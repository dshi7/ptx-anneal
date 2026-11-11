# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Toolchain detection for ``ptx-anneal doctor``.

Reports whether the three things a real search needs are present:
1. ``ptxas`` >= the minimum version (applying an ACF via ``--apply-controls`` is a ptxas 13.3+ GA
   feature) — needed to *assemble* tuning artifacts;
2. an optimization engine (bring-your-own; e.g. NVIDIA CompileIQ) registered as a search backend;
3. a GPU + driver stack (cuda-python / torch) — needed to *launch* candidates.

The baseline backend + the store/ABC layer work without any of these; ``tune`` with a real engine
needs all three. ``doctor`` only inspects and prints next steps — it never installs anything.

Engine discovery is automatic: an installed engine plugin registers under the
``ptx_anneal.search_backends`` entry-point group and is found here with no env var to point at it.
The only env that may matter is ``TRITON_PTXAS_BLACKWELL_PATH`` (when ptxas isn't on ``PATH``); any
engine-internal config (its search space, etc.) is the engine's own concern, not ptx-anneal's.

Caveat: ``tune`` drives the engine through a standalone *adapter* subprocess (see ``cli.py``), which
does not need an entry-point plugin. So "no engine registered" here does not mean ``tune`` will fail
with the bundled adapter -- it only reports the plugin registry.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess

from .search.base import available_backends

# ACF apply (`ptxas --apply-controls`) is GA from this ptxas version on. Not overridable: an older
# ptxas cannot apply an ACF at all, so "relaxing" the floor only turns a clear error into a silent
# no-op (and, on the frontend side, a store MISS -- see the PTX-ISA note in min_ptxas()).
MIN_PTXAS = "13.3"


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _wheel_ptxas() -> list[str]:
    """ptxas binaries shipped by the ``nvidia-cuda-nvcc`` wheels, newest CUDA major first.

    A pip-installed ptxas is not on ``PATH``. Without looking here, a box whose only new-enough
    ptxas came from a wheel -- the normal outcome of provisioning by pip, and the one the "no
    vendored binaries" rule pushes people toward -- reports as unprovisioned.
    """
    import glob
    import sysconfig

    found: list[str] = []
    for key in ("purelib", "platlib"):
        root = sysconfig.get_paths().get(key)
        if root:
            found += glob.glob(os.path.join(root, "nvidia", "cu*", "bin", "ptxas"))
    return sorted({p for p in found if os.path.exists(p)}, reverse=True)


def find_ptxas() -> str | None:
    """The ptxas ptx-anneal would use: explicit env, then PATH, then a pip-installed wheel.

    An explicitly pointed-at ptxas is returned verbatim even when it is too old. The whole
    toolchain -- collect, factory, consume -- must agree on ONE ptxas, so silently substituting a
    different one would trade a loud version error for a silent store MISS later.

    With nothing pinned, prefer the first candidate that MEETS the floor rather than the first one
    found: a PATH ptxas below MIN_PTXAS cannot apply an ACF at all, so letting it shadow a new
    enough wheel would fail a box that is in fact provisioned. If none qualify, return the first so
    the caller still reports what it saw.
    """
    for env in ("TRITON_PTXAS_BLACKWELL_PATH", "TRITON_PTXAS_PATH"):
        p = os.environ.get(env)
        if p and os.path.exists(p):
            return p

    on_path = shutil.which("ptxas")
    candidates = ([on_path] if on_path else []) + [p for p in _wheel_ptxas() if p != on_path]
    need = _version_tuple(MIN_PTXAS)
    for c in candidates:
        v = ptxas_version(c)
        if v and _version_tuple(v) >= need:
            return c
    return candidates[0] if candidates else None


def ptxas_version(path: str | None) -> str | None:
    # A None path is "no ptxas was found", which is a reportable state like any other bad path --
    # not a TypeError from deep inside subprocess. Callers already handle a None return.
    if not path:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        # Not executable, wrong arch, hung past the timeout: "unknown version", not a crash. The
        # caller decides (an unparseable version simply fails the floor check).
        return None
    m = re.search(r"release\s+([0-9]+\.[0-9]+)", out)
    if m:
        return m.group(1)
    m = re.search(r"V([0-9]+\.[0-9]+\.[0-9]+)", out)
    return m.group(1) if m else None


def min_ptxas() -> str:
    """The required ptxas floor.

    Fixed, by design. Beyond `--apply-controls` being GA only from here on, the frontend derives the
    PTX ISA version it emits from *its* ptxas version, so a task collected under a different ptxas
    hashes to a different store key -- a lower floor would silently produce ACFs nothing can consume.
    """
    return MIN_PTXAS


def visible_gpus() -> list[str]:
    """Probe CUDA-visible devices for tuning, without importing torch or cuda-python.

    CUDA applies CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER during initialization. Preserve
    that order and the caller's device identifiers when pinning fresh scorer subprocesses.
    This initializes the driver, so only tuning invokes it; doctor remains best-effort.
    """
    try:
        driver = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
        status = driver.cuInit(0)
        if status == 100:  # CUDA_ERROR_NO_DEVICE
            return []
        if status != 0:
            raise RuntimeError(f"CUDA GPU discovery: cuInit failed with error {status}")
        count = ctypes.c_int()
        status = driver.cuDeviceGetCount(ctypes.byref(count))
    except (OSError, AttributeError) as e:
        raise RuntimeError(f"CUDA GPU discovery requires a working NVIDIA driver: {e}") from e
    if status != 0:
        raise RuntimeError(f"CUDA GPU discovery: cuDeviceGetCount failed with error {status}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(index) for index in range(count.value)]
    # CUDA stops enumeration at an invalid entry (including -1). Use only the prefix it found.
    return [gpu.strip() for gpu in visible.split(",")][: count.value]


def gpu_status() -> dict:
    """Whether the launch stack (cuda-python + torch + a visible GPU) is available.

    Every probe here is best-effort by design: this backs ``doctor``, whose entire job is to report
    what is missing. An absent or broken optional dependency is the answer, not an error -- so the
    blind excepts below are deliberate, and must stay blind: a half-installed torch can raise almost
    anything on import, and `doctor` failing is strictly worse than `doctor` saying "no GPU".
    """
    status = {"cuda_python": False, "torch": False, "gpu": False, "detail": ""}
    try:
        import cuda.bindings.driver  # noqa: F401

        status["cuda_python"] = True
    except Exception:  # noqa: BLE001, S110 - absent/broken cuda-python is a reportable state
        pass
    try:
        import torch

        status["torch"] = True
        status["gpu"] = bool(torch.cuda.is_available())
        if status["gpu"]:
            status["detail"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001, S110 - ditto for torch / a driver that fails to initialise
        pass
    return status


def report() -> dict:
    """A structured toolchain report (consumed by ``doctor``)."""
    ptxas = find_ptxas()
    ver = ptxas_version(ptxas) if ptxas else None
    need = min_ptxas()
    ptxas_ok = bool(ver) and _version_tuple(ver) >= _version_tuple(need)
    engines = [b for b in available_backends() if b != "baseline"]
    return {
        "ptxas_path": ptxas,
        "ptxas_version": ver,
        "ptxas_min": need,
        "ptxas_ok": ptxas_ok,
        "engines": engines,
        "gpu": gpu_status(),
    }


def format_report(rep: dict) -> str:
    """A human-readable doctor report + the next provisioning step."""
    lines = ["ptx-anneal doctor", "=================="]

    if rep["ptxas_path"]:
        mark = "OK " if rep["ptxas_ok"] else "!! "
        lines.append(
            f"[{mark}] ptxas: {rep['ptxas_path']} (version {rep['ptxas_version'] or '?'}, need >= {rep['ptxas_min']})"
        )
        if not rep["ptxas_ok"]:
            lines.append("       -> install/point at a newer CUDA Toolkit; set TRITON_PTXAS_BLACKWELL_PATH.")
    else:
        lines.append("[!! ] ptxas: not found")
        lines.append("       -> install a CUDA Toolkit and set TRITON_PTXAS_BLACKWELL_PATH (or add ptxas to PATH).")

    if rep["engines"]:
        lines.append(f"[OK ] engine(s): {', '.join(rep['engines'])}")
    else:
        lines.append("[-- ] engine: no plugin registered (only the built-in 'baseline' backend)")
        lines.append("       -> this line reports the 'ptx_anneal.search_backends' plugin registry only. `tune` does")
        lines.append("          NOT need it: it drives the engine through an adapter subprocess, so after")
        lines.append("          `pip install ptx-anneal[compileiq]` tuning works while this still reads '--'.")
        lines.append("       -> a plugin is only needed for the library API (factory.tune); install one and it")
        lines.append("          self-registers under that entry point and appears here — no env var needed.")

    g = rep["gpu"]
    if g["gpu"]:
        lines.append(f"[OK ] gpu: {g['detail']} (cuda-python={g['cuda_python']}, torch={g['torch']})")
    else:
        lines.append(f"[!! ] gpu: not available (cuda-python={g['cuda_python']}, torch={g['torch']})")
        lines.append("       -> 'doctor' and the store/ABCs work without a GPU, but 'tune' needs one.")

    ready = rep["ptxas_ok"] and g["gpu"]
    lines.append("")
    lines.append("baseline flow ready (no engine): " + ("YES" if g["gpu"] else "no — needs a GPU"))
    lines.append("real tuning ready: " + ("YES" if (ready and rep["engines"]) else "no — see the items marked !! / --"))
    return "\n".join(lines)
