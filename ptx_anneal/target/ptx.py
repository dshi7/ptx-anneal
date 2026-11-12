# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The ``ptx`` target — tune ptxas advanced controls (ACFs) for a fixed ``kernel.ptx``.

``score_one`` is the PTX-direct per-candidate objective: ptxas-assemble the FIXED PTX (with the
candidate ACF applied via ``--apply-controls``), launch the cubin via the CUDA driver on real-shaped
inputs, check the output is self-consistent with the no-ACF baseline, and time it. No frontend
recompile is involved -- only ptxas + the driver -- so what we tune is byte-identical to production.

This module is also runnable as the isolated per-candidate worker a :class:`~ptx_anneal.run` Runner
spawns::

    python -m ptx_anneal.target.ptx <task_dir> <acf_path|NONE> [warmup] [rep]
    # prints exactly one line: "MS <float>" on success, or "INVALID".
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys

from . import base
from .base import ScoreInvalid

_SECTION_RE = re.compile(r"^\s*\.section\s+(\S+)")
_LOC_FILE_RE = re.compile(r"^\s*\.(loc|file)(\s|$)")


def normalize_ptx(ptx: str) -> str:
    """Strip volatile debug / source-path info so the content hash reflects only the code -- not where
    the source happened to live. A frontend bakes the source file path into PTX debug info (`.file`,
    the `// path:line` tail of every `.loc`, and the `.debug_*` DWARF sections), so the SAME kernel
    would otherwise hash differently depending on its on-disk path. Dropping these makes the key
    location-independent; the executable instructions are byte-identical without them, so collect /
    factory / consume all agree on one key.

    This MUST stay byte-identical to the frontend's normalizer (a contract test checks it), or the
    factory and the consumer would compute different keys and never agree.
    """
    out = []
    in_debug = False
    for line in ptx.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            in_debug = m.group(1).startswith(".debug")
            if in_debug:
                continue
        if in_debug:
            continue
        if _LOC_FILE_RE.match(line):
            continue
        out.append(line)
    return "\n".join(out)


def ptx_ir_hash(ptx: str) -> str:
    """The store key for a PTX kernel: sha256 of the normalized PTX."""
    return hashlib.sha256(normalize_ptx(ptx).encode("utf-8")).hexdigest()


def _resolve_ptxas(spec: dict) -> str:
    ptxas = (
        spec.get("ptxas")
        or os.environ.get("TRITON_PTXAS_BLACKWELL_PATH")
        or os.environ.get("TRITON_PTXAS_PATH")
        or shutil.which("ptxas")
    )
    if not ptxas:
        raise ScoreInvalid("no ptxas found (set spec['ptxas'] / TRITON_PTXAS_BLACKWELL_PATH, or add ptxas to PATH)")
    return ptxas


class PtxTarget(base.Target):
    """Tune ptxas ACFs for a fixed PTX kernel via the PTX-direct launch bridge."""

    name = "ptx"

    def ir_hash(self, ir: str) -> str:
        return ptx_ir_hash(ir)

    def arch(self, ir: str) -> str:
        from .. import ptx_launch

        return ptx_launch.parse_arch(ir)

    def score_one(
        self,
        task,
        artifact_path: str | None,
        *,
        warmup: int = 100,
        rep: int = 200,
        bench: str = "cudagraph",
        ptxas: str | None = None,
    ) -> float:
        try:
            import torch
        except ImportError as e:
            raise ScoreInvalid("this task requires torch: install 'ptx-anneal[harness,cudapy]'") from e

        from .. import ptx_launch as L

        ptx = task.ir
        spec = task.spec
        # The CLI's resolved pin must win over the collector's machine-specific spec path.
        ptxas = ptxas or _resolve_ptxas(spec)
        if bench not in ("cudagraph", "do_bench"):
            raise ScoreInvalid(f"unknown benchmark method: {bench}")

        def run_and_snapshot(acf):
            cubin = L.ptxas_compile(ptx, ptxas, arch=spec["arch"], acf_path=acf)
            k = L.load_cubin(cubin, spec["entry"], spec["shared"], cluster=spec.get("cluster"))
            try:
                tensors = L.alloc_tensors(spec, seed=0)
                tms, addrs = L.build_tensormaps(spec, tensors)  # [] for non-TMA kernels
                scratch = L.alloc_global_scratch(spec)  # None unless the kernel uses global scratch
                ka = L.kernel_args_from_spec(spec, tensors, addrs, scratch)
                torch.cuda.synchronize()  # finish input initialization before the driver launch
                k.launch(spec["grid"], spec["block"], ka)
                torch.cuda.synchronize()
                return k, tensors, ka, (tms, scratch)
            except Exception:
                k.unload()
                raise

        try:
            # 1) baseline reference (no ACF).
            baseline, ref, _, _tms0 = run_and_snapshot(None)
            baseline.unload()
            # 2) candidate run (with the ACF, or baseline again when artifact_path is None). Keep `tms`
            #    alive: the TMA descriptors and global scratch it holds are referenced by `ka` for the
            #    benchmark below.
            k, tensors, ka, tms = run_and_snapshot(artifact_path)
            try:
                # 3) compare every buffer against independent untuned execution, including BF16.
                # Keep the reference tensors themselves: the candidate has separate allocations,
                # so an extra clone of multi-GB buffers is unnecessary.
                from ..validate import load_validator

                deviations = []
                for r, cur in zip(ref, tensors, strict=True):
                    denom = max(r.float().abs().max().item(), 1e-9)
                    deviations.append((cur.float() - r.float()).abs().max().item() / denom)
                load_validator().check(deviations)
                # 4) preserve the CLI's selected timing method. Keep `tms` alive through timing.
                if bench == "do_bench":
                    ms = L.bench_ms_do_bench(k, spec["grid"], spec["block"], ka, warmup=warmup, reps=rep)
                else:
                    ms = L.bench_ms_cudagraph(k, spec["grid"], spec["block"], ka, warmup=warmup, reps=max(rep // 10, 3))
                del tms
                return float(ms)
            finally:
                k.unload()
        except ScoreInvalid:
            raise
        except Exception as e:
            raise ScoreInvalid(f"{type(e).__name__}: {e}") from e


def _main(argv):
    """Isolated per-candidate worker. Prints 'MS <float>' or 'INVALID'."""
    from .. import task as task_mod

    task_dir, acf = argv[1], argv[2]
    warmup = int(argv[3]) if len(argv) > 3 else 100
    rep = int(argv[4]) if len(argv) > 4 else 200
    artifact_path = None if acf == "NONE" else acf
    try:
        t = task_mod.load(task_dir)
        ms = PtxTarget().score_one(t, artifact_path, warmup=warmup, rep=rep)
        print(f"MS {ms}")
    except Exception as e:  # noqa: BLE001 - the worker reports any failure as INVALID to the parent
        sys.stderr.write(f"[ptx.score_one] {type(e).__name__}: {e}\n")
        print("INVALID")


if __name__ == "__main__":
    _main(sys.argv)
