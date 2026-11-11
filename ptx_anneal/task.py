# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The ACF tuning-task schema (``spec_version`` 1).

A *task* is the unit of work a frontend (e.g. fbtriton's collection hook) hands to
the factory. It is **source-free** — the factory never recompiles from Python — and
lives in a directory containing exactly two files:

    <task_dir>/kernel.ptx   the fixed IR the artifact is tuned for
    <task_dir>/spec.json    the launch description defined by this module

``spec.json`` schema (v1)::

    {
      "spec_version": 1,                 # optional; absent == 1 (legacy tasks)
      "entry":  "add_kernel",            # the PTX .visible .entry name
      "arch":   "sm_100a",               # the PTX .target
      "shared": 0,                       # dynamic shared-memory bytes
      "block":  [128, 1, 1],             # CTA dims (threads)
      "grid":   [8, 1, 1],               # launch grid (CTAs)
      "ptxas":  "",                      # ptxas path (filled at run; never hardcode)
      "tensors": [{"shape", "dtype", "strides"}, ...],
      "args":    [{"t": i} | {"i32"|"i64"|"f32": v} | {"tma": j} | {"null": true}, ...],
      "tensordescs": [...],              # optional; host-TMA descriptors
      "ir_hash": "<sha256>",             # content key; alias "ptx_sha256" (legacy)
      "kernel_name": "...", "fn_name": "..."   # provenance (optional)
    }

The store key is ``(target, arch, ir_hash)``. For the ``ptx`` target ``ir_hash`` is
``sha256`` of the normalized PTX (see :mod:`ptx_anneal.target.ptx`), so one artifact
transfers across runtime shapes of the same kernel. ``"args"`` is the final
kernel-param order (post constexpr / ``equal_to_1`` specialization), including the two
trailing null scratch params.

The loader accepts specs that omit ``spec_version`` (treated as v1) so tasks already
collected by older frontends keep working.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

SPEC_VERSION = 1

# Interop env var (external contract — shared with the frontend; do NOT rename).
TASK_DIR_ENV = "COMPILE_IQ_TASK_DIR"

# Fields every v1 spec must carry to describe a launch.
REQUIRED_FIELDS = ("entry", "arch", "shared", "block", "grid", "tensors", "args")


class TaskError(ValueError):
    """Raised when a task cannot be fetched or its files are missing/malformed."""


def _manifold_task_path(task_dir: str) -> str:
    """Validate a Manifold directory URL and return the CLI's bucket/path form."""
    url = urlsplit(task_dir)
    if url.scheme != "manifold":
        raise TaskError(f"unsupported task URL scheme: {url.scheme!r}")
    path = url.path.rstrip("/")
    if not url.netloc or not path.strip("/") or url.query or url.fragment:
        raise TaskError(f"expected manifold://<bucket>/<task directory>, got {task_dir!r}")
    if path.rsplit("/", 1)[-1] in ("spec.json", "kernel.ptx"):
        parent = "manifold://" + url.netloc + path.rsplit("/", 1)[0]
        raise TaskError(f"Manifold --task must name a directory, not a file; use {parent}")
    return url.netloc + path


@contextmanager
def local_task_dir(task_dir: str | os.PathLike[str]) -> Iterator[str]:
    """Yield a local task directory, downloading a Manifold task for this context's lifetime.

    Remote inputs must name a directory. Only the standalone ``manifold`` CLI
    is needed; local tasks have no remote dependencies. Keep the context open until all scoring
    subprocesses have finished using the downloaded files.
    """
    task_dir = os.fspath(task_dir)
    if "://" not in task_dir:
        yield task_dir
        return
    source = _manifold_task_path(task_dir)
    manifold = shutil.which("manifold")
    if not manifold:
        raise TaskError("a manifold:// task requires the 'manifold' CLI on PATH; alternatively download the task first")
    with tempfile.TemporaryDirectory(prefix="ptx_anneal_task_") as local_dir:
        for filename in ("spec.json", "kernel.ptx"):
            remote = f"{source}/{filename}"
            dest = os.path.join(local_dir, filename)
            try:
                result = subprocess.run(
                    [manifold, "get", remote, dest], capture_output=True, text=True, timeout=120, check=False
                )
            except (OSError, subprocess.SubprocessError) as e:
                raise TaskError(f"failed to download manifold://{remote}: {e}") from e
            if result.returncode != 0 or not os.path.isfile(dest):
                detail = result.stderr.strip() or result.stdout.strip() or "download did not create the requested file"
                raise TaskError(f"failed to download manifold://{remote} (exit {result.returncode}): {detail}")
        yield local_dir


def manifold_store_url(task_dir: str) -> str:
    """Map a captured task's ``.../tasks/<id>`` URL to the run's sibling ``.../store``."""
    source = _manifold_task_path(task_dir)
    parts = source.split("/")
    if len(parts) < 4 or parts[-2] != "tasks" or any(part in ("", ".", "..") for part in parts):
        raise TaskError("--upload-manifold requires --task manifold://<bucket>/<run>/tasks/<task-id>")
    return "manifold://" + "/".join([*parts[:-2], "store"])


def manifold_tasks(task_list: str) -> list[str]:
    """Recursively discover task directories containing both spec.json and kernel.ptx.

    Use the external CLI for discovery, then download one task at a time during tuning. Sorting
    and deduplicating the URLs makes the run order independent of Manifold's listing order.
    """
    source = _manifold_task_path(task_list)
    if any(part in ("", ".", "..") for part in source.split("/")):
        raise TaskError(f"invalid Manifold task-list directory: {task_list!r}")
    manifold = shutil.which("manifold")
    if not manifold:
        raise TaskError("--task-list requires the 'manifold' CLI on PATH")
    try:
        result = subprocess.run(
            [manifold, "find", source, r"^(spec\.json|kernel\.ptx)$", "--regex", "--type", "f"],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise TaskError(f"failed to list tasks under {task_list}: {e}") from e
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise TaskError(f"failed to list tasks under {task_list} (exit {result.returncode}): {detail}")
    files: dict[str, set[str]] = {}
    for line in result.stdout.splitlines():
        path = line.strip().removeprefix("manifold://")
        if not path:
            continue
        parent, _, filename = path.rpartition("/")
        if (
            not path.startswith(source + "/")
            or any(part in ("", ".", "..") for part in path.split("/"))
            or filename not in ("spec.json", "kernel.ptx")
        ):
            raise TaskError(f"unexpected Manifold task-list entry: {line!r}")
        files.setdefault(parent, set()).add(filename)
    tasks = sorted("manifold://" + parent for parent, names in files.items() if names == {"spec.json", "kernel.ptx"})
    if not tasks:
        raise TaskError(f"no tasks containing spec.json and kernel.ptx found under {task_list}")
    return tasks


def upload_store(task_dir: str, store_root: str, acf_path: str) -> str:
    """Publish this task's store subtree under ``.../store/<arch>/<hash16>/``.

    Include all versioned ACFs, their provenance and optional cubins for this kernel, preserving
    paths relative to the store root. Other kernels and temporary files are not part of this task's
    upload. Return the remote store URL after every upload succeeds; retain local files on failure.
    """
    destination = manifold_store_url(task_dir)
    manifold = shutil.which("manifold")
    if not manifold:
        raise TaskError("--upload-manifold requires the 'manifold' CLI on PATH")
    root, acf = Path(store_root).resolve(), Path(acf_path).resolve()
    try:
        relative_dir = acf.parent.relative_to(root)
    except ValueError as e:
        raise TaskError(f"cannot upload store: ACF {acf} is outside {root}") from e
    if len(relative_dir.parts) != 2 or not acf.name.startswith(relative_dir.name + ".") or acf.suffix != ".acf":
        raise TaskError("cannot upload store: expected <store>/<arch>/<ir_hash>/<ir_hash>[.<ptxas>].acf")
    if not acf.is_file():
        raise TaskError(f"cannot upload store: missing local file {acf}")
    files = []
    for artifact in sorted(acf.parent.iterdir()):
        if not artifact.name.startswith(relative_dir.name + ".") or artifact.suffix != ".acf":
            continue
        sidecar = Path(str(artifact) + ".json")
        for path in (artifact, sidecar):
            if not path.is_file():
                raise TaskError(f"cannot upload store: missing local file {path}")
            files.append(path)
        cubin = Path(str(artifact) + ".cubin")
        if cubin.is_file():
            files.append(cubin)
    remote_dir = f"{destination.removeprefix('manifold://')}/{relative_dir.as_posix()}"
    # Idempotent directory creation and file replacement let later tuning runs refresh the store.
    commands = [[manifold, "mkdir", "-p", remote_dir]]
    commands += [[manifold, "put", "--overwrite", str(path), f"{remote_dir}/{path.name}"] for path in files]
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError) as e:
            raise TaskError(f"Manifold {command[1]} failed for manifold://{command[-1]}: {e}") from e
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise TaskError(
                f"Manifold {command[1]} failed for manifold://{command[-1]} (exit {result.returncode}): {detail}"
            )
    return destination


def default_task_root() -> str:
    """Where tasks live by default (overridable via ``COMPILE_IQ_TASK_DIR``)."""
    return os.environ.get(TASK_DIR_ENV, os.path.expanduser("~/.compile_iq/tasks"))


def spec_ir_hash(spec: dict) -> str | None:
    """The content key from a spec, accepting the legacy ``ptx_sha256`` alias."""
    return spec.get("ir_hash") or spec.get("ptx_sha256")


def validate_spec(spec: dict) -> None:
    """Raise :class:`TaskError` unless ``spec`` is a well-formed v1 launch spec."""
    if not isinstance(spec, dict):
        raise TaskError("spec.json must be a JSON object")
    version = int(spec.get("spec_version", 1))
    if version != SPEC_VERSION:
        raise TaskError(f"unsupported spec_version {version} (this build supports {SPEC_VERSION})")
    missing = [f for f in REQUIRED_FIELDS if f not in spec]
    if missing:
        raise TaskError(f"spec.json missing required field(s): {missing}")
    if not spec_ir_hash(spec):
        raise TaskError("spec.json missing ir_hash (or legacy ptx_sha256)")
    if not isinstance(spec["tensors"], list) or not isinstance(spec["args"], list):
        raise TaskError("spec.json 'tensors' and 'args' must be lists")
    for axis in ("block", "grid"):
        v = spec[axis]
        if not isinstance(v, list) or not v:
            raise TaskError(f"spec.json '{axis}' must be a non-empty list")


@dataclass
class Task:
    """A loaded tuning task: its directory, fixed IR, and validated spec."""

    dir: str
    ir: str  # the fixed kernel source IR (PTX for the ``ptx`` target)
    spec: dict

    @property
    def entry(self) -> str:
        return self.spec["entry"]

    @property
    def arch(self) -> str:
        return self.spec["arch"]

    @property
    def ir_hash(self) -> str:
        h = spec_ir_hash(self.spec)
        assert h is not None  # guaranteed by validate_spec
        return h

    @property
    def spec_version(self) -> int:
        return int(self.spec.get("spec_version", 1))


def load(task_dir: str) -> Task:
    """Load and validate a task directory (``kernel.ptx`` + ``spec.json``)."""
    spec_path = os.path.join(task_dir, "spec.json")
    ir_path = os.path.join(task_dir, "kernel.ptx")
    if not os.path.isfile(spec_path):
        raise TaskError(f"no spec.json in task dir: {task_dir}")
    if not os.path.isfile(ir_path):
        raise TaskError(f"no kernel.ptx in task dir: {task_dir}")
    try:
        with open(spec_path) as f:
            spec = json.load(f)
    except json.JSONDecodeError as e:
        raise TaskError(f"spec.json is not valid JSON: {e}") from e
    with open(ir_path) as f:
        ir = f.read()
    validate_spec(spec)
    return Task(dir=task_dir, ir=ir, spec=spec)
