# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Torch-free per-candidate worker for the ``ptx`` target (cuda-python + numpy).

Mirrors ``ptx_anneal.target.ptx`` (ptxas-assemble the FIXED PTX with the ACF applied, launch via the
CUDA driver, check self-consistency vs the no-ACF baseline, time it) but uses **cuda-python + numpy**
instead of torch -- so candidate scoring runs in environments without a (working) torch install. This
is the worker spawned by :class:`ptx_anneal.run.cudapy.CudaPyRunner`. Non-TMA, contiguous tensors only;
for TMA/strided kernels use the torch path (``ptx_anneal.target.ptx`` via ``LocalRunner``).

Usage: python -m ptx_anneal.run.cudapy_worker <task_dir> <acf_path|NONE> [warmup] [rep]
Prints exactly one line: "MS <float>" on success, else "INVALID".
"""

from __future__ import annotations

import os
import sys

import numpy as np
from cuda.bindings import driver as d

from ptx_anneal import ptx_launch as L
from ptx_anneal import task as task_mod
from ptx_anneal.validate import load_validator

_NP = {
    "float32": np.float32,
    "float16": np.float16,
    "int32": np.int32,
    "int64": np.int64,
    "int8": np.int8,
}


def _chk(r):
    if r[0] != d.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA error: {r[0]}")
    return r[1] if len(r) > 1 else None


def _resolve_ptxas(spec):
    p = spec.get("ptxas") or os.environ.get("TRITON_PTXAS_BLACKWELL_PATH") or os.environ.get("TRITON_PTXAS_PATH")
    if not p:
        from shutil import which

        p = which("ptxas")
    if not p:
        raise RuntimeError("no ptxas (set spec['ptxas'] / TRITON_PTXAS_BLACKWELL_PATH)")
    return p


def _host_inputs(spec, seed=0):
    """Deterministic host arrays (contiguous), one per spec tensor."""
    rng = np.random.default_rng(seed)
    arrs = []
    for t in spec["tensors"]:
        dt = _NP.get(t["dtype"])
        if dt is None:
            raise RuntimeError(f"unsupported dtype {t['dtype']!r}")
        numel = int(np.prod(t["shape"])) if t["shape"] else 1
        if np.issubdtype(dt, np.floating):
            arrs.append(rng.standard_normal(numel).astype(dt))
        else:
            arrs.append(np.zeros(numel, dtype=dt))
    return arrs


def _alloc_global_scratch(spec):
    """(allocation, aligned address) of a zeroed grid_size * size byte buffer, or None when the
    kernel uses no global scratch (see ptx_launch.alloc_global_scratch)."""
    gs = spec.get("global_scratch")
    if not gs:
        return None
    gx, gy, gz = spec["grid"]
    size, align = gx * gy * gz * int(gs["size"]), int(gs["align"])
    p = _chk(d.cuMemAlloc(size + align))
    _chk(d.cuMemsetD8(p, 0, size + align))
    return p, int(p) + (-int(p) % align)


def _launch_and_snapshot(ptx, spec, ptxas, acf_path, host_inputs):
    """ptxas(ptx[,acf]) -> cubin -> alloc+HtoD -> launch -> sync -> DtoH all buffers. Returns
    (loaded_kernel, kernel_args, device_ptrs, [np output arrays]). Caller frees device_ptrs."""
    cubin = L.ptxas_compile(ptx, ptxas, arch=spec["arch"], acf_path=acf_path)
    k = L.load_cubin(cubin, spec["entry"], spec["shared"], cluster=spec.get("cluster"))
    ptrs = []
    for h in host_inputs:
        p = _chk(d.cuMemAlloc(h.nbytes))
        _chk(d.cuMemcpyHtoD(p, h.ctypes.data, h.nbytes))
        ptrs.append(p)
    scratch = _alloc_global_scratch(spec)
    # Build launch args from the spec's final arg order using our device pointers.
    ka = []
    for a in spec["args"]:
        if "t" in a:
            ka.append(("ptr", int(ptrs[a["t"]])))
        elif a.get("global_scratch"):
            ka.append(("ptr", scratch[1]))
        elif a.get("null"):
            ka.append(("null",))
        elif "i32" in a:
            ka.append(("i32", a["i32"]))
        elif "i64" in a:
            ka.append(("i64", a["i64"]))
        elif "f32" in a:
            ka.append(("f32", a["f32"]))
        else:
            raise RuntimeError(f"unsupported/non-TMA spec arg: {a}")
    k.launch(spec["grid"], spec["block"], ka)
    _chk(d.cuCtxSynchronize())
    snaps = []
    for h, p in zip(host_inputs, ptrs, strict=True):
        out = np.empty_like(h)
        _chk(d.cuMemcpyDtoH(out.ctypes.data, p, h.nbytes))
        snaps.append(out)
    if scratch is not None:
        ptrs.append(scratch[0])  # freed with the tensors, after the benchmark that reuses `ka`
    return k, ka, ptrs, snaps


def score(task_dir, acf_path, warmup, rep, bench="cudagraph", *, ptxas=None):
    t = task_mod.load(task_dir)
    spec, ptx = t.spec, t.ir
    ptxas = ptxas or _resolve_ptxas(spec)
    host = _host_inputs(spec)

    # 1) baseline (no ACF) reference snapshot.
    _, _, ptrs0, ref = _launch_and_snapshot(ptx, spec, ptxas, None, host)
    for p in ptrs0:
        d.cuMemFree(p)
    # 2) candidate run.
    k, ka, ptrs, cur = _launch_and_snapshot(ptx, spec, ptxas, acf_path, host)
    try:
        # 3) self-consistency: per-buffer relative deviation vs the no-ACF baseline, then let the
        #    pluggable validator accept/reject (an ACF only changes codegen, never the math).
        deviations = []
        for r, c in zip(ref, cur, strict=True):
            if not np.issubdtype(r.dtype, np.floating):
                continue
            denom = max(float(np.abs(r).max()), 1e-9)
            deviations.append(float(np.abs(c.astype(np.float64) - r.astype(np.float64)).max()) / denom)
        load_validator().check(deviations)  # raises ValidationError -> scored INVALID by main()
        # 4) time the candidate. cudagraph (default) amortizes host launch overhead via graph replay --
        #    fast + deterministic, right for a smoke; do_bench flushes the L2 and takes the median of
        #    single launches -- slower but faithful to the consumer's launch-time A/B.
        if bench == "do_bench":
            return L.bench_ms_do_bench(k, spec["grid"], spec["block"], ka, warmup=warmup, reps=rep)
        return L.bench_ms_cudagraph(k, spec["grid"], spec["block"], ka, warmup=warmup, reps=max(rep // 10, 3))
    finally:
        for p in ptrs:
            d.cuMemFree(p)


def main(argv):
    task_dir, acf = argv[1], argv[2]
    warmup = int(argv[3]) if len(argv) > 3 else 100
    rep = int(argv[4]) if len(argv) > 4 else 200
    bench = argv[5] if len(argv) > 5 else "cudagraph"
    acf_path = None if acf == "NONE" else acf
    try:
        print(f"MS {score(task_dir, acf_path, warmup, rep, bench)}")
    except Exception as e:  # noqa: BLE001 - any failure is INVALID to the parent
        sys.stderr.write(f"[cudapy_worker] {type(e).__name__}: {e}\n")
        print("INVALID")


if __name__ == "__main__":
    main(sys.argv)
