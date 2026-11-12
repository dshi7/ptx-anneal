# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Standalone PTX replay with synthetic tensors, including BF16, strides and TMA.

Only the captured PTX is compiled. Torch allocates inputs and Triton encodes host TMA descriptors;
neither a frontend kernel nor an optimization engine is invoked. GPU dependencies are lazy.
"""

from .task import Task


def replay(task: Task, ptxas: str, *, warmup: int = 25, rep: int = 50, bench: str = "cudagraph") -> float:
    """Return per-launch milliseconds. This is a benchmark, not an output-correctness check."""
    if warmup < 0 or rep < 1:
        raise ValueError("warmup must be nonnegative and rep must be positive")
    if bench not in ("cudagraph", "do_bench"):
        raise ValueError(f"unknown benchmark method: {bench}")
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("replay requires GPU dependencies: pip install 'ptx-anneal[harness,cudapy]'") from e

    from . import ptx_launch as L

    spec = task.spec
    cubin = L.ptxas_compile(task.ir, ptxas, arch=task.arch)
    kernel = L.load_cubin(cubin, task.entry, spec["shared"], cluster=spec.get("cluster"))
    try:
        tensors = L.alloc_tensors(spec, seed=0)
        # Both the tensors and descriptor objects must outlive graph capture and its replays.
        tensormaps, addresses = L.build_tensormaps(spec, tensors)
        scratch = L.alloc_global_scratch(spec)
        args = L.kernel_args_from_spec(spec, tensors, addresses, scratch)
        # The benchmark uses a nonblocking stream; finish torch's input initialization first.
        torch.cuda.synchronize()
        kernel.launch(spec["grid"], spec["block"], args)
        torch.cuda.synchronize()
        if bench == "do_bench":
            ms = L.bench_ms_do_bench(kernel, spec["grid"], spec["block"], args, warmup=warmup, reps=rep)
        else:
            ms = L.bench_ms_cudagraph(kernel, spec["grid"], spec["block"], args, warmup=warmup, reps=max(rep // 10, 3))
        return float(ms)
    finally:
        kernel.unload()
