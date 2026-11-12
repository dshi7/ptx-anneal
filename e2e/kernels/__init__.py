# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Import each kernel module so it self-registers in the registry.

New kernels: add a guarded ``from . import <module>`` below.

Every import is guarded, and that is the point: these modules import torch, triton and (for the TLX
ones) tutorial kernels that only exist on some pins/arches. An unguarded import means one
unavailable kernel empties the whole registry -- including the canary, whose entire job is to still
work when something else is broken. Report and continue instead.
"""

import sys


def pin_autotuner(kernel, index: int = 0, match: dict | None = None, prune_args: dict | None = None) -> dict:
    """Narrow an ``@triton.autotune``'d kernel to exactly ONE config, in place. Returns it as a dict.

    This is how a TLX tutorial gets tuned without being edited. Triton's own ``Autotuner.run`` only
    benchmarks when ``len(configs) > 1``; with one config it uses that config directly, so narrowing
    the list turns the autotuner into a fixed choice -- deterministic, no sweep, and unambiguous
    about which kernel the search is measuring. The runway's autotune refusal accepts this form.

    ``match`` selects by config kwargs (e.g. ``{"BLOCK_SIZE_M": 4}``) instead of by position; a
    positional default is fine but fragile, since upstream reorders config lists freely.

    **``prune_args`` is mandatory when the kernel has an ``early_config_prune`` hook**, and this is
    the subtle part. Triton runs that hook only on the ``len(configs) > 1`` path, and TLX tutorials
    use it to *derive* constexprs rather than merely to filter: the multi-CTA LayerNorm hook rewrites
    ``BLOCK_SIZE_N`` from a placeholder 8192 to ``next_power_of_2(N // num_reduction_ctas)`` and sets
    ``SHOULD_MASK_ROW``/``SHOULD_MASK_COL``. Pin without running it and the kernel launches happily
    with the placeholder and computes garbage -- observed as a silently all-zero LayerNorm output, not
    as an error. So the hook is run here with the shapes the KernelSpec will actually use, and
    pinning refuses rather than guessing when those shapes were not supplied.
    """
    configs = list(getattr(kernel, "configs", []) or [])
    if not configs:
        raise ValueError(f"{kernel!r} has no autotune configs to pin")

    prune = getattr(kernel, "early_config_prune", None)
    if prune is not None:
        if not prune_args:
            raise ValueError(
                f"{getattr(kernel, 'base_fn', kernel)} has an early_config_prune hook, so its configs "
                "carry DERIVED constexprs that triton only computes on the multi-config path. Pinning "
                "without running it would launch with placeholder values and silently produce wrong "
                "numbers. Pass prune_args={'M': ..., 'N': ...} (the shapes this KernelSpec uses)."
            )
        # The hook mutates conf.kwargs in place and returns the surviving configs.
        configs = list(prune(configs, {}, **prune_args))
        if not configs:
            raise ValueError(f"early_config_prune rejected every config for {prune_args}")

    if match:
        picked = [c for c in configs if all(c.kwargs.get(k) == v for k, v in match.items())]
        if len(picked) != 1:
            raise ValueError(f"match {match} selected {len(picked)} of {len(configs)} configs; need exactly 1")
        chosen = picked[0]
    else:
        chosen = configs[index]
    kernel.configs = [chosen]
    # Autotuner caches the selected config per key; stale entries would defeat the pin.
    if hasattr(kernel, "cache"):
        try:
            kernel.cache.clear()
        except AttributeError:  # some builds use a non-dict cache
            pass
    return {
        **dict(chosen.kwargs),
        "num_warps": getattr(chosen, "num_warps", None),
        "num_stages": getattr(chosen, "num_stages", None),
        "num_ctas": getattr(chosen, "num_ctas", None),
        "pinned_from": f"pin_autotuner({getattr(getattr(kernel, 'base_fn', kernel), '__name__', kernel)}, "
        f"{match or f'index={index}'}) of {len(configs)} configs",
    }


def _load(name: str) -> None:
    try:
        __import__(f"{__name__}.{name}")
    except Exception as e:  # noqa: BLE001 - a registered kernel may fail to import for any reason
        # (missing torch/triton, a bad @triton.jit body, an unsupported arch, an absent tlx tutorial).
        # Reporting and continuing is deliberate: one broken kernel must not take out the registry.
        print(f"[e2e] {name} kernel unavailable: {e}", file=sys.stderr)


# doc_matmul first: it is the canary (NVIDIA's known-good doc example), so it should be registered
# before anything that is more likely to fail.
_load("doc_matmul")
# ws_gemm: authentic TLX warp-specialized Blackwell GEMM (needs the tlx tutorials on this pin).
_load("ws_gemm")
# gemm_2cta: TMA + tcgen5 WITHOUT warp specialization -- the WS deconfounder.
_load("gemm_2cta")
# fa_ws: TLX WS flash attention -- TMA + tcgen5 + WS; the kernel that faulted rather than failed.
_load("fa_ws")
# layernorm_ws: warp specialization WITHOUT tcgen5/TMA -- the control that exonerates or implicates WS.
_load("layernorm_ws")
# gemm_pipelined / gemm_clc: more points in the tcgen5+TMA cells (no-WS and WS+CLC).
_load("gemm_pipelined")
_load("gemm_clc")
# layernorm_mc: the non-WS partner of layernorm_ws -- isolates warp specialization as one variable.
_load("layernorm_mc")
# fa_family: the four TLX Blackwell flash-attention variants (TMA + tcgen5 + WS).
_load("fa_family")
# addmm_2cta: 2-CTA host-TMA matmul with and without WS -- autotune-free upstream.
_load("addmm_2cta")
# synth_acf: register-spilling synthetic kernel -- real ptxas-ACF headroom + a fidelity probe.
_load("synth_acf")
# pt2: the torch.compile use cases -- an Inductor-generated kernel and a user-defined one.
_load("pt2")
