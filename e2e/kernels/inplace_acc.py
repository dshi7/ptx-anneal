# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""In-place accumulating kernel (``y += f(x)``) for the consume A/B safety rule.

The free-win A/B launches the kernel many times before the real launch. Run on the caller's
buffers, those extra launches would add ``f(x)`` to ``y`` again and again, so the first checked
result would be wrong. The check below catches that.
"""

import torch
import triton
import triton.language as tl

from ..registry import KernelSpec, register

_BLOCK = 1024
_N = 1 << 20


@triton.jit
def _inplace_acc_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, tl.load(y_ptr + offs) + 0.5 * tl.sin(x) * tl.cos(0.5 * x))


def _run(x: torch.Tensor, y: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    # pyre-ignore[6, 28]: @triton.jit launch meta-params
    _inplace_acc_kernel[(x.numel() // _BLOCK,)](x, y, BLOCK=_BLOCK, num_warps=4)
    return y


def _ref(x: torch.Tensor, y: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    return y0 + 0.5 * torch.sin(x) * torch.cos(0.5 * x)


def _inputs():
    x = torch.randn((_N,), device="cuda", dtype=torch.float32)
    y = torch.randn((_N,), device="cuda", dtype=torch.float32)
    return x, y, y.clone()


register(
    KernelSpec(
        name="inplace_acc",
        make_inputs=_inputs,
        run_fn=_run,
        ref_fn=_ref,
        rtol=1e-5,
        desc="in-place accumulation; fails if the consume A/B writes to the caller's buffers",
    )
)
