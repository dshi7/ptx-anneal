# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PT2 use cases for the 3-step e2e: kernels that reach Triton through ``torch.compile``.

- ``pt2_inductor``: an Inductor-generated pointwise kernel (``triton_poi_*``), launched by Inductor's
  generated launcher through ``CompiledKernel.run``.
- ``pt2_user_defined``: the ``synth_acf`` ``@triton.jit`` kernel called inside a ``torch.compile``
  region, which Inductor wraps and launches the same way.

Inductor caches are disabled so the collect and consume runs each compile through ``triton.compile``,
the path these smokes cover; warm-cache reloads are not exercised here.
"""

import torch

from ..registry import KernelSpec, register
from .synth_acf import _BLOCK, _ITERS, _NACC, _inputs, _ref, _synth_acf_kernel

torch._inductor.config.force_disable_caches = True


def _pointwise(x: torch.Tensor) -> torch.Tensor:
    a, b = torch.sin(x), torch.cos(x)
    for _ in range(8):  # one fused kernel with enough live values to give ptxas something to schedule
        a, b = torch.sin(a + 0.5 * b), torch.cos(b - 0.5 * a)
    return a * b


def _user_defined(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    # pyre-ignore[6, 28]: @triton.jit launch meta-params
    _synth_acf_kernel[(x.numel() // _BLOCK,)](x, y, BLOCK=_BLOCK, ITERS=_ITERS, NACC=_NACC, num_warps=4)
    return y


_compiled_pointwise = torch.compile(_pointwise, fullgraph=True)
_compiled_user_defined = torch.compile(_user_defined, fullgraph=True)

register(
    KernelSpec(
        name="pt2_inductor",
        make_inputs=_inputs,
        run_fn=_compiled_pointwise,
        ref_fn=_pointwise,
        rtol=1e-3,
        desc="PT2 use case A: Inductor-generated pointwise kernel via CompiledKernel.run",
    )
)
register(
    KernelSpec(
        name="pt2_user_defined",
        make_inputs=_inputs,
        run_fn=_compiled_user_defined,
        ref_fn=_ref,
        rtol=5e-2,
        desc="PT2 use case B: user-defined @triton.jit inside torch.compile",
    )
)
