# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the GPU-light bits of ptx_launch: auto-N math + kernelParams building (no GPU)."""

import json
from types import SimpleNamespace

import pytest

from ptx_anneal.ptx_launch import (
    _build_kernel_params,
    _pick_n,
    build_spec,
    build_spec_from_schema,
    kernel_args_from_spec,
)


# --- CUDA-graph unroll count (auto-N) -----------------------------------------------------------
def test_pick_n_scales_region_to_rep_ms():
    assert _pick_n(0.5, 100.0) == 200  # 100ms / 0.5ms
    assert _pick_n(2.0, 100.0) == 50


def test_pick_n_floor_and_degenerate():
    assert _pick_n(1000.0, 100.0) == 1  # est bigger than target -> at least 1
    assert _pick_n(0.0, 100.0) == 1  # nonpositive est -> 1
    assert _pick_n(float("inf"), 100.0) == 1
    assert _pick_n(float("nan"), 100.0) == 1


def test_pick_n_clamped_to_max():
    assert _pick_n(1e-12, 100.0) == 100000  # tiny est -> clamp at n_max


# --- kernelParams building --------------------------------------------------------------------
def test_build_kernel_params_len_and_holders():
    args = [("ptr", 0x1000), ("i32", 7), ("i64", 9), ("f32", 1.5), ("null",), ("tma", 0x2000)]
    arr, holders = _build_kernel_params(args)
    assert len(arr) == len(args)
    # ptr/i32/i64/f32/null each back a value holder; tma is passed by value (no holder).
    assert len(holders) == 5
    assert arr[5] == 0x2000  # tma slot points straight at the descriptor address


def test_build_kernel_params_rejects_unknown_kind():
    with pytest.raises(ValueError):
        _build_kernel_params([("weird", 1)])


# --- build_spec cluster handling (ctas_per_cga; no GPU) ----------------------------------------
# Minimal kernel: one user pointer param + the two trailing scratch pointers.
_PTX = """//
.version 8.3
.target sm_90a
.address_size 64

.visible .entry k(
    .param .u64 .ptr .global .align 16 p0,
    .param .u64 .ptr .global .align 16 p1,
    .param .u64 .ptr .global .align 16 p2
)
{
    ret;
}
"""
_ARGS = [("tensor", {"id": 0, "shape": [16], "dtype": "float32", "strides": [1]})]


def _md(**over):
    base = {
        "global_scratch_size": 0,
        "profile_scratch_size": 0,
        "num_ctas": 1,
        "ctas_per_cga": None,
        "tensordesc_meta": [],
        "num_warps": 4,
        "shared": 0,
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_build_spec_records_ctas_per_cga_cluster():
    spec = build_spec(_PTX, _md(ctas_per_cga=(2, 1, 1)), (4, 1, 1), _ARGS, "ptxas")
    assert spec["cluster"] == [2, 1, 1]
    assert spec["arch"] == "sm_90a"
    assert spec["grid"] == [4, 1, 1]


def test_build_spec_no_cluster_key_when_unclustered():
    spec = build_spec(_PTX, _md(ctas_per_cga=None), (4, 1, 1), _ARGS, "ptxas")
    assert "cluster" not in spec


def test_build_spec_reads_ctas_per_cga_only_ignoring_cluster_dims():
    # The decision under test: cluster shape comes from ctas_per_cga alone. A stray cluster_dims must
    # NOT be read (the nvidia backend mirrors ctas_per_cga into cluster_dims, so ctas_per_cga is the
    # single source of truth here).
    spec = build_spec(_PTX, _md(ctas_per_cga=None, cluster_dims=(2, 1, 1)), (4, 1, 1), _ARGS, "ptxas")
    assert "cluster" not in spec


def test_build_spec_rejects_num_ctas_gt_1():
    with pytest.raises(NotImplementedError):
        build_spec(_PTX, _md(num_ctas=2), (4, 1, 1), _ARGS, "ptxas")


def test_build_spec_rejects_grid_not_divisible_by_cluster():
    with pytest.raises(NotImplementedError):
        build_spec(_PTX, _md(ctas_per_cga=(2, 1, 1)), (3, 1, 1), _ARGS, "ptxas")


def test_build_spec_cluster_survives_json_roundtrip():
    # The spec IS the ACF task's spec.json, so the cluster must survive serialization intact.
    spec = build_spec(_PTX, _md(ctas_per_cga=(2, 1, 1)), (4, 1, 1), _ARGS, "ptxas")
    assert json.loads(json.dumps(spec))["cluster"] == [2, 1, 1]


# --- global scratch (device-side TMA descriptors etc.; no GPU) ----------------------------------
def test_build_spec_records_global_scratch():
    spec = build_spec(_PTX, _md(global_scratch_size=128, global_scratch_align=128), (4, 1, 1), _ARGS, "ptxas")
    assert spec["global_scratch"] == {"size": 128, "align": 128}
    assert spec["args"] == [{"t": 0}, {"global_scratch": True}, {"null": True}]


def test_build_spec_no_global_scratch_key_when_unused():
    spec = build_spec(_PTX, _md(), (4, 1, 1), _ARGS, "ptxas")
    assert "global_scratch" not in spec
    assert spec["args"][-2:] == [{"null": True}, {"null": True}]


def test_build_spec_rejects_profile_scratch():
    with pytest.raises(NotImplementedError):
        build_spec(_PTX, _md(profile_scratch_size=64), (4, 1, 1), _ARGS, "ptxas")


class _Buf:
    def __init__(self, ptr):
        self.ptr = ptr

    def data_ptr(self):
        return self.ptr


def test_kernel_args_from_spec_passes_global_scratch_pointer():
    spec = build_spec(_PTX, _md(global_scratch_size=128, global_scratch_align=128), (4, 1, 1), _ARGS, "ptxas")
    ka = kernel_args_from_spec(spec, [_Buf(0x1000)], global_scratch=_Buf(0x2000))
    assert ka == [("ptr", 0x1000), ("ptr", 0x2000), ("null",)]


def test_kernel_args_from_spec_requires_global_scratch_buffer():
    spec = build_spec(_PTX, _md(global_scratch_size=128, global_scratch_align=128), (4, 1, 1), _ARGS, "ptxas")
    with pytest.raises(ValueError):
        kernel_args_from_spec(spec, [_Buf(0x1000)])


# --- build_spec_from_schema: Level 0 launch-metadata replaces PTX parsing (prototype) --------------
# A PTX with one user scalar param (p1) between the user pointer (p0) and the two trailing scratch ptrs.
_PTX_SCALAR = """//
.version 8.3
.target sm_90a
.address_size 64

.visible .entry ks(
    .param .u64 .ptr .global .align 16 p0,
    .param .u32 p1,
    .param .u64 .ptr .global .align 16 p2,
    .param .u64 .ptr .global .align 16 p3
)
{
    ret;
}
"""


def _schema(**over):
    """A Level 0 launch-metadata schema (see nvidia backend make_launch_metadata) for kernel `k`:
    one *fp32 pointer param, entry `k`, no cluster/scratch. Override any field via kwargs."""
    base = {
        "entry_name": "k",
        "num_warps": 4,
        "num_ctas": 1,
        "shared_mem": 0,
        "cluster_dims": [1, 1, 1],
        "global_scratch_size": 0,
        "profile_scratch_size": 0,
        "args": [{"name": "p0", "type": "*fp32", "index": 0}],
        "tensordesc_meta": [],
    }
    base.update(over)
    return base


def test_from_schema_matches_build_spec_cluster():
    # The whole point: schema-driven spec == PTX-parsed spec for the same 2-CTA kernel.
    want = build_spec(_PTX, _md(ctas_per_cga=(2, 1, 1)), (4, 1, 1), _ARGS, "ptxas")
    got = build_spec_from_schema(_schema(cluster_dims=[2, 1, 1]), _PTX, (4, 1, 1), _ARGS, "ptxas")
    assert got == want
    assert got["cluster"] == [2, 1, 1]


def test_from_schema_matches_build_spec_no_cluster():
    want = build_spec(_PTX, _md(ctas_per_cga=None), (4, 1, 1), _ARGS, "ptxas")
    got = build_spec_from_schema(_schema(cluster_dims=[1, 1, 1]), _PTX, (4, 1, 1), _ARGS, "ptxas")
    assert got == want
    assert "cluster" not in got


def test_from_schema_matches_build_spec_with_scalar():
    # A pointer + an i32 scalar: schema types (*fp32, i32) reproduce the PTX-derived kinds.
    args = [("tensor", {"id": 0, "shape": [16], "dtype": "float32", "strides": [1]}), ("scalar", 7)]
    want = build_spec(_PTX_SCALAR, _md(), (4, 1, 1), args, "ptxas")
    schema = _schema(
        entry_name="ks",
        args=[{"name": "p0", "type": "*fp32", "index": 0}, {"name": "p1", "type": "i32", "index": 1}],
    )
    got = build_spec_from_schema(schema, _PTX_SCALAR, (4, 1, 1), args, "ptxas")
    assert got == want
    assert got["args"] == [{"t": 0}, {"i32": 7}, {"null": True}, {"null": True}]


def test_from_schema_matches_build_spec_global_scratch():
    want = build_spec(_PTX, _md(global_scratch_size=128, global_scratch_align=128), (4, 1, 1), _ARGS, "ptxas")
    schema = _schema(global_scratch_size=128, global_scratch_align=128)
    got = build_spec_from_schema(schema, _PTX, (4, 1, 1), _ARGS, "ptxas")
    assert got == want


def test_from_schema_parity_mismatch_raises():
    # Schema says 2 params but only 1 runtime arg survives -> fail-open (never mis-collect).
    schema = _schema(args=[{"type": "*fp32", "index": 0}, {"type": "i32", "index": 1}])
    with pytest.raises(NotImplementedError):
        build_spec_from_schema(schema, _PTX, (4, 1, 1), _ARGS, "ptxas")


def test_from_schema_rejects_num_ctas_gt_1():
    with pytest.raises(NotImplementedError):
        build_spec_from_schema(_schema(num_ctas=2), _PTX, (4, 1, 1), _ARGS, "ptxas")


def test_from_schema_tensordesc_not_supported():
    # Documented prototype limitation: Level 0 schema lacks the flattened TMA leaf kinds.
    td_args = [
        (
            "tensordesc",
            {
                "base": {"id": 0, "shape": [16, 16], "dtype": "float16", "strides": [16, 1]},
                "desc_shape": [16, 16],
                "desc_strides": [16, 1],
                "block_shape": [16, 16],
                "padding": 0,
            },
        )
    ]
    schema = _schema(args=[{"name": "d0", "type": "tensordesc<fp16>", "index": 0}])
    with pytest.raises(NotImplementedError):
        build_spec_from_schema(schema, _PTX, (4, 1, 1), td_args, "ptxas")
