# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Standalone replay and clustered driver launch coverage, with CPU-only stubs."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ptx_anneal import cli, replay
from ptx_anneal import ptx_launch as L
from ptx_anneal.task import Task

SAMPLE_TASK = str(Path(__file__).resolve().parents[1] / "sample_tasks" / "sample_task")


@pytest.fixture(autouse=True)
def isolated_ptxas_env(monkeypatch):
    monkeypatch.setenv("TRITON_PTXAS_BLACKWELL_PATH", "")


def test_replay_cli_skips_engine_and_overrides_captured_ptxas(monkeypatch, capsys):
    run = Mock(return_value=0.25)
    monkeypatch.setattr(replay, "replay", run)
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    monkeypatch.setattr(cli, "_cmd_tune", Mock(side_effect=AssertionError("tuning was invoked")))
    monkeypatch.setattr(cli, "_bundled_adapter", Mock(side_effect=AssertionError("engine was resolved")))
    assert cli.main(["replay", "--task", SAMPLE_TASK, "--ptxas", "/explicit/ptxas"]) == 0
    assert run.call_args.args[1] == "/explicit/ptxas"
    assert "MS 0.25" in capsys.readouterr().out


@pytest.mark.parametrize("version", [None, "13.2"])
def test_replay_enforces_ptxas_floor(monkeypatch, version):
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: version)
    monkeypatch.setattr(replay, "replay", Mock(side_effect=AssertionError("replay started")))
    with pytest.raises(SystemExit, match="13.3"):
        cli.main(["replay", "--task", SAMPLE_TASK, "--ptxas", "/old/ptxas"])


@pytest.mark.parametrize("outcome", [RuntimeError("launch failed"), float("nan"), float("inf"), 0.0])
def test_replay_cli_failure_is_nonzero(monkeypatch, outcome, capsys):
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    run = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
    monkeypatch.setattr(replay, "replay", run)
    assert cli.main(["replay", "--task", SAMPLE_TASK, "--ptxas", "/ptxas"]) == 1
    captured = capsys.readouterr()
    assert "[replay]" in captured.err
    assert "MS " not in captured.out


@pytest.mark.parametrize("bench", ["cudagraph", "do_bench"])
def test_replay_launches_spec_without_acf(monkeypatch, bench):
    events = []
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: events.append("sync")))
    )
    spec = {
        "entry": "k",
        "arch": "sm_100a",
        "shared": 100,
        "cluster": [2, 1, 1],
        "grid": [4, 1, 1],
        "block": [256, 1, 1],
    }
    task = Task("/task", "PTX", spec)
    tensors, descriptors, addresses = [object()], [object()], [1234]
    args = [("tma", 1234)]
    compile_ptx = Mock(return_value=b"cubin")
    kernel = SimpleNamespace(launch=lambda *a: events.append("launch"), unload=Mock())
    load = Mock(return_value=kernel)
    monkeypatch.setattr(L, "ptxas_compile", compile_ptx)
    monkeypatch.setattr(L, "load_cubin", load)
    monkeypatch.setattr(L, "alloc_tensors", Mock(return_value=tensors))
    maps = Mock(return_value=(descriptors, addresses))
    monkeypatch.setattr(L, "build_tensormaps", maps)
    scratch = object()
    alloc_scratch = Mock(return_value=scratch)
    monkeypatch.setattr(L, "alloc_global_scratch", alloc_scratch)
    make_args = Mock(return_value=args)
    monkeypatch.setattr(L, "kernel_args_from_spec", make_args)

    def measure(*a, **kw):
        assert events == ["sync", "launch", "sync"]
        assert a == (kernel, spec["grid"], spec["block"], args)
        assert kw == {"warmup": 2, "reps": 3 if bench == "cudagraph" else 20}
        return 0.125

    monkeypatch.setattr(L, "bench_ms_" + bench, measure)
    assert replay.replay(task, "/ptxas", warmup=2, rep=20, bench=bench) == 0.125
    compile_ptx.assert_called_once_with("PTX", "/ptxas", arch="sm_100a")
    load.assert_called_once_with(b"cubin", "k", 100, cluster=[2, 1, 1])
    maps.assert_called_once_with(spec, tensors)
    alloc_scratch.assert_called_once_with(spec)
    make_args.assert_called_once_with(spec, tensors, addresses, scratch)
    kernel.unload.assert_called_once()

    def failed_measure(*a, **kw):
        raise RuntimeError("device failure")

    monkeypatch.setattr(L, "bench_ms_" + bench, failed_measure)
    with pytest.raises(RuntimeError, match="device failure"):
        replay.replay(task, "/ptxas", bench=bench)
    assert kernel.unload.call_count == 2


@pytest.mark.parametrize("cluster", [None, [2, 1, 1]])
def test_driver_uses_extended_launch_for_clusters(monkeypatch, cluster):
    calls = []
    driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        cuModuleLoadData=lambda b: (0, "module"),
        cuModuleGetFunction=lambda *a: (0, "function"),
        CUlaunchAttribute=lambda: SimpleNamespace(value=SimpleNamespace(clusterDim=SimpleNamespace())),
        CUlaunchAttributeID=SimpleNamespace(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION="cluster"),
        CUlaunchConfig=SimpleNamespace,
        cuLaunchKernel=lambda *a: (calls.append(("plain", a)) or 0,),
        cuLaunchKernelEx=lambda *a: (calls.append(("extended", a)) or 0,),
    )
    monkeypatch.setattr(L, "_drv", driver)
    monkeypatch.setattr(L, "_ensure_init", lambda: None)
    kernel = L.load_cubin(b"cubin", "k", 123, cluster=cluster)
    kernel.launch([4, 1, 1], [256, 1, 1], [("i32", 7)], stream=42)
    kind, args = calls[0]
    assert kind == ("extended" if cluster else "plain")
    if cluster:
        config = args[0]
        assert (config.gridDimX, config.gridDimY, config.gridDimZ) == (4, 1, 1)
        assert (config.blockDimX, config.blockDimY, config.blockDimZ) == (256, 1, 1)
        assert config.hStream == 42 and config.sharedMemBytes == 123
        assert config.numAttrs == 1 and config.attrs[0].id == "cluster"
        dim = config.attrs[0].value.clusterDim
        assert (dim.x, dim.y, dim.z) == (2, 1, 1)
        with pytest.raises(ValueError, match="not divisible"):
            kernel.launch([3, 1, 1], [256, 1, 1], [])


@pytest.mark.parametrize("cluster", [[0, 1, 1], [2, 1], [1.5, 1, 1]])
def test_bad_cluster_rejected_before_cuda_initialization(cluster):
    with pytest.raises(ValueError, match="invalid cluster"):
        L.load_cubin(b"cubin", "k", 0, cluster=cluster)
