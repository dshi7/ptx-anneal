# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CLI tests: argument parsing, the _score subcommand, adapter wiring, and doctor formatting (no GPU)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptx_anneal import cli, provision


def test_tune_parser_defaults():
    # engine is provisioning (env), not a required flag; per-run surface is just --ss/--task.
    a = cli.build_parser().parse_args(["--ss", "/s.bin", "--task", "/t", "--output", "/o"])
    assert a.ss == "/s.bin" and a.task == "/t" and a.output == "/o"
    assert a.target == "ptx" and a.engine_python is None and a.engine_adapter is None
    assert not a.full_tune


def test_engine_overrides_parse():
    a = cli.build_parser().parse_args(["--engine-python", "/py", "--engine-adapter", "/a.py", "--task", "/t"])
    assert a.engine_python == "/py" and a.engine_adapter == "/a.py"


@pytest.mark.parametrize("workers", ["0", "-1"])
def test_workers_must_be_positive(workers):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--task", "/t", "--workers", workers])


@pytest.mark.parametrize("measured_ms", [0.5, 2.0, float("inf")])
@pytest.mark.parametrize("full_tune", [False, True])
def test_parallel_winner_is_admitted_using_baseline_gpu_timing(monkeypatch, tmp_path, measured_ms, full_tune):
    sample = Path(__file__).parents[1] / "sample_tasks" / "sample_task"
    argv = ["--task", str(sample), "--workers", "2", "--output", str(tmp_path)]
    args = cli.build_parser().parse_args([*argv, "--full-tune"] if full_tune else argv)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setenv("PTX_ANNEAL_NUM_WORKERS", "9")  # Explicit flag wins over the environment.
    original_budget = {"CIQ_GENERATIONS": "3", "CIQ_POOL": "16", "CIQ_CULL": "6"}
    expected_budget = {"CIQ_GENERATIONS": "20", "CIQ_POOL": "32", "CIQ_CULL": "24"} if full_tune else original_budget
    for name, value in original_budget.items():
        monkeypatch.setenv(name, value)
    budget = {
        "num_workers": 2,
        "gpus": ["3", "7"],
        "generations": int(expected_budget["CIQ_GENERATIONS"]),
        "pool_size": int(expected_budget["CIQ_POOL"]),
        "cull_size": int(expected_budget["CIQ_CULL"]),
    }
    monkeypatch.delenv("PTX_ANNEAL_FORCE_ADMIT", raising=False)
    monkeypatch.setattr(cli, "_require_ptxas", lambda args: ("/ptxas", "13.3"))
    monkeypatch.setattr(cli, "_fingerprint", lambda *a: {})
    calls = []

    def probe():
        calls.append("probe")
        return ["3", "7"]

    monkeypatch.setattr(provision, "visible_gpus", probe)

    def score(task, acf, warmup, rep, bench, ptxas):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,7"
        assert ptxas == "/ptxas"
        if acf is None:
            calls.append("baseline")
            return 1.0
        assert Path(acf).read_bytes() == b"acf"
        calls.append("winner")
        return measured_ms

    def run_adapter(command, *, env, check):
        calls.append("search")
        assert env["PTX_ANNEAL_NUM_WORKERS"] == "2"
        assert env["CUDA_VISIBLE_DEVICES"] == "3,7"
        assert {name: env[name] for name in expected_budget} == expected_budget
        Path(env["PTX_ANNEAL_RESULT"]).write_text(
            json.dumps(
                {
                    "acf": b"acf".hex(),
                    "best_ms": 0.1,
                    "evaluated": 4,
                    "valid": 4,
                    "budget": budget,
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli, "_score_one", score)
    monkeypatch.setattr(cli.subprocess, "run", run_adapter)
    assert cli._cmd_tune(args) == 0
    assert {name: os.environ[name] for name in original_budget} == original_budget
    assert calls == (["probe"] if full_tune else []) + ["baseline", "search", "winner"]
    artifacts = list(tmp_path.rglob("*.acf.json"))
    if measured_ms < 1:
        assert len(artifacts) == 1
        metadata = json.loads(artifacts[0].read_text())
        assert artifacts[0].parent == tmp_path / "sm_100a" / metadata["ir_hash"][:16]
        assert metadata["best_ms"] == measured_ms
        assert metadata["search_best_ms"] == 0.1
        assert metadata["budget"] == budget
    else:
        assert artifacts == []  # A fast score from another GPU cannot cause a false admission.


@pytest.mark.parametrize(
    ("visible", "gpus", "workers"),
    [
        (None, ["0", "1", "2", "3"], None),
        ("3,7", ["3", "7"], None),
        ("GPU-abc,GPU-def", ["GPU-abc", "GPU-def"], None),
        (None, ["0"], None),
        (None, ["0", "1", "2", "3"], 2),
    ],
)
def test_full_tune_discovers_gpus_before_scoring(monkeypatch, tmp_path, visible, gpus, workers):
    sample = Path(__file__).parents[1] / "sample_tasks" / "sample_task"
    argv = ["--task", str(sample), "--full-tune"]
    if workers is not None:
        argv += ["--workers", str(workers)]
    args = cli.build_parser().parse_args(argv)
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("PTX_ANNEAL_NUM_WORKERS", "99")
    monkeypatch.setattr(cli, "_require_ptxas", lambda args: ("/ptxas", "13.3"))
    monkeypatch.setattr(cli, "_fingerprint", lambda *a: {})
    calls = []

    def probe():
        calls.append("probe")
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible
        return gpus

    def baseline(*args):
        assert calls == ["probe"]
        calls.append("baseline")
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible
        return 1.0

    def run_adapter(command, *, env, check):
        assert calls == ["probe", "baseline"]
        calls.append("search")
        assert env["CUDA_VISIBLE_DEVICES"] == ",".join(gpus)
        assert env["PTX_ANNEAL_NUM_WORKERS"] == str(len(gpus) if workers is None else workers)
        assert (env["CIQ_GENERATIONS"], env["CIQ_POOL"], env["CIQ_CULL"]) == ("20", "32", "24")
        Path(env["PTX_ANNEAL_RESULT"]).write_text(json.dumps({"evaluated": 0, "valid": 0}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(provision, "visible_gpus", probe)
    monkeypatch.setattr(cli, "_score_one", baseline)
    monkeypatch.setattr(cli.subprocess, "run", run_adapter)
    assert cli._cmd_tune(args) == 0
    assert calls == ["probe", "baseline", "search"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible
    assert os.environ["PTX_ANNEAL_NUM_WORKERS"] == "99"


@pytest.mark.parametrize("failure", ["no_devices", "driver_error", "too_many_workers"])
def test_full_tune_rejects_unavailable_gpus_before_scoring(monkeypatch, failure):
    argv = ["--task", "/unused", "--full-tune"]
    if failure == "too_many_workers":
        argv += ["--workers", "3"]
    args = cli.build_parser().parse_args(argv)
    monkeypatch.setattr(cli, "_require_ptxas", lambda args: ("/ptxas", "13.3"))

    def probe():
        if failure == "driver_error":
            raise RuntimeError("driver failed")
        return [] if failure == "no_devices" else ["0", "1"]

    monkeypatch.setattr(provision, "visible_gpus", probe)
    monkeypatch.setattr(cli, "_score_one", lambda *a: pytest.fail("scoring started after a failed GPU probe"))
    with pytest.raises(SystemExit, match="no CUDA-visible GPUs|driver failed|exceeds"):
        cli._cmd_tune(args)


def test_output_aliases_store():
    assert cli.build_parser().parse_args(["--task", "/t", "--store", "/o"]).output == "/o"


def test_unknown_flag_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--bogus"])


def test_task_is_required():
    # The parser owns this, not a late SystemExit from _cmd_tune -- so the user gets a usage message.
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--output", "/o"])


def test_score_parser():
    a = cli._score_parser().parse_args(["--task", "/t", "--acf", "/a.acf", "--ptxas", "/p"])
    assert a.task == "/t" and a.acf == "/a.acf" and a.ptxas == "/p"
    assert cli._score_parser().parse_args(["--task", "/t"]).acf == "NONE"  # default = untuned baseline


def test_harness_cmd_invokes_this_module():
    # dev (not frozen): the adapter re-invokes `python -m ptx_anneal.cli` for scoring.
    assert "ptx_anneal.cli" in cli._harness_cmd()


def test_bundled_compileiq_adapter_exists():
    p = cli._bundled_adapter()
    assert p.endswith(os.path.join("adapters", "compileiq_adapter.py")) and os.path.exists(p)


def test_format_report_no_gpu():
    rep = {
        "ptxas_path": None,
        "ptxas_version": None,
        "ptxas_min": "13.3",
        "ptxas_ok": False,
        "engines": [],
        "gpu": {"cuda_python": False, "torch": False, "gpu": False, "detail": ""},
    }
    text = provision.format_report(rep)
    assert "ptxas: not found" in text
    assert "real tuning ready: no" in text
