# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compile-only spill tuning: no engine, GPU, or GPU Python packages required."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptx_anneal import cli, ptx_launch

PTX = ".version 9.3\n.target sm_100a\n.address_size 64\n.visible .entry kernel() { ret; }\n"


@pytest.fixture
def compiler(monkeypatch, tmp_path):
    state = SimpleNamespace(
        diagnostics="ptxas info : 128 bytes stack frame, 8 bytes spill stores, 16 bytes spill loads\n",
        returncode=0,
        calls=[],
        task=tmp_path / "kernel with spaces.ptx",
    )
    state.task.write_text(PTX)

    def run(cmd, **kwargs):
        assert kwargs["capture_output"] and kwargs["text"] and kwargs["check"] is False
        state.calls.append(cmd)
        if cmd[-1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, "release 13.3, V13.3.42", "")
        assert cmd[:2] == ["/tools/ptxas", "-arch=sm_100a"]
        assert "-v" in cmd
        assert Path(cmd[2]).read_text() == PTX
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"cubin")
        return subprocess.CompletedProcess(cmd, state.returncode, "", state.diagnostics)

    monkeypatch.setattr(ptx_launch.subprocess, "run", run)
    monkeypatch.setattr(cli, "_resolve_ptxas", lambda args: "/tools/ptxas")
    # _require_ptxas mutates this variable; keep that change scoped to the test.
    monkeypatch.setenv("TRITON_PTXAS_BLACKWELL_PATH", "/tools/ptxas")
    return state


@pytest.mark.parametrize(
    ("diagnostics", "expected"),
    [
        ("128 bytes stack frame, 8 bytes spill stores, 16 bytes spill loads", 24),
        ("128 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads", 0),
        ("8 bytes spill stores, 16 bytes spill loads\n4 bytes spill stores, 4 bytes spill loads", 32),
    ],
)
def test_spill_scoring_uses_all_functions_and_preserves_acf(compiler, diagnostics, expected):
    compiler.diagnostics = diagnostics
    assert ptx_launch.ptxas_spill_bytes(PTX, "/tools/ptxas", acf_path="/a path/controls.acf") == expected
    cmd = compiler.calls[0]
    assert "--apply-controls=/a path/controls.acf" in cmd
    assert not Path(cmd[2]).exists()  # temporary PTX and cubin are cleaned up
    assert not Path(cmd[cmd.index("-o") + 1]).exists()


@pytest.mark.parametrize(
    ("returncode", "diagnostics"),
    [(1, "ptxas fatal : invalid controls"), (0, "ptxas info : Used 255 registers")],
)
def test_failed_compile_or_missing_statistics_is_invalid(compiler, capsys, returncode, diagnostics):
    compiler.returncode = returncode
    compiler.diagnostics = diagnostics
    assert cli.main(["_score", "--task", str(compiler.task), "--metric", "spill_bytes"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "INVALID\n"
    assert diagnostics in captured.err


def test_score_protocol_accepts_zero_without_an_acf(compiler, capsys):
    compiler.diagnostics = "0 bytes spill stores, 0 bytes spill loads"
    assert cli.main(["_score", "--task", str(compiler.task), "--metric", "spill_bytes"]) == 0
    assert capsys.readouterr().out == "SPILL_BYTES 0\n"
    assert not any(arg.startswith("--apply-controls") for arg in compiler.calls[0])


@pytest.fixture
def search(compiler, monkeypatch, tmp_path):
    state = SimpleNamespace(
        result={
            "acf": "00abff",
            "best_spill_bytes": 0.0,
            "evaluated": 8,
            "valid": 7,
            "timeout": 1,
            "engine_info": {"name": "compileiq", "version": "test", "path": "/engine", "python": "/engine-python"},
            "search_space": {"resolved_tag": "test", "sha256": "abcd", "variant": "default", "source": "test"},
            "budget": {"generations": 1, "pool_size": 8},
        },
        store=tmp_path / "store",
        env=None,
    )
    compile_run = subprocess.run

    def run(cmd, **kwargs):
        if cmd[0] != "/engine-python":
            return compile_run(cmd, **kwargs)
        assert cmd == ["/engine-python", cli._bundled_adapter()]
        state.env = kwargs["env"]
        Path(state.env["PTX_ANNEAL_RESULT"]).write_text(json.dumps(state.result))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setenv("PTX_ANNEAL_ENGINE_PYTHON", "/engine-python")
    monkeypatch.delenv("PTX_ANNEAL_ENGINE_ADAPTER", raising=False)
    monkeypatch.setenv("COMPILE_IQ_STORE", str(state.store))
    return state


def test_register_spill_saves_zero_spill_winner_and_provenance(compiler, search, tmp_path, capsys):
    ss = tmp_path / "space.bin"
    ss.write_bytes(b"search-space")
    assert cli.main(["register_spill", "--task", str(compiler.task), "--ss", str(ss)]) == 0
    acf = next((search.store / "register_spill").rglob("*.acf"))
    assert acf.read_bytes() == bytes.fromhex("00abff")
    assert acf.name.endswith(".13.3.acf")
    meta = json.loads(Path(str(acf) + ".json").read_text())
    assert meta["objective"] == "spill_bytes"
    assert meta["baseline_spill_bytes"] == 24 and meta["best_spill_bytes"] == 0
    assert meta["ptxas_version"] == "13.3"
    assert meta["engine_info"] == search.result["engine_info"]
    assert meta["search_space"] == search.result["search_space"]
    assert meta["budget"] == search.result["budget"]
    assert "best_ms" not in meta
    assert search.env["PTX_ANNEAL_TASK"] == str(compiler.task)
    assert search.env["PTX_ANNEAL_PTXAS"] == "/tools/ptxas"
    assert search.env["PTX_ANNEAL_METRIC"] == "spill_bytes"
    assert search.env["PTX_ANNEAL_SS"] == str(ss)
    assert json.loads(search.env["PTX_ANNEAL_SCORE_CMD"]) == [
        *cli._harness_cmd(),
        "_score",
        "--metric",
        "spill_bytes",
    ]
    assert not Path(search.env["PTX_ANNEAL_RESULT"]).exists()
    assert "baseline=24 best=0 spill bytes" in capsys.readouterr().out


def test_register_spill_respects_output(compiler, search, tmp_path):
    output = tmp_path / "chosen-store"
    assert cli.main(["register_spill", "--task", str(compiler.task), "--output", str(output)]) == 0
    assert list(output.rglob("*.acf"))
    assert not search.store.exists()


@pytest.mark.parametrize("best", [24, 40, None, float("inf"), float("nan"), -1, 0.5])
def test_no_improvement_or_invalid_candidate_is_not_saved(compiler, search, best, capsys):
    search.result["best_spill_bytes"] = best
    assert cli.main(["register_spill", "--task", str(compiler.task)]) == 0
    assert not search.store.exists()
    assert "no candidate admitted" in capsys.readouterr().out


def test_zero_baseline_is_a_success_without_admission(compiler, search):
    compiler.diagnostics = "0 bytes spill stores, 0 bytes spill loads"
    assert cli.main(["register_spill", "--task", str(compiler.task)]) == 0
    assert not search.store.exists()


def test_baseline_failure_aborts_search(compiler, search, capsys):
    compiler.returncode = 1
    assert cli.main(["register_spill", "--task", str(compiler.task)]) == 1
    assert search.env is None
    assert "ptxas failed" in capsys.readouterr().err


def test_engine_failure_is_reported(compiler, search, capsys):
    search.result["engine_error"] = "engine crashed"
    assert cli.main(["register_spill", "--task", str(compiler.task)]) == 1
    assert not search.store.exists()
    assert "engine crashed" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["missing", "directory", "malformed"])
def test_bad_task_reports_failure_before_search(compiler, search, tmp_path, capsys, kind):
    task = tmp_path / kind
    if kind == "directory":
        task.mkdir()
    elif kind == "malformed":
        task.write_text("not PTX")
    assert cli.main(["register_spill", "--task", str(task)]) == 1
    assert search.env is None
    assert "[register_spill]" in capsys.readouterr().err


def test_register_spill_keeps_toolchain_floor(compiler, search, monkeypatch):
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.2")
    with pytest.raises(SystemExit, match="13.2 < 13.3"):
        cli.main(["register_spill", "--task", str(compiler.task)])
    assert search.env is None
