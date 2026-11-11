# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Recursive Manifold discovery and batch tuning, with no network or GPU dependencies."""

import subprocess
from types import SimpleNamespace

import pytest

from ptx_anneal import cli, task

ROOT = "manifold://bucket/tree/run/tasks"
SOURCES = [ROOT + "/a", ROOT + "/group/tasks/b", ROOT + "/z"]
HARNESS = ["/python", "-m", "ptx_anneal.cli"]


@pytest.fixture
def batch(monkeypatch):
    state = SimpleNamespace(
        listing="\n".join(
            source.removeprefix("manifold://") + "/" + name
            for source in reversed(SOURCES)
            for name in ("spec.json", "kernel.ptx")
        ),
        calls=[],
        children=[],
        codes={},
        discovery_error=None,
    )
    monkeypatch.setattr(task.shutil, "which", lambda name: "/tools/manifold")
    monkeypatch.setattr(cli, "_harness_cmd", lambda: HARNESS)
    monkeypatch.setattr(cli, "_cmd_tune", lambda args: pytest.fail("batch parent must not initialize the GPU"))

    def run(argv, **kwargs):
        state.calls.append(argv)
        assert kwargs["check"] is False
        if argv[0] == "/tools/manifold":
            assert argv == [
                "/tools/manifold",
                "find",
                ROOT.removeprefix("manifold://"),
                r"^(spec\.json|kernel\.ptx)$",
                "--regex",
                "--type",
                "f",
            ]
            assert kwargs["capture_output"] and kwargs["text"] and kwargs["timeout"] > 0
            if isinstance(state.discovery_error, Exception):
                raise state.discovery_error
            return SimpleNamespace(returncode=state.discovery_error or 0, stdout=state.listing, stderr="listing failed")
        assert argv[: len(HARNESS)] == HARNESS
        assert kwargs == {"check": False}  # Inherit the environment and stream each task's output.
        args = cli.build_parser().parse_args(argv[len(HARNESS) :])
        state.children.append(args)
        code = state.codes.get(args.task, 0)
        if isinstance(code, BaseException):
            raise code
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(task.subprocess, "run", run)
    return state


@pytest.mark.parametrize("suffix", ["", "/"])
def test_discovery_recurses_sorts_and_deduplicates_complete_tasks(batch, suffix):
    batch.listing += (
        f"\n{SOURCES[0]}/spec.json\n{SOURCES[0]}/kernel.ptx\n{ROOT}/incomplete/spec.json\n{ROOT}/ptx-only/kernel.ptx\n"
    )
    assert task.manifold_tasks(ROOT + suffix) == SOURCES
    assert len(batch.calls) == 1
    assert batch.children == []


@pytest.mark.parametrize("listing", ["", ROOT + "/partial/spec.json"])
def test_empty_task_list_reports_error(batch, listing, capsys):
    batch.listing = listing
    assert cli.main(["--task-list", ROOT]) == 1
    assert "no tasks containing spec.json and kernel.ptx" in capsys.readouterr().err
    assert batch.children == []


@pytest.mark.parametrize(
    "entry",
    [
        "bucket/tree/other/spec.json",
        "bucket/tree/run/tasks-other/spec.json",
        "bucket/tree/run/tasks/a/../spec.json",
        "bucket/tree/run/tasks//spec.json",
        "bucket/tree/run/tasks/a/notes.txt",
    ],
)
def test_discovery_rejects_unexpected_listing_paths(batch, entry):
    batch.listing = entry
    with pytest.raises(task.TaskError, match="unexpected Manifold task-list entry"):
        task.manifold_tasks(ROOT)
    assert batch.children == []


@pytest.mark.parametrize("error", [1, subprocess.TimeoutExpired("manifold", 600), OSError("cannot execute manifold")])
def test_discovery_failures_prevent_tuning(batch, error, capsys):
    batch.discovery_error = error
    assert cli.main(["--task-list", ROOT]) == 1
    assert "failed to list tasks" in capsys.readouterr().err
    assert batch.children == []


def test_missing_manifold_is_actionable(batch, monkeypatch, capsys):
    monkeypatch.setattr(task.shutil, "which", lambda name: None)
    assert cli.main(["--task-list", ROOT]) == 1
    assert "requires the 'manifold' CLI on PATH" in capsys.readouterr().err
    assert batch.calls == []


@pytest.mark.parametrize("source", ["/local/tasks", "https://bucket/tree/tasks"])
def test_non_manifold_task_list_rejected_before_io(batch, source, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--task-list", source])
    assert error.value.code == 2
    assert "--task-list requires a manifold:// directory URL" in capsys.readouterr().err
    assert batch.calls == []


@pytest.mark.parametrize("source", ["manifold://bucket", ROOT + "/spec.json", ROOT + "?query=yes", ROOT + "/../tasks"])
def test_invalid_task_list_directory_rejected_before_io(batch, source):
    assert cli.main(["--task-list", source]) == 1
    assert batch.calls == []


def test_batch_forwards_all_tuning_options_to_fresh_processes(batch, capsys):
    options = [
        "--full-tune",
        "--workers",
        "4",
        "--upload-manifold",
        "--output",
        "/store with spaces",
        "--engine-python",
        "/engine/python",
        "--engine-adapter",
        "/adapter.py",
        "--ptxas",
        "/tools/ptxas",
        "--ss",
        "/catalog.bin",
        "--warmup",
        "7",
        "--rep",
        "12",
        "--bench",
        "do_bench",
        "--target",
        "ptx",
    ]
    expected = vars(cli.build_parser().parse_args(["--task-list", ROOT, *options]))
    assert cli.main(["--task-list", ROOT, *options]) == 0
    assert [args.task for args in batch.children] == SOURCES
    for args in batch.children:
        assert vars(args) == {**expected, "task": args.task, "task_list": None}
    captured = capsys.readouterr()
    assert "Found 3 tasks; tuning sequentially." in captured.out
    assert "3 completed, 0 failed, 3 total" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("code", [1, -9, OSError("child launch failed")])
def test_batch_continues_after_task_failure_and_returns_nonzero(batch, code, capsys):
    batch.codes[SOURCES[0]] = code
    assert cli.main(["--task-list", ROOT]) == 1
    assert [args.task for args in batch.children] == SOURCES
    assert all(not args.upload_manifold and not args.full_tune for args in batch.children)
    captured = capsys.readouterr()
    assert SOURCES[0] in captured.err
    assert "2 completed, 1 failed, 3 total" in captured.out


def test_keyboard_interrupt_stops_batch(batch):
    batch.codes[SOURCES[0]] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--task-list", ROOT])
    assert len(batch.children) == 1


def test_upload_validates_all_task_urls_before_starting_batch(batch, capsys):
    batch.listing += f"\n{ROOT}/group/bad/spec.json\n{ROOT}/group/bad/kernel.ptx"
    assert cli.main(["--task-list", ROOT, "--upload-manifold"]) == 1
    assert "--upload-manifold requires" in capsys.readouterr().err
    assert batch.children == []


@pytest.mark.parametrize("argv", [[], ["--task", "/task", "--task-list", ROOT]])
def test_exactly_one_task_source_required(batch, argv):
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2
    assert batch.calls == []


@pytest.mark.parametrize("command", ["doctor", "replay", "_score"])
def test_task_list_is_tuning_only(batch, command):
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--task-list", ROOT])
    assert error.value.code == 2
    assert batch.calls == []
