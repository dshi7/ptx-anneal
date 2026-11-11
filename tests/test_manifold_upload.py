# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Manifold publication through the tuning CLI, without network, engine or GPU dependencies."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptx_anneal import cli, task
from ptx_anneal.store import LocalStore
from ptx_anneal.target.ptx import ptx_ir_hash

SAMPLE_TASK = Path(__file__).resolve().parents[1] / "sample_tasks" / "sample_task"
REMOTE = "manifold://bucket/tree/run/tasks/task"
DESTINATION = "bucket/tree/run/store"


@pytest.fixture
def tuning(monkeypatch, tmp_path):
    sample = task.load(SAMPLE_TASK)
    ir_hash = ptx_ir_hash(sample.ir)
    state = SimpleNamespace(
        calls=[],
        uploads={},
        fail=None,
        timeout=False,
        result={
            "acf": "00abff",
            "best_ms": 0.5,
            "evaluated": 8,
            "valid": 8,
            "engine_info": {"name": "compileiq", "version": "test", "path": "/engine", "python": "/python"},
            "search_space": {"resolved_tag": "test-catalog", "sha256": "abcd", "variant": "default", "source": "test"},
        },
        store=tmp_path / "store",
        arch=sample.arch,
        ir_hash=ir_hash,
        remote_dir=f"{DESTINATION}/{sample.arch}/{ir_hash[:16]}",
    )
    monkeypatch.setattr(task.shutil, "which", lambda name: "/tools/manifold" if name == "manifold" else None)
    monkeypatch.setattr(cli, "_require_ptxas", lambda args: ("/ptxas", "13.3"))
    monkeypatch.setattr(cli, "_score_one", lambda *args: 1.0)
    monkeypatch.setattr(cli, "_fingerprint", lambda *args: {"digest": "test-fingerprint"})
    monkeypatch.delenv("PTX_ANNEAL_FORCE_ADMIT", raising=False)
    monkeypatch.delenv("PTX_ANNEAL_ENGINE_ADAPTER", raising=False)

    def run(argv, **kwargs):
        state.calls.append(argv)
        if argv[0] != "/tools/manifold":
            # The engine adapter's subprocess contract: consume a local task and emit result JSON.
            env = kwargs["env"]
            assert task.load(env["PTX_ANNEAL_TASK"]).entry == "add_kernel"
            Path(env["PTX_ANNEAL_RESULT"]).write_text(json.dumps(state.result))
        else:
            operation = argv[1]
            if state.fail == operation or (state.fail == "sidecar" and argv[-1].endswith(".acf.json")):
                if state.timeout:
                    raise subprocess.TimeoutExpired(argv, 120)
                return SimpleNamespace(returncode=1, stderr="permission denied or destination exists", stdout="")
            if operation == "get":
                shutil.copyfile(SAMPLE_TASK / Path(argv[2]).name, argv[3])
            elif operation == "mkdir":
                assert argv == ["/tools/manifold", "mkdir", "-p", state.remote_dir]
            elif operation == "put":
                assert kwargs["check"] is False
                assert argv[:3] == ["/tools/manifold", "put", "--overwrite"]
                state.uploads[argv[-1]] = Path(argv[-2]).read_bytes()
            else:
                pytest.fail(f"unexpected Manifold command: {argv}")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(task.subprocess, "run", run)
    return state


@pytest.mark.parametrize("source", [REMOTE, REMOTE + "/"])
def test_tuning_uploads_store_directly_under_run_store(tuning, source, capsys):
    assert cli.main(["--task", source, "--output", str(tuning.store), "--upload-manifold"]) == 0
    acf = next(tuning.store.rglob("*.acf"))
    assert acf.parent == tuning.store / "sm_100a" / acf.name.split(".", 1)[0]
    assert acf.name == f"{tuning.ir_hash[:16]}.13.3.acf"
    remote_acf = f"{DESTINATION}/{acf.relative_to(tuning.store).as_posix()}"
    assert tuning.uploads[remote_acf] == acf.read_bytes() == bytes.fromhex("00abff")
    assert tuning.uploads[remote_acf + ".json"] == Path(str(acf) + ".json").read_bytes()
    meta = json.loads(tuning.uploads[remote_acf + ".json"])
    assert meta["ptxas_version"] == "13.3"
    assert meta["ir_hash"] == tuning.ir_hash
    assert meta["engine_info"] == tuning.result["engine_info"]
    assert meta["search_space"] == tuning.result["search_space"]
    assert len(tuning.uploads) == 2
    assert all(path.startswith(f"{DESTINATION}/sm_100a/{tuning.ir_hash[:16]}/") for path in tuning.uploads)
    assert f"uploaded task store -> manifold://{DESTINATION}" in capsys.readouterr().out


@pytest.mark.parametrize("suffix", ["", "/"])
def test_manifold_store_url_matches_requested_layout(suffix):
    run = "tc_bench_ci/tree/magnon/aps-aps-v5_combo_inductorfix3_0910-0911-row9_dpp_dcb0-36ec92e9dd4c"
    assert task.manifold_store_url(f"manifold://{run}/tasks/1e359f0358047d9e{suffix}") == f"manifold://{run}/store"


def test_upload_includes_task_versions_and_cubin_but_not_other_kernels(tuning):
    store = LocalStore(str(tuning.store))
    previous = store.write("ptx", tuning.arch, tuning.ir_hash, b"previous-version", toolchain_version="13.4")
    cubin = store.write_cubin(tuning.arch, tuning.ir_hash, "13.4", b"cached-cubin")
    store.write("ptx", tuning.arch, "other-hash", b"unrelated", toolchain_version="13.3")
    Path(previous + ".tmp.123").write_bytes(b"partial-write")
    assert cli.main(["--task", REMOTE, "--output", str(tuning.store), "--upload-manifold"]) == 0
    assert tuning.uploads[f"{DESTINATION}/{Path(previous).relative_to(tuning.store).as_posix()}"] == b"previous-version"
    assert tuning.uploads[f"{DESTINATION}/{Path(cubin).relative_to(tuning.store).as_posix()}"] == b"cached-cubin"
    assert len(tuning.uploads) == 5  # Two ACF/sidecar pairs and the optional cubin.
    assert all(path.startswith(tuning.remote_dir + "/") for path in tuning.uploads)
    assert not any(".tmp." in path or "other-hash" in path for path in tuning.uploads)


def test_repeated_upload_refreshes_same_store_directory(tuning):
    argv = ["--task", REMOTE, "--output", str(tuning.store), "--upload-manifold"]
    assert cli.main(argv) == 0
    tuning.result["acf"] = "aabbcc"
    assert cli.main(argv) == 0
    assert len(tuning.uploads) == 2
    acf_data = next(data for path, data in tuning.uploads.items() if path.endswith(".acf"))
    assert acf_data == bytes.fromhex("aabbcc")


def test_upload_requires_a_loadable_remote_task_directory(tuning):
    tuning.fail = "get"
    assert cli.main(["--task", REMOTE, "--upload-manifold"]) == 1
    assert not tuning.uploads
    assert all(argv[:2] == ["/tools/manifold", "get"] for argv in tuning.calls)


@pytest.mark.parametrize("outcome", ["slower", "invalid", "no_baseline"])
def test_no_admission_does_not_publish(tuning, monkeypatch, capsys, outcome):
    if outcome == "slower":
        tuning.result["best_ms"] = 2.0
    elif outcome == "invalid":
        tuning.result["acf"] = None
    else:
        monkeypatch.setattr(cli, "_score_one", lambda *args: float("inf"))
    assert cli.main(["--task", REMOTE, "--output", str(tuning.store), "--upload-manifold"]) == 0
    assert not tuning.uploads
    assert not any(argv[1] == "mkdir" for argv in tuning.calls)
    assert "Manifold upload skipped: no ACF admitted." in capsys.readouterr().out


def test_force_admitted_acf_can_be_uploaded(tuning, monkeypatch):
    tuning.result["best_ms"] = 2.0
    monkeypatch.setenv("PTX_ANNEAL_FORCE_ADMIT", "1")
    assert cli.main(["--task", REMOTE, "--output", str(tuning.store), "--upload-manifold"]) == 0
    sidecar = next(data for path, data in tuning.uploads.items() if path.endswith(".json"))
    assert json.loads(sidecar)["forced"] is True


def test_upload_is_opt_in(tuning):
    assert cli.main(["--task", REMOTE, "--output", str(tuning.store)]) == 0
    assert list(tuning.store.rglob("*.acf"))
    assert not tuning.uploads
    assert not any(argv[1] == "mkdir" for argv in tuning.calls)


@pytest.mark.parametrize("failure", ["mkdir", "put", "sidecar", "timeout"])
def test_upload_failure_is_nonzero_and_keeps_local_result(tuning, capsys, failure):
    tuning.fail = "put" if failure == "timeout" else failure
    tuning.timeout = failure == "timeout"
    assert cli.main(["--task", REMOTE, "--output", str(tuning.store), "--upload-manifold"]) == 1
    acf = next(tuning.store.rglob("*.acf"))
    assert acf.read_bytes() == bytes.fromhex("00abff")
    assert Path(str(acf) + ".json").is_file()
    captured = capsys.readouterr()
    assert f"Local ACF retained at {acf}" in captured.err
    assert "uploaded task store" not in captured.out
    assert "test-fingerprint" in captured.out
    if failure == "mkdir":
        assert not any(argv[1] == "put" for argv in tuning.calls)


@pytest.mark.parametrize("source", ["/local/task", "https://bucket/tree/task"])
def test_upload_rejects_non_manifold_tasks_before_io(tuning, source, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--task", source, "--upload-manifold"])
    assert error.value.code == 2
    assert "--upload-manifold requires --task manifold://" in capsys.readouterr().err
    assert tuning.calls == []


@pytest.mark.parametrize(
    "source",
    [
        REMOTE + "/spec.json",
        REMOTE + "/kernel.ptx",
        REMOTE + "?query=yes",
        REMOTE + "#fragment",
        "manifold://bucket/tree/run/task",
        "manifold://bucket/tree/run/tasks/../task",
        "manifold://bucket/tree/run//tasks/task",
        "manifold://bucket",
    ],
)
def test_upload_rejects_invalid_task_directory_layout_before_io(tuning, source):
    with pytest.raises(SystemExit) as error:
        cli.main(["--task", source, "--upload-manifold"])
    assert error.value.code == 2
    assert tuning.calls == []


@pytest.mark.parametrize("command", ["replay", "doctor", "_score"])
def test_non_tuning_commands_reject_upload_option(tuning, command):
    args = [] if command == "doctor" else ["--task", REMOTE]
    with pytest.raises(SystemExit) as error:
        cli.main([command, *args, "--upload-manifold"])
    assert error.value.code == 2
    assert tuning.calls == []


def test_missing_provenance_fails_before_remote_writes(tuning, tmp_path):
    acf = tmp_path / tuning.arch / tuning.ir_hash / f"{tuning.ir_hash}.13.3.acf"
    acf.parent.mkdir(parents=True)
    acf.write_bytes(b"acf")
    with pytest.raises(task.TaskError, match="missing local file"):
        task.upload_store(REMOTE, str(tmp_path), str(acf))
    assert tuning.calls == []
