# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CompileIQ adapter: search-space resolution + engine identity (no GPU, no engine installed).

The adapter is the bring-your-own-engine boundary: it imports ONLY ``compileiq`` + stdlib. That is
what lets these tests run anywhere -- we inject a stub ``compileiq`` into ``sys.modules``, so CI
never needs the real engine (which is >=3.11,<3.14 + glibc 2.34) to cover this logic.
"""

import hashlib
import importlib.util
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

ADAPTER = "ptx_anneal.adapters.compileiq_adapter"


def _load_adapter():
    """Import the adapter fresh (it is a standalone script, normally run by its own interpreter)."""
    sys.modules.pop(ADAPTER, None)
    spec = importlib.util.find_spec(ADAPTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_compileiq(monkeypatch):
    """A minimal stand-in for the engine, recording what the adapter asked it for."""
    calls = {}

    class LocalSearchSpaceBin:
        def __init__(self, path):
            calls["local_path"] = str(path)
            self.path = str(path)

    class PtxasSearchSpace:
        def __init__(self, **kw):
            calls["selector"] = kw
            self.resolution_metadata = None

        def retrieve(self):
            calls["retrieved"] = True
            self.resolution_metadata = types.SimpleNamespace(
                as_dict=lambda: {
                    "compiler": "ptxas",
                    "compiler_version": "13.3",
                    "variant": "default",
                    "requested_tag": "latest",
                    "resolved_tag": "search-spaces-2026.05.22",
                    "sha256": "deadbeef",
                    "size_bytes": 13632,
                    "source": "github_release",
                }
            )
            return "/cache/ptxas13.3_search_space.bin"

    pkg = types.ModuleType("compileiq")
    pkg.__path__ = []  # namespace-ish package, like the real one
    compilers = types.ModuleType("compileiq.search_spaces.compilers")
    compilers.LocalSearchSpaceBin = LocalSearchSpaceBin
    compilers.PtxasSearchSpace = PtxasSearchSpace
    search_spaces = types.ModuleType("compileiq.search_spaces")
    ciq = types.ModuleType("compileiq.ciq")
    ciq.__file__ = "/fake/site-packages/compileiq/ciq.py"
    ciq.Search = object

    # A real `import a.b` binds b onto a; sys.modules injection alone does not, so do it by hand or
    # attribute access (`compileiq.ciq.__file__`) fails in a way the real engine never would.
    pkg.ciq = ciq
    pkg.search_spaces = search_spaces
    search_spaces.compilers = compilers

    for name, mod in [
        ("compileiq", pkg),
        ("compileiq.search_spaces", search_spaces),
        ("compileiq.search_spaces.compilers", compilers),
        ("compileiq.ciq", ciq),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_explicit_ss_wins_and_is_recorded(fake_compileiq, tmp_path):
    ss = tmp_path / "my.bin"
    ss.write_bytes(b"\x00")
    provider, meta = _load_adapter()._resolve_search_space(str(ss))
    assert provider.path == str(ss)
    assert meta["source"] == "PTX_ANNEAL_SS" and meta["path"] == str(ss)
    assert "selector" not in fake_compileiq  # the catalog was never consulted


def test_missing_explicit_ss_fails_loudly(fake_compileiq, tmp_path):
    with pytest.raises(SystemExit, match="search space not found"):
        _load_adapter()._resolve_search_space(str(tmp_path / "nope.bin"))


def test_unset_ss_resolves_the_catalog_and_returns_provenance(fake_compileiq, monkeypatch):
    for var in ("CIQ_SS_VERSION", "CIQ_SS_VARIANT", "CIQ_SS_TAG"):
        monkeypatch.delenv(var, raising=False)
    _, meta = _load_adapter()._resolve_search_space("")
    # Nothing forwarded: the catalog's own defaults stay the single source of truth.
    assert fake_compileiq["selector"] == {}
    # Resolved eagerly, so a fetch failure surfaces here and the sidecar gets its provenance.
    assert fake_compileiq["retrieved"] is True
    assert meta["resolved_tag"] == "search-spaces-2026.05.22" and meta["sha256"] == "deadbeef"


def test_ciq_ss_env_is_forwarded_as_a_selector(fake_compileiq, monkeypatch):
    monkeypatch.setenv("CIQ_SS_TAG", "search-spaces-2026.05.22")
    monkeypatch.setenv("CIQ_SS_VARIANT", "att")
    monkeypatch.delenv("CIQ_SS_VERSION", raising=False)
    _load_adapter()._resolve_search_space("")
    assert fake_compileiq["selector"] == {"variant": "att", "tag": "search-spaces-2026.05.22"}


@pytest.fixture(autouse=True)
def no_manifold(monkeypatch):
    """Hide any real `manifold` on PATH: a devserver has one, CI does not, and tests must not care."""
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda cmd, *a, **kw: None if cmd == "manifold" else real_which(cmd, *a, **kw))


@pytest.fixture
def fake_manifold(tmp_path, monkeypatch):
    """A stub `manifold get <bucket>/<path> <dest>` serving objects from a local dir; returns that dir."""
    remote = tmp_path / "remote"
    knobs = remote / "tc_bench_ci" / "tree" / "compileiq" / "ptxas_knobs"
    knobs.mkdir(parents=True)
    (knobs / "ptxas13.3_search_space.bin").write_bytes(b"ss-13.3")
    tool = tmp_path / "manifold"
    tool.write_text(
        f'#!/bin/sh\n[ "$1" = get ] || exit 2\n'
        f'[ -f "{remote}/$2" ] || {{ echo "no such object: $2" >&2; exit 3; }}\ncp "{remote}/$2" "$3"\n'
    )
    tool.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda cmd, *a, **kw: str(tool) if cmd == "manifold" else None)
    return remote


def test_manifold_downloads_the_search_space_for_this_ptxas(fake_compileiq, fake_manifold):
    provider, meta = _load_adapter()._resolve_search_space("", "13.3")
    with open(provider.path, "rb") as f:
        assert f.read() == b"ss-13.3"
    # The sidecar records which object was downloaded (not the temp path) and what it hashed to.
    assert meta["source"] == "manifold"
    assert meta["url"] == "manifold://tc_bench_ci/tree/compileiq/ptxas_knobs/ptxas13.3_search_space.bin"
    assert meta["sha256"] == hashlib.sha256(b"ss-13.3").hexdigest()
    assert "selector" not in fake_compileiq  # the catalog was never consulted


def test_without_manifold_the_catalog_is_used(fake_compileiq):
    _, meta = _load_adapter()._resolve_search_space("", "13.3")
    assert fake_compileiq["retrieved"] is True
    assert meta["resolved_tag"] == "search-spaces-2026.05.22"


def test_failed_manifold_download_is_fatal_not_a_silent_catalog_fallback(fake_compileiq, fake_manifold):
    with pytest.raises(SystemExit, match="search-space download failed"):
        _load_adapter()._resolve_search_space("", "13.4")
    assert "selector" not in fake_compileiq


def test_unknown_ptxas_version_with_manifold_fails_loudly(fake_compileiq, fake_manifold):
    with pytest.raises(SystemExit, match="ptxas --version"):
        _load_adapter()._resolve_search_space("", "")


def test_explicit_ss_beats_manifold(fake_compileiq, fake_manifold, tmp_path):
    ss = tmp_path / "my.bin"
    ss.write_bytes(b"\x00")
    _, meta = _load_adapter()._resolve_search_space(str(ss), "13.3")
    assert meta["source"] == "PTX_ANNEAL_SS"


def test_ptxas_version_is_asked_of_the_ptxas_itself(tmp_path):
    ptxas = tmp_path / "ptxas"
    ptxas.write_text('#!/bin/sh\necho "Cuda compilation tools, release 13.3, V13.3.42"\n')
    ptxas.chmod(0o755)
    adapter = _load_adapter()
    assert adapter._ptxas_version(str(ptxas)) == "13.3"
    # Unrunnable or unset is "unknown", not a crash -- the manifold download then fails with a clear message.
    assert adapter._ptxas_version(str(tmp_path / "missing")) == ""
    assert adapter._ptxas_version("") == ""


def test_engine_info_identifies_the_engine(fake_compileiq):
    info = _load_adapter()._engine_info()
    assert info["name"] == "compileiq"
    # compileiq is a namespace package (__file__ is None), so the path comes from a real submodule.
    assert info["path"] == "/fake/site-packages/compileiq"
    assert info["python"] == sys.executable
    assert info["version"]  # "unknown" when metadata is unavailable, never missing


def test_task_timeout_on_and_clock_off_by_default(fake_compileiq, monkeypatch):
    """A hung candidate must not stall the search, so the timeout is on unless disabled with 0.

    The clock knob stays off: unset means *absent*, so the GPU clocks are not touched at all
    (``gpu_benchmark_mode(None)`` would warn, or raise).
    """
    mod = _load_adapter()
    for var in ("CIQ_TASK_TIMEOUT", "CIQ_CLOCK_MHZ"):
        monkeypatch.delenv(var, raising=False)
    assert mod._task_timeout() == mod.DEFAULT_TASK_TIMEOUT_S
    monkeypatch.setenv("CIQ_TASK_TIMEOUT", "0")
    assert mod._task_timeout() is None
    assert mod._env_num("CIQ_CLOCK_MHZ", int) is None
    # nullcontext, not the engine's clock manager -- and reached without importing compileiq.utils.
    with mod._benchmark_mode(None):
        pass


def test_task_timeout_and_clock_are_read_when_set(fake_compileiq, monkeypatch):
    mod = _load_adapter()
    monkeypatch.setenv("CIQ_TASK_TIMEOUT", "20")
    monkeypatch.setenv("CIQ_CLOCK_MHZ", "1965")
    assert mod._task_timeout() == 20.0
    assert mod._env_num("CIQ_CLOCK_MHZ", int) == 1965


def test_clock_lock_uses_the_engines_context_manager(fake_compileiq, monkeypatch):
    """When a clock IS requested we must use the engine's own gpu_benchmark_mode, non-fatally: a box
    that cannot lock clocks should still produce a (noisier) result, and the fingerprint records the
    clock state that actually applied."""
    seen = {}
    gpu = types.ModuleType("compileiq.utils.gpu")

    def gpu_benchmark_mode(**kw):
        seen.update(kw)
        return __import__("contextlib").nullcontext()

    gpu.gpu_benchmark_mode = gpu_benchmark_mode
    utils = types.ModuleType("compileiq.utils")
    utils.gpu = gpu
    sys.modules["compileiq"].utils = utils
    monkeypatch.setitem(sys.modules, "compileiq.utils", utils)
    monkeypatch.setitem(sys.modules, "compileiq.utils.gpu", gpu)

    with _load_adapter()._benchmark_mode(1965):
        pass
    assert seen["clock_mhz"] == 1965 and seen["raise_on_failure"] is False


def test_adapter_never_imports_ptx_anneal():
    """The engine-free boundary: the adapter must be runnable in an interpreter without ptx_anneal."""
    src = importlib.util.find_spec(ADAPTER).origin
    with open(src) as f:
        body = f.read()
    assert "import ptx_anneal" not in body and "from ptx_anneal" not in body


@pytest.mark.parametrize("metric", ["ms", "spill_bytes"])
def test_search_metric_protocol_and_zero_winner(fake_compileiq, monkeypatch, tmp_path, metric):
    """Exercise Search's objective callback, including failed/malformed scorers and zero scores."""
    adapter = _load_adapter()
    result_path = tmp_path / "result.json"
    for name, value in {
        "PTX_ANNEAL_SCORE_CMD": json.dumps(["scorer"]),
        "PTX_ANNEAL_TASK": "/task.ptx",
        "PTX_ANNEAL_PTXAS": "/ptxas",
        "PTX_ANNEAL_RESULT": str(result_path),
        "CIQ_TASK_TIMEOUT": "20",
        "CIQ_POOL": "10",
        "CIQ_GENERATIONS": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("PTX_ANNEAL_SS", raising=False)
    if metric == "spill_bytes":
        monkeypatch.setenv("PTX_ANNEAL_METRIC", metric)
        # A clock pin from a benchmarking session must not make compile-only tuning require a GPU.
        monkeypatch.setenv("CIQ_CLOCK_MHZ", "1965")
    else:
        monkeypatch.delenv("PTX_ANNEAL_METRIC", raising=False)  # existing adapter clients default to ms
        monkeypatch.delenv("CIQ_CLOCK_MHZ", raising=False)
    monkeypatch.setattr(adapter, "_ptxas_version", lambda p: "13.3")
    replies = iter(["0", "3", "1.5", "INVALID", "abc", "nan", "inf", "-1", "exit", "timeout"])
    scores = []

    def run(cmd, **kwargs):
        assert cmd[:3] == ["scorer", "--task", "/task.ptx"]
        assert cmd[cmd.index("--ptxas") + 1] == "/ptxas"
        assert Path(cmd[cmd.index("--acf") + 1]).is_file()
        assert ("--warmup" in cmd) == (metric == "ms")
        assert kwargs["timeout"] == 20
        value = next(replies)
        if value == "timeout":
            raise subprocess.TimeoutExpired(cmd, 20)
        if value == "exit":
            return subprocess.CompletedProcess(cmd, 1, f"{metric.upper()} 0\n", "failed")
        return subprocess.CompletedProcess(cmd, 0, f"{metric.upper()} {value}\n", "")

    class Search:
        def __init__(self, objective_function, search_space, search_config, exit_on_failure):
            self.objective = objective_function
            assert search_config.problem_type == "min"
            assert exit_on_failure is False

        def start(self, **kwargs):
            assert kwargs == {"task_timeout": 25}
            for index in range(10):
                scores.append(self.objective(f"{index:02x}"))
            return types.SimpleNamespace(
                get_results=lambda: scores,
                get_best_result=lambda: {"score_1": 0, "params": "00"},
            )

    ciq_types = types.ModuleType("compileiq.types")
    ciq_types.INVALID_SCORE = float("inf")
    ciq_types.SearchConfiguration = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, "compileiq.types", ciq_types)
    monkeypatch.setattr(sys.modules["compileiq.ciq"], "Search", Search)
    monkeypatch.setattr(adapter.subprocess, "run", run)

    assert adapter.main() == 0
    result = json.loads(result_path.read_text())
    assert result[f"best_{metric}"] == 0 and result["acf"] == "00"
    assert result["valid"] == (3 if metric == "ms" else 2)
    assert result["evaluated"] == 10 and result["timeout"] == 1
    assert scores[:2] == [0, 3]
    assert all(score == float("inf") for score in scores[3:])
    assert result["budget"]["clock_mhz"] is None
    assert result["search_space"]["sha256"] == "deadbeef"
    assert result["engine_info"]["name"] == "compileiq"
    if metric == "spill_bytes":
        assert "best_ms" not in result
        assert result["budget"]["warmup"] is None and result["budget"]["rep"] is None


@pytest.mark.parametrize("visible", ["", "0", "0,0", "0,00", "0,", "-1,0", "all", "0,1,-1"])
def test_parallel_workers_require_explicit_distinct_devices(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        _load_adapter()._worker_gpus(2)


def test_workers_respect_visible_device_order(monkeypatch):
    mod = _load_adapter()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 3, 7, 5 ")
    assert mod._worker_gpus(2) == ["3", "7"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,GPU-def")
    assert mod._worker_gpus(2) == ["GPU-abc", "GPU-def"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert mod._worker_gpus(1) == []
    with pytest.raises(ValueError, match="positive"):
        mod._worker_gpus(0)


@pytest.fixture
def scorer_script(tmp_path):
    """CPU substitute for _score: records device affinity and detects overlapping use of a GPU."""
    script = tmp_path / "score.py"
    script.write_text("""\
import json
import os
from pathlib import Path
import sys
import time

root, mode = Path(sys.argv[1]), sys.argv[2]
acf_path = Path(sys.argv[sys.argv.index("--acf") + 1])
acf = acf_path.read_bytes().hex()
gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
(root / f"input-{acf}.json").write_text(json.dumps({"pid": os.getpid(), "path": str(acf_path), "gpu": gpu}))
if mode == "timeout":
    time.sleep(30)
if mode == "invalid":
    print("INVALID")
    sys.exit(0)
if mode == "crash":
    print("MS 1.0")
    sys.exit(2)
if mode == "parallel":
    lease = root / f"busy-{gpu}"
    # Exclusive creation fails if two scorers receive the same GPU concurrently.
    with lease.open("x"):
        (root / f"ready-{gpu}").touch()
        deadline = time.monotonic() + 10
        while len(list(root.glob("ready-*"))) < 2:
            if time.monotonic() > deadline:
                raise RuntimeError("scorers did not run on two GPUs concurrently")
            time.sleep(0.01)
        time.sleep(0.05)
    lease.unlink()
print("MS 1.0")
""")
    return [sys.executable, str(script), str(tmp_path)]


def _score_in_process(scorer, acf, barrier):
    """Transfer the actual objective and its manager proxies through spawn/forkserver serialization."""
    barrier.wait(timeout=20)
    return scorer(acf)


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_gpu_leases_and_counts_cross_processes(monkeypatch, scorer_script, tmp_path, start_method):
    mod = importlib.import_module(ADAPTER)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    with mod._scoring_state(["3", "7"]) as state, multiprocessing.Manager() as manager:
        barrier = manager.Barrier(4)
        scorer = mod._CandidateScorer([*scorer_script, "parallel"], 15, *state, float("inf"))
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context(start_method)) as pool:
            futures = [pool.submit(_score_in_process, scorer, f"{i:02x}", barrier) for i in range(4)]
            assert [future.result(timeout=30) for future in futures] == [1.0] * 4
        available, counts, _ = state
        assert dict(counts) == {"n": 4, "valid": 4, "timeout": 0}
        assert {available.get_nowait(), available.get_nowait()} == {"3", "7"}
    records = [json.loads(path.read_text()) for path in tmp_path.glob("input-*.json")]
    assert {record["gpu"] for record in records} == {"3", "7"}
    assert all(not Path(record["path"]).exists() for record in records)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,7"


@pytest.mark.parametrize("mode", ["timeout", "invalid", "crash"])
def test_failed_scorer_releases_gpu_and_cleans_acf(scorer_script, tmp_path, mode):
    mod = _load_adapter()
    with mod._scoring_state(["7"]) as state:
        scorer = mod._CandidateScorer([*scorer_script, mode], 1, *state, float("inf"))
        assert scorer("00") == float("inf")
        record = json.loads((tmp_path / "input-00.json").read_text())
        assert not Path(record["path"]).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(record["pid"], 0)  # Child must be dead/reaped before the GPU is reused.
        scorer.command = [*scorer_script, "ok"]
        assert scorer("01") == 1.0
        assert dict(state[1]) == {"n": 2, "valid": 1, "timeout": int(mode == "timeout")}
        assert state[0].get_nowait() == "7"


def test_serial_scorer_preserves_cuda_visibility(monkeypatch, scorer_script, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,3")
    mod = _load_adapter()
    with mod._scoring_state([]) as state:
        assert mod._CandidateScorer([*scorer_script, "ok"], 5, *state, float("inf"))("00") == 1.0
    assert json.loads((tmp_path / "input-00.json").read_text())["gpu"] == "5,3"


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("engine_fails", [False, True])
@pytest.mark.parametrize(
    ("pool_env", "cull_env", "expected_pool", "expected_cull"),
    [(None, None, 8, 4), ("16", "6", 16, 6), ("0", "4", None, 4)],
)
def test_adapter_search_wiring_and_shared_results(
    fake_compileiq, monkeypatch, tmp_path, workers, engine_fails, pool_env, cull_env, expected_pool, expected_cull
):
    mod = _load_adapter()
    monkeypatch.setenv("PTX_ANNEAL_SCORE_CMD", json.dumps([sys.executable, "-c", "print('MS 1.0')"]))
    monkeypatch.setenv("PTX_ANNEAL_TASK", str(tmp_path))
    monkeypatch.setenv("PTX_ANNEAL_PTXAS", "unused")
    monkeypatch.setenv("PTX_ANNEAL_RESULT", str(tmp_path / "result.json"))
    monkeypatch.setenv("PTX_ANNEAL_NUM_WORKERS", str(workers))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setenv("CIQ_TASK_TIMEOUT", "5")
    monkeypatch.setenv("CIQ_GENERATIONS", "3")
    for name, value in (("CIQ_POOL", pool_env), ("CIQ_CULL", cull_env)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.delenv("CIQ_CLOCK_MHZ", raising=False)
    monkeypatch.delenv("PTX_ANNEAL_SS", raising=False)
    calls = {}

    class Search:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def start(self, **kwargs):
            calls["start"] = kwargs
            assert calls["objective_function"]("00") == 1.0
            if engine_fails:
                raise RuntimeError("engine failed after candidate")
            return types.SimpleNamespace(
                get_results=lambda: [1], get_best_result=lambda: {"params": "00", "score_1": 1.0}
            )

    sys.modules["compileiq.ciq"].Search = Search
    engine_types = types.ModuleType("compileiq.types")
    engine_types.SearchConfiguration = lambda **kw: kw
    engine_types.INVALID_SCORE = float("inf")
    engine_types.WorkerTypes = types.SimpleNamespace(NATIVE="native")
    monkeypatch.setitem(sys.modules, "compileiq.types", engine_types)
    assert mod.main() == 0
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["valid"] == 1 and payload["evaluated"] == 1 and payload["timeout"] == 0
    assert bool(payload["engine_error"]) == engine_fails
    assert payload["budget"]["num_workers"] == workers
    assert payload["budget"]["gpus"] == (["3", "7"] if workers == 2 else [])
    assert calls["search_config"] == {
        "problem_type": "min",
        "generations": 3,
        "pool_size": expected_pool,
        "cull_size": expected_cull,
    }
    assert payload["budget"]["pool_size"] == expected_pool
    assert payload["budget"]["cull_size"] == expected_cull
    assert calls["start"] == ({"task_timeout": 10, "num_workers": 2} if workers == 2 else {"task_timeout": 10})
    assert calls.get("worker_type") == ("native" if workers == 2 else None)
