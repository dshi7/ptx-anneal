# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The ``ptx_anneal`` command-line interface -- one binary for the whole ACF factory.

    ptx_anneal --task <task_dir> [--output <store>] [--ss <ss>]
    ptx_anneal --task-list <manifold_dir> [--full-tune] [--upload-manifold]
    ptx_anneal replay --task <task_dir>
    ptx_anneal register_spill --task <kernel.ptx> [--output <store>]
    ptx_anneal doctor

Per-run surface is just the task; the engine, its search space and ptxas are all provisioning, and
each self-provisions by default (pip-installed engine, fetched search-space catalog, discovered
ptxas). ``--ss`` only exists to pin a specific search space.

The harness is engine-free: it assembles/benchmarks candidates with ptxas + the CUDA driver
(``_score``) and admits the winner. The search *engine* (NVIDIA CompileIQ, or an in-house one) is
bring-your-own, driven by a small standalone *adapter* that scores each candidate by calling this
binary's ``_score`` subcommand -- so the harness never imports the engine and can be a frozen binary.

Provisioning (env; flags override):
    PTXAS / TRITON_PTXAS_BLACKWELL_PATH   ptxas to assemble/apply ACFs with
    PTX_ANNEAL_ENGINE_PYTHON              interpreter that has the engine (default: this interpreter,
                                          i.e. install the engine into the same env as ptx_anneal)
    PTX_ANNEAL_ENGINE_ADAPTER             bring-your-own engine adapter (default: bundled CompileIQ)

Version gate: ``ptxas >= MIN_PTXAS`` is required and not overridable -- ``--apply-controls`` (how an
ACF is applied at all) is GA only from that version on, so an older ptxas cannot consume what this
factory produces. A lower ptxas is a hard error, not a fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

from . import fingerprint as fp_mod
from . import provision
from . import task as task_mod
from .ptx_launch import DEFAULT_PTXAS_FLAGS
from .store import LocalStore, default_store_root
from .target import load_target

INVALID = float("inf")

# How each --bench choice actually reduces its samples. Recorded in the fingerprint because "we both
# used do_bench" is not agreement if one took a median of L2-cold single launches and the other the
# mean of a back-to-back graph replay -- those are different questions about the same kernel.
_BENCH_RETURN_MODE = {"cudagraph": "mean(graph-replay)", "do_bench": "median(L2-cold)"}


def _fingerprint(args, t, ptxas: str, ver: str | None, res: dict) -> dict:
    """The search problem this run actually posed (see :mod:`ptx_anneal.fingerprint`).

    Built AFTER the engine returns, never before: the search space resolves to ``latest`` by default,
    so only the post-run metadata says which catalog was really used -- and the requested one is
    exactly the thing that must not be recorded in its place.
    """
    spec = t.spec
    tensors = spec.get("tensors") or []
    return fp_mod.build(
        channel="ptx-direct",
        kernel={
            "name": spec.get("kernel_name") or spec.get("entry"),
            "entry": spec.get("entry"),
            "arch": spec.get("arch"),
            "shapes": [tt.get("shape") for tt in tensors],
            "dtypes": [tt.get("dtype") for tt in tensors],
            # A task is a frozen PTX, so "the pinned config" is the launch geometry baked into it --
            # there is no autotuner left to be ambiguous about.
            "pinned_config": {"grid": spec.get("grid"), "block": spec.get("block"), "shared": spec.get("shared")},
            "ir_hash": t.ir_hash,
        },
        # No frontend at tune time: this channel replays a frozen kernel.ptx, so whichever Triton
        # produced it did its work at COLLECT time and is not an input here. Recorded as empty rather
        # than omitted, so a diff against the triton-native channel shows the difference explicitly.
        frontend={},
        ptxas={"path": ptxas, "version": ver, "flags": [*DEFAULT_PTXAS_FLAGS, *fp_mod.ptxas_flags()]},
        search_space=res.get("search_space") or {},
        engine={**(res.get("engine_info") or {}), "budget": res.get("budget") or {}},
        objective={
            "metric": "min_ms",
            "timing": {
                "method": args.bench,
                "warmup": args.warmup,
                "rep": args.rep,
                "return_mode": _BENCH_RETURN_MODE.get(args.bench, args.bench),
            },
            # The PTX-direct channel has no independent reference: an ACF cannot change the math, so
            # a candidate is checked against the NO-ACF BASELINE's own output. That is the weaker,
            # self-referential oracle -- named as such here so a diff against a channel using a real
            # reference shows the difference instead of hiding it.
            "correctness": {
                "oracle": "baseline-output (self-referential)",
                "strength": "weaker",
                "validator": os.environ.get("PTX_ANNEAL_VALIDATOR") or "RelTolValidator",
                "rel_tol": float(os.environ.get("PTX_ANNEAL_REL_TOL", "1e-2")),
            },
        },
    )


def _resolve_ptxas(args) -> str | None:
    # An explicit flag/env is honoured verbatim; everything else goes through the shared discovery
    # path (env -> PATH -> pip-installed nvidia-cuda-nvcc wheel) so `doctor` and `tune` can never
    # disagree about which ptxas this box would use.
    return getattr(args, "ptxas", None) or os.environ.get("PTXAS") or provision.find_ptxas()


def _harness_cmd() -> list[str]:
    """How the adapter re-invokes this binary's ``_score``: the frozen executable, or ``python -m``."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "ptx_anneal.cli"]


def _bundled_adapter() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "adapters", "compileiq_adapter.py")


# -- _score: the per-candidate scorer (engine-free; selects a scorer from the task spec) ----------
def _cmd_score(args) -> int:
    if args.ptxas:
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = args.ptxas
    acf = None if args.acf in (None, "NONE") else args.acf
    try:
        if args.metric == "spill_bytes":
            print(f"SPILL_BYTES {_score_spills(args.task, acf, _resolve_ptxas(args))}")
        else:
            from . import scoring

            ms = scoring.score(args.task, acf, args.warmup, args.rep, args.bench, ptxas=_resolve_ptxas(args))
            print(f"MS {float(ms)}")
    except Exception as e:  # noqa: BLE001 - any failure is INVALID to the caller
        sys.stderr.write(f"[_score] {type(e).__name__}: {e}\n")
        print("INVALID")
    return 0


def _score_spills(task: str, acf: str | None, ptxas: str) -> int:
    from .ptx_launch import ptxas_spill_bytes

    with open(task) as f:
        return ptxas_spill_bytes(f.read(), ptxas, acf_path=acf)


def _score_one(
    task_dir: str, acf: str | None, warmup: int, rep: int, bench: str = "cudagraph", ptxas: str | None = None
) -> float:
    from . import scoring

    try:
        return float(scoring.score(task_dir, acf, warmup, rep, bench, ptxas=ptxas))
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[baseline] {type(e).__name__}: {e}\n")
        return INVALID


def _require_ptxas(args) -> tuple[str, str]:
    """Resolve and check the one toolchain used for tuning and standalone replay."""
    ptxas = _resolve_ptxas(args)
    if not ptxas:
        raise SystemExit("ptx_anneal: no ptxas (set --ptxas / PTXAS / TRITON_PTXAS_BLACKWELL_PATH, or add to PATH)")
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas

    # Hard version gate. An ACF is applied with `ptxas --apply-controls`, GA only from MIN_PTXAS on,
    # so an older ptxas can neither assemble nor consume one -- there is no fallback to offer.
    ver = provision.ptxas_version(ptxas)
    need = provision.min_ptxas()
    if not ver or provision._version_tuple(ver) < provision._version_tuple(need):
        raise SystemExit(
            f"ptx_anneal: ptxas {ver or '?'} < {need}. Applying an ACF (`ptxas --apply-controls`) is "
            f"GA from {need} on; point PTXAS / TRITON_PTXAS_BLACKWELL_PATH at a {need}+ ptxas."
        )
    return ptxas, ver


def _cmd_replay(args) -> int:
    from .replay import replay

    try:
        t = task_mod.load(args.task)
        ptxas, ver = _require_ptxas(args)
        print(f"replay: {t.entry} ; arch={t.arch} ; ptxas={ptxas} (V{ver}) ; inputs=synthetic", flush=True)
        ms = replay(t, ptxas, warmup=args.warmup, rep=args.rep, bench=args.bench)
        if not math.isfinite(ms) or ms <= 0:
            raise RuntimeError(f"invalid replay timing: {ms}")
        print(f"MS {ms}")
        return 0
    except Exception as e:  # noqa: BLE001 - a standalone replay must report failure with a nonzero exit
        sys.stderr.write(f"[replay] {type(e).__name__}: {e}\n")
        return 1


def _engine_command(args) -> tuple[str, str]:
    # Override only when the engine lives in a separate interpreter.
    engine_python = args.engine_python or os.environ.get("PTX_ANNEAL_ENGINE_PYTHON") or sys.executable
    # The engine stays bring-your-own: the bundled CompileIQ adapter is only the default.
    adapter = args.engine_adapter or os.environ.get("PTX_ANNEAL_ENGINE_ADAPTER") or _bundled_adapter()
    if not os.path.exists(adapter):
        raise SystemExit(f"ptx_anneal: engine adapter not found: {adapter!r}")
    return engine_python, adapter


def _run_engine(
    args, ptxas: str, engine_python: str, adapter: str, *, metric: str = "ms", extra_env: dict | None = None
) -> dict:
    """Drive the subprocess adapter with either runtime or assembly-only scoring."""
    result_fd, result_path = tempfile.mkstemp(suffix=".json")
    os.close(result_fd)
    env = dict(os.environ)
    score_args = ["--bench", args.bench] if metric == "ms" else ["--metric", metric]
    env["PTX_ANNEAL_SCORE_CMD"] = json.dumps([*_harness_cmd(), "_score", *score_args])
    env["PTX_ANNEAL_METRIC"] = metric
    env["PTX_ANNEAL_TASK"] = os.path.abspath(args.task)
    env["PTX_ANNEAL_PTXAS"] = ptxas
    env["PTX_ANNEAL_RESULT"] = result_path
    env["PTX_ANNEAL_WARMUP"] = str(getattr(args, "warmup", 0))
    env["PTX_ANNEAL_REP"] = str(getattr(args, "rep", 0))
    env.update(extra_env or {})
    if args.ss:
        env["PTX_ANNEAL_SS"] = os.path.abspath(args.ss)
    try:
        proc = subprocess.run([engine_python, adapter], env=env, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"ptx_anneal: engine adapter exited {proc.returncode}")
        with open(result_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)


# -- tune: baseline -> spawn engine adapter -> admit ----------------------------------------------
def _cmd_tune(args) -> int:
    # Single-task invocations reach here; --task-list delegates to fresh --task subprocesses.
    ptxas, ver = _require_ptxas(args)
    gpus = None
    workers = args.workers
    if args.full_tune:
        try:
            gpus = provision.visible_gpus()
        except RuntimeError as e:
            raise SystemExit(f"ptx_anneal: --full-tune: {e}") from e
        if not gpus:
            raise SystemExit("ptx_anneal: --full-tune: no CUDA-visible GPUs available")
        workers = len(gpus) if workers is None else workers
        if workers > len(gpus):
            raise SystemExit(f"ptx_anneal: --workers {workers} exceeds the {len(gpus)} CUDA-visible GPUs")
        print(f"full tune: detected GPUs={','.join(gpus)} ; workers={workers}", flush=True)
    extra_env = {}
    if workers is not None:
        extra_env["PTX_ANNEAL_NUM_WORKERS"] = str(workers)
    if args.full_tune:
        extra_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        extra_env.update(CIQ_GENERATIONS="20", CIQ_POOL="32", CIQ_CULL="24")
    engine_python, adapter = _engine_command(args)
    t = task_mod.load(args.task)
    target = load_target(args.target)
    arch, ir_hash = target.arch(t.ir), target.ir_hash(t.ir)

    print(
        f"ptxas={ptxas} (V{ver or '?'}) ; engine-python={engine_python} ; "
        f"adapter={os.path.basename(adapter)} ; task={args.task}"
    )

    # 1) baseline (no ACF), scored in-process by the harness.
    baseline_ms = _score_one(args.task, None, args.warmup, args.rep, args.bench, ptxas)
    baseline_ok = math.isfinite(baseline_ms)
    print(f"baseline: {f'{baseline_ms:.4f} ms' if baseline_ok else 'FAILED'} (bench={args.bench})")

    # 2) run the engine in its own interpreter via the adapter; it scores candidates through `_score`.
    res = _run_engine(args, ptxas, engine_python, adapter, extra_env=extra_env)
    evaluated = int(res.get("evaluated") or 0)
    valid = res.get("valid")
    # Surfaced, not logged: timed-out candidates are search scope that was dropped.
    timed_out = int(res.get("timeout") or 0)
    timeout_tag = f" timeout={timed_out}" if timed_out else ""
    best_ms = res.get("best_ms")
    acf_hex = res.get("acf")
    engine_name = os.path.basename(adapter).removesuffix(".py").removesuffix("_adapter")
    search_best_ms = best_ms
    budget = res.get("budget") or {}
    if (
        budget.get("num_workers", 1) > 1
        and acf_hex
        and isinstance(best_ms, (int, float))
        and math.isfinite(best_ms)
        and baseline_ok
    ):
        # Search timings may come from another GPU. Admission must compare the winner on the
        # original baseline device, after all parallel scorers have exited.
        with tempfile.TemporaryDirectory(prefix="ptx_anneal_winner_") as tmp:
            acf_path = os.path.join(tmp, "winner.acf")
            with open(acf_path, "wb") as f:
                f.write(bytes.fromhex(acf_hex))
            best_ms = _score_one(args.task, acf_path, args.warmup, args.rep, args.bench, ptxas)
        print(f"winner on baseline GPU: {best_ms:.4f} ms (search={search_best_ms:.4f} ms)")

    # 3) admit. Normally only a finite candidate that BEAT the baseline is admitted. PTX_ANNEAL_FORCE_ADMIT
    #    (default off) admits the best *valid* candidate even without a win -- this is a SMOKE knob: it
    #    guarantees a real (different-from-baseline) ACF lands in the store so the consume/apply path is
    #    always exercised end-to-end, even on kernels/shapes with no reproducible ptxas headroom.
    force_admit = os.environ.get("PTX_ANNEAL_FORCE_ADMIT", "") not in ("", "0")
    best_finite = bool(acf_hex) and isinstance(best_ms, (int, float)) and math.isfinite(best_ms)
    is_win = best_finite and baseline_ok and best_ms < baseline_ms
    store_path = None
    if best_finite and baseline_ok and (is_win or force_admit):
        win = (baseline_ms - best_ms) / baseline_ms
        meta = {
            "target": target.name,
            "arch": arch,
            "ir_hash": ir_hash,
            "entry": t.spec.get("entry"),
            "baseline_ms": baseline_ms,
            "best_ms": best_ms,
            "search_best_ms": search_best_ms,
            "budget": budget,
            "search_win": win,
            "forced": not is_win,
            "evaluated": evaluated,
            "engine": engine_name,
            "valid": valid,
            "ptxas_version": ver,
            "ptxas_flags": list(DEFAULT_PTXAS_FLAGS),
            # Which search space produced this ACF. The search space is a tuning *input*, and the
            # engine resolves it to "latest" by default, so recording it is what keeps an admitted
            # ACF explainable (and a pinned CIQ_SS_TAG meaningful) after the catalog moves on.
            "search_space": res.get("search_space") or None,
            # Which engine build produced it. `engine` above is just the adapter's name; an engine is
            # an installed library with no path to print, so the adapter reports its own identity.
            "engine_info": res.get("engine_info") or None,
        }
        # Group artifacts under <arch>/<ir_hash[:16]>/, retaining version-tagged filenames. An ACF is
        # bound to its ptxas version, so different toolchains must coexist without overwriting one
        # another; the kernel identity stays version-free. Direct consumers must use this layout.
        store_path = LocalStore(args.output or default_store_root()).write(
            target.name,
            arch,
            ir_hash,
            bytes.fromhex(acf_hex),
            meta,
            toolchain_version=ver or "",
        )
        forced_tag = "" if is_win else " [FORCE_ADMIT: no genuine win; admitted to exercise consume]"
        print(
            f"admitted ACF for {target.name}/{arch}/{ir_hash[:16]} "
            f"(baseline={baseline_ms:.4f}ms best={best_ms:.4f}ms search-win={win * 100:+.2f}%){forced_tag} "
            f"engine={engine_name} evaluated={evaluated} valid={valid}{timeout_tag} -> {store_path}"
        )
        print("note: the search-time win is noisy; the trustworthy decision is the consumer's A/B vs baseline.")
    else:
        reason = (
            "no baseline"
            if not baseline_ok
            else "no valid candidate"
            if not best_finite
            else "no candidate beat the baseline (set PTX_ANNEAL_FORCE_ADMIT=1 to admit best anyway)"
        )
        print(
            f"no candidate admitted ({reason}); engine={engine_name} evaluated={evaluated} valid={valid}{timeout_tag}"
        )

    # Always, including on a no-admission run: the fingerprint describes the *problem*, and a
    # channel that found nothing is exactly when you need to know whether it was even asked the same
    # question as the channel that did.
    print(fp_mod.render(_fingerprint(args, t, ptxas, ver, res)))
    if args.upload_manifold:
        if store_path is None:
            print("Manifold upload skipped: no ACF admitted.")
        else:
            try:
                uploaded = task_mod.upload_store(args.task_source, args.output or default_store_root(), store_path)
            except task_mod.TaskError as e:
                sys.stderr.write(f"[upload-manifold] {e}\nLocal ACF retained at {store_path}\n")
                return 1
            print(f"uploaded task store -> {uploaded}")
    return 0


def _cmd_tune_list(args) -> int:
    print(f"Discovering tasks under {args.task_list} ...", flush=True)
    try:
        sources = task_mod.manifold_tasks(args.task_list)
        if args.upload_manifold:
            # Validate every upload destination before starting any search.
            for source in sources:
                task_mod.manifold_store_url(source)
    except task_mod.TaskError as e:
        sys.stderr.write(f"[task-list] {e}\n")
        return 1

    # The tuning parser's destinations match its long option names (underscores become hyphens).
    # Forward every tuning option to each child, including pins and the shared local store root.
    options = []
    for name, value in vars(args).items():
        if name in ("task", "task_list") or value is None or value is False:
            continue
        flag = "--" + name.replace("_", "-")
        options.extend([flag] if value is True else [f"{flag}={value}"])

    failed = 0
    print(f"Found {len(sources)} tasks; tuning sequentially.", flush=True)
    for index, source in enumerate(sources, 1):
        print(f"\n[task-list {index}/{len(sources)}] {source}", flush=True)
        # A fresh process releases CUDA contexts and tensor caches between tasks. Within each
        # search, --full-tune / --workers still control parallel candidate scoring across GPUs.
        try:
            result = subprocess.run([*_harness_cmd(), "--task", source, *options], check=False)
            code = result.returncode
        except OSError as e:
            sys.stderr.write(f"[task-list] {source}: {e}\n")
            code = 1
        if code != 0:
            failed += 1
            sys.stderr.write(f"[task-list] FAILED (exit {code}): {source}\n")
    print(f"Task-list complete: {len(sources) - failed} completed, {failed} failed, {len(sources)} total.", flush=True)
    return 1 if failed else 0


def _cmd_register_spill(args) -> int:
    from .ptx_launch import ptxas_spill_bytes

    try:
        with open(args.task) as f:
            ptx = f.read()
        target = load_target("ptx")
        arch, ir_hash = target.arch(ptx), target.ir_hash(ptx)
        ptxas, ver = _require_ptxas(args)
        engine_python, adapter = _engine_command(args)
        print(
            f"register_spill: task={args.task} ; arch={arch} ; ptxas={ptxas} (V{ver}) ; "
            f"engine-python={engine_python} ; adapter={os.path.basename(adapter)}",
            flush=True,
        )
        baseline = ptxas_spill_bytes(ptx, ptxas)
        print(f"baseline: {baseline} spill bytes (stores + loads)", flush=True)
        res = _run_engine(args, ptxas, engine_python, adapter, metric="spill_bytes")
        if res.get("engine_error"):
            raise RuntimeError(f"engine search failed: {res['engine_error']}")
        best = res.get("best_spill_bytes")
        acf_hex = res.get("acf")
        valid = (
            bool(acf_hex) and isinstance(best, (int, float)) and math.isfinite(best) and best >= 0 and best == int(best)
        )
        counts = f"evaluated={res.get('evaluated', 0)} valid={res.get('valid')} timeout={res.get('timeout', 0)}"
        if valid and best < baseline:
            # This objective has no runtime correctness or performance check. Its default store
            # must not replace the validated runtime winner for the same PTX and toolchain.
            root = args.output or os.path.join(default_store_root(), "register_spill")
            meta = {
                "objective": "spill_bytes",
                "correctness": "not checked (compile-only)",
                "baseline_spill_bytes": baseline,
                "best_spill_bytes": int(best),
                "evaluated": res.get("evaluated", 0),
                "valid": res.get("valid"),
                "ptxas_version": ver,
                "ptxas_flags": [f"-arch={arch}", *DEFAULT_PTXAS_FLAGS, "-v"],
                "search_space": res.get("search_space") or None,
                "engine_info": res.get("engine_info") or None,
                "budget": res.get("budget") or {},
            }
            path = LocalStore(root).write("ptx", arch, ir_hash, bytes.fromhex(acf_hex), meta, toolchain_version=ver)
            print(f"admitted ACF (baseline={baseline} best={int(best)} spill bytes) {counts} -> {path}")
        else:
            reason = "no valid candidate" if not valid else "no candidate reduced spill bytes"
            print(f"no candidate admitted ({reason}); {counts}")
        return 0
    except Exception as e:  # noqa: BLE001 - report compile/search/file failures without a traceback
        sys.stderr.write(f"[register_spill] {type(e).__name__}: {e}\n")
        return 1


def _cmd_doctor() -> int:
    print(provision.format_report(provision.report()))
    return 0


# Subcommands are intercepted in main() before the parser sees them, so argparse cannot
# advertise them on its own -- spell the real surface out here. `_score` stays undocumented: it is
# the internal per-candidate scorer the adapter calls back into, not a user command.
_USAGE = (
    "ptx_anneal --task <dir> [--output <store>] [options]\n"
    "       ptx_anneal --task-list <manifold_dir> [--output <store>] [options]\n"
    "       ptx_anneal replay --task <dir> [options]\n"
    "       ptx_anneal register_spill --task <kernel.ptx> [options]\n"
    "       ptx_anneal doctor"
)

_EPILOG = """\
environment settings:
  PTX_ANNEAL_FORCE_ADMIT   admit the best valid candidate even when it did not beat the baseline
                           (smoke knob: guarantees a real ACF lands so consume can be exercised)
  CIQ_POOL / CIQ_CULL / CIQ_GENERATIONS   engine budget (default 8 / 4 / 1 -- one batch, no evolution)
  CIQ_SS_TAG / CIQ_SS_VERSION / CIQ_SS_VARIANT   pin the published search-space catalog
  CIQ_SEARCH_SPACES_DIR    offline search-space mirror (no network)

notes:
  The exit code does NOT report admission -- 0 means the run completed, not that an ACF was
  admitted ("no headroom" is an outcome, not an error). To gate on it, match "admitted ACF"
  in the output.

  The ptxas >= 13.3 floor is fixed, not a default: `--apply-controls` is GA only from there
  on, so an older ptxas can neither produce nor consume an ACF. A pinned --ptxas is used
  verbatim even when it is too old -- the whole toolchain must agree on one ptxas.
"""


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptx_anneal",
        usage=_USAGE,
        description="Offline ptxas ACF tuning. The engine, its search space and ptxas all "
        "self-provision, so `--task <dir>` on its own is a complete invocation.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("task")
    source = g.add_mutually_exclusive_group(required=True)
    source.add_argument("--task", help="local task directory or manifold:// task directory URL")
    source.add_argument(
        "--task-list",
        help="Manifold directory URL; recursively discover and tune all tasks sequentially, continuing after failures",
    )
    g.add_argument(
        "--output",
        "--store",
        dest="output",
        help="ACF store root (default: $COMPILE_IQ_STORE or ~/.compile_iq/store)",
    )
    g.add_argument(
        "--upload-manifold",
        action="store_true",
        help="upload the task's <arch>/<hash16>/ store subtree to manifold://<bucket>/<run>/store; "
        "requires each task URL to have the form manifold://<bucket>/<run>/tasks/<task-id>",
    )

    g = p.add_argument_group("search")
    g.add_argument(
        "--full-tune",
        action="store_true",
        help="set 20 generations, pool size 32, and cull size 24, and use all CUDA-visible GPUs "
        "unless --workers is set; overrides CIQ_GENERATIONS, CIQ_POOL, CIQ_CULL, and PTX_ANNEAL_NUM_WORKERS",
    )

    g = p.add_argument_group("scoring")
    g.add_argument(
        "--bench",
        choices=["cudagraph", "do_bench"],
        default="cudagraph",
        help="scoring metric: cudagraph (fast/deterministic, default) or do_bench (L2-flush median, "
        "faithful to the consumer's A/B)",
    )
    g.add_argument("--warmup", type=int, default=25, help="benchmark warmup iters (default: %(default)s)")
    g.add_argument("--rep", type=int, default=50, help="benchmark measured iters (default: %(default)s)")
    g.add_argument(
        "--workers",
        type=_positive_int,
        help="CompileIQ native workers (default: all visible GPUs with --full-tune, otherwise "
        "$PTX_ANNEAL_NUM_WORKERS or 1); uses the first N GPUs, one candidate per GPU; "
        "requires CUDA_VISIBLE_DEVICES unless --full-tune probes the device list",
    )
    g.add_argument("--target", default="ptx", help="target (default: %(default)s)")

    _add_provisioning_pins(p)
    return p


def _add_provisioning_pins(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("provisioning pins", "all optional -- each of these self-provisions")
    g.add_argument(
        "--ptxas",
        help="ptxas to use (else $PTXAS / $TRITON_PTXAS_BLACKWELL_PATH / PATH / an installed nvidia-cuda-nvcc wheel)",
    )
    g.add_argument("--ss", help="engine search-space path (else the engine fetches its published catalog)")
    g.add_argument(
        "--engine-python",
        dest="engine_python",
        help="interpreter that has the engine (default: this interpreter / $PTX_ANNEAL_ENGINE_PYTHON)",
    )
    g.add_argument(
        "--engine-adapter",
        dest="engine_adapter",
        help="bring-your-own engine adapter script (default: bundled CompileIQ / $PTX_ANNEAL_ENGINE_ADAPTER)",
    )


def _register_spill_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptx_anneal register_spill",
        description="Tune CompileIQ controls to minimize ptxas spill-store + spill-load bytes across all "
        "functions in a PTX file. Compile-only: no GPU, launch spec, or runtime validation required.",
        epilog="CIQ_POOL / CIQ_GENERATIONS set the search budget. Only candidates with fewer spill bytes "
        "than the baseline are saved; zero spill bytes is a valid score.",
    )
    p.add_argument("--task", required=True, help="local PTX file (not a task directory)")
    p.add_argument(
        "--output",
        "--store",
        dest="output",
        help="ACF store root (default: register_spill/ under the default ACF store)",
    )
    _add_provisioning_pins(p)
    return p


def _replay_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptx_anneal replay",
        description="Assemble and benchmark captured PTX without an engine or ACF. Uses synthetic inputs; "
        "does not validate against original outputs. Requires the harness and cudapy extras.",
    )
    p.add_argument("--task", required=True, help="local task directory or manifold:// task directory URL")
    p.add_argument("--ptxas", help="ptxas >= 13.3 (otherwise use environment / PATH / installed nvcc wheel)")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=50)
    p.add_argument("--bench", choices=["cudagraph", "do_bench"], default="cudagraph")
    return p


def _score_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ptx_anneal _score", description="Score one candidate (internal).")
    p.add_argument("--task", required=True)
    p.add_argument("--acf", default="NONE", help="ACF file path, or NONE for the untuned baseline")
    p.add_argument("--ptxas")
    p.add_argument("--metric", choices=["ms", "spill_bytes"], default="ms")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=50)
    p.add_argument("--bench", choices=["cudagraph", "do_bench"], default="cudagraph")
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "doctor":
        parser = argparse.ArgumentParser(prog="ptx_anneal doctor", description="Inspect the tuning toolchain.")
        parser.parse_args(argv[1:])
        return _cmd_doctor()
    if argv and argv[0] == "_score":
        return _cmd_score(_score_parser().parse_args(argv[1:]))
    if argv and argv[0] == "register_spill":
        return _cmd_register_spill(_register_spill_parser().parse_args(argv[1:]))
    if argv and argv[0] == "replay":
        args, command = _replay_parser().parse_args(argv[1:]), _cmd_replay
    else:
        parser = build_parser()
        args, command = parser.parse_args(argv), _cmd_tune
        if args.task_list is not None:
            if not args.task_list.startswith("manifold://"):
                parser.error("--task-list requires a manifold:// directory URL")
            return _cmd_tune_list(args)
        if args.upload_manifold:
            if not args.task.startswith("manifold://"):
                parser.error("--upload-manifold requires --task manifold://<bucket>/<run>/tasks/<task-id>")
            try:
                task_mod.manifold_store_url(args.task)
            except task_mod.TaskError as e:
                parser.error(str(e))
    args.task_source = args.task
    try:
        # Fetch once per invocation. The engine and its scorers share the same local files until
        # the whole run completes; cleanup also happens on failure or interruption.
        with task_mod.local_task_dir(args.task) as local_dir:
            args.task = local_dir
            return command(args)
    except task_mod.TaskError as e:
        sys.stderr.write(f"[task] {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
