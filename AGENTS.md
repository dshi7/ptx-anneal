# AGENTS.md — guide for coding agents working on ptx-anneal

This file orients automated coding agents. Humans: see `README.md`.

## What this repo is

`ptx-anneal` is an offline **ptxas tuning framework** (the "ACF factory") built on NVIDIA CompileIQ. 
It searches `ptxas` advanced controls for a compiled GPU kernel, validates the
result is numerically correct against the untuned baseline, and caches the
winning tuning artifact (an **ACF**, Advanced Control File) in a
content-addressed store. It is the orchestration **harness only** — the
optimization engine is bring-your-own (see below).

PTX in, ACF out

## Hard rules

- **Must run from a plain git checkout.** `ptx_anneal` (the ACF factory) and the
  `MODE=tune` validation are what we share with the NVIDIA CompileIQ team, so they
  must work with nothing but Python, `ptxas` and the engine — no build system, no
  fbcode paths, no `fb/`. This is the constraint most likely to be broken by a
  well-meaning change; check it before adding any dependency. `e2e.sh` therefore
  pins `PYTHONPATH` to its own checkout: it runs from `e2e/`, so an editable
  `ptx_anneal` install elsewhere (an fbcode tree) would otherwise be imported and
  the harness would silently validate the wrong tree.
- **Provisioning self-serves; nothing is required by hand.** The engine is
  `pip install ptx-anneal[compileiq]`, its search space is fetched by the engine, and `ptxas`
  is discovered (env → `PATH` → `nvidia-cuda-nvcc` wheel). Keep `--ss`, `--ptxas`
  and the `CIQ_SS_*` env vars as *pins*, not prerequisites — a bare
  `ptx_anneal --task <dir>` must keep working.
- **Pure Python, no build system beyond `hatchling`,** under `ptx_anneal/` and
  `e2e/*.sh`. Anything needing Buck lives in `fb/` (site scaffolding, stripped) —
  today the `MODE=full` frontend orchestration. `e2e/BUCK` is the one Buck file
  outside `fb/`: it only *defines* the frontend target from `e2e/` sources and is
  inert in a git checkout.
- **Never vendor the engine — install it.** CompileIQ is on PyPI
  (`pip install compileiq`, or `pip install ptx-anneal[compileiq]`). It must never
  be committed to or downloaded into the tree, and nothing here may `import
  compileiq`: the harness reaches the engine only by spawning an adapter script
  (`ptx_anneal/adapters/`) in the engine's own interpreter. That subprocess seam —
  not the dependency list — is what keeps the harness engine-free and freezable.
  Keeping compileiq out of the base `dependencies` is a **support-matrix**
  decision, not a licensing one: compileiq is `>=3.11,<3.14` and needs glibc
  >= 2.34, while the store/ABC/`doctor` layer is pure Python, needs no GPU, and
  should stay installable anywhere (CI covers 3.10–3.13, and the `bare-install` job
  enforces the no-extras install). Promoting it to a base dependency therefore means
  bumping `requires-python` and dropping the 3.10 CI leg — a deliberate call, not a
  drive-by.
- **No internal-only code in the public tree.** Inhouse APIs/services live either
  out-of-tree (registered via `entry_points`) or in the stripped `fb/`. Nothing
  outside `fb/` may import or require them; `fb/` is optional by construction.
- **Every source file carries the MIT license header.**
- **Leave changes uncommitted** unless explicitly asked to commit.

## Layout

- `ptx_anneal/target/` — `Target` ABC + the `ptx` target (the only v1 target;
  other targets such as AMD `gcn` are future registered plugins).
- `ptx_anneal/store/` — `Store` ABC + `LocalStore` (filesystem). Keys are
  `(target, arch, ir_hash)`.
- `ptx_anneal/run/` — `Runner` ABC + `LocalRunner` (single host) and `CudaPyRunner`
  (torch-free: cuda-python + numpy worker). The CLI uses `scoring.py` to choose this worker
  or the torch-based `PtxTarget` scorer from the task spec — see "Two tuning paths".
- `ptx_anneal/scoring.py` — shared baseline/candidate dispatch for the CLI. Contiguous supported
  NumPy dtypes use `cudapy_worker`; BF16, TMA and strided tensors use `PtxTarget.score_one`.
  Pass the resolved ptxas pin through both paths so a captured spec cannot override it.
- `ptx_anneal/search/` — `SearchBackend` ABC + the entry-point registry.
- `ptx_anneal/log/` — `Logger` ABC + `NullLogger` (default) / `StderrLogger`.
- `ptx_anneal/factory.py` — the library orchestrator (search → validate → admit →
  store). Library API only — the CLI does not call it; see "Two tuning paths".
- `ptx_anneal/task.py` — the task (`kernel.ptx` + `spec.json`) loader. `local_task_dir` also
  materializes `manifold://` directory URLs via the optional external `manifold` CLI;
  its context must outlive the whole replay/search, including scoring subprocesses.
  Tuning's `--upload-manifold` uploads the admitted ACF and its provenance sidecar under the
  original remote task's `compileiq_YYYYmmddHHMMSS/` directory (UTC); preserve the original URL
  when replacing `args.task` with a temporary local path.
- `ptx_anneal/validate.py` — `Validator` ABC + `RelTolValidator`.
- `ptx_anneal/ptx_launch.py` — the PTX-direct driver-launch benchmarking harness.
- `ptx_anneal/adapters/` — engine adapters (standalone scripts run in the engine's
  own interpreter); `compileiq_adapter.py` is the bundled default.
- `ptx_anneal/provision.py` / `cli.py` — toolchain detection (`ptx-anneal doctor`)
  and the CLI. Tuning is the *default* action (`--task`); `doctor` and the internal
  `_score`, plus `replay` and `register_spill`, are the subcommands — there is no `tune` subcommand.
  `register_spill --task <file.ptx>` compiles without launching and minimizes spill-store + spill-load
  bytes reported by `ptxas -v`. It uses the same adapter with `PTX_ANNEAL_METRIC=spill_bytes` and saves
  improvements under a separate `register_spill/` store by default; these have no runtime validation.
- `ptx_anneal/replay.py` — standalone `replay --task`: assemble and benchmark PTX without an
  engine or ACF. Uses lazy torch allocation and Triton TMA encoding with the CUDA driver;
  supports BF16, strides and CTA clusters. Inputs are synthetic; this is not a correctness check.
- `ptx_anneal/fingerprint.py` — the search-problem fingerprint both channels emit (stdlib only).
- `e2e/` — the adhoc validation harness (`e2e.sh`, `kernels/`); never packaged.
  `MODE=tune` needs only the factory's deps; `kernels/` + `run.py` need torch+triton.
- `e2e/triton_native/` — the optional **triton-native channel** (`MODE=native`): scores a registered
  kernel through **fbtriton** under `PTXAS_OPTIONS=--apply-controls`, instead of relaunching captured
  PTX. The frontend is fbtriton specifically — `PTXAS_OPTIONS` is its knob (upstream Triton has none),
  and on the wrong Triton the ACF is ignored while every candidate still scores, so the scorer
  *verifies* `CUDAOptions.ptx_options` before measuring. `runway.py` is pure stdlib and *probes* for
  torch/fbtriton/engine so it degrades cleanly; `score.py` is the only module that imports them, and
  only after the ACF env is set — `PTXAS_OPTIONS` is read when the nvidia backend is imported, so it
  is necessarily one candidate per process. Attribution, not speed: see `e2e/triton_native/README.md`.
- `fb/` — site scaffolding, ShipIt-stripped. Today: `e2e_internal.sh`, the buck-driven
  `MODE=full` frontend. Optional by construction. **`fb/` lives in fbsource only** — it
  must not exist in a git checkout at all, even locally and even though `.gitignore`
  would stop it being committed. A stale local copy is worse than none: it reads as
  live internal wiring to anyone grepping the tree. Edit it in
  `fbcode/triton/tools/ptx_anneal/fb/`, which is the one real copy.
- `sample_tasks/` — pre-captured `kernel.ptx` + `spec.json` fixtures (torch-free data).
- `tests/`, `skills/`.

## One toolchain

`ptxas >= 13.3` + CompileIQ, everywhere. The floor is fixed (`provision.MIN_PTXAS`),
not a default to relax: `--apply-controls` is GA only from 13.3, and the frontend
derives its emitted PTX ISA version from *its* ptxas version — so mixing versions
across collect/factory/consume changes the PTX, changes the store key, and silently
MISSes. Do not add a version-override knob or a fallback path for an older ptxas.
An explicitly pinned `ptxas` (`--ptxas` / `$PTXAS` / `$TRITON_PTXAS_BLACKWELL_PATH`)
is used verbatim even when it is below the floor — substituting a newer one would
trade a loud version error for exactly that silent MISS.

**Tuning inputs are recorded, not assumed.** An ACF is only explainable if you know what
produced it, and the store key `(target, arch, ir_hash)` separates none of these — so the
`.acf.json` sidecar carries all three: `ptxas_version`, `engine_info` (engine name, version,
import path, interpreter) and `search_space` (`resolved_tag`, `sha256`, `variant`, `source`).
Do not drop any of them. The search space matters most because it resolves to `latest` by
default: two ACFs for the same kernel, tuned from different catalogs, are otherwise
indistinguishable on disk. The engine is an installed library with no path to print, so the
adapter — not the harness — must report its own identity.

## Two tuning paths (know which one you are editing)

There are two orchestration paths; they share the torch-based PTX scorer where applicable:

1. **`ptx_anneal --task` (the CLI, what ships).** Tuning is the default action —
   there is no `tune` subcommand (`replay` benchmarks without tuning; `register_spill` tunes compiler
   spill statistics from a PTX file; `doctor` and `_score` also exist).
   `cli._cmd_tune` scores the baseline via `scoring.score`, spawns the engine adapter as a
   subprocess, and admits through `LocalStore`. The `_score` callback uses the same dispatch:
   `cudapy_worker` for ordinary contiguous tasks, `PtxTarget.score_one` for BF16/TMA/strides.
   Both validate against untuned output. The CLI does **not** call `factory`, `SearchBackend`
   or any `Runner`.
2. **`factory.tune` (the library DI API).** Takes `Store`/`Runner`/`SearchBackend`/
   `Logger` implementations as arguments. Exercised by `tests/` only.

So `factory.tune`, `load_backend`, `LocalRunner`, `CudaPyRunner` and the `Logger`
implementations are currently reachable from tests and embedders, not from the CLI.
`provision.report()` (i.e. `doctor`) inspects the `ptx_anneal.search_backends`
registry that path (2) uses, which is why it can report "no engine" on a setup where
path (1) tunes fine. Do not "fix" a doc or a report to claim the CLI uses the ABCs —
either wire the CLI through `factory.tune`, or keep the distinction explicit.

## Tests & CI (two invariants that are easy to break silently)

```bash
pip3 install -e ".[dev]"
pytest -q          # CPU-only, ~2s
ruff check .       # config: [tool.ruff] in pyproject.toml, line-length 120
```

- **Every test is CPU-only.** No GPU, torch, triton or cuda-python — CI installs only
  `.[dev]` (pytest + ruff), so a test that needs any of them fails there while passing
  on a devgpu. This holds because `ptx_launch.py` guards its `cuda.bindings` import and
  imports torch/triton inside functions; keep it that way. To cover engine code, inject a
  stub `compileiq` into `sys.modules` (see `tests/test_adapter_compileiq.py`) rather than
  depending on the real engine. To check the invariant locally, run `pytest` with stub
  modules that raise `ImportError` ahead of the real ones on `PYTHONPATH`.
- **The `bare-install` CI job is a guard, not boilerplate.** It installs with *no extras*
  and runs `import ptx_anneal` + `ptx-anneal doctor` on the oldest supported Python, from
  outside the checkout. It exists to catch an extra being promoted into base
  `dependencies`. If it fails, fix the dependency — do not "fix" it by adding the extra.
- CI (`.github/workflows/ci.yml`): `test` (3.10–3.13), `bare-install`, `lint`. The lint job
  runs `ruff check` only — formatting is not normalized in this tree, so `ruff format` is a
  deliberate act, never a drive-by.

## Conventions

- `Target`/`SearchBackend`/`Runner`/`Store`/`Logger` are ABCs with local/null
  implementations shipped here; out-of-tree ones are injected by constructing them
  and passing them to `factory.tune` (no registry needed for runner/store/logger).
- Interop env vars from a frontend (e.g. fbtriton) — `TRITON_COMPILE_IQ_*`,
  `COMPILE_IQ_STORE`, `COMPILE_IQ_TASK_DIR` — are an external contract; do not
  rename them. ptx-anneal's own knobs use the `PTX_ANNEAL_*` prefix.

## Agent skills

Task-specific playbooks live in `skills/` as plain markdown (provisioning,
factory search, ACF store). They are vendor-agnostic and usable by any agent.
