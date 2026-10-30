# PTX-anneal

Offline ptxas tuning framework (ACF factory) for faster GPU kernels.

`ptx-anneal` searches ACF (Advanced Control File, encrypted `ptxas` advanced controls) for a compiled GPU kernel, verifies
the result is numerically correct against the untuned baseline, and caches the winning
ACF in a content-addressed store. Applying a cached ACF at
runtime yields faster SASS with no change to your kernel source or on-disk compilation cache.

`ptx-anneal` is the orchestration harness only. The optimization engine
([NVIDIA CompileIQ](https://github.com/nvidia/compileiq)) is **bring-your-own**: install it with
`pip install ptx-anneal[compileiq]` (or bring a different one), and `ptx-anneal` treats it as a black
box behind a plugin interface. The harness never imports the engine — it drives it through an
adapter subprocess, so the engine can live in a different interpreter entirely.

## How it works — a complete local flow with [fbtriton](https://github.com/facebookexperimental/triton)

Step 1. Task collection: fbtriton's JIT hook dumps each compiled PTX as an ACF tuning task.
```bash
TRITON_COMPILE_IQ_COLLECT=1 COMPILE_IQ_TASK_DIR=$TASKS python your_workload.py
```

Step 2. ACF factory: Search for the ACF whose ptxas-assembled SASS runs fastest.
```bash
# one engine-free binary. Everything else is provisioning and self-provisions by default, so the
# per-run command is just the task:
ptx_anneal --task $TASKS/<task> --output $STORE
```
Run `ptx_anneal doctor` first to check the toolchain. Each piece can be pinned when the default
isn't what you want:

| Piece | Default | Pin with |
|-------|---------|----------|
| Search engine | `pip install ptx-anneal[compileiq]`, in the same env as `ptx_anneal` | `--engine-python` / `--engine-adapter` |
| Search space | the engine fetches its published catalog (`latest`) and caches it under `~/.cache/compileiq/<tag>/` | `--ss <bin>`, or `CIQ_SS_TAG` / `CIQ_SS_VERSION` / `CIQ_SS_VARIANT`, or `CIQ_SEARCH_SPACES_DIR` for an offline mirror (with no `--ss` and `manifold` on `PATH`, it is downloaded from `manifold://tc_bench_ci/tree/compileiq/ptxas_knobs/ptxas<ptxas_version>_search_space.bin` instead) |
| `ptxas` (>= 13.3) | `$PTXAS` / `$TRITON_PTXAS_BLACKWELL_PATH`, else `PATH`, else a pip-installed `nvidia-cuda-nvcc` wheel | `--ptxas` |

Because the search space is a tuning *input* and resolves to `latest` by default, every admitted ACF
records which one produced it (`resolved_tag` + `sha256`) in its `.acf.json` sidecar, alongside the
ptxas version and the engine build (`engine_info`) — so a result stays explainable after the catalog
moves on. Pin `CIQ_SS_TAG` for reproducible runs.

For a task stored on Manifold, add `--upload-manifold` to upload the admitted ACF and its
`.acf.json` provenance sidecar back to that task directory:

```bash
ptx_anneal --task manifold://<bucket>/<task_directory> --upload-manifold
```

The files keep their version-tagged names under a new `compileiq_YYYYmmddHHMMSS` subdirectory
(UTC, at upload time). This tuning-only flag requires a `manifold://` task directory. If no ACF
is admitted, upload is skipped. An upload failure exits nonzero and preserves the local ACF
and sidecar so they can be recovered without rerunning the search.

Step 3. ACF consumption: On an ACF-store hit, re-assemble the kernel with the ACF (ptxas --apply-controls) and run the tuned SASS.
```bash
TRITON_COMPILE_IQ_APPLY=1 COMPILE_IQ_STORE=$STORE python your_workload.py
```

## Tune register spills from a PTX file

`register_spill` uses CompileIQ to minimize the sum of spill-store and spill-load bytes reported
by `ptxas -v`, summed across all functions in the file:

```bash
ptx_anneal register_spill --task /path/to/kernel.ptx
# Optional compiler/search-space pins and a larger search budget:
CIQ_POOL=16 CIQ_GENERATIONS=4 ptx_anneal register_spill --task kernel.ptx --ptxas /path/to/ptxas --ss space.bin
```

Here `--task` is a **local PTX file**. The architecture comes from its `.target` directive. Only
Python, ptxas >= 13.3 and the installed engine are needed; there is no launch spec or GPU execution.
The usual `--engine-python`, `--engine-adapter`, and search-space environment pins also apply.

Each candidate is compiled with `--apply-controls`. Compilation failures or missing spill diagnostics
are invalid; zero spill bytes is valid. Only a candidate that reduces spills against the untuned
baseline is saved. A completed search returns 0 even if no candidate improves on the baseline.

Winning ACFs and `.acf.json` provenance are written under `register_spill/` in the default ACF
store (`$COMPILE_IQ_STORE` or `~/.compile_iq/store`), or to the store selected by `--output`/`--store`.
The sidecar records the spill objective and scores, ptxas version, engine identity, and search space.
Spill bytes are a static compiler metric: this command does not measure speed or validate numerical
correctness, so its default output is separate from the runtime tuning store.

## Replay PTX without tuning

`replay` assembles a captured task and benchmarks it without CompileIQ, a search space, or an ACF:

```bash
pip3 install '.[harness,cudapy]'  # from this checkout; no engine required
ptx_anneal replay --task /path/to/task
```

The task directory must contain `kernel.ptx` and `spec.json`. Both replay and tuning accept a local
directory or a `manifold://` directory URL (file URLs such as `spec.json` are rejected):

```bash
ptx_anneal replay --task manifold://tc_bench_ci/tree/magnon/aba5e3348816add4
```

Remote tasks require an authenticated `manifold` CLI on `PATH`. The two files are downloaded once
to a temporary directory, which is removed when the run finishes, including on failure.
Local tasks do not need Manifold.

Replay supports BF16, strided tensors, host TMA descriptors and
CTA clusters. Torch allocates tensors; Triton is used only to encode TMA descriptors, not to
recompile the kernel. TMA tasks require a Triton version with the descriptor encoder used by
the capture frontend (fbtriton for fbtriton captures).

Inputs are deterministic synthetic tensors matching the captured shapes, dtypes and strides.
Replay does not restore captured tensor values or check numerical correctness against original
outputs. It prints `MS <milliseconds>` on success and exits nonzero on failure.

Use `--bench do_bench` for L2-cold timing (default: `cudagraph`), and `--warmup` / `--rep` to set
the timing budget. `--ptxas` pins the assembler; otherwise normal environment/PATH/wheel discovery
applies. The captured spec's machine-specific `ptxas` path is ignored; the >= 13.3 floor still applies.

## An optional second channel: triton-native

Alongside the shipped PTX-direct path above, the repo carries an **optional, on-demand** runway that
scores the *same* kernel a different way: it sets `PTXAS_OPTIONS=--apply-controls=<acf>` and lets
[fbtriton](https://github.com/facebookexperimental/triton) compile and launch the kernel normally —
no PTX capture, no launch spec. (`PTXAS_OPTIONS` is fbtriton's knob; the runway checks for fbtriton
and verifies per candidate that the ACF actually reached the compiler, since a Triton without that
knob would compile untuned and still report a score.)

```bash
python -m e2e.triton_native.runway               # the canary
python -m e2e.triton_native.runway --kernel k    # canary, then your kernel
```

It exists to make results **attributable**. The two channels share an engine, an adapter and a
scoring contract, and differ only in how a candidate is executed — so comparing them separates a
question about the compiler from a question about our harness:

| triton-native | PTX-direct | conclusion |
|---|---|---|
| fail | — | the finding is in the compiler/engine path, reproducible with no `ptx-anneal` code |
| pass | fail | the finding is in `ptx-anneal` — the captured repro or the consume path |
| pass | pass, numbers disagree | also `ptx-anneal`, and the easiest kind to miss: both look green |
| pass | pass, numbers agree | trustworthy |

That comparison only means something if both channels posed the **same search problem**, so both
print a **search-problem fingerprint** at the end of a run — kernel identity, ptxas path/version/flags,
the *resolved* search space, the frontend that compiled it, engine and budget, the objective (metric,
timing method and budget, correctness oracle and tolerance) and the environment (GPU, CUDA, driver,
clock state), plus a digest
over all of it. Equal digests mean the two runs are comparable; `diff` shows exactly what moved when
they are not. The fingerprint is also what makes a result shareable: it is a complete, self-contained
statement of what was measured and under what conditions.

The channel is **off by default and not in the wheel or sdist** — it needs torch and fbtriton, and
without them it reports itself unavailable and nothing else changes. Defaults follow NVIDIA's
documented CompileIQ example wherever that is practical, and the deviations that remain are recorded
in the fingerprint rather than left implicit. See
[`e2e/triton_native/README.md`](e2e/triton_native/README.md).

## How the CLI runs a search

`ptx_anneal --task ...` is local and single-host: it scores the untuned baseline in-process,
then spawns the **engine adapter** as a subprocess and lets the engine drive the search, scoring
each candidate by calling this same binary back as `ptx_anneal _score`. The winner is admitted to a
local filesystem store. The harness never imports the engine, so it can be a frozen binary and the
engine can live in its own interpreter.

Baseline and candidate scoring select the same implementation from the task spec. Ordinary
contiguous tensors use cuda-python + NumPy. BF16, TMA and strided tensors use the torch-based
PTX scorer (install `.[harness,cudapy]` from the checkout); TMA captures need the matching Triton
descriptor encoder. Both paths check candidate outputs against an untuned run before timing,
honor `--bench`, and use the selected `--ptxas` instead of a captured machine-specific path.

```
ptx_anneal --task ──> baseline score (in-process)
                └───> $PTX_ANNEAL_ENGINE_ADAPTER (subprocess, engine's own interpreter)
                        └──> ptx_anneal _score <candidate>   (one call per candidate)
                    admit winner ──> LocalStore
```

## Extending ptx-anneal

Two seams are wired into the CLI:

| Seam | How | Ships here |
|------|-----|-----------|
| Search engine | `--engine-adapter` / `$PTX_ANNEAL_ENGINE_ADAPTER` — a standalone script run in the engine's interpreter (see `ptx_anneal/adapters/`) | `compileiq_adapter.py` (bundled default) |
| Per-IR backend | `--target`, resolved from built-ins + the `ptx_anneal.targets` entry point | `PtxTarget` (`ptx`) |

Env knobs: `PTX_ANNEAL_ENGINE_PYTHON` (interpreter that has the engine) and
`PTX_ANNEAL_WORKER_PYTHON` (interpreter for the per-candidate benchmark, when it differs).

There is also a **library-level** dependency-injection API — `factory.tune(task, target=, store=,
runner=, search=, logger=)` over the `Store` / `Runner` / `SearchBackend` / `Logger` ABCs, with
`LocalStore`, `LocalRunner`, `CudaPyRunner`, `BaselineSearch` and `NullLogger`/`StderrLogger`
shipped. Use it to embed ptx-anneal in your own orchestration (e.g. a remote store or fleet
runner). Note the CLI does **not** currently go through it — it uses the adapter path above — so
these ABCs are exercised by the test suite rather than by `ptx_anneal --task`.

The `ptxas >= 13.3` floor is fixed and not overridable — see [Requirements](#requirements).

## Install

```bash
pip3 install ptx-anneal[compileiq]   # harness + the reference search engine
```
(Until it's published, install from a checkout: `pip3 install .[compileiq]`.)

Extras: `compileiq` (the reference engine), `cudapy` (torch-free scorer), `harness` (PTX-direct
benchmarking; needs torch+triton), `dev` (pytest + ruff). Plain `pip3 install ptx-anneal` gets the
store/ABC/`doctor` layer only — pure Python, no GPU, no engine.

The wheel contains the `ptx_anneal` package and nothing else. The validation harnesses (`e2e/`,
including the triton-native channel) are development tools and are excluded from both the wheel and
the sdist, so installing `ptx-anneal` never pulls in torch or triton.

## Requirements

- `ptxas` >= 13.3 — from the CUDA Toolkit, or `pip3 install nvidia-cuda-nvcc` (a wheel-installed
  `ptxas` isn't on `PATH`, but `ptx-anneal` looks for it).
- An optimization engine — NVIDIA CompileIQ, via the `compileiq` extra above. It is installed, never
  vendored, and the harness never imports it. It also supplies the search space, so there is nothing
  else to fetch by hand. Run `ptx-anneal doctor` to check your toolchain and print provisioning
  instructions.
- The engine is an **extra** rather than a base dependency because it narrows the support matrix
  (Python `>=3.11,<3.14`, glibc >= 2.34, Linux/Windows wheels) while the rest of `ptx-anneal`
  installs anywhere. Bring your own engine instead by pointing `--engine-adapter` at your own
  adapter script.
- A frontend that emits ACF tasks and can apply an ACF (e.g. fbtriton).
- Network access to the CompileIQ search-space releases on first use. Air-gapped or CI hosts should
  mirror the catalog and set `CIQ_SEARCH_SPACES_DIR`, or pass an explicit `--ss`.

> **Use one `ptxas` for all three steps.** The 13.3 floor is a hard requirement, not a default:
> `--apply-controls` is GA only from 13.3 on, *and* the frontend derives the PTX ISA version it emits
> from its own `ptxas` version. Collecting under a different `ptxas` therefore produces different PTX,
> a different store key, and a guaranteed MISS at consume — which fails open to the untuned kernel, so
> it looks healthy while delivering nothing.

## Development

```bash
pip3 install -e ".[dev]"
pytest -q                 # CPU-only: no GPU, torch, triton or cuda-python needed
ruff check .              # lint  — CI gates on this
ruff format --check .     # format — CI gates on this
ruff check --fix . && ruff format .   # apply both
```

Lint and format config live in `[tool.ruff]` / `[tool.ruff.lint]` in `pyproject.toml` (line length
120), so an editor, CI and the internal linter all read the same rules. `ruff` is pinned in the
`dev` extra: it is a CI gate, and an unpinned release can turn the build red with no code change.

CI runs three jobs — `test` (Python 3.10–3.13), `lint`, and `bare-install`, which installs with **no
extras** and runs `ptx-anneal doctor` to prove the pure-Python core still works without a GPU or an
engine.

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out, and our
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

`ptx-anneal` is MIT licensed, as found in the [LICENSE](LICENSE) file.
