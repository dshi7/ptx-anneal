# Skill: Inspecting and managing the tuning-artifact store

Use this to find, inspect, or clear cached tuning artifacts (ACFs) — e.g. when a
consume run unexpectedly misses, or after re-tuning.

## Layout

The store is content-addressed by `(target, arch, ir_hash)`. For the `ptx`
target the `ir_hash` is `sha256` of the normalized PTX (line/debug info stripped
so cosmetic differences don't change the key). Paths use `hash16 = ir_hash[:16]`; the full
hash stays in the `.acf.json` sidecar and in listings when that metadata is available.
A local store is a directory tree:

```
<store>/<arch>/<hash16>/<hash16>.<ptxas>.acf          # the tuning artifact (controls)
<store>/<arch>/<hash16>/<hash16>.<ptxas>.acf.json     # provenance sidecar
<store>/<arch>/<hash16>/<hash16>.<ptxas>.acf.cubin    # optional: assembled cubin cache
```

e.g. `sm_100a/1e359f0358047d9e/1e359f0358047d9e.13.3.acf`. The **ptxas version is part of the filename**:
an ACF is bound to the ptxas that produced it, so ACFs from different ptxas
versions coexist instead of overwriting each other. The kernel identity
(`ir_hash`) stays version-free. (`LocalStore` also supports an untagged
`<hash16>.acf` when written with no `toolchain_version`; the CLI always tags.)

New writes use the short kernel directory. Reads and listings support the older full-hash
directories and flat `<store>/<arch>/<filename>` layout. Short nested artifacts take precedence,
followed by full-hash nested artifacts, then flat artifacts. Existing files are not moved or deleted.
Consumers constructing paths directly must adopt the 16-character nested layout to find new ACFs.

## Inspecting

There is no `store` subcommand — the store is plain files, so read them directly:

```bash
ls <store>/<arch>/<hash16>/                          # one kernel's cached artifacts
cat <store>/<arch>/<hash16>/<hash16>.<ptxas>.acf.json  # one artifact's provenance
```

Or from Python, via the `Store` ABC:

```python
from ptx_anneal.store import LocalStore, default_store_root  # default: ~/.compile_iq/store

s = LocalStore(default_store_root())
s.list()  # every cached artifact
s.read_meta("ptx", arch, ir_hash, toolchain_version="13.3")  # the sidecar
```

## The sidecar

Records the result *and* every input that could change it — start here when a
result is surprising or won't reproduce:

```json
{
  "target": "ptx", "arch": "sm_100a", "ir_hash": "8455732c…", "entry": "add_kernel",
  "baseline_ms": 0.000961, "best_ms": 0.000932, "search_win": 0.0300,
  "forced": false, "evaluated": 8, "valid": 8,
  "ptxas_version": "13.3",
  "engine": "compileiq",
  "engine_info": {"name": "compileiq", "version": "1.0.0.dev1", "path": "…/compileiq", "python": "…"},
  "search_space": {
    "compiler": "ptxas", "compiler_version": "13.3", "variant": "default",
    "requested_tag": "latest", "resolved_tag": "search-spaces-2026.05.22",
    "filename": "ptxas13.3_search_space.bin", "sha256": "edae9fe3…",
    "size_bytes": 13632, "source": "github_release"
  }
}
```

- `forced: true` → admitted under `PTX_ANNEAL_FORCE_ADMIT` **without** beating the
  baseline. Not a win; it exists so the consume path can be exercised.
- `search_space` matters because the catalog defaults to `latest`. Two ACFs for the
  same kernel can come from different catalogs, and the store key does **not**
  distinguish them — `resolved_tag`/`sha256` is the only record of which was used.
  When `--ss`/`PTX_ANNEAL_SS` pinned an explicit file, `source` is `PTX_ANNEAL_SS`
  and only the path is recorded. When it was downloaded from manifold, `source`
  is `manifold` with the `url` and `sha256`.
- `search_win` is a noisy search-time number, not a promise; see
  `skills/factory-search.md`.

## Common questions

- **Why did consume miss?** The consumer computes the key from the *post-compile*
  IR. If the kernel changed (different PTX → different `ir_hash`) the key won't
  match. Confirm the `ir_hash` of the current kernel equals a stored one.
- **Version mismatch.** An artifact is only valid for the `ptxas` that produced it,
  so both the `.acf` and the assembled-cubin cache are tagged by ptxas version. Note
  the frontend's ptxas also decides the PTX ISA version it emits, so a version
  mismatch usually shows up one step earlier, as an `ir_hash` that never matches.
- **Re-tuning / staleness.** Writing a new `.acf` for the same key should
  invalidate any derived `.acf.cubin` for that key, or later runs may serve the
  stale assembly. When in doubt, delete the derived `.acf.cubin` files.

## Keying note

`ir_hash` subsumes autotune-config and dtype/alignment specialization (they
change the IR) but **not** runtime shape (M, N, K) when the kernel is
shape-agnostic — so one artifact can transfer across shapes of the same kernel.
