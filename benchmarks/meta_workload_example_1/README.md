# meta_workload_example_1 — Triton kernels from a production recommendation-model training step

This directory is a ready-to-tune benchmark set for `ptx_anneal`: **59 kernel tasks** captured from a real
Meta training workload on NVIDIA B200 (`sm_100a`), together with the ptxas ACFs that `ptx_anneal --full-tune`
found for them. Use it to benchmark ACF search engines and search spaces on a realistic kernel mix, rather than on
hand-picked GEMM/attention kernels.

## The workload

- **Model:** a large multi-task ads recommendation (ranking) model. 
- **Run:** a local training run on 4× B200 GPUs (one trainer rank per GPU), 200 training batches, compiled with
  PyTorch 2 (`torch.compile` / Inductor) on Meta's Triton fork. ptxas 13.3 targets `sm_100a`.
- **Capture:** Triton's compile-IQ collector recorded each kernel's PTX and a source-free launch spec on its first
  real (non-benchmarking) launch. The task key is `sha256(normalized PTX)`, and the directory name is its first
  16 hex digits.
- **Selection:** the run captured about 1.5k kernels. These 59 are the ones a later training run of the same model
  looked up in the ACF store, so they are kernels the training step actually compiles and runs, not autotuning
  candidates that lost.

### Kernel mix

| Kind | Tasks |
|---|---|
| Inductor persistent reduction (`triton_per_*`) | 25 |
| Inductor pointwise (`triton_poi_*`) | 15 |
| Inductor looped reduction (`triton_red_*`) | 12 |
| Inductor combo/foreach (`triton_for_*`) | 5 |
| Hand-written TLX kernel (`_block_attention*`) | 2 |

Almost all of the Inductor kernels are fusions around **LayerNorm forward/backward** (often with `sigmoid` gating, `mul`/`addcmul`,
dtype casts or a fused `mm` epilogue), plus **`sum` reductions** that produce gradients. The two TLX kernels are a
warp-specialized block-attention forward and backward.

The kernels are **small**. Baselines range from 1.2 µs to 277 µs (median 3.2 µs),
and 37 of the 57 measurable tasks run in under 10 µs. That makes this set a useful stress test for search
noise: on microsecond kernels, a percentage win can easily be measurement jitter.

## How the ACFs were produced

```
ptx_anneal --task <task_dir> --full-tune --ptxas <ptxas 13.3>
```

- **Engine:** CompileIQ 1.0.2, using the published `ptxas13.3_search_space.bin` catalog.
- **Budget (`--full-tune`):** 20 generations, pool 32, cull 24, which came to 526–590 candidates per kernel. Candidates
  were scored on 8 B200 workers.
- **Scoring:** the CUDA-graph benchmark with warmup 25 and rep 50, on seeded random inputs. Every candidate's outputs
  were checked against the untuned baseline for self-consistency.
- **Admission:** an ACF is admitted only when the re-measured best beats the baseline.

## Results

| Outcome | Tasks |
|---|---|
| ACF admitted | 38 |
| ACF admitted with search win ≥ 2% | 19 |
| No candidate beat the baseline | 19 |
| Not tunable (corrupt `kernel.ptx`, see Known issues) | 2 |
| **Total** | **59** |

### ACFs with a search win of 2% or more

| # | Kernel | Task | Baseline (ms) | Best (ms) | Win | Saved (µs/launch) |
|---|---|---|---|---|---|---|
| 1 | `triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_sigmoid_backward_16` | `52342344baf17dda` | 0.0084 | 0.0056 | 33.58% | 2.8 |
| 2 | `triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_sigmoid_backward_8` | `4a30d8c7403d60a8` | 0.0115 | 0.0092 | 19.72% | 2.3 |
| 3 | `triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_sigmoid_backward_1` | `8e41b0bdbecfb80c` | 0.0155 | 0.0139 | 10.71% | 1.6 |
| 4 | `triton_per_fused_mul_native_layer_norm_sigmoid_0` | `8dac94c9c4bf5ebd` | 0.0018 | 0.0016 | 10.14% | 0.2 |
| 5 | `triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_sigmoid_backward_4` | `6ddfaa9bf3fadd82` | 0.1195 | 0.1096 | 8.21% | 9.9 |
| 6 | `triton_per_fused__to_copy_add_mm_mul_native_layer_norm_native_layer_norm_backward_10` | `1c87d475ee4bf97b` | 0.0440 | 0.0413 | 6.22% | 2.7 |
| 7 | `triton_red_fused__unsafe_view_clone_native_layer_norm_transpose_view_8` | `b9e5728dec5ecae1` | 0.0264 | 0.0252 | 4.52% | 1.2 |
| 8 | `triton_red_fused_sum_5` | `a9f336a37daf2792` | 0.0262 | 0.0250 | 4.41% | 1.2 |
| 9 | `triton_poi_fused_native_layer_norm_sigmoid_0` | `8a19422329c4c2c6` | 0.0016 | 0.0015 | 4.30% | 0.1 |
| 10 | `triton_red_fused_mul_native_layer_norm_sigmoid_2` | `83d0f0773af8ed56` | 0.0026 | 0.0025 | 4.08% | 0.1 |
| 11 | `triton_poi_fused_native_layer_norm_sigmoid_7` | `3e2234eaaf8b49b5` | 0.0014 | 0.0014 | 3.67% | 0.0 |
| 12 | `triton_poi_fused_native_layer_norm_sigmoid_15` | `c4747be8d727e4cc` | 0.0013 | 0.0012 | 3.56% | 0.1 |
| 13 | `triton_per_fused_addcmul_mul_native_layer_norm_native_layer_norm_backward_54` | `8bdcc7613b710f2c` | 0.2195 | 0.2121 | 3.40% | 7.4 |
| 14 | `triton_per_fused_addcmul_mul_native_layer_norm_native_layer_norm_backward_47` | `8e93fc6c4084b3e8` | 0.2196 | 0.2123 | 3.33% | 7.3 |
| 15 | `_block_attention` | `4bb6c1c96def3eef` | 0.0043 | 0.0042 | 3.27% | 0.1 |
| 16 | `_block_attention_backward` | `7d17e0c9bc719fe1` | 0.0046 | 0.0044 | 3.22% | 0.2 |
| 17 | `triton_red_fused_native_layer_norm_5` | `451d70a3b5b12d5c` | 0.0080 | 0.0077 | 2.96% | 0.3 |
| 18 | `triton_red_fused_sum_9` | `920b353744d3e664` | 0.0016 | 0.0015 | 2.55% | 0.1 |
| 19 | `triton_poi_fused_native_layer_norm_sigmoid_7` | `3f4c7c82d6cac4ff` | 0.0013 | 0.0013 | 2.16% | 0.0 |

Together these save about 37.6 µs per launch. Most of that comes from the three ~0.12–0.22 ms LayerNorm-backward
kernels (#5, #13, #14). The largest percentage wins (#1–#4) are on very short kernels, and wins on 1–3 µs kernels
(≈0.1 µs saved) are within measurement noise.

These are **search-time** numbers: the kernel runs standalone on synthetic inputs. In training, Triton re-times each
ACF against the plain kernel on first launch and keeps whichever is faster. Expect in-training wins to be smaller and noisier.

## Directory layout

```
meta_workload_example_1/
  <task id = first 16 hex of ir_hash>/
    kernel.ptx                 the frozen PTX to assemble (never recompiled)
    spec.json                  source-free launch spec (entry, arch, shared, block, grid, tensors, args)
    launch_metadata.json       compiler launch metadata (Level 0 schema)
    launcher.c                 the compiler-generated C launcher (Level 1)
    <ir_hash>.13.3.acf         tuned ACF, present for the 38 admitted tasks
    <ir_hash>.13.3.acf.json    its provenance: baseline/best ms, search win, budget, engine, search space
```

## Reproducing

```bash
# baseline timing of one task (no ACF)
ptx_anneal replay --task benchmarks/meta_workload_example_1/<task> --ptxas <ptxas 13.3>

# re-tune one task into a scratch store
ptx_anneal --task benchmarks/meta_workload_example_1/<task> --full-tune --ptxas <ptxas 13.3> --output /tmp/store

# assemble with the shipped ACF
ptxas -arch=sm_100a --apply-controls=<task>/<ir_hash>.13.3.acf <task>/kernel.ptx -o out.cubin
```

Pin `--ptxas` to a **13.3** ptxas. An ACF is bound to the ptxas version that produced it, and the `ptxas` path
inside each `spec.json` points at the capture machine.

## Known issues

- **Corrupt PTX in 2 tasks:** `6def2f96a7cd572a` and `80e35b95cb6901cf` have a `kernel.ptx` that ptxas rejects
  (`Parsing error near '{'` on the last line). All four trainer ranks captured the same kernel simultaneously,
  and the collector wrote `kernel.ptx` in place. Raw PTX differs in length per rank (debug info paths), so a shorter
  write left the tail of a longer one behind. They are kept as-is for completeness; delete the trailing junk after
  `.section .debug_macinfo { }` to use them.
- **No captured inputs:** the collector's `*.pkl` input captures were removed. `ptx_anneal` never reads them, since it
  benchmarks with seeded random inputs.

## All tasks

| Task | Kernel | Grid | Block | Baseline (ms) | Result |
|---|---|---|---|---|---|
| `4bb6c1c96def3eef` | `_block_attention` | 148×1×1 | 640 | 0.0043 | ACF, +3.27% |
| `7d17e0c9bc719fe1` | `_block_attention_backward` | 148×1×1 | 768 | 0.0046 | ACF, +3.22% |
| `c60a8933414d9a6c` | `triton_for_fused_0` | 8000×1×1 | 256 | 0.0297 | no win |
| `6ad2defd6f1333a9` | `triton_for_fused_1` | 3136×1×1 | 256 | 0.0057 | ACF, +0.14% |
| `4b3c4785e401d246` | `triton_for_fused_3` | 3180×1×1 | 256 | 0.0131 | no win |
| `b3e1bf481bb9d47f` | `triton_for_fused_4` | 3916×1×1 | 256 | 0.0145 | ACF, +0.00% |
| `0d2cca4a26668de3` | `triton_for_fused_5` | 1385×1×1 | 256 | 0.0035 | no win |
| `1c87d475ee4bf97b` | `triton_per_fused__to_copy_add_mm_mul_native_layer_norm_native_layer_norm_backward_10` | 16×1×1 | 256 | 0.0440 | ACF, +6.22% |
| `04faeb5d4b75e826` | `triton_per_fused__to_copy_add_mul_native_layer_norm_native_layer_norm_backward_13` | 16×1×1 | 256 | 0.0524 | no win |
| `8e41b0bdbecfb80c` | `triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_sigmoid_backward_1` | 16×1×1 | 256 | 0.0155 | ACF, +10.71% |
| `52342344baf17dda` | `triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_sigmoid_backward_16` | 16×1×1 | 64 | 0.0084 | ACF, +33.58% |
| `6ddfaa9bf3fadd82` | `triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_sigmoid_backward_4` | 1196×1×1 | 64 | 0.1195 | ACF, +8.21% |
| `4a30d8c7403d60a8` | `triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_sigmoid_backward_8` | 16×1×1 | 128 | 0.0115 | ACF, +19.72% |
| `f7356671e578376d` | `triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_sigmoid_backward_9` | 16×1×1 | 256 | 0.0146 | no win |
| `80e35b95cb6901cf` | `triton_per_fused_addcmul_mul_native_layer_norm_native_layer_norm_backward_22` | 16×1×1 | 512 | — | corrupt PTX |
| `8e93fc6c4084b3e8` | `triton_per_fused_addcmul_mul_native_layer_norm_native_layer_norm_backward_47` | 16×1×1 | 512 | 0.2196 | ACF, +3.33% |
| `dc4c06ca209d10fe` | `triton_per_fused_addcmul_mul_native_layer_norm_native_layer_norm_backward_53` | 16×1×1 | 512 | 0.2196 | ACF, +1.98% |
| `8bdcc7613b710f2c` | `triton_per_fused_addcmul_mul_native_layer_norm_native_layer_norm_backward_54` | 16×1×1 | 512 | 0.2195 | ACF, +3.40% |
| `6def2f96a7cd572a` | `triton_per_fused_mul_native_layer_norm_native_layer_norm_backward_17` | 16×1×1 | 512 | — | corrupt PTX |
| `2002bde47a2f5d98` | `triton_per_fused_mul_native_layer_norm_native_layer_norm_backward_39` | 16×1×1 | 512 | 0.2771 | no win |
| `ad0b1cf3d9d3bb71` | `triton_per_fused_mul_native_layer_norm_native_layer_norm_backward_43` | 16×1×1 | 512 | 0.2771 | no win |
| `ec6db5c5c7a7f26b` | `triton_per_fused_mul_native_layer_norm_native_layer_norm_backward_44` | 16×1×1 | 512 | 0.2772 | no win |
| `1c6b416f790ad2ac` | `triton_per_fused_mul_native_layer_norm_sigmoid_0` | 256×1×1 | 256 | 0.0019 | no win |
| `8dac94c9c4bf5ebd` | `triton_per_fused_mul_native_layer_norm_sigmoid_0` | 256×1×1 | 128 | 0.0018 | ACF, +10.14% |
| `2188b1dcdc44400b` | `triton_per_fused_mul_native_layer_norm_sigmoid_8` | 38272×1×1 | 32 | 0.0548 | no win |
| `5a28b36855b8e784` | `triton_per_fused_sum_10` | 8×1×1 | 64 | 0.0014 | no win |
| `b63814ef8ee69fc2` | `triton_per_fused_sum_11` | 16×1×1 | 64 | 0.0014 | no win |
| `9ae7f7d186815d4d` | `triton_per_fused_sum_14` | 50×1×1 | 64 | 0.0014 | no win |
| `6e134ae14a2638e3` | `triton_per_fused_sum_18` | 2×1×1 | 128 | 0.0013 | ACF, +0.21% |
| `2ed8334a5d0e65cb` | `triton_per_fused_sum_3` | 16×1×1 | 64 | 0.0014 | no win |
| `4dc7f1eedd22055d` | `triton_per_fused_sum_6` | 50×1×1 | 64 | 0.0014 | ACF, +0.06% |
| `2e4c067996a40e77` | `triton_per_fused_sum_7` | 1×1×1 | 128 | 0.0014 | no win |
| `89614588ebbddbd2` | `triton_poi_fused_add_mul_native_layer_norm_native_layer_norm_backward_sigmoid_sigmoid_backward_8` | 1024×1×1 | 256 | 0.0020 | ACF, +0.57% |
| `e0c0197ab5b7802e` | `triton_poi_fused_clamp_nan_to_num_permute_7` | 76544×1×1 | 128 | 0.0930 | ACF, +0.23% |
| `1c29ebe92727be54` | `triton_poi_fused_mul_sigmoid_1` | 1594×1×1 | 128 | 0.0025 | ACF, +1.86% |
| `9c6229afea201c5f` | `triton_poi_fused_mul_sigmoid_sigmoid_backward_12` | 1594×1×1 | 128 | 0.0028 | ACF, +1.65% |
| `9f34f59c1138a38d` | `triton_poi_fused_mul_sigmoid_sigmoid_backward_12` | 3188×1×1 | 256 | 0.0032 | no win |
| `30ccec41948eed8f` | `triton_poi_fused_mul_sigmoid_sigmoid_backward_4` | 1594×1×1 | 128 | 0.0028 | ACF, +1.94% |
| `faee837e7ac8bd18` | `triton_poi_fused_mul_sigmoid_sigmoid_backward_4` | 3188×1×1 | 256 | 0.0032 | no win |
| `8a19422329c4c2c6` | `triton_poi_fused_native_layer_norm_sigmoid_0` | 512×1×1 | 128 | 0.0016 | ACF, +4.30% |
| `921d38730f2c0ab3` | `triton_poi_fused_native_layer_norm_sigmoid_0` | 1024×1×1 | 256 | 0.0017 | ACF, +0.71% |
| `c4747be8d727e4cc` | `triton_poi_fused_native_layer_norm_sigmoid_15` | 128×1×1 | 128 | 0.0013 | ACF, +3.56% |
| `e8b2a2b0637a71f6` | `triton_poi_fused_native_layer_norm_sigmoid_15` | 256×1×1 | 256 | 0.0012 | ACF, +0.59% |
| `6369baf64dbdbde0` | `triton_poi_fused_native_layer_norm_sigmoid_3` | 38272×1×1 | 128 | 0.0517 | no win |
| `3e2234eaaf8b49b5` | `triton_poi_fused_native_layer_norm_sigmoid_7` | 256×1×1 | 128 | 0.0014 | ACF, +3.67% |
| `3f4c7c82d6cac4ff` | `triton_poi_fused_native_layer_norm_sigmoid_7` | 512×1×1 | 256 | 0.0013 | ACF, +2.16% |
| `4458ede4f86a7819` | `triton_poi_fused_pack_segments_transpose_6` | 8481×1×1 | 128 | 0.0068 | ACF, +0.13% |
| `b9e5728dec5ecae1` | `triton_red_fused__unsafe_view_clone_native_layer_norm_transpose_view_8` | 512×1×1 | 1024 | 0.0264 | ACF, +4.52% |
| `83d0f0773af8ed56` | `triton_red_fused_mul_native_layer_norm_sigmoid_2` | 128×1×1 | 256 | 0.0026 | ACF, +4.08% |
| `451d70a3b5b12d5c` | `triton_red_fused_native_layer_norm_5` | 256×1×1 | 1024 | 0.0080 | ACF, +2.96% |
| `f3fcd2cadacfd2cd` | `triton_red_fused_sum_10` | 256×1×1 | 128 | 0.0017 | ACF, +0.60% |
| `c5407358395cac0a` | `triton_red_fused_sum_13` | 797×1×1 | 128 | 0.0024 | ACF, +1.45% |
| `f09effc206db8538` | `triton_red_fused_sum_17` | 128×1×1 | 128 | 0.0016 | ACF, +0.37% |
| `50db800e596dbdc6` | `triton_red_fused_sum_2` | 256×1×1 | 128 | 0.0017 | ACF, +1.15% |
| `d27d5f0619629578` | `triton_red_fused_sum_2` | 1196×1×1 | 128 | 0.0072 | no win |
| `a9f336a37daf2792` | `triton_red_fused_sum_5` | 2392×1×1 | 256 | 0.0262 | ACF, +4.41% |
| `e96c44ca46757285` | `triton_red_fused_sum_5` | 797×1×1 | 128 | 0.0024 | ACF, +1.13% |
| `c1d9d92371ab7a17` | `triton_red_fused_sum_6` | 128×1×1 | 64 | 0.0016 | ACF, +0.14% |
| `920b353744d3e664` | `triton_red_fused_sum_9` | 128×1×1 | 128 | 0.0016 | ACF, +2.55% |
