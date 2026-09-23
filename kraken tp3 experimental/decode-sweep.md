# Final decode sweep — kraken-port and the running image, 2026-09-22

One more pass for anything that could still raise decode speed, guided by the
C1 step budget in [profile-20260922](../profile-20260922/RESULTS.md) and
checked against every earlier port's results before spending GPU time.

Production was recreated twice (one failed candidate, one restore) and ended
on `glm53-kraken-tp3:baked-20260921`, default/MTP3/DCP1, `FLASHINFER_CUTLASS`
MoE, health 200, 8/8 exact arithmetic. No image, compose, launcher or source
change.

## Nothing left at the configuration level

| Candidate | Result | Evidence |
| --- | --- | --- |
| Async scheduling | Already on. MTP is in `EagleModelTypes`, so `async_scheduling=None` resolves to `True`; no disable warning in the log. | `vllm/config/vllm.py` |
| `B12X_PCIE_ALLREDUCE_ALGORITHM` | Only `oneshot` accepts world size 3 (`hierarchical` 12/16, `island_rs` 16). | profile RESULTS |
| PCIe two-shot all-reduce | World size 4 only ("Only world size 4 is supported"). | `b12x/docs/evidence/pcie_twoshot_bf16_sm120.md` |
| `VLLM_GLM53_L2_PREFETCH_PERSIST_MB=max` | 0.0% at R34, not adopted. | `r34-port/optimization/decode-20260919` |
| Quarter L2 budgets | -1.1 to -1.5% at R34; half remains best measured. | `r34-port/optimization/dcp3-20260910` |
| Inductor compilation (`mode: 3`) | -4.8% C1, and the B12X fusion still does not register. | [compile-fusion-20260922](../compile-fusion-20260922/RESULTS.md) |
| MoE `FLASHINFER_CUTEDSL` | `_supports_current_device` requires the SM100 family. | `experts/flashinfer_cutedsl_moe.py` |
| MoE `VLLM_CUTLASS` (`--moe-backend cutlass`) | Supports SM120 and was never measured on any port -- `auto` stops at FlashInfer first -- but **the engine refuses to start**: "does not support parallel config ... ep_size=3". TP3 requires EP3 because the routed-expert width is not divisible by 3. | `compose.cutlass.yaml`, this run |
| MoE `FLASHINFER_TRTLLM` | Unsupported on SM120 (R34). | R34 notes |

## The low-latency GEMM plan is still correct on this image

`glm53_low_latency_gemm.py` routes BF16 projections to the CuTe skinny GEMM by
exact `(N, K)` from a plan measured on R34. The worry was that kraken's
geometry or kernels had drifted. They have not:

- The log reports **168** target projections swapped, which is 34 KDA layers x
  3 (`in_proj` 8598x4096, `g_a_proj`, `o_proj`) + 11 DSA layers x 6. Kraken's
  checkpoint uses the non-full-rank-gate KDA layout, so `in_proj` is still 8598
  wide and matches.
- R34's own `bench_linear_shapes.py`, re-run unchanged on this image
  (`bench_linear_shapes-kraken.json`), reproduces R34's numbers to within a few
  percent. Every planned `(shape, M)` still beats cuBLAS.
- One marginal gap: `kda.o_proj` at M=4 now wins on skinny (5.90 vs 6.27 us,
  -5.9%, over the plan's 3% bar). 34 layers x 0.37 us = ~13 us/step, **~0.1%**.
  Not worth a rebuild on its own.

## SM80 fallback GEMMs: identified, not worth fixing

The profile shows legacy `cutlass_80_*` cuBLAS kernels on an SM120 GPU. Launch
grids `(8, 22)` of 16x16 tiles give N = 2816 = 22 heads x 128, and the plan
count identifies them as the two KDA gate up-projections, `f_b_proj` and
`g_b_proj` (both 2816x128, neither in the plan). `f_b_proj` takes the `align2`
variant because its input is a slice of the 8598-wide `in_proj` output (row
pitch 17,196 B, not 16-byte aligned; the full-rank-gate path pads this, the
path this checkpoint uses does not).

At C1 they are **70% hidden** (68 us/step exposed, at most 0.5% of occupied
GPU time). R34 already measured skinny as no faster for these shapes. The C8
capture showed them 92% exposed, but those launches are ~4/step rather than 34
-- prefill-sized batches from that contaminated window, not decode.

## Lower-precision dense weights: the largest lever left, and a quality change

BF16 dense projections are ~3.7 ms of the 13.09 ms C1 step, all weight
streaming, because `modelopt_mixed` keeps attention/KDA projections in BF16.

`dense-fp8-roofline.py` (supersedes `dense-precision-bench.py`, whose skinny
baseline and L2 handling were both wrong) compares the exact production path
-- skinny GEMM where the plan routes (shape, M), cuBLAS elsewhere -- against
Marlin FP8 W8A16, in two regimes: **hbm** (rotating weight copies past the
128 MB L2) and **l2** (one weight re-read, the fully-prefetched best case).
Results in `dense-fp8-roofline.json`.

Harness pitfall found on the way: the skinny kernel compiles asynchronously.
Capturing a CUDA graph immediately after the first call bakes in the
uncompiled fallback (~45 us for `in_proj` instead of 12.6 us). Warm up with a
few hundred eager calls and a pause before capture.

Valid C1 (M=4) numbers for the shapes that matter:

| Shape | uses/step | BF16 hbm -> FP8 | BF16 l2 -> FP8 |
| --- | ---: | ---: | ---: |
| `kda.in_proj_qkvgfab` 8598x4096 (skinny) | 34 | 44.4 -> **26.9 us** | 12.6 -> 12.4 |
| `dsa.o_proj` 4096x6144 (skinny) | 11 | 31.8 -> **19.3** | 9.6 -> 8.5 |
| `kda.o_proj` 4096x2816 (cuBLAS) | 34 | 16.5 -> **10.7** | 6.6 -> 8.4 |
| `dsa.q_b_proj` 6144x1536 (cuBLAS) | 11 | 13.9 -> 10.3 | 6.5 -> 8.3 |
| LM head 51648x4096 (cuBLAS) | 1 | 260.8 -> **136.7** | -- |

C8 (M=32, all cuBLAS): `in_proj` 50.7 -> 27.7, `kda.o_proj` 17.9 -> 11.5,
`dsa.o_proj` 34.9 -> 20.2, LM head 267 -> 140 us.

Small shapes lose badly and stay BF16: `g_a_proj`, `idx.wk`,
`idx.weights_proj` go from ~2 us to ~28 us under Marlin; `f_b`/`g_b` break even.

FP8 halves the bytes where a weight streams from HBM and ties where it already
sits in L2. Production is between: the profile's skinny total (1.95 ms over 112
launches/step) implies `in_proj` runs near its HBM figure, and a fixed prefetch
budget covers twice the fraction of a half-size FP8 weight. Estimated C1
saving from `in_proj`, both `o_proj`s, `q_b_proj` and the LM head:
**~0.85-1.1 ms/step, roughly 7-9%**; at C8 ~1.2 ms of a ~30 ms engine step,
~4%. Dropping the BF16 copies of those layers would also free ~2 GB per GPU
for KV.

What it would take:
1. A load-time hook like `enable_glm53_low_latency_gemm`: quantize the chosen
   layers' weights to FP8 per channel and route decode through Marlin W8A16.
2. A large-M path so prefill does not regress (Marlin is weak at large M; a
   CUTLASS FP8 W8A8 path on the same weights would serve prefill chunks).
3. Bracketed serving A/B.
4. Quality qualification (lavd / hotel-lights): the checkpoint authors kept
   these layers BF16 while quantizing experts to NVFP4. Per-channel FP8 E4M3
   weight-only is the mildest quantization available, but it is not free.

The fork's existing `VLLM_MXFP8_LM_HEAD` switch is **not** this: it uses
b12x's MXFP8 a16 kernel and measured **-0.8% C1 / -0.9% C8 steps/s** in a
tight bracket ([fp8-lmhead-20260922](../fp8-lmhead-20260922/RESULTS.md)).

## Where that leaves TP3 decode

| Lever | Size | Cost |
| --- | ---: | --- |
| 256 KiB one-shot cutoff | +2-4% at C8, 0 at C1 | env-only, already plumbed; measured twice; mechanism now proven (C8's 256 KiB messages fall to NCCL ring at 44.9 us vs 16.1 us) |
| FP8 (Marlin W8A16) for large BF16 dense layers incl. LM head | ~7-9% C1, ~4% C8 (microbench estimate) | load-time hook + prefill path + A/B + quality qualification |
| TP3 push transport for the one-shot | ~3% C1 | b12x kernel + IPC layout work; hang risk; needs downtime ([pcie-fastpath-20260922](../pcie-fastpath-20260922/RESULTS.md)) |
| All-reduce -> mHC PDL | measured +0.2% C1 (noise), 0 at C8 | done as a mounted overlay; live and correct |
| `kda.o_proj` M=4 plan entry | ~0.1% | one-line plan edit + rebuild |
| KDA gate GEMM alignment | <0.5% | model code change |
