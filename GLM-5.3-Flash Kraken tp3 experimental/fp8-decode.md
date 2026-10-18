# FP8 weight-only decode for the large BF16 projections — speed confirmed, 2026-09-23

## What it is

`src/vllm/models/glm5next/nvidia/glm53_fp8_dense.py` (new) plus 6 lines in
`model.py`. Opt-in via `VLLM_GLM53_FP8_DENSE=1`; mounted by
`compose.fp8-dense.yaml` on top of the production compose.

- Converts 90 projections per rank -- KDA `in_proj_qkvgfab` (8598x4096) and
  `o_proj` (4096x2816) x34, DSA `o_proj` (4096x6144) and `q_b_proj`
  (6144x1536) x11 -- plus the LM head (51648x4096).
- After loading, each weight is quantized to FP8 E4M3 with one scale per output
  channel and packed for Marlin W8A16 (activations stay BF16).
- Batches of at most 32 rows (`VLLM_GLM53_FP8_DENSE_MAX_M`; every decode and
  verify step up to 8 x 4) use FP8. Larger batches -- prefill chunks -- call the
  original method's own GEMM on the retained BF16 weight, so prefill numerics
  and kernels are unchanged.
- The packed FP8 tensor becomes `layer.weight` so the L2 prefetcher streams
  what decode reads; the BF16 original sits in a private attribute it skips.
  The LM head keeps its BF16 `weight` and carries the FP8 copy alongside.
- Small projections (`g_a_proj`, indexer `wk`/`weights_proj`) are left BF16:
  Marlin is ~15x slower on them.

Unit test before serving (real pack/apply functions, random weights with
all-zero padding rows): cosine >= 0.99964 and relative error <= 2.7% per
projection against BF16 at M = 1/4/8/32, padding rows exactly zero, CUDA-graph
replay identical to eager.

Serving smoke: 90 projections + LM head converted (log), 8/8 arithmetic, a
coherent code answer with natural stop. KV capacity 3,033,780 tokens vs
3,410,670 (-11%), the cost of keeping BF16 copies for prefill.

## Speed bracket

Control is `compose.yaml` as it stood during the run: the baked image **without** PDL (its two mount lines had been commented out), so both arms lack PDL and the gain is FP8 alone.

| Arm | C1 steps/s 0 / 32K | C1 tok/s 0 / 32K | C8 steps/s | C8 tok/s |
| --- | --- | --- | ---: | ---: |
| control-open | 79.43 / 79.06 | 205.9 / 203.2 | 261.58 | 653.6 |
| **fp8-dense** | **86.03 / 85.61** | 212.5 / 213.1 | **274.00** | 704.3 |
| control-close | 78.97 / 78.88 | 194.9 / 196.8 | 262.70 | 675.2 |

| | vs control mean | controls differ |
| --- | ---: | ---: |
| C1 steps/s, 0 ctx | **+8.6%** | 0.6% |
| C1 steps/s, 32K | **+8.4%** | 0.2% |
| C8 steps/s | **+4.5%** | 0.4% |

Exactly the microbenchmark estimate (7-9% C1, ~4% C8). C1 at 86 steps/s is
the TP2 (73) / TP4 (100+) interpolation for TP3. tok/s moves with acceptance,
which varies ~5% between identical controls; steps/s is the metric.

## Status

Changes decode numerics of layers the checkpoint left in BF16, so it is **not
production** until qualified: `../../qualification-20260923-fp8/`.

Possible follow-ups if adopted: drop the retained BF16 copies (recovering the
11% KV and more) by giving prefill an FP8 path; extend to the MTP draft layer.

## Run log

```
2026-09-23T01:04:45.290193+00:00 — control-open up | FP8 decode active: False | arithmetic 8/8
2026-09-23T01:07:28.433693+00:00 —   control-open C1: [(0, 205.9, 79.43, 2.591, False), (32768, 203.2, 79.06, 2.57, False)]
2026-09-23T01:09:11.170226+00:00 —   control-open C8: [(0, 653.6, 261.58, 2.499, False)]
2026-09-23T01:11:35.672819+00:00 — fp8-dense up | FP8 decode active: True | arithmetic 8/8
2026-09-23T01:14:19.000137+00:00 —   fp8-dense C1: [(0, 212.5, 86.03, 2.47, False), (32768, 213.1, 85.61, 2.489, False)]
2026-09-23T01:16:01.755258+00:00 —   fp8-dense C8: [(0, 704.3, 274.0, 2.571, False)]
2026-09-23T01:18:26.520924+00:00 — control-close up | FP8 decode active: False | arithmetic 8/8
2026-09-23T01:21:09.191548+00:00 —   control-close C1: [(0, 194.9, 78.97, 2.468, False), (32768, 196.8, 78.88, 2.495, False)]
2026-09-23T01:22:51.699670+00:00 —   control-close C8: [(0, 675.2, 262.7, 2.57, False)]
2026-09-23T01:22:51.699832+00:00 — restoring production compose
2026-09-23T01:25:01.208915+00:00 — production restored | arithmetic 8/8
```

## PDL on top of FP8 (2026-09-23)

Alternating C1 bracket (`run_pdl_on_fp8.py`, `pdl-on-fp8.json`,
`PDL-ON-FP8.log`), FP8 build with and without `../pcie-fastpath-20260922/
compose.pdl.yaml`:

| | C1 steps/s |
| --- | --- |
| FP8 | 85.98, 85.72, 86.06 (spread 0.40%) |
| FP8 + PDL | 86.36, 86.44 |

**+0.56% mean; both PDL arms above all three FP8 arms** (lowest PDL beats the
highest FP8 by 0.35%). Slightly larger than PDL's +0.37% on the plain baseline,
as expected with a shorter step. Against the original baseline (79.2):
FP8 **+8.5%**, FP8 + PDL **+9.1%**. C8 not re-run: C8 all-reduces use NCCL and
PDL cannot engage there (measured neutral).

## Prefill: keep BF16 or convert too?

Prefill-size microbenchmark, summed over all converted layers per chunk
(same shapes and use counts):

| Chunk | BF16 cuBLAS (current) | Marlin FP8 | CUTLASS FP8 W8A8 |
| ---: | ---: | ---: | ---: |
| 512 | 7.21 ms | 7.86 ms (1.09x) | 4.82 ms (0.67x) |
| 2048 | 27.13 ms | 31.47 ms (1.16x) | 16.07 ms (0.59x) |
| 4096 | 54.04 ms | 66.54 ms (1.23x) | 41.56 ms (0.77x) |

Recorded cold prefill is ~11,300 tok/s, so a 4096 chunk is ~362 ms and these
layers are ~15% of it: Marlin-only prefill ~3.5% slower, W8A8 prefill ~3.5%
faster.

| Option | Weights kept | KV capacity | Prefill | Prefill numerics |
| --- | --- | --- | --- | --- |
| A (current) | BF16 + Marlin FP8 | 3.03M (-11%) | unchanged | unchanged (qualification valid) |
| B | Marlin FP8 only | ~3.5M (est., above original 3.41M) | ~3.5% slower | FP8 weights: requalify long context |
| C | Marlin FP8 + CUTLASS FP8 | ~3.3M (est.) | ~3.5% faster | FP8 weights **and** activations: most lossy, requalify |

## FP8 v2 additions and the 256 KiB cutoff (2026-09-23)

A re-profile of FP8 + PDL (`../profile-20260922/traces-c1-fp8pdl/`, 12.11
ms/step) showed the LM head's 260 us BF16 GEMM still firing once per step: the
LM-head wrapper was not a `QuantizeMethodBase`, so the loader never called its
`process_weights_after_loading` and `apply()` fell back to BF16. **The v1
qualification therefore covered the decoder projections only, with a BF16 LM
head.** Fixed (subclass `QuantizeMethodBase`, but not
`UnquantizedEmbeddingMethod`, or `LogitsProcessor` bypasses `apply()`).

Remaining BF16 cuBLAS GEMMs identified by kernel sequence and benchmarked
(`../sweep-20260922/dense-fp8-roofline-round2.json`): the three dense FFN layers
(8192x4096 43.5 -> 24.2 us, 4096x4096 22.8 -> 14.2 us, HBM) convert; shared-expert
gate_up/down (1408x4096, 4096x704) and DSA `fused_qkv_a` are slower or flat
under Marlin and stay BF16.

v2 adds, each behind a flag (default 1): `VLLM_GLM53_FP8_LM_HEAD`,
`VLLM_GLM53_FP8_FFN` (6 projections), `VLLM_GLM53_FP8_MTP` (the MTP draft's DSA
`o_proj`/`q_b_proj`, via `mtp.py`). All three at 0 reproduces v1.

Bracket A / B / C / B / A (`run_v2.py`, `v2.json`, `V2-RESULTS.log`), all with PDL:

| Arm | C1 steps/s | C8 steps/s |
| --- | --- | --- |
| A: v1 | 85.76 / 86.12 | 278.01 / 274.42 |
| B: v2 | 88.16 / 88.25 | 273.20 / 277.16 |
| C: v2 + 256 KiB cutoff | 87.94 | 283.35 |

- v2 vs v1: **+2.64% C1** (repeat spreads 0.42% / 0.10%); C8 -0.37%, inside
  its 1.3-1.4% noise.
- 256 KiB cutoff vs v2: **+2.97% C8** (above both adjacent B arms); C1 -0.30%,
  neutral as expected (C1 messages are 32 KiB).
- Against the original baseline: **C1 88.2 steps/s (+11.4%), C8 283 (+8.1%)**
  for v2 + PDL + cutoff.

KV capacity with v2: 2,983,403 tokens.

Quality: v2's LM head and dense FFN change target numerics (the LM head most
sensitively); MTP draft FP8 cannot change output quality (rejection sampling
preserves the target distribution); the cutoff changes only which all-reduce
algorithm sums the same values. Full candidate qualification:
`../../qualification-20260923-fp8v2/`.
