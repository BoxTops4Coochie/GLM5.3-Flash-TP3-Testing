# GLM-5.3 Flash TP3 — MTP1–MTP5 Qualification

This file tracks the deterministic MTP qualification sweep for GLM-5.3 Flash on TP3. The planned sweep is **MTP1 through MTP5**, using the same checkpoint, runtime, correctness prompt, two vision images, and measurement method for every level.

## Fixed test baseline

- Model: `local-inference-lab/GLM-5.3-Flash-NVFP4`
- Snapshot: `378ca54585c46542bad1f3cb3ed0d73ae51cdb62`
- Served model: `GLM-5.3-Flash-TP3`
- API: `http://127.0.0.1:15015`
- TP=3, EP enabled
- Max model length: 1,048,576
- KV cache: FP8
- Target attention backend: B12X
- MTP attention backend: B12X
- FlashInfer autotune: **disabled**
- KDA decode backend: Triton
- Vision encoder TP mode: `data`
- Requested CUDA graph mode: `FULL`
- Effective graph mode: `FULL_DECODE_ONLY`
- Max sequences: 8
- Max batched tokens: 8192
- GPU memory utilization: 0.95

Deterministic non-MTP vision-loaded reference decode: approximately **109.97 tok/s**.

## Test matrix

| MTP level | Startup | 5x decode | Arithmetic 20x | Vision 2x | Acceptance | Status |
|---|---:|---:|---:|---:|---:|---|
| MTP1 | PASS | PASS | PASS | PASS | PASS | Qualified |
| MTP2 | Pending | Pending | Pending | Pending | Pending | Pending |
| MTP3 | Pending | Pending | Pending | Pending | Pending | Pending |
| MTP4 | Pending | Pending | Pending | Pending | Pending | Pending |
| MTP5 | Pending | Pending | Pending | Pending | Pending | Pending |

# MTP1

## Startup / runtime findings

MTP1 was launched with:

```text
speculative_config={
  'method': 'mtp',
  'num_speculative_tokens': 1,
  'attention_backend': 'B12X'
}
```

The runtime resolved the draft model as `Glm5NextMTPModel`, using the same LIL snapshot as the target.

### Important points from logs

- FlashInfer autotune remained off:
  ```text
  enable_flashinfer_autotune=False
  Skipping FlashInfer autotune because it is disabled.
  ```
- Target routed-expert MoE backend:
  ```text
  Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
  ```
- EP3 layout:
  ```text
  Local/global number of experts: 96/288
  ```
- MTP routed-expert backend:
  ```text
  Using 'MARLIN' MxFp8 MoE backend.
  ```
- Target weight load: `25.01 s`.
- MTP/draft weight load: `23.42 s`.
- Model runner reported about `63.64 GiB` memory per rank and about `51.26 s` total model loading time.
- Vision stayed enabled through FlashAttention for ViT/MM encoder.
- Encoder cache budget: `32,242` tokens.
- Hybrid cache:
  ```text
  attention block size = 3072
  mamba page padding = 4.21%
  Add 2 padding layers, may waste at most 5.88% KV cache memory
  ```
- `FULL` graph mode was requested, but runtime downgraded to:
  ```text
  FULL_DECODE_ONLY
  ```
- Target and speculator CUDA graph capture both completed.
- Final graph pool was about `0.26 GiB` per GPU.
- Final KV cache:
  ```text
  GPU KV cache size: 1,992,593 tokens
  Maximum concurrency at 1,048,576 tokens/request: 1.90x
  ```
- TP and EP communication both resolved to PyNCCL.
- Existing KDA hot path remained active:
  ```text
  W8A16, N=8598->8600, K=4096
  FP8-E4M3 weights, BF16 activations, Humming backend
  ```

### Multimodal warning

The log states:

```text
Draft model Glm5NextMTP does not support external multimodal embeddings.
Embeddings from the target model will not be passed to the drafter;
using text-only draft inputs instead.
```

This did **not** prevent target-model vision inference; both vision tests completed successfully.

## 1. Five-run decode test

512 completion tokens, deterministic sampling.

| Run | tok/s | Tokens | Decode | TTFT |
|---:|---:|---:|---:|---:|
| 1 | 111.48 | 512 | 4.584 s | 9.878 s |
| 2 | 125.25 | 512 | 4.080 s | 0.116 s |
| 3 | 122.08 | 512 | 4.186 s | 0.107 s |
| 4 | 123.64 | 512 | 4.133 s | 0.113 s |
| 5 | 122.97 | 512 | 4.156 s | 0.106 s |

Summary:

```text
Mean:   121.08 tok/s
Median: 122.97 tok/s
Min:    111.48 tok/s
Max:    125.25 tok/s
```

Run 1 clearly includes first-request/JIT warmup. Runs 2–5 average about **123.49 tok/s**, roughly **12.3% above** the ~109.97 tok/s deterministic non-MTP reference.

## 2. Arithmetic determinism — 20 runs

Prompt:

```text
Calculate 375 * 48 + 625 * 32. Answer only with the number.
```

Expected answer: `38000`

Result:

```text
CORRECT: 20/20
UNIQUE ANSWERS:
20x '38000'
```

Speed:

```text
MEAN:   147.24 tok/s
MEDIAN: 147.31 tok/s
MIN:    146.14 tok/s
MAX:    147.65 tok/s
```

Every request generated 34 completion tokens. TTFT stayed around 0.11–0.12 s.

MTP1 therefore did **not** reintroduce the deterministic arithmetic instability that was eliminated by disabling FlashInfer autotune.

## 3. Vision qualification

Same two files as the deterministic baseline:

```text
images/anime_hill.png
images/retro_anime_portrait.png
```

Prompt:

```text
Analyze this image in detail.
```

### anime_hill.png

```text
Total request:      13.036 s
TTFT:                8.472 s
Post-first-token:    4.564 s
Decode:            111.95 tok/s
Prompt tokens:       1390
Completion tokens:    512
Reasoning tokens:     512
Total tokens:        1902
```

The first vision request shows a large one-time warmup/JIT TTFT. The reasoning correctly described the anime-style person on a grassy hill, dark hair/jacket, rocks, grasses/wildflowers, and large sky.

### retro_anime_portrait.png

```text
Total request:       4.957 s
TTFT:                0.236 s
Post-first-token:    4.721 s
Decode:            108.25 tok/s
Prompt tokens:       1390
Completion tokens:    512
Reasoning tokens:     512
Total tokens:        1902
```

The reasoning correctly identified the central male figure, thobe/dishdasha, ghutra/keffiyeh, black agal, green belt, curved sword, floral framing, purple background, and retro-anime style.

Raw two-image mean decode: **110.10 tok/s**.

Both requests consumed the entire 512-token budget in reasoning, so neither emitted final-content text.

## 4. MTP acceptance

Prometheus counters:

```text
vllm:spec_decode_num_drafts_total                   2672
vllm:spec_decode_num_draft_tokens_total             2672
vllm:spec_decode_num_accepted_tokens_total           1568
vllm:spec_decode_num_accepted_tokens_per_pos_total   1568
```

The first helper script incorrectly printed `117.37%` because its regex summed both the global accepted-token counter and the per-position accepted-token counter.

Those are not independent totals. For MTP1, position 0 is the only speculative position, so the positional counter equals the global accepted-token count.

Correct calculation:

```text
accepted = 1568
drafted  = 2672

1568 / 2672 = 58.68%
```

**Correct MTP1 acceptance: 58.68%**

For MTP2–MTP5, use:

```text
spec_decode_num_accepted_tokens_total
/
spec_decode_num_draft_tokens_total
```

Do not add `accepted_tokens_per_pos_total` into the global accepted total. The per-position counters should only be used to analyze how acceptance falls off by speculative position.

## MTP1 summary

| Metric | Result |
|---|---:|
| Startup | PASS |
| LIL checkpoint | PASS |
| TP3 / EP3 | PASS |
| MTP attention | B12X |
| Target NVFP4 MoE | FlashInfer CUTLASS |
| MTP MXFP8 MoE | MARLIN |
| FlashInfer autotune | OFF |
| Effective graph mode | FULL_DECODE_ONLY |
| Model memory | ~63.64 GiB/rank |
| Graph pool | ~0.26 GiB/rank |
| KV cache | 1,992,593 tokens |
| 1M concurrency | 1.90x |
| 5-run decode mean | 121.08 tok/s |
| Warm decode mean | 123.49 tok/s |
| Arithmetic | 20/20 |
| Arithmetic mean | 147.24 tok/s |
| anime_hill | 111.95 tok/s |
| retro_anime_portrait | 108.25 tok/s |
| Vision mean | 110.10 tok/s |
| Acceptance | **58.68%** |
| Status | **QUALIFIED** |

# MTP2

Status: **FAIL — DETERMINISTIC ARITHMETIC CORRECTNESS REGRESSION**

## Startup / runtime findings

MTP2 launched successfully with:

```text
speculative_config:
  method: mtp
  num_speculative_tokens: 2
  attention_backend: B12X
```

Relevant startup observations:

```text
Model loading:              ~63.27 GiB / GPU
Draft MoE backend:          MARLIN (MXFP8)
Target MoE backend:         FLASHINFER_CUTLASS
FlashInfer autotune:        disabled
```

The MTP draft emitted the expected nonfatal multimodal warning that external target-model image embeddings are not passed to the drafter.

## 1. Five-run decode test

```text
RUN 1: 107.49 tok/s
RUN 2: 123.68 tok/s
RUN 3: 117.71 tok/s
RUN 4: 113.90 tok/s
RUN 5: 117.31 tok/s

MEAN:   116.02 tok/s
MEDIAN: 117.31 tok/s
MIN:    107.49 tok/s
MAX:    123.68 tok/s

WARM RUNS 2-5:
MEAN:   118.15 tok/s
MEDIAN: 117.51 tok/s
MIN:    113.90 tok/s
MAX:    123.68 tok/s
```

Compared with the deterministic no-MTP vision-loaded text reference of ~109.97 tok/s, the MTP2 warm mean is about **7.4% faster**.

## 2. Arithmetic correctness

### Original 256-token run

The canonical arithmetic test at `max_tokens=256` produced the same wrong final answer on all 20 runs:

```text
20/20 -> '38375'
completion_tokens = 129 on every run
reasoning_tokens  = 124 on every run
```

Because every request stopped naturally at only 129 completion tokens, this was **not** a max-token truncation event.

### 512-token confirmation run

MTP2 was relaunched and the exact same arithmetic prompt was tested again with:

```text
max_tokens = 512
```

Result:

```text
CORRECT FINAL ANSWERS: 0/20

UNIQUE FINAL ANSWERS:
20x '38375'

REASONING TOKEN COUNTS:
20x 124

MEAN SPEED:   157.52 tok/s
MEDIAN SPEED: 160.73 tok/s
MIN SPEED:     96.67 tok/s
MAX SPEED:    161.16 tok/s
```

Every run again terminated at:

```text
completion_tokens = 129
reasoning_tokens  = 124
final answer      = 38375
```

Run 1 was a first-request warmup outlier at 96.67 tok/s; runs 2–20 were tightly clustered around ~160–161 tok/s.

This conclusively rules out the earlier hypothesis that MTP2's wrong result was caused by the 256-token output limit. **MTP2 has a deterministic arithmetic correctness regression in this runtime/checkpoint combination.**

## 3. Vision qualification

```text
anime_hill.png:          110.33 tok/s
retro_anime_portrait:    108.77 tok/s
Two-image mean:          109.55 tok/s
```

Both images were interpreted correctly.

## 4. MTP acceptance

```text
draft events:       3285
draft tokens:       6570
accepted tokens:    2855
global acceptance:  43.46%

position 0:         1830
position 1:         1025
```

The per-position sum equals the global accepted-token counter.

## MTP2 summary

| Metric | Result |
|---|---:|
| Decode mean | 116.02 tok/s |
| Warm decode mean | **118.15 tok/s** |
| Arithmetic @ 256 | **0/20 — deterministic `38375`** |
| Arithmetic @ 512 | **0/20 — deterministic `38375`** |
| Arithmetic @ 512 mean | 157.52 tok/s |
| Vision image 1 | 110.33 tok/s |
| Vision image 2 | 108.77 tok/s |
| Vision mean | 109.55 tok/s |
| Acceptance | **43.46%** |
| Status | **FAIL — correctness** |

# MTP3

Status: **QUALIFIED**

## Startup / runtime findings

```text
Model loading:              63.27 GiB / GPU
CUDA graph pool:             0.49 GiB actual
GPU KV cache size:       2,608,222 tokens
Maximum 1M concurrency:      2.49x
```

## Five-run decode

```text
116.54, 116.01, 114.57, 113.83, 111.15 tok/s

MEAN:       114.42 tok/s
WARM MEAN:  113.89 tok/s
```

## Arithmetic

```text
20/20 PASS
answer: 38000
MEAN SPEED: 181.80 tok/s
MEDIAN:     181.82 tok/s
MIN:        180.95 tok/s
MAX:        182.67 tok/s
```

## Vision

```text
anime_hill.png:          111.78 tok/s
retro_anime_portrait:    104.94 tok/s
Two-image mean:          108.36 tok/s
```

Both image descriptions remained semantically correct.

## Acceptance

```text
draft events:      3322
draft tokens:      9966
accepted tokens:   3247
acceptance:        32.58%

position 0: 1802
position 1:  960
position 2:  485
```

## MTP3 summary

| Metric | Result |
|---|---:|
| Decode mean | 114.42 tok/s |
| Warm decode mean | 113.89 tok/s |
| Arithmetic | **20/20** |
| Arithmetic mean | **181.80 tok/s** |
| Vision mean | 108.36 tok/s |
| Acceptance | **32.58%** |
| KV cache tokens | 2,608,222 |
| Status | **QUALIFIED** |

# MTP4

Status: **QUALIFIED WITH OUTPUT-BUDGET SENSITIVITY**

## Startup / runtime findings

```text
Model loading:              63.27 GiB / GPU
CUDA graph pool:             0.67 GiB actual
GPU KV cache size:       2,554,447 tokens
Maximum 1M concurrency:      2.44x
```

## Decode

Two five-run passes were recorded.

Pass 1:

```text
MEAN:       114.61 tok/s
WARM MEAN:  114.16 tok/s
```

Pass 2:

```text
MEAN:       113.34 tok/s
WARM MEAN:  113.99 tok/s
```

The two passes are consistent at roughly **114 tok/s steady-state**.

## Arithmetic

### 256-token run

```text
CORRECT: 4/20
16x final answer ''
 4x final answer '38000'

MEAN SPEED: 176.61 tok/s
```

All requests reached `completion_tokens=256`. Most consumed the entire generation budget in reasoning and emitted no final answer.

### 512-token retest

The same arithmetic prompt was rerun with only the output budget increased to `max_tokens=512`.

Result:

```text
CORRECT FINAL ANSWERS: 20/20
UNIQUE FINAL ANSWERS:
20x '38000'

MEAN SPEED:   177.83 tok/s
MEDIAN SPEED: 179.77 tok/s
MIN SPEED:    138.51 tok/s
MAX SPEED:    182.73 tok/s
```

Run 1 was a slower first-request/warmup outlier:

```text
RUN 01: 138.51 tok/s
completion tokens: 335
reasoning tokens:  331
```

Runs 2–20 were mostly ~176.7–182.7 tok/s.

Reasoning-token counts across the 20 correct runs:

```text
 4x 253
 3x 296
 1x 303
 1x 311
 9x 316
 1x 329
 1x 331
```

This confirms that MTP4's initial 256-token failures were **generation-budget exhaustion**, not a demonstrated arithmetic correctness failure. At 512 tokens it produced the canonical `38000` on all 20 runs.

## Vision

Two vision passes were recorded.

Pass 1:

```text
anime_hill.png:          107.05 tok/s
retro_anime_portrait:     98.42 tok/s
mean:                    102.74 tok/s
```

Pass 2:

```text
anime_hill.png:          101.69 tok/s
retro_anime_portrait:     97.38 tok/s
mean:                     99.54 tok/s
```

Vision remained semantically correct, but decode throughput was materially below MTP1–MTP3.

## Acceptance

```text
draft events:      5937
draft tokens:     23748
accepted tokens:   7003
acceptance:        29.49%

position 0: 3392
position 1: 1878
position 2: 1143
position 3:  590
```

## MTP4 summary

| Metric | Result |
|---|---:|
| Decode mean | ~114 tok/s |
| Arithmetic @ 256 | 4/20 final answers; 16/20 exhausted budget |
| Arithmetic @ 512 | **20/20 — `38000`** |
| Arithmetic @ 512 mean | 177.83 tok/s |
| Vision mean | 99.54–102.74 tok/s |
| Acceptance | **29.49%** |
| KV cache tokens | 2,554,447 |
| Status | **QUALIFIED WITH BUDGET CAVEAT** |

# MTP5

Status: **QUALIFIED, BUT PERFORMANCE REGRESSION**

## Startup / runtime findings

```text
Model loading:              63.27 GiB / GPU
CUDA graph pool:             0.85 GiB actual
GPU KV cache size:       2,476,755 tokens
Maximum 1M concurrency:      2.36x
Attention page size:         3328 tokens
Mamba page padding:          8.05%
```

MTP5 has the largest graph pool and lowest KV capacity of the tested MTP levels.

## Five-run decode

```text
96.32, 95.98, 99.60, 95.73, 97.02 tok/s

MEAN:       96.93 tok/s
WARM MEAN:  97.08 tok/s
```

This is slower than the deterministic no-MTP reference (~109.97 tok/s).

## Arithmetic

```text
CORRECT FINAL ANSWERS: 20/20
MEAN SPEED:   179.04 tok/s
MEDIAN SPEED: 178.97 tok/s
MIN SPEED:    177.61 tok/s
MAX SPEED:    180.47 tok/s

completion tokens: 163
reasoning tokens:  159
```

## Vision

```text
anime_hill.png:           98.68 tok/s
retro_anime_portrait:     89.42 tok/s
Two-image mean:           94.05 tok/s
```

Both images remained semantically correct, but vision decode performance regressed substantially.

## Acceptance

```text
draft events:      3343
draft tokens:     16715
accepted tokens:   4043
acceptance:        24.19%
draft tokens/event: 5.000000

position 0: 1881
position 1: 1029   survival from previous 54.70%
position 2:  586   survival from previous 56.95%
position 3:  351   survival from previous 59.90%
position 4:  196   survival from previous 55.84%
```

## MTP5 summary

| Metric | Result |
|---|---:|
| Decode mean | 96.93 tok/s |
| Warm decode mean | 97.08 tok/s |
| Arithmetic | **20/20** |
| Arithmetic mean | 179.04 tok/s |
| Vision mean | 94.05 tok/s |
| Acceptance | **24.19%** |
| KV cache tokens | 2,476,755 |
| Status | **QUALIFIED, NOT PERFORMANCE-COMPETITIVE** |

# Current comparison

| Metric | No MTP | MTP1 | MTP2 | MTP3 | MTP4 | MTP5 |
|---|---:|---:|---:|---:|---:|---:|
| Decode mean | ~109.97 | 121.08 | 116.02 | 114.42 | ~114 | 96.93 |
| Warm decode mean | ~109.97 | **123.49** | **118.15** | 113.89 | ~114 | 97.08 |
| Arithmetic @512 | 20/20 | 20/20 | **0/20** | **20/20** | **20/20** | **20/20** |
| Arithmetic result | `38000` | `38000` | **`38375`** | `38000` | `38000` | `38000` |
| Arithmetic tok/s | — | 147.24 | 157.52 | **181.80** | 177.83 | 179.04 |
| Vision image 1 | 109.89 | 111.95 | 110.33 | 111.78 | 101.69–107.05 | 98.68 |
| Vision image 2 | 109.62 | 108.25 | 108.77 | 104.94 | 97.38–98.42 | 89.42 |
| Vision mean | 109.76 | **110.10** | 109.55 | 108.36 | 99.54–102.74 | 94.05 |
| Acceptance | N/A | **58.68%** | 43.46% | 32.58% | 29.49% | 24.19% |
| KV cache tokens | ~1.76M* | 1.993M* | 2.648M | 2.608M | 2.554M | 2.477M |
| Current status | Reference | **Qualified / best overall** | **FAIL correctness** | Qualified | Qualified / budget caveat | Qualified / slow |

`*` Hybrid-cache packing/layout differs between runs, so raw KV token counts should not be treated as a simple one-variable memory comparison.

## Sweep interpretation

At the end of the MTP1–MTP5 sweep:

- **MTP1 remains the strongest general-purpose choice**: highest steady-state long-generation decode throughput, highest acceptance, and essentially no vision penalty.
- **MTP2 is disqualified for this exact runtime/checkpoint** by a reproducible deterministic arithmetic error (`38375`) at both 256 and 512 max-token budgets.
- **MTP3 is correctness-clean** and produces the highest arithmetic microbenchmark throughput, but its long-form decode gain is much smaller than MTP1.
- **MTP4 is correctness-clean at 512 tokens**, but needs a larger reasoning budget on the arithmetic test and gives poorer vision throughput.
- **MTP5 is correctness-clean but performance-negative** for long-form text and vision because acceptance has fallen far enough that repeated draft forwards cost more than they save.

The acceptance curve falls monotonically:

```text
MTP1: 58.68%
MTP2: 43.46%
MTP3: 32.58%
MTP4: 29.49%
MTP5: 24.19%
```

That trend matches the observed crossover: deeper reuse of the single MTP layer continues to help some short/highly predictable sequences, but becomes progressively less effective for long-form and multimodal generation.

# Qualification policy

A level is qualified only if it reaches API-ready state, keeps the canonical arithmetic test at 20/20, preserves correct vision behavior, has no output degeneration, reports valid speculative acceptance, and has its steady-state decode speed plus memory/KV-cache cost recorded.

After MTP1–MTP5 are complete, choose based on the combined tradeoff among **steady-state decode throughput, acceptance, correctness, vision behavior, and memory cost**, not the highest short-run number alone.
