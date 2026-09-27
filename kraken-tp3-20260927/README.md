# GLM-5.3 Flash Kraken TP3 — 20260927

Local image: **`glm53-kraken-tp3:20260927`** (`sha256:7a380c432ca8…`, rebuilt with the DFlash fix below; first build kept as `20260927-pre-dflash-fix`), derived
from `glm53-kraken-tp3:20260925`. Not pushed to a registry. Main compose
(`../../compose.yaml`) now defaults to this image; the previous compose and
launcher are kept as `compose.pre-20260927.yaml` and
`serve-glm53-flash-tp3-kraken.pre-20260927.py`.

Defaults: default checkpoint, MTP3, DCP1, **MOE_TP=1**, top-p 0.95, half-L2,
batch 4096, graph 32, 8 slots, max length 1,048,576. Host settings unchanged:
350 W/GPU (never exceed 400 W), +6000 memory offset.

## What changed since 20260925

| Change | Switch | Effect |
| --- | --- | --- |
| Pooled-selection startup-owner cleanup (warmup tensors no longer retained, 481 MiB/GPU live) | always | memory |
| LM-head BF16 release: all logits use the existing FP8 copy; released at the first logits call, after the MTP draft builds its NVFP4 head (403.5 MiB/GPU) | `KV_RECLAIM=auto` | memory |
| Shared TP/EP PyNCCL communicator | `KV_RECLAIM=auto` | memory |
| Explicit KV budget 24,707,662,848 B/GPU (1,779 blocks) | `KV_RECLAIM=auto`, optional `KV_CACHE_MEMORY_BYTES` | **KV 2,951,685 -> 3,319,246** |
| Upstream b12x tensor-core BF16 GEMV (b12x #421) + b12x cuBLAS ("torch") tuning backend + vLLM routing (vllm #873), plus a guard that skips FP8-weight layers | `BF16_GEMV=1` (sets `VLLM_B12X_BF16_GEMV`) | **C1 ~+2.6-3%** |
| Prefill-only step uncapping (vllm #880) — code only | `VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS=0` | none (untested) |

`KV_RECLAIM=auto` applies the three memory items **only** for default
checkpoint / MTP3 / MOE_TP=1 at DCP1 (1,779 blocks) or DCP3 (1,774 blocks,
experimental) with the default slot/batch/graph/length settings — the
configurations they were sized and stress-tested in. Any other setting
(MOE_TP=0, DFlash, MTP0, other checkpoint, different slots...) keeps vLLM's
automatic KV sizing and launches exactly as before; the launcher prints
`GLM-5.3 KV reclaim active|inactive`. `KV_RECLAIM=0` turns it off everywhere.
BF16 GEMV applies to BF16 linears with at most 8 rows in any mode; C8 (32
rows) and FP8 layers are unaffected.

## Mode overview (this image)

| Mode | KV tokens | C1 output tok/s (0/32K/128K) | Quality on this image |
| --- | ---: | --- | --- |
| **MTP3 / DCP1 / MOE_TP=1 (default)** | **3,319,246** | 287.6 / 282.1 / 266.3 | qualified (lavd 30/30, hotel-lights 29+1 trunc, retrieval, long-gen 4/4) |
| MTP3 / DCP3 / MOE_TP=1 (experimental) | 8,417,076 | not measured | 8/8 arithmetic, 8x895K stress |
| MTP3 / DCP3 / EP3 | 8,203,565 | not measured | launch only |
| DFlash2 / DCP1 / EP3 | 2,807,197 | 206.6 / 205.6 / 227.7 | lavd 27+3 near; hotel-lights 83/90 over 3 runs; retrieval, long-gen 4/4 |
| DFlash2 / DCP3 / EP3 | 6,498,779 | 190.2 / 196.2 / 187.8 | 8/8, retrieval pass, hotel-lights 28/1/1 trunc |

MTP3/DCP1/MOE_TP=1 is the recommended default: fastest and best-scoring.

```bash
cd /home/aabduh/glm53-tp3-patch-guide/kraken-port
docker compose up -d                          # defaults above
DCP=3 docker compose up -d                    # MTP3/DCP3/MOE_TP=1 + KV reclaim (experimental)
MODE=dflash2 MOE_TP=0 docker compose up -d    # DFlash2 (EP3 required)
KV_RECLAIM=0 docker compose up -d             # automatic KV sizing
BF16_GEMV=0 docker compose up -d              # cuBLAS for BF16 decode linears
GLM53_IMAGE=glm53-kraken-tp3:20260925 docker compose -f compose.pre-20260927.yaml up -d   # previous
```

## Measured

Speed (user's `llm_decode_bench.py --concurrency 1,8 --contexts 0,32k,128k
--duration 30`, GEMV candidate; 128K runs are power-capped at 1,050 W, so
temperature matters — about 3 steps/s per +9 °C):

| | 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| C1 steps/s | **112.37** (59.8 °C) | **110.34** (65.6 °C) | **106.16** (70.1 °C) |
| C8 steps/s | 314.34 | 308.29 | 298.98 |

At matched temperature, C1 128K is 106.16 vs 103.44 for the pre-GEMV server
(≈+2.6%); C8 is unchanged. Prefill (same run): 12,202 / 11,941 / 12,372 /
10,977 tok/s at 8K/32K/64K/128K. The larger KV pool itself showed no
repeatable speed effect in cooled, matched comparisons (107.26 vs 107.26).

KV capacity and memory (DCP1/MOE_TP1):

| Step | KV tokens |
| --- | ---: |
| 20260925, MOE_TP=1 | 2,951,685 |
| + startup-owner cleanup, explicit budget | 3,003,927 |
| + LM-head release, shared PyNCCL, +45 blocks | 3,087,888 |
| + 34 blocks (3.01x 1M) | 3,151,325 |
| **+ 90 blocks (this image)** | **3,319,246** (3.17x 1M) |

Stress (`optimization/kv-lmhead-nccl-20260926/stress3x1m.py`, pre-GEMV
overlays, same KV): three distinct ~0.9-1.03M-token requests, up to three
resident at 85.2% KV, 0 preemptions, no OOM; peak 96,374 MiB/GPU (1.5 GB
free), reached early in the first long prefill and independent of context
length or residency. **Note:** with GEMV the idle footprint is ~0.4 GB higher
(95.5 vs 95.1 GB/GPU); the 3x1M stress has not been re-run on this exact image.

Quality (`../../qualification-20260927-gemv/`, GEMV candidate, protocols of
the 2026-09-23 FP8 qualification):

| Test | FP8 v2 (09-23) | This stack |
| --- | --- | --- |
| lavd, 30 runs, C8 | 30 exact | **30 exact** |
| hotel-lights, 30 runs, C8 | 30 exact | **29 exact / 1 truncated at 100K, 0 wrong** |
| Retrieval 128K / 900K | pass / pass | **pass / pass** |
| Long generation @826K, 4 seeds | 4/4 clean (worst 0.0052) | **4/4 clean (worst 0.0020)** |

Not covered by this table: MTP0, uncensored checkpoint, image inputs; DFlash2 and DCP3 results are in the sections below.

## Deployment check (this image, main compose)

`GLM-5.3 KV reclaim active`, `GPU KV cache size: 3,319,246 tokens`,
`b12x ready gemm.bf16_gemv: 118/118`, shared PyNCCL on all ranks, 8/8
arithmetic natural EOS, 350 W on every GPU. Receipts: `deploy-server.log`,
`deploy-runtime.json`, `deploy-correctness.json`, `image-inspect.json`.

## Files

- `Dockerfile`, `src/` (13 source files byte-identical to the qualified
  overlays, plus `glm53_tp3.py` with the DFlash draft fix), `serve-glm53-flash-tp3-kraken.py` (+ `launcher.before.py`),
  `support-receipt.json`, `compose.yaml` (copy of main), `build.log`.
- `source.patch`: all source changes versus `glm53-kraken-tp3:20260925`.
- Evidence: `../../optimization/kv-reclaim-20260926/`,
  `../../optimization/kv-layout-20260926/`,
  `../../optimization/kv-lmhead-nccl-20260926/`,
  `../../optimization/upstream-gemv-20260927/`,
  `../../qualification-20260927-gemv/`.

## DCP3 launch checks (2026-09-27, this image, main compose)

| Setting | Launch | KV tokens | Notes |
| --- | --- | ---: | --- |
| `DCP=3 MOE_TP=0` (EP3), native DCP on | OK | 8,203,565 (7.82x 1M) | KV reclaim inactive (automatic sizing), GEMV 99/99, runtime proof EP3 |
| `DCP=3 MOE_TP=1` (TP experts), native DCP on | OK, 8/8 arithmetic | 7,634,202 (7.28x 1M) before reclaim | **experimental**: launcher now allows MOE_TP=1 at DCP3; runtime proof `tp_experts_verified`, EP size 1; see KV reclaim below (8,417,076) |

MOE_TP=1 at DCP3 costs 569,363 KV tokens (-6.9%, same fraction as at DCP1).
Speed and quality at DCP3 not yet measured for either setting.

## KV reclaim at DCP3 (MOE_TP=1, experimental)

`KV_RECLAIM=auto` now also applies to default/MTP3/**DCP3**/MOE_TP=1 with a
1,774-block budget (24,638,220,288 B/GPU). LM-head release + shared PyNCCL
(which at DCP3 also covers the DCP group: two reuses per rank) free ~0.97 GB/GPU
(idle 93.86 -> 92.89 GB). Peak under a 128K prefill, C8 and an 895K prefill at
the automatic budget: 94.16 GB -> +165 blocks keeps 1.5 GB free.

| DCP3 setting | KV tokens | 1M requests |
| --- | ---: | ---: |
| EP3, automatic | 8,203,565 | 7.82x |
| MOE_TP=1, automatic | 7,634,202 | 7.28x |
| **MOE_TP=1 + KV_RECLAIM (1,774 blocks)** | **8,417,076** | **8.03x** |

Stress (`optimization/kv-reclaim-dcp3-20260927/`): 8 distinct ~895K-token
requests x 100K output tokens: all OK, up to 6 resident, max KV 63.5%,
0 preemptions, no OOM/CUDA errors, peak 96,364 MiB/GPU (1,523 MiB free).
8/8 arithmetic afterwards. DCP3 speed/quality not yet measured.

## DFlash fix and launch check

The 20260925 MOE_TP patch pasted its expert-width block into
`apply_glm53_tp3_draft_geometry` too, where `moe_tp_width` is undefined, so
**every DFlash launch failed** on 20260925 and the first 20260927 build
(`NameError: name 'moe_tp_width' is not defined`). DFlash drafts are dense, so
the block is removed from the draft function ([patch](dflash-draft-fix.patch));
the image was rebuilt with only that change.

`MODE=dflash2 DCP=1 MOE_TP=0`: launches, depth 7, KV **2,807,197** (2.68x 1M,
automatic sizing; KV reclaim inactive), runtime proof EP3, BF16 GEMV planned for
target and draft layers, 8/8 arithmetic, draft acceptance 203/350 on the smoke
requests. MOE_TP=1 remains MTP-only (launcher rejects it for DFlash); speed and
quality below.

Second draft: `DFLASH_MODEL=incoai/GLM-5.3-Flash-DFlash2
DFLASH_MODEL_REVISION=dc77ff1c99eeb2df044ee3d4f0094eb033fee410` (BF16, 2.2 GB;
the default local-inference-lab draft is its MXFP8 conversion, 1.2 GB). Same
architecture and TP3 padding. DCP1/EP3: launches, KV **2,753,380** (-53,817 vs
the MXFP8 draft), 8/8 arithmetic.

## DFlash2 speed and quality (MXFP8 draft, DCP1, EP3)

Speed (user's bench): C1 206.6/205.6/227.7 output tok/s
(84.09/83.28/79.49 verify steps/s) at 0/32K/128K;
C8 606.2/641.2/586.3 tok/s. MTP3 remains faster everywhere.

Quality (`../../qualification-20260927-dflash/`): lavd 27 exact / 3 near;
hotel-lights 27 exact / **3 wrong** (all "49", expected 48; natural stop);
retrieval 128K/900K pass; 826K long generation 4/4 clean. Coherent output, but
the hotel-lights misses are the most seen on that profile — repeat before
claiming parity with MTP3. The incoai BF16 draft was only launch-checked.

## DFlash2 at DCP3 (MXFP8 draft, EP3, native DCP)

Launches: KV **6,498,779** (6.20x 1M, automatic; KV reclaim inactive), runtime
proof EP3, 8/8 arithmetic, retrieval 128K/900K pass (natural EOS).

| Conc. | 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| C1 verify steps/s | 79.29 | 78.96 | 77.54 |
| C1 output tok/s | 190.2 | 196.2 | 187.8 |
| C8 verify steps/s | 220.63 | 218.25 | 217.82 |
| C8 output tok/s | 570.9 | 592.2 | 565.2 |

User's bench with `--dcp-size 3`, cooled start (47/57/65 °C; this run was
hotter, 69-76 °C). Versus DFlash DCP1: about -6% C1 / -8% C8 steps/s at 0
context, closer at 128K (-2.5% C1), for 2.3x the KV. Quality at DCP3 not run
(DCP1 qualification above applies to the model path, not DCP3 collectives).

## hotel-lights repeats (seed base 203, 30 runs each, C8/max/100K)

| Run | Exact | Wrong | Truncated | Wrong answers |
| --- | ---: | ---: | ---: | --- |
| DCP1 first run (seed 103) | 27 | 3 | 0 | 49, 49, 49 |
| DCP1 repeat | 28 | 2 | 0 | 49 (6.5K tokens), 47 (40K) |
| DCP3 repeat | 27 (+1 scored wrong but correct: final line "49 - 1 = 48") | 1 | 1 | 49 (8K tokens) |

Across the three DFlash runs: 83 exact of 90 (92%) on DCP1+DCP3, the same
off-by-one "49" as the dominant miss. MTP3 on the same image scored 29/30 and
30/30 in its runs (0 wrong). DFlash looks slightly more error-prone on this
puzzle (~2 wrong per 30 vs ~0 for MTP3); with n=30 per run this is suggestive,
not conclusive. Output is always coherent with natural EOS.

## Bench tool note

`llm_decode_bench.py` v0.6.2 (used for every number here) now prompts
"New version available: v0.7.2 ... Upgrade and restart? [Y/n]" and Enter
accepts. Answer `n` (e.g. `echo n | python3 llm_decode_bench.py ...`) to keep
results comparable. The repeat runner `qualification-20260927-dflash/hotel_repeat.py`
does this automatically.
