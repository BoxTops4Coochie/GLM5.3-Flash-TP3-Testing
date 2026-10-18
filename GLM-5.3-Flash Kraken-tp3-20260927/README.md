# GLM-5.3 Flash Kraken TP3 — 20260927

Image: **`azallaza/glm53-kraken-tp3:20260927`** (local `glm53-kraken-tp3:20260927`,
`sha256:ed0340333e9fd26e901e60096ab9adac7f5e2452005951246a7bc61f11f90ce2`), derived from `glm53-kraken-tp3:20260925`; includes `lil-bench` and fastokens.
Earlier local builds of this tag are kept as `20260927-pre-dflash-fix`,
`20260927-pre-lmcache`, `20260927-pre-lilbench` and
`20260927-pre-uncensored-moetp`, `20260927-pre-moebackend`, `20260927-pre-fastokens`
(earlier uploads `sha256:697f2787…`, `sha256:4281708c…` and `sha256:823716b9…`;
the newest adds fastokens on top of 823716b9).

```bash
docker pull azallaza/glm53-kraken-tp3:20260927
```

Main compose (`../../compose.yaml`) uses the local tag; the previous compose and
launcher are kept as `compose.pre-20260927.yaml` and
`serve-glm53-flash-tp3-kraken.pre-20260927.py`. The public example is below.

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
| DFlash draft geometry fix (DFlash could not start on 20260925) | always | DFlash works again |
| LMCache L1 at TP3: lil `cache.py` accepts TP3 (contract hash updated), launcher `LMCACHE=l1`, stale-arena cleanup | `LMCACHE=l1` | host-RAM prefix tier |
| `lil-bench` (upstream standardized benchmark, llm-inference-bench v0.7.3, p2pmark) | `docker exec … lil-bench` | tooling only |
| MOE_TP=1 + KV reclaim for the uncensored checkpoint (Marlin TP experts) | `CHECKPOINT=uncensored MOE_TP=1` | KV 3,211,030 (DCP1) / 8,398,097 (DCP3) |
| fastokens 0.3.2 Rust tokenizer (upstream hash-pinned wheel, docker #109) + `max_token_id` fix (vllm #934); token parity identical | `FASTOKENS=1` (sets `VLLM_USE_FASTOKENS`) | **cached-prefix TTFT 396K 0.58 -> 0.21 s, 894K 1.28 -> 0.41 s**; decode unchanged |

`KV_RECLAIM=auto` applies the three memory items **only** for MTP3 / MOE_TP=1
with the default slot/batch/graph/length settings, in the configurations they
were sized and stress-tested in: default checkpoint DCP1 (1,779 blocks; 1,720
with `LMCACHE=l1`) and DCP3 (1,774), uncensored DCP1 (1,721) and DCP3 (1,770);
DCP3 and uncensored are experimental. Any other setting (MOE_TP=0, DFlash, MTP0,
uncensored with L1, different slots...) keeps vLLM's automatic KV sizing and
launches exactly as before; the launcher prints
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
| MTP3 / DCP1 / MOE_TP=1 + LMCache L1 96 GB | 3,180,865 | not re-measured | 8/8, L1 reload verified (below) |
| Uncensored MTP3 / DCP1 / MOE_TP=1 (experimental) | 3,211,030 | not measured | 8/8, retrieval 128K/900K, 3x895K stress |
| Uncensored MTP3 / DCP3 / MOE_TP=1 (experimental) | 8,398,097 (8.01x) | not measured | 8/8, retrieval 128K/900K, 8x895K stress |
| Uncensored MTP3 / DCP1 / EP3 | 3,166,251 | not measured | launch (automatic sizing) |

MTP3/DCP1/MOE_TP=1 is the recommended default: fastest and best-scoring.

```bash
export HF_CACHE_DIR=/path/to/huggingface           # with the example compose below
docker compose up -d                               # defaults above
DCP=3 docker compose up -d                         # MTP3/DCP3/MOE_TP=1 + KV reclaim (experimental)
MODE=dflash2 MOE_TP=0 docker compose up -d         # DFlash2 (EP3 required)
LMCACHE=l1 docker compose up -d                    # + LMCache L1 host-RAM tier
CHECKPOINT=uncensored docker compose up -d         # uncensored, MOE_TP=1 + reclaim (experimental)
CHECKPOINT=uncensored DCP=3 docker compose up -d   # uncensored DCP3, 8.01x 1M (experimental)
KV_RECLAIM=0 docker compose up -d                  # automatic KV sizing
BF16_GEMV=0 docker compose up -d                   # cuBLAS for BF16 decode linears
FASTOKENS=0 docker compose up -d                   # Hugging Face tokenizer instead of fastokens
```

To go back to the previous release, use its own compose and launcher from
`../kraken-20260923/` (image `azallaza/glm53-kraken-tp3:20260923`).

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
  overlays, plus `glm53_tp3.py` with the DFlash draft fix), `lil/` (TP3
  `cache.py` + updated `image-contract.json`, `lil-cache-tp3.patch`),
  `serve-glm53-flash-tp3-kraken.py` (+ `launcher.before.py`),
  `support-receipt.json`, `install_lil_bench.py` + `lil-bench.lock.json`
  (upstream, unchanged), `compose.yaml` (the public example below), build logs.
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

## LMCache L1 (host-RAM prefix tier)

A user report that TP3 works with LMCache L1 is confirmed here. The lil launcher
refused GLM external cache below TP4 except TP2; its `cache.py` now accepts TP3
(only that file's hash changed in `image-contract.json`; the runtime-lock
identity and compile caches are unchanged). `LMCACHE=l1` selects
`--cache-mode lmcache --cache-l1-gib $LMCACHE_L1_GB --no-cache-l2-enabled`;
the disk tier (L2) is not tested at TP3 and is rejected.

- **Reload works:** a 99,728-token prompt, then ~3.2M tokens of other prompts
  to evict it from GPU memory, then the same prompt again: time to first token
  8.5 s cold, 0.24 s after eviction, with all 99,728 tokens loaded from L1 and
  none recomputed (`../../optimization/lmcache-l1-20260927/`).
- **L1 size:** default 96 GB. 64 GB overflowed in that test (the prompt was
  evicted from RAM too). The arena lives in the host's `/dev/shm` (`ipc: host`,
  124 GB here), so `LMCACHE_L1_GB` must fit it.
- **GPU memory:** LMCache's transfer buffers need about 0.8 GB more headroom;
  1,779 blocks with L1 hung in startup autotuning. With `KV_RECLAIM=auto` the
  launcher uses 1,720 blocks for DCP1 + L1: **3,180,865 tokens**, peak
  96,168 MiB/GPU under long prefills, eviction and C8 (1.7 GB free). DCP3 + L1
  uses automatic sizing (untested budget).
- **Restarts:** LMCache does not unlink its arena when the container stops, and
  lil refuses to start over an existing one. The launcher holds a lock beside
  the arena and removes a stale arena at start (verified over two restarts).
  After stopping for good the 96 GB file stays in `/dev/shm`; remove
  `/dev/shm/lmcache_l1_pool_lmcache-glm53-flash-8000-18000` to free the RAM.
- A benign `resource_tracker` `KeyError` traceback for the arena name can appear
  at startup while ranks attach; serving is unaffected.

## lil-bench

The image ships upstream's standardized benchmark, installed by upstream's own
`install_lil_bench.py` and lock (`local-inference-lab/llm-inference-bench`
v0.7.3, commit `05ec1803`, archive checksum verified) at `/opt/lil/bench`, with
p2pmark compiled for this image. It runs inside the serving container against
the running server, records the launch command, hardware and PCIe topology,
runs p2pmark and the prefill (32K/128K) and decode (C1/C8/C16 at 0/64K/128K)
matrix while sampling clocks, power and throttling, then saves to
`/cache/lil-bench` and uploads to docker.local-inference-lab.ai.

```bash
# identifier from https://docker.local-inference-lab.ai/bench/token
docker exec --privileged -it glm53-kraken-tp3 lil-bench --profile quick --no-upload   # setup check
docker exec --privileged -it -e LIL_BENCH_TOKEN=lilb_... glm53-kraken-tp3 lil-bench
```

With 8 slots (`MAX_NUM_SEQS=8`) C16 is skipped automatically. p2pmark sizes
its buffers to free GPU memory (the KV budgets leave 1.5-2.5 GB/GPU). It
refuses to run while the server has other requests (`--allow-busy` overrides).
Run it at the tested host settings (350 W/GPU, +6000 memory offset) with the
GPUs cool.

## Uncensored checkpoint with MOE_TP and KV reclaim (experimental)

`orcarouter/GLM-5.3-Flash-Uncensored-NVFP4` (compressed-tensors
`nvfp4-pack-quantized`) now runs with TP-sharded experts. The padded
2048 -> 2112 expert loading is in the routed-expert weight loader and works for
compressed-tensors unchanged (runtime proof `tp_experts_verified`); the launcher
keeps this checkpoint on the Marlin MoE backend (EP off) instead of FlashInfer
CUTLASS. The `w1_weight_global_scale must match w3_weight_global_scale` warning
comes from the checkpoint and also appears on EP3. Startup briefly runs
out of allocator headroom while Marlin repacks the experts (expandable-segment
warnings); loading completes and serving is unaffected.

| Setting | KV tokens | Peak free (GPU) | Checks |
| --- | ---: | ---: | --- |
| Uncensored DCP1 EP3, automatic | 3,166,251 | — | running before this change |
| Uncensored DCP1 MOE_TP=1, automatic | 2,875,187 | — | 8/8 |
| **Uncensored DCP1 MOE_TP=1 + reclaim (1,721 blocks)** | **3,211,030** | 1.85 GB | 8/8, retrieval 128K/900K, 3 x 895K resident (86.8% KV), 0 preemptions |
| **Uncensored DCP3 MOE_TP=1 + reclaim (1,770 blocks)** | **8,398,097** (8.01x 1M) | 1.39 GB | 8/8, retrieval 128K/900K, 8 x 895K (5 resident), 0 preemptions |

Reclaim frees about 0.8 GB/GPU for this checkpoint. At DCP3, 1,686 blocks
(7,999,543) also passed the same stress with 2.5 GB free; 1,770 was chosen for
8 full 1M contexts, with a slightly smaller margin than the default checkpoint.
Speed and answer-quality qualification are not yet run for the uncensored
MOE_TP configurations. Evidence: `../../optimization/uncensored-moetp-20260927/`,
`../../optimization/uncensored-moetp-dcp3-20260927/`.

## TP-expert MoE backend

`MOE_TP_BACKEND` selects the MoE kernels for TP experts (empty = FlashInfer
CUTLASS for the default checkpoint, Marlin for uncensored). b12x was measured
with real weights on default MTP3/DCP1: C1 -4.5%, C8 -3.4%, prefill -4 to -9%
versus FlashInfer CUTLASS at 32K, despite running cooler; it stays opt-in only
(`../../optimization/b12x-moe-20260928/RESULTS.md`).

## Example Compose

Download [compose.yaml](compose.yaml) and
[serve-glm53-flash-tp3-kraken.py](serve-glm53-flash-tp3-kraken.py) into one
directory. Set `HF_CACHE_DIR` to a populated Hugging Face cache (offline loading
is enabled) and adjust `CPUSET` for your host. Every variation is an
environment variable; stop active inference before switching.

```bash
export HF_CACHE_DIR=/path/to/huggingface
docker compose up -d                                   # MTP3 / DCP1 / MOE_TP=1, KV 3,319,246
LMCACHE=l1 docker compose up -d                        # + LMCache L1 96 GB (KV 3,180,865)
LMCACHE=l1 LMCACHE_L1_GB=64 docker compose up -d       # smaller L1
DCP=3 docker compose up -d                             # MTP3 / DCP3 / MOE_TP=1, KV 8,417,076 (experimental)
DCP=3 MOE_TP=0 docker compose up -d                    # MTP3 / DCP3 / EP3, automatic KV
MODE=dflash2 MOE_TP=0 docker compose up -d             # DFlash2 depth 7 (MXFP8 draft)
MODE=dflash2 MOE_TP=0 DCP=3 docker compose up -d       # DFlash2 at DCP3
MODE=mtp0 MOE_TP=0 docker compose up -d                # no speculation
CHECKPOINT=uncensored docker compose up -d             # orcarouter NVFP4, MOE_TP=1 + reclaim (experimental)
CHECKPOINT=uncensored DCP=3 docker compose up -d       # uncensored DCP3, 8,398,097 KV (experimental)
CHECKPOINT=uncensored MOE_TP=0 docker compose up -d    # uncensored EP3, automatic KV
KV_RECLAIM=0 docker compose up -d                      # automatic KV sizing
BF16_GEMV=0 docker compose up -d                       # cuBLAS for BF16 decode linears
FASTOKENS=0 docker compose up -d                       # Hugging Face tokenizer instead of fastokens
```

| Variable | Default | Values |
| --- | --- | --- |
| `MODE` | `mtp` | `mtp` (depth 3), `dflash2` (depth 7), `mtp0` (off) |
| `DCP` | `1` | `1`, `3` (KV sharded across the 3 GPUs) |
| `MOE_TP` | `1` | `1` TP experts (MTP only; uncensored experimental), `0` EP3 |
| `KV_RECLAIM` | `auto` | `auto`, `0` |
| `KV_CACHE_MEMORY_BYTES` | empty | explicit bytes/GPU for the reclaim configurations |
| `LMCACHE` / `LMCACHE_L1_GB` | `off` / `96` | `off`, `l1` / GiB of host RAM |
| `BF16_GEMV` | `1` | `1`, `0` |
| `FASTOKENS` | `1` | `1` fastokens tokenizer, `0` Hugging Face tokenizer |
| `NATIVE_DCP` | `1` | native PCIe DCP collectives at DCP3 (`0` = off) |
| `CHECKPOINT` | `default` | `default`, `uncensored` (experimental with MOE_TP) |

```yaml
# Kraken TP3 release example, image azallaza/glm53-kraken-tp3:20260927.
# Defaults: default checkpoint / MTP3 / DCP1 / MOE_TP=1, FP8 decode + prefill,
# BF16 GEMV, KV reclaim (3,319,246 tokens), half-L2, top-p 0.95. All source
# changes are built into the image; only the launcher is mounted.
# GPU power is a host setting: tested 350W/GPU; never exceed 400W.
services:
  glm53-kraken-tp3:
    image: "${GLM53_IMAGE:-azallaza/glm53-kraken-tp3:20260927}"
    container_name: glm53-kraken-tp3
    restart: "no"
    # Room for LMCache/vLLM to shut down cleanly. A stale LMCache L1 arena left in
    # /dev/shm is removed by the launcher at the next start (lock-protected).
    stop_grace_period: 90s
    ipc: host
    shm_size: "64gb"
    cpuset: "${CPUSET:-8-47}"
    gpus: all
    ports:
      - "15015:8000"
    volumes:
      - ${HF_CACHE_DIR:?Set HF_CACHE_DIR to your populated Hugging Face cache}:/root/.cache/huggingface:ro
      - runtime-cache:/cache
      - ./serve-glm53-flash-tp3-kraken.py:/usr/local/bin/serve-glm53-flash-tp3-kraken.py:ro
    environment:
      # Set here, in .env, or before `docker compose up -d`.
      # default = released NVFP4; uncensored = orcarouter NVFP4 checkpoint.
      # Both checkpoints are qualified in all four mode/DCP combinations;
      # see release/kraken-20260921/VARIATIONS.md.
      CHECKPOINT: "${CHECKPOINT:-uncensored}"
      MODEL: "${MODEL:-}"
      MODEL_REVISION: "${MODEL_REVISION:-}"
      SERVED_MODEL_NAME: "${SERVED_MODEL_NAME:-GLM-5.3-Flash-TP3}"

      # mtp = depth3; dflash2 (or dflash) = depth7; mtp0 = speculation off.
      # Draft arguments are added only in DFlash mode.
      MODE: "${MODE:-mtp}"
      # 0 = baseline EP3; 1 = TP-sharded routed experts (704 channels/rank),
      # +2.8-4% C1. MTP only: default checkpoint qualified at DCP1; DCP3 and the
      # uncensored checkpoint (Marlin TP experts) experimental; DFlash/MTP0 need 0.
      MOE_TP: "${MOE_TP:-0}"
      # auto = LM-head BF16 release + shared PyNCCL + explicit KV budget, applied
      # only for MTP3/MOE_TP=1 with default slots/batch/graph/length: default
      # checkpoint DCP1 3,319,246 KV (3,180,865 with LMCACHE=l1), DCP3 8,417,076;
      # uncensored DCP1 3,211,030, DCP3 8,398,097 (DCP3/uncensored experimental).
      # Any other setting uses automatic KV sizing. 0 = off everywhere.
      # TP-expert MoE backend (MOE_TP=1): empty = flashinfer_cutlass (default
      # checkpoint) / marlin (uncensored); b12x measured slower, opt-in only.
      MOE_TP_BACKEND: "${MOE_TP_BACKEND:-}"
      KV_RECLAIM: "${KV_RECLAIM:-auto}"
      # off = GPU prefix cache only; l1 = LMCache host-RAM tier (LMCACHE_L1_GB GiB,
      # must fit the host's /dev/shm; ipc: host). Reloads evicted prefixes from RAM
      # (100K prefix: 8.5 s -> 0.24 s) at ~138K fewer GPU KV tokens (3,180,865).
      # The LMCache disk tier (L2) is not tested at TP3 and is rejected.
      LMCACHE: "${LMCACHE:-off}"
      LMCACHE_L1_GB: "${LMCACHE_L1_GB:-96}"
      # Optional explicit KV bytes/GPU for those configurations (empty = tested budget).
      KV_CACHE_MEMORY_BYTES: "${KV_CACHE_MEMORY_BYTES:-}"
      # b12x tensor-core BF16 GEMV for decode BF16 linears <=8 rows (C1): ~+2.6-3%
      # C1 steps/s, C8 unchanged, quality-qualified 2026-09-27. 0 = cuBLAS as before.
      VLLM_B12X_BF16_GEMV: "${BF16_GEMV:-1}"
      VLLM_USE_FASTOKENS: "${FASTOKENS:-1}"
      DFLASH_MODEL: "${DFLASH_MODEL:-local-inference-lab/GLM-5.3-Flash-DFlash2}"
      DFLASH_MODEL_REVISION: "${DFLASH_MODEL_REVISION:-}"

      # DCP1 = measured speed baseline; DCP3 = shard KV across all3 GPUs.
      # DCP3 is qualified; its rank-padding correction is baked into the image.
      DCP: "${DCP:-1}"
      # Native PCIe DCP gather/reduce + fused empty-shard mask; 1 enables, 0 disables.
      # Screened default/MTP3/DCP3, C1: +2.6-3.0% steps/s, no meaningful KV loss.
      # Other modes and long natural-output correctness not yet qualified for this option.
      # Included in this image; larger batches retain existing collectives.
      VLLM_TP3_PCIE_DCP: "${NATIVE_DCP:-1}"
      MAX_NUM_BATCHED_TOKENS: "${MAX_NUM_BATCHED_TOKENS:-4096}"
      # Empty chooses32 for MTP/off or64 for DFlash (covers C8 verification).
      MAX_CUDAGRAPH_CAPTURE_SIZE: "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
      # Empty: finer MTP/off captures for default/DCP1, graph32, 8 slots.
      # Other settings retain their ladder; explicit space-separated lists override.
      CUDAGRAPH_CAPTURE_SIZES: "${CUDAGRAPH_CAPTURE_SIZES:-}"
      MAX_NUM_SEQS: "${MAX_NUM_SEQS:-8}"
      MAX_MODEL_LEN: "${MAX_MODEL_LEN:-1048576}"
      GPU_MEMORY_UTILIZATION: "${GPU_MEMORY_UTILIZATION:-0.95}"
      REASONING_EFFORT: "${REASONING_EFFORT:-max}"
      CLEAR_THINKING: "${CLEAR_THINKING:-true}"

      # 256KB keeps C8's 256 KiB all-reduces on the b12x one-shot instead of
      # the NCCL ring: +3.0% C8 steps/s, neutral at C1 (measured 2026-09-23 on
      # the FP8 stack). Empty restores the upstream 84KiB cutoff.
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: "${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE-256KB}"

      # KDA backends. The Kraken profile ships kda_prefill_backend=b12x, which
      # `resolve_kda_prefill_backend` refuses to select under `auto` because it
      # is not serving-qualified. It degrades the recurrent state built across
      # very long prompts: on an 826K history it cost uncensored/DFlash2 one
      # failure in four, and held uncensored/MTP3 to 38% of the default
      # checkpoint's output length. flashkda -- what R34 served in all 429
      # recorded receipts -- takes both arms to 4/4 at full length.
      # Decode stays on b12x, so this remains a b12x build.
      # See qualification-20260920/kda-prefill-backend/RESULTS.md.
      # The image's profile still ships kda_prefill_backend=b12x and is not
      # patched: /opt/lil/image-contract.json pins the profile hashes and the
      # entrypoint refuses to start if they change. So the qualified backend is
      # selected here. Decode stays b12x; only recurrent prefill moves.
      ADDITIONAL_CONFIG: "${ADDITIONAL_CONFIG:-{\"glm53_kda_decode_backend\":\"b12x\",\"kda_prefill_backend\":\"flashkda\"}}"
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
      VLLM_NO_USAGE_STATS: "1"
      # 1 asserts the resolved backend set matches the qualified one. The
      # baked vllm/v1/worker/utils.py accepts either b12x or flashkda for
      # prefill and requires b12x collectives and b12x KDA decode.
      # Expert proof checks EP3 normally, or actual TP3 partitioning with MOE_TP=1.
      GLM53_TP3_REQUIRE_RUNTIME_PROOF: "${GLM53_TP3_REQUIRE_RUNTIME_PROOF:-1}"

      # MTP acceptance-length adaptation. Empty = disabled, the qualified
      # configuration. A positive integer averages accepted draft lengths over
      # that many verification steps and trims the speculative-token count,
      # with depth 3 as the upper bound. Measured 0.0% at C1, where the
      # controller never trims; under evaluation at C8.
      MTP_ADAPTIVE_WINDOW: "${MTP_ADAPTIVE_WINDOW:-}"
      # FP8 weight-only decode (Marlin W8A16) for the large BF16 projections:
      # KDA in_proj/o_proj, DSA o_proj/q_b_proj, dense FFN, LM head, MTP draft.
      # +11.4% C1 / +8.1% C8 steps/s with PDL and the cutoff. 0 = BF16 as in
      # 20260921. Sub-flags switch parts off individually.
      VLLM_GLM53_FP8_DENSE: "${VLLM_GLM53_FP8_DENSE:-1}"
      VLLM_GLM53_FP8_LM_HEAD: "${VLLM_GLM53_FP8_LM_HEAD:-1}"
      VLLM_GLM53_FP8_FFN: "${VLLM_GLM53_FP8_FFN:-1}"
      VLLM_GLM53_FP8_MTP: "${VLLM_GLM53_FP8_MTP:-1}"
      # w8a8: prefill on CUTLASS FP8, BF16 copies dropped (+5.5-6.5% prefill,
      # +6% KV). bf16: prefill keeps the BF16 weights (more memory, unchanged
      # prefill numerics). Only read when VLLM_GLM53_FP8_DENSE=1.
      VLLM_GLM53_FP8_PREFILL: "${VLLM_GLM53_FP8_PREFILL:-w8a8}"
      # Half-L2 budgets retained from R34; re-tested with FP8 weights and kept.
      VLLM_GLM53_L2_PREFETCH_BUDGET_A_MB: "10"
      VLLM_GLM53_L2_PREFETCH_BUDGET_B_MB: "25"
      VLLM_GLM53_L2_PREFETCH_BUDGET_C_MB: "7.5"
      VLLM_GLM53_L2_PREFETCH_BUDGET_A_MLA_MB: "18"
      VLLM_GLM53_L2_PREFETCH_PERSIST_MB: "${VLLM_GLM53_L2_PREFETCH_PERSIST_MB:-0}"
      VLLM_GLM53_L2_PREFETCH_A_NEXT_MB: "${VLLM_GLM53_L2_PREFETCH_A_NEXT_MB:-0}"
    entrypoint: ["/opt/venv/bin/python", "/usr/local/bin/serve-glm53-flash-tp3-kraken.py"]
    command: []

volumes:
  runtime-cache:
```
