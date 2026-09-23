# CUTLASS FP8 prefill (VLLM_GLM53_FP8_PREFILL=w8a8) — speed and quality, 2026-09-23

On top of the qualified full stack (FP8 v2 decode + PDL + 256 KiB cutoff), the
candidate drops the BF16 copies of the converted layers and runs batches above
32 rows on CUTLASS FP8 with per-token dynamic activation scales. Decode and
prefill read the same FP8 weight values (Marlin-packed and row-major copies).
The LM head keeps BF16 for large batches, so prompt log-probs stay exact.

Unit test (random weights, zero padding rows): W8A8 relative error 3.75%,
cosine 0.9993 at M=64/512/4096, versus 2.7% for the decode path.

## Phase A: bracket current / W8A8 / current

| | KV tokens | 8K | 32K | 64K | 128K | C1 steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current (BF16 prefill) | 2,989,001 | 11,294 | 11,307 | 10,756 | 10,105 | 87.77 |
| **W8A8 prefill** | **3,169,983** | **11,912** | **11,941** | **11,432** | **10,760** | 87.36 |
| current (close) | 2,989,001 | 10,854 | 10,803 | 10,377 | 9,859 | 87.73 |

Prefill tok/s. The controls drifted ~4% apart; against the **better** control
W8A8 is **+5.5% / +5.6% / +6.3% / +6.5%** (8K/32K/64K/128K), +7.9% against
the control mean. Decode unchanged. **KV +6.1%** (2,989,001 -> 3,169,983; still
7% below original production's 3,410,670): this option holds two FP8 copies,
the same bytes as the BF16 it replaces, so it frees only BF16 - FP8 (~2 GiB);
~1 GiB of that did not reach the KV pool (allocator/conversion peak).

Prompt log-likelihood (prefill numerics, teacher-forced):

| Text | tokens | current NLL (two runs) | W8A8 NLL | mean abs dlogprob: current vs current / current vs W8A8 |
| --- | ---: | --- | ---: | --- |
| prose | 21,518 | 1.45488 / 1.45361 | 1.45952 | 0.1005 / 0.1057 |
| code | 14,347 | 0.76752 / 0.77142 | 0.77001 | 0.1143 / 0.1174 |

Identical builds already differ by ~0.10 nats per token (per-restart autotuner
tactics); W8A8 adds 3-5% on top. Code NLL falls between the two controls;
prose is +0.36%, just outside their 0.09% spread. Small but measurable.

## Phase B: prefill-sensitive quality

| Test | Production | Full stack (BF16 prefill) | **W8A8 prefill** |
| --- | --- | --- | --- |
| Retrieval 128K / 900K | pass / pass | pass / pass | **pass / pass** |
| Long generation @826K | 1 fail in 12 | 4/4 clean | **6/6 clean**, worst repeat 0.0092 |
| lavd, 30 runs | 27 exact / 3 near | 30 exact | **29 exact / 1 near** (the same `73, 46.5` near production gets) |

hotel-lights not run: short prompt, long generation, so it is dominated by
decode, which this change does not touch.

| Seed | Content chars | Completion tokens | repeat_8gram | Verdict |
| --- | ---: | ---: | ---: | --- |
| 201 | 54,431 | 14,696 | 0.0025 | clean |
| 202 | 61,441 | 32,648 | 0.0010 | clean |
| 203 | 73,278 | 20,803 | 0.0026 | clean |
| 204 | 65,180 | 32,824 | 0.0004 | clean |
| 205 | 54,715 | 21,292 | 0.0009 | clean |
| 206 | 73,474 | 19,866 | 0.0092 | clean |

## Verdict

Prefill **+5.5-6.5%**, KV **+6%**, decode unchanged, and no degradation on
the tests most sensitive to prefill numerics. Left serving for review.

## Run log

```
2026-09-23T12:01:03.116507+00:00 — cur-open up | {'kv_tokens': 2989001, 'prefill': 'bf16', 'lm_head': True, 'arithmetic': 8}
2026-09-23T12:06:21.597467+00:00 — ERROR: TypeError: dict() got multiple values for keyword argument 'prefill'
2026-09-23T12:08:31.126143+00:00 — restored current stack (bf16 prefill) up | {'kv_tokens': 2989001, 'prefill': 'bf16', 'lm_head': True, 'arithmetic': 8}
2026-09-23T12:18:59.654359+00:00 — cur-open up | {'kv_tokens': 2989001, 'prefill_mode': 'bf16', 'lm_head': True, 'arithmetic': 8}
2026-09-23T12:24:19.868396+00:00 —   cur-open: prefill {'8192': 11294.0, '32768': 11307.0, '65536': 10756.0, '131072': 10105.0} | C1 87.77 steps/s | NLL {'prose': {'tokens': 21518, 'mean_nll': 1.45488}, 'code': {'tokens': 14347, 'mean_nll': 0.76752}}
2026-09-23T12:26:34.389725+00:00 — w8a8 up | {'kv_tokens': 3169983, 'prefill_mode': 'w8a8', 'lm_head': True, 'arithmetic': 8}
2026-09-23T12:31:50.929724+00:00 —   w8a8: prefill {'8192': 11912.0, '32768': 11941.0, '65536': 11432.0, '131072': 10760.0} | C1 87.36 steps/s | NLL {'prose': {'tokens': 21518, 'mean_nll': 1.45952}, 'code': {'tokens': 14347, 'mean_nll': 0.77001}}
2026-09-23T12:34:00.602255+00:00 — cur-close up | {'kv_tokens': 2989001, 'prefill_mode': 'bf16', 'lm_head': True, 'arithmetic': 8}
2026-09-23T12:39:18.527431+00:00 —   cur-close: prefill {'8192': 10854.0, '32768': 10803.0, '65536': 10377.0, '131072': 9859.0} | C1 87.73 steps/s | NLL {'prose': {'tokens': 21518, 'mean_nll': 1.45361}, 'code': {'tokens': 14347, 'mean_nll': 0.77142}}
2026-09-23T12:39:18.527486+00:00 —   prefill 8192: w8a8 11912.0 vs controls [11294.0, 10854.0] -> +7.57% (beats best control: True)
2026-09-23T12:39:18.527519+00:00 —   prefill 32768: w8a8 11941.0 vs controls [11307.0, 10803.0] -> +8.01% (beats best control: True)
2026-09-23T12:39:18.527548+00:00 —   prefill 65536: w8a8 11432.0 vs controls [10756.0, 10377.0] -> +8.19% (beats best control: True)
2026-09-23T12:39:18.527573+00:00 —   prefill 131072: w8a8 10760.0 vs controls [10105.0, 9859.0] -> +7.79% (beats best control: True)
2026-09-23T12:39:18.548330+00:00 —   NLL prose: mean |dlogprob| per token cur-vs-cur 0.10054, cur-vs-w8a8 0.10567; mean NLL 1.45488 / 1.45361 vs 1.45952
2026-09-23T12:39:18.568601+00:00 —   NLL code: mean |dlogprob| per token cur-vs-cur 0.11429, cur-vs-w8a8 0.11738; mean NLL 0.76752 / 0.77142 vs 0.77001
2026-09-23T12:39:18.568636+00:00 — GATE: mean prefill gain +7.89% -> quality phase
2026-09-23T12:41:28.179182+00:00 — w8a8 for quality up | {'kv_tokens': 3169983, 'prefill_mode': 'w8a8', 'lm_head': True, 'arithmetic': 8}
2026-09-23T12:44:11.006300+00:00 — retrieval rc=0 passed=['true', 'true']
2026-09-23T12:47:18.320391+00:00 — {"seed": 201, "content_chars": 54431, "completion": 14696, "repeat_8gram": 0.0025, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T12:51:38.341380+00:00 — {"seed": 202, "content_chars": 61441, "completion": 32648, "repeat_8gram": 0.001, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T12:55:12.377997+00:00 — {"seed": 203, "content_chars": 73278, "completion": 20803, "repeat_8gram": 0.0026, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T12:59:38.340363+00:00 — {"seed": 204, "content_chars": 65180, "completion": 32824, "repeat_8gram": 0.0004, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T13:03:14.544480+00:00 — {"seed": 205, "content_chars": 54715, "completion": 21292, "repeat_8gram": 0.0009, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T13:06:45.373636+00:00 — {"seed": 206, "content_chars": 73474, "completion": 19866, "repeat_8gram": 0.0092, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T13:16:49.486736+00:00 — lavd rc=0
2026-09-23T13:16:49.486779+00:00 — leaving the W8A8-prefill candidate serving for review
```
