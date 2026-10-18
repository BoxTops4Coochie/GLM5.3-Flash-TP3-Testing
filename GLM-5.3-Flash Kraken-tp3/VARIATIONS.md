# GLM-5.3-Flash — Kraken TP3, per-variation qualification

Companion to the main Kraken TP3 guide. Each of the eight served
variations gets its own section: correctness at 0 / 32K / 128K for
decode, standalone cold prefill to 128K, exact retrieval at 128K and
900K, and long-context generation integrity.

Measured on three RTX PRO 6000 Blackwell Workstation GPUs at 350 W each,
TP3/EP3, FP8 KV, 1,048,576-token maximum context, temperature 1,
top_p .95, reasoning effort `max`. Recurrent prefill runs on FlashKDA;
attention, dense projections, routed experts, TP collectives and
recurrent decode all run on B12X.

_Rendered 2026-09-21._

## Summary

All eight variations are qualified. Every one passes the arithmetic
smoke with natural EOS, exact 3-key retrieval at both 128K and 900K,
decode at 0 / 32K / 128K at concurrency 1 and 8, and standalone cold
prefill to 128K.

| Checkpoint | Mode | DCP | Smoke | 128K / 900K retrieval | Long-generation integrity |
| --- | --- | ---: | --- | --- | --- |
| default | MTP depth 3 | 1 | 8/8 exact | exact / exact | 11/12 (+1 tool call) |
| default | MTP depth 3 | 3 | 8/8 exact | exact / exact | 4/4 (+1 tool call) |
| default | DFlash2 depth 7 | 1 | 8/8 exact | exact / exact | 4/4 |
| default | DFlash2 depth 7 | 3 | 8/8 exact | exact / exact | 4/4 |
| uncensored | MTP depth 3 | 1 | 8/8 exact | exact / exact | 4/4 |
| uncensored | MTP depth 3 | 3 | 8/8 exact | exact / exact | 4/4 |
| uncensored | DFlash2 depth 7 | 1 | 8/8 exact | exact / exact | 4/4 |
| uncensored | DFlash2 depth 7 | 3 | 8/8 exact | exact / exact | 4/4 |

The single long-generation failure is default / MTP3 / DCP1 seed 202:
49,194 chars of coherent reasoning (repeat_8gram 0.0042) followed by an
end of turn with no answer. One further response per the table wrote its
answer into a `write` tool call instead of the answer channel, which is a
property of the agentic test prompt rather than of generation integrity;
those are counted separately rather than as failures.

## default (released NVFP4)

### default (released NVFP4) — MTP depth 3 — DCP1

| Check | Result |
| --- | --- |
| Arithmetic smoke, natural EOS | 8/8 exact |
| Exact retrieval at 128K (127,992 prompt tokens) | 3/3 keys exact |
| Exact retrieval at 900K (899,994 prompt tokens) | 3/3 keys exact |
| Long-generation integrity, 826K prompt | 11/12 pass; 40,680–80,308 chars; worst repeat_8gram 0.0074; 1 failed (empty); 1 answered via a tool call rather than the answer channel, scored separately (flashkda) |

**Decode, tokens/s (verifier steps/s, acceptance length)**

| Concurrency | 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| C1 | 190.1 (79.4, 2.39) | 199.2 (79.2, 2.51) | 188.8 (77.9, 2.42) |
| C8 | 655.0 (259.5, 2.52) | 664.0 (261.4, 2.54) | 665.7 (255.8, 2.60) |

**Standalone cold prefill, tokens/s**

| 8K | 32K | 64K | 128K |
| ---: | ---: | ---: | ---: |
| 10,145 | 10,306 | 10,056 | 9,670 |

### default (released NVFP4) — MTP depth 3 — DCP3

| Check | Result |
| --- | --- |
| Arithmetic smoke, natural EOS | 8/8 exact |
| Exact retrieval at 128K (127,992 prompt tokens) | 3/3 keys exact |
| Exact retrieval at 900K (899,994 prompt tokens) | 3/3 keys exact |
| Long-generation integrity, 826K prompt | 4/4 pass; 54,576–72,227 chars; worst repeat_8gram 0.0031; 1 answered via a tool call rather than the answer channel, scored separately (flashkda) |

**Decode, tokens/s (verifier steps/s, acceptance length)**

| Concurrency | 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| C1 | 172.2 (73.1, 2.36) | 189.5 (72.7, 2.61) | 186.6 (72.5, 2.57) |
| C8 | 621.7 (246.3, 2.52) | 629.4 (244.5, 2.57) | 628.2 (243.3, 2.58) |

**Standalone cold prefill, tokens/s**

| 8K | 32K | 64K | 128K |
| ---: | ---: | ---: | ---: |
| 9,725 | 10,100 | 9,944 | 9,690 |

### default (released NVFP4) — DFlash2 depth 7 — DCP1

| Check | Result |
| --- | --- |
| Arithmetic smoke, natural EOS | 8/8 exact |
| Exact retrieval at 128K (127,992 prompt tokens) | 3/3 keys exact |
| Exact retrieval at 900K (899,994 prompt tokens) | 3/3 keys exact |
| Long-generation integrity, 826K prompt | 4/4 pass; 51,727–84,952 chars; worst repeat_8gram 0.0052 (flashkda) |

**Decode, tokens/s (verifier steps/s, acceptance length)**

| Concurrency | 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| C1 | 163.8 (64.1, 2.56) | 159.8 (63.7, 2.51) | 163.0 (62.1, 2.62) |
| C8 | 536.2 (211.8, 2.53) | 538.9 (209.5, 2.57) | 544.8 (200.6, 2.72) |

**Standalone cold prefill, tokens/s**

| 8K | 32K | 64K | 128K |
| ---: | ---: | ---: | ---: |
| 10,300 | 10,518 | 10,308 | 9,919 |

### default (released NVFP4) — DFlash2 depth 7 — DCP3

| Check | Result |
| --- | --- |
| Arithmetic smoke, natural EOS | 8/8 exact |
| Exact retrieval at 128K (127,992 prompt tokens) | 3/3 keys exact |
| Exact retrieval at 900K (899,994 prompt tokens) | 3/3 keys exact |
| Long-generation integrity, 826K prompt | 4/4 pass; 55,331–80,351 chars; worst repeat_8gram 0.0072 (flashkda) |

**Decode, tokens/s (verifier steps/s, acceptance length)**

| Concurrency | 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| C1 | 146.9 (60.6, 2.43) | 145.6 (59.6, 2.44) | 147.9 (59.2, 2.50) |
| C8 | 507.0 (198.5, 2.55) | 512.4 (194.4, 2.64) | 505.4 (193.0, 2.62) |

**Standalone cold prefill, tokens/s**

| 8K | 32K | 64K | 128K |
| ---: | ---: | ---: | ---: |
| 9,876 | 10,267 | 10,136 | 9,881 |

## uncensored (orcarouter NVFP4)

### uncensored (orcarouter NVFP4) — MTP depth 3 — DCP1

| Check | Result |
| --- | --- |
| Arithmetic smoke, natural EOS | 8/8 exact |
| Exact retrieval at 128K (127,992 prompt tokens) | 3/3 keys exact |
| Exact retrieval at 900K (899,994 prompt tokens) | 3/3 keys exact |
| Long-generation integrity, 826K prompt | 4/4 pass; 23,513–30,807 chars; worst repeat_8gram 0.0044 (flashkda) |

**Decode, tokens/s (verifier steps/s, acceptance length)**

| Concurrency | 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| C1 | 200.1 (79.3, 2.52) | 197.6 (78.8, 2.51) | 192.3 (77.3, 2.49) |
| C8 | 657.6 (258.2, 2.55) | 667.5 (258.4, 2.58) | 665.1 (250.1, 2.66) |

**Standalone cold prefill, tokens/s**

| 8K | 32K | 64K | 128K |
| ---: | ---: | ---: | ---: |
| 8,082 | 8,187 | 7,979 | 7,752 |

### uncensored (orcarouter NVFP4) — MTP depth 3 — DCP3

| Check | Result |
| --- | --- |
| Arithmetic smoke, natural EOS | 8/8 exact |
| Exact retrieval at 128K (127,992 prompt tokens) | 3/3 keys exact |
| Exact retrieval at 900K (899,994 prompt tokens) | 3/3 keys exact |
| Long-generation integrity, 826K prompt | 4/4 pass; 29,999–48,291 chars; worst repeat_8gram 0.0151 (flashkda) |

**Decode, tokens/s (verifier steps/s, acceptance length)**

| Concurrency | 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| C1 | 179.5 (74.5, 2.41) | 196.5 (73.9, 2.66) | 184.5 (73.4, 2.51) |
| C8 | 623.5 (241.4, 2.58) | 639.6 (244.0, 2.62) | 623.3 (240.1, 2.60) |

**Standalone cold prefill, tokens/s**

| 8K | 32K | 64K | 128K |
| ---: | ---: | ---: | ---: |
| 7,867 | 8,099 | 7,920 | 7,725 |

### uncensored (orcarouter NVFP4) — DFlash2 depth 7 — DCP1

| Check | Result |
| --- | --- |
| Arithmetic smoke, natural EOS | 8/8 exact |
| Exact retrieval at 128K (127,992 prompt tokens) | 3/3 keys exact |
| Exact retrieval at 900K (899,994 prompt tokens) | 3/3 keys exact |
| Long-generation integrity, 826K prompt | 4/4 pass; 23,501–36,390 chars; worst repeat_8gram 0.0050 (flashkda) |

**Decode, tokens/s (verifier steps/s, acceptance length)**

| Concurrency | 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| C1 | 156.9 (64.6, 2.43) | 160.2 (63.7, 2.52) | 154.2 (62.0, 2.49) |
| C8 | 536.1 (205.9, 2.60) | 535.6 (202.1, 2.65) | 536.0 (196.8, 2.72) |

**Standalone cold prefill, tokens/s**

| 8K | 32K | 64K | 128K |
| ---: | ---: | ---: | ---: |
| 8,084 | 8,286 | 8,092 | 7,904 |

### uncensored (orcarouter NVFP4) — DFlash2 depth 7 — DCP3

| Check | Result |
| --- | --- |
| Arithmetic smoke, natural EOS | 8/8 exact |
| Exact retrieval at 128K (127,992 prompt tokens) | 3/3 keys exact |
| Exact retrieval at 900K (899,994 prompt tokens) | 3/3 keys exact |
| Long-generation integrity, 826K prompt | 4/4 pass; 21,350–39,651 chars; worst repeat_8gram 0.0030 (flashkda) |

**Decode, tokens/s (verifier steps/s, acceptance length)**

| Concurrency | 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| C1 | 150.1 (61.4, 2.44) | 151.7 (60.6, 2.51) | 146.9 (60.2, 2.44) |
| C8 | 515.0 (197.0, 2.61) | 543.0 (193.5, 2.81) | 535.5 (190.1, 2.82) |

**Standalone cold prefill, tokens/s**

| 8K | 32K | 64K | 128K |
| ---: | ---: | ---: | ---: |
| 7,989 | 8,195 | 8,052 | 7,865 |

