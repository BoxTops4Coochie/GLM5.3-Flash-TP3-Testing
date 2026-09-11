# R34 TP3 port — current state (2026-09-10, late)

Local image: **`glm53-r34-tp3:dcp-20260910`** (31 overlay files, overlay identity
`7447b246…`); no publication performed. Production `glm53-r34-tp3` serves MTP3 on
port 15015 with **DCP=3** (Compose default), half-L2 prefetch budgets, graph
maximum 32, top_p .95, 300 W/GPU. `DCP=1 docker compose up -d` restores the
single-copy KV layout and full DCP1 decode speed. R30 container/image/cache are
preserved, stopped.

## What the build contains

All R30 TP3 production source changes (physical TP3 padding/loaders, three-rank
PCIe collectives, target/MTP/DFlash/vision geometry, qualified GDN profile
component, explicit-`tool_choice=none` parser fix) carried onto the pulled R34
base with native libraries unchanged, plus one DCP fix: the DCP log-sum-exp
combine kernel in `vllm/v1/attention/ops/dcp.py` used the DCP world size as a
Triton `arange` extent (power of two required), so world size 3 could not start.
The TP3 launcher accepts `DCP=1|3`. See PORT-CHANGES.md and PATCH-LEDGER.md.

## Qualification

- 3,191 component tests; MTP0, DFlash7 and MTP3 exact-answer smokes; MTP3
  concurrency, vision OCR and 17.6K retrieval (DCP1, `ported-20260910`).
- DCP1: 826,266-token retained history → 28,883 tokens, normal stop, no
  sustained degeneration; exact retrieval at 127,992 and 899,994 prompt tokens.
- DCP3 (`dcp-20260910`): smokes, concurrency, OCR, 17.6K/128K/900K exact
  retrieval; 826,266-token retained history → 27,592 tokens, normal stop, clean
  content channel (reasoning re-quoted a user message in a recap; not a loop).
- Fenced-JSON formatting misses on the retrieval replies are model behavior,
  identical in both modes. These finite checks do not prove all model issues fixed.

## Capacity and speed

| | DCP1 | DCP3 |
| --- | ---: | ---: |
| GPU KV capacity (graph max 32) | 2,091,238 tokens | 7,488,260–7,510,219 tokens |
| C1 verifier steps/s, 300 W, 0 → 512K ctx | 77.6 → 70.2 | 71.6 → 69.4 |
| C1 verifier steps/s, 350 W, 0 → 512K ctx | 79.0 → 73.8 | 72.9 → 71.3 |
| C8 aggregate tok/s, 350 W, 0 / 256K / 512K | 628 / 588 / skipped (KV) | 603 / 578 / 575 |

DCP3 costs ~8% decode below 128K context and ~1–3% at 512K for 3.6× the KV
capacity; only DCP3 serves eight concurrent 512K requests. Full tables:
optimization/dcp3-20260910/COMPARISON.md and release-bench/TABLE.md; the
shareable post release/glm-5.3-flash-tp3.md carries the same numbers.

## Decode optimization record

Measured and rejected or neutral (adjacent A/B/A): R30 MoE tactic transfer,
MTP depth 5, Marlin experts, DCP a2a comm, quarter L2 budgets, sequential or
high-priority KDA gate stream. Adopted: half L2 budgets (+2%). Power: 350 W is
+6–8% (user cap 400 W; normal operation 300 W). Padded TP3 routed experts
(no EP3) load and are correct; with the shipped B12X profile they were 9–13%
slower (heuristic policy); a generated `moe.decode` profile for the 704-wide
geometry is under test (optimization/dcp3-20260910/RESULTS.md).
