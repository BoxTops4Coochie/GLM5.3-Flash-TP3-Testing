# L2 prefetch budgets re-sized for FP8 weights — no change adopted, 2026-09-23

With FP8 decode the prefetched projections are half their BF16 size, so the
R34 half-L2 budgets (A/B/C/A_MLA = 10/25/7.5/18 MB) cover more of each weight:
KDA `in_proj` ~75%, KDA `o_proj` 87%, DSA `o_proj` 72% (from the logged plans).
Bracket on the serving stack (FP8 v2 + PDL + 256 KiB cutoff + W8A8 prefill),
`compose.l2.yaml` overriding the budgets, `run.py`, `l2.json`:

| Budgets | C1 steps/s | C8 steps/s |
| --- | --- | --- |
| half 10/25/7.5/18 (current) | 88.25 / 88.10 | 289.13 / 283.86 |
| fit 12/25/17/26 (each FP8 weight fully covered) | 88.24 / 88.79 | 284.42 / 285.27 |
| full 20/50/15/36 (upstream defaults) | 86.49 | 286.63 |

- fit vs half: C1 +0.39% (one fit arm equal to half, one above), C8 -0.58%
  inside a 1.8% control spread. **No clear gain.**
- full: **C1 -1.9%**, C8 flat. Too much fill for the idle windows.

The current half budgets stay. Full coverage does not pay: the remaining
misses are cheaper than the extra fill traffic on the side stream.
