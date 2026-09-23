# Full image candidate — quality qualification, 2026-09-23

Candidate: `glm53-kraken-tp3:baked-20260921` + `compose.fp8-dense.yaml` (FP8 v2:
96 decoder projections, LM head, MTP draft DSA x2, all Marlin FP8 W8A16 at
<=32 rows) + `compose.pdl.yaml` + `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=256KB`,
default/MTP3/DCP1. Speed: C1 88.2 steps/s (+11.4%), C8 283 (+8.1%) vs the
original baseline. Protocols identical to the production and FP8 v1 runs.

| Test | Production | FP8 v1 | **Candidate** |
| --- | --- | --- | --- |
| lavd, 30 runs, C8, max | 27 exact / 3 near | 28 / 2 near | **30 exact** |
| hotel-lights, 30 runs, C8, max | 27 exact / 3 truncated | 28 / 1 fail / 1 trunc | **30 exact, no cap hits** |
| Retrieval 128K / 900K | pass / pass | pass / pass | **pass / pass** |
| Long generation @826K, seeds 201-204 | 1 fail in 12, worst repeat 0.0151 | 4/4 clean, worst 0.0044 | **4/4 clean, worst 0.0052** |

Long generation detail:

| Seed | Content chars | Completion tokens | repeat_8gram | Verdict |
| --- | ---: | ---: | ---: | --- |
| 201 | 64,250 | 18,360 | 0.0052 | clean |
| 202 | 49,929 | 21,450 | 0.0006 | clean |
| 203 | 45,265 | 25,453 | 0.0020 | clean |
| 204 | 36,206 | 15,293 | 0.0000 | clean |

Median completion tokens: lavd 17,296 (production 18,426); hotel-lights
42,687 (production 47,350).

## Verdict

No degradation on any test; the candidate is the best-scoring configuration
measured on this harness. The perfect lavd/hotel-lights scores are partly
sampling luck at n=30 and should not be read as FP8 improving accuracy -- the
supported claim is "no detectable harm". Not covered: MTP0, DFlash2, DCP3,
uncensored checkpoint; no same-day production control.

Left serving afterwards per the user's instruction.

## Run log

```
2026-09-23T10:23:49.162169+00:00 — fp8-v2 + pdl + cutoff256 up | FP8 decode active: True | arithmetic 8/8
2026-09-23T10:24:04.412050+00:00 — START lavd
2026-09-23T10:34:07.868807+00:00 — DONE lavd rc=0
2026-09-23T10:34:23.083213+00:00 — START hotel-lights
2026-09-23T11:03:36.720102+00:00 — DONE hotel-lights rc=0
2026-09-23T11:04:42.593815+00:00 — START long retrieval 128K/900K
2026-09-23T11:07:27.755681+00:00 — DONE long retrieval rc=0
2026-09-23T11:12:44.878437+00:00 — {"seed": 201, "content_chars": 64250, "completion": 18360, "finish": null, "repeat_8gram": 0.0052, "ttr": 0.54, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T11:16:24.995129+00:00 — {"seed": 202, "content_chars": 49929, "completion": 21450, "finish": null, "repeat_8gram": 0.0006, "ttr": 0.506, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T11:20:24.804831+00:00 — {"seed": 203, "content_chars": 45265, "completion": 25453, "finish": null, "repeat_8gram": 0.002, "ttr": 0.624, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T11:23:46.463822+00:00 — {"seed": 204, "content_chars": 36206, "completion": 15293, "finish": null, "repeat_8gram": 0.0, "ttr": 0.646, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T11:23:46.463858+00:00 — left serving the candidate (per user instruction); not restoring
```
