# FP8 weight-only decode build — quality qualification, 2026-09-23

Build: `glm53-kraken-tp3:baked-20260921` + `optimization/fp8-dense-20260923/
compose.fp8-dense.yaml` (90 projections + LM head on Marlin FP8 W8A16 for
batches of <=32 rows), default/MTP3/DCP1, no PDL (its mounts were commented out
in `compose.yaml` during this run). Protocols identical to the production kraken
qualification; references are production's results from those protocols.

## Results

| Test | Production reference | FP8 build |
| --- | --- | --- |
| lavd, 30 runs, C8, max | 27 exact / 3 near / 0 fail | **28 exact / 2 near / 0 fail** |
| hotel-lights, 30 runs, C8, max | 27 exact / 3 truncated | **28 exact / 1 fail / 1 truncated** |
| Retrieval 128K | pass | **pass** (3/3 markers exact) |
| Retrieval 900K | pass | **pass** (3/3 markers exact) |
| Long generation @826K, seeds 201-204 | 1 failure in 12 (seed 202, empty answer); worst clean repeat 0.0151 | **4/4 clean**; worst repeat 0.0044 |

Long generation detail:

| Seed | Content chars | Completion tokens | repeat_8gram | Verdict |
| --- | ---: | ---: | ---: | --- |
| 201 | 66,293 | 17,769 | 0.0012 | clean |
| 202 | 49,171 | 13,863 | 0.0044 | clean |
| 203 | 36,179 | 10,617 | 0.0000 | clean |
| 204 | 52,612 | 19,619 | 0.0038 | clean |

All lavd runs finished `stop` with no cap hits (median completion 16,712
tokens vs production 18,426). hotel-lights had one 100K cap hit (production:
three) and **one wrong final answer, `49`** -- production's 30 had none, but
R34's BF16 hotel-lights run had 2 wrong answers, so an occasional miss is not
FP8-specific. Non-exact totals: FP8 2 vs production 3 (hotel-lights), 2 vs 3
(lavd).

## Verdict

No degradation detected; at n=30 per profile the FP8 build is level with
production on every measure. Not covered: MTP0, DFlash2, DCP3 and the
uncensored checkpoint (all use the converted layers), and no same-day
production control was run -- the comparison is against the recorded
production results from the identical harness.

Per the user's instruction the FP8 build was brought back up after the run
and left serving (8/8 arithmetic, 90 projections + LM head converted).

## Run log

```
2026-09-23T01:30:09.152616+00:00 — fp8-dense up | FP8 decode active: True | arithmetic 8/8
2026-09-23T01:30:24.394766+00:00 — START lavd
2026-09-23T01:41:56.684045+00:00 — DONE lavd rc=0
2026-09-23T01:42:11.916567+00:00 — START hotel-lights
2026-09-23T02:16:03.341033+00:00 — DONE hotel-lights rc=0
2026-09-23T02:16:18.552117+00:00 — START long retrieval 128K/900K
2026-09-23T02:19:03.670054+00:00 — DONE long retrieval rc=0
2026-09-23T02:22:38.206572+00:00 — {"seed": 201, "content_chars": 66293, "completion": 17769, "finish": null, "repeat_8gram": 0.0012, "ttr": 0.518, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T02:25:55.775874+00:00 — {"seed": 202, "content_chars": 49171, "completion": 13863, "finish": null, "repeat_8gram": 0.0044, "ttr": 0.549, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T02:28:59.466326+00:00 — {"seed": 203, "content_chars": 36179, "completion": 10617, "finish": null, "repeat_8gram": 0.0, "ttr": 0.528, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T02:32:39.754959+00:00 — {"seed": 204, "content_chars": 52612, "completion": 19619, "finish": null, "repeat_8gram": 0.0038, "ttr": 0.519, "fail": false, "empty": false, "repeat": false, "thin": false}
2026-09-23T02:35:08.264005+00:00 — production restored up | FP8 decode active: False | arithmetic 8/8
```
