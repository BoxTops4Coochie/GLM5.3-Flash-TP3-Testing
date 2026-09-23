# PCIe collective fast path — PDL between one-shot all-reduce and mHC, 2026-09-22

## What was missing

The glm53-flash lil profile sets `B12X_PCIE_ONESHOT_PDL=1` and `B12X_MHC_PDL=1`,
the pair qualified at TP4 in b12x's
`docs/evidence/glm53_pcie_mhc_dependent_launch_20260901.md` (+4.46% verifier
steps/s there). In this image neither half applies to the pair that matters:

- **Nothing reads `B12X_PCIE_ONESHOT_PDL`.** This b12x 1.3.0 has no producer
  release (`griddepcontrol.launch_dependents`) in the plain one-shot kernel;
  the TP4 evidence came from branch `b12x-glm53-dflash-pdl-20260901`.
- **The consumer of each all-reduce is `MHCPostPrePartialKernel`, which has no
  `use_pdl` and no `griddepcontrol_wait`.** The PDL pair that does exist is
  `MHCPostPrePartial -> MHCFinalizeGram`.

Trace evidence (`../profile-20260922/pdl_check.py`, baseline C1): 0.0% of
41,860 all-reduce -> mHC-partial pairs overlap; mean launch gap 1.33 us versus
~0.45 us after an all-reduce for other kernels.

Note: `/proc/<pid>/environ` of EngineCore and workers is unusable for checking
this -- setproctitle (`VLLM::Worker_TP0_EP0`) overwrites that region (3,132
"lines" vs 173 for pid 1). Use kernel behaviour in a trace instead.

## Patch (two-sided, 8 functional lines)

`overlay/b12x/...`, diffs `oneshot-pdl.patch`, `mhc-partial-pdl.patch`:

- `comm/pcie/_oneshot_cute.py`: the pull kernel calls
  `griddepcontrol_launch_dependents()` right after its cross-GPU barrier (which
  ends in `sync_threads`), gated by `B12X_PCIE_ONESHOT_PDL`. The flag joins the
  persistent compile key and the key version goes 4 -> 5, so the `/cache` JIT
  cannot serve the pre-patch kernel.
- `norm/mhc/_kernels.py`: `MHCPostPrePartialKernel` launches with
  `use_pdl=_MHC_PDL` and calls `griddepcontrol_wait()` at entry, before any
  global read. All three compile sites of that class get version bumps.

Correctness is by construction: `griddepcontrol_wait` returns only after the
producer grid has completed and its writes are visible, so the early release
can only hide launch latency. After a non-PDL producer (NCCL at C8, prefill
paths) the wait degenerates to ordinary stream order.

Tested by mounting the two files (`compose.pdl.yaml`), no rebuild.

## Results

- **Live:** 98.8% of 42,042 pairs now overlap (`verify_pdl.py`, traces in
  `../profile-20260922/traces-c1-pdl/`).
- **Correct:** 8/8 exact arithmetic in every PDL arm; first-token logprob
  agreement within the control-vs-control noise floor.
- **Speed:**

| Arm | C1 steps/s | C8 steps/s |
| --- | ---: | ---: |
| control-open | *69.25* | 263.30 |
| oneshot-pdl | 79.56 | 263.32 |
| control-close | 79.38 | 263.67 |

  C8 is neutral, as expected: C8's 256 KiB messages take NCCL, not the
  one-shot, so PDL never engages. C1 control-open is an unexplained outlier
  (its C8 cell and every other arm are normal; request logs show only the
  runner's own traffic), so the only fair C1 comparison is against
  control-close: +0.2%.

  **Alternating C1 repeat** (`run_repeat_c1.py`, `repeat-c1.json`,
  `REPEAT-C1.log`), control / PDL / control / PDL / control:

  | | steps/s |
  | --- | --- |
  | controls | 79.67, 79.57, 79.67 (spread 0.13%) |
  | PDL | 79.82, 80.04 |

  **+0.37% mean; both PDL arms above all three controls** (lowest PDL beats
  highest control by 0.19%). Small, consistent, and free at C8.

Reading: PDL works and is worth keeping, but reclaiming ~1 us of launch latency
on 91 pairs per step is bounded near 1% of a 12.6 ms step; it measures ~0.4%.
To adopt it, copy the two overlay files into `kraken-port/src/b12x/...` and
rebuild, or mount them as `compose.pdl.yaml` does.

## The larger collective lever: a TP3 push transport

The exposed cost is latency *inside* each all-reduce (16.1 us at 32 KiB in
production), not the launch gap. TP2's plain push transport
(`tp2_remote_push*` in `_oneshot_cute.py`) is NCCL-LL style: 16 B of payload
travels as 32 B lines carrying two generation tags, written with relaxed
system-scope stores directly into the peer's incoming scratch; the receiver
polls local memory until the tags match. No barrier and no remote reads in
steady state. b12x's TP2 evidence (`docs/evidence/pcie_tp2/...json`) measures
**8.96 us** at 32 KiB versus 16.1 us for our pull.

A three-rank version is feasible (the kernel loops are simple to generalise)
but is the largest piece of work considered here: the kernel ABI passes exactly
one peer's slot pointers, the IPC storage layout needs a per-source incoming
region for each of two peers, the epoch-wrap clear protocol must cover three
ranks, and ~70 references across `pcie_oneshot.py` / `_oneshot_cute.py` plus
vLLM routing assume world size 2. Failure mode is a spin-wait hang. Estimated
prize ~5 us x 100 calls x 72% exposed = **~3% of the C1 step**. Development
needs the GPUs free (service down) for a standalone 3-rank stress harness.

## Run log (first bracket)

```
2026-09-22T23:59:18.504307+00:00 — control-open up | (pdl overlay) | arithmetic 8/8
2026-09-23T00:01:09.821190+00:00 —   control-open C1: [(0, 179.2, 69.25, 2.589, False)]
2026-09-23T00:03:33.060051+00:00 —   control-open C8: [(0, 679.2, 263.3, 2.58, False)]
2026-09-23T00:05:42.561449+00:00 — oneshot-pdl up | (pdl overlay) | arithmetic 8/8
2026-09-23T00:07:31.581430+00:00 —   oneshot-pdl C1: [(0, 203.8, 79.56, 2.561, False)]
2026-09-23T00:09:14.443185+00:00 —   oneshot-pdl C8: [(0, 654.4, 263.32, 2.485, False)]
2026-09-23T00:11:23.992391+00:00 — control-close up | (pdl overlay) | arithmetic 8/8
2026-09-23T00:13:12.952712+00:00 —   control-close C1: [(0, 194.9, 79.38, 2.455, False)]
2026-09-23T00:14:55.754821+00:00 —   control-close C8: [(0, 674.4, 263.67, 2.558, False)]
2026-09-23T00:14:55.755127+00:00 — fidelity control-open vs control-close (noise floor): {'top1_agree': '16/24', 'mean_abs_dlogprob': 0.3248, 'mean_shared_top20': 16.5}
2026-09-23T00:14:55.755270+00:00 — fidelity control-open vs oneshot-pdl: {'top1_agree': '19/24', 'mean_abs_dlogprob': 0.3527, 'mean_shared_top20': 16.4}
2026-09-23T00:14:55.755398+00:00 — fidelity control-close vs oneshot-pdl: {'top1_agree': '18/24', 'mean_abs_dlogprob': 0.3596, 'mean_shared_top20': 16.6}
2026-09-23T00:14:55.755424+00:00 — restoring production compose
2026-09-23T00:17:05.306091+00:00 — production restored | arithmetic 8/8
```
