# Where the decode step actually goes — torch profile, 2026-09-22

Every config-level decode lever is closed and the last two mechanism theories
(attention head padding, 400 W) were wrong, so this profiles the production
configuration instead of guessing again.

Method: `run_profile.py` brings up default/MTP3/DCP1 on the baked image with
`--profiler-config` (the launcher forwards it as an unmanaged native option;
lil's own `--compilation-config` is the only other one it passes), holds
steady-state decode for 25 s, profiles a 6 s window, then restores
`compose.yaml` and verifies 8/8 arithmetic. `analyze.py` aggregates GPU kernel
time; `overlap.py` measures what is actually exposed on the critical path.
Traces and per-kernel caches are retained under `traces-c1/` and `traces-c8/`.

## C1: 13.09 ms/step, 76.4 steps/s

Step count is inferred from once-per-step kernel launches and lands within 3%
of the benchmark's 78.4 steps/s, so these per-step numbers are trustworthy.
The GPU is **94.3% occupied** — this is not a launch-gap or scheduling problem.

| category | us/step | % of step | launches/step |
| --- | ---: | ---: | ---: |
| moe-expert-gemm | 3681 | 28.1 | 90.1 |
| dense-gemm | 2827 | 21.6 | 459.2 |
| skinny-gemm | 1946 | 14.9 | 113.1 |
| collective | 1718 | 13.1 | 107.1 |
| l2-prefetch | 1478 | 11.3 | 135.0 |
| norm | 780 | 6.0 | 271.2 |
| moe-routing | 655 | 5.0 | 138.1 |
| elementwise | 388 | 3.0 | 342.2 |
| kda-recurrent | 369 | 2.8 | 72.0 |
| mla-attention | 329 | 2.5 | 68.1 |
| quant-cast | 190 | 1.5 | 54.0 |
| sample-logits | 86 | 0.7 | 29.1 |

Percentages exceed 100 because kernels on different streams overlap; the union
of all kernel intervals is 12.35 ms of the 13.09 ms step.

MLA attention is 2.5%, independently confirming the head-padding rejection in
[c1-20260922](../c1-20260922/RESULTS.md).

## Exposure matters more than kernel time

`overlap.py` subtracts every other kernel's interval from each category's:

| | kernel time | exposed | hidden |
| --- | ---: | ---: | ---: |
| l2-prefetch | 1478 us/step | **229 us/step** | 84.5% |
| collective | 1607 us/step | **1155 us/step** | 28.1% |

**L2 prefetch is not a lever.** It looks like 11% of the step but runs on its
own stream (tid 4659) inside the idle windows it was written for; removing it
could save at most 1.9% of occupied GPU time, and would give back the dense
projection speedups it buys (its own docstring claims +7% C1). Do not disable
it on the strength of its kernel-time share.

**The collective is the lever.** 100 one-shot all-reduces per step at 16.1 us,
72% of them with nothing else resident: **1.16 ms of a 13.09 ms step, ~9%** —
almost exactly the gap between our 78.4 steps/s and the ~86 that interpolating
TP2's 73 and TP4's 100+ predicts for TP3.

## Why the collective is expensive at world size 3

At 32 KiB per message (4 verify rows x 4096 BF16), 16.1 us is not bandwidth —
the payload moves in 2-3 us on PCIe. It is barrier latency, paid 100 times per
step, and world size 3 has the least support of any size in the stack:

- `b12x.comm.pcie` offers three algorithms. `hierarchical` is world sizes
  (12, 16), `island_rs` is (16,). Only `oneshot` accepts 3, so
  `B12X_PCIE_ALLREDUCE_ALGORITHM` has nothing else to select.
- The remote-push fast paths are written per world size and stop at powers of
  two: `B12X_PCIE_TP2_REMOTE_PUSH`, `B12X_PCIE_TP2_PLAIN_REMOTE_PUSH`,
  `B12X_PCIE_TP4_REMOTE_PUSH`, `B12X_PCIE_TP8_OWNER_REDUCE`. There is no TP3
  equivalent, and `recommended_max_bytes` gives TP2 a 512 KiB capacity bump
  that TP3 does not get.
- The remaining knobs are launch geometry, `B12X_PCIE_ONESHOT_THREADS` (256)
  and `B12X_PCIE_ONESHOT_BLOCK_LIMIT` (8). At 32 KiB these already give one
  pack per thread, so geometry cannot address a latency-bound barrier.

This is the same "world-size-3 support enabled by constants only, untuned"
finding the R27 profile recorded, now quantified.

## C8 falls off the one-shot cutoff into NCCL

The C8 capture mixes in prefill (long generations restarting inside the
window), so its step inference is unreliable and per-step absolutes are not
quoted. The composition is still clear, and one line dominates:

```
4177 us  13.6%   93.0/step   44.9 us each   ncclDevKernel_AllReduce_bf16_RING
```

At C8 a message is 32 rows x 4096 x BF16 = **256 KiB**, above the 84 KiB
`VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE` default, so all-reduce leaves the b12x
one-shot and falls back to the NCCL ring at **44.9 us — 2.8x the one-shot's
16.1 us**. That is the mechanism behind the previously measured but unadopted
"256 KiB cutoff, +2-4% C8 step rate"
([decode-20260920](../decode-20260920/CONFIRMED-RESULTS.md)): it is not a
tuning coincidence, it moves 93 collectives per step off NCCL. It cannot help
C1, where 32 KiB messages already take the one-shot — which is exactly what
those runs found.

## The fusion that targets this is switched off

`vllm/compilation/passes/fusion/allreduce_rms_fusion.py` contains
`B12XAllReduceFusedAddRMSNormPattern`, which fuses all-reduce with the
residual-add RMSNorm that follows it. Two things make it interesting here:

1. It is **not** subject to the FlashInfer world-size gate. `AllReduceFusionPass`
   checks `get_b12x_pcie_allreduce()` first and returns early; only the
   FlashInfer fallback consults `FI_SUPPORTED_WORLD_SIZES = [2, 4, 8, 16]`,
   which excludes 3. TP3 is eligible for the B12X pattern.
2. `pass_config` already has `fuse_allreduce_rms: True`.

But the resolved engine config is `'mode': <CompilationMode.NONE: 0>`, so no
pattern-matcher pass runs at all. lil's profile passes only
`--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}` and never sets
a mode. The profile shows the unfused result directly: 100 all-reduce kernels
plus 180 separate MHC norm kernels per step, no fused op.

**Tested and rejected** in [compile-fusion-20260922](../compile-fusion-20260922/RESULTS.md):
compilation mode 3 costs -4.8% C1 / -4.2% C8 steps/s, and the B12X pattern
still never registered, because `get_b12x_pcie_allreduce()` is None when the
pass is constructed.

## Ranked from here

1. **Adopt the 256 KiB one-shot cutoff for C8.** Already measured twice at
   +2-4%, now with a proven mechanism. Env-only, already plumbed.
2. ~~Compilation mode 3 for the B12X all-reduce/RMSNorm fusion~~ — rejected,
   -4.8% C1.
3. **A TP3 remote-push path in b12x.** Would address the 9% directly and is
   what TP2/TP4/TP8 already have. Kernel work in a closed library; not
   available to us.

## Housekeeping

An interrupted 3-rank collective microbenchmark (`allreduce-sweep.py`,
`ar-probe.py`) left GPUs 1 and 2 spinning at 100% / 125 W while the service
still answered normally; a container recreate cleared it. Those scripts need
the b12x preparation protocol to declare plan shapes before
`custom_all_reduce` will route, which they do not yet do — all 160 cells
returned "not routed to the PCIe runtime". **Do not run them against GPUs held
by the live service.**

Service restored: `glm53-kraken-tp3:baked-20260921`, default/MTP3/DCP1,
health 200, 8/8 arithmetic, GPUs idle at ~81 W.

## Run log

```
profiling build up: default/MTP3/DCP1, baked image, torch profiler enabled
c1: profiled 6.0s at C1
c1: traces {"dp0_pp0_tp0_dcp0_ep0_rank0.1790040960758299316.pt.trace.json.gz": 100260268, "dp0_pp0_tp1_dcp0_ep1_rank1.1790040960731204727.pt.trace.json.gz": 100226574, "dp0_pp0_tp2_dcp0_ep2_rank2.1790040960709817084.pt.trace.json.gz": 100309104}
c8: profiled 6.0s at C8
c8: traces {"dp0_pp0_tp0_dcp0_ep0_rank0.1790041081364819430.pt.trace.json.gz": 46771049, "dp0_pp0_tp1_dcp0_ep1_rank1.1790041081352080614.pt.trace.json.gz": 46758235, "dp0_pp0_tp2_dcp0_ep2_rank2.1790041081354358878.pt.trace.json.gz": 46792602}
restoring production compose
production restored: 8/8 exact arithmetic
```
