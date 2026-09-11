# R34 TP3 port log

Port in progress. R30 optimization work stopped; R30 preserved for rollback.
Parent: localinferencelab/vllm@sha256:d2d13141fc158f3e5f989930c4be4637eaf28322307288724c2faa7cd4e9bcc7.
Upstream instructions saved as upstream-glm-5.3-flash.md; image metadata and source.lock retained.

All 36 changed Python source/test files from the current R30 working tree were
three-way merged against R30 pristine and R34 pristine without textual conflicts.
This includes the tested local explicit-tool-choice-none parser fix, which was
not in the published R30 image. Diagnostic overlays and rejected optimizations
are excluded; the source delta and published-image differences will be audited.

The GDN profile is the sole R30 generated-profile change. Its R30/R34 pristine
component, GPU targets, decode kernel directory and shared math are identical;
carry the qualified R30 component and preserve every other R34 profile component.
No new performance tuning. profile-port.json retains qualification lineage.

Latest R30 launcher settings carry forward: MTP3/.95, clear_thinking default,
configurable batched tokens and graph maximum/list, TP3 cache fingerprint,
pinned checkpoint, GPU-only cache and EP-capable MoE auto override. Native R34
libraries remain inherited. Source and serving qualification pending.

## Built and component-qualified

30-file overlay built, delta 1b1d1c379ef7cb3ff8f47e245633c188a9ab1c7fc98b86bc7d38faa59905b693.
CPU: 42 TP3 geometry/loader, 7 sparse allocator, 120 collective dispatch,
3,009 parser replay tests passed. GPU: 6 boundary restore tests (large offsets)
and 7 TP3 collective eager/graph/torture tests passed. Ten active vLLM native
libraries byte-identical between parent and candidate. Minor copied test lint
issues (imports/formatting) corrected; production source unchanged. R30 stopped
after three idle observations. R34 MTP0 serving qualification started.

## Serving modes

MTP0 and DFlash7 each passed 3/3 exact-answer smokes and default top_p=.95
render checks. Both emitted the required TP3/EP3/B12X/FlashKDA proof under
normal graph serving. GPU KV capacity: MTP0 2,344,230 tokens; DFlash7
1,372,072 tokens. Mode switches followed three idle observations. No new speed
tuning or forced-output benchmark was run. MTP3 extended qualification started.

## MTP3 extended checks passed

MTP3 passed 3 exact-answer requests, 4 concurrent requests, image OCR and
17,642-token retrieval. Normal graphs and runtime proof intact; draft-only
NVFP4 vocabulary copy 113.48 MiB/rank, verifier head BF16. GPU KV capacity
1,596,516 tokens at max model length 1,048,576. Retained-history long-context
qualification started with .95 / min_tokens=0 / ignore_eos=false, seed103,
65,536-token cap. No repetition penalty or token bans. No tuning or speed claims.

## Final long-context checks and handoff

Completed per user instruction; no further optimization or experiments planned.
R34 MTP3/.95, normal EOS, repetition penalty 1, configured GPU limits 300 W:

- Original retained history: 826,266 prompt / 28,883 completion tokens,
  normal stop/DONE, no tool calls, 58,505 final-content characters and 45,927
  reasoning characters. Full-channel screens and distributed manual review
  found no sustained degeneration. Factual accuracy was not audited.
- Exact three-marker retrieval at 127,992 prompt tokens: all three values
  correct, 170 completion tokens, normal stop. The initial strict JSON parser
  rejected Markdown fences. Raw failure is preserved; retrieval-review.json
  records exact values separately from the failed bare-JSON instruction.
- Exact three-marker retrieval at 899,994 prompt tokens: all three values
  correct, 149 completion tokens, normal stop. Same fenced-JSON formatting miss.
  The revised checker only strips a full-response Markdown fence, then checks
  exact JSON key/value equality; no fuzzy matching. Synthetic filler and three
  needle positions are bounded retrieval checks, not broad long-context QA.

Shareable basic post: release/glm-5.3-flash-tp3.md. Companion release/compose.yaml
has no image default; Docker artifact section intentionally blank for the user.
Detailed changes/provenance: PORT-CHANGES.md and PATCH-LEDGER.md. Public Compose
validated using the local image through GLM53_IMAGE; no registry push performed.
Local operational compose.yaml remains configured for the tested local image.
R30 stopped and preserved; R34 MTP3 left running. All test clients finished.


## Decode follow-up completed — 2026-09-10

All three requested investigations complete at 300 W/GPU. Matched C1 0–128K
R30/R34 sweeps do not reproduce meaningful R34 execution slowdown. Correction:
2048-token split target pages were already the R30 default; effective target /
recurrent page geometry and 1,596,516-token KV capacity match. R30's M=4 GEMM2
tactic 57 on R34 gives -0.23% verifier change versus surrounding baselines;
retain the original R34 tactic 56.

Half L2 budgets (10/25/7.5/18 decimal MB) improved verifier speed by +1.80%
and +2.04% in two alternating comparisons, positive at every context. Mean
output changes +1.39% / +2.27%, with stochastic acceptance variation.
27 basic checks and 54 timed cells passed; no new 840K qualification.

Original R34 container restored and idle at the initial running settings
(4096 batched tokens, graph max 256, original cache, default L2 budgets).
R30 preserved/stopped. Original launcher sources and R34 autotune decisions
verified unchanged. Pending on-disk Compose edits retained but not applied.
No production optimization adopted, no new image/push, all GPU limits 300 W.

Details: [investigation report](optimization/decode-20260910/RESULTS.md),
[comparisons](optimization/decode-20260910/COMPARISON.md), and the optional
[half-L2 override](optimization/decode-20260910/l2-half.override.yaml).

## DCP3 and decode follow-up — 2026-09-10 (evening)

DCP=3 (TP3/EP3/DCP3) now starts and serves after a one-kernel fix in
`vllm/v1/attention/ops/dcp.py` (Triton `arange` needs a power-of-two extent;
world size 3 failed the DCP profiling pass). Verified on the candidate: 3/3
smokes, 4 concurrent, OCR, 17.6K and exact 128K/900K three-marker retrieval.
KV capacity 7,488,260 tokens (vs 2,091,238 at DCP1 with the same graph max 32).
Decode: −8% verifier rate at 0–128K, −1% at 512K; crossover ≈1M tokens. DCP3
is a capacity feature, not a speedup, for this host. Production stays DCP1.

Env/patch decode arms (MTP depth 5, Marlin MoE, DCP a2a, sequential or
high-priority KDA gate stream, quarter L2) were all neutral or worse against
adjacent identical-config baselines. A fresh MTP3 profile locates the
remaining time (MoE at HBM roofline, allreduce skew, ~1,750 launches/step);
the one large lever is TP3 routed experts on the B12X fused MoE path, which
is excluded today only by its no-EP guard. Details, receipts and the ranked
list: `optimization/dcp3-20260910/RESULTS.md` and `COMPARISON.md`.
Production container restored (DCP1, 300 W); no image rebuilt or pushed.

## DCP3 adopted as Compose default — 2026-09-10 (late evening)

Built `glm53-r34-tp3:dcp-20260910` (31 overlay files, delta 7447b246…): adds
the `vllm/v1/attention/ops/dcp.py` world-size fix and a TP3 launcher that
accepts `DCP=1|3`. compose.yaml and release/compose.yaml expose `DCP`
(default 3) next to MODE/graph/batch knobs; the release Compose now uses the
same MODE-switch layout as the local one. Production recreated at DCP3:
KV 7,510,219 tokens, runtime proof intact. DCP3/MTP3 passed the 826,266-token
retained-history test (27,592 completion tokens, normal stop, clean content;
reasoning re-quoted one user message in a recap, not a loop).

No-EP3 test (padded TP routed experts, `tp3-routed-experts.patch`, kept out of
the tree): loads and is correct on B12X fused NVFP4 MoE, but −9 to −13%
decode because the shipped B12X profile has no `moe.decode` cases for the
704-wide/288-expert geometry (heuristic policy). `flashinfer_b12x` cannot run
this checkpoint (no swiglu_limit clamp). Not adopted; needs a generated
moe.decode profile for a fair retest.

Release benchmark (llm-inference-bench C1+C8, 0–512K, 350 W/GPU per user
request) running for DCP3 then DCP1; results go to
optimization/dcp3-20260910/release-bench/ and the release post.

Release benchmark done (350 W, C1+C8, 0–512K, DCP3 then DCP1); table in
release/glm-5.3-flash-tp3.md and optimization/dcp3-20260910/release-bench/.
Production restored: glm53-r34-tp3 on dcp-20260910 at DCP3, all GPUs 300 W.
