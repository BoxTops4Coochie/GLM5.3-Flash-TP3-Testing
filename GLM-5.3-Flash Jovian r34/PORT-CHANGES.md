# R34 TP3 port: detailed changes and provenance

## Source identities

- Base: `localinferencelab/vllm@sha256:d2d13141fc158f3e5f989930c4be4637eaf28322307288724c2faa7cd4e9bcc7`.
- vLLM: `c496604123b1f4441007b952a7ee37ab12c8f6ad`.
- B12X: `59d51a36a942d56a9c36265855cdc7856fa7712e`.
- Source lock SHA256: `e7b5712d12676c8daf0a000398cfa2d57eedb3e290cedf611fcee28fe2413dd0`.
- Local image: `glm53-r34-tp3:ported-20260910`.
- Local image ID: `sha256:d391f61f66e73c5274804d910c3ea586481ab247d910ab469ec0bf2106f0379e`.
- 30 shipped overlay files. `build/source-delta.json` records individual hashes.
- Overlay identity: `1b1d1c379ef7cb3ff8f47e245633c188a9ab1c7fc98b86bc7d38faa59905b693`.
- DCP build (2026-09-10 evening): `glm53-r34-tp3:dcp-20260910`, image ID
  `sha256:34d5ad37b8bb2d9d627ebee946f522beb97aa7af62f2067114e26aea79d655bc`,
  31 overlay files, overlay identity
  `7447b24601a4ad8a866790479b99fb3e95415ef4037111b176fa3ba95e4b155c`. Adds
  `vllm/v1/attention/ops/dcp.py` (masked LSE-combine kernel for non-power-of-two
  DCP world sizes) and the TP3 launcher's `DCP=1|3` gate. Compose default DCP=3.

No registry publication performed. R30 image, source, cache and container retained.
The shareable post intentionally leaves its Docker artifact section blank.

## Changes carried forward

1. **TP3 geometry and weight loading.** Preserve logical checkpoint sizes while
   padding physical attention/recurrent head, projection, vocabulary storage,
   shared-expert and vision dimensions for three ranks. Load the correct logical
   shard, handle quantized storage units, and zero physical tails. Non-TP3
   geometry remains unchanged. Config/linear/parameter/model tests cover these
   behaviors, including invalid/truncated checkpoint rejection.
2. **Target, MTP and DFlash integration.** Carry padded target/MTP projection and
   draft geometry paths, model configuration validation, TP3 vision loading,
   engine/runner integration and runtime proof checks. Preserve R34's draft-only
   NVFP4 MTP vocabulary head and BF16 verifier head. Retain the qualified R30
   incoai BF16 DFlash7 revision rather than changing draft checkpoint during port.
3. **Three-rank PCIe collectives.** Allow world size three in DMA/oneshot support
   with matching dispatch and GPU regression expectations. Preserve kernel math
   and 64-bit addressing. Eager/graph/torture checks run on all three GPUs.
4. **Profile coverage.** Carry TP3 corpus geometry and the previously measured
   attention.gdn component. The R30 and R34 pristine profile components, GPU
   targets, GDN decode sources and shared math match. Preserve every other R34
   profile component. This is profile lineage reuse, not new performance tuning.
5. **Explicit no-tool parsing.** Include the narrow tested local R30 fix that
   honors explicitly supplied `tool_choice="none"` without requiring tool schemas.
   Preserve omitted-choice behavior. It suppresses forbidden API tool events;
   it is not a model-degeneration fix or a reasoning-only-response fix.
6. **Launcher and Compose.** Add R34 TP3 launcher/dispatcher routing, pin the
   target checkpoint, scope supported serving to TP3/EP3 with DCP 1 or 3 and GPU-only cache,
   retain B12X attention/linear/collectives and FlashKDA prefill, explicitly select
   EP-capable MoE `auto`, and preserve temperature 1 / top_p .95 / history defaults.
   Expose batched tokens and maximum graph size plus an optional capture list.
   Default capture list adjusts to the maximum. Use separate R34 cache namespace.
7. **Decode-context parallelism (DCP3).** `vllm/v1/attention/ops/dcp.py` masks
   the LSE-combine Triton kernel so a DCP world size of 3 (not a power of two)
   works; the TP3 launcher gate accepts `DCP=1|3`; Compose exposes `DCP`
   (default 3). Attention KV is sharded across the three ranks (7.5M-token
   capacity vs 2.1M) at about −8% decode speed below 128K context and −1% at
   512K. DCP-group collectives run over PyNCCL. Recurrent (KDA) state is
   replicated, as upstream. Receipts: optimization/dcp3-20260910/.

## Upstream changes preserved

The existing Python files touched by the R30 delta are byte-identical in the
R30 and R34 pristine trees. All 36 source/test changes therefore ported without
textual conflicts. R34 changes in untouched files and its launchers remain.
The 2,048-token split target-page default is inherited and is also present in
the pristine R30 launcher; it is not new in R34. R34's base launcher is retained. Explicit MoE auto is retained for
EP3 rather than silently accepting R34's global B12X MoE default.

No native library is replaced in the overlay. Ten active vLLM native shared
libraries were hashed in base and candidate and match exactly. Other inherited
packages remain in the pinned base image. Source patch: `r34-tp3.patch`; exact
file disposition: `PATCH-LEDGER.md`; source hashes: `source-audit.json`.

## Validation

- 42 TP3 geometry/loader/vision/corpus CPU tests.
- 7 sparse Mamba allocator regressions.
- 120 collective-dispatch CPU tests.
- 3,009 parser replay tests.
- 6 GPU boundary-state restore tests, including large offsets.
- 7 GPU collective eager/graph/torture tests.
- MTP0 and DFlash7: three exact-answer smokes each; .95 default render check.
- MTP3: three exact-answer smokes, four concurrent requests, exact image OCR,
  17,642-token retrieval, and .95 default render check.
- 826,266-token retained history / 28,883-token completion: normal stop,
  nonempty final answer and no sustained degeneration in screens/sampled review.
- DCP3 on `dcp-20260910`: smokes, 4 concurrent, OCR, 17.6K/128K/900K exact
  retrieval; 826,266-token retained history / 27,592-token completion, normal
  stop, clean content channel.
- Exact marker retrieval at 127,992 and 899,994 prompt tokens: 3/3 values each.
  Both used Markdown JSON fences, so bare-JSON formatting compliance failed.
  Raw results and the distinction are recorded in STATUS.md and results/.

All component checks above passed. Copied test-only import/formatting lint issues
were corrected; production delta stayed fixed during serving qualification.
Launcher dry run validates 8192 batched tokens / graph maximum 128. No GPU power
changes (300 W per GPU), tuning sweeps, throughput claims or forced continuation.
Mode switches waited for three idle observations. MTP3 is the final mode.

## Known limits and exclusions

A source port does not establish a cure for R30's retained-history degeneration
or occasional reasoning-only output. Long prose is screened and sampled, not
factually audited. Known-answer retrieval tests check exact values separately.
External cache, DCP=2, strict structured-output concurrency, TP4/TP8 hardware and
new draft checkpoints are not qualified by these TP3 checks. DCP=3 is qualified
by the bounded checks listed in SUMMARY.md (smokes, concurrency, OCR, 128K/900K
retrieval, 826K retained history), not by a repeated full matrix. Rejected R30
optimization experiments and diagnostic mounts are not included. The image's
development-unqualified build label is not a broader production guarantee;
these documents record the bounded local checks on the exact tested image.
