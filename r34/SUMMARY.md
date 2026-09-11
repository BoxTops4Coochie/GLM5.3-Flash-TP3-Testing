# R34 TP3 port complete — bounded local checks finished

Local image: `glm53-r34-tp3:ported-20260910`; no publication performed.
R34 MTP3 is running on port15015 with default top_p=.95 and 300 W GPU limits.
R30 container/image/cache preserved, stopped. No further optimization running.

All R30 TP3 production source changes and the tested local explicit-none parser
fix were ported; qualified GDN component preserved against identical source
contracts. Latest batched-token/graph configuration controls included. Native
R34 libraries inherited unchanged. See PORT-CHANGES.md and PATCH-LEDGER.md.

3,191 component tests passed. MTP0, DFlash7 and MTP3 exact-answer smokes passed.
MTP3 concurrent requests, vision OCR and 17.6K retrieval passed. Retained-history
826,266-token prompt generated 28,883 tokens and stopped normally with no
sustained degeneration found. Retrieval at 127,992 and 899,994 prompt tokens
recovered all three known keys; both replies had Markdown JSON fences rather
than bare JSON. These finite tests do not prove all model/channel issues fixed.

Shareable post: release/glm-5.3-flash-tp3.md (Docker section intentionally blank).
Companion Compose: release/compose.yaml (image unset). Local operations and
rollback: RUN.md. Exact receipts, raw output and logs: results/.


Separate decode follow-up (2026-09-10): matched R30/R34 execution effectively
unchanged; no benefit from R30 MoE tactic transfer. Half L2 budgets measured
+1.80% / +2.04% verifier speed in two comparisons. Original runtime restored,
all GPUs 300 W; optional override saved, not enabled. See
[decode report](optimization/decode-20260910/RESULTS.md). The port image is unchanged.

DCP3 follow-up (2026-09-10 evening): DCP=3 works after a dcp.py world-size fix;
correct through 900K retrieval; 7.49M-token KV capacity; −8% decode at short
context, −1% at 512K. Six env/patch decode arms neutral or worse. See
[DCP3 report](optimization/dcp3-20260910/RESULTS.md). Production unchanged (DCP1).

Late evening: DCP3 is now the Compose default on image `glm53-r34-tp3:dcp-20260910`
(7.51M-token KV; `DCP=1` restores full speed). 826K retained-history passed under
DCP3. Padded TP routed experts (no EP3) load and are correct but run 9–13% slower
on B12X's heuristic MoE policy; not adopted. Release C1/C8 benchmark in progress.
