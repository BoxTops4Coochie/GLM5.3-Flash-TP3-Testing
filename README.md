# GLM-5.3 Flash Native TP3 / EP3 Notes

This folder contains the current write-up for the native GLM-5.3 Flash TP3/EP3 experiment on three RTX PRO 6000 Blackwell GPUs.

- [NATIVE-TP3.md](NATIVE-TP3.md) - full native TP3 history, geometry, debug versions, compose examples, findings through v13, and the planned v14 shared-expert / TP-reduction diagnostic.

## Resume point

When work resumes, start at **“Next planned step: v14 shared-expert / TP-reduction diagnostic”** near the end of `NATIVE-TP3.md`.

The current leading hypothesis is a possible reduction-semantic issue involving the replicated shared expert (`2048 % 3 != 0`). This is not yet proven. If that path is correct, the documented fallback is layer-boundary hidden-state fingerprinting to locate the first TP3 numerical divergence.
