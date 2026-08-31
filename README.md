# GLM-5.3 Flash Native TP3 / EP3 Notes

This folder contains the current write-up for the native GLM-5.3 Flash TP3/EP3 experiment on three RTX PRO 6000 Blackwell GPUs.

- [NATIVE-TP3.md](NATIVE-TP3.md) - full native TP3 history, geometry, debug versions, compose examples, findings through v16, including the v14 shared-expert TP-reduction fix, v15 parser isolation, and the v16 default-template validation.
- [V17.md](V17.md) - current confirmed correctness/performance baseline: official checkpoint template, `glm45` reasoning parser, 1M-context compose, KV pool, CUDA-graph behavior, FlashInfer autotuning, decode benchmarks, and Estonia / Estonia-long results.
- [VTP1-5.md](VTP1-5.md) - earlier generic/Transformers-based TP3 padding experiments. Work stopped after VTP5 because the generic path lacked the desired native EP3/Glm5Next behavior and ModelOpt metadata handling became increasingly fragile, so development moved to the native ModelOpt NVFP4 path.

## Prebuilt images

All native TP3 debug builds through v15 plus the current v17 baseline are published as public Docker Hub images, so reproducing a version does not require building from source.

Example:

```bash
docker pull azallaza/glm53-tp3-testing:v17-official-template-parser
```

Published tags documented here:

```text
v1 … v7
v8-debug
v9-kda-debug
v10-kda-debug
v11-kda-debug
v12-mla-debug
v13-indexer-debug
v14-shared-expert-fix
v15-parser-clean
v17-official-template-parser
```

v16 intentionally has no image because it was a configuration/template-only validation performed with the v15 image; there was no new source/container build to preserve.

Each published tag maps 1:1 to the corresponding `voipmonitor/vllm:glm53-native-tp3-bf16-*` build referenced in the version notes. No login required. Images expect the `LibertAIDAI/GLM-5.3-Flash-NVFP4` checkpoint plus 3× SM120 GPUs.

## Current baseline

**v17 is the current confirmed correctness/performance baseline.**

The major correctness failure present through v13 was fixed in v14. The replicated shared expert was being included in a generic TP SUM, multiplying its contribution by the TP world size. Scaling only that replicated shared-expert contribution by `1 / tp_size` before the SUM restored the intended math.

v15 and v16 then isolated parser/template behavior. v17 returned to the official checkpoint template and restored the `glm45` reasoning parser while keeping the v14 numerical fix.

Current v17 headline results:

```text
TP / EP:               3 / 3
Context:               1,048,576
KV Pool                4,134,385
Vision:                disabled
MTP / DFlash:          disabled
KV dtype:              FP8
KDA decode:            Triton
MoE backend:           auto -> FLASHINFER_CUTLASS
FlashInfer autotune:   enabled
CUDA graph:            FULL requested -> FULL_DECODE_ONLY effective
Custom all-reduce:     disabled
Warm 512-token decode: ~95.8 tok/s
Estonia:               PASS 29 / FAIL 1 @ 92.2 gen tok/s
Estonia-long:          PASS 30 / FAIL 0 @ 92.4 gen tok/s
```

See [V17.md](V17.md) for the full compose, memory/KV details, correctness validation, backend experiments, and benchmark results.
