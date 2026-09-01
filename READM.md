# GLM-5.3 Flash Native TP3 / EP3 Notes

This repository documents the native GLM-5.3 Flash TP3/EP3 bring-up and optimization work on three RTX PRO 6000 Blackwell GPUs.

The current path uses the native NVIDIA `Glm5Next` implementation with ModelOpt NVFP4 routed experts, native sparse MLA, native KDA/Gated DeltaNet, FP8 KV cache, and text-only serving.

## Documentation

- `NATIVE-TP3.md` - native TP3 bring-up and debugging through v16.
- `V17.md` - official checkpoint template/parser baseline and first stable post-correctness performance baseline.
- `V18.md` - TP3-sharded shared expert.
- `V19.md` - KDA `in_proj_qkvgfab` FP8 E4M3 weight-only Humming W8A16 optimization.
- `V22.md` - current baseline: zero-overhead refined expert placement, scheduler/KV trade-offs, prefill, Estonia validation, concurrency scaling, and both recommended compose configurations.
- `VTP1-5.md` - older generic/Transformers-based TP3 padding experiments.

v20 and v21 were profiling/experimental bridge builds and intentionally do not have standalone Markdown files. Their important findings are summarized in `V22.md`.

---

# Current baseline: v22

v22 is the current confirmed text-only correctness/performance baseline.

```text
v14: fix shared-expert TP SUM correctness bug
v17: official/default checkpoint template + glm45 parser
v18: TP-sharded shared expert
v19: KDA W8A16 Humming optimization
v20: TP3 collective / MoE profiling
v21: runtime expert-remap experiment + identity control
v22: refined expert placement encoded entirely at model load
```

Headline configuration:

```text
checkpoint:              LibertAIDAI/GLM-5.3-Flash-NVFP4
TP / EP:                 3 / 3
context:                 1,048,576
vision:                  disabled
MTP / DFlash:            disabled
KV dtype:                FP8
KDA decode:              Triton
KDA hot projection:      FP8 E4M3 W8A16 / Humming
MLA backend:             B12X
MoE backend:             FLASHINFER_CUTLASS
expert placement:        refined, load-time only
runtime remap kernel:     none
TP/EP communication:     PyNCCL / NCCL 2.31.2
```

Headline results:

```text
512-token decode:         ~112.51 tok/s

Estonia:                  30 / 30 PASS @ 108.9 tok/s
Estonia-long:             30 / 30 PASS @ 108.4 tok/s
combined correctness:     60 / 60 PASS

repeated prefill:
8k:                       10,250 tok/s
16k:                       9,881 tok/s
32k:                       9,726 tok/s
64k:                       9,235 tok/s
128k:                      8,991 tok/s

aggregate decode:
C1 @ 0 ctx:               111.4 tok/s
C4 @ 0 ctx:               301.7 tok/s
C8 @ 0 ctx:               505.7 tok/s
```

The preferred general-purpose scheduler configuration is:

```text
max_num_seqs=8
max_num_batched_tokens=8192
```

A lower `max_num_batched_tokens` value can recover substantially more KV-cache capacity at the cost of smaller prefill chunks and potentially lower prefill throughput.

For the full v22 implementation history, v20/v21 findings, load-time expert permutation design, build/image lineage, detailed memory logs, complete `1 / 2048` and `8 / 8192` compose files, Estonia results, prefill measurements, and full C1/C2/C4/C8 concurrency table, see:

**[`V22.md`](V22.md)**

---

# Performance progression

| Version | Main change | 512-token decode |
|---|---|---:|
| v17 | official template/parser + corrected TP3 path | ~95.76 tok/s |
| v18 | TP-sharded shared expert | 104.19 tok/s |
| v19 | KDA W8A16 Humming | 111.91 tok/s |
| **v22** | **load-time refined expert placement** | **112.51 tok/s** |

No MTP or DFlash is enabled in these measurements.

---

# Docker images

Existing published images use:

```text
azallaza/glm53-tp3-testing
```

Current local v22 image:

```text
voipmonitor/vllm:glm53-native-tp3-bf16-v22-refined-zero-overhead
```

Recommended public v22 tag:

```text
azallaza/glm53-tp3-testing:v22-refined-zero-overhead
```

---

# Current status / next targets

v22 is the text-only baseline to freeze before multimodal/speculative work.

```text
1. enable vision on native TP3
2. verify text-only correctness/performance remains intact
3. enable MTP
4. benchmark correctness and speculative acceptance/performance
5. enable DFlash
6. revisit smaller conventional tuning afterward
```

World-size-3 communication remains PyNCCL because the tested custom all-reduce paths do not support this TP3 geometry.
