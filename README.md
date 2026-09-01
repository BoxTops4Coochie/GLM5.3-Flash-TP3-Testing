# GLM-5.3 Flash Native TP3 / EP3 Notes

This repository documents the native GLM-5.3 Flash TP3/EP3 bring-up and optimization work on three RTX PRO 6000 Blackwell GPUs.

The current baseline uses the native NVIDIA `Glm5Next` implementation with ModelOpt mixed-precision weights, native sparse MLA, native KDA/Gated DeltaNet, FP8 KV cache, replicated vision encoder mode, and deterministic FlashInfer CUTLASS MoE execution with autotuning disabled.

## Documentation

- `NATIVE-TP3.md` - native TP3 bring-up and debugging through v16.
- `V17.md` - official checkpoint template/parser baseline and first stable post-correctness performance baseline.
- `V18.md` - TP3-sharded shared expert.
- `V19.md` - KDA `in_proj_qkvgfab` FP8 E4M3 weight-only Humming W8A16 optimization.
- `V22.md` - current deterministic baseline: zero-overhead refined expert placement, vision, memory/KV behavior, determinism fix, Estonia validation, and final serving compose.
- `VTP1-5.md` - older generic/Transformers-based TP3 padding experiments.

v20 and v21 were profiling/experimental bridge builds and intentionally do not have standalone Markdown files. Their important findings are summarized in `V22.md`.

---

# Current baseline: v22

v22 is the current frozen target-model baseline before renewed MTP work.

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
checkpoint:              local-inference-lab/GLM-5.3-Flash-NVFP4
snapshot:                378ca54585c46542bad1f3cb3ed0d73ae51cdb62
TP / EP:                 3 / 3
context:                 1,048,576
vision:                  enabled
vision TP mode:          data / replicated encoder
MTP / DFlash:            disabled
KV dtype:                FP8
KDA decode:              Triton
KDA hot projection:      FP8 E4M3 W8A16 / Humming
MLA backend:             B12X
MoE backend:             FLASHINFER_CUTLASS
FlashInfer autotune:     disabled
expert placement:        refined, load-time only
runtime remap kernel:     none
graphs:                  FULL -> FULL_DECODE_ONLY
TP/EP communication:     PyNCCL / NCCL 2.31.2
```

Headline results:

```text
determinism:
20 / 20 routed-expert captures identical
20 / 20 strict arithmetic PASS
first-token logprob spread: 0.000000

text decode:
109.91 tok/s mean, text-only diagnostic
109.97 tok/s mean, vision loaded

Estonia:                  28 / 30 PASS @ 102.6 tok/s
Estonia-long:             30 / 30 PASS @ 103.0 tok/s

vision:
anime_hill.png:           109.89 tok/s
retro_anime_portrait.png: 109.62 tok/s
KV pool:                  1,761,607 tokens
1M max-context capacity:  1.68x
```

The preferred general-purpose scheduler configuration is:

```text
max_num_seqs=8
max_num_batched_tokens=8192
gpu_memory_utilization=0.95
```

For implementation details, build lineage, refined expert placement, the FlashInfer autotune determinism investigation, final compose, Estonia results, and vision validation, see:

**[`V22.md`](V22.md)**

---

# Performance progression

| Version | Main change | 512-token decode |
|---|---|---:|
| v17 | official template/parser + corrected TP3 path | ~95.76 tok/s |
| v18 | TP-sharded shared expert | 104.19 tok/s |
| v19 | KDA W8A16 Humming | 111.91 tok/s |
| v22 pre-determinism-fix | load-time refined expert placement | ~112.51 tok/s |
| **v22 final** | **same target path, FlashInfer autotune disabled** | **109.91 tok/s** |

The small throughput reduction in the final v22 baseline is intentional: it removes the observed run-to-run target-MoE nondeterminism.

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

v22 text + vision is now frozen as the deterministic target baseline.

```text
1. requalify MTP1
2. requalify MTP2
3. requalify MTP3
4. select the best speculative baseline
5. evaluate Infernix/LIL TP3 improvements
6. add DFlash2
7. revisit smaller conventional tuning
```

World-size-3 communication remains PyNCCL because the tested custom all-reduce paths do not support this TP3 geometry.
