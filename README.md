# GLM-5.3 Flash Native TP3 / EP3 Notes

This repository documents the native GLM-5.3 Flash TP3/EP3 bring-up and optimization work on three RTX PRO 6000 Blackwell GPUs.

The current path uses the native NVIDIA `Glm5Next` implementation with ModelOpt NVFP4 routed experts, native sparse MLA, native KDA/Gated DeltaNet, and text-only serving.

## Documentation

- [NATIVE-TP3.md](NATIVE-TP3.md) - full native TP3 history and debugging through v16, including KDA/MLA/vocab padding, EP3 bring-up, the v14 shared-expert TP-reduction correctness fix, and template/parser isolation.
- [V17.md](V17.md) - official checkpoint template/parser baseline, 1M-context compose, CUDA-graph behavior, FlashInfer autotuning, and the first stable performance baseline after correctness was restored.
- [V18.md](V18.md) - replaces replicated shared experts with a padded TP3-sharded shared expert, saving about 1.3 GiB/GPU and raising short decode to ~104.19 tok/s.
- [V19.md](V19.md) - converts the 34 bandwidth-bound KDA `in_proj_qkvgfab` projections to runtime FP8 E4M3 weight-only Humming W8A16, reaching ~111.91 tok/s and passing both Estonia suites 30/30.
- [VTP1-5.md](VTP1-5.md) - earlier generic/Transformers-based TP3 padding experiments. Development moved away from this path in favor of the native ModelOpt NVFP4 implementation.

---

# Current baseline: v19

**v19 is the current confirmed correctness/performance baseline.**

The major progression is:

```text
v14: fix replicated shared-expert TP SUM correctness bug
v17: restore official/default checkpoint template + glm45 parser
v18: TP-shard shared expert instead of replicating it
v19: FP8 W8A16 Humming for the 34 hot KDA merged projections
```

Current headline configuration:

```text
checkpoint:              LibertAIDAI/GLM-5.3-Flash-NVFP4
TP / EP:                 3 / 3
context:                 1,048,576
routed experts:          288 global -> 96/GPU
shared expert:           2048 logical -> 2052 physical -> 684/GPU
KDA heads:               64 logical -> 66 physical -> 22/GPU
MLA heads:               64 logical -> 72 physical -> 24/GPU
vision:                  disabled
MTP / DFlash:            disabled
KV dtype:                FP8
KDA decode:              Triton
KDA hot projection:      FP8 E4M3 W8A16 / Humming
MoE backend:             auto -> FLASHINFER_CUTLASS
FlashInfer autotune:     enabled
custom all-reduce:       disabled
GPU power cap:           300 W per GPU (900 W total for the 3-GPU node)
```

---

# Performance progression

| Version | Main change | 512-token decode | Estonia | Estonia-long |
|---|---|---:|---:|---:|
| v17 | official template/parser + v14 correctness fix | ~95.76 tok/s | historical baseline | historical baseline |
| v18 | TP-sharded shared expert | **104.19 tok/s** | **30/30 @ ~100.4 tok/s** | **29/30 @ ~99.8 tok/s** |
| v19 | KDA W8A16 Humming | **111.91 tok/s** | **30/30 @ 108.5 tok/s** | **30/30 @ 108.0 tok/s** |

v19 improvement over v18:

```text
512-token decode: +7.4%
Estonia:          ~+8.1%
Estonia-long:     ~+8.2%
```

No MTP or DFlash is enabled in these measurements.

---

# v19 Estonia results

Normal Estonia:

```text
completed:                  30/30
score:                      PASS 30 / FAIL 0
completion tokens avg:      3,463
TTFT avg:                   0.37 s
aggregate generation speed: 108.5 tok/s
mean per-request speed:     108.6 tok/s
```

Estonia-long:

```text
completed:                  30/30
score:                      PASS 30 / FAIL 0
completion tokens avg:      3,707
TTFT avg:                   0.40 s
aggregate generation speed: 108.0 tok/s
mean per-request speed:     108.2 tok/s
```

See [V19.md](V19.md) for the complete benchmark statistics, Humming microbenchmark, integration details, and correctness checks.

---

# Docker images

Earlier native TP3 debug builds through v15 plus v17 were published as public Docker Hub images under:

```text
azallaza/glm53-tp3-testing
```

Example:

```bash
docker pull azallaza/glm53-tp3-testing:v17-official-template-parser
```

Published tags already documented in the repository include:

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

v16 intentionally has no separate image because it was a configuration/template-only validation performed with the v15 image.

The v18/v19 pages currently identify the exact local build tags used for the measurements:

```text
voipmonitor/vllm:glm53-native-tp3-bf16-v18-shared-tp
voipmonitor/vllm:glm53-native-tp3-bf16-v19-kda-w8a16
```

No public v18/v19 Docker Hub tag is asserted here until those images are explicitly published.

---

# Current v19 compose

```yaml
services:
  glm53-native-tp3:
    image: voipmonitor/vllm:glm53-native-tp3-bf16-v19-kda-w8a16
    container_name: glm53-native-tp3
    restart: "no"

    ipc: host
    shm_size: "64gb"

    ports:
      - "15015:8000"

    volumes:
      - /m2-2/huggingface:/root/.cache/huggingface

    environment:
      HF_HUB_OFFLINE: "1"
      GLM53_DEBUG_KDA_STATE: "0"
      GLM53_DEBUG_MLA_HEADS: "0"
      GLM53_DEBUG_INDEXER: "0"
      GLM53_TORCH_PROFILE: "0"

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0", "1", "2"]
              capabilities: [gpu]

    command:
      - LibertAIDAI/GLM-5.3-Flash-NVFP4
      - --tensor-parallel-size
      - "3"
      - --enable-expert-parallel
      - --language-model-only
      - --max-model-len
      - "1048576"
      - --max-num-seqs
      - "1"
      - --max-num-batched-tokens
      - "2048"
      - --gpu-memory-utilization
      - "0.95"
      - --kv-cache-dtype
      - fp8
      - --disable-custom-all-reduce
      - --served-model-name
      - GLM-5.3-Flash-TP3
      - --moe-backend
      - auto
      - --linear-backend
      - auto
      - --enable-flashinfer-autotune
      - --additional-config
      - '{"glm53_kda_decode_backend":"triton"}'
```

---

# Current status / next targets

The obvious low-risk runtime/backend switches have already been explored around the v18 baseline. v19 addressed the largest identified conventional KDA projection hotspot.

Remaining conventional optimization targets before MTP/DFlash are:

```text
1. 42-layer MoE path
2. MHC launch/fusion overhead
3. only then proceed to MTP / DFlash
```

World-size-3 communication remains largely constrained to PyNCCL because the tested custom all-reduce paths do not support this geometry.
