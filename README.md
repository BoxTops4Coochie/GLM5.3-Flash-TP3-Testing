# GLM-5.3 Flash Native TP3 / EP3 Notes

This repository documents the native GLM-5.3 Flash TP3/EP3 bring-up and optimization work on three RTX PRO 6000 Blackwell GPUs.

The current qualified path uses the native NVIDIA `Glm5Next` implementation with ModelOpt mixed-precision weights, B12X sparse MLA, native KDA/Gated DeltaNet, FP8 KV cache, replicated vision encoder mode, deterministic FlashInfer CUTLASS target MoE execution, and native MTP speculative decoding.

## Documentation

- [`NATIVE-TP3.md`](NATIVE-TP3.md) — native TP3 bring-up and debugging through v16.
- [`V17.md`](V17.md) — official checkpoint template/parser baseline and first stable post-correctness performance baseline.
- [`V18.md`](V18.md) — TP3-sharded shared expert.
- [`V19.md`](V19.md) — KDA `in_proj_qkvgfab` FP8 E4M3 weight-only Humming W8A16 optimization.
- [`V22.md`](V22.md) — deterministic target baseline: refined load-time expert placement, vision, memory/KV behavior, FlashInfer autotune determinism fix, Estonia validation, and serving configuration.
- [`V23-WIP.md`](V23.md) — native MTP TP3 bring-up, TP-aware MTP loading, draft-backend selection, and the qualified v23d serving configuration.
- [`GLM53-TP3-MTP1-5-qualification.md`](GLM53-TP3-MTP1-5-qualification.md) — full MTP1–MTP5 qualification sweep, including correctness, decode performance, vision, and acceptance-rate measurements on WIP v23
- [`VTP1-5.md`](VTP1-5.md) — older generic / Transformers-based TP3 padding experiments.

v20 and v21 were profiling/experimental bridge builds and intentionally do not have standalone Markdown files. Their important findings are summarized in `V22.md`.

---

## v22 baseline compose

```yaml
services:
  glm53-native-tp3:
    image: voipmonitor/vllm:glm53-native-tp3-bf16-v22-refined-zero-overhead
    container_name: glm53-native-tp3
    restart: "no"
    ipc: host
    shm_size: "64gb"
    ports: ["15015:8000"]

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
      - local-inference-lab/GLM-5.3-Flash-NVFP4
      - --tensor-parallel-size
      - "3"
      - --enable-expert-parallel
      - --mm-encoder-tp-mode
      - data
      - --max-model-len
      - "1048576"
      - --max-num-seqs
      - "8"
      - --max-num-batched-tokens
      - "8192"
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
      - --compilation-config
      - '{"cudagraph_mode":"FULL"}'
      - --additional-config
      - '{"glm53_kda_decode_backend":"triton"}'
```

See [`V22.md`](V22.md) for the full deterministic target-model baseline.

---

## v23 baseline compose

```yaml
services:
  glm53-native-tp3:
    image: voipmonitor/vllm:glm53-native-tp3-bf16-v23d-mtp-tp-loader
    container_name: glm53-native-tp3
    restart: "no"
    ipc: host
    shm_size: "64gb"
    ports: ["15015:8000"]

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
      - local-inference-lab/GLM-5.3-Flash-NVFP4
      - --tensor-parallel-size
      - "3"
      - --enable-expert-parallel
      - --mm-encoder-tp-mode
      - data
      - --max-model-len
      - "1048576"
      - --max-num-seqs
      - "8"
      - --max-num-batched-tokens
      - "4096"
      - --gpu-memory-utilization
      - "0.97"
      - --kv-cache-dtype
      - fp8
      - --disable-custom-all-reduce
      - --served-model-name
      - GLM-5.3-Flash-TP3
      - --moe-backend
      - auto
      - --linear-backend
      - auto
      - --speculative-config
      - '{"method":"mtp","num_speculative_tokens":1,"attention_backend":"B12X","moe_backend":"marlin"}'
      - --compilation-config
      - '{"cudagraph_mode":"FULL"}'
      - --additional-config
      - '{"glm53_kda_decode_backend":"triton"}'
```

MTP1 is the recommended qualified depth. See [`V23.md`](V23.md) and [`GLM53-TP3-MTP1-MTP5-qualification.md`](GLM53-TP3-MTP1-MTP5-qualification.md).
