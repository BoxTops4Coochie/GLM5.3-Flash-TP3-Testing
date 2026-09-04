# GLM-5.3-Flash TP3 — DFlash2 Runtime Record

**Configuration:** TP3 / EP3 / DFlash2 speculative decoding  
**Target model:** `local-inference-lab/GLM-5.3-Flash-NVFP4`  
**Draft model:** `local-inference-lab/GLM-5.3-Flash-DFlash2`  
**Container image:** `infernix/vllm@sha256:187432cdb974645089bdcb9eb526a5e7b99ff00f25493698f037f24722884d4c`  
**Served model name:** `GLM-5.3-Flash-TP3`  
**vLLM build:** `0.26.1rc0+glm53.flash.nvfp4.luke.clean.r1.vllme75bcfd.b12x58a046f`

---

## Full Docker Compose

```yaml
services:
  glm53-native-tp3:
    image: infernix/vllm@sha256:187432cdb974645089bdcb9eb526a5e7b99ff00f25493698f037f24722884d4c
    container_name: glm53-native-tp3

    restart: "no"

    ipc: host
    shm_size: "64gb"

    ports:
      - "15015:8000"

    volumes:
      - /m2-2/huggingface:/root/.cache/huggingface
      - /m2-2/vllm/glm53-r17-tp3:/cache

    environment:
      # Exact qualified revisions.
      MODEL_REVISION: "9c712132678ee8ec869db9f848042ab8314c7685"
      DFLASH_MODEL_REVISION: "dfa270d7eb8df37e0cd0d4420f8dd0bd24ffcd50"

      # Enable only after target + draft are cached.
      # HF_HUB_OFFLINE: "1"

      # Leave old v23i tuning disabled for first baseline.
      # NCCL_MIN_NCHANNELS: "32"
      # NCCL_MAX_NCHANNELS: "32"
      # NCCL_CUMEM_ENABLE: "0"
      # NCCL_IB_DISABLE: "1"
      # NCCL_P2P_LEVEL: "SYS"
      # NCCL_PROTO: "LL,LL128,Simple"

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0", "1", "2"]
              capabilities: [gpu]

    entrypoint:
      - /bin/bash
      - -lc

    command:
      - |
          # ============================================================
          # MODE
          #
          # mtp      = MTP depth 3
          # dflash2  = DFlash2 depth 7
          # ============================================================
          MODE=dflash2

          # ============================================================
          # QUALIFIED TP3 SETTINGS
          # ============================================================
          export TP=3
          export DCP=1

          export MAX_MODEL_LEN=1048576
          export MAX_NUM_SEQS=8
          export MAX_NUM_BATCHED_TOKENS=8192
          export PREFILL_SCHEDULE_INTERVAL=8

          export GPU_MEMORY_UTILIZATION=0.91

          export CACHE_MODE=vram
          export KV_CACHE_QUANT=fp8_ds_mla

          export GLM53_KDA_PREFILL_BACKEND=flashkda

          # ============================================================
          # MODE-SPECIFIC SETTINGS
          # ============================================================
          case "$$MODE" in
            mtp)
              export SPECULATOR=mtp
              export MTP=3
              ;;

            dflash2)
              export SPECULATOR=dflash2
              unset MTP
              ;;

            *)
              echo "Invalid MODE=$$MODE"
              exit 1
              ;;
          esac

          # ============================================================
          # SHOW ACTIVE SETTINGS
          # ============================================================
          echo "============================================================"
          echo "GLM-5.3-Flash TP3"
          echo "============================================================"
          echo "MODE=$$MODE"
          echo "SPECULATOR=$$SPECULATOR"
          echo "MTP=$${MTP:-N/A}"
          echo "TP=$$TP"
          echo "DCP=$$DCP"
          echo "MAX_MODEL_LEN=$$MAX_MODEL_LEN"
          echo "MAX_NUM_SEQS=$$MAX_NUM_SEQS"
          echo "MAX_NUM_BATCHED_TOKENS=$$MAX_NUM_BATCHED_TOKENS"
          echo "PREFILL_SCHEDULE_INTERVAL=$$PREFILL_SCHEDULE_INTERVAL"
          echo "GPU_MEMORY_UTILIZATION=$$GPU_MEMORY_UTILIZATION"
          echo "CACHE_MODE=$$CACHE_MODE"
          echo "KV_CACHE_QUANT=$$KV_CACHE_QUANT"
          echo "GLM53_KDA_PREFILL_BACKEND=$$GLM53_KDA_PREFILL_BACKEND"
          echo "============================================================"

          # ============================================================
          # START EXACT TP3 R17 LAUNCHER
          # ============================================================
          exec /usr/local/bin/serve-glm53-flash-tp3-r17.sh \
            --served-model-name GLM-5.3-Flash-TP3
```

---

## Qualified Runtime Summary

| Item | Runtime value |
|---|---|
| Mode | `dflash2` |
| Speculative decoder | `DFlash2` |
| MTP | `N/A` in this run |
| Speculative draft depth | **7 tokens** |
| Draft KV layers | **5** |
| Draft KV cache groups | **1** |
| Tensor parallelism | `TP=3` |
| Expert parallelism | `EP=3` |
| Decode context parallelism | `DCP=1` |
| Pipeline parallelism | `PP=1` |
| Maximum model length | **1,048,576 tokens** |
| Maximum sequences | `8` |
| Maximum batched tokens | `8192` |
| Prefill schedule interval | `8` |
| Chunked prefill | Enabled |
| Prefix caching | Enabled |
| GPU memory utilization target | `0.91` |
| KV cache generic dtype | **FP8** |
| KV cache backend format | **fp8_ds_mla** |
| KV cache layout | **BLHNC** |
| Attention block/page size | **3328 tokens** |
| Total reported GPU KV token capacity | **2,143,243 tokens** |
| 1,048,576-token theoretical max concurrency | **2.04×** |
| GPU 0 current KV allocation | **17.79 GiB** |
| GPU 1 current KV allocation | **17.73 GiB** |
| GPU 2 current KV allocation | **18.04 GiB** |
| Initial reported available KV memory | **17.79 GiB** |
| Target checkpoint load | **184 GiB** |
| Draft checkpoint load | **1.20 GiB** |
| Model load memory | **61.74 GiB/GPU** |
| Peak activation memory | **4.19 GiB/GPU** |
| CUDA graph pool | **0.25 GiB/GPU actual** |
| KDA decode backend | `b12x` |
| KDA prefill backend | `flashkda` |
| Target attention backend | `B12X` / MLA path |
| Draft attention backend | `FLASH_ATTN` |
| DFlash draft attention window | **2048** |
| DFlash `hkv` | `3` |
| MoE backend | `FLASHINFER_CUTLASS` |
| Linear backend | `b12x` |
| All-reduce | `B12X PCIe oneshot` |
| NCCL | `2.31.2` |
| Total experts | **288** |
| Experts local per EP rank | **96** |
| Eagle3 auxiliary layers | `6, 15, 25, 34, 43` |

> **Important:** The value `7` is the **DFlash2 speculative depth**, not MTP depth.  
> This run explicitly reports `MTP=N/A`. The DFlash draft model proposes up to seven speculative tokens, while its independent draft KV cache contains five KV layers.

---

# Proof From Startup / Runtime Logs

## 1. Active Compose-Supplied Settings

```text
============================================================
GLM-5.3-Flash TP3
============================================================
MODE=dflash2
SPECULATOR=dflash2
MTP=N/A
TP=3
DCP=1
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=8192
PREFILL_SCHEDULE_INTERVAL=8
GPU_MEMORY_UTILIZATION=0.91
CACHE_MODE=vram
KV_CACHE_QUANT=fp8_ds_mla
GLM53_KDA_PREFILL_BACKEND=flashkda
============================================================
```

This proves the intended launch mode and user-facing environment values.

---

## 2. vLLM Build and Target Model

```text
(APIServer pid=1) INFO 09-04 10:58:28 [api_utils.py:347]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.26.1rc0+glm53.flash.nvfp4.luke.clean.r1.vllme75bcfd.b12x58a046f
(APIServer pid=1) INFO 09-04 10:58:28 [api_utils.py:347]   █▄█▀ █     █     █     █  model   local-inference-lab/GLM-5.3-Flash-NVFP4
```

---

## 3. Effective vLLM Arguments

```text
(APIServer pid=1) INFO 09-04 10:58:28 [api_utils.py:286] non-default args: {'model_tag': 'local-inference-lab/GLM-5.3-Flash-NVFP4', 'enable_auto_tool_choice': True, 'tool_call_parser': 'glm47', 'host': '0.0.0.0', 'model': 'local-inference-lab/GLM-5.3-Flash-NVFP4', 'dtype': 'bfloat16', 'revision': '9c712132678ee8ec869db9f848042ab8314c7685', 'max_model_len': 1048576, 'quantization': 'modelopt_mixed', 'served_model_name': ['GLM-5.3-Flash-TP3'], 'load_format': 'instanttensor', 'attention_backend': 'B12X', 'reasoning_parser': 'glm45', 'tensor_parallel_size': 3, 'dcp_kv_cache_interleave_size': 4, 'cp_kv_cache_interleave_size': 4, 'enable_expert_parallel': True, 'block_size': 256, 'gpu_memory_utilization': 0.91, 'kv_cache_dtype': 'fp8', 'enable_prefix_caching': True, 'mamba_cache_mode': 'align', 'max_num_batched_tokens': 8192, 'max_num_seqs': 8, 'enable_chunked_prefill': True, 'prefill_schedule_interval': 8, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16], 'max_cudagraph_capture_size': 16, 'enable_flashinfer_autotune': False, 'linear_backend': 'b12x', 'speculative_config': {'method': 'dflash', 'model': 'local-inference-lab/GLM-5.3-Flash-DFlash2', 'revision': 'dfa270d7eb8df37e0cd0d4420f8dd0bd24ffcd50', 'num_speculative_tokens': 7, 'kv_cache_dtype': 'auto', 'attention_backend': 'FLASH_ATTN'}, 'additional_config': {'glm53_kda_decode_backend': 'b12x', 'kda_prefill_backend': 'flashkda'}}
```

Key proof extracted from this line:

```text
max_model_len=1048576
tensor_parallel_size=3
enable_expert_parallel=True
gpu_memory_utilization=0.91
kv_cache_dtype=fp8
enable_prefix_caching=True
max_num_batched_tokens=8192
max_num_seqs=8
enable_chunked_prefill=True
prefill_schedule_interval=8
attention_backend=B12X
linear_backend=b12x
speculative method=dflash
draft model=local-inference-lab/GLM-5.3-Flash-DFlash2
num_speculative_tokens=7
draft attention_backend=FLASH_ATTN
glm53_kda_decode_backend=b12x
kda_prefill_backend=flashkda
```

---

## 4. Speculative Depth = 7, Not MTP

The engine configuration independently confirms the draft depth:

```text
(EngineCore pid=986) INFO 09-04 10:58:50 [core.py:123] Initializing a V1 LLM engine (...) with config: model='local-inference-lab/GLM-5.3-Flash-NVFP4', speculative_config=SpeculativeConfig(method='dflash', model='local-inference-lab/GLM-5.3-Flash-DFlash2', num_spec_tokens=7), ... tensor_parallel_size=3, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, ... kv_cache_dtype=fp8, ... enable_prefix_caching=True, enable_chunked_prefill=True, ...
```

Runtime metrics also prove the active depth:

```text
(APIServer pid=1) INFO 09-04 11:01:16 [metrics.py:125] SpecDecoding metrics: Mean acceptance length: 6.82, Current speculative depth: 7, Accepted throughput: 6.46 tokens/s, Drafted throughput: 7.77 tokens/s, Accepted: 256 tokens, Drafted: 308 tokens, Per-position acceptance rate: 1.000, 1.000, 1.000, 1.000, 0.795, 0.614, 0.409, Avg Draft acceptance rate: 83.1%
```

Therefore:

```text
MTP=N/A
SPECULATOR=DFlash2
SPECULATIVE_DRAFT_DEPTH=7
```

---

## 5. KV Cache Generic Dtype = FP8

vLLM explicitly reports:

```text
(APIServer pid=1) INFO 09-04 10:58:31 [cache.py:337] Using fp8 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, it may cause accuracy drop without a proper scaling factor
```

The engine configuration also contains:

```text
kv_cache_dtype=fp8
```

---

## 6. KV Cache Actual MLA Format = `fp8_ds_mla`

The B12X MLA backend explicitly confirms the more specific cache format:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:04 [mla_attention.py:634] Using fp8_ds_mla KV cache format for B12X backend.
```

Therefore, record both fields:

```text
KV_CACHE_DTYPE=fp8
KV_CACHE_FORMAT=fp8_ds_mla
```

---

## 7. KV Cache Layout = BLHNC

```text
(EngineCore pid=986) INFO 09-04 10:59:45 [utils.py:305] Using BLHNC KV cache layout.
```

---

## 8. KV / Attention Page Size = 3328 Tokens

All three workers report the same effective attention block size:

```text
(Worker_TP1_EP1 pid=1160) INFO 09-04 10:59:45 [interface.py:1021] Setting attention block size to 3328 tokens to ensure that attention page size is >= mamba page size.
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:45 [interface.py:1021] Setting attention block size to 3328 tokens to ensure that attention page size is >= mamba page size.
(Worker_TP2_EP2 pid=1161) INFO 09-04 10:59:45 [interface.py:1021] Setting attention block size to 3328 tokens to ensure that attention page size is >= mamba page size.
```

The associated Mamba page padding is:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:45 [interface.py:1054] Padding mamba page size by 5.79% to ensure that mamba page size and attention page size are exactly equal.
```

---

## 9. DFlash Draft KV = 5 Layers

This is distinct from the seven-token speculative depth.

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 11:00:01 [kv_cache_utils.py:1885] Keeping 5 DFlash draft KV layers in 1 independent cache groups
```

The same is repeated on all TP ranks:

```text
(Worker_TP2_EP2 pid=1161) INFO 09-04 11:00:01 [kv_cache_utils.py:1885] Keeping 5 DFlash draft KV layers in 1 independent cache groups
(Worker_TP1_EP1 pid=1160) INFO 09-04 11:00:01 [kv_cache_utils.py:1885] Keeping 5 DFlash draft KV layers in 1 independent cache groups
```

There is also a cache-padding warning:

```text
(Worker_TP0_EP0 pid=1159) WARNING 09-04 11:00:01 [kv_cache_utils.py:1253] Add 10 padding layers, may waste at most 29.41% KV cache memory
```

---

## 10. KV Pool Token Capacity

This is the most useful capacity line for comparing builds:

```text
(EngineCore pid=986) INFO 09-04 11:00:12 [kv_cache_utils.py:2033] GPU KV cache size: 2,143,243 tokens, Maximum concurrency for 1,048,576 tokens per request: 2.04x
```

Record:

```text
GPU_KV_CACHE_SIZE=2143243 tokens
MAX_CONCURRENCY_AT_1048576_CONTEXT=2.04x
```

---

## 11. KV Cache Memory Per GPU

Initial available KV cache memory on TP0:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 11:00:12 [gpu_worker.py:624] Available KV cache memory: 17.79 GiB
```

Final memory-accounting lines:

### GPU 0

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 11:00:17 [gpu_worker.py:855] Free memory on device (91.64/94.97 GiB) on startup. Desired GPU memory utilization is (0.91, 86.42 GiB). Actual usage is 64.44 GiB for consumed memory (weights + non-torch), 4.19 GiB for peak activation, and 0.25 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=18676332626` (17.39 GiB) to fit into requested memory, or `--kv-cache-memory=24274632704` (22.61 GiB) to fully utilize gpu memory. Current kv cache memory in use is 17.79 GiB.
```

### GPU 1

```text
(Worker_TP1_EP1 pid=1160) INFO 09-04 11:00:17 [gpu_worker.py:855] Free memory on device (91.45/94.97 GiB) on startup. Desired GPU memory utilization is (0.91, 86.42 GiB). Actual usage is 64.5 GiB for consumed memory (weights + non-torch), 4.19 GiB for peak activation, and 0.25 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=18609223762` (17.33 GiB) to fit into requested memory, or `--kv-cache-memory=24006197248` (22.36 GiB) to fully utilize gpu memory. Current kv cache memory in use is 17.73 GiB.
```

### GPU 2

```text
(Worker_TP2_EP2 pid=1161) INFO 09-04 11:00:17 [gpu_worker.py:855] Free memory on device (92.36/94.97 GiB) on startup. Desired GPU memory utilization is (0.91, 86.42 GiB). Actual usage is 64.19 GiB for consumed memory (weights + non-torch), 4.19 GiB for peak activation, and 0.25 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=18944768082` (17.64 GiB) to fit into requested memory, or `--kv-cache-memory=25316786176` (23.58 GiB) to fully utilize gpu memory. Current kv cache memory in use is 18.04 GiB.
```

Summary:

```text
GPU0_KV_CACHE=17.79 GiB
GPU1_KV_CACHE=17.73 GiB
GPU2_KV_CACHE=18.04 GiB
```

The slight per-GPU variation comes from runtime memory accounting and does not mean the TP ranks are configured differently.

---

## 12. CUDA Graph Memory

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 11:00:17 [gpu_worker.py:792] CUDA graph pool memory: 0.25 GiB (actual), 0.38 GiB (estimated), difference: 0.12 GiB (50.0%).
```

All three ranks report the same actual pool:

```text
CUDAGRAPH_POOL_MEMORY=0.25 GiB/GPU
```

vLLM also notes that CUDA graph profiling slightly changes effective GPU-memory-utilization accounting:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 11:00:12 [gpu_worker.py:639] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9100 is equivalent to --gpu-memory-utilization=0.9061 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9139.
```

---

## 13. Target and Draft Model Load Sizes

Target checkpoint:

```text
(Worker_TP0_EP0 pid=1159) Loading safetensors using InstantTensor loader: 100% Completed | 184G/184G [00:23<00:00, 8.41GB/s]
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:30 [default_loader.py:489] Loading weights took 25.14 seconds
```

DFlash2 draft checkpoint:

```text
(Worker_TP0_EP0 pid=1159) Loading safetensors using InstantTensor loader: 100% Completed | 1.20G/1.20G [00:00<00:00, 10.5GB/s]
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:44 [default_loader.py:489] Loading weights took 0.77 seconds
```

Per-worker model-load memory:

```text
(Worker_TP1_EP1 pid=1160) INFO 09-04 10:59:45 [model_runner.py:408] Model loading took 61.74 GiB memory and 42.523385 seconds
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:45 [model_runner.py:408] Model loading took 61.74 GiB memory and 42.528694 seconds
(Worker_TP2_EP2 pid=1161) INFO 09-04 10:59:45 [model_runner.py:408] Model loading took 61.74 GiB memory and 42.527944 seconds
```

---

## 14. Quantization / Compute Kernels

The checkpoint contains multiple ModelOpt formats:

```text
(APIServer pid=1) WARNING 09-04 10:58:39 [modelopt.py:379] Detected ModelOpt fp8 checkpoint (quant_algo=FP8). Please note that the format is experimental and could change.
(APIServer pid=1) WARNING 09-04 10:58:39 [modelopt.py:1029] Detected ModelOpt NVFP4 checkpoint (quant_algo=NVFP4). Please note that the format is experimental and could change in future.
(APIServer pid=1) WARNING 09-04 10:58:39 [modelopt.py:1029] Detected ModelOpt NVFP4 checkpoint (quant_algo=W4A16_NVFP4). Please note that the format is experimental and could change in future.
(APIServer pid=1) WARNING 09-04 10:58:39 [modelopt.py:1699] Detected ModelOpt MXFP8 checkpoint. Please note that the format is experimental and could change in future.
```

The target MoE kernel:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:04 [nvfp4.py:304] Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN', 'HUMMING', 'EMULATION'].
```

MXFP8 linear GEMM:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:31 [__init__.py:869] Using B12xMxfp8LinearKernel for MXFP8 GEMM
```

DFlash fused context K/V projection:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:44 [qwen3_dflash.py:701] Using B12xMxfp8LinearKernel for the fused DFlash context K/V projection.
```

---

## 15. Expert Parallelism

The engine has expert parallelism enabled:

```text
enable_expert_parallel=True
```

Runtime expert placement proves `EP=3` and the expert counts:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:04 [expert_map_manager.py:245] [EP Rank 0/3] Expert parallelism is enabled. Expert placement strategy: linear. Local/global number of experts: 96/288.
```

Record:

```text
EP=3
LOCAL_EXPERTS=96
GLOBAL_EXPERTS=288
```

---

## 16. TP / Distributed Runtime

```text
(Worker pid=1159) INFO 09-04 10:59:00 [parallel_state.py:1647] world_size=3 rank=0 local_rank=0 distributed_init_method=file:///tmp/vllm_dist_57f20b9998fa483db7588148ce130828 backend=nccl
(Worker pid=1160) INFO 09-04 10:59:01 [parallel_state.py:1647] world_size=3 rank=1 local_rank=1 distributed_init_method=file:///tmp/vllm_dist_57f20b9998fa483db7588148ce130828 backend=nccl
(Worker pid=1161) INFO 09-04 10:59:01 [parallel_state.py:1647] world_size=3 rank=2 local_rank=2 distributed_init_method=file:///tmp/vllm_dist_57f20b9998fa483db7588148ce130828 backend=nccl
```

---

## 17. NCCL and B12X PCIe All-Reduce

NCCL:

```text
(Worker pid=1159) INFO 09-04 10:59:01 [nccl.py:25] Found nccl from environment variable VLLM_NCCL_SO_PATH=/opt/local-inference/nccl/lib/libnccl.so.2.31.2
(Worker pid=1159) INFO 09-04 10:59:01 [pynccl.py:113] vLLM is using nccl==2.31.2
```

B12X PCIe collective:

```text
(Worker pid=1159) INFO 09-04 10:59:02 [b12x_pcie_all_reduce.py:244] Using B12X PCIe all-reduce (algorithm=oneshot, one-shot max=86016, fused max=86016, two-shot bf16 max=off, DMA min=6291456).
```

TP dispatch order:

```text
(Worker pid=1159) INFO 09-04 10:59:02 [cuda_communicator.py:318] Using ['B12X_PCIE', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'
```

The build's own runtime proof line summarizes this:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:45 [utils.py:118] GLM53_R17_TP3_RUNTIME_PROOF {"collective_backend":"b12x_pcie_oneshot","expert_parallel_size":3,"kda_decode_backend":"b12x","kda_prefill_backend":"flashkda","mm_encoder_tp_mode":"weights"}
```

---

## 18. DFlash Attention Runtime

The custom DFlash attention library loads successfully:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 11:00:09 [__init__.py:139] [dflash_attn] loaded /cache/jit/cu133-torch213-glm53-r17-tp3-vllm87945c472d37-b12x4b604b9a326b-dense-ctx1m-seq8-bt8192/vllm/dflash_attn/cefe35a0abca1442/libdflash_attn.so
```

Its active configuration:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 11:00:09 [flash_attn.py:330] [dflash_attn] split-KV draft attention enabled: hkv=3 window=2048 max_reqs=12 ws_reqs=64
```

Therefore:

```text
DFLASH_HKV=3
DFLASH_WINDOW=2048
DFLASH_MAX_REQS=12
DFLASH_WS_REQS=64
```

---

## 19. Eagle3 Auxiliary Layers

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:30 [eagle3_utils.py:28] Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)
```

Repeated on each TP rank.

Record:

```text
EAGLE3_AUX_LAYERS=6,15,25,34,43
```

---

## 20. CUDAGraph Effective Mode

The requested/full graph mode is reduced for the GDN attention backend:

```text
(Worker_TP0_EP0 pid=1159) WARNING 09-04 11:00:01 [compilation.py:1412] CUDAGraphMode.FULL is not supported with GDNAttentionBackend backend (support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_DECODE_ONLY
```

Thus the effective runtime mode should be recorded as:

```text
CUDAGRAPH_MODE=FULL_DECODE_ONLY
```

---

## 21. L2 Prefetch State

The kernel itself initializes with:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 11:00:01 [l2_prefetch.py:255] [l2_prefetch] CuTe kernel ready (grid=16 block=128 chunk=4096; budgets A/B/C/A_mla=20/50/15/36 MB)
```

However persistent L2 set-aside is effectively disabled:

```text
(Worker_TP0_EP0 pid=1159) INFO 09-04 10:59:51 [l2_prefetch.py:445] [l2_prefetch] persisting L2 set-aside on cuda:0: L2 134 MB, max 84 MB, requested 0 MB, now 0 MB
```

So this run should **not** be described as reserving persistent L2 capacity.

---

# Observed Speculative-Decoding Behavior

These runtime lines are useful for later DFlash-depth comparisons.

### Early high-acceptance sample

```text
(APIServer pid=1) INFO 09-04 11:01:16 [metrics.py:125] SpecDecoding metrics: Mean acceptance length: 6.82, Current speculative depth: 7, Accepted throughput: 6.46 tokens/s, Drafted throughput: 7.77 tokens/s, Accepted: 256 tokens, Drafted: 308 tokens, Per-position acceptance rate: 1.000, 1.000, 1.000, 1.000, 0.795, 0.614, 0.409, Avg Draft acceptance rate: 83.1%
```

### Sustained generation sample

```text
(APIServer pid=1) INFO 09-04 11:47:26 [loggers.py:310] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 168.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.6%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 09-04 11:47:26 [metrics.py:125] SpecDecoding metrics: Mean acceptance length: 3.00, Current speculative depth: 7, Accepted throughput: 112.09 tokens/s, Drafted throughput: 391.98 tokens/s, Accepted: 1121 tokens, Drafted: 3920 tokens, Per-position acceptance rate: 0.732, 0.468, 0.330, 0.195, 0.129, 0.086, 0.062, Avg Draft acceptance rate: 28.6%
```

### Another sustained sample

```text
(APIServer pid=1) INFO 09-04 11:48:16 [loggers.py:310] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 155.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.6%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 09-04 11:48:16 [metrics.py:125] SpecDecoding metrics: Mean acceptance length: 2.83, Current speculative depth: 7, Accepted throughput: 100.29 tokens/s, Drafted throughput: 383.58 tokens/s, Accepted: 1003 tokens, Drafted: 3836 tokens, Per-position acceptance rate: 0.690, 0.449, 0.299, 0.170, 0.113, 0.071, 0.038, Avg Draft acceptance rate: 26.1%
```

These show that a configured draft depth of seven does **not** imply seven accepted tokens per speculative step. During sustained decode in these samples, mean acceptance length was typically around `2.6–3.0`, with the later speculative positions seeing much lower acceptance.

---

# Compact Benchmark Record

For future side-by-side comparisons, the most important fields from this run are:

```text
GLM-5.3-Flash TP3 R17

MODE=dflash2
TARGET_MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4
DRAFT_MODEL=local-inference-lab/GLM-5.3-Flash-DFlash2

TP=3
EP=3
DCP=1
PP=1

MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=8192
PREFILL_SCHEDULE_INTERVAL=8

GPU_MEMORY_UTILIZATION=0.91

KV_CACHE_DTYPE=fp8
KV_CACHE_FORMAT=fp8_ds_mla
KV_CACHE_LAYOUT=BLHNC
KV_ATTENTION_PAGE_SIZE=3328
GPU_KV_CACHE_SIZE=2143243 tokens
MAX_1M_CONTEXT_CONCURRENCY=2.04x

GPU0_KV_CACHE_MEMORY=17.79 GiB
GPU1_KV_CACHE_MEMORY=17.73 GiB
GPU2_KV_CACHE_MEMORY=18.04 GiB

SPEC_METHOD=DFlash2
SPEC_DEPTH=7
MTP=N/A
DRAFT_KV_LAYERS=5
DRAFT_KV_GROUPS=1
DFLASH_HKV=3
DFLASH_WINDOW=2048

TARGET_CHECKPOINT_SIZE=184 GiB
DRAFT_CHECKPOINT_SIZE=1.20 GiB
MODEL_LOAD_MEMORY=61.74 GiB/GPU
PEAK_ACTIVATION_MEMORY=4.19 GiB/GPU
CUDAGRAPH_POOL_MEMORY=0.25 GiB/GPU

KDA_DECODE_BACKEND=b12x
KDA_PREFILL_BACKEND=flashkda
MOE_BACKEND=FLASHINFER_CUTLASS
LINEAR_BACKEND=b12x
ALL_REDUCE=b12x_pcie_oneshot
NCCL=2.31.2

GLOBAL_EXPERTS=288
LOCAL_EXPERTS_PER_EP_RANK=96
EAGLE3_AUX_LAYERS=6,15,25,34,43
```

---

## Notes

1. `KV_CACHE_QUANT=fp8_ds_mla` is the launcher-facing configuration value, while vLLM's generic engine config reports `kv_cache_dtype=fp8`. The B12X MLA runtime explicitly verifies that the actual format is `fp8_ds_mla`.
2. DFlash2 uses **seven speculative tokens** in this run. This should not be recorded as MTP depth.
3. DFlash2 maintains **five draft KV layers** in one independent cache group. This is separate from speculative depth.
4. The reported `2,143,243` KV tokens are the best single capacity value to preserve when comparing future builds because it directly represents runtime KV token capacity.
5. The reported 1M-context concurrency of `2.04x` is based on the configured maximum request length of `1,048,576` tokens.
6. The three ranks have slightly different reported KV-memory allocations because of per-device runtime memory accounting.
7. CUDAGraph falls back from `FULL` to the effective runtime mode `FULL_DECODE_ONLY`.
8. Persistent L2 set-aside is `0 MB` in this run, despite the L2 prefetch kernel being available.
