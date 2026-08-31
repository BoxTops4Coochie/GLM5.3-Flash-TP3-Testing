# Native GLM-5.3 Flash TP3 / EP3 on 3× RTX PRO 6000 Blackwell

This document tracks the current native vLLM TP3/EP3 experiment for **GLM-5.3 Flash** on three SM120 GPUs.

The target configuration is:

```text
TP = 3
EP = 3
SM120
ModelOpt NVFP4 checkpoint
native Glm5Next implementation
native sparse MLA
native KDA / Gated DeltaNet
text-only serving
```

The server now loads and serves requests successfully. The major long-decode correctness bug documented through v13 was identified and fixed in v14. The historical sections below are intentionally preserved because they show how the failure was isolated. v15 and v16 then separated template/parser behavior from the numerical TP3 fix. Starting with v17, each major version is documented on its own page; see [V17.md](V17.md).

---

## Hardware / runtime

Test host:

```text
3× NVIDIA RTX PRO 6000 Blackwell Workstation Edition
96 GB ECC each
PCIe only, no NVLink
```

Base image:

```text
voipmonitor/vllm:glm53-dflash2-mxfp8-dev-20260828-4
```

Runtime source tree:

```text
/opt/glm53-flash/vllm/vllm
```

Observed source provenance:

```text
repository: local-inference-lab/vllm
branch: codex/glm53-dflash2-perf-015dcd42-20260828
commit: 479f2eef391516a6e5aedafc5783867b6b77ade9
tree: bbe1d1f078ef027e93b2eb7f0a9db04a5b52a956
```

Package / stack:

```text
vLLM: 0.26.1rc0+glm53.flash.nvfp4.luke.clean.r1.vllme75bcfd.b12x58a046f
CUDA: 13.3
PyTorch: 2.13
Transformers: 5.15
NCCL: 2.31.2
FlashInfer: 0.6.18+cu133
```

Checkpoint used for most testing:

```text
LibertAIDAI/GLM-5.3-Flash-NVFP4
```

Successful load characteristics:

```text
checkpoint size: ~181 GB
load time: ~23.7 s
weights + non-torch memory: ~61.9 GiB/GPU
peak activation: ~3.44 GiB/GPU
KV cache: ~24.8 GiB/GPU
KV tokens: ~2.42M tokens
EP3 routed experts: 96 / GPU
```

---

# Geometry changes required for TP3

## Routed experts

```text
288 global routed experts
/ 3 GPUs
= 96 routed experts per GPU
```

This is naturally divisible and does not require expert-width padding.

## Shared expert

The shared expert intermediate width is:

```text
2048
```

which is not divisible by 3. Rather than padding the shared MLP and changing its math, the shared expert is replicated on each rank with TP disabled.

## Vocabulary

Original vocab:

```text
154880
```

Padded physical vocab:

```text
154944
```

Per-rank vocab:

```text
51648
```

## KDA heads

Logical KDA geometry:

```text
64 heads × 128 dim
```

Physical TP3 geometry:

```text
64 logical heads
-> 66 physical heads
-> 22 heads / rank
```

Rank layout:

```text
rank 0: global heads  0-21  (22 real)
rank 1: global heads 22-43  (22 real)
rank 2: global heads 44-63  (20 real) + 2 dummy
```

The output projection columns and all KDA fused inputs are zero-padded for the two synthetic heads.

## MLA heads

The SM120 B12X sparse MLA path needs a local head count compatible with its kernel geometry. Generic 64 -> 66 -> 22 was insufficient because 22 local heads is not compatible with the sparse MLA implementation.

The current physical MLA geometry is:

```text
64 logical heads
-> 72 physical heads
-> 24 heads / rank
```

Rank layout:

```text
rank 0: global heads  0-23  (24 real)
rank 1: global heads 24-47  (24 real)
rank 2: global heads 48-63  (16 real) + 8 dummy
```

The synthetic rank-2 MLA heads are zero-padded in Q/K/V-related projection weights and the output projection.

---

# Modified files

The current native TP3 overlay modifies:

```text
vllm/models/glm5next/nvidia/tp_padding.py         NEW
vllm/models/glm5next/nvidia/attention.py
vllm/models/glm5next/nvidia/kda.py
vllm/models/glm5next/nvidia/model.py
vllm/models/glm5next/nvidia/pooled_indexer.py     debug-only in v13
vllm/config/model.py
```

The build context uses staged copies under:

```text
image/vllm/...
```

while the working tree is:

```text
vllm/vllm/...
```

This matters: changes must be copied into `image/vllm/...` before building.

---

# Version history

## v1 - Native Glm5Next TP3 starting point

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v1`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v1` in the compose examples below.


Moved away from the generic Transformers/VTP route and began patching the native NVIDIA `Glm5Next` implementation.

Initial padding helpers were added so model construction could progress under TP3.

Result: native code path established, but generic vLLM divisibility validation still blocked startup.

---

## v2 - Generic TP divisibility validator bypass

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v2`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v2` in the compose examples below.


Patched:

```text
vllm/config/model.py
```

so generic head-divisibility validation is bypassed specifically for native Glm5Next architectures.

Reason:

```text
64 attention heads % TP3 != 0
```

The native implementation must be allowed to apply its own physical padding before the generic validator aborts.

Result: startup advanced to the vision path.

---

## v3 - Text-only / vision-tower bypass

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v3`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v3` in the compose examples below.


Added a language-only construction path so the vision tower is skipped when using:

```text
--language-model-only
```

Reason: the vision model has its own incompatible TP3 head geometry and is not needed for this experiment.

Result: startup advanced to the shared expert.

---

## v4 - Replicated shared expert + EP3 routed experts

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v4`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v4` in the compose examples below.


The shared expert is replicated with TP disabled because its 2048 intermediate width is not divisible by 3.

The routed experts remain EP3:

```text
288 -> 96 experts/GPU
```

A forced B12X MoE backend was not compatible with EP3 in this configuration, so testing moved to:

```text
--moe-backend auto
```

which chooses a compatible backend for the NVFP4 checkpoint.

---

## v5 - Loader diagnostics

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v5`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v5` in the compose examples below.


Added weight-loader instrumentation to identify checkpoint/runtime shape mismatches.

A key failure appeared around the fused KDA parameter:

```text
layers.0.self_attn.in_proj_qkvgfab.weight
```

The checkpoint KDA tensors are BF16, while the runtime object had been created under the global ModelOpt NVFP4 quantization config.

Conclusion: the KDA path should not inherit the global NVFP4 quantizer.

---

## v6 - KDA quantization config override

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v6`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v6` in the compose examples below.


Before constructing the shared Kimi/GDN KDA implementation, a shallow copy of the vLLM config is made and:

```python
quant_config = None
```

is applied for KDA.

Reason: the pure NVFP4 checkpoint keeps KDA in BF16.

This allowed the KDA runtime shapes to align with the padded BF16 checkpoint tensors.

Caveat: this is specific to the pure NVFP4 checkpoint line and can conflict with checkpoints that actually quantize KDA.

---

## v7 - Stable TP3 geometry / running server

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v7`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v7` in the compose examples below.


This was the first important milestone where the model could fully load and serve requests under native TP3/EP3.

Important geometry at this point:

```text
KDA: 64 -> 66 -> 22 local
MLA: 64 -> 72 -> 24 local
shared expert: replicated
routed experts: EP3, 96/GPU
vocab: 154880 -> 154944 -> 51648/GPU
```

Observed server behavior:

```text
server starts
checkpoint fully loads
API works
short completions can be coherent
longer generations degrade into repetition / semantic corruption
```

A representative raw completion test:

```text
Prompt: "The capital of France is"
1-16 tokens: mostly coherent
32+ tokens: increasing repetition / semantic drift
```

This suggested recurrent/decode instability rather than immediate catastrophic weight corruption.

<details>
<summary>Representative v7-style compose baseline</summary>

```yaml
services:
  glm53-native-tp3:
    image: voipmonitor/vllm:glm53-native-tp3-bf16-v7
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
      - "32768"
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
      - --additional-config
      - '{"glm53_kda_decode_backend":"triton"}'
```

</details>

---

## v8 - First KDA debug attempt did not reach the image

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v8-debug`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v8-debug` in the compose examples below.


A KDA cache diagnostic was added to the working source but did not appear in the built container.

Root cause: Docker was copying from:

```text
image/vllm/...
```

while the edits had only been made under:

```text
vllm/vllm/...
```

This established the build-staging rule used for all later debug versions.

---

## v9 - KDA debug instrumentation active

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v9-kda-debug`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v9-kda-debug` in the compose examples below.


Image:

```text
voipmonitor/vllm:glm53-native-tp3-bf16-v9-kda-debug
```

The debug logger was confirmed inside the running container.

Initial instrumentation attempted statistics across the whole recurrent cache and used `.item()`.

Two debug-only failures were found:

1. `.item()` failed during CUDA graph capture.
2. full-cache statistics created multi-GB temporary tensors and OOMed.

Neither failure was a model correctness failure.

<details>
<summary>v9 debug compose pattern</summary>

```yaml
services:
  glm53-native-tp3:
    image: voipmonitor/vllm:glm53-native-tp3-bf16-v9-kda-debug
    container_name: glm53-native-tp3
    restart: "no"
    ipc: host
    shm_size: "64gb"

    ports:
      - "15015:8000"

    volumes:
      - /m2-2/huggingface:/root/.cache/huggingface
      - /home/aabduh/glm53-native-tp3/template-test/minimal-no-think.jinja:/minimal-no-think.jinja:ro

    environment:
      HF_HUB_OFFLINE: "1"
      GLM53_DEBUG_KDA_STATE: "1"

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
      - "32768"
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
      - --additional-config
      - '{"glm53_kda_decode_backend":"triton"}'
      - --chat-template
      - /minimal-no-think.jinja
```

</details>

---

## v10 - Lightweight KDA cache sampling

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v10-kda-debug`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v10-kda-debug` in the compose examples below.


Image:

```text
voipmonitor/vllm:glm53-native-tp3-bf16-v10-kda-debug
```

CUDA graphs were disabled for the debug run:

```text
--compilation-config {"cudagraph_mode":"NONE"}
```

The logger was changed to inspect only one recurrent/cache slot instead of the whole cache.

The first attempt sampled slot 0 and showed all zeros. This was **not** evidence that KDA state was unused; it revealed that slot 0 was not the active request slot.

<details>
<summary>v10 KDA debug compose</summary>

```yaml
services:
  glm53-native-tp3:
    image: voipmonitor/vllm:glm53-native-tp3-bf16-v10-kda-debug
    container_name: glm53-native-tp3
    restart: "no"
    ipc: host
    shm_size: "64gb"

    ports:
      - "15015:8000"

    volumes:
      - /m2-2/huggingface:/root/.cache/huggingface
      - /home/aabduh/glm53-native-tp3/template-test/minimal-no-think.jinja:/minimal-no-think.jinja:ro

    environment:
      HF_HUB_OFFLINE: "1"
      GLM53_DEBUG_KDA_STATE: "1"

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
      - "32768"
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
      - --additional-config
      - '{"glm53_kda_decode_backend":"triton"}'
      - --chat-template
      - /minimal-no-think.jinja
      - --compilation-config
      - '{"cudagraph_mode":"NONE"}'
```

</details>

---

## v11 - Active KDA state-slot instrumentation

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v11-kda-debug`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v11-kda-debug` in the compose examples below.


Image:

```text
voipmonitor/vllm:glm53-native-tp3-bf16-v11-kda-debug
```

The logger now reads the active request state index from forward metadata rather than hard-coding slot 0.

For the 32-token reproduction, the active slot was:

```text
state_idx = 1
```

### KDA result

Rank 2 real state became nonzero immediately and remained finite:

```text
conv_real RMS: ~0.18 - 0.26
recurrent real RMS: ~5.7e-4 - 7.7e-4
NaN: 0
Inf: 0
```

At the same time, the two synthetic KDA heads stayed exactly zero through prefill and decode:

```text
rec_dummy  rms=0 absmax=0
conv_dummy rms=0 absmax=0
```

### Conclusion

This strongly argues against direct contamination from the KDA 64 -> 66 padding.

The model output was still broken, so KDA dummy-state accumulation is not the cause.

<details>
<summary>v11 KDA active-slot debug compose</summary>

```yaml
services:
  glm53-native-tp3:
    image: voipmonitor/vllm:glm53-native-tp3-bf16-v11-kda-debug
    container_name: glm53-native-tp3
    restart: "no"
    ipc: host
    shm_size: "64gb"

    ports:
      - "15015:8000"

    volumes:
      - /m2-2/huggingface:/root/.cache/huggingface
      - /home/aabduh/glm53-native-tp3/template-test/minimal-no-think.jinja:/minimal-no-think.jinja:ro

    environment:
      HF_HUB_OFFLINE: "1"
      GLM53_DEBUG_KDA_STATE: "1"

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
      - "32768"
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
      - --additional-config
      - '{"glm53_kda_decode_backend":"triton"}'
      - --chat-template
      - /minimal-no-think.jinja
      - --compilation-config
      - '{"cudagraph_mode":"NONE"}'
```

</details>

---

## v12 - MLA real-vs-dummy head instrumentation

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v12-mla-debug`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v12-mla-debug` in the compose examples below.


Image:

```text
voipmonitor/vllm:glm53-native-tp3-bf16-v12-mla-debug
```

The first sparse MLA layer (`layers.3.self_attn`) was instrumented on TP rank 2.

Rank 2 geometry:

```text
24 local heads
0-15  = real
16-23 = synthetic
```

The logger measured both:

1. Q immediately before MLA.
2. MLA output immediately before `o_proj`.

### MLA Q result

Real Q heads were active:

```text
RMS: ~1.4 - 1.6
absmax: ~5 - 8
```

All eight synthetic Q heads were exactly zero:

```text
dummy RMS = 0
dummy absmax = 0
NaN = 0
Inf = 0
```

### MLA output result

Real attention output was active during the actual request:

```text
RMS: ~0.075 - 0.097
```

All eight synthetic MLA output heads remained exactly zero:

```text
dummy RMS = 0
dummy absmax = 0
```

### Conclusion

The B12X sparse MLA path is not visibly leaking values into the padded rank-2 heads.

This strongly weakens direct MLA dummy-head contamination as the root cause.

<details>
<summary>v12 MLA debug compose</summary>

```yaml
services:
  glm53-native-tp3:
    image: voipmonitor/vllm:glm53-native-tp3-bf16-v12-mla-debug
    container_name: glm53-native-tp3
    restart: "no"
    ipc: host
    shm_size: "64gb"

    ports:
      - "15015:8000"

    volumes:
      - /m2-2/huggingface:/root/.cache/huggingface
      - /home/aabduh/glm53-native-tp3/template-test/minimal-no-think.jinja:/minimal-no-think.jinja:ro

    environment:
      HF_HUB_OFFLINE: "1"
      GLM53_DEBUG_KDA_STATE: "0"
      GLM53_DEBUG_MLA_HEADS: "1"

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
      - "32768"
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
      - --additional-config
      - '{"glm53_kda_decode_backend":"triton"}'
      - --chat-template
      - /minimal-no-think.jinja
      - --compilation-config
      - '{"cudagraph_mode":"NONE"}'
```

</details>

---

## v13 - Sparse indexer cross-rank consistency instrumentation

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v13-indexer-debug`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v13-indexer-debug` in the compose examples below.


Image:

```text
voipmonitor/vllm:glm53-native-tp3-bf16-v13-indexer-debug
```

`pooled_indexer.py` was added to the overlay and instrumented to fingerprint the complete `[rows, 2051]` selection record after `expand_pool_ids()`.

The test compares on all three TP ranks:

```text
valid count
sum
weighted sum
first 12 entries
last 12 entries
```

### Result

For every observed prefill/decode step, ranks 0/1/2 produced matching records.

Examples:

```text
prefill rows=2:
rank 0 == rank 1 == rank 2
valid=1 / 2, same sums, same first/last entries
```

```text
prefill rows=18:
rank 0 == rank 1 == rank 2
last row valid=18
first=[0,1,2,3,4,5,6,7,8,9,10,11]
```

Decode then advanced consistently across all ranks:

```text
valid=19
valid=20
valid=21
...
valid=49
```

with matching:

```text
sum
weighted sum
first[]
last[]
```

on every rank.

### Conclusion

The sparse pooled selector is not diverging between TP ranks in the observed reproduction.

That moves cross-rank indexer disagreement down the suspect list.

<details>
<summary>v13 indexer debug compose</summary>

```yaml
services:
  glm53-native-tp3:
    image: voipmonitor/vllm:glm53-native-tp3-bf16-v13-indexer-debug
    container_name: glm53-native-tp3
    restart: "no"

    ipc: host
    shm_size: "64gb"

    ports:
      - "15015:8000"

    volumes:
      - /m2-2/huggingface:/root/.cache/huggingface
      - /home/aabduh/glm53-native-tp3/template-test/minimal-no-think.jinja:/minimal-no-think.jinja:ro

    environment:
      HF_HUB_OFFLINE: "1"
      GLM53_DEBUG_KDA_STATE: "0"
      GLM53_DEBUG_MLA_HEADS: "0"
      GLM53_DEBUG_INDEXER: "1"

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
      - "32768"
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
      - --additional-config
      - '{"glm53_kda_decode_backend":"triton"}'
      - --chat-template
      - /minimal-no-think.jinja
      - --compilation-config
      - '{"cudagraph_mode":"NONE"}'
```

</details>

---

# Reproduction used for v10-v13 correctness testing

Minimal no-thinking chat template:

```jinja
[gMASK]<sop>
{%- for m in messages -%}
{%- if m.role == 'system' -%}
<|system|>{{ m.content }}
{%- elif m.role == 'user' -%}
<|user|>{{ m.content }}
{%- elif m.role == 'assistant' -%}
<|assistant|>{{ m.content }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|assistant|>
{%- endif -%}
```

32-token request:

```bash
curl -s http://127.0.0.1:15015/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.3-Flash-TP3",
    "messages": [{
      "role": "user",
      "content": "What is 2 + 2? Answer only with the number."
    }],
    "temperature": 0,
    "max_tokens": 32
  }' | jq -r '.choices[0].message.content'
```

Representative broken output:

```text
The query is about computing 2 + 2, which is trivial since the base ten two 2 two two two two is even four two all two
```

This same failure pattern remained after disabling CUDA graphs and while KDA/MLA/indexer diagnostics showed sane state.

---

# Findings ruled out or strongly weakened

## Chat response parser

Ruled out as the primary cause.

Rendered prompt IDs fed directly to `/v1/completions` reproduced the same corruption.

## Official vs custom chat template

Strongly weakened.

The official current template and a minimal no-thinking template both reproduce the issue.

## Immediate checkpoint destruction

Strongly weakened.

Raw completions can remain coherent for the first several tokens and usually do not begin as random garbage.

## CUDA graph capture

Strongly weakened.

The same output corruption occurs with:

```text
{"cudagraph_mode":"NONE"}
```

## KDA backend choice

Strongly weakened as the sole cause.

Forced B12X and forced Triton KDA decode backends both fail, though not always identically.

## KDA synthetic-head state contamination

Strongly weakened.

Rank-2 dummy KDA recurrent and convolution regions stay exactly zero while real state evolves normally.

## MLA synthetic-head contamination

Strongly weakened.

Rank-2 dummy Q heads and dummy MLA outputs stay exactly zero while real heads are active and finite.

## Sparse indexer cross-rank disagreement

Strongly weakened.

Ranks 0/1/2 produce matching selector records through the observed prefill/decode sequence.

## MoE backend alone

Strongly weakened.

A MoE emulation experiment still produced broken output, so the auto-selected FLASHINFER_CUTLASS MoE backend is unlikely to be the only cause.

---

# Still suspicious / not yet cleared

The remaining problem is likely subtler than simple padded-head leakage.

High-value areas still worth testing:

1. **Cross-rank hidden-state agreement around row-parallel reductions**
   - Verify the hidden state after MLA `o_proj` and after KDA output projection is identical across TP ranks where it should be replicated.
   - A reduction/shard semantic bug can exist even if dummy heads stay zero.

2. **Shared expert path**
   - It is replicated specifically to work around the 2048 % 3 geometry.
   - Verify its output is numerically identical on all ranks and is combined exactly once.

3. **Residual / post-attention synchronization**
   - Compare checksums or small deterministic fingerprints of hidden states after every block family.
   - Find the first layer where TP ranks diverge.

4. **KDA recurrent semantics rather than dummy-state leakage**
   - KDA state is finite, but finite does not prove it matches TP1/TP2 reference math.
   - A head-shard ordering or output-reduction error could still corrupt recurrence.

5. **MLA real-head math / output reduction**
   - Dummy heads are inert, but the real 64 logical heads may still be combined incorrectly after padding.

6. **Loader shard ordering / fused parameter layout**
   - Especially around fused KDA inputs and row/column-parallel projections.

7. **Synthetic vocab padding interaction**
   - High-vocab token rows were not obviously shifted, but final LM-head shard/reduction behavior should still be verified explicitly.

---

# v14 - Shared-expert TP reduction bug identified and fixed

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v14-shared-expert-fix`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v14-shared-expert-fix` in the compose examples below.

v14 was the major correctness turning point.

## Root cause

The shared expert has an intermediate width of:

```text
2048
```

which is not divisible by TP3. The native TP3 adaptation therefore instantiated the shared expert as a fully replicated MLP:

```python
disable_tp=True
reduce_results=False
```

That construction itself was valid. The problem was what happened later in the generic FusedMoE reduction path.

Each TP rank computed the same complete shared-expert contribution:

```text
rank 0 = S
rank 1 = S
rank 2 = S
```

The generic MoE path then performed its normal TP SUM over the shared contribution:

```text
S + S + S = 3S
```

instead of the intended:

```text
S
```

This explains the pre-v14 failure mode unusually well: short output could remain plausible, but every affected MoE block systematically over-weighted the same valid shared-expert signal, and longer decode progressively destabilized into repetition / semantic collapse.

## v14 fix

`Glm5NextMLP` gained an optional output scale:

```python
output_scale: float = 1.0
```

and after `down_proj`:

```python
if self.output_scale != 1.0:
    x = x * self.output_scale
```

The replicated shared expert is instantiated with:

```python
output_scale=1.0 / self.tp_size
```

For TP3, each rank therefore contributes:

```text
S / 3
```

and the existing generic TP SUM restores the correct result:

```text
S/3 + S/3 + S/3 = S
```

Only the replicated shared-expert contribution is compensated. No global `/3` correction is applied to the model.

## Result

The old decode collapse disappeared.

Observed after v14:

```text
2 + 2 correctness:         fixed
150-250 token generations: coherent
old 16/32/64 collapse:     gone
KDA/MLA padding geometry:   unchanged
EP3 routed experts:         unchanged
```

The earlier diagnostics remained important because they had already shown:

```text
KDA dummy recurrent state: exactly zero
KDA dummy conv state:      exactly zero
MLA dummy Q heads:         exactly zero
MLA dummy outputs:         exactly zero
pooled sparse indexer:     matched across all TP ranks
```

Those results helped narrow the issue to reduction semantics rather than padded-head contamination.

---

# v15 - Parser-clean isolation

**Prebuilt image (public, no login required):** `docker pull azallaza/glm53-tp3-testing:v15-parser-clean`
→ replaces the local build `voipmonitor/vllm:glm53-native-tp3-bf16-v15-parser-clean`.

v15 kept the v14 numerical fix and removed only:

```text
--reasoning-parser glm45
```

from the launcher.

The goal was to isolate whether reasoning-parser handling explained the remaining strange arithmetic behavior seen with the custom minimal no-thinking template.

## Result

With the minimal custom template, a deterministic arithmetic test could still produce:

```text
17 * 23 -> 289289
```

while ordinary prompts remained coherent.

That showed:

```text
v14 numerical fix:       still valid
reasoning parser:        not the TP3 correctness root cause
minimal custom template: still suspicious
```

The catastrophic long-decode corruption fixed in v14 did not return.

---

# v16 - Default-template validation; no separate image built

There is intentionally **no v16 Docker image**.

v16 was not a source-code/container change. It was a configuration-only validation using the existing v15 image, so creating another image tag would have added no useful provenance.

The custom minimal no-thinking template was removed and the checkpoint/default template was used instead.

## Result

With the default checkpoint template:

```text
17 * 23 -> 391
```

and normal prompts remained coherent.

This established:

1. the v14 shared-expert reduction fix genuinely solved the TP3 numerical corruption;
2. the remaining short arithmetic oddity was tied to the custom minimal template experiment rather than TP3 execution;
3. the official/default checkpoint template should be the production/reference path.

The next version restored the official/default template and restored the `glm45` reasoning parser in the launcher.

---

# Continue with v17

Starting with v17, each major version is documented on its own page rather than extending this already long historical file.

**Next:** [V17.md](V17.md) - official-template/parser baseline, current 1M-context compose, KV-cache pool, CUDA-graph behavior, decode tuning, and Estonia / Estonia-long benchmarks.
