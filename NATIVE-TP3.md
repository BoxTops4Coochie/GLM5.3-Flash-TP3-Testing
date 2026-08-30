# Native GLM5Next TP3 / EP3 — Base through v7

This document covers the newer GLM-5.3 Flash TP3 line based on the **native NVIDIA `Glm5Next` implementation in vLLM**.

The goal is:

```text
TP = 3
EP = 3
SM120
ModelOpt NVFP4
native sparse MLA
native KDA / Gated DeltaNet
```

The current image line reaches a running API server. Correctness is still being investigated.

---

# Base image

Base:

```text
voipmonitor/vllm:glm53-dflash2-mxfp8-dev-20260828-4
```

Runtime source:

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

Package:

```text
0.26.1rc0+glm53.flash.nvfp4.luke.clean.r1.vllme75bcfd.b12x58a046f
```

Approximate stack:

```text
CUDA 13.3
PyTorch 2.13
Transformers 5.15
NCCL 2.31.2
FlashInfer 0.6.18+cu133
```

The runtime overlay from the image was captured and committed locally before the TP3 changes so the experiments could be reproduced against a known base.

---

# Image progression

## v1 — Native GLM5Next TP3 starting point

### Change from base

The first native image moved the effort away from the generic Transformers/VTP route and into the native:

```text
Glm5NextForConditionalGeneration
```

implementation.

Initial TP3 padding helpers and native GLM5Next modifications were introduced.

### Goal

Make the native model construct under TP3 rather than immediately rejecting dimensions that are not divisible by three.

### Important takeaway

This established the native code path as the new development base, but generic vLLM model validation still prevented the full TP3 configuration.

---

## v2 — TP divisibility validator bypass for Glm5Next

### Change from v1

Patched:

```text
vllm/config/model.py
```

so the generic attention-head divisibility check is skipped specifically for native `Glm5Next` architectures.

Relevant architectures:

```text
Glm5NextForCausalLM
Glm5NextForConditionalGeneration
Glm5NextMTPModel
```

### Why

The generic validator sees:

```text
64 attention heads % TP3 != 0
```

and normally aborts before the native model can apply its own padded physical geometry.

### Result

Construction advanced past the generic model-config validator.

The next blocker came from the multimodal/vision path, whose own head geometry was also incompatible with TP3.

---

## v3 — Text-only / vision-tower bypass

### Change from v2

Added a text-only construction path for:

```text
Glm5NextForConditionalGeneration
```

When:

```text
multimodal_config.language_model_only = true
```

the model skips the vision tower and uses a missing-layer placeholder instead.

### Why

The vision configuration contains TP-incompatible geometry, including a 16-head configuration:

```text
16 % 3 != 0
```

For this project, vision was not needed; the target is language-only serving.

### Result

The model advanced beyond the prior vision-tower TP3 failure.

The next blocker was the shared expert.

---

## v4 — Replicate the shared expert; keep routed experts on EP3

### Change from v3

Added a `disable_tp` path to `Glm5NextMLP`.

The shared expert is constructed with TP disabled / replicated on each GPU, approximately:

```text
disable_tp = true
reduce_results = false
```

The routed experts remain under EP3.

### Why

Shared expert dimensions:

```text
hidden_size = 4096
intermediate_size = 2048
```

The 2048 intermediate width is not divisible by three.

Instead of padding and changing the shared-expert mathematics, the shared expert is replicated.

### Routed-expert result

Native EP3 cleanly maps:

```text
288 global routed experts
-> 96 experts/GPU
```

with routed expert intermediate size remaining:

```text
2048
```

No routed-expert width padding is required.

### Additional backend finding

Forcing:

```text
--moe-backend b12x
```

was not compatible with this EP3 arrangement.

The runtime was changed to:

```text
--moe-backend auto
```

which later selected a compatible NVFP4 backend.

---

## v5 — Weight-loader diagnostics

### Change from v4

Added diagnostic instrumentation around native GLM5Next weight loading.

### Why

Model construction now advanced far enough that the remaining blocker was a checkpoint/runtime parameter mismatch.

### Key diagnostic

The loader exposed a failure around:

```text
layers.0.self_attn.in_proj_qkvgfab.weight
```

for the checkpoint's `b_proj` shard.

Observed diagnostic geometry included:

```text
loaded_shape = (66, 4096)
runtime param shape = (8598, 2048)
shard_id = 3
TP = 3
```

### Interpretation

The source BF16 KDA tensors were being padded correctly, but the fused runtime KDA parameter had been created with NVFP4-packed geometry.

That meant the fused runtime parameter's quantization treatment was wrong.

---

## v6 — Keep KDA fused projections BF16

### Change from v5

The GLM-5.3 KDA checkpoint tensors are BF16, but the shared Kimi/GDN implementation fuses several projections into:

```text
self_attn.in_proj_qkvgfab
```

The checkpoint ModelOpt ignore rules refer to the original unfused names and therefore did not match this fused runtime name.

v6 creates a shallow copy of the vLLM config specifically for KDA and disables quantization there:

```python
kda_vllm_config = copy(vllm_config)
kda_vllm_config.quant_config = None

super().__init__(physical_config, kda_vllm_config, prefix)
```

### Result

This was a major milestone.

The primary ModelOpt NVFP4 checkpoint could now:

- load the entire checkpoint
- initialize model memory
- create cache
- capture CUDA graphs
- proceed into warmup

### KDA TP3 geometry

Logical KDA geometry:

```text
64 heads
head_dim = 128
projection width = 8192
```

TP3 physical geometry:

```text
64 -> 66 heads
66 / 3 = 22 heads/GPU

8192 -> 8448 physical projection width
8448 / 3 = 2816 local width
```

The loader pads the relevant checkpoint dimensions before standard TP sharding.

### New blocker

Warmup reached the SM120 B12X sparse-MLA kernel and failed with:

```text
ValueError:
SM120 sparse MLA prefill requires heads divisible by 8, got 22
```

That established that the **local MLA head count**, not merely the global count, has an additional SM120 kernel constraint.

---

## v7 — MLA-specific 64 → 72 → 24/GPU padding

### Change from v6

v6 used generic TP3 padding for MLA:

```text
64 -> 66 -> 22/GPU
```

but SM120 B12X sparse MLA requires:

```text
local_heads % 8 == 0
```

v7 added an MLA-specific helper using:

```python
lcm(tp_size, 8)
```

At TP3:

```text
lcm(3, 8) = 24
```

so MLA now becomes:

```text
64 logical heads
-> 72 physical heads
-> 24 heads/GPU
```

KDA intentionally remains:

```text
64 -> 66 -> 22/GPU
```

because KDA did not expose the same local-head-divisible-by-8 requirement.

### MLA checkpoint padding

With 72 physical MLA heads:

```text
q_b:
64 * 256 = 16384
-> 72 * 256 = 18432

kv_b:
64 * 512 = 32768
-> 72 * 512 = 36864

o_proj input:
16384 -> 18432
```

These dimensions are zero-padded before the normal TP parameter loaders run.

### Result

v7 clears the previous SM120 sparse-MLA warmup failure and starts the API server.

This is the first image in the native line that completes the full startup path under TP3/EP3.

---

# Files changed by the native TP3 work

```text
vllm/models/glm5next/nvidia/tp_padding.py
vllm/models/glm5next/nvidia/attention.py
vllm/models/glm5next/nvidia/kda.py
vllm/models/glm5next/nvidia/model.py
vllm/config/model.py
```

`tp_padding.py` is new.

---

# Current representative runtime configuration

```text
TP = 3
EP = enabled
PP = 1
DP = 1

dtype = bfloat16
quantization = modelopt_mixed / modelopt_fp4
load format = instanttensor

attention backend = B12X
linear backend = b12x
MoE backend = auto

KV cache dtype = fp8
max model length = 32768
max sequences = 1
max batched tokens = 2048

prefix caching = enabled
chunked prefill = enabled
language-model-only = true
disable custom all-reduce = true
```

With world size 3, the tested environment falls back to:

```text
PYNCCL
```

for TP and EP collectives.

---

# Fully loaded NVFP4 checkpoint results

The native v7 image was successfully brought all the way to a running API server with **two independent pure ModelOpt NVFP4 checkpoints**. These two runs are the main focus of this document because they exercise the full native TP3/EP3 path.

Both checkpoints load successfully and expose very similar GPU/KV characteristics, but both currently generate corrupted output. That makes the shared runtime/TP3 implementation the primary debugging target rather than either checkpoint individually.

## 1. LibertAIDAI/GLM-5.3-Flash-NVFP4

This is the preferred primary checkpoint for documenting the native ModelOpt NVFP4 path.

It fully loads, starts the server, and achieved the higher observed decode speed of the two successful NVFP4 runs:

```text
~96 tok/s via curl
```

The runtime identifies it as:

```text
Detected ModelOpt NVFP4 checkpoint (quant_algo=NVFP4)
```

and resolves:

```text
Glm5NextForConditionalGeneration
```

with:

```text
TP3
EP3
96 / 288 routed experts per EP rank
FLASHINFER_CUTLASS NVFP4 MoE backend
fp8_ds_mla KV cache format
```

### Weight loading

Checkpoint payload:

```text
~181 GB
```

InstantTensor progress reached approximately:

```text
11.9 GB/s
```

during the fastest portions of loading.

Reported total weight-load time:

```text
23.39 s
```

Reported model load memory/time per rank:

```text
60.81 GiB
~25.83–25.87 s
```

### GPU memory / cache

Startup memory observations:

```text
GPU total usable reported: ~94.97 GiB
free at startup: ~93.79–93.84 GiB

weights + non-torch consumed:
~61.93–61.97 GiB/GPU

peak activation:
~3.44 GiB/GPU

actual CUDA graph pool:
~0.09 GiB/GPU
```

Available KV cache:

```text
~24.82 GiB
```

Current per-rank KV use during sizing:

```text
~24.81–24.85 GiB
```

vLLM reported that fully utilizing the remaining GPU memory could allow approximately:

```text
~28.15–28.24 GiB KV cache/GPU
```

with an explicit `--kv-cache-memory` value.

### KV token capacity

```text
GPU KV cache size:
2,423,107 tokens
```

At:

```text
max_model_len = 32768
```

reported theoretical maximum concurrency was:

```text
73.95x
```

The cache setup also reports:

```text
10 padding layers
up to 29.41% potential KV cache waste
```

due to aligning the Mamba and attention cache/page requirements.

### Graphs / warmup

Initial graph capture:

```text
~6 s
~1.66 GiB estimated capture allocation
```

Later decode graph capture:

```text
~2 s
~0.09 GiB actual graph pool
```

Full graph mode is not supported by the GDN backend, so vLLM automatically falls back to:

```text
FULL_DECODE_ONLY
```

Total engine initialization:

```text
68.97 s
```

### Inference-side observations

The first requests triggered additional TileLang/Triton JIT compilation, so early latency and throughput are not representative steady-state numbers.

Observed vLLM generation-throughput log samples included:

```text
6.4 tok/s
12.8 tok/s
66.2 tok/s while a request was actively running
36.2 tok/s in the following reporting window
```

These are **server reporting-window samples**, not a controlled benchmark.

The notable peak active-request log was:

```text
Avg generation throughput: 66.2 tokens/s
GPU KV cache usage: 0.4%
```

A direct curl generation test reached approximately:

```text
~96 tok/s
```

on the LibertAIDAI checkpoint.

This should still be treated as a preliminary single-request result rather than a controlled multi-run benchmark, but it is the best observed decode figure from the two fully loaded pure NVFP4 checkpoints so far.

### Correctness result

Although the LibertAIDAI checkpoint is a cleaner pure ModelOpt NVFP4 starting point and loads successfully, **its generated output was also broken/corrupted in the v7 TP3 runtime**.

This is an important A/B result because two independent pure ModelOpt NVFP4 checkpoints now:

```text
LibertAIDAI/GLM-5.3-Flash-NVFP4
dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4
```

both load and reach the API server under the same TP3/EP3 implementation, yet both produce invalid generation.

That makes a checkpoint-specific corruption problem substantially less likely and shifts the investigation toward the TP3 runtime modifications or the kernel/backend path.

### Important warning

The runtime still reports:

```text
w1_weight_scale_2 must match w3_weight_scale_2.
Accuracy may be affected.
```

This warning should remain part of the correctness investigation.

---

## 2. dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4

This checkpoint was the first one to make the entire native v7 startup path work, but it is now documented after the LibertAIDAI checkpoint because it is a less desirable primary reference model.

### Weight loading

Checkpoint:

```text
~181 GB
```

Observed InstantTensor load:

```text
21.11 s
```

Reported model memory:

```text
~60.81 GiB/GPU
```

### Cache / GPU memory

Available KV cache:

```text
~24.82 GiB/GPU
```

GPU KV token capacity:

```text
2,423,107 tokens
```

At 32768 context:

```text
~73.95x theoretical maximum concurrency
```

The memory profile was effectively the same class as the LibertAIDAI run because both are similarly sized ModelOpt NVFP4 checkpoints under the same v7 geometry.

### Preliminary generation speed

A 1024-token request completed in approximately:

```text
13.3048 s
```

which gives a wall-clock approximation of:

```text
~76.96 tok/s
```

So the two fully loaded pure NVFP4 checkpoints currently compare approximately as:

```text
LibertAIDAI:  ~96 tok/s
dealignai:    ~77 tok/s
```

These are preliminary single-request figures rather than controlled multi-run benchmarks.

Because both checkpoints generated corrupted output, neither speed number should be treated as a final quality/performance result.

### Correctness failure

Raw completion:

```text
The capital of France is
```

produced:

```text
locklocklocklocklock...
```

Chat with thinking disabled also produced repeated `lock`.

Sampling changed the attractor to repeated punctuation:

```text
!!!!!!!!!!!!!!!!!!!!
```

This demonstrates that the issue is below the chat-template/reasoning-parser layer.

The server and kernels are running, but the resulting logits are numerically wrong for this checkpoint/runtime combination.

---

# Other checkpoint variants

Other GLM-5.3 Flash checkpoint/export variants were inspected during development but did not complete the same end-to-end native v7 run.

They are intentionally **not detailed in this document**, which is focused on the two ModelOpt NVFP4 checkpoints that fully loaded and reached inference.

See the repository [README](./README.md) for the full model/checkpoint list and why the other variants were not used as primary native TP3 test cases.

---

# Known runtime warnings / limitations

These are currently expected in the tested TP3 environment:

```text
SymmMemCommunicator:
Device capability 12.0 not supported

FlashInfer All Reduce:
not supported for world_size=3
```

so TP and EP collectives use:

```text
PYNCCL
```

The GDN backend also forces:

```text
CUDAGraphMode.FULL
-> FULL_DECODE_ONLY
```

and first inference requests may trigger additional TileLang/Triton JIT compilation.

The base wrapper also exposes several stale EXL3-related environment variables that this vLLM build reports as unknown. These are noisy but were not the root cause of the startup transitions documented above.

---

# Current unresolved correctness questions

Because **both independent pure ModelOpt NVFP4 checkpoints load successfully, reach the API server, and then generate broken output**, the next debugging step should focus on separating **backend/kernel corruption** from **TP3 geometry/padding corruption**.

The fact that LibertAIDAI reached roughly **96 tok/s** while dealignai reached roughly **77 tok/s** shows that the runtime is performing real decode work at plausible speeds; the remaining issue is correctness, not simply failure to execute.

## Recommended next experiment: force the reference NVFP4 MoE backend

Keep the v7 image, LibertAIDAI checkpoint, TP3/EP3 settings, B12X attention, and every other runtime option unchanged.

Change only:

```text
--moe-backend auto
```

to:

```text
--moe-backend emulation
```

The purpose is not performance. `emulation` is expected to be much slower; it is being used as a numerical reference path.

Interpretation:

```text
emulation output is correct
    -> the TP3 model geometry is probably viable
    -> focus on FLASHINFER_CUTLASS / NVFP4 scale handling / EP3 MoE backend behavior

emulation output is still broken
    -> corruption is likely above the optimized MoE backend
    -> focus on MLA/KDA TP3 padding and logical-vs-physical head semantics
```

If emulation fixes correctness, the next backend A/B tests should be tried one at a time, for example:

```text
vllm_cutlass
flashinfer_trtllm
flashinfer_cutedsl
marlin
```

using whichever backends report support for the checkpoint and EP3 in the current build.

If emulation is also broken, the **highest-priority suspect becomes MLA 64 -> 72 padding**. The current implementation changes the model-visible physical head count in order to satisfy the SM120 sparse-MLA kernel. A more correct implementation may need to keep:

```text
logical MLA heads = 64
```

throughout model semantics and only create/mask temporary padded storage:

```text
kernel-facing physical heads = 72
```

at the B12X sparse-MLA boundary.

The major remaining investigation areas are:

1. **MLA padding semantics**

   The physical kernel representation is padded:

   ```text
   64 -> 72 heads
   ```

   to satisfy SM120's local 8-head multiple.

   The extra synthetic heads must be proven mathematically neutral through sparse indexing, top-k selection, normalization, and attention.

2. **KDA padding semantics**

   KDA uses:

   ```text
   64 -> 66 heads
   ```

   and needs numerical comparison against a known-good configuration.

3. **NVFP4 scale handling**

   Both pure NVFP4 runs report:

   ```text
   w1_weight_scale_2 must match w3_weight_scale_2.
   Accuracy may be affected.
   ```

4. **MoE backend behavior**

   `auto` selects:

   ```text
   FLASHINFER_CUTLASS
   ```

   for the pure ModelOpt NVFP4 checkpoints.

   Alternate backends remain useful for numerical isolation.

5. **Logical vs physical MLA head count**

   A better long-term implementation may need to preserve:

   ```text
   logical heads = 64
   ```

   throughout model semantics while presenting:

   ```text
   physical kernel heads = 72
   ```

   only at the kernel boundary.

---

# Useful next debugging work

High-value experiments include:

```text
- layer-by-layer activation comparison against a known-good TP configuration
- mask synthetic MLA heads before sparse/top-k/normalization operations
- verify padded KDA heads remain exactly neutral
- compare logits before/after the first MLA and KDA layers
- force alternate NVFP4 MoE backends
- audit ModelOpt scales after EP3 expert sharding
- inspect B12X sparse-MLA assumptions around logical vs physical heads
- extend warmup coverage to avoid first-request TileLang/Triton JIT spikes
```

This repository intentionally documents the incomplete state so other developers can continue from the current startup milestone rather than reproducing the same base → v7 sequence.
