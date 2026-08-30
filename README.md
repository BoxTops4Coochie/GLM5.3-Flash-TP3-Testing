# GLM-5.3 Flash TP3 / SM120 Experiments

Experimental work to run **GLM-5.3 Flash** on **3 × NVIDIA Blackwell / SM120 GPUs** with vLLM.

This repository documents two separate development tracks:

| Track | Description | Status |
|---|---|---|
| [VTP1–VTP5](./VTP1-5.md) | Earlier generic / Transformers-based TP3 padding experiments | Stopped after VTP5 to focus on the native ModelOpt NVFP4 path |
| [Native GLM5Next TP3](./NATIVE-TP3.md) | Native `Glm5Next` TP3/EP3 implementation, image-by-image from base through v7 | Server starts successfully on pure ModelOpt NVFP4 checkpoints; inference correctness is still under investigation |

> **Development status:** This repository is intentionally sharing incomplete work. The native v7 line reaches a running OpenAI-compatible API server, but at least one tested checkpoint produced corrupted output. The purpose of publishing now is to let others reproduce the progress, inspect the patches, and continue debugging rather than rediscover the same TP3 blockers.

## Test hardware

- 3 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- 96 GB VRAM per GPU
- SM120
- Ubuntu 26.04
- CUDA 13.x container stack
- vLLM development build

The model does not fit the intended deployment on TP1/TP2, so the work here targets **TP3** specifically.

## Checkpoints considered

The available GLM-5.3 Flash checkpoints are **not interchangeable**. Different repositories use different quantization/export layouts, and that matters because the native patches touch KDA, MLA, ModelOpt loading, expert parallelism, and TP sharding.

| Checkpoint | Format / observed layout | Tested in native v7? | Result / why used or not used |
|---|---|---:|---|
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` | Pure ModelOpt NVFP4 detected by vLLM | Yes | **Preferred clean NVFP4 test checkpoint.** Loads all 181 GB, starts TP3/EP3, allocates ~24.82 GiB KV cache/GPU, and reaches the API server. |
| `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` | ModelOpt NVFP4 with BF16/FP32 exclusions | Yes | Loads and starts on v7, but generated output was corrupted (`locklock...` / repeated punctuation). Kept as a secondary diagnostic checkpoint rather than the primary starting point. |
| `local-inference-lab/GLM-5.3-Flash-NVFP4-4p67` | Mixed ModelOpt metadata: FP8 + NVFP4 + W4A16_NVFP4 + MXFP8 | Yes | Loader reaches ~94% then fails on `layers.0.self_attn.in_proj_qkvgfab.weight_scale`. Current KDA BF16 workaround conflicts with this checkpoint's quantized fused-KDA metadata. |
| `brandonmusic/GLM-5.3-Flash-tr3-4bpw` | EXL3/MCG-packed routed experts + native BF16/FP32 remainder | No, not on the ModelOpt NVFP4 path | Not a ModelOpt NVFP4 checkpoint. Requires the EXL3 loader/kernel path (`trellis` / `suh` / `svh` / `mcg` tensors), so it was intentionally excluded from the newer native NVFP4 line. |
| `incoai/GLM-5.3-Flash-DFlash2` | DFlash2 variant | Not used as the primary native NVFP4 checkpoint | Different execution/quantization path from the pure ModelOpt NVFP4 target. |
| `local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8` | DFlash2 + MXFP8 | Not used as the primary native NVFP4 checkpoint | Different MXFP8/DFlash2 path; useful as a related implementation reference, but not an apples-to-apples NVFP4 checkpoint. |

## Why focus on ModelOpt NVFP4?

The newer image line is specifically trying to make the **native NVIDIA `Glm5Next` implementation + TP3 + EP3 + ModelOpt NVFP4** work on SM120.

That gives us:

- native `Glm5Next` architecture support
- native sparse MLA
- proper 288-expert EP3 topology
- ModelOpt NVFP4 routed experts
- KDA/Gated DeltaNet support
- fewer artificial changes to routed-expert geometry than the older generic VTP route

See [NATIVE-TP3.md](./NATIVE-TP3.md) for the full base → v1 → … → v7 progression and runtime data.
