# GLM-5.3-Flash TP3 port — diffs against the Kraken base

Two patches, repository-relative paths, against exactly what ships in
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta`
@ `sha256:5927520c447fdcbc0990567f9237756ff66a54e360438915a5f70cd5cf9d530d`:

| Patch | Base | Files | Lines |
| --- | --- | ---: | ---: |
| [vllm-tp3-20260923.patch](vllm-tp3-20260923.patch) | vLLM `0.1.dev21537+g67bb922f6` (commit `67bb922f6`) | 28 (3 new) | +1966 / -157 |
| [b12x-tp3-20260923.patch](b12x-tp3-20260923.patch) | b12x `1.3.0` as installed in that image (no commit recorded; base-file sha256s below) | 7 | +38 / -23 |

```bash
git -C vllm apply vllm-tp3-20260923.patch   # at 67bb922f6
git -C b12x apply b12x-tp3-20260923.patch
```

Both were checked with `git apply` against the image's own copies of the base
files; applying them reproduces the files in the released image byte for byte.
This is the full TP3 port (R34 port to Kraken, 2026-09-19..21) plus the
2026-09-23 performance work, i.e. everything in
`azallaza/glm53-kraken-tp3:20260923`.

Hardware: 3x RTX PRO 6000 Blackwell (SM120, PCIe, no NVLink), TP3 / EP3 / DCP1,
checkpoint `local-inference-lab/GLM-5.3-Flash-NVFP4` @ `175ae8ce`.

## TP3 geometry and padded loading

GLM-5.3-Flash is dimensioned in powers of two; TP3 pads to the next
divisible size and loads the logical checkpoint into the physical layout with
zeroed tails. Keyed on GLM-5.3 at `tensor_parallel_size == 3`; other models and
TP sizes are unchanged.

| File | Lines | Change | Scope |
| --- | ---: | --- | --- |
| `vllm/transformers_utils/configs/glm53_tp3.py` | +217/-0 | New. Physical TP3 geometry for the target and MTP/DFlash drafts: attention/KV heads 64->72, KDA heads 64->66, shared-expert intermediate 2048->2112, vocab storage 154944; logical sizes kept as `original_*`. Requires EP (routed width 2048 is not divisible by 3). | GLM-5.3 at TP3 only; exact no-op otherwise |
| `vllm/config/vllm.py` | +8/-0 | Applies the TP3 geometry from ParallelConfig before generic parallel-shape validation. | GLM-5.3 TP3 |
| `vllm/config/speculative.py` | +85/-0 | Applies the target's TP3 contract to MTP/DFlash draft configs; MTP follows the target MoE topology, DFlash (dense) does not inherit EP/DCP. | GLM-5.3 TP3 |
| `vllm/model_executor/layers/linear.py` | +573/-74 | Strict padded weight loading: `loaded_output_size`/`loaded_input_size`, direct destination copies, zeroed tails, rejection of truncated checkpoints, packed-storage factors for quantized weights. | opt-in per layer; unchanged when the kwargs are absent |
| `vllm/model_executor/parameter.py` | +150/-40 | `load_tensor_parallel_weight` with validated shard bounds and packed input-storage factors, used by the padded loaders. | opt-in |
| `vllm/model_executor/layers/quantization/modelopt.py` | +3/-0 | Declares input-dim storage factors (NVFP4 packed x2, block scales x group_size, MXFP8 scales x32) for the padded loaders. | metadata only |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` | +2/-0 | Same storage-factor declarations for the uncensored (compressed-tensors) checkpoint. | metadata only |
| `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` | +145/-17 | KDA padded-head loading: logical (64) vs physical (66) heads, rank-local tail loaders for `dt_bias`, `A_log` and the fused conv1d weights. | GLM-5.3 TP3 (`glm53_tp3_padding`) |
| `vllm/models/glm5next/nvidia/attention.py` | +20/-0 | MLA q_b/kv_b/o_proj load the 64 logical heads into 72 physical. | GLM-5.3 TP3 |
| `vllm/models/glm5next/nvidia/model.py` | +51/-1 | TP3 padding in the decoder, shared expert and head; hooks for the low-latency GEMM plan and the FP8 decode path. | GLM-5.3; FP8 hook is env-gated off |
| `vllm/models/glm5next/nvidia/mtp.py` | +59/-9 | MTP draft TP3 projection/head padding; hooks for the GEMM plan and FP8 draft path. | GLM-5.3; FP8 hook env-gated off |
| `vllm/models/glm5next/common/multimodal.py` | +86/-5 | Vision tower TP3 (weights mode): heads 16->18, padded merger MLP sharded, the 4096-wide projection replicated. | GLM-5.3 TP3 weights-mode vision |
| `vllm/model_executor/models/qwen3_dflash.py` | +81/-7 | DFlash2 draft TP3 head/vocab padding (heads 32->36, KV 8->9). | GLM-5.3 TP3 target |
| `vllm/model_executor/models/qwen3_dflash2.py` | +3/-2 | Uses the draft vocab size helper instead of the target's padded vocab. | GLM-5.3 TP3 target |
| `vllm/v1/spec_decode/dflash.py` | +13/-0 | DFlash drafter at TP3 uses the draft parallel config with the target's rank. | GLM-5.3 TP3 |
| `vllm/v1/engine/core.py` | +22/-1 | `_sync_speculative_draft_dp_identity`: copies the target's data-parallel index/rank/local rank into the draft parallel config before the engine core is built. | speculative decoding |

## Fixes (several are not TP3-specific)

| File | Lines | Change | Scope |
| --- | ---: | --- | --- |
| `vllm/v1/worker/mamba_utils.py` | +4/-1 | **Generic bug fix.** Padded rows in aligned recurrent state indices used -1; convolution/GDN/KDA consumers expect NULL_BLOCK_ID 0. Under full-graph replay (seen with MTP0) -1 addressed memory before the state pool. | all hybrid models with padded rows |
| `vllm/model_executor/warmup/flashinfer_autotune_cache.py` | +45/-0 | **Generic bug fix.** Saves the union of every rank's FlashInfer autotune entries. Leader-only saves miss EP-rank-specific MoE keys; on warm start rank 0 skips tuning while peers enter synchronized profiling and wait forever. | EP > 1 with a saved tuning cache |
| `vllm/model_executor/warmup/kernel_warmup.py` | +14/-1 | Uses a fresh versioned cache name and the union save above. | EP > 1 |
| `vllm/v1/attention/ops/dcp.py` | +9/-2 | **Generic bug fix.** `tl.arange` needs a power-of-two extent: round the LSE rank vector up and mask the extra lanes. DCP world size 3 failed to compile. | DCP with non-power-of-two world size |
| `vllm/parser/engine/parser_engine.py` | +2/-1 | Honour an explicit `tool_choice="none"` even when no tools list is sent (the model otherwise emits tool calls into an answer that parses empty). | all models |
| `vllm/distributed/device_communicators/cuda_communicator.py` | +5/-1 | RoCE all-reduce now also initialises as a fallback when the B12X PCIe communicator was constructed but came up disabled; the previous `elif` skipped RoCE whenever the PCIe branch was entered. | B12X PCIe + RoCE builds |
| `vllm/v1/core/kv_cache_utils.py` | +1/-4 | Split target/recurrent block sizing for GLM5Next MLA is always on instead of behind `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE`. | GLM5Next |
| `vllm/v1/worker/utils.py` | +84/-0 | Startup runtime proof for the qualified GLM-5.3 TP3 backend set (EP3, B12X PCIe one-shot, B12X KDA decode, flashkda or b12x KDA prefill, weights-mode vision); enforced with `GLM53_TP3_REQUIRE_RUNTIME_PROOF=1`. | GLM-5.3 TP3; logs otherwise |
| `vllm/v1/worker/gpu_model_runner.py` | +3/-0 | Calls the runtime proof after model load. |  |
| `vllm/v1/worker/gpu/model_runner.py` | +3/-0 | Same for the V2 model runner. |  |

## Performance

| File | Lines | Change | Scope |
| --- | ---: | --- | --- |
| `vllm/models/glm5next/nvidia/glm53_low_latency_gemm.py` | +119/-0 | New. Routes measured (N, K, M<=8) BF16 projections to the CuTe skinny GEMM where it beat cuBLAS by >3% at TP3 shapes (+1.3-1.7% decode at R34; re-validated on this base). | GLM-5.3 TP3 shapes, SM120; `VLLM_GLM53_LOW_LATENCY_GEMM=0` disables |
| `vllm/models/glm5next/nvidia/glm53_fp8_dense.py` | +316/-0 | New. FP8 E4M3 weight-only decode (Marlin W8A16, per-output-channel scales) for the large BF16 projections, LM head, dense FFN and MTP draft at <=32 rows; optional CUTLASS FP8 W8A8 prefill. +11.4% C1 / +8.1% C8 decode with the b12x PDL change and a 256 KiB one-shot cutoff; qualified for default/MTP3/DCP1. | `VLLM_GLM53_FP8_DENSE=1` (default off) |

## b12x

| File | Lines | Change | Scope |
| --- | ---: | --- | --- |
| `b12x/comm/pcie/pcie_oneshot.py` | +1/-1 | World size 3 added to `SUPPORTED_WORLD_SIZES` (constants only; the kernels are world-size generic). | TP3 |
| `b12x/comm/pcie/pcie_dma.py` | +1/-1 | World size 3 added to the DMA all-reduce. | TP3 |
| `b12x/comm/pcie/_dma_kernels.py` | +1/-1 | World size 3 accepted by the DMA compile guard. | TP3 |
| `b12x/comm/pcie/_tuning.py` | +15/-15 | World size 3 added to the prepared-surface world-size tables (one-shot, DMA, DCP all-to-all). | TP3 |
| `b12x/comm/pcie/pcie_dcp_a2a.py` | +1/-1 | World size 3 added to DCP all-to-all. **Enabled by constant only, not validated** -- DCP3 here uses vLLM's ag_rs backend, not a2a. | TP3 |
| `b12x/comm/pcie/_oneshot_cute.py` | +13/-1 | PDL producer: the pull kernel calls `griddepcontrol.launch_dependents` after its cross-GPU barrier when `B12X_PCIE_ONESHOT_PDL=1`; flag added to the compile key (version 4->5). In 1.3.0 nothing read that flag. | any world size, flag-gated |
| `b12x/norm/mhc/_kernels.py` | +8/-3 | PDL consumer: `MHCPostPrePartialKernel` launches with `use_pdl` and `griddepcontrol_wait`s at entry when `B12X_MHC_PDL=1` (it had neither, although it is what consumes each all-reduce). Compile-key versions bumped. | flag-gated |

## Runtime settings outside the code

- `additional_config: {"kda_prefill_backend": "flashkda", "glm53_kda_decode_backend": "b12x"}`.
  The glm53-flash profile ships `kda_prefill_backend: b12x`, which
  `resolve_kda_prefill_backend` will not pick under `auto`. On 826K-token prompts
  it degraded long-generation output (repetition / thin answers; one failure in
  four on uncensored + DFlash2, and uncensored/MTP3 answers at 38% of normal
  length); flashkda fixed both at no measured prefill cost.
- `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=256KB`: at C8 each all-reduce is
  32 x 4096 x BF16 = 256 KiB, over the 84 KiB default, and fell back to the NCCL
  ring at 44.9 us versus the one-shot's 16.1 us. +3.0% C8 steps/s.
- `VLLM_GLM53_FP8_DENSE=1`, `VLLM_GLM53_FP8_PREFILL=w8a8` for the FP8 paths.
- The lil profile was not modified (its hashes are pinned by the image contract).

## Findings that may matter upstream

- **`B12X_PCIE_ONESHOT_PDL` is dead in b12x 1.3.0** and the all-reduce's consumer
  (`MHCPostPrePartialKernel`) has no PDL wait, so the profile's two PDL flags
  produce 0% overlap (0 of 41,860 all-reduce -> mHC pairs in a C1 trace). The
  b12x patch makes it 98.8% and +0.4-0.6% C1. The TP4 evidence doc
  (`glm53_pcie_mhc_dependent_launch_20260901.md`) came from a branch whose
  one-shot change is not in this build.
- **At world size 3, most of the all-reduce time is EP load imbalance, not
  transport.** One-shot all-reduce after attention: 6.0 us median. After MoE:
  15-19 us median, 26-28 us mean -- ranks spinning in the barrier for the one
  that received the most experts. ~1 ms per C1 step (~8%). Per layer the
  hottest rank is 12% over the mean systematically and the step-to-step spread
  is 26%. World size 3 also gets none of the TP2/TP4/TP8 remote-push transports.
- `modelopt_mixed` leaves attention/KDA projections BF16 (~3.7 ms of a 13.1 ms
  C1 step); FP8 weight-only for the large ones is the single largest decode win
  found (+8.6% C1 alone) with no measured quality loss on lavd / hotel-lights /
  128K-900K retrieval / 826K long generation.

## Base-file sha256 (the files the patches expect)

```
1d01269822a1a32b47fba1670d6aa02ab5d36da0750749041c644839df29ce15  b12x/comm/pcie/_dma_kernels.py
d595e45a9eba95610a2df0685e4d22eb933eecb77818c277509e44e12dc8a08b  b12x/comm/pcie/_oneshot_cute.py
9182e5e4fd6652a297a11dfd761b8eae9e6bb16791aa9ac1e9cb9929071bd770  b12x/comm/pcie/_tuning.py
c519c5b9118a3bd0319044816fab44003226c83743e672fe4cc5fb0e4969e3cc  b12x/comm/pcie/pcie_dcp_a2a.py
20076abe48f53084e580c95f573a9758fc7ec9b5d8cfca58a8e6a0fbc68fa7ef  b12x/comm/pcie/pcie_dma.py
ad4b47f2b45c2366f5bddc7742953c5b36bca25b01bb6ce42a1cc11334845f8a  b12x/comm/pcie/pcie_oneshot.py
91a574e16dcdc978716f87f5deb49d2682442258653b4f8fefcf1236a4dccc06  b12x/norm/mhc/_kernels.py
4539dc4e5f8ae6f6af7ee705cfdfb1bbe8628489486b5f815c595d9ced171998  vllm/config/speculative.py
c53a4c3fa12443b301bc0dfdcb20f4dadecd7c924d6748b91f23a3ff5bee25ac  vllm/config/vllm.py
138c5ed01f010bf42937a745a8be3aa6ae5ebdbd09b430239d0ab26fdcbbe66d  vllm/distributed/device_communicators/cuda_communicator.py
6be929f5ae4b1f12a970bd630f272550931a1caca7335388a49c521dd7e3a817  vllm/model_executor/layers/linear.py
d485cad73dac6b6274ebbd2cb88d86ebac8b5c4e9fe807a2c7402d39fa145e43  vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
17c515ce7d5f3e8464ce70674f542facff430c083c1fd6a52831b322c14e737c  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py
9bd665fdb4885dc8de21bb7b10f5c309d7a54e84949efb4dc17fcac88801d5a1  vllm/model_executor/layers/quantization/modelopt.py
3ac6b916df501cd04d1dbd806a53f9e75c91e4e9f755c2cb908c8f91a139c1c9  vllm/model_executor/models/qwen3_dflash.py
e751759734f6859c89d68be29d0e90743f86830b378c3a41a97692fd825be495  vllm/model_executor/models/qwen3_dflash2.py
797fe8e400be45f61670a4674c2c7891ec8cfed529e0fe493a02ef49df6a9a61  vllm/model_executor/parameter.py
be537ec70c49f25a5328110ac3c137e624b21752149e8bcf6972ca5b133ab252  vllm/model_executor/warmup/flashinfer_autotune_cache.py
804925ae11644396d338b7661327448d4809d1730df4d7a11fa9fd01517bf09a  vllm/model_executor/warmup/kernel_warmup.py
c9681e4ae2367cd4ef111bd7ca2f7736dc2644f697dbd64e6d63bc856d8cfa0e  vllm/models/glm5next/common/multimodal.py
73197be525688a023b9ba1d1cb0e1aa64d708117a48e01a4643618ed793bba94  vllm/models/glm5next/nvidia/attention.py
new file  vllm/models/glm5next/nvidia/glm53_fp8_dense.py
new file  vllm/models/glm5next/nvidia/glm53_low_latency_gemm.py
04b45bc183fe53f3636baf6e680c6bfd18d6bf6a414a995016fa23119f534ce8  vllm/models/glm5next/nvidia/model.py
468e7f4e122ce7f24d8f3f67856a2f4bc51d81e7951854750491b5ca2e3a18ef  vllm/models/glm5next/nvidia/mtp.py
7825e6d9c701f967cf3087d0040e0b0939fd452616649676bf8fc5ec1952b2a9  vllm/parser/engine/parser_engine.py
new file  vllm/transformers_utils/configs/glm53_tp3.py
1353f04b44ac3d9480179fdd5d51b5228a4f39fdc9d3d43519fd1d8731346deb  vllm/v1/attention/ops/dcp.py
fe3b5c0d4584521082660edea4d2a88c9713d93ec0649190910dd9c234cfe818  vllm/v1/core/kv_cache_utils.py
205f2c3d965e75c9ee6de442766eb34c9df4753582ea8c39a594ce52c0704877  vllm/v1/engine/core.py
076a7cef13a781a6d2cc2d40f3cd6f9043ee4b64a00a63ba2ece7866b86319da  vllm/v1/spec_decode/dflash.py
5612c3f8cd668fa113af1c4069f87d3d3eefaa19d31d5b92ce27e3de2800b8e1  vllm/v1/worker/gpu/model_runner.py
8fa6fe9a6bfbb3e9e8d6eca47c84ce657fb10e38639dfa40ff3abeb4e87e856d  vllm/v1/worker/gpu_model_runner.py
2e84a5269dccfce0c0e05f931af9b526e098c5ee8c9d3a9f66b38fc25c78e72c  vllm/v1/worker/mamba_utils.py
260042152d342e4a9202149ebfe9d5cd3ddbf4eed1272f86d6e2faebbf67c057  vllm/v1/worker/utils.py
```
