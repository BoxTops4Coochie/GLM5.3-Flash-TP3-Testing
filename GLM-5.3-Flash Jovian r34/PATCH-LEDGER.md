# R30 to R34 TP3 patch ledger

All ported existing Python files are byte-identical between the pristine R30
and R34 bases; three-way application retained the complete local R30 delta.
New upstream files and untargeted changes remain from R34.

| Component | Path | Disposition |
| --- | --- | --- |
| vllm | `tests/config/test_glm53_tp3_geometry.py` | NEW_FILE |
| vllm | `tests/models/test_glm53_tp3_model.py` | NEW_FILE |
| vllm | `tests/models/test_glm5next_vision_tp3.py` | NEW_FILE |
| vllm | `tests/parser/engine/test_replay.py` | MERGED |
| vllm | `tests/v1/worker/test_gpu_boundary_checkpoint.py` | NEW_FILE |
| vllm | `vllm/config/speculative.py` | MERGED |
| vllm | `vllm/config/vllm.py` | MERGED |
| vllm | `vllm/distributed/device_communicators/cuda_communicator.py` | MERGED |
| vllm | `vllm/model_executor/layers/linear.py` | MERGED |
| vllm | `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` | MERGED |
| vllm | `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` | MERGED |
| vllm | `vllm/model_executor/layers/quantization/modelopt.py` | MERGED |
| vllm | `vllm/model_executor/models/qwen3_dflash.py` | MERGED |
| vllm | `vllm/model_executor/models/qwen3_dflash2.py` | MERGED |
| vllm | `vllm/model_executor/parameter.py` | MERGED |
| vllm | `vllm/models/glm5next/nvidia/attention.py` | MERGED |
| vllm | `vllm/models/glm5next/nvidia/model.py` | MERGED |
| vllm | `vllm/models/glm5next/nvidia/mtp.py` | MERGED |
| vllm | `vllm/models/glm5next/nvidia/multimodal.py` | MERGED |
| vllm | `vllm/parser/engine/parser_engine.py` | MERGED |
| vllm | `vllm/transformers_utils/configs/glm53_tp3.py` | NEW_FILE |
| vllm | `vllm/v1/core/kv_cache_utils.py` | MERGED |
| vllm | `vllm/v1/attention/ops/dcp.py` | MERGED (DCP3 build; masked `_correct_attn_cp_out_kernel` for world size 3) |
| vllm | `vllm/v1/engine/core.py` | MERGED |
| vllm | `vllm/v1/spec_decode/dflash.py` | MERGED |
| vllm | `vllm/v1/worker/gpu/model_runner.py` | MERGED |
| vllm | `vllm/v1/worker/gpu_model_runner.py` | MERGED |
| vllm | `vllm/v1/worker/utils.py` | MERGED |
| b12x | `b12x/comm/pcie/pcie_dma.py` | MERGED |
| b12x | `b12x/comm/pcie/pcie_oneshot.py` | MERGED |
| b12x | `b12x/policy/generation/attention_corpus.py` | MERGED |
| b12x | `b12x/policy/generation/moe_corpus.py` | MERGED |
| b12x | `b12x/policy/generation/providers/blockscaled.py` | MERGED |
| b12x | `tests/comm/test_pcie_dma_kernels.py` | MERGED |
| b12x | `tests/comm/test_pcie_hierarchical.py` | MERGED |
| b12x | `tests/comm/test_pcie_oneshot_torture.py` | MERGED |
| b12x | `tests/policy/test_glm53_tp3_port.py` | NEW_FILE |

Additional artifacts: qualified attention.gdn profile component carried with
identical base component/kernel contract; latest configurable Compose and TP3
launcher ported; R34 dispatcher branches TP3 to that launcher. The other profile
components, native libraries and upstream launchers remain from R34.

The narrow explicit-none parser fix and its tests were local post-publication
R30 changes. Included and revalidated (3,009 replay tests). This does not claim
to repair reasoning-only output or model degeneration. Reproducer tools/results
remain host-side and are not part of the serving-image overlay.

Rejected MTP/L2/MoE experiments and diagnostic state-tracing mounts are not
production changes and were not imported. No new tuning is being performed.
