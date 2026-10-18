# GLM-5.3-Flash TP3: upstream 034e6c20 vs ours (20260923)

- **a (base):** `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260924-034e6c20d80260d4`
  (vLLM `f6496cbd`, b12x `9001944d`)
- **b (ours):** `azallaza/glm53-kraken-tp3:20260923` (local `glm53-kraken-tp3:20260923`)

## All differences

Ours has, upstream lacks (TP3 port):

1. **FP8 LM head works.** Upstream's gate rejects the modelopt LM head's `UnquantizedLinearMethod`, so it silently stays BF16; fixed by the 2-line patch here.
2. **DFlash / DFlash2 at TP3.** Draft vocab/geometry padding and draft rank mapping; absent upstream, so `MODE=dflash2` at TP3 likely fails.
3. **MTP draft head.** Padded TP3 shared head, `eh_proj` sharded instead of replicated, skinny-GEMM entry `(1366, 8192)`.
4. **Padded NVFP4 loading.** Packed storage factors for modelopt / compressed-tensors; other NVFP4 checkpoints (uncensored) untested upstream.
5. **Draft data-parallel sync.** Draft copies the target's DP rank/index (`engine/core.py`).
6. **`tool_choice="none"`** honoured with tools present or explicitly set.
7. **Vision tower sharded** across GPUs with padded heads; upstream uses a full copy per GPU (likely part of its 122K lower KV).
8. **TP3 runtime proof.** Logs/asserts the resolved TP3 backends at startup; diagnostics.
9. **KV stride rounding** always on; upstream only with `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE`. Not a lever.
10. **b12x DCP all-to-all lists world size 3.** Unvalidated on ours too.

Upstream has, ours lacks (newer components):

11. **vLLM 241 commits newer** (`67bb922f` -> `f6496cbd`), incl. the RecoverSSM committed-state fix (`mamba_utils.py`) and a newer autotune cache. Compiled extensions identical.
12. **Newer b12x** (`9001944d`): IQ2/Q8 quant kernels, fused-MoE and embedding updates, compile-cache integrity check. Not used by GLM-5.3 at TP3.
13. **Newer LMCache** (`dev365` -> `dev398`): cache/eviction fixes, new L2 options; `cuda_ops` extension rebuilt.

Launcher / profiles (`/opt/lil`):

14. **Built-in `glm53-tp3` preset upstream** with our settings (FP8 dense, W8A8 prefill, 256KB cutoff, L2 budgets, flashkda prefill, EP, flashinfer_cutlass, MTP), capture ladder `1,2,4,8,12..32`. Our compose/launcher does not use it.
15. **GLM-5.3 profile speculative default** `mtp` upstream vs `off` in ours. No effect with our launcher (sets MTP).
16. **New launcher options upstream:** `hf-overrides`, `max-num-prefill-tokens-per-step`, `enable-expert-parallel`, `enable-prompt-tokens-details`.
17. **`tp3-source-delta.json`** in ours only (record of our source changes; no runtime effect).

FlashInfer is identical in both images.

## Files

| File | What it is |
| --- | --- |
| [034e6c20-fp8-lm-head-gate.patch](034e6c20-fp8-lm-head-gate.patch) | **Ready to apply to 034e6c20.** Item 1, see below. |
| [vllm-034e6c20-to-ours.diff](vllm-034e6c20-to-ours.diff) | Focused: the 28 vLLM files our TP3 port touches (+1831 / -1111). Items 2-9. |
| [b12x-034e6c20-to-ours.diff](b12x-034e6c20-to-ours.diff) | Focused: 2 b12x files (+11 / -12). Item 10. The other 5 b12x files of our port (ws3 one-shot/DMA, PDL) are identical upstream. |
| [full-vllm-034e6c20-to-ours.diff](full-vllm-034e6c20-to-ours.diff) | Complete vLLM Python tree: 136 files (+2414 / -5198). Items 2-9 and 11. |
| [full-b12x-034e6c20-to-ours.diff](full-b12x-034e6c20-to-ours.diff) | Complete b12x tree: 137 files (+5486 / -8834). Items 10 and 12. |
| [full-lmcache-034e6c20-to-ours.diff](full-lmcache-034e6c20-to-ours.diff) | Complete LMCache Python tree: 25 files (+179 / -1493). Item 13 (the rebuilt `cuda_ops` .so is binary, not in the diff). |
| [full-lil-034e6c20-to-ours.diff](full-lil-034e6c20-to-ours.diff) | Complete `/opt/lil` launcher/profiles: 15 files (+86 / -488). Items 14-17. |

All diffs run upstream (`a/`) -> ours (`b/`); `-` lines are upstream-only, `+`
lines ours-only. `__pycache__`, `.pyc`, `.so` and `.cubin` are excluded
(the only differing binary is LMCache's `cuda_ops` extension).

The focused diffs were checked with `git apply` on the image's own copies:
applied to 034e6c20 they reproduce our files byte for byte. **None of the diffs
are merge patches.** Upstream reimplemented the TP3 padding in
`vllm/model_executor/models/config.py` (not in these diffs), so applying ours
wholesale would pad twice, and the `-` lines also include upstream work newer
than our base (for example the RecoverSSM fix in `mamba_utils.py`, autotune
cache changes, `config/vllm.py` additions), which would be reverted. Use the
table below to find the hunks that matter.

## FP8 LM head gate (the one ready patch)

`enable_glm53_fp8_lm_head` requires `type(lm_head.quant_method) is
UnquantizedEmbeddingMethod`. With the modelopt checkpoint the excluded LM head
gets `UnquantizedLinearMethod`, so the FP8 LM head never engages; the only
sign is the startup warning "LM head is not an unquantized BF16 (51648, 4096)
shard". The patch accepts both; the wrapper already delegates to either.
Tested on 034e6c20: "LM head packed to Marlin FP8", 8/8 arithmetic on three
starts, KV -26K. Speed within upstream's restart-to-restart spread (C1
87.2-87.5 steps/s), so this is a correctness fix, not a measured speed gain
(`kraken-port/optimization/upstream-lmhead-20260924/RESULTS.md`).

## File by file

**Missing upstream (worth porting)**

| File(s) | Change in ours |
| --- | --- |
| `vllm/model_executor/models/qwen3_dflash.py`, `qwen3_dflash2.py`, `vllm/v1/spec_decode/dflash.py`, `vllm/config/speculative.py` | DFlash/DFlash2 draft at TP3: padded draft vocab (`_get_dflash_draft_vocab_size`), draft geometry, draft `ParallelConfig` rank taken from the target. Without it DFlash at TP3 is unsupported. |
| `vllm/models/glm5next/nvidia/mtp.py`, `glm53_low_latency_gemm.py` | MTP shared head with TP3 vocab storage (`_Glm53TP3SharedHead`), `eh_proj` as a padded `ColumnParallelLinear` (upstream: replicated `nn.Linear`), skinny-GEMM plan entry `(1366, 8192)` for it. |
| `vllm/model_executor/layers/quantization/modelopt.py`, `compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` (+ loader parts of `linear.py` / `parameter.py`) | Input-dim storage factors (NVFP4 packed x2, block scales x group size) so padded NVFP4 weights load correctly. Needed for other NVFP4 checkpoints at TP3 (uncensored untested upstream). |
| `vllm/v1/engine/core.py` | `_sync_speculative_draft_dp_identity`: copies the target's data-parallel index/rank into the draft parallel config. |
| `vllm/parser/engine/parser_engine.py` | `tool_choice="none"` honoured when tools are given or the field is set explicitly. |
| `vllm/models/glm5next/common/multimodal.py` | Vision tower sharded in weights mode with padded heads. Upstream switches `mm_encoder_tp_mode` to data (full tower per GPU), which likely costs KV (upstream has 122K fewer KV tokens with the same compose file). |
| `vllm/v1/worker/utils.py`, `gpu_model_runner.py`, `gpu/model_runner.py` | `log_glm53_tp3_runtime_proof` / `GLM53_TP3_REQUIRE_RUNTIME_PROOF`: logs and optionally asserts the resolved TP3 backend set. Diagnostics. |
| `vllm/v1/core/kv_cache_utils.py` | Ours rounds the GLM-5.3 block stride to the C4 index-page unit unconditionally; upstream only with `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE`. Stride padding only; not a speed or KV lever. |
| `b12x/comm/pcie/pcie_dcp_a2a.py`, `_tuning.py` | World size 3 listed for DCP all-to-all. Never validated on ours either. |

**Reimplemented upstream (do not port ours)**

| File(s) | Note |
| --- | --- |
| `vllm/transformers_utils/configs/glm53_tp3.py` (new in ours), `vllm/config/vllm.py`, TP3 parts of `config/speculative.py` | Upstream applies the TP3 geometry in `model_executor/models/config.py`. |
| `vllm/model_executor/layers/linear.py`, `parameter.py`, `models/glm5next/nvidia/attention.py`, `model.py`, `layers/mamba/gdn/kimi_gdn_linear_attn.py` | Padded loading via `loaded_*_size` kwargs in ours; upstream pads through its config hook and loaders. |
| `vllm/models/glm5next/nvidia/glm53_fp8_dense.py` | Upstream reworked our module (TP3-only guard, runtime-quantized heads left alone); only the LM head gate above is a gap. |
| `vllm/model_executor/warmup/flashinfer_autotune_cache.py`, `kernel_warmup.py` | Autotune union: upstream has its own, newer version. |
| `vllm/distributed/device_communicators/cuda_communicator.py` | RoCE fallback: upstream has an equivalent condition. |
| `vllm/v1/attention/ops/dcp.py` | Formatting only. |
| `vllm/v1/worker/mamba_utils.py` | Upstream-only change (RecoverSSM committed-state fix); ours predates it. |

## Measured (2026-09-24, same compose.yaml, DCP1, user's bench command)

| Steps/s | ours (2 starts) | 034e6c20 (3 starts) | 034e6c20 + LM head fix (3 starts) |
| --- | --- | --- | --- |
| C1, 0 / 32K / 128K | 88.2 / 88.4 / 86.7 | 88.6 / 88.4 / 86.9 (range 86.0-90.1 at 0) | 87.4 / 86.8 / 85.6 |
| C8, 0 / 32K / 128K | 281.9 / 284.1 / 274.2 | 276.9 / 277.2 / 268.7 | 277.4 / 273.2 / 268.3 |
| KV tokens | 3,169,983 | 3,049,718 | 3,023,362 |

One of the three plain-upstream starts (90.1 at C1/0) came from the first bracket,
which used temperature 1 and 45 s windows; the others and all other columns use
the user's command. Evidence: `kraken-port/optimization/upstream-tp3-20260924/`,
`upstream-tp3-usercmd-20260924/`, `upstream-lmhead-20260924/`.
