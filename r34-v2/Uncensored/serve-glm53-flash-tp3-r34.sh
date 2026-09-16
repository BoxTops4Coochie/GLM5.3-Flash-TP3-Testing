#!/usr/bin/env bash
set -euo pipefail
# Initial correctness envelope; scheduler and kernel defaults remain R34's.
fail() { printf '%s\n' "$*" >&2; exit 2; }
[[ ${TP:-3} == 3 ]] || fail 'TP3 launcher requires TP=3'
[[ ${DCP:-1} == 1 || ${DCP:-1} == 3 ]] || fail "TP3 supports DCP=1 or DCP=3 (decode-context ranks must divide TP=3), got ${DCP:-1}"
[[ ${CACHE_MODE:-vram} == vram && ${LMCACHE_ENABLED:-0} == 0 ]] ||
  fail 'TP3 initial qualification requires GPU-only cache'
[[ ${MM_ENCODER_TP_MODE:-weights} == weights ]] || fail 'TP3 requires weights vision mode'
(($# == 0)) || fail 'TP3 launcher accepts environment configuration only'
# CHECKPOINT selects which qualified checkpoint to serve. Each variant pins its
# own revision and the kernel backends its quantization requires; the launcher
# is the single source of truth and rejects a MODEL that disagrees with it.
#
#   default     the released GLM-5.3-Flash NVFP4 build (MIXED_PRECISION:
#               NVFP4 routed experts, MXFP8 MTP experts)
#   uncensored  orcarouter GLM-5.3-Flash-Uncensored-NVFP4. Same architecture and
#               tokenizer, but compressed-tensors W4A16 NVFP4 with no static
#               activation scales and an unquantized BF16 MTP layer, so it needs
#               the Marlin weights-only expert kernel and a Triton MTP kernel.
#               Qualified 2026-09-16: C1/C8 smokes, OCR, 17.6K/128K/900K
#               retrieval, 768K retained history, tool calls, DCP1 and DCP3.
export CHECKPOINT=${CHECKPOINT:-default}
case ${CHECKPOINT} in
  default)
    checkpoint_model=local-inference-lab/GLM-5.3-Flash-NVFP4
    checkpoint_revision=46aaae8a82032f77100f2f03e9cc11b391df3b4d
    ;;
  uncensored)
    checkpoint_model=orcarouter/GLM-5.3-Flash-Uncensored-NVFP4
    checkpoint_revision=ec0adf4f49c9570807cc11a5f650538c1893ae54
    export LOAD_FORMAT=${LOAD_FORMAT:-auto}
    export QUANTIZATION=${QUANTIZATION:-compressed-tensors}
    export MOE_BACKEND=${MOE_BACKEND:-marlin}
    export MTP_MOE_BACKEND=${MTP_MOE_BACKEND:-triton}
    ;;
  *) fail "CHECKPOINT must be default or uncensored, got ${CHECKPOINT}" ;;
esac
[[ ${MODEL:-${checkpoint_model}} == "${checkpoint_model}" ]] ||
  fail "CHECKPOINT=${CHECKPOINT} serves ${checkpoint_model}, got MODEL=${MODEL}"
[[ ${MODEL_REVISION:-${checkpoint_revision}} == "${checkpoint_revision}" ]] ||
  fail "CHECKPOINT=${CHECKPOINT} pins revision ${checkpoint_revision}, got ${MODEL_REVISION}"
export MODEL=${checkpoint_model}
export MODEL_REVISION=${checkpoint_revision}
export TP=3 DCP=${DCP:-1} LMCACHE_ENABLED=0 MM_ENCODER_TP_MODE=weights
export SPECULATOR=${SPECULATOR:-mtp}
case ${SPECULATOR} in
  mtp)
    depth=${MTP_DEPTH:-${MTP:-${NUM_SPECULATIVE_TOKENS:-0}}}
    [[ ${depth} == 0 || ${depth} == 3 ]] || fail 'MTP depth must be 0 or 3'
    ;;
  dflash|dflash2)
    export SPECULATOR=dflash2
    depth=${DFLASH_DEPTH:-${NUM_SPECULATIVE_TOKENS:-7}}
    [[ ${depth} == 7 ]] || fail 'DFlash2 depth must be 7'
    [[ ${DFLASH_MODEL_REVISION:-} =~ ^[0-9a-f]{40}$ ]] ||
      fail 'DFlash2 qualification requires an explicit immutable draft revision'
    ;;
  *) fail 'SPECULATOR must be mtp or dflash2' ;;
esac
export NUM_SPECULATIVE_TOKENS=${depth}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.91}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
export MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}
for name in MAX_NUM_BATCHED_TOKENS MAX_CUDAGRAPH_CAPTURE_SIZE; do
  [[ ${!name} =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer"
done
if [[ -z ${CUDAGRAPH_CAPTURE_SIZES:-} ]]; then
  capture_sizes=()
  for size in 1 2 4 8 16 32 40 48 64 96 128 192 256; do
    ((size < MAX_CUDAGRAPH_CAPTURE_SIZE)) && capture_sizes+=("${size}")
  done
  capture_sizes+=("${MAX_CUDAGRAPH_CAPTURE_SIZE}")
  export CUDAGRAPH_CAPTURE_SIZES="${capture_sizes[*]}"
fi
export GLM53_KDA_DECODE_BACKEND=b12x GLM53_KDA_PREFILL_BACKEND=flashkda
export B12X_PCIE_ALLREDUCE=1 VLLM_ENABLE_PCIE_ALLREDUCE=1
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export GLM53_TP3_REQUIRE_RUNTIME_PROOF=1
# Measured best for ws3 decode messages (optimization/NOTES.md sweep); the
# 24us/call seen in serving profiles is peer-arrival skew, not geometry.
export B12X_PCIE_ONESHOT_THREADS=${B12X_PCIE_ONESHOT_THREADS:-512}
# EP3 requires an EP-capable MoE implementation; R34 B12X rejects EP.
export MOE_BACKEND=${MOE_BACKEND:-auto}

# Default chat-template behaviour. Both remain per-request overridable; these
# only set the server-side default. Defaults match the qualified configuration.
# The checkpoint's chat template recognises exactly 'low' and 'high' and renders
# every other value as "Reasoning Effort: Max", so only these three strings are
# accepted here; anything else would silently become Max (asking for "minimal"
# would give you the most expensive tier).
export REASONING_EFFORT=${REASONING_EFFORT:-high}
case ${REASONING_EFFORT} in
  low|high|max) ;;
  *) fail "REASONING_EFFORT must be low, high or max, got ${REASONING_EFFORT}" ;;
esac
export CLEAR_THINKING=${CLEAR_THINKING:-true}
case ${CLEAR_THINKING} in
  true|false) ;;
  *) fail "CLEAR_THINKING must be true or false, got ${CLEAR_THINKING}" ;;
esac

# Prefix caching cannot be turned off for this model. The inherited launcher
# treats ENABLE_PREFIX_CACHING=0 as "omit --enable-prefix-caching", but vLLM V1
# defaults prefix caching ON, so the knob silently does nothing. Forcing it off
# with --no-enable-prefix-caching does not work either: that drops
# mamba_cache_mode from 'align', and the GLM-5.3 split target/recurrent-state
# page layout requires align, so the engine core aborts at startup with
# "Split GLM-5.3 target and recurrent-state pages require mamba_cache_mode=
# 'align'". Fail loudly rather than pretend the setting took effect.
case ${ENABLE_PREFIX_CACHING:-1} in
  1) ;;
  0) fail "ENABLE_PREFIX_CACHING=0 is not supported for GLM-5.3-Flash: the split target/recurrent page layout requires mamba_cache_mode='align', which requires prefix caching. The setting was previously silent and had no effect." ;;
  *) fail "ENABLE_PREFIX_CACHING must be 0 or 1, got ${ENABLE_PREFIX_CACHING}" ;;
esac

delta=$(sha256sum /opt/glm53-flash/tp3-source-delta.json)
delta=${delta%% *}
fingerprint="r34-tp3-vllm26948c19-b12xf5366673-${delta:0:16}"
export LOCAL_INFERENCE_CACHE_FINGERPRINT=${fingerprint}
cache_root=/cache/jit/${fingerprint}
export XDG_CACHE_HOME=${cache_root}
for name in VLLM_CACHE_ROOT VLLM_CACHE_DIR TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR \
  TORCHINDUCTOR_CACHE_DIR VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR FLASHINFER_WORKSPACE_BASE \
  TVM_FFI_CACHE_DIR TVM_CACHE_DIR TILELANG_CACHE_DIR CUTE_DSL_CACHE_DIR \
  B12X_CUTE_COMPILE_CACHE_DIR B12X_COMPILE_CACHE_DIR SPARKINFER_COMPILE_CACHE_DIR \
  DG_JIT_CACHE_DIR MM_SPARSE_ATTN_AOT_CACHE MINFER_FMHA_CACHE_DIR NUMBA_CACHE_DIR \
  CUDA_CACHE_PATH CUPY_CACHE_DIR; do
  export "${name}=${cache_root}/${name}"
done
# The inherited launcher passes --no-enable-flashinfer-autotune; appended args
# win, restoring the R34 vLLM default (tune, persist to the cache fingerprint).
autotune_args=(--enable-flashinfer-autotune)
[[ ${FLASHINFER_AUTOTUNE:-1} == 0 ]] && autotune_args=()
printf 'R34 TP3 candidate: TP=3 EP=3 DCP=%s spec=%s:%s autotune=%s cache=%s\n' "${DCP}" \
  "${SPECULATOR}" "${depth}" "${FLASHINFER_AUTOTUNE:-1}" "${fingerprint}"
# The inherited launcher hardcodes --quantization modelopt_mixed and appends
# "$@" last, so a trailing --quantization wins for checkpoints that need another
# scheme. Empty for the default checkpoint, which keeps its arguments unchanged.
quant_args=()
[[ -n ${QUANTIZATION:-} ]] && quant_args=(--quantization "${QUANTIZATION}")
exec /usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh \
  --enable-expert-parallel --mm-encoder-tp-mode weights "${autotune_args[@]}" \
  --default-chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT}\",\"clear_thinking\":${CLEAR_THINKING}}" \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' "${quant_args[@]}"
