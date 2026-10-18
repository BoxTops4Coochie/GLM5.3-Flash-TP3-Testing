# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash TP3 FP8 weight-only decode path for the large BF16 projections.

``modelopt_mixed`` keeps attention and KDA projections in BF16. At decode batch
sizes those GEMMs are weight streams, so halving the bytes halves the time
wherever a weight is not already L2-resident. Measured per TP3 rank at M=4
(``optimization/sweep-20260922/dense-fp8-roofline.py``, weights from HBM):
KDA ``in_proj`` 44.4 -> 26.9 us, DSA ``o_proj`` 31.8 -> 19.3 us, KDA ``o_proj``
16.5 -> 10.7 us, LM head 261 -> 137 us. Small projections lose badly under
Marlin and are not converted.

For each selected layer the weight is quantized after loading to FP8 E4M3 with
one scale per output channel and repacked for the Marlin W8A16 kernel (BF16
activations). Batches of at most ``VLLM_GLM53_FP8_DENSE_MAX_M`` rows -- every
decode and verify step up to 8 sequences x 4 positions -- use it; larger
batches (prefill chunks) keep the original BF16 GEMM, so prefill numerics and
kernels are unchanged.

For the decoder projections the packed FP8 tensor replaces ``layer.weight``
and the BF16 original moves to a private attribute. The L2 prefetcher collects
public parameters and attributes only, so it streams the bytes decode actually
reads, and a fixed budget covers twice the fraction of a half-size weight.
The LM head keeps ``weight`` as is (other modules may read it) and carries the
FP8 copy alongside.

``VLLM_GLM53_FP8_DENSE=1`` enables it (default off). Sub-flags, all default
on once enabled: ``VLLM_GLM53_FP8_LM_HEAD``, ``VLLM_GLM53_FP8_FFN`` (the three
dense FFN layers), ``VLLM_GLM53_FP8_MTP`` (the MTP draft's DSA projections).
``VLLM_GLM53_FP8_PREFILL=w8a8`` (default ``bf16``) runs prefill-sized batches
on CUTLASS FP8 and drops the BF16 weights; see ``_prefill_mode``.
Qualified 2026-09-23 with all three effectively off (the LM head path had a
bug and fell back to BF16); set them to 0 to reproduce that configuration. This changes numerics of
decode-time projections and needs quality qualification before production.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

logger = init_logger(__name__)

# (N, K) per TP3 rank of the BF16 weights to convert. Chosen where FP8 won in
# both the HBM and the C8 measurements; see the module docstring.
GLM53_TP3_FP8_SHAPES: dict[tuple[int, int], str] = {
    (8598, 4096): "KDA in_proj_qkvgfab",
    (4096, 2816): "KDA o_proj",
    (4096, 6144): "DSA o_proj",
    (6144, 1536): "DSA q_b_proj",
}
# The three dense FFN layers (first_k_dense_replace=3, intermediate 12288/3).
# Shared-expert projections (1408x4096, 4096x704) are slower under Marlin and
# stay BF16. VLLM_GLM53_FP8_FFN=0 leaves the dense FFN in BF16.
GLM53_TP3_FP8_FFN_SHAPES: dict[tuple[int, int], str] = {
    (8192, 4096): "dense FFN gate_up",
    (4096, 4096): "dense FFN down",
}
LM_HEAD_SHAPE = (51648, 4096)


def enabled() -> bool:
    return os.environ.get("VLLM_GLM53_FP8_DENSE", "0") not in ("", "0")


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("", "0")


def _prefill_mode() -> str:
    """How batches above the decode threshold run.

    ``bf16`` (default, the qualified behaviour): keep the BF16 weight and use the
    original GEMM, so prefill is unchanged. ``w8a8``: drop the BF16 weight and
    run CUTLASS FP8 on a row-major copy of the same FP8 weights, with dynamic
    per-token activation scales; frees ~1 GB per converted GB of BF16 and is
    ~1.3-1.7x faster at prefill sizes, but changes prefill numerics.
    """
    mode = os.environ.get("VLLM_GLM53_FP8_PREFILL", "bf16").strip().lower()
    if mode not in ("bf16", "w8a8"):
        raise ValueError("VLLM_GLM53_FP8_PREFILL must be bf16 or w8a8")
    return mode


def _max_m() -> int:
    return int(os.environ.get("VLLM_GLM53_FP8_DENSE_MAX_M", "32"))


def _pack_marlin_fp8(
    weight: torch.Tensor, keep_rowmajor: bool = False
) -> nn.Module:
    """Quantize an (N, K) BF16 weight per output channel and pack it for Marlin.

    With ``keep_rowmajor`` the same FP8 values are also kept row-major (with an
    (N, 1) scale) for the CUTLASS prefill path, so decode and prefill read
    identical quantized weights.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_fp8_layer_for_marlin,
    )

    n, k = weight.shape
    qweight, scale = ops.scaled_fp8_quant(
        weight.contiguous(), use_per_token_if_dynamic=True
    )
    packed = nn.Module()
    packed.rowmajor = qweight if keep_rowmajor else None
    packed.scale_col = scale.reshape(n, 1).to(torch.float32) if keep_rowmajor else None
    packed.weight = nn.Parameter(qweight.t().contiguous(), requires_grad=False)
    packed.weight_scale = nn.Parameter(
        scale.reshape(1, n).to(torch.float32), requires_grad=False
    )
    packed.input_size_per_partition = k
    packed.output_size_per_partition = n
    packed.orig_dtype = weight.dtype
    prepare_fp8_layer_for_marlin(packed, size_k_first=True)
    return packed


def _decode_eligible(x: torch.Tensor, bias: torch.Tensor | None, max_m: int) -> bool:
    return (
        bias is None
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and 0 < x.shape[0] <= max_m
    )


class GLM53Fp8DecodeLinearMethod(UnquantizedLinearMethod):
    """Marlin FP8 W8A16 for decode-sized batches, the original method otherwise."""

    def __init__(self, inner: UnquantizedLinearMethod, label: str) -> None:
        super().__init__()
        self._inner = inner
        self._label = label
        self._max_m = _max_m()
        self._prefill = _prefill_mode()

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self._inner.process_weights_after_loading(layer)
        bf16 = layer.weight.data
        n, k = bf16.shape
        w8a8 = self._prefill == "w8a8"
        packed = _pack_marlin_fp8(bf16, keep_rowmajor=w8a8)
        if w8a8:
            # Prefill reads the row-major FP8 copy; the BF16 weight is released.
            layer._glm53_bf16_weight = None
            layer._glm53_fp8_rowmajor = packed.rowmajor
            layer._glm53_fp8_scale_col = packed.scale_col
        else:
            layer._glm53_bf16_weight = bf16
        del bf16
        layer._glm53_fp8_scale = packed.weight_scale
        layer._glm53_fp8_workspace = packed.workspace
        layer._glm53_fp8_nk = (n, k)
        # Decode reads the packed tensor, so expose it as the public weight the
        # L2 prefetcher discovers.
        layer.weight = packed.weight

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
        )

        if _decode_eligible(x, bias, self._max_m):
            n, k = layer._glm53_fp8_nk
            return apply_fp8_marlin_linear(
                x,
                layer.weight,
                layer._glm53_fp8_scale,
                layer._glm53_fp8_workspace,
                n,
                k,
                None,
            )
        if self._prefill == "w8a8":
            from vllm import _custom_ops as ops

            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            xq, xs = ops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
            out = ops.cutlass_scaled_mm(
                xq,
                layer._glm53_fp8_rowmajor.t(),
                scale_a=xs,
                scale_b=layer._glm53_fp8_scale_col,
                out_dtype=x.dtype,
                bias=bias,
            )
            return out.reshape(*shape[:-1], out.shape[-1])
        return self._inner._gemm_impl(layer, x, layer._glm53_bf16_weight, bias)


class GLM53Fp8DecodeLMHeadMethod(QuantizeMethodBase):
    """Wraps the LM head's method; keeps its BF16 weight and adds an FP8 copy.

    Must be a QuantizeMethodBase so the loader calls
    process_weights_after_loading, and must not be an
    UnquantizedEmbeddingMethod, or LogitsProcessor takes its vocab-projection
    branch on the raw BF16 weight and never calls apply().
    """

    def __init__(self, inner) -> None:
        super().__init__()
        self._inner = inner
        self._max_m = _max_m()
        self._packed: nn.Module | None = None

    def __getattr__(self, name: str):
        if name.startswith("__") or name in ("_inner", "_max_m", "_packed"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def create_weights(self, *args, **kwargs):
        return self._inner.create_weights(*args, **kwargs)

    def embedding(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        return self._inner.embedding(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self._inner.process_weights_after_loading(layer)
        weight = layer.weight.data
        if tuple(weight.shape) != LM_HEAD_SHAPE or weight.dtype != torch.bfloat16:
            logger.warning(
                "GLM-5.3 FP8 decode: LM head is %s %s, not converting",
                tuple(weight.shape),
                weight.dtype,
            )
            return
        self._packed = _pack_marlin_fp8(weight)
        logger.info_once("GLM-5.3 FP8 decode: LM head packed to Marlin FP8")

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
        )

        p = self._packed
        if p is not None and _decode_eligible(x, bias, self._max_m):
            return apply_fp8_marlin_linear(
                x,
                p.weight,
                p.weight_scale,
                p.workspace,
                p.output_size_per_partition,
                p.input_size_per_partition,
                None,
            )
        return self._inner.apply(layer, x, bias)


def enable_glm53_fp8_dense(module: nn.Module, *, draft: bool = False) -> int:
    """Wrap the selected BF16 projections; conversion happens after loading.

    ``draft=True`` is the MTP draft module (its DSA projections share the
    target's shapes); ``VLLM_GLM53_FP8_MTP=0`` leaves it BF16.
    """
    if not enabled() or (draft and not _flag("VLLM_GLM53_FP8_MTP")):
        return 0
    shapes = dict(GLM53_TP3_FP8_SHAPES)
    if _flag("VLLM_GLM53_FP8_FFN"):
        shapes.update(GLM53_TP3_FP8_FFN_SHAPES)
    counts: dict[str, int] = {}
    for child in module.modules():
        if not isinstance(child, LinearBase) or not isinstance(
            child.quant_method, UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        label = shapes.get((int(weight.shape[0]), int(weight.shape[1])))
        if label is None:
            continue
        child.quant_method = GLM53Fp8DecodeLinearMethod(child.quant_method, label)
        counts[label] = counts.get(label, 0) + 1
    wrapped = sum(counts.values())
    summary = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
    if draft:
        logger.info_once(
            "GLM-5.3 FP8 decode (MTP draft): %d projections [%s]", wrapped, summary
        )
    else:
        logger.info_once(
            "GLM-5.3 FP8 decode: %d BF16 projections use Marlin FP8 W8A16 for "
            "batches of at most %d rows, prefill %s [%s]",
            wrapped,
            _max_m(),
            _prefill_mode(),
            summary,
        )
    return wrapped


def enable_glm53_fp8_lm_head(lm_head: nn.Module) -> bool:
    if not enabled() or not _flag("VLLM_GLM53_FP8_LM_HEAD"):
        return False
    lm_head.quant_method = GLM53Fp8DecodeLMHeadMethod(lm_head.quant_method)
    logger.info_once("GLM-5.3 FP8 decode: LM head uses Marlin FP8 W8A16 for decode")
    return True
