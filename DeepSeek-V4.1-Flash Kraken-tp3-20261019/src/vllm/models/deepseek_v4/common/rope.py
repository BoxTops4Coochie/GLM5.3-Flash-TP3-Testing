# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 RoPE builder with a serving-length materialization cap.

Frequency correction keeps checkpoint training geometry. Only the number of
cached lookup rows is bounded by ``DS41_ROPE_CACHE_ROWS``; unset keeps upstream.
"""
import os

import torch
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekV4ScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


def _compute_capped_cache(self) -> torch.Tensor:
    """Preserve YaRN frequency math; cap only allocated lookup positions."""

    inv_freq = self._compute_inv_freq(self.scaling_factor)
    full_rows = int(self.max_position_embeddings * self.scaling_factor)
    requested = int(os.environ.get("DS41_ROPE_CACHE_ROWS", "0"))
    rows = min(full_rows, requested) if requested else full_rows
    if rows <= 0:
        raise ValueError("DS41_ROPE_CACHE_ROWS must be positive")
    t = torch.arange(rows, device=inv_freq.device, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos() * self.mscale, freqs.sin() * self.mscale), dim=-1)


DeepseekV4ScalingRotaryEmbedding._compute_cos_sin_cache = _compute_capped_cache

def build_deepseek_v4_rope(
    config,
    *,
    head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    compress_ratio: int,
) -> RotaryEmbedding:
    rope_parameters = config.rope_parameters
    if isinstance(rope_parameters.get("main"), dict) and isinstance(
        rope_parameters.get("compress"), dict
    ):
        key = "compress" if compress_ratio > 1 else "main"
        rope_parameters = dict(rope_parameters[key])
    else:
        rope_parameters = dict(rope_parameters)

    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if compress_ratio > 1 and rope_parameters["rope_type"] != "default":
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    else:
        rope_parameters["rope_type"] = "deepseek_yarn"
        rope_parameters["factor"] = 1.0
        rope_parameters["original_max_position_embeddings"] = max_position_embeddings
    rope_parameters["mscale"] = 0
    rope_parameters["mscale_all_dim"] = 0
    rope_parameters["is_deepseek_v4"] = True
    rope_parameters["rope_dim"] = rope_head_dim

    requested_rows = int(os.environ.get("DS41_ROPE_CACHE_ROWS", "0"))
    if requested_rows and not 0 < requested_rows <= max_position_embeddings:
        raise ValueError(
            f"DS41_ROPE_CACHE_ROWS={requested_rows} must be inside "
            f"[1, {max_position_embeddings}]"
        )

    rotary = get_rope(
        head_dim,
        max_position=max_position_embeddings,
        rope_parameters=rope_parameters,
        is_neox_style=False,
    )
    if requested_rows and rotary.cos_sin_cache.shape[0] != requested_rows:
        raise RuntimeError(
            "DeepSeek RoPE cache cap was not applied during construction: "
            f"got {rotary.cos_sin_cache.shape[0]}, expected {requested_rows}"
        )
    return rotary
