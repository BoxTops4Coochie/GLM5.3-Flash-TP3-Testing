# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read retained native tensors from a lossless MXFP4 container.

Compressed routed experts are owned by the model's quantization method.
Engram tensors retain their immutable file-range descriptors, avoiding full
CPU/GPU staging allocations for the host-mapped embedding tables.
"""

import time
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open

from vllm.model_executor.model_loader.csf_utils import (
    CsfMatrix,
    CsfTensorReader,
    read_csf_contract,
    tp_extent,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    file_source_tensor,
    safetensors_file_sources,
)

FAMILIES = frozenset(("deepseek_v41", "deepseek_v4_flash", "kimi_k3"))


@lru_cache(maxsize=4)
def checkpoint_contract(root: str) -> dict:
    """Validate the MXFP4-CSF container before loading model tensors."""
    return read_csf_contract(root, families=FAMILIES)


def expert_tensor_names(family, layer_index, expert):
    """Source weight and scale names of an expert's gate, up and down matrices."""
    if family == "kimi_k3":
        prefix = (
            f"language_model.model.layers.{layer_index}."
            f"block_sparse_moe.experts.{expert}"
        )
        weight, scale = "weight_packed", "weight_scale"
    else:
        # DeepSeek-V4.1-Flash and DeepSeek-V4-Flash, including its vision
        # variant, keep DeepSeek's native routed-expert names.
        prefix = f"layers.{layer_index}.ffn.experts.{expert}"
        weight, scale = "weight", "scale"
    return tuple(
        (f"{prefix}.{p}.{weight}", f"{prefix}.{p}.{scale}") for p in ("w1", "w3", "w2")
    )


def read_mxfp4_csf_layer(
    root,
    layer_index,
    *,
    num_experts,
    hidden_size,
    intermediate_size,
    tp_rank,
    tp_size,
    device,
    w13_scale_scratch,
    w2_scale_scratch,
    expert_order=None,
    out_w13=None,
    out_w2=None,
):
    """Read rank-local gate/up/down tensors and unprepared compressed scales.

    ``expert_order[slot]`` names the checkpoint expert stored at ``slot``
    (default identity); ``out_w13``/``out_w2`` receive the packed weights in
    place of fresh device tensors, e.g. host-backed VMM expert storage.
    """
    contract = checkpoint_contract(str(Path(root).resolve()))
    family = contract["family"]
    tp_extent(intermediate_size, tp_rank, tp_size, 32)
    first_weight = expert_tensor_names(family, layer_index, 0)[0][0]
    if first_weight not in contract["source_names"]:
        raise ValueError("MXFP4-CSF layer is outside the compressed expert inventory")
    with CsfTensorReader(root, contract["source_names"], "mxfp4") as reader:

        order = range(num_experts) if expert_order is None else expert_order

        def experts():
            for expert in order:
                yield tuple(
                    reader.matrix(weight, scale)
                    for weight, scale in expert_tensor_names(
                        family, layer_index, expert
                    )
                )

        return _load_mxfp4_csf_weights(
            experts(),
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            device=device,
            w13_scale_scratch=w13_scale_scratch,
            w2_scale_scratch=w2_scale_scratch,
            out_w13=out_w13,
            out_w2=out_w2,
        )


def _slice_scale_plane(scales, rows, columns, row_slice, column_slice):
    """Describe a TP rectangle; payload validation and regrouping run on CUDA."""
    from b12x.loader._csf_source import upload_csf_planes

    if (scales.rows, scales.columns, scales.quantization) != (rows, columns, "mxfp4"):
        raise ValueError("MXFP4-CSF scale geometry differs from its packed weights")
    return upload_csf_planes(
        (scales,), "cpu", row_range=row_slice, column_range=column_slice
    )


def _load_mxfp4_csf_weights(
    experts: Iterable[tuple[CsfMatrix, CsfMatrix, CsfMatrix]],
    *,
    num_experts,
    hidden_size,
    intermediate_size,
    tp_rank,
    tp_size,
    device,
    w13_scale_scratch,
    w2_scale_scratch,
    out_w13=None,
    out_w2=None,
):
    """Slice and upload gate/up/down-ordered expert projections.

    The iterable must yield exactly ``num_experts`` projection triples. Tensor
    stores, manifests and model-specific tensor names belong to the caller.
    Expanded scale buffers remain caller-owned for serialized layer execution.
    """
    from b12x.loader._csf_blocks import CsfBlockPlane
    from b12x.loader._csf_source import CsfSource, batch_csf_sources
    from b12x.moe.fused_moe import Mxfp4CsfWeights

    if num_experts <= 0 or hidden_size <= 0 or hidden_size % 64:
        raise ValueError("MXFP4-CSF requires experts and 64-aligned hidden channels")
    first, last = tp_extent(intermediate_size, tp_rank, tp_size, 32)
    local = last - first
    shapes = (
        (num_experts, 2 * local, hidden_size // 2),
        (num_experts, hidden_size, local // 2),
    )
    w13, w2 = (
        torch.empty(shape, dtype=torch.uint8, device=device)
        if out is None
        else out
        for out, shape in zip((out_w13, out_w2), shapes)
    )
    if tuple(w13.shape) != shapes[0] or tuple(w2.shape) != shapes[1]:
        raise ValueError("MXFP4-CSF destination weights differ from the layer geometry")
    scales13: list[CsfBlockPlane | CsfSource] = []
    scales2: list[CsfBlockPlane | CsfSource] = []
    for expert, (first_projection, second_projection, down) in zip(
        range(num_experts), experts, strict=True
    ):
        for matrix, projection in enumerate(
            (first_projection, second_projection, down)
        ):
            view = projection.weight
            expected = (
                [intermediate_size, hidden_size // 2]
                if matrix < 2
                else [hidden_size, intermediate_size // 2]
            )
            if view.get_shape() != expected or view.get_dtype() not in ("I8", "U8"):
                raise ValueError(
                    f"MXFP4-CSF nibble geometry/dtype mismatch: "
                    f"expert={expert}, projection={matrix}"
                )
            if matrix < 2:
                w13[expert, matrix * local : (matrix + 1) * local].copy_(
                    view[first:last, :].view(torch.uint8)
                )
                rows, columns = intermediate_size, hidden_size // 32
                row_slice, column_slice = (first, last), (0, columns)
            else:
                w2[expert].copy_(view[:, first // 2 : last // 2].view(torch.uint8))
                rows, columns = hidden_size, intermediate_size // 32
                row_slice, column_slice = (0, rows), (first // 32, last // 32)
            scales = projection.scales
            if (scales.rows, scales.columns, scales.quantization) != (
                rows,
                columns,
                "mxfp4",
            ):
                raise ValueError(
                    "MXFP4-CSF scale geometry differs from its packed weights"
                )
            if isinstance(scales, CsfSource):
                scales = scales.select(row_range=row_slice, column_range=column_slice)
            (scales13 if matrix < 2 else scales2).append(scales)
    return Mxfp4CsfWeights(
        w13=w13,
        w2=w2,
        w13_scales=batch_csf_sources(
            scales13,
            device,
            join=2,
            **(
                {}
                if isinstance(scales13[0], CsfSource)
                else dict(row_range=(first, last))
            ),
        ),
        w2_scales=batch_csf_sources(
            scales2,
            device,
            **(
                {}
                if isinstance(scales2[0], CsfSource)
                else dict(column_range=(first // 32, last // 32))
            ),
        ),
        w13_scale_scratch=w13_scale_scratch,
        w2_scale_scratch=w2_scale_scratch,
    )


class Mxfp4CsfModelLoader(DefaultModelLoader):
    def _root(self, model_config):
        text_config = getattr(model_config, "hf_text_config", model_config.hf_config)
        quant = getattr(model_config.hf_config, "quantization_config", None)
        quant = quant or text_config.quantization_config
        if quant.get("quant_method") not in ("mxfp4_csf",):
            raise ValueError(
                "MXFP4-CSF loading requires a compressed-scale model config"
            )
        root = Path(quant["checkpoint_root"])
        if not root.is_absolute():
            raise ValueError("MXFP4-CSF checkpoint_root must be an absolute local path")
        return root, checkpoint_contract(str(root.resolve()))

    def download_model(self, model_config):
        self._root(model_config)

    def get_all_weights(self, model_config, model):
        root, contract = self._root(model_config)
        if getattr(model, "secondary_weights", ()):
            raise NotImplementedError(
                "MXFP4-CSF does not support secondary weight sources"
            )
        file_filter = getattr(model, "checkpoint_file_weight_filter", None)
        prefixes = getattr(model, "checkpoint_weight_name_prefixes", None)
        text_config = getattr(model_config, "hf_text_config", model_config.hf_config)
        layers = text_config.num_hidden_layers
        self.counter_before_loading_weights = time.perf_counter()
        for filename in sorted(set(contract["source_names"].values())):
            path = str(root / filename)
            descriptors = safetensors_file_sources(path)
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in sorted(descriptors):
                    if prefixes is not None and not name.startswith(prefixes):
                        continue
                    match = re.search(
                        r"(?:^|\.)layers\.(\d+)\.(?:ffn|block_sparse_moe)\.experts\.",
                        name,
                    )
                    if match and int(match.group(1)) < layers:
                        continue
                    if callable(file_filter) and file_filter(name):
                        yield name, file_source_tensor(descriptors[name])
                    else:
                        yield name, handle.get_tensor(name)
