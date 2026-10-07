# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load retained tensors while the quantization method owns compressed experts."""

import time
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
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

SCHEMA = "lil-nvfp4-csf-checkpoint/1"
CODEC = "byte-window4-fixed-stream-u24-exceptions/1"
FAMILIES = {
    "glm53_nvfp4": (288, 4096, 2048, range(3, 45)),
    "qwen38_flash_next_nvfp4": (512, 2560, 640, range(48)),
}


@lru_cache(maxsize=4)
def checkpoint_contract(root: str) -> dict:
    """Validate the NVFP4-CSF container before loading model tensors."""
    return read_csf_contract(root, schema=SCHEMA, codec=CODEC, families=FAMILIES)


def read_nvfp4_csf_layer(
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
):
    """Read rank-local up/gate/down tensors and unprepared compressed scales."""
    contract = checkpoint_contract(str(Path(root).resolve()))
    e, h, n, layers = FAMILIES[contract["family"]]
    logical_intermediate_size = _logical_intermediate_size(
        contract["family"], n, intermediate_size, tp_size
    )
    if (num_experts, hidden_size, logical_intermediate_size) != (e, h, n):
        raise ValueError("NVFP4-CSF expert geometry differs from the checkpoint family")
    if layer_index not in layers:
        raise ValueError("NVFP4-CSF layer is outside the compressed expert inventory")
    with CsfTensorReader(root, contract["source_names"], "nvfp4") as reader:

        def experts():
            for expert in range(num_experts):
                prefix = (
                    f"model.language_model.layers.{layer_index}.mlp.experts.{expert}"
                )
                yield tuple(
                    reader.matrix(
                        f"{prefix}.{p}.weight",
                        f"{prefix}.{p}.weight_scale",
                        global_scale=f"{prefix}.{p}.weight_scale_2",
                        input_scale=f"{prefix}.{p}.input_scale",
                    )
                    for p in ("up_proj", "gate_proj", "down_proj")
                )

        return _load_nvfp4_csf_weights(
            experts(),
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            device=device,
            w13_scale_scratch=w13_scale_scratch,
            w2_scale_scratch=w2_scale_scratch,
            logical_intermediate_size=logical_intermediate_size,
        )


def _logical_intermediate_size(family, logical, intermediate_size, tp_size):
    """Checkpoint expert width for a (possibly Kraken TP3 MOE_TP-padded) layer.

    Kraken TP3 shards the routed experts across TP instead of EP by padding the
    2048-channel expert width to 2112 (704 per rank, VLLM_GLM53_TP3_MOE_TP).
    The checkpoint keeps 2048; rank-local tails past it load as zeros.
    """
    import os

    if intermediate_size == logical:
        return logical
    padded = int(os.environ.get("VLLM_GLM53_TP3_MOE_TP", "0") or 0)
    if (
        family == "glm53_nvfp4"
        and tp_size == 3
        and padded == intermediate_size
        and intermediate_size > logical
    ):
        return logical
    raise ValueError("NVFP4-CSF expert geometry differs from the checkpoint family")


def _slice_scale_plane(
    fixed,
    exceptions,
    rows,
    columns,
    row_slice,
    column_slice,
    out_rows=None,
    out_columns=None,
):
    """Copy aligned compressed extents and rebase exception positions.

    out_rows / out_columns (Kraken TP3 MOE_TP) extend the slice with zero
    nibbles: padded rows get zero bases (scale 0) and padded columns keep the
    row base. Their FP4 weights are zero, so their scales never contribute.
    """
    r0, r1 = row_slice
    c0, c1 = column_slice
    out_rows = r1 - r0 if out_rows is None else out_rows
    out_columns = c1 - c0 if out_columns is None else out_columns
    if out_rows < r1 - r0 or out_columns < c1 - c0 or out_rows % 16 or out_columns % 2:
        raise ValueError("NVFP4-CSF padded extents must cover the slice in slabs/pairs")
    if not (0 <= r0 < r1 <= rows and r0 % 16 == r1 % 16 == 0):
        raise ValueError("NVFP4-CSF row slices require complete 16-row slabs")
    if not (0 <= c0 < c1 <= columns and c0 % 2 == c1 % 2 == columns % 2 == 0):
        raise ValueError("NVFP4-CSF column slices require complete nibble pairs")
    if fixed.dtype != torch.uint8 or exceptions.dtype != torch.uint32:
        raise TypeError("NVFP4-CSF requires uint8 fixed and uint32 exception tensors")
    stream = fixed.numpy().reshape(rows // 16, 16 * (1 + columns // 2))
    bases = stream[:, :16]
    if np.any(bases > 240):
        raise ValueError("NVFP4-CSF row bases must be in 0..240")
    packed = stream[:, 16:].reshape(rows, columns // 2)
    selected = packed[r0:r1, c0 // 2 : c1 // 2]
    if out_rows != r1 - r0 or out_columns != c1 - c0:
        padded = np.zeros((out_rows, out_columns // 2), dtype=np.uint8)
        padded[: r1 - r0, : (c1 - c0) // 2] = selected
        selected = padded
        selected_bases = np.zeros((out_rows // 16, 16), dtype=np.uint8)
        selected_bases[: (r1 - r0) // 16] = bases[r0 // 16 : r1 // 16]
    else:
        selected_bases = bases[r0 // 16 : r1 // 16]
    result = np.concatenate(
        (selected_bases, selected.reshape(out_rows // 16, -1)), 1
    )
    words = exceptions.numpy().reshape(-1)
    positions = words & np.uint32(0xFFFFFF)
    if len(words) and (
        positions[-1] >= rows * columns or np.any(positions[1:] <= positions[:-1])
    ):
        raise ValueError("NVFP4-CSF exception positions must be sorted and unique")
    rr, cc = positions // columns, positions % columns
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    local_positions = (rr[keep] - r0) * out_columns + cc[keep] - c0
    selected_words = (words[keep] & np.uint32(0xFF000000)) | local_positions
    return result.copy(), selected_words.astype(np.uint32)


def _load_nvfp4_csf_weights(
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
    logical_intermediate_size=None,
):
    """Slice and upload up/gate/down-ordered expert projections.

    The iterable must yield exactly ``num_experts`` projection triples. Tensor
    stores, manifests and model-specific tensor names belong to the caller.
    Expanded scale buffers remain caller-owned for serialized layer execution.
    """
    from b12x.moe.fused_moe import CsfScalePlanes, Nvfp4CsfWeights, PackedWeights

    if num_experts <= 0 or hidden_size <= 0 or hidden_size % 128:
        raise ValueError("NVFP4-CSF requires experts and 128-aligned hidden channels")
    first, last = tp_extent(intermediate_size, tp_rank, tp_size, 64)
    local = last - first
    # Kraken TP3 MOE_TP: the physical width exceeds the checkpoint's; the
    # rank-local channels past it are zero weights (and zero/neutral scales).
    logical = intermediate_size if logical_intermediate_size is None else logical_intermediate_size
    real_last = min(last, logical)
    real = max(0, real_last - first)
    if real % 64:
        raise ValueError("NVFP4-CSF padded TP extents must keep 64-channel slabs")
    padded = real != local
    alloc = torch.zeros if padded else torch.empty
    w13 = alloc(
        (num_experts, 2 * local, hidden_size // 2), dtype=torch.uint8, device="cpu"
    )
    w2 = alloc(
        (num_experts, hidden_size, local // 2), dtype=torch.uint8, device="cpu"
    )
    # Model loaders may set BF16 as the default dtype. Calibration belongs to
    # the source FP32 contract and must not be rounded with the model weights.
    g13, g2 = (
        torch.empty(num_experts, dtype=torch.float32, device="cpu") for _ in range(2)
    )
    a13, a2 = (
        torch.empty(num_experts, dtype=torch.float32, device="cpu") for _ in range(2)
    )
    fixed13, fixed2, exceptions13, exceptions2 = [], [], [], []

    def scalar(value, role):
        if value is None or value.numel() != 1 or value.dtype != torch.float32:
            raise ValueError(f"NVFP4 {role} must be one FP32 value")
        if not bool(torch.isfinite(value).all() and (value > 0).all()):
            raise ValueError(f"NVFP4 {role} must be positive and finite")
        return value.reshape(())

    for expert, (first_projection, second_projection, down) in zip(
        range(num_experts), experts, strict=True
    ):
        f13, e13, global13, input13 = [], [], [], []
        for matrix, projection in enumerate(
            (first_projection, second_projection, down)
        ):
            view = projection.weight
            expected = (
                [logical, hidden_size // 2]
                if matrix < 2
                else [hidden_size, logical // 2]
            )
            if view.get_shape() != expected or view.get_dtype() != "U8":
                raise ValueError(
                    f"NVFP4 nibble geometry/dtype mismatch: "
                    f"expert={expert}, projection={matrix}"
                )
            global_scale, input_scale = (
                scalar(projection.global_scale, "global_scale"),
                scalar(projection.input_scale, "input_scale"),
            )
            if matrix < 2:
                if real:
                    w13[expert, matrix * local : matrix * local + real].copy_(
                        view[first:real_last, :]
                    )
                rows, columns = logical, hidden_size // 16
                row_slice, column_slice = (first, real_last), (0, columns)
                out_rows, out_columns = local, columns
                global13.append(global_scale)
                input13.append(input_scale)
            else:
                if real:
                    w2[expert, :, : real // 2].copy_(view[:, first // 2 : real_last // 2])
                rows, columns = hidden_size, logical // 16
                row_slice, column_slice = (0, rows), (first // 16, real_last // 16)
                out_rows, out_columns = rows, local // 16
                g2[expert], a2[expert] = global_scale, input_scale
            fixed, exceptions = _slice_scale_plane(
                projection.fixed,
                projection.exceptions,
                rows,
                columns,
                row_slice,
                column_slice,
                out_rows=out_rows,
                out_columns=out_columns,
            )
            if matrix < 2:
                f13.append(fixed)
                if matrix:
                    exceptions += np.uint32(local * columns)
                e13.append(exceptions)
            else:
                fixed2.append(fixed)
                exceptions2.append(exceptions)
        if not torch.equal(global13[0], global13[1]):
            raise ValueError("NVFP4-CSF requires equal gate/up global weight scales")
        g13[expert] = global13[0]
        a13[expert] = torch.stack(input13).amax()
        fixed13.append(np.concatenate(f13))
        exceptions13.append(np.concatenate(e13))
    return Nvfp4CsfWeights(
        packed=PackedWeights(
            w13=w13.to(device),
            w2=w2.to(device),
            w13_block_scales=w13_scale_scratch,
            w2_block_scales=w2_scale_scratch,
            w13_global_scales=g13.to(device),
            w2_global_scales=g2.to(device),
            input_scale=a13.to(device).reciprocal(),
            intermediate_scale=a2.to(device).reciprocal(),
            immutable_input_scales=True,
        ),
        w13_scales=CsfScalePlanes(
            tuple(torch.from_numpy(p) for p in fixed13),
            tuple(torch.from_numpy(p) for p in exceptions13),
        ),
        w2_scales=CsfScalePlanes(
            tuple(torch.from_numpy(p) for p in fixed2),
            tuple(torch.from_numpy(p) for p in exceptions2),
        ),
    )


class Nvfp4CsfModelLoader(DefaultModelLoader):
    def _root(self, model_config):
        quant = getattr(model_config.hf_config, "quantization_config", None)
        quant = quant or model_config.hf_text_config.quantization_config
        if quant.get("quant_method") != "nvfp4_csf":
            raise ValueError("NVFP4-CSF loader requires quant_method=nvfp4_csf")
        root = Path(quant["checkpoint_root"])
        if not root.is_absolute():
            raise ValueError("NVFP4-CSF checkpoint_root must be an absolute local path")
        return root, checkpoint_contract(str(root.resolve()))

    def download_model(self, model_config):
        self._root(model_config)

    def get_all_weights(self, model_config, model):
        root, contract = self._root(model_config)
        if getattr(model, "secondary_weights", ()):
            raise NotImplementedError(
                "NVFP4-CSF does not support secondary weight sources"
            )
        prefixes = getattr(model, "checkpoint_weight_name_prefixes", None)
        file_filter = getattr(model, "checkpoint_file_weight_filter", None)
        layers = model_config.hf_text_config.num_hidden_layers
        self.counter_before_loading_weights = time.perf_counter()
        for filename in sorted(set(contract["source_names"].values())):
            descriptors = safetensors_file_sources(str(root / "tensors" / filename))
            with safe_open(
                root / "tensors" / filename, framework="pt", device="cpu"
            ) as handle:
                for name in sorted(handle.keys()):
                    if prefixes is not None and not name.startswith(prefixes):
                        continue
                    match = re.search(
                        r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.", name
                    )
                    if match and int(match.group(1)) < layers:
                        continue
                    if callable(file_filter) and file_filter(name):
                        yield name, file_source_tensor(descriptors[name])
                    else:
                        yield name, handle.get_tensor(name)
