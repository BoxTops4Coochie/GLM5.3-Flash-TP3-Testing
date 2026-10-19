# SPDX-License-Identifier: Apache-2.0
"""WS3 P4 walk variant: mixed-residency expert layers via VMM shim.

For decoder MoE layers with ordinal in [DS41_OFFLOAD_FROM_ORDINAL, DS41_OFFLOAD_TO_ORDINAL]:
  1. ONE contiguous VMM region PER routed-expert parameter
     (w13_weight, w13_weight_scale, w2_weight, w2_weight_scale), sized from
     the live tensor shape; experts stay exactly contiguous inside the region
     so the b12x staging arithmetic (expert_idx * row_words) needs no change.
  2. build one contiguous view per param over its region
  3. copy the module's current param values into the views (device->VMM)
  4. swap module param .data to the views

Within each region the device-hot part is a whole-page prefix (hot first,
HOST_NUMA cold tail); the hot fraction comes from DS41_VMM_COLD_FRACTION.
Histogram-driven non-prefix hot sets need WS4 bank pointers (see PR draft);
until then the uniform prior keeps hot a prefix.
"""

from __future__ import annotations

import json
import os

import torch

from vllm.logger import init_logger

from .vmm_shim import MixedResidencyAllocation, register_vmm_owner
from . import vmm_policy

logger = init_logger(__name__)

_PARAM_ORDER = ("w13_weight", "w13_scale", "w2_weight", "w2_scale")
# Preferred suffix first: this stack names scales w13_weight_scale /
# w2_weight_scale; bare *_scale kept as fallback for other checkpoints.
_CANDIDATES = {
    "w13_weight": ("w13_weight",),
    "w13_scale": ("w13_weight_scale", "w13_scale"),
    "w2_weight": ("w2_weight",),
    "w2_scale": ("w2_weight_scale", "w2_scale"),
}


def _load_histogram() -> dict[int, int] | None:
    path = os.environ.get("DS41_VMM_HISTOGRAM", "")
    if not path:
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}
    except Exception as exc:  # pragma: no cover
        logger.warning("ds41-p4: histogram load failed (%s); uniform prior", exc)
        return None


def maybe_vmm_wrap_layer(module: torch.nn.Module, ordinal: int) -> bool:
    """If ordinal is in range and env enabled, swap layer experts to VMM storage.

    Returns True if the module was converted. The module is an MoE layer whose
    routed-expert parameters carry the four stacked tensors.
    """
    if os.environ.get("DS41_VMM_ENABLE", "0") != "1":
        return False
    lo = int(os.environ.get("DS41_OFFLOAD_FROM_ORDINAL", "20"))
    hi = int(os.environ.get("DS41_OFFLOAD_TO_ORDINAL", "22"))
    if not (lo <= ordinal <= hi):
        return False

    found: dict[str, tuple[str, torch.nn.Parameter]] = {}
    for name, p in module.named_parameters():
        for key in _PARAM_ORDER:
            if key in found:
                continue
            for cand in _CANDIDATES[key]:
                if name == cand or name.endswith("." + cand):
                    found[key] = (name, p)
                    break
    # MXFP4-CSF layers keep only compressed scales (on device), so their
    # scale tensors are optional; the packed weights are always required.
    missing = [k for k in ("w13_weight", "w2_weight") if k not in found]
    if missing:
        logger.debug("ds41-p4: ordinal %d missing params %s; skip", ordinal, missing)
        return False

    cold_fraction = float(os.environ.get("DS41_VMM_COLD_FRACTION", "0.5"))

    converted: list[str] = []
    # from_blob has no deleter: retain physical owners independently of views.
    owners = getattr(module, "_ds41_vmm_allocations", None)
    if owners is None:
        owners = {}
        module._ds41_vmm_allocations = owners
    device_index = torch.cuda.current_device()
    # Run driver allocation and tensor operations in the worker's context.
    # The wrapper supplies its explicit target device for HOST_NUMA pointers;
    # a device scope alone cannot determine their CUDA access device.
    with torch.cuda.device(device_index):
        for key in (k for k in _PARAM_ORDER if k in found):
            _, p = found[key]
            shape = tuple(p.data.shape)
            num_experts = shape[0]
            row_bytes = p.data.numel() // num_experts * p.data.element_size()
            num_pages, hot_pages = vmm_policy.region_plan(row_bytes, num_experts, cold_fraction)
            total_bytes = vmm_policy.region_bytes(num_pages)
            chunk_pages = None
            chunk_experts = int(os.environ.get("DS41_VMM_CHUNK_EXPERTS", "0"))
            if chunk_experts > 0 and hot_pages == 0 and key in ("w13_weight", "w2_weight"):
                chunk_pages = max(1, -(-chunk_experts * row_bytes // vmm_policy.GRAN))
            alloc = MixedResidencyAllocation(
                num_bytes=row_bytes * num_experts,
                device_index=device_index,
                hot_page_indices=set(range(hot_pages)),
                cold_chunk_pages=chunk_pages,
            )
            alloc.operand_role = "raw_scale" if "scale" in key or "sfb" in key else "weight"
            alloc.layer_ordinal = ordinal
            owners[key] = alloc
            base = alloc.wrap_torch(torch, numel=total_bytes, dtype=torch.uint8)
            # keep only the live-byte window, then reinterpret at the param dtype
            live = base[: row_bytes * num_experts]
            view = live.view(p.data.dtype).view(shape)
            view.copy_(p.data.to(device="cpu", non_blocking=True))
            p.data = view
            register_vmm_owner(view, alloc)
            converted.append(f"{key}:{hot_pages}/{num_pages}p")

    logger.info(
        "ds41-p4: ordinal %d -> per-param VMM regions (%s), device_index %d",
        ordinal, ", ".join(converted), device_index,
    )
    return True
