# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

import os
from collections.abc import Generator
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


class UVAOffloader(BaseOffloader):
    """Offloader using Unified Virtual Addressing (UVA) for zero-copy access.

    This offloader moves parameters to pinned CPU memory and creates CUDA views
    using UVA. The GPU can then directly access the CPU memory without explicit
    transfers, at the cost of PCIe bandwidth (slower than GPU memory).

    When UVA is disabled via env var, falls back to a functional_call-based
    approach that moves parameters on-demand.

    Args:
        cpu_offload_max_bytes: Maximum bytes to offload to CPU.
        cpu_offload_params: Set of parameter name segments to selectively
            offload. If empty, all parameters are eligible up to the byte limit.
    """

    supports_tower_offload = True

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
    ):
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0
        self.cpu_offload_params = cpu_offload_params or set()

        self.pin_memory = should_pin_memory()
        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        prefix: str = "",
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading.

        ds41-tp3 overlay: the walk stays lazy (construction and offloading
        interleave, as stock), but on the main layer walk the budget is only spent
        on modules whose ordinal is >= DS41_OFFLOAD_FROM_ORDINAL (default 20, the
        CED decoder boundary). Tower walks (prefix != "") keep stock behaviour.
        """
        min_ordinal = int(os.environ.get("DS41_OFFLOAD_FROM_ORDINAL", "20"))
        max_ordinal = int(os.environ.get("DS41_OFFLOAD_TO_ORDINAL", "999"))
        prefix_dot = f"{prefix}." if prefix else ""
        modules: list[nn.Module] = []
        overlay_hits: list[tuple[int, int]] = []
        for ordinal, module in enumerate(modules_generator):
            if prefix or min_ordinal <= ordinal <= max_ordinal:
                p4_converted = False
                if not prefix and os.environ.get("DS41_VMM_ENABLE", "0") == "1":
                    try:
                        from . import uva_p4

                        p4_converted = uva_p4.maybe_vmm_wrap_layer(module, ordinal)
                    except Exception as exc:
                        logger.warning(
                            "ds41-p4: ordinal %d VMM conversion failed (%s); "
                            "falling back to stock offload",
                            ordinal,
                            exc,
                        )
                if not p4_converted:
                    before = self.cpu_offload_bytes
                    self._maybe_offload_to_cpu(module, prefix_dot)
                    delta = self.cpu_offload_bytes - before
                else:
                    delta = 0
                if delta > 0:
                    overlay_hits.append((ordinal, delta))
                    # 54972: freed accelerator blocks are retained by the caching
                    # allocator under expandable_segments; release them now so
                    # deeper walk steps do not OOM on reserved-but-free VA.
                    torch.accelerator.empty_cache()
            modules.append(module)
        if overlay_hits:
            logger.info(
                "ds41-tp3 overlay: offload walk from ordinal %d, "
                "(ordinal, bytes) = %s",
                min_ordinal,
                [(i, format_gib(b)) for i, b in overlay_hits],
            )
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return modules

    def _maybe_offload_to_cpu(self, module: nn.Module, prefix: str = "") -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            # Skip parameters an earlier wrap_modules call already offloaded.
            # The UVA path leaves p.device as the accelerator (a view of CPU
            # memory), so the marker is the only way to recognize those.
            if p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False):
                continue

            if self.cpu_offload_params:
                # Check if parameter belongs to the offloading set
                # Add dots here to ensure we match full segments only
                # e.g., "experts.w2_weight" matches "mlp.experts.w2_weight"
                # but not "mlp.experts.w2_weight_scale"
                should_offload = any(
                    f".{param}." in f".{prefix}{name}."
                    for param in self.cpu_offload_params
                )
                if not should_offload:
                    continue

            cpu_data = p.data.to(device="cpu")
            # 56413: pin only the fallback path. The UVA helper maps its own
            # host memory for unpinned tensors; pinning here first doubles
            # the transient host footprint (~1.9x, upstream #56410).
            if self.pin_memory and not self.uva_offloading:
                cpu_data = cpu_data.pin_memory()

            if not self.uva_offloading:
                p.data = cpu_data
            else:
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True

            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters and not self.uva_offloading:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                with nullcontext() if self.pin_memory else gpu_sync_allowed():
                    device_state = {
                        # here we blindly call `to(device)`
                        # if the parameter is already on the device,
                        # it will be a no-op
                        k: v.to(device, non_blocking=True)
                        for k, v in module.state_dict().items()
                    }

                # set `tie_weights=False` as tied weights in original model
                # become untied when calling .to(device) individually
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    tie_weights=False,
                )
                module.forward = forward
                return output

            module.forward = forward

        return module
