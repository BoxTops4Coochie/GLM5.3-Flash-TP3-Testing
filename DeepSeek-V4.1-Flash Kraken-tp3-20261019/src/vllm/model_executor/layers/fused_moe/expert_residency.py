# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-expert pointer-bank residency for b12x w4a8_mx MoE weights.

Single policy path when bank mode is enabled. It REPLACES the prefix-only
page-residency plan for targeted layers on opt-in: hot experts become
dense-packed in GPU memory at true per-expert quanta, cold experts stay in
the existing pinned device-readable host backing. When bank mode is disabled
nothing here runs and the existing prefix/whole-parameter behaviour is
untouched; there is intentionally no second silent policy layered beside it.

Opt-in is per placement through the launcher environment
(``B12X_EXPERT_BANK_BUDGET_BYTES`` with optional
``B12X_EXPERT_BANK_HISTOGRAM``/``B12X_EXPERT_BANK_ENABLE``); it is never
inferred from ambient state. Podman ``-e`` is required for container env
delivery; launcher argument separation is preserved.

Frozen contract with the kernel side:

- four device Int64 tables ``w13_bank``/``down_bank``/``sfbw13_bank``/
  ``sfbdown_bank``, length ``E``; entry ``e`` is the absolute BYTE address
  of logical/local expert ``e``'s prepared slice inside its owned backing.
  Intra-expert offsets keep their existing units (word offsets stay word
  offsets; the loaded value is bytes).
- Tables are indexed by the logical/local expert identity routing produces;
  this module never renumbers experts. Gate/up order, packed bytes, scale
  encoding, TP-local shapes, alphas and routing weights are preserved
  verbatim by copying whole per-expert slices.
- Supported recipes: N128-repacked ``w4a8_mx`` (``n64_repack=False``,
  ``w4a8_b_tma=False``). Every other recipe rejects bank mode explicitly
  with :class:`BankModeRejected` instead of silently falling back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import json
import logging
import os
from pathlib import Path
import threading
import time

import torch

BANK_ENABLE_ENV = "B12X_EXPERT_BANK_ENABLE"
BANK_BUDGET_ENV = "B12X_EXPERT_BANK_BUDGET_BYTES"
BANK_HISTOGRAM_ENV = "B12X_EXPERT_BANK_HISTOGRAM"

# The four canonical expert operands, in launch-ABI binding order.
BANK_SLOTS: tuple[str, ...] = (
    "w13_bank",
    "down_bank",
    "sfbw13_bank",
    "sfbdown_bank",
)

_ALIGNMENT_BYTES = 8
_ROUTE_KERNELS = None


def _route_kernels():
    global _ROUTE_KERNELS
    if _ROUTE_KERNELS is not None:
        return _ROUTE_KERNELS
    # Lazy import keeps the non-CUDA residency bookkeeping usable on CPU.
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["N"])
    def count(IDS, COUNTS, ENABLED, N, EXPERTS: tl.constexpr,
              BLOCK: tl.constexpr):
        if tl.load(ENABLED) != 0:
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            ids = tl.load(IDS + offsets, offsets < N, other=-1)
            tl.atomic_add(COUNTS + ids, 1,
                          (offsets < N) & (ids >= 0) & (ids < EXPERTS),
                          sem="relaxed")

    @triton.jit
    def sample(COUNTS, OUT, EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
        ids = tl.arange(0, BLOCK)
        values = tl.atomic_add(COUNTS + ids, 0, ids < EXPERTS, sem="relaxed")
        tl.store(OUT + ids, values, ids < EXPERTS)

    _ROUTE_KERNELS = (count, sample)
    return _ROUTE_KERNELS


class BankModeRejected(RuntimeError):
    """The prepared recipe is outside the supported bank-mode matrix."""


@dataclass(frozen=True)
class ExpertBankSet:
    """One layer's owned pointer banks and backing.

    The descriptor references this object for the whole graph/plan lifetime,
    and it owns the hot GPU slab, so every address stored in the four tables
    stays valid while the descriptor is alive. ``hot_ids`` are logical IDs
    resident in GPU memory in a dense slab; cold IDs keep pointing into the
    existing pinned device-readable stacked storage. ``table_for`` is the
    accessor the host ABI validates against.
    """

    w13_bank: torch.Tensor
    down_bank: torch.Tensor
    sfbw13_bank: torch.Tensor
    sfbdown_bank: torch.Tensor
    hot_slabs: tuple[torch.Tensor, ...]
    hot_ids: tuple[int, ...]
    cold_ids: tuple[int, ...]
    slot_stride_bytes: tuple[int, ...] = tuple()

    def table_for(self, slot: str) -> torch.Tensor:
        tables = {name: getattr(self, name) for name in BANK_SLOTS}
        return tables[slot]


class ExpertRouteMonitor:
    """Counter-only routed-expert monitor for GPU-resident layers."""

    def __init__(self, *, layer_ordinal, num_experts, device):
        self.layer_ordinal = int(layer_ordinal)
        self.num_experts = _require_positive_int(num_experts, "num_experts")
        self.device = torch.device(device)
        self.rank = (torch.distributed.get_rank() if torch.distributed.is_initialized()
                     else int(os.environ.get("RANK", "0")))
        self.route_counts = torch.zeros(
            self.num_experts, dtype=torch.int64, device=self.device
        )
        self.route_enabled = torch.zeros((), dtype=torch.int32, device=self.device)
        self._route_sample = torch.empty_like(self.route_counts)
        self._route_host = torch.empty(
            self.num_experts, dtype=torch.int64, pin_memory=True
        )
        self._route_stream = torch.cuda.Stream(device=self.device)
        self._count_routes, self._sample_routes = _route_kernels()
        for dtype in (torch.int32, torch.int64):
            ids = torch.zeros(256, dtype=dtype, device=self.device)
            self.record_routes(ids)
        self._sample_routes[(1,)](
            self.route_counts, self._route_sample, self.num_experts, 512
        )
        self._route_ready = torch.cuda.Event()
        self._route_ready.record()
        self._route_interval = float(
            os.environ.get("B12X_EXPERT_ROUTES_INTERVAL", "30")
        )
        if self._route_interval <= 0:
            raise ValueError("route snapshot interval must be positive")
        self._started = False

    def record_routes(self, topk_ids):
        if (topk_ids.dtype not in (torch.int32, torch.int64)
                or not topk_ids.is_contiguous()
                or topk_ids.device != self.route_counts.device):
            raise ValueError("routes require contiguous int32/int64 IDs on monitor device")
        n = topk_ids.numel()
        if n:
            self._count_routes[((n + 255) // 256,)](
                topk_ids, self.route_counts, self.route_enabled,
                n, self.num_experts, 256,
            )

    def enable(self):
        if self._started:
            return
        self.route_enabled.fill_(1)
        self._route_ready.record()
        self._started = True
        threading.Thread(
            target=self._dump_loop,
            daemon=True,
            name=f"all-expert-routes-{self.rank}-{self.layer_ordinal}",
        ).start()

    def _dump_loop(self):
        while True:
            time.sleep(self._route_interval)
            try:
                self.dump_routes()
            except Exception:
                logging.getLogger(__name__).exception("all-layer route snapshot failed")

    def dump_routes(self, directory="/cache"):
        with torch.cuda.device(self.device), torch.cuda.stream(self._route_stream):
            self._route_stream.wait_event(self._route_ready)
            self._sample_routes[(1,)](
                self.route_counts, self._route_sample, self.num_experts, 512
            )
            self._route_host.copy_(self._route_sample, non_blocking=True)
        self._route_stream.synchronize()
        counts = self._route_host.tolist()
        payload = dict(
            rank=self.rank,
            layer=self.layer_ordinal,
            total_routes=sum(counts),
            counts=counts,
            timestamp=time.time(),
        )
        path = Path(directory) / (
            f"all-expert-routes-rank{self.rank}-layer{self.layer_ordinal}.json"
        )
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload) + "\n")
        os.replace(temporary, path)
        return payload

    # Counter-only monitor intentionally has no expert-bank table interface.


class DeferredWeightBanks:
    """All-cold codec banks, then a replaceable weight-only VRAM cache.

    Caller owns launch exclusion for construction, promotion and eviction.
    Tables never move; RAM tensors and their VMM owners remain retained.
    Scale ABI slots alias weight tables solely as unused pointer carriers;
    the codec specialization never loads them. No scale slabs exist.
    """

    def __init__(self, prepared, *, num_experts, device, owners,
                 quant_recipe="w4a8_mx", n128_repacked=True, banks=None):
        if banks is not None or quant_recipe != "w4a8_mx" or not n128_repacked:
            raise BankModeRejected("deferred codec banks require non-bank N128 w4a8_mx")
        self.num_experts = _require_positive_int(num_experts, "num_experts")
        self.device = torch.device(device)
        self.owners = tuple(owners)
        if len(self.owners) != 2 or any(getattr(o, "operand_role", None) != "weight" for o in self.owners):
            raise BankModeRejected("deferred banks require two retained VMM weight owners")
        self.sources = tuple(prepared[name] for name in ("w13_bank", "down_bank"))
        for source, owner in zip(self.sources, self.owners):
            if not source.is_contiguous() or source.numel() % num_experts:
                raise BankModeRejected("weights require contiguous whole expert slices")
            if source.data_ptr() != owner.va or owner.hot_bytes != 0:
                raise BankModeRejected("weights must use all-cold canonical VMM storage")
        self.scales = tuple(prepared[name] for name in ("sfbw13_bank", "sfbdown_bank"))
        if any(s.dtype != torch.uint32 or not s.is_contiguous() for s in self.scales):
            raise BankModeRejected("deferred banks require contiguous Uint32 codec containers")
        self.slot_stride_bytes = tuple(_slot_stride_bytes(s, num_experts) for s in self.sources)
        self.expert_bytes = sum(self.slot_stride_bytes)
        self._cold_tables = tuple(
            torch.tensor([s.data_ptr() + e * stride for e in range(num_experts)], dtype=torch.int64)
            for s, stride in zip(self.sources, self.slot_stride_bytes)
        )
        self.w13_bank, self.down_bank = tuple(t.to(self.device) for t in self._cold_tables)
        self.hot_slabs = ()
        self.hot_ids = ()
        self.cold_ids = tuple(range(num_experts))
        self._committed_tables = self._cold_tables
        self._failed_backing = ()
        self._publication_failed = False
        self.layer_ordinal = getattr(self.owners[0], "layer_ordinal", None)
        self._route_lock = threading.RLock()
        self._routes = os.environ.get("B12X_EXPERT_ROUTES", "0") == "1"
        self._routes_started = False
        if self._routes:
            if self.device.type != "cuda" or self.num_experts != 384:
                raise BankModeRejected("route instrumentation requires CUDA and 384 experts")
            if self.layer_ordinal is None or any(
                getattr(o, "layer_ordinal", None) != self.layer_ordinal for o in self.owners
            ):
                raise BankModeRejected("route instrumentation requires matching layer ordinals")
            self.rank = (torch.distributed.get_rank() if torch.distributed.is_initialized()
                         else int(os.environ.get("RANK", "0")))
            self.route_counts = torch.zeros(384, dtype=torch.int64, device=self.device)
            self.route_enabled = torch.zeros((), dtype=torch.int32, device=self.device)
            self._route_sample = torch.empty_like(self.route_counts)
            self._route_host = torch.empty(384, dtype=torch.int64, pin_memory=True)
            self._route_stream = torch.cuda.Stream(device=self.device)
            self._count_routes, self._sample_routes = _route_kernels()
            # Compile both ID representations before capture. N is runtime, so
            # different token counts reuse these binaries without new tensors.
            for dtype in (torch.int32, torch.int64):
                ids = torch.zeros(256, dtype=dtype, device=self.device)
                self.record_routes(ids)
            self._sample_routes[(1,)](self.route_counts, self._route_sample, 384, 512)
            self._route_ready = torch.cuda.Event()
            self._route_ready.record()
            self._route_interval = float(os.environ.get("B12X_EXPERT_ROUTES_INTERVAL", "30"))
            if self._route_interval <= 0:
                raise ValueError("route snapshot interval must be positive")

    def record_routes(self, topk_ids):
        """Always called during capture; device flag excludes preparation routes."""
        if not self._routes:
            return
        if (topk_ids.dtype not in (torch.int32, torch.int64)
                or not topk_ids.is_contiguous() or topk_ids.device != self.route_counts.device):
            raise ValueError("routes require contiguous int32/int64 IDs on the bank device")
        n = topk_ids.numel()
        if n:
            self._count_routes[((n + 255) // 256,)](
                topk_ids, self.route_counts, self.route_enabled, n, 384, 256)

    def enable_routes(self):
        """Called only after armed promotion has published its pointer tables."""
        if not self._routes or self._routes_started:
            return
        self.route_enabled.fill_(1)
        self._route_ready.record()
        self._routes_started = True
        threading.Thread(target=self._route_dump_loop, daemon=True,
                         name=f"expert-routes-{self.rank}-{self.layer_ordinal}").start()

    def _route_dump_loop(self):
        while True:
            time.sleep(self._route_interval)
            try:
                self.dump_routes()
            except Exception:
                logging.getLogger(__name__).exception("expert route snapshot failed")

    def dump_routes(self, directory="/cache"):
        """Atomic per-counter sample; not a global pause of concurrent routing.

        Only this background stream waits for its copy. Replacement is excluded
        while deriving hot/cold accounting; counts are cumulative, not reset.
        """
        if not self._routes:
            raise RuntimeError("route instrumentation is disabled")
        with self._route_lock, torch.cuda.device(self.device):
            with torch.cuda.stream(self._route_stream):
                self._route_stream.wait_event(self._route_ready)
                self._sample_routes[(1,)](self.route_counts, self._route_sample, 384, 512)
                self._route_host.copy_(self._route_sample, non_blocking=True)
            self._route_stream.synchronize()
            counts = self._route_host.tolist()
            total = sum(counts)
            hot = sum(counts[e] for e in self.hot_ids)
            payload = dict(rank=self.rank, layer=self.layer_ordinal,
                           total_routes=total, cold_routes=total - hot,
                           hot_hit_ratio=hot / total if total else 0.0,
                           cold_hit_ratio=(total - hot) / total if total else 0.0,
                           hot_ids=list(self.hot_ids), counts=counts, timestamp=time.time())
            path = Path(directory) / f"expert-routes-rank{self.rank}-layer{self.layer_ordinal}.json"
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload) + "\n")
            os.replace(temporary, path)
            return payload

    def table_for(self, slot):
        return {"w13_bank": self.w13_bank, "down_bank": self.down_bank,
                "sfbw13_bank": self.w13_bank, "sfbdown_bank": self.down_bank}[slot]

    @property
    def promoted_bytes(self):
        return len(self.hot_ids) * self.expert_bytes

    def receipt(self):
        return {"gpu_expert_bytes": self.promoted_bytes,
                "retained_host_bytes": self.num_experts * self.expert_bytes,
                "pointer_table_bytes": self.num_experts * 16,
                "scale_copy_bytes": 0}

    def replace_hot(self, hot_ids, *, budget_bytes):
        with self._route_lock:
            return self._replace_hot(hot_ids, budget_bytes=budget_bytes)

    def _replace_hot(self, hot_ids, *, budget_bytes):
        """Copy whole experts, synchronize, publish tables in place, retain RAM.

        Budget is the new cache size; replacement needs old+new peak bytes.
        Rollback failure is fatal: retain both candidate and old backing so
        partially published pointers cannot become dangling allocations.
        """
        hot = tuple(sorted(int(e) for e in hot_ids))
        if len(set(hot)) != len(hot) or any(e < 0 or e >= self.num_experts for e in hot):
            raise ValueError("invalid hot expert ids")
        if budget_bytes < 0 or len(hot) * self.expert_bytes > budget_bytes:
            raise ValueError("hot experts exceed weight byte budget")
        if self._publication_failed:
            raise RuntimeError("bank publication failed irrecoverably; restart required")
        if hot == self.hot_ids:
            return self.promoted_bytes
        torch.cuda.synchronize(self.device)
        slabs, tables = [], []
        for source, stride, cold in zip(self.sources, self.slot_stride_bytes, self._cold_tables):
            table = cold.clone()
            if hot:
                slab = torch.empty(len(hot) * stride, dtype=torch.uint8, device=self.device)
                slabs.append(slab)
                flat = source.view(torch.uint8).reshape(-1)
                for slot, expert in enumerate(hot):
                    slab.narrow(0, slot * stride, stride).copy_(flat.narrow(0, expert * stride, stride))
                    table[expert] = slab.data_ptr() + slot * stride
            tables.append(table)
        torch.cuda.synchronize(self.device)
        targets = (self.w13_bank, self.down_bank)
        try:
            for target, table in zip(targets, tables):
                target.copy_(table)
            torch.cuda.synchronize(self.device)
        except Exception:
            self._publication_failed = True
            self._failed_backing = tuple(slabs)
            for target, old in zip(targets, self._committed_tables):
                target.copy_(old)
            torch.cuda.synchronize(self.device)
            self._failed_backing = ()
            self._publication_failed = False
            raise
        self.hot_slabs = tuple(slabs)
        self.hot_ids = hot
        self.cold_ids = tuple(e for e in range(self.num_experts) if e not in hot)
        self._committed_tables = tuple(tables)
        return self.promoted_bytes

    def promote_up_to(self, budget_bytes):
        if budget_bytes < 0:
            raise ValueError("negative budget")
        count = min(self.num_experts, int(budget_bytes) // self.expert_bytes)
        return self.replace_hot(self.preferred_order()[:count], budget_bytes=budget_bytes)

    def preferred_order(self):
        """Experts most-routed first, from a previous run's route snapshot.

        B12X_DEFERRED_BANK_ROUTE_DIR names the directory holding the
        expert-routes-rank<r>-layer<l>.json files that dump_routes() writes
        (B12X_EXPERT_ROUTES=1). Without a usable file this is logical-id
        order, the original behavior.
        """
        directory = os.environ.get("B12X_DEFERRED_BANK_ROUTE_DIR", "")
        if not directory or self.layer_ordinal is None:
            return tuple(range(self.num_experts))
        rank = (torch.distributed.get_rank() if torch.distributed.is_initialized()
                else int(os.environ.get("RANK", "0")))
        path = Path(directory) / f"expert-routes-rank{rank}-layer{self.layer_ordinal}.json"
        try:
            counts = json.loads(path.read_text())["counts"]
            histogram = load_route_histogram(counts, num_experts=self.num_experts)
        except (OSError, ValueError, KeyError) as exc:
            logging.getLogger(__name__).warning(
                "deferred bank route order unavailable (%s); using id order", exc)
            return tuple(range(self.num_experts))
        # Same ranking as select_hot_experts: most routes first, ties by id.
        return tuple(e for e, _ in sorted(histogram, key=lambda item: (-item[1], item[0])))

    def evict_all(self):
        released = self.promoted_bytes
        self.replace_hot((), budget_bytes=0)
        return released


def _require_positive_int(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return number


def load_route_histogram(
    raw: Mapping[int | str, Any] | Sequence[Any] | None,
    *,
    num_experts: int,
) -> tuple[tuple[int, int], ...]:
    """Normalize a measured per-expert route histogram.

    Accepted shapes: a mapping from logical expert id to a route count, or a
    full sequence of ``num_experts`` counts in logical-id order. Missing ids
    count as zero routes (never as "equal to the average"); ties keep
    ascending logical-id order so the plan is deterministic across ranks.
    With no counts at all the caller gets the explicitly named deterministic
    baseline order (ascending logical id), not an invented heuristic.
    """

    _require_positive_int(num_experts, "num_experts")
    counts: dict[int, int] = {}
    if raw is None:
        return tuple((e, 0) for e in range(num_experts))
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            expert = int(str(key))
            if not 0 <= expert < num_experts:
                raise ValueError(f"histogram id {expert} outside [0, {num_experts})")
            count = int(value)
            if count < 0:
                raise ValueError(f"route count for expert {expert} is negative")
            counts[expert] = count
    else:
        if len(raw) != num_experts:
            raise ValueError(
                f"histogram has {len(raw)} entries for {num_experts} experts"
            )
        for expert, value in enumerate(raw):
            count = int(value)
            if count < 0:
                raise ValueError(f"route count for expert {expert} is negative")
            counts[expert] = count
    return tuple((e, counts.get(e, 0)) for e in range(num_experts))


def parse_histogram_env(raw: str | None, *, num_experts: int) -> tuple[tuple[int, int], ...]:
    """Parse ``id=count`` pairs (``;`` separated) from the launcher env."""

    if raw is None or not raw.strip():
        return load_route_histogram(None, num_experts=num_experts)
    counts: dict[str, int] = {}
    for item in raw.split(";"):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        if not value:
            raise ValueError(f"histogram entry {item!r} is not id=count")
        counts[key.strip()] = int(value)
    return load_route_histogram(counts, num_experts=num_experts)


def select_hot_experts(
    histogram: Sequence[tuple[int, int]],
    *,
    num_experts: int,
    expert_bytes: int,
    budget_bytes: int,
) -> tuple[int, ...]:
    """Deterministic byte-budgeted hot-set selection for one layer/rank.

    Equal-size experts (this placement): higher measured route counts win,
    ties broken by ascending logical id, until the next whole expert would
    exceed ``budget_bytes``. All four operand arrays of a chosen expert move
    together, so the budget is charged once per expert at ``expert_bytes``.
    Zero budget selects nobody; ``>= E * expert_bytes`` selects everybody.
    No fractional or per-parameter splitting.
    """

    expert_bytes = _require_positive_int(expert_bytes, "expert_bytes")
    budget = int(budget_bytes)
    if budget < 0:
        raise ValueError(f"budget_bytes must be >= 0, got {budget}")
    if len(histogram) != num_experts:
        raise ValueError(
            f"histogram covers {len(histogram)} experts, expected {num_experts}"
        )
    capacity = budget // expert_bytes
    if capacity >= num_experts:
        return tuple(e for e, _ in histogram)
    measured = any(count > 0 for _, count in histogram)
    ranked = sorted(
        histogram,
        key=lambda item: (-item[1], item[0]) if measured else (item[0],),
    )
    return tuple(expert for expert, _ in ranked[:capacity])


def _slot_stride_bytes(source: torch.Tensor, num_experts: int) -> int:
    """Per-expert byte stride of one canonical prepared operand.

    Derived live from the actual tensor so the byte addresses in the banks
    match the real packed layouts instead of a guessed constant.
    """

    rows = int(source.numel()) // int(num_experts)
    stride = rows * int(source.element_size())
    if stride % _ALIGNMENT_BYTES:
        raise BankModeRejected(
            f"per-expert stride {stride}B is not {_ALIGNMENT_BYTES}B aligned; "
            "bank mode requires Int64-loadable table entries"
        )
    return stride


def build_expert_banks(
    prepared: Mapping[str, torch.Tensor],
    *,
    num_experts: int,
    hot_ids: Sequence[int],
    device: torch.device | str,
) -> ExpertBankSet:
    """Own dense GPU slabs for hot experts; cold experts keep host backing.

    ``prepared`` maps each bank slot to the canonical prepared operand
    ([E, rows] storage). Hot experts are copied whole into one dense GPU
    slab per slot (byte-identical slices: gate/up order, packed payload and
    scale encoding preserved). Cold entries point into the existing pinned
    stacked storage, which the UVA offload path already makes device-
    readable; no second host copy is allocated. On any failure the freshly
    allocated slab/tables are dropped and the exception propagates with the
    descriptor untouched (transactional publication).
    """

    num_experts = _require_positive_int(num_experts, "num_experts")
    for name in BANK_SLOTS:
        source = prepared.get(name)
        if not isinstance(source, torch.Tensor):
            raise BankModeRejected(
                f"prepared operand {name!r} missing; bank mode needs the "
                "QMMA-repacked w4a8_mx representation"
            )
        if int(source.shape[0]) != num_experts:
            raise BankModeRejected(
                f"{name} has {int(source.shape[0])} experts, expected {num_experts}"
            )
    hot = tuple(sorted(int(e) for e in hot_ids))
    if len(set(hot)) != len(hot) or any(not 0 <= e < num_experts for e in hot):
        raise ValueError("hot expert ids must be unique and inside [0, E)")
    cold = tuple(e for e in range(num_experts) if e not in set(hot))
    slabs: list[torch.Tensor] = []
    tables: dict[str, torch.Tensor] = {}
    slot_strides: list[int] = []
    try:
        for name in BANK_SLOTS:
            source = prepared[name]
            stride = _slot_stride_bytes(source, num_experts)
            slot_strides.append(stride)
            if hot:
                slab = torch.empty(len(hot) * stride, dtype=torch.uint8, device=device)
            else:
                slab = torch.empty(1, dtype=torch.uint8, device=device)
            slabs.append(slab)
            source_bytes = source.view(torch.uint8)
            source_flat = source_bytes.reshape(-1)
            table = torch.empty(num_experts, dtype=torch.int64)
            slab_base = int(slab.data_ptr())
            host_base = int(source_bytes.data_ptr())
            for slot, expert in enumerate(hot):
                dest = slab.narrow(0, slot * stride, stride)
                dest.copy_(source_flat.narrow(0, expert * stride, stride))
                table[expert] = slab_base + slot * stride
            for expert in cold:
                table[expert] = host_base + expert * stride
            if not bool((table % _ALIGNMENT_BYTES == 0).all()):
                raise BankModeRejected(
                    f"{name}: computed bank entry is not "
                    f"{_ALIGNMENT_BYTES}B aligned"
                )
            tables[name] = table.to(device, non_blocking=True)
    except Exception:
        # Explicit rollback so the pre-bank representation remains the sole
        # owner; the caller never observes a half-published bank set.
        del slabs, tables
        raise
    return ExpertBankSet(
        w13_bank=tables["w13_bank"],
        down_bank=tables["down_bank"],
        sfbw13_bank=tables["sfbw13_bank"],
        sfbdown_bank=tables["sfbdown_bank"],
        hot_slabs=tuple(slabs),
        slot_stride_bytes=slot_strides,
        hot_ids=hot,
        cold_ids=cold,
    )


def bank_receipt(banks: ExpertBankSet, *, expert_bytes: int) -> dict[str, Any]:
    """One-time ledger of the bytes this bank owns, counted exactly once."""

    return {
        "gpu_expert_bytes": len(banks.hot_ids) * expert_bytes,
        "host_expert_bytes": len(banks.cold_ids) * expert_bytes,
        "pointer_table_bytes": sum(
            int(getattr(banks, name).numel()) * 8 for name in BANK_SLOTS
        ),
        "hot_experts": len(banks.hot_ids),
        "cold_experts": len(banks.cold_ids),
    }
