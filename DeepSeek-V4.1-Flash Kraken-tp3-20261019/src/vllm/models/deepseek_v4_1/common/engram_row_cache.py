# SPDX-License-Identifier: Apache-2.0
"""Host RAM row cache in front of b12x's disk Engram reader (DS41 TP3 port).

Disk-mode Engram lookups read every row from NVMe: the io_uring reader
bypasses the page cache, and b12x's DiskRowCache only removes duplicates
inside one batch. Engram rows are reused heavily across batches (measured on
layer 14: 62% of 6.5M lookups hit a row already read), so a modest cache in
host RAM removes most NVMe reads.

The cache is direct-mapped by row id: slot = hash(id) % capacity, with a tag
array holding the cached id. DiskRowCache._read_staged() is wrapped so that
hits are copied into the reader's output buffers and only the remaining
positions go to the native reader, through side buffers of the same capacity.
The output contract is unchanged: output position i holds the row of ids[i].

Enabled by DS41_ENGRAM_ROW_CACHE_ROWS (rows per disk table, e.g. 8388608 =
2 GiB at 256 bytes per row); 0 or unset keeps the stock reader.
"""

import os
import threading

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

_HASH = np.uint64(0x9E3779B97F4A7C15)


class _RowCache:
    def __init__(self, reader, capacity: int):
        self.capacity = capacity
        self.weight_bytes = reader.weight_row_bytes
        self.scale_bytes = reader.scale_row_bytes if reader._scale_buffer is not None else 0
        self.tags = np.full(capacity, -1, dtype=np.int64)
        self.weight = np.empty((capacity, self.weight_bytes), dtype=np.uint8)
        self.scale = (
            np.empty((capacity, self.scale_bytes), dtype=np.uint8)
            if self.scale_bytes
            else None
        )
        n = reader.max_lookups
        self.side_ids = np.empty(n, dtype=np.int64)
        self.side_weight = np.empty((n, self.weight_bytes), dtype=np.uint8)
        self.side_scale = (
            np.empty((n, self.scale_bytes), dtype=np.uint8) if self.scale_bytes else None
        )
        self.side_ids_buffer = memoryview(self.side_ids)
        self.side_weight_buffer = memoryview(self.side_weight)
        self.side_scale_buffer = (
            memoryview(self.side_scale) if self.side_scale is not None else None
        )
        self.lookups = 0
        self.hits = 0

    def slots(self, ids):
        mixed = ids.astype(np.uint64, copy=False) * _HASH
        return (mixed % np.uint64(self.capacity)).astype(np.int64)


def _cached_read_staged(self, count: int) -> None:
    cache = self.__dict__.get("_ds41_row_cache")
    if cache is None or self._gds is not None or count == 0:
        return _STOCK_READ_STAGED(self, count)
    import torch

    with torch.cuda.device(self.device):
        self._ids_ready.synchronize()
        ids = self.ids_host.numpy()[:count]
        weight_out = self.weight_host.numpy()
        scale_out = self.scale_host.numpy() if cache.scale_bytes else None
        valid = (ids >= self.shard_start) & (ids < self.shard_end)
        slots = cache.slots(ids)
        hit = valid & (cache.tags[slots] == ids)
        hit_pos = np.flatnonzero(hit)
        if hit_pos.size:
            weight_out[hit_pos] = cache.weight[slots[hit_pos]]
            if scale_out is not None:
                scale_out[hit_pos] = cache.scale[slots[hit_pos]]
        rest = np.flatnonzero(~hit)
        if rest.size:
            n = rest.size
            cache.side_ids[:n] = ids[rest]
            self._native.ple_reader_run(
                self._reader,
                cache.side_ids_buffer,
                cache.side_weight_buffer,
                cache.side_scale_buffer,
                n,
            )
            weight_out[rest] = cache.side_weight[:n]
            if scale_out is not None:
                scale_out[rest] = cache.side_scale[:n]
            fill = rest[valid[rest]]
            if fill.size:
                src = np.flatnonzero(valid[rest])
                fill_slots = slots[fill]
                cache.tags[fill_slots] = ids[fill]
                cache.weight[fill_slots] = cache.side_weight[src]
                if cache.scale is not None:
                    cache.scale[fill_slots] = cache.side_scale[src]
        cache.lookups += int(valid.sum())
        cache.hits += int(hit_pos.size)
        cache.calls = getattr(cache, "calls", 0) + 1
        if cache.calls % 2000 == 0:
            logger.info(
                "Engram row cache [%d, %d): %.1f%% hits over %d lookups",
                self.shard_start, self.shard_end,
                100.0 * cache.hits / max(1, cache.lookups), cache.lookups,
            )


_STOCK_READ_STAGED = None
_INSTALL_LOCK = threading.Lock()


def install() -> None:
    """Wrap DiskRowCache once; caches attach per reader on first construction."""
    global _STOCK_READ_STAGED
    rows = int(os.environ.get("DS41_ENGRAM_ROW_CACHE_ROWS", "0") or 0)
    if rows <= 0:
        return
    from b12x.sequence._shared.disk_table import DiskRowCache

    with _INSTALL_LOCK:
        if _STOCK_READ_STAGED is not None:
            return
        _STOCK_READ_STAGED = DiskRowCache._read_staged
        stock_init = DiskRowCache.__init__

        def _init(self, *args, **kwargs):
            stock_init(self, *args, **kwargs)
            if self._gds is None:
                self._ds41_row_cache = _RowCache(self, rows)
                logger.info(
                    "Engram row cache: %d rows (%.2f GiB) for shard [%d, %d)",
                    rows,
                    rows * (self.weight_row_bytes + self.scale_row_bytes) / (1 << 30),
                    self.shard_start,
                    self.shard_end,
                )

        DiskRowCache.__init__ = _init
        DiskRowCache._read_staged = _cached_read_staged


def stats(reader) -> tuple[int, int] | None:
    cache = getattr(reader, "_ds41_row_cache", None)
    return None if cache is None else (cache.lookups, cache.hits)
