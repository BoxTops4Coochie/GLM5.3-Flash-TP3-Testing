# SPDX-License-Identifier: Apache-2.0
"""WS3/WS4 placement policy for DS4.1 TP3 expert layers.

One contiguous VA region PER PARAMETER (not one padded per-expert-block bank):
region bytes = num_experts * row_bytes rounded up to the 2 MiB page
granularity. Experts are then exactly contiguous inside the region, which is
what the b12x staging kernels assume when they compute
``expert_idx * expert_words`` from tensor shapes - no stride bank needed as
long as every region is internally contiguous.

Within a region the device-hot part is a contiguous page prefix (hot pages
first, HOST_NUMA cold tail), so a hot-prefix-only shim stays valid.

cold_experts/hot_blocks keep the block-granular view used by the histogram
policy tests: blocks are 3 pages (6 MiB) and hot must remain a prefix.
"""

from __future__ import annotations

import math

GRAN = 2 * 1024 * 1024  # driver minimum granularity on GB202
PAGES_PER_BLOCK = 3  # smallest whole-page expert row group (row_bytes ~2 MiB)
BLOCK_BYTES = PAGES_PER_BLOCK * GRAN
NUM_EXPERTS = 384


def region_plan(row_bytes: int, num_experts: int, cold_fraction: float) -> tuple[int, int]:
    """Return (num_pages, hot_pages) for one contiguous per-param region.

    hot_pages is the largest whole-page prefix covering (1 - cold_fraction)
    of the live bytes, never exceeding the region.
    """
    total_bytes = num_experts * row_bytes
    num_pages = math.ceil(total_bytes / GRAN)
    hot_pages = min(num_pages, math.ceil(num_pages * (1.0 - cold_fraction)))
    return num_pages, hot_pages


def region_bytes(num_pages: int) -> int:
    return num_pages * GRAN


def cold_experts(histogram: dict[int, int] | None, cold_fraction: float) -> set[int]:
    """Expert ids to keep HOST-backed; hottest ids first when a histogram exists."""
    n_cold = int(NUM_EXPERTS * cold_fraction)
    if histogram is None:
        return set(range(NUM_EXPERTS - n_cold, NUM_EXPERTS))
    ranked = sorted(histogram, key=lambda e: histogram[e], reverse=True)
    return {e for e in ranked[n_cold:]}


def hot_blocks(cold: set[int]) -> set[int]:
    return {e for e in range(NUM_EXPERTS) if e not in cold}


def summary(cold_fraction: float, histogram: dict[int, int] | None = None) -> dict:
    cold = cold_experts(histogram, cold_fraction)
    hot = hot_blocks(cold)
    return {
        "experts": NUM_EXPERTS,
        "hot_experts": len(hot),
        "cold_experts": len(cold),
        "device_hot_GiB": len(hot) * BLOCK_BYTES / 2**30,
        "host_cold_GiB": len(cold) * BLOCK_BYTES / 2**30,
    }
