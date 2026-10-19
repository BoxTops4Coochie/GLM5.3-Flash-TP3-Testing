# SPDX-License-Identifier: Apache-2.0
"""WS3 P4 shim: VMM HOST_NUMA-backed mixed-residency torch tensors.

Allocates a virtual address range whose physical pages can be backed by
either DEVICE memory or HOST_NUMA (pinned host) memory, mapped into the
device VA space and wrapped into a torch CUDA tensor via at::from_blob.
Kernels address the whole range through one base pointer; residency is a
page-granular (2 MiB) property invisible to the kernel.

Allocation uses the caller's CUDA context (the worker's primary context).
The from_blob wrapper is compiled lazily and supplies an explicit target
device: HOST_NUMA pointer attributes do not identify the CUDA access device.
The caller must retain the allocation until all tensor views and GPU work
using it are finished; tensor destruction does not free the VMM range.
Struct classes are defined after CDLL load; field names avoid 'location'
(ctypes/driver interplay artifact, see 25-ws2-vmm-probe.md).
"""

from __future__ import annotations

import ctypes
import os
import threading

_CU_MEM_ALLOC_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_LOCATION_TYPE_HOST_NUMA = 2
_CU_MEM_ACCESS_PROT_READWRITE = 3

_GRANULARITY_BYTES = 2 * 1024 * 1024  # verified on GB202 / driver 610.43.02


class _VmmError(RuntimeError):
    pass


class _VmmLibrary:
    """Lazy singleton driver binding with explicit argtypes."""

    _instance: "_VmmLibrary | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.cuda = ctypes.CDLL("libcuda.so.1")
        c = self.cuda
        for f in (
            "cuInit",
            "cuCtxCreate_v2",
            "cuCtxGetDevice",
            "cuMemCreate",
            "cuMemAddressReserve",
            "cuMemMap",
            "cuMemSetAccess",
            "cuMemUnmap",
            "cuMemRelease",
            "cuMemAddressFree",
            "cuDeviceGetAttribute",
        ):
            getattr(c, f).restype = ctypes.c_int
        c.cuInit.argtypes = [ctypes.c_uint]
        c.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        c.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]

        class _Loc(ctypes.Structure):
            _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

        class _Comp(ctypes.Structure):
            _fields_ = [("type", ctypes.c_int)]

        class _Prop(ctypes.Structure):
            # CUmemAllocationProp per CUDA 12/13 headers:
            # type(4) | requestedHandleTypes(4) | location(8) |
            # compressionType(4) | (pad 4) | gpuStorageDeadlineMs(8)
            # -> sizeof 32. Missing tail fields shifted nothing but keep
            # the full declared layout for driver validation parity.
            _fields_ = [
                ("type", ctypes.c_int),
                ("requestedHandleTypes", ctypes.c_int),
                ("loc", _Loc),
                ("compressionType", _Comp),
                ("gpuStorageDeadlineMs", ctypes.c_ulonglong),
            ]

        class _Acc(ctypes.Structure):
            _fields_ = [("acc_loc", _Loc), ("flags", ctypes.c_int)]

        self._Loc = _Loc
        self._Prop = _Prop
        self._Acc = _Acc
        c.cuMemCreate.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(_Prop),
            ctypes.c_uint64,
        ]
        c.cuMemAddressReserve.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        c.cuMemSetAccess.argtypes = [
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.POINTER(_Acc),
            ctypes.c_int,
        ]
        c.cuMemUnmap.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
        c.cuMemRelease.argtypes = [ctypes.c_uint64]
        c.cuMemAddressFree.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
        c.cuCtxSynchronize.argtypes = []
        c.cuCtxSynchronize.restype = ctypes.c_int
        c.cuMemcpyDtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
        c.cuMemcpyDtoD_v2.restype = ctypes.c_int

        # cuMemMap via plain dlsym FAILS under torch's primary context
        # (compat stub rejects VMM maps); the cuGetProcAddress-resolved
        # entry works. Resolve the VMM family through the proc-address API.
        c.cuGetProcAddress.restype = ctypes.c_int
        c.cuGetProcAddress.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int),
        ]

        def _resolve(name: str) -> int:
            fn = ctypes.c_void_p()
            st = ctypes.c_int()
            rc = c.cuGetProcAddress(
                name.encode(), ctypes.byref(fn), 12000, 0, ctypes.byref(st)
            )
            if rc != 0 or not fn.value:
                raise _VmmError(f"cuGetProcAddress({name}): rc={rc}")
            return fn.value

        proto_map = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
        )
        proto_unmap = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t)
        proto_setacc = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.POINTER(_Acc),
            ctypes.c_int,
        )
        self.cuMemMap = proto_map(_resolve("cuMemMap"))
        self.cuMemUnmap = proto_unmap(_resolve("cuMemUnmap"))
        self.cuMemSetAccess = proto_setacc(_resolve("cuMemSetAccess"))

        self._initialized = False
        self._numa_id: int | None = None

    @classmethod
    def get(cls) -> "_VmmLibrary":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def ensure_init(self, device_index: int) -> None:
        """cuInit only; the CALLER's context (torch primary ctx in the
        engine) owns all subsequent driver calls. Never create a private
        context here - private-context sequences failed in probe harnesses
        while the primary-context step-by-step sequence passes."""
        if self._initialized:
            return
        if self.cuda.cuInit(0) != 0:
            raise _VmmError("cuInit failed")
        self._initialized = True

    def host_numa_id(self, device_index: int) -> int:
        """CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 149."""
        if self._numa_id is None:
            dev = ctypes.c_int(device_index)
            out = ctypes.c_int()
            # device handle: ordinal == devid for this purpose
            rc = self.cuda.cuDeviceGetAttribute(
                ctypes.byref(out), 149, dev
            )
            if rc != 0 or out.value < 0:
                out.value = 0
            self._numa_id = out.value
        return self._numa_id

    def check(self, rc: int, what: str) -> None:
        if rc != 0:
            s = ctypes.c_char_p()
            self.cuda.cuGetErrorString(rc, ctypes.byref(s))
            raise _VmmError(f"{what}: rc={rc} {s.value.decode() if s.value else ''}")

_OWNER_BY_DATA_PTR: dict[int, "MixedResidencyAllocation"] = {}


def register_vmm_owner(tensor, owner: "MixedResidencyAllocation") -> None:
    """Retain and publish an owner by its canonical tensor data pointer."""

    pointer = int(tensor.data_ptr())
    if pointer != owner.va:
        raise ValueError("VMM owner registry requires the canonical base pointer")
    previous = _OWNER_BY_DATA_PTR.get(pointer)
    if previous is not None and previous is not owner:
        raise RuntimeError("canonical VMM pointer already has a different owner")
    _OWNER_BY_DATA_PTR[pointer] = owner


def vmm_owner_for_tensor(tensor):
    """Resolve ownership after in-place B12X repacking preserves storage."""

    return _OWNER_BY_DATA_PTR.get(int(tensor.untyped_storage().data_ptr()))


class MixedResidencyAllocation:
    """One VA range, TWO physical handles: a device-backed hot region
    [0, hot_bytes) and a HOST_NUMA-backed cold region [hot_bytes, size).
    Expert blocks MUST be ordered hot-first (see vmm_policy.hot_blocks +
    uva_p4's block permutation note) so each region is contiguous.

    Two handles per layer (not per-page): the driver fails rapid-fire
    small HOST_NUMA creates after ~4-5 successes in one process, while
    few large creates are reliable (25/26 probe notes). Base pointer is
    fixed for the process lifetime -> kernel ABI, data_ptr-keyed plans,
    and CUDA graphs all stay valid.
    """

    def __init__(
        self,
        num_bytes: int,
        device_index: int,
        hot_page_indices: set[int] | None = None,
        cold_chunk_pages: int | None = None,
    ) -> None:
        if num_bytes <= 0:
            raise ValueError("num_bytes must be positive")
        if hot_page_indices is not None:
            n = (num_bytes + _GRANULARITY_BYTES - 1) // _GRANULARITY_BYTES
            k = len(hot_page_indices)
            if k > n or hot_page_indices != set(range(k)):
                raise _VmmError(
                    "two-region shim requires hot pages to be a contiguous "
                    f"prefix set(range(k)); got {k} pages not in prefix form"
                )
        hot_all = hot_page_indices is None or len(hot_page_indices) == (
            num_bytes + _GRANULARITY_BYTES - 1
        ) // _GRANULARITY_BYTES
        self._all_hot = hot_all
        self._all_cold = hot_page_indices is not None and len(hot_page_indices) == 0
        lib = _VmmLibrary.get()
        lib.ensure_init(device_index)
        self.lib = lib
        self.device_index = device_index
        self.num_pages = (num_bytes + _GRANULARITY_BYTES - 1) // _GRANULARITY_BYTES
        self.size = self.num_pages * _GRANULARITY_BYTES
        self.hot_bytes = (
            len(hot_page_indices) * _GRANULARITY_BYTES
            if hot_page_indices is not None
            else self.size
        )

        numa = lib.host_numa_id(device_index)
        # CREATE FIRST, reserve later: a large VA reservation before device
        # creation deterministically breaks the subsequent cuMemCreate
        # (driver VA-accounting interaction, 26-ws3-notes). Handle creation
        # order: hot, cold, then reserve, then map.
        self._handles: list[tuple[str, int]] = []
        self._mappings: dict[int, tuple[int, int]] = {}
        self._cold_mapping: tuple[int, int, int] | None = None
        self._promoted_handle: int | None = None
        # All-cold regions created with cold_chunk_pages: one HOST_NUMA handle
        # per chunk, so a prefix of chunks can be promoted in place.
        self._cold_chunks: list[tuple[int, int, int]] = []
        self._chunk_promoted = 0
        self._freed = False
        handle_values: list[tuple[str, int]] = []
        if not self._all_cold:
            prop = lib._Prop()
            prop.type = _CU_MEM_ALLOC_TYPE_PINNED
            prop.requestedHandleTypes = 0
            prop.loc.type = _CU_MEM_LOCATION_TYPE_DEVICE
            prop.loc.id = device_index
            h = ctypes.c_uint64()
            lib.check(
                lib.cuda.cuMemCreate(ctypes.byref(h), self.hot_bytes, ctypes.byref(prop), 0),
                "hot region create",
            )
            handle_values.append(("hot", h.value))
        chunked = bool(cold_chunk_pages) and self._all_cold
        chunk_sizes: list[int] = []
        if chunked:
            remaining = self.size
            step = int(cold_chunk_pages) * _GRANULARITY_BYTES
            while remaining:
                chunk_sizes.append(min(step, remaining))
                remaining -= chunk_sizes[-1]
            for nbytes in chunk_sizes:
                prop = lib._Prop()
                prop.type = _CU_MEM_ALLOC_TYPE_PINNED
                prop.requestedHandleTypes = 0
                prop.loc.type = _CU_MEM_LOCATION_TYPE_HOST_NUMA
                prop.loc.id = numa
                h = ctypes.c_uint64()
                lib.check(
                    lib.cuda.cuMemCreate(ctypes.byref(h), nbytes, ctypes.byref(prop), 0),
                    "cold chunk create",
                )
                handle_values.append(("chunk", h.value))
        elif not self._all_hot:
            cold_bytes = self.size - self.hot_bytes
            prop = lib._Prop()
            prop.type = _CU_MEM_ALLOC_TYPE_PINNED
            prop.requestedHandleTypes = 0
            prop.loc.type = _CU_MEM_LOCATION_TYPE_HOST_NUMA
            prop.loc.id = numa
            h = ctypes.c_uint64()
            lib.check(
                lib.cuda.cuMemCreate(ctypes.byref(h), cold_bytes, ctypes.byref(prop), 0),
                "cold region create",
            )
            handle_values.append(("cold", h.value))

        va = ctypes.c_uint64()
        lib.check(
            lib.cuda.cuMemAddressReserve(
                ctypes.byref(va), self.size, _GRANULARITY_BYTES, 0, 0
            ),
            "cuMemAddressReserve",
        )
        self.va = va.value
        offset = 0
        chunk_iter = iter(chunk_sizes)
        for kind, handle in handle_values:
            if kind == "chunk":
                bytes_here = next(chunk_iter)
                lib.check(lib.cuMemMap(self.va + offset, bytes_here, 0, handle, 0), "chunk map")
                self._handles.append((kind, handle))
                self._mappings[self.va + offset] = (bytes_here, handle)
                self._cold_chunks.append((self.va + offset, bytes_here, handle))
                offset += bytes_here
                continue
            bytes_here = self.hot_bytes if kind == "hot" else self.size - self.hot_bytes
            lib.check(
                lib.cuMemMap(self.va + offset, bytes_here, 0, handle, 0),
                f"{kind} region map",
            )
            self._handles.append((kind, handle))
            self._mappings[self.va + offset] = (bytes_here, handle)
            if kind == "cold":
                self._cold_mapping = (self.va + offset, bytes_here, handle)
            offset += bytes_here

        acc = lib._Acc()
        acc.acc_loc.type = _CU_MEM_LOCATION_TYPE_DEVICE
        acc.acc_loc.id = device_index
        acc.flags = _CU_MEM_ACCESS_PROT_READWRITE
        lib.check(
            lib.cuMemSetAccess(self.va, self.size, ctypes.byref(acc), 1),
            "cuMemSetAccess",
        )

    @property
    def hot_pages(self) -> set[int]:
        return set(range(self.hot_bytes // _GRANULARITY_BYTES))

    def _access(self, address: int, size: int) -> None:
        acc = self.lib._Acc()
        acc.acc_loc.type = _CU_MEM_LOCATION_TYPE_DEVICE
        acc.acc_loc.id = self.device_index
        acc.flags = _CU_MEM_ACCESS_PROT_READWRITE
        self.lib.check(self.lib.cuMemSetAccess(address, size, ctypes.byref(acc), 1), "set access")

    def _map(self, address: int, size: int, handle: int) -> None:
        self.lib.check(self.lib.cuMemMap(address, size, 0, handle, 0), "map")
        self._mappings[address] = (size, handle)
        self._access(address, size)

    def _unmap(self, address: int) -> None:
        size, _ = self._mappings[address]
        self.lib.check(self.lib.cuMemUnmap(address, size), "unmap")
        del self._mappings[address]

    def promote_up_to(self, budget_bytes: int) -> int:
        """Promote the complete cold mapping without changing canonical VA.

        Caller must exclude launches/capture on every stream for this owner.
        CUDA prohibits partial unmaps and nonzero allocation offsets. Thus
        this two-handle owner has a whole-cold-region promotion quantum, NOT
        a page quantum: an insufficient budget returns zero. Retained host
        backing is immutable and makes eviction copy-free.
        """
        if budget_bytes < 0 or self._freed:
            raise ValueError("negative budget or freed VMM owner")
        if self._cold_mapping is None or self._promoted_handle is not None:
            return 0
        address, size, cold = self._cold_mapping
        if budget_bytes < size:
            return 0
        lib = self.lib
        lib.check(lib.cuda.cuCtxSynchronize(), "promotion quiesce")
        prop = lib._Prop()
        prop.type = _CU_MEM_ALLOC_TYPE_PINNED
        prop.loc.type = _CU_MEM_LOCATION_TYPE_DEVICE
        prop.loc.id = self.device_index
        handle = ctypes.c_uint64()
        lib.check(lib.cuda.cuMemCreate(ctypes.byref(handle), size, ctypes.byref(prop), 0), "promotion create")
        hot = handle.value
        self._handles.append(("cache", hot))
        temporary = ctypes.c_uint64()
        reserved = False
        try:
            lib.check(lib.cuda.cuMemAddressReserve(ctypes.byref(temporary), size, _GRANULARITY_BYTES, 0, 0), "promotion reserve")
            reserved = True
            self._map(temporary.value, size, hot)
            lib.check(lib.cuda.cuMemcpyDtoD_v2(temporary.value, address, size), "promotion copy")
            lib.check(lib.cuda.cuCtxSynchronize(), "promotion copy complete")
            self._unmap(address)
            try:
                self._map(address, size, hot)
            except Exception:
                if address in self._mappings:
                    self._unmap(address)
                self._map(address, size, cold)
                raise
            self._promoted_handle = hot
            self.hot_bytes += size
        finally:
            if temporary.value in self._mappings:
                self._unmap(temporary.value)
            if reserved:
                lib.check(lib.cuda.cuMemAddressFree(temporary.value, size), "promotion temporary free")
            if not any(h == hot for _, h in self._mappings.values()):
                lib.check(lib.cuda.cuMemRelease(hot), "promotion unused release")
                self._handles.remove(("cache", hot))
        return size

    @property
    def num_chunks(self) -> int:
        return len(self._cold_chunks)

    def promote_chunk(self) -> int:
        """Promote the next cold chunk to device memory at its canonical VA.

        Caller excludes launches on every stream. Returns the chunk's bytes,
        or 0 when no chunk remains. The chunk's host backing is released
        (chunks are never evicted) unless DS41_VMM_KEEP_PROMOTED_HOST=1.
        """
        if self._freed or self._chunk_promoted >= len(self._cold_chunks):
            return 0
        address, size, cold = self._cold_chunks[self._chunk_promoted]
        lib = self.lib
        lib.check(lib.cuda.cuCtxSynchronize(), "chunk promotion quiesce")
        prop = lib._Prop()
        prop.type = _CU_MEM_ALLOC_TYPE_PINNED
        prop.loc.type = _CU_MEM_LOCATION_TYPE_DEVICE
        prop.loc.id = self.device_index
        handle = ctypes.c_uint64()
        lib.check(lib.cuda.cuMemCreate(ctypes.byref(handle), size, ctypes.byref(prop), 0), "chunk promotion create")
        hot = handle.value
        self._handles.append(("chunk-hot", hot))
        temporary = ctypes.c_uint64()
        lib.check(lib.cuda.cuMemAddressReserve(ctypes.byref(temporary), size, _GRANULARITY_BYTES, 0, 0), "chunk promotion reserve")
        try:
            self._map(temporary.value, size, hot)
            lib.check(lib.cuda.cuMemcpyDtoD_v2(temporary.value, address, size), "chunk promotion copy")
            lib.check(lib.cuda.cuCtxSynchronize(), "chunk promotion copy complete")
            self._unmap(temporary.value)
            self._unmap(address)
            self._map(address, size, hot)
        finally:
            if temporary.value in self._mappings:
                self._unmap(temporary.value)
            lib.check(lib.cuda.cuMemAddressFree(temporary.value, size), "chunk promotion temporary free")
        if os.environ.get("DS41_VMM_KEEP_PROMOTED_HOST", "0") != "1":
            lib.check(lib.cuda.cuMemRelease(cold), "chunk promotion host release")
            self._handles.remove(("chunk", cold))
        self._chunk_promoted += 1
        self.hot_bytes += size
        return size

    def evict_promoted(self) -> int:
        """Restore immutable retained RAM backing; caller excludes launches."""
        if self._freed:
            raise ValueError("freed VMM owner")
        if self._promoted_handle is None:
            return 0
        address, size, cold = self._cold_mapping
        hot = self._promoted_handle
        self.lib.check(self.lib.cuda.cuCtxSynchronize(), "eviction quiesce")
        self._unmap(address)
        try:
            self._map(address, size, cold)
        except Exception:
            if address in self._mappings:
                self._unmap(address)
            self._map(address, size, hot)
            raise
        self._promoted_handle = None
        self.hot_bytes -= size
        self.lib.check(self.lib.cuda.cuMemRelease(hot), "eviction release")
        self._handles.remove(("cache", hot))
        return size

    def set_page_residency(self, page_index: int, hot: bool) -> None:
        raise _VmmError("partial remaps unsupported; use promote_up_to/evict_promoted")

    def wrap_torch(self, torch, numel: int, dtype=None):
        """Wrap the VA range into a torch CUDA tensor (from_blob, no deleter)."""
        ext = _get_wrap_ext(torch)
        if dtype is None:
            dtype = torch.float32
        itemsize = torch.tensor([], dtype=dtype).element_size()
        shape = [numel]
        t = ext.wrap_ptr(self.va, shape, self.device_index, itemsize, str(dtype).split(".")[-1])
        return t

    def free(self) -> None:
        """Release tracked mappings/handles once, after caller quiescence."""
        if self._freed:
            return
        for address in tuple(self._mappings):
            self._unmap(address)
        for kind, handle in tuple(self._handles):
            self.lib.check(self.lib.cuda.cuMemRelease(handle), "free handle")
            self._handles.remove((kind, handle))
        self.lib.check(self.lib.cuda.cuMemAddressFree(self.va, self.size), "free VA")
        self._freed = True

    def __repr__(self) -> str:
        return (
            f"MixedResidencyAllocation(va=0x{self.va:x}, pages={self.num_pages}, "
            f"hot_bytes={self.hot_bytes}/{self.size})"
        )


_WRAP_EXT = None
_WRAP_LOCK = threading.Lock()


class LazyHostTable:
    """Device-visible host table whose pages are created only when mapped.

    Reserves the full VA range and maps HOST_NUMA chunks on demand: creating
    a HOST_NUMA handle commits its host RAM, so unmapped chunks cost nothing.
    The first chunk is mapped at construction (kernel preparation reads its
    leading rows); map_all() backs the rest at the same VA, so tensors and
    lookup bindings built on wrap_torch() stay valid. Touching an unmapped
    chunk faults; callers must not read rows beyond chunk 0 before map_all().
    """

    def __init__(self, num_bytes: int, device_index: int, chunk_bytes: int = 1 << 30):
        if num_bytes <= 0:
            raise ValueError("num_bytes must be positive")
        lib = _VmmLibrary.get()
        lib.ensure_init(device_index)
        self.lib = lib
        self.device_index = device_index
        gran = _GRANULARITY_BYTES
        self.size = (num_bytes + gran - 1) // gran * gran
        step = max(gran, chunk_bytes // gran * gran)
        self.chunks = [(off, min(step, self.size - off)) for off in range(0, self.size, step)]
        self._numa = lib.host_numa_id(device_index)
        self._handles: dict[int, int] = {}
        va = ctypes.c_uint64()
        lib.check(
            lib.cuda.cuMemAddressReserve(ctypes.byref(va), self.size, gran, 0, 0),
            "lazy table reserve",
        )
        self.va = va.value
        self._freed = False
        self._map_chunk(0)

    @property
    def mapped_bytes(self) -> int:
        return sum(self.chunks[i][1] for i in self._handles)

    def _map_chunk(self, index: int) -> None:
        if index in self._handles:
            return
        offset, nbytes = self.chunks[index]
        lib = self.lib
        prop = lib._Prop()
        prop.type = _CU_MEM_ALLOC_TYPE_PINNED
        prop.requestedHandleTypes = 0
        prop.loc.type = _CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.loc.id = self._numa
        handle = ctypes.c_uint64()
        lib.check(
            lib.cuda.cuMemCreate(ctypes.byref(handle), nbytes, ctypes.byref(prop), 0),
            "lazy table chunk create",
        )
        try:
            lib.check(lib.cuMemMap(self.va + offset, nbytes, 0, handle.value, 0), "lazy table map")
        except Exception:
            lib.check(lib.cuda.cuMemRelease(handle.value), "lazy table release")
            raise
        acc = lib._Acc()
        acc.acc_loc.type = _CU_MEM_LOCATION_TYPE_DEVICE
        acc.acc_loc.id = self.device_index
        acc.flags = _CU_MEM_ACCESS_PROT_READWRITE
        lib.check(lib.cuMemSetAccess(self.va + offset, nbytes, ctypes.byref(acc), 1), "lazy table access")
        self._handles[index] = handle.value

    def map_all(self) -> None:
        for index in range(len(self.chunks)):
            self._map_chunk(index)

    def wrap_torch(self, torch, numel: int, dtype=None):
        ext = _get_wrap_ext(torch)
        dtype = torch.uint8 if dtype is None else dtype
        itemsize = torch.tensor([], dtype=dtype).element_size()
        return ext.wrap_ptr(self.va, [numel], self.device_index, itemsize, str(dtype).split(".")[-1])

    def free(self) -> None:
        if self._freed:
            return
        for index, handle in tuple(self._handles.items()):
            offset, nbytes = self.chunks[index]
            self.lib.check(self.lib.cuMemUnmap(self.va + offset, nbytes), "lazy table unmap")
            self.lib.check(self.lib.cuda.cuMemRelease(handle), "lazy table release")
            del self._handles[index]
        self.lib.check(self.lib.cuda.cuMemAddressFree(self.va, self.size), "lazy table free VA")
        self._freed = True


def _get_wrap_ext(torch):
    """Compile (once per process) the tiny from_blob wrapper extension."""
    global _WRAP_EXT
    with _WRAP_LOCK:
        if _WRAP_EXT is not None:
            return _WRAP_EXT
        from torch.utils.cpp_extension import load_inline

        cpp = r"""
#include <torch/extension.h>
#include <ATen/Dispatch.h>
torch::Tensor wrap_ptr(int64_t ptr, std::vector<int64_t> shape,
                       int64_t device_index, int64_t itemsize,
                       std::string dtype_name) {
    auto opts = torch::TensorOptions().device(torch::kCUDA, device_index);
    if (dtype_name == "float32") {
        opts = opts.dtype(torch::kFloat32);
    } else if (dtype_name == "uint8") {
        opts = opts.dtype(torch::kUInt8);
    } else if (dtype_name == "float16") {
        opts = opts.dtype(torch::kFloat16);
    } else if (dtype_name == "bfloat16") {
        opts = opts.dtype(torch::kBFloat16);
    } else if (dtype_name == "int32") {
        opts = opts.dtype(torch::kInt32);
    } else {
        TORCH_CHECK(false, "unsupported dtype ", dtype_name);
    }
    // HOST_NUMA pointers can report device 0 regardless of the CUDA device
    // granted access by cuMemSetAccess. Supply the known target so
    // TensorMaker does not infer a CUDA ordinal from pointer attributes.
    // The caller retains ownership of the VMM allocation.
    return at::from_blob((void*)ptr, shape, [](void*){}, opts, opts.device());
}
"""
        _WRAP_EXT = load_inline(
            name="ds41_vmm_blob_wrap_v4",
            cpp_sources=[cpp],
            functions=["wrap_ptr"],
            with_cuda=True,
            verbose=False,
        )
        return _WRAP_EXT
