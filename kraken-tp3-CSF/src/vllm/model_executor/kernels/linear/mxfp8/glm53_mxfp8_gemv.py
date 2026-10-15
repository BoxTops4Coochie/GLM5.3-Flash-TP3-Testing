"""Small-row A16 GEMV for ModelOpt MXFP8 weights (E4M3 values, UE8M0 scales per 32).

At decode row counts the dense MXFP8 GEMM runs its A16 regime (BF16 activations, weights
dequantized in-kernel) as tiled split-K with a second reduce pass, which is latency-bound.
This kernel follows b12x's block-FP8 GEMV (``b12x/gemm/blockscaled/_block_fp8_gemv.py``):
each CTA owns 16 output features, its 8 warps split K by 128-element chunks, and the warps
reduce through shared memory in a fixed order (deterministic). Differences:

- A16 numerics: weights are expanded to BF16 in registers and multiplied with the BF16
  activations by a BF16 ``m16n8k16`` MMA with FP32 accumulation, as the dense A16 path does.
- The expansion is exact: the E4M3 bit fields are placed into the BF16 fields (``prmt``,
  shift, mask), which yields ``value * 2**-120`` for every E4M3 code, subnormals included
  (b12x's ``fp8x4_e4m3_to_bfloat2x2``); one ``mul.bf16x2`` by ``2**(e - 127 + 120)`` then
  applies the UE8M0 block scale and the bias together (a power of two: exact).
- Scaled before the MMA, so steps need not align with scale blocks: as in the block-FP8 GEMV,
  K is permuted identically for both operands within each 128-element chunk so that thread
  ``quad`` reads the 32 contiguous bytes of block ``quad`` (two 16-byte loads per row) and the
  matching 64 activation bytes, with one scale per thread per chunk.

Reads the packed ``MXFP8LinearWeight`` storage in place (row-major ``[N, Kp]`` values,
``[N, Kp/32]`` UE8M0 rows; ``Kp`` = K padded to 128), so it needs no extra memory.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    bfloat2_mul,
    byte_perm,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    u32_as_f32,
)
from b12x._lib.utils import cuda_stream_from_int_or_current, current_cuda_stream, make_ptr
from b12x.gemm.bf16_gemv._kernel import _flat

MAX_ROWS = 8  # 16/32-row variants exist (tiles=2/4); 16 rows was neutral at C4 served, 32 latency-bound
TILE_ROWS = 8
_WARPS = 8
_LANES = 32
_TYPES = (BFloat16, Uint8, Uint8, BFloat16)
_ALIGN = (16, 16, 16, 2)


def supports(rows: int, out_features: int, in_features: int, padded_in_features: int) -> bool:
    return (
        1 <= rows <= MAX_ROWS
        and out_features > 0
        and in_features > 0
        and in_features % 32 == 0
        and in_features % 8 == 0
        and padded_in_features % 128 == 0
        and padded_in_features >= in_features
    )


@cute.jit
def _e4m3x4_to_bf16x2x2(word: Uint32, scale: Uint32):
    """Four E4M3 bytes of ``word`` as two BF16x2 (bytes 0,1 and 2,3), times ``scale``."""
    spread = byte_perm(word, Uint32(0), Int32(0x1302))
    lo = (spread & Uint32(0x80008000)) | ((spread >> Uint32(4)) & Uint32(0x07F007F0))
    hi = ((spread << Uint32(8)) & Uint32(0x80008000)) | ((spread << Uint32(4)) & Uint32(0x07F007F0))
    return bfloat2_mul(lo, scale), bfloat2_mul(hi, scale)


# Feature tiles per CTA by token tiles: wider CTAs reuse each activation load at 32 rows.
FTILES = {1: 1, 2: 1, 4: 2}


def tiles_for(rows: int) -> int:
    """Token tiles (8 rows each) of the variant that serves ``rows``."""
    return 1 if rows <= 8 else 2 if rows <= 16 else 4


class Mxfp8A16Gemv:
    """``out[m, n] = sum_k x[m, k] * w[n, k] * 2**(s[n, k // 32] - 127)``, BF16 out.

    ``tiles`` 8-row token tiles share every weight load (up to 32 rows)."""

    def __init__(self, n: int, k: int, kp: int, tiles: int = 1, ftiles: int = 1, warps: int = 8):
        if k % 32 or kp % 128 or kp < k:
            raise ValueError("MXFP8 GEMV needs K % 32 == 0 and a 128-padded row stride")
        self.n, self.k, self.kp = int(n), int(k), int(kp)
        if tiles not in (1, 2, 4):
            raise ValueError('tiles must be 1, 2 or 4')
        self.tiles = int(tiles)
        if ftiles not in (1, 2):
            raise ValueError('ftiles must be 1 or 2')
        self.ftiles = int(ftiles)
        self.features = 16 * self.ftiles
        if warps not in (4, 8, 16):
            raise ValueError('warps must be 4, 8 or 16')
        self.warps = int(warps)
        self.blocks = self.k // 32
        self.chunks = (self.blocks + 3) // 4
        self.chunks_per_warp = (self.chunks + self.warps - 1) // self.warps
        self.scale_stride = self.kp // 32

    def _shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "partials": cute.struct.Align[
                cute.struct.MemRange[Float32, self.warps * _LANES * 4 * self.tiles * self.ftiles], 16
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        lhs: cute.Pointer,
        rhs: cute.Pointer,
        rhs_scale: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(_flat(lhs), _flat(rhs), _flat(rhs_scale), _flat(output), rows).launch(
            grid=((self.n + self.features - 1) // self.features, 1, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        lhs: cute.Tensor,
        rhs: cute.Tensor,
        rhs_scale: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = Int32(thread) // _LANES
        lane = Int32(thread) % _LANES
        group = lane // 4
        quad = lane % 4
        first = Int32(block) * self.features
        # Clamped loads keep partial tiles in bounds; their stores are masked.
        rhs_base = get_ptr_as_int64(rhs, Int64(0))
        lhs_base = get_ptr_as_int64(lhs, Int64(0))
        scale_base = get_ptr_as_int64(rhs_scale, Int64(0))
        kp = Int64(self.kp)
        k = Int64(self.k)
        # Weight rows of this thread: per feature tile f, rows first + 16f + group (+8).
        wrows = [
            cutlass.min(first + 16 * (i // 2) + group + 8 * (i % 2), Int32(self.n - 1))
            for i in range(2 * self.ftiles)
        ]
        wptr = [rhs_base + Int64(r) * kp for r in wrows]
        sptr = [scale_base + Int64(r) * Int64(self.scale_stride) for r in wrows]
        sources = [
            lhs_base + Int64(cutlass.min(group + 8 * t, rows - 1)) * k * Int64(2)
            for t in range(self.tiles)
        ]
        acc = [Float32(0.0)] * (4 * self.tiles * self.ftiles)
        for step in cutlass.range_constexpr(self.chunks_per_warp):
            chunk = warp * self.chunks_per_warp + step
            if chunk < self.chunks:
                blk = chunk * 4 + quad
                valid = blk < self.blocks
                nrows = 2 * self.ftiles
                w = [Uint32(0)] * (8 * nrows)
                scales = [Uint32(0)] * nrows
                offset = Int64(blk * 32)
                if valid:
                    shift = Uint32(quad * 8)
                    for r in cutlass.range_constexpr(nrows):
                        for h in cutlass.range_constexpr(2):
                            v0, v1, v2, v3 = ld_global_nc_v4_u32(wptr[r] + offset + Int64(16 * h))
                            i = 8 * r + 4 * h
                            w[i], w[i + 1], w[i + 2], w[i + 3] = v0, v1, v2, v3
                        e = cutlass.min((ld_global_nc_u32(sptr[r] + Int64(chunk * 4)) >> shift) & Uint32(0xFF), Uint32(134))
                        # BF16 2**(e - 7) in both halves: 2**(e - 127) block scale times 2**120 bias.
                        scales[r] = ((e + Uint32(120)) << Uint32(7)) * Uint32(0x00010001)
                for t in cutlass.range_constexpr(self.tiles):
                    xs = [Uint32(0)] * 16
                    if valid:
                        for h in cutlass.range_constexpr(4):
                            v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                sources[t] + offset * Int64(2) + Int64(16 * h))
                            xs[4 * h], xs[4 * h + 1], xs[4 * h + 2], xs[4 * h + 3] = v0, v1, v2, v3
                    for st in cutlass.range_constexpr(8):
                        for f in cutlass.range_constexpr(self.ftiles):
                            a_lo, a_hi = _e4m3x4_to_bf16x2x2(w[8 * (2 * f) + st], scales[2 * f])
                            b_lo, b_hi = _e4m3x4_to_bf16x2x2(w[8 * (2 * f + 1) + st], scales[2 * f + 1])
                            j = 4 * (t * self.ftiles + f)
                            acc[j], acc[j + 1], acc[j + 2], acc[j + 3] = bf16_mma_m16n8k16_f32(
                                acc[j], acc[j + 1], acc[j + 2], acc[j + 3],
                                a_lo, b_lo, a_hi, b_hi, xs[2 * st], xs[2 * st + 1],
                            )

        storage = cutlass.utils.SmemAllocator().allocate(self._shared_storage_cls())
        width = 4 * self.tiles * self.ftiles
        partials = storage.partials.get_tensor(cute.make_layout((self.warps * _LANES * width,)))
        base = (warp * _LANES + lane) * width
        for e in cutlass.range_constexpr(width):
            partials[base + e] = acc[e]
        cute.arch.sync_threads()
        if warp == 0:
            for e in cutlass.range_constexpr(width):
                total = Float32(0.0)
                for w in cutlass.range_constexpr(self.warps):
                    total = total + partials[Int32(w * _LANES * width) + lane * width + e]
                tile_t = (e // 4) // self.ftiles
                tile_f = (e // 4) % self.ftiles
                feature = first + 16 * tile_f + group + (8 if e % 4 >= 2 else 0)
                row = quad * 2 + (e % 2) + 8 * tile_t
                if feature < self.n and row < rows:
                    output[Int64(row) * self.n + Int64(feature)] = total.to(BFloat16)


_PROGRAMS = {}


def _compile(ordinal: int, n: int, k: int, kp: int, tiles: int, ftiles: int, warps: int = 8):
    key = (ordinal, n, k, kp, tiles, ftiles, warps)
    if key not in _PROGRAMS:
        fake = tuple(
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=align)
            for dtype, align in zip(_TYPES, _ALIGN, strict=True)
        )
        with torch.cuda.device(ordinal):
            _PROGRAMS[key] = b12x_compile(
                Mxfp8A16Gemv(n, k, kp, tiles, ftiles, warps), *fake, Int32(1), current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key("gemm.mxfp8_a16.gemv", 7, key),
            )
    return _PROGRAMS[key]


def compile_mxfp8_gemv(ordinal: int, n: int, k: int, kp: int, max_rows: int = MAX_ROWS):
    """Compile the variants serving up to ``max_rows`` rows; the runner picks by row count."""
    raws = {t: _compile(ordinal, n, k, kp, t, FTILES[t]) for t in (1, 2, 4) if t == 1 or t <= tiles_for(max_rows)}

    def run(lhs, rhs_values, rhs_scale_rows, output, stream=None):
        tensors = (lhs, rhs_values, rhs_scale_rows, output)
        pointers = tuple(
            make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)
            for dtype, align, tensor in zip(_TYPES, _ALIGN, tensors, strict=True)
        )
        rows = int(lhs.shape[0])
        run_compiled(raws[tiles_for(rows)], (*pointers, Int32(rows), cuda_stream_from_int_or_current(stream)))

    return run
