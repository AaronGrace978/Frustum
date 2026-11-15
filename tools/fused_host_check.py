"""Developer check for fused_kernels.SOURCE without a GPU.

Compiles the exact CUDA source with g++ behind a small shim (each warp runs as 32
std::threads so __shfl_down_sync behaves as on hardware), then compares against
NumPy ports of the PyTorch reference decoders:

- one-hot activations: every output must equal one decoded weight, bit for bit
- random activations: matrix-vector products agree to rounding of the summation order

    python tools/fused_host_check.py        (needs g++ with C++20 and _Float16, and numpy)

On the target machine the same comparison runs on the GPU automatically on first use.
"""
import ctypes
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fused_kernels import SOURCE  # noqa: E402

SHIM = r'''
#include <barrier>
#include <cstring>
#include <thread>
#include <vector>
struct dim { unsigned x, y, z; };
thread_local dim threadIdx, blockIdx;
dim blockDim = {256, 1, 1};
static std::barrier<>* warp_barrier;
static float lanes[32];
static float __shfl_down_sync(unsigned, float v, int offset) {
    int lane = threadIdx.x & 31;
    lanes[lane] = v; warp_barrier->arrive_and_wait();
    float r = lane + offset < 32 ? lanes[lane + offset] : v; warp_barrier->arrive_and_wait();
    return r;
}
static inline unsigned __float_as_uint(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
static inline float __uint_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline float host_h2f(unsigned short h) { _Float16 v; std::memcpy(&v, &h, 2); return (float)v; }
static inline float host_round_half(float v) { return (float)(_Float16)v; }
#define __device__
#define __forceinline__ inline
#define __global__
#define __restrict__
#define __constant__ static const
'''

DRIVER = r'''
template <typename F> static void run(int rows, F kernel) {
    std::barrier<> bar(32); warp_barrier = &bar;
    int warps_per_block = blockDim.x / 32, blocks = (rows + warps_per_block - 1) / warps_per_block;
    for (int b = 0; b < blocks; ++b)
        for (int w = 0; w < warps_per_block; ++w) {
            std::vector<std::thread> pool;
            for (int lane = 0; lane < 32; ++lane)
                pool.emplace_back([&, b, w, lane] {
                    blockIdx = {(unsigned)b, 0, 0}; threadIdx = {(unsigned)(w * 32 + lane), 0, 0};
                    kernel();
                });
            for (auto& t : pool) t.join();
        }
}
extern "C" void host_gguf(const u8* W, int kind, int rows, long long row_bytes, int units, const float* X, int T,
                          int K, float* Y, int mode) {
    run(rows, [&] { gguf_matvec(W, kind, rows, row_bytes, units, X, T, K, Y, mode); });
}
extern "C" void host_block(const u8* W, const u8* S, int fp4, int rows, int cols, const float* X, int T, float* Y) {
    run(rows, [&] { block_matvec(W, S, fp4, rows, cols, X, T, Y); });
}
'''


def host_source():
    src = SOURCE
    src = re.sub(r'__device__ __forceinline__ float h2f\(u16 h\) \{.*?\n\}',
                 '__device__ __forceinline__ float h2f(u16 h) { return host_h2f(h); }', src, flags=re.S)
    src = src.replace('if (mode == 1) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v)); return h2f(h); }',
                      'if (mode == 1) return host_round_half(v);')
    assert 'asm(' not in src, 'unshimmed inline PTX left'
    src = src.replace('extern "C" __global__ void', 'static void')
    return SHIM + src + DRIVER


def build(workdir):
    cpp = Path(workdir) / 'fused_host.cpp'
    lib = Path(workdir) / 'fused_host.so'
    cpp.write_text(host_source())
    subprocess.run(['g++', '-std=c++20', '-O1', '-ffp-contract=off', '-shared', '-fPIC', '-pthread',
                    str(cpp), '-o', str(lib)], check=True)
    return ctypes.CDLL(str(lib))


# ---- NumPy ports of the PyTorch reference decoders, float32 op for op ----
def ref_gguf(raw, kind):
    if kind == 'Q8_0':
        blocks = raw.reshape(-1, 34)
        scale = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        return (scale * blocks[:, 2:].view(np.int8).astype(np.float32)).reshape(raw.shape[0], -1)
    rows = raw.shape[0]
    width = {'Q4_K': 144, 'Q5_K': 176, 'Q6_K': 210}[kind]
    blocks = raw.reshape(-1, width)
    shifts = np.arange(8, dtype=np.uint8)
    if kind == 'Q6_K':
        low = (blocks[:, :128].reshape(-1, 2, 1, 64) >> (shifts[:2] * 4).reshape(1, 1, 2, 1)) & 15
        high = (blocks[:, 128:192].reshape(-1, 2, 1, 32) >> (shifts[:4] * 2).reshape(1, 1, 4, 1)) & 3
        codes = (low.reshape(-1, 16, 16) | (high.reshape(-1, 16, 16) << 4)).astype(np.float32) - 32
        scale = blocks[:, 192:208].view(np.int8).astype(np.float32)
        factor = blocks[:, 208:210].copy().view(np.float16).astype(np.float32)
        return (codes * (scale * factor)[..., None]).reshape(rows, -1)
    factors = blocks[:, :4].copy().view(np.float16).astype(np.float32)
    d, m, tail = blocks[:, 4:8], blocks[:, 8:12], blocks[:, 12:16]
    scale = np.concatenate((d & 63, (tail & 15) | ((d >> 2) & 48)), 1).astype(np.float32)
    minimum = np.concatenate((m & 63, (tail >> 4) | ((m >> 2) & 48)), 1).astype(np.float32)
    codes = (blocks[:, width - 128:].reshape(-1, 4, 1, 32) >> (shifts[:2] * 4).reshape(1, 1, 2, 1)) & 15
    codes = codes.reshape(-1, 8, 32)
    if kind == 'Q5_K':
        codes = codes | (((blocks[:, None, 16:48] >> shifts.reshape(1, 8, 1)) & 1) << 4)
    return ((scale * factors[:, :1])[..., None] * codes.astype(np.float32)
            - (minimum * factors[:, 1:])[..., None]).reshape(rows, -1)


def ref_fp4(packed, scale):
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    out = np.empty((packed.shape[0], packed.shape[1] * 2), np.float32)
    out[:, 0::2] = lut[packed & 15]
    out[:, 1::2] = lut[packed >> 4]
    factor = np.exp2(scale.astype(np.float32) - 127).astype(np.float32)
    return (out.reshape(out.shape[0], -1, 32) * factor[..., None]).reshape(out.shape)


def ref_fp8(raw, scale):
    e = ((raw >> 3) & 15).astype(np.int32)
    m = (raw & 7).astype(np.float64)
    mag = np.where(e == 0, np.ldexp(m, -9), np.ldexp(1 + m / 8, e - 7))
    value = np.where(raw & 128, -mag, mag).astype(np.float32)
    value[(raw & 127) == 127] = np.nan
    factor = np.exp2(scale.astype(np.float32) - 127).astype(np.float32)
    return value * np.repeat(np.repeat(factor, 32, 0), 32, 1)


def realistic_gguf(rng, kind, rows, cols):
    bytes_per, per = {'Q4_K': (144, 256), 'Q5_K': (176, 256), 'Q6_K': (210, 256), 'Q8_0': (34, 32)}[kind]
    raw = rng.integers(0, 256, (rows, cols // per * bytes_per), dtype=np.uint8)
    blocks = raw.reshape(-1, bytes_per)
    halves = rng.uniform(1e-4, 2e-2, (blocks.shape[0], 2)).astype(np.float16).view(np.uint8)
    if kind == 'Q6_K':
        blocks[:, 208:210] = halves[:, :2]
    elif kind == 'Q8_0':
        blocks[:, 0:2] = halves[:, :2]
    else:
        blocks[:, 0:4] = halves
    return raw


def main():
    rng = np.random.default_rng(7)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        lib = build(tmp)
        f32 = np.float32

        def gguf(raw, kind, x, mode):
            rows, row_bytes = raw.shape
            T, K = x.shape
            y = np.zeros((T, rows), f32)
            code = {'Q4_K': 0, 'Q5_K': 1, 'Q6_K': 2, 'Q8_0': 3}[kind]
            lib.host_gguf(raw.ctypes.data, code, rows, ctypes.c_longlong(row_bytes), (K + 255) // 256,
                          x.ctypes.data, T, K, y.ctypes.data, mode)
            return y

        def block(w, s, fp4, x):
            T, cols = x.shape
            y = np.zeros((T, w.shape[0]), f32)
            lib.host_block(w.ctypes.data, s.ctypes.data, int(fp4), w.shape[0], cols, x.ctypes.data, T, y.ctypes.data)
            return y

        def check(name, got, decoded, x, onehot_cols=None):
            nonlocal failures
            if onehot_cols is not None:
                want = decoded[:, onehot_cols].T
                same = np.array_equal(got.view(np.uint32), want.astype(f32).view(np.uint32)) or \
                    np.array_equal(np.nan_to_num(got), np.nan_to_num(want))
                status = 'exact' if same else 'MISMATCH'
                failures += not same
            else:
                want = x.astype(np.float64) @ decoded.astype(np.float64).T
                err = np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)
                status = f'rel {err:.1e}' + ('' if err < 1e-5 else '  MISMATCH')
                failures += err >= 1e-5
            print(f'{name:<34} {status}')

        for kind, cols in (('Q4_K', 512), ('Q5_K', 768), ('Q6_K', 512), ('Q8_0', 288)):
            raw = realistic_gguf(rng, kind, 37, cols)
            decoded = ref_gguf(raw, kind)
            for mode, label in ((0, 'fp32'), (1, 'fp16')):
                w = decoded if mode == 0 else decoded.astype(np.float16).astype(f32)
                for trial in range(3):
                    picks = rng.choice(cols, 8, replace=False)
                    onehot = np.zeros((8, cols), f32)
                    onehot[np.arange(8), picks] = 1
                    check(f'{kind} {label} one-hot {trial}', gguf(raw, kind, onehot, mode), w, onehot, picks)
                x = rng.standard_normal((3, cols)).astype(f32)
                check(f'{kind} {label} random T=3', gguf(raw, kind, x, mode), w, x)
            # Whole-matrix bit check through one-hot rows covering every column.
            for start in range(0, cols, 8):
                picks = np.arange(start, min(start + 8, cols))
                onehot = np.zeros((len(picks), cols), f32)
                onehot[np.arange(len(picks)), picks] = 1
                got = gguf(raw, kind, onehot, 0)
                if not np.array_equal(got, decoded[:, picks].T):
                    failures += 1
                    print(f'{kind} full sweep MISMATCH at columns {picks}')
                    break
            else:
                print(f'{kind + " all columns bit-exact":<34} exact')

        rows, cols = 64, 512
        packed = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
        scale = rng.integers(110, 140, (rows, cols // 32), dtype=np.uint8)
        scale[0, 0] = 0  # 2^-127, a subnormal factor
        decoded = ref_fp4(packed, scale)
        for start in range(0, cols, 8):
            picks = np.arange(start, start + 8)
            onehot = np.zeros((8, cols), f32)
            onehot[np.arange(8), picks] = 1
            if not np.array_equal(block(packed, scale, True, onehot), decoded[:, picks].T):
                failures += 1
                print('FP4 sweep MISMATCH', picks)
                break
        else:
            print(f'{"FP4 all columns bit-exact":<34} exact')
        x = rng.standard_normal((5, cols)).astype(f32)
        check('FP4 random T=5', block(packed, scale, True, x), decoded, x)

        raw = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
        raw[(raw & 127) == 127] = 0
        scale = rng.integers(110, 140, (rows // 32, cols // 32), dtype=np.uint8)
        decoded = ref_fp8(raw, scale)
        for start in range(0, cols, 8):
            picks = np.arange(start, start + 8)
            onehot = np.zeros((8, cols), f32)
            onehot[np.arange(8), picks] = 1
            if not np.array_equal(block(raw, scale, False, onehot), decoded[:, picks].T):
                failures += 1
                print('FP8 sweep MISMATCH', picks)
                break
        else:
            print(f'{"FP8 all columns bit-exact":<34} exact')
        x = rng.standard_normal((8, cols)).astype(f32)
        check('FP8 random T=8', block(raw, scale, False, x), decoded, x)
        odd = rng.integers(0, 256, (32, 96), dtype=np.uint8)
        odd[(odd & 127) == 127] = 0
        oscale = rng.integers(120, 130, (1, 3), dtype=np.uint8)
        x = rng.standard_normal((2, 96)).astype(f32)
        check('FP8 cols=96 (partial chunk)', block(odd, oscale, False, x), ref_fp8(odd, oscale), x)
    print('PASS' if not failures else f'FAIL ({failures})')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
