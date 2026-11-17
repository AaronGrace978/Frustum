"""Fused dequantize-and-multiply CUDA kernels for decode steps (Frustum_loader v0.6).

Why
    Every backend used to turn a packed weight matrix into a full FP32 copy on every
    token and then multiply. For a token that only multiplies one or two rows of
    activations, writing and re-reading those copies costs far more memory traffic
    than the packed bytes themselves. These kernels read the packed bytes once,
    decode them in registers and accumulate the dot products directly.

How
    The CUDA source below is compiled at load time with NVRTC, which ships inside
    the PyTorch CUDA wheel, and launched through the CUDA driver API on PyTorch's
    current stream. Nothing new has to be installed.

Formats
    gguf  Q4_K, Q5_K, Q6_K, Q8_0 rows (Phi-4, GLM-4.7-Flash)
    fp8   E4M3 with E8M0 scales per 32x32 block (DeepSeek dense and shared experts)
    fp4   packed E2M1 with E8M0 scales per 32 columns (DeepSeek routed experts)

Numerics
    Each weight is decoded with the same operations, in the same order, as the PyTorch
    reference decoders, with fused multiply-add disabled, so decoded values are
    bit-identical. When the reference converts weights to FP16 before multiplying,
    the kernel rounds them to FP16 too. Only the order of the final summation differs
    from cuBLAS, which changes results in the last bits.

Safety
    The first call for each (format, shape) also runs the reference path and compares.
    Any mismatch, compile failure or launch error turns the fused path off for the
    rest of the process and the reference path carries on. FRUSTUM_FUSED=0 turns it
    off from the start. Only inputs of at most MAX_TOKENS rows use the kernels;
    prefill keeps the reference path, where cuBLAS is the better tool.
"""
import ctypes
import ctypes.util
import os
from pathlib import Path
import sys
import threading

MAX_TOKENS = 8
GGUF_KINDS = {'Q4_K': (0, 144, 256), 'Q5_K': (1, 176, 256), 'Q6_K': (2, 210, 256), 'Q8_0': (3, 34, 32)}
ROUND = {'none': 0, 'fp16': 1, 'bf16': 2}

SOURCE = r'''
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef signed char i8;
#define MAXT 8

__device__ __forceinline__ float h2f(u16 h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ float half_at(const u8* p) { return h2f((u16)p[0] | ((u16)p[1] << 8)); }
__device__ __forceinline__ float round_to(float v, int mode) {
    if (mode == 1) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v)); return h2f(h); }
    if (mode == 2) {
        u32 u = __float_as_uint(v);
        if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
        return __uint_as_float(u & 0xffff0000u);
    }
    return v;
}
__device__ __forceinline__ float e8m0(u8 s) {
    // 2^(s-127); s == 255 is rejected on the host before launch.
    return s == 0 ? __uint_as_float(0x00400000u) : __uint_as_float((u32)s << 23);
}
__device__ __forceinline__ float e4m3fn(u8 b) {
    u32 e = (b >> 3) & 15u, m = b & 7u;
    float mag;
    if ((b & 127u) == 127u) return __uint_as_float(0x7fc00000u);
    if (e == 0) mag = (float)m * 0.001953125f;                       // m * 2^-9
    else mag = __uint_as_float(((e + 120u) << 23) | (m << 20));      // (1 + m/8) * 2^(e-7)
    return (b & 128u) ? -mag : mag;
}
__constant__ float FP4[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                              -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// Eight consecutive weights of one 256-weight super-block, for one lane (lane*8 .. lane*8+7).
__device__ __forceinline__ void q4k(const u8* b, int lane, float* w, bool five) {
    int j = lane >> 2, e0 = (lane & 3) * 8;
    float d = half_at(b), dmin = half_at(b + 2);
    const u8 *D = b + 4, *M = b + 8, *T = b + 12;
    u32 s, m;
    if (j < 4) { s = D[j] & 63u; m = M[j] & 63u; }
    else { s = (T[j - 4] & 15u) | ((D[j - 4] >> 2) & 48u); m = (T[j - 4] >> 4) | ((M[j - 4] >> 2) & 48u); }
    float a = (float)s * d, c = (float)m * dmin;
    const u8* qs = b + (five ? 48 : 16) + 32 * (j >> 1);
    int shift = (j & 1) * 4;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        u32 q = (qs[e0 + i] >> shift) & 15u;
        if (five) q |= ((b[16 + e0 + i] >> j) & 1u) << 4;
        w[i] = a * (float)q - c;
    }
}
__device__ __forceinline__ void q6k(const u8* b, int lane, float* w) {
    int L0 = lane * 8;
    float scale = (float)(i8)b[192 + (L0 >> 4)] * half_at(b + 208);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int L = L0 + i, h = L >> 7, rem = L & 127;
        u32 low = (b[h * 64 + (rem & 63)] >> (4 * (rem >> 6))) & 15u;
        u32 high = (b[128 + h * 32 + (rem & 31)] >> (2 * (rem >> 5))) & 3u;
        w[i] = ((float)(low | (high << 4)) - 32.f) * scale;
    }
}

// One warp per output row. y[t * rows + row] = sum_k W[row, k] * x[t * K + k].
extern "C" __global__ void gguf_matvec(const u8* __restrict__ W, int kind, int rows, long long row_bytes,
                                       int units, const float* __restrict__ X, int T, int K,
                                       float* __restrict__ Y, int mode) {
    int row = (int)((blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const u8* base = W + (long long)row * row_bytes;
    float acc[MAXT];
    #pragma unroll
    for (int t = 0; t < MAXT; ++t) acc[t] = 0.f;
    int nblocks = K >> 5;
    for (int u = 0; u < units; ++u) {
        float w[8];
        int col = u * 256 + lane * 8;
        if (kind == 3) {
            int blk = u * 8 + (lane >> 2);
            if (blk >= nblocks) continue;
            const u8* b = base + (long long)blk * 34;
            float d = half_at(b);
            int e0 = (lane & 3) * 8;
            #pragma unroll
            for (int i = 0; i < 8; ++i) w[i] = d * (float)(i8)b[2 + e0 + i];
        } else if (kind == 2) {
            q6k(base + (long long)u * 210, lane, w);
        } else {
            q4k(base + (long long)u * (kind == 1 ? 176 : 144), lane, w, kind == 1);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = round_to(w[i], mode);
        #pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t < T) {
                const float* x = X + (long long)t * K + col;
                float s = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) s += w[i] * x[i];
                acc[t] += s;
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t < T) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) Y[(long long)t * rows + row] = v;
        }
    }
}

// E4M3 weights [rows, cols] with E8M0 scales [rows/32, cols/32] (fp8) or
// packed E2M1 weights [rows, cols/2] with E8M0 scales [rows, cols/32] (fp4).
extern "C" __global__ void block_matvec(const u8* __restrict__ W, const u8* __restrict__ S, int fp4,
                                        int rows, int cols, const float* __restrict__ X, int T,
                                        float* __restrict__ Y) {
    int row = (int)((blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    int scols = cols >> 5;
    float acc[MAXT];
    #pragma unroll
    for (int t = 0; t < MAXT; ++t) acc[t] = 0.f;
    for (int c0 = 0; c0 < cols; c0 += 256) {
        int col = c0 + lane * 8;
        if (col >= cols) break;
        float w[8];
        if (fp4) {
            float f = e8m0(S[(long long)row * scols + (col >> 5)]);
            const u8* p = W + (long long)row * (cols >> 1) + (col >> 1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                u8 b = p[i];
                w[2 * i] = FP4[b & 15] * f;
                w[2 * i + 1] = FP4[b >> 4] * f;
            }
        } else {
            float f = e8m0(S[(long long)(row >> 5) * scols + (col >> 5)]);
            const u8* p = W + (long long)row * cols + col;
            #pragma unroll
            for (int i = 0; i < 8; ++i) w[i] = e4m3fn(p[i]) * f;
        }
        #pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t < T) {
                const float* x = X + (long long)t * cols + col;
                float s = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) s += w[i] * x[i];
                acc[t] += s;
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t < T) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) Y[(long long)t * rows + row] = v;
        }
    }
}

// v0.10: expert_correctness.activation in one launch instead of about thirteen. Same operations in
// the same order: per 32-value group, scale = exp2(ceil(log2(max(amax|x|, 1e-4) / 448))), then
// x / scale clamped to +-448, rounded to FP8 E4M3 (round to nearest even, E4M3 subnormals
// included), times scale. amax propagates NaN like torch. Checked bit for bit against the
// PyTorch version before use (Fused.activation).
__device__ __forceinline__ float round_e4m3(float q) {
    if (q != q) return q;
    if (fabsf(q) < 0.015625f) return rintf(q * 512.0f) / 512.0f;   // below 2^-6: steps of 2^-9
    u32 u = __float_as_uint(q);
    u += 0x7ffffu + ((u >> 20) & 1u);                               // keep 3 mantissa bits, ties to even
    return __uint_as_float(u & 0xfff00000u);
}
extern "C" __global__ void act_quant(const void* __restrict__ X, int bf16, long long n, float* __restrict__ Y) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;                                             // n % 32 == 0: whole warps only
    float v = bf16 ? __uint_as_float(((u32)((const u16*)X)[i]) << 16) : ((const float*)X)[i];
    float m = fabsf(v);
    int nan = v != v;
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, m, o);
        m = other > m ? other : m;
        nan |= __shfl_xor_sync(0xffffffffu, nan, o);
    }
    if (nan) m = __int_as_float(0x7fc00000);
    float c = m < 1e-4f ? 1e-4f : m;
    float s = exp2f(ceilf(log2f(c / 448.0f)));
    float q = v / s;
    q = q < -448.0f ? -448.0f : (q > 448.0f ? 448.0f : q);
    Y[i] = round_e4m3(q) * s;
}
'''


def _note(message):
    print(f'[frustum fused] {message}', file=sys.stderr, flush=True)


def _library(names):
    for name in names:
        try:
            return ctypes.CDLL(str(name))
        except OSError:
            continue
    return None


def _nvrtc_candidates(torch):
    lib = Path(torch.__file__).resolve().parent / 'lib'
    found = []
    if os.name == 'nt':
        found += sorted((p for p in lib.glob('nvrtc64_*.dll') if 'alt' not in p.name and 'builtins' not in p.name),
                        reverse=True)
    else:
        site = lib.parent.parent
        found += sorted(lib.glob('libnvrtc*.so*'), reverse=True)
        found += sorted((site / 'nvidia' / 'cuda_nvrtc' / 'lib').glob('libnvrtc.so*'), reverse=True)
    for name in ('nvrtc', 'nvrtc64_120_0'):
        path = ctypes.util.find_library(name)
        if path:
            found.append(path)
    return found


class _Compiled:
    """NVRTC compile once, then a module per CUDA context (one in practice)."""

    def __init__(self, torch):
        self.t = torch
        if os.name == 'nt':
            lib = Path(torch.__file__).resolve().parent / 'lib'
            if hasattr(os, 'add_dll_directory') and lib.is_dir():
                os.add_dll_directory(str(lib))
        self.nvrtc = _library(_nvrtc_candidates(torch))
        if self.nvrtc is None:
            raise OSError('NVRTC library not found next to PyTorch')
        self.cuda = _library(['nvcuda.dll'] if os.name == 'nt' else ['libcuda.so.1', 'libcuda.so'])
        if self.cuda is None:
            raise OSError('CUDA driver library not found')
        self.nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
        self.ptx = self._compile()
        self.lock = threading.Lock()
        self.modules = {}

    def _check_nvrtc(self, result, prog=None):
        if result:
            log = b''
            if prog is not None:
                size = ctypes.c_size_t()
                self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
                buffer = ctypes.create_string_buffer(size.value)
                self.nvrtc.nvrtcGetProgramLog(prog, buffer)
                log = buffer.value
            raise RuntimeError(f'NVRTC: {self.nvrtc.nvrtcGetErrorString(result).decode()} {log.decode(errors="replace")}')

    def _compile(self):
        major, minor = self.t.cuda.get_device_capability()
        last = None
        for arch in (f'compute_{major}{minor}', 'compute_90', 'compute_80'):
            prog = ctypes.c_void_p()
            self._check_nvrtc(self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), SOURCE.encode(), b'frustum_fused.cu',
                                                            0, None, None))
            options = [f'--gpu-architecture={arch}'.encode(), b'--fmad=false', b'-default-device']
            array = (ctypes.c_char_p * len(options))(*options)
            try:
                self._check_nvrtc(self.nvrtc.nvrtcCompileProgram(prog, len(options), array), prog)
                size = ctypes.c_size_t()
                self._check_nvrtc(self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size)))
                buffer = ctypes.create_string_buffer(size.value)
                self._check_nvrtc(self.nvrtc.nvrtcGetPTX(prog, buffer))
                return buffer.raw
            except RuntimeError as error:
                last = error
            finally:
                self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        raise last

    def _check_cuda(self, result, what):
        if result:
            text = ctypes.c_char_p()
            self.cuda.cuGetErrorString(result, ctypes.byref(text))
            raise RuntimeError(f'CUDA {what}: {(text.value or b"error").decode()} ({result})')

    def functions(self):
        context = ctypes.c_void_p()
        self._check_cuda(self.cuda.cuCtxGetCurrent(ctypes.byref(context)), 'cuCtxGetCurrent')
        if not context.value:
            device = ctypes.c_int()
            self._check_cuda(self.cuda.cuDeviceGet(ctypes.byref(device), self.t.cuda.current_device()), 'cuDeviceGet')
            self._check_cuda(self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), 'cuDevicePrimaryCtxRetain')
            self._check_cuda(self.cuda.cuCtxSetCurrent(context), 'cuCtxSetCurrent')
        with self.lock:
            entry = self.modules.get(context.value)
            if entry is None:
                module = ctypes.c_void_p()
                self._check_cuda(self.cuda.cuModuleLoadData(ctypes.byref(module), self.ptx), 'cuModuleLoadData')
                entry = {}
                for name in ('gguf_matvec', 'block_matvec', 'act_quant'):
                    function = ctypes.c_void_p()
                    self._check_cuda(self.cuda.cuModuleGetFunction(ctypes.byref(function), module, name.encode()),
                                     'cuModuleGetFunction')
                    entry[name] = function
                self.modules[context.value] = entry
        return entry

    def launch(self, name, rows, args):
        function = self.functions()[name]
        values = [a if isinstance(a, ctypes._SimpleCData) else ctypes.c_int(a) for a in args]
        params = (ctypes.c_void_p * len(values))(*[ctypes.addressof(v) for v in values])
        threads = 256
        blocks = (rows * 32 + threads - 1) // threads
        stream = ctypes.c_void_p(self.t.cuda.current_stream().cuda_stream)
        self._check_cuda(self.cuda.cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, stream, params, None),
                         'cuLaunchKernel')


class Fused:
    def __init__(self, torch):
        self.t = torch
        self.enabled = os.environ.get('FRUSTUM_FUSED', '1').strip().lower() not in ('0', 'false', 'off', 'no')
        self.compiled = None
        self.checked = set()
        self.calls = 0
        self.reason = None if self.enabled else 'FRUSTUM_FUSED=0'
        # v0.10 fused activation quantizer: on unless FRUSTUM_FUSED_ACT=0; switches only itself off.
        self.act_enabled = os.environ.get('FRUSTUM_FUSED_ACT', '1').strip().lower() not in ('0', 'false', 'off', 'no')
        self.act_reason = None if self.act_enabled else 'FRUSTUM_FUSED_ACT=0'
        self.act_tested = False
        self.act_compared = 0
        self.act_calls = 0

    ACT_COMPARE_CALLS = 64  # real calls compared bit for bit with the reference before trusting the kernel

    @staticmethod
    def _same_bits(t, a, b):
        """Bit-identical, except that any NaN equals any NaN (payloads are not significant)."""
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        nan_a, nan_b = t.isnan(a), t.isnan(b)
        return bool(t.equal(nan_a, nan_b)) and bool(t.equal(a.masked_fill(nan_a, 0).view(t.int32),
                                                            b.masked_fill(nan_b, 0).view(t.int32)))

    def _act_off(self, reason):
        if self.act_enabled:
            _note(f'fused activation off: {reason}. Using the reference path.')
        self.act_enabled = False
        self.act_reason = reason

    def _act_launch(self, x):
        t = self.t
        xs = x.contiguous()
        out = t.empty(xs.shape, device=xs.device, dtype=t.float32)
        n = xs.numel()
        self.compiled.launch('act_quant', n // 32, [
            ctypes.c_void_p(xs.data_ptr()), int(xs.dtype == t.bfloat16), ctypes.c_longlong(n),
            ctypes.c_void_p(out.data_ptr())])
        return out

    def _act_selftest(self, reference):
        """Synthetic edge cases: every magnitude range, zero groups, FP8 ties and subnormals,
        float32 subnormals, values that clamp at 448, NaN groups, both input dtypes."""
        t = self.t
        g = t.Generator(device='cpu').manual_seed(1234)
        parts = [t.randn(64, 32, generator=g) * 10.0 ** e for e in range(-12, 7)]
        parts.append(t.zeros(4, 32))
        parts.append(t.full((2, 32), 1e-5))                       # all below the 1e-4 floor
        parts.append(t.full((2, 32), 1.4e-45))                    # float32 subnormals
        ties = t.arange(1, 449, dtype=t.float32)[:448] / 4        # many exact FP8 ties after scaling
        parts.append(t.cat((ties, -ties)).reshape(-1, 32))
        spread = t.randn(32, 32, generator=g)
        spread[:, 0] = 1e4                                        # one large value: the rest go subnormal
        parts.append(spread)
        nan_group = t.randn(2, 32, generator=g)
        nan_group[0, 5] = float('nan')
        parts.append(nan_group)
        x = t.cat(parts).reshape(-1, 32).cuda()                  # one 32-value group per row
        wide = x[:x.shape[0] - x.shape[0] % 8].reshape(-1, 256)   # several groups per row
        for case in (x, x.to(t.bfloat16), wide, wide.to(t.bfloat16), wide[:3].contiguous()):
            if not self._same_bits(t, self._act_launch(case), reference(case)):
                raise RuntimeError(f'self-test mismatch ({case.dtype}, {tuple(case.shape)})')

    def activation(self, x, reference):
        """expert_correctness.activation(x) in one kernel launch, bit-identical. reference is that
        function. Falls back to it for anything other than a 2-D CUDA float32/bfloat16 tensor whose
        width is a multiple of 32, and switches itself off (only itself) on any mismatch."""
        t = self.t
        if (not self.act_enabled or not x.is_cuda or x.dim() != 2 or x.shape[1] % 32 or not x.numel()
                or x.dtype not in (t.float32, t.bfloat16) or not self._ready()):
            return reference(x)
        try:
            if not self.act_tested:
                self._act_selftest(reference)
                self.act_tested = True
            out = self._act_launch(x)
            if self.act_compared < self.ACT_COMPARE_CALLS:
                expected = reference(x)
                self.act_compared += 1
                if not self._same_bits(t, out, expected):
                    raise RuntimeError(f'mismatch on a real input {tuple(x.shape)} {x.dtype}')
        except Exception as error:  # noqa: BLE001 - any failure means: use the reference path
            self._act_off(str(error))
            return reference(x)
        self.act_calls += 1
        return out

    def _ready(self):
        if not self.enabled:
            return False
        if self.compiled is None:
            try:
                self.compiled = _Compiled(self.t)
            except Exception as error:  # noqa: BLE001 - any failure means: use the reference path
                self.disable(f'unavailable ({error})')
                return False
        return True

    def disable(self, reason):
        if self.enabled:
            _note(f'fused kernels off: {reason}. Using the reference path.')
        self.enabled = False
        self.reason = reason

    def _run(self, key, launch, reference):
        """launch() -> fp32 output or raises. reference() -> output in the final dtype."""
        if key not in self.checked:
            expected = reference()
            try:
                actual = launch()
                exp = expected.float()
                bad = ~self.t.isfinite(actual) & self.t.isfinite(exp)
                scale = float(exp.abs().nan_to_num(0.0).amax().clamp_min(1e-30))
                error = float((actual.float() - exp).abs().nan_to_num(0.0).amax()) / scale
                # A few units in the last place of the output type, relative to the largest output.
                # A wrong kernel is off by orders of magnitude, not by rounding.
                tolerance = {self.t.float16: 1e-2, self.t.bfloat16: 3e-2}.get(expected.dtype, 1e-3)
                if bool(bad.any()) or error > tolerance:
                    raise RuntimeError(f'{key[0]} {key[1:]} differs from reference (relative {error:.2e})')
            except Exception as error:  # noqa: BLE001
                self.disable(str(error))
                return expected
            self.checked.add(key)
            return expected
        try:
            out = launch()
        except Exception as error:  # noqa: BLE001
            self.disable(str(error))
            return reference()
        self.calls += 1
        return out

    def _rows(self, x):
        if x.shape[-1] == 0 or x.numel() == 0:
            return None
        flat = x.reshape(-1, x.shape[-1])
        return flat if flat.shape[0] <= MAX_TOKENS else None

    def gguf(self, x, raw, kind, reference):
        """y = x @ decode(raw).T for a GGUF row-quantized matrix raw [rows, row_bytes].
        reference() must return exactly what the old path returns. Returns a tensor shaped like it."""
        t = self.t
        flat = self._rows(x)
        info = GGUF_KINDS.get(kind)
        if (flat is None or info is None or raw.ndim != 2 or raw.dtype != t.uint8 or not raw.is_cuda
                or not self._ready()):
            return reference()
        code, block_bytes, block_weights = info
        rows, row_bytes = raw.shape
        width = flat.shape[1]
        if width % block_weights or row_bytes != width // block_weights * block_bytes or (code < 3 and width % 256):
            return reference()
        mode = {t.float16: 1, t.bfloat16: 2}.get(x.dtype, 0)
        units = (width + 255) // 256

        def launch():
            xs = flat.float().contiguous()
            weight = raw if raw.is_contiguous() else raw.contiguous()
            out = t.empty(xs.shape[0], rows, device=raw.device, dtype=t.float32)
            self.compiled.launch('gguf_matvec', rows, [
                ctypes.c_void_p(weight.data_ptr()), code, rows, ctypes.c_longlong(row_bytes), units,
                ctypes.c_void_p(xs.data_ptr()), xs.shape[0], width, ctypes.c_void_p(out.data_ptr()), mode])
            return out.to(x.dtype).reshape(*x.shape[:-1], rows)

        return self._run(('gguf', kind, rows, width, str(x.dtype)), launch, reference)

    def block(self, x, weight, scale, fp4, reference, out_dtype):
        """y = x @ decode(weight, scale).T, x already activation-quantized (float32).
        fp4: weight [rows, cols/2] packed E2M1, scale [rows, cols/32]. Otherwise weight is
        E4M3 [rows, cols] and scale [rows/32, cols/32]. Scale 255 must be rejected by the caller."""
        t = self.t
        flat = self._rows(x)
        if flat is None or not weight.is_cuda or not self._ready():
            return reference()
        cols = flat.shape[1]
        rows = weight.shape[0]
        shape = (rows, cols // 32) if fp4 else (rows // 32, cols // 32)
        width = cols // 2 if fp4 else cols
        if (cols % 32 or (not fp4 and rows % 32) or tuple(weight.shape) != (rows, width)
                or tuple(scale.shape) != shape):
            return reference()

        def launch():
            xs = flat.float().contiguous()
            w = weight.contiguous().view(t.uint8)
            s = scale.contiguous().view(t.uint8)
            out = t.empty(xs.shape[0], rows, device=weight.device, dtype=t.float32)
            self.compiled.launch('block_matvec', rows, [
                ctypes.c_void_p(w.data_ptr()), ctypes.c_void_p(s.data_ptr()), int(fp4), rows, cols,
                ctypes.c_void_p(xs.data_ptr()), xs.shape[0], ctypes.c_void_p(out.data_ptr())])
            return out.to(out_dtype).reshape(*x.shape[:-1], rows)

        return self._run(('fp4' if fp4 else 'fp8', rows, cols), launch, reference)


_SHARED = {}


def get(torch):
    """One Fused instance per process."""
    fused = _SHARED.get('fused')
    if fused is None:
        fused = _SHARED['fused'] = Fused(torch)
    return fused
