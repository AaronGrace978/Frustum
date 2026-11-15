"""GPU check and micro-benchmark for the v0.6 fused kernels. No model weights needed.

For each format: random packed weights at a real model's matrix size, one decode token.
Compares the fused kernel with the reference decode-then-multiply path and times both.

    python fused_check.py            (or .\\run_fused_check.ps1)
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import fused_kernels
from expert_correctness import activation, gpu_decode
from phi4_chat import unpack
from glm_chat import decode_weight

GGUF = {'Q4_K': (144, 256), 'Q5_K': (176, 256), 'Q6_K': (210, 256), 'Q8_0': (34, 32)}


def gguf_matrix(kind, rows, cols, rng):
    nbytes, per = GGUF[kind]
    raw = rng.integers(0, 256, (rows, cols // per * nbytes), dtype=np.uint8)
    blocks = raw.reshape(-1, nbytes)
    halves = rng.uniform(1e-4, 2e-2, (blocks.shape[0], 2)).astype(np.float16).view(np.uint8)
    if kind == 'Q6_K':
        blocks[:, 208:210] = halves[:, :2]
    elif kind == 'Q8_0':
        blocks[:, 0:2] = halves[:, :2]
    else:
        blocks[:, 0:4] = halves
    return torch.from_numpy(raw).cuda()


def timed(fn, repeat=20):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) / repeat


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault('FRUSTUM_FUSED', '1')
    fused = fused_kernels.get(torch)
    rng = np.random.default_rng(0)
    rows_out = []
    cases = [('Q4_K 7680x5120', 'Q4_K', 7680, 5120), ('Q5_K 5120x17920', 'Q5_K', 5120, 17920),
             ('Q6_K 100352x5120 (head-sized)', 'Q6_K', 100352, 5120), ('Q8_0 2048x2048', 'Q8_0', 2048, 2048)]
    for label, kind, rows, cols in cases:
        raw = gguf_matrix(kind, rows, cols, rng)
        x = torch.randn(1, cols, device='cuda').half()
        decode = (lambda c: decode_weight(c, kind)) if kind == 'Q8_0' else (lambda c: unpack(c, kind))

        def reference():
            return torch.cat([F.linear(x, decode(chunk).to(x.dtype)) for chunk in raw.split(2048)], -1)
        expected, ref_s = timed(reference)
        fused.gguf(x, raw, kind, reference)  # first call checks against the reference
        if not fused.enabled:
            raise SystemExit(f'Fused path turned itself off: {fused.reason}')
        got, fused_s = timed(lambda: fused.gguf(x, raw, kind, reference))
        err = float((got.float() - expected.float()).abs().max() / expected.float().abs().max())
        rows_out.append(dict(case=label, reference_ms=ref_s * 1e3, fused_ms=fused_s * 1e3,
                             speedup=ref_s / fused_s, relative_error=err))

    for label, fp4, rows, cols in (('FP4 expert 2304x5120', True, 2304, 5120),
                                   ('FP8 dense 5120x5120', False, 5120, 5120)):
        x = torch.randn(1, cols, device='cuda').to(torch.bfloat16)
        xa = activation(x)
        if fp4:
            weight = torch.from_numpy(rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)).cuda()
            scale = torch.from_numpy(rng.integers(118, 128, (rows, cols // 32), dtype=np.uint8)).cuda()

            def reference():
                return F.linear(xa, gpu_decode(weight, scale)).to(torch.bfloat16)
        else:
            raw = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
            raw[(raw & 127) == 127] = 0
            weight = torch.from_numpy(raw).cuda()
            scale = torch.from_numpy(rng.integers(118, 128, (rows // 32, cols // 32), dtype=np.uint8)).cuda()

            def reference():
                m = weight.view(torch.float8_e4m3fn).float()
                m.view(rows // 32, 32, cols // 32, 32).mul_(torch.exp2(scale.float() - 127)[:, None, :, None])
                return F.linear(xa, m).to(torch.bfloat16)
        expected, ref_s = timed(reference)
        fused.block(xa, weight, scale, fp4, reference, torch.bfloat16)
        if not fused.enabled:
            raise SystemExit(f'Fused path turned itself off: {fused.reason}')
        got, fused_s = timed(lambda: fused.block(xa, weight, scale, fp4, reference, torch.bfloat16))
        err = float((got.float() - expected.float()).abs().max() / expected.float().abs().max())
        rows_out.append(dict(case=label, reference_ms=ref_s * 1e3, fused_ms=fused_s * 1e3,
                             speedup=ref_s / fused_s, relative_error=err))

    for r in rows_out:
        print(f"{r['case']:<32} reference {r['reference_ms']:8.2f} ms   fused {r['fused_ms']:7.2f} ms   "
              f"{r['speedup']:5.1f}x   rel err {r['relative_error']:.1e}")
    print(json.dumps(dict(status='PASS', device=torch.cuda.get_device_name(), results=rows_out), indent=2))


if __name__ == '__main__':
    main()
