"""One real streamed MXFP4 expert. Correctness baseline, not fused FP4 inference.
GPU: unpack/dequantize one matrix at a time, FP32 GEMM, BF16 projection outputs.
Reference: independent NumPy FP4 decode and float64 CPU GEMM.
Activation quantization follows local inference/kernel.py: E4M3, per-32 power-of-2 scales.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from frustum_loader import Catalog, FrustumLoader

GIB=1024**3

FP4_VALUES=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]
_PAIR_LUT={}

def gpu_decode_reference(packed, scale):
    """Original v0 decoder: two masked table lookups and strided writes. Kept as the oracle."""
    if bool((scale==255).any()): raise ValueError('Invalid E8M0 NaN scale')
    lut=torch.tensor(FP4_VALUES,device=packed.device)
    out=torch.empty((packed.shape[0],packed.shape[1]*2),device=packed.device,dtype=torch.float32)
    out[:,0::2]=lut[(packed & 15).long()]
    out[:,1::2]=lut[(packed >> 4).long()]
    return (out.reshape(out.shape[0],-1,32)*torch.exp2(scale.float()-127).unsqueeze(-1)).flatten(1)

def _pair_lut(device):
    # One row per packed byte: (low nibble value, high nibble value). Built once per device.
    lut=_PAIR_LUT.get(device)
    if lut is None:
        codes=torch.arange(256)
        values=torch.tensor(FP4_VALUES)
        lut=_PAIR_LUT[device]=torch.stack((values[codes & 15],values[codes >> 4]),1).to(device)
    return lut

def gpu_decode(packed, scale):
    """Same values as gpu_decode_reference: one byte-pair lookup, then the block scale in place.

    The table entries and the power-of-two scale products are exact, so the result is
    bit-identical; decoder_test checks that on the GPU in use.
    """
    if bool((scale==255).any()): raise ValueError('Invalid E8M0 NaN scale')
    rows,cols=packed.shape
    out=_pair_lut(packed.device).index_select(0,packed.reshape(-1).int()).view(rows,cols*2)
    out.view(rows,-1,32).mul_(torch.exp2(scale.float()-127).unsqueeze(-1))
    return out

def cpu_decode(packed, scale):
    # Decode sign/exponent/mantissa arithmetically, independently of GPU LUT.
    bits=np.empty((packed.shape[0],packed.shape[1]*2),dtype=np.uint8)
    bits[:,0::2]=packed%16
    bits[:,1::2]=packed//16
    exponent=((bits>>1)&3).astype(np.int32)
    mantissa=(bits&1).astype(np.float64)
    magnitude=np.where(exponent==0,mantissa*.5,np.ldexp(1+mantissa*.5,exponent-1))
    signed=np.where(bits&8,-magnitude,magnitude)
    if np.any(scale==255): raise ValueError('Invalid E8M0 NaN scale')
    return (signed.reshape(signed.shape[0],-1,32)*np.exp2(scale.astype(np.float64)-127)[...,None]).reshape(signed.shape)

def activation(x):
    groups=x.float().reshape(x.shape[0],-1,32)
    scales=torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4)/448)))
    quant=(groups/scales.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn).float()
    return (quant*scales.unsqueeze(-1)).reshape(x.shape)

def decoder_test():
    # Exhaustive 256 packed bytes, all 16 codes and multiple positive scales.
    p=np.arange(256,dtype=np.uint8).reshape(16,16)
    s=np.arange(119,135,dtype=np.uint8).reshape(16,1)
    actual=gpu_decode(torch.from_numpy(p).cuda(),torch.from_numpy(s).cuda()).cpu().numpy()
    np.testing.assert_array_equal(actual,cpu_decode(p,s))
    assert actual[0,2]==.5*2**(119-127)
    # Fast decoder against the original on a multi-block matrix, signed zeros included.
    generator=np.random.default_rng(0)
    p2=torch.from_numpy(generator.integers(0,256,(64,96),dtype=np.uint8)).cuda()
    s2=torch.from_numpy(generator.integers(110,140,(64,6),dtype=np.uint8)).cuda()
    fast=gpu_decode(p2,s2); slow=gpu_decode_reference(p2,s2)
    assert fast.shape==slow.shape and torch.equal(fast.view(torch.int32),slow.view(torch.int32)),'Fast FP4 decode differs from reference'
    return True

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=r'G:\Models\DeepSeek-V4.1-Flash')
    parser.add_argument('--layer',type=int,default=0)
    parser.add_argument('--expert',type=int,default=0)
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--output',default='expert_correctness_results.json')
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    free,total=torch.cuda.mem_get_info()
    if total<14*GIB or free<3*GIB: raise SystemExit('Insufficient dedicated VRAM/headroom')
    torch.cuda.set_per_process_memory_fraction(12*GIB/total)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4)
    decoder_test()
    catalog=Catalog(args.model)
    cfg=catalog.config['text_config']
    if not 0<=args.layer<cfg['num_hidden_layers'] or not 0<=args.expert<cfg['n_routed_experts']:
        raise ValueError('Invalid layer or expert')
    loader=FrustumLoader(catalog,'cuda',64*1024**2)
    torch.manual_seed(args.seed)
    x_cpu=torch.randn(3,cfg['hidden_size'],dtype=torch.float32).to(torch.bfloat16)
    # Zero, typical, large activations exercise quantization and clamps.
    x_cpu[0].zero_(); x_cpu[2]*=8
    x=x_cpu.cuda()
    projections=[]
    started=time.perf_counter()
    def linear_pair(x_gpu,x_ref,w):
        name=f'layers.{args.layer}.ffn.experts.{args.expert}.{w}'
        wi,si=catalog.tensors[name+'.weight'],catalog.tensors[name+'.scale']
        if wi.dtype!='I8' or si.dtype!='F8_E8M0': raise ValueError('Expected packed FP4 with E8M0 scales')
        if tuple(si.shape)!=(wi.shape[0],wi.shape[1]*2//32): raise ValueError('Unexpected scale shape')
        with loader.acquire(name+'.weight') as raw, loader.acquire(name+'.scale') as scales:
            p=raw.reshape(wi.shape); s=scales.reshape(si.shape)
            a=activation(x_gpu)
            ar=activation(x_ref).double().numpy()
            weights=gpu_decode(p,s)
            out=F.linear(a,weights)
            reference_weights=cpu_decode(p.cpu().numpy(),s.cpu().numpy())
            # Independent FP64 reference for exactly the GPU's quantized input.
            same_input=a.cpu().double().numpy()@reference_weights.T
            got=out.cpu().double().numpy()
            np.testing.assert_allclose(got,same_input,rtol=2e-4,atol=2e-4)
            reference=ar@reference_weights.T
            projections.append(dict(matrix=w,max_abs_gemm_error=float(np.abs(got-same_input).max())))
            return out.to(torch.bfloat16),torch.from_numpy(reference).to(torch.bfloat16)
    try:
        gate,gate_ref=linear_pair(x,x_cpu,'w1')
        up,up_ref=linear_pair(x,x_cpu,'w3')
        limit=cfg['swiglu_limit']
        def swiglu(g,u):
            g=g.float();u=u.float()
            if limit>0:g=g.clamp(max=limit);u=u.clamp(-limit,limit)
            return (F.silu(g)*u).to(torch.bfloat16)
        y,y_ref=linear_pair(swiglu(gate,up),swiglu(gate_ref,up_ref),'w2')
        torch.cuda.synchronize()
        actual=y.cpu().float(); expected=y_ref.float()
        if not torch.isfinite(actual).all(): raise AssertionError('Non-finite expert output')
        rel=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-12))
        cosine=float(F.cosine_similarity(actual[1:].flatten(),expected[1:].flatten(),dim=0))
        if rel>0.01 or cosine<0.999: raise AssertionError(f'Expert reference mismatch: relative L2={rel}, cosine={cosine}')
        result=dict(status='PASS',layer=args.layer,expert=args.expert,seed=args.seed,
            input_cases=['zero','normal','8x normal'],output_shape=list(y.shape),
            independent_fp4_decode_test='all 256 packed bytes passed',
            projections=projections,relative_L2_error=rel,cosine_similarity=cosine,
            max_abs_output_error=float((actual-expected).abs().max()),
            logical_weight_read_MB=loader.reader.bytes_read/1e6,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/GIB,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/GIB,
            total_seconds=time.perf_counter()-started,
            reference='Independent NumPy decode + FP64 GEMM; shared PyTorch FP8 activation quantization; NOT original TileLang kernel parity',
            implementation='One-matrix-at-a-time FP32 dequantization; not native FP4 Tensor Core execution')
        Path(args.output).write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally: loader.close()

if __name__=='__main__': main()
