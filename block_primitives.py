"""RMSNorm and mHC residual operations for future full-block integration.
Pure PyTorch, preserving the equations in the supplied inference/model.py.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from frusium_loader import Catalog,FrusiumLoader


def rms_norm(x,weight,eps):
    xf=x.float()
    return (weight*xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+eps)).to(x.dtype)


def hc_mixes(x,fn,scale,base,hc=4,iters=20,eps=1e-6,norm_eps=1e-20):
    flat=x.flatten(2).float()
    mix=F.linear(flat,fn)*torch.rsqrt(flat.square().mean(-1,keepdim=True)+norm_eps)
    pre=torch.sigmoid(mix[...,:hc]*scale[0]+base[:hc])+eps
    post=2*torch.sigmoid(mix[...,hc:2*hc]*scale[1]+base[hc:2*hc])
    logits=(mix[...,2*hc:]*scale[2]+base[2*hc:]).reshape(*mix.shape[:-1],hc,hc)
    comb=logits.softmax(-1)+eps
    comb=comb/(comb.sum(-2,keepdim=True)+eps)
    for _ in range(iters-1):
        comb=comb/(comb.sum(-1,keepdim=True)+eps)
        comb=comb/(comb.sum(-2,keepdim=True)+eps)
    return pre,post,comb


def hc_pre(x,pre):
    return (pre[...,None]*x.float()).sum(-2).to(x.dtype)


def hc_post(x,residual,post,comb):
    # comb[source, destination]; sum source dimension, as local Block.hc_post.
    return (post[...,None]*x.unsqueeze(-2)+(comb[...,None]*residual.unsqueeze(-2)).sum(-3)).to(x.dtype)

@torch.inference_mode()
def main():
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    free,total=torch.cuda.mem_get_info()
    if free<2*1024**3:raise SystemExit('Insufficient VRAM headroom')
    torch.cuda.set_per_process_memory_fraction(min(1,12*1024**3/total))
    torch.backends.cuda.matmul.allow_tf32=False
    c=Catalog(r'G:\Models\DeepSeek-V4.1-Flash');cfg=c.config['text_config']
    loader=FrusiumLoader(c,'cuda',32*1024**2)
    def get(name):
        item=c.tensors[name]
        with loader.acquire(name) as raw:
            return raw.view({'F32':torch.float32,'BF16':torch.bfloat16}[item.dtype]).reshape(item.shape).clone()
    try:
        fn=get('layers.0.hc_attn_fn');scale=get('layers.0.hc_attn_scale');base=get('layers.0.hc_attn_base')
        norm=get('layers.0.attn_norm.weight')
        torch.manual_seed(42);x=torch.randn(1,3,cfg['hc_mult'],cfg['hidden_size'],device='cuda').to(torch.bfloat16)
        pre,post,comb=hc_mixes(x,fn,scale,base,hc=cfg['hc_mult'],iters=cfg['hc_sinkhorn_iters'],eps=cfg['hc_eps'],norm_eps=cfg['rms_norm_eps'])
        # Independent NumPy float64 coefficient calculation.
        arr=x.float().cpu().numpy().astype(np.float64); flat=arr.reshape(1,3,-1)
        mix=(flat@fn.cpu().double().numpy().T)/np.sqrt(np.mean(flat**2,-1,keepdims=True)+cfg['rms_norm_eps'])
        sc=scale.cpu().double().numpy();bs=base.cpu().double().numpy();h=cfg['hc_mult'];eps=cfg['hc_eps']
        sigmoid=lambda z:np.exp(-np.logaddexp(0,-z))
        pr=sigmoid(mix[...,:h]*sc[0]+bs[:h])+eps
        po=2*sigmoid(mix[...,h:2*h]*sc[1]+bs[h:2*h])
        co=(mix[...,2*h:]*sc[2]+bs[2*h:]).reshape(1,3,h,h)
        co=np.exp(co-co.max(-1,keepdims=True));co=co/co.sum(-1,keepdims=True)+eps
        co/=co.sum(-2,keepdims=True)+eps
        for _ in range(cfg['hc_sinkhorn_iters']-1):
            co/=co.sum(-1,keepdims=True)+eps;co/=co.sum(-2,keepdims=True)+eps
        for actual,expected in ((pre,pr),(post,po),(comb,co)):
            np.testing.assert_allclose(actual.cpu().numpy(),expected,rtol=2e-4,atol=2e-5)
        collapsed=hc_pre(x,pre)
        np.testing.assert_allclose(collapsed.float().cpu().numpy(),torch.tensor((arr*pr[...,None]).sum(-2)).to(torch.bfloat16).float().numpy(),rtol=.01,atol=.01)
        normal=rms_norm(collapsed,norm,cfg['rms_norm_eps'])
        v=collapsed.float().cpu().double().numpy(); nr=norm.float().cpu().double().numpy()*v/np.sqrt(np.mean(v*v,-1,keepdims=True)+cfg['rms_norm_eps'])
        torch.testing.assert_close(normal.cpu(),torch.from_numpy(nr).to(torch.bfloat16),rtol=.01,atol=.01)
        expanded=hc_post(normal,x,post,comb)
        # Use actual FP32 coefficients to isolate expansion orientation from quantization.
        ref=post.double().cpu().numpy()[...,None]*normal.float().cpu().double().numpy()[:,:,None,:]+np.einsum('bsij,bsid->bsjd',comb.double().cpu().numpy(),arr)
        torch.testing.assert_close(expanded.cpu(),torch.from_numpy(ref).to(torch.bfloat16),rtol=.01,atol=.01)
        report=dict(status='PASS',components=['RMSNorm','mHC coefficients/Sinkhorn','residual collapse','residual expansion'],
            real_parameter_layer=0,input_shape=list(x.shape),logical_weight_read_MB=loader.reader.bytes_read/1e6,
            note='Primitives validated; full attention-plus-FFN transformer block not yet integrated.')
        Path('block_primitives_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:loader.close()

if __name__=='__main__':main()
