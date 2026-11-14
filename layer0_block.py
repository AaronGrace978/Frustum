"""Complete layer-0 block baseline with causal sliding-window KV state.
No compressed-KV, Engram, full-model generation or original TileLang parity claim.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from frusium_loader import Catalog,FrusiumLoader
from reproducibility import tensor_sha256,provenance,cache_sha256
from expert_correctness import activation
from streamed_moe import StreamedMoE
from block_primitives import rms_norm,hc_mixes,hc_pre,hc_post

GIB=1024**3

def quant_dequant(x):
    shape=x.shape
    return activation(x.reshape(-1,shape[-1])).reshape(shape).to(torch.bfloat16)

def rope(x,positions,base=10000.,inverse=False):
    # Adjacent pairs, as in the supplied apply_rotary_emb; pure SWA disables YaRN.
    dim=x.shape[-1]
    freq=base**(-torch.arange(0,dim,2,device=x.device,dtype=torch.float32)/dim)
    angle=positions.float()[:,None]*freq[None,:]
    if inverse:angle=-angle
    shape=[1,len(positions)]+[1]*(x.ndim-3)+[dim//2]
    c=angle.cos().reshape(shape);s=angle.sin().reshape(shape)
    pairs=x.float().reshape(*x.shape[:-1],dim//2,2)
    a,b=pairs[...,0],pairs[...,1]
    return torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-2).to(x.dtype)

class RingKV:
    def __init__(self,window):self.window=window;self.reset()
    def reset(self):self.data=None;self.next_position=0
    def append(self,kv,start):
        b,n,d=kv.shape
        if start!=self.next_position:raise ValueError('Non-contiguous KV position; reset before a new sequence')
        if self.data is None:self.data=torch.zeros(b,self.window,d,device=kv.device,dtype=kv.dtype)
        if self.data.shape[0]!=b:raise ValueError('Batch size changed without reset')
        for i in range(n):self.data[:,(start+i)%self.window]=kv[:,i]
        self.next_position+=n
    def latest(self):
        lo=max(0,self.next_position-self.window)
        ids=torch.arange(lo,self.next_position,device=self.data.device)%self.window
        return self.data[:,ids]


def window_attention(q,kv,sink,window,start):
    # BF16 exponent cast mirrors the reference kernel's probability operand.
    # Full selected window is reduced at once; not TileLang's online block ordering.
    b,n,h,d=q.shape
    out=torch.empty_like(q)
    for token in range(n):
        if start==0:values=kv[:,max(0,token-window+1):token+1]
        else:values=kv
        for batch in range(b):
            scores=(q[batch,token].float()@values[batch].float().T)*(d**-.5)
            maximum=scores.max(-1,keepdim=True).values
            prob=torch.exp(scores-maximum)
            den=prob.sum(-1,keepdim=True)+torch.exp(sink[:,None]-maximum)
            out[batch,token]=(prob.to(torch.bfloat16).float()@values[batch].float()/den).to(q.dtype)
    return out

class LayerZeroBlock:
    def __init__(self,catalog):
        self.c=catalog;self.cfg=catalog.config['text_config']
        if self.cfg['compress_ratios'][0]!=0 or 0 in self.cfg['engram_layer_ids']:
            raise ValueError('This baseline only supports layer 0 without compression or Engram')
        self.dense=FrusiumLoader(catalog,'cuda',256*1024**2)
        self.moe=StreamedMoE(catalog,256)
        self.kv=RingKV(self.cfg['sliding_window'])
    def reset(self):self.kv.reset()
    def tensor(self,name):
        item=self.c.tensors[name]
        with self.dense.acquire(name) as raw:
            return raw.view({'BF16':torch.bfloat16,'F32':torch.float32}[item.dtype]).reshape(item.shape).clone()
    def matrix(self,name):
        wi=self.c.tensors[name+'.weight']
        if wi.dtype=='BF16':return self.tensor(name+'.weight')
        if wi.dtype!='F8_E4M3':raise ValueError('Unsupported dense format')
        si=self.c.tensors[name+'.scale']
        if si.dtype!='F8_E8M0':raise ValueError('Unsupported scale format')
        with self.dense.acquire(name+'.weight') as w,self.dense.acquire(name+'.scale') as s:
            raw=w.view(torch.float8_e4m3fn).reshape(wi.shape).float()
            scales=s.reshape(si.shape)
            if bool((scales==255).any()):raise ValueError('NaN scale')
            return raw*torch.exp2(scales.float()-127).repeat_interleave(32,0).repeat_interleave(32,1)[:wi.shape[0],:wi.shape[1]]
    def linear(self,x,name):
        w=self.matrix(name)
        if self.c.tensors[name+'.weight'].dtype=='F8_E4M3':
            z=activation(x.reshape(-1,x.shape[-1])).reshape(x.shape)
        else:z=x.float()
        return F.linear(z,w.float()).to(torch.bfloat16)
    def norm(self,x,name):return rms_norm(x,self.tensor(name+'.weight'),self.cfg['rms_norm_eps'])
    def mixes(self,x,part):
        return hc_mixes(x,self.tensor('layers.0.hc_'+part+'_fn'),self.tensor('layers.0.hc_'+part+'_scale'),
            self.tensor('layers.0.hc_'+part+'_base'),hc=self.cfg['hc_mult'],iters=self.cfg['hc_sinkhorn_iters'],eps=self.cfg['hc_eps'],norm_eps=self.cfg['rms_norm_eps'])
    @torch.inference_mode()
    def attention(self,x,start):
        cfg=self.cfg;b,n,_=x.shape
        if start!=0 and n!=1:raise ValueError('Decode requires one token per step')
        if start!=self.kv.next_position:raise ValueError('KV position mismatch')
        p='layers.0.attn.';rd=cfg['qk_rope_head_dim'];heads=cfg['num_attention_heads'];dim=cfg['head_dim']
        positions=torch.arange(start,start+n,device=x.device)
        qr=self.norm(self.linear(x,p+'wq_a'),p+'q_norm')
        q=self.linear(qr,p+'wq_b').reshape(b,n,heads,dim)
        q[...,-rd:]=rope(q[...,-rd:],positions,cfg['rope_theta'])
        kv=self.norm(self.linear(x,p+'wkv'),p+'kv_norm')
        kv[...,-rd:]=rope(kv[...,-rd:],positions,cfg['rope_theta'])
        kv=quant_dequant(kv)
        self.kv.append(kv,start)
        values=kv if start==0 else self.kv.latest()
        o=window_attention(q,values,self.tensor(p+'attn_sink'),cfg['sliding_window'],start)
        o[...,-rd:]=rope(o[...,-rd:],positions,cfg['rope_theta'],True)
        groups=cfg['o_groups'];rank=cfg['o_lora_rank']
        o=quant_dequant(o.reshape(b,n,groups,-1))
        # convert.py explicitly dequantizes wo_a to BF16 before grouped projection.
        wa=self.matrix(p+'wo_a').to(torch.bfloat16).reshape(groups,rank,-1)
        latent=torch.einsum('bsgd,grd->bsgr',o,wa)
        return self.linear(latent.flatten(2),p+'wo_b')
    @torch.inference_mode()
    def forward(self,h,start=0,pre_mix=None,validate_moe=False):
        if pre_mix is None:
            pre_mix=torch.zeros(*h.shape[:-1],dtype=torch.float32,device=h.device);pre_mix[...,0]=1
        self.last_trace={'block_input':tensor_sha256(h),'pre_mix':tensor_sha256(pre_mix)}
        attn_pre,attn_post,attn_comb=self.mixes(h,'attn')
        a=self.attention(self.norm(hc_pre(h,pre_mix),'layers.0.attn_norm'),start)
        self.last_trace['attention_output']=tensor_sha256(a)
        h=hc_post(a,h,attn_post,attn_comb)
        self.last_trace['attention_residual']=tensor_sha256(h)
        ffn_pre,ffn_post,ffn_comb=self.mixes(h,'ffn')
        x=self.norm(hc_pre(h,attn_pre),'layers.0.ffn_norm')
        self.last_trace['moe_input']=tensor_sha256(x)
        f,record=self.moe.forward(x.reshape(-1,x.shape[-1]),0,validate=validate_moe)
        h=hc_post(f.reshape(x.shape),h,ffn_post,ffn_comb)
        if not torch.isfinite(h).all():raise AssertionError('Nonfinite block output')
        self.last_trace['block_output']=tensor_sha256(h)
        return h,ffn_pre,record
    def close(self):self.moe.close();self.dense.close()


def primitive_tests():
    torch.manual_seed(7)
    q=torch.randn(1,4,3,8,device='cuda').to(torch.bfloat16)
    kv=torch.randn(1,4,8,device='cuda').to(torch.bfloat16)
    sink=torch.tensor([-.2,.3,1.],device='cuda')
    actual=window_attention(q,kv,sink,3,0).cpu().float().numpy()
    expected=np.empty_like(actual)
    qn=q.cpu().float().numpy().astype(np.float64);kn=kv.cpu().float().numpy().astype(np.float64)
    for i in range(4):
        v=kn[0,max(0,i-2):i+1]
        z=qn[0,i]@v.T/np.sqrt(8);mx=z.max(-1,keepdims=True);p=np.exp(z-mx)
        # Mirror BF16 probability storage; remaining oracle arithmetic is NumPy FP64.
        pb=torch.from_numpy(p).to(torch.bfloat16).float().numpy()
        expected[0,i]=pb@v/(p.sum(-1,keepdims=True)+np.exp(sink.cpu().numpy()[:,None]-mx))
    np.testing.assert_allclose(actual,expected,rtol=.015,atol=.01)
    changed=kv.clone();changed[:,3]*=100
    torch.testing.assert_close(window_attention(q,changed,sink,3,0)[:,:3],torch.from_numpy(actual[:,:3]).cuda().to(torch.bfloat16),rtol=0,atol=0)
    ring=RingKV(3)
    for i in range(8):
        row=torch.full((1,1,8),float(i),device='cuda',dtype=torch.bfloat16);ring.append(row,i)
        assert ring.latest()[0,:,0].tolist()==list(range(max(0,i-2),i+1))
    try:ring.append(row,99)
    except ValueError:pass
    else:raise AssertionError('KV position mismatch accepted')
    t=torch.randn(1,4,3,8,device='cuda');pos=torch.arange(4,device='cuda')
    torch.testing.assert_close(rope(rope(t,pos),pos,inverse=True),t,rtol=2e-5,atol=2e-5)
    return ['attention NumPy oracle','causal future-token masking','KV ring wraparound','KV position rejection','RoPE inverse']

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=r'G:\Models\DeepSeek-V4.1-Flash')
    p.add_argument('--output',default='layer0_block_results.json')
    args=p.parse_args()
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    free,total=torch.cuda.mem_get_info()
    if total<14*GIB or free<3*GIB:raise SystemExit('Insufficient dedicated VRAM/headroom')
    torch.cuda.set_per_process_memory_fraction(12*GIB/total)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    tests=primitive_tests();print(json.dumps(dict(primitive_tests='PASS',tests=tests)),flush=True)
    c=Catalog(args.model);block=LayerZeroBlock(c)
    torch.manual_seed(2026)
    # Use real token embeddings instead of fabricated full-block inputs.
    from sparse_rows import SparseRows
    lookup=SparseRows(c)
    embeds=lookup.get('embed.weight',[100,200,300]).unsqueeze(0)
    h=embeds.unsqueeze(2).repeat(1,1,c.config['text_config']['hc_mult'],1)
    torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    try:
        full,pre,moe_record=block.forward(h,0,validate_moe=True)
        full_trace=dict(block.last_trace)
        print(json.dumps(dict(full_prefill='PASS',moe=moe_record)),flush=True)
        block.reset()
        prefix,_,_=block.forward(h[:,:2],0,validate_moe=False)
        last,last_pre,_=block.forward(h[:,2:],2,validate_moe=False)
        inc=torch.cat((prefix,last),1)
        a=full.float();b=inc.float()
        rel=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-12))
        if rel>.02:raise AssertionError(f'Prefill/decode relative L2 mismatch {rel}')
        torch.testing.assert_close(pre[:,2:],last_pre,rtol=.02,atol=.01)
        report=dict(status='PASS',provenance=provenance(args.model),full_prefill_trace=full_trace,
            resident_dense_sha256=cache_sha256(block.dense),resident_router_sha256=cache_sha256(block.moe.router),
            output_sha256=tensor_sha256(full),scope='Complete layer-0 block only; no full-model generation or later-layer compressed attention',
            actual_token_embedding_ids=[100,200,300],output_shape=list(full.shape),primitive_tests=tests,
            full_prefill_vs_prefix_plus_decode_relative_L2=rel,max_abs_block_difference=float((a-b).abs().max()),
            full_prefill_MoE_reference=moe_record,
            logical_weight_read_MB=(block.dense.reader.bytes_read+block.moe.cache.reader.bytes_read+block.moe.router.reader.bytes_read+lookup.reader.bytes_read)/1e6,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/GIB,peak_reserved_GiB=torch.cuda.max_memory_reserved()/GIB,
            seconds_including_CPU_validation=time.perf_counter()-begin,
            caveat='Attention oracle uses tiny synthetic dimensions; full-block check compares two schedules of the same implementation, not original TileLang parity. Full attention-window wrap validated at primitive level only.')
        Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:block.close()

if __name__=='__main__':main()
