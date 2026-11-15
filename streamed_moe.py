"""Streamed real MoE sublayer with actual text routing and shared FP8 expert.
Run directly, this is the MoE correctness test: no attention, residual block, or
text generation. The arithmetic is unchanged from v0; only where the packed
expert bytes come from (expert_cache.TieredExpertCache) has changed.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from frustum_loader import Catalog, FrustumLoader, default_model_root
from expert_cache import TieredExpertCache
import fused_kernels
import os

GIB = 1024**3
MIB = 1024**2
from expert_correctness import gpu_decode, cpu_decode, activation, decoder_test


def fp8_decode_cpu(raw, scales):
    # Independent arithmetic E4M3FN decoder; 0x7f/0xff are NaN.
    bits=raw.astype(np.uint8)
    e=((bits>>3)&15).astype(np.int32)
    m=(bits&7).astype(np.float64)
    mag=np.where(e==0,np.ldexp(m,-9),np.ldexp(1+m/8,e-7))
    v=np.where(bits&128,-mag,mag)
    if np.any((bits&127)==127) or np.any(scales==255):
        raise ValueError('Nonfinite FP8 weights/scales')
    scale=np.exp2(scales.astype(np.float64)-127).repeat(32,0).repeat(32,1)
    return v*scale[:v.shape[0],:v.shape[1]]


def route(x,weight,bias,cfg):
    scores=F.softplus(F.linear(x.float(),weight.float())).sqrt()
    ids=(scores+bias).topk(cfg['num_experts_per_tok'],dim=-1).indices
    selected=scores.gather(1,ids)
    if cfg['norm_topk_prob']:
        selected=selected/(selected.sum(-1,keepdim=True)+1e-20)
    return selected*cfg['routed_scaling_factor'],ids


def route_reference(x,weight,bias,cfg):
    logits=x.float().numpy().astype(np.float64)@weight.float().numpy().astype(np.float64).T
    scores=np.sqrt(np.logaddexp(0,logits))
    ids=np.argsort(-(scores+bias.numpy()),axis=-1)[:,:cfg['num_experts_per_tok']]
    weights=np.take_along_axis(scores,ids,axis=-1)
    if cfg['norm_topk_prob']: weights/=weights.sum(-1,keepdims=True)+1e-20
    return torch.from_numpy(weights*cfg['routed_scaling_factor']).float(),torch.from_numpy(ids.copy())


def expert_forward(x,weight,linear,limit):
    gate=linear(x,'w1').float()
    up=linear(x,'w3').float()
    if limit>0:
        gate=gate.clamp(max=limit); up=up.clamp(-limit,limit)
    hidden=F.silu(gate)*up
    if weight is not None: hidden=hidden*weight
    return linear(hidden.to(x.dtype),'w2')


class StreamedMoE:
    def __init__(self,catalog,cache_mib=256,*,host_cache_mib=0,router_cache_mib=32,io_threads=4,
                 gpu_policy='lfu',host_policy='lru',pin_shared=False,cap_bytes=12*GIB,headroom_bytes=2*GIB,
                 lookahead=None,predict_next=None,predict_mode=None,predict_threshold=0.6,predict_warmup=40):
        """cache_mib: GPU budget for packed experts. host_cache_mib: optional RAM tier behind it.
        router_cache_mib: give it the size of all router weights to stop re-reading them per token.
        pin_shared: never evict shared experts (the budget must have room for all of them).
        io_threads=0 restores strictly sequential reads.
        predict_next: during a decode step, run the next layer's router on this layer's input and
        start reading that many of its likely experts early (v0.6). The real routing still decides
        which experts run, so a wrong guess costs drive time, never accuracy. 0 disables.
        Default: FRUSTUM_PREDICT_NEXT, else 0 (off until measured: wrong guesses compete with real reads).
        predict_mode (v0.7; FRUSTUM_PREDICT_MODE): what to do with the guesses.
          shadow  score them, never read. Costs one small matrix product per layer and no drive time.
          gated   (default) read a source's guesses only after predict_warmup scored layers and while
                  the measured share of its not-yet-resident guesses that were then routed stays at
                  or above predict_threshold; otherwise behave like shadow.
          always  read every guess, as v0.6 did.
        Guesses come from two sources, scored separately: 'router' (next layer's router on this
        layer's input) and 'history' (what the next layer used on the previous step). Reads are
        issued after this layer's experts have run, so they never queue ahead of its own reads.
        """
        self.catalog=catalog
        self.cfg=catalog.config['text_config']
        if self.cfg['scoring_func']!='sqrtsoftplus' or self.cfg['n_shared_experts']!=1:
            raise ValueError('Unsupported MoE configuration')
        self.router=FrustumLoader(catalog,'cuda',router_cache_mib*MIB)
        pinned=(lambda key:key[1]=='shared') if pin_shared else None
        self.cache=TieredExpertCache(torch,catalog,torch.device('cuda:0'),cache_mib*MIB,host_cache_mib*MIB,
            io_threads=io_threads,gpu_policy=gpu_policy,host_policy=host_policy,pinned=pinned,
            cap_bytes=cap_bytes,headroom_bytes=headroom_bytes)
        # How many upcoming experts to read in the background while the GPU works.
        self.lookahead=max(1,2*io_threads) if lookahead is None else lookahead
        self._last_layer=None
        if predict_next is None:
            predict_next=int(os.environ.get('FRUSTUM_PREDICT_NEXT','0'))
        self.predict_next=0 if io_threads==0 else max(0,predict_next)
        self.predict_mode=(predict_mode or os.environ.get('FRUSTUM_PREDICT_MODE','gated')).strip().lower()
        if self.predict_mode not in ('shadow','gated','always'):
            raise ValueError('predict_mode is shadow, gated or always')
        self.predict_threshold,self.predict_warmup=float(predict_threshold),int(predict_warmup)
        self._predicted=None  # (layer, {source: ids}, {source: ids not resident when guessed})
        self._history={}      # layer -> expert ids routed on the previous decode step
        self.prediction=dict(guessed=0,routed=0,routed_and_guessed=0)
        self.sources={name:dict(layers=0,guessed=0,correct=0,cold_guessed=0,cold_correct=0,
                                cold_precision_ema=None,reads_issued=0) for name in ('router','history')}
        decoder_test()  # tiny: proves the fast FP4 decoder is exact on this GPU before any weights are used

    def _group(self,layer,expert):
        key=(layer,expert)
        if key not in self.cache.plan:
            prefix=f'layers.{layer}.ffn.'+(f'experts.{expert}' if expert!='shared' else 'shared_experts')
            names=[prefix+f'.{w}.{kind}' for w in ('w1','w2','w3') for kind in ('weight','scale')]
            for n in names:
                if n not in self.catalog.tensors: raise ValueError('Missing '+n)
            self.cache.plan[key]=names
        return key

    def _confident(self,source):
        st=self.sources[source]
        if self.predict_mode=='always': return True
        if self.predict_mode=='shadow': return False
        return (st['layers']>=self.predict_warmup and st['cold_precision_ema'] is not None
                and st['cold_precision_ema']>=self.predict_threshold)

    def _predict(self,x,layer):
        nxt=f'layers.{layer+1}.ffn.gate.'
        if (not self.predict_next or x.shape[0]>2 or nxt+'weight' not in self.catalog.tensors
                or nxt+'bias' not in self.catalog.tensors):
            return
        weight=self._router_tensor(nxt+'weight'); bias=self._router_tensor(nxt+'bias')
        scores=F.softplus(F.linear(x.float(),weight.float())).sqrt()+bias
        k=min(self.predict_next,scores.shape[-1])
        guesses={'router':{int(e) for e in scores.topk(k,dim=-1).indices.flatten().tolist()},
                 'history':set(self._history.get(layer+1,()))}
        cold={name:{e for e in ids if not self.cache.resident(self._group(layer+1,e))}
              for name,ids in guesses.items()}
        self._predicted=(layer+1,guesses,cold)
        self.prediction['guessed']+=len(guesses['router'])
        issue=[]
        for name in ('history','router'):
            if self._confident(name):
                fresh=sorted(cold[name]-set(issue))
                self.sources[name]['reads_issued']+=len(fresh)
                issue+=fresh
        if issue:
            self.cache.prefetch([self._group(layer+1,e) for e in issue],speculative=True)

    def _score(self,layer,routed):
        """Compare the guesses made for this layer with what its router really chose."""
        if self._predicted is not None and self._predicted[0]==layer:
            _,guesses,cold=self._predicted
            self.prediction['routed']+=len(routed)
            self.prediction['routed_and_guessed']+=len(routed&guesses['router'])
            for name,ids in guesses.items():
                st=self.sources[name]
                st['layers']+=1; st['guessed']+=len(ids); st['correct']+=len(ids&routed)
                st['cold_guessed']+=len(cold[name]); st['cold_correct']+=len(cold[name]&routed)
                if cold[name]:
                    p=len(cold[name]&routed)/len(cold[name])
                    ema=st['cold_precision_ema']
                    st['cold_precision_ema']=p if ema is None else 0.9*ema+0.1*p
        self._predicted=None
        if self.predict_next:
            self._history[layer]=set(routed)

    def prediction_stats(self):
        p=self.prediction
        sources={}
        for name,st in self.sources.items():
            sources[name]=dict(st,precision=st['correct']/st['guessed'] if st['guessed'] else None,
                cold_precision=st['cold_correct']/st['cold_guessed'] if st['cold_guessed'] else None,
                reading_now=self._confident(name))
        return dict(p,recall=p['routed_and_guessed']/p['routed'] if p['routed'] else None,
                    experts_guessed_per_layer=self.predict_next,mode=self.predict_mode,
                    threshold=self.predict_threshold,warmup_layers=self.predict_warmup,sources=sources)

    def _router_tensor(self,name):
        item=self.catalog.tensors[name]
        dtype={'BF16':torch.bfloat16,'F32':torch.float32}[item.dtype]
        with self.router.acquire(name) as raw:
            return raw.view(dtype).reshape(item.shape).clone()

    def _linear(self,entry,key,x,w,reference=False,decoded=False):
        i=('w1','w2','w3').index(w)*2
        wi=self.catalog.tensors[self.cache.plan[key][i]]
        si=self.catalog.tensors[self.cache.plan[key][i+1]]
        if si.dtype!='F8_E8M0': raise ValueError('Unsupported scale format')
        raw=entry.tensors[i].reshape(wi.shape)
        scale=entry.tensors[i+1].reshape(si.shape)
        if reference:
            if wi.dtype=='I8': matrix=cpu_decode(raw.cpu().numpy(),scale.cpu().numpy())
            elif wi.dtype=='F8_E4M3': matrix=fp8_decode_cpu(raw.cpu().numpy(),scale.cpu().numpy())
            else: raise ValueError('Unsupported matrix format')
            return torch.from_numpy(activation(x).double().numpy()@matrix.T).to(torch.bfloat16)
        fused=fused_kernels.get(torch)
        if not decoded and fused.enabled and wi.dtype in ('I8','F8_E4M3') and x.shape[0]<=fused_kernels.MAX_TOKENS:
            checked=getattr(entry,'scales_ok',None)
            if checked is None: checked=entry.scales_ok=set()
            if i not in checked:  # E8M0 255 is NaN; the reference rejects it on every call
                if bool((scale==255).any()): raise ValueError('NaN scale')
                checked.add(i)
            xa=activation(x)
            return fused.block(xa,raw,scale,wi.dtype=='I8',
                               lambda:self._linear(entry,key,x,w,decoded=True),torch.bfloat16)
        if wi.dtype=='I8': matrix=gpu_decode(raw,scale)
        elif wi.dtype=='F8_E4M3':
            if bool((scale==255).any()): raise ValueError('NaN scale')
            matrix=raw.view(torch.float8_e4m3fn).float()
            factor=torch.exp2(scale.float()-127)
            r,c=matrix.shape
            if r%32==0 and c%32==0 and tuple(scale.shape)==(r//32,c//32):
                matrix.view(r//32,32,c//32,32).mul_(factor[:,None,:,None])  # same products, no expanded copy
            else:
                matrix=matrix*factor.repeat_interleave(32,0).repeat_interleave(32,1)[:r,:c]
        else: raise ValueError('Unsupported matrix format')
        return F.linear(activation(x),matrix).to(torch.bfloat16)

    @torch.inference_mode()
    def forward(self,x,layer,validate=True):
        if x.ndim!=2 or x.shape[-1]!=self.cfg['hidden_size'] or x.dtype!=torch.bfloat16:
            raise ValueError('Expected [tokens, hidden_size] BF16 input')
        cfg=self.cfg
        prefix=f'layers.{layer}.ffn.gate.'
        weight=self._router_tensor(prefix+'weight'); bias=self._router_tensor(prefix+'bias')
        weights,ids=route(x,weight,bias,cfg)
        ids_cpu=ids.cpu()
        x_cpu=x.cpu()
        if validate:
            wr,ir=route_reference(x_cpu,weight.cpu(),bias.cpu(),cfg)
            if not torch.equal(ids_cpu,ir): raise AssertionError('GPU and CPU selected different experts')
            torch.testing.assert_close(weights.cpu(),wr,rtol=2e-5,atol=2e-6)
        keys=[self._group(layer,int(e)) for e in sorted(set(ids_cpu.flatten().tolist()))]
        keys.append(self._group(layer,'shared'))
        if self._last_layer is None or layer<=self._last_layer:
            self.cache.new_pass()
        self._last_layer=layer
        # v0.7: nothing this layer needs may be evicted while the layer runs.
        self.cache.begin_layer(keys)
        self.cache.discard_pending(keep=keys)
        self._score(layer,{int(e) for e in ids_cpu.flatten().tolist()})
        self.cache.prefetch(keys[:1+self.lookahead])
        y=torch.zeros_like(x,dtype=torch.float32)
        yr=torch.zeros_like(x_cpu,dtype=torch.float32)
        for pos,key in enumerate(keys):
            # The selected experts are known exactly after routing; read the next few in the background.
            self.cache.prefetch(keys[pos:pos+1+self.lookahead])
            entry=self.cache.get(key,(key,))
            if key[1]=='shared':
                rows=torch.arange(x.shape[0]);cols=None
            else:
                rows,cols=torch.where(ids_cpu==key[1])
            # Small index/mix tensors prepared before dispatching compute.
            rows_gpu=rows.to(x.device)
            coeff=None if cols is None else weights[rows_gpu,cols.to(x.device),None]
            part=expert_forward(x[rows_gpu],coeff,
                lambda z,w:self._linear(entry,key,z,w),cfg['swiglu_limit'])
            y.index_add_(0,rows_gpu,part.float())
            if validate:
                cr=None if cols is None else wr[rows,cols,None]
                ref=expert_forward(x_cpu[rows],cr,
                    lambda z,w:self._linear(entry,key,z,w,True),cfg['swiglu_limit'])
                yr.index_add_(0,rows,ref.float())
            del part
        # v0.7: guesses for the next layer go out only now, when this layer's own reads are done,
        # so a guess can never delay an expert the router has actually selected.
        self._predict(x,layer)
        result=y.to(torch.bfloat16)
        if not torch.isfinite(result).all(): raise AssertionError('Nonfinite MoE output')
        metrics={}
        if validate:
            a=result.cpu().float(); b=yr.to(torch.bfloat16).float()
            rel=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
            cos=float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0).clamp(-1,1))
            if rel>0.01 or cos<0.999: raise AssertionError(f'MoE mismatch L2={rel} cosine={cos}')
            metrics=dict(relative_L2_error=rel,cosine_similarity=cos,max_abs_error=float((a-b).abs().max()))
        return result,dict(layer=layer,selected_experts=ids_cpu.tolist(),routing_weights=weights.cpu().tolist(),
            unique_routed_experts=len(keys)-1,shared_expert=True,**metrics)

    def close(self):
        self.cache.close();self.router.close()


def fp8_decoder_test():
    raw=np.arange(256,dtype=np.uint8).reshape(8,32)
    # Validate all finite codes, separately checking rejection of NaNs.
    raw[(raw&127)==127]=0
    scale=np.full((1,1),127,dtype=np.uint8)
    expected=torch.from_numpy(raw).view(torch.float8_e4m3fn).float().numpy()
    np.testing.assert_array_equal(fp8_decode_cpu(raw,scale),expected)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=default_model_root())
    p.add_argument('--layers',default='0,1,2,3')
    p.add_argument('--tokens',type=int,default=2)
    p.add_argument('--cache-mib',type=int,default=256)
    p.add_argument('--output',default='moe_correctness_results.json')
    args=p.parse_args()
    if not 1<=args.tokens<=4 or not 128<=args.cache_mib<=8192: p.error('tokens 1..4, cache 128..8192 MiB')
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    free,total=torch.cuda.mem_get_info()
    if total<14*GIB or free<3*GIB: raise SystemExit('Insufficient VRAM/headroom')
    torch.cuda.set_per_process_memory_fraction(12*GIB/total)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4)
    fp8_decoder_test()
    # Warm the GPU before timed real-weight work.
    warm=torch.randn(32,32,device='cuda'); (warm@warm).sum().item();del warm
    torch.cuda.reset_peak_memory_stats()
    catalog=Catalog(args.model)
    layers=[int(v) for v in args.layers.split(',')]
    if any(l<0 or l>=catalog.config['text_config']['num_hidden_layers'] for l in layers): p.error('Invalid layer')
    engine=StreamedMoE(catalog,args.cache_mib)
    records=[]
    begin=time.perf_counter()
    try:
        for layer in layers:
            torch.manual_seed(1234+layer)
            x=torch.randn(args.tokens,engine.cfg['hidden_size']).to(torch.bfloat16).cuda()
            y,record=engine.forward(x,layer)
            records.append(record)
            print(json.dumps(dict(status='PASS',**record)),flush=True)
        report=dict(status='PASS',scope='Independent MoE sublayers, NOT sequential transformer layers or generation',
            reference='NumPy FP64 routing and matrix products; independent FP4/FP8 decode; shared PyTorch activation quantization/SwiGLU; not TileLang kernel parity',
            layers=records,logical_weight_read_MB=(engine.cache.reader.bytes_read+engine.router.reader.bytes_read)/1e6,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/GIB,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/GIB,
            evictions=engine.cache.evictions,gpu_cache_hits=engine.cache.hits,expert_cache=engine.cache.stats(),
            expert_requests=engine.cache.requests,elapsed_including_CPU_validation=time.perf_counter()-begin)
        Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:engine.close()

if __name__=='__main__':main()
