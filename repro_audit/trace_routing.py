import hashlib,json,sys,platform
from pathlib import Path
import torch
from layer0_block import LayerZeroBlock
from frustum_loader import Catalog
from sparse_rows import SparseRows
from streamed_moe import route

def digest(x):return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
@torch.inference_mode()
def main():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
 torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
 c=Catalog(r'G:\Models\DeepSeek-V4.1-Flash');b=LayerZeroBlock(c);trace={}
 for method in ['tensor','linear','norm','mixes','attention']:
  original=getattr(b,method)
  def wrapped(*a,_f=original,_name=method,**kw):
   y=_f(*a,**kw);label=_name+':'+str(a[1] if len(a)>1 else '')
   trace[label]=[digest(v) for v in y] if isinstance(y,tuple) else digest(y)
   return y
  setattr(b,method,wrapped)
 def routed(x,layer,validate=False):
  trace['moe_input']=digest(x)
  w=b.moe._router_tensor('layers.0.ffn.gate.weight');bias=b.moe._router_tensor('layers.0.ffn.gate.bias')
  mix,ids=route(x,w,bias,b.cfg);trace['ids']=ids.cpu().tolist();trace['mix']=mix.cpu().tolist()
  return torch.zeros_like(x),{}
 b.moe.forward=routed
 lookup=SparseRows(c);x=lookup.get('embed.weight',[100,200,300]).unsqueeze(0)
 trace['embedding']=digest(x)
 h=x.unsqueeze(2).repeat(1,1,c.config['text_config']['hc_mult'],1)
 try:b.forward(h)
 finally:b.close()
 trace['torch']=torch.__version__;trace['python']=sys.version
 trace['code_hashes']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['frustum_loader.py','layer0_block.py','block_primitives.py','sparse_rows.py','streamed_moe.py']}
 Path(sys.argv[1]).write_text(json.dumps(trace,indent=2));print(json.dumps({'ids':trace['ids'],'output':sys.argv[1]}))
main()
