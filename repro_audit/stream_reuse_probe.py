import json,struct,tempfile
from pathlib import Path
import torch
from frustum_loader import Catalog,FrustumLoader
N=1024*1024
with tempfile.TemporaryDirectory() as td:
 p=Path(td);header=json.dumps({'probe':{'dtype':'U8','shape':[N],'data_offsets':[0,N]}}).encode()
 (p/'s.safetensors').write_bytes(struct.pack('<Q',len(header))+header+bytes([19])*N)
 (p/'config.json').write_text('{}');(p/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':N},'weight_map':{'probe':'s.safetensors'}}))
 loader=FrustumLoader(Catalog(p),'cuda',budget_bytes=2*N,chunk_bytes=N)
 old=torch.empty(N,dtype=torch.uint8,device='cuda');pointer=old.data_ptr()
 torch.cuda._sleep(100_000_000)
 old.fill_(73)
 del old
 with loader.acquire('probe') as value:
  reused=value.data_ptr()==pointer
  actual=value.cpu()
  correct=bool((actual==19).all())
  report={'same_allocation_reused':reused,'correct_bytes':correct,'first_byte':int(actual[0])}
 loader.close()
 print(json.dumps(report))
