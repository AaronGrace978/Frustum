"""Bounded row lookup for BF16 token embeddings and FP8 Engram tables.
Never maps or copies an entire table. Rows live in a small GPU LRU cache.
"""
from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from frustum_loader import Catalog, RangeReader

class SparseRows:
    def __init__(self,catalog,device='cuda',budget_bytes=32*1024**2):
        self.catalog=catalog;self.device=device;self.budget=budget_bytes
        self.reader=RangeReader(1024*1024)
        self.cache=OrderedDict();self.used=self.hits=self.requests=0
    def _bytes(self,name,row):
        item=self.catalog.tensors[name]
        if len(item.shape)!=2 or not 0<=row<item.shape[0]:raise IndexError('Row outside table')
        size=item.size//item.shape[0]
        if item.size%item.shape[0]:raise ValueError('Nonuniform rows')
        selected=replace(item,offset=item.offset+row*size,size=size,shape=(item.shape[1],))
        data=bytearray(size)
        def put(offset,view):data[offset:offset+len(view)]=view
        self.reader.copy(selected,put)
        return torch.frombuffer(data,dtype=torch.uint8).clone()
    def get(self,name,rows,scale_name=None):
        item=self.catalog.tensors[name]
        if item.dtype not in ('BF16','F8_E4M3'):raise ValueError('Unsupported row dtype')
        if item.dtype=='F8_E4M3':
            if scale_name is None:raise ValueError('FP8 rows require scales')
            scale=self.catalog.tensors[scale_name]
            if scale.dtype!='F8_E8M0' or scale.shape!=(item.shape[0],item.shape[1]//32):
                raise ValueError('Invalid row-scale layout')
        result=[]
        for row in rows:
            row=int(row);key=(name,scale_name,row);self.requests+=1
            if key in self.cache:
                self.hits+=1;self.cache.move_to_end(key)
            else:
                raw=self._bytes(name,row)
                if item.dtype=='BF16':value=raw.view(torch.bfloat16).to(self.device)
                else:
                    scales=self._bytes(scale_name,row)
                    if bool((scales==255).any()):raise ValueError('NaN scale')
                    value=(raw.view(torch.float8_e4m3fn).float().reshape(-1,32)*torch.exp2(scales.float()-127)[:,None]).flatten().to(torch.bfloat16).to(self.device)
                size=value.numel()*value.element_size()
                if size>self.budget:raise ValueError('One decoded row exceeds cache budget')
                while self.used+size>self.budget:
                    _,old=self.cache.popitem(last=False);self.used-=old.numel()*old.element_size()
                self.cache[key]=value;self.used+=size
            result.append(self.cache[key])
        # Return owned result storage, not a borrowed cache view.
        return torch.stack(result) if result else torch.empty((0,item.shape[1]),dtype=torch.bfloat16,device=self.device)

@torch.inference_mode()
def main():
    root=Path(r'G:\Models\DeepSeek-V4.1-Flash')
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    free,total=torch.cuda.mem_get_info()
    if free<2*1024**3:raise SystemExit('Insufficient VRAM headroom')
    torch.cuda.set_per_process_memory_fraction(min(1,12*1024**3/total))
    c=Catalog(root);lookup=SparseRows(c)
    name='layers.1.engram.embed.weight';scales='layers.1.engram.embed.scale'
    rows=[0,17,c.tensors[name].shape[0]-1,17]
    a=lookup.get(name,rows,scales)
    # An independent scalar E4M3 decoder for selected rows, direct file reads.
    expected=[]
    for row in rows:
        def read_row(n):
            item=c.tensors[n];stride=item.size//item.shape[0]
            with item.path.open('rb') as f:f.seek(item.offset+row*stride);return np.frombuffer(f.read(stride),dtype=np.uint8)
        bits=read_row(name);s=read_row(scales)
        v=[]
        for j,b in enumerate(bits):
            b=int(b);e=(b>>3)&15;m=b&7
            if (b&127)==127 or s[j//32]==255:raise ValueError('Nonfinite value')
            mag=m*2**-9 if e==0 else (1+m/8)*2**(e-7)
            v.append((-mag if b&128 else mag)*2.**(int(s[j//32])-127))
        expected.append(v)
    b=torch.tensor(expected).to(torch.bfloat16)
    torch.testing.assert_close(a.cpu(),b,rtol=0,atol=0)
    # Check token embeddings with a direct reference read, not the row loader.
    token_ids=[0,1,129279,1]
    token=lookup.get('embed.weight',token_ids)
    refs=[];it=c.tensors['embed.weight'];stride=it.size//it.shape[0]
    with it.path.open('rb') as f:
        for row in token_ids:
            f.seek(it.offset+row*stride)
            refs.append(torch.frombuffer(bytearray(f.read(stride)),dtype=torch.bfloat16).clone())
    torch.testing.assert_close(token.cpu(),torch.stack(refs),rtol=0,atol=0)
    before=lookup.reader.bytes_read
    lookup.get(name,rows,scales);lookup.get('embed.weight',token_ids)
    assert lookup.reader.bytes_read==before,'Cache reread bytes on warm lookup'
    try:lookup.get(name,[c.tensors[name].shape[0]],scales)
    except IndexError:pass
    else:raise AssertionError('Out-of-range row not rejected')
    report=dict(status='PASS',engram_table_rows=c.tensors[name].shape[0],
        engram_test_rows=rows,token_test_rows=token_ids,logical_lookup_bytes=before,
        reference_read_bytes=4*264+4*stride,cache_payload_bytes=lookup.used,
        note='Sparse lookup only. N-gram hashing and full Engram residual integration remain pending.')
    Path('sparse_rows_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
