"""Four-layer expert streaming benchmark. Real bytes, synthetic compute; NOT inference."""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import time
from frustum_loader import Catalog, RangeReader

GIB = 1024**3
MIB = 1024**2

def expert_names(catalog, layer, expert):
    prefix = f'layers.{layer}.ffn.experts.{expert}.'
    names = [prefix + f'{w}.{kind}' for w in ('w1','w2','w3') for kind in ('weight','scale')]
    for name in names:
        if name not in catalog.tensors:
            raise ValueError('Missing tensor: ' + name)
    return names

def make_plan(catalog, ids):
    return {(layer, expert): expert_names(catalog, layer, expert)
            for layer in range(4) for expert in ids}

@dataclass
class Entry:
    tensors: list
    size: int
    ready: object
    done: object = None

class AsyncExpertCache:
    # Host calls are single-threaded. GPU compute and copy use distinct streams.
    def __init__(self, torch, catalog, plan, device, cache_bytes, cap_bytes, headroom_bytes):
        self.t = torch
        self.catalog, self.plan, self.device = catalog, plan, device
        self.budget, self.cap, self.headroom = cache_bytes, cap_bytes, headroom_bytes
        self.copy_stream = torch.cuda.Stream(device=device)
        self.compute_stream = torch.cuda.Stream(device=device)
        self.cache = OrderedDict()
        self.used = self.hits = self.requests = self.evictions = 0
        self.reader = RangeReader(8*MIB)
        self.stages = [torch.empty(8*MIB,dtype=torch.uint8,pin_memory=True) for _ in range(2)]
        self.views = [memoryview(x.numpy()) for x in self.stages]
        self.stage_events = [None,None]
        self.slot = 0

    def check_memory(self, additional=0):
        t = self.t
        free, _ = t.cuda.mem_get_info(self.device)
        reserved = t.cuda.memory_reserved(self.device)
        allocated = t.cuda.memory_allocated(self.device)
        # Conservative: reject rather than rely on Windows shared memory.
        if allocated + additional > self.cap or reserved > self.cap:
            raise RuntimeError('12 GiB process allocation guard reached')
        if free < self.headroom + additional:
            raise RuntimeError('Insufficient free dedicated VRAM; stop before allocating')

    def ensure(self, key, protected):
        if key in self.cache:
            self.cache.move_to_end(key)
            return
        names = self.plan[key]
        size = sum(self.catalog.tensors[n].size for n in names)
        if size > self.budget:
            raise RuntimeError('One expert exceeds cache budget')
        while self.used + size > self.budget:
            victim = next((k for k in self.cache if k not in protected), None)
            if victim is None:
                raise RuntimeError('Cache too small for current and next layer')
            old = self.cache.pop(victim)
            (old.done or old.ready).synchronize()
            self.used -= old.size
            del old
            self.evictions += 1
        self.check_memory(size)
        tensors = []
        with self.t.cuda.stream(self.copy_stream):
            for name in names:
                item = self.catalog.tensors[name]
                target = self.t.empty(item.size,dtype=self.t.uint8,device=self.device)
                tensors.append(target)
                def consume(offset, view):
                    slot = self.slot % 2
                    event = self.stage_events[slot]
                    if event is not None:
                        event.synchronize()  # CPU cannot overwrite in-flight pinned bytes.
                    n = len(view)
                    self.views[slot][:n] = view
                    target[offset:offset+n].copy_(self.stages[slot][:n],non_blocking=True)
                    event = self.t.cuda.Event()
                    event.record(self.copy_stream)
                    self.stage_events[slot] = event
                    self.slot += 1
                self.reader.copy(item,consume)
            ready = self.t.cuda.Event()
            ready.record(self.copy_stream)
        self.cache[key] = Entry(tensors,size,ready)
        self.used += size

    def prefetch(self, keys, protected):
        for key in keys:
            self.ensure(key, protected)

    def consume(self, keys, repeats):
        # Counts demand residency, including successful prefetch; not router accuracy.
        self.requests += len(keys)
        self.hits += sum(key in self.cache for key in keys)
        self.prefetch(keys,set(keys))
        start, end = self.t.cuda.Event(enable_timing=True), self.t.cuda.Event(enable_timing=True)
        with self.t.cuda.stream(self.compute_stream):
            for key in keys:
                self.compute_stream.wait_event(self.cache[key].ready)
            start.record(self.compute_stream)
            # Bounded synthetic GPU work touches every packed byte. No FP4 interpretation.
            checksum = self.t.zeros((),dtype=self.t.int64,device=self.device)
            for _ in range(repeats):
                for key in keys:
                    entry = self.cache[key]
                    for raw in entry.tensors:
                        raw.record_stream(self.compute_stream)
                        checksum.add_(raw.sum(dtype=self.t.int64))
            end.record(self.compute_stream)
            for key in keys:
                self.cache[key].done = end
                self.cache.move_to_end(key)
        return start,end,checksum

    def close(self):
        self.t.cuda.synchronize(self.device)
        self.cache.clear()
        for view in self.views:
            view.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=r'G:\Models\DeepSeek-V4.1-Flash')
    p.add_argument('--run',action='store_true',help='Read real weights and run CUDA benchmark; default is metadata-only')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--experts',default='0,1,2,3,4,5',help='Fixed synthetic hot-expert IDs, NOT real routing')
    p.add_argument('--cache-mib',type=int,default=512)
    p.add_argument('--compute-repeats',type=int,default=4)
    p.add_argument('--no-prefetch',action='store_true',help='Control run for overlap comparison')
    args=p.parse_args()
    ids=list(dict.fromkeys(int(x) for x in args.experts.split(',')))
    if not ids or min(ids)<0 or args.rounds<1 or args.compute_repeats<1 or not 1<=args.cache_mib<=10240:
        p.error('Use nonnegative expert IDs, positive rounds/repeats, cache 1..10240 MiB')
    catalog=Catalog(args.model)
    plan=make_plan(catalog,ids)
    sizes={k:sum(catalog.tensors[n].size for n in ns) for k,ns in plan.items()}
    per_layer=[sum(v for (l,e),v in sizes.items() if l==layer) for layer in range(4)]
    minimum=max(per_layer[i]+per_layer[(i+1)%4] for i in range(4))
    if args.cache_mib*MIB<minimum:
        p.error(f'Cache must hold adjacent layers: at least {(minimum+MIB-1)//MIB} MiB')
    print(json.dumps(dict(layers=[0,1,2,3],experts=ids,layer_payload_MB=[x/1e6 for x in per_layer],
          cache_MiB=args.cache_mib,process_allocator_cap_GiB=12,headroom_GiB=2,
          workload='packed-byte checksum, not model inference',metadata_only=not args.run),indent=2))
    if not args.run: return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Install CUDA-enabled PyTorch; no weights read.')
    device=torch.device(args.device)
    if device.type!='cuda': p.error('This benchmark requires CUDA')
    torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    cap=12*GIB
    if total < cap + 2*GIB or free < 2*GIB+args.cache_mib*MIB:
        raise SystemExit('Insufficient dedicated VRAM/headroom; no weights read.')
    # Applies to PyTorch caching allocator, not Windows or other processes.
    torch.cuda.set_per_process_memory_fraction(cap/total,device)
    torch.cuda.reset_peak_memory_stats(device)
    cache=AsyncExpertCache(torch,catalog,plan,device,args.cache_mib*MIB,cap,2*GIB)
    sequence=[[(layer,e) for e in ids] for _ in range(args.rounds) for layer in range(4)]
    print('step layer allocated_GiB reserved_GiB device_used_GiB file_read_MB hit_pct compute_ms wall_ms checksum')
    print('file_read_MB = cumulative logical payload reads; physical SSD reads may be lower due to OS cache.')
    try:
        for step,keys in enumerate(sequence):
            wall=time.perf_counter()
            start,end,checksum=cache.consume(keys,args.compute_repeats)
            if not args.no_prefetch and step+1<len(sequence):
                upcoming=sequence[step+1]
                # Current entries cannot be evicted while their compute is running.
                cache.prefetch(upcoming,set(keys+upcoming))
            end.synchronize()
            value=checksum.item()
            cache.check_memory()
            free,total=torch.cuda.mem_get_info(device)
            print(f'{step:4d} {keys[0][0]:5d} '
                  f'{torch.cuda.memory_allocated(device)/GIB:13.3f} '
                  f'{torch.cuda.memory_reserved(device)/GIB:12.3f} '
                  f'{(total-free)/GIB:15.3f} {cache.reader.bytes_read/1e6:12.2f} '
                  f'{100*cache.hits/cache.requests:7.2f} {start.elapsed_time(end):10.2f} '
                  f'{(time.perf_counter()-wall)*1000:7.2f} {value}',flush=True)
        print(json.dumps(dict(evictions=cache.evictions,expert_hits=cache.hits,
              expert_requests=cache.requests,cache_payload_MiB=cache.used/MIB,
              peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/GIB,
              peak_reserved_GiB=torch.cuda.max_memory_reserved(device)/GIB),indent=2))
    finally:
        cache.close()

if __name__=='__main__': main()
