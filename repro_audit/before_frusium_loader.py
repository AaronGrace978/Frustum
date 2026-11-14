"""Frusium_loader: bounded safetensors streaming prototype, not an inference engine.
Default inspect reads JSON/headers only. No model source is imported or executed.
"""
from __future__ import annotations
import argparse
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import json
import mmap
from pathlib import Path
import struct
import tempfile
import time
import unittest

@dataclass(frozen=True)
class TensorRange:
    name: str
    path: Path
    offset: int
    size: int
    dtype: str
    shape: tuple

class Catalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / 'config.json').read_text())
        self.index = json.loads((self.root / 'model.safetensors.index.json').read_text())
        self.tensors = {}
        self.header_bytes = 0
        weights = self.index['weight_map']
        for filename in sorted(set(weights.values())):
            path = (self.root / filename).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Shard path escapes model directory')
            with path.open('rb') as f:
                prefix = f.read(8)
                if len(prefix) != 8:
                    raise ValueError('Truncated header prefix')
                length = struct.unpack('<Q', prefix)[0]
                if not 2 <= length <= 64 * 1024 * 1024:
                    raise ValueError('Invalid or oversized header')
                payload = f.read(length)
                if len(payload) != length:
                    raise ValueError('Truncated header')
                header = json.loads(payload)
                self.header_bytes += 8 + length
            file_size = path.stat().st_size
            for name, info in header.items():
                if name == '__metadata__':
                    continue
                if weights.get(name) != filename:
                    raise ValueError('Header/index mismatch: ' + name)
                start, end = info['data_offsets']
                if not 0 <= start <= end <= file_size - 8 - length:
                    raise ValueError('Invalid tensor extent: ' + name)
                if name in self.tensors:
                    raise ValueError('Duplicate tensor: ' + name)
                self.tensors[name] = TensorRange(name, path, 8 + length + start,
                                               end-start, info['dtype'], tuple(info['shape']))
        if set(weights) != set(self.tensors):
            raise ValueError('Index references missing tensors')

    def report(self):
        total = sum(t.size for t in self.tensors.values())
        expected = self.index.get('metadata', {}).get('total_size')
        if expected is not None and total != expected:
            raise ValueError('Index total_size differs from headers')
        return dict(architectures=self.config.get('architectures'),
                    configured_dtype=self.config.get('dtype'),
                    quantization=self.config.get('quantization_config'),
                    text_config=self.config.get('text_config'),
                    shards=len(set(self.index['weight_map'].values())),
                    tensors=len(self.tensors), tensor_bytes=total,
                    stored_dtypes=dict(Counter(t.dtype for t in self.tensors.values())),
                    header_bytes_read=self.header_bytes, weight_payload_bytes_read=0)

class RangeReader:
    """Map one aligned window at a time; no whole-shard copy or retained mapping.
    OS page caching remains OS-managed; this is not a total process-RAM limit.
    """
    def __init__(self, chunk_bytes=8*1024*1024):
        if chunk_bytes <= 0:
            raise ValueError('chunk_bytes must be positive')
        self.chunk_bytes = chunk_bytes
        self.bytes_read = 0

    def copy(self, item, consume):
        with item.path.open('rb') as f:
            for relative in range(0, item.size, self.chunk_bytes):
                size = min(self.chunk_bytes, item.size-relative)
                absolute = item.offset + relative
                base = absolute - absolute % mmap.ALLOCATIONGRANULARITY
                delta = absolute - base
                with mmap.mmap(f.fileno(), delta+size, access=mmap.ACCESS_READ,
                               offset=base) as mm:
                    view = memoryview(mm)[delta:delta+size]
                    try:
                        consume(relative, view)
                    finally:
                        view.release()
                self.bytes_read += size

class FrusiumLoader:
    """Single-threaded byte cache with leased entries and bounded staging.
    CUDA copies use a dedicated stream but synchronize each chunk before reuse.
    v0 prioritizes correctness; double-buffer overlap is intentionally deferred.
    Consumers must release a lease only after submitting all tensor operations.
    Never retain a tensor reference outside its lease.
    """
    def __init__(self, catalog, device='cpu', budget_bytes=256*1024*1024,
                 chunk_bytes=8*1024*1024):
        import torch
        if budget_bytes <= 0 or chunk_bytes <= 0:
            raise ValueError('Budgets must be positive')
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only cpu and cuda supported')
        self.cuda = self.device.type == 'cuda'
        if self.cuda and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: install a CUDA-enabled PyTorch build')
        self.catalog = catalog
        self.budget = budget_bytes
        self.reader = RangeReader(chunk_bytes)
        self.cache = OrderedDict()
        self.leases = Counter()
        self.used = self.peak = self.hits = self.misses = self.evictions = 0
        self.stage = torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=self.cuda)
        self.stage_view = memoryview(self.stage.numpy())
        self.stream = torch.cuda.Stream(device=self.device) if self.cuda else None

    def _sync(self):
        if self.cuda:
            self.torch.cuda.synchronize(self.device)

    def _load(self, name):
        if name in self.cache:
            self.hits += 1
            self.cache.move_to_end(name)
            return self.cache[name]
        item = self.catalog.tensors[name]
        if item.size > self.budget:
            raise ValueError('Tensor exceeds cache budget; use row/chunk adapter: ' + name)
        while self.used + item.size > self.budget:
            victim = next((k for k in self.cache if not self.leases[k]), None)
            if victim is None:
                raise RuntimeError('Cache exhausted by live leases')
            self._sync()
            old = self.cache.pop(victim)
            self.used -= old.numel()
            del old
            self.evictions += 1
        self.misses += 1
        target = self.torch.empty(item.size, dtype=self.torch.uint8, device=self.device)
        def consume(offset, view):
            n = len(view)
            self.stage_view[:n] = view
            if self.cuda:
                with self.torch.cuda.stream(self.stream):
                    target[offset:offset+n].copy_(self.stage[:n], non_blocking=True)
                self.stream.synchronize()
            else:
                target[offset:offset+n].copy_(self.stage[:n])
        self.reader.copy(item, consume)
        self.cache[name] = target
        self.used += item.size
        self.peak = max(self.peak, self.used)
        return target

    @contextmanager
    def acquire(self, name):
        tensor = self._load(name)
        self.leases[name] += 1
        try:
            yield tensor
        finally:
            self._sync()
            self.leases[name] -= 1

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    payload_bytes_read=self.reader.bytes_read,
                    cache_bytes=self.used, peak_cache_bytes=self.peak,
                    budget_bytes=self.budget, staging_bytes=len(self.stage_view))

    def close(self):
        if any(self.leases.values()):
            raise RuntimeError('Cannot close with live leases')
        self._sync()
        self.cache.clear()
        self.used = 0
        self.stage_view.release()

class PrototypeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        header = {k: dict(dtype='U8', shape=[16], data_offsets=[i*16,(i+1)*16])
                  for i,k in enumerate(('a','b','c'))}
        raw = json.dumps(header).encode()
        (self.root/'s.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+bytes(range(48)))
        (self.root/'config.json').write_text('{}')
        (self.root/'model.safetensors.index.json').write_text(json.dumps(
            dict(metadata=dict(total_size=48), weight_map={k:'s.safetensors' for k in header})))
        self.catalog = Catalog(self.root)
    def tearDown(self):
        self.tmp.cleanup()
    def test_header_only(self):
        self.assertEqual(self.catalog.report()['weight_payload_bytes_read'],0)
    def test_unaligned_chunk_copy(self):
        out = bytearray(16)
        reader = RangeReader(3)
        def consume(offset, view):
            out[offset:offset+len(view)] = view
        reader.copy(self.catalog.tensors['b'],consume)
        self.assertEqual(out,bytes(range(16,32)))
        self.assertEqual(reader.bytes_read,16)
    def test_cache_and_leases(self):
        loader = FrusiumLoader(self.catalog,budget_bytes=32,chunk_bytes=3)
        with loader.acquire('a') as a:
            self.assertEqual(a.tolist(),list(range(16)))
            with loader.acquire('b'):
                with self.assertRaises(RuntimeError):
                    with loader.acquire('c'): pass
            with loader.acquire('c') as c:
                self.assertEqual(c.tolist(),list(range(32,48)))
        with loader.acquire('a'): pass
        self.assertEqual(loader.hits,1)
        self.assertEqual(loader.evictions,1)
        self.assertLessEqual(loader.peak,32)
        loader.close()
    def test_oversized_rejected_before_read(self):
        loader = FrusiumLoader(self.catalog,budget_bytes=8,chunk_bytes=3)
        with self.assertRaises(ValueError):
            with loader.acquire('a'): pass
        self.assertEqual(loader.reader.bytes_read,0)
        loader.close()
    def test_truncated_file(self):
        path=self.root/'s.safetensors'
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError): Catalog(self.root)
    def test_path_escape(self):
        (self.root/'model.safetensors.index.json').write_text(json.dumps(
            dict(weight_map={'a':'../outside.safetensors'})))
        with self.assertRaises(ValueError): Catalog(self.root)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=r'G:\Models\DeepSeek-V4.1-Flash')
    sub=p.add_subparsers(dest='command')
    sub.add_parser('inspect')
    sub.add_parser('self-test')
    bench=sub.add_parser('bench',help='Explicitly read selected weight payloads; no inference')
    bench.add_argument('--tensor',action='append',required=True)
    bench.add_argument('--device',default='cpu')
    bench.add_argument('--budget-mib',type=int,default=256)
    bench.add_argument('--chunk-mib',type=int,default=8)
    bench.add_argument('--rounds',type=int,default=2)
    args=p.parse_args()
    if args.command=='self-test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrototypeTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    catalog=Catalog(args.model)
    if args.command in (None,'inspect'):
        print(json.dumps(catalog.report(),indent=2)); return
    if args.rounds<1: p.error('--rounds must be positive')
    for name in args.tensor:
        if name not in catalog.tensors: p.error('Unknown tensor: '+name)
    loader=FrusiumLoader(catalog,args.device,args.budget_mib*1024**2,args.chunk_mib*1024**2)
    start=time.perf_counter()
    try:
        for _ in range(args.rounds):
            for name in args.tensor:
                with loader.acquire(name): pass
        elapsed=time.perf_counter()-start
        print(json.dumps(dict(**loader.stats(),seconds=elapsed,
              effective_read_GBps=loader.reader.bytes_read/max(elapsed,1e-9)/1e9,
              note='Includes transfer/allocation; OS caching may inflate SSD throughput. Not tokens/sec.'),indent=2))
    finally:
        loader.close()

if __name__=='__main__': main()
