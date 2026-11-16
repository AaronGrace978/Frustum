"""Frustum_loader: bounded safetensors streaming core (catalog, reader, byte cache).
Default inspect reads JSON/headers only. No model source is imported or executed.
"""
from __future__ import annotations
import argparse
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import io
import mmap
import os
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest


def default_model_root():
    """FRUSTUM_MODEL if set, otherwise the folder that contains this project."""
    return os.environ.get('FRUSTUM_MODEL') or str(Path(__file__).resolve().parent.parent)

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

UNBUFFERED_ALIGN = 4096  # a multiple of every common sector size (512, 4096)


class _UnbufferedFile:
    """One Windows handle opened with FILE_FLAG_NO_BUFFERING (v0.8).

    The OS file cache never sees these reads, so expert bytes are not held twice in RAM
    (once by Windows, once by the host tier). Offsets, lengths and the buffer address must
    be sector-aligned, so each read goes through an aligned bounce buffer owned by the
    calling thread and the wanted slice is copied out. Positioned reads use an OVERLAPPED
    offset on a synchronous handle: no shared file pointer.
    """

    def __init__(self, path):
        import ctypes
        from ctypes import wintypes
        self._ctypes, self._wintypes = ctypes, wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                    ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel = kernel
        # GENERIC_READ, share read/write/delete, OPEN_EXISTING,
        # FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN
        handle = kernel.CreateFileW(str(path), 0x80000000, 0x7, None, 3, 0x20000000 | 0x08000000, None)
        if handle is None or handle == wintypes.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), 'Unbuffered open failed', str(path))
        self._handle = handle

        class Overlapped(ctypes.Structure):
            _fields_ = [('internal', ctypes.c_size_t), ('internal_high', ctypes.c_size_t),
                        ('offset', wintypes.DWORD), ('offset_high', wintypes.DWORD),
                        ('event', wintypes.HANDLE)]
        self._overlapped = Overlapped

    def read_aligned(self, offset, address, length):
        """One ReadFile of `length` bytes at `offset` into `address` (all aligned). Returns bytes read."""
        ctypes = self._ctypes
        overlapped = self._overlapped(0, 0, offset & 0xFFFFFFFF, offset >> 32, None)
        got = self._wintypes.DWORD(0)
        if not self._kernel.ReadFile(self._handle, ctypes.c_void_p(address), length, ctypes.byref(got),
                                     ctypes.byref(overlapped)):
            error = ctypes.get_last_error()
            if error == 38:  # ERROR_HANDLE_EOF
                return 0
            raise OSError(error, 'Unbuffered read failed')
        return got.value

    def close(self):
        if self._handle is not None:
            self._kernel.CloseHandle(self._handle)
            self._handle = None


class RangeReader:
    """Positioned reads from persistent file handles; no whole-shard copy.

    v0 mapped a fresh 8 MiB mmap window per chunk and reopened the shard for
    every tensor. Cold pages then arrive through page faults, a few dozen KiB
    per request, which is slow on USB storage. This version keeps one handle
    per (thread, shard) and issues large sequential read calls straight into
    the destination buffer. mode='mmap' (or FRUSTUM_READER=mmap in the
    environment) selects the old path for A/B runs. mode='unbuffered' (v0.8,
    Windows only) bypasses the OS file cache; elsewhere it falls back to direct
    and sets `fallback_reason`.

    Thread-safe: every thread owns its handles. bytes_read counts logical
    payload bytes; the OS cache may serve them without touching the drive.
    OS page caching remains OS-managed; this is not a total process-RAM limit.
    """
    MAX_HANDLES = 12  # per thread; a Windows process defaults to 512 CRT handles

    def __init__(self, chunk_bytes=8*1024*1024, mode=None, access='sequential'):
        mode = mode or os.environ.get('FRUSTUM_READER', 'direct')
        if chunk_bytes <= 0:
            raise ValueError('chunk_bytes must be positive')
        if mode not in ('direct', 'mmap', 'unbuffered') or access not in ('sequential', 'random'):
            raise ValueError('mode is direct|mmap|unbuffered and access is sequential|random')
        self.fallback_reason = None
        if mode == 'unbuffered' and os.name != 'nt':
            mode, self.fallback_reason = 'direct', 'unbuffered reads are implemented for Windows only'
        self.chunk_bytes = chunk_bytes
        self.mode = mode
        self.bytes_read = 0
        self.read_seconds = 0.0
        self._lock = threading.Lock()
        self._local = threading.local()
        self._all = []
        hint = 'O_SEQUENTIAL' if access == 'sequential' else 'O_RANDOM'
        self._flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, hint, 0)

    def _handle(self, path):
        handles = getattr(self._local, 'handles', None)
        if handles is None:
            handles = self._local.handles = OrderedDict()
            with self._lock:
                self._all.append(handles)
        handle = handles.get(path)
        if handle is None:
            while len(handles) >= self.MAX_HANDLES:
                _, old = handles.popitem(last=False)
                old.close()
            if self.mode == 'unbuffered':
                handle = handles[path] = _UnbufferedFile(path)
            else:
                handle = handles[path] = io.FileIO(os.open(path, self._flags), 'rb', closefd=True)
        else:
            handles.move_to_end(path)
        return handle

    def _account(self, size, seconds):
        with self._lock:
            self.bytes_read += size
            self.read_seconds += seconds

    def readinto(self, item, dest, start=0):
        """Fill `dest` (a writable bytes-like object) from item[start:start+len(dest)]."""
        view = memoryview(dest).cast('B')
        size = len(view)
        if start < 0 or start + size > item.size:
            raise ValueError('Read outside tensor extent')
        began = time.perf_counter()
        if self.mode == 'mmap':
            self._mmap_into(item, view, start)
        elif self.mode == 'unbuffered':
            self._unbuffered_into(item, view, start)
        else:
            handle = self._handle(item.path)
            handle.seek(item.offset + start)
            done = 0
            while done < size:
                got = handle.readinto(view[done:min(size, done + (1 << 30))])
                if not got:
                    raise ValueError('Truncated tensor payload: ' + item.name)
                done += got
        view.release()
        self._account(size, time.perf_counter() - began)

    def _bounce(self):
        """This thread's aligned bounce buffer: (memoryview of chunk_bytes, its address)."""
        cached = getattr(self._local, 'bounce', None)
        if cached is None:
            import ctypes
            align = UNBUFFERED_ALIGN
            length = -(-self.chunk_bytes // align) * align
            raw = bytearray(length + align)  # never resized, so its address stays fixed
            base = ctypes.addressof(ctypes.c_char.from_buffer(raw))
            pad = (-base) % align
            cached = self._local.bounce = (raw, memoryview(raw)[pad:pad + length], base + pad)
        return cached[1], cached[2]

    def _unbuffered_into(self, item, view, start):
        handle = self._handle(item.path)
        bounce, address = self._bounce()
        align = UNBUFFERED_ALIGN
        position = item.offset + start
        end = position + len(view)
        done = 0
        while position < end:
            aligned = position - position % align
            length = min(len(bounce), -(-(end - aligned) // align) * align)
            got = handle.read_aligned(aligned, address, length)
            skip = position - aligned
            usable = min(got - skip, end - position)
            if usable <= 0:
                raise ValueError('Truncated tensor payload: ' + item.name)
            view[done:done + usable] = bounce[skip:skip + usable]
            done += usable
            position += usable

    def _mmap_into(self, item, view, start):
        with open(item.path, 'rb') as f:
            for relative in range(0, len(view), self.chunk_bytes):
                size = min(self.chunk_bytes, len(view) - relative)
                absolute = item.offset + start + relative
                base = absolute - absolute % mmap.ALLOCATIONGRANULARITY
                delta = absolute - base
                with mmap.mmap(f.fileno(), delta + size, access=mmap.ACCESS_READ, offset=base) as mm:
                    source = memoryview(mm)[delta:delta + size]
                    try:
                        view[relative:relative + size] = source
                    finally:
                        source.release()

    def read(self, item):
        """Return the whole payload as a new bytearray."""
        data = bytearray(item.size)
        if item.size:
            self.readinto(item, data)
        return data

    def read_group(self, items, max_gap=65536):
        """Read several tensors with as few positioned reads as possible (v0.7).

        Ranges in the same file that touch, or are separated by at most max_gap
        bytes, are fetched with one read into one buffer; each tensor is returned
        as a writable memoryview into it, in the order given. The bytes are the
        same bytes read() returns. Gap bytes that were read are counted in
        bytes_read, so the counter stays an honest record of what was requested.
        mode='mmap' keeps the one-read-per-tensor behaviour.
        """
        items = list(items)
        if self.mode == 'mmap' or len(items) < 2:
            return [self.read(item) for item in items]
        order = sorted(range(len(items)), key=lambda i: (str(items[i].path), items[i].offset))
        out = [None] * len(items)
        run = []

        def flush():
            if not run:
                return
            first, last = items[run[0]], items[run[-1]]
            span = replace(first, name=first.name + '..' + last.name,
                           size=last.offset + last.size - first.offset)
            data = self.read(span)
            view = memoryview(data)
            for i in run:
                start = items[i].offset - first.offset
                out[i] = view[start:start + items[i].size]
            run.clear()

        for i in order:
            if run:
                prev = items[run[-1]]
                gap = items[i].offset - (prev.offset + prev.size)
                if items[i].path != prev.path or not 0 <= gap <= max_gap:
                    flush()
            run.append(i)
        flush()
        return out

    def copy(self, item, consume):
        """Chunked callback API kept for existing callers: consume(offset, view).

        The view is only valid during the callback.
        """
        buffer = getattr(self._local, 'buffer', None)
        if buffer is None or len(buffer) < min(self.chunk_bytes, item.size):
            buffer = self._local.buffer = bytearray(min(self.chunk_bytes, max(item.size, 1)))
        for relative in range(0, item.size, self.chunk_bytes):
            size = min(self.chunk_bytes, item.size - relative)
            view = memoryview(buffer)[:size]
            try:
                self.readinto(item, view, relative)
                consume(relative, view)
            finally:
                view.release()

    def close(self):
        with self._lock:
            groups, self._all = self._all, []
        for handles in groups:
            for handle in list(handles.values()):
                handle.close()
            handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

class FrustumLoader:
    """Byte cache with leased entries and one bounded staging buffer.

    Single-threaded. Each tensor is read from disk straight into a page-locked
    staging buffer and uploaded with a blocking copy on the current CUDA
    stream, so no cross-stream lifetime tracking is needed. Give it a budget
    that covers every tensor you request and it behaves as a resident store:
    after the first pass nothing is read again.
    Consumers must finish using a tensor before releasing its lease.
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
            old = self.cache.pop(victim)
            self.used -= old.numel()
            del old
            self.evictions += 1
        self.misses += 1
        target = self.torch.empty(item.size, dtype=self.torch.uint8, device=self.device)
        chunk = len(self.stage_view)
        for offset in range(0, item.size, chunk):
            n = min(chunk, item.size - offset)
            self.reader.readinto(item, self.stage_view[:n], offset)
            # Blocking copy: the staging buffer is free again when this returns.
            target[offset:offset+n].copy_(self.stage[:n])
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
        self.reader.close()

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
        for mode in ('direct', 'mmap'):
            out = bytearray(16)
            reader = RangeReader(3, mode=mode)
            def consume(offset, view):
                out[offset:offset+len(view)] = view
            reader.copy(self.catalog.tensors['b'],consume)
            self.assertEqual(out,bytes(range(16,32)))
            self.assertEqual(reader.bytes_read,16)
            reader.close()
    def test_positioned_reads(self):
        reader = RangeReader()
        item = self.catalog.tensors['c']
        self.assertEqual(reader.read(item), bytes(range(32,48)))
        part = bytearray(5)
        reader.readinto(item, part, 9)
        self.assertEqual(part, bytes(range(41,46)))
        with self.assertRaises(ValueError):
            reader.readinto(item, bytearray(5), 12)
        self.assertEqual(reader.bytes_read, 21)
        reader.close()
    def test_threaded_reads_and_handle_cap(self):
        from concurrent.futures import ThreadPoolExecutor
        reader = RangeReader()
        reader.MAX_HANDLES = 1
        names = ['a','b','c'] * 40
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda n: bytes(reader.read(self.catalog.tensors[n])), names))
        expected = dict(a=bytes(range(16)), b=bytes(range(16,32)), c=bytes(range(32,48)))
        self.assertEqual(results, [expected[n] for n in names])
        self.assertEqual(reader.bytes_read, 16 * len(names))
        self.assertTrue(all(len(h) <= 1 for h in reader._all))
        reader.close()
    def test_truncated_payload_detected_on_read(self):
        item = self.catalog.tensors['c']
        path = self.root/'s.safetensors'
        path.write_bytes(path.read_bytes()[:-4])
        reader = RangeReader()
        with self.assertRaises(ValueError):
            reader.read(item)
        reader.close()
    @unittest.skipUnless(os.name == 'nt', 'unbuffered reads are Windows-only')
    def test_unbuffered_matches_direct(self):
        """v0.8: same bytes as direct reads at unaligned offsets, across bounce-buffer chunks and threads."""
        from concurrent.futures import ThreadPoolExecutor
        import random
        data = random.Random(7).randbytes(3 * UNBUFFERED_ALIGN * 5 + 1234)  # not a multiple of the sector
        path = self.root / 'big.bin'
        path.write_bytes(data)
        rng = random.Random(11)
        items = []
        for i in range(60):
            offset = rng.randrange(len(data) - 1)
            items.append(TensorRange(f't{i}', path, offset, rng.randrange(1, len(data) - offset + 1), 'U8', ()))
        items.append(TensorRange('tail', path, len(data) - 5, 5, 'U8', ()))
        for chunk in (3, UNBUFFERED_ALIGN, 2 * UNBUFFERED_ALIGN + 1):
            reader = RangeReader(chunk, mode='unbuffered')
            with ThreadPoolExecutor(4) as pool:
                got = list(pool.map(lambda item: bytes(reader.read(item)), items))
            self.assertEqual(got, [data[t.offset:t.offset + t.size] for t in items])
            grouped = reader.read_group(items[:3])
            self.assertEqual([bytes(g) for g in grouped], [data[t.offset:t.offset + t.size] for t in items[:3]])
            whole = TensorRange('whole', path, 0, len(data), 'U8', ())
            part = bytearray(4099)
            reader.readinto(whole, part, 4093)  # starts 3 bytes before a sector boundary
            self.assertEqual(bytes(part), data[4093:4093 + 4099])
            reader.close()
        short = TensorRange('past_end', path, len(data) - 3, 10, 'U8', ())
        reader = RangeReader(mode='unbuffered')
        with self.assertRaises(ValueError):
            reader.read(short)
        reader.close()

    def test_unbuffered_falls_back_off_windows(self):
        reader = RangeReader(mode='unbuffered')
        if os.name == 'nt':
            self.assertEqual((reader.mode, reader.fallback_reason), ('unbuffered', None))
        else:
            self.assertEqual(reader.mode, 'direct')
            self.assertTrue(reader.fallback_reason)
        self.assertEqual(reader.read(self.catalog.tensors['b']), bytes(range(16, 32)))
        reader.close()

    def test_cache_and_leases(self):
        loader = FrustumLoader(self.catalog,budget_bytes=32,chunk_bytes=3)
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
        loader = FrustumLoader(self.catalog,budget_bytes=8,chunk_bytes=3)
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

def bench_io(catalog, total_mib=384, threads=4, seed=None, min_mib=8):
    """Compare read paths on real shard bytes. No GPU, no inference.

    Three disjoint tensor groups are read once each (cold if the OS has not
    cached them) and then again (warm). Each run picks different tensors unless
    a seed is given, so repeat runs stay mostly cold on a large checkpoint.
    """
    import random
    from concurrent.futures import ThreadPoolExecutor
    seed = int(time.time()) if seed is None else seed
    pool = sorted((t for t in catalog.tensors.values() if t.size >= min_mib * 1024**2),
                  key=lambda t: t.name)
    random.Random(seed).shuffle(pool)
    groups, target = [], total_mib * 1024**2
    for _ in range(3):
        group, size = [], 0
        while pool and size < target:
            group.append(pool.pop())
            size += group[-1].size
        groups.append(group)
    if not all(groups):
        raise SystemExit('Not enough large tensors for an I/O benchmark')

    def serial(reader, group):
        for item in group:
            reader.read(item)

    def threaded(reader, group):
        with ThreadPoolExecutor(threads) as workers:
            list(workers.map(lambda item: len(reader.read(item)), group))

    cases = [('v0 mmap windows', RangeReader(mode='mmap'), serial),
             ('direct reads, 1 thread', RangeReader(), serial),
             (f'direct reads, {threads} threads', RangeReader(), threaded)]
    rows = []
    for (label, reader, run), group in zip(cases, groups):
        size = sum(t.size for t in group)
        row = dict(path=label, MB=round(size / 1e6, 1))
        for phase in ('first_read', 'repeat_read'):
            began = time.perf_counter()
            run(reader, group)
            row[phase + '_MBps'] = round(size / 1e6 / max(time.perf_counter() - began, 1e-9), 1)
        reader.close()
        rows.append(row)
    cold = rows[1]['first_read_MBps']
    if cold < 60:
        verdict = ('First reads are under 60 MB/s even with large sequential requests. That is the ceiling of a '
                   'USB 2.0 link: check that the drive is on a USB 3.x port with its original cable, not a hub.')
    elif cold > 1500:
        verdict = ('These first reads ran at memory speed, so the OS cache served them. Run again (a new seed picks '
                   'other tensors) to measure the drive itself.')
    elif rows[0]['first_read_MBps'] < 0.6 * cold:
        verdict = 'The drive is fine; the old mmap path was the limit. Direct reads fix it.'
    else:
        verdict = 'Both paths read at similar speed here (possibly already cached). Re-run for fresh tensors.'
    return dict(seed=seed, results=rows, verdict=verdict,
                note='first_read may be served by the OS cache if those bytes were read recently; '
                     'repeat_read normally is. MB = 1e6 bytes.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=default_model_root())
    sub=p.add_subparsers(dest='command')
    sub.add_parser('inspect')
    sub.add_parser('self-test')
    io_bench=sub.add_parser('bench-io',help='Measure drive read throughput: old mmap path vs direct reads')
    io_bench.add_argument('--total-mib',type=int,default=384,help='Bytes read per case')
    io_bench.add_argument('--threads',type=int,default=4)
    io_bench.add_argument('--seed',type=int)
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
    if args.command=='bench-io':
        if args.total_mib<16 or args.threads<1: p.error('Use --total-mib >= 16 and --threads >= 1')
        print(json.dumps(bench_io(catalog,args.total_mib,args.threads,args.seed),indent=2)); return
    if args.command in (None,'inspect'):
        print(json.dumps(catalog.report(),indent=2)); return
    if args.rounds<1: p.error('--rounds must be positive')
    for name in args.tensor:
        if name not in catalog.tensors: p.error('Unknown tensor: '+name)
    loader=FrustumLoader(catalog,args.device,args.budget_mib*1024**2,args.chunk_mib*1024**2)
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
