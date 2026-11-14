"""Two-tier packed-weight cache for routed experts: GPU, then host RAM, then disk.

Replaces the v0 AsyncExpertCache in the inference path (benchmark_layers.py
still carries the original for the synthetic transfer benchmark).

Design
- Disk reads run on a small thread pool, so the experts a layer needs are
  fetched concurrently while the GPU works on the ones already present. File
  reads release the GIL.
- Everything that touches CUDA happens on the calling thread and the current
  stream, with blocking uploads. There are no side streams or events to keep
  consistent, so a tensor is safe to use as soon as get() returns it.
- The host tier holds packed bytes in ordinary RAM. A host hit costs one
  upload (milliseconds) instead of a drive read. It keeps a copy of whatever
  it admits whether or not the GPU also holds it, which makes GPU eviction
  free. Pinned keys (shared experts) never leave the GPU, so they are not
  duplicated in RAM.
- Victim selection lives in cache_policy.Tier so the replay tool can run the
  same logic on recorded routing.

Weights stay packed (FP4/FP8 bytes) in both tiers; decoding happens per use.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

from cache_policy import Frequency, Tier
from frusium_loader import RangeReader


@dataclass
class Entry:
    tensors: list
    size: int


class TieredExpertCache:
    def __init__(self, torch, catalog, device, gpu_bytes, host_bytes=0, *, io_threads=4,
                 gpu_policy='lfu', host_policy='lru', pinned=None, half_life=8,
                 cap_bytes=None, headroom_bytes=0):
        if gpu_bytes <= 0 or host_bytes < 0 or io_threads < 0:
            raise ValueError('Invalid cache configuration')
        self.t = torch
        self.catalog = catalog
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.plan = {}
        self.reader = RangeReader()
        self.frequency = Frequency(half_life)
        self.pinned = pinned or (lambda key: False)
        self.gpu = Tier(gpu_bytes, gpu_policy, self.pinned)
        self.host = Tier(host_bytes, host_policy)
        self.cache = {}    # key -> Entry on the compute device
        self.ram = {}      # key -> list of CPU uint8 tensors
        self.pending = {}  # key -> Future[list[bytearray]]
        self.pool = ThreadPoolExecutor(io_threads, thread_name_prefix='frusium-io') if io_threads else None
        self.cap, self.headroom = cap_bytes, headroom_bytes
        self.requests = self.hits = self.host_hits = self.disk_reads = self.evictions = 0
        self.upload_seconds = self.wait_seconds = 0.0
        self.upload_bytes = 0

    # Attribute names kept from the v0 cache so existing reports keep working.
    @property
    def used(self):
        return self.gpu.used

    @property
    def budget(self):
        return self.gpu.budget

    def size(self, key):
        return sum(self.catalog.tensors[name].size for name in self.plan[key])

    def _read(self, key):
        return [self.reader.read(self.catalog.tensors[name]) for name in self.plan[key]]

    def new_pass(self):
        """Call once per pass over the layers (one prefill or one decode step)."""
        self.frequency.tick()

    def prefetch(self, keys):
        """Start background reads for keys that are in neither tier. Never blocks."""
        if self.pool is None:
            return
        for key in keys:
            if key not in self.cache and key not in self.ram and key not in self.pending:
                self.pending[key] = self.pool.submit(self._read, key)

    def discard_pending(self):
        """Forget background reads nobody collected, e.g. after an interrupted forward pass."""
        for future in self.pending.values():
            future.cancel()
        self.pending.clear()

    def _memory_ok(self, additional):
        if not self.cuda or self.cap is None:
            return True
        free, _ = self.t.cuda.mem_get_info(self.device)
        allocated = self.t.cuda.memory_allocated(self.device)
        # Blocks PyTorch has reserved but is not using are available to this process.
        reusable = self.t.cuda.memory_reserved(self.device) - allocated
        return allocated + additional <= self.cap and free + reusable >= self.headroom + additional

    def _evict(self, key):
        del self.cache[key]
        self.evictions += 1

    def _remember(self, key, tensors, size):
        """Keep a host copy when the host tier wants one."""
        if not self.host.budget or self.pinned(key):
            return
        evicted = self.host.admit(key, size, self.frequency)
        if evicted is None:
            return
        for victim in evicted:
            del self.ram[victim]
        self.ram[key] = tensors
        self.host.add(key, size)

    def get(self, key, protected=()):
        """Return the Entry for key, loading it if needed. Blocks until it is usable."""
        self.requests += 1
        self.frequency.touch(key)
        entry = self.cache.get(key)
        if entry is not None:
            self.hits += 1
            self.gpu.touch(key)
            return entry
        size = self.size(key)
        if size > self.gpu.budget:
            raise RuntimeError('One expert exceeds cache budget')
        tensors = self.ram.get(key)
        if tensors is not None:
            self.host_hits += 1
            self.host.touch(key)
        else:
            began = time.perf_counter()
            future = self.pending.pop(key, None)
            buffers = self._read(key) if future is None else future.result()
            self.wait_seconds += time.perf_counter() - began
            self.disk_reads += 1
            tensors = [self.t.frombuffer(buffer, dtype=self.t.uint8) for buffer in buffers]
            self._remember(key, tensors, size)
        keep = set(protected)
        keep.add(key)
        for victim in self.gpu.make_room(size, self.frequency, keep):
            self._evict(victim)
        # The byte budget is a plan; the allocator is the truth. Shed more if it is tight.
        while not self._memory_ok(size):
            victim = self.gpu.victim(self.frequency, keep)
            if victim is None:
                raise RuntimeError('Insufficient free dedicated VRAM; stop before allocating')
            self.gpu.remove(victim)
            self._evict(victim)
        began = time.perf_counter()
        entry = Entry([tensor.to(self.device) for tensor in tensors], size)
        self.upload_seconds += time.perf_counter() - began
        self.upload_bytes += size
        self.cache[key] = entry
        self.gpu.add(key, size)
        return entry

    def stats(self):
        return dict(requests=self.requests, gpu_hits=self.hits, host_hits=self.host_hits,
                    disk_reads=self.disk_reads, evictions=self.evictions,
                    payload_bytes_read=self.reader.bytes_read,
                    reader_seconds_all_threads=self.reader.read_seconds,
                    blocked_on_disk_seconds=self.wait_seconds,
                    upload_seconds=self.upload_seconds, upload_bytes=self.upload_bytes,
                    gpu_bytes=self.gpu.used, gpu_budget_bytes=self.gpu.budget, gpu_policy=self.gpu.policy,
                    host_bytes=self.host.used, host_budget_bytes=self.host.budget, host_policy=self.host.policy)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.cuda:
            self.t.cuda.synchronize(self.device)
        self.pending.clear()
        self.cache.clear()
        self.ram.clear()
        self.reader.close()
