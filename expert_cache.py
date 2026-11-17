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

v0.7 (none of it changes which expert runs or any arithmetic):
- begin_layer(keys) protects every expert of the layer in progress in both
  tiers. Before, loading one expert of a layer could evict another expert of
  the same layer that had not been used yet, turning a hit into a re-read.
  FRUSTUM_PROTECT_LAYER=0 restores the old behaviour.
- One positioned read per expert instead of one per tensor when the tensors
  are adjacent on disk (RangeReader.read_group). FRUSTUM_COALESCE=0 restores
  per-tensor reads.
- Speculative reads (next-layer guesses) are tracked separately from demand
  reads: a guess that is not needed is cancelled if it has not started and
  counted as waste if it has; get() adopts a guess that turned out right.
- io_ledger.IOLedger records what the roofline needs: unique and repeated
  bytes, drive-busy time, queue depth, blocked time and overlap.
- Optional RAM guard: shrink the host tier when free RAM falls below a floor.
- Optional spill tier (spill_tier.SpillTier, set cache.spill): verified exact copies of
  experts on a faster local drive, consulted before the checkpoint drive.

v0.8 pipelined uploads (opt-in: FRUSTUM_ASYNC_UPLOAD=1, CUDA only). OFF by default because
it measured slower on the target machine: 4.33 vs 4.07 s/token median, token ids identical
(bench_runs/v08_pipeline_and_gated). Uploads were already hidden behind drive reads, and the
extra pinned allocations and copies cost more than they removed. Kept for A/B runs and for
machines where reads are not the bottleneck.
- When a reader thread finishes an expert it hands it to one of two staging threads, which
  copy it into page-locked (pinned) memory; prefetch() stages host-tier hits the same way.
  Staging threads only allocate pinned host memory (thread-safe); they launch no GPU work.
- get() allocates the GPU copy on a dedicated copy stream, starts a non-blocking copy from
  the pinned bytes, and makes the current stream wait on an event. The calling thread no
  longer blocks on uploads, so it can queue the next expert while the GPU copies and computes.
- Optional inspect(key, cpu_tensors) runs in the staging thread and returns attributes to set
  on the Entry; streamed_moe uses it to do the NaN-scale check on the CPU instead of forcing
  a GPU synchronisation on first use. Bytes, kernels and summation order are unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import time

from cache_policy import Frequency, Tier
from frustum_loader import RangeReader
from io_ledger import IOLedger


def _flag(name, default=True):
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in ('0', 'false', 'off', 'no')


class _Buffers(list):
    """A read result that remembers which storage it came from ('disk' or 'spill')."""
    source = 'disk'


@dataclass
class Entry:
    tensors: list
    size: int


@dataclass
class _Staged:
    """Pinned copies of an expert's packed tensors, ready for a non-blocking upload."""
    pinned: list
    attrs: dict


class TieredExpertCache:
    def __init__(self, torch, catalog, device, gpu_bytes, host_bytes=0, *, io_threads=4,
                 gpu_policy='lfu', host_policy='lru', pinned=None, half_life=8,
                 cap_bytes=None, headroom_bytes=0, coalesce=None, protect_layer=None,
                 ram_probe=None, ram_floor_bytes=0, async_upload=None):
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
        self.pending = {}  # key -> Future[list of buffers]
        self.speculative = set()  # keys in pending that were issued as guesses
        self.pool = ThreadPoolExecutor(io_threads, thread_name_prefix='frustum-io') if io_threads else None
        self.cap, self.headroom = cap_bytes, headroom_bytes
        self.coalesce = _flag('FRUSTUM_COALESCE') if coalesce is None else coalesce
        self.protect_layer = _flag('FRUSTUM_PROTECT_LAYER') if protect_layer is None else protect_layer
        self.layer_keys = frozenset()
        self.ledger = IOLedger()
        self.ram_probe, self.ram_floor = ram_probe, ram_floor_bytes
        self.host_shrinks = 0
        self.spill = None  # optional SpillTier
        self.spill_hits = 0
        self.requests = self.hits = self.host_hits = self.disk_reads = self.evictions = 0
        self.upload_seconds = self.wait_seconds = 0.0
        self.upload_bytes = 0
        # v0.8 pipelined uploads
        wanted = _flag('FRUSTUM_ASYNC_UPLOAD', False) if async_upload is None else bool(async_upload)
        self.async_upload = wanted and self.cuda
        self.copy_stream = torch.cuda.Stream(self.device) if self.async_upload else None
        self.stage_pool = (ThreadPoolExecutor(2, thread_name_prefix='frustum-stage')
                           if self.async_upload and io_threads else None)
        self.staged = {}     # key -> Future[_Staged] for host-tier keys about to be uploaded
        self.inspect = None  # optional callable(key, cpu_tensors) -> dict of Entry attributes
        self.staged_uploads = self.blocking_stages = 0

    # Attribute names kept from the v0 cache so existing reports keep working.
    @property
    def used(self):
        return self.gpu.used

    @property
    def budget(self):
        return self.gpu.budget

    def size(self, key):
        return sum(self.catalog.tensors[name].size for name in self.plan[key])

    def resident(self, key):
        """True when key can be served without the drive (either tier, or a read under way)."""
        return key in self.cache or key in self.ram or key in self.pending

    def _read(self, key, queued=False):
        items = [self.catalog.tensors[name] for name in self.plan[key]]
        size = 0
        self.ledger.read_begin(queued)
        try:
            identity = None
            if self.spill is not None:
                identity = self.spill.identity(items)
                copy = self.spill.load(key, identity)
                if copy is not None:
                    result = _Buffers(copy)
                    result.source = 'spill'
                    return result
            if self.coalesce:
                buffers = self.reader.read_group(items)
            else:
                buffers = [self.reader.read(item) for item in items]
            # Bytes actually requested: each underlying buffer once (a coalesced read may span gaps).
            owners = {id(getattr(b, 'obj', b)): getattr(b, 'obj', b) for b in buffers}
            size = sum(len(owner) for owner in owners.values())
            if self.spill is not None:
                self.spill.store(key, buffers, identity)
            return _Buffers(buffers)
        finally:
            self.ledger.read_end(size)

    def _stage(self, key, tensors):
        """Copy CPU tensors into pinned memory (no CUDA calls) and run inspect. Thread-safe."""
        pinned = []
        for tensor in tensors:
            copy = self.t.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
            copy.copy_(tensor)
            pinned.append(copy)
        attrs = self.inspect(key, tensors) if self.inspect is not None else {}
        return _Staged(pinned, attrs)

    def _read_staged(self, key, queued=False):
        """_read, then (v0.8) hand pinned staging to a staging thread. The reader thread is free
        for its next read at once: staging in the reader thread left the drive idle during each
        copy (measured 866 -> 800 MB/s)."""
        buffers = self._read(key, queued)
        if self.async_upload:
            buffers.tensors = [self.t.frombuffer(buffer, dtype=self.t.uint8) for buffer in buffers]
            if self.stage_pool is not None:
                buffers.staged = self.stage_pool.submit(self._stage, key, buffers.tensors)
        return buffers

    def _upload_async(self, pinned):
        """Non-blocking upload on the copy stream; the current stream waits for it on the GPU."""
        t = self.t
        main = t.cuda.current_stream(self.device)
        with t.cuda.stream(self.copy_stream):
            out = [t.empty(p.shape, dtype=p.dtype, device=self.device) for p in pinned]
            for target, source in zip(out, pinned):
                target.copy_(source, non_blocking=True)
            done = t.cuda.Event()
            done.record(self.copy_stream)
        main.wait_event(done)
        for target in out:
            target.record_stream(main)  # freed blocks are not reused before main-stream work ends
        return out

    def _submit(self, key):
        self.ledger.submitted()
        return self.pool.submit(self._read_staged, key, True)

    def new_pass(self):
        """Call once per pass over the layers (one prefill or one decode step)."""
        self.frequency.tick()
        self._guard_ram()

    def begin_layer(self, keys):
        """Declare the experts the layer in progress will use; they are not evicted until the
        next begin_layer. Their use is counted now, so the frequency ranking already knows
        they belong to this pass when the first of them needs room."""
        self.layer_keys = frozenset(keys) if self.protect_layer else frozenset()
        for key in self.layer_keys:
            self.frequency.touch(key)

    def _guard_ram(self):
        """Give RAM back when the machine is short: the host tier only ever shrinks here."""
        if self.ram_probe is None or not self.host.budget:
            return
        available = self.ram_probe()
        if available is None or available >= self.ram_floor:
            return
        target = max(0, self.host.used - (self.ram_floor - available))
        self.host.budget = min(self.host.budget, target)
        while self.host.used > self.host.budget:
            victim = self.host.victim(self.frequency, self.layer_keys)
            if victim is None:
                break
            self.host.remove(victim)
            del self.ram[victim]
        self.host_shrinks += 1

    def prefetch(self, keys, speculative=False):
        """Start background reads for keys that are in neither tier. Never blocks.

        speculative=True marks guesses: they are accounted separately, and a later
        demand prefetch or get() for the same key adopts the read already under way."""
        if self.pool is None:
            return
        for key in keys:
            if key in self.pending:
                if not speculative and key in self.speculative:
                    self._adopt(key)
                continue
            if key not in self.cache and key not in self.ram:
                self.pending[key] = self._submit(key)
                if speculative:
                    self.speculative.add(key)
                    self.ledger.speculative('issued')
            elif (not speculative and self.stage_pool is not None and key not in self.cache
                  and key in self.ram and key not in self.staged):
                self.staged[key] = self.stage_pool.submit(self._stage, key, self.ram[key])

    def _adopt(self, key):
        self.speculative.discard(key)
        self.ledger.speculative('used', self.size(key))

    def discard_pending(self, keep=()):
        """Forget background reads nobody collected, e.g. after an interrupted forward pass.
        Reads for keys in keep stay queued (next-layer prediction issued them early)."""
        keep = set(keep)
        for key in [key for key in self.staged if key not in keep]:
            self.staged.pop(key).cancel()
        for key in [key for key in self.pending if key not in keep]:
            future = self.pending.pop(key)
            guess = key in self.speculative
            self.speculative.discard(key)
            if future.cancel():
                self.ledger.unsubmitted()
                if guess:
                    self.ledger.speculative('cancelled')
            elif guess:
                # Already reading (or read): the drive time is spent. Record it; the bytes are dropped.
                self.ledger.speculative('wasted', self.size(key))

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
        evicted = self.host.admit(key, size, self.frequency, self.layer_keys)
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
            self.ledger.served(key, entry.size, 'gpu')
            return entry
        size = self.size(key)
        if size > self.gpu.budget:
            raise RuntimeError('One expert exceeds cache budget')
        staged = None
        tensors = self.ram.get(key)
        if tensors is not None:
            self.host_hits += 1
            self.host.touch(key)
            self.ledger.served(key, size, 'host')
            future = self.staged.pop(key, None)
            if future is not None:
                staged = future.result()
        else:
            began = time.perf_counter()
            future = self.pending.pop(key, None)
            if key in self.speculative:
                self._adopt(key)
            self.ledger.wait_begin()
            try:
                buffers = self._read_staged(key) if future is None else future.result()
            finally:
                self.ledger.wait_end()
            self.wait_seconds += time.perf_counter() - began
            if getattr(buffers, 'source', 'disk') == 'spill':
                self.spill_hits += 1
                self.ledger.served(key, size, 'spill')
            else:
                self.disk_reads += 1
                self.ledger.served(key, size, 'disk')
            tensors = getattr(buffers, 'tensors', None)
            if tensors is None:
                tensors = [self.t.frombuffer(buffer, dtype=self.t.uint8) for buffer in buffers]
            staged = getattr(buffers, 'staged', None)
            if staged is not None:
                staged = staged.result()
            self._remember(key, tensors, size)
        self.staged.pop(key, None)
        keep = set(protected)
        keep.add(key)
        wide = keep | self.layer_keys
        try:
            victims = self.gpu.make_room(size, self.frequency, wide)
        except RuntimeError:
            # The tier cannot hold a whole layer at once: protect only what the caller named.
            wide = keep
            victims = self.gpu.make_room(size, self.frequency, keep)
        for victim in victims:
            self._evict(victim)
        # The byte budget is a plan; the allocator is the truth. Shed more if it is tight.
        while not self._memory_ok(size):
            victim = self.gpu.victim(self.frequency, wide)
            if victim is None and wide is not keep:
                wide = keep
                victim = self.gpu.victim(self.frequency, keep)
            if victim is None:
                raise RuntimeError('Insufficient free dedicated VRAM; stop before allocating')
            self.gpu.remove(victim)
            self._evict(victim)
        began = time.perf_counter()
        if self.async_upload:
            if staged is None:  # not staged in the background (no pool, or not prefetched)
                staged = self._stage(key, tensors)
                self.blocking_stages += 1
            entry = Entry(self._upload_async(staged.pinned), size)
            for name, value in staged.attrs.items():
                setattr(entry, name, value)
            self.staged_uploads += 1
        else:
            entry = Entry([tensor.to(self.device) for tensor in tensors], size)
        # With async_upload this is the time to queue the copy, not the copy itself.
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
                    host_bytes=self.host.used, host_budget_bytes=self.host.budget, host_policy=self.host.policy,
                    host_shrinks=self.host_shrinks, coalesced_reads=self.coalesce,
                    layer_protection=self.protect_layer, io=self.ledger.snapshot(),
                    spill_hits=self.spill_hits, spill=self.spill.stats() if self.spill is not None else None,
                    async_upload=self.async_upload, staged_uploads=self.staged_uploads,
                    blocking_stages=self.blocking_stages)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.stage_pool is not None:
            self.stage_pool.shutdown(wait=True, cancel_futures=True)
        self.staged.clear()
        if self.cuda:
            self.t.cuda.synchronize(self.device)
        self.pending.clear()
        self.speculative.clear()
        self.cache.clear()
        self.ram.clear()
        if self.spill is not None:
            self.spill.close()
        self.reader.close()
