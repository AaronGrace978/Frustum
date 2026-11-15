"""I/O accounting for the routed-expert path. Pure Python, thread-safe, no torch.

v0.7 instrumentation. It changes no arithmetic and no cache decision; it only
answers the roofline question "is a token's time drive time or GPU time?".

What it measures (all for expert reads issued through TieredExpertCache):
  bytes_read            logical bytes requested from the checkpoint drive (what the code asked
                        for; reads answered by the optional spill tier are not included)
  unique_bytes          bytes of experts read for the first time in this process
  repeat_bytes          bytes of experts that had been read before and were evicted
                        from both tiers: the part a better cache could remove
  reread_amplification  bytes_read / unique_bytes
  busy_seconds          wall time with at least one read in flight (interval union)
  effective_MBps        bytes_read / busy_seconds
  queue_depth           time-weighted mean and max of reads in flight; queued_max is the
                        most reads waiting for a reader thread
  blocked_seconds       wall time the compute thread sat waiting for a read
  overlap_fraction      share of busy_seconds during which the compute thread was NOT
                        blocked, i.e. transfer genuinely overlapped compute
  speculative_*         guessed reads: issued, used, cancelled before starting, wasted

Physical device bytes are not visible from here; os_cache.DeviceCounters reads
them from the OS and bench_e2e.py / frustum_chat.py report both side by side.
"""
import threading
import time

_COUNTERS = ('requests', 'gpu_hits', 'host_hits', 'spill_hits', 'disk_reads', 'bytes_read', 'unique_bytes',
             'repeat_bytes', 'reads_started', 'busy_seconds', 'busy_blocked_seconds',
             'depth_seconds', 'blocked_seconds', 'speculative_issued', 'speculative_used',
             'speculative_cancelled', 'speculative_wasted', 'speculative_wasted_bytes',
             'speculative_used_bytes')


class IOLedger:
    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._lock = threading.Lock()
        self._seen = set()
        self._inflight = 0
        self._queued = 0
        self._blocked = 0
        self._last = clock()
        self._mark = None
        self.created = self._last
        self.max_depth = self.max_queued = 0
        for name in _COUNTERS:
            setattr(self, name, 0)

    # -- time integration: call with the lock held, before any state change --
    def _advance(self):
        now = self._clock()
        dt = now - self._last
        self._last = now
        if dt <= 0:
            return
        if self._inflight:
            self.busy_seconds += dt
            self.depth_seconds += dt * self._inflight
            if self._blocked:
                self.busy_blocked_seconds += dt
        if self._blocked:
            self.blocked_seconds += dt

    # -- reader threads --
    def submitted(self):
        with self._lock:
            self._queued += 1
            self.max_queued = max(self.max_queued, self._queued)

    def unsubmitted(self):
        """A queued read was cancelled before a reader thread picked it up."""
        with self._lock:
            self._queued = max(0, self._queued - 1)

    def read_begin(self, queued=True):
        with self._lock:
            self._advance()
            if queued:
                self._queued = max(0, self._queued - 1)
            self._inflight += 1
            self.reads_started += 1
            self.max_depth = max(self.max_depth, self._inflight)

    def read_end(self, size):
        with self._lock:
            self._advance()
            self._inflight -= 1
            self.bytes_read += size

    # -- compute thread --
    def wait_begin(self):
        with self._lock:
            self._advance()
            self._blocked += 1

    def wait_end(self):
        with self._lock:
            self._advance()
            self._blocked -= 1

    def served(self, key, size, source):
        """One cache request was answered from 'gpu', 'host', 'spill' (fast local copy) or 'disk'."""
        with self._lock:
            self.requests += 1
            if source == 'gpu':
                self.gpu_hits += 1
            elif source == 'host':
                self.host_hits += 1
            elif source == 'spill':
                self.spill_hits += 1
            else:
                self.disk_reads += 1
                if key in self._seen:
                    self.repeat_bytes += size
                else:
                    self._seen.add(key)
                    self.unique_bytes += size

    def speculative(self, event, size=0):
        """event: issued | used | cancelled | wasted."""
        with self._lock:
            setattr(self, 'speculative_' + event, getattr(self, 'speculative_' + event) + 1)
            if event in ('used', 'wasted'):
                name = f'speculative_{event}_bytes'
                setattr(self, name, getattr(self, name) + size)

    # -- reporting --
    def totals(self):
        with self._lock:
            self._advance()
            values = {name: getattr(self, name) for name in _COUNTERS}
            values['wall_seconds'] = self._last - self.created
            values['max_depth'], values['max_queued'] = self.max_depth, self.max_queued
            return values

    @staticmethod
    def derive(v):
        """Add the ratios to a totals() or step() dict. Ratios are None when undefined."""
        ratio = lambda a, b: (a / b) if b else None
        out = dict(v)
        out['effective_MBps'] = ratio(v['bytes_read'] / 1e6, v['busy_seconds'])
        out['drive_utilization'] = ratio(v['busy_seconds'], v.get('wall_seconds'))
        out['mean_queue_depth'] = ratio(v['depth_seconds'], v['busy_seconds'])
        out['overlap_fraction'] = ratio(v['busy_seconds'] - v['busy_blocked_seconds'], v['busy_seconds'])
        out['reread_amplification'] = ratio(v['unique_bytes'] + v['repeat_bytes'], v['unique_bytes'])
        for tier in ('gpu_hits', 'host_hits', 'spill_hits', 'disk_reads'):
            out[tier.replace('_hits', '').replace('_reads', '') + '_rate'] = ratio(v[tier], v['requests'])
        return out

    def snapshot(self):
        return self.derive(self.totals())

    def step(self):
        """Counters accumulated since the previous step() call (or creation), with ratios.

        max_depth / max_queued are process-wide maxima, not per step."""
        now = self.totals()
        before = self._mark or dict.fromkeys(now, 0)
        self._mark = now
        delta = {k: now[k] - before[k] for k in now if k not in ('max_depth', 'max_queued')}
        delta['max_depth'], delta['max_queued'] = now['max_depth'], now['max_queued']
        return self.derive(delta)
