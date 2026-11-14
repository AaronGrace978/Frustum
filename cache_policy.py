"""Cache bookkeeping shared by the live expert cache and the replay tool.

Pure Python: no torch, no file access. A Tier tracks which keys are resident in
one memory tier and chooses eviction victims. Frequency keeps a decayed use
count for every key ever requested, resident or not, so a tier can keep the
experts that recur across tokens instead of the ones that were merely recent.

Why not plain LRU: one decode step walks layers 0..N-1 in order and touches a
different key set in each. When a tier is smaller than one step's working set,
LRU has always evicted layer k's experts by the time the next token reaches
layer k, so its hit rate collapses. Frequency ranking does not have that
failure mode. Run `python decode_cache_replay.py` on a recorded profile to
compare both on real routing before trusting either.
"""
from collections import OrderedDict
import math


class Frequency:
    """Decayed count of the passes (decode steps) in which each key was requested.

    Call tick() when a new pass over the layers begins. Counting per pass, not
    per request, means keys with the same history tie exactly, and the tier can
    then break the tie by recency. rank(key) = log2(decayed count) +
    last_pass / half_life; the second term cancels the decay, so ranks stay
    comparable without recomputation as passes go by.
    """
    def __init__(self, half_life=8):
        if half_life <= 0:
            raise ValueError('half_life must be positive')
        self.half_life = float(half_life)
        self.clock = 0
        self._count = {}
        self._rank = {}

    def tick(self):
        self.clock += 1

    def touch(self, key):
        previous = self._count.get(key)
        if previous is None:
            value = 1.0
        elif previous[1] == self.clock:
            return previous[0]
        else:
            value = previous[0] * 0.5 ** ((self.clock - previous[1]) / self.half_life) + 1.0
        self._count[key] = (value, self.clock)
        self._rank[key] = math.log2(value) + self.clock / self.half_life
        return value

    def rank(self, key):
        return self._rank.get(key, -math.inf)

    def count(self, key):
        item = self._count.get(key)
        if item is None:
            return 0.0
        return item[0] * 0.5 ** ((self.clock - item[1]) / self.half_life)


class Tier:
    """Byte-budgeted set of resident keys with a pluggable victim policy.

    policy 'lru': evict the least recently used unprotected key.
    policy 'lfu': evict the unprotected key with the lowest Frequency rank;
                  among equals evict the most recently used, because under
                  layer-cyclic access that key is needed again last.
    """
    def __init__(self, budget, policy='lfu', pinned=None):
        if budget < 0:
            raise ValueError('budget must be nonnegative')
        if policy not in ('lru', 'lfu'):
            raise ValueError('policy must be lru or lfu')
        self.budget = int(budget)
        self.policy = policy
        self.pinned = pinned or (lambda key: False)
        self.used = 0
        self.order = OrderedDict()  # key -> size, oldest first

    def __contains__(self, key):
        return key in self.order

    def __len__(self):
        return len(self.order)

    def touch(self, key):
        self.order.move_to_end(key)

    def add(self, key, size):
        if key in self.order:
            raise KeyError('Key already resident')
        self.order[key] = size
        self.used += size

    def remove(self, key):
        self.used -= self.order.pop(key)

    def victim(self, frequency, protected=()):
        if self.policy == 'lru':
            for key in self.order:
                if key not in protected and not self.pinned(key):
                    return key
            return None
        best, best_rank = None, math.inf
        for key in reversed(self.order):
            if key in protected or self.pinned(key):
                continue
            rank = frequency.rank(key)
            if rank < best_rank:
                best, best_rank = key, rank
        return best

    def make_room(self, size, frequency, protected=()):
        """Evict until `size` more bytes fit. Returns the evicted keys.

        Raises RuntimeError when only protected or pinned keys remain.
        """
        evicted = []
        while self.used + size > self.budget:
            key = self.victim(frequency, protected)
            if key is None:
                raise RuntimeError('Cache tier too small for the keys in use')
            self.remove(key)
            evicted.append(key)
        return evicted

    def admit(self, key, size, frequency):
        """Decide whether an optional copy of `key` is worth keeping here.

        Returns the list of keys to evict (possibly empty) when admitted, or
        None when the newcomer ranks below what it would displace.
        """
        if size > self.budget or key in self.order:
            return None
        evicted, freed = [], 0
        rank = frequency.rank(key)
        skip = set()
        while self.used - freed + size > self.budget:
            victim = self.victim(frequency, skip)
            if victim is None:
                return None
            if self.policy == 'lfu' and frequency.rank(victim) > rank:
                return None
            skip.add(victim)
            evicted.append(victim)
            freed += self.order[victim]
        for victim in evicted:
            self.remove(victim)
        return evicted


def simulate(steps, sizes, gpu_bytes, host_bytes=0, policy='lfu', half_life=8, pinned=None,
             host_policy='lru'):
    """Replay recorded requests through a GPU tier and an optional host tier.

    steps: list of decode steps, each an ordered list of keys.
    sizes: key -> bytes.
    policy applies to the GPU tier, host_policy to the host tier.
    Returns one record per step with gpu hits, host hits and bytes read from
    disk. The host tier keeps a copy of whatever it admits, GPU-resident or
    not, matching TieredExpertCache.
    """
    frequency = Frequency(half_life)
    gpu = Tier(gpu_bytes, policy, pinned)
    host = Tier(host_bytes, host_policy)
    results = []
    for step in steps:
        frequency.tick()
        record = dict(requests=len(step), gpu_hits=0, host_hits=0, disk_reads=0, disk_bytes=0)
        for key in step:
            size = sizes[key]
            if size > gpu.budget:
                raise ValueError('An expert exceeds the GPU tier budget')
            frequency.touch(key)
            if key in gpu:
                gpu.touch(key)
                record['gpu_hits'] += 1
                continue
            if key in host:
                host.touch(key)
                record['host_hits'] += 1
            else:
                record['disk_reads'] += 1
                record['disk_bytes'] += size
                if host.budget and not gpu.pinned(key) and host.admit(key, size, frequency) is not None:
                    host.add(key, size)
            gpu.make_room(size, frequency, (key,))
            gpu.add(key, size)
        results.append(record)
    return results
