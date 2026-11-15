"""v0.7 tests: I/O ledger, coalesced reads, layer protection, speculative-read bookkeeping,
thread safety of the expert cache, RAM guard, spill tier, prefix-reuse equivalence, and
reference checks for routing, FP4/FP8 decode and residual/norm arithmetic.

    python selftest_v07.py

No GPU and no model weights are needed. Two groups:
  * storage / cache / generation tests need only the standard library and NumPy. If
    PyTorch is not importable the cache tests use a small NumPy stand-in for the four
    torch calls the cache makes, so they run anywhere.
  * ReferenceArithmeticTests need PyTorch (CPU is enough) and are skipped without it.
"""
import json
import os
from pathlib import Path
import random
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from cache_policy import Frequency, Tier, simulate
from frustum_loader import RangeReader
from generation import drive
from io_ledger import IOLedger
from selftest import write_checkpoint
from spill_tier import SpillTier

try:
    import torch as real_torch
except ImportError:
    real_torch = None


class _Array:
    """Stand-in for a CPU uint8 tensor: the cache only moves it and tests read it back."""
    def __init__(self, data):
        self.data = data

    def to(self, device):
        return _Array(self.data.copy())

    def numpy(self):
        return self.data


STANDIN = SimpleNamespace(uint8=np.uint8, device=lambda name: SimpleNamespace(type=str(name).split(':')[0]),
                          frombuffer=lambda buffer, dtype: _Array(np.frombuffer(buffer, dtype=dtype)))
TORCH = real_torch or STANDIN


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class LedgerTests(unittest.TestCase):
    def test_busy_time_is_an_interval_union_and_depth_is_time_weighted(self):
        clock = Clock()
        ledger = IOLedger(clock)
        ledger.submitted(); ledger.submitted()
        clock.now = 1.0; ledger.read_begin()          # A starts
        clock.now = 2.0; ledger.read_begin()          # B starts, A still running
        clock.now = 4.0; ledger.read_end(100)         # A ends
        clock.now = 5.0; ledger.read_end(300)         # B ends
        clock.now = 9.0                               # idle
        ledger.read_begin(queued=False); clock.now = 10.0; ledger.read_end(100)
        total = ledger.snapshot()
        self.assertAlmostEqual(total['busy_seconds'], 5.0)            # [1,5] and [9,10], not 3+3+1
        self.assertAlmostEqual(total['depth_seconds'], 1 + 2 * 2 + 1 + 1)
        self.assertAlmostEqual(total['mean_queue_depth'], 7 / 5)
        self.assertEqual((total['max_depth'], total['max_queued'], total['bytes_read']), (2, 2, 500))
        self.assertAlmostEqual(total['effective_MBps'], 500 / 1e6 / 5.0)
        self.assertAlmostEqual(total['drive_utilization'], 0.5)

    def test_overlap_is_drive_time_while_compute_was_not_waiting(self):
        clock = Clock()
        ledger = IOLedger(clock)
        ledger.read_begin(queued=False)               # read runs over [0, 10]
        clock.now = 4.0; ledger.wait_begin()          # compute blocks over [4, 10]
        clock.now = 10.0; ledger.read_end(1); ledger.wait_end()
        total = ledger.snapshot()
        self.assertAlmostEqual(total['blocked_seconds'], 6.0)
        self.assertAlmostEqual(total['busy_blocked_seconds'], 6.0)
        self.assertAlmostEqual(total['overlap_fraction'], 0.4)

    def test_unique_and_repeated_bytes_and_step_deltas(self):
        ledger = IOLedger(Clock())
        ledger.served('a', 10, 'disk'); ledger.served('b', 10, 'disk'); ledger.served('a', 10, 'gpu')
        first = ledger.step()
        self.assertEqual((first['unique_bytes'], first['repeat_bytes'], first['gpu_hits']), (20, 0, 1))
        ledger.served('a', 10, 'disk'); ledger.served('c', 10, 'host'); ledger.served('d', 10, 'spill')
        second = ledger.step()
        self.assertEqual((second['unique_bytes'], second['repeat_bytes']), (0, 10))
        self.assertEqual((second['requests'], second['host_hits'], second['spill_hits'], second['disk_reads']),
                         (3, 1, 1, 1))
        self.assertAlmostEqual(ledger.snapshot()['reread_amplification'], 1.5)
        self.assertIsNone(IOLedger(Clock()).snapshot()['effective_MBps'])

    def test_threaded_updates_balance(self):
        ledger = IOLedger()
        def work():
            for _ in range(500):
                ledger.submitted(); ledger.read_begin(); ledger.read_end(3)
        threads = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in threads]; [t.join() for t in threads]
        self.assertEqual((ledger._inflight, ledger._queued, ledger.bytes_read, ledger.reads_started),
                         (0, 0, 12000, 4000))


class Fixture(unittest.TestCase):
    layers, experts = 4, 8

    def setUp(self):
        from expert_cache import TieredExpertCache
        self.cls = TieredExpertCache
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'model'
        self.root.mkdir()
        self.catalog = write_checkpoint(self.root, layers=self.layers, experts=self.experts)
        reader = RangeReader()
        self.truth = {}
        for layer in range(self.layers):
            for expert in list(range(self.experts)) + ['shared']:
                self.truth[(layer, expert)] = [bytes(reader.read(self.catalog.tensors[n]))
                                               for n in self.names(layer, expert)]
        reader.close()

    @staticmethod
    def names(layer, expert):
        prefix = f'layers.{layer}.ffn.' + ('shared_experts' if expert == 'shared' else f'experts.{expert}')
        return [f'{prefix}.{w}.{kind}' for w in ('w1', 'w2', 'w3') for kind in ('weight', 'scale')]

    def make(self, gpu_keys, host_keys=0, **options):
        cache = self.cls(TORCH, self.catalog, 'cpu', gpu_keys * 384, host_keys * 384, **options)
        self.addCleanup(cache.close)
        for key in self.truth:
            cache.plan[key] = self.names(*key)
        return cache

    def check(self, cache, key, entry):
        self.assertEqual([bytes(np.asarray(t.numpy()).tobytes()) for t in entry.tensors], self.truth[key], key)
        self.assertLessEqual(cache.gpu.used, cache.gpu.budget)
        self.assertLessEqual(cache.host.used, max(cache.host.budget, 0))
        self.assertEqual(set(cache.cache), set(cache.gpu.order))
        self.assertEqual(set(cache.ram), set(cache.host.order))


class ReadGroupTests(Fixture):
    def test_one_read_same_bytes(self):
        calls = []
        reader = RangeReader()
        original = reader.readinto
        reader.readinto = lambda item, dest, start=0: (calls.append(item.size), original(item, dest, start))[1]
        items = [self.catalog.tensors[n] for n in self.names(2, 5)]
        buffers = reader.read_group(items[::-1])
        self.assertEqual([bytes(b) for b in buffers], self.truth[(2, 5)][::-1])
        self.assertEqual(calls, [6 * 64])           # six adjacent tensors, one positioned read
        self.assertEqual(reader.bytes_read, 6 * 64)
        reader.close()

    def test_gaps_split_reads_and_mmap_mode_is_unchanged(self):
        scattered = [self.catalog.tensors[n] for n in
                     ('head.weight', self.names(0, 0)[0], 'layers.3.ffn.gate.bias', self.names(0, 0)[1])]
        reader = RangeReader()
        expected = [bytes(reader.read(item)) for item in scattered]
        for gap, mode in ((0, 'direct'), (1 << 20, 'direct'), (0, 'mmap')):
            other = RangeReader(mode=mode)
            self.assertEqual([bytes(b) for b in other.read_group(scattered, max_gap=gap)], expected)
            if mode == 'direct' and gap == 0:
                self.assertEqual(other.bytes_read, sum(i.size for i in scattered))   # no gap bytes read
            other.close()
        reader.close()

    def test_coalescing_flag_changes_reads_not_bytes(self):
        for coalesce in (True, False):
            cache = self.make(4, coalesce=coalesce)
            self.check(cache, (1, 2), cache.get((1, 2)))
            self.assertEqual(cache.reader.bytes_read, 384)
            self.assertEqual(cache.ledger.snapshot()['bytes_read'], 384)


class LayerProtectionTests(Fixture):
    def run_layer(self, cache, protect):
        a, b, c = (0, 0), (0, 1), (0, 2)
        for key in (a, b):
            cache.get(key)                              # a is now the least recently used
        cache.new_pass()
        if protect:
            cache.begin_layer([c, a])
        for key in (c, a):                              # same layer needs c, then a
            self.check(cache, key, cache.get(key, (key,)))
        return cache.disk_reads

    def test_same_layer_expert_is_not_evicted_before_use(self):
        old = self.run_layer(self.make(2, gpu_policy='lru', io_threads=0, protect_layer=False), False)
        new = self.run_layer(self.make(2, gpu_policy='lru', io_threads=0, protect_layer=True), True)
        self.assertEqual((old, new), (4, 3))            # v0.6 re-reads a; v0.7 evicts b instead

    def test_tier_smaller_than_a_layer_falls_back(self):
        cache = self.make(1, protect_layer=True)
        keys = [(1, 0), (1, 1), (1, 2)]
        cache.begin_layer(keys)
        for key in keys:
            self.check(cache, key, cache.get(key, (key,)))
        self.assertEqual(cache.evictions, 2)

    def test_host_copy_of_layer_expert_survives_admission(self):
        a, b = (2, 0), (2, 1)
        extra = {}
        for protect in (False, True):
            cache = self.make(1, host_keys=2, io_threads=0, protect_layer=protect)
            cache.get(a); cache.new_pass()              # a: on GPU and in RAM
            cache.get((3, 7))                           # a leaves the GPU; RAM holds a (oldest) and (3, 7)
            cache.new_pass(); cache.begin_layer([b, a])
            reads = cache.disk_reads
            cache.get(b, (b,))                          # RAM must make room: not by dropping a
            self.check(cache, a, cache.get(a, (a,)))
            extra[protect] = (cache.disk_reads - reads, cache.host_hits)
        self.assertEqual(extra, {False: (2, 0), True: (1, 1)})

    def test_policy_protected_admission_and_replay_accounting(self):
        frequency, tier = Frequency(), Tier(20, 'lru')
        tier.add('x', 10); tier.add('y', 10)
        self.assertEqual(tier.admit('z', 10, frequency, protected=('x',)), ['y'])
        self.assertIsNone(Tier(10, 'lru').admit('z', 20, frequency))
        steps = [['L0.ffn.a', 'L0.ffn.b'], ['L0.ffn.c', 'L0.ffn.a']]
        sizes = dict.fromkeys(['L0.ffn.a', 'L0.ffn.b', 'L0.ffn.c'], 10)
        group = lambda key: key.split('.ffn.')[0]
        plain = simulate(steps, sizes, 20, policy='lru')
        protected = simulate(steps, sizes, 20, policy='lru', group=group)
        self.assertEqual((plain[1]['disk_reads'], protected[1]['disk_reads']), (2, 1))
        for record in plain + protected:
            self.assertEqual(record['unique_disk_bytes'] + record['repeat_disk_bytes'], record['disk_bytes'])
        self.assertEqual(plain[1]['repeat_disk_bytes'], 10)


class SpeculativeTests(Fixture):
    def test_wrong_guess_is_cancelled_or_counted_and_right_guess_is_adopted(self):
        cache = self.make(4, io_threads=1)
        gate = threading.Event()
        original = cache.reader.read_group
        cache.reader.read_group = lambda items, **kw: (gate.wait(5), original(items, **kw))[1]
        guesses = [(1, e) for e in range(4)]
        cache.prefetch(guesses, speculative=True)        # the single reader starts guess 0, queues the rest
        self.assertEqual(cache.speculative, set(guesses))
        cache.begin_layer([(1, 1), (1, 6)])
        cache.discard_pending(keep=[(1, 1), (1, 6)])     # the router chose 1 and 6
        gate.set()
        for key in ((1, 1), (1, 6)):
            self.check(cache, key, cache.get(key, (key,)))
        totals = cache.ledger.snapshot()
        self.assertEqual(totals['speculative_issued'], 4)
        self.assertEqual(totals['speculative_used'], 1)
        self.assertEqual(totals['speculative_cancelled'] + totals['speculative_wasted'], 3)
        self.assertLessEqual(totals['speculative_wasted'], 1)     # at most the one already reading
        self.assertFalse(cache.pending)
        self.assertFalse(cache.speculative)
        cache.pool.shutdown(wait=True)
        self.assertEqual((cache.ledger._inflight, cache.ledger._queued), (0, 0))

    def test_guess_for_resident_expert_reads_nothing(self):
        cache = self.make(4)
        cache.get((0, 0))
        before = cache.reader.bytes_read
        cache.prefetch([(0, 0)], speculative=True)
        self.assertFalse(cache.pending)
        self.assertEqual(cache.reader.bytes_read, before)
        self.assertTrue(cache.resident((0, 0)))
        self.assertFalse(cache.resident((0, 1)))


class AsyncSafetyTests(Fixture):
    def test_random_prefetch_discard_get_always_returns_exact_bytes(self):
        keys = sorted((k for k in self.truth if k[1] != 'shared'), key=str)
        for seed in range(6):
            rng = random.Random(seed)
            cache = self.make(rng.choice((1, 2, 5)), host_keys=rng.choice((0, 3, 8)), io_threads=rng.choice((1, 4)),
                              gpu_policy=rng.choice(('lru', 'lfu')), protect_layer=rng.random() < 0.7,
                              coalesce=rng.random() < 0.7)
            for step in range(250):
                layer = rng.randrange(self.layers)
                chosen = rng.sample([k for k in keys if k[0] == layer], 3)
                if layer == 0:
                    cache.new_pass()
                cache.begin_layer(chosen)
                cache.discard_pending(keep=chosen)
                cache.prefetch(chosen)
                for key in chosen:
                    self.check(cache, key, cache.get(key, (key,)))
                guess = rng.sample(keys, 2)
                cache.prefetch(guess, speculative=True)
                self.assertTrue(cache.speculative <= set(cache.pending))
            cache.discard_pending()
            cache.pool.shutdown(wait=True)
            totals = cache.ledger.snapshot()
            self.assertEqual((cache.ledger._inflight, cache.ledger._queued, cache.ledger._blocked), (0, 0, 0))
            self.assertEqual(totals['requests'], 750)
            self.assertEqual(totals['gpu_hits'] + totals['host_hits'] + totals['disk_reads'], 750)
            self.assertEqual(totals['speculative_issued'],
                             totals['speculative_used'] + totals['speculative_cancelled'] + totals['speculative_wasted'])

    def test_failed_read_surfaces_and_leaves_cache_usable(self):
        cache = self.make(2, io_threads=2)
        original = cache.reader.read_group
        def broken(items, **kw):
            raise OSError('drive unplugged')
        cache.reader.read_group = broken
        cache.prefetch([(0, 0)])
        with self.assertRaises(OSError):
            cache.get((0, 0))
        self.assertNotIn((0, 0), cache.cache)
        self.assertEqual(cache.ledger._blocked, 0)
        cache.reader.read_group = original
        self.check(cache, (0, 0), cache.get((0, 0)))


class RamGuardTests(Fixture):
    def test_host_tier_shrinks_when_ram_is_short_and_never_grows(self):
        free = [10_000]
        cache = self.make(1, host_keys=4, io_threads=0, ram_probe=lambda: free[0], ram_floor_bytes=1000)
        for expert in range(4):
            cache.get((0, expert))
        self.assertEqual(len(cache.ram), 4)
        cache.new_pass()
        self.assertEqual((cache.host.budget, cache.host_shrinks), (4 * 384, 0))
        free[0] = 1000 - 2 * 384                        # 768 bytes under the floor
        cache.new_pass()
        self.assertEqual((len(cache.ram), cache.host.budget, cache.host_shrinks), (2, 2 * 384, 1))
        free[0] = 10_000
        cache.new_pass()
        self.assertEqual(cache.host.budget, 2 * 384)
        self.check(cache, (0, 0), cache.get((0, 0)))


class SpillTests(Fixture):
    def spill(self, budget=10 ** 6):
        return SpillTier(Path(self.tmp.name) / 'spill', budget, 'test-checkpoint')

    def test_second_session_reads_the_copy_and_bytes_are_exact(self):
        first = self.make(4)
        first.spill = self.spill()
        for key in ((0, 1), (2, 3)):
            self.check(first, key, first.get(key))
        first.close()
        self.assertEqual(first.spill.stats()['stores'], 2)
        second = self.make(4)
        second.spill = self.spill()
        for key in ((0, 1), (2, 3), (1, 1)):
            self.check(second, key, second.get(key))
        self.assertEqual((second.spill_hits, second.disk_reads), (2, 1))
        self.assertEqual(second.reader.bytes_read, 384)            # only the expert that had no copy
        second.close()
        on_disk = sum(p.stat().st_size for p in second.spill.root.rglob('*.bin'))
        self.assertEqual(second.spill.stats()['used_bytes'], on_disk)
        self.assertEqual(len(list(second.spill.root.rglob('*.bin'))), 3)

    def test_corrupt_stale_and_truncated_copies_are_ignored(self):
        first = self.make(4)
        first.spill = self.spill()
        for expert in range(3):
            first.get((0, expert))
        first.close()
        files = {int(p.stem): p for p in (first.spill.root / '0').glob('*.bin')}
        blob = bytearray(files[0].read_bytes()); blob[-1] ^= 1; files[0].write_bytes(blob)      # flipped bit
        files[1].write_bytes(files[1].read_bytes()[:-5])                                         # truncated
        second = self.make(4)
        second.spill = self.spill()
        for expert in range(3):
            self.check(second, (0, expert), second.get((0, expert)))
        self.assertEqual((second.spill_hits, second.disk_reads, second.spill.stats()['rejected']), (1, 2, 2))
        second.close()
        shard = next(self.root.glob('*.safetensors'))
        os.utime(shard, ns=(1, 1))                                                               # checkpoint changed
        third = self.make(4)
        third.spill = self.spill()
        self.check(third, (0, 2), third.get((0, 2)))
        self.assertEqual((third.spill_hits, third.disk_reads), (0, 1))

    def test_budget_stops_admission(self):
        cache = self.make(4)
        cache.spill = self.spill(budget=384 + 1024 + 100)
        for expert in range(3):
            self.check(cache, (1, expert), cache.get((1, expert)))
        cache.close()
        stats = cache.spill.stats()
        self.assertEqual((stats['stores'], stats['skipped_full']), (1, 2))
        self.assertLessEqual(stats['used_bytes'], stats['budget_bytes'])


class PrefixReuseTests(unittest.TestCase):
    """Reuse must be invisible: same tokens as rebuilding the whole conversation every turn."""
    class Engine:
        def __init__(self):
            self.state, self.prefills, self.fed_tokens = [], 0, 0

        def run(self, tokens, phase, want_logits):
            if phase == 'prefill':
                self.state, self.prefills = list(tokens), self.prefills + 1
            else:
                assert len(tokens) == 1, 'incremental steps feed one token'
                self.state += tokens
            self.fed_tokens += len(tokens)
            return (tuple(self.state) if want_logits else None), {}

    @staticmethod
    def pick(state):
        return hash(state) % 11        # a function of the entire processed sequence; 0 acts as EOS

    def converse(self, turns, reuse):
        engine, fed, outputs = self.Engine(), [], []
        for ids, limit in turns:
            out, _, reused, _ = drive(fed, list(ids), limit, 0, self.pick, engine.run, reuse)
            self.assertEqual(fed, engine.state)                       # bookkeeping matches the engine
            self.assertTrue(reused == 0 or reuse)
            outputs.append((out, reused))
        return outputs, engine

    def test_reuse_matches_rebuild_on_random_conversations(self):
        rng = random.Random(7)
        for _ in range(200):
            turns, history = [], [rng.randrange(1, 50) for _ in range(rng.randrange(1, 6))]
            for _ in range(rng.randrange(1, 5)):
                turns.append((list(history), rng.randrange(1, 6)))
                out, *_ = drive([], list(history), turns[-1][1], 0, self.pick, self.Engine().run, False)
                mode = rng.random()
                if mode < 0.6:      # normal follow-up: previous prompt + reply + new user tokens
                    history = history + out + [rng.randrange(1, 50) for _ in range(rng.randrange(1, 4))]
                elif mode < 0.8:    # edited history: must not reuse
                    history = [rng.randrange(1, 50)] + history[1:] + [5]
                else:               # identical or shorter prompt: nothing new to extend
                    history = history[:rng.randrange(1, len(history) + 1)]
            with_reuse, engine = self.converse(turns, True)
            without, rebuilt = self.converse(turns, False)
            self.assertEqual([o for o, _ in with_reuse], [o for o, _ in without])
            self.assertLessEqual(engine.fed_tokens, rebuilt.fed_tokens)

    def test_reuse_requires_a_strict_exact_prefix(self):
        engine, fed = self.Engine(), []
        drive(fed, [3, 4, 5], 2, 0, self.pick, engine.run, True)
        covered = list(fed)
        for ids, expect in ((covered + [9], len(covered)), (covered, 0), (covered[:-1], 0),
                            ([8] + covered[1:] + [9], 0)):
            copy, other = list(covered), self.Engine()
            other.state = list(covered)
            _, _, reused, _ = drive(copy, ids, 1, 0, self.pick, other.run, True)
            self.assertEqual(reused, expect, ids)

    def test_interrupted_step_forces_rebuild(self):
        engine, fed = self.Engine(), []
        drive(fed, [3, 4], 2, 0, self.pick, engine.run, True)
        def failing(tokens, phase, want_logits):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            drive(fed, fed + [7], 2, 0, self.pick, failing, True)
        self.assertEqual(fed, [])


class RooflineTests(Fixture):
    def test_verdict_follows_measured_drive_utilisation(self):
        import roofline
        self.assertEqual(roofline.arithmetic()['all_miss_bytes_per_token'], 40 * 6 * 18_800_640)
        def evidence(busy, wall, unique, repeat):
            io = dict.fromkeys(('requests', 'gpu_hits', 'host_hits', 'spill_hits', 'disk_reads', 'depth_seconds',
                                'blocked_seconds', 'busy_blocked_seconds', 'speculative_issued', 'speculative_used',
                                'speculative_wasted', 'speculative_wasted_bytes'), 0)
            io.update(busy_seconds=busy, bytes_read=unique + repeat, unique_bytes=unique, repeat_bytes=repeat,
                      requests=4, disk_reads=4)
            return dict(token_timings=[dict(phase='prefill', total_seconds=9.0),
                                       dict(phase='decode', total_seconds=wall, io=io)])
        bound = roofline.measured(evidence(7.5, 8.0, 1e9, 3e9))
        self.assertTrue(bound['verdict'].startswith('Storage-bound'))
        self.assertIn('75% of the bytes were re-reads', bound['verdict'])
        self.assertAlmostEqual(bound['effective_MBps'], 4e9 / 1e6 / 7.5)
        self.assertTrue(roofline.measured(evidence(2.0, 8.0, 1e9, 0))['verdict'].startswith('Compute-bound'))
        self.assertTrue(roofline.measured(evidence(5.6, 8.0, 1e9, 0))['verdict'].startswith('Mixed'))
        self.assertIsNone(roofline.measured(dict(token_timings=[dict(phase='decode', total_seconds=1.0)])))

    def test_live_ledger_step_feeds_roofline(self):
        import roofline
        cache = self.make(2, io_threads=2, gpu_policy='lru')
        cache.ledger.step()
        for key in ((0, 0), (0, 1), (0, 2), (0, 0)):       # the last one was evicted: a re-read
            cache.get(key)
        step = cache.ledger.step()
        result = roofline.measured(dict(token_timings=[dict(phase='decode', total_seconds=1.0, io=step)]))
        self.assertAlmostEqual(result['first_time_GB_per_token'] * 1e9, 3 * 384)
        self.assertAlmostEqual(result['reread_GB_per_token'] * 1e9, 384)
        self.assertEqual(result['disk_rate'], 1.0)


@unittest.skipIf(real_torch is None, 'PyTorch not installed: reference arithmetic checks skipped')
class ReferenceArithmeticTests(unittest.TestCase):
    """CPU PyTorch against independent NumPy float64 references. Needs no GPU or weights."""
    def test_fp4_decode_exhaustive_and_fast_path_bit_exact(self):
        from expert_correctness import cpu_decode, gpu_decode, gpu_decode_reference
        packed = np.arange(256, dtype=np.uint8).reshape(16, 16)
        scale = np.arange(119, 135, dtype=np.uint8).reshape(16, 1)
        fast = gpu_decode(real_torch.from_numpy(packed), real_torch.from_numpy(scale))
        np.testing.assert_array_equal(fast.numpy(), cpu_decode(packed, scale))
        rng = np.random.default_rng(3)
        p = real_torch.from_numpy(rng.integers(0, 256, (64, 96), dtype=np.uint8))
        s = real_torch.from_numpy(rng.integers(100, 150, (64, 6), dtype=np.uint8))
        a, b = gpu_decode(p, s), gpu_decode_reference(p, s)
        self.assertTrue(real_torch.equal(a.view(real_torch.int32), b.view(real_torch.int32)))
        with self.assertRaises(ValueError):
            gpu_decode(p, real_torch.full_like(s, 255))

    def test_fp8_decode_all_finite_codes_with_block_scales(self):
        from streamed_moe import fp8_decode_cpu
        raw = np.arange(256, dtype=np.uint8).repeat(16).reshape(64, 64)
        raw[(raw & 127) == 127] = 0
        scale = np.random.default_rng(4).integers(110, 140, (2, 2), dtype=np.uint8)
        value = real_torch.from_numpy(raw.copy()).view(real_torch.float8_e4m3fn).float()
        value.view(2, 32, 2, 32).mul_(real_torch.exp2(real_torch.from_numpy(scale).float() - 127)[:, None, :, None])
        np.testing.assert_array_equal(value.numpy().astype(np.float64), fp8_decode_cpu(raw, scale))
        with self.assertRaises(ValueError):
            fp8_decode_cpu(np.full((32, 32), 0x7f, dtype=np.uint8), scale[:1, :1])

    def test_routing_matches_float64_reference(self):
        from streamed_moe import route, route_reference
        cfg = dict(num_experts_per_tok=6, norm_topk_prob=True, routed_scaling_factor=1.5)
        generator = real_torch.Generator().manual_seed(5)
        for _ in range(20):
            x = real_torch.randn(3, 64, generator=generator).to(real_torch.bfloat16)
            weight = real_torch.randn(384, 64, generator=generator).to(real_torch.bfloat16)
            bias = real_torch.randn(384, generator=generator) * 0.1
            weights, ids = route(x, weight, bias, cfg)
            ref_weights, ref_ids = route_reference(x, weight, bias, cfg)
            self.assertTrue(real_torch.equal(ids, ref_ids))
            real_torch.testing.assert_close(weights, ref_weights, rtol=2e-5, atol=2e-6)
            real_torch.testing.assert_close(weights.sum(-1), real_torch.full((3,), 1.5), rtol=1e-5, atol=1e-5)

    def test_rmsnorm_and_residual_mixing_match_float64(self):
        from block_primitives import hc_mixes, hc_post, hc_pre, rms_norm
        generator = real_torch.Generator().manual_seed(6)
        hc, dim, eps = 4, 48, 1e-6
        x = real_torch.randn(1, 3, hc, dim, generator=generator)
        fn = real_torch.randn(2 * hc + hc * hc, hc * dim, generator=generator) * 0.1
        scale = real_torch.rand(3, generator=generator) + 0.5
        base = real_torch.randn(2 * hc + hc * hc, generator=generator) * 0.1
        pre, post, comb = hc_mixes(x, fn, scale, base, hc=hc, iters=20, eps=eps, norm_eps=1e-20)
        flat = x.double().numpy().reshape(1, 3, -1)
        mix = flat @ fn.double().numpy().T / np.sqrt((flat ** 2).mean(-1, keepdims=True) + 1e-20)
        sc, bs = scale.double().numpy(), base.double().numpy()
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        pr = sigmoid(mix[..., :hc] * sc[0] + bs[:hc]) + eps
        po = 2 * sigmoid(mix[..., hc:2 * hc] * sc[1] + bs[hc:2 * hc])
        co = (mix[..., 2 * hc:] * sc[2] + bs[2 * hc:]).reshape(1, 3, hc, hc)
        co = np.exp(co - co.max(-1, keepdims=True)); co = co / co.sum(-1, keepdims=True) + eps
        co /= co.sum(-2, keepdims=True) + eps
        for _ in range(19):
            co /= co.sum(-1, keepdims=True) + eps; co /= co.sum(-2, keepdims=True) + eps
        for actual, expected in ((pre, pr), (post, po), (comb, co)):
            np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-4, atol=2e-5)
        arr = x.double().numpy()
        np.testing.assert_allclose(hc_pre(x, pre).numpy(), (pr[..., None] * arr).sum(-2), rtol=1e-4, atol=1e-5)
        part = real_torch.randn(1, 3, dim, generator=generator)
        expected = po[..., None] * part.double().numpy()[..., None, :] + (co[..., None] * arr[..., None, :]).sum(-3)
        np.testing.assert_allclose(hc_post(part, x, post, comb).numpy(), expected, rtol=1e-4, atol=1e-5)
        weight = real_torch.rand(dim, generator=generator) + 0.5
        v = part.double().numpy()
        norm = weight.double().numpy() * v / np.sqrt((v * v).mean(-1, keepdims=True) + eps)
        np.testing.assert_allclose(rms_norm(part, weight, eps).numpy(), norm, rtol=1e-5, atol=1e-6)

    def test_prediction_gate_reads_only_when_measurably_right(self):
        from streamed_moe import StreamedMoE
        moe = StreamedMoE.__new__(StreamedMoE)
        moe.predict_next, moe.predict_mode, moe.predict_threshold, moe.predict_warmup = 6, 'gated', 0.6, 3
        moe._history, moe.prediction = {}, dict(guessed=0, routed=0, routed_and_guessed=0)
        moe.sources = {n: dict(layers=0, guessed=0, correct=0, cold_guessed=0, cold_correct=0,
                               cold_precision_ema=None, reads_issued=0) for n in ('router', 'history')}
        for layer in range(4):                       # router guesses wrong, history guesses right
            moe._predicted = (layer, {'router': {1, 2}, 'history': {7, 8}}, {'router': {1, 2}, 'history': {7, 8}})
            self.assertFalse(moe._confident('history') and layer < 3)
            moe._score(layer, {7, 8, 9})
        self.assertTrue(moe._confident('history'))
        self.assertFalse(moe._confident('router'))
        stats = moe.prediction_stats()
        self.assertEqual((stats['sources']['router']['cold_precision'], stats['sources']['history']['cold_precision']),
                         (0.0, 1.0))
        self.assertEqual(stats['recall'], 0.0)
        moe.predict_mode = 'shadow'
        self.assertFalse(moe._confident('history'))
        moe.predict_mode = 'always'
        self.assertTrue(moe._confident('router'))


if __name__ == '__main__':
    result = unittest.main(exit=False, verbosity=2).result
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                          skipped=len(result.skipped), failures=len(result.failures), errors=len(result.errors),
                          torch='real' if real_torch is not None else 'NumPy stand-in for the cache tests; '
                                'ReferenceArithmeticTests skipped')))
    raise SystemExit(0 if result.wasSuccessful() else 1)
