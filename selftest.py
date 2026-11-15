"""Hardware-free checks for the loader, cache policy, expert cache and placement planner.

Runs on CPU against a small synthetic checkpoint written to a temp folder.
No model weights and no GPU are needed:

    python selftest.py

This does not exercise model arithmetic. Use verify_upgrade.py for that.
"""
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from cache_policy import Frequency, Tier, simulate
from frustum_loader import Catalog, PrototypeTests, default_model_root  # noqa: F401  (tests collected below)
from generation import chat_prompt, drive, reusable_prefix
import placement


def write_checkpoint(root, layers=3, experts=8, expert_bytes=64):
    """Tiny safetensors checkpoint using the real tensor naming scheme."""
    names = ['head.weight', 'norm.weight', 'embed.weight']
    for layer in range(layers):
        p = f'layers.{layer}.'
        names += [p + 'attn.wq_a.weight', p + 'attn.wq_a.scale', p + 'attn_norm.weight',
                  p + 'ffn.gate.weight', p + 'ffn.gate.bias', p + 'engram.embed.weight']
        for expert in [f'experts.{i}' for i in range(experts)] + ['shared_experts']:
            names += [f'{p}ffn.{expert}.{w}.{kind}' for w in ('w1', 'w2', 'w3') for kind in ('weight', 'scale')]
    header, offset = {}, 0
    payload = bytearray()
    for index, name in enumerate(names):
        size = expert_bytes if '.ffn.experts.' in name or '.shared_experts.' in name else 40
        header[name] = dict(dtype='U8', shape=[size], data_offsets=[offset, offset + size])
        payload += bytes((index * 7 + j) % 251 for j in range(size))
        offset += size
    raw = json.dumps(header).encode()
    (root / 'model-00001.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
    (root / 'config.json').write_text(json.dumps(dict(text_config=dict(
        num_hidden_layers=layers, n_routed_experts=experts, num_experts_per_tok=2))))
    (root / 'model.safetensors.index.json').write_text(json.dumps(dict(
        metadata=dict(total_size=offset), weight_map={name: 'model-00001.safetensors' for name in names})))
    return Catalog(root)


class PolicyTests(unittest.TestCase):
    def test_frequency_rank_orders_by_decayed_count(self):
        f = Frequency(half_life=2)
        for _ in range(5):
            f.tick(); f.touch('hot'); f.touch('hot')  # twice in one pass counts once
        f.tick(); f.touch('cold')
        self.assertGreater(f.rank('hot'), f.rank('cold'))
        self.assertAlmostEqual(f.count('hot'), sum(0.5 ** (i / 2) for i in range(1, 6)), places=9)
        for _ in range(20):
            f.tick()
        f.touch('cold')
        # 'hot' has decayed for 20 passes; a key used again now outranks it.
        self.assertGreater(f.rank('cold'), f.rank('hot'))
        self.assertEqual(f.rank('never'), float('-inf'))

    def test_lru_and_lfu_victims(self):
        f = Frequency()
        lru, lfu = Tier(30, 'lru'), Tier(30, 'lfu')
        for key in 'abc':
            f.touch(key)
            lru.add(key, 10)
            lfu.add(key, 10)
        f.tick(); f.touch('a'); lru.touch('a'); lfu.touch('a')
        self.assertEqual(lru.victim(f), 'b')
        # b and c tie on frequency; the more recent one goes first under LFU.
        self.assertEqual(lfu.victim(f), 'c')
        self.assertEqual(lfu.victim(f, protected=('c',)), 'b')
        self.assertEqual(lfu.make_room(20, f, protected=('a',)), ['c', 'b'])
        self.assertEqual(lfu.used, 10)

    def test_pinned_and_exhaustion(self):
        f = Frequency()
        tier = Tier(20, 'lfu', pinned=lambda key: key == 'shared')
        tier.add('shared', 10)
        tier.add('x', 10)
        self.assertEqual(tier.make_room(10, f), ['x'])
        with self.assertRaises(RuntimeError):
            tier.make_room(20, f)

    def test_admission(self):
        f = Frequency()
        tier = Tier(20, 'lfu')
        for _ in range(3):
            f.tick(); f.touch('hot')
        f.touch('warm')
        tier.add('hot', 10)
        tier.add('warm', 10)
        f.touch('once')
        self.assertIsNone(Tier(5, 'lru').admit('once', 10, f))
        self.assertIsNone(tier.admit('hot', 10, f))
        f.touch('newer'); f.tick(); f.touch('newer')
        self.assertEqual(tier.admit('newer', 10, f), ['warm'])
        self.assertEqual(tier.used, 10)

    def test_small_tier_cyclic_access(self):
        # Three layers, two keys each, tier holds four keys: LRU never hits, LFU does.
        step = [(layer, expert) for layer in range(3) for expert in range(2)]
        sizes = {key: 10 for key in step}
        lru = simulate([step] * 6, sizes, 40, policy='lru')
        lfu = simulate([step] * 6, sizes, 40, policy='lfu')
        self.assertEqual(sum(r['gpu_hits'] for r in lru), 0)
        self.assertGreater(sum(r['gpu_hits'] for r in lfu[1:]), 10)
        tiered = simulate([step] * 3, sizes, 40, host_bytes=60, policy='lru')
        self.assertEqual(tiered[1]['disk_reads'], 0)
        self.assertEqual(tiered[1]['host_hits'], 6)


class ExpertCacheTests(unittest.TestCase):
    def setUp(self):
        import torch
        from expert_cache import TieredExpertCache
        self.torch, self.cls = torch, TieredExpertCache
        self.tmp = tempfile.TemporaryDirectory()
        # Registered first so it runs last: Windows cannot delete files with open handles.
        self.addCleanup(self.tmp.cleanup)
        self.catalog = write_checkpoint(Path(self.tmp.name))

    def make(self, gpu_keys, host_keys=0, **options):
        cache = self.cls(self.torch, self.catalog, 'cpu', gpu_keys * 384, host_keys * 384, **options)
        self.addCleanup(cache.close)
        return cache

    def key(self, cache, layer, expert):
        key = (layer, expert)
        prefix = f'layers.{layer}.ffn.' + ('shared_experts' if expert == 'shared' else f'experts.{expert}')
        cache.plan[key] = [f'{prefix}.{w}.{kind}' for w in ('w1', 'w2', 'w3') for kind in ('weight', 'scale')]
        return key

    def expected(self, cache, key):
        return [bytes(cache.reader.read(self.catalog.tensors[name])) for name in cache.plan[key]]

    def content(self, entry):
        return [bytes(tensor.numpy().tobytes()) for tensor in entry.tensors]

    def test_contents_and_tiers(self):
        cache = self.make(gpu_keys=2, host_keys=4, io_threads=2, gpu_policy='lru')
        keys = [self.key(cache, 0, e) for e in range(4)]
        truth = {key: self.expected(cache, key) for key in keys}
        cache.reader.bytes_read = 0
        cache.prefetch(keys)
        for key in keys:
            self.assertEqual(self.content(cache.get(key, (key,))), truth[key])
        self.assertEqual((cache.disk_reads, cache.hits, cache.host_hits, cache.evictions), (4, 0, 0, 2))
        self.assertEqual(cache.reader.bytes_read, 4 * 384)
        self.assertLessEqual(cache.used, cache.budget)
        # Second pass: two come from the GPU tier or host tier, none from disk.
        for key in keys:
            self.assertEqual(self.content(cache.get(key, (key,))), truth[key])
        self.assertEqual(cache.disk_reads, 4)
        self.assertEqual(cache.reader.bytes_read, 4 * 384)
        self.assertEqual(cache.hits + cache.host_hits, 4)
        self.assertFalse(cache.pending)

    def test_without_threads_or_host_tier(self):
        cache = self.make(gpu_keys=1, io_threads=0)
        a, b = self.key(cache, 1, 0), self.key(cache, 1, 1)
        truth = self.expected(cache, a)
        cache.prefetch([a, b])
        self.assertFalse(cache.pending)
        self.assertEqual(self.content(cache.get(a)), truth)
        cache.get(b)
        self.assertEqual(self.content(cache.get(a)), truth)
        self.assertEqual((cache.disk_reads, cache.evictions, len(cache.ram)), (3, 2, 0))

    def test_pinned_shared_stays(self):
        pinned = lambda key: key[1] == 'shared'
        cache = self.make(gpu_keys=2, host_keys=4, pinned=pinned)
        shared = self.key(cache, 2, 'shared')
        cache.get(shared)
        for expert in range(5):
            cache.get(self.key(cache, 2, expert))
        self.assertIn(shared, cache.cache)
        self.assertNotIn(shared, cache.ram)
        self.assertEqual(cache.get(shared).size, 384)
        self.assertEqual(cache.hits, 1)

    def test_oversized_and_protected(self):
        cache = self.make(gpu_keys=1)
        a, b = self.key(cache, 0, 0), self.key(cache, 0, 1)
        cache.get(a)
        with self.assertRaises(RuntimeError):
            cache.get(b, protected=(a,))
        small = self.cls(self.torch, self.catalog, 'cpu', 100)
        self.addCleanup(small.close)
        with self.assertRaises(RuntimeError):
            small.get(self.key(small, 0, 0))

    def test_discard_pending(self):
        cache = self.make(gpu_keys=2, io_threads=1)
        keys = [self.key(cache, 0, e) for e in range(6)]
        cache.prefetch(keys)
        cache.discard_pending()
        self.assertFalse(cache.pending)
        self.assertEqual(len(self.content(cache.get(keys[5]))), 6)


class GenerationTests(unittest.TestCase):
    """A toy engine whose 'logits' are just the tokens it has been fed."""
    def engine(self):
        state, calls = [], []

        def run(tokens, phase, want_logits):
            if phase == 'prefill':
                state.clear()
            elif len(tokens) != 1:
                raise ValueError('Decode requires one token per step')
            state.extend(tokens)
            calls.append((phase, list(tokens), want_logits))
            return (list(state) if want_logits else None), dict(phase=phase, token_id=None)
        return state, calls, run

    def test_prefill_then_decode(self):
        state, calls, run = self.engine()
        fed = []
        out, timings, reused, stopped = drive(fed, [1, 2, 3], 3, eos=99, pick=lambda seen: 10 + len(seen), run=run)
        self.assertEqual((out, reused, stopped), ([13, 14, 15], 0, False))
        # The last generated token is not fed back; state and bookkeeping agree.
        self.assertEqual(fed, [1, 2, 3, 13, 14])
        self.assertEqual(fed, state)
        self.assertEqual([c[0] for c in calls], ['prefill', 'decode', 'decode'])
        self.assertEqual([t['token_id'] for t in timings], [13, 14, 15])

    def test_eos_and_reuse(self):
        state, calls, run = self.engine()
        fed = []
        replies = iter([7, 8, 99])
        out, _, _, stopped = drive(fed, [1, 2], 8, eos=99, pick=lambda seen: next(replies), run=run)
        self.assertEqual((out, stopped, fed), ([7, 8], True, [1, 2, 7, 8]))
        calls.clear()
        follow = [1, 2, 7, 8, 99, 5, 6]
        self.assertEqual(reusable_prefix(fed, follow), 4)
        out, timings, reused, _ = drive(fed, follow, 1, eos=99, pick=lambda seen: 42, run=run)
        self.assertEqual((out, reused), ([42], 4))
        self.assertEqual(calls, [('reuse_feed', [99], False), ('reuse_feed', [5], False), ('reuse_feed', [6], True)])
        self.assertEqual(state, follow)
        self.assertEqual([t['token_id'] for t in timings], [None, None, 42])

    def test_mismatch_and_disabled_reuse_rebuild(self):
        state, calls, run = self.engine()
        fed = []
        drive(fed, [1, 2, 3], 1, eos=99, pick=lambda seen: 4, run=run)
        for ids, reuse in (([1, 9, 3, 4], True), ([1, 2, 3, 4], False), ([1, 2, 3], True)):
            calls.clear()
            fed[:] = [1, 2, 3]
            _, _, reused, _ = drive(fed, ids, 1, eos=99, pick=lambda seen: 4, run=run, reuse=reuse)
            self.assertEqual((reused, calls[0][0], state), (0, 'prefill', ids))

    def test_failure_invalidates_state(self):
        fed = [1, 2]

        def broken(tokens, phase, want_logits):
            raise RuntimeError('boom')
        with self.assertRaises(RuntimeError):
            drive(fed, [1, 2, 3], 1, eos=99, pick=lambda seen: 4, run=broken)
        self.assertEqual(fed, [])


class TemplateTests(unittest.TestCase):
    CASES = [
        (None, [], 'Say hello.'),
        (None, [('Say hello.', 'Hello! How can I help?')], 'Reply with only the word Dino.'),
        ('Be brief.', [('a', 'b'), ('c', 'd')], 'e'),
    ]

    def test_fixed_strings(self):
        self.assertEqual(chat_prompt([], 'Say hello.'),
                         '<｜begin▁of▁sentence｜><｜User｜>Say hello.<｜Assistant｜></think>')
        self.assertEqual(chat_prompt([('a', 'b')], 'c', 'Be brief.'),
                         '<｜begin▁of▁sentence｜><｜System｜>Be brief.<｜User｜>a<｜Assistant｜></think>b'
                         '<｜end▁of▁sentence｜><｜User｜>c<｜Assistant｜></think>')

    def test_matches_official_encoder(self):
        folder = Path(default_model_root()) / 'encoding'
        if not (folder / 'encoding.py').exists():
            self.skipTest('Official encoding/encoding.py not found next to the checkpoint')
        import importlib.util
        spec = importlib.util.spec_from_file_location('deepseek_official_encoding', folder / 'encoding.py')
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        for system, history, prompt in self.CASES:
            messages = [dict(role='system', content=system)] if system else []
            for user, reply in history:
                messages += [dict(role='user', content=user), dict(role='assistant', content=reply)]
            messages.append(dict(role='user', content=prompt))
            expected = official.encode_messages(messages, thinking_mode='chat')
            if isinstance(expected, tuple):
                expected = expected[0]
            self.assertEqual(chat_prompt(history, prompt, system), expected)


class PlacementTests(unittest.TestCase):
    def catalog(self):
        # Sizes taken from the DeepSeek-V4.1-Flash decode profile, metadata only.
        tensors = {'head.weight': 1323827200, 'norm.weight': 10240, 'embed.weight': 1323827200}
        for layer in range(40):
            p = f'layers.{layer}.'
            tensors[p + 'attn.wq_a.weight'] = 140_600_000
            tensors[p + 'ffn.gate.weight'] = 3_933_696
            tensors[p + 'engram.embed.weight'] = 98_000_000_000 if layer in (1, 14) else 0
            tensors[p + 'ffn.shared_experts.w1.weight'] = 35_424_000
            for expert in range(384):
                tensors[f'{p}ffn.experts.{expert}.w1.weight'] = 18_800_640
        tensors['layers.40.attn.wq_a.weight'] = 10 ** 12  # beyond the text stack: must be ignored
        config = dict(text_config=dict(num_hidden_layers=40, n_routed_experts=384, num_experts_per_tok=6))
        return SimpleNamespace(config=config, tensors={k: SimpleNamespace(size=v) for k, v in tensors.items()})

    def test_sixteen_gib_card(self):
        p = placement.plan(self.catalog(), 12 * placement.GIB, ram_available=32 * placement.GIB)
        s = p['sizes']
        self.assertEqual(s['dense'], 40 * 140_600_000 + 10240)
        self.assertEqual(s['one_routed_expert'], 18_800_640)
        self.assertEqual((p['dense'], p['head'], p['pin_shared'], p['gpu_policy']), ('resident', 'resident', True, 'lfu'))
        total = (p['dense_cache_mib'] + p['router_cache_mib'] + p['expert_cache_mib']) * placement.MIB + p['head_bytes']
        self.assertLessEqual(total, 10 * placement.GIB + 2 * placement.MIB)
        self.assertGreater(p['expert_cache_mib'], 3000)
        self.assertEqual(p['host_cache_mib'], 13 * 1024)
        self.assertIn('resident', placement.describe(p))

    def test_small_card_falls_back(self):
        p = placement.plan(self.catalog(), 8 * placement.GIB, ram_available=10 * placement.GIB)
        self.assertEqual((p['dense'], p['head']), ('streamed', 'resident'))
        self.assertTrue(p['notes'])
        self.assertEqual(p['host_cache_mib'], 2048)
        # A host tier too small to hold 64 experts is switched off.
        self.assertEqual(placement.plan(self.catalog(), 8 * placement.GIB, ram_available=8 * placement.GIB)['host_cache_mib'], 0)
        p = placement.plan(self.catalog(), 24 * placement.GIB, host_cache_gib=0)
        self.assertEqual((p['gpu_policy'], p['host_cache_mib']), ('lru', 0))

    def test_explicit_requests(self):
        p = placement.plan(self.catalog(), 12 * placement.GIB, dense='streamed', head='streamed',
                           expert_cache_mib=256, host_cache_gib=2, ram_available=64 * placement.GIB)
        self.assertEqual((p['dense_cache_mib'], p['head_bytes'], p['expert_cache_mib'], p['pin_shared']),
                         (256, 0, 256, False))
        self.assertEqual(p['host_cache_mib'], 2048)
        with self.assertRaises(ValueError):
            placement.plan(self.catalog(), 3 * placement.GIB, dense='resident', head='resident')


if __name__ == '__main__':
    unittest.main(verbosity=2)
