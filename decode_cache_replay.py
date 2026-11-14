"""Estimate expert payload reads from recorded single-token routing."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path

from frusium_loader import Catalog


def requests(catalog, step):
    for layer in step['layers']:
        prefix = f"layers.{layer['layer']}.ffn."
        selected, = layer['selected_experts']
        for expert in [*(f'experts.{index}' for index in sorted(set(selected))), 'shared_experts']:
            key = prefix + expert
            size = sum(catalog.tensors[f'{key}.{projection}.{kind}'].size
                       for projection in ('w1', 'w2', 'w3') for kind in ('weight', 'scale'))
            yield key, size


def replay(passes, capacity):
    cache = OrderedDict()
    used = 0
    results = []
    for step in passes:
        hits = read = 0
        for key, size in step:
            if size > capacity:
                raise ValueError('An expert exceeds the cache budget')
            if key in cache:
                cache.move_to_end(key)
                hits += 1
                continue
            while used + size > capacity:
                _, evicted = cache.popitem(last=False)
                used -= evicted
            cache[key] = size
            used += size
            read += size
        results.append(dict(requests=len(step), hits=hits, logical_bytes=read))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    evidence = json.loads(args.profile.read_text(encoding='utf-8'))
    catalog = Catalog(args.model)
    passes = [list(requests(catalog, step)) for step in evidence['token_timings'] if 'layers' in step]
    if len(passes) < 2:
        parser.error('At least two profiled decode passes are required')
    data = {
        'scope': 'Demand-order LRU estimate with cold start. Actual prefetch and allocator behavior may differ.',
        'working_set_bytes': [sum(size for _, size in step) for step in passes],
        'replay_by_cache_mib': {size: replay(passes, size * 2**20) for size in (256, 2048, 4096, 5120, 6144)},
    }
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
