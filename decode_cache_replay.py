"""Replay recorded routing through the cache policies to estimate disk reads per token.

Input is an evidence file written by `frustum_chat.py --profile-decode`. Every
profiled step (decode or reuse_feed) contributes one pass. The more passes the
profile holds, the more the comparison means: two passes say little about
which policy wins.

    python decode_cache_replay.py decode_profile_6144_results.json
    python decode_cache_replay.py profile.json --gpu-mib 2048 3400 --host-gib 0 8
"""
import argparse
import json
from pathlib import Path

from cache_policy import simulate
from frustum_loader import Catalog, default_model_root


def requests(catalog, step):
    for layer in step['layers']:
        prefix = f"layers.{layer['layer']}.ffn."
        selected, = layer['selected_experts']
        for expert in [*(f'experts.{index}' for index in sorted(set(selected))), 'shared_experts']:
            key = prefix + expert
            size = sum(catalog.tensors[f'{key}.{projection}.{kind}'].size
                       for projection in ('w1', 'w2', 'w3') for kind in ('weight', 'scale'))
            yield key, size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--model', type=Path, default=Path(default_model_root()))
    parser.add_argument('--gpu-mib', type=int, nargs='+', default=[256, 2048, 3400, 6144])
    parser.add_argument('--host-gib', type=float, nargs='+', default=[0, 8])
    parser.add_argument('--pin-shared', action='store_true', help='Treat shared experts as never evicted')
    args = parser.parse_args()
    evidence = json.loads(args.profile.read_text(encoding='utf-8'))
    catalog = Catalog(args.model)
    passes = [list(requests(catalog, step)) for step in evidence['token_timings'] if 'layers' in step]
    if len(passes) < 2:
        parser.error('At least two profiled passes are required')
    sizes = {key: size for step in passes for key, size in step}
    steps = [[key for key, _ in step] for step in passes]
    repeats = [len(set(a) & set(b)) / len(b) for a, b in zip(steps, steps[1:])]
    pinned = (lambda key: key.endswith('shared_experts')) if args.pin_shared else None
    table = []
    for gpu_mib in args.gpu_mib:
        for host_gib in args.host_gib:
            for policy in ('lru', 'lfu'):
                try:
                    result = simulate(steps, sizes, gpu_mib * 2**20, int(host_gib * 2**30), policy, pinned=pinned)
                except (RuntimeError, ValueError) as error:
                    table.append(dict(gpu_mib=gpu_mib, host_gib=host_gib, gpu_policy=policy, error=str(error)))
                    continue
                later = result[1:]
                table.append(dict(
                    gpu_mib=gpu_mib, host_gib=host_gib, gpu_policy=policy,
                    disk_GB_per_pass_after_first=round(sum(r['disk_bytes'] for r in later) / len(later) / 1e9, 3),
                    gpu_hit_rate=round(sum(r['gpu_hits'] for r in later) / sum(r['requests'] for r in later), 3),
                    host_hit_rate=round(sum(r['host_hits'] for r in later) / sum(r['requests'] for r in later), 3),
                    per_pass=result))
    print(json.dumps({
        'scope': 'Demand-order replay from a cold start; host tier is LRU. An estimate, not a latency measurement.',
        'passes': len(passes),
        'working_set_bytes': [sum(size for _, size in step) for step in passes],
        'fraction_of_requests_repeated_from_previous_pass': [round(value, 3) for value in repeats],
        'replay': table,
    }, indent=2))


if __name__ == '__main__':
    main()
