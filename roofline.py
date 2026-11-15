"""Bytes-per-token roofline for DeepSeek decode: is a token's time drive time or GPU time?

    python roofline.py                         arithmetic only, from checkpoint constants
    python roofline.py run.json [more.json]    add what a v0.7 run measured (--evidence files)

A decode token cannot be faster than (bytes it must read) / (drive throughput), and it
cannot be faster than its compute. Whichever floor is higher is the bottleneck, and work
on the other one cannot help. With evidence, this prints both floors per token and says
which one binds, then splits the bytes into first-time reads (no cache can remove them)
and re-reads (a better cache can).

Everything under "arithmetic" is calculation, not measurement.
"""
import argparse
import json
from pathlib import Path

EXPERT_BYTES = 18_800_640          # three FP4 matrices + per-32 scales (whitepaper section 1)
LAYERS, TOP_K = 40, 6


def arithmetic(expert_bytes=EXPERT_BYTES, layers=LAYERS, top_k=TOP_K):
    all_miss = expert_bytes * layers * top_k
    rows = []
    for miss in (1.0, 0.75, 0.5, 0.25, 0.1):
        for mbps in (250, 500, 1000, 3000):
            rows.append(dict(miss_fraction=miss, drive_MBps=mbps, GB_per_token=round(all_miss * miss / 1e9, 2),
                             drive_seconds_per_token=round(all_miss * miss / 1e6 / mbps, 2)))
    return dict(all_miss_bytes_per_token=all_miss, scenarios=rows)


def measured(evidence):
    steps = [t for t in evidence['token_timings'] if t['phase'] == 'decode' and 'io' in t]
    if not steps:
        return None
    n = len(steps)
    total = lambda name: sum(t['io'][name] for t in steps)
    wall = sum(t['total_seconds'] for t in steps)
    busy, blocked, read = total('busy_seconds'), total('blocked_seconds'), total('bytes_read')
    mbps = read / 1e6 / busy if busy else None
    utilization = busy / wall if wall else None
    out = dict(
        decode_tokens=n, seconds_per_token=wall / n,
        expert_GB_per_token=read / n / 1e9,
        first_time_GB_per_token=total('unique_bytes') / n / 1e9,
        reread_GB_per_token=total('repeat_bytes') / n / 1e9,
        gpu_rate=total('gpu_hits') / max(total('requests'), 1),
        host_rate=total('host_hits') / max(total('requests'), 1),
        spill_rate=total('spill_hits') / max(total('requests'), 1),
        disk_rate=total('disk_reads') / max(total('requests'), 1),
        effective_MBps=mbps, drive_utilization=utilization,
        mean_queue_depth=total('depth_seconds') / busy if busy else None,
        overlap_fraction=(busy - total('busy_blocked_seconds')) / busy if busy else None,
        drive_floor_seconds_per_token=busy / n,
        not_waiting_on_drive_seconds_per_token=(wall - blocked) / n,
        speculative=dict(issued=total('speculative_issued'), used=total('speculative_used'),
                         wasted=total('speculative_wasted'), wasted_GB=total('speculative_wasted_bytes') / 1e9),
        physical_over_logical=evidence.get('summary', {}).get('physical_over_logical'))
    if utilization is None:
        out['verdict'] = 'No expert reads during decode: storage is not the limit for this run.'
    elif utilization >= 0.85:
        share = out['reread_GB_per_token'] / out['expert_GB_per_token'] if out['expert_GB_per_token'] else 0
        out['verdict'] = (
            f'Storage-bound: the drive was busy {100 * utilization:.0f}% of decode time. Kernels and overlap cannot '
            f'help; only fewer bytes or a faster source can. {100 * share:.0f}% of the bytes were re-reads '
            '(removable by a larger or better cache tier); the rest were first-time reads (removable only by a '
            'faster copy, e.g. --spill-dir, or by reading them before they are needed).')
    elif utilization <= 0.5:
        out['verdict'] = (
            f'Compute-bound: the drive was idle {100 * (1 - utilization):.0f}% of decode time. Reducing bytes will '
            'not help much; GPU-side work (kernels, uploads) or using the idle drive time (accurate prediction) can.')
    else:
        out['verdict'] = (
            f'Mixed: drive busy {100 * utilization:.0f}% of decode time. Both floors matter; up to '
            f'{out["seconds_per_token"] * (1 - utilization):.2f} s/token of idle drive time exists for overlap.')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('evidence', nargs='*', type=Path)
    args = parser.parse_args()
    report = dict(arithmetic=arithmetic(), runs={})
    for path in args.evidence:
        result = measured(json.loads(path.read_text(encoding='utf-8')))
        report['runs'][path.name] = result or 'no v0.7 decode I/O records in this file'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
