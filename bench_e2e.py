"""Repeatable end-to-end benchmark for DeepSeek chat: cold and warm OS cache, several runs.

    python bench_e2e.py                               patch (current code) vs in-tree baseline flags
    python bench_e2e.py --baseline-dir ..\\Frustum_v06  also run an untouched v0.6 checkout
    python bench_e2e.py --configs patch spill --spill-dir D:\\frustum_spill
    python bench_e2e.py --runs 5 --tokens 24 --prompt "..."

Every measurement is one fresh process running frustum_chat.py with --evidence, so the
application caches (GPU tier, RAM tier) always start empty. Per configuration and repetition:
  cold  the checkpoint shards are purged from the OS file cache first (os_cache.py)
  warm  the same command again straight after, with whatever the OS kept in RAM
Cold is verified, not assumed: a cold run whose physical device bytes are under 80% of the
logical bytes it requested is flagged `cold_not_achieved` (the purge did not work, or the
counter is unavailable) and should not be read as a cold number.

Reported per configuration and cache state, as median and p95 over the runs (with three
runs p95 is close to the slowest run): time to first token, seconds per decode token,
physical device bytes, logical bytes, first-time vs re-read expert bytes per token,
re-read amplification, GPU / RAM / spill / drive hit rates, effective MB/s while the drive
was busy, mean queue depth, drive utilisation and compute/transfer overlap. Token ids are
compared across configurations: the patch changes no arithmetic, so they must be equal.

Configurations
  baseline  current code with v0.7 behaviour switched off (FRUSTUM_COALESCE=0,
            FRUSTUM_PROTECT_LAYER=0, no prediction, buffered --reader direct). Same
            instrumentation, old I/O behaviour.
  patch     current defaults.
  shadow    patch + --predict-next 6 --predict-mode shadow: measures guess precision, reads nothing.
  gated     patch + --predict-next 6 (reads guesses only while measurably right).
  spill     patch + --spill-dir (needs --spill-dir; its first cold run fills the tier).
  unbuffered patch + --reader unbuffered (v0.8, Windows): reads bypass the OS file cache and the
            auto host tier grows. Its "warm" runs stay cold by design: nothing is left in the
            OS cache between processes. Default for frustum_chat.py on Windows since v0.8, so
            patch and unbuffered are the same there.
  direct    patch + --reader direct: the v0.7 buffered reader, for A/B runs against v0.8.
  async_upload patch with FRUSTUM_ASYNC_UPLOAD=1: v0.8 pinned staging and copy-stream
            uploads (opt-in; measured slower than the default blocking uploads).
  v06       frustum_chat.py from --baseline-dir. It has no I/O ledger, so only timings,
            token ids and physical device bytes are available for it.

STATUS: the aggregation is unit-checked (python bench_e2e.py --self-test). The benchmark
itself is UNRUN: it needs the GPU and the checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import os_cache

HERE = Path(__file__).resolve().parent
CONFIGS = {
    'baseline': (['--predict-next', '0', '--reader', 'direct'], {'FRUSTUM_COALESCE': '0', 'FRUSTUM_PROTECT_LAYER': '0'}),
    'patch': (['--predict-next', '0'], {}),
    'shadow': (['--predict-next', '6', '--predict-mode', 'shadow'], {}),
    'gated': (['--predict-next', '6', '--predict-mode', 'gated'], {}),
    'spill': (['--predict-next', '0'], {}),
    'unbuffered': (['--predict-next', '0', '--reader', 'unbuffered'], {}),
    'direct': (['--predict-next', '0', '--reader', 'direct'], {}),
    'async_upload': (['--predict-next', '0'], {'FRUSTUM_ASYNC_UPLOAD': '1'}),
    'v06': (['--predict-next', '0'], {}),
}


def percentile(values, q):
    """Linear-interpolated percentile of a non-empty list."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(median=statistics.median(values), p95=percentile(values, 0.95), min=min(values), max=max(values),
                n=len(values))


def metrics(evidence, physical, seconds):
    """Flatten one run into the reported numbers. Works for v0.6 evidence too (fewer fields)."""
    summary = evidence['summary']
    io = summary.get('decode_io') or {}
    steps = [t['io'] for t in evidence['token_timings'] if t['phase'] == 'decode' and 'io' in t]
    total = lambda name: sum(step[name] for step in steps) if steps else None
    requests = total('requests')
    rate = lambda name: (total(name) / requests) if requests else None
    unique, repeat = io.get('unique_bytes_per_token'), io.get('repeat_bytes_per_token')
    return dict(
        ttft_seconds=summary['first_token_seconds'],
        seconds_per_token=summary['mean_decode_seconds'],
        process_seconds=seconds,
        physical_bytes=physical['bytes_read'] if physical else None,
        logical_bytes=summary['disk_bytes'],
        physical_over_logical=(physical['bytes_read'] / summary['disk_bytes'])
        if physical and summary['disk_bytes'] else None,
        unique_expert_bytes_per_token=unique, repeat_expert_bytes_per_token=repeat,
        reread_amplification=((unique + repeat) / unique) if unique else None,
        gpu_hit_rate=rate('gpu_hits'), host_hit_rate=rate('host_hits'),
        spill_hit_rate=rate('spill_hits') if steps and 'spill_hits' in steps[0] else None,
        drive_hit_rate=rate('disk_reads'),
        effective_MBps=io.get('effective_MBps'), mean_queue_depth=io.get('mean_queue_depth'),
        drive_utilization=io.get('drive_utilization'), overlap_fraction=io.get('overlap_fraction'))


def aggregate(runs):
    """runs: list of dict(config, state, metrics, token_ids, flags). Returns the report table."""
    table = {}
    for run in runs:
        table.setdefault(run['config'], {}).setdefault(run['state'], []).append(run)
    report = {}
    for config, states in table.items():
        report[config] = {}
        for state, items in states.items():
            names = items[0]['metrics'].keys()
            report[config][state] = dict(
                {name: summarize([item['metrics'][name] for item in items]) for name in names},
                runs=len(items), flags=sorted({flag for item in items for flag in item['flags']}))
    tokens = {config: {tuple(item['token_ids']) for state in states.values() for item in state}
              for config, states in table.items()}
    first = next(iter(tokens.values())) if tokens else set()
    report['_token_ids'] = dict(
        identical_within_each_config=all(len(ids) == 1 for ids in tokens.values()),
        identical_across_configs=all(ids == first for ids in tokens.values()) and len(first) == 1)
    return report


def markdown(report):
    order = ['ttft_seconds', 'seconds_per_token', 'physical_bytes', 'logical_bytes', 'physical_over_logical',
             'unique_expert_bytes_per_token', 'repeat_expert_bytes_per_token', 'reread_amplification',
             'gpu_hit_rate', 'host_hit_rate', 'spill_hit_rate', 'drive_hit_rate', 'effective_MBps',
             'mean_queue_depth', 'drive_utilization', 'overlap_fraction']
    def cell(stat, name):
        if stat is None:
            return 'n/a'
        scale = 1e9 if name.endswith('bytes') or 'bytes_per_token' in name else 1
        unit = ' GB' if scale != 1 else ''
        return f'{stat["median"] / scale:.3g} / {stat["p95"] / scale:.3g}{unit}'
    lines = []
    for state in ('cold', 'warm'):
        configs = [c for c in report if not c.startswith('_') and state in report[c]]
        if not configs:
            continue
        lines += [f'### {state} OS cache (median / p95)', '', '| metric | ' + ' | '.join(configs) + ' |',
                  '|---|' + '---|' * len(configs)]
        for name in order:
            lines.append(f'| {name} | ' + ' | '.join(cell(report[c][state].get(name), name) for c in configs) + ' |')
        lines.append('| runs / flags | ' + ' | '.join(
            f'{report[c][state]["runs"]} {",".join(report[c][state]["flags"]) or "-"}' for c in configs) + ' |')
        lines.append('')
    lines.append(f'Token ids: {json.dumps(report["_token_ids"])}')
    return '\n'.join(lines)


def self_test():
    assert percentile([1, 2, 3], 0.95) == 2.9 and percentile([5], 0.95) == 5
    def fake(seconds, ids, unique):
        io = dict(requests=10, gpu_hits=5, host_hits=2, spill_hits=0, disk_reads=3)
        evidence = dict(summary=dict(first_token_seconds=seconds, mean_decode_seconds=seconds / 10,
                                     disk_bytes=1000,
                                     decode_io=dict(unique_bytes_per_token=unique, repeat_bytes_per_token=50,
                                                    effective_MBps=400.0, mean_queue_depth=2.0,
                                                    drive_utilization=0.9, overlap_fraction=0.2)),
                        token_timings=[dict(phase='prefill'), dict(phase='decode', io=io)])
        return dict(metrics=metrics(evidence, dict(bytes_read=900), seconds), token_ids=ids, flags=[])
    runs = [dict(fake(s, [1, 2], 100), config='patch', state='cold') for s in (10.0, 12.0, 11.0)]
    runs += [dict(fake(20.0, [1, 2], 100), config='baseline', state='cold')]
    report = aggregate(runs)
    cold = report['patch']['cold']
    assert cold['ttft_seconds']['median'] == 11.0 and abs(cold['ttft_seconds']['p95'] - 11.9) < 1e-9
    assert cold['physical_over_logical']['median'] == 0.9 and cold['reread_amplification']['median'] == 1.5
    assert cold['drive_hit_rate']['median'] == 0.3 and report['_token_ids']['identical_across_configs']
    runs.append(dict(fake(9.0, [1, 3], 100), config='baseline', state='warm'))
    assert not aggregate(runs)['_token_ids']['identical_across_configs']
    old = dict(summary=dict(first_token_seconds=1.0, mean_decode_seconds=2.0, disk_bytes=10),
               token_timings=[dict(phase='decode')])
    assert metrics(old, None, 3.0)['gpu_hit_rate'] is None      # v0.6 evidence has no ledger
    assert 'cold OS cache' in markdown(report)
    print('bench_e2e self-test: PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=Path(os.environ.get('FRUSTUM_MODEL') or HERE.parent))
    parser.add_argument('--configs', nargs='+', choices=sorted(CONFIGS), default=['baseline', 'patch'])
    parser.add_argument('--runs', type=int, default=3, help='Repetitions per configuration and cache state (>= 3)')
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--prompt', default='Do you like Dino Buddies? Reply in one short sentence.')
    parser.add_argument('--baseline-dir', type=Path, help='An untouched v0.6 checkout, for the v06 configuration')
    parser.add_argument('--spill-dir', type=Path, help='Folder on a fast local drive, for the spill configuration')
    parser.add_argument('--spill-gib', type=float, default=64.0)
    parser.add_argument('--no-purge', action='store_true', help='Skip the cold runs (warm only)')
    parser.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                        help='Arguments passed to every frustum_chat.py run, e.g. --extra --host-cache-gib 8')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.runs < 3:
        parser.error('--runs must be at least 3')
    if args.baseline_dir:
        args.configs = list(dict.fromkeys(args.configs + ['v06']))
    if 'v06' in args.configs and not args.baseline_dir:
        parser.error('v06 needs --baseline-dir')
    if 'spill' in args.configs and not args.spill_dir:
        parser.error('spill needs --spill-dir')
    out = args.output or HERE / 'bench_runs' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    shards = sorted(args.model.glob('*.safetensors'))
    counters = os_cache.DeviceCounters(args.model)
    log = (out / 'bench.log').open('w', encoding='utf-8')

    def say(text):
        print(text, flush=True)
        log.write(text + '\n'); log.flush()

    say(f'model {args.model} ({len(shards)} shards); device counters: '
        f'{"ok" if counters.read() else "UNAVAILABLE (" + str(counters.error) + ")"}')
    runs = []
    for repetition in range(args.runs):
        for config in args.configs:
            extra, env_extra = CONFIGS[config]
            folder = args.baseline_dir if config == 'v06' else HERE
            if config == 'spill':
                extra = extra + ['--spill-dir', str(args.spill_dir), '--spill-gib', str(args.spill_gib)]
            for state in (('warm',) if args.no_purge else ('cold', 'warm')):
                flags = []
                if state == 'cold':
                    purged, failed = os_cache.purge_file_cache(shards)
                    if failed:
                        flags.append(f'purge_failed_{failed}')
                evidence_path = out / f'{config}_{state}_{repetition}.json'
                command = [sys.executable, str(folder / 'frustum_chat.py'), '--model', str(args.model),
                           '--prompt', args.prompt, '--max-new-tokens', str(args.tokens),
                           '--evidence', str(evidence_path), *extra, *args.extra]
                if config != 'v06':
                    command.append('--trace-routing')
                say(f'[{repetition + 1}/{args.runs}] {config} {state}: {" ".join(command[1:])} '
                    + ' '.join(f'{k}={v}' for k, v in env_extra.items()))
                before, began = counters.read(), time.perf_counter()
                done = subprocess.run(command, cwd=folder, env=dict(os.environ, PYTHONUTF8='1', **env_extra),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
                seconds = time.perf_counter() - began
                physical = os_cache.delta(before, counters.read())
                log.write(done.stdout + '\n'); log.flush()
                if done.returncode:
                    say(f'  FAILED with exit code {done.returncode}; see bench.log')
                    raise SystemExit(1)
                evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
                numbers = metrics(evidence, physical, seconds)
                ratio = numbers['physical_over_logical']
                if physical is None:
                    flags.append('physical_bytes_unavailable')
                elif state == 'cold' and ratio is not None and ratio < 0.8:
                    flags.append('cold_not_achieved')
                if config == 'spill' and state == 'cold' and repetition == 0:
                    flags.append('first_run_fills_spill')
                say(f'  first token {numbers["ttft_seconds"]:.1f}s, '
                    f'{numbers["seconds_per_token"] or float("nan"):.2f} s/token, physical/logical '
                    f'{"n/a" if ratio is None else format(ratio, ".2f")} {" ".join(flags)}')
                runs.append(dict(config=config, state=state, repetition=repetition, metrics=numbers,
                                 token_ids=evidence['token_ids'], flags=flags, evidence=evidence_path.name))
    report = aggregate(runs)
    (out / 'results.json').write_text(json.dumps(dict(
        prompt=args.prompt, tokens=args.tokens, runs_per_cell=args.runs, configs=args.configs,
        note='physical_bytes is device-wide for the model drive; other programs reading it are included',
        report=report, runs=runs), indent=2), encoding='utf-8')
    text = markdown(report)
    (out / 'results.md').write_text(text + '\n', encoding='utf-8')
    say('\n' + text + f'\n\nWrote {out}')
    log.close()


if __name__ == '__main__':
    main()
