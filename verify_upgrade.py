"""Check the v0.5 changes on this machine against the results saved before them.

Run with the CUDA Python environment (or .\\run_verify.ps1):

    python verify_upgrade.py            # everything
    python verify_upgrade.py --skip chat
    python verify_upgrade.py --host-cache-gib 0   # unknown options go to frustum_chat.py

Steps
  selftest  hardware-free unit tests (loader, caches, placement, generation flow)
  io        drive read speed: old mmap path vs direct reads (no GPU)
  layer0    layer-0 block; its output hash, routing and stage fingerprints must
            equal repro_baseline.json, which was recorded with the v0.4 code
  chat      "Say hello." for six tokens; token ids must equal chat_smoke_results.json
            and the timings are set beside the saved v0.4 run

Nothing is overwritten: outputs go to verify_runs/<timestamp>/.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def run(command, log):
    print('>', ' '.join(str(part) for part in command), flush=True)
    started = time.perf_counter()
    with log.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen([str(part) for part in command], cwd=HERE, text=True, encoding='utf-8',
                                   errors='replace', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            handle.write(line)
            print('   ', line.rstrip(), flush=True)
        code = process.wait()
    return code, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', help='Checkpoint folder. Default: FRUSTUM_MODEL or the parent folder')
    parser.add_argument('--skip', nargs='*', default=[], choices=('selftest', 'io', 'layer0', 'chat'))
    args, chat_options = parser.parse_known_args()
    folder = HERE / 'verify_runs' / time.strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True)
    model = ['--model', args.model] if args.model else []
    python = sys.executable
    report = dict(python=python, folder=str(folder), steps={})

    def record(name, **fields):
        report['steps'][name] = fields
        (folder / 'verify_upgrade_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    if 'selftest' not in args.skip:
        code, seconds = run([python, 'selftest.py'], folder / 'selftest.log')
        record('selftest', status='PASS' if code == 0 else 'FAIL', seconds=seconds)

    if 'io' not in args.skip:
        code, seconds = run([python, 'frustum_loader.py', *model, 'bench-io'], folder / 'io.log')
        fields = dict(status='DONE' if code == 0 else 'FAIL', seconds=seconds)
        if code == 0:
            text = (folder / 'io.log').read_text(encoding='utf-8')
            fields.update(json.loads(text[text.index('{'):]))
        record('io', **fields)

    if 'layer0' not in args.skip:
        output = folder / 'layer0.json'
        code, seconds = run([python, 'layer0_block.py', *model, '--output', output], folder / 'layer0.log')
        fields = dict(status='FAIL', seconds=seconds)
        baseline = HERE / 'repro_baseline.json'
        if code == 0 and baseline.exists():
            old, new = json.loads(baseline.read_text()), json.loads(output.read_text())
            checks = dict(
                same_output=old['output_sha256'] == new['output_sha256'],
                same_stage_fingerprints=old['full_prefill_trace'] == new['full_prefill_trace'],
                same_experts=old['full_prefill_MoE_reference']['selected_experts'] == new['full_prefill_MoE_reference']['selected_experts'],
                same_routing_weights=old['full_prefill_MoE_reference']['routing_weights'] == new['full_prefill_MoE_reference']['routing_weights'],
                same_resident_dense_weights=old['resident_dense_sha256'] == new['resident_dense_sha256'],
                same_router_weights=old['resident_router_sha256'] == new['resident_router_sha256'])
            fields.update(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                          output_sha256=new['output_sha256'],
                          seconds_before=old.get('seconds_including_CPU_validation'),
                          seconds_now=new.get('seconds_including_CPU_validation'),
                          logical_read_MB_before=old.get('logical_weight_read_MB'),
                          logical_read_MB_now=new.get('logical_weight_read_MB'),
                          note='Source hashes differ by design; the arithmetic results must not.')
        elif code == 0:
            fields.update(status='DONE', note='repro_baseline.json not found; nothing to compare against.')
        record('layer0', **fields)

    if 'chat' not in args.skip:
        output = folder / 'chat.json'
        extra = [option for option in chat_options if option != '--']
        code, seconds = run([python, 'frustum_chat.py', *model, '--prompt', 'Say hello.', '--max-new-tokens', '6',
                             '--evidence', output, *extra], folder / 'chat.log')
        fields = dict(status='FAIL', seconds=seconds)
        saved = HERE / 'chat_resident_head_results.json'
        if code == 0:
            new = json.loads(output.read_text(encoding='utf-8'))
            fields.update(status='DONE', token_ids=new['token_ids'], reply=new['reply'], summary=new['summary'],
                          peak_reserved_GiB=new['peak_reserved_GiB'],
                          step_seconds=[round(t['total_seconds'], 2) for t in new['token_timings']])
            if saved.exists():
                old = json.loads(saved.read_text(encoding='utf-8'))
                fields.update(status='PASS' if old['token_ids'] == new['token_ids'] else 'FAIL',
                              token_ids_before=old['token_ids'], total_seconds_before=old['seconds'],
                              total_seconds_now=new['seconds'],
                              step_seconds_before=[round(t['total_seconds'], 2) for t in old['token_timings']])
        record('chat', **fields)

    print(json.dumps(report, indent=2))
    failed = [name for name, step in report['steps'].items() if step['status'] == 'FAIL']
    print('FAILED: ' + ', '.join(failed) if failed else 'All requested steps completed.')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
