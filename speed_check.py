"""Same-day A/B of the v0.6 speed-ups on real models: reference path first, then v0.6.

    python speed_check.py                 Phi-4 and GLM-4.7-Flash (a few minutes)
    python speed_check.py --deepseek      also DeepSeek V4.1 Flash (much longer)
    python speed_check.py --only phi4     one model

Each model runs twice in fresh processes with the same prompt and greedy decoding:
  before  FRUSTUM_FUSED=0 (and, for DeepSeek, --predict-next 0)
  after   defaults: fused decode kernels on (DeepSeek also runs a third time with --predict-next 6)
Reports mean seconds per decode token, the speed-up, whether the generated token ids
match, and DeepSeek's prediction recall. Writes speed_check_results.json.
Token ids can legitimately differ after many tokens: the fused kernels sum in a different
order than cuBLAS, so a near-tie between two tokens can flip.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PROMPT = 'Do you like Dino Buddies? Reply in one short sentence.'


def run(script, args, env_extra, evidence):
    env = dict(os.environ, PYTHONUTF8='1', **env_extra)
    command = [sys.executable, str(HERE / script), *args, '--evidence', str(evidence)]
    print('  $', ' '.join(command[1:]), ' '.join(f'{k}={v}' for k, v in env_extra.items()), flush=True)
    done = subprocess.run(command, cwd=HERE, env=env)
    if done.returncode:
        raise SystemExit(f'{script} failed with exit code {done.returncode}')
    return json.loads(Path(evidence).read_text(encoding='utf-8'))


def gguf_numbers(record):
    decode = record['decode_seconds']
    return dict(tokens=record['token_ids'], first_token_seconds=record['first_token_seconds'],
                mean_decode_seconds=sum(decode) / len(decode) if decode else None)


def deepseek_numbers(record):
    s = record['summary']
    return dict(tokens=[t.get('token_id') for t in record['token_timings'] if 'token_id' in t],
                first_token_seconds=s['first_token_seconds'], mean_decode_seconds=s['mean_decode_seconds'],
                prediction=record.get('prediction'), fused=record.get('fused'))


def compare(name, before, after):
    b, a = before['mean_decode_seconds'], after['mean_decode_seconds']
    result = dict(model=name, before=before, after=after,
                  decode_speedup=(b / a) if a and b else None,
                  same_tokens=before['tokens'] == after['tokens'])
    speed = f"{b:.3f} -> {a:.3f} s/token ({result['decode_speedup']:.1f}x)" if a and b else 'no decode steps'
    extra = ''
    if after.get('prediction') and after['prediction'].get('recall') is not None:
        extra = f", prediction recall {100 * after['prediction']['recall']:.0f}%"
    print(f'{name}: {speed}, same tokens: {result["same_tokens"]}{extra}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--deepseek', action='store_true', help='Include DeepSeek V4.1 Flash (slow)')
    parser.add_argument('--only', choices=('phi4', 'glm', 'deepseek'))
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--output', type=Path, default=HERE / 'speed_check_results.json')
    args = parser.parse_args()
    models = [args.only] if args.only else ['phi4', 'glm'] + (['deepseek'] if args.deepseek else [])
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in models:
            print(f'\n== {name} ==', flush=True)
            if name == 'phi4':
                common = ['--prompt', PROMPT, '--max-new-tokens', str(args.tokens)]
                before = gguf_numbers(run('phi4_chat.py', common, {'FRUSTUM_FUSED': '0'}, tmp / 'p0.json'))
                after = gguf_numbers(run('phi4_chat.py', common, {'FRUSTUM_FUSED': '1'}, tmp / 'p1.json'))
            elif name == 'glm':
                common = ['--prompt', PROMPT, '--max-new-tokens', str(min(args.tokens, 64))]
                before = gguf_numbers(run('glm_chat.py', common, {'FRUSTUM_FUSED': '0'}, tmp / 'g0.json'))
                after = gguf_numbers(run('glm_chat.py', common, {'FRUSTUM_FUSED': '1'}, tmp / 'g1.json'))
            else:
                common = ['--prompt', 'Say hello.', '--max-new-tokens', str(min(args.tokens, 12))]
                before = deepseek_numbers(run('frustum_chat.py', common + ['--no-fused', '--predict-next', '0'],
                                              {}, tmp / 'd0.json'))
                after = deepseek_numbers(run('frustum_chat.py', common, {}, tmp / 'd1.json'))
                results.append(compare('deepseek (fused only)', before, after))
                after = deepseek_numbers(run('frustum_chat.py', common + ['--predict-next', '6'], {}, tmp / 'd2.json'))
                name = 'deepseek (fused + prediction)'
            results.append(compare(name, before, after))
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\nWrote {args.output.name}')


if __name__ == '__main__':
    main()
