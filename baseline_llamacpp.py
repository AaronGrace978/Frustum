"""Measure the same GGUF with llama.cpp and set it beside Frusium's saved numbers.

Phi-4 and GLM-4.7-Flash both fit on a 16 GB card, so llama.cpp is the baseline
any reader will compare against. This runs llama-bench (from a llama.cpp
release with CUDA support) and reports tokens per second for both.

    python baseline_llamacpp.py --llama-bench C:\\path\\to\\llama-bench.exe \
        --gguf G:\\Models\\Phi-4\\phi-4-Q4_K_S.gguf --frusium phi4_chat_results.json

No GPU code of this project runs here; nothing is loaded by Frusium_loader.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--llama-bench', default=shutil.which('llama-bench') or 'llama-bench')
    parser.add_argument('--frusium', type=Path, help='Saved Frusium run for the same model (has decode_seconds)')
    parser.add_argument('--gpu-layers', type=int, default=99)
    parser.add_argument('--prompt-tokens', type=int, default=64)
    parser.add_argument('--gen-tokens', type=int, default=64)
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    if not args.gguf.exists():
        parser.error('GGUF not found')
    if args.save and args.save.exists():
        parser.error('--save needs a new file')
    command = [args.llama_bench, '-m', str(args.gguf), '-ngl', str(args.gpu_layers),
               '-p', str(args.prompt_tokens), '-n', str(args.gen_tokens), '-o', 'json']
    try:
        done = subprocess.run(command, text=True, capture_output=True)
    except FileNotFoundError:
        raise SystemExit('llama-bench not found. Download a CUDA build of llama.cpp and pass --llama-bench.')
    if done.returncode:
        raise SystemExit(f'llama-bench failed:\n{done.stderr[-2000:]}')
    text = done.stdout
    try:
        runs = json.loads(text[text.index('['):text.rindex(']') + 1])
    except ValueError:
        raise SystemExit('Could not find JSON in llama-bench output:\n' + text[-2000:])
    report = dict(gguf=str(args.gguf), command=command, llama_bench_runs=runs)
    generation = [run for run in runs if run.get('n_gen') and run.get('avg_ts')]
    prompt = [run for run in runs if run.get('n_prompt') and not run.get('n_gen') and run.get('avg_ts')]
    if generation:
        report['llamacpp_generation_tokens_per_second'] = generation[0]['avg_ts']
    if prompt:
        report['llamacpp_prompt_tokens_per_second'] = prompt[0]['avg_ts']
    if args.frusium:
        saved = json.loads(args.frusium.read_text(encoding='utf-8'))
        steps = saved.get('decode_seconds') or []
        if steps:
            report['frusium_generation_tokens_per_second'] = len(steps) / sum(steps)
            report['frusium_source'] = str(args.frusium)
        if generation and steps:
            report['llamacpp_over_frusium'] = report['llamacpp_generation_tokens_per_second'] / report['frusium_generation_tokens_per_second']
    report['note'] = ('Different prompts and token counts; a throughput comparison, not an output comparison. '
                      'If the expected fields are missing, read llama_bench_runs directly.')
    print(json.dumps(report, indent=2))
    if args.save:
        with args.save.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)


if __name__ == '__main__':
    main()
