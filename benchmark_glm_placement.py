"""Compare GLM cache policies and validate warm conversation prefix reuse."""
import argparse
import gc
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from tokenizers import Tokenizer
import torch

from glm_chat import EOS, GIB, Glm, glm_prompt, prompt_suffix


def run_turn(model, tokenizer, messages, limit, *, reuse, label):
    ids = tokenizer.encode(glm_prompt(messages), add_special_tokens=False).ids
    suffix, reused = prompt_suffix(model, ids, reuse)
    before = model.metrics()
    output, times = [], []
    for index in range(limit):
        started = perf_counter()
        logits = model.forward(suffix)
        if not torch.isfinite(logits).all().item():
            raise RuntimeError(f'Nonfinite logits during {label}')
        token = logits.argmax().item()
        times.append(perf_counter() - started)
        if index == 0:
            print(f'{label}: first token in {times[0]:.2f}s ({reused} tokens reused)', flush=True)
        if token in EOS:
            break
        output.append(token)
        suffix = [token]
    after = model.metrics()
    counters = {name: after[name] - before[name] for name in before
                if name.endswith(('_hits', '_seconds')) or name in ('file_reads', 'file_bytes', 'transfer_bytes')}
    return dict(reply=tokenizer.decode(output), token_ids=output, stopped_at_eos=len(output) < len(times),
                prompt_tokens=len(ids), reused_prompt_tokens=reused,
                time_to_first_token_seconds=times[0], mean_decode_seconds=mean(times[1:]) if len(times) > 1 else None,
                total_seconds=sum(times), step_seconds=times, **counters)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(r'G:\Models\GLM-4.7-Flash\zai-org_GLM-4.7-Flash-Q4_K_M.gguf'))
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--order', default='baseline,split,tiered')
    args = parser.parse_args()
    modes = {'baseline': (False, 0), 'split': (True, 0), 'tiered': (True, 8 * GIB)}
    order = args.order.split(',')
    if args.evidence.exists() or not order or any(mode not in modes for mode in order):
        parser.error('Use a fresh evidence path and an order containing baseline, split, or tiered')
    free, total = torch.cuda.mem_get_info()
    budget = min(9 * GIB, free - 2 * GIB)
    if budget < 4 * GIB:
        raise RuntimeError('Insufficient free GPU memory for comparison')
    torch.cuda.set_per_process_memory_fraction(min(12 * GIB / total, 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = Tokenizer.from_file(str(args.model.with_name('tokenizer.json')))
    prompt = 'Do you like Dino Buddies? 🦖 Reply in one short sentence.'
    reference = json.loads(Path(__file__).with_name('glm_chat_results.json').read_text(encoding='utf-8'))
    report = dict(prompt=prompt, followup='Reply with only the word Dino.',
                  scope='One run per policy per listed occurrence. Fresh GPU and application RAM caches per run; OS file cache uncontrolled.',
                  gpu=torch.cuda.get_device_name(), gpu_cache_budget_bytes=budget, runs=[])
    for mode in order:
        print(f'Benchmarking {mode}', flush=True)
        torch.cuda.reset_peak_memory_stats()
        split, ram_budget = modes[mode]
        model = Glm(args.model, budget, split=split, ram_budget=ram_budget)
        first = run_turn(model, tokenizer, [dict(role='user', content=prompt)], 24,
                         reuse=False, label=f'{mode}/first')
        if first['token_ids'] != reference['token_ids'] or not first['stopped_at_eos']:
            raise AssertionError(f'{mode} did not reproduce the saved GLM response')
        warm = run_turn(model, tokenizer, [dict(role='user', content=prompt)], 24,
                        reuse=False, label=f'{mode}/warm')
        chat = [dict(role='user', content=prompt), dict(role='assistant', content=warm['reply']),
                dict(role='user', content=report['followup'])]
        reused = run_turn(model, tokenizer, chat, 3, reuse=True, label=f'{mode}/reuse')
        rebuilt = run_turn(model, tokenizer, chat, 3, reuse=False, label=f'{mode}/rebuild')
        if warm['token_ids'] != reference['token_ids'] or reused['token_ids'] != rebuilt['token_ids']:
            raise AssertionError(f'{mode} token parity failed')
        report['runs'].append(dict(mode=mode, initial=first, warm=warm, followup_reuse=reused,
                                   followup_rebuild=rebuilt, ram_budget_bytes=model.ram_budget,
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   final_cache=model.metrics()))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with args.evidence.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f'Saved {args.evidence}', flush=True)


if __name__ == '__main__':
    main()
