"""Compare Frustum_loader's next-token distributions with a reference.

Why: every correctness check so far compares this implementation with itself
or with NumPy re-derivations of single pieces. Whether the whole 40-layer
stack matches the published model is still open. This script measures it.

1. Record what this implementation predicts (GPU needed):

    python frustum_chat.py --prompt "Say hello." --max-new-tokens 12 \
        --top-logprobs 10 --evidence ours.json

2. Compare with one of:

   another Frustum run (e.g. --legacy, a different placement, another machine):

    python reference_parity.py ours.json --against other.json

   a provider that serves the same checkpoint through an OpenAI-compatible API.
   The key is read from the FRUSTUM_API_KEY environment variable.

    python reference_parity.py ours.json --api-base https://host/v1 --api-model NAME
    python reference_parity.py ours.json --api-base https://host/v1 --api-model NAME --endpoint completions

   `completions` sends the exact prompt string this implementation tokenized
   and is the cleaner comparison where a provider offers it. `chat` lets the
   provider apply its own chat template, so a template difference shows up as
   a mismatch from the first token. Providers may also serve a different
   quantization or revision. Treat a mismatch as a question to investigate,
   not as proof of a bug on either side.

Output: per-step agreement of the greedy token, overlap of the top-K sets and
the largest log-probability gap on shared tokens.
"""
import argparse
import json
import os
from pathlib import Path
import urllib.request

from frustum_loader import default_model_root


def ours_as_text(evidence, tokenizer):
    steps = []
    for step in evidence.get('top_logprobs') or []:
        steps.append(dict(token=tokenizer.decode([step['token_id']], skip_special_tokens=False),
                          top={tokenizer.decode([token], skip_special_tokens=False): logprob
                               for token, logprob in step['top']}))
    if not steps:
        raise SystemExit('Evidence has no top_logprobs. Re-run frustum_chat.py with --top-logprobs K.')
    return steps


def fetch(args, evidence, count):
    key = os.environ.get('FRUSTUM_API_KEY')
    if not key:
        raise SystemExit('Set FRUSTUM_API_KEY to call a provider.')
    k = len(evidence['top_logprobs'][0]['top'])
    if args.endpoint == 'chat':
        body = dict(model=args.api_model, messages=[dict(role='user', content=evidence['prompt'])],
                    max_tokens=count, temperature=0, logprobs=True, top_logprobs=k)
        path = '/chat/completions'
    else:
        body = dict(model=args.api_model, prompt=evidence['prompt_text'], max_tokens=count,
                    temperature=0, logprobs=k)
        path = '/completions'
    request = urllib.request.Request(args.api_base.rstrip('/') + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    choice = payload['choices'][0]
    logprobs = choice.get('logprobs')
    if not logprobs:
        raise SystemExit('The provider returned no logprobs; it may not support them for this model.')
    if args.endpoint == 'chat':
        steps = [dict(token=item['token'], top={alt['token']: alt['logprob'] for alt in item['top_logprobs']})
                 for item in logprobs['content']]
    else:
        steps = [dict(token=token, top=dict(top)) for token, top in zip(logprobs['tokens'], logprobs['top_logprobs'])]
    return steps, payload


def compare(ours, theirs):
    rows = []
    for index, (a, b) in enumerate(zip(ours, theirs)):
        shared = set(a['top']) & set(b['top'])
        rows.append(dict(step=index, ours=a['token'], reference=b['token'], same_token=a['token'] == b['token'],
                         top_overlap=round(len(shared) / max(len(a['top']), 1), 3),
                         max_logprob_gap=round(max((abs(a['top'][t] - b['top'][t]) for t in shared), default=float('nan')), 4)))
        if a['token'] != b['token']:
            break  # after the first divergence the two runs condition on different text
    agree = sum(row['same_token'] for row in rows)
    return dict(steps_compared=len(rows), greedy_tokens_matching=agree,
                first_divergence=next((row['step'] for row in rows if not row['same_token']), None),
                mean_top_overlap=round(sum(row['top_overlap'] for row in rows) / max(len(rows), 1), 3),
                steps=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ours', type=Path)
    parser.add_argument('--against', type=Path, help='Another frustum_chat.py evidence file')
    parser.add_argument('--api-base')
    parser.add_argument('--api-model')
    parser.add_argument('--endpoint', choices=('chat', 'completions'), default='chat')
    parser.add_argument('--model', type=Path, default=Path(default_model_root()), help='Folder with tokenizer.json')
    parser.add_argument('--save', type=Path, help='Write the comparison (and raw API response) here')
    args = parser.parse_args()
    if bool(args.against) == bool(args.api_base):
        parser.error('Give exactly one of --against or --api-base')
    if args.api_base and not args.api_model:
        parser.error('--api-base requires --api-model')
    if args.save and args.save.exists():
        parser.error('--save needs a new file')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    evidence = json.loads(args.ours.read_text(encoding='utf-8'))
    ours = ours_as_text(evidence, tokenizer)
    raw = None
    if args.against:
        other = json.loads(args.against.read_text(encoding='utf-8'))
        if other.get('prompt_token_ids') != evidence.get('prompt_token_ids'):
            raise SystemExit('The two evidence files were produced from different prompts.')
        theirs, reference = ours_as_text(other, tokenizer), str(args.against)
    else:
        theirs, raw = fetch(args, evidence, len(ours))
        reference = f'{args.api_base} {args.api_model} ({args.endpoint})'
    result = dict(prompt=evidence['prompt'], reference=reference, **compare(ours, theirs))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.save:
        with args.save.open('x', encoding='utf-8') as handle:
            json.dump(dict(result, raw_reference_response=raw), handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
