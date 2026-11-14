"""Run bounded, text-only chat with the local DeepSeek-V4.1-Flash checkpoint."""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from block_primitives import hc_pre
from frusium_loader import Catalog, RangeReader
from layers0_3_block import GIB, SequentialBlocks


class OutputHead:
    def __init__(self, tensor, resident=False, rows_per_chunk=4096):
        self.tensor = tensor
        self.resident = resident
        self.rows_per_chunk = rows_per_chunk
        self.reader = RangeReader()
        self.weights = None

    def read_chunk(self, start, count):
        head = self.tensor
        dim = head.shape[1]
        payload = bytearray(count * dim * 2)
        chunk = replace(head, offset=head.offset + start * dim * 2, size=len(payload))

        def copy(offset, source):
            payload[offset:offset + len(source)] = source

        self.reader.copy(chunk, copy)
        return torch.frombuffer(payload, dtype=torch.bfloat16).reshape(count, dim)

    def logits(self, x):
        rows, dim = self.tensor.shape
        if self.tensor.dtype != 'BF16' or dim != x.numel() or self.tensor.size != rows * dim * 2:
            raise ValueError('Unexpected output head layout')
        starts = range(0, rows, self.rows_per_chunk)
        if self.resident and self.weights is None:
            if x.is_cuda and torch.cuda.mem_get_info(x.device)[0] < self.tensor.size + 2 * GIB:
                raise RuntimeError('Insufficient GPU headroom for resident output head. Use --output-head streamed.')
            weights = torch.empty((rows, dim), dtype=torch.bfloat16, device=x.device)
            for start in starts:
                end = min(start + self.rows_per_chunk, rows)
                weights[start:end].copy_(self.read_chunk(start, end - start))
            self.weights = weights
        scores = []
        for start in starts:
            end = min(start + self.rows_per_chunk, rows)
            weight = self.read_chunk(start, end - start) if self.weights is None else self.weights[start:end]
            scores.append(F.linear(x, weight.to(x.device, dtype=torch.float32)))
        return torch.cat(scores)


def chat_prompt(history, prompt):
    text = '<｜begin▁of▁sentence｜>'
    for user, reply in history:
        text += f'<｜User｜>{user}<｜Assistant｜></think>{reply}<｜end▁of▁sentence｜>'
    return text + f'<｜User｜>{prompt}<｜Assistant｜></think>'


@torch.inference_mode()
def generate(engine, tokenizer, head, prompt, history, max_new_tokens, profile_decode=False):
    ids = tokenizer.encode(chat_prompt(history, prompt), add_special_tokens=False).ids
    if len(ids) + max_new_tokens > engine.capacity:
        raise ValueError('Conversation exceeds the configured context capacity')
    eos = tokenizer.token_to_id('<｜end▁of▁sentence｜>')
    output = []
    timings = []
    start = 0
    for _ in range(max_new_tokens):
        step_started = time.perf_counter()
        inputs = torch.tensor([ids], device='cuda', dtype=torch.long)
        embeds = engine.lookup.get('embed.weight', ids)[None, :, None, :]
        hidden = embeds.repeat(1, 1, engine.cfg['hc_mult'], 1)
        def progress(done, total):
            if done % 10 == 0 or done == total:
                print(f'Processed {done}/{total} layers for token {len(output) + 1}', flush=True)

        layer_profile = []
        readers = {
            'dense': engine.ops.loader.reader,
            'experts': engine.moe.cache.reader,
            'router': engine.moe.router.reader,
            'rows': engine.lookup.reader,
        }
        layer_started = time.perf_counter()
        previous = {name: reader.bytes_read for name, reader in readers.items()}

        def profile_progress(done, total):
            nonlocal layer_started, previous
            torch.cuda.synchronize()
            ended = time.perf_counter()
            current = {name: reader.bytes_read for name, reader in readers.items()}
            layer_profile.append(dict(layer=done - 1, seconds=ended - layer_started,
                                      logical_bytes={name: value - previous[name] for name, value in current.items()}))
            previous = current
            progress(done, total)
            layer_started = time.perf_counter()

        callback = profile_progress if profile_decode and start else progress
        hidden, mix, records, _ = engine.forward(inputs, hidden, start, progress=callback)
        if layer_profile:
            for measurement, record in zip(layer_profile, records, strict=True):
                measurement['selected_experts'] = record['selected_experts']
        torch.cuda.synchronize()
        head_started = time.perf_counter()
        head_bytes_before = head.reader.bytes_read
        x = engine.ops.norm(hc_pre(hidden[:, -1:], mix[:, -1:]), 'norm').float().flatten()
        logits = head.logits(x)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Output head produced nonfinite logits')
        token = logits.argmax().item()
        ended = time.perf_counter()
        timings.append(dict(phase='prefill' if start == 0 else 'decode', token_id=token,
                            backbone_seconds=head_started - step_started,
                            output_head_seconds=ended - head_started,
                            total_seconds=ended - step_started,
                            head_logical_bytes=head.reader.bytes_read - head_bytes_before))
        if layer_profile:
            timings[-1]['layers'] = layer_profile
        if token == eos:
            break
        output.append(token)
        print(f'Generated token {len(output)}/{max_new_tokens} in {ended - step_started:.1f}s '
              f'(head {ended - head_started:.2f}s)', flush=True)
        start += len(ids)
        ids = [token]
    return tokenizer.decode(output, skip_special_tokens=True), output, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--prompt', help='Run one prompt and exit. Omit for interactive chat.')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--output-head', choices=('resident', 'streamed'), default='streamed')
    parser.add_argument('--profile-decode', action='store_true',
                        help='Synchronize each decode layer and record time, logical reads and routing in evidence')
    parser.add_argument('--expert-cache-mib', type=int, default=256,
                        help='GPU budget for streamed experts across all layers')
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens < args.context <= 512:
        parser.error('Require 1 <= max-new-tokens < context <= 512')
    if args.evidence and (args.prompt is None or args.evidence.exists()):
        parser.error('--evidence requires --prompt and a new output file')
    if args.profile_decode and (not args.evidence or args.max_new_tokens < 2):
        parser.error('--profile-decode requires --evidence and at least two generated tokens')
    if not 128 <= args.expert_cache_mib <= 6144:
        parser.error('--expert-cache-mib must be between 128 and 6144')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    free, total = torch.cuda.mem_get_info()
    if total < 14 * GIB or free < 3 * GIB:
        raise SystemExit('Insufficient dedicated VRAM/headroom')
    torch.cuda.set_per_process_memory_fraction(12 * GIB / total)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    catalog = Catalog(args.model)
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    head = OutputHead(catalog.tensors['head.weight'], resident=args.output_head == 'resident')
    engine = SequentialBlocks(catalog, args.context, 256, args.expert_cache_mib,
                               catalog.config['text_config']['num_hidden_layers'])
    history = []
    try:
        while True:
            if args.prompt is not None:
                prompt = args.prompt
            else:
                try:
                    prompt = input('You: ')
                except (EOFError, KeyboardInterrupt):
                    break
                if prompt.strip() == '/quit':
                    break
                if not prompt.strip():
                    continue
            print('Generating with the local checkpoint...', flush=True)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                reply, tokens, timings = generate(engine, tokenizer, head, prompt, history,
                                                  args.max_new_tokens, args.profile_decode)
            except ValueError as error:
                if args.prompt is not None:
                    raise
                print(error)
                continue
            print(f'Assistant: {reply}', flush=True)
            if args.evidence:
                with args.evidence.open('x', encoding='utf-8') as handle:
                    json.dump(dict(prompt=prompt, reply=reply, token_ids=tokens,
                                   layers=len(engine.blocks), seconds=time.perf_counter() - started,
                                   peak_allocated_GiB=torch.cuda.max_memory_allocated() / GIB,
                                   peak_reserved_GiB=torch.cuda.max_memory_reserved() / GIB,
                                   output_head_mode=args.output_head,
                                   expert_cache_mib=args.expert_cache_mib,
                                   output_head_logical_bytes=sum(t['head_logical_bytes'] for t in timings),
                                   token_timings=timings), handle, indent=2)
            if args.prompt is not None:
                break
            history.append((prompt, reply))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
