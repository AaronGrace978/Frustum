"""Check the engine beyond the 128-token sliding window.

Every earlier schedule test used three tokens. This one runs a real prompt
longer than the window, so the KV ring wraps and the compressed-attention path
(layers 2 and up) holds dozens of entries, then compares two schedules of the
same sequence:

  A. one prefill over all N tokens
  B. a prefill over the first N-k tokens, then k single-token decode steps
  C. (v0.9, --append-chunk M > 1) a prefill over the first N-k tokens, then the k
     tokens appended M at a time: the batched follow-up-prompt path

The last k positions must agree. Prefill uses the batched attention path and
decode the reference loop, so this also compares those two at scale. C is
compared with B, the schedule it replaces; it should differ from B only by the
last-bit rounding of matrix products whose result depends on the batch size.

    python long_context_check.py                    # 160 tokens, first 4 layers
    python long_context_check.py --tokens 300 --layers 8
    python long_context_check.py --tail 24 --append-chunk 8

Cost: prefill touches most routed experts of every layer it runs. Four layers
at 160 tokens read on the order of 15-20 GB per prefill, and there are two.
All 40 layers would read several hundred GB; raise --layers with that in mind.

Like the existing block tests this compares the implementation with itself.
It does not establish parity with the reference kernels.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from frustum_loader import Catalog, default_model_root
from layers0_3_block import GIB, SequentialBlocks
import placement

TEXT = ('The loader keeps only a small part of a very large model in memory at any moment. '
        'Each layer asks for the experts it needs, the cache decides what stays, and everything '
        'else is read back from the drive when it is wanted again. ')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=Path(default_model_root()))
    parser.add_argument('--tokens', type=int, default=160, help='Sequence length; above 128 to wrap the window')
    parser.add_argument('--tail', type=int, default=8, help='Positions produced by decode steps in schedule B')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--append-chunk', type=int, default=0,
                        help='Also run schedule C: append the tail this many tokens at a time (v0.9). 0 skips it')
    parser.add_argument('--dense-cache-mib', type=int, default=1024)
    parser.add_argument('--expert-cache-mib', type=int, default=4096)
    parser.add_argument('--output', type=Path, default=Path('long_context_results.json'))
    args = parser.parse_args()
    if not 16 <= args.tokens <= 512 or not 1 <= args.tail < args.tokens or args.output.exists():
        parser.error('Require 16 <= tokens <= 512, 1 <= tail < tokens, and a new --output file')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    free, total = torch.cuda.mem_get_info()
    if total < 14 * GIB or free < 8 * GIB:
        raise SystemExit('Insufficient dedicated VRAM/headroom')
    torch.cuda.set_per_process_memory_fraction(12 * GIB / total)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    catalog = Catalog(args.model)
    cfg = catalog.config['text_config']
    if not 1 <= args.layers <= cfg['num_hidden_layers']:
        parser.error('Invalid --layers')
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    ids = tokenizer.encode('<｜begin▁of▁sentence｜>' + TEXT * 14, add_special_tokens=False).ids[:args.tokens]
    if len(ids) < args.tokens:
        raise SystemExit('Built-in text is too short for this token count')
    router_mib = -(-placement.measure(catalog)['router'] // 2**20) + 8
    engine = SequentialBlocks(catalog, args.tokens, args.dense_cache_mib, args.expert_cache_mib, args.layers,
                              dict(router_cache_mib=router_mib))
    token_ids = torch.tensor([ids], dtype=torch.long, device='cuda')
    embeds = engine.lookup.get('embed.weight', ids).unsqueeze(0)
    h = embeds.unsqueeze(2).repeat(1, 1, cfg['hc_mult'], 1)
    split = args.tokens - args.tail
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def progress(label):
        return lambda done, count: print(f'{label}: layer {done}/{count}', flush=True)

    try:
        full, full_mix, records, _ = engine.forward(token_ids, h, 0, progress=progress('full prefill'))
        full_seconds = time.perf_counter() - started
        full_tail, full_mix_tail = full[:, split:].float().clone(), full_mix[:, split:].clone()
        unique = [record['unique_routed_experts'] for record in records]
        del full, full_mix
        engine.forward(token_ids[:, :split], h[:, :split], 0, progress=progress('prefix prefill'))
        outputs, mixes = [], []
        for position in range(split, args.tokens):
            last, mix, _, _ = engine.forward(token_ids[:, position:position + 1], h[:, position:position + 1], position)
            outputs.append(last.float())
            mixes.append(mix)
            print(f'decode position {position + 1}/{args.tokens}', flush=True)
        incremental, incremental_mix = torch.cat(outputs, 1), torch.cat(mixes, 1)
        norm = torch.linalg.vector_norm
        appended = None
        if args.append_chunk > 1:
            engine.forward(token_ids[:, :split], h[:, :split], 0, progress=progress('prefix prefill (C)'))
            outputs, mixes = [], []
            for begin in range(split, args.tokens, args.append_chunk):
                end = min(begin + args.append_chunk, args.tokens)
                piece, mix, _, _ = engine.forward(token_ids[:, begin:end], h[:, begin:end], begin)
                outputs.append(piece.float())
                mixes.append(mix)
                print(f'appended positions {begin + 1}-{end}/{args.tokens}', flush=True)
            batched, batched_mix = torch.cat(outputs, 1), torch.cat(mixes, 1)
            appended = dict(
                append_chunk=args.append_chunk,
                relative_L2_vs_decode=float(norm(batched - incremental) / norm(incremental).clamp_min(1e-12)),
                relative_L2_vs_decode_per_position=[
                    float(norm(batched[:, i] - incremental[:, i]) / norm(incremental[:, i]).clamp_min(1e-12))
                    for i in range(args.tail)],
                max_abs_difference_vs_decode=float((batched - incremental).abs().max()),
                bitwise_equal_to_decode=bool(torch.equal(batched, incremental)),
                max_abs_residual_mix_difference_vs_decode=float((batched_mix - incremental_mix).abs().max()),
                relative_L2_vs_full_prefill=float(norm(batched - full_tail) / norm(full_tail).clamp_min(1e-12)),
                passed=bool(torch.isfinite(batched).all()
                            and torch.allclose(incremental, batched, rtol=.02, atol=.01)
                            and torch.allclose(incremental_mix, batched_mix, rtol=.02, atol=.01)))
        per_position = [float(norm(full_tail[:, i] - incremental[:, i]) / norm(full_tail[:, i]).clamp_min(1e-12))
                        for i in range(args.tail)]
        relative = float(norm(full_tail - incremental) / norm(full_tail).clamp_min(1e-12))
        mix_gap = float((full_mix_tail - incremental_mix).abs().max())
        # Same tolerances as the saved three-token schedule test in layers0_3_block.py.
        passed = (bool(torch.isfinite(incremental).all())
                  and torch.allclose(full_tail, incremental, rtol=.02, atol=.01)
                  and torch.allclose(full_mix_tail, incremental_mix, rtol=.02, atol=.01)
                  and (appended is None or appended['passed']))
        window = cfg['sliding_window']
        report = dict(
            status='PASS' if passed else 'FAIL',
            scope=f'Blocks 0-{args.layers - 1}, {args.tokens} tokens, window {window}; two schedules of this implementation',
            tokens=args.tokens, decode_positions=args.tail, layers=args.layers,
            window_wrapped=args.tokens > window,
            compressed_entries_at_end={str(ratio): args.tokens // ratio
                                       for ratio in sorted(set(cfg['compress_ratios'][:args.layers])) if ratio},
            relative_L2_tail=relative, relative_L2_per_position=per_position,
            max_abs_difference=float((full_tail - incremental).abs().max()),
            max_abs_residual_mix_difference=mix_gap,
            tolerance=dict(rtol=.02, atol=.01),
            batched_append=appended,
            unique_routed_experts_per_layer_full_prefill=unique,
            full_prefill_seconds=full_seconds, total_seconds=time.perf_counter() - started,
            attention_path=dict(prefill='batched', decode='reference loop'),
            expert_cache=engine.moe.cache.stats(), dense_cache=engine.ops.loader.stats(),
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / GIB,
            peak_reserved_GiB=torch.cuda.max_memory_reserved() / GIB,
            caveat='Self-consistency only. Not parity with the reference kernels, and not a test of layers beyond --layers.')
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
    finally:
        engine.close()
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
