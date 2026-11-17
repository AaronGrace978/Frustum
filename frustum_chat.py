"""Run bounded, text-only chat with the local DeepSeek-V4.1-Flash checkpoint.

v0.5 changes, none of which alter model arithmetic:
- A placement plan keeps always-used weights (dense, routers, shared experts,
  output head) resident on the GPU when they fit, instead of re-reading them
  for every token. See placement.py. `--legacy` restores the v0.4 behaviour.
- Routed experts go through a GPU tier, an optional host-RAM tier and threaded
  disk reads (expert_cache.py).
- A follow-up turn reuses the attention state of the previous turn when the
  new prompt starts with exactly the tokens already processed.
- Tokens stream as they are produced; sampling is available (greedy default).
"""
import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from block_primitives import hc_pre
from frustum_loader import Catalog, RangeReader, default_model_root
from generation import chat_prompt, drive, reusable_prefix  # chat_prompt re-exported for callers
from layers0_3_block import GIB, SequentialBlocks
import fused_kernels
import os_cache
import placement
from spill_tier import SpillTier


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
        chunk = replace(head, offset=head.offset + start * dim * 2, size=count * dim * 2)
        return torch.frombuffer(self.reader.read(chunk), dtype=torch.bfloat16).reshape(count, dim)

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


def make_sampler(temperature=0.0, top_k=0, top_p=1.0, seed=None):
    """Greedy when temperature is 0 (the validated path); otherwise temperature/top-k/top-p sampling."""
    if temperature <= 0:
        return lambda logits: int(logits.argmax().item())
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    def pick(logits):
        scores = logits.float().cpu() / temperature
        if top_k:
            kth = scores.topk(min(top_k, scores.numel())).values[-1]
            scores = scores.masked_fill(scores < kth, float('-inf'))
        probs = scores.softmax(-1)
        if top_p < 1:
            ordered, order = probs.sort(descending=True)
            keep = ordered.cumsum(-1) - ordered < top_p
            probs = torch.zeros_like(probs).scatter_(0, order, ordered * keep)
        return int(torch.multinomial(probs, 1, generator=generator).item())
    return pick


DEFAULT_APPEND_CHUNK = 32


class Session:
    """Engine plus the exact token sequence its attention state currently covers.

    append_chunk (v0.9): new prompt tokens fed per pass on a follow-up turn. Default:
    FRUSTUM_APPEND_CHUNK, else 32: on the target machine a 29-token follow-up took 123.7 s one
    token per pass, 87.7 s with 8 and 60.1 s with 32 (bench_runs/v09_append). 1 restores the v0.8
    one-token-per-pass behaviour."""
    def __init__(self, engine, tokenizer, head, reuse=True, append_chunk=None):
        self.engine, self.tokenizer, self.head, self.reuse = engine, tokenizer, head, reuse
        self.eos = tokenizer.token_to_id('<｜end▁of▁sentence｜>')
        self.fed = []
        if append_chunk is None:
            append_chunk = int(os.environ.get('FRUSTUM_APPEND_CHUNK', DEFAULT_APPEND_CHUNK))
        if append_chunk < 1:
            raise ValueError('append_chunk must be at least 1')
        self.append_chunk = append_chunk

    def readers(self):
        engine = self.engine
        return {'dense': engine.ops.loader.reader, 'experts': engine.moe.cache.reader,
                'router': engine.moe.router.reader, 'rows': engine.lookup.reader}

    def disk_bytes(self):
        return sum(reader.bytes_read for reader in self.readers().values()) + self.head.reader.bytes_read


@torch.inference_mode()
def generate(session, ids, max_new_tokens, pick, on_token=None, profile_decode=False, verbose=False,
             trace_routing=False):
    """Generate up to max_new_tokens after the prompt token ids.

    Returns (token_ids, timings, reused_prompt_tokens, stopped_at_eos).
    """
    engine, head = session.engine, session.head
    if len(ids) + max_new_tokens > engine.capacity:
        raise ValueError('Conversation exceeds the configured context capacity')
    readers = session.readers()
    def run(tokens, phase, want_logits):
        start = 0 if phase == 'prefill' else len(session.fed)
        step_started = time.perf_counter()
        inputs = torch.tensor([tokens], device='cuda', dtype=torch.long)
        embeds = engine.lookup.get('embed.weight', tokens)[None, :, None, :]
        hidden = embeds.repeat(1, 1, engine.cfg['hc_mult'], 1)

        def progress(done, total):
            if (verbose or phase == 'prefill') and (done % 10 == 0 or done == total):
                print(f'  {phase}: layer {done}/{total}', flush=True)

        layer_profile = []
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
        logits = None
        if want_logits:
            x = engine.ops.norm(hc_pre(hidden[:, -1:], mix[:, -1:]), 'norm').float().flatten()
            logits = head.logits(x)
            if not torch.isfinite(logits).all():
                raise RuntimeError('Output head produced nonfinite logits')
        ended = time.perf_counter()
        timing = dict(phase=phase, token_id=None, input_tokens=len(tokens),
                      backbone_seconds=head_started - step_started,
                      output_head_seconds=ended - head_started,
                      total_seconds=ended - step_started,
                      head_logical_bytes=head.reader.bytes_read - head_bytes_before)
        if layer_profile:
            timing['layers'] = layer_profile
        # v0.7: expert I/O for this step (no synchronisation, so it does not perturb timing).
        timing['io'] = engine.moe.cache.ledger.step()
        if trace_routing:
            timing['routing'] = [record['selected_experts'] for record in records]
        return logits, timing

    reused = reusable_prefix(session.fed, ids) if session.reuse else 0
    if reused:
        print(f'  reusing {reused} processed tokens, adding {len(ids) - reused}', flush=True)
    return drive(session.fed, ids, max_new_tokens, session.eos, pick, run, session.reuse, on_token,
                 append_chunk=getattr(session, 'append_chunk', 1))


def build_plan(catalog, args, total, free):
    cap = int(args.gpu_cap_gib * GIB)
    notes = []
    if free - int(1.5 * GIB) < cap:
        cap = free - int(1.5 * GIB)
        notes.append(f'Only {free / GIB:.1f} GiB of VRAM is free; GPU cap lowered to {cap / GIB:.1f} GiB.')
    if cap < 6 * GIB:
        raise SystemExit('Insufficient dedicated VRAM/headroom')
    if args.legacy:
        sizes = placement.measure(catalog)
        head = 'streamed' if args.output_head == 'auto' else args.output_head
        p = dict(gpu_cap_bytes=cap, reserve_bytes=0, dense='streamed', dense_cache_mib=256, head=head,
                 head_bytes=sizes['head'] if head == 'resident' else 0, router_cache_mib=32,
                 expert_cache_mib=args.expert_cache_mib or 256, pin_shared=False, gpu_policy='lru',
                 host_policy='lru', host_cache_mib=0, sizes=sizes, notes=['Legacy v0.4 placement.'])
    else:
        try:
            p = placement.plan(catalog, cap, reserve_bytes=int(args.reserve_gib * GIB), dense=args.dense,
                               head=args.output_head, expert_cache_mib=args.expert_cache_mib,
                               host_cache_gib=args.host_cache_gib, unbuffered=args.reader == 'unbuffered')
        except ValueError as error:
            raise SystemExit(str(error))
        if args.cache_policy != 'auto':
            p['gpu_policy'] = args.cache_policy
    p['notes'] = notes + p['notes']
    return p


def fused_report():
    fused = fused_kernels.get(torch)
    return dict(enabled=fused.enabled, fused_calls=fused.calls, checked_shapes=len(fused.checked),
                off_reason=fused.reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=Path(default_model_root()))
    parser.add_argument('--prompt', help='Run one prompt and exit. Omit for interactive chat.')
    parser.add_argument('--system', help='Optional system message placed at the start of the conversation')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--output-head', choices=('auto', 'resident', 'streamed'), default='auto')
    parser.add_argument('--dense', choices=('auto', 'resident', 'streamed'), default='auto',
                        help='Keep dense weights on the GPU (resident) or re-read them per token (streamed)')
    parser.add_argument('--profile-decode', action='store_true',
                        help='Synchronize each decode layer and record time, logical reads and routing in evidence')
    parser.add_argument('--expert-cache-mib', type=int,
                        help='GPU budget for packed experts. Default: whatever the placement plan leaves')
    parser.add_argument('--host-cache-gib', default='auto',
                        help='RAM tier for packed experts behind the GPU cache: a number, 0 to disable, or auto')
    parser.add_argument('--gpu-cap-gib', type=float, default=12.0, help='PyTorch allocator cap')
    parser.add_argument('--reserve-gib', type=float, default=2.0,
                        help='Part of the cap kept free for activations and decode temporaries')
    parser.add_argument('--io-threads', type=int, default=4, help='Background disk readers for experts; 0 disables')
    parser.add_argument('--cache-policy', choices=('auto', 'lfu', 'lru'), default='auto',
                        help='GPU expert eviction. auto picks lfu when the cache is smaller than one token needs')
    parser.add_argument('--temperature', type=float, default=0.0, help='0 = greedy (default)')
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--top-logprobs', type=int, default=0,
                        help='Record the K most likely tokens and their log-probabilities per step in --evidence')
    parser.add_argument('--no-reuse', action='store_true', help='Re-prefill the whole conversation every turn')
    parser.add_argument('--append-chunk', type=int,
                        help='New prompt tokens per pass on a follow-up turn (v0.9). Default: FRUSTUM_APPEND_CHUNK, '
                             'else 32; 1 feeds them one at a time as v0.8 did')
    parser.add_argument('--followup', action='append', metavar='TEXT',
                        help='With --prompt: another user turn to run after it in the same conversation (repeatable). '
                             '--evidence then describes the last turn and lists every turn under "turns"')
    parser.add_argument('--verbose', action='store_true', help='Per-token timing lines instead of streamed text')
    parser.add_argument('--no-fused', action='store_true',
                        help='Decode steps use the reference decode-then-multiply path instead of fused kernels')
    parser.add_argument('--predict-next', type=int,
                        help='Experts of the next layer to start reading early during decode (0 disables). '
                             'Default: FRUSTUM_PREDICT_NEXT, else 0 (off)')
    parser.add_argument('--predict-mode', choices=('shadow', 'gated', 'always'),
                        help='What --predict-next does with its guesses: shadow scores them without reading, '
                             'gated (default) reads only while they are measurably right, always is v0.6')
    parser.add_argument('--trace-routing', action='store_true',
                        help='Record the routed experts of every step in --evidence (for decode_cache_replay.py '
                             'and roofline.py). Unlike --profile-decode it adds no synchronisation')
    parser.add_argument('--spill-dir', type=Path,
                        help='Folder on a faster local drive for verified exact copies of experts already read '
                             '(spill_tier.py). Off unless given')
    parser.add_argument('--spill-gib', type=float, default=64.0, help='Byte budget for --spill-dir')
    parser.add_argument('--ram-floor-gib', type=float, default=3.0,
                        help='Shrink the host RAM tier when free system RAM falls below this; 0 disables')
    parser.add_argument('--reader', choices=('direct', 'unbuffered'),
                        help='How checkpoint bytes are read. direct goes through the OS file cache; unbuffered '
                             '(Windows) bypasses it, so the host RAM tier can take most of the spare RAM. '
                             'Default: FRUSTUM_READER, else unbuffered on Windows (v0.8: 29%% faster decode in '
                             'bench_runs/v08_patch_vs_unbuffered_2), direct elsewhere')
    parser.add_argument('--legacy', action='store_true',
                        help='v0.4 behaviour for A/B runs: mmap reader, 256 MiB caches, streamed head, no reuse')
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens < args.context <= 512:
        parser.error('Require 1 <= max-new-tokens < context <= 512')
    if args.evidence and (args.prompt is None or args.evidence.exists()):
        parser.error('--evidence requires --prompt and a new output file')
    if args.profile_decode and (not args.evidence or args.max_new_tokens < 2):
        parser.error('--profile-decode requires --evidence and at least two generated tokens')
    if args.expert_cache_mib is not None and args.expert_cache_mib < 128:
        parser.error('--expert-cache-mib must be at least 128')
    if args.host_cache_gib != 'auto':
        try:
            if float(args.host_cache_gib) < 0:
                raise ValueError
        except ValueError:
            parser.error('--host-cache-gib takes a nonnegative number or auto')
    if args.temperature < 0 or args.top_k < 0 or not 0 < args.top_p <= 1 or args.io_threads < 0:
        parser.error('Require temperature >= 0, top-k >= 0, 0 < top-p <= 1, io-threads >= 0')
    if not 0 <= args.top_logprobs <= 20 or (args.top_logprobs and not args.evidence):
        parser.error('--top-logprobs takes 0..20 and requires --evidence')
    if args.gpu_cap_gib < 6 or args.reserve_gib < 0.5:
        parser.error('Require --gpu-cap-gib >= 6 and --reserve-gib >= 0.5')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    if args.predict_next is not None and args.predict_next < 0:
        parser.error('--predict-next takes 0 or more')
    if args.followup and args.prompt is None:
        parser.error('--followup requires --prompt')
    if args.append_chunk is not None and args.append_chunk < 1:
        parser.error('--append-chunk takes 1 or more')
    if args.legacy:
        if args.reader:
            parser.error('--reader cannot be combined with --legacy (which uses the mmap reader)')
        os.environ['FRUSTUM_READER'] = 'mmap'
    else:
        args.reader = (args.reader or os.environ.get('FRUSTUM_READER')
                       or ('unbuffered' if os.name == 'nt' else 'direct'))
        if args.reader not in ('direct', 'unbuffered'):
            parser.error('FRUSTUM_READER must be direct or unbuffered (or use --legacy for mmap)')
        if args.reader == 'unbuffered' and os.name != 'nt':
            print('Note: unbuffered reads are Windows-only; using direct reads.', flush=True)
            args.reader = 'direct'
        os.environ['FRUSTUM_READER'] = args.reader
    if args.legacy or args.no_fused:
        os.environ['FRUSTUM_FUSED'] = '0'
    free, total = torch.cuda.mem_get_info()
    catalog = Catalog(args.model)
    plan = build_plan(catalog, args, total, free)
    print(placement.describe(plan), flush=True)
    torch.cuda.set_per_process_memory_fraction(min(1.0, plan['gpu_cap_bytes'] / total))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    head = OutputHead(catalog.tensors['head.weight'], resident=plan['head'] == 'resident')
    moe_options = dict(host_cache_mib=plan['host_cache_mib'], router_cache_mib=plan['router_cache_mib'],
                       io_threads=0 if args.legacy else args.io_threads, gpu_policy=plan['gpu_policy'],
                       host_policy=plan['host_policy'], pin_shared=plan['pin_shared'],
                       cap_bytes=plan['gpu_cap_bytes'],
                       predict_next=0 if args.legacy else args.predict_next,
                       predict_mode=args.predict_mode)
    engine = SequentialBlocks(catalog, args.context, plan['dense_cache_mib'], plan['expert_cache_mib'],
                              catalog.config['text_config']['num_hidden_layers'], moe_options)
    if args.ram_floor_gib > 0 and not args.legacy:
        engine.moe.cache.ram_probe = placement.available_ram
        engine.moe.cache.ram_floor = int(args.ram_floor_gib * GIB)
    if args.spill_dir and not args.legacy:
        checkpoint_id = (str(catalog.root), catalog.index.get('metadata', {}).get('total_size'), len(catalog.tensors))
        engine.moe.cache.spill = SpillTier(args.spill_dir, int(args.spill_gib * GIB), checkpoint_id)
    device_counters = os_cache.DeviceCounters(args.model)
    session = Session(engine, tokenizer, head, reuse=not (args.no_reuse or args.legacy),
                      append_chunk=1 if args.legacy else args.append_chunk)
    scripted = None if args.prompt is None else [args.prompt] + list(args.followup or [])
    turns = []
    sample = make_sampler(args.temperature, args.top_k, args.top_p, args.seed)
    distributions = []

    def pick(logits):
        token = sample(logits)
        if args.top_logprobs:
            logprobs = logits.float().log_softmax(-1)
            best = logprobs.topk(args.top_logprobs)
            distributions.append(dict(token_id=token, logprob=float(logprobs[token]),
                                      top=[[int(i), float(v)] for i, v in zip(best.indices.tolist(), best.values.tolist())]))
        return token
    history = []
    try:
        while True:
            if scripted is not None:
                prompt = scripted.pop(0)
            else:
                try:
                    prompt = input('You: ')
                except (EOFError, KeyboardInterrupt):
                    break
                command = prompt.strip()
                if command == '/quit':
                    break
                if command == '/reset':
                    history.clear()
                    session.fed = []
                    print('Conversation cleared.')
                    continue
                if command == '/stats':
                    print(json.dumps(dict(expert_cache=engine.moe.cache.stats(), dense=engine.ops.loader.stats(),
                                          prediction=engine.moe.prediction_stats(), fused=fused_report(),
                                          memory=engine.memory()), indent=2))
                    continue
                if not command:
                    continue
            prompt_text = chat_prompt(history, prompt, args.system)
            ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
            distributions.clear()
            torch.cuda.reset_peak_memory_stats()
            cache = engine.moe.cache
            before = dict(disk=session.disk_bytes(), requests=cache.requests, gpu=cache.hits,
                          host=cache.host_hits, read=cache.disk_reads, spill=cache.spill_hits)
            cache.ledger.step()  # start this turn's I/O window
            physical_before = device_counters.read()
            shown, opened = 0, False

            def on_token(output, timing):
                nonlocal shown, opened
                if args.verbose:
                    print(f'Generated token {len(output)}/{args.max_new_tokens} in {timing["total_seconds"]:.1f}s '
                          f'(head {timing["output_head_seconds"]:.2f}s)', flush=True)
                    return
                text = tokenizer.decode(output, skip_special_tokens=True)
                if not text.endswith('\ufffd'):  # wait for the rest of a multi-token character
                    print(('' if opened else 'Assistant: ') + text[shown:], end='', flush=True)
                    shown, opened = len(text), True

            print('Generating with the local checkpoint...', flush=True)
            started = time.perf_counter()
            try:
                tokens, timings, reused, stopped = generate(session, ids, args.max_new_tokens, pick, on_token,
                                                            args.profile_decode, args.verbose, args.trace_routing)
            except ValueError as error:
                if args.prompt is not None:
                    raise
                print(error)
                continue
            seconds = time.perf_counter() - started
            reply = tokenizer.decode(tokens, skip_special_tokens=True)
            print(('' if opened else 'Assistant: ') + reply[shown:], flush=True)
            decode = [t['total_seconds'] for t in timings if t['phase'] == 'decode']
            first = sum(t['total_seconds'] for t in timings if t['phase'] != 'decode')
            requests = max(cache.requests - before['requests'], 1)
            summary = dict(
                generated_tokens=len(tokens), prompt_tokens=len(ids), reused_prompt_tokens=reused,
                stopped_at_eos=stopped, first_token_seconds=first,
                mean_decode_seconds=sum(decode) / len(decode) if decode else None,
                disk_bytes=session.disk_bytes() - before['disk'],
                expert_requests=cache.requests - before['requests'],
                expert_gpu_hits=cache.hits - before['gpu'], expert_host_hits=cache.host_hits - before['host'],
                expert_disk_reads=cache.disk_reads - before['read'],
                expert_spill_hits=cache.spill_hits - before['spill'])
            # v0.7 roofline inputs. Physical bytes are device-wide (other programs on the drive count).
            physical = os_cache.delta(physical_before, device_counters.read())
            decode_io = [t['io'] for t in timings if t['phase'] == 'decode' and 'io' in t]
            total = lambda name: sum(step[name] for step in decode_io)
            summary['physical_device'] = physical
            summary['physical_device_error'] = device_counters.error
            summary['physical_over_logical'] = (physical['bytes_read'] / summary['disk_bytes']
                                                if physical and summary['disk_bytes'] else None)
            if decode_io:
                busy, wall = total('busy_seconds'), sum(decode)
                summary['decode_io'] = dict(
                    steps=len(decode_io), expert_bytes_per_token=total('bytes_read') / len(decode_io),
                    unique_bytes_per_token=total('unique_bytes') / len(decode_io),
                    repeat_bytes_per_token=total('repeat_bytes') / len(decode_io),
                    drive_busy_seconds_per_token=busy / len(decode_io),
                    blocked_seconds_per_token=total('blocked_seconds') / len(decode_io),
                    drive_utilization=busy / wall if wall else None,
                    effective_MBps=total('bytes_read') / 1e6 / busy if busy else None,
                    mean_queue_depth=total('depth_seconds') / busy if busy else None,
                    overlap_fraction=(busy - total('busy_blocked_seconds')) / busy if busy else None)
            rate = f'{summary["mean_decode_seconds"]:.1f} s/token' if decode else 'no decode steps'
            print(f'[{len(tokens)} tokens, first after {first:.1f}s, then {rate}; '
                  f'{reused}/{len(ids)} prompt tokens reused; disk {summary["disk_bytes"] / 1e9:.2f} GB; experts '
                  f'{100 * summary["expert_gpu_hits"] / requests:.0f}% GPU, '
                  f'{100 * summary["expert_host_hits"] / requests:.0f}% RAM, '
                  f'{100 * summary["expert_disk_reads"] / requests:.0f}% disk]', flush=True)
            if summary.get('decode_io'):
                d = summary['decode_io']
                show = lambda value, form: 'n/a' if value is None else format(value, form)
                print(f'[decode I/O per token: {d["expert_bytes_per_token"] / 1e9:.2f} GB experts '
                      f'({d["repeat_bytes_per_token"] / 1e9:.2f} GB re-read), drive busy '
                      f'{show(d["drive_utilization"] and 100 * d["drive_utilization"], ".0f")}% of the time at '
                      f'{show(d["effective_MBps"], ".0f")} MB/s, queue {show(d["mean_queue_depth"], ".1f")}, '
                      f'physical/logical {show(summary["physical_over_logical"], ".2f")}]', flush=True)
            feed = [t for t in timings if t['phase'] == 'reuse_feed']
            summary['prompt_passes'] = len(feed) if reused else 1
            turns.append(dict(prompt=prompt, reply=reply, token_ids=tokens, prompt_token_ids=ids,
                              reused_prompt_tokens=reused, new_prompt_tokens=len(ids) - reused,
                              prompt_passes=summary['prompt_passes'], append_chunk=session.append_chunk,
                              first_token_seconds=first, mean_decode_seconds=summary['mean_decode_seconds'],
                              disk_bytes=summary['disk_bytes']))
            if args.evidence and (scripted is None or not scripted):
                with args.evidence.open('x', encoding='utf-8') as handle:
                    json.dump(dict(prompt=prompt, prompt_text=prompt_text, prompt_token_ids=ids,
                                   reply=reply, token_ids=tokens,
                                   layers=len(engine.blocks), seconds=seconds,
                                   peak_allocated_GiB=torch.cuda.max_memory_allocated() / GIB,
                                   peak_reserved_GiB=torch.cuda.max_memory_reserved() / GIB,
                                   output_head_mode=plan['head'],
                                   expert_cache_mib=plan['expert_cache_mib'],
                                   output_head_logical_bytes=sum(t['head_logical_bytes'] for t in timings),
                                   summary=summary,
                                   placement={k: v for k, v in plan.items() if k != 'sizes'},
                                   checkpoint_sizes=plan['sizes'],
                                   sampling=dict(temperature=args.temperature, top_k=args.top_k,
                                                 top_p=args.top_p, seed=args.seed),
                                   legacy=args.legacy, io_threads=moe_options['io_threads'],
                                   reader=os.environ.get('FRUSTUM_READER', 'direct'),
                                   expert_cache=cache.stats(), dense_cache=engine.ops.loader.stats(),
                                   prediction=engine.moe.prediction_stats(), fused=fused_report(),
                                   top_logprobs=distributions, append_chunk=session.append_chunk,
                                   turns=turns, token_timings=timings), handle, indent=2)
            if scripted is not None and not scripted:
                break
            history.append((prompt, reply))
    finally:
        engine.close()
        head.reader.close()


if __name__ == '__main__':
    main()
