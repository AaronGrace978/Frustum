"""Where does a DeepSeek decode token go? Per-section timing for frustum_chat.py (v0.8).

    python tools\\profile_decode.py --prompt "Say hello." --max-new-tokens 24 --predict-next 0

Takes frustum_chat.py's arguments and runs it normally, but during decode steps it wraps the
main sections (attention, hyper-connection mixes, Engram, MoE routing, expert waits/uploads,
expert compute, trace hashing) with torch.cuda.synchronize() on both sides and prints the
seconds per token spent in each. Prefill is not timed.

The synchronisation removes whatever CPU/GPU overlap the engine normally gets, so the total
runs a little slower than an untimed run, and the sections are an upper bound on each part's
share. Use it to find the bottleneck, not as a speed result. Diagnostic only: model output is
unchanged.
"""
import atexit
import collections
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import expert_cache  # noqa: E402
import layers0_3_block as blocks  # noqa: E402
import sparse_rows  # noqa: E402
import streamed_moe  # noqa: E402

TOTAL = 'TOTAL decode backbone'
seconds = collections.defaultdict(float)
calls = collections.Counter()
state = {'decode': False}


def timed(name, fn):
    def wrapper(*args, **kwargs):
        if not state['decode']:
            return fn(*args, **kwargs)
        torch.cuda.synchronize()
        began = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            seconds[name] += time.perf_counter() - began
            calls[name] += 1
    return wrapper


_forward = blocks.SequentialBlocks.forward


def forward(self, ids, h, start=0, *args, **kwargs):
    state['decode'] = start > 0
    if not state['decode']:
        return _forward(self, ids, h, start, *args, **kwargs)
    torch.cuda.synchronize()
    began = time.perf_counter()
    out = _forward(self, ids, h, start, *args, **kwargs)
    torch.cuda.synchronize()
    seconds[TOTAL] += time.perf_counter() - began
    calls[TOTAL] += 1
    return out


blocks.SequentialBlocks.forward = forward
blocks.Attention.__call__ = timed('attention', blocks.Attention.__call__)
blocks.Engram.__call__ = timed('engram (incl. row reads)', blocks.Engram.__call__)
blocks.EngramHashes.__call__ = timed('engram hashes', blocks.EngramHashes.__call__)
blocks.Block.mixes = timed('hyper-connection mixes', blocks.Block.mixes)
blocks.tensor_sha256 = timed('tensor_sha256 traces', blocks.tensor_sha256)
streamed_moe.StreamedMoE.forward = timed('moe forward (all)', streamed_moe.StreamedMoE.forward)
expert_cache.TieredExpertCache.get = timed('  moe: cache.get (drive wait + upload)', expert_cache.TieredExpertCache.get)
streamed_moe.expert_forward = timed('  moe: expert compute', streamed_moe.expert_forward)
streamed_moe.route = timed('  moe: route', streamed_moe.route)
sparse_rows.SparseRows.get = timed('  engram: row lookup', sparse_rows.SparseRows.get)


@atexit.register
def report():
    steps = calls[TOTAL]
    if not steps:
        print('profile_decode: no decode steps were timed')
        return
    print(f'\n=== seconds per decode token ({steps} steps, synchronised sections) ===')
    for name, value in sorted(seconds.items(), key=lambda item: -item[1]):
        print(f'{value / steps:8.3f}  {name}  ({calls[name] // steps} calls/token)')


if __name__ == '__main__':
    import frustum_chat
    sys.argv[0] = 'frustum_chat.py'
    frustum_chat.main()
