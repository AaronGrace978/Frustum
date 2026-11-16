"""Decide what lives where for the DeepSeek text path. Pure Python, header metadata only.

Per generated token the model touches the same dense weights, routers, shared
experts and output head every time, plus a changing set of routed experts.
v0 pushed all of it through small LRU caches, so about 5.6 GB of dense weights
and 1.3 GB of output head were re-read for every token. The plan here keeps
the always-used set resident on the GPU when it fits and gives what is left to
routed experts, with an optional host-RAM tier behind that.
"""
import ctypes
import os
import re

GIB = 1024 ** 3
MIB = 1024 ** 2
LAYER = re.compile(r'^layers\.(\d+)\.(.+)$')
HOST_SHARE_BUFFERED = 0.5     # of RAM above 6 GiB, when the OS file cache also holds expert bytes
HOST_SHARE_UNBUFFERED = 0.85  # v0.8: no duplicate in the OS file cache


def available_ram():
    """Bytes of physical RAM currently available, or None if it cannot be determined."""
    try:
        if os.name == 'nt':
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                            ('total', ctypes.c_uint64), ('available', ctypes.c_uint64),
                            ('remaining_counters', ctypes.c_uint64 * 5)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.available)
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, AttributeError):
        return None


def classify(name, layers):
    """Role of a checkpoint tensor in the text path, or None if the text path never reads it."""
    if name == 'head.weight':
        return 'head'
    if name == 'norm.weight':
        return 'dense'
    match = LAYER.match(name)
    if not match or int(match.group(1)) >= layers:
        return None
    rest = match.group(2)
    if rest.startswith('ffn.experts.'):
        return 'routed'
    if rest.startswith('ffn.shared_experts.'):
        return 'shared'
    if rest.startswith('ffn.gate.'):
        return 'router'
    if rest.startswith('engram.embed.'):
        return 'rows'
    return 'dense'


def measure(catalog):
    cfg = catalog.config['text_config']
    layers = cfg['num_hidden_layers']
    totals = dict(dense=0, router=0, shared=0, routed=0, head=0, rows=0)
    for name, tensor in catalog.tensors.items():
        role = classify(name, layers)
        if role:
            totals[role] += tensor.size
    routed_count = layers * cfg['n_routed_experts']
    one_expert = totals['routed'] // routed_count if routed_count else 0
    totals['one_routed_expert'] = one_expert
    # Upper bound for one decode token: every layer picks its experts with no overlap.
    totals['routed_working_set'] = layers * cfg['num_experts_per_tok'] * one_expert
    return totals


def plan(catalog, gpu_cap_bytes, *, reserve_bytes=2 * GIB, dense='auto', head='auto',
         expert_cache_mib=None, host_cache_gib='auto', ram_available=None, unbuffered=False):
    """Return a placement dict. Raises ValueError when an explicit request cannot fit.

    unbuffered=True means expert reads bypass the OS file cache (FRUSTUM_READER=unbuffered).
    The automatic host tier then takes 85% of the spare RAM instead of 50%: with buffered
    reads Windows keeps its own copy of every expert read, so half the spare RAM was left
    for that duplicate.
    """
    if dense not in ('auto', 'resident', 'streamed') or head not in ('auto', 'resident', 'streamed'):
        raise ValueError('dense and head accept auto, resident or streamed')
    sizes = measure(catalog)
    margin = 64 * MIB
    usable = gpu_cap_bytes - reserve_bytes
    router = sizes['router'] + 8 * MIB
    streamed_dense = 256 * MIB
    floor = sizes['shared'] + 1 * GIB  # least expert space worth running with
    notes = []

    def room(dense_bytes, head_bytes):
        return usable - router - dense_bytes - head_bytes

    choices = [(d, h) for d in (('resident', 'streamed') if dense == 'auto' else (dense,))
               for h in (('resident', 'streamed') if head == 'auto' else (head,))]
    # Preference: both resident, then head resident, then dense resident, then neither.
    choices.sort(key=lambda c: (c != ('resident', 'resident'), c[1] != 'resident', c[0] != 'resident'))
    chosen = None
    for d, h in choices:
        dense_bytes = sizes['dense'] + margin if d == 'resident' else streamed_dense
        head_bytes = sizes['head'] if h == 'resident' else 0
        if room(dense_bytes, head_bytes) >= floor or (d, h) == choices[-1]:
            chosen = (d, h, dense_bytes, head_bytes)
            break
    dense_mode, head_mode, dense_bytes, head_bytes = chosen
    expert_bytes = room(dense_bytes, head_bytes)
    if expert_cache_mib is not None:
        if expert_cache_mib * MIB > expert_bytes:
            notes.append(f'--expert-cache-mib {expert_cache_mib} exceeds the planned {expert_bytes // MIB} MiB; '
                         'the allocator guard decides if it fits.')
        expert_bytes = expert_cache_mib * MIB
    if expert_bytes < 128 * MIB:
        raise ValueError('GPU cap leaves under 128 MiB for experts. Raise --gpu-cap-gib or stream more.')
    if (dense_mode, head_mode) != ('resident', 'resident') and dense == 'auto' and head == 'auto':
        notes.append('Not everything always-used fits under the GPU cap; part of it is streamed per token.')
    pin_shared = sizes['shared'] + 512 * MIB <= expert_bytes
    routed_room = expert_bytes - (sizes['shared'] if pin_shared else 0)
    gpu_policy = 'lru' if routed_room >= sizes['routed_working_set'] + sizes['shared'] * (not pin_shared) else 'lfu'

    if ram_available is None:
        ram_available = available_ram()
    if host_cache_gib == 'auto':
        share = HOST_SHARE_UNBUFFERED if unbuffered else HOST_SHARE_BUFFERED
        host_bytes = 0 if ram_available is None else int(min(16 * GIB, max(0, ram_available - 6 * GIB) * share))
        if ram_available is None:
            notes.append('Could not read available RAM; host tier off. Set --host-cache-gib to enable it.')
    else:
        host_bytes = int(float(host_cache_gib) * GIB)
        if ram_available is not None and host_bytes > max(0, ram_available - 4 * GIB):
            host_bytes = max(0, ram_available - 4 * GIB)
            notes.append('Host tier reduced to leave 4 GiB of RAM free.')
    if host_bytes and host_bytes < 64 * sizes['one_routed_expert']:
        host_bytes = 0  # too small to matter
    return dict(
        gpu_cap_bytes=gpu_cap_bytes, reserve_bytes=reserve_bytes,
        dense=dense_mode, dense_cache_mib=-(-dense_bytes // MIB),
        head=head_mode, head_bytes=head_bytes,
        router_cache_mib=-(-router // MIB),
        expert_cache_mib=expert_bytes // MIB, pin_shared=pin_shared,
        gpu_policy=gpu_policy, host_policy='lru', host_cache_mib=host_bytes // MIB,
        sizes=sizes, notes=notes)


def describe(p):
    s = p['sizes']
    gib = lambda b: f'{b / GIB:.2f} GiB'
    routed_room = p['expert_cache_mib'] * MIB - (s['shared'] if p['pin_shared'] else 0)
    experts = routed_room // s['one_routed_expert'] if s['one_routed_expert'] else 0
    lines = [
        f"GPU cap {gib(p['gpu_cap_bytes'])}, {gib(p['reserve_bytes'])} kept for working memory",
        f"  dense weights   {gib(s['dense'])}  {p['dense']}",
        f"  output head     {gib(s['head'])}  {p['head']}",
        f"  routers         {gib(s['router'])}  resident",
        f"  shared experts  {gib(s['shared'])}  {'pinned' if p['pin_shared'] else 'cached with routed experts'}",
        f"  routed experts  {gib(routed_room)} cache, about {experts} experts, policy {p['gpu_policy']} "
        f"(one token needs up to {gib(s['routed_working_set'])})",
        f"Host RAM tier {p['host_cache_mib'] / 1024:.1f} GiB" if p['host_cache_mib'] else 'Host RAM tier off',
    ]
    return '\n'.join(lines + ['Note: ' + n for n in p['notes']])
