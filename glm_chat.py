"""GLM-4.7-Flash GGUF smoke test using Frusium's bounded reader and CUDA PyTorch."""
import argparse
import ctypes
import os
from pathlib import Path
from types import SimpleNamespace
import json
import time

from gguf import GGUFReader
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F

from expert_cache import TieredExpertCache
from frusium_loader import TensorRange
from phi4_chat import unpack

GIB = 2**30
EOS = {154820, 154827, 154829}


def available_ram():
    if os.name == 'nt':
        # MEMORYSTATUSEX: two DWORDs followed by seven 64-bit counters.
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                        ('total', ctypes.c_uint64), ('available', ctypes.c_uint64),
                        ('remaining_counters', ctypes.c_uint64 * 5)]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Could not query available system RAM')
        return status.available
    return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')


def glm_prompt(messages):
    return '[gMASK]<sop>' + ''.join(
        f"<|{message['role']}|>" + ('</think>' if message['role'] == 'assistant' else '')
        + message['content'] for message in messages
    ) + '<|assistant|></think>'


def prompt_suffix(model, ids, reuse):
    cached = model.tokens
    if reuse and cached and len(cached) < len(ids) and ids[:len(cached)] == cached:
        return ids[len(cached):], len(cached)
    model.reset_kv()
    return ids, 0


def decode_weight(raw, kind):
    if kind == 'Q8_0':
        blocks = raw.reshape(-1, 34)
        scale = blocks[:, :2].contiguous().view(torch.float16).float()
        return (scale * blocks[:, 2:].view(torch.int8).float()).reshape(*raw.shape[:-1], -1)
    return unpack(raw.reshape(-1, raw.shape[-1]), kind).reshape(*raw.shape[:-1], -1)


class Glm:
    def __init__(self, path, budget, *, split=False, ram_budget=0, io_threads=4, lookahead=4):
        """budget: GPU bytes for packed weights. split pins always-used tensors so only routed
        expert slices compete for the rest. ram_budget adds a host-RAM tier for those slices.

        Storage is the shared expert_cache.TieredExpertCache (the same code the DeepSeek path
        uses), with LRU in both tiers as before. Arithmetic is unchanged.
        """
        self.path = path
        self.gguf = GGUFReader(path)
        if self.gguf.fields['general.architecture'].contents() != 'deepseek2' or self.gguf.fields['general.name'].contents() != 'GLM 4.7 Flash':
            raise ValueError('Expected the GLM-4.7-Flash deepseek2 GGUF')
        self.tensors = {t.name: t for t in self.gguf.tensors}
        self.budget = budget
        self.split = split
        self.fixed_budget = sum(t.n_bytes for name, t in self.tensors.items()
                                if '_exps.' not in name) if split else 0
        self.dynamic_budget = budget - self.fixed_budget
        if self.dynamic_budget <= 0:
            raise ValueError('GPU budget cannot reserve all always-used tensors')
        self.ram_budget = min(ram_budget, max(0, available_ram() - 4 * GIB)) if ram_budget else 0
        # One cache entry per whole tensor (expert None) or per expert slice of a stacked tensor.
        self.ranges = SimpleNamespace(tensors={})
        self.pinned = (lambda key: key[1] is None) if split else (lambda key: False)
        self.store = TieredExpertCache(torch, self.ranges, 'cuda', budget, self.ram_budget, io_threads=io_threads,
                                       gpu_policy='lru', host_policy='lru', pinned=self.pinned)
        self.reader = self.store.reader
        self.lookahead = lookahead
        self.norm_eps = 1e-5
        self.freq = 1e6 ** (-torch.arange(0, 64, 2, device='cuda').float() / 64)
        self.cache = {}
        self.pos = 0
        self.tokens = []

    def reset_kv(self):
        self.cache.clear()
        self.tokens.clear()
        self.pos = 0

    def metrics(self):
        store = self.store
        fixed = sum(size for key, size in store.gpu.order.items() if self.pinned(key))
        return dict(gpu_hits=store.hits, ram_hits=store.host_hits, file_reads=store.disk_reads,
                    transfer_bytes=store.upload_bytes, file_read_seconds=store.wait_seconds,
                    blocking_upload_seconds=store.upload_seconds, file_bytes=self.reader.bytes_read,
                    ram_bytes=store.host.used, gpu_fixed_bytes=fixed, gpu_dynamic_bytes=store.used - fixed)

    def _key(self, name, expert=None):
        key = (name, expert)
        if key not in self.store.plan:
            tensor = self.tensors[name]
            shape = tensor.data.shape
            size = tensor.n_bytes
            offset = tensor.data_offset
            if expert is not None:
                size //= shape[0]
                offset += expert * size
                shape = shape[1:]
            self.ranges.tensors[key] = TensorRange(name, self.path, offset, size, tensor.tensor_type.name, shape)
            self.store.plan[key] = [key]
        return key

    def prefetch(self, keys):
        """Start background reads for (name, expert) pairs that are about to be used."""
        self.store.prefetch([self._key(name, expert) for name, expert in keys])

    def packed(self, name, expert=None):
        key = self._key(name, expert)
        entry = self.store.get(key, (key,))
        shaped = getattr(entry, 'shaped', None)
        if shaped is None:
            view = self.ranges.tensors[key]
            raw = entry.tensors[0]
            raw = raw.view(torch.float32) if view.dtype == 'F32' else raw
            shaped = entry.shaped = (raw.reshape(view.shape), view.dtype)
        return shaped

    def close(self):
        self.store.close()

    def linear(self, x, name, expert=None):
        raw, kind = self.packed(name + '.weight', expert)
        return torch.cat([F.linear(x, decode_weight(chunk, kind).to(x.dtype))
                          for chunk in raw.split(2048)], -1)

    def norm(self, x, name):
        return F.rms_norm(x.float(), (x.shape[-1],), self.packed(name + '.weight')[0], self.norm_eps).to(x.dtype)

    def rotate(self, x, pos):
        phase = (pos[:, None] * self.freq)[:, None]
        a, b = x.float()[..., ::2], x.float()[..., 1::2]
        return torch.stack((a * phase.cos() - b * phase.sin(), b * phase.cos() + a * phase.sin()), -1).flatten(-2).to(x.dtype)

    def attention(self, x, p, layer, pos):
        q = self.linear(self.norm(self.linear(x, p+'attn_q_a'), p+'attn_q_a_norm'), p+'attn_q_b')
        q_base, q_rot = q.reshape(-1, 20, 256).split((192, 64), -1)
        latent, k_rot = self.linear(x, p+'attn_kv_a_mqa').split((512, 64), -1)
        latent = self.norm(latent, p+'attn_kv_a_norm')
        k_weight = decode_weight(*self.packed(p+'attn_k_b.weight')).to(x.dtype)
        v_weight = decode_weight(*self.packed(p+'attn_v_b.weight')).to(x.dtype)
        absorbed = torch.einsum('thd,hrd->thr', q_base, k_weight)
        keys = torch.cat((latent, self.rotate(k_rot[:, None], pos).squeeze(1)), -1)
        if layer in self.cache:
            keys = torch.cat((self.cache[layer], keys))
        self.cache[layer] = keys
        queries = torch.cat((absorbed, self.rotate(q_rot, pos)), -1)
        logits = torch.einsum('thr,sr->hts', queries.float(), keys.float()) / 16
        visible = torch.arange(len(keys), device=x.device)[None] <= pos[:, None]
        probs = logits.masked_fill(~visible, -torch.inf).softmax(-1).to(x.dtype)
        mixed = torch.einsum('hts,sr->thr', probs, keys[:, :512])
        out = torch.einsum('thr,hdr->thd', mixed, v_weight).flatten(1)
        return self.linear(out, p+'attn_output')

    def mlp(self, x, p, suffix='', expert=None):
        return self.linear(F.silu(self.linear(x, p+'ffn_gate'+suffix, expert))
                           * self.linear(x, p+'ffn_up'+suffix, expert), p+'ffn_down'+suffix, expert)

    def moe(self, x, p):
        score = self.linear(x.float(), p+'ffn_gate_inp').sigmoid()
        indices = (score + self.packed(p+'exp_probs_b.bias')[0]).topk(4).indices
        mix = score.gather(-1, indices)
        mix = mix / mix.sum(-1, keepdim=True) * 1.8
        result = self.mlp(x, p, '_shexp').float()
        experts = sorted(set(indices.flatten().tolist()))
        wanted = [(p + part + '_exps.weight', expert) for expert in experts
                  for part in ('ffn_gate', 'ffn_up', 'ffn_down')]
        for position, expert in enumerate(experts):
            # Routing is known; read the next few expert slices while this one computes.
            self.prefetch(wanted[3 * position:3 * (position + 1 + self.lookahead)])
            rows, rank = torch.where(indices == expert)
            result.index_add_(0, rows, self.mlp(x[rows], p, '_exps', expert).float() * mix[rows, rank, None])
        return result.to(x.dtype)

    def forward(self, ids, progress=False):
        raw, kind = self.packed('token_embd.weight')
        x = decode_weight(raw[ids], kind).half()
        pos = torch.arange(self.pos, self.pos + len(ids), device=x.device)
        for layer in range(47):
            p = f'blk.{layer}.'
            x = x + self.attention(self.norm(x, p+'attn_norm'), p, layer, pos)
            norm = self.norm(x, p+'ffn_norm')
            x = x + (self.mlp(norm, p) if layer == 0 else self.moe(norm, p))
            if progress and (layer + 1) % 10 == 0:
                print(f'Prefill layer {layer + 1}/47', flush=True)
        logits = self.linear(self.norm(x[-1:], 'output_norm'), 'output').float().flatten()
        self.pos += len(ids)
        self.tokens.extend(ids)
        return logits


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(r'G:\Models\GLM-4.7-Flash\zai-org_GLM-4.7-Flash-Q4_K_M.gguf'))
    parser.add_argument('--prompt', default='Do you like Dino Buddies? 🦖 Reply in one short sentence.')
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.evidence.exists() or not 1 <= args.max_new_tokens <= 64:
        parser.error('Use a fresh evidence path and a token limit from 1 to 64')
    free, total = torch.cuda.mem_get_info()
    cache_bytes = min(9 * 2**30, free - 2 * 2**30)
    if cache_bytes < 2**30:
        raise RuntimeError('Insufficient free GPU memory')
    torch.cuda.set_per_process_memory_fraction(min(12 * 2**30 / total, 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model = Glm(args.model, cache_bytes)
    tok = Tokenizer.from_file(str(args.model.with_name('tokenizer.json')))
    ids = tok.encode(f'[gMASK]<sop><|user|>{args.prompt}<|assistant|></think>', add_special_tokens=False).ids
    output, timings = [], []
    for i in range(args.max_new_tokens):
        start = time.perf_counter()
        logits = model.forward(ids, progress=i == 0)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite GLM logits')
        token = logits.argmax().item()
        timings.append(time.perf_counter() - start)
        if token in (154820, 154827, 154829):
            break
        ids = [token]
        output.append(token)
        print(f'Token {i+1}: {tok.decode(output)!r}', flush=True)
    reply = tok.decode(output)
    evidence = dict(prompt=args.prompt, reply=reply, token_ids=output, layers=47,
                    first_token_seconds=timings[0], decode_seconds=timings[1:],
                    logical_weight_read_bytes=model.reader.bytes_read,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                    cache_GiB=cache_bytes/2**30, stopped_at_eos=len(output)<len(timings))
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    with args.evidence.open('x', encoding='utf-8') as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
