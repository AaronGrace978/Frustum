"""Qwen3-Next-80B-A3B (Qwen3NextForCausalLM) on one GPU with Frustum's tiered expert streaming.

Placement
- Dense weights (attention, linear attention, routers, shared experts, norms, output head; about
  3.7 GiB in BF16) are read once and stay on the GPU.
- Routed experts (48 layers x 512 experts, 10 per token, 6 MiB each in BF16) stream from the
  checkpoint through expert_cache.TieredExpertCache: GPU tier, host-RAM tier, then the drive.
  This is the same cache the DeepSeek and GLM paths use.
- The token embedding stays on the drive: each new token reads one 4 KiB row.

Architecture (a plain-PyTorch reimplementation of transformers' modeling_qwen3_next; no model
source is imported or executed):
- Every fourth layer is gated full attention (16 query heads, 2 KV heads, head_dim 256, rotary on
  the first 64 dims, output scaled by sigmoid(gate)). The other 36 are Gated DeltaNet linear
  attention with a causal depthwise conv and a fixed-size recurrent state per layer.
- RMSNorm multiplies by (1 + weight); the gated norm inside linear attention multiplies by weight.
- MoE: softmax router, top 10, renormalised, plus a shared expert scaled by sigmoid(gate).

The engine keeps its state between calls, so a follow-up turn whose token ids extend the previous
ones only processes the new tokens (linear-attention state cannot be rolled back, so anything
else starts over). qwen_next_parity.py checks this file against the transformers reference.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from expert_cache import TieredExpertCache
from frustum_loader import Catalog, TensorRange

GIB = 2**30
DTYPES = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}
EOS = (151645, 151643)  # <|im_end|>, <|endoftext|>


def qwen_prompt(messages):
    """The Qwen chat template for plain user/assistant/system turns (no tools)."""
    return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + '<|im_start|>assistant\n'


def available_ram():
    from glm_chat import available_ram as probe
    return probe()


def auto_host_bytes(reserve=6 * GIB, cap=16 * GIB):
    """85% of the RAM left after a reserve for the OS and other programs, up to cap."""
    return int(min(cap, max(0, available_ram() - reserve) * 0.85))


class QwenNext:
    def __init__(self, folder, gpu_bytes, host_bytes=0, *, io_threads=4, cap_bytes=None, device='cuda'):
        """gpu_bytes: GPU bytes for dense weights plus the routed-expert tier."""
        self.catalog = catalog = Catalog(folder)
        cfg = self.cfg = catalog.config
        if cfg.get('model_type') != 'qwen3_next':
            raise ValueError('Expected a Qwen3-Next checkpoint (model_type qwen3_next)')
        if cfg.get('mlp_only_layers') or cfg.get('decoder_sparse_step', 1) != 1:
            raise ValueError('This engine expects every layer to be a MoE layer')
        self.device = torch.device(device)
        self.n_layers = cfg['num_hidden_layers']
        self.interval = cfg['full_attention_interval']
        self.eps = cfg['rms_norm_eps']
        self.heads, self.kv_heads, self.head_dim = cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim']
        self.nk, self.nv = cfg['linear_num_key_heads'], cfg['linear_num_value_heads']
        self.dk, self.dv = cfg['linear_key_head_dim'], cfg['linear_value_head_dim']
        self.kernel = cfg['linear_conv_kernel_dim']
        self.top_k, self.n_experts = cfg['num_experts_per_tok'], cfg['num_experts']
        self.norm_topk = cfg.get('norm_topk_prob', True)
        rotary = int(self.head_dim * cfg.get('partial_rotary_factor', 1.0))
        if cfg.get('rope_scaling'):
            raise ValueError('rope_scaling is not supported')
        self.inv_freq = 1.0 / (cfg['rope_theta'] ** (torch.arange(0, rotary, 2, dtype=torch.int64).float().to(self.device) / rotary))

        tensors = catalog.tensors
        self.embed = tensors['model.embed_tokens.weight']
        if self.embed.dtype not in DTYPES:
            raise ValueError('Unsupported embedding dtype ' + self.embed.dtype)
        dense = [name for name in tensors if (name.startswith('model.') or name == 'lm_head.weight')
                 and '.mlp.experts.' not in name and name != self.embed.name]
        dense_bytes = sum(tensors[name].size for name in dense)
        expert_bytes = gpu_bytes - dense_bytes
        one_expert = sum(tensors[f'model.layers.0.mlp.experts.0.{p}_proj.weight'].size for p in ('gate', 'up', 'down'))
        if expert_bytes < 2 * self.top_k * one_expert:
            raise ValueError(f'GPU budget {gpu_bytes / GIB:.1f} GiB leaves too little room for experts '
                             f'after {dense_bytes / GIB:.1f} GiB of dense weights')
        self.store = TieredExpertCache(torch, catalog, self.device, expert_bytes, host_bytes, io_threads=io_threads,
                                       gpu_policy='lru', host_policy='lru', cap_bytes=cap_bytes,
                                       ram_probe=available_ram if host_bytes else None, ram_floor_bytes=3 * GIB)
        self.reader = self.store.reader
        for layer in range(self.n_layers):
            for e in range(self.n_experts):
                p = f'model.layers.{layer}.mlp.experts.{e}.'
                self.store.plan[(layer, e)] = [p + 'gate_proj.weight', p + 'up_proj.weight', p + 'down_proj.weight']
        self.dense_bytes, self.expert_bytes, self.host_bytes = dense_bytes, expert_bytes, host_bytes
        self.w = {}
        with ThreadPoolExecutor(max(1, io_threads)) as pool:
            for name, data in zip(dense, pool.map(lambda n: self.reader.read(tensors[n]), dense)):
                self.w[name] = self._tensor(data, tensors[name])
        self.reset()

    def _tensor(self, data, item):
        raw = torch.frombuffer(data, dtype=torch.uint8).to(self.device)
        return raw.view(DTYPES[item.dtype]).reshape(item.shape)

    # ---- state ----
    def reset(self):
        self.tokens = []
        self.kv = {}      # full-attention layer -> (keys, values), each (positions, kv_heads, head_dim)
        self.conv = {}    # linear layer -> last kernel-1 conv inputs (kernel-1, conv_dim)
        self.state = {}   # linear layer -> recurrent state (nv, dk, dv) float32

    def close(self):
        self.store.close()

    def metrics(self):
        return self.store.stats()

    # ---- building blocks ----
    def norm(self, x, weight):
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + weight.float())).to(x.dtype)

    def gated_norm(self, x, gate, weight):
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = weight * out.to(x.dtype)
        return (out * F.silu(gate.float())).to(x.dtype)

    def rope(self, x, pos):
        freqs = pos[:, None].float() * self.inv_freq[None]
        emb = torch.cat((freqs, freqs), -1)
        cos, sin = emb.cos().to(x.dtype)[:, None], emb.sin().to(x.dtype)[:, None]
        d = emb.shape[-1]
        rot, rest = x[..., :d], x[..., d:]
        half = d // 2
        rotated = torch.cat((-rot[..., half:], rot[..., :half]), -1)
        return torch.cat((rot * cos + rotated * sin, rest), -1)

    def attention(self, x, p, layer, pos):
        w, T, H, D = self.w, x.shape[0], self.heads, self.head_dim
        q, gate = F.linear(x, w[p + 'q_proj.weight']).view(T, H, 2 * D).chunk(2, -1)
        gate = gate.reshape(T, H * D)
        q = self.rope(self.norm(q, w[p + 'q_norm.weight']), pos)
        k = self.rope(self.norm(F.linear(x, w[p + 'k_proj.weight']).view(T, self.kv_heads, D), w[p + 'k_norm.weight']), pos)
        v = F.linear(x, w[p + 'v_proj.weight']).view(T, self.kv_heads, D)
        if layer in self.kv:
            k, v = torch.cat((self.kv[layer][0], k)), torch.cat((self.kv[layer][1], v))
        self.kv[layer] = (k, v)
        groups = H // self.kv_heads
        keys = k.repeat_interleave(groups, 1).transpose(0, 1).float()     # (H, S, D)
        values = v.repeat_interleave(groups, 1).transpose(0, 1)           # (H, S, D)
        scores = q.transpose(0, 1).float() @ keys.transpose(1, 2) * D ** -0.5
        visible = torch.arange(k.shape[0], device=x.device)[None] <= pos[:, None]
        probs = scores.masked_fill(~visible, -torch.inf).softmax(-1).to(x.dtype)
        out = (probs @ values).transpose(0, 1).reshape(T, H * D)
        return F.linear(out * torch.sigmoid(gate), w[p + 'o_proj.weight'])

    def linear_attention(self, x, p, layer):
        w, T = self.w, x.shape[0]
        nk, nv, dk, dv = self.nk, self.nv, self.dk, self.dv
        r = nv // nk
        qkvz = F.linear(x, w[p + 'in_proj_qkvz.weight']).view(T, nk, 2 * dk + 2 * r * dv)
        q, k, v, z = torch.split(qkvz, (dk, dk, r * dv, r * dv), -1)
        ba = F.linear(x, w[p + 'in_proj_ba.weight']).view(T, nk, 2 * r)
        b, a = torch.split(ba, (r, r), -1)
        v, z = v.reshape(T, nv, dv), z.reshape(T, nv, dv)
        b, a = b.reshape(T, nv), a.reshape(T, nv)
        mixed = torch.cat((q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)), -1)  # (T, conv_dim)

        # Causal depthwise conv over [previous kernel-1 inputs, new inputs].
        history = self.conv.get(layer)
        if history is None:
            history = mixed.new_zeros(self.kernel - 1, mixed.shape[1])
        seq = torch.cat((history, mixed))
        self.conv[layer] = seq[-(self.kernel - 1):]
        weight = w[p + 'conv1d.weight']  # (conv_dim, 1, kernel)
        mixed = F.silu(F.conv1d(seq.t()[None], weight, groups=weight.shape[0])[0].t())  # (T, conv_dim)

        q, k, v = torch.split(mixed, (nk * dk, nk * dk, nv * dv), -1)
        q = q.reshape(T, nk, dk).repeat_interleave(r, 1)
        k = k.reshape(T, nk, dk).repeat_interleave(r, 1)
        v = v.reshape(T, nv, dv).float()
        beta = b.sigmoid().float()
        g = (-w[p + 'A_log'].float().exp() * F.softplus(a.float() + w[p + 'dt_bias'])).exp()
        # l2norm as transformers/FLA define it (rsqrt of sum of squares + eps), then the 1/sqrt(dk) scale.
        q, k = q.float(), k.float()
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * dk ** -0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)

        state = self.state.get(layer)
        if state is None:
            state = torch.zeros(nv, dk, dv, device=x.device, dtype=torch.float32)
        out = torch.empty(T, nv, dv, device=x.device, dtype=torch.float32)
        for t in range(T):  # gated delta rule, one position at a time
            state = state * g[t][:, None, None]
            remembered = torch.einsum('hk,hkv->hv', k[t], state)
            delta = (v[t] - remembered) * beta[t][:, None]
            state = state + k[t][:, :, None] * delta[:, None, :]
            out[t] = torch.einsum('hk,hkv->hv', q[t], state)
        self.state[layer] = state

        out = self.gated_norm(out.to(x.dtype).reshape(-1, dv), z.reshape(-1, dv), w[p + 'norm.weight'])
        return F.linear(out.reshape(T, nv * dv), w[p + 'out_proj.weight'])

    def mlp(self, x, gate, up, down):
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

    def expert(self, layer, e):
        entry = self.store.get((layer, e))
        shaped = getattr(entry, 'shaped', None)
        if shaped is None:
            names = self.store.plan[(layer, e)]
            shaped = entry.shaped = [raw.view(DTYPES[self.catalog.tensors[n].dtype]).reshape(self.catalog.tensors[n].shape)
                                     for raw, n in zip(entry.tensors, names)]
        return shaped

    def moe(self, x, layer, p):
        w = self.w
        probs = F.softmax(F.linear(x, w[p + 'gate.weight']), dim=-1, dtype=torch.float)
        weights, chosen = probs.topk(self.top_k, dim=-1)
        if self.norm_topk:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(x.dtype)
        experts = sorted(set(chosen.flatten().tolist()))
        keys = [(layer, e) for e in experts]
        store = self.store
        store.begin_layer(keys)
        store.prefetch(keys)
        # Experts already on the GPU first, then RAM hits, then drive reads in the order issued,
        # so the GPU works while the drive is still reading.
        keys.sort(key=lambda key: 0 if key in store.cache else 1 if key in store.ram else 2)
        result = torch.zeros_like(x, dtype=torch.float32)
        for key in keys:
            rows, rank = torch.where(chosen == key[1])
            out = self.mlp(x[rows], *self.expert(*key)) * weights[rows, rank, None]
            result.index_add_(0, rows, out.float())
        shared = self.mlp(x, w[p + 'shared_expert.gate_proj.weight'], w[p + 'shared_expert.up_proj.weight'],
                          w[p + 'shared_expert.down_proj.weight'])
        shared = torch.sigmoid(F.linear(x, w[p + 'shared_expert_gate.weight'])) * shared
        return (result + shared.float()).to(x.dtype)

    def embed_rows(self, ids):
        row = self.embed.size // self.embed.shape[0]
        items = [TensorRange(self.embed.name, self.embed.path, self.embed.offset + i * row, row, self.embed.dtype,
                             (self.embed.shape[1],)) for i in ids]
        rows = [torch.frombuffer(self.reader.read(item), dtype=torch.uint8) for item in items]
        return torch.stack(rows).to(self.device).view(DTYPES[self.embed.dtype])

    def forward(self, ids, progress=None, all_positions=False):
        """Feed new token ids after the ones already processed; return the last position's logits
        (every position's with all_positions=True, used by the parity check)."""
        if not ids:
            raise ValueError('No tokens to process')
        start = len(self.tokens)
        pos = torch.arange(start, start + len(ids), device=self.device)
        x = self.embed_rows(ids)
        self.store.new_pass()
        try:
            for layer in range(self.n_layers):
                p = f'model.layers.{layer}.'
                h = self.norm(x, self.w[p + 'input_layernorm.weight'])
                if (layer + 1) % self.interval == 0:
                    x = x + self.attention(h, p + 'self_attn.', layer, pos)
                else:
                    x = x + self.linear_attention(h, p + 'linear_attn.', layer)
                x = x + self.moe(self.norm(x, self.w[p + 'post_attention_layernorm.weight']), layer, p + 'mlp.')
                if progress:
                    progress(layer + 1, self.n_layers)
        except BaseException:
            self.store.discard_pending()
            self.reset()  # partial updates to the recurrent state cannot be undone
            raise
        self.tokens.extend(ids)
        x = x if all_positions else x[-1:]
        logits = F.linear(self.norm(x, self.w['model.norm.weight']), self.w['lm_head.weight']).float()
        return logits if all_positions else logits.flatten()


def prompt_suffix(model, ids, reuse=True):
    """New tokens to feed when ids extend what the engine already holds; otherwise reset."""
    cached = model.tokens
    if reuse and cached and len(cached) < len(ids) and ids[:len(cached)] == cached:
        return ids[len(cached):], len(cached)
    model.reset()
    return ids, 0


def budgets(free, total, host='auto'):
    """GPU cap (bytes the process may use), tier budget for weights, and host-RAM tier bytes."""
    cap = min(12 * GIB, free - 3 * GIB // 2)
    weights = cap - 5 * GIB // 4  # working room for activations and attention state
    host = auto_host_bytes() if host in (None, '', 'auto') else int(float(host) * GIB)
    return cap, weights, max(0, host)


@torch.inference_mode()
def main():
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', type=Path, default=Path(os.environ.get('FRUSTUM_MODELS', 'G:/Models')) / 'Qwen3-Next-80B-A3B-Instruct')
    parser.add_argument('--prompt', default='Do you like Dino Buddies? Reply in one short sentence.')
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--host-gib', default=os.environ.get('FRUSTUM_QWEN_HOST_GIB', 'auto'))
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if args.evidence and args.evidence.exists():
        parser.error('Use a fresh evidence path')
    free, total = torch.cuda.mem_get_info()
    cap, weights, host = budgets(free, total, args.host_gib)
    torch.cuda.set_per_process_memory_fraction(min(cap / total, 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    loaded = time.perf_counter()
    model = QwenNext(args.model, weights, host, cap_bytes=cap)
    print(f'Loaded dense weights ({model.dense_bytes / GIB:.2f} GiB) in {time.perf_counter() - loaded:.1f} s; '
          f'experts {model.expert_bytes / GIB:.1f} GiB GPU + {host / GIB:.1f} GiB RAM', flush=True)
    tok = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    ids = tok.encode(qwen_prompt([dict(role='user', content=args.prompt)]), add_special_tokens=False).ids
    output, timings = [], []
    for i in range(args.max_new_tokens):
        start = time.perf_counter()
        logits = model.forward(ids, progress=(lambda d, n: print(f'Prefill layer {d}/{n}', flush=True) if d % 12 == 0 else None) if i == 0 else None)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite logits')
        token = logits.argmax().item()
        timings.append(time.perf_counter() - start)
        if token in EOS:
            break
        ids = [token]
        output.append(token)
        print(f'Token {i + 1} ({timings[-1]:.2f} s): {tok.decode(output)!r}', flush=True)
    stats = model.metrics()
    evidence = dict(prompt=args.prompt, reply=tok.decode(output), token_ids=output,
                    first_token_seconds=timings[0], decode_seconds=timings[1:],
                    gpu_hits=stats['gpu_hits'], host_hits=stats['host_hits'], disk_reads=stats['disk_reads'],
                    payload_bytes_read=stats['payload_bytes_read'], dense_GiB=model.dense_bytes / GIB,
                    expert_gpu_GiB=model.expert_bytes / GIB, host_GiB=host / GIB,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved() / GIB)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    if args.evidence:
        with args.evidence.open('x', encoding='utf-8') as handle:
            json.dump(evidence, handle, ensure_ascii=False, indent=2)
    model.close()


if __name__ == '__main__':
    main()
