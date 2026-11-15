"""Phi-4 GGUF backend for Frustum_loader, using packed GPU weights and PyTorch."""
import argparse
import json
from pathlib import Path
import time

from gguf import GGUFReader, dequantize
import numpy as np
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F

from frustum_loader import RangeReader, TensorRange

GIB = 1024**3


def unpack(raw, kind):
    if kind == 'F32':
        return raw.view(torch.float32)
    rows = raw.shape[0]
    width = {'Q4_K': 144, 'Q5_K': 176, 'Q6_K': 210}[kind]
    blocks = raw.reshape(-1, width)
    shifts = torch.arange(8, device=raw.device, dtype=torch.uint8)
    if kind == 'Q6_K':
        low = (blocks[:, :128].reshape(-1, 2, 1, 64) >> (shifts[:2] * 4).view(1, 1, 2, 1)) & 15
        high = (blocks[:, 128:192].reshape(-1, 2, 1, 32) >> (shifts[:4] * 2).view(1, 1, 4, 1)) & 3
        codes = (low.reshape(-1, 16, 16) | (high.reshape(-1, 16, 16) << 4)).float() - 32
        scale = blocks[:, 192:208].view(torch.int8).float()
        factor = blocks[:, 208:210].contiguous().view(torch.float16).float()
        return (codes * (scale * factor).unsqueeze(-1)).reshape(rows, -1)
    factors = blocks[:, :4].contiguous().view(torch.float16).float()
    d, m, tail = blocks[:, 4:16].split(4, dim=1)
    scale = torch.cat((d & 63, (tail & 15) | ((d >> 2) & 48)), dim=1).float()
    minimum = torch.cat((m & 63, (tail >> 4) | ((m >> 2) & 48)), dim=1).float()
    codes = (blocks[:, width - 128:].reshape(-1, 4, 1, 32) >> (shifts[:2] * 4).view(1, 1, 2, 1)) & 15
    codes = codes.reshape(-1, 8, 32)
    if kind == 'Q5_K':
        codes = codes | (((blocks[:, None, 16:48] >> shifts.view(1, 8, 1)) & 1) << 4)
    return ((scale * factors[:, :1]).unsqueeze(-1) * codes.float()
            - (minimum * factors[:, 1:]).unsqueeze(-1)).reshape(rows, -1)


class Phi4:
    def __init__(self, path, context):
        self.gguf = GGUFReader(path)
        self.meta = {key: field.contents() for key, field in self.gguf.fields.items()
                     if key.startswith(('phi3.', 'general.architecture', 'tokenizer.ggml.eos'))}
        if self.meta['general.architecture'] != 'phi3' or self.meta['phi3.attention.sliding_window']:
            raise ValueError('Expected dense Phi-4 with full attention')
        self.layers = self.meta['phi3.block_count']
        self.heads = self.meta['phi3.attention.head_count']
        self.kv_heads = self.meta['phi3.attention.head_count_kv']
        self.dim = self.meta['phi3.rope.dimension_count']
        self.eps = self.meta['phi3.attention.layer_norm_rms_epsilon']
        self.context = context
        self.reader = RangeReader()
        self.weights = {}
        self.cache = {}
        self.position = 0
        free, total = torch.cuda.mem_get_info()
        if free < path.stat().st_size + 2 * GIB:
            raise RuntimeError('Insufficient free VRAM for packed weights and inference workspace')
        torch.cuda.set_per_process_memory_fraction(min(12 * GIB / total, 1))
        for tensor in self.gguf.tensors:
            kind = tensor.tensor_type.name
            if kind not in ('F32', 'Q4_K', 'Q5_K', 'Q6_K'):
                raise ValueError(f'Unsupported tensor type {kind}')
            view = TensorRange(tensor.name, path, tensor.data_offset, tensor.n_bytes, kind, tuple(tensor.data.shape))
            host = bytearray(view.size)
            self.reader.copy(view, lambda offset, data: host.__setitem__(slice(offset, offset + len(data)), data))
            raw = torch.frombuffer(host, dtype=torch.uint8).cuda()
            target = raw.view(torch.float32) if kind == 'F32' else raw.reshape(view.shape[0], -1)
            self.weights[tensor.name] = (target, kind)
            if kind != 'F32':
                expected = dequantize(np.asarray(tensor.data[:2]).copy(), tensor.tensor_type)
                torch.testing.assert_close(unpack(target[:2], kind).cpu(), torch.from_numpy(expected), rtol=0, atol=0)
        self.freqs = self.meta['phi3.rope.freq_base'] ** (
            -torch.arange(0, self.dim, 2, device='cuda', dtype=torch.float32) / self.dim)

    def linear(self, x, name):
        weight, kind = self.weights[name + '.weight']
        return torch.cat([F.linear(x, unpack(chunk, kind).to(x.dtype))
                          for chunk in weight.split(2048)], dim=-1)

    def norm(self, x, name):
        scale = self.weights[name + '.weight'][0]
        return (F.rms_norm(x.float(), (x.shape[-1],), scale, self.eps)).to(x.dtype)

    def rotate(self, x, positions):
        angle = positions[:, None] * self.freqs
        c, s = angle.cos()[None, None], angle.sin()[None, None]
        left, right = x.float().chunk(2, dim=-1)
        return torch.cat((left * c - right * s, right * c + left * s), dim=-1).to(x.dtype)

    def forward(self, ids):
        length = len(ids)
        end = self.position + length
        if not length or end > self.context:
            raise ValueError('Sequence exceeds configured context')
        weight, kind = self.weights['token_embd.weight']
        x = unpack(weight[ids], kind).half()[None]
        pos = torch.arange(self.position, end, device=x.device)
        mask = torch.arange(end, device=x.device)[None, :] <= pos[:, None]
        for layer in range(self.layers):
            prefix = f'blk.{layer}.'
            qkv = self.linear(self.norm(x, prefix + 'attn_norm'), prefix + 'attn_qkv')
            q, k, v = [t.reshape(1, length, -1, self.dim).transpose(1, 2)
                       for t in qkv.split([self.heads * self.dim, self.kv_heads * self.dim,
                                           self.kv_heads * self.dim], dim=-1)]
            q, k = self.rotate(q, pos), self.rotate(k, pos)
            if layer not in self.cache:
                self.cache[layer] = tuple(x.new_empty(1, self.kv_heads, self.context, self.dim)
                                          for _ in range(2))
            keys, values = self.cache[layer]
            keys[:, :, self.position:end] = k
            values[:, :, self.position:end] = v
            repeat = self.heads // self.kv_heads
            attn = F.scaled_dot_product_attention(
                q, keys[:, :, :end].repeat_interleave(repeat, dim=1),
                values[:, :, :end].repeat_interleave(repeat, dim=1), attn_mask=mask)
            x = x + self.linear(attn.transpose(1, 2).reshape(1, length, -1), prefix + 'attn_output')
            gate, up = self.linear(self.norm(x, prefix + 'ffn_norm'), prefix + 'ffn_up').chunk(2, dim=-1)
            x = x + self.linear(F.silu(gate) * up, prefix + 'ffn_down')
        self.position = end
        return self.linear(self.norm(x[:, -1:], 'output_norm'), 'output').float().flatten()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(r'G:\Models\Phi-4\phi-4-Q4_K_S.gguf'))
    parser.add_argument('--prompt', default='Do you like Dino Buddies? 🦖 Reply in one short sentence.')
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256 or (args.evidence and args.evidence.exists()):
        parser.error('Use a token limit of 1–256 and a new evidence file')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Phi4(args.model, context=512)
    tokenizer = Tokenizer.from_file(str(args.model.with_name('tokenizer.json')))
    eos = model.meta['tokenizer.ggml.eos_token_id']
    if tokenizer.token_to_id('<|im_end|>') != eos:
        raise ValueError('Tokenizer and GGUF EOS disagree')
    ids = tokenizer.encode(f'<|im_start|>user<|im_sep|>{args.prompt}<|im_end|>'
                           '<|im_start|>assistant<|im_sep|>', add_special_tokens=False).ids
    if len(ids) + args.max_new_tokens > model.context:
        parser.error('Prompt exceeds the available context')
    loaded = time.perf_counter()
    print(f'Loaded packed weights in {loaded - started:.2f}s', flush=True)
    output, timings = [], []
    for _ in range(args.max_new_tokens):
        step = time.perf_counter()
        logits = model.forward(ids)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite Phi-4 logits')
        token = logits.argmax().item()
        timings.append(time.perf_counter() - step)
        if token == eos:
            break
        output.append(token)
        ids = [token]
    reply = tokenizer.decode(output)
    result = dict(prompt=args.prompt, reply=reply, token_ids=output, layers=model.layers,
                  load_seconds=loaded - started, first_token_seconds=timings[0],
                  decode_seconds=timings[1:], peak_reserved_GiB=torch.cuda.max_memory_reserved() / GIB,
                  logical_weight_read_bytes=model.reader.bytes_read, stopped_at_eos=len(output) < len(timings))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.evidence:
        with args.evidence.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
