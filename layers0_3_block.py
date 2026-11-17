"""Bounded sequential execution of text blocks 0--3 from the local checkpoint.

This correctness baseline implements the supplied model.py schedule through layer 3:
pure-SWA layers 0--1, layer-1 Engram, and the layer-2 compressed-KV/index source
consumed by layer 3.  It intentionally uses dequantize-and-compute PyTorch kernels,
not the supplied TileLang kernels or a full-model generation loop.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from block_primitives import hc_mixes, hc_post, hc_pre, rms_norm
from expert_correctness import activation
from frustum_loader import Catalog, FrustumLoader, default_model_root
import fused_kernels
from layer0_block import RingKV
from reproducibility import cache_sha256, tensor_sha256
from sparse_rows import SparseRows
from streamed_moe import StreamedMoE

GIB = 1024 ** 3
MIB = 1024 ** 2


def rope(x, positions, base, original_max=0, factor=1.0, beta_fast=32, beta_slow=1, inverse=False):
    """Supplied precompute_freqs_cis/apply_rotary_emb equation, without a large cache."""
    dim = x.shape[-1]
    freq = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    if original_max:
        corrected = lambda rotations: dim * math.log(original_max / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low, high = max(math.floor(corrected(beta_fast)), 0), min(math.ceil(corrected(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, device=x.device, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freq = freq / factor * (1 - smooth) + freq * smooth
    angle = positions.float()[:, None] * freq[None, :]
    if inverse:
        angle = -angle
    shape = [1, len(positions)] + [1] * (x.ndim - 3) + [dim // 2]
    c, s = angle.cos().reshape(shape), angle.sin().reshape(shape)
    pairs = x.float().reshape(*x.shape[:-1], dim // 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2).to(x.dtype)


def fp4_quant_dequant(x, block_size, power2_scale):
    """Reference-style in-place FP4 quant/dequant for compressed KV and index vectors.

    The model's TileLang path stores FP4 values only transiently.  Keeping BF16 here avoids
    allocating packed activation buffers while retaining the documented E2M1 finite values.
    """
    shape = x.shape
    groups = x.float().reshape(-1, shape[-1] // block_size, block_size)
    floor = 6 * (2 ** (-126 if power2_scale else -9))
    scale = groups.abs().amax(-1).clamp_min(floor) / 6
    if power2_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    else:
        scale = scale.to(torch.float8_e4m3fn).float()
    z = (groups / scale[..., None]).clamp(-6, 6)
    levels = z.new_tensor([-6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6.])
    nearest = (z.unsqueeze(-1) - levels).abs().argmin(-1)
    return (levels[nearest] * scale[..., None]).reshape(shape).to(torch.bfloat16)


def sparse_attention_reference(q, kv, sink, idxs):
    """Original eager form: gather selected rows per token, then causal softmax. Kept as the oracle."""
    bsz, seqlen, heads, dim = q.shape
    out = torch.zeros_like(q)
    for batch in range(bsz):
        for token in range(seqlen):
            ids = idxs[batch, token]
            ids = ids[ids >= 0].long()
            if not ids.numel():
                continue
            values = kv[batch, ids].float()
            score = (q[batch, token].float() @ values.T) * (dim ** -0.5)
            maximum = score.max(-1, keepdim=True).values
            prob = torch.exp(score - maximum)
            denom = prob.sum(-1, keepdim=True) + torch.exp(sink.float()[:, None] - maximum)
            out[batch, token] = (prob.to(torch.bfloat16).float() @ values / denom).to(q.dtype)
    return out


def sparse_attention(q, kv, sink, idxs, chunk=64):
    """Sparse attention over selected rows; idxs < 0 marks an unused slot.

    A single-token call (every decode step) runs the reference loop, so decode
    arithmetic is exactly what it was. Longer calls (prefill) process `chunk`
    tokens at a time with padded gathers instead of one Python iteration per
    token. Same equation, but batched matrix products may round differently in
    the last bit. FRUSTUM_ATTENTION=loop forces the reference everywhere.
    """
    bsz, seqlen, heads, dim = q.shape
    if seqlen == 1 or idxs.shape[-1] == 0 or os.environ.get('FRUSTUM_ATTENTION') == 'loop':
        return sparse_attention_reference(q, kv, sink, idxs)
    out = torch.zeros_like(q)
    scale = dim ** -0.5
    sink_f = sink.float()[None, None, :, None]
    batch = torch.arange(bsz, device=q.device)[:, None, None]
    for start in range(0, seqlen, chunk):
        ids = idxs[:, start:start + chunk]
        valid = ids >= 0
        values = kv[batch, ids.clamp_min(0).long()].float()  # [batch, tokens, slots, dim]
        score = torch.einsum('bchd,bckd->bchk', q[:, start:start + chunk].float(), values) * scale
        score = score.masked_fill(~valid[:, :, None, :], -torch.inf)
        maximum = score.amax(-1, keepdim=True)
        prob = torch.exp(score - maximum)
        denom = prob.sum(-1, keepdim=True) + torch.exp(sink_f - maximum)
        mixed = torch.einsum('bchk,bckd->bchd', prob.to(torch.bfloat16).float(), values) / denom
        # A token with no selected rows contributes zeros, as in the reference.
        mixed = torch.where(valid.any(-1)[:, :, None, None], mixed, torch.zeros_like(mixed))
        out[:, start:start + chunk] = mixed.to(q.dtype)
    return out


def group_positions(start, count, ratio, device):
    """First positions of the `count` compressed groups completed by tokens appended at `start`.

    The first group a call can complete is start // ratio (its last position is the first one
    at or after `start` with position % ratio == ratio - 1). For a single decode token this is
    the old positions[-1] - (ratio - 1); for a prefill it is positions[::ratio]."""
    first = start // ratio
    return torch.arange(first, first + count, device=device) * ratio


def append_window_indices(batch, seqlen, window, start, lo, device):
    """Sliding-window rows for tokens appended at `start` to state, over a key sequence that
    holds positions lo .. start + seqlen - 1. Each token gets the `window` positions ending at
    itself, in ascending order (the order RingKV.latest() gives a decode step); slots before
    position 0 are -1."""
    positions = torch.arange(start, start + seqlen, device=device).unsqueeze(1)
    keys = positions - window + 1 + torch.arange(window, device=device)
    ids = torch.where(keys >= 0, keys - lo, -1).int()
    return ids.unsqueeze(0).expand(batch, -1, -1).contiguous()


def window_indices(batch, seqlen, window, start, cache_len, device):
    if start:
        return torch.arange(cache_len, device=device, dtype=torch.int32).view(1, 1, -1).expand(batch, seqlen, -1).contiguous()
    width = min(seqlen, window)
    end = torch.arange(seqlen, device=device).unsqueeze(1)
    ids = (end - window + 1).clamp_min(0) + torch.arange(width, device=device)
    ids = torch.where(ids > end, -1, ids).int()
    return ids.unsqueeze(0).expand(batch, -1, -1).contiguous()


class DenseOps:
    """Quantized dense tensor reader shared by all four blocks and the Engram path."""
    def __init__(self, catalog, cache_mib):
        self.catalog, self.cfg = catalog, catalog.config['text_config']
        self.loader = FrustumLoader(catalog, 'cuda', cache_mib * MIB)
        self.scales_ok = set()

    def tensor(self, name):
        item = self.catalog.tensors[name]
        dtypes = {'BF16': torch.bfloat16, 'F32': torch.float32}
        if item.dtype not in dtypes:
            raise ValueError(f'Expected BF16/F32 tensor for {name}, got {item.dtype}')
        with self.loader.acquire(name) as raw:
            return raw.view(dtypes[item.dtype]).reshape(item.shape).clone()

    def _fp8_formats(self, name):
        weight = self.catalog.tensors[name + '.weight']
        if weight.dtype != 'F8_E4M3':
            raise ValueError(f'Unsupported dense weight format {weight.dtype} for {name}')
        scales = self.catalog.tensors[name + '.scale']
        if scales.dtype != 'F8_E8M0':
            raise ValueError(f'Unsupported dense scale format for {name}')
        return weight, scales

    @staticmethod
    def _decode_fp8(raw, scale_raw, weight, scales, name):
        value = raw.view(torch.float8_e4m3fn).reshape(weight.shape).float()
        scale = scale_raw.reshape(scales.shape)
        if bool((scale == 255).any()):
            raise ValueError(f'NaN E8M0 scale in {name}')
        factor = torch.exp2(scale.float() - 127)
        rows, cols = weight.shape
        if rows % 32 == 0 and cols % 32 == 0 and tuple(scales.shape) == (rows // 32, cols // 32):
            # Same element-by-element products as the expanded form below, without
            # materialising two more full-size FP32 copies of the matrix.
            value.view(rows // 32, 32, cols // 32, 32).mul_(factor[:, None, :, None])
            return value
        return value * factor.repeat_interleave(32, 0).repeat_interleave(32, 1)[:rows, :cols]

    def matrix(self, name):
        weight = self.catalog.tensors[name + '.weight']
        if weight.dtype in ('BF16', 'F32'):
            return self.tensor(name + '.weight')
        weight, scales = self._fp8_formats(name)
        with self.loader.acquire(name + '.weight') as raw, self.loader.acquire(name + '.scale') as scale_raw:
            return self._decode_fp8(raw, scale_raw, weight, scales, name)

    def linear(self, x, name):
        if self.catalog.tensors[name + '.weight'].dtype != 'F8_E4M3':
            return F.linear(x.float(), self.matrix(name).float()).to(torch.bfloat16)
        x = activation(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        weight, scales = self._fp8_formats(name)
        with self.loader.acquire(name + '.weight') as raw, self.loader.acquire(name + '.scale') as scale_raw:
            def reference():
                return F.linear(x.float(), self._decode_fp8(raw, scale_raw, weight, scales, name).float()).to(torch.bfloat16)
            fused = fused_kernels.get(torch)
            if not fused.enabled or x.reshape(-1, x.shape[-1]).shape[0] > fused_kernels.MAX_TOKENS:
                return reference()
            if name not in self.scales_ok:  # E8M0 255 is NaN; the reference rejects it on every call
                if bool((scale_raw == 255).any()):
                    raise ValueError(f'NaN E8M0 scale in {name}')
                self.scales_ok.add(name)
            return fused.block(x, raw.reshape(weight.shape), scale_raw.reshape(scales.shape), False, reference,
                               torch.bfloat16)

    def float_linear(self, x, name):
        """Compressor projection promoted to FP32 before learned pooling.

        The supplied module declares the ratio>1 projections in F32; this checkpoint stores
        their payload in BF16, so load it faithfully and promote only for the documented pooling.
        """
        if self.catalog.tensors[name + '.weight'].dtype not in ('F32', 'BF16'):
            raise ValueError(f'Expected F32/BF16 compressor projection: {name}')
        return F.linear(x.float(), self.matrix(name).float())

    def norm(self, x, name):
        return rms_norm(x, self.tensor(name + '.weight'), self.cfg['rms_norm_eps'])

    def close(self):
        self.loader.close()


class CompressedState:
    def __init__(self, capacity, ratio):
        self.capacity, self.ratio = capacity, ratio
        self.reset()

    def reset(self):
        self.kv = self.index_k = self.kv_tail = self.score_tail = None

    def _ensure(self, name, bsz, dim, dtype, device):
        value = getattr(self, name)
        if value is None:
            value = torch.zeros(bsz, (self.capacity // self.ratio) + 1, dim, dtype=dtype, device=device)
            setattr(self, name, value)
        if value.shape[0] != bsz:
            raise ValueError('Batch changed without reset')
        return value

    def write_kv(self, value, group_start):
        cache = self._ensure('kv', value.shape[0], value.shape[-1], value.dtype, value.device)
        cache[:, group_start:group_start + value.shape[1]] = value

    def write_index(self, value, group_start):
        cache = self._ensure('index_k', value.shape[0], value.shape[-1], value.dtype, value.device)
        cache[:, group_start:group_start + value.shape[1]] = value


class Compressor:
    def __init__(self, ops, layer, state):
        self.ops, self.layer, self.state = ops, layer, state
        self.cfg, self.ratio = ops.cfg, ops.cfg['compress_ratios'][layer]

    def __call__(self, x, start):
        bsz, seqlen, _ = x.shape
        p = f'layers.{self.layer}.attn.compressor.'
        if self.ratio == 1:
            return self.ops.norm(self.ops.linear(x, p + 'wkv'), p + 'norm')
        kv = self.ops.float_linear(x, p + 'wkv')
        score = self.ops.float_linear(x, p + 'wgate')
        if start == 0:
            cutoff, remainder = seqlen - seqlen % self.ratio, seqlen % self.ratio
            if remainder:
                self.state.kv_tail = F.pad(kv[:, cutoff:], (0, 0, 0, self.ratio - remainder))
                self.state.score_tail = F.pad(score[:, cutoff:], (0, 0, 0, self.ratio - remainder), value=-torch.inf)
            if not cutoff:
                return None
            kv = kv[:, :cutoff].unflatten(1, (-1, self.ratio))
            score = score[:, :cutoff].unflatten(1, (-1, self.ratio))
            pooled = (kv * score.softmax(2)).sum(2)
        else:
            # Appending to existing state. One token is a decode step; v0.9 also accepts several
            # (a batched follow-up prompt). Each token goes through the decode step's slot logic
            # in order, so every pooled group is computed exactly as one-at-a-time decode does.
            if self.state.kv_tail is None:
                self.state.kv_tail = torch.zeros(bsz, self.ratio, kv.shape[-1], device=x.device, dtype=torch.float32)
                self.state.score_tail = torch.full_like(self.state.kv_tail, -torch.inf)
            groups = []
            for i in range(seqlen):
                slot = (start + i) % self.ratio
                self.state.kv_tail[:, slot] = kv[:, i]
                self.state.score_tail[:, slot] = score[:, i]
                if slot == self.ratio - 1:
                    groups.append((self.state.kv_tail * self.state.score_tail.softmax(1)).sum(1, keepdim=True))
            if not groups:
                return None
            pooled = groups[0] if len(groups) == 1 else torch.cat(groups, 1)
        return self.ops.norm(pooled.to(x.dtype), p + 'norm')


class Indexer:
    def __init__(self, ops, layer, shared, owns_keys):
        self.ops, self.layer, self.shared = ops, layer, shared
        self.owns_keys = owns_keys
        self.cfg, self.ratio = ops.cfg, ops.cfg['compress_ratios'][layer]

    def __call__(self, x, qr, latent, start, offset, positions):
        bsz, seqlen, _ = x.shape
        compressed_len = (start + seqlen) // self.ratio
        p = f'layers.{self.layer}.attn.indexer.'
        rd = self.cfg['qk_rope_head_dim']
        state = self.shared.active
        if self.owns_keys and latent is not None:
            freq_positions = (positions[:seqlen - seqlen % self.ratio:self.ratio] if start == 0
                              else group_positions(start, latent.shape[1], self.ratio, x.device))
            k = self.ops.norm(self.ops.linear(latent, p + 'wk'), p + 'k_norm')
            k[..., -rd:] = rope(k[..., -rd:], freq_positions, self.cfg['compress_rope_theta'], self.cfg['rope_scaling']['original_max_position_embeddings'], self.cfg['rope_scaling']['factor'], self.cfg['rope_scaling']['beta_fast'], self.cfg['rope_scaling']['beta_slow'])
            k = fp4_quant_dequant(k, 32, power2_scale=True)
            state.write_index(k, start // self.ratio)
        if compressed_len == 0:
            return torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        if state.index_k is None:
            raise RuntimeError('Indexer requested compressed keys before source wrote them')
        heads, dim = self.cfg['index_n_heads'], self.cfg['index_head_dim']
        q = self.ops.linear(qr, p + 'wq_b').reshape(bsz, seqlen, heads, dim)
        q[..., -rd:] = rope(q[..., -rd:], positions, self.cfg['compress_rope_theta'], self.cfg['rope_scaling']['original_max_position_embeddings'], self.cfg['rope_scaling']['factor'], self.cfg['rope_scaling']['beta_fast'], self.cfg['rope_scaling']['beta_slow'])
        q = fp4_quant_dequant(q, 32, power2_scale=True)
        weights = self.ops.linear(x, p + 'weights_proj') * (dim ** -0.5 * heads ** -0.5)
        key = state.index_k[:, :compressed_len]
        score = (torch.einsum('bshd,btd->bsht', q, key).relu() * weights.unsqueeze(-1)).sum(2)
        if start == 0:
            visible = (torch.arange(1, seqlen + 1, device=x.device) // self.ratio).view(1, seqlen, 1)
        elif seqlen == 1:
            visible = torch.full((1, seqlen, 1), compressed_len, device=x.device)
        else:
            # v0.9 batched append: each token sees the groups completed up to its own position.
            visible = ((positions + 1) // self.ratio).view(1, seqlen, 1)
        score.masked_fill_(torch.arange(compressed_len, device=x.device).view(1, 1, -1) >= visible, -torch.inf)
        candidate_layer = self.cfg['candidate_source_layer_id']
        if self.layer == candidate_layer:
            block = self.cfg['candidate_block_size']
            scores = F.pad(score, (0, -compressed_len % block), value=-torch.inf)
            scores = scores.unflatten(-1, (-1, block)).amax(-1)
            scores = scores.masked_fill(torch.arange(scores.size(-1), device=x.device) == (visible - 1) // block, torch.inf)
            best = scores.topk(min(self.cfg['candidate_topk_blocks'], scores.size(-1)), dim=-1)
            keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, best.indices, best.values > -torch.inf)
            self.shared.candidates = keep.repeat_interleave(block, -1)[..., :compressed_len]
        elif self.layer > candidate_layer >= 0:
            score.masked_fill_(~self.shared.candidates, -torch.inf)
        topk = min(self.cfg['index_topk'], compressed_len)
        ids = score.topk(topk, -1, sorted=False).indices.sort(-1).values
        return torch.where(ids < visible, ids + offset, -1).int()


class Attention:
    def __init__(self, ops, layer, shared, capacity):
        self.ops, self.layer, self.shared = ops, layer, shared
        self.cfg = ops.cfg
        self.ring = RingKV(self.cfg['sliding_window'])
        self.ratio = self.cfg['compress_ratios'][layer]
        self.is_source = layer in self.cfg['kv_source_layer_ids']
        self.is_index_source = layer in self.cfg['index_source_layer_ids']
        if self.is_source:
            self.source_state = shared.sources[layer] = CompressedState(capacity, self.ratio)
            self.compressor = Compressor(ops, layer, self.source_state)
        else:
            self.source_state = None
            self.compressor = None
        self.indexer = Indexer(ops, layer, shared, self.is_source) if self.is_index_source else None

    def reset(self):
        self.ring.reset()

    def _window(self, x, positions, start):
        p, rd = f'layers.{self.layer}.attn.', self.cfg['qk_rope_head_dim']
        kv = self.ops.norm(self.ops.linear(x, p + 'wkv'), p + 'kv_norm')
        kv[..., -rd:] = rope(
            kv[..., -rd:], positions,
            self.cfg['rope_theta'] if not self.ratio else self.cfg['compress_rope_theta'],
            0 if not self.ratio else self.cfg['rope_scaling']['original_max_position_embeddings'],
            1 if not self.ratio else self.cfg['rope_scaling']['factor'],
            self.cfg['rope_scaling']['beta_fast'], self.cfg['rope_scaling']['beta_slow'],
        )
        kv = activation(kv.reshape(-1, kv.shape[-1])).reshape(kv.shape).to(torch.bfloat16)
        window_size = self.cfg['sliding_window']
        if start and x.shape[1] > 1:
            # v0.9 batched append. The ring only holds `window` positions, so appending first would
            # overwrite rows the earlier new tokens still need. Take the window-1 most recent rows
            # before appending, put the new rows after them, and give each token its own window.
            older = self.ring.recent(window_size - 1)
            lo = start - older.shape[1]
            keys = torch.cat((older, kv), 1)
            self.ring.append(kv, start)
            return keys, append_window_indices(x.shape[0], x.shape[1], window_size, start, lo, x.device)
        self.ring.append(kv, start)
        window = kv if start == 0 else self.ring.latest()
        return window, window_indices(x.shape[0], x.shape[1], window_size, start, window.shape[1], x.device)

    def _compressed(self, x, qr, start, offset, positions):
        compressed_len = (start + x.shape[1]) // self.ratio
        if self.is_source:
            self.shared.active = self.source_state
        latent = self.compressor(x, start) if self.is_source else None
        if self.is_index_source:
            indexes = self.indexer(x, qr, latent, start, offset, positions)
            self.shared.topk = indexes
        elif self.shared.topk is None:
            raise RuntimeError('Compressed-attention consumer ran before its index source')
        else:
            indexes = self.shared.topk
        if latent is not None:
            rd = self.cfg['qk_rope_head_dim']
            latent_positions = (positions[:x.shape[1] - x.shape[1] % self.ratio:self.ratio] if start == 0
                                else group_positions(start, latent.shape[1], self.ratio, x.device))
            latent[..., -rd:] = rope(latent[..., -rd:], latent_positions, self.cfg['compress_rope_theta'], self.cfg['rope_scaling']['original_max_position_embeddings'], self.cfg['rope_scaling']['factor'], self.cfg['rope_scaling']['beta_fast'], self.cfg['rope_scaling']['beta_slow'])
            latent = fp4_quant_dequant(latent, 16, power2_scale=False)
            self.source_state.write_kv(latent, start // self.ratio)
        if compressed_len == 0:
            return torch.empty(x.shape[0], 0, self.cfg['head_dim'], dtype=x.dtype, device=x.device), indexes
        if self.shared.active is None or self.shared.active.kv is None:
            raise RuntimeError('Compressed KV requested before source wrote it')
        return self.shared.active.kv[:, :compressed_len], indexes

    def __call__(self, x, start):
        bsz, seqlen, _ = x.shape
        positions = torch.arange(start, start + seqlen, device=x.device)
        p, rd = f'layers.{self.layer}.attn.', self.cfg['qk_rope_head_dim']
        qr = self.ops.norm(self.ops.linear(x, p + 'wq_a'), p + 'q_norm')
        q = self.ops.linear(qr, p + 'wq_b').reshape(bsz, seqlen, self.cfg['num_attention_heads'], self.cfg['head_dim'])
        q[..., -rd:] = rope(q[..., -rd:], positions, self.cfg['rope_theta'] if not self.ratio else self.cfg['compress_rope_theta'], 0 if not self.ratio else self.cfg['rope_scaling']['original_max_position_embeddings'], 1 if not self.ratio else self.cfg['rope_scaling']['factor'], self.cfg['rope_scaling']['beta_fast'], self.cfg['rope_scaling']['beta_slow'])
        kv, indexes = self._window(x, positions, start)
        if self.ratio:
            compressed, compressed_indexes = self._compressed(x, qr, start, kv.shape[1], positions)
            kv, indexes = torch.cat((kv, compressed), 1), torch.cat((indexes, compressed_indexes), -1)
        sink = self.ops.tensor(p + 'attn_sink')
        # Appending to state (decode, or a v0.9 batched follow-up prompt) uses the per-token reference
        # loop, so each token gathers and sums its rows in the same order as one-at-a-time decode.
        o = sparse_attention_reference(q, kv, sink, indexes) if start else sparse_attention(q, kv, sink, indexes)
        o[..., -rd:] = rope(o[..., -rd:], positions, self.cfg['rope_theta'] if not self.ratio else self.cfg['compress_rope_theta'], 0 if not self.ratio else self.cfg['rope_scaling']['original_max_position_embeddings'], 1 if not self.ratio else self.cfg['rope_scaling']['factor'], self.cfg['rope_scaling']['beta_fast'], self.cfg['rope_scaling']['beta_slow'], inverse=True)
        groups, rank = self.cfg['o_groups'], self.cfg['o_lora_rank']
        o = activation(o.reshape(bsz, seqlen, groups, -1).reshape(-1, self.cfg['head_dim'] * self.cfg['num_attention_heads'] // groups)).reshape(bsz, seqlen, groups, -1).to(torch.bfloat16)
        wa = self.ops.matrix(p + 'wo_a').to(torch.bfloat16).reshape(groups, rank, -1)
        latent = torch.einsum('bsgd,grd->bsgr', o, wa)
        return self.ops.linear(latent.flatten(2), p + 'wo_b')


class SharedAttention:
    def __init__(self):
        self.sources = {}
        self.begin_step()

    def begin_step(self):
        self.active = self.topk = self.candidates = None

    def reset(self):
        for state in self.sources.values():
            state.reset()
        self.begin_step()


def normalized_token_map(model_root, expected_size):
    """Exact tokenizer/normalizer route used by supplied inference/engram.py."""
    from tokenizers import Regex, Tokenizer, normalizers
    tokenizer = Tokenizer.from_file(str(Path(model_root) / 'tokenizer.json'))
    sentinel = '\ue000'
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r'[ \t\r\n]+'), ' '), normalizers.Replace(Regex(r'^ $'), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, ' '),
    ])
    keys, result = {}, []
    for token_id in range(tokenizer.get_vocab_size(with_added_tokens=True)):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if '\ufffd' in text:
            key = tokenizer.id_to_token(token_id)
        else:
            value = normalizer.normalize_str(text)
            key = value if value else text
        if key not in keys:
            keys[key] = len(keys)
        result.append(keys[key])
    if len(keys) != expected_size:
        raise AssertionError(f'Compressed tokenizer vocabulary mismatch: {len(keys)} != {expected_size}')
    return torch.tensor(result, dtype=torch.long), len(keys)


def next_prime(value, seen):
    def prime(candidate):
        if candidate < 2:
            return False
        if candidate % 2 == 0:
            return candidate == 2
        limit = int(candidate ** .5)
        return all(candidate % divisor for divisor in range(3, limit + 1, 2))
    value += 1
    while value in seen or not prime(value):
        value += 1
    seen.add(value)
    return value


class EngramHashes:
    def __init__(self, model_root, cfg):
        self.cfg = cfg
        self.token_map, self.compressed_vocab_size = normalized_token_map(model_root, cfg['engram_compressed_vocab_size'])
        seen = set()
        self.tables = {}
        bound = (np.iinfo(np.int64).max // cfg['engram_compressed_vocab_size']) // 2
        for layer, size in zip(cfg['engram_layer_ids'], cfg['engram_num_embeddings'], strict=True):
            primes, offsets = [], []
            offset = 0
            for _ in range(cfg['engram_max_ngram_size'] - 1):
                row = [next_prime(cfg['engram_vocab_size'] - 1, seen) for _ in range(cfg['engram_n_heads'])]
                primes.append(row)
                offsets.append([offset + sum(row[:i]) for i in range(len(row))])
                offset += sum(row)
            if offset != size:
                raise ValueError(f'Engram bucket count mismatch for layer {layer}')
            multipliers = np.random.default_rng(10007 * layer).integers(
                0, bound, size=cfg['engram_max_ngram_size'], dtype=np.int64) * 2 + 1
            self.tables[layer] = (torch.from_numpy(multipliers), torch.tensor(primes), torch.tensor(offsets))
        self.pad_id = int(self.token_map[cfg['engram_pad_token_id']])
        self.history = None

    def reset(self):
        self.history = None

    def __call__(self, ids, start):
        source = self.token_map[ids.cpu()].to(ids.device)
        if start == 0:
            self.history = source.clone()
        else:
            if self.history is None or self.history.shape[0] != source.shape[0] or self.history.shape[1] != start:
                raise ValueError('Engram hash position mismatch')
            self.history = torch.cat((self.history, source), 1)
        positions = torch.arange(start, start + ids.shape[1], device=ids.device).expand_as(source)
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        tokens = []
        for shift in range(self.cfg['engram_max_ngram_size']):
            gathered = self.history.gather(1, (positions - shift).clamp_min(0))
            blocked |= positions < shift
            tokens.append(torch.where(blocked, self.pad_id, gathered))
        tokens = torch.stack(tokens, -1)
        hashes = {}
        for layer, (multipliers, primes, offsets) in self.tables.items():
            products = tokens * multipliers.to(ids.device)
            rolling, values = products[..., 0], []
            for shift in range(1, self.cfg['engram_max_ngram_size']):
                rolling = torch.bitwise_xor(rolling, products[..., shift])
                values.append(rolling[..., None] % primes[shift - 1].to(ids.device) + offsets[shift - 1].to(ids.device))
            hashes[layer] = torch.cat(values, -1)
        return hashes


class Engram:
    def __init__(self, ops, lookup):
        self.ops, self.lookup = ops, lookup
        self.cfg = ops.cfg

    def __call__(self, h, hashes, layer):
        p = f'layers.{layer}.engram.'
        rows = self.lookup.get(p + 'embed.weight', hashes.flatten().tolist(), p + 'embed.scale').reshape(*hashes.shape, self.cfg['engram_head_dim'])
        kv = self.ops.linear(rows.flatten(-2), p + 'wkv')
        key, value = kv.split([self.cfg['hc_mult'] * self.cfg['hidden_size'], self.cfg['hidden_size']], -1)
        key = key.float().reshape(*key.shape[:-1], self.cfg['hc_mult'], self.cfg['hidden_size'])
        h_float = h.float()
        weight = self.ops.tensor(p + 'q_weight').float() * self.ops.tensor(p + 'k_weight').float()
        rstd = torch.rsqrt(h_float.square().mean(-1) + self.cfg['rms_norm_eps']) * torch.rsqrt(key.square().mean(-1) + self.cfg['rms_norm_eps'])
        dot = (h_float * weight * key).sum(-1) * rstd * self.cfg['hidden_size'] ** -.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (h_float + gate[..., None] * value.float().unsqueeze(-2)).to(h.dtype)


class Block:
    def __init__(self, ops, moe, layer, shared, capacity):
        self.ops, self.moe, self.layer = ops, moe, layer
        self.attention = Attention(ops, layer, shared, capacity)

    def reset(self):
        self.attention.reset()

    def mixes(self, h, part):
        p = f'layers.{self.layer}.hc_{part}_'
        return hc_mixes(h, self.ops.tensor(p + 'fn'), self.ops.tensor(p + 'scale'), self.ops.tensor(p + 'base'), hc=self.ops.cfg['hc_mult'], iters=self.ops.cfg['hc_sinkhorn_iters'], eps=self.ops.cfg['hc_eps'], norm_eps=self.ops.cfg['rms_norm_eps'])

    def __call__(self, h, start, pre_mix, validate):
        prefix = f'layers.{self.layer}.'
        trace = {'input': tensor_sha256(h)}
        attn_pre, attn_post, attn_comb = self.mixes(h, 'attn')
        a = self.attention(self.ops.norm(hc_pre(h, pre_mix), prefix + 'attn_norm'), start)
        trace['attention_output'] = tensor_sha256(a)
        h = hc_post(a, h, attn_post, attn_comb)
        trace['attention_residual'] = tensor_sha256(h)
        ffn_pre, ffn_post, ffn_comb = self.mixes(h, 'ffn')
        x = self.ops.norm(hc_pre(h, attn_pre), prefix + 'ffn_norm')
        trace['moe_input'] = tensor_sha256(x)
        f, record = self.moe.forward(x.reshape(-1, x.shape[-1]), self.layer, validate=validate)
        h = hc_post(f.reshape(x.shape), h, ffn_post, ffn_comb)
        if not torch.isfinite(h).all():
            raise AssertionError(f'Nonfinite output in layer {self.layer}')
        trace['output'] = tensor_sha256(h)
        return h, ffn_pre, record, trace


class SequentialBlocks:
    def __init__(self, catalog, sequence_capacity, dense_cache_mib, expert_cache_mib, layer_count=4, moe_options=None):
        """dense_cache_mib at least the size of all dense weights makes them resident.
        moe_options are passed to StreamedMoE (host tier, router cache, policies, threads).
        """
        self.catalog, self.cfg = catalog, catalog.config['text_config']
        if self.cfg['compress_ratios'][:4] != [0, 0, 2, 2] or tuple(self.cfg['kv_source_layer_ids'][:1]) != (2,) or tuple(self.cfg['index_source_layer_ids'][:1]) != (2,):
            raise ValueError('Unexpected first-four-layer attention schedule')
        self.capacity = sequence_capacity
        self.hasher = EngramHashes(catalog.root, self.cfg)
        self.ops = DenseOps(catalog, dense_cache_mib)
        self.moe = StreamedMoE(catalog, expert_cache_mib, **(moe_options or {}))
        self.lookup = SparseRows(catalog)
        self.shared = SharedAttention()
        self.engram = Engram(self.ops, self.lookup)
        self.blocks = [Block(self.ops, self.moe, layer, self.shared, sequence_capacity) for layer in range(layer_count)]

    def reset(self):
        self.shared.reset()
        self.hasher.reset()
        for block in self.blocks:
            block.reset()

    def forward(self, ids, h, start=0, pre_mix=None, validate=False, progress=None):
        if not 0 <= start < start + ids.shape[1] <= self.capacity:
            raise ValueError('Sequence exceeds configured capacity')
        if start == 0:
            self.reset()
        self.shared.begin_step()
        hashes = self.hasher(ids, start)
        if pre_mix is None:
            pre_mix = torch.zeros(*h.shape[:-1], dtype=torch.float32, device=h.device)
            pre_mix[..., 0] = 1
        traces, records = [], []
        for layer, block in enumerate(self.blocks):
            if layer in hashes:
                h = self.engram(h, hashes[layer], layer)
            h, pre_mix, record, trace = block(h, start, pre_mix, validate)
            traces.append(trace)
            records.append(record)
            if progress is not None:
                progress(layer + 1, len(self.blocks))
        return h, pre_mix, records, traces

    def memory(self):
        cap = 12 * GIB
        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        if self.ops.loader.used > self.ops.loader.budget or self.moe.cache.used > self.moe.cache.budget:
            raise AssertionError('A bounded cache exceeded its configured budget')
        if allocated > cap or reserved > cap:
            raise AssertionError('12 GiB allocator guard exceeded')
        return dict(allocated_GiB=allocated / GIB, reserved_GiB=reserved / GIB,
                    dense_cache_MiB=self.ops.loader.used / MIB, dense_cache_budget_MiB=self.ops.loader.budget / MIB,
                    expert_cache_MiB=self.moe.cache.used / MIB, expert_cache_budget_MiB=self.moe.cache.budget / MIB)

    def close(self):
        self.moe.close()
        self.ops.close()


def primitive_tests():
    torch.manual_seed(7)
    q = torch.randn(1, 4, 3, 8, device='cuda').to(torch.bfloat16)
    kv = torch.randn(1, 4, 8, device='cuda').to(torch.bfloat16)
    sink = torch.tensor([-.2, .3, 1.], device='cuda')
    ids = window_indices(1, 4, 3, 0, 4, q.device)
    actual = sparse_attention(q, kv, sink, ids).cpu().float().numpy()
    expected = np.zeros_like(actual)
    qn, kn = q.cpu().float().numpy().astype(np.float64), kv.cpu().float().numpy().astype(np.float64)
    for pos in range(4):
        values = kn[0, max(0, pos - 2):pos + 1]
        z = qn[0, pos] @ values.T / np.sqrt(8)
        mx, p = z.max(-1, keepdims=True), np.exp(z - z.max(-1, keepdims=True))
        pb = torch.from_numpy(p).to(torch.bfloat16).float().numpy()
        expected[0, pos] = pb @ values / (p.sum(-1, keepdims=True) + np.exp(sink.cpu().numpy()[:, None] - mx))
    np.testing.assert_allclose(actual, expected, rtol=.015, atol=.01)
    changed = kv.clone(); changed[:, 3] *= 100
    # Causality: scaling the last key must not move earlier tokens. Exact for the reference loop;
    # the batched path is held to rounding tolerance (a leak would be a 100x change).
    torch.testing.assert_close(sparse_attention_reference(q, changed, sink, ids)[:, :3], sparse_attention_reference(q, kv, sink, ids)[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(sparse_attention(q, changed, sink, ids)[:, :3].float(), torch.from_numpy(actual[:, :3]).cuda(), rtol=.02, atol=.01)
    # Batched path against the reference loop: padded slots, holes and a token with no rows.
    generator = torch.Generator().manual_seed(11)
    bq = torch.randn(2, 70, 3, 8, generator=generator).to(torch.bfloat16).cuda()
    bkv = torch.randn(2, 40, 8, generator=generator).to(torch.bfloat16).cuda()
    bids = torch.randint(0, 40, (2, 70, 12), generator=generator).sort(-1).values
    bids[torch.rand(2, 70, 12, generator=generator) < .3] = -1
    bids[:, 5] = -1
    bids = bids.int().cuda()
    batched, looped = sparse_attention(bq, bkv, sink, bids, chunk=32), sparse_attention_reference(bq, bkv, sink, bids)
    torch.testing.assert_close(batched.float(), looped.float(), rtol=.02, atol=.01)
    if batched[:, 5].abs().max().item() != 0 or not torch.isfinite(batched).all():
        raise AssertionError('Batched attention mishandled a token with no selected rows')
    ring = RingKV(3)
    for pos in range(8):
        ring.append(torch.full((1, 1, 8), float(pos), device='cuda', dtype=torch.bfloat16), pos)
        assert ring.latest()[0, :, 0].tolist() == list(range(max(0, pos - 2), pos + 1))
    value = torch.randn(1, 4, 3, 8, device='cuda'); positions = torch.arange(4, device='cuda')
    torch.testing.assert_close(rope(rope(value, positions, 160000, 65536, 16), positions, 160000, 65536, 16, inverse=True), value, rtol=2e-5, atol=2e-5)
    compressed = fp4_quant_dequant(torch.randn(2, 32, device='cuda'), 16, power2_scale=False)
    if not torch.isfinite(compressed).all():
        raise AssertionError('FP4 compressed activation quantization produced nonfinite values')
    return ['sparse attention NumPy oracle', 'batched attention vs reference loop', 'causal future-token masking', 'KV ring wraparound', 'YaRN RoPE inverse', 'compressed FP4 quantization finite']


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=default_model_root())
    parser.add_argument('--tokens', default='100,200,300', help='Three or more comma-separated text token IDs')
    parser.add_argument('--dense-cache-mib', type=int, default=256)
    parser.add_argument('--expert-cache-mib', type=int, default=256)
    parser.add_argument('--output', default='layers0_3_results.json')
    args = parser.parse_args()
    ids_list = [int(item) for item in args.tokens.split(',')]
    if len(ids_list) < 3 or any(item < 0 or item >= 129280 for item in ids_list):
        parser.error('--tokens requires at least three valid token IDs')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    free, total = torch.cuda.mem_get_info()
    if total < 14 * GIB or free < 3 * GIB:
        raise SystemExit('Insufficient dedicated VRAM/headroom')
    torch.cuda.set_per_process_memory_fraction(12 * GIB / total)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    tests = primitive_tests()
    catalog = Catalog(args.model)
    engine = SequentialBlocks(catalog, len(ids_list), args.dense_cache_mib, args.expert_cache_mib)
    token_ids = torch.tensor([ids_list], dtype=torch.long, device='cuda')
    embeds = engine.lookup.get('embed.weight', ids_list).unsqueeze(0)
    h = embeds.unsqueeze(2).repeat(1, 1, engine.cfg['hc_mult'], 1)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        full, pre, records, traces = engine.forward(token_ids, h, 0, validate=True)
        full_memory = engine.memory()
        engine.reset()
        split = len(ids_list) - 1
        prefix, _, _, _ = engine.forward(token_ids[:, :split], h[:, :split], 0, validate=False)
        last, last_pre, _, _ = engine.forward(token_ids[:, split:], h[:, split:], split, validate=False)
        incremental = torch.cat((prefix, last), 1)
        a, b = full.float(), incremental.float()
        relative = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-12))
        torch.testing.assert_close(a, b, rtol=.02, atol=.01)
        torch.testing.assert_close(pre[:, split:], last_pre, rtol=.02, atol=.01)
        report = dict(
            status='PASS',
            scope='Sequential complete text blocks 0--3 only; no head, generation, or later source-sharing schedule',
            attention_schedule=dict(layers_0_1='sliding-window raw KV', layer_1='Engram applied before Block 1', layer_2='2:1 compressor and index source', layer_3='consumes layer-2 shared compressed KV/index selections'),
            checkpoint_attention_source='local inference/model.py Attention, Compressor, Indexer, Block, and Transformer.forward inspected before implementation',
            token_ids=ids_list, output_shape=list(full.shape), output_sha256=tensor_sha256(full),
            tokenizer_compressed_vocab_size=engine.hasher.compressed_vocab_size,
            primitive_tests=tests, full_prefill_trace=traces, full_prefill_moe=records,
            full_prefill_vs_prefix_decode_relative_L2=relative,
            max_abs_prefill_decode_difference=float((a - b).abs().max()),
            correctness_checks=dict(moe='Per-layer NumPy FP64 routing/matrix-product references passed', attention='Tiny NumPy sparse-attention oracle plus causal masking passed', schedule='Full prefill versus prefix+decode agrees within 0.02 rtol / 0.01 atol'),
            memory_full_prefill=full_memory, memory_after_schedule=engine.memory(),
            dense_cache_stats=engine.ops.loader.stats(),
            expert_cache_stats=dict(payload_bytes_read=engine.moe.cache.reader.bytes_read, cache_bytes=engine.moe.cache.used, budget_bytes=engine.moe.cache.budget, evictions=engine.moe.cache.evictions, requests=engine.moe.cache.requests, hits=engine.moe.cache.hits),
            sparse_row_stats=dict(payload_bytes_read=engine.lookup.reader.bytes_read, cache_bytes=engine.lookup.used, requests=engine.lookup.requests, hits=engine.lookup.hits),
            logical_weight_read_MB=(engine.ops.loader.reader.bytes_read + engine.moe.cache.reader.bytes_read + engine.moe.router.reader.bytes_read + engine.lookup.reader.bytes_read) / 1e6,
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / GIB,
            peak_reserved_GiB=torch.cuda.max_memory_reserved() / GIB,
            seconds_including_CPU_validation=time.perf_counter() - started,
            caveat='The eager attention/compressed-activation implementation follows supplied scheduling/equations but is not TileLang kernel parity. Full-sequence agreement compares two schedules of this implementation; long-context sparse-selection and all later layers remain unvalidated.'
        )
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
