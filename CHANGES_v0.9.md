# Frustum_loader v0.9: batched follow-up prompts

Base: `7bbee8b` (v0.8). Measured on the target machine on 2026-11-16: RTX 5060 Ti 16 GB,
Ryzen 7 3700X, 32 GB RAM, DeepSeek V4.1 Flash on a Samsung T7 at a 10 Gbps USB port, v0.8
defaults (unbuffered reads). Evidence is in `v09_logs/`.

## 1. The problem

On a follow-up turn the engine reuses the attention state of the conversation so far, but
v0.8 then fed the new prompt tokens one per pass through all 40 layers
(`generation.drive`), because attention only accepted one new token when state existed
(`layers0_3_block.Attention` raised "Decode requires one token per step"). Every pass pays
the per-layer expert-read stall measured in v0.8, so an 8-token follow-up such as
"Say one word." spent about 36 s before its first reply token.

## 2. What changed

Attention can now extend existing state by several tokens in one pass:

- **Sliding window.** The ring buffer holds only `sliding_window` (128) positions, so
  appending first would overwrite rows that earlier new tokens still need. The window-1 most
  recent rows are taken before appending (`RingKV.recent`), the new rows are placed after them,
  and each new token gets its own window of the 128 positions ending at itself, in ascending
  order (`append_window_indices`), which is the order `RingKV.latest()` gives a decode step.
- **Compressor.** Each new token goes through the decode step's slot logic in order, so every
  pooled group is computed with the same operations as one-at-a-time decode. Groups completed
  in the pass are written together (`group_positions` gives their positions).
- **Indexer.** Each token sees the compressed groups completed up to its own position,
  `(position + 1) // ratio`, the formula prefill already uses.
- **Attention arithmetic.** Appends use the per-token reference attention loop that decode
  uses, not prefill's batched path, so each token gathers and sums its rows in decode's order.
- Engram hashing and the MoE already handled several tokens.

`generation.drive(..., append_chunk=N)` feeds follow-up prompt tokens N per pass.
`frustum_chat.Session(append_chunk=...)`, `--append-chunk` and `FRUSTUM_APPEND_CHUNK` set N;
the default is 32. `--append-chunk 1` restores v0.8. `--followup TEXT` (repeatable, with
`--prompt`) scripts a multi-turn conversation in one process; the evidence JSON lists every turn
under `turns`, including how many passes each prompt took.

## 3. Results: full 40-layer, three-turn conversation (`v09_logs/followup_turns.json`)

Greedy, `--max-new-tokens 16 --predict-next 0`. Time to first token is the time to process the
new prompt tokens and produce the first reply token.

| Append chunk | Turn 1 (18-token prompt, prefill) | Turn 2 (8 new tokens) | Turn 3 (29 new tokens) | Tokens vs chunk 1 |
|---|---|---|---|---|
| 1 (v0.8) | 58.0 s | 35.8 s (8 passes) | 123.7 s (29 passes, 93 GB) | |
| 8 | 57.7 s | 22.5 s (1 pass) | 87.7 s (4 passes, 102 GB) | turns 1-2 identical; turn 3 diverges after ~10 tokens |
| **32 (default)** | 57.7 s | **22.1 s** (1 pass) | **60.1 s** (1 pass, 83 GB) | **all three turns identical** |

Follow-up time to first token falls by 38% for 8 new tokens and 51% for 29. Decode speed and
the first turn are unchanged. With chunk 8, turn 3's reply has the same meaning with different
wording ("USB interfaces have much lower..." vs "USB has much lower data...").

## 4. Correctness checks (`v09_logs/long_context_*`)

`long_context_check.py --append-chunk M` adds schedule C: prefill a prefix, then append the
remaining positions M at a time, compared with schedule B (one token per step, the v0.8 path).
160 tokens, so the 128-token window wraps; 16 or 24 appended positions; chunk 8.

| Layers | A (full prefill) vs B (decode) | C (batched append) vs B |
|---|---|---|
| 1 (layer 0: sliding window only) | 0.72% | **0.04%** (13 of 16 positions bit-identical) |
| 2 | 1.98% | 1.29% |
| 3 | 3.41% | 3.97% |
| 4 | 4.58% | 4.75% |

At layer 0 the batched path matches decode 17 times more closely than prefill does. The
remaining differences start at single positions (last-bit rounding in matrix products whose
result depends on the number of rows, such as the router's) and then spread to later tokens
through attention, which is the same mechanism as the existing prefill/decode gap.

**Existing issue found:** `long_context_check.py` fails its own tolerance (rtol 0.02, atol 0.01)
at 4 layers on unchanged v0.8 code too, with exactly the same A-vs-B numbers
(`long_context_4layers_v08code.json`). Prefill and one-at-a-time decode drift apart by a few
percent over 160 tokens and 4 layers. That predates v0.9; it also shows v0.9 left the decode
path unchanged. It is not investigated here.

Unit tests (`selftest.py BatchedAppendTests`, `GenerationTests.test_batched_append_chunks`):
window rows for appended tokens equal what decode gives each token, including wrap-around;
the compressor's pooled groups and tail state equal decode's bit for bit; group positions
match the decode and prefill formulas; drive splits prompts into chunks and asks for logits
only on the last one. All `selftest.py`, `frustum_loader.py` and `selftest_v07.py` tests pass.

## 5. Next

DSpark drafting: the checkpoint's three MTP draft layers propose several tokens, and the full
model checks them in one pass. That verification pass is this same multi-token append.
