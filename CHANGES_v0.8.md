# Frustum_loader v0.8: first measured v0.7 baseline, unbuffered reads, and the per-layer stall

Base: `bd8d047` (v0.7 docs). Every number below was measured on the target machine on
2026-11-16: RTX 5060 Ti 16 GB, Ryzen 7 3700X, 32 GB RAM, Windows 11, DeepSeek V4.1 Flash on a
Samsung T7 connected to a 10 Gbps USB port. Benchmarks are `bench_e2e.py` runs (fresh process
per run, 3 repetitions, prompt "Do you like Dino Buddies? Reply in one short sentence.",
24 max new tokens, greedy; the reply ends after 15 tokens). Summaries are in `v08_logs/`.

## 1. The drive link comes first

A first benchmark attempt ran with the T7 at about 42 MB/s (a slow port or cable) and was
stopped. After reconnecting to a 10 Gbps port, unbuffered 19 MiB expert-sized reads measured:

| Parallel reads | 1 | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| MB/s | 719 | 848 | 909 | 905 | 883 | 843 |

Four reader threads (the default) are already at the drive's ceiling. More threads do not help.

## 2. First measured v0.7 numbers (`v08_logs/v08_patch_vs_unbuffered_2_*`)

| Metric (median) | v0.7 defaults (`patch`, buffered reads) |
|---|---|
| Time to first token | 63.0 s |
| Decode | **5.69 s/token** |
| Expert bytes requested per token | 2.43 GB (1.11 GB first-time, 1.32 GB re-read) |
| Physical / logical bytes | 1.27 |
| Effective drive throughput while busy | 580 MB/s |
| Drive utilisation / read-compute overlap | 73% / 23% |

Two findings: Windows read 27% more from the drive than the engine asked for (unused
read-ahead), and the OS file cache gave nothing back: "warm" runs were as slow as cold ones
(5.66 vs 5.69 s/token), because 32 GB of RAM cannot hold enough of the model.

## 3. Unbuffered reads and a larger RAM tier (on by default on Windows)

- `RangeReader(mode='unbuffered')`: Windows `FILE_FLAG_NO_BUFFERING` reads, sector-aligned
  through a per-thread aligned bounce buffer. The OS cache never holds a duplicate of expert bytes.
- `placement.plan(unbuffered=True)`: the automatic host tier takes 85% of RAM above 6 GiB
  instead of 50% (still capped at 16 GiB, still shrunk by the 3 GiB free-RAM guard).
- `frustum_chat.py --reader {direct,unbuffered}` / `FRUSTUM_READER`; unbuffered is the default on
  Windows, direct elsewhere. The evidence JSON records the reader.

| Metric (median of 3 cold + 3 warm) | Buffered (v0.7) | Unbuffered (v0.8) |
|---|---|---|
| Decode | 5.69 s/token | **4.06 s/token (-29%)** |
| Time to first token | 63.0 s | 58.1 s |
| Physical / logical bytes | 1.27 | 1.00 |
| Effective throughput | 580 MB/s | 863 MB/s |
| Host RAM tier | 8.2 GiB | 14.7 GiB |
| Re-read bytes per token | 1.29 GB | 0.97 GB |
| Token ids | identical across all 12 runs | |

The RAM-tier size and the reader changed together, so this measures the combination.

## 4. Where the remaining 4 s goes (`v08_logs/decode_section_profile.txt`)

`tools/profile_decode.py` brackets each section of a decode step with GPU synchronisation:

| Section | s/token |
|---|---|
| Waiting for experts (`cache.get`: drive reads + uploads) | 2.78 |
| Expert compute (280 calls) | 0.70 |
| Attention | 0.54 |
| Hyper-connection mixes | 0.37 |
| Other MoE bookkeeping | ~0.27 |
| Engram row reads, trace hashing, routing | ~0.08 |

Each layer alternates about 65 ms of reads (GPU idle) with about 55 ms of GPU work (drive
idle). Reads for layer N+1 cannot start until layer N's output exists, so the two halves only
overlap if layer N+1's experts are guessed.

## 5. Things tried that did not help (kept as opt-in switches)

`v08_logs/v08_pipeline_and_gated_*`, 3 runs each, token ids identical across all 9 runs:

| Configuration | Decode (median) | Notes |
|---|---|---|
| v0.8 default (unbuffered, blocking uploads) | **4.07 s/token** | |
| + pinned staging and copy-stream uploads (`FRUSTUM_ASYNC_UPLOAD=1`) | 4.33 s/token | Throughput fell to 823 MB/s. Uploads were already hidden behind reads; the extra pinned allocations and copies cost more than they removed |
| + gated next-layer prefetch (`--predict-next 6`, async uploads on) | 4.42 s/token | Overlap doubled (24% -> 47%) and disk-blocked time fell 1.85 -> 1.57 s/token, but wrong guesses added ~0.37 GB/token of reads |

Prediction accuracy in shadow mode (`v08_logs/v08_shadow_prediction_*`, reads nothing):

| Guess source | Precision | Precision on experts not already cached |
|---|---|---|
| Next layer's router on this layer's input | 71.7% | 59.9% (recent EMA ~0.70) |
| Same layer, previous token ("history") | 25.2% | 8.5% |

The gate (>= 0.6 cold precision) correctly switched router guesses on. They are right often
enough to keep the drive busier, but not right enough to pay for the bytes they waste on a
drive that is already the bottleneck. Prediction stays off by default.

## 6. Other changes in this release

- `streamed_moe.forward` uploads every expert's (rows, cols) for a layer in one pinned,
  non-blocking copy instead of one pageable copy per expert, and only copies the hidden state to
  the CPU when validating. Part of the measured 4.07 s/token default.
- `expert_cache.TieredExpertCache(async_upload=...)` and `inspect` hook (CPU-side NaN-scale
  check), used only when `FRUSTUM_ASYNC_UPLOAD=1`.
- `bench_e2e.py`: new `unbuffered`, `direct` and `async_upload` configurations; `baseline`
  pins the v0.7 buffered reader.
- Tests: unbuffered reads vs direct reads at unaligned offsets, across bounce-buffer chunks
  and threads, truncation detection; host-tier sizing; async uploads byte-identical to blocking
  uploads on CUDA, including host-tier staging and immediate use on the main stream.
  `selftest.py` 52 tests, `selftest_v07.py` 29 tests, all passing on the target machine.

## 7. What this points to next

1. Batch prompt tokens on follow-up turns. Follow-ups currently feed new prompt tokens one at
   a time (`generation.drive`) because attention only accepts one token per step when state
   exists (`layers0_3_block.py`, Attention, Compressor, Indexer). Expected gain is fewer
   per-layer stalls, not fewer bytes: repeated experts were already cache hits.
2. DSpark drafting. The checkpoint ships three MTP draft layers (`mtp.0-2`, 128 experts, top-3,
   plus a confidence head). Verification needs the same multi-token pass as item 1.
3. Cheaper GPU work per layer (attention, mixes and 280 small expert calls are ~1.6 s/token),
   for example CUDA graphs, once the drive stall is smaller.
4. Native FP4: `torch.nn.functional.scaled_mm` runs NVFP4 (1x16 E4M3 scales) on this GPU at
   about 326 TFLOPS for 1,024 rows, but is launch-bound (~73 us per call) for 1-8 rows, and the
   checkpoint's MXFP4 (1x32 E8M0) needs FBGEMM_GENAI. Unverified numerically; useful for prefill
   and draft verification, not single-token decode.
