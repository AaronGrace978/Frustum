# Frustum_loader v0.7 implementation and validation notes (GPU path unrun)

Base: `ed3a77f` (v0.6.0). The implementation described here is committed as v0.7. Nothing here has run on the RTX 5060 Ti
or against the checkpoint: the development sandbox had no GPU, no PyTorch and no weights.
Every number below is either arithmetic, a replay of routing you recorded earlier, or a
unit-test result, and is labelled as such. No benchmark result is claimed.

## 1. Audit of v0.6

| Finding | Where | Consequence |
|---|---|---|
| No saved run separates drive time from GPU time for v0.5.1/v0.6. The only per-layer profiles are v0.4 (USB 2.0 era). `wait_seconds` exists but is not reported per token. | `frustum_chat.py`, saved JSONs | The roofline cannot be closed from existing evidence. |
| The "8 s/token" figure is 192.2 s / 24 tokens and includes prefill. | README, whitepaper | Decode-only s/token for v0.5.1 is not recorded. |
| Byte counters are logical. Reads are buffered, so the Windows file cache can serve them. | `RangeReader` | "Cold" and "warm" were never distinguished. |
| Next-layer guesses were queued at the *start* of layer N, and nothing cancelled a guess once a reader thread picked it up. By the time layer N+1 routed, the four readers had worked through the queue. | `streamed_moe.forward`, `discard_pending` | Nearly every wrong guess was read in full, then dropped. |
| The guess uses layer N's MoE input on layer N+1's router. The real input to that router also contains layer N's MoE output and layer N+1's attention. Recall was never saved. | `_predict` | Guess quality is unknown and plausibly poor. |
| `get(key, (key,))` protected only the expert being loaded. Loading one expert of a layer could evict another expert of the same layer that had not run yet (GPU tier), or drop its only RAM copy (host tier). | `expert_cache.get`, `_remember` | A hit becomes a re-read. Rare under LFU (ties only), possible under LRU. |
| Each expert was six positioned reads (three weights, three scales). | `_read` | Six syscalls where one suffices if the tensors are adjacent on disk. |
| Host tier is sized once at start-up and never gives RAM back. | `placement.plan` | Can push the machine into paging if other programs grow. |
| Component tests need CUDA; `selftest.py` cache tests need PyTorch. Full-model parity is still open. | tests | Unchanged by this work. |

## 2. Roofline (arithmetic and replay, not measurement)

One routed expert is 18,800,640 bytes; 6 experts x 40 layers = 4.51 GB if nothing is cached.

| Bytes read per token | at 250 MB/s | at 500 MB/s | at 1 GB/s | at 3 GB/s |
|---|---|---|---|---|
| 4.51 GB (no reuse) | 18.0 s | 9.0 s | 4.5 s | 1.5 s |
| 2.26 GB (half reused) | 9.0 s | 4.5 s | 2.3 s | 0.8 s |
| 1.13 GB | 4.5 s | 2.3 s | 1.1 s | 0.4 s |

Replay of the only routing on file (`decode_profile_results.json`, two decode steps, v0.4,
"Say hello."; log in `v07_logs/replay_saved_v04_profile_sandbox.log`):

- 49.6% of the second token's experts were used by the first token.
- With any RAM tier of 8 GiB or more, the second token still reads 121 experts = 2.27 GB, and
  **all of it is first-time reads**. No cache size or policy changes that number.
- Layer protection makes no difference in this replay.

Reading: at about 500 MB/s, 2.27 GB is about 4.5 s, so storage plausibly accounts for half or
more of an 8 s token early in a conversation, and those bytes are ones a RAM cache cannot
remove. Two decode steps cannot show how fast first-time reads decay as a conversation goes
on. That is the measurement to take first (section 5).

### Why next-layer prediction lost (inference from arithmetic; recall was not recorded)

6 guesses x 40 layers x 18.8 MB = 4.5 GB per token. At 500 MB/s that is about 9 s. The
observed slowdown was 192 s -> about 420 s for 24 tokens, i.e. about +9.5 s per token. The
slowdown equals the cost of reading essentially every guess and using essentially none. That
fits the two audit findings above: guesses were not cancellable in time, and the predictor's
input is not the router's real input. It also means the drive had no idle time to absorb them.

## 3. Ranked plan

| # | Bottleneck | Mechanism | Expected benefit | Correctness risk | How to measure | Status |
|---|---|---|---|---|---|---|
| 1 | Unknown split of a token between drive and GPU | I/O ledger, physical device counters, `roofline.py`, `bench_e2e.py` | None by itself; decides everything below | None (no decision or arithmetic changes) | It is the measurement | Implemented |
| 2 | First-time expert reads over USB (2.27 GB on token 2 in the replay) | Spill tier: CRC-verified exact copies of experts on a faster internal drive, read before the checkpoint drive | Up to (USB time for those bytes) x (1 - USB speed / internal speed), if the roofline says storage-bound. Zero if compute-bound or the internal drive is not faster | Low: bytes are verified against identity + CRC-32; any mismatch falls back to the checkpoint | `bench_e2e.py --configs patch spill --spill-dir ...`: physical bytes on the model drive, s/token | Implemented, opt-in |
| 3 | Wrong guesses stealing drive time | Guesses issued after the layer's own reads; scored in shadow; read only while measured precision of not-yet-resident guesses >= 0.6; cancelled or counted as waste | Unknown until shadow precision is measured. Negative if precision is low, which the gate prevents | None: the real router still selects; guesses only move reads earlier | `--configs shadow gated`: `prediction.sources.*.cold_precision`, `speculative_wasted_bytes`, s/token | Implemented, off by default |
| 4 | Same-layer self-eviction | `begin_layer` protects the whole layer in both tiers | Small. Zero in the saved replay | None (eviction choice only) | `repeat_expert_bytes_per_token` baseline vs patch | Implemented, on |
| 5 | Six reads per expert | One coalesced read per contiguous expert | Small; fewer syscalls, same bytes | None (same bytes; tested) | `effective_MBps` baseline vs patch | Implemented, on |
| 6 | Host tier never yields RAM | Shrink-only guard below a free-RAM floor | Safety, not speed | None | `host_shrinks` in stats | Implemented, on (3 GiB) |
| 7 | Drive idle while a resident expert computes | Run resident experts first, accumulate in the original order | Small within a layer | Medium: must keep summation order bit-identical | `overlap_fraction` | Not built; only if the roofline shows idle drive time |
| 8 | Buffered reads double-cache in RAM | Unbuffered (`FILE_FLAG_NO_BUFFERING`) aligned reads | Unknown | Low | `physical_over_logical`, `effective_MBps` | Not built |
| 9 | Follow-up turns re-prefill when re-tokenised history differs | Build the next prompt from generated token ids | Fewer rebuilds | Medium: differs from the official encoder's tokenisation | `reused_prompt_tokens` | Not built |
| 10 | Fused kernels, FP4 on Tensor Cores | - | Only if the roofline says compute-bound | - | `run_fused_check.ps1`, `speed_check.py` | Untouched. No native FP4 Tensor Core path exists or is claimed |
| 11 | ActivatePrime streaming / cancel / progress | - | Usability | - | - | Not touched: its source is outside the folder shared for this session |

## 4. What changed (small patches, all switchable)

| File | Change | Switch back |
|---|---|---|
| `io_ledger.py` (new) | Thread-safe accounting: unique vs repeated bytes, drive-busy interval union, queue depth, blocked time, overlap, speculative outcomes | - |
| `os_cache.py` (new) | Physical device bytes (Windows `IOCTL_DISK_PERFORMANCE`, Linux sysfs); best-effort file-cache purge | - |
| `spill_tier.py` (new) | Optional verified local copies of experts | omit `--spill-dir` |
| `roofline.py`, `bench_e2e.py`, `run_bench.ps1` (new) | Roofline report; cold/warm, >= 3 runs, median/p95 | - |
| `selftest_v07.py`, `run_selftest_v07.ps1` (new) | Tests below | - |
| `frustum_loader.py` | `RangeReader.read_group` | `FRUSTUM_COALESCE=0` |
| `cache_policy.py` | `Tier.admit(protected=)`; `simulate(group=)`; unique/repeat bytes in replay | default arguments = old behaviour |
| `expert_cache.py` | `begin_layer`, coalesced reads, speculative bookkeeping, ledger, RAM guard, spill hook | `FRUSTUM_PROTECT_LAYER=0`, `FRUSTUM_COALESCE=0`, `--ram-floor-gib 0` |
| `streamed_moe.py` | `begin_layer` call; guesses issued after the layer; two scored sources; `predict_mode` | `--predict-next 0` (default); `--predict-mode always` = v0.6 policy |
| `frustum_chat.py` | `--predict-mode`, `--trace-routing`, `--spill-dir/--spill-gib`, `--ram-floor-gib`; per-step `io`, physical bytes, a second summary line | flags above; `--legacy` unchanged |
| `decode_cache_replay.py` | Accepts `--trace-routing` evidence; host policy and layer protection sweeps | - |

Not changed: model arithmetic, routing, weights, fused kernels, attention, prefix-reuse logic,
GLM/Phi backends (they use the same cache through the unchanged API), README, whitepaper,
version number. `v07_logs/v0.6_to_v0.7.patch` is the full diff against `ed3a77f`.

Hard constraints: a selected expert is always the one that runs (guesses only start reads
early; a miss still reads the selected expert). No weight is modified; the spill tier writes
copies elsewhere and never touches the checkpoint. The reference path (`--no-fused`,
`--legacy`, `validate=True`) is intact.

## 5. Tests and their status

`python selftest_v07.py` (or `.\run_selftest_v07.ps1`): 29 tests.

| Area | Tests | Sandbox result |
|---|---|---|
| I/O ledger: interval union, queue depth, overlap, unique/repeat, thread balance | 4 | PASS |
| Coalesced reads equal per-tensor reads; gaps; mmap mode | 3 | PASS |
| Cache eviction: layer protection (GPU and host tier), fallback when a tier is smaller than a layer, protected admission, replay accounting | 4 | PASS (cache driven through a NumPy stand-in for four torch calls) |
| Speculative reads: cancel / waste / adopt accounting | 2 | PASS (stand-in) |
| Async safety: 6 seeds x 250 random prefetch/discard/get rounds with 1 or 4 reader threads, every returned buffer byte-compared with the checkpoint; failed read surfaces and leaves the cache usable | 2 | PASS (stand-in) |
| RAM guard | 1 | PASS (stand-in) |
| Spill tier: exact bytes across sessions; flipped bit, truncation and changed checkpoint rejected; budget | 3 | PASS (stand-in, Linux) |
| Prefix reuse: 200 random multi-turn conversations give the same tokens with and without reuse; strict-prefix rule; interrupted step forces rebuild | 3 | PASS |
| Roofline verdicts | 2 | PASS |
| Routing vs float64 reference; FP4 exhaustive + fast-path bit equality; FP8 all finite codes with block scales; RMSNorm / mHC pre, post, mixes vs float64; prediction gate | 5 | **SKIPPED: need PyTorch. UNRUN.** |

Existing `selftest.py`: 30 tests, 9 errors, 1 skip both before and after the patch in the
sandbox; all 9 are `No module named 'torch'`. Its six `ExpertCacheTests` pass against the
patched cache with the stand-in. Logs are in `v07_logs/`.

**Unrun or unverified**

- Everything on the GPU: `streamed_moe.py` and `frustum_chat.py` changes were only byte-compiled.
- All Windows-specific code: `IOCTL_DISK_PERFORMANCE`, the cache purge, spill-file replace.
- The purge relies on observed Windows behaviour, not a documented guarantee. The benchmark
  flags `cold_not_achieved` instead of trusting it.
- Whether expert tensors are adjacent in the shards (coalescing falls back to separate reads if not).
- Any speed effect of any change.

## 6. First session on the real machine

    .\run_selftest.ps1 ; .\run_selftest_v07.ps1          # both should pass with PyTorch present
    .\run_verify.ps1                                      # layer-0 hash and "Say hello." token ids vs v0.4
    .\run_bench.ps1 --configs baseline patch shadow       # about 18 fresh processes; hours on DeepSeek
    python roofline.py bench_runs\<stamp>\patch_cold_0.json bench_runs\<stamp>\patch_warm_0.json

Then decide from the verdict line:

- Storage-bound, mostly first-time bytes: `.\run_bench.ps1 --configs patch spill --spill-dir D:\frustum_spill`
  (any internal SSD with room; 64 GiB default budget holds about 3,400 experts).
- Storage-bound, mostly re-reads: raise `--host-cache-gib`, and replay the trace:
  `python decode_cache_replay.py bench_runs\<stamp>\patch_cold_0.json --summary --host-policy lru lfu`.
- Idle drive time and `shadow` shows `cold_precision` of 0.6 or more for a source: add `gated`.
- Compute-bound: storage work is finished; profile kernels and uploads.

For a true v0.6 baseline alongside the flag baseline:

    git worktree add ..\Frustum_v06 ed3a77f
    .\run_bench.ps1 --baseline-dir ..\Frustum_v06

Token ids must be identical across configurations (`results.md`, last line). If they are not,
stop and keep the evidence: these patches are not supposed to be able to change a token.

## 7. Rollback

Per feature: the switches in section 4.
Everything (revert the committed v0.7 implementation rather than deleting files by hand):

    git revert 6e3f1ca

Or apply in reverse: `git apply -R v07_logs\v0.6_to_v0.7.patch`. The spill folder, if you
created one, is separate: delete it by hand.
