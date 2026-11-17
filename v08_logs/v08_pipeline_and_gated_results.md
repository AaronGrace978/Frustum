### warm OS cache (median / p95)

| metric | patch | sync_upload | gated |
|---|---|---|---|
| ttft_seconds | 58.5 / 58.6 | 57.8 / 57.9 | 58.5 / 58.7 |
| seconds_per_token | 4.33 / 4.34 | 4.07 / 4.08 | 4.42 / 4.42 |
| physical_bytes | 79.1 / 79.1 GB | 79.1 / 79.1 GB | 84.6 / 84.6 GB |
| logical_bytes | 79 / 79 GB | 79 / 79 GB | 84.5 / 84.5 GB |
| physical_over_logical | 1 / 1 | 1 / 1 | 1 / 1 |
| unique_expert_bytes_per_token | 1.11 / 1.11 GB | 1.11 / 1.11 GB | 1.11 / 1.11 GB |
| repeat_expert_bytes_per_token | 0.974 / 0.974 GB | 0.973 / 0.974 GB | 0.974 / 0.974 GB |
| reread_amplification | 1.88 / 1.88 | 1.87 / 1.87 | 1.88 / 1.88 |
| gpu_hit_rate | 0.31 / 0.31 | 0.31 / 0.31 | 0.31 / 0.31 |
| host_hit_rate | 0.293 / 0.294 | 0.294 / 0.294 | 0.293 / 0.293 |
| spill_hit_rate | 0 / 0 | 0 / 0 | 0 / 0 |
| drive_hit_rate | 0.396 / 0.396 | 0.396 / 0.396 | 0.396 / 0.396 |
| effective_MBps | 823 / 827 | 864 / 867 | 836 / 839 |
| mean_queue_depth | 2.82 / 2.82 | 2.86 / 2.86 | 2.87 / 2.88 |
| drive_utilization | 0.587 / 0.588 | 0.594 / 0.596 | 0.664 / 0.666 |
| overlap_fraction | 0.292 / 0.292 | 0.237 / 0.241 | 0.467 / 0.472 |
| runs / flags | 3 - | 3 - | 3 - |

Token ids: {"identical_within_each_config": true, "identical_across_configs": true}
