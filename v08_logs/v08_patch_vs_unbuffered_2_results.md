### cold OS cache (median / p95)

| metric | patch | unbuffered |
|---|---|---|
| ttft_seconds | 63 / 63 | 58.1 / 58.7 |
| seconds_per_token | 5.69 / 5.73 | 4.06 / 4.11 |
| physical_bytes | 106 / 107 GB | 79 / 79.4 GB |
| logical_bytes | 83.7 / 84.1 GB | 78.9 / 79.4 GB |
| physical_over_logical | 1.27 / 1.27 | 1 / 1 |
| unique_expert_bytes_per_token | 1.11 / 1.11 GB | 1.11 / 1.11 GB |
| repeat_expert_bytes_per_token | 1.29 / 1.32 GB | 0.968 / 0.997 GB |
| reread_amplification | 2.16 / 2.18 | 1.87 / 1.9 |
| gpu_hit_rate | 0.31 / 0.31 | 0.31 / 0.31 |
| host_hit_rate | 0.234 / 0.234 | 0.295 / 0.297 |
| spill_hit_rate | 0 / 0 | 0 / 0 |
| drive_hit_rate | 0.456 / 0.462 | 0.395 / 0.401 |
| effective_MBps | 580 / 588 | 863 / 866 |
| mean_queue_depth | 2.51 / 2.54 | 2.86 / 2.86 |
| drive_utilization | 0.731 / 0.732 | 0.592 / 0.595 |
| overlap_fraction | 0.23 / 0.231 | 0.229 / 0.23 |
| runs / flags | 3 - | 3 - |

### warm OS cache (median / p95)

| metric | patch | unbuffered |
|---|---|---|
| ttft_seconds | 62.1 / 62.2 | 57.7 / 59 |
| seconds_per_token | 5.66 / 5.73 | 4.12 / 4.19 |
| physical_bytes | 106 / 107 GB | 79.1 / 79.1 GB |
| logical_bytes | 84.3 / 84.3 GB | 79 / 79 GB |
| physical_over_logical | 1.27 / 1.27 | 1 / 1 |
| unique_expert_bytes_per_token | 1.11 / 1.11 GB | 1.11 / 1.11 GB |
| repeat_expert_bytes_per_token | 1.33 / 1.33 GB | 0.974 / 0.976 GB |
| reread_amplification | 2.19 / 2.19 | 1.88 / 1.88 |
| gpu_hit_rate | 0.31 / 0.31 | 0.31 / 0.31 |
| host_hit_rate | 0.226 / 0.233 | 0.293 / 0.297 |
| spill_hit_rate | 0 / 0 | 0 / 0 |
| drive_hit_rate | 0.463 / 0.463 | 0.396 / 0.397 |
| effective_MBps | 584 / 592 | 861 / 864 |
| mean_queue_depth | 2.52 / 2.54 | 2.85 / 2.86 |
| drive_utilization | 0.727 / 0.728 | 0.583 / 0.591 |
| overlap_fraction | 0.224 / 0.231 | 0.229 / 0.234 |
| runs / flags | 3 - | 3 - |

Token ids: {"identical_within_each_config": true, "identical_across_configs": true}
