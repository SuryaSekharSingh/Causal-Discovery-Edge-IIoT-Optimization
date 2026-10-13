# Phase 6 Review

## Observed simulator results

| Representation | Policy | Mean reward/action | Immediate service | Latency proxy (ms) | Energy proxy |
|---|---|---:|---:|---:|---:|
| causal | dqn | 0.3894 | 0.9467 | 5010.2 | 1.542 |
| causal | screened_dqn | 0.4081 | 0.9646 | 3423.7 | 1.569 |
| gnn | dqn | 0.3837 | 0.9438 | 5852.8 | 1.752 |
| gnn | least_queue | 0.3870 | 0.9445 | 1670.2 | 1.602 |
| gnn | random | 0.3730 | 0.9356 | 5817.6 | 1.955 |
| gnn | screened_dqn | 0.4049 | 0.9646 | 3437.4 | 1.773 |
| minimal | dqn | 0.4029 | 0.9655 | 2999.9 | 1.554 |
| minimal | screened_dqn | 0.4077 | 0.9646 | 3384.1 | 1.626 |

Values aggregate all configured training seeds on one shared test stream.
See paired_intervals.json for descriptive paired intervals and shift_summary.csv for post-onset surges.

## Simulator assumptions
The current packet batch is announced before routing. Historical graph embeddings are not modified during synthetic demand surges.
Latency and energy are simulator proxies. PDR means immediate service of current arrivals.

## Unsupported claims
No real-network safety, measured latency or energy gain, topology attack detection, or Phase 2 DAG action effect is established.
The two Phase 1 split boundaries remain inside shared graph windows.
