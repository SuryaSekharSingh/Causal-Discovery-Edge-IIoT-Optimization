# Project Review Guide: Phases 1-6

This guide explains what the repository actually implements, how data moves between phases, and how to read the final results. The **timing-corrected** Phase 4/5 runs are `phase4_v3` and `phase5_v3`; the final evaluation is `phase6_v3`. Earlier Phase 4/5 runs used an invalid same-window decision observation and are retained only as legacy artifacts.

## One-minute overview

The project turns sampled Edge-IIoTset traffic into directed, 60-second graphs; derives graph-level variables; learns a graph embedding; trains a routing agent in a **synthetic** three-server simulator; screens actions with an exact model of that simulator; and compares screened and unscreened agents with simpler state representations. The model does **not** control a real network, and its latency and energy values are simulation proxies.

```text
Traffic CSV
  | Phase 1: clean, sample, split, make 60-second directed PyG graphs
  v
data/processed/{train,validation,test}/graph_*.pt
  | Phase 2: graph-level variables, NOTEARS candidate DAG, linear SCM
  +------------------------> data/interim/causal_dataset.csv
  | Phase 3: edge-aware bidirectional GATv2; current/next auxiliary targets
  +------------------------> artifacts/phase3/embeddings/{split}.pt [N, 64]
                               |
Decision for window t:        | use completed graph t-1 only
  embedding(t-1) + announced demand(t) + queues(before t) [68]
                               | Phase 4: DQN proposes one of six routes
                               | Phase 5: exact simulator SCM screens six routes
                               v
                         execute once for window t
                               |
                               v
                    Phase 6: paired evaluation and ablations
```

The Phase 2 *observational* DAG is **not** the action-intervention model in Phase 5. Phase 5 has a separate, explicit structural model of the Phase 4 simulator. These two uses of the word "causal" must not be conflated.

## Data and decision contract

| Item | Contract |
|---|---|
| Splits | Chronological train, validation, and test; 542, 113, and 146 graphs (801 total). |
| Graph tensor | `x` has 7 node features; `edge_attr` has 66 edge features; `edge_index` is directed; metadata includes graph identity and `window_start`. |
| Phase 2 row | Exactly one `(split, graph_file, window_start)` row per graph; nine retained variables in the order in `artifacts/phase2/refined_dag.json`. |
| Phase 3 embedding | One finite 64-value vector per graph, tied to the encoder hash, filename, split, and timestamp. |
| Legal Phase 4 decision | Completed observation graph at `t-1` and demand graph at `t` must be exactly 60 seconds apart **within the same split**. Counts are 537 train, 112 validation, and 143 test decisions. |
| Policy information at `t` | 64 values from graph `t-1`, one training-normalized `log1p(announced demand_t)`, and three pre-action queue ratios. No embedding or other derived variables from completed graph `t`. |
| Phase 5 information at `t` | The same announced demand, actual queues, and saved simulator calibration. It uses no completed topology from `t`. |
| Evaluation | Validation selects checkpoints; test is loaded only for the frozen final evaluation. Surges alter announced demand, not historical embeddings or service capacities. |

The announcement of the current packet batch is a **simulator assumption**. It is not evidence that a deployed controller can forecast or observe its entire future minute of traffic. Adjacent split periods also retain two partially shared window starts; see [boundary_audit.json](reports/phase6_v3/boundary_audit.json). They are not a fully window-isolated test.

## Phase 1: traffic to graph snapshots

**Purpose.** Turn packet-level records into topology-bearing snapshots. [phase1_pipeline.py](src/phase1_pipeline.py) audits and cleans the source CSV, takes a deterministic 200,000-record development sample, partitions time into train/validation/test, and aggregates traffic into 60-second directed graphs. Each endpoint is a node; directed source-to-destination traffic becomes an edge. The seven node features summarize inbound/outbound packets and bytes, packet size, interarrival, and peer count. Edge attributes combine six edge aggregates with transformed record features (66 total). Node and edge scalers are fitted on training graphs only.

**Crucial implementation:**

```python
# src/phase1_pipeline.py, simplified from build_raw_graphs
frame["_window"] = ((frame["frame.time"] - start).dt.total_seconds() // window_seconds).astype(int)
for (split, window), group in frame.groupby(["_split", "_window"], sort=True):
    graph = nx.DiGraph(window=int(window), split=str(split))
    # Aggregate nodes and directed (source, destination) edges in this group.
```

Grouping by split prevents a single graph file from containing records from both sides of a split. It does **not** remove the two boundary minutes that share the same nominal 60-second start across adjacent splits. Fixing that would require rebuilding Phase 1 and downstream artifacts.

**Output and handoff.** `data/processed/{train,validation,test}/graph_*.pt`, Phase 1 scalers and feature mapping, and graph statistics. Phase 2 reads these graph files; Phase 3 also reads their tensors directly. The `.pt` graph files are trusted local PyG objects, not arbitrary remote downloads.

## Phase 2: graph-level variables and observational discovery

**Purpose.** Convert each graph to interpretable window-level statistics, then discover a candidate observational dependency structure. [causal_dataset.py](src/causal_dataset.py) inverses the training-fitted graph scaling to calculate quantities such as packet count, size, interarrival, active nodes, and degree. [phase2_pipeline.py](src/phase2_pipeline.py) selects nine retained variables, runs linear NOTEARS on **training** rows, applies stability and domain constraints, validates the refined DAG, and fits an interpretable linear SCM.

**Crucial implementation:**

```python
# src/phase2_pipeline.py, abridged from run
candidate_adjacency, candidate_dag = discover_candidate_dag(train_frame, features, config)
constrained_adjacency, refinement_log = apply_constraints(
    candidate_adjacency, features, config["constraints"]
)
refined_dag = adjacency_to_dag(constrained_adjacency, features)
validate_dag(refined_dag, features)
scm = LinearSCM(refined_dag, features).fit(train_frame)
```

The retained order is `total_packets`, `total_bytes_proxy`, `mean_packet_size_proxy`, `std_packet_size_proxy`, `mean_interarrival_time`, `std_interarrival_time`, `active_nodes`, `average_degree`, `average_peer_count`. These are graph summaries, not measured routing-action outcomes. A discovered edge is not proof of a real-world intervention effect.

**Output and handoff.** `data/interim/causal_dataset.csv` has 801 keyed rows. `artifacts/phase2/refined_dag.json` fixes the nine-variable order, and `artifacts/phase2/scm.pkl` holds the observational linear SCM. Phase 3 uses the rows as auxiliary labels; Phase 4 uses `total_packets` as simulated demand. The Phase 5 action screen does **not** query this Phase 2 SCM.

## Phase 3: directed graph representation learning

**Purpose.** Compress each graph into a 64-dimensional network-state representation. [gnn_model.py](src/gnn_model.py) projects seven node features to 64 hidden features, passes messages through two four-head GATv2 blocks using all 66 edge attributes, and separately processes original and reversed edges. Mean and max graph pooling, plus normalized log node/edge counts, feed a graph-level readout. Two heads train against current-window and exact-next-window nine-variable targets; only the encoder is exported for later inference.

**Crucial implementation:**

```python
# src/gnn_model.py, from BidirectionalGATBlock.forward
incoming = self.incoming(x, edge_index, edge_attr)
outgoing = self.outgoing(x, edge_index.flip(0), edge_attr)
messages = self.fuse(torch.cat([incoming, outgoing], dim=-1))
return self.norm(x + self.dropout(F.elu(messages)))
```

Training minimizes standardized current-state MSE plus `0.5` times next-state MSE. The next loss is masked when no exact 60-second successor exists. Train-only target and graph-size normalization is saved. Validation loss chooses the checkpoint; future targets and prediction heads are **not** stored in embedding files.

**Output and handoff.** `artifacts/phase3/encoder.pt` and `embeddings/{train,validation,test}.pt`; `reports/phase3/evaluation_metrics.json` and diagnostics. Phase 4 verifies encoder hashes, shapes, graph identities, and timestamps before using embeddings. Phase 3 test current-state standardized MSE is about `1.632` versus training-mean baseline `4.259`; next-state MSE is about `2.052` versus persistence `1.400`. Thus the encoder reconstructs current variables better than a constant mean, but **next-state prediction does not beat persistence on test**. Neither result establishes that its embeddings are best for RL.

## Phase 4: routing simulator and DQN

**Purpose.** Learn a routing action in a controlled, reproducible simulator. Use the **v3** configuration and outputs: [phase4_v3_config.json](config/phase4_v3_config.json), [rl_environment.py](src/rl_environment.py), [phase4_pipeline.py](src/phase4_pipeline.py). The three simulated servers have heterogeneous training-calibrated service capacities and buffers. Six discrete actions route all packets to A/B/C or split them evenly, performance-weighted, or energy-weighted. The 68-value observation is `embedding(t-1) + announced demand(t) + queues(before t)`.

**Crucial implementation:**

```python
# src/rl_environment.py, abridged from simulate_action
served_old = np.minimum(pre_queue, calibration.service_capacity)
remaining_capacity = calibration.service_capacity - served_old
arrivals = fractions * demand
served_current = np.minimum(arrivals, remaining_capacity)
enqueued = np.minimum(arrivals - served_current, available_buffer)
dropped = np.maximum(arrivals - served_current - enqueued, 0.0)
```

Old queued packets consume service first. `pdr` is the fraction of **current** arrivals served immediately; it is not eventual packet delivery. Throughput includes old and current packets served. `latency_proxy_ms` uses base latency, **pre-action** queue waiting time, and an undelivered-current-arrivals penalty. `energy_proxy` uses idle and utilization-dependent terms. Task completion requires `pdr >= 0.99` and latency within 15,000 ms. Reward combines normalized throughput (`+0.30`), immediate service (`+0.25`), completion (`+0.15`), latency (`-0.15`), energy (`-0.10`), and drops (`-0.05`), then clips to `[-1,1]`.

Stable Baselines3 DQN trains for 100,000 steps with validation checkpoint selection. The timing-v2 environment truncates at 64 actions or the end of a contiguous segment and treats placeholder terminal observations as learning terminals in replay. Calibration is fitted only on train demand. A balanced action-outcome table evaluates all six actions from each sampled pre-action state without letting the query advance the environment.

**Output and handoff.** `artifacts/phase4_v3/best_model.zip`, `environment_calibration.json`, `model_metadata.json`, and `data/interim/phase4_v3_action_outcomes.csv`. Phase 5 reads these with hash and schema checks. The original `phase4` and superseded `phase4_v2` outputs must not be mixed with v3.

## Phase 5: exact one-step counterfactual screening

**Purpose.** Intercept a frozen DQN proposal and ask what the **Phase 4 simulator** would do under each action from the same demand and queue state. [counterfactual_controller.py](src/counterfactual_controller.py) independently implements the queue and outcome equations as an exact action-aware structural model. [phase5_pipeline.py](src/phase5_pipeline.py) validates it against saved outcomes and realized simulator steps. A separate ExtraTrees surrogate is trained on training action-outcome rows for diagnostic approximation/coverage reports; it does **not** select actions.

**Crucial implementation:**

```python
# src/counterfactual_controller.py, abridged from select_action
ranking = tuple(sorted(range(6), key=lambda a: (-q_values[a], a)))
outcomes = self.scm.evaluate_actions(state)  # Same immutable state, six interventions
assessments = tuple(self.assess(outcome) for outcome in outcomes)
safe = [a for a in ranking if not assessments[a][0]]
selected = safe[0] if safe else min(
    range(6), key=lambda a: (assessments[a][1], -q_values[a], a)
)
```

An action is considered safe when `latency_proxy_ms <= 15000`, `energy_proxy <= 0.8 * 6.60 = 5.28`, `drop_ratio <= 0.01`, and `pdr >= 0.99`, with `1e-8` comparison tolerance. If the DQN's top action is safe, it remains unchanged. Otherwise the highest-Q safe alternative is selected. If **none** is safe, the controller chooses the smallest normalized total violation and labels `no_safe_action`: this is a fallback, **not a safe action**. Q-values rank actions; they are not probabilities.

**Output and handoff.** `artifacts/phase5_v3/` contains the exact SCM specification, surrogate, and frozen thresholds. `reports/phase5_v3/` contains audits, surrogate diagnostics, policy comparisons, and six-action explanations. Phase 6 reuses the frozen exact screen and verifies predicted and realized outcomes. This is a **one-step simulator guarantee where a feasible action exists**, not a long-horizon or real-network safety guarantee.

## Phase 6: final comparative evaluation

**Purpose.** Test what each component contributes, using the same six actions, simulator equations, training budget, validation selection, and frozen test schedule. [phase6_pipeline.py](src/phase6_pipeline.py) compares three state representations:

1. `gnn`: previous completed graph's 64-value embedding + announced demand + queues (68 inputs).
2. `causal`: previous completed graph's **nine Phase 2 variables** + announced demand + queues (13 inputs). This label means causal-variable features, **not** the Phase 2 SCM controlling actions.
3. `minimal`: announced demand + queues only (4 inputs).

Each DQN is evaluated both normally and with the same Phase 5 screen. Random and least-normalized-queue heuristics provide extra context. Seeds are `11, 23, 42, 67, 101`; the corrected Phase 4 GNN seed-42 agent is reused, while the other matched agents are trained for 100,000 steps each. Validation chooses the best checkpoint for each seed; the final test uses non-overlapping episodes covering each eligible decision window once per policy. Policies start from the same initial queues and demand stream, then evolve their own queues.

**Crucial implementation:**

```python
# src/phase6_pipeline.py, from decision_blocks
for index in env.valid_starts:
    if index != previous + 1 or span >= env.episode_horizon:
        starts.append(index)
        span = 0
    span += 1
    previous = index
```

The main test is normal traffic. Two additional scenarios multiply announced demand by `1.5` or `2.0` from decision 8 for up to 16 actions where a segment is long enough. They leave the prior graph embedding and server calibration unchanged, so they test **exogenous workload robustness**, not attack or topology detection. Paired bootstrap intervals resample the three non-overlapping test episodes per seed; they are **descriptive**, not strong population confidence intervals.

**Outputs.** Final agent ZIPs and causal-feature normalization live in `artifacts/phase6_v3/`. Numeric results, shift trajectories, explanations, plots, and hashes live in `reports/phase6_v3/`. `review_report.md` is the quick narrative; `evaluation_steps.csv`, `evaluation_episodes.csv`, `policy_metrics.json`, and `explanations.jsonl` are the detailed evidence.

## How to read the final results

Start with [review_report.md](reports/phase6_v3/review_report.md), then check [policy_metrics.json](reports/phase6_v3/policy_metrics.json) and [evaluation_steps.csv](reports/phase6_v3/evaluation_steps.csv). The table below is the **normal test** result averaged over the same 143 legal decisions for each of five training seeds (`715` step records per row). Higher reward and immediate service are better; lower latency/energy proxies are better. These are **simulator** values.

| Representation | Policy | Reward/action | Immediate service | Latency proxy ms | Energy proxy |
|---|---|---:|---:|---:|---:|
| Nine previous-window variables | DQN | 0.3894 | 0.9467 | 5010.2 | 1.542 |
| Nine previous-window variables | Screened DQN | 0.4081 | 0.9646 | 3423.7 | 1.569 |
| GNN embedding | DQN | 0.3837 | 0.9438 | 5852.8 | 1.752 |
| GNN embedding | Screened DQN | 0.4049 | 0.9646 | 3437.4 | 1.773 |
| Demand + queues | DQN | 0.4029 | 0.9655 | 2999.9 | 1.554 |
| Demand + queues | Screened DQN | 0.4077 | 0.9646 | 3384.1 | 1.626 |
| Random heuristic | Unscreened | 0.3730 | 0.9356 | 5817.6 | 1.955 |
| Least-queue heuristic | Unscreened | 0.3870 | 0.9445 | 1670.2 | 1.602 |

**What this establishes.** On this frozen synthetic test stream, screening raises mean reward for all three DQN representations and prevents the executed action from violating a safety limit whenever **some** action is safe. The GNN DQN beats the random heuristic on reward, but it does **not** outperform the simpler demand-plus-queues DQN. The GNN embedding therefore has **no demonstrated advantage** in this Phase 6 comparison. Screening can change reward, latency, energy, and queue trajectories; discuss the complete trade-off, not just one metric.

**Safety denominator matters.** Across all three screened representations and five seeds, the normal test has `2,145` screened actions. `135` are `no_safe_action` fallbacks (`45/715`, about `6.3%`, for each representation); the selected action can violate a limit there because no candidate passes. In the other `2,010` feasible decision states, selected-action violations are **zero**. Thus "zero violations" without the feasible-state qualifier would be false. In [explanations.jsonl](reports/phase6_v3/explanations.jsonl), compare `ranking`, `reasons`, all six `outcomes`, `executed_action`, `status`, and `realized` for a concrete example. Predictions agree with the executed simulator outcomes to the audit tolerance; they do not validate real-world physics.

**Other metrics.** `reward/action` is step-weighted; `mean_episode_reward` and `cumulative_reward` are per-episode quantities, so do not compare them as if episodes had equal length. `throughput_packets` includes service of previous queues; `pdr` is immediate service of *this* batch. `drop_ratio` is overflow divided by current demand, so low drops can coexist with a large queue and poor immediate service. `task_completion` is the fraction of windows passing both the `0.99` immediate-service requirement and 15,000-ms latency budget. `latency_proxy_ms` and `energy_proxy` have no physical measurement validation. `feasible_fraction` is the share of states in which at least one of six actions passes all limits; `no_safe_action_fraction` is the share forced to fallback. Inspect action distributions and queue trajectories to understand *why* rewards differ.

**Distribution shift.** Use [shift_summary.csv](reports/phase6_v3/shift_summary.csv) for post-onset reward, immediate service, drops, queues, latency, and energy; use [shift_trajectories.csv](reports/phase6_v3/shift_trajectories.csv) for decision-by-decision paths and recovery/censoring. First separate episodes too short to reach the onset. A screened policy's improvement under a synthetic surge shows the exact simulator screen reacts to higher **announced** demand. It does not prove anomaly detection, adaptive retraining, or recovery from a real link failure.

**Uncertainty.** [paired_intervals.json](reports/phase6_v3/paired_intervals.json) compares screened minus unscreened cumulative episode reward for matching starts and seeds. For GNN seed 42 on the normal test, the paired mean is about `+1.68` with descriptive interval `[0, 3.49]` over only **three** episodes. Report all five seeds and do not interpret this as independent real-network replications or a statistically definitive superiority claim.

## Verification and known limits

The saved Phase 6 full run completed with `ALL_PHASE6_CHECKS_PASSED`. A subsequent `--audit-only` run, Phase 5 v3 artifact audit, Phase 6 `--validate-only` input check, and all **32** repository tests also passed. An independent read of `evaluation_steps.csv` and `explanations.jsonl` found finite numeric values, all six counterfactual actions per explanation, 143 normal-test decisions per seed/policy, and zero screened feasible-state violations. The saved comparison PNG also renders. These checks establish that the implemented pipeline and saved artifacts are internally consistent; they cannot certify the simulator's realism or guarantee that no untested software defect exists.

Known limitations for the project review:

- Two Phase 1 split boundaries share partial 60-second window starts (`2021-01-01T19:44:11` and `2021-01-01T21:36:11`); this was intentionally left unrepaired. Do not describe the splits as fully window-isolated.
- Phase 2 causal discovery uses observational graph summaries. Its DAG does not establish the effect of routing interventions.
- Phase 3 test next-state forecasting is worse than a simple persistence baseline; the GNN representation does not win the Phase 6 RL ablation.
- The announced packet batch, server capacities, reward, latency, energy, and safety thresholds are model assumptions. No real-network throughput, energy saving, delay reduction, or deployment safety follows directly.
- The exact Phase 5 screen gives **one-step simulator** consistency, not long-horizon guarantees. Some states have no safe action under the chosen thresholds.
- The test stream is small (143 eligible decision windows, only three non-overlapping episodes per seed), and all training seeds share it. Descriptive bootstrap intervals are not a substitute for independent datasets.

## Commands for an existing run

Run from the repository root with the Python interpreter that has the requirements installed. These commands **inspect** existing artifacts and do not retrain Phases 1-6:

```powershell
python -m src.phase5_pipeline --config config/phase5_v3_config.json --audit-only
python -m src.phase6_pipeline --config config/phase6_config.json --validate-only
python -m src.phase6_pipeline --config config/phase6_config.json --audit-only
python -m unittest discover -s tests -q
```

For a presentation, show the flow diagram, the corrected `t-1 -> t` decision contract, one six-action explanation, the normal-test comparison table, one synthetic-surge trajectory, and the explicit limitations. The defensible conclusion is: **the exact screen improves one-step simulator constraint compliance when feasible; the current GNN is functional but is not the strongest tested state representation for this simulated routing task.**
