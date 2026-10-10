# Causal Discovery Edge-IIoT Optimization

## Phase 1: temporal graph construction

Phase 1 audits the verified Edge-IIoTset schema, copies the source CSV into `data/raw/` without modifying it, cleans records, creates a deterministic temporally stratified development sample, and writes validated directed PyTorch Geometric snapshots.

## Phase 2: causal discovery and SCM

Phase 2 runs separately from Phase 1 and reuses the existing graph snapshots in `data/processed/train/`, `data/processed/validation/`, and `data/processed/test/`. It builds graph-level causal variables, runs NOTEARS on training-period observations only, applies configured DAG constraints and stability filtering, then fits a reusable linear SCM.

```powershell
python -m src.phase2_pipeline --config config/phase2_config.json
```

Main outputs are written to `data/interim/causal_dataset.csv`, `artifacts/phase2/`, and `reports/phase2/`. The test split is loaded only into the causal dataset for future phases; it is not used for DAG discovery, refinement, stability filtering, or SCM fitting.

## Phase 3: edge-aware GNN embeddings

Phase 3 runs independently from the earlier pipelines. It joins the existing PyG snapshots with the Phase 2 causal table, trains a bidirectional edge-aware GATv2 encoder using current- and next-window causal targets, and exports one 64-dimensional embedding per graph.

```powershell
python -m src.phase3_pipeline --config config/phase3_config.json
```

Use `--validate-only` to check all graph/causal joins and tensor schemas without training or writing outputs. Model checkpoints and embeddings are written to `artifacts/phase3/`; metrics and training reports are written to `reports/phase3/`. The exported `encoder.pt` is frozen by default for use as the Phase 4 state encoder.

## Phase 4: simulated network-control DQN

Phase 4 reuses the frozen 64-dimensional Phase 3 embeddings and trains a DQN to select among six routing profiles in a reproducible three-server Gymnasium environment. The observation also includes three normalized server queues. Latency and energy are transparent simulation proxies, not measurements from Edge-IIoTset.

```powershell
python -m src.phase4_pipeline --config config/phase4_config.json --validate-only
python -m src.phase4_pipeline --config config/phase4_config.json --smoke-test
python -m src.phase4_pipeline --config config/phase4_config.json
```

The best and final agents are written to `artifacts/phase4/`, evaluation and TensorBoard reports to `reports/phase4/`, and the balanced six-action dataset for Phase 5 to `data/interim/phase4_action_outcomes.csv`.

## Phase 5: counterfactual routing safety

Phase 5 independently reuses the saved Phase 4 DQN, calibration, and six-action outcome dataset, plus Phase 3 embeddings and Phase 2 causal rows. An exact action-aware SCM screens all routing profiles in DQN Q-value order. A training-only learned surrogate provides separate approximation diagnostics.

```powershell
python -m src.phase5_pipeline --config config/phase5_config.json --validate-only
python -m src.phase5_pipeline --config config/phase5_config.json --smoke-test
python -m src.phase5_pipeline --config config/phase5_config.json
python -m src.phase5_pipeline --config config/phase5_config.json --audit-only
```

Artifacts go to `artifacts/phase5/`; evaluations, six-action explanations, and audits go to `reports/phase5/`. Outputs are protected from overwrite by default. Limits are latency proxy <=15,000 ms, energy proxy <=80% of calibrated maximum (currently 5.28), drop ratio <=1%, and PDR >=0.99. If no action meets them, the controller executes the minimum-violation profile and explicitly records infeasibility. Screening uses actual demand and queues in addition to the DQN embedding; it checks one-step simulator outcomes, with synthetic latency and energy, and does not establish real-network or long-term safety. Reports distinguish episode-average from step-weighted metrics and flag unsupported surrogate states or uninformative prediction bands.

## Quick start

### Method 1: Standard project environment (original method)

This is the usual workflow for a clean machine and is kept as the default fallback:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.main --config config/phase1_config.json
```

### Method 2: User-level virtual environment (working workaround)

On some Windows machines, the project-local `.venv` can run into Windows Application Control / DLL blocking issues when importing PyTorch, even though the package installs successfully. A reliable workaround is to create the virtual environment in the user-level AppData folder instead, then install the same dependencies and run the project from that interpreter:

```powershell
& "$env:LOCALAPPDATA\Python\pythoncore-3.10-64\python.exe" -m venv "$env:LOCALAPPDATA\venvs\edge-iiot"
& "$env:LOCALAPPDATA\venvs\edge-iiot\Scripts\python.exe" -m pip install --upgrade pip
& "$env:LOCALAPPDATA\venvs\edge-iiot\Scripts\python.exe" -m pip install -r requirements.txt
& "$env:LOCALAPPDATA\venvs\edge-iiot\Scripts\python.exe" -m src.main --config config/phase1_config.json
& "$env:LOCALAPPDATA\venvs\edge-iiot\Scripts\python.exe" -m src.phase2_pipeline --config config/phase2_config.json
& "$env:LOCALAPPDATA\Python\pythoncore-3.10-64\python.exe" -m src.phase3_pipeline --config config/phase3_config.json
```

If you are already inside the project folder and want to activate that environment manually:

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& "$env:LOCALAPPDATA\venvs\edge-iiot\Scripts\Activate.ps1")
python -m src.main --config config/phase1_config.json
```

> The user-level venv method is the one that was verified to work in this environment. The project-local venv remains the fallback if the user-level environment is unavailable or unsupported on another machine.

## Notes

The default `sample_size` is 200,000 for development. Set it to `null` in the JSON configuration to process every cleaned row, or adjust `window_seconds`, sample size, and split fractions there. Outputs are written under `data/`, `artifacts/`, and `reports/`. The pipeline refuses to overwrite non-empty graph output directories unless `overwrite_outputs` is set to `true`.
