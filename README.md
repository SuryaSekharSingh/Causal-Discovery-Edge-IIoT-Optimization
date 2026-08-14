# Causal Discovery Edge-IIoT Optimization

## Phase 1: temporal graph construction

Only Phase 1 is implemented. It audits the verified Edge-IIoTset schema, copies the source CSV into `data/raw/` without modifying it, cleans records, creates a deterministic temporally stratified development sample, and writes validated directed PyTorch Geometric snapshots.

Install the dependencies and run:

```powershell
py -m pip install -r requirements.txt
py -m src.main --config config/phase1_config.json
```

The default `sample_size` is 200,000 for development. Set it to `null` in the JSON configuration to process every cleaned row, or adjust `window_seconds`, sample size, and split fractions there. Outputs are written under `data/`, `artifacts/`, and `reports/`. The pipeline refuses to overwrite non-empty graph output directories unless `overwrite_outputs` is set to `true`.
