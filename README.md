# Causal Discovery Edge-IIoT Optimization

## Phase 1: temporal graph construction

Only Phase 1 is implemented. It audits the verified Edge-IIoTset schema, copies the source CSV into `data/raw/` without modifying it, cleans records, creates a deterministic temporally stratified development sample, and writes validated directed PyTorch Geometric snapshots.

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
```

If you are already inside the project folder and want to activate that environment manually:

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& "$env:LOCALAPPDATA\venvs\edge-iiot\Scripts\Activate.ps1")
python -m src.main --config config/phase1_config.json
```

> The user-level venv method is the one that was verified to work in this environment. The project-local venv remains the fallback if the user-level environment is unavailable or unsupported on another machine.

## Notes

The default `sample_size` is 200,000 for development. Set it to `null` in the JSON configuration to process every cleaned row, or adjust `window_seconds`, sample size, and split fractions there. Outputs are written under `data/`, `artifacts/`, and `reports/`. The pipeline refuses to overwrite non-empty graph output directories unless `overwrite_outputs` is set to `true`.
