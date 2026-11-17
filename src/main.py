"""Run the complete Edge-IIoT workflow from Phase 1 through Phase 6."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Callable

from src import phase1_pipeline, phase2_pipeline, phase3_pipeline
from src import phase4_pipeline, phase5_pipeline, phase6_pipeline

LOG = logging.getLogger("pipeline")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIGS = {
    1: "config/phase1_config.json",
    2: "config/phase2_config.json",
    3: "config/phase3_config.json",
    4: "config/phase4_v3_config.json",
    5: "config/phase5_v3_config.json",
    6: "config/phase6_config.json",
}


def _read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a JSON object: {path}")
    return value


def _project_path(config: dict, key: str, config_path: Path) -> Path:
    if key not in config:
        raise ValueError(f"Missing '{key}' in {config_path}")
    path = Path(config[key])
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Pipeline paths must stay inside the project: {key}={path}")
    return resolved


def _outputs(configs: dict[int, tuple[Path, dict]]) -> list[Path]:
    """Return exact output files/directories owned by this six-phase chain."""
    p1_path, p1 = configs[1]
    p2_path, p2 = configs[2]
    p3_path, p3 = configs[3]
    p4_path, p4 = configs[4]
    p5_path, p5 = configs[5]
    p6_path, p6 = configs[6]

    targets = [
        _project_path(p1, key, p1_path)
        for key in ("raw_csv", "interim_csv", "processed_dir")
    ]
    p1_artifacts = _project_path(p1, "artifacts_dir", p1_path)
    p1_reports = _project_path(p1, "reports_dir", p1_path)
    targets.extend(p1_artifacts / name for name in (
        "feature_mapping.json", "node_mapping.json", "scaler.pkl",
        "preprocessing_metadata.json", "dataset_audit.json",
    ))
    targets.append(p1_reports / "graph_statistics.csv")

    targets.extend([
        _project_path(p2, "causal_dataset_csv", p2_path),
        _project_path(p2, "phase2_artifacts_dir", p2_path),
        _project_path(p2, "phase2_reports_dir", p2_path),
        _project_path(p3, "phase3_artifacts_dir", p3_path),
        _project_path(p3, "phase3_reports_dir", p3_path),
        _project_path(p4, "transition_dataset_csv", p4_path),
        _project_path(p4, "phase4_artifacts_dir", p4_path),
        _project_path(p4, "phase4_reports_dir", p4_path),
        _project_path(p5, "artifacts_dir", p5_path),
        _project_path(p5, "reports_dir", p5_path),
        _project_path(p6, "artifacts_dir", p6_path),
        _project_path(p6, "reports_dir", p6_path),
    ])

    # Refuse ambiguous nested targets rather than risk deleting a parent that
    # contains another phase's output or a caller-supplied input.
    unique = sorted(set(targets), key=lambda item: (len(item.parts), str(item)))
    for index, target in enumerate(unique):
        for other in unique[index + 1:]:
            if other.is_relative_to(target):
                raise ValueError(f"Overlapping output paths are not supported: {target} and {other}")
    return unique


def _check_config_chain(configs: dict[int, tuple[Path, dict]]) -> None:
    def same_path(phase: int, key: str, referenced: str, label: str) -> None:
        config_path, config = configs[phase]
        actual = _project_path(config, key, config_path)
        expected_path = (ROOT / referenced).resolve()
        if actual != expected_path:
            raise ValueError(f"{label} must point to {expected_path}, got {actual}")

    p1_path, p1 = configs[1]
    p2_path, p2 = configs[2]
    p3_path, p3 = configs[3]
    p4_path, p4 = configs[4]
    p5_path, p5 = configs[5]
    p6_path, p6 = configs[6]

    same_path(2, "processed_dir", p1["processed_dir"], "Phase 2 processed_dir")
    same_path(2, "phase1_artifacts_dir", p1["artifacts_dir"], "Phase 2 phase1_artifacts_dir")
    same_path(3, "processed_dir", p1["processed_dir"], "Phase 3 processed_dir")
    same_path(3, "causal_dataset_csv", p2["causal_dataset_csv"], "Phase 3 causal_dataset_csv")
    same_path(3, "phase2_artifacts_dir", p2["phase2_artifacts_dir"], "Phase 3 phase2_artifacts_dir")
    same_path(3, "phase2_reports_dir", p2["phase2_reports_dir"], "Phase 3 phase2_reports_dir")
    same_path(4, "causal_dataset_csv", p2["causal_dataset_csv"], "Phase 4 causal_dataset_csv")
    same_path(4, "phase3_artifacts_dir", p3["phase3_artifacts_dir"], "Phase 4 phase3_artifacts_dir")
    same_path(5, "phase4_artifacts_dir", p4["phase4_artifacts_dir"], "Phase 5 phase4_artifacts_dir")
    same_path(5, "phase4_reports_dir", p4["phase4_reports_dir"], "Phase 5 phase4_reports_dir")
    same_path(6, "artifacts_dir", p6["artifacts_dir"], "Phase 6 artifacts_dir")
    for phase, field, referenced_phase in ((5, "phase4_config", 4), (6, "phase4_config", 4), (6, "phase5_config", 5)):
        config_path, config = configs[phase]
        referenced = Path(config[field])
        resolved = ((ROOT / referenced).resolve() if not referenced.is_absolute() else referenced.resolve())
        if resolved != configs[referenced_phase][0]:
            raise ValueError(f"Phase {phase} {field} must reference {configs[referenced_phase][0]}")


def _preflight(config_paths: dict[int, Path], overwrite: bool) -> list[Path]:
    configs = {phase: (path, _read_config(path)) for phase, path in config_paths.items()}
    _check_config_chain(configs)
    source = _project_path(configs[1][1], "source_csv", configs[1][0])
    if not source.is_file():
        raise FileNotFoundError(f"Phase 1 source dataset was not found: {source}")

    targets = _outputs(configs)
    protected = [source, *config_paths.values()]
    for target in targets:
        if target == ROOT or any(path == target or path.is_relative_to(target) for path in protected):
            raise ValueError(f"Refusing an output target that contains an input or config: {target}")
    conflicts = [target for target in targets if target.exists()]
    if conflicts and not overwrite:
        listing = "\n".join(f"  - {path.relative_to(ROOT)}" for path in conflicts)
        raise FileExistsError(
            "Outputs from a prior run already exist. Use --overwrite to replace only "
            f"the configured outputs for this pipeline:\n{listing}"
        )
    return targets


def _remove_outputs(targets: list[Path]) -> None:
    for target in targets:
        if target.is_dir():
            def make_writable(function, path, _error):
                os.chmod(path, stat.S_IWRITE)
                function(path)

            shutil.rmtree(target, onerror=make_writable)
        elif target.exists():
            os.chmod(target, stat.S_IWRITE)
            target.unlink()


def _run_phase(phase: int, config: Path) -> None:
    runners: dict[int, Callable] = {
        1: phase1_pipeline.run,
        2: phase2_pipeline.run,
        3: phase3_pipeline.run,
        4: phase4_pipeline.run,
        5: phase5_pipeline.run,
        6: phase6_pipeline.run,
    }
    if phase == 6:
        runners[phase](config, validate_only=False, smoke_test=False, audit_only=False)
    elif phase in (4, 5):
        runners[phase](config, validate_only=False, smoke_test=False)
    elif phase == 3:
        runners[phase](config, validate_only=False)
    else:
        runners[phase](config)


def run_pipeline(config_paths: dict[int, Path], overwrite: bool = False) -> None:
    targets = _preflight(config_paths, overwrite)
    if overwrite:
        LOG.warning("Removing configured outputs from the integrated Phases 1–6 chain.")
        _remove_outputs(targets)

    for phase in range(1, 7):
        config = config_paths[phase]
        LOG.info("========== Phase %s/6 starting (%s) =========", phase, config)
        try:
            _run_phase(phase, config)
        except Exception:
            LOG.exception("Phase %s failed; later phases were not run.", phase)
            raise
        LOG.info("========== Phase %s/6 complete =========", phase)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--phase1-config", dest="phase1_config", type=Path,
                        default=Path(DEFAULT_CONFIGS[1]),
                        help="Phase 1 config (alias: --phase1-config)")
    for phase in range(2, 7):
        parser.add_argument(f"--phase{phase}-config", type=Path,
                            default=Path(DEFAULT_CONFIGS[phase]))
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace configured outputs from this Phase 1–6 pipeline run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    config_paths = {phase: getattr(args, f"phase{phase}_config") for phase in range(1, 7)}
    config_paths = {
        phase: ((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
        for phase, path in config_paths.items()
    }
    try:
        for path in config_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Configuration file was not found: {path}")
        run_pipeline(config_paths, overwrite=args.overwrite)
    except Exception as error:
        LOG.error("Integrated pipeline stopped: %s", error)
        return 1
    LOG.info("All six phases completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
