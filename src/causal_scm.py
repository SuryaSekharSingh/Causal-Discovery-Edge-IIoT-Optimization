"""Interpretable structural causal model for Phase 2."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass
class StructuralEquation:
    variable: str
    parents: list[str]
    coefficients: dict[str, float]
    intercept: float
    residual_mean: float
    residual_std: float
    r2_train: float | None
    model_type: str = "linear_regression"


@dataclass
class LinearSCM:
    dag: dict[str, list[dict[str, Any]]]
    variables: list[str]
    equations: dict[str, StructuralEquation] = field(default_factory=dict)

    def topological_variables(self) -> list[str]:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.variables)
        for source, edges in self.dag.items():
            for edge in edges:
                graph.add_edge(source, edge["target"])
        return list(nx.topological_sort(graph))

    def parents_of(self, variable: str) -> list[str]:
        return [source for source, edges in self.dag.items() for edge in edges if edge["target"] == variable]

    def fit(self, frame: pd.DataFrame) -> "LinearSCM":
        self.equations = {}
        for variable in self.variables:
            parents = self.parents_of(variable)
            y = frame[variable].to_numpy(dtype=float)
            if parents:
                X = frame[parents].to_numpy(dtype=float)
                model = LinearRegression().fit(X, y)
                prediction = model.predict(X)
                coefficients = {parent: float(coef) for parent, coef in zip(parents, model.coef_)}
                intercept = float(model.intercept_)
                r2 = float(r2_score(y, prediction)) if len(np.unique(y)) > 1 else None
            else:
                prediction = np.full_like(y, fill_value=float(np.mean(y)), dtype=float)
                coefficients = {}
                intercept = float(np.mean(y))
                r2 = None
            residual = y - prediction
            self.equations[variable] = StructuralEquation(
                variable=variable,
                parents=parents,
                coefficients=coefficients,
                intercept=intercept,
                residual_mean=float(np.mean(residual)),
                residual_std=float(np.std(residual, ddof=0)),
                r2_train=r2,
            )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(index=frame.index)
        for variable in self.variables:
            equation = self.equations[variable]
            values = np.full(len(frame), equation.intercept, dtype=float)
            for parent, coefficient in equation.coefficients.items():
                values += coefficient * frame[parent].to_numpy(dtype=float)
            predictions[variable] = values
        return predictions

    def intervene(self, frame: pd.DataFrame, interventions: dict[str, float]) -> pd.DataFrame:
        simulated = frame[self.variables].copy()
        for variable, value in interventions.items():
            if variable not in self.variables:
                raise ValueError(f"Unknown intervention variable: {variable}")
            simulated[variable] = float(value)
        for variable in self.topological_variables():
            if variable in interventions:
                continue
            equation = self.equations[variable]
            if not equation.parents:
                continue
            values = np.full(len(simulated), equation.intercept, dtype=float)
            for parent, coefficient in equation.coefficients.items():
                values += coefficient * simulated[parent].to_numpy(dtype=float)
            simulated[variable] = values
        return simulated

    def metadata(self, train_frame: pd.DataFrame, validation_frame: pd.DataFrame | None = None) -> dict[str, Any]:
        train_predictions = self.predict(train_frame)
        payload = {
            "variables": self.variables,
            "dag": self.dag,
            "equations": {name: equation.__dict__ for name, equation in self.equations.items()},
            "fit_quality": {"train": self._quality(train_frame, train_predictions)},
        }
        if validation_frame is not None and not validation_frame.empty:
            payload["fit_quality"]["validation"] = self._quality(validation_frame, self.predict(validation_frame))
        return payload

    def _quality(self, observed: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, dict[str, float | None]]:
        quality: dict[str, dict[str, float | None]] = {}
        for variable in self.variables:
            y = observed[variable].to_numpy(dtype=float)
            y_hat = predicted[variable].to_numpy(dtype=float)
            quality[variable] = {
                "mae": float(mean_absolute_error(y, y_hat)),
                "r2": float(r2_score(y, y_hat)) if len(np.unique(y)) > 1 else None,
            }
        return quality


def save_scm(scm: LinearSCM, model_path: Path, metadata_path: Path, metadata: dict[str, Any]) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(scm, model_path)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_scm(model_path: Path) -> LinearSCM:
    return load(model_path)
