from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from src.gnn_dataset import build_phase3_data
from src.gnn_model import EdgeAwareGATEncoder, Phase3GNN, masked_phase3_loss


class Phase3ModelTests(unittest.TestCase):
    def _graph(self) -> Data:
        return Data(
            x=torch.randn(3, 7),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            edge_attr=torch.randn(3, 66),
            node_ids=torch.tensor([10, 11, 12], dtype=torch.long),
            window_start="2021-01-01T00:00:00",
        )

    def _encoder(self) -> EdgeAwareGATEncoder:
        return EdgeAwareGATEncoder(
            node_dim=7,
            edge_dim=66,
            hidden_dim=64,
            embedding_dim=64,
            num_layers=2,
            attention_heads=4,
            dropout=0.0,
            graph_size_mean=(0.0, 0.0),
            graph_size_scale=(1.0, 1.0),
        )

    def test_forward_shapes_include_single_node_graph(self) -> None:
        single = Data(
            x=torch.randn(1, 7),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_attr=torch.randn(1, 66),
            node_ids=torch.tensor([1]),
            window_start="2021-01-01T00:00:00",
        )
        batch = Batch.from_data_list([single, self._graph()])
        model = Phase3GNN(self._encoder(), causal_dim=9).eval()
        outputs = model(batch)
        self.assertEqual(tuple(outputs["embedding"].shape), (2, 64))
        self.assertEqual(tuple(outputs["current_prediction"].shape), (2, 9))
        self.assertEqual(tuple(outputs["next_prediction"].shape), (2, 9))
        self.assertTrue(torch.isfinite(outputs["embedding"]).all())

    def test_node_permutation_invariance(self) -> None:
        graph = self._graph()
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(len(permutation))
        permuted = Data(
            x=graph.x[permutation],
            edge_index=inverse[graph.edge_index],
            edge_attr=graph.edge_attr.clone(),
            node_ids=graph.node_ids[permutation],
            window_start=graph.window_start,
        )
        encoder = self._encoder().eval()
        with torch.no_grad():
            original_embedding = encoder(graph)
            permuted_embedding = encoder(permuted)
        self.assertTrue(torch.allclose(original_embedding, permuted_embedding, atol=1e-5, rtol=1e-5))

    def test_both_message_directions_receive_gradients(self) -> None:
        encoder = self._encoder().train()
        loss = encoder(self._graph()).square().mean()
        loss.backward()
        incoming = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in encoder.named_parameters()
            if ".incoming." in name and parameter.grad is not None
        )
        outgoing = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in encoder.named_parameters()
            if ".outgoing." in name and parameter.grad is not None
        )
        self.assertGreater(incoming, 0.0)
        self.assertGreater(outgoing, 0.0)

    def test_masked_next_loss_handles_no_successor(self) -> None:
        outputs = {
            "current_prediction": torch.zeros(2, 3, requires_grad=True),
            "next_prediction": torch.ones(2, 3, requires_grad=True),
        }
        total, current, following = masked_phase3_loss(
            outputs,
            torch.ones(2, 3),
            torch.zeros(2, 3),
            torch.zeros(2),
        )
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(float(following.detach()), 0.0)
        self.assertEqual(float(current.detach()), 1.0)


class Phase3DatasetTests(unittest.TestCase):
    FEATURES = ["feature_a", "feature_b"]

    def _write_fixture(self, root: Path, duplicate: bool = False, mismatch: bool = False) -> dict:
        for split in ("train", "validation", "test"):
            (root / "data" / "processed" / split).mkdir(parents=True, exist_ok=True)
        (root / "artifacts" / "phase2").mkdir(parents=True)
        (root / "reports" / "phase2").mkdir(parents=True)
        (root / "data" / "interim").mkdir(parents=True)
        dag = {
            "metadata": {"features": self.FEATURES},
            "dag": {"feature_a": [{"target": "feature_b", "weight": 1.0}], "feature_b": []},
        }
        (root / "artifacts" / "phase2" / "refined_dag.json").write_text(json.dumps(dag), encoding="utf-8")
        decisions = {"used_features": self.FEATURES, "removed_features": []}
        (root / "reports" / "phase2" / "causal_variable_decisions.json").write_text(
            json.dumps(decisions), encoding="utf-8"
        )
        rows = []
        starts = {
            "train": ["2021-01-01T00:00:00", "2021-01-01T00:01:00"],
            "validation": ["2021-01-01T01:00:00", "2021-01-01T01:01:00"],
            "test": ["2021-01-01T02:00:00", "2021-01-01T02:02:00"],
        }
        value = 1.0
        for split, timestamps in starts.items():
            for index, timestamp in enumerate(timestamps):
                graph = Data(
                    x=torch.full((2, 7), value),
                    edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                    edge_attr=torch.full((1, 66), value),
                    node_ids=torch.tensor([0, 1]),
                    window_start=("2021-01-01T09:00:00" if mismatch and split == "train" and index == 0 else timestamp),
                )
                graph_file = f"graph_{index:03d}.pt"
                torch.save(graph, root / "data" / "processed" / split / graph_file)
                rows.append({
                    "split": split,
                    "graph_file": graph_file,
                    "window_start": timestamp,
                    "feature_a": value,
                    "feature_b": value * 10,
                })
                value += 1.0
        if duplicate:
            rows.append(rows[0].copy())
        pd.DataFrame(rows).to_csv(root / "data" / "interim" / "causal_dataset.csv", index=False)
        return {
            "processed_dir": "data/processed",
            "causal_dataset_csv": "data/interim/causal_dataset.csv",
            "phase2_artifacts_dir": "artifacts/phase2",
            "phase2_reports_dir": "reports/phase2",
            "window_seconds": 60,
            "graph_splits": {"train": "train", "validation": "validation", "test": "test"},
            "expected_schema": {"node_features": 7, "edge_features": 66},
        }

    def test_join_temporal_pairs_and_train_only_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_fixture(root)
            bundle = build_phase3_data(root, config)
            self.assertEqual(bundle.split_counts, {"train": 2, "validation": 2, "test": 2})
            self.assertEqual(bundle.temporal_pair_counts, {"train": 1, "validation": 1, "test": 0})
            np.testing.assert_allclose(bundle.normalization.causal_mean, [1.5, 15.0])
            item = bundle.datasets["train"][0]
            self.assertEqual(float(item.next_mask), 1.0)
            self.assertEqual(tuple(item.y_current.shape), (1, 2))

    def test_duplicate_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_fixture(root, duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build_phase3_data(root, config)

    def test_timestamp_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_fixture(root, mismatch=True)
            with self.assertRaisesRegex(ValueError, "Timestamp mismatch"):
                build_phase3_data(root, config)


if __name__ == "__main__":
    unittest.main()
