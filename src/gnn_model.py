"""Edge-aware bidirectional GATv2 encoder used by Phase 3 and later phases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class BidirectionalGATBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention_heads.")
        per_head = hidden_dim // heads
        kwargs = {
            "heads": heads,
            "concat": True,
            "edge_dim": edge_dim,
            "dropout": dropout,
            "add_self_loops": True,
            "fill_value": "mean",
        }
        self.incoming = GATv2Conv(hidden_dim, per_head, **kwargs)
        self.outgoing = GATv2Conv(hidden_dim, per_head, **kwargs)
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        incoming = self.incoming(x, edge_index, edge_attr)
        outgoing = self.outgoing(x, edge_index.flip(0), edge_attr)
        messages = self.fuse(torch.cat([incoming, outgoing], dim=-1))
        return self.norm(x + self.dropout(F.elu(messages)))


class EdgeAwareGATEncoder(nn.Module):
    """Encode one or more directed network graphs into fixed-width vectors."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.2,
        graph_size_mean: Iterable[float] = (0.0, 0.0),
        graph_size_scale: Iterable[float] = (1.0, 1.0),
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [BidirectionalGATBlock(hidden_dim, edge_dim, attention_heads, dropout) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.register_buffer("graph_size_mean", torch.tensor(list(graph_size_mean), dtype=torch.float32))
        self.register_buffer("graph_size_scale", torch.tensor(list(graph_size_scale), dtype=torch.float32))

    def forward(self, data: Any) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        hidden = self.input_projection(x)
        for block in self.blocks:
            hidden = block(hidden, edge_index, edge_attr)

        mean_pool = global_mean_pool(hidden, batch)
        max_pool = global_max_pool(hidden, batch)
        graph_count = mean_pool.shape[0]
        node_count = torch.bincount(batch, minlength=graph_count).to(hidden.dtype)
        if edge_index.shape[1]:
            edge_batch = batch[edge_index[0]]
            edge_count = torch.bincount(edge_batch, minlength=graph_count).to(hidden.dtype)
        else:
            edge_count = torch.zeros(graph_count, dtype=hidden.dtype, device=hidden.device)
        sizes = torch.stack([torch.log1p(node_count), torch.log1p(edge_count)], dim=-1)
        sizes = (sizes - self.graph_size_mean) / self.graph_size_scale
        return self.readout(torch.cat([mean_pool, max_pool, sizes], dim=-1))


class Phase3GNN(nn.Module):
    def __init__(self, encoder: EdgeAwareGATEncoder, causal_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.current_head = self._head(encoder.embedding_dim, causal_dim)
        self.next_head = self._head(encoder.embedding_dim, causal_dim)

    @staticmethod
    def _head(embedding_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ELU(),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        embedding = self.encoder(data)
        return {
            "embedding": embedding,
            "current_prediction": self.current_head(embedding),
            "next_prediction": self.next_head(embedding),
        }


def masked_phase3_loss(
    outputs: dict[str, torch.Tensor],
    current_target: torch.Tensor,
    next_target: torch.Tensor,
    next_mask: torch.Tensor,
    current_weight: float = 1.0,
    next_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_loss = F.mse_loss(outputs["current_prediction"], current_target)
    per_graph_next = F.mse_loss(outputs["next_prediction"], next_target, reduction="none").mean(dim=1)
    mask = next_mask.reshape(-1).to(per_graph_next.dtype)
    if float(mask.sum().detach().cpu()) > 0:
        next_loss = (per_graph_next * mask).sum() / mask.sum()
    else:
        next_loss = per_graph_next.sum() * 0.0
    total = current_weight * current_loss + next_weight * next_loss
    return total, current_loss, next_loss


def _encoder_from_payload(payload: dict[str, Any]) -> EdgeAwareGATEncoder:
    model_config = payload["model_config"]
    normalization = payload["normalization"]
    return EdgeAwareGATEncoder(
        node_dim=int(model_config["node_dim"]),
        edge_dim=int(model_config["edge_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_layers=int(model_config["num_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
        graph_size_mean=normalization["graph_size_mean"],
        graph_size_scale=normalization["graph_size_scale"],
    )


def load_phase3_encoder(checkpoint: str | Path, device: str | torch.device = "cpu") -> EdgeAwareGATEncoder:
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    encoder = _encoder_from_payload(payload)
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_graphs(encoder: EdgeAwareGATEncoder, loader: Any, device: str | torch.device = "cpu") -> torch.Tensor:
    encoder.eval()
    encoded: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        encoded.append(encoder(batch).cpu())
    if not encoded:
        return torch.empty((0, encoder.embedding_dim), dtype=torch.float32)
    return torch.cat(encoded, dim=0)
