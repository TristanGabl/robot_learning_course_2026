"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        chunk_size,
        d_model: int = 300,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        self.input_layer = nn.Sequential(nn.Linear(state_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Dropout(dropout))
        layers = []
        for i in range(depth):
            layers += [nn.Linear(d_model, d_model),
                       nn.LayerNorm(d_model),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.body = nn.Sequential(*layers)
        self.output_layer = nn.Linear(d_model, chunk_size * action_dim)

    def forward(
        self,
        state
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        return self.output_layer(self.body(self.input_layer(state))).reshape(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state,
        action_chunk
    ) -> torch.Tensor:
        y = action_chunk
        y_hat = self.forward(state)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(state=state)
        


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def compute_loss(
        self,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(
        self,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        raise NotImplementedError


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model: int = 300,
    depth: int = 3,
    dropout: float = 0.1,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            d_model=d_model,
            depth=depth,
            chunk_size=chunk_size,
            dropout=dropout,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
