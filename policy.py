from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """Two-layer MLP that outputs a valid action probability distribution."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits = self.net(state)

        # Zero out logits for invalid actions before computing probabilities
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(action_mask == 0, -1e9)

        probs = F.softmax(logits, dim=-1)

        # Guard against NaN/Inf that can appear when all actions are masked
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Re-normalise to ensure probabilities sum to exactly 1
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        return probs


class PolicyNetwork(MLPPolicy):
    """Convenience wrapper that builds the state dimension from grammar sizes."""

    def __init__(self, n_symbols: int, n_actions: int, hidden_dim: int = 128):
        # State = current symbol + depth + action mask + parent action + sibling symbols
        state_dim = n_symbols + 1 + n_actions + n_actions + n_symbols
        super().__init__(state_dim=state_dim, num_actions=n_actions, hidden_dim=hidden_dim)


def encode_state(state: np.ndarray, *_, **__) -> torch.Tensor:
    # Convert a numpy state array to a float32 tensor for the policy network
    return torch.tensor(np.asarray(state, dtype=np.float32), dtype=torch.float32)
