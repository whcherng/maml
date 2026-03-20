#!/usr/bin/env python3
"""
Baseline 3: PPO Reinforcement Learning Agent

A Proximal Policy Optimization agent that learns a trading policy by
maximizing cumulative risk-adjusted returns. Unlike MAML, it doesn't
have a fast-adaptation mechanism — it learns a single policy during
training and deploys it fixed during testing.

The PPO agent uses the same features and position sizing as the other
models for fair comparison. The key difference is the training objective:
supervised classification (LSTM baselines) vs. reward maximization (PPO)
vs. meta-learning (MAML).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple

from maml_trading.config import MAMLConfig
from maml_trading.metrics import PerformanceMetrics


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, seq_len: int = 30):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # Actor (policy) — outputs position in [0, 1]
        self.actor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # position between 0 and 1
        )
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        position = self.actor(h)
        value = self.critic(h)
        return position, value


class PPOBaseline:
    """
    PPO agent for trading. Learns a policy that maps market state → position.
    """

    def __init__(
        self,
        config: MAMLConfig,
        feature_columns: List[str],
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        n_epochs_per_update: int = 4,
        batch_size: int = 64,
    ):
        self.cfg = config
        self.feature_columns = feature_columns
        self.device = torch.device(config.device)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.n_epochs_per_update = n_epochs_per_update
        self.batch_size = batch_size

        self.network = PPONetwork(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            seq_len=config.sequence_length,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def train_agent(
        self,
        processed_data: Dict[str, "pd.DataFrame"],
        train_end: int,
        n_episodes: int = 100,
    ) -> None:
        """
        Train PPO on the training data by simulating trading episodes.
        Each episode: pick a random ticker, trade through a random
        window, collect rewards, update policy.
        """
        tickers = list(processed_data.keys())
        seq_len = self.cfg.sequence_length

        for episode in range(n_episodes):
            # Pick random ticker and starting point
            ticker = np.random.choice(tickers)
            df = processed_data[ticker]
            features = df[self.feature_columns].values[:train_end].astype(np.float32)
            raw_returns = df["Raw_Returns"].values[:train_end] if "Raw_Returns" in df.columns else np.zeros(train_end)

            # Random episode start
            max_start = len(features) - seq_len - 60
            if max_start < seq_len:
                continue
            start = np.random.randint(seq_len, max_start)
            episode_len = min(60, len(features) - start)

            # Collect trajectory
            states, actions, rewards, values, log_probs = [], [], [], [], []
            prev_position = 0.0

            self.network.train()
            for t in range(start, start + episode_len):
                if t < seq_len:
                    continue
                x = torch.tensor(
                    features[t - seq_len:t], dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                position, value = self.network(x)
                pos_val = position.item()

                # Add exploration noise
                noise = np.random.normal(0, 0.1)
                pos_val = np.clip(pos_val + noise, 0, 1)

                # Compute reward (daily return * position - cost)
                daily_ret = raw_returns[t] if t < len(raw_returns) else 0.0
                trade_cost = abs(pos_val - prev_position) * 0.0015  # 15bps
                reward = daily_ret * pos_val - trade_cost

                states.append(x)
                actions.append(pos_val)
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(self._log_prob(position, pos_val))
                prev_position = pos_val

            if len(rewards) < 5:
                continue

            # Compute returns and advantages
            returns = self._compute_returns(rewards)
            advantages = np.array(returns) - np.array(values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update
            self._ppo_update(states, actions, log_probs, returns, advantages)

    def _log_prob(self, position_tensor: torch.Tensor, action: float) -> torch.Tensor:
        """Approximate log probability using Gaussian around the mean."""
        std = 0.1
        mean = position_tensor.squeeze()
        diff = torch.tensor(action, dtype=torch.float32).to(self.device) - mean
        return -0.5 * (diff / std) ** 2

    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """Discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def _ppo_update(self, states, actions, old_log_probs, returns, advantages):
        """PPO clipped objective update."""
        returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        for _ in range(self.n_epochs_per_update):
            for i in range(0, len(states), self.batch_size):
                batch_end = min(i + self.batch_size, len(states))
                batch_states = torch.cat(states[i:batch_end], dim=0)
                batch_actions = actions[i:batch_end]
                batch_old_lp = old_log_probs[i:batch_end]
                batch_returns = returns_t[i:batch_end]
                batch_adv = advantages_t[i:batch_end]

                positions, values = self.network(batch_states)

                # New log probs
                new_log_probs = torch.stack([
                    self._log_prob(positions[j], batch_actions[j])
                    for j in range(len(batch_actions))
                ])
                old_lp = torch.stack([lp.detach() for lp in batch_old_lp])

                # Ratio and clipped loss
                ratio = torch.exp(new_log_probs - old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = actor_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

    @torch.no_grad()
    def get_position(self, x: torch.Tensor) -> float:
        """Get position for a single state sequence."""
        self.network.eval()
        x = x.unsqueeze(0).to(self.device)
        position, _ = self.network(x)
        return float(position.item())
