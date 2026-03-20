#!/usr/bin/env python3
"""
Baseline 2: Online-Learning LSTM

The standard industry approach: periodically retrain the model on a
sliding window of recent data. Every N days, the model is fine-tuned
on the most recent `retrain_window` days of data.

This is "reactive" adaptation — it only adjusts AFTER the regime has
already changed, which means it's always lagging behind.

Compared to MAML: Online-Learning adapts slowly (full retraining every
N days) while MAML adapts quickly (3 gradient steps on 30 samples daily).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple

from maml_trading.config import MAMLConfig
from maml_trading.model import AdaptiveTradingNetwork
from maml_trading.metrics import PerformanceMetrics


class OnlineLSTMBaseline:
    """
    Sliding-window retraining every `retrain_every` days.
    """

    def __init__(
        self,
        config: MAMLConfig,
        feature_columns: List[str],
        retrain_window: int = 252,   # 1 year of recent data
        retrain_every: int = 21,     # retrain monthly
        retrain_epochs: int = 10,    # quick fine-tuning
    ):
        self.cfg = config
        self.feature_columns = feature_columns
        self.device = torch.device(config.device)
        self.retrain_window = retrain_window
        self.retrain_every = retrain_every
        self.retrain_epochs = retrain_epochs

        self.model = AdaptiveTradingNetwork(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _build_sequences(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.cfg.sequence_length
        X, Y = [], []
        for i in range(seq_len, len(features)):
            X.append(features[i - seq_len:i])
            Y.append(labels[i - 1])
        if not X:
            return torch.zeros(0), torch.zeros(0, dtype=torch.long)
        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(Y), dtype=torch.long),
        )

    def _retrain(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fine-tune on recent window of data."""
        X, Y = self._build_sequences(features, labels)
        if len(X) < 10:
            return
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        for _ in range(self.retrain_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

    def initial_train(
        self,
        processed_data: Dict[str, "pd.DataFrame"],
        train_end: int,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> None:
        """Initial training on the full training set (same as Static)."""
        all_X, all_Y = [], []
        for ticker, df in processed_data.items():
            features = df[self.feature_columns].values[:train_end].astype(np.float32)
            labels = df["Target"].values[:train_end].astype(np.int64)
            X, Y = self._build_sequences(features, labels)
            all_X.append(X)
            all_Y.append(Y)

        X_train = torch.cat(all_X, dim=0)
        Y_train = torch.cat(all_Y, dim=0)
        dataset = TensorDataset(X_train, Y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

    def should_retrain(self, day_idx: int, test_start: int) -> bool:
        """Check if it's time to retrain."""
        days_since_start = day_idx - test_start
        return days_since_start > 0 and days_since_start % self.retrain_every == 0

    def retrain_on_recent(
        self, features: np.ndarray, labels: np.ndarray, current_idx: int
    ) -> None:
        """Retrain on the most recent `retrain_window` days."""
        start = max(0, current_idx - self.retrain_window)
        self._retrain(features[start:current_idx], labels[start:current_idx])

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[int, np.ndarray]:
        self.model.eval()
        x = x.unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
        return int(pred), probs
