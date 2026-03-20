#!/usr/bin/env python3
"""
Baseline 1: Static LSTM

Trained ONCE on the training set, then deployed on the test set without
any retraining or adaptation. This is the "control" model that demonstrates
how serious model decay is — a model trained on 2015-2020 data will
degrade when market regime shifts in 2022+.

This is the simplest possible approach and represents what most practitioners
do: train a model, deploy it, and hope it keeps working.
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


class StaticLSTMBaseline:
    """
    Train once, test forever. No adaptation, no retraining.
    """

    def __init__(self, config: MAMLConfig, feature_columns: List[str]):
        self.cfg = config
        self.feature_columns = feature_columns
        self.device = torch.device(config.device)

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
        """Build (seq_len, F) sequences from flat arrays."""
        seq_len = self.cfg.sequence_length
        X, Y = [], []
        for i in range(seq_len, len(features)):
            X.append(features[i - seq_len:i])
            Y.append(labels[i - 1])
        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(Y), dtype=torch.long),
        )

    def train_model(
        self,
        processed_data: Dict[str, "pd.DataFrame"],
        train_end: int,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> None:
        """Train on all tickers' training data combined."""
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
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[int, np.ndarray]:
        """Predict class and probabilities for a single sequence."""
        self.model.eval()
        x = x.unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
        return int(pred), probs
