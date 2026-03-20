"""
Task Generation Module using learn2learn interfaces.

Defines a "task" as a market regime window (60-90 days) and splits
it into K-shot support sets (recent data for fast adaptation) and
query sets (subsequent data for meta-evaluation).

Uses learn2learn's MetaDataset / TaskDataset patterns to procedurally
generate episodic tasks from the preprocessed time-series data.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from maml_trading.config import MAMLConfig


# ════════════════════════════════════════════════════════════════════
# 1. Single-Ticker Time-Series Dataset
# ════════════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    """
    Wraps a preprocessed ticker DataFrame into a PyTorch Dataset.
    Each sample is a (feature_sequence, label) pair where the
    feature_sequence has shape (sequence_length, num_features).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 30,
    ):
        """
        Args:
            features: (T, F) array of normalized features.
            labels:   (T,) array of ternary targets.
            sequence_length: lookback window for each sample.
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.seq_len = sequence_length
        # Valid start indices so every sample has a full lookback
        self.valid_indices = list(range(self.seq_len, len(self.features)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = self.valid_indices[idx]
        start = end - self.seq_len
        x = torch.tensor(self.features[start:end])   # (seq_len, F)
        y = torch.tensor(self.labels[end - 1])        # scalar
        return x, y


# ════════════════════════════════════════════════════════════════════
# 2. Market Regime Task Generator
# ════════════════════════════════════════════════════════════════════

class MarketRegimeTask:
    """
    A single meta-learning task representing one market regime window.

    Attributes:
        support_x: (K, seq_len, F) tensor — recent regime data.
        support_y: (K,) tensor — labels for support set.
        query_x:   (Q, seq_len, F) tensor — subsequent regime data.
        query_y:   (Q,) tensor — labels for query set.
        ticker:    source ticker symbol.
        start_date / end_date: regime window boundaries.
    """

    def __init__(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        ticker: str = "",
        start_idx: int = 0,
        end_idx: int = 0,
    ):
        self.support_x = support_x
        self.support_y = support_y
        self.query_x = query_x
        self.query_y = query_y
        self.ticker = ticker
        self.start_idx = start_idx
        self.end_idx = end_idx


class MarketRegimeTaskGenerator:
    """
    Procedurally generates meta-learning tasks by slicing time-series
    into regime windows and splitting into support / query sets.

    Mirrors the learn2learn TaskDataset pattern:
      - Each call to `sample_tasks()` returns a batch of
        MarketRegimeTask objects ready for the MAML inner/outer loop.
    """

    def __init__(
        self,
        processed_data: Dict[str, "pd.DataFrame"],
        feature_columns: List[str],
        config: MAMLConfig,
    ):
        """
        Args:
            processed_data: {ticker: DataFrame} from DataPipeline.
            feature_columns: list of feature column names.
            config: MAMLConfig instance.
        """
        self.cfg = config
        self.feature_columns = feature_columns

        # Pre-convert each ticker's data to numpy for fast slicing
        self.ticker_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for ticker, df in processed_data.items():
            feats = df[feature_columns].values.astype(np.float32)
            labels = df["Target"].values.astype(np.int64)
            # Only keep tickers with enough data for at least one task
            min_len = (
                config.regime_window_max
                + config.sequence_length
            )
            if len(feats) >= min_len:
                self.ticker_data[ticker] = (feats, labels)

        self.tickers = list(self.ticker_data.keys())
        if not self.tickers:
            raise ValueError(
                "No tickers have enough data for task generation. "
                "Check date range and regime_window_max."
            )
        print(
            f"[TaskGenerator] {len(self.tickers)} tickers available "
            f"for task sampling."
        )

    def _sample_single_task(
        self,
        ticker: Optional[str] = None,
        start_range: Optional[Tuple[int, int]] = None,
    ) -> MarketRegimeTask:
        """
        Sample one task (regime window) from a random or specified ticker.

        Steps:
          1. Pick a random ticker (or use the one provided).
          2. Sample a regime window length ∈ [regime_window_min, regime_window_max].
          3. Pick a random start index within the valid range.
          4. Slice features/labels for the window.
          5. Split into support (first K samples) and query (next Q samples).
          6. Build sequences of length `sequence_length` for each sample.

        Args:
            ticker: optional ticker to sample from.
            start_range: optional (lo, hi) to constrain the start index
                         (used for walk-forward splits).
        """
        if ticker is None:
            ticker = random.choice(self.tickers)

        feats, labels = self.ticker_data[ticker]
        T = len(feats)

        # Regime window length
        window = random.randint(self.cfg.regime_window_min, self.cfg.regime_window_max)
        total_needed = self.cfg.sequence_length + window

        # Determine valid start range
        if start_range is not None:
            lo, hi = start_range
            lo = max(lo, 0)
            hi = min(hi, T - total_needed)
        else:
            lo = 0
            hi = T - total_needed

        if hi <= lo:
            # Fallback: use the latest available window
            start = max(0, T - total_needed)
        else:
            start = random.randint(lo, hi)

        # Slice the regime window (after the sequence lookback)
        window_start = start + self.cfg.sequence_length
        window_end = window_start + window

        # Build (x, y) pairs with lookback sequences
        k = self.cfg.k_shot
        q = self.cfg.query_size
        # Ensure we don't exceed the window
        actual_samples = min(k + q, window)
        k = min(k, actual_samples // 2)
        q = actual_samples - k

        support_xs, support_ys = [], []
        query_xs, query_ys = [], []

        for i in range(actual_samples):
            idx = window_start + i
            seq = feats[idx - self.cfg.sequence_length : idx]  # (seq_len, F)
            label = labels[idx - 1]

            if i < k:
                support_xs.append(seq)
                support_ys.append(label)
            else:
                query_xs.append(seq)
                query_ys.append(label)

        return MarketRegimeTask(
            support_x=torch.tensor(np.array(support_xs)),
            support_y=torch.tensor(np.array(support_ys)),
            query_x=torch.tensor(np.array(query_xs)),
            query_y=torch.tensor(np.array(query_ys)),
            ticker=ticker,
            start_idx=window_start,
            end_idx=window_end,
        )

    def sample_tasks(
        self,
        n_tasks: Optional[int] = None,
        start_range: Optional[Tuple[int, int]] = None,
    ) -> List[MarketRegimeTask]:
        """
        Sample a meta-batch of tasks.

        Args:
            n_tasks: number of tasks (defaults to cfg.tasks_per_batch).
            start_range: optional index range constraint for walk-forward.

        Returns:
            List of MarketRegimeTask objects.
        """
        if n_tasks is None:
            n_tasks = self.cfg.tasks_per_batch

        tasks = []
        for _ in range(n_tasks):
            task = self._sample_single_task(start_range=start_range)
            tasks.append(task)
        return tasks

    def sample_tasks_for_period(
        self,
        ticker: str,
        period_start: int,
        period_end: int,
        n_tasks: int = 4,
    ) -> List[MarketRegimeTask]:
        """
        Sample tasks constrained to a specific time period for a ticker.
        Used during walk-forward backtesting to respect temporal splits.
        """
        return [
            self._sample_single_task(
                ticker=ticker,
                start_range=(period_start, period_end),
            )
            for _ in range(n_tasks)
        ]
