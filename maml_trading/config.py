"""
Configuration dataclass for the MAML Trading Framework.

Centralizes all hyperparameters for data ingestion, task generation,
meta-learning optimization, backtesting, and execution simulation.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MAMLConfig:
    """Master configuration for the MAML trading pipeline."""

    # ── Data Ingestion ──────────────────────────────────────────────
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "NVDA", "JPM", "V", "JNJ", "UNH",
        "PG", "HD", "MA", "DIS", "BAC",
        "XOM", "PFE", "CSCO", "ADBE", "CRM"
    ])
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"

    # ── Feature Engineering ─────────────────────────────────────────
    sma_short: int = 50
    sma_long: int = 200
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    outlier_lower_pct: float = 0.01   # 1st percentile trim
    outlier_upper_pct: float = 0.99   # 99th percentile trim
    rolling_norm_window: int = 60     # rolling z-score window

    # ── Sentiment (FinBERT) ─────────────────────────────────────────
    finbert_model_name: str = "ProsusAI/finbert"
    use_sentiment: bool = False  # toggle; requires GPU + news data

    # ── Market Index Features ───────────────────────────────────────
    index_tickers: List[str] = field(default_factory=lambda: [
        "^IXIC",   # NASDAQ Composite
        "^GSPC",   # S&P 500
    ])

    # ── Task Generation ─────────────────────────────────────────────
    regime_window_min: int = 60       # minimum task window (days)
    regime_window_max: int = 90       # maximum task window (days)
    k_shot: int = 20                  # support set size (days)
    query_size: int = 20              # query set size (days)
    tasks_per_batch: int = 12          # meta-batch size

    # ── Model Architecture ──────────────────────────────────────────
    input_dim: int = 0                # set dynamically after features
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    output_dim: int = 2               # flat / long (binary)

    # ── MAML Optimization ───────────────────────────────────────────
    meta_lr: float = 1e-3             # outer-loop learning rate
    inner_lr: float = 1e-2            # inner-loop learning rate
    inner_steps: int = 5              # gradient steps in inner loop
    meta_epochs: int = 200            # outer-loop iterations
    use_kronecker: bool = True        # KroneckerTransform for meta-update
    first_order: bool = False         # False = full second-order MAML

    # ── Walk-Forward Backtesting ────────────────────────────────────
    train_ratio: float = 0.60
    val_ratio: float = 0.15
    test_ratio: float = 0.25
    embargo_days: int = 10            # gap between val and test

    # ── Execution Simulation ────────────────────────────────────────
    transaction_cost_bps: float = 15.0   # round-trip cost in bps
    almgren_chriss_eta: float = 0.1      # temporary impact coefficient
    almgren_chriss_gamma: float = 0.001  # permanent impact coefficient
    volatility_lookback: int = 20        # days for realized vol estimate

    # ── General ─────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cpu"  # "cuda" if available
    sequence_length: int = 30  # lookback window for LSTM input
