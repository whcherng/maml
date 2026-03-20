#!/usr/bin/env python3
"""
MAML Trading Framework — Main Entry Point.

Orchestrates the full pipeline:
  1. Data download & feature engineering
  2. Task generation (market regime windows)
  3. MAML meta-training (bi-level optimization)
  4. Walk-forward backtesting with execution simulation
  5. Performance evaluation

Usage:
    python main.py

The MAML agent is structured so it can be compared against baselines
(Static LSTM, Online-Learning LSTM, PPO-DRL) by swapping the engine
while keeping the same data pipeline, task generator, and backtester.
"""

import random
import numpy as np
import torch

from maml_trading.config import MAMLConfig
from maml_trading.data_pipeline import DataPipeline
from maml_trading.task_generator import MarketRegimeTaskGenerator
from maml_trading.maml_engine import MAMLTradingEngine
from maml_trading.backtester import WalkForwardBacktester
from maml_trading.metrics import PerformanceMetrics


def set_seed(seed: int) -> None:
    """Reproducibility across numpy, torch, and python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # ── 1. Configuration ────────────────────────────────────────────
    config = MAMLConfig(
        tickers=["AAPL", "MSFT", "GOOGL", "JPM", "NVDA", "TEAM"],
        start_date="2015-01-01",
        end_date="2024-12-31",
        # Meta-learning — tuned for small model
        meta_epochs=200,
        inner_steps=3,
        inner_lr=0.01,
        meta_lr=0.001,
        tasks_per_batch=12,
        k_shot=30,
        query_size=20,
        # Architecture — small model for MAML
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        output_dim=2,  # binary: flat / long
        # Task generation
        regime_window_min=60,
        regime_window_max=90,
        sequence_length=30,
        # Backtesting
        train_ratio=0.60,
        val_ratio=0.15,
        test_ratio=0.25,
        embargo_days=10,
        transaction_cost_bps=15.0,
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    set_seed(config.seed)
    print(f"[Main] Device: {config.device}")

    # ── 2. Data Pipeline ────────────────────────────────────────────
    pipeline = DataPipeline(config)
    processed_data = pipeline.run()

    feature_columns = pipeline.get_feature_columns()
    config.input_dim = len(feature_columns)
    print(f"[Main] Feature dimension: {config.input_dim}")
    print(f"[Main] Features: {feature_columns}")

    # ── 3. Walk-Forward Backtest ────────────────────────────────────
    # The backtester internally creates task generators and MAML
    # engines for each fold, respecting temporal boundaries.
    backtester = WalkForwardBacktester(
        config=config,
        processed_data=processed_data,
        feature_columns=feature_columns,
    )

    results = backtester.run(
        meta_train_epochs=config.meta_epochs,
        verbose=True,
    )

    # ── 4. Final Report ─────────────────────────────────────────────
    if results:
        print("\n" + "=" * 60)
        print("  FINAL MAML TRADING FRAMEWORK RESULTS")
        print("=" * 60)
        PerformanceMetrics.print_report(results)

        # ── 5. Benchmark Comparison ─────────────────────────────────
        benchmarks = pipeline.get_benchmark_returns(
            start_idx=backtester.test_start_idx,
            end_idx=backtester.test_end_idx,
        )
        if benchmarks:
            PerformanceMetrics.print_benchmark_comparison(results, benchmarks)

    return results


if __name__ == "__main__":
    main()
