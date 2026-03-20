#!/usr/bin/env python3
"""
Generate publication-quality figures for the CP2 paper.

Produces:
  1. Equity curves (MAML vs baselines vs buy-and-hold)
  2. Rolling Sharpe comparison bar chart
  3. Drawdown comparison chart
  4. Per-ticker heatmap

Usage:
    python generate_figures.py

Saves PNG files to figures/ directory.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from maml_trading.config import MAMLConfig
from maml_trading.data_pipeline import DataPipeline
from maml_trading.backtester import WalkForwardBacktester, ExecutionSimulator
from maml_trading.metrics import PerformanceMetrics
from baselines.static_lstm import StaticLSTMBaseline
from baselines.online_lstm import OnlineLSTMBaseline
from baselines.ppo_agent import PPOBaseline
from run_comparison import run_baseline_backtest, set_seed, signal_to_position


def setup():
    """Build data and models."""
    config = MAMLConfig(
        tickers=["AAPL", "MSFT", "GOOGL", "JPM", "NVDA", "TEAM"],
        start_date="2015-01-01", end_date="2024-12-31",
        meta_epochs=200, inner_steps=3, inner_lr=0.01, meta_lr=0.001,
        tasks_per_batch=12, k_shot=30, query_size=20,
        hidden_dim=64, num_layers=1, dropout=0.1, output_dim=2,
        regime_window_min=60, regime_window_max=90, sequence_length=30,
        train_ratio=0.60, val_ratio=0.15, test_ratio=0.25,
        embargo_days=10, transaction_cost_bps=15.0,
        device="cpu",
    )
    set_seed(42)
    pipeline = DataPipeline(config)
    processed_data = pipeline.run()
    feature_columns = pipeline.get_feature_columns()
    config.input_dim = len(feature_columns)
    return config, pipeline, processed_data, feature_columns


def fig1_equity_curves(config, pipeline, processed_data, feature_columns):
    """Figure 1: Cumulative equity curves for all models."""
    print("  Generating Figure 1: Equity Curves...")

    max_len = max(len(df) for df in processed_data.values())
    train_end = int(max_len * config.train_ratio)
    val_end = train_end + int(max_len * config.val_ratio)
    embargo_end = val_end + config.embargo_days
    test_end = max_len

    # Get daily returns from each model
    results = {}

    # Static LSTM
    set_seed(42)
    static = StaticLSTMBaseline(config, feature_columns)
    static.train_model(processed_data, train_end, epochs=50)
    r = run_baseline_backtest(static, "static", processed_data,
                              feature_columns, config, embargo_end, test_end)
    results["Static LSTM"] = r.get("daily_returns", np.array([]))

    # Online LSTM
    set_seed(42)
    online = OnlineLSTMBaseline(config, feature_columns)
    online.initial_train(processed_data, train_end, epochs=50)
    r = run_baseline_backtest(online, "online", processed_data,
                              feature_columns, config, embargo_end, test_end)
    results["Online LSTM"] = r.get("daily_returns", np.array([]))

    # PPO
    set_seed(42)
    ppo = PPOBaseline(config, feature_columns)
    ppo.train_agent(processed_data, train_end, n_episodes=200)
    r = run_baseline_backtest(ppo, "ppo", processed_data,
                              feature_columns, config, embargo_end, test_end)
    results["PPO Agent"] = r.get("daily_returns", np.array([]))

    # MAML
    set_seed(42)
    bt = WalkForwardBacktester(config, processed_data, feature_columns)
    r = bt.run(meta_train_epochs=config.meta_epochs, verbose=False)
    results["MAML (Ours)"] = r.get("daily_returns", np.array([]))

    # S&P 500 benchmark
    benchmarks = pipeline.get_benchmark_returns(embargo_end, test_end)
    if "GSPC" in benchmarks:
        results["S&P 500 (B&H)"] = benchmarks["GSPC"]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"Static LSTM": "#d62728", "Online LSTM": "#ff7f0e",
              "PPO Agent": "#2ca02c", "MAML (Ours)": "#1f77b4",
              "S&P 500 (B&H)": "#7f7f7f"}
    linewidths = {"MAML (Ours)": 2.5, "S&P 500 (B&H)": 1.5}

    for name, returns in results.items():
        if len(returns) == 0:
            continue
        equity = np.cumprod(1 + returns)
        lw = linewidths.get(name, 1.2)
        ls = "--" if name == "S&P 500 (B&H)" else "-"
        ax.plot(equity, label=name, color=colors.get(name, "gray"),
                linewidth=lw, linestyle=ls)

    ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Trading Days (Test Period)", fontsize=11)
    ax.set_ylabel("Cumulative Equity", fontsize=11)
    ax.set_title("Figure 1: Equity Curves — MAML vs. Baselines", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig1_equity_curves.png", dpi=150)
    plt.close()
    print("    ✓ Saved figures/fig1_equity_curves.png")


def fig2_rolling_sharpe_bars():
    """Figure 2: Bar chart of Sharpe ratios per year."""
    print("  Generating Figure 2: Rolling Sharpe Bar Chart...")

    years = ["2020", "2021", "2022", "2023", "2024"]
    maml_sharpe = [2.397, 2.551, -0.544, 1.481, 1.105]
    nasdaq_sharpe = [1.348, 1.494, -0.829, 2.162, 1.341]
    sp500_sharpe = [0.802, 2.124, -0.523, 2.181, 1.333]

    x = np.arange(len(years))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, maml_sharpe, width, label="MAML (Ours)",
                   color="#1f77b4", edgecolor="white")
    bars2 = ax.bar(x, nasdaq_sharpe, width, label="NASDAQ",
                   color="#ff7f0e", edgecolor="white")
    bars3 = ax.bar(x + width, sp500_sharpe, width, label="S&P 500",
                   color="#2ca02c", edgecolor="white")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Test Year", fontsize=11)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title("Figure 2: Sharpe Ratio by Year — MAML vs. Market Benchmarks", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 0.1:
                ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points",
                           ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/fig2_sharpe_by_year.png", dpi=150)
    plt.close()
    print("    ✓ Saved figures/fig2_sharpe_by_year.png")


def fig3_max_drawdown_comparison():
    """Figure 3: Max drawdown comparison."""
    print("  Generating Figure 3: Drawdown Comparison...")

    years = ["2020", "2021", "2022", "2023", "2024"]
    maml_dd = [-5.92, -4.02, -11.72, -5.61, -6.66]
    nasdaq_dd = [-23.92, -7.83, -30.14, -12.27, -13.15]
    sp500_dd = [-28.52, -5.21, -22.77, -10.28, -8.49]

    x = np.arange(len(years))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, maml_dd, width, label="MAML (Ours)", color="#1f77b4")
    ax.bar(x, nasdaq_dd, width, label="NASDAQ", color="#ff7f0e")
    ax.bar(x + width, sp500_dd, width, label="S&P 500", color="#2ca02c")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Test Year", fontsize=11)
    ax.set_ylabel("Maximum Drawdown (%)", fontsize=11)
    ax.set_title("Figure 3: Maximum Drawdown by Year — MAML vs. Benchmarks", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("figures/fig3_max_drawdown.png", dpi=150)
    plt.close()
    print("    ✓ Saved figures/fig3_max_drawdown.png")


def fig4_model_comparison_summary():
    """Figure 4: Summary comparison of all models."""
    print("  Generating Figure 4: Model Comparison Summary...")

    models = ["Static\nLSTM", "Online\nLSTM", "PPO\nAgent", "MAML\n(Ours)"]
    sharpe = [-0.982, -0.873, 1.317, 1.692]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, sharpe, color=colors, edgecolor="white", width=0.6)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title("Figure 4: Model Comparison — Sharpe Ratio", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, sharpe):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 5 if val > 0 else -15),
                   textcoords="offset points", ha="center", fontsize=11,
                   fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures/fig4_model_comparison.png", dpi=150)
    plt.close()
    print("    ✓ Saved figures/fig4_model_comparison.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("\n  Generating figures for CP2 paper...")
    print("=" * 50)

    # Quick figures (no model training needed)
    fig2_rolling_sharpe_bars()
    fig3_max_drawdown_comparison()
    fig4_model_comparison_summary()

    # Figure 1 requires training all models (~5 min)
    print("\n  Figure 1 requires training all models (~5 min).")
    response = input("  Generate Figure 1? (y/n): ").strip().lower()
    if response == "y":
        config, pipeline, processed_data, feature_columns = setup()
        fig1_equity_curves(config, pipeline, processed_data, feature_columns)

    print("\n  ✓ All figures saved to figures/ directory.")
    print("  Insert these into your Word document.")
