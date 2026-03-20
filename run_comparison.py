#!/usr/bin/env python3
"""
Model Comparison: MAML vs 3 Baselines

Runs all 4 models on the same data with the same evaluation protocol:
  1. Static LSTM — trained once, no adaptation
  2. Online-Learning LSTM — retrained monthly on sliding window
  3. PPO RL Agent — reward-maximizing policy, no adaptation
  4. MAML (ours) — meta-learned initialization with fast adaptation

All models use:
  - Same features (11 curated technical indicators)
  - Same train/test split
  - Same execution cost model (15 bps + Almgren-Chriss)
  - Same position sizing (sigmoid mapping)
  - Same risk management (graduated drawdown, vol-scaling)

This ensures the ONLY difference is the learning/adaptation mechanism.

Usage:
    python run_comparison.py
"""

import random
import numpy as np
import torch
from typing import Dict

from maml_trading.config import MAMLConfig
from maml_trading.data_pipeline import DataPipeline
from maml_trading.backtester import WalkForwardBacktester, ExecutionSimulator
from maml_trading.task_generator import MarketRegimeTask
from maml_trading.metrics import PerformanceMetrics

from baselines.static_lstm import StaticLSTMBaseline
from baselines.online_lstm import OnlineLSTMBaseline
from baselines.ppo_agent import PPOBaseline


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def signal_to_position(probs: np.ndarray) -> float:
    """Same sigmoid mapping as MAML backtester for fair comparison."""
    p_long = float(probs[1]) if len(probs) > 1 else float(probs[0])
    steepness = 8.0
    center = 0.48
    position = 1.0 / (1.0 + np.exp(-steepness * (p_long - center)))
    return float(np.clip(position, 0.0, 1.0))


def vol_scale_position(position: float, realized_vol: float) -> float:
    """Same vol-scaling as MAML backtester."""
    vol_threshold = 0.025
    if realized_vol > vol_threshold:
        vol_scalar = vol_threshold / realized_vol
        return position * max(vol_scalar, 0.2)
    return position


def compute_realized_vol(returns: np.ndarray, idx: int, lookback: int = 20) -> float:
    start = max(0, idx - lookback)
    window = returns[start:idx]
    if len(window) < 2:
        return 0.01
    return float(np.std(window, ddof=1))


def run_baseline_backtest(
    model,
    model_type: str,  # "static", "online", "ppo"
    processed_data: Dict,
    feature_columns: list,
    config: MAMLConfig,
    test_start: int,
    test_end: int,
) -> Dict[str, float]:
    """
    Run backtest for a baseline model using the same logic as MAML backtester.
    """
    exec_sim = ExecutionSimulator(config)
    seq_len = config.sequence_length
    ticker_daily_returns = {}

    for ticker, df in processed_data.items():
        features = df[feature_columns].values.astype(np.float32)
        labels = df["Target"].values.astype(np.int64)
        raw_returns = (
            df["Raw_Returns"].values if "Raw_Returns" in df.columns
            else np.zeros(len(df))
        )

        ticker_returns = []
        prev_position = 0.0
        peak_equity = 1.0
        current_equity = 1.0
        n_trades = 0

        ticker_test_end = min(test_end, len(features))

        for t in range(test_start + seq_len, ticker_test_end):
            if t >= len(features):
                break

            # Online LSTM: check if we should retrain
            if model_type == "online" and model.should_retrain(t, test_start):
                model.retrain_on_recent(features, labels, t)

            # Build input sequence
            x = torch.tensor(features[t - seq_len:t], dtype=torch.float32)

            # Get position from model
            if model_type == "ppo":
                raw_position = model.get_position(x)
            else:
                pred, probs = model.predict(x)
                raw_position = signal_to_position(probs)

            # Vol-scaling
            realized_vol = compute_realized_vol(raw_returns, t)
            raw_position = vol_scale_position(raw_position, realized_vol)

            # Graduated risk management
            drawdown = (current_equity - peak_equity) / peak_equity
            if drawdown < -0.10:
                dd_severity = min(abs(drawdown) / 0.20, 1.0)
                raw_position *= (1.0 - 0.7 * dd_severity)

            # Asymmetric smoothing
            if raw_position < prev_position:
                alpha = 0.85
            else:
                alpha = 0.6
            position = prev_position + alpha * (raw_position - prev_position)

            if abs(position) < 0.03:
                position = 0.0
            if abs(position - prev_position) < 0.08:
                position = prev_position

            # Compute return
            daily_raw_return = raw_returns[t] * position if t < len(raw_returns) else 0.0
            trade_fraction = abs(position - prev_position)
            net_return = exec_sim.apply_costs(daily_raw_return, trade_fraction, realized_vol)

            current_equity *= (1 + net_return)
            peak_equity = max(peak_equity, current_equity)
            ticker_returns.append(net_return)
            if trade_fraction > 0.05:
                n_trades += 1
            prev_position = position

        ticker_daily_returns[ticker] = np.array(ticker_returns)

    # Portfolio aggregation (same as MAML)
    if not ticker_daily_returns:
        return {}

    max_days = max(len(r) for r in ticker_daily_returns.values())
    padded = np.zeros((len(ticker_daily_returns), max_days))
    mask = np.zeros((len(ticker_daily_returns), max_days))
    for i, (ticker, r) in enumerate(ticker_daily_returns.items()):
        padded[i, :len(r)] = r
        mask[i, :len(r)] = 1.0
    active_count = mask.sum(axis=0).clip(min=1)
    portfolio_returns = padded.sum(axis=0) / active_count

    return {**PerformanceMetrics.compute_all(portfolio_returns), "daily_returns": portfolio_returns}


def main():
    print("=" * 72)
    print("  MODEL COMPARISON: MAML vs BASELINES")
    print("  Same data, same costs, same risk management")
    print("=" * 72)

    # ── Configuration ───────────────────────────────────────────────
    config = MAMLConfig(
        tickers=["AAPL", "MSFT", "GOOGL", "JPM", "NVDA", "TEAM"],
        start_date="2015-01-01",
        end_date="2024-12-31",
        meta_epochs=200, inner_steps=3, inner_lr=0.01, meta_lr=0.001,
        tasks_per_batch=12, k_shot=30, query_size=20,
        hidden_dim=64, num_layers=1, dropout=0.1, output_dim=2,
        regime_window_min=60, regime_window_max=90, sequence_length=30,
        train_ratio=0.60, val_ratio=0.15, test_ratio=0.25,
        embargo_days=10, transaction_cost_bps=15.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    set_seed(config.seed)

    # ── Data Pipeline ───────────────────────────────────────────────
    pipeline = DataPipeline(config)
    processed_data = pipeline.run()
    feature_columns = pipeline.get_feature_columns()
    config.input_dim = len(feature_columns)

    # Compute split
    max_len = max(len(df) for df in processed_data.values())
    train_end = int(max_len * config.train_ratio)
    val_end = train_end + int(max_len * config.val_ratio)
    embargo_end = val_end + config.embargo_days
    test_end = max_len

    print(f"\n  Split: train={train_end}, embargo={embargo_end}, test={test_end}")
    print(f"  Tickers: {len(processed_data)}, Features: {len(feature_columns)}")

    all_results = {}

    # ── 1. Static LSTM ──────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  Training: STATIC LSTM (train once, no adaptation)")
    print("─" * 72)
    set_seed(config.seed)
    static = StaticLSTMBaseline(config, feature_columns)
    static.train_model(processed_data, train_end, epochs=50)
    static_results = run_baseline_backtest(
        static, "static", processed_data, feature_columns,
        config, embargo_end, test_end,
    )
    all_results["Static LSTM"] = static_results
    print(f"  Sharpe: {static_results.get('sharpe_ratio', 0):.3f}  "
          f"Return: {static_results.get('total_return', 0):.2%}  "
          f"MaxDD: {static_results.get('max_drawdown', 0):.2%}")

    # ── 2. Online-Learning LSTM ─────────────────────────────────────
    print("\n" + "─" * 72)
    print("  Training: ONLINE LSTM (retrain monthly on 1-year window)")
    print("─" * 72)
    set_seed(config.seed)
    online = OnlineLSTMBaseline(config, feature_columns)
    online.initial_train(processed_data, train_end, epochs=50)
    online_results = run_baseline_backtest(
        online, "online", processed_data, feature_columns,
        config, embargo_end, test_end,
    )
    all_results["Online LSTM"] = online_results
    print(f"  Sharpe: {online_results.get('sharpe_ratio', 0):.3f}  "
          f"Return: {online_results.get('total_return', 0):.2%}  "
          f"MaxDD: {online_results.get('max_drawdown', 0):.2%}")

    # ── 3. PPO Agent ────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  Training: PPO AGENT (reward maximization, no adaptation)")
    print("─" * 72)
    set_seed(config.seed)
    ppo = PPOBaseline(config, feature_columns)
    ppo.train_agent(processed_data, train_end, n_episodes=200)
    ppo_results = run_baseline_backtest(
        ppo, "ppo", processed_data, feature_columns,
        config, embargo_end, test_end,
    )
    all_results["PPO Agent"] = ppo_results
    print(f"  Sharpe: {ppo_results.get('sharpe_ratio', 0):.3f}  "
          f"Return: {ppo_results.get('total_return', 0):.2%}  "
          f"MaxDD: {ppo_results.get('max_drawdown', 0):.2%}")

    # ── 4. MAML (ours) ──────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  Training: MAML (meta-learning with fast adaptation)")
    print("─" * 72)
    set_seed(config.seed)
    backtester = WalkForwardBacktester(config, processed_data, feature_columns)
    maml_results = backtester.run(meta_train_epochs=config.meta_epochs, verbose=False)
    all_results["MAML (Ours)"] = maml_results
    print(f"  Sharpe: {maml_results.get('sharpe_ratio', 0):.3f}  "
          f"Return: {maml_results.get('total_return', 0):.2%}  "
          f"MaxDD: {maml_results.get('max_drawdown', 0):.2%}")

    # ── Summary Table ───────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  FINAL COMPARISON TABLE")
    print("=" * 72)
    print(f"  {'Model':<20s} {'Sharpe':>8s} {'Sortino':>8s} "
          f"{'MaxDD':>8s} {'Return':>9s} {'Ann.Ret':>8s}")
    print("  " + "─" * 62)

    for name, metrics in all_results.items():
        if not metrics:
            continue
        print(f"  {name:<20s} "
              f"{metrics.get('sharpe_ratio', 0):>8.3f} "
              f"{metrics.get('sortino_ratio', 0):>8.3f} "
              f"{metrics.get('max_drawdown', 0):>8.2%} "
              f"{metrics.get('total_return', 0):>9.2%} "
              f"{metrics.get('annualized_return', 0):>8.2%}")

    print("=" * 72)

    # ── Benchmark ───────────────────────────────────────────────────
    benchmarks = pipeline.get_benchmark_returns(
        start_idx=embargo_end, end_idx=test_end,
    )
    if benchmarks:
        print("\n  BENCHMARKS (Buy-and-Hold):")
        print(f"  {'Benchmark':<20s} {'Sharpe':>8s} {'Return':>9s} {'MaxDD':>8s}")
        print("  " + "─" * 50)
        for name, returns in benchmarks.items():
            if len(returns) == 0:
                continue
            bm = PerformanceMetrics.compute_all(returns)
            print(f"  {name:<20s} {bm['sharpe_ratio']:>8.3f} "
                  f"{bm['total_return']:>9.2%} {bm['max_drawdown']:>8.2%}")

    # ── Diebold-Mariano Test ────────────────────────────────────────
    # Tests whether MAML's forecast errors are significantly different
    # from each baseline's forecast errors (two-sided test).
    print("\n" + "=" * 72)
    print("  DIEBOLD-MARIANO STATISTICAL SIGNIFICANCE TEST")
    print("  H0: No significant difference in predictive accuracy")
    print("  H1: MAML produces significantly different (better) returns")
    print("=" * 72)
    diebold_mariano_test(all_results, "MAML (Ours)")

    print()
    return all_results


def diebold_mariano_test(
    all_results: Dict[str, Dict],
    reference_model: str,
) -> None:
    """
    Diebold-Mariano test comparing the reference model against each baseline.

    Uses squared return differences as the loss differential. A significant
    negative DM statistic means the reference model has smaller losses.

    The test accounts for autocorrelation in the loss differential series
    using Newey-West HAC standard errors.
    """
    from scipy import stats

    ref_metrics = all_results.get(reference_model)
    if not ref_metrics or "daily_returns" not in ref_metrics:
        # If daily returns weren't stored, skip
        print("  (Daily returns not available — skipping D-M test)")
        print("  Run with stored returns to enable statistical testing.")
        return

    ref_returns = ref_metrics["daily_returns"]

    print(f"\n  {'Baseline':<20s} {'DM Stat':>10s} {'p-value':>10s} {'Significant':>12s}")
    print("  " + "─" * 56)

    for name, metrics in all_results.items():
        if name == reference_model:
            continue
        if not metrics or "daily_returns" not in metrics:
            continue

        baseline_returns = metrics["daily_returns"]

        # Align lengths
        min_len = min(len(ref_returns), len(baseline_returns))
        r1 = ref_returns[:min_len]
        r2 = baseline_returns[:min_len]

        # Loss differential: negative values mean MAML is better
        # Using squared negative returns as loss (penalizes losses more)
        loss_ref = np.where(r1 < 0, r1 ** 2, 0)
        loss_base = np.where(r2 < 0, r2 ** 2, 0)
        d = loss_base - loss_ref  # positive = baseline has more loss

        # DM statistic with Newey-West variance (lag = sqrt(T))
        n = len(d)
        d_mean = np.mean(d)
        lag = int(np.sqrt(n))

        # Newey-West HAC variance
        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0
        for k in range(1, lag + 1):
            weight = 1 - k / (lag + 1)
            gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
            gamma_sum += 2 * weight * gamma_k
        var_d = (gamma_0 + gamma_sum) / n

        if var_d <= 0:
            print(f"  {name:<20s} {'N/A':>10s} {'N/A':>10s} {'—':>12s}")
            continue

        dm_stat = d_mean / np.sqrt(var_d)
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))  # two-sided

        sig = "Yes (p<0.05)" if p_value < 0.05 else "No"
        if p_value < 0.01:
            sig = "Yes (p<0.01)"

        print(f"  {name:<20s} {dm_stat:>10.3f} {p_value:>10.4f} {sig:>12s}")


if __name__ == "__main__":
    main()
