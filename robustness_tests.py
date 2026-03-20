#!/usr/bin/env python3
"""
Robustness Tests for the MAML Trading Framework.

Addresses common backtest validity concerns:
  1. Transaction cost sensitivity analysis
  2. Multi-period walk-forward (rolling folds)
  3. Expanded universe (survivorship bias mitigation)
  4. Hold-out year test (true out-of-sample)

Usage:
    python robustness_tests.py              # run all tests
    python robustness_tests.py --cost       # cost sensitivity only
    python robustness_tests.py --folds      # multi-fold only
    python robustness_tests.py --holdout    # hold-out year only
    python robustness_tests.py --universe   # expanded universe only
"""

import argparse
import copy
import random
import numpy as np
import torch

from maml_trading.config import MAMLConfig
from maml_trading.data_pipeline import DataPipeline
from maml_trading.backtester import WalkForwardBacktester
from maml_trading.metrics import PerformanceMetrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def base_config() -> MAMLConfig:
    return MAMLConfig(
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


# ════════════════════════════════════════════════════════════════════
# Test 1: Transaction Cost Sensitivity
# ════════════════════════════════════════════════════════════════════

def test_cost_sensitivity():
    """
    Run the backtest at multiple transaction cost levels to show
    how robust the strategy is to execution cost assumptions.
    """
    print("\n" + "=" * 72)
    print("  TEST 1: TRANSACTION COST SENSITIVITY ANALYSIS")
    print("=" * 72)

    cost_levels = [5, 10, 15, 25, 50, 100]  # bps round-trip
    results_table = []

    # Build data once, reuse across cost levels
    cfg = base_config()
    set_seed(cfg.seed)
    pipeline = DataPipeline(cfg)
    processed_data = pipeline.run()
    feature_columns = pipeline.get_feature_columns()
    cfg.input_dim = len(feature_columns)

    for bps in cost_levels:
        set_seed(cfg.seed)
        test_cfg = copy.deepcopy(cfg)
        test_cfg.transaction_cost_bps = float(bps)

        print(f"\n  Running with transaction_cost_bps = {bps}...")
        bt = WalkForwardBacktester(test_cfg, processed_data, feature_columns)
        metrics = bt.run(meta_train_epochs=test_cfg.meta_epochs, verbose=False)

        if metrics:
            results_table.append({
                "cost_bps": bps,
                "sharpe": metrics["sharpe_ratio"],
                "sortino": metrics["sortino_ratio"],
                "max_dd": metrics["max_drawdown"],
                "total_ret": metrics["total_return"],
                "ann_ret": metrics["annualized_return"],
            })

    # Print summary table
    print("\n" + "─" * 72)
    print(f"  {'Cost (bps)':>10s} {'Sharpe':>8s} {'Sortino':>8s} "
          f"{'MaxDD':>8s} {'Return':>9s} {'Ann.Ret':>8s}")
    print("  " + "─" * 62)
    for r in results_table:
        print(f"  {r['cost_bps']:>10d} {r['sharpe']:>8.3f} {r['sortino']:>8.3f} "
              f"{r['max_dd']:>8.2%} {r['total_ret']:>9.2%} {r['ann_ret']:>8.2%}")
    print("=" * 72)

    return results_table


# ════════════════════════════════════════════════════════════════════
# Test 2: Multi-Period Walk-Forward (Rolling Folds)
# ════════════════════════════════════════════════════════════════════

def test_rolling_folds():
    """
    Train on expanding windows and test on successive years:
      Fold 1: Train 2015-2019, Test 2020
      Fold 2: Train 2015-2020, Test 2021
      Fold 3: Train 2015-2021, Test 2022
      Fold 4: Train 2015-2022, Test 2023
      Fold 5: Train 2015-2023, Test 2024

    Reports per-fold and average Sharpe to show consistency.
    """
    print("\n" + "=" * 72)
    print("  TEST 2: MULTI-PERIOD ROLLING WALK-FORWARD")
    print("=" * 72)

    folds = [
        ("2015-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2015-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ]

    fold_results = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"\n  Fold {i+1}: Train {train_start}→{train_end}, "
              f"Test {test_start}→{test_end}")

        cfg = base_config()
        cfg.start_date = train_start
        cfg.end_date = test_end
        set_seed(cfg.seed)

        # Compute train/test ratio based on dates
        import pandas as pd
        total_days = len(pd.bdate_range(train_start, test_end))
        train_days = len(pd.bdate_range(train_start, train_end))
        cfg.train_ratio = train_days / total_days
        cfg.val_ratio = 0.02  # minimal validation
        cfg.test_ratio = 1.0 - cfg.train_ratio - cfg.val_ratio

        pipeline = DataPipeline(cfg)
        processed_data = pipeline.run()
        feature_columns = pipeline.get_feature_columns()
        cfg.input_dim = len(feature_columns)

        bt = WalkForwardBacktester(cfg, processed_data, feature_columns)
        metrics = bt.run(meta_train_epochs=cfg.meta_epochs, verbose=False)

        if metrics:
            # Get benchmark returns for this fold
            benchmarks = pipeline.get_benchmark_returns(
                start_idx=bt.test_start_idx, end_idx=bt.test_end_idx,
            )
            bm_metrics = {}
            for bm_name, bm_ret in benchmarks.items():
                if len(bm_ret) > 0:
                    bm_metrics[bm_name] = PerformanceMetrics.compute_all(bm_ret)

            fold_entry = {
                "fold": i + 1,
                "test_year": test_start[:4],
                "sharpe": metrics["sharpe_ratio"],
                "sortino": metrics["sortino_ratio"],
                "max_dd": metrics["max_drawdown"],
                "total_ret": metrics["total_return"],
            }
            # Add benchmark Sharpe and return for each benchmark
            for bm_name, bm_m in bm_metrics.items():
                fold_entry[f"{bm_name}_sharpe"] = bm_m["sharpe_ratio"]
                fold_entry[f"{bm_name}_ret"] = bm_m["total_return"]
                fold_entry[f"{bm_name}_dd"] = bm_m["max_drawdown"]

            fold_results.append(fold_entry)

            # Print fold result with benchmarks
            print(f"    MAML   → Sharpe: {metrics['sharpe_ratio']:.3f}  "
                  f"Return: {metrics['total_return']:.2%}  "
                  f"MaxDD: {metrics['max_drawdown']:.2%}")
            for bm_name, bm_m in bm_metrics.items():
                label = bm_name[:10]
                print(f"    {label:<10s}→ Sharpe: {bm_m['sharpe_ratio']:.3f}  "
                      f"Return: {bm_m['total_return']:.2%}  "
                      f"MaxDD: {bm_m['max_drawdown']:.2%}")

    # ── Summary table ───────────────────────────────────────────────
    # Determine which benchmarks are available
    bm_names = []
    if fold_results:
        for key in fold_results[0]:
            if key.endswith("_sharpe"):
                bm_names.append(key.replace("_sharpe", ""))

    print("\n" + "─" * 100)
    header = f"  {'Fold':>4s} {'Year':>4s} │ {'Sharpe':>7s} {'Return':>8s} {'MaxDD':>8s}"
    for bm in bm_names:
        label = bm[:8]
        header += f" │ {label+' SR':>10s} {label+' Ret':>10s} {label+' DD':>10s}"
    print(header)
    print("  " + "─" * 96)

    for r in fold_results:
        row = (f"  {r['fold']:>4d} {r['test_year']:>4s} │ "
               f"{r['sharpe']:>7.3f} {r['total_ret']:>8.2%} {r['max_dd']:>8.2%}")
        for bm in bm_names:
            bm_sr = r.get(f"{bm}_sharpe", 0)
            bm_rt = r.get(f"{bm}_ret", 0)
            bm_dd = r.get(f"{bm}_dd", 0)
            row += f" │ {bm_sr:>10.3f} {bm_rt:>10.2%} {bm_dd:>10.2%}"
        print(row)

    if fold_results:
        avg_sharpe = np.mean([r["sharpe"] for r in fold_results])
        std_sharpe = np.std([r["sharpe"] for r in fold_results])
        print("  " + "─" * 96)
        avg_row = f"  {'AVG':>4s} {'':>4s} │ {avg_sharpe:>7.3f}±{std_sharpe:.2f}"
        for bm in bm_names:
            bm_avg = np.mean([r.get(f"{bm}_sharpe", 0) for r in fold_results])
            avg_row += f" │ {bm_avg:>10.3f}"
            avg_row += f" {'':>10s} {'':>10s}"
        print(avg_row)

        win_rate = np.mean([r["sharpe"] > 0 for r in fold_results])
        print(f"  Profitable folds: {win_rate:.0%} "
              f"({sum(r['sharpe'] > 0 for r in fold_results)}/{len(fold_results)})")

        # Beat benchmark rate
        for bm in bm_names:
            beat = sum(
                r["sharpe"] > r.get(f"{bm}_sharpe", 0)
                for r in fold_results
            )
            print(f"  Beat {bm}: {beat}/{len(fold_results)} folds")

    print("=" * 100)

    return fold_results


# ════════════════════════════════════════════════════════════════════
# Test 3: Hold-Out Year (True Out-of-Sample)
# ════════════════════════════════════════════════════════════════════

def test_holdout_year():
    """
    Train/tune on 2015-2023, test on 2024 only.
    This is the purest out-of-sample test — 2024 data was never
    used for any hyperparameter decisions.
    """
    print("\n" + "=" * 72)
    print("  TEST 3: HOLD-OUT YEAR (2024) — TRUE OUT-OF-SAMPLE")
    print("=" * 72)

    cfg = base_config()
    cfg.start_date = "2015-01-01"
    cfg.end_date = "2024-12-31"
    # ~90% train (2015-2023), ~2% val, ~8% test (2024)
    cfg.train_ratio = 0.82
    cfg.val_ratio = 0.05
    cfg.test_ratio = 0.13
    set_seed(cfg.seed)

    pipeline = DataPipeline(cfg)
    processed_data = pipeline.run()
    feature_columns = pipeline.get_feature_columns()
    cfg.input_dim = len(feature_columns)

    bt = WalkForwardBacktester(cfg, processed_data, feature_columns)
    metrics = bt.run(meta_train_epochs=cfg.meta_epochs, verbose=True)

    if metrics:
        print("\n" + "=" * 60)
        print("  HOLD-OUT 2024 RESULTS")
        print("=" * 60)
        PerformanceMetrics.print_report(metrics)

        benchmarks = pipeline.get_benchmark_returns(
            start_idx=bt.test_start_idx, end_idx=bt.test_end_idx,
        )
        if benchmarks:
            PerformanceMetrics.print_benchmark_comparison(metrics, benchmarks)

    return metrics


# ════════════════════════════════════════════════════════════════════
# Test 4: Expanded Universe (Survivorship Bias Mitigation)
# ════════════════════════════════════════════════════════════════════

def test_expanded_universe():
    """
    Test with 20 stocks including underperformers and sector diversity.
    Includes stocks that struggled (GE, INTC, BA) to mitigate
    survivorship bias.
    """
    print("\n" + "=" * 72)
    print("  TEST 4: EXPANDED 20-STOCK UNIVERSE")
    print("=" * 72)

    cfg = base_config()
    cfg.tickers = [
        # Original 6
        "AAPL", "MSFT", "GOOGL", "JPM", "NVDA", "TEAM",
        # Additional large-caps (diverse sectors)
        "AMZN", "META", "V", "UNH",
        "PG", "HD", "MA", "XOM",
        # Underperformers / troubled stocks (survivorship bias test)
        "GE",    # struggled 2015-2020, restructured
        "INTC",  # lost market share to AMD/NVDA
        "BA",    # 737 MAX crisis, COVID hit
        "PFE",   # COVID spike then decline
        "DIS",   # streaming losses
        "BAC",   # banking sector volatility
    ]
    set_seed(cfg.seed)

    pipeline = DataPipeline(cfg)
    processed_data = pipeline.run()
    feature_columns = pipeline.get_feature_columns()
    cfg.input_dim = len(feature_columns)

    print(f"  Universe: {len(processed_data)} stocks")
    print(f"  Features: {len(feature_columns)}")

    bt = WalkForwardBacktester(cfg, processed_data, feature_columns)
    metrics = bt.run(meta_train_epochs=cfg.meta_epochs, verbose=True)

    if metrics:
        print("\n" + "=" * 60)
        print("  EXPANDED UNIVERSE RESULTS")
        print("=" * 60)
        PerformanceMetrics.print_report(metrics)

        benchmarks = pipeline.get_benchmark_returns(
            start_idx=bt.test_start_idx, end_idx=bt.test_end_idx,
        )
        if benchmarks:
            PerformanceMetrics.print_benchmark_comparison(metrics, benchmarks)

    return metrics


# ════════════════════════════════════════════════════════════════════
# Test 5: True Out-of-Sample (2025 Q1)
# ════════════════════════════════════════════════════════════════════

def test_true_oos():
    """
    The purest out-of-sample test possible.

    Train on 2015-2024 (including the regime the model was tuned on),
    test on 2025 Q1 — data that was NEVER seen or optimized against
    during any iteration of development.

    By including 2024 in training, the model sees the late-2024 regime
    shift (rate uncertainty, tariff fears) which is more representative
    of the 2025 Q1 environment.
    """
    print("\n" + "=" * 72)
    print("  TEST 5: TRUE OUT-OF-SAMPLE (2025 Q1)")
    print("  Train: 2015-2024 | Test: Jan-Apr 2025")
    print("=" * 72)

    cfg = base_config()
    cfg.start_date = "2015-01-01"
    cfg.end_date = "2025-04-11"

    # ~95% train (2015-2024), ~1% val, ~4% test (2025)
    cfg.train_ratio = 0.93
    cfg.val_ratio = 0.02
    cfg.test_ratio = 0.05
    cfg.embargo_days = 5
    set_seed(cfg.seed)

    pipeline = DataPipeline(cfg)
    processed_data = pipeline.run()
    feature_columns = pipeline.get_feature_columns()
    cfg.input_dim = len(feature_columns)

    bt = WalkForwardBacktester(cfg, processed_data, feature_columns)
    metrics = bt.run(meta_train_epochs=cfg.meta_epochs, verbose=True)

    if metrics:
        print("\n" + "=" * 60)
        print("  TRUE OUT-OF-SAMPLE (2025) RESULTS")
        print("=" * 60)
        PerformanceMetrics.print_report(metrics)

        benchmarks = pipeline.get_benchmark_returns(
            start_idx=bt.test_start_idx, end_idx=bt.test_end_idx,
        )
        if benchmarks:
            PerformanceMetrics.print_benchmark_comparison(metrics, benchmarks)

    return metrics


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAML Trading Robustness Tests")
    parser.add_argument("--cost", action="store_true", help="Transaction cost sensitivity")
    parser.add_argument("--folds", action="store_true", help="Multi-period walk-forward")
    parser.add_argument("--holdout", action="store_true", help="Hold-out year (2024)")
    parser.add_argument("--universe", action="store_true", help="Expanded 20-stock universe")
    parser.add_argument("--oos", action="store_true", help="True out-of-sample (2025 Q1)")
    args = parser.parse_args()

    run_all = not (args.cost or args.folds or args.holdout or args.universe or args.oos)

    if args.cost or run_all:
        test_cost_sensitivity()
    if args.folds or run_all:
        test_rolling_folds()
    if args.holdout or run_all:
        test_holdout_year()
    if args.universe or run_all:
        test_expanded_universe()
    if args.oos or run_all:
        test_true_oos()

    print("\n  All requested robustness tests complete.")
