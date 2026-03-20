#!/usr/bin/env python3
"""
Iteration V4 — Inverse Volatility Scaling (Negative Result, Reverted in V5)

Expected: Sharpe ≈ 0.245, Total Return ≈ 3.41%

Changes from V3:
  - Added inverse vol-scaling: target 15% annualized vol,
    scale positions inversely with realized vol.
  - This was too conservative for high-beta tech stocks and
    was reverted in V5.

All other parameters identical to V3.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import numpy as np
import torch

from maml_trading.config import MAMLConfig
from maml_trading.data_pipeline import DataPipeline
from maml_trading.task_generator import MarketRegimeTask, MarketRegimeTaskGenerator
from maml_trading.maml_engine import MAMLTradingEngine
from maml_trading.backtester import ExecutionSimulator
from maml_trading.metrics import PerformanceMetrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class V4DataPipeline(DataPipeline):
    def _generate_labels(self, df):
        fwd_ret = df["Close"].pct_change().shift(-1)
        roll_median = fwd_ret.rolling(window=60, min_periods=20).median()
        roll_median = roll_median.fillna(fwd_ret.median())
        df["Target"] = (fwd_ret > roll_median).astype(np.int64)
        return df

    def get_feature_columns(self):
        curated = ["SMA_Cross", "RSI", "Mom_5d", "Mom_20d", "BB_width",
                    "BB_position", "Returns", "Volatility_20d", "Vol_Ratio",
                    "Price_Zscore", "Volume_Ratio"]
        sample = next(iter(self.processed_data.values()))
        return [c for c in curated if c in sample.columns]


class V4Backtester:
    """V4: same as V3 but with inverse vol-scaling."""

    def __init__(self, config, processed_data, feature_columns):
        self.cfg = config
        self.processed_data = processed_data
        self.feature_columns = feature_columns
        self.exec_sim = ExecutionSimulator(config)
        self.fold_results = []

    def _signal_to_position(self, prediction, probs):
        p_long = float(probs[1])
        if p_long >= 0.55:
            return 1.0
        if p_long >= 0.45:
            return 0.6
        if p_long >= 0.35:
            return 0.3
        return 0.0

    def _compute_realized_vol(self, returns, idx, lookback):
        start = max(0, idx - lookback)
        window = returns[start:idx]
        if len(window) < 2:
            return 0.01
        return float(np.std(window, ddof=1))

    def _compute_splits(self, total_len):
        train_end = int(total_len * self.cfg.train_ratio)
        val_end = train_end + int(total_len * self.cfg.val_ratio)
        embargo_end = val_end + self.cfg.embargo_days
        test_end = total_len
        if embargo_end >= test_end:
            embargo_end = val_end
        return [(0, train_end, val_end, embargo_end, test_end)]

    def run(self, meta_train_epochs=50, verbose=True):
        min_len = min(len(df) for df in self.processed_data.values())
        splits = self._compute_splits(min_len)
        train_start, train_end, val_end, embargo_end, test_end = splits[0]

        train_data = {t: df.iloc[train_start:train_end].copy()
                      for t, df in self.processed_data.items()}
        task_gen = MarketRegimeTaskGenerator(
            processed_data=train_data,
            feature_columns=self.feature_columns, config=self.cfg)

        engine = MAMLTradingEngine(config=self.cfg, task_generator=task_gen)
        engine.meta_train(n_epochs=meta_train_epochs, verbose=verbose)

        ticker_daily_returns = {}

        for ticker, df in self.processed_data.items():
            if verbose:
                print(f"\n{'─'*50}\n  Testing: {ticker}\n{'─'*50}")

            features = df[self.feature_columns].values.astype(np.float32)
            labels = df["Target"].values.astype(np.int64)
            raw_returns = (df["Raw_Returns"].values if "Raw_Returns" in df.columns
                          else np.zeros(len(df)))

            ticker_returns, active_returns = [], []
            prev_position = 0.0
            n_trades = 0
            seq_len, k_shot = self.cfg.sequence_length, self.cfg.k_shot
            peak_equity, current_equity = 1.0, 1.0
            risk_off, risk_off_cooldown = False, 0

            for t in range(embargo_end + seq_len + k_shot, test_end):
                if t >= len(features):
                    break

                support_start = t - k_shot - seq_len
                support_xs, support_ys = [], []
                for i in range(k_shot):
                    s = support_start + i
                    e = s + seq_len
                    if e <= len(features):
                        support_xs.append(features[s:e])
                        support_ys.append(labels[e - 1])

                if len(support_xs) < 2:
                    ticker_returns.append(0.0)
                    continue

                query_x = features[t - seq_len:t]
                task = MarketRegimeTask(
                    support_x=torch.tensor(np.array(support_xs)),
                    support_y=torch.tensor(np.array(support_ys)),
                    query_x=torch.tensor(np.array([query_x])),
                    query_y=torch.tensor(np.array([labels[t - 1]])),
                    ticker=ticker,
                )

                preds, probs = engine.adapt_and_predict(task)
                raw_position = self._signal_to_position(preds[0], probs[0])

                # V4: inverse vol-scaling (the key difference)
                realized_vol = self._compute_realized_vol(
                    raw_returns, t, self.cfg.volatility_lookback)
                target_daily_vol = 0.15 / np.sqrt(252)
                vol_scalar = min(target_daily_vol / max(realized_vol, 0.001), 1.5)
                raw_position *= vol_scalar

                drawdown = (current_equity - peak_equity) / peak_equity
                if drawdown < -0.20:
                    risk_off = True
                    risk_off_cooldown = 5
                if risk_off:
                    raw_position *= 0.3
                    risk_off_cooldown -= 1
                    if risk_off_cooldown <= 0:
                        risk_off = False

                position = prev_position + 0.7 * (raw_position - prev_position)
                if abs(position) < 0.03:
                    position = 0.0

                daily_raw_return = raw_returns[t] * position if t < len(raw_returns) else 0.0
                trade_fraction = abs(position - prev_position)
                net_return = self.exec_sim.apply_costs(
                    daily_raw_return, trade_fraction, realized_vol)

                current_equity *= (1 + net_return)
                peak_equity = max(peak_equity, current_equity)
                ticker_returns.append(net_return)
                if abs(position) > 0.05:
                    active_returns.append(net_return)
                if trade_fraction > 0.05:
                    n_trades += 1
                prev_position = position

            ticker_daily_returns[ticker] = np.array(ticker_returns)

            if verbose and ticker_returns:
                fm = PerformanceMetrics.compute_all(np.array(ticker_returns))
                n_active, n_days = len(active_returns), len(ticker_returns)
                ah = np.mean(np.array(active_returns) > 0) if active_returns else 0.0
                print(f"  [{ticker}] Sharpe: {fm['sharpe_ratio']:.3f}  "
                      f"MaxDD: {fm['max_drawdown']:.2%}  Hit: {ah:.1%}  "
                      f"Active: {n_active}/{n_days} ({n_active/n_days*100:.0f}%)  "
                      f"Trades: {n_trades}")
                self.fold_results.append({"ticker": ticker, "n_trades": n_trades,
                    "active_hit_rate": float(ah), "active_days": n_active,
                    "total_days": n_days, **fm})

        if not ticker_daily_returns:
            return {}

        min_days = min(len(r) for r in ticker_daily_returns.values())
        aligned = np.array([r[:min_days] for r in ticker_daily_returns.values()])
        portfolio_returns = aligned.mean(axis=0)
        results = PerformanceMetrics.compute_all(portfolio_returns)

        total_active = sum(f.get("active_days", 0) for f in self.fold_results)
        total_days = sum(f.get("total_days", 0) for f in self.fold_results)
        results["active_hit_rate"] = (sum(f.get("active_hit_rate", 0) * f.get("active_days", 0)
            for f in self.fold_results) / max(total_active, 1))
        results["active_days"] = total_active
        results["total_days"] = len(portfolio_returns)
        results["total_trades"] = sum(f.get("n_trades", 0) for f in self.fold_results)
        results["active_pct"] = total_active / max(total_days, 1)

        if verbose:
            print("\n" + "=" * 50)
            print("  V4 AGGREGATE RESULTS")
            PerformanceMetrics.print_report(results)
        return results


def main():
    config = MAMLConfig(
        tickers=["AAPL", "MSFT", "GOOGL", "JPM", "NVDA"],
        start_date="2015-01-01", end_date="2024-12-31",
        meta_epochs=150, inner_steps=3, inner_lr=0.01, meta_lr=0.001,
        tasks_per_batch=8, k_shot=20, query_size=20,
        hidden_dim=64, num_layers=1, dropout=0.1, output_dim=2,
        regime_window_min=60, regime_window_max=90, sequence_length=30,
        train_ratio=0.60, val_ratio=0.15, test_ratio=0.25, embargo_days=10,
        transaction_cost_bps=15.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    set_seed(config.seed)
    print(f"[V4] Device: {config.device}")

    pipeline = V4DataPipeline(config)
    processed_data = pipeline.run()
    feature_columns = pipeline.get_feature_columns()
    config.input_dim = len(feature_columns)
    print(f"[V4] Feature dimension: {config.input_dim}")

    backtester = V4Backtester(config, processed_data, feature_columns)
    results = backtester.run(meta_train_epochs=config.meta_epochs, verbose=True)

    if results:
        print("\n" + "=" * 60)
        print("  V4 FINAL RESULTS")
        print("=" * 60)
        PerformanceMetrics.print_report(results)
    return results


if __name__ == "__main__":
    main()
