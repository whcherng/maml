#!/usr/bin/env python3
"""
Iteration V2 — Binary Labels + Small Model

Expected: Sharpe ≈ -0.431, Total Return ≈ -29.87%

Changes from V1:
  - Binary labels (flat/long) via rolling median
  - Small model: hidden_dim=64, num_layers=1, output_dim=2
  - inner_lr=0.01, meta_lr=0.001, meta_epochs=150
  - Inference uses same inner_steps (not 2x)
  - Still per-ticker training, sequential aggregation
  - Still all 29 features
  - Still binary position mapping with 0.55 threshold
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
from maml_trading.metrics import PerformanceMetrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class V2DataPipeline(DataPipeline):
    """V2: binary labels via rolling median, all features."""

    def _generate_labels(self, df):
        import pandas as pd
        fwd_ret = df["Close"].pct_change().shift(-1)
        roll_median = fwd_ret.rolling(window=60, min_periods=20).median()
        global_median = fwd_ret.median()
        roll_median = roll_median.fillna(global_median)
        df["Target"] = (fwd_ret > roll_median).astype(np.int64)
        return df

    def get_feature_columns(self):
        """V2 still uses ALL numeric features (29 total)."""
        exclude = {"Target", "Raw_Returns"}
        sample = next(iter(self.processed_data.values()))
        return [c for c in sample.select_dtypes(include=[np.number]).columns
                if c not in exclude]


class V2Backtester:
    """V2: per-ticker training, sequential aggregation, binary threshold."""

    def __init__(self, config, processed_data, feature_columns):
        self.cfg = config
        self.processed_data = processed_data
        self.feature_columns = feature_columns
        from maml_trading.backtester import ExecutionSimulator
        self.exec_sim = ExecutionSimulator(config)

    def _signal_to_position(self, prediction, confidence):
        """V2: binary mapping, 0.55 threshold."""
        if prediction == 0:
            return 0.0
        if confidence < 0.55:
            return 0.0
        return min(confidence, 1.0)

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
        all_test_returns = []

        for ticker, df in self.processed_data.items():
            if verbose:
                print(f"\n{'─'*50}\nBacktesting: {ticker}\n{'─'*50}")

            total_len = len(df)
            splits = self._compute_splits(total_len)
            train_start, train_end, val_end, embargo_end, test_end = splits[0]

            train_data = {ticker: df.iloc[train_start:train_end].copy()}
            task_gen = MarketRegimeTaskGenerator(
                processed_data=train_data,
                feature_columns=self.feature_columns,
                config=self.cfg,
            )

            engine = MAMLTradingEngine(config=self.cfg, task_generator=task_gen)
            engine.meta_train(n_epochs=meta_train_epochs, verbose=verbose)

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
                confidence = float(probs[0, preds[0]])
                raw_position = self._signal_to_position(preds[0], confidence)

                drawdown = (current_equity - peak_equity) / peak_equity
                if drawdown < -0.12:
                    risk_off = True
                    risk_off_cooldown = 10
                if risk_off:
                    raw_position = 0.0
                    risk_off_cooldown -= 1
                    if risk_off_cooldown <= 0:
                        risk_off = False

                position = prev_position + 0.6 * (raw_position - prev_position)
                if abs(position) < 0.05:
                    position = 0.0

                daily_raw_return = raw_returns[t] * position if t < len(raw_returns) else 0.0
                trade_fraction = abs(position - prev_position)
                realized_vol = self._compute_realized_vol(raw_returns, t, self.cfg.volatility_lookback)
                net_return = self.exec_sim.apply_costs(daily_raw_return, trade_fraction, realized_vol)

                current_equity *= (1 + net_return)
                peak_equity = max(peak_equity, current_equity)
                ticker_returns.append(net_return)
                if abs(position) > 0.05:
                    active_returns.append(net_return)
                if trade_fraction > 0.05:
                    n_trades += 1
                prev_position = position

            all_test_returns.extend(ticker_returns)

            if verbose and ticker_returns:
                fm = PerformanceMetrics.compute_all(np.array(ticker_returns))
                n_active, n_days = len(active_returns), len(ticker_returns)
                ah = np.mean(np.array(active_returns) > 0) if active_returns else 0.0
                print(f"  [{ticker}] Sharpe: {fm['sharpe_ratio']:.3f}  "
                      f"MaxDD: {fm['max_drawdown']:.2%}  Hit: {ah:.1%}  "
                      f"Active: {n_active}/{n_days} ({n_active/n_days*100:.0f}%)  "
                      f"Trades: {n_trades}")

        if not all_test_returns:
            return {}

        all_returns = np.array(all_test_returns)
        results = PerformanceMetrics.compute_all(all_returns)
        if verbose:
            print("\n" + "=" * 50)
            print("  V2 AGGREGATE RESULTS")
            PerformanceMetrics.print_report(results)
        return results


def main():
    config = MAMLConfig(
        tickers=["AAPL", "MSFT", "GOOGL", "JPM", "NVDA"],
        start_date="2015-01-01",
        end_date="2024-12-31",
        meta_epochs=150,
        inner_steps=5,
        inner_lr=0.01,
        meta_lr=0.001,
        tasks_per_batch=8,
        k_shot=20,
        query_size=20,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        output_dim=2,
        regime_window_min=60,
        regime_window_max=90,
        sequence_length=30,
        train_ratio=0.60,
        val_ratio=0.15,
        test_ratio=0.25,
        embargo_days=10,
        transaction_cost_bps=15.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    set_seed(config.seed)
    print(f"[V2] Device: {config.device}")

    pipeline = V2DataPipeline(config)
    processed_data = pipeline.run()
    feature_columns = pipeline.get_feature_columns()
    config.input_dim = len(feature_columns)
    print(f"[V2] Feature dimension: {config.input_dim}")

    backtester = V2Backtester(config, processed_data, feature_columns)
    results = backtester.run(meta_train_epochs=config.meta_epochs, verbose=True)

    if results:
        print("\n" + "=" * 60)
        print("  V2 FINAL RESULTS")
        print("=" * 60)
        PerformanceMetrics.print_report(results)
    return results


if __name__ == "__main__":
    main()
