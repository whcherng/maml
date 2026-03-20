#!/usr/bin/env python3
"""
Iteration V1 — Baseline (Ternary Labels, Large Model, Per-Ticker Training)

Expected: Sharpe ≈ -0.972, Total Return ≈ -48.82%

Key characteristics:
  - Ternary labels (long/flat/short) via rolling tercile thresholds
  - Large model: hidden_dim=128, num_layers=2, output_dim=3
  - Per-ticker training (1 ticker at a time in backtester)
  - Binary position mapping with 0.55 confidence threshold
  - 2x inner steps at inference
  - Sequential return concatenation (not equal-weight portfolio)
  - Aggressive circuit breaker (-12% DD, 10-day cooldown, full flat)
  - All 29 features used
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import numpy as np
import torch
import torch.nn as nn

from maml_trading.config import MAMLConfig
from maml_trading.data_pipeline import DataPipeline
from maml_trading.task_generator import MarketRegimeTask, MarketRegimeTaskGenerator
from maml_trading.maml_engine import MAMLTradingEngine
from maml_trading.metrics import PerformanceMetrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── V1 Data Pipeline: ternary labels + all features ────────────────

class V1DataPipeline(DataPipeline):
    """Override label generation to use ternary (rolling tercile) labels."""

    def _generate_labels(self, df):
        import pandas as pd
        fwd_ret = df["Close"].pct_change().shift(-1)
        roll_upper = fwd_ret.rolling(window=60, min_periods=20).quantile(0.667)
        roll_lower = fwd_ret.rolling(window=60, min_periods=20).quantile(0.333)
        global_upper = fwd_ret.quantile(0.667)
        global_lower = fwd_ret.quantile(0.333)
        roll_upper = roll_upper.fillna(global_upper)
        roll_lower = roll_lower.fillna(global_lower)
        target = np.where(fwd_ret > roll_upper, 2,
                 np.where(fwd_ret < roll_lower, 0, 1))
        df["Target"] = pd.Series(target, index=df.index).astype(np.int64)
        return df

    def get_feature_columns(self):
        """V1 uses ALL numeric features (29 total)."""
        exclude = {"Target", "Raw_Returns"}
        sample = next(iter(self.processed_data.values()))
        return [c for c in sample.select_dtypes(include=[np.number]).columns
                if c not in exclude]


# ── V1 MAML Engine: 2x inner steps at inference ───────────────────

class V1MAMLEngine(MAMLTradingEngine):
    """Override inference to use 2x inner steps."""

    @torch.no_grad()
    def adapt_and_predict(self, task):
        with torch.enable_grad():
            sx = task.support_x.to(self.device)
            sy = task.support_y.to(self.device)
            qx = task.query_x.to(self.device)
            learner = self.maml.clone()
            inference_steps = self.cfg.inner_steps * 2  # V1: 2x inner steps
            for _ in range(inference_steps):
                logits = learner(sx)
                loss = self.criterion(logits, sy)
                learner.adapt(loss)
        learner.eval()
        q_logits = learner(qx)
        probs = torch.softmax(q_logits, dim=1).cpu().numpy()
        preds = q_logits.argmax(dim=1).cpu().numpy()
        return preds, probs


# ── V1 Backtester: per-ticker training, sequential aggregation ─────

class V1Backtester:
    """V1 backtester: trains per-ticker, sequential return concat."""

    def __init__(self, config, processed_data, feature_columns):
        self.cfg = config
        self.processed_data = processed_data
        self.feature_columns = feature_columns
        from maml_trading.backtester import ExecutionSimulator
        self.exec_sim = ExecutionSimulator(config)
        self.fold_results = []

    def _signal_to_position(self, prediction, confidence):
        """V1: ternary mapping with 0.55 threshold."""
        if prediction == 1:  # flat
            return 0.0
        min_confidence = 0.55
        if confidence < min_confidence:
            return 0.0
        if prediction == 2:  # long
            return min(confidence, 1.0)
        if prediction == 0:  # short
            return -min(confidence, 1.0)
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
        all_test_returns = []

        for ticker, df in self.processed_data.items():
            if verbose:
                print(f"\n{'─'*50}\nBacktesting: {ticker}\n{'─'*50}")

            total_len = len(df)
            splits = self._compute_splits(total_len)
            train_start, train_end, val_end, embargo_end, test_end = splits[0]

            # Per-ticker task generator
            train_data = {ticker: df.iloc[train_start:train_end].copy()}
            task_gen = MarketRegimeTaskGenerator(
                processed_data=train_data,
                feature_columns=self.feature_columns,
                config=self.cfg,
            )

            # Per-ticker MAML engine (V1 with 2x inference steps)
            engine = V1MAMLEngine(config=self.cfg, task_generator=task_gen)
            engine.meta_train(n_epochs=meta_train_epochs, verbose=verbose)

            features = df[self.feature_columns].values.astype(np.float32)
            labels = df["Target"].values.astype(np.int64)
            raw_returns = (df["Raw_Returns"].values if "Raw_Returns" in df.columns
                          else np.zeros(len(df)))

            ticker_returns = []
            active_returns = []
            prev_position = 0.0
            n_trades = 0
            seq_len = self.cfg.sequence_length
            k_shot = self.cfg.k_shot

            peak_equity = 1.0
            current_equity = 1.0
            risk_off = False
            risk_off_cooldown = 0

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

                query_start = t - seq_len
                query_x = features[query_start:t]

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

                # V1 risk management: aggressive
                drawdown = (current_equity - peak_equity) / peak_equity
                if drawdown < -0.12:
                    risk_off = True
                    risk_off_cooldown = 10
                if risk_off:
                    raw_position = 0.0
                    risk_off_cooldown -= 1
                    if risk_off_cooldown <= 0:
                        risk_off = False

                # V1 position smoothing
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
                fold_metrics = PerformanceMetrics.compute_all(np.array(ticker_returns))
                n_active = len(active_returns)
                n_days = len(ticker_returns)
                active_hit = np.mean(np.array(active_returns) > 0) if active_returns else 0.0
                print(f"  [{ticker}] Sharpe: {fold_metrics['sharpe_ratio']:.3f}  "
                      f"MaxDD: {fold_metrics['max_drawdown']:.2%}  "
                      f"Hit: {active_hit:.1%}  "
                      f"Active: {n_active}/{n_days} ({n_active/n_days*100:.0f}%)  "
                      f"Trades: {n_trades}")

        if not all_test_returns:
            return {}

        # V1: sequential concatenation (NOT equal-weight portfolio)
        all_returns = np.array(all_test_returns)
        aggregate_metrics = PerformanceMetrics.compute_all(all_returns)

        if verbose:
            print("\n" + "=" * 50)
            print("  V1 AGGREGATE RESULTS")
            PerformanceMetrics.print_report(aggregate_metrics)

        return aggregate_metrics


def main():
    config = MAMLConfig(
        tickers=["AAPL", "MSFT", "GOOGL", "JPM", "NVDA"],
        start_date="2015-01-01",
        end_date="2024-12-31",
        # V1: original hyperparameters
        meta_epochs=200,
        inner_steps=5,
        inner_lr=0.005,
        meta_lr=0.0005,
        tasks_per_batch=8,
        k_shot=20,
        query_size=20,
        # V1: large model, ternary
        hidden_dim=128,
        num_layers=2,
        dropout=0.15,
        output_dim=3,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    set_seed(config.seed)
    print(f"[V1] Device: {config.device}")

    pipeline = V1DataPipeline(config)
    processed_data = pipeline.run()

    feature_columns = pipeline.get_feature_columns()
    config.input_dim = len(feature_columns)
    print(f"[V1] Feature dimension: {config.input_dim}")

    backtester = V1Backtester(config, processed_data, feature_columns)
    results = backtester.run(meta_train_epochs=config.meta_epochs, verbose=True)

    if results:
        print("\n" + "=" * 60)
        print("  V1 FINAL RESULTS")
        print("=" * 60)
        PerformanceMetrics.print_report(results)

    return results


if __name__ == "__main__":
    main()
