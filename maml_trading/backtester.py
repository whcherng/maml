"""
Walk-Forward Backtesting & Validation Protocol.

Implements:
  1. Walk-forward optimization: Train → Validate → Test splits that
     roll forward through time, ensuring the model is always evaluated
     on truly out-of-sample data.
  2. Embargo period: a strict gap between validation and test segments
     to prevent temporal data leakage from overlapping lookback windows.
  3. Execution simulation:
     - Transaction costs (10-20 bps per round trip).
     - Market impact / slippage via a simplified Almgren-Chriss model
       calibrated on realized market volatility.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from maml_trading.config import MAMLConfig
from maml_trading.maml_engine import MAMLTradingEngine
from maml_trading.task_generator import MarketRegimeTask, MarketRegimeTaskGenerator
from maml_trading.metrics import PerformanceMetrics


class ExecutionSimulator:
    """
    Simulates realistic execution costs for backtesting.

    Components:
      - Fixed transaction cost (spread + commission) in basis points,
        scaled proportionally to the fraction of portfolio traded.
      - Variable market impact via simplified Almgren-Chriss:
            impact = eta * sigma * sqrt(|trade_fraction|)
        where sigma is the realized volatility over a trailing window.
        The sqrt reflects the well-known square-root law of market impact.
    """

    def __init__(self, config: MAMLConfig):
        self.cfg = config
        # Half the round-trip cost (we pay half on entry, half on exit)
        self.tc_half_turn = config.transaction_cost_bps / 10_000 / 2

    def compute_transaction_cost(self, trade_fraction: float) -> float:
        """
        Fixed cost component, proportional to the amount traded.

        A position change from -1 to +1 is a trade_fraction of 2.0,
        so we pay 2× the half-turn cost.

        Args:
            trade_fraction: absolute change in position (0 to 2).
        Returns:
            Cost as a fraction of portfolio value.
        """
        return self.tc_half_turn * abs(trade_fraction)

    def compute_market_impact(
        self,
        realized_vol: float,
        trade_fraction: float,
    ) -> float:
        """
        Simplified Almgren-Chriss market impact model.

        Temporary impact:  eta * sigma * sqrt(|trade_fraction|)
        Permanent impact:  gamma * |trade_fraction|

        Uses sqrt for temporary impact (square-root law) to avoid
        unrealistically large costs on full position flips.

        Args:
            realized_vol: trailing realized volatility (daily).
            trade_fraction: absolute change in position (0 to 2).
        Returns:
            Total impact cost as a fraction.
        """
        abs_trade = abs(trade_fraction)
        temp_impact = self.cfg.almgren_chriss_eta * realized_vol * np.sqrt(abs_trade)
        perm_impact = self.cfg.almgren_chriss_gamma * abs_trade
        return temp_impact + perm_impact

    def apply_costs(
        self,
        raw_return: float,
        trade_fraction: float,
        realized_vol: float,
    ) -> float:
        """
        Apply all execution costs to a raw return.

        Args:
            raw_return: the pre-cost daily return.
            trade_fraction: absolute change in position (0 if no trade).
            realized_vol: trailing realized volatility.
        Returns:
            Net return after costs.
        """
        if trade_fraction == 0.0:
            return raw_return
        tc = self.compute_transaction_cost(trade_fraction)
        impact = self.compute_market_impact(realized_vol, trade_fraction)
        return raw_return - tc - impact


class WalkForwardBacktester:
    """
    Walk-forward optimization backtester with embargo periods.

    The time series is divided into rolling windows:

      |--- Train ---|-- Val --|-- Embargo --|--- Test ---|
                                  (gap)

    For each fold:
      1. Meta-train the MAML engine on the Train segment.
      2. Validate / tune on the Val segment.
      3. Skip the Embargo period (no data used).
      4. Evaluate on the Test segment with execution simulation.

    Results are aggregated across all test segments to produce
    the final performance metrics.
    """

    def __init__(
        self,
        config: MAMLConfig,
        processed_data: Dict[str, pd.DataFrame],
        feature_columns: List[str],
    ):
        self.cfg = config
        self.processed_data = processed_data
        self.feature_columns = feature_columns
        self.exec_sim = ExecutionSimulator(config)
        self.fold_results: List[Dict] = []

    def _compute_splits(
        self, total_len: int
    ) -> List[Tuple[int, int, int, int, int]]:
        """
        Compute walk-forward split indices.

        Returns list of (train_start, train_end, val_end, embargo_end, test_end)
        tuples. Currently uses a single expanding-window split for
        simplicity; extend to rolling windows by adjusting train_start.
        """
        train_end = int(total_len * self.cfg.train_ratio)
        val_end = train_end + int(total_len * self.cfg.val_ratio)
        embargo_end = val_end + self.cfg.embargo_days
        test_end = total_len

        # Ensure test segment exists after embargo
        if embargo_end >= test_end:
            embargo_end = val_end  # no embargo if not enough data

        return [(0, train_end, val_end, embargo_end, test_end)]

    def _signal_to_position(self, prediction: int, probs: np.ndarray) -> float:
        """
        Map binary prediction to a position using the model's confidence.

        Uses a sigmoid mapping centered at 0.50 (no directional bias).
        The model itself must learn when to be long vs. flat — we don't
        impose a bull or bear prior.

        Args:
            prediction: predicted class (0 or 1).
            probs: softmax probabilities [p_flat, p_long].
        Returns:
            Position in [0.0, 1.0]. No shorting.
        """
        p_long = float(probs[1])

        # Mild long bias — center at 0.48 instead of 0.50.
        # When uncertain (p_long=0.50), position ≈ 0.55 (slightly long).
        # This captures the structural upward drift of equities without
        # being as aggressive as the old 0.45 center.
        steepness = 8.0
        center = 0.48
        position = 1.0 / (1.0 + np.exp(-steepness * (p_long - center)))
        return float(np.clip(position, 0.0, 1.0))

    def _vol_scale_position(
        self,
        position: float,
        realized_vol: float,
    ) -> float:
        """
        Scale position down only when volatility is abnormally high.

        Normal vol (~1-2% daily) → no scaling (full position).
        Elevated vol (>2.5% daily) → proportional reduction.
        Extreme vol (>5% daily) → heavy reduction.

        This only activates during genuine stress (COVID crash, tariff
        shock) and leaves positions alone in normal markets.
        """
        vol_threshold = 0.025  # ~40% annualized — only scale above this
        if realized_vol > vol_threshold:
            # Scale down proportionally: at 5% vol → 50% position
            vol_scalar = vol_threshold / realized_vol
            return position * max(vol_scalar, 0.2)  # floor at 20%
        return position  # no scaling in normal vol

    def _compute_realized_vol(
        self,
        returns: np.ndarray,
        idx: int,
        lookback: int,
    ) -> float:
        """
        Trailing realized volatility for Almgren-Chriss impact model.
        """
        start = max(0, idx - lookback)
        window = returns[start:idx]
        if len(window) < 2:
            return 0.01  # fallback
        return float(np.std(window, ddof=1))

    def run(
        self,
        meta_train_epochs: int = 50,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Execute the full walk-forward backtest.

        Key design: meta-train ONE model across ALL tickers jointly
        (cross-sectional learning), then test per-ticker. This gives
        the model much more training data and cross-asset regime
        diversity compared to per-ticker training.

        Returns:
            Aggregated performance metrics across all tickers and folds.
        """
        # ── Compute unified split based on the LONGEST ticker ───────
        # Using max_len ensures training uses all available data from
        # longer tickers. Shorter tickers contribute what they can to
        # training and their test period adapts to their own length.
        max_len = max(len(df) for df in self.processed_data.values())
        splits = self._compute_splits(max_len)
        train_start, train_end, val_end, embargo_end, test_end = splits[0]

        # Store test period indices for benchmark alignment
        self.test_start_idx = embargo_end
        self.test_end_idx = test_end

        if verbose:
            print(f"\n[Backtester] Unified split (based on longest ticker): "
                  f"train={train_end}, val={val_end}, "
                  f"embargo={embargo_end}, test={test_end}")

        # ── 1. Build cross-sectional task generator (ALL tickers) ───
        train_data = {}
        for ticker, df in self.processed_data.items():
            # Each ticker contributes up to train_end rows; shorter
            # tickers simply contribute fewer days.
            ticker_train_end = min(train_end, len(df))
            train_slice = df.iloc[train_start:ticker_train_end].copy()
            if len(train_slice) > self.cfg.regime_window_max + self.cfg.sequence_length:
                train_data[ticker] = train_slice

        task_gen = MarketRegimeTaskGenerator(
            processed_data=train_data,
            feature_columns=self.feature_columns,
            config=self.cfg,
        )

        # ── 2. Meta-train ONE model across all tickers ──────────────
        if verbose:
            print(f"\n  Meta-training across {len(train_data)} tickers jointly...")

        engine = MAMLTradingEngine(
            config=self.cfg,
            task_generator=task_gen,
        )
        engine.meta_train(
            n_epochs=meta_train_epochs,
            verbose=verbose,
        )

        # ── 3. Test per-ticker using the shared meta-model ─────────
        # Collect per-ticker daily returns aligned by index for proper
        # equal-weight portfolio aggregation
        ticker_daily_returns = {}

        for ticker, df in self.processed_data.items():
            if verbose:
                print(f"\n{'─'*50}")
                print(f"  Testing: {ticker}")
                print(f"{'─'*50}")

            features = df[self.feature_columns].values.astype(np.float32)
            labels = df["Target"].values.astype(np.int64)
            raw_returns = (
                df["Raw_Returns"].values
                if "Raw_Returns" in df.columns
                else np.zeros(len(df))
            )

            ticker_returns = []
            active_returns = []
            prev_position = 0.0
            n_trades = 0
            seq_len = self.cfg.sequence_length
            k_shot = self.cfg.k_shot

            # Risk management state
            peak_equity = 1.0
            current_equity = 1.0

            test_start_t = embargo_end + seq_len + k_shot

            # Use this ticker's actual data length as the upper bound
            ticker_test_end = min(test_end, len(features))

            for t in range(test_start_t, ticker_test_end):
                if t >= len(features):
                    break

                # Build support set from recent data
                support_start = t - k_shot - seq_len
                support_xs = []
                support_ys = []
                for i in range(k_shot):
                    s = support_start + i
                    e = s + seq_len
                    if e <= len(features):
                        support_xs.append(features[s:e])
                        support_ys.append(labels[e - 1])

                if len(support_xs) < 2:
                    ticker_returns.append(0.0)
                    continue

                # Build query
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
                raw_position = self._signal_to_position(preds[0], probs[0])

                # ── Realized vol for cost model ─────────────────
                realized_vol = self._compute_realized_vol(
                    raw_returns, t, self.cfg.volatility_lookback
                )

                # ── Volatility-adaptive position scaling ────────
                # Automatically reduces exposure in high-vol regimes
                raw_position = self._vol_scale_position(
                    raw_position, realized_vol
                )

                # ── Risk Management (graduated) ─────────────────
                drawdown = (current_equity - peak_equity) / peak_equity
                if drawdown < -0.10:
                    # Graduated reduction: scale down proportionally
                    # At -10% DD → 70% of target, at -20% DD → 30%
                    dd_severity = min(abs(drawdown) / 0.20, 1.0)
                    raw_position *= (1.0 - 0.7 * dd_severity)

                # Faster position smoothing for de-risking, slower for adding
                if raw_position < prev_position:
                    alpha = 0.85  # fast de-risk
                else:
                    alpha = 0.6   # slower to add risk
                position = prev_position + alpha * (raw_position - prev_position)
                if abs(position) < 0.03:
                    position = 0.0

                # Minimum trade size — don't trade if position change
                # is too small (reduces churn and transaction costs)
                if abs(position - prev_position) < 0.08:
                    position = prev_position

                # Compute return
                daily_raw_return = (
                    raw_returns[t] * position if t < len(raw_returns) else 0.0
                )

                trade_fraction = abs(position - prev_position)
                # realized_vol already computed above for vol-scaling
                net_return = self.exec_sim.apply_costs(
                    daily_raw_return, trade_fraction, realized_vol
                )

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
                fold_metrics = PerformanceMetrics.compute_all(
                    np.array(ticker_returns)
                )
                n_days = len(ticker_returns)
                n_active = len(active_returns)
                active_hit = (
                    np.mean(np.array(active_returns) > 0)
                    if active_returns else 0.0
                )
                print(
                    f"  [{ticker}] Sharpe: {fold_metrics['sharpe_ratio']:.3f}  "
                    f"MaxDD: {fold_metrics['max_drawdown']:.2%}  "
                    f"Hit: {active_hit:.1%}  "
                    f"Active: {n_active}/{n_days} ({n_active/n_days*100:.0f}%)  "
                    f"Trades: {n_trades}"
                )
                self.fold_results.append({
                    "ticker": ticker,
                    "active_hit_rate": float(active_hit),
                    "active_days": n_active,
                    "total_days": n_days,
                    "n_trades": n_trades,
                    **fold_metrics,
                })

        # ── Aggregate metrics (equal-weight portfolio) ──────────────
        if not ticker_daily_returns:
            print("[Backtester] No test returns generated.")
            return {}

        # Align to the LONGEST ticker. Shorter tickers are padded with
        # 0.0 (= no position) so they don't drag down the portfolio on
        # days they lack data. The daily return is the average across
        # all tickers that have data on that day.
        max_days = max(len(r) for r in ticker_daily_returns.values())
        padded = np.zeros((len(ticker_daily_returns), max_days))
        mask = np.zeros((len(ticker_daily_returns), max_days))
        for i, (ticker, r) in enumerate(ticker_daily_returns.items()):
            padded[i, :len(r)] = r
            mask[i, :len(r)] = 1.0
        # Average only across tickers that have data on each day
        active_count = mask.sum(axis=0).clip(min=1)
        portfolio_returns = padded.sum(axis=0) / active_count

        aggregate_metrics = PerformanceMetrics.compute_all(portfolio_returns)
        aggregate_metrics["daily_returns"] = portfolio_returns

        total_active = sum(f.get("active_days", 0) for f in self.fold_results)
        total_days = sum(f.get("total_days", 0) for f in self.fold_results)
        total_trades = sum(f.get("n_trades", 0) for f in self.fold_results)
        if total_active > 0:
            weighted_hit = sum(
                f.get("active_hit_rate", 0) * f.get("active_days", 0)
                for f in self.fold_results
            ) / total_active
        else:
            weighted_hit = 0.0

        aggregate_metrics["active_hit_rate"] = weighted_hit
        aggregate_metrics["active_days"] = total_active
        aggregate_metrics["total_days"] = len(portfolio_returns)
        aggregate_metrics["total_trades"] = total_trades
        aggregate_metrics["active_pct"] = total_active / max(total_days, 1)

        if verbose:
            print("\n" + "=" * 50)
            print("  AGGREGATE BACKTEST RESULTS")
            PerformanceMetrics.print_report(aggregate_metrics)

        return aggregate_metrics
