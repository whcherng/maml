"""
Performance Metrics Module.

Computes standard quantitative trading evaluation metrics:
  - Sharpe Ratio (annualized)
  - Sortino Ratio (annualized, downside deviation only)
  - Maximum Drawdown
  - Hit Rate (fraction of profitable trades)
"""

from typing import Dict

import numpy as np


class PerformanceMetrics:
    """Calculate and store trading performance metrics."""

    TRADING_DAYS_PER_YEAR = 252

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Annualized Sharpe Ratio.

        SR = (mean(R) - Rf) / std(R) * sqrt(252)

        Args:
            returns: array of daily returns.
            risk_free_rate: daily risk-free rate (default 0).
        """
        excess = returns - risk_free_rate
        std = np.std(excess, ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0
        return float(
            np.mean(excess) / std * np.sqrt(PerformanceMetrics.TRADING_DAYS_PER_YEAR)
        )

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Annualized Sortino Ratio — penalizes only downside volatility.

        Sortino = (mean(R) - Rf) / downside_std * sqrt(252)
        """
        excess = returns - risk_free_rate
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf")  # no losing days
        downside_std = np.std(downside, ddof=1)
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return float(
            np.mean(excess)
            / downside_std
            * np.sqrt(PerformanceMetrics.TRADING_DAYS_PER_YEAR)
        )

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """
        Maximum Drawdown — largest peak-to-trough decline.

        Returns a negative float (e.g., -0.15 means 15% drawdown).
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        return float(np.min(drawdowns))

    @staticmethod
    def hit_rate(returns: np.ndarray) -> float:
        """
        Hit Rate — fraction of days with positive returns.
        """
        if len(returns) == 0:
            return 0.0
        return float(np.mean(returns > 0))

    @classmethod
    def compute_all(
        cls,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute all metrics and return as a dictionary.

        Args:
            returns: array of daily strategy returns.
            risk_free_rate: daily risk-free rate.

        Returns:
            Dict with keys: sharpe, sortino, max_drawdown, hit_rate,
            total_return, annualized_return.
        """
        total_ret = float(np.prod(1 + returns) - 1)
        n_days = len(returns)
        # Guard against negative cumulative wealth (which would produce
        # a complex number when raised to a fractional power).
        cum_wealth = 1 + total_ret
        if cum_wealth > 0 and n_days > 0:
            ann_ret = float(
                cum_wealth ** (cls.TRADING_DAYS_PER_YEAR / n_days) - 1
            )
        else:
            ann_ret = -1.0  # total loss

        return {
            "sharpe_ratio": cls.sharpe_ratio(returns, risk_free_rate),
            "sortino_ratio": cls.sortino_ratio(returns, risk_free_rate),
            "max_drawdown": cls.max_drawdown(returns),
            "hit_rate": cls.hit_rate(returns),
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "n_trading_days": n_days,
        }

    @staticmethod
    def print_report(metrics: Dict[str, float]) -> None:
        """Pretty-print a metrics dictionary."""
        print("\n" + "=" * 50)
        print("  PERFORMANCE REPORT")
        print("=" * 50)
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>10.3f}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']:>10.3f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']:>10.2%}")
        print(f"  Total Return:      {metrics['total_return']:>10.2%}")
        print(f"  Annualized Return: {metrics['annualized_return']:>10.2%}")
        print(f"  Trading Days:      {metrics['n_trading_days']:>10d}")
        # Active trading stats (present when run through backtester)
        if "active_hit_rate" in metrics:
            print("  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─")
            active_pct = metrics.get("active_pct", 0)
            print(f"  Active Hit Rate:   {metrics['active_hit_rate']:>10.1%}")
            print(f"  Overall Hit Rate:  {metrics['hit_rate']:>10.1%}")
            print(f"  Active Days:       {metrics.get('active_days', 0):>10d}"
                  f"  ({active_pct:.0%} of total)")
            print(f"  Total Trades:      {metrics.get('total_trades', 0):>10d}")
        else:
            print(f"  Hit Rate:          {metrics['hit_rate']:>10.2%}")
        print("=" * 50 + "\n")

    @classmethod
    def print_benchmark_comparison(
        cls,
        strategy_metrics: Dict[str, float],
        benchmarks: Dict[str, "np.ndarray"],
    ) -> None:
        """
        Print a side-by-side comparison of the strategy vs benchmarks.

        Args:
            strategy_metrics: pre-computed strategy metrics dict.
            benchmarks: {name: daily_returns_array} from DataPipeline.
        """
        import numpy as np

        print("\n" + "=" * 72)
        print("  BENCHMARK COMPARISON")
        print("=" * 72)

        header = f"  {'':>16s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'Return':>9s} {'Ann.Ret':>8s}"
        print(header)
        print("  " + "─" * 68)

        # Strategy row
        print(f"  {'MAML Strategy':>16s} "
              f"{strategy_metrics['sharpe_ratio']:>8.3f} "
              f"{strategy_metrics['sortino_ratio']:>8.3f} "
              f"{strategy_metrics['max_drawdown']:>8.2%} "
              f"{strategy_metrics['total_return']:>9.2%} "
              f"{strategy_metrics['annualized_return']:>8.2%}")

        # Benchmark rows
        for name, returns in benchmarks.items():
            if len(returns) == 0:
                continue
            bm = cls.compute_all(returns)
            label = name if len(name) <= 16 else name[:16]
            print(f"  {label:>16s} "
                  f"{bm['sharpe_ratio']:>8.3f} "
                  f"{bm['sortino_ratio']:>8.3f} "
                  f"{bm['max_drawdown']:>8.2%} "
                  f"{bm['total_return']:>9.2%} "
                  f"{bm['annualized_return']:>8.2%}")

        print("=" * 72 + "\n")
