"""
Data Preprocessing & Feature Engineering Module.

Handles:
  1. OHLCV download via yfinance for S&P 500 liquid stocks (2015-2024).
  2. Technical indicators: SMA50, SMA200, RSI, Bollinger Bands.
  3. Optional FinBERT sentiment scores.
  4. Preprocessing: forward-fill, outlier trimming (1st/99th pct),
     rolling z-score normalization for distribution-shift robustness.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from maml_trading.config import MAMLConfig

warnings.filterwarnings("ignore", category=FutureWarning)


class DataPipeline:
    """End-to-end data pipeline: download → features → preprocess."""

    def __init__(self, config: MAMLConfig):
        self.cfg = config
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.index_data: Dict[str, pd.DataFrame] = {}  # market index data

    # ── 1. Data Download ────────────────────────────────────────────

    def download_ohlcv(self) -> Dict[str, pd.DataFrame]:
        """
        Download daily OHLCV data for each ticker via yfinance.
        Returns dict of {ticker: DataFrame} with DatetimeIndex.
        """
        print(f"[DataPipeline] Downloading OHLCV for {len(self.cfg.tickers)} tickers...")
        for ticker in self.cfg.tickers:
            try:
                df = yf.download(
                    ticker,
                    start=self.cfg.start_date,
                    end=self.cfg.end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    print(f"  ⚠ No data for {ticker}, skipping.")
                    continue
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index)
                self.raw_data[ticker] = df
            except Exception as e:
                print(f"  ⚠ Failed to download {ticker}: {e}")
        print(f"  ✓ Downloaded {len(self.raw_data)} tickers.")
        return self.raw_data

    # ── 1b. Market Index Download ───────────────────────────────────

    def download_index_data(self) -> Dict[str, pd.DataFrame]:
        """
        Download daily data for market indices (NASDAQ, S&P 500).
        These provide macro-level context features for each stock.
        """
        if not self.cfg.index_tickers:
            return {}

        print(f"[DataPipeline] Downloading {len(self.cfg.index_tickers)} market indices...")
        for idx_ticker in self.cfg.index_tickers:
            try:
                df = yf.download(
                    idx_ticker,
                    start=self.cfg.start_date,
                    end=self.cfg.end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    print(f"  ⚠ No data for {idx_ticker}, skipping.")
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index)
                self.index_data[idx_ticker] = df
            except Exception as e:
                print(f"  ⚠ Failed to download {idx_ticker}: {e}")
        print(f"  ✓ Downloaded {len(self.index_data)} indices.")
        return self.index_data

    def _compute_index_features(self) -> pd.DataFrame:
        """
        Derive features from market index data:
          - Daily returns and momentum (5d, 20d)
          - Realized volatility (20d)
          - RSI
          - Distance from 50-day SMA (trend strength)

        Returns a DataFrame indexed by date with prefixed column names.
        """
        all_features = []

        for idx_ticker, df in self.index_data.items():
            # Clean name for column prefix (e.g., "^IXIC" → "IXIC")
            name = idx_ticker.replace("^", "")
            close = df["Close"]

            idx_df = pd.DataFrame(index=df.index)
            idx_df[f"{name}_Ret"] = close.pct_change()
            idx_df[f"{name}_Mom5"] = close.pct_change(5)
            idx_df[f"{name}_Mom20"] = close.pct_change(20)
            idx_df[f"{name}_Vol20"] = idx_df[f"{name}_Ret"].rolling(20).std()
            idx_df[f"{name}_RSI"] = RSIIndicator(close=close, window=14).rsi()
            sma50 = close.rolling(50, min_periods=1).mean()
            idx_df[f"{name}_SMA50_Dist"] = (close - sma50) / sma50

            all_features.append(idx_df)

        if not all_features:
            return pd.DataFrame()

        return pd.concat(all_features, axis=1)

    def _merge_index_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge pre-computed index features into a ticker DataFrame."""
        if not hasattr(self, '_index_features_cache'):
            self._index_features_cache = self._compute_index_features()

        if self._index_features_cache.empty:
            return df

        # Left-join on date index, forward-fill any gaps
        df = df.join(self._index_features_cache, how="left")
        idx_cols = self._index_features_cache.columns.tolist()
        df[idx_cols] = df[idx_cols].ffill()
        return df

    # ── 2. Technical Indicators ─────────────────────────────────────

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append technical indicators and alpha features to a
        single-ticker OHLCV DataFrame.
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # ── Trend Indicators ────────────────────────────────────────
        df["SMA50"] = close.rolling(window=self.cfg.sma_short, min_periods=1).mean()
        df["SMA200"] = close.rolling(window=self.cfg.sma_long, min_periods=1).mean()
        # SMA crossover signal (normalized)
        df["SMA_Cross"] = (df["SMA50"] - df["SMA200"]) / df["SMA200"]

        # ── Momentum ────────────────────────────────────────────────
        rsi = RSIIndicator(close=close, window=self.cfg.rsi_period)
        df["RSI"] = rsi.rsi()

        # Multi-horizon momentum (captures different trend speeds)
        for period in [5, 10, 20]:
            df[f"Mom_{period}d"] = close.pct_change(period)

        # Rate of change of momentum (acceleration)
        df["Mom_Accel"] = df["Mom_5d"] - df["Mom_5d"].shift(5)

        # ── Volatility ──────────────────────────────────────────────
        bb = BollingerBands(
            close=close, window=self.cfg.bb_period, window_dev=self.cfg.bb_std,
        )
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_middle"] = bb.bollinger_mavg()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
        # BB position: where price sits within the bands (-1 to +1)
        bb_range = df["BB_upper"] - df["BB_lower"]
        df["BB_position"] = np.where(
            bb_range > 0,
            2 * (close - df["BB_lower"]) / bb_range - 1,
            0.0,
        )

        # ── Price-derived features ──────────────────────────────────
        df["Returns"] = close.pct_change()
        df["Log_Returns"] = np.log(close / close.shift(1))
        df["Volatility_20d"] = df["Returns"].rolling(20).std()
        df["Volatility_5d"] = df["Returns"].rolling(5).std()
        # Vol regime change (rising vol = risk-off)
        df["Vol_Ratio"] = df["Volatility_5d"] / df["Volatility_20d"].replace(0, 1)

        # ── Mean-Reversion Signals ──────────────────────────────────
        # Distance from 20-day mean (z-score of price)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std().replace(0, 1)
        df["Price_Zscore"] = (close - ma20) / std20

        # ── Microstructure / Volume Features ────────────────────────
        avg_vol = volume.rolling(20).mean().replace(0, 1)
        df["Volume_Ratio"] = volume / avg_vol
        # On-balance volume trend
        df["OBV_Slope"] = (
            (volume * np.sign(df["Returns"].fillna(0)))
            .rolling(10).mean()
        ) / avg_vol

        # ── Candlestick Features ────────────────────────────────────
        body = close - df["Open"]
        candle_range = high - low
        df["Body_Ratio"] = body / candle_range.replace(0, 1)
        df["Upper_Shadow"] = (high - np.maximum(close, df["Open"])) / candle_range.replace(0, 1)

        return df

    # ── 3. FinBERT Sentiment (Optional) ─────────────────────────────

    def _add_sentiment_placeholder(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a sentiment score column. When `use_sentiment` is True,
        this would call FinBERT on aligned news headlines. Here we
        provide a neutral placeholder (0.0) so the feature vector
        shape is consistent regardless of toggle.

        To integrate real FinBERT scores:
          1. Collect daily news headlines per ticker.
          2. Tokenize with AutoTokenizer.from_pretrained(cfg.finbert_model_name).
          3. Run inference and map softmax outputs to [-1, +1].
          4. Merge on date index.
        """
        if self.cfg.use_sentiment:
            # ── Real FinBERT integration point ──
            # from transformers import AutoTokenizer, AutoModelForSequenceClassification
            # tokenizer = AutoTokenizer.from_pretrained(self.cfg.finbert_model_name)
            # model = AutoModelForSequenceClassification.from_pretrained(
            #     self.cfg.finbert_model_name
            # )
            # ... score headlines and merge ...
            pass
        # Default: neutral sentiment placeholder
        df["Sentiment"] = 0.0
        return df

    # ── 4. Preprocessing ────────────────────────────────────────────

    def _forward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill missing prices (weekends, holidays, halts)."""
        return df.ffill()

    def _trim_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize numeric columns at the 1st and 99th percentiles
        to limit the influence of extreme values on gradient updates.
        Excludes the Target column (categorical label).
        """
        exclude = {"Target", "Raw_Returns"}
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        for col in numeric_cols:
            lo = df[col].quantile(self.cfg.outlier_lower_pct)
            hi = df[col].quantile(self.cfg.outlier_upper_pct)
            df[col] = df[col].clip(lower=lo, upper=hi)
        return df

    def _rolling_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling z-score normalization to handle distribution
        shifts across market regimes. Each feature is standardized
        using a trailing window of `rolling_norm_window` days.
        Excludes the Target and Raw_Returns columns.
        """
        exclude = {"Target", "Raw_Returns"}
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        for col in numeric_cols:
            roll_mean = df[col].rolling(
                window=self.cfg.rolling_norm_window, min_periods=1
            ).mean()
            roll_std = df[col].rolling(
                window=self.cfg.rolling_norm_window, min_periods=1
            ).std().replace(0, 1)  # avoid division by zero
            df[col] = (df[col] - roll_mean) / roll_std
        return df

    # ── 5. Target Label Generation ──────────────────────────────────

    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a binary classification target:
          0 = flat/avoid  (next-day return below median)
          1 = long        (next-day return above median)

        Binary classification is more learnable than ternary and
        avoids shorting long-biased equities. Uses a LAGGED rolling
        median of past realized returns as the threshold — no future
        data is used in the threshold computation.
        """
        fwd_ret = df["Close"].pct_change().shift(-1)

        # Compute rolling median on PAST realized returns (not fwd_ret)
        # to avoid look-ahead bias. The threshold at day t is the median
        # of returns from day t-60 to t-1, which would be known at time t.
        past_ret = df["Close"].pct_change()
        roll_median = past_ret.rolling(window=60, min_periods=20).median()
        global_median = past_ret.median()
        roll_median = roll_median.fillna(global_median)

        df["Target"] = (fwd_ret > roll_median).astype(np.int64)
        return df

    # ── 6. Full Pipeline ────────────────────────────────────────────

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the complete pipeline:
          download → indicators → sentiment → ffill → outlier trim
          → labels → rolling normalization → drop NaNs.
        Returns dict of {ticker: processed_DataFrame}.
        """
        if not self.raw_data:
            self.download_ohlcv()

        # Download market index data for benchmarking
        if not self.index_data and self.cfg.index_tickers:
            self.download_index_data()

        print("[DataPipeline] Processing features...")
        for ticker, df in self.raw_data.items():
            df = df.copy()
            df = self._add_technical_indicators(df)
            df = self._add_sentiment_placeholder(df)
            df = self._forward_fill(df)
            df = self._trim_outliers(df)
            df = self._generate_labels(df)
            # Preserve raw (un-normalized) returns for backtesting P&L.
            # These must NOT be z-scored — they represent actual daily
            # percentage changes used to compute strategy returns.
            df["Raw_Returns"] = df["Returns"].copy()
            df = self._rolling_normalize(df)
            df = df.dropna()
            self.processed_data[ticker] = df

        print(f"  ✓ Processed {len(self.processed_data)} tickers.")
        return self.processed_data

    def get_feature_columns(self) -> list:
        """
        Return a curated list of the most predictive features.

        Fewer, higher-quality features help the small LSTM generalize
        better. We keep: trend (SMA_Cross), momentum (RSI, Mom),
        volatility (BB_width, BB_position, Vol_Ratio), returns,
        mean-reversion (Price_Zscore), and volume (Volume_Ratio).
        """
        curated = [
            # Per-stock features
            "SMA_Cross",       # trend direction
            "RSI",             # momentum oscillator
            "Mom_5d",          # short-term momentum
            "Mom_20d",         # medium-term momentum
            "BB_width",        # volatility regime
            "BB_position",     # mean-reversion within bands
            "Returns",         # recent return
            "Volatility_20d",  # realized vol
            "Vol_Ratio",       # vol regime change
            "Price_Zscore",    # mean-reversion signal
            "Volume_Ratio",    # volume anomaly
        ]
        # Only include features that actually exist in the data
        sample = next(iter(self.processed_data.values()))
        return [c for c in curated if c in sample.columns]

    def get_benchmark_returns(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """
        Compute daily buy-and-hold returns for each market index
        and each individual stock over the test period.

        Args:
            start_idx: start index into the processed data (aligned to
                       the shortest ticker).
            end_idx: end index.

        Returns:
            Dict of {name: daily_returns_array} for benchmarks.
        """
        benchmarks = {}

        # Index benchmarks
        for idx_ticker, df in self.index_data.items():
            name = idx_ticker.replace("^", "")
            close = df["Close"]
            daily_ret = close.pct_change().dropna().values

            # Align to the test period by date range from processed data
            sample_ticker = next(iter(self.processed_data.keys()))
            sample_df = self.processed_data[sample_ticker]
            test_dates = sample_df.index[start_idx:end_idx]

            if len(test_dates) == 0:
                continue

            idx_ret = close.pct_change()
            idx_ret = idx_ret.reindex(test_dates).fillna(0.0).values
            benchmarks[name] = idx_ret

        # Equal-weight buy-and-hold of the stock universe
        stock_returns = []
        for ticker, df in self.processed_data.items():
            raw_ret = df["Raw_Returns"].values if "Raw_Returns" in df.columns else df["Returns"].values
            sliced = raw_ret[start_idx:end_idx]
            stock_returns.append(sliced)

        if stock_returns:
            min_len = min(len(r) for r in stock_returns)
            aligned = np.array([r[:min_len] for r in stock_returns])
            benchmarks["EW_Stocks"] = aligned.mean(axis=0)

        return benchmarks

    def get_combined_dataframe(self) -> pd.DataFrame:
        """
        Stack all tickers into a single DataFrame with a 'Ticker'
        column, useful for cross-sectional task sampling.
        """
        frames = []
        for ticker, df in self.processed_data.items():
            tmp = df.copy()
            tmp["Ticker"] = ticker
            frames.append(tmp)
        return pd.concat(frames, axis=0).sort_index()
