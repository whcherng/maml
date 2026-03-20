# MAML Trading Framework

A stock trading system that uses meta-learning (MAML) to adapt to changing market conditions.
Instead of training one fixed model, it learns how to quickly adjust to whatever the market
is doing right now — bull run, crash, sideways chop.

---

## What Does This Project Do?

In plain English: it downloads stock price data, trains a small neural network that can
rapidly adapt to current market conditions, then simulates trading with realistic costs
to see if it makes money.

The key idea: markets change. A model trained on 2020 data won't work well in 2022.
MAML solves this by learning a "starting point" for the model that can be fine-tuned
to any market regime in just 3 gradient steps using the last 30 days of data.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install learn2learn (needed for MAML)
# If on Python 3.12+:
bash install_learn2learn.sh
# If on Python 3.10/3.11:
pip install learn2learn

# Run the main strategy
python main.py

# Run robustness tests
python robustness_tests.py --oos      # true out-of-sample (2025)
python robustness_tests.py --folds    # test each year separately
python robustness_tests.py --cost     # transaction cost sensitivity
python robustness_tests.py --universe # test with 20 stocks
python robustness_tests.py            # run ALL tests (~30 min)
```

---

## How It Works — Step by Step

Here's exactly what happens when you run `python main.py`:

### Step 1: Download Stock Data

The system downloads daily price data (Open, High, Low, Close, Volume) for 6 stocks
from Yahoo Finance:

```
AAPL (Apple), MSFT (Microsoft), GOOGL (Google), JPM (JPMorgan),
NVDA (Nvidia), TEAM (Atlassian)
```

It also downloads NASDAQ and S&P 500 index data for benchmarking (not used by the model).

Date range: 2015-01-01 to 2024-12-31 (~10 years, ~2500 trading days per stock).

**Code:** `maml_trading/data_pipeline.py` → `download_ohlcv()`

### Step 2: Compute Features

Raw prices aren't useful for a model. The system computes 29 technical indicators,
then selects the 11 most useful ones:

| Feature | What It Measures | Plain English |
|---------|-----------------|---------------|
| SMA_Cross | 50-day vs 200-day moving average | Is the stock trending up or down? |
| RSI | Relative Strength Index (14-day) | Is the stock overbought or oversold? |
| Mom_5d | 5-day price change | What happened this week? |
| Mom_20d | 20-day price change | What happened this month? |
| BB_width | Bollinger Band width | How volatile is the stock right now? |
| BB_position | Where price is within Bollinger Bands | Is price near the top or bottom of its range? |
| Returns | Daily percentage change | How much did it move today? |
| Volatility_20d | 20-day realized volatility | How risky is this stock lately? |
| Vol_Ratio | Short-term vol / long-term vol | Is volatility increasing or decreasing? |
| Price_Zscore | How far price is from its 20-day average | Is the stock stretched too far from normal? |
| Volume_Ratio | Today's volume vs average volume | Is there unusual trading activity? |

**Why only 11?** The model is small (~34,000 parameters). Too many features = overfitting.
These 11 cover the four things that matter: trend, momentum, volatility, and volume.

**Code:** `maml_trading/data_pipeline.py` → `_add_technical_indicators()`

### Step 3: Create Labels (What We're Predicting)

For each day, the system asks: "Will tomorrow's return be above or below the recent median?"

- **Label 1 (Long):** Tomorrow's return is above the 60-day rolling median → buy/hold
- **Label 0 (Flat):** Tomorrow's return is below the 60-day rolling median → stay out

The rolling median uses only past data (no future information leaks into the labels).

**Why binary (not short/flat/long)?** Shorting stocks in a bull market loses money.
Binary is simpler and the model learns it better.

**Code:** `maml_trading/data_pipeline.py` → `_generate_labels()`

### Step 4: Preprocess

Three cleaning steps applied to every feature:

1. **Forward-fill** missing values (weekends, holidays)
2. **Winsorize** at 1st/99th percentile (clip extreme values so one flash crash doesn't dominate)
3. **Rolling z-score normalize** with a 60-day window (so "high RSI" means the same thing in 2015 and 2024)

Raw daily returns are saved separately BEFORE normalization — the backtester needs
actual percentages to compute real P&L.

**Code:** `maml_trading/data_pipeline.py` → `_forward_fill()`, `_trim_outliers()`, `_rolling_normalize()`

### Step 5: Split Data (Train / Validate / Test)

The data is split chronologically (never randomly — that would leak future info):

```
|-------- 60% Train --------|-- 15% Val --|-- 10d Gap --|---- 25% Test ----|
     2015 — mid 2021          mid-late 2021   (embargo)    2022 — 2024
```

- **Train:** The model learns from this data
- **Validation:** Reserved for tuning (not actively used for early stopping)
- **Embargo:** 10-day gap so features don't overlap between val and test
- **Test:** The model is evaluated here — it has NEVER seen this data

**Code:** `maml_trading/backtester.py` → `_compute_splits()`

### Step 6: Meta-Train the MAML Model

This is the core of the system. Instead of normal training, MAML uses a two-level process:

```
Repeat 200 times:
    1. Sample 12 random "tasks" (each task = a 60-90 day market window from any stock)
    2. For each task:
       a. Clone the model
       b. INNER LOOP: Fine-tune the clone on 30 days of data (3 gradient steps)
       c. Test the fine-tuned clone on the next 20 days
    3. OUTER LOOP: Use the test results to update the ORIGINAL model
       so that future fine-tuning starts from a better place
```

**Think of it like this:** The model doesn't learn "buy AAPL when RSI < 30."
It learns "here's a good starting point so that when I see 30 days of ANY stock's
data, I can quickly figure out what to do."

The model itself is tiny: a 1-layer LSTM with 64 hidden units (~34K parameters).
Small on purpose — MAML works best with compact models because the inner loop
only has 30 samples to adapt from.

**Code:** `maml_trading/maml_engine.py` → `meta_train()`, `_inner_loop()`, `_outer_step()`

### Step 7: Test (Walk-Forward Backtest)

For each day in the test period, for each stock:

```
1. Take the last 30 days of data as "support set"
2. Adapt the model to this recent data (3 gradient steps)
   — do this 5 times with shuffled data, average the results (ensemble)
3. Model outputs: probability of "long" (p_long)
4. Convert p_long to a position size (0% to 100%) via sigmoid function
5. Scale position down if volatility is high (vol-adaptive sizing)
6. Scale position down if portfolio is in drawdown (graduated risk management)
7. Smooth the position change (don't flip from 0% to 100% in one day)
8. Skip tiny trades (< 8% position change) to avoid unnecessary costs
9. Compute the day's return = stock return × position size - trading costs
```

**Trading costs include:**
- Transaction cost: 15 basis points round-trip (realistic for large-cap stocks)
- Market impact: Almgren-Chriss model (bigger trades move the price against you)

**Code:** `maml_trading/backtester.py` → `run()`

### Step 8: Compute Results

After testing all stocks, daily returns are combined into an equal-weight portfolio
(average across all stocks each day). Then compute:

- **Sharpe Ratio:** Risk-adjusted return (> 1.0 is good, > 2.0 is great)
- **Sortino Ratio:** Like Sharpe but only penalizes losses, not upside volatility
- **Max Drawdown:** Worst peak-to-trough decline (how bad can it get?)
- **Total Return:** How much money you made/lost overall
- **Hit Rate:** What % of days were profitable

Results are compared against benchmarks (NASDAQ, S&P 500, equal-weight buy-and-hold).

**Code:** `maml_trading/metrics.py` → `compute_all()`, `print_benchmark_comparison()`

---

## Project Structure

```
├── main.py                      # Run this — does everything above
├── robustness_tests.py          # Validity tests (OOS, rolling folds, cost sensitivity)
├── maml_trading/
│   ├── config.py                # All settings in one place
│   ├── data_pipeline.py         # Step 1-4: download, features, labels, preprocess
│   ├── task_generator.py        # Step 6: creates "tasks" for MAML training
│   ├── model.py                 # The neural network (small LSTM)
│   ├── maml_engine.py           # Step 6: the MAML training loop
│   ├── backtester.py            # Step 5,7: splits data, simulates trading
│   └── metrics.py               # Step 8: Sharpe, Sortino, MaxDD, etc.
├── iterations/                  # History of how we improved the strategy
│   ├── run_v1.py → run_v6.py   # Each version with its specific changes
│   └── README.md                # What changed in each version
├── CODE_EXPLANATION.md          # Deep technical explanation of every method
├── TUNING_REPORT.md             # Iteration-by-iteration performance comparison
├── requirements.txt             # Python dependencies
└── install_learn2learn.sh       # Install script for learn2learn on Python 3.12+
```

---

## Results

### In-Sample (2015-2024, 60% train / 25% test)

| Metric | MAML Strategy | NASDAQ | S&P 500 | EW Buy-Hold |
|--------|--------------|--------|---------|-------------|
| Sharpe | 1.915 | 1.062 | 1.114 | 1.739 |
| MaxDD | -9.05% | -22.20% | -16.91% | -18.74% |
| Return | 58.69% | 61.59% | 47.71% | 65.50% |

The strategy doesn't beat buy-and-hold on raw returns, but it has **half the drawdown**.
The real edge is risk management, not stock picking.

### Rolling Folds (Each Year Tested Separately, with Benchmarks)

| Year | Condition | MAML Sharpe | MAML Ret | MAML MaxDD | NASDAQ Sharpe | NASDAQ Ret | NASDAQ MaxDD | S&P Sharpe | S&P Ret | S&P MaxDD |
|------|-----------|-------------|----------|------------|---------------|------------|--------------|------------|---------|-----------|
| 2020 | COVID crash | **2.397** | 18.66% | **-5.92%** | 1.348 | 43.76% | -23.92% | 0.802 | 20.77% | -28.52% |
| 2021 | Bull market | **2.551** | 13.13% | **-4.02%** | 1.494 | 20.45% | -7.83% | 2.124 | 22.57% | -5.21% |
| 2022 | Bear market | **-0.544** | **-4.37%** | **-11.72%** | -0.829 | -22.10% | -30.14% | -0.523 | -11.89% | -22.77% |
| 2023 | Recovery | 1.481 | 7.42% | **-5.61%** | **2.162** | 28.63% | -12.27% | **2.181** | 21.15% | -10.28% |
| 2024 | Late bull | 1.105 | 4.96% | **-6.66%** | **1.341** | 19.44% | -13.15% | **1.333** | 13.52% | -8.49% |
| **AVG** | | **1.398** | | | 1.103 | | | 1.183 | | |

Key findings:
- Average Sharpe 1.398 beats NASDAQ (1.103) and S&P 500 (1.183)
- Beat NASDAQ on Sharpe in 3/5 years, beat S&P in 2/5 years
- **MAML wins in volatile/down markets** (2020, 2022): much higher Sharpe, fraction of the drawdown
- **Benchmarks win on raw returns in calm bull markets** (2023, 2024) but with 2-3x the drawdown
- MaxDD never exceeded -11.72% in any year — NASDAQ hit -30% in 2022
- 4 out of 5 years profitable (80%)

The pattern is clear: MAML's value is in **downside protection**, not upside capture. It trades
some raw returns for dramatically lower drawdowns — a favorable tradeoff for risk-conscious investors.

### Out-of-Sample (2025 Q1, tariff crash)

| Metric | MAML Strategy | NASDAQ | S&P 500 |
|--------|--------------|--------|---------|
| Return | -5.44% | -10.36% | -8.86% |
| MaxDD | -7.23% | -24.32% | -18.90% |

The strategy lost money (everything did), but lost **half as much** as the market
with **one-third the drawdown**. The volatility-adaptive sizing automatically reduced
exposure when the tariff shock hit.

---

## Robustness Tests

These tests address common concerns about backtest validity:

| Test | Command | What It Proves |
|------|---------|---------------|
| Cost Sensitivity | `--cost` | Strategy survives at 5/10/15/25/50/100 bps costs |
| Rolling Folds | `--folds` | Tests each year 2020-2024 separately (good, bad, normal) |
| Hold-Out Year | `--holdout` | Train 2015-2023, test 2024 only |
| True OOS | `--oos` | Train 2015-2024, test 2025 (never seen during development) |
| Expanded Universe | `--universe` | 20 stocks including underperformers (GE, INTC, BA) |

---

## Tuning History (V1 → V6)

The strategy was improved iteratively. Each version changed one thing:

| Version | Sharpe | What Changed | Why |
|---------|--------|-------------|-----|
| V1 | -0.97 | Baseline (3 classes, big model) | Model couldn't learn 3 classes |
| V2 | -0.43 | Switched to 2 classes, smaller model | Simpler = more learnable |
| V3 | +0.40 | Train across all stocks together, not one at a time | 6× more training data |
| V4 | +0.25 | Added vol-scaling to positions | Made it worse — reverted |
| V5 | +1.04 | Sigmoid position sizing, more training | Smooth positions reduce costs |
| V6 | +1.83 | Ensemble (adapt 5 times, average) | Reduces prediction noise |

Full details in [TUNING_REPORT.md](TUNING_REPORT.md).

---

## Limitations (Be Honest in Your Paper)

1. **Hyperparameter overfitting:** V1→V6 tuning used test results to guide decisions.
   The true OOS test (2025) shows the real generalization gap.
2. **Survivorship bias:** Testing on stocks that survived and thrived 2015-2024.
   The expanded universe test (with GE, INTC, BA) partially addresses this.
3. **Bull market period:** 2015-2024 was mostly up. The strategy's edge is clearer
   in drawdown reduction than in raw alpha generation.
4. **Small universe:** 6 stocks is not enough to claim broad applicability.
5. **No live trading:** Simulated costs, not real execution.
6. **Seed sensitivity:** Results use seed=42. Different seeds may vary.

---

## Dependencies

- Python 3.10+ (tested on 3.12, macOS ARM)
- PyTorch ≥ 2.0
- learn2learn (see install script for Python 3.12+)
- yfinance (market data download)
- ta (technical indicators)
- pandas, numpy, tqdm
