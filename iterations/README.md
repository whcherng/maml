# Replicable Iteration Versions

Each `run_vN.py` file is a self-contained script that reproduces the exact
configuration and behavior of iteration N from the tuning process.

All scripts share the base `maml_trading/` package but override specific
classes/methods inline where behavior changed between iterations.

## Usage

```bash
python iterations/run_v1.py   # Baseline: ternary labels, large model
python iterations/run_v2.py   # Binary labels, small model
python iterations/run_v3.py   # Bull-bias position sizing, EW portfolio
python iterations/run_v4.py   # Inverse vol-scaling (negative result)
python iterations/run_v5.py   # Sigmoid mapping, more training
python iterations/run_v6.py   # Ensemble inference, final (best)
```

## Expected Results

| Version | Sharpe | Total Return | Description |
|---------|--------|-------------|-------------|
| V1 | -0.972 | -48.82% | Baseline |
| V2 | -0.431 | -29.87% | Binary + small model |
| V3 | 0.404 | 9.55% | Bull-bias + EW portfolio |
| V4 | 0.245 | 3.41% | Vol-scaling (worse) |
| V5 | 1.037 | 26.71% | Sigmoid + training |
| V6 | 1.834 | 62.15% | Ensemble + optimized |

Note: Results may vary slightly due to yfinance data updates and
stochastic training, but relative ordering should be preserved.
