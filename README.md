# Copper futures vs. China manufacturing PMI: which PMI component leads copper, and by how much?

A Python time-series model that tests whether China's manufacturing PMI and its sub-indices lead
copper futures returns, and identifies the component and horizon the data supports. On the
internship data the answer was the **New Orders sub-index, one month ahead**; this repository is the
reproducible version of that study, built to be rerun on any monthly sample.

Built from the CITIC Futures winter internship project (Dec 2025 to Jan 2026). The first version was a
12-month OLS; this one applies the standard monthly macro toolkit to ten-plus years of data, compares
every PMI component on the same footing, and evaluates the finding out of sample.

## Method

| Step | Method | Question it answers |
|---|---|---|
| 1 | ADF and KPSS tests | Which series are stationary, so models run on returns and index changes, not price levels |
| 2 | Lead-lag cross-correlation, k = -6..+6, both forms (index minus 50, and month-on-month change), with the 1.96/sqrt(n) band | At what horizon each component co-moves with copper returns |
| 2b | Sub-index ranking: every PMI component, both forms, correlation at lead 1 and Granger p at lag 1 | Which component is the one-month leading indicator |
| 3 | Granger causality, both directions, lags 1-6 | Does the component forecast copper beyond copper's own past, and not the reverse |
| 4 | VAR with BIC lag selection, orthogonalised impulse responses | Dynamic response of copper to a component shock |
| 5 | Lagged regression with Newey-West (HAC) standard errors | Size and significance of each lag, robust to autocorrelation |
| 6 | Expanding-window out-of-sample forecasts; RMSE vs random walk and historical mean; Diebold-Mariano test; hit rate | Is the predictive content real out of sample |
| 7 | Signal backtest with transaction costs; Sharpe, drawdown | Economic significance |

The signal used in steps 6 and 7 (which component, which form) is chosen on the first 60 months only, so
the evaluation period never informs the choice.

Outputs in `--out`: `report.md` (headline paragraph plus every table), one CSV per step, `var.json`, and
figures `levels.png`, `lead_lag.png`, `var_irf.png`, `oos_forecast.png`, `backtest.png`.

## Run

```bash
python3 -m venv .venv && ./.venv/bin/pip install pandas numpy scipy statsmodels matplotlib tabulate
./.venv/bin/python copper_pmi_ts.py --demo --out results_demo                      # synthetic data, checks the pipeline
./.venv/bin/python copper_pmi_ts.py --pmi data/pmi.csv --copper data/copper.csv --out results
```

## Data

- `data/pmi.csv`: `date,pmi,new_orders[,production,new_export_orders,raw_material_inventories,employment,...]`.
  Monthly NBS manufacturing PMI and sub-indices in percent; every extra column is tested as a candidate.
  Source: National Bureau of Statistics monthly PMI release (Wind and Choice export all components).
- `data/copper.csv`: `date,price`. SHFE copper main contract (CNY/t) monthly average or month-end, or LME
  3-month copper (USD/t). FRED series PCOPPUSDM (global copper price, monthly) also works.
- Dates may be any day of the month; everything is aligned to month end. At least 36 overlapping months
  are required; 120 or more is what the tests need to be meaningful.
- `--fetch` tries akshare (headline PMI) and FRED (copper) and tells you what to supply by hand if a
  download fails; the sub-indices come from `data/pmi.csv`.

## What the demo shows

`--demo` generates 180 months in which only New Orders carries a one-month lead on copper returns
(production co-moves without extra lead, export orders are unrelated, inventories are weakly
counter-cyclical). A correct pipeline ranks New Orders (level) first with the highest lead-1 correlation
and a Granger p near zero, shows the cross-correlation peak at k = 1, finds copper does not Granger-cause
New Orders, and beats the random walk out of sample. Real data are noisier; read the p-values with the
sample size in mind.

![Lead-lag cross-correlation on the demo data](figures/lead_lag_demo.png)

## Reading the result on real data

- `report.md` opens with the component and horizon the data supports and the three numbers that back it:
  the lead-1 correlation with its white-noise band, the Granger p-value, and the out-of-sample RMSE
  against the random walk with the Diebold-Mariano p-value.
- A component that leads in the `level` form means the index being above or below 50 matters; in the
  `change` form it is the month's acceleration that matters. The ranking table shows both.
- Copper also responds to the US dollar, exchange inventories, policy and supply. The model isolates the
  PMI channel and does not control for those; the backtest is a sanity check of economic significance,
  not a trading strategy.
