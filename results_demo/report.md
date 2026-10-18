# Copper vs PMI: time-series results (DEMO DATA)

Sample: 2011-01 to 2025-12, 180 months. Indicators: new_orders, pmi.

**Headline.** Among the PMI components, new_orders (level) ranks first as a one-month leading indicator of copper returns (corr 0.396 at lead 1, Granger p=0.0). Across horizons, the strongest positive lead is new_orders (level) at k=1 month(s) ahead of copper returns (corr 0.396, 95% band +-0.146, significant). Granger test pmi (level) -> copper is most significant at lag 1 (p=0.0). Out of sample the model's RMSE is 4.725 vs 4.922 for the random walk (Diebold-Mariano p=0.2493, hit rate 0.559).

## 1. Stationarity

| series           |   ADF p |   KPSS p | verdict        |   n |
|:-----------------|--------:|---------:|:---------------|----:|
| log copper price |  0.981  |     0.01 | non-stationary | 180 |
| copper return    |  0      |     0.1  | stationary     | 179 |
| PMI              |  0.0004 |     0.1  | stationary     | 180 |
| dPMI             |  0      |     0.1  | stationary     | 179 |
| New Orders       |  0.0001 |     0.1  | stationary     | 180 |
| dNew Orders      |  0      |     0.1  | stationary     | 179 |

Returns and PMI changes should be stationary; price and PMI levels usually are not, which is why the analysis runs on changes.

## 2. Lead-lag cross-correlation

|   lag_k |   new_orders change |   new_orders level |   pmi change |   pmi level |
|--------:|--------------------:|-------------------:|-------------:|------------:|
|      -6 |              -0.045 |              0.187 |        0.049 |       0.239 |
|      -5 |               0.131 |              0.217 |        0.106 |       0.196 |
|      -4 |              -0.163 |              0.131 |       -0.104 |       0.105 |
|      -3 |              -0.05  |              0.238 |       -0.019 |       0.193 |
|      -2 |              -0.102 |              0.268 |       -0.079 |       0.206 |
|      -1 |              -0.102 |              0.335 |       -0.096 |       0.275 |
|       0 |               0.001 |              0.398 |       -0.001 |       0.354 |
|       1 |               0.107 |              0.396 |        0.148 |       0.354 |
|       2 |               0.159 |              0.33  |        0.044 |       0.23  |
|       3 |               0.019 |              0.219 |       -0.028 |       0.186 |
|       4 |               0.126 |              0.197 |        0.207 |       0.202 |
|       5 |              -0.009 |              0.105 |       -0.061 |       0.014 |
|       6 |               0.02  |              0.112 |        0.034 |       0.068 |

k > 0 means the indicator leads copper. Values outside +-0.149 are outside the white-noise band.

## 2b. Which PMI component leads copper by one month?

| indicator                | form   |   corr_lead |   granger_p |   n |
|:-------------------------|:-------|------------:|------------:|----:|
| new_orders               | level  |       0.396 |      0      | 179 |
| pmi                      | level  |       0.354 |      0      | 179 |
| production               | level  |       0.292 |      0.0002 | 179 |
| pmi                      | change |       0.148 |      0.0476 | 178 |
| new_orders               | change |       0.107 |      0.156  | 178 |
| production               | change |       0.098 |      0.12   | 178 |
| raw_material_inventories | change |       0.038 |      0.6538 | 178 |
| new_export_orders        | change |      -0.042 |      0.5069 | 178 |
| new_export_orders        | level  |      -0.098 |      0.1742 | 179 |
| raw_material_inventories | level  |      -0.098 |      0.2303 | 179 |

All components are compared on the same footing (indicator at t-1 vs copper return at t, in both forms); the top row is the indicator the data supports.

## 3. Granger causality (F-test p-values)

|   lag |   copper -> new_orders (change) |   copper -> new_orders (level) |   copper -> pmi (change) |   copper -> pmi (level) |   new_orders (change) -> copper |   new_orders (level) -> copper |   pmi (change) -> copper |   pmi (level) -> copper |
|------:|--------------------------------:|-------------------------------:|-------------------------:|------------------------:|--------------------------------:|-------------------------------:|-------------------------:|------------------------:|
|     1 |                          0.1735 |                         0.6122 |                   0.1893 |                  0.335  |                          0.156  |                         0      |                   0.0476 |                  0      |
|     2 |                          0.1663 |                         0.9184 |                   0.2416 |                  0.7324 |                          0.0189 |                         0      |                   0.0549 |                  0.0003 |
|     3 |                          0.2624 |                         0.9664 |                   0.4452 |                  0.872  |                          0.0242 |                         0.0001 |                   0.0846 |                  0.0015 |
|     4 |                          0.0879 |                         0.7077 |                   0.2549 |                  0.8077 |                          0.007  |                         0.0003 |                   0.0023 |                  0.0045 |
|     5 |                          0.0798 |                         0.1395 |                   0.3747 |                  0.3478 |                          0.0124 |                         0.0006 |                   0.0045 |                  0.0009 |
|     6 |                          0.2059 |                         0.25   |                   0.4983 |                  0.2355 |                          0.0156 |                         0.0014 |                   0.0056 |                  0.0021 |

## 4. VAR

Lag order by BIC: 1; AIC 2.799, BIC 2.906.

Copper-return equation coefficients: const=-0.6549, L1.ret=0.1689, L1.d_no=0.6091

Orthogonalised impulse response of copper return to a one-s.d. indicator shock, months 0-6: 0.0, 0.5057, -0.0077, 0.0104, -0.0003, 0.0002, -0.0

## 5. Lagged regression with Newey-West standard errors

| indicator   | form   | term                  |    coef |   t (HAC) |      p |     R2 |   n |
|:------------|:-------|:----------------------|--------:|----------:|-------:|-------:|----:|
| new_orders  | level  | const                 |  0.3277 |      0.71 | 0.4787 | 0.1644 | 177 |
| new_orders  | level  | new_orders_level_lag1 |  1.4433 |      3.9  | 0.0001 | 0.1644 | 177 |
| new_orders  | level  | new_orders_level_lag2 |  0.4742 |      0.84 | 0.3982 | 0.1644 | 177 |
| new_orders  | level  | new_orders_level_lag3 | -0.4529 |     -1    | 0.3159 | 0.1644 | 177 |
| pmi         | level  | const                 |  0.1703 |      0.33 | 0.7414 | 0.1278 | 177 |
| pmi         | level  | pmi_level_lag1        |  1.6653 |      3.98 | 0.0001 | 0.1278 | 177 |
| pmi         | level  | pmi_level_lag2        | -0.0606 |     -0.11 | 0.9105 | 0.1278 | 177 |
| pmi         | level  | pmi_level_lag3        |  0.1771 |      0.33 | 0.7392 | 0.1278 | 177 |

## 6. Out-of-sample forecast evaluation

| signal             |   n_oos |   rmse_model |   rmse_random_walk |   rmse_hist_mean |   hit_rate_model |   dm_stat_vs_rw |   dm_p |
|:-------------------|--------:|-------------:|-------------------:|-----------------:|-----------------:|----------------:|-------:|
| new_orders (level) |     118 |        4.725 |              4.922 |            4.849 |            0.559 |           -1.15 | 0.2493 |

## 7. Signal backtest (position for month t = sign of the chosen signal at t-1: `change` form = last month's rise/fall, `level` form = index above/below 50; 10 bp per position change)

| signal             |   months |   ann_return_net |   ann_vol |   sharpe_net |   sharpe_buy_hold |   max_drawdown |   hit_rate |   months_in_market |
|:-------------------|---------:|-----------------:|----------:|-------------:|------------------:|---------------:|-----------:|-------------------:|
| new_orders (level) |      179 |           0.2254 |    0.1559 |         1.45 |             -0.58 |        -0.1987 |       0.62 |                173 |

## Caveats

- Monthly macro data: a 10-year sample is ~120 observations; treat p-values between 0.01 and 0.1 as suggestive.
- Copper is driven by USD, inventories, policy and supply; PMI is one driver. The VAR and HAC regressions control for the indicator's own dynamics only.
- The backtest is a sanity check of economic significance, not a strategy: no slippage model, no sizing, no risk management.
- Figures: levels.png, lead_lag.png, var_irf.png, oos_forecast.png, backtest.png.