# Copper vs China PMI, v2: tradable out-of-sample results

Daily COMEX copper futures (HG=F) and NBS manufacturing PMI sub-indices. Release-aligned months 2005-02 to 2026-09; out-of-sample from 2010-02 after a 60-month burn-in.

## 1. Averaging bias

Returns computed from monthly-average prices are autocorrelated by construction (Working, 1960) and inflate apparent predictability; v2 uses tradable returns.

| return definition              |   AR(1) autocorrelation |   corr with New Orders level (m-1) |
|:-------------------------------|------------------------:|-----------------------------------:|
| monthly-average price (v1)     |                   0.336 |                              0.231 |
| release-aligned, tradable (v2) |                   0.103 |                              0.156 |

## 2. Out-of-sample forecasts vs the prevailing historical mean

OOS R^2 > 0 means the forecast beats the historical mean (Campbell & Thompson, 2008). Clark & West (2007) tests the nested comparison; p < 0.10 is the usual threshold in this literature. `combo` = mean of all univariate forecasts (Rapach, Strauss & Zhou, 2010); `combo_ct` adds the sign restriction.

| forecast                        |   n_oos |   oos_r2_pct |   clark_west_t |   cw_p_one_sided |   direction_hit_rate |
|:--------------------------------|--------:|-------------:|---------------:|-----------------:|---------------------:|
| pmi_level                       |     200 |         1.06 |           1.16 |           0.1226 |                0.49  |
| production_level                |     200 |         1.06 |           1.18 |           0.12   |                0.5   |
| combo                           |     200 |         0.88 |           0.85 |           0.1981 |                0.545 |
| combo_ct                        |     200 |         0.88 |           0.85 |           0.1981 |                0.545 |
| new_orders_level                |     200 |         0.52 |           1.19 |           0.1168 |                0.485 |
| new_export_orders_level         |     200 |         0.07 |           0.74 |           0.2304 |                0.515 |
| raw_material_inventories_level  |     200 |        -0.44 |          -0.92 |           0.8224 |                0.525 |
| new_orders_change               |     200 |        -0.65 |           0.38 |           0.352  |                0.525 |
| production_change               |     200 |        -0.71 |           0.16 |           0.4381 |                0.525 |
| pmi_change                      |     200 |        -0.78 |           0.42 |           0.3358 |                0.52  |
| raw_material_inventories_change |     200 |        -1.37 |          -0.23 |           0.5892 |                0.515 |
| new_export_orders_change        |     200 |        -1.5  |           1.02 |           0.155  |                0.56  |

**Reading.** The combination forecast has OOS R^2 of 0.88% (Clark-West p = 0.1981). Monthly OOS R^2 of 0.5% or more is economically meaningful for a mean-variance investor (Campbell & Thompson, 2008).

## 3. Strategies (same out-of-sample months, 10 bp per unit of turnover, rf = 0)

| strategy                     | CAGR   | ann_vol   |   Sharpe | max_drawdown   | hit_rate   |   avg_turnover |   Sharpe to 2018-06 |   Sharpe from 2018-06 |
|:-----------------------------|:-------|:----------|---------:|:---------------|:-----------|---------------:|--------------------:|----------------------:|
| Buy & hold                   | 4.6%   | 22.6%     |     0.31 | -55.2%         | 52.5%      |           0    |                0.12 |                  0.5  |
| Buy & hold, vol-targeted     | 1.8%   | 10.6%     |     0.22 | -37.3%         | 52.5%      |           0.05 |                0.01 |                  0.44 |
| v1 signal: New Orders > 50   | -0.8%  | 22.6%     |     0.08 | -69.3%         | 51.3%      |           0.42 |               -0.09 |                  0.24 |
| PMI combination (RSZ + CT)   | 0.9%   | 22.7%     |     0.16 | -54.1%         | 54.5%      |           0.56 |                0.15 |                  0.17 |
| 12m momentum (TSMOM)         | 1.5%   | 22.7%     |     0.18 | -63.4%         | 51.0%      |           0.2  |               -0    |                  0.36 |
| PMI + momentum 50/50         | 2.2%   | 18.4%     |     0.21 | -46.6%         | 54.3%      |           0.35 |                0.09 |                  0.3  |
| PMI + momentum, vol-targeted | 1.2%   | 8.4%      |     0.19 | -25.8%         | 54.3%      |           0.19 |                0.05 |                  0.3  |

Figure: `nav_v2.png`.

## Caveats

- HG=F is the continuous front-month contract from Yahoo Finance; roll returns are approximated by the continuous series.
- hit_rate counts only months with a non-zero position (the 50/50 blend is flat when the two signals disagree).
- Sharpe ratios use rf = 0; ~200 monthly observations give a Sharpe standard error of roughly 0.25.
- Parameters were fixed in advance at literature values; no parameter was tuned on this sample.

## References

- Campbell, J. Y. and Thompson, S. B. (2008). Predicting excess stock returns out of sample: can anything beat the historical average? Review of Financial Studies 21(4).
- Clark, T. E. and West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. Journal of Econometrics 138(1).
- Moreira, A. and Muir, T. (2017). Volatility-managed portfolios. Journal of Finance 72(4).
- Moskowitz, T. J., Ooi, Y. H. and Pedersen, L. H. (2012). Time series momentum. Journal of Financial Economics 104(2).
- Rapach, D. E., Strauss, J. K. and Zhou, G. (2010). Out-of-sample equity premium prediction: combination forecasts and links to the real economy. Review of Financial Studies 23(2).
- Welch, I. and Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. Review of Financial Studies 21(4).
- Working, H. (1960). Note on the correlation of first differences of averages in a random chain. Econometrica 28(4).