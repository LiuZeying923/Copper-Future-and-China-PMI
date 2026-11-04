#!/usr/bin/env python
"""Copper vs China PMI, v2: a tradable, out-of-sample version of the study.

What changes relative to copper_pmi_ts.py (v1), and why:

  1. Tradable returns. v1 used monthly-average prices; averaging induces spurious autocorrelation in
     returns (Working, 1960) and nobody can trade at a monthly average. v2 uses daily COMEX copper
     futures closes and release-aligned months: enter at the close of the 2nd trading day of month m
     (the NBS PMI for month m-1 is public by then: it is released on the last day of m-1 or the 1st
     of m), exit at the close of the 2nd trading day of m+1. No look-ahead.
  2. Out-of-sample evaluation against the right benchmark. Expanding-window forecasts are compared with
     the prevailing historical mean (Welch & Goyal, 2008): out-of-sample R^2 (Campbell & Thompson,
     2008) and the Clark & West (2007) test for nested models (Diebold-Mariano is biased against the
     larger model when models are nested).
  3. Forecast combination. The mean of the univariate forecasts from every PMI component and form
     (Rapach, Strauss & Zhou, 2010), with the Campbell-Thompson sign restriction (an activity index
     with a negative estimated slope falls back to the historical mean).
  4. Trend. 12-month time-series momentum on copper itself (Moskowitz, Ooi & Pedersen, 2012),
     blended 50/50 with the PMI signal.
  5. Risk scaling. Positions scaled to a 10% annualised volatility target using trailing 63-day
     realised volatility, capped at 2x (Moreira & Muir, 2017).

All parameters (60-month burn-in, 12-month momentum, 63-day volatility, 10% target, 2x cap, 10 bp per
unit of turnover) are fixed in advance at the values used in those papers, and every variant is
reported, not only the best one.

Usage
  python copper_pmi_v2.py --pmi data/pmi.csv --copper-daily data/copper_daily.csv --out results_v2
  python copper_pmi_v2.py --fetch --out results_v2      # akshare (NBS sub-indices) + yfinance (HG=F)
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INDICATORS = ("pmi", "new_orders", "production", "new_export_orders", "raw_material_inventories")
SIGN_RESTRICTED = {"pmi", "new_orders", "production", "new_export_orders"}   # theory: higher activity -> higher copper
MIN_TRAIN = 60          # months of burn-in before the first out-of-sample forecast
MOM_LOOKBACK = 12       # months, time-series momentum
VOL_WINDOW = 63         # trading days, realised volatility
VOL_TARGET = 0.10       # annualised
LEV_CAP = 2.0


# ----------------------------------------------------------------------------- data
def fetch() -> tuple[pd.DataFrame, pd.Series]:
    import akshare as ak
    import yfinance as yf
    raw = ak.macro_china_nbs_nation(kind="月度数据", path="采购经理指数 > 制造业采购经理指数", period="2005-")
    names = {"制造业采购经理指数(%)": "pmi", "新订单指数(%)": "new_orders", "生产指数(%)": "production",
             "新出口订单指数(%)": "new_export_orders", "原材料库存指数(%)": "raw_material_inventories"}
    pmi = raw.loc[list(names)].T.rename(columns=names)
    pmi.index = pd.to_datetime(pmi.index.str.replace("年", "-").str.replace("月", ""), format="%Y-%m")
    pmi = pmi.apply(pd.to_numeric, errors="coerce").sort_index().dropna()
    px = yf.download("HG=F", start="2000-01-01", auto_adjust=False, progress=False)["Close"].squeeze().dropna()
    return pmi, px


def load(pmi_path: str, copper_daily_path: str) -> tuple[pd.DataFrame, pd.Series]:
    pmi = pd.read_csv(pmi_path, parse_dates=["date"]).set_index("date").sort_index()
    pmi.index = pmi.index.to_period("M").to_timestamp()                  # month start = reference month
    cu = pd.read_csv(copper_daily_path, parse_dates=["date"]).set_index("date")["price"].sort_index().dropna()
    return pmi, cu


def monthly_panel(pmi: pd.DataFrame, px: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Release-aligned monthly returns. Row m: return from the 2nd trading day of m to the 2nd trading day
    of m+1, the predictors are PMI readings for month m-1, and `vol` is realised vol up to the entry date."""
    daily_ret = np.log(px).diff()
    by_month = px.groupby(px.index.to_period("M"))
    entry = by_month.apply(lambda s: s.index[1] if len(s) > 1 else s.index[0])          # 2nd trading day
    rows = []
    months = list(entry.index)
    for i in range(len(months) - 1):
        m, d0, d1 = months[i], entry.iloc[i], entry.iloc[i + 1]
        rv = daily_ret.loc[:d0].iloc[-VOL_WINDOW:]
        rows.append({"month": m.to_timestamp(), "entry": d0, "exit": d1,
                     "ret": 100 * math.log(px.loc[d1] / px.loc[d0]),
                     "vol": float(rv.std() * math.sqrt(252)) if len(rv) == VOL_WINDOW else np.nan})
    df = pd.DataFrame(rows).set_index("month")
    avg = px.groupby(px.index.to_period("M")).mean()
    avg.index = avg.index.to_timestamp()
    avg_ret = 100 * np.log(avg).diff()                                  # v1-style monthly-average return
    for c in INDICATORS:
        if c in pmi:
            df[f"{c}_level"] = (pmi[c] - 50).shift(1).reindex(df.index)      # month m-1 reading
            df[f"{c}_change"] = pmi[c].diff().shift(1).reindex(df.index)
    df["mom12"] = df["ret"].shift(1).rolling(MOM_LOOKBACK).sum()             # months m-12..m-1, all realised before entry
    return df.dropna(subset=[f"pmi_level"]), avg_ret


def predictors(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_level") or c.endswith("_change")]


# ----------------------------------------------------------------------------- forecasts
def oos_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window univariate OLS forecasts r_m = a + b x_{m-1}; prevailing mean benchmark;
    equal-weight combination with and without the Campbell-Thompson sign restriction."""
    cols = predictors(df)
    out = []
    for i in range(MIN_TRAIN, len(df)):
        train, row = df.iloc[:i], df.iloc[i]
        hm = float(train["ret"].mean())
        rec = {"month": df.index[i], "actual": float(row["ret"]), "hist_mean": hm}
        unr, res = [], []
        for c in cols:
            d = train[["ret", c]].dropna()
            if len(d) < 24 or pd.isna(row[c]):
                continue
            x, y = d[c].to_numpy(), d["ret"].to_numpy()
            b = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
            a = float(y.mean() - b * x.mean())
            f = a + b * float(row[c])
            rec[c] = f
            unr.append(f)
            restricted = c.rsplit("_", 1)[0] in SIGN_RESTRICTED and b < 0
            res.append(hm if restricted else f)
        rec["combo"] = float(np.mean(unr))
        rec["combo_ct"] = float(np.mean(res))
        out.append(rec)
    return pd.DataFrame(out).set_index("month")


def clark_west(actual: np.ndarray, f_bench: np.ndarray, f_model: np.ndarray) -> tuple[float, float]:
    """Clark & West (2007) MSPE-adjusted statistic, one-sided p-value, Newey-West (lag 3) s.e."""
    import statsmodels.api as sm
    from scipy import stats
    fadj = (actual - f_bench) ** 2 - ((actual - f_model) ** 2 - (f_bench - f_model) ** 2)
    fit = sm.OLS(fadj, np.ones_like(fadj)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    t = float(fit.tvalues[0])
    return t, float(1 - stats.norm.cdf(t))


def forecast_table(fc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a, hm = fc["actual"].to_numpy(), fc["hist_mean"].to_numpy()
    msfe_hm = float(np.mean((a - hm) ** 2))
    for c in [c for c in fc.columns if c not in ("actual", "hist_mean")]:
        f = fc[c].to_numpy()
        ok = ~np.isnan(f)
        r2 = 1 - float(np.mean((a[ok] - f[ok]) ** 2)) / float(np.mean((a[ok] - hm[ok]) ** 2))
        t, p = clark_west(a[ok], hm[ok], f[ok])
        rows.append({"forecast": c, "n_oos": int(ok.sum()), "oos_r2_pct": round(100 * r2, 2),
                     "clark_west_t": round(t, 2), "cw_p_one_sided": round(p, 4),
                     "direction_hit_rate": round(float((np.sign(f[ok]) == np.sign(a[ok])).mean()), 3)})
    t = pd.DataFrame(rows).sort_values("oos_r2_pct", ascending=False)
    t.attrs["msfe_hist_mean"] = msfe_hm
    return t


# ----------------------------------------------------------------------------- strategies
def strategies(df: pd.DataFrame, fc: pd.DataFrame, cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly positions decided at the entry close; simple returns; cost per unit of position change."""
    d = df.loc[fc.index]
    r = np.exp(d["ret"] / 100) - 1
    scale = (VOL_TARGET / d["vol"]).clip(upper=LEV_CAP).fillna(1.0)
    pmi_sig = np.sign(fc["combo_ct"])
    mom_sig = np.sign(d["mom12"]).fillna(0)
    blend = 0.5 * pmi_sig + 0.5 * mom_sig
    pos = pd.DataFrame({
        "Buy & hold": 1.0,
        "Buy & hold, vol-targeted": scale,
        "v1 signal: New Orders > 50": np.sign(d["new_orders_level"]),
        "PMI combination (RSZ + CT)": pmi_sig,
        "12m momentum (TSMOM)": mom_sig,
        "PMI + momentum 50/50": blend,
        "PMI + momentum, vol-targeted": blend * scale,
    }, index=d.index)
    turnover = pos.diff().abs()
    turnover.iloc[0] = pos.iloc[0].abs()
    net = pos.mul(r, axis=0) - turnover * cost_bps / 1e4
    return net, pos


def perf(net: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    half = net.index[len(net) // 2]
    for c in net.columns:
        x = net[c]
        nav = (1 + x).cumprod()
        yrs = len(x) / 12
        sr = lambda s: float(s.mean() / s.std() * math.sqrt(12)) if s.std() > 0 else 0.0
        rows.append({"strategy": c, "CAGR": nav.iloc[-1] ** (1 / yrs) - 1, "ann_vol": x.std() * math.sqrt(12),
                     "Sharpe": sr(x), "max_drawdown": float((nav / nav.cummax() - 1).min()),
                     "hit_rate": float((x[pos[c] != 0] > 0).mean()), "avg_turnover": float(pos[c].diff().abs().mean()),
                     f"Sharpe to {half:%Y-%m}": sr(x[x.index < half]), f"Sharpe from {half:%Y-%m}": sr(x[x.index >= half])})
    return pd.DataFrame(rows).set_index("strategy")


def plot(net: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = {"Buy & hold": "#9AA5B1", "Buy & hold, vol-targeted": "#C8553D", "v1 signal: New Orders > 50": "#E0A458",
            "PMI + momentum, vol-targeted": "#1F3A5F"}
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for c, col in show.items():
        nav = (1 + net[c]).cumprod()
        ax.plot(nav.index, nav, color=col, lw=1.6 if c.startswith("PMI") else 1.2, label=c)
    ax.set_yscale("log")
    ax.axhline(1, color="#999", lw=0.8, ls="--")
    ax.set_title(f"Out-of-sample, tradable: COMEX copper, release-aligned months {net.index[0]:%Y-%m} to {net.index[-1]:%Y-%m}, 10 bp per unit turnover",
                 fontsize=8.5)
    ax.set_ylabel("NAV (log scale)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "nav_v2.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def averaging_check(df: pd.DataFrame, avg_ret: pd.Series) -> pd.DataFrame:
    a = avg_ret.reindex(df.index).dropna()
    t = df["ret"].reindex(a.index)
    return pd.DataFrame([
        {"return definition": "monthly-average price (v1)", "AR(1) autocorrelation": round(float(a.autocorr(1)), 3),
         "corr with New Orders level (m-1)": round(float(a.corr(df.loc[a.index, "new_orders_level"])), 3)},
        {"return definition": "release-aligned, tradable (v2)", "AR(1) autocorrelation": round(float(t.autocorr(1)), 3),
         "corr with New Orders level (m-1)": round(float(t.corr(df.loc[a.index, "new_orders_level"])), 3)}])


def write_report(out: Path, df: pd.DataFrame, avg_tab: pd.DataFrame, ftab: pd.DataFrame, ptab: pd.DataFrame) -> None:
    fmt = ptab.copy()
    for c in fmt.columns:
        if c in ("CAGR", "ann_vol", "max_drawdown", "hit_rate"):
            fmt[c] = (100 * fmt[c]).round(1).astype(str) + "%"
        else:
            fmt[c] = fmt[c].round(2)
    combo = ftab.set_index("forecast").loc["combo_ct"]
    lines = [
        "# Copper vs China PMI, v2: tradable out-of-sample results", "",
        f"Daily COMEX copper futures (HG=F) and NBS manufacturing PMI sub-indices. Release-aligned months "
        f"{df.index[0]:%Y-%m} to {df.index[-1]:%Y-%m}; out-of-sample from {ptab.attrs['start']:%Y-%m} after a {MIN_TRAIN}-month burn-in.", "",
        "## 1. Averaging bias", "",
        "Returns computed from monthly-average prices are autocorrelated by construction (Working, 1960) and inflate "
        "apparent predictability; v2 uses tradable returns.", "", avg_tab.to_markdown(index=False), "",
        "## 2. Out-of-sample forecasts vs the prevailing historical mean", "",
        "OOS R^2 > 0 means the forecast beats the historical mean (Campbell & Thompson, 2008). Clark & West (2007) "
        "tests the nested comparison; p < 0.10 is the usual threshold in this literature. `combo` = mean of all "
        "univariate forecasts (Rapach, Strauss & Zhou, 2010); `combo_ct` adds the sign restriction.", "",
        ftab.to_markdown(index=False), "",
        f"**Reading.** The combination forecast has OOS R^2 of {combo['oos_r2_pct']}% (Clark-West p = {combo['cw_p_one_sided']}). "
        "Monthly OOS R^2 of 0.5% or more is economically meaningful for a mean-variance investor (Campbell & Thompson, 2008).", "",
        "## 3. Strategies (same out-of-sample months, 10 bp per unit of turnover, rf = 0)", "",
        fmt.reset_index().to_markdown(index=False), "",
        "Figure: `nav_v2.png`.", "",
        "## Caveats", "",
        "- HG=F is the continuous front-month contract from Yahoo Finance; roll returns are approximated by the continuous series.",
        "- hit_rate counts only months with a non-zero position (the 50/50 blend is flat when the two signals disagree).",
        "- Sharpe ratios use rf = 0; ~200 monthly observations give a Sharpe standard error of roughly 0.25.",
        "- Parameters were fixed in advance at literature values; no parameter was tuned on this sample.",
        "",
        "## References", "",
        "- Campbell, J. Y. and Thompson, S. B. (2008). Predicting excess stock returns out of sample: can anything beat the historical average? Review of Financial Studies 21(4).",
        "- Clark, T. E. and West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. Journal of Econometrics 138(1).",
        "- Moreira, A. and Muir, T. (2017). Volatility-managed portfolios. Journal of Finance 72(4).",
        "- Moskowitz, T. J., Ooi, Y. H. and Pedersen, L. H. (2012). Time series momentum. Journal of Financial Economics 104(2).",
        "- Rapach, D. E., Strauss, J. K. and Zhou, G. (2010). Out-of-sample equity premium prediction: combination forecasts and links to the real economy. Review of Financial Studies 23(2).",
        "- Welch, I. and Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. Review of Financial Studies 21(4).",
        "- Working, H. (1960). Note on the correlation of first differences of averages in a random chain. Econometrica 28(4).",
    ]
    (out / "report_v2.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmi", default="data/pmi.csv")
    ap.add_argument("--copper-daily", default="data/copper_daily.csv")
    ap.add_argument("--fetch", action="store_true", help="download NBS PMI sub-indices (akshare) and HG=F (yfinance)")
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pmi, px = fetch() if a.fetch else load(a.pmi, a.copper_daily)
    df, avg_ret = monthly_panel(pmi, px)
    if len(df) < MIN_TRAIN + 36:
        sys.exit(f"only {len(df)} months; need at least {MIN_TRAIN + 36}")
    fc = oos_forecasts(df)
    ftab = forecast_table(fc)
    net, pos = strategies(df, fc, a.cost_bps)
    ptab = perf(net, pos)
    ptab.attrs["start"] = fc.index[0]
    avg_tab = averaging_check(df, avg_ret)
    df.to_csv(out / "panel_v2.csv")
    fc.to_csv(out / "oos_forecasts.csv")
    ftab.to_csv(out / "forecast_eval.csv", index=False)
    net.to_csv(out / "strategy_returns.csv")
    ptab.to_csv(out / "strategy_performance.csv")
    plot(net, out)
    write_report(out, df, avg_tab, ftab, ptab)
    print((out / "report_v2.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
