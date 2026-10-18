#!/usr/bin/env python
"""Copper futures vs. China manufacturing PMI: a time-series study of the lead-lag relationship.

Replaces the 12-month OLS notebook from the CITIC Futures internship with the standard toolkit for
this question, applied to ten-plus years of monthly data:

  1. stationarity (ADF, KPSS) on levels and first differences;
  2. lead-lag cross-correlation between PMI changes (headline + New Orders) and copper returns;
  3. Granger causality in both directions, lags 1-6;
  4. a VAR with information-criterion lag selection and impulse responses;
  5. a lagged-regressor model of copper returns with Newey-West (HAC) standard errors;
  6. out-of-sample expanding-window forecasts vs. the random-walk benchmark (RMSE, hit rate,
     Diebold-Mariano test);
  7. a transaction-cost-aware signal backtest.

Usage
  python copper_pmi_ts.py --pmi data/pmi.csv --copper data/copper.csv --out results
  python copper_pmi_ts.py --demo --out results_demo        # synthetic data with a planted 1-month lead
  python copper_pmi_ts.py --fetch --out results            # best-effort download (see README)

Input formats (monthly; the date is any day of the month)
  pmi.csv     date,pmi,new_orders[,production,new_export_orders,...]   NBS manufacturing PMI and sub-indices (%)
  copper.csv  date,price                                                 monthly average or month-end copper price
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MAX_LAG = 6


# ----------------------------------------------------------------------------- data
def load_inputs(pmi_path: str, copper_path: str) -> pd.DataFrame:
    pmi = pd.read_csv(pmi_path)
    cu = pd.read_csv(copper_path)
    for d in (pmi, cu):
        d.columns = [c.strip().lower() for c in d.columns]
        d["date"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp("M")
    cu = cu.rename(columns={"close": "price"})[["date", "price"]]
    df = pmi.merge(cu, on="date", how="inner").sort_values("date").set_index("date")
    return df.dropna(subset=["pmi", "price"])


def demo_data(n_months: int = 180, seed: int = 7) -> pd.DataFrame:
    """Synthetic data with a KNOWN structure, to prove the pipeline recovers it:
    New Orders is an AR(1) around 50; headline PMI follows New Orders with noise; copper log
    returns respond to LAST month's New Orders deviation (a planted 1-month lead)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-31", periods=n_months, freq="ME")
    no = np.empty(n_months)
    no[0] = 50.5
    for t in range(1, n_months):
        no[t] = 50 + 0.8 * (no[t - 1] - 50) + rng.normal(0, 0.9)
    pmi = 50 + 0.7 * (no - 50) + rng.normal(0, 0.5, n_months)
    production = 50 + 0.5 * (no - 50) + rng.normal(0, 1.0, n_months)          # co-moves, no extra lead
    export_orders = 49 + rng.normal(0, 1.2, n_months)                           # unrelated
    inventories = 48 - 0.2 * (no - 50) + rng.normal(0, 0.8, n_months)           # weakly counter-cyclical
    ret = np.zeros(n_months)
    for t in range(1, n_months):
        ret[t] = 0.012 * (no[t - 1] - 50) + rng.normal(0, 0.045)
    price = 55000 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"pmi": pmi.round(1), "new_orders": no.round(1), "production": production.round(1),
                         "new_export_orders": export_orders.round(1), "raw_material_inventories": inventories.round(1),
                         "price": price.round(0)}, index=dates)


def fetch_data() -> pd.DataFrame:
    """Best effort: NBS PMI via akshare, global copper price via FRED (pandas_datareader).
    The New Orders sub-index is not in every akshare table; put it in data/pmi.csv if missing."""
    try:
        import akshare as ak
        raw = ak.macro_china_pmi()
        pmi = pd.DataFrame({"date": pd.to_datetime(raw["月份"].astype(str).str.replace("年", "-").str.replace("月份", ""), errors="coerce"),
                            "pmi": pd.to_numeric(raw["制造业-指数"], errors="coerce")}).dropna()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"PMI download failed ({type(e).__name__}: {e}); supply data/pmi.csv instead")
    try:
        from pandas_datareader import data as pdr
        cu = pdr.DataReader("PCOPPUSDM", "fred").rename(columns={"PCOPPUSDM": "price"}).reset_index().rename(columns={"DATE": "date"})
    except Exception as e:  # noqa: BLE001
        sys.exit(f"copper download failed ({type(e).__name__}: {e}); supply data/copper.csv instead")
    for d in (pmi, cu):
        d["date"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp("M")
    df = pmi.merge(cu, on="date").sort_values("date").set_index("date")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_price"] = np.log(df["price"])
    df["ret"] = 100 * df["log_price"].diff()                    # monthly log return, %
    df["d_pmi"] = df["pmi"].diff()
    df["pmi_dev"] = df["pmi"] - 50
    if "new_orders" in df:
        df["d_no"] = df["new_orders"].diff()
        df["no_dev"] = df["new_orders"] - 50
    return df


def indicators(df: pd.DataFrame) -> list[str]:
    return [c for c in ("new_orders", "pmi") if c in df]


FORMS = ("change", "level")


def series_for(df: pd.DataFrame, ind: str, form: str) -> pd.Series:
    """`change` = month-on-month change of the diffusion index; `level` = index minus 50 (expansion /
    contraction). A diffusion index is usually stationary, so both are legitimate regressors for returns;
    which one carries the information is an empirical question, so every test looks at both."""
    return df[ind].diff() if form == "change" else df[ind] - 50.0


# ----------------------------------------------------------------------------- 1. stationarity
def stationarity(df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller, kpss
    rows = []
    series = {"log copper price": df["log_price"], "copper return": df["ret"], "PMI": df["pmi"], "dPMI": df["d_pmi"]}
    if "new_orders" in df:
        series["New Orders"] = df["new_orders"]
        series["dNew Orders"] = df["d_no"]
    for name, s in series.items():
        s = s.dropna()
        adf_p = adfuller(s, autolag="AIC")[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_p = kpss(s, regression="c", nlags="auto")[1]
        verdict = "stationary" if (adf_p < 0.05 and kpss_p > 0.05) else ("non-stationary" if (adf_p >= 0.05 and kpss_p <= 0.05) else "ambiguous")
        rows.append({"series": name, "ADF p": round(adf_p, 4), "KPSS p": round(kpss_p, 4), "verdict": verdict, "n": len(s)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2. lead-lag cross-correlation
def lead_lag(df: pd.DataFrame, max_lag: int = MAX_LAG) -> pd.DataFrame:
    """corr(indicator at t-k, copper return at t) for k = -max_lag..max_lag and both forms.
    k > 0: the indicator LEADS copper by k months; k < 0: copper leads the indicator."""
    rows = []
    for ind in indicators(df):
        for form in FORMS:
            x = series_for(df, ind, form)
            for k in range(-max_lag, max_lag + 1):
                pair = pd.concat([x.shift(k), df["ret"]], axis=1).dropna()
                r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                n = len(pair)
                band = 1.96 / math.sqrt(n)
                rows.append({"indicator": ind, "form": form, "lag_k": k, "corr": round(r, 3), "n": n, "ci95": round(band, 3), "significant": abs(r) > band})
    return pd.DataFrame(rows)


def subindex_ranking(df: pd.DataFrame, lead: int = 1) -> pd.DataFrame:
    """Every PMI component in the file, in both forms, compared on the same footing: correlation of the
    indicator at t-lead with copper return at t, and the Granger p-value at that lag. The top row is the
    leading indicator (and form) the data supports."""
    from statsmodels.tsa.stattools import grangercausalitytests
    derived = {"price", "log_price", "ret", "d_pmi", "pmi_dev", "d_no", "no_dev"}
    cands = [c for c in df.columns if c not in derived]
    rows = []
    for c in cands:
        for form in FORMS:
            x = series_for(df, c, form)
            pair = pd.concat([x.shift(lead), df["ret"]], axis=1).dropna()
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            try:
                g = grangercausalitytests(pd.concat([df["ret"], x], axis=1).dropna(), maxlag=lead)[lead][0]["ssr_ftest"][1]
            except Exception:  # noqa: BLE001
                g = float("nan")
            rows.append({"indicator": c, "form": form, "corr_lead": round(r, 3), "granger_p": round(g, 4), "n": len(pair)})
    return pd.DataFrame(rows).sort_values("corr_lead", ascending=False).reset_index(drop=True)


def granger(df: pd.DataFrame, max_lag: int = MAX_LAG) -> pd.DataFrame:
    from statsmodels.tsa.stattools import grangercausalitytests
    rows = []
    for ind in indicators(df):
        for form in FORMS:
            x = series_for(df, ind, form).rename("x")
            for cause, effect in (("x", "ret"), ("ret", "x")):
                data = pd.concat([df["ret"], x], axis=1).dropna()[[effect, cause]]
                res = grangercausalitytests(data, maxlag=max_lag)
                label = f"{ind} ({form}) -> copper" if cause == "x" else f"copper -> {ind} ({form})"
                for lag, r in res.items():
                    rows.append({"direction": label, "lag": lag, "F p-value": round(r[0]["ssr_ftest"][1], 4)})
    return pd.DataFrame(rows)


def var_model(df: pd.DataFrame, out: Path, max_lag: int = MAX_LAG) -> dict:
    from statsmodels.tsa.api import VAR
    cols = ["ret"] + (["d_no"] if "new_orders" in df else ["d_pmi"])
    data = df[cols].dropna()
    model = VAR(data)
    sel = model.select_order(maxlags=max_lag)
    p = int(sel.bic) if sel.bic and sel.bic > 0 else 1
    fit = model.fit(p)
    irf = fit.irf(12)
    try:
        import matplotlib
        matplotlib.use("Agg")
        fig = irf.plot(orth=True, impulse=cols[1], response="ret")
        fig.suptitle(f"Response of copper return to a one-s.d. shock in {cols[1]} (VAR({p}))")
        fig.savefig(out / "var_irf.png", dpi=130, bbox_inches="tight")
    except Exception:  # noqa: BLE001
        pass
    coef = fit.params["ret"].to_dict()
    return {"lag_order_bic": p, "aic": round(float(fit.aic), 3), "bic": round(float(fit.bic), 3),
            "ret_equation": {k: round(float(v), 4) for k, v in coef.items()},
            "irf_ret_to_indicator": [round(float(v), 4) for v in irf.orth_irfs[:7, 0, 1]]}


# ----------------------------------------------------------------------------- 5. lagged regression with HAC
def hac_regression(df: pd.DataFrame, rank: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Copper return on lags 1-3 of each indicator, in the form that ranked higher at lead 1."""
    import statsmodels.api as sm
    rows = []
    for ind in indicators(df):
        form = rank[rank["indicator"] == ind].iloc[0]["form"]
        x = series_for(df, ind, form)
        X = pd.concat({f"{ind}_{form}_lag{k}": x.shift(k) for k in lags}, axis=1)
        data = pd.concat([df["ret"], X], axis=1).dropna()
        fit = sm.OLS(data["ret"], sm.add_constant(data.drop(columns="ret"))).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        for name in fit.params.index:
            rows.append({"indicator": ind, "form": form, "term": name, "coef": round(float(fit.params[name]), 4),
                         "t (HAC)": round(float(fit.tvalues[name]), 2), "p": round(float(fit.pvalues[name]), 4), "R2": round(float(fit.rsquared), 4), "n": int(fit.nobs)})
    return pd.DataFrame(rows)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM statistic for squared-error loss (negative = model 1 better); HAC variance with h-1 lags."""
    from scipy import stats
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    mean = d.mean()
    gamma0 = ((d - mean) ** 2).sum() / n
    var = gamma0
    for k in range(1, h):
        gk = ((d[k:] - mean) * (d[:-k] - mean)).sum() / n
        var += 2 * gk
    dm = mean / math.sqrt(var / n) if var > 0 else 0.0
    return dm, 2 * (1 - stats.norm.cdf(abs(dm)))


def choose_signal(df: pd.DataFrame, min_train: int) -> tuple[str, str, float]:
    """Pick indicator and form on the FIRST min_train months only (lowest Granger p at lag 1), so the
    out-of-sample period never informs the choice."""
    from statsmodels.tsa.stattools import grangercausalitytests
    best = ("pmi", "change", 1.0)
    train = df.iloc[:min_train]
    for ind in indicators(df):
        for form in FORMS:
            data = pd.concat([train["ret"], series_for(train, ind, form).rename("x")], axis=1).dropna()
            try:
                pv = grangercausalitytests(data, maxlag=1)[1][0]["ssr_ftest"][1]
            except Exception:  # noqa: BLE001
                continue
            if pv < best[2]:
                best = (ind, form, float(pv))
    return best


def oos_forecast(df: pd.DataFrame, ind: str, form: str, min_train: int = 60) -> tuple[pd.DataFrame, dict]:
    import statsmodels.api as sm
    x = series_for(df, ind, form)
    data = pd.DataFrame({"ret": df["ret"], "x1": x.shift(1), "x2": x.shift(2)}).dropna()
    preds = []
    for i in range(min_train, len(data)):
        train, test = data.iloc[:i], data.iloc[i]
        fit = sm.OLS(train["ret"], sm.add_constant(train[["x1", "x2"]])).fit()
        pred = float(fit.params["const"] + fit.params["x1"] * test["x1"] + fit.params["x2"] * test["x2"])
        preds.append({"date": data.index[i], "actual": test["ret"], "model": pred, "random_walk": 0.0, "hist_mean": float(train["ret"].mean())})
    p = pd.DataFrame(preds).set_index("date")
    e_model = (p["actual"] - p["model"]).to_numpy()
    e_rw = (p["actual"] - p["random_walk"]).to_numpy()
    e_mean = (p["actual"] - p["hist_mean"]).to_numpy()
    dm, dm_p = diebold_mariano(e_model, e_rw)
    stats = {"signal": f"{ind} ({form})", "n_oos": len(p),
             "rmse_model": round(float(np.sqrt((e_model ** 2).mean())), 3),
             "rmse_random_walk": round(float(np.sqrt((e_rw ** 2).mean())), 3),
             "rmse_hist_mean": round(float(np.sqrt((e_mean ** 2).mean())), 3),
             "hit_rate_model": round(float((np.sign(p["model"]) == np.sign(p["actual"])).mean()), 3),
             "dm_stat_vs_rw": round(dm, 2), "dm_p": round(dm_p, 4)}
    return p, stats


def backtest(df: pd.DataFrame, ind: str, form: str, cost_bps: float = 10.0) -> tuple[pd.DataFrame, dict]:
    """Position for month t = sign of the chosen indicator series at t-1 (known before the month starts):
    `change` form = long after a rise, short after a fall; `level` form = long above 50, short below.
    Costs charged per unit of position change."""
    sig = np.sign(series_for(df, ind, form).shift(1)).fillna(0)
    gross = sig * df["ret"] / 100
    turnover = sig.diff().abs().fillna(0)
    net = gross - turnover * cost_bps / 1e4
    bt = pd.DataFrame({"signal": sig, "ret": df["ret"] / 100, "gross": gross, "net": net}).dropna()
    bt["equity"] = (1 + bt["net"]).cumprod()
    bt["buy_hold"] = (1 + bt["ret"]).cumprod()
    ann = 12
    def sharpe(r):
        return float(r.mean() / r.std() * math.sqrt(ann)) if r.std() > 0 else 0.0
    dd = float((bt["equity"] / bt["equity"].cummax() - 1).min())
    stats = {"signal": f"{ind} ({form})", "months": len(bt), "ann_return_net": round(float(bt["net"].mean() * ann), 4), "ann_vol": round(float(bt["net"].std() * math.sqrt(ann)), 4),
             "sharpe_net": round(sharpe(bt["net"]), 2), "sharpe_buy_hold": round(sharpe(bt["ret"]), 2), "max_drawdown": round(dd, 4),
             "hit_rate": round(float((np.sign(bt["gross"]) > 0).mean()), 3), "months_in_market": int((bt["signal"] != 0).sum())}
    return bt, stats


def figures(df: pd.DataFrame, ll: pd.DataFrame, oos: pd.DataFrame, bt: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["price"], color="tab:blue", lw=1.6, label="copper price")
    ax.set_ylabel("copper price")
    ax2 = ax.twinx()
    for ind, c in (("new_orders", "tab:red"), ("pmi", "tab:orange")):
        if ind in df:
            ax2.plot(df.index, df[ind], color=c, lw=1.2, alpha=0.8, label=ind)
    ax2.axhline(50, color="grey", ls="--", lw=0.8)
    ax2.set_ylabel("PMI (%)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    ax.set_title("Copper price and manufacturing PMI")
    fig.savefig(out / "levels.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    band = ll["ci95"].iloc[0]
    for ind, c in (("new_orders", "tab:red"), ("pmi", "tab:orange")):
        for form, ls in (("level", "-"), ("change", "--")):
            sub = ll[(ll["indicator"] == ind) & (ll["form"] == form)]
            if len(sub):
                ax.plot(sub["lag_k"], sub["corr"], marker="o", ms=4, ls=ls, color=c, label=f"{ind} {form} (t-k) vs copper return (t)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhspan(-band, band, color="grey", alpha=0.15, label="95% band (white noise)")
    ax.set_xlabel("k (months); k > 0 = indicator leads copper")
    ax.set_ylabel("correlation")
    ax.set_title("Lead-lag cross-correlation")
    ax.legend(fontsize=8)
    fig.savefig(out / "lead_lag.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(oos.index, oos["actual"], color="grey", lw=1, label="actual monthly return (%)")
    ax.plot(oos.index, oos["model"], color="tab:red", lw=1.4, label="model forecast")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Out-of-sample forecasts (expanding window)")
    ax.legend()
    fig.savefig(out / "oos_forecast.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(bt.index, bt["equity"], color="tab:red", lw=1.6, label="signal strategy (net of costs)")
    ax.plot(bt.index, bt["buy_hold"], color="tab:blue", lw=1.2, label="buy and hold")
    ax.set_title("Signal backtest: equity curves")
    ax.legend()
    fig.savefig(out / "backtest.png", dpi=130, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- report
def write_report(out: Path, df: pd.DataFrame, st: pd.DataFrame, ll: pd.DataFrame, gr: pd.DataFrame, var: dict, hac: pd.DataFrame,
                 oos_stats: dict, bt_stats: dict, demo: bool, rank: pd.DataFrame) -> None:
    top = rank.iloc[0]
    ll_lead = ll[ll["form"] == top["form"]]
    best = ll[ll["lag_k"] > 0].sort_values("corr", ascending=False).iloc[0]
    g_ind = gr[gr["direction"].str.endswith("-> copper")]
    g_best = g_ind.sort_values("F p-value").iloc[0]
    lines = [f"# Copper vs PMI: time-series results{' (DEMO DATA)' if demo else ''}", "",
             f"Sample: {df.index[0]:%Y-%m} to {df.index[-1]:%Y-%m}, {len(df)} months. Indicators: {', '.join(indicators(df))}.", "",
             f"**Headline.** Among the PMI components, {top['indicator']} ({top['form']}) ranks first as a one-month leading indicator of copper returns "
             f"(corr {top['corr_lead']} at lead 1, Granger p={top['granger_p']}). Across horizons, the strongest positive lead is {best['indicator']} ({best['form']}) at k={int(best['lag_k'])} month(s) ahead of copper returns "
             f"(corr {best['corr']}, 95% band +-{best['ci95']}, {'significant' if best['significant'] else 'NOT significant'}). "
             f"Granger test {g_best['direction']} is most significant at lag {int(g_best['lag'])} (p={g_best['F p-value']}). "
             f"Out of sample the model's RMSE is {oos_stats['rmse_model']} vs {oos_stats['rmse_random_walk']} for the random walk "
             f"(Diebold-Mariano p={oos_stats['dm_p']}, hit rate {oos_stats['hit_rate_model']}).", "",
             "## 1. Stationarity", "", st.to_markdown(index=False), "",
             "Returns and PMI changes should be stationary; price and PMI levels usually are not, which is why the analysis runs on changes.", "",
             "## 2. Lead-lag cross-correlation", "", ll.assign(series=ll["indicator"] + " " + ll["form"]).pivot(index="lag_k", columns="series", values="corr").to_markdown(), "",
             f"k > 0 means the indicator leads copper. Values outside +-{ll['ci95'].iloc[0]} are outside the white-noise band.", "",
             "## 2b. Which PMI component leads copper by one month?", "", rank.to_markdown(index=False), "",
             "All components are compared on the same footing (indicator at t-1 vs copper return at t, in both forms); the top row is the indicator the data supports.", "",
             "## 3. Granger causality (F-test p-values)", "", gr.pivot(index="lag", columns="direction", values="F p-value").to_markdown(), "",
             "## 4. VAR", "", f"Lag order by BIC: {var['lag_order_bic']}; AIC {var['aic']}, BIC {var['bic']}.", "",
             "Copper-return equation coefficients: " + ", ".join(f"{k}={v}" for k, v in var["ret_equation"].items()), "",
             "Orthogonalised impulse response of copper return to a one-s.d. indicator shock, months 0-6: " + ", ".join(map(str, var["irf_ret_to_indicator"])), "",
             "## 5. Lagged regression with Newey-West standard errors", "", hac.to_markdown(index=False), "",
             "## 6. Out-of-sample forecast evaluation", "", pd.DataFrame([oos_stats]).to_markdown(index=False), "",
             "## 7. Signal backtest (position for month t = sign of the chosen signal at t-1: `change` form = last month's rise/fall, `level` form = index above/below 50; 10 bp per position change)", "", pd.DataFrame([bt_stats]).to_markdown(index=False), "",
             "## Caveats", "",
             "- Monthly macro data: a 10-year sample is ~120 observations; treat p-values between 0.01 and 0.1 as suggestive.",
             "- Copper is driven by USD, inventories, policy and supply; PMI is one driver. The VAR and HAC regressions control for the indicator's own dynamics only.",
             "- The backtest is a sanity check of economic significance, not a strategy: no slippage model, no sizing, no risk management.",
             "- Figures: levels.png, lead_lag.png, var_irf.png, oos_forecast.png, backtest.png."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmi", help="CSV: date,pmi[,new_orders,...]")
    ap.add_argument("--copper", help="CSV: date,price")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data with a planted 1-month lead")
    ap.add_argument("--fetch", action="store_true", help="best-effort download (akshare PMI, FRED copper)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.demo:
        df = demo_data()
    elif a.fetch:
        df = fetch_data()
    elif a.pmi and a.copper:
        df = load_inputs(a.pmi, a.copper)
    else:
        ap.error("give --pmi and --copper, or --demo, or --fetch")
    df = prepare(df)
    if len(df) < 36:
        sys.exit(f"only {len(df)} months of overlapping data; need at least 36 (ideally 120+) for these tests")
    df.to_csv(out / "panel.csv")
    st = stationarity(df); st.to_csv(out / "stationarity.csv", index=False)
    ll = lead_lag(df); ll.to_csv(out / "lead_lag.csv", index=False)
    rank = subindex_ranking(df); rank.to_csv(out / "subindex_ranking.csv", index=False)
    gr = granger(df); gr.to_csv(out / "granger.csv", index=False)
    var = var_model(df, out); (out / "var.json").write_text(json.dumps(var, indent=1))
    hac = hac_regression(df, rank); hac.to_csv(out / "hac_regression.csv", index=False)
    ind, form, pv = choose_signal(df, min_train=60)
    oos, oos_stats = oos_forecast(df, ind, form); oos.to_csv(out / "oos_forecast.csv")
    bt, bt_stats = backtest(df, ind, form, a.cost_bps); bt.to_csv(out / "backtest.csv")
    figures(df, ll, oos, bt, out)
    write_report(out, df, st, ll, gr, var, hac, oos_stats, bt_stats, a.demo, rank)
    print((out / "report.md").read_text(encoding="utf-8")[:2500])
    print(f"\n... full report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
