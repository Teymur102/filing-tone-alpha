"""Statistical tests: cross-sectional regressions, tercile sorts, and an out-of-sample trading test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C

SIGNALS = ["lm_tone", "finbert_tone", "lm_tone_chg", "finbert_tone_chg"]
SIGNAL_LABELS = {
    "lm_tone": "LM tone (level)",
    "finbert_tone": "FinBERT tone (level)",
    "lm_tone_chg": "LM tone change vs firm's last 4 releases",
    "finbert_tone_chg": "FinBERT tone change vs firm's last 4 releases",
}


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def add_tone_changes(df: pd.DataFrame, n_prior: int = 4, min_prior: int = 2) -> pd.DataFrame:
    """Tone minus the mean of the same firm's previous `n_prior` releases (uses only past data)."""
    df = df.sort_values(["ticker", "day0"]).copy()
    for col in ["lm_tone", "finbert_tone"]:
        prior = df.groupby("ticker")[col].transform(
            lambda s: s.shift(1).rolling(n_prior, min_periods=min_prior).mean())
        df[f"{col}_chg"] = df[col] - prior
    return df


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def regress(df: pd.DataFrame, y: str, xs: list[str], controls: list[str]) -> dict:
    """OLS with calendar-quarter fixed effects; SEs two-way clustered by firm and quarter."""
    d = df.dropna(subset=[y, *xs, *controls]).copy()
    X = d[xs + controls].copy()
    for x in xs:
        X[x] = zscore(X[x])                     # coefficient = CAR change per 1 s.d. of tone
    fe = pd.get_dummies(d["yq"], prefix="q", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([X, fe], axis=1))
    groups = np.column_stack([pd.factorize(d["ticker"])[0], pd.factorize(d["yq"])[0]])
    fit = sm.OLS(d[y].astype(float), X.astype(float)).fit(cov_type="cluster", cov_kwds={"groups": groups})
    out = {"y": y, "n": int(fit.nobs), "r2": fit.rsquared}
    for v in xs + controls:
        out[f"b_{v}"], out[f"t_{v}"] = fit.params[v], fit.tvalues[v]
    return out


def trailing_terciles(df: pd.DataFrame, col: str, lookback_days: int = 365, min_obs: int = 60) -> pd.Series:
    """Assign terciles using breakpoints from events in the prior `lookback_days` only (no look-ahead).
    1 = lowest tone, 3 = highest. NaN when there is too little history."""
    d = df[["day0", col]].dropna().sort_values("day0")
    days, vals = d["day0"].to_numpy(), d[col].to_numpy()
    out = pd.Series(np.nan, index=d.index)
    for i, (t, v) in enumerate(zip(days, vals)):
        lo = np.searchsorted(days, t - np.timedelta64(lookback_days, "D"))
        hi = np.searchsorted(days, t)               # strictly earlier days only
        hist = vals[lo:hi]
        if len(hist) < min_obs:
            continue
        q1, q2 = np.quantile(hist, [1 / 3, 2 / 3])
        out.iloc[i] = 1 if v <= q1 else (3 if v > q2 else 2)
    return out.reindex(df.index)


def tercile_table(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    """Mean CAR by tercile and the high-minus-low spread.
    t_stat clusters by calendar quarter (releases in the same season share market shocks and
    overlapping windows); t_welch treats events as independent and is shown for comparison only."""
    rows = []
    for w in C.CAR_WINDOWS:
        d = df.dropna(subset=[tcol, w])
        g = d.groupby(tcol)[w]
        hi, lo = d.loc[d[tcol] == 3, w], d.loc[d[tcol] == 1, w]
        se = np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
        hl = d[d[tcol].isin([1, 3])]
        fit = sm.OLS(hl[w].astype(float), sm.add_constant((hl[tcol] == 3).astype(float))).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(hl["yq"])[0]})
        rows.append({"window": w, "low": g.mean().get(1.0), "mid": g.mean().get(2.0), "high": g.mean().get(3.0),
                     "high_minus_low": hi.mean() - lo.mean(), "t_stat": float(fit.tvalues.iloc[1]),
                     "t_welch": (hi.mean() - lo.mean()) / se, "n_low": len(lo), "n_high": len(hi)})
    return pd.DataFrame(rows)


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    x = np.asarray(x, float)
    fit = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def calendar_portfolio(df: pd.DataFrame, rets: pd.DataFrame, tcol: str, start: str, end: str,
                       hold=(2, 60), cost_bps: float = C.COST_BPS_PER_SIDE) -> pd.DataFrame:
    """Daily long-short portfolio: buy top-tercile names at the close of day +1, short bottom-tercile
    names, hold through day +60, equal weight within each leg (rebalanced daily).
    Cost = cost_bps x daily one-way turnover of each leg."""
    dates = rets.index
    sel = df[(df["day0"] >= start) & (df["day0"] <= end)].dropna(subset=[tcol])
    sel = sel[sel[tcol].isin([1, 3])]
    pos = {1: np.zeros((len(dates), rets.shape[1])), 3: np.zeros((len(dates), rets.shape[1]))}
    col_ix = {c: i for i, c in enumerate(rets.columns)}
    for ev in sel.itertuples(index=False):
        a, b = ev.t0 + hold[0], min(ev.t0 + hold[1], len(dates) - 1)
        pos[int(getattr(ev, tcol))][a:b + 1, col_ix[ev.ticker]] += 1
    R = rets.fillna(0.0).to_numpy()
    out = {}
    for leg in (1, 3):
        n = pos[leg].sum(axis=1, keepdims=True)
        w = np.divide(pos[leg], n, out=np.zeros_like(pos[leg]), where=n > 0)
        out[f"ret_{leg}"] = (w * R).sum(axis=1)
        out[f"turn_{leg}"] = np.abs(np.diff(w, axis=0, prepend=0)).sum(axis=1)
        out[f"n_{leg}"] = n[:, 0]
    p = pd.DataFrame(out, index=dates)
    p["mkt"] = rets[C.MARKET].to_numpy()
    active = (p["n_1"] > 0) & (p["n_3"] > 0)
    p["gross"] = np.where(active, p["ret_3"] - p["ret_1"], 0.0)
    p["cost"] = (p["turn_1"] + p["turn_3"]) * cost_bps / 1e4
    p["net"] = p["gross"] - p["cost"]
    first, last = sel["day0"].min(), sel["day0"].max() + pd.Timedelta(days=95)
    return p.loc[(p.index >= first) & (p.index <= last)]


def portfolio_stats(p: pd.DataFrame, label: str) -> dict:
    out = {"portfolio": label, "days": len(p), "start": p.index.min().date(), "end": p.index.max().date()}
    for k in ("gross", "net"):
        r = p[k].to_numpy()
        out[f"ann_ret_{k}"] = r.mean() * 252
        out[f"ann_vol_{k}"] = r.std(ddof=1) * np.sqrt(252)
        out[f"sharpe_{k}"] = out[f"ann_ret_{k}"] / out[f"ann_vol_{k}"] if out[f"ann_vol_{k}"] > 0 else np.nan
        out[f"nw_t_{k}"] = newey_west_t(r)
    fit = sm.OLS(p["net"], sm.add_constant(p["mkt"])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    out["capm_alpha_net_ann"], out["capm_alpha_net_t"], out["capm_beta"] = (
        fit.params["const"] * 252, fit.tvalues["const"], fit.params["mkt"])
    out["avg_daily_turnover"] = (p["turn_1"] + p["turn_3"]).mean()
    out["avg_names_long"], out["avg_names_short"] = p["n_3"].mean(), p["n_1"].mean()
    return out
