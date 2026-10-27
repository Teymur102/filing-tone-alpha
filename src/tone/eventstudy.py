"""Market-model event study.

Day 0 is the first trading session in which investors could trade on the release:
the filing date if it was accepted before 16:00 ET, otherwise the next trading day.
Abnormal return AR_t = r_t - (alpha + beta * r_m,t), with alpha/beta estimated by OLS
over trading days [-250, -30] relative to day 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def event_day_index(dates: pd.DatetimeIndex, accepted: pd.Timestamp) -> int | None:
    """Position of day 0 in the trading calendar `dates` (sorted). `accepted` is in US/Eastern."""
    day = accepted.normalize()
    after_close = accepted.hour >= 16
    pos = dates.searchsorted(day, side="right" if after_close else "left")
    return int(pos) if pos < len(dates) else None


def market_model(ri: np.ndarray, rm: np.ndarray) -> tuple[float, float, int]:
    """OLS of stock on market returns, ignoring days where either is missing."""
    ok = ~(np.isnan(ri) | np.isnan(rm))
    n = int(ok.sum())
    if n < 3:
        return np.nan, np.nan, n
    x, y = rm[ok], ri[ok]
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha = y.mean() - beta * x.mean()
    return float(alpha), float(beta), n


def abnormal_returns(ri: np.ndarray, rm: np.ndarray, t0: int,
                     est=C.EST_WINDOW, path=C.PATH_RANGE, min_obs=C.MIN_EST_OBS):
    """Return (alpha, beta, AR path over `path` as a dict {k: AR_k}) for one event, or None."""
    lo, hi = t0 + est[0], t0 + est[1]
    if lo < 0 or t0 + path[1] >= len(ri):
        return None
    a, b, n = market_model(ri[lo:hi + 1], rm[lo:hi + 1])
    if n < min_obs:
        return None
    ks = np.arange(path[0], path[1] + 1)
    ar = ri[t0 + ks] - (a + b * rm[t0 + ks])
    return a, b, dict(zip(ks.tolist(), ar.tolist()))


def car(ar_path: dict, window: tuple[int, int]) -> float:
    """Cumulative abnormal return: sum of AR_k for k in [window]; NaN if any day missing."""
    vals = [ar_path.get(k, np.nan) for k in range(window[0], window[1] + 1)]
    return float(np.sum(vals)) if not np.any(np.isnan(vals)) else np.nan


def bhr(r: np.ndarray, start: int, end: int) -> float:
    """Buy-and-hold return over positions [start, end]."""
    if start < 0 or end >= len(r):
        return np.nan
    seg = r[start:end + 1]
    return float(np.prod(1 + seg) - 1) if not np.isnan(seg).any() else np.nan


def run(events: pd.DataFrame, rets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute CARs and controls for every event. Returns (event table, AR paths long)."""
    dates = rets.index
    rm = rets[C.MARKET].to_numpy()
    rows, paths = [], []
    for ev in events.itertuples(index=False):
        if ev.ticker not in rets.columns:
            continue
        t0 = event_day_index(dates, ev.accepted_et)
        if t0 is None:
            continue
        ri = rets[ev.ticker].to_numpy()
        res = abnormal_returns(ri, rm, t0)
        if res is None:
            continue
        a, b, ar = res
        row = {"accessionNumber": ev.accessionNumber, "day0": dates[t0], "alpha": a, "beta": b}
        for name, w in C.CAR_WINDOWS.items():
            row[name] = car(ar, w)
        row["mom_12_1"] = bhr(ri, t0 - 252, t0 - 21)
        row["ret_pre20"] = bhr(ri, t0 - 20, t0 - 2)
        row["t0"] = t0
        rows.append(row)
        paths.append(pd.DataFrame({"accessionNumber": ev.accessionNumber, "k": list(ar), "ar": list(ar.values())}))
    return pd.DataFrame(rows), pd.concat(paths, ignore_index=True)
