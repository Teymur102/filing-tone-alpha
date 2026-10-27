"""Daily prices (yfinance, keyless) and shares outstanding (SEC XBRL) for size controls."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import requests

from . import config as C
from . import edgar

PRICE_PATH = C.DATA / "prices.parquet"


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """Long frame: date, ticker, adj_close, close (split-adjusted only), split."""
    if PRICE_PATH.exists():
        return pd.read_parquet(PRICE_PATH)
    import yfinance as yf

    raw = yf.download(sorted(set(tickers) | {C.MARKET}), start=C.PRICE_START, auto_adjust=False,
                      actions=True, progress=False, group_by="column", threads=True)
    out = []
    for t in raw["Adj Close"].columns:
        df = pd.DataFrame({
            "adj_close": raw["Adj Close"][t],
            "close": raw["Close"][t],
            "split": raw["Stock Splits"][t].replace(0, np.nan),
        })
        df["ticker"] = t
        out.append(df.dropna(subset=["adj_close"]))
    long = pd.concat(out).rename_axis("date").reset_index()
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
    missing = set(tickers) - set(long.ticker)
    if missing:
        print(f"  [prices] yfinance returned nothing for {sorted(missing)}")
    long.to_parquet(PRICE_PATH)
    return long


def returns_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide daily simple returns from dividend- and split-adjusted closes."""
    px = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return px.pct_change(fill_method=None).iloc[1:]


def unadjusted_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Undo yfinance split adjustment: multiply by the product of all later split ratios."""
    out = {}
    for t, g in prices.sort_values("date").groupby("ticker"):
        split = g["split"].fillna(1.0).to_numpy()
        later = np.cumprod(split[::-1])[::-1]          # product of splits on or after each date
        later = np.append(later[1:], 1.0)              # strictly after
        out[t] = pd.Series(g["close"].to_numpy() * later, index=g["date"].to_numpy())
    return pd.DataFrame(out).sort_index()


def _concept(cik: int, taxonomy: str, tag: str) -> dict | None:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json"
    try:
        return json.loads(edgar._cached(C.RAW_EDGAR / "shares" / f"{cik}_{tag}.json.gz", url))
    except requests.HTTPError as e:            # 404 = company never reported this tag
        if e.response is not None and e.response.status_code == 404:
            return None
        raise


def _share_frame(raw: dict | None) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame(columns=["filed", "shares"])
    rows = [(u["accn"], u["filed"], u["end"], u["val"]) for u in raw["units"].get("shares", [])]
    df = pd.DataFrame(rows, columns=["accn", "filed", "end", "val"])
    # several classes reported in one filing at the same date -> add them up
    df = df.groupby(["accn", "filed", "end"], as_index=False)["val"].sum()
    df = df.sort_values(["filed", "end"]).groupby("filed", as_index=False).last()
    return pd.DataFrame({"filed": pd.to_datetime(df["filed"]), "shares": df["val"].astype(float)})


def shares_outstanding(cik: int) -> pd.DataFrame:
    """Share counts with filing dates. First non-empty of: cover-page dei:EntityCommonStockSharesOutstanding
    (summed across classes), us-gaap:CommonStockSharesOutstanding, diluted weighted-average shares."""
    for taxonomy, tag in (("dei", "EntityCommonStockSharesOutstanding"),
                          ("us-gaap", "CommonStockSharesOutstanding"),
                          ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")):
        df = _share_frame(_concept(cik, taxonomy, tag))
        if len(df):
            return df
    return pd.DataFrame(columns=["filed", "shares"])
