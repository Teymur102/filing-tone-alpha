"""SEC EDGAR download: 8-K Item 2.02 earnings releases (exhibit 99.x) for the ticker universe.

Everything fetched is cached under data/edgar, so re-runs make no network calls.
"""
from __future__ import annotations

import gzip
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from . import config as C

_session = requests.Session()
_session.headers.update({"User-Agent": C.USER_AGENT, "Accept-Encoding": "gzip, deflate"})
_lock = threading.Lock()
_last = [0.0]


def _throttle() -> None:
    """Global limiter shared by all threads: at most SEC_MAX_RPS requests per second."""
    with _lock:
        wait = _last[0] + 1.0 / C.SEC_MAX_RPS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()


def get(url: str, retries: int = 5) -> bytes:
    for i in range(retries):
        _throttle()
        try:
            r = _session.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** i)
            continue
        if r.status_code == 200:
            return r.content
        if r.status_code in (429, 503):
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed after {retries} tries: {url}")


def _cached(path, url: str) -> bytes:
    """Fetch url once; store gzipped at path."""
    if path.exists():
        return gzip.decompress(path.read_bytes())
    data = get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data))
    return data


def ticker_map() -> dict[str, int]:
    raw = _cached(C.RAW_EDGAR / "company_tickers.json.gz", "https://www.sec.gov/files/company_tickers.json")
    return {v["ticker"]: int(v["cik_str"]) for v in json.loads(raw).values()}


def _filings_frame(block: dict) -> pd.DataFrame:
    keys = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "primaryDocument"]
    return pd.DataFrame({k: block.get(k, []) for k in keys})


def earnings_8ks(ticker: str, cik: int) -> pd.DataFrame:
    """All original 8-K filings with Item 2.02 in the sample window for one company."""
    sub = json.loads(_cached(C.RAW_EDGAR / "submissions" / f"{cik}.json.gz",
                             f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    frames = [_filings_frame(sub["filings"]["recent"])]
    for f in sub["filings"].get("files", []):
        if f["filingTo"] >= C.SAMPLE_START:
            more = json.loads(_cached(C.RAW_EDGAR / "submissions" / f"{f['name']}.gz",
                                      f"https://data.sec.gov/submissions/{f['name']}"))
            frames.append(_filings_frame(more))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["form"] == "8-K") & df["items"].str.contains("2.02", regex=False)]
    df = df[(df["filingDate"] >= C.SAMPLE_START) & (df["filingDate"] <= C.SAMPLE_END)]
    df = df.drop_duplicates("accessionNumber").sort_values("acceptanceDateTime")
    # One release per firm-quarter: drop any 2.02 filing within 45 days of the previous kept one
    kept, last = [], None
    for _, row in df.iterrows():
        d = pd.Timestamp(row["filingDate"])
        if last is None or (d - last).days > 45:
            kept.append(row)
            last = d
    out = pd.DataFrame(kept)
    out["ticker"], out["cik"] = ticker, cik
    return out


_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_HREF = re.compile(r'href="([^"]+)"', re.I)


def exhibit_url(cik: int, accession: str) -> str | None:
    """Find the press-release exhibit (EX-99.1, else first EX-99*) in the filing index."""
    nodash = accession.replace("-", "")
    html = _cached(C.RAW_EDGAR / "index" / f"{accession}.htm.gz",
                   f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{accession}-index.htm").decode("latin-1")
    cands = []
    for row in _ROW.findall(html):
        cells = _CELL.findall(row)
        if len(cells) < 4:
            continue
        typ = re.sub(r"<[^>]+>", "", cells[3]).strip().upper()
        m = _HREF.search(cells[2])
        if m and typ.startswith("EX-99") and m.group(1).lower().endswith((".htm", ".html", ".txt")):
            href = m.group(1)
            if href.startswith("/ix?doc="):
                href = href[len("/ix?doc="):]
            cands.append((0 if typ in ("EX-99.1", "EX-99.01", "EX-99") else 1, "https://www.sec.gov" + href))
    return sorted(cands)[0][1] if cands else None


def fetch_exhibit(row) -> dict:
    url = exhibit_url(row.cik, row.accessionNumber)
    if url is None:
        return {"accessionNumber": row.accessionNumber, "exhibit_url": None}
    _cached(C.RAW_EDGAR / "exhibits" / f"{row.accessionNumber}.htm.gz", url)
    return {"accessionNumber": row.accessionNumber, "exhibit_url": url}


def build_filing_list() -> pd.DataFrame:
    """Return one row per earnings-release event, with the exhibit cached locally."""
    cmap = ticker_map()
    frames = []
    for t in C.TICKERS:
        cik = cmap.get(t)
        if cik is None:
            print(f"  [edgar] no CIK for {t}; skipped")
            continue
        frames.append(earnings_8ks(t, cik))
    events = pd.concat(frames, ignore_index=True)
    print(f"  [edgar] {len(events)} Item 2.02 8-Ks across {events.ticker.nunique()} firms; fetching exhibits")
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(fetch_exhibit, events.itertuples(index=False)))
    events = events.merge(pd.DataFrame(res), on="accessionNumber")
    missing = events.exhibit_url.isna().sum()
    print(f"  [edgar] exhibits found for {len(events) - missing}/{len(events)} filings")
    return events.dropna(subset=["exhibit_url"]).reset_index(drop=True)


def load_exhibit(accession: str) -> str:
    return gzip.decompress((C.RAW_EDGAR / "exhibits" / f"{accession}.htm.gz").read_bytes()).decode("utf-8", "ignore")
