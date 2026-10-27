import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tone import analysis as A  # noqa: E402
from tone import prices as P  # noqa: E402


def _toy():
    dates = pd.bdate_range("2022-01-03", periods=120)
    rets = pd.DataFrame({"AAA": 0.01, "BBB": -0.002, "SPY": 0.0}, index=dates)
    ev = pd.DataFrame({"ticker": ["AAA", "BBB"], "day0": [dates[10], dates[10]], "t0": [10, 10],
                       "terc": [3.0, 1.0]})
    return dates, rets, ev


def test_calendar_portfolio_holds_days_2_to_60_and_charges_turnover():
    dates, rets, ev = _toy()
    p = A.calendar_portfolio(ev, rets, "terc", "2022-01-01", "2022-12-31", cost_bps=10)
    held = p[(p.n_1 > 0) & (p.n_3 > 0)]
    assert held.index.min() == dates[12] and held.index.max() == dates[70]   # day +2 .. day +60
    assert held["gross"].to_numpy() == pytest.approx(0.012)                   # long +1%, short -0.2%
    # one entry and one exit per leg, each a full unit of one-way turnover at 10 bps
    assert p["cost"].sum() == pytest.approx(4 * 10 / 1e4)
    assert p["net"].sum() == pytest.approx(p["gross"].sum() - p["cost"].sum())


def test_tercile_table_high_minus_low():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"terc": np.tile([1.0, 2.0, 3.0], n // 3), "yq": np.repeat(["2020Q1", "2020Q2", "2020Q3"], n // 3)})
    for w in ("car_0_1", "car_2_20", "car_2_60"):
        df[w] = 0.01 * (df.terc - 2) + rng.normal(0, 1e-4, n)
    t = A.tercile_table(df, "terc").set_index("window")
    assert t.loc["car_0_1", "high_minus_low"] == pytest.approx(0.02, abs=1e-4)
    assert t.loc["car_0_1", "t_stat"] > 10


def test_unadjusted_close_undoes_split():
    px = pd.DataFrame({"date": pd.to_datetime(["2024-06-06", "2024-06-07", "2024-06-10"]), "ticker": "NVDA",
                       "close": [120.0, 121.0, 122.0], "adj_close": [120.0, 121.0, 122.0],
                       "split": [np.nan, np.nan, 10.0]})
    u = P.unadjusted_close(px)["NVDA"]
    assert u.tolist() == pytest.approx([1200.0, 1210.0, 122.0])
