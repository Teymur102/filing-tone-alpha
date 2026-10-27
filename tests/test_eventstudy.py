import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tone import analysis as A  # noqa: E402
from tone import eventstudy as E  # noqa: E402


def test_market_model_recovers_alpha_beta():
    rng = np.random.default_rng(0)
    rm = rng.normal(0, 0.01, 300)
    ri = 0.0005 + 1.3 * rm
    a, b, n = E.market_model(ri, rm)
    assert n == 300
    assert a == pytest.approx(0.0005, abs=1e-12)
    assert b == pytest.approx(1.3, abs=1e-10)


def test_market_model_skips_missing_days():
    rm = np.array([0.01, 0.02, np.nan, -0.01, 0.0, 0.03])
    ri = 2 * rm
    ri[4] = np.nan
    a, b, n = E.market_model(ri, rm)
    assert n == 4 and b == pytest.approx(2.0) and a == pytest.approx(0.0, abs=1e-12)


def test_abnormal_returns_isolate_event_shock_and_car_sums():
    rng = np.random.default_rng(1)
    T, t0 = 400, 320
    rm = rng.normal(0, 0.01, T)
    ri = 0.0002 + 0.8 * rm                  # exact market model -> AR = 0 except injected shocks
    ri[t0] += 0.05
    ri[t0 + 1] += 0.01
    ri[t0 + 10] -= 0.02
    a, b, ar = E.abnormal_returns(ri, rm, t0)
    assert b == pytest.approx(0.8) and a == pytest.approx(0.0002)
    assert ar[0] == pytest.approx(0.05) and ar[-3] == pytest.approx(0.0, abs=1e-12)
    assert E.car(ar, (0, 1)) == pytest.approx(0.06)
    assert E.car(ar, (2, 20)) == pytest.approx(-0.02)
    assert E.car(ar, (2, 60)) == pytest.approx(-0.02)


def test_estimation_window_excludes_event_period():
    # A huge shock inside [-29, +60] must not move the market-model estimates.
    rng = np.random.default_rng(2)
    rm = rng.normal(0, 0.01, 400)
    ri = 1.1 * rm
    ri[300 - 10] += 0.5
    a, b, _ = E.abnormal_returns(ri, rm, 300)
    assert b == pytest.approx(1.1) and a == pytest.approx(0.0, abs=1e-12)


def test_abnormal_returns_need_full_windows():
    rm = np.zeros(300)
    assert E.abnormal_returns(rm, rm, 200) is None   # estimation window starts before data
    assert E.abnormal_returns(rm, rm, 280) is None   # +60 days run past the end


def test_car_is_nan_if_a_day_is_missing():
    ar = {k: 0.01 for k in range(-5, 61)}
    ar[5] = np.nan
    assert np.isnan(E.car(ar, (2, 20)))
    assert E.car(ar, (0, 1)) == pytest.approx(0.02)


def test_event_day_respects_market_close():
    dates = pd.DatetimeIndex(["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
    assert E.event_day_index(dates, pd.Timestamp("2024-01-30 07:00")) == 1   # pre-market -> same day
    assert E.event_day_index(dates, pd.Timestamp("2024-01-30 16:05")) == 2   # after close -> next day
    assert E.event_day_index(dates, pd.Timestamp("2024-01-27 10:00")) == 0   # weekend -> Monday
    assert E.event_day_index(dates, pd.Timestamp("2024-02-02 17:00")) is None


def test_bhr():
    r = np.array([0.1, -0.1, 0.05])
    assert E.bhr(r, 0, 2) == pytest.approx(1.1 * 0.9 * 1.05 - 1)
    assert np.isnan(E.bhr(r, -1, 1))


def test_trailing_terciles_use_only_past_events():
    days = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({"day0": days, "x": np.arange(200, dtype=float)})
    t = A.trailing_terciles(df, "x", lookback_days=365, min_obs=60)
    assert t.iloc[:60].isna().all()                  # not enough history yet
    # an ever-increasing signal is always above all past breakpoints -> top tercile
    assert (t.iloc[60:] == 3).all()
    df.loc[150, "x"] = -1.0
    assert A.trailing_terciles(df, "x").iloc[150] == 1
