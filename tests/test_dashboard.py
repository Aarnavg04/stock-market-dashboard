"""
Integration tests: module imports, SQLite storage, and the earnings-volatility backtest.

Network-dependent checks (Yahoo Finance) skip instead of failing when offline.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.backtest import EarningsVolatilityBacktest
from src.data_fetcher import StockDataFetcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "stock_data.db")


def test_dashboard_app_builds():
    from src.dashboard import app
    assert app.layout is not None


def test_fetcher_creates_schema(tmp_path):
    db = tmp_path / "test.db"
    StockDataFetcher(db_path=str(db))
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"stock_prices", "earnings_calendar"} <= tables


def test_fetcher_db_round_trip(tmp_path):
    fetcher = StockDataFetcher(db_path=str(tmp_path / "test.db"))
    rows = [("TEST", "2024-01-03", 1, 2, 0.5, 1.5, 100), ("TEST", "2024-01-02", 1, 2, 0.5, 1.4, 90)]
    with sqlite3.connect(fetcher.db_path) as conn:
        conn.executemany("INSERT OR REPLACE INTO stock_prices VALUES (?,?,?,?,?,?,?)", rows)
        conn.executemany("INSERT OR REPLACE INTO stock_prices VALUES (?,?,?,?,?,?,?)", rows[:1])  # upsert
    df = fetcher.get_prices_from_db("TEST")
    assert fetcher.get_record_count() == 2
    assert list(df["close"]) == [1.4, 1.5]  # sorted oldest first
    assert fetcher.get_prices_from_db("MISSING").empty


@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="run initialize_data.py to build data/stock_data.db")
def test_local_database_has_prices():
    with sqlite3.connect(DB_PATH) as conn:
        assert conn.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0] > 0


def test_yahoo_fetch(tmp_path):
    fetcher = StockDataFetcher(db_path=str(tmp_path / "test.db"))
    data = fetcher.fetch_historical_data("AAPL", period="1mo")
    if data.empty:
        pytest.skip("Yahoo Finance unreachable")
    assert {"Open", "High", "Low", "Close", "Volume"} <= set(data.columns)
    assert fetcher.get_record_count() == len(data)


def test_backtest_on_synthetic_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=500, freq="B")
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))), index=dates)

    result = EarningsVolatilityBacktest(train_window=250, post_window=5, tol=1.5).run(
        prices, [dates[260], dates[380], dates[10], dates[498]]
    )
    # dates[10] lacks 250 training days and dates[498] lacks 5 post-event days -> skipped
    assert result["events_tested"] == 2
    assert 0.0 <= result["accuracy_pct"] <= 100.0
    detail = result["detail"]
    assert list(detail.columns) == ["earnings_date", "forecast_vol", "realized_vol", "hit"]
    band_ok = (detail["realized_vol"] >= detail["forecast_vol"] / 1.5 - 0.01) & (
        detail["realized_vol"] <= detail["forecast_vol"] * 1.5 + 0.01
    )
    assert (band_ok == detail["hit"]).all()


def test_backtest_accepts_tz_aware_dates():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=400, freq="B", tz="America/New_York")
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=dates)
    result = EarningsVolatilityBacktest().run(prices, [dates[300]])
    assert result["events_tested"] == 1


def test_next_event_date_picks_earliest_upcoming_and_handles_timezones():
    from src.dashboard import next_event_date
    tz = "America/New_York"
    dates = [pd.Timestamp("2026-07-30 16:30", tz=tz), pd.Timestamp("2026-04-30 16:30", tz=tz),
             pd.Timestamp("2026-10-29 16:30", tz=tz)]  # newest-first order is not assumed
    assert next_event_date(dates, pd.Timestamp("2026-06-05")) == pd.Timestamp("2026-07-30")
    assert next_event_date(dates, pd.Timestamp("2026-07-30")) == pd.Timestamp("2026-10-29")  # strictly after
    assert next_event_date(dates, pd.Timestamp("2026-11-07")) is None  # all in the past
    assert next_event_date([], pd.Timestamp("2026-01-01")) is None
