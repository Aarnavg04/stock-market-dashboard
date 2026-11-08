"""
Recompute every data/model number quoted in README.md from data/stock_data.db.

    python scripts/verify_readme_numbers.py

Prints:
  * stock_prices row count, ticker count and date range
  * the earnings-volatility backtest (events, hits, in-band rate, date range) for every
    ticker that has earnings dates stored in earnings_calendar
  * GARCH(1,1) alpha/beta mean and median across every ticker in the database

Results depend on the database contents, which depend on when initialize_data.py ran
(Yahoo Finance returns a rolling window). Timing numbers come from tests/benchmark.py.
"""
import logging
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import EarningsVolatilityBacktest  # noqa: E402
from src.garch_model import GARCHVolatilityModel  # noqa: E402

DB = "data/stock_data.db"


def main():
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    if not os.path.exists(DB):
        sys.exit(f"{DB} not found - run `python initialize_data.py` first.")
    conn = sqlite3.connect(DB)

    rows, tickers, first, last = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(date), MAX(date) FROM stock_prices"
    ).fetchone()
    print(f"stock_prices: {rows:,} rows, {tickers} tickers, {first} to {last}")

    prices = pd.read_sql_query(
        "SELECT ticker, date, close FROM stock_prices ORDER BY ticker, date", conn, parse_dates=["date"]
    )
    closes = {t: g.set_index("date")["close"].dropna() for t, g in prices.groupby("ticker")}

    # Earnings-volatility backtest (same settings the dashboard uses)
    earnings = pd.read_sql_query("SELECT ticker, earnings_date FROM earnings_calendar", conn)
    events = []
    for t, g in earnings.groupby("ticker"):
        if t not in closes:
            continue
        res = EarningsVolatilityBacktest(train_window=250, post_window=5, tol=1.5).run(
            closes[t], list(pd.to_datetime(g["earnings_date"]))
        )
        if len(res["detail"]):
            events.append(res["detail"].assign(ticker=t))
    if events:
        bt = pd.concat(events)
        print(
            f"Backtest: {len(bt)} events across {bt['ticker'].nunique()} tickers "
            f"({', '.join(sorted(bt['ticker'].unique()))}); {int(bt['hit'].sum())} in band "
            f"= {100 * bt['hit'].mean():.1f}%; events {bt['earnings_date'].min()} to {bt['earnings_date'].max()}"
        )
    else:
        print("Backtest: no testable earnings events in earnings_calendar")

    # GARCH(1,1) parameters across all tickers, full stored history
    alphas, betas = [], []
    for t, s in closes.items():
        try:
            p = GARCHVolatilityModel().fit(s)
            alphas.append(p["alpha"])
            betas.append(p["beta"])
        except Exception:
            pass
    a, b = np.array(alphas), np.array(betas)
    print(
        f"GARCH(1,1) over {len(a)} tickers: alpha mean {a.mean():.3f} / median {np.median(a):.3f}; "
        f"beta mean {b.mean():.3f} / median {np.median(b):.3f}"
    )


if __name__ == "__main__":
    main()
