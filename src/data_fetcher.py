"""
Module for fetching and managing financial data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    def __init__(self, db_path: str = "data/stock_data.db"):
        """Initialize data fetcher with database connection"""
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self):
        """Create a new database connection for thread safety"""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _create_tables(self):
        """Create necessary database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Table for historical prices
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        ''')

        # Table for earnings dates
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS earnings_calendar (
                ticker TEXT,
                earnings_date DATE,
                PRIMARY KEY (ticker, earnings_date)
            )
        ''')

        conn.commit()
        conn.close()

    def fetch_historical_data(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """Fetch historical stock data and APPEND it to the database.

        Uses an INSERT OR REPLACE keyed on (ticker, date) so repeated runs and
        multiple tickers accumulate in one table instead of overwriting it.
        """
        logger.info(f"Fetching {period} of data for {ticker}")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                return pd.DataFrame()

            # Shape the frame to match the table schema
            df = hist.reset_index()[
                ["Date", "Open", "High", "Low", "Close", "Volume"]
            ].copy()
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df.insert(0, "ticker", ticker)

            # Upsert row-by-row on the (ticker, date) primary key so we ACCUMULATE
            conn = self._get_connection()
            conn.executemany(
                """INSERT OR REPLACE INTO stock_prices
                   (ticker, date, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                df[["ticker", "date", "open", "high", "low", "close", "volume"]]
                .itertuples(index=False, name=None),
            )
            conn.commit()
            conn.close()

            return hist

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def get_prices_from_db(self, ticker: str) -> pd.DataFrame:
        """Read a ticker's stored prices back out of SQLite, oldest first."""
        conn = self._get_connection()
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume "
            "FROM stock_prices WHERE ticker = ? ORDER BY date ASC",
            conn,
            params=(ticker,),
        )
        conn.close()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def get_record_count(self) -> int:
        """Return total rows in stock_prices (for honest pipeline reporting)."""
        conn = self._get_connection()
        n = conn.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
        conn.close()
        return n

    def fetch_earnings_dates(self, ticker: str) -> List[datetime]:
        """Fetch earnings dates and persist them to earnings_calendar."""
        try:
            stock = yf.Ticker(ticker)
            dates: List[pd.Timestamp] = []

            # Historical + upcoming earnings dates (yfinance >= 0.2)
            try:
                ed = stock.get_earnings_dates(limit=24)
                if ed is not None and not ed.empty:
                    dates = [pd.to_datetime(d) for d in ed.index]
            except Exception:
                pass

            # Fallback: forward calendar
            if not dates:
                cal = stock.calendar
                if isinstance(cal, dict) and cal.get("Earnings Date"):
                    dates = [pd.to_datetime(cal["Earnings Date"][0])]
                elif hasattr(cal, "empty") and not cal.empty:
                    dates = [pd.to_datetime(cal.iloc[0, 0])]

            # Persist to earnings_calendar (now actually populated)
            if dates:
                conn = self._get_connection()
                conn.executemany(
                    "INSERT OR REPLACE INTO earnings_calendar (ticker, earnings_date) VALUES (?, ?)",
                    [(ticker, d.strftime("%Y-%m-%d")) for d in dates],
                )
                conn.commit()
                conn.close()

            return dates
        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker}: {e}")
            return []

    def get_sp500_tickers(self) -> List[str]:
        """Get the full list of S&P 500 tickers from Wikipedia.

        Falls back to a short liquid set only if the network fetch fails, so the
        dashboard still runs offline.
        """
        try:
            import requests, io
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            logger.info(f"Loaded {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers
        except Exception as e:
            logger.warning(f"Could not load full S&P 500 list ({e}); using fallback set")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
                    'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']