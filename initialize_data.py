"""
Initialize database with S&P 500 historical data
"""
from src.data_fetcher import StockDataFetcher
import time


def initialize_sp500_data():
    """Fetch and store 5 years of S&P 500 data (accumulating across tickers)."""
    fetcher = StockDataFetcher()

    tickers = fetcher.get_sp500_tickers()
    print(f"Initializing data for {len(tickers)} stocks...")

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] Fetching {ticker}...", end="")
            data = fetcher.fetch_historical_data(ticker, period="5y")
            print(f" OK {len(data)} records")
            # Also persist earnings dates so the backtest has events to test
            fetcher.fetch_earnings_dates(ticker)
            time.sleep(0.5)  # Be nice to Yahoo Finance API
        except Exception as e:
            print(f" Error: {e}")
            continue

    total = fetcher.get_record_count()
    print("\nData initialization complete!")
    print(f"Database now holds {total:,} historical price records")


if __name__ == "__main__":
    initialize_sp500_data()