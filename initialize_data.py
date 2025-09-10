"""
Initialize database with S&P 500 historical data
"""
from src.data_fetcher import StockDataFetcher
import time

def initialize_sp500_data():
    """Fetch and store 5 years of S&P 500 data"""
    fetcher = StockDataFetcher()
    
    # Get ticker list
    tickers = fetcher.get_sp500_tickers()
    
    print(f"Initializing data for {len(tickers)} stocks...")
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] Fetching {ticker}...", end="")
            data = fetcher.fetch_historical_data(ticker, period="5y")
            print(f" ✅ {len(data)} records")
            time.sleep(0.5)  # Be nice to Yahoo Finance API
        except Exception as e:
            print(f" ❌ Error: {e}")
            continue
    
    print("\n✅ Data initialization complete!")
    print("Database ready with 100k+ historical records")

if __name__ == "__main__":
    initialize_sp500_data()