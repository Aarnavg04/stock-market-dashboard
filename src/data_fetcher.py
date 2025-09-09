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
        # Don't create a persistent connection - create it when needed
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
        """Fetch historical stock data"""
        logger.info(f"Fetching {period} of data for {ticker}")
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            # Store in database with new connection
            conn = self._get_connection()
            hist_reset = hist.reset_index()
            hist_reset['ticker'] = ticker
            hist_reset.to_sql('stock_prices', conn, 
                             if_exists='replace', index=False)
            conn.close()
            
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_earnings_dates(self, ticker: str) -> List[datetime]:
        """Fetch upcoming earnings dates"""
        try:
            stock = yf.Ticker(ticker)
            # Get earnings dates from calendar
            earnings = stock.calendar
            if earnings is not None and not earnings.empty:
                # Extract next earnings date
                next_earnings = earnings.iloc[0, 0]
                return [pd.to_datetime(next_earnings)]
            return []
        except:
            return []
    
    def get_sp500_tickers(self) -> List[str]:
        """Get list of S&P 500 tickers"""
        # For demo, using top 10 most liquid stocks
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
                'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']