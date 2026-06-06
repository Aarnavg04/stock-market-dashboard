"""
Test script to verify all dashboard components
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def test_imports():
    """Test all required imports"""
    try:
        import dash
        import plotly
        import yfinance
        import arch
        from src.dashboard import app
        from src.data_fetcher import StockDataFetcher
        from src.monte_carlo import MonteCarloSimulator
        from src.risk_metrics import RiskAnalyzer
        from src.garch_model import GARCHVolatilityModel
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_fetch():
    """Test data fetching capability"""
    try:
        from src.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        data = fetcher.fetch_historical_data("AAPL", period="1mo")
        if not data.empty:
            print(f"✅ Data fetch successful: {len(data)} records")
            return True
        else:
            print("❌ Data fetch returned empty")
            return False
    except Exception as e:
        print(f"❌ Data fetch error: {e}")
        return False

def test_simulation():
    """Test Monte Carlo simulation"""
    try:
        from src.monte_carlo import MonteCarloSimulator
        from src.data_fetcher import StockDataFetcher
        
        fetcher = StockDataFetcher()
        simulator = MonteCarloSimulator(n_simulations=100)
        
        # Get sample data
        prices = fetcher.fetch_historical_data("AAPL", period="6mo")
        
        # Fit GARCH
        from src.garch_model import GARCHVolatilityModel
        garch = GARCHVolatilityModel()
        params = garch.fit(prices['Close'])
        
        # Run simulation
        paths = simulator.simulate_gbm_garch(prices['Close'], 10, params)
        
        if paths.shape == (100, 10):
            print("✅ Simulation successful")
            return True
        else:
            print(f"❌ Simulation shape incorrect: {paths.shape}")
            return False
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return False

def test_database():
    """Test database initialization"""
    try:
        import sqlite3
        import os
        
        # Check if database exists
        if os.path.exists("data/stock_data.db"):
            conn = sqlite3.connect("data/stock_data.db")
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if len(tables) >= 2:
                print(f"✅ Database initialized with {len(tables)} tables")
                
                # Check record count
                cursor.execute("SELECT COUNT(*) FROM stock_prices")
                count = cursor.fetchone()[0]
                print(f"   📊 Current records: {count}")
                
                conn.close()
                return True
            else:
                print("❌ Database tables not properly created")
                conn.close()
                return False
        else:
            print("❌ Database file not found")
            return False
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_backtest():
    """Test EarningsVolatilityBacktest produces valid results on synthetic data"""
    try:
        import numpy as np
        import pandas as pd
        from src.backtest import EarningsVolatilityBacktest

        rng = np.random.default_rng(42)
        dates = pd.date_range("2015-01-01", periods=500, freq="B")
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))), index=dates)

        earnings_dates = [dates[260], dates[380]]
        bt = EarningsVolatilityBacktest(train_window=250, post_window=5, tol=1.5)
        result = bt.run(prices, earnings_dates)

        assert "events_tested" in result and "accuracy_pct" in result, "Missing keys"
        assert result["events_tested"] > 0, "No events tested"
        assert 0.0 <= result["accuracy_pct"] <= 100.0, "Accuracy out of range"
        print(f"✅ Backtest test passed: {result['events_tested']} events, {result['accuracy_pct']}% accuracy")
        return True
    except Exception as e:
        print(f"❌ Backtest test error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*50)
    print("DASHBOARD COMPONENT TEST")
    print("="*50 + "\n")
    
    all_pass = True
    
    # Run tests
    all_pass &= test_imports()
    all_pass &= test_data_fetch()
    all_pass &= test_simulation()
    all_pass &= test_database()
    all_pass &= test_backtest()
    
    print("\n" + "="*50)
    if all_pass:
        print("✅ ALL TESTS PASSED - Dashboard ready to run!")
        print("\nRun your dashboard with: python main.py")
    else:
        print("❌ Some tests failed - please fix issues above")
    print("="*50 + "\n")