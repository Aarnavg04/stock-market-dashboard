"""
Main entry point for the Stock Market Analytics Dashboard
"""
import logging
from src.dashboard import run_dashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("="*50)
    print("Stock Market Analytics Dashboard")
    print("="*50)
    print("Starting server...")
    print("Navigate to http://localhost:8050 in your browser")
    print("Press Ctrl+C to stop the server")
    print("="*50)
    run_dashboard()