"""
Performance benchmark for the simulation pipeline.

Measures, on the machine it runs on:
  * SQLite read time for one ticker's full price history (get_prices_from_db)
  * GARCH(1,1) fit time
  * Monte Carlo simulation time (simulate_gbm_garch) for 1,000 / 5,000 / 10,000 paths
  * Peak traced Python memory during the 10,000-path simulation (tracemalloc)
  * Peak process RSS at the end of the run (resource.getrusage)

Usage:
    python tests/benchmark.py                # uses AAPL from data/stock_data.db
    python tests/benchmark.py --ticker MSFT --days 30

Requires data/stock_data.db (run `python initialize_data.py` first).
This file is a script, not a pytest test (pytest only collects test_*.py).
"""
import argparse
import logging
import os
import resource
import sys
import time
import tracemalloc
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import StockDataFetcher  # noqa: E402
from src.garch_model import GARCHVolatilityModel  # noqa: E402
from src.monte_carlo import MonteCarloSimulator  # noqa: E402
from src.risk_metrics import RiskAnalyzer  # noqa: E402


def _timed(fn, repeats=1):
    """Return (best wall-clock seconds over `repeats`, last result)."""
    best, result = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--days", type=int, default=30, help="forecast horizon in trading days")
    parser.add_argument("--db", default="data/stock_data.db")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    if not os.path.exists(args.db):
        sys.exit(f"{args.db} not found - run `python initialize_data.py` first.")

    fetcher = StockDataFetcher(db_path=args.db)

    t_db, prices_df = _timed(lambda: fetcher.get_prices_from_db(args.ticker), repeats=5)
    if prices_df.empty:
        sys.exit(f"No rows for {args.ticker} in {args.db}.")
    prices = prices_df["close"]
    print(f"Ticker: {args.ticker}  ({len(prices):,} daily closes, "
          f"{prices.index.min().date()} to {prices.index.max().date()})")
    print(f"SQLite read (get_prices_from_db, best of 5): {t_db * 1000:.1f} ms")

    t_fit, params = _timed(lambda: GARCHVolatilityModel().fit(prices), repeats=3)
    print(f"GARCH(1,1) fit (best of 3): {t_fit * 1000:.1f} ms  "
          f"(alpha={params['alpha']:.3f}, beta={params['beta']:.3f})")

    print(f"\nMonte Carlo simulate_gbm_garch, {args.days}-day horizon:")
    for n in (1000, 5000, 10000):
        sim = MonteCarloSimulator(n_simulations=n, random_seed=42)
        t_sim, paths = _timed(lambda: sim.simulate_gbm_garch(prices, args.days, params))
        print(f"  {n:>6,} paths: {t_sim:6.2f} s   output shape {paths.shape}")

    sim = MonteCarloSimulator(n_simulations=10000, random_seed=42)
    tracemalloc.start()
    paths = sim.simulate_gbm_garch(prices, args.days, params)
    RiskAnalyzer().calculate_metrics(paths, float(prices.iloc[-1]))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nPeak traced Python allocations, 10,000-path run + risk metrics: {peak / 1e6:.1f} MB")

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # bytes on macOS, KiB on Linux
    print(f"Peak process RSS (whole benchmark, incl. imports): {rss_mb:.0f} MB")


if __name__ == "__main__":
    main()
