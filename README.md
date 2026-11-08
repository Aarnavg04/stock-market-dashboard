# Stock Market Analytics Dashboard

## Overview

A Plotly Dash dashboard that runs **Monte Carlo price simulations** driven by a **GARCH(1,1)
volatility model**, computes risk metrics on the simulated distribution, and backtests how well
GARCH(1,1) forecasts post-earnings volatility. Price history for every S&P 500 constituent is
stored locally in SQLite (10 years of daily data).

### Key Features

- **🎲 Monte Carlo simulation**: 1,000 / 5,000 / 10,000 Geometric Brownian Motion paths over a
  5–90 day horizon, with a GARCH(1,1) variance recursion driving each path's volatility
- **📊 GARCH(1,1) volatility modeling**: fit by maximum likelihood with the `arch` library
- **📅 Earnings-event simulation**: simulates up to the next earnings date, then 5 trading days
  with raised baseline variance (ω × 1.5), each path continuing from its own pre-event state
- **🧪 Post-earnings volatility backtest**: 117 earnings events across 5 tickers, with 47.9% of
  realized 5-day volatilities falling within 1.5× of the GARCH forecast
- **💾 1.2M+ rows of price history**: 1,229,526 daily OHLCV rows for 503 S&P 500 tickers
  (2016-06-06 to 2026-06-05) in SQLite
- **📈 Risk metrics**: VaR and CVaR (95% / 99%), Sharpe ratio, probability of profit, skewness,
  kurtosis, percentile bands
- **🎨 Interactive visualizations**: Plotly Dash interface with a dark theme (price-path fan chart,
  return histogram, volatility term structure, backtest chart)

## Quick Start

### Prerequisites

- Python 3.12 (tested). The dependency minimums in `requirements.txt` need Python 3.9 or newer.
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aarnavg04/stock-market-dashboard.git
cd stock-market-dashboard
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Build the database**
```bash
python initialize_data.py
```
This downloads 10 years of daily prices and recent earnings dates from Yahoo Finance for every
S&P 500 ticker listed on Wikipedia. It pauses 0.5 s between tickers, so expect it to take several
minutes. The database (`data/stock_data.db`) is git-ignored and built locally.

5. **Run the dashboard**
```bash
python main.py
```

6. **Open your browser** at `http://localhost:8050`

## Technical Architecture

### Data flow

1. `initialize_data.py` → `StockDataFetcher` downloads prices and earnings dates from Yahoo
   Finance (`yfinance`) and upserts them into SQLite.
2. Clicking **Run Simulation** downloads the selected ticker's last 5 years of prices, fits
   GARCH(1,1), simulates paths, and computes risk metrics. With "Analyze Earnings Event" on, it
   simulates through the next earnings date Yahoo lists. If none is listed, it falls back to the
   standard horizon.
3. The backtest panel reads the 10-year price history from SQLite and runs the earnings backtest
   for the selected ticker.

Data refreshes only when you click Run Simulation or change the ticker. Nothing streams or polls.

### Monte Carlo simulation engine
```
S(t+1) = S(t) × exp(μ·dt + σ(t)·√dt·Z)

Where:
- μ    = drift (historical mean log return, annualized)
- σ(t) = time-varying volatility from the GARCH(1,1) recursion, starting from the
         one-step-ahead GARCH forecast
- Z ~ N(0,1) = random shock
- dt   = 1/252
```

### GARCH(1,1) model
```
σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

- α = reaction to recent shocks
- β = volatility persistence
- ω = baseline variance
```
Fitted on the full 10-year history of all 503 tickers in the database, the median α is 0.093 and
the median β is 0.850 (means: 0.106 and 0.812).

### Risk metrics

All metrics are computed on each path's return over the full horizon, `(S_T − S_0) / S_0`.

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **VaR (95% / 99%)** | Loss threshold exceeded in 5% / 1% of scenarios | 5th / 1st percentile of returns |
| **CVaR** | Expected loss beyond VaR | Mean of returns at or below VaR |
| **Sharpe Ratio** | Annualized risk-adjusted return | (mean return − 5%·T/252) / std × √(252/T), T = horizon in trading days |
| **Probability of Profit** | Chance of a positive return | Share of paths with return > 0 |

### Earnings-volatility backtest

For each earnings date, the backtest fits GARCH(1,1) on the previous 250 trading days of returns
and forecasts volatility for the next 5 trading days. It then compares that forecast with the
volatility actually realized. An event counts as **in band** if the realized volatility is
between forecast ÷ 1.5 and forecast × 1.5.

| | |
|---|---|
| Events tested | 117 (AAPL 24, AMZN 24, JPM 21, MSFT 24, NVDA 24) |
| In band | 56 / 117 = **47.9%** |
| Event dates | 2019-10-23 to 2026-04-14 |
| Earnings dates | 24 most recent per ticker returned by Yahoo Finance when the database was built (June 2026) |

See [docs/technical_spec.md](docs/technical_spec.md) for full definitions.

## Project Structure

```
stock-market-dashboard/
│
├── src/
│   ├── dashboard.py          # Dash app: layout and callbacks
│   ├── monte_carlo.py        # Monte Carlo simulation engine
│   ├── garch_model.py        # GARCH(1,1) volatility model
│   ├── risk_metrics.py       # VaR, CVaR, Sharpe and other metrics
│   ├── data_fetcher.py       # Yahoo Finance → SQLite pipeline
│   └── backtest.py           # Post-earnings volatility backtest
│
├── tests/
│   ├── test_simulation.py    # Monte Carlo engine
│   ├── test_garch.py         # GARCH model
│   ├── test_risk_metrics.py  # Risk metrics
│   ├── test_dashboard.py     # App build, SQLite storage, backtest
│   ├── conftest.py           # Shared fixtures
│   └── benchmark.py          # Performance benchmark (script)
│
├── scripts/
│   └── verify_readme_numbers.py  # Recomputes the data/backtest/GARCH numbers in this README
│
├── docs/
│   └── technical_spec.md     # How the code works, measured results, limitations
│
├── data/                     # stock_data.db is created here (git-ignored)
├── initialize_data.py        # Builds the database
├── main.py                   # Entry point
├── requirements.txt
└── LICENSE
```

## Performance

Measured with `python tests/benchmark.py` on an Apple M1 Pro (Python 3.12, AAPL, 2,515 daily
closes). Timings cover the simulation function only, not the Yahoo download or chart rendering.

| Paths | 30-day horizon | 90-day horizon |
|---|---|---|
| 1,000 | 0.07 s | 0.22 s |
| 5,000 | 0.30 s | 0.90 s |
| 10,000 | 0.6 s | 1.9 s |

- **SQLite read** of one ticker's 10-year history: ~3 ms
- **GARCH(1,1) fit** on 10 years of daily data: ~10 ms
- **Peak memory** of the benchmark process (10,000 paths): 160 MB (30-day) / 172 MB (90-day)

## Configuration

```python
from src.monte_carlo import MonteCarloSimulator
from src.garch_model import GARCHVolatilityModel

simulator = MonteCarloSimulator(
    n_simulations=10000,  # number of paths
    random_seed=42        # for reproducibility
)

model = GARCHVolatilityModel(
    p=1,  # ARCH (α) lag order
    q=1   # GARCH (β) lag order
)
```

## Usage Example

```python
from src.data_fetcher import StockDataFetcher
from src.garch_model import GARCHVolatilityModel
from src.monte_carlo import MonteCarloSimulator
from src.risk_metrics import RiskAnalyzer

# Load prices from the local database (run initialize_data.py first)
prices = StockDataFetcher().get_prices_from_db("AAPL")["close"]

# Fit GARCH(1,1)
params = GARCHVolatilityModel().fit(prices)

# Simulate 10,000 paths 30 trading days ahead
simulator = MonteCarloSimulator(n_simulations=10000, random_seed=42)
paths = simulator.simulate_gbm_garch(prices, days_ahead=30, garch_params=params)

# Risk metrics
metrics = RiskAnalyzer().calculate_metrics(paths, initial_price=prices.iloc[-1])
print(f"VaR (95%): {metrics['var_95']:.2f}%")
print(f"Probability of profit: {metrics['probability_profit']:.1f}%")
```

## Testing

```bash
pytest tests/ -v                       # unit and integration tests
python tests/benchmark.py              # performance benchmark
python scripts/verify_readme_numbers.py  # recompute the README's data and backtest numbers
```

Current result: 29 passed. The tests cover the simulation engine (including agreement with the
`arch` library's analytic GARCH forecast), the GARCH model, every risk metric, SQLite storage,
the backtest, and earnings-date handling.

## Future Enhancements

- [ ] Options pricing with Black-Scholes
- [ ] Portfolio optimization
- [ ] Real-time data streaming
- [ ] Machine learning predictions
- [ ] Multi-asset correlation analysis
- [ ] Backtesting framework expansion

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Yahoo Finance** (via `yfinance`) for historical prices and earnings dates
- **Wikipedia** for the S&P 500 constituent list
- **Plotly Dash** for the visualization framework
- **ARCH** library for GARCH modeling
- **NumPy/Pandas/SciPy** for numerical computing

## Contact

Aarnav Gandhi - aarnavgandhi04@gmail.com
