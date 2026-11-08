# Technical Specification

This document describes the code as it currently behaves.
File and function names refer to `src/` unless stated otherwise.

## 1. Components

| Module | Class / function | Responsibility |
|---|---|---|
| `data_fetcher.py` | `StockDataFetcher` | Yahoo Finance download (`yfinance`), SQLite upserts, earnings dates, S&P 500 ticker list |
| `garch_model.py` | `GARCHVolatilityModel` | GARCH(1,1) fit via the `arch` library |
| `monte_carlo.py` | `MonteCarloSimulator` | GBM price paths driven by a GARCH(1,1) variance recursion |
| `risk_metrics.py` | `RiskAnalyzer` | Return statistics, VaR/CVaR, Sharpe, percentile bands |
| `backtest.py` | `EarningsVolatilityBacktest` | Post-earnings volatility forecast vs. realized |
| `dashboard.py` | Dash `app`, `run_dashboard()` | Single-page UI on `http://localhost:8050` |
| `initialize_data.py` (root) | `initialize_sp500_data()` | One-time database build |
| `main.py` (root) | — | Starts the Dash server |

## 2. Data layer

### 2.1 Storage

SQLite file `data/stock_data.db` (git-ignored, built locally). `StockDataFetcher` opens a new
connection per call.

```sql
stock_prices      (ticker TEXT, date DATE, open REAL, high REAL, low REAL,
                   close REAL, volume INTEGER, PRIMARY KEY (ticker, date))
earnings_calendar (ticker TEXT, earnings_date DATE, PRIMARY KEY (ticker, earnings_date))
```

Writes use `INSERT OR REPLACE`, so repeated fetches update rows in place instead of duplicating them.

### 2.2 Sources

- **Ticker list**: `get_sp500_tickers()` scrapes the constituents table from Wikipedia's
  "List of S&P 500 companies" page (`.` replaced with `-`, e.g. `BRK-B`). If the request fails it
  falls back to 10 hard-coded large caps.
- **Prices**: `fetch_historical_data(ticker, period)` calls `yf.Ticker(ticker).history(period=...)`,
  which returns split/dividend-adjusted daily OHLCV, and upserts it into `stock_prices`.
- **Earnings dates**: `fetch_earnings_dates(ticker)` calls `get_earnings_dates(limit=24)` (the 24
  most recent past and scheduled dates), falling back to `Ticker.calendar`, and upserts them into
  `earnings_calendar`.

### 2.3 Database build

`initialize_data.py` loops over the ticker list, fetching `period="10y"` of prices and the
earnings dates for each ticker, sleeping 0.5 s between tickers.

Snapshot of the database used for the README numbers (reproduce with
`python scripts/verify_readme_numbers.py`):

| Table | Contents |
|---|---|
| `stock_prices` | 1,229,526 rows; 503 tickers; 2016-06-06 to 2026-06-05 |
| `earnings_calendar` | 120 rows; 5 tickers (AAPL, AMZN, JPM, MSFT, NVDA), 24 dates each |

Earnings dates are stored for only 5 tickers in this snapshot. A fresh run of
`initialize_data.py` attempts earnings dates for every ticker, and because Yahoo returns a
rolling window, its row counts and date ranges will differ from the table above.

## 3. GARCH(1,1) model (`GARCHVolatilityModel`)

- Input: a price series. Returns are `r_t = 100 · ln(P_t / P_{t-1})` (daily, in percent).
- Model: `arch_model(r, vol='Garch', p=1, q=1)` with the library defaults (constant mean,
  normal innovations), fit by maximum likelihood:

  ```
  σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
  ```

  In `arch`, `p` is the ARCH (α) lag order and `q` the GARCH (β) lag order.
- `fit()` returns `{omega, alpha, beta, long_run_variance = ω / (1 − α − β), aic, bic}`. `ω` and
  the long-run variance are in **daily percent² units**.
- `forecast_volatility(horizon)` returns the forecast daily σ (percent) for each step.
  `get_conditional_volatility()` returns the in-sample σ_t series (percent).

Fitted parameters across all 503 tickers in the snapshot (full 10-year history per ticker):

| | mean | median |
|---|---|---|
| α | 0.106 | 0.093 |
| β | 0.812 | 0.850 |

## 4. Monte Carlo engine (`MonteCarloSimulator`)

`MonteCarloSimulator(n_simulations=10000, random_seed=42)` seeds NumPy's global RNG in its
constructor. All variances use the GARCH model's units: **daily variance of percent log
returns** (for example, `h = 2.25` means 1.5% daily volatility).

### 4.1 `simulate_gbm_garch(prices, days_ahead, garch_params)`

1. `μ` = mean daily log return × 252.
2. Fits a `GARCHVolatilityModel` on `prices` and starts every path from the **one-step-ahead
   forecast variance** `h_{T+1}`.
3. For each path and each step (pure-Python double loop):
   ```
   z   ~ N(0, 1)
   S  ← S · exp(μ/252 + (√h / 100) · z)     # today's return uses today's variance
   ε   = z · √h                              # shock in percent units
   h   = ω + α · ε² + β · h                  # variance for the next day
   ```
4. Returns an array of shape `(n_simulations, days_ahead)`.

On average, the variance at each simulated step matches `arch`'s analytic multi-step GARCH
forecast (checked in `tests/test_simulation.py`). On AAPL's last 5 years in the database,
simulated daily volatility is 1.56% against 1.72% historical. The gap is expected, because
GARCH starts from the current conditional variance rather than the long-run average.

### 4.2 `simulate_around_event(prices, event_date, current_date, vol_multiplier=1.5, days_after_event=5)`

1. `days_to_event` = weekdays from `current_date` up to `event_date`
   (`np.busday_count`, timezone-safe; exchange holidays are not excluded). Raises `ValueError`
   if the event is not after `current_date`.
2. Fits GARCH and simulates `days_to_event` steps, as in §4.1.
3. Multiplies `ω` by `vol_multiplier` and simulates `days_after_event` more steps. **Each path
   continues from its own pre-event price and variance.**
4. Returns the `(n_simulations, days_to_event + days_after_event)` array and `days_to_event`.

### 4.3 Measured performance

`python tests/benchmark.py` on an Apple M1 Pro, Python 3.12, AAPL (2,515 daily closes from the
database). Times cover `simulate_gbm_garch` only: no Yahoo download, no JSON serialization,
no rendering.

| Paths | 30-day horizon | 90-day horizon |
|---|---|---|
| 1,000 | 0.07 s | 0.22 s |
| 5,000 | 0.30 s | 0.90 s |
| 10,000 | 0.6 s | 1.9 s |

Other measurements from the same runs: `get_prices_from_db` (one ticker, 10 years) 3.2 ms;
GARCH fit 9.7 ms; peak process RSS 160 MB (30-day) / 172 MB (90-day).

## 5. Risk metrics (`RiskAnalyzer`)

`calculate_metrics(paths, initial_price)` uses only the **final** simulated price of each path:
`R = (S_T − S_0) / S_0`. All outputs except skewness, kurtosis and Sharpe are percentages.

| Key | Definition |
|---|---|
| `expected_return`, `volatility`, `median_return` | mean, std, median of `R` |
| `skewness`, `kurtosis` | `scipy.stats.skew`, `scipy.stats.kurtosis` (excess) of `R` |
| `probability_profit` | share of paths with `R > 0` |
| `probability_loss_10pct` | share of paths with `R < −10%` |
| `max_drawdown`, `max_gain` | min and max of `R` across paths (not a path-wise drawdown) |
| `percentile_25`, `percentile_75` | 25th and 75th percentiles of `R` |
| `var_95`, `var_99` | 5th and 1st percentiles of `R` |
| `cvar_95`, `cvar_99` | mean of `R` at or below the matching VaR |
| `sharpe_ratio` | `(mean(R) − 0.05·T/252) / std(R) × √(252/T)`, T = horizon in trading days |

`calculate_path_statistics(paths)` returns, per time step, the mean, std, and the 5th/25th/50th/75th/95th
percentiles across paths (used for the fan chart).

## 6. Earnings-volatility backtest (`EarningsVolatilityBacktest`)

Parameters: `train_window=250`, `post_window=5`, `tol=1.5`.

For each earnings date `E`:

1. Training set: the 250 daily percent log returns strictly before `E`. Evaluation set: the
   first 5 returns on or after `E`. Events lacking either are skipped.
2. Fit GARCH(1,1) on the training set. Forecast 5 steps ahead and convert the mean forecast
   variance to annualized volatility: `√(mean σ²) · √252`.
3. Realized volatility = std of the 5 post-event returns × √252.
4. **Hit** if `forecast / 1.5 ≤ realized ≤ forecast × 1.5`.

Output: `events_tested`, `accuracy_pct` (hit rate), and a per-event `detail` DataFrame.

Result on the snapshot (`scripts/verify_readme_numbers.py`, using the stored `earnings_calendar`):

| Ticker | Events | In-band |
|---|---|---|
| AAPL | 24 | 45.8% |
| AMZN | 24 | 45.8% |
| JPM | 21 | 47.6% |
| MSFT | 24 | 45.8% |
| NVDA | 24 | 54.2% |
| **Total** | **117** | **56 hits = 47.9%** |

Events range from 2019-10-23 to 2026-04-14. JPM has 3 fewer events because its stored dates
include scheduled future dates with no post-event data yet.

The dashboard's backtest panel runs the same class per ticker, but it fetches the earnings dates
live from Yahoo, so its numbers can drift from this table.

## 7. Dashboard (`dashboard.py`)

Dash + `dash-bootstrap-components` (CYBORG dark theme), served by `app.run(debug=True,
port=8050, host='0.0.0.0')`. The only HTTP routes are Dash's own: the page at `/` and Dash's
internal callback and asset endpoints. There is no custom REST API.

Callbacks:

1. **`run_simulation`** (Run Simulation button). Downloads 5 years of prices for the selected
   ticker from Yahoo (`fetch_historical_data(ticker, "5y")`, which also upserts them into SQLite)
   and fits GARCH. If "Analyze Earnings Event" is on, `next_event_date()` picks the earliest
   earnings date after the last close, and the callback runs `simulate_around_event` up to that
   date plus 5 post-event days. Otherwise, or when Yahoo lists no upcoming date, it runs
   `simulate_gbm_garch` over the slider horizon (5–90 days). Results (paths serialized as JSON
   lists, with the actual simulated horizon) go into a `dcc.Store`.
2. **`update_main_graphs`**: price-path fan chart (5/25/50/75/95 percentile bands plus 100 sample
   paths, with an earnings marker in event mode), final-return histogram, and a volatility term
   structure. The term structure compares simulated volatility with historical volatility and
   the GARCH long-run volatility `√(long_run_variance) · √252`.
3. **`update_metrics_panels`**: risk metric cards, probability panel, and quick stats.
4. **`update_bottom_panels`** (ticker dropdown). Backtest chart (prices from SQLite, earnings
   dates from Yahoo); next upcoming earnings date, or the latest listed date if none are upcoming;
   pipeline status showing the data sources and the live `stock_prices` row count.

Data updates only when the user clicks Run Simulation or changes the ticker. Nothing streams or polls.

## 8. Tests

`pytest tests/ -v` runs 29 tests:

- `test_simulation.py`: path shape and positivity; seed reproducibility; the variance recursion
  with α = β = 0 (volatility exactly `√ω / 100`, drift `μ/252`); simulated variance matching
  `arch`'s analytic forecast; start price; event simulation (trading-day count, tz-aware dates,
  rejecting past events, path continuity, higher post-event volatility).
- `test_garch.py`: parameter recovery on synthetic GARCH(1,1) data (ω = 0.05, α = 0.10, β = 0.85),
  output keys, long-run variance, forecast and conditional-volatility shapes, errors before fitting.
- `test_risk_metrics.py`: every metric checked against hand-computed values, including the
  horizon-scaled Sharpe ratio.
- `test_dashboard.py`: app construction; SQLite schema and upsert round-trip; earnings-date
  selection; a backtest on synthetic data (including skipped events and tz-aware dates); a
  local-DB check (skipped if there is no DB); a Yahoo fetch (skipped when offline).

`tests/benchmark.py` is a standalone script, not collected by pytest.

## 9. Limitations

- **Upcoming earnings dates depend on Yahoo.** If `get_earnings_dates` lists no future date for
  a ticker, event mode falls back to the standard simulation. With `yfinance` 0.2.65 in November
  2026 this happened for AAPL, whose latest returned date was 2025-05-01.
- **Event volatility is a heuristic.** Post-event volatility is raised by scaling ω (× 1.5 by
  default), not estimated from past earnings moves.
- **Trading-day counts** skip weekends but not exchange holidays.
- **Speed.** The simulation is a pure-Python loop; see §4.3 for timings.
