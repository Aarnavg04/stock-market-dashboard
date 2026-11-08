"""Tests for src/monte_carlo.py (MonteCarloSimulator)."""
import numpy as np
import pandas as pd
import pytest

from src.garch_model import GARCHVolatilityModel
from src.monte_carlo import MonteCarloSimulator


def _log_returns(paths, start_price):
    full = np.column_stack([np.full(len(paths), start_price), paths])
    return np.diff(np.log(full), axis=1)


def test_output_shape_and_validity(garch_prices):
    params = GARCHVolatilityModel().fit(garch_prices)
    paths = MonteCarloSimulator(n_simulations=200).simulate_gbm_garch(garch_prices, 15, params)
    assert paths.shape == (200, 15)
    assert np.isfinite(paths).all()
    assert (paths > 0).all()  # GBM prices stay strictly positive


def test_same_seed_is_reproducible(garch_prices):
    params = GARCHVolatilityModel().fit(garch_prices)
    a = MonteCarloSimulator(n_simulations=100, random_seed=7).simulate_gbm_garch(garch_prices, 10, params)
    b = MonteCarloSimulator(n_simulations=100, random_seed=7).simulate_gbm_garch(garch_prices, 10, params)
    c = MonteCarloSimulator(n_simulations=100, random_seed=8).simulate_gbm_garch(garch_prices, 10, params)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_constant_variance_recursion_and_drift():
    # With alpha = beta = 0 the recursion gives h_t = omega from the second step on.
    # omega is a daily variance of percent returns, so the daily log-return std must be
    # sqrt(omega) / 100 and the mean must equal mu / 252.
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2018-01-01", periods=1000)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 1000))), index=dates)
    omega = 2.25  # 1.5% daily volatility
    params = {"omega": omega, "alpha": 0.0, "beta": 0.0}

    paths = MonteCarloSimulator(n_simulations=4000, random_seed=0).simulate_gbm_garch(prices, 20, params)
    lr = _log_returns(paths, prices.iloc[-1])[:, 1:]  # step 0 uses the fitted starting variance

    mu_daily = np.log(prices / prices.shift(1)).dropna().mean()
    assert lr.std() == pytest.approx(np.sqrt(omega) / 100, rel=0.02)
    assert lr.mean() == pytest.approx(mu_daily, abs=4 * (np.sqrt(omega) / 100) / np.sqrt(lr.size))


def test_paths_start_from_last_observed_price(garch_prices):
    params = GARCHVolatilityModel().fit(garch_prices)
    paths = MonteCarloSimulator(n_simulations=2000).simulate_gbm_garch(garch_prices, 1, params)
    # One step of zero-median noise around the last close: median ratio should be ~1.
    assert np.median(paths[:, 0] / garch_prices.iloc[-1]) == pytest.approx(1.0, abs=0.02)


def _event_prices():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=600)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 600))), index=dates)


def test_simulate_around_event_counts_trading_days():
    prices = _event_prices()
    now = prices.index[-1]
    event = now + pd.Timedelta(days=12)
    paths, days_to_event = MonteCarloSimulator(n_simulations=100).simulate_around_event(prices, event, now)
    assert days_to_event == np.busday_count(now.date(), event.date())  # weekdays, not calendar days
    assert paths.shape == (100, days_to_event + 5)  # pre-event days + 5 post-event days
    assert (paths > 0).all()


def test_simulate_around_event_accepts_tz_aware_event_date():
    prices = _event_prices()
    now = prices.index[-1]
    event = pd.Timestamp(now + pd.Timedelta(days=7)).tz_localize("America/New_York")
    _, days_to_event = MonteCarloSimulator(n_simulations=20).simulate_around_event(prices, event, now)
    assert days_to_event == np.busday_count(now.date(), event.date())


def test_simulate_around_event_rejects_past_event():
    prices = _event_prices()
    now = prices.index[-1]
    with pytest.raises(ValueError):
        MonteCarloSimulator(n_simulations=10).simulate_around_event(prices, now - pd.Timedelta(days=30), now)


def test_fitted_simulation_matches_garch_forecast(garch_prices):
    # The simulated variance at each step should match arch's own analytic GARCH forecast.
    model = GARCHVolatilityModel()
    params = model.fit(garch_prices)
    days = 20
    paths = MonteCarloSimulator(n_simulations=3000).simulate_gbm_garch(garch_prices, days, params)
    lr = _log_returns(paths, garch_prices.iloc[-1])
    forecast_vol = np.sqrt((model.forecast_volatility(horizon=days) ** 2).mean()) / 100
    assert lr.std() == pytest.approx(forecast_vol, rel=0.05)
    # First simulated day uses the one-step-ahead forecast variance
    assert lr[:, 0].std() == pytest.approx(model.forecast_volatility(horizon=1)[0] / 100, rel=0.05)


def test_event_paths_are_continuous():
    prices = _event_prices()
    now = prices.index[-1]
    paths, d = MonteCarloSimulator(n_simulations=300).simulate_around_event(
        prices, now + pd.Timedelta(days=14), now
    )
    # Each post-event path continues from its own pre-event price: one step later the
    # prices are strongly correlated and the first post-event move is small.
    assert np.corrcoef(paths[:, d - 1], paths[:, d])[0, 1] > 0.9
    assert np.median(np.abs(np.log(paths[:, d] / paths[:, d - 1]))) < 0.05


def test_event_window_has_higher_volatility():
    prices = _event_prices()
    now = prices.index[-1]
    params_free = MonteCarloSimulator(n_simulations=2000, random_seed=1)
    paths, d = params_free.simulate_around_event(prices, now + pd.Timedelta(days=14), now, vol_multiplier=5.0)
    lr = _log_returns(paths, prices.iloc[-1])
    assert lr[:, d:].std() > lr[:, :d].std()
