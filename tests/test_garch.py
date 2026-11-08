"""Tests for src/garch_model.py (GARCHVolatilityModel)."""
import numpy as np
import pytest

from src.garch_model import GARCHVolatilityModel


def test_fit_recovers_known_parameters(garch_prices):
    # Data generated with omega=0.05, alpha=0.10, beta=0.85 (percent-return units)
    params = GARCHVolatilityModel().fit(garch_prices)
    assert params["alpha"] == pytest.approx(0.10, abs=0.04)
    assert params["beta"] == pytest.approx(0.85, abs=0.06)
    assert 0 < params["alpha"] + params["beta"] < 1


def test_fit_returns_expected_keys_and_long_run_variance(garch_prices):
    params = GARCHVolatilityModel().fit(garch_prices)
    for key in ("omega", "alpha", "beta", "long_run_variance", "aic", "bic"):
        assert key in params
    expected = params["omega"] / (1 - params["alpha"] - params["beta"])
    assert params["long_run_variance"] == pytest.approx(expected)


def test_conditional_volatility_matches_return_count(garch_prices):
    model = GARCHVolatilityModel()
    model.fit(garch_prices)
    vol = model.get_conditional_volatility()
    assert len(vol) == len(garch_prices) - 1  # one return per pair of prices
    assert (vol > 0).all()


def test_forecast_volatility_shape_and_positivity(garch_prices):
    model = GARCHVolatilityModel()
    model.fit(garch_prices)
    fc = model.forecast_volatility(horizon=10)
    assert fc.shape == (10,)
    assert np.all(fc > 0)


def test_methods_require_fit_first():
    model = GARCHVolatilityModel()
    with pytest.raises(ValueError):
        model.forecast_volatility()
    with pytest.raises(ValueError):
        model.get_conditional_volatility()
