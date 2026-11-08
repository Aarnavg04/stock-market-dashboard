"""Tests for src/risk_metrics.py (RiskAnalyzer)."""
import numpy as np
import pytest

from src.risk_metrics import RiskAnalyzer


@pytest.fixture
def known_paths():
    # 100 two-step paths; final prices 51..150 against an initial price of 100,
    # so final returns are exactly -0.49, -0.48, ..., +0.50.
    finals = np.arange(51, 151, dtype=float)
    return np.column_stack([np.full(100, 100.0), finals])


def test_basic_statistics(known_paths):
    m = RiskAnalyzer().calculate_metrics(known_paths, initial_price=100.0)
    returns = (known_paths[:, -1] - 100.0) / 100.0
    assert m["expected_return"] == pytest.approx(returns.mean() * 100)
    assert m["volatility"] == pytest.approx(returns.std() * 100)
    assert m["median_return"] == pytest.approx(np.median(returns) * 100)
    assert m["max_drawdown"] == pytest.approx(-49.0)
    assert m["max_gain"] == pytest.approx(50.0)
    assert m["skewness"] == pytest.approx(0.0, abs=1e-9)  # symmetric distribution


def test_probabilities(known_paths):
    m = RiskAnalyzer().calculate_metrics(known_paths, initial_price=100.0)
    assert m["probability_profit"] == pytest.approx(50.0)       # returns +0.01..+0.50
    assert m["probability_loss_10pct"] == pytest.approx(39.0)   # returns -0.49..-0.11


def test_var_and_cvar(known_paths):
    m = RiskAnalyzer().calculate_metrics(known_paths, initial_price=100.0)
    returns = (known_paths[:, -1] - 100.0) / 100.0
    for conf in (95, 99):
        var = np.percentile(returns, 100 - conf)
        assert m[f"var_{conf}"] == pytest.approx(var * 100)
        assert m[f"cvar_{conf}"] == pytest.approx(returns[returns <= var].mean() * 100)
        assert m[f"cvar_{conf}"] <= m[f"var_{conf}"]  # tail mean is at least as bad as VaR
    assert m["var_99"] <= m["var_95"]


def test_sharpe_zero_when_no_dispersion():
    paths = np.full((50, 5), 100.0)
    m = RiskAnalyzer().calculate_metrics(paths, initial_price=100.0)
    assert m["sharpe_ratio"] == 0


def test_path_statistics_columns_and_ordering():
    rng = np.random.default_rng(1)
    paths = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (1000, 20)), axis=1))
    stats = RiskAnalyzer().calculate_path_statistics(paths)
    assert list(stats.columns) == ["mean", "std", "p5", "p25", "p50", "p75", "p95"]
    assert len(stats) == 20
    assert (stats["p5"] <= stats["p50"]).all() and (stats["p50"] <= stats["p95"]).all()


def test_sharpe_scales_with_horizon():
    # Final returns with mean 2% and std 10% over a 21-trading-day horizon
    rng = np.random.default_rng(0)
    r = rng.normal(0.02, 0.10, 5000)
    days = 21
    paths = np.column_stack([np.full((5000, days - 1), 100.0), 100.0 * (1 + r)])
    m = RiskAnalyzer().calculate_metrics(paths, initial_price=100.0)
    expected = (r.mean() - 0.05 * days / 252) / r.std() * np.sqrt(252 / days)
    assert m["sharpe_ratio"] == pytest.approx(expected)
