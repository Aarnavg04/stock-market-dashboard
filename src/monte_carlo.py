"""
Monte Carlo simulation engine with GARCH volatility
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.garch_model import GARCHVolatilityModel
import logging

logger = logging.getLogger(__name__)


def _to_naive_day(ts) -> pd.Timestamp:
    """Timestamp -> tz-naive midnight of the same local calendar day."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


class MonteCarloSimulator:
    """GBM price paths whose daily variance follows a GARCH(1,1) recursion.

    Units follow GARCHVolatilityModel: omega and the variance h are daily variances of
    percent log returns (e.g. h = 2.25 means a 1.5% daily volatility).
    """

    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        """Initialize Monte Carlo simulator"""
        self.n_simulations = n_simulations
        np.random.seed(random_seed)

    @staticmethod
    def _annual_drift(prices: pd.Series) -> float:
        """Historical mean daily log return, annualized (decimal)."""
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns.mean() * 252

    @staticmethod
    def _next_variance(model: GARCHVolatilityModel) -> float:
        """One-step-ahead GARCH variance for the first simulated day (daily, percent^2)."""
        return float(model.forecast_volatility(horizon=1)[0] ** 2)

    def _run_paths(self, start_prices: np.ndarray, start_variances: np.ndarray,
                   mu: float, garch_params: Dict, days: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate `days` steps from per-path starting prices and variances.

        Returns (paths of shape (n_simulations, days), variance after the last step per path).
        """
        omega = garch_params['omega']
        alpha = garch_params['alpha']
        beta = garch_params['beta']

        paths = np.zeros((self.n_simulations, days))
        final_variances = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            h_t = start_variances[i]
            price_t = start_prices[i]

            for t in range(days):
                z = np.random.normal(0, 1)

                # Today's return uses today's conditional variance (percent^2 -> decimal vol)
                daily_vol = np.sqrt(h_t) / 100.0
                price_t = price_t * np.exp(mu / 252 + daily_vol * z)
                paths[i, t] = price_t

                # GARCH(1,1) update for tomorrow: h = omega + alpha * eps^2 + beta * h
                eps = z * np.sqrt(h_t)  # shock in percent-return units
                h_t = omega + alpha * eps ** 2 + beta * h_t

            final_variances[i] = h_t

        return paths, final_variances

    def simulate_gbm_garch(self,
                           prices: pd.Series,
                           days_ahead: int,
                           garch_params: Dict) -> np.ndarray:
        """
        Simulate price paths using Geometric Brownian Motion with GARCH volatility

        Returns:
            Array of shape (n_simulations, days_ahead) with simulated prices
        """
        logger.info(f"Running {self.n_simulations} simulations for {days_ahead} days")

        n = self.n_simulations
        model = GARCHVolatilityModel()
        model.fit(prices)
        paths, _ = self._run_paths(
            start_prices=np.full(n, float(prices.iloc[-1])),
            start_variances=np.full(n, self._next_variance(model)),
            mu=self._annual_drift(prices),
            garch_params=garch_params,
            days=days_ahead,
        )
        return paths

    def simulate_around_event(self,
                              prices: pd.Series,
                              event_date: pd.Timestamp,
                              current_date: pd.Timestamp,
                              vol_multiplier: float = 1.5,
                              days_after_event: int = 5) -> Tuple[np.ndarray, int]:
        """
        Simulate prices up to an event (like earnings), then `days_after_event` more
        trading days with omega scaled by `vol_multiplier`. Each post-event path continues
        from its own pre-event price and variance.

        Returns (paths of shape (n_simulations, days_to_event + days_after_event),
        days_to_event), where days_to_event counts weekdays from current_date up to the
        event date (exchange holidays are not excluded).
        """
        days_to_event = int(np.busday_count(_to_naive_day(current_date).date(),
                                            _to_naive_day(event_date).date()))
        if days_to_event < 1:
            raise ValueError(f"Event date {event_date} is not after {current_date}")

        garch = GARCHVolatilityModel()
        garch_params = garch.fit(prices)
        next_variance = self._next_variance(garch)
        mu = self._annual_drift(prices)
        n = self.n_simulations

        pre_event_paths, event_variances = self._run_paths(
            np.full(n, float(prices.iloc[-1])), np.full(n, next_variance),
            mu, garch_params, days_to_event,
        )

        # Earnings typically raise volatility: scale omega for the post-event window
        event_params = dict(garch_params, omega=garch_params['omega'] * vol_multiplier)
        post_event_paths, _ = self._run_paths(
            pre_event_paths[:, -1], event_variances, mu, event_params, days_after_event,
        )

        full_paths = np.concatenate([pre_event_paths, post_event_paths], axis=1)
        return full_paths, days_to_event
