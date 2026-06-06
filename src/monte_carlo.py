"""
Monte Carlo simulation engine with GARCH volatility
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.garch_model import GARCHVolatilityModel
import logging

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        """Initialize Monte Carlo simulator"""
        self.n_simulations = n_simulations
        np.random.seed(random_seed)
        
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
        
        # Initialize arrays
        paths = np.zeros((self.n_simulations, days_ahead))
        current_price = prices.iloc[-1]
        
        # Calculate drift (historical mean return)
        returns = np.log(prices / prices.shift(1)).dropna()
        mu = returns.mean() * 252  # Annualized
        
        # Extract GARCH parameters
        omega = garch_params['omega']
        alpha = garch_params['alpha']
        beta = garch_params['beta']
        
        # Get last conditional variance
        garch_model = GARCHVolatilityModel()
        garch_model.fit(prices)
        last_variance = garch_model.get_conditional_volatility().iloc[-1]**2
        
        for i in range(self.n_simulations):
            h_t = last_variance  # Initialize with last known (annualized) variance
            price_t = current_price

            for t in range(days_ahead):
                # Generate standard-normal shock
                z = np.random.normal(0, 1)

                # Residual in variance units, then correct GARCH(1,1) recursion:
                # h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
                eps = z * np.sqrt(h_t)
                h_t = omega + alpha * (eps ** 2) + beta * h_t

                # Daily return with drift and GARCH volatility
                daily_vol = np.sqrt(h_t / 252)  # annual variance -> daily vol
                daily_return = mu / 252 + daily_vol * z

                # Update price
                price_t = price_t * np.exp(daily_return)
                paths[i, t] = price_t
        
        return paths
    
    def simulate_around_event(self,
                            prices: pd.Series,
                            event_date: pd.Timestamp,
                            current_date: pd.Timestamp,
                            vol_multiplier: float = 1.5) -> Tuple[np.ndarray, int]:
        """
        Simulate prices around an event (like earnings) with volatility adjustment
        """
        days_to_event = (event_date - current_date).days
        days_after_event = 5  # Simulate 5 days post-event
        
        # Fit GARCH model
        garch = GARCHVolatilityModel()
        garch_params = garch.fit(prices)
        
        # Simulate to event
        pre_event_paths = self.simulate_gbm_garch(prices, days_to_event, garch_params)
        
        # Adjust volatility for event (earnings typically increase vol)
        event_params = garch_params.copy()
        event_params['omega'] *= vol_multiplier
        
        # Simulate post-event with higher volatility
        post_event_prices = pd.Series(pre_event_paths[:, -1])
        post_event_paths = self.simulate_gbm_garch(post_event_prices, 
                                                   days_after_event, 
                                                   event_params)
        
        # Combine paths
        full_paths = np.concatenate([pre_event_paths, post_event_paths], axis=1)
        
        return full_paths, days_to_event