"""
GARCH modeling for volatility forecasting
"""
import numpy as np
import pandas as pd
from arch import arch_model
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class GARCHVolatilityModel:
    def __init__(self, p: int = 1, q: int = 1):
        """Initialize GARCH(p,q) model"""
        self.p = p
        self.q = q
        self.model = None
        self.model_fit = None
        
    def fit(self, prices: pd.Series) -> Dict:
        """Fit GARCH model to price series"""
        # Calculate returns
        returns = np.log(prices / prices.shift(1)).dropna() * 100
        
        # Fit GARCH model
        self.model = arch_model(returns, vol='Garch', p=self.p, q=self.q)
        self.model_fit = self.model.fit(disp='off', show_warning=False)
        
        # Extract parameters
        params = {
            'omega': self.model_fit.params['omega'],
            'alpha': self.model_fit.params[f'alpha[1]'],
            'beta': self.model_fit.params[f'beta[1]'],
            'long_run_variance': self.model_fit.params['omega'] / 
                                (1 - self.model_fit.params[f'alpha[1]'] - 
                                 self.model_fit.params[f'beta[1]'])
        }
        
        # Add diagnostics
        params['aic'] = self.model_fit.aic
        params['bic'] = self.model_fit.bic
        
        logger.info(f"GARCH model fitted - AIC: {params['aic']:.2f}")
        
        return params
    
    def forecast_volatility(self, horizon: int = 30) -> np.ndarray:
        """Forecast volatility for given horizon"""
        if self.model_fit is None:
            raise ValueError("Model must be fitted first")
            
        forecast = self.model_fit.forecast(horizon=horizon)
        return np.sqrt(forecast.variance.values[-1, :])
    
    def get_conditional_volatility(self) -> pd.Series:
        """Get historical conditional volatility"""
        if self.model_fit is None:
            raise ValueError("Model must be fitted first")
        return self.model_fit.conditional_volatility