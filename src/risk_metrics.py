"""
Risk metrics calculation
"""
import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats

class RiskAnalyzer:
    def __init__(self, confidence_levels: list = [0.95, 0.99]):
        """Initialize risk analyzer"""
        self.confidence_levels = confidence_levels
        
    def calculate_metrics(self, 
                         simulated_paths: np.ndarray,
                         initial_price: float) -> Dict:
        """Calculate comprehensive risk metrics"""
        
        # Final prices and returns
        final_prices = simulated_paths[:, -1]
        returns = (final_prices - initial_price) / initial_price
        
        metrics = {
            # Basic statistics
            'expected_return': np.mean(returns) * 100,
            'volatility': np.std(returns) * 100,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            
            # Probability metrics
            'probability_profit': np.mean(returns > 0) * 100,
            'probability_loss_10pct': np.mean(returns < -0.10) * 100,
            
            # Risk metrics
            'max_drawdown': np.min(returns) * 100,
            'max_gain': np.max(returns) * 100,
            
            # Percentiles
            'median_return': np.median(returns) * 100,
            'percentile_25': np.percentile(returns, 25) * 100,
            'percentile_75': np.percentile(returns, 75) * 100,
        }
        
        # VaR and CVaR for different confidence levels
        for conf in self.confidence_levels:
            var_level = (1 - conf) * 100
            var = np.percentile(returns, var_level) * 100
            cvar = np.mean(returns[returns <= np.percentile(returns, var_level)]) * 100
            
            metrics[f'var_{int(conf*100)}'] = var
            metrics[f'cvar_{int(conf*100)}'] = cvar
        
        # Sharpe ratio (assuming risk-free rate of 5% annually)
        risk_free_daily = 0.05 / 252
        sharpe = (np.mean(returns) - risk_free_daily) / np.std(returns) if np.std(returns) > 0 else 0
        metrics['sharpe_ratio'] = sharpe * np.sqrt(252)  # Annualized
        
        return metrics
    
    def calculate_path_statistics(self, paths: np.ndarray) -> pd.DataFrame:
        """Calculate statistics for each time point in paths"""
        percentiles = [5, 25, 50, 75, 95]
        
        stats_df = pd.DataFrame({
            'mean': np.mean(paths, axis=0),
            'std': np.std(paths, axis=0),
            **{f'p{p}': np.percentile(paths, p, axis=0) for p in percentiles}
        })
        
        return stats_df