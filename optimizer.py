import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Tuple, Dict

def get_portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculates the expected annualized return and volatility for a given set of weights.
    """
    returns = np.sum(mean_returns * weights) * 252
    variance = np.dot(weights.T, np.dot(cov_matrix, weights)) * 252
    volatility = np.sqrt(variance)
    return returns, volatility

def maximize_sharpe_ratio(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.0) -> np.ndarray:
    """
    Uses SciPy's SLSQP optimizer to find the exact asset weights 
    that give the highest possible Sharpe Ratio.
    """
    num_assets = len(mean_returns)
    
    def neg_sharpe(weights):
        p_ret, p_vol = get_portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - risk_free_rate) / p_vol if p_vol != 0 else 0

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    
    optimized = sco.minimize(
        neg_sharpe, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    return optimized.x

def generate_random_portfolios(returns_df: pd.DataFrame, num_portfolios: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation generating random weights to visualize the Efficient Frontier.
    """
    num_assets = len(returns_df.columns)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        port_ret, port_vol = get_portfolio_performance(weights, mean_returns, cov_matrix)
        
        results[0, i] = port_ret
        results[1, i] = port_vol
        results[2, i] = port_ret / port_vol if port_vol != 0 else 0
        
    return results, np.array(weights_record)

def get_fuzzy_weighted_portfolio(fuzzy_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Allocates capital strictly based on the calculated Fuzzy Engine scores.
    """
    total_score = sum(fuzzy_scores.values())
    
    if total_score == 0:
        num_assets = len(fuzzy_scores)
        return {ticker: 1.0 / num_assets for ticker in fuzzy_scores}
        
    fuzzy_weights = {ticker: score / total_score for ticker, score in fuzzy_scores.items()}
    return fuzzy_weights