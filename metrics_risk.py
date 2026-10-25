import numpy as np
import pandas as pd

def calculate_annualized_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    ann_return = returns.mean() * 252
    ann_volatility = returns.std() * np.sqrt(252)
    
    if ann_volatility > 0:
        sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility
    else:
        sharpe_ratio = 0.0
        
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        "Annual Return": ann_return,
        "Annual Volatility": ann_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

def calculate_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    cov = asset_returns.cov(benchmark_returns)
    var = benchmark_returns.var()
    return cov / var if var > 0 else 1.0

def calculate_tail_risk(returns: pd.Series, confidence_level: float = 0.95) -> dict:
    var_95 = np.percentile(returns.dropna(), (1 - confidence_level) * 100)
    cvar_95 = returns[returns <= var_95].mean()
    return {
        f"Historical VaR ({confidence_level*100}%)": var_95,
        f"Historical CVaR ({confidence_level*100}%)": cvar_95
    }