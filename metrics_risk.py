import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_annualized_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Calculates key annualized performance and risk metrics for a single asset or portfolio.
    Assumes 252 trading days in a year.
    """
    TRADING_DAYS = 252
    
    # Annualized Return & Volatility
    annual_return = returns.mean() * TRADING_DAYS
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    
    # Sharpe Ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
    
    # Sortino Ratio (Downside deviation only)
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(TRADING_DAYS)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if (downside_std != 0 and not pd.isna(downside_std)) else 0
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    max_drawdown = drawdown.min()
    
    return {
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown
    }

def calculate_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculates the Beta of an asset relative to a benchmark.
    Beta = Covariance(Asset, Benchmark) / Variance(Benchmark)
    """
    covariance_matrix = np.cov(asset_returns, benchmark_returns)
    covariance = covariance_matrix[0, 1]
    benchmark_variance = covariance_matrix[1, 1]
    
    return covariance / benchmark_variance if benchmark_variance != 0 else 1.0

def calculate_tail_risk(returns: pd.Series, confidence_level: float = 0.95) -> dict:
    """
    Calculates Value at Risk (VaR) and Conditional VaR (Expected Shortfall) 
    using both Historical and Parametric (Normal Distribution) methods.
    """
    alpha = 1 - confidence_level
    
    # Historical VaR & CVaR
    hist_var = returns.quantile(alpha)
    hist_cvar = returns[returns <= hist_var].mean()
    
    # Parametric (Normal) VaR & CVaR
    mu = returns.mean()
    sigma = returns.std()
    z_score = norm.ppf(alpha)
    
    param_var = mu + z_score * sigma
    
    # CVaR formulation for Normal Distribution
    param_cvar = mu - sigma * (norm.pdf(z_score) / alpha)
    
    return {
        f"Historical VaR ({confidence_level*100}%)": hist_var,
        f"Historical CVaR ({confidence_level*100}%)": hist_cvar,
        f"Parametric VaR ({confidence_level*100}%)": param_var,
        f"Parametric CVaR ({confidence_level*100}%)": param_cvar
    }


# --- TEST BLOCK ---
if __name__ == "__main__":
    from data_loader import fetch_market_data
    
    print("Fetching data for Risk Engine Test...")
    test_tickers = ["THYAO.IS"]
    prices, bench_p, returns_df, bench_r = fetch_market_data(test_tickers, market="BIST-30", period="3y")
    
    asset_ret = returns_df["THYAO.IS"]
    
    print("\n--- PERFORMANCE METRICS ---")
    metrics = calculate_annualized_metrics(asset_ret)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\n--- MARKET RISK ---")
    beta = calculate_beta(asset_ret, bench_r)
    print(f"Beta vs BIST-100: {beta:.4f}")
    
    print("\n--- TAIL RISK (95% Confidence) ---")
    tail_risk = calculate_tail_risk(asset_ret, confidence_level=0.95)
    for k, v in tail_risk.items():
        print(f"{k}: {v:.4f}")