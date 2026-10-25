import numpy as np
import pandas as pd
from typing import Dict, Tuple
from optimizer import maximize_sharpe_ratio, get_fuzzy_weighted_portfolio

def _compute_window_rsi(series: pd.Series, period: int = 14) -> float:
    """Calculates the latest RSI value within a historical lookback window."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    val = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0

def _compute_rolling_fuzzy_scores(window_returns: pd.DataFrame) -> Dict[str, float]:
    """
    Computes dynamic Fuzzy membership scores for each asset strictly using 
    data available inside the current lookback window (preventing look-ahead bias).
    """
    scores = {}
    cum_prices = (1.0 + window_returns).cumprod()
    
    for col in window_returns.columns:
        ret_s = window_returns[col]
        ann_ret = ret_s.mean() * 252
        ann_vol = ret_s.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        rsi = _compute_window_rsi(cum_prices[col])
        
        # Continuous fuzzy membership functions (Sharpe, Volatility, RSI)
        mu_sharpe = np.clip((sharpe + 0.5) / 2.5, 0.05, 1.0)
        mu_low_vol = np.clip(1.0 - (ann_vol / 0.65), 0.05, 1.0)
        
        # Prefer oversold/healthy momentum (RSI ~ 35-55) over extreme overbought (>75)
        if rsi < 30:
            mu_rsi = 0.85
        elif rsi <= 60:
            mu_rsi = 0.75 - 0.25 * ((rsi - 30.0) / 30.0)
        else:
            mu_rsi = np.clip(0.50 - 0.40 * ((rsi - 60.0) / 40.0), 0.10, 0.50)
            
        # Defuzzified composite score (0 - 100 scale)
        score = (0.45 * mu_sharpe + 0.30 * mu_low_vol + 0.25 * mu_rsi) * 100.0
        scores[col] = float(score)
        
    return scores

def run_walk_forward_backtest(
    returns_df: pd.DataFrame, 
    lookback_days: int = 126, 
    rebalance_days: int = 21, 
    tc_bps: float = 15.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes an out-of-sample Walk-Forward backtest comparing:
    1. Dynamic Markowitz (SLSQP Max Sharpe)
    2. Dynamic Fuzzy Logic Portfolio
    3. Equal-Weighted Benchmark (1/N)
    Deducts transaction costs (tc_bps in basis points) at each rebalancing date.
    """
    clean_df = returns_df.dropna(how='all').fillna(0.0)
    n_assets = len(clean_df.columns)
    tc_rate = tc_bps / 10000.0  # e.g., 15 bps = 0.0015 (0.15%)
    
    if len(clean_df) <= lookback_days + rebalance_days:
        lookback_days = max(30, len(clean_df) // 3)
        
    dates = clean_df.index[lookback_days:]
    
    markowitz_daily_ret = pd.Series(index=dates, dtype=float)
    fuzzy_daily_ret = pd.Series(index=dates, dtype=float)
    equal_daily_ret = pd.Series(index=dates, dtype=float)
    
    w_markowitz_prev = np.zeros(n_assets)
    w_fuzzy_prev = np.zeros(n_assets)
    w_equal = np.ones(n_assets) / n_assets
    
    turnover_markowitz = []
    turnover_fuzzy = []
    
    for start_idx in range(lookback_days, len(clean_df), rebalance_days):
        end_idx = min(start_idx + rebalance_days, len(clean_df))
        
        train_window = clean_df.iloc[start_idx - lookback_days : start_idx]
        test_window = clean_df.iloc[start_idx : end_idx]
        
        # 1. Optimize Markowitz strictly on training window
        mean_ret = train_window.mean()
        cov_mat = train_window.cov()
        try:
            w_markowitz = maximize_sharpe_ratio(mean_ret, cov_mat)
        except Exception:
            w_markowitz = w_equal.copy()
            
        # 2. Compute Fuzzy Weights strictly on training window
        fuzzy_scores = _compute_rolling_fuzzy_scores(train_window)
        fuzzy_dict = get_fuzzy_weighted_portfolio(fuzzy_scores)
        w_fuzzy = np.array([fuzzy_dict[col] for col in clean_df.columns])
        
        # 3. Calculate Turnover & Transaction Costs
        to_m = float(np.sum(np.abs(w_markowitz - w_markowitz_prev)))
        to_f = float(np.sum(np.abs(w_fuzzy - w_fuzzy_prev)))
        turnover_markowitz.append(to_m)
        turnover_fuzzy.append(to_f)
        
        cost_m = to_m * tc_rate
        cost_f = to_f * tc_rate
        
        w_markowitz_prev = w_markowitz.copy()
        w_fuzzy_prev = w_fuzzy.copy()
        
        # 4. Out-of-Sample Evaluation on unseen test_window
        m_rets = test_window.values @ w_markowitz
        f_rets = test_window.values @ w_fuzzy
        e_rets = test_window.values @ w_equal
        
        # Deduct transaction cost on the first day of the rebalancing period
        if len(m_rets) > 0:
            m_rets[0] -= cost_m
            f_rets[0] -= cost_f
            
        markowitz_daily_ret.iloc[start_idx - lookback_days : end_idx - lookback_days] = m_rets
        fuzzy_daily_ret.iloc[start_idx - lookback_days : end_idx - lookback_days] = f_rets
        equal_daily_ret.iloc[start_idx - lookback_days : end_idx - lookback_days] = e_rets

    # Build Cumulative Wealth Index (Starting at $1.00)
    cum_df = pd.DataFrame({
        'Fuzzy Logic (Out-of-Sample)': (1.0 + fuzzy_daily_ret).cumprod(),
        'Markowitz SLSQP (Out-of-Sample)': (1.0 + markowitz_daily_ret).cumprod(),
        'Equal Weight (1/N Benchmark)': (1.0 + equal_daily_ret).cumprod()
    })
    
    # Calculate Summary Performance Table
    summary_rows = []
    for name, s_ret, to_list in [
        ('Fuzzy Logic (Out-of-Sample)', fuzzy_daily_ret, turnover_fuzzy),
        ('Markowitz SLSQP (Out-of-Sample)', markowitz_daily_ret, turnover_markowitz),
        ('Equal Weight (1/N Benchmark)', equal_daily_ret, [0.0])
    ]:
        ann_ret = s_ret.mean() * 252
        ann_vol = s_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        
        cum_s = (1.0 + s_ret).cumprod()
        peak = cum_s.cummax()
        mdd = ((cum_s - peak) / peak).min()
        
        summary_rows.append({
            'Strategy': name,
            'Total Return': f"{(cum_s.iloc[-1] - 1.0) * 100:.1f}%",
            'Ann. Return': f"{ann_ret * 100:.1f}%",
            'Ann. Volatility': f"{ann_vol * 100:.1f}%",
            'Out-of-Sample Sharpe': f"{sharpe:.2f}",
            'Max Drawdown': f"{mdd * 100:.1f}%",
            'Avg Monthly Turnover': f"{np.mean(to_list) * 100:.1f}%"
        })
        
    summary_df = pd.DataFrame(summary_rows)
    return cum_df, summary_df