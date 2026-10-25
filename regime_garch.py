import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.mixture import GaussianMixture
from typing import Dict, Tuple

def fit_garch_volatility(returns_series: pd.Series) -> Tuple[pd.Series, float, Dict[str, float]]:
    """
    Fits a GARCH(1,1) econometric model to a return series to estimate 
    conditional volatility and forecast next-period annualized volatility.
    """
    clean_returns = returns_series.dropna() * 100.0
    
    model = arch_model(clean_returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
    res = model.fit(disp='off')
    
    cond_vol_annualized = (res.conditional_volatility / 100.0) * np.sqrt(252)
    
    forecast = res.forecast(horizon=5)
    next_vol_daily = np.sqrt(forecast.variance.values[-1, :]).mean() / 100.0
    forecasted_annual_vol = float(next_vol_daily * np.sqrt(252))
    
    alpha = float(res.params.get('alpha[1]', 0.0))
    beta = float(res.params.get('beta[1]', 0.0))
    persistence = alpha + beta
    
    params = {
        'omega': float(res.params.get('omega', 0.0)),
        'alpha': alpha,
        'beta': beta,
        'persistence': persistence
    }
    
    return cond_vol_annualized, forecasted_annual_vol, params

def detect_market_regimes(benchmark_returns: pd.Series, n_regimes: int = 3) -> Tuple[pd.DataFrame, str]:
    """
    Uses an Unsupervised Gaussian Mixture Model (GMM) on rolling return and 
    volatility features to classify market states into Bull, Transition, or Crisis regimes.
    """
    df = pd.DataFrame({'return': benchmark_returns.dropna()})
    df['rolling_ret'] = df['return'].rolling(window=21).mean() * 252
    df['rolling_vol'] = df['return'].rolling(window=21).std() * np.sqrt(252)
    df = df.dropna().copy()
    
    features = df[['rolling_ret', 'rolling_vol']].values
    
    gmm = GaussianMixture(n_components=n_regimes, covariance_type='full', random_state=42)
    df['cluster'] = gmm.fit_predict(features)
    
    cluster_vols = df.groupby('cluster')['rolling_vol'].mean().sort_values()
    ordered_clusters = cluster_vols.index.tolist()
    
    regime_labels = {
        ordered_clusters[0]: 'Bull / Low Volatility',
        ordered_clusters[1]: 'Neutral / Transition',
        ordered_clusters[2]: 'Bear / High-Vol Crisis'
    }
    
    df['Regime'] = df['cluster'].map(regime_labels)
    df['Cumulative_Market'] = (1 + df['return']).cumprod()
    
    current_regime = df['Regime'].iloc[-1]
    return df, current_regime