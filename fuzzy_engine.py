import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculates the latest Relative Strength Index (RSI) value for a price series."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    clean_rsi = rsi.dropna()
    return float(clean_rsi.iloc[-1]) if not clean_rsi.empty else 50.0

def compute_fuzzy_score(sharpe: float, volatility: float, rsi: float) -> float:
    """Evaluates an asset (0-100) using a Mamdani Fuzzy Logic inference system."""
    sharpe_univ = np.arange(-1.0, 3.01, 0.05)
    vol_univ = np.arange(0.0, 1.01, 0.01)
    rsi_univ = np.arange(0.0, 100.1, 1.0)
    score_univ = np.arange(0.0, 100.1, 1.0)

    sharpe_ctrl = ctrl.Antecedent(sharpe_univ, 'sharpe')
    vol_ctrl = ctrl.Antecedent(vol_univ, 'volatility')
    rsi_ctrl = ctrl.Antecedent(rsi_univ, 'rsi')
    score_ctrl = ctrl.Consequent(score_univ, 'score')

    sharpe_ctrl['poor'] = fuzz.trapmf(sharpe_univ, [-1.0, -1.0, 0.0, 0.6])
    sharpe_ctrl['good'] = fuzz.trimf(sharpe_univ, [0.2, 1.0, 1.8])
    sharpe_ctrl['excellent'] = fuzz.trapmf(sharpe_univ, [1.2, 2.0, 3.0, 3.0])

    vol_ctrl['low'] = fuzz.trapmf(vol_univ, [0.0, 0.0, 0.18, 0.30])
    vol_ctrl['medium'] = fuzz.trimf(vol_univ, [0.20, 0.35, 0.55])
    vol_ctrl['high'] = fuzz.trapmf(vol_univ, [0.45, 0.65, 1.0, 1.0])

    rsi_ctrl['oversold'] = fuzz.trapmf(rsi_univ, [0.0, 0.0, 30.0, 45.0])
    rsi_ctrl['neutral'] = fuzz.trimf(rsi_univ, [35.0, 50.0, 65.0])
    rsi_ctrl['overbought'] = fuzz.trapmf(rsi_univ, [55.0, 70.0, 100.0, 100.0])

    score_ctrl['low'] = fuzz.trimf(score_univ, [0.0, 20.0, 45.0])
    score_ctrl['medium'] = fuzz.trimf(score_univ, [35.0, 50.0, 65.0])
    score_ctrl['high'] = fuzz.trimf(score_univ, [55.0, 80.0, 100.0])

    rules = [
        ctrl.Rule(sharpe_ctrl['excellent'] & (vol_ctrl['low'] | vol_ctrl['medium']), score_ctrl['high']),
        ctrl.Rule(sharpe_ctrl['good'] & rsi_ctrl['oversold'], score_ctrl['high']),
        ctrl.Rule(sharpe_ctrl['good'] & rsi_ctrl['neutral'] & vol_ctrl['medium'], score_ctrl['medium']),
        ctrl.Rule(sharpe_ctrl['poor'] | vol_ctrl['high'], score_ctrl['low']),
        ctrl.Rule(rsi_ctrl['overbought'] & vol_ctrl['high'], score_ctrl['low']),
        ctrl.Rule(sharpe_ctrl['good'] & rsi_ctrl['overbought'], score_ctrl['medium'])
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)

    sim.input['sharpe'] = float(np.clip(sharpe, -0.95, 2.95))
    sim.input['volatility'] = float(np.clip(volatility, 0.01, 0.99))
    sim.input['rsi'] = float(np.clip(rsi, 1.0, 99.0))

    try:
        sim.compute()
        return float(sim.output['score'])
    except Exception:
        return 50.0

def get_fuzzy_score(vol_val: float, rsi_val: float, sharpe_val: float) -> float:
    """Wrapper function matching app.py signature."""
    return compute_fuzzy_score(sharpe=sharpe_val, volatility=vol_val, rsi=rsi_val)