import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def calculate_rsi(prices: pd.Series, window: int = 14) -> float:
    """
    A quick helper function to calculate the Relative Strength Index (RSI).
    We need this to feed momentum into our fuzzy engine. 
    Returns the most recent RSI value.
    """
    deltas = prices.diff()
    
    # Separate the positive and negative gains
    gains = deltas.where(deltas > 0, 0.0)
    losses = -deltas.where(deltas < 0, 0.0)
    
    # Calculate the exponential moving average of gains and losses
    avg_gain = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = losses.ewm(com=window - 1, min_periods=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    
    # We only care about the latest momentum right now
    return rsi_series.iloc[-1] if not rsi_series.empty else 50.0

def build_fuzzy_system():
    """
    This is the brain of our scoring engine. 
    Instead of using hard thresholds (like "if volatility > 0.5 then bad"), 
    we use fuzzy sets to mimic how a human portfolio manager actually thinks.
    """
    
    # 1. Define our universe of inputs (Antecedents) and the final score (Consequent).
    # We set reasonable market ranges for each metric.
    volatility = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'volatility')
    rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
    sharpe = ctrl.Antecedent(np.arange(-3, 4.1, 0.1), 'sharpe')
    
    score = ctrl.Consequent(np.arange(0, 101, 1), 'score')

    # 2. Map the data into fuzzy categories (Low, Medium, High).
    # automf() automatically creates overlapping triangles for us.
    volatility.automf(names=['low', 'medium', 'high'])
    
    # For RSI: <30 is oversold (cheap/good), ~50 is neutral, >70 is overbought (expensive/risky)
    rsi.automf(names=['oversold', 'neutral', 'overbought'])
    
    # For Sharpe: <0 is poor, ~1 is acceptable, >2 is excellent
    sharpe.automf(names=['poor', 'acceptable', 'excellent'])
    
    # The final output score categories: 0-33 Avoid, 34-66 Hold, 67-100 Buy
    score.automf(names=['avoid', 'hold', 'buy'])

    # 3. Time to write the rules! 
    # This reads almost like plain English. It's how we balance risk and reward.
    rules = [
        # Deal-breakers: High risk or bad returns mean we stay away.
        ctrl.Rule(sharpe['poor'] | volatility['high'], score['avoid']),
        
        # The holy grail: Low risk, great returns.
        ctrl.Rule(sharpe['excellent'] & volatility['low'], score['buy']),
        
        # Finding bargains: The asset is oversold (cheap) and has decent historical returns.
        ctrl.Rule(rsi['oversold'] & sharpe['acceptable'], score['buy']),
        
        # Overheated assets: Everyone is buying (overbought), so we should be careful and just hold.
        ctrl.Rule(rsi['overbought'], score['hold']),
        
        # Middle of the road: Nothing crazy going on, just keeping the status quo.
        ctrl.Rule(sharpe['acceptable'] & volatility['medium'], score['hold']),
        
        # Good fundamentals but normal momentum.
        ctrl.Rule(rsi['neutral'] & sharpe['excellent'], score['buy'])
    ]

    # 4. Wire everything together into a simulation environment
    scoring_ctrl = ctrl.ControlSystem(rules)
    scoring_sim = ctrl.ControlSystemSimulation(scoring_ctrl)
    
    return scoring_sim

def get_fuzzy_score(vol_val: float, rsi_val: float, sharpe_val: float) -> float:
    """
    Feeds the raw financial metrics into our fuzzy engine and returns a clean 0-100 score.
    """
    engine = build_fuzzy_system()
    
    # Clip the inputs just in case the real-world data goes crazy (e.g., GME squeeze)
    engine.input['volatility'] = np.clip(vol_val, 0, 1.0)
    engine.input['rsi'] = np.clip(rsi_val, 0, 100)
    engine.input['sharpe'] = np.clip(sharpe_val, -3.0, 4.0)
    
    # Crunch the numbers
    try:
        engine.compute()
        return engine.output['score']
    except Exception:
        # Fallback if the logic breaks due to weird data
        return 50.0

# --- TEST BLOCK ---
if __name__ == "__main__":
    from data_loader import fetch_market_data
    from metrics_risk import calculate_annualized_metrics
    
    print("Waking up the Fuzzy Engine...")
    test_ticker = ["THYAO.IS"]
    prices, _, returns_df, _ = fetch_market_data(test_ticker, period="1y")
    
    asset_prices = prices["THYAO.IS"]
    asset_returns = returns_df["THYAO.IS"]
    
    # Get the raw metrics
    metrics = calculate_annualized_metrics(asset_returns)
    volatility = metrics["Annual Volatility"]
    sharpe = metrics["Sharpe Ratio"]
    rsi = calculate_rsi(asset_prices)
    
    # Feed them to the fuzzy engine
    final_score = get_fuzzy_score(vol_val=volatility, rsi_val=rsi, sharpe_val=sharpe)
    
    print("\n--- FUZZY SCORING RESULTS ---")
    print(f"Asset: {test_ticker[0]}")
    print(f"Raw Volatility : {volatility:.4f}")
    print(f"Raw RSI        : {rsi:.2f}")
    print(f"Raw Sharpe     : {sharpe:.4f}")
    print(f"-> FINAL FUZZY SCORE: {final_score:.2f} / 100")