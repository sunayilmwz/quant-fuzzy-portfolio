import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, List

# Pre-defined popular tickers for the UI
BIST30_TICKERS = [
    "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ISCTR.IS", "YKBNK.IS",
    "TUPRS.IS", "ASELS.IS", "KCHOL.IS", "SAHOL.IS", "BIMAS.IS",
    "SISE.IS", "EREGL.IS", "FROTO.IS", "TOASO.IS", "TCELL.IS",
    "TTKOM.IS", "PGSUS.IS", "MGROS.IS", "ASTOR.IS", "KONTR.IS"
]

SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "MA", "JNJ", "PG",
    "HD", "AVGO", "MRK", "COST", "ABBV"
]

MARKET_BENCHMARKS = {
    "BIST-30": "XU100.IS",
    "S&P 500": "^GSPC"
}


def fetch_market_data(
    tickers: List[str],
    market: str = "BIST-30",
    period: str = "3y"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Fetches historical closing prices for selected assets and the market benchmark,
    cleans the data, and calculates daily returns.

    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols (e.g., ['THYAO.IS', 'GARAN.IS']).
    market : str
        Market name for benchmark selection ('BIST-30' or 'S&P 500').
    period : str
        Data retrieval period ('1y', '3y', '5y').

    Returns:
    --------
    prices_df : pd.DataFrame
        Daily closing prices of the selected assets.
    benchmark_prices : pd.Series
        Daily closing prices of the market benchmark.
    returns_df : pd.DataFrame
        Daily percentage returns of the selected assets.
    benchmark_returns : pd.Series
        Daily percentage returns of the market benchmark.
    """
    benchmark_ticker = MARKET_BENCHMARKS.get(market, "XU100.IS")
    all_symbols = list(set(tickers + [benchmark_ticker]))

    # Fetch adjusted data via yfinance (handles splits and dividends)
    raw_data = yf.download(
        all_symbols,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if raw_data.empty:
        raise ValueError("Failed to fetch data. Please check your internet connection or the ticker symbols.")

    # Handle MultiIndex returned by newer yfinance versions
    if isinstance(raw_data.columns, pd.MultiIndex):
        close_prices = raw_data["Close"].copy()
    else:
        close_prices = raw_data.copy()

    # Forward fill missing data (e.g., holidays) then drop remaining NaNs
    close_prices = close_prices.ffill().dropna()

    # Separate benchmark from portfolio assets
    benchmark_prices = close_prices[benchmark_ticker].copy()
    prices_df = close_prices[tickers].copy()

    # Calculate simple daily returns
    returns_df = prices_df.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()

    # Align dates precisely (intersection of indices)
    common_index = returns_df.index.intersection(benchmark_returns.index)
    returns_df = returns_df.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]

    return prices_df, benchmark_prices, returns_df, benchmark_returns


# --- TEST BLOCK ---
if __name__ == "__main__":
    test_tickers = ["THYAO.IS", "GARAN.IS", "ASELS.IS", "TUPRS.IS"]
    print(f"Fetching data for {test_tickers} and BIST-100 benchmark...")

    prices, bench_p, returns, bench_r = fetch_market_data(
        tickers=test_tickers,
        market="BIST-30",
        period="3y"
    )

    print("\n✅ Data fetched successfully!")
    print(f"Total Trading Days: {len(returns)}")
    print("\nLast 5 Days of Closing Prices:")
    print(prices.tail())
    print("\nAverage Annualized Returns (%):")
    print((returns.mean() * 252 * 100).round(2))