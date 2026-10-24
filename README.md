# 📈 Quant-Fuzzy Portfolio Optimizer 🧠

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quant-fuzzy-portfolio.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An institutional-grade, interactive quantitative finance web application that optimizes asset allocation by comparing traditional **Markowitz Mean-Variance** optimization (SLSQP) against a custom **Fuzzy Logic (Mamdani)** inference system.

**[Try the Live Web App Here!](https://quant-fuzzy-portfolio.streamlit.app/)**

## 🚀 The Story & Problem it Solves
Traditional optimization algorithms (like SciPy's SLSQP maximizing the Sharpe ratio) often hyper-concentrate capital into just a few assets, completely ignoring market momentum or qualitative risk transitions. In the real world, financial markets aren't black and white. 

This project solves that by introducing a **Fuzzy Logic Scoring Engine**. Instead of hard thresholds, it evaluates assets based on fuzzy sets (e.g., *Is the RSI oversold? Is the volatility medium?*), mimicking how a human portfolio manager balances risk and reward to generate a highly diversified, mathematically sound portfolio.

## ⚙️ Core Features
- **Live Market Data Pipeline:** Automatically fetches multi-year historical data for BIST-30 or S&P 500 assets via `yfinance`.
- **Econometric & Risk Engine:** Calculates annualized volatility, Beta, Sharpe Ratio, Sortino Ratio, and Max Drawdown.
- **Tail Risk Analytics:** Computes 95% Historical and Parametric Value at Risk (VaR) and Conditional VaR (Expected Shortfall).
- **Fuzzy Intelligence Module:** Uses `scikit-fuzzy` to map Volatility, RSI, and Sharpe ratios into fuzzy logic rules, generating an asset strength score (0-100).
- **Monte Carlo Simulation:** Runs 10,000 randomized iterations to visualize the Efficient Frontier.
- **Interactive UI:** Fully dynamic dashboard built with `Streamlit` and `Plotly`.

## 🏗️ Technical Architecture
The codebase is structured following software engineering best practices, split into modular engines:
* `data_loader.py`: Handles API connections, multi-index parsing, and data cleaning.
* `metrics_risk.py`: Core statistical and risk calculations.
* `fuzzy_engine.py`: Defines antecedents, consequents, and the rules of the Mamdani control system.
* `optimizer.py`: Markowitz calculations, Monte Carlo simulations, and Fuzzy allocation mapping.
* `app.py`: The Streamlit presentation layer.

## 🛠️ Local Installation & Usage
If you want to run this quantitative engine on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/quant-fuzzy-portfolio.git](https://github.com/YOUR_GITHUB_USERNAME/quant-fuzzy-portfolio.git)
   cd quant-fuzzy-portfolio
