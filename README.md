# 📈 Quant-Fuzzy Portfolio Optimizer 🧠

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quant-fuzzy-portfolio.streamlit.app/)

Hi there! 👋 Welcome to a project where I bring classic quantitative finance and modern machine learning together. 

**[Try the Live Web App Here!](https://quant-fuzzy-portfolio.streamlit.app/)**

## 🚀 Why Did I Build This?
If you've ever played around with traditional portfolio optimization (like the Markowitz Mean-Variance model), you might have noticed a recurring issue: the algorithm is mathematically ruthless. It will happily dump 100% of your budget into just two assets if it means maximizing the Sharpe ratio. 

But human portfolio managers don't work like that. The market isn't just black and white; there are gray areas and transitions. 

I built this interactive engine to see if I could teach a machine to think a bit more flexibly. By using a **Fuzzy Logic (Mamdani) scoring system**, the app evaluates assets much like a human would—balancing momentum (RSI), volatility, and risk-adjusted returns—to create a much more realistic, well-diversified portfolio.

## 🧠 What's Happening Under the Hood?
I wanted this to be more than just a static Jupyter Notebook, so I built it as a fully interactive web app.

- **Live Data Pipeline:** Pulls multi-year historical data for BIST-30 or S&P 500 assets on the fly.
- **Risk Analytics:** Calculates the heavy quantitative metrics: Beta, Sharpe, Sortino, Max Drawdown, and 95% Value at Risk (VaR & CVaR).
- **The Showdown:** Pits a classic SciPy SLSQP optimizer directly against my custom Fuzzy Logic weighting logic. 
- **Monte Carlo Simulation:** Runs 10,000 random portfolio iterations to dynamically plot the Efficient Frontier.

## 🏗️ Code Structure
I'm a big fan of clean, readable code. Instead of one massive spaghetti script, the logic is broken down into specific modular engines:
* `data_loader.py`: Handles the API connections and data wrangling.
* `metrics_risk.py`: The statistical and econometric math.
* `fuzzy_engine.py`: Where the fuzzy logic magic happens (defining the rules and memberships).
* `optimizer.py`: The portfolio allocation algorithms.
* `app.py`: The Streamlit dashboard that ties it all together.

## 🛠️ Run It Locally
Want to test the engine on your own machine? It's super simple:

```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/quant-fuzzy-portfolio.git](https://github.com/YOUR_GITHUB_USERNAME/quant-fuzzy-portfolio.git)
cd quant-fuzzy-portfolio
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
