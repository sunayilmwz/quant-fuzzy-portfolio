# 📈 Quant Fuzzy Portfolio Optimizer & Risk Terminal 

Hi there! 👋 Welcome to an interactive quantitative finance engine where I combine classic portfolio theory, econometric volatility modeling, and machine learning. 

**[Try the Live Web App Here!](https://quant-fuzzy-portfolio.streamlit.app/)**

## 🚀 Why Did I Build This?
Traditional portfolio optimization models like Markowitz Mean Variance have a major flaw. They are mathematically ruthless and suffer from look ahead bias when tested only on static historical data. They often dump the entire budget into just two assets to maximize the Sharpe ratio, resulting in massive portfolio turnover and heavy trading costs.

I built this multi tab terminal to solve these real world quantitative challenges. By designing a Mamdani Fuzzy Logic scoring system, the engine evaluates assets much like a human portfolio manager would. It balances momentum, volatility, and risk adjusted returns to build a well diversified portfolio that survives out of sample market stress.

## 🧠 What is Happening Under the Hood?
I designed this project as a full scale web application divided into three core analytical modules:

* **Portfolio Allocation & Markowitz (Tab 1):** Pulls live historical data for BIST 30 and S&P 500 universes[cite: 4, 5]. Calculates Beta, Sharpe, Maximum Drawdown, and Historical Value at Risk[cite: 5, 10], while running a 10,000 iteration Monte Carlo simulation to plot the Efficient Frontier[cite: 5, 9].
* **GARCH Volatility & ML Market Regimes (Tab 2):** Replaces static standard deviation with an econometric GARCH(1,1) model to capture volatility clustering and forecast next period shocks. Uses an unsupervised Gaussian Mixture Model (GMM) to classify current market conditions into Bull, Transition, or Crisis regimes[cite: 5, 7].
* **Walk Forward Backtesting Engine (Tab 3):** Eliminates look ahead bias by simulating rolling training and testing windows over time. Deducts real world transaction costs (in basis points) at each monthly rebalancing step to prove out of sample performance against Markowitz and Equal Weight benchmarks[cite: 5, 6].

## 🏗️ Modular Code Architecture
Instead of a single monolithic script, the mathematical engines are separated into clean modules:

* `data_loader.py`: Handles live market API connections, benchmark alignment, and time series cleaning[cite: 4].
* `metrics_risk.py`: Computes annualized performance metrics, systematic Beta, and tail risk (VaR & CVaR)[cite: 10].
* `fuzzy_engine.py`: Defines membership functions and Mamdani inference rules for asset scoring[cite: 3].
* `optimizer.py`: Runs SciPy SLSQP optimization, Monte Carlo simulations, and fuzzy capital allocation[cite: 9].
* `regime_garch.py`: Executes GARCH(1,1) econometric modeling and GMM unsupervised regime clustering[cite: 7].
* `backtester.py`: Powers the rolling window Walk Forward simulation and turnover cost deductions[cite: 6].
* `app.py`: The multi tab Streamlit dashboard interface.

## 🛠️ Run It Locally
1. Clone this repository to your local machine
2. Create and activate your Python virtual environment
3. Install the dependencies from the requirements file
4. Run the application file using Streamlit

Feel free to reach out if you want to chat about quantitative analysis, econometrics, or data science!
