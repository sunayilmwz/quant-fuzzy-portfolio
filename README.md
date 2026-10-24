# 📈 Quant Fuzzy Portfolio Optimizer 🧠

Hi there! 👋 Welcome to a project where I bring classic quantitative finance and modern machine learning together. 

**[Try the Live Web App Here!](https://quant-fuzzy-portfolio.streamlit.app/)**

## 🚀 Why Did I Build This?
If you have ever played around with traditional portfolio optimization, you might have noticed a recurring issue. The algorithm is mathematically ruthless. It will happily dump all your budget into just two assets if it means maximizing the Sharpe ratio. 

But human portfolio managers do not work like that. The market is not just black and white. There are gray areas and transitions. 

I built this interactive engine to see if I could teach a machine to think a bit more flexibly. By using a Fuzzy Logic scoring system, the app evaluates assets much like a human would. It balances momentum, volatility, and risk adjusted returns to create a much more realistic, well diversified portfolio.

## 🧠 What is Happening Under the Hood?
I wanted this to be more than just a static Jupyter Notebook, so I built it as a fully interactive web app.

* **Live Data Pipeline:** Pulls multi year historical data for BIST 30 or S&P 500 assets on the fly.
* **Risk Analytics:** Calculates the heavy quantitative metrics like Beta, Sharpe, Sortino, Max Drawdown, and Value at Risk.
* **The Showdown:** Pits a classic SciPy optimizer directly against my custom Fuzzy Logic weighting logic. 
* **Monte Carlo Simulation:** Runs 10000 random portfolio iterations to dynamically plot the Efficient Frontier.

## 🏗️ Code Structure
I am a big fan of clean and readable code. Instead of one massive spaghetti script, the logic is broken down into specific modular engines:

* `data_loader.py`: Handles the API connections and data wrangling.
* `metrics_risk.py`: The statistical and econometric math.
* `fuzzy_engine.py`: Where the fuzzy logic magic happens.
* `optimizer.py`: The portfolio allocation algorithms.
* `app.py`: The Streamlit dashboard that ties it all together.

## 🛠️ Run It Locally
Want to test the engine on your own machine? It is super simple:

1. Clone this repository
2. Set up your Python virtual environment
3. Install the requirements file
4. Run the app file with Streamlit

Feel free to reach out if you want to chat about quantitative analysis, statistics, or fuzzy logic!
