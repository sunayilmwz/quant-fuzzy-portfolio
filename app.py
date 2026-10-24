import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import our custom quantitative modules
from data_loader import fetch_market_data, BIST30_TICKERS, SP500_TICKERS, MARKET_BENCHMARKS
from metrics_risk import calculate_annualized_metrics, calculate_beta, calculate_tail_risk
from fuzzy_engine import calculate_rsi, get_fuzzy_score
from optimizer import maximize_sharpe_ratio, generate_random_portfolios, get_fuzzy_weighted_portfolio

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quant & Fuzzy Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 2. SIDEBAR SETUP ---
st.sidebar.title("⚙️ Portfolio Parameters")
st.sidebar.markdown("Configure your investment universe and constraints below.")

market_choice = st.sidebar.radio("Select Market:", ["BIST-30", "S&P 500"])
benchmark_ticker = MARKET_BENCHMARKS[market_choice]

# Dynamically change available tickers based on market selection
if market_choice == "BIST-30":
    available_tickers = BIST30_TICKERS
    default_tickers = ["THYAO.IS", "GARAN.IS", "TUPRS.IS", "BIMAS.IS", "SISE.IS"]
else:
    available_tickers = SP500_TICKERS
    default_tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]

selected_tickers = st.sidebar.multiselect(
    "Select Assets (Min 3 recommended):",
    options=available_tickers,
    default=default_tickers
)

time_period = st.sidebar.selectbox("Historical Data Range:", ["1y", "3y", "5y"], index=1)
budget = st.sidebar.number_input("Investment Budget ($ / ₺):", min_value=1000, value=100000, step=5000)

run_button = st.sidebar.button("🚀 Run Quantitative Analysis", use_container_width=True)

# --- 3. MAIN DASHBOARD AREA ---
st.title("🧠 Quantitative Portfolio Optimization & Fuzzy Logic Scoring")
st.markdown(f"""
Welcome to the interactive engine! This tool fetches live data, calculates institutional-grade risk metrics (VaR, CVaR), 
and compares a traditional **Markowitz Mean-Variance** optimization against a custom **Fuzzy Logic** scoring system.
*Current Benchmark: **{benchmark_ticker}***
""")

if run_button:
    if len(selected_tickers) < 2:
        st.error("Please select at least 2 assets to form a portfolio.")
    else:
        with st.spinner("Fetching market data and running quantitative engines..."):
            
            # --- A. DATA FETCHING ---
            try:
                prices_df, bench_prices, returns_df, bench_returns = fetch_market_data(
                    tickers=selected_tickers, 
                    market=market_choice, 
                    period=time_period
                )
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.stop()

            # --- B. ENGINE CALCULATIONS ---
            # 1. Asset Level Metrics & Fuzzy Scoring
            asset_stats = []
            fuzzy_scores_dict = {}
            
            for ticker in selected_tickers:
                asset_ret = returns_df[ticker]
                asset_price = prices_df[ticker]
                
                # Risk & Return
                metrics = calculate_annualized_metrics(asset_ret)
                beta = calculate_beta(asset_ret, bench_returns)
                tail = calculate_tail_risk(asset_ret, confidence_level=0.95)
                
                # Fuzzy Logic
                rsi_val = calculate_rsi(asset_price)
                vol = metrics["Annual Volatility"]
                sharpe = metrics["Sharpe Ratio"]
                f_score = get_fuzzy_score(vol_val=vol, rsi_val=rsi_val, sharpe_val=sharpe)
                
                fuzzy_scores_dict[ticker] = f_score
                
                asset_stats.append({
                    "Asset": ticker,
                    "Fuzzy Score (0-100)": round(f_score, 1),
                    "RSI": round(rsi_val, 1),
                    "Beta": round(beta, 2),
                    "Ann. Return": f"{metrics['Annual Return']*100:.1f}%",
                    "Volatility": f"{metrics['Annual Volatility']*100:.1f}%",
                    "Sharpe": round(metrics['Sharpe Ratio'], 2),
                    "Max Drawdown": f"{metrics['Max Drawdown']*100:.1f}%",
                    "VaR 95% (Hist)": f"{tail['Historical VaR (95.0%)']*100:.2f}%"
                })

            stats_df = pd.DataFrame(asset_stats).set_index("Asset")
            
            # 2. Optimization (Markowitz vs Fuzzy)
            mean_returns = returns_df.mean()
            cov_matrix = returns_df.cov()
            
            markowitz_weights = maximize_sharpe_ratio(mean_returns, cov_matrix)
            fuzzy_weights_dict = get_fuzzy_weighted_portfolio(fuzzy_scores_dict)
            fuzzy_weights = np.array([fuzzy_weights_dict[t] for t in selected_tickers])
            
            # --- C. VISUALIZATIONS ---
            
            st.divider()
            st.subheader("📊 1. Asset Diagnostics & Fuzzy Intelligence")
            st.dataframe(stats_df, use_container_width=True)
            
            st.divider()
            st.subheader("⚖️ 2. Portfolio Allocation: Math vs. Machine Intelligence")
            st.markdown("Traditional algorithms often hyper-concentrate into a few assets. Our Fuzzy Logic engine mimics a human manager, ensuring better diversification based on momentum and risk-adjusted returns.")
            
            col1, col2 = st.columns(2)
            
            # Pie Chart 1: Markowitz
            fig_markowitz = px.pie(
                values=markowitz_weights, 
                names=selected_tickers, 
                title="Max Sharpe Portfolio (SciPy SLSQP)",
                hole=0.4
            )
            fig_markowitz.update_traces(textposition='inside', textinfo='percent+label')
            col1.plotly_chart(fig_markowitz, use_container_width=True)
            
            # Pie Chart 2: Fuzzy
            fig_fuzzy = px.pie(
                values=fuzzy_weights, 
                names=selected_tickers, 
                title="Fuzzy Logic Weighted Portfolio",
                hole=0.4
            )
            fig_fuzzy.update_traces(textposition='inside', textinfo='percent+label')
            col2.plotly_chart(fig_fuzzy, use_container_width=True)
            
            # Investment Budget Breakdown
            st.markdown("#### 💰 Budget Allocation based on Fuzzy Weights")
            budget_cols = st.columns(len(selected_tickers))
            for i, ticker in enumerate(selected_tickers):
                allocated = budget * fuzzy_weights[i]
                budget_cols[i].metric(label=f"{ticker}", value=f"{allocated:,.2f}")

            st.divider()
            st.subheader("🎯 3. Monte Carlo Simulation & Efficient Frontier")
            st.markdown("Running 10,000 random portfolio combinations to visualize the risk-return landscape.")
            
            # Run Monte Carlo
            mc_results, mc_weights = generate_random_portfolios(returns_df, num_portfolios=10000)
            
            # Create Scatter Plot for Efficient Frontier
            mc_returns = mc_results[0, :] * 100  # Convert to percentage
            mc_vols = mc_results[1, :] * 100
            mc_sharpes = mc_results[2, :]
            
            fig_mc = go.Figure()
            
            # Add random portfolios
            fig_mc.add_trace(go.Scatter(
                x=mc_vols, 
                y=mc_returns, 
                mode='markers',
                marker=dict(
                    color=mc_sharpes,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                    size=5,
                    opacity=0.6
                ),
                name='Random Portfolios'
            ))
            
            # Highlight Max Sharpe
            max_sharpe_idx = np.argmax(mc_sharpes)
            fig_mc.add_trace(go.Scatter(
                x=[mc_vols[max_sharpe_idx]],
                y=[mc_returns[max_sharpe_idx]],
                mode='markers',
                marker=dict(color='red', size=15, symbol='star'),
                name='Max Sharpe (Optimal)'
            ))
            
            fig_mc.update_layout(
                title="Efficient Frontier (10,000 Iterations)",
                xaxis_title="Expected Annual Volatility (%)",
                yaxis_title="Expected Annual Return (%)",
                template="plotly_white",
                height=600
            )
            
            st.plotly_chart(fig_mc, use_container_width=True)
            st.success("✅ Analysis Complete! You can now adjust parameters in the sidebar to re-run the engine.")
else:
    st.info("👈 Please select your parameters from the sidebar and click 'Run Quantitative Analysis'.")