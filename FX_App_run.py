import streamlit as st
import pymc as pm
import pyomo.environ as pyo
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt

# ==========================================
# 1. OANDA BROKER CONFIGURATION
# ==========================================
class OandaBroker:
    def __init__(self, api_key, account_id, environment="practice"):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = "https://api-fxpractice.oanda.com/v3" if environment == "practice" else "https://api-fxtrade.oanda.com/v3"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_allied_prices(self):
        instruments = "EUR_USD,GBP_USD,USD_CAD,AUD_USD,USD_JPY"
        endpoint = f"{self.base_url}/accounts/{self.account_id}/pricing"
        params = {"instruments": instruments}
        response = requests.get(endpoint, headers=self.headers, params=params)
        
        if response.status_code == 200:
            prices = response.json().get('prices', [])
            return {p['instrument']: float(p['closeoutAsk']) for p in prices}
        return None

    def execute_trade(self, instrument, units):
        endpoint = f"{self.base_url}/accounts/{self.account_id}/orders"
        order_payload = {
            "order": {
                "instrument": instrument,
                "units": str(int(units)),
                "type": "MARKET",
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }
        response = requests.post(endpoint, headers=self.headers, data=json.dumps(order_payload))
        return response.status_code == 201

def calculate_oanda_units(allocation_usd, pair, current_price, expected_return):
    direction = 1 if expected_return > 0 else -1
    base_currency, quote_currency = pair.split('_')
    
    if base_currency == 'USD':
        units = allocation_usd
    elif quote_currency == 'USD':
        units = allocation_usd / current_price
    else:
        units = 0
        
    return int(round(units)) * direction

# ==========================================
# 2. DATA PIPELINE (FX + MACRO FEATURES)
# ==========================================
FX_MAP = {
    "EURUSD=X": "EUR_USD",
    "GBPUSD=X": "GBP_USD",
    "CAD=X": "USD_CAD",
    "AUDUSD=X": "AUD_USD",
    "JPY=X": "USD_JPY"
}

@st.cache_data
def fetch_market_data(days=300):
    fx_data = yf.download(list(FX_MAP.keys()), period=f"{days}d")['Close']
    fx_returns = fx_data.pct_change().dropna()
    
    macro_data = yf.download(["^TNX", "^VIX"], period=f"{days}d")['Close']
    macro_features = macro_data.pct_change().dropna()
    
    aligned_data = fx_returns.join(macro_features, how='inner').dropna()
    
    return aligned_data[list(FX_MAP.keys())], aligned_data[["^TNX", "^VIX"]]

# ==========================================
# 3. BAYESIAN STRUCTURAL TIME SERIES (MCMC)
# ==========================================
@st.cache_resource
def run_bsts_feature_selection(y_returns, X_macro_features):
    num_predictors = X_macro_features.shape[1]
    
    with pm.Model() as bsts_model:
        volatility = pm.Exponential("volatility", 1.0)
        drift = pm.Normal("drift", mu=0, sigma=0.01)
        
        tau = pm.HalfCauchy("tau", beta=0.1) 
        lam = pm.HalfCauchy("lam", beta=1.0, shape=num_predictors)
        beta = pm.Normal("beta", mu=0, sigma=tau * lam, shape=num_predictors)
        
        regression_effect = pm.math.dot(X_macro_features.values, beta)
        mu_t = drift + regression_effect
        
        obs = pm.StudentT("obs", nu=3, mu=mu_t, sigma=volatility, observed=y_returns.values)
        
        trace = pm.sample(draws=300, tune=300, target_accept=0.95, progressbar=False)
        
    posterior_mu = trace.posterior['drift'].mean().item()
    posterior_var = trace.posterior['volatility'].mean().item() ** 2
    
    return posterior_mu, posterior_var

# ==========================================
# 4. PYOMO OPTIMIZATION (FRACTIONAL KELLY)
# ==========================================
def optimize_portfolio(posterior_results, total_capital=4000, max_position=1000):
    model = pyo.ConcreteModel()
    
    model.pairs = pyo.Set(initialize=list(posterior_results.keys()))
    model.mu = pyo.Param(model.pairs, initialize={k: v[0] for k, v in posterior_results.items()})
    model.var = pyo.Param(model.pairs, initialize={k: v[1] for k, v in posterior_results.items()})
    
    model.x = pyo.Var(model.pairs, domain=pyo.NonNegativeReals, bounds=(0, max_position))
    
    risk_aversion = 2.0 
    def objective_rule(model):
        expected_profit = sum(model.x[i] * abs(model.mu[i]) for i in model.pairs)
        risk_penalty = sum((model.x[i]**2) * model.var[i] for i in model.pairs) * risk_aversion
        return expected_profit - risk_penalty
        
    model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    def capital_limit_rule(model):
        return sum(model.x[i] for i in model.pairs) <= total_capital
    model.CapitalConstraint = pyo.Constraint(rule=capital_limit_rule)
    
    # ---------------------------------------------------------
    # DUAL SOLVER LOGIC: Try Gurobi first, fallback to IPOPT
    # ---------------------------------------------------------
    solver_used = "None"
    try:
        solver = pyo.SolverFactory('gurobi')
        if solver.available(exception_flag=False):
            solver.solve(model)
            solver_used = "Gurobi"
        else:
            raise ValueError("Gurobi unavailable")
    except Exception:
        solver = pyo.SolverFactory('ipopt')
        solver.solve(model)
        solver_used = "Ipopt (Open-Source Fallback)"
        
    allocations = {i: pyo.value(model.x[i]) for i in model.pairs}
    return allocations, solver_used

# ==========================================
# 5. STREAMLIT DASHBOARD
# ==========================================
st.set_page_config(page_title="Bayesian OR Swing Trader", layout="wide")
st.title("Bayesian FX Simulation & Optimization Engine")
st.markdown("Targeting structural swing trades within the allied currency universe using a $4000 constraint.")

try:
    api_key = st.secrets["oanda"]["api_key"]
    account_id = st.secrets["oanda"]["account_id"]
    broker = OandaBroker(api_key, account_id)
except Exception:
    st.error("Missing `.streamlit/secrets.toml` file with OANDA credentials or credentials not set in Streamlit Cloud.")
    st.stop()

if st.button("Run Daily Quantitative Pipeline"):
    with st.spinner("Fetching market data and macro features..."):
        fx_returns, macro_features = fetch_market_data()
        live_prices = broker.get_allied_prices()
        
    if live_prices is None:
        st.error("Failed to fetch live prices from OANDA. Check API token.")
        st.stop()
        
    posteriors = {}
    
    st.write("### 1. Running MCMC Inference (Horseshoe Prior)")
    progress_bar = st.progress(0)
    
    for i, (yf_ticker, oanda_pair) in enumerate(FX_MAP.items()):
        with st.spinner(f"Sampling posterior geometry for {oanda_pair}..."):
            mu, var = run_bsts_feature_selection(fx_returns[yf_ticker], macro_features)
            posteriors[oanda_pair] = (mu, var)
        progress_bar.progress((i + 1) / len(FX_MAP))
        
    st.write("### 2. Pyomo Fractional Kelly Allocations")
    
    # Run optimization and capture which solver was used
    allocations_usd, solver_used = optimize_portfolio(posteriors, total_capital=4000, max_position=1000)
    st.info(f"Optimization computed successfully using: **{solver_used}**")
    
    execution_plan = []
    for pair, alloc_usd in allocations_usd.items():
        if alloc_usd > 10:
            expected_ret = posteriors[pair][0]
            price = live_prices[pair]
            units = calculate_oanda_units(alloc_usd, pair, price, expected_ret)
            
            execution_plan.append({
                "Pair": pair,
                "USD Allocation": f"${alloc_usd:.2f}",
                "Expected 5-Day Drift": f"{expected_ret:.4%}",
                "Direction": "LONG" if expected_ret > 0 else "SHORT",
                "OANDA Units": units
            })
            
    df_plan = pd.DataFrame(execution_plan)
    st.table(df_plan)
    st.session_state['execution_plan'] = execution_plan

if 'execution_plan' in st.session_state and len(st.session_state['execution_plan']) > 0:
    st.write("### 3. Market Execution")
    if st.button("Transmit Orders to OANDA Practice Net"):
        for trade in st.session_state['execution_plan']:
            success = broker.execute_trade(trade["Pair"], trade["OANDA Units"])
            if success:
                st.success(f"Filled: {trade['OANDA Units']} units of {trade['Pair']}")
            else:
                st.error(f"Failed to execute {trade['Pair']}")
        del st.session_state['execution_plan']
