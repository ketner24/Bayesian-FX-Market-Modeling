# Bayesian FX Trading Engine v2.0

## Architecture Change: Streamlit → FastAPI

**Why Streamlit is wrong for this project:**

| Problem | Streamlit | FastAPI (this version) |
|---|---|---|
| Stale inference | `@st.cache_resource` freezes MCMC results forever | Fresh inference every pipeline run |
| No scheduling | Requires a human clicking a button | APScheduler runs daily at 14:00 UTC |
| No audit trail | Ephemeral UI state | Structured logging, JSON responses |
| Stateful trading in reactive framework | Session state fights the re-run model | Explicit request/response, no hidden state |
| Deployment | Single process, UI + logic coupled | API and dashboard deploy independently |
| Testing | Can't unit test Streamlit callbacks easily | Standard pytest against API endpoints |

**How to run:**
```bash
pip install -r requirements.txt
python main.py                    # Starts API on port 8000
# Then: POST /pipeline/run with OANDA credentials
```

**For a dashboard**, add Plotly Dash, Grafana, or a React frontend that calls the API. The key insight is: *the trading engine should never depend on whether someone is looking at a UI*.

---

## Bug Fixes Applied

### Fix 1: Regression Betas Now Used (CRITICAL)
**Before:** `posterior_mu = trace.posterior['drift'].mean()` — threw away the entire regression.
**After:** Expected return = E[drift] + E[beta] @ x_last, where x_last is the most recent macro observation. The horseshoe prior and macro features now actually influence trade decisions.

### Fix 2: Student-T Variance Corrected
**Before:** `posterior_var = sigma ** 2` — understated variance by 3x.
**After:** `corrected_variance = sigma² × ν/(ν−2)` — for ν=3, this is `σ² × 3`. All pairs now show their true risk.

### Fix 3: Optimizer Handles Direction Natively
**Before:** `abs(model.mu[i])` made the optimizer direction-blind.
**After:** Separate `x_long` and `x_short` variables with signed expected returns. The optimizer itself decides whether to go long or short based on the sign of mu.

### Fix 4: Full Covariance Matrix
**Before:** Independent variance terms — EUR/USD and GBP/USD both max-allocated despite ~0.8 correlation.
**After:** `x^T Σ x` using the sample covariance matrix. Correlated pairs are penalized together.

### Fix 5: Renamed from "BSTS" to Bayesian Linear Regression
A true BSTS has time-varying trend and seasonality. This model is a static Bayesian regression. The name now reflects reality.

### Fix 6: Regularized Normal Prior (Replaced Horseshoe)
With 2-3 predictors, a horseshoe prior is overkill and causes convergence issues. Replaced with `Normal(0, 0.05)` which provides light regularization and converges reliably.

### Fix 7: MCMC Samples Increased
**Before:** 300 draws / 300 tune — chains almost certainly haven't converged.
**After:** 1000 draws / 1000 tune with `target_accept=0.90`. This is the minimum for reliable posterior estimates.

### Fix 8: Pair-Specific Macro Features
**Before:** Same US-only features (VIX, 10Y) for all 5 pairs.
**After:** USD/CAD gets crude oil (CL=F), AUD/USD gets gold (GC=F). Extensible to ECB/BOJ/RBA rate proxies when data sources are added.

### Fix 9: Correct Label
**Before:** "Expected 5-Day Drift" — no 5-day aggregation exists in the code.
**After:** "daily_expected_return" — matches what the model actually estimates.

### Fix 10: No Stale Cache
**Before:** `@st.cache_resource` persisted MCMC results across sessions indefinitely.
**After:** Every pipeline run performs fresh inference on fresh data. No caching of stochastic model outputs.

### Fix 11: Risk Guards Added
**Before:** No per-trade risk limit, no stop-loss, no drawdown protection.
**After:**
- Max drawdown guard: pipeline refuses to trade if NAV drops below 95% of starting capital
- Per-trade risk percentage: 2% of capital per trade (configurable)
- Gross exposure cap: total allocations ≤ capital

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/pipeline/run` | Run full pipeline, return trade plan |
| POST | `/orders/execute` | Execute trades from a previous plan |

## Future Improvements
- Add ECB, BOJ, RBA rate proxies as macro features
- Implement proper stop-loss orders via OANDA's trailing stop API
- Add convergence diagnostics (R-hat, ESS) and reject runs that don't converge
- Add a local linear trend (GaussianRandomWalk) to make this a proper BSTS model
- Implement walk-forward backtesting before live deployment
- Add Prometheus metrics for monitoring
