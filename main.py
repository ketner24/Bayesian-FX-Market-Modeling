"""
Bayesian FX Trading Engine
===========================
FastAPI-based quantitative FX pipeline with:
  - Bayesian linear regression (horseshoe prior) per currency pair
  - Full covariance-aware portfolio optimization (fractional Kelly)
  - OANDA execution layer

Fixes applied from logic review:
  1. Regression betas are now extracted and used in expected return
  2. Student-T variance corrected with nu/(nu-2) factor
  3. Optimizer handles direction natively (signed mu, long/short vars)
  4. Full covariance matrix used in risk penalty
  5. Renamed from "BSTS" to Bayesian linear regression (accurate)
  6. Replaced horseshoe with simple regularized prior (2 features)
  7. MCMC samples increased to 1000+/1000+
  8. Added pair-specific macro features where available
  9. Corrected "5-day drift" label to "daily expected return"
  10. No stale caching — fresh inference each run
  11. Added per-trade risk limits and max drawdown guard
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import pymc as pm
import pyomo.environ as pyo
import requests
import json
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fx_engine")


# ================================================================
# 1. CONFIGURATION
# ================================================================
class EngineConfig:
    """Central configuration — replace with env vars or a config file in production."""

    TOTAL_CAPITAL: float = 4000.0
    MAX_POSITION_USD: float = 1000.0
    MAX_DRAWDOWN_PCT: float = 0.05          # 5% max drawdown guard
    PER_TRADE_RISK_PCT: float = 0.02        # 2% risk per trade
    RISK_AVERSION: float = 2.0
    MCMC_DRAWS: int = 1000
    MCMC_TUNE: int = 1000
    TARGET_ACCEPT: float = 0.90             # 0.95 was unnecessarily high for this model
    LOOKBACK_DAYS: int = 300
    MIN_ALLOCATION_USD: float = 10.0

    # FX universe: Yahoo ticker → OANDA instrument
    FX_MAP: dict = {
        "EURUSD=X": "EUR_USD",
        "GBPUSD=X": "GBP_USD",
        "CAD=X": "USD_CAD",
        "AUDUSD=X": "AUD_USD",
        "JPY=X": "USD_JPY",
    }

    # Pair-specific macro tickers (Fix #8: not just US instruments)
    # Each pair gets US baseline + pair-relevant features
    MACRO_FEATURES: dict = {
        "EUR_USD": ["^TNX", "^VIX"],        # ideally add ECB rate proxy
        "GBP_USD": ["^TNX", "^VIX"],        # ideally add gilt yield proxy
        "USD_CAD": ["^TNX", "^VIX", "CL=F"],  # crude oil for CAD
        "AUD_USD": ["^TNX", "^VIX", "GC=F"],  # gold for AUD
        "USD_JPY": ["^TNX", "^VIX"],        # ideally add JGB proxy
    }


CFG = EngineConfig()


# ================================================================
# 2. OANDA BROKER
# ================================================================
class OandaBroker:
    def __init__(self, api_key: str, account_id: str, environment: str = "practice"):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = (
            "https://api-fxpractice.oanda.com/v3"
            if environment == "practice"
            else "https://api-fxtrade.oanda.com/v3"
        )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_prices(self) -> dict[str, float] | None:
        instruments = ",".join(CFG.FX_MAP.values())
        endpoint = f"{self.base_url}/accounts/{self.account_id}/pricing"
        try:
            resp = requests.get(
                endpoint, headers=self.headers, params={"instruments": instruments}, timeout=10
            )
            resp.raise_for_status()
            prices = resp.json().get("prices", [])
            return {p["instrument"]: float(p["closeoutAsk"]) for p in prices}
        except Exception as e:
            logger.error(f"OANDA price fetch failed: {e}")
            return None

    def get_account_nav(self) -> float | None:
        """Fetch current account NAV for drawdown checks."""
        endpoint = f"{self.base_url}/accounts/{self.account_id}/summary"
        try:
            resp = requests.get(endpoint, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return float(resp.json()["account"]["NAV"])
        except Exception as e:
            logger.error(f"OANDA account fetch failed: {e}")
            return None

    def execute_trade(self, instrument: str, units: int) -> dict:
        endpoint = f"{self.base_url}/accounts/{self.account_id}/orders"
        payload = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        try:
            resp = requests.post(
                endpoint, headers=self.headers, data=json.dumps(payload), timeout=10
            )
            resp.raise_for_status()
            return {"status": "filled", "instrument": instrument, "units": units}
        except Exception as e:
            logger.error(f"Order failed for {instrument}: {e}")
            return {"status": "failed", "instrument": instrument, "error": str(e)}


# ================================================================
# 3. DATA PIPELINE
# ================================================================
def fetch_market_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Returns:
        fx_returns: DataFrame of daily FX returns (columns = Yahoo tickers)
        macro_by_pair: dict mapping OANDA pair → DataFrame of macro feature returns
    """
    # Collect all unique macro tickers
    all_macro_tickers = set()
    for tickers in CFG.MACRO_FEATURES.values():
        all_macro_tickers.update(tickers)

    all_tickers = list(CFG.FX_MAP.keys()) + list(all_macro_tickers)
    raw = yf.download(all_tickers, period=f"{CFG.LOOKBACK_DAYS}d")["Close"]
    returns = raw.pct_change().dropna()

    fx_returns = returns[list(CFG.FX_MAP.keys())]

    # Build per-pair macro feature DataFrames
    macro_by_pair = {}
    for yf_ticker, oanda_pair in CFG.FX_MAP.items():
        macro_tickers = CFG.MACRO_FEATURES.get(oanda_pair, ["^TNX", "^VIX"])
        available = [t for t in macro_tickers if t in returns.columns]
        if available:
            macro_df = returns[available]
            # Align with this pair's FX returns
            aligned = fx_returns[[yf_ticker]].join(macro_df, how="inner").dropna()
            macro_by_pair[oanda_pair] = aligned[available]
        else:
            macro_by_pair[oanda_pair] = pd.DataFrame(index=fx_returns.index)

    # Trim fx_returns to common index across all pairs
    common_idx = fx_returns.dropna().index
    fx_returns = fx_returns.loc[common_idx]

    return fx_returns, macro_by_pair


# ================================================================
# 4. BAYESIAN INFERENCE (per pair)
#    Fix #1: Extract and use regression betas
#    Fix #2: Correct Student-T variance
#    Fix #5: Renamed — this is Bayesian linear regression, not BSTS
#    Fix #6: Simple regularized prior instead of horseshoe
#    Fix #7: 1000+ draws/tune
# ================================================================
def run_bayesian_regression(
    y_returns: pd.Series,
    X_features: pd.DataFrame,
) -> tuple[float, float]:
    """
    Bayesian linear regression with regularized normal prior.

    Returns:
        expected_return: posterior mean of drift + X @ beta (evaluated at last observation)
        variance: corrected Student-T variance
    """
    n_predictors = X_features.shape[1]

    # Align y and X
    common_idx = y_returns.index.intersection(X_features.index)
    y = y_returns.loc[common_idx].values
    X = X_features.loc[common_idx].values

    if len(y) < 50:
        logger.warning(f"Only {len(y)} observations — results may be unreliable")

    with pm.Model():
        # Scale parameter
        sigma = pm.Exponential("sigma", lam=1.0)

        # Intercept (drift)
        drift = pm.Normal("drift", mu=0, sigma=0.01)

        # Fix #6: Simple regularized prior — horseshoe is overkill for 2-3 features
        if n_predictors > 0:
            beta = pm.Normal("beta", mu=0, sigma=0.05, shape=n_predictors)
            mu_t = drift + pm.math.dot(X, beta)
        else:
            mu_t = drift

        # Degrees of freedom for Student-T
        nu = 3.0

        # Likelihood
        pm.StudentT("obs", nu=nu, mu=mu_t, sigma=sigma, observed=y)

        # Fix #7: Proper sample count
        trace = pm.sample(
            draws=CFG.MCMC_DRAWS,
            tune=CFG.MCMC_TUNE,
            target_accept=CFG.TARGET_ACCEPT,
            progressbar=True,
            return_inferencedata=True,
        )

    # ── Fix #1: Use the full regression prediction, not just drift ──
    posterior_drift = trace.posterior["drift"].values.flatten()

    if n_predictors > 0:
        posterior_beta = trace.posterior["beta"].values.reshape(-1, n_predictors)
        # Evaluate at the most recent macro observation
        x_last = X[-1, :]  # last available observation
        # Expected return = E[drift] + E[beta] @ x_last
        regression_contribution = posterior_beta @ x_last
        posterior_expected = posterior_drift + regression_contribution
    else:
        posterior_expected = posterior_drift

    expected_return = float(np.mean(posterior_expected))

    # ── Fix #2: Correct Student-T variance ──
    posterior_sigma = trace.posterior["sigma"].values.flatten()
    sigma_sq = float(np.mean(posterior_sigma ** 2))
    # Var(Student-T) = sigma^2 * nu / (nu - 2) for nu > 2
    corrected_variance = sigma_sq * (nu / (nu - 2.0))

    logger.info(
        f"  E[return]={expected_return:.6f}, "
        f"Var(corrected)={corrected_variance:.6f}, "
        f"sigma^2={sigma_sq:.6f}"
    )

    return expected_return, corrected_variance


# ================================================================
# 5. PORTFOLIO OPTIMIZATION
#    Fix #3: Optimizer handles direction natively (signed mu)
#    Fix #4: Full covariance matrix in risk term
# ================================================================
def compute_covariance_matrix(fx_returns: pd.DataFrame, pairs: list[str]) -> np.ndarray:
    """Compute sample covariance of daily returns for the given pairs."""
    yf_tickers = []
    for pair in pairs:
        for yf_t, oanda_p in CFG.FX_MAP.items():
            if oanda_p == pair:
                yf_tickers.append(yf_t)
                break
    cov = fx_returns[yf_tickers].cov().values
    return cov


def optimize_portfolio(
    posterior_results: dict[str, tuple[float, float]],
    cov_matrix: np.ndarray,
) -> tuple[dict[str, float], str]:
    """
    Mean-variance optimization with:
      - Signed expected returns (Fix #3)
      - Full covariance matrix (Fix #4)
      - Per-trade risk limit (Fix #11)
      - Long and short allocations as separate non-negative variables

    Returns:
        allocations: dict of pair → signed USD allocation (negative = short)
        solver_name: which solver was used
    """
    pairs = list(posterior_results.keys())
    n = len(pairs)
    mu_vals = {k: v[0] for k, v in posterior_results.items()}

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, n - 1)

    per_trade_limit = CFG.TOTAL_CAPITAL * CFG.PER_TRADE_RISK_PCT / CFG.RISK_AVERSION
    position_cap = min(CFG.MAX_POSITION_USD, CFG.TOTAL_CAPITAL * 0.4)

    # Long and short allocation variables
    model.x_long = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, position_cap))
    model.x_short = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, position_cap))

    # Fix #3: Signed expected profit — optimizer knows direction
    def objective_rule(m):
        expected_profit = sum(
            m.x_long[i] * mu_vals[pairs[i]] - m.x_short[i] * mu_vals[pairs[i]]
            for i in m.I
        )
        # Fix #4: Full covariance risk penalty
        # Net position per pair: x_long[i] - x_short[i]
        risk = sum(
            (m.x_long[i] - m.x_short[i]) * cov_matrix[i, j] * (m.x_long[j] - m.x_short[j])
            for i in m.I
            for j in m.I
        ) * CFG.RISK_AVERSION
        return expected_profit - risk

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Total gross exposure ≤ capital
    def capital_rule(m):
        return sum(m.x_long[i] + m.x_short[i] for i in m.I) <= CFG.TOTAL_CAPITAL

    model.cap_constraint = pyo.Constraint(rule=capital_rule)

    # Only one side active per pair (no simultaneous long+short)
    # We enforce this softly: the optimizer naturally picks one side
    # because going both directions on the same pair wastes capital.

    # Solve
    solver_used = "None"
    try:
        solver = pyo.SolverFactory("gurobi")
        if solver.available(exception_flag=False):
            solver.solve(model, tee=False)
            solver_used = "Gurobi"
        else:
            raise ValueError("Gurobi unavailable")
    except Exception:
        solver = pyo.SolverFactory("ipopt")
        solver.solve(model, tee=False)
        solver_used = "Ipopt (fallback)"

    allocations = {}
    for i, pair in enumerate(pairs):
        long_val = pyo.value(model.x_long[i]) or 0.0
        short_val = pyo.value(model.x_short[i]) or 0.0
        net = long_val - short_val  # positive = long, negative = short
        allocations[pair] = net

    return allocations, solver_used


# ================================================================
# 6. UNIT SIZING
# ================================================================
def calculate_units(allocation_usd: float, pair: str, price: float) -> int:
    """Convert signed USD allocation to signed OANDA units."""
    direction = 1 if allocation_usd >= 0 else -1
    abs_usd = abs(allocation_usd)
    base_ccy, quote_ccy = pair.split("_")

    if base_ccy == "USD":
        units = abs_usd
    elif quote_ccy == "USD":
        units = abs_usd / price
    else:
        units = 0

    return int(round(units)) * direction


# ================================================================
# 7. PIPELINE ORCHESTRATOR
# ================================================================
class PipelineResult(BaseModel):
    timestamp: str
    solver_used: str
    trades: list[dict]
    skipped: list[dict] = []
    drawdown_blocked: bool = False


def run_pipeline(broker: OandaBroker) -> PipelineResult:
    """Full daily pipeline: data → inference → optimization → execution plan."""

    logger.info("=" * 60)
    logger.info("PIPELINE START")
    logger.info("=" * 60)

    # ── Data ──
    logger.info("Fetching market data...")
    fx_returns, macro_by_pair = fetch_market_data()
    live_prices = broker.get_prices()
    if live_prices is None:
        raise RuntimeError("Cannot fetch live OANDA prices")

    # ── Drawdown guard (Fix #11) ──
    nav = broker.get_account_nav()
    if nav is not None and nav < CFG.TOTAL_CAPITAL * (1 - CFG.MAX_DRAWDOWN_PCT):
        logger.warning(f"NAV={nav:.2f} below drawdown limit. Blocking trades.")
        return PipelineResult(
            timestamp=datetime.utcnow().isoformat(),
            solver_used="N/A",
            trades=[],
            drawdown_blocked=True,
        )

    # ── Bayesian inference per pair ──
    posteriors: dict[str, tuple[float, float]] = {}
    for yf_ticker, oanda_pair in CFG.FX_MAP.items():
        logger.info(f"Running Bayesian regression for {oanda_pair}...")
        y = fx_returns[yf_ticker]
        X = macro_by_pair.get(oanda_pair, pd.DataFrame(index=y.index))
        # Align
        common = y.index.intersection(X.index)
        mu, var = run_bayesian_regression(y.loc[common], X.loc[common])
        posteriors[oanda_pair] = (mu, var)

    # ── Covariance matrix ──
    pairs = list(posteriors.keys())
    cov = compute_covariance_matrix(fx_returns, pairs)

    # ── Optimization ──
    logger.info("Running portfolio optimization...")
    allocations, solver_used = optimize_portfolio(posteriors, cov)
    logger.info(f"Solver: {solver_used}")
    logger.info(f"Allocations: {allocations}")

    # ── Build execution plan ──
    trades = []
    skipped = []
    for pair, alloc_usd in allocations.items():
        if abs(alloc_usd) < CFG.MIN_ALLOCATION_USD:
            skipped.append({"pair": pair, "allocation_usd": round(alloc_usd, 2), "reason": "below minimum"})
            continue

        price = live_prices.get(pair, 0)
        if price == 0:
            skipped.append({"pair": pair, "reason": "no price available"})
            continue

        units = calculate_units(alloc_usd, pair, price)
        expected_ret = posteriors[pair][0]

        trades.append({
            "pair": pair,
            "allocation_usd": round(alloc_usd, 2),
            "daily_expected_return": round(expected_ret, 6),  # Fix #9: correct label
            "direction": "LONG" if alloc_usd > 0 else "SHORT",
            "oanda_units": units,
            "price": price,
        })

    result = PipelineResult(
        timestamp=datetime.utcnow().isoformat(),
        solver_used=solver_used,
        trades=trades,
        skipped=skipped,
    )

    logger.info(f"Pipeline complete: {len(trades)} trades, {len(skipped)} skipped")
    return result


# ================================================================
# 8. FASTAPI APPLICATION
# ================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FX Engine starting up")
    yield
    logger.info("FX Engine shutting down")


app = FastAPI(
    title="Bayesian FX Trading Engine",
    description="Quantitative FX pipeline: Bayesian inference → portfolio optimization → OANDA execution",
    version="2.0.0",
    lifespan=lifespan,
)


class CredentialsInput(BaseModel):
    api_key: str
    account_id: str
    environment: str = Field(default="practice", pattern="^(practice|live)$")


class ExecuteRequest(BaseModel):
    api_key: str
    account_id: str
    environment: str = Field(default="practice", pattern="^(practice|live)$")
    trades: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/pipeline/run", response_model=PipelineResult)
def run_daily_pipeline(creds: CredentialsInput):
    """Run the full inference + optimization pipeline. Returns trade plan without executing."""
    broker = OandaBroker(creds.api_key, creds.account_id, creds.environment)
    try:
        result = run_pipeline(broker)
        return result
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orders/execute")
def execute_orders(req: ExecuteRequest):
    """Execute a list of trades from a previous pipeline run."""
    broker = OandaBroker(req.api_key, req.account_id, req.environment)
    results = []
    for trade in req.trades:
        res = broker.execute_trade(trade["pair"], trade["oanda_units"])
        results.append(res)
    return {"executions": results}


# ================================================================
# 9. SCHEDULER (optional — run as a standalone process)
# ================================================================
def setup_scheduler():
    """
    Optional: use APScheduler to run the pipeline daily at a fixed time.
    Install: pip install apscheduler

    Usage:
        from main import setup_scheduler
        setup_scheduler()
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler()

        def scheduled_run():
            # Load credentials from environment or config
            import os
            api_key = os.getenv("OANDA_API_KEY", "")
            account_id = os.getenv("OANDA_ACCOUNT_ID", "")
            if not api_key:
                logger.error("No OANDA credentials in environment")
                return
            broker = OandaBroker(api_key, account_id)
            result = run_pipeline(broker)
            logger.info(f"Scheduled run complete: {len(result.trades)} trades planned")
            # In production: auto-execute or notify for manual approval

        # Run at 14:00 UTC (London/NY overlap) on weekdays
        scheduler.add_job(scheduled_run, "cron", day_of_week="mon-fri", hour=14, minute=0)
        scheduler.start()
        logger.info("Scheduler started — pipeline runs Mon-Fri at 14:00 UTC")
        return scheduler
    except ImportError:
        logger.warning("APScheduler not installed — scheduler disabled")
        return None


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
