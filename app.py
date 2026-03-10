"""
FastAPI application — HTTP interface to the FX trading engine.

v4.0 additions:
  - API key authentication middleware
  - Rate limiting middleware
  - Trade journal (persistent JSONL audit log)
  - Position reconciliation before execution
  - Rolling window backtesting
  - Journal query endpoints
  - Reconciliation endpoint
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fx_engine import __version__
from fx_engine.backtesting import run_walk_forward_backtest
from fx_engine.broker.oanda import OandaBroker
from fx_engine.config import EngineConfig, OandaSettings, get_config
from fx_engine.data.pipeline import fetch_market_data
from fx_engine.journal import TradeJournal
from fx_engine.logging_config import setup_logging
from fx_engine.middleware import APIKeyMiddleware, RateLimitMiddleware
from fx_engine.monitoring import init_metrics
from fx_engine.pipeline import PipelineOutput, run_pipeline
from fx_engine.reconciliation import reconcile_positions

logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    setup_logging(level=cfg.log_level, json_output=True)
    logger.info(f"FX Engine v{__version__} starting")

    # Create directories
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.backtest_dir.mkdir(parents=True, exist_ok=True)

    # Initialize journal
    app.state.journal = TradeJournal(cfg.journal_path)
    logger.info(f"Trade journal: {cfg.journal_path}")

    # Start Prometheus metrics server
    if cfg.metrics.enabled:
        init_metrics(port=cfg.metrics.port, version=__version__)

    # Start scheduler if configured
    scheduler = None
    if cfg.scheduler.enabled:
        from fx_engine.scheduler import setup_scheduler
        scheduler = setup_scheduler(cfg)

    yield

    if scheduler:
        scheduler.shutdown(wait=False)
    logger.info("FX Engine shutting down")


# ── App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Bayesian FX Trading Engine",
    description=(
        "Quantitative FX pipeline: BSTS inference → "
        "covariance-aware optimization → risk management → OANDA execution. "
        "Authenticate with X-API-Key header (if configured)."
    ),
    version=__version__,
    lifespan=lifespan,
)

# Apply middleware (order matters — outermost first)
_cfg = get_config()
app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)
app.add_middleware(APIKeyMiddleware, api_key=_cfg.api_key)


# ── Request/Response Models ──────────────────────────────────────

class CredentialsInput(BaseModel):
    api_key: str = Field(description="OANDA API bearer token")
    account_id: str = Field(description="OANDA account ID")
    environment: str = Field(default="practice", pattern="^(practice|live)$")


class ExecuteInput(BaseModel):
    api_key: str = Field(description="OANDA API bearer token")
    account_id: str = Field(description="OANDA account ID")
    environment: str = Field(default="practice", pattern="^(practice|live)$")
    reconcile_first: bool = Field(
        default=True,
        description="Run position reconciliation before execution",
    )
    close_unknown_positions: bool = Field(
        default=False,
        description="Close positions not in our universe during reconciliation",
    )


class BacktestRequest(BaseModel):
    min_train_days: int = Field(default=120, ge=60)
    hold_days: int = Field(default=5, ge=1, le=20)
    rebalance_every: int = Field(default=5, ge=1, le=20)
    window_type: str = Field(
        default="expanding",
        pattern="^(expanding|rolling)$",
        description="'expanding' uses all past data; 'rolling' uses a fixed window",
    )
    rolling_window_days: int | None = Field(
        default=None,
        ge=60,
        description="Window size for rolling mode (required if window_type='rolling')",
    )


class ReconcileRequest(BaseModel):
    api_key: str
    account_id: str
    environment: str = Field(default="practice", pattern="^(practice|live)$")
    close_unknown: bool = Field(
        default=False,
        description="Close positions not in our trading universe",
    )


# ── Helper ───────────────────────────────────────────────────────

def _get_journal() -> TradeJournal:
    """Safely get the trade journal from app state."""
    journal = getattr(app.state, "journal", None)
    if journal is None:
        # Fallback: create a journal on the fly (should only happen in tests)
        cfg = get_config()
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        journal = TradeJournal(cfg.journal_path)
        app.state.journal = journal
    return journal


def _make_broker(creds: CredentialsInput | ExecuteInput | ReconcileRequest) -> OandaBroker:
    settings = OandaSettings(
        api_key=creds.api_key,
        account_id=creds.account_id,
        environment=creds.environment,
    )
    return OandaBroker(settings)


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/config")
def get_current_config():
    """Return current configuration with secrets redacted."""
    cfg = get_config()
    return {
        "version": __version__,
        "auth_enabled": bool(cfg.api_key),
        "trading": {
            "total_capital": cfg.trading.total_capital,
            "max_position_usd": cfg.trading.max_position_usd,
            "max_drawdown_pct": cfg.trading.max_drawdown_pct,
            "per_trade_risk_pct": cfg.trading.per_trade_risk_pct,
            "risk_aversion": cfg.trading.risk_aversion,
            "stop_loss_atr_multiple": cfg.trading.stop_loss_atr_multiple,
        },
        "inference": {
            "mcmc_draws": cfg.inference.mcmc_draws,
            "mcmc_tune": cfg.inference.mcmc_tune,
            "mcmc_chains": cfg.inference.mcmc_chains,
            "target_accept": cfg.inference.target_accept,
            "student_t_nu": cfg.inference.student_t_nu,
            "min_ess_per_chain": cfg.inference.min_ess_per_chain,
            "max_rhat": cfg.inference.max_rhat,
            "use_local_trend": cfg.inference.use_local_trend,
            "lookback_days": cfg.inference.lookback_days,
        },
        "universe": {
            "fx_pairs": list(cfg.universe.FX_MAP.values()),
            "macro_features": cfg.universe.MACRO_FEATURES,
        },
        "scheduler": {
            "enabled": cfg.scheduler.enabled,
            "run_time_utc": f"{cfg.scheduler.run_hour_utc:02d}:{cfg.scheduler.run_minute_utc:02d}",
            "auto_execute": cfg.scheduler.auto_execute,
        },
    }


@app.post("/pipeline/run")
def pipeline_run(creds: CredentialsInput) -> dict:
    """
    Run the full pipeline: data → inference → optimization → risk checks.
    Returns trade plan WITHOUT executing.
    """
    cfg = get_config()
    broker = _make_broker(creds)
    journal = _get_journal()

    result: PipelineOutput = run_pipeline(broker, cfg, auto_execute=False)
    journal.record_pipeline_run(result.__dict__)

    if result.status == "error":
        raise HTTPException(status_code=500, detail=result.error)

    return result.__dict__


@app.post("/pipeline/execute")
def pipeline_execute(req: ExecuteInput) -> dict:
    """
    Run the full pipeline AND execute approved trades.
    Optionally runs position reconciliation first.
    """
    cfg = get_config()
    broker = _make_broker(req)
    journal = _get_journal()

    # Pre-flight reconciliation
    if req.reconcile_first:
        recon = reconcile_positions(
            broker=broker,
            allowed_instruments=list(cfg.universe.FX_MAP.values()),
            close_unknown=req.close_unknown_positions,
        )
        journal.record_reconciliation(
            pipeline_id="pre_execute",
            open_positions=recon.open_positions,
            stale_closed=recon.stale_positions_closed,
            account_nav=recon.account_summary.nav if recon.account_summary else None,
        )
        if not recon.passed:
            logger.warning(f"Reconciliation warnings: {recon.warnings}")

    result: PipelineOutput = run_pipeline(broker, cfg, auto_execute=True)
    journal.record_pipeline_run(result.__dict__)

    # Log individual trade executions
    for exec_result in result.execution_results:
        journal.record_trade_execution(
            pipeline_id=result.pipeline_id,
            pair=exec_result.get("pair", ""),
            direction=exec_result.get("direction", ""),
            units=exec_result.get("units", 0),
            status=exec_result.get("status", ""),
            fill_price=exec_result.get("fill_price"),
            order_id=exec_result.get("order_id"),
            trailing_stop=exec_result.get("trailing_stop"),
            error=exec_result.get("error"),
        )

    if result.status == "error":
        raise HTTPException(status_code=500, detail=result.error)

    return result.__dict__


@app.post("/reconcile")
def reconcile(req: ReconcileRequest) -> dict:
    """
    Run position reconciliation without a full pipeline run.
    Useful for checking account state and cleaning up stale positions.
    """
    cfg = get_config()
    broker = _make_broker(req)
    journal = _get_journal()

    result = reconcile_positions(
        broker=broker,
        allowed_instruments=list(cfg.universe.FX_MAP.values()),
        close_unknown=req.close_unknown,
    )

    journal.record_reconciliation(
        pipeline_id="manual_reconcile",
        open_positions=result.open_positions,
        stale_closed=result.stale_positions_closed,
        account_nav=result.account_summary.nav if result.account_summary else None,
    )

    return {
        "passed": result.passed,
        "account_nav": result.account_summary.nav if result.account_summary else None,
        "margin_used": result.account_summary.margin_used if result.account_summary else None,
        "margin_available": result.account_summary.margin_available if result.account_summary else None,
        "open_positions": len(result.open_positions),
        "stale_closed": result.stale_positions_closed,
        "warnings": result.warnings,
    }


@app.post("/backtest/run")
def backtest_run(params: BacktestRequest) -> dict:
    """
    Run walk-forward backtest using historical data.
    Supports both expanding and rolling window modes.
    """
    cfg = get_config()

    if params.window_type == "rolling" and params.rolling_window_days is None:
        raise HTTPException(
            status_code=422,
            detail="rolling_window_days is required when window_type='rolling'",
        )

    try:
        data = fetch_market_data(cfg)
        result = run_walk_forward_backtest(
            fx_returns=data.fx_returns,
            macro_by_pair=data.macro_by_pair,
            cfg=cfg,
            min_train_days=params.min_train_days,
            hold_days=params.hold_days,
            rebalance_every=params.rebalance_every,
            window_type=params.window_type,
            rolling_window_days=params.rolling_window_days,
        )

        # Log backtest to journal
        journal = _get_journal()
        journal.record("backtest_completed", {
            "window_type": params.window_type,
            "rolling_window_days": params.rolling_window_days,
            "total_return_pct": result.total_return_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "total_trades": result.total_trades,
        })

        return {
            "status": "success",
            "window_type": params.window_type,
            "rolling_window_days": params.rolling_window_days,
            "total_return_pct": result.total_return_pct,
            "annualized_return_pct": result.annualized_return_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "calmar_ratio": result.calmar_ratio,
            "total_trades": result.total_trades,
            "avg_pnl_per_trade": result.avg_pnl_per_trade,
            "best_trade": result.best_trade_pnl,
            "worst_trade": result.worst_trade_pnl,
            "equity_curve": result.equity_curve,
            "rebalance_dates": result.rebalance_dates,
        }
    except Exception as e:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/positions/close-all")
def close_all_positions(creds: CredentialsInput) -> dict:
    """Emergency: close all open positions."""
    broker = _make_broker(creds)
    journal = _get_journal()

    results = broker.close_all_positions()
    for r in results:
        journal.record_trade_execution(
            pipeline_id="emergency_close",
            pair=r.instrument,
            direction="CLOSE",
            units=r.units,
            status=r.status,
            fill_price=r.fill_price,
            order_id=r.order_id,
            error=r.error,
        )

    return {
        "closed": [
            {"instrument": r.instrument, "units": r.units, "status": r.status}
            for r in results
        ]
    }


# ── Journal Query Endpoints ──────────────────────────────────────

@app.get("/journal/recent")
def journal_recent(n: int = 50) -> dict:
    """Query the last N trade journal entries."""
    journal = _get_journal()
    entries = journal.read_recent(n=min(n, 500))
    return {"count": len(entries), "entries": entries}


@app.get("/journal/pipeline/{pipeline_id}")
def journal_by_pipeline(pipeline_id: str) -> dict:
    """Query all journal entries for a specific pipeline run."""
    journal = _get_journal()
    entries = journal.read_by_pipeline(pipeline_id)
    return {"pipeline_id": pipeline_id, "count": len(entries), "entries": entries}
