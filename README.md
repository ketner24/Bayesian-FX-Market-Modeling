# Bayesian FX Trading Engine v4.0

A production-grade quantitative FX trading pipeline built on Bayesian Structural Time Series inference, covariance-aware portfolio optimization, and OANDA execution with full risk management, audit logging, and monitoring.

## Architecture

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│  Reconcile  │ ──► │  Data Layer   │ ──► │  Inference   │ ──► │  Optimizer  │
│ (positions) │     │  (yfinance)   │     │ (PyMC BSTS)  │     │ (Pyomo MVO) │
└─────────────┘     └───────────────┘     └──────────────┘     └─────────────┘
                                                                      │
       ┌──────────────────────────────────────────────────────────────┘
       ▼
┌──────────────┐     ┌───────────┐     ┌──────────────────┐     ┌──────────┐
│    Risk      │ ──► │ Execution │ ──► │  Trade Journal   │ ──► │ Grafana  │
│ Management   │     │  (OANDA)  │     │  (JSONL audit)   │     │  + Prom  │
└──────────────┘     └───────────┘     └──────────────────┘     └──────────┘
```

## Quick Start

```bash
# 1. Clone and install
pip install -e ".[dev,scheduling,monitoring]"

# 2. Configure
cp .env.example .env
# Edit .env with your OANDA credentials

# 3. Run
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Docker (recommended for production)

```bash
docker-compose up -d
# API:        http://localhost:8000
# Grafana:    http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9091
```

Grafana auto-provisions with the FX Engine dashboard — no manual setup required.

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Liveness check |
| GET | `/config` | Yes | Current configuration (secrets redacted) |
| POST | `/pipeline/run` | Yes | Run inference + optimization, return trade plan |
| POST | `/pipeline/execute` | Yes | Run pipeline AND execute trades |
| POST | `/reconcile` | Yes | Check account state, close stale positions |
| POST | `/backtest/run` | Yes | Walk-forward backtest (expanding or rolling) |
| POST | `/positions/close-all` | Yes | Emergency: close all open positions |
| GET | `/journal/recent?n=50` | Yes | Query last N trade journal entries |
| GET | `/journal/pipeline/{id}` | Yes | Query journal for specific pipeline run |

Authentication is via the `X-API-Key` header. Set `FX_ENGINE_API_KEY` in your `.env` to enable; leave blank for development.

### Example: Run Pipeline

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-engine-api-key" \
  -d '{
    "api_key": "your-oanda-key",
    "account_id": "your-account-id",
    "environment": "practice"
  }'
```

### Example: Rolling Window Backtest

```bash
curl -X POST http://localhost:8000/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "min_train_days": 120,
    "hold_days": 5,
    "rebalance_every": 5,
    "window_type": "rolling",
    "rolling_window_days": 180
  }'
```

## Project Structure

```
fx_engine/
├── __init__.py              # Package version
├── app.py                   # FastAPI application + all endpoints
├── config.py                # Pydantic-settings centralized configuration
├── logging_config.py        # JSON structured logging (ELK/Datadog ready)
├── middleware.py             # API key auth + rate limiting
├── journal.py               # Append-only JSONL trade audit log
├── pipeline.py              # Orchestrator: data → inference → optimize → risk → execute
├── reconciliation.py        # Pre-flight position reconciliation
├── scheduler.py             # APScheduler daily automation
├── py.typed                 # PEP 561 type stub marker
├── broker/
│   └── oanda.py             # OANDA v20 REST client with retry + trailing stops
├── data/
│   └── pipeline.py          # Market data fetch, validation, ATR computation
├── inference/
│   └── bayesian.py          # BSTS model: GaussianRandomWalk + convergence diagnostics
├── optimization/
│   └── portfolio.py         # Covariance-aware MVO with Pyomo
├── risk/
│   └── __init__.py          # Drawdown breaker, ATR stops, position sizing
├── backtesting/
│   └── __init__.py          # Walk-forward backtest (expanding + rolling windows)
├── monitoring/
│   └── __init__.py          # Prometheus metrics
tests/
├── conftest.py              # Shared fixtures
├── test_config.py           # Configuration + integration tests
├── test_data.py             # Data pipeline tests
├── test_journal.py          # Trade journal tests
├── test_middleware.py        # Auth + rate limiting tests
├── test_optimization.py     # Portfolio optimization tests
├── test_risk.py             # Risk management tests
config/
├── prometheus.yml           # Prometheus scrape config
├── grafana/
│   ├── fx-engine-dashboard.json    # Pre-built Grafana dashboard
│   └── provisioning/               # Auto-provisioning for datasources + dashboards
.github/
└── workflows/
    └── ci.yml               # GitHub Actions: lint → typecheck → test → docker build
```

## What Changed From v3

### 1. API Security
API key authentication via `X-API-Key` header with rate limiting (60 req/min). Configurable via `FX_ENGINE_API_KEY` — leave blank to disable for development.

### 2. Trade Journal (Audit Log)
Every pipeline run, trade execution, reconciliation, and backtest is recorded as a JSON line in `data/trade_journal.jsonl`. This provides a complete, immutable audit trail queryable via the `/journal/*` endpoints. Each entry includes timestamps, pipeline IDs for tracing, and full execution details.

### 3. Position Reconciliation
Before every execution run, the engine now checks the OANDA account state: verifies open positions are in our trading universe, checks margin utilization, and optionally closes unknown/stale positions. This prevents orphaned positions from failed previous runs.

### 4. Rolling Window Backtesting
The backtester now supports both expanding windows (all past data) and rolling fixed-width windows. Rolling windows are better for detecting regime changes and avoiding fitting to stale data. Configure via the `window_type` and `rolling_window_days` parameters.

### 5. Grafana Dashboard Template
A complete Grafana dashboard JSON ships with the repo, auto-provisioned via docker-compose. Panels cover engine health (NAV, exposure, risk), inference quality (R-hat, ESS, convergence rate), expected returns, trade activity, and solver usage.

### 6. CI/CD Pipeline
GitHub Actions workflow: ruff lint → mypy type-check → pytest with coverage → Docker build + smoke test. Runs on push to main/develop and on all PRs.

### 7. Comprehensive Test Suite
Tests split into focused modules: data pipeline, risk management, optimization, journal, middleware, and configuration. Coverage of edge cases: drawdown breaker, unconverged pairs, rate limiting, missing prices, ATR fallbacks, and more.

## Configuration

All settings configurable via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `FX_ENGINE_API_KEY` | *(blank)* | API authentication key (blank = auth disabled) |
| `INFERENCE_USE_LOCAL_TREND` | `true` | GaussianRandomWalk (true BSTS) vs static drift |
| `INFERENCE_MAX_RHAT` | `1.05` | Reject inference if R-hat exceeds this |
| `TRADING_STOP_LOSS_ATR_MULTIPLE` | `2.0` | Trailing stop = ATR × this multiple |
| `TRADING_MAX_DRAWDOWN_PCT` | `0.05` | Circuit breaker: block trades if NAV drops 5% |
| `SCHEDULER_AUTO_EXECUTE` | `false` | Whether scheduled runs auto-submit orders |

## Testing

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=fx_engine --cov-report=html

# Specific module
pytest tests/test_risk.py -v
```

## Remaining Future Work

- Source actual ECB/BOJ/RBA rate differential data (currently using equity index proxies)
- WebSocket streaming for real-time price monitoring
- Redis-backed rate limiting for multi-instance deployments
- Alertmanager integration for Prometheus alerts (convergence failures, drawdown events)
- Position reconciliation scheduling (independent of pipeline runs)
