"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fx_engine.broker.oanda import AccountSummary
from fx_engine.config import EngineConfig, InferenceSettings, TradingSettings
from fx_engine.inference.bayesian import ConvergenceDiagnostics, InferenceResult


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Generate realistic FX return series."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    data = {
        "EURUSD=X": np.random.normal(0.0001, 0.005, 200),
        "GBPUSD=X": np.random.normal(0.0002, 0.006, 200),
        "CAD=X": np.random.normal(-0.0001, 0.004, 200),
        "AUDUSD=X": np.random.normal(0.0001, 0.007, 200),
        "JPY=X": np.random.normal(-0.0002, 0.005, 200),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def trading_settings() -> TradingSettings:
    return TradingSettings(
        total_capital=4000.0,
        max_position_usd=1000.0,
        max_drawdown_pct=0.05,
        per_trade_risk_pct=0.02,
        risk_aversion=2.0,
        min_allocation_usd=10.0,
        stop_loss_atr_multiple=2.0,
    )


@pytest.fixture
def converged_diagnostics() -> ConvergenceDiagnostics:
    return ConvergenceDiagnostics(
        rhat_max=1.01,
        ess_bulk_min=500,
        ess_tail_min=400,
        divergence_count=0,
        converged=True,
    )


@pytest.fixture
def unconverged_diagnostics() -> ConvergenceDiagnostics:
    return ConvergenceDiagnostics(
        rhat_max=1.15,
        ess_bulk_min=50,
        ess_tail_min=30,
        divergence_count=5,
        converged=False,
    )


@pytest.fixture
def sample_inference_results(converged_diagnostics) -> list[InferenceResult]:
    return [
        InferenceResult("EUR_USD", 0.002, 0.00005, 0.002, [0.01, -0.005], ["^TNX", "^VIX"], converged_diagnostics, 200),
        InferenceResult("GBP_USD", -0.001, 0.00006, -0.001, [0.008, -0.003], ["^TNX", "^VIX"], converged_diagnostics, 200),
        InferenceResult("USD_CAD", 0.0005, 0.00004, 0.0005, [0.005], ["^TNX"], converged_diagnostics, 200),
    ]


@pytest.fixture
def healthy_account() -> AccountSummary:
    return AccountSummary(
        nav=4100.0,
        unrealized_pl=50.0,
        open_trade_count=2,
        margin_used=500.0,
        margin_available=3600.0,
    )


@pytest.fixture
def drawdown_account() -> AccountSummary:
    return AccountSummary(
        nav=3700.0,
        unrealized_pl=-300.0,
        open_trade_count=3,
        margin_used=1000.0,
        margin_available=2700.0,
    )
