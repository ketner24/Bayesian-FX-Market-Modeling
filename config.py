"""
Centralized configuration via pydantic-settings.
Loads from environment variables, .env file, or defaults.

Corporate pattern: all secrets come from env vars or a vault,
never hardcoded. Config is immutable after startup.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class OandaSettings(BaseSettings):
    """OANDA broker credentials and connection settings."""

    api_key: str = Field(default="", description="OANDA API bearer token")
    account_id: str = Field(default="", description="OANDA account ID")
    environment: str = Field(
        default="practice",
        description="'practice' or 'live'",
        pattern="^(practice|live)$",
    )

    @property
    def base_url(self) -> str:
        if self.environment == "practice":
            return "https://api-fxpractice.oanda.com/v3"
        return "https://api-fxtrade.oanda.com/v3"

    model_config = {"env_prefix": "OANDA_"}


class TradingSettings(BaseSettings):
    """Capital, risk limits, and position sizing."""

    total_capital: float = Field(default=4000.0, ge=100.0)
    max_position_usd: float = Field(default=1000.0, ge=50.0)
    max_drawdown_pct: float = Field(default=0.05, ge=0.01, le=0.50)
    per_trade_risk_pct: float = Field(default=0.02, ge=0.005, le=0.10)
    risk_aversion: float = Field(default=2.0, ge=0.1, le=10.0)
    min_allocation_usd: float = Field(default=10.0, ge=1.0)
    stop_loss_atr_multiple: float = Field(
        default=2.0, ge=0.5, le=5.0,
        description="Trailing stop distance as a multiple of 14-day ATR",
    )

    model_config = {"env_prefix": "TRADING_"}


class InferenceSettings(BaseSettings):
    """MCMC sampler and model configuration."""

    mcmc_draws: int = Field(default=1000, ge=500)
    mcmc_tune: int = Field(default=1000, ge=500)
    mcmc_chains: int = Field(default=4, ge=2)
    target_accept: float = Field(default=0.90, ge=0.65, le=0.99)
    student_t_nu: float = Field(default=5.0, ge=2.1, le=30.0)
    min_ess_per_chain: float = Field(
        default=100.0, ge=50.0,
        description="Minimum effective sample size per chain to accept results",
    )
    max_rhat: float = Field(
        default=1.05, ge=1.0, le=1.20,
        description="Maximum R-hat to accept convergence",
    )
    lookback_days: int = Field(default=500, ge=100)
    use_local_trend: bool = Field(
        default=True,
        description="Use GaussianRandomWalk local trend (true BSTS) vs static drift",
    )

    model_config = {"env_prefix": "INFERENCE_"}


class UniverseSettings(BaseSettings):
    """FX universe and macro feature mappings."""

    # Yahoo Finance ticker → OANDA instrument
    FX_MAP: dict[str, str] = {
        "EURUSD=X": "EUR_USD",
        "GBPUSD=X": "GBP_USD",
        "CAD=X": "USD_CAD",
        "AUDUSD=X": "AUD_USD",
        "JPY=X": "USD_JPY",
    }

    # Pair-specific macro features
    # Improvement: ECB/BOJ/RBA proxy tickers added
    MACRO_FEATURES: dict[str, list[str]] = {
        # EUR: ECB main refi rate proxy via Euro Stoxx bank index + DE 10Y bund
        "EUR_USD": ["^TNX", "^VIX", "^STOXX50E"],
        # GBP: UK gilt yield proxy via FTSE 100
        "GBP_USD": ["^TNX", "^VIX", "^FTSE"],
        # CAD: crude oil is the dominant driver
        "USD_CAD": ["^TNX", "^VIX", "CL=F", "BZ=F"],
        # AUD: gold + iron ore proxy (BHP as liquid proxy)
        "AUD_USD": ["^TNX", "^VIX", "GC=F", "BHP"],
        # JPY: Nikkei as BOJ policy proxy + gold as safe-haven
        "USD_JPY": ["^TNX", "^VIX", "^N225", "GC=F"],
    }

    model_config = {"env_prefix": "UNIVERSE_"}


class SchedulerSettings(BaseSettings):
    """APScheduler / cron configuration."""

    enabled: bool = Field(default=False)
    run_hour_utc: int = Field(default=14, ge=0, le=23)
    run_minute_utc: int = Field(default=0, ge=0, le=59)
    run_days: str = Field(default="mon-fri")
    auto_execute: bool = Field(
        default=False,
        description="If True, automatically execute trades after pipeline run",
    )

    model_config = {"env_prefix": "SCHEDULER_"}


class MetricsSettings(BaseSettings):
    """Prometheus / monitoring configuration."""

    enabled: bool = Field(default=True)
    port: int = Field(default=9090)

    model_config = {"env_prefix": "METRICS_"}


class EngineConfig(BaseSettings):
    """Root configuration aggregating all sub-configs."""

    oanda: OandaSettings = OandaSettings()
    trading: TradingSettings = TradingSettings()
    inference: InferenceSettings = InferenceSettings()
    universe: UniverseSettings = UniverseSettings()
    scheduler: SchedulerSettings = SchedulerSettings()
    metrics: MetricsSettings = MetricsSettings()
    log_level: str = Field(default="INFO")
    api_key: str = Field(
        default="",
        description="API key for endpoint authentication. Blank = auth disabled.",
    )
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory for backtest results, trade logs, diagnostics",
    )

    model_config = {"env_prefix": "FX_ENGINE_"}

    @property
    def journal_path(self) -> Path:
        return self.data_dir / "trade_journal.jsonl"

    @property
    def backtest_dir(self) -> Path:
        return self.data_dir / "backtests"

    model_config = {"env_prefix": "FX_ENGINE_"}


@lru_cache(maxsize=1)
def get_config() -> EngineConfig:
    """Singleton config — loaded once, immutable thereafter."""
    return EngineConfig()
