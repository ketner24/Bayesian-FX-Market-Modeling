"""
Walk-forward backtesting engine.

Simulates the full pipeline historically using expanding or rolling windows:
  1. At each rebalance date, fit the model on past data only
  2. Optimize portfolio using only information available at that point
  3. Hold positions for `hold_days`, then rebalance
  4. Track PnL, Sharpe, max drawdown, win rate

This is the correct way to evaluate a trading strategy — in-sample
metrics are meaningless for live trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from fx_engine.config import EngineConfig
from fx_engine.inference.bayesian import run_bayesian_inference, InferenceResult
from fx_engine.optimization.portfolio import optimize_portfolio

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single backtested trade."""

    date: str
    pair: str
    direction: str
    allocation_usd: float
    entry_return: float        # actual return over hold period
    pnl_usd: float
    expected_return: float     # model's prediction


@dataclass
class BacktestResult:
    """Aggregate backtest performance metrics."""

    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float       # gross profit / gross loss
    total_trades: int
    avg_pnl_per_trade: float
    worst_trade_pnl: float
    best_trade_pnl: float
    calmar_ratio: float        # annualized return / max drawdown
    equity_curve: list[float]
    trade_log: list[BacktestTrade]
    rebalance_dates: list[str]


def run_walk_forward_backtest(
    fx_returns: pd.DataFrame,
    macro_by_pair: dict[str, pd.DataFrame],
    cfg: EngineConfig,
    min_train_days: int = 120,
    hold_days: int = 5,
    rebalance_every: int = 5,
    window_type: str = "expanding",
    rolling_window_days: int | None = None,
) -> BacktestResult:
    """
    Walk-forward backtest with expanding or rolling training window.

    Args:
        fx_returns: full history of FX log returns
        macro_by_pair: dict of pair → macro feature returns
        cfg: engine configuration
        min_train_days: minimum observations before first trade
        hold_days: days to hold each position
        rebalance_every: rebalance frequency in trading days
        window_type: "expanding" (all past data) or "rolling" (fixed window)
        rolling_window_days: window size for rolling mode (required if window_type="rolling")
    """
    if window_type == "rolling" and rolling_window_days is None:
        rolling_window_days = min_train_days
        logger.info(f"Rolling window defaulting to min_train_days={min_train_days}")

    universe = cfg.universe
    settings_inf = cfg.inference
    settings_trd = cfg.trading

    # Use fewer MCMC samples for backtest speed
    import copy
    bt_inference = copy.deepcopy(settings_inf)
    bt_inference.mcmc_draws = max(500, settings_inf.mcmc_draws // 2)
    bt_inference.mcmc_tune = max(500, settings_inf.mcmc_tune // 2)
    bt_inference.mcmc_chains = 2  # speed vs accuracy tradeoff

    dates = fx_returns.index
    n_dates = len(dates)
    pairs = list(universe.FX_MAP.values())
    yf_to_oanda = universe.FX_MAP

    # Rebalance dates
    rebalance_indices = list(range(min_train_days, n_dates - hold_days, rebalance_every))
    logger.info(
        f"Backtest: {len(rebalance_indices)} rebalance points, "
        f"train_min={min_train_days}, hold={hold_days}d, "
        f"window={window_type}" + (f"({rolling_window_days}d)" if window_type == "rolling" else "")
    )

    trade_log: list[BacktestTrade] = []
    equity = [settings_trd.total_capital]
    rebalance_dates: list[str] = []

    for step, t in enumerate(rebalance_indices):
        rebal_date = str(dates[t].date())
        rebalance_dates.append(rebal_date)

        if step % 10 == 0:
            logger.info(
                f"Backtest step {step}/{len(rebalance_indices)} — "
                f"date={rebal_date}, equity={equity[-1]:.2f}"
            )

        # ── Train on data up to (not including) t ──
        if window_type == "rolling" and rolling_window_days is not None:
            window_start = max(0, t - rolling_window_days)
            train_returns = fx_returns.iloc[window_start:t]
        else:
            train_returns = fx_returns.iloc[:t]

        # ── Inference per pair ──
        inference_results: list[InferenceResult] = []
        for yf_ticker, oanda_pair in yf_to_oanda.items():
            try:
                y_train = train_returns[yf_ticker]
                X_macro = macro_by_pair.get(oanda_pair, pd.DataFrame(index=y_train.index))
                common = y_train.index.intersection(X_macro.index)

                if len(common) < 60:
                    continue

                result = run_bayesian_inference(
                    pair=oanda_pair,
                    y_returns=y_train.loc[common],
                    X_features=X_macro.loc[common],
                    settings=bt_inference,
                )
                inference_results.append(result)
            except Exception as e:
                logger.warning(f"Backtest inference failed for {oanda_pair} at {rebal_date}: {e}")
                continue

        if not inference_results:
            equity.append(equity[-1])
            continue

        # ── Covariance from training data ──
        converged_pairs = [r.pair for r in inference_results if r.diagnostics.converged]
        if not converged_pairs:
            equity.append(equity[-1])
            continue

        yf_tickers_for_cov = []
        for p in [r.pair for r in inference_results]:
            for yf_t, op in yf_to_oanda.items():
                if op == p:
                    yf_tickers_for_cov.append(yf_t)
                    break

        cov_matrix = train_returns[yf_tickers_for_cov].cov().values

        # ── Optimize ──
        portfolio = optimize_portfolio(inference_results, cov_matrix, settings_trd)

        # ── Simulate hold period returns ──
        period_pnl = 0.0
        hold_end = min(t + hold_days, n_dates)
        actual_returns = fx_returns.iloc[t:hold_end]

        for alloc in portfolio.allocations:
            if abs(alloc.usd_amount) < settings_trd.min_allocation_usd:
                continue

            # Find the Yahoo ticker for this pair
            yf_tick = None
            for yt, op in yf_to_oanda.items():
                if op == alloc.pair:
                    yf_tick = yt
                    break
            if yf_tick is None or yf_tick not in actual_returns.columns:
                continue

            # Cumulative return over hold period
            cum_return = actual_returns[yf_tick].sum()  # log returns are additive
            # PnL: allocation * return (sign handles direction)
            # If SHORT (negative allocation), negative return = positive PnL
            trade_pnl = alloc.usd_amount * cum_return

            period_pnl += trade_pnl
            trade_log.append(BacktestTrade(
                date=rebal_date,
                pair=alloc.pair,
                direction=alloc.direction,
                allocation_usd=alloc.usd_amount,
                entry_return=cum_return,
                pnl_usd=round(trade_pnl, 4),
                expected_return=alloc.expected_return,
            ))

        equity.append(equity[-1] + period_pnl)

    # ── Compute performance metrics ──
    equity_arr = np.array(equity)
    returns_series = np.diff(equity_arr) / equity_arr[:-1]
    returns_series = returns_series[np.isfinite(returns_series)]

    total_return = (equity_arr[-1] / equity_arr[0]) - 1

    # Annualize (approximate: rebalance_every trading days per period)
    n_periods = len(rebalance_indices)
    periods_per_year = 252 / rebalance_every
    annualized = (1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1

    # Sharpe
    if len(returns_series) > 1 and np.std(returns_series) > 0:
        sharpe = (np.mean(returns_series) / np.std(returns_series)) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - peak) / peak
    max_dd = float(np.min(drawdowns))

    # Win rate and profit factor
    pnls = [t.pnl_usd for t in trade_log]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / max(len(pnls), 1)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss

    calmar = abs(annualized / max_dd) if max_dd != 0 else 0.0

    result = BacktestResult(
        total_return_pct=round(total_return * 100, 2),
        annualized_return_pct=round(annualized * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown_pct=round(max_dd * 100, 2),
        win_rate=round(win_rate, 3),
        profit_factor=round(profit_factor, 3),
        total_trades=len(trade_log),
        avg_pnl_per_trade=round(np.mean(pnls), 4) if pnls else 0,
        worst_trade_pnl=round(min(pnls), 4) if pnls else 0,
        best_trade_pnl=round(max(pnls), 4) if pnls else 0,
        calmar_ratio=round(calmar, 3),
        equity_curve=[round(e, 2) for e in equity],
        trade_log=trade_log,
        rebalance_dates=rebalance_dates,
    )

    logger.info(
        f"Backtest complete: return={result.total_return_pct}%, "
        f"sharpe={result.sharpe_ratio}, max_dd={result.max_drawdown_pct}%, "
        f"win_rate={result.win_rate}, trades={result.total_trades}"
    )

    return result
