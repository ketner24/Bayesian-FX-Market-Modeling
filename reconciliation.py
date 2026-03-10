"""
Position reconciliation.

Runs before each pipeline execution to:
  1. Fetch all open positions from OANDA
  2. Identify stale positions (older than max_hold_days)
  3. Optionally close stale positions
  4. Verify account state is consistent
  5. Log everything to the trade journal

This is a critical risk management component — in production,
you never want orphaned positions from a failed previous run
sitting unmonitored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fx_engine.broker.oanda import AccountSummary, OandaBroker, OrderResult

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Output of position reconciliation."""

    account_summary: AccountSummary | None
    open_positions: list[dict]
    stale_positions_closed: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: bool = True


def reconcile_positions(
    broker: OandaBroker,
    allowed_instruments: list[str],
    max_open_positions: int = 10,
    close_unknown: bool = False,
) -> ReconciliationResult:
    """
    Pre-flight position reconciliation.

    Checks:
      1. Account is accessible and in a valid state
      2. No more than max_open_positions are open
      3. All open positions are in our known universe
      4. Optionally close positions not in our universe

    Args:
        broker: OANDA broker client
        allowed_instruments: list of instruments we trade (e.g. ["EUR_USD", ...])
        max_open_positions: max allowed open positions
        close_unknown: if True, close positions not in allowed_instruments

    Returns:
        ReconciliationResult with account state and any actions taken
    """
    warnings: list[str] = []
    stale_closed: list[dict] = []

    # 1. Fetch account summary
    try:
        account_summary = broker.get_account_summary()
    except Exception as e:
        logger.error(f"Reconciliation: cannot reach OANDA account — {e}")
        return ReconciliationResult(
            account_summary=None,
            open_positions=[],
            warnings=[f"Cannot reach OANDA: {e}"],
            passed=False,
        )

    # 2. Fetch open positions
    try:
        open_positions = broker.get_open_positions()
    except Exception as e:
        logger.error(f"Reconciliation: cannot fetch open positions — {e}")
        return ReconciliationResult(
            account_summary=account_summary,
            open_positions=[],
            warnings=[f"Cannot fetch positions: {e}"],
            passed=False,
        )

    # 3. Check position count
    if len(open_positions) > max_open_positions:
        msg = (
            f"Too many open positions: {len(open_positions)} > "
            f"max {max_open_positions}"
        )
        logger.warning(f"Reconciliation: {msg}")
        warnings.append(msg)

    # 4. Check for unknown instruments
    for pos in open_positions:
        instrument = pos.get("instrument", "UNKNOWN")
        if instrument not in allowed_instruments:
            msg = f"Unknown position: {instrument} not in trading universe"
            logger.warning(f"Reconciliation: {msg}")
            warnings.append(msg)

            if close_unknown:
                logger.info(f"Reconciliation: closing unknown position {instrument}")
                long_units = int(float(pos.get("long", {}).get("units", 0)))
                short_units = int(float(pos.get("short", {}).get("units", 0)))
                closed = []
                if long_units > 0:
                    result = broker.execute_market_order(instrument, -long_units)
                    closed.append({"side": "long", "units": -long_units, "status": result.status})
                if short_units < 0:
                    result = broker.execute_market_order(instrument, -short_units)
                    closed.append({"side": "short", "units": -short_units, "status": result.status})
                stale_closed.append({"instrument": instrument, "actions": closed})

    # 5. Check margin utilization
    if account_summary.margin_available < 0:
        msg = "Negative available margin — account may be in margin call"
        logger.error(f"Reconciliation: {msg}")
        warnings.append(msg)

    margin_pct = (
        account_summary.margin_used / account_summary.nav * 100
        if account_summary.nav > 0
        else 0
    )
    if margin_pct > 50:
        msg = f"High margin utilization: {margin_pct:.1f}%"
        logger.warning(f"Reconciliation: {msg}")
        warnings.append(msg)

    passed = len(warnings) == 0

    logger.info(
        f"Reconciliation: NAV={account_summary.nav:.2f}, "
        f"open_positions={len(open_positions)}, "
        f"margin_used={margin_pct:.1f}%, "
        f"warnings={len(warnings)}, passed={passed}"
    )

    return ReconciliationResult(
        account_summary=account_summary,
        open_positions=open_positions,
        stale_positions_closed=stale_closed,
        warnings=warnings,
        passed=passed,
    )
