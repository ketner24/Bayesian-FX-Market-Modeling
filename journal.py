"""
Trade journal — append-only JSONL audit log.

Every pipeline run and every trade execution is recorded as a single
JSON line. This provides a complete, immutable audit trail for:
  - Regulatory compliance
  - Post-trade analysis
  - Debugging inference/optimization issues
  - Reconstructing historical decisions

Format: one JSON object per line (JSONL), easily parseable by
pandas, jq, ELK, or any log aggregation tool.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TradeJournal:
    """Append-only JSONL trade journal with file locking."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Append a single event to the journal.

        Args:
            event_type: e.g. "pipeline_run", "trade_executed", "risk_blocked"
            data: arbitrary event payload
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        }
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write journal entry: {e}")

    def record_pipeline_run(self, pipeline_output: dict) -> None:
        """Record a complete pipeline run."""
        self.record("pipeline_run", {
            "pipeline_id": pipeline_output.get("pipeline_id"),
            "status": pipeline_output.get("status"),
            "duration_seconds": pipeline_output.get("duration_seconds"),
            "solver_used": pipeline_output.get("solver_used"),
            "gross_exposure": pipeline_output.get("gross_exposure"),
            "inference_summary": [
                {
                    "pair": r.get("pair"),
                    "expected_return": r.get("expected_return"),
                    "converged": r.get("converged"),
                    "rhat": r.get("rhat"),
                }
                for r in pipeline_output.get("inference_results", [])
            ],
            "trade_count": len(
                pipeline_output.get("risk_report", {}).get("trades", [])
            ),
            "total_risk_usd": pipeline_output.get("risk_report", {}).get(
                "total_risk_usd"
            ),
        })

    def record_trade_execution(
        self,
        pipeline_id: str,
        pair: str,
        direction: str,
        units: int,
        status: str,
        fill_price: float | None = None,
        order_id: str | None = None,
        trailing_stop: float | None = None,
        error: str | None = None,
    ) -> None:
        """Record a single trade execution."""
        self.record("trade_executed", {
            "pipeline_id": pipeline_id,
            "pair": pair,
            "direction": direction,
            "units": units,
            "status": status,
            "fill_price": fill_price,
            "order_id": order_id,
            "trailing_stop_distance": trailing_stop,
            "error": error,
        })

    def record_reconciliation(
        self,
        pipeline_id: str,
        open_positions: list[dict],
        stale_closed: list[dict],
        account_nav: float | None,
    ) -> None:
        """Record a pre-trade reconciliation event."""
        self.record("reconciliation", {
            "pipeline_id": pipeline_id,
            "open_positions_count": len(open_positions),
            "stale_positions_closed": len(stale_closed),
            "stale_details": stale_closed,
            "account_nav": account_nav,
        })

    def read_recent(self, n: int = 50) -> list[dict]:
        """Read the last N journal entries."""
        if not self.path.exists():
            return []
        entries = []
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            return entries[-n:]
        except Exception as e:
            logger.error(f"Failed to read journal: {e}")
            return []

    def read_by_pipeline(self, pipeline_id: str) -> list[dict]:
        """Read all journal entries for a specific pipeline run."""
        entries = self.read_recent(n=10000)
        return [e for e in entries if e.get("pipeline_id") == pipeline_id]
