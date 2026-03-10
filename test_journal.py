"""Tests for trade journal module."""

import json
import tempfile
from pathlib import Path

import pytest

from fx_engine.journal import TradeJournal


@pytest.fixture
def journal(tmp_path) -> TradeJournal:
    return TradeJournal(tmp_path / "test_journal.jsonl")


class TestTradeJournal:

    def test_record_creates_file(self, journal):
        journal.record("test_event", {"key": "value"})
        assert journal.path.exists()

    def test_record_appends_jsonl(self, journal):
        journal.record("event_1", {"a": 1})
        journal.record("event_2", {"b": 2})

        lines = journal.path.read_text().strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["event_type"] == "event_1"
        assert entry1["a"] == 1

        entry2 = json.loads(lines[1])
        assert entry2["event_type"] == "event_2"

    def test_record_includes_timestamp(self, journal):
        journal.record("test", {})
        entries = journal.read_recent()
        assert "timestamp" in entries[0]

    def test_read_recent_returns_last_n(self, journal):
        for i in range(20):
            journal.record("event", {"index": i})

        recent = journal.read_recent(n=5)
        assert len(recent) == 5
        assert recent[0]["index"] == 15  # last 5

    def test_read_recent_empty_file(self, journal):
        entries = journal.read_recent()
        assert entries == []

    def test_record_pipeline_run(self, journal):
        pipeline_output = {
            "pipeline_id": "abc12345",
            "status": "success",
            "duration_seconds": 120.5,
            "solver_used": "Ipopt",
            "gross_exposure": 3500.0,
            "inference_results": [
                {"pair": "EUR_USD", "expected_return": 0.001, "converged": True, "rhat": 1.01},
            ],
            "risk_report": {
                "trades": [{"pair": "EUR_USD"}],
                "total_risk_usd": 80.0,
            },
        }
        journal.record_pipeline_run(pipeline_output)
        entries = journal.read_recent()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "pipeline_run"
        assert entries[0]["pipeline_id"] == "abc12345"

    def test_record_trade_execution(self, journal):
        journal.record_trade_execution(
            pipeline_id="abc",
            pair="EUR_USD",
            direction="LONG",
            units=463,
            status="filled",
            fill_price=1.0812,
            order_id="12345",
        )
        entries = journal.read_recent()
        assert entries[0]["event_type"] == "trade_executed"
        assert entries[0]["units"] == 463

    def test_read_by_pipeline(self, journal):
        journal.record("event", {"pipeline_id": "aaa"})
        journal.record("event", {"pipeline_id": "bbb"})
        journal.record("event", {"pipeline_id": "aaa"})

        aaa = journal.read_by_pipeline("aaa")
        assert len(aaa) == 2

    def test_record_reconciliation(self, journal):
        journal.record_reconciliation(
            pipeline_id="recon1",
            open_positions=[{"instrument": "EUR_USD"}],
            stale_closed=[],
            account_nav=4100.0,
        )
        entries = journal.read_recent()
        assert entries[0]["event_type"] == "reconciliation"
        assert entries[0]["account_nav"] == 4100.0
