"""Tests for risk management module."""

import pytest

from fx_engine.optimization.portfolio import Allocation
from fx_engine.risk import (
    RiskCheckedTrade,
    calculate_units,
    compute_trailing_stop,
    run_risk_checks,
)


class TestUnitSizing:

    def test_usd_base_pair(self):
        """USD/CAD: USD is base, units ≈ allocation."""
        units = calculate_units(500.0, "USD_CAD", 1.3600)
        assert units == 500

    def test_usd_quote_pair(self):
        """EUR/USD: USD is quote, units = allocation / price."""
        units = calculate_units(500.0, "EUR_USD", 1.0800)
        expected = int(round(500.0 / 1.0800))
        assert units == expected

    def test_short_direction(self):
        """Negative allocation → negative units."""
        units = calculate_units(-500.0, "EUR_USD", 1.0800)
        assert units < 0

    def test_zero_price_returns_zero(self):
        assert calculate_units(500.0, "EUR_USD", 0) == 0

    def test_zero_allocation_returns_zero(self):
        assert calculate_units(0.0, "EUR_USD", 1.08) == 0

    def test_symmetry(self):
        """Long and short of same size should produce opposite units."""
        long_units = calculate_units(500.0, "EUR_USD", 1.08)
        short_units = calculate_units(-500.0, "EUR_USD", 1.08)
        assert long_units == -short_units


class TestTrailingStop:

    def test_basic_computation(self):
        stop = compute_trailing_stop(0.0080, 2.0)
        assert abs(stop - 0.0160) < 1e-10

    def test_zero_atr(self):
        assert compute_trailing_stop(0.0, 2.0) == 0.0

    def test_higher_multiple_wider_stop(self):
        s1 = compute_trailing_stop(0.008, 1.5)
        s2 = compute_trailing_stop(0.008, 3.0)
        assert s2 > s1


class TestRiskChecks:

    def test_drawdown_breaker_blocks_all(self, trading_settings, drawdown_account):
        allocations = [
            Allocation("EUR_USD", 500.0, "LONG", 0.001, 0.0001, True),
        ]
        report = run_risk_checks(
            allocations=allocations,
            atr_by_pair={"EUR_USD": 0.008},
            live_prices={"EUR_USD": 1.08},
            account_summary=drawdown_account,
            settings=trading_settings,
        )
        assert report.drawdown_blocked is True
        assert len(report.trades) == 0

    def test_healthy_account_passes(self, trading_settings, healthy_account):
        allocations = [
            Allocation("EUR_USD", 500.0, "LONG", 0.001, 0.0001, True),
        ]
        report = run_risk_checks(
            allocations=allocations,
            atr_by_pair={"EUR_USD": 0.008},
            live_prices={"EUR_USD": 1.08},
            account_summary=healthy_account,
            settings=trading_settings,
        )
        assert report.drawdown_blocked is False
        assert len(report.trades) == 1

    def test_unconverged_pair_rejected(self, trading_settings):
        allocations = [
            Allocation("EUR_USD", 500.0, "LONG", 0.001, 0.0001, converged=False),
        ]
        report = run_risk_checks(
            allocations=allocations,
            atr_by_pair={"EUR_USD": 0.008},
            live_prices={"EUR_USD": 1.08},
            account_summary=None,
            settings=trading_settings,
        )
        assert len(report.trades) == 0
        assert any(r["reason"] == "mcmc_unconverged" for r in report.rejected)

    def test_tiny_allocation_rejected(self, trading_settings):
        allocations = [
            Allocation("EUR_USD", 5.0, "LONG", 0.001, 0.0001, True),  # below $10 min
        ]
        report = run_risk_checks(
            allocations=allocations,
            atr_by_pair={"EUR_USD": 0.008},
            live_prices={"EUR_USD": 1.08},
            account_summary=None,
            settings=trading_settings,
        )
        assert len(report.trades) == 0
        assert any(r["reason"] == "below_minimum" for r in report.rejected)

    def test_no_price_rejected(self, trading_settings):
        allocations = [
            Allocation("EUR_USD", 500.0, "LONG", 0.001, 0.0001, True),
        ]
        report = run_risk_checks(
            allocations=allocations,
            atr_by_pair={"EUR_USD": 0.008},
            live_prices={},  # no prices
            account_summary=None,
            settings=trading_settings,
        )
        assert len(report.trades) == 0
        assert any(r["reason"] == "no_price" for r in report.rejected)

    def test_risk_scaling_enforced(self, trading_settings):
        """Large allocation should be scaled down to fit risk budget."""
        allocations = [
            Allocation("EUR_USD", 900.0, "LONG", 0.001, 0.0001, True),
        ]
        report = run_risk_checks(
            allocations=allocations,
            atr_by_pair={"EUR_USD": 0.008},
            live_prices={"EUR_USD": 1.08},
            account_summary=None,
            settings=trading_settings,
        )
        if report.trades:
            max_risk = trading_settings.total_capital * trading_settings.per_trade_risk_pct
            assert report.trades[0].risk_usd <= max_risk + 0.01

    def test_no_account_summary_still_works(self, trading_settings):
        """Pipeline should work without OANDA account access."""
        allocations = [
            Allocation("EUR_USD", 500.0, "LONG", 0.001, 0.0001, True),
        ]
        report = run_risk_checks(
            allocations=allocations,
            atr_by_pair={"EUR_USD": 0.008},
            live_prices={"EUR_USD": 1.08},
            account_summary=None,
            settings=trading_settings,
        )
        assert report.drawdown_blocked is False
        assert len(report.trades) == 1

    def test_atr_fallback_when_no_atr(self, trading_settings):
        """When ATR is missing, should use 1% fallback stop."""
        allocations = [
            Allocation("EUR_USD", 500.0, "LONG", 0.001, 0.0001, True),
        ]
        report = run_risk_checks(
            allocations=allocations,
            atr_by_pair={},
            live_prices={"EUR_USD": 1.08},
            account_summary=None,
            settings=trading_settings,
        )
        assert len(report.trades) == 1
        # Fallback stop = 1% of price
        assert abs(report.trades[0].trailing_stop_distance - 0.0108) < 0.001
