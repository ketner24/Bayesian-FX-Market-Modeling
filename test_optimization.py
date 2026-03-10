"""Tests for portfolio optimization module."""

import numpy as np
import pytest

from fx_engine.inference.bayesian import ConvergenceDiagnostics, InferenceResult
from fx_engine.optimization.portfolio import optimize_portfolio


class TestOptimization:

    def test_positive_mu_gets_long(self, trading_settings, converged_diagnostics):
        results = [
            InferenceResult("EUR_USD", 0.002, 0.00005, 0.002, [], [], converged_diagnostics, 200),
            InferenceResult("GBP_USD", -0.001, 0.00006, -0.001, [], [], converged_diagnostics, 200),
        ]
        cov = np.array([[0.00005, 0.00003], [0.00003, 0.00006]])
        portfolio = optimize_portfolio(results, cov, trading_settings)

        eur = next(a for a in portfolio.allocations if a.pair == "EUR_USD")
        gbp = next(a for a in portfolio.allocations if a.pair == "GBP_USD")
        assert eur.usd_amount > 0
        assert gbp.usd_amount <= 0

    def test_correlation_reduces_allocation(self, trading_settings, converged_diagnostics):
        results = [
            InferenceResult("EUR_USD", 0.002, 0.00005, 0.002, [], [], converged_diagnostics, 200),
            InferenceResult("GBP_USD", 0.002, 0.00005, 0.002, [], [], converged_diagnostics, 200),
        ]

        cov_high = np.array([[0.00005, 0.000045], [0.000045, 0.00005]])
        cov_low = np.array([[0.00005, 0.000005], [0.000005, 0.00005]])

        port_high = optimize_portfolio(results, cov_high, trading_settings)
        port_low = optimize_portfolio(results, cov_low, trading_settings)

        gross_high = sum(abs(a.usd_amount) for a in port_high.allocations)
        gross_low = sum(abs(a.usd_amount) for a in port_low.allocations)
        assert gross_high <= gross_low + 1.0

    def test_unconverged_excluded(self, trading_settings, converged_diagnostics, unconverged_diagnostics):
        results = [
            InferenceResult("EUR_USD", 0.002, 0.00005, 0.002, [], [], converged_diagnostics, 200),
            InferenceResult("GBP_USD", 0.005, 0.00003, 0.005, [], [], unconverged_diagnostics, 200),
        ]
        cov = np.array([[0.00005, 0.00002], [0.00002, 0.00003]])
        portfolio = optimize_portfolio(results, cov, trading_settings)

        gbp = next(a for a in portfolio.allocations if a.pair == "GBP_USD")
        assert gbp.usd_amount == 0.0
        assert gbp.converged is False

    def test_all_unconverged_returns_empty(self, trading_settings, unconverged_diagnostics):
        results = [
            InferenceResult("EUR_USD", 0.002, 0.00005, 0.002, [], [], unconverged_diagnostics, 200),
        ]
        cov = np.array([[0.00005]])
        portfolio = optimize_portfolio(results, cov, trading_settings)
        assert portfolio.solver_used == "N/A"
        assert portfolio.gross_exposure == 0.0

    def test_respects_capital_constraint(self, trading_settings, converged_diagnostics):
        """Gross exposure should not exceed total capital."""
        results = [
            InferenceResult("EUR_USD", 0.005, 0.00003, 0.005, [], [], converged_diagnostics, 200),
            InferenceResult("GBP_USD", 0.004, 0.00003, 0.004, [], [], converged_diagnostics, 200),
            InferenceResult("USD_CAD", 0.003, 0.00004, 0.003, [], [], converged_diagnostics, 200),
        ]
        cov = np.eye(3) * 0.00005
        portfolio = optimize_portfolio(results, cov, trading_settings)
        assert portfolio.gross_exposure <= trading_settings.total_capital + 0.01

    def test_zero_mu_gets_no_allocation(self, trading_settings, converged_diagnostics):
        """A pair with zero expected return shouldn't get allocated."""
        results = [
            InferenceResult("EUR_USD", 0.0, 0.00005, 0.0, [], [], converged_diagnostics, 200),
        ]
        cov = np.array([[0.00005]])
        portfolio = optimize_portfolio(results, cov, trading_settings)
        eur = next(a for a in portfolio.allocations if a.pair == "EUR_USD")
        assert abs(eur.usd_amount) < 1.0  # effectively zero
