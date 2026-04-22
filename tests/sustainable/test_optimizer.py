# tests/sustainable/test_optimizer.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sustainable.config import (
    DURABLE_ASSET_ORDER, DURABLE_ASSET_NAMES_FR, DURABLE_DEFAULT_WEIGHTS,
    DURABLE_MIN_WEIGHTS, DURABLE_MAX_WEIGHTS,
    get_durable_returns, get_durable_covariance_matrix,
    get_score_matrix, DEFAULT_DIM_WEIGHTS,
)
from sustainable.optimizer import DurableOptimizer, DurableResult


@pytest.fixture
def optimizer():
    mu = get_durable_returns()
    cov = get_durable_covariance_matrix()
    return DurableOptimizer(
        expected_returns=mu,
        cov_matrix=cov,
        risk_free_rate=0.03,
        asset_names=DURABLE_ASSET_NAMES_FR,
        min_weights=DURABLE_MIN_WEIGHTS,
        max_weights=DURABLE_MAX_WEIGHTS,
    )


@pytest.fixture
def sustainability_scores():
    S_cur, _ = get_score_matrix(DEFAULT_DIM_WEIGHTS)
    return S_cur


def test_optimize_durable_lambda_zero_feasible(optimizer, sustainability_scores):
    result = optimizer.optimize_durable(lam=0.0, gamma=2.5,
                                        sustainability_scores=sustainability_scores)
    assert result.status == "optimal"
    assert abs(result.weights.sum() - 1.0) < 1e-4
    assert np.all(result.weights >= -1e-6)


def test_optimize_durable_weights_sum_to_one(optimizer, sustainability_scores):
    result = optimizer.optimize_durable(lam=0.5, gamma=2.5,
                                        sustainability_scores=sustainability_scores)
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_optimize_durable_higher_lambda_higher_sustainability(optimizer, sustainability_scores):
    r0 = optimizer.optimize_durable(lam=0.0, gamma=2.5,
                                    sustainability_scores=sustainability_scores)
    r1 = optimizer.optimize_durable(lam=2.0, gamma=2.5,
                                    sustainability_scores=sustainability_scores)
    assert r1.sustainability_score >= r0.sustainability_score - 1e-4


def test_durable_result_has_breakdown(optimizer, sustainability_scores):
    result = optimizer.optimize_durable(lam=1.0, gamma=2.5,
                                        sustainability_scores=sustainability_scores)
    assert isinstance(result, DurableResult)
    assert isinstance(result.sustainability_score, float)
    assert 1.0 <= result.sustainability_score <= 5.0
    assert result.lambda_used == 1.0


def test_pareto_frontier_length(optimizer, sustainability_scores):
    frontier = optimizer.pareto_frontier(
        n_points=10, gamma=2.5,
        sustainability_scores=sustainability_scores,
    )
    assert len(frontier) >= 5


def test_pareto_frontier_monotone_sustainability(optimizer, sustainability_scores):
    frontier = optimizer.pareto_frontier(
        n_points=15, gamma=2.5,
        sustainability_scores=sustainability_scores,
    )
    scores = [r.sustainability_score for r in frontier]
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i-1] - 0.01


def test_optimize_durable_respects_max_weights(optimizer, sustainability_scores):
    result = optimizer.optimize_durable(lam=1.0, gamma=2.5,
                                        sustainability_scores=sustainability_scores)
    assert np.all(result.weights <= DURABLE_MAX_WEIGHTS + 1e-4)
    assert np.all(result.weights >= DURABLE_MIN_WEIGHTS - 1e-4)
