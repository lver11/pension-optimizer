import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sustainable.config import (
    DurableAsset, DURABLE_ASSETS, DURABLE_ASSET_ORDER,
    get_durable_returns, get_durable_volatilities,
    get_durable_correlation_matrix, get_durable_covariance_matrix,
    compute_composite_score, get_score_matrix,
)


def test_asset_count():
    assert len(DURABLE_ASSETS) == 15
    assert len(DURABLE_ASSET_ORDER) == 15


def test_asset_has_required_fields():
    asset = DURABLE_ASSETS["obligations_univers"]
    assert 0 < asset.rendement < 0.20
    assert 0 < asset.volatilite < 0.40
    assert 1 <= asset.score_durabilite <= 5
    assert asset.has_durable_variant is True
    assert asset.nom_durable != ""


def test_durable_variant_has_higher_durability_score():
    asset = DURABLE_ASSETS["obligations_univers"]
    assert asset.score_durabilite_durable >= asset.score_durabilite


def test_get_durable_returns_shape():
    r = get_durable_returns()
    assert r.shape == (15,)
    assert np.all(r > 0)


def test_get_durable_volatilities_shape():
    v = get_durable_volatilities()
    assert v.shape == (15,)
    assert np.all(v > 0)


def test_get_durable_correlation_matrix_valid():
    corr = get_durable_correlation_matrix()
    assert corr.shape == (15, 15)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(corr)
    assert np.min(eigvals) >= -1e-8, "Correlation matrix must be PSD"


def test_get_durable_covariance_matrix_consistent():
    cov = get_durable_covariance_matrix()
    vols = get_durable_volatilities()
    assert np.allclose(np.diag(cov), vols**2, atol=1e-10)
    assert cov.shape == (15, 15)


def test_compute_composite_score_equal_weights():
    weights_port = np.ones(15) / 15
    dim_weights = {"durabilite": 0.2, "additionnalite": 0.2,
                   "disponibilite": 0.2, "retombees_qc": 0.2, "liquidite": 0.2}
    score = compute_composite_score(weights_port, dim_weights, use_durable={})
    # Scores range 1–5 (source: Excel cartography, liquidité dimension uses 1–5)
    assert 1.0 <= score <= 5.0


def test_compute_composite_score_durable_variant_higher():
    weights_port = np.zeros(15)
    idx = DURABLE_ASSET_ORDER.index("obligations_univers")
    weights_port[idx] = 1.0
    dim_weights = {"durabilite": 1.0, "additionnalite": 0.0,
                   "disponibilite": 0.0, "retombees_qc": 0.0, "liquidite": 0.0}
    score_std = compute_composite_score(weights_port, dim_weights, use_durable={})
    score_dur = compute_composite_score(weights_port, dim_weights,
                                        use_durable={"obligations_univers": True})
    assert score_dur > score_std


def test_get_score_matrix_shape():
    dim_weights = {"durabilite": 0.4, "additionnalite": 0.3,
                   "disponibilite": 0.1, "retombees_qc": 0.1, "liquidite": 0.1}
    S_std, S_dur = get_score_matrix(dim_weights)
    assert S_std.shape == (15,)
    assert S_dur.shape == (15,)
    # Durable scores should be >= standard for assets with durable variant
    for i, aid in enumerate(DURABLE_ASSET_ORDER):
        if DURABLE_ASSETS[aid].has_durable_variant:
            assert S_dur[i] >= S_std[i]
