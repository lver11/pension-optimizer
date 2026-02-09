"""
Estimateurs de matrice de covariance.
Sample, Ledoit-Wolf, EWMA, et correction PSD.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.covariance import LedoitWolf


class CovarianceEstimator:
    """Methodes d'estimation de la matrice de covariance."""

    @staticmethod
    def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
        """Covariance echantillonnale standard: S = (1/(T-1)) * X^T * X"""
        return returns.cov().values

    @staticmethod
    def ledoit_wolf(returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Retrait de Ledoit-Wolf via sklearn.
        Retourne (matrice_retiree, coefficient_retrait).
        """
        lw = LedoitWolf().fit(returns.values)
        return lw.covariance_, lw.shrinkage_

    @staticmethod
    def exponential_weighted(
        returns: pd.DataFrame, halflife: int = 60
    ) -> np.ndarray:
        """
        Covariance a poids exponentiels (EWMA).
        lambda = 1 - 2/(halflife+1)
        Donne plus de poids aux observations recentes.
        """
        lam = 1 - 2 / (halflife + 1)
        T, n = returns.shape
        weights = np.array([lam ** (T - 1 - t) for t in range(T)])
        weights /= weights.sum()

        centered = returns.values - returns.values.mean(axis=0)
        cov = np.zeros((n, n))
        for t in range(T):
            cov += weights[t] * np.outer(centered[t], centered[t])

        return cov

    @staticmethod
    def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Trouve la matrice semi-definie positive la plus proche.
        Algorithme de Higham (2002) par projections alternees.
        """
        n = matrix.shape[0]
        Y = matrix.copy()
        delta_S = np.zeros_like(matrix)

        for _ in range(100):
            R = Y - delta_S
            # Projection sur les matrices PSD
            eigvals, eigvecs = np.linalg.eigh(R)
            eigvals = np.maximum(eigvals, epsilon)
            X = eigvecs @ np.diag(eigvals) @ eigvecs.T
            delta_S = X - R
            # Projection sur les matrices avec diagonale unitaire (pour correlation)
            Y = X.copy()
            np.fill_diagonal(Y, np.diag(matrix))

            if np.linalg.norm(X - Y) < 1e-10:
                break

        return (Y + Y.T) / 2

    @staticmethod
    def denoise_covariance(
        returns: pd.DataFrame, method: str = "marcenko_pastur"
    ) -> np.ndarray:
        """
        Debruitage par theorie des matrices aleatoires (Marcenko-Pastur).
        Supprime les valeurs propres de bruit sous le seuil theorique.
        """
        T, n = returns.shape
        q = T / n
        cov = returns.cov().values

        # Decomposition spectrale
        eigvals, eigvecs = np.linalg.eigh(cov)
        sigma2 = np.median(eigvals)

        # Seuil Marcenko-Pastur
        lambda_plus = sigma2 * (1 + 1 / np.sqrt(q)) ** 2

        # Remplacer les valeurs propres de bruit
        eigvals_clean = eigvals.copy()
        noise_mask = eigvals_clean < lambda_plus
        if noise_mask.any():
            eigvals_clean[noise_mask] = eigvals_clean[noise_mask].mean()

        return eigvecs @ np.diag(eigvals_clean) @ eigvecs.T

    @staticmethod
    def estimate(
        returns: pd.DataFrame, method: str = "ledoit_wolf", **kwargs
    ) -> np.ndarray:
        """Interface unifiee pour l'estimation de covariance."""
        if method == "sample" or method == "echantillon":
            return CovarianceEstimator.sample_covariance(returns)
        elif method == "ledoit_wolf" or method == "ledoit-wolf":
            cov, _ = CovarianceEstimator.ledoit_wolf(returns)
            return cov
        elif method == "ewma" or method == "poids_exponentiels":
            halflife = kwargs.get("halflife", 60)
            return CovarianceEstimator.exponential_weighted(returns, halflife)
        elif method == "denoise" or method == "marcenko_pastur":
            return CovarianceEstimator.denoise_covariance(returns)
        else:
            return CovarianceEstimator.sample_covariance(returns)
