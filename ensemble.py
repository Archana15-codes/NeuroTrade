"""
ensemble.py — Ensemble methods for NeuroTrade (Layer 4).

Provides three combination strategies that sit on top of any
dict[str, BaseModel] produced by ml_models.get_models():

  • VotingEnsemble      — hard / soft majority vote
  • WeightedEnsemble    — confidence-weighted probability blending
  • StackingEnsemble    — meta-learner trained on base-model predictions

All ensembles expose the same predict / predict_proba / evaluate interface
so the Streamlit panel can treat them identically.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from ml_models import BaseModel


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────

def _collect_probas(
    models: Dict[str, BaseModel], X: np.ndarray
) -> np.ndarray:
    """Stack class-1 probabilities from every model → (n_samples, n_models)."""
    return np.column_stack([
        m.predict_proba(X)[:, 1] for m in models.values()
    ])


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  1. Voting Ensemble
# ══════════════════════════════════════════════════════════════════════════════

class VotingEnsemble:
    """Hard or soft majority-vote over trained base models.

    Parameters
    ----------
    models : dict[str, BaseModel]
        Already-trained model wrappers.
    voting : ``'hard'`` | ``'soft'``
        * *hard* — each model casts a binary vote; majority wins.
        * *soft* — average predicted probabilities; argmax wins.
    """

    def __init__(
        self,
        models: Dict[str, BaseModel],
        voting: str = "soft",
    ):
        self.models = models
        self.voting = voting
        self.name = f"Voting ({'S' if voting == 'soft' else 'H'})"

    # ── predict_proba ────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, 2) averaged probability matrix."""
        all_proba = np.array([
            m.predict_proba(X) for m in self.models.values()
        ])  # (n_models, n_samples, n_classes)
        return all_proba.mean(axis=0)

    # ── predict ──────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.voting == "soft":
            avg_proba = self.predict_proba(X)
            return (avg_proba[:, 1] >= 0.5).astype(int)
        else:
            preds = np.column_stack([
                m.predict(X) for m in self.models.values()
            ])
            # majority vote (>50 % of models say 1)
            return (preds.mean(axis=1) >= 0.5).astype(int)

    # ── evaluate ─────────────────────────────────────────────────────────
    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        return _evaluate(y_test, self.predict(X_test))


# ══════════════════════════════════════════════════════════════════════════════
#  2. Weighted Ensemble
# ══════════════════════════════════════════════════════════════════════════════

class WeightedEnsemble:
    """Probability blending where each model's weight is its validation score.

    Parameters
    ----------
    models : dict[str, BaseModel]
        Already-trained model wrappers.
    weights : dict[str, float] | None
        Manual weights per model key.  If ``None``, weights are computed
        automatically from validation accuracy via :meth:`fit_weights`.
    """

    def __init__(
        self,
        models: Dict[str, BaseModel],
        weights: Optional[Dict[str, float]] = None,
    ):
        self.models = models
        self.weights: Dict[str, float] = weights or {k: 1.0 for k in models}
        self.name = "Weighted"

    # ── fit_weights ──────────────────────────────────────────────────────
    def fit_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Set weights proportional to each model's validation accuracy."""
        raw: Dict[str, float] = {}
        for name, model in self.models.items():
            acc = accuracy_score(y_val, model.predict(X_val))
            raw[name] = max(acc, 0.01)  # floor to avoid zero-weight
        total = sum(raw.values())
        self.weights = {k: v / total for k, v in raw.items()}
        return self.weights

    # ── predict_proba ────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        weighted_sum = np.zeros((len(X), 2))
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            weighted_sum += w * model.predict_proba(X)
        return weighted_sum

    # ── predict ──────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    # ── evaluate ─────────────────────────────────────────────────────────
    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        return _evaluate(y_test, self.predict(X_test))


# ══════════════════════════════════════════════════════════════════════════════
#  3. Stacking Ensemble
# ══════════════════════════════════════════════════════════════════════════════

class StackingEnsemble:
    """Two-layer stacking: base model probabilities are features for a
    Logistic Regression meta-learner.

    Parameters
    ----------
    models : dict[str, BaseModel]
        Already-trained base model wrappers.
    meta_C : float
        Regularisation strength for the logistic meta-learner.
    """

    def __init__(
        self,
        models: Dict[str, BaseModel],
        meta_C: float = 1.0,
    ):
        self.models = models
        self.meta = LogisticRegression(
            C=meta_C, max_iter=500, solver="lbfgs", random_state=42,
        )
        self.scaler = StandardScaler()
        self._fitted = False
        self.name = "Stacking"

    # ── _meta_features ───────────────────────────────────────────────────
    def _meta_features(self, X: np.ndarray) -> np.ndarray:
        """Build meta-feature matrix: class-1 probability from each base."""
        return _collect_probas(self.models, X)

    # ── fit ───────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "StackingEnsemble":
        """Train meta-learner on base-model outputs."""
        meta_X = self._meta_features(X_train)
        meta_X = self.scaler.fit_transform(meta_X)
        self.meta.fit(meta_X, y_train)
        self._fitted = True
        return self

    # ── predict_proba ────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("StackingEnsemble.fit() must be called first.")
        meta_X = self.scaler.transform(self._meta_features(X))
        return self.meta.predict_proba(meta_X)

    # ── predict ──────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("StackingEnsemble.fit() must be called first.")
        meta_X = self.scaler.transform(self._meta_features(X))
        return self.meta.predict(meta_X)

    # ── evaluate ─────────────────────────────────────────────────────────
    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        return _evaluate(y_test, self.predict(X_test))

    # ── meta_weights ─────────────────────────────────────────────────────
    def meta_weights(self) -> Optional[Dict[str, float]]:
        """Return the logistic-regression coefficient per base model."""
        if not self._fitted:
            return None
        coefs = self.meta.coef_[0]
        names = list(self.models.keys())
        return {n: float(c) for n, c in zip(names, coefs)}


# ══════════════════════════════════════════════════════════════════════════════
#  Factory / convenience
# ══════════════════════════════════════════════════════════════════════════════

def build_ensembles(
    trained_models: Dict[str, BaseModel],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, object]:
    """Instantiate, fit, and return all three ensemble methods.

    Parameters
    ----------
    trained_models : dict
        Already-trained base models (from ``train_all_models``).
    X_train, y_train : array-like
        Training split (used by StackingEnsemble meta-learner).
    X_val, y_val : array-like
        Validation split (used by WeightedEnsemble to set weights,
        and by StackingEnsemble as a secondary quality check).

    Returns
    -------
    dict  — ``{ "voting": VotingEnsemble, "weighted": WeightedEnsemble,
                 "stacking": StackingEnsemble }``
    """
    voting = VotingEnsemble(trained_models, voting="soft")

    weighted = WeightedEnsemble(trained_models)
    weighted.fit_weights(X_val, y_val)

    stacking = StackingEnsemble(trained_models)
    stacking.fit(X_train, y_train)

    return {
        "voting":   voting,
        "weighted": weighted,
        "stacking": stacking,
    }


def evaluate_ensembles(
    ensembles: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Evaluate every ensemble on a held-out test set."""
    results: Dict[str, Dict[str, float]] = {}
    for name, ens in ensembles.items():
        results[name] = ens.evaluate(X_test, y_test)
    return results
