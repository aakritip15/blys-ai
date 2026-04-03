from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths (resolved relative to this file's directory) ──────────────────────
_BASE     = Path(__file__).parent
_MODEL_PATH = _BASE / "outputs" / "models" / "recommendation_model.pkl"
_DATA_PATH  = _BASE / "outputs" / "data"  / "customer_data_cleaned.csv"

# Fallback services used when the model file is absent (dev / demo mode)
_FALLBACK_SERVICES = [
    "Swedish Massage",
    "Deep Tissue Massage",
    "Hot Stone Massage",
    "Aromatherapy",
    "Reflexology",
    "Sports Massage",
    "Prenatal Massage",
    "Couples Massage",
]


# ── RecommendationEngine ─────────────────────────────────────────────────────
class RecommendationEngine:
    """Load once at startup, call .recommend() per request."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_type: str = "similarity"
        self._customer_ids: list[str] = []
        self._services: list[str] = _FALLBACK_SERVICES[:]
        self._matrix: np.ndarray | None = None   # customer × service utility
        self._customer_df: pd.DataFrame | None = None
        self._ready = False

        self._load()

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load pkl bundle + cleaned CSV. Degrades gracefully if files missing."""
        # 1. Load customer data for feature-based fallback
        if _DATA_PATH.exists():
            try:
                self._customer_df = pd.read_csv(_DATA_PATH)
                logger.info("Customer data loaded: %d rows", len(self._customer_df))
            except Exception as exc:
                logger.warning("Could not load customer data: %s", exc)

        # 2. Load model bundle
        if not _MODEL_PATH.exists():
            logger.warning(
                "recommendation_model.pkl not found at %s — running in fallback mode.",
                _MODEL_PATH,
            )
            return

        try:
            with open(_MODEL_PATH, "rb") as fh:
                bundle = pickle.load(fh)

            # Support both plain-model pkl and dict-bundle pkl
            if isinstance(bundle, dict):
                self._model       = bundle.get("model")
                self._model_type  = bundle.get("model_type", "similarity")
                self._customer_ids = [str(c) for c in bundle.get("customer_ids", [])]
                self._services    = bundle.get("services", _FALLBACK_SERVICES)
                self._matrix      = bundle.get("customer_service_matrix")
            else:
                # Plain model object — treat as similarity / predict-based
                self._model      = bundle
                self._model_type = "similarity"

            self._ready = True
            logger.info(
                "Recommendation model loaded (type=%s, customers=%d, services=%d)",
                self._model_type, len(self._customer_ids), len(self._services),
            )
        except Exception as exc:
            logger.error("Failed to load recommendation model: %s", exc)

    def _fallback_recommend(
        self, customer_id: str, top_n: int
    ) -> list[dict[str, Any]]:
        """
        Data-driven fallback when no trained model is available.
        Uses Preferred_Service from customer data + popularity ranking.
        """
        recommendations: list[dict[str, Any]] = []

        if self._customer_df is not None and "Customer_ID" in self._customer_df.columns:
            row = self._customer_df[
                self._customer_df["Customer_ID"].astype(str) == str(customer_id)
            ]
            if not row.empty and "Preferred_Service" in row.columns:
                preferred = row.iloc[0]["Preferred_Service"]
                recommendations.append(
                    {"service": preferred, "score": 1.0, "reason": "Your preferred service"}
                )

        # Fill remaining slots with popularity-ranked services
        used = {r["service"] for r in recommendations}
        if self._customer_df is not None and "Preferred_Service" in self._customer_df.columns:
            popular = (
                self._customer_df["Preferred_Service"]
                .value_counts()
                .index.tolist()
            )
        else:
            popular = _FALLBACK_SERVICES

        rank = 0.9
        for svc in popular:
            if len(recommendations) >= top_n:
                break
            if svc not in used:
                recommendations.append({"service": svc, "score": round(rank, 2), "reason": "Popular with similar customers"})
                used.add(svc)
                rank -= 0.05

        return recommendations[:top_n]

    def _similarity_recommend(
        self, customer_id: str, top_n: int
    ) -> list[dict[str, Any]]:
        """Item-item or user-item similarity matrix lookup."""
        if self._matrix is None or customer_id not in self._customer_ids:
            return self._fallback_recommend(customer_id, top_n)

        idx = self._customer_ids.index(customer_id)
        scores = self._matrix[idx]                         # shape: (n_services,)
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [
            {
                "service": self._services[i],
                "score": round(float(scores[i]), 4),
                "reason": "Based on your booking history",
            }
            for i in top_idx
        ]

    def _knn_recommend(
        self, customer_id: str, top_n: int
    ) -> list[dict[str, Any]]:
        """KNN — find similar customers, surface their top services."""
        if self._matrix is None or customer_id not in self._customer_ids:
            return self._fallback_recommend(customer_id, top_n)

        idx = self._customer_ids.index(customer_id)
        user_vec = self._matrix[idx].reshape(1, -1)

        # distances, indices of neighbours
        distances, neighbour_ids = self._model.kneighbors(user_vec, n_neighbors=min(6, len(self._customer_ids)))
        neighbour_matrix = self._matrix[neighbour_ids[0]]
        # Aggregate neighbour scores, zero out already-booked services
        agg_scores = neighbour_matrix.mean(axis=0)
        # Suppress services the customer already has high scores for
        agg_scores = agg_scores * (1 - np.clip(self._matrix[idx], 0, 1))
        top_idx = np.argsort(agg_scores)[::-1][:top_n]
        return [
            {
                "service": self._services[i],
                "score": round(float(agg_scores[i]), 4),
                "reason": "Customers like you also booked this",
            }
            for i in top_idx
        ]

    def _svd_recommend(
        self, customer_id: str, top_n: int
    ) -> list[dict[str, Any]]:
        """SVD (surprise / implicit) — predict ratings for all services."""
        if customer_id not in self._customer_ids:
            return self._fallback_recommend(customer_id, top_n)

        scored: list[tuple[float, str]] = []
        for service in self._services:
            try:
                pred = self._model.predict(customer_id, service)
                est = pred.est if hasattr(pred, "est") else float(pred)
            except Exception:
                est = 0.0
            scored.append((est, service))

        scored.sort(reverse=True)
        return [
            {
                "service": svc,
                "score": round(score, 4),
                "reason": "Predicted rating from your history",
            }
            for score, svc in scored[:top_n]
        ]

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def known_customer_ids(self) -> list[str]:
        return self._customer_ids

    def recommend(
        self,
        customer_id: str,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """
        Return top-N service recommendations for a customer.

        Returns
        -------
        {
            "customer_id": str,
            "recommendations": [{"service": str, "score": float, "reason": str}, ...],
            "model_used": str,
            "is_known_customer": bool,
        }
        """
        cid = str(customer_id)
        is_known = cid in self._customer_ids

        if not self._ready:
            recs = self._fallback_recommend(cid, top_n)
            model_used = "fallback (no model loaded)"
        elif self._model_type == "svd":
            recs = self._svd_recommend(cid, top_n)
            model_used = "svd"
        elif self._model_type == "knn":
            recs = self._knn_recommend(cid, top_n)
            model_used = "knn"
        else:
            recs = self._similarity_recommend(cid, top_n)
            model_used = "similarity"

        return {
            "customer_id": cid,
            "recommendations": recs,
            "model_used": model_used,
            "is_known_customer": is_known,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_engine: RecommendationEngine | None = None


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine