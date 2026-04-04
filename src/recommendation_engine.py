from __future__ import annotations

import logging
import os
import pickle
import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#   Paths (resolved relative to this file's directory)  
_BASE     = Path(__file__).parent
_MODEL_PATH = _BASE.parent / "models" / "recommendation_model.pkl"
_DATA_PATH  = _BASE.parent / "data"  / "cleaned_customer_data.csv"

print(_MODEL_PATH, _DATA_PATH)  # sanity check
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


#   RecommendationEngine  
class RecommendationEngine:
    """Load once at startup, call .recommend() per request."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_type: str = "similarity"
        self._customer_ids: list[str] = []
        self._services: list[str] = _FALLBACK_SERVICES[:]
        self._matrix: np.ndarray | None = None   # customer × service utility
        self._train_latent: np.ndarray | None = None  # reduced dim vectors for training customers
        self._customer_latent: np.ndarray | None = None  # reduced dim vectors for all known customers
        self._customer_df: pd.DataFrame | None = None
        self._ready = False

        # Hybrid content-based layer artifacts
        self._svc_feature_vecs: np.ndarray | None = None  # (n_services, feat_dim) one-hot vectors
        self._alpha: float = 0.65  # SVD blend weight (matches notebook default)

        self._load()

    #   private  

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
            try:
                bundle = joblib.load(_MODEL_PATH)
            except Exception:
                with open(_MODEL_PATH, "rb") as fh:
                    bundle = pickle.load(fh)

            # Support both plain-model pkl and dict-bundle pkl
            if isinstance(bundle, dict):
                # model may be stored under various keys
                self._model = bundle.get("model") or bundle.get("svd_model") or bundle.get("knn_model")
                self._model_type = bundle.get("model_type", "similarity")

                # Data consistency: unify ID fields
                id_list = bundle.get("customer_ids") or bundle.get("client_ids") or bundle.get("user_ids")
                self._customer_ids = [str(c) for c in id_list] if id_list is not None else []

                # Service catalog from coefficients/categorization
                self._services = bundle.get("services") or bundle.get("catalog") or _FALLBACK_SERVICES

                # Matrix may use different names
                self._matrix = bundle.get("customer_service_matrix") or bundle.get("interaction_matrix")
                self._train_latent = bundle.get("train_latent")
                self._customer_latent = bundle.get("customer_latent")

                # Load hybrid content-based artifacts
                self._svc_feature_vecs = bundle.get("svc_feature_vecs")
                self._alpha = float(bundle.get("alpha", 0.65))

                # If model_type uses hybrid labels, normalize to supported engine types
                if self._model_type in ("hybrid_svd_content", "hybrid"):
                    self._model_type = "svd"

                # If we loaded explicit train matrix as sparse form, densify for in-memory ops
                if self._matrix is not None and hasattr(self._matrix, "toarray"):
                    self._matrix = self._matrix.toarray()
                if self._train_latent is not None and hasattr(self._train_latent, "toarray"):
                    self._train_latent = self._train_latent.toarray()
                if self._customer_latent is not None and hasattr(self._customer_latent, "toarray"):
                    self._customer_latent = self._customer_latent.toarray()
            else:
                # Plain model object — treat as similarity / predict-based
                self._model = bundle
                self._model_type = "similarity"

            self._ready = True
            logger.info(
                "Recommendation model loaded (type=%s, customers=%d, services=%d)",
                self._model_type,
                len(self._customer_ids),
                len(self._services),
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
        """SVD (TruncatedSVD/hybrid) — predict ratings for all services."""
        if customer_id not in self._customer_ids:
            return self._fallback_recommend(customer_id, top_n)

        idx = self._customer_ids.index(customer_id)

        user_latent = None

        if self._customer_latent is not None and idx < len(self._customer_latent):
            user_latent = np.asarray(self._customer_latent[idx])
        elif self._train_latent is not None and idx < len(self._train_latent):
            user_latent = np.asarray(self._train_latent[idx])
        elif self._matrix is not None and self._model is not None and hasattr(self._model, "transform"):
            user_vec = np.asarray(self._matrix[idx]).reshape(1, -1)
            user_latent = np.asarray(self._model.transform(user_vec)[0])
        elif self._matrix is not None and idx < len(self._matrix):
            user_latent = np.asarray(self._matrix[idx])

        if user_latent is None:
            return self._fallback_recommend(customer_id, top_n)

        if self._model is not None and hasattr(self._model, "components_"):
            comp = np.asarray(self._model.components_)
            if user_latent.shape[0] == comp.shape[0]:
                svd_scores = user_latent.dot(comp)
            else:
                # Fallback to matrix row if dimension mismatch
                svd_scores = np.asarray(self._matrix[idx]) if self._matrix is not None else np.zeros(len(self._services))
        else:
            svd_scores = np.asarray(self._matrix[idx]) if self._matrix is not None else np.zeros(len(self._services))

        # Normalize SVD scores to [0, 1] row-wise (avoids near-zero display bug)
        svd_max = svd_scores.max()
        svd_norm = svd_scores / svd_max if svd_max > 1e-9 else svd_scores

        #   Hybrid blend: add content-based layer if artifacts are available  
        if self._svc_feature_vecs is not None:
            if self._matrix is not None:
                row_affinities = np.asarray(self._matrix[idx], dtype=np.float32)
            else:
                row_affinities = np.zeros(len(self._services), dtype=np.float32)

            total = row_affinities.sum()
            if total > 1e-9:
                taste_profile = (row_affinities[:, None] * self._svc_feature_vecs).sum(axis=0) / total
            else:
                taste_profile = self._svc_feature_vecs.mean(axis=0)

            from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
            content_scores = _cos_sim(taste_profile.reshape(1, -1), self._svc_feature_vecs)[0]
            content_max = content_scores.max()
            content_norm = content_scores / content_max if content_max > 1e-9 else content_scores

            scores = self._alpha * svd_norm + (1.0 - self._alpha) * content_norm
            reason = "Hybrid personalised recommendation"
        else:
            scores = svd_norm
            reason = "Predicted rating from your history"

        # Degrade already-known services to encourage discovery
        if self._matrix is not None:
            existing = np.asarray(self._matrix[idx], dtype=float)
            scores = np.where(existing > 0, scores * 0.75, scores)

        top_idx = np.argsort(scores)[::-1][:top_n]
        recommendations = []
        for i in top_idx:
            recommendations.append(
                {
                    "service": self._services[i],
                    "score": round(float(scores[i]), 4),
                    "reason": reason,
                }
            )

        return recommendations

    #   public  

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

        # Normalize service representation for API compatibility
        # (some model artifacts store service as tuple `(name, category, type)`).
        def _normalise_service_item(item: dict[str, Any]) -> dict[str, Any]:
            svc = item.get("service")
            if isinstance(svc, tuple) and svc:
                item["service"] = str(svc[0])
            elif hasattr(svc, "__iter__") and not isinstance(svc, str) and len(svc) > 0:
                # also cover lists
                item["service"] = str(svc[0])
            else:
                item["service"] = str(svc)
            return item

        recs = [_normalise_service_item(r.copy()) for r in recs]

        return {
            "customer_id": cid,
            "recommendations": recs,
            "model_used": model_used,
            "is_known_customer": is_known,
        }


#   Singleton  
_engine: RecommendationEngine | None = None


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine