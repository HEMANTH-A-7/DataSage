"""
ModelRecommender — Scores and ranks ML models based on dataset characteristics.
Uses a multi-criteria scoring system: data size, feature types, quality, interpretability.
"""
from typing import List
from core.model_registry import MODEL_REGISTRY


class ModelRecommender:
    """Ranks applicable models for a given task using heuristic scoring."""

    def recommend(self, task_type: str, schema: dict, eda: dict) -> List[dict]:
        candidates = {k: v for k, v in MODEL_REGISTRY.items() if v["task"] == task_type}
        n_rows = schema.get("n_rows", 1000)
        n_cols = schema.get("n_cols", 10)
        quality_score = eda.get("data_quality_score", 70)
        n_missing_cols = len(eda.get("missing", {}))
        n_outlier_cols = len(eda.get("outliers", {}))
        n_numeric = schema.get("columns", {})
        numeric_count = sum(1 for m in schema.get("columns", {}).values() if m["col_type"] == "numeric")
        cat_count = sum(1 for m in schema.get("columns", {}).values() if m["col_type"] == "categorical")

        scored = []
        for key, model in candidates.items():
            score = self._score_model(
                key, model, n_rows, n_cols, numeric_count, cat_count,
                quality_score, n_missing_cols, n_outlier_cols
            )
            scored.append({
                "key": key,
                "name": model["name"],
                "score": round(score, 2),
                "complexity": model["complexity"],
                "interpretability": model["interpretability"],
                "pros": model["pros"],
                "cons": model["cons"],
                "use_when": model["use_when"],
                "params": model["params"],
                "reasoning": self._generate_reasoning(key, model, n_rows, score),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        # Top recommendation gets a badge
        if scored:
            scored[0]["recommended"] = True
        return scored

    def _score_model(
        self, key, model, n_rows, n_cols, numeric_count, cat_count,
        quality_score, n_missing_cols, n_outlier_cols
    ) -> float:
        score = 50.0

        # Data size
        if n_rows < 500:
            if model["complexity"] == "low":
                score += 15
            elif model["complexity"] == "high":
                score -= 10
        elif n_rows < 10_000:
            if model["complexity"] in ("low", "medium"):
                score += 10
            else:
                score += 5
        else:
            if model["complexity"] == "high":
                score += 15
            else:
                score += 5

        # Data quality penalties
        if quality_score < 60:
            if "Robust" in " ".join(model["pros"]):
                score += 10
            if "Handles missing values" in " ".join(model["pros"]):
                score += 8

        # Outliers — robust models get bonus
        if n_outlier_cols > 2:
            if key in ("random_forest_reg", "random_forest_clf", "xgboost_reg", "xgboost_clf", "dbscan"):
                score += 8

        # Interpretability bonus for low complexity
        if model["interpretability"] in ("high", "very_high"):
            score += 5

        # XGBoost/GBDT general power bonus
        if key in ("xgboost_reg", "xgboost_clf", "gradient_boosting_reg", "gradient_boosting_clf"):
            score += 10

        # Linear models need mostly numeric
        if key in ("linear_regression", "ridge", "logistic_regression", "svr", "svm_clf"):
            if cat_count > numeric_count:
                score -= 8

        return min(score, 100.0)

    def _generate_reasoning(self, key: str, model: dict, n_rows: int, score: float) -> str:
        reasons = []
        if score >= 75:
            reasons.append(f"**{model['name']}** is highly recommended for this dataset.")
        elif score >= 60:
            reasons.append(f"**{model['name']}** is a solid choice.")
        else:
            reasons.append(f"**{model['name']}** may work but consider alternatives.")

        reasons.append(f"Best suited when: *{model['use_when']}*.")
        if n_rows < 500:
            reasons.append("With a small dataset, simpler models may generalize better.")
        elif n_rows > 50_000:
            reasons.append("With large data, ensemble methods will shine.")

        return " ".join(reasons)
