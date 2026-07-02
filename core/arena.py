"""
ModelArena — Multi-model comparison engine.
Runs ALL applicable models, ranks them on a leaderboard,
and produces radar-chart comparison data.
"""
import time
import numpy as np
import pandas as pd
from typing import List, Dict
from core.model_registry import MODEL_REGISTRY
from core.workflow_runner import WorkflowRunner


class ModelArena:
    """Run all applicable models and produce a ranked comparison."""

    def __init__(self):
        self.runner = WorkflowRunner()

    def compete(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict:
        """Run all models for the given task type and return ranked results."""
        candidates = {k: v for k, v in MODEL_REGISTRY.items() if v["task"] == task_type}
        results = []
        errors = []

        for key, meta in candidates.items():
            try:
                start = time.time()
                result = self.runner.run(df=df, target_column=target_column, task_type=task_type, model_key=key)
                result["training_time_sec"] = round(time.time() - start, 2)
                results.append(result)
            except Exception as e:
                errors.append({"model_key": key, "model_name": meta["name"], "error": str(e)})

        leaderboard = self._build_leaderboard(results, task_type)
        radar_data = self._build_radar_data(results, task_type)
        winner = leaderboard[0] if leaderboard else None

        return {
            "leaderboard": leaderboard,
            "radar_data": radar_data,
            "total_models": len(candidates),
            "successful": len(results),
            "failed": len(errors),
            "errors": errors,
            "winner": winner,
            "task_type": task_type,
            "full_results": {r["model_key"]: r for r in results},
        }

    def _build_leaderboard(self, results: List[Dict], task_type: str) -> List[Dict]:
        primary = self._get_primary_metric(task_type)
        higher = primary not in ("rmse", "mae", "aic")
        entries = []
        for r in results:
            m = r.get("metrics", {})
            val = m.get(primary, 0)
            if val is None:
                val = 0
            entries.append({
                "rank": 0, "model_key": r["model_key"], "model_name": r["model_name"],
                "primary_metric": primary, "primary_value": round(float(val), 4),
                "all_metrics": m, "training_time_sec": r.get("training_time_sec", 0),
                "n_features": r.get("n_features", 0),
            })
        entries.sort(key=lambda x: x["primary_value"], reverse=higher)
        badges = ["🥇 CHAMPION", "🥈 RUNNER-UP", "🥉 BRONZE"]
        for i, e in enumerate(entries):
            e["rank"] = i + 1
            e["badge"] = badges[i] if i < 3 else ""
        return entries

    def _build_radar_data(self, results: List[Dict], task_type: str) -> Dict:
        if not results:
            return {}
        if task_type == "regression":
            axes = ["r2", "cv_r2_mean", "consistency"]
            labels = ["R²", "CV R²", "Consistency"]
        elif task_type == "classification":
            axes = ["accuracy", "f1", "precision", "recall"]
            labels = ["Accuracy", "F1", "Precision", "Recall"]
        elif task_type == "clustering":
            axes = ["silhouette_score", "calinski_harabasz_scaled", "davies_bouldin_scaled"]
            labels = ["Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin"]
        else:
            return {}

        # Precompute max Calinski-Harabasz for normalization in clustering
        max_ch = 1.0
        if task_type == "clustering":
            ch_vals = []
            for r in results:
                v = r.get("metrics", {}).get("calinski_harabasz")
                if v is not None:
                    ch_vals.append(float(v))
            if ch_vals:
                max_ch = max(ch_vals) or 1.0

        models = []
        for r in results[:5]:
            m = r.get("metrics", {})
            vals = []
            for ax in axes:
                if ax == "consistency":
                    v = 1.0 - float(m.get("cv_r2_std", 0) or 0)
                    v = max(0.0, min(1.0, v)) * 100.0
                elif ax == "calinski_harabasz_scaled":
                    ch = float(m.get("calinski_harabasz", 0) or 0)
                    v = (ch / max_ch) * 100.0
                elif ax == "davies_bouldin_scaled":
                    db = float(m.get("davies_bouldin", 0) or 0)
                    v = (1.0 / (1.0 + db)) * 100.0
                elif ax == "silhouette_score":
                    sil = float(m.get("silhouette_score", 0) or 0)
                    v = ((sil + 1.0) / 2.0) * 100.0
                else:
                    v = m.get(ax, 0)
                    if v is None:
                        v = 0
                    v = float(v) * 100.0 if v <= 1.0 else float(v)
                vals.append(round(v, 1))
            models.append({"model_name": r["model_name"], "model_key": r["model_key"], "values": vals})
        return {"axis_labels": labels, "models": models}

    def _get_primary_metric(self, task_type: str) -> str:
        return {"regression": "r2", "classification": "f1", "clustering": "silhouette_score"}.get(task_type, "rmse")
