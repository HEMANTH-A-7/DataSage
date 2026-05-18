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
            axes = ["r2", "cv_r2_mean"]
            labels = ["R²", "CV R²"]
        elif task_type == "classification":
            axes = ["accuracy", "f1", "precision", "recall"]
            labels = ["Accuracy", "F1", "Precision", "Recall"]
        elif task_type == "clustering":
            return {}
        else:
            return {}
        models = []
        for r in results[:5]:
            m = r.get("metrics", {})
            vals = []
            for ax in axes:
                v = m.get(ax, 0)
                if v is None:
                    v = 0
                vals.append(round(float(v) * 100 if float(v) <= 1 else float(v), 1))
            models.append({"model_name": r["model_name"], "model_key": r["model_key"], "values": vals})
        return {"axis_labels": labels, "models": models}

    def _get_primary_metric(self, task_type: str) -> str:
        return {"regression": "r2", "classification": "f1", "clustering": "silhouette_score"}.get(task_type, "rmse")
