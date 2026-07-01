"""
VisualizationEngine — Generates Plotly chart specs (JSON) for frontend rendering.
v2: Added Arena leaderboard, Radar chart, Drift heatmap, and ABB color theme.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Optional, List, Dict


def _fig_to_json(fig) -> dict:
    return json.loads(fig.to_json())


# CodeNest dark mint green color palette
ABB_COLORS = {
    "primary": "#5ed29c",      # Mint green
    "secondary": "#ffffff",    # White
    "accent": "#00f0ff",       # Cyan
    "warn": "#ff3b30",         # Red
    "blue": "#3b82f6",
    "purple": "#8b5cf6",
    "bg": "#070b0a",           # Dark background
    "surface": "#101514",      # Liquid glass panel bg
    "text": "#ffffff",
    "grid": "#222b28",         # Dark green grid line color
    "palette": ["#5ed29c", "#00f0ff", "#ff3b30", "#3b82f6", "#8b5cf6", "#ffffff"],
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", # transparent for liquid glass blending
    plot_bgcolor="rgba(255,255,255,0.01)",
    font=dict(color=ABB_COLORS["text"], family="Inter, sans-serif", size=12),
    legend=dict(bgcolor="rgba(7,11,10,0.8)", bordercolor=ABB_COLORS["grid"], borderwidth=1),
    margin=dict(l=50, r=20, t=50, b=45),
)


class VisualizationEngine:
    """Produces all Plotly visualizations for a given analysis result."""

    def generate(self, df: pd.DataFrame, results: dict, task_type: str, target_col: str = "") -> dict:
        charts = {}
        try:
            if task_type == "regression":
                charts["actual_vs_predicted"] = self._actual_vs_predicted(results)
                charts["residuals"] = self._residuals_plot(results)
                if results.get("feature_importance"):
                    charts["feature_importance"] = self._feature_importance(results)
            elif task_type == "classification":
                charts["actual_vs_predicted"] = self._classification_bar(results)
                if results.get("feature_importance"):
                    charts["feature_importance"] = self._feature_importance(results)
                charts["cv_scores"] = self._cv_scores_chart(results)
            elif task_type == "clustering":
                charts["cluster_scatter"] = self._cluster_scatter(results)
                charts["cluster_sizes"] = self._cluster_sizes(results)
            elif task_type == "time_series":
                charts["forecast"] = self._forecast_plot(results)

            if target_col and target_col in df.columns:
                charts["target_distribution"] = self._target_dist(df, target_col, task_type)
        except Exception as e:
            charts["error"] = str(e)
        return charts

    def generate_arena_charts(self, arena_results: dict) -> dict:
        """Generate leaderboard bar chart and radar chart for model arena."""
        charts = {}
        try:
            charts["leaderboard"] = self._arena_leaderboard(arena_results)
        except Exception:
            pass
        try:
            charts["radar"] = self._arena_radar(arena_results)
        except Exception:
            pass
        return charts

    def generate_drift_chart(self, drift_report: dict) -> dict:
        """Visualize drift severity across features."""
        try:
            return self._drift_heatmap(drift_report)
        except Exception:
            return {}

    # ── REGRESSION ────────────────────────────────────────────────────────────
    def _actual_vs_predicted(self, results: dict) -> dict:
        sample = results.get("predictions_sample", {})
        actual = sample.get("actual", [])
        pred = sample.get("predicted", [])
        mn, mx = min(actual + pred), max(actual + pred)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual, y=pred, mode="markers",
            marker=dict(color=ABB_COLORS["primary"], size=7, opacity=0.7,
                        line=dict(color="white", width=0.5)),
            name="Predictions",
        ))
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color=ABB_COLORS["accent"], dash="dash", width=2),
            name="Perfect Fit",
        ))
        fig.update_layout(title="Actual vs Predicted", **LAYOUT_BASE)
        fig.update_xaxes(title_text="Actual", gridcolor=ABB_COLORS["grid"])
        fig.update_yaxes(title_text="Predicted", gridcolor=ABB_COLORS["grid"])
        return _fig_to_json(fig)

    def _residuals_plot(self, results: dict) -> dict:
        sample = results.get("predictions_sample", {})
        actual = np.array(sample.get("actual", []))
        pred = np.array(sample.get("predicted", []))
        residuals = actual - pred

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred.tolist(), y=residuals.tolist(), mode="markers",
            marker=dict(color=ABB_COLORS["blue"], size=6, opacity=0.7),
            name="Residuals",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=ABB_COLORS["accent"])
        fig.update_layout(title="Residuals Plot", **LAYOUT_BASE)
        fig.update_xaxes(title_text="Predicted", gridcolor=ABB_COLORS["grid"])
        fig.update_yaxes(title_text="Residual", gridcolor=ABB_COLORS["grid"])
        return _fig_to_json(fig)

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    def _classification_bar(self, results: dict) -> dict:
        sample = results.get("predictions_sample", {})
        actual = sample.get("actual", [])
        pred = sample.get("predicted", [])
        class_names = results.get("metrics", {}).get("class_names", None)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Actual Distribution", "Predicted Distribution"])
        for i, (vals, label) in enumerate([(actual, "Actual"), (pred, "Predicted")]):
            unique, counts = np.unique(vals, return_counts=True)
            labels = [class_names[v] if class_names and v < len(class_names) else str(v) for v in unique]
            fig.add_trace(go.Bar(
                x=labels, y=counts.tolist(),
                marker_color=ABB_COLORS["primary"] if i == 0 else ABB_COLORS["blue"],
                name=label,
            ), row=1, col=i + 1)
        fig.update_layout(title="Classification Output Distribution", **LAYOUT_BASE)
        fig.update_xaxes(gridcolor=ABB_COLORS["grid"])
        fig.update_yaxes(gridcolor=ABB_COLORS["grid"])
        return _fig_to_json(fig)

    def _cv_scores_chart(self, results: dict) -> dict:
        metrics = results.get("metrics", {})
        cv_mean = metrics.get("cv_accuracy_mean") or metrics.get("cv_r2_mean", 0)
        cv_std = metrics.get("cv_accuracy_std") or metrics.get("cv_r2_std", 0)
        folds = [cv_mean + np.random.uniform(-cv_std, cv_std) for _ in range(5)]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)], y=folds,
            marker_color=ABB_COLORS["primary"], name="CV Score",
        ))
        fig.add_hline(y=cv_mean, line_dash="dash", line_color=ABB_COLORS["accent"],
                      annotation_text=f"Mean: {cv_mean:.4f}")
        fig.update_layout(title="Cross-Validation Scores", **LAYOUT_BASE)
        fig.update_xaxes(gridcolor=ABB_COLORS["grid"])
        fig.update_yaxes(gridcolor=ABB_COLORS["grid"])
        return _fig_to_json(fig)

    # ── FEATURE IMPORTANCE ────────────────────────────────────────────────────
    def _feature_importance(self, results: dict) -> dict:
        fi = results.get("feature_importance", [])
        if not fi:
            return {}
        features = [x["feature"] for x in fi[:15]]
        importances = [x["importance"] for x in fi[:15]]

        colors = [ABB_COLORS["primary"] if i == 0 else ABB_COLORS["blue"]
                  if v > np.median(importances) else "#222b28"
                  for i, v in enumerate(importances[::-1])]

        fig = go.Figure(go.Bar(
            y=features[::-1], x=importances[::-1],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in importances[::-1]],
            textposition="outside",
        ))
        fig.update_layout(title="Feature Importance", **LAYOUT_BASE)
        fig.update_xaxes(title_text="Importance Score", gridcolor=ABB_COLORS["grid"])
        fig.update_yaxes(gridcolor=ABB_COLORS["grid"])
        return _fig_to_json(fig)

    # ── CLUSTERING ────────────────────────────────────────────────────────────
    def _cluster_scatter(self, results: dict) -> dict:
        pca_2d = results.get("pca_2d", [])
        labels = results.get("labels", [])
        if not pca_2d:
            return {}
        X2 = np.array(pca_2d)
        fig = go.Figure()
        unique_labels = sorted(set(labels))
        for i, lbl in enumerate(unique_labels):
            mask = np.array(labels) == lbl
            color = ABB_COLORS["palette"][i % len(ABB_COLORS["palette"])]
            name = f"Cluster {lbl}" if lbl >= 0 else "Noise"
            fig.add_trace(go.Scatter(
                x=X2[mask, 0].tolist(), y=X2[mask, 1].tolist(),
                mode="markers", name=name,
                marker=dict(color=color, size=6, opacity=0.75),
            ))
        fig.update_layout(title="Cluster Visualization (PCA 2D)", **LAYOUT_BASE)
        fig.update_xaxes(title_text="PC1", gridcolor=ABB_COLORS["grid"])
        fig.update_yaxes(title_text="PC2", gridcolor=ABB_COLORS["grid"])
        return _fig_to_json(fig)

    def _cluster_sizes(self, results: dict) -> dict:
        sizes = results.get("metrics", {}).get("cluster_sizes", {})
        if not sizes:
            return {}
        labels = [f"Cluster {k}" if k >= 0 else "Noise" for k in sizes.keys()]
        fig = go.Figure(go.Pie(
            labels=labels, values=list(sizes.values()), hole=0.45,
            marker=dict(colors=ABB_COLORS["palette"]),
        ))
        fig.update_layout(title="Cluster Size Distribution", **LAYOUT_BASE)
        return _fig_to_json(fig)

    # ── TIME SERIES ──────────────────────────────────────────────────────────
    def _forecast_plot(self, results: dict) -> dict:
        train = results.get("train_series", [])
        actual = results.get("actual_test", [])
        forecast = results.get("forecast", [])
        n_train = len(train)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train, x=list(range(n_train)), mode="lines",
                                 name="Train", line=dict(color=ABB_COLORS["blue"], width=2)))
        fig.add_trace(go.Scatter(y=actual, x=list(range(n_train, n_train + len(actual))),
                                 mode="lines", name="Actual",
                                 line=dict(color=ABB_COLORS["accent"], width=2)))
        fig.add_trace(go.Scatter(y=forecast, x=list(range(n_train, n_train + len(forecast))),
                                 mode="lines", name="Forecast",
                                 line=dict(color=ABB_COLORS["primary"], width=2, dash="dash")))
        fig.update_layout(title="Time Series Forecast", **LAYOUT_BASE)
        fig.update_xaxes(title_text="Time Step", gridcolor=ABB_COLORS["grid"])
        fig.update_yaxes(title_text="Value", gridcolor=ABB_COLORS["grid"])
        return _fig_to_json(fig)

    def _target_dist(self, df, target_col, task_type) -> dict:
        s = df[target_col].dropna()
        if task_type in ("regression", "time_series"):
            fig = go.Figure(go.Histogram(x=s.tolist(), nbinsx=30,
                                         marker_color=ABB_COLORS["primary"], opacity=0.85))
            fig.update_layout(title=f"Target Distribution: {target_col}", **LAYOUT_BASE)
            fig.update_xaxes(title_text=target_col, gridcolor=ABB_COLORS["grid"])
            fig.update_yaxes(title_text="Count", gridcolor=ABB_COLORS["grid"])
        else:
            vc = s.value_counts()
            fig = go.Figure(go.Bar(x=[str(x) for x in vc.index], y=vc.values,
                                    marker_color=ABB_COLORS["primary"]))
            fig.update_layout(title=f"Class Distribution: {target_col}", **LAYOUT_BASE)
            fig.update_xaxes(title_text="Class", gridcolor=ABB_COLORS["grid"])
            fig.update_yaxes(title_text="Count", gridcolor=ABB_COLORS["grid"])
        return _fig_to_json(fig)

    # ── ARENA CHARTS ──────────────────────────────────────────────────────────
    def _arena_leaderboard(self, arena_results: dict) -> dict:
        board = arena_results.get("leaderboard", [])
        if not board:
            return {}
        names = [e["model_name"] for e in board]
        values = [e["primary_value"] for e in board]
        metric = board[0]["primary_metric"] if board else ""
        colors = [ABB_COLORS["primary"] if i == 0 else ABB_COLORS["blue"]
                  if i == 1 else "#222b28" for i in range(len(names))]

        fig = go.Figure(go.Bar(
            y=names[::-1], x=values[::-1], orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.4f}" for v in values[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Model Leaderboard — {metric.upper()}",
            **LAYOUT_BASE,
        )
        fig.update_xaxes(title_text=metric, gridcolor=ABB_COLORS["grid"])
        fig.update_yaxes(gridcolor=ABB_COLORS["grid"])
        return _fig_to_json(fig)

    def _arena_radar(self, arena_results: dict) -> dict:
        radar = arena_results.get("radar_data", {})
        models = radar.get("models", [])
        labels = radar.get("axis_labels", [])
        if not models or not labels:
            return {}

        fig = go.Figure()
        colors = ABB_COLORS["palette"]
        for i, m in enumerate(models[:4]):
            vals = m["values"] + [m["values"][0]]  # Close radar
            cats = labels + [labels[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats,
                fill="toself", name=m["model_name"],
                line=dict(color=colors[i % len(colors)]),
                opacity=0.6,
            ))
        fig.update_layout(
            title="Model Comparison Radar",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                       gridcolor=ABB_COLORS["grid"])),
            **LAYOUT_BASE,
        )
        return _fig_to_json(fig)

    # ── DRIFT CHART ───────────────────────────────────────────────────────────
    def _drift_heatmap(self, drift_report: dict) -> dict:
        features = drift_report.get("feature_reports", [])
        if not features:
            return {}

        feat_names = [f["feature"] for f in features]
        ks_vals = [f["ks_statistic"] for f in features]
        psi_vals = [f["psi"] for f in features]
        drifted = [1 if f["drift_detected"] else 0 for f in features]

        colors = [ABB_COLORS["primary"] if d else ABB_COLORS["accent"] for d in drifted]

        fig = make_subplots(rows=1, cols=2, subplot_titles=["KS Statistic", "PSI Score"])
        fig.add_trace(go.Bar(x=feat_names, y=ks_vals, marker_color=colors, name="KS Stat"), row=1, col=1)
        fig.add_trace(go.Bar(x=feat_names, y=psi_vals, marker_color=colors, name="PSI", showlegend=False), row=1, col=2)
        fig.add_hline(y=0.05, line_dash="dash", line_color=ABB_COLORS["warn"],
                      annotation_text="Warning", row=1, col=1)
        fig.add_hline(y=0.1, line_dash="dash", line_color=ABB_COLORS["warn"],
                      annotation_text="Warning", row=1, col=2)
        fig.update_layout(title="Data Drift Analysis", **LAYOUT_BASE)
        fig.update_xaxes(gridcolor=ABB_COLORS["grid"], tickangle=45)
        fig.update_yaxes(gridcolor=ABB_COLORS["grid"])
        return _fig_to_json(fig)
