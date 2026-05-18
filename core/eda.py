"""
EDAEngine — Automated Exploratory Data Analysis.
Produces statistics, correlations, distribution info, and data quality flags.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any


class EDAEngine:
    """Performs comprehensive automated EDA on a DataFrame."""

    def analyze(self, df: pd.DataFrame, schema: dict) -> dict:
        report = {
            "summary": self._basic_summary(df, schema),
            "missing": self._missing_analysis(df),
            "distributions": self._distribution_analysis(df, schema),
            "correlations": self._correlation_analysis(df, schema),
            "outliers": self._outlier_analysis(df, schema),
            "data_quality_score": 0.0,
        }
        report["data_quality_score"] = self._compute_quality_score(df, report)
        return report

    # ──────────────────────────────────────────────────────────────────────────
    def _basic_summary(self, df: pd.DataFrame, schema: dict) -> dict:
        numeric_cols = [c for c, m in schema["columns"].items() if m["col_type"] == "numeric"]
        categorical_cols = [c for c, m in schema["columns"].items() if m["col_type"] == "categorical"]
        datetime_cols = schema.get("datetime_cols", [])

        summary: dict[str, Any] = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "numeric_count": len(numeric_cols),
            "categorical_count": len(categorical_cols),
            "datetime_count": len(datetime_cols),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing_cells": int(df.isna().sum().sum()),
        }

        # Per-column statistics for numeric
        stats_rows = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            stats_rows[col] = {
                "mean": _sf(s.mean()),
                "median": _sf(s.median()),
                "std": _sf(s.std()),
                "min": _sf(s.min()),
                "max": _sf(s.max()),
                "q25": _sf(s.quantile(0.25)),
                "q75": _sf(s.quantile(0.75)),
                "skewness": _sf(s.skew()),
                "kurtosis": _sf(s.kurtosis()),
            }
        summary["numeric_stats"] = stats_rows

        # Categorical value counts (top 10)
        cat_freq = {}
        for col in categorical_cols:
            vc = df[col].value_counts(normalize=True).head(10)
            cat_freq[col] = {str(k): round(float(v), 4) for k, v in vc.items()}
        summary["categorical_freq"] = cat_freq

        return summary

    def _missing_analysis(self, df: pd.DataFrame) -> dict:
        total = len(df)
        missing = {}
        for col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                missing[col] = {"count": n, "pct": round(n / total * 100, 2)}
        return missing

    def _distribution_analysis(self, df: pd.DataFrame, schema: dict) -> dict:
        numeric_cols = [c for c, m in schema["columns"].items() if m["col_type"] == "numeric"]
        result = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 5:
                continue
            # Normality test (Shapiro-Wilk for n<5000, else D'Agostino)
            try:
                if len(s) <= 5000:
                    stat, p = stats.shapiro(s.sample(min(len(s), 2000), random_state=42))
                else:
                    stat, p = stats.normaltest(s)
                is_normal = bool(p > 0.05)
            except Exception:
                is_normal = False
            # Histogram buckets (20 bins)
            counts, edges = np.histogram(s, bins=20)
            result[col] = {
                "is_normal": is_normal,
                "skewness": _sf(s.skew()),
                "hist_counts": counts.tolist(),
                "hist_edges": [_sf(e) for e in edges.tolist()],
            }
        return result

    def _correlation_analysis(self, df: pd.DataFrame, schema: dict) -> dict:
        numeric_cols = [c for c, m in schema["columns"].items() if m["col_type"] == "numeric"]
        if len(numeric_cols) < 2:
            return {}
        sub = df[numeric_cols].select_dtypes(include=[np.number])
        corr = sub.corr(method="pearson")
        # Return as nested dict; round to 4dp
        result = {}
        for col in corr.columns:
            result[col] = {k: _sf(v) for k, v in corr[col].items()}
        return result

    def _outlier_analysis(self, df: pd.DataFrame, schema: dict) -> dict:
        numeric_cols = [c for c, m in schema["columns"].items() if m["col_type"] == "numeric"]
        result = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = int(((s < lower) | (s > upper)).sum())
            if n_outliers > 0:
                result[col] = {
                    "n_outliers": n_outliers,
                    "pct": round(n_outliers / len(s) * 100, 2),
                    "lower_bound": _sf(lower),
                    "upper_bound": _sf(upper),
                }
        return result

    def _compute_quality_score(self, df: pd.DataFrame, report: dict) -> float:
        score = 100.0
        n = len(df)
        if n == 0:
            return 0.0
        missing_pct = report["summary"]["total_missing_cells"] / (n * len(df.columns)) * 100
        score -= min(missing_pct * 2, 30)
        dup_pct = report["summary"]["duplicate_rows"] / n * 100
        score -= min(dup_pct, 20)
        if report["outliers"]:
            avg_outlier_pct = sum(v["pct"] for v in report["outliers"].values()) / len(report["outliers"])
            score -= min(avg_outlier_pct, 15)
        return round(max(score, 0.0), 1)


def _sf(val) -> float:
    """Safe float — handles NaN/Inf."""
    try:
        f = float(val)
        return round(f, 4) if not (np.isnan(f) or np.isinf(f)) else 0.0
    except Exception:
        return 0.0
