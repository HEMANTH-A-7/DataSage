"""
FeatureEngineer — Automated feature engineering suggestions and execution.
Uses heuristics and optional LLM reasoning to recommend transformations
based on data characteristics (skewness, cardinality, correlations).
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class FeatureEngineer:
    """Analyzes dataset features and suggests/applies engineering transformations."""

    def suggest(self, df: pd.DataFrame, schema: dict, eda: dict) -> List[Dict]:
        """Generate smart feature engineering suggestions based on data profile."""
        suggestions = []

        columns = schema.get("columns", {})
        distributions = eda.get("distributions", {})
        correlations = eda.get("correlations", {})
        outliers = eda.get("outliers", {})

        for col, meta in columns.items():
            if meta["col_type"] != "numeric":
                continue

            dist = distributions.get(col, {})
            skewness = abs(dist.get("skewness", 0))

            # High skewness → log transform
            if skewness > 2.0:
                suggestions.append({
                    "feature": col,
                    "transform": "log1p",
                    "reason": f"High skewness ({skewness:.2f}) — log transform will normalize distribution",
                    "priority": "high",
                    "impact": "Reduces outlier influence, improves linear model performance",
                    "code_hint": f"df['{col}_log'] = np.log1p(df['{col}'])",
                })
            elif skewness > 1.0:
                suggestions.append({
                    "feature": col,
                    "transform": "sqrt",
                    "reason": f"Moderate skewness ({skewness:.2f}) — square root can help",
                    "priority": "medium",
                    "impact": "Reduces tail effects for parametric models",
                    "code_hint": f"df['{col}_sqrt'] = np.sqrt(df['{col}'].clip(lower=0))",
                })

            # Outlier-heavy columns → robust scaling
            if col in outliers:
                pct = outliers[col].get("pct", 0)
                if pct > 5:
                    suggestions.append({
                        "feature": col,
                        "transform": "robust_scale",
                        "reason": f"{pct}% outliers detected — robust scaling recommended",
                        "priority": "high",
                        "impact": "IQR-based scaling resists outlier distortion",
                        "code_hint": f"from sklearn.preprocessing import RobustScaler",
                    })

        # Interaction features for highly correlated pairs
        if correlations:
            checked = set()
            for col_a, corr_row in correlations.items():
                for col_b, corr_val in corr_row.items():
                    if col_a == col_b:
                        continue
                    pair = tuple(sorted([col_a, col_b]))
                    if pair in checked:
                        continue
                    checked.add(pair)

                    if abs(corr_val) > 0.7:
                        suggestions.append({
                            "feature": f"{col_a} × {col_b}",
                            "transform": "interaction",
                            "reason": f"High correlation ({corr_val:.2f}) — interaction term may capture non-linear relationship",
                            "priority": "medium",
                            "impact": "Captures multiplicative effects between related features",
                            "code_hint": f"df['{col_a}_x_{col_b}'] = df['{col_a}'] * df['{col_b}']",
                        })

        # Categorical encoding suggestions
        for col, meta in columns.items():
            if meta["col_type"] == "categorical":
                n_unique = meta.get("n_unique", 0)
                if n_unique > 10:
                    suggestions.append({
                        "feature": col,
                        "transform": "target_encode",
                        "reason": f"High cardinality ({n_unique} categories) — target encoding preferred over one-hot",
                        "priority": "high",
                        "impact": "Avoids dimensionality explosion from one-hot encoding",
                        "code_hint": f"# Use category_encoders.TargetEncoder for '{col}'",
                    })
                elif n_unique <= 10:
                    suggestions.append({
                        "feature": col,
                        "transform": "one_hot",
                        "reason": f"Low cardinality ({n_unique} categories) — one-hot encoding is safe",
                        "priority": "low",
                        "impact": "Standard encoding for tree-based and linear models",
                        "code_hint": f"pd.get_dummies(df, columns=['{col}'])",
                    })

        # Datetime feature extraction
        for col in schema.get("datetime_cols", []):
            suggestions.append({
                "feature": col,
                "transform": "datetime_extract",
                "reason": "Datetime column — extract year, month, day_of_week, hour for temporal patterns",
                "priority": "high",
                "impact": "Unlocks seasonal and cyclical patterns in the data",
                "code_hint": f"df['{col}_month'] = pd.to_datetime(df['{col}']).dt.month",
            })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return suggestions

    def apply_transforms(
        self, df: pd.DataFrame, suggestions: List[Dict]
    ) -> pd.DataFrame:
        """Apply selected feature engineering transforms to the dataset."""
        df = df.copy()

        for s in suggestions:
            col = s["feature"]
            transform = s["transform"]

            try:
                if transform == "log1p" and col in df.columns:
                    df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
                elif transform == "sqrt" and col in df.columns:
                    df[f"{col}_sqrt"] = np.sqrt(df[col].clip(lower=0))
                elif transform == "interaction" and "×" in col:
                    parts = col.split(" × ")
                    if len(parts) == 2 and parts[0] in df.columns and parts[1] in df.columns:
                        df[f"{parts[0]}_x_{parts[1]}"] = df[parts[0]] * df[parts[1]]
            except Exception:
                continue

        return df
