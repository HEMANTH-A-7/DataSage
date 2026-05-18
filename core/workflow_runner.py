"""
WorkflowRunner — Executes the full ML pipeline for any task type.
Handles: preprocessing → training → evaluation → feature importance.
"""
import importlib
import numpy as np
import pandas as pd
from typing import Any
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from core.model_registry import MODEL_REGISTRY


class WorkflowRunner:
    """Executes a complete ML pipeline end-to-end."""

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        model_key: str,
    ) -> dict:
        model_meta = MODEL_REGISTRY.get(model_key)
        if not model_meta:
            raise ValueError(f"Unknown model key: {model_key}")

        if task_type == "clustering":
            return self._run_clustering(df, model_meta, model_key)
        elif task_type == "time_series":
            return self._run_time_series(df, target_column, model_meta, model_key)
        elif task_type == "regression":
            return self._run_supervised(df, target_column, model_meta, model_key, "regression")
        elif task_type == "classification":
            return self._run_supervised(df, target_column, model_meta, model_key, "classification")
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    # ──────────────────────────────────────────────────────────────────────────
    def _preprocess(self, df: pd.DataFrame, target_col: str = "") -> tuple:
        """Generic preprocessing: encode categoricals, impute, return X."""
        exclude = [target_col] if target_col else []
        feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols].copy()
        encoders = {}

        # Encode categoricals
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        # Drop non-numeric remaining
        X = X.select_dtypes(include=[np.number])

        # Impute
        imputer = SimpleImputer(strategy="median")
        X_arr = imputer.fit_transform(X)

        return X_arr, list(X.columns), encoders, imputer

    def _instantiate_model(self, model_meta: dict) -> Any:
        module = importlib.import_module(model_meta["module"])
        cls = getattr(module, model_meta["class"])
        return cls(**model_meta["params"])

    # ── SUPERVISED ────────────────────────────────────────────────────────────
    def _run_supervised(self, df, target_col, model_meta, model_key, task) -> dict:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        X, feature_names, _, _ = self._preprocess(df, target_col)
        y_raw = df[target_col].copy()

        # Encode target for classification
        le_target = None
        if task == "classification":
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
        else:
            y = y_raw.to_numpy(dtype=float)

        # Remove rows where y is NaN
        valid = ~np.isnan(y) if task == "regression" else np.ones(len(y), dtype=bool)
        X, y = X[valid], y[valid]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task == "classification" else None,
        )

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = self._instantiate_model(model_meta)

        # Use raw for tree-based, scaled for linear/SVM
        use_scaled = model_key in ("linear_regression", "ridge", "logistic_regression", "svr", "svm_clf")
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        # Metrics
        if task == "regression":
            metrics = {
                "r2": round(float(r2_score(y_test, y_pred)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            }
            cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring="r2")
            metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
            metrics["cv_r2_std"] = round(float(cv_scores.std()), 4)
        else:
            avg = "binary" if len(np.unique(y)) == 2 else "weighted"
            metrics = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "f1": round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                "precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            }
            cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy")
            metrics["cv_accuracy_mean"] = round(float(cv_scores.mean()), 4)
            metrics["cv_accuracy_std"] = round(float(cv_scores.std()), 4)
            # Class names
            if le_target:
                metrics["class_names"] = le_target.classes_.tolist()

        # Feature importance
        feature_importance = self._get_feature_importance(model, feature_names)

        # Predictions sample
        n_sample = min(50, len(y_test))
        predictions_sample = {
            "actual": y_test[:n_sample].tolist(),
            "predicted": y_pred[:n_sample].tolist(),
        }

        return {
            "task_type": task,
            "model_key": model_key,
            "model_name": model_meta["name"],
            "metrics": metrics,
            "feature_importance": feature_importance,
            "predictions_sample": predictions_sample,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": len(feature_names),
            "feature_names": feature_names,
        }

    # ── CLUSTERING ────────────────────────────────────────────────────────────
    def _run_clustering(self, df, model_meta, model_key) -> dict:
        X, feature_names, _, _ = self._preprocess(df)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        model = self._instantiate_model(model_meta)
        labels = model.fit_predict(X_s)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        metrics = {"n_clusters_found": n_clusters}

        if n_clusters >= 2:
            try:
                sil = silhouette_score(X_s, labels, sample_size=min(5000, len(X_s)))
                metrics["silhouette_score"] = round(float(sil), 4)
            except Exception:
                metrics["silhouette_score"] = None

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        metrics["cluster_sizes"] = {int(k): int(v) for k, v in zip(unique, counts)}

        # Cluster centers (KMeans only)
        cluster_centers = None
        if hasattr(model, "cluster_centers_"):
            cluster_centers = model.cluster_centers_.tolist()

        # PCA 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_s)

        return {
            "task_type": "clustering",
            "model_key": model_key,
            "model_name": model_meta["name"],
            "metrics": metrics,
            "labels": labels.tolist(),
            "pca_2d": X_2d.tolist(),
            "cluster_centers": cluster_centers,
            "feature_names": feature_names,
            "n_samples": len(df),
        }

    # ── TIME SERIES ──────────────────────────────────────────────────────────
    def _run_time_series(self, df, target_col, model_meta, model_key) -> dict:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in dataset.")

        # Use lag-feature approach for RF; ARIMA/ES for statsmodels
        series = df[target_col].dropna().reset_index(drop=True)
        n = len(series)

        if n < 10:
            raise ValueError("Time series requires at least 10 data points.")

        if model_key == "arima":
            return self._run_arima(series, model_meta, model_key)
        elif model_key == "exponential_smoothing":
            return self._run_exp_smoothing(series, model_meta, model_key)
        else:
            return self._run_ts_ml(series, model_meta, model_key)

    def _run_arima(self, series, model_meta, model_key) -> dict:
        from statsmodels.tsa.arima.model import ARIMA
        n = len(series)
        train_size = int(n * 0.8)
        train, test = series[:train_size], series[train_size:]

        order = model_meta["params"].get("order", (1, 1, 1))
        model = ARIMA(train, order=order)
        fit = model.fit()

        forecast = fit.forecast(steps=len(test))
        actual = test.values

        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(actual, forecast))), 4),
            "mae": round(float(mean_absolute_error(actual, forecast)), 4),
            "aic": round(float(fit.aic), 4),
        }

        return {
            "task_type": "time_series",
            "model_key": model_key,
            "model_name": model_meta["name"],
            "metrics": metrics,
            "forecast": forecast.tolist(),
            "actual_test": actual.tolist(),
            "train_series": series[:train_size].tolist(),
        }

    def _run_exp_smoothing(self, series, model_meta, model_key) -> dict:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        n = len(series)
        train_size = int(n * 0.8)
        train, test = series[:train_size], series[train_size:]

        try:
            params = {k: v for k, v in model_meta["params"].items()}
            model = ExponentialSmoothing(train, **params)
            fit = model.fit()
        except Exception:
            model = ExponentialSmoothing(train, trend="add")
            fit = model.fit()

        forecast = fit.forecast(len(test))
        actual = test.values
        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(actual, forecast))), 4),
            "mae": round(float(mean_absolute_error(actual, forecast)), 4),
        }

        return {
            "task_type": "time_series",
            "model_key": model_key,
            "model_name": model_meta["name"],
            "metrics": metrics,
            "forecast": list(forecast),
            "actual_test": actual.tolist(),
            "train_series": train.tolist(),
        }

    def _run_ts_ml(self, series, model_meta, model_key) -> dict:
        """Lag-feature based ML for time series."""
        lags = 5
        X_list, y_list = [], []
        for i in range(lags, len(series)):
            X_list.append(series[i - lags:i].values)
            y_list.append(series[i])
        X = np.array(X_list)
        y = np.array(y_list)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = self._instantiate_model(model_meta)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
        }

        return {
            "task_type": "time_series",
            "model_key": model_key,
            "model_name": model_meta["name"],
            "metrics": metrics,
            "forecast": y_pred.tolist(),
            "actual_test": y_test.tolist(),
            "train_series": series.tolist()[:split + lags],
        }

    # ── HELPERS ───────────────────────────────────────────────────────────────
    def _get_feature_importance(self, model, feature_names: list) -> list:
        importances = []
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
            importances = sorted(
                [{"feature": f, "importance": round(float(v), 5)} for f, v in zip(feature_names, vals)],
                key=lambda x: x["importance"], reverse=True,
            )[:20]
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            importances = sorted(
                [{"feature": f, "importance": round(float(v), 5)} for f, v in zip(feature_names, coef)],
                key=lambda x: x["importance"], reverse=True,
            )[:20]
        return importances
