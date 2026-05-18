"""
Tests for DataSage AI core modules.
Run with: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ingestor import DataIngestor
from core.eda import EDAEngine
from core.task_detector import TaskDetector
from core.model_recommender import ModelRecommender
from core.workflow_runner import WorkflowRunner
from core.drift_detector import DriftDetector
from core.feature_engineer import FeatureEngineer
from core.arena import ModelArena


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def regression_df():
    rng = np.random.default_rng(42)
    n = 200
    X1 = rng.uniform(100, 5000, n)
    X2 = rng.integers(1, 6, n).astype(float)
    y = X1 * 2 + X2 * 1000 + rng.normal(0, 500, n)
    return pd.DataFrame({"feature_a": X1, "feature_b": X2, "target": y})


@pytest.fixture
def classification_df():
    rng = np.random.default_rng(42)
    n = 200
    X1 = rng.uniform(0, 10, n)
    X2 = rng.uniform(0, 10, n)
    y = (X1 + X2 > 10).astype(str)
    y[y == "True"] = "Yes"
    y[y == "False"] = "No"
    return pd.DataFrame({"feature_a": X1, "feature_b": X2, "label": y})


@pytest.fixture
def csv_bytes(regression_df):
    return regression_df.to_csv(index=False).encode()


# ── Ingestor ────────────────────────────────────────────────────────────────
class TestDataIngestor:
    def test_load_csv(self, csv_bytes):
        ingestor = DataIngestor()
        df, schema = ingestor.load(csv_bytes, "test.csv")
        assert len(df) == 200
        assert "feature_a" in schema["columns"]

    def test_schema_has_types(self, csv_bytes):
        ingestor = DataIngestor()
        _, schema = ingestor.load(csv_bytes, "test.csv")
        for col, meta in schema["columns"].items():
            assert "col_type" in meta

    def test_unsupported_raises(self):
        ingestor = DataIngestor()
        with pytest.raises(ValueError):
            ingestor.load(b"data", "file.parquet")


# ── EDA Engine ───────────────────────────────────────────────────────────────
class TestEDAEngine:
    def test_analyze_returns_quality_score(self, regression_df, csv_bytes):
        ingestor = DataIngestor()
        df, schema = ingestor.load(csv_bytes, "test.csv")
        eda = EDAEngine()
        report = eda.analyze(df, schema)
        assert 0 <= report["data_quality_score"] <= 100
        assert "summary" in report
        assert "correlations" in report
        assert "outliers" in report


# ── Task Detector ─────────────────────────────────────────────────────────────
class TestTaskDetector:
    def test_regression_detected(self, regression_df, csv_bytes):
        ingestor = DataIngestor()
        df, schema = ingestor.load(csv_bytes, "test.csv")
        detector = TaskDetector()
        import asyncio
        result = asyncio.run(detector.detect(df, schema, "target", "", "intermediate"))
        assert result["task_type"] == "regression"

    def test_clustering_when_no_target(self, regression_df, csv_bytes):
        ingestor = DataIngestor()
        df, schema = ingestor.load(csv_bytes, "test.csv")
        detector = TaskDetector()
        import asyncio
        result = asyncio.run(detector.detect(df, schema, "", "", "intermediate"))
        assert result["task_type"] == "clustering"


# ── Workflow Runner ───────────────────────────────────────────────────────────
class TestWorkflowRunner:
    def test_regression_pipeline(self, regression_df):
        runner = WorkflowRunner()
        result = runner.run(regression_df, "target", "regression", "linear_regression")
        assert "metrics" in result
        assert "r2" in result["metrics"]
        assert -1 <= result["metrics"]["r2"] <= 1

    def test_classification_pipeline(self, classification_df):
        runner = WorkflowRunner()
        result = runner.run(classification_df, "label", "classification", "logistic_regression")
        assert "accuracy" in result["metrics"]
        assert 0 <= result["metrics"]["accuracy"] <= 1

    def test_clustering_pipeline(self, regression_df):
        runner = WorkflowRunner()
        result = runner.run(regression_df, "", "clustering", "kmeans")
        assert "n_clusters_found" in result["metrics"]


# ── Drift Detector ─────────────────────────────────────────────────────────────
class TestDriftDetector:
    def test_no_drift_same_distribution(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, (200, 3))
        X_test = rng.normal(0, 1, (50, 3))
        detector = DriftDetector()
        report = detector.detect(X_train, X_test, ["f1", "f2", "f3"])
        assert "drift_severity" in report
        assert report["total_features"] == 3

    def test_critical_drift_different_distribution(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, (200, 2))
        X_test = rng.normal(10, 1, (50, 2))  # Shifted by 10 std
        detector = DriftDetector()
        report = detector.detect(X_train, X_test, ["f1", "f2"])
        assert report["overall_drift_detected"] is True


# ── Feature Engineer ──────────────────────────────────────────────────────────
class TestFeatureEngineer:
    def test_suggests_log_for_skewed(self, csv_bytes):
        ingestor = DataIngestor()
        df, schema = ingestor.load(csv_bytes, "test.csv")
        eda = EDAEngine()
        report = eda.analyze(df, schema)
        fe = FeatureEngineer()
        suggestions = fe.suggest(df, schema, report)
        assert isinstance(suggestions, list)
        for s in suggestions:
            assert "feature" in s
            assert "transform" in s
            assert "priority" in s


# ── Model Recommender ─────────────────────────────────────────────────────────
class TestModelRecommender:
    def test_recommends_for_regression(self, csv_bytes):
        ingestor = DataIngestor()
        df, schema = ingestor.load(csv_bytes, "test.csv")
        eda = EDAEngine().analyze(df, schema)
        rec = ModelRecommender()
        recs = rec.recommend("regression", schema, eda)
        assert len(recs) > 0
        assert recs[0].get("recommended") is True
        for r in recs:
            assert 0 <= r["score"] <= 100
