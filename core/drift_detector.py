"""
DriftDetector — Statistical drift detection between train/test distributions.
Uses KS-test, PSI, and Jensen-Shannon divergence to flag distribution shifts.
Production-grade feature that demonstrates ML monitoring maturity.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import List, Dict, Optional


class DriftDetector:
    """Detect data drift between training and test distributions."""

    SEVERITY_THRESHOLDS = {
        "ks_pvalue": {"warning": 0.05, "critical": 0.01},
        "psi": {"warning": 0.1, "critical": 0.25},
        "js_divergence": {"warning": 0.05, "critical": 0.15},
    }

    def detect(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
    ) -> Dict:
        """Run comprehensive drift analysis across all features."""
        results = {
            "overall_drift_detected": False,
            "drift_severity": "none",
            "features_drifted": 0,
            "total_features": len(feature_names),
            "feature_reports": [],
            "summary": "",
        }

        drift_count = 0
        critical_count = 0
        feature_reports = []

        for i, feat in enumerate(feature_names):
            if i >= X_train.shape[1] or i >= X_test.shape[1]:
                break

            train_col = X_train[:, i]
            test_col = X_test[:, i]

            # Remove NaNs
            train_col = train_col[~np.isnan(train_col)]
            test_col = test_col[~np.isnan(test_col)]

            if len(train_col) < 5 or len(test_col) < 5:
                continue

            report = self._analyze_feature(feat, train_col, test_col)
            feature_reports.append(report)

            if report["drift_detected"]:
                drift_count += 1
            if report["severity"] == "critical":
                critical_count += 1

        # Overall assessment
        drift_ratio = drift_count / max(len(feature_names), 1)
        if critical_count > 0 or drift_ratio > 0.3:
            severity = "critical"
        elif drift_count > 0:
            severity = "warning"
        else:
            severity = "none"

        results["features_drifted"] = drift_count
        results["overall_drift_detected"] = drift_count > 0
        results["drift_severity"] = severity
        results["feature_reports"] = feature_reports
        results["summary"] = self._generate_summary(
            drift_count, critical_count, len(feature_names), severity
        )

        return results

    def _analyze_feature(
        self, name: str, train: np.ndarray, test: np.ndarray
    ) -> Dict:
        """Analyze drift for a single feature using multiple tests."""
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(train, test)

        # Population Stability Index (PSI)
        psi = self._compute_psi(train, test)

        # Jensen-Shannon divergence
        js_div = self._compute_js_divergence(train, test)

        # Determine drift
        ks_drift = ks_pvalue < self.SEVERITY_THRESHOLDS["ks_pvalue"]["warning"]
        psi_drift = psi > self.SEVERITY_THRESHOLDS["psi"]["warning"]
        js_drift = js_div > self.SEVERITY_THRESHOLDS["js_divergence"]["warning"]

        drift_detected = sum([ks_drift, psi_drift, js_drift]) >= 2  # Majority vote

        # Severity
        if drift_detected:
            ks_critical = ks_pvalue < self.SEVERITY_THRESHOLDS["ks_pvalue"]["critical"]
            psi_critical = psi > self.SEVERITY_THRESHOLDS["psi"]["critical"]
            severity = "critical" if (ks_critical or psi_critical) else "warning"
        else:
            severity = "none"

        return {
            "feature": name,
            "drift_detected": drift_detected,
            "severity": severity,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "psi": round(float(psi), 4),
            "js_divergence": round(float(js_div), 4),
            "train_mean": round(float(np.mean(train)), 4),
            "test_mean": round(float(np.mean(test)), 4),
            "train_std": round(float(np.std(train)), 4),
            "test_std": round(float(np.std(test)), 4),
        }

    def _compute_psi(
        self, train: np.ndarray, test: np.ndarray, bins: int = 10
    ) -> float:
        """Compute Population Stability Index."""
        try:
            breakpoints = np.percentile(train, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)
            if len(breakpoints) < 3:
                return 0.0

            train_counts = np.histogram(train, bins=breakpoints)[0]
            test_counts = np.histogram(test, bins=breakpoints)[0]

            # Avoid zero-division
            train_pct = (train_counts + 1) / (len(train) + len(breakpoints) - 1)
            test_pct = (test_counts + 1) / (len(test) + len(breakpoints) - 1)

            psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
            return float(psi)
        except Exception:
            return 0.0

    def _compute_js_divergence(
        self, train: np.ndarray, test: np.ndarray, bins: int = 20
    ) -> float:
        """Compute Jensen-Shannon divergence between distributions."""
        try:
            all_data = np.concatenate([train, test])
            edges = np.histogram_bin_edges(all_data, bins=bins)

            p = np.histogram(train, bins=edges, density=True)[0] + 1e-10
            q = np.histogram(test, bins=edges, density=True)[0] + 1e-10

            p = p / p.sum()
            q = q / q.sum()

            return float(jensenshannon(p, q))
        except Exception:
            return 0.0

    def _generate_summary(
        self,
        drift_count: int,
        critical_count: int,
        total: int,
        severity: str,
    ) -> str:
        if severity == "none":
            return (
                "No significant distribution drift detected between train and test sets. "
                "Model predictions should be reliable on this data split."
            )
        elif severity == "warning":
            return (
                f"{drift_count}/{total} features show distribution drift (KS test + PSI). "
                "Consider investigating these features — they may reduce model generalizability."
            )
        else:
            return (
                f"⚠ CRITICAL: {critical_count} features show severe distribution shift. "
                f"{drift_count}/{total} features drifted total. "
                "Model performance on test data may be unreliable. "
                "Recommend re-sampling or stratified splitting."
            )
