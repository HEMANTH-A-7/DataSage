"""
DataIngestor — Loads CSV/Excel/JSON datasets and extracts schema metadata.
"""
import io
import json
import numpy as np
import pandas as pd
from typing import Tuple


class DataIngestor:
    """Handles dataset loading from raw bytes and extracts schema information."""

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}

    def load(self, content: bytes, filename: str) -> Tuple[pd.DataFrame, dict]:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(content))
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(content))
        elif ext == ".json":
            df = pd.read_json(io.BytesIO(content))
        else:
            raise ValueError(f"Cannot parse {ext}")

        df = self._clean(df)
        schema = self._extract_schema(df)
        return df, schema

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: strip whitespace from column names, drop all-null rows."""
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all")
        return df

    def _extract_schema(self, df: pd.DataFrame) -> dict:
        """Build a rich schema dictionary describing each column."""
        columns = {}
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            n_unique = int(series.nunique())
            n_missing = int(series.isna().sum())
            missing_pct = round(n_missing / len(df) * 100, 2) if len(df) > 0 else 0.0

            col_type = self._infer_column_type(series, dtype, n_unique, len(df))

            columns[col] = {
                "dtype": dtype,
                "col_type": col_type,
                "n_unique": n_unique,
                "n_missing": n_missing,
                "missing_pct": missing_pct,
                "sample_values": series.dropna().head(5).tolist(),
            }

            if col_type == "numeric":
                columns[col].update({
                    "min": _safe_float(series.min()),
                    "max": _safe_float(series.max()),
                    "mean": _safe_float(series.mean()),
                    "std": _safe_float(series.std()),
                })

        # Heuristics for potential target columns
        potential_targets = [
            col for col, meta in columns.items()
            if meta["col_type"] in ("numeric", "categorical")
               and meta["n_unique"] > 1
        ]

        # Detect datetime-like columns
        datetime_cols = [c for c, m in columns.items() if m["col_type"] == "datetime"]

        return {
            "columns": columns,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "potential_targets": potential_targets,
            "datetime_cols": datetime_cols,
            "has_datetime": len(datetime_cols) > 0,
        }

    def _infer_column_type(self, series, dtype: str, n_unique: int, n_rows: int) -> str:
        if "int" in dtype or "float" in dtype:
            if n_unique <= 20 and n_unique / max(n_rows, 1) < 0.05:
                return "categorical"
            return "numeric"
        if "datetime" in dtype or "timestamp" in dtype:
            return "datetime"
        # Try parsing strings as datetime
        if "object" in dtype:
            sample = series.dropna().head(20).astype(str)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                return "datetime"
            except Exception:
                pass
            if n_unique <= 50:
                return "categorical"
            return "text"
        if "bool" in dtype:
            return "categorical"
        return "other"


def _safe_float(val) -> float:
    try:
        f = float(val)
        return round(f, 4) if not (np.isnan(f) or np.isinf(f)) else 0.0
    except Exception:
        return 0.0
