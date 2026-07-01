"""
TaskDetector — Hybrid heuristic + LLM task type detection.
Determines: regression | classification | clustering | time_series
"""
import json
import re
import asyncio
import numpy as np
import pandas as pd
from typing import Optional

import httpx

class TaskDetector:
    TASK_TYPES = ["regression", "classification", "clustering", "time_series"]

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    async def detect(
        self,
        df: pd.DataFrame,
        schema: dict,
        target_column: str,
        user_hint: str,
        expertise: str,
    ) -> dict:
        heuristic = self._heuristic_detect(df, schema, target_column)
        llm_result = None

        if self.api_key and (user_hint or target_column):
            try:
                llm_result = await self._llm_detect(df, schema, target_column, user_hint, expertise)
            except Exception:
                pass

        # Merge: LLM overrides heuristic if confident
        if llm_result and llm_result.get("confidence", 0) > 0.7:
            task_type = llm_result["task_type"]
            reasoning = llm_result.get("reasoning", "")
            confidence = llm_result["confidence"]
            source = "llm"
        else:
            task_type = heuristic["task_type"]
            reasoning = heuristic["reasoning"]
            confidence = heuristic["confidence"]
            source = "heuristic"

        # Sub-task info
        subtask_info = self._subtask_details(task_type, df, schema, target_column)

        return {
            "task_type": task_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "source": source,
            "subtask_info": subtask_info,
            "target_column": target_column,
            "heuristic_result": heuristic,
            "llm_result": llm_result,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _heuristic_detect(self, df: pd.DataFrame, schema: dict, target_column: str) -> dict:
        cols = schema["columns"]
        has_datetime = schema.get("has_datetime", False)

        # Time series: has datetime + numeric target
        if has_datetime and target_column and cols.get(target_column, {}).get("col_type") == "numeric":
            return {
                "task_type": "time_series",
                "confidence": 0.85,
                "reasoning": "Dataset has datetime column(s) and a numeric target, suggesting time-series forecasting.",
            }

        if not target_column:
            return {
                "task_type": "clustering",
                "confidence": 0.75,
                "reasoning": "No target column specified — unsupervised clustering is the natural approach.",
            }

        target_meta = cols.get(target_column, {})
        col_type = target_meta.get("col_type", "numeric")
        n_unique = target_meta.get("n_unique", 10)
        n_rows = schema.get("n_rows", 1)

        if col_type == "categorical" or (col_type == "numeric" and n_unique <= 20):
            return {
                "task_type": "classification",
                "confidence": 0.88,
                "reasoning": f"Target '{target_column}' has {n_unique} unique values — classification task.",
            }

        if col_type == "numeric":
            return {
                "task_type": "regression",
                "confidence": 0.90,
                "reasoning": f"Target '{target_column}' is continuous numeric — regression task.",
            }

        return {
            "task_type": "classification",
            "confidence": 0.60,
            "reasoning": "Fallback to classification based on target column analysis.",
        }

    async def _llm_detect(
        self,
        df: pd.DataFrame,
        schema: dict,
        target_column: str,
        user_hint: str,
        expertise: str,
    ) -> Optional[dict]:
        col_summary = "\n".join(
            f"  - {col}: {meta['col_type']}, {meta['n_unique']} unique, {meta['missing_pct']}% missing"
            for col, meta in schema["columns"].items()
        )
        sample_rows = df.head(3).to_dict(orient="records")

        system_prompt = "You are a senior data scientist. Analyze the dataset metadata and determine the ML task type. Respond ONLY with valid JSON."
        prompt = f"""Dataset columns:
{col_summary}

Target column: {target_column or "None specified"}
User's problem description: {user_hint or "Not provided"}
Sample rows: {json.dumps(sample_rows, default=str)}

Respond ONLY with valid JSON in this format:
{{
  "task_type": "<regression|classification|clustering|time_series>",
  "confidence": <0.0-1.0>,
  "reasoning": "<2-3 sentence explanation>"
}}"""

        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                text = result["choices"][0]["message"]["content"].strip()
                
            # Extract JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return None

    def _subtask_details(self, task_type: str, df: pd.DataFrame, schema: dict, target: str) -> dict:
        if task_type == "classification":
            n_classes = int(df[target].nunique()) if target and target in df.columns else 2
            return {
                "n_classes": n_classes,
                "is_binary": n_classes == 2,
                "class_distribution": df[target].value_counts(normalize=True).round(4).to_dict() if target and target in df.columns else {},
            }
        if task_type == "regression":
            if target and target in df.columns:
                s = df[target].dropna()
                return {
                    "target_range": [float(s.min()), float(s.max())],
                    "target_mean": float(s.mean()),
                    "target_std": float(s.std()),
                }
        if task_type == "clustering":
            return {"suggested_k": min(max(2, int(len(df) ** 0.5 // 5)), 10)}
        if task_type == "time_series":
            return {"datetime_cols": schema.get("datetime_cols", []), "freq_hint": "inferred"}
        return {}
