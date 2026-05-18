"""
ExplainerEngine — LLM-powered natural language explanations for ML results.
Adapts language complexity to user expertise level.
"""
import asyncio
import json
import re
from typing import Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

EXPERTISE_PROMPTS = {
    "beginner": "Explain as if to someone with no programming or statistics background. Use simple analogies and avoid jargon. Keep it friendly and encouraging.",
    "intermediate": "Explain to a developer or analyst who understands basic ML concepts. Be precise but not overly academic.",
    "expert": "Explain to a senior data scientist. Include statistical nuances, potential issues (overfitting, data leakage, distribution shift), and production considerations.",
}


class ExplainerEngine:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.model = None
        if GEMINI_AVAILABLE and api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def explain(self, results: dict, task: dict, expertise: str) -> dict:
        """Generate a comprehensive explanation of workflow results."""
        if not self.model:
            return self._fallback_explanation(results, task, expertise)

        expertise_style = EXPERTISE_PROMPTS.get(expertise, EXPERTISE_PROMPTS["intermediate"])
        task_type = task.get("task_type", "unknown")
        model_name = results.get("model_name", "Unknown Model")
        metrics = results.get("metrics", {})
        feature_importance = results.get("feature_importance", [])[:5]

        prompt = f"""You are DataSage AI, an expert data science assistant.

Task type: {task_type}
Model used: {model_name}
Metrics: {json.dumps(metrics, indent=2)}
Top features: {json.dumps(feature_importance, indent=2)}
Task reasoning: {task.get("reasoning", "")}

Audience: {expertise_style}

Generate a structured explanation with:
1. **Summary** (2-3 sentences): What was done and the key result
2. **Performance Analysis**: What the metrics mean, whether they're good/bad
3. **Key Drivers**: What features matter most and why (if available)
4. **Recommendations**: Next steps to improve the model
5. **Cautions**: Any concerns about the data or results

Respond ONLY with valid JSON:
{{
  "summary": "...",
  "performance_analysis": "...",
  "key_drivers": "...",
  "recommendations": "...",
  "cautions": "..."
}}"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.model.generate_content, prompt)
            text = response.text.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return self._fallback_explanation(results, task, expertise)

    async def answer_question(
        self, question: str, results: dict, task: dict, eda: dict, expertise: str
    ) -> str:
        """Answer a follow-up question about the analysis."""
        if not self.model:
            return "LLM service unavailable. Please configure GEMINI_API_KEY."

        expertise_style = EXPERTISE_PROMPTS.get(expertise, EXPERTISE_PROMPTS["intermediate"])
        context = {
            "task": task.get("task_type"),
            "model": results.get("model_name"),
            "metrics": results.get("metrics", {}),
            "data_quality": eda.get("data_quality_score"),
        }

        prompt = f"""You are DataSage AI. A user is asking about their ML analysis.

Context:
{json.dumps(context, indent=2)}

Question: {question}

Audience style: {expertise_style}

Answer clearly, helpfully, and concisely (2-5 sentences max). Cite specific numbers from the context where relevant."""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.model.generate_content, prompt)
            return response.text.strip()
        except Exception as e:
            return f"Could not generate answer: {str(e)}"

    def _fallback_explanation(self, results: dict, task: dict, expertise: str) -> dict:
        """Rule-based fallback when LLM is unavailable."""
        metrics = results.get("metrics", {})
        task_type = task.get("task_type", "analysis")
        model_name = results.get("model_name", "the model")

        summary = f"Completed {task_type} using {model_name}."

        perf = ""
        if "r2" in metrics:
            r2 = metrics["r2"]
            perf = f"R² = {r2}. " + ("Excellent fit." if r2 > 0.9 else "Good fit." if r2 > 0.7 else "Moderate fit — consider feature engineering.")
        elif "accuracy" in metrics:
            acc = metrics["accuracy"]
            perf = f"Accuracy = {acc*100:.1f}%. " + ("Excellent." if acc > 0.95 else "Good." if acc > 0.80 else "Consider tuning or more data.")
        elif "silhouette_score" in metrics:
            sil = metrics.get("silhouette_score", 0)
            perf = f"Silhouette score = {sil}. " + ("Good cluster separation." if sil and sil > 0.5 else "Moderate clustering — try different k.")

        return {
            "summary": summary,
            "performance_analysis": perf or "See metrics above.",
            "key_drivers": "Check feature importance chart for top predictors.",
            "recommendations": "Consider cross-validation and hyperparameter tuning for better results.",
            "cautions": "Always validate results on unseen data before production deployment.",
        }
