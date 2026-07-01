"""
DataSage AI — FastAPI Application Entry Point
Industry-grade intelligent data science workflow engine.
Version 2.0 — Hackathon Edition with Arena, Drift Detection, Feature Engineering.
"""
import os
import uuid
import json
import logging
import asyncio
import time
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from dotenv import load_dotenv

from core.ingestor import DataIngestor
from core.eda import EDAEngine
from core.task_detector import TaskDetector
from core.model_recommender import ModelRecommender
from core.workflow_runner import WorkflowRunner
from core.explainer import ExplainerEngine
from core.visualizer import VisualizationEngine
from core.arena import ModelArena
from core.drift_detector import DriftDetector
from core.feature_engineer import FeatureEngineer

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("datasage")

# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class DetectTaskRequest(BaseModel):
    session_id: str
    target_column: str = ""
    user_hint: str = ""

class RecommendRequest(BaseModel):
    session_id: str

class RunWorkflowRequest(BaseModel):
    session_id: str
    model_key: str

class AskRequest(BaseModel):
    session_id: str
    question: str

class ArenaRequest(BaseModel):
    session_id: str

class DriftRequest(BaseModel):
    session_id: str

class FeatureSuggestRequest(BaseModel):
    session_id: str

# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataSage AI",
    description="Intelligent Data Science Workflow Engine — Hackathon Edition",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

demo_dir = Path(__file__).parent / "demo_data"
demo_dir.mkdir(exist_ok=True)
app.mount("/demo_data", StaticFiles(directory=str(demo_dir)), name="demo_data")

# ── Session Store (with TTL) ───────────────────────────────────────────────────
sessions: dict = {}
SESSION_TTL_MINUTES = 60


def _get_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    sess = sessions[session_id]
    # Refresh TTL on access
    sess["last_accessed"] = datetime.utcnow()
    return sess


def _cleanup_sessions():
    """Remove sessions older than TTL."""
    cutoff = datetime.utcnow() - timedelta(minutes=SESSION_TTL_MINUTES)
    expired = [sid for sid, s in sessions.items() if s.get("last_accessed", datetime.utcnow()) < cutoff]
    for sid in expired:
        del sessions[sid]
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")


# ── Request Logging Middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed}ms)")
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    expertise: str = Form(default="intermediate"),
):
    """Upload a dataset and run automated EDA."""
    _cleanup_sessions()
    session_id = str(uuid.uuid4())
    logger.info(f"Upload: {file.filename} | expertise={expertise}")

    try:
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Max 50 MB.")

        filename = file.filename or "dataset.csv"
        ingestor = DataIngestor()
        df, schema = ingestor.load(content, filename)

        eda_engine = EDAEngine()
        eda_report = eda_engine.analyze(df, schema)

        # Feature engineering suggestions
        fe = FeatureEngineer()
        fe_suggestions = fe.suggest(df, schema, eda_report)

        sessions[session_id] = {
            "df_json": df.to_json(orient="split"),
            "schema": schema,
            "eda": eda_report,
            "expertise": expertise,
            "filename": filename,
            "fe_suggestions": fe_suggestions,
            "last_accessed": datetime.utcnow(),
        }

        logger.info(f"Session {session_id[:8]} created | rows={len(df)} cols={len(df.columns)}")

        return JSONResponse({
            "session_id": session_id,
            "schema": schema,
            "eda": eda_report,
            "rows": len(df),
            "columns": list(df.columns),
            "fe_suggestions": fe_suggestions[:8],  # Top 8 suggestions
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=400, detail=f"Dataset ingestion failed: {str(e)}")


@app.post("/api/detect-task")
async def detect_task(payload: DetectTaskRequest):
    """Detect the ML task type using heuristics + LLM."""
    session = _get_session(payload.session_id)
    df = pd.read_json(session["df_json"], orient="split")
    schema = session["schema"]

    detector = TaskDetector(api_key=os.getenv("GROQ_API_KEY", ""))
    detection = await detector.detect(
        df, schema, payload.target_column, payload.user_hint, session["expertise"]
    )

    session["task"] = detection
    session["target_column"] = payload.target_column
    logger.info(f"Task detected: {detection['task_type']} (confidence={detection['confidence']:.2f})")
    return JSONResponse(detection)


@app.post("/api/recommend-models")
async def recommend_models(payload: RecommendRequest):
    """Get ranked model recommendations for the detected task."""
    session = _get_session(payload.session_id)
    task = session.get("task", {})

    recommender = ModelRecommender()
    recommendations = recommender.recommend(
        task_type=task.get("task_type", "classification"),
        schema=session["schema"],
        eda=session["eda"],
    )
    session["recommendations"] = recommendations
    return JSONResponse({"recommendations": recommendations})


@app.post("/api/run-workflow")
async def run_workflow(payload: RunWorkflowRequest):
    """Execute the selected ML workflow and return results with drift analysis."""
    session = _get_session(payload.session_id)
    df = pd.read_json(session["df_json"], orient="split")
    target_column = session.get("target_column", "")
    task = session.get("task", {})
    task_type = task.get("task_type", "classification")

    runner = WorkflowRunner()
    results = runner.run(
        df=df,
        target_column=target_column,
        task_type=task_type,
        model_key=payload.model_key,
    )

    # Drift detection on train/test split (for supervised tasks)
    drift_report = None
    if task_type in ("regression", "classification") and results.get("feature_names"):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.impute import SimpleImputer
            import numpy as np

            feature_names = results["feature_names"]
            exclude = [target_column] if target_column else []
            X = df[[c for c in df.columns if c not in exclude]].copy()
            for col in X.select_dtypes(include=["object", "category"]).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            X = X.select_dtypes(include=[np.number])
            imputer = SimpleImputer(strategy="median")
            X_arr = imputer.fit_transform(X)
            y_dummy = np.zeros(len(X_arr))
            X_train, X_test = train_test_split(X_arr, test_size=0.2, random_state=42)
            detector = DriftDetector()
            drift_report = detector.detect(X_train, X_test, list(X.columns))
        except Exception as e:
            logger.warning(f"Drift detection failed: {e}")
            drift_report = None

    viz_engine = VisualizationEngine()
    charts = viz_engine.generate(df, results, task_type, target_column)

    explainer = ExplainerEngine(api_key=os.getenv("GROQ_API_KEY", ""))
    explanation = await explainer.explain(results, task, session["expertise"])

    session["results"] = results
    session["drift_report"] = drift_report

    return JSONResponse({
        "results": results,
        "charts": charts,
        "explanation": explanation,
        "drift_report": drift_report,
    })


@app.post("/api/arena")
async def run_arena(payload: ArenaRequest):
    """Run ALL models for the detected task and return a ranked leaderboard."""
    session = _get_session(payload.session_id)
    df = pd.read_json(session["df_json"], orient="split")
    target_column = session.get("target_column", "")
    task = session.get("task", {})
    task_type = task.get("task_type", "classification")

    logger.info(f"Arena: running all {task_type} models for session {payload.session_id[:8]}")

    arena = ModelArena()
    arena_results = arena.compete(df=df, target_column=target_column, task_type=task_type)

    session["arena_results"] = arena_results

    # Generate AI commentary on winner
    if arena_results.get("winner"):
        try:
            explainer = ExplainerEngine(api_key=os.getenv("GROQ_API_KEY", ""))
            winner_results = arena_results["full_results"].get(
                arena_results["winner"]["model_key"], {}
            )
            explanation = await explainer.explain(winner_results, task, session["expertise"])
            arena_results["winner_explanation"] = explanation
        except Exception:
            arena_results["winner_explanation"] = None

    return JSONResponse(arena_results)


@app.post("/api/feature-suggestions")
async def feature_suggestions(payload: FeatureSuggestRequest):
    """Return automated feature engineering suggestions for the dataset."""
    session = _get_session(payload.session_id)
    suggestions = session.get("fe_suggestions", [])
    return JSONResponse({"suggestions": suggestions})


@app.post("/api/ask")
async def ask_question(payload: AskRequest):
    """Answer follow-up questions about the analysis."""
    session = _get_session(payload.session_id)
    explainer = ExplainerEngine(api_key=os.getenv("GROQ_API_KEY", ""))
    answer = await explainer.answer_question(
        payload.question,
        session.get("results", {}),
        session.get("task", {}),
        session.get("eda", {}),
        session["expertise"],
    )
    return JSONResponse({"answer": answer})


@app.get("/api/health")
async def health():
    """Health check with dependency status."""
    groq_ok = bool(os.getenv("GROQ_API_KEY", ""))
    return {
        "status": "ok",
        "service": "DataSage AI",
        "version": "2.0.0",
        "active_sessions": len(sessions),
        "groq_configured": groq_ok,
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
