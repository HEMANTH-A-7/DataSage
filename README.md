---
title: DataSage AI
emoji: ⚡
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
---

<div align="center">

# ⚡ DataSage AI

**Intelligent Data Science Workflow Engine**

An end-to-end automated machine learning platform that ingests any dataset, detects the optimal task type, benchmarks multiple models in a competitive arena, monitors data drift, and delivers AI-powered explanations — all through a single, professional interface.

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face-yellow?style=for-the-badge)](https://hemanth021-datasage.hf.space)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

##  Live Deployment

> **Try it now:** [https://hemanth021-datasage.hf.space](https://hemanth021-datasage.hf.space)

---

##  Overview

DataSage AI eliminates the repetitive overhead of exploratory data analysis, model selection, and pipeline configuration. Users upload a dataset (CSV, Excel, or JSON), and the platform automatically:

1. **Ingests & profiles** the data — schema inference, missing value analysis, distribution statistics.
2. **Detects the ML task** — Hybrid heuristic + Gemini LLM approach to classify the problem as regression, classification, clustering, or time-series.
3. **Recommends models** — Ranks applicable algorithms by suitability based on dataset characteristics.
4. **Trains & evaluates** — Executes the full ML pipeline with cross-validation, feature importance, and metric computation.
5. **Runs the Model Arena** — Benchmarks *all* applicable models head-to-head and produces a ranked leaderboard with radar-chart comparisons.
6. **Detects data drift** — Statistical monitoring using KS-test, PSI, and Jensen-Shannon divergence to flag distribution shifts.
7. **Explains results with AI** — Gemini-powered natural language explanations tailored to the user's expertise level.

---

##  Key Features

| Feature | Description |
|---|---|
| **Multi-Format Ingestion** | CSV, Excel (.xlsx/.xls), and JSON files up to 50 MB |
| **Automated EDA** | Schema detection, missing values, distribution profiling, correlation analysis |
| **Hybrid Task Detection** | Heuristic rules + Gemini LLM for high-confidence task classification |
| **18+ ML Models** | Covers regression, classification, clustering, and time-series forecasting |
| **Model Arena** | Automated head-to-head benchmarking with ranked leaderboard and radar charts |
| **Data Drift Detection** | KS-test, PSI, and JS-divergence across all features with severity grading |
| **Feature Engineering** | Automated suggestions for transformations, interactions, and encodings |
| **AI Explanations** | Context-aware, expertise-adapted explanations powered by Gemini |
| **Interactive Q&A** | Ask follow-up questions about your results in natural language |
| **Demo Datasets** | One-click ABB Motor Data, House Prices, Customer Churn, Segments, and Sales Forecast |

---

##  Architecture

```
datasage/
├── app.py                    # FastAPI application entry point
├── core/
│   ├── ingestor.py           # Multi-format dataset ingestion
│   ├── eda.py                # Exploratory data analysis engine
│   ├── task_detector.py      # Hybrid heuristic + LLM task detection
│   ├── model_recommender.py  # Dataset-aware model ranking
│   ├── model_registry.py     # Centralized catalog of 18+ ML models
│   ├── workflow_runner.py    # End-to-end ML pipeline execution
│   ├── arena.py              # Multi-model competition engine
│   ├── drift_detector.py     # Statistical drift monitoring (KS, PSI, JS)
│   ├── feature_engineer.py   # Automated feature suggestions
│   ├── explainer.py          # Gemini-powered AI explanations
│   └── visualizer.py         # Plotly chart generation
├── static/
│   ├── index.html            # Single-page application
│   ├── style.css             # ABB-inspired design system
│   └── app.js                # Frontend logic & API integration
├── tests/
│   └── test_core.py          # Unit tests for core modules
├── demo_data/                # Pre-generated demo datasets
├── Dockerfile                # Production container image
├── docker-compose.yml        # Local orchestration
└── requirements.txt          # Python dependencies
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **ML / Data** | scikit-learn, XGBoost, statsmodels, pandas, NumPy, SciPy |
| **AI / LLM** | Google Gemini 1.5 Flash |
| **Visualization** | Plotly.js |
| **Frontend** | Vanilla HTML/CSS/JS with ABB-inspired design system |
| **Deployment** | Docker, Hugging Face Spaces |

---

## Getting Started

### Prerequisites

- Python 3.11+
- (Optional) [Gemini API Key](https://aistudio.google.com/apikey) for AI explanations

### Local Setup

```bash
# Clone the repository
git clone https://github.com/HEMANTH-A-7/DataSage.git
cd DataSage

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Generate demo datasets
python generate_demo_data.py

# Set environment variable (optional — enables AI explanations)
export GEMINI_API_KEY="your_key_here"       # Linux/Mac
set GEMINI_API_KEY=your_key_here            # Windows

# Run the application
python app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Docker

```bash
docker-compose up --build
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the web application |
| `POST` | `/api/upload` | Upload and profile a dataset |
| `POST` | `/api/detect-task` | Detect the ML task type |
| `POST` | `/api/recommend-models` | Get ranked model recommendations |
| `POST` | `/api/run-workflow` | Execute a selected ML pipeline |
| `POST` | `/api/arena` | Benchmark all models (Model Arena) |
| `POST` | `/api/feature-suggestions` | Get feature engineering suggestions |
| `POST` | `/api/ask` | Ask follow-up questions via AI |
| `GET` | `/api/health` | Health check with dependency status |

Full interactive API documentation available at `/docs` (Swagger UI).

---

##  Supported Models

<details>
<summary><strong>Regression (6 models)</strong></summary>

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Support Vector Regressor
</details>

<details>
<summary><strong>Classification (6 models)</strong></summary>

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- Support Vector Classifier
- Decision Tree
</details>

<details>
<summary><strong>Clustering (3 models)</strong></summary>

- K-Means
- DBSCAN
- Agglomerative Clustering
</details>

<details>
<summary><strong>Time Series (3 models)</strong></summary>

- ARIMA
- Exponential Smoothing (Holt-Winters)
- Random Forest (Lag Features)
</details>

---

##  License

This project is intended for educational and demonstration purposes.

---

<div align="center">

**Built for intelligent, automated data science.**

</div>
