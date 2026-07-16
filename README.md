<div align="center">

# DataSage AI

**Intelligent Data Science Workflow Engine**

DataSage AI is an end-to-end automated machine learning platform that eliminates the manual overhead of data science workflows. Upload any dataset, and the platform automatically profiles it, detects the right ML task, benchmarks every applicable model head-to-head, monitors data drift, and delivers plain-language AI explanations — all without writing a single line of code.

<!-- Deployment -->
[![Live Demo](https://img.shields.io/badge/Live_Demo-Railway-101514?style=for-the-badge&logo=railway&logoColor=5ed29c)](https://datasage-production-f1ea.up.railway.app)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

<!-- Backend -->
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-499848?style=for-the-badge&logo=gunicorn&logoColor=white)](https://www.uvicorn.org)

<!-- ML & Data -->
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-EA4335?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![pandas](https://img.shields.io/badge/pandas-DataFrames-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-Statistics-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![statsmodels](https://img.shields.io/badge/statsmodels-TimeSeries-4B8BBE?style=for-the-badge&logo=python&logoColor=white)](https://www.statsmodels.org)

<!-- AI & Visualization -->
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-FF6B35?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![Plotly](https://img.shields.io/badge/Plotly.js-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/javascript)

</div>

---

## Live Deployment

> **Try it now:** https://datasage-c97d.onrender.com/

---

## Screenshots

All workflow screenshots are stored in the [`screenshots/`](https://github.com/HEMANTH-A-7/DataSage/tree/main/screenshots) folder.

| Step | Preview |
|---|---|
| Landing Page | [View](https://github.com/HEMANTH-A-7/DataSage/blob/main/screenshots/01_landing_page.png) |
| Dataset Upload | [View](https://github.com/HEMANTH-A-7/DataSage/blob/main/screenshots/02_upload_section.png) |
| Data Profiling & EDA | [View](https://github.com/HEMANTH-A-7/DataSage/blob/main/screenshots/03_data_profiling.png) |
| Task Detection | [View](https://github.com/HEMANTH-A-7/DataSage/blob/main/screenshots/04_task_detection.png) |
| Model Recommendations | [View](https://github.com/HEMANTH-A-7/DataSage/blob/main/screenshots/05_model_recommendations.png) |
| Model Arena — Leaderboard & Radar | [View](https://github.com/HEMANTH-A-7/DataSage/blob/main/screenshots/06_model_arena.png) |
| Results & AI Explanations | [View](https://github.com/HEMANTH-A-7/DataSage/blob/main/screenshots/07_results_explanation.png) |

---

## Overview

Modern data science projects typically require a data engineer to clean data, a data scientist to select and tune models, and an ML engineer to productionize the pipeline. DataSage AI compresses this entire workflow into a single browser session. It is designed for teams and individuals who need fast, reliable, and explainable machine learning results without the setup burden.

The platform handles four classes of ML problems out of the box:

- **Regression** — predicting continuous numerical targets (sales forecasts, price estimation, energy output)
- **Classification** — sorting records into categories (customer churn, fraud detection, disease prediction)
- **Clustering** — discovering hidden structure in unlabeled data (customer segmentation, anomaly grouping)
- **Time Series** — forecasting sequential data with trend and seasonality (demand planning, stock indices)

### How It Works

1. **Ingest & Profile** — Upload CSV, Excel, or JSON. The platform infers data types, computes distribution statistics, flags missing values, detects skew and outliers, and builds a full correlation matrix.
2. **Detect the ML Task** — A hybrid engine first applies deterministic heuristic rules (target column cardinality, data types, column name patterns), then queries the Groq Llama 3.3 language model for a second-opinion classification. The two signals are combined into a high-confidence task label.
3. **Recommend Models** — A dataset-aware scoring function ranks the 18 supported algorithms by suitability, factoring in dataset size, feature count, task type, and linearity signals. Each recommendation includes pros, cons, complexity rating, and a plain-language explanation of when to use it.
4. **Train & Evaluate** — The selected model is trained on an 80/20 stratified split with StandardScaler preprocessing for linear and kernel-based models. Five-fold cross-validation is run automatically and reported alongside test-set metrics.
5. **Run the Model Arena** — Every applicable model for the detected task is trained simultaneously on the same data split. Results are ranked on a primary metric leaderboard, visualised as a bar chart, and compared across multiple axes on an interactive radar chart.
6. **Detect Data Drift** — The platform compares the uploaded dataset against a reference distribution using the Kolmogorov-Smirnov test, Population Stability Index, and Jensen-Shannon divergence. Each feature is graded individually and an overall drift severity (none / warning / critical) is returned.
7. **Explain with AI** — The Groq Llama 3.3 model generates a structured natural language report covering the summary, performance analysis, key drivers, and actionable recommendations — adapted to the user's selected expertise level (Beginner / Intermediate / Expert).

---

## Use Cases

### Rapid Proof-of-Concept for Business Teams

Business analysts and product managers often need a quick answer to "can we predict X from this data?" before committing engineering resources. DataSage AI delivers a full benchmarked answer in under five minutes — complete with feature importances, accuracy metrics, and a plain-language summary — without requiring any Python knowledge.

### Academic and Research Prototyping

Researchers who need a baseline model comparison across multiple algorithms can use the Model Arena to benchmark every relevant model on their dataset in a single click. The radar chart comparison across metrics like Accuracy, F1, Precision, and Recall provides a publication-ready multi-dimensional view of model behaviour.

### Customer Segmentation and Market Analysis

Upload a customer behavioural dataset (without a target label) and DataSage detects the clustering task automatically, runs K-Means, DBSCAN, and Agglomerative Clustering, evaluates each with Silhouette Score, Calinski-Harabasz, and Davies-Bouldin indices, and visualises cluster structures in a PCA 2D scatter plot alongside a cluster size distribution chart.

### Predictive Maintenance and Industrial Monitoring

Upload sensor or equipment data and let the platform detect whether the task is regression (predicting remaining useful life) or classification (fault vs. no-fault). The data drift monitor is particularly useful here — it can flag when incoming production data starts deviating from the training distribution, an early warning sign for model degradation.

### Sales and Demand Forecasting

Upload time-series sales records and DataSage automatically detects the temporal task, compares ARIMA, Exponential Smoothing (Holt-Winters), and a lag-feature Random Forest, and provides AI commentary on which model best captures the trend and seasonal patterns in your data.

### Educational and Teaching Tool

The three expertise levels (Beginner, Intermediate, Expert) make DataSage a strong teaching aid. Instructors can demonstrate the full ML workflow on real datasets in a live lecture, while students at different levels receive explanations calibrated to their background knowledge.

---

## Why DataSage AI Over Alternatives

The AutoML landscape is dominated by a handful of established tools. DataSage AI occupies a distinct niche in that space.

| Capability | DataSage AI | Google AutoML | H2O AutoML | TPOT | Weka |
|---|---|---|---|---|---|
| Zero-code browser interface | Yes | Yes | Partial | No | Partial |
| Instant local deployment | Yes | No (cloud only) | Yes | Yes | Yes |
| Hybrid LLM task detection | Yes | No | No | No | No |
| Natural language result explanations | Yes (Groq LLM) | Limited | No | No | No |
| Expertise-level adapted output | Yes | No | No | No | No |
| Real-time data drift monitoring | Yes | Limited | Yes | No | No |
| Multi-axis radar chart comparison | Yes | No | No | No | No |
| Open source & self-hostable | Yes | No | Yes | Yes | Yes |
| Setup time | Under 2 minutes | Hours (GCP config) | 10-30 minutes | Complex | Moderate |

**Key differentiators:**

- **LLM-assisted task detection.** Unlike pure AutoML tools that rely solely on heuristics, DataSage combines deterministic rules with a large language model query, reducing task misclassification on ambiguous datasets.
- **Explanation depth.** Most AutoML tools return a number. DataSage returns a structured narrative — what the result means, what drives it, what to do next — tailored to the user's background.
- **Expertise levels.** The same result can be explained in plain English to a business stakeholder or in statistical detail to a data scientist, with a single toggle.
- **Instant deployment.** No cloud account, no API keys required to start. Clone, install, run. The Groq API key is optional and only needed for the AI explanation features.
- **Model Arena transparency.** Rather than returning a single "best" model, DataSage shows the full ranked leaderboard so users can make informed trade-offs between accuracy, training speed, and interpretability.

---

## Key Features

| Feature | Description |
|---|---|
| Multi-Format Ingestion | CSV, Excel (.xlsx/.xls), and JSON files up to 50 MB |
| Automated EDA | Schema detection, missing values, distribution profiling, correlation analysis |
| Hybrid Task Detection | Heuristic rules combined with Groq LLM for high-confidence task classification |
| 18+ ML Models | Covers regression, classification, clustering, and time-series forecasting |
| Model Arena | Automated head-to-head benchmarking with ranked leaderboard and radar charts |
| Data Drift Detection | KS-test, PSI, and JS-divergence across all features with severity grading |
| Feature Engineering | Automated suggestions for log transforms, interactions, and encodings |
| AI Explanations | Context-aware, expertise-adapted explanations powered by Groq Llama 3.3 |
| Interactive Q&A | Ask follow-up questions about your results in natural language |
| Demo Datasets | One-click ABB Motor Data, House Prices, Customer Churn, Segments, and Sales Forecast |

---

## Architecture

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
│   ├── explainer.py          # Groq-powered AI explanations
│   └── visualizer.py         # Plotly chart generation
├── static/
│   ├── index.html            # Single-page application
│   ├── style.css             # CodeNest dark design system
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
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML / Data | scikit-learn, XGBoost, statsmodels, pandas, NumPy, SciPy |
| AI / LLM | Groq Llama 3.3 70B |
| Visualization | Plotly.js |
| Frontend | Vanilla HTML, CSS, JavaScript — CodeNest dark design system |
| Deployment | Docker, Railway |

---

## Getting Started

### Prerequisites

- Python 3.11+
- (Optional) [Groq API Key](https://console.groq.com/keys) for AI explanations

### Local Setup

```bash
# Clone the repository
git clone https://github.com/HEMANTH-A-7/DataSage.git
cd DataSage

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Generate demo datasets
python generate_demo_data.py

# Set environment variable (optional — enables AI explanations)
export GROQ_API_KEY="your_key_here"       # Linux / Mac
set GROQ_API_KEY=your_key_here            # Windows

# Run the application
python app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Docker

```bash
docker-compose up --build
```

---

## API Endpoints

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

Full interactive API documentation is available at `/docs` (Swagger UI).

---

## Supported Models

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

## License

This project is intended for educational and demonstration purposes.

---

<div align="center">

**Built for intelligent, automated data science.**

</div>
