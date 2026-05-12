# CreditAI — Fraud Detection & Credit Risk Analysis


Machine learning system for credit risk scoring and fraud detection, featuring supervised,
unsupervised and neural network models, a NLP sentiment module, and a production-ready
MLOps pipeline. Built as part of the Banco do Brasil Tech Roadmap — Big Tech edition.

---

## Demo

> 📸 Screenshots and demo GIF coming after Phase 6 (Serving & MLOps)

---

## Overview

CreditAI is structured around three parallel model tracks and one NLP module:

| Track | Approach | Algorithm |
|---|---|---|
| Supervised | Classification | Random Forest + XGBoost |
| Unsupervised | Anomaly detection | Isolation Forest + K-Means |
| Neural Network | Deep learning | PyTorch MLP |
| NLP | Sentiment analysis | BERT fine-tuned (BERTimbau) |

The system is served via **FastAPI**, tracked with **MLflow**, and — in the Big Tech phase —
extended with a feature store, drift monitoring, explainability and A/B testing.

---

## Tech Stack

### Data & ML
| Layer | Technology |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Supervised ML | `scikit-learn` (Random Forest, XGBoost) |
| Neural network | `PyTorch` (MLP with `nn.Module`) |
| NLP / Transformers | `HuggingFace Transformers` (BERT fine-tuning) |
| Statistics | `scipy` |

### MLOps & Serving
| Layer | Technology |
|---|---|
| Experiment tracking | `MLflow` (Model Registry) |
| API serving | `FastAPI` + `Pydantic` |
| Containerization | `Docker` (multi-stage build) |
| CI/CD | `GitHub Actions` |
| Explainability | `SHAP` / `LIME` |

### Big Tech Phase
| Layer | Technology |
|---|---|
| Feature store | `Feast` |
| Drift monitoring | `Evidently` / `Arize` |
| Observability | `Prometheus` + `Grafana` + `OpenTelemetry` |
| Load testing | `k6` (p99 < 50ms SLO) |
| Vector DB | `pgvector` (semantic retrieval via BERT embeddings) |
| Inter-service comm | `gRPC` |
| Fairness audit | `Fairlearn` |

---

## Project Structure

```
creditai/
├── notebooks/
│   ├── 01_eda_exploratory_analysis.ipynb
│   ├── 02_feature_engineering_baseline.ipynb
│   ├── 03_unsupervised_models.ipynb
│   ├── 04_pytorch_mlp.ipynb
│   └── 05_bert_nlp_sentiment.ipynb
├── src/
│   ├── features/          # Feature engineering pipeline
│   ├── models/            # Model definitions and training scripts
│   ├── serving/           # FastAPI app
│   └── monitoring/        # Drift detection and metrics
├── tests/
├── docs/
│   ├── adr/               # Architecture Decision Records
│   ├── runbook.md
│   └── design_doc.md
├── mlruns/                # MLflow tracking (gitignored)
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Implementation Roadmap

| Phase | Description | Key Technologies | Status |
|---|---|---|---|
| 1 | Exploratory Data Analysis | Pandas, NumPy, Matplotlib, Seaborn | 🔄 In progress |
| 2 | Feature Engineering & Supervised Baseline | Scikit-learn, XGBoost, MLflow | ⏳ Pending |
| 3 | Unsupervised Models | Isolation Forest, K-Means | ⏳ Pending |
| 4 | Neural Network | PyTorch MLP | ⏳ Pending |
| 5 | NLP with BERT | HuggingFace Transformers | ⏳ Pending |
| 6 | Serving & MLOps | FastAPI, Docker, GitHub Actions | ⏳ Pending |
| 7 | Big Tech-tier | Feast, Evidently, SHAP, gRPC, k6 | ⏳ Pending |

---

## Getting Started

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/creditai.git
cd creditai

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the notebooks

Open the project in DataSpell or JupyterLab and navigate to the `notebooks/` folder.
Start with `01_eda_exploratory_analysis.ipynb`.

### Running the API (Phase 6+)

```bash
# Build and run with Docker
docker build -t creditai .
docker run -p 8000:8000 creditai

# API docs available at
http://localhost:8000/docs
```

---

## Dataset

**Credit Card Fraud Detection** — [Kaggle / ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions with 492 fraud cases (~0.17%)
- Features anonymized via PCA (V1–V28) + `Time` and `Amount`
- Highly imbalanced — representative of real-world fraud scenarios

> Dataset files are gitignored. Download from Kaggle and place in `data/raw/`.

---

## Architecture Decision Records

ADRs are documented in [`/docs/adr/`](./docs/adr/). Key decisions:

- `ADR-001` — Model selection: Random Forest vs XGBoost as baseline
- `ADR-002` — MLflow vs Weights & Biases for experiment tracking
- `ADR-003` — FastAPI vs Flask for model serving

---

## References

| Book | Used in Phase |
|---|---|
| *Python for Data Analysis* — McKinney | 1 |
| *Practical Statistics for Data Scientists* — Bruce et al. | 1, 2 |
| *Hands-On Machine Learning* — Géron | 2, 3 |
| *Introducing MLOps* — Treveil et al. | 2 |
| *Programming PyTorch for Deep Learning* — Pointer | 4 |
| *NLP with Transformers* — Tunstall et al. | 5 |
| *Practical MLOps* — Gift, Deza | 6 |
| *Designing Machine Learning Systems* — Chip Huyen | 7 |
| *Reliable Machine Learning* — Cathy Chen et al. | 7 |

---

## License

MIT