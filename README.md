# Blys AI – Customer Intelligence Platform

AI-powered customer intelligence platform: segmentation, recommendations, and NLP chatbot powered by **FastAPI** backend and **Streamlit** frontend.

---

## Architecture

```
blys-ai/
├── data/
│   ├── customers.csv            # Original synthetic dataset
│   └── cleaned_customer_data.csv # Cleaned/processed data
├── src/
│   ├── __init__.py
│   ├── chat_engine.py           # Chatbot logic & intent classification
│   └── recommendation_engine.py # Recommendation engine with collaborative filtering
├── frontend/
│   └── app.py                   # Streamlit frontend UI
├── models/                      # Serialised model artifacts
├── reports/                     # Generated customer analysis reports (PDF/HTML)
├── notebooks/
│   ├── chatbot_model.ipynb
│   ├── customer_behaviour_analysis.ipynb
│   └── recommendation_model.ipynb
├── images/                      # Visual assets & plots
├── api.py                       # FastAPI backend service
├── generate_data.py             # Synthetic dataset generator
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Multi-container orchestration
├── Dockerfile.backend           # Backend (FastAPI) container
├── Dockerfile.frontend          # Frontend (Streamlit) container
├── .dockerignore
└── README.md                    # This file
```

---

## Stack

| Component | Technology | Port |
|-----------|-----------|------|
| **Backend API** | FastAPI + Uvicorn | 8000 |
| **Frontend** | Streamlit | 8501 |
| **ML Models** | scikit-learn, joblib | — |
| **Python Version** | 3.10+ | — |

---

## Quick Start

### Option 1: Docker Compose (Recommended)

**Requirements:** Docker & Docker Compose

```bash
cd blys-ai
docker-compose up --build
```

This will start:
- **Backend API**: http://localhost:8000 (FastAPI docs at `/docs`)
- **Frontend**: http://localhost:8501 (Streamlit app)

To rebuild images:
```bash
docker-compose build
docker-compose up
```

To stop:
```bash
docker-compose down
```

---

### Option 2: Local Development

**Requirements:** Python 3.10+

```bash
git clone <repo-url> && cd blys-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

#### Generate Dataset
```bash
python generate_data.py
# → data/customers.csv (500 records)
```

#### Start Backend API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
API docs: http://localhost:8000/docs

#### Start Frontend (in new terminal)
```bash
cd frontend
streamlit run app.py
```
Frontend: http://localhost:8501

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "models_loaded": true, "customer_count": 500 }
```

### `POST /recommend`
```json
// Request
{ "customer_id": 1042, "top_n": 5 }

// Response
{
  "customer_id": 1042,
  "segment": "Loyal Regular",
  "recommendations": ["Deep Tissue Massage", "Wellness Package", "Aromatherapy", "Body Scrub", "Reflexology"]
}
```

### `POST /chatbot`
Supports multi-turn dialogue via `session_id`. Omit `session_id` on first message; pass it back on subsequent turns.

```json
// Turn 1
{ "query": "I need to reschedule my booking" }

// Response
{
  "response": "Sure, I can reschedule your booking. What is the new date and time?",
  "intent": "reschedule",
  "confidence": 0.9312,
  "session_id": "a3f9b1c2"
}

// Turn 2 (continue the dialogue)
{ "query": "30 Mar 2025 10am", "session_id": "a3f9b1c2" }

// Response
{
  "response": "✅ Reschedule request sent to your therapist for 30 Mar 2025 10am...",
  "intent": "reschedule",
  "confidence": 0.4210,
  "session_id": "a3f9b1c2"
}
```

### `GET /customers/{customer_id}/segment`
```json
{ "customer_id": 1042, "segment": "Loyal Regular" }
```

---

## Docker & Containerization

### Architecture

The project uses **separate containers** for frontend and backend, orchestrated with Docker Compose:

```
┌─────────────────┐
│  Streamlit UI   │ (port 8501)
│  Dockerfile.frontend
└────────┬────────┘
         │ HTTP requests
         ↓
┌─────────────────┐
│  FastAPI        │ (port 8000)
│  Dockerfile.backend
└─────────────────┘
         │
      ↙  ↓  ↘
   data models reports
```

### Why Separate Dockerfiles?

| Aspect | Benefit |
|--------|---------|
| **Scalability** | Scale backend independently if needed via load balancer |
| **Development** | Developers can work on frontend without rebuilding backend |
| **Dependencies** | Different packages per service (Streamlit vs Uvicorn) |
| **Deployment** | Deploy frontend and backend to separate instances/clusters |
| **Monitoring** | Monitor container metrics per service |

### Build & Run

**Build images:**
```bash
docker-compose build
```

**Start services:**
```bash
docker-compose up
```

**View logs:**
```bash
docker-compose logs -f backend    # Backend logs
docker-compose logs -f frontend   # Frontend logs
docker-compose logs -f            # All logs
```

**Stop services:**
```bash
docker-compose down
```

**Rebuild specific service:**
```bash
docker-compose up --build backend
# or
docker-compose up --build frontend
```

### Build Individual Images

```bash
# Backend only
docker build -f Dockerfile.backend -t blys-api:latest .
docker run -p 8000:8000 blys-api:latest

# Frontend only
docker build -f Dockerfile.frontend -t blys-ui:latest .
docker run -p 8501:8501 blys-ui:latest
```

### Health Checks

The compose file includes health checks:
- **Backend**: Hits `/health` endpoint every 30s
- **Frontend**: Waits for backend to be healthy before starting

### Volume Mounts

The compose file mounts:
- `./data`, `./models`, `./reports` → shared between services
- Optional: uncomment to use code volumes for live reload during development

### Environment Variables

The frontend and backend communicate via service names in Docker:
- **Frontend** → needs `API_BASE_URL=http://backend:8000` (automatically set in compose)
- **Local dev** → set `API_BASE_URL=http://localhost:8000` before running Streamlit

Copy and customize `.env.example`:
```bash
cp .env.example .env
# edit .env as needed
```

To use custom env file:
```bash
docker-compose --env-file .env up
```

For production deployment:

1. **Environment variables** — update `docker-compose.yml` with prod database/auth configs
2. **Image registry** — push to Docker Hub, AWS ECR, or private registry
3. **Orchestration** — migrate to Kubernetes, AWS ECS, or similar
4. **Reverse proxy** — add Nginx/Traefik frontend for SSL/routing
5. **Logging** — integrate with ELK, Datadog, or cloud logging

---

## Tests

```bash
pytest tests/ -v
```

Tests cover:
- Missing value imputation
- Recency / sentiment / churn risk feature engineering
- Intent classification accuracy across all 5 intents
- Date entity extraction
- Multi-turn reschedule and cancel dialogue flows

---

## ML Design Decisions

| Component | Approach | Rationale |
|-----------|----------|-----------|
| Sentiment | VADER | Fast, no training required, strong on short review text |
| Clustering | KMeans + silhouette k-selection | Interpretable segments; auto-selects k in [3,6] |
| Recommendations | Truncated SVD (collaborative filtering) | Captures latent service preferences; frequency-weighted |
| Chatbot intent | TF-IDF + LogisticRegression pipeline | High accuracy on small corpus; single-artefact serialisation |
| Dialogue management | Finite state machine | Explicit, debuggable multi-turn flows without external dependencies |

---

## Customer Analysis Report

After training, open `reports/customer_analysis.md` for:
- Sentiment distribution
- Segment profiles (booking frequency, spend, churn risk)
- High-value retention strategies
- At-risk re-engagement tactics
- Recommendation model evaluation (Precision@5, Recall@5)
