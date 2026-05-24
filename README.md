# Criminal Face Detection & Identification System

> **Decision-support platform for real-time face detection and identification.**
>
> **Disclaimer:** This system is a **decision-support tool only**. It must never autonomously dispatch, flag, or act on a match without a human-in-the-loop confirmation step.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0+-green.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3+-blue.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.4+-blue.svg)](https://www.typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.3+-blue.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Quick Start with Docker Compose](#quick-start-with-docker-compose)
- [Local Development Setup](#local-development-setup)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Database Setup](#database-setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Authentication](#authentication)
  - [Register a Suspect Profile](#register-a-suspect-profile)
  - [Search for a Match](#search-for-a-match)
  - [Manage Alerts](#manage-alerts)
  - [Audit Log](#audit-log)
- [API Reference](#api-reference)
- [Five-Stage Face Pipeline](#five-stage-face-pipeline)
- [Project Directory Structure](#project-directory-structure)
- [Testing](#testing)
- [Performance Targets](#performance-targets)
- [Security & Fairness](#security--fairness)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

---

## Overview

The **Criminal Face Detection & Identification System** is a real-time, decision-support platform that uses deep learning–based face recognition to detect, embed, and match facial biometrics against a database of suspect profiles. Built on the [`deepface`](https://github.com/serengil/deepface) library with an ArcFace backbone, it provides:

- **Real-time face detection and alignment** via RetinaFace
- **512-dimensional facial embeddings** stored in PostgreSQL with pgvector
- **Approximate nearest-neighbour (ANN) search** using HNSW indexing
- **Anti-spoofing / liveness detection** to prevent photo and mask attacks
- **Human-in-the-loop alert confirmation** — no match is acted upon without operator review
- **Append-only audit logging** for full regulatory compliance
- **React/Leaflet dashboard** with live camera feed and GPS mapping

---

## Core Features

| Feature | Description |
| --- | --- |
| **Face Registration** | Upload a frontal face image + metadata; system extracts and stores a 512-d ArcFace embedding |
| **Face Search** | Upload a query image; the pipeline runs detection, embedding, and ANN matching against all suspect profiles |
| **Liveness Detection** | Anti-spoofing module blocks photo, video, and mask attacks; failed attempts are logged as `SPOOF_BLOCKED` |
| **Alert Management** | Matches are created as alerts with `PENDING_REVIEW` status; an operator confirms or dismisses each alert |
| **Audit Trail** | Every search, registration, and spoof attempt is recorded in an append-only `audit_log` table |
| **GPS Mapping** | Leaflet-based map in the frontend visualises alert locations |
| **JWT Authentication** | All API endpoints (except `/health`) require a Bearer token (HS256, 8-hour expiry) |
| **Docker Compose** | One-command orchestration of PostgreSQL, FastAPI backend, and React frontend |

---

## Technology Stack

| Layer | Technology |
| --- | --- |
| **Face Analysis** | `deepface` >= 0.0.100 (ArcFace backbone, RetinaFace detector) |
| **ML Runtime** | TensorFlow 2.16.0+ (Python 3.11+) |
| **Video Capture** | OpenCV 4.7.0+ |
| **Backend API** | FastAPI >= 0.110.0 (async, Python 3.11+) |
| **Database** | PostgreSQL 15.3 + pgvector 0.4.1 |
| **Vector Index** | HNSW via pgvector (`vector_cosine_ops`) |
| **Frontend** | React 18.3 + Vite 5.x + TypeScript 5.x + Leaflet JS |
| **Containerisation** | Docker + Docker Compose |
| **OS** | Ubuntu 22.04 LTS (recommended) |

---

## Prerequisites

| Tool | Minimum Version | Purpose |
| --- | --- | --- |
| [Python](https://www.python.org/downloads/) | 3.11+ | Backend runtime |
| [Node.js](https://nodejs.org/) | 18+ | Frontend build |
| [npm](https://www.npmjs.com/) | 9+ | Package management |
| [Docker](https://www.docker.com/) | 24+ | Container runtime |
| [Docker Compose](https://docs.docker.com/compose/) | 2.20+ | Service orchestration |
| [Git](https://git-scm.com/) | 2.40+ | Version control |

> **Note:** Python 3.11+ is required for TensorFlow 2.16+. If you need TensorFlow 2.12, use Python 3.10.

---

## Quick Start with Docker Compose

The fastest way to run the full stack is Docker Compose.

### 1. Clone and configure

```bash
git clone https://github.com/your-org/criminal-face-detection.git
cd criminal-face-detection
cp .env.example .env
```

Edit `.env` and set secure values for `POSTGRES_PASSWORD`, `JWT_SECRET`, and `DB_ENCRYPTION_KEY`.

### 2. Start all services

```bash
docker-compose up --build
```

This starts:

| Service | Container | Port |
| --- | --- | --- |
| PostgreSQL + pgvector | `criminal_db` | `5432` |
| FastAPI backend | `criminal_backend` | `8000` |
| React frontend (via Nginx) | `criminal_frontend` | `5173` |

### 3. Access the application

- **Frontend Dashboard:** http://localhost:5173
- **Backend API docs (Swagger):** http://localhost:8000/api/docs
- **Backend ReDoc:** http://localhost:8000/api/redoc
- **Health check:** http://localhost:8000/api/v1/health

### 4. Log in

Admin credentials are configured via environment variables.

Obtain a JWT token via:

```bash
curl -X POST http://localhost:8000/api/v1/login \
  -H "Content-Type: application/json" \
  -d '{"username":"<admin_username>","password":"<admin_password>"}'
```

---

## Local Development Setup

### Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure PostgreSQL is running locally (or use Docker)
docker run -d --name criminal_db \
  -e POSTGRES_DB=criminaldb \
  -e POSTGRES_USER=appuser \
  -e POSTGRES_PASSWORD=<strong-password> \
  -p 5432:5432 \
  pgvector/pgvector:pg15

# Run the database initialisation
psql -U appuser -d criminaldb -f ../db/init.sql

# Start the FastAPI dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at http://localhost:8000.

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the Vite dev server (with API proxy to backend)
npm run dev
```

The frontend will be available at http://localhost:5173.

---

## Database Setup

The database schema is defined in [`db/init.sql`](db/init.sql). It creates three tables:

### Tables

**`suspect_profiles`** — Stores encrypted 512-d ArcFace embeddings with an HNSW index for fast ANN search.

```sql
CREATE TABLE suspect_profiles (
    id               SERIAL PRIMARY KEY,
    suspect_name     VARCHAR(100) NOT NULL,
    alias            VARCHAR(100),
    demographics     JSONB,
    face_embedding   vector(512) NOT NULL,
  face_embedding_enc BYTEA,
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX suspect_embedding_hnsw_idx
    ON suspect_profiles
    USING hnsw (face_embedding vector_cosine_ops);
```

**`audit_log`** — Append-only audit trail (INSERT only; no UPDATE or DELETE).

```sql
CREATE TABLE audit_log (
    id           SERIAL PRIMARY KEY,
    event_type   VARCHAR(50) NOT NULL,
    query_hash   TEXT NOT NULL,
    result_name  VARCHAR(100),
    distance     FLOAT,
    gps_lat      FLOAT,
    gps_lon      FLOAT,
    timestamp    TIMESTAMPTZ DEFAULT now()
);
```

**`alerts`** — Human-in-the-loop confirmation workflow.

```sql
CREATE TABLE alerts (
    id            SERIAL PRIMARY KEY,
    audit_log_id  INTEGER REFERENCES audit_log(id),
    suspect_id    INTEGER REFERENCES suspect_profiles(id),
    event_type    VARCHAR(50) NOT NULL,
    distance      FLOAT,
    status        VARCHAR(20) NOT NULL DEFAULT 'PENDING_REVIEW',
    gps_lat       FLOAT,
    gps_lon       FLOAT,
    created_at    TIMESTAMPTZ DEFAULT now(),
    confirmed_at  TIMESTAMPTZ
);
```

### Seed synthetic data (optional)

```bash
cd db
python seed_synthetic.py
```

---

## Configuration

All settings are injected via environment variables. Copy `.env.example` to `.env` and adjust. **Never commit a `.env` file.**

```bash
# ── Model Configuration ─────────────────────────────────────────
DEEPFACE_MODEL=ArcFace          # ArcFace | FaceNet | VGG-Face | SFace
DEEPFACE_DETECTOR=retinaface    # retinaface | mtcnn | opencv
MATCH_THRESHOLD=0.58            # Cosine distance threshold for a match

# ── Database ────────────────────────────────────────────────────
POSTGRES_HOST=db                # Use 'db' in Docker, 'localhost' locally
POSTGRES_PORT=5432
POSTGRES_DB=criminaldb
POSTGRES_USER=appuser
POSTGRES_PASSWORD=<strong-password>
DB_ENCRYPTION_KEY=<base64-32-byte-key>

# ── API ─────────────────────────────────────────────────────────
JWT_SECRET=replace-with-strong-secret-in-production
JWT_EXPIRY_HOURS=8
API_HOST=0.0.0.0
API_PORT=8000

# ── Admin Auth ──────────────────────────────────────────────────
ADMIN_USERNAME=<admin-username>
ADMIN_PASSWORD_HASH=<bcrypt-hash>
ALLOW_ADMIN_INIT=false

# ── Frontend ────────────────────────────────────────────────────
VITE_API_BASE_URL=http://localhost:8000
```

---

## Usage

### Authentication

All endpoints except `/api/v1/health` require a Bearer token.

**Login:**

```bash
curl -X POST http://localhost:8000/api/v1/login \
  -H "Content-Type: application/json" \
  -d '{"username":"<admin_username>","password":"<admin_password>"}'
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 28800
}
```

**Include the token in subsequent requests:**

```bash
curl -H "Authorization: Bearer <your_token>" http://localhost:8000/api/v1/search
```

---

### Register a Suspect Profile

**Endpoint:** `POST /api/v1/register`

Upload a clear frontal face image along with the suspect's name and optional metadata.

```bash
curl -X POST http://localhost:8000/api/v1/register \
  -H "Authorization: Bearer <your_token>" \
  -F "file=@suspect_photo.jpg" \
  -F "suspect_name=John Doe" \
  -F "alias=JD" \
  -F 'demographics={"age_band":"36-60","gender":"M","ethnicity":"Caucasian"}'
```

**Response (201 Created):**

```json
{
  "status": "REGISTERED",
  "profile_id": 1,
  "query_hash": "a1b2c3d4e5f6...",
  "embedding_dim": 512
}
```

---

### Search for a Match

**Endpoint:** `POST /api/v1/search`

Upload a query image to search against all registered suspect profiles.

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Authorization: Bearer <your_token>" \
  -F "file=@query_photo.jpg" \
  -F 'search_data={"gps_lat":40.7128,"gps_lon":-74.0060}' \
  -H "Content-Type: multipart/form-data"
```

**Response (MATCH):**

```json
{
  "status": "MATCH",
  "query_hash": "a1b2c3d4e5f6...",
  "matches": [
    {
      "id": 1,
      "suspect_name": "John Doe",
      "alias": "JD",
      "distance": 0.42
    }
  ],
  "gps_lat": 40.7128,
  "gps_lon": -74.0060
}
```

**Response (NO_MATCH):**

```json
{
  "status": "NO_MATCH",
  "query_hash": "a1b2c3d4e5f6...",
  "matches": [],
  "gps_lat": 40.7128,
  "gps_lon": -74.0060
}
```

**Response (SPOOF_BLOCKED):**

```json
{
  "status": "SPOOF_BLOCKED",
  "query_hash": "a1b2c3d4e5f6...",
  "matches": []
}
```

---

### Manage Alerts

**List alerts (paginated):**

```bash
curl http://localhost:8000/api/v1/alerts?page=1&page_size=20\&status_filter=PENDING_REVIEW \
  -H "Authorization: Bearer <your_token>"
```

**Confirm or dismiss an alert:**

```bash
curl -X POST http://localhost:8000/api/v1/alerts/1/confirm \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{"confirmed": true}'
```

**Response:**

```json
{
  "alert_id": 1,
  "status": "CONFIRMED",
  "confirmed_at": "2026-05-24T14:30:00+00:00"
}
```

---

### Audit Log

**Retrieve audit entries (read-only):**

```bash
curl "http://localhost:8000/api/v1/audit?page=1&page_size=50\&event_type=MATCH" \
  -H "Authorization: Bearer <your_token>"
```

**Response:**

```json
[
  {
    "id": 1,
    "event_type": "MATCH",
    "query_hash": "a1b2c3d4e5f6...",
    "result_name": "John Doe",
    "distance": 0.42,
    "gps_lat": 40.7128,
    "gps_lon": -74.0060,
    "timestamp": "2026-05-24T14:30:00+00:00"
  }
]
```

---

## API Reference

| Method   | Path                                         | Description                                        | Auth   |
| -------- | -------------------------------------------- | -------------------------------------------------- | ------ |
| `POST`   | [`/api/v1/login`](backend/app/api/routes/auth.py:47)              | Authenticate and receive a JWT token               | None   |
| `POST`   | [`/api/v1/token/refresh`](backend/app/api/routes/auth.py:73)      | Refresh an expiring JWT token                      | None   |
| `POST`   | [`/api/v1/register`](backend/app/api/routes/register.py:22)       | Register a suspect profile                         | JWT    |
| `POST`   | [`/api/v1/search`](backend/app/api/routes/search.py:23)           | Search for a matching suspect                      | JWT    |
| `GET`    | [`/api/v1/alerts`](backend/app/api/routes/alerts.py:22)           | List recent alerts (paginated, filterable)         | JWT    |
| `POST`   | [`/api/v1/alerts/{id}/confirm`](backend/app/api/routes/alerts.py:83) | Confirm or dismiss an alert                     | JWT    |
| `GET`    | [`/api/v1/audit`](backend/app/api/routes/audit.py:22)             | Read-only audit log                                | JWT    |
| `GET`    | [`/api/v1/health`](backend/app/main.py:51)                        | Liveness probe                                     | None   |

### Request / Response Schemas

See [`backend/app/schemas/face.py`](backend/app/schemas/face.py) for full Pydantic model definitions. All response models output **camelCase** JSON keys to match the TypeScript client types.

| Schema | Description |
| --- | --- |
| `RegisterRequest` | `suspect_name`, `alias`, `demographics` |
| `RegisterResponse` | `status`, `profile_id`, `query_hash`, `embedding_dim`, `error` |
| `SearchRequest` | `gps_lat`, `gps_lon` |
| `SearchResponse` | `status`, `query_hash`, `matches[]`, `gps_lat`, `gps_lon` |
| `MatchResult` | `id`, `suspect_name`, `alias`, `distance` |
| `AlertResponse` | `id`, `audit_log_id`, `suspect_id`, `event_type`, `distance`, `status`, `gps_lat`, `gps_lon`, `created_at`, `confirmed_at` |
| `AuditEntryResponse` | `id`, `event_type`, `query_hash`, `result_name`, `distance`, `gps_lat`, `gps_lon`, `timestamp` |
| `ConfirmRequest` | `confirmed` (boolean) |
| `HealthResponse` | `status`, `database` |

---

## Five-Stage Face Pipeline

Every face image passes through exactly five stages in order. This pipeline is implemented in [`backend/app/core/pipeline.py`](backend/app/core/pipeline.py).

```
Stage 1 — DETECT   : OpenCV frame grab → RetinaFace bounding box
                     (fallback: MTCNN → OpenCV)

Stage 2 — ALIGN    : Eye-landmark affine transform
                     (eliminates tilt, rotation, skew)

Stage 3 — NORMALIZE: Resize to 112x112 (ArcFace) or 224x224 (FaceNet/VGG-Face)
                     L2 pixel normalisation to reduce covariate shift

Stage 4 — REPRESENT: deepface.represent() → 512-d ArcFace embedding vector

Stage 5 — VERIFY   : pgvector cosine ANN query
                     Flag match if distance <= MATCH_THRESHOLD (default 0.58)
```

### Distance Metrics

**Cosine distance (primary):**

$$D_C(u, v) = 1 - \\frac{{u \\cdot v}}{{\\|u\\|_2 \\times \\|v\\|_2}}$$

**L2-normalised Euclidean (secondary):**

$$\\hat{u} = \\frac{{u}}{{\\|u\\|_2}}, \\quad D_{L2}(\\hat{u}, \\hat{v}) = \\sqrt{{\\sum (\\hat{u}_i - \\hat{v}_i)^2}}$$

Match threshold `T = 0.58` (cosine) lives in [`backend/app/core/config.py`](backend/app/core/config.py) as `MATCH_THRESHOLD`.

### Model Options

| Model | Embedding Dims | Loss | Use-case |
| --- | --- | --- | --- |
| **ArcFace** *(default)* | 512 | Additive Angular Margin | Primary: occlusion-robust production matching |
| FaceNet | 128 / 512 | Triplet Loss | Benchmark comparison only |
| VGG-Face | 4096 | Softmax / Cross-Entropy | Benchmark comparison only |
| SFace | 512 | Sigmoid-Constrained Hypersphere | Low-resolution CCTV benchmark |

---

## Project Directory Structure

```
project-root/
├── README.md                         ← You are here
├── AGENTS.md                         ← Agent development rules
├── docker-compose.yml                # Multi-service orchestration
├── .env.example                      # Environment variable template
│
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI entry point
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── register.py       # POST /api/v1/register
│   │   │       ├── search.py         # POST /api/v1/search
│   │   │       ├── alerts.py         # GET/POST /api/v1/alerts
│   │   │       ├── audit.py          # GET /api/v1/audit
│   │   │       └── auth.py           # POST /api/v1/login
│   │   ├── core/
│   │   │   ├── pipeline.py           # Five-stage face pipeline
│   │   │   ├── liveness.py           # Anti-spoofing module
│   │   │   ├── config.py             # Environment-driven settings
│   │   │   ├── auth.py               # JWT creation & verification
│   │   │   └── middleware.py          # JWT validation middleware
│   │   ├── db/
│   │   │   ├── session.py            # Async SQLAlchemy engine
│   │   │   ├── models.py             # ORM: SuspectProfile, AuditLog, Alert
│   │   │   └── vector_ops.py         # pgvector ANN query helpers
│   │   └── schemas/
│   │       └── face.py               # Pydantic request/response models
│   ├── tests/
│   │   ├── unit/
│   │   │   ├── test_pipeline.py      # Mock deepface stages
│   │   │   ├── test_vector_ops.py    # Mock pgvector queries
│   │   │   ├── test_liveness.py      # SPOOF_BLOCKED path
│   │   │   └── test_auth.py          # JWT auth
│   │   └── integration/
│   │       ├── test_register_flow.py # POST /register end-to-end
│   │       ├── test_search_flow.py   # POST /search with known embedding
│   │       ├── test_alerts.py        # Alert CRUD
│   │       ├── test_audit.py         # Audit log queries
│   │       └── test_audit_immutability.py
│   ├── requirements.txt              # Python dependencies
│   ├── pyproject.toml
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.tsx         # Main dashboard
│   │   │   ├── CameraFeed.tsx        # Live camera via getUserMedia
│   │   │   ├── AlertPanel.tsx        # Alert list + confirm/dismiss
│   │   │   ├── SuspectMap.tsx        # Leaflet GPS map
│   │   │   └── Login.tsx             # Login form
│   │   ├── api/
│   │   │   └── client.ts             # Typed axios wrappers
│   │   ├── hooks/
│   │   │   ├── useAlerts.ts          # Alert data-fetching
│   │   │   └── useAuth.tsx           # Auth state management
│   │   ├── types/
│   │   │   └── index.ts              # Shared domain types
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── vite-env.d.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   └── Dockerfile
│
├── db/
│   ├── init.sql                      # Database schema + indexes
│   └── seed_synthetic.py             # 100k synthetic profiles
│
├── notebooks/
│   ├── 01_detector_benchmark.ipynb   # Detector comparison
│   ├── 02_model_comparison.ipynb     # Model benchmark
│   ├── 03_threshold_calibration.ipynb # Threshold tuning
│   └── 04_bias_audit.ipynb           # Fairness & bias analysis
│
└── docs/
    ├── architecture.md               # System architecture
    └── api_reference.md              # Detailed API docs
```

---

## Testing

### Backend

Run all unit and integration tests with coverage enforcement:

```bash
cd backend
pytest tests/ -v --cov=app --cov-fail-under=80
```

Run a specific test file:

```bash
pytest tests/unit/test_pipeline.py -v
pytest tests/integration/test_search_flow.py -v
```

### Frontend

Type-check the entire codebase:

```bash
cd frontend
npx tsc --noEmit
```

Build for production:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

---

## Performance Targets

| Metric | Target |
| --- | --- |
| End-to-end latency (single frame, 100k profiles) | <= 1,000 ms |
| ANN query time (pgvector HNSW, 100k profiles) | <= 50 ms |
| RetinaFace detection (1080p frame) | <= 200 ms |
| ArcFace embedding extraction | <= 300 ms |
| API p95 response time | <= 800 ms |
| False Acceptance Rate (FAR) | <= 0.1% |

---

## Security & Fairness

### Security Rules

| Rule | Detail |
| --- | --- |
| **No raw image storage** | Only `vector(512)` embeddings are persisted. Raw frames are discarded post-pipeline. |
| **Encrypted embeddings** | AES-256 at rest; TLS 1.3 in transit. |
| **Append-only audit** | `audit_log` has no `DELETE` / `UPDATE` grants. |
| **Token auth** | JWT on all non-health endpoints. Tokens expire in 8 hours. |
| **Input validation** | All uploads validated for MIME type, file size (<= 5 MB), and dimensionality. |
| **No PII in logs** | Application logs use `suspect_id` (integer) only. |
| **Legal alignment** | Every match result displays a "Decision Support Only" disclaimer. |

### Fairness & Bias

- Subgroup performance (age, gender, ethnicity) is tracked in [`notebooks/04_bias_audit.ipynb`](notebooks/04_bias_audit.ipynb).
- Group-specific thresholds may be used **only** if a fairness audit demonstrates statistically significant disparity (p < 0.05).
- The fairness dataset must cover: age bands (18-35, 36-60, 60+), gender (M/F/non-binary), and at least 4 ethnic subgroups.

---

## Contributing

### Commit Convention

```
<type>(<scope>): <short description>

Types : feat | fix | perf | refactor | test | docs | chore | security
Scopes: pipeline | db | api | frontend | liveness | fairness | infra | docs
```

**Examples:**

```
feat(pipeline): add MTCNN fallback detector
fix(db): prevent UPDATE on audit_log table
perf(db): switch to HNSW index for ANN queries
security(api): enforce JWT expiry to 8h
test(pipeline): add unit test for SPOOF_BLOCKED path
```

### Pre-commit Checklist

1. **Run backend tests:** `cd backend && pytest tests/ -v --cov=app --cov-fail-under=80`
2. **Run frontend type-check:** `cd frontend && npx tsc --noEmit`
3. **No credentials committed:** Verify `.env` is not tracked.
4. **No hard-coded thresholds:** `MATCH_THRESHOLD` and `JWT_SECRET` must come from env.
5. **Human-in-the-loop preserved:** No commit may bypass the alert confirmation step.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Support

| Resource | Link |
| --- | --- |
| **API Documentation** | http://localhost:8000/api/docs (Swagger UI) |
| **Architecture Docs** | [`docs/architecture.md`](docs/architecture.md) |
| **API Reference** | [`docs/api_reference.md`](docs/api_reference.md) |
| **Issue Tracker** | GitHub Issues (configure your repo URL) |
| **Project Maintainer** | Contact the repository owner |

---

*Last updated: 2026-05-24*
