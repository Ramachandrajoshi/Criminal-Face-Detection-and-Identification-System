# Architecture — Criminal Face Detection & Identification System

## 1. System Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  React/Vite  │────▶│   FastAPI    │────▶│ PostgreSQL  │
│  Frontend    │◀────│  Backend     │◀────│ + pgvector  │
│  (Port 5173) │     │  (Port 8000) │     │ (Port 5432) │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │
       │  Camera            │  deepface + ArcFace
       │  (MediaStream)     │  + RetinaFace detector
       ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│              Five-Stage Face Pipeline                    │
│  DETECT → ALIGN → NORMALIZE → REPRESENT → VERIFY       │
└─────────────────────────────────────────────────────────┘
```

## 2. Component Details

### 2.1 Frontend (React/Vite/TypeScript)

- **Dashboard.tsx** — File upload, search, and registration UI
- **CameraFeed.tsx** — Live camera via `getUserMedia`, frame capture for search
- **AlertPanel.tsx** — Paginated alert list with confirm/dismiss actions
- **SuspectMap.tsx** — Leaflet GPS map (placeholder; GPS-enabled cameras needed)
- **api/client.ts** — Typed axios wrappers for all FastAPI endpoints
- **hooks/useAlerts.ts** — Data-fetching with refresh and confirmation
- **types/index.ts** — Shared domain types (Alert, SuspectMatch, AuditEntry)

### 2.2 Backend (FastAPI/Python 3.10)

- **main.py** — FastAPI app with CORS, auth middleware, routers
- **core/pipeline.py** — Five-stage face processing pipeline
- **core/liveness.py** — Anti-spoofing module
- **core/auth.py** — JWT creation/verification
- **core/middleware.py** — JWT validation middleware
- **core/config.py** — Environment-driven settings (thresholds, models)
- **db/models.py** — SQLAlchemy ORM: SuspectProfile, AuditLog, Alert
- **db/session.py** — Async SQLAlchemy engine
- **db/vector_ops.py** — pgvector ANN query helpers
- **schemas/face.py** — Pydantic request/response models
- **api/routes/** — Register, Search, Alerts, Audit endpoints

### 2.3 Database (PostgreSQL + pgvector)

- **suspect_profiles** — Stores pgvector `face_embedding` for ANN queries plus AES-256 `face_embedding_enc` payload
- **audit_log** — Append-only audit trail (INSERT only)
- **alerts** — Human-in-the-loop confirmation workflow

## 3. Data Flow

```
1. User uploads image (or camera captures frame)
2. Frontend sends multipart/form-data to POST /api/v1/search
3. Backend validates MIME type and file size (≤ 5 MB)
4. Stage 1-3: OpenCV + RetinaFace detect and align face
5. Stage 4: deepface.represent() extracts 512-d ArcFace embedding
6. Stage 5: pgvector HNSW cosine ANN search (threshold ≤ 0.58)
7. If match found → create Alert (PENDING_REVIEW) + log to audit_log
8. Frontend displays results with "Decision Support Only" disclaimer
9. Human operator confirms/dismisses via POST /api/v1/alerts/{id}/confirm
```

## 4. Security Model

- **JWT auth** on all non-health endpoints (HS256, 8h expiry)
- **Input validation**: MIME type, file size, image dimensions
- **AES-256 encryption** of embeddings at rest (`face_embedding_enc`) with volume encryption for pgvector storage
- **TLS 1.3** in transit (Docker Compose with reverse proxy)
- **Append-only audit_log**: no UPDATE or DELETE grants
- **No raw image storage**: only embeddings persist

## 5. Performance

- pgvector HNSW index provides ≤ 50ms ANN queries at 100k profiles
- RetinaFace detection: ≤ 200ms on 1080p frames
- ArcFace embedding: ≤ 300ms
- End-to-end: ≤ 1000ms for 100k profile database
