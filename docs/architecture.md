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

### Preprocessing compatibility

New enrollments and re-enrollments use embedding version 3: symmetric square
padding before resizing, an explicit RGB-to-BGR handoff to DeepFace's ndarray
API, and a gamma guard for fully saturated images. Versions 1 and 2 retain
their historical geometry and color handling when searching existing profiles.

`SUPPORTED_EMBEDDING_VERSIONS` in `app/core/config.py` lists every recipe the
preprocessor can still reproduce, newest first, and `CURRENT_EMBEDDING_VERSION`
points at the one used for new enrollments. An import-time assertion keeps them
in step, so adding a recipe cannot silently drop the previous one out of search.

#### How a search picks recipes

A stored vector is only comparable to a query embedded with the same recipe.
Before embedding anything, search asks which `embedding_version` values
actually have rows for the tenant and embeds only those, plus the current
recipe. Immediately after a version bump that is one legacy recipe plus the
current one; once re-enrollment finishes it is a single recipe. The
per-recipe embeddings are independent and run concurrently on the inference
pool, so wall-clock stays at roughly one embedding regardless of how many
recipes are in play.

Each recipe is queried against its own vectors, scoped to the tenant, with the
existing cosine-distance threshold (distance <= MATCH_THRESHOLD). Hits from
every recipe are then **merged and ranked by distance together** — a closer
match under an older recipe is never masked by a weaker one under a newer
recipe — and `limit` bounds the merged set.

The current recipe is embedded on every search even when nothing is enrolled
under it yet. That one extra embedding is what makes `query_hash` a function of
the query image alone rather than of whichever recipe held the match, so audit
entries for the same image stay comparable across a re-enrollment.

Failure handling: if a recipe that holds enrolled vectors cannot be embedded,
those vectors were never compared, so search returns ERROR rather than a
misleading NO_MATCH. A failure of the current recipe while nothing is enrolled
under it costs only hash stability, not the search. Liveness rejection stops
the entire search. Database errors propagate; they are not treated as identity
mismatches.

#### Deploying a version bump

The `embedding_version` column already exists and needs no change. One
migration ships alongside this: an index on `(tenant_id, embedding_version)`
so the per-search recipe census is an index-only scan rather than a table
scan. It adds no columns or constraints and is safe to apply to a live table.

Do not relabel stored vectors — re-enroll to generate version 3 embeddings.
Raw images are not retained, so existing embeddings cannot be converted in
place. Existing version 1/2 enrollments remain searchable without
re-enrollment, and searches get cheaper as re-enrollment progresses: once a
tenant is fully on version 3, a search embeds once and queries once.

The backend runs as a single Compose service with no replicas, so a deploy
replaces it wholesale and there is no window in which an older release
searches a version 3 profile. That window would open if the backend were ever
scaled out or deployed with rolling replicas; in that case, upgrade every
search-serving instance before allowing version 3 enrollment.

Keep enhancement settings and model normalization consistent with enrollment;
these settings are not individually captured by the version number. Thresholds
and liveness policy are unchanged by this work. Recognition accuracy still
needs evaluation with representative consenting participants after
re-enrollment — in particular `MATCH_THRESHOLD`, which governs false accepts
and has not been re-tuned against version 3 distances.

- pgvector HNSW index provides ≤ 50ms ANN queries at 100k profiles
- RetinaFace detection: ≤ 200ms on 1080p frames
- ArcFace embedding: ≤ 300ms
- End-to-end: ≤ 1000ms for 100k profile database
