# AGENTS.md — Criminal Face Detection & Identification System

## Agentic Development Harness

This file governs all AI-agent behaviour across the project. Every agent (Claude Code,
Codex, Cursor, or any other agentic tool) **must read this file in full before making any
change** to the codebase.

---

## 1. Project Overview

**Goal:** A real-time criminal face detection and identification decision-support platform
built with the `deepface` Python library, backed by a PostgreSQL/pgvector database, and
fronted by a FastAPI + React/Vite dashboard with Leaflet GPS mapping.

**Stack at a glance:**

| Layer                                                                                         | Technology                                                       |
| --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Face analysis                                                                                 | `deepface` ≥ v0.0.100 (ArcFace backbone, RetinaFace detector) |
| ML runtime                                                                                    | TensorFlow 2.16.0+ (Python 3.11+)                                |
| Note: Python 3.10 recommended for TensorFlow 2.12; Python 3.11+ required for TensorFlow 2.16+ |                                                                  |
| Video capture                                                                                 | OpenCV 4.7.0                                                     |
| Backend API                                                                                   | FastAPI ≥ 0.110.0 (async, Python 3.11+)                         |
| Database                                                                                      | PostgreSQL 15.3 + pgvector 0.4.1                                 |
| Vector index                                                                                  | HNSW via pgvector (`vector_cosine_ops`)                        |
| Frontend                                                                                      | React 18.3 + Vite 5.x + TypeScript 5.x + Leaflet JS              |
| OS                                                                                            | Ubuntu 22.04 LTS                                                 |
| Containers                                                                                    | Docker + Docker Compose                                          |

**System is a decision-support tool only.** It must never autonomously dispatch, flag, or
act on a match without a human-in-the-loop confirmation step.

---

## 2. Repository Layout

```
project-root/
├── AGENTS.md                  ← this file
├── docker-compose.yml
├── .env.example
│
├── backend/
│   ├── app/
│   │   ├── main.py            ← FastAPI entry point
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── register.py
│   │   │   │   ├── search.py
│   │   │   │   ├── alerts.py
│   │   │   │   └── audit.py
│   │   ├── core/
│   │   │   ├── pipeline.py    ← deepface wrapper (detect→align→normalize→represent→verify)
│   │   │   ├── liveness.py    ← anti-spoofing module
│   │   │   └── config.py      ← env-driven settings (thresholds, model names)
│   │   ├── db/
│   │   │   ├── session.py     ← async SQLAlchemy engine
│   │   │   ├── models.py      ← ORM: SuspectProfile, AuditLog
│   │   │   └── vector_ops.py  ← pgvector ANN query helpers
│   │   └── schemas/
│   │       └── face.py        ← Pydantic request/response models
│   ├── tests/
│   │   ├── unit/
│   │   └── integration/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── CameraFeed.tsx
│   │   │   ├── AlertPanel.tsx
│   │   │   └── SuspectMap.tsx  ← Leaflet integration
│   │   ├── api/
│   │   │   └── client.ts       ← typed axios wrappers for FastAPI endpoints
│   │   ├── types/
│   │   │   └── index.ts        ← shared domain types (Alert, SuspectMatch, AuditEntry…)
│   │   ├── hooks/
│   │   │   └── useAlerts.ts    ← data-fetching hooks
│   │   ├── main.tsx
│   │   └── vite-env.d.ts
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   ├── package.json
│   └── Dockerfile
│
├── db/
│   ├── init.sql               ← schema + HNSW index creation
│   └── seed_synthetic.py      ← generates 100k synthetic vector profiles
│
├── notebooks/
│   ├── 01_detector_benchmark.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_threshold_calibration.ipynb
│   └── 04_bias_audit.ipynb
│
└── docs/
    ├── architecture.md
    └── api_reference.md
```

Agents **must not** create files outside this layout without first adding an entry here.

---

## 3. Pipeline Stages — Canonical Implementation

The five-stage pipeline is the core of the system. All agents must implement and preserve
these stages in order. Never collapse, skip, or reorder them.

```
Stage 1 — DETECT   : OpenCV frame grab → RetinaFace bounding box (fallback: MTCNN → OpenCV)
Stage 2 — ALIGN    : Eye-landmark affine transform (eliminates tilt, rotation, skew)
Stage 3 — NORMALIZE: Resize to 112×112 (ArcFace) or 224×224 (FaceNet/VGG-Face);
                     L2 pixel normalization to reduce covariate shift
Stage 4 — REPRESENT: deepface.represent() → 512-d ArcFace embedding vector
Stage 5 — VERIFY   : pgvector cosine ANN query; flag match if distance ≤ T (default 0.58)
```

### Distance metrics

```python
# Cosine distance (primary)
D_C(u, v) = 1 - (u · v) / (||u||₂ × ||v||₂)

# L2-normalized Euclidean (secondary)
û = u / ||u||₂
D_L2(û, v̂) = sqrt(sum((û_i - v̂_i)²))
```

Match threshold `T = 0.58` (cosine). This value lives in `backend/app/core/config.py`
as `MATCH_THRESHOLD` and must **never** be hard-coded elsewhere.

---

## 4. Database Schema — Do Not Modify Without Migration

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE suspect_profiles (
    id               SERIAL PRIMARY KEY,
    suspect_name     VARCHAR(100) NOT NULL,
    alias            VARCHAR(100),
    demographics     JSONB,
    face_embedding   vector(512) NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX suspect_embedding_hnsw_idx
    ON suspect_profiles
    USING hnsw (face_embedding vector_cosine_ops);

-- Immutable audit log (INSERT only — no UPDATE or DELETE)
CREATE TABLE audit_log (
    id           SERIAL PRIMARY KEY,
    event_type   VARCHAR(50) NOT NULL,    -- 'MATCH' | 'NO_MATCH' | 'REGISTER' | 'SPOOF_BLOCKED'
    query_hash   TEXT NOT NULL,           -- SHA-256 of raw embedding bytes
    result_name  VARCHAR(100),
    distance     FLOAT,
    gps_lat      FLOAT,
    gps_lon      FLOAT,
    timestamp    TIMESTAMPTZ DEFAULT now()
);
```

**Rules:**

- The `audit_log` table is append-only. No agent may add `UPDATE` or `DELETE` statements
  targeting it.
- Raw face images are **never** stored. Only 512-d embeddings are persisted.
- Embeddings must be stored encrypted at rest (AES-256). Decryption happens in
  `vector_ops.py` immediately before the ANN query.

---

## 5. Model Configuration

Agents selecting or switching models must use only entries from this table.
No other backbone may be introduced without a PR updating this file.

| Model                           | Embedding Dims | Loss                            | Use-case                                         |
| ------------------------------- | -------------- | ------------------------------- | ------------------------------------------------ |
| **ArcFace** *(default)* | 512            | Additive Angular Margin         | Primary: occlusion-robust production matching    |
| FaceNet                         | 128 / 512      | Triplet Loss                    | Benchmark comparison only                        |
| VGG-Face                        | 4096           | Softmax / Cross-Entropy         | Benchmark comparison only (high-quality frontal) |
| SFace                           | 512            | Sigmoid-Constrained Hypersphere | Low-resolution CCTV benchmark                    |

Detector backends in priority order: `retinaface` → `mtcnn` → `opencv`.

The active model and detector are set via environment variables:

```
DEEPFACE_MODEL=ArcFace
DEEPFACE_DETECTOR=retinaface
MATCH_THRESHOLD=0.58
```

---

## 6. API Endpoints Contract

Agents building or modifying API routes must conform to these signatures.
Response schemas live in `backend/app/schemas/face.py`.

| Method   | Path                            | Description                                                |
| -------- | ------------------------------- | ---------------------------------------------------------- |
| `POST` | `/api/v1/register`            | Upload image + metadata; extract & store embedding         |
| `POST` | `/api/v1/search`              | Upload query image; run pipeline; return match or NO_MATCH |
| `GET`  | `/api/v1/alerts`              | Paginated list of recent MATCH events                      |
| `POST` | `/api/v1/alerts/{id}/confirm` | Human operator confirms/dismisses a match                  |
| `GET`  | `/api/v1/audit`               | Read-only audit log (admin token required)                 |
| `GET`  | `/api/v1/health`              | Liveness probe (no auth)                                   |

**Authentication:** All endpoints except `/health` require a Bearer token (JWT, HS256).
The secret is loaded from env `JWT_SECRET`. Never hard-code it.

**Human-in-the-loop rule:** A `MATCH` event sets status `PENDING_REVIEW`. It transitions
to `CONFIRMED` or `DISMISSED` only via `POST /api/v1/alerts/{id}/confirm`. No downstream
action (alert dispatch, map pin) becomes permanent before that transition.

---

## 7. Liveness / Anti-Spoofing

```python
# Always pass enforce_detection=True and anti_spoofing=True
result = DeepFace.verify(
    img1_path=query_frame,
    img2_path=db_image,
    model_name="ArcFace",
    detector_backend="retinaface",
    distance_metric="cosine",
    enforce_detection=True,
    anti_spoofing=True,
)
```

If liveness fails, the pipeline **must** return `SPOOF_BLOCKED`, log the event to
`audit_log`, and **not** proceed to the ANN query.

---

## 8. Fairness & Bias Constraints

- Subgroup performance (age, gender, ethnicity) must be tracked in
  `notebooks/04_bias_audit.ipynb` before any threshold change.
- Group-specific thresholds `T_group` may be used **only** if the fairness audit
  demonstrates a statistically significant disparity (p < 0.05) for a subgroup.
- Any PR that changes `MATCH_THRESHOLD` must include updated bias-audit notebook output.
- The fairness dataset must cover at minimum: age bands (18–35, 36–60, 60+),
  gender (M/F/non-binary), and at least 4 ethnic subgroups.

---

## 9. Security Rules — Non-Negotiable

| Rule                 | Detail                                                                                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| No raw image storage | Store only `vector(512)` embeddings. Raw frames are discarded post-pipeline.                                                                                  |
| Encrypted embeddings | AES-256 at rest; TLS 1.3 in transit.                                                                                                                            |
| Append-only audit    | `audit_log` has no `DELETE`/`UPDATE` grants.                                                                                                              |
| Token auth           | JWT on all non-health endpoints. Tokens expire in 8 hours.                                                                                                      |
| Input validation     | All uploaded images validated for MIME type, file size (≤ 5 MB), and dimensionality before entering the pipeline.                                              |
| No PII in logs       | Application logs must not contain suspect names, GPS coordinates, or embedding values. Use `suspect_id` (integer) only.                                       |
| Legal alignment      | System must display a "Decision Support Only" disclaimer on every match result in the UI. References: UK DPA 2018, Ed Bridges v. South Wales Police, EU AI Act. |

---

## 10. Testing Requirements

Every agent-written function must be covered by tests before the PR is considered complete.

```
backend/tests/unit/
    test_pipeline.py        — mock deepface; test each stage independently
    test_vector_ops.py      — mock pgvector; test cosine query + threshold logic
    test_liveness.py        — test SPOOF_BLOCKED path

backend/tests/integration/
    test_register_flow.py   — POST /register end-to-end with test DB
    test_search_flow.py     — POST /search with known embedding; assert distance ≤ T
    test_audit_immutability.py — assert UPDATE/DELETE on audit_log raises PermissionError
```

**Coverage gate:** `pytest --cov=app --cov-fail-under=80`

Run before every commit:

```bash
cd backend && pytest tests/ -v --cov=app --cov-fail-under=80
```

---

## 11. Performance Targets

Agents must not degrade these baselines. Regression tests in CI enforce them.

| Metric                                           | Target      |
| ------------------------------------------------ | ----------- |
| End-to-end latency (single frame, 100k profiles) | ≤ 1 000 ms |
| ANN query time (pgvector HNSW, 100k profiles)    | ≤ 50 ms    |
| RetinaFace detection (1080p frame)               | ≤ 200 ms   |
| ArcFace embedding extraction                     | ≤ 300 ms   |
| API p95 response time                            | ≤ 800 ms   |
| False Acceptance Rate (FAR)                      | ≤ 0.1%     |

## 12. Commit & PR Conventions

```
<type>(<scope>): <short description>

Types : feat | fix | perf | refactor | test | docs | chore | security
Scopes: pipeline | db | api | frontend | liveness | fairness | infra | docs

Examples:
  feat(pipeline): add MTCNN fallback detector
  fix(db): prevent UPDATE on audit_log table
  perf(db): switch to HNSW index for ANN queries
  security(api): enforce JWT expiry to 8h
  test(pipeline): add unit test for SPOOF_BLOCKED path
```

- No commit may remove or bypass the human-in-the-loop confirmation step.
- No commit may add raw image persistence to the database layer.
- No commit may hard-code `MATCH_THRESHOLD`, `JWT_SECRET`, or any credential.

---

## 13. Environment Variables

All configuration is injected via environment. Copy `.env.example` and fill values.
Never commit a `.env` file.

```bash
# Model
DEEPFACE_MODEL=ArcFace
DEEPFACE_DETECTOR=retinaface
MATCH_THRESHOLD=0.58

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=criminaldb
POSTGRES_USER=appuser
POSTGRES_PASSWORD=<secret>
DB_ENCRYPTION_KEY=<aes256-key>

# API
JWT_SECRET=<secret>
JWT_EXPIRY_HOURS=8
API_HOST=0.0.0.0
API_PORT=8000

# Frontend
VITE_API_BASE_URL=http://localhost:8000
```

---

## 14. TypeScript Conventions (Frontend)

The frontend is **strictly TypeScript**. Agents must never create `.js` or `.jsx` files
inside `frontend/src/`. All React components use `.tsx`, all other modules use `.ts`.

### tsconfig.json baseline

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

`"strict": true` is non-negotiable. No agent may disable or override individual strict
flags (`noImplicitAny`, `strictNullChecks`, etc.).

### Shared domain types

All API response shapes are defined once in `frontend/src/types/index.ts` and imported
everywhere. Agents must not redefine inline types that duplicate an entry in that file.

```typescript
// frontend/src/types/index.ts — canonical shapes (extend as needed, never duplicate)

export type EventType = 'MATCH' | 'NO_MATCH' | 'REGISTER' | 'SPOOF_BLOCKED';
export type AlertStatus = 'PENDING_REVIEW' | 'CONFIRMED' | 'DISMISSED';

export interface SuspectMatch {
  suspectName: string;
  alias: string | null;
  distance: number;        // cosine distance, 0–1
  status: AlertStatus;
}

export interface Alert {
  id: number;
  eventType: EventType;
  match: SuspectMatch | null;
  gpsLat: number | null;
  gpsLon: number | null;
  timestamp: string;       // ISO-8601
}

export interface AuditEntry {
  id: number;
  eventType: EventType;
  queryHash: string;
  resultName: string | null;
  distance: number | null;
  timestamp: string;
}
```

### API client pattern

All HTTP calls go through `frontend/src/api/client.ts` using typed axios instances.
No component may call `fetch` or `axios` directly.

```typescript
// frontend/src/api/client.ts
import axios from 'axios';
import type { Alert, AuditEntry, SuspectMatch } from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
});

// Attach JWT automatically
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('jwt');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export const searchFace = (formData: FormData): Promise<SuspectMatch> =>
  api.post<SuspectMatch>('/api/v1/search', formData).then((r) => r.data);

export const getAlerts = (page = 1): Promise<Alert[]> =>
  api.get<Alert[]>('/api/v1/alerts', { params: { page } }).then((r) => r.data);

export const confirmAlert = (id: number, confirmed: boolean): Promise<void> =>
  api.post(`/api/v1/alerts/${id}/confirm`, { confirmed }).then(() => undefined);
```

### vite.config.ts baseline

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE_URL ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
```

### Frontend type-check gate

Run before every commit:

```bash
cd frontend && npx tsc --noEmit
```

This must exit with code 0. Any type error blocks the commit.

### Required dev dependencies

```json
{
  "devDependencies": {
    "typescript": "^5.4.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@types/leaflet": "^1.9.0",
    "@vitejs/plugin-react": "^4.3.0",
    "vite": "^5.3.0"
  }
}
```

---

## 15. Out-of-Scope — Agents Must Refuse These Tasks

The following are explicitly outside project boundaries. If instructed to implement
any of these, the agent must stop and flag for human review.

- Training any DCNN backbone from scratch.
- Adding autonomous alert dispatch or any mechanism that acts on a match without
  human confirmation.
- Storing raw facial images, video frames, or any biometric data beyond 512-d embeddings.
- Integrating 3D facial reconstruction or thermal/infrared sensor feeds.
- Modifying the `audit_log` table to allow `UPDATE` or `DELETE`.
- Deploying to a public-facing server without a completed security hardening checklist (P6).

---

*Last updated: 2026-05-24*
