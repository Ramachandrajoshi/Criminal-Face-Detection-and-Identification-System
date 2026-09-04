# API Reference

Base URL: `http://localhost:8000`

All endpoints except `/api/v1/health` require a Bearer JWT token in the `Authorization` header.

---

## Health

```
GET /api/v1/health
```

**Response:** `200 OK`
```json
{ "status": "ok", "database": "connected" }
```

---

## Register Suspect

```
POST /api/v1/register
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

**Form Fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Face image (JPEG/PNG, ≤ 5 MB) |
| `person_name` | string | Yes | Full name (1–100 chars) |
| `alias` | string | No | Known alias |
| `demographics` | JSON | No | `{ "age_band": "36-60", "gender": "M", "ethnicity": "Caucasian" }` |

**Response:** `201 Created`
```json
{
  "status": "REGISTERED",
  "profile_id": 42,
  "query_hash": "a1b2c3d4...",
  "embedding_dim": 512
}
```

**Errors:**
- `400` — Invalid MIME type or file > 5 MB
- `400` — Missing person_name
- `422` — No face detected in image

---

## Search Face

```
POST /api/v1/search
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

**Form Fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Face image to search |
| `gps_lat` | float | No | GPS latitude |
| `gps_lon` | float | No | GPS longitude |

**Query Params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int | 10 | Max matches (1–50) |

**Response:** `200 OK`
```json
{
  "status": "MATCH",
  "query_hash": "a1b2c3d4...",
  "matches": [
    { "id": 42, "suspectName": "John Doe", "alias": "JD", "distance": 0.32 }
  ],
  "gpsLat": 40.7128,
  "gpsLon": -74.006,
  "matchThreshold": 0.58,
  "alertId": 101
}
```

`status` values: `"MATCH"`, `"NO_MATCH"`, `"SPOOF_BLOCKED"`, `"ERROR"`

---

## Liveness Check

```
POST /api/v1/liveness
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

Standalone anti-spoofing check on a single camera frame — runs DeepFace's
anti-spoofing model and reports real-vs-spoof, without running the
detect → embed → match pipeline and without writing to the audit log. Use
this for a pre-flight liveness check; use `POST /api/v1/search` with
`is_live_capture=true` when you want liveness enforced as part of an actual
search (which also logs `SPOOF_BLOCKED` to the audit trail).

Intended for **live camera captures only** — static photo uploads will
generally score as not-live.

**Form Fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Camera frame to verify (JPEG/PNG, ≤ 5 MB) |

**Response:** `200 OK`
```json
{
  "isLive": true,
  "spoofProbability": 0.04,
  "message": "Live face detected."
}
```

**Errors:**
- `400` — Invalid MIME type, file > 5 MB, or invalid/undersized image

---

## Video Liveness Check

```
POST /api/v1/liveness/video
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

Stronger, multi-frame variant of the liveness check: samples several frames
spread across a short video clip and runs anti-spoofing on each. Harder to
defeat than the single-frame check since an attack (printed photo, or a
phone/monitor replaying a photo or video) has to fool anti-spoofing on
every sampled frame, not just once. Read-only — no audit entry, no face
match.

**Form Fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Short face video — MP4/WebM/QuickTime, **2.5-5.5s** duration (nominal 3-5s + tolerance for encoder rounding), ≤ 5 MB |

**Response:** `200 OK`
```json
{
  "isLive": true,
  "spoofProbability": 0.08,
  "framesAnalyzed": 6,
  "message": "Live face detected across 6 sampled frames."
}
```

**Errors:**
- `400` — Invalid MIME type (must be MP4/WebM/QuickTime), file > 5 MB
- `400` — Duration outside 2.5-5.5s
- `400` — Video unreadable/corrupt or unsupported codec

---

## List Alerts

```
GET /api/v1/alerts?page=1&page_size=20&status_filter=PENDING_REVIEW
Authorization: Bearer <token>
```

**Query Params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `page` | int | 1 | Page number (≥ 1) |
| `page_size` | int | 20 | Items per page (1–100) |
| `status_filter` | string | — | `PENDING_REVIEW`, `CONFIRMED`, `DISMISSED` |

**Response:** `200 OK`
```json
[
  {
    "id": 1,
    "auditLogId": 1,
    "suspectId": 42,
    "eventType": "MATCH",
    "distance": 0.32,
    "status": "PENDING_REVIEW",
    "gpsLat": 40.7128,
    "gpsLon": -74.006,
    "createdAt": "2026-05-24T12:00:00Z",
    "confirmedAt": null
  }
]
```

---

## Confirm Alert

```
POST /api/v1/alerts/{alert_id}/confirm
Authorization: Bearer <token>
Content-Type: application/json
```

**Body:**
```json
{ "confirmed": true }
```

**Response:** `200 OK`
```json
{
  "alert_id": 1,
  "status": "CONFIRMED",
  "confirmed_at": "2026-05-24T12:01:00Z"
}
```

---

## Audit Log

```
GET /api/v1/audit?page=1&page_size=50&event_type=MATCH
Authorization: Bearer <token>
```

**Access:** admin only

**Response:** `200 OK`
```json
[
  {
    "id": 1,
    "eventType": "MATCH",
    "queryHash": "a1b2c3d4...",
    "resultName": "John Doe",
    "distance": 0.32,
    "gpsLat": 40.7128,
    "gpsLon": -74.006,
    "timestamp": "2026-05-24T12:00:00Z"
  }
]
```

---

## Authentication

All endpoints require JWT in the `Authorization` header:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

Token format (HS256):
```json
{ "sub": "user_id", "exp": 1716500000 }
```

Default expiry: 8 hours. Configurable via `JWT_EXPIRY_HOURS`.
