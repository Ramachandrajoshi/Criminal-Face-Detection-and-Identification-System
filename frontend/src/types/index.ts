// frontend/src/types/index.ts — canonical shapes

export type EventType = 'MATCH' | 'NO_MATCH' | 'REGISTER' | 'SPOOF_BLOCKED';
export type AlertStatus = 'PENDING_REVIEW' | 'CONFIRMED' | 'DISMISSED';
export type UserRole = 'admin' | 'analyst';

export interface FaceMatch {
  faceName: string;
  alias: string | null;
  distance: number;
  status: AlertStatus;
}

export interface Alert {
  id: number;
  eventType: EventType;
  match: FaceMatch | null;
  gpsLat: number | null;
  gpsLon: number | null;
  timestamp: string;
}

export interface AuditEntry {
  id: number;
  eventType: EventType;
  queryHash: string;
  resultName: string | null;
  distance: number | null;
  timestamp: string;
}

export interface SearchResponse {
  status: 'MATCH' | 'NO_MATCH' | 'SPOOF_BLOCKED' | 'ERROR';
  queryHash: string;
  matches: Array<{
    id: number;
    faceName: string;
    alias: string | null;
    distance: number;
  }>;
  gpsLat: number | null;
  gpsLon: number | null;
  matchThreshold?: number | null;
  alertId?: number | null;
}

export interface RegisterResponse {
  status: 'REGISTERED' | 'ERROR';
  profileId: number | null;
  queryHash: string;
  embeddingDim: number | null;
  error: string | null;
}

export interface AlertItem {
  id: number;
  auditLogId: number | null;
  faceId: number | null;
  faceName?: string | null;
  faceAlias?: string | null;
  eventType: string;
  distance: number | null;
  status: AlertStatus;
  gpsLat: number | null;
  gpsLon: number | null;
  createdAt: string;
  confirmedAt: string | null;
}

// ── Auth types ──────────────────────────────────────────────────

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface TokenResponse {
  accessToken: string;
  tokenType: string;
  expiresIn: number;
}

export interface AuthUser {
  sub: string;
  role: UserRole;
  tokenExpiry: number;  // epoch seconds
}

export interface AuthContextType {
  user: AuthUser | null;
  isAuthenticated: boolean;
  login: (creds: LoginCredentials) => Promise<void>;
  logout: () => void;
  getToken: () => string | null;
  isLoading: boolean;
}

// ── Face CRUD ─────────────────────────────────────────────────────
export interface FaceProfile {
  id: number;
  faceName: string;
  alias: string | null;
  demographics: Record<string, unknown> | null;
  createdAt: string; // ISO-8601
}

// ── Pose-aware enrollment & search (/api/v1/pose/*) ──────────────
// yaw > 0: nose toward image-right of the raw (unmirrored) camera frame,
// i.e. the subject turned to *their* left. pitch > 0: looking up.
export interface PoseInfo {
  yaw: number;
  pitch: number;
  roll: number;
  bucket: string;
  quality: number;
}

export interface PoseEstimateResponse {
  faceFound: boolean;
  pose: PoseInfo | null;
  bbox: [number, number, number, number] | null; // x1, y1, x2, y2 as fractions
  usable: boolean;
  reason: string | null;
}

export interface PoseImageResult {
  filename: string;
  status: 'ACCEPTED' | 'REJECTED';
  poseId: number | null;
  pose: PoseInfo | null;
  error: string | null;
}

export interface PoseRegisterResponse {
  status: 'REGISTERED' | 'POSES_ADDED' | 'POSES_REPLACED';
  profileId: number;
  accepted: number;
  rejected: number;
  poseBuckets: string[];
  images: PoseImageResult[];
}

export interface PoseMatch {
  id: number;
  faceName: string;
  alias: string | null;
  distance: number;
  matchedPose: string;
  matchedYaw: number | null;
  matchedPitch: number | null;
}

export interface PoseSearchResponse {
  status: 'MATCH' | 'NO_MATCH' | 'LOW_QUALITY' | 'ERROR';
  queryHash: string;
  queryPose: PoseInfo | null;
  matches: PoseMatch[];
  matchThreshold: number | null;
  alertId: number | null;
  error: string | null;
}

// ── Batch Search ─────────────────────────────────────────────────
export interface BatchSearchMatch {
  id: number;
  faceName: string;
  alias: string | null;
  distance: number;
}

export interface BatchSearchResultEntry {
  filename: string;
  status: 'MATCH' | 'NO_MATCH' | 'SPOOF_BLOCKED' | 'ERROR' | 'pending' | 'processing';
  queryHash: string;
  matches: BatchSearchMatch[];
  alertId: number | null;
  fileMs?: number;
  error?: string | null;
}

export interface BatchSearchSseDoneEvent {
  type: 'done';
  processed: number;
  total: number;
  matched: number;
  noMatch: number;
  errors: number;
  totalMs: number;
}

export interface BatchSearchSseProgressEvent {
  type: 'start' | 'progress';
  processed: number;
  total: number;
  filename?: string;
  status?: 'MATCH' | 'NO_MATCH' | 'SPOOF_BLOCKED' | 'ERROR';
  queryHash?: string;
  matches?: BatchSearchMatch[];
  alertId?: number | null;
  elapsedMs: number;
  fileMs?: number;
  error?: string | null;
}

export type BatchSearchSseEvent = BatchSearchSseProgressEvent | BatchSearchSseDoneEvent;
