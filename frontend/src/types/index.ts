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
