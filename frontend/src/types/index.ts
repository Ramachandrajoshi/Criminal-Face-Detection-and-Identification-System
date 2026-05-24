// frontend/src/types/index.ts — canonical shapes

export type EventType = 'MATCH' | 'NO_MATCH' | 'REGISTER' | 'SPOOF_BLOCKED';
export type AlertStatus = 'PENDING_REVIEW' | 'CONFIRMED' | 'DISMISSED';
export type UserRole = 'admin' | 'analyst';

export interface SuspectMatch {
  suspectName: string;
  alias: string | null;
  distance: number;
  status: AlertStatus;
}

export interface Alert {
  id: number;
  eventType: EventType;
  match: SuspectMatch | null;
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
    suspectName: string;
    alias: string | null;
    distance: number;
  }>;
  gpsLat: number | null;
  gpsLon: number | null;
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
  suspectId: number | null;
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
