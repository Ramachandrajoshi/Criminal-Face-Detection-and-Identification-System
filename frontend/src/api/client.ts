import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';
import type {
  SearchResponse,
  RegisterResponse,
  AlertItem,
  AuditEntry,
  LoginCredentials,
  TokenResponse,
  SuspectProfile,
  BatchSearchSseEvent,
} from '../types';

// ── Token storage helpers ────────────────────────────────────────

const TOKEN_KEY = 'jwt';
const EXPIRY_KEY = 'jwt_expiry';
const USER_KEY = 'jwt_user';

export function storeToken(token: string, expiresIn: number, userPayload: object): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(EXPIRY_KEY, String(Date.now() + expiresIn * 1000));
  localStorage.setItem(USER_KEY, JSON.stringify({ ...userPayload, expiresAt: Date.now() + expiresIn * 1000 }));
}

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(EXPIRY_KEY);
  localStorage.removeItem(USER_KEY);
}

export function isTokenExpired(): boolean {
  const expiry = localStorage.getItem(EXPIRY_KEY);
  if (!expiry) return true;
  return Date.now() >= Number(expiry);
}

export function getUserPayload(): object | null {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

// ── Axios instance ──────────────────────────────────────────────

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 30000,
});

// ── Request interceptor: attach JWT ─────────────────────────────

api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = getToken();
  if (token && !config.url?.includes('/login') && !config.url?.includes('/health')) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Response interceptor: handle 401 ────────────────────────────

let isRefreshing = false;
type QueueItem = {
  resolve: (value: string) => void;
  reject: (reason?: unknown) => void;
};
let failedQueue: QueueItem[] = [];

function processQueue(error: AxiosError | null, token: string | null = null): void {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else if (token) {
      prom.resolve(token);
    }
  });
  failedQueue = [];
}

api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

    // Never retry login or health endpoints
    if (originalRequest.url?.includes('/login') || originalRequest.url?.includes('/health')) {
      return Promise.reject(error);
    }

    // 401 — token expired or invalid
    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        try {
          const token = await new Promise<string>((resolve, reject) => {
            failedQueue.push({ resolve, reject });
          });
          originalRequest.headers.Authorization = `Bearer ${token}`;
          return api(originalRequest);
        } catch (err) {
          // Refresh failed → force logout
          clearToken();
          window.location.href = '/login';
          return Promise.reject(err);
        }
      }

      originalRequest._retry = true;
      isRefreshing = true;

      const currentToken = getToken();

      try {
        // Try to refresh the token if we have one
        if (currentToken) {
          const refreshResponse = await api.post<any>('/api/v1/token/refresh', {
            access_token: currentToken,
          });
          const { access_token, expires_in } = refreshResponse.data;
          const user = getUserPayload();
          storeToken(access_token, expires_in, user ?? { sub: 'unknown', role: 'admin' });

          api.defaults.headers.common.Authorization = `Bearer ${access_token}`;
          originalRequest.headers.Authorization = `Bearer ${access_token}`;

          processQueue(null, access_token);
          return api(originalRequest);
        } else {
          // No token — force logout and redirect to login
          clearToken();
          processQueue(error, null);
          window.location.href = '/login';
          return Promise.reject(error);
        }
      } catch (_refreshErr) {
        // Refresh failed — force logout
        clearToken();
        processQueue(error, null);
        window.location.href = '/login';
        return Promise.reject(error);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);

// ── Auth API ────────────────────────────────────────────────────

export const login = (creds: LoginCredentials): Promise<TokenResponse> =>
  api.post<any>('/api/v1/login', creds).then((r) => ({
    accessToken: r.data.access_token,
    tokenType: r.data.token_type || 'bearer',
    expiresIn: r.data.expires_in,
  }));

export const refreshToken = (token: string): Promise<TokenResponse> =>
  api.post<any>('/api/v1/token/refresh', { access_token: token }).then((r) => ({
    accessToken: r.data.access_token,
    tokenType: r.data.token_type || 'bearer',
    expiresIn: r.data.expires_in,
  }));

export const logout = (): void => clearToken();

// ── Search ───────────────────────────────────────────────────────

export const searchFace = (
  formData: FormData,
  isLiveCapture = false,
): Promise<SearchResponse> => {
  // Append the liveness flag so the backend knows whether to run anti-spoofing.
  // Photo uploads always pass false; live camera captures pass true.
  formData.append('is_live_capture', String(isLiveCapture));
  return api
    .post<SearchResponse>('/api/v1/search', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data);
};


// ── Register ─────────────────────────────────────────────────────

export const registerFace = (formData: FormData): Promise<RegisterResponse> =>
  api
    .post<RegisterResponse>('/api/v1/register', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data);

export interface BatchFileResult {
  filename: string;
  status: 'REGISTERED' | 'ERROR' | 'SPOOF_BLOCKED';
  profileId: number | null;
  suspectName: string;
  error: string | null;
}

export interface BatchRegisterResponse {
  totalFiles: number;
  registered: number;
  failed: number;
  results: BatchFileResult[];
}

export const registerFacesBatch = (formData: FormData): Promise<BatchRegisterResponse> =>
  api
    .post<BatchRegisterResponse>('/api/v1/register/batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data);

// ── Streaming batch registration (SSE) ───────────────────────────

export type SseEventType = 'start' | 'progress' | 'done';

export interface SseProgressEvent {
  type: 'start' | 'progress';
  processed: number;
  total: number;
  filename?: string;
  suspectName?: string;
  status?: 'REGISTERED' | 'ERROR' | 'SPOOF_BLOCKED';
  profileId?: number | null;
  error?: string | null;
  elapsedMs: number;
  fileMs?: number;
}

export interface SseDoneEvent {
  type: 'done';
  processed: number;
  total: number;
  registered: number;
  failed: number;
  totalMs: number;
}

export type SseEvent = SseProgressEvent | SseDoneEvent;

/**
 * Stream-register multiple files via POST /api/v1/register/batch/stream.
 * Calls `onEvent` for every SSE event received.
 * Returns a cancel function that aborts the request.
 */
export function registerFacesBatchStream(
  formData: FormData,
  onEvent: (event: SseEvent) => void,
  onError?: (err: Error) => void,
): () => void {
  const controller = new AbortController();

  const token = getToken();
  const baseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';

  fetch(`${baseUrl}/api/v1/register/batch/stream`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const text = await response.text().catch(() => response.statusText);
        throw new Error(`Server error ${response.status}: ${text}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE lines look like: "data: {...}\n\n"
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';          // keep incomplete tail

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data:')) continue;
          const json = line.slice('data:'.length).trim();
          try {
            const event = JSON.parse(json) as SseEvent;
            onEvent(event);
          } catch {
            // malformed SSE frame — skip
          }
        }
      }
    })
    .catch((err: unknown) => {
      if (err instanceof Error && err.name === 'AbortError') return; // cancelled
      onError?.(err instanceof Error ? err : new Error(String(err)));
    });

  return () => controller.abort();
}

// ── Alerts ───────────────────────────────────────────────────────

export const getAlerts = (
  page = 1,
  pageSize = 20,
  statusFilter?: string
): Promise<AlertItem[]> =>
  api
    .get<AlertItem[]>('/api/v1/alerts', {
      params: { page, page_size: pageSize, status_filter: statusFilter },
    })
    .then((r) => r.data);

export const confirmAlert = (
  alertId: number,
  confirmed: boolean
): Promise<void> =>
  api
    .post(`/api/v1/alerts/${alertId}/confirm`, { confirmed })
    .then(() => undefined);

// ── Audit ────────────────────────────────────────────────────────

export const getAuditLog = (
  page = 1,
  pageSize = 50,
  eventType?: string
): Promise<AuditEntry[]> =>
  api
    .get<AuditEntry[]>('/api/v1/audit', {
      params: { page, page_size: pageSize, event_type: eventType },
    })
    .then((r) => r.data);

// ── Health ───────────────────────────────────────────────────────

export const healthCheck = (): Promise<{ status: string }> =>
  api.get('/api/v1/health').then((r) => r.data);

// ── Suspects CRUD ────────────────────────────────────────────────

export const getSuspects = (): Promise<SuspectProfile[]> =>
  api.get<SuspectProfile[]>('/api/v1/suspects').then((r) => r.data);

export const getSuspect = (id: number): Promise<SuspectProfile> =>
  api.get<SuspectProfile>(`/api/v1/suspects/${id}`).then((r) => r.data);

export const updateSuspect = (
  id: number,
  patch: { suspectName?: string; alias?: string | null; demographics?: Record<string, unknown> | null },
): Promise<SuspectProfile> =>
  api.patch<SuspectProfile>(`/api/v1/suspects/${id}`, patch).then((r) => r.data);

export const deleteSuspect = (id: number): Promise<void> =>
  api.delete(`/api/v1/suspects/${id}`).then(() => undefined);

// ── Batch Search SSE ─────────────────────────────────────────────

export function searchFacesBatchStream(
  formData: FormData,
  onEvent: (event: BatchSearchSseEvent) => void,
  onError?: (err: Error) => void,
): () => void {
  const controller = new AbortController();
  const token = getToken();
  const baseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';

  fetch(`${baseUrl}/api/v1/search/batch/stream`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const text = await response.text().catch(() => response.statusText);
        throw new Error(`Server error ${response.status}: ${text}`);
      }
      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data:')) continue;
          const json = line.slice('data:'.length).trim();
          try {
            const event = JSON.parse(json) as BatchSearchSseEvent;
            onEvent(event);
          } catch {
            // malformed SSE frame — skip
          }
        }
      }
    })
    .catch((err: unknown) => {
      if (err instanceof Error && err.name === 'AbortError') return;
      onError?.(err instanceof Error ? err : new Error(String(err)));
    });

  return () => controller.abort();
}
