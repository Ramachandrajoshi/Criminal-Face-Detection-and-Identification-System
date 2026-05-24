import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';
import type {
  SearchResponse,
  RegisterResponse,
  AlertItem,
  AuditEntry,
  LoginCredentials,
  TokenResponse,
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
          const refreshResponse = await api.post<TokenResponse>('/api/v1/token/refresh', {
            access_token: currentToken,
          });
          const { accessToken, expiresIn } = refreshResponse.data;
          const user = getUserPayload();
          storeToken(accessToken, expiresIn, user ?? { sub: 'unknown', role: 'admin' });

          api.defaults.headers.common.Authorization = `Bearer ${accessToken}`;
          originalRequest.headers.Authorization = `Bearer ${accessToken}`;

          processQueue(null, accessToken);
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
  api.post<TokenResponse>('/api/v1/login', creds).then((r) => r.data);

export const refreshToken = (token: string): Promise<TokenResponse> =>
  api.post<TokenResponse>('/api/v1/token/refresh', { access_token: token }).then((r) => r.data);

export const logout = (): void => clearToken();

// ── Search ───────────────────────────────────────────────────────

export const searchFace = (formData: FormData): Promise<SearchResponse> =>
  api
    .post<SearchResponse>('/api/v1/search', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data);

// ── Register ─────────────────────────────────────────────────────

export const registerFace = (formData: FormData): Promise<RegisterResponse> =>
  api
    .post<RegisterResponse>('/api/v1/register', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data);

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
