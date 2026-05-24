import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react';
import { login as apiLogin, logout as apiLogout, getToken, isTokenExpired, getUserPayload, storeToken } from '../api/client';
import type { AuthContextType, AuthUser, LoginCredentials } from '../types';

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }): ReactNode {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Restore session on mount
  useEffect(() => {
    const token = getToken();
    if (token && !isTokenExpired()) {
      const payload = getUserPayload();
      if (payload && typeof payload === 'object' && 'sub' in payload) {
        setUser(payload as AuthUser);
      } else {
        // Stale payload — clear and redirect
        apiLogout();
      }
    }
    setIsLoading(false);
  }, []);

  const login = useCallback(async (creds: LoginCredentials) => {
    const data = await apiLogin(creds);
    const authUser: AuthUser = {
      sub: creds.username,
      role: 'admin',
      tokenExpiry: Date.now() + data.expiresIn * 1000,
    };
    setUser(authUser);
    storeToken(data.accessToken, data.expiresIn, authUser);
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    apiLogout();
  }, []);

  const isAuthenticated = !!user && !isTokenExpired();

  return (
    <AuthContext.Provider value={{ user, isAuthenticated, login, logout, getToken, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
