import React, { Suspense, lazy, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { AuthProvider, useAuth } from './hooks/useAuth';
import './index.css';

// ── Lazy-load components ────────────────────────────────────────

const App = lazy(() => import('./App'));
const LoginPage = lazy(() => import('./components/Login'));

// ── Loading fallback ────────────────────────────────────────────

function LoadingFallback(): JSX.Element {
  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#0a0e17',
      color: '#38bdf8',
      fontSize: '1rem',
    }}>
      <span
        style={{
          display: 'inline-block',
          width: '20px',
          height: '20px',
          border: '2px solid #475569',
          borderTopColor: '#38bdf8',
          borderRadius: '50%',
          marginRight: '0.75rem',
          animation: 'spin 0.6s linear infinite',
          verticalAlign: 'middle',
        }}
      />
      Loading…
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ── Route resolver ──────────────────────────────────────────────

function AppShell(): JSX.Element {
  const { isAuthenticated, isLoading } = useAuth();
  const isLoginPage = window.location.pathname === '/login';

  useEffect(() => {
    if (isAuthenticated && isLoginPage) {
      window.location.href = '/';
    }
  }, [isAuthenticated, isLoginPage]);

  if (isLoading) {
    return <LoadingFallback />;
  }

  return (
    <Suspense fallback={<LoadingFallback />}>
      {isLoginPage && !isAuthenticated ? <LoginPage /> : <App />}
    </Suspense>
  );
}

// ── Mount ───────────────────────────────────────────────────────

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  </React.StrictMode>
);
