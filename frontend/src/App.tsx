import { useState, useCallback, useEffect, useRef } from 'react';
import { useAuth } from './hooks/useAuth';
import { useAlerts } from './hooks/useAlerts';
import OverviewScreen from './components/screens/OverviewScreen';
import SearchScreen from './components/screens/SearchScreen';
import LiveCameraScreen from './components/screens/LiveCameraScreen';
import MapScreen from './components/screens/MapScreen';
import RegisterScreen from './components/screens/RegisterScreen';
import AlertsScreen from './components/screens/AlertsScreen';

export type Tab = 'overview' | 'search' | 'camera' | 'map' | 'register' | 'alerts';

// ── Auth Guard: redirect to login if not authenticated ──────────

function AuthGuard({ children }: { children: React.ReactNode }): JSX.Element {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
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
        Initializing system…
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (!isAuthenticated) {
    window.location.href = '/login';
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0a0e17',
        color: '#38bdf8',
      }}>
        Redirecting to login…
      </div>
    );
  }

  return <>{children}</>;
}

// ── Main App ────────────────────────────────────────────────────

function AppContent(): JSX.Element {
  const { logout, user, isAuthenticated } = useAuth();
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const { alerts } = useAlerts();
  const prevPendingRef = useRef<number>(0);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (isAuthenticated && !user) {
      window.location.href = '/login';
    }
  }, [isAuthenticated, user]);

  const playAlertSound = useCallback(() => {
    try {
      const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: new () => AudioContext }).webkitAudioContext;
      if (!AudioContextCtor) return;
      const ctx = new AudioContextCtor();
      const oscillator = ctx.createOscillator();
      const gain = ctx.createGain();

      oscillator.type = 'sine';
      oscillator.frequency.value = 880;
      gain.gain.value = 0.05;

      oscillator.connect(gain);
      gain.connect(ctx.destination);
      oscillator.start();
      oscillator.stop(ctx.currentTime + 0.25);
      oscillator.onended = () => {
        void ctx.close();
      };
    } catch {
      // Ignore audio failures
    }
  }, []);

  const pendingCount = alerts.filter((a) => a.status === 'PENDING_REVIEW').length;

  // Play alert sound if a new pending alert arrives in the background
  useEffect(() => {
    if (pendingCount > prevPendingRef.current) {
      playAlertSound();
    }
    prevPendingRef.current = pendingCount;
  }, [pendingCount, playAlertSound]);

  const handleLogout = useCallback(() => {
    logout();
    window.location.href = '/login';
  }, [logout]);

  const tabs: Array<{ id: Tab; label: string; icon: string }> = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'search', label: 'Search Suspect', icon: '🔍' },
    { id: 'camera', label: 'Live Feed', icon: '📷' },
    { id: 'map', label: 'GPS Map', icon: '🗺️' },
    { id: 'register', label: 'Register Roster', icon: '👤' },
    { id: 'alerts', label: 'Alerts Panel', icon: '🔔' },
  ];

  return (
    <div className="app-container" style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      {/* Header */}
      <header className="header" style={{
        background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
        padding: '1rem 2rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid #1e293b'
      }}>
        <div>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#38bdf8', letterSpacing: '0.05em', margin: 0 }}>
            ⚠ CRIMINAL FACE DETECTION SYSTEM
          </h1>
          <span className="disclaimer" style={{
            fontSize: '0.75rem',
            color: '#f59e0b',
            background: 'rgba(245, 158, 11, 0.1)',
            padding: '0.375rem 0.75rem',
            borderRadius: '4px',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            display: 'inline-block',
            marginTop: '0.25rem'
          }}>
            ⚖️ Decision Support Only — Requires Human Confirmation
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontFamily: 'monospace' }}>
            👤 {user?.sub ?? 'unknown'}
          </span>
          <button
            onClick={handleLogout}
            style={{
              padding: '0.375rem 0.75rem',
              background: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            🔒 Logout
          </button>
        </div>
      </header>

      {/* Navigation */}
      <nav style={{
        background: '#1e293b',
        display: 'flex',
        borderBottom: '1px solid #334155',
      }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'none',
              border: 'none',
              color: activeTab === tab.id ? '#38bdf8' : '#94a3b8',
              borderBottom: activeTab === tab.id ? '2px solid #38bdf8' : '2px solid transparent',
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              position: 'relative',
              transition: 'color 0.2s',
            }}
          >
            {tab.icon} {tab.label}
            {tab.id === 'alerts' && pendingCount > 0 && (
              <span style={{
                position: 'absolute', top: '0.25rem', right: '0.5rem',
                background: '#ef4444', color: 'white', fontSize: '0.625rem',
                width: '18px', height: '18px', borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                {pendingCount}
              </span>
            )}
          </button>
        ))}
      </nav>

      {/* Main Content */}
      <main style={{ display: 'flex', flex: 1, overflow: 'hidden', background: '#0a0e17' }}>
        {activeTab === 'overview' && <OverviewScreen />}
        {activeTab === 'search' && <SearchScreen />}
        {activeTab === 'camera' && <LiveCameraScreen />}
        {activeTab === 'map' && <MapScreen />}
        {activeTab === 'register' && <RegisterScreen />}
        {activeTab === 'alerts' && <AlertsScreen />}
      </main>
    </div>
  );
}

// ── Root Export ──────────────────────────────────────────────────

export default function App(): JSX.Element {
  return (
    <AuthGuard>
      <AppContent />
    </AuthGuard>
  );
}
