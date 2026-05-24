import { useState, useCallback, useEffect } from 'react';
import { useAuth } from './hooks/useAuth';
import Dashboard from './components/Dashboard';
import CameraFeed from './components/CameraFeed';
import AlertPanel from './components/AlertPanel';
import SuspectMap from './components/SuspectMap';
import { useAlerts } from './hooks/useAlerts';
import { searchFace } from './api/client';

export type Tab = 'dashboard' | 'camera' | 'map' | 'alerts';

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
  const [activeTab, setActiveTab] = useState<Tab>('dashboard');
  const [isSearching, setIsSearching] = useState(false);
  const { alerts, refresh } = useAlerts();

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
      // Ignore audio failures (browser permissions, unsupported API)
    }
  }, []);

  const handleFrameCaptured = useCallback(async (blob: Blob, gpsLat?: number | null, gpsLon?: number | null) => {
    setIsSearching(true);
    try {
      const formData = new FormData();
      formData.append('file', blob, 'capture.jpg');
      if (gpsLat !== undefined && gpsLat !== null && gpsLon !== undefined && gpsLon !== null) {
        formData.append('gps_lat', gpsLat.toString());
        formData.append('gps_lon', gpsLon.toString());
      }

      const response = await searchFace(formData, true);

      if (response.status === 'MATCH') {
        playAlertSound();
        await refresh();
      }
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setIsSearching(false);
    }
  }, [playAlertSound, refresh]);

  const handleLogout = useCallback(() => {
    logout();
    window.location.href = '/login';
  }, [logout]);

  const pendingCount = alerts.filter((a: { status: string }) => a.status === 'PENDING_REVIEW').length;

  const mapAlerts = alerts.map((a: { id: number; gpsLat: number | null; gpsLon: number | null; status: string; eventType: string }) => ({
    id: a.id,
    gpsLat: a.gpsLat,
    gpsLon: a.gpsLon,
    status: a.status,
    eventType: a.eventType,
  }));

  const tabs: Array<{ id: Tab; label: string; icon: string }> = [
    { id: 'dashboard', label: 'Dashboard', icon: '📋' },
    { id: 'camera', label: 'Live Camera', icon: '📷' },
    { id: 'map', label: 'Map', icon: '🗺️' },
    { id: 'alerts', label: 'Alerts', icon: '🔔' },
  ];

  return (
    <div className="app-container">
      {/* ── Header ──────────────────────────────────────── */}
      <header className="header">
        <div>
          <h1>⚠ CRIMINAL FACE DETECTION SYSTEM</h1>
          <span className="disclaimer">
            ⚖️ Decision Support Only — Requires Human Confirmation
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{
            fontSize: '0.75rem',
            color: '#94a3b8',
            fontFamily: 'monospace',
          }}>
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

      {/* ── Navigation ──────────────────────────────────── */}
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

      {/* ── Main Content ────────────────────────────────── */}
      <main style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {activeTab === 'dashboard' && (
          <>
            <Dashboard />
            <div style={{ flex: 1, minWidth: 0 }}>
              <SuspectMap alerts={mapAlerts} />
            </div>
            <AlertPanel />
          </>
        )}
        {activeTab === 'camera' && (
          <>
            <Dashboard />
            <div style={{ flex: 1, minWidth: 0, overflow: 'auto' }}>
              <CameraFeed onFrameCaptured={handleFrameCaptured} isSearching={isSearching} />
            </div>
            <AlertPanel />
          </>
        )}
        {activeTab === 'map' && (
          <>
            <div style={{ width: '300px', borderRight: '1px solid #334155', padding: '1rem', overflowY: 'auto' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '1rem',
              }}>
                <h2 style={{
                  fontSize: '0.875rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.1em',
                  color: '#94a3b8',
                }}>
                  Recent Activity
                </h2>
                <button
                  className="btn"
                  onClick={() => refresh()}
                  style={{ background: '#334155', color: '#e2e8f0', fontSize: '0.75rem' }}
                >
                  ↻ Refresh
                </button>
              </div>
              {alerts.slice(0, 20).map((a: { id: number; gpsLat: number | null; gpsLon: number | null; status: string; eventType: string }) => (
                <div key={a.id} style={{
                  background: '#1e293b',
                  borderRadius: '6px',
                  padding: '0.75rem',
                  marginBottom: '0.5rem',
                  fontSize: '0.8125rem',
                  border: a.status === 'PENDING_REVIEW'
                    ? '1px solid rgba(251, 191, 36, 0.3)'
                    : '1px solid #334155',
                }}>
                  <div style={{
                    color: a.status === 'CONFIRMED'
                      ? '#ef4444'
                      : a.status === 'PENDING_REVIEW'
                        ? '#fbbf24'
                        : '#64748b',
                    fontWeight: 600,
                  }}>
                    {a.eventType} — #{a.id}
                  </div>
                  {a.gpsLat !== null && (
                    <div style={{
                      fontSize: '0.75rem',
                      color: '#94a3b8',
                      fontFamily: 'monospace',
                      marginTop: '0.25rem',
                    }}>
                      📍 {a.gpsLat.toFixed(4)}, {a.gpsLon?.toFixed(4)}
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <SuspectMap alerts={mapAlerts} />
            </div>
            <AlertPanel />
          </>
        )}
        {activeTab === 'alerts' && (
          <>
            <Dashboard />
            <div style={{ flex: 1 }} />
            <AlertPanel />
          </>
        )}
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
