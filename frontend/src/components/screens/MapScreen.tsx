import { useAlerts } from '../../hooks/useAlerts';
import SuspectMap from '../SuspectMap';

export default function MapScreen(): JSX.Element {
  const { alerts, loading, error, refresh } = useAlerts();

  // Filter alerts that have GPS locations
  const gpsAlerts = alerts.filter(
    (a) => a.gpsLat !== null && a.gpsLon !== null
  );

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      {/* Sidebar */}
      <div style={{
        width: '320px',
        borderRight: '1px solid #334155',
        background: '#0f172a',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
      }}>
        <div style={{
          padding: '1.25rem',
          borderBottom: '1px solid #334155',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '1rem', color: '#e2e8f0' }}>GPS Alerts</h3>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
              {gpsAlerts.length} locations mapped
            </span>
          </div>
          <button
            onClick={() => void refresh()}
            style={{
              background: '#334155', color: '#e2e8f0', border: 'none',
              borderRadius: '6px', padding: '0.375rem 0.75rem',
              fontSize: '0.75rem', cursor: 'pointer', fontWeight: 600
            }}
          >
            ↻
          </button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '0.75rem' }}>
          {loading && <div style={{ color: '#64748b', padding: '1rem', textAlign: 'center' }}>Loading…</div>}
          {error && <div style={{ color: '#ef4444', padding: '1rem' }}>{error}</div>}
          {!loading && gpsAlerts.length === 0 && (
            <div style={{ color: '#475569', textAlign: 'center', padding: '2rem', fontSize: '0.875rem' }}>
              No GPS-enabled alert locations recorded yet.
            </div>
          )}

          {!loading && gpsAlerts.map((alert) => (
            <div
              key={alert.id}
              style={{
                background: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '8px',
                padding: '0.75rem',
                marginBottom: '0.5rem',
                fontSize: '0.8125rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, color: alert.status === 'CONFIRMED' ? '#ef4444' : '#f59e0b' }}>
                  {alert.eventType}
                </span>
                <span style={{
                  fontSize: '0.7rem',
                  color: alert.status === 'CONFIRMED' ? '#ef4444' : alert.status === 'PENDING_REVIEW' ? '#fbbf24' : '#64748b',
                  fontWeight: 700
                }}>
                  {alert.status}
                </span>
              </div>
              <div style={{ color: '#94a3b8', fontSize: '0.75rem', marginTop: '0.25rem', fontFamily: 'monospace' }}>
                📍 {alert.gpsLat?.toFixed(4)}, {alert.gpsLon?.toFixed(4)}
              </div>
              <div style={{ color: '#64748b', fontSize: '0.7rem', marginTop: '0.25rem' }}>
                {new Date(alert.createdAt).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Map area */}
      <div style={{ flex: 1, height: '100%', position: 'relative' }}>
        <SuspectMap alerts={gpsAlerts} />
      </div>
    </div>
  );
}
