import { useAlerts } from '../hooks/useAlerts';

export default function AlertPanel() {
  const { alerts, loading, error, refresh, confirm } = useAlerts();

  if (loading) return <div style={{ padding: '1.25rem', color: '#64748b' }}>Loading alerts…</div>;
  if (error) return <div style={{ padding: '1.25rem', color: '#ef4444' }}>Error: {error}</div>;

  return (
    <div className="alert-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 style={{ fontSize: '0.875rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em', color: '#94a3b8' }}>
          Alerts
        </h2>
        <button className="btn" onClick={refresh} style={{ background: '#334155', color: '#e2e8f0', fontSize: '0.75rem' }}>
          ↻ Refresh
        </button>
      </div>

      {alerts.length === 0 && (
        <div style={{ color: '#475569', fontSize: '0.875rem', textAlign: 'center', marginTop: '2rem' }}>
          No alerts yet
        </div>
      )}

      {alerts.map((alert) => (
        <div key={alert.id} className="alert-item">
          <div className="alert-header">
            <span style={{ fontSize: '0.8125rem', fontWeight: 600 }}>
              {alert.suspectId ? 'Suspect Match' : 'Detection'}
            </span>
            <span className={`alert-badge ${alert.status.toLowerCase()}`}>
              {alert.status}
            </span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
            {alert.distance !== null && (
              <div>Distance: {alert.distance.toFixed(4)}</div>
            )}
            {alert.gpsLat !== null && alert.gpsLon !== null && (
              <div>📍 {alert.gpsLat.toFixed(4)}, {alert.gpsLon.toFixed(4)}</div>
            )}
            <div>{new Date(alert.createdAt).toLocaleString()}</div>
          </div>
          {alert.status === 'PENDING_REVIEW' && (
            <div className="alert-actions">
              <button className="btn btn-confirm" onClick={() => confirm(alert.id, true)}>
                ✓ Confirm
              </button>
              <button className="btn btn-dismiss" onClick={() => confirm(alert.id, false)}>
                ✗ Dismiss
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
