import { useAlerts } from '../../hooks/useAlerts';
import QueryImage from '../QueryImage';
import SuspectImage from '../SuspectImage';

const statusColor = (s: string): string =>
  s === 'CONFIRMED' ? '#ef4444'
  : s === 'PENDING_REVIEW' ? '#fbbf24'
  : s === 'DISMISSED' ? '#64748b'
  : '#22c55e';

const eventColor = (e: string): string =>
  e === 'MATCH' ? '#ef4444'
  : e === 'NO_MATCH' ? '#22c55e'
  : e === 'SPOOF_BLOCKED' ? '#f59e0b'
  : '#64748b';

export default function AlertsScreen(): JSX.Element {
  const { alerts, loading, error, refresh, confirm } = useAlerts();

  return (
    <div style={{ padding: '2rem', maxWidth: '1000px', margin: '0 auto', overflowY: 'auto', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#e2e8f0', margin: 0 }}>
            Alerts Panel
          </h2>
          <p style={{ color: '#64748b', fontSize: '0.875rem', marginTop: '0.25rem' }}>
            Review and confirm or dismiss identification match alerts
          </p>
        </div>
        <button
          onClick={() => void refresh()}
          style={{
            background: '#334155', color: '#e2e8f0', border: 'none',
            borderRadius: '8px', padding: '0.5rem 1rem', fontSize: '0.8125rem',
            cursor: 'pointer', fontWeight: 600
          }}
        >
          ↻ Refresh
        </button>
      </div>

      {/* Disclaimer */}
      <div style={{
        background: 'rgba(245, 158, 11, 0.08)',
        border: '1px solid rgba(245, 158, 11, 0.3)',
        borderRadius: '8px',
        padding: '0.875rem 1.25rem',
        color: '#f59e0b',
        fontSize: '0.8125rem',
        marginBottom: '1.5rem',
      }}>
        ⚖️ <strong>Decision Support Only</strong> — All match confirmations require human review and positive confirmation before action is taken.
      </div>

      {loading && <div style={{ color: '#64748b', padding: '3rem', textAlign: 'center' }}>Loading alerts…</div>}
      {error && <div style={{ color: '#ef4444', padding: '1rem' }}>Error: {error}</div>}

      {!loading && alerts.length === 0 && (
        <div style={{
          background: 'linear-gradient(135deg, #1e293b, #0f172a)',
          border: '1px solid #334155',
          borderRadius: '12px',
          padding: '4rem',
          textAlign: 'center',
          color: '#475569',
        }}>
          No alerts recorded
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {alerts.map((alert) => (
          <div
            key={alert.id}
            style={{
              background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
              border: alert.status === 'PENDING_REVIEW' ? '1px solid rgba(251, 191, 36, 0.4)' : '1px solid #334155',
              borderRadius: '12px',
              padding: '1.25rem',
              boxShadow: alert.status === 'PENDING_REVIEW' ? '0 4px 20px rgba(251, 191, 36, 0.08)' : 'none',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: '1.5rem',
            }}
          >
            <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'center', flex: 1 }}>
              {/* Image previews */}
              <div style={{ display: 'flex', gap: '0.5rem', flexShrink: 0 }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.2rem' }}>
                  <span style={{ fontSize: '0.6rem', color: '#64748b', fontWeight: 600 }}>QUERY</span>
                  <QueryImage alertId={alert.id} style={{ width: '56px', height: '56px' }} />
                </div>
                {alert.suspectName && (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.2rem' }}>
                    <span style={{ fontSize: '0.6rem', color: '#ef4444', fontWeight: 600 }}>MATCH</span>
                    <SuspectImage name={alert.suspectName} style={{ width: '56px', height: '56px' }} />
                  </div>
                )}
              </div>

              {/* Alert details */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                  <span style={{
                    background: `${eventColor(alert.eventType)}22`,
                    color: eventColor(alert.eventType),
                    padding: '0.125rem 0.5rem',
                    borderRadius: '999px',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                  }}>
                    {alert.eventType}
                  </span>
                  <span style={{ color: '#64748b', fontSize: '0.8125rem', fontFamily: 'monospace' }}>
                    #{alert.id}
                  </span>
                  {alert.suspectName && (
                    <span style={{ color: '#ef4444', fontWeight: 700, fontSize: '0.875rem' }}>
                      👤 {alert.suspectName} {alert.suspectAlias ? `(${alert.suspectAlias})` : ''}
                    </span>
                  )}
                  <span style={{
                    color: statusColor(alert.status),
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    textTransform: 'uppercase',
                  }}>
                    {alert.status}
                  </span>
                </div>

                <div style={{ fontSize: '0.8125rem', color: '#94a3b8' }}>
                  {alert.distance !== null && (
                    <div>Distance: <span style={{ fontFamily: 'monospace' }}>{alert.distance.toFixed(4)}</span></div>
                  )}
                  {alert.gpsLat !== null && alert.gpsLon !== null && (
                    <div>📍 Location: <span style={{ fontFamily: 'monospace' }}>{alert.gpsLat.toFixed(4)}, {alert.gpsLon.toFixed(4)}</span></div>
                  )}
                  <div style={{ color: '#64748b', marginTop: '0.25rem' }}>
                    Time: {new Date(alert.createdAt).toLocaleString()}
                  </div>
                </div>
              </div>
            </div>

            {alert.status === 'PENDING_REVIEW' && (
              <div style={{ display: 'flex', gap: '0.5rem', flexShrink: 0 }}>
                <button
                  onClick={() => void confirm(alert.id, true)}
                  style={{
                    background: '#22c55e', color: '#0f172a', border: 'none',
                    borderRadius: '6px', padding: '0.5rem 1rem', fontSize: '0.8125rem',
                    cursor: 'pointer', fontWeight: 700
                  }}
                >
                  ✓ Confirm
                </button>
                <button
                  onClick={() => void confirm(alert.id, false)}
                  style={{
                    background: 'transparent', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.4)',
                    borderRadius: '6px', padding: '0.5rem 1rem', fontSize: '0.8125rem',
                    cursor: 'pointer', fontWeight: 600
                  }}
                >
                  ✗ Dismiss
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
