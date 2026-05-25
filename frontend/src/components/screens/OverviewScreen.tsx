import { useState, useEffect } from 'react';
import { useAlerts } from '../../hooks/useAlerts';
import { getSuspects } from '../../api/client';
import type { AlertItem } from '../../types';

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

export default function OverviewScreen(): JSX.Element {
  const { alerts, refresh } = useAlerts();
  const [suspectCount, setSuspectCount] = useState<number | null>(null);
  const [loadingCount, setLoadingCount] = useState(true);

  useEffect(() => {
    setLoadingCount(true);
    getSuspects()
      .then((s) => setSuspectCount(s.length))
      .catch(() => setSuspectCount(null))
      .finally(() => setLoadingCount(false));
  }, []);

  const pendingCount = alerts.filter((a: AlertItem) => a.status === 'PENDING_REVIEW').length;
  const matchCount = alerts.filter((a: AlertItem) => a.eventType === 'MATCH').length;
  const recent = alerts.slice(0, 10);

  const cards = [
    {
      icon: '👤',
      label: 'Registered Suspects',
      value: loadingCount ? '…' : (suspectCount ?? '—'),
      color: '#38bdf8',
      glow: 'rgba(56,189,248,0.25)',
    },
    {
      icon: '⚠',
      label: 'Pending Review',
      value: pendingCount,
      color: '#fbbf24',
      glow: 'rgba(251,191,36,0.25)',
    },
    {
      icon: '🎯',
      label: 'Total Matches',
      value: matchCount,
      color: '#ef4444',
      glow: 'rgba(239,68,68,0.25)',
    },
  ] as const;

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto', overflowY: 'auto', height: '100%' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#e2e8f0', margin: 0 }}>
          System Overview
        </h2>
        <p style={{ color: '#64748b', fontSize: '0.875rem', marginTop: '0.25rem' }}>
          Real-time criminal identification decision support platform
        </p>
      </div>

      {/* Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem', marginBottom: '2rem' }}>
        {cards.map((card) => (
          <div
            key={card.label}
            style={{
              background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
              border: `1px solid ${card.color}44`,
              borderRadius: '12px',
              padding: '1.5rem',
              boxShadow: `0 0 24px ${card.glow}`,
              transition: 'transform 0.2s, box-shadow 0.2s',
              cursor: 'default',
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-3px)';
              (e.currentTarget as HTMLDivElement).style.boxShadow = `0 8px 32px ${card.glow}`;
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)';
              (e.currentTarget as HTMLDivElement).style.boxShadow = `0 0 24px ${card.glow}`;
            }}
          >
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>{card.icon}</div>
            <div style={{ fontSize: '2.5rem', fontWeight: 800, color: card.color, fontFamily: 'monospace', lineHeight: 1 }}>
              {card.value}
            </div>
            <div style={{ fontSize: '0.8125rem', color: '#94a3b8', marginTop: '0.5rem', fontWeight: 500 }}>
              {card.label}
            </div>
          </div>
        ))}
      </div>

      {/* Recent Activity Table */}
      <div style={{
        background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
        border: '1px solid #334155',
        borderRadius: '12px',
        overflow: 'hidden',
        marginBottom: '1.5rem',
      }}>
        <div style={{
          padding: '1rem 1.5rem',
          borderBottom: '1px solid #334155',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <h3 style={{ margin: 0, fontSize: '0.9375rem', fontWeight: 600, color: '#e2e8f0' }}>
            Recent Activity
          </h3>
          <button
            onClick={() => void refresh()}
            style={{
              background: '#334155', color: '#e2e8f0', border: 'none',
              borderRadius: '6px', padding: '0.375rem 0.75rem',
              fontSize: '0.75rem', cursor: 'pointer', fontWeight: 600,
              transition: 'background 0.2s',
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = '#475569'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = '#334155'; }}
          >
            ↻ Refresh
          </button>
        </div>

        {recent.length === 0 ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: '#475569' }}>
            No activity yet
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
            <thead>
              <tr style={{ background: 'rgba(51,65,85,0.5)' }}>
                {['ID', 'Event', 'Status', 'Distance', 'Time'].map((h) => (
                  <th key={h} style={{
                    padding: '0.625rem 1.5rem', textAlign: 'left',
                    color: '#64748b', fontWeight: 600, fontSize: '0.75rem',
                    textTransform: 'uppercase', letterSpacing: '0.05em',
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {recent.map((a, i) => (
                <tr
                  key={a.id}
                  style={{
                    background: i % 2 === 0 ? 'transparent' : 'rgba(15,23,42,0.4)',
                    borderTop: '1px solid #1e293b',
                  }}
                >
                  <td style={{ padding: '0.625rem 1.5rem', color: '#64748b', fontFamily: 'monospace' }}>#{a.id}</td>
                  <td style={{ padding: '0.625rem 1.5rem' }}>
                    <span style={{
                      background: `${eventColor(a.eventType)}22`,
                      color: eventColor(a.eventType),
                      padding: '0.125rem 0.5rem',
                      borderRadius: '999px',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                    }}>
                      {a.eventType}
                    </span>
                  </td>
                  <td style={{ padding: '0.625rem 1.5rem' }}>
                    <span style={{ color: statusColor(a.status), fontWeight: 600 }}>{a.status}</span>
                  </td>
                  <td style={{ padding: '0.625rem 1.5rem', color: '#94a3b8', fontFamily: 'monospace' }}>
                    {a.distance !== null ? a.distance.toFixed(4) : '—'}
                  </td>
                  <td style={{ padding: '0.625rem 1.5rem', color: '#64748b' }}>
                    {new Date(a.createdAt).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Disclaimer */}
      <div style={{
        background: 'rgba(245,158,11,0.08)',
        border: '1px solid rgba(245,158,11,0.3)',
        borderRadius: '8px',
        padding: '0.875rem 1.25rem',
        color: '#f59e0b',
        fontSize: '0.8125rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
      }}>
        ⚖️ <strong>Decision Support Only</strong> — All matches require human confirmation before any action is taken.
        References: UK DPA 2018 · Ed Bridges v. South Wales Police · EU AI Act.
      </div>
    </div>
  );
}
