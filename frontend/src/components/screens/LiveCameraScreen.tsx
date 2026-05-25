import { useState, useCallback } from 'react';
import CameraFeed from '../CameraFeed';
import { searchFace } from '../../api/client';
import type { SearchResponse } from '../../types';

export default function LiveCameraScreen(): JSX.Element {
  const [isSearching, setIsSearching] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFrameCaptured = useCallback(async (blob: Blob, gpsLat?: number | null, gpsLon?: number | null) => {
    setIsSearching(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append('file', blob, 'capture.jpg');
      if (gpsLat !== undefined && gpsLat !== null) {
        fd.append('gps_lat', String(gpsLat));
      }
      if (gpsLon !== undefined && gpsLon !== null) {
        fd.append('gps_lon', String(gpsLon));
      }

      // Live captures pass isLiveCapture=true to enforce liveness / anti-spoofing
      const res = await searchFace(fd, true);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setIsSearching(false);
    }
  }, []);

  return (
    <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto', overflowY: 'auto', height: '100%' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#e2e8f0', margin: 0 }}>
          Live Camera Feed
        </h2>
        <p style={{ color: '#64748b', fontSize: '0.875rem', marginTop: '0.25rem' }}>
          Real-time stream capture with anti-spoofing verification
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '2rem', justifyItems: 'center' }}>
        <div style={{
          background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
          border: '1px solid #334155',
          borderRadius: '16px',
          padding: '1.5rem',
          boxShadow: '0 10px 30px rgba(0,0,0,0.3)',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.5rem'
        }}>
          <CameraFeed onFrameCaptured={handleFrameCaptured} isSearching={isSearching} />
        </div>

        {/* Search Results */}
        {(result || error || isSearching) && (
          <div style={{
            width: '100%',
            background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
            border: `1px solid ${
              error ? '#ef444444' 
              : result?.status === 'MATCH' ? '#ef444488' 
              : result?.status === 'NO_MATCH' ? '#22c55e44' 
              : result?.status === 'SPOOF_BLOCKED' ? '#f59e0b44' 
              : '#334155'
            }`,
            borderRadius: '12px',
            padding: '1.5rem',
            boxShadow: result?.status === 'MATCH' ? '0 0 20px rgba(239, 68, 68, 0.15)' : 'none',
          }}>
            {isSearching && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#38bdf8' }}>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                <span>Running face recognition and anti-spoofing checks…</span>
              </div>
            )}

            {error && !isSearching && (
              <div style={{ color: '#ef4444', fontWeight: 600 }}>
                ✗ Error: {error}
              </div>
            )}

            {result && !isSearching && (
              <div>
                <div style={{ 
                  fontSize: '1.125rem', 
                  fontWeight: 700, 
                  color: result.status === 'MATCH' ? '#ef4444' 
                         : result.status === 'NO_MATCH' ? '#22c55e' 
                         : '#f59e0b',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  marginBottom: '0.75rem'
                }}>
                  {result.status === 'MATCH' && '⚠️ SUSPECT MATCH IDENTIFIED'}
                  {result.status === 'NO_MATCH' && '✓ NO MATCH FOUND'}
                  {result.status === 'SPOOF_BLOCKED' && '🎭 SPOOFING ATTEMPT BLOCKED'}
                </div>

                <div style={{ fontSize: '0.75rem', color: '#64748b', fontFamily: 'monospace', marginBottom: '1rem' }}>
                  Query SHA-256: {result.queryHash}
                </div>

                {result.status === 'MATCH' && result.matches.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {result.matches.map((m) => (
                      <div key={m.id} style={{ 
                        background: 'rgba(239, 68, 68, 0.05)', 
                        border: '1px solid rgba(239, 68, 68, 0.2)', 
                        borderRadius: '8px', 
                        padding: '1rem' 
                      }}>
                        <div style={{ fontSize: '1rem', fontWeight: 700, color: '#ef4444' }}>
                          {m.suspectName}
                        </div>
                        {m.alias && (
                          <div style={{ fontSize: '0.8125rem', color: '#94a3b8', marginTop: '0.125rem' }}>
                            Alias: {m.alias}
                          </div>
                        )}
                        <div style={{ fontSize: '0.8125rem', color: '#64748b', marginTop: '0.25rem', fontFamily: 'monospace' }}>
                          Confidence Distance: {m.distance.toFixed(4)} (Threshold: {(result.matchThreshold ?? 0.58).toFixed(2)})
                        </div>
                      </div>
                    ))}
                    
                    <div style={{ 
                      marginTop: '0.5rem', 
                      background: 'rgba(245, 158, 11, 0.08)', 
                      border: '1px solid rgba(245, 158, 11, 0.2)', 
                      borderRadius: '8px', 
                      padding: '0.75rem', 
                      fontSize: '0.8125rem', 
                      color: '#f59e0b' 
                    }}>
                      ⚖️ <strong>Decision Support Only</strong> — Operator review required. Verify match in the Alerts screen.
                    </div>
                  </div>
                )}

                {result.status === 'SPOOF_BLOCKED' && (
                  <div style={{ color: '#f59e0b', fontSize: '0.875rem' }}>
                    The system detected potential face spoofing (presentation attack). This incident has been logged.
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
