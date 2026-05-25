import { useState, useCallback, useRef, useEffect } from 'react';
import { searchFace, searchFacesBatchStream } from '../../api/client';
import type {
  SearchResponse,
  BatchSearchResultEntry,
  BatchSearchSseEvent,
  BatchSearchSseDoneEvent,
  BatchSearchSseProgressEvent,
} from '../../types';
import { storeAlertImage } from '../../utils/db';
import SuspectImage from '../SuspectImage';

function FileImagePreview({ file, style }: { file?: File; style?: React.CSSProperties }): JSX.Element | null {
  const [src, setSrc] = useState('');
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  if (!src) return null;
  return <img src={src} alt="query preview" style={style} />;
}

// ── Status helpers ────────────────────────────────────────────────

const STATUS_COLOR: Record<string, string> = {
  MATCH: '#ef4444',
  NO_MATCH: '#22c55e',
  SPOOF_BLOCKED: '#f59e0b',
  ERROR: '#ef4444',
  pending: '#475569',
  processing: '#38bdf8',
};

const STATUS_ICON: Record<string, string> = {
  MATCH: '⚠',
  NO_MATCH: '✓',
  SPOOF_BLOCKED: '🎭',
  ERROR: '✗',
  pending: '○',
  processing: '◌',
};

function StatusBadge({ status }: { status: string }): JSX.Element {
  return (
    <span style={{
      background: `${STATUS_COLOR[status] ?? '#475569'}22`,
      color: STATUS_COLOR[status] ?? '#475569',
      padding: '0.15rem 0.6rem',
      borderRadius: '999px',
      fontSize: '0.75rem',
      fontWeight: 700,
      display: 'inline-flex',
      alignItems: 'center',
      gap: '0.3rem',
    }}>
      {status === 'processing'
        ? <span className="spinner" style={{ width: 10, height: 10, borderWidth: 2, display: 'inline-block' }} />
        : STATUS_ICON[status]}
      {' '}{status}
    </span>
  );
}

function ProgressBar({ pct, color = '#38bdf8' }: { pct: number; color?: string }): JSX.Element {
  return (
    <div style={{ width: '100%', height: '6px', background: '#0f172a', borderRadius: '999px', overflow: 'hidden' }}>
      <div style={{
        height: '100%', width: `${Math.min(100, pct)}%`,
        background: color, borderRadius: '999px',
        transition: 'width 0.35s cubic-bezier(0.4,0,0.2,1)',
        boxShadow: `0 0 8px ${color}88`,
      }} />
    </div>
  );
}

// ── Single mode ───────────────────────────────────────────────────

function SingleSearchPanel(): JSX.Element {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const runSearch = useCallback(async (file: File | Blob) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append('file', file, 'query.jpg');
      const res = await searchFace(fd, false);
      setResult(res);
      if (res.status === 'MATCH' && res.alertId && file instanceof Blob) {
        void storeAlertImage(res.alertId, file);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFile = useCallback((file: File) => {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    void runSearch(file);
  }, [runSearch, preview]);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const borderColor = '#334155';

  return (
    <div style={{ maxWidth: '640px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f?.type.startsWith('image/')) handleFile(f); }}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${borderColor}`, borderRadius: '16px',
          padding: '3rem', textAlign: 'center', cursor: 'pointer',
          background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
          transition: 'border-color 0.2s, background 0.2s',
          minHeight: '200px', display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', gap: '1rem',
        }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.borderColor = '#38bdf8'; }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.borderColor = borderColor; }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/jpg"
          style={{ display: 'none' }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        />
        {preview ? (
          <img src={preview} alt="Query" style={{ maxHeight: '160px', maxWidth: '100%', borderRadius: '8px', objectFit: 'cover' }} />
        ) : (
          <>
            <div style={{ fontSize: '3rem' }}>🔍</div>
            <div style={{ color: '#94a3b8', fontSize: '0.9375rem' }}>
              Drop a photo to search or click to upload
              <br />
              <span style={{ fontSize: '0.75rem', color: '#475569' }}>JPEG / PNG, max 5 MB</span>
            </div>
          </>
        )}
      </div>

      {preview && (
        <button
          onClick={() => inputRef.current?.click()}
          style={{ background: '#334155', color: '#e2e8f0', border: 'none', borderRadius: '8px', padding: '0.5rem 1rem', fontSize: '0.8125rem', cursor: 'pointer', fontWeight: 600 }}
        >
          🔄 Choose Different Photo
        </button>
      )}

      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '1.25rem', background: '#1e293b', borderRadius: '10px', color: '#38bdf8' }}>
          <span className="spinner" /> Analyzing face…
        </div>
      )}

      {!loading && result && (
        <div style={{
          background: result.status === 'MATCH'
            ? 'linear-gradient(135deg, #1e293b, #1a0a0a)'
            : 'linear-gradient(135deg, #1e293b, #0a1a0a)',
          border: `1px solid ${result.status === 'MATCH' ? '#ef444466' : result.status === 'NO_MATCH' ? '#22c55e44' : '#f59e0b44'}`,
          borderRadius: '12px',
          padding: '1.25rem',
        }}>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: result.status === 'MATCH' ? '#ef4444' : result.status === 'NO_MATCH' ? '#22c55e' : '#f59e0b' }}>
            {result.status === 'MATCH' ? '⚠ Match Found' : result.status === 'NO_MATCH' ? '✓ No Match Found' : '🎭 Spoof Detected'}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#64748b', fontFamily: 'monospace', marginTop: '0.4rem' }}>
            Hash: {result.queryHash.slice(0, 24)}…
          </div>
          {result.status === 'MATCH' && result.matches.length > 0 && (
            <>
              <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {result.matches.map((m) => (
                  <div key={m.id} style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: '8px', padding: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <SuspectImage name={m.suspectName} style={{ width: '40px', height: '40px' }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 700, color: '#ef4444' }}>{m.suspectName}</div>
                      {m.alias && <div style={{ fontSize: '0.75rem', color: '#64748b' }}>{m.alias}</div>}
                      <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.25rem', fontFamily: 'monospace' }}>
                        Distance: {m.distance.toFixed(4)} / Threshold: {(result.matchThreshold ?? 0.58).toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: '0.75rem', padding: '0.5rem 0.75rem', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: '6px', fontSize: '0.75rem', color: '#f59e0b' }}>
                ⚖️ Decision Support Only — Requires Human Confirmation
              </div>
            </>
          )}
        </div>
      )}

      {error && !loading && (
        <div style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '10px', padding: '1rem', color: '#ef4444', fontWeight: 600 }}>
          ✗ {error}
        </div>
      )}
    </div>
  );
}

// ── Batch mode ────────────────────────────────────────────────────

interface BatchState {
  total: number;
  processed: number;
  matched: number;
  noMatch: number;
  errors: number;
  elapsedMs: number;
  totalMs?: number;
  done: boolean;
  entries: BatchSearchResultEntry[];
}

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function exportCsv(entries: BatchSearchResultEntry[]): void {
  const rows = [
    ['#', 'Filename', 'Status', 'Best Match', 'Distance', 'Alert ID', 'Time (ms)'],
    ...entries.map((e, i) => [
      String(i + 1),
      e.filename,
      e.status,
      e.matches[0]?.suspectName ?? '',
      e.matches[0]?.distance.toFixed(4) ?? '',
      e.alertId !== null ? String(e.alertId) : '',
      e.fileMs !== undefined ? String(e.fileMs) : '',
    ]),
  ];
  const csv = rows.map((r) => r.map((c) => `"${c.replace(/"/g, '""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'batch_search_results.csv';
  a.click();
  URL.revokeObjectURL(url);
}

function BatchSearchPanel(): JSX.Element {
  const [files, setFiles] = useState<File[]>([]);
  const [batch, setBatch] = useState<BatchState | null>(null);
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());
  const cancelRef = useRef<(() => void) | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const isRunning = batch !== null && !batch.done;

  const handleFiles = useCallback((incoming: FileList | null) => {
    if (!incoming) return;
    const arr = Array.from(incoming).filter((f) => f.type.startsWith('image/')).slice(0, 20);
    setFiles((prev) => [...prev, ...arr].slice(0, 20));
    setBatch(null);
  }, []);

  const handleRun = useCallback(() => {
    if (files.length === 0 || isRunning) return;

    const initialEntries: BatchSearchResultEntry[] = files.map((f) => ({
      filename: f.name,
      status: 'pending' as const,
      queryHash: '',
      matches: [],
      alertId: null,
    }));

    setBatch({ total: files.length, processed: 0, matched: 0, noMatch: 0, errors: 0, elapsedMs: 0, done: false, entries: initialEntries });
    setExpandedRows(new Set());

    const fd = new FormData();
    files.forEach((f) => fd.append('files', f));

    const cancel = searchFacesBatchStream(
      fd,
      (event: BatchSearchSseEvent) => {
        if (event.type === 'start') {
          setBatch((prev) => prev ? {
            ...prev,
            entries: prev.entries.map((e, i) => i === 0 ? { ...e, status: 'processing' as const } : e),
          } : prev);
          return;
        }

        if (event.type === 'progress') {
          const ev = event as BatchSearchSseProgressEvent;
          if (ev.status === 'MATCH' && ev.alertId) {
            const file = files[ev.processed - 1];
            if (file) {
              void storeAlertImage(ev.alertId, file);
            }
          }

          setBatch((prev) => {
            if (!prev) return prev;
            const updatedEntries = prev.entries.map((e, idx) => {
              if (idx === ev.processed - 1) {
                return {
                  ...e,
                  status: (ev.status ?? 'ERROR') as BatchSearchResultEntry['status'],
                  queryHash: ev.queryHash ?? '',
                  matches: ev.matches ?? [],
                  alertId: ev.alertId ?? null,
                  fileMs: ev.fileMs,
                  error: ev.error,
                };
              }
              if (idx === ev.processed) return { ...e, status: 'processing' as const };
              return e;
            });
            return {
              ...prev,
              processed: ev.processed,
              matched: prev.matched + (ev.status === 'MATCH' ? 1 : 0),
              noMatch: prev.noMatch + (ev.status === 'NO_MATCH' ? 1 : 0),
              errors: prev.errors + (ev.status === 'ERROR' || ev.status === 'SPOOF_BLOCKED' ? 1 : 0),
              elapsedMs: ev.elapsedMs,
              entries: updatedEntries,
            };
          });
        }

        if (event.type === 'done') {
          const ev = event as BatchSearchSseDoneEvent;
          setBatch((prev) => prev ? {
            ...prev,
            processed: ev.processed,
            matched: ev.matched,
            noMatch: ev.noMatch,
            errors: ev.errors,
            totalMs: ev.totalMs,
            done: true,
            entries: prev.entries.map((e) =>
              e.status === 'pending' || e.status === 'processing'
                ? { ...e, status: 'ERROR' as const, error: 'No response received' }
                : e
            ),
          } : prev);
        }
      },
      (err: Error) => {
        setBatch((prev) => prev ? {
          ...prev,
          done: true,
          entries: prev.entries.map((e) =>
            e.status === 'pending' || e.status === 'processing'
              ? { ...e, status: 'ERROR' as const, error: err.message }
              : e
          ),
        } : prev);
      }
    );
    cancelRef.current = cancel;
  }, [files, isRunning]);

  const toggleRow = useCallback((idx: number) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

  const pct = batch && batch.total > 0 ? Math.round((batch.processed / batch.total) * 100) : 0;
  const dropBorderColor = '#334155';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* File drop zone */}
      {!isRunning && (
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); handleFiles(e.dataTransfer.files); }}
          onClick={() => inputRef.current?.click()}
          style={{
            border: `2px dashed ${dropBorderColor}`, borderRadius: '16px',
            padding: '2.5rem', textAlign: 'center', cursor: 'pointer',
            background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
            transition: 'border-color 0.2s',
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.borderColor = '#38bdf8'; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.borderColor = dropBorderColor; }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            multiple
            style={{ display: 'none' }}
            onChange={(e) => handleFiles(e.target.files)}
          />
          <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>📂</div>
          <div style={{ color: '#94a3b8', fontSize: '0.9375rem' }}>
            Drop up to 20 photos or click to select
            {files.length > 0 && (
              <span style={{ color: '#38bdf8', fontWeight: 700, marginLeft: '0.5rem' }}>
                {files.length} selected
              </span>
            )}
            <br />
            <span style={{ fontSize: '0.75rem', color: '#475569' }}>JPEG / PNG, max 5 MB each</span>
          </div>
        </div>
      )}

      {/* File list preview */}
      {!isRunning && files.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {files.map((f, i) => (
            <div
              key={`${f.name}-${i}`}
              style={{
                background: '#1e293b', border: '1px solid #334155', borderRadius: '8px',
                padding: '0.4rem 0.75rem', fontSize: '0.75rem', color: '#94a3b8',
                display: 'flex', alignItems: 'center', gap: '0.5rem',
              }}
            >
              📄 {f.name}
              <button
                onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: 0, fontSize: '0.875rem' }}
              >
                ✕
              </button>
            </div>
          ))}
          <button
            onClick={() => { setFiles([]); setBatch(null); }}
            style={{ background: 'none', border: '1px solid #334155', borderRadius: '8px', padding: '0.4rem 0.75rem', fontSize: '0.75rem', color: '#64748b', cursor: 'pointer' }}
          >
            Clear all
          </button>
        </div>
      )}

      {/* Action buttons */}
      {!isRunning && (
        <button
          onClick={handleRun}
          disabled={files.length === 0}
          style={{
            background: files.length === 0 ? '#1e293b' : 'linear-gradient(135deg, #0ea5e9, #38bdf8)',
            color: files.length === 0 ? '#475569' : '#0f172a',
            border: 'none', borderRadius: '10px',
            padding: '0.75rem 2rem', fontSize: '0.9375rem',
            fontWeight: 700, cursor: files.length === 0 ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
          }}
        >
          🔍 Run Batch Search{files.length > 0 ? ` (${files.length} photo${files.length > 1 ? 's' : ''})` : ''}
        </button>
      )}

      {isRunning && (
        <button
          onClick={() => { cancelRef.current?.(); setBatch((prev) => prev ? { ...prev, done: true } : prev); }}
          style={{ background: '#ef4444', color: 'white', border: 'none', borderRadius: '10px', padding: '0.625rem 1.5rem', fontSize: '0.875rem', fontWeight: 700, cursor: 'pointer' }}
        >
          ✕ Cancel
        </button>
      )}

      {/* Progress + Results */}
      {batch && (
        <div style={{
          background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
          border: '1px solid #334155', borderRadius: '12px', overflow: 'hidden',
        }}>
          {/* Progress header */}
          <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid #334155' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <div style={{ fontWeight: 700, color: batch.done ? '#22c55e' : '#38bdf8', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {batch.done
                  ? '✓ Complete'
                  : <><span className="spinner" style={{ width: 12, height: 12, borderWidth: 2 }} /> Processing…</>}
              </div>
              <div style={{ fontSize: '0.8125rem', color: '#64748b', fontFamily: 'monospace' }}>
                {batch.processed} / {batch.total}
              </div>
            </div>
            <ProgressBar pct={pct} color={batch.done ? '#22c55e' : '#38bdf8'} />
            <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.75rem', fontSize: '0.75rem' }}>
              <span style={{ color: '#ef4444' }}>⚠ {batch.matched} MATCH</span>
              <span style={{ color: '#22c55e' }}>✓ {batch.noMatch} NO_MATCH</span>
              <span style={{ color: '#f59e0b' }}>✗ {batch.errors} Error</span>
              <span style={{ color: '#64748b' }}>⏱ {fmtMs(batch.done ? (batch.totalMs ?? batch.elapsedMs) : batch.elapsedMs)}</span>
            </div>
          </div>

          {/* Results table */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
              <thead>
                <tr style={{ background: 'rgba(51,65,85,0.5)' }}>
                  {['#', 'Filename', 'Status', 'Best Match', 'Distance', 'Time'].map((h) => (
                    <th key={h} style={{ padding: '0.625rem 1rem', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {batch.entries.map((entry, idx) => (
                  <>
                    <tr
                      key={idx}
                      onClick={() => entry.status === 'MATCH' && entry.matches.length > 1 && toggleRow(idx)}
                      style={{
                        background: idx % 2 === 0 ? 'transparent' : 'rgba(15,23,42,0.4)',
                        borderTop: '1px solid #1e293b',
                        cursor: entry.status === 'MATCH' && entry.matches.length > 1 ? 'pointer' : 'default',
                      }}
                    >
                      <td style={{ padding: '0.625rem 1rem', color: '#64748b', fontFamily: 'monospace' }}>{idx + 1}</td>
                      <td style={{ padding: '0.625rem 1rem', color: '#94a3b8', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <FileImagePreview file={files[idx]} style={{ width: '28px', height: '28px', borderRadius: '4px', objectFit: 'cover' }} />
                          <span>{entry.filename}</span>
                        </div>
                      </td>
                      <td style={{ padding: '0.625rem 1rem' }}>
                        <StatusBadge status={entry.status} />
                      </td>
                      <td style={{ padding: '0.625rem 1rem', color: entry.status === 'MATCH' ? '#ef4444' : '#64748b', fontWeight: entry.status === 'MATCH' ? 700 : 400 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          {entry.status === 'MATCH' && entry.matches[0] && (
                            <SuspectImage name={entry.matches[0].suspectName} style={{ width: '28px', height: '28px' }} />
                          )}
                          <div>
                            <div>{entry.matches[0]?.suspectName ?? '—'}</div>
                            {entry.matches.length > 1 && (
                              <span style={{ color: '#38bdf8', fontSize: '0.7rem' }}>
                                +{entry.matches.length - 1} {expandedRows.has(idx) ? '▲' : '▼'}
                              </span>
                            )}
                          </div>
                        </div>
                      </td>
                      <td style={{ padding: '0.625rem 1rem', color: '#94a3b8', fontFamily: 'monospace' }}>
                        {entry.matches[0] !== undefined ? entry.matches[0].distance.toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '0.625rem 1rem', color: '#64748b', fontFamily: 'monospace' }}>
                        {entry.fileMs !== undefined ? fmtMs(entry.fileMs) : '—'}
                      </td>
                    </tr>
                    {expandedRows.has(idx) && entry.matches.length > 1 && (
                      <tr key={`${idx}-expanded`} style={{ background: 'rgba(239,68,68,0.05)' }}>
                        <td colSpan={6} style={{ padding: '0.5rem 1rem 0.75rem 3rem' }}>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                            {entry.matches.slice(1).map((m) => (
                              <div key={m.id} style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                                <SuspectImage name={m.suspectName} style={{ width: '24px', height: '24px' }} />
                                <span style={{ color: '#ef4444', fontWeight: 600 }}>{m.suspectName}</span>
                                {m.alias && <span style={{ color: '#64748b' }}>{m.alias}</span>}
                                <span style={{ fontFamily: 'monospace' }}>{m.distance.toFixed(4)}</span>
                              </div>
                            ))}
                          </div>
                        </td>
                      </tr>
                    )}
                    {entry.error && (
                      <tr key={`${idx}-error`} style={{ background: 'rgba(239,68,68,0.05)' }}>
                        <td colSpan={6} style={{ padding: '0.25rem 1rem 0.5rem 3rem', fontSize: '0.7rem', color: '#ef4444' }}>
                          ✗ {entry.error}
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>

          {/* Export CSV */}
          {batch.done && (
            <div style={{ padding: '1rem 1.5rem', borderTop: '1px solid #334155', display: 'flex', justifyContent: 'flex-end', gap: '1rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.75rem', color: '#f59e0b' }}>⚖️ Decision Support Only — confirm matches via Alerts</span>
              <button
                onClick={() => exportCsv(batch.entries)}
                style={{ background: '#334155', color: '#e2e8f0', border: 'none', borderRadius: '8px', padding: '0.5rem 1rem', fontSize: '0.8125rem', cursor: 'pointer', fontWeight: 600 }}
              >
                ⬇ Export CSV
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main SearchScreen ─────────────────────────────────────────────

export default function SearchScreen(): JSX.Element {
  const [mode, setMode] = useState<'single' | 'batch'>('single');

  return (
    <div style={{ padding: '2rem', maxWidth: '1000px', margin: '0 auto', overflowY: 'auto', height: '100%' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#e2e8f0', margin: 0 }}>Face Search</h2>
        <p style={{ color: '#64748b', fontSize: '0.875rem', marginTop: '0.25rem' }}>
          Search the database by uploading a photo or a batch of photos
        </p>
      </div>

      {/* Mode toggle */}
      <div style={{ display: 'flex', gap: '0.25rem', marginBottom: '2rem', background: '#0f172a', padding: '0.25rem', borderRadius: '10px', width: 'fit-content' }}>
        {(['single', 'batch'] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            style={{
              padding: '0.5rem 1.25rem', borderRadius: '8px', border: 'none',
              fontWeight: 600, fontSize: '0.875rem', cursor: 'pointer',
              background: mode === m ? '#38bdf8' : 'transparent',
              color: mode === m ? '#0f172a' : '#94a3b8',
              transition: 'all 0.2s',
            }}
          >
            {m === 'single' ? '📁 Single Photo' : '📂 Batch Photos'}
          </button>
        ))}
      </div>

      {mode === 'single' ? <SingleSearchPanel /> : <BatchSearchPanel />}
    </div>
  );
}
