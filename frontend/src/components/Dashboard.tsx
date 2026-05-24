import { useState, useCallback, useRef, useEffect } from 'react';
import {
  searchFace,
  registerFace,
  registerFacesBatchStream,
} from '../api/client';
import type { SseEvent, SseProgressEvent, SseDoneEvent } from '../api/client';
import type { SearchResponse } from '../types';

// ── Local helpers ────────────────────────────────────────────────

function nameFromFilename(filename: string): string {
  const stem = filename.replace(/\.[^.]+$/, '');
  const cleaned = stem.replace(/[_\-]+/g, ' ').trim();
  return cleaned.replace(/\b\w/g, (c) => c.toUpperCase());
}

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60000);
  const s = Math.round((ms % 60000) / 1000);
  return `${m}m ${s}s`;
}

function fmtEta(remainingMs: number): string {
  if (remainingMs <= 0) return '0s';
  return fmtMs(remainingMs);
}

// ── Types ────────────────────────────────────────────────────────

interface FileEntry {
  file: File;
  name: string;
  preview: string;
  id: string;
}

interface ProgressEntry {
  filename: string;
  suspectName: string;
  status: 'pending' | 'processing' | 'REGISTERED' | 'ERROR' | 'SPOOF_BLOCKED';
  fileMs?: number;
  error?: string | null;
}

interface BatchProgress {
  total: number;
  processed: number;
  registered: number;
  failed: number;
  elapsedMs: number;
  etaMs: number;
  currentFile: string;
  entries: ProgressEntry[];
  done: boolean;
  totalMs?: number;
}

interface DashboardProps {
  onSearchResult?: (result: SearchResponse | null, error: string | null) => void;
}

// ── Status colours ───────────────────────────────────────────────

const STATUS_ICON: Record<string, string> = {
  REGISTERED: '✓',
  ERROR: '✗',
  SPOOF_BLOCKED: '⚠',
  pending: '○',
  processing: '◌',
};

const STATUS_COLOR: Record<string, string> = {
  REGISTERED: '#22c55e',
  ERROR: '#ef4444',
  SPOOF_BLOCKED: '#f59e0b',
  pending: '#475569',
  processing: '#38bdf8',
};

// ── ProgressBar ──────────────────────────────────────────────────

function ProgressBar({ pct, color = '#38bdf8' }: { pct: number; color?: string }) {
  return (
    <div style={{
      width: '100%', height: '8px', background: '#0f172a',
      borderRadius: '999px', overflow: 'hidden',
    }}>
      <div
        style={{
          height: '100%',
          width: `${Math.min(100, pct)}%`,
          background: color,
          borderRadius: '999px',
          transition: 'width 0.35s cubic-bezier(0.4,0,0.2,1)',
          boxShadow: `0 0 8px ${color}88`,
        }}
      />
    </div>
  );
}

// ── Live Progress Panel ──────────────────────────────────────────

function LiveProgressPanel({ progress }: { progress: BatchProgress }) {
  const pct = progress.total > 0
    ? Math.round((progress.processed / progress.total) * 100)
    : 0;

  const barColor = progress.done
    ? (progress.failed === 0 ? '#22c55e' : '#f59e0b')
    : '#38bdf8';

  return (
    <div className="progress-panel">
      {/* Header row */}
      <div className="progress-header">
        <div className="progress-title">
          {progress.done ? (
            progress.failed === 0
              ? <span style={{ color: '#22c55e' }}>✓ Batch Complete</span>
              : <span style={{ color: '#f59e0b' }}>⚠ Batch Complete with errors</span>
          ) : (
            <span style={{ color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
              Processing…
            </span>
          )}
        </div>
        <div className="progress-count">
          <strong style={{ color: '#e2e8f0' }}>{progress.processed}</strong>
          <span style={{ color: '#475569' }}> / {progress.total}</span>
        </div>
      </div>

      {/* Progress bar */}
      <ProgressBar pct={pct} color={barColor} />

      {/* Stats row */}
      <div className="progress-stats">
        <div className="progress-stat">
          <span className="stat-label">Elapsed</span>
          <span className="stat-value">{fmtMs(progress.done ? (progress.totalMs ?? progress.elapsedMs) : progress.elapsedMs)}</span>
        </div>
        {!progress.done && (
          <div className="progress-stat">
            <span className="stat-label">ETA</span>
            <span className="stat-value eta">{fmtEta(progress.etaMs)}</span>
          </div>
        )}
        <div className="progress-stat">
          <span className="stat-label">Rate</span>
          <span className="stat-value">
            {progress.processed > 0 && progress.elapsedMs > 0
              ? `${((progress.processed / progress.elapsedMs) * 1000).toFixed(1)}/s`
              : '—'}
          </span>
        </div>
        <div className="progress-stat">
          <span className="stat-label" style={{ color: '#22c55e' }}>✓</span>
          <span className="stat-value" style={{ color: '#22c55e' }}>{progress.registered}</span>
        </div>
        <div className="progress-stat">
          <span className="stat-label" style={{ color: '#ef4444' }}>✗</span>
          <span className="stat-value" style={{ color: '#ef4444' }}>{progress.failed}</span>
        </div>
      </div>

      {/* Current file */}
      {!progress.done && progress.currentFile && (
        <div className="progress-current">
          <span style={{ color: '#64748b' }}>Processing: </span>
          <span style={{ color: '#94a3b8', fontFamily: 'monospace', fontSize: '0.75rem' }}>
            {progress.currentFile}
          </span>
        </div>
      )}

      {/* Per-file result list */}
      <div className="progress-list">
        {progress.entries.map((e, i) => (
          <div
            key={i}
            className="progress-entry"
            style={{
              opacity: e.status === 'pending' ? 0.45 : 1,
              borderLeft: `3px solid ${STATUS_COLOR[e.status] ?? '#334155'}`,
              transition: 'opacity 0.3s, border-color 0.3s',
            }}
          >
            <span
              className="progress-entry-icon"
              style={{ color: STATUS_COLOR[e.status] ?? '#475569' }}
            >
              {e.status === 'processing'
                ? <span className="spinner" style={{ width: 10, height: 10, borderWidth: 2, display: 'inline-block' }} />
                : STATUS_ICON[e.status]}
            </span>
            <div className="progress-entry-info">
              <span className="progress-entry-name">{e.suspectName}</span>
              <span className="progress-entry-file">{e.filename}</span>
            </div>
            <div className="progress-entry-meta">
              {e.fileMs !== undefined && (
                <span className="progress-entry-time">{fmtMs(e.fileMs)}</span>
              )}
              {e.error && (
                <span style={{ color: '#ef4444', fontSize: '0.65rem' }} title={e.error}>
                  {e.error.slice(0, 28)}{e.error.length > 28 ? '…' : ''}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── SearchPanel ──────────────────────────────────────────────────

interface SearchPanelProps {
  onResult: (r: SearchResponse | null, e: string | null) => void;
  onCameraToggle: () => void;
  cameraMode: boolean;
}

function SearchPanel({ onResult, onCameraToggle, cameraMode }: SearchPanelProps) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const runSearch = useCallback(async (file: File | Blob, isLiveCapture = false) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', file, 'query.jpg');
      // isLiveCapture=true enables anti-spoofing on the backend.
      // Photo uploads always pass false; camera captures pass true.
      const res = await searchFace(formData, isLiveCapture);
      setResult(res);
      onResult(res, null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Search failed';
      setError(msg);
      onResult(null, msg);
    } finally {
      setLoading(false);
    }
  }, [onResult]);


  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    void runSearch(file);
  }, [runSearch, preview]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      if (preview) URL.revokeObjectURL(preview);
      setPreview(URL.createObjectURL(file));
      void runSearch(file);
    }
  }, [runSearch, preview]);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Mode toggle */}
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <button
          id="search-mode-upload"
          onClick={() => { if (cameraMode) onCameraToggle(); }}
          style={{
            flex: 1, padding: '0.5rem', borderRadius: '6px', border: 'none',
            fontWeight: 600, fontSize: '0.8125rem', cursor: 'pointer',
            background: !cameraMode ? '#38bdf8' : '#1e293b',
            color: !cameraMode ? '#0f172a' : '#94a3b8',
            transition: 'all 0.2s',
          }}
        >
          📁 Upload Photo
        </button>
        <button
          id="search-mode-camera"
          onClick={() => { if (!cameraMode) onCameraToggle(); }}
          style={{
            flex: 1, padding: '0.5rem', borderRadius: '6px', border: 'none',
            fontWeight: 600, fontSize: '0.8125rem', cursor: 'pointer',
            background: cameraMode ? '#38bdf8' : '#1e293b',
            color: cameraMode ? '#0f172a' : '#94a3b8',
            transition: 'all 0.2s',
          }}
        >
          📷 Live Camera
        </button>
      </div>

      {!cameraMode && (
        <>
          <div
            id="search-drop-zone"
            className="upload-area"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            style={{ minHeight: '130px', cursor: 'pointer' }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            {preview ? (
              <img
                src={preview}
                alt="Query preview"
                style={{ maxHeight: '110px', maxWidth: '100%', borderRadius: '6px', objectFit: 'cover' }}
              />
            ) : (
              <>
                <div className="upload-icon">🔍</div>
                <div className="upload-text">
                  Drop a photo to search or click to upload
                  <br />
                  <span style={{ fontSize: '0.75rem', color: '#64748b' }}>JPEG / PNG, max 5 MB</span>
                </div>
              </>
            )}
          </div>
          {preview && (
            <button
              id="search-re-upload"
              className="btn"
              onClick={() => inputRef.current?.click()}
              style={{ background: '#334155', color: '#e2e8f0', fontSize: '0.8125rem' }}
            >
              🔄 Choose Different Photo
            </button>
          )}
        </>
      )}

      {cameraMode && (
        <div style={{
          padding: '1rem', background: '#1e293b', borderRadius: '8px',
          textAlign: 'center', color: '#94a3b8', fontSize: '0.875rem',
        }}>
          📷 Camera panel is in the Live Camera tab.
          <br />
          <span style={{ fontSize: '0.75rem', color: '#475569', display: 'block', marginTop: '0.5rem' }}>
            Switch to "Live Camera" to capture frames.
          </span>
        </div>
      )}

      {loading && (
        <div className="result-box" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="spinner" /> Analyzing face…
        </div>
      )}

      {!loading && result && (
        <div className={`result-box ${result.status.toLowerCase().replace(/_/g, '-')}`}>
          <div className="result-label">
            {result.status === 'MATCH' ? '⚠ Match Found'
              : result.status === 'NO_MATCH' ? '✓ No Match'
              : result.status === 'SPOOF_BLOCKED' ? '🎭 Spoof Detected'
              : '⚠ Error'}
          </div>
          <div style={{ marginTop: '0.4rem', fontSize: '0.7rem', color: '#64748b', fontFamily: 'monospace' }}>
            Hash: {result.queryHash.slice(0, 20)}…
          </div>
          {result.status === 'MATCH' && result.matches.length > 0 && (
            <div className="match-list" style={{ marginTop: '0.75rem' }}>
              {result.matches.map((m) => (
                <div key={m.id} className="match-item">
                  <div className="match-name">{m.suspectName}</div>
                  {m.alias && <div style={{ fontSize: '0.75rem', color: '#64748b' }}>{m.alias}</div>}
                  <div className="match-distance">
                    Dist: {m.distance.toFixed(4)} / threshold: {(result.matchThreshold ?? 0.58).toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          )}
          {result.status === 'MATCH' && (
            <div style={{
              marginTop: '0.75rem', padding: '0.5rem',
              background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)',
              borderRadius: '4px', fontSize: '0.7rem', color: '#f59e0b',
            }}>
              ⚠️ Decision support only — confirm via Alerts panel
            </div>
          )}
        </div>
      )}

      {error && !loading && (
        <div className="result-box error">
          <div className="result-label">✗ {error}</div>
        </div>
      )}
    </div>
  );
}

// ── RegisterPanel ────────────────────────────────────────────────

function RegisterPanel() {
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [alias, setAlias] = useState('');
  const [description, setDescription] = useState('');
  const [progress, setProgress] = useState<BatchProgress | null>(null);
  const [singleError, setSingleError] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  // Store cancel function for aborting in-progress stream
  const cancelRef = useRef<(() => void) | null>(null);

  // Clock tick for elapsed time animation while running
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  useEffect(() => () => {
    files.forEach((f) => URL.revokeObjectURL(f.preview));
    cancelRef.current?.();
    if (tickRef.current) clearInterval(tickRef.current);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const addFiles = useCallback((incoming: FileList | null) => {
    if (!incoming) return;
    const entries: FileEntry[] = Array.from(incoming)
      .filter((f) => f.type.startsWith('image/'))
      .map((f) => ({
        file: f,
        name: nameFromFilename(f.name),
        preview: URL.createObjectURL(f),
        id: `${f.name}-${f.lastModified}-${Math.random()}`,
      }));
    setFiles((prev) => [...prev, ...entries]);
    setProgress(null);
    setSingleError(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const handleRemove = useCallback((id: string) => {
    setFiles((prev) => {
      const entry = prev.find((f) => f.id === id);
      if (entry) URL.revokeObjectURL(entry.preview);
      return prev.filter((f) => f.id !== id);
    });
  }, []);

  const handleNameChange = useCallback((id: string, value: string) => {
    setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, name: value } : f)));
  }, []);

  const handleRegister = useCallback(() => {
    if (files.length === 0) return;
    setSingleError(null);

    // Build the initial progress state with all entries as "pending"
    const initialEntries: ProgressEntry[] = files.map((f) => ({
      filename: f.file.name,
      suspectName: f.name.trim() || nameFromFilename(f.file.name),
      status: 'pending',
    }));

    const total = files.length;
    setProgress({
      total,
      processed: 0,
      registered: 0,
      failed: 0,
      elapsedMs: 0,
      etaMs: 0,
      currentFile: '',
      entries: initialEntries,
      done: false,
    });

    // Start elapsed timer
    startTimeRef.current = Date.now();
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(() => {
      const elapsed = Date.now() - startTimeRef.current;
      setProgress((prev) => {
        if (!prev || prev.done) return prev;
        // Recompute ETA based on average ms per file processed so far
        const avgMs = prev.processed > 0 ? elapsed / prev.processed : 0;
        const remaining = prev.total - prev.processed;
        return { ...prev, elapsedMs: elapsed, etaMs: Math.round(avgMs * remaining) };
      });
    }, 200);

    if (files.length === 1) {
      // Single file — use regular endpoint but simulate progress
      const entry = files[0];
      setProgress((prev) => prev ? {
        ...prev,
        currentFile: entry.file.name,
        entries: prev.entries.map((e) =>
          e.filename === entry.file.name ? { ...e, status: 'processing' } : e
        ),
      } : prev);

      const formData = new FormData();
      formData.append('file', entry.file);
      formData.append('suspect_name', entry.name.trim() || nameFromFilename(entry.file.name));
      if (alias.trim()) formData.append('alias', alias.trim());
      if (description.trim()) {
        formData.append('demographics', JSON.stringify({ description: description.trim() }));
      }

      const t0 = Date.now();
      registerFace(formData)
        .then(() => {
          const fileMs = Date.now() - t0;
          if (tickRef.current) clearInterval(tickRef.current);
          const totalMs = Date.now() - startTimeRef.current;
          setProgress((prev) => prev ? {
            ...prev,
            processed: 1,
            registered: 1,
            failed: 0,
            elapsedMs: totalMs,
            etaMs: 0,
            currentFile: '',
            done: true,
            totalMs,
            entries: prev.entries.map((e) =>
              e.filename === entry.file.name
                ? { ...e, status: 'REGISTERED', fileMs }
                : e
            ),
          } : prev);
          // Clear file list on success
          files.forEach((f) => URL.revokeObjectURL(f.preview));
          setFiles([]);
          setAlias('');
          setDescription('');
        })
        .catch((err: unknown) => {
          if (tickRef.current) clearInterval(tickRef.current);
          const fileMs = Date.now() - t0;
          const msg = err instanceof Error ? err.message : 'Registration failed';
          setProgress((prev) => prev ? {
            ...prev,
            processed: 1,
            failed: 1,
            elapsedMs: Date.now() - startTimeRef.current,
            etaMs: 0,
            currentFile: '',
            done: true,
            entries: prev.entries.map((e) =>
              e.filename === entry.file.name
                ? { ...e, status: 'ERROR', fileMs, error: msg }
                : e
            ),
          } : prev);
          setSingleError(msg);
        });

      return;
    }

    // Batch path — SSE stream
    const formData = new FormData();
    files.forEach((entry) => {
      const ext = entry.file.name.split('.').pop() ?? 'jpg';
      const renamedFile = new File(
        [entry.file],
        `${entry.name.trim().replace(/\s+/g, '_')}.${ext}`,
        { type: entry.file.type }
      );
      formData.append('files', renamedFile);
    });
    if (alias.trim()) formData.append('alias', alias.trim());
    if (description.trim()) {
      formData.append('demographics', JSON.stringify({ description: description.trim() }));
    }

    const cancel = registerFacesBatchStream(
      formData,
      (event: SseEvent) => {
        if (event.type === 'start') return;

        if (event.type === 'progress') {
          const ev = event as SseProgressEvent;
          const fileStatus = ev.status ?? 'ERROR';
          setProgress((prev) => {
            if (!prev) return prev;
            const elapsed = Date.now() - startTimeRef.current;
            const avgMs = ev.processed > 0 ? elapsed / ev.processed : 0;
            const remaining = prev.total - ev.processed;
            const etaMs = Math.round(avgMs * remaining);

            // Find the entry by filename (server echoes back the renamed filename)
            const updatedEntries = prev.entries.map((e, idx) => {
              // Match by position (entries are in upload order)
              if (idx === ev.processed - 1) {
                return {
                  ...e,
                  status: fileStatus as ProgressEntry['status'],
                  fileMs: ev.fileMs,
                  error: ev.error ?? undefined,
                };
              }
              // Mark the next one as processing
              if (idx === ev.processed) {
                return { ...e, status: 'processing' as const };
              }
              return e;
            });

            return {
              ...prev,
              processed: ev.processed,
              registered: prev.registered + (fileStatus === 'REGISTERED' ? 1 : 0),
              failed: prev.failed + (fileStatus !== 'REGISTERED' ? 1 : 0),
              elapsedMs: elapsed,
              etaMs,
              currentFile: ev.filename ?? '',
              entries: updatedEntries,
            };
          });
        }

        if (event.type === 'done') {
          const ev = event as SseDoneEvent;
          if (tickRef.current) clearInterval(tickRef.current);
          setProgress((prev) => prev ? {
            ...prev,
            processed: ev.processed,
            registered: ev.registered,
            failed: ev.failed,
            elapsedMs: ev.totalMs,
            etaMs: 0,
            currentFile: '',
            done: true,
            totalMs: ev.totalMs,
            // Ensure all entries are no longer "pending"
            entries: prev.entries.map((e) =>
              e.status === 'pending' || e.status === 'processing'
                ? { ...e, status: 'ERROR', error: 'No response received' }
                : e
            ),
          } : prev);
          // Clear file list
          files.forEach((f) => URL.revokeObjectURL(f.preview));
          setFiles([]);
          setAlias('');
          setDescription('');
        }
      },
      (err: Error) => {
        if (tickRef.current) clearInterval(tickRef.current);
        setSingleError(err.message);
        setProgress((prev) => prev ? { ...prev, done: true, currentFile: '' } : prev);
      }
    );

    cancelRef.current = cancel;
  }, [files, alias, description]);

  const handleCancel = useCallback(() => {
    cancelRef.current?.();
    if (tickRef.current) clearInterval(tickRef.current);
    setProgress((prev) => prev ? { ...prev, done: true, currentFile: '' } : prev);
    setSingleError('Cancelled by user');
  }, []);

  const isRunning = progress !== null && !progress.done;
  const totalReady = files.length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Drop zone — hidden while running */}
      {!isRunning && (
        <div
          id="register-drop-zone"
          className="upload-area"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          style={{ minHeight: '90px', cursor: 'pointer' }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            multiple
            style={{ display: 'none' }}
            onChange={(e) => addFiles(e.target.files)}
          />
          <div className="upload-icon">📂</div>
          <div className="upload-text">
            Drop one or more photos, or click to select
            <br />
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
              Names extracted from filenames • JPEG / PNG, max 5 MB each
            </span>
          </div>
        </div>
      )}

      {/* File list */}
      {!isRunning && files.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '220px', overflowY: 'auto' }}>
          {files.map((entry) => (
            <div
              key={entry.id}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.6rem',
                background: '#1e293b', borderRadius: '8px',
                padding: '0.4rem 0.6rem', border: '1px solid #334155',
              }}
            >
              <img
                src={entry.preview}
                alt={entry.name}
                style={{ width: 40, height: 40, borderRadius: '6px', objectFit: 'cover', flexShrink: 0 }}
              />
              <div style={{ flex: 1, minWidth: 0 }}>
                <input
                  type="text"
                  value={entry.name}
                  onChange={(e) => handleNameChange(entry.id, e.target.value)}
                  placeholder="Suspect name"
                  style={{
                    width: '100%', background: '#0f172a', border: '1px solid #334155',
                    borderRadius: '4px', color: '#e2e8f0', fontSize: '0.8rem',
                    padding: '0.25rem 0.5rem', outline: 'none', boxSizing: 'border-box',
                  }}
                />
                <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '0.15rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {entry.file.name} · {(entry.file.size / 1024).toFixed(0)} KB
                </div>
              </div>
              <button
                id={`remove-${entry.id}`}
                onClick={() => handleRemove(entry.id)}
                title="Remove"
                style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '0.9rem', flexShrink: 0, padding: '0.25rem' }}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Shared fields */}
      {!isRunning && (
        <>
          <div className="form-group">
            <label>Alias (optional)</label>
            <input
              id="register-alias"
              type="text"
              value={alias}
              onChange={(e) => setAlias(e.target.value)}
              placeholder="e.g. street name / nickname"
            />
          </div>
          <div className="form-group">
            <label>Description (optional)</label>
            <input
              id="register-description"
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g. age band, ethnicity, gender"
            />
          </div>
        </>
      )}

      {/* Action buttons */}
      {!isRunning ? (
        <button
          id="register-submit"
          className="btn btn-register"
          onClick={handleRegister}
          disabled={totalReady === 0}
          style={{ opacity: totalReady === 0 ? 0.5 : 1 }}
        >
          Register {totalReady > 0
            ? `${totalReady} Suspect${totalReady > 1 ? 's' : ''}`
            : 'Suspects'}
        </button>
      ) : (
        <button
          id="register-cancel"
          className="btn"
          onClick={handleCancel}
          style={{ background: '#ef4444', color: 'white', width: '100%', padding: '0.5rem' }}
        >
          ✕ Cancel
        </button>
      )}

      {/* Real-time progress panel */}
      {progress && <LiveProgressPanel progress={progress} />}

      {singleError && !isRunning && progress?.done && (
        <div className="result-box error" style={{ marginTop: 0 }}>
          <div className="result-label">✗ {singleError}</div>
        </div>
      )}
    </div>
  );
}

// ── Dashboard (tabbed) ───────────────────────────────────────────

export default function Dashboard(props: DashboardProps): JSX.Element {
  const { onSearchResult } = props;
  const [activePanel, setActivePanel] = useState<'search' | 'register'>('search');
  const [cameraMode, setCameraMode] = useState(false);

  const handleSearchResult = useCallback(
    (result: SearchResponse | null, error: string | null) => {
      onSearchResult?.(result, error);
    },
    [onSearchResult]
  );

  const panelStyle = (active: boolean): React.CSSProperties => ({
    flex: 1,
    padding: '0.625rem 1rem',
    background: active ? '#38bdf8' : 'transparent',
    color: active ? '#0f172a' : '#94a3b8',
    border: 'none',
    borderBottom: active ? '2px solid #38bdf8' : '2px solid transparent',
    cursor: 'pointer',
    fontWeight: 700,
    fontSize: '0.875rem',
    transition: 'all 0.2s',
    borderRadius: active ? '6px 6px 0 0' : '0',
  });

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', borderBottom: '2px solid #334155', marginBottom: '1rem', gap: '0.25rem' }}>
        <button id="panel-tab-search" style={panelStyle(activePanel === 'search')} onClick={() => setActivePanel('search')}>
          🔍 Detection
        </button>
        <button id="panel-tab-register" style={panelStyle(activePanel === 'register')} onClick={() => setActivePanel('register')}>
          ➕ Register
        </button>
      </div>

      {activePanel === 'search' && (
        <SearchPanel
          onResult={handleSearchResult}
          cameraMode={cameraMode}
          onCameraToggle={() => setCameraMode((v) => !v)}
        />
      )}

      {activePanel === 'register' && <RegisterPanel />}
    </div>
  );
}
