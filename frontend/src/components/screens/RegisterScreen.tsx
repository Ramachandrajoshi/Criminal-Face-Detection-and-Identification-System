import { useState, useCallback, useRef, useEffect } from 'react';
import {
  registerFace,
  registerFacesBatchStream,
  getSuspects,
  updateSuspect,
  deleteSuspect,
} from '../../api/client';
import type { SseEvent, SseProgressEvent, SseDoneEvent } from '../../api/client';
import type { SuspectProfile } from '../../types';

// ── Local helpers ────────────────────────────────────────────────

function nameFromFilename(filename: string): string {
  const stem = filename.replace(/\.[^.]+$/, '');
  const cleaned = stem.replace(/[-_]+/g, ' ').trim();
  return cleaned.replace(/\b\w/g, (c) => c.toUpperCase());
}


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
  error?: string;
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

export default function RegisterScreen(): JSX.Element {
  // Roster state
  const [roster, setRoster] = useState<SuspectProfile[]>([]);
  const [loadingRoster, setLoadingRoster] = useState(true);
  const [rosterError, setRosterError] = useState<string | null>(null);

  // Edit modal / inline edit state
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState('');
  const [editAlias, setEditAlias] = useState('');

  // Register panel state
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [alias, setAlias] = useState('');
  const [description, setDescription] = useState('');
  const [progress, setProgress] = useState<BatchProgress | null>(null);
  const [singleError, setSingleError] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const cancelRef = useRef<(() => void) | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  const fetchRoster = useCallback(async () => {
    setLoadingRoster(true);
    setRosterError(null);
    try {
      const data = await getSuspects();
      setRoster(data);
    } catch (err: unknown) {
      setRosterError(err instanceof Error ? err.message : 'Failed to fetch roster');
    } finally {
      setLoadingRoster(false);
    }
  }, []);

  useEffect(() => {
    void fetchRoster();
  }, [fetchRoster]);

  useEffect(() => () => {
    files.forEach((f) => URL.revokeObjectURL(f.preview));
    cancelRef.current?.();
    if (tickRef.current) clearInterval(tickRef.current);
  }, [files]);

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

    startTimeRef.current = Date.now();
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(() => {
      const elapsed = Date.now() - startTimeRef.current;
      setProgress((prev) => {
        if (!prev || prev.done) return prev;
        const avgMs = prev.processed > 0 ? elapsed / prev.processed : 0;
        const remaining = prev.total - prev.processed;
        return { ...prev, elapsedMs: elapsed, etaMs: Math.round(avgMs * remaining) };
      });
    }, 200);

    if (files.length === 1) {
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
          files.forEach((f) => URL.revokeObjectURL(f.preview));
          setFiles([]);
          setAlias('');
          setDescription('');
          void fetchRoster();
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

    // Batch path SSE
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

            const updatedEntries = prev.entries.map((e, idx) => {
              if (idx === ev.processed - 1) {
                return {
                  ...e,
                  status: fileStatus as ProgressEntry['status'],
                  fileMs: ev.fileMs,
                  error: ev.error ?? undefined,
                };
              }
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
            entries: prev.entries.map((e) =>
              e.status === 'pending' || e.status === 'processing'
                ? { ...e, status: 'ERROR', error: 'No response received' }
                : e
            ),
          } : prev);
          files.forEach((f) => URL.revokeObjectURL(f.preview));
          setFiles([]);
          setAlias('');
          setDescription('');
          void fetchRoster();
        }
      },
      (err: Error) => {
        if (tickRef.current) clearInterval(tickRef.current);
        setSingleError(err.message);
        setProgress((prev) => prev ? { ...prev, done: true, currentFile: '' } : prev);
      }
    );

    cancelRef.current = cancel;
  }, [files, alias, description, fetchRoster]);

  const handleCancel = useCallback(() => {
    cancelRef.current?.();
    if (tickRef.current) clearInterval(tickRef.current);
    setProgress((prev) => prev ? { ...prev, done: true, currentFile: '' } : prev);
    setSingleError('Cancelled by user');
  }, []);

  // CRUD actions
  const startEdit = (suspect: SuspectProfile) => {
    setEditingId(suspect.id);
    setEditName(suspect.suspectName);
    setEditAlias(suspect.alias ?? '');
  };

  const saveEdit = async (id: number) => {
    try {
      await updateSuspect(id, { suspectName: editName, alias: editAlias });
      setEditingId(null);
      void fetchRoster();
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Update failed');
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Are you sure you want to delete this suspect profile? This is append-only audit logged.')) {
      return;
    }
    try {
      await deleteSuspect(id);
      void fetchRoster();
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Delete failed');
    }
  };

  const isRunning = progress !== null && !progress.done;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '400px 1fr', gap: '2rem', padding: '2rem', height: '100%', overflow: 'hidden' }}>
      {/* Registration Column */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', overflowY: 'auto' }}>
        <h2 style={{ fontSize: '1.25rem', color: '#e2e8f0', margin: 0, fontWeight: 700 }}>
          Suspect Registration
        </h2>

        {!isRunning && (
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            style={{
              border: '2px dashed #334155', borderRadius: '12px',
              padding: '2rem', textAlign: 'center', cursor: 'pointer',
              background: 'linear-gradient(135deg, #1e293b, #0f172a)',
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              multiple
              style={{ display: 'none' }}
              onChange={(e) => addFiles(e.target.files)}
            />
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📂</div>
            <div style={{ color: '#94a3b8', fontSize: '0.8125rem' }}>
              Drop photos or click to upload
            </div>
          </div>
        )}

        {files.length > 0 && !isRunning && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '180px', overflowY: 'auto' }}>
            {files.map((f) => (
              <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: '#1e293b', padding: '0.4rem', borderRadius: '6px' }}>
                <img src={f.preview} alt="preview" style={{ width: 32, height: 32, borderRadius: '4px', objectFit: 'cover' }} />
                <input
                  type="text"
                  value={f.name}
                  onChange={(e) => handleNameChange(f.id, e.target.value)}
                  style={{ flex: 1, background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#fff', fontSize: '0.75rem', padding: '0.2rem' }}
                />
                <button onClick={() => handleRemove(f.id)} style={{ border: 'none', background: 'none', color: '#ef4444', cursor: 'pointer' }}>✕</button>
              </div>
            ))}
          </div>
        )}

        {!isRunning && (
          <>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <label style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Alias (optional)</label>
              <input
                type="text"
                value={alias}
                onChange={(e) => setAlias(e.target.value)}
                placeholder="e.g. nickname"
                style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '6px', color: '#fff', padding: '0.4rem', fontSize: '0.8125rem' }}
              />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <label style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Description (optional)</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="e.g. height, build"
                style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '6px', color: '#fff', padding: '0.4rem', fontSize: '0.8125rem' }}
              />
            </div>
          </>
        )}

        {!isRunning ? (
          <button
            onClick={handleRegister}
            disabled={files.length === 0}
            style={{
              background: files.length === 0 ? '#1e293b' : 'linear-gradient(135deg, #38bdf8, #0284c7)',
              color: files.length === 0 ? '#64748b' : '#0f172a',
              border: 'none', borderRadius: '8px', padding: '0.6rem', fontWeight: 700, cursor: files.length === 0 ? 'default' : 'pointer'
            }}
          >
            Register Suspects
          </button>
        ) : (
          <button onClick={handleCancel} style={{ background: '#ef4444', color: '#fff', border: 'none', borderRadius: '8px', padding: '0.6rem', cursor: 'pointer' }}>
            ✕ Cancel
          </button>
        )}

        {progress && (
          <div style={{ background: '#1e293b', padding: '1rem', borderRadius: '8px', fontSize: '0.75rem' }}>
            <div>Progress: {progress.processed} / {progress.total}</div>
            <div style={{ color: '#38bdf8', marginTop: '0.25rem' }}>Success: {progress.registered} | Failed: {progress.failed}</div>
            {progress.currentFile && <div style={{ color: '#94a3b8', marginTop: '0.25rem' }}>Active: {progress.currentFile}</div>}
          </div>
        )}

        {singleError && !isRunning && (
          <div style={{ color: '#ef4444', fontSize: '0.75rem', background: 'rgba(239,68,68,0.1)', padding: '0.5rem', borderRadius: '6px' }}>
            {singleError}
          </div>
        )}
      </div>

      {/* Roster Column */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', overflow: 'hidden' }}>
        <h2 style={{ fontSize: '1.25rem', color: '#e2e8f0', margin: 0, fontWeight: 700 }}>
          Registered Suspect Roster
        </h2>

        <div style={{ flex: 1, overflowY: 'auto', border: '1px solid #334155', borderRadius: '12px', background: '#0f172a' }}>
          {loadingRoster ? (
            <div style={{ padding: '2rem', textRendering: 'optimizeSpeed', color: '#64748b', textAlign: 'center' }}>Loading roster…</div>
          ) : rosterError ? (
            <div style={{ padding: '2rem', color: '#ef4444', textAlign: 'center' }}>Error: {rosterError}</div>
          ) : roster.length === 0 ? (
            <div style={{ padding: '2rem', color: '#475569', textAlign: 'center' }}>No suspect profiles found.</div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ background: '#1e293b', borderBottom: '1px solid #334155' }}>
                  <th style={{ padding: '0.75rem' }}>ID</th>
                  <th style={{ padding: '0.75rem' }}>Name</th>
                  <th style={{ padding: '0.75rem' }}>Alias</th>
                  <th style={{ padding: '0.75rem' }}>Created At</th>
                  <th style={{ padding: '0.75rem', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {roster.map((s) => (
                  <tr key={s.id} style={{ borderBottom: '1px solid #1e293b' }}>
                    <td style={{ padding: '0.75rem', color: '#64748b', fontFamily: 'monospace' }}>#{s.id}</td>
                    <td style={{ padding: '0.75rem' }}>
                      {editingId === s.id ? (
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          style={{ background: '#0f172a', border: '1px solid #38bdf8', color: '#fff', fontSize: '0.75rem', padding: '0.2rem', borderRadius: '4px' }}
                        />
                      ) : (
                        s.suspectName
                      )}
                    </td>
                    <td style={{ padding: '0.75rem' }}>
                      {editingId === s.id ? (
                        <input
                          type="text"
                          value={editAlias}
                          onChange={(e) => setEditAlias(e.target.value)}
                          style={{ background: '#0f172a', border: '1px solid #38bdf8', color: '#fff', fontSize: '0.75rem', padding: '0.2rem', borderRadius: '4px' }}
                        />
                      ) : (
                        s.alias || '—'
                      )}
                    </td>
                    <td style={{ padding: '0.75rem', color: '#64748b' }}>{new Date(s.createdAt).toLocaleDateString()}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right' }}>
                      {editingId === s.id ? (
                        <>
                          <button onClick={() => void saveEdit(s.id)} style={{ border: 'none', background: '#22c55e', color: '#0f172a', borderRadius: '4px', padding: '0.2rem 0.4rem', cursor: 'pointer', marginRight: '0.3rem', fontWeight: 600 }}>Save</button>
                          <button onClick={() => setEditingId(null)} style={{ border: 'none', background: '#64748b', color: '#fff', borderRadius: '4px', padding: '0.2rem 0.4rem', cursor: 'pointer' }}>Cancel</button>
                        </>
                      ) : (
                        <>
                          <button onClick={() => startEdit(s)} style={{ border: 'none', background: '#334155', color: '#38bdf8', borderRadius: '4px', padding: '0.2rem 0.4rem', cursor: 'pointer', marginRight: '0.3rem' }}>Edit</button>
                          <button onClick={() => void handleDelete(s.id)} style={{ background: '#2a1a1a', color: '#ef4444', border: '1px solid #ef444433', borderRadius: '4px', padding: '0.2rem 0.4rem', cursor: 'pointer' }}>Delete</button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
