import { useState, useCallback, useRef } from 'react';
import { searchFace, registerFace } from '../api/client';
import type { SearchResponse } from '../types';

interface DashboardProps {
  onSearchResult?: (result: SearchResponse | null, error: string | null) => void;
}

export default function Dashboard(props: DashboardProps): JSX.Element {
  const { onSearchResult } = props;
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [registerName, setRegisterName] = useState('');
  const [registerAlias, setRegisterAlias] = useState('');
  const [registerSuccess, setRegisterSuccess] = useState(false);
  const [registerError, setRegisterError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback(
    async (file: File) => {
      setSearchLoading(true);
      setSearchError(null);
      setSearchResult(null);

      try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await searchFace(formData);
        setSearchResult(response);
        onSearchResult?.(response, null);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Search failed';
        setSearchError(message);
        onSearchResult?.(null, message);
      } finally {
        setSearchLoading(false);
      }
    },
    [onSearchResult]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const handleRegister = useCallback(async () => {
    if (!registerName.trim()) return;
    setRegisterError(null);
    setRegisterSuccess(false);

    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      setRegisterError('Please select an image first');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('suspect_name', registerName.trim());
      if (registerAlias.trim()) {
        formData.append('alias', registerAlias.trim());
      }
      await registerFace(formData);
      setRegisterSuccess(true);
      setTimeout(() => setRegisterSuccess(false), 3000);
      setRegisterName('');
      setRegisterAlias('');
    } catch (err: unknown) {
      setRegisterError(err instanceof Error ? err.message : 'Registration failed');
    }
  }, [registerName, registerAlias]);

  return (
    <div className="card">
      <h2>Face Detection</h2>

      {/* Upload area */}
      <div
        className="upload-area"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/jpg"
          onChange={handleFileInput}
        />
        <div className="upload-icon">📷</div>
        <div className="upload-text">
          Drop an image here or click to upload
          <br />
          <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
            JPEG / PNG, max 5 MB
          </span>
        </div>
      </div>

      {/* Registration form */}
      <div className="form-group">
        <label>Suspect Name</label>
        <input
          type="text"
          value={registerName}
          onChange={(e) => setRegisterName(e.target.value)}
          placeholder="Full name"
        />
      </div>
      <div className="form-group">
        <label>Alias (optional)</label>
        <input
          type="text"
          value={registerAlias}
          onChange={(e) => setRegisterAlias(e.target.value)}
          placeholder="Known alias"
        />
      </div>
      <button
        className="btn btn-register"
        onClick={handleRegister}
        disabled={!registerName.trim()}
      >
        Register Suspect
      </button>

      {registerSuccess && (
        <div className="result-box no-match" style={{ marginTop: '0.75rem' }}>
          <div className="result-label" style={{ color: '#22c55e' }}>
            ✓ Registered
          </div>
        </div>
      )}
      {registerError && (
        <div className="result-box error" style={{ marginTop: '0.75rem' }}>
          <div className="result-label">✗ {registerError}</div>
        </div>
      )}

      {/* Search result */}
      {searchLoading && (
        <div className="result-box" style={{ marginTop: '1rem' }}>
          <span className="spinner" /> Processing face…
        </div>
      )}

      {!searchLoading && searchResult && (
        <div className={`result-box ${searchResult.status.toLowerCase().replace(' ', '-')}`}>
          <div className="result-label">
            {searchResult.status === 'MATCH' ? '⚠ Match Found' :
             searchResult.status === 'NO_MATCH' ? '✓ No Match' :
             searchResult.status === 'SPOOF_BLOCKED' ? '🎭 Spoof Detected' :
             '⚠ Error'}
          </div>
          <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#64748b' }}>
            Query Hash: {searchResult.queryHash.slice(0, 16)}…
          </div>
          {searchResult.status === 'MATCH' && searchResult.matches.length > 0 && (
            <div className="match-list" style={{ marginTop: '0.75rem' }}>
              {searchResult.matches.map((m) => (
                <div key={m.id} className="match-item">
                  <div className="match-name">{m.suspectName}</div>
                  {m.alias && <div style={{ fontSize: '0.75rem', color: '#64748b' }}>{m.alias}</div>}
                  <div className="match-distance">
                    Distance: {m.distance.toFixed(4)} (threshold: 0.58)
                  </div>
                </div>
              ))}
            </div>
          )}
          {searchResult.status === 'MATCH' && (
            <div style={{
              marginTop: '0.75rem',
              padding: '0.5rem',
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '4px',
              fontSize: '0.7rem',
              color: '#f59e0b',
            }}>
              ⚠️ Decision support only — confirm via Alerts panel
            </div>
          )}
        </div>
      )}

      {searchError && !searchLoading && (
        <div className="result-box error" style={{ marginTop: '1rem' }}>
          <div className="result-label">✗ {searchError}</div>
        </div>
      )}
    </div>
  );
}
