import { useState, useCallback, type FormEvent } from 'react';
import { useAuth } from '../hooks/useAuth';

export default function LoginPage() {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      setError(null);
      setIsSubmitting(true);

      try {
        await login({ username: username.trim(), password });
        // AuthProvider will set the user; App component redirects automatically
      } catch (err: unknown) {
        if (err instanceof Error) {
          if (err.message.includes('401') || err.message.includes('Invalid')) {
            setError('Invalid username or password');
          } else {
            setError(`Login failed: ${err.message}`);
          }
        } else {
          setError('Login failed. Please try again.');
        }
      } finally {
        setIsSubmitting(false);
      }
    },
    [login, username, password]
  );

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      padding: '1rem',
    }}>
      <div style={{
        width: '100%',
        maxWidth: '420px',
        background: '#1e293b',
        borderRadius: '12px',
        padding: '2.5rem',
        boxShadow: '0 25px 60px rgba(0,0,0,0.5)',
        border: '1px solid #334155',
      }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ fontSize: '3rem', marginBottom: '0.5rem' }}>🛡️</div>
          <h1 style={{
            fontSize: '1.5rem',
            fontWeight: 700,
            color: '#38bdf8',
            marginBottom: '0.5rem',
          }}>
            Criminal Detection System
          </h1>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
            Decision Support Platform
          </p>
        </div>

        {/* Disclaimer */}
        <div style={{
          background: 'rgba(245, 158, 11, 0.1)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: '6px',
          padding: '0.75rem',
          marginBottom: '1.5rem',
          fontSize: '0.75rem',
          color: '#f59e0b',
          textAlign: 'center',
        }}>
          ⚖️ All actions are logged. Authorized personnel only.
        </div>

        {/* Error message */}
        {error && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '6px',
            padding: '0.75rem',
            marginBottom: '1rem',
            fontSize: '0.875rem',
            color: '#ef4444',
            textAlign: 'center',
          }}>
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{
              display: 'block',
              fontSize: '0.75rem',
              fontWeight: 600,
              color: '#94a3b8',
              marginBottom: '0.375rem',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}>
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              autoFocus
              disabled={isSubmitting}
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                background: '#0f172a',
                border: '1px solid #334155',
                borderRadius: '6px',
                color: '#e2e8f0',
                fontSize: '0.875rem',
                outline: 'none',
                transition: 'border-color 0.2s',
              }}
              onFocus={(e) => (e.target.style.borderColor = '#38bdf8')}
              onBlur={(e) => (e.target.style.borderColor = '#334155')}
            />
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{
              display: 'block',
              fontSize: '0.75rem',
              fontWeight: 600,
              color: '#94a3b8',
              marginBottom: '0.375rem',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}>
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              disabled={isSubmitting}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSubmit(e as unknown as FormEvent);
              }}
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                background: '#0f172a',
                border: '1px solid #334155',
                borderRadius: '6px',
                color: '#e2e8f0',
                fontSize: '0.875rem',
                outline: 'none',
                transition: 'border-color 0.2s',
              }}
              onFocus={(e) => (e.target.style.borderColor = '#38bdf8')}
              onBlur={(e) => (e.target.style.borderColor = '#334155')}
            />
          </div>

          <button
            type="submit"
            disabled={isSubmitting || !username.trim() || !password.trim()}
            style={{
              width: '100%',
              padding: '0.875rem',
              background: isSubmitting || !username.trim() || !password.trim()
                ? '#475569'
                : '#38bdf8',
              color: '#0f172a',
              border: 'none',
              borderRadius: '6px',
              fontSize: '0.875rem',
              fontWeight: 700,
              cursor: isSubmitting || !username.trim() || !password.trim()
                ? 'not-allowed'
                : 'pointer',
              transition: 'background 0.2s',
              letterSpacing: '0.05em',
              textTransform: 'uppercase',
            }}
          >
            {isSubmitting ? (
              <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                <span
                  style={{
                    display: 'inline-block',
                    width: '16px',
                    height: '16px',
                    border: '2px solid #0f172a',
                    borderTopColor: 'transparent',
                    borderRadius: '50%',
                    animation: 'spin 0.6s linear infinite',
                  }}
                />
                Authenticating…
              </span>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        {/* Default credentials hint */}
        <div style={{
          marginTop: '1.5rem',
          padding: '1rem',
          background: 'rgba(56, 189, 248, 0.05)',
          border: '1px solid rgba(56, 189, 248, 0.2)',
          borderRadius: '6px',
          fontSize: '0.75rem',
          color: '#64748b',
        }}>
          <div style={{ fontWeight: 600, color: '#38bdf8', marginBottom: '0.25rem' }}>
            Default credentials:
          </div>
          <div>
            Username: <code style={{ background: '#0f172a', padding: '0.125rem 0.375rem', borderRadius: '3px' }}>admin</code>
          </div>
          <div>
            Password: <code style={{ background: '#0f172a', padding: '0.125rem 0.375rem', borderRadius: '3px' }}>admin123</code>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
