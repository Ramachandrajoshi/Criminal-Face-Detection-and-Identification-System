import { useEffect, useState } from 'react';
import { getSuspectImage } from '../utils/db';
import { getToken } from '../api/client';

interface SuspectImageProps {
  name: string;
  style?: React.CSSProperties;
}

export default function SuspectImage({ name, style }: SuspectImageProps): JSX.Element {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let localUrl = '';
    let remoteUrl = '';

    const loadImage = async () => {
      // 1. Try local IndexedDB database
      const blob = await getSuspectImage(name);
      if (!active) return;
      if (blob) {
        localUrl = URL.createObjectURL(blob);
        setSrc(localUrl);
        return;
      }

      // 2. Try backend testdata folder
      try {
        const token = getToken();
        const baseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';
        const response = await fetch(`${baseUrl}/api/v1/face/image/${encodeURIComponent(name)}`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (active && response.ok) {
          const remoteBlob = await response.blob();
          if (active) {
            remoteUrl = URL.createObjectURL(remoteBlob);
            setSrc(remoteUrl);
            return;
          }
        }
      } catch (err) {
        console.error('Failed to fetch suspect image from backend:', err);
      }

      // 3. Fallback to null (show text avatar)
      if (active) {
        setSrc(null);
      }
    };

    void loadImage();

    return () => {
      active = false;
      if (localUrl) URL.revokeObjectURL(localUrl);
      if (remoteUrl) URL.revokeObjectURL(remoteUrl);
    };
  }, [name]);

  if (src) {
    return <img src={src} alt={name} style={{ objectFit: 'cover', borderRadius: '4px', ...style }} />;
  }

  const initials = name
    .split(' ')
    .map((n) => n[0])
    .slice(0, 2)
    .join('')
    .toUpperCase();

  return (
    <div style={{
      width: '40px',
      height: '40px',
      borderRadius: '4px',
      background: 'linear-gradient(135deg, #334155, #1e293b)',
      color: '#38bdf8',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontWeight: 700,
      fontSize: '0.85rem',
      border: '1px solid #38bdf833',
      textShadow: '0 0 4px rgba(56, 189, 248, 0.4)',
      ...style
    }}>
      {initials || '👤'}
    </div>
  );
}
