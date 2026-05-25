import { useEffect, useState } from 'react';
import { getAlertImage } from '../utils/db';

interface QueryImageProps {
  alertId: number;
  style?: React.CSSProperties;
}

export default function QueryImage({ alertId, style }: QueryImageProps): JSX.Element {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let url = '';
    getAlertImage(alertId).then((blob) => {
      if (!active) return;
      if (blob) {
        url = URL.createObjectURL(blob);
        setSrc(url);
      } else {
        setSrc(null);
      }
    });
    return () => {
      active = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [alertId]);

  if (src) {
    return <img src={src} alt={`Alert #${alertId}`} style={{ objectFit: 'cover', borderRadius: '4px', ...style }} />;
  }

  return (
    <div style={{
      width: '40px',
      height: '40px',
      borderRadius: '4px',
      background: '#1e293b',
      color: '#64748b',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '0.75rem',
      border: '1px solid #334155',
      ...style
    }}>
      🔍
    </div>
  );
}
