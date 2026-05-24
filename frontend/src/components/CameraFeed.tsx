import { useEffect, useRef, useState, useCallback } from 'react';

interface CameraFeedProps {
  onFrameCaptured?: (blob: Blob, gpsLat?: number | null, gpsLon?: number | null) => void;
  isSearching?: boolean;
}

export default function CameraFeed({ onFrameCaptured, isSearching }: CameraFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [streamActive, setStreamActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStreamActive(true);
      setError(null);
    } catch {
      setError('Camera access denied. Please allow camera access and reload.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStreamActive(false);
  }, []);

  const getGps = useCallback(() => new Promise<{ lat: number | null; lon: number | null }>((resolve) => {
    if (!navigator.geolocation) {
      resolve({ lat: null, lon: null });
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      () => resolve({ lat: null, lon: null }),
      { enableHighAccuracy: false, timeout: 3000, maximumAge: 60000 }
    );
  }), []);

  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg')
    );

    if (!blob || !onFrameCaptured) return;

    const gps = await getGps();
    onFrameCaptured(blob, gps.lat, gps.lon);
  }, [getGps, onFrameCaptured]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', padding: '2rem' }}>
      <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', boxShadow: '0 4px 24px rgba(0,0,0,0.5)' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ display: streamActive ? 'block' : 'none', maxWidth: '100%' }}
        />
        {!streamActive && (
          <div style={{
            width: 640, height: 480, background: '#0f172a',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#475569', fontSize: '0.875rem',
          }}>
            {error || 'Camera is off. Click Start Camera.'}
          </div>
        )}
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {/* Overlay when searching */}
        {isSearching && (
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(15, 23, 42, 0.6)', display: 'flex',
            alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{ color: '#38bdf8', fontSize: '1rem' }}>
              <span className="spinner" style={{ marginRight: '0.5rem' }} />
              Analyzing…
            </div>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: '0.75rem' }}>
        {!streamActive ? (
          <button className="btn" style={{ background: '#38bdf8', color: '#0f172a', padding: '0.5rem 1.5rem' }}
                  onClick={startCamera}>
            ▶ Start Camera
          </button>
        ) : (
          <>
            <button className="btn" style={{ background: '#ef4444', color: 'white', padding: '0.5rem 1.5rem' }}
                    onClick={stopCamera}>
              ⏹ Stop Camera
            </button>
            <button className="btn" style={{ background: '#38bdf8', color: '#0f172a', padding: '0.5rem 1.5rem' }}
                  onClick={() => void captureFrame()}
                    disabled={isSearching}>
              📸 Capture & Search
            </button>
          </>
        )}
      </div>
    </div>
  );
}
