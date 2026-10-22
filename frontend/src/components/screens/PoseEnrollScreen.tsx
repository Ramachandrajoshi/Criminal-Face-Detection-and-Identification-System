import { useState, useCallback, useEffect, useRef } from 'react';
import { addFacePoses, estimatePose, poseSearch, registerWithPoses } from '../../api/client';
import type { PoseEstimateResponse, PoseInfo, PoseRegisterResponse, PoseSearchResponse } from '../../types';

// ── Guided pose steps ────────────────────────────────────────────
//
// Angles come from the backend in raw (unmirrored) camera space:
//   yaw > 0   — the subject has turned to *their* left
//   pitch > 0 — looking up
// Pitch targets are relative to the pitch measured on the frontal capture,
// because a laptop webcam usually sits above eye level and reads a "straight"
// face as slightly looking down.

interface PoseTarget {
  yaw: [number, number];
  pitch: [number, number];
}

interface PoseStep {
  id: string;
  label: string;
  hint: string;
  optional?: boolean;
  target: PoseTarget;
}

const STEPS: PoseStep[] = [
  { id: 'front', label: 'Look straight', hint: 'Face the camera directly', target: { yaw: [-10, 10], pitch: [-25, 15] } },
  { id: 'left', label: 'Turn left', hint: 'Turn your head to your left (about 30°)', target: { yaw: [18, 45], pitch: [-15, 15] } },
  { id: 'right', label: 'Turn right', hint: 'Turn your head to your right (about 30°)', target: { yaw: [-45, -18], pitch: [-15, 15] } },
  { id: 'up', label: 'Look up', hint: 'Tilt your chin up', target: { yaw: [-15, 15], pitch: [12, 45] } },
  { id: 'down', label: 'Look down', hint: 'Tilt your chin down', target: { yaw: [-15, 15], pitch: [-45, -12] } },
  { id: 'left_far', label: 'Far left', hint: 'Turn further to your left (about 60°)', optional: true, target: { yaw: [45, 75], pitch: [-20, 20] } },
  { id: 'right_far', label: 'Far right', hint: 'Turn further to your right (about 60°)', optional: true, target: { yaw: [-75, -45], pitch: [-20, 20] } },
];

const POLL_MS = 250;            // pause between pose estimates
const HOLD_FRAMES = 2;          // consecutive in-target estimates before auto-capture
const MAX_FRAME_WIDTH = 960;    // frames are downscaled before upload

interface Frame {
  blob: Blob;
}

interface Capture {
  blob: Blob;
  url: string;
  pose: PoseInfo;
}

type Mode = 'enroll' | 'search';

// ── Helpers ──────────────────────────────────────────────────────

function errMsg(err: unknown): string {
  const ax = err as { response?: { data?: { detail?: unknown } }; message?: string };
  const detail = ax.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d: { msg?: string }) => d.msg ?? '').join(', ');
  if (detail && typeof detail === 'object' && 'error' in detail) {
    const d = detail as { error: string; images?: Array<{ filename: string; error: string | null }> };
    return [d.error, ...(d.images ?? []).map((i) => `${i.filename}: ${i.error}`)].join(' · ');
  }
  return ax.message ?? 'Request failed';
}

function inTarget(pose: PoseInfo, target: PoseTarget, baselinePitch: number): boolean {
  const pitch = pose.pitch - baselinePitch;
  return pose.yaw >= target.yaw[0] && pose.yaw <= target.yaw[1]
    && pitch >= target.pitch[0] && pitch <= target.pitch[1];
}

function guidance(pose: PoseInfo, target: PoseTarget, baselinePitch: number): string {
  const pitch = pose.pitch - baselinePitch;
  if (pose.yaw < target.yaw[0]) return 'Turn more to your left';
  if (pose.yaw > target.yaw[1]) return 'Turn more to your right';
  if (pitch < target.pitch[0]) return 'Tilt your head up';
  if (pitch > target.pitch[1]) return 'Tilt your head down';
  return 'Hold still…';
}

/** Human description of a raw yaw, from the subject's point of view. */
function yawText(yaw: number): string {
  if (Math.abs(yaw) < 3) return 'straight';
  return `${Math.abs(yaw).toFixed(0)}° ${yaw > 0 ? 'your left' : 'your right'}`;
}

// ── Pose gauge ───────────────────────────────────────────────────
// A mirror-view dot: turning to your left moves it left, looking up moves it
// up. The box is the target pose for the current step.

function PoseGauge({ pose, target, baselinePitch }: {
  pose: PoseInfo | null;
  target: PoseTarget | null;
  baselinePitch: number;
}): JSX.Element {
  const size = 160;
  const c = size / 2;
  const scale = c / 80; // ±80° fills the gauge
  const x = (yaw: number) => c - yaw * scale;
  const y = (pitch: number) => c - pitch * scale;
  const hit = pose && target ? inTarget(pose, target, baselinePitch) : false;

  return (
    <svg width={size} height={size} style={{ background: '#0f172a', borderRadius: '12px', border: '1px solid #334155' }}>
      <line x1={c} y1={0} x2={c} y2={size} stroke="#1e293b" />
      <line x1={0} y1={c} x2={size} y2={c} stroke="#1e293b" />
      <text x={6} y={c - 4} fill="#475569" fontSize={9}>your left</text>
      <text x={size - 50} y={c - 4} fill="#475569" fontSize={9}>your right</text>
      <text x={c + 4} y={11} fill="#475569" fontSize={9}>up</text>
      <text x={c + 4} y={size - 4} fill="#475569" fontSize={9}>down</text>
      {target && (
        <rect
          x={x(target.yaw[1])}
          y={y(target.pitch[1])}
          width={(target.yaw[1] - target.yaw[0]) * scale}
          height={(target.pitch[1] - target.pitch[0]) * scale}
          fill="rgba(56,189,248,0.12)"
          stroke="#38bdf8"
          strokeDasharray="4 3"
          rx={4}
        />
      )}
      {pose && (
        <circle
          cx={x(pose.yaw)}
          cy={y(pose.pitch - baselinePitch)}
          r={7}
          fill={hit ? '#22c55e' : '#f59e0b'}
          stroke="#0f172a"
          strokeWidth={2}
          style={{ transition: 'cx 0.2s, cy 0.2s' }}
        />
      )}
    </svg>
  );
}

// ── Screen ───────────────────────────────────────────────────────

export default function PoseEnrollScreen(): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const loopGenRef = useRef(0);

  const [streamActive, setStreamActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>('enroll');

  const [live, setLive] = useState<PoseEstimateResponse | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);

  const [captures, setCaptures] = useState<Record<string, Capture>>({});
  const [stepIdx, setStepIdx] = useState(0);
  const holdRef = useRef(0);

  const [faceName, setFaceName] = useState('');
  const [alias, setAlias] = useState('');
  const [existingFaceId, setExistingFaceId] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [registerResult, setRegisterResult] = useState<PoseRegisterResponse | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const [searching, setSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<PoseSearchResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  const step: PoseStep | null = stepIdx < STEPS.length ? STEPS[stepIdx] : null;
  const baselinePitch = captures.front?.pose.pitch ?? 0;
  const capturedCount = Object.keys(captures).length;
  const requiredDone = STEPS.filter((s) => !s.optional).every((s) => captures[s.id]);

  // ── Camera ─────────────────────────────────────────────────────

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
      setCameraError(null);
    } catch {
      setCameraError('Camera access denied. Allow camera access in the browser and try again.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStreamActive(false);
    setLive(null);
  }, []);

  useEffect(() => () => {
    loopGenRef.current += 1;
    streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  const grabFrame = useCallback(async (): Promise<Frame | null> => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return null;
    if (!canvasRef.current) canvasRef.current = document.createElement('canvas');
    const canvas = canvasRef.current;
    const scale = Math.min(1, MAX_FRAME_WIDTH / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    // Raw, unmirrored frame — the preview is mirrored with CSS only.
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
    return blob ? { blob } : null;
  }, []);

  // ── Auto-capture on each live estimate ────────────────────────
  // Re-assigned every render so the polling loop always sees current state.

  const onEstimateRef = useRef<(est: PoseEstimateResponse, frame: Frame) => void>(() => undefined);
  onEstimateRef.current = (est, frame) => {
    if (mode !== 'enroll' || !step || !est.pose || !est.usable) {
      holdRef.current = 0;
      return;
    }
    if (!inTarget(est.pose, step.target, baselinePitch)) {
      holdRef.current = 0;
      return;
    }
    holdRef.current += 1;
    if (holdRef.current < HOLD_FRAMES) return;

    holdRef.current = 0;
    const capture: Capture = { blob: frame.blob, url: URL.createObjectURL(frame.blob), pose: est.pose };
    const next = { ...captures, [step.id]: capture };
    setCaptures(next);
    const nextIdx = STEPS.findIndex((s) => !next[s.id]);
    setStepIdx(nextIdx === -1 ? STEPS.length : nextIdx);
  };

  // ── Live pose polling loop ─────────────────────────────────────

  useEffect(() => {
    if (!streamActive) return undefined;
    const gen = ++loopGenRef.current;
    let timer: number | undefined;

    const tick = async () => {
      if (gen !== loopGenRef.current) return;
      let delay = POLL_MS;
      const frame = await grabFrame();
      if (frame) {
        try {
          const est = await estimatePose(frame.blob);
          if (gen !== loopGenRef.current) return;
          setLive(est);
          setLiveError(null);
          onEstimateRef.current(est, frame);
        } catch (err) {
          if (gen !== loopGenRef.current) return;
          setLiveError(errMsg(err));
          delay = 1500; // back off while the backend is unhappy
        }
      }
      timer = window.setTimeout(() => void tick(), delay);
    };
    void tick();

    return () => {
      loopGenRef.current += 1;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [streamActive, grabFrame]);

  // ── Step controls ──────────────────────────────────────────────

  const goToStep = useCallback((idx: number) => {
    holdRef.current = 0;
    setStepIdx(idx);
  }, []);

  const skipStep = useCallback(() => {
    holdRef.current = 0;
    setStepIdx((idx) => {
      const later = STEPS.findIndex((s, i) => i > idx && !captures[s.id]);
      return later === -1 ? STEPS.length : later;
    });
  }, [captures]);

  const retake = useCallback((id: string) => {
    setCaptures((prev) => {
      const { [id]: removed, ...rest } = prev;
      if (removed) URL.revokeObjectURL(removed.url);
      return rest;
    });
    goToStep(STEPS.findIndex((s) => s.id === id));
  }, [goToStep]);

  const resetAll = useCallback(() => {
    Object.values(captures).forEach((c) => URL.revokeObjectURL(c.url));
    setCaptures({});
    setRegisterResult(null);
    setSubmitError(null);
    goToStep(0);
  }, [captures, goToStep]);

  useEffect(() => () => {
    Object.values(captures).forEach((c) => URL.revokeObjectURL(c.url));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Submit ─────────────────────────────────────────────────────

  const faceIdNum = existingFaceId.trim() ? Number(existingFaceId.trim()) : null;
  const faceIdInvalid = faceIdNum !== null && (!Number.isInteger(faceIdNum) || faceIdNum < 1);
  const canSubmit = capturedCount > 0 && !submitting && !faceIdInvalid
    && (faceIdNum !== null || faceName.trim().length > 0);

  const submit = useCallback(async () => {
    setSubmitting(true);
    setSubmitError(null);
    setRegisterResult(null);
    try {
      const fd = new FormData();
      STEPS.forEach((s) => {
        const c = captures[s.id];
        if (c) fd.append('files', c.blob, `${s.id}.jpg`);
      });
      let res: PoseRegisterResponse;
      if (faceIdNum !== null) {
        res = await addFacePoses(faceIdNum, fd);
      } else {
        fd.append('face_name', faceName.trim());
        if (alias.trim()) fd.append('alias', alias.trim());
        res = await registerWithPoses(fd);
      }
      setRegisterResult(res);
    } catch (err) {
      setSubmitError(errMsg(err));
    } finally {
      setSubmitting(false);
    }
  }, [captures, faceIdNum, faceName, alias]);

  // ── Search ─────────────────────────────────────────────────────

  const runSearch = useCallback(async () => {
    const frame = await grabFrame();
    if (!frame) return;
    setSearching(true);
    setSearchError(null);
    setSearchResult(null);
    try {
      const fd = new FormData();
      fd.append('file', frame.blob, 'capture.jpg');
      setSearchResult(await poseSearch(fd));
    } catch (err) {
      setSearchError(errMsg(err));
    } finally {
      setSearching(false);
    }
  }, [grabFrame]);

  // ── Render ─────────────────────────────────────────────────────

  const livePose = live?.pose ?? null;
  const liveHit = !!(livePose && step && mode === 'enroll' && inTarget(livePose, step.target, baselinePitch));
  const bbox = live?.bbox;

  const card: React.CSSProperties = {
    background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
    border: '1px solid #334155',
    borderRadius: '16px',
    padding: '1.25rem',
  };

  const modeButton = (m: Mode, label: string) => (
    <button
      key={m}
      onClick={() => { holdRef.current = 0; setMode(m); }}
      style={{
        flex: 1, padding: '0.5rem', borderRadius: '6px', border: 'none',
        fontWeight: 600, fontSize: '0.8125rem', cursor: 'pointer',
        background: mode === m ? '#38bdf8' : '#1e293b',
        color: mode === m ? '#0f172a' : '#94a3b8',
      }}
    >
      {label}
    </button>
  );

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto', overflowY: 'auto', height: '100%', width: '100%' }}>
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#e2e8f0', margin: 0 }}>
          Pose Enrolment
        </h2>
        <p style={{ color: '#64748b', fontSize: '0.875rem', marginTop: '0.25rem' }}>
          Enrol a face from several head angles using the webcam, so CCTV frames taken from the side,
          above or below still match. Frames are captured automatically when you hold each pose.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.6fr) minmax(280px, 1fr)', gap: '1.25rem', alignItems: 'start' }}>
        {/* ── Left: camera ─────────────────────────────────────── */}
        <div style={{ ...card, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {modeButton('enroll', '👤 Enrol poses')}
            {modeButton('search', '🔍 Pose search')}
          </div>

          {/* Instruction banner */}
          {mode === 'enroll' && streamActive && (
            <div style={{
              padding: '0.75rem 1rem', borderRadius: '10px',
              background: step ? (liveHit ? 'rgba(34,197,94,0.12)' : 'rgba(56,189,248,0.08)') : 'rgba(34,197,94,0.12)',
              border: `1px solid ${step ? (liveHit ? '#22c55e66' : '#38bdf844') : '#22c55e66'}`,
            }}>
              {step ? (
                <>
                  <div style={{ fontSize: '1.125rem', fontWeight: 700, color: '#e2e8f0' }}>
                    Step {stepIdx + 1}/{STEPS.length}: {step.label}
                    {step.optional && <span style={{ fontSize: '0.75rem', color: '#64748b', marginLeft: '0.5rem' }}>(optional)</span>}
                  </div>
                  <div style={{ fontSize: '0.875rem', color: liveHit ? '#22c55e' : '#94a3b8', marginTop: '0.2rem' }}>
                    {!live?.faceFound ? 'No face detected — move into the frame'
                      : livePose && !live.usable ? live.reason
                      : livePose ? `${step.hint} — ${guidance(livePose, step.target, baselinePitch)}`
                      : step.hint}
                  </div>
                </>
              ) : (
                <div style={{ fontSize: '1rem', fontWeight: 700, color: '#22c55e' }}>
                  ✓ All poses captured — enter a name and register
                </div>
              )}
            </div>
          )}

          {/* Video + face box. Mirrored together so the preview feels natural. */}
          <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', background: '#0f172a', minHeight: 240 }}>
            <div style={{ position: 'relative', transform: 'scaleX(-1)' }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ display: streamActive ? 'block' : 'none', width: '100%' }}
              />
              {streamActive && bbox && (
                <svg
                  viewBox="0 0 1 1"
                  preserveAspectRatio="none"
                  style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                >
                  <rect
                    x={bbox[0]} y={bbox[1]} width={bbox[2] - bbox[0]} height={bbox[3] - bbox[1]}
                    fill="none"
                    stroke={mode === 'search' ? '#38bdf8' : liveHit ? '#22c55e' : '#f59e0b'}
                    strokeWidth={3}
                    vectorEffect="non-scaling-stroke"
                    rx={0.01}
                  />
                </svg>
              )}
            </div>
            {!streamActive && (
              <div style={{
                height: 360, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: '#475569', fontSize: '0.875rem', padding: '1rem', textAlign: 'center',
              }}>
                {cameraError || 'Camera is off. Click Start Camera.'}
              </div>
            )}
            {(searching || submitting) && (
              <div style={{
                position: 'absolute', inset: 0, background: 'rgba(15,23,42,0.6)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#38bdf8',
              }}>
                <span className="spinner" style={{ marginRight: '0.5rem' }} />
                {searching ? 'Searching…' : 'Registering…'}
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            {!streamActive ? (
              <button className="btn" style={{ background: '#38bdf8', color: '#0f172a', padding: '0.5rem 1.5rem' }}
                      onClick={() => void startCamera()}>
                ▶ Start Camera
              </button>
            ) : (
              <>
                <button className="btn" style={{ background: '#ef4444', color: 'white', padding: '0.5rem 1.5rem' }}
                        onClick={stopCamera}>
                  ⏹ Stop Camera
                </button>
                {mode === 'enroll' && step && (
                  <button className="btn" style={{ background: '#334155', color: '#e2e8f0', padding: '0.5rem 1.25rem' }}
                          onClick={skipStep}>
                    ⏭ Skip this pose
                  </button>
                )}
                {mode === 'search' && (
                  <button className="btn" style={{ background: '#38bdf8', color: '#0f172a', padding: '0.5rem 1.5rem' }}
                          onClick={() => void runSearch()} disabled={searching}>
                    📸 Capture & Pose Search
                  </button>
                )}
              </>
            )}
          </div>

          {liveError && streamActive && (
            <div style={{ color: '#fbbf24', fontSize: '0.8rem' }}>⚠ Pose estimate failed: {liveError}</div>
          )}
        </div>

        {/* ── Right: live pose + captures / results ────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div style={{ ...card, display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <PoseGauge pose={livePose} target={mode === 'enroll' && step ? step.target : null} baselinePitch={baselinePitch} />
            <div style={{ fontSize: '0.8125rem', color: '#94a3b8', display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
              <div style={{ color: '#e2e8f0', fontWeight: 700 }}>Live head pose</div>
              {livePose ? (
                <>
                  <div>Turn: <span style={{ color: '#e2e8f0' }}>{yawText(livePose.yaw)}</span></div>
                  <div>Tilt: <span style={{ color: '#e2e8f0' }}>{(livePose.pitch - baselinePitch).toFixed(0)}°</span>
                    {captures.front && <span style={{ color: '#475569' }}> (vs. straight)</span>}</div>
                  <div>Roll: <span style={{ color: '#e2e8f0' }}>{livePose.roll.toFixed(0)}°</span></div>
                  <div>Bucket: <span style={{ color: '#38bdf8', fontFamily: 'monospace' }}>{livePose.bucket}</span></div>
                  <div>Detector: <span style={{ color: '#e2e8f0' }}>{(livePose.quality * 100).toFixed(0)}%</span></div>
                </>
              ) : (
                <div style={{ color: '#475569' }}>{streamActive ? 'Looking for a face…' : 'Start the camera'}</div>
              )}
            </div>
          </div>

          {mode === 'enroll' && (
            <div style={{ ...card, display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ color: '#e2e8f0', fontWeight: 700 }}>Captured poses ({capturedCount}/{STEPS.length})</div>
                {capturedCount > 0 && (
                  <button onClick={resetAll} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '0.75rem' }}>
                    Reset all
                  </button>
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(92px, 1fr))', gap: '0.5rem' }}>
                {STEPS.map((s, i) => {
                  const cap = captures[s.id];
                  const current = i === stepIdx;
                  return (
                    <div
                      key={s.id}
                      onClick={() => (cap ? retake(s.id) : goToStep(i))}
                      title={cap ? 'Click to retake' : 'Click to capture this pose next'}
                      style={{
                        cursor: 'pointer', borderRadius: '8px', overflow: 'hidden',
                        border: `2px solid ${cap ? '#22c55e' : current ? '#38bdf8' : '#334155'}`,
                        background: '#0f172a',
                      }}
                    >
                      <div style={{ aspectRatio: '1', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {cap ? (
                          <img src={cap.url} alt={s.label}
                               style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} />
                        ) : (
                          <span style={{ color: current ? '#38bdf8' : '#475569', fontSize: '1.25rem' }}>
                            {current ? '◉' : '○'}
                          </span>
                        )}
                      </div>
                      <div style={{ padding: '0.25rem 0.35rem', fontSize: '0.68rem', color: cap ? '#22c55e' : '#94a3b8' }}>
                        {cap ? '✓ ' : ''}{s.label}{s.optional ? '*' : ''}
                        {cap && <div style={{ color: '#64748b', fontFamily: 'monospace' }}>{cap.pose.bucket}</div>}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div style={{ fontSize: '0.7rem', color: '#475569' }}>
                * optional. Click a captured pose to retake it.
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label>Name {faceIdNum !== null ? '(not used when adding to an existing face)' : ''}</label>
                <input type="text" value={faceName} maxLength={100} disabled={faceIdNum !== null}
                       onChange={(e) => setFaceName(e.target.value)} placeholder="Full name" />
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label>Alias (optional)</label>
                <input type="text" value={alias} maxLength={100} disabled={faceIdNum !== null}
                       onChange={(e) => setAlias(e.target.value)} placeholder="Known alias" />
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label>Or add to existing face ID (optional)</label>
                <input type="text" inputMode="numeric" value={existingFaceId}
                       onChange={(e) => setExistingFaceId(e.target.value)} placeholder="e.g. 12" />
                {faceIdInvalid && <div style={{ color: '#ef4444', fontSize: '0.7rem', marginTop: '0.25rem' }}>Enter a positive whole number</div>}
              </div>

              {!requiredDone && capturedCount > 0 && (
                <div style={{ fontSize: '0.75rem', color: '#f59e0b' }}>
                  Tip: capturing all five main poses gives the best CCTV matching.
                </div>
              )}

              <button
                className="btn btn-register"
                onClick={() => void submit()}
                disabled={!canSubmit}
                style={{ opacity: canSubmit ? 1 : 0.5 }}
              >
                {submitting ? 'Registering…'
                  : faceIdNum !== null ? `Add ${capturedCount} pose${capturedCount === 1 ? '' : 's'} to face #${faceIdNum}`
                  : `Register with ${capturedCount} pose${capturedCount === 1 ? '' : 's'}`}
              </button>

              {submitError && (
                <div className="result-box error" style={{ marginTop: 0 }}>
                  <div className="result-label">✗ {submitError}</div>
                </div>
              )}

              {registerResult && (
                <div className="result-box no-match" style={{ marginTop: 0 }}>
                  <div className="result-label">
                    ✓ {registerResult.status === 'REGISTERED' ? 'Registered' : 'Poses added'} — face #{registerResult.profileId}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.4rem' }}>
                    {registerResult.accepted} accepted, {registerResult.rejected} rejected ·
                    poses: <span style={{ fontFamily: 'monospace' }}>{registerResult.poseBuckets.join(', ')}</span>
                  </div>
                  {registerResult.images.filter((i) => i.status === 'REJECTED').map((i) => (
                    <div key={i.filename} style={{ fontSize: '0.72rem', color: '#f59e0b', marginTop: '0.25rem' }}>
                      ✗ {i.filename}: {i.error}
                    </div>
                  ))}
                  <button className="btn" onClick={resetAll}
                          style={{ marginTop: '0.6rem', background: '#334155', color: '#e2e8f0', fontSize: '0.75rem' }}>
                    Enrol another person
                  </button>
                </div>
              )}
            </div>
          )}

          {mode === 'search' && (
            <div style={{ ...card, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div style={{ color: '#e2e8f0', fontWeight: 700 }}>Pose search result</div>
              {!searchResult && !searchError && !searching && (
                <div style={{ fontSize: '0.8125rem', color: '#64748b' }}>
                  Turn your head to any angle and click <strong>Capture & Pose Search</strong>.
                  The frame is matched against every enrolled pose.
                </div>
              )}
              {searchError && <div style={{ color: '#ef4444', fontSize: '0.875rem' }}>✗ {searchError}</div>}
              {searchResult && (
                <div>
                  <div style={{
                    fontWeight: 700, fontSize: '1rem',
                    color: searchResult.status === 'MATCH' ? '#ef4444'
                      : searchResult.status === 'NO_MATCH' ? '#22c55e' : '#f59e0b',
                  }}>
                    {searchResult.status === 'MATCH' && '⚠ MATCH FOUND'}
                    {searchResult.status === 'NO_MATCH' && '✓ NO MATCH'}
                    {searchResult.status === 'LOW_QUALITY' && '⚠ POSE TOO EXTREME — NOT SEARCHED'}
                    {searchResult.status === 'ERROR' && `⚠ ${searchResult.error ?? 'Error'}`}
                  </div>
                  {searchResult.queryPose && (
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.3rem' }}>
                      Query pose: <span style={{ fontFamily: 'monospace', color: '#38bdf8' }}>{searchResult.queryPose.bucket}</span>
                      {' '}({yawText(searchResult.queryPose.yaw)}, pitch {searchResult.queryPose.pitch.toFixed(0)}°)
                    </div>
                  )}
                  {searchResult.status === 'LOW_QUALITY' && searchResult.error && (
                    <div style={{ fontSize: '0.75rem', color: '#f59e0b', marginTop: '0.3rem' }}>{searchResult.error}</div>
                  )}
                  {searchResult.matches.map((m) => (
                    <div key={m.id} style={{
                      marginTop: '0.6rem', padding: '0.75rem', borderRadius: '8px',
                      background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.2)',
                    }}>
                      <div style={{ fontWeight: 700, color: '#ef4444' }}>{m.faceName} <span style={{ color: '#64748b', fontWeight: 400 }}>#{m.id}</span></div>
                      {m.alias && <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Alias: {m.alias}</div>}
                      <div style={{ fontSize: '0.75rem', color: '#64748b', fontFamily: 'monospace', marginTop: '0.2rem' }}>
                        distance {m.distance.toFixed(4)} / threshold {(searchResult.matchThreshold ?? 0).toFixed(2)}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                        matched enrolled pose: <span style={{ fontFamily: 'monospace', color: '#38bdf8' }}>{m.matchedPose}</span>
                      </div>
                    </div>
                  ))}
                  {searchResult.status === 'MATCH' && (
                    <div style={{ marginTop: '0.6rem', fontSize: '0.72rem', color: '#f59e0b' }}>
                      ⚖️ Decision support only — an alert was raised for operator review.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
