import { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface MapLocation {
  id: number;
  gpsLat: number | null;
  gpsLon: number | null;
  status: string;
  eventType: string;
}

interface SuspectMapProps {
  alerts: MapLocation[];
}

// ── Color mapping based on alert status ────────────────────────

function getStatusColor(status: string): string {
  switch (status) {
    case 'CONFIRMED':
      return '#ef4444';
    case 'PENDING_REVIEW':
      return '#f59e0b';
    case 'DISMISSED':
      return '#6b7280';
    default:
      return '#38bdf8';
  }
}

function getStatusIcon(status: string): string {
  switch (status) {
    case 'CONFIRMED':
      return '⚠️';
    case 'PENDING_REVIEW':
      return '🔍';
    case 'DISMISSED':
      return '❌';
    default:
      return '📍';
  }
}

// ── Custom icon factory ────────────────────────────────────────

function createCustomIcon(status: string): L.Icon {
  const color = getStatusColor(status);

  return L.divIcon({
    className: 'custom-marker',
    html: `
      <div style="
        width: 32px; height: 32px;
        background: ${color}; border: 3px solid white;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        display: flex; align-items: center; justify-content: center;
        font-size: 14px;
      ">
        ${getStatusIcon(status)}
      </div>
    `,
    iconSize: [32, 32] as [number, number],
    iconAnchor: [16, 16] as [number, number],
    popupAnchor: [0, -20] as [number, number],
  }) as unknown as L.Icon;
}

export default function SuspectMap({ alerts }: SuspectMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const markersRef = useRef<L.Marker[]>([]);
  const [mapReady, setMapReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const locations = alerts
    .filter((a) => a.gpsLat !== null && a.gpsLon !== null)
    .filter((a, idx, arr) =>
      arr.findIndex((b) => b.gpsLat === a.gpsLat && b.gpsLon === a.gpsLon) === idx
    )
    .slice(0, 50);

  // Initialize Leaflet map
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    try {
      mapRef.current = L.map(containerRef.current).setView([51.505, -0.09], 13);

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      }).addTo(mapRef.current);

      L.control.zoom({ position: 'topright' }).addTo(mapRef.current);
      setMapReady(true);
    } catch (err) {
      setError(`Map initialization failed: ${err}`);
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // Update markers when alerts change
  const updateMarkers = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;

    markersRef.current.forEach((marker) => map.removeLayer(marker));
    markersRef.current = [];

    if (locations.length === 0) return;

    const latLngs: L.LatLng[] = [];

    locations.forEach((loc) => {
      if (loc.gpsLat === null || loc.gpsLon === null) return;

      const latLng = L.latLng(loc.gpsLat, loc.gpsLon);
      const icon = createCustomIcon(loc.status);
      const marker = L.marker(latLng, { icon }).addTo(map);

      const popupContent = `
        <div style="font-family: system-ui, sans-serif; min-width: 200px;">
          <div style="font-weight: 600; margin-bottom: 0.5rem;">
            ${getStatusIcon(loc.status)} ${loc.eventType}
          </div>
          <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">
            Alert ID: #${loc.id}
          </div>
          <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">
            Status: <span style="color: ${getStatusColor(loc.status)}; font-weight: 600;">
              ${loc.status}
            </span>
          </div>
          <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.5rem; font-family: monospace;">
            📍 ${loc.gpsLat.toFixed(6)}, ${loc.gpsLon.toFixed(6)}
          </div>
        </div>
      `;

      marker.bindPopup(popupContent, { maxWidth: 300 });
      markersRef.current.push(marker);
      latLngs.push(latLng);
    });

    if (latLngs.length > 0) {
      const bounds = L.latLngBounds(latLngs);
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [locations]);

  useEffect(() => {
    updateMarkers();
  }, [updateMarkers]);

  if (error) {
    return (
      <div className="map-container" style={{ background: '#0f172a', minHeight: '100%' }}>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#ef4444' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⚠️</div>
          <div>{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="map-container" style={{ position: 'relative', minHeight: '100%' }}>
      {/* Map stats overlay */}
      {locations.length > 0 && (
        <div style={{
          position: 'absolute', top: '1rem', left: '1rem', zIndex: 1000,
          background: 'rgba(30, 41, 59, 0.9)', border: '1px solid #334155',
          borderRadius: '8px', padding: '0.75rem 1rem',
          backdropFilter: 'blur(8px)',
        }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.5rem' }}>
            GPS MARKERS
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.75rem' }}>
            <span style={{ color: '#ef4444' }}>
              ⚠ {locations.filter((a) => a.status === 'CONFIRMED').length} Confirmed
            </span>
            <span style={{ color: '#f59e0b' }}>
              🔍 {locations.filter((a) => a.status === 'PENDING_REVIEW').length} Pending
            </span>
          </div>
        </div>
      )}

      <div
        ref={containerRef}
        style={{ width: '100%', height: '100%', minHeight: '500px' }}
      />

      {/* Legend */}
      <div style={{
        position: 'absolute', bottom: '1rem', left: '1rem', zIndex: 1000,
        background: 'rgba(30, 41, 59, 0.9)', border: '1px solid #334155',
        borderRadius: '8px', padding: '0.75rem 1rem',
        backdropFilter: 'blur(8px)', fontSize: '0.75rem',
      }}>
        <div style={{ fontWeight: 600, color: '#94a3b8', marginBottom: '0.5rem' }}>LEGEND</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ color: '#ef4444' }}>●</span> Confirmed Match
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ color: '#f59e0b' }}>●</span> Pending Review
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ color: '#6b7280' }}>●</span> Dismissed
          </div>
        </div>
      </div>

      {/* No markers state */}
      {locations.length === 0 && mapReady && (
        <div style={{
          position: 'absolute', top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)', textAlign: 'center', zIndex: 1000,
          background: 'rgba(30, 41, 59, 0.9)', border: '1px solid #334155',
          borderRadius: '12px', padding: '2rem', backdropFilter: 'blur(8px)',
        }}>
          <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>🗺️</div>
          <div style={{ fontSize: '0.875rem', color: '#94a3b8', fontWeight: 600 }}>
            No GPS Locations Yet
          </div>
          <div style={{ fontSize: '0.75rem', color: '#475569', marginTop: '0.5rem' }}>
            GPS-enabled cameras will display alert locations here
          </div>
        </div>
      )}
    </div>
  );
}
