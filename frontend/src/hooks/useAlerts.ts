import { useState, useEffect, useCallback, useRef } from 'react';
import { getAlerts, confirmAlert } from '../api/client';
import type { AlertItem } from '../types';

export function useAlerts(refreshIntervalMs = 30000) {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const abortRef = useRef<AbortController | null>(null);

  const fetchAlerts = useCallback(async (pageNum: number = 1) => {
    // Cancel previous in-flight request
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);
    try {
      const data = await getAlerts(pageNum);
      setAlerts(data);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      setError(err instanceof Error ? err.message : 'Failed to load alerts');
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch + auto-refresh
  useEffect(() => {
    fetchAlerts(1);
  }, [fetchAlerts]);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchAlerts(1);
    }, refreshIntervalMs);
    return () => clearInterval(interval);
  }, [fetchAlerts, refreshIntervalMs]);

  // Cleanup abort on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const handleConfirm = useCallback(async (alertId: number, confirmed: boolean) => {
    try {
      await confirmAlert(alertId, confirmed);
      // Optimistic update
      setAlerts((prev) =>
        prev.map((a) =>
          a.id === alertId
            ? {
                ...a,
                status: confirmed ? 'CONFIRMED' : 'DISMISSED',
                confirmedAt: new Date().toISOString(),
              }
            : a
        )
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to confirm alert');
      // Rollback on failure
      fetchAlerts(page);
    }
  }, [fetchAlerts, page]);

  return {
    alerts,
    loading,
    error,
    refresh: () => fetchAlerts(1),
    confirm: handleConfirm,
    nextPage: () => {
      setPage((p) => p + 1);
      fetchAlerts(page + 1);
    },
  };
}
