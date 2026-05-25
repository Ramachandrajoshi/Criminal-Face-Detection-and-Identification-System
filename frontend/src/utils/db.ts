const DB_NAME = 'CriminalFaceDetectionCache';
const DB_VERSION = 1;

let dbInstance: IDBDatabase | null = null;

function getDB(): Promise<IDBDatabase> {
  if (dbInstance) return Promise.resolve(dbInstance);

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('suspects')) {
        db.createObjectStore('suspects');
      }
      if (!db.objectStoreNames.contains('alerts')) {
        db.createObjectStore('alerts');
      }
    };

    request.onsuccess = () => {
      dbInstance = request.result;
      resolve(request.result);
    };

    request.onerror = () => {
      reject(request.error);
    };
  });
}

export async function storeSuspectImage(name: string, blob: Blob): Promise<void> {
  try {
    const db = await getDB();
    const tx = db.transaction('suspects', 'readwrite');
    const store = tx.objectStore('suspects');
    store.put(blob, name.trim().toLowerCase());
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (err) {
    console.error('Failed to store suspect image in IndexedDB:', err);
  }
}

export async function getSuspectImage(name: string): Promise<Blob | null> {
  try {
    const db = await getDB();
    const tx = db.transaction('suspects', 'readonly');
    const store = tx.objectStore('suspects');
    const request = store.get(name.trim().toLowerCase());
    return new Promise<Blob | null>((resolve, reject) => {
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  } catch (err) {
    console.error('Failed to get suspect image from IndexedDB:', err);
    return null;
  }
}

export async function storeAlertImage(alertId: number, blob: Blob): Promise<void> {
  try {
    const db = await getDB();
    const tx = db.transaction('alerts', 'readwrite');
    const store = tx.objectStore('alerts');
    store.put(blob, alertId);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (err) {
    console.error('Failed to store alert image in IndexedDB:', err);
  }
}

export async function getAlertImage(alertId: number): Promise<Blob | null> {
  try {
    const db = await getDB();
    const tx = db.transaction('alerts', 'readonly');
    const store = tx.objectStore('alerts');
    const request = store.get(alertId);
    return new Promise<Blob | null>((resolve, reject) => {
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  } catch (err) {
    console.error('Failed to get alert image from IndexedDB:', err);
    return null;
  }
}
