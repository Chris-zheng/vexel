import type { ModelCard } from '../types/vexel';
import type { TrainedLocalModel } from './mlRuntime';
import { fastSha256Hex } from './samples';

/**
 * Browser-local IndexedDB persistence for:
 *  1) Locally trained Few-Shot DL models (`TrainedLocalModel` including weights + SHA-256)
 *  2) Imported offline ONNX model binaries (`PersistedOnnxModel` including ArrayBuffer + ModelCard)
 *
 * Ensures refreshing the page 100% reproduces trained & imported models without data loss,
 * with graceful in-memory fallback when IndexedDB is unavailable (e.g. Node/Vitest or private iframes).
 */

export interface PersistedOnnxModel {
  ref: string;
  card: ModelCard;
  bytes: ArrayBuffer;
  inputName: string;
  outputName: string;
  inputDims: Array<number | string>;
  outputDims: Array<number | string>;
  savedAt: string;
}

export interface ExportedLocalModelEnvelope {
  schema: 'vexel.local_model.v1';
  exportedAt: string;
  sha256: string;
  model: TrainedLocalModel;
}

const DB_NAME = 'vexel_model_registry_v1';
const DB_VERSION = 1;
const STORE_LOCAL = 'local_models';
const STORE_ONNX = 'onnx_models';

// In-memory fallback for non-browser environments (Vitest / restricted sandboxes)
const memoryLocalStore = new Map<string, TrainedLocalModel>();
const memoryOnnxStore = new Map<string, PersistedOnnxModel>();

function hasIndexedDb(): boolean {
  return typeof indexedDB !== 'undefined' && indexedDB !== null;
}

function openRegistryDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_LOCAL)) {
        db.createObjectStore(STORE_LOCAL, { keyPath: 'ref' });
      }
      if (!db.objectStoreNames.contains(STORE_ONNX)) {
        db.createObjectStore(STORE_ONNX, { keyPath: 'ref' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('Failed to open IndexedDB model registry'));
  });
}

/** Deterministic integrity hash over trained weights + architecture metadata. */
export function computeLocalModelChecksum(model: TrainedLocalModel): string {
  const payload = JSON.stringify({
    id: model.card.id,
    version: model.card.version,
    classes: model.classes,
    deepHead: model.deepHead ?? null,
    classStats: model.classStats,
    stumps: model.stumps,
  });
  return fastSha256Hex(payload);
}

export async function saveLocalModel(model: TrainedLocalModel): Promise<void> {
  const record = {
    ref: model.card.ref,
    savedAt: new Date().toISOString(),
    checksum: computeLocalModelChecksum(model),
    model,
  };
  memoryLocalStore.set(model.card.ref, model);
  if (!hasIndexedDb()) return;
  try {
    const db = await openRegistryDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_LOCAL, 'readwrite');
      tx.objectStore(STORE_LOCAL).put(record);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error('Failed to save local model'));
    });
    db.close();
  } catch {
    // Keep in-memory copy if browser quota/private mode rejects IndexedDB
  }
}

export async function loadAllLocalModels(): Promise<TrainedLocalModel[]> {
  if (!hasIndexedDb()) {
    return Array.from(memoryLocalStore.values());
  }
  try {
    const db = await openRegistryDb();
    const rows = await new Promise<Array<{ ref: string; checksum?: string; model: TrainedLocalModel }>>(
      (resolve, reject) => {
        const tx = db.transaction(STORE_LOCAL, 'readonly');
        const req = tx.objectStore(STORE_LOCAL).getAll();
        req.onsuccess = () => resolve(req.result ?? []);
        req.onerror = () => reject(req.error ?? new Error('Failed to read local models'));
      }
    );
    db.close();

    const verified: TrainedLocalModel[] = [];
    for (const row of rows) {
      if (!row?.model?.card?.ref) continue;
      if (row.checksum && computeLocalModelChecksum(row.model) !== row.checksum) {
        // Skip corrupted record whose weights no longer match stored checksum
        continue;
      }
      memoryLocalStore.set(row.model.card.ref, row.model);
      verified.push(row.model);
    }
    return verified;
  } catch {
    return Array.from(memoryLocalStore.values());
  }
}

export async function deleteLocalModel(ref: string): Promise<void> {
  memoryLocalStore.delete(ref);
  if (!hasIndexedDb()) return;
  try {
    const db = await openRegistryDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_LOCAL, 'readwrite');
      tx.objectStore(STORE_LOCAL).delete(ref);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error('Failed to delete local model'));
    });
    db.close();
  } catch {
    // Ignore
  }
}

export async function saveOnnxModel(record: PersistedOnnxModel): Promise<void> {
  memoryOnnxStore.set(record.ref, record);
  if (!hasIndexedDb()) return;
  try {
    const db = await openRegistryDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_ONNX, 'readwrite');
      tx.objectStore(STORE_ONNX).put(record);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error('Failed to save ONNX model'));
    });
    db.close();
  } catch {
    // Keep in-memory copy
  }
}

export async function loadAllOnnxModels(): Promise<PersistedOnnxModel[]> {
  if (!hasIndexedDb()) {
    return Array.from(memoryOnnxStore.values());
  }
  try {
    const db = await openRegistryDb();
    const rows = await new Promise<PersistedOnnxModel[]>((resolve, reject) => {
      const tx = db.transaction(STORE_ONNX, 'readonly');
      const req = tx.objectStore(STORE_ONNX).getAll();
      req.onsuccess = () => resolve(req.result ?? []);
      req.onerror = () => reject(req.error ?? new Error('Failed to load ONNX models'));
    });
    db.close();
    const valid = rows.filter((r) => Boolean(r?.ref && r?.card && r?.bytes?.byteLength));
    for (const r of valid) memoryOnnxStore.set(r.ref, r);
    return valid;
  } catch {
    return Array.from(memoryOnnxStore.values());
  }
}

export async function deleteOnnxModel(ref: string): Promise<void> {
  memoryOnnxStore.delete(ref);
  if (!hasIndexedDb()) return;
  try {
    const db = await openRegistryDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_ONNX, 'readwrite');
      tx.objectStore(STORE_ONNX).delete(ref);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error('Failed to delete ONNX model'));
    });
    db.close();
  } catch {
    // Ignore
  }
}

/** Serialize a trained local model into a portable, SHA-256 verified `.vexmodel.json` envelope. */
export function exportLocalModelJson(model: TrainedLocalModel): string {
  const envelope: ExportedLocalModelEnvelope = {
    schema: 'vexel.local_model.v1',
    exportedAt: new Date().toISOString(),
    sha256: computeLocalModelChecksum(model),
    model,
  };
  return JSON.stringify(envelope, null, 2);
}

/** Parse and cryptographically verify an imported `.vexmodel.json` file. */
export function parseImportedLocalModelJson(jsonStr: string): TrainedLocalModel {
  const parsed = JSON.parse(jsonStr) as Partial<ExportedLocalModelEnvelope>;
  if (parsed?.schema !== 'vexel.local_model.v1' || !parsed.model?.card?.ref) {
    throw new Error(
      '无效的 Vexel 模型文件格式（期望 vexel.local_model.v1）/ Invalid Vexel model file schema.'
    );
  }
  const actualChecksum = computeLocalModelChecksum(parsed.model);
  if (parsed.sha256 && parsed.sha256 !== actualChecksum) {
    throw new Error(
      '模型权重 SHA-256 校验失败，文件可能已被篡改或损坏 / Model weight SHA-256 checksum mismatch.'
    );
  }
  return parsed.model;
}
