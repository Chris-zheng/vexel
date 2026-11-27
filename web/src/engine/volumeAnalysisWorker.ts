import { analyzeVolume3D, type VolumeAnalyzeInput } from './volumeAnalysisCore';
import type { VolumeAnalysisResult } from '../types/vexel';

export interface VolumeWorkerRequest {
  jobId: string;
  input: VolumeAnalyzeInput;
}

export interface VolumeWorkerResponse {
  jobId: string;
  ok: boolean;
  result?: VolumeAnalysisResult;
  error?: string;
}

const workerCtx = self as unknown as {
  onmessage: ((event: MessageEvent<VolumeWorkerRequest>) => void) | null;
  postMessage: (message: VolumeWorkerResponse, transfer?: Transferable[]) => void;
};

workerCtx.onmessage = (event: MessageEvent<VolumeWorkerRequest>) => {
  const msg = event.data;
  if (!msg?.input) return;
  try {
    const result = analyzeVolume3D(msg.input);
    const resp: VolumeWorkerResponse = {
      jobId: msg.jobId,
      ok: true,
      result,
    };
    workerCtx.postMessage(resp, [result.labelVolume.buffer]);
  } catch (err) {
    workerCtx.postMessage({
      jobId: msg.jobId,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
};
