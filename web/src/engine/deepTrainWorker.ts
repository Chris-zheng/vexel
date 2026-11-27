import {
  DEEP_FEATURE_DIM,
  extractMultiScaleFeatures24,
  trainDeepFewShotHead,
  type DeepTrainOutcome,
  type DeepTrainProgressEvent,
} from './deepFewShotCore';
import type { DeepFewShotConfig } from '../types/vexel';

export interface DeepTrainWorkerRequest {
  type: 'train' | 'cancel';
  jobId: string;
  cacheKey?: string;
  gray?: Float32Array;
  mask?: Uint8Array;
  width?: number;
  height?: number;
  classes?: string[];
  config?: DeepFewShotConfig;
}

export interface DeepTrainWorkerResponse {
  type: 'progress' | 'done' | 'error';
  jobId: string;
  progress?: DeepTrainProgressEvent;
  outcome?: DeepTrainOutcome;
  error?: string;
}

// Single-entry incremental feature pyramid cache inside worker memory
// so adding brush/SAM annotations on the same image/ROI skips feature re-extraction (<1s retrain)
let cachedKey: string | null = null;
let cachedFeats: Float32Array | null = null;
let activeJobId: string | null = null;
let cancelledJobId: string | null = null;

self.onmessage = (event: MessageEvent<DeepTrainWorkerRequest>) => {
  const msg = event.data;
  if (!msg) return;

  if (msg.type === 'cancel') {
    cancelledJobId = msg.jobId;
    return;
  }

  if (msg.type === 'train') {
    activeJobId = msg.jobId;
    cancelledJobId = null;

    try {
      const { cacheKey, gray, mask, width, height, classes, config } = msg;
      if (!gray || !mask || !width || !height || !classes || !config) {
        throw new Error('Missing required training payload in worker.');
      }

      let feats: Float32Array;
      let cacheHit = false;
      if (
        cacheKey &&
        cachedKey === cacheKey &&
        cachedFeats &&
        cachedFeats.length === width * height * DEEP_FEATURE_DIM
      ) {
        feats = cachedFeats;
        cacheHit = true;
      } else {
        const extractProgress: DeepTrainProgressEvent = {
          phase: 'extracting',
          epoch: 0,
          totalEpochs: config.epochs,
          trainLoss: 0,
          valLoss: 0,
          valMeanIoU: 0,
          valMacroF1: 0,
          cacheHit: false,
        };
        self.postMessage({
          type: 'progress',
          jobId: msg.jobId,
          progress: extractProgress,
        } satisfies DeepTrainWorkerResponse);

        feats = extractMultiScaleFeatures24(gray, width, height);
        cachedKey = cacheKey ?? null;
        cachedFeats = feats;
      }

      const outcome = trainDeepFewShotHead(
        feats,
        mask,
        width,
        classes,
        config,
        cacheHit,
        (progress) => {
          self.postMessage({
            type: 'progress',
            jobId: msg.jobId,
            progress,
          } satisfies DeepTrainWorkerResponse);
        },
        () => cancelledJobId === msg.jobId || activeJobId !== msg.jobId
      );

      self.postMessage({
        type: 'done',
        jobId: msg.jobId,
        outcome,
      } satisfies DeepTrainWorkerResponse);
    } catch (err) {
      self.postMessage({
        type: 'error',
        jobId: msg.jobId,
        error: err instanceof Error ? err.message : String(err),
      } satisfies DeepTrainWorkerResponse);
    }
  }
};
