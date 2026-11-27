import * as ort from 'onnxruntime-web';
import type { ModelCard } from '../types/vexel';
import type { PersistedOnnxModel } from './modelStore';
import { fastSha256Hex } from './samples';

/**
 * Browser & Node Offline ONNX Runtime Engine (`onnxruntime-web`).
 *
 * Supports:
 *  - Real `.onnx` binary inspection (validates ONNX graph + extracts input/output tensor specs + SHA-256)
 *  - Automatic NCHW / NHWC & 1-channel / 3-channel grayscale adaptation
 *  - Fixed spatial resolution (resize + bilinear un-resize) OR dynamic spatial tiling (`512×512` + cosine feathering)
 *  - Sigmoid (1-channel logit) and Softmax foreground channel extraction
 */

// Safe single-threaded WASM defaults so iframe environments without COOP/COEP headers never crash.
try {
  if (ort.env?.wasm) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    if (typeof window !== 'undefined') {
      ort.env.wasm.wasmPaths = '/ort/';
    }
  }
} catch {
  // Ignore if env is frozen in certain runtimes
}

export const MAX_ONNX_FILE_BYTES = 50 * 1024 * 1024; // 50 MB guardrail matching implementation plan

export interface InspectedOnnxMetadata {
  sha256: string;
  byteLength: number;
  inputName: string;
  outputName: string;
  inputDims: Array<number | string>;
  outputDims: Array<number | string>;
}

const sessionCache = new Map<string, ort.InferenceSession>();

async function computeBufferSha256Hex(bytes: ArrayBuffer): Promise<string> {
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const hashBuf = await crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(hashBuf))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }
  // Fallback for environments without WebCrypto subtle
  const u8 = new Uint8Array(bytes);
  let sample = `${u8.length}:`;
  const stride = Math.max(1, Math.floor(u8.length / 2048));
  for (let i = 0; i < u8.length; i += stride) sample += u8[i].toString(16);
  return fastSha256Hex(sample);
}

function readTensorDims(
  metaMap: ReadonlyArray<ort.InferenceSession.ValueMetadata> | undefined
): Array<number | string> {
  if (!metaMap || metaMap.length === 0) return [];
  const first = metaMap[0];
  if (first && 'shape' in first && Array.isArray(first.shape)) {
    return first.shape.map((d) => (typeof d === 'number' || typeof d === 'string' ? d : -1));
  }
  return [];
}

async function createOnnxSession(bytes: ArrayBuffer): Promise<ort.InferenceSession> {
  const u8 = new Uint8Array(bytes);
  return ort.InferenceSession.create(u8, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

/** Validate an uploaded `.onnx` binary via onnxruntime-web and inspect its tensor signature & SHA-256. */
export async function inspectOnnxBinary(bytes: ArrayBuffer): Promise<InspectedOnnxMetadata> {
  if (!bytes || bytes.byteLength < 16) {
    throw new Error('无效的 ONNX 文件（体积过小）/ Invalid ONNX binary (file too small).');
  }
  if (bytes.byteLength > MAX_ONNX_FILE_BYTES) {
    throw new Error(
      `ONNX 权重超出 50 MB 浏览器安全上限（当前 ${(bytes.byteLength / (1024 * 1024)).toFixed(1)} MB）/ ONNX binary exceeds 50 MB browser safety limit.`
    );
  }
  const [sha256, session] = await Promise.all([
    computeBufferSha256Hex(bytes),
    createOnnxSession(bytes),
  ]);

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (!inputName || !outputName) {
    await session.release();
    throw new Error('ONNX 模型缺少有效输入或输出节点 / ONNX model missing input or output tensor.');
  }

  const inputDims = readTensorDims(session.inputMetadata);
  const outputDims = readTensorDims(session.outputMetadata);
  await session.release();

  return {
    sha256,
    byteLength: bytes.byteLength,
    inputName,
    outputName,
    inputDims,
    outputDims,
  };
}

/** Build a verified ModelCard + PersistedOnnxModel record directly from a user-selected `.onnx` binary. */
export async function buildPersistedOnnxModel(
  fileName: string,
  bytes: ArrayBuffer,
  options: {
    modelId?: string;
    version?: string;
    modality?: ModelCard['modality'];
    task?: ModelCard['task'];
    classes?: string[];
  } = {}
): Promise<PersistedOnnxModel> {
  const meta = await inspectOnnxBinary(bytes);
  const cleanBase =
    (options.modelId || fileName.replace(/\.onnx$/i, ''))
      .replace(/[^a-zA-Z0-9._-]/g, '_')
      .slice(0, 48) || 'custom_onnx';
  const version = (options.version || '1.0').replace(/[^a-zA-Z0-9._-]/g, '').slice(0, 16) || '1.0';
  const ref = `${cleanBase}@${version}`;
  const sizeMb = (bytes.byteLength / (1024 * 1024)).toFixed(2);
  const inShapeStr = meta.inputDims.length ? `[${meta.inputDims.join('×')}]` : 'dynamic';

  const card: ModelCard = {
    id: cleanBase,
    version,
    ref,
    nameZh: `离线 ONNX 模型 (${cleanBase})`,
    nameEn: `Offline ONNX (${cleanBase})`,
    task: options.task ?? 'segment.instances',
    framework: 'onnx',
    modality: options.modality ?? 'shared',
    license: 'User-Imported-ONNX',
    commercialUse: true,
    trainingData: [{ name: `${fileName} (${sizeMb} MB · ${inShapeStr})`, license: 'User-Owned' }],
    sha256: meta.sha256,
    tiledInference: true,
    tileSize: 512,
    fileStatus: 'verified',
    metrics: {},
    descriptionZh: `已通过 onnxruntime-web 验证图结构并持久化至浏览器 IndexedDB，输入 ${meta.inputName} ${inShapeStr} → 输出 ${meta.outputName}，支持 512×512 羽化瓦片离线推理。`,
    descriptionEn: `Validated via onnxruntime-web & persisted in IndexedDB. Input ${meta.inputName} ${inShapeStr} → Output ${meta.outputName} with cosine-feathered tiled offline inference.`,
    isLocalTrained: true,
    classes: options.classes ?? ['Foreground', 'Background'],
  };

  return {
    ref,
    card,
    bytes: bytes.slice(0),
    inputName: meta.inputName,
    outputName: meta.outputName,
    inputDims: meta.inputDims,
    outputDims: meta.outputDims,
    savedAt: new Date().toISOString(),
  };
}

export async function disposeOnnxSession(ref: string): Promise<void> {
  const sess = sessionCache.get(ref);
  if (sess) {
    sessionCache.delete(ref);
    try {
      await sess.release();
    } catch {
      // Ignore release errors
    }
  }
}

/** Detect tensor channel/spatial layout from ONNX inputDims (defaults to NCHW 1-channel dynamic). */
function resolveInputSpec(dims: Array<number | string>): {
  layout: 'nchw' | 'nhwc';
  channels: number;
  fixedH: number | null;
  fixedW: number | null;
} {
  if (dims.length === 4) {
    const d1 = typeof dims[1] === 'number' && dims[1] > 0 ? dims[1] : null;
    const d2 = typeof dims[2] === 'number' && dims[2] > 0 ? dims[2] : null;
    const d3 = typeof dims[3] === 'number' && dims[3] > 0 ? dims[3] : null;
    // Heuristic: NHWC if last dim is 1 or 3 while d1/d2 are larger spatial dims
    if ((d3 === 1 || d3 === 3) && (d1 === null || d1 > 4)) {
      return { layout: 'nhwc', channels: d3, fixedH: d1, fixedW: d2 };
    }
    return {
      layout: 'nchw',
      channels: d1 === 3 ? 3 : 1,
      fixedH: d2,
      fixedW: d3,
    };
  }
  return { layout: 'nchw', channels: 1, fixedH: null, fixedW: null };
}

/** Bilinear sample a 2D grayscale tile (srcW×srcH) into dstW×dstH normalized [0, 1]. */
function resamplePatchNormalized(
  gray: Float32Array,
  fullW: number,
  x0: number,
  y0: number,
  patchW: number,
  patchH: number,
  dstW: number,
  dstH: number
): Float32Array {
  const out = new Float32Array(dstW * dstH);
  const scaleX = patchW / dstW;
  const scaleY = patchH / dstH;
  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(patchH - 1, Math.max(0, (y + 0.5) * scaleY - 0.5));
    const iy0 = Math.floor(sy);
    const iy1 = Math.min(patchH - 1, iy0 + 1);
    const fy = sy - iy0;
    const row0 = (y0 + iy0) * fullW + x0;
    const row1 = (y0 + iy1) * fullW + x0;
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(patchW - 1, Math.max(0, (x + 0.5) * scaleX - 0.5));
      const ix0 = Math.floor(sx);
      const ix1 = Math.min(patchW - 1, ix0 + 1);
      const fx = sx - ix0;
      const v00 = gray[row0 + ix0];
      const v10 = gray[row0 + ix1];
      const v01 = gray[row1 + ix0];
      const v11 = gray[row1 + ix1];
      const top = v00 + (v10 - v00) * fx;
      const bot = v01 + (v11 - v01) * fx;
      out[y * dstW + x] = Math.max(0, Math.min(1, (top + (bot - top) * fy) / 255));
    }
  }
  return out;
}

/** Convert ONNX output tensor (1-ch probability/logit OR multi-ch softmax) into a 2D [0,1] foreground map. */
function decodeOnnxOutputToProb(
  outputTensor: ort.Tensor,
  fallbackW: number,
  fallbackH: number,
  foregroundChannel = 1
): { prob: Float32Array; outW: number; outH: number } {
  const data = outputTensor.data as Float32Array | Float64Array | Int32Array | Uint8Array;
  const dims = outputTensor.dims;
  let channels = 1;
  let outH = fallbackH;
  let outW = fallbackW;
  let isNhwc = false;

  if (dims.length === 4) {
    if (dims[3] <= 16 && dims[1] > 16) {
      isNhwc = true;
      outH = dims[1];
      outW = dims[2];
      channels = dims[3];
    } else {
      channels = dims[1];
      outH = dims[2];
      outW = dims[3];
    }
  } else if (dims.length === 3) {
    outH = dims[1];
    outW = dims[2];
  } else if (dims.length === 2) {
    outH = dims[0];
    outW = dims[1];
  }

  const n = outW * outH;
  const prob = new Float32Array(n);

  if (channels <= 1) {
    let minV = Infinity;
    let maxV = -Infinity;
    for (let i = 0; i < Math.min(n, data.length); i++) {
      const v = Number(data[i]);
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    const alreadyProb = minV >= -1e-4 && maxV <= 1 + 1e-4;
    for (let i = 0; i < Math.min(n, data.length); i++) {
      const v = Number(data[i]);
      prob[i] = alreadyProb ? Math.max(0, Math.min(1, v)) : 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, v))));
    }
  } else {
    const fgIdx = Math.min(channels - 1, Math.max(0, foregroundChannel));
    for (let i = 0; i < n; i++) {
      let maxLogit = -Infinity;
      for (let c = 0; c < channels; c++) {
        const rawIdx = isNhwc ? i * channels + c : c * n + i;
        const v = Number(data[rawIdx] ?? 0);
        if (v > maxLogit) maxLogit = v;
      }
      let sumExp = 0;
      let fgExp = 0;
      for (let c = 0; c < channels; c++) {
        const rawIdx = isNhwc ? i * channels + c : c * n + i;
        const e = Math.exp(Math.max(-20, Number(data[rawIdx] ?? 0) - maxLogit));
        sumExp += e;
        if (c === fgIdx) fgExp = e;
      }
      prob[i] = sumExp > 0 ? fgExp / sumExp : 0;
    }
  }

  return { prob, outW, outH };
}

/**
 * Execute real offline ONNX segmentation inference with cosine-feathered tiling over `gray`.
 * Works seamlessly in both browser (WASM) and Node.js (Vitest).
 */
export async function predictOnnxModelTiled(
  model: PersistedOnnxModel,
  gray: Float32Array,
  width: number,
  height: number,
  options: {
    cropRows?: number;
    tileSize?: number;
    overlap?: number;
    foregroundChannel?: number;
  } = {}
): Promise<Float32Array> {
  let session = sessionCache.get(model.ref);
  if (!session) {
    session = await createOnnxSession(model.bytes);
    sessionCache.set(model.ref, session);
  }

  const maxY = Math.min(height, options.cropRows ?? height);
  const spec = resolveInputSpec(model.inputDims);
  const tileW = spec.fixedW ?? Math.min(width, options.tileSize ?? 512);
  const tileH = spec.fixedH ?? Math.min(maxY, options.tileSize ?? 512);
  const overlap = Math.min(options.overlap ?? 32, Math.floor(Math.min(tileW, tileH) * 0.25));
  const stepX = Math.max(1, tileW - overlap);
  const stepY = Math.max(1, tileH - overlap);

  const probAcc = new Float32Array(width * height);
  const weightAcc = new Float32Array(width * height);

  // Cosine feather weights along tile boundaries
  const makeFeather = (len: number, ov: number) => {
    const w = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      let f = 1;
      if (ov > 0 && i < ov) f = Math.min(f, 0.5 - 0.5 * Math.cos((Math.PI * (i + 0.5)) / ov));
      if (ov > 0 && i >= len - ov) {
        const d = len - 1 - i;
        f = Math.min(f, 0.5 - 0.5 * Math.cos((Math.PI * (d + 0.5)) / ov));
      }
      w[i] = Math.max(0.05, f);
    }
    return w;
  };

  const xs: number[] = [];
  for (let x = 0; x < width; x += stepX) {
    xs.push(Math.min(x, Math.max(0, width - tileW)));
    if (x + tileW >= width) break;
  }
  const ys: number[] = [];
  for (let y = 0; y < maxY; y += stepY) {
    ys.push(Math.min(y, Math.max(0, maxY - tileH)));
    if (y + tileH >= maxY) break;
  }

  for (const ty of ys) {
    const patchH = Math.min(tileH, maxY - ty);
    const runH = spec.fixedH ?? patchH;
    const featherY = makeFeather(patchH, ys.length > 1 ? overlap : 0);

    for (const tx of xs) {
      const patchW = Math.min(tileW, width - tx);
      const runW = spec.fixedW ?? patchW;
      const featherX = makeFeather(patchW, xs.length > 1 ? overlap : 0);

      const normPatch = resamplePatchNormalized(gray, width, tx, ty, patchW, patchH, runW, runH);
      const planeSize = runW * runH;
      const tensorData = new Float32Array(spec.channels * planeSize);
      if (spec.channels === 1) {
        tensorData.set(normPatch);
      } else if (spec.layout === 'nchw') {
        for (let c = 0; c < spec.channels; c++) {
          tensorData.set(normPatch, c * planeSize);
        }
      } else {
        for (let i = 0; i < planeSize; i++) {
          const v = normPatch[i];
          for (let c = 0; c < spec.channels; c++) {
            tensorData[i * spec.channels + c] = v;
          }
        }
      }

      const dims =
        spec.layout === 'nhwc'
          ? [1, runH, runW, spec.channels]
          : [1, spec.channels, runH, runW];
      const inputTensor = new ort.Tensor('float32', tensorData, dims);
      const outputs = await session.run({ [model.inputName]: inputTensor });
      const outTensor = outputs[model.outputName] ?? Object.values(outputs)[0];
      if (!outTensor) continue;

      const { prob, outW, outH } = decodeOnnxOutputToProb(
        outTensor,
        runW,
        runH,
        options.foregroundChannel ?? 1
      );

      // Accumulate feathered probability back onto full-resolution image coordinates
      const sx = outW / patchW;
      const sy = outH / patchH;
      for (let py = 0; py < patchH; py++) {
        const fy = featherY[py];
        const oy = Math.min(outH - 1, Math.floor((py + 0.5) * sy));
        const dstRow = (ty + py) * width + tx;
        const srcRow = oy * outW;
        for (let px = 0; px < patchW; px++) {
          const ox = Math.min(outW - 1, Math.floor((px + 0.5) * sx));
          const w = fy * featherX[px];
          probAcc[dstRow + px] += prob[srcRow + ox] * w;
          weightAcc[dstRow + px] += w;
        }
      }
    }
  }

  const finalProb = new Float32Array(width * height);
  for (let i = 0; i < width * maxY; i++) {
    finalProb[i] = weightAcc[i] > 0 ? Math.max(0, Math.min(1, probAcc[i] / weightAcc[i])) : 0;
  }
  return finalProb;
}
