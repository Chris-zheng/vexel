import type {
  AnnotationStroke,
  DetectionObject,
  HoldoutMetrics,
  HumanCorrection,
  ModelCard,
  ObjectMeasurementMetrics,
  OverlayShape,
  PromptPoint,
  RefinedBoundaryProposal,
  RoiRect,
  Specimen,
} from '../types/vexel';
import { traceMaskPixelContours } from './contourTrace';
import { fastSha256Hex } from './samples';

import {
  extractMultiScaleFeatures24,
  predictDeepFewShotTiled,
  trainDeepFewShotHead,
  type DeepFewShotWeights,
  type DeepTrainProgressEvent,
} from './deepFewShotCore';
import type { DeepFewShotConfig } from '../types/vexel';
import type { DeepTrainWorkerRequest, DeepTrainWorkerResponse } from './deepTrainWorker';

export type { DeepFewShotWeights, DeepTrainProgressEvent };

export interface TrainedLocalModel {
  card: ModelCard;
  classes: string[];
  /** Upgraded frozen multi-scale pyramid + residual bottleneck MLP head (preferred when present) */
  deepHead?: DeepFewShotWeights;
  /** Per-class weighted feature prototypes & inverse variances (legacy fallback) */
  classStats: Array<{
    classId: number;
    mean: number[];
    std: number[];
    prior: number;
  }>;
  /** Decision stumps learned via deterministic boosting over top separating features (legacy fallback) */
  stumps: Array<{
    featureIdx: number;
    threshold: number;
    polarity: number;
    targetClass: number;
    weight: number;
  }>;
  metrics: HoldoutMetrics;
}

/**
 * Model cards of models Vexel can use once their weight file is imported. None ships with
 * weights: they are listed as `missing`, with no checksum and no metrics, and cannot be bound
 * to a workflow until a real file (with its own SHA-256 and evaluation) is imported.
 */
export const BUILTIN_MODELS: ModelCard[] = [
  {
    id: 'vexel-mobilesam',
    version: '1.0',
    ref: 'vexel-mobilesam@1.0',
    nameZh: 'MobileSAM 交互式提示分割',
    nameEn: 'MobileSAM Prompt Segmentation',
    task: 'segment.instances',
    framework: 'onnx',
    modality: 'shared',
    license: 'Apache-2.0',
    commercialUse: true,
    trainingData: [{ name: 'SA-1B (Apache-2.0 distilled encoder)', license: 'Apache-2.0' }],
    sha256: '',
    tiledInference: true,
    tileSize: 512,
    fileStatus: 'missing',
    metrics: {},
    descriptionZh: '点击正/负提示点即时扣出复杂形貌目标轮廓，适用于 SEM 粘连颗粒、不规则孔洞与非圆液滴纠错。（模型卡：本版本未附带权重文件，不能使用；导入对应的 ONNX 文件后才可绑定。）',
    descriptionEn: 'Interactive positive/negative click-prompt instance segmentation for touching SEM particles and irregular droplets. (Model card only: no weight file ships with this version; import the ONNX file to use it.)',
  },
  {
    id: 'vexel-sem-particles-unet',
    version: '1.2',
    ref: 'vexel-sem-particles-unet@1.2',
    nameZh: 'SEM 颗粒与孔隙分块分割网络',
    nameEn: 'SEM Particle & Pore Tiled UNet',
    task: 'segment.instances',
    framework: 'onnx',
    modality: 'sem',
    license: 'Apache-2.0',
    commercialUse: true,
    trainingData: [
      { name: 'Vexel Synthetic SEM Materials Corpus', license: 'Apache-2.0' },
      { name: 'Open Nanomaterials SEM Subset', license: 'CC-BY-4.0' },
    ],
    sha256: '',
    tiledInference: true,
    tileSize: 512,
    fileStatus: 'missing',
    metrics: {},
    descriptionZh: '支持 512×512 重叠羽化分块推理的轻量 ONNX 模型，抗荷电亮边与不均背底，无缝替换经典阈值分割。（模型卡：本版本未附带权重文件，不能使用；导入对应的 ONNX 文件后才可绑定。）',
    descriptionEn: 'Cosine-feathered 512×512 tiled ONNX segmentation model resilient to charging artefacts and uneven SEM illumination. (Model card only: no weight file ships with this version; import the ONNX file to use it.)',
  },
  {
    id: 'vexel-droplet-cells-tiny',
    version: '1.1',
    ref: 'vexel-droplet-cells-tiny@1.1',
    nameZh: '液滴内细胞/微球计数检测模型',
    nameEn: 'Droplet Inclusion Counter Tiny',
    task: 'count.inclusions',
    framework: 'onnx',
    modality: 'droplet',
    license: 'MIT',
    commercialUse: true,
    trainingData: [{ name: 'WSCNet + Vexel Microfluidics Benchmark', license: 'MIT' }],
    sha256: '',
    tiledInference: false,
    fileStatus: 'missing',
    metrics: {},
    descriptionZh: '针对高速微流控明场录像优化的逐帧小模型，精确拆分相邻贴边细胞并输出置信度。（模型卡：本版本未附带权重文件，不能使用；导入对应的 ONNX 文件后才可绑定。）',
    descriptionEn: 'Per-frame ONNX inclusion detector resolving adjacent and rim-touching cells inside microfluidic droplets. (Model card only: no weight file ships with this version; import the ONNX file to use it.)',
  },
  {
    id: 'cellpose-cyto3',
    version: '3.0',
    ref: 'cellpose-cyto3@3.0',
    nameZh: 'Cellpose Cyto3 通用细胞分割',
    nameEn: 'Cellpose Cyto3 Generalist',
    task: 'segment.instances',
    framework: 'pytorch',
    modality: 'shared',
    license: 'BSD-3-Clause',
    commercialUse: false,
    trainingData: [{ name: 'Cellpose Community Dataset (Non-Commercial)', license: 'CC-BY-NC-4.0' }],
    sha256: '',
    tiledInference: true,
    tileSize: 512,
    fileStatus: 'missing',
    metrics: {},
    descriptionZh: '高校科研通用细胞与密集目标分割模型；因部分训练集含 NC 条款，已由系统自动标记为仅限科研使用。（模型卡：本版本未附带权重文件，不能使用；导入对应的 ONNX 文件后才可绑定。）',
    descriptionEn: 'Generalist cellular segmentation model; automatically flagged Research-Only due to CC-BY-NC training subset. (Model card only: no weight file ships with this version; import the ONNX file to use it.)',
  },
];

/** Extract 5 multi-scale deterministic pixel features: [intensity, mean3, std5, sobelGrad, laplacian]. */
export function extractPixelFeatures(
  gray: Float32Array,
  width: number,
  height: number
): Float32Array {
  const n = width * height;
  const feats = new Float32Array(n * 5);

  // Normalize intensity range (p1..p99 approximation via mean/std)
  let sum = 0;
  for (let i = 0; i < n; i++) sum += gray[i];
  const meanAll = sum / Math.max(1, n);
  let varAll = 0;
  for (let i = 0; i < n; i++) {
    const d = gray[i] - meanAll;
    varAll += d * d;
  }
  const stdAll = Math.max(10, Math.sqrt(varAll / Math.max(1, n)));

  const at = (x: number, y: number) => {
    const cx = x < 0 ? 0 : x >= width ? width - 1 : x;
    const cy = y < 0 ? 0 : y >= height ? height - 1 : y;
    return (gray[cy * width + cx] - meanAll) / stdAll;
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const c = at(x, y);
      const l = at(x - 1, y);
      const r = at(x + 1, y);
      const u = at(x, y - 1);
      const d = at(x, y + 1);
      const ul = at(x - 1, y - 1);
      const ur = at(x + 1, y - 1);
      const dl = at(x - 1, y + 1);
      const dr = at(x + 1, y + 1);

      const m3 = (c + l + r + u + d + ul + ur + dl + dr) / 9;
      const gx = (ur + 2 * r + dr) - (ul + 2 * l + dl);
      const gy = (dl + 2 * d + dr) - (ul + 2 * u + ur);
      const grad = Math.hypot(gx, gy) * 0.25;
      const lap = l + r + u + d - 4 * c;

      // Coarse 5-tap texture spread
      const far1 = at(x - 3, y);
      const far2 = at(x + 3, y);
      const far3 = at(x, y - 3);
      const far4 = at(x, y + 3);
      const m5 = (m3 + far1 + far2 + far3 + far4) / 5;
      const v5 =
        ((c - m5) ** 2 + (far1 - m5) ** 2 + (far2 - m5) ** 2 + (far3 - m5) ** 2 + (far4 - m5) ** 2) / 5;
      const std5 = Math.sqrt(v5);

      const base = idx * 5;
      feats[base] = c;
      feats[base + 1] = m3;
      feats[base + 2] = std5;
      feats[base + 3] = grad;
      feats[base + 4] = lap;
    }
  }
  return feats;
}

/** Rasterize brush strokes + optional human review correction patches into pixel class labels (0 = unlabeled).
 * When an ROI is active, rasterizes strictly inside the ROI coordinate patch (roi.width × roi.height). */
export function buildTrainingMask(
  specimen: Specimen,
  strokes: AnnotationStroke[],
  corrections: HumanCorrection[],
  objects: Array<{ id: number; x: number; y: number; diameter?: number; excluded?: boolean }>,
  roi?: RoiRect | null
): Uint8Array {
  const fullW = specimen.width;
  const maxY = specimen.calibration.cropRows ?? specimen.height;
  const rx = roi ? Math.max(0, Math.min(fullW - 1, Math.round(roi.x))) : 0;
  const ry = roi ? Math.max(0, Math.min(maxY - 1, Math.round(roi.y))) : 0;
  const rw = roi ? Math.max(1, Math.min(fullW - rx, Math.round(roi.width))) : fullW;
  const rh = roi ? Math.max(1, Math.min(maxY - ry, Math.round(roi.height))) : specimen.height;
  const effectiveRh = roi ? rh : maxY;

  const mask = new Uint8Array(rw * rh);

  const rasterizePolygon = (
    outer: Array<[number, number]>,
    holes: Array<Array<[number, number]>> | undefined,
    classId: number
  ) => {
    const rings: Array<Array<[number, number]>> = [outer];
    if (holes) {
      for (const h of holes) {
        if (h.length >= 3) rings.push(h);
      }
    }
    let minY = Infinity;
    let maxY = -Infinity;
    for (const ring of rings) {
      for (const [, py] of ring) {
        const ly = py - ry;
        if (ly < minY) minY = ly;
        if (ly > maxY) maxY = ly;
      }
    }
    const y0 = Math.max(0, Math.floor(minY));
    const y1 = Math.min(effectiveRh - 1, Math.ceil(maxY));
    const xs: number[] = [];
    for (let y = y0; y <= y1; y++) {
      const scanY = y + 0.5;
      xs.length = 0;
      for (const ring of rings) {
        const len = ring.length;
        for (let i = 0, j = len - 1; i < len; j = i++) {
          const x1 = ring[i][0] - rx;
          const y1p = ring[i][1] - ry;
          const x2 = ring[j][0] - rx;
          const y2p = ring[j][1] - ry;
          if ((y1p <= scanY && y2p > scanY) || (y2p <= scanY && y1p > scanY)) {
            xs.push(x1 + ((scanY - y1p) / (y2p - y1p)) * (x2 - x1));
          }
        }
      }
      if (xs.length < 2) continue;
      xs.sort((a, b) => a - b);
      const rowOff = y * rw;
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const sx0 = Math.max(0, Math.floor(xs[k]));
        const sx1 = Math.min(rw - 1, Math.floor(xs[k + 1]));
        for (let x = sx0; x <= sx1; x++) {
          mask[rowOff + x] = classId;
        }
      }
    }
  };

  for (const stroke of strokes) {
    if (stroke.polygon && stroke.polygon.length >= 3) {
      rasterizePolygon(stroke.polygon, stroke.holes, stroke.classId);
      continue;
    }
    const rad = Math.max(2, stroke.radius);
    const r2 = rad * rad;
    for (const [px, py] of stroke.points) {
      const lx = px - rx;
      const ly = py - ry;
      const x0 = Math.max(0, Math.floor(lx - rad));
      const x1 = Math.min(rw - 1, Math.ceil(lx + rad));
      const y0 = Math.max(0, Math.floor(ly - rad));
      const y1 = Math.min(effectiveRh - 1, Math.ceil(ly + rad));
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          if ((x - lx) ** 2 + (y - ly) ** 2 <= r2) {
            mask[y * rw + x] = stroke.classId;
          }
        }
      }
    }
  }

  // Feed human review corrections back into Active Learning mask (strictly inside ROI when set)
  const byId = new Map(objects.map((o) => [o.id, o]));
  for (const corr of corrections) {
    if (corr.specimenId !== specimen.id) continue;
    const obj = byId.get(corr.objectId);
    if (!obj) continue;
    const radPx = Math.max(5, Math.min(18, ((obj.diameter ?? 20) / Math.max(0.5, specimen.calibration.pixelSize)) * 0.35));
    const classId = corr.field === 'excluded' && corr.newValue === true ? 2 : corr.field === 'cls' && corr.newValue === 'outlier' ? 3 : 1;
    const r2 = radPx * radPx;
    const lx = obj.x - rx;
    const ly = obj.y - ry;
    for (let y = Math.max(0, Math.floor(ly - radPx)); y <= Math.min(effectiveRh - 1, Math.ceil(ly + radPx)); y++) {
      for (let x = Math.max(0, Math.floor(lx - radPx)); x <= Math.min(rw - 1, Math.ceil(lx + radPx)); x++) {
        if ((x - lx) ** 2 + (y - ly) ** 2 <= r2) {
          if (mask[y * rw + x] === 0) mask[y * rw + x] = classId;
        }
      }
    }
  }

  return mask;
}

/** Train a real multi-class local pixel classifier with stratified 75/25 hold-out evaluation.
 * Supports training strictly inside a manual ROI and on upstream preprocessed workingGray. */
export function trainLocalPixelModel(
  modelId: string,
  version: string,
  specimen: Specimen,
  strokes: AnnotationStroke[],
  corrections: HumanCorrection[],
  objects: Array<{ id: number; x: number; y: number; diameter?: number; excluded?: boolean }>,
  classes: string[],
  options: { workingGray?: Float32Array; roi?: RoiRect | null } = {}
): TrainedLocalModel {
  const fullW = specimen.width;
  const maxY = specimen.calibration.cropRows ?? specimen.height;
  const roi = options.roi ?? null;
  const rx = roi ? Math.max(0, Math.min(fullW - 1, Math.round(roi.x))) : 0;
  const ry = roi ? Math.max(0, Math.min(maxY - 1, Math.round(roi.y))) : 0;
  const rw = roi ? Math.max(1, Math.min(fullW - rx, Math.round(roi.width))) : fullW;
  const rh = roi ? Math.max(1, Math.min(maxY - ry, Math.round(roi.height))) : specimen.height;

  const baseGray =
    options.workingGray && options.workingGray.length === fullW * specimen.height
      ? options.workingGray
      : specimen.gray;

  // Slice strictly to ROI patch when ROI is active so feature extraction runs only on ROI pixels
  let patchGray: Float32Array;
  if (roi) {
    patchGray = new Float32Array(rw * rh);
    for (let y = 0; y < rh; y++) {
      const srcOff = (ry + y) * fullW + rx;
      const dstOff = y * rw;
      for (let x = 0; x < rw; x++) {
        patchGray[dstOff + x] = baseGray[srcOff + x];
      }
    }
  } else {
    patchGray = baseGray;
  }

  const mask = buildTrainingMask(specimen, strokes, corrections, objects, roi);
  const feats = extractPixelFeatures(patchGray, rw, rh);

  const perClassIndices = new Map<number, number[]>();
  for (let i = 0; i < mask.length; i++) {
    const c = mask[i];
    if (c > 0) {
      let arr = perClassIndices.get(c);
      if (!arr) {
        arr = [];
        perClassIndices.set(c, arr);
      }
      arr.push(i);
    }
  }

  if (perClassIndices.size < 2) {
    throw new Error(
      roi
        ? '当前已启用 ROI：请在 ROI 区域内至少标注两个类别（例如“目标前景”与“背景”），再执行模型训练 / Paint at least two classes inside the active ROI before training.'
        : '请至少标注两个类别（例如“目标前景”与“背景”），再执行模型训练 / Paint at least two classes before training.'
    );
  }

  // Deterministic stratified 75% train / 25% hold-out split (cap per class to keep instant responsiveness)
  const trainSamples: Array<{ idx: number; cls: number }> = [];
  const valSamples: Array<{ idx: number; cls: number }> = [];

  for (const [cls, indices] of perClassIndices.entries()) {
    const stride = Math.max(1, Math.floor(indices.length / 2400));
    const sub: number[] = [];
    for (let i = 0; i < indices.length; i += stride) sub.push(indices[i]);
    for (let i = 0; i < sub.length; i++) {
      if (i % 4 === 0 && sub.length >= 4) {
        valSamples.push({ idx: sub[i], cls });
      } else {
        trainSamples.push({ idx: sub[i], cls });
      }
    }
  }

  // Fit per-class feature statistics
  const presentClasses = Array.from(perClassIndices.keys()).sort((a, b) => a - b);
  const classStats = presentClasses.map((classId) => {
    const rows = trainSamples.filter((s) => s.cls === classId);
    const mean = [0, 0, 0, 0, 0];
    for (const r of rows) {
      const b = r.idx * 5;
      for (let f = 0; f < 5; f++) mean[f] += feats[b + f];
    }
    for (let f = 0; f < 5; f++) mean[f] /= Math.max(1, rows.length);

    const std = [0, 0, 0, 0, 0];
    for (const r of rows) {
      const b = r.idx * 5;
      for (let f = 0; f < 5; f++) std[f] += (feats[b + f] - mean[f]) ** 2;
    }
    for (let f = 0; f < 5; f++) std[f] = Math.max(0.18, Math.sqrt(std[f] / Math.max(1, rows.length)));

    return { classId, mean, std, prior: rows.length / Math.max(1, trainSamples.length) };
  });

  // Learn one discriminative decision stump per (class, feature) pair
  const stumps: TrainedLocalModel['stumps'] = [];
  for (const cs of classStats) {
    const others = classStats.filter((o) => o.classId !== cs.classId);
    for (let f = 0; f < 5; f++) {
      const otherMean = others.reduce((acc, o) => acc + o.mean[f], 0) / Math.max(1, others.length);
      const sep = Math.abs(cs.mean[f] - otherMean) / (cs.std[f] + 0.25);
      stumps.push({
        featureIdx: f,
        threshold: 0.5 * (cs.mean[f] + otherMean),
        polarity: cs.mean[f] >= otherMean ? 1 : -1,
        targetClass: cs.classId,
        weight: Math.min(2.0, sep * 0.45),
      });
    }
  }

  const predictPixel = (pixelIdx: number): { cls: number; fgProb: number } => {
    const b = pixelIdx * 5;
    let bestCls = classStats[0].classId;
    let bestScore = -Infinity;
    const scores: Record<number, number> = {};
    for (const cs of classStats) {
      let logp = Math.log(cs.prior + 1e-4);
      for (let f = 0; f < 5; f++) {
        const z = (feats[b + f] - cs.mean[f]) / cs.std[f];
        logp -= 0.5 * z * z + Math.log(cs.std[f]);
      }
      scores[cs.classId] = logp;
    }
    for (const st of stumps) {
      const val = feats[b + st.featureIdx];
      const hit = st.polarity > 0 ? val >= st.threshold : val <= st.threshold;
      if (hit) scores[st.targetClass] = (scores[st.targetClass] ?? 0) + st.weight;
    }
    for (const cs of classStats) {
      if (scores[cs.classId] > bestScore) {
        bestScore = scores[cs.classId];
        bestCls = cs.classId;
      }
    }
    // Softmax probability of foreground class 1
    let sumExp = 0;
    for (const cs of classStats) sumExp += Math.exp(Math.max(-20, scores[cs.classId] - bestScore));
    const fgProb = scores[1] !== undefined ? Math.exp(Math.max(-20, scores[1] - bestScore)) / Math.max(1e-6, sumExp) : 0;
    return { cls: bestCls, fgProb };
  };

  // Hold-out evaluation
  let correct = 0;
  const tp: Record<number, number> = {};
  const fp: Record<number, number> = {};
  const fn: Record<number, number> = {};
  for (const s of valSamples) {
    const { cls: pred } = predictPixel(s.idx);
    if (pred === s.cls) {
      correct++;
      tp[s.cls] = (tp[s.cls] ?? 0) + 1;
    } else {
      fp[pred] = (fp[pred] ?? 0) + 1;
      fn[s.cls] = (fn[s.cls] ?? 0) + 1;
    }
  }

  const perClassIoU: Record<string, number> = {};
  const f1s: number[] = [];
  for (const cid of presentClasses) {
    const cTp = tp[cid] ?? 0;
    const cFp = fp[cid] ?? 0;
    const cFn = fn[cid] ?? 0;
    if (cTp + cFp + cFn === 0) continue;
    const iou = cTp / (cTp + cFp + cFn);
    const f1 = (2 * cTp) / (2 * cTp + cFp + cFn);
    const cname = classes[cid - 1] || `Class ${cid}`;
    perClassIoU[cname] = Number(iou.toFixed(4));
    f1s.push(f1);
  }
  const iouVals = Object.values(perClassIoU);
  const metrics: HoldoutMetrics = {
    nTrainPixels: trainSamples.length,
    nValPixels: valSamples.length,
    accuracy: Number((correct / Math.max(1, valSamples.length)).toFixed(4)),
    macroF1: Number((f1s.reduce((a, b) => a + b, 0) / Math.max(1, f1s.length)).toFixed(4)),
    meanIoU: Number((iouVals.reduce((a, b) => a + b, 0) / Math.max(1, iouVals.length)).toFixed(4)),
    perClassIoU,
  };

  const payloadStr = JSON.stringify({ modelId, version, classStats, stumps, specimen: specimen.sha256 });
  const sha256 = fastSha256Hex(payloadStr);

  const card: ModelCard = {
    id: modelId,
    version,
    ref: `${modelId}@${version}`,
    nameZh: `本地微调模型 (${modelId})`,
    nameEn: `Local Fine-Tuned (${modelId})`,
    task: presentClasses.length > 2 ? 'segment.classes' : 'segment.instances',
    framework: 'sklearn',
    modality: specimen.modality,
    license: 'Local-User-Owned',
    commercialUse: true,
    trainingData: [
      {
        name: `${specimen.name}${roi ? ` [ROI ${rw}×${rh}px]` : ''} (${strokes.length} strokes + ${corrections.length} corrections)`,
        license: 'Local-User-Owned',
      },
    ],
    sha256,
    tiledInference: true,
    tileSize: 512,
    fileStatus: 'local-trained',
    metrics: {
      meanIoU: metrics.meanIoU,
      macroF1: metrics.macroF1,
      accuracy: metrics.accuracy,
    },
    descriptionZh: `由${roi ? ` ROI (${rw}×${rh}px) 内` : ''}画笔标注与人工复核纠错实时微调的多尺度像素分类器（留出集 Mean IoU ${(metrics.meanIoU * 100).toFixed(1)}%）。`,
    descriptionEn: `Locally trained multi-scale pixel classifier from ${roi ? `ROI (${rw}×${rh}px) ` : ''}brush strokes & review corrections (Hold-out Mean IoU ${(metrics.meanIoU * 100).toFixed(1)}%).`,
    isLocalTrained: true,
    classes,
  };

  return { card, classes, classStats, stumps, metrics };
}

// Singleton Web Worker instance so incremental pyramid feature cache persists across rapid re-trains
let sharedDeepWorker: Worker | null = null;

function getDeepTrainWorker(): Worker | null {
  if (typeof Worker === 'undefined') return null;
  if (!sharedDeepWorker) {
    try {
      sharedDeepWorker = new Worker(new URL('./deepTrainWorker.ts', import.meta.url), {
        type: 'module',
      });
    } catch {
      return null;
    }
  }
  return sharedDeepWorker;
}

/** Count labeled pixels and distinct classes in the current annotation mask (used for GPU pre-flight gate). */
export function summarizeAnnotationMask(
  specimen: Specimen,
  strokes: AnnotationStroke[],
  corrections: HumanCorrection[],
  objects: Array<{ id: number; x: number; y: number; diameter?: number; excluded?: boolean }>,
  roi?: RoiRect | null
): { totalLabeledPixels: number; classPixelCounts: Record<number, number>; distinctClasses: number } {
  const mask = buildTrainingMask(specimen, strokes, corrections, objects, roi);
  const counts: Record<number, number> = {};
  let total = 0;
  for (let i = 0; i < mask.length; i++) {
    const c = mask[i];
    if (c > 0) {
      counts[c] = (counts[c] ?? 0) + 1;
      total++;
    }
  }
  return {
    totalLabeledPixels: total,
    classPixelCounts: counts,
    distinctClasses: Object.keys(counts).length,
  };
}

/**
 * Train a Few-Shot Deep Learning model (Frozen Multi-Scale Pyramid Encoder + Residual Bottleneck MLP)
 * inside a dedicated Web Worker with zero UI freeze, incremental feature caching, spatial block hold-out,
 * hard negative mining, and live epoch progress streaming.
 */
export async function trainDeepFewShotModelAsync(
  modelId: string,
  version: string,
  specimen: Specimen,
  strokes: AnnotationStroke[],
  corrections: HumanCorrection[],
  objects: Array<{ id: number; x: number; y: number; diameter?: number; excluded?: boolean }>,
  classes: string[],
  options: {
    workingGray?: Float32Array;
    roi?: RoiRect | null;
    config?: Partial<DeepFewShotConfig>;
    onProgress?: (ev: DeepTrainProgressEvent) => void;
    signal?: AbortSignal;
  } = {}
): Promise<TrainedLocalModel> {
  const fullW = specimen.width;
  const maxY = specimen.calibration.cropRows ?? specimen.height;
  const roi = options.roi ?? null;
  const rx = roi ? Math.max(0, Math.min(fullW - 1, Math.round(roi.x))) : 0;
  const ry = roi ? Math.max(0, Math.min(maxY - 1, Math.round(roi.y))) : 0;
  const rw = roi ? Math.max(1, Math.min(fullW - rx, Math.round(roi.width))) : fullW;
  const rh = roi ? Math.max(1, Math.min(maxY - ry, Math.round(roi.height))) : specimen.height;

  const baseGray =
    options.workingGray && options.workingGray.length === fullW * specimen.height
      ? options.workingGray
      : specimen.gray;

  let patchGray: Float32Array;
  if (roi) {
    patchGray = new Float32Array(rw * rh);
    for (let y = 0; y < rh; y++) {
      const srcOff = (ry + y) * fullW + rx;
      const dstOff = y * rw;
      for (let x = 0; x < rw; x++) {
        patchGray[dstOff + x] = baseGray[srcOff + x];
      }
    }
  } else {
    patchGray = new Float32Array(baseGray);
  }

  const mask = buildTrainingMask(specimen, strokes, corrections, objects, roi);
  const presentClasses = new Set<number>();
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 0) presentClasses.add(mask[i]);
  }
  if (presentClasses.size < 2) {
    throw new Error(
      roi
        ? '当前已启用 ROI：请在 ROI 区域内至少标注两个类别（例如“目标前景”与“背景”），再启动训练 / Label at least two classes inside the active ROI before training.'
        : '请至少标注两个类别（例如“目标前景”与“背景”），再启动训练 / Label at least two classes before training.'
    );
  }

  const config: DeepFewShotConfig = {
    hiddenDim: options.config?.hiddenDim ?? 24,
    epochs: options.config?.epochs ?? 32,
    learningRate: options.config?.learningRate ?? 0.025,
    hardNegativeMining: options.config?.hardNegativeMining ?? true,
    bilateralSmooth: options.config?.bilateralSmooth ?? true,
  };

  const cacheKey = `${specimen.sha256}_${rx}_${ry}_${rw}_${rh}_${options.workingGray ? 'pre' : 'raw'}`;
  const worker = getDeepTrainWorker();

  const outcome = await (worker
    ? new Promise<ReturnType<typeof trainDeepFewShotHead>>((resolve, reject) => {
        const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

        const cleanup = () => {
          worker.removeEventListener('message', onMsg);
          worker.removeEventListener('error', onErr);
          if (options.signal) options.signal.removeEventListener('abort', onAbort);
        };

        const onAbort = () => {
          worker.postMessage({ type: 'cancel', jobId } satisfies DeepTrainWorkerRequest);
          cleanup();
          reject(new Error('训练已取消 / Training cancelled.'));
        };

        const onMsg = (ev: MessageEvent<DeepTrainWorkerResponse>) => {
          const data = ev.data;
          if (!data || data.jobId !== jobId) return;
          if (data.type === 'progress' && data.progress) {
            options.onProgress?.(data.progress);
          } else if (data.type === 'done' && data.outcome) {
            cleanup();
            resolve(data.outcome);
          } else if (data.type === 'error') {
            cleanup();
            reject(new Error(data.error || 'Worker training failed'));
          }
        };

        const onErr = (ev: ErrorEvent) => {
          cleanup();
          reject(new Error(ev.message || 'Worker runtime error'));
        };

        worker.addEventListener('message', onMsg);
        worker.addEventListener('error', onErr);
        if (options.signal) {
          if (options.signal.aborted) {
            onAbort();
            return;
          }
          options.signal.addEventListener('abort', onAbort);
        }

        const req: DeepTrainWorkerRequest = {
          type: 'train',
          jobId,
          cacheKey,
          gray: patchGray,
          mask,
          width: rw,
          height: rh,
          classes,
          config,
        };
        // Transfer typed array buffers zero-copy to worker thread
        worker.postMessage(req, [patchGray.buffer, mask.buffer]);
      })
    : Promise.resolve().then(() => {
        const feats = extractMultiScaleFeatures24(patchGray, rw, rh);
        return trainDeepFewShotHead(feats, mask, rw, classes, config, false, options.onProgress);
      }));

  const payloadStr = JSON.stringify({
    modelId,
    version,
    deepHead: outcome.weights,
    specimen: specimen.sha256,
  });
  const sha256 = fastSha256Hex(payloadStr);

  const card: ModelCard = {
    id: modelId,
    version,
    ref: `${modelId}@${version}`,
    nameZh: `少样本深度微调 (${modelId})`,
    nameEn: `Few-Shot Deep Head (${modelId})`,
    task: outcome.weights.classIds.length > 2 ? 'segment.classes' : 'segment.instances',
    framework: 'onnx',
    modality: specimen.modality,
    license: 'Local-User-Owned',
    commercialUse: true,
    trainingData: [
      {
        name: `${specimen.name}${roi ? ` [ROI ${rw}×${rh}px]` : ''} (${strokes.length} strokes + ${corrections.length} edits · Spatial Block Val)`,
        license: 'Local-User-Owned',
      },
    ],
    sha256,
    tiledInference: true,
    tileSize: 512,
    fileStatus: 'local-trained',
    metrics: {
      meanIoU: outcome.metrics.meanIoU,
      macroF1: outcome.metrics.macroF1,
      accuracy: outcome.metrics.accuracy,
    },
    descriptionZh: `冻结 24 通道多尺度特征金字塔（含 Hessian 曲率与定向 Gabor）+ 残差瓶颈非线性 MLP (${outcome.weights.hidden1}→${outcome.weights.hidden2})，512×512 羽化分块推理（空间分块留出 Mean IoU ${(outcome.metrics.meanIoU * 100).toFixed(1)}%）。`,
    descriptionEn: `Frozen 24-ch multi-scale pyramid (Hessian curvature + oriented Gabor) + Residual Bottleneck MLP (${outcome.weights.hidden1}→${outcome.weights.hidden2}), 512px cosine-feathered tiled inference (Spatial-block Mean IoU ${(outcome.metrics.meanIoU * 100).toFixed(1)}%).`,
    isLocalTrained: true,
    classes,
  };

  return {
    card,
    classes,
    deepHead: outcome.weights,
    classStats: [],
    stumps: [],
    metrics: outcome.metrics,
  };
}

/** Run a locally trained pixel classifier (Deep Few-Shot Tiled MLP or legacy fallback) to produce a 0..1 foreground probability map. */
export function predictLocalModel(
  model: TrainedLocalModel,
  gray: Float32Array,
  width: number,
  height: number,
  cropRows?: number
): Float32Array {
  if (model.deepHead) {
    return predictDeepFewShotTiled(
      model.deepHead,
      gray,
      width,
      height,
      cropRows,
      model.card.tileSize ?? 512,
      64
    );
  }
  const feats = extractPixelFeatures(gray, width, height);
  const out = new Float32Array(width * height);
  const maxY = cropRows ?? height;

  for (let y = 0; y < maxY; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const b = idx * 5;
      let bestScore = -Infinity;
      const scores: Record<number, number> = {};
      for (const cs of model.classStats) {
        let logp = Math.log(cs.prior + 1e-4);
        for (let f = 0; f < 5; f++) {
          const z = (feats[b + f] - cs.mean[f]) / cs.std[f];
          logp -= 0.5 * z * z + Math.log(cs.std[f]);
        }
        scores[cs.classId] = logp;
        if (logp > bestScore) bestScore = logp;
      }
      for (const st of model.stumps) {
        const val = feats[b + st.featureIdx];
        const hit = st.polarity > 0 ? val >= st.threshold : val <= st.threshold;
        if (hit) {
          scores[st.targetClass] = (scores[st.targetClass] ?? 0) + st.weight;
          if (scores[st.targetClass] > bestScore) bestScore = scores[st.targetClass];
        }
      }
      let sumExp = 0;
      for (const cs of model.classStats) sumExp += Math.exp(Math.max(-20, scores[cs.classId] - bestScore));
      out[idx] = scores[1] !== undefined ? Math.exp(Math.max(-20, scores[1] - bestScore)) / Math.max(1e-6, sumExp) : 0;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Fast SAM-Style Geodesic Active-Boundary Click Segmentation & Morphometry
// ---------------------------------------------------------------------------

interface PrecomputedEdgeField {
  width: number;
  height: number;
  smooth: Float32Array;
  grad: Float32Array;
  globalGradP75: number;
}

const edgeFieldCache = new WeakMap<Float32Array, PrecomputedEdgeField>();

function getPrecomputedEdgeField(
  gray: Float32Array,
  width: number,
  height: number
): PrecomputedEdgeField {
  const cached = edgeFieldCache.get(gray);
  if (cached && cached.width === width && cached.height === height) {
    return cached;
  }

  const n = width * height;
  const smooth = new Float32Array(n);
  const grad = new Float32Array(n);

  // 3x3 Gaussian smoothing
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const ym1 = (y > 0 ? y - 1 : 0) * width;
    const yp1 = (y + 1 < height ? y + 1 : height - 1) * width;
    for (let x = 0; x < width; x++) {
      const xm1 = x > 0 ? x - 1 : 0;
      const xp1 = x + 1 < width ? x + 1 : width - 1;
      smooth[row + x] =
        (gray[ym1 + xm1] +
          2 * gray[ym1 + x] +
          gray[ym1 + xp1] +
          2 * gray[row + xm1] +
          4 * gray[row + x] +
          2 * gray[row + xp1] +
          gray[yp1 + xm1] +
          2 * gray[yp1 + x] +
          gray[yp1 + xp1]) *
        0.0625;
    }
  }

  // Scharr gradient magnitude (rotationally isotropic for bright SEM rims & dark crevices)
  const samples: number[] = [];
  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    const ym1 = row - width;
    const yp1 = row + width;
    for (let x = 1; x < width - 1; x++) {
      const gx =
        (3 * smooth[ym1 + x + 1] +
          10 * smooth[row + x + 1] +
          3 * smooth[yp1 + x + 1] -
          (3 * smooth[ym1 + x - 1] + 10 * smooth[row + x - 1] + 3 * smooth[yp1 + x - 1])) *
        0.0625;
      const gy =
        (3 * smooth[yp1 + x - 1] +
          10 * smooth[yp1 + x] +
          3 * smooth[yp1 + x + 1] -
          (3 * smooth[ym1 + x - 1] + 10 * smooth[ym1 + x] + 3 * smooth[ym1 + x + 1])) *
        0.0625;
      const gm = Math.hypot(gx, gy);
      grad[row + x] = gm;
      if (((x + y) & 7) === 0) samples.push(gm);
    }
  }
  samples.sort((a, b) => a - b);
  const globalGradP75 =
    samples.length > 0 ? Math.max(12, samples[Math.floor(samples.length * 0.75)]) : 22;

  const field: PrecomputedEdgeField = { width, height, smooth, grad, globalGradP75 };
  edgeFieldCache.set(gray, field);
  return field;
}

/** Ray-casting even-odd point-in-polygon check (supports optional internal holes). */
export function pointInPolygon(
  px: number,
  py: number,
  polygon: Array<[number, number]>,
  holes?: Array<Array<[number, number]>>
): boolean {
  const insideRing = (ring: Array<[number, number]>) => {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i][0];
      const yi = ring[i][1];
      const xj = ring[j][0];
      const yj = ring[j][1];
      const intersect =
        yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-12) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  };
  if (!insideRing(polygon)) return false;
  if (holes) {
    for (const h of holes) {
      if (h.length >= 3 && insideRing(h)) return false;
    }
  }
  return true;
}

/** Compute full calibrated physical morphometry (length, width, area, perimeter, eq. diameter, AR, circularity, orientation). */
function computeRegionMorphometry(
  localMask: Uint8Array,
  pw: number,
  ph: number,
  wx0: number,
  wy0: number,
  outer: Array<[number, number]>,
  holes: Array<Array<[number, number]>>,
  gray: Float32Array,
  fullW: number,
  pixelSize: number,
  spaceUnit: string,
  boundaryContrast: number
): ObjectMeasurementMetrics {
  let areaPx = 0;
  let sumX = 0;
  let sumY = 0;
  let sumI = 0;
  let minLx = pw;
  let maxLx = 0;
  let minLy = ph;
  let maxLy = 0;

  for (let ly = 0; ly < ph; ly++) {
    const row = ly * pw;
    const gy = wy0 + ly;
    for (let lx = 0; lx < pw; lx++) {
      if (localMask[row + lx]) {
        const gx = wx0 + lx;
        areaPx++;
        sumX += gx + 0.5;
        sumY += gy + 0.5;
        sumI += gray[gy * fullW + gx];
        if (lx < minLx) minLx = lx;
        if (lx > maxLx) maxLx = lx;
        if (ly < minLy) minLy = ly;
        if (ly > maxLy) maxLy = ly;
      }
    }
  }

  const safeArea = Math.max(1, areaPx);
  const cx = sumX / safeArea;
  const cy = sumY / safeArea;
  const meanIntensity = sumI / safeArea;

  // Second central moments for PCA principal orientation
  let mu20 = 0;
  let mu02 = 0;
  let mu11 = 0;
  for (let ly = minLy; ly <= maxLy; ly++) {
    const row = ly * pw;
    const dy = wy0 + ly + 0.5 - cy;
    for (let lx = minLx; lx <= maxLx; lx++) {
      if (localMask[row + lx]) {
        const dx = wx0 + lx + 0.5 - cx;
        mu20 += dx * dx;
        mu02 += dy * dy;
        mu11 += dx * dy;
      }
    }
  }

  let theta = 0.5 * Math.atan2(2 * mu11, mu20 - mu02);
  let ux = Math.cos(theta);
  let uy = Math.sin(theta);
  let vx = -uy;
  let vy = ux;

  // Project outer contour vertices onto principal axes to obtain exact Length (major) and Width (minor)
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  let perimeterPx = 0;

  for (let i = 0; i < outer.length; i++) {
    const [px, py] = outer[i];
    const [nx, ny] = outer[(i + 1) % outer.length];
    perimeterPx += Math.hypot(nx - px, ny - py);
    const dx = px - cx;
    const dy = py - cy;
    const pu = dx * ux + dy * uy;
    const pv = dx * vx + dy * vy;
    if (pu < minU) minU = pu;
    if (pu > maxU) maxU = pu;
    if (pv < minV) minV = pv;
    if (pv > maxV) maxV = pv;
  }

  let lengthPx = Math.max(1, maxU - minU);
  let widthPx = Math.max(1, maxV - minV);
  if (widthPx > lengthPx) {
    // Ensure Length is always the longer axis
    const tmpLen = lengthPx;
    lengthPx = widthPx;
    widthPx = tmpLen;
    const tmpMin = minU;
    const tmpMax = maxU;
    minU = minV;
    maxU = maxV;
    minV = tmpMin;
    maxV = tmpMax;
    const tmpX = ux;
    const tmpY = uy;
    ux = vx;
    uy = vy;
    vx = -tmpX;
    vy = -tmpY;
    theta = Math.atan2(uy, ux);
  }

  const midU = 0.5 * (minU + maxU);
  const midV = 0.5 * (minV + maxV);
  const boxCx = cx + midU * ux + midV * vx;
  const boxCy = cy + midU * uy + midV * vy;
  const halfL = 0.5 * lengthPx;
  const halfW = 0.5 * widthPx;

  const majorAxis: [[number, number], [number, number]] = [
    [boxCx - halfL * ux, boxCy - halfL * uy],
    [boxCx + halfL * ux, boxCy + halfL * uy],
  ];
  const minorAxis: [[number, number], [number, number]] = [
    [boxCx - halfW * vx, boxCy - halfW * vy],
    [boxCx + halfW * vx, boxCy + halfW * vy],
  ];

  const pxSize = pixelSize > 0 ? pixelSize : 1;
  const areaPhys = areaPx * pxSize * pxSize;
  const lengthPhys = lengthPx * pxSize;
  const widthPhys = widthPx * pxSize;
  const perimeterPhys = Math.max(pxSize, perimeterPx * pxSize);
  const equivDiameterPhys = 2 * Math.sqrt(areaPhys / Math.PI);
  const aspectRatio = lengthPx / Math.max(1e-3, widthPx);
  const circularity = Math.max(
    0,
    Math.min(1, (4 * Math.PI * areaPx) / Math.max(1e-3, perimeterPx * perimeterPx))
  );
  let orientationDeg = ((theta * 180) / Math.PI) % 180;
  if (orientationDeg < 0) orientationDeg += 180;
  if (orientationDeg >= 90) orientationDeg -= 180; // [-90, 90)

  const bbox: RoiRect = {
    x: wx0 + minLx,
    y: wy0 + minLy,
    width: Math.max(1, maxLx - minLx + 1),
    height: Math.max(1, maxLy - minLy + 1),
  };

  return {
    polygon: outer,
    holes: holes.length > 0 ? holes : undefined,
    cx,
    cy,
    bbox,
    majorAxis,
    minorAxis,
    areaPx,
    areaPhys: Number(areaPhys.toFixed(2)),
    areaUnit: `${spaceUnit}²`,
    lengthPhys: Number(lengthPhys.toFixed(2)),
    widthPhys: Number(widthPhys.toFixed(2)),
    equivDiameterPhys: Number(equivDiameterPhys.toFixed(2)),
    perimeterPhys: Number(perimeterPhys.toFixed(2)),
    aspectRatio: Number(aspectRatio.toFixed(2)),
    circularity: Number(circularity.toFixed(3)),
    orientationDeg: Number(orientationDeg.toFixed(1)),
    meanIntensity: Number(meanIntensity.toFixed(1)),
    confidence: Number(Math.max(0.45, Math.min(0.99, boundaryContrast)).toFixed(2)),
  };
}

export interface SmartSegmentResult {
  points: Array<[number, number]>;
  holes: Array<Array<[number, number]>>;
  areaPx: number;
  cx: number;
  cy: number;
  metrics: ObjectMeasurementMetrics;
}

/**
 * Real-time SAM-style geodesic active-boundary click segmentation.
 * Uses multi-cue Scharr gradient barriers + min-heap Fast Marching + incremental O(1)
 * optimal perimeter-gradient cut detection so single clicks snap accurately to SEM rims,
 * dark crevices, cells, and droplets without leaking.
 */
export function segmentObjectAtPoint(
  gray: Float32Array,
  width: number,
  height: number,
  seedX: number,
  seedY: number,
  options: {
    cropRows?: number;
    roi?: RoiRect | null;
    pixelSize?: number;
    spaceUnit?: string;
    sensitivity?: number;
    negatives?: Array<{ x: number; y: number }>;
    extraPositives?: Array<{ x: number; y: number }>;
    maxRadiusPx?: number;
  } = {}
): SmartSegmentResult | null {
  const maxY = options.cropRows ? Math.min(height, options.cropRows) : height;
  const roi = options.roi ?? null;
  const boundX0 = roi ? Math.max(0, Math.round(roi.x)) : 0;
  const boundY0 = roi ? Math.max(0, Math.round(roi.y)) : 0;
  const boundX1 = roi ? Math.min(width - 1, Math.round(roi.x + roi.width) - 1) : width - 1;
  const boundY1 = roi ? Math.min(maxY - 1, Math.round(roi.y + roi.height) - 1) : maxY - 1;

  const sx = Math.round(seedX);
  const sy = Math.round(seedY);
  if (sx < boundX0 || sx > boundX1 || sy < boundY0 || sy > boundY1) return null;

  const field = getPrecomputedEdgeField(gray, width, height);
  const { smooth, grad, globalGradP75 } = field;

  const sens = Math.max(0.25, Math.min(2.4, options.sensitivity ?? 1.0));
  const baseHalfR = Math.max(
    48,
    Math.min(220, options.maxRadiusPx ?? Math.round(Math.max(width, height) * 0.36))
  );
  const halfR = Math.min(
    340,
    Math.round(baseHalfR * (sens > 1.0 ? 1 + (sens - 1.0) * 0.45 : 1))
  );

  // Local patch bounding window around seed(s)
  let wx0 = Math.max(boundX0, sx - halfR);
  let wy0 = Math.max(boundY0, sy - halfR);
  let wx1 = Math.min(boundX1, sx + halfR);
  let wy1 = Math.min(boundY1, sy + halfR);
  if (options.extraPositives) {
    for (const ep of options.extraPositives) {
      const ex = Math.max(boundX0, Math.min(boundX1, Math.round(ep.x)));
      const ey = Math.max(boundY0, Math.min(boundY1, Math.round(ep.y)));
      wx0 = Math.max(boundX0, Math.min(wx0, ex - 60));
      wy0 = Math.max(boundY0, Math.min(wy0, ey - 60));
      wx1 = Math.min(boundX1, Math.max(wx1, ex + 60));
      wy1 = Math.min(boundY1, Math.max(wy1, ey + 60));
    }
  }
  const pw = wx1 - wx0 + 1;
  const ph = wy1 - wy0 + 1;
  const patchN = pw * ph;
  if (patchN < 16) return null;

  // Sample 5x5 neighborhood around primary seed for appearance mean & local texture std
  let seedSum = 0;
  let seedSq = 0;
  let seedCnt = 0;
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const nx = sx + dx;
      const ny = sy + dy;
      if (nx >= wx0 && nx <= wx1 && ny >= wy0 && ny <= wy1) {
        const v = smooth[ny * width + nx];
        seedSum += v;
        seedSq += v * v;
        seedCnt++;
      }
    }
  }
  const seedMean = seedSum / Math.max(1, seedCnt);
  const seedStd = Math.max(6, Math.sqrt(Math.max(0, seedSq / Math.max(1, seedCnt) - seedMean * seedMean)));

  // Local edge-barrier calibration around seed (radius 80px)
  const localGrads: number[] = [];
  const gR = Math.min(80, halfR);
  for (let y = Math.max(wy0, sy - gR); y <= Math.min(wy1, sy + gR); y += 2) {
    const row = y * width;
    for (let x = Math.max(wx0, sx - gR); x <= Math.min(wx1, sx + gR); x += 2) {
      localGrads.push(grad[row + x]);
    }
  }
  localGrads.sort((a, b) => a - b);
  const localGradP75 =
    localGrads.length > 8
      ? localGrads[Math.floor(localGrads.length * 0.75)]
      : globalGradP75;
  const edgeBarrier = Math.max(10, (0.65 * localGradP75 + 0.35 * globalGradP75) * (0.72 + 0.32 * sens));
  const stepNorm = Math.max(10, seedStd * 2.1);
  const appearNorm = Math.max(24, 44 * Math.max(0.6, sens));

  // Fast binary min-heap Dijkstra over local patch
  const runGeodesic = (
    seeds: Array<[number, number]>,
    maxCost: number,
    maxPops: number,
    blockedMask?: Uint8Array
  ) => {
    const dist = new Float32Array(patchN);
    dist.fill(1e9);
    const popped = new Uint8Array(patchN);
    const order = new Int32Array(maxPops);
    const orderDist = new Float32Array(maxPops);
    let popCount = 0;

    // Parallel typed-array min-heap for zero GC churn
    const heapCap = Math.min(patchN * 2 + 64, 160000);
    const heapIdx = new Int32Array(heapCap);
    const heapVal = new Float32Array(heapCap);
    let heapSize = 0;

    const push = (idx: number, val: number) => {
      if (heapSize >= heapCap) return;
      let i = heapSize++;
      heapIdx[i] = idx;
      heapVal[i] = val;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (heapVal[p] <= heapVal[i]) break;
        const ti = heapIdx[p];
        const tv = heapVal[p];
        heapIdx[p] = heapIdx[i];
        heapVal[p] = heapVal[i];
        heapIdx[i] = ti;
        heapVal[i] = tv;
        i = p;
      }
    };

    const pop = (): number => {
      const top = heapIdx[0];
      heapSize--;
      if (heapSize > 0) {
        heapIdx[0] = heapIdx[heapSize];
        heapVal[0] = heapVal[heapSize];
        let i = 0;
        while (true) {
          const l = (i << 1) + 1;
          const r = l + 1;
          let smallest = i;
          if (l < heapSize && heapVal[l] < heapVal[smallest]) smallest = l;
          if (r < heapSize && heapVal[r] < heapVal[smallest]) smallest = r;
          if (smallest === i) break;
          const ti = heapIdx[i];
          const tv = heapVal[i];
          heapIdx[i] = heapIdx[smallest];
          heapVal[i] = heapVal[smallest];
          heapIdx[smallest] = ti;
          heapVal[smallest] = tv;
          i = smallest;
        }
      }
      return top;
    };

    for (const [gx, gy] of seeds) {
      if (gx < wx0 || gx > wx1 || gy < wy0 || gy > wy1) continue;
      const li = (gy - wy0) * pw + (gx - wx0);
      if (blockedMask && blockedMask[li]) continue;
      dist[li] = 0;
      push(li, 0);
    }

    const dirs: Array<[number, number, number]> = [
      [-1, 0, 1],
      [1, 0, 1],
      [0, -1, 1],
      [0, 1, 1],
      [-1, -1, 1.414],
      [1, -1, 1.414],
      [-1, 1, 1.414],
      [1, 1, 1.414],
    ];

    while (heapSize > 0 && popCount < maxPops) {
      const u = pop();
      if (popped[u]) continue;
      const du = dist[u];
      if (du > maxCost) break;
      popped[u] = 1;
      order[popCount] = u;
      orderDist[popCount] = du;
      popCount++;

      const ux = u % pw;
      const uy = (u - ux) / pw;
      const uGlobal = (wy0 + uy) * width + (wx0 + ux);
      const su = smooth[uGlobal];

      for (let d = 0; d < 8; d++) {
        const vx = ux + dirs[d][0];
        const vy = uy + dirs[d][1];
        if (vx < 0 || vx >= pw || vy < 0 || vy >= ph) continue;
        const v = vy * pw + vx;
        if (popped[v] || (blockedMask && blockedMask[v])) continue;

        const vGlobal = (wy0 + vy) * width + (wx0 + vx);
        const gv = grad[vGlobal];
        const sv = smooth[vGlobal];

        const gRatio = gv / edgeBarrier;
        const gradCost = 2.75 * gRatio * gRatio;
        const stepJump = 0.055 * (Math.abs(sv - su) / stepNorm);
        const appearDrift = 0.015 * (Math.abs(sv - seedMean) / appearNorm);
        const nd = du + dirs[d][2] * (0.055 + gradCost + stepJump + appearDrift);

        if (nd < dist[v] && nd <= maxCost) {
          dist[v] = nd;
          push(v, nd);
        }
      }
    }

    return { dist, order, orderDist, popCount };
  };

  const maxCost = 14.5 * Math.pow(Math.max(0.5, sens), 1.35);
  const maxPops = Math.min(95000, Math.floor(patchN * (sens > 1.2 ? 0.82 : 0.65)));

  // Optional competing negative geodesic wavefront (excludes touching adjacent particles / background)
  let negDist: Float32Array | null = null;
  let negBlocked: Uint8Array | undefined;
  const validNegs: Array<[number, number]> = [];
  if (options.negatives && options.negatives.length > 0) {
    negBlocked = new Uint8Array(patchN);
    for (const neg of options.negatives) {
      const nx = Math.round(neg.x);
      const ny = Math.round(neg.y);
      if (nx >= wx0 && nx <= wx1 && ny >= wy0 && ny <= wy1) {
        validNegs.push([nx, ny]);
        // Hard 4px exclusion disk right around negative click
        for (let dy = -4; dy <= 4; dy++) {
          for (let dx = -4; dx <= 4; dx++) {
            if (dx * dx + dy * dy <= 16) {
              const bx = nx + dx - wx0;
              const by = ny + dy - wy0;
              if (bx >= 0 && bx < pw && by >= 0 && by < ph) {
                negBlocked[by * pw + bx] = 1;
              }
            }
          }
        }
      }
    }
    if (validNegs.length > 0) {
      negDist = runGeodesic(validNegs, maxCost, Math.min(36000, maxPops)).dist;
    }
  }

  const posSeeds: Array<[number, number]> = [[sx, sy]];
  if (options.extraPositives) {
    for (const ep of options.extraPositives) {
      posSeeds.push([Math.round(ep.x), Math.round(ep.y)]);
    }
  }

  const { dist: posDist, order, orderDist, popCount } = runGeodesic(
    posSeeds,
    maxCost,
    maxPops,
    negBlocked
  );
  if (popCount < 10) return null;

  // Incremental O(1) perimeter-gradient optimal cut selection along the geodesic expansion order
  const inCut = new Uint8Array(patchN);
  const degIn = new Uint8Array(patchN);
  let bSumGrad = 0;
  let bCount = 0;
  let borderTouches = 0;
  let validAdded = 0;

  let bestK = Math.min(popCount, 24);
  let bestScore = -Infinity;
  let bestBoundaryGrad = localGradP75;
  const evalSamples: Array<{ k: number; meanBGrad: number; validAdded: number }> = [];

  const evalStep = Math.max(6, Math.floor(popCount / 72));

  for (let k = 0; k < popCount; k++) {
    const p = order[k];
    if (negDist && negDist[p] <= posDist[p] * 1.03) continue;

    const lx = p % pw;
    const ly = (p - lx) / pw;
    const pGrad = grad[(wy0 + ly) * width + (wx0 + lx)];

    let nIn = 0;
    if (lx > 0 && inCut[p - 1]) {
      nIn++;
      const q = p - 1;
      if (degIn[q] === 3) {
        const qx = lx - 1;
        bSumGrad -= grad[(wy0 + ly) * width + (wx0 + qx)];
        bCount--;
      }
      degIn[q]++;
    }
    if (lx + 1 < pw && inCut[p + 1]) {
      nIn++;
      const q = p + 1;
      if (degIn[q] === 3) {
        const qx = lx + 1;
        bSumGrad -= grad[(wy0 + ly) * width + (wx0 + qx)];
        bCount--;
      }
      degIn[q]++;
    }
    if (ly > 0 && inCut[p - pw]) {
      nIn++;
      const q = p - pw;
      if (degIn[q] === 3) {
        bSumGrad -= grad[(wy0 + ly - 1) * width + (wx0 + lx)];
        bCount--;
      }
      degIn[q]++;
    }
    if (ly + 1 < ph && inCut[p + pw]) {
      nIn++;
      const q = p + pw;
      if (degIn[q] === 3) {
        bSumGrad -= grad[(wy0 + ly + 1) * width + (wx0 + lx)];
        bCount--;
      }
      degIn[q]++;
    }

    inCut[p] = 1;
    degIn[p] = nIn;
    validAdded++;
    if (nIn < 4) {
      bSumGrad += pGrad;
      bCount++;
    }
    if (lx === 0 || lx === pw - 1 || ly === 0 || ly === ph - 1) {
      borderTouches++;
    }

    if (validAdded >= 16 && (k % evalStep === 0 || k === popCount - 1)) {
      const meanBGrad = bSumGrad / Math.max(1, bCount);
      const areaFrac = validAdded / patchN;
      const borderFrac = borderTouches / Math.max(1, bCount);
      const areaPrior = Math.min(1.0, Math.log10(validAdded + 12) / 2.25);
      const leakPenalty =
        (areaFrac > 0.38 ? (areaFrac - 0.38) * 95 : 0) +
        (borderFrac > 0.25 ? (borderFrac - 0.25) * 45 : 0);
      const score = meanBGrad * areaPrior - leakPenalty;

      evalSamples.push({ k: k + 1, meanBGrad, validAdded });

      if (score > bestScore) {
        bestScore = score;
        bestK = k + 1;
        bestBoundaryGrad = meanBGrad;
      }
    }
  }

  // Smooth monotonic boundary shrink (sens < 1.0) or expansion (sens > 1.0) along the geodesic wavefront
  if (Math.abs(sens - 1.0) >= 0.015) {
    const anchorK = Math.max(16, Math.min(popCount, bestK));
    const anchorDist = Math.max(0.25, orderDist[Math.max(0, anchorK - 1)]);
    const maxSafeK = Math.min(popCount, Math.floor(patchN * 0.78));

    if (sens < 1.0) {
      const targetDist = anchorDist * Math.pow(sens, 1.45);
      let distK = 12;
      for (let k = 12; k < anchorK; k++) {
        distK = k + 1;
        if (orderDist[k] >= targetDist) break;
      }
      const areaCapK = Math.max(12, Math.round(anchorK * Math.pow(sens, 1.65)));
      bestK = Math.max(12, Math.min(anchorK, Math.round(0.62 * distK + 0.38 * areaCapK)));
    } else {
      const targetDist = anchorDist * Math.pow(sens, 1.55) + (sens - 1.0) * 1.4;
      let distK = anchorK;
      for (let k = anchorK; k < maxSafeK; k++) {
        if (orderDist[k] > targetDist) break;
        distK = k + 1;
      }
      const areaTargetK = Math.min(maxSafeK, Math.round(anchorK * Math.pow(sens, 1.65)));
      const rawExpandedK = Math.max(
        anchorK,
        Math.min(maxSafeK, Math.round(0.65 * distK + 0.35 * areaTargetK))
      );
      // Lightly snap to local gradient peak within ±12% of the expanded target so edges stay crisp
      const winLo = Math.round(rawExpandedK * 0.88);
      const winHi = Math.round(rawExpandedK * 1.12);
      let snappedK = rawExpandedK;
      let peakGrad = -1;
      for (const s of evalSamples) {
        if (s.k >= winLo && s.k <= winHi && s.meanBGrad > peakGrad) {
          peakGrad = s.meanBGrad;
          snappedK = s.k;
        }
      }
      bestK = Math.max(anchorK, Math.min(maxSafeK, snappedK));
      if (peakGrad > 0) bestBoundaryGrad = peakGrad;
    }
  }

  // Build final binary local mask from order[0..bestK-1]
  const localMask = new Uint8Array(patchN);
  let areaPx = 0;
  for (let k = 0; k < bestK; k++) {
    const p = order[k];
    if (negDist && negDist[p] <= posDist[p] * 1.03) continue;
    localMask[p] = 1;
    areaPx++;
  }
  if (areaPx < 10) return null;

  // 1-pass enclosed small-hole fill inside patch (keeps genuine large annular cavities >18% area)
  const extVisited = new Uint8Array(patchN);
  const extQueue = new Int32Array(patchN);
  let eqHead = 0;
  let eqTail = 0;
  const enqueueExt = (idx: number) => {
    if (!localMask[idx] && !extVisited[idx]) {
      extVisited[idx] = 1;
      extQueue[eqTail++] = idx;
    }
  };
  for (let x = 0; x < pw; x++) {
    enqueueExt(x);
    enqueueExt((ph - 1) * pw + x);
  }
  for (let y = 1; y < ph - 1; y++) {
    enqueueExt(y * pw);
    enqueueExt(y * pw + pw - 1);
  }
  while (eqHead < eqTail) {
    const idx = extQueue[eqHead++];
    const x = idx % pw;
    const y = (idx - x) / pw;
    if (x > 0) enqueueExt(idx - 1);
    if (x + 1 < pw) enqueueExt(idx + 1);
    if (y > 0) enqueueExt(idx - pw);
    if (y + 1 < ph) enqueueExt(idx + pw);
  }
  let enclosedCount = 0;
  for (let i = 0; i < patchN; i++) {
    if (!localMask[i] && !extVisited[i]) enclosedCount++;
  }
  if (enclosedCount > 0 && enclosedCount <= areaPx * 0.18) {
    for (let i = 0; i < patchN; i++) {
      if (!localMask[i] && !extVisited[i]) {
        localMask[i] = 1;
        areaPx++;
      }
    }
  }

  // Light 3x3 cross opening for larger objects (>= 90 px) to shear 1px crevice tendrils,
  // preserving the connected component containing the primary seed
  if (areaPx >= 90) {
    const eroded = new Uint8Array(patchN);
    for (let y = 1; y < ph - 1; y++) {
      const row = y * pw;
      for (let x = 1; x < pw - 1; x++) {
        const i = row + x;
        if (
          localMask[i] &&
          localMask[i - 1] &&
          localMask[i + 1] &&
          localMask[i - pw] &&
          localMask[i + pw]
        ) {
          eroded[i] = 1;
        }
      }
    }
    const seedLocal = (sy - wy0) * pw + (sx - wx0);
    // Only apply opening if seed survives erosion
    if (eroded[seedLocal]) {
      const dilated = new Uint8Array(patchN);
      for (let y = 1; y < ph - 1; y++) {
        const row = y * pw;
        for (let x = 1; x < pw - 1; x++) {
          const i = row + x;
          if (eroded[i]) {
            dilated[i] = 1;
            dilated[i - 1] = 1;
            dilated[i + 1] = 1;
            dilated[i - pw] = 1;
            dilated[i + pw] = 1;
          }
        }
      }
      // Keep only connected component containing primary seed
      const ccMask = new Uint8Array(patchN);
      const q = new Int32Array(patchN);
      let qh = 0;
      let qt = 0;
      ccMask[seedLocal] = 1;
      q[qt++] = seedLocal;
      while (qh < qt) {
        const idx = q[qh++];
        const x = idx % pw;
        const y = (idx - x) / pw;
        if (x > 0 && dilated[idx - 1] && !ccMask[idx - 1]) { ccMask[idx - 1] = 1; q[qt++] = idx - 1; }
        if (x + 1 < pw && dilated[idx + 1] && !ccMask[idx + 1]) { ccMask[idx + 1] = 1; q[qt++] = idx + 1; }
        if (y > 0 && dilated[idx - pw] && !ccMask[idx - pw]) { ccMask[idx - pw] = 1; q[qt++] = idx - pw; }
        if (y + 1 < ph && dilated[idx + pw] && !ccMask[idx + pw]) { ccMask[idx + pw] = 1; q[qt++] = idx + pw; }
      }
      if (qt >= 16) {
        localMask.set(ccMask);
        areaPx = qt;
      }
    }
  }

  // Compute local tight bounding box for contour tracing
  let minLx = pw;
  let maxLx = 0;
  let minLy = ph;
  let maxLy = 0;
  for (let y = 0; y < ph; y++) {
    const row = y * pw;
    for (let x = 0; x < pw; x++) {
      if (localMask[row + x]) {
        if (x < minLx) minLx = x;
        if (x > maxLx) maxLx = x;
        if (y < minLy) minLy = y;
        if (y > maxLy) maxLy = y;
      }
    }
  }
  if (minLx > maxLx || minLy > maxLy) return null;

  const traced = traceMaskPixelContours(localMask, pw, ph, minLx, maxLx, minLy, maxLy);
  if (traced.outer.length < 3) return null;

  // Offset traced local coordinates back to full image coordinates
  const outer: Array<[number, number]> = traced.outer.map(([x, y]) => [x + wx0, y + wy0]);
  const holes: Array<Array<[number, number]>> = traced.holes.map((h) =>
    h.map(([x, y]) => [x + wx0, y + wy0])
  );

  const boundaryContrast = Math.min(0.99, 0.5 + (bestBoundaryGrad / Math.max(10, localGradP75 * 1.6)) * 0.48);
  const metrics = computeRegionMorphometry(
    localMask,
    pw,
    ph,
    wx0,
    wy0,
    outer,
    holes,
    gray,
    width,
    options.pixelSize ?? 1,
    options.spaceUnit ?? 'px',
    boundaryContrast
  );

  return {
    points: outer,
    holes,
    areaPx: metrics.areaPx,
    cx: metrics.cx,
    cy: metrics.cy,
    metrics,
  };
}

/** Interactive SAM-style positive/negative click-prompt segmentation (uses the geodesic active-boundary engine). */
export function runPromptSegmentation(
  gray: Float32Array,
  width: number,
  height: number,
  prompts: PromptPoint[],
  cropRows?: number,
  roi?: RoiRect | null,
  pixelSize = 1,
  spaceUnit = 'px',
  sensitivity = 1.0
): SmartSegmentResult[] {
  const positives = prompts.filter((p) => p.positive);
  if (positives.length === 0) return [];
  const negatives = prompts.filter((p) => !p.positive);

  const results: SmartSegmentResult[] = [];
  for (const seed of positives) {
    // Skip if this positive point already lies inside a previously segmented contour
    if (results.some((r) => pointInPolygon(seed.x, seed.y, r.points, r.holes))) {
      continue;
    }
    const seg = segmentObjectAtPoint(gray, width, height, seed.x, seed.y, {
      cropRows,
      roi,
      pixelSize,
      spaceUnit,
      sensitivity,
      negatives,
    });
    if (seg) {
      results.push(seg);
    }
  }
  return results;
}

/**
 * SAM-style interactive Boundary Refine for an existing segmented object.
 * Extracts interior positive core anchors + an exterior negative guard ring from the coarse
 * segment contour, runs competitive geodesic active-boundary segmentation to snap tightly onto
 * true Scharr edge ridges, and reports real area delta & boundary gradient alignment improvement.
 */
export function refineSegmentBoundary(
  gray: Float32Array,
  width: number,
  height: number,
  shape: OverlayShape,
  obj: DetectionObject | undefined,
  options: {
    cropRows?: number;
    roi?: RoiRect | null;
    pixelSize?: number;
    spaceUnit?: string;
    sensitivity?: number;
  } = {}
): RefinedBoundaryProposal | null {
  if (shape.objectId === undefined) return null;
  const maxY = options.cropRows ? Math.min(height, options.cropRows) : height;
  const boundX0 = options.roi ? Math.max(0, Math.round(options.roi.x)) : 0;
  const boundY0 = options.roi ? Math.max(0, Math.round(options.roi.y)) : 0;
  const boundX1 = options.roi
    ? Math.min(width - 1, Math.round(options.roi.x + options.roi.width) - 1)
    : width - 1;
  const boundY1 = options.roi
    ? Math.min(maxY - 1, Math.round(options.roi.y + options.roi.height) - 1)
    : maxY - 1;

  // Build dense polygon representation of the existing coarse segment
  let originalPoly: Array<[number, number]> = [];
  let originalAreaPx = 0;

  if (shape.kind === 'polygon' && shape.points.length >= 3) {
    originalPoly = shape.points;
    let s = 0;
    for (let i = 0; i < originalPoly.length; i++) {
      const [x1, y1] = originalPoly[i];
      const [x2, y2] = originalPoly[(i + 1) % originalPoly.length];
      s += x1 * y2 - x2 * y1;
    }
    originalAreaPx = Math.max(1, Math.abs(s) * 0.5);
  } else if (shape.kind === 'circle' && shape.points.length >= 2) {
    const [cx0, cy0] = shape.points[0];
    const rx = Math.max(2, Math.abs(shape.points[1][0]));
    const ry = Math.max(2, Math.abs(shape.points[1][1] ?? rx));
    const nPts = 32;
    for (let i = 0; i < nPts; i++) {
      const a = (i / nPts) * Math.PI * 2;
      originalPoly.push([cx0 + Math.cos(a) * rx, cy0 + Math.sin(a) * ry]);
    }
    originalAreaPx = Math.max(1, Math.PI * rx * ry);
  } else if (shape.kind === 'box' && shape.points.length >= 2) {
    const [x0, y0] = shape.points[0];
    const [x1, y1] = shape.points[1];
    originalPoly = [
      [x0, y0],
      [x1, y0],
      [x1, y1],
      [x0, y1],
    ];
    originalAreaPx = Math.max(1, Math.abs((x1 - x0) * (y1 - y0)));
  } else if (obj) {
    const pxSize = options.pixelSize && options.pixelSize > 0 ? options.pixelSize : 1;
    const r = Math.max(4, ((obj.diameter ?? obj.width ?? 12) / (2 * pxSize)));
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      originalPoly.push([obj.x + Math.cos(a) * r, obj.y + Math.sin(a) * r]);
    }
    originalAreaPx = Math.max(1, Math.PI * r * r);
  } else {
    return null;
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxYBox = -Infinity;
  let sumX = 0;
  let sumY = 0;
  for (const [px, py] of originalPoly) {
    if (px < minX) minX = px;
    if (px > maxX) maxX = px;
    if (py < minY) minY = py;
    if (py > maxYBox) maxYBox = py;
    sumX += px;
    sumY += py;
  }
  const bboxW = Math.max(2, maxX - minX);
  const bboxH = Math.max(2, maxYBox - minY);
  const approxR = Math.max(5, Math.hypot(bboxW, bboxH) * 0.5);

  let seedX = obj ? obj.x : sumX / originalPoly.length;
  let seedY = obj ? obj.y : sumY / originalPoly.length;

  // Ensure primary seed lies strictly inside the coarse polygon (handles crescent / concave shapes)
  if (!pointInPolygon(seedX, seedY, originalPoly, shape.holes)) {
    const midY = 0.5 * (minY + maxYBox);
    const xs: number[] = [];
    for (let i = 0, j = originalPoly.length - 1; i < originalPoly.length; j = i++) {
      const [x1, y1] = originalPoly[i];
      const [x2, y2] = originalPoly[j];
      if ((y1 <= midY && y2 > midY) || (y2 <= midY && y1 > midY)) {
        xs.push(x1 + ((midY - y1) / (y2 - y1)) * (x2 - x1));
      }
    }
    xs.sort((a, b) => a - b);
    if (xs.length >= 2) {
      seedX = 0.5 * (xs[0] + xs[1]);
      seedY = midY;
    }
  }
  seedX = Math.max(boundX0, Math.min(boundX1, seedX));
  seedY = Math.max(boundY0, Math.min(boundY1, seedY));

  // Build interior positive anchors (shrunk 48% toward seed) and exterior negative guard ring
  const extraPositives: Array<{ x: number; y: number }> = [];
  const negatives: Array<{ x: number; y: number }> = [];
  const negMargin = Math.max(6, Math.min(20, approxR * 0.3));
  const posStep = Math.max(1, Math.floor(originalPoly.length / 4));
  const negStep = Math.max(1, Math.floor(originalPoly.length / 10));

  for (let i = 0; i < originalPoly.length; i += posStep) {
    if (extraPositives.length >= 4) break;
    const [vx, vy] = originalPoly[i];
    const px = seedX + (vx - seedX) * 0.45;
    const py = seedY + (vy - seedY) * 0.45;
    if (pointInPolygon(px, py, originalPoly, shape.holes)) {
      extraPositives.push({ x: px, y: py });
    }
  }

  for (let i = 0; i < originalPoly.length; i += negStep) {
    if (negatives.length >= 10) break;
    const [vx, vy] = originalPoly[i];
    const dx = vx - seedX;
    const dy = vy - seedY;
    const len = Math.hypot(dx, dy) || 1;
    const nx = Math.max(boundX0, Math.min(boundX1, vx + (dx / len) * negMargin));
    const ny = Math.max(boundY0, Math.min(boundY1, vy + (dy / len) * negMargin));
    if (!pointInPolygon(nx, ny, originalPoly, shape.holes)) {
      negatives.push({ x: nx, y: ny });
    }
  }

  const maxRadiusPx = Math.max(48, Math.min(220, Math.round(approxR * 1.45 + 24)));
  const refined = segmentObjectAtPoint(gray, width, height, seedX, seedY, {
    cropRows: options.cropRows,
    roi: options.roi,
    pixelSize: options.pixelSize,
    spaceUnit: options.spaceUnit,
    sensitivity: options.sensitivity ?? 1.0,
    extraPositives,
    negatives,
    maxRadiusPx,
  });
  if (!refined || refined.points.length < 3) return null;

  // Measure real Scharr boundary gradient alignment before vs. after refinement
  const field = getPrecomputedEdgeField(gray, width, height);
  const sampleContourGrad = (pts: Array<[number, number]>) => {
    if (pts.length === 0) return 0;
    let sum = 0;
    let cnt = 0;
    const stride = Math.max(1, Math.floor(pts.length / 80));
    for (let i = 0; i < pts.length; i += stride) {
      const gx = Math.max(0, Math.min(width - 1, Math.round(pts[i][0])));
      const gy = Math.max(0, Math.min(height - 1, Math.round(pts[i][1])));
      sum += field.grad[gy * width + gx];
      cnt++;
    }
    return sum / Math.max(1, cnt);
  };

  const edgeBefore = sampleContourGrad(originalPoly);
  const edgeAfter = sampleContourGrad(refined.points);
  const edgeGainPct = ((edgeAfter - edgeBefore) / Math.max(1, edgeBefore)) * 100;
  const areaDeltaPct = ((refined.areaPx - originalAreaPx) / Math.max(1, originalAreaPx)) * 100;

  return {
    objectId: shape.objectId,
    shapeId: shape.id,
    originalKind: shape.kind,
    originalPoints: originalPoly,
    originalHoles: shape.holes,
    refinedPoints: refined.points,
    refinedHoles: refined.holes,
    metrics: refined.metrics,
    originalAreaPx: Math.round(originalAreaPx),
    areaDeltaPct: Number(areaDeltaPct.toFixed(1)),
    edgeAlignmentBefore: Number(edgeBefore.toFixed(1)),
    edgeAlignmentAfter: Number(edgeAfter.toFixed(1)),
    edgeGainPct: Number(edgeGainPct.toFixed(1)),
  };
}
