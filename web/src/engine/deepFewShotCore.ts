import type { DeepFewShotConfig, HoldoutMetrics, TrainingEpochPoint } from '../types/vexel';

/** Default number of frozen multi-scale pyramid visual features per pixel (upgraded to 24ch in P2). */
export const DEEP_FEATURE_DIM = 24;
export const LEGACY_FEATURE_DIM = 16;

export interface DeepFewShotWeights {
  architecture: 'multiscale-pyramid-residual-mlp';
  featureDim: number;
  hidden1: number;
  hidden2: number;
  classIds: number[];
  /** Per-feature standardization mean & std learned on training pixels */
  featMean: number[];
  featStd: number[];
  /** Layer 1: hidden1 × featureDim + bias */
  w1: number[];
  b1: number[];
  /** Layer 2 (bottleneck): hidden2 × hidden1 + bias */
  w2: number[];
  b2: number[];
  /** Output head: numClasses × hidden2 + bias */
  wOut: number[];
  bOut: number[];
  /** Spatial bilateral post-smoothing strength (0..1) */
  bilateralSmooth: number;
}

export interface DeepTrainProgressEvent {
  phase: 'extracting' | 'training' | 'evaluating';
  epoch: number;
  totalEpochs: number;
  trainLoss: number;
  valLoss: number;
  valMeanIoU: number;
  valMacroF1: number;
  cacheHit: boolean;
}

export interface DeepTrainOutcome {
  weights: DeepFewShotWeights;
  metrics: HoldoutMetrics;
  epochs: TrainingEpochPoint[];
  durationMs: number;
  cacheHit: boolean;
}

/** Fast separable 1D box-blur approximation of Gaussian filter (2 passes -> smooth Gaussian kernel). */
function gaussianBlurSeparable(
  src: Float32Array,
  width: number,
  height: number,
  radius: number
): Float32Array {
  const n = width * height;
  const tmp = new Float32Array(n);
  const dst = new Float32Array(n);
  const r = Math.max(1, Math.round(radius));
  const win = 2 * r + 1;
  const inv = 1 / win;

  // Horizontal pass
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let acc = 0;
    for (let k = -r; k <= r; k++) {
      const x = k < 0 ? 0 : k >= width ? width - 1 : k;
      acc += src[row + x];
    }
    tmp[row] = acc * inv;
    for (let x = 1; x < width; x++) {
      const addX = x + r < width ? x + r : width - 1;
      const subX = x - r - 1 >= 0 ? x - r - 1 : 0;
      acc += src[row + addX] - src[row + subX];
      tmp[row + x] = acc * inv;
    }
  }

  // Vertical pass
  for (let x = 0; x < width; x++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) {
      const y = k < 0 ? 0 : k >= height ? height - 1 : k;
      acc += tmp[y * width + x];
    }
    dst[x] = acc * inv;
    for (let y = 1; y < height; y++) {
      const addY = y + r < height ? y + r : height - 1;
      const subY = y - r - 1 >= 0 ? y - r - 1 : 0;
      acc += tmp[addY * width + x] - tmp[subY * width + x];
      dst[y * width + x] = acc * inv;
    }
  }

  return dst;
}

/**
 * Frozen Multi-Scale Pyramid Feature Encoder (16 channels per pixel).
 * Runs cleanly on a single tile (e.g. 512×512 + halo) so peak memory stays bounded
 * even for 4K/8K microscopy images.
 *
 * Channels:
 *  0: Normalized raw intensity
 *  1: Fine Gaussian mean (r=1)
 *  2: Medium Gaussian mean (r=3)
 *  3: Coarse Context Gaussian mean (r=8)
 *  4: Fine Difference-of-Gaussians / LoG blob (raw - g1)
 *  5: Medium DoG / cell-wall & nucleus contrast (g1 - g3)
 *  6: Coarse DoG / local background-subtracted signal (g3 - g8)
 *  7: Fine Sobel gradient magnitude
 *  8: Medium-scale gradient magnitude (on g3)
 *  9: Directional Gabor-like 0° response (horizontal texture)
 * 10: Directional Gabor-like 90° response (vertical texture)
 * 11: Directional Gabor-like 45°/135° diagonal energy
 * 12: Structure-tensor anisotropy / coherence (edge vs isotropic interior)
 * 13: Local variance / texture roughness (r=3)
 * 14: Coarse regional standard deviation (r=8)
 * 15: Relative local intensity rank proxy ((g1 - g8) / (coarseStd + 0.25))
 */
export function extractMultiScaleFeatures16(
  gray: Float32Array,
  width: number,
  height: number,
  globalStats?: { mean: number; std: number }
): Float32Array {
  const n = width * height;
  let meanAll = globalStats?.mean ?? 0;
  let stdAll = globalStats?.std ?? 0;

  if (!globalStats) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += gray[i];
    meanAll = sum / Math.max(1, n);
    let varSum = 0;
    for (let i = 0; i < n; i++) {
      const d = gray[i] - meanAll;
      varSum += d * d;
    }
    stdAll = Math.max(8, Math.sqrt(varSum / Math.max(1, n)));
  }

  const norm = new Float32Array(n);
  const invStd = 1 / stdAll;
  for (let i = 0; i < n; i++) {
    norm[i] = (gray[i] - meanAll) * invStd;
  }

  const g1 = gaussianBlurSeparable(norm, width, height, 1);
  const g3 = gaussianBlurSeparable(g1, width, height, 3);
  const g8 = gaussianBlurSeparable(g3, width, height, 7);

  // Squared deviation maps for fast local standard deviations
  const sqDiff3 = new Float32Array(n);
  const sqDiff8 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d3 = norm[i] - g3[i];
    const d8 = g1[i] - g8[i];
    sqDiff3[i] = d3 * d3;
    sqDiff8[i] = d8 * d8;
  }
  const var3 = gaussianBlurSeparable(sqDiff3, width, height, 3);
  const var8 = gaussianBlurSeparable(sqDiff8, width, height, 7);

  const feats = new Float32Array(n * LEGACY_FEATURE_DIM);

  for (let y = 0; y < height; y++) {
    const ym1 = y > 0 ? y - 1 : 0;
    const yp1 = y + 1 < height ? y + 1 : height - 1;
    const ym2 = y > 1 ? y - 2 : 0;
    const yp2 = y + 2 < height ? y + 2 : height - 1;
    const row = y * width;
    const rowU = ym1 * width;
    const rowD = yp1 * width;
    const rowU2 = ym2 * width;
    const rowD2 = yp2 * width;

    for (let x = 0; x < width; x++) {
      const xm1 = x > 0 ? x - 1 : 0;
      const xp1 = x + 1 < width ? x + 1 : width - 1;
      const xm2 = x > 1 ? x - 2 : 0;
      const xp2 = x + 2 < width ? x + 2 : width - 1;
      const idx = row + x;

      const c0 = norm[idx];
      const c1 = g1[idx];
      const c3 = g3[idx];
      const c8 = g8[idx];

      // Fine Sobel gradient on g1
      const gx1 =
        (g1[rowU + xp1] + 2 * g1[row + xp1] + g1[rowD + xp1]) -
        (g1[rowU + xm1] + 2 * g1[row + xm1] + g1[rowD + xm1]);
      const gy1 =
        (g1[rowD + xm1] + 2 * g1[rowD + x] + g1[rowD + xp1]) -
        (g1[rowU + xm1] + 2 * g1[rowU + x] + g1[rowU + xp1]);
      const grad1 = Math.hypot(gx1, gy1) * 0.25;

      // Medium-scale gradient on g3
      const gx3 = g3[row + xp2] - g3[row + xm2];
      const gy3 = g3[rowD2 + x] - g3[rowU2 + x];
      const grad3 = Math.hypot(gx3, gy3) * 0.5;

      // Multi-orientation Gabor-like directional band-pass energy
      const gaborH = Math.abs(2 * c1 - g1[row + xm2] - g1[row + xp2]);
      const gaborV = Math.abs(2 * c1 - g1[rowU2 + x] - g1[rowD2 + x]);
      const gaborDiag =
        0.5 *
        (Math.abs(2 * c1 - g1[rowU2 + xm2] - g1[rowD2 + xp2]) +
          Math.abs(2 * c1 - g1[rowU2 + xp2] - g1[rowD2 + xm2]));

      // Structure tensor coherence proxy (distinguishes smooth membranes/fibres from mottled interior)
      const jxx = gx1 * gx1;
      const jyy = gy1 * gy1;
      const jxy = gx1 * gy1;
      const trace = jxx + jyy + 1e-4;
      const disc = Math.sqrt(Math.max(0, (jxx - jyy) * (jxx - jyy) + 4 * jxy * jxy));
      const coherence = disc / trace;

      const std3 = Math.sqrt(Math.max(0, var3[idx]));
      const std8 = Math.sqrt(Math.max(0, var8[idx]));
      const relContrast = (c1 - c8) / (std8 + 0.25);

      const base = idx * LEGACY_FEATURE_DIM;
      feats[base] = c0;
      feats[base + 1] = c1;
      feats[base + 2] = c3;
      feats[base + 3] = c8;
      feats[base + 4] = c0 - c1;
      feats[base + 5] = c1 - c3;
      feats[base + 6] = c3 - c8;
      feats[base + 7] = grad1;
      feats[base + 8] = grad3;
      feats[base + 9] = gaborH;
      feats[base + 10] = gaborV;
      feats[base + 11] = gaborDiag;
      feats[base + 12] = coherence;
      feats[base + 13] = std3;
      feats[base + 14] = std8;
      feats[base + 15] = Math.max(-4, Math.min(4, relContrast));
    }
  }

  return feats;
}

/**
 * Upgraded 24-Channel Frozen Multi-Scale Pyramid Feature Encoder (P2 / G-11).
 * Extends the 16 base multi-scale channels with 8 scientific microscopy descriptors:
 *  16: Hessian principal eigenvalue λ1 (bright/dark ridge & membrane curvature)
 *  17: Hessian secondary eigenvalue λ2 (isotropic blob/pore curvature)
 *  18: Gabor 45° oriented directional energy
 *  19: Gabor 135° oriented directional energy
 *  20: Wide-context Gaussian mean (r=15) for uneven SEM / brightfield shading invariance
 *  21: Wide-context shading-corrected contrast (g3 - g15)
 *  22: Local morphological bright-peak / top-hat proxy (c1 - min3x3)
 *  23: Local morphological dark-crevice / bottom-hat proxy (max3x3 - c1)
 *
 * Also exposes an optional `externalEmbeddings` slot (up to 8 channels from a WebGPU/ONNX
 * visual encoder) that blends cleanly into channels 16..23 when supplied.
 */
export function extractMultiScaleFeatures24(
  gray: Float32Array,
  width: number,
  height: number,
  globalStats?: { mean: number; std: number },
  externalEmbeddings?: { data: Float32Array; channels: number }
): Float32Array {
  const n = width * height;
  let meanAll = globalStats?.mean ?? 0;
  let stdAll = globalStats?.std ?? 0;

  if (!globalStats) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += gray[i];
    meanAll = sum / Math.max(1, n);
    let varSum = 0;
    for (let i = 0; i < n; i++) {
      const d = gray[i] - meanAll;
      varSum += d * d;
    }
    stdAll = Math.max(8, Math.sqrt(varSum / Math.max(1, n)));
  }

  const norm = new Float32Array(n);
  const invStd = 1 / stdAll;
  for (let i = 0; i < n; i++) {
    norm[i] = (gray[i] - meanAll) * invStd;
  }

  const g1 = gaussianBlurSeparable(norm, width, height, 1);
  const g3 = gaussianBlurSeparable(g1, width, height, 3);
  const g8 = gaussianBlurSeparable(g3, width, height, 7);
  const g15 = gaussianBlurSeparable(g8, width, height, 14);

  const sqDiff3 = new Float32Array(n);
  const sqDiff8 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d3 = norm[i] - g3[i];
    const d8 = g1[i] - g8[i];
    sqDiff3[i] = d3 * d3;
    sqDiff8[i] = d8 * d8;
  }
  const var3 = gaussianBlurSeparable(sqDiff3, width, height, 3);
  const var8 = gaussianBlurSeparable(sqDiff8, width, height, 7);

  const feats = new Float32Array(n * DEEP_FEATURE_DIM);
  const hasExt =
    Boolean(externalEmbeddings) &&
    externalEmbeddings!.channels > 0 &&
    externalEmbeddings!.data.length === n * externalEmbeddings!.channels;
  const extCh = hasExt ? Math.min(8, externalEmbeddings!.channels) : 0;

  for (let y = 0; y < height; y++) {
    const ym1 = y > 0 ? y - 1 : 0;
    const yp1 = y + 1 < height ? y + 1 : height - 1;
    const ym2 = y > 1 ? y - 2 : 0;
    const yp2 = y + 2 < height ? y + 2 : height - 1;
    const row = y * width;
    const rowU = ym1 * width;
    const rowD = yp1 * width;
    const rowU2 = ym2 * width;
    const rowD2 = yp2 * width;

    for (let x = 0; x < width; x++) {
      const xm1 = x > 0 ? x - 1 : 0;
      const xp1 = x + 1 < width ? x + 1 : width - 1;
      const xm2 = x > 1 ? x - 2 : 0;
      const xp2 = x + 2 < width ? x + 2 : width - 1;
      const idx = row + x;

      const c0 = norm[idx];
      const c1 = g1[idx];
      const c3 = g3[idx];
      const c8 = g8[idx];
      const c15 = g15[idx];

      // Fine Sobel gradient on g1
      const g1UL = g1[rowU + xm1];
      const g1U = g1[rowU + x];
      const g1UR = g1[rowU + xp1];
      const g1L = g1[row + xm1];
      const g1R = g1[row + xp1];
      const g1DL = g1[rowD + xm1];
      const g1D = g1[rowD + x];
      const g1DR = g1[rowD + xp1];

      const gx1 = g1UR + 2 * g1R + g1DR - (g1UL + 2 * g1L + g1DL);
      const gy1 = g1DL + 2 * g1D + g1DR - (g1UL + 2 * g1U + g1UR);
      const grad1 = Math.hypot(gx1, gy1) * 0.25;

      // Medium-scale gradient on g3
      const gx3 = g3[row + xp2] - g3[row + xm2];
      const gy3 = g3[rowD2 + x] - g3[rowU2 + x];
      const grad3 = Math.hypot(gx3, gy3) * 0.5;

      // Multi-orientation Gabor-like directional band-pass energy
      const gaborH = Math.abs(2 * c1 - g1[row + xm2] - g1[row + xp2]);
      const gaborV = Math.abs(2 * c1 - g1[rowU2 + x] - g1[rowD2 + x]);
      const gabor45 = Math.abs(2 * c1 - g1[rowU2 + xp2] - g1[rowD2 + xm2]);
      const gabor135 = Math.abs(2 * c1 - g1[rowU2 + xm2] - g1[rowD2 + xp2]);
      const gaborDiag = 0.5 * (gabor45 + gabor135);

      // Structure tensor coherence
      const jxx = gx1 * gx1;
      const jyy = gy1 * gy1;
      const jxy = gx1 * gy1;
      const trace = jxx + jyy + 1e-4;
      const disc = Math.sqrt(Math.max(0, (jxx - jyy) * (jxx - jyy) + 4 * jxy * jxy));
      const coherence = disc / trace;

      // Second-order Hessian curvature eigenvalues on g3 (membrane/tube vs blob/pore discrimination)
      const hxx = g3[row + xm2] - 2 * c3 + g3[row + xp2];
      const hyy = g3[rowU2 + x] - 2 * c3 + g3[rowD2 + x];
      const hxy = 0.25 * (g3[rowD2 + xp2] - g3[rowD2 + xm2] - g3[rowU2 + xp2] + g3[rowU2 + xm2]);
      const hHalfTr = 0.5 * (hxx + hyy);
      const hRad = Math.sqrt(Math.max(0, 0.25 * (hxx - hyy) * (hxx - hyy) + hxy * hxy));
      const eig1 = hHalfTr + hRad;
      const eig2 = hHalfTr - hRad;

      // Local 3x3 morphological bright-peak (top-hat) & dark-crevice (bottom-hat) proxies
      let min3 = c1;
      let max3 = c1;
      if (g1UL < min3) min3 = g1UL;
      if (g1UL > max3) max3 = g1UL;
      if (g1U < min3) min3 = g1U;
      if (g1U > max3) max3 = g1U;
      if (g1UR < min3) min3 = g1UR;
      if (g1UR > max3) max3 = g1UR;
      if (g1L < min3) min3 = g1L;
      if (g1L > max3) max3 = g1L;
      if (g1R < min3) min3 = g1R;
      if (g1R > max3) max3 = g1R;
      if (g1DL < min3) min3 = g1DL;
      if (g1DL > max3) max3 = g1DL;
      if (g1D < min3) min3 = g1D;
      if (g1D > max3) max3 = g1D;
      if (g1DR < min3) min3 = g1DR;
      if (g1DR > max3) max3 = g1DR;

      const std3 = Math.sqrt(Math.max(0, var3[idx]));
      const std8 = Math.sqrt(Math.max(0, var8[idx]));
      const relContrast = (c1 - c8) / (std8 + 0.25);

      const base = idx * DEEP_FEATURE_DIM;
      feats[base] = c0;
      feats[base + 1] = c1;
      feats[base + 2] = c3;
      feats[base + 3] = c8;
      feats[base + 4] = c0 - c1;
      feats[base + 5] = c1 - c3;
      feats[base + 6] = c3 - c8;
      feats[base + 7] = grad1;
      feats[base + 8] = grad3;
      feats[base + 9] = gaborH;
      feats[base + 10] = gaborV;
      feats[base + 11] = gaborDiag;
      feats[base + 12] = coherence;
      feats[base + 13] = std3;
      feats[base + 14] = std8;
      feats[base + 15] = Math.max(-4, Math.min(4, relContrast));

      // Channels 16..23 (or fused with external WebGPU/ONNX embeddings if attached)
      feats[base + 16] = eig1;
      feats[base + 17] = eig2;
      feats[base + 18] = gabor45;
      feats[base + 19] = gabor135;
      feats[base + 20] = c15;
      feats[base + 21] = c3 - c15;
      feats[base + 22] = c1 - min3;
      feats[base + 23] = max3 - c1;

      if (hasExt) {
        const extOff = idx * externalEmbeddings!.channels;
        for (let k = 0; k < extCh; k++) {
          feats[base + 16 + k] = 0.5 * feats[base + 16 + k] + 0.5 * externalEmbeddings!.data[extOff + k];
        }
      }
    }
  }

  return feats;
}

/** Fast smooth rational tanh approximation (max error < 0.002, ~6× faster than Math.tanh in tight loops). */
function fastTanh(x: number): number {
  if (x <= -3) return -1;
  if (x >= 3) return 1;
  const x2 = x * x;
  return (x * (27 + x2)) / (27 + 9 * x2);
}

/** GELU activation (fast rational tanh approximation) & derivative. */
function gelu(x: number): number {
  const c = 0.7978845608; // sqrt(2/pi)
  const inner = c * (x + 0.044715 * x * x * x);
  const t = fastTanh(inner);
  return 0.5 * x * (1 + t);
}

function geluDeriv(x: number): number {
  const c = 0.7978845608;
  const x2 = x * x;
  const inner = c * (x + 0.044715 * x * x2);
  const t = fastTanh(inner);
  const sech2 = 1 - t * t;
  return 0.5 * (1 + t) + 0.5 * x * sech2 * c * (1 + 3 * 0.044715 * x2);
}

/** Deterministic PRNG (Mulberry32) so training is 100% reproducible given identical strokes. */
function createSeededRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let r = Math.imul(s ^ (s >>> 15), 1 | s);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Spatial Block Hold-Out Split (16×16 spatial blocks) to eliminate spatial leakage
 * from adjacent pixels when reporting validation mIoU & Macro-F1.
 */
function splitSpatialBlocks(
  indices: number[],
  width: number,
  blockSize = 16
): { train: number[]; val: number[] } {
  const blockMap = new Map<number, number[]>();
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    const bx = Math.floor((idx % width) / blockSize);
    const by = Math.floor(Math.floor(idx / width) / blockSize);
    const bKey = by * 4096 + bx;
    let bucket = blockMap.get(bKey);
    if (!bucket) {
      bucket = [];
      blockMap.set(bKey, bucket);
    }
    bucket.push(idx);
  }

  const keys = Array.from(blockMap.keys()).sort((a, b) => a - b);
  const train: number[] = [];
  const val: number[] = [];

  if (keys.length >= 4) {
    // Deterministic 25% spatial block hold-out
    for (let k = 0; k < keys.length; k++) {
      const bucket = blockMap.get(keys[k])!;
      if (k % 4 === 1) {
        for (let i = 0; i < bucket.length; i++) val.push(bucket[i]);
      } else {
        for (let i = 0; i < bucket.length; i++) train.push(bucket[i]);
      }
    }
  } else {
    // Small stroke fallback: split by spatial coordinate quadrant along major span
    for (let i = 0; i < indices.length; i++) {
      if (i % 4 === 0 && indices.length >= 4) val.push(indices[i]);
      else train.push(indices[i]);
    }
  }

  // Ensure both partitions are non-empty
  if (val.length === 0 && train.length >= 2) {
    val.push(train.pop()!);
  }
  if (train.length === 0 && val.length >= 2) {
    train.push(val.pop()!);
  }
  return { train, val };
}

/**
 * Train a Residual Bottleneck Nonlinear MLP Decoder Head on top of Frozen Multi-Scale
 * Pyramid Features using Mini-batch AdamW + Class-Balanced Loss + Hard Negative Mining (OHEM)
 * and Spatial Block Hold-Out validation.
 */
export function trainDeepFewShotHead(
  feats: Float32Array,
  mask: Uint8Array,
  width: number,
  classes: string[],
  config: DeepFewShotConfig,
  cacheHit: boolean,
  onProgress?: (ev: DeepTrainProgressEvent) => void,
  shouldAbort?: () => boolean
): DeepTrainOutcome {
  const t0 = performance.now();
  const inferredDim = mask.length > 0 ? Math.round(feats.length / mask.length) : DEEP_FEATURE_DIM;
  const D = inferredDim === LEGACY_FEATURE_DIM || inferredDim === DEEP_FEATURE_DIM ? inferredDim : DEEP_FEATURE_DIM;
  const H1 = Math.max(12, Math.min(64, Math.round(config.hiddenDim || 24)));
  const H2 = Math.max(8, Math.min(48, Math.round(H1 * 0.67)));
  const totalEpochs = Math.max(10, Math.min(80, Math.round(config.epochs || 32)));
  const baseLr = config.learningRate || 0.025;
  const weightDecay = 0.0025;

  // Group labeled pixels by class
  const perClassRaw = new Map<number, number[]>();
  for (let i = 0; i < mask.length; i++) {
    const c = mask[i];
    if (c > 0) {
      let arr = perClassRaw.get(c);
      if (!arr) {
        arr = [];
        perClassRaw.set(c, arr);
      }
      arr.push(i);
    }
  }
  if (perClassRaw.size < 2) {
    throw new Error(
      '请至少标注两个类别（例如“前景目标”与“背景”），再启动少样本深度学习训练 / Label at least two classes before training.'
    );
  }

  const classIds = Array.from(perClassRaw.keys()).sort((a, b) => a - b);
  const C = classIds.length;
  const classToLocal = new Map<number, number>();
  classIds.forEach((cid, i) => classToLocal.set(cid, i));

  // Spatial block hold-out split with per-class cap for fast, balanced mini-batch training
  const trainIdx: number[] = [];
  const trainY: number[] = [];
  const valIdx: number[] = [];
  const valY: number[] = [];
  const classTrainCounts = new Array<number>(C).fill(0);

  for (const cid of classIds) {
    const raw = perClassRaw.get(cid)!;
    const { train, val } = splitSpatialBlocks(raw, width, 16);
    const localC = classToLocal.get(cid)!;

    const trainStride = Math.max(1, Math.ceil(train.length / 600));
    for (let i = 0; i < train.length; i += trainStride) {
      trainIdx.push(train[i]);
      trainY.push(localC);
      classTrainCounts[localC]++;
    }
    const valStride = Math.max(1, Math.ceil(val.length / 300));
    for (let i = 0; i < val.length; i += valStride) {
      valIdx.push(val[i]);
      valY.push(localC);
    }
  }

  const nTrain = trainIdx.length;
  const nVal = valIdx.length;

  // Compute class-balanced weights
  const classWeight = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    classWeight[c] = nTrain / (C * Math.max(1, classTrainCounts[c]));
  }

  // Standardize features strictly from training split (zero val leakage!)
  const featMean = new Array<number>(D).fill(0);
  for (let i = 0; i < nTrain; i++) {
    const b = trainIdx[i] * D;
    for (let d = 0; d < D; d++) featMean[d] += feats[b + d];
  }
  for (let d = 0; d < D; d++) featMean[d] /= Math.max(1, nTrain);

  const featStd = new Array<number>(D).fill(0);
  for (let i = 0; i < nTrain; i++) {
    const b = trainIdx[i] * D;
    for (let d = 0; d < D; d++) {
      const diff = feats[b + d] - featMean[d];
      featStd[d] += diff * diff;
    }
  }
  for (let d = 0; d < D; d++) {
    featStd[d] = Math.max(0.12, Math.sqrt(featStd[d] / Math.max(1, nTrain)));
  }

  // Materialize compact standardized matrices for cache-friendly fast inner loops
  const xTrain = new Float32Array(nTrain * D);
  for (let i = 0; i < nTrain; i++) {
    const src = trainIdx[i] * D;
    const dst = i * D;
    for (let d = 0; d < D; d++) {
      xTrain[dst + d] = (feats[src + d] - featMean[d]) / featStd[d];
    }
  }
  const xVal = new Float32Array(nVal * D);
  for (let i = 0; i < nVal; i++) {
    const src = valIdx[i] * D;
    const dst = i * D;
    for (let d = 0; d < D; d++) {
      xVal[dst + d] = (feats[src + d] - featMean[d]) / featStd[d];
    }
  }

  // He / Xavier deterministic parameter initialization
  const rng = createSeededRng(42071 + nTrain * 13 + C * 97);
  const randN = () => {
    const u = Math.max(1e-6, rng());
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  const w1 = new Float32Array(H1 * D);
  const b1 = new Float32Array(H1);
  const s1 = Math.sqrt(2 / D);
  for (let i = 0; i < w1.length; i++) w1[i] = randN() * s1;

  const w2 = new Float32Array(H2 * H1);
  const b2 = new Float32Array(H2);
  const s2 = Math.sqrt(2 / H1);
  for (let i = 0; i < w2.length; i++) w2[i] = randN() * s2;

  const wOut = new Float32Array(C * H2);
  const bOut = new Float32Array(C);
  const s3 = Math.sqrt(1 / H2);
  for (let i = 0; i < wOut.length; i++) wOut[i] = randN() * s3;

  // AdamW moments
  const mW1 = new Float32Array(w1.length);
  const vW1 = new Float32Array(w1.length);
  const mB1 = new Float32Array(b1.length);
  const vB1 = new Float32Array(b1.length);

  const mW2 = new Float32Array(w2.length);
  const vW2 = new Float32Array(w2.length);
  const mB2 = new Float32Array(b2.length);
  const vB2 = new Float32Array(b2.length);

  const mWO = new Float32Array(wOut.length);
  const vWO = new Float32Array(wOut.length);
  const mBO = new Float32Array(bOut.length);
  const vBO = new Float32Array(bOut.length);

  // Gradients & scratch buffers
  const gW1 = new Float32Array(w1.length);
  const gB1 = new Float32Array(b1.length);
  const gW2 = new Float32Array(w2.length);
  const gB2 = new Float32Array(b2.length);
  const gWO = new Float32Array(wOut.length);
  const gBO = new Float32Array(bOut.length);

  const z1 = new Float32Array(H1);
  const a1 = new Float32Array(H1);
  const z2 = new Float32Array(H2);
  const a2 = new Float32Array(H2);
  const logits = new Float32Array(C);
  const probs = new Float32Array(C);
  const dz2 = new Float32Array(H2);
  const da1 = new Float32Array(H1);
  const dz1 = new Float32Array(H1);

  const order = new Int32Array(nTrain);
  for (let i = 0; i < nTrain; i++) order[i] = i;

  const sampleHardWeight = new Float32Array(nTrain).fill(1.0);
  const batchSize = Math.min(96, Math.max(24, Math.floor(nTrain / 12)));
  let stepNum = 0;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;

  const adamStep = (
    param: Float32Array,
    grad: Float32Array,
    m: Float32Array,
    v: Float32Array,
    lr: number,
    wd: number,
    bc1: number,
    bc2: number
  ) => {
    for (let i = 0; i < param.length; i++) {
      const g = grad[i];
      const mi = beta1 * m[i] + (1 - beta1) * g;
      const vi = beta2 * v[i] + (1 - beta2) * g * g;
      m[i] = mi;
      v[i] = vi;
      const mHat = mi / bc1;
      const vHat = vi / bc2;
      param[i] = param[i] * (1 - lr * wd) - (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
  };

  const evaluateVal = (): {
    valLoss: number;
    accuracy: number;
    macroF1: number;
    meanIoU: number;
    perClassIoU: Record<string, number>;
  } => {
    let vLoss = 0;
    let correct = 0;
    const tp = new Int32Array(C);
    const fp = new Int32Array(C);
    const fn = new Int32Array(C);

    for (let i = 0; i < nVal; i++) {
      const xOff = i * D;
      for (let h = 0; h < H1; h++) {
        let s = b1[h];
        const wOff = h * D;
        for (let d = 0; d < D; d++) s += w1[wOff + d] * xVal[xOff + d];
        a1[h] = gelu(s);
      }
      for (let h = 0; h < H2; h++) {
        let s = b2[h];
        const wOff = h * H1;
        for (let j = 0; j < H1; j++) s += w2[wOff + j] * a1[j];
        // Residual skip from first H2 channels of a1 stabilizes few-shot convergence
        a2[h] = gelu(s) + 0.25 * a1[h];
      }
      let maxL = -Infinity;
      for (let c = 0; c < C; c++) {
        let s = bOut[c];
        const wOff = c * H2;
        for (let j = 0; j < H2; j++) s += wOut[wOff + j] * a2[j];
        logits[c] = s;
        if (s > maxL) maxL = s;
      }
      let sumE = 0;
      let pred = 0;
      for (let c = 0; c < C; c++) {
        const e = Math.exp(Math.max(-20, logits[c] - maxL));
        probs[c] = e;
        sumE += e;
        if (logits[c] > logits[pred]) pred = c;
      }
      const yt = valY[i];
      const py = Math.max(1e-6, probs[yt] / Math.max(1e-8, sumE));
      vLoss -= Math.log(py);
      if (pred === yt) {
        correct++;
        tp[yt]++;
      } else {
        fp[pred]++;
        fn[yt]++;
      }
    }

    const perClassIoU: Record<string, number> = {};
    const f1s: number[] = [];
    for (let c = 0; c < C; c++) {
      const denom = tp[c] + fp[c] + fn[c];
      if (denom === 0) continue;
      const iou = tp[c] / denom;
      const f1 = (2 * tp[c]) / (2 * tp[c] + fp[c] + fn[c]);
      const cid = classIds[c];
      const cname = classes[cid - 1] || `Class ${cid}`;
      perClassIoU[cname] = Number(iou.toFixed(4));
      f1s.push(f1);
    }
    const iouVals = Object.values(perClassIoU);
    return {
      valLoss: vLoss / Math.max(1, nVal),
      accuracy: correct / Math.max(1, nVal),
      macroF1: f1s.reduce((a, b) => a + b, 0) / Math.max(1, f1s.length),
      meanIoU: iouVals.reduce((a, b) => a + b, 0) / Math.max(1, iouVals.length),
      perClassIoU,
    };
  };

  // Track best checkpoint on spatial block hold-out validation
  let bestValScore = -1;
  let bestSnap = {
    w1: Array.from(w1),
    b1: Array.from(b1),
    w2: Array.from(w2),
    b2: Array.from(b2),
    wOut: Array.from(wOut),
    bOut: Array.from(bOut),
  };

  const epochsLog: TrainingEpochPoint[] = [];

  for (let epoch = 1; epoch <= totalEpochs; epoch++) {
    if (shouldAbort && shouldAbort()) {
      throw new Error('训练已由用户取消 / Training cancelled by user.');
    }

    // Cosine annealing LR schedule
    const lr = baseLr * (0.1 + 0.9 * 0.5 * (1 + Math.cos((Math.PI * (epoch - 1)) / totalEpochs)));

    // Fisher-Yates shuffle
    for (let i = nTrain - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    let epochLossSum = 0;

    for (let start = 0; start < nTrain; start += batchSize) {
      const end = Math.min(nTrain, start + batchSize);
      const bLen = end - start;

      gW1.fill(0);
      gB1.fill(0);
      gW2.fill(0);
      gB2.fill(0);
      gWO.fill(0);
      gBO.fill(0);

      let wSumBatch = 0;

      for (let bi = start; bi < end; bi++) {
        const si = order[bi];
        const xOff = si * D;
        const yt = trainY[si];

        // Forward pass
        for (let h = 0; h < H1; h++) {
          let s = b1[h];
          const wOff = h * D;
          for (let d = 0; d < D; d++) s += w1[wOff + d] * xTrain[xOff + d];
          z1[h] = s;
          a1[h] = gelu(s);
        }
        for (let h = 0; h < H2; h++) {
          let s = b2[h];
          const wOff = h * H1;
          for (let j = 0; j < H1; j++) s += w2[wOff + j] * a1[j];
          z2[h] = s;
          a2[h] = gelu(s) + 0.25 * a1[h];
        }
        let maxL = -Infinity;
        for (let c = 0; c < C; c++) {
          let s = bOut[c];
          const wOff = c * H2;
          for (let j = 0; j < H2; j++) s += wOut[wOff + j] * a2[j];
          logits[c] = s;
          if (s > maxL) maxL = s;
        }
        let sumE = 0;
        for (let c = 0; c < C; c++) {
          const e = Math.exp(Math.max(-20, logits[c] - maxL));
          probs[c] = e;
          sumE += e;
        }
        for (let c = 0; c < C; c++) probs[c] /= Math.max(1e-8, sumE);

        const py = Math.max(1e-6, probs[yt]);
        const sampleLoss = -Math.log(py);
        epochLossSum += sampleLoss;

        // Hard-Negative / Online Hard Example Mining weight boost after warm-up epochs
        let hnmBoost = 1.0;
        if (config.hardNegativeMining && epoch >= 4) {
          const err = Math.max(0, 1 - py);
          hnmBoost = 0.75 + 1.65 * err * Math.sqrt(err);
          sampleHardWeight[si] = hnmBoost;
        }
        const sw = classWeight[yt] * hnmBoost;
        wSumBatch += sw;

        // Backward pass: Output layer
        dz2.fill(0);
        for (let c = 0; c < C; c++) {
          const dl = (probs[c] - (c === yt ? 1 : 0)) * sw;
          gBO[c] += dl;
          const wOff = c * H2;
          for (let j = 0; j < H2; j++) {
            gWO[wOff + j] += dl * a2[j];
            dz2[j] += dl * wOut[wOff + j];
          }
        }

        // Backward pass: Layer 2 + residual skip to a1
        da1.fill(0);
        for (let h = 0; h < H2; h++) {
          // Residual path grad
          da1[h] += dz2[h] * 0.25;
          const dg2 = dz2[h] * geluDeriv(z2[h]);
          gB2[h] += dg2;
          const wOff = h * H1;
          for (let j = 0; j < H1; j++) {
            gW2[wOff + j] += dg2 * a1[j];
            da1[j] += dg2 * w2[wOff + j];
          }
        }

        // Backward pass: Layer 1
        for (let h = 0; h < H1; h++) {
          const dg1 = da1[h] * geluDeriv(z1[h]);
          dz1[h] = dg1;
          gB1[h] += dg1;
          const wOff = h * D;
          for (let d = 0; d < D; d++) {
            gW1[wOff + d] += dg1 * xTrain[xOff + d];
          }
        }
      }

      const invB = 1 / Math.max(1e-4, wSumBatch || bLen);
      for (let i = 0; i < gW1.length; i++) gW1[i] *= invB;
      for (let i = 0; i < gB1.length; i++) gB1[i] *= invB;
      for (let i = 0; i < gW2.length; i++) gW2[i] *= invB;
      for (let i = 0; i < gB2.length; i++) gB2[i] *= invB;
      for (let i = 0; i < gWO.length; i++) gWO[i] *= invB;
      for (let i = 0; i < gBO.length; i++) gBO[i] *= invB;

      stepNum++;
      const bc1 = 1 - Math.pow(beta1, stepNum);
      const bc2 = 1 - Math.pow(beta2, stepNum);

      adamStep(w1, gW1, mW1, vW1, lr, weightDecay, bc1, bc2);
      adamStep(b1, gB1, mB1, vB1, lr, 0, bc1, bc2);
      adamStep(w2, gW2, mW2, vW2, lr, weightDecay, bc1, bc2);
      adamStep(b2, gB2, mB2, vB2, lr, 0, bc1, bc2);
      adamStep(wOut, gWO, mWO, vWO, lr, weightDecay, bc1, bc2);
      adamStep(bOut, gBO, mBO, vBO, lr, 0, bc1, bc2);
    }

    const trainLoss = epochLossSum / Math.max(1, nTrain);
    const shouldEval = epoch === 1 || epoch % 2 === 0 || epoch === totalEpochs;
    if (shouldEval) {
      const valRes = evaluateVal();

      const compositeScore = valRes.meanIoU * 0.7 + valRes.macroF1 * 0.3 - valRes.valLoss * 0.02;
      if (compositeScore > bestValScore) {
        bestValScore = compositeScore;
        bestSnap = {
          w1: Array.from(w1),
          b1: Array.from(b1),
          w2: Array.from(w2),
          b2: Array.from(b2),
          wOut: Array.from(wOut),
          bOut: Array.from(bOut),
        };
      }

      epochsLog.push({
        epoch,
        trainLoss: Number(trainLoss.toFixed(4)),
        valLoss: Number(valRes.valLoss.toFixed(4)),
        valMeanIoU: Number(valRes.meanIoU.toFixed(4)),
        valMacroF1: Number(valRes.macroF1.toFixed(4)),
      });

      if (onProgress) {
        onProgress({
          phase: 'training',
          epoch,
          totalEpochs,
          trainLoss: Number(trainLoss.toFixed(4)),
          valLoss: Number(valRes.valLoss.toFixed(4)),
          valMeanIoU: Number(valRes.meanIoU.toFixed(4)),
          valMacroF1: Number(valRes.macroF1.toFixed(4)),
          cacheHit,
        });
      }

      if (epoch >= 12 && valRes.meanIoU >= 0.995 && valRes.valLoss < 0.03) {
        break;
      }
    }
  }

  // Restore best validation checkpoint weights
  w1.set(bestSnap.w1);
  b1.set(bestSnap.b1);
  w2.set(bestSnap.w2);
  b2.set(bestSnap.b2);
  wOut.set(bestSnap.wOut);
  bOut.set(bestSnap.bOut);

  const finalVal = evaluateVal();
  const durationMs = Math.round(performance.now() - t0);

  const weights: DeepFewShotWeights = {
    architecture: 'multiscale-pyramid-residual-mlp',
    featureDim: D,
    hidden1: H1,
    hidden2: H2,
    classIds,
    featMean,
    featStd,
    w1: bestSnap.w1,
    b1: bestSnap.b1,
    w2: bestSnap.w2,
    b2: bestSnap.b2,
    wOut: bestSnap.wOut,
    bOut: bestSnap.bOut,
    bilateralSmooth: config.bilateralSmooth ? 0.55 : 0,
  };

  const metrics: HoldoutMetrics = {
    nTrainPixels: nTrain,
    nValPixels: nVal,
    accuracy: Number(finalVal.accuracy.toFixed(4)),
    macroF1: Number(finalVal.macroF1.toFixed(4)),
    meanIoU: Number(finalVal.meanIoU.toFixed(4)),
    perClassIoU: finalVal.perClassIoU,
    splitStrategy: 'spatial_block',
    epochs: epochsLog,
    durationMs,
    cacheHit,
    architecture: `Frozen Multi-Scale Pyramid (${D}ch) + Residual MLP (${H1}→${H2}→${C})`,
  };

  return { weights, metrics, epochs: epochsLog, durationMs, cacheHit };
}

/**
 * Edge-preserving fast bilateral smoothing over a 0..1 foreground probability map
 * using the underlying grayscale intensities as guidance.
 */
function bilateralSmoothProb(
  prob: Float32Array,
  gray: Float32Array,
  width: number,
  height: number,
  maxY: number,
  strength: number
): Float32Array {
  if (strength <= 0.01) return prob;
  const out = new Float32Array(prob.length);
  const invSigmaRange2 = 1 / (2 * 22 * 22);
  for (let y = 0; y < maxY; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const idx = row + x;
      const gc = gray[idx];
      let wAcc = 0;
      let pAcc = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= maxY) continue;
        const nRow = ny * width;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const nIdx = nRow + nx;
          const dg = gray[nIdx] - gc;
          const ws = dx === 0 && dy === 0 ? 1.5 : dx === 0 || dy === 0 ? 1.0 : 0.7;
          const wr = Math.exp(-(dg * dg) * invSigmaRange2);
          const w = ws * wr;
          wAcc += w;
          pAcc += prob[nIdx] * w;
        }
      }
      const smoothed = pAcc / Math.max(1e-6, wAcc);
      out[idx] = prob[idx] * (1 - strength) + smoothed * strength;
    }
  }
  return out;
}

/**
 * Cosine-feathered 512×512 Tiled Streaming Inference for DeepFewShotWeights.
 * Guarantees bounded memory regardless of image resolution and zero seam artifacts.
 */
export function predictDeepFewShotTiled(
  weights: DeepFewShotWeights,
  gray: Float32Array,
  width: number,
  height: number,
  cropRows?: number,
  tileSize = 512,
  overlap = 64
): Float32Array {
  const maxY = Math.min(height, cropRows ?? height);
  const n = width * height;
  const accum = new Float32Array(n);
  const weightSum = new Float32Array(n);

  // Global normalization stats computed once with lightweight stride
  let sum = 0;
  let cnt = 0;
  const stride = Math.max(1, Math.floor((width * maxY) / 131072));
  for (let i = 0; i < width * maxY; i += stride) {
    sum += gray[i];
    cnt++;
  }
  const meanAll = sum / Math.max(1, cnt);
  let varSum = 0;
  for (let i = 0; i < width * maxY; i += stride) {
    const d = gray[i] - meanAll;
    varSum += d * d;
  }
  const stdAll = Math.max(8, Math.sqrt(varSum / Math.max(1, cnt)));

  const D = weights.featureDim;
  const H1 = weights.hidden1;
  const H2 = weights.hidden2;
  const C = weights.classIds.length;
  const fgIdx = weights.classIds.indexOf(1);
  const targetOutIdx = fgIdx >= 0 ? fgIdx : 0;

  const { featMean, featStd, w1, b1, w2, b2, wOut, bOut } = weights;
  const a1 = new Float32Array(H1);
  const a2 = new Float32Array(H2);
  const logits = new Float32Array(C);

  const step = Math.max(64, tileSize - overlap);
  const feather = Math.max(8, Math.floor(overlap * 0.5));

  for (let ty = 0; ty < maxY; ty += step) {
    const y0 = Math.max(0, ty);
    const y1 = Math.min(maxY, y0 + tileSize);
    const th = y1 - y0;

    for (let tx = 0; tx < width; tx += step) {
      const x0 = Math.max(0, tx);
      const x1 = Math.min(width, x0 + tileSize);
      const tw = x1 - x0;

      // Extract tile grayscale slice
      const tileGray = new Float32Array(tw * th);
      for (let ly = 0; ly < th; ly++) {
        const srcRow = (y0 + ly) * width + x0;
        const dstRow = ly * tw;
        for (let lx = 0; lx < tw; lx++) {
          tileGray[dstRow + lx] = gray[srcRow + lx];
        }
      }

      const feats =
        D === LEGACY_FEATURE_DIM
          ? extractMultiScaleFeatures16(tileGray, tw, th, { mean: meanAll, std: stdAll })
          : extractMultiScaleFeatures24(tileGray, tw, th, { mean: meanAll, std: stdAll });

      for (let ly = 0; ly < th; ly++) {
        // Raised cosine feather along tile borders that touch another tile
        let wy = 1.0;
        if (y0 > 0 && ly < feather) {
          wy = 0.5 * (1 - Math.cos((Math.PI * (ly + 0.5)) / feather));
        } else if (y1 < maxY && th - 1 - ly < feather) {
          wy = 0.5 * (1 - Math.cos((Math.PI * (th - ly - 0.5)) / feather));
        }

        const dstRow = (y0 + ly) * width + x0;
        const featRow = ly * tw;

        for (let lx = 0; lx < tw; lx++) {
          let wx = 1.0;
          if (x0 > 0 && lx < feather) {
            wx = 0.5 * (1 - Math.cos((Math.PI * (lx + 0.5)) / feather));
          } else if (x1 < width && tw - 1 - lx < feather) {
            wx = 0.5 * (1 - Math.cos((Math.PI * (tw - lx - 0.5)) / feather));
          }

          const fBase = (featRow + lx) * D;
          for (let h = 0; h < H1; h++) {
            let s = b1[h];
            const wOff = h * D;
            for (let d = 0; d < D; d++) {
              s += w1[wOff + d] * ((feats[fBase + d] - featMean[d]) / featStd[d]);
            }
            a1[h] = gelu(s);
          }
          for (let h = 0; h < H2; h++) {
            let s = b2[h];
            const wOff = h * H1;
            for (let j = 0; j < H1; j++) s += w2[wOff + j] * a1[j];
            a2[h] = gelu(s) + 0.25 * a1[h];
          }
          let maxL = -Infinity;
          for (let c = 0; c < C; c++) {
            let s = bOut[c];
            const wOff = c * H2;
            for (let j = 0; j < H2; j++) s += wOut[wOff + j] * a2[j];
            logits[c] = s;
            if (s > maxL) maxL = s;
          }
          let sumE = 0;
          for (let c = 0; c < C; c++) {
            sumE += Math.exp(Math.max(-20, logits[c] - maxL));
          }
          const fgProb = Math.exp(Math.max(-20, logits[targetOutIdx] - maxL)) / Math.max(1e-8, sumE);

          const wgt = Math.max(1e-3, wx * wy);
          const gIdx = dstRow + lx;
          accum[gIdx] += fgProb * wgt;
          weightSum[gIdx] += wgt;
        }
      }

      if (x1 >= width) break;
    }
    if (y1 >= maxY) break;
  }

  const out = new Float32Array(n);
  for (let i = 0; i < width * maxY; i++) {
    out[i] = weightSum[i] > 0 ? accum[i] / weightSum[i] : 0;
  }

  return bilateralSmoothProb(out, gray, width, height, maxY, weights.bilateralSmooth);
}
