import type {
  DetectionObject,
  LowConfidenceSegment,
  OverlayShape,
} from '../types/vexel';

// ---------------------------------------------------------------------------
// Curved / Low-Contrast / Uneven Multilayer Enhancement & Spectral Metrology
//   1. Local Contrast Normalization (LCN / CLAHE-style integral-image local z-norm)
//   2. Structure-Tensor Oriented Anisotropic Coherence Enhancement (curved streamline)
//   3. Fast Radix-2 2D FFT Dominant Spatial Period Verification
//   4. Arbitrary-Direction Narrow-Band Line-Profile Layer Counter
//   5. Low-Contrast / Low-Confidence Region & Segment Scanner
// ---------------------------------------------------------------------------

export interface OrientationField {
  tx: Float32Array;
  ty: Float32Array;
  nx: Float32Array;
  ny: Float32Array;
  coherence: Float32Array;
}

export interface FftPeriodResult {
  valid: boolean;
  periodPx: number;
  singleLayerPx: number;
  dominantAngleDeg: number;
  snr: number;
}

export interface LineProfileLayerAnalysis {
  positions: number[];
  intensities: number[];
  edges: number[];
  widthPhys: number | null;
  layerCount: number;
  layerThicknesses: number[];
  meanLayerThickness: number | null;
  totalStackThickness: number | null;
  fftPeriodPhys: number | null;
}

function bilinearSampleClamped(
  gray: Float32Array,
  w: number,
  h: number,
  x: number,
  y: number
): number {
  const cx = x < 0 ? 0 : x > w - 1 ? w - 1 : x;
  const cy = y < 0 ? 0 : y > h - 1 ? h - 1 : y;
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = x0 + 1 < w ? x0 + 1 : x0;
  const y1 = y0 + 1 < h ? y0 + 1 : y0;
  const fx = cx - x0;
  const fy = cy - y0;
  const row0 = y0 * w;
  const row1 = y1 * w;
  return (
    (1 - fx) * (1 - fy) * gray[row0 + x0] +
    fx * (1 - fy) * gray[row0 + x1] +
    (1 - fx) * fy * gray[row1 + x0] +
    fx * fy * gray[row1 + x1]
  );
}

function separableGaussian2D(
  src: Float32Array,
  w: number,
  cropRows: number,
  sigma: number
): Float32Array {
  const n = w * cropRows;
  if (sigma <= 0.25 || w < 3 || cropRows < 3) return Float32Array.from(src.subarray(0, n));

  const rad = Math.max(1, Math.min(18, Math.ceil(sigma * 2.6)));
  const kernel = new Float32Array(rad * 2 + 1);
  let kSum = 0;
  const denom = 2 * sigma * sigma;
  for (let k = -rad; k <= rad; k++) {
    const val = Math.exp(-(k * k) / denom);
    kernel[k + rad] = val;
    kSum += val;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;

  const tmp = new Float32Array(n);
  for (let y = 0; y < cropRows; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -rad; k <= rad; k++) {
        let sx = x + k;
        if (sx < 0) sx = 0;
        else if (sx >= w) sx = w - 1;
        acc += src[row + sx] * kernel[k + rad];
      }
      tmp[row + x] = acc;
    }
  }

  const out = new Float32Array(n);
  for (let y = 0; y < cropRows; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -rad; k <= rad; k++) {
        let sy = y + k;
        if (sy < 0) sy = 0;
        else if (sy >= cropRows) sy = cropRows - 1;
        acc += tmp[sy * w + x] * kernel[k + rad];
      }
      out[row + x] = acc;
    }
  }
  return out;
}

/**
 * 1. Local Contrast Normalization (LCN / CLAHE-like adaptive equalization)
 * Uses O(W*H) 2D summed-area tables for local mean and variance so dark and bright
 * unevenly-illuminated zones achieve uniform lamellar contrast without noise explosion.
 */
export function applyLocalContrastNorm(
  src: Float32Array,
  w: number,
  h: number,
  cropRows: number,
  options: { windowPx?: number; clipLimit?: number; blend?: number } = {}
): Float32Array {
  const out = new Float32Array(src);
  const rows = Math.max(1, Math.min(h, cropRows));
  const windowPx = Math.max(8, Math.min(128, Math.round(options.windowPx ?? 24)));
  const clipLimit = Math.max(1.1, Math.min(6.0, Number(options.clipLimit ?? 2.5)));
  const blend = Math.max(0.05, Math.min(1.0, Number(options.blend ?? 0.88)));

  const rad = Math.max(4, Math.floor(windowPx * 0.5));
  const stride = w + 1;
  const sat1 = new Float64Array(stride * (rows + 1));
  const sat2 = new Float64Array(stride * (rows + 1));

  // Build summed-area tables for I and I^2
  let globalSum = 0;
  let globalSum2 = 0;
  for (let y = 0; y < rows; y++) {
    let rowS1 = 0;
    let rowS2 = 0;
    const srcRow = y * w;
    const dstRow = (y + 1) * stride;
    const prevRow = y * stride;
    for (let x = 0; x < w; x++) {
      const v = src[srcRow + x];
      rowS1 += v;
      rowS2 += v * v;
      sat1[dstRow + x + 1] = sat1[prevRow + x + 1] + rowS1;
      sat2[dstRow + x + 1] = sat2[prevRow + x + 1] + rowS2;
    }
    globalSum += rowS1;
    globalSum2 += rowS2;
  }

  const totalN = Math.max(1, w * rows);
  const globalMean = globalSum / totalN;
  const globalStd = Math.sqrt(Math.max(1, globalSum2 / totalN - globalMean * globalMean));
  const sigmaFloor = Math.max(6.5, globalStd * 0.24);

  for (let y = 0; y < rows; y++) {
    const y0 = Math.max(0, y - rad);
    const y1 = Math.min(rows - 1, y + rad);
    const r0 = y0 * stride;
    const r1 = (y1 + 1) * stride;
    const srcRow = y * w;
    const boxH = y1 - y0 + 1;

    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - rad);
      const x1 = Math.min(w - 1, x + rad);
      const area = (x1 - x0 + 1) * boxH;

      const s1 = sat1[r1 + x1 + 1] - sat1[r0 + x1 + 1] - sat1[r1 + x0] + sat1[r0 + x0];
      const s2 = sat2[r1 + x1 + 1] - sat2[r0 + x1 + 1] - sat2[r1 + x0] + sat2[r0 + x0];

      const mean = s1 / area;
      const variance = Math.max(0, s2 / area - mean * mean);
      const std = Math.sqrt(variance);
      const denom = std > sigmaFloor ? std : sigmaFloor;

      const v = src[srcRow + x];
      let z = (v - mean) / denom;
      if (z > clipLimit) z = clipLimit;
      else if (z < -clipLimit) z = -clipLimit;

      const normVal = 127.5 + (z / clipLimit) * 116;
      const blended = v * (1 - blend) + normVal * blend;
      out[srcRow + x] = blended < 0 ? 0 : blended > 255 ? 255 : blended;
    }
  }

  return out;
}

/**
 * 2a. Smoothed 2x2 Structure Tensor Orientation & Coherence Field
 * Uses rotation-invariant 3x3 Scharr derivatives followed by Gaussian tensor smoothing.
 * Returns per-pixel local layer tangent (tx, ty), normal (nx, ny), and coherence C in [0, 1].
 */
export function computeOrientationField(
  gray: Float32Array,
  w: number,
  cropRows: number,
  tensorSigma = 3.0
): OrientationField {
  const rows = Math.max(1, cropRows);
  const n = w * rows;
  const rawJxx = new Float32Array(n);
  const rawJyy = new Float32Array(n);
  const rawJxy = new Float32Array(n);

  // Rotation-invariant Scharr gradient operator
  for (let y = 1; y < rows - 1; y++) {
    const rPrev = (y - 1) * w;
    const rCur = y * w;
    const rNext = (y + 1) * w;
    for (let x = 1; x < w - 1; x++) {
      const gx =
        (3 * gray[rPrev + x + 1] + 10 * gray[rCur + x + 1] + 3 * gray[rNext + x + 1] -
          (3 * gray[rPrev + x - 1] + 10 * gray[rCur + x - 1] + 3 * gray[rNext + x - 1])) *
        0.03125;
      const gy =
        (3 * gray[rNext + x - 1] + 10 * gray[rNext + x] + 3 * gray[rNext + x + 1] -
          (3 * gray[rPrev + x - 1] + 10 * gray[rPrev + x] + 3 * gray[rPrev + x + 1])) *
        0.03125;
      const idx = rCur + x;
      rawJxx[idx] = gx * gx;
      rawJyy[idx] = gy * gy;
      rawJxy[idx] = gx * gy;
    }
  }

  const sJxx = separableGaussian2D(rawJxx, w, rows, tensorSigma);
  const sJyy = separableGaussian2D(rawJyy, w, rows, tensorSigma);
  const sJxy = separableGaussian2D(rawJxy, w, rows, tensorSigma);

  const tx = new Float32Array(n);
  const ty = new Float32Array(n);
  const nx = new Float32Array(n);
  const ny = new Float32Array(n);
  const coherence = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const a = sJxx[i];
    const b = sJyy[i];
    const c = sJxy[i];
    const trace = a + b;
    const diff = a - b;
    const disc = Math.hypot(diff, 2 * c);
    const l1 = 0.5 * (trace + disc);
    const l2 = Math.max(0, 0.5 * (trace - disc));

    // Normal eigenvector angle (points perpendicular to layers, along strongest gradient)
    const phi = 0.5 * Math.atan2(2 * c, diff);
    const cPhi = Math.cos(phi);
    const sPhi = Math.sin(phi);
    nx[i] = cPhi;
    ny[i] = sPhi;
    // Tangent vector is perpendicular to gradient normal (runs along curved layer)
    tx[i] = -sPhi;
    ty[i] = cPhi;

    const coh = trace > 1e-4 ? (l1 - l2) / (l1 + l2 + 1e-4) : 0;
    coherence[i] = coh * coh;
  }

  return { tx, ty, nx, ny, coherence };
}

/**
 * 2b. Coherence-Enhancing Curved Anisotropic Layer Filter
 * Performs curved streamline-guided Gaussian smoothing strictly along the local
 * layer tangent field (suppressing granular speckle & reconnecting broken arcs)
 * while keeping normal boundaries sharp (with optional mild normal sharpening).
 */
export function applyCoherenceEnhance(
  src: Float32Array,
  w: number,
  h: number,
  cropRows: number,
  options: {
    alongSigma?: number;
    tensorSigma?: number;
    iterations?: number;
    sharpen?: number;
  } = {}
): Float32Array {
  const rows = Math.max(1, Math.min(h, cropRows));
  const alongSigma = Math.max(0.8, Math.min(10.0, Number(options.alongSigma ?? 3.2)));
  const tensorSigma = Math.max(1.2, Math.min(9.0, Number(options.tensorSigma ?? 3.0)));
  const iterations = Math.max(1, Math.min(5, Math.round(options.iterations ?? 2)));
  const sharpen = Math.max(0, Math.min(1.5, Number(options.sharpen ?? 0.45)));

  const rad = Math.max(2, Math.min(14, Math.ceil(alongSigma * 2.1)));
  const gWeights = new Float32Array(rad + 1);
  const inv2s2 = 1 / (2 * alongSigma * alongSigma);
  for (let s = 0; s <= rad; s++) {
    gWeights[s] = Math.exp(-(s * s) * inv2s2);
  }

  let cur = new Float32Array(src);
  const next = new Float32Array(src);

  for (let iter = 0; iter < iterations; iter++) {
    const field = computeOrientationField(cur, w, rows, tensorSigma);
    const { tx, ty, nx, ny, coherence } = field;

    for (let y = 0; y < rows; y++) {
      const rowOff = y * w;
      for (let x = 0; x < w; x++) {
        const idx = rowOff + x;
        const baseVal = cur[idx];
        const coh = coherence[idx];
        const t0x = tx[idx];
        const t0y = ty[idx];

        let acc = baseVal * gWeights[0];
        let wSum = gWeights[0];

        // March in +tangent and -tangent directions following local curved tangent field
        for (const dir of [1, -1]) {
          let px = x;
          let py = y;
          let vx = t0x * dir;
          let vy = t0y * dir;

          for (let s = 1; s <= rad; s++) {
            px += vx;
            py += vy;
            if (px < 0 || py < 0 || px > w - 1 || py > rows - 1) break;

            const ix = Math.round(px);
            const iy = Math.round(py);
            const pIdx = iy * w + ix;
            let ltx = tx[pIdx];
            let lty = ty[pIdx];
            const dot = ltx * vx + lty * vy;
            if (dot < 0) {
              ltx = -ltx;
              lty = -lty;
            }
            // Stop smoothly if tangent turns sharply (e.g. hitting a vertical crack/domain wall)
            const align = Math.abs(dot);
            if (align < 0.45) break;

            // Gentle streamline curvature tracking
            const blendDirX = 0.65 * ltx + 0.35 * vx;
            const blendDirY = 0.65 * lty + 0.35 * vy;
            const dLen = Math.hypot(blendDirX, blendDirY) || 1;
            vx = blendDirX / dLen;
            vy = blendDirY / dLen;

            const sampleVal = bilinearSampleClamped(cur, w, rows, px, py);
            const wt = gWeights[s] * (0.35 + 0.65 * align);
            acc += sampleVal * wt;
            wSum += wt;
          }
        }

        const filtered = acc / wSum;
        // Weight diffusion by local coherence so isotropic areas/cracks stay preserved
        const alpha = 0.25 + 0.75 * coh;
        let outVal = baseVal * (1 - alpha) + filtered * alpha;

        // Optional perpendicular (normal-axis) boundary sharpening on final pass
        if (sharpen > 0 && iter === iterations - 1 && coh > 0.12) {
          const n0x = nx[idx];
          const n0y = ny[idx];
          const vPlus = bilinearSampleClamped(cur, w, rows, x + n0x * 1.4, y + n0y * 1.4);
          const vMinus = bilinearSampleClamped(cur, w, rows, x - n0x * 1.4, y - n0y * 1.4);
          const lapNormal = vPlus - 2 * outVal + vMinus;
          outVal -= 0.28 * sharpen * coh * lapNormal;
        }

        next[idx] = outVal < 0 ? 0 : outVal > 255 ? 255 : outVal;
      }
    }

    // Copy uncropped bottom region if present and swap buffers
    for (let i = rows * w; i < src.length; i++) next[i] = src[i];
    cur = new Float32Array(next);
  }

  return cur;
}

// ---------------------------------------------------------------------------
// 3. Fast Radix-2 1D & 2D FFT Dominant Period Estimation
// ---------------------------------------------------------------------------

function fft1D(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (-2 * Math.PI) / len;
    const wLenRe = Math.cos(ang);
    const wLenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < half; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + half] * wRe - im[i + j + half] * wIm;
        const vIm = re[i + j + half] * wIm + im[i + j + half] * wRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + half] = uRe - vRe;
        im[i + j + half] = uIm - vIm;
        const nextRe = wRe * wLenRe - wIm * wLenIm;
        wIm = wRe * wLenIm + wIm * wLenRe;
        wRe = nextRe;
      }
    }
  }
}

/**
 * Estimates dominant repeating layer/bilayer period via a Hann-windowed 128x128 2D FFT
 * centered inside the active ROI/image. Noise-immune independent cross-check for low-contrast stacks.
 */
export function compute2DFFTDominantPeriod(
  gray: Float32Array,
  w: number,
  cropRows: number,
  minLayerPx = 3
): FftPeriodResult {
  const N = 128;
  const minDim = Math.min(w, cropRows);
  if (minDim < 16) {
    return { valid: false, periodPx: 0, singleLayerPx: 0, dominantAngleDeg: 0, snr: 0 };
  }

  // Isotropic sampling step so FFT radial frequency has zero aspect-ratio distortion
  const spanPx = Math.min(minDim, Math.max(64, Math.floor(minDim * 0.88)));
  const stepPx = spanPx / N;
  const startX = (w - spanPx) * 0.5;
  const startY = (cropRows - spanPx) * 0.5;

  const patch = new Float64Array(N * N);
  let mean = 0;
  for (let y = 0; y < N; y++) {
    const py = startY + (y + 0.5) * stepPx;
    for (let x = 0; x < N; x++) {
      const px = startX + (x + 0.5) * stepPx;
      const v = bilinearSampleClamped(gray, w, cropRows, px, py);
      patch[y * N + x] = v;
      mean += v;
    }
  }
  mean /= N * N;

  // Apply 2D Hann window
  const hann = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
  }

  const re = new Float64Array(N * N);
  const im = new Float64Array(N * N);
  for (let y = 0; y < N; y++) {
    const wy = hann[y];
    const row = y * N;
    for (let x = 0; x < N; x++) {
      re[row + x] = (patch[row + x] - mean) * wy * hann[x];
    }
  }

  // 1D FFT across rows
  const rowRe = new Float64Array(N);
  const rowIm = new Float64Array(N);
  for (let y = 0; y < N; y++) {
    const off = y * N;
    for (let x = 0; x < N; x++) {
      rowRe[x] = re[off + x];
      rowIm[x] = 0;
    }
    fft1D(rowRe, rowIm);
    for (let x = 0; x < N; x++) {
      re[off + x] = rowRe[x];
      im[off + x] = rowIm[x];
    }
  }

  // 1D FFT down columns
  const colRe = new Float64Array(N);
  const colIm = new Float64Array(N);
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) {
      colRe[y] = re[y * N + x];
      colIm[y] = im[y * N + x];
    }
    fft1D(colRe, colIm);
    for (let y = 0; y < N; y++) {
      re[y * N + x] = colRe[y];
      im[y * N + x] = colIm[y];
    }
  }

  // Search physically valid frequency annulus
  // Wavelength lambda_px = spanPx / freqRadius
  const minPeriodPx = Math.max(4, minLayerPx * 1.7);
  const maxPeriodPx = Math.max(minPeriodPx * 1.5, spanPx * 0.42);
  const fMin = Math.max(2.2, spanPx / maxPeriodPx);
  const fMax = Math.min(N * 0.46, spanPx / minPeriodPx);

  const half = N >> 1;
  const bandPowers: number[] = [];
  let bestPower = 0;
  let bestU = 0;
  let bestV = 0;

  // Scan upper half-plane (v in [0..half-1], u in [-half+1..half-1], excluding negative u when v===0)
  for (let v = 0; v < half; v++) {
    for (let u = -half + 1; u < half; u++) {
      if (v === 0 && u <= 0) continue;
      const fr = Math.hypot(u, v);
      if (fr < fMin || fr > fMax) continue;
      const iu = u < 0 ? u + N : u;
      const idx = v * N + iu;
      const pwr = re[idx] * re[idx] + im[idx] * im[idx];
      bandPowers.push(pwr);
      if (pwr > bestPower) {
        bestPower = pwr;
        bestU = u;
        bestV = v;
      }
    }
  }

  if (bandPowers.length < 10 || bestPower <= 0) {
    return { valid: false, periodPx: 0, singleLayerPx: 0, dominantAngleDeg: 0, snr: 0 };
  }

  // Refine spectral peak via weighted 3x3 centroid in frequency space
  let wSum = 0;
  let uSum = 0;
  let vSum = 0;
  for (let dv = -1; dv <= 1; dv++) {
    for (let du = -1; du <= 1; du++) {
      const uu = bestU + du;
      const vv = bestV + dv;
      const iu = ((uu % N) + N) % N;
      const iv = ((vv % N) + N) % N;
      const idx = iv * N + iu;
      const pwr = re[idx] * re[idx] + im[idx] * im[idx];
      wSum += pwr;
      uSum += uu * pwr;
      vSum += vv * pwr;
    }
  }
  const refU = wSum > 0 ? uSum / wSum : bestU;
  const refV = wSum > 0 ? vSum / wSum : bestV;
  const peakFreq = Math.hypot(refU, refV);
  const periodPx = peakFreq > 1e-4 ? spanPx / peakFreq : 0;

  // Compute robust spectral peak-to-median SNR
  bandPowers.sort((a, b) => a - b);
  const medPower = Math.max(1e-9, bandPowers[bandPowers.length >> 1]);
  const snr = bestPower / medPower;

  // Layer tangent is perpendicular to frequency wavevector (refU, refV)
  const layerAngleDeg = Number(
    ((((Math.atan2(-refU, refV) * 180) / Math.PI) % 180 + 180) % 180).toFixed(1)
  );

  return {
    valid: snr >= 3.2 && periodPx >= minPeriodPx && periodPx <= maxPeriodPx,
    periodPx: Number(periodPx.toFixed(2)),
    singleLayerPx: Number((periodPx * 0.5).toFixed(2)),
    dominantAngleDeg: layerAngleDeg,
    snr: Number(snr.toFixed(1)),
  };
}

/**
 * 4. Arbitrary-Direction Line-Profile Layer Counter
 * Samples a narrow perpendicular strip (±2 px) along a user-drawn line p1 -> p2,
 * applies local detrending & sub-pixel crossing detection + 1D FFT period verification.
 */
export function analyzeLineProfileLayers(
  gray: Float32Array,
  w: number,
  h: number,
  p1: [number, number],
  p2: [number, number],
  pxSize: number
): LineProfileLayerAnalysis {
  const dx = p2[0] - p1[0];
  const dy = p2[1] - p1[1];
  const distPx = Math.hypot(dx, dy);
  const lengthPhys = distPx * pxSize;

  const samples = Math.min(320, Math.max(32, Math.round(distPx * 1.25)));
  const stepPx = distPx / Math.max(1, samples - 1);
  // Perpendicular unit vector for narrow 5-sample strip averaging (cuts speckle without arc phase cancellation)
  const perpX = distPx > 1e-5 ? -dy / distPx : 0;
  const perpY = distPx > 1e-5 ? dx / distPx : 0;

  const raw = new Float64Array(samples);
  const positions: number[] = [];
  for (let i = 0; i < samples; i++) {
    const frac = i / (samples - 1);
    const cx = p1[0] + dx * frac;
    const cy = p1[1] + dy * frac;
    let sum = 0;
    for (let u = -2; u <= 2; u++) {
      sum += bilinearSampleClamped(gray, w, h, cx + perpX * u, cy + perpY * u);
    }
    raw[i] = sum / 5;
    positions.push(Number((frac * lengthPhys).toFixed(2)));
  }

  // Light 1D Gaussian smooth (sigma = 1.0 sample)
  const smooth = new Float64Array(samples);
  const kRad = 3;
  const kW = [0.006, 0.061, 0.242, 0.382, 0.242, 0.061, 0.006];
  for (let i = 0; i < samples; i++) {
    let acc = 0;
    for (let k = -kRad; k <= kRad; k++) {
      const idx = Math.max(0, Math.min(samples - 1, i + k));
      acc += raw[idx] * kW[k + kRad];
    }
    smooth[i] = acc;
  }

  const intensities = Array.from(smooth, (v) => Number(v.toFixed(1)));

  // Adaptive local detrending to handle strong uneven illumination along the drawn profile
  const winHalf = Math.max(6, Math.min(Math.floor(samples * 0.22), Math.round(12 / Math.max(0.5, stepPx))));
  const prefix = new Float64Array(samples + 1);
  for (let i = 0; i < samples; i++) prefix[i + 1] = prefix[i] + smooth[i];
  const detrended = new Float64Array(samples);
  for (let i = 0; i < samples; i++) {
    const a = Math.max(0, i - winHalf);
    const b = Math.min(samples - 1, i + winHalf);
    detrended[i] = smooth[i] - (prefix[b + 1] - prefix[a]) / (b - a + 1);
  }

  // Local contrast envelope
  const env = new Float64Array(samples);
  const envRad = Math.max(4, Math.min(20, Math.round(winHalf * 0.85)));
  let maxEnv = 0;
  for (let i = 0; i < samples; i++) {
    let amp = 0;
    const a = Math.max(0, i - envRad);
    const b = Math.min(samples - 1, i + envRad);
    for (let j = a; j <= b; j++) {
      const av = Math.abs(detrended[j]);
      if (av > amp) amp = av;
    }
    env[i] = amp;
    if (amp > maxEnv) maxEnv = amp;
  }

  const ampThresh = Math.max(2.2, maxEnv * 0.22);
  const minLayerSamples = Math.max(2.0, 2.5 / Math.max(0.4, stepPx));
  const rawCrossings: number[] = [];
  for (let i = 1; i < samples; i++) {
    if (env[i] < ampThresh) continue;
    const d0 = detrended[i - 1];
    const d1 = detrended[i];
    if ((d0 <= 0 && d1 > 0) || (d0 >= 0 && d1 < 0)) {
      const frac = Math.abs(d1 - d0) > 1e-9 ? -d0 / (d1 - d0) : 0.5;
      rawCrossings.push(i - 1 + Math.max(0, Math.min(1, frac)));
    }
  }

  // Filter micro-jitter crossings
  const cleanIdx: number[] = [];
  for (const pos of rawCrossings) {
    if (cleanIdx.length === 0) {
      cleanIdx.push(pos);
    } else {
      const prev = cleanIdx[cleanIdx.length - 1];
      if (pos - prev >= minLayerSamples) {
        const i0 = Math.max(0, Math.floor(prev));
        const i1 = Math.min(samples - 1, Math.ceil(pos));
        let segPeak = 0;
        for (let j = i0; j <= i1; j++) {
          const av = Math.abs(detrended[j]);
          if (av > segPeak) segPeak = av;
        }
        if (segPeak >= ampThresh * 0.65) {
          cleanIdx.push(pos);
        } else {
          cleanIdx[cleanIdx.length - 1] = pos;
        }
      }
    }
  }

  const edges = cleanIdx.map((idx) => Number(((idx / (samples - 1)) * lengthPhys).toFixed(2)));
  const layerThicknesses: number[] = [];
  for (let i = 0; i + 1 < edges.length; i++) {
    const dt = edges[i + 1] - edges[i];
    if (dt > 0) layerThicknesses.push(Number(dt.toFixed(2)));
  }

  const layerCount = layerThicknesses.length;
  const totalStackThickness =
    layerCount > 0 ? Number(layerThicknesses.reduce((a, b) => a + b, 0).toFixed(2)) : null;
  const meanLayerThickness =
    layerCount > 0 && totalStackThickness !== null
      ? Number((totalStackThickness / layerCount).toFixed(2))
      : null;

  // 1D FFT dominant period along the drawn line
  let fftPeriodPhys: number | null = null;
  const fftN = 256;
  const fRe = new Float64Array(fftN);
  const fIm = new Float64Array(fftN);
  for (let i = 0; i < samples && i < fftN; i++) {
    const win = 0.5 * (1 - Math.cos((2 * Math.PI * i) / Math.max(1, samples - 1)));
    fRe[i] = detrended[i] * win;
  }
  fft1D(fRe, fIm);
  let bestP = 0;
  let bestK = 0;
  const kMin = Math.max(2, Math.floor((fftN * stepPx) / Math.max(6, distPx * 0.45)));
  const kMax = Math.min(fftN >> 1, Math.floor((fftN * stepPx) / 4));
  const pwrList: number[] = [];
  for (let k = kMin; k <= kMax; k++) {
    const p = fRe[k] * fRe[k] + fIm[k] * fIm[k];
    pwrList.push(p);
    if (p > bestP) {
      bestP = p;
      bestK = k;
    }
  }
  if (bestK > 0 && pwrList.length >= 4) {
    pwrList.sort((a, b) => a - b);
    const medP = Math.max(1e-9, pwrList[pwrList.length >> 1]);
    if (bestP / medP >= 3.0) {
      const periodSamples = fftN / bestK;
      fftPeriodPhys = Number((periodSamples * stepPx * pxSize).toFixed(2));
    }
  }

  return {
    positions,
    intensities,
    edges,
    widthPhys: meanLayerThickness ?? (edges.length >= 2 ? Number((edges[1] - edges[0]).toFixed(2)) : null),
    layerCount,
    layerThicknesses,
    meanLayerThickness,
    totalStackThickness,
    fftPeriodPhys,
  };
}

// ---------------------------------------------------------------------------
// 5. Low-Contrast / Low-Confidence Segment & Blind-Zone Scanner
// ---------------------------------------------------------------------------

function measureBoxContrast(
  gray: Float32Array,
  w: number,
  cropRows: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number
): { contrast: number; std: number; meanGrad: number } {
  const cx0 = Math.max(0, Math.min(w - 1, Math.floor(x0)));
  const cy0 = Math.max(0, Math.min(cropRows - 1, Math.floor(y0)));
  const cx1 = Math.max(cx0 + 1, Math.min(w, Math.ceil(x1)));
  const cy1 = Math.max(cy0 + 1, Math.min(cropRows, Math.ceil(y1)));

  const bw = cx1 - cx0;
  const bh = cy1 - cy0;
  const step = Math.max(1, Math.floor(Math.sqrt((bw * bh) / 160)));
  const vals: number[] = [];
  let sum = 0;
  let gradSum = 0;
  let gradCnt = 0;

  for (let y = cy0; y < cy1; y += step) {
    const row = y * w;
    for (let x = cx0; x < cx1; x += step) {
      const v = gray[row + x];
      vals.push(v);
      sum += v;
      if (x > 0 && x < w - 1 && y > 0 && y < cropRows - 1) {
        const gx = 0.5 * (gray[row + x + 1] - gray[row + x - 1]);
        const gy = 0.5 * (gray[row + w + x] - gray[row - w + x]);
        gradSum += Math.hypot(gx, gy);
        gradCnt++;
      }
    }
  }

  if (vals.length === 0) return { contrast: 0, std: 0, meanGrad: 0 };
  vals.sort((a, b) => a - b);
  const mean = sum / vals.length;
  let varSum = 0;
  for (let i = 0; i < vals.length; i++) {
    const d = vals[i] - mean;
    varSum += d * d;
  }
  const p10 = vals[Math.floor(vals.length * 0.1)];
  const p90 = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.9))];
  return {
    contrast: p90 - p10,
    std: Math.sqrt(varSum / vals.length),
    meanGrad: gradCnt > 0 ? gradSum / gradCnt : 0,
  };
}

/**
 * Evaluates local contrast & confidence for all detected objects/layers and scans for
 * unsegmented extremely low-contrast blind zones. Returns candidate segments (confidence < 0.88)
 * remapped to full working-image coordinates so the UI threshold slider filters instantaneously.
 */
export function detectLowConfidenceSegments(
  gray: Float32Array,
  w: number,
  cropRows: number,
  objects: DetectionObject[],
  shapes: OverlayShape[],
  roiOffset?: { x: number; y: number }
): LowConfidenceSegment[] {
  const ox = roiOffset?.x ?? 0;
  const oy = roiOffset?.y ?? 0;
  const candidates: LowConfidenceSegment[] = [];

  if (w < 16 || cropRows < 16) return candidates;

  // Build lookup of shape bounding boxes per objectId (in ROI-local coordinates)
  const bboxByObj = new Map<number, { x0: number; y0: number; x1: number; y1: number }>();
  for (const s of shapes) {
    if (s.objectId === undefined || s.layer === 'roi') continue;
    let cur = bboxByObj.get(s.objectId);
    const expand = (px: number, py: number) => {
      if (!cur) {
        cur = { x0: px, y0: py, x1: px, y1: py };
        bboxByObj.set(s.objectId!, cur);
      } else {
        if (px < cur.x0) cur.x0 = px;
        if (py < cur.y0) cur.y0 = py;
        if (px > cur.x1) cur.x1 = px;
        if (py > cur.y1) cur.y1 = py;
      }
    };
    if (s.kind === 'circle' && s.points.length >= 2) {
      const [cx, cy] = s.points[0];
      const r = Math.max(3, s.points[1][0]);
      expand(cx - r, cy - r);
      expand(cx + r, cy + r);
    } else if (s.kind === 'polygon' || s.kind === 'polyline' || s.kind === 'box') {
      for (const [px, py] of s.points) expand(px, py);
    }
  }

  // Global reference contrast across the working FOV
  const globalStats = measureBoxContrast(gray, w, cropRows, 0, 0, w, cropRows);
  const refContrast = Math.max(24, globalStats.contrast * 0.72);

  // Pass 1: Score detected objects / layers with local contrast awareness
  const coveredBoxes: Array<{ x0: number; y0: number; x1: number; y1: number }> = [];
  for (const obj of objects) {
    if (obj.excluded) continue;
    const rawBox = bboxByObj.get(obj.id);
    const pad = obj.kind === 'layer' ? 2 : 4;
    const fallbackR = Math.max(8, Math.min(28, (obj.diameter ?? obj.width ?? 16) * 0.5));
    const bx0 = Math.max(0, (rawBox ? rawBox.x0 : obj.x - fallbackR) - pad);
    const by0 = Math.max(0, (rawBox ? rawBox.y0 : obj.y - fallbackR) - pad);
    const bx1 = Math.min(w, (rawBox ? rawBox.x1 : obj.x + fallbackR) + pad);
    const by1 = Math.min(cropRows, (rawBox ? rawBox.y1 : obj.y + fallbackR) + pad);
    const bw = Math.max(6, bx1 - bx0);
    const bh = Math.max(6, by1 - by0);

    const stats = measureBoxContrast(gray, w, cropRows, bx0 - 3, by0 - 3, bx1 + 3, by1 + 3);
    const localContrast = Number(stats.contrast.toFixed(1));

    // Calibrate confidence using local contrast if the region is extremely low-contrast
    let conf = obj.confidence;
    if (obj.kind !== 'layer') {
      const cRatio = Math.min(1, stats.contrast / Math.max(22, refContrast * 0.65));
      const gRatio = Math.min(1, stats.meanGrad / Math.max(3.5, globalStats.meanGrad * 0.65));
      const contrastConf = 0.22 + 0.72 * (0.65 * cRatio + 0.35 * gRatio);
      if (stats.contrast < refContrast * 0.62 || stats.contrast < 24) {
        conf = Number(Math.min(conf, contrastConf).toFixed(2));
        obj.confidence = conf;
      }
    }

    if (conf < 0.88) {
      if (conf < 0.65) {
        obj.flagged = true;
        if (!obj.flagReason) {
          obj.flagReason = `极低对比度/弱置信 (${Math.round(conf * 100)}%, ΔI=${localContrast})`;
        }
      }
      coveredBoxes.push({ x0: bx0, y0: by0, x1: bx1, y1: by1 });
      const tagZh = obj.kind === 'layer' ? `第 ${obj.id} 层` : `目标 #${obj.id}`;
      const tagEn = obj.kind === 'layer' ? `Layer L${obj.id}` : `Object #${obj.id}`;
      candidates.push({
        id: `lowconf_obj_${obj.id}`,
        objectId: obj.id,
        kind: obj.kind === 'layer' ? 'layer' : 'object',
        bbox: {
          x: Math.round(bx0 + ox),
          y: Math.round(by0 + oy),
          width: Math.round(bw),
          height: Math.round(bh),
        },
        confidence: conf,
        localContrast,
        labelZh: tagZh,
        labelEn: tagEn,
        reasonZh:
          localContrast < 22
            ? `极低对比度片段 (ΔI=${localContrast}, 置信度 ${Math.round(conf * 100)}%)`
            : `弱边界/低置信片段 (ΔI=${localContrast}, 置信度 ${Math.round(conf * 100)}%)`,
        reasonEn:
          localContrast < 22
            ? `Very low-contrast segment (ΔI=${localContrast}, conf ${Math.round(conf * 100)}%)`
            : `Weak-boundary segment (ΔI=${localContrast}, conf ${Math.round(conf * 100)}%)`,
      });
    }
  }

  // Pass 2: Scan for unsegmented extremely low-contrast image zones (blind spots)
  const cols = Math.max(4, Math.min(7, Math.round(w / 90)));
  const rows = Math.max(4, Math.min(6, Math.round(cropRows / 80)));
  const cellW = w / cols;
  const cellH = cropRows / rows;

  const cellContrasts: number[] = [];
  const cellData: Array<{ r: number; c: number; contrast: number; meanGrad: number; low: boolean }> = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const st = measureBoxContrast(
        gray,
        w,
        cropRows,
        c * cellW,
        r * cellH,
        (c + 1) * cellW,
        (r + 1) * cellH
      );
      cellContrasts.push(st.contrast);
      cellData.push({ r, c, contrast: st.contrast, meanGrad: st.meanGrad, low: false });
    }
  }

  const sortedC = [...cellContrasts].sort((a, b) => a - b);
  const medC = sortedC[sortedC.length >> 1] || 20;
  const lowContrastCutoff = Math.max(14, Math.min(34, medC * 0.56));

  for (const cell of cellData) {
    if (cell.contrast <= lowContrastCutoff) {
      cell.low = true;
    }
  }

  // Merge adjacent low-contrast grid cells into clean bounding boxes (BFS connected components)
  const visited = new Uint8Array(rows * cols);
  const regionCandidates: LowConfidenceSegment[] = [];
  for (let i = 0; i < cellData.length; i++) {
    if (!cellData[i].low || visited[i]) continue;
    const queue = [i];
    visited[i] = 1;
    let minC = cellData[i].c;
    let maxC = cellData[i].c;
    let minR = cellData[i].r;
    let maxR = cellData[i].r;
    let cSum = 0;
    let cnt = 0;

    while (queue.length > 0) {
      const idx = queue.shift()!;
      const cur = cellData[idx];
      if (cur.c < minC) minC = cur.c;
      if (cur.c > maxC) maxC = cur.c;
      if (cur.r < minR) minR = cur.r;
      if (cur.r > maxR) maxR = cur.r;
      cSum += cur.contrast;
      cnt++;

      const dirs = [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
      ];
      for (const [dr, dc] of dirs) {
        const nr = cur.r + dr;
        const nc = cur.c + dc;
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
        const nIdx = nr * cols + nc;
        if (!visited[nIdx] && cellData[nIdx].low) {
          visited[nIdx] = 1;
          queue.push(nIdx);
        }
      }
    }

    // Avoid highlighting the entire frame if the whole image is uniformly low-contrast
    if (cnt >= rows * cols * 0.72) continue;

    const rx0 = minC * cellW;
    const ry0 = minR * cellH;
    const rx1 = (maxC + 1) * cellW;
    const ry1 = (maxR + 1) * cellH;
    const rw = rx1 - rx0;
    const rh = ry1 - ry0;
    if (rw < 22 || rh < 22) continue;

    // Skip if heavily overlapped by already-flagged object boxes
    const area = rw * rh;
    let coveredArea = 0;
    for (const cb of coveredBoxes) {
      const ix0 = Math.max(rx0, cb.x0);
      const iy0 = Math.max(ry0, cb.y0);
      const ix1 = Math.min(rx1, cb.x1);
      const iy1 = Math.min(ry1, cb.y1);
      if (ix1 > ix0 && iy1 > iy0) {
        coveredArea += (ix1 - ix0) * (iy1 - iy0);
      }
    }
    if (coveredArea / Math.max(1, area) > 0.45) continue;

    const avgContrast = Number((cSum / Math.max(1, cnt)).toFixed(1));
    const conf = Number(
      Math.max(0.16, Math.min(0.62, 0.14 + 0.52 * (avgContrast / Math.max(24, medC)))).toFixed(2)
    );
    const regIdx = regionCandidates.length + 1;
    regionCandidates.push({
      id: `lowconf_reg_${minR}_${minC}_${maxR}_${maxC}`,
      kind: 'region',
      bbox: {
        x: Math.round(rx0 + ox),
        y: Math.round(ry0 + oy),
        width: Math.round(rw),
        height: Math.round(rh),
      },
      confidence: conf,
      localContrast: avgContrast,
      labelZh: `低对比区 #${regIdx}`,
      labelEn: `Low-Contrast #${regIdx}`,
      reasonZh: `极低局部对比度盲区 (ΔI=${avgContrast}, 置信度 ${Math.round(conf * 100)}%)，点击可局部增强或手动数层`,
      reasonEn: `Extremely low local contrast zone (ΔI=${avgContrast}, conf ${Math.round(conf * 100)}%) — click to correct`,
    });
  }

  // Keep up to 6 lowest-confidence blind zones so canvas remains clean
  regionCandidates.sort((a, b) => a.confidence - b.confidence);
  candidates.push(...regionCandidates.slice(0, 6));

  // Sort all candidates ascending by confidence (weakest first)
  candidates.sort((a, b) => a.confidence - b.confidence);
  return candidates;
}
