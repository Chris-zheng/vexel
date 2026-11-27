import type {
  DetectionObject,
  HumanCorrection,
  OverlayShape,
  RunResult,
  Specimen,
  SummaryMetric,
} from '../types/vexel';
import {
  compute2DFFTDominantPeriod,
  computeOrientationField,
  type OrientationField,
} from './curvedLayerEnhance';

export interface MultilayerResult {
  objects: DetectionObject[];
  shapes: OverlayShape[];
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  warnings: string[];
  histogram?: RunResult['histogram'];
  profileChart?: RunResult['profileChart'];
  qcLabel: string;
}

function meanStdLocal(vals: number[]): { mean: number; sd: number } {
  if (vals.length === 0) return { mean: 0, sd: 0 };
  let sum = 0;
  for (let i = 0; i < vals.length; i++) sum += vals[i];
  const mean = sum / vals.length;
  let vSum = 0;
  for (let i = 0; i < vals.length; i++) {
    const d = vals[i] - mean;
    vSum += d * d;
  }
  return { mean, sd: Math.sqrt(vSum / vals.length) };
}

function medianOf(arr: Float64Array | number[]): number {
  const copy = Array.from(arr).sort((a, b) => a - b);
  if (copy.length === 0) return 0;
  const mid = copy.length >> 1;
  return copy.length % 2 === 1 ? copy[mid] : 0.5 * (copy[mid - 1] + copy[mid]);
}

function bilinearSample(
  gray: Float32Array,
  w: number,
  h: number,
  x: number,
  y: number
): number | null {
  if (x < 0 || y < 0 || x > w - 1 || y > h - 1) return null;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(w - 1, x0 + 1);
  const y1 = Math.min(h - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const i00 = gray[y0 * w + x0];
  const i10 = gray[y0 * w + x1];
  const i01 = gray[y1 * w + x0];
  const i11 = gray[y1 * w + x1];
  return (1 - fx) * (1 - fy) * i00 + fx * (1 - fy) * i10 + (1 - fx) * fy * i01 + fx * fy * i11;
}

function gaussianSmooth1D(src: Float64Array, sigma: number): Float64Array {
  const n = src.length;
  if (n < 3 || sigma <= 0.15) return Float64Array.from(src);
  const rad = Math.max(1, Math.min(24, Math.ceil(sigma * 3)));
  const kernel = new Float64Array(rad * 2 + 1);
  let kSum = 0;
  const denom = 2 * sigma * sigma;
  for (let k = -rad; k <= rad; k++) {
    const w = Math.exp(-(k * k) / denom);
    kernel[k + rad] = w;
    kSum += w;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;

  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let k = -rad; k <= rad; k++) {
      const idx = Math.max(0, Math.min(n - 1, i + k));
      acc += src[idx] * kernel[k + rad];
    }
    out[i] = acc;
  }
  return out;
}

function refinePeakSubpixel(arr: Float64Array, idx: number): number {
  if (idx <= 0 || idx >= arr.length - 1) return idx;
  const y0 = arr[idx - 1];
  const y1 = arr[idx];
  const y2 = arr[idx + 1];
  const denom = y0 - 2 * y1 + y2;
  if (Math.abs(denom) < 1e-9) return idx;
  const delta = 0.5 * (y0 - y2) / denom;
  return idx + Math.max(-0.5, Math.min(0.5, delta));
}

function nmsPeaks(
  candidates: Array<{ pos: number; score: number }>,
  minDist: number
): number[] {
  const sorted = [...candidates].sort((a, b) => b.score - a.score);
  const kept: Array<{ pos: number; score: number }> = [];
  for (const c of sorted) {
    if (kept.every((k) => Math.abs(k.pos - c.pos) >= minDist)) {
      kept.push(c);
    }
  }
  return kept.map((k) => k.pos).sort((a, b) => a - b);
}

/**
 * Center Multilayer Stack Count & Layer Thickness Metrology
 *
 * 1. Estimates dominant layer orientation & slight sample tilt (±22°) using the 2x2 structure tensor
 *    inside the central region.
 * 2. Samples a tilt-compensated 1D intensity profile averaged across a tunable center ROI band so
 *    electron-microscopy speckle noise is suppressed without blurring thin tilted layers.
 * 3. Detects sub-pixel layer interfaces via one of 4 physical modes:
 *    - gradient: Step-contrast interface gradient peaks |dI/ds| (general multilayer films/coatings)
 *    - alternating: Detrended zero-crossings inside active contrast envelope (bright/dark periodic stacks)
 *    - dark_seams: Local dark seam valleys between layers (delamination, grain/glue seams, dark trenches)
 *    - bright_seams: Local bright interface ridges between layers
 * 4. Measures each individual layer thickness (sub-pixel -> physical unit), total center stack thickness,
 *    mean/SD/CV thickness, and emits rich canvas polygons, interface lines, and 1D profile markers.
 */
export function runMultilayerAnalysis(
  specimen: Specimen,
  gray: Float32Array,
  params: Record<string, number | boolean | string>,
  corrections: HumanCorrection[] = []
): MultilayerResult {
  const { width, height, calibration } = specimen;
  const cropRows = calibration.cropRows ?? height;
  const pxSize = calibration.pixelSize;
  const unit = calibration.spaceUnit;

  const geometryMode = String(params.geometryMode ?? 'straight') as 'straight' | 'curved';
  const orientation = String(params.orientation ?? 'auto') as 'auto' | 'horizontal' | 'vertical';
  const detectionMode = String(params.detectionMode ?? 'gradient') as
    | 'gradient'
    | 'alternating'
    | 'dark_seams'
    | 'bright_seams';
  const sensitivity = Math.max(0.05, Math.min(0.98, Number(params.sensitivity ?? 0.48)));
  const minLayerPx = Math.max(2, Math.min(120, Number(params.minLayerPx ?? 5)));
  const smoothSigma = Math.max(0.3, Math.min(8, Number(params.smoothSigma ?? 1.25)));
  const roiWidthPct = Math.max(5, Math.min(95, Number(params.roiWidthPct ?? 36)));
  const roiCenterOffsetPct = Math.max(-45, Math.min(45, Number(params.roiCenterOffsetPct ?? 0)));
  const rawStartPct = Math.max(0, Math.min(92, Number(params.stackStartPct ?? 4)));
  const rawEndPct = Math.max(8, Math.min(100, Number(params.stackEndPct ?? 96)));
  const stackStartPct = Math.min(rawStartPct, rawEndPct - 6);
  const stackEndPct = Math.max(rawEndPct, stackStartPct + 6);
  const autoTrimExterior = params.autoTrimExterior !== false;

  // 1. Structure tensor inside central 60% x 60% region to determine stack normal & tilt
  const xStart = Math.max(2, Math.floor(width * 0.2));
  const xEnd = Math.min(width - 3, Math.floor(width * 0.8));
  const yStart = Math.max(2, Math.floor(cropRows * 0.2));
  const yEnd = Math.min(cropRows - 3, Math.floor(cropRows * 0.8));

  let jxx = 0;
  let jyy = 0;
  let jxy = 0;
  const strideStep = Math.max(1, Math.floor(Math.min(width, cropRows) / 220));
  for (let y = yStart; y <= yEnd; y += strideStep) {
    const row = y * width;
    for (let x = xStart; x <= xEnd; x += strideStep) {
      const gx = 0.5 * (gray[row + x + 1] - gray[row + x - 1]);
      const gy = 0.5 * (gray[row + width + x] - gray[row - width + x]);
      jxx += gx * gx;
      jyy += gy * gy;
      jxy += gx * gy;
    }
  }

  // Dominant eigenvector angle of J points along gradient (normal to layers)
  const rawNormalPhi = 0.5 * Math.atan2(2 * jxy, jxx - jyy); // in [-pi/2, pi/2]
  const maxTiltRad = (22 * Math.PI) / 180;

  let isHorizLayers: boolean;
  if (orientation === 'horizontal') {
    isHorizLayers = true;
  } else if (orientation === 'vertical') {
    isHorizLayers = false;
  } else {
    // Horizontal layers have strong vertical gradient (jyy >= jxx)
    isHorizLayers = jyy >= jxx;
  }

  let normalAngle: number;
  if (isHorizLayers) {
    // Expected normal near +pi/2 (pointing downward)
    let tilt = 0;
    if (Math.abs(rawNormalPhi) > Math.PI / 4) {
      tilt = rawNormalPhi > 0 ? rawNormalPhi - Math.PI / 2 : rawNormalPhi + Math.PI / 2;
    }
    tilt = Math.max(-maxTiltRad, Math.min(maxTiltRad, tilt));
    normalAngle = Math.PI / 2 + tilt;
  } else {
    // Expected normal near 0 (pointing rightward)
    let tilt = Math.abs(rawNormalPhi) <= Math.PI / 4 ? rawNormalPhi : 0;
    tilt = Math.max(-maxTiltRad, Math.min(maxTiltRad, tilt));
    normalAngle = tilt;
  }

  const nx = Math.cos(normalAngle);
  const ny = Math.sin(normalAngle);
  // Layer tangent unit vector (parallel to interfaces)
  const tx = -ny;
  const ty = nx;
  const layerAngleDeg = Number(
    ((((Math.atan2(ty, tx) * 180) / Math.PI) % 180 + 180) % 180).toFixed(1)
  );
  const tiltDeg = isHorizLayers
    ? Number((((normalAngle - Math.PI / 2) * 180) / Math.PI).toFixed(1))
    : Number(((normalAngle * 180) / Math.PI).toFixed(1));

  // 2. Center ROI band geometry & 1D intensity projection (Straight tilt-compensated OR Curved streamline)
  const stackLen = isHorizLayers ? cropRows : width;
  const bandBaseLen = isHorizLayers ? width : cropRows;
  const halfBand = Math.max(4, 0.5 * (roiWidthPct / 100) * bandBaseLen);
  const offsetPx = (roiCenterOffsetPct / 100) * bandBaseLen;

  const cx0 = width * 0.5 + tx * offsetPx;
  const cy0 = cropRows * 0.5 + ty * offsetPx;

  const sMin = -0.5 * stackLen + (stackStartPct / 100) * stackLen;
  const sMax = -0.5 * stackLen + (stackEndPct / 100) * stackLen;

  let sampleCount = Math.max(16, Math.floor(sMax - sMin) + 1);
  let rawProfile = new Float64Array(sampleCount);

  // Curved mode geometric state (spine points + helper to trace curved iso-layer arcs)
  let orientField: OrientationField | null = null;
  let curvedSpine: Array<{ x: number; y: number; tx: number; ty: number }> = [];

  const traceIsoLayerArc = (
    sx: number,
    sy: number,
    t0x: number,
    t0y: number,
    maxArcHalf: number,
    stepSize: number,
    field: OrientationField
  ): { pts: Array<[number, number]>; meanGray: number } => {
    const traceSide = (dir: 1 | -1): { pts: Array<[number, number]>; sum: number; cnt: number } => {
      const pts: Array<[number, number]> = [];
      let px = sx;
      let py = sy;
      let vx = t0x * dir;
      let vy = t0y * dir;
      let sum = 0;
      let cnt = 0;
      const nSteps = Math.max(1, Math.floor(maxArcHalf / stepSize));

      for (let st = 1; st <= nSteps; st++) {
        px += vx * stepSize;
        py += vy * stepSize;
        if (px < 1 || py < 1 || px > width - 2 || py > cropRows - 2) break;

        const ix = Math.round(px);
        const iy = Math.round(py);
        const pIdx = iy * width + ix;
        let ltx = field.tx[pIdx];
        let lty = field.ty[pIdx];
        const dot = ltx * vx + lty * vy;
        if (dot < 0) {
          ltx = -ltx;
          lty = -lty;
        }
        // Halt arc tracking gently if tangent makes a sharp turn (e.g., hitting a vertical crack/domain wall)
        if (Math.abs(dot) < 0.42) break;

        const coh = field.coherence[pIdx];
        const wLocal = 0.55 * (0.35 + 0.65 * coh);
        const bdx = wLocal * ltx + (1 - wLocal) * vx;
        const bdy = wLocal * lty + (1 - wLocal) * vy;
        const blen = Math.hypot(bdx, bdy) || 1;
        vx = bdx / blen;
        vy = bdy / blen;

        pts.push([px, py]);
        const v = bilinearSample(gray, width, cropRows, px, py);
        if (v !== null) {
          sum += v;
          cnt++;
        }
      }
      return { pts, sum, cnt };
    };

    const left = traceSide(-1);
    const right = traceSide(1);
    const centerVal = bilinearSample(gray, width, cropRows, sx, sy);
    let totalSum = left.sum + right.sum;
    let totalCnt = left.cnt + right.cnt;
    if (centerVal !== null) {
      totalSum += centerVal;
      totalCnt++;
    }
    const pts: Array<[number, number]> = [...left.pts.reverse(), [sx, sy], ...right.pts];
    return {
      pts,
      meanGray: totalCnt > 0 ? totalSum / totalCnt : 0,
    };
  };

  if (geometryMode === 'curved') {
    orientField = computeOrientationField(
      gray,
      width,
      cropRows,
      Math.max(2.6, Math.min(6.5, smoothSigma * 2.4))
    );

    // Clamp seed point into active frame
    const seedX = Math.max(4, Math.min(width - 5, cx0));
    const seedY = Math.max(4, Math.min(cropRows - 5, cy0));
    const seedIdx = Math.round(seedY) * width + Math.round(seedX);
    let n0x = orientField.nx[seedIdx];
    let n0y = orientField.ny[seedIdx];
    if (n0x * nx + n0y * ny < 0) {
      n0x = -n0x;
      n0y = -n0y;
    }
    if (orientField.coherence[seedIdx] < 0.08) {
      n0x = nx;
      n0y = ny;
    }

    const marchSpine = (dir: 1 | -1, maxSteps: number) => {
      const pts: Array<{ x: number; y: number; tx: number; ty: number }> = [];
      let px = seedX;
      let py = seedY;
      let vx = n0x * dir;
      let vy = n0y * dir;

      for (let st = 1; st <= maxSteps; st++) {
        px += vx;
        py += vy;
        if (px < 2 || py < 2 || px > width - 3 || py > cropRows - 3) break;
        const ix = Math.round(px);
        const iy = Math.round(py);
        const pIdx = iy * width + ix;
        let lnx = orientField!.nx[pIdx];
        let lny = orientField!.ny[pIdx];
        const dot = lnx * vx + lny * vy;
        if (dot < 0) {
          lnx = -lnx;
          lny = -lny;
        }
        const coh = orientField!.coherence[pIdx];
        // Blend smoothly with prior normal so noisy specks or tiny cracks do not derail the spine
        const wLocal = Math.abs(dot) > 0.45 ? 0.45 * (0.2 + 0.8 * coh) : 0.08;
        const bnx = wLocal * lnx + (1 - wLocal) * vx;
        const bny = wLocal * lny + (1 - wLocal) * vy;
        const blen = Math.hypot(bnx, bny) || 1;
        vx = bnx / blen;
        vy = bny / blen;

        // Tangent perpendicular to current forward normal (oriented consistently with global tx, ty)
        const fwdNx = vx * dir;
        const fwdNy = vy * dir;
        let cTx = -fwdNy;
        let cTy = fwdNx;
        if (cTx * tx + cTy * ty < 0) {
          cTx = -cTx;
          cTy = -cTy;
        }
        pts.push({ x: px, y: py, tx: cTx, ty: cTy });
      }
      return pts;
    };

    const maxHalfSpine = Math.max(16, Math.round(Math.hypot(width, cropRows) * 0.55));
    const backPts = marchSpine(-1, maxHalfSpine).reverse();
    const fwdPts = marchSpine(1, maxHalfSpine);
    let seedTx = -n0y;
    let seedTy = n0x;
    if (seedTx * tx + seedTy * ty < 0) {
      seedTx = -seedTx;
      seedTy = -seedTy;
    }
    const fullSpine = [...backPts, { x: seedX, y: seedY, tx: seedTx, ty: seedTy }, ...fwdPts];
    const iStart = Math.max(0, Math.floor((stackStartPct / 100) * (fullSpine.length - 1)));
    const iEnd = Math.min(
      fullSpine.length - 1,
      Math.max(iStart + 15, Math.ceil((stackEndPct / 100) * (fullSpine.length - 1)))
    );
    curvedSpine = fullSpine.slice(iStart, iEnd + 1);
    sampleCount = curvedSpine.length;
    rawProfile = new Float64Array(sampleCount);

    const arcStep = Math.max(1.25, halfBand / 28);
    for (let k = 0; k < sampleCount; k++) {
      const sp = curvedSpine[k];
      const arc = traceIsoLayerArc(sp.x, sp.y, sp.tx, sp.ty, halfBand, arcStep, orientField);
      rawProfile[k] = arc.meanGray;
    }
  } else {
    const uStep = Math.max(1, halfBand / 90);
    for (let k = 0; k < sampleCount; k++) {
      const s = sMin + k;
      const bx = cx0 + nx * s;
      const by = cy0 + ny * s;
      let acc = 0;
      let cnt = 0;
      for (let u = -halfBand; u <= halfBand; u += uStep) {
        const val = bilinearSample(gray, width, cropRows, bx + tx * u, by + ty * u);
        if (val !== null) {
          acc += val;
          cnt++;
        }
      }
      rawProfile[k] = cnt > 0 ? acc / cnt : k > 0 ? rawProfile[k - 1] : 0;
    }
  }

  // 3. 1D Gaussian smoothing & derivative
  const smoothP = gaussianSmooth1D(rawProfile, smoothSigma);
  const grad = new Float64Array(sampleCount);
  for (let k = 1; k < sampleCount - 1; k++) {
    grad[k] = Math.abs(0.5 * (smoothP[k + 1] - smoothP[k - 1]));
  }
  grad[0] = grad[1] ?? 0;
  grad[sampleCount - 1] = grad[sampleCount - 2] ?? 0;

  // 4. Detect sub-pixel layer interfaces according to selected physical mode
  let interfaceIndices: number[] = [];

  if (detectionMode === 'gradient') {
    let gradMax = 0;
    for (let k = 1; k < sampleCount - 1; k++) {
      if (grad[k] > gradMax) gradMax = grad[k];
    }
    const gradMed = medianOf(grad);
    const absDevs = new Float64Array(sampleCount);
    for (let k = 0; k < sampleCount; k++) absDevs[k] = Math.abs(grad[k] - gradMed);
    const gradMad = Math.max(0.4, medianOf(absDevs) * 1.4826);

    const floorThresh = Math.max(
      1.0,
      gradMed + (1 - sensitivity) * 4.2 * gradMad,
      gradMax * (0.05 + (1 - sensitivity) * 0.42)
    );

    const cands: Array<{ pos: number; score: number }> = [];
    const promWin = Math.max(3, Math.min(28, Math.round(minLayerPx * 1.6)));
    for (let k = 2; k < sampleCount - 2; k++) {
      const v = grad[k];
      if (v < floorThresh) continue;
      if (v >= grad[k - 1] && v > grad[k + 1]) {
        let leftMin = v;
        for (let j = Math.max(0, k - promWin); j < k; j++) {
          if (grad[j] < leftMin) leftMin = grad[j];
        }
        let rightMin = v;
        for (let j = k + 1; j <= Math.min(sampleCount - 1, k + promWin); j++) {
          if (grad[j] < rightMin) rightMin = grad[j];
        }
        const prom = v - Math.max(leftMin, rightMin);
        if (prom >= floorThresh * 0.25) {
          cands.push({ pos: refinePeakSubpixel(grad, k), score: v + prom * 0.5 });
        }
      }
    }
    interfaceIndices = nmsPeaks(cands, minLayerPx);
  } else if (detectionMode === 'alternating') {
    // Detrend with adaptive local moving average so illumination drift does not break layers
    const winHalf = Math.max(
      6,
      Math.min(Math.floor(sampleCount * 0.28), Math.round(minLayerPx * 2.5))
    );
    const prefix = new Float64Array(sampleCount + 1);
    for (let k = 0; k < sampleCount; k++) prefix[k + 1] = prefix[k] + smoothP[k];
    const detrended = new Float64Array(sampleCount);
    for (let k = 0; k < sampleCount; k++) {
      const a = Math.max(0, k - winHalf);
      const b = Math.min(sampleCount - 1, k + winHalf);
      const localMean = (prefix[b + 1] - prefix[a]) / (b - a + 1);
      detrended[k] = smoothP[k] - localMean;
    }

    // Active local contrast envelope to isolate the center layered region from flat air/substrate
    const env = new Float64Array(sampleCount);
    const envRad = Math.max(4, Math.min(30, Math.round(minLayerPx * 2)));
    let maxEnv = 0;
    for (let k = 0; k < sampleCount; k++) {
      let localAmp = 0;
      const a = Math.max(0, k - envRad);
      const b = Math.min(sampleCount - 1, k + envRad);
      for (let j = a; j <= b; j++) {
        const av = Math.abs(detrended[j]);
        if (av > localAmp) localAmp = av;
      }
      env[k] = localAmp;
      if (localAmp > maxEnv) maxEnv = localAmp;
    }

    const ampThresh = Math.max(2.0, maxEnv * (0.07 + (1 - sensitivity) * 0.36));
    const rawCrossings: number[] = [];
    for (let k = 1; k < sampleCount - 1; k++) {
      if (env[k] < ampThresh) continue;
      const d0 = detrended[k - 1];
      const d1 = detrended[k];
      if ((d0 <= 0 && d1 > 0) || (d0 >= 0 && d1 < 0)) {
        const frac = Math.abs(d1 - d0) > 1e-9 ? -d0 / (d1 - d0) : 0.5;
        const pos = k - 1 + Math.max(0, Math.min(1, frac));
        rawCrossings.push(pos);
      }
    }

    // Filter out micro-intervals narrower than minLayerPx or weaker than ampThresh
    const cleaned: number[] = [];
    for (let i = 0; i < rawCrossings.length; i++) {
      const pos = rawCrossings[i];
      if (cleaned.length === 0) {
        cleaned.push(pos);
      } else {
        const prev = cleaned[cleaned.length - 1];
        if (pos - prev >= minLayerPx) {
          // Verify the segment between prev and pos actually has sufficient contrast
          const i0 = Math.max(0, Math.floor(prev));
          const i1 = Math.min(sampleCount - 1, Math.ceil(pos));
          let peakAmp = 0;
          for (let j = i0; j <= i1; j++) {
            const av = Math.abs(detrended[j]);
            if (av > peakAmp) peakAmp = av;
          }
          if (peakAmp >= ampThresh * 0.65) {
            cleaned.push(pos);
          } else {
            cleaned[cleaned.length - 1] = pos;
          }
        }
      }
    }
    interfaceIndices = cleaned;
  } else {
    // dark_seams or bright_seams: detect local valleys / ridges between layers
    const isDark = detectionMode === 'dark_seams';
    const sig = new Float64Array(sampleCount);
    for (let k = 0; k < sampleCount; k++) {
      sig[k] = isDark ? -smoothP[k] : smoothP[k];
    }

    const searchRad = Math.max(4, Math.min(36, Math.round(minLayerPx * 1.8)));
    const proms = new Float64Array(sampleCount);
    let maxProm = 0;
    for (let k = 2; k < sampleCount - 2; k++) {
      const v = sig[k];
      if (v >= sig[k - 1] && v > sig[k + 1]) {
        let leftMin = v;
        for (let j = Math.max(0, k - searchRad); j < k; j++) {
          if (sig[j] < leftMin) leftMin = sig[j];
        }
        let rightMin = v;
        for (let j = k + 1; j <= Math.min(sampleCount - 1, k + searchRad); j++) {
          if (sig[j] < rightMin) rightMin = sig[j];
        }
        const prom = v - Math.max(leftMin, rightMin);
        proms[k] = prom;
        if (prom > maxProm) maxProm = prom;
      }
    }

    const promThresh = Math.max(1.8, maxProm * (0.06 + (1 - sensitivity) * 0.42));
    const cands: Array<{ pos: number; score: number }> = [];
    for (let k = 2; k < sampleCount - 2; k++) {
      if (proms[k] >= promThresh) {
        cands.push({ pos: refinePeakSubpixel(sig, k), score: proms[k] });
      }
    }
    interfaceIndices = nmsPeaks(cands, minLayerPx);
  }

  // Optionally include exterior segments within [stackStartPct, stackEndPct] when autoTrimExterior is off
  const boundaryIndices = [...interfaceIndices];
  if (!autoTrimExterior) {
    if (boundaryIndices.length === 0) {
      boundaryIndices.push(0, sampleCount - 1);
    } else {
      if (boundaryIndices[0] >= minLayerPx) boundaryIndices.unshift(0);
      if (sampleCount - 1 - boundaryIndices[boundaryIndices.length - 1] >= minLayerPx) {
        boundaryIndices.push(sampleCount - 1);
      }
    }
  }

  // 5. Construct per-layer measurements & canvas overlays
  const LAYER_PALETTE = ['#06B6D4', '#10B981', '#F59E0B', '#A78BFA', '#38BDF8', '#EC4899'];
  const corrForSpec = corrections.filter((c) => c.specimenId === specimen.id);

  // Helper: map (sIndex, uOffset) -> [x, y] image coords (supports both straight & curved spine)
  const sampleSpineAt = (sIdx: number): { x: number; y: number; tx: number; ty: number } => {
    if (geometryMode === 'curved' && curvedSpine.length >= 2) {
      const clamped = Math.max(0, Math.min(curvedSpine.length - 1, sIdx));
      const i0 = Math.floor(clamped);
      const i1 = Math.min(curvedSpine.length - 1, i0 + 1);
      const f = clamped - i0;
      const a = curvedSpine[i0];
      const b = curvedSpine[i1];
      const vx = a.tx * (1 - f) + b.tx * f;
      const vy = a.ty * (1 - f) + b.ty * f;
      const vl = Math.hypot(vx, vy) || 1;
      return {
        x: a.x * (1 - f) + b.x * f,
        y: a.y * (1 - f) + b.y * f,
        tx: vx / vl,
        ty: vy / vl,
      };
    }
    const s = sMin + sIdx;
    return { x: cx0 + nx * s, y: cy0 + ny * s, tx, ty };
  };

  const toImgPt = (sIdx: number, u: number): [number, number] => {
    const sp = sampleSpineAt(sIdx);
    return [sp.x + sp.tx * u, sp.y + sp.ty * u];
  };

  // Precompute curved or straight polyline for each detected interface
  const interfacePolylines: Array<Array<[number, number]>> = boundaryIndices.map((eIdx) => {
    if (geometryMode === 'curved' && orientField && curvedSpine.length >= 2) {
      const sp = sampleSpineAt(eIdx);
      const step = Math.max(1.5, halfBand / 20);
      return traceIsoLayerArc(sp.x, sp.y, sp.tx, sp.ty, halfBand, step, orientField).pts;
    }
    return [toImgPt(eIdx, -halfBand), toImgPt(eIdx, halfBand)];
  });

  // Compute global stack median gray for Bright/Dark tagging
  const stackRegionVals: number[] = [];
  if (boundaryIndices.length >= 2) {
    const a0 = Math.max(0, Math.floor(boundaryIndices[0]));
    const b0 = Math.min(sampleCount - 1, Math.ceil(boundaryIndices[boundaryIndices.length - 1]));
    for (let k = a0; k <= b0; k++) stackRegionVals.push(rawProfile[k]);
  }
  const stackMedianGray = stackRegionVals.length > 0 ? medianOf(stackRegionVals) : medianOf(rawProfile);

  const objects: DetectionObject[] = [];
  const shapes: OverlayShape[] = [];
  const warnings: string[] = [];

  // Draw the Center Measurement ROI frame (curved envelope or straight tilted box)
  if (geometryMode === 'curved' && orientField && curvedSpine.length >= 4) {
    const leftEdge: Array<[number, number]> = [];
    const rightEdge: Array<[number, number]> = [];
    const envStep = Math.max(1, Math.floor(curvedSpine.length / 24));
    for (let k = 0; k < curvedSpine.length; k += envStep) {
      const sp = curvedSpine[k];
      const arc = traceIsoLayerArc(sp.x, sp.y, sp.tx, sp.ty, halfBand, Math.max(2, halfBand / 14), orientField);
      if (arc.pts.length >= 2) {
        leftEdge.push(arc.pts[0]);
        rightEdge.push(arc.pts[arc.pts.length - 1]);
      }
    }
    if (leftEdge.length >= 2 && rightEdge.length >= 2) {
      shapes.push({
        id: 'ml_roi_box',
        kind: 'polygon',
        points: [...leftEdge, ...rightEdge.reverse()],
        color: 'rgba(148, 163, 184, 0.55)',
        dashed: true,
        layer: 'roi',
      });
    }
  } else {
    shapes.push({
      id: 'ml_roi_box',
      kind: 'polygon',
      points: [
        toImgPt(0, -halfBand),
        toImgPt(0, halfBand),
        toImgPt(sampleCount - 1, halfBand),
        toImgPt(sampleCount - 1, -halfBand),
      ],
      color: 'rgba(148, 163, 184, 0.55)',
      dashed: true,
      layer: 'roi',
    });
  }

  // Draw the center normal spine across the measured stack
  if (boundaryIndices.length >= 2) {
    if (geometryMode === 'curved' && curvedSpine.length >= 2) {
      const aIdx = Math.max(0, Math.floor(boundaryIndices[0]));
      const bIdx = Math.min(curvedSpine.length - 1, Math.ceil(boundaryIndices[boundaryIndices.length - 1]));
      const spinePts: Array<[number, number]> = [];
      const stride = Math.max(1, Math.floor((bIdx - aIdx) / 60));
      for (let k = aIdx; k <= bIdx; k += stride) {
        spinePts.push([curvedSpine[k].x, curvedSpine[k].y]);
      }
      const endPt = sampleSpineAt(boundaryIndices[boundaryIndices.length - 1]);
      spinePts.push([endPt.x, endPt.y]);
      shapes.push({
        id: 'ml_spine',
        kind: 'polyline',
        points: spinePts,
        color: 'rgba(226, 232, 240, 0.72)',
        dashed: true,
        layer: 'marks',
      });
    } else {
      shapes.push({
        id: 'ml_spine',
        kind: 'polyline',
        points: [
          toImgPt(boundaryIndices[0], 0),
          toImgPt(boundaryIndices[boundaryIndices.length - 1], 0),
        ],
        color: 'rgba(226, 232, 240, 0.65)',
        dashed: true,
        layer: 'marks',
      });
    }
  }

  // Reference gradient and stack gray dynamic range for physics-based per-layer confidence
  const stackGrads: number[] = [];
  if (boundaryIndices.length >= 2) {
    const a0 = Math.max(0, Math.floor(boundaryIndices[0]));
    const b0 = Math.min(sampleCount - 1, Math.ceil(boundaryIndices[boundaryIndices.length - 1]));
    for (let k = a0; k <= b0; k++) stackGrads.push(grad[k]);
  }
  stackGrads.sort((u, v) => u - v);
  const refGrad =
    stackGrads.length > 0 ? Math.max(6, stackGrads[Math.floor(stackGrads.length * 0.78)]) : 10;
  const sortedStackGray = [...stackRegionVals].sort((u, v) => u - v);
  const stackGraySpan =
    sortedStackGray.length >= 4
      ? Math.max(
          14,
          sortedStackGray[Math.floor(sortedStackGray.length * 0.92)] -
            sortedStackGray[Math.floor(sortedStackGray.length * 0.08)]
        )
      : 32;

  // Precompute per-segment mean gray so we can evaluate contrast against immediate neighbors
  const segMeanGrays: number[] = [];
  for (let i = 0; i + 1 < boundaryIndices.length; i++) {
    const k0 = Math.max(0, Math.ceil(boundaryIndices[i]));
    const k1 = Math.min(sampleCount - 1, Math.floor(boundaryIndices[i + 1]));
    let sSum = 0;
    let sCnt = 0;
    for (let k = k0; k <= k1; k++) {
      sSum += rawProfile[k];
      sCnt++;
    }
    segMeanGrays.push(
      sCnt > 0
        ? sSum / sCnt
        : rawProfile[Math.min(sampleCount - 1, Math.round(0.5 * (boundaryIndices[i] + boundaryIndices[i + 1])))]
    );
  }

  let layerIdx = 0;
  for (let i = 0; i + 1 < boundaryIndices.length; i++) {
    const a = boundaryIndices[i];
    const b = boundaryIndices[i + 1];
    const thickPx = b - a;
    if (thickPx < Math.max(1.5, minLayerPx * 0.7)) continue;

    layerIdx++;
    const objId = layerIdx;
    const thickPhys = thickPx * pxSize;

    const meanGray = segMeanGrays[i] ?? stackMedianGray;
    const isBright = meanGray >= stackMedianGray;

    // Physics-based local contrast & interface confidence (drops naturally in very low-contrast zones)
    const idxA = Math.max(0, Math.min(sampleCount - 1, Math.round(a)));
    const idxB = Math.max(0, Math.min(sampleCount - 1, Math.round(b)));
    const edgeGrad = 0.5 * (grad[idxA] + grad[idxB]);
    let neighborDelta = Math.abs(meanGray - stackMedianGray) * 1.4;
    if (i > 0 && i + 1 < segMeanGrays.length) {
      neighborDelta = 0.5 * (Math.abs(meanGray - segMeanGrays[i - 1]) + Math.abs(meanGray - segMeanGrays[i + 1]));
    } else if (i > 0) {
      neighborDelta = Math.abs(meanGray - segMeanGrays[i - 1]);
    } else if (i + 1 < segMeanGrays.length) {
      neighborDelta = Math.abs(meanGray - segMeanGrays[i + 1]);
    }

    let avgCoh = 1.0;
    if (geometryMode === 'curved' && orientField && curvedSpine.length >= 2) {
      let cSum = 0;
      let cCnt = 0;
      const k0 = Math.max(0, Math.floor(a));
      const k1 = Math.min(curvedSpine.length - 1, Math.ceil(b));
      for (let k = k0; k <= k1; k++) {
        const sp = curvedSpine[k];
        const ix = Math.max(0, Math.min(width - 1, Math.round(sp.x)));
        const iy = Math.max(0, Math.min(cropRows - 1, Math.round(sp.y)));
        cSum += orientField.coherence[iy * width + ix];
        cCnt++;
      }
      avgCoh = cCnt > 0 ? cSum / cCnt : 0.5;
    }

    const gradScore = Math.min(1, edgeGrad / Math.max(7, refGrad * 0.85));
    const contrastScore = Math.min(1, neighborDelta / Math.max(12, stackGraySpan * 0.34));
    const cohScore = geometryMode === 'curved' ? Math.min(1, avgCoh / 0.42) : 1.0;
    const rawConf = 0.42 * gradScore + 0.43 * contrastScore + 0.15 * cohScore;
    const confidence = Number(Math.max(0.18, Math.min(0.98, 0.2 + 0.78 * rawConf)).toFixed(2));
    const isLowConf = confidence < 0.65;

    let excluded = false;
    for (const corr of corrForSpec) {
      if (corr.objectId !== objId) continue;
      if (corr.field === 'excluded') excluded = Boolean(corr.newValue);
    }

    const midS = 0.5 * (a + b);
    const [midX, midY] = toImgPt(midS, 0);
    const color =
      detectionMode === 'alternating'
        ? isBright
          ? '#06B6D4'
          : '#F59E0B'
        : LAYER_PALETTE[(objId - 1) % LAYER_PALETTE.length];

    const toneTag =
      detectionMode === 'alternating' ? (isBright ? '亮层' : '暗层') : isBright ? '偏亮' : '偏暗';

    objects.push({
      id: objId,
      kind: 'layer',
      x: midX,
      y: midY,
      thickness: Number(thickPhys.toFixed(2)),
      width: Number(thickPhys.toFixed(2)),
      orientationDeg: layerAngleDeg,
      cls: isBright ? 'positive' : 'normal',
      reason: `${thickPx.toFixed(1)} px · I=${meanGray.toFixed(0)} · ΔI=${neighborDelta.toFixed(0)} · ${toneTag}`,
      flagged: isLowConf,
      flagReason: isLowConf
        ? `极低对比度/弱界面 (置信度 ${Math.round(confidence * 100)}%, ΔI=${neighborDelta.toFixed(0)})`
        : undefined,
      excluded,
      confidence,
    });

    if (!excluded) {
      // Translucent layer band (curved polygon in curved mode, tilted quad in straight mode)
      const topArc = interfacePolylines[i];
      const botArc = interfacePolylines[i + 1];
      const layerPoly: Array<[number, number]> =
        topArc && botArc && topArc.length >= 2 && botArc.length >= 2
          ? [...topArc, ...[...botArc].reverse()]
          : [
              toImgPt(a, -halfBand),
              toImgPt(a, halfBand),
              toImgPt(b, halfBand),
              toImgPt(b, -halfBand),
            ];
      shapes.push({
        id: `ml_layer_${objId}`,
        kind: 'polygon',
        points: layerPoly,
        color,
        fill: `${color}28`,
        objectId: objId,
        layer: 'segmentation',
      });

      // Center thickness dimension segment + label right beside the ROI band
      const labelPos = toImgPt(midS, halfBand * 0.55);
      shapes.push({
        id: `ml_tick_${objId}`,
        kind: 'polyline',
        points: [toImgPt(a, 0), toImgPt(b, 0)],
        color,
        objectId: objId,
        layer: 'marks',
      });
      shapes.push({
        id: `ml_lbl_${objId}`,
        kind: 'point',
        points: [labelPos],
        color,
        label: `L${objId}: ${thickPhys.toFixed(1)} ${unit}`,
        objectId: objId,
        layer: 'marks',
      });
    }
  }

  // Draw crisp sub-pixel interface lines (curved polylines or straight tilted segments)
  interfacePolylines.forEach((pts, idx) => {
    if (pts.length < 2) return;
    shapes.push({
      id: `ml_edge_${idx}`,
      kind: 'polyline',
      points: pts,
      color: '#FFFFFF',
      layer: 'edges',
    });
  });

  // 6. Quantitative statistics
  const activeLayers = objects.filter((o) => !o.excluded);
  const thicknesses = activeLayers.map((o) => o.thickness ?? o.width ?? 0);
  const totalThickness = thicknesses.reduce((acc, v) => acc + v, 0);
  const { mean: meanThick, sd: sdThick } = meanStdLocal(thicknesses);
  const cvPct = meanThick > 0 ? (sdThick / meanThick) * 100 : 0;
  let minThick = thicknesses.length > 0 ? Infinity : 0;
  let maxThick = 0;
  for (const v of thicknesses) {
    if (v < minThick) minThick = v;
    if (v > maxThick) maxThick = v;
  }

  // Compute bilayer period when in alternating mode with >= 2 layers
  const bilayerPeriod =
    detectionMode === 'alternating' && activeLayers.length >= 2
      ? Number(((totalThickness / activeLayers.length) * 2).toFixed(2))
      : null;

  if (activeLayers.length === 0) {
    warnings.push(
      '当前灵敏度下未检出完整的中心分层：请调高「界面检测灵敏度」、切换「检测模式（明暗交替/暗缝分隔）」，或微调中心测量带位置。'
    );
  } else if (activeLayers.length === 1) {
    warnings.push(
      '仅检出 1 层（2 条界面）：若中心存在更薄子层，请适当调高「界面检测灵敏度」并减小「最小单层厚度 (px)」。'
    );
  }

  // Build 1D profileChart data (subsampled gently if > 320 samples so SVG renders crisply)
  const chartStep = Math.max(1, Math.floor(sampleCount / 280));
  const profilePositions: number[] = [];
  const profileInts: number[] = [];
  for (let k = 0; k < sampleCount; k += chartStep) {
    profilePositions.push(Number((k * pxSize).toFixed(2)));
    profileInts.push(Number(smoothP[k].toFixed(1)));
  }
  const edgePositions = boundaryIndices.map((e) => Number((e * pxSize).toFixed(2)));

  // Independent 2D FFT dominant spatial period verification (noise-immune spectral cross-check)
  const fftInfo = compute2DFFTDominantPeriod(gray, width, cropRows, minLayerPx);
  const fftPeriodPhys = fftInfo.valid ? Number((fftInfo.periodPx * pxSize).toFixed(2)) : null;
  const fftSinglePhys = fftInfo.valid ? Number((fftInfo.singleLayerPx * pxSize).toFixed(2)) : null;

  const modeLabelZh: Record<string, string> = {
    gradient: '阶跃界面法',
    alternating: '明暗交替法',
    dark_seams: '暗缝分隔法',
    bright_seams: '亮缝分隔法',
  };
  const geomTagZh = geometryMode === 'curved' ? '弯曲沿层追踪' : isHorizLayers ? `水平分层 (倾角 ${tiltDeg}°)` : `垂直分层 (倾角 ${tiltDeg}°)`;
  const geomTagEn = geometryMode === 'curved' ? 'Curved streamline' : isHorizLayers ? `Horizontal (${tiltDeg}°)` : `Vertical (${tiltDeg}°)`;

  const summary: SummaryMetric[] = [
    {
      key: 'layer_count',
      labelZh: '中心层数',
      labelEn: 'Center Layer Count',
      value: activeLayers.length,
      unit: 'layers',
      detailZh: `${boundaryIndices.length} 条界面 · ${geomTagZh}`,
      detailEn: `${boundaryIndices.length} interfaces · ${geomTagEn}`,
      highlight: true,
    },
    {
      key: 'total_thickness',
      labelZh: '中心多层总厚',
      labelEn: 'Total Stack Thickness',
      value: Number(totalThickness.toFixed(2)),
      unit,
      detailZh: modeLabelZh[detectionMode] ?? detectionMode,
      detailEn: detectionMode,
      highlight: true,
    },
    {
      key: 'mean_layer_thickness',
      labelZh: '平均单层厚度',
      labelEn: 'Mean Layer Thickness',
      value: activeLayers.length > 0 ? Number(meanThick.toFixed(2)) : 0,
      unit,
      detailZh: `SD ${sdThick.toFixed(2)} ${unit} · CV ${cvPct.toFixed(1)}%`,
      detailEn: `SD ${sdThick.toFixed(2)} ${unit} · CV ${cvPct.toFixed(1)}%`,
      highlight: true,
    },
    fftPeriodPhys !== null
      ? {
          key: 'fft_period',
          labelZh: '2D FFT 主周期厚',
          labelEn: '2D FFT Dominant Period',
          value: fftPeriodPhys,
          unit,
          detailZh: `估测单层 ≈ ${fftSinglePhys} ${unit} · 频谱 SNR ${fftInfo.snr}`,
          detailEn: `Est. single ≈ ${fftSinglePhys} ${unit} · SNR ${fftInfo.snr}`,
        }
      : bilayerPeriod !== null
      ? {
          key: 'bilayer_period',
          labelZh: '双层周期厚度',
          labelEn: 'Bilayer Period',
          value: bilayerPeriod,
          unit,
          detailZh: `单层范围 ${minThick.toFixed(1)}–${maxThick.toFixed(1)} ${unit}`,
          detailEn: `Range ${minThick.toFixed(1)}–${maxThick.toFixed(1)} ${unit}`,
        }
      : {
          key: 'thickness_range',
          labelZh: '单层厚度范围',
          labelEn: 'Thickness Min–Max',
          value:
            activeLayers.length > 0
              ? `${minThick.toFixed(1)} – ${maxThick.toFixed(1)}`
              : '—',
          unit,
          detailZh: `中心带宽 ${roiWidthPct}%`,
          detailEn: `Center ROI ${roiWidthPct}%`,
        },
  ];

  const summaryMap: Record<string, number | string | null> = {
    layer_count: activeLayers.length,
    interface_count: boundaryIndices.length,
    total_thickness: Number(totalThickness.toFixed(2)),
    mean_layer_thickness: activeLayers.length > 0 ? Number(meanThick.toFixed(2)) : 0,
    layer_thickness_sd: Number(sdThick.toFixed(2)),
    layer_thickness_cv: Number(cvPct.toFixed(2)),
    min_layer_thickness: Number(minThick.toFixed(2)),
    max_layer_thickness: Number(maxThick.toFixed(2)),
    fft_period: fftPeriodPhys,
    fft_single_layer: fftSinglePhys,
    tilt_deg: tiltDeg,
  };

  return {
    objects,
    shapes,
    summary,
    summaryMap,
    warnings,
    histogram:
      thicknesses.length >= 2
        ? {
            labelZh: '各层厚度分布',
            labelEn: 'Layer Thickness Distribution',
            unit,
            values: thicknesses,
            mean: meanThick,
          }
        : undefined,
    profileChart: {
      positions: profilePositions,
      intensities: profileInts,
      edges: edgePositions,
      unit,
    },
    qcLabel: `${activeLayers.length} layers · total ${totalThickness.toFixed(1)} ${unit}`,
  };
}
