import { traceComponentPixelContours } from './contourTrace';

export interface SeparatedParticleComponent {
  id: number;
  areaPx: number;
  cx: number;
  cy: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  eqDiameterPx: number;
  circularity: number;
  solidity: number; // exact 2D convex-hull solidity (area / convexHullArea)
  aspectRatio: number;
  meanGray: number;
  rimGrayP90: number;
  touchingBorder: boolean;
  occluded: boolean;
  occlusionScore: number;
  overlapConfidence: number; // 0..1 combined curvature + neighbor overlap confidence
  concaveCuspCount: number; // detected high-negative-curvature overlap pinch cusps
  boundaryFitScore: number; // 0..1 post-optimization boundary fit quality
  occludedReason?: string;
  contactRatio: number;
  recessedContactRatio: number;
  contour: Array<[number, number]>;
  holeContours: Array<Array<[number, number]>>;
}

export type EdgeRefinementMode =
  | 'off'
  | 'sobel_3x3'
  | 'sobel_5x5'
  | 'canny_mild'
  | 'canny_strong';

export interface EdgeRefinementField {
  mode: Exclude<EdgeRefinementMode, 'off'>;
  gx: Float32Array;
  gy: Float32Array;
  mag: Float32Array;
  /** 0..1 normalized edge map (Sobel gradient magnitude or Canny NMS + hysteresis ridge). */
  edgeResponse: Float32Array;
  /** 1.0 on Canny hysteresis edge pixels, 0 elsewhere (always 0 for Sobel modes). */
  cannyMask: Float32Array;
}

export interface SemEdgeWatershedOptions {
  isBright: boolean;
  minDiameterPx: number;
  edgeBarrierStrength: number; // 0..1
  splitSensitivity: number; // 0.15..0.95
  occlusionSensitivity: number; // 0.1..0.95
  excludeBorder: boolean;
  curvatureBoundaryFit?: boolean;
  overlapConfidenceThreshold?: number;
  edgeRefinement?: EdgeRefinementMode;
  probMap?: Float32Array;
  probThreshold?: number;
}

/** Exact O(W*H) 2D Euclidean Distance Transform to nearest 0-pixel (Felzenszwalb-Huttenlocher). */
function computeEDT(binary: Uint8Array, w: number, h: number): Float32Array {
  const INF = 1e12;
  const maxDim = Math.max(w, h);
  const f = new Float64Array(maxDim);
  const v = new Int32Array(maxDim);
  const z = new Float64Array(maxDim + 1);
  const tmp = new Float64Array(w * h);
  const dist = new Float32Array(w * h);

  const dt1d = (len: number, outArr: Float64Array, outOffset: number, outStride: number) => {
    let k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    for (let q = 1; q < len; q++) {
      let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (s <= z[k]) {
        k--;
        s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k + 1] = INF;
    }
    k = 0;
    for (let q = 0; q < len; q++) {
      while (z[k + 1] < q) k++;
      const d = q - v[k];
      outArr[outOffset + q * outStride] = d * d + f[v[k]];
    }
  };

  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = binary[y * w + x] ? INF : 0;
    dt1d(h, tmp, x, w);
  }
  const rowOut = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = 0; x < w; x++) f[x] = tmp[off + x];
    dt1d(w, rowOut, 0, 1);
    for (let x = 0; x < w; x++) dist[off + x] = Math.sqrt(rowOut[x]);
  }
  return dist;
}

/** Fast 5-tap separable Gaussian blur (~sigma 1.0). */
function smooth5Tap(src: Float32Array, w: number, h: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = 0; x < w; x++) {
      const xm2 = x >= 2 ? x - 2 : 0;
      const xm1 = x >= 1 ? x - 1 : 0;
      const xp1 = x + 1 < w ? x + 1 : w - 1;
      const xp2 = x + 2 < w ? x + 2 : w - 1;
      tmp[off + x] =
        (src[off + xm2] +
          4 * src[off + xm1] +
          6 * src[off + x] +
          4 * src[off + xp1] +
          src[off + xp2]) *
        0.0625;
    }
  }
  for (let y = 0; y < h; y++) {
    const ym2 = (y >= 2 ? y - 2 : 0) * w;
    const ym1 = (y >= 1 ? y - 1 : 0) * w;
    const y0 = y * w;
    const yp1 = (y + 1 < h ? y + 1 : h - 1) * w;
    const yp2 = (y + 2 < h ? y + 2 : h - 1) * w;
    for (let x = 0; x < w; x++) {
      out[y0 + x] =
        (tmp[ym2 + x] + 4 * tmp[ym1 + x] + 6 * tmp[y0 + x] + 4 * tmp[yp1 + x] + tmp[yp2 + x]) *
        0.0625;
    }
  }
  return out;
}

/** O(W*H) integral-image local box mean with half-window radius r. */
function computeLocalBoxMean(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const iw = w + 1;
  const integral = new Float64Array(iw * (h + 1));
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    const srcOff = y * w;
    const prevOff = y * iw;
    const curOff = (y + 1) * iw;
    for (let x = 0; x < w; x++) {
      rowSum += src[srcOff + x];
      integral[curOff + x + 1] = integral[prevOff + x + 1] + rowSum;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h - 1, y + r);
    const row0 = y0 * iw;
    const row1 = (y1 + 1) * iw;
    const dstOff = y * w;
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      const sum =
        integral[row1 + x1 + 1] -
        integral[row0 + x1 + 1] -
        integral[row1 + x0] +
        integral[row0 + x0];
      out[dstOff + x] = sum / ((x1 - x0 + 1) * (y1 - y0 + 1));
    }
  }
  return out;
}

function quickOtsu(values: Float32Array, stride = 3): number {
  const bins = 64;
  const hist = new Float64Array(bins);
  let total = 0;
  for (let i = 0; i < values.length; i += stride) {
    const b = Math.max(0, Math.min(bins - 1, Math.floor((values[i] / 255) * bins)));
    hist[b]++;
    total++;
  }
  if (total < 2) return 128;
  let sumAll = 0;
  for (let i = 0; i < bins; i++) sumAll += i * hist[i];
  let sumB = 0;
  let wB = 0;
  let maxBetween = -1;
  let bestBin = 32;
  for (let i = 0; i < bins; i++) {
    wB += hist[i];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += i * hist[i];
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const between = wB * wF * (mB - mF) ** 2;
    if (between > maxBetween) {
      maxBetween = between;
      bestBin = i;
    }
  }
  return ((bestBin + 0.5) / bins) * 255;
}

function approxPercentile(values: Float32Array, p: number, stride = 4): number {
  const sample: number[] = [];
  for (let i = 0; i < values.length; i += stride) sample.push(values[i]);
  if (sample.length === 0) return 0;
  sample.sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sample.length - 1, Math.floor(p * (sample.length - 1))));
  return sample[idx];
}

/**
 * Shared Sobel (3x3 / 5x5) and Canny (NMS + double-threshold hysteresis) edge refinement engine.
 * Used by both `sem.particles` and `sem.lines` for pixel-precise localization on low-contrast boundaries.
 */
export function computeEdgeRefinementField(
  gray: Float32Array,
  w: number,
  h: number,
  mode: EdgeRefinementMode
): EdgeRefinementField | null {
  if (mode === 'off' || w < 5 || h < 5) return null;

  const n = w * h;
  const gx = new Float32Array(n);
  const gy = new Float32Array(n);
  const mag = new Float32Array(n);
  const edgeResponse = new Float32Array(n);
  const cannyMask = new Float32Array(n);

  const isCanny = mode === 'canny_mild' || mode === 'canny_strong';
  const use5x5 = mode === 'sobel_5x5' || mode === 'canny_strong';
  const src = isCanny ? smooth5Tap(gray, w, h) : gray;

  if (use5x5) {
    // Separable 5x5 Sobel: smooth [1, 4, 6, 4, 1] x diff [-1, -2, 0, 2, 1], normalized by 48
    const inv48 = 1 / 48;
    for (let y = 2; y < h - 2; y++) {
      const rM2 = (y - 2) * w;
      const rM1 = (y - 1) * w;
      const r0 = y * w;
      const rP1 = (y + 1) * w;
      const rP2 = (y + 2) * w;
      for (let x = 2; x < w - 2; x++) {
        const dxM2 = -src[rM2 + x - 2] - 2 * src[rM2 + x - 1] + 2 * src[rM2 + x + 1] + src[rM2 + x + 2];
        const dxM1 = -src[rM1 + x - 2] - 2 * src[rM1 + x - 1] + 2 * src[rM1 + x + 1] + src[rM1 + x + 2];
        const dx0 = -src[r0 + x - 2] - 2 * src[r0 + x - 1] + 2 * src[r0 + x + 1] + src[r0 + x + 2];
        const dxP1 = -src[rP1 + x - 2] - 2 * src[rP1 + x - 1] + 2 * src[rP1 + x + 1] + src[rP1 + x + 2];
        const dxP2 = -src[rP2 + x - 2] - 2 * src[rP2 + x - 1] + 2 * src[rP2 + x + 1] + src[rP2 + x + 2];
        const vx = (dxM2 + 4 * dxM1 + 6 * dx0 + 4 * dxP1 + dxP2) * inv48;

        const syM2 = src[rM2 + x - 2] + 4 * src[rM2 + x - 1] + 6 * src[rM2 + x] + 4 * src[rM2 + x + 1] + src[rM2 + x + 2];
        const syM1 = src[rM1 + x - 2] + 4 * src[rM1 + x - 1] + 6 * src[rM1 + x] + 4 * src[rM1 + x + 1] + src[rM1 + x + 2];
        const syP1 = src[rP1 + x - 2] + 4 * src[rP1 + x - 1] + 6 * src[rP1 + x] + 4 * src[rP1 + x + 1] + src[rP1 + x + 2];
        const syP2 = src[rP2 + x - 2] + 4 * src[rP2 + x - 1] + 6 * src[rP2 + x] + 4 * src[rP2 + x + 1] + src[rP2 + x + 2];
        const vy = (-syM2 - 2 * syM1 + 2 * syP1 + syP2) * inv48;

        const idx = r0 + x;
        gx[idx] = vx;
        gy[idx] = vy;
        mag[idx] = Math.hypot(vx, vy);
      }
    }
  } else {
    // Standard 3x3 Sobel normalized by 4
    for (let y = 1; y < h - 1; y++) {
      const rM1 = (y - 1) * w;
      const r0 = y * w;
      const rP1 = (y + 1) * w;
      for (let x = 1; x < w - 1; x++) {
        const vx =
          (-src[rM1 + x - 1] +
            src[rM1 + x + 1] -
            2 * src[r0 + x - 1] +
            2 * src[r0 + x + 1] -
            src[rP1 + x - 1] +
            src[rP1 + x + 1]) *
          0.25;
        const vy =
          (-src[rM1 + x - 1] -
            2 * src[rM1 + x] -
            src[rM1 + x + 1] +
            src[rP1 + x - 1] +
            2 * src[rP1 + x] +
            src[rP1 + x + 1]) *
          0.25;
        const idx = r0 + x;
        gx[idx] = vx;
        gy[idx] = vy;
        mag[idx] = Math.hypot(vx, vy);
      }
    }
  }

  const magP96 = Math.max(4, approxPercentile(mag, 0.96, 4));

  if (!isCanny) {
    for (let i = 0; i < n; i++) {
      edgeResponse[i] = Math.min(1, mag[i] / magP96);
    }
    return { mode, gx, gy, mag, edgeResponse, cannyMask };
  }

  // Canny Step 1: 4-direction Non-Maximum Suppression (NMS)
  const nms = new Float32Array(n);
  const nmsSample: number[] = [];
  for (let y = 2; y < h - 2; y++) {
    const row = y * w;
    for (let x = 2; x < w - 2; x++) {
      const idx = row + x;
      const m = mag[idx];
      if (m <= 1e-4) continue;
      let deg = (Math.atan2(gy[idx], gx[idx]) * 180) / Math.PI;
      if (deg < 0) deg += 180;

      let m1 = 0;
      let m2 = 0;
      if (deg < 22.5 || deg >= 157.5) {
        m1 = mag[idx - 1];
        m2 = mag[idx + 1];
      } else if (deg < 67.5) {
        m1 = mag[idx - w - 1];
        m2 = mag[idx + w + 1];
      } else if (deg < 112.5) {
        m1 = mag[idx - w];
        m2 = mag[idx + w];
      } else {
        m1 = mag[idx - w + 1];
        m2 = mag[idx + w - 1];
      }

      if (m >= m1 && m >= m2) {
        nms[idx] = m;
        if (((x + y) & 3) === 0) nmsSample.push(m);
      }
    }
  }

  // Canny Step 2: Adaptive double-threshold hysteresis edge tracking
  nmsSample.sort((a, b) => a - b);
  const pct = (p: number) =>
    nmsSample.length > 0
      ? nmsSample[Math.max(0, Math.min(nmsSample.length - 1, Math.floor(p * (nmsSample.length - 1))))]
      : magP96 * 0.5;

  const tHigh =
    mode === 'canny_strong'
      ? Math.max(2.0, pct(0.76))
      : Math.max(3.5, pct(0.88));
  const tLow = tHigh * (mode === 'canny_strong' ? 0.3 : 0.42);

  // state: 0 = suppressed, 1 = weak candidate, 2 = confirmed hysteresis edge
  const state = new Uint8Array(n);
  const queue = new Int32Array(n);
  let head = 0;
  let tail = 0;
  for (let i = 0; i < n; i++) {
    const v = nms[i];
    if (v >= tHigh) {
      state[i] = 2;
      queue[tail++] = i;
    } else if (v >= tLow) {
      state[i] = 1;
    }
  }

  while (head < tail) {
    const idx = queue[head++];
    const px = idx % w;
    const py = (idx - px) / w;
    for (let dy = -1; dy <= 1; dy++) {
      const ny = py + dy;
      if (ny <= 0 || ny >= h - 1) continue;
      const nRow = ny * w;
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = px + dx;
        if (nx <= 0 || nx >= w - 1) continue;
        const nIdx = nRow + nx;
        if (state[nIdx] === 1) {
          state[nIdx] = 2;
          queue[tail++] = nIdx;
        }
      }
    }
  }

  for (let i = 0; i < n; i++) {
    const normG = Math.min(1, mag[i] / magP96);
    if (state[i] === 2) {
      cannyMask[i] = 1;
      edgeResponse[i] = Math.min(1, 0.58 + 0.42 * normG);
    } else {
      edgeResponse[i] = normG * 0.22;
    }
  }

  return { mode, gx, gy, mag, edgeResponse, cannyMask };
}

/** Exact 2D convex hull (Andrew's monotone chain) in O(N log N). */
export function computeConvexHull(pts: Array<[number, number]>): Array<[number, number]> {
  if (pts.length <= 3) return pts.slice();
  const sorted = pts.slice().sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]));
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  const lower: Array<[number, number]> = [];
  for (let i = 0; i < sorted.length; i++) {
    const p = sorted[i];
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper: Array<[number, number]> = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

export function polygonSignedArea(poly: Array<[number, number]>): number {
  if (poly.length < 3) return 0;
  let acc = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    acc += x1 * y2 - x2 * y1;
  }
  return acc * 0.5;
}

export function polygonArea(poly: Array<[number, number]>): number {
  return Math.abs(polygonSignedArea(poly));
}

/** Distance from point (px, py) to nearest edge of a convex hull polygon. */
function pointToHullEdgeDistance(px: number, py: number, hull: Array<[number, number]>): number {
  const n = hull.length;
  if (n < 3) return 0;
  let minDistSq = Infinity;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = hull[i];
    const [x2, y2] = hull[(i + 1) % n];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    let t = 0;
    if (lenSq > 1e-9) {
      t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq));
    }
    const cx = x1 + t * dx;
    const cy = y1 + t * dy;
    const dSq = (px - cx) * (px - cx) + (py - cy) * (py - cy);
    if (dSq < minDistSq) minDistSq = dSq;
  }
  return Math.sqrt(minDistSq);
}

/**
 * Compute per-vertex signed curvature kappa_i (rad/px) along a closed contour,
 * normalized so positive = convex outward turn and negative = concave inward pinch/overlap.
 */
function computeSignedContourCurvature(contour: Array<[number, number]>): Float32Array {
  const m = contour.length;
  const kappa = new Float32Array(m);
  if (m < 8) return kappa;

  // In image coordinates (x right, y down), positive signed shoelace area corresponds to clockwise screen orientation.
  // Align sign so that outward convex turns always yield positive curvature and inward dents yield negative curvature.
  const orientSign = polygonSignedArea(contour) >= 0 ? 1 : -1;
  const stride = Math.max(2, Math.min(6, Math.floor(m / 18)));

  for (let i = 0; i < m; i++) {
    const pPrev = contour[(i - stride + m) % m];
    const pCurr = contour[i];
    const pNext = contour[(i + stride) % m];
    const v1x = pCurr[0] - pPrev[0];
    const v1y = pCurr[1] - pPrev[1];
    const v2x = pNext[0] - pCurr[0];
    const v2y = pNext[1] - pCurr[1];
    const l1 = Math.hypot(v1x, v1y);
    const l2 = Math.hypot(v2x, v2y);
    if (l1 < 1e-4 || l2 < 1e-4) continue;

    const cross = v1x * v2y - v1y * v2x;
    const dot = v1x * v2x + v1y * v2y;
    const turnAngle = Math.atan2(cross, dot) * orientSign;
    kappa[i] = turnAngle / (0.5 * (l1 + l2));
  }
  return kappa;
}

/**
 * Curvature- & Overlap-Confidence-Guided Boundary Fitting Optimization.
 *
 * - Analyzes per-vertex signed curvature (kappa) and local neighbor overlap confidence.
 * - Counts sharp concave pinch cusps (overlap junctions where upper particles cut into lower ones).
 * - Along high-overlap-confidence seams: applies tangential regularization + sub-pixel normal snapping
 *   onto the maximum SEM gradient/rim edge (within +-1.15 px) to eliminate jagged seam bleed.
 * - Along free outer boundaries: applies corner-preserving curvature-weighted fitting so sharp
 *   crystalline vertices remain crisp while curved spherical rims fit smoothly.
 */
function analyzeAndOptimizeBoundaryFit(
  contour: Array<[number, number]>,
  hull: Array<[number, number]>,
  labels: Int32Array,
  gray: Float32Array,
  w: number,
  h: number,
  selfLabel: number,
  validLabel: Uint8Array,
  meanGrayArr: Float32Array,
  rimP90Arr: Float32Array,
  recessDepthThresh: number,
  enableOptimization: boolean,
  edgeField?: EdgeRefinementField | null
): {
  optimizedContour: Array<[number, number]>;
  overlapConfidence: number;
  concaveCuspCount: number;
  concaveArcRatio: number;
  boundaryFitScore: number;
} {
  const m = contour.length;
  if (m < 8) {
    return {
      optimizedContour: contour,
      overlapConfidence: 0,
      concaveCuspCount: 0,
      concaveArcRatio: 0,
      boundaryFitScore: 0.9,
    };
  }

  const kappa = computeSignedContourCurvature(contour);
  const vertexOverlap = new Float32Array(m);

  const probeOffsets: Array<[number, number]> = [
    [-1, 0], [1, 0], [0, -1], [0, 1],
    [-2, 0], [2, 0], [0, -2], [0, 2],
    [-2, -2], [2, -2], [-2, 2], [2, 2],
  ];

  const selfMean = meanGrayArr[selfLabel];
  const selfRim = rimP90Arr[selfLabel];
  let concaveContactVerts = 0;
  let overlapSum = 0;
  let overlapContactVerts = 0;

  for (let i = 0; i < m; i++) {
    const [vx, vy] = contour[i];
    const ix = Math.max(0, Math.min(w - 1, Math.round(vx)));
    const iy = Math.max(0, Math.min(h - 1, Math.round(vy)));

    let neighborLb = 0;
    let neighborSeamGray = 0;
    for (let p = 0; p < probeOffsets.length; p++) {
      const nx = ix + probeOffsets[p][0];
      const ny = iy + probeOffsets[p][1];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const nl = labels[ny * w + nx];
      if (nl > 0 && nl !== selfLabel && validLabel[nl]) {
        neighborLb = nl;
        neighborSeamGray = gray[ny * w + nx];
        break;
      }
    }

    const negCurv = Math.max(0, -kappa[i]);
    const dHull = pointToHullEdgeDistance(vx, vy, hull);
    const hullNorm = Math.min(1, dHull / Math.max(1.5, recessDepthThresh * 1.4));
    const curvNorm = Math.min(1, negCurv / 0.11);

    let conf = 0;
    if (neighborLb > 0) {
      const brighterCue =
        rimP90Arr[neighborLb] > selfRim + 5 || meanGrayArr[neighborLb] > selfMean + 8 ? 1 : 0;
      const shadowCue = neighborSeamGray > gray[iy * w + ix] + 10 ? 1 : 0;
      conf = Math.min(
        1,
        0.34 + 0.28 * hullNorm + 0.22 * curvNorm + 0.1 * brighterCue + 0.06 * shadowCue
      );
      overlapContactVerts++;
      overlapSum += conf;
      if (negCurv > 0.035 || dHull >= recessDepthThresh) {
        concaveContactVerts++;
      }
    } else if (negCurv > 0.06 && dHull >= recessDepthThresh) {
      // Strong concave notch even across a wider dark shadow gap
      conf = Math.min(0.65, 0.22 + 0.25 * curvNorm + 0.18 * hullNorm);
    }
    vertexOverlap[i] = conf;
  }

  // Detect distinct concave pinch cusps (local minima of signed curvature kappa_i < -0.065)
  let concaveCuspCount = 0;
  const cuspMinGap = Math.max(3, Math.floor(m / 24));
  let lastCuspIdx = -cuspMinGap;
  for (let i = 0; i < m; i++) {
    const ki = kappa[i];
    if (ki > -0.065) continue;
    const prevK = kappa[(i - 1 + m) % m];
    const nextK = kappa[(i + 1) % m];
    if (ki <= prevK && ki <= nextK && i - lastCuspIdx >= cuspMinGap) {
      if (vertexOverlap[i] >= 0.28) {
        concaveCuspCount++;
        lastCuspIdx = i;
      }
    }
  }

  const concaveArcRatio = concaveContactVerts / m;
  const contactFraction = overlapContactVerts / m;
  const meanContactConf = overlapContactVerts > 0 ? overlapSum / overlapContactVerts : 0;
  const overlapConfidence = Math.max(
    0,
    Math.min(
      1,
      meanContactConf * Math.min(1, contactFraction / 0.18) * 0.65 +
        Math.min(1, concaveArcRatio / 0.14) * 0.25 +
        Math.min(1, concaveCuspCount / 2) * 0.1
    )
  );

  if (!enableOptimization && !edgeField) {
    return {
      optimizedContour: contour,
      overlapConfidence: Number(overlapConfidence.toFixed(3)),
      concaveCuspCount,
      concaveArcRatio: Number(concaveArcRatio.toFixed(3)),
      boundaryFitScore: Number(Math.max(0.55, 0.92 - overlapConfidence * 0.25).toFixed(3)),
    };
  }

  // Helper: bilinearly sampled refined edge energy (Sobel/Canny edgeResponse when enabled, else raw gradient)
  const sampleBilinearField = (field: Float32Array, px: number, py: number): number => {
    const x = Math.max(0, Math.min(w - 1.001, px));
    const y = Math.max(0, Math.min(h - 1.001, py));
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    const idx = y0 * w + x0;
    const v00 = field[idx];
    const v10 = field[idx + 1];
    const v01 = field[idx + w];
    const v11 = field[idx + w + 1];
    return (
      v00 * (1 - fx) * (1 - fy) +
      v10 * fx * (1 - fy) +
      v01 * (1 - fx) * fy +
      v11 * fx * fy
    );
  };

  const sampleGradMag = (px: number, py: number): number => {
    if (edgeField) {
      return sampleBilinearField(edgeField.edgeResponse, px, py);
    }
    const x = Math.max(1, Math.min(w - 2, Math.round(px)));
    const y = Math.max(1, Math.min(h - 2, Math.round(py)));
    const idx = y * w + x;
    const gx = gray[idx + 1] - gray[idx - 1];
    const gy = gray[idx + w] - gray[idx - w];
    return Math.hypot(gx, gy);
  };

  // Pass 1: Curvature- & overlap-weighted tangential regularization (preserves sharp crystal corners)
  const fitted: Array<[number, number]> = contour.map(([x, y]) => [x, y]);
  if (enableOptimization) {
    for (let pass = 0; pass < 2; pass++) {
      const src = fitted.map(([x, y]) => [x, y] as [number, number]);
      for (let i = 0; i < m; i++) {
        const pPrev = src[(i - 1 + m) % m];
        const pCurr = src[i];
        const pNext = src[(i + 1) % m];
        const ov = vertexOverlap[i];
        const absK = Math.abs(kappa[i]);

        // High-curvature sharp corners (|kappa| > 0.18) stay anchored;
        // overlapping seams receive stronger tangential tension to clean jagged contact noise.
        const cornerPreserve = Math.exp(-Math.pow(absK / 0.18, 2));
        const weight = ov >= 0.45 ? 0.34 : 0.22 * cornerPreserve;
        const mx = 0.5 * (pPrev[0] + pNext[0]);
        const my = 0.5 * (pPrev[1] + pNext[1]);
        fitted[i] = [
          pCurr[0] + (mx - pCurr[0]) * weight,
          pCurr[1] + (my - pCurr[1]) * weight,
        ];
      }
    }
  }

  // Pass 2: Sub-pixel normal snapping onto peak edge gradient (Sobel/Canny snaps all low-contrast boundaries)
  let gradAlignmentAcc = 0;
  const isCannyMode = edgeField && (edgeField.mode === 'canny_mild' || edgeField.mode === 'canny_strong');
  const refinedSteps = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5];

  for (let i = 0; i < m; i++) {
    const pPrev = fitted[(i - 1 + m) % m];
    const pCurr = fitted[i];
    const pNext = fitted[(i + 1) % m];
    const tx = pNext[0] - pPrev[0];
    const ty = pNext[1] - pPrev[1];
    const tLen = Math.hypot(tx, ty);
    if (tLen < 1e-4) continue;
    const nx = -ty / tLen;
    const ny = tx / tLen;

    const ov = vertexOverlap[i];
    const baseG = sampleGradMag(pCurr[0], pCurr[1]);
    let bestG = baseG;
    let bestShift = 0;

    if (edgeField) {
      const vals = refinedSteps.map((s) => sampleGradMag(pCurr[0] + nx * s, pCurr[1] + ny * s));
      let bestIdx = 3; // center (s = 0)
      for (let k = 0; k < vals.length; k++) {
        if (vals[k] > bestG * 1.04) {
          bestG = vals[k];
          bestIdx = k;
        }
      }
      if (bestIdx !== 3 && bestG > 0.08) {
        let subShift = refinedSteps[bestIdx];
        if (bestIdx > 0 && bestIdx < vals.length - 1) {
          const fPrev = vals[bestIdx - 1];
          const fCurr = vals[bestIdx];
          const fNext = vals[bestIdx + 1];
          const denom = fPrev - 2 * fCurr + fNext;
          if (denom < -1e-6) {
            const delta = Math.max(-0.5, Math.min(0.5, (0.5 * (fPrev - fNext)) / denom));
            subShift += delta * 0.5;
          }
        }
        const cornerDamp = Math.abs(kappa[i]) > 0.22 ? 0.55 : 1.0;
        const pull = (isCannyMode ? 0.85 : 0.72) * cornerDamp;
        fitted[i] = [
          Math.max(0, Math.min(w, pCurr[0] + nx * subShift * pull)),
          Math.max(0, Math.min(h, pCurr[1] + ny * subShift * pull)),
        ];
      }
      gradAlignmentAcc += Math.min(1, bestG / 0.65);
    } else {
      if (ov >= 0.38) {
        const maxShift = 1.1;
        for (const s of [-maxShift, -0.55, 0.55, maxShift]) {
          const candG = sampleGradMag(pCurr[0] + nx * s, pCurr[1] + ny * s);
          if (candG > bestG * 1.06) {
            bestG = candG;
            bestShift = s;
          }
        }
        if (bestShift !== 0) {
          fitted[i] = [
            Math.max(0, Math.min(w, pCurr[0] + nx * bestShift * 0.65)),
            Math.max(0, Math.min(h, pCurr[1] + ny * bestShift * 0.65)),
          ];
        }
      }
      gradAlignmentAcc += Math.min(1, bestG / 38);
    }
  }

  const meanGradFit = gradAlignmentAcc / m;
  const boundaryFitScore = Number(
    Math.max(0.5, Math.min(0.99, 0.62 + 0.34 * meanGradFit - 0.1 * overlapConfidence)).toFixed(3)
  );

  return {
    optimizedContour: fitted,
    overlapConfidence: Number(overlapConfidence.toFixed(3)),
    concaveCuspCount,
    concaveArcRatio: Number(concaveArcRatio.toFixed(3)),
    boundaryFitScore,
  };
}

/** Compact typed-array binary min-heap for fast O(N log N) priority-queue watershed. */
class PixelMinHeap {
  private keys: Float32Array;
  private vals: Int32Array;
  public size = 0;

  constructor(capacity: number) {
    this.keys = new Float32Array(Math.max(1024, capacity));
    this.vals = new Int32Array(Math.max(1024, capacity));
  }

  private grow() {
    const nextCap = this.keys.length * 2;
    const nk = new Float32Array(nextCap);
    const nv = new Int32Array(nextCap);
    nk.set(this.keys);
    nv.set(this.vals);
    this.keys = nk;
    this.vals = nv;
  }

  push(key: number, val: number) {
    if (this.size >= this.keys.length) this.grow();
    let i = this.size++;
    this.keys[i] = key;
    this.vals[i] = val;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.keys[parent] <= this.keys[i]) break;
      const tk = this.keys[parent];
      const tv = this.vals[parent];
      this.keys[parent] = this.keys[i];
      this.vals[parent] = this.vals[i];
      this.keys[i] = tk;
      this.vals[i] = tv;
      i = parent;
    }
  }

  pop(): number {
    const top = this.vals[0];
    this.size--;
    if (this.size > 0) {
      this.keys[0] = this.keys[this.size];
      this.vals[0] = this.vals[this.size];
      let i = 0;
      const n = this.size;
      while (true) {
        const l = (i << 1) + 1;
        const r = l + 1;
        let smallest = i;
        if (l < n && this.keys[l] < this.keys[smallest]) smallest = l;
        if (r < n && this.keys[r] < this.keys[smallest]) smallest = r;
        if (smallest === i) break;
        const tk = this.keys[i];
        const tv = this.vals[i];
        this.keys[i] = this.keys[smallest];
        this.vals[i] = this.vals[smallest];
        this.keys[smallest] = tk;
        this.vals[smallest] = tv;
        i = smallest;
      }
    }
    return top;
  }
}

/** Fast LUT-accelerated 5x5 edge-preserving bilateral filter (flattens SEM shot noise while keeping bright rims & crevices razor-sharp). */
function bilateralFilterFast(src: Float32Array, w: number, h: number, sigmaRange = 24): Float32Array {
  const out = new Float32Array(w * h);
  const rangeLut = new Float32Array(256);
  const inv2sr2 = 1 / (2 * sigmaRange * sigmaRange);
  for (let d = 0; d < 256; d++) {
    rangeLut[d] = Math.exp(-(d * d) * inv2sr2);
  }
  // 5x5 sparse star/ring offsets (13 taps: center + 4-ring + 8-ring) for high speed & isotropic response
  const offsets: Array<[number, number, number]> = [
    [0, 0, 1.0],
    [-1, 0, 0.85], [1, 0, 0.85], [0, -1, 0.85], [0, 1, 0.85],
    [-1, -1, 0.72], [1, -1, 0.72], [-1, 1, 0.72], [1, 1, 0.72],
    [-2, 0, 0.45], [2, 0, 0.45], [0, -2, 0.45], [0, 2, 0.45],
  ];
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const c = src[row + x];
      let wSum = 0;
      let vSum = 0;
      for (let k = 0; k < offsets.length; k++) {
        const nx = x + offsets[k][0];
        const ny = y + offsets[k][1];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const nv = src[ny * w + nx];
        const diff = Math.min(255, Math.abs(nv - c) | 0);
        const wt = offsets[k][2] * rangeLut[diff];
        wSum += wt;
        vSum += nv * wt;
      }
      out[row + x] = wSum > 0 ? vSum / wSum : c;
    }
  }
  return out;
}

/**
 * SEM Bright-Edge & Stacked Particle Separation + Multi-Scale Dipole Barrier + RAG Merging + Curvature Boundary Optimization.
 */
export function segmentSemBrightEdgeParticles(
  gray: Float32Array,
  w: number,
  h: number,
  options: SemEdgeWatershedOptions
): SeparatedParticleComponent[] {
  const n = w * h;
  const {
    isBright,
    minDiameterPx,
    edgeBarrierStrength,
    splitSensitivity,
    occlusionSensitivity,
    excludeBorder,
    curvatureBoundaryFit = true,
    overlapConfidenceThreshold,
    edgeRefinement = 'off',
    probMap,
    probThreshold = 0.5,
  } = options;

  // 1. Polarity-normalized signal (target particles always bright)
  const img = new Float32Array(n);
  if (isBright) {
    for (let i = 0; i < n; i++) img[i] = gray[i];
  } else {
    for (let i = 0; i < n; i++) img[i] = 255 - gray[i];
  }

  // 2. Edge-preserving bilateral pre-filter + coarse scale for multi-scale SE dipole detection
  const smooth = bilateralFilterFast(img, w, h, 24);
  const coarse = smooth5Tap(smooth, w, h);
  const localRadius = Math.max(10, Math.min(42, Math.round(minDiameterPx * 1.1)));
  const localMean = computeLocalBoxMean(smooth, w, h, localRadius);

  // 3. Crevice-aware foreground mask (handles both sparse substrates & densely packed edge-to-edge SEM fields)
  const fgMask = new Uint8Array(n);
  if (probMap) {
    for (let i = 0; i < n; i++) {
      fgMask[i] = probMap[i] >= probThreshold ? 1 : 0;
    }
  } else {
    const tOtsu = quickOtsu(smooth, 3);
    const p05 = approxPercentile(smooth, 0.05, 4);
    const p10 = approxPercentile(smooth, 0.1, 4);
    const p50 = approxPercentile(smooth, 0.5, 4);
    const p90 = approxPercentile(smooth, 0.9, 4);
    const dyn = Math.max(12, p90 - p10);
    const packingRatio = (p50 - p10) / dyn;
    // Densely packed SEM (particles touch edge-to-edge): only exclude true deep shadow crevices,
    // avoiding carving holes into moderately dark tilted crystal faces.
    const creviceThresh =
      packingRatio > 0.34
        ? p05 + 0.26 * Math.max(10, tOtsu - p05)
        : tOtsu;

    const localValleyDrop = Math.max(20, dyn * 0.22);
    for (let i = 0; i < n; i++) {
      const v = smooth[i];
      const isDeepDark = v < creviceThresh;
      const isCastShadow =
        packingRatio > 0.34 &&
        v < localMean[i] - localValleyDrop &&
        v < tOtsu * 0.78;
      fgMask[i] = !isDeepDark && !isCastShadow ? 1 : 0;
    }
  }

  // 4. Multi-scale SE Boundary Dipole Energy:
  //    Fuses Fine+Coarse Scharr Gradient + Multi-Scale Bright SE Rim Ridge (r=2,4) + Dark Crevice Valley (r=2,4)
  const grad = new Float32Array(n);
  const rim = new Float32Array(n);
  const valley = new Float32Array(n);
  for (let y = 4; y < h - 4; y++) {
    const row = y * w;
    for (let x = 4; x < w - 4; x++) {
      const idx = row + x;
      if (!fgMask[idx]) continue;

      // Fine Scharr gradient on bilateral-filtered image
      const gx1 =
        -3 * smooth[idx - w - 1] +
        3 * smooth[idx - w + 1] -
        10 * smooth[idx - 1] +
        10 * smooth[idx + 1] -
        3 * smooth[idx + w - 1] +
        3 * smooth[idx + w + 1];
      const gy1 =
        -3 * smooth[idx - w - 1] -
        10 * smooth[idx - w] -
        3 * smooth[idx - w + 1] +
        3 * smooth[idx + w - 1] +
        10 * smooth[idx + w] +
        3 * smooth[idx + w + 1];
      const gFine = Math.hypot(gx1, gy1) * 0.0625;

      // Coarse gradient (r=2 on smoothed image) captures broader 4-8px SE bright-to-shadow edges
      const gx2 = coarse[idx + 2] - coarse[idx - 2];
      const gy2 = coarse[idx + 2 * w] - coarse[idx - 2 * w];
      const gCoarse = Math.hypot(gx2, gy2) * 0.5;
      grad[idx] = Math.max(gFine, gCoarse * 0.95);

      const c2 = 2 * smooth[idx];
      // Multi-scale bright SE rim ridge (r=2 & r=4)
      const r2H = c2 - smooth[idx - 2] - smooth[idx + 2];
      const r2V = c2 - smooth[idx - 2 * w] - smooth[idx + 2 * w];
      const r2D1 = c2 - smooth[idx - 2 * w - 2] - smooth[idx + 2 * w + 2];
      const r2D2 = c2 - smooth[idx - 2 * w + 2] - smooth[idx + 2 * w - 2];
      const ridge2 = Math.max(0, r2H, r2V, r2D1, r2D2);

      const r4H = c2 - smooth[idx - 4] - smooth[idx + 4];
      const r4V = c2 - smooth[idx - 4 * w] - smooth[idx + 4 * w];
      const r4D1 = c2 - smooth[idx - 3 * w - 3] - smooth[idx + 3 * w + 3];
      const r4D2 = c2 - smooth[idx - 3 * w + 3] - smooth[idx + 3 * w - 3];
      const ridge4 = Math.max(0, r4H, r4V, r4D1, r4D2);

      const brightExcess = Math.max(0, smooth[idx] - localMean[idx]);
      rim[idx] = Math.max(ridge2, ridge4 * 0.85) * 0.62 + brightExcess * 0.38;

      // Multi-scale dark seam/crevice valley (r=2 & r=4) — catches thin dark contact lines between touching particles
      const val2 = Math.max(0, -r2H, -r2V, -r2D1, -r2D2);
      const val4 = Math.max(0, -r4H, -r4V, -r4D1, -r4D2);
      const darkDeficit = Math.max(0, localMean[idx] - smooth[idx]);
      valley[idx] = Math.max(val2, val4 * 0.85) * 0.65 + darkDeficit * 0.35;
    }
  }

  const gradP95 = Math.max(8, approxPercentile(grad, 0.95, 4));
  const rimP95 = Math.max(8, approxPercentile(rim, 0.95, 4));
  const valP95 = Math.max(8, approxPercentile(valley, 0.95, 4));

  const edgeField =
    edgeRefinement !== 'off'
      ? computeEdgeRefinementField(smooth, w, h, edgeRefinement)
      : null;

  const rawBarrier = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (!fgMask[i]) {
      rawBarrier[i] = 1;
      continue;
    }
    const gN = Math.min(1, grad[i] / gradP95);
    const rN = Math.min(1, rim[i] / rimP95);
    const vN = Math.min(1, valley[i] / valP95);
    // SE dipole: strong when gradient coincides with either bright SE rim OR dark contact crevice
    const dipole = Math.max(rN, vN * 0.92);
    let b =
      gN * (0.58 - 0.24 * edgeBarrierStrength) +
      edgeBarrierStrength * (0.42 * dipole + 0.38 * Math.min(1, gN * (0.6 + 0.8 * dipole)));
    if (edgeField) {
      const er = edgeField.edgeResponse[i];
      if (edgeField.mode === 'canny_mild' || edgeField.mode === 'canny_strong') {
        if (edgeField.cannyMask[i] > 0.5) {
          const damWeight = edgeField.mode === 'canny_strong' ? 0.9 : 0.8;
          b = Math.max(b, er * damWeight);
        } else {
          b = b * 0.9 + er * 0.18;
        }
      } else {
        const sobelWeight = edgeField.mode === 'sobel_5x5' ? 0.48 : 0.36;
        b = Math.max(b, b * (1 - sobelWeight * 0.45) + er * sobelWeight);
      }
    }
    rawBarrier[i] = Math.min(1, Math.max(0, b));
  }

  // 4b. Directional 1-pass rim-gap bridging: closes 1-2px weak pinholes along SE edges to prevent watershed leaks
  const barrier = new Float32Array(rawBarrier);
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      const idx = row + x;
      if (!fgMask[idx]) continue;
      const cur = rawBarrier[idx];
      if (cur >= 0.65) continue;
      const bH = Math.min(rawBarrier[idx - 1], rawBarrier[idx + 1]);
      const bV = Math.min(rawBarrier[idx - w], rawBarrier[idx + w]);
      const bD1 = Math.min(rawBarrier[idx - w - 1], rawBarrier[idx + w + 1]);
      const bD2 = Math.min(rawBarrier[idx - w + 1], rawBarrier[idx + w - 1]);
      const bridge = Math.max(bH, bV, bD1, bD2);
      if (bridge > 0.42 && bridge * 0.88 > cur) {
        barrier[idx] = bridge * 0.88;
      }
    }
  }

  // 5. Interior core extraction
  const coreThresh = Math.max(0.24, Math.min(0.76, 0.60 - 0.30 * splitSensitivity));
  const coreMask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    coreMask[i] = fgMask[i] && barrier[i] < coreThresh ? 1 : 0;
  }

  const distCoreRaw = computeEDT(coreMask, w, h);
  const distFg = computeEDT(fgMask, w, h);
  let maxDistFg = 1;
  for (let i = 0; i < n; i++) {
    if (distFg[i] > maxDistFg) maxDistFg = distFg[i];
  }

  const distCore = new Float32Array(n);
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      const idx = row + x;
      if (!coreMask[idx]) continue;
      distCore[idx] =
        (distCoreRaw[idx - w - 1] +
          distCoreRaw[idx - w] +
          distCoreRaw[idx - w + 1] +
          distCoreRaw[idx - 1] +
          distCoreRaw[idx] * 2 +
          distCoreRaw[idx + 1] +
          distCoreRaw[idx + w - 1] +
          distCoreRaw[idx + w] +
          distCoreRaw[idx + w + 1]) /
        10;
    }
  }

  // 6. Multi-scale barrier-aware seed peak detection (retains small spheres sitting on large crystals)
  const minSeedDist = Math.max(2.0, minDiameterPx * (0.26 - 0.1 * splitSensitivity));
  const candidates: Array<{ x: number; y: number; r: number }> = [];
  const winR = Math.max(2, Math.min(4, Math.round(minDiameterPx * 0.2)));

  for (let y = winR; y < h - winR; y++) {
    const row = y * w;
    for (let x = winR; x < w - winR; x++) {
      const idx = row + x;
      const d = distCore[idx];
      if (d < minSeedDist) continue;
      let isMax = true;
      for (let dy = -winR; dy <= winR && isMax; dy++) {
        const nRow = (y + dy) * w;
        for (let dx = -winR; dx <= winR; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nd = distCore[nRow + x + dx];
          if (nd > d || (nd === d && (dy < 0 || (dy === 0 && dx < 0)))) {
            isMax = false;
            break;
          }
        }
      }
      if (isMax) {
        candidates.push({ x, y, r: d });
      }
    }
  }
  candidates.sort((a, b) => b.r - a.r);

  const hasSeparatingBarrier = (ax: number, ay: number, bx: number, by: number): boolean => {
    const steps = Math.max(5, Math.min(72, Math.round(Math.hypot(bx - ax, by - ay))));
    let maxBar = 0;
    let strongSteps = 0;
    let hitsBg = false;
    for (let s = 1; s < steps; s++) {
      const t = s / steps;
      const sx = Math.round(ax + (bx - ax) * t);
      const sy = Math.round(ay + (by - ay) * t);
      const idx = sy * w + sx;
      if (!fgMask[idx]) {
        hitsBg = true;
        break;
      }
      const bVal = barrier[idx];
      if (bVal > maxBar) maxBar = bVal;
      if (bVal >= coreThresh * 0.92) strongSteps++;
    }
    return hitsBg || (maxBar >= coreThresh * 0.96 && strongSteps >= 1);
  };

  const seeds: Array<{ x: number; y: number; r: number }> = [];
  const minBaseGap = Math.max(4, minDiameterPx * (0.52 - 0.24 * splitSensitivity));

  for (const cand of candidates) {
    let keep = true;
    for (const ex of seeds) {
      const dist = Math.hypot(cand.x - ex.x, cand.y - ex.y);
      const smallerR = Math.min(cand.r, ex.r);
      const largerR = Math.max(cand.r, ex.r);
      if (dist < Math.max(minBaseGap, smallerR * 0.72)) {
        keep = false;
        break;
      }
      // Only merge if there is NO separating SE rim/crevice between the two peaks.
      // This preserves small spheres resting on/beside much larger faceted crystals!
      if (dist < largerR * (1.22 - 0.28 * splitSensitivity)) {
        if (!hasSeparatingBarrier(cand.x, cand.y, ex.x, ex.y)) {
          keep = false;
          break;
        }
      }
    }
    if (keep) {
      seeds.push(cand);
      if (seeds.length >= 480) break;
    }
  }

  if (seeds.length === 0) {
    for (let y = 2; y < h - 2; y += 4) {
      for (let x = 2; x < w - 2; x += 4) {
        const idx = y * w + x;
        if (distFg[idx] >= minDiameterPx * 0.35) {
          let close = false;
          for (const s of seeds) {
            if (Math.hypot(x - s.x, y - s.y) < minDiameterPx * 0.75) {
              close = true;
              break;
            }
          }
          if (!close) seeds.push({ x, y, r: distFg[idx] });
        }
      }
    }
  }

  // 7. Priority-Queue Marker-Controlled Watershed flooding strictly guided by the sealed SE barrier
  const labels = new Int32Array(n);
  const inQueue = new Uint8Array(n);
  const heap = new PixelMinHeap(Math.min(n, 32768));

  const barWeight = 0.62 + 0.32 * edgeBarrierStrength;
  const distWeight = 1 - barWeight;

  const enqueueNeighbors = (idx: number) => {
    const px = idx % w;
    const py = (idx - px) / w;
    const tryPush = (nIdx: number) => {
      if (!fgMask[nIdx] || labels[nIdx] !== 0 || inQueue[nIdx]) return;
      inQueue[nIdx] = 1;
      const cost = barrier[nIdx] * barWeight + (1 - distFg[nIdx] / maxDistFg) * distWeight;
      heap.push(cost, nIdx);
    };
    if (px > 0) tryPush(idx - 1);
    if (px + 1 < w) tryPush(idx + 1);
    if (py > 0) tryPush(idx - w);
    if (py + 1 < h) tryPush(idx + w);
  };

  for (let sIdx = 0; sIdx < seeds.length; sIdx++) {
    const label = sIdx + 1;
    const { x, y } = seeds[sIdx];
    const cIdx = y * w + x;
    labels[cIdx] = label;
    for (let dy = -1; dy <= 1; dy++) {
      const ny = y + dy;
      if (ny < 0 || ny >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx;
        if (nx < 0 || nx >= w) continue;
        const ni = ny * w + nx;
        if (coreMask[ni] && labels[ni] === 0) {
          labels[ni] = label;
        }
      }
    }
  }

  for (let sIdx = 0; sIdx < seeds.length; sIdx++) {
    const { x, y } = seeds[sIdx];
    for (let dy = -1; dy <= 1; dy++) {
      const ny = y + dy;
      if (ny < 0 || ny >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx;
        if (nx < 0 || nx >= w) continue;
        const ni = ny * w + nx;
        if (labels[ni] === sIdx + 1) enqueueNeighbors(ni);
      }
    }
  }

  while (heap.size > 0) {
    const idx = heap.pop();
    if (labels[idx] !== 0) continue;
    const px = idx % w;
    const py = (idx - px) / w;

    let bestLabel = 0;
    let bestJump = Infinity;
    const checkNeighbor = (ni: number) => {
      const lb = labels[ni];
      if (lb === 0) return;
      const jump = Math.abs(smooth[idx] - smooth[ni]) * 0.35 + barrier[ni] * 90;
      if (jump < bestJump) {
        bestJump = jump;
        bestLabel = lb;
      }
    };
    if (px > 0) checkNeighbor(idx - 1);
    if (px + 1 < w) checkNeighbor(idx + 1);
    if (py > 0) checkNeighbor(idx - w);
    if (py + 1 < h) checkNeighbor(idx + w);

    if (bestLabel > 0) {
      labels[idx] = bestLabel;
      enqueueNeighbors(idx);
    }
  }

  // 8. Region Adjacency Graph (RAG) Weak-Boundary Merging:
  //    Reunites over-segmented halves of a single shaded crystal face when their shared border lacks a true SE rim/crevice.
  const numSeeds = seeds.length;
  if (numSeeds >= 2) {
    const regArea = new Int32Array(numSeeds + 1);
    const regGraySum = new Float64Array(numSeeds + 1);
    const edgeMap = new Map<number, { count: number; barSum: number; strongCnt: number }>();

    const addEdge = (a: number, b: number, barVal: number) => {
      const lo = a < b ? a : b;
      const hi = a < b ? b : a;
      const key = lo * (numSeeds + 2) + hi;
      let rec = edgeMap.get(key);
      if (!rec) {
        rec = { count: 0, barSum: 0, strongCnt: 0 };
        edgeMap.set(key, rec);
      }
      rec.count++;
      rec.barSum += barVal;
      if (barVal >= 0.44) rec.strongCnt++;
    };

    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const idx = row + x;
        const lb = labels[idx];
        if (lb <= 0) continue;
        regArea[lb]++;
        regGraySum[lb] += smooth[idx];
        if (x + 1 < w) {
          const rLb = labels[idx + 1];
          if (rLb > 0 && rLb !== lb) {
            addEdge(lb, rLb, Math.max(barrier[idx], barrier[idx + 1]));
          }
        }
        if (y + 1 < h) {
          const dLb = labels[idx + w];
          if (dLb > 0 && dLb !== lb) {
            addEdge(lb, dLb, Math.max(barrier[idx], barrier[idx + w]));
          }
        }
      }
    }

    const parent = new Int32Array(numSeeds + 1);
    for (let i = 0; i <= numSeeds; i++) parent[i] = i;
    const find = (x: number): number => {
      let r = x;
      while (parent[r] !== r) r = parent[r];
      while (parent[x] !== r) {
        const nxt = parent[x];
        parent[x] = r;
        x = nxt;
      }
      return r;
    };

    // Adaptive RAG merge cutoff: higher splitSensitivity -> stricter merging (fewer merges); lower -> stronger facet merging
    const ragBarCutoff = Math.max(0.18, Math.min(0.46, 0.42 - 0.22 * splitSensitivity));
    const pairs: Array<{ a: number; b: number; meanBar: number; strongRatio: number; count: number }> = [];
    for (const [key, rec] of edgeMap.entries()) {
      if (rec.count < 4) continue;
      const lo = Math.floor(key / (numSeeds + 2));
      const hi = key - lo * (numSeeds + 2);
      pairs.push({
        a: lo,
        b: hi,
        meanBar: rec.barSum / rec.count,
        strongRatio: rec.strongCnt / rec.count,
        count: rec.count,
      });
    }
    pairs.sort((p1, p2) => p1.meanBar - p2.meanBar);

    let mergedAny = false;
    for (const p of pairs) {
      if (p.meanBar > ragBarCutoff || p.strongRatio > 0.3) continue;
      const ra = find(p.a);
      const rb = find(p.b);
      if (ra === rb) continue;
      const meanA = regGraySum[ra] / Math.max(1, regArea[ra]);
      const meanB = regGraySum[rb] / Math.max(1, regArea[rb]);
      if (Math.abs(meanA - meanB) > 26) continue;
      // Merge smaller region into larger region
      if (regArea[ra] >= regArea[rb]) {
        parent[rb] = ra;
        regArea[ra] += regArea[rb];
        regGraySum[ra] += regGraySum[rb];
      } else {
        parent[ra] = rb;
        regArea[rb] += regArea[ra];
        regGraySum[rb] += regGraySum[ra];
      }
      mergedAny = true;
    }

    if (mergedAny) {
      for (let i = 0; i < n; i++) {
        if (labels[i] > 0) labels[i] = find(labels[i]);
      }
    }
  }

  return extractSeparatedComponentsWithOcclusion(labels, smooth, w, h, {
    minDiameterPx,
    occlusionSensitivity,
    excludeBorder,
    curvatureBoundaryFit,
    overlapConfidenceThreshold,
    edgeRefinement,
    edgeField,
  });
}

/**
 * Extracts exact pixel-edge contours, true convex-hull solidity, edge-curvature + overlap-confidence
 * optimized boundary fitting, and multi-cue partial-occlusion scores from any instance label map.
 */
export function extractSeparatedComponentsWithOcclusion(
  labels: Int32Array,
  gray: Float32Array,
  w: number,
  h: number,
  opts: {
    minDiameterPx: number;
    occlusionSensitivity: number;
    excludeBorder: boolean;
    curvatureBoundaryFit?: boolean;
    overlapConfidenceThreshold?: number;
    edgeRefinement?: EdgeRefinementMode;
    edgeField?: EdgeRefinementField | null;
  }
): SeparatedParticleComponent[] {
  const {
    minDiameterPx,
    occlusionSensitivity,
    excludeBorder,
    curvatureBoundaryFit = true,
    overlapConfidenceThreshold,
    edgeRefinement = 'off',
  } = opts;
  const edgeField =
    opts.edgeField !== undefined
      ? opts.edgeField
      : edgeRefinement !== 'off'
      ? computeEdgeRefinementField(gray, w, h, edgeRefinement)
      : null;
  const minAreaPx = Math.PI * (minDiameterPx / 2) ** 2;
  const maxAreaPx = w * h * 0.48;

  let maxLabel = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] > maxLabel) maxLabel = labels[i];
  }
  if (maxLabel === 0) return [];

  const area = new Int32Array(maxLabel + 1);
  const sumX = new Float64Array(maxLabel + 1);
  const sumY = new Float64Array(maxLabel + 1);
  const sumG = new Float64Array(maxLabel + 1);
  const minX = new Int32Array(maxLabel + 1).fill(w);
  const maxX = new Int32Array(maxLabel + 1).fill(-1);
  const minY = new Int32Array(maxLabel + 1).fill(h);
  const maxY = new Int32Array(maxLabel + 1).fill(-1);
  const perim = new Float64Array(maxLabel + 1);
  const touchBorder = new Uint8Array(maxLabel + 1);
  const boundaryCoords: Array<Array<[number, number]>> = Array.from({ length: maxLabel + 1 }, () => []);
  const boundaryGrays: Array<number[]> = Array.from({ length: maxLabel + 1 }, () => []);

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const idx = row + x;
      const lb = labels[idx];
      if (lb <= 0) continue;

      area[lb]++;
      sumX[lb] += x;
      sumY[lb] += y;
      sumG[lb] += gray[idx];
      if (x < minX[lb]) minX[lb] = x;
      if (x > maxX[lb]) maxX[lb] = x;
      if (y < minY[lb]) minY[lb] = y;
      if (y > maxY[lb]) maxY[lb] = y;
      if (x <= 1 || x >= w - 2 || y <= 1 || y >= h - 2) touchBorder[lb] = 1;

      let diffEdges = 0;
      if (x === 0 || labels[idx - 1] !== lb) diffEdges++;
      if (x + 1 >= w || labels[idx + 1] !== lb) diffEdges++;
      if (y === 0 || labels[idx - w] !== lb) diffEdges++;
      if (y + 1 >= h || labels[idx + w] !== lb) diffEdges++;
      if (diffEdges > 0) {
        perim[lb] += diffEdges === 1 ? 1 : diffEdges === 2 ? 1.414 : 2;
        boundaryCoords[lb].push([x, y]);
        boundaryGrays[lb].push(gray[idx]);
      }
    }
  }

  const validLabel = new Uint8Array(maxLabel + 1);
  const meanGrayArr = new Float32Array(maxLabel + 1);
  const rimP90Arr = new Float32Array(maxLabel + 1);

  for (let lb = 1; lb <= maxLabel; lb++) {
    if (area[lb] < minAreaPx || area[lb] > maxAreaPx) continue;
    const eqD = Math.sqrt((4 * area[lb]) / Math.PI);
    if (eqD < minDiameterPx) continue;
    if (excludeBorder && touchBorder[lb]) continue;
    validLabel[lb] = 1;
    meanGrayArr[lb] = sumG[lb] / area[lb];
    const bgs = boundaryGrays[lb];
    if (bgs.length > 0) {
      bgs.sort((a, b) => a - b);
      rimP90Arr[lb] = bgs[Math.min(bgs.length - 1, Math.floor(bgs.length * 0.9))];
    } else {
      rimP90Arr[lb] = meanGrayArr[lb];
    }
  }

  const results: SeparatedParticleComponent[] = [];
  const occThreshold =
    overlapConfidenceThreshold !== undefined
      ? Math.max(0.18, Math.min(0.82, overlapConfidenceThreshold))
      : Math.max(0.2, Math.min(0.68, 0.62 - 0.42 * occlusionSensitivity));

  const probeOffsets: Array<[number, number]> = [
    [-1, 0], [1, 0], [0, -1], [0, 1],
    [-2, 0], [2, 0], [0, -2], [0, 2],
    [-2, -2], [2, -2], [-2, 2], [2, 2],
    [-3, 0], [3, 0], [0, -3], [0, 3],
  ];

  for (let lb = 1; lb <= maxLabel; lb++) {
    if (!validLabel[lb]) continue;

    const a = area[lb];
    const cx = sumX[lb] / a + 0.5;
    const cy = sumY[lb] / a + 0.5;
    const eqD = Math.sqrt((4 * a) / Math.PI);
    const circ = Math.min(1, (4 * Math.PI * a) / Math.max(1, perim[lb] * perim[lb]));
    const bw = Math.max(1, maxX[lb] - minX[lb] + 1);
    const bh = Math.max(1, maxY[lb] - minY[lb] + 1);
    const aspectRatio = Math.max(bw, bh) / Math.min(bw, bh);

    const traced = traceComponentPixelContours(
      labels,
      lb,
      w,
      h,
      minX[lb],
      maxX[lb],
      minY[lb],
      maxY[lb]
    );

    const hull = computeConvexHull(traced.outer);
    const hullArea = Math.max(a, polygonArea(hull));
    const trueSolidity = Math.min(1, a / Math.max(1, hullArea));
    const recessDepthThresh = Math.max(1.8, Math.min(5.0, eqD * 0.055));

    // Boundary curvature + overlap confidence analysis & sub-pixel boundary fitting optimization
    const fitInfo = analyzeAndOptimizeBoundaryFit(
      traced.outer,
      hull,
      labels,
      gray,
      w,
      h,
      lb,
      validLabel,
      meanGrayArr,
      rimP90Arr,
      recessDepthThresh,
      curvatureBoundaryFit,
      edgeField
    );

    const bPts = boundaryCoords[lb];
    const totalB = Math.max(1, bPts.length);
    let contactPx = 0;
    let recessedContactPx = 0;
    let brighterContactPx = 0;
    let shadowDropPx = 0;

    const selfMean = meanGrayArr[lb];
    const selfRim = rimP90Arr[lb];

    for (let i = 0; i < bPts.length; i++) {
      const [bx, by] = bPts[i];
      let neighborLb = 0;
      let neighborGrayAtSeam = 0;

      for (let p = 0; p < probeOffsets.length; p++) {
        const nx = bx + probeOffsets[p][0];
        const ny = by + probeOffsets[p][1];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const nl = labels[ny * w + nx];
        if (nl > 0 && nl !== lb && validLabel[nl]) {
          neighborLb = nl;
          neighborGrayAtSeam = gray[ny * w + nx];
          break;
        }
      }

      if (neighborLb > 0) {
        contactPx++;
        const dHull = pointToHullEdgeDistance(bx + 0.5, by + 0.5, hull);
        if (dHull >= recessDepthThresh) {
          recessedContactPx++;
        }
        if (rimP90Arr[neighborLb] > selfRim + 5 || meanGrayArr[neighborLb] > selfMean + 8) {
          brighterContactPx++;
        }
        if (neighborGrayAtSeam > gray[by * w + bx] + 10) {
          shadowDropPx++;
        }
      }
    }

    const contactRatio = contactPx / totalB;
    const recessedContactRatio = recessedContactPx / totalB;
    const brighterRatio = brighterContactPx / totalB;
    const shadowRatio = shadowDropPx / totalB;
    const concavityDeficit = Math.max(0, 0.945 - trueSolidity) / 0.24;

    const contactGate = contactRatio >= 0.08 ? 1 : contactRatio / 0.08;
    const rawOcc =
      contactGate *
      (0.32 * Math.min(1, recessedContactRatio / 0.13) +
        0.26 * fitInfo.overlapConfidence +
        0.18 * Math.min(1, concavityDeficit) +
        0.14 * Math.min(1, brighterRatio / 0.28) +
        0.1 * Math.min(1, shadowRatio / 0.22));
    const cuspBoost = fitInfo.concaveCuspCount >= 2 && contactRatio >= 0.12 ? 0.1 : 0;
    const encirclementBoost =
      contactRatio >= 0.34 && (brighterRatio >= 0.18 || shadowRatio >= 0.16) ? 0.2 : 0;
    const occlusionScore = Math.max(0, Math.min(1, rawOcc + cuspBoost + encirclementBoost));

    const occluded =
      (occlusionScore >= occThreshold || fitInfo.overlapConfidence >= occThreshold + 0.08) &&
      contactRatio >= 0.1;

    let occludedReason: string | undefined;
    if (occluded) {
      const cues: string[] = [];
      if (fitInfo.concaveCuspCount > 0) {
        cues.push(`曲率凹角 ${fitInfo.concaveCuspCount} 处`);
      }
      if (fitInfo.overlapConfidence >= 0.25) {
        cues.push(`重叠置信 ${Math.round(fitInfo.overlapConfidence * 100)}%`);
      }
      if (trueSolidity < 0.91) {
        cues.push(`凸包实心度 ${trueSolidity.toFixed(2)}`);
      }
      if (brighterRatio >= 0.14 || shadowRatio >= 0.12) {
        cues.push('上层亮边压盖');
      }
      if (cues.length === 0) {
        cues.push(`邻接接触 ${Math.round(contactRatio * 100)}%`);
      }
      occludedReason = `疑似被上层颗粒部分遮挡（${cues.join(' · ')}）`;
    }

    results.push({
      id: results.length + 1,
      areaPx: a,
      cx,
      cy,
      minX: minX[lb],
      maxX: maxX[lb],
      minY: minY[lb],
      maxY: maxY[lb],
      eqDiameterPx: eqD,
      circularity: circ,
      solidity: trueSolidity,
      aspectRatio,
      meanGray: selfMean,
      rimGrayP90: selfRim,
      touchingBorder: Boolean(touchBorder[lb]),
      occluded,
      occlusionScore: Number(occlusionScore.toFixed(3)),
      overlapConfidence: fitInfo.overlapConfidence,
      concaveCuspCount: fitInfo.concaveCuspCount,
      boundaryFitScore: fitInfo.boundaryFitScore,
      occludedReason,
      contactRatio: Number(contactRatio.toFixed(3)),
      recessedContactRatio: Number(recessedContactRatio.toFixed(3)),
      contour: fitInfo.optimizedContour,
      holeContours: traced.holes,
    });
  }

  return results;
}
