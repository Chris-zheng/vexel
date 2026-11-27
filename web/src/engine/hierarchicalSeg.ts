import type {
  DetectionObject,
  HumanCorrection,
  OverlayShape,
  Specimen,
  SummaryMetric,
} from '../types/vexel';
import { traceComponentPixelContours } from './contourTrace';

export interface ParentRegionInstance {
  id: number;
  cx: number;
  cy: number;
  areaPx: number;
  perimeterPx: number;
  circularity: number;
  equivRadiusPx: number;
  bbox: { minX: number; minY: number; maxX: number; maxY: number };
  polygon: Array<[number, number]>;
  holes?: Array<Array<[number, number]>>;
  confidence: number;
  membraneScore?: number;
}

export interface ParentSegmentationOutput {
  parents: ParentRegionInstance[];
  /** Instance label map (0 = background, 1..N = parent.id) at working resolution. */
  parentMask: Int32Array;
  width: number;
  height: number;
  previewGray: Float32Array;
  qcLabel: string;
}

export interface SubRegionInstance {
  id: number;
  parentId: number;
  cx: number;
  cy: number;
  areaPx: number;
  perimeterPx: number;
  circularity: number;
  equivRadiusPx: number;
  meanIntensity: number;
  parentInteriorMedian: number;
  contrastDelta: number;
  polygon: Array<[number, number]>;
  holes?: Array<Array<[number, number]>>;
  confidence: number;
}

export interface SubRegionSegmentationOutput {
  subRegions: SubRegionInstance[];
  /** Sub-region label map (0 = none, 1..M = subRegion.id) at working resolution. */
  subMask: Int32Array;
  width: number;
  height: number;
  previewGray: Float32Array;
  qcLabel: string;
}

export interface HierarchicalMetrologyOutput {
  objects: DetectionObject[];
  shapes: OverlayShape[];
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  warnings: string[];
  histogram?: {
    labelZh: string;
    labelEn: string;
    unit: string;
    values: number[];
    mean: number;
  };
  qcLabel: string;
}

function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  if (r <= 0) return new Float32Array(src);
  const tmp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  const d = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    let sum = 0;
    const row = y * w;
    for (let x = -r; x <= r; x++) {
      sum += src[row + Math.max(0, Math.min(w - 1, x))];
    }
    for (let x = 0; x < w; x++) {
      tmp[row + x] = sum / d;
      sum +=
        src[row + Math.min(w - 1, x + r + 1)] -
        src[row + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) {
      sum += tmp[Math.max(0, Math.min(h - 1, y)) * w + x];
    }
    for (let y = 0; y < h; y++) {
      dst[y * w + x] = sum / d;
      sum +=
        tmp[Math.min(h - 1, y + r + 1) * w + x] -
        tmp[Math.max(0, y - r) * w + x];
    }
  }
  return dst;
}

function sampleNN(buf: Float32Array, w: number, h: number, x: number, y: number): number {
  const xi = Math.max(0, Math.min(w - 1, Math.round(x)));
  const yi = Math.max(0, Math.min(h - 1, Math.round(y)));
  return buf[yi * w + xi];
}

/** Fast O(area) scanline polygon rasterizer writing `labelValue` into `mask`. Returns filled pixel count. */
function rasterizePolygonToMask(
  poly: Array<[number, number]>,
  w: number,
  h: number,
  mask: Int32Array | Uint8Array,
  labelValue: number
): { count: number; minX: number; minY: number; maxX: number; maxY: number } {
  const n = poly.length;
  if (n < 3) return { count: 0, minX: 0, minY: 0, maxX: 0, maxY: 0 };

  let minY = h - 1;
  let maxY = 0;
  let minX = w - 1;
  let maxX = 0;
  for (let i = 0; i < n; i++) {
    const [x, y] = poly[i];
    if (y < minY) minY = Math.floor(y);
    if (y > maxY) maxY = Math.ceil(y);
    if (x < minX) minX = Math.floor(x);
    if (x > maxX) maxX = Math.ceil(x);
  }
  minY = Math.max(0, minY);
  maxY = Math.min(h - 1, maxY);
  minX = Math.max(0, minX);
  maxX = Math.min(w - 1, maxX);

  const xs: number[] = [];
  let count = 0;
  for (let y = minY; y <= maxY; y++) {
    const py = y + 0.5;
    xs.length = 0;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const [x1, y1] = poly[j];
      const [x2, y2] = poly[i];
      if ((y1 <= py && y2 > py) || (y2 <= py && y1 > py)) {
        xs.push(x1 + ((py - y1) * (x2 - x1)) / (y2 - y1));
      }
    }
    if (xs.length < 2) continue;
    xs.sort((a, b) => a - b);
    const rowOff = y * w;
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const xStart = Math.max(0, Math.ceil(xs[k] - 0.5));
      const xEnd = Math.min(w - 1, Math.floor(xs[k + 1] - 0.5));
      for (let x = xStart; x <= xEnd; x++) {
        mask[rowOff + x] = labelValue;
        count++;
      }
    }
  }
  return { count, minX, minY, maxX, maxY };
}

function polygonPerimeter(poly: Array<[number, number]>): number {
  let perim = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    perim += Math.hypot(x2 - x1, y2 - y1);
  }
  return perim;
}

function otsuThresholdLocal(gray: Float32Array, n: number): number {
  const hist = new Uint32Array(256);
  for (let i = 0; i < n; i++) {
    hist[Math.max(0, Math.min(255, Math.round(gray[i])))]++;
  }
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0;
  let wB = 0;
  let maxVar = -1;
  let bestT = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = n - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) {
      maxVar = between;
      bestT = t;
    }
  }
  return bestT;
}

/** Morph closing (dilate then erode) on a local bbox binary mask, clipped to `validMask`. */
function morphCloseClipped(
  bin: Uint8Array,
  validMask: Uint8Array,
  bw: number,
  bh: number,
  radius: number
): Uint8Array {
  if (radius <= 0) return bin;
  const dil = new Uint8Array(bw * bh);
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const idx = y * bw + x;
      if (!validMask[idx]) continue;
      let hit = 0;
      for (let dy = -radius; dy <= radius && !hit; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= bh) continue;
        const row = ny * bw;
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          if (nx >= 0 && nx < bw && bin[row + nx]) {
            hit = 1;
            break;
          }
        }
      }
      dil[idx] = hit;
    }
  }
  const out = new Uint8Array(bw * bh);
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const idx = y * bw + x;
      if (!dil[idx]) continue;
      let all = 1;
      for (let dy = -radius; dy <= radius && all; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= bh) {
          all = 0;
          break;
        }
        const row = ny * bw;
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= bw || !dil[row + nx]) {
            all = 0;
            break;
          }
        }
      }
      out[idx] = all;
    }
  }
  return out;
}

/** Fill enclosed background holes inside a local bbox binary mask via border flood-fill. */
function fillBinaryHoles(bin: Uint8Array, bw: number, bh: number): Uint8Array {
  const outside = new Uint8Array(bw * bh);
  const q = new Int32Array(bw * bh);
  let qh = 0;
  let qt = 0;
  const push = (idx: number) => {
    if (!bin[idx] && !outside[idx]) {
      outside[idx] = 1;
      q[qt++] = idx;
    }
  };
  for (let x = 0; x < bw; x++) {
    push(x);
    push((bh - 1) * bw + x);
  }
  for (let y = 0; y < bh; y++) {
    push(y * bw);
    push(y * bw + bw - 1);
  }
  while (qh < qt) {
    const cur = q[qh++];
    const x = cur % bw;
    const y = (cur - x) / bw;
    if (x > 0) push(cur - 1);
    if (x + 1 < bw) push(cur + 1);
    if (y > 0) push(cur - bw);
    if (y + 1 < bh) push(cur + bw);
  }
  const out = new Uint8Array(bw * bh);
  for (let i = 0; i < bw * bh; i++) {
    out[i] = bin[i] || !outside[i] ? 1 : 0;
  }
  return out;
}

// ---------------------------------------------------------------------------
// STAGE 1: Parent Region Segmentation (`region.parent_boundary`)
// ---------------------------------------------------------------------------

export function runParentRegionSegmentation(
  gray: Float32Array,
  width: number,
  height: number,
  params: Record<string, number | boolean | string>,
  upstreamProbMap?: Float32Array | null,
  upstreamProbThreshold = 0.5
): ParentSegmentationOutput {
  const mode = String(params.mode ?? 'dark_membrane');
  const minRadiusPct = Math.max(3, Math.min(40, Number(params.minRadiusPct ?? 14)));
  const maxRadiusPct = Math.max(minRadiusPct + 4, Math.min(55, Number(params.maxRadiusPct ?? 44)));
  const sensitivity = Math.max(0.15, Math.min(0.95, Number(params.membraneSensitivity ?? 0.58)));
  const minAreaPx = Math.max(64, Number(params.minAreaPx ?? 800));

  const parentMask = new Int32Array(width * height);
  const parents: ParentRegionInstance[] = [];

  if (mode === 'upstream_model' || mode === 'intensity_blobs') {
    const bin = new Uint8Array(width * height);
    if (mode === 'upstream_model' && upstreamProbMap && upstreamProbMap.length >= width * height) {
      for (let i = 0; i < width * height; i++) {
        if (upstreamProbMap[i] >= upstreamProbThreshold) bin[i] = 1;
      }
    } else {
      const smooth = boxBlur(gray, width, height, 2);
      const tOtsu = otsuThresholdLocal(smooth, width * height);
      // Determine foreground polarity by border mean vs interior
      let borderSum = 0;
      let borderCnt = 0;
      for (let x = 0; x < width; x++) {
        borderSum += smooth[x] + smooth[(height - 1) * width + x];
        borderCnt += 2;
      }
      const borderMean = borderSum / Math.max(1, borderCnt);
      const brightFg = borderMean < tOtsu;
      const bias = (0.5 - sensitivity) * 45;
      for (let i = 0; i < width * height; i++) {
        bin[i] = brightFg ? (smooth[i] > tOtsu + bias ? 1 : 0) : smooth[i] < tOtsu - bias ? 1 : 0;
      }
    }

    const filled = fillBinaryHoles(bin, width, height);
    const vis = new Uint8Array(width * height);
    const q = new Int32Array(width * height);
    const comps: Array<{ pixels: number[]; minX: number; minY: number; maxX: number; maxY: number; sx: number; sy: number }> = [];

    for (let i = 0; i < width * height; i++) {
      if (!filled[i] || vis[i]) continue;
      let qh = 0;
      let qt = 0;
      q[qt++] = i;
      vis[i] = 1;
      const pixels: number[] = [];
      let minX = width;
      let minY = height;
      let maxX = 0;
      let maxY = 0;
      let sx = 0;
      let sy = 0;
      while (qh < qt) {
        const cur = q[qh++];
        pixels.push(cur);
        const x = cur % width;
        const y = (cur - x) / width;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        sx += x;
        sy += y;
        if (x > 0 && filled[cur - 1] && !vis[cur - 1]) { vis[cur - 1] = 1; q[qt++] = cur - 1; }
        if (x + 1 < width && filled[cur + 1] && !vis[cur + 1]) { vis[cur + 1] = 1; q[qt++] = cur + 1; }
        if (y > 0 && filled[cur - width] && !vis[cur - width]) { vis[cur - width] = 1; q[qt++] = cur - width; }
        if (y + 1 < height && filled[cur + width] && !vis[cur + width]) { vis[cur + width] = 1; q[qt++] = cur + width; }
      }
      if (pixels.length >= minAreaPx) {
        comps.push({ pixels, minX, minY, maxX, maxY, sx, sy });
      }
    }

    comps.sort((a, b) => a.sx / a.pixels.length - b.sx / b.pixels.length);
    for (let idx = 0; idx < comps.length; idx++) {
      const c = comps[idx];
      const pid = idx + 1;
      for (const p of c.pixels) parentMask[p] = pid;
      const traced = traceComponentPixelContours(
        parentMask,
        pid,
        width,
        height,
        c.minX,
        c.maxX,
        c.minY,
        c.maxY
      );
      const poly = traced.outer;
      const perim = polygonPerimeter(poly);
      const areaPx = c.pixels.length;
      const circ = Math.min(1, (4 * Math.PI * areaPx) / Math.max(1, perim * perim));
      parents.push({
        id: pid,
        cx: c.sx / areaPx,
        cy: c.sy / areaPx,
        areaPx,
        perimeterPx: perim,
        circularity: circ,
        equivRadiusPx: Math.sqrt(areaPx / Math.PI),
        bbox: { minX: c.minX, minY: c.minY, maxX: c.maxX, maxY: c.maxY },
        polygon: poly,
        holes: traced.holes.length > 0 ? traced.holes : undefined,
        confidence: 0.9,
      });
    }
  } else {
    // -----------------------------------------------------------------------
    // Mode: `dark_membrane`
    // Stage 1A: Fast coarse directional radial-valley + interior homogeneity search
    // Stage 1B: Full-resolution cyclic Polar Dynamic Programming membrane snapping & verification
    // -----------------------------------------------------------------------
    const workGray = boxBlur(gray, width, height, Math.max(1, Math.round(Math.min(width, height) / 480)));
    const scale = Math.max(1, width / 192);
    const cw = Math.max(32, Math.round(width / scale));
    const ch = Math.max(32, Math.round(height / scale));
    const cGray = new Float32Array(cw * ch);

    for (let y = 0; y < ch; y++) {
      const sy0 = Math.floor(y * scale);
      const sy1 = Math.min(height, Math.ceil((y + 1) * scale));
      for (let x = 0; x < cw; x++) {
        const sx0 = Math.floor(x * scale);
        const sx1 = Math.min(width, Math.ceil((x + 1) * scale));
        let s = 0;
        let cnt = 0;
        for (let yy = sy0; yy < sy1; yy++) {
          const row = yy * width;
          for (let xx = sx0; xx < sx1; xx++) {
            s += workGray[row + xx];
            cnt++;
          }
        }
        cGray[y * cw + x] = s / Math.max(1, cnt);
      }
    }

    const cSmooth = boxBlur(cGray, cw, ch, 2);
    const minDimC = Math.min(cw, ch);
    const minRC = Math.max(8, Math.round((minDimC * minRadiusPct) / 100));
    const maxRC = Math.max(minRC + 3, Math.round((minDimC * maxRadiusPct) / 100));
    const K = 40;
    const cosT = new Float32Array(K);
    const sinT = new Float32Array(K);
    for (let k = 0; k < K; k++) {
      const a = (k / K) * Math.PI * 2;
      cosT[k] = Math.cos(a);
      sinT[k] = Math.sin(a);
    }

    const coarseCands: Array<{ cx: number; cy: number; r: number; score: number }> = [];
    const rStep = Math.max(2, Math.round((maxRC - minRC) / 16));
    const xyStep = Math.max(2, Math.round(minDimC / 56));
    const inSamples = new Float32Array(K >> 1);

    for (let r = minRC; r <= maxRC; r += rStep) {
      const pad = r + 3;
      for (let cy = pad; cy < ch - pad; cy += xyStep) {
        for (let cx = pad; cx < cw - pad; cx += xyStep) {
          let hit = 0;
          let sumVal = 0;
          let sIdx = 0;
          for (let k = 0; k < K; k++) {
            const ct = cosT[k];
            const st = sinT[k];
            let bestV = 0;
            for (let dr = -1; dr <= 1; dr++) {
              const rr = r + dr;
              const vMid = sampleNN(cGray, cw, ch, cx + rr * ct, cy + rr * st);
              const vIn = sampleNN(cGray, cw, ch, cx + (rr - 2.5) * ct, cy + (rr - 2.5) * st);
              const vOut = sampleNN(cGray, cw, ch, cx + (rr + 2.5) * ct, cy + (rr + 2.5) * st);
              const dIn = vIn - vMid;
              const dOut = vOut - vMid;
              const valley =
                dIn > 0 && dOut > 0
                  ? Math.min(dIn, dOut) + 0.35 * Math.max(dIn, dOut)
                  : Math.max(0, dIn * 0.35);
              if (valley > bestV) bestV = valley;
            }
            sumVal += bestV;
            if (bestV >= 13) hit++;
            if ((k & 1) === 0) {
              inSamples[sIdx++] = sampleNN(cSmooth, cw, ch, cx + r * 0.72 * ct, cy + r * 0.72 * st);
            }
          }
          const cov = hit / K;
          if (cov < 0.54) continue;
          const sorted = Array.from(inSamples.subarray(0, sIdx)).sort((a, b) => a - b);
          const q25 = sorted[Math.floor(sorted.length * 0.25)];
          const q50 = sorted[Math.floor(sorted.length * 0.5)];
          const q75 = sorted[Math.floor(sorted.length * 0.75)];
          if (q50 < 65 || q50 > 235) continue;
          const iqr = q75 - q25;
          const score = (cov * cov * (sumVal / K)) / (1 + Math.pow(iqr / 28, 2));
          coarseCands.push({
            cx: cx * scale,
            cy: cy * scale,
            r: r * scale,
            score,
          });
        }
      }
    }

    coarseCands.sort((a, b) => b.score - a.score);
    const coarsePicked: Array<{ cx: number; cy: number; r: number; score: number }> = [];
    for (const c of coarseCands) {
      if (
        coarsePicked.some(
          (p) => Math.hypot(c.cx - p.cx, c.cy - p.cy) < (c.r + p.r) * 0.78
        )
      ) {
        continue;
      }
      coarsePicked.push(c);
      if (coarsePicked.length >= 12) break;
    }

    // Stage 1B: Full-resolution Cyclic Polar Dynamic Programming Membrane Snapping
    const N = 128;
    const RSteps = 56;
    const minSnappedCov = 0.88 - sensitivity * 0.32; // at default 0.58 => ~0.69
    const minMeanDv = 64 - sensitivity * 38; // at default 0.58 => ~42

    interface SnappedParent {
      cx: number;
      cy: number;
      areaPx: number;
      perimPx: number;
      circularity: number;
      equivRadiusPx: number;
      pts: Array<[number, number]>;
      snappedCov: number;
      darkCov: number;
      meanDv: number;
      confidence: number;
    }

    const verified: SnappedParent[] = [];
    const dp = new Float32Array(N * RSteps);
    const prev = new Int16Array(N * RSteps);
    const energy = new Float32Array(N * RSteps);
    const darkValleyMat = new Float32Array(N * RSteps);
    const darkIntensityMat = new Float32Array(N * RSteps);

    for (const cand of coarsePicked) {
      const rMin = Math.max(10, cand.r * 0.76);
      const rMax = cand.r * 1.24;
      const dr = (rMax - rMin) / (RSteps - 1);
      const delta = Math.max(5, Math.round(cand.r * 0.045));

      for (let i = 0; i < N; i++) {
        const ang = (i / N) * Math.PI * 2;
        const ca = Math.cos(ang);
        const sa = Math.sin(ang);
        const rowOff = i * RSteps;
        for (let j = 0; j < RSteps; j++) {
          const rr = rMin + j * dr;
          const vMid = sampleNN(workGray, width, height, cand.cx + rr * ca, cand.cy + rr * sa);
          const vIn1 = sampleNN(workGray, width, height, cand.cx + (rr - delta) * ca, cand.cy + (rr - delta) * sa);
          const vIn2 = sampleNN(workGray, width, height, cand.cx + (rr - delta * 1.8) * ca, cand.cy + (rr - delta * 1.8) * sa);
          const vOut1 = sampleNN(workGray, width, height, cand.cx + (rr + delta) * ca, cand.cy + (rr + delta) * sa);
          const vOut2 = sampleNN(workGray, width, height, cand.cx + (rr + delta * 1.8) * ca, cand.cy + (rr + delta * 1.8) * sa);
          const vIn = 0.6 * vIn1 + 0.4 * vIn2;
          const vOut = 0.6 * vOut1 + 0.4 * vOut2;
          const dIn = vIn - vMid;
          const dOut = vOut - vMid;
          const dv =
            dIn > 0 && dOut > 0
              ? Math.min(dIn, dOut) + 0.4 * Math.max(dIn, dOut)
              : Math.max(0, dIn * 0.3);
          darkValleyMat[rowOff + j] = dv;
          darkIntensityMat[rowOff + j] = vMid;
          const prior = Math.pow((rr - cand.r) / cand.r, 2) * 18;
          energy[rowOff + j] = -dv - (165 - vMid) * 0.18 + prior;
        }
      }

      // 2-pass cyclic DP for seamless closed contour
      const maxStep = 2;
      const smoothW = 5.5;
      for (let j = 0; j < RSteps; j++) dp[j] = energy[j];
      for (let pass = 0; pass < 2; pass++) {
        for (let i = 1; i < N; i++) {
          const curRow = i * RSteps;
          const prevRow = (i - 1) * RSteps;
          for (let j = 0; j < RSteps; j++) {
            let bestC = Infinity;
            let bestP = j;
            const j0 = Math.max(0, j - maxStep);
            const j1 = Math.min(RSteps - 1, j + maxStep);
            for (let pj = j0; pj <= j1; pj++) {
              const c = dp[prevRow + pj] + smoothW * (j - pj) * (j - pj);
              if (c < bestC) {
                bestC = c;
                bestP = pj;
              }
            }
            dp[curRow + j] = bestC + energy[curRow + j];
            prev[curRow + j] = bestP;
          }
        }
        if (pass === 0) {
          let minEnd = Infinity;
          const lastRow = (N - 1) * RSteps;
          for (let j = 0; j < RSteps; j++) {
            if (dp[lastRow + j] < minEnd) minEnd = dp[lastRow + j];
          }
          for (let j = 0; j < RSteps; j++) {
            dp[j] = dp[lastRow + j] - minEnd + energy[j];
          }
        }
      }

      let bestEndJ = 0;
      let bestEndC = Infinity;
      const lastRow = (N - 1) * RSteps;
      for (let j = 0; j < RSteps; j++) {
        if (dp[lastRow + j] < bestEndC) {
          bestEndC = dp[lastRow + j];
          bestEndJ = j;
        }
      }
      const rIdx = new Int16Array(N);
      rIdx[N - 1] = bestEndJ;
      for (let i = N - 1; i > 0; i--) {
        rIdx[i - 1] = prev[i * RSteps + rIdx[i]];
      }

      let strongHits = 0;
      let darkHits = 0;
      let dvSum = 0;
      const pts: Array<[number, number]> = [];
      for (let i = 0; i < N; i++) {
        const dv = darkValleyMat[i * RSteps + rIdx[i]];
        const iv = darkIntensityMat[i * RSteps + rIdx[i]];
        dvSum += dv;
        if (dv >= 22) strongHits++;
        if (iv < 135 && dv >= 14) darkHits++;
        let rSmooth = 0;
        for (let d = -2; d <= 2; d++) {
          rSmooth += rMin + rIdx[(i + d + N) % N] * dr;
        }
        const rr = rSmooth / 5;
        const ang = (i / N) * Math.PI * 2;
        pts.push([
          Math.max(1, Math.min(width - 2, cand.cx + rr * Math.cos(ang))),
          Math.max(1, Math.min(height - 2, cand.cy + rr * Math.sin(ang))),
        ]);
      }

      const snappedCov = strongHits / N;
      const darkCov = darkHits / N;
      const meanDv = dvSum / N;

      if (snappedCov >= minSnappedCov && darkCov >= minSnappedCov * 0.9 && meanDv >= minMeanDv) {
        let area2 = 0;
        let sx = 0;
        let sy = 0;
        for (let i = 0; i < N; i++) {
          const [x1, y1] = pts[i];
          const [x2, y2] = pts[(i + 1) % N];
          const cr = x1 * y2 - x2 * y1;
          area2 += cr;
          sx += (x1 + x2) * cr;
          sy += (y1 + y2) * cr;
        }
        const areaPx = Math.abs(area2) * 0.5;
        if (areaPx >= minAreaPx) {
          const cxNew = Math.abs(area2) > 1e-6 ? sx / (3 * area2) : cand.cx;
          const cyNew = Math.abs(area2) > 1e-6 ? sy / (3 * area2) : cand.cy;
          const perimPx = polygonPerimeter(pts);
          const circularity = Math.min(1, (4 * Math.PI * areaPx) / Math.max(1, perimPx * perimPx));
          const confidence = Math.min(0.99, Math.max(0.55, snappedCov * 0.65 + Math.min(1, meanDv / 100) * 0.35));
          verified.push({
            cx: cxNew,
            cy: cyNew,
            areaPx,
            perimPx,
            circularity,
            equivRadiusPx: Math.sqrt(areaPx / Math.PI),
            pts,
            snappedCov,
            darkCov,
            meanDv,
            confidence,
          });
        }
      }
    }

    // Left-to-right ordering for consistent human numbering (#1 left cell, #2 right cell)
    verified.sort((a, b) => a.cx - b.cx);

    for (let idx = 0; idx < verified.length; idx++) {
      const v = verified[idx];
      const pid = idx + 1;
      const rast = rasterizePolygonToMask(v.pts, width, height, parentMask, pid);
      parents.push({
        id: pid,
        cx: v.cx,
        cy: v.cy,
        areaPx: rast.count > 0 ? rast.count : Math.round(v.areaPx),
        perimeterPx: v.perimPx,
        circularity: v.circularity,
        equivRadiusPx: Math.sqrt(Math.max(1, rast.count || v.areaPx) / Math.PI),
        bbox: { minX: rast.minX, minY: rast.minY, maxX: rast.maxX, maxY: rast.maxY },
        polygon: v.pts,
        confidence: Number(v.confidence.toFixed(3)),
        membraneScore: Number(v.meanDv.toFixed(1)),
      });
    }
  }

  // Build diagnostic step preview: dim non-parent exterior, keep parent interiors crisp
  const previewGray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    previewGray[i] = parentMask[i] > 0 ? gray[i] : gray[i] * 0.32;
  }

  return {
    parents,
    parentMask,
    width,
    height,
    previewGray,
    qcLabel: `parents=${parents.length} (${mode})`,
  };
}

// ---------------------------------------------------------------------------
// STAGE 2: Constrained Sub-Region Segmentation (`region.subregion_seg`)
// ---------------------------------------------------------------------------

export function runConstrainedSubRegionSegmentation(
  gray: Float32Array,
  width: number,
  height: number,
  parentOut: ParentSegmentationOutput,
  params: Record<string, number | boolean | string>
): SubRegionSegmentationOutput {
  const polarity = String(params.polarity ?? 'dark');
  const contrastZ = Math.max(0.6, Math.min(4.5, Number(params.contrastZ ?? 2.1)));
  const smoothSigma = Math.max(0.5, Math.min(8.0, Number(params.smoothSigma ?? 3.5)));
  const borderMarginPx = Math.max(0, Math.min(40, Math.round(Number(params.borderMarginPx ?? 12))));
  const minAreaRatioPct = Math.max(0.05, Math.min(30, Number(params.minAreaRatioPct ?? 0.4)));
  const maxAreaRatioPct = Math.max(minAreaRatioPct + 5, Math.min(90, Number(params.maxAreaRatioPct ?? 65)));
  const maxPerParent = Math.max(0, Math.round(Number(params.maxSubregionsPerParent ?? 1)));

  const blurR = Math.max(1, Math.round(smoothSigma));
  const smoothGray = boxBlur(gray, width, height, blurR);

  const subMask = new Int32Array(width * height);
  const subRegions: SubRegionInstance[] = [];
  let nextSubId = 1;

  for (const parent of parentOut.parents) {
    const { minX, minY, maxX, maxY } = parent.bbox;
    const bw = maxX - minX + 1;
    const bh = maxY - minY + 1;
    if (bw < 6 || bh < 6) continue;

    // Build inner eroded search mask strictly inside `parent` to exclude the parent membrane itself
    const validInner = new Uint8Array(bw * bh);
    if (borderMarginPx > 0 && parent.polygon.length >= 6) {
      const innerPoly: Array<[number, number]> = parent.polygon.map(([px, py]) => {
        const dx = px - parent.cx;
        const dy = py - parent.cy;
        const d = Math.hypot(dx, dy) || 1;
        const nr = Math.max(3, d - borderMarginPx);
        return [parent.cx + (dx / d) * nr - minX, parent.cy + (dy / d) * nr - minY];
      });
      rasterizePolygonToMask(innerPoly, bw, bh, validInner, 1);
      // Intersect with actual parentMask for strict containment
      for (let y = 0; y < bh; y++) {
        const fullRow = (y + minY) * width + minX;
        const locRow = y * bw;
        for (let x = 0; x < bw; x++) {
          if (validInner[locRow + x] && parentOut.parentMask[fullRow + x] !== parent.id) {
            validInner[locRow + x] = 0;
          }
        }
      }
    } else {
      for (let y = 0; y < bh; y++) {
        const fullRow = (y + minY) * width + minX;
        const locRow = y * bw;
        for (let x = 0; x < bw; x++) {
          if (parentOut.parentMask[fullRow + x] === parent.id) {
            validInner[locRow + x] = 1;
          }
        }
      }
    }

    // Collect interior pixel intensities for robust median & MAD estimation
    const interiorVals: number[] = [];
    for (let y = 0; y < bh; y++) {
      const fullRow = (y + minY) * width + minX;
      const locRow = y * bw;
      for (let x = 0; x < bw; x++) {
        if (validInner[locRow + x]) {
          interiorVals.push(smoothGray[fullRow + x]);
        }
      }
    }
    if (interiorVals.length < 24) continue;

    interiorVals.sort((a, b) => a - b);
    const med = interiorVals[Math.floor(interiorVals.length * 0.5)];
    const devs = new Float32Array(interiorVals.length);
    for (let i = 0; i < interiorVals.length; i++) {
      devs[i] = Math.abs(interiorVals[i] - med);
    }
    devs.sort();
    const mad = devs[Math.floor(devs.length * 0.5)];
    const robustStd = Math.max(7.5, mad * 1.4826);

    const thresh = polarity === 'bright' ? med + contrastZ * robustStd : med - contrastZ * robustStd;

    const rawBin = new Uint8Array(bw * bh);
    for (let y = 0; y < bh; y++) {
      const fullRow = (y + minY) * width + minX;
      const locRow = y * bw;
      for (let x = 0; x < bw; x++) {
        if (!validInner[locRow + x]) continue;
        const v = smoothGray[fullRow + x];
        if (polarity === 'bright' ? v > thresh : v < thresh) {
          rawBin[locRow + x] = 1;
        }
      }
    }

    // Morphological closing + hole fill inside parent to merge dense chromatin texture inside the nucleus
    const closed = morphCloseClipped(rawBin, validInner, bw, bh, 2);
    const cleanBin = fillBinaryHoles(closed, bw, bh);
    for (let i = 0; i < bw * bh; i++) {
      if (!validInner[i]) cleanBin[i] = 0;
    }

    // Connected components inside this parent
    const vis = new Uint8Array(bw * bh);
    const q = new Int32Array(bw * bh);
    const minSubPx = Math.max(16, Math.round((parent.areaPx * minAreaRatioPct) / 100));
    const maxSubPx = Math.round((parent.areaPx * maxAreaRatioPct) / 100);

    interface LocalComp {
      fullPixels: number[];
      minX: number;
      minY: number;
      maxX: number;
      maxY: number;
      sx: number;
      sy: number;
      intensitySum: number;
      contrastSum: number;
      score: number;
    }

    const parentComps: LocalComp[] = [];
    for (let i = 0; i < bw * bh; i++) {
      if (!cleanBin[i] || vis[i]) continue;
      let qh = 0;
      let qt = 0;
      q[qt++] = i;
      vis[i] = 1;
      const fullPixels: number[] = [];
      let cMinX = width;
      let cMinY = height;
      let cMaxX = 0;
      let cMaxY = 0;
      let sx = 0;
      let sy = 0;
      let intensitySum = 0;
      let contrastSum = 0;

      while (qh < qt) {
        const cur = q[qh++];
        const lx = cur % bw;
        const ly = (cur - lx) / bw;
        const fx = lx + minX;
        const fy = ly + minY;
        const fIdx = fy * width + fx;
        fullPixels.push(fIdx);
        if (fx < cMinX) cMinX = fx;
        if (fx > cMaxX) cMaxX = fx;
        if (fy < cMinY) cMinY = fy;
        if (fy > cMaxY) cMaxY = fy;
        sx += fx;
        sy += fy;
        const val = smoothGray[fIdx];
        intensitySum += val;
        contrastSum += polarity === 'bright' ? Math.max(0, val - med) : Math.max(0, med - val);

        if (lx > 0 && cleanBin[cur - 1] && !vis[cur - 1]) { vis[cur - 1] = 1; q[qt++] = cur - 1; }
        if (lx + 1 < bw && cleanBin[cur + 1] && !vis[cur + 1]) { vis[cur + 1] = 1; q[qt++] = cur + 1; }
        if (ly > 0 && cleanBin[cur - bw] && !vis[cur - bw]) { vis[cur - bw] = 1; q[qt++] = cur - bw; }
        if (ly + 1 < bh && cleanBin[cur + bw] && !vis[cur + bw]) { vis[cur + bw] = 1; q[qt++] = cur + bw; }
      }

      if (fullPixels.length >= minSubPx && fullPixels.length <= maxSubPx) {
        const meanContrast = contrastSum / fullPixels.length;
        parentComps.push({
          fullPixels,
          minX: cMinX,
          minY: cMinY,
          maxX: cMaxX,
          maxY: cMaxY,
          sx,
          sy,
          intensitySum,
          contrastSum,
          score: fullPixels.length * (1 + meanContrast / 30),
        });
      }
    }

    parentComps.sort((a, b) => b.score - a.score);
    const kept = maxPerParent > 0 ? parentComps.slice(0, maxPerParent) : parentComps;

    for (const comp of kept) {
      const sid = nextSubId++;
      for (const fIdx of comp.fullPixels) subMask[fIdx] = sid;
      const traced = traceComponentPixelContours(
        subMask,
        sid,
        width,
        height,
        comp.minX,
        comp.maxX,
        comp.minY,
        comp.maxY
      );
      const areaPx = comp.fullPixels.length;
      const perimPx = polygonPerimeter(traced.outer);
      const circ = Math.min(1, (4 * Math.PI * areaPx) / Math.max(1, perimPx * perimPx));
      const meanI = comp.intensitySum / areaPx;
      const contrastDelta = comp.contrastSum / areaPx;
      subRegions.push({
        id: sid,
        parentId: parent.id,
        cx: comp.sx / areaPx,
        cy: comp.sy / areaPx,
        areaPx,
        perimeterPx: perimPx,
        circularity: circ,
        equivRadiusPx: Math.sqrt(areaPx / Math.PI),
        meanIntensity: Number(meanI.toFixed(1)),
        parentInteriorMedian: Number(med.toFixed(1)),
        contrastDelta: Number(contrastDelta.toFixed(1)),
        polygon: traced.outer,
        holes: traced.holes.length > 0 ? traced.holes : undefined,
        confidence: Number(Math.min(0.99, Math.max(0.65, 0.55 + contrastDelta / 90)).toFixed(3)),
      });
    }
  }

  // Diagnostic step preview: dark outside parent, moderate inside parent, bright sub-region highlight
  const previewGray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    if (subMask[i] > 0) {
      previewGray[i] = 248;
    } else if (parentOut.parentMask[i] > 0) {
      previewGray[i] = gray[i] * 0.58;
    } else {
      previewGray[i] = gray[i] * 0.16;
    }
  }

  return {
    subRegions,
    subMask,
    width,
    height,
    previewGray,
    qcLabel: `subregions=${subRegions.length} in ${parentOut.parents.length} parents`,
  };
}

// ---------------------------------------------------------------------------
// STAGE 3: Hierarchical Parent–Subregion Area Ratio Metrology (`region.hierarchy_ratio`)
// ---------------------------------------------------------------------------

export function runHierarchicalRatioMetrology(
  specimen: Specimen,
  parentOut: ParentSegmentationOutput,
  subOut: SubRegionSegmentationOutput | null,
  params: Record<string, number | boolean | string>,
  corrections: HumanCorrection[]
): HierarchicalMetrologyOutput {
  const minRatioPct = Math.max(0, Number(params.minRatioPct ?? 0));
  const maxRatioPct = Math.max(minRatioPct, Number(params.maxRatioPct ?? 100));
  const pxSize = specimen.calibration.pixelSize;
  const unit = specimen.calibration.spaceUnit;
  const areaUnit = `${unit}²`;
  const areaScale = pxSize * pxSize;

  const exclMap = new Map<number, boolean>();
  for (const c of corrections) {
    if (c.specimenId === specimen.id && c.field === 'excluded') {
      exclMap.set(c.objectId, Boolean(c.newValue));
    }
  }

  const objects: DetectionObject[] = [];
  const shapes: OverlayShape[] = [];
  const warnings: string[] = [];

  const subsByParent = new Map<number, SubRegionInstance[]>();
  for (const sr of subOut?.subRegions ?? []) {
    const arr = subsByParent.get(sr.parentId) ?? [];
    arr.push(sr);
    subsByParent.set(sr.parentId, arr);
  }

  for (const parent of parentOut.parents) {
    const subs = subsByParent.get(parent.id) ?? [];
    const subAreaPx = subs.reduce((acc, s) => acc + s.areaPx, 0);
    const ratioPct = parent.areaPx > 0 ? (subAreaPx / parent.areaPx) * 100 : 0;
    const parentAreaPhys = parent.areaPx * areaScale;
    const subAreaPhys = subAreaPx * areaScale;
    const parentDiamPhys = parent.equivRadiusPx * 2 * pxSize;
    const primarySub = subs[0] ?? null;
    const subDiamPhys = primarySub ? primarySub.equivRadiusPx * 2 * pxSize : 0;

    const outOfRange = ratioPct < minRatioPct || ratioPct > maxRatioPct;
    const missingSub = subs.length === 0;
    const excluded = exclMap.get(parent.id) ?? false;

    objects.push({
      id: parent.id,
      kind: 'cell',
      x: Number(parent.cx.toFixed(1)),
      y: Number(parent.cy.toFixed(1)),
      diameter: Number(parentDiamPhys.toFixed(1)),
      area: Number(parentAreaPhys.toFixed(1)),
      circularity: Number(parent.circularity.toFixed(3)),
      subArea: Number(subAreaPhys.toFixed(1)),
      subAreaRatioPct: Number(ratioPct.toFixed(2)),
      subRegionCount: subs.length,
      subDiameter: primarySub ? Number(subDiamPhys.toFixed(1)) : undefined,
      cls: missingSub ? 'irregular' : 'normal',
      flagged: missingSub || outOfRange,
      flagReason: missingSub
        ? '母区内未检测到符合条件的暗色子区'
        : outOfRange
        ? `子区面积占比 ${ratioPct.toFixed(2)}% 超出设定范围 [${minRatioPct}%, ${maxRatioPct}%]`
        : undefined,
      reason:
        subs.length > 0
          ? `核 ${subAreaPhys.toFixed(0)} ${areaUnit} / 细胞 ${parentAreaPhys.toFixed(0)} ${areaUnit} · 占比 ${ratioPct.toFixed(2)}%`
          : `细胞 ${parentAreaPhys.toFixed(0)} ${areaUnit} · 未检出子区`,
      excluded,
      confidence: parent.confidence,
    });

    // Parent Cell Contour (crisp cyan)
    shapes.push({
      id: `hier_parent_${parent.id}`,
      kind: 'polygon',
      points: parent.polygon,
      holes: parent.holes,
      color: excluded ? '#64748B' : '#22D3EE',
      fill: excluded ? undefined : 'rgba(34, 211, 238, 0.08)',
      dashed: excluded,
      label: `#${parent.id} · ${ratioPct.toFixed(2)}%`,
      objectId: parent.id,
      layer: 'segmentation',
    });

    // Constrained Sub-Region Contours inside this parent (crisp amber/gold)
    subs.forEach((sr, sIdx) => {
      const srRatio = parent.areaPx > 0 ? (sr.areaPx / parent.areaPx) * 100 : 0;
      shapes.push({
        id: `hier_sub_${parent.id}_${sIdx + 1}`,
        kind: 'polygon',
        points: sr.polygon,
        holes: sr.holes,
        color: excluded ? '#64748B' : '#F59E0B',
        fill: excluded ? undefined : 'rgba(245, 158, 11, 0.28)',
        dashed: excluded,
        label: `N#${parent.id}${subs.length > 1 ? `.${sIdx + 1}` : ''} ${srRatio.toFixed(2)}%`,
        objectId: parent.id,
        layer: 'segmentation',
      });
    });
  }

  const valid = objects.filter((o) => !o.excluded);
  const totalParentArea = valid.reduce((a, o) => a + (o.area ?? 0), 0);
  const totalSubArea = valid.reduce((a, o) => a + (o.subArea ?? 0), 0);
  const totalSubCount = valid.reduce((a, o) => a + (o.subRegionCount ?? 0), 0);
  const ratios = valid.map((o) => o.subAreaRatioPct ?? 0);
  const meanRatio = ratios.length > 0 ? ratios.reduce((a, b) => a + b, 0) / ratios.length : 0;
  const totalRatio = totalParentArea > 0 ? (totalSubArea / totalParentArea) * 100 : 0;
  const meanParentArea = valid.length > 0 ? totalParentArea / valid.length : 0;
  const meanSubArea = valid.length > 0 ? totalSubArea / valid.length : 0;

  if (valid.length === 0) {
    warnings.push('未检测到符合膜边界条件的母区目标，建议调节最小/最大半径比例或膜检测灵敏度。');
  } else if (totalSubCount === 0) {
    warnings.push('已分割母区边界，但在母区内部未检出暗色子区，建议降低子区对比度阈值 (contrastZ)。');
  }

  const summary: SummaryMetric[] = [
    {
      key: 'mean_sub_area_ratio_pct',
      labelZh: '平均核/细胞面积占比',
      labelEn: 'Mean Sub/Parent Area Ratio',
      value: valid.length > 0 ? Number(meanRatio.toFixed(2)) : null,
      unit: '%',
      detailZh: valid
        .map((o) => `#${o.id}: ${(o.subAreaRatioPct ?? 0).toFixed(2)}%`)
        .join(' · '),
      detailEn: valid
        .map((o) => `#${o.id}: ${(o.subAreaRatioPct ?? 0).toFixed(2)}%`)
        .join(' · '),
      highlight: true,
    },
    {
      key: 'total_sub_area_ratio_pct',
      labelZh: '总核/细胞面积占比',
      labelEn: 'Total Sub/Parent Area Ratio',
      value: valid.length > 0 ? Number(totalRatio.toFixed(2)) : null,
      unit: '%',
      detailZh: `${totalSubArea.toFixed(0)} / ${totalParentArea.toFixed(0)} ${areaUnit}`,
      detailEn: `${totalSubArea.toFixed(0)} / ${totalParentArea.toFixed(0)} ${areaUnit}`,
      highlight: true,
    },
    {
      key: 'parent_count',
      labelZh: '检出母区 (细胞数)',
      labelEn: 'Parent Regions (Cells)',
      value: valid.length,
      unit: '',
    },
    {
      key: 'subregion_count',
      labelZh: '域内子区 (细胞核数)',
      labelEn: 'Nested Sub-Regions (Nuclei)',
      value: totalSubCount,
      unit: '',
    },
    {
      key: 'mean_parent_area',
      labelZh: '平均细胞面积',
      labelEn: 'Mean Parent Area',
      value: valid.length > 0 ? Number(meanParentArea.toFixed(1)) : null,
      unit: areaUnit,
    },
    {
      key: 'mean_sub_area',
      labelZh: '平均细胞核面积',
      labelEn: 'Mean Sub-Region Area',
      value: valid.length > 0 ? Number(meanSubArea.toFixed(1)) : null,
      unit: areaUnit,
    },
  ];

  const summaryMap: Record<string, number | string | null> = {};
  for (const m of summary) summaryMap[m.key] = m.value;

  const histogram =
    ratios.length > 0
      ? {
          labelZh: '各细胞核/细胞面积占比分布',
          labelEn: 'Per-Cell Sub/Parent Area Ratio',
          unit: '%',
          values: ratios,
          mean: Number(meanRatio.toFixed(2)),
        }
      : undefined;

  return {
    objects,
    shapes,
    summary,
    summaryMap,
    warnings,
    histogram,
    qcLabel: `cells=${valid.length}, nuclei=${totalSubCount}, ratio=${meanRatio.toFixed(2)}%`,
  };
}
