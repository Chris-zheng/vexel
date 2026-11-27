import type { VolumeAnalysisResult, VolumetricObjectMetric } from '../types/vexel';

export interface VolumeAnalyzeInput {
  volumeId: string;
  data: Uint8Array;
  width: number;
  height: number;
  depth: number;
  voxelSizeXY: number;
  voxelSizeZ: number;
  spaceUnit: 'nm' | 'µm' | 'mm' | 'px';
  /** Intensity threshold for outer 3D parent cells (0..255, <=0 for auto Otsu) */
  cellThreshold?: number;
  /** Relative darkness factor inside parent cell to detect nested 3D nuclei (0.35..0.85) */
  nucleusDarkRatio?: number;
  /** Minimum 3D cell voxel count */
  minCellVoxels?: number;
}

/** Otsu threshold over 8-bit 3D volume histogram */
function computeOtsu3D(data: Uint8Array): number {
  const hist = new Uint32Array(256);
  for (let i = 0; i < data.length; i++) hist[data[i]]++;
  const total = data.length;
  let sumAll = 0;
  for (let i = 0; i < 256; i++) sumAll += i * hist[i];
  let wB = 0;
  let sumB = 0;
  let bestVar = -1;
  let bestT = 80;
  for (let t = 1; t < 255; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const diff = mB - mF;
    const between = wB * wF * diff * diff;
    if (between > bestVar) {
      bestVar = between;
      bestT = t;
    }
  }
  return bestT;
}

/**
 * Fill enclosed 3D interior cavities (such as dark nuclei inside bright cell shells)
 * via 6-connected exterior flood fill from the volume boundary so parent cells are solid
 * before 3D connected-component labeling.
 */
function fillEnclosed3DCavities(
  binary: Uint8Array,
  width: number,
  height: number,
  depth: number
): Uint8Array {
  const total = width * height * depth;
  const plane = width * height;
  const outside = new Uint8Array(total);
  const queue = new Int32Array(total);
  let head = 0;
  let tail = 0;

  const enqueue = (idx: number) => {
    if (binary[idx] === 0 && outside[idx] === 0) {
      outside[idx] = 1;
      queue[tail++] = idx;
    }
  };

  // Seed all 6 faces of the 3D bounding box
  for (let z = 0; z < depth; z++) {
    const zOff = z * plane;
    for (let x = 0; x < width; x++) {
      enqueue(zOff + x);
      enqueue(zOff + (height - 1) * width + x);
    }
    for (let y = 0; y < height; y++) {
      enqueue(zOff + y * width);
      enqueue(zOff + y * width + (width - 1));
    }
  }
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      enqueue(row + x);
      enqueue((depth - 1) * plane + row + x);
    }
  }

  while (head < tail) {
    const idx = queue[head++];
    const z = Math.floor(idx / plane);
    const rem = idx - z * plane;
    const y = Math.floor(rem / width);
    const x = rem - y * width;

    if (x > 0) enqueue(idx - 1);
    if (x + 1 < width) enqueue(idx + 1);
    if (y > 0) enqueue(idx - width);
    if (y + 1 < height) enqueue(idx + width);
    if (z > 0) enqueue(idx - plane);
    if (z + 1 < depth) enqueue(idx + plane);
  }

  const filled = new Uint8Array(total);
  for (let i = 0; i < total; i++) {
    filled[i] = outside[i] === 0 ? 1 : 0;
  }
  return filled;
}

/**
 * Memory-bounded 26-connectivity 3D Connected-Component Labeling (2-pass Union-Find)
 * using compact Uint16Array labels (half the memory of Int32Array).
 */
function labelConnectedComponents3D(
  mask: Uint8Array,
  width: number,
  height: number,
  depth: number,
  minVoxels: number
): { labels: Uint16Array; count: number } {
  const total = width * height * depth;
  const plane = width * height;
  const labels = new Uint16Array(total);

  // Union-Find equivalence table (max 65534 provisional labels)
  const MAX_LABELS = 65534;
  const parent = new Uint16Array(MAX_LABELS + 1);
  let nextLabel = 1;

  const findRoot = (a: number): number => {
    let r = a;
    while (parent[r] !== r) r = parent[r];
    let cur = a;
    while (parent[cur] !== r) {
      const nxt = parent[cur];
      parent[cur] = r;
      cur = nxt;
    }
    return r;
  };

  const union = (a: number, b: number): number => {
    const ra = findRoot(a);
    const rb = findRoot(b);
    if (ra === rb) return ra;
    if (ra < rb) {
      parent[rb] = ra;
      return ra;
    }
    parent[ra] = rb;
    return rb;
  };

  // Pass 1: Z-slab forward scan checking 13 causal neighbors in 26-connectivity
  for (let z = 0; z < depth; z++) {
    const zOff = z * plane;
    for (let y = 0; y < height; y++) {
      const row = zOff + y * width;
      for (let x = 0; x < width; x++) {
        const idx = row + x;
        if (mask[idx] === 0) continue;

        let minNb = 0;
        const checkNb = (nbIdx: number) => {
          const l = labels[nbIdx];
          if (l > 0) {
            minNb = minNb === 0 ? l : union(minNb, l);
          }
        };

        // Current Z slice causal neighbors (4)
        if (x > 0) checkNb(idx - 1);
        if (y > 0) {
          checkNb(idx - width);
          if (x > 0) checkNb(idx - width - 1);
          if (x + 1 < width) checkNb(idx - width + 1);
        }
        // Previous Z slice neighbors (9)
        if (z > 0) {
          const prev = idx - plane;
          checkNb(prev);
          if (x > 0) checkNb(prev - 1);
          if (x + 1 < width) checkNb(prev + 1);
          if (y > 0) {
            checkNb(prev - width);
            if (x > 0) checkNb(prev - width - 1);
            if (x + 1 < width) checkNb(prev - width + 1);
          }
          if (y + 1 < height) {
            checkNb(prev + width);
            if (x > 0) checkNb(prev + width - 1);
            if (x + 1 < width) checkNb(prev + width + 1);
          }
        }

        if (minNb === 0) {
          if (nextLabel < MAX_LABELS) {
            const lbl = nextLabel++;
            parent[lbl] = lbl;
            labels[idx] = lbl;
          }
        } else {
          labels[idx] = minNb;
        }
      }
    }
  }

  // Count voxels per canonical root
  const rootCounts = new Uint32Array(nextLabel);
  for (let i = 0; i < total; i++) {
    const l = labels[i];
    if (l > 0) {
      const r = findRoot(l);
      labels[i] = r;
      rootCounts[r]++;
    }
  }

  // Compact contiguous IDs (1..count) filtering out dust below minVoxels
  const remap = new Uint16Array(nextLabel);
  let count = 0;
  for (let r = 1; r < nextLabel; r++) {
    if (rootCounts[r] >= minVoxels) {
      remap[r] = ++count;
    }
  }

  for (let i = 0; i < total; i++) {
    const l = labels[i];
    if (l > 0) labels[i] = remap[l];
  }

  return { labels, count };
}

/**
 * Exact 3x3 symmetric matrix eigenvalues via cyclic Jacobi rotations.
 * Returns eigenvalues sorted descending: [lambda1 >= lambda2 >= lambda3].
 */
function symmetric3x3Eigenvalues(
  m00: number,
  m11: number,
  m22: number,
  m01: number,
  m02: number,
  m12: number
): [number, number, number] {
  const a = [
    [m00, m01, m02],
    [m01, m11, m12],
    [m02, m12, m22],
  ];

  for (let iter = 0; iter < 24; iter++) {
    // Pick largest magnitude off-diagonal entry among (0,1), (0,2), (1,2)
    let p = 0;
    let q = 1;
    let maxOff = Math.abs(a[0][1]);
    if (Math.abs(a[0][2]) > maxOff) {
      p = 0;
      q = 2;
      maxOff = Math.abs(a[0][2]);
    }
    if (Math.abs(a[1][2]) > maxOff) {
      p = 1;
      q = 2;
      maxOff = Math.abs(a[1][2]);
    }
    if (maxOff < 1e-12) break;

    const app = a[p][p];
    const aqq = a[q][q];
    const apq = a[p][q];
    const tau = (aqq - app) / (2 * apq);
    const t =
      tau >= 0
        ? 1 / (tau + Math.sqrt(1 + tau * tau))
        : -1 / (-tau + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    a[p][p] = app - t * apq;
    a[q][q] = aqq + t * apq;
    a[p][q] = 0;
    a[q][p] = 0;

    const r = 3 - p - q;
    const arp = a[r][p];
    const arq = a[r][q];
    a[r][p] = c * arp - s * arq;
    a[p][r] = a[r][p];
    a[r][q] = s * arp + c * arq;
    a[q][r] = a[r][q];
  }

  const evs = [Math.max(0, a[0][0]), Math.max(0, a[1][1]), Math.max(0, a[2][2])];
  evs.sort((x, y) => y - x);
  return [evs[0], evs[1], evs[2]];
}

/**
 * Full 3D Hierarchical Volumetric Metrology Engine:
 * 1. Segments 3D parent cells (outer shell + enclosed cavity fill + 26-connected 3D CCA)
 * 2. Inside each 3D parent cell volume, detects nested dark 3D nuclei
 * 3. Computes anisotropic physical Volume (V), Crofton Surface Area (A), Sphericity (Psi),
 *    Equivalent Spherical Diameter, 3D Centroid, PCA Principal Axes [L1, L2, L3] + Elongation + Flatness,
 *    and 3D Nuclear-to-Cytoplasmic Volume Ratio.
 */
export function analyzeVolume3D(input: VolumeAnalyzeInput): VolumeAnalysisResult {
  const t0 = performance.now();
  const { volumeId, data, width, height, depth, voxelSizeXY, voxelSizeZ, spaceUnit } = input;
  const total = width * height * depth;
  const plane = width * height;
  const minCellVoxels = Math.max(30, input.minCellVoxels ?? Math.round(total * 0.00045));
  const nucleusDarkRatio = input.nucleusDarkRatio ?? 0.68;

  // Step 1: Segment outer bright/cellular structure & fill enclosed nuclear cavities
  const autoT = input.cellThreshold && input.cellThreshold > 0 ? input.cellThreshold : computeOtsu3D(data);
  const rawBinary = new Uint8Array(total);
  for (let i = 0; i < total; i++) {
    rawBinary[i] = data[i] >= autoT ? 1 : 0;
  }
  const solidCellsMask = fillEnclosed3DCavities(rawBinary, width, height, depth);
  const { labels: cellLabels, count: cellCount } = labelConnectedComponents3D(
    solidCellsMask,
    width,
    height,
    depth,
    minCellVoxels
  );

  // Per-cell accumulators (including 2nd-order moments for rotation-invariant 3D PCA principal axes)
  const vxCount = new Uint32Array(cellCount + 1);
  const sumX = new Float64Array(cellCount + 1);
  const sumY = new Float64Array(cellCount + 1);
  const sumZ = new Float64Array(cellCount + 1);
  const sumXX = new Float64Array(cellCount + 1);
  const sumYY = new Float64Array(cellCount + 1);
  const sumZZ = new Float64Array(cellCount + 1);
  const sumXY = new Float64Array(cellCount + 1);
  const sumXZ = new Float64Array(cellCount + 1);
  const sumYZ = new Float64Array(cellCount + 1);
  const sumInt = new Float64Array(cellCount + 1);
  const cytopIntSum = new Float64Array(cellCount + 1);
  const cytopCnt = new Uint32Array(cellCount + 1);
  const minX = new Int32Array(cellCount + 1).fill(width);
  const maxX = new Int32Array(cellCount + 1).fill(0);
  const minY = new Int32Array(cellCount + 1).fill(height);
  const maxY = new Int32Array(cellCount + 1).fill(0);
  const minZ = new Int32Array(cellCount + 1).fill(depth);
  const maxZ = new Int32Array(cellCount + 1).fill(0);
  // Exposed face counts along X, Y, Z axes for anisotropic Crofton surface area
  const faceX = new Uint32Array(cellCount + 1);
  const faceY = new Uint32Array(cellCount + 1);
  const faceZ = new Uint32Array(cellCount + 1);

  for (let z = 0; z < depth; z++) {
    const zOff = z * plane;
    for (let y = 0; y < height; y++) {
      const row = zOff + y * width;
      for (let x = 0; x < width; x++) {
        const idx = row + x;
        const id = cellLabels[idx];
        if (id === 0) continue;

        vxCount[id]++;
        sumX[id] += x;
        sumY[id] += y;
        sumZ[id] += z;
        sumXX[id] += x * x;
        sumYY[id] += y * y;
        sumZZ[id] += z * z;
        sumXY[id] += x * y;
        sumXZ[id] += x * z;
        sumYZ[id] += y * z;
        const val = data[idx];
        sumInt[id] += val;
        if (rawBinary[idx] === 1) {
          cytopIntSum[id] += val;
          cytopCnt[id]++;
        }

        if (x < minX[id]) minX[id] = x;
        if (x > maxX[id]) maxX[id] = x;
        if (y < minY[id]) minY[id] = y;
        if (y > maxY[id]) maxY[id] = y;
        if (z < minZ[id]) minZ[id] = z;
        if (z > maxZ[id]) maxZ[id] = z;

        // 6-neighbor boundary faces
        if (x === 0 || cellLabels[idx - 1] !== id) faceX[id]++;
        if (x + 1 === width || cellLabels[idx + 1] !== id) faceX[id]++;
        if (y === 0 || cellLabels[idx - width] !== id) faceY[id]++;
        if (y + 1 === height || cellLabels[idx + width] !== id) faceY[id]++;
        if (z === 0 || cellLabels[idx - plane] !== id) faceZ[id]++;
        if (z + 1 === depth || cellLabels[idx + plane] !== id) faceZ[id]++;
      }
    }
  }

  // Step 2: Inside each 3D parent cell, detect dark nested 3D nuclear sub-regions
  const nucleusMask = new Uint8Array(total);
  const cellCytoMean = new Float32Array(cellCount + 1);
  for (let id = 1; id <= cellCount; id++) {
    cellCytoMean[id] =
      cytopCnt[id] > 0 ? cytopIntSum[id] / cytopCnt[id] : sumInt[id] / Math.max(1, vxCount[id]);
  }

  for (let i = 0; i < total; i++) {
    const id = cellLabels[i];
    if (id === 0) continue;
    const nucCut = cellCytoMean[id] * nucleusDarkRatio;
    if (data[i] < nucCut) {
      nucleusMask[i] = 1;
    }
  }

  const minNucVoxels = Math.max(12, Math.round(minCellVoxels * 0.08));
  const { labels: nucLabels, count: nucRawCount } = labelConnectedComponents3D(
    nucleusMask,
    width,
    height,
    depth,
    minNucVoxels
  );

  // Assign each 3D nucleus component to its enclosing parent cell
  const nucVx = new Uint32Array(nucRawCount + 1);
  const nucParent = new Uint16Array(nucRawCount + 1);
  const nucSumX = new Float64Array(nucRawCount + 1);
  const nucSumY = new Float64Array(nucRawCount + 1);
  const nucSumZ = new Float64Array(nucRawCount + 1);

  const labelVolume = new Uint8Array(total);
  for (let z = 0; z < depth; z++) {
    const zOff = z * plane;
    for (let y = 0; y < height; y++) {
      const row = zOff + y * width;
      for (let x = 0; x < width; x++) {
        const idx = row + x;
        const cid = cellLabels[idx];
        const nid = nucLabels[idx];
        if (cid > 0) {
          labelVolume[idx] = nid > 0 ? 2 : 1;
        }
        if (nid > 0 && cid > 0) {
          nucVx[nid]++;
          nucParent[nid] = cid;
          nucSumX[nid] += x;
          nucSumY[nid] += y;
          nucSumZ[nid] += z;
        }
      }
    }
  }

  // Aggregate nuclear volumes per parent cell
  const cellNucVx = new Uint32Array(cellCount + 1);
  const cellNucCount = new Uint16Array(cellCount + 1);
  let validNucleusCount = 0;
  for (let nid = 1; nid <= nucRawCount; nid++) {
    const pid = nucParent[nid];
    if (pid > 0 && nucVx[nid] >= minNucVoxels) {
      // Discard nuclei that occupy >85% of the cell (artifact)
      if (nucVx[nid] < vxCount[pid] * 0.85) {
        cellNucVx[pid] += nucVx[nid];
        cellNucCount[pid]++;
        validNucleusCount++;
      }
    }
  }

  const voxelVol = voxelSizeXY * voxelSizeXY * voxelSizeZ;
  // Crofton factor (~0.6667) corrects Manhattan staircase overestimation of smooth 3D surface area
  const croftonFactor = 0.667;
  const areaXY = voxelSizeXY * voxelSizeXY;
  const areaXZ = voxelSizeXY * voxelSizeZ;

  const objects: VolumetricObjectMetric[] = [];
  for (let id = 1; id <= cellCount; id++) {
    const nv = vxCount[id];
    if (nv === 0) continue;
    const volumePhys = nv * voxelVol;
    const rawSurface = faceZ[id] * areaXY + (faceX[id] + faceY[id]) * areaXZ;
    const surfaceAreaPhys = Math.max(1e-4, rawSurface * croftonFactor);
    const sphericity = Math.min(
      1,
      (Math.pow(Math.PI, 1 / 3) * Math.pow(6 * volumePhys, 2 / 3)) / surfaceAreaPhys
    );
    const equivDiameterPhys = Math.cbrt((6 * volumePhys) / Math.PI);

    const subVx = cellNucVx[id];
    const subVolumePhys = subVx * voxelVol;
    const cytoplasmVolumePhys = Math.max(1e-6, volumePhys - subVolumePhys);
    const ncVolumeRatio = subVx > 0 ? subVolumePhys / cytoplasmVolumePhys : 0;
    const ncVolumePct = ncVolumeRatio * 100;
    const ncellVolumePct = subVx > 0 ? (subVolumePhys / Math.max(1e-6, volumePhys)) * 100 : 0;

    const meanX = sumX[id] / nv;
    const meanY = sumY[id] / nv;
    const meanZ = sumZ[id] / nv;

    // Anisotropic 3x3 spatial covariance matrix in physical units (including discrete voxel box variance s^2/12)
    const sx = voxelSizeXY;
    const sy = voxelSizeXY;
    const sz = voxelSizeZ;
    const covXX = Math.max(0, sumXX[id] / nv - meanX * meanX) * sx * sx + (sx * sx) / 12;
    const covYY = Math.max(0, sumYY[id] / nv - meanY * meanY) * sy * sy + (sy * sy) / 12;
    const covZZ = Math.max(0, sumZZ[id] / nv - meanZ * meanZ) * sz * sz + (sz * sz) / 12;
    const covXY = (sumXY[id] / nv - meanX * meanY) * sx * sy;
    const covXZ = (sumXZ[id] / nv - meanX * meanZ) * sx * sz;
    const covYZ = (sumYZ[id] / nv - meanY * meanZ) * sy * sz;

    const [lam1, lam2, lam3] = symmetric3x3Eigenvalues(
      covXX,
      covYY,
      covZZ,
      covXY,
      covXZ,
      covYZ
    );
    // For a uniform 3D ellipsoid with semi-axes (a, b, c), eigenvalue lambda_i = a_i^2 / 5 -> full axis L_i = 2 * sqrt(5 * lambda_i)
    const axis1 = 2 * Math.sqrt(5 * Math.max(0, lam1));
    const axis2 = 2 * Math.sqrt(5 * Math.max(0, lam2));
    const axis3 = 2 * Math.sqrt(5 * Math.max(0, lam3));
    const elongation = axis1 / Math.max(1e-4, axis2);
    const flatness = axis2 / Math.max(1e-4, axis3);

    objects.push({
      id,
      kind: 'cell_3d',
      voxelCount: nv,
      volumePhys: Number(volumePhys.toFixed(2)),
      surfaceAreaPhys: Number(surfaceAreaPhys.toFixed(2)),
      sphericity: Number(sphericity.toFixed(3)),
      equivDiameterPhys: Number(equivDiameterPhys.toFixed(2)),
      centroid: [
        Number(meanX.toFixed(1)),
        Number(meanY.toFixed(1)),
        Number(meanZ.toFixed(1)),
      ],
      bboxVoxel: {
        x0: minX[id],
        y0: minY[id],
        z0: minZ[id],
        x1: maxX[id],
        y1: maxY[id],
        z1: maxZ[id],
      },
      axesPhys: [
        Number(axis1.toFixed(2)),
        Number(axis2.toFixed(2)),
        Number(axis3.toFixed(2)),
      ],
      elongation: Number(elongation.toFixed(3)),
      flatness: Number(flatness.toFixed(3)),
      meanIntensity: Number((sumInt[id] / nv).toFixed(1)),
      subRegionCount: cellNucCount[id],
      subVolumePhys: Number(subVolumePhys.toFixed(2)),
      cytoplasmVolumePhys: Number(Math.max(0, volumePhys - subVolumePhys).toFixed(2)),
      ncVolumeRatio: Number(ncVolumeRatio.toFixed(4)),
      ncVolumePct: Number(ncVolumePct.toFixed(2)),
      ncellVolumePct: Number(ncellVolumePct.toFixed(2)),
    });
  }

  objects.sort((a, b) => b.volumePhys - a.volumePhys);

  const nObjs = Math.max(1, objects.length);
  const meanCellVolume = objects.reduce((acc, o) => acc + o.volumePhys, 0) / nObjs;
  const meanSphericity = objects.reduce((acc, o) => acc + o.sphericity, 0) / nObjs;
  const withNuc = objects.filter((o) => (o.subVolumePhys ?? 0) > 0);
  const meanNcVolumePct =
    withNuc.length > 0
      ? withNuc.reduce((acc, o) => acc + (o.ncVolumePct ?? 0), 0) / withNuc.length
      : 0;
  const meanNcellVolumePct =
    withNuc.length > 0
      ? withNuc.reduce((acc, o) => acc + (o.ncellVolumePct ?? 0), 0) / withNuc.length
      : 0;

  return {
    volumeId,
    executedAt: new Date().toISOString(),
    durationMs: Math.round(performance.now() - t0),
    cellCount: objects.length,
    nucleusCount: validNucleusCount,
    meanCellVolume: Number(meanCellVolume.toFixed(2)),
    meanSphericity: Number(meanSphericity.toFixed(3)),
    meanNcVolumePct: Number(meanNcVolumePct.toFixed(2)),
    meanNcellVolumePct: Number(meanNcellVolumePct.toFixed(2)),
    spaceUnit,
    objects,
    labelVolume,
  };
}
