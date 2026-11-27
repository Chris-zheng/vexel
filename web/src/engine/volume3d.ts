import type { VolumeStackDataset } from '../types/vexel';
import { evaluate3DBudget } from './resourceBudget';
import { decodeMultiPageTiffBuffer, decodeTiffBuffer } from './microscopyDecoder';

/** Natural filename sorter so slice_2.png precedes slice_10.png */
function naturalCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

/** Deterministic pseudo-random noise for realistic microscopy cellular texture */
function hash3(x: number, y: number, z: number): number {
  let n = (x * 374761393 + y * 668265263 + z * 2147483647) | 0;
  n = (n ^ (n >>> 13)) * 1274126177;
  return ((n ^ (n >>> 16)) >>> 0) / 4294967295;
}

/**
 * Generates a compact, realistic synthetic 3D Kidney Cell & Dark Nucleus Z-Stack specimen
 * (128×128×48 = 786 KB Uint8) with 5 distinct 3D ellipsoidal cells containing nested dark
 * 3D nuclei and calibrated anisotropic spacing (XY=0.25 µm, Z=0.75 µm) for zero-setup
 * 3D WebGL2 raymarching, ortho-slicing, and 3D N/C volumetric metrology verification.
 */
export function generateSyntheticKidney3DStack(): VolumeStackDataset {
  const width = 128;
  const height = 128;
  const depth = 48;
  const total = width * height * depth;
  const data = new Uint8Array(total);

  interface SynthCell {
    cx: number;
    cy: number;
    cz: number;
    rx: number;
    ry: number;
    rz: number;
    nx: number;
    ny: number;
    nz: number;
    nrx: number;
    nry: number;
    nrz: number;
  }

  // 5 well-separated 3D kidney cells with dark nested nuclei (true 3D N/Cell volume ratios ~15%-22%)
  const cells: SynthCell[] = [
    {
      cx: 28, cy: 30, cz: 23,
      rx: 18, ry: 17, rz: 13,
      nx: 27, ny: 29, nz: 23,
      nrx: 10.0, nry: 9.2, nrz: 7.2,
    },
    {
      cx: 98, cy: 30, cz: 25,
      rx: 19, ry: 18, rz: 14,
      nx: 99, ny: 31, nz: 25,
      nrx: 10.8, nry: 10.2, nrz: 7.8,
    },
    {
      cx: 30, cy: 98, cz: 22,
      rx: 18, ry: 17, rz: 13,
      nx: 29, ny: 99, nz: 22,
      nrx: 9.8, nry: 9.2, nrz: 7.0,
    },
    {
      cx: 98, cy: 96, cz: 27,
      rx: 19, ry: 18, rz: 13,
      nx: 97, ny: 95, nz: 27,
      nrx: 10.8, nry: 10.0, nrz: 7.5,
    },
    {
      cx: 64, cy: 64, cz: 20,
      rx: 15, ry: 14, rz: 10,
      nx: 64, ny: 64, nz: 20,
      nrx: 8.2, nry: 7.8, nrz: 5.8,
    },
  ];

  for (let z = 0; z < depth; z++) {
    const zOff = z * width * height;
    for (let y = 0; y < height; y++) {
      const rowOff = zOff + y * width;
      for (let x = 0; x < width; x++) {
        // Extracellular low-signal background matrix with subtle tissue noise
        const bgNoise = (hash3(x, y, z) - 0.5) * 14;
        let val = 28 + bgNoise;

        for (let ci = 0; ci < cells.length; ci++) {
          const c = cells[ci];
          const dx = (x - c.cx) / c.rx;
          const dy = (y - c.cy) / c.ry;
          const dz = (z - c.cz) / c.rz;
          const r2 = dx * dx + dy * dy + dz * dz;

          if (r2 <= 1.0) {
            const distFromEdge = 1 - Math.sqrt(r2);
            // Bright cytoplasm + defined bright plasma membrane rim (fluorescence / inverted contrast friendly)
            const membraneRim = distFromEdge < 0.14 ? 46 * (1 - distFromEdge / 0.14) : 0;
            const granule = (hash3(x >> 1, y >> 1, z) - 0.5) * 28;
            let cellVal = 158 + 26 * distFromEdge + membraneRim + granule;

            // Check nested 3D nucleus inside cytoplasm (distinct dark core with chromatin clumping)
            const ndx = (x - c.nx) / c.nrx;
            const ndy = (y - c.ny) / c.nry;
            const ndz = (z - c.nz) / c.nrz;
            const nr2 = ndx * ndx + ndy * ndy + ndz * ndz;
            if (nr2 <= 1.0) {
              const nRim = 1 - Math.sqrt(nr2);
              const chromatin = (hash3(x, y, z + 17) - 0.5) * 20;
              // Dark nuclear cavity inside bright cell body (clearly separable in 3D!)
              cellVal = 56 + 18 * nRim + chromatin;
            }
            if (cellVal > val) val = cellVal;
          }
        }

        data[rowOff + x] = Math.max(0, Math.min(255, Math.round(val)));
      }
    }
  }

  return {
    id: 'builtin_kidney_3d_zstack',
    name: 'Kidney_Cells_Nuclei_ZStack_128x128x48',
    width,
    height,
    depth,
    data,
    voxelSizeXY: 0.25,
    voxelSizeZ: 0.75,
    spaceUnit: 'µm',
    source: 'synthetic_kidney_3d',
    originalDimensions: { width, height, depth },
    lodApplied: false,
    descriptionZh:
      '内置 3D 肾细胞-细胞核共聚焦 Z-Stack 演示体数据（5 枚完整三维细胞及内部暗色细胞核，XY=0.25 µm / Z=0.75 µm 各向异性标定）。',
    descriptionEn:
      'Built-in 3D kidney cell & dark nucleus confocal Z-stack specimen (5 complete 3D cells with nested nuclei, XY=0.25 µm / Z=0.75 µm anisotropic calibration).',
  };
}

/** Downsample a 3D Uint8 volume via 2×/k× box averaging so WebGL2 display texture stays safely within limits. */
export function downsampleVolume3D(
  data: Uint8Array,
  width: number,
  height: number,
  depth: number,
  factorXY: number,
  factorZ = 1
): { width: number; height: number; depth: number; data: Uint8Array } {
  const fXY = Math.max(1, Math.round(factorXY));
  const fZ = Math.max(1, Math.round(factorZ));
  const nw = Math.max(16, Math.floor(width / fXY));
  const nh = Math.max(16, Math.floor(height / fXY));
  const nd = Math.max(4, Math.floor(depth / fZ));
  const out = new Uint8Array(nw * nh * nd);

  for (let z = 0; z < nd; z++) {
    const z0 = z * fZ;
    const z1 = Math.min(depth, z0 + fZ);
    for (let y = 0; y < nh; y++) {
      const y0 = y * fXY;
      const y1 = Math.min(height, y0 + fXY);
      for (let x = 0; x < nw; x++) {
        const x0 = x * fXY;
        const x1 = Math.min(width, x0 + fXY);
        let acc = 0;
        let cnt = 0;
        for (let sz = z0; sz < z1; sz++) {
          const zOff = sz * width * height;
          for (let sy = y0; sy < y1; sy++) {
            const row = zOff + sy * width;
            for (let sx = x0; sx < x1; sx++) {
              acc += data[row + sx];
              cnt++;
            }
          }
        }
        out[z * nw * nh + y * nw + x] = Math.round(acc / Math.max(1, cnt));
      }
    }
  }

  return { width: nw, height: nh, depth: nd, data: out };
}

function isTiffFileName(name: string): boolean {
  return /\.(ome\.)?tiff?$/i.test(name) || /\.lsm$/i.test(name);
}

/** Normalize an array of 2D Float32 grayscale slices into a budget-safe Uint8 VolumeStackDataset. */
function packFloatSlicesToVolume(
  slices: Float32Array[],
  origW: number,
  origH: number,
  origD: number,
  meta: {
    baseName: string;
    voxelSizeXY: number;
    voxelSizeZ: number;
    spaceUnit: VolumeStackDataset['spaceUnit'];
    formatNote?: string;
    zStrideAlreadyApplied?: number;
  }
): { volume: VolumeStackDataset; warningZh?: string; warningEn?: string } {
  const budget = evaluate3DBudget(origW, origH, origD, '3d_analyze');
  let targetW = origW;
  let targetH = origH;
  let xyScale = 1;
  if (budget.tier === 'lod_guarded' || budget.tier === 'offload_gpu_required' || origW > 512 || origH > 512) {
    const maxXY = 512;
    const scale = Math.min(1, maxXY / Math.max(origW, origH));
    targetW = Math.max(16, Math.round(origW * scale));
    targetH = Math.max(16, Math.round(origH * scale));
    xyScale = origW / targetW;
  }

  const maxDepth = 128;
  const extraZStride = slices.length > maxDepth ? Math.ceil(slices.length / maxDepth) : 1;
  const sampledSlices =
    extraZStride > 1 ? slices.filter((_, idx) => idx % extraZStride === 0) : slices;
  const targetD = sampledSlices.length;
  const rawData = new Uint8Array(targetW * targetH * targetD);
  const hist = new Uint32Array(256);

  for (let z = 0; z < targetD; z++) {
    const src = sampledSlices[z];
    const zOff = z * targetW * targetH;
    if (targetW === origW && targetH === origH) {
      for (let i = 0; i < targetW * targetH; i++) {
        const v = Math.max(0, Math.min(255, Math.round(src[i])));
        rawData[zOff + i] = v;
        hist[v]++;
      }
    } else {
      for (let y = 0; y < targetH; y++) {
        const sy = Math.min(origH - 1, Math.floor((y + 0.5) * (origH / targetH)));
        const srcRow = sy * origW;
        const dstRow = zOff + y * targetW;
        for (let x = 0; x < targetW; x++) {
          const sx = Math.min(origW - 1, Math.floor((x + 0.5) * (origW / targetW)));
          const v = Math.max(0, Math.min(255, Math.round(src[srcRow + sx])));
          rawData[dstRow + x] = v;
          hist[v]++;
        }
      }
    }
  }

  // P0.5 - P99.5 contrast stretch
  const totalVx = rawData.length;
  const loTarget = totalVx * 0.005;
  const hiTarget = totalVx * 0.995;
  let cum = 0;
  let pLo = 0;
  let pHi = 255;
  for (let v = 0; v < 256; v++) {
    cum += hist[v];
    if (cum >= loTarget) {
      pLo = v;
      break;
    }
  }
  cum = 0;
  for (let v = 0; v < 256; v++) {
    cum += hist[v];
    if (cum >= hiTarget) {
      pHi = v;
      break;
    }
  }
  const span = Math.max(12, pHi - pLo);
  for (let i = 0; i < totalVx; i++) {
    const norm = ((rawData[i] - pLo) / span) * 255;
    rawData[i] = norm < 0 ? 0 : norm > 255 ? 255 : Math.round(norm);
  }

  const lodApplied = targetW !== origW || targetH !== origH || targetD !== origD;
  const totalZStride = (meta.zStrideAlreadyApplied ?? 1) * extraZStride;

  const volume: VolumeStackDataset = {
    id: `zstack_${Date.now().toString(36)}`,
    name: `${meta.baseName}_${targetW}x${targetH}x${targetD}`,
    width: targetW,
    height: targetH,
    depth: targetD,
    data: rawData,
    voxelSizeXY: Number((meta.voxelSizeXY * xyScale).toFixed(4)),
    voxelSizeZ: Number((meta.voxelSizeZ * extraZStride).toFixed(4)),
    spaceUnit: meta.spaceUnit,
    source: 'uploaded_zstack',
    originalDimensions: { width: origW, height: origH, depth: origD },
    lodApplied,
    descriptionZh: `${meta.formatNote ? `${meta.formatNote} ` : ''}Z-Stack 体数据（原始 ${origW}×${origH}×${origD}，工作体素 ${targetW}×${targetH}×${targetD}${totalZStride > 1 ? ` · Z 抽样 ${totalZStride}×` : ''}）。`,
    descriptionEn: `${meta.formatNote ? `${meta.formatNote} ` : ''}Z-Stack volume (original ${origW}×${origH}×${origD}, working ${targetW}×${targetH}×${targetD}${totalZStride > 1 ? ` · Z stride ${totalZStride}×` : ''}).`,
  };

  return {
    volume,
    warningZh: lodApplied
      ? `检测到原始体数据 ${origW}×${origH}×${origD} 超过浏览器实时上限，已自动安全降采样至 ${targetW}×${targetH}×${targetD} 并按比例同步校正体素间距。`
      : undefined,
    warningEn: lodApplied
      ? `Original stack ${origW}×${origH}×${origD} exceeded browser real-time cap; automatically downsampled to ${targetW}×${targetH}×${targetD} with scaled voxel spacing.`
      : undefined,
  };
}

/**
 * Decode a single Multi-Page TIFF / OME-TIFF Z-Stack ArrayBuffer directly into a VolumeStackDataset.
 * Works cleanly in both browser and Node.js (Vitest), extracting OME-XML / ImageJ XY & Z spacing.
 */
export async function loadZStackFromTiffBuffer(
  buffer: ArrayBuffer,
  fileName: string,
  options: { voxelSizeXY?: number; voxelSizeZ?: number; spaceUnit?: VolumeStackDataset['spaceUnit'] } = {}
): Promise<{ volume: VolumeStackDataset; warningZh?: string; warningEn?: string }> {
  const decoded = await decodeMultiPageTiffBuffer(buffer, { maxPages: 128 });
  if (decoded.depth < 3) {
    throw new Error(
      `该 TIFF 文件仅包含 ${decoded.depth} 页切片，构建 3D 体数据至少需要 3 个 Z 切片 / TIFF contains only ${decoded.depth} page(s); at least 3 Z-slices are required for 3D volume reconstruction.`
    );
  }

  const baseName = fileName.replace(/(\.ome)?\.tiff?$/i, '') || 'Tiff_ZStack';
  const zStrideAlreadyApplied =
    decoded.totalPagesInFile > decoded.depth
      ? Math.ceil(decoded.totalPagesInFile / decoded.depth)
      : 1;

  return packFloatSlicesToVolume(decoded.slices, decoded.width, decoded.height, decoded.totalPagesInFile, {
    baseName,
    voxelSizeXY: decoded.voxelSizeXY ?? options.voxelSizeXY ?? 0.25,
    voxelSizeZ:
      decoded.voxelSizeZ ??
      Number(((options.voxelSizeZ ?? 0.75) * zStrideAlreadyApplied).toFixed(4)),
    spaceUnit: decoded.spaceUnit ?? options.spaceUnit ?? 'µm',
    formatNote: `${decoded.formatLabel} (${decoded.bitDepth}-bit)`,
    zStrideAlreadyApplied,
  });
}

/**
 * Load either:
 *  1) A single Multi-Page TIFF / OME-TIFF Z-Stack file (.tif/.tiff/.ome.tif)
 *  2) A multi-file TIFF slice sequence (≥3 single-page .tif files)
 *  3) A multi-file PNG/JPEG/WebP/BMP slice sequence (≥3 slices)
 * into a compact Uint8 VolumeStackDataset with automatic 3D memory budget LOD protection.
 */
export async function loadZStackFromFiles(
  files: FileList | File[],
  options: { voxelSizeXY?: number; voxelSizeZ?: number; spaceUnit?: VolumeStackDataset['spaceUnit'] } = {}
): Promise<{ volume: VolumeStackDataset; warningZh?: string; warningEn?: string }> {
  const rawList = Array.from(files);

  // Case 1: Single Multi-Page TIFF / OME-TIFF file
  if (rawList.length === 1 && isTiffFileName(rawList[0].name)) {
    const buf = await rawList[0].arrayBuffer();
    return loadZStackFromTiffBuffer(buf, rawList[0].name, options);
  }

  // Case 2: Multi-file TIFF slice sequence
  const tiffFiles = rawList
    .filter((f) => isTiffFileName(f.name))
    .sort((a, b) => naturalCompare(a.name, b.name));
  if (tiffFiles.length >= 3) {
    const maxDepth = 128;
    const zStride = tiffFiles.length > maxDepth ? Math.ceil(tiffFiles.length / maxDepth) : 1;
    const sampled = tiffFiles.filter((_, idx) => idx % zStride === 0);
    const first = await decodeTiffBuffer(await sampled[0].arrayBuffer());
    const slices: Float32Array[] = [first.gray];
    for (let i = 1; i < sampled.length; i++) {
      const dec = await decodeTiffBuffer(await sampled[i].arrayBuffer());
      if (dec.width === first.width && dec.height === first.height) {
        slices.push(dec.gray);
      }
    }
    const baseName =
      tiffFiles[0].name.replace(/(\.ome)?\.tiff?$/i, '').replace(/[_-]?\d+$/, '') || 'Tiff_Sequence';
    return packFloatSlicesToVolume(slices, first.width, first.height, tiffFiles.length, {
      baseName,
      voxelSizeXY: first.metadataCalibration?.pixelSize ?? options.voxelSizeXY ?? 0.25,
      voxelSizeZ: Number(((options.voxelSizeZ ?? 0.75) * zStride).toFixed(4)),
      spaceUnit: first.metadataCalibration?.spaceUnit ?? options.spaceUnit ?? 'µm',
      formatNote: `${first.formatLabel} Seq`,
      zStrideAlreadyApplied: zStride,
    });
  }

  // Case 3: Standard raster image sequence (PNG / JPEG / WebP / BMP)
  const list = rawList
    .filter((f) => f.type.startsWith('image/') || /\.(png|jpe?g|webp|bmp)$/i.test(f.name))
    .sort((a, b) => naturalCompare(a.name, b.name));

  if (list.length < 3) {
    throw new Error(
      '请选择单个多页 TIFF / OME-TIFF 文件，或至少 3 张连续 Z-Stack 图像切片（TIFF/PNG/JPG/WebP）/ Select a multi-page TIFF / OME-TIFF stack or at least 3 sequential slice images.'
    );
  }

  const decodeBitmap = async (file: File): Promise<ImageBitmap> => {
    return createImageBitmap(file);
  };

  const firstBmp = await decodeBitmap(list[0]);
  const origW = firstBmp.width;
  const origH = firstBmp.height;
  const origD = list.length;

  const budget = evaluate3DBudget(origW, origH, origD, '3d_analyze');
  // Enforce safe browser ceiling: cap stored XY to <= 512 if budget exceeds tiled_safe
  let targetW = origW;
  let targetH = origH;
  let xyScale = 1;
  if (budget.tier === 'lod_guarded' || budget.tier === 'offload_gpu_required' || origW > 512 || origH > 512) {
    const maxXY = 512;
    const scale = Math.min(1, maxXY / Math.max(origW, origH));
    targetW = Math.max(32, Math.round(origW * scale));
    targetH = Math.max(32, Math.round(origH * scale));
    xyScale = origW / targetW;
  }

  const maxDepth = 128;
  const zStride = origD > maxDepth ? Math.ceil(origD / maxDepth) : 1;
  const sampledFiles = list.filter((_, idx) => idx % zStride === 0);
  const targetD = sampledFiles.length;

  const canvas = document.createElement('canvas');
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    firstBmp.close();
    throw new Error('Failed to acquire 2D canvas context for Z-stack decoding.');
  }

  const rawData = new Uint8Array(targetW * targetH * targetD);
  const hist = new Uint32Array(256);

  for (let z = 0; z < targetD; z++) {
    const bmp = z === 0 ? firstBmp : await decodeBitmap(sampledFiles[z]);
    ctx.drawImage(bmp, 0, 0, targetW, targetH);
    if (z > 0) bmp.close();
    const imgData = ctx.getImageData(0, 0, targetW, targetH).data;
    const zOff = z * targetW * targetH;
    for (let i = 0; i < targetW * targetH; i++) {
      const r = imgData[i * 4];
      const g = imgData[i * 4 + 1];
      const b = imgData[i * 4 + 2];
      const lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      rawData[zOff + i] = lum;
      hist[lum]++;
    }
  }
  firstBmp.close();

  // P0.5 - P99.5 percentile normalization for crisp 3D contrast
  const totalVx = rawData.length;
  const loTarget = totalVx * 0.005;
  const hiTarget = totalVx * 0.995;
  let cum = 0;
  let pLo = 0;
  let pHi = 255;
  for (let v = 0; v < 256; v++) {
    cum += hist[v];
    if (cum >= loTarget) {
      pLo = v;
      break;
    }
  }
  cum = 0;
  for (let v = 0; v < 256; v++) {
    cum += hist[v];
    if (cum >= hiTarget) {
      pHi = v;
      break;
    }
  }
  const span = Math.max(12, pHi - pLo);
  for (let i = 0; i < totalVx; i++) {
    const norm = ((rawData[i] - pLo) / span) * 255;
    rawData[i] = norm < 0 ? 0 : norm > 255 ? 255 : Math.round(norm);
  }

  const lodApplied = targetW !== origW || targetH !== origH || targetD !== origD;
  const baseName = list[0].name.replace(/\.[^.]+$/, '').replace(/[_-]?\d+$/, '') || 'Uploaded_ZStack';

  const volume: VolumeStackDataset = {
    id: `zstack_${Date.now().toString(36)}`,
    name: `${baseName}_${targetW}x${targetH}x${targetD}`,
    width: targetW,
    height: targetH,
    depth: targetD,
    data: rawData,
    voxelSizeXY: Number(((options.voxelSizeXY ?? 0.25) * xyScale).toFixed(4)),
    voxelSizeZ: Number(((options.voxelSizeZ ?? 0.75) * zStride).toFixed(4)),
    spaceUnit: options.spaceUnit ?? 'µm',
    source: 'uploaded_zstack',
    originalDimensions: { width: origW, height: origH, depth: origD },
    lodApplied,
    descriptionZh: `用户导入 Z-Stack 序列（原始 ${origW}×${origH}×${origD}，工作体素 ${targetW}×${targetH}×${targetD}）。`,
    descriptionEn: `Uploaded Z-stack sequence (original ${origW}×${origH}×${origD}, working volume ${targetW}×${targetH}×${targetD}).`,
  };

  return {
    volume,
    warningZh: lodApplied
      ? `检测到原始体数据 ${origW}×${origH}×${origD} 超过浏览器实时上限，已自动安全降采样至 ${targetW}×${targetH}×${targetD} 并按比例同步校正体素间距。`
      : undefined,
    warningEn: lodApplied
      ? `Original stack ${origW}×${origH}×${origD} exceeded browser real-time cap; automatically downsampled to ${targetW}×${targetH}×${targetD} with scaled voxel spacing.`
      : undefined,
  };
}

/** Compute 2.5D areal surface roughness parameters (Sa, Sq, Sz) from normalized heightfield [0..1]. */
export function computeSurfaceRoughness(
  heights: Float32Array,
  zPhysicalScale: number
): { sa: number; sq: number; sz: number; meanH: number } {
  const n = heights.length;
  if (n === 0) return { sa: 0, sq: 0, sz: 0, meanH: 0 };
  let sum = 0;
  let minH = Infinity;
  let maxH = -Infinity;
  for (let i = 0; i < n; i++) {
    const h = heights[i] * zPhysicalScale;
    sum += h;
    if (h < minH) minH = h;
    if (h > maxH) maxH = h;
  }
  const meanH = sum / n;
  let absSum = 0;
  let sqSum = 0;
  for (let i = 0; i < n; i++) {
    const d = heights[i] * zPhysicalScale - meanH;
    absSum += Math.abs(d);
    sqSum += d * d;
  }
  return {
    sa: Number((absSum / n).toFixed(3)),
    sq: Number(Math.sqrt(sqSum / n).toFixed(3)),
    sz: Number((maxH - minH).toFixed(3)),
    meanH: Number(meanH.toFixed(3)),
  };
}
