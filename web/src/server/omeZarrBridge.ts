import fs from 'fs';
import path from 'path';
import { isBlockedPrivateOrMetadataHost } from './securityGuardrails';

export type OmeSpaceUnit = 'µm' | 'nm' | 'mm' | 'px';

export interface OmeZarrPyramidLevel {
  path: string;
  levelIndex: number;
  shape: [number, number, number]; // [depthZ, heightY, widthX]
  voxelCount: number;
  voxelSizeXY: number;
  voxelSizeZ: number;
  spaceUnit: OmeSpaceUnit;
  downscaleXY: number;
  downscaleZ: number;
  fitsBrowser3DBudget: boolean;
}

export interface OmeZarrInspectionResult {
  name: string;
  specVersion: string;
  axes: string[];
  levels: OmeZarrPyramidLevel[];
  recommendedLevel: OmeZarrPyramidLevel;
  sourceKind: 'metadata_json' | 'local_zarr' | 'remote_https';
  outOfCoreRequired: boolean;
  summaryZh: string;
  summaryEn: string;
}

const DEFAULT_BROWSER_3D_VOXEL_BUDGET = 512 * 512 * 128; // ~33.55M voxels

function normalizeSpaceUnit(unit?: string): OmeSpaceUnit {
  if (!unit) return 'µm';
  const u = unit.toLowerCase();
  if (u === 'micrometer' || u === 'micrometre' || u === 'um' || u === 'µm' || u === 'μm') return 'µm';
  if (u === 'nanometer' || u === 'nanometre' || u === 'nm') return 'nm';
  if (u === 'millimeter' || u === 'millimetre' || u === 'mm') return 'mm';
  if (u === 'px' || u === 'pixel' || u === 'pixels') return 'px';
  return 'µm';
}

/**
 * Pure OME-NGFF (v0.4 / Zarr v3) multiscale metadata parser + browser 3D pyramid level selector.
 * Extracts calibrated anisotropic XY/Z spacing and selects the highest-resolution pyramid level
 * that stays safely within WebGL2 / WebGPU 3D texture memory budgets.
 */
export function parseOmeZarrMetadata(
  rawMeta: Record<string, unknown>,
  options: {
    shapesByPath?: Record<string, number[]>;
    maxVoxels?: number;
    sourceKind?: OmeZarrInspectionResult['sourceKind'];
  } = {}
): OmeZarrInspectionResult {
  const maxVoxels = options.maxVoxels ?? DEFAULT_BROWSER_3D_VOXEL_BUDGET;

  // Support both Zarr v2 (.zattrs -> multiscales) and Zarr v3 (zarr.json -> attributes.ome.multiscales)
  const attrs = (rawMeta.attributes as Record<string, unknown> | undefined) ?? rawMeta;
  const omeBlock = (attrs.ome as Record<string, unknown> | undefined) ?? attrs;
  const multiscales = Array.isArray(omeBlock.multiscales)
    ? (omeBlock.multiscales as Array<Record<string, unknown>>)
    : [];

  if (multiscales.length === 0) {
    throw new Error('Invalid OME-Zarr metadata: missing "multiscales" array in .zattrs / zarr.json.');
  }

  const ms = multiscales[0];
  const name = typeof ms.name === 'string' && ms.name.trim() ? ms.name.trim().slice(0, 80) : 'OME-Zarr Volume';
  const specVersion =
    typeof ms.version === 'string'
      ? ms.version
      : typeof rawMeta.zarr_format === 'number'
      ? `zarr-v${rawMeta.zarr_format}`
      : '0.4';

  const rawAxes = Array.isArray(ms.axes) ? (ms.axes as Array<Record<string, unknown> | string>) : [];
  const axesNames: string[] = [];
  let spaceUnit: OmeSpaceUnit = 'µm';
  for (const ax of rawAxes) {
    if (typeof ax === 'string') {
      axesNames.push(ax.toLowerCase());
    } else if (ax && typeof ax === 'object') {
      const axName = typeof ax.name === 'string' ? ax.name.toLowerCase() : '';
      if (axName) axesNames.push(axName);
      if (typeof ax.unit === 'string' && (axName === 'x' || axName === 'y' || axName === 'z')) {
        spaceUnit = normalizeSpaceUnit(ax.unit);
      }
    }
  }
  const axes = axesNames.length > 0 ? axesNames : ['z', 'y', 'x'];
  const idxZ = axes.indexOf('z');
  const idxY = axes.indexOf('y');
  const idxX = axes.indexOf('x');

  const datasets = Array.isArray(ms.datasets)
    ? (ms.datasets as Array<Record<string, unknown>>)
    : [];
  if (datasets.length === 0) {
    throw new Error('OME-Zarr multiscales entry contains no resolution datasets.');
  }

  const extractScale = (ds: Record<string, unknown>, levelIdx: number): [number, number, number] => {
    const transforms = Array.isArray(ds.coordinateTransformations)
      ? (ds.coordinateTransformations as Array<Record<string, unknown>>)
      : [];
    const scaleObj = transforms.find((t) => t?.type === 'scale' && Array.isArray(t.scale));
    if (scaleObj && Array.isArray(scaleObj.scale)) {
      const sc = scaleObj.scale.map((n) => Number(n));
      const sz = idxZ >= 0 && Number.isFinite(sc[idxZ]) ? sc[idxZ] : 0.8 * Math.pow(2, Math.max(0, levelIdx - 1));
      const sy = idxY >= 0 && Number.isFinite(sc[idxY]) ? sc[idxY] : 0.25 * Math.pow(2, levelIdx);
      const sx = idxX >= 0 && Number.isFinite(sc[idxX]) ? sc[idxX] : sy;
      return [Math.max(1e-4, sz), Math.max(1e-4, sy), Math.max(1e-4, sx)];
    }
    const factor = Math.pow(2, levelIdx);
    return [0.8 * factor, 0.25 * factor, 0.25 * factor];
  };

  const baseScale = extractScale(datasets[0], 0);
  const shapesByPath = options.shapesByPath ?? {};

  const levels: OmeZarrPyramidLevel[] = datasets.map((ds, idx) => {
    const levelPath = typeof ds.path === 'string' && ds.path.trim() ? ds.path.trim() : String(idx);
    const [sz, sy, sx] = extractScale(ds, idx);
    const downscaleXY = Math.max(1, Number((sx / Math.max(1e-6, baseScale[2])).toFixed(2)));
    const downscaleZ = Math.max(1, Number((sz / Math.max(1e-6, baseScale[0])).toFixed(2)));

    // Resolve shape from attached .zarray / zarr.json metadata, inline dataset.shape, or scale ratio
    const inlineShape = Array.isArray(ds.shape) ? ds.shape.map(Number) : undefined;
    const rawShape = shapesByPath[levelPath] ?? inlineShape;
    let dZ = Math.max(8, Math.round(128 / downscaleZ));
    let dY = Math.max(32, Math.round(1024 / downscaleXY));
    let dX = Math.max(32, Math.round(1024 / downscaleXY));

    if (rawShape && rawShape.length >= 3) {
      if (idxZ >= 0 && idxY >= 0 && idxX >= 0 && rawShape.length === axes.length) {
        dZ = Math.max(1, Math.round(rawShape[idxZ]));
        dY = Math.max(1, Math.round(rawShape[idxY]));
        dX = Math.max(1, Math.round(rawShape[idxX]));
      } else {
        dZ = Math.max(1, Math.round(rawShape[rawShape.length - 3]));
        dY = Math.max(1, Math.round(rawShape[rawShape.length - 2]));
        dX = Math.max(1, Math.round(rawShape[rawShape.length - 1]));
      }
    }

    const voxelCount = dZ * dY * dX;
    return {
      path: levelPath,
      levelIndex: idx,
      shape: [dZ, dY, dX],
      voxelCount,
      voxelSizeXY: Number(((sx + sy) * 0.5).toFixed(4)),
      voxelSizeZ: Number(sz.toFixed(4)),
      spaceUnit,
      downscaleXY,
      downscaleZ,
      fitsBrowser3DBudget: voxelCount <= maxVoxels,
    };
  });

  // Pick the highest-resolution pyramid level fitting inside browser 3D budget; else coarsest level
  const recommendedLevel =
    levels.find((lv) => lv.fitsBrowser3DBudget) ?? levels[levels.length - 1];
  const outOfCoreRequired = !levels[0].fitsBrowser3DBudget;

  return {
    name,
    specVersion,
    axes,
    levels,
    recommendedLevel,
    sourceKind: options.sourceKind ?? 'metadata_json',
    outOfCoreRequired,
    summaryZh: outOfCoreRequired
      ? `全分辨率层 (${levels[0].shape[2]}×${levels[0].shape[1]}×${levels[0].shape[0]}) 超出浏览器 3D 预算，已自动选择金字塔层 "${recommendedLevel.path}" (${recommendedLevel.shape[2]}×${recommendedLevel.shape[1]}×${recommendedLevel.shape[0]}，XY=${recommendedLevel.voxelSizeXY}${spaceUnit}, Z=${recommendedLevel.voxelSizeZ}${spaceUnit})。`
      : `OME-Zarr 全分辨率层 "${recommendedLevel.path}" (${recommendedLevel.shape[2]}×${recommendedLevel.shape[1]}×${recommendedLevel.shape[0]}) 处于浏览器 3D 安全预算内，已提取标定间距。`,
    summaryEn: outOfCoreRequired
      ? `Full-res level (${levels[0].shape[2]}×${levels[0].shape[1]}×${levels[0].shape[0]}) exceeds browser 3D budget; auto-selected pyramid level "${recommendedLevel.path}" (${recommendedLevel.shape[2]}×${recommendedLevel.shape[1]}×${recommendedLevel.shape[0]}, XY=${recommendedLevel.voxelSizeXY}${spaceUnit}, Z=${recommendedLevel.voxelSizeZ}${spaceUnit}).`
      : `Full-res OME-Zarr level "${recommendedLevel.path}" (${recommendedLevel.shape[2]}×${recommendedLevel.shape[1]}×${recommendedLevel.shape[0]}) fits within browser 3D budget.`,
  };
}

/**
 * Inspect a local OME-Zarr directory strictly confined inside `zarrRootDir` (`VEXEL_ZARR_ROOT`).
 * Uses `path.resolve` + trailing `path.sep` verification to block any directory traversal.
 */
export function inspectLocalOmeZarr(
  relPath: string,
  zarrRootDir: string
): OmeZarrInspectionResult {
  if (!zarrRootDir || !zarrRootDir.trim()) {
    throw new Error('Local OME-Zarr root is not configured (set VEXEL_ZARR_ROOT on server).');
  }
  const cleanRel = relPath.trim().replace(/^[/\\]+/, '');
  if (!cleanRel || cleanRel.includes('\0')) {
    throw new Error('Invalid relative OME-Zarr dataset path.');
  }

  const rootResolved = path.resolve(zarrRootDir);
  const targetResolved = path.resolve(rootResolved, cleanRel);
  if (targetResolved !== rootResolved && !targetResolved.startsWith(rootResolved + path.sep)) {
    throw new Error('Security check rejected path outside configured VEXEL_ZARR_ROOT.');
  }

  const zattrsPath = path.join(targetResolved, '.zattrs');
  const zarrV3Path = path.join(targetResolved, 'zarr.json');

  let rawMeta: Record<string, unknown> | null = null;
  if (fs.existsSync(zattrsPath)) {
    rawMeta = JSON.parse(fs.readFileSync(zattrsPath, 'utf8')) as Record<string, unknown>;
  } else if (fs.existsSync(zarrV3Path)) {
    rawMeta = JSON.parse(fs.readFileSync(zarrV3Path, 'utf8')) as Record<string, unknown>;
  } else {
    throw new Error('Target directory does not contain .zattrs (Zarr v2) or zarr.json (Zarr v3).');
  }

  // Probe child dataset shapes safely within targetResolved
  const shapesByPath: Record<string, number[]> = {};
  const attrs = (rawMeta.attributes as Record<string, unknown> | undefined) ?? rawMeta;
  const omeBlock = (attrs.ome as Record<string, unknown> | undefined) ?? attrs;
  const multiscales = Array.isArray(omeBlock.multiscales)
    ? (omeBlock.multiscales as Array<Record<string, unknown>>)
    : [];
  const datasets = Array.isArray(multiscales[0]?.datasets)
    ? (multiscales[0].datasets as Array<Record<string, unknown>>)
    : [];

  for (const ds of datasets.slice(0, 8)) {
    const p = typeof ds.path === 'string' ? ds.path.trim() : '';
    if (!p || p.includes('..') || path.isAbsolute(p)) continue;
    const levelDir = path.resolve(targetResolved, p);
    if (!levelDir.startsWith(targetResolved + path.sep)) continue;

    for (const metaFile of ['.zarray', 'zarr.json']) {
      const candidate = path.join(levelDir, metaFile);
      if (fs.existsSync(candidate)) {
        try {
          const parsed = JSON.parse(fs.readFileSync(candidate, 'utf8')) as { shape?: number[] };
          if (Array.isArray(parsed.shape)) {
            shapesByPath[p] = parsed.shape;
            break;
          }
        } catch {
          // ignore unreadable child metadata
        }
      }
    }
  }

  return parseOmeZarrMetadata(rawMeta, { shapesByPath, sourceKind: 'local_zarr' });
}

/**
 * Inspect a public HTTPS OME-Zarr store (e.g., IDR / OpenOrganelle / BioImage Archive)
 * while blocking loopback, private RFC-1918, and cloud metadata endpoints.
 */
export async function inspectRemoteOmeZarr(storeUrl: string): Promise<OmeZarrInspectionResult> {
  const trimmed = storeUrl.trim().replace(/\/+$/, '');
  if (!/^https:\/\//i.test(trimmed)) {
    throw new Error('Remote OME-Zarr URL must use https:// protocol.');
  }
  if (isBlockedPrivateOrMetadataHost(trimmed)) {
    throw new Error('Remote OME-Zarr host blocked: private, loopback, or metadata addresses are not permitted.');
  }

  const fetchJson = async (url: string): Promise<Record<string, unknown> | null> => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    try {
      const r = await fetch(url, {
        signal: controller.signal,
        redirect: 'error', // refuse unvalidated redirects to private IPs
        headers: { Accept: 'application/json' },
      });
      if (!r.ok) return null;
      const text = await r.text();
      if (text.length > 256 * 1024) {
        throw new Error('Remote OME-Zarr metadata exceeds 256 KB limit.');
      }
      return JSON.parse(text) as Record<string, unknown>;
    } finally {
      clearTimeout(timer);
    }
  };

  let rawMeta = await fetchJson(`${trimmed}/.zattrs`);
  if (!rawMeta) {
    rawMeta = await fetchJson(`${trimmed}/zarr.json`);
  }
  if (!rawMeta) {
    throw new Error('Unable to fetch .zattrs or zarr.json from remote OME-Zarr store.');
  }

  const shapesByPath: Record<string, number[]> = {};
  const attrs = (rawMeta.attributes as Record<string, unknown> | undefined) ?? rawMeta;
  const omeBlock = (attrs.ome as Record<string, unknown> | undefined) ?? attrs;
  const multiscales = Array.isArray(omeBlock.multiscales)
    ? (omeBlock.multiscales as Array<Record<string, unknown>>)
    : [];
  const datasets = Array.isArray(multiscales[0]?.datasets)
    ? (multiscales[0].datasets as Array<Record<string, unknown>>)
    : [];

  for (const ds of datasets.slice(0, 6)) {
    const p = typeof ds.path === 'string' ? ds.path.trim().replace(/^\/+/, '') : '';
    if (!p || !/^[a-zA-Z0-9._/-]{1,48}$/.test(p) || p.includes('..')) continue;
    const arrMeta = (await fetchJson(`${trimmed}/${p}/.zarray`)) ?? (await fetchJson(`${trimmed}/${p}/zarr.json`));
    if (arrMeta && Array.isArray(arrMeta.shape)) {
      shapesByPath[p] = arrMeta.shape.map(Number);
    }
  }

  return parseOmeZarrMetadata(rawMeta, { shapesByPath, sourceKind: 'remote_https' });
}
