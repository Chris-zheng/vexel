import * as XLSX from 'xlsx';
import type { VolumeAnalysisResult, VolumeStackDataset } from '../types/vexel';
import { analyzeVolume3D, type VolumeAnalyzeInput } from './volumeAnalysisCore';
import type { VolumeWorkerRequest, VolumeWorkerResponse } from './volumeAnalysisWorker';

let sharedVolumeWorker: Worker | null = null;

function getVolumeWorker(): Worker | null {
  if (typeof Worker === 'undefined') return null;
  if (!sharedVolumeWorker) {
    try {
      sharedVolumeWorker = new Worker(
        new URL('./volumeAnalysisWorker.ts', import.meta.url),
        { type: 'module' }
      );
    } catch {
      return null;
    }
  }
  return sharedVolumeWorker;
}

export async function runVolumeAnalysis3DAsync(
  stack: VolumeStackDataset,
  options: {
    cellThreshold?: number;
    nucleusDarkRatio?: number;
    minCellVoxels?: number;
  } = {}
): Promise<VolumeAnalysisResult> {
  const input: VolumeAnalyzeInput = {
    volumeId: stack.id,
    data: stack.data,
    width: stack.width,
    height: stack.height,
    depth: stack.depth,
    voxelSizeXY: stack.voxelSizeXY,
    voxelSizeZ: stack.voxelSizeZ,
    spaceUnit: stack.spaceUnit,
    cellThreshold: options.cellThreshold,
    nucleusDarkRatio: options.nucleusDarkRatio ?? 0.65,
    minCellVoxels: options.minCellVoxels ?? 45,
  };

  const worker = getVolumeWorker();
  if (!worker) {
    return analyzeVolume3D(input);
  }

  return new Promise<VolumeAnalysisResult>((resolve, reject) => {
    const jobId = `vol_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const onMsg = (ev: MessageEvent<VolumeWorkerResponse>) => {
      const msg = ev.data;
      if (!msg || msg.jobId !== jobId) return;
      cleanup();
      if (msg.ok && msg.result) {
        resolve(msg.result);
      } else {
        reject(new Error(msg.error || '3D volume analysis worker failed'));
      }
    };
    const onErr = (ev: ErrorEvent) => {
      cleanup();
      reject(new Error(ev.message || '3D worker runtime error'));
    };
    const cleanup = () => {
      worker.removeEventListener('message', onMsg);
      worker.removeEventListener('error', onErr);
    };
    worker.addEventListener('message', onMsg);
    worker.addEventListener('error', onErr);
    worker.postMessage({
      jobId,
      input: {
        ...input,
        data: new Uint8Array(stack.data),
      },
    } satisfies VolumeWorkerRequest);
  });
}

function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function exportVolumeMetricsCsv(
  stack: VolumeStackDataset,
  analysis: VolumeAnalysisResult
): void {
  const u = stack.spaceUnit;
  const headers = [
    'ObjectID',
    'Kind',
    'ParentID',
    'VoxelCount',
    `Volume (${u}^3)`,
    `SurfaceArea (${u}^2)`,
    'Sphericity',
    `EqSphereDiameter (${u})`,
    `PCA_L1 (${u})`,
    `PCA_L2 (${u})`,
    `PCA_L3 (${u})`,
    'Elongation_L1_L2',
    'Flatness_L2_L3',
    'CentroidX_px',
    'CentroidY_px',
    'CentroidZ_slice',
    `NuclearVolume (${u}^3)`,
    `CytoplasmVolume (${u}^3)`,
    'NC_VolumeRatio_Pct',
    'Nuclear_CellVolumeRatio_Pct',
    'MeanIntensity',
  ];
  const rows = analysis.objects.map((o) => [
    o.id,
    o.kind,
    o.parentId ?? '',
    o.voxelCount,
    o.volumePhys.toFixed(3),
    o.surfaceAreaPhys.toFixed(3),
    o.sphericity.toFixed(4),
    o.equivDiameterPhys.toFixed(3),
    o.axesPhys[0].toFixed(2),
    o.axesPhys[1].toFixed(2),
    o.axesPhys[2].toFixed(2),
    o.elongation !== undefined ? o.elongation.toFixed(3) : '',
    o.flatness !== undefined ? o.flatness.toFixed(3) : '',
    o.centroid[0].toFixed(2),
    o.centroid[1].toFixed(2),
    o.centroid[2].toFixed(2),
    o.subVolumePhys !== undefined ? o.subVolumePhys.toFixed(3) : '',
    o.cytoplasmVolumePhys !== undefined ? o.cytoplasmVolumePhys.toFixed(3) : '',
    o.ncVolumePct !== undefined ? o.ncVolumePct.toFixed(2) : '',
    o.ncellVolumePct !== undefined ? o.ncellVolumePct.toFixed(2) : '',
    o.meanIntensity.toFixed(1),
  ]);
  const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
  const bom = new Uint8Array([0xef, 0xbb, 0xbf]);
  downloadBlob(
    `${stack.name.replace(/[^a-zA-Z0-9_-]/g, '_')}_3D_metrology.csv`,
    new Blob([bom, csv], { type: 'text/csv;charset=utf-8' })
  );
}

export function exportVolumeMetricsExcel(
  stack: VolumeStackDataset,
  analysis: VolumeAnalysisResult
): void {
  const wb = XLSX.utils.book_new();
  const u = stack.spaceUnit;
  const summaryRows = [
    { Metric: 'Volume Name', Value: stack.name },
    { Metric: 'Dimensions (W×H×D)', Value: `${stack.width} × ${stack.height} × ${stack.depth}` },
    { Metric: `Voxel Spacing XY (${u})`, Value: stack.voxelSizeXY },
    { Metric: `Voxel Spacing Z (${u})`, Value: stack.voxelSizeZ },
    { Metric: 'Segmented 3D Cells', Value: analysis.cellCount },
    { Metric: 'Nested 3D Nuclei', Value: analysis.nucleusCount },
    { Metric: `Mean Cell Volume (${u}^3)`, Value: Number(analysis.meanCellVolume.toFixed(3)) },
    { Metric: 'Mean Sphericity', Value: Number(analysis.meanSphericity.toFixed(4)) },
    { Metric: 'Mean 3D N/C Volume Ratio (%)', Value: Number(analysis.meanNcVolumePct.toFixed(2)) },
    { Metric: 'Mean 3D N/Cell Volume Ratio (%)', Value: Number(analysis.meanNcellVolumePct.toFixed(2)) },
    { Metric: 'Compute Time (ms)', Value: analysis.durationMs },
  ];
  const objRows = analysis.objects.map((o) => ({
    ID: o.id,
    Kind: o.kind,
    ParentID: o.parentId ?? null,
    Voxels: o.voxelCount,
    [`Volume_${u}3`]: Number(o.volumePhys.toFixed(3)),
    [`SurfaceArea_${u}2`]: Number(o.surfaceAreaPhys.toFixed(3)),
    Sphericity: Number(o.sphericity.toFixed(4)),
    [`EqDiameter_${u}`]: Number(o.equivDiameterPhys.toFixed(3)),
    [`NuclearVol_${u}3`]:
      o.subVolumePhys !== undefined ? Number(o.subVolumePhys.toFixed(3)) : null,
    [`CytoVol_${u}3`]:
      o.cytoplasmVolumePhys !== undefined ? Number(o.cytoplasmVolumePhys.toFixed(3)) : null,
    NC_VolRatio_Pct: o.ncVolumePct !== undefined ? Number(o.ncVolumePct.toFixed(2)) : null,
    N_Cell_VolRatio_Pct:
      o.ncellVolumePct !== undefined ? Number(o.ncellVolumePct.toFixed(2)) : null,
    CentroidX: Number(o.centroid[0].toFixed(2)),
    CentroidY: Number(o.centroid[1].toFixed(2)),
    CentroidZ: Number(o.centroid[2].toFixed(2)),
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(summaryRows), '3D_Summary');
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(objRows), '3D_Objects');
  const buf = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
  downloadBlob(
    `${stack.name.replace(/[^a-zA-Z0-9_-]/g, '_')}_3D_metrology.xlsx`,
    new Blob([buf], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    })
  );
}
