import { useState } from 'react';
import { ChevronDown, ChevronUp, Database, Globe, Layers } from 'lucide-react';
import type { Lang, VolumeStackDataset } from '../types/vexel';

interface OmeZarrLevelView {
  path: string;
  levelIndex: number;
  shape: [number, number, number];
  voxelCount: number;
  voxelSizeXY: number;
  voxelSizeZ: number;
  spaceUnit: VolumeStackDataset['spaceUnit'];
  downscaleXY: number;
  downscaleZ: number;
  fitsBrowser3DBudget: boolean;
}

interface OmeZarrInspectionView {
  name: string;
  specVersion: string;
  axes: string[];
  levels: OmeZarrLevelView[];
  recommendedLevel: OmeZarrLevelView;
  outOfCoreRequired: boolean;
  summaryZh: string;
  summaryEn: string;
}

const DEMO_OME_ZARR_V04 = {
  multiscales: [
    {
      version: '0.4',
      name: 'Kidney_LightSheet_OME_Zarr_Pyramid',
      axes: [
        { name: 'z', type: 'space', unit: 'micrometer' },
        { name: 'y', type: 'space', unit: 'micrometer' },
        { name: 'x', type: 'space', unit: 'micrometer' },
      ],
      datasets: [
        {
          path: '0',
          coordinateTransformations: [{ type: 'scale', scale: [0.5, 0.16, 0.16] }],
        },
        {
          path: '1',
          coordinateTransformations: [{ type: 'scale', scale: [0.5, 0.32, 0.32] }],
        },
        {
          path: '2',
          coordinateTransformations: [{ type: 'scale', scale: [1.0, 0.64, 0.64] }],
        },
        {
          path: '3',
          coordinateTransformations: [{ type: 'scale', scale: [2.0, 1.28, 1.28] }],
        },
      ],
    },
  ],
};

const DEMO_SHAPES: Record<string, number[]> = {
  '0': [512, 4096, 4096],
  '1': [512, 2048, 2048],
  '2': [256, 1024, 1024],
  '3': [128, 512, 512],
};

interface OmeZarrBridgePanelProps {
  lang: Lang;
  onApplyCalibration: (patch: {
    voxelSizeXY: number;
    voxelSizeZ: number;
    spaceUnit: VolumeStackDataset['spaceUnit'];
  }) => void;
  onNotify?: (msg: string) => void;
}

export function OmeZarrBridgePanel({
  lang,
  onApplyCalibration,
  onNotify,
}: OmeZarrBridgePanelProps) {
  const [open, setOpen] = useState(false);
  const [remoteUrl, setRemoteUrl] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inspection, setInspection] = useState<OmeZarrInspectionView | null>(null);

  const inspectDemo = async () => {
    setBusy(true);
    setError(null);
    try {
      const resp = await fetch('/api/v1/zarr/inspect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'metadata',
          metadata: DEMO_OME_ZARR_V04,
          shapesByPath: DEMO_SHAPES,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || !data?.ok) throw new Error(data?.error || `HTTP ${resp.status}`);
      setInspection(data.inspection as OmeZarrInspectionView);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const inspectUrl = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!remoteUrl.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const resp = await fetch('/api/v1/zarr/inspect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'remote_https', url: remoteUrl.trim() }),
      });
      const data = await resp.json();
      if (!resp.ok || !data?.ok) throw new Error(data?.error || `HTTP ${resp.status}`);
      setInspection(data.inspection as OmeZarrInspectionView);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const applyRecommended = () => {
    if (!inspection) return;
    const lv = inspection.recommendedLevel;
    onApplyCalibration({
      voxelSizeXY: lv.voxelSizeXY,
      voxelSizeZ: lv.voxelSizeZ,
      spaceUnit: lv.spaceUnit,
    });
    onNotify?.(
      lang === 'zh'
        ? `已同步 OME-Zarr 金字塔层 "${lv.path}" 物理标定：XY=${lv.voxelSizeXY}${lv.spaceUnit}, Z=${lv.voxelSizeZ}${lv.spaceUnit}`
        : `Synced OME-Zarr level "${lv.path}" calibration: XY=${lv.voxelSizeXY}${lv.spaceUnit}, Z=${lv.voxelSizeZ}${lv.spaceUnit}`
    );
  };

  return (
    <div className="rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-[#0B0F1A] overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full px-3 py-2 flex items-center justify-between text-[11.5px] font-semibold text-slate-800 dark:text-slate-200 hover:bg-slate-100/70 dark:hover:bg-slate-900/60 cursor-pointer"
      >
        <span className="flex items-center gap-1.5">
          <Database className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
          <span>{lang === 'zh' ? 'OME-Zarr 多尺度金字塔桥接 (超大图)' : 'OME-Zarr Multiscale Pyramid Bridge'}</span>
        </span>
        {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>

      {open && (
        <div className="px-3 pb-3 pt-1.5 border-t border-slate-200 dark:border-slate-800 space-y-2.5 text-[11px]">
          <form onSubmit={inspectUrl} className="flex gap-1.5">
            <input
              type="url"
              value={remoteUrl}
              onChange={(e) => setRemoteUrl(e.target.value)}
              placeholder="https://uk1s3.embassy.ebi.ac.uk/.../0.zarr"
              className="flex-1 h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[10.5px] text-slate-800 dark:text-slate-100"
            />
            <button
              type="submit"
              disabled={busy || !remoteUrl.trim()}
              className="h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 font-medium flex items-center gap-1 cursor-pointer disabled:opacity-50"
            >
              <Globe className="w-3 h-3" />
              <span>{lang === 'zh' ? '探测' : 'Probe'}</span>
            </button>
          </form>

          <button
            type="button"
            disabled={busy}
            onClick={() => void inspectDemo()}
            className="w-full h-7 px-2.5 rounded border border-cyan-500/40 bg-cyan-500/10 hover:bg-cyan-500/20 text-blue-700 dark:text-cyan-300 font-medium flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <Layers className="w-3 h-3" />
            <span>
              {lang === 'zh'
                ? '演示：探测 4096×4096×512 OME-Zarr 光片金字塔'
                : 'Demo: Probe 4096×4096×512 OME-Zarr Pyramid'}
            </span>
          </button>

          {error && (
            <div className="p-2 rounded border border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300">
              {error}
            </div>
          )}

          {inspection && (
            <div className="space-y-2 pt-1">
              <div className="text-[10.5px] text-slate-600 dark:text-slate-300 leading-relaxed">
                {lang === 'zh' ? inspection.summaryZh : inspection.summaryEn}
              </div>

              <div className="rounded border border-slate-200 dark:border-slate-800 overflow-hidden font-mono text-[10px]">
                {inspection.levels.map((lv) => {
                  const isRec = lv.path === inspection.recommendedLevel.path;
                  return (
                    <div
                      key={lv.path}
                      className={`px-2 py-1 flex items-center justify-between border-b last:border-b-0 border-slate-200/70 dark:border-slate-800/70 ${
                        isRec ? 'bg-cyan-500/12 text-blue-800 dark:text-cyan-200 font-semibold' : 'text-slate-500'
                      }`}
                    >
                      <span>
                        L{lv.path}: {lv.shape[2]}×{lv.shape[1]}×{lv.shape[0]}
                      </span>
                      <span>
                        XY {lv.voxelSizeXY} / Z {lv.voxelSizeZ} {lv.spaceUnit}
                      </span>
                      <span>
                        {isRec
                          ? lang === 'zh'
                            ? '推荐层'
                            : 'Auto'
                          : lv.fitsBrowser3DBudget
                          ? 'OK'
                          : 'Out-of-Core'}
                      </span>
                    </div>
                  );
                })}
              </div>

              <button
                type="button"
                onClick={applyRecommended}
                className="w-full h-7 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 font-medium cursor-pointer"
              >
                {lang === 'zh'
                  ? `同步推荐层 L${inspection.recommendedLevel.path} 物理体素标定`
                  : `Apply L${inspection.recommendedLevel.path} Voxel Calibration`}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
