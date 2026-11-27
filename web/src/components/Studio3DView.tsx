import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Box,
  Cpu,
  FileSpreadsheet,
  FolderUp,
  Play,
  Sparkles,
  Table,
} from 'lucide-react';
import type {
  Lang,
  OccludedDisplayMode,
  RoiRect,
  RunResult,
  Specimen,
  VolumeAnalysisResult,
  VolumeStackDataset,
} from '../types/vexel';
import { generateSyntheticKidney3DStack, loadZStackFromFiles } from '../engine/volume3d';
import { evaluate3DBudget } from '../engine/resourceBudget';
import {
  exportVolumeMetricsCsv,
  exportVolumeMetricsExcel,
  runVolumeAnalysis3DAsync,
} from '../engine/volumeAnalysisRunner';
import { WebGL3DViewport, type Viewport3DMode } from './WebGL3DViewport';
import { NcRatioHistogramPanel } from './NcRatioHistogramPanel';
import { OmeZarrBridgePanel } from './OmeZarrBridgePanel';

interface Studio3DViewProps {
  lang: Lang;
  specimen: Specimen;
  runResult: RunResult | null;
  roi: RoiRect | null;
  selected2DObjectId?: number | null;
  occludedDisplayMode?: OccludedDisplayMode;
  onAnalysisChange?: (analysis: VolumeAnalysisResult | null) => void;
  onNotify?: (msg: string) => void;
}

export function Studio3DView({
  lang,
  specimen,
  runResult,
  roi,
  selected2DObjectId,
  occludedDisplayMode,
  onAnalysisChange,
  onNotify,
}: Studio3DViewProps) {
  const [mode, setMode] = useState<Viewport3DMode>('volume3d');
  const [volume, setVolume] = useState<VolumeStackDataset>(() => generateSyntheticKidney3DStack());
  const [analysis, setAnalysis] = useState<VolumeAnalysisResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedObjId, setSelectedObjId] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // 3D Analysis controls
  const [autoThreshold, setAutoThreshold] = useState(true);
  const [manualThreshold, setManualThreshold] = useState(95);
  const [nucleusDarkRatio, setNucleusDarkRatio] = useState(0.65);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const budget3d = useMemo(
    () => evaluate3DBudget(volume.width, volume.height, volume.depth, '3d_analyze'),
    [volume.width, volume.height, volume.depth]
  );

  // Run initial 3D volumetric analysis automatically for the built-in synthetic 3D kidney specimen
  const runAnalysis = async (targetVol: VolumeStackDataset = volume) => {
    setErrorMsg(null);
    setAnalyzing(true);
    try {
      const res = await runVolumeAnalysis3DAsync(targetVol, {
        cellThreshold: autoThreshold ? 0 : manualThreshold,
        nucleusDarkRatio,
        minCellVoxels: 45,
      });
      setAnalysis(res);
      onAnalysisChange?.(res);
      setSelectedObjId(null);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setAnalyzing(false);
    }
  };

  useEffect(() => {
    void runAnalysis(volume);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volume.id]);

  const handleLoadBuiltin3D = () => {
    const synth = generateSyntheticKidney3DStack();
    setVolume(synth);
    setMode('volume3d');
    onNotify?.(
      lang === 'zh'
        ? '已载入内置 3D 肾细胞-细胞核 Z-Stack 演示体数据 (128×128×48)'
        : 'Loaded built-in 3D Kidney Cell-Nucleus Z-Stack (128×128×48)'
    );
  };

  const handleUploadZStack = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    setErrorMsg(null);
    try {
      const { volume: loaded, warningZh, warningEn } = await loadZStackFromFiles(files, {
        voxelSizeXY: volume.voxelSizeXY,
        voxelSizeZ: volume.voxelSizeZ,
        spaceUnit: volume.spaceUnit,
      });
      setVolume(loaded);
      setMode('volume3d');
      const warn = lang === 'zh' ? warningZh : warningEn;
      onNotify?.(
        warn ||
          (lang === 'zh'
            ? `已载入 Z-Stack 体数据：${loaded.width}×${loaded.height}×${loaded.depth}`
            : `Loaded Z-Stack volume: ${loaded.width}×${loaded.height}×${loaded.depth}`)
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const updateVoxelSpacing = (patch: Partial<Pick<VolumeStackDataset, 'voxelSizeXY' | 'voxelSizeZ' | 'spaceUnit'>>) => {
    const next: VolumeStackDataset = { ...volume, ...patch };
    setVolume(next);
  };

  const parentCells = useMemo(
    () => (analysis ? analysis.objects.filter((o) => o.kind === 'cell_3d') : []),
    [analysis]
  );

  const budgetTone =
    budget3d.tier === 'optimal'
      ? 'border-emerald-500/35 bg-emerald-500/8 text-emerald-700 dark:text-emerald-300'
      : budget3d.tier === 'tiled_safe'
      ? 'border-cyan-500/35 bg-cyan-500/8 text-blue-700 dark:text-cyan-300'
      : 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300';

  return (
    <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-hidden bg-slate-50 dark:bg-[#0B0E17]">
      {/* Left: Full WebGL2 Dual-Mode 3D Viewport */}
      <WebGL3DViewport
        lang={lang}
        mode={mode}
        onChangeMode={setMode}
        volume={volume}
        volumeAnalysis={analysis}
        selectedVolumeObjectId={selectedObjId}
        onSelectVolumeObject={setSelectedObjId}
        specimen={specimen}
        runResult={runResult}
        roi={roi}
        selected2DObjectId={selected2DObjectId}
        occludedDisplayMode={occludedDisplayMode}
      />

      {/* Right: 3D Z-Stack Loader, Anisotropic Calibration, Volumetric Metrology & 3D N/C Histogram */}
      <aside className="w-full lg:w-[416px] shrink-0 border-t lg:border-t-0 lg:border-l border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E1320] overflow-y-auto p-4 space-y-4">
        {/* Data Source & Z-Stack Uploader */}
        <div className="space-y-2.5 pb-3 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5 text-[14px] font-semibold text-slate-900 dark:text-slate-100">
              <Box className="w-4 h-4 text-blue-600 dark:text-cyan-400" />
              <span>{lang === 'zh' ? '3D 体数据与表面实验室' : '3D Volume & Surface Lab'}</span>
            </div>
            <button
              type="button"
              onClick={handleLoadBuiltin3D}
              className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11px] font-medium text-slate-700 dark:text-slate-200 flex items-center gap-1 cursor-pointer"
            >
              <Sparkles className="w-3 h-3 text-cyan-500" />
              <span>{lang === 'zh' ? '3D 肾细胞演示' : '3D Kidney Demo'}</span>
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".tif,.tiff,.ome.tif,.ome.tiff,.lsm,image/tiff,image/png,image/jpeg,image/webp,image/bmp"
            onChange={handleUploadZStack}
            className="hidden"
          />
          <button
            type="button"
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
            className="w-full h-8 px-3 rounded-md border border-dashed border-slate-300 dark:border-slate-700 hover:border-blue-500 dark:hover:border-cyan-400 bg-slate-50/70 dark:bg-[#0B0F1A] text-[12px] font-medium text-slate-700 dark:text-slate-200 flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <FolderUp className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
            <span>
              {uploading
                ? lang === 'zh'
                  ? '正在解码 Z-Stack / 多页 TIFF 并执行内存保护…'
                  : 'Decoding Z-Stack / Multi-Page TIFF…'
                : lang === 'zh'
                ? '导入多页 TIFF / OME-TIFF 或切片序列 (PNG/JPG)'
                : 'Import Multi-Page TIFF / OME-TIFF or Slice Sequence'}
            </span>
          </button>

          <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
            {lang === 'zh' ? volume.descriptionZh : volume.descriptionEn}
          </p>
        </div>

        {/* 3D Resource Budget Guardrail */}
        <div className={`p-2.5 rounded-md border text-[11px] space-y-1 ${budgetTone}`}>
          <div className="flex items-center justify-between font-mono font-semibold">
            <span className="flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5" />
              <span>
                {volume.width}×{volume.height}×{volume.depth} ({(budget3d.elementCount / 1e6).toFixed(2)}M vx)
              </span>
            </span>
            <span>~{budget3d.estimatedPeakMB} MB</span>
          </div>
          <div className="text-[10.5px] opacity-90">
            {lang === 'zh' ? budget3d.recommendationZh : budget3d.recommendationEn}
          </div>
        </div>

        {/* G-11 OME-Zarr Multiscale Pyramid Backend Bridge */}
        <OmeZarrBridgePanel
          lang={lang}
          onApplyCalibration={(patch) => updateVoxelSpacing(patch)}
          onNotify={onNotify}
        />

        {/* Anisotropic Voxel Spacing Calibration */}
        <div className="space-y-2 pb-3 border-b border-slate-200 dark:border-slate-800 text-[11.5px]">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            {lang === 'zh' ? '各向异性体素标定 (Voxel Spacing)' : 'Anisotropic Voxel Calibration'}
          </div>
          <div className="grid grid-cols-3 gap-2 font-mono">
            <label className="space-y-1">
              <span className="text-[10px] text-slate-500 block">XY ({volume.spaceUnit})</span>
              <input
                type="number"
                step={0.05}
                min={0.01}
                value={volume.voxelSizeXY}
                onChange={(e) =>
                  updateVoxelSpacing({ voxelSizeXY: Math.max(0.01, Number(e.target.value) || 0.25) })
                }
                className="w-full h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="space-y-1">
              <span className="text-[10px] text-slate-500 block">Z ({volume.spaceUnit})</span>
              <input
                type="number"
                step={0.05}
                min={0.01}
                value={volume.voxelSizeZ}
                onChange={(e) =>
                  updateVoxelSpacing({ voxelSizeZ: Math.max(0.01, Number(e.target.value) || 0.75) })
                }
                className="w-full h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="space-y-1">
              <span className="text-[10px] text-slate-500 block">
                {lang === 'zh' ? '单位' : 'Unit'}
              </span>
              <select
                value={volume.spaceUnit}
                onChange={(e) =>
                  updateVoxelSpacing({
                    spaceUnit: e.target.value as VolumeStackDataset['spaceUnit'],
                  })
                }
                className="w-full h-7 px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
              >
                <option value="µm">µm</option>
                <option value="nm">nm</option>
                <option value="mm">mm</option>
                <option value="px">px</option>
              </select>
            </label>
          </div>
        </div>

        {/* 3D Segmentation & Volumetric Metrology Trigger */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between text-[11px]">
            <span className="font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-500" />
              <span>{lang === 'zh' ? '3D 体积与核质比计量 (Worker)' : '3D Volumetric Metrology'}</span>
            </span>
            <label className="flex items-center gap-1 text-slate-600 dark:text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={autoThreshold}
                onChange={(e) => setAutoThreshold(e.target.checked)}
                className="accent-blue-600 dark:accent-cyan-400"
              />
              <span>{lang === 'zh' ? '自动 Otsu' : 'Auto Otsu'}</span>
            </label>
          </div>

          {!autoThreshold && (
            <label className="block text-[11px] font-mono">
              <div className="flex justify-between text-slate-500">
                <span>{lang === 'zh' ? '细胞体强度阈值' : 'Cell Intensity Threshold'}</span>
                <span>{manualThreshold}</span>
              </div>
              <input
                type="range"
                min={20}
                max={220}
                value={manualThreshold}
                onChange={(e) => setManualThreshold(Number(e.target.value))}
                className="w-full accent-blue-600 dark:accent-cyan-400"
              />
            </label>
          )}

          <label className="block text-[11px] font-mono">
            <div className="flex justify-between text-slate-500">
              <span>{lang === 'zh' ? '暗核相对亮度系数' : 'Nested Nucleus Dark Ratio'}</span>
              <span>{nucleusDarkRatio.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.4}
              max={0.85}
              step={0.02}
              value={nucleusDarkRatio}
              onChange={(e) => setNucleusDarkRatio(Number(e.target.value))}
              className="w-full accent-blue-600 dark:accent-cyan-400"
            />
          </label>

          {errorMsg && (
            <div className="p-2 rounded border border-amber-500/40 bg-amber-500/10 text-[11.5px] text-amber-700 dark:text-amber-300">
              {errorMsg}
            </div>
          )}

          <button
            type="button"
            disabled={analyzing}
            onClick={() => void runAnalysis()}
            className="w-full h-9 rounded-md bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[12.5px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5" />
            <span>
              {analyzing
                ? lang === 'zh'
                  ? 'Web Worker 26-邻域 3D 分析中…'
                  : 'Running 26-Connectivity 3D Analysis…'
                : lang === 'zh'
                ? '运行 3D 体积与核质比分析'
                : 'Run 3D Volumetric & N/C Analysis'}
            </span>
          </button>
        </div>

        {/* 3D Volumetric KPI Strip + D3 3D N/C Histogram + Interactive Object Table */}
        {analysis && (
          <div className="space-y-3 pt-2 border-t border-slate-200 dark:border-slate-800">
            <div className="grid grid-cols-2 gap-2 font-mono tabular-nums">
              <div className="p-2 rounded border border-slate-200 dark:border-slate-800">
                <div className="text-[10px] text-slate-500">
                  {lang === 'zh' ? '3D 细胞 / 细胞核' : '3D Cells / Nuclei'}
                </div>
                <div className="text-[16px] font-semibold text-blue-600 dark:text-cyan-400">
                  {analysis.cellCount} / {analysis.nucleusCount}
                </div>
              </div>
              <div className="p-2 rounded border border-slate-200 dark:border-slate-800">
                <div className="text-[10px] text-slate-500">
                  {lang === 'zh' ? '平均 3D 核质体积比' : 'Mean 3D N/C Vol'}
                </div>
                <div className="text-[16px] font-semibold text-slate-900 dark:text-slate-100">
                  {analysis.meanNcVolumePct.toFixed(1)}%
                </div>
              </div>
              <div className="p-2 rounded border border-slate-200 dark:border-slate-800">
                <div className="text-[10px] text-slate-500">
                  {lang === 'zh' ? `平均细胞体积 (${volume.spaceUnit}³)` : `Mean Vol (${volume.spaceUnit}³)`}
                </div>
                <div className="text-[14px] font-semibold text-slate-900 dark:text-slate-100">
                  {analysis.meanCellVolume.toFixed(1)}
                </div>
              </div>
              <div className="p-2 rounded border border-slate-200 dark:border-slate-800">
                <div className="text-[10px] text-slate-500">
                  {lang === 'zh' ? '平均三维球度 Ψ · 耗时' : 'Sphericity Ψ · Time'}
                </div>
                <div className="text-[14px] font-semibold text-slate-900 dark:text-slate-100">
                  {analysis.meanSphericity.toFixed(2)} · {analysis.durationMs}ms
                </div>
              </div>
            </div>

            {/* D3 3D Nuclear-to-Cytoplasmic Volume Ratio Distribution Histogram */}
            <NcRatioHistogramPanel
              dimension="3d"
              volumeMetrics={analysis.objects}
              selectedObjectId={selectedObjId}
              onSelectObject={setSelectedObjId}
              spaceUnit={volume.spaceUnit}
              lang={lang}
            />

            {/* Export Buttons */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => exportVolumeMetricsCsv(volume, analysis)}
                className="flex-1 h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11px] font-medium text-slate-700 dark:text-slate-200 flex items-center justify-center gap-1.5 cursor-pointer"
              >
                <Table className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
                <span>{lang === 'zh' ? '导出 3D CSV' : 'Export 3D CSV'}</span>
              </button>
              <button
                type="button"
                onClick={() => exportVolumeMetricsExcel(volume, analysis)}
                className="flex-1 h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11px] font-medium text-slate-700 dark:text-slate-200 flex items-center justify-center gap-1.5 cursor-pointer"
              >
                <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                <span>{lang === 'zh' ? '导出 3D Excel' : 'Export 3D .xlsx'}</span>
              </button>
            </div>

            {/* Interactive 3D Cell Table (Bidirectional Highlight with WebGL Viewport) */}
            <div className="rounded border border-slate-200 dark:border-slate-800 overflow-hidden">
              <table className="w-full text-left border-collapse text-[11px] font-mono tabular-nums">
                <thead className="bg-slate-50 dark:bg-[#0B0F1A] text-slate-500 border-b border-slate-200 dark:border-slate-800">
                  <tr>
                    <th className="py-1.5 px-2">ID</th>
                    <th className="py-1.5 px-2">Vol ({volume.spaceUnit}³)</th>
                    <th className="py-1.5 px-2">N/C %</th>
                    <th className="py-1.5 px-2">PCA L1/L2</th>
                    <th className="py-1.5 px-2 text-right">Ψ</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800/70">
                  {parentCells.map((cell) => {
                    const active = selectedObjId === cell.id;
                    return (
                      <tr
                        key={cell.id}
                        onClick={() => setSelectedObjId(active ? null : cell.id)}
                        className={`cursor-pointer transition-colors ${
                          active
                            ? 'bg-amber-500/15 text-amber-800 dark:text-amber-200'
                            : 'hover:bg-slate-100 dark:hover:bg-slate-800/60 text-slate-700 dark:text-slate-300'
                        }`}
                      >
                        <td className="py-1.5 px-2 font-semibold">#{cell.id}</td>
                        <td className="py-1.5 px-2">{cell.volumePhys.toFixed(1)}</td>
                        <td className="py-1.5 px-2 text-blue-600 dark:text-cyan-400 font-semibold">
                          {cell.ncVolumePct !== undefined ? `${cell.ncVolumePct.toFixed(1)}%` : '—'}
                        </td>
                        <td
                          className="py-1.5 px-2 text-slate-500 dark:text-slate-400"
                          title={`PCA [${cell.axesPhys.join(' × ')}] ${volume.spaceUnit} · Flatness ${cell.flatness?.toFixed(2) ?? '—'}`}
                        >
                          {cell.elongation !== undefined ? `${cell.elongation.toFixed(2)}×` : '—'}
                        </td>
                        <td className="py-1.5 px-2 text-right">{cell.sphericity.toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}
