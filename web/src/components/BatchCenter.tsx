import { useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  FileCode2,
  FileSpreadsheet,
  FolderPlus,
  Play,
  Trash2,
  Upload,
} from 'lucide-react';
import type { BatchRow, Lang, Specimen } from '../types/vexel';
import {
  buildDropletEncapsulationSpecimen,
  buildSemLinesSpecimen,
  buildSemParticlesSpecimen,
  decodeUploadedImage,
} from '../engine/samples';
import { computeBatchStatistics } from '../engine/batchStats';
import { exportBatchExcel } from '../engine/exporters';
import { t } from '../i18n/strings';

interface BatchCenterProps {
  lang: Lang;
  rows: BatchRow[];
  onChangeRows: (rows: BatchRow[]) => void;
  onRunBatch: () => void;
  running: boolean;
  onInspectRowOnCanvas: (row: BatchRow) => void;
}

export function BatchCenter({
  lang,
  rows,
  onChangeRows,
  onRunBatch,
  running,
  onInspectRowOnCanvas,
}: BatchCenterProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const { groupStats, comparisons } = useMemo(() => computeBatchStatistics(rows), [rows]);

  const abnormalRows = useMemo(
    () =>
      rows.filter(
        (r) =>
          r.status === 'warning' ||
          r.status === 'failed' ||
          r.warnings.length > 0 ||
          (r.result?.objects.some((o) => o.flagged) ?? false)
      ),
    [rows]
  );

  const loadDropletCohort = () => {
    const specs: Array<{ spec: Specimen; group: string; rep: string }> = [
      { spec: buildDropletEncapsulationSpecimen(101, 'ctrl_r1'), group: 'Control', rep: 'R1' },
      { spec: buildDropletEncapsulationSpecimen(107, 'ctrl_r2'), group: 'Control', rep: 'R2' },
      { spec: buildDropletEncapsulationSpecimen(113, 'ctrl_r3'), group: 'Control', rep: 'R3' },
      { spec: buildDropletEncapsulationSpecimen(211, 'hiFlow_r1'), group: 'HighFlow', rep: 'R1' },
      { spec: buildDropletEncapsulationSpecimen(223, 'hiFlow_r2'), group: 'HighFlow', rep: 'R2' },
      { spec: buildDropletEncapsulationSpecimen(239, 'hiFlow_r3'), group: 'HighFlow', rep: 'R3' },
    ];
    onChangeRows(
      specs.map((item, idx) => ({
        id: `batch_drop_${idx + 1}`,
        specimen: item.spec,
        group: item.group,
        replicate: item.rep,
        status: 'pending',
        warnings: [],
      }))
    );
  };

  const loadSemCohort = () => {
    const specs: Array<{ spec: Specimen; group: string; rep: string }> = [
      { spec: buildSemLinesSpecimen(303, 'lotA_r1'), group: 'Litho_LotA', rep: 'W1' },
      { spec: buildSemLinesSpecimen(311, 'lotA_r2'), group: 'Litho_LotA', rep: 'W2' },
      { spec: buildSemLinesSpecimen(317, 'lotA_r3'), group: 'Litho_LotA', rep: 'W3' },
      { spec: buildSemParticlesSpecimen(505, 'anneal_r1'), group: 'Anneal_450C', rep: 'S1' },
      { spec: buildSemParticlesSpecimen(521, 'anneal_r2'), group: 'Anneal_450C', rep: 'S2' },
      { spec: buildSemParticlesSpecimen(541, 'anneal_r3'), group: 'Anneal_450C', rep: 'S3' },
    ];
    onChangeRows(
      specs.map((item, idx) => ({
        id: `batch_sem_${idx + 1}`,
        specimen: item.spec,
        group: item.group,
        replicate: item.rep,
        status: 'pending',
        warnings: [],
      }))
    );
  };

  const handleUploadFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    setUploadError(null);
    try {
      const decoded = await Promise.all(files.map((f) => decodeUploadedImage(f, { autoCrop: true })));
      const appended: BatchRow[] = decoded.map((bundle, idx) => ({
        id: `batch_up_${Date.now()}_${idx}`,
        specimen: bundle.specimen,
        group: 'Group_A',
        replicate: `R${rows.length + idx + 1}`,
        status: 'pending',
        warnings: [],
      }));
      onChangeRows([...rows, ...appended]);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Failed to decode uploaded file');
    } finally {
      e.target.value = '';
    }
  };

  const updateRow = (id: string, patch: Partial<BatchRow>) => {
    onChangeRows(rows.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  };

  const removeRow = (id: string) => {
    onChangeRows(rows.filter((r) => r.id !== id));
  };

  const exportBatchReproducePy = () => {
    const manifest = rows.map((r) => ({
      file: r.specimen.name,
      modality: r.specimen.modality,
      group: r.group,
      replicate: r.replicate,
      pixel_size: r.pixelSizeOverride ?? r.specimen.calibration.pixelSize,
      space_unit: r.specimen.calibration.spaceUnit,
    }));
    const py = `#!/usr/bin/env python3
"""Vexel 2.0 Unified Batch Reproducibility Script."""
import json
from pathlib import Path
from vexel.batch.stats import run_batch_stats

DESIGN = ${JSON.stringify(manifest, null, 2)}

if __name__ == "__main__":
    print(f"Loaded {len(DESIGN)} batch entries across modalities.")
    print(json.dumps(DESIGN, indent=2, ensure_ascii=False))
`;
    const blob = new Blob([py], { type: 'text/x-python;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'reproduce_batch.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const getPrimaryMetricText = (row: BatchRow): string => {
    const m = row.result?.summary[0];
    if (!m) return '—';
    return `${m.value ?? '—'} ${m.unit}`;
  };

  const completedCount = rows.filter((r) => r.status === 'completed' || r.status === 'warning').length;

  return (
    <div className="flex-1 min-h-0 overflow-y-auto bg-slate-50 dark:bg-[#0B0E17] p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header & Action Bar */}
        <div className="pb-4 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-[19px] font-display font-semibold text-slate-900 dark:text-slate-100">
                {t('batch_title', lang)}
              </h1>
              <span className="px-2 py-0.5 rounded border border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-mono text-[10.5px]">
                {lang === 'zh' ? '全自动批量模式' : 'Full-Auto Batch'}
              </span>
            </div>
            <p className="mt-1 text-[13px] text-slate-500 dark:text-slate-400">
              {t('batch_subtitle', lang)}
            </p>
          </div>

          <div className="flex items-center flex-wrap gap-2">
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleUploadFiles}
              className="hidden"
            />
            <button
              type="button"
              onClick={loadDropletCohort}
              className="h-8 px-3 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[12px] text-slate-700 dark:text-slate-200 flex items-center gap-1.5 cursor-pointer"
            >
              <FolderPlus className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
              <span>{lang === 'zh' ? '微流控演示组 (6)' : 'Microfluidics Cohort (6)'}</span>
            </button>
            <button
              type="button"
              onClick={loadSemCohort}
              className="h-8 px-3 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[12px] text-slate-700 dark:text-slate-200 flex items-center gap-1.5 cursor-pointer"
            >
              <FolderPlus className="w-3.5 h-3.5 text-amber-500" />
              <span>{lang === 'zh' ? 'SEM 电镜演示组 (6)' : 'SEM Cohort (6)'}</span>
            </button>
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="h-8 px-3 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[12px] text-slate-700 dark:text-slate-200 flex items-center gap-1.5 cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>{t('batch_upload_files', lang)}</span>
            </button>
            <button
              type="button"
              disabled={rows.length === 0 || running}
              onClick={onRunBatch}
              className="h-8 px-3.5 rounded-md bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[12px] font-semibold flex items-center gap-1.5 disabled:opacity-45 cursor-pointer"
            >
              <Play className="w-3.5 h-3.5" />
              <span>{running ? '...' : t('batch_run_all', lang)}</span>
            </button>
          </div>
        </div>

        {uploadError && (
          <div className="p-3 rounded border border-rose-500/40 bg-rose-500/10 text-[12px] text-rose-500">
            {uploadError}
          </div>
        )}

        {/* Unified Experiment Design Table */}
        <div className="border border-slate-200 dark:border-slate-800 rounded-lg bg-white dark:bg-[#0E1320] overflow-hidden">
          <div className="px-4 py-2.5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between text-[12px]">
            <span className="font-semibold text-slate-800 dark:text-slate-200">
              {lang === 'zh'
                ? `实验设计表 (${completedCount}/${rows.length} 已完成)`
                : `Experiment Design Table (${completedCount}/${rows.length} completed)`}
            </span>
            {completedCount > 0 && (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => exportBatchExcel(rows, groupStats, comparisons)}
                  className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11.5px] text-slate-700 dark:text-slate-200 flex items-center gap-1.5 cursor-pointer"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-500" />
                  <span>{t('batch_export_xlsx', lang)}</span>
                </button>
                <button
                  type="button"
                  onClick={exportBatchReproducePy}
                  className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11.5px] text-slate-700 dark:text-slate-200 flex items-center gap-1.5 cursor-pointer"
                >
                  <FileCode2 className="w-3.5 h-3.5" />
                  <span>{t('batch_export_reproduce', lang)}</span>
                </button>
              </div>
            )}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-[12.5px] border-collapse">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 text-[11px] font-mono uppercase text-slate-500 bg-slate-50/70 dark:bg-slate-900/50">
                  <th className="py-2 px-3">{lang === 'zh' ? '样本文件' : 'Specimen'}</th>
                  <th className="py-2 px-3">{lang === 'zh' ? '模态' : 'Modality'}</th>
                  <th className="py-2 px-3">{lang === 'zh' ? '分组 (Group)' : 'Group'}</th>
                  <th className="py-2 px-3">{lang === 'zh' ? '重复 (Replicate)' : 'Replicate'}</th>
                  <th className="py-2 px-3">{lang === 'zh' ? '像素标定' : 'Pixel Size'}</th>
                  <th className="py-2 px-3">{lang === 'zh' ? '状态' : 'Status'}</th>
                  <th className="py-2 px-3 text-right">{lang === 'zh' ? '主指标' : 'Primary Metric'}</th>
                  <th className="py-2 px-3 text-right">{lang === 'zh' ? '操作' : 'Action'}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-10 text-center text-slate-500 text-[13px]">
                      {lang === 'zh'
                        ? '点击右上角载入“微流控演示组”、“SEM 电镜演示组”或批量上传本地图像。'
                        : 'Load a Microfluidics or SEM demo cohort, or upload image files to start batch analysis.'}
                    </td>
                  </tr>
                ) : (
                  rows.map((r) => (
                    <tr key={r.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/30">
                      <td className="py-2 px-3 font-mono text-[12px] text-slate-900 dark:text-slate-100">
                        {r.specimen.name}
                      </td>
                      <td className="py-2 px-3 font-mono text-[11.5px] text-slate-500 uppercase">
                        {r.specimen.modality}
                      </td>
                      <td className="py-2 px-3">
                        <input
                          type="text"
                          value={r.group}
                          onChange={(e) => updateRow(r.id, { group: e.target.value })}
                          className="w-28 h-7 px-2 rounded border border-slate-200 dark:border-slate-700 bg-transparent font-mono text-[12px] text-slate-900 dark:text-slate-100"
                        />
                      </td>
                      <td className="py-2 px-3">
                        <input
                          type="text"
                          value={r.replicate}
                          onChange={(e) => updateRow(r.id, { replicate: e.target.value })}
                          className="w-16 h-7 px-2 rounded border border-slate-200 dark:border-slate-700 bg-transparent font-mono text-[12px] text-slate-900 dark:text-slate-100"
                        />
                      </td>
                      <td className="py-2 px-3 font-mono tabular-nums text-[12px] text-slate-600 dark:text-slate-300">
                        <input
                          type="number"
                          step="0.01"
                          value={r.pixelSizeOverride ?? r.specimen.calibration.pixelSize}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            if (Number.isFinite(v) && v > 0) updateRow(r.id, { pixelSizeOverride: v });
                          }}
                          className="w-16 h-7 px-1.5 rounded border border-slate-200 dark:border-slate-700 bg-transparent mr-1"
                        />
                        <span>{r.specimen.calibration.spaceUnit}</span>
                      </td>
                      <td className="py-2 px-3 font-mono text-[11.5px]">
                        {r.status === 'completed' && (
                          <span className="text-emerald-600 dark:text-emerald-400 inline-flex items-center gap-1">
                            <CheckCircle2 className="w-3.5 h-3.5" /> OK
                          </span>
                        )}
                        {r.status === 'warning' && (
                          <span className="text-amber-600 dark:text-amber-400 inline-flex items-center gap-1">
                            <AlertTriangle className="w-3.5 h-3.5" /> QC ({r.warnings.length})
                          </span>
                        )}
                        {r.status === 'pending' && <span className="text-slate-400">pending</span>}
                        {r.status === 'running' && <span className="text-cyan-400">running…</span>}
                        {r.status === 'failed' && <span className="text-rose-500">failed</span>}
                      </td>
                      <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-900 dark:text-slate-100">
                        {getPrimaryMetricText(r)}
                      </td>
                      <td className="py-2 px-3 text-right">
                        <div className="inline-flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => onInspectRowOnCanvas(r)}
                            className="h-7 px-2 rounded border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11px] text-slate-700 dark:text-slate-200 inline-flex items-center gap-1 cursor-pointer"
                          >
                            <Eye className="w-3 h-3" />
                            <span>{lang === 'zh' ? '查看' : 'Inspect'}</span>
                          </button>
                          <button
                            type="button"
                            onClick={() => removeRow(r.id)}
                            className="w-7 h-7 rounded border border-slate-200 dark:border-slate-700 hover:text-rose-500 text-slate-400 inline-flex items-center justify-center cursor-pointer"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Abnormal / QC Review Queue */}
        {abnormalRows.length > 0 && (
          <div className="border border-amber-500/35 rounded-lg bg-white dark:bg-[#0E1320] p-4 space-y-2.5">
            <div className="flex items-center gap-2 text-[13px] font-semibold text-amber-600 dark:text-amber-400">
              <AlertTriangle className="w-4 h-4" />
              <span>
                {t('batch_abnormal_queue', lang)} ({abnormalRows.length})
              </span>
            </div>
            <div className="divide-y divide-slate-200 dark:divide-slate-800">
              {abnormalRows.map((r) => {
                const nFlagged = r.result?.objects.filter((o) => o.flagged).length ?? 0;
                return (
                  <div key={r.id} className="py-2 flex items-center justify-between gap-3 text-[12px]">
                    <div className="min-w-0">
                      <span className="font-mono font-medium text-slate-900 dark:text-slate-100">
                        {r.specimen.name}
                      </span>
                      <span className="mx-2 text-slate-400">·</span>
                      <span className="text-slate-500 dark:text-slate-400">
                        {r.warnings[0] ?? `${nFlagged} flagged object(s)`}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => onInspectRowOnCanvas(r)}
                      className="h-7 px-2.5 rounded bg-amber-500 hover:bg-amber-400 text-slate-950 font-medium text-[11.5px] shrink-0 cursor-pointer"
                    >
                      {t('batch_jump_review', lang)}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Replicate-Unit Group Statistics & Pairwise Welch's t-test Tables */}
        {groupStats.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="border border-slate-200 dark:border-slate-800 rounded-lg bg-white dark:bg-[#0E1320] overflow-hidden">
              <div className="px-4 py-2.5 border-b border-slate-200 dark:border-slate-800 text-[12.5px] font-semibold text-slate-800 dark:text-slate-100">
                {lang === 'zh'
                  ? '分组描述统计（以 Replicate 为统计单位）'
                  : 'Group Statistics (Replicate-Unit)'}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[12px] font-mono tabular-nums">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-800 text-[10.5px] uppercase text-slate-500 bg-slate-50/70 dark:bg-slate-900/50">
                      <th className="py-2 px-3">{lang === 'zh' ? '指标' : 'Metric'}</th>
                      <th className="py-2 px-3">{lang === 'zh' ? '组别' : 'Group'}</th>
                      <th className="py-2 px-3 text-right">n</th>
                      <th className="py-2 px-3 text-right">Mean ± SD</th>
                      <th className="py-2 px-3 text-right">95% CI</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                    {groupStats.map((g, i) => (
                      <tr key={i}>
                        <td className="py-2 px-3 font-sans text-slate-800 dark:text-slate-200">
                          {lang === 'zh' ? g.metricLabelZh : g.metricLabelEn}
                        </td>
                        <td className="py-2 px-3 text-slate-600 dark:text-slate-300">{g.group}</td>
                        <td className="py-2 px-3 text-right text-slate-500">{g.nReplicates}</td>
                        <td className="py-2 px-3 text-right text-slate-900 dark:text-slate-100">
                          {g.mean} ± {g.sd} {g.unit}
                        </td>
                        <td className="py-2 px-3 text-right text-slate-500">
                          [{g.ci95Low}, {g.ci95High}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="border border-slate-200 dark:border-slate-800 rounded-lg bg-white dark:bg-[#0E1320] overflow-hidden">
              <div className="px-4 py-2.5 border-b border-slate-200 dark:border-slate-800 text-[12.5px] font-semibold text-slate-800 dark:text-slate-100">
                {lang === 'zh'
                  ? "组间假设检验（Welch's t-test · Holm 校正 · Hedges' g）"
                  : "Pairwise Hypothesis Tests (Welch's t · Holm · Hedges' g)"}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[12px] font-mono tabular-nums">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-800 text-[10.5px] uppercase text-slate-500 bg-slate-50/70 dark:bg-slate-900/50">
                      <th className="py-2 px-3">{lang === 'zh' ? '指标' : 'Metric'}</th>
                      <th className="py-2 px-3">{lang === 'zh' ? '对比' : 'Pair'}</th>
                      <th className="py-2 px-3 text-right">Δ</th>
                      <th className="py-2 px-3 text-right">p (Holm)</th>
                      <th className="py-2 px-3 text-right">Hedges&apos; g</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                    {comparisons.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="py-6 text-center font-sans text-slate-500">
                          {lang === 'zh'
                            ? '需至少 2 个包含 ≥2 个重复的同指标实验组以计算检验'
                            : 'Need at least 2 groups with ≥2 replicates on a shared metric'}
                        </td>
                      </tr>
                    ) : (
                      comparisons.map((c, i) => (
                        <tr key={i}>
                          <td className="py-2 px-3 font-sans text-slate-800 dark:text-slate-200">
                            {lang === 'zh' ? c.metricLabelZh : c.metricLabelEn}
                          </td>
                          <td className="py-2 px-3 text-slate-600 dark:text-slate-300">
                            {c.groupA} vs {c.groupB}
                          </td>
                          <td className="py-2 px-3 text-right text-slate-800 dark:text-slate-200">
                            {c.diff > 0 ? `+${c.diff}` : c.diff}
                          </td>
                          <td
                            className={`py-2 px-3 text-right font-semibold ${
                              c.significant
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-slate-500'
                            }`}
                          >
                            {c.pValue.toFixed(4)}
                            {c.significant ? ' *' : ''}
                          </td>
                          <td className="py-2 px-3 text-right text-slate-700 dark:text-slate-300">
                            {c.hedgesG}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
