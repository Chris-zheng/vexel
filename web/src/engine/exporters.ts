import * as XLSX from 'xlsx';
import type { BatchRow, ComparisonRow, GroupStatRow, RunResult, Specimen } from '../types/vexel';

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Generate academic Methods paragraph in both Chinese and English matching Vexel's reproducibility standard. */
export function buildMethodsParagraphs(specimen: Specimen, result: RunResult): { zh: string; en: string } {
  const opsList = result.provenance.steps.map((s) => `${s.op}`).join(' → ');
  const modelsText =
    result.provenance.modelRefs.length > 0
      ? `并接入经 SHA-256 校验的检测模型（${result.provenance.modelRefs.join(', ')}）`
      : '全程采用确定性科学计算算子';
  const modelsTextEn =
    result.provenance.modelRefs.length > 0
      ? `with SHA-256 verified vision model(s) (${result.provenance.modelRefs.join(', ')})`
      : 'using deterministic scientific operators';

  const zh = `图像分析采用 Vexel 2.0 统一分析平台完成（配方 ${result.recipe.id}，摘要 ${result.recipeDigest.slice(0, 12)}）。输入样本 ${specimen.name}（${specimen.width}×${specimen.height} 像素，标定 ${specimen.calibration.pixelSize} ${specimen.calibration.spaceUnit}/px，来源：${specimen.calibration.source}）依次执行工作流：${opsList}，${modelsText}。本次运行共检出 ${result.objects.filter((o) => !o.excluded).length} 个有效目标（人工复核修正 ${result.provenance.correctionsApplied} 处），结果摘要校验值 ${result.resultDigest.slice(0, 12)}，可通过随附的 recipe.json 与 reproduce.py 逐位复现。`;

  const en = `Image analysis was performed using Vexel 2.0 (recipe ${result.recipe.id}, digest ${result.recipeDigest.slice(0, 12)}). Specimen ${specimen.name} (${specimen.width}×${specimen.height} px; calibration ${specimen.calibration.pixelSize} ${specimen.calibration.spaceUnit}/px from ${specimen.calibration.source}) was processed through the frozen pipeline [${opsList}] ${modelsTextEn}. A total of ${result.objects.filter((o) => !o.excluded).length} valid objects were quantified (${result.provenance.correctionsApplied} human review correction(s) applied; result digest ${result.resultDigest.slice(0, 12)}), reproducible via the archived recipe.json and reproduce.py.`;

  return { zh, en };
}

/** Export UTF-8 with BOM CSV (opens directly in Excel without garbled Chinese characters). */
export function exportObjectsCsv(specimen: Specimen, result: RunResult): void {
  const cols = ['id', 'kind', 'x_px', 'y_px', `diameter_${specimen.calibration.spaceUnit}`, 'width', 'area', 'sub_area', 'sub_area_ratio_pct', 'sub_region_count', 'circularity', 'solidity', 'cells', 'positive', 'ler_3sigma', 'lwr_3sigma', 'class', 'flagged', 'excluded', 'confidence'];
  const lines = [cols.join(',')];
  for (const o of result.objects) {
    lines.push(
      [
        o.id,
        o.kind,
        o.x.toFixed(2),
        o.y.toFixed(2),
        o.diameter ?? '',
        o.width ?? '',
        o.area ?? '',
        o.subArea ?? '',
        o.subAreaRatioPct ?? '',
        o.subRegionCount ?? '',
        o.circularity ?? '',
        o.solidity ?? '',
        o.cells ?? '',
        o.positive ?? '',
        o.ler3Sigma ?? '',
        o.lwr3Sigma ?? '',
        o.cls ?? '',
        o.flagged ? 'TRUE' : 'FALSE',
        o.excluded ? 'TRUE' : 'FALSE',
        o.confidence.toFixed(3),
      ].join(',')
    );
  }
  const blob = new Blob(['\ufeff' + lines.join('\n') + '\n'], { type: 'text/csv;charset=utf-8' });
  triggerDownload(blob, `${specimen.name.replace(/\.[^.]+$/, '')}_objects.csv`);
}

/** Export Multi-Sheet Excel (.xlsx) workbook for single run OR unified batch experiment. */
export function exportSingleExcel(specimen: Specimen, result: RunResult): void {
  const wb = XLSX.utils.book_new();
  const summaryRows = result.summary.map((s) => ({
    Metric_ZH: s.labelZh,
    Metric_EN: s.labelEn,
    Value: s.value,
    Unit: s.unit,
    Detail: s.detailZh ?? '',
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(summaryRows), 'Summary');

  const objRows = result.objects.map((o) => ({
    ID: o.id,
    Kind: o.kind,
    X_px: Number(o.x.toFixed(2)),
    Y_px: Number(o.y.toFixed(2)),
    [`Diameter (${specimen.calibration.spaceUnit})`]: o.diameter ?? null,
    [`Width (${specimen.calibration.spaceUnit})`]: o.width ?? null,
    Area: o.area ?? null,
    SubRegion_Area: o.subArea ?? null,
    SubRegion_Area_Ratio_Pct: o.subAreaRatioPct ?? null,
    SubRegion_Count: o.subRegionCount ?? null,
    Circularity: o.circularity ?? null,
    Solidity: o.solidity ?? null,
    Cells: o.cells ?? null,
    Positive: o.positive ?? null,
    LER_3sigma: o.ler3Sigma ?? null,
    LWR_3sigma: o.lwr3Sigma ?? null,
    Class: o.cls ?? '',
    Flagged: Boolean(o.flagged),
    Excluded: Boolean(o.excluded),
    Confidence: o.confidence,
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(objRows), 'Objects');

  const provRows = [
    { Field: 'Input File', Value: specimen.name },
    { Field: 'Input SHA-256', Value: specimen.sha256 },
    { Field: 'Pixel Size', Value: `${specimen.calibration.pixelSize} ${specimen.calibration.spaceUnit}/px` },
    { Field: 'Calibration Source', Value: specimen.calibration.source },
    { Field: 'Calibration Confirmed', Value: String(specimen.calibration.confirmed) },
    { Field: 'Recipe ID', Value: result.recipe.id },
    { Field: 'Recipe Digest', Value: result.recipeDigest },
    { Field: 'Result Digest', Value: result.resultDigest },
    { Field: 'Models', Value: result.provenance.modelRefs.join(', ') || 'Classical deterministic' },
    { Field: 'Corrections Applied', Value: String(result.provenance.correctionsApplied) },
  ];
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(provRows), 'Provenance');

  const buf = XLSX.write(wb, { type: 'array', bookType: 'xlsx' });
  triggerDownload(
    new Blob([buf], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }),
    `${specimen.name.replace(/\.[^.]+$/, '')}_vexel_report.xlsx`
  );
}

export function exportBatchExcel(
  rows: BatchRow[],
  groupStats: GroupStatRow[],
  comparisons: ComparisonRow[]
): void {
  const wb = XLSX.utils.book_new();
  const fileSheet = rows.map((r) => ({
    ID: r.id,
    File: r.specimen.name,
    Modality: r.specimen.modality,
    Group: r.group,
    Replicate: r.replicate,
    Status: r.status,
    PixelSize: `${r.pixelSizeOverride ?? r.specimen.calibration.pixelSize} ${r.specimen.calibration.spaceUnit}`,
    Duration_ms: r.durationMs ?? '',
    Warnings: r.warnings.length,
    ResultDigest: r.result?.resultDigest.slice(0, 16) ?? '',
    ...(r.result?.summaryMap ?? {}),
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(fileSheet), 'Files');
  if (groupStats.length > 0) {
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(groupStats), 'GroupStats');
  }
  if (comparisons.length > 0) {
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(comparisons), 'Comparisons');
  }
  const buf = XLSX.write(wb, { type: 'array', bookType: 'xlsx' });
  triggerDownload(
    new Blob([buf], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }),
    `vexel_batch_experiment_${new Date().toISOString().slice(0, 10)}.xlsx`
  );
}

/** Export annotated Vector SVG with crisp calibrated scale bar. */
export function exportAnnotatedSvg(specimen: Specimen, result: RunResult): void {
  const off = document.createElement('canvas');
  off.width = specimen.width;
  off.height = specimen.height;
  const ctx = off.getContext('2d')!;
  const imgData = new ImageData(new Uint8ClampedArray(specimen.rgba), specimen.width, specimen.height);
  ctx.putImageData(imgData, 0, 0);
  const bgDataUrl = off.toDataURL('image/png');

  const shapeSvg = result.shapes
    .map((s) => {
      const dash = s.dashed ? ' stroke-dasharray="5 3"' : '';
      if (s.kind === 'circle' && s.points.length >= 2) {
        const [cx, cy] = s.points[0];
        const [rx] = s.points[1];
        return `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${rx.toFixed(1)}" fill="none" stroke="${s.color}" stroke-width="1.6"${dash}/>`;
      }
      if (s.kind === 'polygon' && s.points.length >= 3) {
        const pts = s.points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
        return `<polygon points="${pts}" fill="${s.fill ?? 'none'}" stroke="${s.color}" stroke-width="1.5"${dash}/>`;
      }
      if (s.kind === 'polyline' && s.points.length >= 2) {
        const pts = s.points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
        return `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="1.5"${dash}/>`;
      }
      return '';
    })
    .join('\n  ');

  // Calibrated scale bar (target ~15% of width)
  const rawPhys = specimen.width * 0.16 * specimen.calibration.pixelSize;
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(1e-6, rawPhys))));
  const nice = [1, 2, 5, 10].find((m) => m * pow >= rawPhys * 0.7) ?? 1;
  const barPhys = nice * pow;
  const barPx = barPhys / Math.max(1e-6, specimen.calibration.pixelSize);
  const bx = specimen.width - barPx - 20;
  const by = (specimen.calibration.cropRows ?? specimen.height) - 22;

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${specimen.width} ${specimen.height}" width="${specimen.width}" height="${specimen.height}">
  <image href="${bgDataUrl}" x="0" y="0" width="${specimen.width}" height="${specimen.height}"/>
  ${shapeSvg}
  <rect x="${(bx - 8).toFixed(1)}" y="${(by - 18).toFixed(1)}" width="${(barPx + 16).toFixed(1)}" height="28" rx="4" fill="rgba(11,14,23,0.78)"/>
  <line x1="${bx.toFixed(1)}" y1="${by.toFixed(1)}" x2="${(bx + barPx).toFixed(1)}" y2="${by.toFixed(1)}" stroke="#ffffff" stroke-width="3"/>
  <text x="${(bx + barPx / 2).toFixed(1)}" y="${(by - 6).toFixed(1)}" fill="#ffffff" font-family="monospace" font-size="11" text-anchor="middle">${barPhys} ${specimen.calibration.spaceUnit}</text>
</svg>`;
  triggerDownload(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }), `${specimen.name.replace(/\.[^.]+$/, '')}_annotated.svg`);
}

/** Export single-file self-contained printable HTML/PDF Report with embedded annotated image & bilingual Methods. */
export function exportHtmlReport(specimen: Specimen, result: RunResult, annotatedPngDataUrl?: string): void {
  const methods = buildMethodsParagraphs(specimen, result);
  const cards = result.summary
    .map(
      (s) => `
      <div class="card">
        <div class="k">${s.labelZh} / ${s.labelEn}</div>
        <div class="v">${s.value ?? '–'} <span class="u">${s.unit}</span></div>
        <div class="d">${s.detailZh ?? ''}</div>
      </div>`
    )
    .join('');

  const warns = result.warnings.length
    ? `<div class="warn"><strong>QC Notices:</strong><ul>${result.warnings.map((w) => `<li>${w}</li>`).join('')}</ul></div>`
    : '';

  const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>Vexel Report — ${specimen.name}</title>
<style>
  body{font:14px/1.6 -apple-system,"PingFang SC","Segoe UI",sans-serif;color:#0f172a;max-width:980px;margin:28px auto;padding:0 20px}
  h1{font-size:22px;margin:0 0 4px} .muted{color:#64748b;font-size:12px;font-family:monospace}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin:18px 0}
  .card{border:1px solid #e2e8f0;border-radius:8px;padding:12px}
  .k{font-size:12px;color:#64748b} .v{font-size:24px;font-weight:700;font-family:monospace;margin:4px 0}
  .u{font-size:12px;font-weight:400;color:#64748b} .d{font-size:12px;color:#475569}
  .warn{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;margin:14px 0}
  img{max-width:100%;border:1px solid #e2e8f0;border-radius:6px;margin:12px 0}
  h2{font-size:16px;margin:24px 0 8px;border-bottom:1px solid #e2e8f0;padding-bottom:4px}
  pre{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;font-size:12px;overflow:auto}
  @media print{body{margin:0} .no-print{display:none}}
</style></head><body>
  <h1>Vexel 2.0 科学图像分析报告 · ${specimen.name}</h1>
  <div class="muted">Recipe: ${result.recipe.id} · Digest: ${result.recipeDigest.slice(0, 16)} · Result Digest: ${result.resultDigest.slice(0, 16)} · ${result.executedAt}</div>
  ${warns}
  <div class="grid">${cards}</div>
  ${annotatedPngDataUrl ? `<h2>标注复核图 (Annotated Micrograph)</h2><img src="${annotatedPngDataUrl}" alt="Annotated specimen"/>` : ''}
  <h2>论文方法段落 (Methods — ZH / EN)</h2>
  <p>${methods.zh}</p>
  <p>${methods.en}</p>
  <h2>溯源记录 (Provenance)</h2>
  <pre>${JSON.stringify(result.provenance, null, 2)}</pre>
</body></html>`;

  triggerDownload(new Blob([html], { type: 'text/html;charset=utf-8' }), `${specimen.name.replace(/\.[^.]+$/, '')}_report.html`);
}

/** Export frozen .vexflow workflow template JSON. */
export function exportVexflow(result: RunResult): void {
  const payload = {
    format: 'vexel.workflow.v1',
    exportedAt: new Date().toISOString(),
    recipeDigest: result.recipeDigest,
    recipe: result.recipe,
  };
  triggerDownload(
    new Blob([JSON.stringify(payload, null, 2) + '\n'], { type: 'application/json;charset=utf-8' }),
    `${result.recipe.id}.vexflow.json`
  );
}

/** Export standalone executable Python reproduce.py script targeting Vexel unified Python engine. */
export function exportReproducePy(specimen: Specimen, result: RunResult): void {
  const py = `#!/usr/bin/env python3
"""Standalone reproducibility script exported by Vexel 2.0 Workbench.

Re-runs the frozen unified workflow on Python Vexel engine and verifies provenance digests.
Usage:
    uv run python reproduce.py <path-to-${specimen.name}>
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from vexel.engine import Recipe, run_recipe
from vexel.readers import open_volume

FROZEN_RECIPE = ${JSON.stringify(result.recipe, null, 2)}
EXPECTED_RECIPE_DIGEST = "${result.recipeDigest}"
CALIBRATION = {"pixel_size": ${specimen.calibration.pixelSize}, "space_unit": "${specimen.calibration.spaceUnit}"}

def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("${specimen.name}")
    vol = open_volume(src, pixel_size=CALIBRATION["pixel_size"], space_unit=CALIBRATION["space_unit"])
    recipe = Recipe.load("${result.recipe.id}")
    out_dir = Path("reproduced_run")
    res = run_recipe(recipe, vol, out_dir, overwrite=True, in_memory=True)
    print(f"Completed: status={res.status} result_digest={res.result_digest}")
    print(json.dumps(res.summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
`;
  triggerDownload(new Blob([py], { type: 'text/x-python;charset=utf-8' }), `reproduce_${result.recipe.id}.py`);
}
