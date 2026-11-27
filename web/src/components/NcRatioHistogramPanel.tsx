import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import type { DetectionObject, Lang, VolumetricObjectMetric } from '../types/vexel';

type RatioDisplayMode = 'nc_pct' | 'nc_ratio' | 'ncell_pct';
type BinPreset = 'auto' | 5 | 8 | 12;

interface NcCellDatum {
  id: number;
  cellArea: number;
  nuclearArea: number;
  cytoplasmArea: number;
  /** Nuclear-to-cytoplasmic area ratio: A_nucleus / A_cytoplasm */
  ncRatio: number;
  /** Nuclear-to-cytoplasmic percentage: (A_nucleus / A_cytoplasm) * 100 */
  ncPct: number;
  /** Nuclear-to-total-cell percentage: (A_nucleus / A_cell) * 100 */
  ncellPct: number;
  excluded: boolean;
}

interface HoveredBinInfo {
  x0: number;
  x1: number;
  cells: NcCellDatum[];
  sharePct: number;
}

interface NcRatioHistogramPanelProps {
  objects?: DetectionObject[];
  volumeMetrics?: VolumetricObjectMetric[];
  dimension?: '2d' | '3d';
  selectedObjectId: number | null;
  onSelectObject: (id: number | null) => void;
  spaceUnit: string;
  lang: Lang;
}

export function NcRatioHistogramPanel({
  objects = [],
  volumeMetrics,
  dimension = '2d',
  selectedObjectId,
  onSelectObject,
  spaceUnit,
  lang,
}: NcRatioHistogramPanelProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const [mode, setMode] = useState<RatioDisplayMode>('nc_pct');
  const [binPreset, setBinPreset] = useState<BinPreset>('auto');
  const [chartWidth, setChartWidth] = useState<number>(320);
  const [hoveredBin, setHoveredBin] = useState<HoveredBinInfo | null>(null);
  const [hoveredCell, setHoveredCell] = useState<NcCellDatum | null>(null);

  const is3D = dimension === '3d' || (volumeMetrics !== undefined && objects.length === 0);

  // Compute exact nuclear-to-cytoplasmic ratios (2D Area or 3D Volume)
  const cellData = useMemo<NcCellDatum[]>(() => {
    const out: NcCellDatum[] = [];
    if (is3D && volumeMetrics) {
      for (const v of volumeMetrics) {
        if (v.kind !== 'cell_3d') continue;
        if (v.subVolumePhys === undefined || v.volumePhys <= 0) continue;
        const cellVol = Math.max(0, v.volumePhys);
        const nucVol = Math.max(0, v.subVolumePhys);
        const cytoVol = Math.max(1e-6, v.cytoplasmVolumePhys ?? cellVol - nucVol);
        const ncRatio = v.ncVolumeRatio ?? nucVol / cytoVol;
        const ncPct = v.ncVolumePct ?? ncRatio * 100;
        const ncellPct = v.ncellVolumePct ?? (nucVol / cellVol) * 100;
        out.push({
          id: v.id,
          cellArea: cellVol,
          nuclearArea: nucVol,
          cytoplasmArea: Math.max(0, cellVol - nucVol),
          ncRatio,
          ncPct,
          ncellPct,
          excluded: false,
        });
      }
      return out;
    }

    for (const o of objects) {
      if (o.excluded) continue;
      if (o.subArea === undefined && o.subAreaRatioPct === undefined) continue;
      const cellArea = Math.max(0, o.area ?? 0);
      const nuclearArea =
        o.subArea !== undefined
          ? Math.max(0, o.subArea)
          : cellArea * ((o.subAreaRatioPct ?? 0) / 100);
      if (cellArea <= 0) continue;
      const cytoplasmArea = Math.max(1e-6, cellArea - nuclearArea);
      const ncRatio = nuclearArea / cytoplasmArea;
      const ncPct = ncRatio * 100;
      const ncellPct = o.subAreaRatioPct ?? (nuclearArea / cellArea) * 100;
      out.push({
        id: o.id,
        cellArea,
        nuclearArea,
        cytoplasmArea: Math.max(0, cellArea - nuclearArea),
        ncRatio,
        ncPct,
        ncellPct,
        excluded: Boolean(o.excluded),
      });
    }
    return out;
  }, [is3D, volumeMetrics, objects]);

  const valueOf = useMemo(() => {
    return (d: NcCellDatum) => {
      if (mode === 'nc_ratio') return d.ncRatio;
      if (mode === 'ncell_pct') return d.ncellPct;
      return d.ncPct;
    };
  }, [mode]);

  const formatVal = useMemo(() => {
    return (v: number, compact = false) => {
      if (mode === 'nc_ratio') return compact ? v.toFixed(2) : v.toFixed(3);
      return `${v.toFixed(compact ? 1 : 2)}%`;
    };
  }, [mode]);

  // Summary statistics computed via D3
  const stats = useMemo(() => {
    const vals = cellData.map(valueOf);
    if (vals.length === 0) {
      return { mean: 0, median: 0, sd: 0, min: 0, max: 0, n: 0 };
    }
    const mean = d3.mean(vals) ?? 0;
    const median = d3.median(vals) ?? 0;
    const sd = vals.length > 1 ? d3.deviation(vals) ?? 0 : 0;
    const [min, max] = d3.extent(vals) as [number, number];
    return { mean, median, sd, min, max, n: vals.length };
  }, [cellData, valueOf]);

  // Responsive container measurement via ResizeObserver
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      const w = Math.floor(el.getBoundingClientRect().width);
      if (w >= 180) setChartWidth(w);
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Render D3 Histogram + Individual Cell Rug + Mean/Median Guides
  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    if (cellData.length === 0) return;

    const width = Math.max(240, chartWidth);
    const height = 164;
    const margin = { top: 20, right: 14, bottom: 30, left: 32 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    svg.attr('viewBox', `0 0 ${width} ${height}`);

    const vals = cellData.map(valueOf);
    const rawMin = d3.min(vals) ?? 0;
    const rawMax = d3.max(vals) ?? 1;
    const rawSpan = rawMax - rawMin;

    // Smart padded domain so both small-n (e.g., 2 kidney cells) and dense-n distributions frame cleanly
    const pad =
      rawSpan > 1e-6
        ? Math.max(rawSpan * 0.25, mode === 'nc_ratio' ? 0.012 : 1.2)
        : Math.max(rawMax * 0.3, mode === 'nc_ratio' ? 0.025 : 2.5);
    const domMin = Math.max(0, rawMin - pad);
    const domMax = rawMax + pad;

    const xScale = d3.scaleLinear().domain([domMin, domMax]).nice(5).range([0, innerW]);

    const autoBins = Math.min(12, Math.max(5, Math.round(Math.sqrt(cellData.length) * 2.2) + 2));
    const targetBinCount = binPreset === 'auto' ? autoBins : binPreset;
    const [d0, d1] = xScale.domain();
    const step = (d1 - d0) / targetBinCount;
    const thresholds = d3.range(1, targetBinCount).map((i) => d0 + i * step);

    const binner = d3
      .bin<NcCellDatum, number>()
      .value(valueOf)
      .domain([d0, d1])
      .thresholds(thresholds);

    const bins = binner(cellData);
    const maxCount = Math.max(1, d3.max(bins, (b) => b.length) ?? 1);
    const yScale = d3.scaleLinear().domain([0, maxCount]).nice(Math.min(4, maxCount)).range([innerH, 0]);

    const root = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Subtle horizontal gridlines (scientific hairline style)
    const yTicks = yScale.ticks(Math.min(4, maxCount)).filter((t) => Number.isInteger(t));
    root
      .append('g')
      .attr('class', 'nc-grid')
      .selectAll('line')
      .data(yTicks)
      .join('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', (d) => yScale(d))
      .attr('y2', (d) => yScale(d))
      .attr('stroke', 'currentColor')
      .attr('class', (d) =>
        d === 0
          ? 'text-slate-300 dark:text-slate-700'
          : 'text-slate-200/90 dark:text-slate-800/90'
      )
      .attr('stroke-width', (d) => (d === 0 ? 1 : 0.75))
      .attr('stroke-dasharray', (d) => (d === 0 ? null : '2,3'));

    // X Axis (D3)
    const xAxis = d3
      .axisBottom(xScale)
      .ticks(5)
      .tickSize(4)
      .tickPadding(5)
      .tickFormat((d) => formatVal(Number(d), true));

    const xAxisG = root
      .append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(xAxis);

    xAxisG.select('.domain').attr('stroke', 'currentColor').attr('class', 'text-slate-300 dark:text-slate-700');
    xAxisG.selectAll('line').attr('stroke', 'currentColor').attr('class', 'text-slate-300 dark:text-slate-700');
    xAxisG
      .selectAll('text')
      .attr('fill', 'currentColor')
      .attr('class', 'text-slate-500 dark:text-slate-400 font-mono text-[9.5px]');

    // Y Axis (D3, integer counts)
    const yAxis = d3
      .axisLeft(yScale)
      .tickValues(yTicks)
      .tickSize(0)
      .tickPadding(6)
      .tickFormat((d) => `${Number(d)}`);

    const yAxisG = root.append('g').call(yAxis);
    yAxisG.select('.domain').remove();
    yAxisG
      .selectAll('text')
      .attr('fill', 'currentColor')
      .attr('class', 'text-slate-500 dark:text-slate-400 font-mono text-[9.5px]');

    // Y-axis unit caption at top-left
    root
      .append('text')
      .attr('x', -6)
      .attr('y', -8)
      .attr('text-anchor', 'end')
      .attr('fill', 'currentColor')
      .attr('class', 'text-slate-400 dark:text-slate-500 font-mono text-[8.5px]')
      .text(lang === 'zh' ? '细胞' : 'n');

    // Histogram Bars
    const barGroups = root
      .selectAll<SVGGElement, d3.Bin<NcCellDatum, number>>('g.nc-bin')
      .data(bins)
      .join('g')
      .attr('class', 'nc-bin');

    barGroups
      .append('rect')
      .attr('x', (b) => xScale(b.x0 ?? 0) + 1)
      .attr('width', (b) => Math.max(2, xScale(b.x1 ?? 0) - xScale(b.x0 ?? 0) - 2))
      .attr('y', innerH)
      .attr('height', 0)
      .attr('rx', 2)
      .attr('class', (b) => {
        const hasSelected =
          selectedObjectId !== null && b.some((c) => c.id === selectedObjectId);
        if (b.length === 0) return 'fill-transparent';
        if (hasSelected) {
          return 'fill-amber-500 dark:fill-amber-400 stroke-amber-600 dark:stroke-amber-300 stroke-[1] cursor-pointer';
        }
        return 'fill-blue-600/80 dark:fill-cyan-500/75 hover:fill-blue-600 dark:hover:fill-cyan-400 cursor-pointer transition-colors';
      })
      .on('mouseenter', (_event, b) => {
        if (b.length === 0) return;
        setHoveredCell(null);
        setHoveredBin({
          x0: b.x0 ?? 0,
          x1: b.x1 ?? 0,
          cells: [...b],
          sharePct: (b.length / cellData.length) * 100,
        });
      })
      .on('mouseleave', () => {
        setHoveredBin(null);
      })
      .on('click', (_event, b) => {
        if (b.length === 0) return;
        const curIdx = b.findIndex((c) => c.id === selectedObjectId);
        const nextCell = curIdx >= 0 ? b[(curIdx + 1) % b.length] : b[0];
        onSelectObject(nextCell.id === selectedObjectId ? null : nextCell.id);
      })
      .transition()
      .duration(180)
      .ease(d3.easeCubicOut)
      .attr('y', (b) => (b.length > 0 ? yScale(b.length) : innerH))
      .attr('height', (b) => (b.length > 0 ? Math.max(2, innerH - yScale(b.length)) : 0));

    // Count labels directly above non-empty bars when bar width allows
    barGroups
      .filter((b) => b.length > 0)
      .append('text')
      .attr('x', (b) => (xScale(b.x0 ?? 0) + xScale(b.x1 ?? 0)) * 0.5)
      .attr('y', (b) => Math.max(8, yScale(b.length) - 4))
      .attr('text-anchor', 'middle')
      .attr('pointer-events', 'none')
      .attr('fill', 'currentColor')
      .attr('class', 'text-slate-600 dark:text-slate-300 font-mono text-[9px] font-semibold')
      .text((b) => `${b.length}`);

    // Mean vertical reference hairline (amber dashed)
    const meanX = xScale(stats.mean);
    root
      .append('line')
      .attr('x1', meanX)
      .attr('x2', meanX)
      .attr('y1', -3)
      .attr('y2', innerH)
      .attr('stroke', '#F59E0B')
      .attr('stroke-width', 1.25)
      .attr('stroke-dasharray', '3,2')
      .attr('pointer-events', 'none');

    root
      .append('text')
      .attr('x', Math.min(innerW - 2, Math.max(2, meanX + 3)))
      .attr('y', -7)
      .attr('text-anchor', meanX > innerW - 44 ? 'end' : 'start')
      .attr('fill', '#F59E0B')
      .attr('class', 'font-mono text-[9px] font-semibold')
      .text(`μ ${formatVal(stats.mean, true)}`);

    // Median reference hairline (emerald dotted) when meaningfully distinct from mean
    const medianX = xScale(stats.median);
    if (Math.abs(medianX - meanX) > 10 && cellData.length >= 3) {
      root
        .append('line')
        .attr('x1', medianX)
        .attr('x2', medianX)
        .attr('y1', 2)
        .attr('y2', innerH)
        .attr('stroke', '#10B981')
        .attr('stroke-width', 1.1)
        .attr('stroke-dasharray', '1.5,2.5')
        .attr('pointer-events', 'none');
    }

    // Individual cell position rug dots along the X-axis baseline (click to locate cell on canvas)
    root
      .selectAll<SVGCircleElement, NcCellDatum>('circle.nc-cell-dot')
      .data(cellData)
      .join('circle')
      .attr('class', (d) =>
        d.id === selectedObjectId
          ? 'nc-cell-dot cursor-pointer fill-amber-500 dark:fill-amber-400 stroke-white dark:stroke-slate-950'
          : 'nc-cell-dot cursor-pointer fill-blue-600 dark:fill-cyan-400 stroke-white dark:stroke-slate-950'
      )
      .attr('cx', (d) => xScale(valueOf(d)))
      .attr('cy', innerH)
      .attr('r', (d) => (d.id === selectedObjectId ? 4.4 : 3.2))
      .attr('stroke-width', 1.25)
      .on('mouseenter', (_event, d) => {
        setHoveredBin(null);
        setHoveredCell(d);
      })
      .on('mouseleave', () => {
        setHoveredCell(null);
      })
      .on('click', (_event, d) => {
        onSelectObject(d.id === selectedObjectId ? null : d.id);
      });
  }, [cellData, chartWidth, mode, binPreset, selectedObjectId, stats, valueOf, formatVal, lang, onSelectObject]);

  if (cellData.length === 0) return null;

  const areaUnit = is3D ? `${spaceUnit}³` : `${spaceUnit}²`;
  const sym = is3D ? 'V' : 'A';

  return (
    <div
      ref={containerRef}
      className="pt-3 border-t border-slate-200 dark:border-slate-800 space-y-2"
    >
      {/* Header Row: Title + Metric Mode Switch */}
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="text-[12px] font-semibold text-slate-800 dark:text-slate-100 truncate">
            {is3D
              ? lang === 'zh'
                ? '3D 核质体积比分布直方图 (N/C Vol)'
                : '3D Nuclear-to-Cytoplasmic Volume Ratio'
              : lang === 'zh'
              ? '核质面积比分布直方图 (N/C)'
              : 'Nuclear-to-Cytoplasmic Ratio Distribution'}
          </div>
          <div className="text-[10.5px] text-slate-500 dark:text-slate-400 font-mono tabular-nums truncate">
            {mode === 'nc_pct'
              ? lang === 'zh'
                ? `${sym}_核 / ${sym}_胞质 (%) · 点击柱体或基线点定位细胞`
                : `${sym}_nuc / ${sym}_cyto (%) · Click bar or dot to locate cell`
              : mode === 'nc_ratio'
              ? lang === 'zh'
                ? `${sym}_核 / ${sym}_胞质 (无量纲比值)`
                : `${sym}_nuc / ${sym}_cyto (dimensionless ratio)`
              : lang === 'zh'
              ? `${sym}_核 / ${sym}_全细胞 (%)`
              : `${sym}_nuc / ${sym}_cell (%)`}
          </div>
        </div>

        {/* Interactive Mode Segmented Switch */}
        <div className="flex items-center gap-0.5 p-0.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#0B0F1A] shrink-0 text-[10px] font-mono">
          {(
            [
              { id: 'nc_pct', label: 'N/C %' },
              { id: 'nc_ratio', label: 'N/C' },
              { id: 'ncell_pct', label: 'N/Cell' },
            ] as Array<{ id: RatioDisplayMode; label: string }>
          ).map((opt) => (
            <button
              key={opt.id}
              type="button"
              onClick={() => setMode(opt.id)}
              className={`px-1.5 py-0.5 rounded whitespace-nowrap cursor-pointer transition-colors ${
                mode === opt.id
                  ? 'bg-white dark:bg-slate-800 text-blue-600 dark:text-cyan-400 font-semibold'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Live Hover / Summary Readout Bar (Zero Layout Shift) */}
      <div className="flex items-center justify-between gap-2 text-[10.5px] font-mono tabular-nums text-slate-600 dark:text-slate-300 min-h-[18px]">
        {hoveredCell ? (
          <div className="flex items-center gap-2 truncate">
            <span className="font-semibold text-blue-600 dark:text-cyan-400">
              {lang === 'zh' ? `细胞 #${hoveredCell.id}` : `Cell #${hoveredCell.id}`}
            </span>
            <span>N/C {formatVal(valueOf(hoveredCell))}</span>
            <span className="text-slate-400">·</span>
            <span className="text-slate-500 dark:text-slate-400 truncate">
              {lang === 'zh'
                ? `核 ${hoveredCell.nuclearArea.toFixed(0)} / 质 ${hoveredCell.cytoplasmArea.toFixed(0)} ${areaUnit}`
                : `N ${hoveredCell.nuclearArea.toFixed(0)} / C ${hoveredCell.cytoplasmArea.toFixed(0)} ${areaUnit}`}
            </span>
          </div>
        ) : hoveredBin ? (
          <div className="flex items-center gap-2 truncate">
            <span className="font-semibold text-blue-600 dark:text-cyan-400">
              [{formatVal(hoveredBin.x0, true)}, {formatVal(hoveredBin.x1, true)})
            </span>
            <span>
              n={hoveredBin.cells.length} ({hoveredBin.sharePct.toFixed(0)}%)
            </span>
            <span className="text-slate-400">·</span>
            <span className="text-slate-500 dark:text-slate-400 truncate">
              {hoveredBin.cells.map((c) => `#${c.id}`).join(', ')}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 truncate">
            <span>
              μ = <strong className="text-slate-800 dark:text-slate-200">{formatVal(stats.mean)}</strong>
            </span>
            <span aria-hidden="true">·</span>
            <span>
              M = <strong className="text-slate-800 dark:text-slate-200">{formatVal(stats.median)}</strong>
            </span>
            {stats.n > 1 && (
              <>
                <span aria-hidden="true">·</span>
                <span>σ = {formatVal(stats.sd)}</span>
              </>
            )}
          </div>
        )}

        {/* Bin Count Control */}
        <div className="flex items-center gap-1 shrink-0 text-[10px] text-slate-500 dark:text-slate-400">
          <span>{lang === 'zh' ? '分箱' : 'Bins'}</span>
          <select
            aria-label={lang === 'zh' ? '直方图分箱数' : 'Histogram bin count'}
            value={String(binPreset)}
            onChange={(e) => {
              const v = e.target.value;
              setBinPreset(v === 'auto' ? 'auto' : (Number(v) as BinPreset));
            }}
            className="h-5 px-1 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 font-mono text-[10px] cursor-pointer focus:outline-none focus:border-blue-500 dark:focus:border-cyan-400"
          >
            <option value="auto">{lang === 'zh' ? '自动' : 'Auto'}</option>
            <option value="5">5</option>
            <option value="8">8</option>
            <option value="12">12</option>
          </select>
        </div>
      </div>

      {/* D3 SVG Plot Canvas */}
      <div className="rounded border border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-[#0B0F1A] px-1.5 pt-1 pb-0.5">
        <svg ref={svgRef} className="w-full h-auto block select-none overflow-visible" />
      </div>

      {/* Compact Tabular Stats Footer Strip (Unboxed Metadata, Hairline Dividers) */}
      <div className="grid grid-cols-3 gap-2 pt-1 text-[10.5px] font-mono tabular-nums text-slate-500 dark:text-slate-400">
        <div>
          <div className="text-[9.5px] text-slate-400 dark:text-slate-500">
            {lang === 'zh' ? '均值 N/C' : 'Mean N/C'}
          </div>
          <div className="font-semibold text-slate-800 dark:text-slate-100">
            {formatVal(stats.mean)}
          </div>
        </div>
        <div>
          <div className="text-[9.5px] text-slate-400 dark:text-slate-500">
            {lang === 'zh' ? '分布区间' : 'Min – Max'}
          </div>
          <div className="font-semibold text-slate-800 dark:text-slate-100 truncate">
            {formatVal(stats.min, true)}–{formatVal(stats.max, true)}
          </div>
        </div>
        <div className="text-right">
          <div className="text-[9.5px] text-slate-400 dark:text-slate-500">
            {lang === 'zh' ? '有效细胞' : 'Cells (n)'}
          </div>
          <div className="font-semibold text-slate-800 dark:text-slate-100">
            {stats.n}
          </div>
        </div>
      </div>
    </div>
  );
}
