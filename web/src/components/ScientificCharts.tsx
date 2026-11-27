import { useState } from 'react';
import type { Lang, RunResult } from '../types/vexel';

interface HistogramProps {
  data: NonNullable<RunResult['histogram']>;
  lang: Lang;
}

export function DistributionHistogram({ data, lang }: HistogramProps) {
  const [hoverBin, setHoverBin] = useState<number | null>(null);
  if (!data.values.length) return null;

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.values.length; i++) {
    const v = data.values[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = Math.max(1e-6, max - min);
  const numBins = Math.min(14, Math.max(6, Math.round(Math.sqrt(data.values.length) * 1.8)));
  const counts = new Array(numBins).fill(0);
  for (const v of data.values) {
    const idx = Math.min(numBins - 1, Math.floor(((v - min) / span) * numBins));
    counts[idx]++;
  }
  const maxCount = Math.max(1, ...counts);
  const meanX = ((data.mean - min) / span) * 280 + 18;

  const w = 316;
  const h = 112;
  const plotW = 280;
  const plotH = 74;
  const padL = 18;
  const padT = 14;
  const barW = plotW / numBins;

  return (
    <div className="pt-3 border-t border-slate-200 dark:border-slate-800">
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-[12px] font-medium text-slate-700 dark:text-slate-200">
          {lang === 'zh' ? data.labelZh : data.labelEn}
        </span>
        <span className="font-mono text-[11px] tabular-nums text-slate-500 dark:text-slate-400">
          {hoverBin !== null
            ? `${(min + (hoverBin / numBins) * span).toFixed(1)}–${(min + ((hoverBin + 1) / numBins) * span).toFixed(1)} ${data.unit} · n=${counts[hoverBin]}`
            : `μ = ${data.mean.toFixed(2)} ${data.unit} · n = ${data.values.length}`}
        </span>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto block select-none">
        {/* Horizontal hairlines */}
        <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="currentColor" className="text-slate-300 dark:text-slate-700" strokeWidth="1" />
        <line x1={padL} y1={padT + plotH * 0.5} x2={padL + plotW} y2={padT + plotH * 0.5} stroke="currentColor" className="text-slate-200 dark:text-slate-800" strokeWidth="0.75" strokeDasharray="2 3" />

        {counts.map((c, idx) => {
          const bh = (c / maxCount) * plotH;
          const x = padL + idx * barW + 1;
          const y = padT + plotH - bh;
          const isHover = hoverBin === idx;
          return (
            <rect
              key={idx}
              x={x}
              y={y}
              width={Math.max(2, barW - 2)}
              height={Math.max(0, bh)}
              rx="1.5"
              className={
                isHover
                  ? 'fill-blue-600 dark:fill-cyan-400 transition-colors'
                  : 'fill-blue-500/75 dark:fill-cyan-500/70 transition-colors'
              }
              onMouseEnter={() => setHoverBin(idx)}
              onMouseLeave={() => setHoverBin(null)}
            />
          );
        })}

        {/* Mean marker line */}
        <line
          x1={meanX}
          y1={padT - 2}
          x2={meanX}
          y2={padT + plotH}
          stroke="#F59E0B"
          strokeWidth="1.25"
          strokeDasharray="3 2"
        />

        {/* Axis labels */}
        <text x={padL} y={h - 4} fontSize="9.5" className="fill-slate-500 dark:fill-slate-400 font-mono">
          {min.toFixed(1)}
        </text>
        <text x={padL + plotW * 0.5} y={h - 4} textAnchor="middle" fontSize="9.5" className="fill-slate-500 dark:fill-slate-400 font-mono">
          {((min + max) * 0.5).toFixed(1)} {data.unit}
        </text>
        <text x={padL + plotW} y={h - 4} textAnchor="end" fontSize="9.5" className="fill-slate-500 dark:fill-slate-400 font-mono">
          {max.toFixed(1)}
        </text>
      </svg>
    </div>
  );
}

interface PoissonProps {
  data: NonNullable<RunResult['poissonChart']>;
  lang: Lang;
}

export function PoissonComparisonChart({ data, lang }: PoissonProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const w = 316;
  const h = 118;
  const padL = 24;
  const padT = 14;
  const plotW = 276;
  const plotH = 76;
  const maxPct = Math.max(10, ...data.observedPct, ...data.expectedPct);
  const groupW = plotW / Math.max(1, data.counts.length);

  return (
    <div className="pt-3 border-t border-slate-200 dark:border-slate-800">
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-[12px] font-medium text-slate-700 dark:text-slate-200">
          {lang === 'zh' ? '细胞包裹分布 vs 泊松理论' : 'Encapsulation vs Poisson'}
        </span>
        <span className="font-mono text-[11px] tabular-nums text-slate-500 dark:text-slate-400">
          {hoverIdx !== null
            ? `k=${data.counts[hoverIdx]} · Obs ${data.observedPct[hoverIdx].toFixed(1)}% · Pois ${data.expectedPct[hoverIdx].toFixed(1)}%`
            : `λ = ${data.lambda.toFixed(2)}`}
        </span>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto block select-none">
        <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="currentColor" className="text-slate-300 dark:text-slate-700" strokeWidth="1" />
        <line x1={padL} y1={padT} x2={padL + plotW} y2={padT} stroke="currentColor" className="text-slate-200 dark:text-slate-800" strokeWidth="0.75" strokeDasharray="2 3" />
        <text x={padL - 4} y={padT + 4} textAnchor="end" fontSize="9" className="fill-slate-400 font-mono">
          {Math.round(maxPct)}%
        </text>
        <text x={padL - 4} y={padT + plotH} textAnchor="end" fontSize="9" className="fill-slate-400 font-mono">
          0
        </text>

        {data.counts.map((k, idx) => {
          const cx = padL + idx * groupW + groupW * 0.5;
          const bw = Math.min(16, groupW * 0.34);
          const obsH = (data.observedPct[idx] / maxPct) * plotH;
          const expH = (data.expectedPct[idx] / maxPct) * plotH;
          return (
            <g
              key={k}
              onMouseEnter={() => setHoverIdx(idx)}
              onMouseLeave={() => setHoverIdx(null)}
              className="cursor-default"
            >
              <rect
                x={cx - bw - 1}
                y={padT + plotH - obsH}
                width={bw}
                height={Math.max(0, obsH)}
                rx="1.5"
                className="fill-blue-600 dark:fill-cyan-400"
              />
              <rect
                x={cx + 1}
                y={padT + plotH - expH}
                width={bw}
                height={Math.max(0, expH)}
                rx="1.5"
                className="fill-slate-400/60 dark:fill-slate-500/70"
              />
              <text
                x={cx}
                y={h - 5}
                textAnchor="middle"
                fontSize="9.5"
                className="fill-slate-500 dark:fill-slate-400 font-mono"
              >
                {k === data.counts[data.counts.length - 1] && k >= 4 ? `${k}+` : `${k}`}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="flex items-center gap-4 mt-1 text-[11px] text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2 rounded-[1px] bg-blue-600 dark:bg-cyan-400" />
          {lang === 'zh' ? '实测比例' : 'Observed'}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2 rounded-[1px] bg-slate-400/70 dark:bg-slate-500" />
          {lang === 'zh' ? '泊松理论' : 'Poisson'}
        </span>
      </div>
    </div>
  );
}

interface ProfileProps {
  positions: number[];
  intensities: number[];
  edges: number[];
  unit: string;
  title: string;
}

export function IntensityProfileChart({ positions, intensities, edges, unit, title }: ProfileProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  if (positions.length < 2) return null;

  const w = 316;
  const h = 112;
  const padL = 26;
  const padR = 8;
  const padT = 12;
  const plotW = w - padL - padR;
  const plotH = 74;

  const minPos = positions[0];
  const maxPos = positions[positions.length - 1];
  const posSpan = Math.max(1e-6, maxPos - minPos);
  const minI = Math.min(...intensities);
  const maxI = Math.max(...intensities);
  const iSpan = Math.max(1, maxI - minI);

  const toX = (p: number) => padL + ((p - minPos) / posSpan) * plotW;
  const toY = (val: number) => padT + plotH - ((val - minI) / iSpan) * plotH;

  const polyPoints = positions
    .map((p, i) => `${toX(p).toFixed(1)},${toY(intensities[i]).toFixed(1)}`)
    .join(' ');

  return (
    <div className="pt-3 border-t border-slate-200 dark:border-slate-800">
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-[12px] font-medium text-slate-700 dark:text-slate-200">{title}</span>
        <span className="font-mono text-[11px] tabular-nums text-slate-500 dark:text-slate-400">
          {hoverIdx !== null
            ? `${positions[hoverIdx].toFixed(1)} ${unit} · I=${intensities[hoverIdx].toFixed(0)}`
            : `${edges.length} edges`}
        </span>
      </div>

      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full h-auto block select-none"
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const relX = ((e.clientX - rect.left) / rect.width) * w;
          const frac = Math.max(0, Math.min(1, (relX - padL) / plotW));
          setHoverIdx(Math.min(positions.length - 1, Math.round(frac * (positions.length - 1))));
        }}
        onMouseLeave={() => setHoverIdx(null)}
      >
        <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="currentColor" className="text-slate-300 dark:text-slate-700" strokeWidth="1" />
        <line x1={padL} y1={padT} x2={padL + plotW} y2={padT} stroke="currentColor" className="text-slate-200 dark:text-slate-800" strokeWidth="0.75" strokeDasharray="2 3" />

        <text x={padL - 4} y={padT + 4} textAnchor="end" fontSize="8.5" className="fill-slate-400 font-mono">
          {Math.round(maxI)}
        </text>
        <text x={padL - 4} y={padT + plotH} textAnchor="end" fontSize="8.5" className="fill-slate-400 font-mono">
          {Math.round(minI)}
        </text>

        {edges.map((ep, idx) => (
          <line
            key={idx}
            x1={toX(ep)}
            y1={padT}
            x2={toX(ep)}
            y2={padT + plotH}
            stroke="#F59E0B"
            strokeWidth="1"
            strokeDasharray="2 2"
            opacity="0.85"
          />
        ))}

        <polyline
          fill="none"
          strokeWidth="1.5"
          className="stroke-blue-600 dark:stroke-cyan-400"
          points={polyPoints}
        />

        {hoverIdx !== null && (
          <g>
            <line
              x1={toX(positions[hoverIdx])}
              y1={padT}
              x2={toX(positions[hoverIdx])}
              y2={padT + plotH}
              stroke="currentColor"
              className="text-slate-400 dark:text-slate-500"
              strokeWidth="0.8"
            />
            <circle
              cx={toX(positions[hoverIdx])}
              cy={toY(intensities[hoverIdx])}
              r="2.8"
              className="fill-blue-600 dark:fill-cyan-300"
            />
          </g>
        )}

        <text x={padL} y={h - 4} fontSize="9" className="fill-slate-500 dark:fill-slate-400 font-mono">
          {minPos.toFixed(0)}
        </text>
        <text x={padL + plotW} y={h - 4} textAnchor="end" fontSize="9" className="fill-slate-500 dark:fill-slate-400 font-mono">
          {maxPos.toFixed(0)} {unit}
        </text>
      </svg>
    </div>
  );
}
