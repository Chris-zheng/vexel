import type { BatchRow, ComparisonRow, GroupStatRow } from '../types/vexel';

/** Regularized incomplete beta / Student-t two-tailed p-value approximation (accurate to ~1e-4). */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-ax * ax);
  return sign * y;
}

function twoTailedPFromT(tAbs: number, df: number): number {
  if (!Number.isFinite(tAbs) || df <= 0) return 1;
  // Hill's normal-approximation transform for Student's t -> z (accurate even at modest df >= 2)
  const a = df - 0.5;
  const b = 48 * a * a;
  let z = a * Math.log(1 + (tAbs * tAbs) / df);
  z = Math.sqrt(z * (1 + (((-2 * z - 3) / (b + 1e-9)) + 1 / (24 * a))));
  const p = 1 - erf(z / Math.SQRT2);
  return Math.max(0.0001, Math.min(1, p));
}

function tCrit95(df: number): number {
  const table: Record<number, number> = { 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228 };
  const d = Math.max(1, Math.round(df));
  return table[d] ?? (d < 30 ? 2.08 : 1.96);
}

const METRIC_DEFS: Array<{ key: string; zh: string; en: string; unit: string }> = [
  { key: 'diameter_mean', zh: '平均直径 / 孔径', en: 'Mean Diameter', unit: 'µm/nm' },
  { key: 'diameter_cv', zh: '粒径 CV', en: 'Diameter CV', unit: '%' },
  { key: 'lambda', zh: '平均包裹数 λ', en: 'Encapsulation λ', unit: 'cells/drop' },
  { key: 'positive_fraction', zh: '荧光阳性比例', en: 'Positive Fraction', unit: '' },
  { key: 'cd_mean', zh: '平均线宽 CD', en: 'Line CD Mean', unit: 'nm' },
  { key: 'ler_3sigma_unbiased', zh: 'LER 3σ（去噪）', en: 'LER 3σ (Unbiased)', unit: 'nm' },
  { key: 'd50_n', zh: '颗粒中位径 d50', en: 'Particle Median d50', unit: 'nm' },
  { key: 'porosity', zh: '孔隙率', en: 'Porosity', unit: '' },
];

/**
 * Replicate-unit experiment statistics (mirrors vexel/batch/stats.py):
 * Multiple files sharing (group, replicate) are averaged first to prevent pseudoreplication.
 */
export function computeBatchStatistics(rows: BatchRow[]): {
  groupStats: GroupStatRow[];
  comparisons: ComparisonRow[];
} {
  const completed = rows.filter((r) => (r.status === 'completed' || r.status === 'warning') && r.result);
  if (completed.length === 0) return { groupStats: [], comparisons: [] };

  const groupStats: GroupStatRow[] = [];
  const comparisons: ComparisonRow[] = [];

  for (const m of METRIC_DEFS) {
    // Step 1: Aggregate per (group, replicate)
    const unitBuckets = new Map<string, { group: string; replicate: string; vals: number[] }>();
    let detectedUnit = m.unit;

    for (const row of completed) {
      const val = row.result!.summaryMap[m.key];
      if (typeof val !== 'number' || !Number.isFinite(val)) continue;
      if (m.key === 'diameter_mean') detectedUnit = row.specimen.calibration.spaceUnit;
      const repKey = row.replicate.trim() || row.id;
      const bKey = `${row.group}|||${repKey}`;
      let b = unitBuckets.get(bKey);
      if (!b) {
        b = { group: row.group || 'Default', replicate: repKey, vals: [] };
        unitBuckets.set(bKey, b);
      }
      b.vals.push(val);
    }

    if (unitBuckets.size === 0) continue;

    // Step 2: Group replicate values
    const byGroup = new Map<string, number[]>();
    for (const b of unitBuckets.values()) {
      const repMean = b.vals.reduce((a, c) => a + c, 0) / b.vals.length;
      let arr = byGroup.get(b.group);
      if (!arr) {
        arr = [];
        byGroup.set(b.group, arr);
      }
      arr.push(repMean);
    }

    const groupNames = Array.from(byGroup.keys()).sort();
    for (const g of groupNames) {
      const vals = [...byGroup.get(g)!].sort((a, b) => a - b);
      const n = vals.length;
      const mean = vals.reduce((a, b) => a + b, 0) / n;
      const variance = n > 1 ? vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : 0;
      const sd = Math.sqrt(variance);
      const se = sd / Math.sqrt(n);
      const halfCi = n > 1 ? tCrit95(n - 1) * se : 0;
      const median = n % 2 === 1 ? vals[(n - 1) / 2] : 0.5 * (vals[n / 2 - 1] + vals[n / 2]);

      groupStats.push({
        metricKey: m.key,
        metricLabelZh: m.zh,
        metricLabelEn: m.en,
        unit: detectedUnit,
        group: g,
        nReplicates: n,
        mean: Number(mean.toFixed(3)),
        sd: Number(sd.toFixed(3)),
        ci95Low: Number((mean - halfCi).toFixed(3)),
        ci95High: Number((mean + halfCi).toFixed(3)),
        median: Number(median.toFixed(3)),
      });
    }

    // Step 3: Pairwise Welch's t-test with Holm-Bonferroni correction across pairs
    const rawPairs: ComparisonRow[] = [];
    for (let i = 0; i < groupNames.length; i++) {
      for (let j = i + 1; j < groupNames.length; j++) {
        const a = byGroup.get(groupNames[i])!;
        const b = byGroup.get(groupNames[j])!;
        if (a.length < 2 || b.length < 2) continue;
        const ma = a.reduce((s, v) => s + v, 0) / a.length;
        const mb = b.reduce((s, v) => s + v, 0) / b.length;
        const va = a.reduce((s, v) => s + (v - ma) ** 2, 0) / (a.length - 1);
        const vb = b.reduce((s, v) => s + (v - mb) ** 2, 0) / (b.length - 1);
        const se2 = va / a.length + vb / b.length;
        const tStat = se2 > 1e-12 ? Math.abs(ma - mb) / Math.sqrt(se2) : 0;
        const df =
          se2 > 1e-12
            ? (se2 * se2) / ((va / a.length) ** 2 / (a.length - 1) + (vb / b.length) ** 2 / (b.length - 1))
            : a.length + b.length - 2;
        const pRaw = twoTailedPFromT(tStat, df);

        // Hedges' g effect size with small-sample J correction
        const sp = Math.sqrt(((a.length - 1) * va + (b.length - 1) * vb) / Math.max(1, a.length + b.length - 2));
        const jCorr = 1 - 3 / (4 * (a.length + b.length) - 9);
        const g = sp > 1e-12 ? ((mb - ma) / sp) * jCorr : 0;

        rawPairs.push({
          metricKey: m.key,
          metricLabelZh: m.zh,
          metricLabelEn: m.en,
          groupA: groupNames[i],
          groupB: groupNames[j],
          testName: groupNames.length > 2 ? "Welch's t (Holm)" : "Welch's t-test",
          diff: Number((mb - ma).toFixed(3)),
          pValue: pRaw,
          hedgesG: Number(g.toFixed(2)),
          significant: false,
        });
      }
    }

    // Holm adjustment
    const sortedIdx = rawPairs.map((_, idx) => idx).sort((i, j) => rawPairs[i].pValue - rawPairs[j].pValue);
    let runningMax = 0;
    for (let rank = 0; rank < sortedIdx.length; rank++) {
      const pair = rawPairs[sortedIdx[rank]];
      const adj = Math.min(1, pair.pValue * (sortedIdx.length - rank));
      runningMax = Math.max(runningMax, adj);
      pair.pValue = Number(runningMax.toFixed(4));
      pair.significant = pair.pValue < 0.05;
    }
    comparisons.push(...rawPairs);
  }

  return { groupStats, comparisons };
}
