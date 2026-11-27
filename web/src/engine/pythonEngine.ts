/**
 * Client of the validated Python engine (`vexel serve`, default http://127.0.0.1:8765).
 *
 * The browser engine in this workbench gives a quick preview; the numbers that count come from
 * the Python engine, whose measurements passed the synthetic and real-image acceptance. This
 * module sends the image the workbench shows (its grey pixels, the person's calibration and the
 * data-bar crop) and returns the engine's result - numbers, units, warnings and outlines.
 * The engine runs on this computer: nothing leaves it.
 */

import type { Specimen } from '../types/vexel';

/** Web workflow id -> Python SEM measurement and its settings. Workflows not listed have no
 * validated engine yet. Layer stacks may be tilted, bent or vertical: the Python engine
 * straightens them and measures each thickness along the local normal. */
export const PYTHON_SEM_RECIPE: Record<string, { recipe: string; params: Record<string, unknown> }> = {
  sem_lines: { recipe: 'lines', params: {} },
  sem_holes: { recipe: 'holes', params: {} },
  sem_particles: { recipe: 'particles', params: {} },
  sem_fibres: { recipe: 'fibres', params: {} },
  sem_multilayer: { recipe: 'layers', params: { geometry: 'auto', orientation: 'auto' } },
  // cells bounded by dark membranes, each with a dark nucleus (the web workflow's default)
  nested_cell_nucleus: { recipe: 'nested', params: { parent: 'membrane', child: 'dark' } },
  sem_curved_multilayer: { recipe: 'layers', params: { geometry: 'curved', orientation: 'auto' } },
};

export interface PythonSummaryItem {
  key: string;
  label: string;
  value: number | string | null;
  unit: string;
  kind: string;
}

export interface PythonShape {
  kind: 'polygon' | 'polyline' | 'points' | 'ellipse';
  points: number[][] | number[];
  color: string;
  anchor: number[];
  label: string;
}

export interface PythonMeasureResult {
  engine: string;
  version: string;
  validated: boolean;
  recipe: string;
  recipe_label: string;
  calibration: { source: string; confirmed: boolean; text: string; warnings?: string[] };
  unit: string;
  summary: PythonSummaryItem[];
  objects: Array<Record<string, unknown>>;
  shapes: PythonShape[];
  warnings: string[];
  error: string | null;
  region: number[] | null;
  crop_rows: number | null;
  image: { width: number; height: number; sha256: string };
  settings: Record<string, unknown>;
}

const UNIT_OK = new Set(['nm', 'µm', 'mm']);

function float32ToBase64(data: Float32Array): string {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export function pythonRecipeFor(recipeId: string): string | null {
  return PYTHON_SEM_RECIPE[recipeId]?.recipe ?? null;
}

/** The browser preview's quantity that the Python engine measures too, per Python recipe: the
 * browser summary key and how to read the same quantity from the Python summary. Only these
 * pairs are compared; other numbers measure different things. */
export const SAME_QUANTITY: Record<string, { web: string; python: (s: PythonSummaryItem[], unit: string) => { value: number; unit: string } | null }> = {
  lines: { web: 'cd_mean', python: (s, u) => withUnit(num(s, 'cd_mean'), u) },
  holes: { web: 'diameter_mean', python: (s, u) => withUnit(num(s, 'diameter_mean'), u) },
  particles: { web: 'd50_n', python: (s, u) => withUnit(num(s, 'd50_n'), u) },
  fibres: { web: 'diameter_mean', python: (s, u) => withUnit(num(s, 'diameter_mean'), u) },
  // the stack between its first and last interface: the sum of the layers
  layers: {
    web: 'total_thickness',
    python: (s, u) => {
      const t = s.filter((x) => /^layer\d+$/.test(x.key) && typeof x.value === 'number');
      return withUnit(t.length ? t.reduce((a, x) => a + (x.value as number), 0) : null, u);
    },
  },
  // mean nucleus / cell area, in percent on both sides (unit-free: compared even uncalibrated)
  nested: { web: 'mean_sub_area_ratio_pct', python: (s) => withUnit(num(s, 'ratio_mean') === null ? null : (num(s, 'ratio_mean') as number) * 100, '%') },
};

/** How many objects each engine found, per Python recipe (browser summary key, Python summary
 * key): a ratio or mean that agrees is no agreement when one engine found 1 cell and the other 13. */
export const SAME_COUNT: Record<string, { web: string; python: string }> = {
  holes: { web: 'holes', python: 'holes' },
  particles: { web: 'particles', python: 'particles' },
  layers: { web: 'layer_count', python: 'interfaces' }, // Python counts interfaces: layers = interfaces - 1
  nested: { web: 'parent_count', python: 'parents' },
};

/** The Python count comparable with the browser's (layers: interfaces - 1). */
export function pythonCount(recipe: string, s: PythonSummaryItem[]): number | null {
  const pair = SAME_COUNT[recipe];
  const v = pair ? num(s, pair.python) : null;
  if (v === null) return null;
  return recipe === 'layers' ? Math.max(0, v - 1) : v;
}

function withUnit(value: number | null, unit: string): { value: number; unit: string } | null {
  return value === null ? null : { value, unit };
}

function num(s: PythonSummaryItem[], key: string): number | null {
  const v = s.find((x) => x.key === key)?.value;
  return typeof v === 'number' ? v : null;
}

/** The settings the Python measurement gets for this workflow. */
export function pythonParamsFor(recipeId: string): Record<string, unknown> {
  return { ...(PYTHON_SEM_RECIPE[recipeId]?.params ?? {}) };
}

/** The request body for one specimen (exported for tests). */
export function measureRequest(specimen: Specimen, recipe: string, lang: 'zh' | 'en', params: Record<string, unknown> = {}) {
  const cal = specimen.calibration;
  // only a measured calibration is sent: the instrument's metadata, the person's own (drawn on
  // the scale bar), or a data bar the person confirmed. The workbench's defaults for uploads
  // (2 nm/px for SEM without a bar; a bar assumed to read 500 nm) are not measurements - the
  // Python engine then reports lengths in pixels and says the image is uncalibrated.
  const measured = cal.source === 'metadata' || cal.source === 'manual' || (cal.source === 'databar' && cal.confirmed);
  const calibrated = measured && UNIT_OK.has(cal.spaceUnit) && cal.pixelSize > 0;
  const gray = specimen.gray instanceof Float32Array ? specimen.gray : new Float32Array(specimen.gray);
  return {
    recipe,
    params,
    pixels: float32ToBase64(gray),
    width: specimen.width,
    height: specimen.height,
    pixel_size: calibrated ? cal.pixelSize : null,
    unit: calibrated ? cal.spaceUnit : 'px',
    crop_rows: cal.cropRows && cal.cropRows > 0 && cal.cropRows < specimen.height ? cal.cropRows : null,
    lang,
  };
}

export async function measureWithPython(
  baseUrl: string,
  specimen: Specimen,
  recipe: string,
  lang: 'zh' | 'en',
  params: Record<string, unknown> = {},
  signal?: AbortSignal
): Promise<PythonMeasureResult> {
  const url = baseUrl.replace(/\/+$/, '') + '/api/v1/sem/measure';
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(measureRequest(specimen, recipe, lang, params)),
    signal,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(typeof data?.detail === 'string' ? data.detail : `Python engine returned HTTP ${resp.status}`);
  }
  return data as PythonMeasureResult;
}
