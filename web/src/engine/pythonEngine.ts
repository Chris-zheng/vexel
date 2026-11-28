/**
 * Client of the validated Python engine (`vexel serve`, default http://127.0.0.1:8765).
 *
 * The browser engine in this workbench gives a quick preview; the numbers that count come from
 * the Python engine, whose measurements passed the synthetic and real-image acceptance. This
 * module sends the uploaded file itself (full resolution and bit depth, the instrument's metadata)
 * or, for the built-in samples, the grey pixels the workbench shows - with the person's
 * calibration and data-bar crop - and returns the engine's result: numbers, units, warnings and
 * the outlines to draw on the canvas.
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

/** Files the Python engine reads itself (TIFF keeps the instrument metadata; the others it reads
 * at full resolution). Other uploads (Gatan DM3/DM4) and the built-in samples go as pixels. */
const FILE_TYPES = /\.(tiff?|png|jpe?g|bmp)$/i;

/** What the Python engine is sent for this specimen: its original file, or the pixels shown. */
export interface SourceInput {
  name: string;
  file: Blob;
  width: number; // pixels of the original file
  height: number;
}

export function sourceInputFor(specimen: Specimen): SourceInput | null {
  const f = specimen.sourceFile;
  if (!f || !FILE_TYPES.test(specimen.name) || !specimen.sourceWidth || !specimen.sourceHeight) return null;
  return { name: specimen.name, file: f, width: specimen.sourceWidth, height: specimen.sourceHeight };
}

/** Original-file pixels per displayed pixel (the workbench shows uploads at most 960 px wide). */
function sourcePerShown(specimen: Specimen, src: SourceInput): { x: number; y: number } {
  const w = specimen.rawWidth ?? specimen.width;
  const h = specimen.rawHeight ?? specimen.height;
  return { x: src.width / Math.max(1, w), y: src.height / Math.max(1, h) };
}

/** The request body for one specimen (exported for tests). With ``fileBase64`` (the specimen's
 * original file, see ``sourceInputFor``) Python reads the file itself: full resolution and bit
 * depth, and - unless the person calibrated - the instrument's metadata or data bar. */
export function measureRequest(
  specimen: Specimen,
  recipe: string,
  lang: 'zh' | 'en',
  params: Record<string, unknown> = {},
  fileBase64?: string
) {
  const cal = specimen.calibration;
  // only a measured calibration is sent: the instrument's metadata, the person's own (drawn on
  // the scale bar), or a data bar the person confirmed. The workbench's defaults for uploads
  // (2 nm/px for SEM without a bar; a bar assumed to read 500 nm) are not measurements - the
  // Python engine then reports lengths in pixels and says the image is uncalibrated.
  const measured = cal.source === 'metadata' || cal.source === 'manual' || (cal.source === 'databar' && cal.confirmed);
  const calibrated = measured && UNIT_OK.has(cal.spaceUnit) && cal.pixelSize > 0;
  const src = fileBase64 ? sourceInputFor(specimen) : null;
  if (src && fileBase64) {
    const k = sourcePerShown(specimen, src);
    // the instrument's metadata: Python reads it from the file itself. The person's calibration
    // is per displayed pixel; the file's pixels are smaller by the downscale.
    const own = calibrated && cal.source !== 'metadata';
    // a data bar the person cut off: at the bottom, Python measures the rows above it; at the top,
    // the region below it. Not cut: Python finds the data bar itself.
    const cut = specimen.rawCroppedBarPx && specimen.rawHeight ? specimen.rawCroppedBarPx : 0;
    const bottom = cut > 0 && specimen.rawBarPosition !== 'top';
    const top = cut > 0 && specimen.rawBarPosition === 'top';
    return {
      recipe,
      params,
      file: fileBase64,
      filename: src.name,
      pixel_size: own ? cal.pixelSize / k.x : null,
      unit: own ? cal.spaceUnit : 'nm',
      crop_rows: bottom ? Math.floor(((specimen.rawHeight as number) - cut) * k.y + 1e-6) : null,
      region: top ? [0, Math.ceil(cut * k.y - 1e-6), src.width, src.height] : null,
      lang,
    };
  }
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

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** A Python result with what it was measured on. ``fileRefused`` is Python's reason when it could
 * not read the original file and the shown pixels were measured instead. */
export interface PythonMeasurement {
  result: PythonMeasureResult;
  input: 'file' | 'pixels';
  fileRefused?: string;
}

async function post(url: string, body: unknown, signal?: AbortSignal): Promise<PythonMeasureResult> {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(typeof data?.detail === 'string' ? data.detail : `Python engine returned HTTP ${resp.status}`);
  }
  return data as PythonMeasureResult;
}

export async function measureWithPython(
  baseUrl: string,
  specimen: Specimen,
  recipe: string,
  lang: 'zh' | 'en',
  params: Record<string, unknown> = {},
  signal?: AbortSignal
): Promise<PythonMeasurement> {
  const url = baseUrl.replace(/\/+$/, '') + '/api/v1/sem/measure';
  const src = sourceInputFor(specimen);
  let fileRefused: string | undefined;
  if (src) {
    const b64 = bytesToBase64(new Uint8Array(await src.file.arrayBuffer()));
    try {
      return { result: await post(url, measureRequest(specimen, recipe, lang, params, b64), signal), input: 'file' };
    } catch (err) {
      if (signal?.aborted || !(err instanceof Error) || err.name === 'TypeError') throw err; // stopped, or no connection
      fileRefused = err.message; // Python could not read the file (format, size): measure the shown pixels
    }
  }
  return { result: await post(url, measureRequest(specimen, recipe, lang, params), signal), input: 'pixels', fileRefused };
}

/** Python's outlines in the coordinates of the image the canvas shows: an original file was
 * measured at full resolution, before the person's data-bar crop; the canvas shows it downscaled
 * and cropped. */
export function shapesForCanvas(specimen: Specimen, m: PythonMeasurement): PythonShape[] {
  const src = m.input === 'file' ? sourceInputFor(specimen) : null;
  const k = src ? sourcePerShown(specimen, src) : { x: 1, y: 1 };
  const dy = src && specimen.rawBarPosition === 'top' ? specimen.rawCroppedBarPx ?? 0 : 0;
  const pt = (p: number[]) => [p[0] / k.x, p[1] / k.y - dy];
  return m.result.shapes.map((sh) => ({
    ...sh,
    points: sh.kind === 'ellipse' ? sh.points : (sh.points as number[][]).map(pt),
    anchor: sh.anchor.length >= 2 ? pt(sh.anchor) : sh.anchor,
  }));
}
