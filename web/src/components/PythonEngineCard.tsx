import { useState } from 'react';
import { CheckCircle2, Loader2, ShieldCheck, XCircle } from 'lucide-react';
import type { Specimen } from '../types/vexel';
import { measureWithPython, pythonCount, pythonParamsFor, pythonRecipeFor, SAME_COUNT, SAME_QUANTITY, shapesForCanvas, type PythonMeasureResult, type PythonMeasurement, type PythonShape } from '../engine/pythonEngine';
import { workflowMismatch } from '../engine/workflowFit';

interface Props {
  specimen: Specimen;
  recipeId: string;
  pythonUrl: string;
  connected: boolean;
  lang: 'zh' | 'en';
  /** The browser preview's summary (key, label, value, unit): the quantity both engines measure
   * (``SAME_QUANTITY``) is compared, nothing else. */
  previewSummary?: Array<{ key: string; label: string; value: number | string | null; unit: string }> | null;
  /** Python's outlines in the canvas's coordinates, keyed by specimen, image and workflow (null: none). */
  onShapes?: (overlay: { key: string; shapes: PythonShape[] } | null) => void;
  overlayVisible?: boolean;
  onToggleOverlay?: (visible: boolean) => void;
}

function fmt(v: number | string | null, unit: string): string {
  if (v === null || v === undefined) return '–';
  if (typeof v === 'string') return v;
  if (Number.isInteger(v)) return unit ? `${v} ${unit}` : String(v);
  const s = Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(3);
  return unit ? `${s} ${unit}` : s;
}

/** The validated Python engine's result for the current image and workflow, beside the browser
 * preview: the numbers that count, with the engine and its version named. */
/** Key of a Python overlay: the outlines belong to this image (as cropped) and this workflow. */
export function overlayKey(specimen: Specimen, recipeId: string): string {
  return `${specimen.id}:${specimen.sha256}:${recipeId}`;
}

export function PythonEngineCard({ specimen, recipeId, pythonUrl, connected, lang, previewSummary, onShapes, overlayVisible = true, onToggleOverlay }: Props) {
  const recipe = pythonRecipeFor(recipeId);
  const [state, setState] = useState<{ status: 'idle' | 'running' | 'done' | 'error'; result?: PythonMeasureResult; m?: PythonMeasurement; error?: string; key?: string }>({
    status: 'idle',
  });
  const zh = lang === 'zh';
  const key = overlayKey(specimen, recipeId);
  const stale = state.key !== key;

  const run = async () => {
    if (!recipe) return;
    setState({ status: 'running', key });
    try {
      const m = await measureWithPython(pythonUrl, specimen, recipe, lang, pythonParamsFor(recipeId));
      setState({ status: 'done', result: m.result, m, key });
      onShapes?.({ key, shapes: shapesForCanvas(specimen, m) });
    } catch (err) {
      setState({ status: 'error', error: err instanceof Error ? err.message : String(err), key });
      onShapes?.(null);
    }
  };

  const box = 'rounded-lg border p-3 text-[12px] space-y-2';
  const looksLike = workflowMismatch(specimen, recipeId, lang);
  const mismatch = looksLike ? (
    <div className={`${box} border-amber-500/50 bg-amber-500/10 text-amber-800 dark:text-amber-200`}>
      {zh
        ? `这张图看起来是“${looksLike}”，和所选工作流不符：下面的数字多半没有意义。请换成合适的工作流，或确认图像类型后再用。`
        : `This image looks like "${looksLike}", which does not fit the chosen workflow: the numbers below are probably meaningless. Choose a fitting workflow, or confirm the image type first.`}
    </div>
  ) : null;
  if (!recipe) {
    return (
      <>
        {mismatch}
        <div className={`${box} border-slate-200 dark:border-slate-800 text-slate-500`}>
          {zh
            ? '这个工作流还没有经过验收的 Python 计量：下方数字仅为浏览器预览。'
            : 'This workflow has no validated Python measurement yet: the numbers below are a browser preview.'}
        </div>
      </>
    );
  }
  const result = !stale ? state.result : undefined;
  const pair = recipe ? SAME_QUANTITY[recipe] : undefined;
  const py = result && pair ? pair.python(result.summary, result.unit) : null;
  const pyValue = py?.value ?? null;
  const pyUnit = py?.unit ?? '';
  const previewValue = (() => {
    const m = pair && previewSummary?.find((x) => x.key === pair.web);
    return m && typeof m.value === 'number' ? { label: m.label, value: m.value, unit: m.unit } : null;
  })();
  const pyCount = result && recipe ? pythonCount(recipe, result.summary) : null;
  const webCount = (() => {
    const key = recipe ? SAME_COUNT[recipe]?.web : undefined;
    const m = key && previewSummary?.find((x) => x.key === key);
    return m && typeof m.value === 'number' ? m.value : null;
  })();
  const countDiffers = pyCount !== null && webCount !== null && pyCount !== webCount;
  const diff =
    pyValue && previewValue && previewValue.unit === pyUnit && pyUnit !== 'px' ? (previewValue.value / pyValue - 1) * 100 : null;
  return (
    <>
    {mismatch}
    <div className={`${box} border-emerald-500/40 bg-emerald-500/5`}>
      <div className="flex items-center gap-2">
        <ShieldCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
        <span className="font-semibold">{zh ? 'Python 引擎（已验证）' : 'Python engine (validated)'}</span>
        <span className="ml-auto text-[11px] text-slate-500">{result ? `${result.engine} ${result.version}` : recipe}</span>
      </div>
      {!connected && (
        <div className="text-slate-500">
          {zh
            ? '未连接：在终端运行 uv run vexel serve，然后在设置里“测试连接”。下方数字仅为浏览器预览。'
            : 'Not connected: run `uv run vexel serve`, then "Test" in settings. The numbers below are a browser preview.'}
        </div>
      )}
      {connected && (
        <button
          type="button"
          onClick={run}
          disabled={state.status === 'running' && !stale}
          className="h-7 px-3 rounded bg-emerald-600 hover:bg-emerald-700 text-white text-[12px] font-medium cursor-pointer inline-flex items-center gap-1.5"
        >
          {state.status === 'running' && !stale ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
          {zh ? '用 Python 引擎测量' : 'Measure with the Python engine'}
        </button>
      )}
      {!stale && state.status === 'error' && (
        <div className="text-amber-600 dark:text-amber-400 flex items-start gap-1">
          <XCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          <span>{state.error}</span>
        </div>
      )}
      {result && (
        <>
          <div className="text-slate-500">
            {result.recipe_label} · {result.calibration.text}
          </div>
          {state.m && (
            <div className="text-slate-500">
              {state.m.input === 'file'
                ? zh
                  ? `测量的是原始文件 ${specimen.name}（${result.image.width}×${result.image.height} px，全分辨率与原始位深）。`
                  : `Measured the original file ${specimen.name} (${result.image.width}×${result.image.height} px, full resolution and bit depth).`
                : zh
                  ? `测量的是工作台显示的像素（${result.image.width}×${result.image.height} px）。`
                  : `Measured the pixels the workbench shows (${result.image.width}×${result.image.height} px).`}
            </div>
          )}
          {state.m?.fileRefused && (
            <div className="text-amber-600 dark:text-amber-400">
              {zh
                ? `Python 读不了原始文件（${state.m.fileRefused}），改为测量工作台显示的像素：分辨率和位深可能低于原图。`
                : `Python could not read the original file (${state.m.fileRefused}), so it measured the pixels the workbench shows: resolution and bit depth may be lower than the original.`}
            </div>
          )}
          {result.shapes.length > 0 && onToggleOverlay && (
            <label className="flex items-center gap-1.5 text-slate-600 dark:text-slate-300 cursor-pointer select-none">
              <input type="checkbox" checked={overlayVisible} onChange={(e) => onToggleOverlay(e.target.checked)} />
              {zh ? `在图上显示 Python 的测量轮廓（${result.shapes.length} 个，虚线）` : `Show Python's outlines on the image (${result.shapes.length}, dashed)`}
            </label>
          )}
          {result.error ? (
            <div className="text-amber-600 dark:text-amber-400">{result.error}</div>
          ) : (
            <div className="grid grid-cols-2 gap-x-3 gap-y-1">
              {result.summary
                .filter((s) => s.value !== null)
                .slice(0, 12)
                .map((s) => (
                  <div key={s.key} className="flex justify-between gap-2">
                    <span className="text-slate-500 truncate">{s.label}</span>
                    <span className="font-mono tabular-nums">
                      {s.kind === 'ratio' && typeof s.value === 'number' ? fmt(s.value * 100, '%') : fmt(s.value, s.unit)}
                    </span>
                  </div>
                ))}
            </div>
          )}
          {diff !== null && previewValue && (
            <div className={Math.abs(diff) > 2 ? 'text-amber-600 dark:text-amber-400' : 'text-slate-500'}>
              {zh
                ? `浏览器预览的${previewValue.label}与 Python 相差 ${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%，以 Python 为准。`
                : `The browser preview's ${previewValue.label} differs from Python by ${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%; Python's counts.`}
            </div>
          )}
          {countDiffers && (
            <div className="text-amber-600 dark:text-amber-400">
              {zh
                ? `浏览器预览找到 ${webCount} 个，Python 找到 ${pyCount} 个：两边测的不是同一批对象，上面的差异不能说明一致与否。`
                : `The browser preview found ${webCount}, Python found ${pyCount}: the two did not measure the same objects, so the difference above says little.`}
            </div>
          )}
          {result.calibration.source === 'none' && specimen.calibration.spaceUnit !== 'px' && (
            <div className="text-amber-600 dark:text-amber-400">
              {zh
                ? `工作台显示的 ${specimen.calibration.pixelSize} ${specimen.calibration.spaceUnit}/px 是默认值，不是标定，所以 Python 按像素给出长度。请用“手动描尺”在比例尺上标定，或导入带仪器元数据的原始 TIFF。`
                : `The workbench's ${specimen.calibration.pixelSize} ${specimen.calibration.spaceUnit}/px is a default, not a calibration, so Python gives lengths in pixels. Calibrate on the scale bar ("draw scale"), or import the original TIFF with the instrument's metadata.`}
            </div>
          )}
          {[...(result.calibration.warnings ?? []), ...result.warnings].map((w, i) => (
            <div key={i} className="text-amber-600 dark:text-amber-400">
              • {w}
            </div>
          ))}
        </>
      )}
    </div>
    </>
  );
}
