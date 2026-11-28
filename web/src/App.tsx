import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CheckCircle2,
  Globe,
  Microscope,
  Moon,
  Server,
  Sun,
  XCircle,
} from 'lucide-react';
import { useClickOutside } from './hooks/useClickOutside';
import type {
  AnnotationStroke,
  AppView,
  BatchRow,
  Calibration,
  HumanCorrection,
  LLMProviderConfig,
  LLMProviderInfo,
  Lang,
  LowConfidenceSegment,
  ManualMeasurement,
  ModelCard,
  ModelOcclusionMode,
  OccludedDisplayMode,
  PlanProposal,
  PromptPoint,
  RefinedBoundaryProposal,
  RoiRect,
  RunResult,
  RunVersionSnapshot,
  Specimen,
  ToolMode,
  UnifiedRecipe,
  VolumeAnalysisResult,
} from './types/vexel';
import { ACCEPTED_MICROSCOPY_EXTENSIONS } from './engine/microscopyDecoder';
import { generateSyntheticKidney3DStack } from './engine/volume3d';
import { runVolumeAnalysis3DAsync } from './engine/volumeAnalysisRunner';
import {
  applyInfoBarCropToSpecimen,
  applyVisionInspectionToSpecimen,
  decodeUploadedImage,
  getBuiltinSpecimens,
  loadKidneySectionSpecimen,
} from './engine/samples';
import { BUILTIN_MODELS, type TrainedLocalModel } from './engine/mlRuntime';
import { overlayKey, PythonEngineCard } from './components/PythonEngineCard';
import type { PythonShape } from './engine/pythonEngine';
import {
  deleteLocalModel,
  deleteOnnxModel,
  loadAllLocalModels,
  loadAllOnnxModels,
  saveLocalModel,
  saveOnnxModel,
  type PersistedOnnxModel,
} from './engine/modelStore';
import { disposeOnnxSession, predictOnnxModelTiled } from './engine/onnxRuntime';
import {
  BUILTIN_RECIPES,
  OPERATOR_CATALOG,
  RAW_SOURCE_STEP_ID,
  UNRECOGNIZED_MANUAL_RECIPE,
  UNRECOGNIZED_MANUAL_RECIPE_ID,
  buildAutoPlan,
  executeRecipe,
  probeSpecimenFingerprint,
} from './engine/operators';
import {
  exportAnnotatedSvg,
  exportHtmlReport,
  exportObjectsCsv,
  exportReproducePy,
  exportSingleExcel,
  exportVexflow,
} from './engine/exporters';
import { t } from './i18n/strings';
import { TopBar } from './components/TopBar';
import { AgentCommandBar } from './components/AgentCommandBar';
import { ScientificCanvas } from './components/ScientificCanvas';
import {
  InspectorPanel,
  type CorrectionSuggestion,
  type ResultsSubTab,
  type WorkbenchStage,
} from './components/InspectorPanel';
import { WorkflowStudio } from './components/WorkflowStudio';
import { ModelHub } from './components/ModelHub';
import { BatchCenter } from './components/BatchCenter';
import { Studio3DView } from './components/Studio3DView';

function specimenToDownsampledDataUrl(specimen: Specimen): string {
  const maxDim = 512;
  const scale = Math.min(1, maxDim / Math.max(specimen.width, specimen.height));
  const w = Math.max(16, Math.round(specimen.width * scale));
  const h = Math.max(16, Math.round(specimen.height * scale));
  const src = document.createElement('canvas');
  src.width = specimen.width;
  src.height = specimen.height;
  const sctx = src.getContext('2d')!;
  sctx.putImageData(new ImageData(new Uint8ClampedArray(specimen.rgba), specimen.width, specimen.height), 0, 0);

  const dst = document.createElement('canvas');
  dst.width = w;
  dst.height = h;
  const dctx = dst.getContext('2d')!;
  dctx.drawImage(src, 0, 0, w, h);
  return dst.toDataURL('image/jpeg', 0.82);
}

export default function App() {
  const initialSpecimens = useMemo(() => getBuiltinSpecimens(), []);

  const [lang, setLang] = useState<Lang>('zh');
  const [darkMode, setDarkMode] = useState(true);
  const [view, setView] = useState<AppView>('workbench');
  const [workbenchStage, setWorkbenchStage] = useState<WorkbenchStage>('results');
  const [resultsSubTab, setResultsSubTab] = useState<ResultsSubTab>('metrics');

  // Multi-LLM Provider selection (server-side secret status; zero browser API key inputs)
  const [providers, setProviders] = useState<LLMProviderInfo[]>([]);
  const [providerConfig, setProviderConfig] = useState<LLMProviderConfig>(() => {
    try {
      const raw = localStorage.getItem('vexel_llm_provider_v1');
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed?.provider && parsed?.model) return parsed;
      }
    } catch {
      // ignore storage errors
    }
    return {
      provider: 'gemini',
      model: 'gemini-4-hydrogen',
      ollamaBaseUrl: 'http://127.0.0.1:11434',
    };
  });

  // Engine / Compliance settings
  const [enginePopoverOpen, setEnginePopoverOpen] = useState(false);
  const engineTriggerRef = useRef<HTMLButtonElement>(null);
  const enginePopoverRef = useRef<HTMLDivElement>(null);
  useClickOutside(enginePopoverOpen, [engineTriggerRef, enginePopoverRef], () =>
    setEnginePopoverOpen(false)
  );
  const [engineMode, setEngineMode] = useState<'web' | 'python'>('web');
  const [pythonUrl, setPythonUrl] = useState('http://127.0.0.1:8765');
  // Python's outlines for one image and workflow, in canvas coordinates, and whether they are shown
  const [pythonOverlay, setPythonOverlay] = useState<{ key: string; shapes: PythonShape[] } | null>(null);
  const [showPythonOverlay, setShowPythonOverlay] = useState(true);
  const [pythonCheckState, setPythonCheckState] = useState<{
    status: 'idle' | 'checking' | 'ok' | 'error';
    message?: string;
  }>({ status: 'idle' });
  const [offlinePolicy, setOfflinePolicy] = useState(false);

  // Specimens & active state
  const [specimens, setSpecimens] = useState<Specimen[]>(initialSpecimens);
  const [activeSpecimenId, setActiveSpecimenId] = useState<string>(initialSpecimens[0].id);

  // Trained & imported models (built-in + IndexedDB-persisted Few-Shot DL & offline ONNX binaries)
  const [models, setModels] = useState<ModelCard[]>(BUILTIN_MODELS);
  const [localModels, setLocalModels] = useState<Map<string, TrainedLocalModel>>(() => new Map());
  const [onnxModels, setOnnxModels] = useState<Map<string, PersistedOnnxModel>>(() => new Map());
  const onnxModelsRef = useRef<Map<string, PersistedOnnxModel>>(new Map());
  const onnxProbCacheRef = useRef<Map<string, Float32Array>>(new Map());
  const onnxRunSeqRef = useRef<number>(0);

  // Workflow, Result-First plan, custom saved workflows, and execution results
  const [activeRecipe, setActiveRecipe] = useState<UnifiedRecipe>(() =>
    JSON.parse(JSON.stringify(BUILTIN_RECIPES[0]))
  );
  const [savedWorkflows, setSavedWorkflows] = useState<UnifiedRecipe[]>(() => {
    try {
      const raw = localStorage.getItem('vexel_saved_workflows_v1');
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });
  const [importInspecting, setImportInspecting] = useState(false);
  const [autoRunOnImport, setAutoRunOnImport] = useState(false);
  const [planProposal, setPlanProposal] = useState<PlanProposal | null>(null);
  const [planAccepted, setPlanAccepted] = useState(false);
  const [runResult, setRunResult] = useState<RunResult | null>(null);
  const [resultsBySpecimen, setResultsBySpecimen] = useState<Record<string, RunResult | null>>({});
  const [recipesBySpecimen, setRecipesBySpecimen] = useState<Record<string, UnifiedRecipe>>({});
  const [pendingCropDraftBySpecimen, setPendingCropDraftBySpecimen] = useState<
    Record<string, { cropPx: number; barPosition: 'bottom' | 'top' | 'none' }>
  >({});
  const [manualCalLineBySpecimen, setManualCalLineBySpecimen] = useState<
    Record<string, { p1: [number, number]; p2: [number, number]; lengthPx: number; snapped?: boolean } | null>
  >({});
  const [aiLoading, setAiLoading] = useState(false);

  // Canvas interaction state
  const [selectedStepId, setSelectedStepId] = useState<string | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<number | null>(null);
  const [toolMode, setToolMode] = useState<ToolMode>('pan');
  const [brushClassId, setBrushClassId] = useState(1);
  const [brushRadius, setBrushRadius] = useState(9);

  // Per-specimen annotations, prompts, measurements, ROI, and human review corrections
  const [strokesBySpecimen, setStrokesBySpecimen] = useState<Record<string, AnnotationStroke[]>>({});
  const [promptsBySpecimen, setPromptsBySpecimen] = useState<Record<string, PromptPoint[]>>({});
  const [measurementsBySpecimen, setMeasurementsBySpecimen] = useState<Record<string, ManualMeasurement[]>>({});
  const [roiBySpecimen, setRoiBySpecimen] = useState<Record<string, RoiRect | null>>({});
  const roiBySpecimenRef = useRef<Record<string, RoiRect | null>>({});
  const [corrections, setCorrections] = useState<HumanCorrection[]>([]);
  const [showLowConfHighlight, setShowLowConfHighlight] = useState(false);
  const [lowConfThreshold, setLowConfThreshold] = useState(0.65);
  const [occludedDisplayMode, setOccludedDisplayMode] = useState<OccludedDisplayMode>('highlight');
  const [dismissedLowConfBySpecimen, setDismissedLowConfBySpecimen] = useState<Record<string, string[]>>({});
  const [activeCorrectionSegId, setActiveCorrectionSegId] = useState<string | null>(null);
  const [runHistoryBySpecimen, setRunHistoryBySpecimen] = useState<
    Record<string, RunVersionSnapshot[]>
  >({});

  // Batch processing state
  const [batchRows, setBatchRows] = useState<BatchRow[]>([]);
  const [batchRunning, setBatchRunning] = useState(false);

  // Shared 3D volumetric analysis summary (synced between 3D Lab & Workbench Stage 4 Results)
  const [sharedVolumeAnalysis, setSharedVolumeAnalysis] = useState<VolumeAnalysisResult | null>(null);
  const [analyzing3DQuick, setAnalyzing3DQuick] = useState(false);

  // Toast notice (iframe-safe replacement for window.alert)
  const [toast, setToast] = useState<string | null>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const canvasSnapshotRef = useRef<() => string | undefined>(() => undefined);
  const snapshotDebounceRef = useRef<number | null>(null);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast((cur) => (cur === msg ? null : cur)), 3200);
  }, []);

  const handleRunQuick3DAnalysis = useCallback(async () => {
    setAnalyzing3DQuick(true);
    try {
      const vol = generateSyntheticKidney3DStack();
      const res = await runVolumeAnalysis3DAsync(vol, {
        cellThreshold: 0,
        nucleusDarkRatio: 0.65,
        minCellVoxels: 45,
      });
      setSharedVolumeAnalysis(res);
      showToast(
        lang === 'zh'
          ? `3D 体分析完成：检出 ${res.cellCount} 个 3D 细胞，平均核质比 ${res.meanNcVolumePct.toFixed(1)}%`
          : `3D Analysis done: ${res.cellCount} cells, mean N/C ${res.meanNcVolumePct.toFixed(1)}%`
      );
    } catch (err) {
      showToast(err instanceof Error ? err.message : String(err));
    } finally {
      setAnalyzing3DQuick(false);
    }
  }, [lang, showToast]);

  const activeSpecimen = useMemo(
    () => specimens.find((s) => s.id === activeSpecimenId) ?? specimens[0],
    [specimens, activeSpecimenId]
  );

  // Sync dark mode class on <html>
  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) root.classList.add('dark');
    else root.classList.remove('dark');
  }, [darkMode]);

  // Persist user-saved custom workflows
  useEffect(() => {
    try {
      localStorage.setItem('vexel_saved_workflows_v1', JSON.stringify(savedWorkflows));
    } catch {
      // Ignore quota errors
    }
  }, [savedWorkflows]);

  // Persist chosen LLM provider & model (no secret stored)
  useEffect(() => {
    try {
      localStorage.setItem('vexel_llm_provider_v1', JSON.stringify(providerConfig));
    } catch {
      // Ignore quota errors
    }
  }, [providerConfig]);

  // Fetch server-side LLM provider readiness
  useEffect(() => {
    let active = true;
    fetch('/api/v1/ai/providers')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (active && Array.isArray(data?.providers)) {
          setProviders(data.providers);
        }
      })
      .catch(() => {
        // Fallback catalog inside AgentCommandBar handles offline state
      });
    return () => {
      active = false;
    };
  }, []);

  // Restore IndexedDB-persisted locally trained DL models & offline ONNX binaries on startup
  useEffect(() => {
    let active = true;
    Promise.all([loadAllLocalModels(), loadAllOnnxModels()])
      .then(([persistedLocals, persistedOnnx]) => {
        if (!active) return;
        if (persistedLocals.length > 0) {
          setLocalModels((prev) => {
            const next = new Map(prev);
            for (const m of persistedLocals) next.set(m.card.ref, m);
            return next;
          });
        }
        if (persistedOnnx.length > 0) {
          setOnnxModels((prev) => {
            const next = new Map(prev);
            for (const o of persistedOnnx) next.set(o.ref, o);
            onnxModelsRef.current = next;
            return next;
          });
        }
        if (persistedLocals.length > 0 || persistedOnnx.length > 0) {
          const restoredCards = [
            ...persistedOnnx.map((o) => o.card),
            ...persistedLocals.map((m) => m.card),
          ];
          setModels((prev) => {
            const restoredRefs = new Set(restoredCards.map((c) => c.ref));
            return [...restoredCards, ...prev.filter((m) => !restoredRefs.has(m.ref))];
          });
        }
      })
      .catch(() => {
        // IndexedDB optional fallback
      });
    return () => {
      active = false;
    };
  }, []);

  const recordRunSnapshot = useCallback(
    (specimenId: string, res: RunResult, label: string) => {
      setRunHistoryBySpecimen((prev) => {
        const list = prev[specimenId] ?? [];
        const last = list[list.length - 1];
        if (last && last.result.resultDigest === res.resultDigest) {
          return prev;
        }
        const nextVer = (last?.version ?? 0) + 1;
        const entry: RunVersionSnapshot = {
          id: `run_${Date.now()}_${nextVer}_${Math.random().toString(36).slice(2, 5)}`,
          version: nextVer,
          label,
          timestamp: res.executedAt,
          result: res,
        };
        return {
          ...prev,
          [specimenId]: [...list.slice(-11), entry],
        };
      });
    },
    []
  );

  // Execute active recipe deterministically for a specimen (immediate compute; triggers real async onnxruntime-web tiled inference when an imported ONNX binary model is active)
  const runWorkflow = useCallback(
    (
      spec: Specimen,
      rec: UnifiedRecipe,
      corrs: HumanCorrection[],
      lModels: Map<string, TrainedLocalModel>,
      versionLabel?: string,
      debounceSnapshot = false,
      roiOverride?: RoiRect | null
    ) => {
      const effectiveRoi =
        roiOverride !== undefined ? roiOverride : (roiBySpecimenRef.current[spec.id] ?? null);
      const roiKey = effectiveRoi
        ? `${Math.round(effectiveRoi.x)}_${Math.round(effectiveRoi.y)}_${Math.round(effectiveRoi.width)}_${Math.round(effectiveRoi.height)}`
        : 'full';

      // Attach any already-cached real ONNX probability maps for enabled model.segment steps
      const precomputedProbMaps = new Map<string, Float32Array>();
      const pendingOnnxModels: PersistedOnnxModel[] = [];
      for (const step of rec.steps) {
        if (!step.enabled || step.op !== 'model.segment' || !step.modelRef) continue;
        const onnxRec = onnxModelsRef.current.get(step.modelRef);
        if (!onnxRec) continue;
        const cacheKey = `${onnxRec.ref}:${spec.sha256}:${roiKey}`;
        const cached = onnxProbCacheRef.current.get(cacheKey);
        if (cached) {
          precomputedProbMaps.set(onnxRec.ref, cached);
        } else {
          pendingOnnxModels.push(onnxRec);
        }
      }

      const res = executeRecipe(spec, rec, {
        corrections: corrs,
        localModels: lModels,
        precomputedProbMaps: precomputedProbMaps.size > 0 ? precomputedProbMaps : undefined,
        roi: effectiveRoi,
      });
      setRunResult(res);
      setResultsBySpecimen((prev) => ({ ...prev, [spec.id]: res }));
      setRecipesBySpecimen((prev) => ({ ...prev, [spec.id]: rec }));

      // Kick off real offline onnxruntime-web tiled inference for uncached imported ONNX binaries
      if (pendingOnnxModels.length > 0) {
        const seq = ++onnxRunSeqRef.current;
        const fullCropRows = spec.calibration.cropRows ?? spec.height;
        const rx = effectiveRoi ? Math.max(0, Math.round(effectiveRoi.x)) : 0;
        const ry = effectiveRoi ? Math.max(0, Math.round(effectiveRoi.y)) : 0;
        const rw = effectiveRoi
          ? Math.max(1, Math.min(spec.width - rx, Math.round(effectiveRoi.width)))
          : spec.width;
        const rh = effectiveRoi
          ? Math.max(1, Math.min(fullCropRows - ry, Math.round(effectiveRoi.height)))
          : spec.height;
        const baseGray = res.preModelGray ?? spec.gray;
        let workGray: Float32Array;
        if (effectiveRoi) {
          workGray = new Float32Array(rw * rh);
          for (let y = 0; y < rh; y++) {
            const srcOff = (ry + y) * spec.width + rx;
            const dstOff = y * rw;
            for (let x = 0; x < rw; x++) workGray[dstOff + x] = baseGray[srcOff + x];
          }
        } else {
          workGray = baseGray;
        }
        const workCropRows = effectiveRoi ? rh : fullCropRows;

        Promise.all(
          pendingOnnxModels.map(async (onnxRec) => {
            const prob = await predictOnnxModelTiled(onnxRec, workGray, rw, rh, {
              cropRows: workCropRows,
              tileSize: onnxRec.card.tileSize ?? 512,
              overlap: 32,
            });
            const cacheKey = `${onnxRec.ref}:${spec.sha256}:${roiKey}`;
            onnxProbCacheRef.current.set(cacheKey, prob);
            return [onnxRec.ref, prob] as const;
          })
        )
          .then((entries) => {
            if (seq !== onnxRunSeqRef.current) return;
            const freshMap = new Map(precomputedProbMaps);
            for (const [ref, prob] of entries) freshMap.set(ref, prob);
            const updatedRes = executeRecipe(spec, rec, {
              corrections: corrs,
              localModels: lModels,
              precomputedProbMaps: freshMap,
              roi: effectiveRoi,
            });
            setRunResult(updatedRes);
            setResultsBySpecimen((prev) => ({ ...prev, [spec.id]: updatedRes }));
            recordRunSnapshot(spec.id, updatedRes, `${entries[0][0]} (ONNX WASM)`);
          })
          .catch((err) => {
            showToast(
              `ONNX inference error: ${err instanceof Error ? err.message : String(err)}`
            );
          });
      }

      const autoLabel =
        versionLabel ??
        (res.provenance.modelRefs.length > 0
          ? res.provenance.modelRefs[0].split('@')[0]
          : res.provenance.correctionsApplied > 0
          ? `${rec.id} (+${res.provenance.correctionsApplied} edits)`
          : rec.id);
      if (debounceSnapshot) {
        if (snapshotDebounceRef.current !== null) {
          window.clearTimeout(snapshotDebounceRef.current);
        }
        snapshotDebounceRef.current = window.setTimeout(() => {
          recordRunSnapshot(spec.id, res, autoLabel);
          snapshotDebounceRef.current = null;
        }, 500);
      } else {
        if (snapshotDebounceRef.current !== null) {
          window.clearTimeout(snapshotDebounceRef.current);
          snapshotDebounceRef.current = null;
        }
        recordRunSnapshot(spec.id, res, autoLabel);
      }
      return res;
    },
    [recordRunSnapshot, showToast]
  );

  // Full-auto recognition & execution (used for built-in demo specimens or opt-in auto-run)
  const selectSpecimenAndPlan = useCallback(
    (spec: Specimen, lModels: Map<string, TrainedLocalModel>, corrs: HumanCorrection[]) => {
      const proposal = buildAutoPlan(spec, '', lModels);
      const clonedRec: UnifiedRecipe = JSON.parse(JSON.stringify(proposal.suggestedRecipe));
      setPlanProposal(proposal);
      setPlanAccepted(false);
      setActiveRecipe(clonedRec);
      setSelectedStepId(null);
      setSelectedObjectId(null);
      setActiveCorrectionSegId(null);
      const specRoi = roiBySpecimenRef.current[spec.id] ?? null;
      const res = executeRecipe(spec, clonedRec, {
        corrections: corrs,
        localModels: lModels,
        roi: specRoi,
      });
      setRunResult(res);
      setResultsBySpecimen((prev) => ({ ...prev, [spec.id]: res }));
      setRecipesBySpecimen((prev) => ({ ...prev, [spec.id]: clonedRec }));
      recordRunSnapshot(spec.id, res, `Auto · ${clonedRec.id}`);
    },
    [recordRunSnapshot]
  );

  // Semi-auto non-executing staging for newly imported images:
  // Proposes info-bar crop, calibration, and workflow suggestion WITHOUT running analysis until user confirms.
  const stageSpecimenForSemiAuto = useCallback(
    (spec: Specimen, lModels: Map<string, TrainedLocalModel>) => {
      const proposal = buildAutoPlan(spec, '', lModels, undefined, { runTrial: false });
      const clonedRec: UnifiedRecipe = JSON.parse(JSON.stringify(proposal.suggestedRecipe));
      setPlanProposal(proposal);
      setPlanAccepted(false);
      setActiveRecipe(clonedRec);
      setSelectedStepId(null);
      setSelectedObjectId(null);
      setActiveCorrectionSegId(null);
      setRunResult(null);
      setResultsBySpecimen((prev) => ({ ...prev, [spec.id]: null }));
      setRecipesBySpecimen((prev) => ({ ...prev, [spec.id]: clonedRec }));
      const rep = spec.importReport;
      const proposedPx = rep?.proposedCropBarPx ?? 0;
      const pos: 'bottom' | 'top' | 'none' =
        proposedPx > 0 && rep?.barPosition && rep.barPosition !== 'none'
          ? rep.barPosition
          : 'bottom';
      setPendingCropDraftBySpecimen((prev) => ({
        ...prev,
        [spec.id]: { cropPx: proposedPx, barPosition: pos },
      }));
    },
    []
  );

  // Manual ROI update — recomputes immediately ONLY if workflow has already executed; otherwise stages ROI cleanly
  const handleChangeRoi = useCallback(
    (nextRoi: RoiRect | null) => {
      const nextMap = { ...roiBySpecimenRef.current, [activeSpecimen.id]: nextRoi };
      roiBySpecimenRef.current = nextMap;
      setRoiBySpecimen(nextMap);
      setSelectedObjectId(null);
      if (runResult) {
        runWorkflow(
          activeSpecimen,
          activeRecipe,
          corrections,
          localModels,
          nextRoi ? `ROI ${Math.round(nextRoi.width)}×${Math.round(nextRoi.height)}` : `${activeRecipe.id} (Full FOV)`,
          false,
          nextRoi
        );
      }
      showToast(
        nextRoi
          ? runResult
            ? lang === 'zh'
              ? `已设置 ROI (${Math.round(nextRoi.width)}×${Math.round(nextRoi.height)} px)，流程与模型训练将仅在 ROI 内执行`
              : `ROI set (${Math.round(nextRoi.width)}×${Math.round(nextRoi.height)} px) — pipeline & training scoped to ROI`
            : lang === 'zh'
            ? `已框选 ROI (${Math.round(nextRoi.width)}×${Math.round(nextRoi.height)} px)，点击「执行分析」后将仅在 ROI 内计算`
            : `ROI set (${Math.round(nextRoi.width)}×${Math.round(nextRoi.height)} px) — will scope computation when workflow runs`
          : lang === 'zh'
          ? '已清除 ROI，恢复全视场范围'
          : 'ROI cleared — restored full FOV scope'
      );
    },
    [activeSpecimen, activeRecipe, corrections, localModels, runResult, runWorkflow, showToast, lang]
  );

  // Initial mount Result-First execution: load uploaded kidney section specimen when available, fallback to demo specimen
  useEffect(() => {
    let cancelled = false;
    loadKidneySectionSpecimen()
      .then((kidneySpec) => {
        if (cancelled) return;
        setSpecimens((prev) => [kidneySpec, ...prev.filter((s) => s.id !== kidneySpec.id)]);
        setActiveSpecimenId(kidneySpec.id);
        selectSpecimenAndPlan(kidneySpec, localModels, corrections);
      })
      .catch(() => {
        if (cancelled) return;
        selectSpecimenAndPlan(initialSpecimens[0], localModels, corrections);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSwitchSpecimen = (id: string) => {
    const found = specimens.find((s) => s.id === id);
    if (!found) return;
    setActiveSpecimenId(found.id);
    setSelectedStepId(null);
    setSelectedObjectId(null);
    setActiveCorrectionSegId(null);

    // Preserve prior execution / semi-auto staging state if already visited
    if (Object.prototype.hasOwnProperty.call(resultsBySpecimen, found.id)) {
      const cachedRes = resultsBySpecimen[found.id];
      const cachedRec = recipesBySpecimen[found.id] ?? activeRecipe;
      setActiveRecipe(cachedRec);
      setRunResult(cachedRes);
      const prop = buildAutoPlan(found, '', localModels, undefined, { runTrial: false });
      setPlanProposal({ ...prop, suggestedRecipe: cachedRec });
      setPlanAccepted(Boolean(cachedRes));
      if (!cachedRes) {
        setWorkbenchStage('import');
      }
      return;
    }

    selectSpecimenAndPlan(found, localModels, corrections);
  };

  const processUploadedFile = async (file: File) => {
    try {
      // Decode image; in default semi-auto mode (autoRunOnImport=false), do NOT auto-crop or auto-run analysis
      const { specimen: localSpec, originalPreviewBase64 } = await decodeUploadedImage(file, {
        autoCrop: autoRunOnImport,
      });
      setSpecimens((prev) => [localSpec, ...prev]);
      setActiveSpecimenId(localSpec.id);
      setView('workbench');

      const rep = localSpec.importReport;
      if (autoRunOnImport) {
        selectSpecimenAndPlan(localSpec, localModels, corrections);
        setWorkbenchStage('workflow');
        showToast(
          lang === 'zh'
            ? `全自动模式已导入「${localSpec.name}」并完成初步分析`
            : `Full-auto imported "${localSpec.name}" and executed analysis`
        );
      } else {
        stageSpecimenForSemiAuto(localSpec, localModels);
        setWorkbenchStage('import');
        const hasProposedCrop = (rep?.proposedCropBarPx ?? 0) > 0;
        showToast(
          lang === 'zh'
            ? `已导入「${localSpec.name}」（半自动模式）：请先完成物理标定（可参考或手动描测信息栏比例尺）${hasProposedCrop ? `，再确认裁切 ${rep?.proposedCropBarPx}px 信息栏` : ''}并选定工作流执行`
            : `Imported "${localSpec.name}" (Semi-Auto): calibrate scale bar first${hasProposedCrop ? `, then confirm ${rep?.proposedCropBarPx}px info-bar crop` : ''} & run workflow`
        );
      }

      // Optional background Vision + Scale Bar OCR refinement (updates reference suggestions ONLY; never auto-runs in semi-auto mode)
      if (!offlinePolicy) {
        setImportInspecting(true);
        try {
          const resp = await fetch('/api/v1/ai/inspect-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              fileName: localSpec.name,
              imageBase64: originalPreviewBase64,
              localHint: rep,
              lang,
              provider: providerConfig.provider,
              model: providerConfig.model,
            }),
          });
          if (resp.ok) {
            const data = await resp.json();
            if (data?.source === 'vision' && data?.inspection) {
              let updatedRef: Specimen | null = null;
              setSpecimens((prev) =>
                prev.map((s) => {
                  if (s.id !== localSpec.id) return s;
                  updatedRef = applyVisionInspectionToSpecimen(s, data.inspection);
                  return updatedRef;
                })
              );
              const refinedSpec = updatedRef ?? applyVisionInspectionToSpecimen(localSpec, data.inspection);
              setActiveSpecimenId((curId) => {
                if (curId === refinedSpec.id) {
                  if (autoRunOnImport) {
                    selectSpecimenAndPlan(refinedSpec, localModels, corrections);
                  } else {
                    const updatedProposal = buildAutoPlan(
                      refinedSpec,
                      '',
                      localModels,
                      undefined,
                      { runTrial: false }
                    );
                    setPlanProposal(updatedProposal);
                  }
                }
                return curId;
              });
              const rRep = refinedSpec.importReport;
              if (rRep?.recognized) {
                showToast(
                  lang === 'zh'
                    ? `AI 参考识别更新：${rRep.sceneLabelZh}${rRep.scaleBarValue && rRep.scaleBarUnit ? ` · 标尺参考 ${rRep.scaleBarValue} ${rRep.scaleBarUnit}` : ''}（待用户确认）`
                    : `AI reference hint: ${rRep.sceneLabelEn}${rRep.scaleBarValue && rRep.scaleBarUnit ? ` · Scale ~${rRep.scaleBarValue} ${rRep.scaleBarUnit}` : ''} (awaiting confirmation)`
                );
              }
            }
          }
        } catch {
          // Graceful silent fallback to local inspection
        } finally {
          setImportInspecting(false);
        }
      }
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Image decode failed');
    }
  };

  const handleUploadSpecimen = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await processUploadedFile(file);
    } finally {
      e.target.value = '';
    }
  };

  // Semi-auto Step 1: Adjust live crop outline on canvas before confirming
  const handleUpdatePendingCropDraft = (draft: {
    cropPx: number;
    barPosition: 'bottom' | 'top' | 'none';
  }) => {
    setPendingCropDraftBySpecimen((prev) => ({
      ...prev,
      [activeSpecimen.id]: draft,
    }));
  };

  // Semi-auto Step 1: Confirm info-bar crop OR skip crop (keeps rawRgba reversible)
  const handleConfirmInfoBarCrop = (
    cropPx: number,
    barPosition: 'bottom' | 'top' | 'none'
  ) => {
    const croppedSpec = applyInfoBarCropToSpecimen(activeSpecimen, cropPx, barPosition);
    const heightChanged = croppedSpec.height !== activeSpecimen.height;
    setSpecimens((prev) => prev.map((s) => (s.id === croppedSpec.id ? croppedSpec : s)));
    setPendingCropDraftBySpecimen((prev) => ({
      ...prev,
      [croppedSpec.id]: {
        cropPx: croppedSpec.importReport?.croppedBarPx ?? 0,
        barPosition: croppedSpec.importReport?.barPosition ?? 'bottom',
      },
    }));

    let effectiveRoi = roiBySpecimenRef.current[croppedSpec.id] ?? null;
    if (heightChanged && effectiveRoi) {
      effectiveRoi = null;
      const nextRoiMap = { ...roiBySpecimenRef.current, [croppedSpec.id]: null };
      roiBySpecimenRef.current = nextRoiMap;
      setRoiBySpecimen(nextRoiMap);
    }

    if (runResult) {
      runWorkflow(
        croppedSpec,
        activeRecipe,
        corrections,
        localModels,
        cropPx > 0 ? `Crop ${cropPx}px` : 'Uncropped',
        false,
        effectiveRoi
      );
    } else if (planProposal) {
      const nextProp = buildAutoPlan(croppedSpec, '', localModels, undefined, { runTrial: false });
      setPlanProposal({ ...nextProp, suggestedRecipe: activeRecipe });
    }

    showToast(
      cropPx > 0 && barPosition !== 'none'
        ? lang === 'zh'
          ? `已确认裁切${barPosition === 'top' ? '顶部' : '底部'} ${cropPx} px 信息栏（有效视场 ${croppedSpec.width}×${croppedSpec.height} px）`
          : `Confirmed ${barPosition} info-bar crop (${cropPx} px → ${croppedSpec.width}×${croppedSpec.height} px)`
        : lang === 'zh'
        ? '已跳过信息栏裁切，保留完整原始视场'
        : 'Skipped info-bar crop — keeping full original FOV'
    );
  };

  // Revert info-bar crop back to pending confirmation so user can re-adjust cut line on canvas
  const handleRevertInfoBarCrop = () => {
    const restored = applyInfoBarCropToSpecimen(activeSpecimen, 0, 'none');
    const prevProposed =
      activeSpecimen.rawCroppedBarPx ??
      activeSpecimen.importReport?.proposedCropBarPx ??
      0;
    const prevPos =
      activeSpecimen.rawBarPosition && activeSpecimen.rawBarPosition !== 'none'
        ? activeSpecimen.rawBarPosition
        : activeSpecimen.importReport?.barPosition && activeSpecimen.importReport.barPosition !== 'none'
        ? activeSpecimen.importReport.barPosition
        : 'bottom';
    const pendingSpec: Specimen = {
      ...restored,
      importReport: restored.importReport
        ? {
            ...restored.importReport,
            barPosition: prevPos,
            proposedCropBarPx: prevProposed,
            cropStatus: 'pending',
          }
        : undefined,
    };
    setSpecimens((prev) => prev.map((s) => (s.id === pendingSpec.id ? pendingSpec : s)));
    setPendingCropDraftBySpecimen((prev) => ({
      ...prev,
      [pendingSpec.id]: { cropPx: prevProposed, barPosition: prevPos },
    }));
    if (runResult) {
      runWorkflow(pendingSpec, activeRecipe, corrections, localModels, 'Revert crop');
    }
    showToast(
      lang === 'zh'
        ? '已恢复完整原图，可重新调整信息栏裁切区域'
        : 'Restored original image — adjust info-bar crop region'
    );
  };

  // Explicit user-confirmed workflow execution
  const handleRunActiveWorkflow = (recipeOverride?: UnifiedRecipe) => {
    const targetRec = recipeOverride ?? activeRecipe;
    setActiveRecipe(targetRec);
    setPlanAccepted(true);
    setSelectedStepId(null);
    setSelectedObjectId(null);
    runWorkflow(
      activeSpecimen,
      targetRec,
      corrections,
      localModels,
      targetRec.isCustom ? targetRec.name : targetRec.id
    );
    if (workbenchStage === 'import') {
      setWorkbenchStage('workflow');
    }
    showToast(
      lang === 'zh'
        ? `已开始执行工作流：${targetRec.isCustom ? targetRec.name : targetRec.nameZh}`
        : `Executed workflow: ${targetRec.isCustom ? targetRec.name : targetRec.nameEn}`
    );
  };

  // Switch workflow from Canvas top bar or Stage 1 selector:
  // If analysis has already started (runResult !== null), runs live immediately;
  // If in semi-auto pending staging (runResult === null), stages the chosen workflow cleanly without premature execution unless runNow=true.
  const handleSelectWorkflow = (workflowId: string, options?: { runNow?: boolean }) => {
    let cloned: UnifiedRecipe | null = null;
    if (workflowId === UNRECOGNIZED_MANUAL_RECIPE_ID) {
      cloned = JSON.parse(JSON.stringify(UNRECOGNIZED_MANUAL_RECIPE));
    } else {
      const found =
        savedWorkflows.find((w) => w.id === workflowId) ??
        BUILTIN_RECIPES.find((r) => r.id === workflowId);
      if (found) cloned = JSON.parse(JSON.stringify(found));
    }
    if (!cloned) return;

    setActiveRecipe(cloned);
    setRecipesBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: cloned! }));
    setSelectedStepId(null);
    setSelectedObjectId(null);

    const shouldExecute = options?.runNow === true || (options?.runNow !== false && Boolean(runResult));
    if (shouldExecute) {
      setPlanAccepted(true);
      runWorkflow(activeSpecimen, cloned, corrections, localModels, cloned.isCustom ? cloned.name : cloned.id);
      showToast(
        lang === 'zh'
          ? `已切换并执行工作流：${cloned.isCustom ? cloned.name : cloned.nameZh}`
          : `Switched & ran workflow: ${cloned.isCustom ? cloned.name : cloned.nameEn}`
      );
    } else {
      showToast(
        lang === 'zh'
          ? `已选定工作流「${cloned.isCustom ? cloned.name : cloned.nameZh}」，点击「执行分析」或调整参数后开始计算`
          : `Selected "${cloned.isCustom ? cloned.name : cloned.nameEn}" — click Run Workflow or tune parameters to execute`
      );
    }
  };

  // Save current workflow under a user-defined custom name
  const handleSaveCustomWorkflow = (customName: string, recipeOverride?: UnifiedRecipe) => {
    const src = recipeOverride ?? activeRecipe;
    const trimmed = customName.trim();
    if (!trimmed) return;
    const id = `custom_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`;
    const saved: UnifiedRecipe = {
      ...JSON.parse(JSON.stringify(src)),
      id,
      name: trimmed,
      nameZh: trimmed,
      nameEn: trimmed,
      isCustom: true,
      savedAt: new Date().toISOString(),
    };
    setSavedWorkflows((prev) => [saved, ...prev]);
    setActiveRecipe(saved);
    setRecipesBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: saved }));
    showToast(
      lang === 'zh'
        ? `已保存自定义工作流「${trimmed}」`
        : `Saved custom workflow "${trimmed}"`
    );
  };

  const handleDeleteCustomWorkflow = (workflowId: string) => {
    const target = savedWorkflows.find((w) => w.id === workflowId);
    setSavedWorkflows((prev) => prev.filter((w) => w.id !== workflowId));
    if (activeRecipe.id === workflowId) {
      const fallback: UnifiedRecipe = JSON.parse(
        JSON.stringify(BUILTIN_RECIPES[0])
      );
      setActiveRecipe(fallback);
      setRecipesBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: fallback }));
      if (runResult) {
        runWorkflow(activeSpecimen, fallback, corrections, localModels);
      }
    }
    if (target) {
      showToast(
        lang === 'zh'
          ? `已删除自定义工作流「${target.name}」`
          : `Deleted custom workflow "${target.name}"`
      );
    }
  };

  // Calibration updates — recomputes ONLY if workflow has already executed; never prematurely triggers analysis in semi-auto mode
  const updateActiveCalibration = (cal: Calibration) => {
    const updatedSpec: Specimen = { ...activeSpecimen, calibration: cal };
    setSpecimens((prev) => prev.map((s) => (s.id === updatedSpec.id ? updatedSpec : s)));
    if (runResult) {
      runWorkflow(updatedSpec, activeRecipe, corrections, localModels);
    }
    if (planProposal) {
      const updatedPlan = buildAutoPlan(updatedSpec, '', localModels, undefined, {
        runTrial: Boolean(runResult),
      });
      setPlanProposal({ ...updatedPlan, suggestedRecipe: activeRecipe });
    }
  };

  const handleConfirmCalibration = () => {
    updateActiveCalibration({ ...activeSpecimen.calibration, confirmed: true });
    showToast(lang === 'zh' ? '物理标定已确认并写入溯源记录' : 'Physical calibration confirmed');
  };

  // AI Natural-Language Goal Planning (calls server /api/v1/ai/plan with graceful offline fallback)
  const handleAiGoal = async (goal: string, sendImage: boolean) => {
    setAiLoading(true);
    try {
      if (offlinePolicy) {
        const proposal = buildAutoPlan(activeSpecimen, goal, localModels, {
          planner: 'rules',
          model: 'offline-deterministic-rules',
          fallbackReason:
            lang === 'zh'
              ? '离线合规模式已启用，已使用本地规则引擎完成规划。'
              : 'Offline compliance active; planned with local deterministic rules.',
        });
        setPlanProposal(proposal);
        setPlanAccepted(false);
        setActiveRecipe(JSON.parse(JSON.stringify(proposal.suggestedRecipe)));
        runWorkflow(activeSpecimen, proposal.suggestedRecipe, corrections, localModels);
        return;
      }

      const payload = {
        goal,
        modality: activeSpecimen.modality,
        fingerprint: probeSpecimenFingerprint(activeSpecimen),
        availableRecipes: BUILTIN_RECIPES.map((r) => ({
          id: r.id,
          modality: r.modality,
          name: r.nameEn,
        })),
        availableModels: models.filter((m) => m.fileStatus !== 'missing').map((m) => ({ ref: m.ref, task: m.task, modality: m.modality })),
        currentSteps: activeRecipe.steps.map((s) => ({
          op: s.op,
          enabled: s.enabled,
          params: s.params,
          modelRef: s.modelRef,
        })),
        sendImage,
        imageBase64: sendImage ? specimenToDownsampledDataUrl(activeSpecimen) : null,
        lang,
        provider: providerConfig.provider,
        model: providerConfig.model,
        ollamaBaseUrl: providerConfig.ollamaBaseUrl,
      };

      const resp = await fetch('/api/v1/ai/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (data?.domainBlocked && typeof data?.fallbackReason === 'string') {
        showToast(data.fallbackReason);
        return;
      }
      const proposal = buildAutoPlan(activeSpecimen, goal, localModels, data);
      setPlanProposal(proposal);
      setPlanAccepted(false);
      setActiveRecipe(JSON.parse(JSON.stringify(proposal.suggestedRecipe)));
      setSelectedStepId(null);
      runWorkflow(activeSpecimen, proposal.suggestedRecipe, corrections, localModels);
    } catch {
      const proposal = buildAutoPlan(activeSpecimen, goal, localModels, {
        planner: 'rules',
        model: 'offline-deterministic-rules',
        fallbackReason:
          lang === 'zh'
            ? '云端规划不可达，已自动回退至本地确定性规则引擎。'
            : 'Planner service unreachable; fell back to local rules engine.',
      });
      setPlanProposal(proposal);
      setPlanAccepted(false);
      setActiveRecipe(JSON.parse(JSON.stringify(proposal.suggestedRecipe)));
      runWorkflow(activeSpecimen, proposal.suggestedRecipe, corrections, localModels);
    } finally {
      setAiLoading(false);
    }
  };

  // Parameter & Step Mutations (immediate live recompute & live canvas feedback)
  const applyRecipeChange = (next: UnifiedRecipe, debounceSnapshot = false) => {
    setActiveRecipe(next);
    runWorkflow(activeSpecimen, next, corrections, localModels, undefined, debounceSnapshot);
  };

  const handleUpdateStepParam = (stepId: string, key: string, value: number | boolean | string) => {
    const targetStep = activeRecipe.steps.find((s) => s.id === stepId);
    const spec = OPERATOR_CATALOG.find((o) => o.op === targetStep?.op);
    if (key === 'occlusionHandling') {
      setOccludedDisplayMode(value === 'hide' ? 'hide' : 'highlight');
    }
    const next: UnifiedRecipe = {
      ...activeRecipe,
      steps: activeRecipe.steps.map((s) =>
        s.id === stepId ? { ...s, enabled: true, params: { ...s.params, [key]: value } } : s
      ),
    };
    applyRecipeChange(next, true);

    // Ensure the canvas immediately displays the visual effect of the parameter being adjusted
    if (spec?.category === 'preprocess' || spec?.category === 'segment' || spec?.category === 'model') {
      setSelectedStepId(stepId);
    } else {
      setSelectedStepId(null);
    }
  };

  // Sync model-level Occlusion Handling & Curvature/Overlap Boundary Fit config across Model Stage, Workflow & Canvas
  const handleUpdateModelOcclusionConfig = (patch: {
    occlusionMode?: ModelOcclusionMode;
    curvatureBoundaryFit?: boolean;
    overlapConfidenceThreshold?: number;
  }) => {
    if (patch.occlusionMode !== undefined) {
      setOccludedDisplayMode(patch.occlusionMode === 'hide' ? 'hide' : 'highlight');
    }
    const next: UnifiedRecipe = {
      ...activeRecipe,
      steps: activeRecipe.steps.map((s) => {
        if (s.op === 'model.segment') {
          return {
            ...s,
            params: {
              ...s.params,
              ...(patch.occlusionMode !== undefined ? { occlusionHandling: patch.occlusionMode } : {}),
              ...(patch.curvatureBoundaryFit !== undefined
                ? { curvatureBoundaryFit: patch.curvatureBoundaryFit }
                : {}),
              ...(patch.overlapConfidenceThreshold !== undefined
                ? { overlapConfidenceThreshold: patch.overlapConfidenceThreshold }
                : {}),
            },
          };
        }
        if (s.op === 'sem.particles') {
          return {
            ...s,
            params: {
              ...s.params,
              ...(patch.curvatureBoundaryFit !== undefined
                ? { curvatureBoundaryFit: patch.curvatureBoundaryFit }
                : {}),
              ...(patch.overlapConfidenceThreshold !== undefined
                ? { overlapConfidenceThreshold: patch.overlapConfidenceThreshold }
                : {}),
            },
          };
        }
        return s;
      }),
    };
    setActiveRecipe(next);
    setRecipesBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: next }));
    if (
      runResult &&
      (patch.curvatureBoundaryFit !== undefined || patch.overlapConfidenceThreshold !== undefined)
    ) {
      runWorkflow(activeSpecimen, next, corrections, localModels, 'Boundary fit update', true);
    }
  };

  const handleResetStepParams = (stepId: string) => {
    const step = activeRecipe.steps.find((s) => s.id === stepId);
    const spec = OPERATOR_CATALOG.find((o) => o.op === step?.op);
    if (!step || !spec) return;
    const defaults: Record<string, number | boolean | string> = {};
    for (const p of spec.params) defaults[p.key] = p.defaultValue;
    const next: UnifiedRecipe = {
      ...activeRecipe,
      steps: activeRecipe.steps.map((s) => (s.id === stepId ? { ...s, params: defaults } : s)),
    };
    applyRecipeChange(next);
  };

  const handleToggleStepEnabled = (stepId: string) => {
    const next: UnifiedRecipe = {
      ...activeRecipe,
      steps: activeRecipe.steps.map((s) => (s.id === stepId ? { ...s, enabled: !s.enabled } : s)),
    };
    applyRecipeChange(next);
    const toggled = next.steps.find((s) => s.id === stepId);
    if (toggled && !toggled.enabled && selectedStepId === stepId) {
      setSelectedStepId(null);
    }
  };

  const handleChangeModelRef = (stepId: string, modelRef: string) => {
    const next: UnifiedRecipe = {
      ...activeRecipe,
      steps: activeRecipe.steps.map((s) =>
        s.id === stepId ? { ...s, enabled: true, modelRef } : s
      ),
    };
    applyRecipeChange(next);
    setSelectedStepId(stepId);
  };

  // Bind a Trained Model from Model Hub directly into active workflow
  const handleBindModelFromHub = (modelRef: string) => {
    if (models.find((m) => m.ref === modelRef)?.fileStatus === 'missing') return; // a card without weights cannot run
    const existing = activeRecipe.steps.find((s) => s.op === 'model.segment');
    let nextSteps = [...activeRecipe.steps];
    if (existing) {
      nextSteps = nextSteps.map((s) =>
        s.id === existing.id ? { ...s, enabled: true, modelRef } : s
      );
    } else {
      const newStep = {
        id: `model_${Date.now().toString(36).slice(-4)}`,
        op: 'model.segment',
        enabled: true,
        params: { threshold: 0.5, tileSize: 512 },
        modelRef,
      };
      const firstMetrology = nextSteps.findIndex((s) => {
        const os = OPERATOR_CATALOG.find((o) => o.op === s.op);
        return os?.category === 'metrology';
      });
      if (firstMetrology >= 0) nextSteps.splice(firstMetrology, 0, newStep);
      else nextSteps.unshift(newStep);
    }
    const nextRecipe: UnifiedRecipe = { ...activeRecipe, steps: nextSteps };
    setActiveRecipe(nextRecipe);
    const res = runWorkflow(activeSpecimen, nextRecipe, corrections, localModels);
    const modelPreview = res.stepPreviews.find((p) => p.op === 'model.segment');
    setSelectedStepId(modelPreview?.stepId ?? null);
    setView('workbench');
    showToast(
      lang === 'zh'
        ? `已将模型 ${modelRef} 绑定至工作流并完成推理`
        : `Bound model ${modelRef} and executed inference`
    );
  };

  // Publish locally trained Active Learning model, persist to IndexedDB & immediately run on Workbench
  const handlePublishLocalModel = (trained: TrainedLocalModel) => {
    const nextMap = new Map(localModels);
    nextMap.set(trained.card.ref, trained);
    setLocalModels(nextMap);
    setModels((prev) => [trained.card, ...prev.filter((m) => m.ref !== trained.card.ref)]);
    void saveLocalModel(trained);

    // Bind into active workflow immediately
    const existing = activeRecipe.steps.find((s) => s.op === 'model.segment');
    let nextSteps = [...activeRecipe.steps];
    if (existing) {
      nextSteps = nextSteps.map((s) =>
        s.id === existing.id ? { ...s, enabled: true, modelRef: trained.card.ref } : s
      );
    } else {
      const newStep = {
        id: `local_model_${Date.now().toString(36).slice(-4)}`,
        op: 'model.segment',
        enabled: true,
        params: { threshold: 0.5, tileSize: 512 },
        modelRef: trained.card.ref,
      };
      const firstMetrology = nextSteps.findIndex((s) => {
        const os = OPERATOR_CATALOG.find((o) => o.op === s.op);
        return os?.category === 'metrology';
      });
      if (firstMetrology >= 0) nextSteps.splice(firstMetrology, 0, newStep);
      else nextSteps.unshift(newStep);
    }
    const nextRecipe: UnifiedRecipe = { ...activeRecipe, steps: nextSteps };
    setActiveRecipe(nextRecipe);
    const res = executeRecipe(activeSpecimen, nextRecipe, {
      corrections,
      localModels: nextMap,
      roi: roiBySpecimenRef.current[activeSpecimen.id] ?? null,
    });
    setRunResult(res);
    recordRunSnapshot(activeSpecimen.id, res, trained.card.ref);
    const mPrev = res.stepPreviews.find((p) => p.op === 'model.segment');
    setSelectedStepId(mPrev?.stepId ?? null);
    showToast(
      lang === 'zh'
        ? `本地模型 ${trained.card.ref} 已保存至 IndexedDB 并应用至工作台`
        : `Saved ${trained.card.ref} to IndexedDB and applied on Workbench`
    );
  };

  // Register SHA-256 verified ONNX model card exported from Local/Cloud GPU LoRA training
  const handleImportOnnxModel = (card: ModelCard) => {
    setModels((prev) => [card, ...prev.filter((m) => m.ref !== card.ref)]);
    showToast(
      lang === 'zh'
        ? `已校验并入库 GPU ONNX 模型：${card.ref}`
        : `Verified and registered GPU ONNX model: ${card.ref}`
    );
  };

  // Persist & register a real offline `.onnx` binary validated by onnxruntime-web
  const handleImportOnnxBinary = (record: PersistedOnnxModel) => {
    const nextOnnx = new Map(onnxModels);
    nextOnnx.set(record.ref, record);
    onnxModelsRef.current = nextOnnx;
    setOnnxModels(nextOnnx);
    setModels((prev) => [record.card, ...prev.filter((m) => m.ref !== record.ref)]);
    void saveOnnxModel(record);
    showToast(
      lang === 'zh'
        ? `ONNX 权重已校验入库（支持浏览器离线 WASM 推理）：${record.ref}`
        : `ONNX binary verified & saved for offline WASM inference: ${record.ref}`
    );
  };

  // Import a portable SHA-256 verified `.vexmodel.json` local DL model into IndexedDB
  const handleImportLocalModelJson = (trained: TrainedLocalModel) => {
    const nextMap = new Map(localModels);
    nextMap.set(trained.card.ref, trained);
    setLocalModels(nextMap);
    setModels((prev) => [trained.card, ...prev.filter((m) => m.ref !== trained.card.ref)]);
    void saveLocalModel(trained);
    showToast(
      lang === 'zh'
        ? `已校验 SHA-256 并导入本地模型：${trained.card.ref}`
        : `Verified SHA-256 and imported local model: ${trained.card.ref}`
    );
  };

  // Delete a persisted local DL model or imported ONNX binary from IndexedDB
  const handleDeletePersistedModel = (ref: string) => {
    if (localModels.has(ref)) {
      const nextMap = new Map(localModels);
      nextMap.delete(ref);
      setLocalModels(nextMap);
      void deleteLocalModel(ref);
    }
    if (onnxModels.has(ref)) {
      const nextOnnx = new Map(onnxModels);
      nextOnnx.delete(ref);
      onnxModelsRef.current = nextOnnx;
      setOnnxModels(nextOnnx);
      for (const key of Array.from(onnxProbCacheRef.current.keys())) {
        if (key.startsWith(`${ref}:`)) onnxProbCacheRef.current.delete(key);
      }
      void disposeOnnxSession(ref);
      void deleteOnnxModel(ref);
    }
    setModels((prev) => prev.filter((m) => m.ref !== ref));
    showToast(
      lang === 'zh' ? `已从本地模型库移除：${ref}` : `Removed persisted model: ${ref}`
    );
  };

  // Human review correction ledger
  const handleAddCorrection = (corr: Omit<HumanCorrection, 'id' | 'timestamp'>) => {
    const nextEntry: HumanCorrection = {
      ...corr,
      id: `corr_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
      timestamp: new Date().toISOString(),
    };
    const nextList = [...corrections, nextEntry];
    setCorrections(nextList);
    runWorkflow(activeSpecimen, activeRecipe, nextList, localModels);
  };

  // Apply SAM-refined segment boundary directly onto active runResult and refresh calibrated morphometry
  const handleApplyRefinedBoundary = (proposal: RefinedBoundaryProposal) => {
    if (!runResult) return;
    const m = proposal.metrics;
    const nextShapes = runResult.shapes.map((s) => {
      const isMatch =
        s.id === proposal.shapeId ||
        (s.objectId === proposal.objectId &&
          s.layer !== 'roi' &&
          s.layer !== 'manual' &&
          s.layer !== 'flags' &&
          (s.kind === 'polygon' || s.kind === 'circle' || s.kind === 'box'));
      if (!isMatch) return s;
      return {
        ...s,
        kind: 'polygon' as const,
        points: proposal.refinedPoints,
        holes: proposal.refinedHoles.length > 0 ? proposal.refinedHoles : undefined,
        dashed: false,
      };
    });

    const nextObjects = runResult.objects.map((o) => {
      if (o.id !== proposal.objectId) return o;
      return {
        ...o,
        x: m.cx,
        y: m.cy,
        area: m.areaPhys,
        diameter: m.equivDiameterPhys,
        width: m.widthPhys,
        circularity: m.circularity,
        aspectRatio: m.aspectRatio,
        orientationDeg: m.orientationDeg,
        boundaryFitScore: m.confidence,
        confidence: Number(Math.max(o.confidence, m.confidence).toFixed(2)),
        flagged: false,
        flagReason: undefined,
        reason: lang === 'zh' ? 'SAM 边界精修已确认' : 'SAM boundary refined',
      };
    });

    // Recompute summary metrics & histogram mean for active non-excluded objects
    const activeObjs = nextObjects.filter((o) => !o.excluded);
    const dias = activeObjs
      .map((o) => o.diameter)
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    const areas = activeObjs
      .map((o) => o.area)
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    const meanDia = dias.length > 0 ? dias.reduce((a, b) => a + b, 0) / dias.length : undefined;
    const meanArea = areas.length > 0 ? areas.reduce((a, b) => a + b, 0) / areas.length : undefined;

    const nextSummary = runResult.summary.map((item) => {
      if (
        meanDia !== undefined &&
        (item.key === 'mean_diameter' || item.key === 'mean_equiv_diameter' || item.key === 'd_mean')
      ) {
        return {
          ...item,
          value: Number(meanDia.toFixed(2)),
        };
      }
      if (meanArea !== undefined && (item.key === 'mean_area' || item.key === 'avg_area')) {
        return { ...item, value: Number(meanArea.toFixed(2)) };
      }
      return item;
    });

    const nextSummaryMap = { ...runResult.summaryMap };
    for (const s of nextSummary) {
      nextSummaryMap[s.key] = s.value;
    }

    const nextHistogram =
      runResult.histogram && dias.length > 0 && meanDia !== undefined
        ? {
            ...runResult.histogram,
            values: dias,
            mean: Number(meanDia.toFixed(2)),
          }
        : runResult.histogram;

    const nextResult: RunResult = {
      ...runResult,
      shapes: nextShapes,
      objects: nextObjects,
      summary: nextSummary,
      summaryMap: nextSummaryMap,
      histogram: nextHistogram,
      resultDigest: `${runResult.resultDigest.slice(0, 14)}_ref${proposal.objectId}_${Date.now().toString(36)}`,
    };

    setRunResult(nextResult);
    setResultsBySpecimen((prev) => ({
      ...prev,
      [activeSpecimen.id]: nextResult,
    }));
    setSelectedObjectId(proposal.objectId);

    const edgeSign = proposal.edgeGainPct >= 0 ? '+' : '';
    showToast(
      lang === 'zh'
        ? `已精修目标 #${proposal.objectId} 边界（面积 ${m.areaPhys.toFixed(1)} ${m.areaUnit}，边缘贴合 ${edgeSign}${proposal.edgeGainPct.toFixed(0)}%）`
        : `Refined boundary for #${proposal.objectId} (Area ${m.areaPhys.toFixed(1)} ${m.areaUnit}, Edge Snap ${edgeSign}${proposal.edgeGainPct.toFixed(0)}%)`
    );
  };

  // Atomic one-click application for anomaly-suggested HumanCorrections & parameter/noise threshold fixes
  const handleApplySuggestedFix = (suggestions: CorrectionSuggestion[]) => {
    if (suggestions.length === 0) return;
    const now = new Date().toISOString();
    const added: HumanCorrection[] = [];
    let nextRecipe: UnifiedRecipe = activeRecipe;
    let recipeChanged = false;

    for (const suggestion of suggestions) {
      suggestion.corrections.forEach((c, idx) => {
        added.push({
          ...c,
          id: `corr_${Date.now()}_${added.length}_${idx}_${Math.random().toString(36).slice(2, 5)}`,
          timestamp: now,
        });
      });

      if (suggestion.paramFix) {
        const pf = suggestion.paramFix;
        nextRecipe = {
          ...nextRecipe,
          steps: nextRecipe.steps.map((s) => {
            if (s.id !== pf.stepId) return s;
            if (pf.toggleEnable) return { ...s, enabled: !s.enabled };
            if (pf.key !== undefined && pf.value !== undefined) {
              return { ...s, params: { ...s.params, [pf.key]: pf.value } };
            }
            return s;
          }),
        };
        recipeChanged = true;
      }
    }

    if (recipeChanged) {
      setActiveRecipe(nextRecipe);
    }

    const nextCorrections = added.length > 0 ? [...corrections, ...added] : corrections;
    if (added.length > 0) {
      setCorrections(nextCorrections);
    }

    runWorkflow(activeSpecimen, nextRecipe, nextCorrections, localModels);
    const label =
      suggestions.length === 1
        ? lang === 'zh'
          ? `已应用修复：${suggestions[0].titleZh}`
          : `Applied fix: ${suggestions[0].titleEn}`
        : lang === 'zh'
        ? `已批量应用 ${suggestions.length} 项异常修复（${added.length} 条纠错）`
        : `Applied ${suggestions.length} anomaly fixes (${added.length} correction edits)`;
    showToast(label);
  };

  // Delete single or all HumanCorrection records for the active specimen
  const handleDeleteCorrection = (id: string) => {
    const nextList = corrections.filter((c) => c.id !== id);
    setCorrections(nextList);
    runWorkflow(activeSpecimen, activeRecipe, nextList, localModels, 'Revert edit');
  };

  const handleClearCorrections = () => {
    const nextList = corrections.filter((c) => c.specimenId !== activeSpecimen.id);
    setCorrections(nextList);
    runWorkflow(activeSpecimen, activeRecipe, nextList, localModels, 'Cleared edits');
  };

  // Individual deletion for canvas strokes, SAM prompts, and manual measurements
  const handleDeleteStroke = (id: string) => {
    setStrokesBySpecimen((prev) => ({
      ...prev,
      [activeSpecimen.id]: (prev[activeSpecimen.id] ?? []).filter((s) => s.id !== id),
    }));
  };

  const handleDeletePrompt = (id: string) => {
    setPromptsBySpecimen((prev) => ({
      ...prev,
      [activeSpecimen.id]: (prev[activeSpecimen.id] ?? []).filter((p) => p.id !== id),
    }));
  };

  const handleDeleteMeasurement = (id: string) => {
    setMeasurementsBySpecimen((prev) => ({
      ...prev,
      [activeSpecimen.id]: (prev[activeSpecimen.id] ?? []).filter((m) => m.id !== id),
    }));
  };

  // Generate an alternate execution version immediately when user clicks Compare on a 1-version specimen
  const handleGenerateComparisonBaseline = () => {
    const altRecipe: UnifiedRecipe = JSON.parse(JSON.stringify(activeRecipe));
    const bgStep = altRecipe.steps.find((s) => s.op === 'sem.background');
    const dropletStep = altRecipe.steps.find((s) => s.op === 'droplet.analyze');
    const linesStep = altRecipe.steps.find((s) => s.op === 'sem.lines');
    const holesStep = altRecipe.steps.find((s) => s.op === 'sem.holes');
    const particlesStep = altRecipe.steps.find((s) => s.op === 'sem.particles');
    const fibresStep = altRecipe.steps.find((s) => s.op === 'sem.fibres');
    const intensitySegStep = altRecipe.steps.find((s) => s.op === 'generic.intensity_seg');

    let altLabel = 'Alternate';
    if (dropletStep) {
      const cur = Number(dropletStep.params.cellContrast ?? 36);
      dropletStep.params.cellContrast = Math.min(120, cur + 14);
      altLabel = `cellContrast=${dropletStep.params.cellContrast}`;
    } else if (linesStep) {
      linesStep.params.method = linesStep.params.method === 'gradient' ? 'threshold' : 'gradient';
      altLabel = `edge=${linesStep.params.method}`;
    } else if (holesStep) {
      const cur = Number(holesStep.params.level ?? 0.5);
      holesStep.params.level = Number(Math.min(0.8, cur + 0.08).toFixed(2));
      altLabel = `level=${holesStep.params.level}`;
    } else if (particlesStep) {
      const cur = Number(particlesStep.params.minDiameterPx ?? 10);
      particlesStep.params.minDiameterPx = Math.min(80, cur + 6);
      altLabel = `minDia=${particlesStep.params.minDiameterPx}px`;
    } else if (fibresStep) {
      const cur = Number(fibresStep.params.thresholdBias ?? 0.5);
      fibresStep.params.thresholdBias = Number(Math.min(0.85, cur + 0.1).toFixed(2));
      altLabel = `bias=${fibresStep.params.thresholdBias}`;
    } else if (intensitySegStep) {
      intensitySegStep.params.method =
        intensitySegStep.params.method === 'adaptive' ? 'otsu' : 'adaptive';
      altLabel = `seg=${intensitySegStep.params.method}`;
    } else if (bgStep) {
      bgStep.enabled = !bgStep.enabled;
      altLabel = `bg=${bgStep.enabled ? 'ON' : 'OFF'}`;
    }

    const res = executeRecipe(activeSpecimen, altRecipe, {
      corrections,
      localModels,
      roi: roiBySpecimenRef.current[activeSpecimen.id] ?? null,
    });
    recordRunSnapshot(activeSpecimen.id, res, altLabel);
    showToast(
      lang === 'zh'
        ? `已生成对照版本 (${altLabel})，可在右侧查看指标差异`
        : `Generated comparison run (${altLabel})`
    );
  };

  // Unified Batch execution (microfluidics + SEM in one queue, failure isolated per row)
  const handleRunBatch = () => {
    if (batchRows.length === 0) return;
    setBatchRunning(true);
    const updated: BatchRow[] = batchRows.map((row) => {
      try {
        const effectiveSpec: Specimen = row.pixelSizeOverride
          ? {
              ...row.specimen,
              calibration: {
                ...row.specimen.calibration,
                pixelSize: row.pixelSizeOverride,
                confirmed: true,
              },
            }
          : row.specimen;

        // Pick matched recipe for row's modality/subType (or activeRecipe if same modality)
        let recipeForRow = activeRecipe;
        if (effectiveSpec.modality !== activeRecipe.modality) {
          const auto = buildAutoPlan(effectiveSpec, '', localModels);
          recipeForRow = auto.suggestedRecipe;
        } else if (effectiveSpec.subType === 'sem_particles' && activeRecipe.id === 'sem_lines') {
          recipeForRow =
            BUILTIN_RECIPES.find((r) => r.id === 'sem_particles') ?? activeRecipe;
        } else if (effectiveSpec.subType === 'sem_lines' && activeRecipe.id === 'sem_particles') {
          recipeForRow =
            BUILTIN_RECIPES.find((r) => r.id === 'sem_lines') ?? activeRecipe;
        }

        const res = executeRecipe(effectiveSpec, recipeForRow, {
          corrections,
          localModels,
        });
        return {
          ...row,
          specimen: effectiveSpec,
          status: res.warnings.length > 0 ? 'warning' : 'completed',
          durationMs: res.durationMs,
          warnings: res.warnings,
          error: undefined,
          result: res,
        };
      } catch (err) {
        return {
          ...row,
          status: 'failed',
          error: err instanceof Error ? err.message : String(err),
          warnings: [],
        };
      }
    });
    setBatchRows(updated);
    setBatchRunning(false);
    showToast(
      lang === 'zh'
        ? `批量分析完成：${updated.length} 个样本已更新分组统计`
        : `Batch completed: ${updated.length} specimens analysed`
    );
  };

  // Jump from Batch Abnormal Queue straight to Canvas Review tab
  const handleInspectBatchRow = (row: BatchRow) => {
    setSpecimens((prev) =>
      prev.some((s) => s.id === row.specimen.id) ? prev : [row.specimen, ...prev]
    );
    setActiveSpecimenId(row.specimen.id);
    if (row.result) {
      setActiveRecipe(row.result.recipe);
      setRunResult(row.result);
      setResultsBySpecimen((prev) => ({ ...prev, [row.specimen.id]: row.result! }));
      setRecipesBySpecimen((prev) => ({ ...prev, [row.specimen.id]: row.result!.recipe }));
      setPlanProposal(buildAutoPlan(row.specimen, '', localModels, undefined, { runTrial: false }));
      setPlanAccepted(true);
    } else {
      selectSpecimenAndPlan(row.specimen, localModels, corrections);
    }
    setSelectedStepId(null);
    setWorkbenchStage('results');
    setResultsSubTab('review');
    setView('workbench');
  };

  // Verify local Python `vexel serve` engine bridge
  const checkPythonBridge = async () => {
    setPythonCheckState({ status: 'checking' });
    try {
      const resp = await fetch('/api/v1/python-bridge/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ baseUrl: pythonUrl }),
      });
      const data = await resp.json();
      if (resp.ok && data.ok) {
        setPythonCheckState({
          status: 'ok',
          message: `Connected (${data.data?.engine ?? 'vexel'} ${data.data?.version ?? ''})`,
        });
        setEngineMode('python');
      } else {
        setPythonCheckState({
          status: 'error',
          message: data.error ?? 'Connection refused',
        });
      }
    } catch (err) {
      setPythonCheckState({
        status: 'error',
        message: err instanceof Error ? err.message : 'Network error',
      });
    }
  };

  const activeModelRef = useMemo(
    () => activeRecipe.steps.find((s) => s.op === 'model.segment' && s.enabled)?.modelRef,
    [activeRecipe]
  );

  const activeStrokes = strokesBySpecimen[activeSpecimen.id] ?? [];
  const activePrompts = promptsBySpecimen[activeSpecimen.id] ?? [];
  const activeMeasurements = measurementsBySpecimen[activeSpecimen.id] ?? [];
  const activeDismissedLowConfIds = dismissedLowConfBySpecimen[activeSpecimen.id] ?? [];

  const handleDismissLowConfSegment = (segId: string) => {
    setDismissedLowConfBySpecimen((prev) => {
      const cur = prev[activeSpecimen.id] ?? [];
      if (cur.includes(segId)) return prev;
      return { ...prev, [activeSpecimen.id]: [...cur, segId] };
    });
    setActiveCorrectionSegId((cur) => (cur === segId ? null : cur));
  };

  const handleTriggerLowConfCorrection = (seg: LowConfidenceSegment | null) => {
    if (!seg) {
      setActiveCorrectionSegId(null);
      return;
    }
    setActiveCorrectionSegId(seg.id);
    if (seg.objectId !== undefined) {
      setSelectedObjectId(seg.objectId);
    }
    setWorkbenchStage('results');
    setResultsSubTab('review');
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-slate-50 dark:bg-[#0B0E17] text-slate-900 dark:text-slate-100">
      {/* Strict 3-Zone Top Bar */}
      <TopBar
        lang={lang}
        view={view}
        onChangeView={setView}
        hasResult={Boolean(runResult)}
        onExportExcel={() => runResult && exportSingleExcel(activeSpecimen, runResult)}
        onExportCsv={() => runResult && exportObjectsCsv(activeSpecimen, runResult)}
        onExportSvg={() => runResult && exportAnnotatedSvg(activeSpecimen, runResult)}
        onExportHtml={() =>
          runResult && exportHtmlReport(activeSpecimen, runResult, canvasSnapshotRef.current())
        }
        onExportVexflow={() => runResult && exportVexflow(runResult)}
        onExportReproducePy={() => runResult && exportReproducePy(activeSpecimen, runResult)}
      />

      {/* Workspace Sub-Ribbon: Active Specimen Context · Interactive Pipeline Stepper · Secondary Utilities */}
      <div className="h-10 shrink-0 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#0E1320] px-4 flex items-center justify-between gap-3 text-[12px] select-none z-20">
        {/* Hidden shared file input triggered exclusively from Stage 1 (Import) */}
        <input
          ref={uploadInputRef}
          type="file"
          accept={ACCEPTED_MICROSCOPY_EXTENSIONS}
          onChange={handleUploadSpecimen}
          className="hidden"
        />

        {/* Left: Compact Active Specimen Readout (clicks to Stage 1 Import / Switch) */}
        <div className="flex items-center gap-2 min-w-0 shrink-0">
          <button
            type="button"
            onClick={() => {
              setView('workbench');
              setWorkbenchStage('import');
            }}
            title={
              lang === 'zh'
                ? '点击前往右侧「1. 导入」切换样本或导入新图像'
                : 'Open Stage 1 (Import) to switch specimen or import a new image'
            }
            className="h-7 max-w-[290px] px-2.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800/90 hover:border-blue-500 dark:hover:border-cyan-400 hover:bg-blue-50/60 dark:hover:bg-slate-700/80 shadow-xs text-left flex items-center gap-1.5 min-w-0 cursor-pointer transition-colors"
          >
            <Microscope className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400 shrink-0" />
            <span className="font-mono text-[11px] font-semibold text-slate-900 dark:text-slate-100 truncate">
              {activeSpecimen.name}
            </span>
            <span className="font-mono text-[10.5px] text-slate-500 dark:text-slate-400 shrink-0 tabular-nums">
              {activeSpecimen.width}×{activeSpecimen.height}
            </span>
          </button>
        </div>

        {/* Center: Clickable Pipeline Stepper (0. Original -> 1. Data Bar Crop (when auto-cropped) -> 2. Background Leveling -> ...) */}
        {view === 'workbench' && runResult && (
          <div className="hidden md:flex items-center gap-1 overflow-x-auto min-w-0 px-1">
            <button
              type="button"
              onClick={() =>
                setSelectedStepId(selectedStepId === RAW_SOURCE_STEP_ID ? null : RAW_SOURCE_STEP_ID)
              }
              title={
                lang === 'zh'
                  ? '查看未裁切的原始导入图像（含信息栏与自动裁切参考线）'
                  : 'Inspect original uncropped source image with info bar & crop guide'
              }
              className={`h-6 px-2 rounded border text-[11px] font-mono whitespace-nowrap transition-colors cursor-pointer ${
                selectedStepId === RAW_SOURCE_STEP_ID
                  ? 'border-blue-600/60 dark:border-cyan-400/70 bg-blue-600/15 dark:bg-cyan-500/20 text-blue-700 dark:text-cyan-200 font-semibold'
                  : 'border-slate-200 dark:border-slate-700/80 bg-white dark:bg-slate-800/70 text-slate-700 dark:text-slate-300 hover:border-blue-400 dark:hover:border-cyan-400 hover:text-blue-700 dark:hover:text-cyan-200'
              }`}
            >
              0. {t('canvas_source', lang)}
            </button>
            {runResult.stepPreviews.map((sp, idx) => {
              const isLast = idx === runResult.stepPreviews.length - 1;
              const active =
                selectedStepId === sp.stepId || (selectedStepId === null && isLast);
              return (
                <div key={sp.stepId} className="flex items-center gap-1 shrink-0">
                  <span className="text-slate-400 dark:text-slate-500 text-[10px]">→</span>
                  <button
                    type="button"
                    onClick={() => {
                      if (isLast) {
                        setSelectedStepId(null);
                      } else {
                        setSelectedStepId(selectedStepId === sp.stepId ? null : sp.stepId);
                      }
                    }}
                    title={`${sp.op} · ${sp.qcSummary} (${sp.durationMs} ms)`}
                    className={`h-6 px-2 rounded border text-[11px] font-mono whitespace-nowrap transition-colors cursor-pointer ${
                      active
                        ? 'border-blue-600/60 dark:border-cyan-400/70 bg-blue-600/15 dark:bg-cyan-500/20 text-blue-700 dark:text-cyan-200 font-semibold'
                        : 'border-slate-200 dark:border-slate-700/80 bg-white dark:bg-slate-800/70 text-slate-700 dark:text-slate-300 hover:border-blue-400 dark:hover:border-cyan-400 hover:text-blue-700 dark:hover:text-cyan-200'
                    }`}
                  >
                    {idx + 1}. {lang === 'zh' ? sp.labelZh : sp.labelEn}
                    {sp.probMap ? ' · AI' : ''}
                  </button>
                </div>
              );
            })}
          </div>
        )}

        {/* Right: Engine Bridge, Language Toggle, Dark/Light Theme Toggle */}
        <div className="relative flex items-center gap-1.5 shrink-0">
          <button
            ref={engineTriggerRef}
            type="button"
            onClick={() => setEnginePopoverOpen((v) => !v)}
            className="h-7 px-2.5 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800/90 hover:border-blue-500 dark:hover:border-cyan-400 hover:bg-blue-50/60 dark:hover:bg-slate-700 shadow-xs text-[11px] font-mono font-medium text-slate-800 dark:text-slate-200 flex items-center gap-1.5 whitespace-nowrap cursor-pointer"
          >
            <Server className="w-3 h-3 text-blue-600 dark:text-cyan-400" />
            <span>
              {engineMode === 'python'
                ? t('engine_python', lang)
                : t('engine_web', lang)}
            </span>
          </button>

          <button
            type="button"
            onClick={() => setLang((l) => (l === 'zh' ? 'en' : 'zh'))}
            className="h-7 px-2.5 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800/90 hover:border-blue-500 dark:hover:border-cyan-400 hover:bg-blue-50/60 dark:hover:bg-slate-700 shadow-xs text-[11px] font-mono font-medium text-slate-800 dark:text-slate-200 flex items-center gap-1 whitespace-nowrap cursor-pointer"
            title="Switch Language"
          >
            <Globe className="w-3 h-3" />
            <span>{lang === 'zh' ? 'EN' : '中文'}</span>
          </button>

          <button
            type="button"
            onClick={() => setDarkMode((d) => !d)}
            className="h-7 px-2.5 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800/90 hover:border-blue-500 dark:hover:border-cyan-400 hover:bg-blue-50/60 dark:hover:bg-slate-700 shadow-xs text-[11px] font-medium text-slate-800 dark:text-slate-200 flex items-center gap-1.5 whitespace-nowrap cursor-pointer"
            title={darkMode ? (lang === 'zh' ? '切换到亮色界面' : 'Switch to Light Mode') : (lang === 'zh' ? '切换到暗色界面' : 'Switch to Dark Mode')}
          >
            {darkMode ? (
              <>
                <Sun className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                <span className="hidden sm:inline">{lang === 'zh' ? '亮色' : 'Light'}</span>
              </>
            ) : (
              <>
                <Moon className="w-3.5 h-3.5 text-blue-600 shrink-0" />
                <span className="hidden sm:inline">{lang === 'zh' ? '暗色' : 'Dark'}</span>
              </>
            )}
          </button>

          {/* Engine & Offline Policy Popover */}
          {enginePopoverOpen && (
            <div
              ref={enginePopoverRef}
              className="absolute right-0 top-9 w-80 p-3.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-[#111827] shadow-xl space-y-3 z-50 text-[12px]"
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-900 dark:text-slate-100">
                  {lang === 'zh' ? '执行引擎与合规策略' : 'Engine & Compliance Policy'}
                </span>
                <button
                  type="button"
                  onClick={() => setEnginePopoverOpen(false)}
                  className="text-slate-400 hover:text-slate-200 cursor-pointer"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-1.5">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="engineMode"
                    checked={engineMode === 'web'}
                    onChange={() => setEngineMode('web')}
                    className="accent-cyan-500"
                  />
                  <span>{t('engine_web', lang)} (TypeScript / Canvas2D)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="engineMode"
                    checked={engineMode === 'python'}
                    onChange={() => setEngineMode('python')}
                    className="accent-cyan-500"
                  />
                  <span>{t('engine_python', lang)} (`vexel serve`)</span>
                </label>
              </div>

              <div className="space-y-1.5 pt-1 border-t border-slate-200 dark:border-slate-800">
                <div className="text-[11px] text-slate-500">
                  {lang === 'zh' ? '本地 Python 引擎地址 (uv run vexel serve)' : 'Local Python Engine URL'}
                </div>
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    value={pythonUrl}
                    onChange={(e) => setPythonUrl(e.target.value)}
                    className="flex-1 h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px]"
                  />
                  <button
                    type="button"
                    onClick={checkPythonBridge}
                    className="h-7 px-2.5 rounded bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 text-[11px] font-medium cursor-pointer"
                  >
                    {pythonCheckState.status === 'checking'
                      ? '...'
                      : lang === 'zh'
                      ? '测试连接'
                      : 'Test'}
                  </button>
                </div>
                {pythonCheckState.status === 'ok' && (
                  <div className="text-[11px] text-emerald-500 flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>{pythonCheckState.message}</span>
                  </div>
                )}
                {pythonCheckState.status === 'error' && (
                  <div className="text-[11px] text-amber-500 flex items-center gap-1">
                    <XCircle className="w-3.5 h-3.5 shrink-0" />
                    <span className="truncate">{pythonCheckState.message}</span>
                  </div>
                )}
              </div>

              <label className="pt-2 border-t border-slate-200 dark:border-slate-800 flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={offlinePolicy}
                  onChange={(e) => setOfflinePolicy(e.target.checked)}
                  className="accent-cyan-500"
                />
                <span className="text-[11.5px] text-slate-600 dark:text-slate-300">
                  {t('engine_offline', lang)} ({lang === 'zh' ? '禁止任何云端 AI 请求' : 'Block external AI calls'})
                </span>
              </label>
            </div>
          )}
        </div>
      </div>

      {/* Dedicated Natural-Language AI Agent Bar (Workbench) */}
      {view === 'workbench' && (
        <AgentCommandBar
          lang={lang}
          modality={activeSpecimen.modality}
          offlinePolicy={offlinePolicy}
          aiLoading={aiLoading}
          plan={planProposal}
          providers={providers}
          providerConfig={providerConfig}
          onChangeProviderConfig={setProviderConfig}
          onSubmitGoal={async (goal, sendImage) => {
            await handleAiGoal(goal, sendImage);
            setWorkbenchStage('workflow');
          }}
        />
      )}

      {/* Main Workspace Area */}
      <main className="flex-1 min-h-0 flex overflow-hidden">
        {view === 'workbench' && (
          <>
            <ScientificCanvas
              specimen={activeSpecimen}
              result={runResult}
              pythonShapes={
                showPythonOverlay && pythonOverlay?.key === overlayKey(activeSpecimen, activeRecipe.id)
                  ? pythonOverlay.shapes
                  : null
              }
              lang={lang}
              darkMode={darkMode}
              activeRecipe={activeRecipe}
              savedWorkflows={savedWorkflows}
              importInspecting={importInspecting}
              pendingCropDraft={
                pendingCropDraftBySpecimen[activeSpecimen.id] ?? {
                  cropPx: activeSpecimen.importReport?.proposedCropBarPx ?? 0,
                  barPosition:
                    activeSpecimen.importReport?.barPosition &&
                    activeSpecimen.importReport.barPosition !== 'none'
                      ? activeSpecimen.importReport.barPosition
                      : 'bottom',
                }
              }
              manualCalLine={manualCalLineBySpecimen[activeSpecimen.id] ?? null}
              onSetManualCalLine={(line) => {
                setManualCalLineBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: line }));
                if (line) {
                  setWorkbenchStage('import');
                  showToast(
                    lang === 'zh'
                      ? `已描测比例尺线段：${line.lengthPx} px，请在右侧第 1 步输入对应实际物理长度并确认标定`
                      : `Traced scale bar: ${line.lengthPx} px — enter its printed physical length in Step 1 to confirm calibration`
                  );
                }
              }}
              onConfirmInfoBarCrop={handleConfirmInfoBarCrop}
              onConfirmCalibration={handleConfirmCalibration}
              onRunActiveWorkflow={() => handleRunActiveWorkflow()}
              onSelectWorkflow={(wfId) => handleSelectWorkflow(wfId)}
              onSaveCustomWorkflow={(name) => handleSaveCustomWorkflow(name)}
              onDeleteCustomWorkflow={handleDeleteCustomWorkflow}
              selectedStepId={selectedStepId}
              selectedObjectId={selectedObjectId}
              onSelectObject={(id) => {
                setSelectedObjectId(id);
                if (id !== null) {
                  setWorkbenchStage('results');
                  if (resultsSubTab === 'provenance') setResultsSubTab('review');
                }
              }}
              toolMode={toolMode}
              onChangeToolMode={setToolMode}
              roi={roiBySpecimen[activeSpecimen.id] ?? null}
              onChangeRoi={handleChangeRoi}
              showLowConfHighlight={showLowConfHighlight}
              onToggleLowConfHighlight={() => setShowLowConfHighlight((v) => !v)}
              lowConfThreshold={lowConfThreshold}
              occludedDisplayMode={occludedDisplayMode}
              onChangeOccludedDisplayMode={setOccludedDisplayMode}
              dismissedLowConfIds={activeDismissedLowConfIds}
              onDismissLowConfSegment={handleDismissLowConfSegment}
              activeCorrectionSegId={activeCorrectionSegId}
              onTriggerLowConfCorrection={handleTriggerLowConfCorrection}
              onAddCorrection={handleAddCorrection}
              onApplyRefinedBoundary={handleApplyRefinedBoundary}
              strokes={activeStrokes}
              onAddStroke={(st) =>
                setStrokesBySpecimen((prev) => ({
                  ...prev,
                  [activeSpecimen.id]: [...(prev[activeSpecimen.id] ?? []), st],
                }))
              }
              onDeleteStroke={handleDeleteStroke}
              onClearStrokes={() =>
                setStrokesBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: [] }))
              }
              brushClassId={brushClassId}
              onChangeBrushClassId={setBrushClassId}
              brushRadius={brushRadius}
              onChangeBrushRadius={setBrushRadius}
              prompts={activePrompts}
              onAddPrompt={(pt) =>
                setPromptsBySpecimen((prev) => ({
                  ...prev,
                  [activeSpecimen.id]: [...(prev[activeSpecimen.id] ?? []), pt],
                }))
              }
              onDeletePrompt={handleDeletePrompt}
              onClearPrompts={() =>
                setPromptsBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: [] }))
              }
              measurements={activeMeasurements}
              onAddMeasurement={(m) =>
                setMeasurementsBySpecimen((prev) => ({
                  ...prev,
                  [activeSpecimen.id]: [...(prev[activeSpecimen.id] ?? []), m],
                }))
              }
              onDeleteMeasurement={handleDeleteMeasurement}
              onClearMeasurements={() =>
                setMeasurementsBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: [] }))
              }
              onRegisterSnapshot={(fn) => {
                canvasSnapshotRef.current = fn;
              }}
            />

            <InspectorPanel
              engineCard={
                <PythonEngineCard
                  specimen={activeSpecimen}
                  recipeId={activeRecipe.id}
                  pythonUrl={pythonUrl}
                  connected={pythonCheckState.status === 'ok'}
                  lang={lang}
                  onShapes={setPythonOverlay}
                  overlayVisible={showPythonOverlay}
                  onToggleOverlay={setShowPythonOverlay}
                  previewSummary={runResult?.summary.map((m) => ({
                    key: m.key,
                    label: lang === 'zh' ? m.labelZh : m.labelEn,
                    value: m.value,
                    unit: m.unit,
                  }))}
                />
              }
              specimen={activeSpecimen}
              specimens={specimens}
              onSwitchSpecimen={handleSwitchSpecimen}
              onTriggerUpload={() => uploadInputRef.current?.click()}
              onDropFile={(file) => {
                void processUploadedFile(file);
              }}
              autoRunOnImport={autoRunOnImport}
              onToggleAutoRunOnImport={() => setAutoRunOnImport((v) => !v)}
              pendingCropDraft={
                pendingCropDraftBySpecimen[activeSpecimen.id] ?? {
                  cropPx: activeSpecimen.importReport?.proposedCropBarPx ?? 0,
                  barPosition:
                    activeSpecimen.importReport?.barPosition &&
                    activeSpecimen.importReport.barPosition !== 'none'
                      ? activeSpecimen.importReport.barPosition
                      : 'bottom',
                }
              }
              manualCalLine={manualCalLineBySpecimen[activeSpecimen.id] ?? null}
              onClearManualCalLine={() =>
                setManualCalLineBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: null }))
              }
              onUpdatePendingCropDraft={handleUpdatePendingCropDraft}
              onConfirmInfoBarCrop={handleConfirmInfoBarCrop}
              onRevertInfoBarCrop={handleRevertInfoBarCrop}
              savedWorkflows={savedWorkflows}
              onSelectWorkflow={(wfId, runNow) => handleSelectWorkflow(wfId, { runNow })}
              onRunActiveWorkflow={() => handleRunActiveWorkflow()}
              recipe={activeRecipe}
              onApplyRecipeSteps={applyRecipeChange}
              plan={planProposal}
              planAccepted={planAccepted}
              result={runResult}
              lang={lang}
              stage={workbenchStage}
              onChangeStage={setWorkbenchStage}
              resultsSubTab={resultsSubTab}
              onChangeResultsSubTab={setResultsSubTab}
              selectedStepId={selectedStepId}
              onSelectStepId={setSelectedStepId}
              selectedObjectId={selectedObjectId}
              onSelectObject={setSelectedObjectId}
              onConfirmCalibration={handleConfirmCalibration}
              onUpdateCalibration={updateActiveCalibration}
              onAcceptPlan={() => {
                handleRunActiveWorkflow();
              }}
              onCustomizeWorkflow={() => setView('workflow')}
              onUpdateStepParam={handleUpdateStepParam}
              onResetStepParams={handleResetStepParams}
              onToggleStepEnabled={handleToggleStepEnabled}
              onChangeModelRef={handleChangeModelRef}
              models={models}
              activeModelRef={
                activeRecipe.steps.find((s) => s.op === 'model.segment' && s.enabled)?.modelRef
              }
              onBindModel={handleBindModelFromHub}
              onPublishLocalModel={handlePublishLocalModel}
              toolMode={toolMode}
              onChangeToolMode={setToolMode}
              roi={roiBySpecimen[activeSpecimen.id] ?? null}
              onChangeRoi={handleChangeRoi}
              showLowConfHighlight={showLowConfHighlight}
              onToggleLowConfHighlight={() => setShowLowConfHighlight((v) => !v)}
              lowConfThreshold={lowConfThreshold}
              onChangeLowConfThreshold={setLowConfThreshold}
              occludedDisplayMode={occludedDisplayMode}
              onChangeOccludedDisplayMode={setOccludedDisplayMode}
              onUpdateModelOcclusionConfig={handleUpdateModelOcclusionConfig}
              dismissedLowConfIds={activeDismissedLowConfIds}
              onDismissLowConfSegment={handleDismissLowConfSegment}
              activeCorrectionSegId={activeCorrectionSegId}
              onTriggerLowConfCorrection={handleTriggerLowConfCorrection}
              brushClassId={brushClassId}
              onChangeBrushClassId={setBrushClassId}
              brushRadius={brushRadius}
              onChangeBrushRadius={setBrushRadius}
              corrections={corrections}
              onAddCorrection={handleAddCorrection}
              onDeleteCorrection={handleDeleteCorrection}
              onClearCorrections={handleClearCorrections}
              onApplySuggestedFix={handleApplySuggestedFix}
              measurements={activeMeasurements}
              onDeleteMeasurement={handleDeleteMeasurement}
              onClearMeasurements={() =>
                setMeasurementsBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: [] }))
              }
              strokes={activeStrokes}
              onDeleteStroke={handleDeleteStroke}
              onClearStrokes={() =>
                setStrokesBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: [] }))
              }
              prompts={activePrompts}
              onDeletePrompt={handleDeletePrompt}
              onClearPrompts={() =>
                setPromptsBySpecimen((prev) => ({ ...prev, [activeSpecimen.id]: [] }))
              }
              runHistory={runHistoryBySpecimen[activeSpecimen.id] ?? []}
              onGenerateComparisonBaseline={handleGenerateComparisonBaseline}
              onExportVexflow={() => runResult && exportVexflow(runResult)}
              onExportReproducePy={() => runResult && exportReproducePy(activeSpecimen, runResult)}
              onOpen3DStudio={() => setView('studio3d')}
              volumeAnalysis={sharedVolumeAnalysis}
              isAnalyzing3D={analyzing3DQuick}
              onRunQuick3DAnalysis={() => void handleRunQuick3DAnalysis()}
            />
          </>
        )}

        {view === 'workflow' && (
          <WorkflowStudio
            lang={lang}
            recipe={activeRecipe}
            models={models}
            savedWorkflows={savedWorkflows}
            onChangeRecipe={(next) => applyRecipeChange(next, true)}
            onSaveCustomWorkflow={handleSaveCustomWorkflow}
            onDeleteCustomWorkflow={handleDeleteCustomWorkflow}
            onApplyAndRun={(next) => {
              setActiveRecipe(next);
              runWorkflow(activeSpecimen, next, corrections, localModels);
              setSelectedStepId(null);
              setView('workbench');
              showToast(lang === 'zh' ? '自定义工作流已执行完成' : 'Custom workflow executed');
            }}
            onExportVexflow={() => runResult && exportVexflow(runResult)}
            onAiEditWorkflow={(goal) => handleAiGoal(goal, false)}
            aiLoading={aiLoading}
          />
        )}

        {view === 'models' && (
          <ModelHub
            lang={lang}
            specimen={activeSpecimen}
            result={runResult}
            roi={roiBySpecimen[activeSpecimen.id] ?? null}
            onChangeRoi={handleChangeRoi}
            models={models}
            localModels={localModels}
            onnxModels={onnxModels}
            activeModelRef={activeModelRef}
            strokes={activeStrokes}
            corrections={corrections}
            onBindModel={handleBindModelFromHub}
            onPublishLocalModel={handlePublishLocalModel}
            onImportOnnxModel={handleImportOnnxModel}
            onImportOnnxBinary={handleImportOnnxBinary}
            onImportLocalModelJson={handleImportLocalModelJson}
            onDeletePersistedModel={handleDeletePersistedModel}
            onGoSmartGtOnCanvas={() => {
              setToolMode('smart_gt');
              setView('workbench');
            }}
            onGoBrushOnCanvas={() => {
              setToolMode('brush');
              setView('workbench');
            }}
            onGoRoiOnCanvas={() => {
              setToolMode('roi');
              setView('workbench');
            }}
          />
        )}

        {view === 'studio3d' && (
          <Studio3DView
            lang={lang}
            specimen={activeSpecimen}
            runResult={runResult}
            roi={roiBySpecimen[activeSpecimen.id] ?? null}
            selected2DObjectId={selectedObjectId}
            occludedDisplayMode={occludedDisplayMode}
            onAnalysisChange={setSharedVolumeAnalysis}
            onNotify={showToast}
          />
        )}

        {view === 'batch' && (
          <BatchCenter
            lang={lang}
            rows={batchRows}
            onChangeRows={setBatchRows}
            onRunBatch={handleRunBatch}
            running={batchRunning}
            onInspectRowOnCanvas={handleInspectBatchRow}
          />
        )}
      </main>

      {/* Non-blocking inline toast */}
      {toast && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-md border border-slate-700 bg-slate-900/95 text-slate-100 text-[12.5px] shadow-lg">
          {toast}
        </div>
      )}
    </div>
  );
}
