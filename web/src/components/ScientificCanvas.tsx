import { useEffect, useMemo, useRef, useState } from 'react';
import { useClickOutside } from '../hooks/useClickOutside';
import {
  AlertTriangle,
  BookmarkPlus,
  Brush,
  Check,
  CheckCircle2,
  ChevronDown,
  Crop,
  Crosshair,
  Eye,
  EyeOff,
  Layers,
  Loader2,
  Maximize2,
  MinusCircle,
  MousePointer2,
  MoveHorizontal,
  Play,
  PlusCircle,
  RotateCcw,
  Ruler,
  ScanSearch,
  Scissors,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Type,
  Activity,
  Undo2,
  Wand2,
  Workflow,
  X,
} from 'lucide-react';
import type {
  AnnotationStroke,
  ContourStyle,
  DetectionObject,
  HumanCorrection,
  Lang,
  LowConfidenceSegment,
  ManualMeasurement,
  OccludedDisplayMode,
  OverlayShape,
  PromptPoint,
  RefinedBoundaryProposal,
  RoiRect,
  RunResult,
  Specimen,
  StepPreview,
  ToolMode,
  UnifiedRecipe,
} from '../types/vexel';
import {
  AUTO_CROP_STEP_ID,
  BUILTIN_RECIPES,
  computeSegmentedResultsBoundingBox,
  DEFAULT_INTENSITY_RECIPE_ID,
  RAW_SOURCE_STEP_ID,
  UNRECOGNIZED_MANUAL_RECIPE_ID,
} from '../engine/operators';
import { smoothPixelContour } from '../engine/contourTrace';
import { analyzeLineProfileLayers } from '../engine/curvedLayerEnhance';
import {
  pointInPolygon,
  refineSegmentBoundary,
  runPromptSegmentation,
  segmentObjectAtPoint,
  type SmartSegmentResult,
} from '../engine/mlRuntime';
import { snapManualLineToScaleBar } from '../engine/samples';
import { t } from '../i18n/strings';

interface ScientificCanvasProps {
  specimen: Specimen;
  result: RunResult | null;
  lang: Lang;
  darkMode?: boolean;
  activeRecipe: UnifiedRecipe;
  savedWorkflows: UnifiedRecipe[];
  importInspecting?: boolean;
  pendingCropDraft?: { cropPx: number; barPosition: 'bottom' | 'top' | 'none' };
  manualCalLine?: { p1: [number, number]; p2: [number, number]; lengthPx: number; snapped?: boolean } | null;
  onSetManualCalLine?: (line: { p1: [number, number]; p2: [number, number]; lengthPx: number; snapped?: boolean } | null) => void;
  onConfirmInfoBarCrop?: (cropPx: number, barPosition: 'bottom' | 'top' | 'none') => void;
  onConfirmCalibration?: () => void;
  onRunActiveWorkflow?: () => void;
  onSelectWorkflow: (workflowId: string) => void;
  onSaveCustomWorkflow: (customName: string) => void;
  onDeleteCustomWorkflow: (workflowId: string) => void;
  selectedStepId: string | null;
  selectedObjectId: number | null;
  onSelectObject: (id: number | null) => void;
  toolMode: ToolMode;
  onChangeToolMode: (mode: ToolMode) => void;
  roi: RoiRect | null;
  onChangeRoi: (roi: RoiRect | null) => void;
  showLowConfHighlight: boolean;
  onToggleLowConfHighlight: () => void;
  lowConfThreshold: number;
  occludedDisplayMode: OccludedDisplayMode;
  onChangeOccludedDisplayMode: (mode: OccludedDisplayMode) => void;
  dismissedLowConfIds: string[];
  onDismissLowConfSegment: (segId: string) => void;
  activeCorrectionSegId: string | null;
  onTriggerLowConfCorrection: (seg: LowConfidenceSegment | null) => void;
  onAddCorrection: (corr: Omit<HumanCorrection, 'id' | 'timestamp'>) => void;
  onApplyRefinedBoundary?: (proposal: RefinedBoundaryProposal) => void;
  strokes: AnnotationStroke[];
  onAddStroke: (stroke: AnnotationStroke) => void;
  onDeleteStroke: (id: string) => void;
  onClearStrokes: () => void;
  brushClassId: number;
  onChangeBrushClassId: (cid: number) => void;
  brushRadius: number;
  onChangeBrushRadius: (r: number) => void;
  prompts: PromptPoint[];
  onAddPrompt: (p: PromptPoint) => void;
  onDeletePrompt: (id: string) => void;
  onClearPrompts: () => void;
  measurements: ManualMeasurement[];
  onAddMeasurement: (m: ManualMeasurement) => void;
  onDeleteMeasurement: (id: string) => void;
  onClearMeasurements: () => void;
  onRegisterSnapshot?: (fn: () => string | undefined) => void;
}

const BRUSH_CLASS_COLORS: Record<number, { color: string; zh: string; en: string }> = {
  1: { color: '#06B6D4', zh: '目标前景', en: 'Foreground' },
  2: { color: '#F43F5E', zh: '背景基体', en: 'Background' },
  3: { color: '#F59E0B', zh: '缺陷/异物', en: 'Defect/Class 3' },
};

export function ScientificCanvas({
  specimen,
  result,
  lang,
  darkMode = true,
  activeRecipe,
  savedWorkflows,
  importInspecting = false,
  pendingCropDraft,
  manualCalLine = null,
  onSetManualCalLine,
  onConfirmInfoBarCrop,
  onConfirmCalibration,
  onRunActiveWorkflow,
  onSelectWorkflow,
  onSaveCustomWorkflow,
  onDeleteCustomWorkflow,
  selectedStepId,
  selectedObjectId,
  onSelectObject,
  toolMode,
  onChangeToolMode,
  roi,
  onChangeRoi,
  showLowConfHighlight,
  onToggleLowConfHighlight,
  lowConfThreshold,
  occludedDisplayMode,
  onChangeOccludedDisplayMode,
  dismissedLowConfIds,
  onDismissLowConfSegment,
  activeCorrectionSegId,
  onTriggerLowConfCorrection,
  onAddCorrection,
  onApplyRefinedBoundary,
  strokes,
  onAddStroke,
  onDeleteStroke,
  onClearStrokes,
  brushClassId,
  onChangeBrushClassId,
  brushRadius,
  onChangeBrushRadius,
  prompts,
  onAddPrompt,
  onDeletePrompt,
  onClearPrompts,
  measurements,
  onAddMeasurement,
  onDeleteMeasurement,
  onClearMeasurements,
  onRegisterSnapshot,
}: ScientificCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [contrast, setContrast] = useState(1);
  const [brightness, setBrightness] = useState(0);
  const [showFluo, setShowFluo] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showOverlayLabels, setShowOverlayLabels] = useState(false);
  const [showOverlayFill, setShowOverlayFill] = useState(false);
  const [contourStyle, setContourStyle] = useState<ContourStyle>('exact');
  const [overlayOpacity, setOverlayOpacity] = useState(0.9);
  const [showOverlayPanel, setShowOverlayPanel] = useState(false);
  const [showLutPanel, setShowLutPanel] = useState(false);
  const [showMarksPanel, setShowMarksPanel] = useState(false);
  const [saveWfOpen, setSaveWfOpen] = useState(false);
  const [customWfName, setCustomWfName] = useState('');

  const marksBtnRef = useRef<HTMLButtonElement>(null);
  const overlayBtnRef = useRef<HTMLButtonElement>(null);
  const lutBtnRef = useRef<HTMLButtonElement>(null);
  const marksPopoverRef = useRef<HTMLDivElement>(null);
  const overlayPopoverRef = useRef<HTMLDivElement>(null);
  const lutPopoverRef = useRef<HTMLDivElement>(null);
  const saveWfFormRef = useRef<HTMLFormElement>(null);
  const correctionBarRef = useRef<HTMLDivElement>(null);

  useClickOutside(
    showMarksPanel || showOverlayPanel || showLutPanel,
    [marksBtnRef, overlayBtnRef, lutBtnRef, marksPopoverRef, overlayPopoverRef, lutPopoverRef],
    () => {
      setShowMarksPanel(false);
      setShowOverlayPanel(false);
      setShowLutPanel(false);
    }
  );

  useClickOutside(saveWfOpen, [saveWfFormRef], () => setSaveWfOpen(false));

  useClickOutside(
    Boolean(activeCorrectionSegId),
    [correctionBarRef, containerRef],
    () => onTriggerLowConfCorrection(null)
  );

  const [cursorInfo, setCursorInfo] = useState<{ x: number; y: number; intensity: number } | null>(null);
  const [hoveredLowConfId, setHoveredLowConfId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ clientX: number; clientY: number; panX: number; panY: number } | null>(null);
  const [activeDragLine, setActiveDragLine] = useState<{ p1: [number, number]; p2: [number, number] } | null>(null);
  const [liveCalSnap, setLiveCalSnap] = useState<{ p1: [number, number]; p2: [number, number]; lengthPx: number } | null>(null);
  const [activeDragRoi, setActiveDragRoi] = useState<{ p1: [number, number]; p2: [number, number] } | null>(null);
  const activeStrokeRef = useRef<AnnotationStroke | null>(null);
  const [liveStroke, setLiveStroke] = useState<AnnotationStroke | null>(null);
  const [samSensitivity, setSamSensitivity] = useState(1.0);
  const samSensitivityRef = useRef<number>(1.0);
  const [sensWheelFlash, setSensWheelFlash] = useState(false);
  const sensFlashTimerRef = useRef<number | null>(null);
  const lastHoverImgPtRef = useRef<[number, number] | null>(null);
  const middleSensDragRef = useRef<{
    startY: number;
    startSens: number;
    anchorPt: [number, number];
    moved: boolean;
  } | null>(null);
  const [hoverSegPreview, setHoverSegPreview] = useState<SmartSegmentResult | null>(null);
  const lastHoverSegTimeRef = useRef<number>(0);
  const [hoveredRefineProposal, setHoveredRefineProposal] =
    useState<RefinedBoundaryProposal | null>(null);
  const [lockedRefineProposal, setLockedRefineProposal] =
    useState<RefinedBoundaryProposal | null>(null);
  const refineCacheRef = useRef<Map<string, RefinedBoundaryProposal | null>>(new Map());
  const activeRefineProposal = lockedRefineProposal ?? hoveredRefineProposal;

  useEffect(() => {
    samSensitivityRef.current = samSensitivity;
  }, [samSensitivity]);

  useEffect(() => {
    return () => {
      if (sensFlashTimerRef.current !== null) {
        window.clearTimeout(sensFlashTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (toolMode !== 'smart_gt' && toolMode !== 'measure_object') {
      setHoverSegPreview(null);
      lastHoverImgPtRef.current = null;
      middleSensDragRef.current = null;
    }
    if (toolMode !== 'boundary_refine') {
      setHoveredRefineProposal(null);
      setLockedRefineProposal(null);
    }
  }, [toolMode, specimen.id]);

  useEffect(() => {
    refineCacheRef.current.clear();
    setHoveredRefineProposal(null);
    setLockedRefineProposal(null);
  }, [specimen.id, result?.resultDigest]);

  const isRawSource = selectedStepId === RAW_SOURCE_STEP_ID;
  const isCropPreview = selectedStepId === AUTO_CROP_STEP_ID;

  const activePreview: StepPreview | null = useMemo(() => {
    if (!selectedStepId || isRawSource || !result) return null;
    return result.stepPreviews.find((p) => p.stepId === selectedStepId) ?? null;
  }, [selectedStepId, isRawSource, result]);

  const segmentedResultsRoi = useMemo(
    () => computeSegmentedResultsBoundingBox(result, specimen, 0.1),
    [result, specimen]
  );

  // Display dimensions:
  // - On "0. Original Image": show full uncropped original image (including info bar if present)
  // - On "1. Data Bar Crop" and downstream steps: show clean cropped working FOV
  const hasBuiltInBarCrop = Boolean(
    specimen.calibration.cropRows && specimen.calibration.cropRows < specimen.height
  );
  const viewWidth =
    isRawSource && specimen.rawRgba && specimen.rawWidth ? specimen.rawWidth : specimen.width;
  const viewHeight =
    isRawSource && specimen.rawRgba && specimen.rawHeight
      ? specimen.rawHeight
      : !isRawSource && hasBuiltInBarCrop
      ? specimen.calibration.cropRows!
      : specimen.height;

  // Grayscale buffer matching exact displayed viewWidth x viewHeight for magnetic scale-bar snapping
  const calibrationGray = useMemo(() => {
    if (isRawSource && specimen.rawRgba && specimen.rawWidth && specimen.rawHeight) {
      const len = specimen.rawWidth * specimen.rawHeight;
      const g = new Float32Array(len);
      const raw = specimen.rawRgba;
      for (let i = 0; i < len; i++) {
        g[i] = 0.299 * raw[i * 4] + 0.587 * raw[i * 4 + 1] + 0.114 * raw[i * 4 + 2];
      }
      return g;
    }
    return specimen.gray;
  }, [isRawSource, specimen.rawRgba, specimen.rawWidth, specimen.rawHeight, specimen.gray]);

  // Locate the selected fibre object and its fitted centerline segment for animated highlight
  const selectedFiberAxis = useMemo(() => {
    if (selectedObjectId === null || !result) return null;
    const obj = result.objects.find((o) => o.id === selectedObjectId && o.kind === 'fibre');
    if (!obj) return null;
    const axisShape = result.shapes.find(
      (s) => s.id === `fib_axis_${obj.id}` && s.points.length >= 2
    );
    if (!axisShape) return null;
    return {
      obj,
      start: axisShape.points[0] as [number, number],
      end: axisShape.points[axisShape.points.length - 1] as [number, number],
    };
  }, [selectedObjectId, result]);

  const fiberAnimStartRef = useRef<number>(0);
  useEffect(() => {
    if (selectedFiberAxis) {
      fiberAnimStartRef.current = performance.now();
    }
  }, [selectedFiberAxis?.obj.id, result?.resultDigest]);

  // Active processed grayscale image reflecting strict step-to-step image propagation
  const effectiveGray: Float32Array | undefined = useMemo(() => {
    if (isRawSource || isCropPreview || showFluo) return undefined;
    if (activePreview?.gray) return activePreview.gray;
    const previews = result?.stepPreviews;
    if (previews && previews.length > 0) {
      for (let i = previews.length - 1; i >= 0; i--) {
        if (previews[i].gray) return previews[i].gray;
      }
    }
    return result?.processedGray;
  }, [isRawSource, isCropPreview, showFluo, activePreview, result]);

  // Low-contrast segments (strictly exclude per-object rectangular boxes "目标 #xx"; only retain layer/blind-zone highlights)
  const visibleLowConfSegments = useMemo(() => {
    if (!result?.lowConfidenceSegments) return [];
    const dismissedSet = new Set(dismissedLowConfIds);
    return result.lowConfidenceSegments.filter(
      (seg) =>
        seg.kind !== 'object' &&
        seg.confidence < lowConfThreshold &&
        !dismissedSet.has(seg.id)
    );
  }, [result, lowConfThreshold, dismissedLowConfIds]);

  const activeCorrectionSeg = useMemo(() => {
    if (!activeCorrectionSegId) return null;
    return (
      visibleLowConfSegments.find((s) => s.id === activeCorrectionSegId) ??
      result?.lowConfidenceSegments?.find((s) => s.id === activeCorrectionSegId) ??
      null
    );
  }, [activeCorrectionSegId, visibleLowConfSegments, result]);

  const occludedCount = useMemo(
    () => result?.objects.filter((o) => !o.excluded && o.occluded).length ?? 0,
    [result]
  );

  // Run real SAM-style click-prompt segmentation on the processed image (scoped to ROI when active)
  const promptContours = useMemo(() => {
    if (prompts.length === 0) return [];
    return runPromptSegmentation(
      effectiveGray ?? specimen.gray,
      specimen.width,
      specimen.height,
      prompts,
      specimen.calibration.cropRows,
      roi,
      specimen.calibration.pixelSize,
      specimen.calibration.spaceUnit,
      samSensitivity
    );
  }, [specimen, effectiveGray, prompts, roi, samSensitivity]);

  // Cached SAM boundary refinement helper for any visible segmented shape
  const computeRefineForShape = (
    shape: OverlayShape,
    obj: DetectionObject | undefined
  ): RefinedBoundaryProposal | null => {
    const cacheKey = `${shape.id}_${samSensitivity.toFixed(1)}_${result?.resultDigest ?? ''}`;
    if (refineCacheRef.current.has(cacheKey)) {
      return refineCacheRef.current.get(cacheKey) ?? null;
    }
    const segGray = effectiveGray ?? (showFluo && specimen.fluoGray ? specimen.fluoGray : specimen.gray);
    const proposal = refineSegmentBoundary(segGray, specimen.width, specimen.height, shape, obj, {
      cropRows: specimen.calibration.cropRows,
      roi,
      pixelSize: specimen.calibration.pixelSize,
      spaceUnit: specimen.calibration.spaceUnit,
      sensitivity: samSensitivity,
    });
    refineCacheRef.current.set(cacheKey, proposal);
    return proposal;
  };

  // Hit-test visible, non-excluded segmented objects under cursor (smallest area wins)
  const findSegmentAtPoint = (
    ix: number,
    iy: number
  ): { shape: OverlayShape; obj: DetectionObject | undefined } | null => {
    if (!result) return null;
    const objMap = new Map<number, DetectionObject>();
    for (const o of result.objects) objMap.set(o.id, o);

    let best: { shape: OverlayShape; obj: DetectionObject | undefined; area: number } | null = null;
    for (const s of result.shapes) {
      if (s.objectId === undefined) continue;
      if (s.layer === 'roi' || s.layer === 'manual' || s.layer === 'flags') continue;
      if (s.kind !== 'polygon' && s.kind !== 'circle' && s.kind !== 'box') continue;
      const obj = objMap.get(s.objectId);
      if (obj?.excluded) continue;
      const isOccluded = Boolean(s.occluded || obj?.occluded);
      if (isOccluded && occludedDisplayMode === 'hide') continue;

      let hit = false;
      let area = Infinity;
      if (s.kind === 'polygon' && s.points.length >= 3) {
        if (pointInPolygon(ix, iy, s.points, s.holes)) {
          hit = true;
          let minX = Infinity,
            minY = Infinity,
            maxX = -Infinity,
            maxY = -Infinity;
          for (const [px, py] of s.points) {
            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;
          }
          area = Math.max(1, (maxX - minX) * (maxY - minY));
        }
      } else if (s.kind === 'circle' && s.points.length >= 2) {
        const [cx, cy] = s.points[0];
        const rx = Math.max(2, s.points[1][0]);
        if (Math.hypot(ix - cx, iy - cy) <= rx + 3 / zoom) {
          hit = true;
          area = Math.PI * rx * rx;
        }
      } else if (s.kind === 'box' && s.points.length >= 2) {
        const [x0, y0] = s.points[0];
        const [x1, y1] = s.points[1];
        const minX = Math.min(x0, x1);
        const maxX = Math.max(x0, x1);
        const minY = Math.min(y0, y1);
        const maxY = Math.max(y0, y1);
        if (ix >= minX && ix <= maxX && iy >= minY && iy <= maxY) {
          hit = true;
          area = Math.max(1, (maxX - minX) * (maxY - minY));
        }
      }

      if (hit && (!best || area < best.area)) {
        best = { shape: s, obj, area };
      }
    }
    if (best) return { shape: best.shape, obj: best.obj };

    // Fallback: centroid proximity for very small segments
    let nearest: { shape: OverlayShape; obj: DetectionObject; dist: number } | null = null;
    const maxDist = 14 / zoom;
    for (const o of result.objects) {
      if (o.excluded || (o.occluded && occludedDisplayMode === 'hide')) continue;
      const d = Math.hypot(o.x - ix, o.y - iy);
      if (d <= maxDist && (!nearest || d < nearest.dist)) {
        const shape = result.shapes.find(
          (s) =>
            s.objectId === o.id &&
            s.layer !== 'roi' &&
            s.layer !== 'manual' &&
            s.layer !== 'flags' &&
            (s.kind === 'polygon' || s.kind === 'circle' || s.kind === 'box')
        );
        if (shape) nearest = { shape, obj: o, dist: d };
      }
    }
    return nearest ? { shape: nearest.shape, obj: nearest.obj } : null;
  };

  const handleConfirmBoundaryRefine = (proposal: RefinedBoundaryProposal) => {
    onApplyRefinedBoundary?.(proposal);
    refineCacheRef.current.clear();
    setLockedRefineProposal(null);
    setHoveredRefineProposal(null);
  };

  // Live recomputation of locked segment preview when sensitivity slider changes
  useEffect(() => {
    if (toolMode !== 'boundary_refine' || !lockedRefineProposal || !result) return;
    const shape = result.shapes.find((s) => s.id === lockedRefineProposal.shapeId);
    if (!shape) return;
    const obj = result.objects.find((o) => o.id === lockedRefineProposal.objectId);
    const updated = computeRefineForShape(shape, obj);
    if (updated) setLockedRefineProposal(updated);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [samSensitivity]);

  // Keyboard confirmation (Enter = Apply, Escape = Cancel lock) in Boundary Refine mode
  useEffect(() => {
    if (toolMode !== 'boundary_refine') return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && lockedRefineProposal) {
        e.preventDefault();
        setLockedRefineProposal(null);
      } else if (e.key === 'Enter' && activeRefineProposal) {
        e.preventDefault();
        handleConfirmBoundaryRefine(activeRefineProposal);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toolMode, lockedRefineProposal, hoveredRefineProposal]);

  // Precompute lightly smoothed contours on demand so toggling Exact <-> Smooth is instant (0 ms)
  const smoothedShapeContours = useMemo(() => {
    if (contourStyle !== 'smooth' || !result) return null;
    const map = new Map<
      string,
      { points: Array<[number, number]>; holes?: Array<Array<[number, number]>> }
    >();
    for (const s of result.shapes) {
      if (s.kind === 'polygon' && s.points.length >= 3 && s.layer !== 'roi' && !s.id.startsWith('ml_')) {
        map.set(s.id, {
          points: smoothPixelContour(s.points),
          holes: s.holes?.map((h) => smoothPixelContour(h)),
        });
      }
    }
    return map;
  }, [contourStyle, result]);

  const displayedPromptContours = useMemo(() => {
    if (contourStyle !== 'smooth') return promptContours;
    return promptContours.map((p) => ({
      ...p,
      points: smoothPixelContour(p.points),
      holes: p.holes.map((h) => smoothPixelContour(h)),
    }));
  }, [contourStyle, promptContours]);

  // Fit specimen into viewport whenever specimen changes
  const fitToView = () => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pad = 36;
    const scaleX = (rect.width - pad * 2) / viewWidth;
    const scaleY = (rect.height - pad * 2) / viewHeight;
    const z = Math.max(0.25, Math.min(4, Math.min(scaleX, scaleY)));
    setZoom(Number(z.toFixed(3)));
    setPan({ x: 0, y: 0 });
  };

  useEffect(() => {
    setShowFluo(false);
    const id = requestAnimationFrame(() => fitToView());
    return () => cancelAnimationFrame(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [specimen.id]);

  // Build offscreen base image with LUT brightness/contrast + step gray / probability heatmap
  const baseImageCanvas = useMemo(() => {
    const off = document.createElement('canvas');
    off.width = viewWidth;
    off.height = viewHeight;
    const ctx = off.getContext('2d')!;
    const out = ctx.createImageData(viewWidth, viewHeight);
    const dst = out.data;

    const useRawUpload = isRawSource && Boolean(specimen.rawRgba && specimen.rawWidth && specimen.rawHeight);
    const previewGray = !isRawSource && !isCropPreview && !showFluo ? effectiveGray : undefined;
    const probMap = !isRawSource && !isCropPreview ? activePreview?.probMap : undefined;
    const srcRgba = useRawUpload
      ? specimen.rawRgba!
      : showFluo && specimen.fluoRgba
      ? specimen.fluoRgba
      : specimen.rgba;

    const n = viewWidth * viewHeight;
    for (let i = 0; i < n; i++) {
      let r: number;
      let g: number;
      let b: number;
      if (previewGray) {
        const v = previewGray[i];
        r = v;
        g = v;
        b = v;
      } else {
        r = srcRgba[i * 4];
        g = srcRgba[i * 4 + 1];
        b = srcRgba[i * 4 + 2];
      }

      // Apply linear LUT windowing (brightness/contrast)
      r = (r - 128) * contrast + 128 + brightness;
      g = (g - 128) * contrast + 128 + brightness;
      b = (b - 128) * contrast + 128 + brightness;

      // Blend probability heatmap if previewing a Trained Model step
      if (probMap) {
        const p = probMap[i];
        if (p > 0.15) {
          const alpha = Math.min(0.68, (p - 0.15) * 0.85);
          // Cyan-to-Amber scientific probability ramp
          const hr = p > 0.65 ? 245 : 6;
          const hg = p > 0.65 ? 158 : 182;
          const hb = p > 0.65 ? 11 : 212;
          r = r * (1 - alpha) + hr * alpha;
          g = g * (1 - alpha) + hg * alpha;
          b = b * (1 - alpha) + hb * alpha;
        }
      }

      dst[i * 4] = r < 0 ? 0 : r > 255 ? 255 : r;
      dst[i * 4 + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
      dst[i * 4 + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
      dst[i * 4 + 3] = 255;
    }

    ctx.putImageData(out, 0, 0);
    return off;
  }, [specimen, effectiveGray, activePreview, isRawSource, isCropPreview, viewWidth, viewHeight, showFluo, contrast, brightness]);

  // Helper: convert mouse event to specimen pixel coordinates
  const clientToImage = (clientX: number, clientY: number): [number, number] | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;
    const originX = rect.width / 2 + pan.x - (viewWidth * zoom) / 2;
    const originY = rect.height / 2 + pan.y - (viewHeight * zoom) / 2;
    const ix = (cx - originX) / zoom;
    const iy = (cy - originY) / zoom;
    return [ix, iy];
  };

  // Render full interactive Canvas2D frame
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    let rafId = 0;

    const render = (now = performance.now()) => {
      const rect = container.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cw = Math.max(1, Math.floor(rect.width));
      const ch = Math.max(1, Math.floor(rect.height));
      if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
        canvas.width = cw * dpr;
        canvas.height = ch * dpr;
      }
      const ctx = canvas.getContext('2d')!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Theme-matched laboratory stage backdrop (daylight neutral lab vs obsidian dark)
      ctx.fillStyle = darkMode ? '#070A12' : '#EEF2F6';
      ctx.fillRect(0, 0, cw, ch);

      // Subtle coordinate reference grid
      ctx.strokeStyle = darkMode ? 'rgba(148, 163, 184, 0.05)' : 'rgba(15, 23, 42, 0.055)';
      ctx.lineWidth = 1;
      for (let gx = 0; gx < cw; gx += 32) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, ch);
        ctx.stroke();
      }
      for (let gy = 0; gy < ch; gy += 32) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(cw, gy);
        ctx.stroke();
      }

      const originX = cw / 2 + pan.x - (viewWidth * zoom) / 2;
      const originY = ch / 2 + pan.y - (viewHeight * zoom) / 2;

      ctx.save();
      ctx.translate(originX, originY);
      ctx.scale(zoom, zoom);

      if (!darkMode) {
        ctx.save();
        ctx.shadowColor = 'rgba(15, 23, 42, 0.14)';
        ctx.shadowBlur = 16;
        ctx.shadowOffsetY = 3;
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, viewWidth, viewHeight);
        ctx.restore();
      }

      // Crisp pixel rendering when zoomed in
      ctx.imageSmoothingEnabled = zoom < 2;
      ctx.drawImage(baseImageCanvas, 0, 0);

      // Subtle border around active frame
      ctx.strokeStyle = darkMode ? 'rgba(148, 163, 184, 0.28)' : 'rgba(15, 23, 42, 0.22)';
      ctx.lineWidth = 1 / zoom;
      ctx.strokeRect(0, 0, viewWidth, viewHeight);

      // On "0. Original Image", highlight the exact info-bar crop cut line
      if (isRawSource) {
        let cutY: number | null = null;
        let cutPx = 0;
        if (specimen.rawRgba && specimen.rawHeight && specimen.rawCroppedBarPx) {
          cutPx = specimen.rawCroppedBarPx;
          cutY =
            specimen.rawBarPosition === 'top'
              ? specimen.rawCroppedBarPx
              : specimen.rawHeight - specimen.rawCroppedBarPx;
        } else if (hasBuiltInBarCrop && specimen.calibration.cropRows) {
          cutY = specimen.calibration.cropRows;
          cutPx = specimen.height - specimen.calibration.cropRows;
        }
        if (cutY !== null && cutPx > 0) {
          ctx.save();
          ctx.fillStyle = 'rgba(245, 158, 11, 0.14)';
          if (specimen.rawBarPosition === 'top') {
            ctx.fillRect(0, 0, viewWidth, cutY);
          } else {
            ctx.fillRect(0, cutY, viewWidth, viewHeight - cutY);
          }
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.9)';
          ctx.lineWidth = 1.6 / zoom;
          ctx.setLineDash([6 / zoom, 4 / zoom]);
          ctx.beginPath();
          ctx.moveTo(0, cutY);
          ctx.lineTo(viewWidth, cutY);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.font = `600 ${Math.max(10, 11 / zoom)}px "JetBrains Mono", monospace`;
          ctx.fillStyle = '#FBBF24';
          ctx.fillText(
            lang === 'zh'
              ? `✂ 自动裁切信息栏 (${cutPx}px)`
              : `✂ Auto-Crop Info Bar (${cutPx}px)`,
            10 / zoom,
            Math.max(14 / zoom, cutY - 6 / zoom)
          );
          ctx.restore();
        }
      }

      // 0b. Semi-auto pending info-bar crop outline (auto-circled on canvas, awaiting user confirmation!)
      if (
        !isRawSource &&
        specimen.importReport?.cropStatus === 'pending' &&
        pendingCropDraft &&
        pendingCropDraft.cropPx > 0 &&
        pendingCropDraft.barPosition !== 'none'
      ) {
        const cropH = Math.min(viewHeight - 24, Math.max(0, pendingCropDraft.cropPx));
        if (cropH > 0) {
          const isTop = pendingCropDraft.barPosition === 'top';
          const boxY = isTop ? 0 : viewHeight - cropH;
          const cutY = isTop ? cropH : viewHeight - cropH;
          ctx.save();
          // Keep tint subtle while user is manually tracing the scale bar inside the info bar
          ctx.fillStyle =
            toolMode === 'calibrate_line'
              ? 'rgba(245, 158, 11, 0.05)'
              : 'rgba(245, 158, 11, 0.14)';
          ctx.fillRect(0, boxY, viewWidth, cropH);

          // Dashed amber bounding frame outlining the detected info bar
          ctx.strokeStyle =
            toolMode === 'calibrate_line'
              ? 'rgba(245, 158, 11, 0.55)'
              : 'rgba(245, 158, 11, 0.92)';
          ctx.lineWidth = 1.8 / zoom;
          ctx.setLineDash([6 / zoom, 4 / zoom]);
          ctx.strokeRect(1 / zoom, boxY + 1 / zoom, viewWidth - 2 / zoom, cropH - 2 / zoom);

          // Stronger horizontal cut line
          ctx.strokeStyle = '#FBBF24';
          ctx.lineWidth = 2.1 / zoom;
          ctx.beginPath();
          ctx.moveTo(0, cutY);
          ctx.lineTo(viewWidth, cutY);
          ctx.stroke();
          ctx.setLineDash([]);

          // High-contrast micro-badge above/below cut line
          const labelText =
            lang === 'zh'
              ? `✂ 信息栏区域 (${isTop ? '顶部' : '底部'} ${cropH}px) · 建议先标定比例尺再裁切`
              : `✂ Info Bar (${pendingCropDraft.barPosition} ${cropH}px) · Calibrate Scale Bar Before Cropping`;
          ctx.font = `600 ${Math.max(10, 11 / zoom)}px "JetBrains Mono", monospace`;
          const tw = ctx.measureText(labelText).width;
          const padX = 6 / zoom;
          const badgeH = 16 / zoom;
          const badgeY = isTop
            ? Math.min(viewHeight - badgeH - 4 / zoom, cutY + 5 / zoom)
            : Math.max(4 / zoom, cutY - badgeH - 5 / zoom);
          ctx.fillStyle = 'rgba(15, 12, 6, 0.88)';
          ctx.fillRect(8 / zoom, badgeY, tw + padX * 2, badgeH);
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
          ctx.lineWidth = 1 / zoom;
          ctx.strokeRect(8 / zoom, badgeY, tw + padX * 2, badgeH);
          ctx.fillStyle = '#FDE68A';
          ctx.fillText(labelText, 8 / zoom + padX, badgeY + 11.5 / zoom);
          ctx.restore();
        }
      }

      // 1. Quantitative analysis overlay shapes (hidden on pure "0. Original Image" and "1. Data Bar Crop" inspection views)
      if (showOverlay && result && !isRawSource && !isCropPreview) {
        const excludedIds = new Set(result.objects.filter((o) => o.excluded).map((o) => o.id));
        const occludedIds = new Set(result.objects.filter((o) => o.occluded).map((o) => o.id));
        const hasFiberHighlight = Boolean(selectedFiberAxis);
        ctx.save();
        ctx.globalAlpha = overlayOpacity;
        for (const s of result.shapes) {
          const isExcluded = s.objectId !== undefined && excludedIds.has(s.objectId);
          const isSelected = s.objectId !== undefined && s.objectId === selectedObjectId;
          const isOccludedShape = Boolean(
            s.occluded || (s.objectId !== undefined && occludedIds.has(s.objectId))
          );

          // Option: completely hide partially occluded particles when user selects 'hide'
          if (isOccludedShape && occludedDisplayMode === 'hide') {
            continue;
          }

          // Skip static centerline for animated selected fiber (drawn with luminous path animation below)
          if (hasFiberHighlight && isSelected && s.id === `fib_axis_${selectedObjectId}`) {
            continue;
          }

          let shapeColor = s.color;
          let shapeFill = s.fill;
          let shapeDashed = Boolean(s.dashed);
          if (isOccludedShape) {
            if (occludedDisplayMode === 'highlight') {
              shapeColor = '#C084FC';
              shapeFill = '#C084FC28';
              shapeDashed = true;
            } else if (occludedDisplayMode === 'normal') {
              shapeColor = '#10B981';
              shapeFill = '#10B98122';
              shapeDashed = false;
            }
          }

          ctx.save();
          if (isExcluded) {
            ctx.globalAlpha = overlayOpacity * 0.25;
          } else if (hasFiberHighlight && !isSelected) {
            ctx.globalAlpha = overlayOpacity * 0.34;
          }
          ctx.strokeStyle = isSelected
            ? s.id.startsWith('fib_perp_')
              ? '#22D3EE'
              : '#FFFFFF'
            : shapeColor;
          ctx.fillStyle = showOverlayFill
            ? shapeFill ?? 'transparent'
            : isSelected
            ? 'rgba(255, 255, 255, 0.08)'
            : 'transparent';
          ctx.lineWidth = (isSelected ? 2.4 : 1.5) / zoom;
          if (shapeDashed && !isSelected) ctx.setLineDash([5 / zoom, 3 / zoom]);

          if (s.kind === 'circle' && s.points.length >= 2) {
            const [cx, cy] = s.points[0];
            const [rx] = s.points[1];
            ctx.beginPath();
            ctx.arc(cx, cy, Math.max(1.5, rx), 0, Math.PI * 2);
            if ((showOverlayFill && shapeFill) || isSelected) ctx.fill();
            ctx.stroke();
          } else if (s.kind === 'polygon' && s.points.length >= 3) {
            const smoothed = smoothedShapeContours?.get(s.id);
            const outerPts = smoothed ? smoothed.points : s.points;
            const holeLoops = smoothed ? smoothed.holes : s.holes;
            ctx.lineJoin = contourStyle === 'exact' ? 'miter' : 'round';
            ctx.miterLimit = 2;
            ctx.beginPath();
            ctx.moveTo(outerPts[0][0], outerPts[0][1]);
            for (let i = 1; i < outerPts.length; i++) ctx.lineTo(outerPts[i][0], outerPts[i][1]);
            ctx.closePath();
            if (holeLoops && holeLoops.length > 0) {
              for (const hole of holeLoops) {
                if (hole.length < 3) continue;
                ctx.moveTo(hole[0][0], hole[0][1]);
                for (let i = 1; i < hole.length; i++) ctx.lineTo(hole[i][0], hole[i][1]);
                ctx.closePath();
              }
            }
            if ((showOverlayFill && shapeFill) || isSelected) ctx.fill('evenodd');
            ctx.stroke();
          } else if (s.kind === 'polyline' && s.points.length >= 2) {
            ctx.beginPath();
            ctx.moveTo(s.points[0][0], s.points[0][1]);
            for (let i = 1; i < s.points.length; i++) ctx.lineTo(s.points[i][0], s.points[i][1]);
            ctx.stroke();
          } else if (s.kind === 'point' && s.points.length >= 1) {
            const [px, py] = s.points[0];
            ctx.fillStyle = isSelected ? '#FFFFFF' : shapeColor;
            ctx.beginPath();
            ctx.arc(px, py, (isSelected ? 3.3 : 2.8) / zoom, 0, Math.PI * 2);
            ctx.fill();
          } else if (s.kind === 'box' && s.points.length >= 2) {
            const [x0, y0] = s.points[0];
            const [x1, y1] = s.points[1];
            ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
          }

          // Text labels are hidden by default to avoid clutter on dense particles; manually enabled via Overlay options
          if (showOverlayLabels && s.label && (isSelected || zoom >= 0.45)) {
            const [lx, ly] = s.points[0];
            ctx.font = `600 ${Math.max(9, 10.5 / zoom)}px "JetBrains Mono", monospace`;
            ctx.fillStyle = isSelected ? '#FFFFFF' : shapeColor;
            ctx.fillText(s.label, lx + 4 / zoom, ly - 4 / zoom);
          }
          ctx.restore();
        }

        // 1b. Animated centerline trace highlight when a specific fiber is selected
        if (selectedFiberAxis) {
          const { obj, start, end } = selectedFiberAxis;
          const elapsed = Math.max(0, now - fiberAnimStartRef.current);
          const drawDuration = 680;
          const rawT = Math.min(1, elapsed / drawDuration);
          // Smooth cubic ease-out path drawing along fiber centerline
          const progress = 1 - Math.pow(1 - rawT, 3);
          const curX = start[0] + (end[0] - start[0]) * progress;
          const curY = start[1] + (end[1] - start[1]) * progress;
          const pulse = 0.5 + 0.5 * Math.sin(elapsed * 0.0055);

          ctx.save();
          ctx.globalAlpha = 1;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';

          // Soft ambient outer halo along traced centerline
          ctx.strokeStyle = `rgba(34, 211, 238, ${(0.24 + 0.14 * pulse).toFixed(3)})`;
          ctx.lineWidth = (10 + 2 * pulse) / zoom;
          ctx.beginPath();
          ctx.moveTo(start[0], start[1]);
          ctx.lineTo(curX, curY);
          ctx.stroke();

          // Warm golden-amber inner glow
          ctx.strokeStyle = 'rgba(250, 204, 21, 0.88)';
          ctx.lineWidth = 4.2 / zoom;
          ctx.beginPath();
          ctx.moveTo(start[0], start[1]);
          ctx.lineTo(curX, curY);
          ctx.stroke();

          // Crisp luminous white centerline core
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 1.9 / zoom;
          ctx.beginPath();
          ctx.moveTo(start[0], start[1]);
          ctx.lineTo(curX, curY);
          ctx.stroke();

          // Animated marching energy flow dashes along centerline direction
          ctx.save();
          ctx.strokeStyle = '#22D3EE';
          ctx.lineWidth = 2.0 / zoom;
          ctx.setLineDash([8 / zoom, 10 / zoom]);
          ctx.lineDashOffset = -(elapsed * 0.04) / zoom;
          ctx.beginPath();
          ctx.moveTo(start[0], start[1]);
          ctx.lineTo(curX, curY);
          ctx.stroke();
          ctx.restore();

          // Pulsing start endpoint beacon
          ctx.strokeStyle = `rgba(34, 211, 238, ${(0.45 + 0.45 * (1 - pulse)).toFixed(3)})`;
          ctx.lineWidth = 1.6 / zoom;
          ctx.beginPath();
          ctx.arc(start[0], start[1], (4.5 + 4.5 * pulse) / zoom, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = '#FDE047';
          ctx.beginPath();
          ctx.arc(start[0], start[1], 3.2 / zoom, 0, Math.PI * 2);
          ctx.fill();

          // End endpoint beacon (appears once draw-in reaches the end)
          if (rawT >= 0.98) {
            ctx.strokeStyle = `rgba(34, 211, 238, ${(0.45 + 0.45 * (1 - pulse)).toFixed(3)})`;
            ctx.lineWidth = 1.6 / zoom;
            ctx.beginPath();
            ctx.arc(end[0], end[1], (4.5 + 4.5 * pulse) / zoom, 0, Math.PI * 2);
            ctx.stroke();
            ctx.fillStyle = '#FDE047';
            ctx.beginPath();
            ctx.arc(end[0], end[1], 3.2 / zoom, 0, Math.PI * 2);
            ctx.fill();
          }

          // Traveling luminous scan bead gliding along centerline (leading comet during draw-in, then smooth scan)
          const beadT =
            rawT < 1
              ? progress
              : 0.5 - 0.5 * Math.cos(((elapsed - drawDuration) / 2200) * Math.PI * 2);
          const bx = start[0] + (end[0] - start[0]) * beadT;
          const by = start[1] + (end[1] - start[1]) * beadT;
          ctx.fillStyle = 'rgba(34, 211, 238, 0.36)';
          ctx.beginPath();
          ctx.arc(bx, by, 7.5 / zoom, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = '#FFFFFF';
          ctx.beginPath();
          ctx.arc(bx, by, 3.0 / zoom, 0, Math.PI * 2);
          ctx.fill();

          // High-contrast floating centerline telemetry badge at fiber midpoint
          const midX = (start[0] + end[0]) * 0.5;
          const midY = (start[1] + end[1]) * 0.5;
          const badgeText = `F#${obj.id} · ${obj.diameter?.toFixed(1) ?? '—'} ${specimen.calibration.spaceUnit} · ${obj.orientationDeg?.toFixed(1) ?? '0'}°`;
          ctx.font = `600 ${11 / zoom}px "JetBrains Mono", monospace`;
          const tw = ctx.measureText(badgeText).width;
          const padX = 6 / zoom;
          const boxH = 17 / zoom;
          const boxX = midX - tw / 2 - padX;
          const boxY = midY - boxH - 10 / zoom;
          ctx.fillStyle = 'rgba(7, 10, 18, 0.9)';
          ctx.strokeStyle = 'rgba(34, 211, 238, 0.85)';
          ctx.lineWidth = 1.2 / zoom;
          ctx.fillRect(boxX, boxY, tw + padX * 2, boxH);
          ctx.strokeRect(boxX, boxY, tw + padX * 2, boxH);
          ctx.fillStyle = '#FDE047';
          ctx.fillText(badgeText, boxX + padX, boxY + 12.2 / zoom);

          ctx.restore();
        }

        // 1c. Automatic subtle red highlight boxes for low-contrast / low-confidence segments
        if (showLowConfHighlight && visibleLowConfSegments.length > 0) {
          for (const seg of visibleLowConfSegments) {
            const { x, y, width: bw, height: bh } = seg.bbox;
            const isHovered = hoveredLowConfId === seg.id;
            const isActive = activeCorrectionSegId === seg.id;

            ctx.save();
            // Subtle translucent rose tint
            ctx.fillStyle = isActive
              ? 'rgba(244, 63, 94, 0.18)'
              : isHovered
              ? 'rgba(244, 63, 94, 0.14)'
              : 'rgba(244, 63, 94, 0.075)';
            ctx.fillRect(x, y, bw, bh);

            // Subtle dashed rose border (solid & slightly brighter when hovered or active)
            ctx.strokeStyle = isActive
              ? 'rgba(251, 113, 133, 0.95)'
              : isHovered
              ? 'rgba(251, 113, 133, 0.85)'
              : 'rgba(244, 63, 94, 0.58)';
            ctx.lineWidth = (isActive ? 2.0 : isHovered ? 1.7 : 1.25) / zoom;
            if (!isActive) ctx.setLineDash([4.5 / zoom, 3 / zoom]);
            ctx.strokeRect(x, y, bw, bh);
            ctx.setLineDash([]);

            // Subtle corner brackets
            const tk = Math.min(8 / zoom, bw * 0.25, bh * 0.25);
            ctx.strokeStyle = isActive || isHovered ? 'rgba(253, 164, 175, 0.95)' : 'rgba(244, 63, 94, 0.78)';
            ctx.lineWidth = (isActive ? 2.0 : 1.5) / zoom;
            ctx.beginPath();
            ctx.moveTo(x, y + tk); ctx.lineTo(x, y); ctx.lineTo(x + tk, y);
            ctx.moveTo(x + bw - tk, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + tk);
            ctx.moveTo(x, y + bh - tk); ctx.lineTo(x, y + bh); ctx.lineTo(x + tk, y + bh);
            ctx.moveTo(x + bw - tk, y + bh); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw, y + bh - tk);
            ctx.stroke();

            // Clean hover-only label without percentage confidence clutter
            if (isHovered || isActive) {
              const tag = `${lang === 'zh' ? seg.labelZh : seg.labelEn} (${lang === 'zh' ? '点击纠偏' : 'Click to fix'})`;
              ctx.font = `600 ${Math.max(8.5, 9.5 / zoom)}px "JetBrains Mono", monospace`;
              const tw = ctx.measureText(tag).width;
              const padX = 4 / zoom;
              const tagH = 13 / zoom;
              const tagY = y > tagH + 2 / zoom ? y - tagH - 1.5 / zoom : y + 1.5 / zoom;
              ctx.fillStyle = isActive ? 'rgba(136, 19, 55, 0.92)' : 'rgba(15, 10, 16, 0.78)';
              ctx.fillRect(x, tagY, tw + padX * 2, tagH);
              ctx.fillStyle = '#FECDD3';
              ctx.fillText(tag, x + padX, tagY + 9.8 / zoom);
            }
            ctx.restore();
          }
        }
        ctx.restore();
      }

      // 2. SAM-style prompt contours & points
      for (const poly of displayedPromptContours) {
        ctx.save();
        ctx.lineJoin = contourStyle === 'exact' ? 'miter' : 'round';
        ctx.miterLimit = 2;
        ctx.beginPath();
        ctx.moveTo(poly.points[0][0], poly.points[0][1]);
        for (let i = 1; i < poly.points.length; i++) ctx.lineTo(poly.points[i][0], poly.points[i][1]);
        ctx.closePath();
        for (const hole of poly.holes) {
          if (hole.length < 3) continue;
          ctx.moveTo(hole[0][0], hole[0][1]);
          for (let i = 1; i < hole.length; i++) ctx.lineTo(hole[i][0], hole[i][1]);
          ctx.closePath();
        }
        ctx.fillStyle = 'rgba(16, 185, 129, 0.22)';
        ctx.fill('evenodd');
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 1.8 / zoom;
        ctx.stroke();
        const areaPhys = poly.areaPx * specimen.calibration.pixelSize * specimen.calibration.pixelSize;
        ctx.font = `600 ${10 / zoom}px "JetBrains Mono", monospace`;
        ctx.fillStyle = '#6EE7B7';
        ctx.fillText(
          `SAM ${areaPhys.toFixed(1)} ${specimen.calibration.spaceUnit}²`,
          poly.cx + 5 / zoom,
          poly.cy - 4 / zoom
        );
        ctx.restore();
      }
      for (const pt of prompts) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4.5 / zoom, 0, Math.PI * 2);
        ctx.fillStyle = pt.positive ? '#10B981' : '#F43F5E';
        ctx.fill();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.2 / zoom;
        ctx.stroke();
        ctx.restore();
      }

      // 3. Active Learning GT annotations (both SAM-click filled polygons and freehand brush strokes)
      const allStrokes = liveStroke ? [...strokes, liveStroke] : strokes;
      for (const st of allStrokes) {
        const meta = BRUSH_CLASS_COLORS[st.classId] ?? BRUSH_CLASS_COLORS[1];
        if (st.polygon && st.polygon.length >= 3) {
          ctx.save();
          ctx.lineJoin = contourStyle === 'exact' ? 'miter' : 'round';
          ctx.miterLimit = 2;
          ctx.beginPath();
          ctx.moveTo(st.polygon[0][0], st.polygon[0][1]);
          for (let i = 1; i < st.polygon.length; i++) ctx.lineTo(st.polygon[i][0], st.polygon[i][1]);
          ctx.closePath();
          if (st.holes) {
            for (const hole of st.holes) {
              if (hole.length < 3) continue;
              ctx.moveTo(hole[0][0], hole[0][1]);
              for (let i = 1; i < hole.length; i++) ctx.lineTo(hole[i][0], hole[i][1]);
              ctx.closePath();
            }
          }
          ctx.fillStyle = meta.color;
          ctx.globalAlpha = 0.28;
          ctx.fill('evenodd');
          ctx.globalAlpha = 0.92;
          ctx.strokeStyle = meta.color;
          ctx.lineWidth = 1.6 / zoom;
          ctx.stroke();
          ctx.restore();
          continue;
        }
        if (st.points.length === 0) continue;
        ctx.save();
        ctx.strokeStyle = meta.color;
        ctx.fillStyle = meta.color;
        ctx.globalAlpha = 0.55;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = st.radius * 2;
        if (st.points.length === 1) {
          ctx.beginPath();
          ctx.arc(st.points[0][0], st.points[0][1], st.radius, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.beginPath();
          ctx.moveTo(st.points[0][0], st.points[0][1]);
          for (let i = 1; i < st.points.length; i++) ctx.lineTo(st.points[i][0], st.points[i][1]);
          ctx.stroke();
        }
        ctx.restore();
      }

      // 3b. Live SAM Hover Preview (for Smart Click GT & Smart Click Measure)
      if (hoverSegPreview && (toolMode === 'smart_gt' || toolMode === 'measure_object')) {
        const previewPts =
          contourStyle === 'smooth' ? smoothPixelContour(hoverSegPreview.points) : hoverSegPreview.points;
        const previewColor =
          toolMode === 'smart_gt'
            ? (BRUSH_CLASS_COLORS[brushClassId] ?? BRUSH_CLASS_COLORS[1]).color
            : '#22D3EE';
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(previewPts[0][0], previewPts[0][1]);
        for (let i = 1; i < previewPts.length; i++) ctx.lineTo(previewPts[i][0], previewPts[i][1]);
        ctx.closePath();
        for (const hole of hoverSegPreview.holes) {
          if (hole.length < 3) continue;
          ctx.moveTo(hole[0][0], hole[0][1]);
          for (let i = 1; i < hole.length; i++) ctx.lineTo(hole[i][0], hole[i][1]);
          ctx.closePath();
        }
        ctx.fillStyle = previewColor;
        ctx.globalAlpha = 0.2;
        ctx.fill('evenodd');
        ctx.globalAlpha = 0.95;
        ctx.strokeStyle = previewColor;
        ctx.lineWidth = 1.8 / zoom;
        ctx.setLineDash([5 / zoom, 3 / zoom]);
        ctx.stroke();
        ctx.setLineDash([]);

        if (toolMode === 'measure_object') {
          const { majorAxis, minorAxis, lengthPhys, widthPhys, areaPhys, areaUnit } =
            hoverSegPreview.metrics;
          // Preview principal Length & Width axes
          ctx.strokeStyle = 'rgba(251, 191, 36, 0.92)';
          ctx.lineWidth = 1.4 / zoom;
          ctx.beginPath();
          ctx.moveTo(majorAxis[0][0], majorAxis[0][1]);
          ctx.lineTo(majorAxis[1][0], majorAxis[1][1]);
          ctx.stroke();
          ctx.strokeStyle = 'rgba(34, 211, 238, 0.92)';
          ctx.setLineDash([3 / zoom, 2.5 / zoom]);
          ctx.beginPath();
          ctx.moveTo(minorAxis[0][0], minorAxis[0][1]);
          ctx.lineTo(minorAxis[1][0], minorAxis[1][1]);
          ctx.stroke();
          ctx.setLineDash([]);

          const u = specimen.calibration.spaceUnit;
          const sensSuffix =
            sensWheelFlash || Math.abs(samSensitivity - 1.0) > 0.01
              ? ` · ${lang === 'zh' ? '阈值' : 'T'} ${samSensitivity.toFixed(2)}`
              : '';
          const tag = `${lengthPhys.toFixed(1)}×${widthPhys.toFixed(1)} ${u} · ${areaPhys.toFixed(1)} ${areaUnit}${sensSuffix}`;
          ctx.font = `600 ${Math.max(9.5, 10.5 / zoom)}px "JetBrains Mono", monospace`;
          const tw = ctx.measureText(tag).width;
          const lx = Math.max(4 / zoom, Math.min(viewWidth - tw - 12 / zoom, hoverSegPreview.cx - tw * 0.5));
          const ly = Math.max(18 / zoom, hoverSegPreview.metrics.bbox.y - 6 / zoom);
          ctx.fillStyle = 'rgba(7, 10, 18, 0.9)';
          ctx.fillRect(lx - 4 / zoom, ly - 12 / zoom, tw + 8 / zoom, 15 / zoom);
          ctx.strokeStyle = sensWheelFlash ? 'rgba(251, 191, 36, 0.95)' : 'rgba(34, 211, 238, 0.7)';
          ctx.lineWidth = (sensWheelFlash ? 1.4 : 1) / zoom;
          ctx.strokeRect(lx - 4 / zoom, ly - 12 / zoom, tw + 8 / zoom, 15 / zoom);
          ctx.fillStyle = sensWheelFlash ? '#FEF3C7' : '#E0F2FE';
          ctx.fillText(tag, lx, ly - 1.5 / zoom);
        } else {
          const { areaPhys, areaUnit } = hoverSegPreview.metrics;
          const meta = BRUSH_CLASS_COLORS[brushClassId] ?? BRUSH_CLASS_COLORS[1];
          const sensSuffix =
            sensWheelFlash || Math.abs(samSensitivity - 1.0) > 0.01
              ? ` · ${lang === 'zh' ? '阈值' : 'T'} ${samSensitivity.toFixed(2)}`
              : '';
          const tag = `+ ${lang === 'zh' ? meta.zh : meta.en} · ${areaPhys.toFixed(1)} ${areaUnit}${sensSuffix}`;
          ctx.font = `600 ${Math.max(9.5, 10 / zoom)}px "JetBrains Mono", monospace`;
          const tw = ctx.measureText(tag).width;
          const lx = Math.max(4 / zoom, Math.min(viewWidth - tw - 12 / zoom, hoverSegPreview.cx - tw * 0.5));
          const ly = Math.max(18 / zoom, hoverSegPreview.metrics.bbox.y - 6 / zoom);
          ctx.fillStyle = 'rgba(7, 10, 18, 0.9)';
          ctx.fillRect(lx - 4 / zoom, ly - 12 / zoom, tw + 8 / zoom, 15 / zoom);
          ctx.strokeStyle = sensWheelFlash ? '#F59E0B' : previewColor;
          ctx.lineWidth = (sensWheelFlash ? 1.4 : 1) / zoom;
          ctx.strokeRect(lx - 4 / zoom, ly - 12 / zoom, tw + 8 / zoom, 15 / zoom);
          ctx.fillStyle = sensWheelFlash ? '#FEF3C7' : '#F8FAFC';
          ctx.fillText(tag, lx, ly - 1.5 / zoom);
        }
        ctx.restore();
      }

      // 3c. SAM Boundary Refine Preview (Original coarse boundary vs SAM-refined boundary)
      if (
        toolMode === 'boundary_refine' &&
        activeRefineProposal &&
        !isRawSource &&
        !isCropPreview
      ) {
        const prop = activeRefineProposal;
        const isLocked = Boolean(lockedRefineProposal);
        const origPts = prop.originalPoints;
        const refinedPts =
          contourStyle === 'smooth'
            ? smoothPixelContour(prop.refinedPoints)
            : prop.refinedPoints;

        ctx.save();
        // Original coarse contour in dashed amber for clear before/after diff
        if (origPts.length >= 3) {
          ctx.beginPath();
          ctx.moveTo(origPts[0][0], origPts[0][1]);
          for (let i = 1; i < origPts.length; i++) ctx.lineTo(origPts[i][0], origPts[i][1]);
          ctx.closePath();
          ctx.strokeStyle = 'rgba(251, 146, 60, 0.9)';
          ctx.lineWidth = 1.5 / zoom;
          ctx.setLineDash([4.5 / zoom, 3 / zoom]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // SAM-refined contour in luminous emerald with subtle translucent fill
        if (refinedPts.length >= 3) {
          ctx.lineJoin = contourStyle === 'exact' ? 'miter' : 'round';
          ctx.miterLimit = 2;
          ctx.beginPath();
          ctx.moveTo(refinedPts[0][0], refinedPts[0][1]);
          for (let i = 1; i < refinedPts.length; i++) {
            ctx.lineTo(refinedPts[i][0], refinedPts[i][1]);
          }
          ctx.closePath();
          for (const hole of prop.refinedHoles) {
            if (hole.length < 3) continue;
            ctx.moveTo(hole[0][0], hole[0][1]);
            for (let i = 1; i < hole.length; i++) ctx.lineTo(hole[i][0], hole[i][1]);
            ctx.closePath();
          }
          ctx.fillStyle = isLocked ? 'rgba(16, 185, 129, 0.26)' : 'rgba(16, 185, 129, 0.19)';
          ctx.fill('evenodd');

          if (isLocked) {
            ctx.strokeStyle = 'rgba(52, 211, 153, 0.45)';
            ctx.lineWidth = 5.2 / zoom;
            ctx.stroke();
          }
          ctx.strokeStyle = '#10B981';
          ctx.lineWidth = (isLocked ? 2.3 : 1.9) / zoom;
          ctx.stroke();
        }

        // Compact floating diff badge above refined segment bbox
        const areaSign = prop.areaDeltaPct >= 0 ? '+' : '';
        const edgeSign = prop.edgeGainPct >= 0 ? '+' : '';
        const hint = isLocked
          ? lang === 'zh'
            ? 'Enter/双击确认'
            : 'Enter/DblClick Apply'
          : lang === 'zh'
          ? '单击锁定'
          : 'Click Lock';
        const tag = `#${prop.objectId} · ΔA ${areaSign}${prop.areaDeltaPct.toFixed(1)}% · Edge ${edgeSign}${prop.edgeGainPct.toFixed(0)}% · ${hint}`;
        ctx.font = `600 ${Math.max(9.5, 10 / zoom)}px "JetBrains Mono", monospace`;
        const tw = ctx.measureText(tag).width;
        const lx = Math.max(
          4 / zoom,
          Math.min(viewWidth - tw - 12 / zoom, prop.metrics.cx - tw * 0.5)
        );
        const ly = Math.max(18 / zoom, prop.metrics.bbox.y - 6 / zoom);
        ctx.fillStyle = isLocked ? 'rgba(6, 28, 22, 0.94)' : 'rgba(7, 10, 18, 0.9)';
        ctx.fillRect(lx - 4 / zoom, ly - 12 / zoom, tw + 8 / zoom, 15 / zoom);
        ctx.strokeStyle = isLocked ? '#34D399' : 'rgba(16, 185, 129, 0.8)';
        ctx.lineWidth = 1.1 / zoom;
        ctx.strokeRect(lx - 4 / zoom, ly - 12 / zoom, tw + 8 / zoom, 15 / zoom);
        ctx.fillStyle = '#A7F3D0';
        ctx.fillText(tag, lx, ly - 1.5 / zoom);
        ctx.restore();
      }

      // 4. Manual ruler / profile lines / smart click-measured objects
      const drawLineOverlay = (
        p1: [number, number],
        p2: [number, number],
        label: string,
        color: string
      ) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8 / zoom;
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
        for (const [px, py] of [p1, p2]) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(px, py, 3 / zoom, 0, Math.PI * 2);
          ctx.fill();
        }
        const mx = (p1[0] + p2[0]) * 0.5;
        const my = (p1[1] + p2[1]) * 0.5;
        ctx.font = `600 ${11 / zoom}px "JetBrains Mono", monospace`;
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, mx + 6 / zoom, my - 6 / zoom);
        ctx.restore();
      };

      measurements.forEach((m, mIdx) => {
        if (m.kind === 'object' && m.objectMetrics) {
          const om = m.objectMetrics;
          const pts = contourStyle === 'smooth' ? smoothPixelContour(om.polygon) : om.polygon;
          if (pts.length < 3) return;
          ctx.save();
          ctx.lineJoin = contourStyle === 'exact' ? 'miter' : 'round';
          ctx.miterLimit = 2;
          ctx.beginPath();
          ctx.moveTo(pts[0][0], pts[0][1]);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
          ctx.closePath();
          if (om.holes) {
            for (const hole of om.holes) {
              if (hole.length < 3) continue;
              ctx.moveTo(hole[0][0], hole[0][1]);
              for (let i = 1; i < hole.length; i++) ctx.lineTo(hole[i][0], hole[i][1]);
              ctx.closePath();
            }
          }
          ctx.fillStyle = 'rgba(16, 185, 129, 0.16)';
          ctx.fill('evenodd');
          ctx.strokeStyle = '#34D399';
          ctx.lineWidth = 1.8 / zoom;
          ctx.stroke();

          // Orthogonal Major (Length, amber solid) & Minor (Width, cyan dashed) axes
          const drawAxisWithCaps = (
            a: [number, number],
            b: [number, number],
            color: string,
            dashed: boolean
          ) => {
            const dx = b[0] - a[0];
            const dy = b[1] - a[1];
            const dLen = Math.hypot(dx, dy) || 1;
            const nx = -dy / dLen;
            const ny = dx / dLen;
            const cap = 4 / zoom;
            ctx.save();
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.4 / zoom;
            if (dashed) ctx.setLineDash([3.5 / zoom, 2.5 / zoom]);
            ctx.beginPath();
            ctx.moveTo(a[0], a[1]);
            ctx.lineTo(b[0], b[1]);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.moveTo(a[0] - nx * cap, a[1] - ny * cap);
            ctx.lineTo(a[0] + nx * cap, a[1] + ny * cap);
            ctx.moveTo(b[0] - nx * cap, b[1] - ny * cap);
            ctx.lineTo(b[0] + nx * cap, b[1] + ny * cap);
            ctx.stroke();
            ctx.restore();
          };

          drawAxisWithCaps(om.majorAxis[0], om.majorAxis[1], '#FBBF24', false);
          drawAxisWithCaps(om.minorAxis[0], om.minorAxis[1], '#22D3EE', true);

          // Compact measurement readout plate above object bbox
          const tag = `#${mIdx + 1} ${om.lengthPhys.toFixed(1)}×${om.widthPhys.toFixed(1)} ${m.unit} · ${om.areaPhys.toFixed(1)} ${om.areaUnit}`;
          ctx.font = `600 ${Math.max(9.5, 10 / zoom)}px "JetBrains Mono", monospace`;
          const tw = ctx.measureText(tag).width;
          const tx = Math.max(4 / zoom, Math.min(viewWidth - tw - 10 / zoom, om.cx - tw * 0.5));
          const ty = Math.max(16 / zoom, om.bbox.y - 5 / zoom);
          ctx.fillStyle = 'rgba(7, 10, 18, 0.88)';
          ctx.fillRect(tx - 4 / zoom, ty - 11.5 / zoom, tw + 8 / zoom, 14.5 / zoom);
          ctx.strokeStyle = 'rgba(52, 211, 153, 0.75)';
          ctx.lineWidth = 1 / zoom;
          ctx.strokeRect(tx - 4 / zoom, ty - 11.5 / zoom, tw + 8 / zoom, 14.5 / zoom);
          ctx.fillStyle = '#A7F3D0';
          ctx.fillText(tag, tx, ty - 1.5 / zoom);
          ctx.restore();
          return;
        }

        const layerTag =
          m.kind === 'profile' && m.profile?.layerCount && m.profile.layerCount > 0
            ? lang === 'zh'
              ? ` · ${m.profile.layerCount}层 (均值 ${m.profile.meanLayerThickness?.toFixed(1) ?? '—'} ${m.unit})`
              : ` · ${m.profile.layerCount}L (avg ${m.profile.meanLayerThickness?.toFixed(1) ?? '—'} ${m.unit})`
            : '';
        drawLineOverlay(
          m.p1,
          m.p2,
          `${m.kind === 'profile' ? 'Profile ' : ''}${m.lengthPhys.toFixed(1)} ${m.unit}${layerTag}`,
          m.kind === 'profile' ? '#A78BFA' : '#FBBF24'
        );

        // Draw crisp perpendicular interface tick marks along manual profile line
        if (m.kind === 'profile' && m.profile && m.profile.edges.length > 0 && m.lengthPhys > 1e-6) {
          const dx = m.p2[0] - m.p1[0];
          const dy = m.p2[1] - m.p1[1];
          const len = Math.hypot(dx, dy) || 1;
          const perpX = -dy / len;
          const perpY = dx / len;
          const tickHalf = 5 / zoom;
          ctx.save();
          ctx.strokeStyle = '#22D3EE';
          ctx.lineWidth = 1.4 / zoom;
          for (const edgePos of m.profile.edges) {
            const frac = Math.max(0, Math.min(1, edgePos / m.lengthPhys));
            const ex = m.p1[0] + dx * frac;
            const ey = m.p1[1] + dy * frac;
            ctx.beginPath();
            ctx.moveTo(ex - perpX * tickHalf, ey - perpY * tickHalf);
            ctx.lineTo(ex + perpX * tickHalf, ey + perpY * tickHalf);
            ctx.stroke();
          }
          ctx.restore();
        }
      });
      if (activeDragLine) {
        const lenPx = Math.hypot(
          activeDragLine.p2[0] - activeDragLine.p1[0],
          activeDragLine.p2[1] - activeDragLine.p1[1]
        );
        if (toolMode === 'calibrate_line') {
          const snapped = liveCalSnap;
          const [x1, y1] = snapped ? snapped.p1 : activeDragLine.p1;
          const [x2, y2] = snapped ? snapped.p2 : activeDragLine.p2;
          const dispLen = snapped ? snapped.lengthPx : lenPx;
          const dx = x2 - x1;
          const dy = y2 - y1;
          const dLen = Math.hypot(dx, dy) || 1;
          const pxNorm = -dy / dLen;
          const pyNorm = dx / dLen;
          const cap = 7 / zoom;
          ctx.save();
          if (snapped) {
            // Subtle dashed guide showing user's rough hand stroke before magnetic lock
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.55)';
            ctx.lineWidth = 1.2 / zoom;
            ctx.setLineDash([4 / zoom, 3 / zoom]);
            ctx.beginPath();
            ctx.moveTo(activeDragLine.p1[0], activeDragLine.p1[1]);
            ctx.lineTo(activeDragLine.p2[0], activeDragLine.p2[1]);
            ctx.stroke();
            ctx.setLineDash([]);
          }
          ctx.strokeStyle = snapped ? '#10B981' : '#34D399';
          ctx.lineWidth = (snapped ? 2.4 : 2.0) / zoom;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.moveTo(x1 - pxNorm * cap, y1 - pyNorm * cap);
          ctx.lineTo(x1 + pxNorm * cap, y1 + pyNorm * cap);
          ctx.moveTo(x2 - pxNorm * cap, y2 - pyNorm * cap);
          ctx.lineTo(x2 + pxNorm * cap, y2 + pyNorm * cap);
          ctx.stroke();
          if (snapped) {
            // Endpoint magnetic lock rings
            const r = 3.5 / zoom;
            ctx.fillStyle = '#34D399';
            ctx.strokeStyle = '#022C22';
            ctx.lineWidth = 1.2 / zoom;
            for (const [ex, ey] of [[x1, y1], [x2, y2]]) {
              ctx.beginPath();
              ctx.arc(ex, ey, r, 0, Math.PI * 2);
              ctx.fill();
              ctx.stroke();
            }
          }
          const label = snapped
            ? lang === 'zh'
              ? `已吸附比例尺: ${dispLen.toFixed(1)} px`
              : `Snapped Bar: ${dispLen.toFixed(1)} px`
            : `Scale Bar: ${dispLen.toFixed(1)} px`;
          ctx.font = `600 ${Math.max(10, 11 / zoom)}px "JetBrains Mono", monospace`;
          const tw = ctx.measureText(label).width;
          const mx = (x1 + x2) * 0.5;
          const my = Math.max(4 / zoom, Math.min(y1, y2) - 20 / zoom);
          ctx.fillStyle = 'rgba(6, 20, 16, 0.92)';
          ctx.fillRect(mx - tw * 0.5 - 6 / zoom, my, tw + 12 / zoom, 16 / zoom);
          ctx.strokeStyle = snapped ? '#10B981' : '#34D399';
          ctx.lineWidth = 1.1 / zoom;
          ctx.strokeRect(mx - tw * 0.5 - 6 / zoom, my, tw + 12 / zoom, 16 / zoom);
          ctx.fillStyle = '#6EE7B7';
          ctx.fillText(label, mx - tw * 0.5, my + 11.5 / zoom);
          ctx.restore();
        } else {
          drawLineOverlay(
            activeDragLine.p1,
            activeDragLine.p2,
            `${(lenPx * specimen.calibration.pixelSize).toFixed(1)} ${specimen.calibration.spaceUnit}`,
            toolMode === 'measure_profile' ? '#A78BFA' : '#FBBF24'
          );
        }
      }

      // 4a. Persisted Manual Scale-Bar Calibration Line (while calibrating before info-bar crop)
      if (manualCalLine && !activeDragLine) {
        const [x1, y1] = manualCalLine.p1;
        const [x2, y2] = manualCalLine.p2;
        const dx = x2 - x1;
        const dy = y2 - y1;
        const dLen = Math.hypot(dx, dy) || 1;
        const pxNorm = -dy / dLen;
        const pyNorm = dx / dLen;
        const cap = 7 / zoom;
        ctx.save();
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 2.1 / zoom;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.moveTo(x1 - pxNorm * cap, y1 - pyNorm * cap);
        ctx.lineTo(x1 + pxNorm * cap, y1 + pyNorm * cap);
        ctx.moveTo(x2 - pxNorm * cap, y2 - pyNorm * cap);
        ctx.lineTo(x2 + pxNorm * cap, y2 + pyNorm * cap);
        ctx.stroke();
        if (manualCalLine.snapped) {
          const r = 3.2 / zoom;
          ctx.fillStyle = '#34D399';
          ctx.strokeStyle = '#022C22';
          ctx.lineWidth = 1.1 / zoom;
          for (const [ex, ey] of [[x1, y1], [x2, y2]]) {
            ctx.beginPath();
            ctx.arc(ex, ey, r, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          }
        }
        const label = manualCalLine.snapped
          ? lang === 'zh'
            ? `已吸附标尺 ${manualCalLine.lengthPx.toFixed(1)} px`
            : `Snapped Bar ${manualCalLine.lengthPx.toFixed(1)} px`
          : lang === 'zh'
          ? `标定线段 ${manualCalLine.lengthPx.toFixed(1)} px`
          : `Cal Bar ${manualCalLine.lengthPx.toFixed(1)} px`;
        ctx.font = `600 ${Math.max(10, 10.5 / zoom)}px "JetBrains Mono", monospace`;
        const tw = ctx.measureText(label).width;
        const mx = (x1 + x2) * 0.5;
        const my = Math.max(4 / zoom, Math.min(y1, y2) - 18 / zoom);
        ctx.fillStyle = 'rgba(6, 20, 16, 0.9)';
        ctx.fillRect(mx - tw * 0.5 - 5 / zoom, my, tw + 10 / zoom, 15 / zoom);
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.85)';
        ctx.lineWidth = 1 / zoom;
        ctx.strokeRect(mx - tw * 0.5 - 5 / zoom, my, tw + 10 / zoom, 15 / zoom);
        ctx.fillStyle = '#6EE7B7';
        ctx.fillText(label, mx - tw * 0.5, my + 11 / zoom);
        ctx.restore();
      }

      // 4b. Manual ROI overlay (dim excluded region outside ROI + crisp cyan dashed frame)
      const displayRoi: RoiRect | null = activeDragRoi
        ? {
            x: Math.round(Math.min(activeDragRoi.p1[0], activeDragRoi.p2[0])),
            y: Math.round(Math.min(activeDragRoi.p1[1], activeDragRoi.p2[1])),
            width: Math.round(Math.abs(activeDragRoi.p2[0] - activeDragRoi.p1[0])),
            height: Math.round(Math.abs(activeDragRoi.p2[1] - activeDragRoi.p1[1])),
          }
        : roi;

      if (displayRoi && displayRoi.width >= 2 && displayRoi.height >= 2 && !isRawSource) {
        const rx = Math.max(0, Math.min(viewWidth - 1, displayRoi.x));
        const ry = Math.max(0, Math.min(viewHeight - 1, displayRoi.y));
        const rw = Math.min(viewWidth - rx, displayRoi.width);
        const rh = Math.min(viewHeight - ry, displayRoi.height);
        if (rw >= 2 && rh >= 2) {
          ctx.save();
          // Dim regions outside ROI using evenodd cutout path
          ctx.fillStyle = activeDragRoi ? 'rgba(5, 8, 15, 0.42)' : 'rgba(5, 8, 15, 0.52)';
          ctx.beginPath();
          ctx.rect(0, 0, viewWidth, viewHeight);
          ctx.rect(rx, ry, rw, rh);
          ctx.fill('evenodd');

          // Crisp cyan dashed border
          ctx.strokeStyle = activeDragRoi ? '#38BDF8' : '#22D3EE';
          ctx.lineWidth = 1.8 / zoom;
          ctx.setLineDash([6 / zoom, 4 / zoom]);
          ctx.strokeRect(rx, ry, rw, rh);
          ctx.setLineDash([]);

          // Corner registration L-ticks
          const tick = Math.min(14 / zoom, rw * 0.25, rh * 0.25);
          ctx.strokeStyle = '#67E8F9';
          ctx.lineWidth = 2.4 / zoom;
          ctx.beginPath();
          ctx.moveTo(rx, ry + tick); ctx.lineTo(rx, ry); ctx.lineTo(rx + tick, ry);
          ctx.moveTo(rx + rw - tick, ry); ctx.lineTo(rx + rw, ry); ctx.lineTo(rx + rw, ry + tick);
          ctx.moveTo(rx, ry + rh - tick); ctx.lineTo(rx, ry + rh); ctx.lineTo(rx + tick, ry + rh);
          ctx.moveTo(rx + rw - tick, ry + rh); ctx.lineTo(rx + rw, ry + rh); ctx.lineTo(rx + rw, ry + rh - tick);
          ctx.stroke();

          // Compact ROI tag at top-left of box
          const roiTag = `ROI ${rw}×${rh} px`;
          ctx.font = `600 ${Math.max(9.5, 10.5 / zoom)}px "JetBrains Mono", monospace`;
          const tw = ctx.measureText(roiTag).width;
          const padX = 5 / zoom;
          const tagH = 15 / zoom;
          const tagY = ry > tagH + 4 / zoom ? ry - tagH - 3 / zoom : ry + 3 / zoom;
          ctx.fillStyle = 'rgba(7, 10, 18, 0.88)';
          ctx.fillRect(rx, tagY, tw + padX * 2, tagH);
          ctx.strokeStyle = 'rgba(34, 211, 238, 0.7)';
          ctx.lineWidth = 1 / zoom;
          ctx.strokeRect(rx, tagY, tw + padX * 2, tagH);
          ctx.fillStyle = '#67E8F9';
          ctx.fillText(roiTag, rx + padX, tagY + 11 / zoom);
          ctx.restore();
        }
      }

      // 5. Calibrated physical scale bar on image bottom-right
      const rawPhys = viewWidth * 0.16 * specimen.calibration.pixelSize;
      const pow = Math.pow(10, Math.floor(Math.log10(Math.max(1e-6, rawPhys))));
      const nice = [1, 2, 5, 10].find((m) => m * pow >= rawPhys * 0.7) ?? 1;
      const barPhys = nice * pow;
      const barPx = barPhys / Math.max(1e-6, specimen.calibration.pixelSize);
      const bx = viewWidth - barPx - 18;
      const by = viewHeight - 18;

      ctx.save();
      ctx.fillStyle = 'rgba(7, 10, 18, 0.76)';
      ctx.fillRect(bx - 7, by - 18, barPx + 14, 25);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(bx, by, barPx, 2.5);
      ctx.font = '600 10.5px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${barPhys} ${specimen.calibration.spaceUnit}`, bx + barPx / 2, by - 5);
      ctx.restore();

      ctx.restore();
    };

    render(performance.now());

    // Drive smooth 60fps highlight animation only while a fiber is selected and overlay is visible
    if (selectedFiberAxis && showOverlay && !isRawSource && !isCropPreview) {
      const tick = (ts: number) => {
        render(ts);
        rafId = requestAnimationFrame(tick);
      };
      rafId = requestAnimationFrame(tick);
    }

    const ro = new ResizeObserver(() => render(performance.now()));
    ro.observe(container);
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      ro.disconnect();
    };
  }, [
    baseImageCanvas,
    specimen,
    result,
    zoom,
    pan,
    showOverlay,
    showOverlayLabels,
    showOverlayFill,
    occludedDisplayMode,
    contourStyle,
    smoothedShapeContours,
    overlayOpacity,
    selectedObjectId,
    selectedFiberAxis,
    strokes,
    liveStroke,
    hoverSegPreview,
    samSensitivity,
    sensWheelFlash,
    hoveredRefineProposal,
    lockedRefineProposal,
    brushClassId,
    prompts,
    displayedPromptContours,
    measurements,
    activeDragLine,
    liveCalSnap,
    manualCalLine,
    roi,
    activeDragRoi,
    showLowConfHighlight,
    visibleLowConfSegments,
    hoveredLowConfId,
    activeCorrectionSegId,
    toolMode,
    isRawSource,
    isCropPreview,
    hasBuiltInBarCrop,
    pendingCropDraft,
    viewWidth,
    viewHeight,
    lang,
    darkMode,
  ]);

  // Register PNG snapshot callback for HTML Report export
  useEffect(() => {
    if (!onRegisterSnapshot) return;
    onRegisterSnapshot(() => canvasRef.current?.toDataURL('image/png'));
  }, [onRegisterSnapshot]);

  const recomputeHoverSegAt = (ix: number, iy: number, sens: number) => {
    const maxYBound =
      specimen.calibration.cropRows && specimen.calibration.cropRows < specimen.height
        ? specimen.calibration.cropRows
        : specimen.height;
    if (ix < 0 || ix >= specimen.width || iy < 0 || iy >= maxYBound) {
      setHoverSegPreview(null);
      return;
    }
    const segGray =
      effectiveGray ?? (showFluo && specimen.fluoGray ? specimen.fluoGray : specimen.gray);
    const preview = segmentObjectAtPoint(segGray, specimen.width, specimen.height, ix, iy, {
      cropRows: specimen.calibration.cropRows,
      roi,
      pixelSize: specimen.calibration.pixelSize,
      spaceUnit: specimen.calibration.spaceUnit,
      sensitivity: sens,
    });
    setHoverSegPreview(preview);
  };

  const updateSamSensitivity = (nextRaw: number, anchorPt?: [number, number] | null, flash = true) => {
    const clamped = Number(Math.max(0.3, Math.min(2.4, Math.round(nextRaw * 20) / 20)).toFixed(2));
    samSensitivityRef.current = clamped;
    setSamSensitivity(clamped);
    if (flash) {
      setSensWheelFlash(true);
      if (sensFlashTimerRef.current !== null) {
        window.clearTimeout(sensFlashTimerRef.current);
      }
      sensFlashTimerRef.current = window.setTimeout(() => {
        setSensWheelFlash(false);
        sensFlashTimerRef.current = null;
      }, 1150);
    }
    if (toolMode === 'smart_gt' || toolMode === 'measure_object') {
      const pt = anchorPt ?? lastHoverImgPtRef.current;
      if (pt) {
        recomputeHoverSegAt(pt[0], pt[1], clamped);
      }
    }
  };

  // Pointer interaction handlers
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    // In Smart Click GT & Smart Click Measure modes, mouse wheel directly adjusts boundary sensitivity/threshold
    // (scroll up = expand boundary, scroll down = shrink boundary; hold Ctrl/Cmd + wheel to zoom)
    if (
      (toolMode === 'smart_gt' || toolMode === 'measure_object') &&
      !e.ctrlKey &&
      !e.metaKey &&
      !e.altKey
    ) {
      const pt = clientToImage(e.clientX, e.clientY) ?? lastHoverImgPtRef.current;
      if (pt) lastHoverImgPtRef.current = pt;
      const step = e.shiftKey ? 0.15 : 0.05;
      const delta = e.deltaY < 0 ? step : -step;
      updateSamSensitivity(samSensitivityRef.current + delta, pt, true);
      return;
    }
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setZoom((z) => Number(Math.max(0.25, Math.min(8, z * factor)).toFixed(3)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const pt = clientToImage(e.clientX, e.clientY);
    if (!pt) return;
    const [ix, iy] = pt;
    const maxY = specimen.calibration.cropRows && specimen.calibration.cropRows < specimen.height
      ? specimen.calibration.cropRows
      : specimen.height;
    const inBounds = ix >= 0 && ix <= specimen.width && iy >= 0 && iy <= maxY;

    // Middle mouse button in smart_gt / measure_object: drag vertically to scrub boundary threshold, or middle-click to reset 1.0
    if (e.button === 1 && (toolMode === 'smart_gt' || toolMode === 'measure_object')) {
      e.preventDefault();
      const anchor: [number, number] = inBounds
        ? [ix, iy]
        : (lastHoverImgPtRef.current ?? [ix, iy]);
      lastHoverImgPtRef.current = anchor;
      middleSensDragRef.current = {
        startY: e.clientY,
        startSens: samSensitivityRef.current,
        anchorPt: anchor,
        moved: false,
      };
      setIsDragging(true);
      return;
    }

    if (toolMode === 'roi') {
      const cx = Math.max(0, Math.min(specimen.width, ix));
      const cy = Math.max(0, Math.min(maxY, iy));
      setActiveDragRoi({ p1: [cx, cy], p2: [cx, cy] });
      setIsDragging(true);
      return;
    }

    if (toolMode === 'boundary_refine') {
      if (e.button !== 0 || !inBounds || !result || isRawSource || isCropPreview) return;
      const hit = findSegmentAtPoint(ix, iy);
      if (!hit) {
        if (lockedRefineProposal) setLockedRefineProposal(null);
        return;
      }
      const proposal = computeRefineForShape(hit.shape, hit.obj);
      if (!proposal) return;
      // Clicking the already-locked segment or double-clicking directly confirms & applies
      if (
        (lockedRefineProposal && lockedRefineProposal.shapeId === proposal.shapeId) ||
        e.detail >= 2
      ) {
        handleConfirmBoundaryRefine(proposal);
      } else {
        setLockedRefineProposal(proposal);
        setHoveredRefineProposal(proposal);
      }
      return;
    }

    if (toolMode === 'smart_gt' || toolMode === 'measure_object') {
      if (e.button !== 0 || !inBounds) return;
      const segGray = effectiveGray ?? (showFluo && specimen.fluoGray ? specimen.fluoGray : specimen.gray);
      const seg = segmentObjectAtPoint(segGray, specimen.width, specimen.height, ix, iy, {
        cropRows: specimen.calibration.cropRows,
        roi,
        pixelSize: specimen.calibration.pixelSize,
        spaceUnit: specimen.calibration.spaceUnit,
        sensitivity: samSensitivityRef.current,
      });
      if (!seg) return;
      if (toolMode === 'smart_gt') {
        onAddStroke({
          id: `sam_gt_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
          classId: brushClassId,
          radius: 4,
          points: [[seg.cx, seg.cy]],
          polygon: seg.points,
          holes: seg.holes.length > 0 ? seg.holes : undefined,
          areaPx: seg.areaPx,
          source: 'sam_click',
        });
      } else {
        onAddMeasurement({
          id: `m_obj_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
          kind: 'object',
          p1: seg.metrics.majorAxis[0],
          p2: seg.metrics.majorAxis[1],
          lengthPhys: seg.metrics.lengthPhys,
          unit: specimen.calibration.spaceUnit,
          objectMetrics: seg.metrics,
        });
      }
      return;
    }

    if (toolMode === 'prompt_pos' || toolMode === 'prompt_neg') {
      if (!inBounds) return;
      onAddPrompt({
        id: `pt_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
        x: ix,
        y: iy,
        positive: toolMode === 'prompt_pos',
      });
      return;
    }

    if (toolMode === 'brush') {
      if (!inBounds) return;
      const st: AnnotationStroke = {
        id: `st_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
        classId: brushClassId,
        radius: brushRadius,
        points: [[ix, iy]],
      };
      activeStrokeRef.current = st;
      setLiveStroke(st);
      setIsDragging(true);
      return;
    }

    if (toolMode === 'calibrate_line') {
      // Allow tracing directly over the info bar / scale bar across the entire uncropped frame
      const cx = Math.max(0, Math.min(viewWidth, ix));
      const cy = Math.max(0, Math.min(viewHeight, iy));
      setActiveDragLine({ p1: [cx, cy], p2: [cx, cy] });
      setLiveCalSnap(null);
      setIsDragging(true);
      return;
    }

    if (toolMode === 'measure_line' || toolMode === 'measure_profile') {
      if (!inBounds) return;
      setActiveDragLine({ p1: [ix, iy], p2: [ix, iy] });
      setIsDragging(true);
      return;
    }

    // Pan mode: also check if clicking near a detected object
    setIsDragging(true);
    dragStartRef.current = { clientX: e.clientX, clientY: e.clientY, panX: pan.x, panY: pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const pt = clientToImage(e.clientX, e.clientY);
    if (pt) {
      const [ix, iy] = pt;
      const rx = Math.floor(ix);
      const ry = Math.floor(iy);
      if (rx >= 0 && rx < specimen.width && ry >= 0 && ry < specimen.height) {
        const arr = effectiveGray ?? (showFluo && specimen.fluoGray ? specimen.fluoGray : specimen.gray);
        setCursorInfo({ x: ix, y: iy, intensity: arr[ry * specimen.width + rx] });
      } else {
        setCursorInfo(null);
      }

      // Live SAM hover boundary refinement preview when hovering over segmented objects
      if (
        !isDragging &&
        toolMode === 'boundary_refine' &&
        !lockedRefineProposal &&
        result &&
        !isRawSource &&
        !isCropPreview
      ) {
        const hit = findSegmentAtPoint(ix, iy);
        if (!hit) {
          if (hoveredRefineProposal !== null) setHoveredRefineProposal(null);
        } else if (hoveredRefineProposal?.shapeId !== hit.shape.id) {
          const prop = computeRefineForShape(hit.shape, hit.obj);
          setHoveredRefineProposal(prop);
        }
      }

      // Middle-button vertical drag scrubs boundary threshold continuously around anchored point
      if (middleSensDragRef.current) {
        const dy = middleSensDragRef.current.startY - e.clientY;
        if (Math.abs(dy) >= 3) {
          middleSensDragRef.current.moved = true;
          const nextSens = middleSensDragRef.current.startSens + dy * 0.008;
          const now = performance.now();
          if (now - lastHoverSegTimeRef.current >= 32) {
            lastHoverSegTimeRef.current = now;
            updateSamSensitivity(nextSens, middleSensDragRef.current.anchorPt, true);
          }
        }
        return;
      }

      // Live SAM hover boundary preview for Smart Click GT & Smart Click Measure (throttled ~45ms)
      if (!isDragging && (toolMode === 'smart_gt' || toolMode === 'measure_object')) {
        const maxYBound =
          specimen.calibration.cropRows && specimen.calibration.cropRows < specimen.height
            ? specimen.calibration.cropRows
            : specimen.height;
        const insideWork = ix >= 0 && ix < specimen.width && iy >= 0 && iy < maxYBound;
        if (!insideWork) {
          lastHoverImgPtRef.current = null;
          if (hoverSegPreview) setHoverSegPreview(null);
        } else {
          lastHoverImgPtRef.current = [ix, iy];
          const now = performance.now();
          if (now - lastHoverSegTimeRef.current >= 45) {
            lastHoverSegTimeRef.current = now;
            recomputeHoverSegAt(ix, iy, samSensitivityRef.current);
          }
        }
      }

      // Hover hit-test subtle red low-confidence boxes in pan mode
      if (!isDragging && toolMode === 'pan' && showOverlay && showLowConfHighlight && !isRawSource && !isCropPreview && visibleLowConfSegments.length > 0) {
        let hitId: string | null = null;
        let smallestArea = Infinity;
        for (const seg of visibleLowConfSegments) {
          const { x, y, width: bw, height: bh } = seg.bbox;
          if (ix >= x && ix <= x + bw && iy >= y && iy <= y + bh) {
            const area = bw * bh;
            if (area < smallestArea) {
              smallestArea = area;
              hitId = seg.id;
            }
          }
        }
        if (hitId !== hoveredLowConfId) setHoveredLowConfId(hitId);
      } else if (hoveredLowConfId !== null) {
        setHoveredLowConfId(null);
      }

      if (isDragging) {
        if (toolMode === 'roi' && activeDragRoi) {
          const maxY = specimen.calibration.cropRows && specimen.calibration.cropRows < specimen.height
            ? specimen.calibration.cropRows
            : specimen.height;
          const cx = Math.max(0, Math.min(specimen.width, ix));
          const cy = Math.max(0, Math.min(maxY, iy));
          setActiveDragRoi({ ...activeDragRoi, p2: [cx, cy] });
          return;
        }
        if (toolMode === 'brush' && activeStrokeRef.current) {
          const prev = activeStrokeRef.current.points[activeStrokeRef.current.points.length - 1];
          if (Math.hypot(ix - prev[0], iy - prev[1]) >= 2) {
            activeStrokeRef.current = {
              ...activeStrokeRef.current,
              points: [...activeStrokeRef.current.points, [ix, iy]],
            };
            setLiveStroke(activeStrokeRef.current);
          }
          return;
        }
        if (toolMode === 'calibrate_line' && activeDragLine) {
          const cx = Math.max(0, Math.min(viewWidth, ix));
          let cy = Math.max(0, Math.min(viewHeight, iy));
          // Auto-snap strictly horizontal when tracing a horizontal scale bar (within 6px vertical drift)
          if (Math.abs(cy - activeDragLine.p1[1]) <= 6 && Math.abs(cx - activeDragLine.p1[0]) >= 12) {
            cy = activeDragLine.p1[1];
          }
          const nextLine: { p1: [number, number]; p2: [number, number] } = {
            p1: activeDragLine.p1,
            p2: [cx, cy],
          };
          setActiveDragLine(nextLine);
          if (!e.altKey && Math.hypot(cx - nextLine.p1[0], cy - nextLine.p1[1]) >= 10) {
            const snap = snapManualLineToScaleBar(
              calibrationGray,
              viewWidth,
              viewHeight,
              nextLine.p1,
              nextLine.p2
            );
            setLiveCalSnap(snap.snapped ? { p1: snap.p1, p2: snap.p2, lengthPx: snap.lengthPx } : null);
          } else {
            setLiveCalSnap(null);
          }
          return;
        }
        if ((toolMode === 'measure_line' || toolMode === 'measure_profile') && activeDragLine) {
          setActiveDragLine({ ...activeDragLine, p2: [ix, iy] });
          return;
        }
      }
    }

    if (isDragging && toolMode === 'pan' && dragStartRef.current) {
      const dx = e.clientX - dragStartRef.current.clientX;
      const dy = e.clientY - dragStartRef.current.clientY;
      setPan({ x: dragStartRef.current.panX + dx, y: dragStartRef.current.panY + dy });
    }
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (middleSensDragRef.current) {
      const drag = middleSensDragRef.current;
      middleSensDragRef.current = null;
      setIsDragging(false);
      if (!drag.moved) {
        updateSamSensitivity(1.0, drag.anchorPt, true);
      }
      return;
    }

    if (toolMode === 'roi' && activeDragRoi) {
      const maxY = specimen.calibration.cropRows && specimen.calibration.cropRows < specimen.height
        ? specimen.calibration.cropRows
        : specimen.height;
      const x0 = Math.max(0, Math.min(specimen.width - 1, Math.round(Math.min(activeDragRoi.p1[0], activeDragRoi.p2[0]))));
      const y0 = Math.max(0, Math.min(maxY - 1, Math.round(Math.min(activeDragRoi.p1[1], activeDragRoi.p2[1]))));
      const x1 = Math.max(0, Math.min(specimen.width, Math.round(Math.max(activeDragRoi.p1[0], activeDragRoi.p2[0]))));
      const y1 = Math.max(0, Math.min(maxY, Math.round(Math.max(activeDragRoi.p1[1], activeDragRoi.p2[1]))));
      const w = x1 - x0;
      const h = y1 - y0;
      if (w >= 16 && h >= 16) {
        onChangeRoi({ x: x0, y: y0, width: w, height: h });
        onChangeToolMode('pan');
      }
      setActiveDragRoi(null);
      setIsDragging(false);
      return;
    }

    if (toolMode === 'brush' && activeStrokeRef.current) {
      onAddStroke(activeStrokeRef.current);
      activeStrokeRef.current = null;
      setLiveStroke(null);
      setIsDragging(false);
      return;
    }

    if (toolMode === 'calibrate_line' && activeDragLine) {
      const { p1, p2 } = activeDragLine;
      const rawDistPx = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
      if (rawDistPx >= 6 && onSetManualCalLine) {
        const snap = !e.altKey
          ? snapManualLineToScaleBar(calibrationGray, viewWidth, viewHeight, p1, p2)
          : { snapped: false, p1, p2, lengthPx: Number(rawDistPx.toFixed(1)) };
        onSetManualCalLine({
          p1: snap.p1,
          p2: snap.p2,
          lengthPx: snap.lengthPx,
          snapped: snap.snapped,
        });
        onChangeToolMode('pan');
      }
      setActiveDragLine(null);
      setLiveCalSnap(null);
      setIsDragging(false);
      return;
    }

    if ((toolMode === 'measure_line' || toolMode === 'measure_profile') && activeDragLine) {
      const { p1, p2 } = activeDragLine;
      const distPx = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
      if (distPx >= 4) {
        const lengthPhys = distPx * specimen.calibration.pixelSize;
        let profile: ManualMeasurement['profile'];
        if (toolMode === 'measure_profile') {
          const grayArr = effectiveGray ?? specimen.gray;
          profile = analyzeLineProfileLayers(
            grayArr,
            specimen.width,
            specimen.height,
            p1,
            p2,
            specimen.calibration.pixelSize
          );
        }

        onAddMeasurement({
          id: `m_${Date.now()}`,
          kind: toolMode === 'measure_profile' ? 'profile' : 'line',
          p1,
          p2,
          lengthPhys: Number(lengthPhys.toFixed(2)),
          unit: specimen.calibration.spaceUnit,
          profile,
        });
      }
      setActiveDragLine(null);
      setIsDragging(false);
      return;
    }

    // Check if it was a click (not pan drag) to select object on canvas
    if (toolMode === 'pan' && dragStartRef.current && result) {
      const moved = Math.hypot(
        e.clientX - dragStartRef.current.clientX,
        e.clientY - dragStartRef.current.clientY
      );
      if (moved < 5) {
        const pt = clientToImage(e.clientX, e.clientY);
        if (pt) {
          // 1. Direct click on a subtle red low-confidence box triggers manual correction immediately
          if (showOverlay && showLowConfHighlight && !isRawSource && !isCropPreview && visibleLowConfSegments.length > 0) {
            let hitSeg: LowConfidenceSegment | null = null;
            let smallestArea = Infinity;
            for (const seg of visibleLowConfSegments) {
              const { x, y, width: bw, height: bh } = seg.bbox;
              if (pt[0] >= x && pt[0] <= x + bw && pt[1] >= y && pt[1] <= y + bh) {
                const area = bw * bh;
                if (area < smallestArea) {
                  smallestArea = area;
                  hitSeg = seg;
                }
              }
            }
            if (hitSeg) {
              onTriggerLowConfCorrection(hitSeg);
              dragStartRef.current = null;
              setIsDragging(false);
              return;
            }
          }

          if (activeCorrectionSeg) {
            onTriggerLowConfCorrection(null);
          }

          let bestId: number | null = null;
          let bestDist = 26 / zoom;
          const distToSeg = (
            px: number,
            py: number,
            x1: number,
            y1: number,
            x2: number,
            y2: number
          ) => {
            const dx = x2 - x1;
            const dy = y2 - y1;
            const l2 = dx * dx + dy * dy;
            if (l2 < 1e-6) return Math.hypot(px - x1, py - y1);
            const tProj = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / l2));
            return Math.hypot(px - (x1 + tProj * dx), py - (y1 + tProj * dy));
          };
          for (const obj of result.objects) {
            if (obj.occluded && occludedDisplayMode === 'hide') continue;
            let d = Math.hypot(obj.x - pt[0], obj.y - pt[1]);
            if (obj.kind === 'fibre') {
              const axis = result.shapes.find(
                (s) => s.id === `fib_axis_${obj.id}` && s.points.length >= 2
              );
              if (axis) {
                const p0 = axis.points[0];
                const p1 = axis.points[axis.points.length - 1];
                d = Math.min(d, distToSeg(pt[0], pt[1], p0[0], p0[1], p1[0], p1[1]));
              }
            }
            if (d < bestDist) {
              bestDist = d;
              bestId = obj.id;
            }
          }
          onSelectObject(bestId === selectedObjectId ? null : bestId);
        }
      }
    }

    dragStartRef.current = null;
    setIsDragging(false);
  };

  const tools: Array<{ id: ToolMode; icon: typeof MousePointer2; labelKey: string }> = [
    { id: 'pan', icon: MousePointer2, labelKey: 'tool_pan' },
    { id: 'calibrate_line', icon: MoveHorizontal, labelKey: 'tool_calibrate_line' },
    { id: 'roi', icon: Crop, labelKey: 'tool_roi' },
    { id: 'measure_line', icon: Ruler, labelKey: 'tool_line' },
    { id: 'measure_profile', icon: Activity, labelKey: 'tool_profile' },
    { id: 'measure_object', icon: ScanSearch, labelKey: 'tool_measure_object' },
    { id: 'boundary_refine', icon: Sparkles, labelKey: 'tool_boundary_refine' },
    { id: 'smart_gt', icon: Wand2, labelKey: 'tool_smart_gt' },
    { id: 'brush', icon: Brush, labelKey: 'tool_brush' },
    { id: 'prompt_pos', icon: PlusCircle, labelKey: 'tool_prompt_pos' },
    { id: 'prompt_neg', icon: MinusCircle, labelKey: 'tool_prompt_neg' },
  ];

  // Right-click on canvas to quickly delete the nearest manual measurement, prompt, or GT annotation
  const handleContextMenu = (e: React.MouseEvent) => {
    const pt = clientToImage(e.clientX, e.clientY);
    if (!pt) return;
    const [ix, iy] = pt;
    const hitTol = 14 / zoom;

    // 1. Check SAM prompt points
    for (let i = prompts.length - 1; i >= 0; i--) {
      const p = prompts[i];
      if (Math.hypot(ix - p.x, iy - p.y) <= hitTol) {
        e.preventDefault();
        onDeletePrompt(p.id);
        return;
      }
    }

    // 2. Check manual measurements (clicked object polygons & line/profile segments)
    const pointToSegDist = (
      px: number,
      py: number,
      x1: number,
      y1: number,
      x2: number,
      y2: number
    ) => {
      const dx = x2 - x1;
      const dy = y2 - y1;
      const len2 = dx * dx + dy * dy;
      if (len2 < 1e-6) return Math.hypot(px - x1, py - y1);
      const tProj = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / len2));
      return Math.hypot(px - (x1 + tProj * dx), py - (y1 + tProj * dy));
    };
    for (let i = measurements.length - 1; i >= 0; i--) {
      const m = measurements[i];
      if (
        m.kind === 'object' &&
        m.objectMetrics &&
        pointInPolygon(ix, iy, m.objectMetrics.polygon, m.objectMetrics.holes)
      ) {
        e.preventDefault();
        onDeleteMeasurement(m.id);
        return;
      }
      if (pointToSegDist(ix, iy, m.p1[0], m.p1[1], m.p2[0], m.p2[1]) <= hitTol) {
        e.preventDefault();
        onDeleteMeasurement(m.id);
        return;
      }
    }

    // 3. Check GT polygons & brush strokes
    for (let i = strokes.length - 1; i >= 0; i--) {
      const st = strokes[i];
      if (st.polygon && st.polygon.length >= 3 && pointInPolygon(ix, iy, st.polygon, st.holes)) {
        e.preventDefault();
        onDeleteStroke(st.id);
        return;
      }
      const tol = Math.max(hitTol, st.radius + 4 / zoom);
      if (st.points.some(([sx, sy]) => Math.hypot(ix - sx, iy - sy) <= tol)) {
        e.preventDefault();
        onDeleteStroke(st.id);
        return;
      }
    }
  };

  const totalManualItems = measurements.length + prompts.length + strokes.length;

  const cursorClass =
    toolMode === 'pan'
      ? hoveredLowConfId
        ? 'cursor-pointer'
        : isDragging
        ? 'cursor-grabbing'
        : 'cursor-grab'
      : 'cursor-crosshair';

  const isUnrecognized = activeRecipe.id === UNRECOGNIZED_MANUAL_RECIPE_ID;
  const isSelectedCustom = savedWorkflows.some((w) => w.id === activeRecipe.id);
  const importRep = specimen.importReport;
  const isUnrecognizedFallbackSeg =
    Boolean(importRep && !importRep.recognized) &&
    activeRecipe.id === DEFAULT_INTENSITY_RECIPE_ID;

  const openSaveWorkflow = () => {
    const defaultLabel =
      activeRecipe.id === UNRECOGNIZED_MANUAL_RECIPE_ID
        ? lang === 'zh'
          ? `${specimen.name.replace(/\.[^.]+$/, '')} 手动流程`
          : `${specimen.name.replace(/\.[^.]+$/, '')} Custom`
        : lang === 'zh'
        ? `${activeRecipe.nameZh} (自定义)`
        : `${activeRecipe.nameEn} (Custom)`;
    setCustomWfName(defaultLabel);
    setSaveWfOpen(true);
  };

  const submitSaveWorkflow = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = customWfName.trim();
    if (!trimmed) return;
    onSaveCustomWorkflow(trimmed);
    setSaveWfOpen(false);
    setCustomWfName('');
  };

  return (
    <div className="relative flex-1 min-h-0 bg-[#EEF2F6] dark:bg-[#070A12] flex flex-col overflow-hidden select-none">
      {/* Top Workflow Selector & Import Triage Bar Directly Above Image */}
      <div className="shrink-0 border-b border-slate-200 dark:border-slate-800/90 bg-white dark:bg-[#0B0F1A] px-3 py-1.5 flex flex-wrap items-center justify-between gap-2 text-[12px] z-30">
        {/* Left: Workflow Dropdown + Save Custom Workflow */}
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <div className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400 shrink-0">
            <Workflow className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
            <span className="text-[11px] font-semibold hidden sm:inline">
              {lang === 'zh' ? '工作流' : 'Workflow'}
            </span>
          </div>

          <select
            aria-label={lang === 'zh' ? '选择图像处理工作流' : 'Select image processing workflow'}
            value={activeRecipe.id}
            onChange={(e) => onSelectWorkflow(e.target.value)}
            className={`h-7 max-w-[320px] flex-1 min-w-[180px] px-2.5 rounded-md border text-[11.5px] font-medium truncate shadow-xs cursor-pointer focus:outline-none ${
              isUnrecognized
                ? 'border-amber-500/70 bg-amber-50 dark:bg-amber-500/10 text-amber-800 dark:text-amber-200'
                : 'border-slate-300 dark:border-slate-600 bg-white dark:bg-[#0E1424] text-slate-900 dark:text-slate-100 hover:border-blue-500 dark:hover:border-cyan-400'
            }`}
          >
            <option value={UNRECOGNIZED_MANUAL_RECIPE_ID} className="bg-white dark:bg-slate-900 text-amber-700 dark:text-amber-300">
              {lang === 'zh'
                ? '⚠ 未能识别 — 请手工标注 / 手动测量'
                : '⚠ Unrecognized — Manual Annotate / Measure'}
            </option>
            <optgroup label={lang === 'zh' ? '内置建议工作流' : 'Built-in Workflows'} className="bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300">
              {BUILTIN_RECIPES.map((r) => (
                <option key={r.id} value={r.id} className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
                  {lang === 'zh' ? r.nameZh : r.nameEn}
                  {importRep?.suggestedRecipeId === r.id
                    ? lang === 'zh'
                      ? ' (自动建议)'
                      : ' (Suggested)'
                    : ''}
                </option>
              ))}
            </optgroup>
            {savedWorkflows.length > 0 && (
              <optgroup
                label={lang === 'zh' ? '已保存的自定义工作流' : 'Saved Custom Workflows'}
                className="bg-white dark:bg-slate-900 text-blue-700 dark:text-cyan-300"
              >
                {savedWorkflows.map((w) => (
                  <option key={w.id} value={w.id} className="bg-white dark:bg-slate-900 text-blue-900 dark:text-cyan-100">
                    ★ {w.name}
                  </option>
                ))}
              </optgroup>
            )}
          </select>

          {!result && onRunActiveWorkflow && (
            <button
              type="button"
              onClick={onRunActiveWorkflow}
              title={lang === 'zh' ? '确认当前工作流并开始执行分析' : 'Confirm workflow and execute analysis'}
              className="h-7 px-2.5 rounded-md bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[11px] font-semibold shadow-xs flex items-center gap-1 whitespace-nowrap shrink-0 cursor-pointer transition-colors"
            >
              <Play className="w-3 h-3 fill-current" />
              <span>{lang === 'zh' ? '执行分析' : 'Run Workflow'}</span>
            </button>
          )}

          {!saveWfOpen ? (
            <div className="flex items-center gap-1 shrink-0">
              <button
                type="button"
                onClick={openSaveWorkflow}
                title={lang === 'zh' ? '自定义名称保存当前工作流' : 'Save current workflow with custom name'}
                className="h-7 px-2.5 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800/90 hover:bg-blue-50/70 dark:hover:bg-slate-700 hover:border-blue-500 dark:hover:border-cyan-400 shadow-xs text-slate-800 dark:text-slate-200 text-[11px] font-medium flex items-center gap-1 whitespace-nowrap cursor-pointer transition-colors"
              >
                <BookmarkPlus className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
                <span>{lang === 'zh' ? '保存工作流' : 'Save Workflow'}</span>
              </button>
              {isSelectedCustom && (
                <button
                  type="button"
                  onClick={() => onDeleteCustomWorkflow(activeRecipe.id)}
                  title={lang === 'zh' ? '删除此自定义工作流' : 'Delete custom workflow'}
                  className="h-7 w-7 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800/90 hover:border-rose-500/70 hover:text-rose-600 dark:hover:text-rose-400 text-slate-500 dark:text-slate-400 flex items-center justify-center cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ) : (
            <form ref={saveWfFormRef} onSubmit={submitSaveWorkflow} className="flex items-center gap-1.5 shrink-0">
              <input
                type="text"
                autoFocus
                value={customWfName}
                onChange={(e) => setCustomWfName(e.target.value)}
                placeholder={lang === 'zh' ? '输入自定义工作流名称…' : 'Custom workflow name…'}
                className="h-7 w-44 px-2 rounded-md border border-blue-500 dark:border-cyan-500/80 bg-white dark:bg-slate-950 text-[11.5px] text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none"
              />
              <button
                type="submit"
                disabled={!customWfName.trim()}
                className="h-7 px-2.5 rounded-md bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[11px] font-semibold flex items-center gap-1 disabled:opacity-45 cursor-pointer"
              >
                <Check className="w-3.5 h-3.5" />
                <span>{lang === 'zh' ? '保存' : 'Save'}</span>
              </button>
              <button
                type="button"
                onClick={() => setSaveWfOpen(false)}
                className="h-7 w-7 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 flex items-center justify-center cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </form>
          )}
        </div>

        {/* Right: Semi-Auto Staging Status or Step 1 Recognition + Step 2 Scale Bar Status */}
        <div className="flex items-center gap-2.5 text-[11px] font-mono shrink-0">
          {importInspecting ? (
            <span className="flex items-center gap-1.5 text-blue-600 dark:text-cyan-300">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>{lang === 'zh' ? '正在检测参考信息…' : 'Scanning hints…'}</span>
            </span>
          ) : !result ? (
            <span className="flex items-center gap-1.5 text-amber-600 dark:text-amber-300">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
              <span>
                {lang === 'zh' ? '半自动 · 等待确认后执行' : 'Semi-Auto · Awaiting Confirm'}
              </span>
            </span>
          ) : importRep ? (
            <>
              <span
                className={`flex items-center gap-1 ${
                  importRep.recognized ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'
                }`}
              >
                {importRep.recognized ? (
                  <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                ) : (
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                )}
                <span>
                  {lang === 'zh' ? importRep.sceneLabelZh : importRep.sceneLabelEn}
                </span>
              </span>
              {importRep.scaleBarCropped && (
                <span className="flex items-center gap-1 text-blue-700 dark:text-cyan-300 border-l border-slate-300 dark:border-slate-800 pl-2">
                  <Scissors className="w-3 h-3 shrink-0" />
                  <span>
                    {lang === 'zh'
                      ? `已裁切信息栏/标尺 ${importRep.croppedBarPx}px`
                      : `Cropped bar ${importRep.croppedBarPx}px`}
                    {importRep.scaleBarPx ? ` · ${importRep.scaleBarPx}px` : ''}
                  </span>
                </span>
              )}
            </>
          ) : (
            <span className="text-slate-500 tabular-nums">
              {activeRecipe.steps.filter((s) => s.enabled).length}{' '}
              {lang === 'zh' ? '步有效算子' : 'active steps'}
            </span>
          )}
        </div>
      </div>

      {/* Unrecognized Image Guidance Strip (only shown after user actually ran analysis) */}
      {result && (isUnrecognized || isUnrecognizedFallbackSeg) && (
        <div className="shrink-0 border-b border-amber-300 dark:border-amber-500/30 bg-amber-50 dark:bg-amber-500/10 px-3 py-1.5 flex flex-wrap items-center justify-between gap-2 text-[11.5px] text-amber-900 dark:text-amber-200 z-20">
          <div className="flex items-center gap-2 min-w-0">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 shrink-0" />
            <span className="truncate">
              {isUnrecognizedFallbackSeg
                ? lang === 'zh'
                  ? '当前运行「通用灰度阈值分割」；可在右侧微调阈值，或使用手工标注/测量工具。'
                  : 'Running default Intensity-Based Segmentation. Tune threshold in Parameters or annotate/measure manually.'
                : lang === 'zh'
                ? '当前为纯手工标注/测量模式，请使用下方快捷工具或从上方下拉菜单选择工作流。'
                : 'Manual annotation/measurement mode active. Use the quick tools or choose a workflow above.'}
            </span>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <button
              type="button"
              onClick={() => onChangeToolMode('measure_line')}
              className={`h-6 px-2 rounded border text-[11px] font-medium flex items-center gap-1 cursor-pointer transition-colors ${
                toolMode === 'measure_line'
                  ? 'border-blue-600 dark:border-cyan-400 bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 font-semibold'
                  : 'border-amber-400/70 dark:border-amber-500/40 bg-white dark:bg-slate-950/60 text-amber-900 dark:text-amber-100 hover:bg-amber-100/80 dark:hover:bg-slate-900'
              }`}
            >
              <Ruler className="w-3 h-3" />
              <span>{lang === 'zh' ? '手动测距' : 'Ruler'}</span>
            </button>
            <button
              type="button"
              onClick={() => onChangeToolMode('measure_profile')}
              className={`h-6 px-2 rounded border text-[11px] font-medium flex items-center gap-1 cursor-pointer transition-colors ${
                toolMode === 'measure_profile'
                  ? 'border-blue-600 dark:border-cyan-400 bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 font-semibold'
                  : 'border-amber-400/70 dark:border-amber-500/40 bg-white dark:bg-slate-950/60 text-amber-900 dark:text-amber-100 hover:bg-amber-100/80 dark:hover:bg-slate-900'
              }`}
            >
              <Activity className="w-3 h-3" />
              <span>{lang === 'zh' ? '线剖面' : 'Profile'}</span>
            </button>
            <button
              type="button"
              onClick={() => onChangeToolMode('measure_object')}
              className={`h-6 px-2 rounded border text-[11px] font-medium flex items-center gap-1 whitespace-nowrap cursor-pointer transition-colors ${
                toolMode === 'measure_object'
                  ? 'border-blue-600 dark:border-cyan-400 bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 font-semibold'
                  : 'border-amber-400/70 dark:border-amber-500/40 bg-white dark:bg-slate-950/60 text-amber-900 dark:text-amber-100 hover:bg-amber-100/80 dark:hover:bg-slate-900'
              }`}
            >
              <ScanSearch className="w-3 h-3" />
              <span>{lang === 'zh' ? '点选测物体' : 'Click Measure'}</span>
            </button>
            <button
              type="button"
              onClick={() => onChangeToolMode('smart_gt')}
              className={`h-6 px-2 rounded border text-[11px] font-medium flex items-center gap-1 whitespace-nowrap cursor-pointer transition-colors ${
                toolMode === 'smart_gt'
                  ? 'border-blue-600 dark:border-cyan-400 bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 font-semibold'
                  : 'border-amber-400/70 dark:border-amber-500/40 bg-white dark:bg-slate-950/60 text-amber-900 dark:text-amber-100 hover:bg-amber-100/80 dark:hover:bg-slate-900'
              }`}
            >
              <Wand2 className="w-3 h-3" />
              <span>{lang === 'zh' ? '点选标注 GT' : 'Click GT'}</span>
            </button>
            <button
              type="button"
              onClick={() => onChangeToolMode('brush')}
              className={`h-6 px-2 rounded border text-[11px] font-medium flex items-center gap-1 whitespace-nowrap cursor-pointer transition-colors ${
                toolMode === 'brush'
                  ? 'border-blue-600 dark:border-cyan-400 bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 font-semibold'
                  : 'border-amber-400/70 dark:border-amber-500/40 bg-white dark:bg-slate-950/60 text-amber-900 dark:text-amber-100 hover:bg-amber-100/80 dark:hover:bg-slate-900'
              }`}
            >
              <Brush className="w-3 h-3" />
              <span>{lang === 'zh' ? '画笔标注' : 'Brush'}</span>
            </button>
          </div>
        </div>
      )}

      {/* Interactive Image Stage Container (HUD overlays positioned cleanly inside this stage) */}
      <div className="relative flex-1 min-h-0 flex flex-col overflow-hidden">
        {/* Left vertical laboratory tool rail */}
        <div className="absolute left-3 top-3 z-20 flex flex-col gap-1 p-1 rounded-lg border border-slate-300 dark:border-slate-700/90 bg-white/95 dark:bg-[#0B0E17]/92 shadow-md dark:shadow-lg backdrop-blur">
        {tools.map((tool) => {
          const Icon = tool.icon;
          const active = toolMode === tool.id;
          return (
            <button
              key={tool.id}
              type="button"
              title={t(tool.labelKey, lang)}
              onClick={() => onChangeToolMode(tool.id)}
              className={`w-8 h-8 rounded-md flex items-center justify-center border transition-colors cursor-pointer ${
                active
                  ? 'border-blue-600 dark:border-cyan-400 bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 shadow-xs'
                  : 'border-transparent text-slate-700 dark:text-slate-200 hover:border-slate-300 dark:hover:border-slate-600 hover:bg-blue-50/80 dark:hover:bg-slate-800 hover:text-blue-700 dark:hover:text-white'
              }`}
            >
              <Icon className="w-4 h-4" />
            </button>
          );
        })}
      </div>

      {/* Contextual Brush / Prompt bar when active */}
      {toolMode === 'brush' && (
        <div className="absolute left-14 top-3 z-20 flex items-center gap-3 px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white/95 dark:bg-[#0B0E17]/94 shadow-md backdrop-blur text-[12px] text-slate-800 dark:text-slate-200">
          <div className="flex items-center gap-1.5">
            {[1, 2, 3].map((cid) => {
              const info = BRUSH_CLASS_COLORS[cid];
              const active = brushClassId === cid;
              return (
                <button
                  key={cid}
                  type="button"
                  onClick={() => onChangeBrushClassId(cid)}
                  className={`px-2 py-0.5 rounded text-[11px] font-medium flex items-center gap-1.5 border transition-colors cursor-pointer ${
                    active
                      ? 'border-blue-600 dark:border-white/70 bg-blue-50 dark:bg-slate-800 text-blue-800 dark:text-white'
                      : 'border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: info.color }}
                  />
                  <span>{lang === 'zh' ? info.zh : info.en}</span>
                </button>
              );
            })}
          </div>
          <label className="flex items-center gap-1.5 text-[11px] text-slate-600 dark:text-slate-400 font-mono">
            <span>R={brushRadius}</span>
            <input
              type="range"
              min={3}
              max={24}
              value={brushRadius}
              onChange={(e) => onChangeBrushRadius(Number(e.target.value))}
              className="w-16 accent-blue-600 dark:accent-cyan-400 cursor-pointer"
            />
          </label>
          {strokes.length > 0 && (
            <div className="flex items-center gap-2 border-l border-slate-300 dark:border-slate-700 pl-2">
              <button
                type="button"
                onClick={() => onDeleteStroke(strokes[strokes.length - 1].id)}
                title={lang === 'zh' ? '撤销上一笔' : 'Undo last stroke'}
                className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-cyan-300 flex items-center gap-1 text-[11px] cursor-pointer"
              >
                <Undo2 className="w-3.5 h-3.5" />
                <span>{lang === 'zh' ? '撤销' : 'Undo'}</span>
              </button>
              <button
                type="button"
                onClick={onClearStrokes}
                title={lang === 'zh' ? '清空全部画笔' : 'Clear all strokes'}
                className="text-slate-500 dark:text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 flex items-center gap-1 text-[11px] cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>{strokes.length}</span>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Contextual SAM Boundary Refine bar */}
      {toolMode === 'boundary_refine' && (
        <div className="absolute left-14 top-3 z-20 flex flex-wrap items-center gap-2.5 px-3 py-1.5 rounded-lg border border-emerald-500/70 bg-white/95 dark:bg-[#0B0E17]/95 backdrop-blur text-[11.5px] text-slate-800 dark:text-slate-200 shadow-lg">
          <span className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-300 font-semibold whitespace-nowrap">
            <Sparkles className="w-3.5 h-3.5 shrink-0" />
            <span>{lang === 'zh' ? 'SAM 边界精修' : 'Boundary Refine'}</span>
          </span>

          <label
            className="flex items-center gap-1.5 text-[11px] text-slate-600 dark:text-slate-400 font-mono border-l border-slate-300 dark:border-slate-700 pl-2"
            title={
              lang === 'zh'
                ? '调节 SAM 边界贴合敏感度（小=收紧，大=扩展）'
                : 'Adjust SAM boundary sensitivity (lower = tighter, higher = wider)'
            }
          >
            <span className="whitespace-nowrap">
              {lang === 'zh' ? '敏感度' : 'Sens'} {samSensitivity.toFixed(1)}
            </span>
            <input
              type="range"
              min={0.5}
              max={1.6}
              step={0.1}
              value={samSensitivity}
              onChange={(e) => setSamSensitivity(Number(e.target.value))}
              className="w-16 accent-emerald-600 dark:accent-emerald-400 cursor-pointer"
            />
          </label>

          {!result ? (
            <span className="text-[11px] text-amber-700 dark:text-amber-300 border-l border-slate-300 dark:border-slate-700 pl-2 whitespace-nowrap">
              {lang === 'zh' ? '请先执行分割流程以生成目标轮廓' : 'Run segmentation first to refine segments'}
            </span>
          ) : activeRefineProposal ? (
            <>
              <div className="flex items-center gap-2 font-mono text-[11px] tabular-nums border-l border-slate-300 dark:border-slate-700 pl-2 whitespace-nowrap">
                <span className="text-slate-900 dark:text-white font-semibold">#{activeRefineProposal.objectId}</span>
                <span className="text-slate-700 dark:text-slate-300">
                  {activeRefineProposal.metrics.areaPhys.toFixed(1)} {activeRefineProposal.metrics.areaUnit}
                  <span
                    className={`ml-1 ${
                      Math.abs(activeRefineProposal.areaDeltaPct) <= 15
                        ? 'text-blue-600 dark:text-cyan-300'
                        : 'text-amber-600 dark:text-amber-300'
                    }`}
                  >
                    ({activeRefineProposal.areaDeltaPct >= 0 ? '+' : ''}
                    {activeRefineProposal.areaDeltaPct.toFixed(1)}%)
                  </span>
                </span>
                <span className="text-emerald-700 dark:text-emerald-300">
                  {lang === 'zh' ? '边缘' : 'Edge'}{' '}
                  {activeRefineProposal.edgeGainPct >= 0 ? '+' : ''}
                  {activeRefineProposal.edgeGainPct.toFixed(0)}%
                </span>
              </div>

              <div className="flex items-center gap-1.5 border-l border-slate-300 dark:border-slate-700 pl-2">
                <button
                  type="button"
                  onClick={() => handleConfirmBoundaryRefine(activeRefineProposal)}
                  className="h-6 px-2.5 rounded bg-emerald-600 hover:bg-emerald-700 dark:bg-emerald-500 dark:hover:bg-emerald-400 text-white dark:text-slate-950 text-[11px] font-semibold flex items-center gap-1 whitespace-nowrap cursor-pointer transition-colors"
                >
                  <Check className="w-3.5 h-3.5 shrink-0" />
                  <span>{lang === 'zh' ? '确认应用' : 'Confirm & Apply'}</span>
                </button>
                {lockedRefineProposal && (
                  <button
                    type="button"
                    onClick={() => setLockedRefineProposal(null)}
                    className="h-6 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:border-slate-500 text-[11px] flex items-center gap-1 whitespace-nowrap cursor-pointer"
                  >
                    <X className="w-3 h-3 shrink-0" />
                    <span>{lang === 'zh' ? '取消' : 'Cancel'}</span>
                  </button>
                )}
              </div>
            </>
          ) : (
            <span className="text-[11px] text-slate-600 dark:text-slate-400 border-l border-slate-300 dark:border-slate-700 pl-2 whitespace-nowrap">
              {lang === 'zh'
                ? '悬停任意分割目标预览精修（橙虚=原边界 · 绿实=精修），单击锁定确认'
                : 'Hover any segment to preview (amber dashed = original, emerald = SAM), click to lock & apply'}
            </span>
          )}
        </div>
      )}

      {/* Contextual Smart Click GT (SAM) bar */}
      {toolMode === 'smart_gt' && (
        <div className="absolute left-14 top-3 z-20 flex flex-wrap items-center gap-2.5 px-3 py-1.5 rounded-lg border border-blue-500/60 dark:border-cyan-500/50 bg-white/95 dark:bg-[#0B0E17]/94 shadow-md backdrop-blur text-[11.5px] text-slate-800 dark:text-slate-200">
          <span className="flex items-center gap-1.5 text-blue-700 dark:text-cyan-300 font-semibold whitespace-nowrap">
            <Wand2 className="w-3.5 h-3.5 shrink-0" />
            <span>{lang === 'zh' ? 'SAM 点选 GT' : 'Smart Click GT'}</span>
          </span>
          <div className="flex items-center gap-1">
            {[1, 2, 3].map((cid) => {
              const info = BRUSH_CLASS_COLORS[cid];
              const active = brushClassId === cid;
              return (
                <button
                  key={cid}
                  type="button"
                  onClick={() => onChangeBrushClassId(cid)}
                  className={`px-2 py-0.5 rounded text-[11px] font-medium flex items-center gap-1.5 border whitespace-nowrap transition-colors cursor-pointer ${
                    active
                      ? 'border-blue-600 dark:border-white/70 bg-blue-600 dark:bg-slate-800 text-white shadow-xs'
                      : 'border-slate-300 dark:border-slate-700/70 bg-slate-50 dark:bg-transparent text-slate-700 dark:text-slate-300 hover:border-slate-400 dark:hover:text-white'
                  }`}
                >
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: info.color }}
                  />
                  <span>{lang === 'zh' ? info.zh : info.en}</span>
                </button>
              );
            })}
          </div>
          <label
            className={`flex items-center gap-1.5 text-[11px] font-mono border-l border-slate-300 dark:border-slate-700 pl-2 transition-colors ${
              sensWheelFlash
                ? 'text-amber-600 dark:text-amber-300 font-semibold'
                : 'text-slate-600 dark:text-slate-400'
            }`}
            title={
              lang === 'zh'
                ? '滚轮上下滚动或按住中键上下拖动可实时扩缩边界（中键单击重置 1.0，Ctrl+滚轮缩放画布）'
                : 'Scroll wheel or middle-drag vertically to expand/shrink boundary (middle-click resets 1.0; Ctrl+wheel zooms)'
            }
          >
            <span className="whitespace-nowrap tabular-nums">
              {lang === 'zh' ? '阈值' : 'Sens'} {samSensitivity.toFixed(2)}
            </span>
            <input
              type="range"
              min={0.3}
              max={2.4}
              step={0.05}
              value={samSensitivity}
              onChange={(e) => updateSamSensitivity(Number(e.target.value), null, false)}
              className="w-16 accent-blue-600 dark:accent-cyan-400 cursor-pointer"
            />
            {Math.abs(samSensitivity - 1.0) > 0.01 && (
              <button
                type="button"
                onClick={() => updateSamSensitivity(1.0, null, true)}
                className="px-1 py-0.2 rounded border border-slate-300 dark:border-slate-700 text-[10px] text-slate-600 dark:text-slate-300 hover:border-blue-500 dark:hover:border-cyan-400 cursor-pointer"
                title={lang === 'zh' ? '重置默认阈值 1.0' : 'Reset threshold to 1.0'}
              >
                1.0
              </button>
            )}
          </label>
          <span className="hidden xl:inline text-[10.5px] text-slate-500 dark:text-slate-400 whitespace-nowrap">
            {lang === 'zh' ? '滚轮扩缩边界 · Ctrl+滚轮缩放' : 'Wheel: Expand/Shrink · Ctrl+Wheel: Zoom'}
          </span>
          <button
            type="button"
            onClick={() => onChangeToolMode('brush')}
            className="px-2 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/70 text-slate-700 dark:text-slate-200 hover:border-blue-500 dark:hover:border-slate-500 hover:bg-blue-50 dark:hover:bg-slate-800 text-[10.5px] font-medium flex items-center gap-1 whitespace-nowrap cursor-pointer shadow-2xs"
          >
            <Brush className="w-3 h-3" />
            <span>{lang === 'zh' ? '切换画笔' : 'Brush'}</span>
          </button>
          {strokes.length > 0 && (
            <div className="flex items-center gap-1.5 border-l border-slate-300 dark:border-slate-700 pl-2">
              <button
                type="button"
                onClick={() => onDeleteStroke(strokes[strokes.length - 1].id)}
                className="px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-blue-500 dark:hover:border-cyan-400 hover:text-blue-700 dark:hover:text-cyan-300 flex items-center gap-1 text-[11px] font-medium whitespace-nowrap cursor-pointer"
              >
                <Undo2 className="w-3.5 h-3.5" />
                <span>{lang === 'zh' ? '撤销' : 'Undo'}</span>
              </button>
              <button
                type="button"
                onClick={onClearStrokes}
                className="px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:border-rose-400 hover:text-rose-600 dark:hover:text-rose-400 flex items-center gap-1 text-[11px] font-mono cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>{strokes.length}</span>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Contextual Smart Click Measure bar */}
      {toolMode === 'measure_object' && (() => {
        const latestObjMeas = [...measurements].reverse().find((m) => m.kind === 'object' && m.objectMetrics);
        return (
          <div className="absolute left-14 top-3 z-20 flex flex-wrap items-center gap-2.5 px-3 py-1.5 rounded-lg border border-emerald-500/60 dark:border-emerald-500/55 bg-white/95 dark:bg-[#0B0E17]/94 shadow-md backdrop-blur text-[11px] text-slate-800 dark:text-slate-200">
            <span className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-300 font-semibold whitespace-nowrap">
              <ScanSearch className="w-3.5 h-3.5 shrink-0" />
              <span>{lang === 'zh' ? '点击物体自动测长宽/面积' : 'Click Object to Measure'}</span>
            </span>
            <label
              className={`flex items-center gap-1.5 text-[11px] font-mono border-l border-slate-300 dark:border-slate-700 pl-2 transition-colors ${
                sensWheelFlash
                  ? 'text-amber-600 dark:text-amber-300 font-semibold'
                  : 'text-slate-600 dark:text-slate-400'
              }`}
              title={
                lang === 'zh'
                  ? '滚轮上下滚动或按住中键上下拖动可实时扩缩边界（中键单击重置 1.0，Ctrl+滚轮缩放画布）'
                  : 'Scroll wheel or middle-drag vertically to expand/shrink boundary (middle-click resets 1.0; Ctrl+wheel zooms)'
              }
            >
              <span className="whitespace-nowrap tabular-nums">
                {lang === 'zh' ? '阈值' : 'Sens'} {samSensitivity.toFixed(2)}
              </span>
              <input
                type="range"
                min={0.3}
                max={2.4}
                step={0.05}
                value={samSensitivity}
                onChange={(e) => updateSamSensitivity(Number(e.target.value), null, false)}
                className="w-16 accent-emerald-600 dark:accent-emerald-400 cursor-pointer"
              />
              {Math.abs(samSensitivity - 1.0) > 0.01 && (
                <button
                  type="button"
                  onClick={() => updateSamSensitivity(1.0, null, true)}
                  className="px-1 py-0.2 rounded border border-slate-300 dark:border-slate-700 text-[10px] text-slate-600 dark:text-slate-300 hover:border-emerald-500 dark:hover:border-emerald-400 cursor-pointer"
                  title={lang === 'zh' ? '重置默认阈值 1.0' : 'Reset threshold to 1.0'}
                >
                  1.0
                </button>
              )}
            </label>
            <span className="hidden xl:inline text-[10.5px] text-slate-500 dark:text-slate-400 whitespace-nowrap">
              {lang === 'zh' ? '滚轮扩缩边界 · Ctrl+滚轮缩放' : 'Wheel: Expand/Shrink · Ctrl+Wheel: Zoom'}
            </span>
            {latestObjMeas?.objectMetrics && (
              <>
                <span className="font-mono tabular-nums text-emerald-800 dark:text-emerald-200 font-medium border-l border-slate-300 dark:border-slate-700 pl-2 whitespace-nowrap">
                  L {latestObjMeas.objectMetrics.lengthPhys.toFixed(1)} × W {latestObjMeas.objectMetrics.widthPhys.toFixed(1)} {latestObjMeas.unit} · A {latestObjMeas.objectMetrics.areaPhys.toFixed(1)} {latestObjMeas.objectMetrics.areaUnit}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    const om = latestObjMeas.objectMetrics!;
                    onAddStroke({
                      id: `sam_gt_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
                      classId: brushClassId,
                      radius: 4,
                      points: [[om.cx, om.cy]],
                      polygon: om.polygon,
                      holes: om.holes,
                      areaPx: om.areaPx,
                      source: 'sam_click',
                    });
                  }}
                  className="px-2 py-0.5 rounded border border-blue-500/70 dark:border-cyan-500/60 bg-blue-50 dark:bg-cyan-500/15 hover:bg-blue-100 dark:hover:bg-cyan-500/30 text-blue-700 dark:text-cyan-200 text-[10.5px] font-semibold flex items-center gap-1 whitespace-nowrap cursor-pointer shadow-2xs"
                >
                  <Wand2 className="w-3 h-3" />
                  <span>{lang === 'zh' ? '加入 GT 标注' : 'Save as GT'}</span>
                </button>
              </>
            )}
            {measurements.length > 0 && (
              <div className="flex items-center gap-1.5 border-l border-slate-300 dark:border-slate-700 pl-2">
                <button
                  type="button"
                  onClick={() => onDeleteMeasurement(measurements[measurements.length - 1].id)}
                  className="px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-blue-500 dark:hover:border-cyan-400 hover:text-blue-700 dark:hover:text-cyan-300 flex items-center gap-1 font-medium whitespace-nowrap cursor-pointer"
                >
                  <Undo2 className="w-3.5 h-3.5" />
                  <span>{lang === 'zh' ? '撤销' : 'Undo'}</span>
                </button>
                <button
                  type="button"
                  onClick={onClearMeasurements}
                  className="px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:border-rose-400 hover:text-rose-600 dark:hover:text-rose-400 flex items-center gap-1 font-mono cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>{measurements.length}</span>
                </button>
              </div>
            )}
          </div>
        );
      })()}

      {(toolMode === 'prompt_pos' || toolMode === 'prompt_neg') && (
        <div className="absolute left-14 top-3 z-20 flex flex-wrap items-center gap-2.5 px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white/95 dark:bg-[#0B0E17]/94 shadow-md backdrop-blur text-[11.5px] text-slate-800 dark:text-slate-200">
          <span className="text-emerald-700 dark:text-emerald-400 font-mono font-semibold text-[11px] whitespace-nowrap">
            MobileSAM · {prompts.length} {lang === 'zh' ? '提示点' : 'prompts'} · {promptContours.length}{' '}
            {lang === 'zh' ? '轮廓' : 'masks'}
          </span>
          <label className="flex items-center gap-1.5 text-[10.5px] text-slate-600 dark:text-slate-400 font-mono border-l border-slate-300 dark:border-slate-700 pl-2">
            <span>{samSensitivity.toFixed(1)}</span>
            <input
              type="range"
              min={0.5}
              max={1.6}
              step={0.1}
              value={samSensitivity}
              onChange={(e) => setSamSensitivity(Number(e.target.value))}
              className="w-14 accent-emerald-600 dark:accent-emerald-400 cursor-pointer"
            />
          </label>
          {promptContours.length > 0 && (
            <div className="flex items-center gap-1.5 border-l border-slate-300 dark:border-slate-700 pl-2">
              <button
                type="button"
                onClick={() => {
                  for (const seg of promptContours) {
                    onAddStroke({
                      id: `sam_gt_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
                      classId: brushClassId,
                      radius: 4,
                      points: [[seg.cx, seg.cy]],
                      polygon: seg.points,
                      holes: seg.holes.length > 0 ? seg.holes : undefined,
                      areaPx: seg.areaPx,
                      source: 'sam_click',
                    });
                  }
                  onClearPrompts();
                }}
                className="px-2 py-0.5 rounded border border-blue-500/70 dark:border-cyan-500/60 bg-blue-50 dark:bg-cyan-500/15 hover:bg-blue-100 dark:hover:bg-cyan-500/30 text-blue-700 dark:text-cyan-200 text-[10.5px] font-semibold whitespace-nowrap cursor-pointer shadow-2xs"
              >
                {lang === 'zh' ? '转为 GT 标注' : 'Save as GT'}
              </button>
              <button
                type="button"
                onClick={() => {
                  for (const seg of promptContours) {
                    onAddMeasurement({
                      id: `m_obj_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
                      kind: 'object',
                      p1: seg.metrics.majorAxis[0],
                      p2: seg.metrics.majorAxis[1],
                      lengthPhys: seg.metrics.lengthPhys,
                      unit: specimen.calibration.spaceUnit,
                      objectMetrics: seg.metrics,
                    });
                  }
                  onClearPrompts();
                }}
                className="px-2 py-0.5 rounded border border-emerald-500/70 dark:border-emerald-500/60 bg-emerald-50 dark:bg-emerald-500/15 hover:bg-emerald-100 dark:hover:bg-emerald-500/30 text-emerald-700 dark:text-emerald-200 text-[10.5px] font-semibold whitespace-nowrap cursor-pointer shadow-2xs"
              >
                {lang === 'zh' ? '转为对象测量' : 'Measure'}
              </button>
            </div>
          )}
          {prompts.length > 0 && (
            <div className="flex items-center gap-1.5 border-l border-slate-300 dark:border-slate-700 pl-2">
              <button
                type="button"
                onClick={() => onDeletePrompt(prompts[prompts.length - 1].id)}
                className="px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-blue-500 dark:hover:border-cyan-400 hover:text-blue-700 dark:hover:text-cyan-300 flex items-center gap-1 text-[11px] font-medium whitespace-nowrap cursor-pointer"
              >
                <Undo2 className="w-3.5 h-3.5" />
                <span>{lang === 'zh' ? '撤销' : 'Undo'}</span>
              </button>
              <button
                type="button"
                onClick={onClearPrompts}
                className="px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:border-rose-400 hover:text-rose-600 dark:hover:text-rose-400 flex items-center gap-1 text-[11px] font-medium whitespace-nowrap cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>{lang === 'zh' ? '清除' : 'Clear'}</span>
              </button>
            </div>
          )}
        </div>
      )}

      {toolMode === 'calibrate_line' && (
        <div className="absolute left-14 top-3 z-20 flex items-center gap-2.5 px-3 py-1.5 rounded-lg border border-emerald-500/70 dark:border-emerald-500/60 bg-white/95 dark:bg-[#0B0E17]/95 shadow-md backdrop-blur text-[11px] font-mono text-emerald-800 dark:text-emerald-200">
          <MoveHorizontal className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
          <span className="font-medium">
            {lang === 'zh'
              ? '请沿信息栏中的比例尺 (Scale Bar) 拖拽描线，松开后在右侧输入实际物理长度（近水平自动吸附）'
              : 'Drag along the scale bar inside the info bar (auto-snaps horizontally), then enter its printed physical length on the right'}
          </span>
          <button
            type="button"
            onClick={() => onChangeToolMode('pan')}
            className="p-0.5 rounded border border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {(toolMode === 'measure_line' || toolMode === 'measure_profile') && measurements.length > 0 && (
        <div className="absolute left-14 top-3 z-20 flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white/95 dark:bg-[#0B0E17]/94 shadow-md backdrop-blur text-[11px] text-slate-800 dark:text-slate-200 font-mono">
          <span className="font-medium">
            {measurements.length} {lang === 'zh' ? '条手动测量' : 'measurements'}
          </span>
          <button
            type="button"
            onClick={() => onDeleteMeasurement(measurements[measurements.length - 1].id)}
            className="px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-blue-500 dark:hover:border-cyan-400 hover:text-blue-700 dark:hover:text-cyan-300 flex items-center gap-1 font-sans font-medium cursor-pointer"
          >
            <Undo2 className="w-3.5 h-3.5" />
            <span>{lang === 'zh' ? '撤销' : 'Undo'}</span>
          </button>
          <button
            type="button"
            onClick={onClearMeasurements}
            className="px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:border-rose-400 hover:text-rose-600 dark:hover:text-rose-400 flex items-center gap-1 font-sans font-medium cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>{lang === 'zh' ? '清除' : 'Clear'}</span>
          </button>
        </div>
      )}

      {/* Contextual ROI Draw / Active ROI Status Bar */}
      {(toolMode === 'roi' || (roi && toolMode === 'pan')) && (
        <div className="absolute left-14 top-3 z-20 flex items-center gap-2.5 px-3 py-1.5 rounded-lg border border-blue-500/60 dark:border-cyan-500/50 bg-white/95 dark:bg-[#0B0E17]/94 shadow-md backdrop-blur text-[11px] font-mono text-slate-800 dark:text-slate-200">
          <Crop className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400 shrink-0" />
          {toolMode === 'roi' && !roi ? (
            <>
              <span className="text-blue-700 dark:text-cyan-300 font-medium">
                {lang === 'zh'
                  ? '在图像上按住鼠标左键拖拽框选 ROI（参数计算与模型训练将仅在 ROI 内执行）'
                  : 'Drag on image to draw ROI (parameters & training run strictly inside ROI)'}
              </span>
              {segmentedResultsRoi && (
                <>
                  <span className="w-px h-3.5 bg-slate-300 dark:bg-slate-700 shrink-0" />
                  <button
                    type="button"
                    onClick={() => {
                      onChangeRoi(segmentedResultsRoi);
                      onChangeToolMode('pan');
                    }}
                    title={
                      lang === 'zh'
                        ? `自动框选全部检测目标外接矩形并留出 10% 边距 (${segmentedResultsRoi.width}×${segmentedResultsRoi.height} px)`
                        : `Set ROI to bounding box of all detected objects with 10% padding (${segmentedResultsRoi.width}×${segmentedResultsRoi.height} px)`
                    }
                    className="h-5 px-2 rounded border border-blue-500/70 dark:border-cyan-500/60 bg-blue-50 dark:bg-cyan-500/15 hover:bg-blue-100 dark:hover:bg-cyan-500/30 text-blue-700 dark:text-cyan-200 text-[10.5px] font-sans font-semibold flex items-center gap-1 whitespace-nowrap shrink-0 cursor-pointer transition-colors shadow-2xs"
                  >
                    <Crosshair className="w-3 h-3 text-blue-600 dark:text-cyan-300 shrink-0" />
                    <span>{lang === 'zh' ? '聚焦结果' : 'Focus on Results'}</span>
                  </button>
                </>
              )}
            </>
          ) : roi ? (
            <>
              <span className="text-blue-700 dark:text-cyan-300 font-semibold tabular-nums whitespace-nowrap">
                ROI {roi.width}×{roi.height} px
                {' · '}
                {(roi.width * specimen.calibration.pixelSize).toFixed(1)}×
                {(roi.height * specimen.calibration.pixelSize).toFixed(1)}{' '}
                {specimen.calibration.spaceUnit}
                {' · '}
                {Math.max(
                  1,
                  Math.round(
                    ((roi.width * roi.height) /
                      Math.max(1, specimen.width * (specimen.calibration.cropRows ?? specimen.height))) *
                      100
                  )
                )}
                %
              </span>
              <span className="w-px h-3.5 bg-slate-300 dark:bg-slate-700 shrink-0" />
              {segmentedResultsRoi && (
                <button
                  type="button"
                  onClick={() => {
                    onChangeRoi(segmentedResultsRoi);
                    if (toolMode === 'roi') onChangeToolMode('pan');
                  }}
                  title={
                    lang === 'zh'
                      ? `自动框选全部检测目标外接矩形并留出 10% 边距 (${segmentedResultsRoi.width}×${segmentedResultsRoi.height} px)`
                      : `Set ROI to bounding box of all detected objects with 10% padding (${segmentedResultsRoi.width}×${segmentedResultsRoi.height} px)`
                  }
                  className="px-2 py-0.5 rounded border border-blue-500/60 dark:border-cyan-500/50 bg-blue-50 dark:bg-cyan-500/15 hover:bg-blue-100 dark:hover:bg-cyan-500/30 text-blue-700 dark:text-cyan-200 font-sans font-semibold flex items-center gap-1 whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <Crosshair className="w-3 h-3 shrink-0" />
                  <span>{lang === 'zh' ? '聚焦结果' : 'Focus on Results'}</span>
                </button>
              )}
              <button
                type="button"
                onClick={() => onChangeToolMode('roi')}
                className={`px-1.5 py-0.5 rounded border font-sans text-[11px] whitespace-nowrap shrink-0 cursor-pointer ${
                  toolMode === 'roi'
                    ? 'border-blue-600 dark:border-cyan-500 bg-blue-600 dark:bg-cyan-500/20 text-white dark:text-cyan-300 font-semibold'
                    : 'border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-blue-500 dark:hover:border-cyan-400 hover:text-blue-700 dark:hover:text-cyan-300'
                }`}
              >
                {lang === 'zh' ? '重画' : 'Redraw'}
              </button>
              <button
                type="button"
                onClick={() => {
                  onChangeRoi(null);
                  if (toolMode === 'roi') onChangeToolMode('pan');
                }}
                className="px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:border-rose-400 hover:text-rose-600 dark:hover:text-rose-400 font-sans font-medium flex items-center gap-1 whitespace-nowrap shrink-0 cursor-pointer"
              >
                <X className="w-3 h-3 shrink-0" />
                <span>{lang === 'zh' ? '清除 ROI' : 'Clear ROI'}</span>
              </button>
            </>
          ) : null}
        </div>
      )}

      {/* Top-right HUD: Channel, Marks Manager, Overlay, LUT, Zoom */}
      <div className="absolute right-3 top-3 z-20 flex items-center gap-1.5">
        {totalManualItems > 0 && (
          <button
            ref={marksBtnRef}
            type="button"
            onClick={() => {
              setShowMarksPanel((v) => !v);
              setShowLutPanel(false);
              setShowOverlayPanel(false);
            }}
            className={`h-7 px-2.5 rounded-md border text-[11px] font-mono font-medium shadow-xs flex items-center gap-1.5 transition-colors cursor-pointer ${
              showMarksPanel
                ? 'border-amber-500 bg-amber-50 dark:bg-amber-500/20 text-amber-800 dark:text-amber-300'
                : 'border-slate-300 dark:border-slate-700 bg-white/95 dark:bg-[#0B0E17]/92 text-slate-800 dark:text-slate-200 hover:border-blue-500 dark:hover:border-cyan-400 hover:bg-blue-50/70 dark:hover:bg-slate-800'
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
            <span>
              {lang === 'zh' ? `测量/标记 (${totalManualItems})` : `Marks (${totalManualItems})`}
            </span>
          </button>
        )}
        {specimen.fluoRgba && (
          <button
            type="button"
            onClick={() => setShowFluo((v) => !v)}
            className={`h-7 px-2.5 rounded-md border text-[11px] font-medium shadow-xs transition-colors cursor-pointer ${
              showFluo
                ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-300'
                : 'border-slate-300 dark:border-slate-700 bg-white/95 dark:bg-[#0B0E17]/92 text-slate-700 dark:text-slate-300 hover:border-blue-500 dark:hover:border-cyan-400 hover:text-slate-950 dark:hover:text-white'
            }`}
          >
            {t('canvas_fluo_channel', lang)}
          </button>
        )}
        {/* Contour boundary rendering mode: Exact Pixel Boundary vs Lightly Smoothed Contour */}
        <div
          role="group"
          aria-label={lang === 'zh' ? '分割边界显示模式' : 'Segmentation contour mode'}
          className="h-7 rounded-md border border-slate-300 dark:border-slate-700 bg-white/95 dark:bg-[#0B0E17]/92 shadow-xs flex items-center overflow-hidden text-[11px]"
        >
          <button
            type="button"
            onClick={() => setContourStyle('exact')}
            title={
              lang === 'zh'
                ? '严格沿阈值前景像素外沿绘制（0px 偏差，保留凹形与内部孔洞）'
                : 'Trace exact pixel-grid edges of thresholded mask (0 px error, preserves concavities & holes)'
            }
            className={`h-full px-2 transition-colors cursor-pointer ${
              contourStyle === 'exact'
                ? 'bg-blue-600/15 dark:bg-cyan-500/20 text-blue-700 dark:text-cyan-300 font-semibold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100/80 dark:hover:bg-slate-800/70'
            }`}
          >
            {lang === 'zh' ? '严格像素边界' : 'Exact Pixel'}
          </button>
          <span className="w-px h-3.5 bg-slate-300 dark:bg-slate-700" />
          <button
            type="button"
            onClick={() => setContourStyle('smooth')}
            title={
              lang === 'zh'
                ? '基于真实像素边界执行轻微保形平滑（<0.6px 偏差，消除斜边锯齿）'
                : 'Corner-preserving light Taubin smoothing on exact pixel edges (<0.6 px deviation)'
            }
            className={`h-full px-2 transition-colors cursor-pointer ${
              contourStyle === 'smooth'
                ? 'bg-blue-600/15 dark:bg-cyan-500/20 text-blue-700 dark:text-cyan-300 font-semibold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100/80 dark:hover:bg-slate-800/70'
            }`}
          >
            {lang === 'zh' ? '轻微平滑轮廓' : 'Light Smooth'}
          </button>
        </div>

        {visibleLowConfSegments.length > 0 && !isRawSource && !isCropPreview && (
          <button
            type="button"
            onClick={onToggleLowConfHighlight}
            title={
              lang === 'zh'
                ? '自动红框标出低对比/低置信片段（点击红框可触发手动纠偏）'
                : 'Auto-highlight low-contrast / low-confidence segments (click box to manually correct)'
            }
            className={`h-7 px-2 rounded-md border text-[11px] font-mono shadow-xs flex items-center gap-1.5 transition-colors cursor-pointer ${
              showLowConfHighlight
                ? 'border-rose-500/70 bg-rose-50 dark:bg-rose-500/15 text-rose-700 dark:text-rose-300'
                : 'border-slate-300 dark:border-slate-700 bg-white/95 dark:bg-[#0B0E17]/92 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${showLowConfHighlight ? 'bg-rose-500 dark:bg-rose-400' : 'bg-slate-400 dark:bg-slate-600'}`} />
            <span>
              {lang === 'zh'
                ? `低置信 ${visibleLowConfSegments.length}`
                : `Low Conf ${visibleLowConfSegments.length}`}
            </span>
          </button>
        )}
        {/* Overlay split control: quick eye toggle + options menu (labels, fill, occluded particles) */}
        <div
          className={`h-7 rounded-md border shadow-xs flex items-center overflow-hidden text-[11px] ${
            showOverlayPanel
              ? 'border-blue-600 dark:border-cyan-500/70 bg-blue-50 dark:bg-cyan-500/15 text-blue-800 dark:text-cyan-200'
              : 'border-slate-300 dark:border-slate-700 bg-white/95 dark:bg-[#0B0E17]/92 text-slate-800 dark:text-slate-200'
          }`}
        >
          <button
            type="button"
            onClick={() => setShowOverlay((v) => !v)}
            title={lang === 'zh' ? '显示/隐藏分割叠加层' : 'Toggle segmentation overlay'}
            className="h-full px-2 hover:bg-blue-50 dark:hover:bg-slate-800/80 font-medium flex items-center gap-1.5 cursor-pointer"
          >
            {showOverlay ? (
              <Eye className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
            ) : (
              <EyeOff className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500" />
            )}
            <span>{t('canvas_layers', lang)}</span>
            {showOverlayLabels && (
              <span className="px-1 rounded bg-blue-600/15 dark:bg-cyan-500/20 text-blue-700 dark:text-cyan-300 font-mono text-[9.5px]">
                Aa
              </span>
            )}
          </button>
          <span className="w-px h-3.5 bg-slate-300 dark:bg-slate-700" />
          <button
            ref={overlayBtnRef}
            type="button"
            onClick={() => {
              setShowOverlayPanel((v) => !v);
              setShowLutPanel(false);
              setShowMarksPanel(false);
            }}
            title={
              lang === 'zh'
                ? '叠加层显示选项（文字标注、填充、被遮挡颗粒）'
                : 'Overlay options (labels, fill, occluded particles)'
            }
            className="h-full px-1.5 hover:bg-blue-50 dark:hover:bg-slate-800/80 flex items-center justify-center cursor-pointer"
          >
            <ChevronDown className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />
          </button>
        </div>
        <button
          ref={lutBtnRef}
          type="button"
          onClick={() => {
            setShowLutPanel((v) => !v);
            setShowMarksPanel(false);
            setShowOverlayPanel(false);
          }}
          className={`h-7 px-2.5 rounded-md border text-[11px] font-medium shadow-xs flex items-center gap-1.5 cursor-pointer ${
            showLutPanel
              ? 'border-blue-600 dark:border-cyan-500/70 bg-blue-50 dark:bg-cyan-500/15 text-blue-800 dark:text-cyan-300'
              : 'border-slate-300 dark:border-slate-700 bg-white/95 dark:bg-[#0B0E17]/92 text-slate-800 dark:text-slate-200 hover:border-blue-500 dark:hover:border-cyan-400 hover:bg-blue-50/70 dark:hover:bg-slate-800'
          }`}
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>LUT</span>
        </button>
        <button
          type="button"
          onClick={fitToView}
          className="h-7 px-2.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white/95 dark:bg-[#0B0E17]/92 text-slate-800 dark:text-slate-200 hover:border-blue-500 dark:hover:border-cyan-400 hover:bg-blue-50/70 dark:hover:bg-slate-800 shadow-xs text-[11px] font-mono font-medium flex items-center gap-1 cursor-pointer"
          title={t('canvas_fit', lang)}
        >
          <Maximize2 className="w-3 h-3" />
          <span>{Math.round(zoom * 100)}%</span>
        </button>
      </div>

      {/* Expandable Manual Measurements & Marks Delete Popover */}
      {showMarksPanel && totalManualItems > 0 && (
        <div
          ref={marksPopoverRef}
          className="absolute right-3 top-12 z-20 w-68 max-h-80 overflow-y-auto p-3 rounded-lg border border-slate-300 dark:border-slate-700 bg-white/97 dark:bg-[#0B0E17]/95 shadow-xl backdrop-blur text-[11px] text-slate-800 dark:text-slate-200 space-y-2.5"
        >
          <div className="flex items-center justify-between pb-1.5 border-b border-slate-200 dark:border-slate-800">
            <span className="font-semibold text-slate-900 dark:text-slate-100">
              {lang === 'zh' ? '手动测量与标记管理' : 'Manual Marks & Measurements'}
            </span>
            <button
              type="button"
              onClick={() => {
                onClearMeasurements();
                onClearPrompts();
                onClearStrokes();
                setShowMarksPanel(false);
              }}
              className="px-1.5 py-0.5 rounded border border-rose-300 dark:border-rose-500/40 bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 hover:bg-rose-100 dark:hover:bg-rose-500/20 font-medium flex items-center gap-1 cursor-pointer"
            >
              <Trash2 className="w-3 h-3" />
              <span>{lang === 'zh' ? '全部清空' : 'Clear All'}</span>
            </button>
          </div>
          <div className="text-[10.5px] text-slate-500 dark:text-slate-400">
            {lang === 'zh' ? '提示：也可直接在画布上右键点击某条测量或标记删除' : 'Tip: Right-click any mark on the canvas to remove it'}
          </div>

          {measurements.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[10.5px] font-mono uppercase text-amber-700 dark:text-amber-400 font-semibold">
                <span>{lang === 'zh' ? `手动测量 (${measurements.length})` : `Measurements (${measurements.length})`}</span>
                <button type="button" onClick={onClearMeasurements} className="text-slate-500 dark:text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 cursor-pointer">
                  {lang === 'zh' ? '清空' : 'Clear'}
                </button>
              </div>
              {measurements.map((m, idx) => (
                <div key={m.id} className="flex items-center justify-between gap-2 py-1 border-b border-slate-200/80 dark:border-slate-800/70 font-mono">
                  <span className="truncate text-slate-700 dark:text-slate-300 tabular-nums">
                    {m.kind === 'object' && m.objectMetrics
                      ? `#${idx + 1} ${lang === 'zh' ? '对象' : 'Obj'} ${m.objectMetrics.lengthPhys.toFixed(1)}×${m.objectMetrics.widthPhys.toFixed(1)} ${m.unit} · ${m.objectMetrics.areaPhys.toFixed(1)} ${m.objectMetrics.areaUnit}`
                      : `#${idx + 1} ${m.kind === 'profile' ? (lang === 'zh' ? '剖面' : 'Profile') : (lang === 'zh' ? '标尺' : 'Ruler')} · ${m.lengthPhys.toFixed(1)} ${m.unit}`}
                  </span>
                  <button
                    type="button"
                    onClick={() => onDeleteMeasurement(m.id)}
                    title={lang === 'zh' ? '删除此测量' : 'Delete measurement'}
                    className="text-slate-400 dark:text-slate-500 hover:text-rose-600 dark:hover:text-rose-400 p-0.5 cursor-pointer shrink-0"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {prompts.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[10.5px] font-mono uppercase text-emerald-700 dark:text-emerald-400 font-semibold">
                <span>{lang === 'zh' ? `提示点标记 (${prompts.length})` : `SAM Prompts (${prompts.length})`}</span>
                <button type="button" onClick={onClearPrompts} className="text-slate-500 dark:text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 cursor-pointer">
                  {lang === 'zh' ? '清空' : 'Clear'}
                </button>
              </div>
              {prompts.map((p, idx) => (
                <div key={p.id} className="flex items-center justify-between gap-2 py-1 border-b border-slate-200/80 dark:border-slate-800/70 font-mono">
                  <span className="truncate text-slate-700 dark:text-slate-300">
                    #{idx + 1} {p.positive ? '(+) FG' : '(-) BG'} · ({p.x.toFixed(0)}, {p.y.toFixed(0)})
                  </span>
                  <button
                    type="button"
                    onClick={() => onDeletePrompt(p.id)}
                    title={lang === 'zh' ? '删除此提示点' : 'Delete prompt'}
                    className="text-slate-400 dark:text-slate-500 hover:text-rose-600 dark:hover:text-rose-400 p-0.5 cursor-pointer shrink-0"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {strokes.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[10.5px] font-mono uppercase text-blue-700 dark:text-cyan-400 font-semibold">
                <span>{lang === 'zh' ? `画笔标注 (${strokes.length})` : `Brush Strokes (${strokes.length})`}</span>
                <button type="button" onClick={onClearStrokes} className="text-slate-500 dark:text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 cursor-pointer">
                  {lang === 'zh' ? '清空' : 'Clear'}
                </button>
              </div>
              {strokes.map((st, idx) => {
                const meta = BRUSH_CLASS_COLORS[st.classId] ?? BRUSH_CLASS_COLORS[1];
                const isSam = st.source === 'sam_click' || Boolean(st.polygon);
                return (
                  <div key={st.id} className="flex items-center justify-between gap-2 py-1 border-b border-slate-200/80 dark:border-slate-800/70 font-mono">
                    <span className="flex items-center gap-1.5 truncate text-slate-700 dark:text-slate-300 tabular-nums">
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: meta.color }} />
                      <span>
                        #{idx + 1} {isSam ? 'SAM ' : ''}{lang === 'zh' ? meta.zh : meta.en}
                        {isSam && st.areaPx ? ` (${st.areaPx}px)` : ` (${st.points.length}pts)`}
                      </span>
                    </span>
                    <button
                      type="button"
                      onClick={() => onDeleteStroke(st.id)}
                      title={lang === 'zh' ? '删除此标注' : 'Delete annotation'}
                      className="text-slate-400 dark:text-slate-500 hover:text-rose-600 dark:hover:text-rose-400 p-0.5 cursor-pointer shrink-0"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Expandable Overlay Options Popover (Labels, Region Fill, Occluded Particles Display) */}
      {showOverlayPanel && (
        <div
          ref={overlayPopoverRef}
          className="absolute right-3 top-12 z-20 w-64 p-3 rounded-lg border border-slate-300 dark:border-slate-700 bg-white/97 dark:bg-[#0B0E17]/96 backdrop-blur text-[11px] text-slate-800 dark:text-slate-200 space-y-2.5 shadow-xl"
        >
          <div className="flex items-center justify-between pb-1.5 border-b border-slate-200 dark:border-slate-800">
            <span className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
              <span>{lang === 'zh' ? '叠加层显示选项 (Overlay)' : 'Overlay Display Options'}</span>
            </span>
            <button
              type="button"
              onClick={() => setShowOverlayPanel(false)}
              className="text-slate-400 dark:text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-1.5">
            <label className="flex items-center justify-between gap-2 py-0.5 cursor-pointer select-none">
              <span className="text-slate-700 dark:text-slate-300 font-medium">
                {lang === 'zh' ? '显示分割物体边框' : 'Show Segment Contours'}
              </span>
              <input
                type="checkbox"
                checked={showOverlay}
                onChange={(e) => setShowOverlay(e.target.checked)}
                className="accent-blue-600 dark:accent-cyan-400 cursor-pointer"
              />
            </label>

            <label className="flex items-center justify-between gap-2 py-0.5 cursor-pointer select-none">
              <span className="flex items-center gap-1.5 text-slate-700 dark:text-slate-300 font-medium">
                <Type className="w-3 h-3 text-slate-500 dark:text-slate-400" />
                <span>{lang === 'zh' ? '显示物体文字标注' : 'Show Object Text Labels'}</span>
              </span>
              <input
                type="checkbox"
                checked={showOverlayLabels}
                onChange={(e) => setShowOverlayLabels(e.target.checked)}
                className="accent-blue-600 dark:accent-cyan-400 cursor-pointer"
              />
            </label>

            <label className="flex items-center justify-between gap-2 py-0.5 cursor-pointer select-none">
              <span className="text-slate-700 dark:text-slate-300 font-medium">
                {lang === 'zh' ? '显示内部半透明填充' : 'Show Translucent Fill'}
              </span>
              <input
                type="checkbox"
                checked={showOverlayFill}
                onChange={(e) => setShowOverlayFill(e.target.checked)}
                className="accent-blue-600 dark:accent-cyan-400 cursor-pointer"
              />
            </label>
          </div>

          <div className="pt-2 border-t border-slate-200 dark:border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-slate-800 dark:text-slate-300 font-semibold">
                {lang === 'zh' ? '被上层遮挡颗粒显示' : 'Occluded Particles'}
              </span>
              {occludedCount > 0 && (
                <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-purple-100 dark:bg-purple-500/20 text-purple-700 dark:text-purple-300 font-medium">
                  {lang === 'zh' ? `检出 ${occludedCount} 个` : `${occludedCount} found`}
                </span>
              )}
            </div>
            <div className="grid grid-cols-3 gap-1">
              {(
                [
                  {
                    id: 'highlight' as OccludedDisplayMode,
                    zh: '异色边框',
                    en: 'Distinct',
                    swatch: '#C084FC',
                  },
                  {
                    id: 'hide' as OccludedDisplayMode,
                    zh: '不显示',
                    en: 'Hide',
                    swatch: null,
                  },
                  {
                    id: 'normal' as OccludedDisplayMode,
                    zh: '同色',
                    en: 'Same',
                    swatch: '#10B981',
                  },
                ] as const
              ).map((opt) => {
                const active = occludedDisplayMode === opt.id;
                return (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => onChangeOccludedDisplayMode(opt.id)}
                    className={`h-6 px-1.5 rounded border text-[10.5px] flex items-center justify-center gap-1 transition-colors cursor-pointer ${
                      active
                        ? 'border-purple-500 dark:border-purple-400/70 bg-purple-50 dark:bg-purple-500/20 text-purple-800 dark:text-purple-200 font-semibold shadow-2xs'
                        : 'border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/70 text-slate-600 dark:text-slate-400 hover:border-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                    }`}
                  >
                    {opt.swatch && (
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: opt.swatch }}
                      />
                    )}
                    <span>{lang === 'zh' ? opt.zh : opt.en}</span>
                  </button>
                );
              })}
            </div>
            <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-relaxed">
              {lang === 'zh'
                ? '被上层物体压盖而仅露出局部区域的下层颗粒，可选择紫色虚线异色标出或直接隐藏。'
                : 'Lower-layer particles partially blocked by upper objects can be highlighted in purple dashed borders or hidden.'}
            </p>
          </div>

          <label className="block pt-1.5 border-t border-slate-200 dark:border-slate-800 space-y-1">
            <div className="flex justify-between font-mono text-slate-600 dark:text-slate-400">
              <span>{t('canvas_opacity', lang)}</span>
              <span>{Math.round(overlayOpacity * 100)}%</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(Number(e.target.value))}
              className="w-full accent-blue-600 dark:accent-cyan-400"
            />
          </label>
        </div>
      )}

      {/* Expandable LUT / Overlay Opacity popover */}
      {showLutPanel && (
        <div
          ref={lutPopoverRef}
          className="absolute right-3 top-12 z-20 w-56 p-3 rounded-lg border border-slate-300 dark:border-slate-700 bg-white/97 dark:bg-[#0B0E17]/95 shadow-xl backdrop-blur text-[11px] text-slate-800 dark:text-slate-200 space-y-2.5"
        >
          <div className="flex items-center justify-between pb-1.5 border-b border-slate-200 dark:border-slate-800">
            <span className="font-semibold text-slate-900 dark:text-slate-100">LUT & {t('canvas_layers', lang)}</span>
            <button
              type="button"
              onClick={() => {
                setContrast(1);
                setBrightness(0);
                setOverlayOpacity(0.9);
              }}
              className="px-1.5 py-0.5 rounded border border-blue-300 dark:border-cyan-500/40 bg-blue-50 dark:bg-cyan-500/10 text-blue-700 dark:text-cyan-400 hover:bg-blue-100 dark:hover:bg-cyan-500/20 font-medium flex items-center gap-1 cursor-pointer"
            >
              <RotateCcw className="w-3 h-3" />
              <span>{t('canvas_reset_lut', lang)}</span>
            </button>
          </div>
          <label className="block space-y-1">
            <div className="flex justify-between font-mono text-slate-600 dark:text-slate-400">
              <span>{lang === 'zh' ? '对比度' : 'Contrast'}</span>
              <span>{contrast.toFixed(2)}×</span>
            </div>
            <input
              type="range"
              min={0.4}
              max={2.4}
              step={0.05}
              value={contrast}
              onChange={(e) => setContrast(Number(e.target.value))}
              className="w-full accent-blue-600 dark:accent-cyan-400"
            />
          </label>
          <label className="block space-y-1">
            <div className="flex justify-between font-mono text-slate-600 dark:text-slate-400">
              <span>{lang === 'zh' ? '亮度偏移' : 'Brightness'}</span>
              <span>{brightness > 0 ? `+${brightness}` : brightness}</span>
            </div>
            <input
              type="range"
              min={-80}
              max={80}
              step={2}
              value={brightness}
              onChange={(e) => setBrightness(Number(e.target.value))}
              className="w-full accent-blue-600 dark:accent-cyan-400"
            />
          </label>
          <label className="block space-y-1">
            <div className="flex justify-between font-mono text-slate-600 dark:text-slate-400">
              <span>{t('canvas_opacity', lang)}</span>
              <span>{Math.round(overlayOpacity * 100)}%</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(Number(e.target.value))}
              className="w-full accent-blue-600 dark:accent-cyan-400"
            />
          </label>
        </div>
      )}

      {/* Main High-DPI Canvas Viewport */}
      <div
        ref={containerRef}
        className={`relative flex-1 min-h-0 w-full ${cursorClass}`}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onContextMenu={handleContextMenu}
        onMouseLeave={() => {
          setCursorInfo(null);
          setHoverSegPreview(null);
          lastHoverImgPtRef.current = null;
          middleSensDragRef.current = null;
          setHoveredRefineProposal(null);
          if (isDragging && toolMode === 'pan') {
            setIsDragging(false);
            dragStartRef.current = null;
          }
        }}
      >
        <canvas ref={canvasRef} className="block w-full h-full" />
      </div>

      {/* Inline Manual Correction Bar when a subtle red low-confidence box is clicked */}
      {activeCorrectionSeg && !isRawSource && !isCropPreview && (
        <div
          ref={correctionBarRef}
          className="absolute left-1/2 -translate-x-1/2 bottom-12 z-30 max-w-[92%] flex flex-wrap items-center gap-2 px-3 py-2 rounded-lg border border-rose-400 dark:border-rose-500/60 bg-white/97 dark:bg-[#0D0B14]/95 backdrop-blur text-[11.5px] text-slate-900 dark:text-slate-100 shadow-lg"
        >
          <div className="flex items-center gap-1.5 font-mono text-rose-700 dark:text-rose-300 pr-2 border-r border-slate-200 dark:border-slate-800">
            <AlertTriangle className="w-3.5 h-3.5 text-rose-600 dark:text-rose-400 shrink-0" />
            <span className="font-semibold">
              {lang === 'zh' ? activeCorrectionSeg.labelZh : activeCorrectionSeg.labelEn}
            </span>
            <span className="text-[10.5px] text-rose-600/80 dark:text-rose-300/80 tabular-nums">
              ΔI={activeCorrectionSeg.localContrast}
            </span>
          </div>

          <div className="flex items-center flex-wrap gap-1.5">
            {activeCorrectionSeg.objectId !== undefined && (() => {
              const targetObj = result?.objects.find((o) => o.id === activeCorrectionSeg.objectId);
              const isExcl = Boolean(targetObj?.excluded);
              return (
                <button
                  type="button"
                  onClick={() => {
                    onAddCorrection({
                      specimenId: specimen.id,
                      objectId: activeCorrectionSeg.objectId!,
                      field: 'excluded',
                      oldValue: isExcl,
                      newValue: !isExcl,
                    });
                    onTriggerLowConfCorrection(null);
                  }}
                  className="h-6 px-2 rounded bg-rose-50 dark:bg-rose-500/20 hover:bg-rose-100 dark:hover:bg-rose-500/35 border border-rose-400 dark:border-rose-500/50 text-rose-700 dark:text-rose-200 text-[11px] font-semibold cursor-pointer shadow-2xs"
                >
                  {isExcl
                    ? lang === 'zh'
                      ? '恢复目标'
                      : 'Restore'
                    : lang === 'zh'
                    ? '排除误检'
                    : 'Exclude'}
                </button>
              );
            })()}

            <button
              type="button"
              onClick={() => {
                // Expand slightly around the low-contrast box and scope ROI for localized LCN + coherence enhancement
                const pad = 16;
                const maxY = specimen.calibration.cropRows ?? specimen.height;
                const rx = Math.max(0, activeCorrectionSeg.bbox.x - pad);
                const ry = Math.max(0, activeCorrectionSeg.bbox.y - pad);
                const rw = Math.min(specimen.width - rx, activeCorrectionSeg.bbox.width + pad * 2);
                const rh = Math.min(maxY - ry, activeCorrectionSeg.bbox.height + pad * 2);
                if (rw >= 20 && rh >= 20) {
                  onChangeRoi({ x: rx, y: ry, width: rw, height: rh });
                }
                onTriggerLowConfCorrection(null);
              }}
              className="h-6 px-2 rounded bg-blue-50 dark:bg-cyan-500/15 hover:bg-blue-100 dark:hover:bg-cyan-500/30 border border-blue-400 dark:border-cyan-500/45 text-blue-700 dark:text-cyan-200 text-[11px] font-semibold flex items-center gap-1 cursor-pointer shadow-2xs"
            >
              <Crop className="w-3 h-3" />
              <span>{lang === 'zh' ? '局部 ROI 重算增强' : 'Focus ROI Re-run'}</span>
            </button>

            <button
              type="button"
              onClick={() => {
                onChangeToolMode('measure_profile');
                onTriggerLowConfCorrection(null);
              }}
              className="h-6 px-2 rounded bg-violet-50 dark:bg-violet-500/15 hover:bg-violet-100 dark:hover:bg-violet-500/30 border border-violet-400 dark:border-violet-500/45 text-violet-700 dark:text-violet-200 text-[11px] font-semibold flex items-center gap-1 cursor-pointer shadow-2xs"
            >
              <Activity className="w-3 h-3" />
              <span>{lang === 'zh' ? '沿线手动数层' : 'Line Profile'}</span>
            </button>

            <button
              type="button"
              onClick={() => {
                onChangeToolMode('brush');
                onTriggerLowConfCorrection(null);
              }}
              className="h-6 px-2 rounded border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-800 dark:text-slate-200 text-[11px] font-medium flex items-center gap-1 cursor-pointer shadow-2xs"
            >
              <Brush className="w-3 h-3 text-blue-600 dark:text-cyan-400" />
              <span>{lang === 'zh' ? '画笔标注纠偏' : 'Brush Label'}</span>
            </button>

            <button
              type="button"
              onClick={() => onDismissLowConfSegment(activeCorrectionSeg.id)}
              className="h-6 px-2 rounded border border-emerald-500/70 dark:border-emerald-500/45 bg-emerald-50 dark:bg-emerald-500/15 hover:bg-emerald-100 dark:hover:bg-emerald-500/25 text-emerald-700 dark:text-emerald-300 text-[11px] font-semibold flex items-center gap-1 cursor-pointer shadow-2xs"
            >
              <Check className="w-3 h-3" />
              <span>{lang === 'zh' ? '确认无误' : 'Confirm OK'}</span>
            </button>

            <button
              type="button"
              onClick={() => onTriggerLowConfCorrection(null)}
              title={lang === 'zh' ? '关闭纠偏条' : 'Close'}
              className="h-6 w-6 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 flex items-center justify-center cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* Semi-Auto Guided Confirmation Bar (Order: ① Calibrate -> ② Crop Info Bar -> ③ Run Workflow) */}
      {!result && (
        <div className="absolute left-1/2 -translate-x-1/2 bottom-4 z-30 max-w-[95%] flex flex-wrap items-center gap-2 px-3 py-2 rounded-lg border border-amber-400 dark:border-amber-500/50 bg-white/97 dark:bg-[#0B0E17]/95 backdrop-blur text-[11.5px] text-slate-900 dark:text-slate-100 shadow-lg">
          <span className="font-mono text-[10.5px] font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-300 pr-2 border-r border-slate-200 dark:border-slate-800 shrink-0">
            {lang === 'zh' ? '半自动待确认' : 'Semi-Auto Setup'}
          </span>

          {/* Step 1: Physical Calibration FIRST (while scale bar in info bar is still visible!) */}
          {!specimen.calibration.confirmed ? (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500 dark:text-slate-400 font-mono text-[11px]">①</span>
              {onConfirmCalibration && (
                <button
                  type="button"
                  onClick={onConfirmCalibration}
                  className="h-6 px-2.5 rounded bg-amber-500 hover:bg-amber-600 dark:hover:bg-amber-400 text-white dark:text-slate-950 text-[11px] font-semibold shadow-xs flex items-center gap-1 cursor-pointer"
                >
                  <span>
                    {lang === 'zh'
                      ? `确认标定 (${specimen.calibration.pixelSize} ${specimen.calibration.spaceUnit}/px)`
                      : `Confirm ${specimen.calibration.pixelSize} ${specimen.calibration.spaceUnit}/px`}
                  </span>
                </button>
              )}
              <button
                type="button"
                onClick={() =>
                  onChangeToolMode(toolMode === 'calibrate_line' ? 'pan' : 'calibrate_line')
                }
                className={`h-6 px-2 rounded border text-[11px] font-medium flex items-center gap-1 cursor-pointer transition-colors shadow-2xs ${
                  toolMode === 'calibrate_line'
                    ? 'border-emerald-600 dark:border-emerald-400 bg-emerald-600 dark:bg-emerald-500 text-white dark:text-slate-950 font-semibold'
                    : 'border-emerald-500/70 dark:border-emerald-500/50 bg-emerald-50 dark:bg-emerald-500/15 hover:bg-emerald-100 dark:hover:bg-emerald-500/25 text-emerald-700 dark:text-emerald-200'
                }`}
              >
                <MoveHorizontal className="w-3 h-3" />
                <span>{lang === 'zh' ? '手动描尺' : 'Trace Scale Bar'}</span>
              </button>
            </div>
          ) : (
            <span className="flex items-center gap-1 text-emerald-700 dark:text-emerald-400 font-mono font-medium text-[11px]">
              <CheckCircle2 className="w-3 h-3" />
              <span>
                ① {specimen.calibration.pixelSize} {specimen.calibration.spaceUnit}/px
              </span>
            </span>
          )}

          <span className="text-slate-400 dark:text-slate-600">→</span>

          {/* Step 2: Info Bar Crop SECOND (after reading/tracing scale bar) */}
          {importRep?.cropStatus === 'pending' && onConfirmInfoBarCrop ? (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500 dark:text-slate-400 font-mono text-[11px]">②</span>
              {(pendingCropDraft?.cropPx ?? 0) > 0 && pendingCropDraft?.barPosition !== 'none' ? (
                <>
                  <button
                    type="button"
                    onClick={() =>
                      onConfirmInfoBarCrop(pendingCropDraft!.cropPx, pendingCropDraft!.barPosition)
                    }
                    className={`h-6 px-2.5 rounded text-[11px] font-semibold flex items-center gap-1 cursor-pointer shadow-2xs ${
                      specimen.calibration.confirmed
                        ? 'bg-amber-500 hover:bg-amber-600 dark:hover:bg-amber-400 text-white dark:text-slate-950'
                        : 'border border-amber-500/70 dark:border-amber-500/50 bg-amber-50 dark:bg-amber-500/15 hover:bg-amber-100 dark:hover:bg-amber-500/25 text-amber-800 dark:text-amber-200'
                    }`}
                  >
                    <Scissors className="w-3 h-3" />
                    <span>
                      {lang === 'zh'
                        ? `裁切信息栏 ${pendingCropDraft!.cropPx}px`
                        : `Crop Bar ${pendingCropDraft!.cropPx}px`}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onConfirmInfoBarCrop(0, 'none')}
                    className="h-6 px-2 rounded border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 text-[11px] font-medium cursor-pointer"
                  >
                    {lang === 'zh' ? '跳过' : 'Skip'}
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={() => onConfirmInfoBarCrop(0, 'none')}
                  className="h-6 px-2 rounded border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 text-[11px] font-medium cursor-pointer"
                >
                  {lang === 'zh' ? '确认无需裁切' : 'No Crop Needed'}
                </button>
              )}
            </div>
          ) : (
            <span className="flex items-center gap-1 text-emerald-700 dark:text-emerald-400 font-mono font-medium text-[11px]">
              <CheckCircle2 className="w-3 h-3" />
              <span>
                {importRep?.scaleBarCropped
                  ? lang === 'zh'
                    ? `② 已裁切 ${importRep.croppedBarPx}px`
                    : `② Cropped ${importRep.croppedBarPx}px`
                  : lang === 'zh'
                  ? '② 原图视场'
                  : '② Full FOV'}
              </span>
            </span>
          )}

          <span className="text-slate-400 dark:text-slate-600">→</span>

          {/* Step 3: Explicit Workflow Execution */}
          {onRunActiveWorkflow && (
            <button
              type="button"
              onClick={onRunActiveWorkflow}
              className="h-6 px-2.5 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[11px] font-semibold shadow-xs flex items-center gap-1 cursor-pointer"
            >
              <Play className="w-3 h-3 fill-current" />
              <span>
                {lang === 'zh'
                  ? `③ 执行「${activeRecipe.isCustom ? activeRecipe.name : activeRecipe.nameZh}」`
                  : `③ Run "${activeRecipe.isCustom ? activeRecipe.name : activeRecipe.nameEn}"`}
              </span>
            </button>
          )}
        </div>
      )}

      {/* Live Computation & Stage HUD (bottom-left inside stage, updates in real time on every parameter change) */}
      {result && (
        <div className="pointer-events-none absolute left-3 bottom-3 z-20 flex items-center gap-2 px-2.5 py-1 rounded-md border border-slate-300 dark:border-slate-800/90 bg-white/92 dark:bg-[#0B0E17]/88 shadow-sm backdrop-blur font-mono text-[11px] tabular-nums text-slate-800 dark:text-slate-200">
          <span className="text-blue-600 dark:text-cyan-400">●</span>
          <span>
            {isRawSource
              ? lang === 'zh'
                ? '0. 原始图像（未裁切）'
                : '0. Original Image'
              : activePreview
              ? `${result.stepPreviews.findIndex((p) => p.stepId === activePreview.stepId) + 1}. ${
                  lang === 'zh' ? activePreview.labelZh : activePreview.labelEn
                }`
              : lang === 'zh'
              ? '实时计算结果'
              : 'Live Result'}
          </span>
          {!isRawSource && !isCropPreview && result.summary[0] && (
            <>
              <span className="text-slate-400 dark:text-slate-600">·</span>
              <span className="text-slate-700 dark:text-slate-300">
                {lang === 'zh' ? result.summary[0].labelZh : result.summary[0].labelEn}:{' '}
                <strong className="text-blue-700 dark:text-cyan-300">{String(result.summary[0].value ?? '—')}</strong>
                {result.summary[0].unit ? ` ${result.summary[0].unit}` : ''}
              </span>
            </>
          )}
          {!isRawSource && !isCropPreview && result.summary[1] && (
            <>
              <span className="text-slate-400 dark:text-slate-600 hidden sm:inline">·</span>
              <span className="text-slate-600 dark:text-slate-400 hidden sm:inline">
                {lang === 'zh' ? result.summary[1].labelZh : result.summary[1].labelEn}:{' '}
                <strong className="text-slate-900 dark:text-slate-200">{String(result.summary[1].value ?? '—')}</strong>
                {result.summary[1].unit ? ` ${result.summary[1].unit}` : ''}
              </span>
            </>
          )}
          <span className="text-slate-400 dark:text-slate-600">·</span>
          <span className="text-slate-500">{result.durationMs.toFixed(1)} ms</span>
        </div>
      )}
      </div>

      {/* Bottom Telemetry Strip */}
      <div className="h-7 shrink-0 border-t border-slate-200 dark:border-slate-800/90 bg-white dark:bg-[#0B0E17] px-3 flex items-center justify-between text-[11px] font-mono tabular-nums text-slate-600 dark:text-slate-400">
        <div className="flex items-center gap-3 min-w-0 truncate">
          <span className="text-slate-900 dark:text-slate-300 font-medium truncate">{specimen.name}</span>
          <span>
            {viewWidth}×{viewHeight} px
          </span>
          {roi && (
            <span className="text-blue-700 dark:text-cyan-400">
              ROI {roi.width}×{roi.height}
            </span>
          )}
          {!result ? (
            <span className="text-amber-600 dark:text-amber-400">
              {lang === 'zh' ? '半自动 · 原始图像（未自动分析）' : 'Semi-Auto · Raw Image (Not Executed)'}
            </span>
          ) : isRawSource ? (
            <span className="text-amber-600 dark:text-amber-400">
              0. {t('canvas_source', lang)}
            </span>
          ) : activePreview ? (
            <span className="text-blue-700 dark:text-cyan-400">
              {t('canvas_step_preview', lang)}: {lang === 'zh' ? activePreview.labelZh : activePreview.labelEn}
              {activePreview.probMap ? ' (ProbMap)' : ''}
            </span>
          ) : (
            <span className="text-emerald-700 dark:text-emerald-400">
              {lang === 'zh' ? '实时叠加结果' : 'Live Overlay Result'}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 shrink-0">
          {cursorInfo ? (
            <span className="flex items-center gap-1.5 text-slate-900 dark:text-slate-200">
              <Crosshair className="w-3 h-3 text-blue-600 dark:text-cyan-400" />
              <span>
                ({(cursorInfo.x * specimen.calibration.pixelSize).toFixed(1)},{' '}
                {(cursorInfo.y * specimen.calibration.pixelSize).toFixed(1)}) {specimen.calibration.spaceUnit}
              </span>
              <span className="text-slate-400 dark:text-slate-500">·</span>
              <span>I={cursorInfo.intensity.toFixed(0)}</span>
            </span>
          ) : (
            <span className="text-slate-500">
              {specimen.calibration.pixelSize} {specimen.calibration.spaceUnit}/px
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
