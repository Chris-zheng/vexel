import { useEffect, useMemo, useRef, useState } from 'react';
import type React from 'react';
import { useClickOutside } from '../hooks/useClickOutside';
import {
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  ArrowUp,
  Box,
  Brain,
  Brush,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Copy,
  Crop,
  Crosshair,
  FileCode2,
  FileJson,
  GitCompareArrows,
  Microscope,
  MoveHorizontal,
  Play,
  Plus,
  RotateCcw,
  ScanSearch,
  Scissors,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  Upload,
  Wand2,
  Wrench,
  X,
} from 'lucide-react';
import type {
  AnnotationStroke,
  Calibration,
  HumanCorrection,
  Lang,
  LowConfidenceSegment,
  ManualMeasurement,
  ModelCard,
  ModelOcclusionMode,
  OccludedDisplayMode,
  OperatorSpec,
  PlanProposal,
  PromptPoint,
  RoiRect,
  RunResult,
  RunVersionSnapshot,
  Specimen,
  ToolMode,
  TrainingEpochPoint,
  UnifiedRecipe,
  VolumeAnalysisResult,
  WorkflowStep,
} from '../types/vexel';
import {
  AUTO_CROP_STEP_ID,
  BUILTIN_RECIPES,
  computeSegmentedResultsBoundingBox,
  OPERATOR_CATALOG,
  UNRECOGNIZED_MANUAL_RECIPE_ID,
} from '../engine/operators';
import {
  MICROSCOPY_FORMAT_REGISTRY,
} from '../engine/microscopyDecoder';
import {
  trainDeepFewShotModelAsync,
  type DeepTrainProgressEvent,
  type TrainedLocalModel,
} from '../engine/mlRuntime';
import { buildMethodsParagraphs } from '../engine/exporters';
import { t } from '../i18n/strings';
import {
  DistributionHistogram,
  IntensityProfileChart,
  PoissonComparisonChart,
} from './ScientificCharts';
import { NcRatioHistogramPanel } from './NcRatioHistogramPanel';
import { Virtual3DPreviewWidget } from './Virtual3DPreviewWidget';

export type WorkbenchStage = 'import' | 'workflow' | 'model' | 'results';
export type ResultsSubTab = 'metrics' | 'review' | 'provenance';

export interface SuggestedParamFix {
  stepId: string;
  key?: string;
  value?: number | boolean | string;
  toggleEnable?: boolean;
  summaryZh: string;
  summaryEn: string;
}

export interface CorrectionSuggestion {
  id: string;
  severity: 'high' | 'medium';
  anomalyZh: string;
  anomalyEn: string;
  titleZh: string;
  titleEn: string;
  detailZh: string;
  detailEn: string;
  corrections: Array<Omit<HumanCorrection, 'id' | 'timestamp'>>;
  paramFix?: SuggestedParamFix;
  affectedObjectIds: number[];
}

const PROCESSING_CATEGORIES: Array<{
  id: OperatorSpec['category'];
  labelZh: string;
  labelEn: string;
}> = [
  { id: 'preprocess', labelZh: '预处理与去噪', labelEn: 'Preprocess & Denoise' },
  { id: 'segment', labelZh: '图像分割与检测', labelEn: 'Segmentation & Detection' },
  { id: 'model', labelZh: 'AI 训练模型推理', labelEn: 'Trained Model Inference' },
  { id: 'filter', labelZh: '形态过滤与筛选', labelEn: 'Morphological Filtering' },
  { id: 'metrology', labelZh: '定量量测与分析', labelEn: 'Quantitative Metrology' },
];

const BRUSH_CLASSES = [
  { id: 1, color: '#06B6D4', zh: '1. 前景目标', en: '1. Foreground' },
  { id: 2, color: '#F43F5E', zh: '2. 背景基体', en: '2. Background' },
  { id: 3, color: '#F59E0B', zh: '3. 缺陷/异物', en: '3. Defect' },
];

/**
 * Inspects RunResult metadata (objects, QC warnings, summary metrics, and active recipe steps)
 * to suggest concrete HumanCorrection entries and one-click noise/threshold adjustments.
 */
export function suggestCorrectionsFromAnomalies(
  specimen: Specimen,
  recipe: UnifiedRecipe,
  result: RunResult
): CorrectionSuggestion[] {
  const suggestions: CorrectionSuggestion[] = [];
  const activeObjects = result.objects.filter((o) => !o.excluded);
  const coveredExcludeIds = new Set<number>();

  // 1. SEM LER/LWR denoise bias anomaly
  const denoiseStep = recipe.steps.find((s) => s.enabled && s.op === 'sem.denoise');
  const hasLinesStep = recipe.steps.some((s) => s.enabled && s.op === 'sem.lines');
  if (denoiseStep && hasLinesStep) {
    suggestions.push({
      id: 'fix_ler_denoise_bias',
      severity: 'high',
      anomalyZh: 'LER/LWR 去噪偏差',
      anomalyEn: 'LER/LWR Denoise Bias',
      titleZh: '关闭粗糙度测量前的高斯去噪',
      titleEn: 'Disable Gaussian denoise before LER/LWR',
      detailZh: '平滑滤波会人为压低线边缘粗糙度 3σ 数值，建议关闭 sem.denoise 以恢复无偏测量。',
      detailEn: 'Gaussian smoothing artificially suppresses 3σ LER/LWR; disable sem.denoise for unbiased edges.',
      corrections: [],
      paramFix: {
        stepId: denoiseStep.id,
        toggleEnable: true,
        summaryZh: 'sem.denoise → 关闭',
        summaryEn: 'sem.denoise → OFF',
      },
      affectedObjectIds: [],
    });
  }

  // 2. Microfluidics droplet size outliers
  const dropletStep = recipe.steps.find((s) => s.enabled && s.op === 'droplet.analyze');
  const sizeOutliers = activeObjects.filter(
    (o) => o.kind === 'droplet' && o.flagged && /Size outlier|粒径异常/i.test(o.flagReason ?? '')
  );
  if (sizeOutliers.length > 0) {
    sizeOutliers.forEach((o) => coveredExcludeIds.add(o.id));
    suggestions.push({
      id: 'fix_droplet_size_outliers',
      severity: 'high',
      anomalyZh: `粒径离群 (${sizeOutliers.length})`,
      anomalyEn: `Size Outlier (${sizeOutliers.length})`,
      titleZh: `排除 ${sizeOutliers.length} 个异常尺寸液滴`,
      titleEn: `Exclude ${sizeOutliers.length} oversized/outlier droplet(s)`,
      detailZh: `目标 ${sizeOutliers.map((o) => `#${o.id}`).join(', ')} 偏离均值超过 2σ，排除后可修正粒径 CV 与平均体积。`,
      detailEn: `Object(s) ${sizeOutliers.map((o) => `#${o.id}`).join(', ')} deviate >2σ from mean; excluding restores unbiased CV.`,
      corrections: sizeOutliers.map((o) => ({
        specimenId: specimen.id,
        objectId: o.id,
        field: 'excluded',
        oldValue: false,
        newValue: true,
      })),
      affectedObjectIds: sizeOutliers.map((o) => o.id),
    });
  }

  // 3. Microfluidics high cell count / noise-contrast threshold anomaly
  const highCellDrops = activeObjects.filter(
    (o) => o.kind === 'droplet' && o.cells !== undefined && o.cells >= 3
  );
  if (dropletStep && highCellDrops.length > 0) {
    const curContrast = Number(dropletStep.params.cellContrast ?? 36);
    const nextContrast = Math.min(120, curContrast + 12);
    if (nextContrast > curContrast) {
      suggestions.push({
        id: 'fix_cell_noise_threshold',
        severity: 'medium',
        anomalyZh: `高包裹/暗噪点 (${highCellDrops.length})`,
        anomalyEn: `High Cell / Noise (${highCellDrops.length})`,
        titleZh: '提高细胞对比度噪声阈值',
        titleEn: 'Raise cell noise contrast threshold',
        detailZh: `检出 ${highCellDrops.length} 个 ≥3 细胞液滴，提高对比度阈值可滤除液滴暗环与微小杂质误计。`,
        detailEn: `${highCellDrops.length} droplet(s) have ≥3 cells; raising contrast suppresses optical-ring noise spots.`,
        corrections: highCellDrops.map((o) => ({
          specimenId: specimen.id,
          objectId: o.id,
          field: 'cells',
          oldValue: o.cells,
          newValue: Math.max(0, (o.cells ?? 1) - 1),
        })),
        paramFix: {
          stepId: dropletStep.id,
          key: 'cellContrast',
          value: nextContrast,
          summaryZh: `cellContrast: ${curContrast} → ${nextContrast}`,
          summaryEn: `cellContrast: ${curContrast} → ${nextContrast}`,
        },
        affectedObjectIds: highCellDrops.map((o) => o.id),
      });
    }
  }

  // 4. Microfluidics borderline fluorescence threshold
  const nearFluoDrops = activeObjects.filter(
    (o) =>
      o.kind === 'droplet' &&
      o.positive !== undefined &&
      o.flagged &&
      /Near threshold|近阳性阈值/i.test(o.flagReason ?? '')
  );
  if (nearFluoDrops.length > 0 && dropletStep) {
    const avgFluo =
      nearFluoDrops.reduce((acc, o) => acc + (o.fluoIntensity ?? 90), 0) / nearFluoDrops.length;
    const suggestedThresh = Math.min(240, Math.max(20, Math.round((avgFluo + 10) / 5) * 5));
    const curThresh = Number(dropletStep.params.fluoThreshold ?? 0);
    suggestions.push({
      id: 'fix_fluo_borderline',
      severity: 'medium',
      anomalyZh: `荧光近阈值 (${nearFluoDrops.length})`,
      anomalyEn: `Borderline Fluo (${nearFluoDrops.length})`,
      titleZh: '校准荧光阳性噪声阈值并复核边界液滴',
      titleEn: 'Tighten fluorescence noise threshold & call borderline drops',
      detailZh: `目标 ${nearFluoDrops.map((o) => `#${o.id}`).join(', ')} 荧光强度贴近自动阈值，锁定噪声阈值并标记边界弱信号为阴性。`,
      detailEn: `Object(s) ${nearFluoDrops.map((o) => `#${o.id}`).join(', ')} sit near auto-Otsu cutoff; locks threshold above noise floor.`,
      corrections: nearFluoDrops
        .filter((o) => o.positive)
        .map((o) => ({
          specimenId: specimen.id,
          objectId: o.id,
          field: 'positive',
          oldValue: true,
          newValue: false,
        })),
      paramFix:
        curThresh !== suggestedThresh
          ? {
              stepId: dropletStep.id,
              key: 'fluoThreshold',
              value: suggestedThresh,
              summaryZh: `fluoThreshold: ${curThresh === 0 ? 'Auto' : curThresh} → ${suggestedThresh}`,
              summaryEn: `fluoThreshold: ${curThresh === 0 ? 'Auto' : curThresh} → ${suggestedThresh}`,
            }
          : undefined,
      affectedObjectIds: nearFluoDrops.map((o) => o.id),
    });
  }

  // 5. SEM Particle outliers
  const particleStep = recipe.steps.find((s) => s.enabled && s.op === 'sem.particles');
  const particleOutliers = activeObjects.filter((o) => o.kind === 'particle' && o.cls === 'outlier');
  if (particleOutliers.length > 0) {
    particleOutliers.forEach((o) => coveredExcludeIds.add(o.id));
    const curMinDia = particleStep ? Number(particleStep.params.minDiameterPx ?? 10) : 10;
    const nextMinDia = Math.min(80, curMinDia + 4);
    suggestions.push({
      id: 'fix_sem_particle_outliers',
      severity: 'high',
      anomalyZh: `异常颗粒/异物 (${particleOutliers.length})`,
      anomalyEn: `Particle Outlier (${particleOutliers.length})`,
      titleZh: `排除 ${particleOutliers.length} 个异物颗粒并收紧尺寸下限`,
      titleEn: `Exclude ${particleOutliers.length} outlier debris & raise noise floor`,
      detailZh: `目标 ${particleOutliers.map((o) => `#${o.id}`).join(', ')} 被判为离群异物，排除后可防止体积分布 d90 偏移。`,
      detailEn: `Object(s) ${particleOutliers.map((o) => `#${o.id}`).join(', ')} flagged as outliers; excluding prevents d90 skew.`,
      corrections: particleOutliers.map((o) => ({
        specimenId: specimen.id,
        objectId: o.id,
        field: 'excluded',
        oldValue: false,
        newValue: true,
      })),
      paramFix:
        particleStep && nextMinDia > curMinDia
          ? {
              stepId: particleStep.id,
              key: 'minDiameterPx',
              value: nextMinDia,
              summaryZh: `minDiameterPx: ${curMinDia} → ${nextMinDia} px`,
              summaryEn: `minDiameterPx: ${curMinDia} → ${nextMinDia} px`,
            }
          : undefined,
      affectedObjectIds: particleOutliers.map((o) => o.id),
    });
  }

  // 6. SEM Particle agglomerates
  const flaggedAggs = activeObjects.filter((o) => o.kind === 'particle' && o.cls === 'agglomerate');
  if (flaggedAggs.length > 0) {
    flaggedAggs.forEach((o) => coveredExcludeIds.add(o.id));
    const filterStep = recipe.steps.find((s) => s.op === 'filter.objects');
    suggestions.push({
      id: 'fix_sem_exclude_agglomerates',
      severity: 'medium',
      anomalyZh: `颗粒团聚体 (${flaggedAggs.length})`,
      anomalyEn: `Agglomerates (${flaggedAggs.length})`,
      titleZh: `从单颗粒统计中排除 ${flaggedAggs.length} 个团聚体`,
      titleEn: `Exclude ${flaggedAggs.length} agglomerate(s) from primary sizing`,
      detailZh: `将粘连体 ${flaggedAggs.map((o) => `#${o.id}`).join(', ')} 移出单颗粒粒径分布统计，保留至纠错记录用于模型训练。`,
      detailEn: `Removes touching clusters from single-particle sizing while logging corrections for Active Learning.`,
      corrections: flaggedAggs.map((o) => ({
        specimenId: specimen.id,
        objectId: o.id,
        field: 'excluded',
        oldValue: false,
        newValue: true,
      })),
      paramFix:
        filterStep && filterStep.enabled && Number(filterStep.params.minCircularity ?? 0) < 0.65
          ? {
              stepId: filterStep.id,
              key: 'minCircularity',
              value: 0.65,
              summaryZh: 'minCircularity → 0.65',
              summaryEn: 'minCircularity → 0.65',
            }
          : undefined,
      affectedObjectIds: flaggedAggs.map((o) => o.id),
    });
  }

  // 7. SEM Contact holes: merged vias
  const holesStep = recipe.steps.find((s) => s.enabled && s.op === 'sem.holes');
  const mergedHoles = activeObjects.filter(
    (o) => o.kind === 'hole' && o.flagged && /Merged via|双孔粘连/i.test(o.flagReason ?? '')
  );
  if (mergedHoles.length > 0) {
    mergedHoles.forEach((o) => coveredExcludeIds.add(o.id));
    const curLevel = holesStep ? Number(holesStep.params.level ?? 0.5) : 0.5;
    const nextLevel = Math.min(0.8, Number((curLevel + 0.05).toFixed(2)));
    suggestions.push({
      id: 'fix_sem_merged_holes',
      severity: 'high',
      anomalyZh: `双孔粘连 (${mergedHoles.length})`,
      anomalyEn: `Merged Vias (${mergedHoles.length})`,
      titleZh: `排除 ${mergedHoles.length} 处粘连孔并微调边缘阈值`,
      titleEn: `Exclude ${mergedHoles.length} merged via(s) & raise edge threshold`,
      detailZh: `目标 ${mergedHoles.map((o) => `#${o.id}`).join(', ')} 疑似相邻孔粘连，排除并提高阈值以锐化孔边缘。`,
      detailEn: `Object(s) ${mergedHoles.map((o) => `#${o.id}`).join(', ')} appear merged; excluding & raising edge level improves separation.`,
      corrections: mergedHoles.map((o) => ({
        specimenId: specimen.id,
        objectId: o.id,
        field: 'excluded',
        oldValue: false,
        newValue: true,
      })),
      paramFix:
        holesStep && nextLevel > curLevel
          ? {
              stepId: holesStep.id,
              key: 'level',
              value: nextLevel,
              summaryZh: `edge level: ${curLevel} → ${nextLevel}`,
              summaryEn: `edge level: ${curLevel} → ${nextLevel}`,
            }
          : undefined,
      affectedObjectIds: mergedHoles.map((o) => o.id),
    });
  }

  // 8. Generic low-confidence detections
  const lowConfObjects = activeObjects.filter(
    (o) => o.confidence < 0.8 && !coveredExcludeIds.has(o.id) && o.cls !== 'missing'
  );
  const modelStep = recipe.steps.find((s) => s.enabled && s.op === 'model.segment');
  if (lowConfObjects.length > 0) {
    const curProb = modelStep ? Number(modelStep.params.probThreshold ?? 0.5) : 0.5;
    const nextProb = Math.min(0.85, Number((curProb + 0.1).toFixed(2)));
    suggestions.push({
      id: 'fix_low_confidence_noise',
      severity: 'medium',
      anomalyZh: `低置信目标 (${lowConfObjects.length})`,
      anomalyEn: `Low Confidence (${lowConfObjects.length})`,
      titleZh: `剔除 ${lowConfObjects.length} 个低置信噪声目标`,
      titleEn: `Exclude ${lowConfObjects.length} low-confidence detection(s)`,
      detailZh: `置信度 < 80% 的目标疑似背景噪声，建议一键排除。`,
      detailEn: `Detections below 80% confidence are likely background noise.`,
      corrections: lowConfObjects.map((o) => ({
        specimenId: specimen.id,
        objectId: o.id,
        field: 'excluded',
        oldValue: false,
        newValue: true,
      })),
      paramFix:
        modelStep && nextProb > curProb
          ? {
              stepId: modelStep.id,
              key: 'probThreshold',
              value: nextProb,
              summaryZh: `probThreshold: ${curProb} → ${nextProb}`,
              summaryEn: `probThreshold: ${curProb} → ${nextProb}`,
            }
          : undefined,
      affectedObjectIds: lowConfObjects.map((o) => o.id),
    });
  }

  // 9. Background noise when preprocess disabled
  const disabledBg = recipe.steps.find((s) => !s.enabled && s.op === 'sem.background');
  if (disabledBg && result.warnings.length > 0 && suggestions.length < 4) {
    suggestions.push({
      id: 'fix_enable_background_level',
      severity: 'medium',
      anomalyZh: '背底不均 / 噪声抑制',
      anomalyEn: 'Background Noise',
      titleZh: '启用背景校平算子以稳定分割阈值',
      titleEn: 'Enable background levelling to stabilize thresholding',
      detailZh: '开启 sem.background 可扣除光照与背底梯度干扰。',
      detailEn: 'Enabling sem.background removes illumination gradient.',
      corrections: [],
      paramFix: {
        stepId: disabledBg.id,
        toggleEnable: true,
        summaryZh: 'sem.background → 开启',
        summaryEn: 'sem.background → ON',
      },
      affectedObjectIds: [],
    });
  }

  return suggestions;
}

interface InspectorPanelProps {
  specimen: Specimen;
  specimens: Specimen[];
  onSwitchSpecimen: (id: string) => void;
  onTriggerUpload: () => void;
  onDropFile: (file: File) => void;
  autoRunOnImport: boolean;
  onToggleAutoRunOnImport: () => void;
  pendingCropDraft: { cropPx: number; barPosition: 'bottom' | 'top' | 'none' };
  manualCalLine?: { p1: [number, number]; p2: [number, number]; lengthPx: number; snapped?: boolean } | null;
  onClearManualCalLine?: () => void;
  onUpdatePendingCropDraft: (draft: { cropPx: number; barPosition: 'bottom' | 'top' | 'none' }) => void;
  onConfirmInfoBarCrop: (cropPx: number, barPosition: 'bottom' | 'top' | 'none') => void;
  onRevertInfoBarCrop: () => void;
  savedWorkflows: UnifiedRecipe[];
  onSelectWorkflow: (workflowId: string, runNow?: boolean) => void;
  onRunActiveWorkflow: () => void;
  recipe: UnifiedRecipe;
  onApplyRecipeSteps: (nextRecipe: UnifiedRecipe) => void;
  plan: PlanProposal | null;
  planAccepted: boolean;
  result: RunResult | null;
  lang: Lang;
  stage: WorkbenchStage;
  onChangeStage: (stage: WorkbenchStage) => void;
  resultsSubTab: ResultsSubTab;
  onChangeResultsSubTab: (sub: ResultsSubTab) => void;
  selectedStepId: string | null;
  onSelectStepId: (stepId: string | null) => void;
  selectedObjectId: number | null;
  onSelectObject: (id: number | null) => void;
  onConfirmCalibration: () => void;
  onUpdateCalibration: (cal: Calibration) => void;
  onAcceptPlan: () => void;
  onCustomizeWorkflow: () => void;
  onUpdateStepParam: (stepId: string, key: string, value: number | boolean | string) => void;
  onResetStepParams: (stepId: string) => void;
  onToggleStepEnabled: (stepId: string) => void;
  onChangeModelRef: (stepId: string, modelRef: string) => void;
  models: ModelCard[];
  activeModelRef?: string;
  onBindModel: (modelRef: string) => void;
  onPublishLocalModel: (trained: TrainedLocalModel) => void;
  toolMode: ToolMode;
  onChangeToolMode: (mode: ToolMode) => void;
  roi: RoiRect | null;
  onChangeRoi: (roi: RoiRect | null) => void;
  showLowConfHighlight: boolean;
  onToggleLowConfHighlight: () => void;
  lowConfThreshold: number;
  onChangeLowConfThreshold: (v: number) => void;
  occludedDisplayMode: OccludedDisplayMode;
  onChangeOccludedDisplayMode: (mode: OccludedDisplayMode) => void;
  onUpdateModelOcclusionConfig: (patch: {
    occlusionMode?: ModelOcclusionMode;
    curvatureBoundaryFit?: boolean;
    overlapConfidenceThreshold?: number;
  }) => void;
  dismissedLowConfIds: string[];
  onDismissLowConfSegment: (segId: string) => void;
  activeCorrectionSegId: string | null;
  onTriggerLowConfCorrection: (seg: LowConfidenceSegment | null) => void;
  brushClassId: number;
  onChangeBrushClassId: (cid: number) => void;
  brushRadius: number;
  onChangeBrushRadius: (r: number) => void;
  corrections: HumanCorrection[];
  onAddCorrection: (corr: Omit<HumanCorrection, 'id' | 'timestamp'>) => void;
  onDeleteCorrection: (id: string) => void;
  onClearCorrections: () => void;
  onApplySuggestedFix?: (suggestions: CorrectionSuggestion[]) => void;
  measurements: ManualMeasurement[];
  onDeleteMeasurement: (id: string) => void;
  onClearMeasurements: () => void;
  strokes: AnnotationStroke[];
  onDeleteStroke: (id: string) => void;
  onClearStrokes: () => void;
  prompts: PromptPoint[];
  onDeletePrompt: (id: string) => void;
  onClearPrompts: () => void;
  runHistory: RunVersionSnapshot[];
  onGenerateComparisonBaseline: () => void;
  onExportVexflow: () => void;
  onExportReproducePy: () => void;
  onOpen3DStudio?: () => void;
  volumeAnalysis?: VolumeAnalysisResult | null;
  isAnalyzing3D?: boolean;
  onRunQuick3DAnalysis?: () => void;
  /** Shown first in the metrics tab: the validated Python engine's result, beside this preview. */
  engineCard?: React.ReactNode;
}

export function InspectorPanel({
  specimen,
  specimens,
  onSwitchSpecimen,
  onTriggerUpload,
  onDropFile,
  autoRunOnImport,
  onToggleAutoRunOnImport,
  pendingCropDraft,
  manualCalLine = null,
  onClearManualCalLine,
  onUpdatePendingCropDraft,
  onConfirmInfoBarCrop,
  onRevertInfoBarCrop,
  savedWorkflows,
  onSelectWorkflow,
  onRunActiveWorkflow,
  recipe,
  onApplyRecipeSteps,
  plan,
  planAccepted,
  result,
  lang,
  stage,
  onChangeStage,
  resultsSubTab,
  onChangeResultsSubTab,
  selectedStepId,
  onSelectStepId,
  selectedObjectId,
  onSelectObject,
  onConfirmCalibration,
  onUpdateCalibration,
  onAcceptPlan,
  onCustomizeWorkflow,
  onUpdateStepParam,
  onResetStepParams,
  onToggleStepEnabled,
  onChangeModelRef,
  models,
  activeModelRef,
  onBindModel,
  onPublishLocalModel,
  toolMode,
  onChangeToolMode,
  roi,
  onChangeRoi,
  showLowConfHighlight,
  onToggleLowConfHighlight,
  lowConfThreshold,
  onChangeLowConfThreshold,
  occludedDisplayMode,
  onChangeOccludedDisplayMode,
  onUpdateModelOcclusionConfig,
  dismissedLowConfIds,
  onDismissLowConfSegment,
  activeCorrectionSegId,
  onTriggerLowConfCorrection,
  brushClassId,
  onChangeBrushClassId,
  brushRadius,
  onChangeBrushRadius,
  corrections,
  onAddCorrection,
  onDeleteCorrection,
  onClearCorrections,
  onApplySuggestedFix,
  measurements,
  onDeleteMeasurement,
  onClearMeasurements,
  strokes,
  onDeleteStroke,
  onClearStrokes,
  prompts,
  onDeletePrompt,
  onClearPrompts,
  runHistory,
  onGenerateComparisonBaseline,
  onExportVexflow,
  onExportReproducePy,
  onOpen3DStudio,
  volumeAnalysis,
  isAnalyzing3D = false,
  onRunQuick3DAnalysis,
  engineCard,
}: InspectorPanelProps) {
  const [editingCal, setEditingCal] = useState(false);
  const [calPx, setCalPx] = useState(String(specimen.calibration.pixelSize));
  const [calUnit, setCalUnit] = useState<Calibration['spaceUnit']>(specimen.calibration.spaceUnit);
  const [calChannelH, setCalChannelH] = useState(String(specimen.calibration.channelHeightUm ?? ''));
  const [step2Px, setStep2Px] = useState(String(specimen.calibration.pixelSize));
  const [step2Unit, setStep2Unit] = useState<Calibration['spaceUnit']>(specimen.calibration.spaceUnit);
  const [scaleBarValInput, setScaleBarValInput] = useState(
    specimen.importReport?.scaleBarValue ? String(specimen.importReport.scaleBarValue) : ''
  );
  const [scaleBarUnitInput, setScaleBarUnitInput] = useState<Calibration['spaceUnit']>(
    specimen.importReport?.scaleBarUnit ?? specimen.calibration.spaceUnit
  );
  const [isDragOver, setIsDragOver] = useState(false);
  const [showFormatMatrix, setShowFormatMatrix] = useState(false);
  const [inlineTraining, setInlineTraining] = useState(false);
  const [inlineProgress, setInlineProgress] = useState<DeepTrainProgressEvent | null>(null);
  const [inlineCurve, setInlineCurve] = useState<TrainingEpochPoint[]>([]);
  const inlineAbortRef = useRef<AbortController | null>(null);
  const [metricsDim, setMetricsDim] = useState<'2d' | '3d'>('2d');

  useEffect(() => {
    setStep2Px(String(specimen.calibration.pixelSize));
    setStep2Unit(specimen.calibration.spaceUnit);
    setScaleBarValInput(
      specimen.importReport?.scaleBarValue ? String(specimen.importReport.scaleBarValue) : ''
    );
    setScaleBarUnitInput(specimen.importReport?.scaleBarUnit ?? specimen.calibration.spaceUnit);
  }, [
    specimen.id,
    specimen.calibration.pixelSize,
    specimen.calibration.spaceUnit,
    specimen.importReport?.scaleBarValue,
    specimen.importReport?.scaleBarUnit,
  ]);

  // Whenever user traces a manual scale bar on canvas, auto-recompute pixelSize if a printed bar value is entered
  useEffect(() => {
    if (!manualCalLine || manualCalLine.lengthPx <= 0) return;
    const num = Number(scaleBarValInput);
    if (Number.isFinite(num) && num > 0) {
      setStep2Px(String(Number((num / manualCalLine.lengthPx).toPrecision(4))));
      setStep2Unit(scaleBarUnitInput);
    }
  }, [manualCalLine, scaleBarValInput, scaleBarUnitInput]);

  // Bounding box of all active segmented results with 10% padding for "Focus on Results" ROI control
  const segmentedResultsRoi = useMemo(
    () => computeSegmentedResultsBoundingBox(result, specimen, 0.1),
    [result, specimen]
  );

  // Stage 2: Inline workflow editor & progressive parameter disclosure
  const [showReasons, setShowReasons] = useState(false);
  const [openParamStepId, setOpenParamStepId] = useState<string | null>(() => {
    const primary = recipe.steps.find(
      (s) => s.op === 'sem.particles' || s.op === 'sem.lines' || OPERATOR_CATALOG.find((o) => o.op === s.op)?.category === 'metrology'
    );
    return primary?.id ?? null;
  });
  useEffect(() => {
    const primary = recipe.steps.find(
      (s) => s.op === 'sem.particles' || s.op === 'sem.lines' || OPERATOR_CATALOG.find((o) => o.op === s.op)?.category === 'metrology'
    );
    setOpenParamStepId(primary?.id ?? null);
  }, [recipe.id]);
  const [addCategory, setAddCategory] = useState<OperatorSpec['category']>('preprocess');
  const [addOpId, setAddOpId] = useState<string>('sem.background');

  // Stage 3: Side-by-side Active Learning Re-Training state
  const [modelName, setModelName] = useState('lab_tuned_seg');
  const [modelVer, setModelVer] = useState('1.0.0');
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [lastTrained, setLastTrained] = useState<TrainedLocalModel | null>(null);

  // Stage 4: Results, Review, Compare & Provenance
  const [reviewFilter, setReviewFilter] = useState<'all' | 'flagged'>('all');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [compareIdA, setCompareIdA] = useState<string | null>(null);
  const [compareIdB, setCompareIdB] = useState<string | null>(null);

  const calTriggerRef = useRef<HTMLButtonElement>(null);
  const calDrawerRef = useRef<HTMLDivElement>(null);
  useClickOutside(editingCal, [calTriggerRef, calDrawerRef], () => setEditingCal(false));

  const openCalEditor = () => {
    if (editingCal) {
      setEditingCal(false);
      return;
    }
    setCalPx(String(specimen.calibration.pixelSize));
    setCalUnit(specimen.calibration.spaceUnit);
    setCalChannelH(String(specimen.calibration.channelHeightUm ?? ''));
    setEditingCal(true);
  };

  const saveCalibration = () => {
    const num = Number(calPx);
    if (!Number.isFinite(num) || num <= 0) return;
    const chNum = Number(calChannelH);
    onUpdateCalibration({
      ...specimen.calibration,
      pixelSize: num,
      spaceUnit: calUnit,
      channelHeightUm: Number.isFinite(chNum) && chNum > 0 ? chNum : undefined,
      source: 'manual',
      confirmed: true,
    });
    setEditingCal(false);
  };

  const specCorrections = useMemo(
    () => corrections.filter((c) => c.specimenId === specimen.id),
    [corrections, specimen.id]
  );

  const methods = useMemo(
    () => (result ? buildMethodsParagraphs(specimen, result) : null),
    [specimen, result]
  );

  const latestProfileMeasurement = useMemo(() => {
    for (let i = measurements.length - 1; i >= 0; i--) {
      if (measurements[i].profile) return measurements[i];
    }
    return null;
  }, [measurements]);

  const filteredObjects = useMemo(() => {
    if (!result) return [];
    if (reviewFilter === 'flagged') {
      return result.objects.filter((o) => o.flagged || o.excluded);
    }
    return result.objects;
  }, [result, reviewFilter]);

  const flaggedCount = useMemo(
    () => (result ? result.objects.filter((o) => o.flagged).length : 0),
    [result]
  );

  const anomalySuggestions = useMemo(
    () => (result ? suggestCorrectionsFromAnomalies(specimen, recipe, result) : []),
    [specimen, recipe, result]
  );

  const opsInCategory = useMemo(
    () => OPERATOR_CATALOG.filter((o) => o.category === addCategory),
    [addCategory]
  );

  const applySuggestions = (list: CorrectionSuggestion[]) => {
    if (list.length === 0) return;
    if (onApplySuggestedFix) {
      onApplySuggestedFix(list);
      return;
    }
    for (const s of list) {
      for (const c of s.corrections) onAddCorrection(c);
      if (s.paramFix) {
        if (s.paramFix.toggleEnable) onToggleStepEnabled(s.paramFix.stepId);
        else if (s.paramFix.key !== undefined && s.paramFix.value !== undefined) {
          onUpdateStepParam(s.paramFix.stepId, s.paramFix.key, s.paramFix.value);
        }
      }
    }
  };

  // Inline Workflow Step Mutations (Side-by-Side with live canvas!)
  const handleAddStepInline = () => {
    const spec =
      opsInCategory.find((o) => o.op === addOpId) ?? opsInCategory[0];
    if (!spec) return;
    const defaults: Record<string, number | boolean | string> = {};
    for (const p of spec.params) defaults[p.key] = p.defaultValue;
    const newStep: WorkflowStep = {
      id: `${spec.op.replace(/\./g, '_')}_${Date.now().toString(36).slice(-4)}`,
      op: spec.op,
      enabled: true,
      params: defaults,
      modelRef: spec.op === 'model.segment' ? models[0]?.ref : undefined,
    };
    const nextSteps = [...recipe.steps];
    if (spec.category === 'preprocess') {
      nextSteps.unshift(newStep);
    } else if (spec.category === 'model' || spec.category === 'segment') {
      const firstMetrology = nextSteps.findIndex(
        (s) => OPERATOR_CATALOG.find((o) => o.op === s.op)?.category === 'metrology'
      );
      if (firstMetrology >= 0) nextSteps.splice(firstMetrology, 0, newStep);
      else nextSteps.push(newStep);
    } else {
      nextSteps.push(newStep);
    }
    onApplyRecipeSteps({ ...recipe, steps: nextSteps });
    setOpenParamStepId(newStep.id);
  };

  const handleRemoveStepInline = (stepId: string) => {
    const nextSteps = recipe.steps.filter((s) => s.id !== stepId);
    onApplyRecipeSteps({ ...recipe, steps: nextSteps });
    if (selectedStepId === stepId) onSelectStepId(null);
    if (openParamStepId === stepId) setOpenParamStepId(null);
  };

  const handleMoveStepInline = (idx: number, dir: -1 | 1) => {
    const target = idx + dir;
    if (target < 0 || target >= recipe.steps.length) return;
    const nextSteps = [...recipe.steps];
    [nextSteps[idx], nextSteps[target]] = [nextSteps[target], nextSteps[idx]];
    onApplyRecipeSteps({ ...recipe, steps: nextSteps });
  };

  // Side-by-side Active Learning training directly alongside ScientificCanvas (non-blocking Web Worker DL)
  const handleTrainSideBySide = async () => {
    setTrainingError(null);
    const totalSamples = strokes.length + specCorrections.length;
    if (totalSamples === 0) {
      setTrainingError(
        lang === 'zh'
          ? '请先点击下方「开启画布画笔」，在左侧图像上涂抹前景与背景样本（或在结果复核中修改目标标签）。'
          : 'Activate Brush on Canvas below and paint foreground/background strokes on the image first (or record object corrections).'
      );
      return;
    }
    const cleanId =
      modelName.trim().replace(/[^a-zA-Z0-9_-]/g, '_') || 'lab_tuned_seg';
    const cleanVer = modelVer.trim() || '1.0.0';
    const controller = new AbortController();
    inlineAbortRef.current = controller;
    setInlineTraining(true);
    setInlineProgress(null);
    setInlineCurve([]);
    try {
      const trained = await trainDeepFewShotModelAsync(
        cleanId,
        cleanVer,
        specimen,
        strokes,
        specCorrections,
        result?.objects ?? [],
        ['foreground', 'background', 'defect'],
        {
          workingGray: result?.preModelGray ?? result?.processedGray,
          roi,
          config: {
            hiddenDim: 24,
            epochs: 28,
            learningRate: 0.025,
            hardNegativeMining: true,
            bilateralSmooth: true,
          },
          signal: controller.signal,
          onProgress: (ev) => {
            setInlineProgress(ev);
            if (ev.phase === 'training') {
              setInlineCurve((prev) => [
                ...prev,
                {
                  epoch: ev.epoch,
                  trainLoss: ev.trainLoss,
                  valLoss: ev.valLoss,
                  valMeanIoU: ev.valMeanIoU,
                  valMacroF1: ev.valMacroF1,
                },
              ]);
            }
          },
        }
      );
      setLastTrained(trained);
    } catch (err) {
      setTrainingError(
        err instanceof Error
          ? err.message
          : lang === 'zh'
            ? '训练样本类别不足：请至少标注「前景」和「背景」两类画笔样本。'
            : 'Need at least 2 classes (Foreground + Background) to train and evaluate hold-out metrics.'
      );
    } finally {
      setInlineTraining(false);
      inlineAbortRef.current = null;
    }
  };

  const handleAbortInlineTraining = () => {
    inlineAbortRef.current?.abort();
  };

  // Version comparison snapshots
  const versionB: RunVersionSnapshot | null = useMemo(() => {
    if (runHistory.length === 0) return null;
    const found = compareIdB ? runHistory.find((h) => h.id === compareIdB) : null;
    return found ?? runHistory[runHistory.length - 1];
  }, [runHistory, compareIdB]);

  const versionA: RunVersionSnapshot | null = useMemo(() => {
    if (runHistory.length === 0) return null;
    const found = compareIdA ? runHistory.find((h) => h.id === compareIdA) : null;
    if (found) return found;
    return runHistory.length >= 2 ? runHistory[runHistory.length - 2] : runHistory[0];
  }, [runHistory, compareIdA]);

  const compareMetricRows = useMemo(() => {
    if (!versionA || !versionB) return [];
    const mapA = new Map(versionA.result.summary.map((m) => [m.key, m]));
    const mapB = new Map(versionB.result.summary.map((m) => [m.key, m]));
    const keys: string[] = [];
    for (const m of versionA.result.summary) keys.push(m.key);
    for (const m of versionB.result.summary) {
      if (!keys.includes(m.key)) keys.push(m.key);
    }
    return keys.map((key) => {
      const mA = mapA.get(key);
      const mB = mapB.get(key);
      const valA = mA?.value ?? null;
      const valB = mB?.value ?? null;
      const numA = typeof valA === 'number' && Number.isFinite(valA) ? valA : null;
      const numB = typeof valB === 'number' && Number.isFinite(valB) ? valB : null;
      let deltaNum: number | null = null;
      let deltaPct: number | null = null;
      let changed = false;
      if (numA !== null && numB !== null) {
        deltaNum = Number((numB - numA).toFixed(3));
        changed = Math.abs(deltaNum) > 1e-6;
        if (Math.abs(numA) > 1e-6) {
          deltaPct = Number((((numB - numA) / Math.abs(numA)) * 100).toFixed(1));
        }
      } else {
        changed = String(valA ?? '') !== String(valB ?? '');
      }
      return {
        key,
        labelZh: mB?.labelZh ?? mA?.labelZh ?? key,
        labelEn: mB?.labelEn ?? mA?.labelEn ?? key,
        unit: mB?.unit ?? mA?.unit ?? '',
        valA,
        valB,
        deltaNum,
        deltaPct,
        changed,
      };
    });
  }, [versionA, versionB]);

  const compareRecipeDiffs = useMemo(() => {
    if (!versionA || !versionB) return [];
    const diffs: string[] = [];
    const stepsA = new Map(versionA.result.recipe.steps.map((s) => [s.op, s]));
    const stepsB = new Map(versionB.result.recipe.steps.map((s) => [s.op, s]));
    const allOps = Array.from(new Set([...stepsA.keys(), ...stepsB.keys()]));
    for (const op of allOps) {
      const a = stepsA.get(op);
      const b = stepsB.get(op);
      if (!a && b) {
        diffs.push(`+ ${op}${b.modelRef ? ` (${b.modelRef})` : ''}`);
        continue;
      }
      if (a && !b) {
        diffs.push(`− ${op}`);
        continue;
      }
      if (a && b) {
        if (a.enabled !== b.enabled) {
          diffs.push(`${op}: ${a.enabled ? 'ON' : 'OFF'} → ${b.enabled ? 'ON' : 'OFF'}`);
        }
        if (a.modelRef !== b.modelRef) {
          diffs.push(`${op}.model: ${a.modelRef ?? '—'} → ${b.modelRef ?? '—'}`);
        }
        const pKeys = Array.from(new Set([...Object.keys(a.params), ...Object.keys(b.params)]));
        for (const pk of pKeys) {
          if (a.params[pk] !== b.params[pk]) {
            diffs.push(`${op}.${pk}: ${String(a.params[pk] ?? '—')} → ${String(b.params[pk] ?? '—')}`);
          }
        }
      }
    }
    const cA = versionA.result.provenance.correctionsApplied;
    const cB = versionB.result.provenance.correctionsApplied;
    if (cA !== cB) {
      diffs.push(lang === 'zh' ? `人工纠错: ${cA} → ${cB} 条` : `HumanCorrections: ${cA} → ${cB}`);
    }
    return diffs;
  }, [versionA, versionB, lang]);

  const copyText = (key: string, text: string) => {
    navigator.clipboard?.writeText(text).catch(() => {});
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1600);
  };

  const calSourceLabel =
    specimen.calibration.source === 'metadata'
      ? t('cal_source_metadata', lang)
      : specimen.calibration.source === 'databar'
      ? t('cal_source_databar', lang)
      : specimen.calibration.source === 'manual'
      ? t('cal_source_manual', lang)
      : t('cal_source_none', lang);

  const stages: Array<{
    id: WorkbenchStage;
    num: string;
    zh: string;
    en: string;
    optional?: boolean;
  }> = [
    { id: 'import', num: '1', zh: '导入', en: 'Import' },
    { id: 'workflow', num: '2', zh: '流程', en: 'Workflow' },
    { id: 'model', num: '3', zh: '模型', en: 'Model', optional: true },
    { id: 'results', num: '4', zh: '结果', en: 'Results' },
  ];

  return (
    <aside className="w-[404px] xl:w-[432px] shrink-0 border-l border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E1320] flex flex-col min-h-0">
      {/* 1. Guided 4-Stage Horizontal Rail: Import -> Workflow -> Model (Optional) -> Results */}
      <div className="grid grid-cols-4 border-b border-slate-200 dark:border-slate-800 bg-slate-100/80 dark:bg-[#0B0F1A] shrink-0">
        {stages.map((st) => {
          const active = !compareMode && stage === st.id;
          return (
            <button
              key={st.id}
              type="button"
              onClick={() => {
                setCompareMode(false);
                onChangeStage(st.id);
              }}
              className={`relative py-2.5 px-1 text-center transition-colors cursor-pointer border-r last:border-r-0 border-slate-200/70 dark:border-slate-800/80 ${
                active
                  ? 'text-blue-700 dark:text-cyan-300 bg-white dark:bg-[#0E1320] font-semibold'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-white/70 dark:hover:bg-slate-800/60 hover:text-slate-900 dark:hover:text-slate-100'
              }`}
            >
              <div className="flex items-center justify-center gap-1 text-[11.5px] font-semibold whitespace-nowrap">
                <span
                  className={`font-mono text-[10.5px] ${
                    active
                      ? 'text-blue-600 dark:text-cyan-400'
                      : 'text-slate-400 dark:text-slate-500'
                  }`}
                >
                  {st.num}.
                </span>
                <span>{lang === 'zh' ? st.zh : st.en}</span>
                {st.id === 'results' && flaggedCount > 0 && (
                  <span className="font-mono text-[10px] text-amber-600 dark:text-amber-400">{flaggedCount}</span>
                )}
              </div>
              {active && (
                <span className="absolute bottom-0 left-1 right-1 h-[2px] bg-blue-600 dark:bg-cyan-400" />
              )}
            </button>
          );
        })}
      </div>

      {/* 2. Single-Line Calibration & Version Compare Bar (Zero vertical bloat!) */}
      <div className="px-4 py-1.5 border-b border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A]/80 shrink-0 flex items-center justify-between gap-2 text-[11px]">
        <div className="flex items-center gap-1.5 min-w-0 font-mono tabular-nums">
          {specimen.calibration.confirmed ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
          ) : (
            <AlertTriangle className="w-3.5 h-3.5 text-amber-500 shrink-0" />
          )}
          <span className="text-slate-800 dark:text-slate-200 font-medium truncate">
            {specimen.calibration.pixelSize} {specimen.calibration.spaceUnit}/px
          </span>
          {!specimen.calibration.confirmed && (
            <button
              type="button"
              onClick={onConfirmCalibration}
              className="h-5 px-1.5 rounded border border-amber-500/60 bg-amber-500/15 hover:bg-amber-500/25 text-amber-700 dark:text-amber-300 font-sans font-semibold shrink-0 cursor-pointer"
            >
              {lang === 'zh' ? '确认' : 'Confirm'}
            </button>
          )}
          <button
            ref={calTriggerRef}
            type="button"
            onClick={openCalEditor}
            className="h-5 px-1.5 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800/90 hover:border-blue-500 dark:hover:border-cyan-400 text-slate-700 dark:text-slate-200 hover:text-blue-700 dark:hover:text-cyan-300 font-sans font-medium shrink-0 cursor-pointer"
          >
            {lang === 'zh' ? '标定' : 'Edit'}
          </button>
        </div>

        {result && (
          <div className="flex items-center gap-2 shrink-0">
            <span className="font-mono text-[10.5px] text-slate-500 dark:text-slate-400 hidden sm:inline">
              v{runHistory.length || 1} · {result.durationMs}ms
            </span>
            <button
              type="button"
              role="switch"
              aria-checked={compareMode}
              onClick={() => setCompareMode((v) => !v)}
              className={`h-6 px-2.5 rounded-md border text-[11px] font-medium flex items-center gap-1 shrink-0 shadow-xs transition-colors cursor-pointer ${
                compareMode
                  ? 'border-blue-600 dark:border-cyan-400 bg-blue-600/12 dark:bg-cyan-500/20 text-blue-700 dark:text-cyan-200'
                  : 'border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800/90 text-slate-700 dark:text-slate-200 hover:border-blue-500 dark:hover:border-cyan-400 hover:text-blue-700 dark:hover:text-cyan-200'
              }`}
            >
              <GitCompareArrows className="w-3 h-3" />
              <span>{lang === 'zh' ? '对比' : 'Compare'}</span>
            </button>
          </div>
        )}
      </div>

      {/* Inline Calibration Pop-down Drawer (only visible when user clicks Edit) */}
      {editingCal && (
        <div
          ref={calDrawerRef}
          className="px-4 py-2.5 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#0C111E] shrink-0 grid grid-cols-3 gap-2 text-[11px]"
        >
          <label className="space-y-1">
            <span className="text-slate-500 dark:text-slate-400 block">
              {lang === 'zh' ? '像素尺度' : 'Pixel Size'}
            </span>
            <input
              type="number"
              step="0.01"
              value={calPx}
              onChange={(e) => setCalPx(e.target.value)}
              className="w-full h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
            />
          </label>
          <label className="space-y-1">
            <span className="text-slate-500 dark:text-slate-400 block">
              {lang === 'zh' ? '物理单位' : 'Unit'}
            </span>
            <select
              value={calUnit}
              onChange={(e) => setCalUnit(e.target.value as Calibration['spaceUnit'])}
              className="w-full h-7 px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
            >
              <option value="nm">nm</option>
              <option value="µm">µm</option>
              <option value="mm">mm</option>
              <option value="px">px</option>
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-slate-500 dark:text-slate-400 block">
              {lang === 'zh' ? '流道高 µm' : 'Ch. H (µm)'}
            </span>
            <input
              type="number"
              step="1"
              placeholder="—"
              value={calChannelH}
              onChange={(e) => setCalChannelH(e.target.value)}
              className="w-full h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
            />
          </label>
          <div className="col-span-3 flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={() => setEditingCal(false)}
              className="h-6 px-2.5 text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 cursor-pointer"
            >
              {lang === 'zh' ? '取消' : 'Cancel'}
            </button>
            <button
              type="button"
              onClick={saveCalibration}
              className="h-6 px-3 rounded bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 font-medium cursor-pointer"
            >
              {lang === 'zh' ? '保存并重算' : 'Save & Recompute'}
            </button>
          </div>
        </div>
      )}

      {/* 3. Scrollable Stage Body */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 py-3.5 space-y-4">
        {/* ================================================================= */}
        {/* COMPARE MODE (Overlaying any stage when toggled)                  */}
        {/* ================================================================= */}
        {compareMode && versionA && versionB && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-2 text-[11px]">
              <label className="space-y-1 min-w-0">
                <span className="font-mono uppercase text-[10px] text-slate-500 block">
                  {lang === 'zh' ? '基线版本 A' : 'Version A (Base)'}
                </span>
                <select
                  value={versionA.id}
                  onChange={(e) => setCompareIdA(e.target.value)}
                  className="w-full h-7 px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px] text-slate-900 dark:text-slate-100 truncate"
                >
                  {runHistory.map((h) => (
                    <option key={h.id} value={h.id}>
                      v{h.version} · {h.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-1 min-w-0">
                <span className="font-mono uppercase text-[10px] text-blue-600 dark:text-cyan-400 block">
                  {lang === 'zh' ? '对比版本 B' : 'Version B (Target)'}
                </span>
                <select
                  value={versionB.id}
                  onChange={(e) => setCompareIdB(e.target.value)}
                  className="w-full h-7 px-1.5 rounded border border-blue-500/50 dark:border-cyan-500/50 bg-white dark:bg-slate-900 font-mono text-[11px] text-slate-900 dark:text-slate-100 truncate"
                >
                  {runHistory.map((h) => (
                    <option key={h.id} value={h.id}>
                      v{h.version} · {h.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {runHistory.length < 2 && (
              <div className="p-3 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-900/40 space-y-2 text-[11.5px]">
                <div className="text-slate-600 dark:text-slate-300 leading-snug">
                  {lang === 'zh'
                    ? '当前仅有 1 个执行版本，调整参数或点击生成对照版本即可对比差异：'
                    : 'Only 1 version recorded. Tweak a parameter or generate an alternate run:'}
                </div>
                <button
                  type="button"
                  onClick={onGenerateComparisonBaseline}
                  className="h-7 px-2.5 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[11px] font-semibold cursor-pointer"
                >
                  {lang === 'zh' ? '一键生成对照版本' : 'Generate Alternate Run'}
                </button>
              </div>
            )}

            <div className="border-y border-slate-200 dark:border-slate-800 divide-y divide-slate-200 dark:divide-slate-800">
              <div className="grid grid-cols-12 gap-1 py-1.5 text-[10px] font-mono uppercase text-slate-400">
                <span className="col-span-4">{lang === 'zh' ? '指标' : 'Metric'}</span>
                <span className="col-span-3 text-right">v{versionA.version}</span>
                <span className="col-span-3 text-right">v{versionB.version}</span>
                <span className="col-span-2 text-right">Δ</span>
              </div>
              {compareMetricRows.map((row) => {
                const deltaColor = !row.changed
                  ? 'text-slate-400'
                  : row.deltaNum !== null && row.deltaNum > 0
                  ? 'text-amber-600 dark:text-amber-400'
                  : 'text-emerald-600 dark:text-emerald-400';
                return (
                  <div
                    key={row.key}
                    className={`grid grid-cols-12 gap-1 py-2 items-baseline text-[11.5px] font-mono tabular-nums ${
                      row.changed ? 'bg-blue-50/50 dark:bg-cyan-500/8' : 'opacity-70'
                    }`}
                  >
                    <div className="col-span-4 font-sans text-[11.5px] text-slate-700 dark:text-slate-200 truncate pr-1">
                      <div>{lang === 'zh' ? row.labelZh : row.labelEn}</div>
                      {row.unit && <div className="font-mono text-[10px] text-slate-400">{row.unit}</div>}
                    </div>
                    <div className="col-span-3 text-right text-slate-600 dark:text-slate-400 truncate">
                      {row.valA ?? '—'}
                    </div>
                    <div className="col-span-3 text-right font-semibold text-slate-900 dark:text-slate-100 truncate">
                      {row.valB ?? '—'}
                    </div>
                    <div className={`col-span-2 text-right font-semibold ${deltaColor}`}>
                      {!row.changed
                        ? '0'
                        : row.deltaNum !== null
                        ? `${row.deltaNum > 0 ? '+' : ''}${row.deltaNum}`
                        : '≠'}
                    </div>
                  </div>
                );
              })}
            </div>

            {compareRecipeDiffs.length > 0 && (
              <div className="space-y-1 font-mono text-[11px] text-slate-700 dark:text-slate-300">
                <div className="font-sans text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  {lang === 'zh' ? '配方差异' : 'Recipe Diff'}
                </div>
                {compareRecipeDiffs.map((d, i) => (
                  <div
                    key={i}
                    className="px-2 py-1 rounded bg-slate-100/80 dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800 truncate"
                  >
                    {d}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ================================================================= */}
        {/* STAGE 1: IMPORT IMAGE & CALIBRATE                                 */}
        {/* ================================================================= */}
        {!compareMode && stage === 'import' && (
          <div className="space-y-4">
            {/* Drag & Drop Microscopy File Importer */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragOver(true);
              }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragOver(false);
                const file = e.dataTransfer.files?.[0];
                if (file) onDropFile(file);
              }}
              onClick={onTriggerUpload}
              className={`p-4 rounded-md border border-dashed text-center space-y-2 transition-colors cursor-pointer ${
                isDragOver
                  ? 'border-blue-600 dark:border-cyan-400 bg-blue-50/70 dark:bg-cyan-500/10'
                  : 'border-slate-300 dark:border-slate-700 hover:border-blue-500 dark:hover:border-cyan-500 bg-slate-50/60 dark:bg-slate-900/40'
              }`}
            >
              <Upload className="w-5 h-5 mx-auto text-blue-600 dark:text-cyan-400" />
              <div className="text-[12.5px] font-semibold text-slate-900 dark:text-slate-100">
                {lang === 'zh' ? '点击或拖拽导入显微 / 电镜图像' : 'Click or Drop Microscopy / EM Image'}
              </div>
              <div className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                TIFF · OME-TIFF · SEM TIFF · Gatan DM3/DM4 · PNG/JPG
              </div>
            </div>

            {/* Semi-Auto vs Full-Auto Policy Strip */}
            <div className="px-2.5 py-1.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] flex items-center justify-between gap-2 text-[11px]">
              <span className="text-slate-600 dark:text-slate-300 truncate">
                {autoRunOnImport
                  ? lang === 'zh'
                    ? '导入策略：全自动执行（识别后直接分析）'
                    : 'Import Mode: Full-Auto Execute'
                  : lang === 'zh'
                  ? '导入策略：半自动逐项确认（批量中心仍全自动）'
                  : 'Import Mode: Semi-Auto Confirm (Batch stays Full-Auto)'}
              </span>
              <label className="inline-flex items-center gap-1.5 shrink-0 text-[10.5px] font-mono text-slate-500 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoRunOnImport}
                  onChange={onToggleAutoRunOnImport}
                  className="accent-cyan-500 cursor-pointer"
                />
                <span>{lang === 'zh' ? '全自动' : 'Auto'}</span>
              </label>
            </div>

            {/* Active Specimen Selector & Metadata */}
            <div className="space-y-2 text-[12px]">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  {lang === 'zh' ? '当前样本' : 'Active Specimen'}
                </span>
                <select
                  value={specimen.id}
                  onChange={(e) => onSwitchSpecimen(e.target.value)}
                  className="h-6 max-w-[200px] px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px] text-slate-800 dark:text-slate-200 truncate"
                >
                  {specimens.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="border-y border-slate-200 dark:border-slate-800 divide-y divide-slate-100 dark:divide-slate-800/80 font-mono text-[11.5px] tabular-nums">
                <div className="py-1.5 flex justify-between gap-2">
                  <span className="text-slate-500 font-sans">{lang === 'zh' ? '尺寸 / 格式' : 'Size / Format'}</span>
                  <span className="text-slate-900 dark:text-slate-100">
                    {specimen.width}×{specimen.height} px
                    {specimen.importReport?.bitDepth ? ` · ${specimen.importReport.bitDepth}-bit` : ''}
                    {specimen.importReport?.fileFormat ? ` · ${specimen.importReport.fileFormat}` : ''}
                  </span>
                </div>
                {specimen.calibration.vendor && (
                  <div className="py-1.5 flex justify-between gap-2">
                    <span className="text-slate-500 font-sans">{lang === 'zh' ? '设备厂商元数据' : 'Vendor Metadata'}</span>
                    <span className="text-blue-600 dark:text-cyan-400 truncate">
                      {specimen.calibration.vendor}
                      {specimen.calibration.acceleratingVoltageKv ? ` · ${specimen.calibration.acceleratingVoltageKv} kV` : ''}
                      {specimen.calibration.workingDistanceMm ? ` · WD ${specimen.calibration.workingDistanceMm} mm` : ''}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* ============================================================== */}
            {/* SEMI-AUTO 3-STEP CONFIRMATION CHECKLIST (Cal -> Crop -> Run)   */}
            {/* ============================================================== */}
            {(specimen.importReport || !result) && (
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                    {lang === 'zh'
                      ? '半自动确认流程（先标定 → 再裁切 → 后执行）'
                      : 'Semi-Auto Flow (Calibrate → Crop → Run)'}
                  </span>
                  {!result && (
                    <span className="font-mono text-[10px] text-amber-500">
                      {lang === 'zh' ? '未开始分析' : 'Not Executed'}
                    </span>
                  )}
                </div>

                {/* STEP 1: PHYSICAL CALIBRATION FIRST (AUTO OR MANUAL SCALE-BAR TRACE BEFORE CROPPING INFO BAR) */}
                <div
                  className={`p-2.5 rounded-md border space-y-2 text-[11.5px] ${
                    !specimen.calibration.confirmed
                      ? 'border-amber-500/50 bg-amber-50/40 dark:bg-amber-500/5'
                      : 'border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-[#0B0F1A]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                      <span className="font-mono text-[10.5px] text-amber-500">1.</span>
                      <MoveHorizontal className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                      <span>{lang === 'zh' ? '物理尺度标定（自动 / 手动）' : 'Scale Calibration (Auto / Manual)'}</span>
                    </span>
                    {specimen.calibration.confirmed ? (
                      <span className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400">
                        ✓ {specimen.calibration.pixelSize} {specimen.calibration.spaceUnit}/px
                      </span>
                    ) : (
                      <span className="font-mono text-[10px] text-amber-600 dark:text-amber-400">
                        {lang === 'zh' ? '待确认' : 'Pending'}
                      </span>
                    )}
                  </div>

                  {!specimen.calibration.confirmed ? (
                    <div className="space-y-2">
                      <div className="text-[11px] text-slate-600 dark:text-slate-300 leading-snug">
                        {specimen.importReport?.vendorMetadataFound
                          ? lang === 'zh'
                            ? `已从厂商元数据读取参考标定（${specimen.calibration.vendor ?? 'Metadata'}）；建议在裁切信息栏前核对比例尺，或点击下方手动描测：`
                            : `Read calibration from vendor metadata (${specimen.calibration.vendor ?? 'Metadata'}). Verify against the info-bar scale bar or trace manually before cropping:`
                          : specimen.importReport?.scaleBarPx
                          ? lang === 'zh'
                            ? `自动检出信息栏比例尺约 ${specimen.importReport.scaleBarPx} px；可直接输入标尺数值换算，或在左侧画布手动描测比例尺线段：`
                            : `Auto-detected ~${specimen.importReport.scaleBarPx} px scale bar in info bar. Enter its printed length or trace it manually on canvas:`
                          : lang === 'zh'
                          ? '信息栏通常包含比例尺 (Scale Bar)，建议先点击下方按钮在左侧画布沿比例尺描线标定，或直接输入像素物理尺度：'
                          : 'Info bars usually contain the scale bar. Trace the scale bar on canvas first, or enter pixel size directly:'}
                      </div>

                      {/* Manual Scale-Bar Line Calibration Trigger */}
                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() =>
                            onChangeToolMode(toolMode === 'calibrate_line' ? 'pan' : 'calibrate_line')
                          }
                          className={`flex-1 h-7 px-2 rounded border text-[11px] font-medium flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                            toolMode === 'calibrate_line'
                              ? 'border-emerald-500 bg-emerald-500 text-slate-950 font-semibold'
                              : 'border-emerald-500/50 bg-emerald-50/70 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100 dark:hover:bg-emerald-500/20'
                          }`}
                        >
                          <MoveHorizontal className="w-3.5 h-3.5" />
                          <span>
                            {toolMode === 'calibrate_line'
                              ? lang === 'zh'
                                ? '请在左侧沿比例尺拖拽描线…'
                                : 'Drag across scale bar on canvas…'
                              : manualCalLine
                              ? lang === 'zh'
                                ? `重描比例尺 (已描 ${manualCalLine.lengthPx} px)`
                                : `Re-trace Scale Bar (${manualCalLine.lengthPx} px)`
                              : lang === 'zh'
                              ? '画布手动描测比例尺 (Scale Bar)'
                              : 'Trace Scale Bar on Canvas'}
                          </span>
                        </button>
                        {manualCalLine && onClearManualCalLine && (
                          <button
                            type="button"
                            onClick={onClearManualCalLine}
                            title={lang === 'zh' ? '清除描线' : 'Clear traced line'}
                            className="h-7 px-2 rounded border border-slate-300 dark:border-slate-700 text-slate-500 hover:text-rose-500 cursor-pointer"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                      {toolMode === 'calibrate_line' && (
                        <div className="text-[10.5px] text-emerald-700 dark:text-emerald-300/90 leading-snug">
                          {lang === 'zh'
                            ? '拉线时将自动吸附至附近比例尺的精确起止端点（按住 Alt 键可自由描线）。'
                            : 'Auto-snaps start & end points onto the nearest scale bar edges (hold Alt for freehand).'}
                        </div>
                      )}

                      {/* Scale Bar Length -> Pixel Size Converter (works for BOTH manual traced line & auto-detected scale bar!) */}
                      {(() => {
                        const activeBarPx =
                          manualCalLine?.lengthPx ?? specimen.importReport?.scaleBarPx ?? 0;
                        if (activeBarPx < 6) return null;
                        const isManualTrace = Boolean(manualCalLine && manualCalLine.lengthPx >= 6);
                        return (
                          <div
                            className={`p-2 rounded border space-y-1.5 ${
                              isManualTrace
                                ? 'border-emerald-500/50 bg-emerald-50/60 dark:bg-emerald-950/30'
                                : 'border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-950/60'
                            }`}
                          >
                            <div className="flex items-center justify-between text-[10.5px] font-mono">
                              <span
                                className={
                                  isManualTrace
                                    ? 'text-emerald-700 dark:text-emerald-300 font-semibold'
                                    : 'text-slate-500'
                                }
                              >
                                {isManualTrace
                                  ? manualCalLine?.snapped
                                    ? lang === 'zh'
                                      ? `已自动吸附比例尺两端：${activeBarPx} px 对应实际长度`
                                      : `Snapped to Scale Bar: ${activeBarPx} px equals`
                                    : lang === 'zh'
                                    ? `手动描尺：${activeBarPx} px 对应实际长度`
                                    : `Traced Bar: ${activeBarPx} px equals`
                                  : lang === 'zh'
                                  ? `自动标尺：${activeBarPx} px 对应实际长度`
                                  : `Auto Bar: ${activeBarPx} px equals`}
                              </span>
                            </div>
                            <div className="flex items-center gap-1.5">
                              <input
                                type="number"
                                step="any"
                                placeholder={lang === 'zh' ? '输入标尺数值，如 2 或 500' : 'e.g. 2 or 500'}
                                value={scaleBarValInput}
                                onChange={(e) => {
                                  const raw = e.target.value;
                                  setScaleBarValInput(raw);
                                  const num = Number(raw);
                                  if (Number.isFinite(num) && num > 0 && activeBarPx > 0) {
                                    setStep2Px(String(Number((num / activeBarPx).toPrecision(4))));
                                    setStep2Unit(scaleBarUnitInput);
                                  }
                                }}
                                className="flex-1 h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px] text-slate-900 dark:text-slate-100"
                              />
                              <select
                                value={scaleBarUnitInput}
                                onChange={(e) => {
                                  const nextU = e.target.value as Calibration['spaceUnit'];
                                  setScaleBarUnitInput(nextU);
                                  setStep2Unit(nextU);
                                  const num = Number(scaleBarValInput);
                                  if (Number.isFinite(num) && num > 0 && activeBarPx > 0) {
                                    setStep2Px(String(Number((num / activeBarPx).toPrecision(4))));
                                  }
                                }}
                                className="h-7 px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px] text-slate-900 dark:text-slate-100"
                              >
                                <option value="nm">nm</option>
                                <option value="µm">µm</option>
                                <option value="mm">mm</option>
                              </select>
                            </div>
                          </div>
                        );
                      })()}

                      <div className="grid grid-cols-12 gap-1.5 items-center">
                        <input
                          type="number"
                          step="any"
                          value={step2Px}
                          onChange={(e) => setStep2Px(e.target.value)}
                          className="col-span-7 h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px] tabular-nums text-slate-900 dark:text-slate-100"
                        />
                        <select
                          value={step2Unit}
                          onChange={(e) => setStep2Unit(e.target.value as Calibration['spaceUnit'])}
                          className="col-span-5 h-7 px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px] text-slate-900 dark:text-slate-100"
                        >
                          <option value="nm">nm / px</option>
                          <option value="µm">µm / px</option>
                          <option value="mm">mm / px</option>
                          <option value="px">px / px</option>
                        </select>
                      </div>

                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => {
                            const num = Number(step2Px);
                            if (!Number.isFinite(num) || num <= 0) return;
                            onUpdateCalibration({
                              ...specimen.calibration,
                              pixelSize: num,
                              spaceUnit: step2Unit,
                              source:
                                manualCalLine
                                  ? 'manual'
                                  : num === specimen.calibration.pixelSize &&
                                    step2Unit === specimen.calibration.spaceUnit
                                  ? specimen.calibration.source
                                  : 'manual',
                              confirmed: true,
                            });
                            onClearManualCalLine?.();
                            if (toolMode === 'calibrate_line') onChangeToolMode('pan');
                          }}
                          className="flex-1 h-7 px-2 rounded bg-amber-500 hover:bg-amber-400 text-slate-950 text-[11px] font-semibold flex items-center justify-center gap-1 cursor-pointer"
                        >
                          <Check className="w-3 h-3" />
                          <span>{lang === 'zh' ? '确认物理标定' : 'Confirm Calibration'}</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            onUpdateCalibration({
                              ...specimen.calibration,
                              pixelSize: 1,
                              spaceUnit: 'px',
                              source: 'manual',
                              confirmed: true,
                            });
                            onClearManualCalLine?.();
                            if (toolMode === 'calibrate_line') onChangeToolMode('pan');
                          }}
                          className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11px] text-slate-700 dark:text-slate-300 cursor-pointer"
                        >
                          {lang === 'zh' ? '使用 px' : 'Use px'}
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-between gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                      <span className="font-mono truncate">
                        {specimen.calibration.pixelSize} {specimen.calibration.spaceUnit}/px · {calSourceLabel}
                      </span>
                      <div className="flex items-center gap-2 shrink-0">
                        <button
                          type="button"
                          onClick={() => {
                            onUpdateCalibration({ ...specimen.calibration, confirmed: false });
                            onChangeToolMode('calibrate_line');
                          }}
                          className="text-emerald-600 dark:text-emerald-400 hover:underline cursor-pointer"
                        >
                          {lang === 'zh' ? '描尺重标' : 'Re-trace'}
                        </button>
                        <button
                          type="button"
                          onClick={openCalEditor}
                          className="text-blue-600 dark:text-cyan-400 hover:underline cursor-pointer"
                        >
                          {lang === 'zh' ? '修改' : 'Edit'}
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* STEP 2: INFO BAR DETECTION & CROP CONFIRMATION (AFTER CALIBRATION) */}
                <div
                  className={`p-2.5 rounded-md border space-y-2 text-[11.5px] ${
                    specimen.importReport?.cropStatus === 'pending' && specimen.calibration.confirmed
                      ? 'border-amber-500/50 bg-amber-50/40 dark:bg-amber-500/5'
                      : 'border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-[#0B0F1A]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                      <span className="font-mono text-[10.5px] text-amber-500">2.</span>
                      <Scissors className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                      <span>{lang === 'zh' ? '信息栏检测与裁切' : 'Info Bar Detection & Crop'}</span>
                    </span>
                    {specimen.importReport?.cropStatus === 'pending' ? (
                      <span className="font-mono text-[10px] text-amber-600 dark:text-amber-400">
                        {lang === 'zh' ? '待确认' : 'Pending'}
                      </span>
                    ) : specimen.importReport?.scaleBarCropped ? (
                      <span className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400">
                        ✓ {specimen.importReport.croppedBarPx}px
                      </span>
                    ) : (
                      <span className="font-mono text-[10px] text-slate-400">
                        ✓ {lang === 'zh' ? '未裁切' : 'Skipped'}
                      </span>
                    )}
                  </div>

                  {specimen.importReport?.cropStatus === 'pending' ? (
                    <div className="space-y-2">
                      <div className="text-[11px] text-slate-600 dark:text-slate-300 leading-snug">
                        {(specimen.importReport.proposedCropBarPx ?? 0) > 0
                          ? lang === 'zh'
                            ? `自动检出${specimen.importReport.barPosition === 'top' ? '顶部' : '底部'}约 ${specimen.importReport.proposedCropBarPx} px 信息栏（左侧琥珀色虚线框）。完成第 1 步比例尺标定后，请确认裁切以排除信息栏干扰：`
                            : `Detected ~${specimen.importReport.proposedCropBarPx} px ${specimen.importReport.barPosition} info bar (amber dashed outline). After calibrating in Step 1, confirm crop to exclude it from analysis:`
                          : lang === 'zh'
                          ? '未自动检测到明显信息栏；若图像含底/顶栏可手动设置高度预览并裁切，或直接跳过：'
                          : 'No info bar auto-detected. Set a manual crop height to preview on canvas, or skip:'}
                      </div>

                      <div className="grid grid-cols-12 gap-1.5 items-center">
                        <select
                          value={pendingCropDraft.barPosition === 'none' ? 'bottom' : pendingCropDraft.barPosition}
                          onChange={(e) =>
                            onUpdatePendingCropDraft({
                              ...pendingCropDraft,
                              barPosition: e.target.value as 'bottom' | 'top',
                            })
                          }
                          className="col-span-4 h-7 px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[11px] text-slate-900 dark:text-slate-100"
                        >
                          <option value="bottom">{lang === 'zh' ? '底部裁切' : 'Bottom'}</option>
                          <option value="top">{lang === 'zh' ? '顶部裁切' : 'Top'}</option>
                        </select>
                        <input
                          type="range"
                          min={0}
                          max={Math.min(260, Math.max(40, Math.floor(specimen.height * 0.35)))}
                          step={2}
                          value={pendingCropDraft.cropPx}
                          onChange={(e) =>
                            onUpdatePendingCropDraft({
                              cropPx: Number(e.target.value),
                              barPosition:
                                pendingCropDraft.barPosition === 'none'
                                  ? 'bottom'
                                  : pendingCropDraft.barPosition,
                            })
                          }
                          className="col-span-5 accent-amber-500 cursor-pointer"
                        />
                        <div className="col-span-3 flex items-center gap-0.5">
                          <input
                            type="number"
                            min={0}
                            max={Math.max(0, specimen.height - 48)}
                            value={pendingCropDraft.cropPx}
                            onChange={(e) => {
                              const n = Math.max(0, Number(e.target.value) || 0);
                              onUpdatePendingCropDraft({
                                cropPx: n,
                                barPosition:
                                  pendingCropDraft.barPosition === 'none'
                                    ? 'bottom'
                                    : pendingCropDraft.barPosition,
                              });
                            }}
                            className="w-full h-7 px-1 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px] text-right tabular-nums text-slate-900 dark:text-slate-100"
                          />
                          <span className="font-mono text-[10px] text-slate-400">px</span>
                        </div>
                      </div>

                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          disabled={pendingCropDraft.cropPx <= 0}
                          onClick={() => {
                            onClearManualCalLine?.();
                            onConfirmInfoBarCrop(
                              pendingCropDraft.cropPx,
                              pendingCropDraft.barPosition === 'none'
                                ? 'bottom'
                                : pendingCropDraft.barPosition
                            );
                          }}
                          className="flex-1 h-7 px-2 rounded bg-amber-500 hover:bg-amber-400 disabled:opacity-40 text-slate-950 text-[11px] font-semibold flex items-center justify-center gap-1 cursor-pointer"
                        >
                          <Check className="w-3 h-3" />
                          <span>
                            {lang === 'zh'
                              ? `确认裁切 (${pendingCropDraft.cropPx} px)`
                              : `Confirm Crop (${pendingCropDraft.cropPx} px)`}
                          </span>
                        </button>
                        <button
                          type="button"
                          onClick={() => onConfirmInfoBarCrop(0, 'none')}
                          className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11px] text-slate-700 dark:text-slate-300 cursor-pointer"
                        >
                          {lang === 'zh' ? '跳过裁切' : 'Skip'}
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-between gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                      <span className="font-mono truncate">
                        {specimen.importReport?.scaleBarCropped
                          ? lang === 'zh'
                            ? `已裁切${specimen.importReport.barPosition === 'top' ? '顶部' : '底部'} ${specimen.importReport.croppedBarPx}px · 保留 ${specimen.width}×${specimen.height}`
                            : `Cropped ${specimen.importReport.croppedBarPx}px · FOV ${specimen.width}×${specimen.height}`
                          : lang === 'zh'
                          ? `保留原始全视场 (${specimen.width}×${specimen.height} px)`
                          : `Kept full original FOV (${specimen.width}×${specimen.height} px)`}
                      </span>
                      {specimen.importReport && (
                        <button
                          type="button"
                          onClick={onRevertInfoBarCrop}
                          className="text-blue-600 dark:text-cyan-400 hover:underline shrink-0 cursor-pointer"
                        >
                          {lang === 'zh' ? '重新调整' : 'Adjust'}
                        </button>
                      )}
                    </div>
                  )}
                </div>

                {/* STEP 3: WORKFLOW SELECTION & EXPLICIT EXECUTION */}
                <div
                  className={`p-2.5 rounded-md border space-y-2 text-[11.5px] ${
                    !result
                      ? 'border-blue-500/50 dark:border-cyan-500/50 bg-blue-50/40 dark:bg-cyan-500/6'
                      : 'border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-[#0B0F1A]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                      <span className="font-mono text-[10.5px] text-blue-600 dark:text-cyan-400">3.</span>
                      <span>{lang === 'zh' ? '选定工作流并执行' : 'Select Workflow & Run'}</span>
                    </span>
                    {result ? (
                      <span className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400">
                        ✓ {lang === 'zh' ? '已执行' : 'Executed'}
                      </span>
                    ) : (
                      <span className="font-mono text-[10px] text-blue-600 dark:text-cyan-400">
                        {lang === 'zh' ? '等待启动' : 'Ready to Run'}
                      </span>
                    )}
                  </div>

                  {specimen.importReport && (
                    <div className="text-[10.5px] font-mono text-slate-500 dark:text-slate-400 leading-snug">
                      {lang === 'zh'
                        ? `参考识别：${specimen.importReport.sceneLabelZh}${specimen.importReport.recognized ? ` (${Math.round(specimen.importReport.confidence * 100)}%)` : ''} · 仅供参考，请手动选定工作流`
                        : `Reference hint: ${specimen.importReport.sceneLabelEn}${specimen.importReport.recognized ? ` (${Math.round(specimen.importReport.confidence * 100)}%)` : ''} · choose workflow manually`}
                    </div>
                  )}

                  <select
                    value={recipe.id}
                    onChange={(e) => onSelectWorkflow(e.target.value, false)}
                    className="w-full h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[11.5px] font-medium text-slate-900 dark:text-slate-100"
                  >
                    <option value={UNRECOGNIZED_MANUAL_RECIPE_ID}>
                      {lang === 'zh' ? '纯手工标注 / 线剖面测量模式' : 'Manual Annotate / Profile Measure'}
                    </option>
                    <optgroup label={lang === 'zh' ? '内置分析工作流' : 'Built-in Workflows'}>
                      {BUILTIN_RECIPES.map((r) => (
                        <option key={r.id} value={r.id}>
                          {lang === 'zh' ? r.nameZh : r.nameEn}
                          {specimen.importReport?.suggestedRecipeId === r.id
                            ? lang === 'zh'
                              ? ' (参考建议)'
                              : ' (Hint)'
                            : ''}
                        </option>
                      ))}
                    </optgroup>
                    {savedWorkflows.length > 0 && (
                      <optgroup label={lang === 'zh' ? '已保存自定义工作流' : 'Saved Custom Workflows'}>
                        {savedWorkflows.map((w) => (
                          <option key={w.id} value={w.id}>
                            ★ {w.name}
                          </option>
                        ))}
                      </optgroup>
                    )}
                  </select>

                  <div className="grid grid-cols-12 gap-1.5 pt-0.5">
                    <button
                      type="button"
                      onClick={onRunActiveWorkflow}
                      className="col-span-7 h-8 px-2.5 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[11.5px] font-semibold flex items-center justify-center gap-1 cursor-pointer"
                    >
                      <Play className="w-3.5 h-3.5 fill-current" />
                      <span>{lang === 'zh' ? '确认并执行分析' : 'Confirm & Run'}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => onChangeStage('workflow')}
                      className="col-span-5 h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11.5px] font-medium text-slate-700 dark:text-slate-200 flex items-center justify-center gap-1 cursor-pointer"
                    >
                      <span>{lang === 'zh' ? '先调参数' : 'Tune Params'}</span>
                      <ArrowRight className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Collapsible Supported Microscopy & EM Vendor Format Matrix */}
            <div className="border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden">
              <button
                type="button"
                onClick={() => setShowFormatMatrix((v) => !v)}
                className="w-full px-3 py-2 bg-slate-50/80 dark:bg-slate-900/50 flex items-center justify-between text-[11.5px] font-medium text-slate-700 dark:text-slate-200 cursor-pointer"
              >
                <span className="flex items-center gap-1.5">
                  <Microscope className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
                  <span>
                    {lang === 'zh'
                      ? '光学 / 电子显微镜主流厂商格式支持矩阵'
                      : 'Light & Electron Microscopy Format Matrix'}
                  </span>
                </span>
                {showFormatMatrix ? (
                  <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                )}
              </button>

              {showFormatMatrix && (
                <div className="divide-y divide-slate-200 dark:divide-slate-800 text-[11px] max-h-60 overflow-y-auto">
                  {MICROSCOPY_FORMAT_REGISTRY.map((fmt) => (
                    <div key={fmt.id} className="p-2.5 space-y-0.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono font-semibold text-slate-900 dark:text-slate-100">
                          {fmt.extensions.join(' ')}
                        </span>
                        <span
                          className={`font-mono text-[10px] shrink-0 ${
                            fmt.supportTier === 'native_web'
                              ? 'text-emerald-600 dark:text-emerald-400'
                              : 'text-amber-600 dark:text-amber-400'
                          }`}
                        >
                          {fmt.supportTier === 'native_web'
                            ? lang === 'zh'
                              ? '● Web 原生解码'
                              : '● Native Web'
                            : lang === 'zh'
                            ? '▲ Python 引擎桥接'
                            : '▲ Python Bridge'}
                        </span>
                      </div>
                      <div className="text-[10.5px] text-slate-500 dark:text-slate-400">
                        {fmt.vendor}
                      </div>
                      <div className="text-[10.5px] text-slate-600 dark:text-slate-300 leading-snug">
                        {lang === 'zh' ? fmt.noteZh : fmt.noteEn}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Stage 1 Next Step CTA (shown when specimen has no importReport, e.g., built-in demos) */}
            {!specimen.importReport && result && (
              <button
                type="button"
                onClick={() => onChangeStage('workflow')}
                className="w-full h-9 px-3 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[12.5px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
              >
                <span>{lang === 'zh' ? '下一步：选择与编辑工作流' : 'Next: Choose & Edit Workflow'}</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            )}
          </div>
        )}

        {/* ================================================================= */}
        {/* STAGE 2: CHOOSE & EDIT WORKFLOW + PARAMETERS (SIDE-BY-SIDE)       */}
        {/* ================================================================= */}
        {!compareMode && stage === 'workflow' && (
          <div className="space-y-4">
            {/* Semi-Auto Pending Execution Gate Banner (shown when analysis has not started yet) */}
            {!result && (
              <div className="p-3 rounded-md border border-amber-500/50 bg-amber-50/50 dark:bg-amber-500/8 space-y-2 text-[11.5px]">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
                  <div className="space-y-0.5">
                    <div className="font-semibold text-slate-900 dark:text-slate-100">
                      {lang === 'zh'
                        ? '当前工作流尚未执行（半自动待确认）'
                        : 'Workflow Not Yet Executed (Semi-Auto)'}
                    </div>
                    <div className="text-[11px] text-slate-600 dark:text-slate-300 leading-snug">
                      {lang === 'zh'
                        ? '可在下方先调整算子或参数（拖动参数滑块也会实时预览），或直接点击按钮启动分析：'
                        : 'Configure steps or parameters below (tweaking any parameter previews live), or click Run to execute:'}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={onRunActiveWorkflow}
                    className="flex-1 h-8 px-3 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[11.5px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>{lang === 'zh' ? '确认并执行当前工作流' : 'Confirm & Run Workflow'}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onChangeStage('import')}
                    className="h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11px] text-slate-700 dark:text-slate-300 cursor-pointer"
                  >
                    {lang === 'zh' ? '返回标定/裁切' : 'Back to Cal/Crop'}
                  </button>
                </div>
              </div>
            )}

            {/* Compact Plan Summary */}
            {plan && (
              <div className="p-2.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] space-y-1.5 text-[11.5px]">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 min-w-0 font-mono text-[11px]">
                    <span
                      className={
                        plan.status === 'ready'
                          ? 'text-emerald-600 dark:text-emerald-400'
                          : 'text-amber-600 dark:text-amber-400'
                      }
                    >
                      {plan.status === 'ready' ? '✓' : '!'}
                    </span>
                    <span className="font-sans font-semibold text-slate-900 dark:text-slate-100 truncate">
                      {lang === 'zh' ? plan.recipeNameZh : plan.recipeNameEn}
                    </span>
                    <span className="text-slate-400">·</span>
                    <span className="text-slate-500 uppercase">{plan.planner}</span>
                  </div>
                  {!planAccepted && (
                    <button
                      type="button"
                      onClick={onAcceptPlan}
                      className="text-[11px] font-semibold text-blue-600 dark:text-cyan-400 hover:underline shrink-0 cursor-pointer"
                    >
                      {result ? t('plan_accept', lang) : lang === 'zh' ? '采用并执行' : 'Accept & Run'}
                    </button>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => setShowReasons((v) => !v)}
                  className="text-[11px] text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 flex items-center gap-1 cursor-pointer"
                >
                  {showReasons ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                  <span>{lang === 'zh' ? '查看规划依据与检查项' : 'View plan rationale & checks'}</span>
                </button>
                {showReasons && (
                  <div className="pt-1 space-y-1 text-[11px] text-slate-600 dark:text-slate-300">
                    {plan.checks.map((ck, i) => (
                      <div key={i} className="flex items-start gap-1.5">
                        <span className="font-mono font-bold text-emerald-500">{ck.mark}</span>
                        <span>{lang === 'zh' ? ck.textZh : ck.textEn}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Active Workflow Header (Preset switching & saving live exclusively in the Canvas top bar) */}
            <div className="flex items-center justify-between gap-2 pb-2 border-b border-slate-200 dark:border-slate-800">
              <div className="min-w-0">
                <div className="text-[12.5px] font-semibold text-slate-900 dark:text-slate-100 truncate">
                  {recipe.isCustom ? `★ ${recipe.name}` : lang === 'zh' ? recipe.nameZh : recipe.nameEn}
                </div>
                <div className="text-[10.5px] text-slate-500 dark:text-slate-400">
                  {lang === 'zh'
                    ? '切换或命名保存工作流请在左侧画布顶部栏操作'
                    : 'Switch or save custom presets from the canvas header bar'}
                </div>
              </div>
              <button
                type="button"
                onClick={onCustomizeWorkflow}
                className="text-[11px] font-medium text-blue-600 dark:text-cyan-400 hover:underline shrink-0 cursor-pointer"
              >
                {lang === 'zh' ? '全屏编排 →' : 'Full Studio →'}
              </button>
            </div>

            {/* Categorized Add-Step Bar (No overcrowded operator list!) */}
            <div className="p-2.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                {lang === 'zh' ? '向当前流程添加处理算子' : 'Add Processing Step'}
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                <select
                  value={addCategory}
                  onChange={(e) => {
                    const cat = e.target.value as OperatorSpec['category'];
                    setAddCategory(cat);
                    const first = OPERATOR_CATALOG.find((o) => o.category === cat);
                    if (first) setAddOpId(first.op);
                  }}
                  className="h-7 px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[11.5px] text-slate-900 dark:text-slate-100"
                >
                  {PROCESSING_CATEGORIES.map((c) => (
                    <option key={c.id} value={c.id}>
                      {lang === 'zh' ? c.labelZh : c.labelEn}
                    </option>
                  ))}
                </select>
                <div className="flex items-center gap-1">
                  <select
                    value={addOpId}
                    onChange={(e) => setAddOpId(e.target.value)}
                    className="flex-1 min-w-0 h-7 px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[11.5px] text-slate-900 dark:text-slate-100 truncate"
                  >
                    {opsInCategory.map((o) => (
                      <option key={o.op} value={o.op}>
                        {lang === 'zh' ? o.labelZh : o.labelEn}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={handleAddStepInline}
                    title={lang === 'zh' ? '添加步骤并实时运行' : 'Add step & run live'}
                    className="h-7 px-2 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 flex items-center justify-center shrink-0 cursor-pointer"
                  >
                    <Plus className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Live Computation Readout Strip (updates immediately as parameters change) */}
            {result && result.summary.length > 0 && (
              <div className="px-2.5 py-1.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] flex items-center justify-between gap-2 font-mono text-[11px] tabular-nums">
                <div className="flex items-center gap-2 min-w-0 truncate">
                  <span className="text-emerald-500 shrink-0">●</span>
                  <span className="text-slate-600 dark:text-slate-300 truncate">
                    {lang === 'zh' ? result.summary[0].labelZh : result.summary[0].labelEn}:{' '}
                    <strong className="text-blue-600 dark:text-cyan-400">
                      {String(result.summary[0].value ?? '—')}
                    </strong>
                    {result.summary[0].unit ? ` ${result.summary[0].unit}` : ''}
                  </span>
                  {result.summary[1] && (
                    <span className="text-slate-500 dark:text-slate-400 truncate">
                      · {String(result.summary[1].value ?? '—')}
                      {result.summary[1].unit ? ` ${result.summary[1].unit}` : ''}
                    </span>
                  )}
                </div>
                <span className="text-[10px] text-slate-400 shrink-0">
                  {result.durationMs.toFixed(1)} ms
                </span>
              </div>
            )}

            {/* Manual ROI Scope Bar: restrict parameter tuning & pipeline execution to ROI */}
            <div
              className={`px-2.5 py-1.5 rounded border flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono ${
                roi
                  ? 'border-cyan-500/50 bg-cyan-500/8 text-cyan-700 dark:text-cyan-300'
                  : 'border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-[#0B0F1A] text-slate-600 dark:text-slate-400'
              }`}
            >
              <div className="flex items-center gap-1.5 min-w-0 flex-1">
                <Crop className="w-3.5 h-3.5 text-cyan-500 shrink-0" />
                {roi ? (
                  <span className="truncate tabular-nums">
                    {lang === 'zh'
                      ? `ROI ${roi.width}×${roi.height} px · 仅计算区域内`
                      : `ROI ${roi.width}×${roi.height} px · Scoped`}
                  </span>
                ) : (
                  <span className="truncate">
                    {lang === 'zh' ? '分析区域：整幅视场（可框选 ROI 提速）' : 'Scope: Full FOV (draw ROI to focus)'}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <button
                  type="button"
                  disabled={!segmentedResultsRoi}
                  onClick={() => {
                    if (!segmentedResultsRoi) return;
                    onChangeRoi(segmentedResultsRoi);
                    if (toolMode === 'roi') onChangeToolMode('pan');
                  }}
                  title={
                    segmentedResultsRoi
                      ? lang === 'zh'
                        ? `自动框选全部检测目标并留出 10% 边距 (${segmentedResultsRoi.width}×${segmentedResultsRoi.height} px)`
                        : `Set ROI to bounding box of all detected objects with 10% padding (${segmentedResultsRoi.width}×${segmentedResultsRoi.height} px)`
                      : lang === 'zh'
                      ? '请先执行分割检测以生成目标边界框'
                      : 'Run segmentation first to focus ROI on detected objects'
                  }
                  className={`h-6 px-2 rounded text-[10.5px] font-sans font-medium flex items-center gap-1 whitespace-nowrap shrink-0 transition-colors ${
                    segmentedResultsRoi
                      ? 'border border-cyan-500/50 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 hover:bg-cyan-500/20 cursor-pointer'
                      : 'border border-slate-200 dark:border-slate-800 text-slate-400 dark:text-slate-600 cursor-not-allowed'
                  }`}
                >
                  <Crosshair className="w-3 h-3 shrink-0" />
                  <span>{lang === 'zh' ? '聚焦结果' : 'Focus on Results'}</span>
                </button>
                <button
                  type="button"
                  onClick={() => onChangeToolMode(toolMode === 'roi' ? 'pan' : 'roi')}
                  className={`h-6 px-2 rounded text-[10.5px] font-sans font-medium whitespace-nowrap shrink-0 cursor-pointer transition-colors ${
                    toolMode === 'roi'
                      ? 'bg-cyan-500 text-slate-950 font-semibold'
                      : 'border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-cyan-500/60'
                  }`}
                >
                  {toolMode === 'roi'
                    ? lang === 'zh'
                      ? '框选中…'
                      : 'Drawing…'
                    : roi
                    ? lang === 'zh'
                      ? '重选'
                      : 'Redraw'
                    : lang === 'zh'
                    ? '框选 ROI'
                    : 'Set ROI'}
                </button>
                {roi && (
                  <button
                    type="button"
                    onClick={() => onChangeRoi(null)}
                    title={lang === 'zh' ? '清除 ROI 恢复全图' : 'Clear ROI (Full FOV)'}
                    className="h-6 px-1.5 rounded border border-slate-300 dark:border-slate-700 text-slate-500 hover:text-rose-500 hover:border-rose-500/50 text-[10.5px] font-sans whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    {lang === 'zh' ? '清除' : 'Clear'}
                  </button>
                )}
              </div>
            </div>

            {/* Progressive-Disclosure Inline Steps + Live Canvas Preview & Parameter Tuning */}
            <div className="space-y-1.5">
              {(() => {
                const autoCropStep = result?.stepPreviews.find(
                  (p) => p.stepId === AUTO_CROP_STEP_ID
                );
                const stepOffset = autoCropStep ? 1 : 0;
                const totalStepsCount = recipe.steps.length + stepOffset;

                return (
                  <>
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-semibold uppercase tracking-wider text-slate-500">
                        {lang === 'zh'
                          ? `流程步骤与实时参数 (${totalStepsCount}) · 修改即刻更新图像`
                          : `Live Pipeline Steps (${totalStepsCount}) · Instant Preview`}
                      </span>
                      {selectedStepId && (
                        <button
                          type="button"
                          onClick={() => onSelectStepId(null)}
                          className="text-blue-600 dark:text-cyan-400 hover:underline cursor-pointer"
                        >
                          {lang === 'zh' ? '返回结果视图' : 'Show Final'}
                        </button>
                      )}
                    </div>

                    {/* Automatic Step 1: Data Bar Crop (synced with top Pipeline Stepper) */}
                    {autoCropStep && (
                      <div
                        className={`border rounded-md overflow-hidden transition-colors ${
                          selectedStepId === AUTO_CROP_STEP_ID
                            ? 'border-blue-500/70 dark:border-cyan-500/70'
                            : 'border-slate-200 dark:border-slate-800'
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() =>
                            onSelectStepId(
                              selectedStepId === AUTO_CROP_STEP_ID ? null : AUTO_CROP_STEP_ID
                            )
                          }
                          className="w-full px-2.5 py-2 flex items-center justify-between gap-2 bg-slate-50/70 dark:bg-slate-900/50 text-left cursor-pointer"
                        >
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span className="font-mono text-[11px] text-cyan-600 dark:text-cyan-400">
                              1.
                            </span>
                            <span className="text-[12px] font-medium text-slate-900 dark:text-slate-100 truncate">
                              {lang === 'zh' ? autoCropStep.labelZh : autoCropStep.labelEn}
                            </span>
                            <span className="font-mono text-[10px] text-slate-500 truncate">
                              · {autoCropStep.qcSummary}
                            </span>
                          </div>
                          <span className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400 shrink-0">
                            {lang === 'zh' ? '自动' : 'Auto'}
                          </span>
                        </button>
                      </div>
                    )}

                    {recipe.steps.map((step, idx) => {
                      const spec = OPERATOR_CATALOG.find((o) => o.op === step.op);
                      const isOpen = openParamStepId === step.id;
                      const isPreviewing = selectedStepId === step.id;
                      const modifiedCount = spec
                        ? spec.params.filter((p) => (step.params[p.key] ?? p.defaultValue) !== p.defaultValue).length
                        : 0;
                      const displayIndex = idx + 1 + stepOffset;

                      return (
                        <div
                          key={step.id}
                          className={`border rounded-md overflow-hidden transition-colors ${
                            isPreviewing
                              ? 'border-blue-500/70 dark:border-cyan-500/70'
                              : 'border-slate-200 dark:border-slate-800'
                          }`}
                        >
                          <div className="px-2.5 py-2 flex items-center justify-between gap-1.5 bg-slate-50/70 dark:bg-slate-900/50">
                            <button
                              type="button"
                              onClick={() => {
                                const nextOpen = isOpen ? null : step.id;
                                setOpenParamStepId(nextOpen);
                                if (
                                  nextOpen &&
                                  (spec?.category === 'preprocess' ||
                                    spec?.category === 'segment' ||
                                    spec?.category === 'model')
                                ) {
                                  onSelectStepId(nextOpen);
                                } else {
                                  onSelectStepId(null);
                                }
                              }}
                              className="flex items-center gap-1.5 min-w-0 flex-1 text-left cursor-pointer"
                            >
                              {isOpen ? (
                                <ChevronDown className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                              ) : (
                                <ChevronRight className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                              )}
                              <span className="font-mono text-[11px] text-slate-400">
                                {displayIndex}.
                              </span>
                              <span
                                className={`text-[12px] font-medium truncate ${
                                  step.enabled
                                    ? 'text-slate-900 dark:text-slate-100'
                                    : 'text-slate-400 line-through'
                                }`}
                              >
                                {spec ? (lang === 'zh' ? spec.labelZh : spec.labelEn) : step.op}
                              </span>
                              {modifiedCount > 0 && (
                                <span className="text-[10px] font-mono text-amber-600 dark:text-amber-400 shrink-0">
                                  · {modifiedCount}
                                </span>
                              )}
                            </button>

                            <div className="flex items-center gap-1 shrink-0">
                              <input
                                type="checkbox"
                                checked={step.enabled}
                                onChange={() => onToggleStepEnabled(step.id)}
                                title={lang === 'zh' ? '启用/禁用该步骤' : 'Enable/disable step'}
                                className="accent-cyan-500 cursor-pointer"
                              />
                              <button
                                type="button"
                                disabled={idx === 0}
                                onClick={() => handleMoveStepInline(idx, -1)}
                                className="p-0.5 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 disabled:opacity-25 cursor-pointer"
                              >
                                <ArrowUp className="w-3 h-3" />
                              </button>
                              <button
                                type="button"
                                disabled={idx === recipe.steps.length - 1}
                                onClick={() => handleMoveStepInline(idx, 1)}
                                className="p-0.5 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 disabled:opacity-25 cursor-pointer"
                              >
                                <ArrowDown className="w-3 h-3" />
                              </button>
                              <button
                                type="button"
                                onClick={() => handleRemoveStepInline(step.id)}
                                className="p-0.5 text-slate-400 hover:text-rose-500 cursor-pointer"
                              >
                                <X className="w-3 h-3" />
                              </button>
                            </div>
                          </div>

                          {isOpen && (
                            <div className="p-3 space-y-2.5 border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0D121F]">
                              {step.op === 'model.segment' && (
                                <label className="block space-y-1 text-[11.5px]">
                                  <span className="text-slate-500 dark:text-slate-400 block">
                                    {lang === 'zh' ? '绑定模型权重' : 'Bound Model'}
                                  </span>
                                  <select
                                    value={step.modelRef ?? models[0]?.ref ?? ''}
                                    onChange={(e) => onChangeModelRef(step.id, e.target.value)}
                                    className="w-full h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px] text-slate-900 dark:text-slate-100"
                                  >
                                    {models.map((m) => (
                                      <option key={m.ref} value={m.ref}>
                                        {m.ref} — {lang === 'zh' ? m.nameZh : m.nameEn}
                                      </option>
                                    ))}
                                  </select>
                                </label>
                              )}

                              {spec?.params.map((p) => {
                                const val = step.params[p.key] ?? p.defaultValue;
                                return (
                                  <div key={p.key} className="space-y-1">
                                    <div className="flex items-center justify-between gap-2 text-[11.5px]">
                                      <span className="text-slate-700 dark:text-slate-300 truncate">
                                        {lang === 'zh' ? p.labelZh : p.labelEn}
                                      </span>
                                      {p.type === 'number' && (
                                        <div className="flex items-center gap-1 shrink-0">
                                          <input
                                            type="number"
                                            min={p.min ?? 0}
                                            max={p.max ?? 100}
                                            step={p.step ?? 1}
                                            value={Number(val)}
                                            onChange={(e) => {
                                              const parsed = Number(e.target.value);
                                              if (Number.isFinite(parsed)) {
                                                onUpdateStepParam(step.id, p.key, parsed);
                                              }
                                            }}
                                            className="w-16 h-6 px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px] tabular-nums text-right text-slate-900 dark:text-slate-100"
                                          />
                                          {p.unit && (
                                            <span className="font-mono text-[10.5px] text-slate-500">
                                              {p.unit}
                                            </span>
                                          )}
                                        </div>
                                      )}
                                    </div>
                                    {p.type === 'number' && (
                                      <input
                                        type="range"
                                        min={p.min ?? 0}
                                        max={p.max ?? 100}
                                        step={p.step ?? 1}
                                        value={Number(val)}
                                        onChange={(e) =>
                                          onUpdateStepParam(step.id, p.key, Number(e.target.value))
                                        }
                                        className="w-full accent-blue-600 dark:accent-cyan-400 cursor-pointer"
                                      />
                                    )}
                              {p.type === 'boolean' && (
                                <label className="inline-flex items-center gap-2 text-[11.5px] text-slate-600 dark:text-slate-300 cursor-pointer">
                                  <input
                                    type="checkbox"
                                    checked={Boolean(val)}
                                    onChange={(e) =>
                                      onUpdateStepParam(step.id, p.key, e.target.checked)
                                    }
                                    className="accent-cyan-500"
                                  />
                                  <span>
                                    {Boolean(val)
                                      ? lang === 'zh'
                                        ? '开启'
                                        : 'Enabled'
                                      : lang === 'zh'
                                      ? '关闭'
                                      : 'Disabled'}
                                  </span>
                                </label>
                              )}
                              {p.type === 'choice' && p.choices && (
                                <select
                                  value={String(val)}
                                  onChange={(e) =>
                                    onUpdateStepParam(step.id, p.key, e.target.value)
                                  }
                                  className="w-full h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[11.5px] text-slate-900 dark:text-slate-100"
                                >
                                  {p.choices.map((ch) => (
                                    <option key={ch.value} value={ch.value}>
                                      {lang === 'zh' ? ch.labelZh : ch.labelEn}
                                    </option>
                                  ))}
                                </select>
                              )}
                              {(p.helpZh || p.helpEn) && (
                                <p className="text-[10.5px] leading-snug text-slate-500 dark:text-slate-400">
                                  {lang === 'zh' ? p.helpZh ?? p.helpEn : p.helpEn ?? p.helpZh}
                                </p>
                              )}
                            </div>
                          );
                        })}

                        {modifiedCount > 0 && (
                          <div className="pt-1 flex justify-end">
                            <button
                              type="button"
                              onClick={() => onResetStepParams(step.id)}
                              className="text-[11px] text-slate-500 hover:text-blue-600 dark:hover:text-cyan-400 flex items-center gap-1 cursor-pointer"
                            >
                              <RotateCcw className="w-3 h-3" />
                              <span>{t('params_reset', lang)}</span>
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                        </div>
                      );
                    })}
                  </>
                );
              })()}
            </div>

            {/* Stage 2 Navigation CTAs */}
            <div className="grid grid-cols-2 gap-2 pt-1">
              <button
                type="button"
                onClick={() => onChangeStage('model')}
                className="h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-[12px] font-medium text-slate-700 dark:text-slate-200 flex items-center justify-center gap-1 cursor-pointer"
              >
                <Brain className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
                <span>{lang === 'zh' ? '模型 / 标注训练' : 'Models / Train'}</span>
              </button>
              {result ? (
                <button
                  type="button"
                  onClick={() => {
                    onSelectStepId(null);
                    onChangeStage('results');
                  }}
                  className="h-8 px-2.5 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[12px] font-semibold flex items-center justify-center gap-1 cursor-pointer"
                >
                  <span>{lang === 'zh' ? '查看分析结果' : 'View Results'}</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={onRunActiveWorkflow}
                  className="h-8 px-2.5 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[12px] font-semibold flex items-center justify-center gap-1 cursor-pointer"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{lang === 'zh' ? '确认并执行' : 'Run Workflow'}</span>
                </button>
              )}
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* STAGE 3: CHOOSE MODEL & SIDE-BY-SIDE MANUAL-LABEL RE-TRAINING     */}
        {/* ================================================================= */}
        {!compareMode && stage === 'model' && (
          <div className="space-y-4">
            <div className="text-[11.5px] text-slate-500 dark:text-slate-400 leading-relaxed">
              {lang === 'zh'
                ? '可选步骤：直接选用已有训练模型，或在左侧图像上涂抹前景/背景手工标注并即时微调重训，边标边看概率图。'
                : 'Optional: bind an existing trained model, or paint foreground/background strokes directly on the canvas to re-train side-by-side.'}
            </div>

            {/* Section A: Existing Trained Models */}
            <div className="space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                {lang === 'zh'
                  ? `A. 选用已有训练模型 (${models.length})`
                  : `A. Existing Trained Models (${models.length})`}
              </div>
              <div className="divide-y divide-slate-200 dark:divide-slate-800 border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden max-h-56 overflow-y-auto">
                {models.map((m) => {
                  const isBound = activeModelRef === m.ref;
                  return (
                    <div
                      key={m.ref}
                      className={`p-2.5 flex items-start justify-between gap-2 transition-colors ${
                        isBound
                          ? 'bg-blue-50/80 dark:bg-cyan-500/12'
                          : 'bg-white dark:bg-[#0D121F]'
                      }`}
                    >
                      <div className="min-w-0 space-y-0.5">
                        <div className="flex items-center gap-1.5 text-[12px] font-semibold text-slate-900 dark:text-slate-100">
                          <span className="truncate">{lang === 'zh' ? m.nameZh : m.nameEn}</span>
                          {m.isLocalTrained && (
                            <span className="text-[10px] font-mono text-amber-600 dark:text-amber-400 shrink-0">
                              · Local
                            </span>
                          )}
                        </div>
                        <div className="font-mono text-[10.5px] text-slate-500 dark:text-slate-400 truncate">
                          {m.ref} · {m.license}
                        </div>
                        <div className="font-mono text-[10.5px] text-slate-600 dark:text-slate-300 tabular-nums">
                          {m.metrics.meanIoU !== undefined && (
                            <span>mIoU {(m.metrics.meanIoU * 100).toFixed(1)}%</span>
                          )}
                          {m.metrics.macroF1 !== undefined && (
                            <span> · F1 {(m.metrics.macroF1 * 100).toFixed(1)}%</span>
                          )}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => onBindModel(m.ref)}
                        className={`h-7 px-2.5 rounded text-[11px] font-semibold shrink-0 cursor-pointer ${
                          isBound
                            ? 'bg-emerald-600/15 text-emerald-600 dark:text-emerald-400'
                            : 'bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950'
                        }`}
                      >
                        {isBound
                          ? lang === 'zh'
                            ? '✓ 已启用'
                            : '✓ Active'
                          : lang === 'zh'
                          ? '应用'
                          : 'Apply'}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Section A2: Model Occlusion Handling & Curvature/Overlap Boundary Fit Config */}
            {(() => {
              const modelStep = recipe.steps.find((s) => s.op === 'model.segment');
              const partStep = recipe.steps.find((s) => s.op === 'sem.particles');
              const curOcclusionMode: ModelOcclusionMode =
                occludedDisplayMode === 'hide' ? 'hide' : 'dashed';
              const curCurvatureFit =
                modelStep?.params.curvatureBoundaryFit !== undefined
                  ? Boolean(modelStep.params.curvatureBoundaryFit)
                  : partStep?.params.curvatureBoundaryFit !== undefined
                  ? Boolean(partStep.params.curvatureBoundaryFit)
                  : true;
              const curOverlapThresh = Number(
                modelStep?.params.overlapConfidenceThreshold ??
                  partStep?.params.overlapConfidenceThreshold ??
                  0.45
              );
              const occObjs = (result?.objects ?? []).filter((o) => !o.excluded && o.occluded);
              const totalCusps = (result?.objects ?? []).reduce(
                (acc, o) => acc + (o.concaveCuspCount ?? 0),
                0
              );
              const fitScores = (result?.objects ?? [])
                .map((o) => o.boundaryFitScore)
                .filter((v): v is number => v !== undefined);
              const avgFit =
                fitScores.length > 0
                  ? fitScores.reduce((a, b) => a + b, 0) / fitScores.length
                  : null;

              return (
                <div className="p-3 rounded-md border border-purple-500/30 bg-purple-50/40 dark:bg-[#100D1C] space-y-2.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-purple-700 dark:text-purple-300">
                      {lang === 'zh'
                        ? '遮挡处理与边界拟合优化 (Occlusion & Curvature Fit)'
                        : 'Occlusion Handling & Curvature Boundary Fit'}
                    </span>
                    {occObjs.length > 0 && (
                      <span className="px-1.5 py-0.5 rounded bg-purple-500/20 font-mono text-[10px] text-purple-700 dark:text-purple-200 tabular-nums">
                        {lang === 'zh' ? `遮挡 ${occObjs.length}` : `${occObjs.length} occl`}
                      </span>
                    )}
                  </div>

                  {/* Occlusion Handling Mode: Dashed Border vs Hide */}
                  <div className="space-y-1">
                    <div className="text-[11px] text-slate-600 dark:text-slate-300">
                      {lang === 'zh' ? '遮挡处理策略' : 'Occlusion Handling Mode'}
                    </div>
                    <div className="grid grid-cols-2 gap-1.5">
                      <button
                        type="button"
                        onClick={() => {
                          onChangeOccludedDisplayMode('highlight');
                          onUpdateModelOcclusionConfig({ occlusionMode: 'dashed' });
                        }}
                        className={`h-7 px-2 rounded border text-[11px] font-medium flex items-center justify-center gap-1.5 cursor-pointer transition-colors ${
                          curOcclusionMode === 'dashed'
                            ? 'border-purple-500 bg-purple-500/20 text-purple-800 dark:text-purple-200'
                            : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                        }`}
                      >
                        <span className="w-3 border-b-2 border-dashed border-purple-400 shrink-0" />
                        <span className="truncate">
                          {lang === 'zh' ? '虚线边框显示' : 'Dashed Border'}
                        </span>
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          onChangeOccludedDisplayMode('hide');
                          onUpdateModelOcclusionConfig({ occlusionMode: 'hide' });
                        }}
                        className={`h-7 px-2 rounded border text-[11px] font-medium flex items-center justify-center gap-1.5 cursor-pointer transition-colors ${
                          curOcclusionMode === 'hide'
                            ? 'border-purple-500 bg-purple-500/20 text-purple-800 dark:text-purple-200'
                            : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                        }`}
                      >
                        <span className="truncate">
                          {lang === 'zh' ? '隐藏被遮挡颗粒' : 'Hide Occluded'}
                        </span>
                      </button>
                    </div>
                  </div>

                  {/* Curvature & Overlap Confidence Boundary Optimization Toggle */}
                  <label className="flex items-start justify-between gap-2 pt-1 border-t border-purple-500/15 cursor-pointer select-none">
                    <div className="space-y-0.5">
                      <div className="text-[11px] font-medium text-slate-800 dark:text-slate-200">
                        {lang === 'zh'
                          ? '基于边缘曲率与重叠置信度优化边界拟合'
                          : 'Optimize Boundary Fit via Edge Curvature & Overlap'}
                      </div>
                      <div className="text-[10px] text-slate-500 dark:text-slate-400 leading-snug">
                        {lang === 'zh'
                          ? '提取轮廓有符号曲率凹角，沿高重叠置信接缝执行亚像素梯度吸附并保持晶体尖角。'
                          : 'Detects signed concave cusps and snaps high-overlap seams onto peak SE gradients while preserving crystal corners.'}
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={curCurvatureFit}
                      onChange={(e) =>
                        onUpdateModelOcclusionConfig({ curvatureBoundaryFit: e.target.checked })
                      }
                      className="mt-0.5 accent-purple-500 cursor-pointer shrink-0"
                    />
                  </label>

                  {/* Overlap Confidence Threshold Slider */}
                  <label className="block space-y-1 text-[11px]">
                    <div className="flex items-center justify-between font-mono text-slate-600 dark:text-slate-300">
                      <span>{lang === 'zh' ? '重叠置信度判定阈值' : 'Overlap Confidence Threshold'}</span>
                      <span className="text-purple-600 dark:text-purple-300 tabular-nums">
                        {Math.round(curOverlapThresh * 100)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0.2}
                      max={0.85}
                      step={0.05}
                      value={curOverlapThresh}
                      onChange={(e) =>
                        onUpdateModelOcclusionConfig({
                          overlapConfidenceThreshold: Number(e.target.value),
                        })
                      }
                      className="w-full accent-purple-500 cursor-pointer"
                    />
                  </label>

                  {(occObjs.length > 0 || totalCusps > 0 || avgFit !== null) && (
                    <div className="pt-1.5 border-t border-purple-500/15 flex items-center justify-between gap-2 font-mono text-[10px] text-slate-500 dark:text-slate-400 tabular-nums">
                      <span>
                        {lang === 'zh' ? `曲率凹角 ${totalCusps}` : `Cusps ${totalCusps}`}
                      </span>
                      {avgFit !== null && (
                        <span>
                          {lang === 'zh'
                            ? `拟合贴合度 ${(avgFit * 100).toFixed(0)}%`
                            : `Fit ${(avgFit * 100).toFixed(0)}%`}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Section B: Side-by-Side Manual Labeling & Active-Learning Re-Train */}
            <div className="p-3 rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-1.5">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-blue-600 dark:text-cyan-400 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>
                    {lang === 'zh'
                      ? 'B. 智能点选 GT 与本地重训'
                      : 'B. Smart GT Label & Re-Train'}
                  </span>
                </span>
                <div className="flex items-center gap-1 shrink-0">
                  <button
                    type="button"
                    onClick={() =>
                      onChangeToolMode(toolMode === 'smart_gt' ? 'pan' : 'smart_gt')
                    }
                    title={
                      lang === 'zh'
                        ? '使用 SAM 自动判断物体边界，单击即可生成整颗目标的 Ground Truth 标注'
                        : 'SAM click-to-segment: single click auto-detects object boundary as Ground Truth'
                    }
                    className={`h-7 px-2 rounded text-[11px] font-semibold flex items-center gap-1 whitespace-nowrap cursor-pointer transition-colors ${
                      toolMode === 'smart_gt'
                        ? 'bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950'
                        : 'border border-cyan-500/50 bg-cyan-500/10 text-blue-700 dark:text-cyan-300 hover:bg-cyan-500/20'
                    }`}
                  >
                    <Wand2 className="w-3 h-3 shrink-0" />
                    <span>
                      {toolMode === 'smart_gt'
                        ? lang === 'zh'
                          ? '点选中'
                          : 'Click GT On'
                        : lang === 'zh'
                        ? 'SAM 点选'
                        : 'Smart GT'}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      onChangeToolMode(toolMode === 'brush' ? 'pan' : 'brush')
                    }
                    className={`h-7 px-2 rounded text-[11px] font-medium flex items-center gap-1 whitespace-nowrap cursor-pointer transition-colors ${
                      toolMode === 'brush'
                        ? 'bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 font-semibold'
                        : 'border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200'
                    }`}
                  >
                    <Brush className="w-3 h-3 shrink-0" />
                    <span>{lang === 'zh' ? '画笔' : 'Brush'}</span>
                  </button>
                </div>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
                {lang === 'zh'
                  ? '点击「SAM 点选」后在画布上单击物体即可自动识别边界并填充为对应类别 GT（免手动涂色），也可用画笔补充精修。'
                  : 'Use Smart GT to auto-detect and fill full object boundaries with one click (no tedious painting), or refine with Brush.'}
              </p>

              {/* Training Scope Indicator: ROI + Upstream Processed Image */}
              <div className="px-2 py-1.5 rounded border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0D121F] flex flex-wrap items-center justify-between gap-2 text-[10.5px] font-mono">
                <div className="flex items-center gap-1.5 min-w-0 flex-1 text-slate-600 dark:text-slate-300">
                  <Crop className="w-3 h-3 text-cyan-500 shrink-0" />
                  <span className="truncate">
                    {roi
                      ? lang === 'zh'
                        ? `仅 ROI 内训练 (${roi.width}×${roi.height} px)`
                        : `ROI-only training (${roi.width}×${roi.height} px)`
                      : lang === 'zh'
                      ? '全图范围 · 建议框选 ROI 排除干扰'
                      : 'Full FOV · Set ROI to speed up'}
                    {result?.preModelGray
                      ? lang === 'zh'
                        ? ' · 基于前序处理图'
                        : ' · Uses processed image'
                      : ''}
                  </span>
                </div>
                <div className="flex items-center gap-1 shrink-0 font-sans">
                  {segmentedResultsRoi && (
                    <button
                      type="button"
                      onClick={() => {
                        onChangeRoi(segmentedResultsRoi);
                        if (toolMode === 'roi') onChangeToolMode('pan');
                      }}
                      title={
                        lang === 'zh'
                          ? `自动框选全部检测目标并留出 10% 边距 (${segmentedResultsRoi.width}×${segmentedResultsRoi.height} px)`
                          : `Set ROI to bounding box of all detected objects with 10% padding (${segmentedResultsRoi.width}×${segmentedResultsRoi.height} px)`
                      }
                      className="h-5 px-1.5 rounded border border-cyan-500/50 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 hover:bg-cyan-500/20 text-[10px] font-medium flex items-center gap-1 whitespace-nowrap shrink-0 cursor-pointer transition-colors"
                    >
                      <Crosshair className="w-2.5 h-2.5 shrink-0" />
                      <span>{lang === 'zh' ? '聚焦结果' : 'Focus on Results'}</span>
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => onChangeToolMode(toolMode === 'roi' ? 'pan' : 'roi')}
                    className={`h-5 px-1.5 rounded text-[10px] whitespace-nowrap shrink-0 cursor-pointer ${
                      toolMode === 'roi'
                        ? 'bg-cyan-500 text-slate-950 font-semibold'
                        : 'border border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-cyan-500/60'
                    }`}
                  >
                    {roi ? (lang === 'zh' ? '重画' : 'Redraw') : lang === 'zh' ? '框选 ROI' : 'Set ROI'}
                  </button>
                  {roi && (
                    <button
                      type="button"
                      onClick={() => onChangeRoi(null)}
                      className="h-5 px-1.5 rounded border border-slate-300 dark:border-slate-700 text-slate-500 hover:text-rose-500 text-[10px] whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      {lang === 'zh' ? '清除' : 'Clear'}
                    </button>
                  )}
                </div>
              </div>

              {/* Direct GT Class Selector synced with Canvas (Smart Click GT & Brush) */}
              <div className="grid grid-cols-3 gap-1.5">
                {BRUSH_CLASSES.map((bc) => {
                  const selected =
                    brushClassId === bc.id && (toolMode === 'smart_gt' || toolMode === 'brush');
                  return (
                    <button
                      key={bc.id}
                      type="button"
                      onClick={() => {
                        onChangeBrushClassId(bc.id);
                        if (toolMode !== 'brush' && toolMode !== 'smart_gt') {
                          onChangeToolMode('smart_gt');
                        }
                      }}
                      className={`h-7 px-1.5 rounded border text-[11px] font-medium flex items-center justify-center gap-1 cursor-pointer transition-colors ${
                        selected
                          ? 'border-blue-600 dark:border-cyan-400 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100'
                          : 'border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                      }`}
                    >
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: bc.color }}
                      />
                      <span className="truncate">{lang === 'zh' ? bc.zh : bc.en}</span>
                    </button>
                  );
                })}
              </div>

              {/* Brush Radius + Sample Counters */}
              <div className="flex items-center justify-between gap-3 text-[11px] font-mono">
                <label className="flex items-center gap-2 flex-1">
                  <span className="text-slate-500 shrink-0">
                    {lang === 'zh' ? '笔刷' : 'Size'} {brushRadius}px
                  </span>
                  <input
                    type="range"
                    min={3}
                    max={28}
                    value={brushRadius}
                    onChange={(e) => onChangeBrushRadius(Number(e.target.value))}
                    className="flex-1 accent-cyan-500"
                  />
                </label>
                {strokes.length > 0 && (
                  <button
                    type="button"
                    onClick={onClearStrokes}
                    className="text-slate-400 hover:text-rose-500 shrink-0 cursor-pointer"
                  >
                    {lang === 'zh' ? `清空标注 (${strokes.length})` : `Clear (${strokes.length})`}
                  </button>
                )}
              </div>

              {(() => {
                const samGtCount = strokes.filter((s) => s.source === 'sam_click' || Boolean(s.polygon)).length;
                const brushOnlyCount = strokes.length - samGtCount;
                return (
                  <div className="grid grid-cols-3 gap-2 py-1.5 border-y border-slate-200 dark:border-slate-800 font-mono text-[11px] tabular-nums text-center">
                    <div>
                      <div className="text-slate-500 text-[10px]">{lang === 'zh' ? 'SAM 轮廓 GT' : 'SAM GT'}</div>
                      <div className="font-semibold text-cyan-600 dark:text-cyan-400">{samGtCount}</div>
                    </div>
                    <div>
                      <div className="text-slate-500 text-[10px]">{lang === 'zh' ? '画笔笔迹' : 'Brush'}</div>
                      <div className="font-semibold text-slate-900 dark:text-slate-100">{brushOnlyCount}</div>
                    </div>
                    <div>
                      <div className="text-slate-500 text-[10px]">{lang === 'zh' ? '人工纠错' : 'Edits'}</div>
                      <div className="font-semibold text-slate-900 dark:text-slate-100">{specCorrections.length}</div>
                    </div>
                  </div>
                );
              })()}

              {/* Model ID + Version + Train Button */}
              <div className="grid grid-cols-3 gap-1.5 text-[11px]">
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="model_id"
                  className="col-span-2 h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono"
                />
                <input
                  type="text"
                  value={modelVer}
                  onChange={(e) => setModelVer(e.target.value)}
                  placeholder="1.0.0"
                  className="h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono"
                />
              </div>

              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  disabled={inlineTraining}
                  onClick={handleTrainSideBySide}
                  className="flex-1 h-8 px-3 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[12px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-55"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>
                    {inlineTraining
                      ? lang === 'zh'
                        ? 'Web Worker 微调训练中…'
                        : 'Training in Worker…'
                      : lang === 'zh'
                      ? '立即训练并评估 (25% 留出验证)'
                      : 'Train & Evaluate (25% Hold-Out)'}
                  </span>
                </button>
                {inlineTraining && (
                  <button
                    type="button"
                    onClick={handleAbortInlineTraining}
                    className="h-8 px-2.5 rounded border border-rose-500/45 bg-rose-500/12 hover:bg-rose-500/20 text-rose-600 dark:text-rose-300 text-[11.5px] font-semibold flex items-center gap-1 cursor-pointer shrink-0"
                  >
                    <Square className="w-3 h-3 fill-current" />
                    <span>{lang === 'zh' ? '中止' : 'Abort'}</span>
                  </button>
                )}
              </div>

              {(inlineTraining || inlineCurve.length > 0) && (() => {
                const pts: TrainingEpochPoint[] = inlineCurve;
                const pct = inlineProgress
                  ? Math.min(100, Math.round((inlineProgress.epoch / Math.max(1, inlineProgress.totalEpochs)) * 100))
                  : pts.length > 0
                  ? 100
                  : 0;
                const w = 260;
                const h = 68;
                const pad = 6;
                const maxLoss = Math.max(0.2, ...pts.map((p) => p.trainLoss));
                const lossPath = pts
                  .map((p, i) => {
                    const x = pad + (i / Math.max(1, pts.length - 1)) * (w - pad * 2);
                    const y = pad + (1 - Math.min(1, p.trainLoss / maxLoss)) * (h - pad * 2);
                    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
                  })
                  .join(' ');
                const iouPath = pts
                  .map((p, i) => {
                    const x = pad + (i / Math.max(1, pts.length - 1)) * (w - pad * 2);
                    const y = pad + (1 - Math.max(0, Math.min(1, p.valMeanIoU))) * (h - pad * 2);
                    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
                  })
                  .join(' ');
                const latest = pts[pts.length - 1];
                return (
                  <div className="p-2 rounded border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#090D16] space-y-1.5 font-mono text-[10.5px]">
                    <div className="flex items-center justify-between text-slate-500 dark:text-slate-400">
                      <span className="truncate">
                        {inlineProgress
                          ? inlineProgress.phase === 'extracting'
                            ? lang === 'zh'
                              ? '提取 16D 多尺度特征金字塔…'
                              : 'Extracting 16D feature pyramid…'
                            : `Epoch ${inlineProgress.epoch}/${inlineProgress.totalEpochs}`
                          : lang === 'zh'
                          ? '实时训练曲线 (Loss / Val mIoU)'
                          : 'Live Training Curve (Loss / Val mIoU)'}
                      </span>
                      <span className="tabular-nums text-cyan-600 dark:text-cyan-400">
                        {latest ? `IoU ${(latest.valMeanIoU * 100).toFixed(1)}%` : `${pct}%`}
                      </span>
                    </div>
                    {inlineTraining && (
                      <div className="h-1.5 w-full rounded bg-slate-200 dark:bg-slate-800 overflow-hidden">
                        <div
                          className="h-full bg-blue-600 dark:bg-cyan-400 transition-all"
                          style={{ width: `${Math.max(8, pct)}%` }}
                        />
                      </div>
                    )}
                    {pts.length >= 2 && (
                      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[68px] block">
                        <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="currentColor" strokeOpacity={0.15} />
                        <path d={lossPath} fill="none" stroke="#f59e0b" strokeWidth={1.6} />
                        <path d={iouPath} fill="none" stroke="#06b6d4" strokeWidth={1.8} />
                      </svg>
                    )}
                    <div className="flex items-center justify-between text-[10px] text-slate-400">
                      <span className="text-amber-500">
                        — Loss {latest ? latest.trainLoss.toFixed(3) : '—'}
                      </span>
                      <span className="text-cyan-500">
                        — Val mIoU {latest ? `${(latest.valMeanIoU * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                  </div>
                );
              })()}

              {trainingError && (
                <div className="text-[11px] text-amber-600 dark:text-amber-400 leading-snug">
                  ▲ {trainingError}
                </div>
              )}

              {lastTrained && (
                <div className="pt-2 border-t border-slate-200 dark:border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                      <ShieldCheck className="w-3.5 h-3.5" />
                      <span>{lang === 'zh' ? '留出评估完成' : 'Hold-Out Verified'}</span>
                    </span>
                    <span className="font-mono text-slate-400">
                      {lastTrained.metrics.nValPixels} val px
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 font-mono text-[11.5px] tabular-nums">
                    <div>
                      <div className="text-[10px] text-slate-500">mIoU</div>
                      <div className="font-semibold text-blue-600 dark:text-cyan-400">
                        {(lastTrained.metrics.meanIoU * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-slate-500">Macro F1</div>
                      <div className="font-semibold text-slate-900 dark:text-slate-100">
                        {(lastTrained.metrics.macroF1 * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-slate-500">Acc</div>
                      <div className="font-semibold text-slate-900 dark:text-slate-100">
                        {(lastTrained.metrics.accuracy * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      onPublishLocalModel(lastTrained);
                      onChangeToolMode('pan');
                    }}
                    className="w-full h-8 px-3 rounded border border-emerald-600/50 bg-emerald-600/12 hover:bg-emerald-600/20 text-emerald-700 dark:text-emerald-300 text-[12px] font-semibold cursor-pointer"
                  >
                    {lang === 'zh'
                      ? `发布并立即应用 ${lastTrained.card.ref} 到画布`
                      : `Publish & Apply ${lastTrained.card.ref} Live`}
                  </button>
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={() => {
                onSelectStepId(null);
                onChangeStage('results');
              }}
              className="w-full h-8 px-3 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-[12px] font-medium text-slate-700 dark:text-slate-200 flex items-center justify-center gap-1.5 cursor-pointer"
            >
              <span>{lang === 'zh' ? '下一步：查看量化结果与复核' : 'Next: View Results & Review'}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* ================================================================= */}
        {/* STAGE 4: DISPLAY RESULTS, HUMAN REVIEW & PROVENANCE               */}
        {/* ================================================================= */}
        {!compareMode && stage === 'results' && !result && (
          <div className="p-4 rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] space-y-3 text-[12px]">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div className="font-semibold text-slate-900 dark:text-slate-100">
                  {lang === 'zh' ? '尚未执行图像分析' : 'Analysis Not Yet Executed'}
                </div>
                <p className="text-[11.5px] text-slate-600 dark:text-slate-300 leading-relaxed">
                  {lang === 'zh'
                    ? '当前图像处于半自动待确认状态。请先在「1. 导入」确认信息栏裁切与物理标定，或在「2. 流程」选定工作流并点击执行。'
                    : 'This image is in semi-auto staging. Confirm info-bar crop & calibration in Stage 1, or choose a workflow and click Run.'}
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => onChangeStage('import')}
                className="h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11.5px] font-medium text-slate-700 dark:text-slate-200 cursor-pointer"
              >
                {lang === 'zh' ? '前往 1. 导入确认' : 'Go to 1. Import'}
              </button>
              <button
                type="button"
                onClick={() => {
                  onRunActiveWorkflow();
                  onChangeStage('results');
                }}
                className="h-8 px-2.5 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[11.5px] font-semibold flex items-center justify-center gap-1 cursor-pointer"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>{lang === 'zh' ? '立即执行分析' : 'Run Now'}</span>
              </button>
            </div>
          </div>
        )}

        {!compareMode && stage === 'results' && result && (
          <div className="space-y-4">
            {/* Sub-view Switcher inside Results */}
            <div className="grid grid-cols-3 p-0.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#0B0F1A] text-[11.5px] font-medium">
              {(
                [
                  { id: 'metrics', zh: '指标与图表', en: 'Metrics' },
                  { id: 'review', zh: '人工复核', en: 'Review' },
                  { id: 'provenance', zh: '溯源与论文', en: 'Provenance' },
                ] as Array<{ id: ResultsSubTab; zh: string; en: string }>
              ).map((sub) => {
                const active = resultsSubTab === sub.id;
                return (
                  <button
                    key={sub.id}
                    type="button"
                    onClick={() => onChangeResultsSubTab(sub.id)}
                    className={`py-1.5 rounded text-center transition-colors cursor-pointer ${
                      active
                        ? 'bg-white dark:bg-slate-800 text-blue-600 dark:text-cyan-400 font-semibold'
                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                    }`}
                  >
                    {lang === 'zh' ? sub.zh : sub.en}
                    {sub.id === 'review' && flaggedCount > 0 && (
                      <span className="ml-1 font-mono text-[10px] text-amber-500">
                        {flaggedCount}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Automatic Low-Contrast / Low-Confidence Red-Box Highlight Bar */}
            {(() => {
              const dismissedSet = new Set(dismissedLowConfIds);
              const lowSegs = (result.lowConfidenceSegments ?? []).filter(
                (s) => s.confidence < lowConfThreshold && !dismissedSet.has(s.id)
              );
              return (
                <div
                  className={`p-2.5 rounded-md border space-y-2 transition-colors ${
                    lowSegs.length > 0 && showLowConfHighlight
                      ? 'border-rose-500/40 bg-rose-500/6'
                      : 'border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 text-[11px]">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <span
                        className={`w-2 h-2 rounded-full shrink-0 ${
                          lowSegs.length > 0 && showLowConfHighlight
                            ? 'bg-rose-500'
                            : 'bg-slate-400 dark:bg-slate-600'
                        }`}
                      />
                      <span className="font-semibold text-slate-800 dark:text-slate-200 truncate">
                        {lowSegs.length > 0
                          ? lang === 'zh'
                            ? `低对比/低置信自动红框 (${lowSegs.length} 处)`
                            : `Low-Contrast Red Boxes (${lowSegs.length})`
                          : lang === 'zh'
                          ? '低置信红框自动检测 · 当前全部达标'
                          : 'Low-Confidence Check · All Above Threshold'}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={onToggleLowConfHighlight}
                      className={`h-5 px-2 rounded text-[10px] font-mono shrink-0 cursor-pointer transition-colors ${
                        showLowConfHighlight
                          ? 'bg-rose-500/20 text-rose-600 dark:text-rose-300 border border-rose-500/45'
                          : 'border border-slate-300 dark:border-slate-700 text-slate-500'
                      }`}
                    >
                      {showLowConfHighlight
                        ? lang === 'zh'
                          ? '红框: 开'
                          : 'Box: ON'
                        : lang === 'zh'
                        ? '红框: 关'
                        : 'Box: OFF'}
                    </button>
                  </div>

                  <div className="flex items-center gap-2 font-mono text-[10.5px] text-slate-500 dark:text-slate-400">
                    <span className="shrink-0">
                      {lang === 'zh' ? '置信度阈值' : 'Conf Thresh'}:{' '}
                      <strong className="text-rose-600 dark:text-rose-300 tabular-nums">
                        {Math.round(lowConfThreshold * 100)}%
                      </strong>
                    </span>
                    <input
                      type="range"
                      min={0.35}
                      max={0.85}
                      step={0.05}
                      value={lowConfThreshold}
                      onChange={(e) => onChangeLowConfThreshold(Number(e.target.value))}
                      className="flex-1 accent-rose-500"
                    />
                  </div>

                  {lowSegs.length > 0 && (
                    <div className="space-y-1">
                      <div className="text-[10px] text-slate-500 dark:text-slate-400">
                        {lang === 'zh'
                          ? '点击画布上的微弱红框或下方片段，立即触发手动纠偏：'
                          : 'Click any subtle red box on canvas or below to trigger manual correction:'}
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {lowSegs.slice(0, 8).map((seg) => {
                          const isAct = activeCorrectionSegId === seg.id;
                          return (
                            <button
                              key={seg.id}
                              type="button"
                              onClick={() => onTriggerLowConfCorrection(seg)}
                              title={lang === 'zh' ? seg.reasonZh : seg.reasonEn}
                              className={`h-6 px-2 rounded border font-mono text-[10.5px] tabular-nums flex items-center gap-1 cursor-pointer transition-colors ${
                                isAct
                                  ? 'border-rose-500 bg-rose-500/25 text-rose-700 dark:text-rose-200 font-semibold'
                                  : 'border-rose-500/35 bg-white/80 dark:bg-slate-900/70 text-rose-600 dark:text-rose-300 hover:border-rose-500/70'
                              }`}
                            >
                              <span>{lang === 'zh' ? seg.labelZh : seg.labelEn}</span>
                              <span className="opacity-75">{Math.round(seg.confidence * 100)}%</span>
                            </button>
                          );
                        })}
                        {lowSegs.length > 0 && (
                          <button
                            type="button"
                            onClick={() => lowSegs.forEach((s) => onDismissLowConfSegment(s.id))}
                            className="h-6 px-1.5 rounded text-[10px] text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 cursor-pointer"
                          >
                            {lang === 'zh' ? '全部忽略' : 'Dismiss All'}
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* SUB-TAB 4A: METRICS & CHARTS (Unified 2D / 3D Switcher) */}
            {resultsSubTab === 'metrics' && (
              <>
                {engineCard}
                <div className="flex items-center justify-between gap-2 pb-1">
                  <div className="grid grid-cols-2 p-0.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#0B0F1A] text-[11px] font-medium flex-1">
                    <button
                      type="button"
                      onClick={() => setMetricsDim('2d')}
                      className={`py-1 rounded text-center cursor-pointer transition-colors ${
                        metricsDim === '2d'
                          ? 'bg-white dark:bg-slate-800 text-blue-600 dark:text-cyan-400 font-semibold'
                          : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                      }`}
                    >
                      {lang === 'zh' ? '2D 平面指标' : '2D Metrics'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setMetricsDim('3d')}
                      className={`py-1 rounded text-center cursor-pointer transition-colors flex items-center justify-center gap-1 ${
                        metricsDim === '3d'
                          ? 'bg-white dark:bg-slate-800 text-blue-600 dark:text-cyan-400 font-semibold'
                          : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                      }`}
                    >
                      <Box className="w-3 h-3" />
                      <span>
                        {lang === 'zh' ? '3D 体计量' : '3D Volume'}
                        {volumeAnalysis ? ` (${volumeAnalysis.cellCount})` : ''}
                      </span>
                    </button>
                  </div>
                </div>

                {metricsDim === '3d' ? (
                  <div className="space-y-3">
                    {volumeAnalysis ? (
                      <>
                        <div className="grid grid-cols-2 gap-2 font-mono tabular-nums">
                          <div className="p-2 rounded border border-slate-200 dark:border-slate-800">
                            <div className="text-[10px] text-slate-500">
                              {lang === 'zh' ? '3D 细胞 / 嵌套核' : '3D Cells / Nuclei'}
                            </div>
                            <div className="text-[16px] font-semibold text-blue-600 dark:text-cyan-400">
                              {volumeAnalysis.cellCount} / {volumeAnalysis.nucleusCount}
                            </div>
                          </div>
                          <div className="p-2 rounded border border-slate-200 dark:border-slate-800">
                            <div className="text-[10px] text-slate-500">
                              {lang === 'zh' ? '平均 3D 核质体积比' : 'Mean 3D N/C Vol'}
                            </div>
                            <div className="text-[16px] font-semibold text-slate-900 dark:text-slate-100">
                              {volumeAnalysis.meanNcVolumePct.toFixed(1)}%
                            </div>
                          </div>
                          <div className="p-2 rounded border border-slate-200 dark:border-slate-800">
                            <div className="text-[10px] text-slate-500">
                              {lang === 'zh'
                                ? `平均细胞体积 (${volumeAnalysis.spaceUnit}³)`
                                : `Mean Vol (${volumeAnalysis.spaceUnit}³)`}
                            </div>
                            <div className="text-[14px] font-semibold text-slate-900 dark:text-slate-100">
                              {volumeAnalysis.meanCellVolume.toFixed(1)}
                            </div>
                          </div>
                          <div className="p-2 rounded border border-slate-200 dark:border-slate-800">
                            <div className="text-[10px] text-slate-500">
                              {lang === 'zh' ? '平均三维球度 Ψ' : 'Mean Sphericity Ψ'}
                            </div>
                            <div className="text-[14px] font-semibold text-slate-900 dark:text-slate-100">
                              {volumeAnalysis.meanSphericity.toFixed(2)} · {volumeAnalysis.durationMs}ms
                            </div>
                          </div>
                        </div>

                        <NcRatioHistogramPanel
                          dimension="3d"
                          volumeMetrics={volumeAnalysis.objects}
                          selectedObjectId={null}
                          onSelectObject={() => onOpen3DStudio?.()}
                          spaceUnit={volumeAnalysis.spaceUnit}
                          lang={lang}
                        />

                        {volumeAnalysis.objects.length > 0 && (
                          <div className="rounded border border-slate-200 dark:border-slate-800 overflow-hidden">
                            <table className="w-full text-left border-collapse text-[11px] font-mono tabular-nums">
                              <thead className="bg-slate-50 dark:bg-[#0B0F1A] text-slate-500 border-b border-slate-200 dark:border-slate-800">
                                <tr>
                                  <th className="py-1 px-2">ID</th>
                                  <th className="py-1 px-2">Vol ({volumeAnalysis.spaceUnit}³)</th>
                                  <th className="py-1 px-2">PCA L1/L2</th>
                                  <th className="py-1 px-2 text-right">N/C %</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800/70">
                                {volumeAnalysis.objects.slice(0, 6).map((cell) => (
                                  <tr key={cell.id} className="text-slate-700 dark:text-slate-300">
                                    <td className="py-1 px-2 font-semibold">#{cell.id}</td>
                                    <td className="py-1 px-2">{cell.volumePhys.toFixed(1)}</td>
                                    <td className="py-1 px-2 text-slate-500">
                                      {cell.elongation !== undefined ? `${cell.elongation.toFixed(2)}×` : '—'}
                                    </td>
                                    <td className="py-1 px-2 text-right text-blue-600 dark:text-cyan-400">
                                      {cell.ncVolumePct !== undefined ? `${cell.ncVolumePct.toFixed(1)}%` : '—'}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}

                        <div className="flex items-center gap-2">
                          {onRunQuick3DAnalysis && (
                            <button
                              type="button"
                              disabled={isAnalyzing3D}
                              onClick={onRunQuick3DAnalysis}
                              className="flex-1 h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11.5px] font-medium cursor-pointer disabled:opacity-50"
                            >
                              {isAnalyzing3D
                                ? lang === 'zh'
                                  ? '3D 重算中…'
                                  : 'Re-running…'
                                : lang === 'zh'
                                ? '重新运行 3D 分析'
                                : 'Re-run 3D Analysis'}
                            </button>
                          )}
                          {onOpen3DStudio && (
                            <button
                              type="button"
                              onClick={onOpen3DStudio}
                              className="flex-1 h-8 px-2.5 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[11.5px] font-semibold cursor-pointer"
                            >
                              {lang === 'zh' ? '进入 3D 实验室' : 'Open 3D Lab'}
                            </button>
                          )}
                        </div>
                      </>
                    ) : (
                      <div className="p-3.5 rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] space-y-2.5 text-[12px]">
                        <div className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                          <Box className="w-4 h-4 text-blue-600 dark:text-cyan-400" />
                          <span>
                            {lang === 'zh' ? '尚未运行 3D 体积与核质比计量' : '3D Volumetric Analysis Not Run Yet'}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
                          {lang === 'zh'
                            ? '可直接在后台 Web Worker 对内置 3D Z-Stack 执行 26-邻域分割与 PCA 主轴计量，或切换至完整 3D 实验室导入多页 OME-TIFF。'
                            : 'Run 26-connectivity 3D segmentation & PCA principal-axis metrology in a background Worker, or open the full 3D Lab to load a multi-page OME-TIFF Z-Stack.'}
                        </p>
                        <div className="flex items-center gap-2">
                          {onRunQuick3DAnalysis && (
                            <button
                              type="button"
                              disabled={isAnalyzing3D}
                              onClick={onRunQuick3DAnalysis}
                              className="flex-1 h-8 px-2.5 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[11.5px] font-semibold cursor-pointer disabled:opacity-50"
                            >
                              {isAnalyzing3D
                                ? lang === 'zh'
                                  ? 'Worker 3D 分析中…'
                                  : 'Analyzing 3D…'
                                : lang === 'zh'
                                ? '一键运行 3D 体分析'
                                : 'Run Quick 3D Analysis'}
                            </button>
                          )}
                          {onOpen3DStudio && (
                            <button
                              type="button"
                              onClick={onOpen3DStudio}
                              className="h-8 px-3 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11.5px] font-medium cursor-pointer"
                            >
                              {lang === 'zh' ? '3D 实验室' : '3D Lab'}
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <>
                <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                  {result.summary.map((m) => (
                    <div
                      key={m.key}
                      className="pb-2.5 border-b border-slate-100 dark:border-slate-800/80 min-w-0"
                    >
                      <div className="text-[11px] text-slate-500 dark:text-slate-400 truncate">
                        {lang === 'zh' ? m.labelZh : m.labelEn}
                      </div>
                      <div className="mt-0.5 flex items-baseline gap-1 font-mono tabular-nums">
                        <span
                          className={`text-[19px] font-semibold tracking-tight ${
                            m.highlight
                              ? 'text-blue-600 dark:text-cyan-400'
                              : 'text-slate-900 dark:text-slate-100'
                          }`}
                        >
                          {m.value ?? '—'}
                        </span>
                        {m.unit && (
                          <span className="text-[11px] text-slate-500 dark:text-slate-400">
                            {m.unit}
                          </span>
                        )}
                      </div>
                      {(m.detailZh || m.detailEn) && (
                        <div className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400 truncate">
                          {lang === 'zh' ? m.detailZh : m.detailEn}
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {(result.warnings.length > 0 || anomalySuggestions.length > 0) && (
                  <div className="pt-1 space-y-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                        {lang === 'zh' ? '质控提示 (QC)' : 'QC Advisories'}
                      </span>
                      {anomalySuggestions.length > 0 && (
                        <button
                          type="button"
                          onClick={() => onChangeResultsSubTab('review')}
                          className="text-[11px] font-medium text-blue-600 dark:text-cyan-400 hover:underline flex items-center gap-1 cursor-pointer"
                        >
                          <Wrench className="w-3 h-3" />
                          <span>
                            {lang === 'zh'
                              ? `${anomalySuggestions.length} 项一键修复`
                              : `${anomalySuggestions.length} Quick Fix(es)`}
                          </span>
                        </button>
                      )}
                    </div>
                    {result.warnings.map((w, idx) => (
                      <div
                        key={idx}
                        className="flex items-start gap-1.5 text-[12px] text-amber-700 dark:text-amber-300/90"
                      >
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                        <span>{w}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Interactive Virtual 3D Preview (WebGL2 2.5D Surface + Open 3D Lab shortcut) */}
                <Virtual3DPreviewWidget
                  specimen={specimen}
                  result={result}
                  lang={lang}
                  selectedObjectId={selectedObjectId}
                  occludedDisplayMode={occludedDisplayMode}
                  roi={roi}
                  onOpen3DStudio={onOpen3DStudio}
                />

                {/* D3 Nuclear-to-Cytoplasmic Area Ratio Histogram Panel */}
                {result.objects.some(
                  (o) => !o.excluded && (o.subArea !== undefined || o.subAreaRatioPct !== undefined)
                ) ? (
                  <NcRatioHistogramPanel
                    objects={result.objects}
                    selectedObjectId={selectedObjectId}
                    onSelectObject={onSelectObject}
                    spaceUnit={specimen.calibration.spaceUnit}
                    lang={lang}
                  />
                ) : (
                  result.histogram && <DistributionHistogram data={result.histogram} lang={lang} />
                )}
                {result.poissonChart && (
                  <PoissonComparisonChart data={result.poissonChart} lang={lang} />
                )}
                {result.profileChart && (
                  <IntensityProfileChart
                    positions={result.profileChart.positions}
                    intensities={result.profileChart.intensities}
                    edges={result.profileChart.edges}
                    unit={result.profileChart.unit}
                    title={
                      result.objects.some((o) => o.kind === 'layer')
                        ? lang === 'zh'
                          ? '中心多层截面灰度剖面（含检测界面）'
                          : 'Center Multilayer Intensity Profile'
                        : lang === 'zh'
                        ? 'SEM 扫描线灰度剖面'
                        : 'SEM Intensity Profile'
                    }
                  />
                )}
                {latestProfileMeasurement?.profile && (
                  <div className="space-y-2">
                    <IntensityProfileChart
                      positions={latestProfileMeasurement.profile.positions}
                      intensities={latestProfileMeasurement.profile.intensities}
                      edges={latestProfileMeasurement.profile.edges}
                      unit={latestProfileMeasurement.unit}
                      title={
                        lang === 'zh'
                          ? `手动线剖面 (${latestProfileMeasurement.lengthPhys} ${latestProfileMeasurement.unit})`
                          : `Manual Profile (${latestProfileMeasurement.lengthPhys} ${latestProfileMeasurement.unit})`
                      }
                    />
                    {(latestProfileMeasurement.profile.layerCount ?? 0) > 0 && (
                      <div className="p-2.5 rounded border border-violet-500/35 bg-violet-500/8 space-y-1.5 font-mono text-[11px] tabular-nums">
                        <div className="flex items-center justify-between gap-2 text-violet-700 dark:text-violet-300 font-semibold">
                          <span>
                            {lang === 'zh'
                              ? `沿线穿过层数: ${latestProfileMeasurement.profile.layerCount} 层`
                              : `Line Profile Layers: ${latestProfileMeasurement.profile.layerCount}`}
                          </span>
                          <span>
                            {lang === 'zh' ? '均厚' : 'Mean'}{' '}
                            {latestProfileMeasurement.profile.meanLayerThickness?.toFixed(2) ?? '—'}{' '}
                            {latestProfileMeasurement.unit}
                          </span>
                        </div>
                        <div className="flex items-center flex-wrap gap-x-3 gap-y-0.5 text-[10.5px] text-slate-600 dark:text-slate-400">
                          {latestProfileMeasurement.profile.totalStackThickness !== null &&
                            latestProfileMeasurement.profile.totalStackThickness !== undefined && (
                              <span>
                                {lang === 'zh' ? '总厚' : 'Total'}:{' '}
                                <strong className="text-slate-800 dark:text-slate-200">
                                  {latestProfileMeasurement.profile.totalStackThickness.toFixed(2)}{' '}
                                  {latestProfileMeasurement.unit}
                                </strong>
                              </span>
                            )}
                          {latestProfileMeasurement.profile.fftPeriodPhys !== null &&
                            latestProfileMeasurement.profile.fftPeriodPhys !== undefined && (
                              <span>
                                1D FFT {lang === 'zh' ? '主周期' : 'Period'}:{' '}
                                <strong className="text-cyan-600 dark:text-cyan-400">
                                  {latestProfileMeasurement.profile.fftPeriodPhys.toFixed(2)}{' '}
                                  {latestProfileMeasurement.unit}
                                </strong>
                              </span>
                            )}
                        </div>
                        {latestProfileMeasurement.profile.layerThicknesses &&
                          latestProfileMeasurement.profile.layerThicknesses.length > 0 && (
                            <div className="flex flex-wrap gap-1 pt-1 border-t border-violet-500/20 text-[10px] text-slate-700 dark:text-slate-300">
                              {latestProfileMeasurement.profile.layerThicknesses.map((th, i) => (
                                <span
                                  key={i}
                                  className="px-1.5 py-0.5 rounded bg-white/70 dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800"
                                >
                                  L{i + 1}: {th} {latestProfileMeasurement.unit}
                                </span>
                              ))}
                            </div>
                          )}
                      </div>
                    )}
                  </div>
                )}

                {/* Clickable Multilayer Stack Layer-by-Layer Thickness Breakdown */}
                {result.objects.some((o) => o.kind === 'layer') && (
                  <div className="pt-3 border-t border-slate-200 dark:border-slate-800 space-y-1.5">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-semibold uppercase tracking-wider text-slate-500">
                        {lang === 'zh'
                          ? `中心逐层厚度明细 (${result.objects.filter((o) => o.kind === 'layer' && !o.excluded).length} 层)`
                          : `Layer-by-Layer Thickness (${result.objects.filter((o) => o.kind === 'layer' && !o.excluded).length})`}
                      </span>
                      {selectedObjectId !== null && (
                        <button
                          type="button"
                          onClick={() => onSelectObject(null)}
                          className="text-blue-600 dark:text-cyan-400 hover:underline cursor-pointer"
                        >
                          {lang === 'zh' ? '取消高亮' : 'Clear'}
                        </button>
                      )}
                    </div>
                    <div className="divide-y divide-slate-100 dark:divide-slate-800/80 border-y border-slate-200 dark:border-slate-800 max-h-60 overflow-y-auto">
                      {result.objects
                        .filter((o) => o.kind === 'layer')
                        .map((lyr) => {
                          const isSel = lyr.id === selectedObjectId;
                          const thk = lyr.thickness ?? lyr.width;
                          return (
                            <div
                              key={lyr.id}
                              onClick={() => onSelectObject(isSel ? null : lyr.id)}
                              className={`w-full py-2 px-2 text-left transition-colors cursor-pointer flex items-center justify-between gap-2 ${
                                isSel
                                  ? 'bg-blue-50/90 dark:bg-cyan-500/15'
                                  : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                              } ${lyr.excluded ? 'opacity-45' : ''}`}
                            >
                              <div className="min-w-0">
                                <div className="flex items-center gap-2 font-mono text-[12px] tabular-nums">
                                  <span
                                    className={`font-semibold ${
                                      isSel
                                        ? 'text-blue-600 dark:text-cyan-300'
                                        : 'text-slate-900 dark:text-slate-100'
                                    }`}
                                  >
                                    L#{lyr.id}
                                  </span>
                                  <span className="text-slate-800 dark:text-slate-200 font-medium">
                                    {thk !== undefined
                                      ? `${thk.toFixed(2)} ${specimen.calibration.spaceUnit}`
                                      : '—'}
                                  </span>
                                </div>
                                <div className="mt-0.5 text-[10.5px] font-mono text-slate-500 dark:text-slate-400 truncate">
                                  {lyr.reason ?? ''}
                                </div>
                              </div>
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onAddCorrection({
                                    specimenId: specimen.id,
                                    objectId: lyr.id,
                                    field: 'excluded',
                                    oldValue: Boolean(lyr.excluded),
                                    newValue: !lyr.excluded,
                                  });
                                }}
                                className="text-[10.5px] font-mono text-slate-400 hover:text-rose-400 shrink-0 cursor-pointer"
                              >
                                {lyr.excluded
                                  ? lang === 'zh'
                                    ? '恢复'
                                    : 'Restore'
                                  : lang === 'zh'
                                  ? '排除'
                                  : 'Exclude'}
                              </button>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                )}

                {/* Clickable Fibre Measurements List (triggers animated centerline highlight on canvas) */}
                {result.objects.some((o) => o.kind === 'fibre') && (
                  <div className="pt-3 border-t border-slate-200 dark:border-slate-800 space-y-1.5">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-semibold uppercase tracking-wider text-slate-500">
                        {lang === 'zh'
                          ? `单条纤维测量结果 (${result.objects.filter((o) => o.kind === 'fibre' && !o.excluded).length})`
                          : `Fibre Measurements (${result.objects.filter((o) => o.kind === 'fibre' && !o.excluded).length})`}
                      </span>
                      {selectedObjectId !== null && (
                        <button
                          type="button"
                          onClick={() => onSelectObject(null)}
                          className="text-blue-600 dark:text-cyan-400 hover:underline cursor-pointer"
                        >
                          {lang === 'zh' ? '取消高亮' : 'Clear'}
                        </button>
                      )}
                    </div>
                    <div className="divide-y divide-slate-100 dark:divide-slate-800/80 border-y border-slate-200 dark:border-slate-800 max-h-56 overflow-y-auto">
                      {result.objects
                        .filter((o) => o.kind === 'fibre')
                        .map((fib) => {
                          const isSel = fib.id === selectedObjectId;
                          return (
                            <button
                              key={fib.id}
                              type="button"
                              onClick={() => onSelectObject(isSel ? null : fib.id)}
                              className={`w-full py-2 px-2 text-left transition-colors cursor-pointer flex items-center justify-between gap-2 ${
                                isSel
                                  ? 'bg-blue-50/90 dark:bg-cyan-500/15'
                                  : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                              } ${fib.excluded ? 'opacity-45' : ''}`}
                            >
                              <div className="min-w-0">
                                <div className="flex items-center gap-2 font-mono text-[12px] tabular-nums">
                                  <span
                                    className={`font-semibold ${
                                      isSel
                                        ? 'text-blue-600 dark:text-cyan-300'
                                        : 'text-slate-900 dark:text-slate-100'
                                    }`}
                                  >
                                    F#{fib.id}
                                  </span>
                                  <span className="text-slate-800 dark:text-slate-200">
                                    {fib.diameter !== undefined
                                      ? `${fib.diameter.toFixed(1)} ${specimen.calibration.spaceUnit}`
                                      : '—'}
                                  </span>
                                  <span className="text-amber-600 dark:text-amber-400">
                                    {fib.orientationDeg !== undefined
                                      ? `${fib.orientationDeg.toFixed(1)}°`
                                      : ''}
                                  </span>
                                </div>
                                <div className="mt-0.5 text-[10.5px] font-mono text-slate-500 dark:text-slate-400 truncate">
                                  {fib.reason ?? ''}
                                </div>
                              </div>
                              <span
                                className={`text-[10.5px] font-mono shrink-0 ${
                                  isSel
                                    ? 'text-blue-600 dark:text-cyan-300 font-semibold'
                                    : 'text-slate-400'
                                }`}
                              >
                                {isSel
                                  ? lang === 'zh'
                                    ? '● 高亮中'
                                    : '● Active'
                                  : lang === 'zh'
                                  ? '定位'
                                  : 'Locate'}
                              </span>
                            </button>
                          );
                        })}
                    </div>
                  </div>
                )}

                {result.objects.some((o) => o.subAreaRatioPct !== undefined) && (
                  <div className="pt-3 border-t border-slate-200 dark:border-slate-800 space-y-1.5">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-semibold uppercase tracking-wider text-slate-500">
                        {lang === 'zh'
                          ? `逐细胞核/细胞面积占比明细 (${result.objects.filter((o) => o.subAreaRatioPct !== undefined && !o.excluded).length})`
                          : `Parent → Sub-Region Area Ratios (${result.objects.filter((o) => o.subAreaRatioPct !== undefined && !o.excluded).length})`}
                      </span>
                      {selectedObjectId !== null && (
                        <button
                          type="button"
                          onClick={() => onSelectObject(null)}
                          className="text-blue-600 dark:text-cyan-400 hover:underline cursor-pointer"
                        >
                          {lang === 'zh' ? '取消高亮' : 'Clear'}
                        </button>
                      )}
                    </div>
                    <div className="divide-y divide-slate-100 dark:divide-slate-800/80 border-y border-slate-200 dark:border-slate-800 max-h-60 overflow-y-auto">
                      {result.objects
                        .filter((o) => o.subAreaRatioPct !== undefined)
                        .map((cell) => {
                          const isSel = cell.id === selectedObjectId;
                          const areaU = `${specimen.calibration.spaceUnit}²`;
                          return (
                            <button
                              key={cell.id}
                              type="button"
                              onClick={() => onSelectObject(isSel ? null : cell.id)}
                              className={`w-full py-2 px-2 text-left transition-colors cursor-pointer flex items-center justify-between gap-2 ${
                                isSel
                                  ? 'bg-cyan-50/90 dark:bg-cyan-500/15'
                                  : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                              } ${cell.excluded ? 'opacity-45' : ''}`}
                            >
                              <div className="min-w-0">
                                <div className="flex items-center gap-2 font-mono text-[12px] tabular-nums">
                                  <span
                                    className={`font-semibold ${
                                      isSel
                                        ? 'text-cyan-700 dark:text-cyan-300'
                                        : 'text-slate-900 dark:text-slate-100'
                                    }`}
                                  >
                                    {lang === 'zh' ? `细胞 #${cell.id}` : `Cell #${cell.id}`}
                                  </span>
                                  <span className="font-bold text-emerald-700 dark:text-emerald-400">
                                    {(cell.subAreaRatioPct ?? 0).toFixed(2)}%
                                  </span>
                                  <span className="text-[10.5px] text-amber-600 dark:text-amber-400">
                                    {lang === 'zh'
                                      ? `核×${cell.subRegionCount ?? 0}`
                                      : `sub×${cell.subRegionCount ?? 0}`}
                                  </span>
                                </div>
                                <div className="mt-0.5 text-[10.5px] font-mono text-slate-500 dark:text-slate-400 tabular-nums truncate">
                                  {lang === 'zh'
                                    ? `核 ${(cell.subArea ?? 0).toFixed(0)} / 细胞 ${(cell.area ?? 0).toFixed(0)} ${areaU}`
                                    : `Sub ${(cell.subArea ?? 0).toFixed(0)} / Parent ${(cell.area ?? 0).toFixed(0)} ${areaU}`}
                                </div>
                              </div>
                              <span
                                className={`text-[10.5px] font-mono shrink-0 ${
                                  isSel
                                    ? 'text-cyan-700 dark:text-cyan-300 font-semibold'
                                    : 'text-slate-400'
                                }`}
                              >
                                {isSel
                                  ? lang === 'zh'
                                    ? '● 高亮中'
                                    : '● Active'
                                  : lang === 'zh'
                                  ? '定位'
                                  : 'Locate'}
                              </span>
                            </button>
                          );
                        })}
                    </div>
                  </div>
                )}

                <div className="pt-3 border-t border-slate-200 dark:border-slate-800 space-y-2">
                  <div className="flex items-center justify-between gap-2 text-[11px]">
                    <span className="font-semibold uppercase tracking-wider text-slate-500">
                      {lang === 'zh'
                        ? `手动与智能点选测量 (${measurements.length})`
                        : `Manual & Click Measurements (${measurements.length})`}
                    </span>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <button
                        type="button"
                        onClick={() =>
                          onChangeToolMode(toolMode === 'measure_object' ? 'pan' : 'measure_object')
                        }
                        className={`h-6 px-2 rounded text-[10.5px] font-medium flex items-center gap-1 whitespace-nowrap cursor-pointer transition-colors ${
                          toolMode === 'measure_object'
                            ? 'bg-emerald-500 text-slate-950 font-semibold'
                            : 'border border-emerald-500/50 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/20'
                        }`}
                      >
                        <ScanSearch className="w-3 h-3 shrink-0" />
                        <span>
                          {toolMode === 'measure_object'
                            ? lang === 'zh'
                              ? '点选测量中'
                              : 'Measuring'
                            : lang === 'zh'
                            ? '点选测物体'
                            : 'Click Measure'}
                        </span>
                      </button>
                      {measurements.some((m) => m.kind === 'object' && m.objectMetrics) && (
                        <button
                          type="button"
                          onClick={() => {
                            const header = `Index,Length(${specimen.calibration.spaceUnit}),Width(${specimen.calibration.spaceUnit}),Area(${specimen.calibration.spaceUnit}2),EqDiameter(${specimen.calibration.spaceUnit}),Perimeter(${specimen.calibration.spaceUnit}),AspectRatio,Circularity,AngleDeg`;
                            const rows = measurements
                              .map((m, i) => {
                                if (m.kind !== 'object' || !m.objectMetrics) return null;
                                const o = m.objectMetrics;
                                return `${i + 1},${o.lengthPhys},${o.widthPhys},${o.areaPhys},${o.equivDiameterPhys},${o.perimeterPhys},${o.aspectRatio},${o.circularity},${o.orientationDeg}`;
                              })
                              .filter(Boolean);
                            copyText('obj_meas_csv', [header, ...rows].join('\n'));
                          }}
                          className="h-6 px-1.5 rounded border border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-cyan-500/60 text-[10px] flex items-center gap-1 whitespace-nowrap cursor-pointer"
                        >
                          {copiedKey === 'obj_meas_csv' ? (
                            <Check className="w-3 h-3 text-emerald-500" />
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                          <span>CSV</span>
                        </button>
                      )}
                      {measurements.length > 0 && (
                        <button
                          type="button"
                          onClick={onClearMeasurements}
                          className="text-slate-400 hover:text-rose-500 flex items-center gap-0.5 cursor-pointer"
                          title={lang === 'zh' ? '清空测量' : 'Clear measurements'}
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  </div>
                  {measurements.length === 0 ? (
                    <div className="text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
                      {lang === 'zh'
                        ? '点击「点选测物体」后直接单击任意目标，自动判定边界并生成长、宽、面积、等效径与圆度。'
                        : 'Click "Click Measure" then click any object on canvas to auto-segment its boundary and compute length, width, area, and circularity.'}
                    </div>
                  ) : (
                    <div className="divide-y divide-slate-100 dark:divide-slate-800/80 font-mono text-[11px] tabular-nums">
                      {measurements.map((m, idx) =>
                        m.kind === 'object' && m.objectMetrics ? (
                          <div key={m.id} className="py-2 space-y-1.5">
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                                #{idx + 1} · {lang === 'zh' ? '智能对象测量' : 'Smart Object'}
                              </span>
                              <button
                                type="button"
                                onClick={() => onDeleteMeasurement(m.id)}
                                className="text-slate-400 hover:text-rose-500 p-0.5 cursor-pointer shrink-0"
                              >
                                <X className="w-3.5 h-3.5" />
                              </button>
                            </div>
                            <div className="grid grid-cols-3 gap-x-2 gap-y-1 text-[10.5px]">
                              <div>
                                <span className="text-slate-400 block text-[9.5px] font-sans">
                                  {lang === 'zh' ? '长 (Major)' : 'Length'}
                                </span>
                                <span className="text-slate-900 dark:text-slate-100 font-semibold">
                                  {m.objectMetrics.lengthPhys.toFixed(2)} {m.unit}
                                </span>
                              </div>
                              <div>
                                <span className="text-slate-400 block text-[9.5px] font-sans">
                                  {lang === 'zh' ? '宽 (Minor)' : 'Width'}
                                </span>
                                <span className="text-slate-900 dark:text-slate-100 font-semibold">
                                  {m.objectMetrics.widthPhys.toFixed(2)} {m.unit}
                                </span>
                              </div>
                              <div>
                                <span className="text-slate-400 block text-[9.5px] font-sans">
                                  {lang === 'zh' ? '面积 (Area)' : 'Area'}
                                </span>
                                <span className="text-emerald-600 dark:text-emerald-400 font-semibold">
                                  {m.objectMetrics.areaPhys.toFixed(2)} {m.objectMetrics.areaUnit}
                                </span>
                              </div>
                              <div>
                                <span className="text-slate-400 block text-[9.5px] font-sans">
                                  {lang === 'zh' ? '等效直径' : 'Eq. Dia'}
                                </span>
                                <span className="text-slate-700 dark:text-slate-300">
                                  {m.objectMetrics.equivDiameterPhys.toFixed(2)} {m.unit}
                                </span>
                              </div>
                              <div>
                                <span className="text-slate-400 block text-[9.5px] font-sans">
                                  {lang === 'zh' ? '周长' : 'Perimeter'}
                                </span>
                                <span className="text-slate-700 dark:text-slate-300">
                                  {m.objectMetrics.perimeterPhys.toFixed(2)} {m.unit}
                                </span>
                              </div>
                              <div>
                                <span className="text-slate-400 block text-[9.5px] font-sans">
                                  {lang === 'zh' ? '长宽比 / 圆度' : 'AR / Circ'}
                                </span>
                                <span className="text-slate-700 dark:text-slate-300">
                                  {m.objectMetrics.aspectRatio.toFixed(2)} / {m.objectMetrics.circularity.toFixed(2)}
                                </span>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div key={m.id} className="py-1.5 flex items-center justify-between gap-2">
                            <span className="text-slate-700 dark:text-slate-300 truncate">
                              #{idx + 1} · {m.lengthPhys.toFixed(1)} {m.unit}
                              {m.kind === 'profile' && m.profile?.layerCount
                                ? lang === 'zh'
                                  ? ` · ${m.profile.layerCount}层 (均 ${m.profile.meanLayerThickness?.toFixed(1) ?? '—'} ${m.unit})`
                                  : ` · ${m.profile.layerCount}L (avg ${m.profile.meanLayerThickness?.toFixed(1) ?? '—'} ${m.unit})`
                                : ''}
                            </span>
                            <button
                              type="button"
                              onClick={() => onDeleteMeasurement(m.id)}
                              className="text-slate-400 hover:text-rose-500 p-0.5 cursor-pointer shrink-0"
                            >
                              <X className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        )
                      )}
                    </div>
                  )}
                </div>
                  </>
                )}
              </>
            )}

            {/* SUB-TAB 4B: HUMAN REVIEW & ANOMALY QUICK-FIXES */}
            {resultsSubTab === 'review' && (
              <div className="space-y-3">
                {anomalySuggestions.length > 0 && (
                  <div className="pb-3 border-b border-slate-200 dark:border-slate-800 space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-blue-600 dark:text-cyan-400">
                        <Wrench className="w-3.5 h-3.5" />
                        <span>
                          {lang === 'zh'
                            ? `异常一键修复 (${anomalySuggestions.length})`
                            : `Suggested Fixes (${anomalySuggestions.length})`}
                        </span>
                      </span>
                      {anomalySuggestions.length > 1 && (
                        <button
                          type="button"
                          onClick={() => applySuggestions(anomalySuggestions)}
                          className="text-[11px] font-medium text-blue-600 dark:text-cyan-400 hover:underline cursor-pointer"
                        >
                          {lang === 'zh' ? '全部应用' : 'Apply All'}
                        </button>
                      )}
                    </div>
                    <div className="divide-y divide-slate-200 dark:divide-slate-800 border-t border-slate-200 dark:border-slate-800">
                      {anomalySuggestions.map((s) => (
                        <div
                          key={s.id}
                          onClick={() => {
                            if (s.affectedObjectIds.length > 0) {
                              onSelectObject(s.affectedObjectIds[0]);
                            }
                          }}
                          className="py-2 flex items-start justify-between gap-2 cursor-pointer"
                        >
                          <div className="min-w-0 space-y-0.5">
                            <div className="text-[11px] font-mono text-amber-600 dark:text-amber-400">
                              {lang === 'zh' ? s.anomalyZh : s.anomalyEn}
                              {s.paramFix ? ` · ${lang === 'zh' ? s.paramFix.summaryZh : s.paramFix.summaryEn}` : ''}
                            </div>
                            <div className="text-[12px] font-medium text-slate-900 dark:text-slate-100 leading-snug">
                              {lang === 'zh' ? s.titleZh : s.titleEn}
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              applySuggestions([s]);
                            }}
                            className="h-7 px-2.5 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[11px] font-semibold shrink-0 cursor-pointer"
                          >
                            {lang === 'zh' ? '应用' : 'Fix'}
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between gap-2 text-[12px]">
                  <div className="flex items-center gap-2 min-w-0">
                    <button
                      type="button"
                      onClick={() => setReviewFilter('all')}
                      className={`whitespace-nowrap cursor-pointer ${
                        reviewFilter === 'all'
                          ? 'font-semibold text-blue-600 dark:text-cyan-400'
                          : 'text-slate-500'
                      }`}
                    >
                      {t('review_all', lang)} ({result.objects.length})
                    </button>
                    <span className="text-slate-300 dark:text-slate-700">|</span>
                    <button
                      type="button"
                      onClick={() => setReviewFilter('flagged')}
                      className={`whitespace-nowrap cursor-pointer ${
                        reviewFilter === 'flagged'
                          ? 'font-semibold text-amber-600 dark:text-amber-400'
                          : 'text-slate-500'
                      }`}
                    >
                      {t('review_flagged', lang)} ({flaggedCount})
                    </button>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={() =>
                        onChangeToolMode(toolMode === 'boundary_refine' ? 'pan' : 'boundary_refine')
                      }
                      title={t('tool_boundary_refine', lang)}
                      className={`h-6 px-2 rounded text-[10.5px] font-medium flex items-center gap-1 whitespace-nowrap shrink-0 cursor-pointer transition-colors ${
                        toolMode === 'boundary_refine'
                          ? 'bg-emerald-500 text-slate-950 font-semibold'
                          : 'border border-emerald-500/50 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/20'
                      }`}
                    >
                      <Sparkles className="w-3 h-3 shrink-0" />
                      <span>
                        {toolMode === 'boundary_refine'
                          ? lang === 'zh'
                            ? '精修中'
                            : 'Refining'
                          : lang === 'zh'
                          ? 'SAM 精修'
                          : 'Boundary Refine'}
                      </span>
                    </button>
                    <span className="font-mono text-[11px] text-slate-500 whitespace-nowrap">
                      {specCorrections.length} {lang === 'zh' ? '条纠错' : 'edits'}
                    </span>
                  </div>
                </div>

                <div className="divide-y divide-slate-200 dark:divide-slate-800 border-y border-slate-200 dark:border-slate-800 max-h-72 overflow-y-auto">
                  {filteredObjects.map((obj) => {
                    const selected = obj.id === selectedObjectId;
                    return (
                      <div
                        key={obj.id}
                        onClick={() => onSelectObject(selected ? null : obj.id)}
                        className={`py-2 px-2 transition-colors cursor-pointer ${
                          selected
                            ? 'bg-blue-50/90 dark:bg-cyan-500/15'
                            : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                        } ${obj.excluded ? 'opacity-50' : ''}`}
                      >
                        <div className="flex items-center justify-between gap-2 text-[12px]">
                          <div className="flex items-center gap-1.5 font-mono tabular-nums min-w-0">
                            <span className="font-semibold text-slate-900 dark:text-slate-100">
                              {obj.kind === 'fibre'
                                ? `F#${obj.id}`
                                : obj.kind === 'layer'
                                ? `L#${obj.id}`
                                : `#${obj.id}`}
                            </span>
                            <span className="text-slate-500 dark:text-slate-400">
                              {obj.thickness !== undefined
                                ? `T ${obj.thickness.toFixed(2)} ${specimen.calibration.spaceUnit}`
                                : obj.diameter !== undefined
                                ? `${obj.diameter.toFixed(1)} ${specimen.calibration.spaceUnit}`
                                : obj.width !== undefined
                                ? `CD ${obj.width.toFixed(1)} ${specimen.calibration.spaceUnit}`
                                : obj.kind}
                            </span>
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onAddCorrection({
                                specimenId: specimen.id,
                                objectId: obj.id,
                                field: 'excluded',
                                oldValue: Boolean(obj.excluded),
                                newValue: !obj.excluded,
                              });
                            }}
                            className="text-[11px] text-slate-500 hover:text-rose-500 shrink-0 cursor-pointer"
                          >
                            {obj.excluded ? t('review_restore', lang) : t('review_exclude', lang)}
                          </button>
                        </div>

                        <div
                          className="mt-1 flex items-center flex-wrap gap-2 text-[11px]"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {obj.kind === 'droplet' && obj.cells !== undefined && (
                            <div className="flex items-center gap-1 font-mono text-slate-600 dark:text-slate-300">
                              <span>{lang === 'zh' ? '细胞:' : 'Cells:'}</span>
                              <button
                                type="button"
                                onClick={() =>
                                  onAddCorrection({
                                    specimenId: specimen.id,
                                    objectId: obj.id,
                                    field: 'cells',
                                    oldValue: obj.cells,
                                    newValue: Math.max(0, (obj.cells ?? 0) - 1),
                                  })
                                }
                                className="w-5 h-5 rounded border border-slate-300 dark:border-slate-700 cursor-pointer"
                              >
                                −
                              </button>
                              <span className="w-4 text-center font-semibold">{obj.cells}</span>
                              <button
                                type="button"
                                onClick={() =>
                                  onAddCorrection({
                                    specimenId: specimen.id,
                                    objectId: obj.id,
                                    field: 'cells',
                                    oldValue: obj.cells,
                                    newValue: (obj.cells ?? 0) + 1,
                                  })
                                }
                                className="w-5 h-5 rounded border border-slate-300 dark:border-slate-700 cursor-pointer"
                              >
                                +
                              </button>
                            </div>
                          )}
                          {obj.kind === 'particle' && (
                            <select
                              value={obj.cls ?? 'normal'}
                              onChange={(e) =>
                                onAddCorrection({
                                  specimenId: specimen.id,
                                  objectId: obj.id,
                                  field: 'cls',
                                  oldValue: obj.cls,
                                  newValue: e.target.value,
                                })
                              }
                              className="h-6 px-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11px]"
                            >
                              <option value="normal">normal</option>
                              <option value="occluded">occluded</option>
                              <option value="agglomerate">agglomerate</option>
                              <option value="irregular">irregular</option>
                              <option value="outlier">outlier</option>
                            </select>
                          )}
                          <span className="ml-auto font-mono text-slate-400">
                            {(obj.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Jump directly to Side-by-Side Stage 3 Re-Train */}
                <button
                  type="button"
                  onClick={() => onChangeStage('model')}
                  className="w-full h-8 px-3 rounded border border-blue-600/40 dark:border-cyan-500/40 bg-blue-50/70 dark:bg-cyan-500/10 hover:bg-blue-100/70 dark:hover:bg-cyan-500/20 text-blue-700 dark:text-cyan-300 text-[12px] font-medium flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <Brain className="w-3.5 h-3.5" />
                  <span>
                    {lang === 'zh'
                      ? '用以上纠错在 Step 3 并排微调模型 →'
                      : 'Re-Train Model Side-by-Side (Step 3) →'}
                  </span>
                </button>

                {specCorrections.length > 0 && (
                  <div className="text-[11px] font-mono text-slate-500 dark:text-slate-400 space-y-1">
                    <div className="flex items-center justify-between text-[10.5px] uppercase tracking-wider text-slate-400">
                      <span>corrections.json ({specCorrections.length})</span>
                      <button
                        type="button"
                        onClick={onClearCorrections}
                        className="text-slate-400 hover:text-rose-500 flex items-center gap-1 normal-case cursor-pointer"
                      >
                        <Trash2 className="w-3 h-3" />
                        <span>{lang === 'zh' ? '清空' : 'Clear'}</span>
                      </button>
                    </div>
                    <div className="max-h-28 overflow-y-auto divide-y divide-slate-100 dark:divide-slate-800/80">
                      {specCorrections.map((c) => (
                        <div key={c.id} className="py-1 flex items-center justify-between gap-2">
                          <span className="truncate">
                            #{c.objectId}.{c.field}: {String(c.oldValue ?? '—')} → {String(c.newValue)}
                          </span>
                          <button
                            type="button"
                            onClick={() => onDeleteCorrection(c.id)}
                            className="text-slate-400 hover:text-rose-500 p-0.5 shrink-0 cursor-pointer"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {(prompts.length > 0 || strokes.length > 0) && (
                  <div className="pt-2 border-t border-slate-200 dark:border-slate-800 space-y-1 text-[11px] font-mono">
                    <div className="flex items-center justify-between text-[10.5px] uppercase text-slate-500">
                      <span>
                        {lang === 'zh'
                          ? `画布标记 (${prompts.length + strokes.length})`
                          : `Canvas Marks (${prompts.length + strokes.length})`}
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          onClearPrompts();
                          onClearStrokes();
                        }}
                        className="text-slate-400 hover:text-rose-500 normal-case cursor-pointer"
                      >
                        {lang === 'zh' ? '清除全部' : 'Clear All'}
                      </button>
                    </div>
                    {prompts.map((p, idx) => (
                      <div key={p.id} className="flex items-center justify-between">
                        <span>
                          SAM #{idx + 1} {p.positive ? '(+)' : '(-)'}
                        </span>
                        <button
                          type="button"
                          onClick={() => onDeletePrompt(p.id)}
                          className="text-slate-400 hover:text-rose-500 cursor-pointer"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                    {strokes.map((st, idx) => (
                      <div key={st.id} className="flex items-center justify-between">
                        <span>
                          {st.source === 'sam_click' || st.polygon ? 'SAM GT' : 'Brush'} #{idx + 1} · Class {st.classId}
                          {st.areaPx ? ` (${st.areaPx}px)` : ''}
                        </span>
                        <button
                          type="button"
                          onClick={() => onDeleteStroke(st.id)}
                          className="text-slate-400 hover:text-rose-500 cursor-pointer"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* SUB-TAB 4C: PROVENANCE & METHODS */}
            {resultsSubTab === 'provenance' && methods && (
              <div className="space-y-4 text-[12px]">
                <div className="space-y-1.5 font-mono text-[11px] tabular-nums">
                  <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800/80">
                    <span className="text-slate-500">Recipe</span>
                    <span className="text-slate-900 dark:text-slate-100">{result.recipe.id}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800/80">
                    <span className="text-slate-500">Recipe SHA-256</span>
                    <span className="text-slate-900 dark:text-slate-100">
                      {result.recipeDigest.slice(0, 16)}
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800/80">
                    <span className="text-slate-500">Input SHA-256</span>
                    <span className="text-slate-900 dark:text-slate-100">
                      {specimen.sha256.slice(0, 16)}
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800/80">
                    <span className="text-slate-500">Result Digest</span>
                    <span className="text-blue-600 dark:text-cyan-400">
                      {result.resultDigest.slice(0, 16)}
                    </span>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                      {lang === 'zh' ? '论文方法段落' : 'Methods Paragraph'}
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        copyText('methods', lang === 'zh' ? methods.zh : methods.en)
                      }
                      className="text-[11px] text-blue-600 dark:text-cyan-400 flex items-center gap-1 cursor-pointer"
                    >
                      {copiedKey === 'methods' ? (
                        <Check className="w-3 h-3" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                      <span>
                        {copiedKey === 'methods'
                          ? lang === 'zh'
                            ? '已复制'
                            : 'Copied'
                          : lang === 'zh'
                          ? '复制'
                          : 'Copy'}
                      </span>
                    </button>
                  </div>
                  <p className="p-2.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60 text-[11.5px] leading-relaxed text-slate-700 dark:text-slate-300">
                    {lang === 'zh' ? methods.zh : methods.en}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={onExportVexflow}
                    className="h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11.5px] flex items-center justify-center gap-1.5 text-slate-700 dark:text-slate-200 cursor-pointer"
                  >
                    <FileJson className="w-3.5 h-3.5" />
                    <span>.vexflow</span>
                  </button>
                  <button
                    type="button"
                    onClick={onExportReproducePy}
                    className="h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11.5px] flex items-center justify-center gap-1.5 text-slate-700 dark:text-slate-200 cursor-pointer"
                  >
                    <FileCode2 className="w-3.5 h-3.5" />
                    <span>reproduce.py</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
