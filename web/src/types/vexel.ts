export type Modality = 'droplet' | 'sem';
export type DetectedSceneType =
  | 'microfluidics_droplet'
  | 'cells'
  | 'pcb_board'
  | 'sem_metrology'
  | 'unknown';
export type Lang = 'zh' | 'en';
export type AppView = 'workbench' | 'workflow' | 'models' | 'studio3d' | 'batch';
export type ToolMode =
  | 'pan'
  | 'roi'
  | 'calibrate_line'
  | 'measure_line'
  | 'measure_profile'
  | 'measure_object'
  | 'smart_gt'
  | 'boundary_refine'
  | 'prompt_pos'
  | 'prompt_neg'
  | 'brush';

export interface RoiRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LowConfidenceSegment {
  id: string;
  objectId?: number;
  kind: 'layer' | 'object' | 'region';
  bbox: RoiRect;
  confidence: number; // 0..1
  localContrast: number; // local gray dynamic range (0..255)
  labelZh: string;
  labelEn: string;
  reasonZh: string;
  reasonEn: string;
}

export interface Calibration {
  pixelSize: number; // in spaceUnit per pixel
  spaceUnit: 'nm' | 'µm' | 'mm' | 'px';
  frameIntervalMs?: number;
  channelHeightUm?: number;
  source: 'metadata' | 'databar' | 'manual' | 'none';
  confirmed: boolean;
  vendor?: string;
  acceleratingVoltageKv?: number;
  workingDistanceMm?: number;
  magnification?: string;
  cropRows?: number; // imaging height excluding SEM data bar
}

export interface ImportRecognitionReport {
  recognized: boolean;
  sceneType: DetectedSceneType;
  sceneLabelZh: string;
  sceneLabelEn: string;
  confidence: number;
  source: 'local' | 'vision';
  reasonZh: string;
  reasonEn: string;
  hasDataBar: boolean;
  barPosition: 'bottom' | 'top' | 'none';
  croppedBarPx: number;
  originalWidth: number;
  originalHeight: number;
  scaleBarDetected: boolean;
  scaleBarPx: number | null;
  scaleBarValue: number | null;
  scaleBarUnit: Calibration['spaceUnit'] | null;
  scaleBarCropped: boolean;
  /** Auto-detected info/scale bar height (px) proposed for user confirmation prior to physical crop. */
  proposedCropBarPx?: number;
  /** Semi-auto crop confirmation gate state for single-image import. */
  cropStatus?: 'pending' | 'confirmed' | 'skipped';
  suggestedRecipeId: string | null;
  fileFormat?: string;
  bitDepth?: number;
  vendorMetadataFound?: boolean;
}

export interface Specimen {
  id: string;
  name: string;
  modality: Modality;
  subType:
    | 'droplet_encapsulation'
    | 'droplet_fluorescence'
    | 'cells_count'
    | 'pcb_traces'
    | 'sem_lines'
    | 'sem_holes'
    | 'sem_particles'
    | 'sem_pores'
    | 'sem_fibres'
    | 'sem_multilayer'
    | 'kidney_nested_cells'
    | 'unknown'
    | 'uploaded';
  width: number;
  height: number;
  frames: number;
  channels: number;
  rgba: Uint8ClampedArray;
  fluoRgba?: Uint8ClampedArray; // optional 2nd channel for fluorescence droplets
  gray: Float32Array;
  fluoGray?: Float32Array;
  rawRgba?: Uint8ClampedArray; // pre-crop original RGBA when info/scale bar was auto-cropped
  rawWidth?: number;
  rawHeight?: number;
  rawBarPosition?: 'bottom' | 'top' | 'none';
  rawCroppedBarPx?: number;
  calibration: Calibration;
  importReport?: ImportRecognitionReport;
  /** The uploaded file itself, kept in memory for the Python engine (full resolution and bit
   * depth, the instrument's metadata), with its size in pixels before the workbench's downscale. */
  sourceFile?: Blob;
  sourceWidth?: number;
  sourceHeight?: number;
  sha256: string;
  descriptionZh: string;
  descriptionEn: string;
}

export interface ParamDef {
  key: string;
  labelZh: string;
  labelEn: string;
  type: 'number' | 'boolean' | 'choice';
  defaultValue: number | boolean | string;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  choices?: Array<{ value: string; labelZh: string; labelEn: string }>;
  helpZh?: string;
  helpEn?: string;
}

export interface OperatorSpec {
  op: string;
  category: 'preprocess' | 'segment' | 'model' | 'filter' | 'metrology';
  modality: Modality | 'shared';
  labelZh: string;
  labelEn: string;
  descZh: string;
  descEn: string;
  inputs: string[];
  outputs: string[];
  params: ParamDef[];
}

export interface WorkflowStep {
  id: string;
  op: string;
  enabled: boolean;
  params: Record<string, number | boolean | string>;
  modelRef?: string;
}

export interface QCRule {
  id: string;
  metric: string;
  min?: number;
  max?: number;
  onFail: 'warn' | 'stop';
}

export interface UnifiedRecipe {
  schemaVersion: 1;
  id: string;
  name: string;
  nameZh: string;
  nameEn: string;
  modality: Modality;
  descriptionZh: string;
  descriptionEn: string;
  seed: number;
  steps: WorkflowStep[];
  qcRules: QCRule[];
  isCustom?: boolean;
  savedAt?: string;
}

export type ContourStyle = 'exact' | 'smooth';
export type OccludedDisplayMode = 'highlight' | 'hide' | 'normal';
export type ModelOcclusionMode = 'dashed' | 'hide';

export interface OverlayShape {
  id: string;
  kind: 'circle' | 'polygon' | 'polyline' | 'point' | 'box';
  points: Array<[number, number]>; // for circle: [[cx, cy], [rx, ry]]
  holes?: Array<Array<[number, number]>>; // optional internal hole contours for polygons
  color: string;
  fill?: string;
  dashed?: boolean;
  label?: string;
  objectId?: number;
  occluded?: boolean;
  layer: 'segmentation' | 'marks' | 'edges' | 'flags' | 'roi' | 'manual';
}

export interface DetectionObject {
  id: number;
  kind: 'droplet' | 'cell' | 'pad' | 'trace' | 'line' | 'hole' | 'particle' | 'pore' | 'fibre' | 'layer' | 'region';
  x: number;
  y: number;
  diameter?: number; // in physical unit
  width?: number;
  thickness?: number; // per-layer thickness in physical unit
  area?: number;
  parentId?: number;
  subArea?: number;
  subAreaRatioPct?: number;
  subRegionCount?: number;
  subDiameter?: number;
  circularity?: number;
  solidity?: number;
  aspectRatio?: number;
  cells?: number;
  fluoIntensity?: number;
  positive?: boolean;
  ler3Sigma?: number;
  lwr3Sigma?: number;
  orientationDeg?: number;
  cls?: 'normal' | 'agglomerate' | 'irregular' | 'outlier' | 'missing' | 'positive' | 'negative' | 'occluded';
  occluded?: boolean;
  occlusionScore?: number;
  overlapConfidence?: number;
  concaveCuspCount?: number;
  boundaryFitScore?: number;
  occludedReason?: string;
  reason?: string;
  flagged?: boolean;
  flagReason?: string;
  excluded?: boolean;
  confidence: number;
}

export interface SummaryMetric {
  key: string;
  labelZh: string;
  labelEn: string;
  value: number | string | null;
  unit: string;
  detailZh?: string;
  detailEn?: string;
  highlight?: boolean;
}

export interface StepPreview {
  stepId: string;
  op: string;
  labelZh: string;
  labelEn: string;
  durationMs: number;
  gray?: Float32Array;
  probMap?: Float32Array; // optional 0..1 model confidence overlay
  qcSummary: string;
}

export interface RunResult {
  recipe: UnifiedRecipe;
  recipeDigest: string;
  resultDigest: string;
  executedAt: string;
  durationMs: number;
  status: 'completed' | 'warning' | 'stopped_by_qc';
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  objects: DetectionObject[];
  shapes: OverlayShape[];
  stepPreviews: StepPreview[];
  warnings: string[];
  /** Active ROI rectangle in full working-image pixel coordinates when analysis was scoped to a manual ROI. */
  roi?: RoiRect | null;
  /** Final cumulative processed grayscale image produced by upstream image-transforming steps (composited to full frame). */
  processedGray?: Float32Array;
  /** Cumulative processed grayscale image right before model inference, used so local model training matches inference input. */
  preModelGray?: Float32Array;
  /** Low-contrast / low-confidence segments eligible for subtle red-box highlight & click-to-correct on the results layer. */
  lowConfidenceSegments?: LowConfidenceSegment[];
  histogram?: {
    labelZh: string;
    labelEn: string;
    unit: string;
    values: number[];
    mean: number;
  };
  poissonChart?: {
    counts: number[];
    observedPct: number[];
    expectedPct: number[];
    lambda: number;
  };
  profileChart?: {
    positions: number[];
    intensities: number[];
    edges: number[];
    unit: string;
  };
  provenance: {
    engine: string;
    engineVersion: string;
    inputName: string;
    inputSha256: string;
    recipeId: string;
    recipeDigest: string;
    resultDigest: string;
    calibrated: boolean;
    pixelSize: number;
    spaceUnit: string;
    modelRefs: string[];
    correctionsApplied: number;
    steps: Array<{ id: string; op: string; durationMs: number; qc: string }>;
  };
}

export interface RunVersionSnapshot {
  id: string;
  version: number;
  label: string;
  timestamp: string;
  result: RunResult;
}

export type LLMProviderId = 'gemini' | 'claude' | 'deepseek' | 'openai' | 'ollama';

export interface LLMProviderInfo {
  id: LLMProviderId;
  name: string;
  envVar: string;
  configured: boolean;
  supportsVision: boolean;
  defaultModel: string;
  models: string[];
  noteZh: string;
  noteEn: string;
}

export interface LLMProviderConfig {
  provider: LLMProviderId;
  model: string;
  ollamaBaseUrl?: string;
}

export interface PlanProposal {
  status: 'ready' | 'check' | 'needs_user';
  planner: 'rules' | 'gemini' | 'claude' | 'deepseek' | 'openai' | 'ollama' | 'python';
  modelName: string;
  modality: Modality;
  recipeId: string;
  recipeNameZh: string;
  recipeNameEn: string;
  confidence: number;
  checks: Array<{ mark: '✓' | '!' | '✗'; textZh: string; textEn: string }>;
  reasonsZh: string[];
  reasonsEn: string[];
  suggestedRecipe: UnifiedRecipe;
  previewResult?: RunResult;
  fallbackNotice?: string;
}

export interface ModelCard {
  id: string;
  version: string;
  ref: string; // id@version
  nameZh: string;
  nameEn: string;
  task: 'segment.instances' | 'segment.classes' | 'count.inclusions' | 'classify.objects';
  framework: 'onnx' | 'sklearn' | 'pytorch';
  modality: Modality | 'shared';
  license: string;
  commercialUse: boolean;
  trainingData: Array<{ name: string; license: string }>;
  sha256: string;
  tiledInference: boolean;
  tileSize?: number;
  fileStatus: 'verified' | 'local-trained' | 'missing';
  metrics: {
    meanIoU?: number;
    macroF1?: number;
    accuracy?: number;
  };
  descriptionZh: string;
  descriptionEn: string;
  isLocalTrained?: boolean;
  classes?: string[];
}

export interface AnnotationStroke {
  id: string;
  classId: number; // 1 = foreground/object, 2 = background, 3 = defect/class3
  radius: number;
  points: Array<[number, number]>;
  /** Optional SAM-segmented filled polygon mask for one-click Ground Truth annotation. */
  polygon?: Array<[number, number]>;
  holes?: Array<Array<[number, number]>>;
  areaPx?: number;
  source?: 'brush' | 'sam_click';
}

export interface PromptPoint {
  id: string;
  x: number;
  y: number;
  positive: boolean;
}

export interface HumanCorrection {
  id: string;
  specimenId: string;
  objectId: number;
  field: 'cells' | 'cls' | 'excluded' | 'positive';
  oldValue: string | number | boolean | undefined;
  newValue: string | number | boolean;
  timestamp: string;
}

export interface TrainingEpochPoint {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  valMeanIoU: number;
  valMacroF1: number;
}

export interface HoldoutMetrics {
  nTrainPixels: number;
  nValPixels: number;
  accuracy: number;
  macroF1: number;
  meanIoU: number;
  perClassIoU: Record<string, number>;
  splitStrategy?: 'spatial_block' | 'stratified';
  epochs?: TrainingEpochPoint[];
  durationMs?: number;
  cacheHit?: boolean;
  architecture?: string;
}

export interface ObjectMeasurementMetrics {
  polygon: Array<[number, number]>;
  holes?: Array<Array<[number, number]>>;
  cx: number;
  cy: number;
  bbox: RoiRect;
  majorAxis: [[number, number], [number, number]];
  minorAxis: [[number, number], [number, number]];
  areaPx: number;
  areaPhys: number;
  areaUnit: string;
  lengthPhys: number;
  widthPhys: number;
  equivDiameterPhys: number;
  perimeterPhys: number;
  aspectRatio: number;
  circularity: number;
  orientationDeg: number;
  meanIntensity: number;
  confidence: number;
}

export interface RefinedBoundaryProposal {
  objectId: number;
  shapeId: string;
  originalKind: OverlayShape['kind'];
  originalPoints: Array<[number, number]>;
  originalHoles?: Array<Array<[number, number]>>;
  refinedPoints: Array<[number, number]>;
  refinedHoles: Array<Array<[number, number]>>;
  metrics: ObjectMeasurementMetrics;
  originalAreaPx: number;
  areaDeltaPct: number;
  edgeAlignmentBefore: number;
  edgeAlignmentAfter: number;
  edgeGainPct: number;
}

export interface ManualMeasurement {
  id: string;
  kind: 'line' | 'profile' | 'object';
  p1: [number, number];
  p2: [number, number];
  lengthPhys: number;
  unit: string;
  profile?: {
    positions: number[];
    intensities: number[];
    edges: number[];
    widthPhys: number | null;
    layerCount?: number;
    layerThicknesses?: number[];
    meanLayerThickness?: number | null;
    totalStackThickness?: number | null;
    fftPeriodPhys?: number | null;
  };
  objectMetrics?: ObjectMeasurementMetrics;
}

export interface BatchRow {
  id: string;
  specimen: Specimen;
  group: string;
  replicate: string;
  pixelSizeOverride?: number;
  status: 'pending' | 'running' | 'completed' | 'warning' | 'failed';
  durationMs?: number;
  warnings: string[];
  error?: string;
  result?: RunResult;
}

export interface GroupStatRow {
  metricKey: string;
  metricLabelZh: string;
  metricLabelEn: string;
  unit: string;
  group: string;
  nReplicates: number;
  mean: number;
  sd: number;
  ci95Low: number;
  ci95High: number;
  median: number;
}

export interface ComparisonRow {
  metricKey: string;
  metricLabelZh: string;
  metricLabelEn: string;
  groupA: string;
  groupB: string;
  testName: string;
  diff: number;
  pValue: number;
  hedgesG: number;
  significant: boolean;
}

export type ResourceBudgetTier =
  | 'optimal'
  | 'tiled_safe'
  | 'lod_guarded'
  | 'offload_gpu_required';

export interface ResourceBudgetReport {
  task: '2d_train' | '2d_infer' | '3d_render' | '3d_analyze';
  dimensions: { width: number; height: number; depth: number };
  elementCount: number;
  rawMemoryMB: number;
  estimatedPeakMB: number;
  tier: ResourceBudgetTier;
  recommendedTileSize: number;
  recommendedOverlapPx: number;
  recommendedDownsampleFactor: number;
  recommendationZh: string;
  recommendationEn: string;
}

export interface DeepFewShotConfig {
  hiddenDim: number;
  epochs: number;
  learningRate: number;
  hardNegativeMining: boolean;
  bilateralSmooth: boolean;
}

export type GpuBackboneId = 'dinov2_vit_s14' | 'convnext_tiny' | 'sam_vit_b_lora' | 'unet_resnet34';
export type GpuLossId = 'focal_dice' | 'ce_dice' | 'tversky';
export type GpuPrecisionId = 'fp16' | 'bf16';

export interface GpuProviderPreset {
  id: string;
  nameZh: string;
  nameEn: string;
  gpuTier: string;
  billingZh: string;
  billingEn: string;
  coldStartSec: number;
  maxTilePx: number;
  recommendedForZh: string;
  recommendedForEn: string;
}

export interface OnlineTrainJobRecord {
  jobId: string;
  createdAt: string;
  provider: string;
  backbone: string;
  loraRank: number;
  epochs: number;
  loss: string;
  precision: string;
  maxWallClockSec: number;
  modelId: string;
  version: string;
  specimenName: string;
  imageSha256: string;
  imageWidth: number;
  imageHeight: number;
  labeledClasses: number;
  labeledPixels: number;
  status: 'manifest_ready' | 'dispatched' | 'rejected';
  sandboxPolicy: string[];
  manifestDigest: string;
  message: string;
}

export interface VolumeStackDataset {
  id: string;
  name: string;
  width: number;
  height: number;
  depth: number;
  /** Compact 1-byte-per-voxel normalized intensity buffer (length = width * height * depth) */
  data: Uint8Array;
  /** Optional downsampled display LOD buffer when source exceeds browser 3D texture budget */
  displayLod?: {
    width: number;
    height: number;
    depth: number;
    data: Uint8Array;
    factor: number;
  };
  voxelSizeXY: number;
  voxelSizeZ: number;
  spaceUnit: 'nm' | 'µm' | 'mm' | 'px';
  source: 'synthetic_kidney_3d' | 'uploaded_zstack';
  originalDimensions?: { width: number; height: number; depth: number };
  lodApplied?: boolean;
  descriptionZh: string;
  descriptionEn: string;
}

export interface VolumetricObjectMetric {
  id: number;
  kind: 'cell_3d' | 'nucleus_3d' | 'particle_3d';
  parentId?: number;
  voxelCount: number;
  volumePhys: number;
  surfaceAreaPhys: number;
  sphericity: number;
  equivDiameterPhys: number;
  centroid: [number, number, number]; // in voxel coordinates (x, y, z)
  bboxVoxel: {
    x0: number;
    y0: number;
    z0: number;
    x1: number;
    y1: number;
    z1: number;
  };
  axesPhys: [number, number, number]; // PCA principal axis lengths [L1 >= L2 >= L3] in spaceUnit
  elongation?: number; // L1 / L2 (>= 1.0)
  flatness?: number; // L2 / L3 (>= 1.0)
  meanIntensity: number;
  // Hierarchical parent-cell 3D nuclear-to-cytoplasmic metrics
  subRegionCount?: number;
  subVolumePhys?: number;
  cytoplasmVolumePhys?: number;
  ncVolumeRatio?: number; // V_nucleus / V_cytoplasm
  ncVolumePct?: number; // (V_nucleus / V_cytoplasm) * 100
  ncellVolumePct?: number; // (V_nucleus / V_cell) * 100
}

export interface VolumeAnalysisResult {
  volumeId: string;
  executedAt: string;
  durationMs: number;
  cellCount: number;
  nucleusCount: number;
  meanCellVolume: number;
  meanSphericity: number;
  meanNcVolumePct: number;
  meanNcellVolumePct: number;
  spaceUnit: VolumeStackDataset['spaceUnit'];
  objects: VolumetricObjectMetric[];
  /** Compact 3D segmentation label volume (0=bg, 1=cell cytoplasm, 2=nucleus) */
  labelVolume: Uint8Array;
}

