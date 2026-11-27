import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Archive,
  Brush,
  CheckCircle2,
  Cloud,
  Cpu,
  Crop,
  Crosshair,
  Download,
  FileCode2,
  FileJson,
  Layers,
  Lock,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  Upload,
  Wand2,
} from 'lucide-react';
import type {
  AnnotationStroke,
  GpuBackboneId,
  GpuLossId,
  GpuPrecisionId,
  GpuProviderPreset,
  HumanCorrection,
  Lang,
  ModelCard,
  OnlineTrainJobRecord,
  RoiRect,
  RunResult,
  Specimen,
  TrainingEpochPoint,
} from '../types/vexel';
import { computeSegmentedResultsBoundingBox } from '../engine/operators';
import {
  buildTrainingMask,
  summarizeAnnotationMask,
  trainDeepFewShotModelAsync,
  type DeepTrainProgressEvent,
  type TrainedLocalModel,
} from '../engine/mlRuntime';
import {
  exportLocalModelJson,
  parseImportedLocalModelJson,
  type PersistedOnnxModel,
} from '../engine/modelStore';
import { buildPersistedOnnxModel } from '../engine/onnxRuntime';
import { evaluate2DBudget } from '../engine/resourceBudget';
import { GovernanceQuotaPanel } from './GovernanceQuotaPanel';
import { t } from '../i18n/strings';

interface ModelHubProps {
  lang: Lang;
  specimen: Specimen;
  result: RunResult | null;
  roi: RoiRect | null;
  onChangeRoi: (roi: RoiRect | null) => void;
  models: ModelCard[];
  localModels?: Map<string, TrainedLocalModel>;
  onnxModels?: Map<string, PersistedOnnxModel>;
  activeModelRef?: string;
  strokes: AnnotationStroke[];
  corrections: HumanCorrection[];
  onBindModel: (modelRef: string) => void;
  onPublishLocalModel: (trained: TrainedLocalModel) => void;
  onImportOnnxModel?: (card: ModelCard) => void;
  onImportOnnxBinary?: (record: PersistedOnnxModel) => void;
  onImportLocalModelJson?: (trained: TrainedLocalModel) => void;
  onDeletePersistedModel?: (ref: string) => void;
  onGoSmartGtOnCanvas?: () => void;
  onGoBrushOnCanvas: () => void;
  onGoRoiOnCanvas: () => void;
}

type TrainTabMode = 'browser_dl' | 'gpu_lora';

const FALLBACK_GPU_PRESETS: GpuProviderPreset[] = [
  {
    id: 'local_cuda',
    nameZh: '本地工作站 (NVIDIA CUDA)',
    nameEn: 'Local Workstation (NVIDIA CUDA)',
    gpuTier: 'RTX 4070 Ti Super (16 GB) / RTX 4090 (24 GB)',
    billingZh: '一次性硬件投入，数据零出网（科研合规首选）',
    billingEn: 'One-time hardware, air-gapped lab data privacy',
    coldStartSec: 0,
    maxTilePx: 1024,
    recommendedForZh: '高频微调、高隐私病理/晶圆数据、3D Z-Stack',
    recommendedForEn: 'Daily fine-tuning, sensitive microscopy & 3D stacks',
  },
  {
    id: 'modal_serverless',
    nameZh: 'Modal Serverless GPU',
    nameEn: 'Modal Serverless GPU',
    gpuTier: 'NVIDIA L4 (24 GB) / A10G (24 GB)',
    billingZh: '按秒计费（空闲零费用），无常驻运维成本',
    billingEn: 'Per-second billing ($0 idle), zero cluster ops',
    coldStartSec: 8,
    maxTilePx: 1024,
    recommendedForZh: '中小团队弹性少样本 LoRA 微调（推荐）',
    recommendedForEn: 'Elastic few-shot LoRA jobs (Recommended)',
  },
  {
    id: 'runpod_serverless',
    nameZh: 'RunPod Serverless / Pod',
    nameEn: 'RunPod Serverless / Pod',
    gpuTier: 'RTX 4090 (24 GB) / L40S (48 GB)',
    billingZh: '高性价比按秒/按小时计费',
    billingEn: 'Cost-effective per-second/hourly containers',
    coldStartSec: 12,
    maxTilePx: 1024,
    recommendedForZh: '大批量 SEM / 3D 体积微调',
    recommendedForEn: 'Cost-sensitive SEM & 3D training',
  },
  {
    id: 'autodl_cn',
    nameZh: 'AutoDL / 算力云 (国内低延迟)',
    nameEn: 'AutoDL (Mainland China)',
    gpuTier: 'RTX 4090 (24 GB) / A800 (80 GB)',
    billingZh: '人民币按量计费，国内高速传输',
    billingEn: 'CNY pay-as-you-go with domestic mirrors',
    coldStartSec: 15,
    maxTilePx: 1024,
    recommendedForZh: '中国大陆高校与企业实验室',
    recommendedForEn: 'Mainland China research & industrial labs',
  },
  {
    id: 'gcp_vertex',
    nameZh: 'Google Cloud Vertex AI',
    nameEn: 'Google Cloud Vertex AI',
    gpuTier: 'NVIDIA L4 (24 GB) / A100-80GB',
    billingZh: '企业级 IAM / VPC-SC 隔离与审计',
    billingEn: 'Enterprise IAM & VPC-SC isolation',
    coldStartSec: 18,
    maxTilePx: 1024,
    recommendedForZh: '需企业合规认证与 GCP 集成',
    recommendedForEn: 'Enterprise compliance & GCP stacks',
  },
  {
    id: 'aws_sagemaker',
    nameZh: 'AWS SageMaker Training',
    nameEn: 'AWS SageMaker Training',
    gpuTier: 'ml.g5.xlarge (A10G 24 GB)',
    billingZh: '按作业秒级计费，S3 + KMS 加密',
    billingEn: 'Per-job billing with S3 + KMS encryption',
    coldStartSec: 25,
    maxTilePx: 1024,
    recommendedForZh: '已有 AWS 数据湖的企业部署',
    recommendedForEn: 'Enterprise AWS pipelines',
  },
];

function triggerTextDownload(filename: string, content: string, mime = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Compact inline SVG chart rendering real training loss & spatial-block validation mIoU curves. */
function TrainingCurveChart({
  history,
  lang,
}: {
  history: TrainingEpochPoint[];
  lang: Lang;
}) {
  if (history.length < 2) return null;
  const w = 320;
  const h = 92;
  const padL = 26;
  const padR = 26;
  const padT = 10;
  const padB = 18;
  const iw = w - padL - padR;
  const ih = h - padT - padB;

  const maxEpoch = Math.max(1, history[history.length - 1].epoch);
  const maxLoss = Math.max(0.2, ...history.map((p) => p.trainLoss));

  const xOf = (ep: number) => padL + (ep / maxEpoch) * iw;
  const yLoss = (v: number) => padT + ih - Math.min(1, v / maxLoss) * ih;
  const yIoU = (v: number) => padT + ih - Math.max(0, Math.min(1, v)) * ih;

  const lossPath = history
    .map(
      (p, i) =>
        `${i === 0 ? 'M' : 'L'}${xOf(p.epoch).toFixed(1)},${yLoss(p.trainLoss).toFixed(1)}`
    )
    .join(' ');
  const iouPath = history
    .map(
      (p, i) =>
        `${i === 0 ? 'M' : 'L'}${xOf(p.epoch).toFixed(1)},${yIoU(p.valMeanIoU).toFixed(1)}`
    )
    .join(' ');

  const latest = history[history.length - 1];

  return (
    <div className="rounded border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] p-2 space-y-1">
      <div className="flex items-center justify-between text-[10px] font-mono">
        <span className="text-slate-500">
          {lang === 'zh' ? '收敛曲线 (Loss / Spatial-Block Val mIoU)' : 'Convergence (Loss / Val mIoU)'}
        </span>
        <div className="flex items-center gap-2.5">
          <span className="text-amber-500">Loss {latest.trainLoss.toFixed(3)}</span>
          <span className="text-cyan-500">Val mIoU {(latest.valMeanIoU * 100).toFixed(1)}%</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto block select-none">
        {[0, 0.5, 1].map((f) => {
          const y = padT + ih * (1 - f);
          return (
            <line
              key={f}
              x1={padL}
              x2={w - padR}
              y1={y}
              y2={y}
              stroke="currentColor"
              className="text-slate-200 dark:text-slate-800"
              strokeWidth={0.75}
              strokeDasharray={f === 0 ? undefined : '2,3'}
            />
          );
        })}
        <path d={lossPath} fill="none" stroke="#F59E0B" strokeWidth={1.6} />
        <path d={iouPath} fill="none" stroke="#06B6D4" strokeWidth={1.8} />
        <text
          x={padL - 3}
          y={padT + 4}
          textAnchor="end"
          className="fill-amber-500 font-mono text-[8px]"
        >
          {maxLoss.toFixed(1)}
        </text>
        <text
          x={w - padR + 3}
          y={padT + 4}
          textAnchor="start"
          className="fill-cyan-500 font-mono text-[8px]"
        >
          100%
        </text>
        <text
          x={w - padR}
          y={h - 4}
          textAnchor="end"
          className="fill-slate-400 font-mono text-[8px]"
        >
          ep {maxEpoch}
        </text>
      </svg>
    </div>
  );
}

export function ModelHub({
  lang,
  specimen,
  result,
  roi,
  onChangeRoi,
  models,
  localModels,
  onnxModels,
  activeModelRef,
  strokes,
  corrections,
  onBindModel,
  onPublishLocalModel,
  onImportOnnxModel,
  onImportOnnxBinary,
  onImportLocalModelJson,
  onDeletePersistedModel,
  onGoSmartGtOnCanvas,
  onGoBrushOnCanvas,
  onGoRoiOnCanvas,
}: ModelHubProps) {
  const [tabMode, setTabMode] = useState<TrainTabMode>('browser_dl');
  const [modelId, setModelId] = useState(`local-${specimen.modality}-dl`);
  const [version, setVersion] = useState('2.0');
  const [classNames, setClassNames] = useState(
    lang === 'zh' ? '目标前景, 背景基体, 缺陷异物' : 'Foreground, Background, Defect'
  );

  // Browser Web Worker Deep Few-Shot hyperparameters
  const [hiddenDim, setHiddenDim] = useState<24 | 32 | 48>(32);
  const [dlEpochs, setDlEpochs] = useState<20 | 32 | 50>(32);
  const [hardNegativeMining, setHardNegativeMining] = useState(true);
  const [bilateralSmooth, setBilateralSmooth] = useState(true);

  const [isTraining, setIsTraining] = useState(false);
  const [progressEv, setProgressEv] = useState<DeepTrainProgressEvent | null>(null);
  const [curveHistory, setCurveHistory] = useState<TrainingEpochPoint[]>([]);
  const [trainedCandidate, setTrainedCandidate] = useState<TrainedLocalModel | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [publishedRef, setPublishedRef] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // GPU LoRA Bridge declarative controls
  const [gpuPresets, setGpuPresets] = useState<GpuProviderPreset[]>(FALLBACK_GPU_PRESETS);
  const [gpuEndpointConfigured, setGpuEndpointConfigured] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>('modal_serverless');
  const [backbone, setBackbone] = useState<GpuBackboneId>('dinov2_vit_s14');
  const [loraRank, setLoraRank] = useState<4 | 8 | 16>(8);
  const [gpuEpochs, setGpuEpochs] = useState<number>(25);
  const [gpuLoss, setGpuLoss] = useState<GpuLossId>('focal_dice');
  const [gpuPrecision, setGpuPrecision] = useState<GpuPrecisionId>('bf16');
  const [gpuSubmitting, setGpuSubmitting] = useState(false);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const [latestGpuJob, setLatestGpuJob] = useState<{
    job: OnlineTrainJobRecord;
    manifestJson: string;
    pythonScript: string;
    sanitization?: {
      reencoded: boolean;
      width: number;
      height: number;
      imageSanitizedSha256: string;
      maskSanitizedSha256?: string;
      strippedChunks: string[];
      trailingBytesStripped: number;
      datasetBundleSha256?: string;
    } | null;
    datasetBundleBase64?: string;
  } | null>(null);
  const [onnxImportMsg, setOnnxImportMsg] = useState<string | null>(null);
  const onnxInputRef = useRef<HTMLInputElement>(null);
  // In-memory Bearer session token from GovernanceQuotaPanel (never stored in localStorage)
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [governanceRefresh, setGovernanceRefresh] = useState(0);

  useEffect(() => {
    let active = true;
    fetch('/api/v1/train/presets')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!active || !data) return;
        if (Array.isArray(data.presets) && data.presets.length > 0) {
          setGpuPresets(data.presets);
        }
        setGpuEndpointConfigured(Boolean(data.endpointConfigured));
      })
      .catch(() => {
        // Offline fallback presets already populated
      });
    return () => {
      active = false;
    };
  }, []);

  const segmentedResultsRoi = useMemo(
    () => computeSegmentedResultsBoundingBox(result, specimen, 0.1),
    [result, specimen]
  );

  const specCorrections = useMemo(
    () => corrections.filter((c) => c.specimenId === specimen.id),
    [corrections, specimen.id]
  );

  const classCounts = useMemo(
    () => ({
      1: strokes.filter((s) => s.classId === 1).length,
      2: strokes.filter((s) => s.classId === 2).length,
      3: strokes.filter((s) => s.classId === 3).length,
    }),
    [strokes]
  );

  // Live 2D Resource Budget Guardrail
  const activeWidth = roi ? roi.width : specimen.width;
  const activeHeight = roi ? roi.height : specimen.calibration.cropRows ?? specimen.height;
  const budget2d = useMemo(
    () => evaluate2DBudget(activeWidth, activeHeight, '2d_train'),
    [activeWidth, activeHeight]
  );

  // Pre-flight annotation pixel gate summary (shared by browser DL & GPU LoRA)
  const annotationSummary = useMemo(
    () =>
      summarizeAnnotationMask(
        specimen,
        strokes,
        specCorrections,
        result?.objects ?? [],
        roi
      ),
    [specimen, strokes, specCorrections, result?.objects, roi]
  );

  const preflightPassed =
    annotationSummary.distinctClasses >= 2 && annotationSummary.totalLabeledPixels >= 64;

  const activePreset = useMemo(
    () => gpuPresets.find((p) => p.id === selectedProvider) ?? gpuPresets[0],
    [gpuPresets, selectedProvider]
  );

  const handleTrainBrowserDl = async () => {
    setTrainError(null);
    setPublishedRef(null);
    setCurveHistory([]);
    setProgressEv(null);
    const controller = new AbortController();
    abortRef.current = controller;
    setIsTraining(true);

    try {
      const classes = classNames
        .split(/[,，]/)
        .map((s) => s.trim())
        .filter(Boolean);
      const trained = await trainDeepFewShotModelAsync(
        modelId.trim() || 'local-dl-detector',
        version.trim() || '2.0',
        specimen,
        strokes,
        specCorrections,
        result?.objects ?? [],
        classes.length >= 2 ? classes : ['Foreground', 'Background'],
        {
          workingGray: result?.preModelGray ?? result?.processedGray,
          roi,
          config: {
            hiddenDim,
            epochs: dlEpochs,
            learningRate: 0.025,
            hardNegativeMining,
            bilateralSmooth,
          },
          signal: controller.signal,
          onProgress: (ev) => {
            setProgressEv(ev);
            if (ev.phase === 'training') {
              setCurveHistory((prev) => [
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
      setTrainedCandidate(trained);
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsTraining(false);
      abortRef.current = null;
    }
  };

  const handleCancelBrowserDl = () => {
    abortRef.current?.abort();
  };

  const handlePublish = () => {
    if (!trainedCandidate) return;
    onPublishLocalModel(trainedCandidate);
    setPublishedRef(trainedCandidate.card.ref);
  };

  const triggerBase64BinaryDownload = (
    filename: string,
    base64: string,
    mimeType: string
  ) => {
    const raw = atob(base64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      bytes[i] = raw.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const encodeSpecimenAndMaskPngs = (): {
    imagePngBase64: string;
    maskPngBase64: string;
    width: number;
    height: number;
  } | null => {
    if (typeof document === 'undefined') return null;
    const fullW = specimen.width;
    const maxY = specimen.calibration.cropRows ?? specimen.height;
    const rx = roi ? Math.max(0, Math.min(fullW - 1, Math.round(roi.x))) : 0;
    const ry = roi ? Math.max(0, Math.min(maxY - 1, Math.round(roi.y))) : 0;
    const rw = roi ? Math.max(1, Math.min(fullW - rx, Math.round(roi.width))) : fullW;
    const rh = roi ? Math.max(1, Math.min(maxY - ry, Math.round(roi.height))) : specimen.height;

    const mask = buildTrainingMask(
      specimen,
      strokes,
      specCorrections,
      result?.objects ?? [],
      roi
    );

    const maxEdge = 1024;
    const scale = Math.min(1, maxEdge / Math.max(rw, rh));
    const outW = Math.max(1, Math.round(rw * scale));
    const outH = Math.max(1, Math.round(rh * scale));

    const imgCanvas = document.createElement('canvas');
    imgCanvas.width = outW;
    imgCanvas.height = outH;
    const imgCtx = imgCanvas.getContext('2d');
    if (!imgCtx) return null;
    const imgData = imgCtx.createImageData(outW, outH);
    const srcRgba = specimen.rgba;

    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = outW;
    maskCanvas.height = outH;
    const maskCtx = maskCanvas.getContext('2d');
    if (!maskCtx) return null;
    const maskData = maskCtx.createImageData(outW, outH);

    for (let y = 0; y < outH; y++) {
      const sy = Math.min(rh - 1, Math.floor((y / outH) * rh));
      const fullY = ry + sy;
      for (let x = 0; x < outW; x++) {
        const sx = Math.min(rw - 1, Math.floor((x / outW) * rw));
        const fullX = rx + sx;
        const dstIdx = (y * outW + x) * 4;
        const srcIdx = (fullY * fullW + fullX) * 4;
        imgData.data[dstIdx] = srcRgba[srcIdx] ?? 0;
        imgData.data[dstIdx + 1] = srcRgba[srcIdx + 1] ?? 0;
        imgData.data[dstIdx + 2] = srcRgba[srcIdx + 2] ?? 0;
        imgData.data[dstIdx + 3] = 255;

        const cls = mask[sy * rw + sx] ?? 0;
        maskData.data[dstIdx] = cls;
        maskData.data[dstIdx + 1] = cls;
        maskData.data[dstIdx + 2] = cls;
        maskData.data[dstIdx + 3] = 255;
      }
    }

    imgCtx.putImageData(imgData, 0, 0);
    maskCtx.putImageData(maskData, 0, 0);

    return {
      imagePngBase64: imgCanvas.toDataURL('image/png'),
      maskPngBase64: maskCanvas.toDataURL('image/png'),
      width: outW,
      height: outH,
    };
  };

  const handleSubmitGpuJob = async () => {
    setGpuError(null);
    setGpuSubmitting(true);
    try {
      const classes = classNames
        .split(/[,，]/)
        .map((s) => s.trim())
        .filter(Boolean);
      const encoded = encodeSpecimenAndMaskPngs();
      const reqHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
      if (authToken) reqHeaders.Authorization = `Bearer ${authToken}`;
      const resp = await fetch('/api/v1/train/jobs', {
        method: 'POST',
        headers: reqHeaders,
        body: JSON.stringify({
          provider: selectedProvider,
          backbone,
          loraRank,
          epochs: gpuEpochs,
          loss: gpuLoss,
          precision: gpuPrecision,
          maxWallClockSec: 360,
          modelId: modelId.trim() || 'vexel_lora_seg',
          version: version.trim() || '2.0',
          specimenName: specimen.name,
          imageSha256: specimen.sha256,
          imageWidth: encoded ? encoded.width : activeWidth,
          imageHeight: encoded ? encoded.height : activeHeight,
          labeledClasses: annotationSummary.distinctClasses,
          labeledPixels: annotationSummary.totalLabeledPixels,
          classNames: classes.length >= 2 ? classes : ['Foreground', 'Background'],
          ...(encoded
            ? {
                imagePngBase64: encoded.imagePngBase64,
                maskPngBase64: encoded.maskPngBase64,
              }
            : {}),
        }),
      });
      const data = await resp.json();
      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || `HTTP ${resp.status}`);
      }
      setLatestGpuJob({
        job: data.job as OnlineTrainJobRecord,
        manifestJson: JSON.stringify(data.manifest, null, 2),
        pythonScript: String(data.pythonScript ?? ''),
        sanitization: data.sanitization ?? null,
        datasetBundleBase64:
          typeof data.datasetBundleBase64 === 'string' ? data.datasetBundleBase64 : undefined,
      });
      setGovernanceRefresh((c) => c + 1);
    } catch (err) {
      setGpuError(err instanceof Error ? err.message : String(err));
    } finally {
      setGpuSubmitting(false);
    }
  };

  const onnxBinaryInputRef = useRef<HTMLInputElement>(null);
  const localModelJsonInputRef = useRef<HTMLInputElement>(null);
  const [binaryImportStatus, setBinaryImportStatus] = useState<{
    busy: boolean;
    message?: string;
    error?: string;
  }>({ busy: false });

  const handleImportOnnxBinaryFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBinaryImportStatus({ busy: true });
    try {
      const buf = await file.arrayBuffer();
      const persisted = await buildPersistedOnnxModel(file.name, buf, {
        modality: specimen.modality,
      });
      onImportOnnxBinary?.(persisted);
      setBinaryImportStatus({
        busy: false,
        message:
          lang === 'zh'
            ? `ONNX 图结构已验证并持久化：${persisted.ref}（输入 ${persisted.inputName} → 输出 ${persisted.outputName}）`
            : `Verified & persisted ONNX binary: ${persisted.ref} (${persisted.inputName} → ${persisted.outputName})`,
      });
    } catch (err) {
      setBinaryImportStatus({
        busy: false,
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      e.target.value = '';
    }
  };

  const handleImportLocalJsonFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBinaryImportStatus({ busy: true });
    try {
      const text = await file.text();
      const trained = parseImportedLocalModelJson(text);
      onImportLocalModelJson?.(trained);
      setBinaryImportStatus({
        busy: false,
        message:
          lang === 'zh'
            ? `已验证 SHA-256 并导入本地模型：${trained.card.ref}`
            : `Verified SHA-256 & imported local model: ${trained.card.ref}`,
      });
    } catch (err) {
      setBinaryImportStatus({
        busy: false,
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      e.target.value = '';
    }
  };

  const handleImportOnnxMetaFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setOnnxImportMsg(null);
    setGpuError(null);
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      const resp = await fetch('/api/v1/train/import-onnx', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parsed),
      });
      const data = await resp.json();
      if (!resp.ok || !data?.ok || !data?.card) {
        throw new Error(data?.error || 'ONNX metadata verification failed');
      }
      const verifiedCard: ModelCard = {
        ...data.card,
        id: parsed.modelId || data.card.ref,
        trainingData: [{ name: `GPU LoRA (${parsed.backbone ?? backbone})`, license: data.card.license }],
        fileStatus: 'verified',
      };
      onImportOnnxModel?.(verifiedCard);
      setOnnxImportMsg(
        lang === 'zh'
          ? `已校验 SHA-256 并注册 ONNX 模型：${verifiedCard.ref}`
          : `Verified SHA-256 and registered ONNX model: ${verifiedCard.ref}`
      );
    } catch (err) {
      setGpuError(err instanceof Error ? err.message : String(err));
    } finally {
      e.target.value = '';
    }
  };

  const budgetToneClass =
    budget2d.tier === 'optimal'
      ? 'border-emerald-500/30 bg-emerald-500/8 text-emerald-700 dark:text-emerald-300'
      : budget2d.tier === 'tiled_safe'
      ? 'border-cyan-500/35 bg-cyan-500/8 text-blue-700 dark:text-cyan-300'
      : budget2d.tier === 'lod_guarded'
      ? 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300'
      : 'border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300';

  return (
    <div className="flex-1 min-h-0 overflow-y-auto bg-slate-50 dark:bg-[#0B0E17] p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="pb-4 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-[19px] font-display font-semibold text-slate-900 dark:text-slate-100">
              {t('models_title', lang)}
            </h1>
            <p className="mt-1 text-[13px] text-slate-500 dark:text-slate-400">
              {t('models_subtitle', lang)}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onGoRoiOnCanvas}
              className="h-8 px-3 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[12.5px] font-medium text-slate-800 dark:text-slate-100 flex items-center gap-1.5 cursor-pointer"
            >
              <Crop className="w-3.5 h-3.5 text-cyan-500" />
              <span>
                {roi
                  ? lang === 'zh'
                    ? `重画 ROI (${roi.width}×${roi.height})`
                    : `Redraw ROI (${roi.width}×${roi.height})`
                  : lang === 'zh'
                  ? '画布框选训练 ROI'
                  : 'Set Training ROI'}
              </span>
            </button>
            {onGoSmartGtOnCanvas && (
              <button
                type="button"
                onClick={onGoSmartGtOnCanvas}
                className="h-8 px-3.5 rounded-md border border-cyan-500/60 bg-cyan-500/10 hover:bg-cyan-500/20 text-[12.5px] font-medium text-blue-700 dark:text-cyan-300 flex items-center gap-1.5 whitespace-nowrap cursor-pointer"
              >
                <Wand2 className="w-3.5 h-3.5 shrink-0" />
                <span>{lang === 'zh' ? 'SAM 点选自动标注 GT' : 'Smart Click GT (SAM)'}</span>
              </button>
            )}
            <button
              type="button"
              onClick={onGoBrushOnCanvas}
              className="h-8 px-3.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[12.5px] font-medium text-slate-800 dark:text-slate-100 flex items-center gap-1.5 whitespace-nowrap cursor-pointer"
            >
              <Brush className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
              <span>{lang === 'zh' ? '画笔标注' : 'Paint Strokes'}</span>
            </button>
          </div>
        </div>

        {/* Live 2D Resource Budget & Tiling Guardrail Banner */}
        <div className={`px-3.5 py-2.5 rounded-md border text-[12px] flex flex-wrap items-center justify-between gap-2 ${budgetToneClass}`}>
          <div className="flex items-center gap-2 min-w-0">
            <Activity className="w-4 h-4 shrink-0" />
            <span className="font-mono font-semibold shrink-0">
              {activeWidth}×{activeHeight} px ({(budget2d.elementCount / 1e6).toFixed(2)} MP · ~{budget2d.estimatedPeakMB} MB)
            </span>
            <span className="truncate">
              {lang === 'zh' ? budget2d.recommendationZh : budget2d.recommendationEn}
            </span>
          </div>
          <span className="font-mono text-[11px] shrink-0 opacity-90">
            Tile {budget2d.recommendedTileSize}px · Halo {budget2d.recommendedOverlapPx}px
          </span>
        </div>

        {/* G-09 Multi-User Auth, Persistent SQLite Quota & Cost Circuit-Breaker Panel */}
        <GovernanceQuotaPanel
          lang={lang}
          refreshSignal={governanceRefresh}
          onAuthTokenChange={setAuthToken}
        />

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left: Verified Model Catalog */}
          <div className="lg:col-span-6 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="text-[12px] font-semibold uppercase tracking-wider text-slate-500">
                {lang === 'zh'
                  ? `模型 (${models.length})`
                  : `Models (${models.length})`}
              </div>
              <div className="flex items-center gap-1.5">
                <input
                  ref={onnxBinaryInputRef}
                  type="file"
                  accept=".onnx,application/octet-stream"
                  onChange={handleImportOnnxBinaryFile}
                  className="hidden"
                />
                <input
                  ref={localModelJsonInputRef}
                  type="file"
                  accept=".json,.vexmodel.json,application/json"
                  onChange={handleImportLocalJsonFile}
                  className="hidden"
                />
                <button
                  type="button"
                  disabled={binaryImportStatus.busy}
                  onClick={() => onnxBinaryInputRef.current?.click()}
                  className="h-7 px-2.5 rounded border border-cyan-500/45 bg-cyan-500/10 hover:bg-cyan-500/20 text-blue-700 dark:text-cyan-300 text-[11px] font-medium flex items-center gap-1 cursor-pointer disabled:opacity-50"
                >
                  <Upload className="w-3 h-3" />
                  <span>{lang === 'zh' ? '导入 .onnx 离线权重' : 'Import .onnx'}</span>
                </button>
                <button
                  type="button"
                  disabled={binaryImportStatus.busy}
                  onClick={() => localModelJsonInputRef.current?.click()}
                  className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200 text-[11px] font-medium flex items-center gap-1 cursor-pointer disabled:opacity-50"
                >
                  <FileJson className="w-3 h-3" />
                  <span>{lang === 'zh' ? '导入 .vexmodel.json' : 'Import Local JSON'}</span>
                </button>
              </div>
            </div>

            {binaryImportStatus.error && (
              <div className="p-2.5 rounded border border-amber-500/40 bg-amber-500/10 text-[11.5px] text-amber-700 dark:text-amber-300">
                {binaryImportStatus.error}
              </div>
            )}
            {binaryImportStatus.message && (
              <div className="p-2.5 rounded border border-emerald-500/35 bg-emerald-500/10 text-[11.5px] text-emerald-700 dark:text-emerald-300 flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                <span>{binaryImportStatus.message}</span>
              </div>
            )}

            <div className="divide-y divide-slate-200 dark:divide-slate-800 border border-slate-200 dark:border-slate-800 rounded-lg bg-white dark:bg-[#0E1320]">
              {models.map((m) => {
                const isBound = activeModelRef === m.ref;
                const localRecord = localModels?.get(m.ref);
                const onnxRecord = onnxModels?.get(m.ref);
                const isPersistedUserModel = Boolean(localRecord || onnxRecord);
                return (
                  <div key={m.ref} className="p-4 space-y-2.5">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-[14px] font-semibold text-slate-900 dark:text-slate-100">
                            {lang === 'zh' ? m.nameZh : m.nameEn}
                          </span>
                          <span className="font-mono text-[11px] text-slate-500">{m.ref}</span>
                          {onnxRecord && (
                            <span className="font-mono text-[10px] px-1.5 py-0.5 rounded border border-cyan-500/40 bg-cyan-500/10 text-blue-700 dark:text-cyan-300">
                              ONNX WASM · IndexedDB
                            </span>
                          )}
                          {localRecord && !onnxRecord && (
                            <span className="font-mono text-[10px] px-1.5 py-0.5 rounded border border-emerald-500/35 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300">
                              IndexedDB
                            </span>
                          )}
                        </div>
                        <div className="mt-1 flex items-center flex-wrap gap-x-2.5 gap-y-1 text-[11.5px] font-mono text-slate-500 dark:text-slate-400">
                          <span>{m.framework.toUpperCase()}</span>
                          <span>·</span>
                          <span>{m.task}</span>
                          <span>·</span>
                          <span>
                            {m.tiledInference
                              ? `Tiled ${m.tileSize ?? 512}px`
                              : lang === 'zh'
                              ? '整图推理'
                              : 'Full-frame'}
                          </span>
                          <span>·</span>
                          <span
                            className={
                              m.commercialUse
                                ? 'text-emerald-600 dark:text-emerald-400 inline-flex items-center gap-1'
                                : 'text-amber-600 dark:text-amber-400 inline-flex items-center gap-1'
                            }
                          >
                            {m.commercialUse ? (
                              <ShieldCheck className="w-3.5 h-3.5" />
                            ) : (
                              <ShieldAlert className="w-3.5 h-3.5" />
                            )}
                            {m.commercialUse
                              ? `${m.license} · ${t('models_commercial_ok', lang)}`
                              : `${m.license} · ${t('models_commercial_no', lang)}`}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center gap-1.5 shrink-0">
                        {localRecord && (
                          <button
                            type="button"
                            title={lang === 'zh' ? '导出可复现模型 (.vexmodel.json)' : 'Export .vexmodel.json'}
                            onClick={() =>
                              triggerTextDownload(
                                `${m.id}_${m.version}.vexmodel.json`,
                                exportLocalModelJson(localRecord),
                                'application/json;charset=utf-8'
                              )
                            }
                            className="h-8 w-8 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 flex items-center justify-center cursor-pointer"
                          >
                            <Download className="w-3.5 h-3.5" />
                          </button>
                        )}
                        {isPersistedUserModel && onDeletePersistedModel && (
                          <button
                            type="button"
                            title={lang === 'zh' ? '从 IndexedDB 删除此模型' : 'Delete persisted model'}
                            onClick={() => onDeletePersistedModel(m.ref)}
                            className="h-8 w-8 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:border-rose-500/50 hover:text-rose-500 text-slate-500 flex items-center justify-center cursor-pointer"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                        {m.fileStatus === 'missing' ? (
                          <span
                            title={lang === 'zh' ? '没有权重文件：导入对应的 ONNX 文件后才能使用' : 'No weight file: import the ONNX file to use it'}
                            className="h-8 px-3 rounded-md text-[12px] font-medium shrink-0 inline-flex items-center border border-amber-500/40 text-amber-700 dark:text-amber-300 bg-amber-500/10"
                          >
                            {lang === 'zh' ? '无权重，不可用' : 'No weights'}
                          </span>
                        ) : (
                          <button
                            type="button"
                            onClick={() => onBindModel(m.ref)}
                            className={`h-8 px-3 rounded-md text-[12px] font-medium shrink-0 transition-colors cursor-pointer ${
                              isBound
                                ? 'bg-emerald-600/15 text-emerald-700 dark:text-emerald-300 border border-emerald-500/40'
                                : 'bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950'
                            }`}
                          >
                            {isBound ? t('models_bound', lang) : t('models_bind', lang)}
                          </button>
                        )}
                      </div>
                    </div>

                    <p className="text-[12.5px] text-slate-600 dark:text-slate-300 leading-relaxed">
                      {lang === 'zh' ? m.descriptionZh : m.descriptionEn}
                    </p>

                    <div className="flex items-center flex-wrap gap-x-5 gap-y-1 text-[11.5px] font-mono tabular-nums text-slate-500 dark:text-slate-400">
                      {m.metrics.meanIoU !== undefined && (
                        <span>
                          Mean IoU:{' '}
                          <strong className="text-slate-800 dark:text-slate-200">
                            {(m.metrics.meanIoU * 100).toFixed(1)}%
                          </strong>
                        </span>
                      )}
                      {m.metrics.macroF1 !== undefined && (
                        <span>
                          Macro-F1:{' '}
                          <strong className="text-slate-800 dark:text-slate-200">
                            {(m.metrics.macroF1 * 100).toFixed(1)}%
                          </strong>
                        </span>
                      )}
                      {m.metrics.accuracy !== undefined && (
                        <span>
                          Acc:{' '}
                          <strong className="text-slate-800 dark:text-slate-200">
                            {(m.metrics.accuracy * 100).toFixed(1)}%
                          </strong>
                        </span>
                      )}
                      {m.sha256 ? <span className="ml-auto text-[10.5px]">SHA-256 {m.sha256.slice(0, 12)}</span> : null}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right: Dual-Mode Hybrid Few-Shot Deep Learning Training Studio */}
          <div className="lg:col-span-6 border border-slate-200 dark:border-slate-800 rounded-lg bg-white dark:bg-[#0E1320] p-5 space-y-4">
            {/* Mode Switcher */}
            <div className="flex items-center justify-between gap-2 pb-3 border-b border-slate-200 dark:border-slate-800">
              <div className="grid grid-cols-2 w-full rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#0B0F1A] p-0.5 text-[12px]">
                <button
                  type="button"
                  onClick={() => setTabMode('browser_dl')}
                  className={`h-8 rounded flex items-center justify-center gap-1.5 font-medium cursor-pointer transition-colors ${
                    tabMode === 'browser_dl'
                      ? 'bg-white dark:bg-slate-800 text-blue-700 dark:text-cyan-300 shadow-xs'
                      : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                  }`}
                >
                  <Cpu className="w-3.5 h-3.5" />
                  <span>{lang === 'zh' ? '浏览器实时微调 (Offline DL)' : 'Browser Few-Shot DL'}</span>
                </button>
                <button
                  type="button"
                  onClick={() => setTabMode('gpu_lora')}
                  className={`h-8 rounded flex items-center justify-center gap-1.5 font-medium cursor-pointer transition-colors ${
                    tabMode === 'gpu_lora'
                      ? 'bg-white dark:bg-slate-800 text-blue-700 dark:text-cyan-300 shadow-xs'
                      : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                  }`}
                >
                  <Cloud className="w-3.5 h-3.5" />
                  <span>{lang === 'zh' ? '本地 / 云端 GPU LoRA' : 'Local / Cloud GPU LoRA'}</span>
                </button>
              </div>
            </div>

            {/* Shared Training Data Inventory & Pre-Flight Gate */}
            <div className="py-2.5 border-b border-slate-200 dark:border-slate-800 grid grid-cols-2 gap-2.5 text-[12px] font-mono tabular-nums">
              <div>
                <div className="text-[10.5px] text-slate-500">
                  {lang === 'zh' ? '当前训练样本' : 'Active Specimen'}
                </div>
                <div className="mt-0.5 text-slate-900 dark:text-slate-100 truncate">
                  {specimen.name}
                </div>
              </div>
              <div>
                <div className="text-[10.5px] text-slate-500">
                  {lang === 'zh' ? '有效标注像素 / 类别' : 'Labeled Pixels / Classes'}
                </div>
                <div className="mt-0.5 flex items-center gap-1.5">
                  <span
                    className={
                      preflightPassed
                        ? 'text-emerald-600 dark:text-emerald-400 font-semibold'
                        : 'text-amber-600 dark:text-amber-400 font-semibold'
                    }
                  >
                    {annotationSummary.totalLabeledPixels.toLocaleString()} px · {annotationSummary.distinctClasses} cls
                  </span>
                </div>
              </div>
              <div className="col-span-2 flex items-center justify-between gap-2 pt-2 border-t border-slate-200/70 dark:border-slate-800/70 text-[11.5px]">
                <span className="flex items-center gap-1.5 text-slate-600 dark:text-slate-300 truncate">
                  <Crop className="w-3.5 h-3.5 text-cyan-500 shrink-0" />
                  <span className="truncate">
                    {roi
                      ? lang === 'zh'
                        ? `ROI 训练区: ${roi.width}×${roi.height} px`
                        : `ROI Scope: ${roi.width}×${roi.height} px`
                      : lang === 'zh'
                      ? '训练区: 整幅视场'
                      : 'Scope: Full FOV'}
                  </span>
                </span>
                <div className="flex items-center gap-2 shrink-0 font-sans">
                  {segmentedResultsRoi && (
                    <button
                      type="button"
                      onClick={() => onChangeRoi(segmentedResultsRoi)}
                      className="text-cyan-600 dark:text-cyan-400 hover:underline text-[11px] flex items-center gap-1 whitespace-nowrap cursor-pointer"
                    >
                      <Crosshair className="w-3 h-3 shrink-0" />
                      <span>{lang === 'zh' ? '聚焦结果' : 'Focus Results'}</span>
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={onGoRoiOnCanvas}
                    className="text-blue-600 dark:text-cyan-400 hover:underline text-[11px] whitespace-nowrap cursor-pointer"
                  >
                    {roi ? (lang === 'zh' ? '重选' : 'Redraw') : lang === 'zh' ? '框选 ROI' : 'Set ROI'}
                  </button>
                  {roi && (
                    <button
                      type="button"
                      onClick={() => onChangeRoi(null)}
                      className="text-slate-400 hover:text-rose-500 text-[11px] whitespace-nowrap cursor-pointer"
                    >
                      {lang === 'zh' ? '清除' : 'Clear'}
                    </button>
                  )}
                </div>
              </div>
              <div className="col-span-2 flex items-center gap-4 text-[11px]">
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-cyan-500" />
                  {lang === 'zh' ? '前景' : 'Class 1'}: {classCounts[1]}
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-rose-500" />
                  {lang === 'zh' ? '背景' : 'Class 2'}: {classCounts[2]}
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-amber-500" />
                  {lang === 'zh' ? '缺陷' : 'Class 3'}: {classCounts[3]}
                </span>
                <span className="ml-auto text-slate-500">
                  {lang === 'zh' ? `复核纠错 ${specCorrections.length} 条` : `${specCorrections.length} edits`}
                </span>
              </div>
            </div>

            {/* Shared Model Metadata Form */}
            <div className="grid grid-cols-3 gap-2.5 text-[12px]">
              <label className="col-span-2 space-y-1">
                <span className="text-[11px] text-slate-500 block">Model ID</span>
                <input
                  type="text"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  className="w-full h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
                />
              </label>
              <label className="space-y-1">
                <span className="text-[11px] text-slate-500 block">Version</span>
                <input
                  type="text"
                  value={version}
                  onChange={(e) => setVersion(e.target.value)}
                  className="w-full h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
                />
              </label>
              <label className="col-span-3 space-y-1">
                <span className="text-[11px] text-slate-500 block">
                  {lang === 'zh' ? '类别名称（逗号分隔）' : 'Class Names (comma-separated)'}
                </span>
                <input
                  type="text"
                  value={classNames}
                  onChange={(e) => setClassNames(e.target.value)}
                  className="w-full h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
                />
              </label>
            </div>

            {/* TAB 1: BROWSER FEW-SHOT DEEP LEARNING (WEB WORKER) */}
            {tabMode === 'browser_dl' && (
              <div className="space-y-3.5">
                <p className="text-[12px] text-slate-500 dark:text-slate-400 leading-relaxed">
                  {lang === 'zh'
                    ? '冻结 16 通道多尺度金字塔编码器 + 轻量残差瓶颈非线性 MLP 解码头（AdamW + 困难负样本挖掘 + 16×16 空间分块留出验证），在独立 Web Worker 后台运行，主线程零卡顿。'
                    : 'Frozen 16-ch Multi-Scale Pyramid Encoder + Trainable Residual Bottleneck MLP Decoder (AdamW + Hard-Negative Mining + 16×16 Spatial-Block Hold-Out) in a non-blocking Web Worker.'}
                </p>

                <div className="grid grid-cols-2 gap-2.5 text-[11.5px]">
                  <label className="space-y-1">
                    <span className="text-slate-500 block">
                      {lang === 'zh' ? '解码头隐层宽度 (Hidden Dim)' : 'Decoder Hidden Dim'}
                    </span>
                    <select
                      value={hiddenDim}
                      onChange={(e) => setHiddenDim(Number(e.target.value) as 24 | 32 | 48)}
                      className="w-full h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
                    >
                      <option value={24}>24 (Fast · ~1.2k params)</option>
                      <option value={32}>32 (Balanced · ~2.1k params)</option>
                      <option value={48}>48 (High Capacity · ~4.2k params)</option>
                    </select>
                  </label>
                  <label className="space-y-1">
                    <span className="text-slate-500 block">
                      {lang === 'zh' ? '迭代轮数 (Epochs)' : 'Training Epochs'}
                    </span>
                    <select
                      value={dlEpochs}
                      onChange={(e) => setDlEpochs(Number(e.target.value) as 20 | 32 | 50)}
                      className="w-full h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
                    >
                      <option value={20}>20 Epochs (Quick)</option>
                      <option value={32}>32 Epochs (Recommended)</option>
                      <option value={50}>50 Epochs (Fine-Grain)</option>
                    </select>
                  </label>
                </div>

                <div className="flex flex-wrap items-center gap-4 text-[11.5px] text-slate-600 dark:text-slate-300">
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={hardNegativeMining}
                      onChange={(e) => setHardNegativeMining(e.target.checked)}
                      className="accent-blue-600 dark:accent-cyan-400"
                    />
                    <span>{lang === 'zh' ? '困难负样本挖掘 (OHEM)' : 'Hard-Negative Mining'}</span>
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={bilateralSmooth}
                      onChange={(e) => setBilateralSmooth(e.target.checked)}
                      className="accent-blue-600 dark:accent-cyan-400"
                    />
                    <span>{lang === 'zh' ? '保边双边概率平滑' : 'Edge-Preserving Smooth'}</span>
                  </label>
                </div>

                {trainError && (
                  <div className="p-2.5 rounded border border-amber-500/40 bg-amber-500/10 text-[12px] text-amber-700 dark:text-amber-300">
                    {trainError}
                  </div>
                )}

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={isTraining}
                    onClick={handleTrainBrowserDl}
                    className="flex-1 h-9 rounded-md bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[13px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-50"
                  >
                    <Sparkles className="w-4 h-4" />
                    <span>
                      {isTraining
                        ? (() => {
                            const pct = progressEv
                              ? progressEv.phase === 'extracting'
                                ? 12
                                : progressEv.phase === 'evaluating'
                                ? 96
                                : Math.round(
                                    15 +
                                      80 *
                                        (progressEv.epoch / Math.max(1, progressEv.totalEpochs))
                                  )
                              : 5;
                            return lang === 'zh'
                              ? `Worker 训练中… ${pct}%`
                              : `Training in Worker… ${pct}%`;
                          })()
                        : lang === 'zh'
                        ? '启动 Web Worker 少样本深度微调'
                        : 'Train Deep Few-Shot Head (Worker)'}
                    </span>
                  </button>
                  {isTraining && (
                    <button
                      type="button"
                      onClick={handleCancelBrowserDl}
                      className="h-9 px-3 rounded-md border border-rose-500/50 bg-rose-500/10 text-rose-600 dark:text-rose-300 text-[12px] font-medium flex items-center gap-1 cursor-pointer"
                    >
                      <Square className="w-3.5 h-3.5" />
                      <span>{lang === 'zh' ? '中止' : 'Abort'}</span>
                    </button>
                  )}
                </div>

                {/* Live Progress Bar */}
                {isTraining && (
                  <div className="space-y-1">
                    <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div
                        className="h-full bg-blue-600 dark:bg-cyan-400 transition-all duration-150"
                        style={{
                          width: `${
                            progressEv
                              ? progressEv.phase === 'extracting'
                                ? 12
                                : progressEv.phase === 'evaluating'
                                ? 96
                                : Math.max(
                                    15,
                                    Math.round(
                                      15 +
                                        80 *
                                          (progressEv.epoch / Math.max(1, progressEv.totalEpochs))
                                    )
                                  )
                              : 6
                          }%`,
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[10.5px] font-mono text-slate-500">
                      <span>
                        {progressEv?.phase === 'training'
                          ? lang === 'zh'
                            ? `残差 MLP 迭代中 (Loss ${progressEv.trainLoss.toFixed(3)})`
                            : `Training Residual MLP (Loss ${progressEv.trainLoss.toFixed(3)})`
                          : progressEv?.phase === 'evaluating'
                          ? lang === 'zh'
                            ? '正在计算 16×16 空间分块留出集指标…'
                            : 'Evaluating 16×16 spatial-block hold-out…'
                          : lang === 'zh'
                          ? '正在提取 16 通道多尺度特征金字塔…'
                          : 'Extracting 16-ch multi-scale feature pyramid…'}
                      </span>
                      {progressEv && progressEv.totalEpochs > 0 && (
                        <span>
                          ep {progressEv.epoch}/{progressEv.totalEpochs}
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* Convergence Curve */}
                <TrainingCurveChart history={curveHistory} lang={lang} />

                {/* Real Spatial-Block Hold-Out Evaluation Results */}
                {trainedCandidate && (
                  <div className="pt-3 border-t border-slate-200 dark:border-slate-800 space-y-3">
                    <div className="flex items-center justify-between gap-2 text-[11.5px]">
                      <span className="font-semibold text-slate-900 dark:text-slate-100">
                        {lang === 'zh'
                          ? '空间分块留出集验证 (16×16 Block Hold-Out)'
                          : 'Spatial-Block Hold-Out Evaluation'}
                      </span>
                      <span className="font-mono text-[10.5px] text-slate-500">
                        {trainedCandidate.metrics.durationMs !== undefined
                          ? `${trainedCandidate.metrics.durationMs} ms`
                          : ''}
                        {trainedCandidate.metrics.cacheHit
                          ? lang === 'zh'
                            ? ' · 特征缓存命中'
                            : ' · Cache Hit'
                          : ''}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 font-mono tabular-nums">
                      <div className="p-2.5 rounded border border-slate-200 dark:border-slate-800">
                        <div className="text-[10.5px] text-slate-500">Mean IoU</div>
                        <div className="text-[17px] font-semibold text-blue-600 dark:text-cyan-400">
                          {(trainedCandidate.metrics.meanIoU * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="p-2.5 rounded border border-slate-200 dark:border-slate-800">
                        <div className="text-[10.5px] text-slate-500">Macro-F1</div>
                        <div className="text-[17px] font-semibold text-slate-900 dark:text-slate-100">
                          {(trainedCandidate.metrics.macroF1 * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="p-2.5 rounded border border-slate-200 dark:border-slate-800">
                        <div className="text-[10.5px] text-slate-500">Accuracy</div>
                        <div className="text-[17px] font-semibold text-slate-900 dark:text-slate-100">
                          {(trainedCandidate.metrics.accuracy * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    <div className="space-y-1 text-[11.5px] font-mono">
                      {Object.entries(trainedCandidate.metrics.perClassIoU).map(([cname, val]) => (
                        <div
                          key={cname}
                          className="flex justify-between text-slate-600 dark:text-slate-300"
                        >
                          <span>{cname}</span>
                          <span>IoU {(val * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>

                    <button
                      type="button"
                      onClick={handlePublish}
                      className="w-full h-9 rounded-md border border-emerald-600/50 bg-emerald-600 hover:bg-emerald-500 text-white text-[12.5px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
                    >
                      <Layers className="w-4 h-4" />
                      <span>{t('models_publish_btn', lang)}</span>
                    </button>

                    {publishedRef && (
                      <div className="flex items-center gap-1.5 text-[12px] text-emerald-600 dark:text-emerald-400">
                        <CheckCircle2 className="w-4 h-4 shrink-0" />
                        <span>
                          {lang === 'zh'
                            ? `已发布并绑定 ${publishedRef} 至工作台（512×512 羽化分块推理）！`
                            : `Published and bound ${publishedRef} (512×512 feathered tiled inference)!`}
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: LOCAL / CLOUD GPU DEEP LORA TRAINING (DECLARATIVE SANDBOX) */}
            {tabMode === 'gpu_lora' && (
              <div className="space-y-3.5">
                {/* Provider Preset Selector */}
                <div className="space-y-1.5 text-[12px]">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-medium text-slate-500">
                      {lang === 'zh' ? '算力部署形态 / 云端服务商' : 'GPU Target / Cloud Provider'}
                    </span>
                    <span className="font-mono text-[10.5px] text-slate-500">
                      {gpuEndpointConfigured
                        ? lang === 'zh'
                          ? '服务端 GPU 网关已就绪'
                          : 'Server GPU endpoint active'
                        : lang === 'zh'
                        ? '离线声明式包 + 沙箱脚本模式'
                        : 'Declarative Manifest + Sandbox Script'}
                    </span>
                  </div>
                  <select
                    value={selectedProvider}
                    onChange={(e) => setSelectedProvider(e.target.value)}
                    className="w-full h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
                  >
                    {gpuPresets.map((p) => (
                      <option key={p.id} value={p.id}>
                        {lang === 'zh' ? p.nameZh : p.nameEn}
                      </option>
                    ))}
                  </select>
                  {activePreset && (
                    <div className="p-2.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] text-[11px] space-y-0.5 text-slate-600 dark:text-slate-300">
                      <div className="font-mono text-slate-800 dark:text-slate-100">
                        {activePreset.gpuTier}
                      </div>
                      <div>{lang === 'zh' ? activePreset.billingZh : activePreset.billingEn}</div>
                      <div className="text-slate-500">
                        {lang === 'zh' ? '适用：' : 'Best for: '}
                        {lang === 'zh'
                          ? activePreset.recommendedForZh
                          : activePreset.recommendedForEn}
                      </div>
                    </div>
                  )}
                </div>

                {/* Strict Declarative Allowlist Parameters */}
                <div className="grid grid-cols-2 gap-2.5 text-[11.5px]">
                  <label className="space-y-1">
                    <span className="text-slate-500 block">
                      {lang === 'zh' ? '白名单视觉骨干 (Backbone)' : 'Allowlisted Backbone'}
                    </span>
                    <select
                      value={backbone}
                      onChange={(e) => setBackbone(e.target.value as GpuBackboneId)}
                      className="w-full h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
                    >
                      <option value="dinov2_vit_s14">DINOv2-ViT-S/14 (Frozen + LoRA)</option>
                      <option value="convnext_tiny">ConvNeXt-Tiny (Microscopy)</option>
                      <option value="sam_vit_b_lora">SAM-ViT-B LoRA Adapter</option>
                      <option value="unet_resnet34">UNet-ResNet34 (Classic)</option>
                    </select>
                  </label>

                  <label className="space-y-1">
                    <span className="text-slate-500 block">LoRA Rank / Epochs</span>
                    <div className="grid grid-cols-2 gap-1.5">
                      <select
                        value={loraRank}
                        onChange={(e) => setLoraRank(Number(e.target.value) as 4 | 8 | 16)}
                        className="h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
                      >
                        <option value={4}>r=4</option>
                        <option value={8}>r=8</option>
                        <option value={16}>r=16</option>
                      </select>
                      <input
                        type="number"
                        min={5}
                        max={80}
                        value={gpuEpochs}
                        onChange={(e) =>
                          setGpuEpochs(Math.max(5, Math.min(80, Number(e.target.value) || 25)))
                        }
                        className="h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
                      />
                    </div>
                  </label>

                  <label className="space-y-1">
                    <span className="text-slate-500 block">
                      {lang === 'zh' ? '损失函数 (Loss)' : 'Loss Objective'}
                    </span>
                    <select
                      value={gpuLoss}
                      onChange={(e) => setGpuLoss(e.target.value as GpuLossId)}
                      className="w-full h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
                    >
                      <option value="focal_dice">Focal + Dice (Imbalanced)</option>
                      <option value="ce_dice">CrossEntropy + Dice</option>
                      <option value="tversky">Tversky (Small Targets)</option>
                    </select>
                  </label>

                  <label className="space-y-1">
                    <span className="text-slate-500 block">
                      {lang === 'zh' ? '混合精度 (Precision)' : 'Mixed Precision'}
                    </span>
                    <select
                      value={gpuPrecision}
                      onChange={(e) => setGpuPrecision(e.target.value as GpuPrecisionId)}
                      className="w-full h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100"
                    >
                      <option value="bf16">BF16 (RTX 40/L4/A100)</option>
                      <option value="fp16">FP16 (T4/RTX 30)</option>
                    </select>
                  </label>
                </div>

                {/* Anti-Abuse Sandbox Policy Strip */}
                <div className="p-2.5 rounded border border-emerald-500/30 bg-emerald-500/8 text-[11px] text-emerald-800 dark:text-emerald-300 space-y-1">
                  <div className="font-semibold flex items-center gap-1.5">
                    <Lock className="w-3.5 h-3.5 shrink-0" />
                    <span>
                      {lang === 'zh'
                        ? '在线训练防滥用安全策略（声明式白名单沙箱）'
                        : 'Online Training Anti-Abuse Policy (Declarative Sandbox)'}
                    </span>
                  </div>
                  <div className="text-[10.5px] opacity-90">
                    {lang === 'zh'
                      ? '禁止任意代码/Shell/pip/Git URL；强制前置标注像素门禁（≥2类且≥64px）；容器断网零出站；单任务上限 ≤480s；仅允许导出 ≤50MB 校验 ONNX。'
                      : 'Blocks arbitrary code/shell/git URLs; enforces pre-flight annotation gate (≥2 classes & ≥64px); zero-egress container; ≤480s wall-clock cap; ONNX-only output (≤50 MB).'}
                  </div>
                </div>

                {gpuError && (
                  <div className="p-2.5 rounded border border-amber-500/40 bg-amber-500/10 text-[12px] text-amber-700 dark:text-amber-300">
                    {gpuError}
                  </div>
                )}

                <button
                  type="button"
                  disabled={gpuSubmitting || !preflightPassed}
                  onClick={handleSubmitGpuJob}
                  className="w-full h-9 rounded-md bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[13px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-45"
                >
                  <Cloud className="w-4 h-4" />
                  <span>
                    {gpuSubmitting
                      ? lang === 'zh'
                        ? '正在校验并生成沙箱任务…'
                        : 'Validating & Generating Sandbox Job…'
                      : !preflightPassed
                      ? lang === 'zh'
                        ? '请先在画布标注 ≥2 类且 ≥64 像素以通过前置门禁'
                        : 'Annotate ≥2 classes & ≥64 px on canvas to pass Pre-Flight Gate'
                      : lang === 'zh'
                      ? '提交声明式 GPU LoRA 微调任务'
                      : 'Submit Declarative GPU LoRA Job'}
                  </span>
                </button>

                {latestGpuJob && (
                  <div className="p-3 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] space-y-2.5 text-[11.5px]">
                    <div className="flex items-center justify-between gap-2 font-mono">
                      <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                        {latestGpuJob.job.jobId}
                      </span>
                      <span className="text-[10.5px] text-slate-500">
                        SHA {latestGpuJob.job.manifestDigest.slice(0, 12)}
                      </span>
                    </div>
                    <p className="text-slate-600 dark:text-slate-300 text-[11px]">
                      {latestGpuJob.job.message}
                    </p>
                    {latestGpuJob.sanitization?.reencoded && (
                      <div className="p-2 rounded border border-emerald-500/30 bg-emerald-500/8 text-[10.5px] font-mono text-emerald-700 dark:text-emerald-300 flex flex-wrap items-center justify-between gap-1.5">
                        <span className="flex items-center gap-1">
                          <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
                          <span>
                            {lang === 'zh'
                              ? `纯像素重编码已校验 (${latestGpuJob.sanitization.width}×${latestGpuJob.sanitization.height})`
                              : `Pure-Pixel Re-encoded (${latestGpuJob.sanitization.width}×${latestGpuJob.sanitization.height})`}
                          </span>
                        </span>
                        <span>
                          IMG {latestGpuJob.sanitization.imageSanitizedSha256.slice(0, 10)} ·{' '}
                          {lang === 'zh'
                            ? `已清洗元数据块 ${latestGpuJob.sanitization.strippedChunks.length}`
                            : `Stripped chunks ${latestGpuJob.sanitization.strippedChunks.length}`}
                        </span>
                      </div>
                    )}
                    <div className="flex flex-wrap items-center gap-2">
                      {latestGpuJob.datasetBundleBase64 && (
                        <button
                          type="button"
                          onClick={() =>
                            triggerBase64BinaryDownload(
                              `vexel_dataset_${latestGpuJob.job.jobId}.zip`,
                              latestGpuJob.datasetBundleBase64!,
                              'application/zip'
                            )
                          }
                          className="h-7 px-2.5 rounded border border-emerald-500/40 bg-emerald-600 hover:bg-emerald-700 text-white text-[11px] font-medium flex items-center gap-1.5 cursor-pointer"
                        >
                          <Archive className="w-3.5 h-3.5" />
                          <span>
                            {lang === 'zh' ? '下载安全训练集 (.zip)' : 'Download Sanitized Dataset (.zip)'}
                          </span>
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() =>
                          triggerTextDownload(
                            `train_vexel_lora_${latestGpuJob.job.jobId}.py`,
                            latestGpuJob.pythonScript,
                            'text/x-python;charset=utf-8'
                          )
                        }
                        className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11px] font-medium flex items-center gap-1.5 cursor-pointer"
                      >
                        <FileCode2 className="w-3.5 h-3.5 text-cyan-500" />
                        <span>{lang === 'zh' ? '下载只读 PyTorch LoRA 脚本' : 'Download Sandbox .py'}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          triggerTextDownload(
                            `manifest_${latestGpuJob.job.jobId}.json`,
                            latestGpuJob.manifestJson,
                            'application/json;charset=utf-8'
                          )
                        }
                        className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-[11px] font-medium flex items-center gap-1.5 cursor-pointer"
                      >
                        <FileJson className="w-3.5 h-3.5 text-amber-500" />
                        <span>{lang === 'zh' ? '下载签名 Manifest JSON' : 'Download Manifest'}</span>
                      </button>
                    </div>
                  </div>
                )}

                {/* Import Verified ONNX Metadata back into Model Hub */}
                <div className="pt-2.5 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between gap-2 text-[11.5px]">
                  <div className="text-slate-500 dark:text-slate-400">
                    {lang === 'zh'
                      ? '训练完成后回传 vexel_model_meta.json 校验入库'
                      : 'Import exported vexel_model_meta.json (SHA-256 verified)'}
                  </div>
                  <input
                    ref={onnxInputRef}
                    type="file"
                    accept=".json,application/json"
                    onChange={handleImportOnnxMetaFile}
                    className="hidden"
                  />
                  <button
                    type="button"
                    onClick={() => onnxInputRef.current?.click()}
                    className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-800 dark:text-slate-100 text-[11px] font-medium flex items-center gap-1.5 shrink-0 cursor-pointer"
                  >
                    <Upload className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
                    <span>{lang === 'zh' ? '导入 ONNX 元数据' : 'Import ONNX Meta'}</span>
                  </button>
                </div>
                {onnxImportMsg && (
                  <div className="flex items-center gap-1.5 text-[11.5px] text-emerald-600 dark:text-emerald-400">
                    <Download className="w-3.5 h-3.5 shrink-0" />
                    <span>{onnxImportMsg}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
