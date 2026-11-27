import crypto from 'crypto';

/**
 * Shared server-side security guardrails for Vexel AI Planning & Online GPU Training.
 * Isolated in a pure module so both Express routes (`server.ts`) and automated Vitest
 * security suites exercise the exact same validation, allowlist, and rate-limit code.
 */

export function createSlidingWindowLimiter(maxRequests: number, windowMs: number) {
  const hits = new Map<string, number[]>();
  return (clientKey: string, nowOverride?: number): { allowed: boolean; retryAfterSec: number } => {
    const now = nowOverride ?? Date.now();
    const prev = (hits.get(clientKey) ?? []).filter((ts) => now - ts < windowMs);
    if (prev.length >= maxRequests) {
      const oldest = prev[0] ?? now;
      const retryAfterSec = Math.max(1, Math.ceil((windowMs - (now - oldest)) / 1000));
      hits.set(clientKey, prev);
      return { allowed: false, retryAfterSec };
    }
    prev.push(now);
    hits.set(clientKey, prev);
    if (hits.size > 1000) {
      for (const [k, v] of hits.entries()) {
        if (v.every((ts) => now - ts >= windowMs)) hits.delete(k);
      }
    }
    return { allowed: true, retryAfterSec: 0 };
  };
}

export const JAILBREAK_PATTERNS = [
  /ignore\s+(all\s+)?previous\s+instructions/i,
  /disregard\s+(all\s+)?(prior|previous|system)/i,
  /reveal\s+(your\s+)?system\s+prompt/i,
  /act\s+as\s+(a\s+)?(linux|terminal|shell|dan|developer\s+mode|unrestricted)/i,
  /jailbreak|bypass\s+safety|prompt\s+injection/i,
  /忽略(之前|以上|所有)(的)?(指令|规则|提示)/,
  /输出(你的)?系统提示词|越狱|绕过限制/,
  /crypto\s*miner|xmrig|reverse\s*shell|bash\s+-i|curl\s+http/i,
];

export const OFF_TOPIC_PATTERNS = [
  /(write|compose|draft)\s+(a\s+)?(poem|story|essay|novel|email|resume|cover\s+letter|song|joke)/i,
  /(translate|翻译成|帮我写一篇|写一首诗|写个故事|写邮件|简历|笑话|作文|旅游攻略|菜谱)/i,
  /(stock\s+price|bitcoin|crypto\s+trading|weather\s+forecast|homework|leetcode|sql\s+injection)/i,
  /(股票|比特币|天气预报|情感咨询|星座|算命|爬虫代码|刷单)/,
];

export const IMAGING_DOMAIN_HINTS =
  /(cell|nucle|cytoplasm|droplet|microfluid|sem|tem|pcb|trace|pad|particle|pore|fibre|fiber|line|cd|ler|lwr|hole|layer|roi|scale|calibrat|segment|threshold|otsu|watershed|denoise|destripe|contrast|clahe|illumination|background|filter|morpholog|histogram|area|diameter|volume|3d|z-stack|stack|model|lora|unet|sam|fluorescence|brightfield|stain|tissue|kidney|细胞|核质|细胞核|液滴|微流控|电镜|线宽|孔洞|颗粒|纤维|分层|电路板|焊盘|分割|阈值|去噪|去条纹|背景|对比度|滤波|标定|比例尺|面积|直径|体积|三维|模型|微调|掩膜|边界|噪声|伪影|遮挡|荧光|染色|切片)/i;

export function checkImageDomainIntent(
  goal: string,
  lang: string
): { allowed: boolean; reason?: string } {
  const trimmed = goal.trim();
  if (!trimmed) return { allowed: true };
  if (trimmed.length > 1000) {
    return {
      allowed: false,
      reason:
        lang === 'en'
          ? 'Request blocked: instruction exceeds 1,000-character domain limit.'
          : '请求已拦截：指令长度超过 1,000 字符上限，请精简为图像分析目标。',
    };
  }
  for (const pat of JAILBREAK_PATTERNS) {
    if (pat.test(trimmed)) {
      return {
        allowed: false,
        reason:
          lang === 'en'
            ? 'Domain Gatekeeper blocked a prompt-injection / off-policy instruction before invoking AI.'
            : '域安全网关已拦截越权/提示词注入指令（未消耗云端 AI 配额）。Vexel AI 仅限于显微与工业图像分析工作流编排。',
      };
    }
  }
  for (const pat of OFF_TOPIC_PATTERNS) {
    if (pat.test(trimmed) && !IMAGING_DOMAIN_HINTS.test(trimmed)) {
      return {
        allowed: false,
        reason:
          lang === 'en'
            ? 'Domain Gatekeeper: Vexel AI is strictly scoped to scientific/industrial image analysis and workflow planning.'
            : '域边界拦截：Vexel AI 助手仅限用于当前项目的图像预处理、分割、计量与模型编排，不支持通用闲聊或非图像任务。',
      };
    }
  }
  return { allowed: true };
}

export const ALLOWED_PREPROCESS_OPS = new Set([
  'img.bg_level',
  'img.clahe',
  'img.gaussian',
  'img.median',
  'sem.destripe',
  'sem.denoise',
  'img.crop_bar',
  'img.normalize',
]);

export const ALLOWED_ALL_OPS = new Set([
  ...ALLOWED_PREPROCESS_OPS,
  'model.segment',
  'droplet.segment',
  'droplet.encapsulation',
  'droplet.fluorescence',
  'cell.segment',
  'cell.nuclei',
  'pcb.inspect',
  'sem.lines',
  'sem.holes',
  'sem.particles',
  'sem.fibres',
  'sem.layers',
]);

export function sanitizePlanProposal(
  raw: Record<string, unknown>,
  availableRecipes: Array<{ id?: string }>,
  availableModels: Array<{ ref?: string }>
): Record<string, unknown> {
  const recipeIds = new Set(
    availableRecipes.map((r) => (typeof r?.id === 'string' ? r.id : '')).filter(Boolean)
  );
  const modelRefs = new Set(
    availableModels.map((m) => (typeof m?.ref === 'string' ? m.ref : '')).filter(Boolean)
  );

  const recipeId =
    typeof raw.recipeId === 'string' && (recipeIds.size === 0 || recipeIds.has(raw.recipeId))
      ? raw.recipeId
      : availableRecipes[0]?.id ?? 'cell_segmentation';

  const modelRef =
    typeof raw.modelRef === 'string' && modelRefs.has(raw.modelRef) ? raw.modelRef : null;

  const addPreprocess = Array.isArray(raw.addPreprocess)
    ? raw.addPreprocess
        .filter((op): op is string => typeof op === 'string' && ALLOWED_PREPROCESS_OPS.has(op))
        .slice(0, 6)
    : [];

  const removePreprocess = Array.isArray(raw.removePreprocess)
    ? raw.removePreprocess
        .filter((op): op is string => typeof op === 'string' && ALLOWED_PREPROCESS_OPS.has(op))
        .slice(0, 6)
    : [];

  const paramOverrides = Array.isArray(raw.paramOverrides)
    ? raw.paramOverrides
        .filter(
          (item): item is { stepOp: string; key: string; value: string } =>
            Boolean(item) &&
            typeof item === 'object' &&
            typeof (item as Record<string, unknown>).stepOp === 'string' &&
            ALLOWED_ALL_OPS.has((item as Record<string, unknown>).stepOp as string) &&
            typeof (item as Record<string, unknown>).key === 'string' &&
            /^[a-zA-Z0-9_.-]{1,48}$/.test((item as Record<string, unknown>).key as string) &&
            (item as Record<string, unknown>).value !== undefined
        )
        .slice(0, 12)
        .map((item) => ({
          stepOp: item.stepOp,
          key: item.key,
          value: String(item.value).slice(0, 64),
        }))
    : [];

  const reasons = Array.isArray(raw.reasons)
    ? raw.reasons
        .filter((r): r is string => typeof r === 'string' && r.trim().length > 0)
        .slice(0, 6)
        .map((r) => r.slice(0, 220))
    : [];

  const caution =
    typeof raw.caution === 'string' && raw.caution.trim() ? raw.caution.slice(0, 220) : null;

  return {
    recipeId,
    modelRef,
    addPreprocess,
    removePreprocess,
    paramOverrides,
    reasons,
    caution,
  };
}

export function isBlockedMetadataHost(urlStr: string): boolean {
  try {
    const parsed = new URL(urlStr);
    const host = parsed.hostname.toLowerCase();
    if (
      host === '169.254.169.254' ||
      host === 'metadata.google.internal' ||
      host === 'metadata.goog' ||
      host.startsWith('169.254.') ||
      host === '0.0.0.0'
    ) {
      return true;
    }
    return false;
  } catch {
    return true;
  }
}

/**
 * Strict outbound URL host check blocking cloud metadata, loopback, RFC-1918 private networks,
 * IPv6 ULA/link-local, and internal hostnames (used for remote OME-Zarr and cloud endpoints).
 */
export function isBlockedPrivateOrMetadataHost(urlStr: string): boolean {
  if (isBlockedMetadataHost(urlStr)) return true;
  try {
    const parsed = new URL(urlStr);
    const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '');
    if (
      host === 'localhost' ||
      host === '::1' ||
      host === '::' ||
      host.endsWith('.local') ||
      host.endsWith('.internal') ||
      host.startsWith('127.') ||
      host.startsWith('10.') ||
      host.startsWith('192.168.') ||
      host.startsWith('fc') ||
      host.startsWith('fd') ||
      host.startsWith('fe80:')
    ) {
      return true;
    }
    const ipv4 = host.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
    if (ipv4) {
      const a = Number(ipv4[1]);
      const b = Number(ipv4[2]);
      if (a === 172 && b >= 16 && b <= 31) return true;
      if (a === 100 && b >= 64 && b <= 127) return true; // CGNAT
      if (a === 0) return true;
    }
    return false;
  } catch {
    return true;
  }
}

/**
 * Timing-safe HMAC-SHA256 verification for cloud GPU worker completion callbacks.
 */
export function verifyGpuCallbackSignature(
  rawBody: string,
  signatureHeader: string | undefined | null,
  secret: string | undefined | null
): boolean {
  if (!secret || secret.trim().length < 12 || !signatureHeader) return false;
  const cleaned = signatureHeader.trim().replace(/^sha256=/i, '');
  if (!/^[a-f0-9]{64}$/i.test(cleaned)) return false;
  const expected = crypto
    .createHmac('sha256', secret.trim())
    .update(rawBody, 'utf8')
    .digest('hex');
  const a = Buffer.from(cleaned.toLowerCase(), 'hex');
  const b = Buffer.from(expected, 'hex');
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

export const GPU_PROVIDER_PRESETS = [
  {
    id: 'local_cuda',
    nameZh: '本地工作站 (NVIDIA CUDA)',
    nameEn: 'Local Workstation (NVIDIA CUDA)',
    gpuTier: 'RTX 4070 Ti Super (16 GB) / RTX 4090 (24 GB) / RTX 5090 (32 GB)',
    billingZh: '一次性硬件投入，数据零出网（科研合规首选）',
    billingEn: 'One-time hardware, air-gapped lab data privacy',
    coldStartSec: 0,
    maxTilePx: 1024,
    recommendedForZh: '高频常规微调、高隐私病理/晶圆数据、3D Z-Stack 批处理',
    recommendedForEn: 'Daily fine-tuning, sensitive pathology/wafer images, 3D stacks',
  },
  {
    id: 'modal_serverless',
    nameZh: 'Modal Serverless GPU',
    nameEn: 'Modal Serverless GPU',
    gpuTier: 'NVIDIA L4 (24 GB) / A10G (24 GB) / A100-40GB',
    billingZh: '按秒计费（空闲零费用），无常驻机器运维成本',
    billingEn: 'Per-second billing ($0 idle), zero cluster ops',
    coldStartSec: 8,
    maxTilePx: 1024,
    recommendedForZh: '中小团队弹性少样本 LoRA 微调（强烈推荐）',
    recommendedForEn: 'Elastic few-shot LoRA jobs for small/mid teams (Recommended)',
  },
  {
    id: 'runpod_serverless',
    nameZh: 'RunPod Serverless / Pod',
    nameEn: 'RunPod Serverless / Pod',
    gpuTier: 'RTX 4090 (24 GB) / L40S (48 GB)',
    billingZh: '高性价比按秒/按小时计费，容器隔离启动快',
    billingEn: 'Cost-effective per-second/hourly isolated containers',
    coldStartSec: 12,
    maxTilePx: 1024,
    recommendedForZh: '大批量 SEM / 3D 体积微调，追求性价比',
    recommendedForEn: 'Cost-sensitive high-throughput SEM & 3D training',
  },
  {
    id: 'autodl_cn',
    nameZh: 'AutoDL / 算力云 (中国大陆低延迟)',
    nameEn: 'AutoDL (Mainland China Low-Latency)',
    gpuTier: 'RTX 4090 (24 GB) / A800 (80 GB)',
    billingZh: '人民币按量计费，国内镜像加速与低延迟传输',
    billingEn: 'CNY pay-as-you-go with domestic mirror & low latency',
    coldStartSec: 15,
    maxTilePx: 1024,
    recommendedForZh: '中国大陆高校与企业实验室在线微调',
    recommendedForEn: 'Mainland China university & industrial labs',
  },
  {
    id: 'gcp_vertex',
    nameZh: 'Google Cloud Vertex AI / Cloud Run GPU',
    nameEn: 'Google Cloud Vertex AI / Cloud Run GPU',
    gpuTier: 'NVIDIA L4 (24 GB) / A100-80GB',
    billingZh: '企业级 IAM / VPC-SC 隔离与合规审计',
    billingEn: 'Enterprise IAM, VPC-SC isolation & audit trail',
    coldStartSec: 18,
    maxTilePx: 1024,
    recommendedForZh: '需企业合规认证与全栈 GCP 集成的生产环境',
    recommendedForEn: 'Enterprise production requiring IAM & compliance',
  },
  {
    id: 'aws_sagemaker',
    nameZh: 'AWS SageMaker Training Jobs',
    nameEn: 'AWS SageMaker Training Jobs',
    gpuTier: 'ml.g5.xlarge (A10G 24 GB) / ml.p4d (A100)',
    billingZh: '按作业秒级计费，S3 + KMS 加密存储',
    billingEn: 'Per-job billing with S3 + KMS encryption',
    coldStartSec: 25,
    maxTilePx: 1024,
    recommendedForZh: '已有 AWS 数据湖的企业级部署',
    recommendedForEn: 'Enterprise teams already running on AWS',
  },
] as const;

export const ALLOWED_PROVIDERS = new Set<string>(GPU_PROVIDER_PRESETS.map((p) => p.id));
export const ALLOWED_BACKBONES = new Set([
  'dinov2_vit_s14',
  'convnext_tiny',
  'sam_vit_b_lora',
  'unet_resnet34',
]);
export const ALLOWED_LORA_RANKS = new Set([4, 8, 16]);
export const ALLOWED_LOSSES = new Set(['focal_dice', 'ce_dice', 'tversky']);
export const ALLOWED_PRECISIONS = new Set(['bf16', 'fp16']);
export const FORBIDDEN_CODE_FIELDS = [
  'code',
  'script',
  'command',
  'cmd',
  'shell',
  'bash',
  'python',
  'gitUrl',
  'repoUrl',
  'dockerImage',
  'pipInstall',
  'packages',
  'entrypoint',
  'eval',
  'exec',
  'webhook',
  'url',
];

export interface TrainJobAuditRecord {
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

export interface ValidatedTrainJobResult {
  ok: true;
  job: TrainJobAuditRecord;
  classNames: string[];
  manifest: Record<string, unknown>;
}

export interface RejectedTrainJobResult {
  ok: false;
  statusCode: number;
  error: string;
}

export function validateDeclarativeTrainJob(
  body: Record<string, unknown>,
  maxWallClockCap = 900
): ValidatedTrainJobResult | RejectedTrainJobResult {
  // Defense 1: Strictly reject any arbitrary code, command, git repo, or container override fields
  for (const forbidden of FORBIDDEN_CODE_FIELDS) {
    if (body[forbidden] !== undefined) {
      return {
        ok: false,
        statusCode: 400,
        error: `Security Gatekeeper blocked forbidden field "${forbidden}". Vexel GPU training only accepts declarative vision allowlist parameters.`,
      };
    }
  }

  const provider = String(body.provider ?? 'local_cuda');
  const backbone = String(body.backbone ?? 'dinov2_vit_s14');
  const loraRank = Number(body.loraRank ?? 8);
  const epochs = Math.round(Number(body.epochs ?? 25));
  const loss = String(body.loss ?? 'focal_dice');
  const precision = String(body.precision ?? 'bf16');
  const requestedWallSec = Math.round(Number(body.maxWallClockSec ?? 360));
  const maxWallClockSec = Math.max(60, Math.min(maxWallClockCap, requestedWallSec));

  const modelId = String(body.modelId ?? 'vexel_lora_seg')
    .replace(/[^a-zA-Z0-9._-]/g, '_')
    .slice(0, 48);
  const version = String(body.version ?? '2.0')
    .replace(/[^a-zA-Z0-9._-]/g, '')
    .slice(0, 16);
  const specimenName = String(body.specimenName ?? 'specimen').slice(0, 96);
  const imageSha256 = String(body.imageSha256 ?? '')
    .replace(/[^a-fA-F0-9]/g, '')
    .slice(0, 64);
  const imageWidth = Math.round(Number(body.imageWidth ?? 0));
  const imageHeight = Math.round(Number(body.imageHeight ?? 0));
  const labeledClasses = Math.round(Number(body.labeledClasses ?? 0));
  const labeledPixels = Math.round(Number(body.labeledPixels ?? 0));
  const classNames = Array.isArray(body.classNames)
    ? body.classNames
        .filter((c): c is string => typeof c === 'string' && c.trim().length > 0)
        .map((c) => c.trim().slice(0, 32))
        .slice(0, 8)
    : ['Foreground', 'Background'];

  if (!ALLOWED_PROVIDERS.has(provider)) {
    return { ok: false, statusCode: 400, error: `Unsupported GPU provider "${provider}".` };
  }
  if (!ALLOWED_BACKBONES.has(backbone)) {
    return {
      ok: false,
      statusCode: 400,
      error: `Backbone "${backbone}" is not in Vexel vision allowlist.`,
    };
  }
  if (!ALLOWED_LORA_RANKS.has(loraRank)) {
    return { ok: false, statusCode: 400, error: `LoRA rank must be one of [4, 8, 16].` };
  }
  if (!Number.isFinite(epochs) || epochs < 5 || epochs > 80) {
    return { ok: false, statusCode: 400, error: `Epochs must be in safe range [5, 80].` };
  }
  if (!ALLOWED_LOSSES.has(loss)) {
    return { ok: false, statusCode: 400, error: `Unsupported loss "${loss}".` };
  }
  if (!ALLOWED_PRECISIONS.has(precision)) {
    return { ok: false, statusCode: 400, error: `Precision must be fp16 or bf16.` };
  }
  if (imageWidth < 64 || imageHeight < 64 || imageWidth > 8192 || imageHeight > 8192) {
    return {
      ok: false,
      statusCode: 400,
      error: `Image dimensions (${imageWidth}×${imageHeight}) outside valid microscopy range [64..8192].`,
    };
  }
  if (labeledClasses < 2 || labeledPixels < 64) {
    return {
      ok: false,
      statusCode: 400,
      error: `Pre-flight data gate failed: at least 2 labeled classes and ≥64 annotated pixels are required before allocating GPU compute.`,
    };
  }

  const jobId = `vxtrain_${Date.now().toString(36)}_${crypto.randomBytes(3).toString('hex')}`;
  const sandboxPolicy = [
    'allowlist_only_declarative_params',
    'zero_arbitrary_code_or_shell',
    'network_egress_none',
    `wall_clock_cap_${maxWallClockSec}s`,
    'onnx_artifact_only_max_50mb',
    'vision_tensor_schema_enforced',
  ];

  const manifestObj: Record<string, unknown> = {
    schema: 'vexel.gpu_lora_job.v1',
    jobId,
    createdAt: new Date().toISOString(),
    provider,
    model: {
      modelId,
      version,
      backbone,
      loraRank,
      epochs,
      loss,
      precision,
      classNames,
    },
    dataset: {
      specimenName,
      imageSha256: imageSha256 || 'unverified',
      width: imageWidth,
      height: imageHeight,
      labeledClasses,
      labeledPixels,
    },
    sandboxPolicy,
    maxWallClockSec,
  };

  const manifestDigest = crypto
    .createHash('sha256')
    .update(JSON.stringify(manifestObj))
    .digest('hex');

  const job: TrainJobAuditRecord = {
    jobId,
    createdAt: String(manifestObj.createdAt),
    provider,
    backbone,
    loraRank,
    epochs,
    loss,
    precision,
    maxWallClockSec,
    modelId,
    version,
    specimenName,
    imageSha256: imageSha256 || 'unverified',
    imageWidth,
    imageHeight,
    labeledClasses,
    labeledPixels,
    status: 'manifest_ready',
    sandboxPolicy,
    manifestDigest,
    message:
      'Signed sandbox training manifest & read-only LoRA script generated. Run locally or connect VEXEL_GPU_TRAIN_ENDPOINT for managed cloud dispatch.',
  };

  return { ok: true, job, classNames, manifest: manifestObj };
}

/** A URL a user may point the server at (their local Python engine or Ollama): this computer, or
 * a public host - never another machine on the private network nor a cloud metadata address. */
export function isAllowedEngineUrl(urlStr: string): boolean {
  let host = '';
  try {
    const u = new URL(urlStr);
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return false;
    host = u.hostname.toLowerCase().replace(/^\[|\]$/g, '');
  } catch {
    return false;
  }
  if (host === 'localhost' || host === '127.0.0.1' || host === '::1') return true;
  return !isBlockedPrivateOrMetadataHost(urlStr);
}
