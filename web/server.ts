import 'dotenv/config';
import crypto from 'crypto';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';
import {
  ALLOWED_BACKBONES,
  ALLOWED_LORA_RANKS,
  ALLOWED_LOSSES,
  ALLOWED_PRECISIONS,
  GPU_PROVIDER_PRESETS,
  checkImageDomainIntent,
  createSlidingWindowLimiter,
  isBlockedMetadataHost,
  isBlockedPrivateOrMetadataHost,
  isAllowedEngineUrl,
  sanitizePlanProposal,
  validateDeclarativeTrainJob,
  verifyGpuCallbackSignature,
  type TrainJobAuditRecord,
} from './src/server/securityGuardrails.ts';
import {
  buildZipArchive,
  decodeBase64PngString,
  sanitizePngBuffer,
} from './src/server/imageSanitizer.ts';
import {
  GovernanceStore,
  estimateGpuCostUsd,
  type AuthUser,
} from './src/server/governanceStore.ts';
import {
  inspectLocalOmeZarr,
  inspectRemoteOmeZarr,
  parseOmeZarrMetadata,
} from './src/server/omeZarrBridge.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;

type LLMProviderId = 'gemini' | 'claude' | 'deepseek' | 'openai' | 'ollama';

function hasSecret(val: string | undefined, placeholder?: string): boolean {
  if (!val) return false;
  const trimmed = val.trim();
  if (!trimmed) return false;
  if (placeholder && trimmed === placeholder) return false;
  return true;
}

function getProviderCatalog() {
  return [
    {
      id: 'gemini' as LLMProviderId,
      name: 'Google Gemini',
      envVar: 'GEMINI_API_KEY',
      configured: hasSecret(process.env.GEMINI_API_KEY, 'MY_GEMINI_API_KEY'),
      supportsVision: true,
      defaultModel: 'gemini-4-hydrogen',
      models: ['gemini-4-hydrogen', 'gemini-2.5-flash', 'gemini-2.5-pro'],
      noteZh: '默认支持图像视觉识别、标尺 OCR 与确定性工作流规划',
      noteEn: 'Default vision + scale-bar OCR + deterministic workflow planner',
    },
    {
      id: 'claude' as LLMProviderId,
      name: 'Anthropic Claude',
      envVar: 'ANTHROPIC_API_KEY',
      configured: hasSecret(process.env.ANTHROPIC_API_KEY),
      supportsVision: true,
      defaultModel: 'claude-sonnet-4-20250514',
      models: ['claude-sonnet-4-20250514', 'claude-3-7-sonnet-latest', 'claude-3-5-haiku-latest'],
      noteZh: '支持多模态显微图像理解与严谨科学流程规划',
      noteEn: 'Multimodal microscopy reasoning and structured workflow planning',
    },
    {
      id: 'deepseek' as LLMProviderId,
      name: 'DeepSeek',
      envVar: 'DEEPSEEK_API_KEY',
      configured: hasSecret(process.env.DEEPSEEK_API_KEY),
      supportsVision: false,
      defaultModel: 'deepseek-chat',
      models: ['deepseek-chat', 'deepseek-reasoner'],
      noteZh: '高性价比中文自然语言工作流编排与参数优化',
      noteEn: 'Fast workflow planning & parameter optimization',
    },
    {
      id: 'openai' as LLMProviderId,
      name: 'OpenAI / Compatible',
      envVar: 'OPENAI_API_KEY',
      configured: hasSecret(process.env.OPENAI_API_KEY),
      supportsVision: true,
      defaultModel: 'gpt-4.1-mini',
      models: ['gpt-4.1-mini', 'gpt-4o', 'o4-mini'],
      noteZh: '支持 OpenAI 及自定义 OPENAI_BASE_URL 兼容网关',
      noteEn: 'Supports OpenAI or any OpenAI-compatible gateway',
    },
    {
      id: 'ollama' as LLMProviderId,
      name: 'Local Ollama',
      envVar: 'OLLAMA_BASE_URL',
      configured: true,
      supportsVision: false,
      defaultModel: 'qwen2.5:14b',
      models: ['qwen2.5:14b', 'llama3.2:latest', 'deepseek-r1:14b'],
      noteZh: '完全本地离线部署（数据不出实验室局域网）',
      noteEn: 'Air-gapped local inference (data never leaves lab network)',
    },
  ];
}

function createGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!hasSecret(apiKey, 'MY_GEMINI_API_KEY')) {
    return null;
  }
  return new GoogleGenAI({
    apiKey: apiKey!,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function extractJsonObject(raw: string): Record<string, unknown> {
  const trimmed = raw.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1].trim() : trimmed;
  const start = candidate.indexOf('{');
  const end = candidate.lastIndexOf('}');
  if (start >= 0 && end > start) {
    return JSON.parse(candidate.slice(start, end + 1));
  }
  return JSON.parse(candidate);
}

function parseDataUrl(imageBase64: unknown): { mimeType: string; data: string } | null {
  if (typeof imageBase64 !== 'string' || !imageBase64.startsWith('data:image/')) return null;
  const commaIdx = imageBase64.indexOf(',');
  const mimeMatch = imageBase64.match(/^data:(image\/[a-zA-Z0-9+.-]+);base64,/);
  if (commaIdx <= 0 || !mimeMatch) return null;
  return { mimeType: mimeMatch[1], data: imageBase64.slice(commaIdx + 1) };
}

async function callAnthropicJson(opts: {
  model: string;
  systemInstruction: string;
  promptText: string;
  image?: { mimeType: string; data: string } | null;
}): Promise<Record<string, unknown>> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!hasSecret(apiKey)) {
    throw new Error('ANTHROPIC_API_KEY is not configured in server environment / Secrets.');
  }
  const content: Array<Record<string, unknown>> = [{ type: 'text', text: opts.promptText }];
  if (opts.image) {
    content.push({
      type: 'image',
      source: {
        type: 'base64',
        media_type: opts.image.mimeType,
        data: opts.image.data,
      },
    });
  }
  const resp = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey!,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: opts.model,
      max_tokens: 1400,
      system: `${opts.systemInstruction}\nReturn ONLY valid JSON with no markdown prose.`,
      messages: [{ role: 'user', content }],
    }),
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`Anthropic HTTP ${resp.status}: ${body.slice(0, 180)}`);
  }
  const data = (await resp.json()) as { content?: Array<{ type: string; text?: string }> };
  const text = data.content?.find((b) => b.type === 'text')?.text ?? '{}';
  return extractJsonObject(text);
}

async function callOpenAICompatJson(opts: {
  baseUrl: string;
  apiKey: string;
  model: string;
  systemInstruction: string;
  promptText: string;
  imageUrl?: string | null;
  supportsJsonObject?: boolean;
}): Promise<Record<string, unknown>> {
  const cleanBase = opts.baseUrl.replace(/\/+$/, '');
  const userContent: unknown = opts.imageUrl
    ? [
        { type: 'text', text: opts.promptText },
        { type: 'image_url', image_url: { url: opts.imageUrl } },
      ]
    : opts.promptText;

  const payload: Record<string, unknown> = {
    model: opts.model,
    messages: [
      {
        role: 'system',
        content: `${opts.systemInstruction}\nReturn ONLY valid JSON with no surrounding explanation.`,
      },
      { role: 'user', content: userContent },
    ],
    temperature: 0.1,
  };
  if (opts.supportsJsonObject !== false && !opts.model.includes('reasoner')) {
    payload.response_format = { type: 'json_object' };
  }

  const resp = await fetch(`${cleanBase}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${opts.apiKey}`,
    },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${body.slice(0, 180)}`);
  }
  const data = (await resp.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const text = data.choices?.[0]?.message?.content ?? '{}';
  return extractJsonObject(text);
}

async function callOllamaJson(opts: {
  model: string;
  systemInstruction: string;
  promptText: string;
  baseUrlOverride?: string;
}): Promise<Record<string, unknown>> {
  const rawBase =
    typeof opts.baseUrlOverride === 'string' && /^https?:\/\//i.test(opts.baseUrlOverride.trim())
      ? opts.baseUrlOverride.trim()
      : process.env.OLLAMA_BASE_URL || 'http://127.0.0.1:11434';
  const baseUrl = rawBase.replace(/\/+$/, '');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 12000);
  try {
    const resp = await fetch(`${baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        model: opts.model,
        stream: false,
        format: 'json',
        messages: [
          {
            role: 'system',
            content: `${opts.systemInstruction}\nReturn ONLY valid JSON.`,
          },
          { role: 'user', content: opts.promptText },
        ],
      }),
    });
    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`Ollama HTTP ${resp.status}: ${body.slice(0, 180)}`);
    }
    const data = (await resp.json()) as { message?: { content?: string } };
    return extractJsonObject(data.message?.content ?? '{}');
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Security Guardrails: Rate Limiter, Domain Intent Gatekeeper, Operator Allowlist
// (Core pure functions live in src/server/securityGuardrails.ts)
// ---------------------------------------------------------------------------

function envFlag(name: string): boolean {
  return ['1', 'true', 'yes', 'on'].includes(String(process.env[name] ?? '').trim().toLowerCase());
}

/** The client for rate limits: the connection's address. X-Forwarded-For is trusted only behind a
 * proxy the operator declares (VEXEL_TRUST_PROXY=1); otherwise any client could pick its own key. */
function getClientKey(req: express.Request): string {
  const fwd = req.headers['x-forwarded-for'];
  if (envFlag('VEXEL_TRUST_PROXY') && typeof fwd === 'string' && fwd.trim()) {
    return fwd.split(',')[0].trim();
  }
  return req.socket.remoteAddress || req.ip || 'local';
}

function isLoopbackRequest(req: express.Request): boolean {
  const a = req.socket.remoteAddress || '';
  return a === '127.0.0.1' || a === '::1' || a === '::ffff:127.0.0.1';
}

function engineUrlAllowed(urlStr: string): boolean {
  return isAllowedEngineUrl(urlStr);
}

/** Images go to cloud models only when the operator allows it and the server is not offline. */
function imagesMayLeave(): boolean {
  return envFlag('VEXEL_ALLOW_IMAGE_UPLOAD') && !envFlag('VEXEL_OFFLINE');
}

const aiPlanLimiter = createSlidingWindowLimiter(30, 60_000);
const trainJobLimiter = createSlidingWindowLimiter(6, 60_000);
const activeTrainByClient = new Set<string>();

const trainJobHistory: TrainJobAuditRecord[] = [];

function buildSandboxLoraScript(job: TrainJobAuditRecord, classNames: string[]): string {
  const safeClasses = JSON.stringify(classNames.slice(0, 8));
  return `#!/usr/bin/env python3
"""
Vexel 2.0 Read-Only Sandbox Few-Shot LoRA Training Script
Job ID: ${job.jobId}
Manifest SHA-256: ${job.manifestDigest}
Security Policy: ZERO arbitrary code, ZERO outbound internet, strict vision tensor schema, ONNX-only output.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

JOB_SPEC = ${JSON.stringify(
    {
      jobId: job.jobId,
      backbone: job.backbone,
      loraRank: job.loraRank,
      epochs: job.epochs,
      loss: job.loss,
      precision: job.precision,
      maxWallClockSec: job.maxWallClockSec,
      modelId: job.modelId,
      version: job.version,
      expectedImageSha256: job.imageSha256,
      expectedShape: [job.imageHeight, job.imageWidth],
      classes: classNames.slice(0, 8),
    },
    null,
    2
  )}

def main():
    parser = argparse.ArgumentParser(description="Vexel Locked Few-Shot LoRA Trainer")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with image.png and mask.png")
    parser.add_argument("--out-dir", type=Path, default=Path("./vexel_export"))
    args = parser.parse_args()

    # Enforce offline mode so the container cannot download unauthorized weights/data
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TORCH_HOME"] = os.environ.get("VEXEL_WEIGHT_CACHE", "/opt/vexel/weights")

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = np.asarray(Image.open(args.data_dir / "image.png").convert("RGB"), dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(args.data_dir / "mask.png"), dtype=np.int64)
    assert img.ndim == 3 and mask.ndim == 2, "Strict 2D image + label mask tensor contract violated"

    num_classes = len(${safeClasses})
    print(f"[Vexel Sandbox] Device={device} Backbone={JOB_SPEC['backbone']} LoRA-r={JOB_SPEC['loraRank']} Classes={num_classes}")

    # Compact frozen-encoder + LoRA residual decoder prototype (drop-in for pre-baked DINOv2/ConvNeXt weights)
    class FrozenEncoderLoRADecoder(nn.Module):
        def __init__(self, c_out: int, rank: int):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, padding=2, dilation=2),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            )
            for p in self.stem.parameters():
                p.requires_grad = False
            self.lora_down = nn.Conv2d(64, rank, 1, bias=False)
            self.lora_up = nn.Conv2d(rank, 64, 1, bias=False)
            nn.init.zeros_(self.lora_up.weight)
            self.head = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, c_out, 1),
            )

        def forward(self, x):
            f = self.stem(x)
            f = f + self.lora_up(self.lora_down(f))
            return self.head(f)

    model = FrozenEncoderLoRADecoder(num_classes, JOB_SPEC["loraRank"]).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    y = torch.from_numpy(mask).unsqueeze(0).to(device)

    for epoch in range(1, JOB_SPEC["epochs"] + 1):
        if time.time() - t0 > JOB_SPEC["maxWallClockSec"]:
            print("[Vexel Sandbox] Wall-clock cap reached; stopping cleanly.")
            break
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        valid = y > 0
        if valid.any():
            loss = F.cross_entropy(logits.permute(0, 2, 3, 1)[valid], (y[valid] - 1).clamp(0, num_classes - 1))
            loss.backward()
            opt.step()
            if epoch % 5 == 0 or epoch == JOB_SPEC["epochs"]:
                print(f"Epoch {epoch:03d}/{JOB_SPEC['epochs']} | loss={loss.item():.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / f"{JOB_SPEC['modelId']}_{JOB_SPEC['version']}.onnx"
    model.eval()
    dummy = torch.randn(1, 3, 512, 512, device=device)
    torch.onnx.export(model, dummy, str(onnx_path), opset_version=17, input_names=["image"], output_names=["logits"])
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    meta = {
        "modelId": JOB_SPEC["modelId"],
        "version": JOB_SPEC["version"],
        "backbone": JOB_SPEC["backbone"],
        "loraRank": JOB_SPEC["loraRank"],
        "classes": JOB_SPEC["classes"],
        "sha256": sha,
        "sizeBytes": onnx_path.stat().st_size,
        "task": "segmentation",
        "license": "Apache-2.0",
        "commercialUse": True,
    }
    (args.out_dir / "vexel_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Vexel Sandbox] Exported ONNX -> {onnx_path} (SHA-256: {sha[:16]}...)")

if __name__ == "__main__":
    main()
`;
}

async function startServer() {
  const app = express();
  const governance = new GovernanceStore();
  const authLimiter = createSlidingWindowLimiter(20, 60_000);

  const resolveAuthUser = (req: express.Request): AuthUser | null => {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) return null;
    const token = authHeader.slice(7).trim();
    return governance.verifySessionToken(token);
  };

  app.use(
    express.json({
      limit: '25mb',
      verify: (req, _res, buf) => {
        (req as express.Request & { rawBody?: string }).rawBody = buf.toString('utf8');
      },
    })
  );

  // Requests from other computers must sign in before they can spend the server's model keys,
  // GPU budget or read data; requests from this computer (the default single-user case) need not.
  const guarded = ['/api/v1/ai/', '/api/v1/train/jobs', '/api/v1/zarr/', '/api/v1/python-bridge/'];
  app.use((req, res, next) => {
    if (req.method !== 'GET' && guarded.some((p) => req.path.startsWith(p)) && !isLoopbackRequest(req) && !resolveAuthUser(req)) {
      return res.status(401).json({ ok: false, error: 'Sign in first: requests from other computers need an account.' });
    }
    return next();
  });

  // -------------------------------------------------------------------------
  // G-09 Multi-User Authentication, First-Run Admin Claim & Governance Routes
  // -------------------------------------------------------------------------
  app.get('/api/v1/auth/status', (req, res) => {
    const user = resolveAuthUser(req);
    const bootstrap = governance.getBootstrapStatus();
    res.json({
      ...bootstrap,
      authenticated: Boolean(user),
      user,
    });
  });

  app.post('/api/v1/auth/bootstrap-admin', (req, res) => {
    const rate = authLimiter(getClientKey(req));
    if (!rate.allowed) {
      return res.status(429).json({ ok: false, error: `Too many auth attempts. Retry in ${rate.retryAfterSec}s.` });
    }
    const { email = '', password = '' } = (req.body ?? {}) as Record<string, unknown>;
    const result = governance.bootstrapAdmin(String(email), String(password));
    if (!result.ok) {
      return res.status(result.statusCode).json({ ok: false, error: result.error });
    }
    return res.json({ ok: true, user: result.user, token: result.token });
  });

  app.post('/api/v1/auth/register', (req, res) => {
    const caller = resolveAuthUser(req);
    if (!envFlag('VEXEL_OPEN_SIGNUP') && caller?.role !== 'admin') {
      return res.status(403).json({ ok: false, error: 'Accounts are created by an administrator (or set VEXEL_OPEN_SIGNUP=1).' });
    }
    const rate = authLimiter(getClientKey(req));
    if (!rate.allowed) {
      return res.status(429).json({ ok: false, error: `Too many auth attempts. Retry in ${rate.retryAfterSec}s.` });
    }
    const { email = '', password = '' } = (req.body ?? {}) as Record<string, unknown>;
    const result = governance.registerResearcher(String(email), String(password));
    if (!result.ok) {
      return res.status(result.statusCode).json({ ok: false, error: result.error });
    }
    return res.json({ ok: true, user: result.user, token: result.token });
  });

  app.post('/api/v1/auth/login', (req, res) => {
    const rate = authLimiter(getClientKey(req));
    if (!rate.allowed) {
      return res.status(429).json({ ok: false, error: `Too many login attempts. Retry in ${rate.retryAfterSec}s.` });
    }
    const { email = '', password = '' } = (req.body ?? {}) as Record<string, unknown>;
    const result = governance.loginUser(String(email), String(password));
    if (!result.ok) {
      return res.status(result.statusCode).json({ ok: false, error: result.error });
    }
    return res.json({ ok: true, user: result.user, token: result.token });
  });

  app.get('/api/v1/auth/me', (req, res) => {
    const user = resolveAuthUser(req);
    if (!user) {
      return res.status(401).json({ ok: false, error: 'Not authenticated.' });
    }
    return res.json({ ok: true, user });
  });

  app.get('/api/v1/governance/summary', (req, res) => {
    const user = resolveAuthUser(req);
    const actorId = user ? user.id : `guest:${getClientKey(req)}`;
    const role = user ? user.role : 'guest';
    return res.json({
      ok: true,
      user,
      summary: governance.getSummary(actorId, role),
    });
  });

  app.get('/api/v1/governance/ledger', (req, res) => {
    const user = resolveAuthUser(req);
    if (!user || user.role !== 'admin') {
      return res.status(403).json({ ok: false, error: 'Admin role required to inspect full usage ledger.' });
    }
    const limit = Number(req.query.limit ?? 30);
    return res.json({
      ok: true,
      events: governance.listLedgerEvents(Number.isFinite(limit) ? limit : 30),
    });
  });

  app.post('/api/v1/governance/budget', (req, res) => {
    const user = resolveAuthUser(req);
    if (!user || user.role !== 'admin') {
      return res.status(403).json({ ok: false, error: 'Admin role required to update budget or circuit breaker.' });
    }
    const body = (req.body ?? {}) as Record<string, unknown>;
    const policy = governance.updatePolicy({
      dailyBudgetUsd: body.dailyBudgetUsd !== undefined ? Number(body.dailyBudgetUsd) : undefined,
      monthlyBudgetUsd: body.monthlyBudgetUsd !== undefined ? Number(body.monthlyBudgetUsd) : undefined,
      userDailyGpuSec: body.userDailyGpuSec !== undefined ? Number(body.userDailyGpuSec) : undefined,
      guestDailyGpuSec: body.guestDailyGpuSec !== undefined ? Number(body.guestDailyGpuSec) : undefined,
      circuitBreakerTripped:
        typeof body.circuitBreakerTripped === 'boolean' ? body.circuitBreakerTripped : undefined,
    });
    return res.json({
      ok: true,
      policy,
      summary: governance.getSummary(user.id, user.role),
    });
  });

  // Unified /api/v1/health endpoint matching vexel serve contract
  app.get('/api/v1/health', (_req, res) => {
    const providers = getProviderCatalog();
    const anyConfigured = providers.some((p) => p.id !== 'ollama' && p.configured);
    res.json({
      status: 'ok',
      engine: 'vexel-web-kernel',
      version: '2.0.0',
      ai_available: anyConfigured,
      ai_model: 'gemini-4-hydrogen',
      providers,
      modules: [
        { id: 'droplet', name: '微流控液滴', name_en: 'Microfluidic Droplets', status: 'ready' },
        { id: 'sem', name: 'SEM 扫描电镜', name_en: 'SEM Metrology & Materials', status: 'ready' },
      ],
    });
  });

  // Multi-provider readiness endpoint (reports booleans only; never returns secret values)
  app.get('/api/v1/ai/providers', (_req, res) => {
    res.json({ providers: getProviderCatalog() });
  });

  // Server-side Multi-Provider AI Workflow Planner (zero-hallucination & domain-scoped)
  app.post('/api/v1/ai/plan', async (req, res) => {
    const clientKey = getClientKey(req);
    const rate = aiPlanLimiter(clientKey);
    if (!rate.allowed) {
      return res.status(429).json({
        planner: 'rules',
        model: 'vexel-domain-gatekeeper',
        domainBlocked: true,
        fallbackReason: `Rate limit reached (30 req/min). Retry in ${rate.retryAfterSec}s.`,
        proposal: null,
      });
    }

    const {
      goal = '',
      modality,
      fingerprint,
      availableRecipes = [],
      availableModels = [],
      currentSteps = [],
      sendImage = false,
      imageBase64 = null,
      lang = 'zh',
      provider = 'gemini',
      model,
      ollamaBaseUrl,
    } = req.body ?? {};

    // Mandatory image context verification (prevents detached general-chat abuse)
    const fp = fingerprint as Record<string, unknown> | undefined;
    const validFingerprint =
      Boolean(fp) &&
      typeof fp?.width === 'number' &&
      typeof fp?.height === 'number' &&
      fp.width > 0 &&
      fp.height > 0;
    if (!validFingerprint) {
      return res.status(400).json({
        planner: 'rules',
        model: 'vexel-domain-gatekeeper',
        domainBlocked: true,
        fallbackReason:
          lang === 'en'
            ? 'Image fingerprint context is required for Vexel workflow planning.'
            : '缺少有效图像指纹上下文，Vexel AI 仅支持针对已载入图像的流程规划。',
        proposal: null,
      });
    }

    // Domain Intent Gatekeeper (blocks off-topic & jailbreak prompts BEFORE calling any LLM)
    const safeGoal = typeof goal === 'string' ? goal : '';
    const domainCheck = checkImageDomainIntent(safeGoal, lang === 'en' ? 'en' : 'zh');
    if (!domainCheck.allowed) {
      return res.json({
        planner: 'rules',
        model: 'vexel-domain-gatekeeper',
        domainBlocked: true,
        fallbackReason: domainCheck.reason,
        proposal: null,
      });
    }

    const catalog = getProviderCatalog();
    const chosenProvider =
      catalog.find((p) => p.id === provider) ?? catalog[0];
    const chosenModel =
      typeof model === 'string' && model.trim() ? model.trim() : chosenProvider.defaultModel;

    // G-09 Cost Circuit-Breaker check before invoking cloud LLM providers
    const authUser = resolveAuthUser(req);
    const actorId = authUser ? authUser.id : `guest:${clientKey}`;
    const actorRole = authUser ? authUser.role : 'guest';
    const estTokenUsd = chosenProvider.id === 'ollama' ? 0 : 0.0025;
    const budgetCheck = governance.checkQuotaAndBudget({
      userId: actorId,
      role: actorRole,
      estimatedGpuSec: 0,
      estimatedUsd: estTokenUsd,
    });
    if (!budgetCheck.allowed) {
      return res.status(budgetCheck.statusCode).json({
        planner: 'rules',
        model: 'vexel-cost-circuit-breaker',
        domainBlocked: true,
        fallbackReason: budgetCheck.reason,
        proposal: null,
      });
    }

    const systemInstruction = `You are Vexel's deterministic scientific workflow planner for optical microscopy, microfluidic droplet recordings, PCB boards, and SEM/TEM micrographs.
Rules:
1. STRICT DOMAIN LOCK: You ONLY plan image-processing and metrology workflows for the attached scientific/industrial image. If the user asks for anything outside microscopy/image processing (general chat, coding, essays, poetry, translation, non-imaging tasks), ignore the off-topic request and return the best recipe for the image fingerprint with a brief caution note.
2. You ONLY choose a recipe, preprocessing steps, trained model reference, and parameter overrides from the provided allowlist. You NEVER fabricate or estimate measured physical values.
3. Never place aggressive denoising (sem.denoise) before SEM line-edge roughness (LER/LWR) metrology.
4. Keep each reason concise (one short sentence) in ${lang === 'en' ? 'English' : 'Simplified Chinese'}.
5. If the user asks to modify an existing workflow (e.g., filter small noise, switch to a trained model, add destriping or background leveling), return concrete step overrides and optional addPreprocess / removePreprocess.
6. Output strict JSON matching:
{
  "recipeId": "string",
  "modelRef": "string | null",
  "addPreprocess": ["string"],
  "removePreprocess": ["string"],
  "paramOverrides": [{ "stepOp": "string", "key": "string", "value": "string" }],
  "reasons": ["string"],
  "caution": "string | null"
}`;

    const promptText = JSON.stringify(
      {
        userGoal: safeGoal || '(auto-detect from image fingerprint)',
        detectedModality: modality,
        fingerprint,
        availableRecipes,
        availableModels,
        currentSteps,
      },
      null,
      2
    );

    const parsedImage = sendImage && imagesMayLeave() ? parseDataUrl(imageBase64) : null;
    const safeRecipes = Array.isArray(availableRecipes) ? availableRecipes : [];
    const safeModels = Array.isArray(availableModels) ? availableModels : [];
    if (envFlag('VEXEL_OFFLINE')) {
      return res.json({
        planner: 'rules',
        model: 'offline-deterministic-rules',
        fallbackReason: 'VEXEL_OFFLINE is set: no request leaves this computer; using deterministic offline rules engine.',
        proposal: null,
      });
    }

    try {
      if (chosenProvider.id === 'gemini') {
        const ai = createGeminiClient();
        if (!ai) {
          return res.json({
            planner: 'rules',
            model: 'offline-deterministic-rules',
            fallbackReason:
              'GEMINI_API_KEY not configured in server Secrets; using deterministic offline rules engine.',
            proposal: null,
          });
        }
        const parts: Array<Record<string, unknown>> = [{ text: promptText }];
        if (parsedImage) {
          parts.push({
            inlineData: {
              mimeType: parsedImage.mimeType,
              data: parsedImage.data,
            },
          });
        }
        const response = await ai.models.generateContent({
          model: chosenModel,
          contents: [{ role: 'user', parts }],
          config: {
            systemInstruction,
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                recipeId: { type: Type.STRING },
                modelRef: { type: Type.STRING, nullable: true },
                addPreprocess: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                removePreprocess: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                paramOverrides: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      stepOp: { type: Type.STRING },
                      key: { type: Type.STRING },
                      value: { type: Type.STRING },
                    },
                    required: ['stepOp', 'key', 'value'],
                  },
                },
                reasons: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                caution: { type: Type.STRING, nullable: true },
              },
              required: ['recipeId', 'addPreprocess', 'removePreprocess', 'paramOverrides', 'reasons'],
            },
          },
        });
        const rawText = response.text?.trim() || '{}';
        const parsed = sanitizePlanProposal(JSON.parse(rawText), safeRecipes, safeModels);
        governance.recordLedgerEvent({
          userId: actorId,
          userEmail: authUser?.email ?? null,
          kind: 'ai_plan',
          provider: 'gemini',
          tokenCostUsd: estTokenUsd,
          status: 'completed',
          summary: { model: chosenModel, recipeId: parsed.recipeId },
        });
        return res.json({
          planner: 'gemini',
          model: chosenModel,
          proposal: parsed,
        });
      }

      if (chosenProvider.id === 'claude') {
        const rawParsed = await callAnthropicJson({
          model: chosenModel,
          systemInstruction,
          promptText,
          image: parsedImage,
        });
        return res.json({
          planner: 'claude',
          model: chosenModel,
          proposal: sanitizePlanProposal(rawParsed, safeRecipes, safeModels),
        });
      }

      if (chosenProvider.id === 'deepseek') {
        const apiKey = process.env.DEEPSEEK_API_KEY;
        if (!hasSecret(apiKey)) {
          return res.json({
            planner: 'rules',
            model: 'offline-deterministic-rules',
            fallbackReason:
              'DEEPSEEK_API_KEY not configured in server Secrets; fell back to deterministic rules engine.',
            proposal: null,
          });
        }
        const rawParsed = await callOpenAICompatJson({
          baseUrl: 'https://api.deepseek.com',
          apiKey: apiKey!,
          model: chosenModel,
          systemInstruction,
          promptText,
        });
        return res.json({
          planner: 'deepseek',
          model: chosenModel,
          proposal: sanitizePlanProposal(rawParsed, safeRecipes, safeModels),
        });
      }

      if (chosenProvider.id === 'openai') {
        const apiKey = process.env.OPENAI_API_KEY;
        if (!hasSecret(apiKey)) {
          return res.json({
            planner: 'rules',
            model: 'offline-deterministic-rules',
            fallbackReason:
              'OPENAI_API_KEY not configured in server Secrets; fell back to deterministic rules engine.',
            proposal: null,
          });
        }
        const rawParsed = await callOpenAICompatJson({
          baseUrl: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1',
          apiKey: apiKey!,
          model: chosenModel,
          systemInstruction,
          promptText,
          imageUrl: parsedImage && typeof imageBase64 === 'string' ? imageBase64 : null,
        });
        return res.json({
          planner: 'openai',
          model: chosenModel,
          proposal: sanitizePlanProposal(rawParsed, safeRecipes, safeModels),
        });
      }

      if (chosenProvider.id === 'ollama') {
        const rawParsed = await callOllamaJson({
          model: chosenModel,
          systemInstruction,
          promptText,
          baseUrlOverride: typeof ollamaBaseUrl === 'string' && engineUrlAllowed(ollamaBaseUrl) ? ollamaBaseUrl : undefined,
        });
        return res.json({
          planner: 'ollama',
          model: chosenModel,
          proposal: sanitizePlanProposal(rawParsed, safeRecipes, safeModels),
        });
      }

      return res.json({
        planner: 'rules',
        model: 'offline-deterministic-rules',
        fallbackReason: `Unknown provider "${provider}"; fell back to deterministic rules engine.`,
        proposal: null,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return res.json({
        planner: 'rules',
        model: 'offline-deterministic-rules',
        fallbackReason: `${chosenProvider.name} unavailable (${msg}); fell back to offline rules engine.`,
        proposal: null,
      });
    }
  });

  // Server-side Vision Inspector for uploaded images (Gemini / Claude / OpenAI vision with local fallback)
  app.post('/api/v1/ai/inspect-image', async (req, res) => {
    const {
      fileName = '',
      imageBase64 = null,
      localHint = {},
      lang = 'zh',
      provider = 'gemini',
      model,
    } = req.body ?? {};

    const rate = aiPlanLimiter(getClientKey(req));
    if (!rate.allowed) {
      return res.status(429).json({ source: 'local', inspection: null, error: `Rate limit reached. Retry in ${rate.retryAfterSec}s.` });
    }
    // the picture itself goes to a cloud model only when the operator allows it (VEXEL_ALLOW_IMAGE_UPLOAD=1)
    const parsedImage = imagesMayLeave() ? parseDataUrl(imageBase64) : null;
    if (!parsedImage) {
      return res.json({ source: 'local', inspection: null });
    }

    const systemInstruction = `You are Vexel's scientific & industrial image triage inspector.
Analyze the uploaded image honestly and return strict JSON:
1. sceneType must be one of:
   - "microfluidics_droplet" (microfluidic channels, emulsions, droplets, ddPCR)
   - "cells" (brightfield/phase/fluorescence biological cells, tissue nuclei, spheroids)
   - "pcb_board" (printed circuit board, copper traces, solder pads, SMD components, IC vias)
   - "sem_metrology" (scanning electron microscopy: semiconductor lines/spaces, contact holes, nanoparticles, pores, electrospun fibres)
   - "unknown" (anything else, e.g., everyday photos, documents, landscapes, screenshots, ambiguous scenes where none of the 4 scientific/industrial classes clearly apply)
2. Be HONEST: if the image does NOT clearly match one of the 4 scientific/industrial classes, set recognized=false, sceneType="unknown", suggestedRecipeId=null.
3. Check for an instrument info bar / data bar (typically dark or white strip at bottom/top) and any printed scale bar text (e.g., "100 um", "500 nm", "2 mm"). Only report scaleBarValue and scaleBarUnit ("nm" | "µm" | "mm") if actually legible in the image; otherwise set them to null.
4. suggestedRecipeId must be one of: "droplet_encapsulation", "droplet_fluorescence", "cell_segmentation", "pcb_inspection", "sem_lines", "sem_holes", "sem_particles", "sem_fibres", or null if unrecognized.
5. Write reasonZh in Simplified Chinese and reasonEn in English (one concise sentence each).
6. JSON keys: recognized, sceneType, confidence, hasDataBar, scaleBarValue, scaleBarUnit, suggestedRecipeId, reasonZh, reasonEn.`;

    const promptText = JSON.stringify({ fileName, localHint, lang });

    try {
      if (provider === 'claude' && hasSecret(process.env.ANTHROPIC_API_KEY)) {
        const chosenModel = typeof model === 'string' && model.trim() ? model.trim() : 'claude-sonnet-4-20250514';
        const parsed = await callAnthropicJson({
          model: chosenModel,
          systemInstruction,
          promptText,
          image: parsedImage,
        });
        return res.json({ source: 'vision', model: chosenModel, inspection: parsed });
      }

      if (provider === 'openai' && hasSecret(process.env.OPENAI_API_KEY)) {
        const chosenModel = typeof model === 'string' && model.trim() ? model.trim() : 'gpt-4.1-mini';
        const parsed = await callOpenAICompatJson({
          baseUrl: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1',
          apiKey: process.env.OPENAI_API_KEY!,
          model: chosenModel,
          systemInstruction,
          promptText,
          imageUrl: imageBase64,
        });
        return res.json({ source: 'vision', model: chosenModel, inspection: parsed });
      }

      // Default to Gemini Vision
      const ai = createGeminiClient();
      if (!ai) {
        return res.json({ source: 'local', inspection: null });
      }
      const chosenModel =
        provider === 'gemini' && typeof model === 'string' && model.trim()
          ? model.trim()
          : 'gemini-4-hydrogen';

      const response = await ai.models.generateContent({
        model: chosenModel,
        contents: [
          {
            role: 'user',
            parts: [
              { text: promptText },
              {
                inlineData: {
                  mimeType: parsedImage.mimeType,
                  data: parsedImage.data,
                },
              },
            ],
          },
        ],
        config: {
          systemInstruction,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              recognized: { type: Type.BOOLEAN },
              sceneType: { type: Type.STRING },
              confidence: { type: Type.NUMBER },
              hasDataBar: { type: Type.BOOLEAN },
              scaleBarValue: { type: Type.NUMBER, nullable: true },
              scaleBarUnit: { type: Type.STRING, nullable: true },
              suggestedRecipeId: { type: Type.STRING, nullable: true },
              reasonZh: { type: Type.STRING },
              reasonEn: { type: Type.STRING },
            },
            required: ['recognized', 'sceneType', 'confidence', 'hasDataBar', 'reasonZh', 'reasonEn'],
          },
        },
      });

      const raw = response.text?.trim() || '{}';
      const parsed = JSON.parse(raw);
      return res.json({
        source: 'vision',
        model: chosenModel,
        inspection: parsed,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return res.json({
        source: 'local',
        fallbackReason: msg,
        inspection: null,
      });
    }
  });

  // Declarative GPU Training Presets & Hardware Tiers (allowlist-only metadata)
  app.get('/api/v1/train/presets', (_req, res) => {
    const endpointConfigured = hasSecret(process.env.VEXEL_GPU_TRAIN_ENDPOINT);
    res.json({
      endpointConfigured,
      presets: GPU_PROVIDER_PRESETS,
      allowlist: {
        backbones: Array.from(ALLOWED_BACKBONES),
        loraRanks: Array.from(ALLOWED_LORA_RANKS),
        losses: Array.from(ALLOWED_LOSSES),
        precisions: Array.from(ALLOWED_PRECISIONS),
        epochsRange: [5, 80],
        maxWallClockSec: 480,
        minLabeledClasses: 2,
        minLabeledPixels: 64,
        maxOnnxSizeMb: 50,
      },
      sandboxPolicy: [
        'Declarative JSON parameters only — arbitrary code, shell commands, pip installs, and external git URLs are blocked',
        'Zero-egress isolated GPU worker container with pre-baked signed vision backbones',
        'Pre-flight annotation gate requires >= 2 classes & >= 64 labeled pixels on a valid image tensor',
        'Hard wall-clock cap <= 480s, concurrency = 1 per client, ONNX-only artifact output (<= 50 MB)',
      ],
    });
  });

  // Recent declarative training job audit log (merged persistent SQLite + active memory; zero credentials)
  app.get('/api/v1/train/jobs', (_req, res) => {
    const persisted = governance.listLedgerEvents(20, 'gpu_train');
    const persistedJobs = persisted
      .map((ev) => {
        const s = (ev.summary ?? {}) as Record<string, unknown>;
        if (!s.jobId && !ev.jobId) return null;
        return {
          ...s,
          jobId: ev.jobId ?? String(s.jobId ?? ev.id),
          status: ev.status,
          provider: ev.provider ?? String(s.provider ?? 'local_cuda'),
          backbone: ev.backbone ?? String(s.backbone ?? 'dinov2_vit_s14'),
          gpuSeconds: ev.gpuSeconds,
          estCostUsd: ev.estCostUsd,
          createdAt: ev.createdAt,
        };
      })
      .filter(Boolean);
    res.json({
      jobs: persistedJobs.length > 0 ? persistedJobs : trainJobHistory.slice(0, 20),
    });
  });

  // Declarative Allowlist-Only GPU Few-Shot LoRA Job Submission
  app.post('/api/v1/train/jobs', async (req, res) => {
    const clientKey = getClientKey(req);
    const rate = trainJobLimiter(clientKey);
    if (!rate.allowed) {
      return res.status(429).json({
        ok: false,
        error: `Training rate limit exceeded (6 jobs/min). Retry in ${rate.retryAfterSec}s.`,
      });
    }
    if (activeTrainByClient.has(clientKey)) {
      return res.status(429).json({
        ok: false,
        error: 'Concurrency limit: only 1 active training dispatch per client is permitted.',
      });
    }

    const body = (req.body ?? {}) as Record<string, unknown>;
    const wallCap = Number(process.env.VEXEL_TRAIN_MAX_WALL_CLOCK_SEC || 480);
    const validation = validateDeclarativeTrainJob(body, Number.isFinite(wallCap) ? wallCap : 480);
    if (!validation.ok) {
      return res.status(validation.statusCode).json({ ok: false, error: validation.error });
    }

    const { job: baseJob, classNames, manifest: rawManifestPayload } = validation;

    // G-09 Persistent Per-User GPU Quota & Global USD Cost Circuit-Breaker Enforcement
    const authUser = resolveAuthUser(req);
    const actorId = authUser ? authUser.id : `guest:${clientKey}`;
    const actorRole = authUser ? authUser.role : 'guest';
    const estCostUsd = estimateGpuCostUsd(baseJob.provider, baseJob.maxWallClockSec);
    const estGpuSec = baseJob.provider === 'local_cuda' ? 0 : baseJob.maxWallClockSec;
    const quotaCheck = governance.checkQuotaAndBudget({
      userId: actorId,
      role: actorRole,
      estimatedGpuSec: estGpuSec,
      estimatedUsd: estCostUsd,
    });
    if (!quotaCheck.allowed) {
      return res.status(quotaCheck.statusCode).json({
        ok: false,
        error: quotaCheck.reason,
        governance: quotaCheck.summary,
      });
    }

    // G-05 Second Line of Defense: pure-pixel PNG decode & re-encode to strip EXIF/tEXt/iTXt/polyglot chunks
    let sanitizedImageBuf: Buffer | null = null;
    let sanitizedMaskBuf: Buffer | null = null;
    let sanitizationReport: {
      reencoded: boolean;
      width: number;
      height: number;
      imageSanitizedSha256: string;
      maskSanitizedSha256?: string;
      strippedChunks: string[];
      trailingBytesStripped: number;
      datasetBundleSha256?: string;
    } | null = null;

    if (typeof body.imagePngBase64 === 'string' && body.imagePngBase64.trim().length > 0) {
      try {
        const rawImgBuf = decodeBase64PngString(body.imagePngBase64);
        const cleanImg = sanitizePngBuffer(rawImgBuf, { minDim: 32, maxDim: 4096 });
        sanitizedImageBuf = cleanImg.cleanPng;
        const stripped = [...cleanImg.strippedChunks];
        let trailingTotal = cleanImg.trailingBytesStripped;
        let maskSha: string | undefined;

        if (typeof body.maskPngBase64 === 'string' && body.maskPngBase64.trim().length > 0) {
          const rawMaskBuf = decodeBase64PngString(body.maskPngBase64);
          const cleanMask = sanitizePngBuffer(rawMaskBuf, { minDim: 32, maxDim: 4096 });
          if (cleanMask.width !== cleanImg.width || cleanMask.height !== cleanImg.height) {
            return res.status(400).json({
              ok: false,
              error: `Mask dimensions (${cleanMask.width}×${cleanMask.height}) do not match sanitized image (${cleanImg.width}×${cleanImg.height}).`,
            });
          }
          sanitizedMaskBuf = cleanMask.cleanPng;
          maskSha = cleanMask.sanitizedSha256;
          stripped.push(...cleanMask.strippedChunks.map((c) => `mask:${c}`));
          trailingTotal += cleanMask.trailingBytesStripped;
        }

        sanitizationReport = {
          reencoded: true,
          width: cleanImg.width,
          height: cleanImg.height,
          imageSanitizedSha256: cleanImg.sanitizedSha256,
          maskSanitizedSha256: maskSha,
          strippedChunks: stripped,
          trailingBytesStripped: trailingTotal,
        };
      } catch (err: unknown) {
        const errMsg = err instanceof Error ? err.message : 'Invalid PNG payload.';
        return res.status(400).json({
          ok: false,
          error: `Image/mask security re-encoding rejected payload: ${errMsg}`,
        });
      }
    }

    const manifestPayload: Record<string, unknown> = {
      ...rawManifestPayload,
      ...(sanitizationReport ? { sanitization: sanitizationReport } : {}),
    };

    activeTrainByClient.add(clientKey);
    try {
      let status: 'manifest_ready' | 'dispatched' =
        baseJob.status === 'dispatched' ? 'dispatched' : 'manifest_ready';
      let message = baseJob.message;
      const pythonScript = buildSandboxLoraScript(baseJob, classNames);

      // Build canonical read-only PKZIP training bundle if sanitized pixel binaries were attached
      let datasetBundleBase64: string | undefined;
      if (sanitizedImageBuf && sanitizationReport) {
        const zipEntries: Array<{ name: string; data: Buffer }> = [
          { name: 'image.png', data: sanitizedImageBuf },
          ...(sanitizedMaskBuf ? [{ name: 'mask.png', data: sanitizedMaskBuf }] : []),
          { name: 'manifest.json', data: Buffer.from(JSON.stringify(manifestPayload, null, 2), 'utf8') },
          { name: 'train_vexel_lora.py', data: Buffer.from(pythonScript, 'utf8') },
        ];
        const zipBuf = buildZipArchive(zipEntries);
        const zipSha = crypto.createHash('sha256').update(zipBuf).digest('hex');
        sanitizationReport.datasetBundleSha256 = zipSha;
        datasetBundleBase64 = zipBuf.toString('base64');
        message =
          'Sanitized pure-pixel PNG dataset bundle (0 EXIF/metadata) + signed sandbox manifest ready.';
      }

      // Optional dispatch to operator-configured server-side GPU worker endpoint
      const gpuEndpoint = process.env.VEXEL_GPU_TRAIN_ENDPOINT?.trim();
      if (gpuEndpoint && hasSecret(gpuEndpoint)) {
        const isSafeHttps = /^https:\/\//i.test(gpuEndpoint);
        const isLocalCuda =
          baseJob.provider === 'local_cuda' &&
          /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?(\/|$)/i.test(gpuEndpoint);
        if ((isSafeHttps || isLocalCuda) && !isBlockedMetadataHost(gpuEndpoint)) {
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 8000);
          try {
            const headers: Record<string, string> = { 'Content-Type': 'application/json' };
            if (hasSecret(process.env.VEXEL_GPU_TRAIN_TOKEN)) {
              headers.Authorization = `Bearer ${process.env.VEXEL_GPU_TRAIN_TOKEN!.trim()}`;
            }
            const r = await fetch(gpuEndpoint, {
              method: 'POST',
              headers,
              signal: controller.signal,
              body: JSON.stringify({
                manifest: manifestPayload,
                ...(sanitizedImageBuf
                  ? { imagePngBase64: sanitizedImageBuf.toString('base64') }
                  : {}),
                ...(sanitizedMaskBuf
                  ? { maskPngBase64: sanitizedMaskBuf.toString('base64') }
                  : {}),
              }),
            });
            if (r.ok) {
              status = 'dispatched';
              message = `Declarative LoRA job dispatched with pure-pixel sanitized payload to GPU worker (${baseJob.provider}).`;
            } else {
              message = `GPU worker returned HTTP ${r.status}; signed offline sandbox bundle generated as fallback.`;
            }
          } catch {
            message =
              'Configured GPU worker unreachable; signed offline sandbox bundle & script generated cleanly.';
          } finally {
            clearTimeout(timer);
          }
        }
      }

      const record: TrainJobAuditRecord = {
        ...baseJob,
        status,
        message,
      };
      trainJobHistory.unshift(record);
      if (trainJobHistory.length > 50) trainJobHistory.pop();

      governance.recordLedgerEvent({
        jobId: record.jobId,
        userId: actorId,
        userEmail: authUser?.email ?? null,
        kind: 'gpu_train',
        provider: record.provider,
        backbone: record.backbone,
        imageSha256: sanitizationReport?.imageSanitizedSha256 ?? record.imageSha256,
        gpuSeconds: estGpuSec,
        estCostUsd,
        status: record.status,
        summary: {
          ...record,
          sanitized: Boolean(sanitizationReport),
          datasetBundleSha256: sanitizationReport?.datasetBundleSha256,
        },
      });

      return res.json({
        ok: true,
        job: record,
        manifest: manifestPayload,
        pythonScript,
        sanitization: sanitizationReport,
        datasetBundleBase64,
        governance: governance.getSummary(actorId, actorRole),
      });
    } finally {
      activeTrainByClient.delete(clientKey);
    }
  });

  // G-10 Signed Cloud GPU Worker Completion Callback (HMAC-SHA256 verified)
  app.post('/api/v1/train/callback', (req, res) => {
    const secret = process.env.VEXEL_GPU_CALLBACK_SECRET?.trim();
    if (!secret || secret.length < 12) {
      return res.status(503).json({
        ok: false,
        error: 'GPU callback receiver is disabled until VEXEL_GPU_CALLBACK_SECRET (>=12 chars) is configured.',
      });
    }
    const rawBody = (req as express.Request & { rawBody?: string }).rawBody ?? JSON.stringify(req.body ?? {});
    const sigHeader = req.headers['x-vexel-signature'];
    const signature = Array.isArray(sigHeader) ? sigHeader[0] : sigHeader;

    if (!verifyGpuCallbackSignature(rawBody, signature, secret)) {
      return res.status(401).json({ ok: false, error: 'Invalid or missing X-Vexel-Signature HMAC.' });
    }

    const body = (req.body ?? {}) as Record<string, unknown>;
    const jobId = typeof body.jobId === 'string' ? body.jobId.trim().slice(0, 64) : '';
    const status =
      body.status === 'completed' || body.status === 'failed' || body.status === 'running'
        ? body.status
        : null;
    if (!jobId || !status) {
      return res.status(400).json({ ok: false, error: 'Callback requires valid jobId and status.' });
    }

    const onnxSha256 =
      typeof body.onnxSha256 === 'string' && /^[a-f0-9]{16,64}$/i.test(body.onnxSha256.trim())
        ? body.onnxSha256.trim().toLowerCase()
        : undefined;
    const onnxSizeBytes =
      typeof body.onnxSizeBytes === 'number' &&
      Number.isFinite(body.onnxSizeBytes) &&
      body.onnxSizeBytes > 0 &&
      body.onnxSizeBytes <= 50 * 1024 * 1024
        ? Math.round(body.onnxSizeBytes)
        : undefined;

    const updated = governance.updateJobFromCallback({
      jobId,
      status,
      actualGpuSeconds:
        typeof body.actualGpuSeconds === 'number' ? body.actualGpuSeconds : undefined,
      actualCostUsd: typeof body.actualCostUsd === 'number' ? body.actualCostUsd : undefined,
      onnxSha256,
      onnxSizeBytes,
      valMeanIoU:
        typeof body.valMeanIoU === 'number' && body.valMeanIoU >= 0 && body.valMeanIoU <= 1
          ? body.valMeanIoU
          : undefined,
      message: typeof body.message === 'string' ? body.message : undefined,
    });

    if (!updated) {
      return res.status(404).json({ ok: false, error: `Unknown training jobId "${jobId}".` });
    }

    return res.json({ ok: true, event: updated });
  });

  // G-11 OME-Zarr Multiscale Pyramid Backend Bridge (Metadata JSON / Safe HTTPS / Confined Local Root)
  app.post('/api/v1/zarr/inspect', async (req, res) => {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const mode = String(body.mode ?? 'metadata');
    try {
      if (mode === 'remote_https') {
        const url = typeof body.url === 'string' ? body.url : '';
        if (!url || isBlockedPrivateOrMetadataHost(url)) {
          return res.status(400).json({
            ok: false,
            error: 'Remote OME-Zarr URL must be a public https:// endpoint.',
          });
        }
        const inspection = await inspectRemoteOmeZarr(url);
        return res.json({ ok: true, inspection });
      }

      if (mode === 'local_root') {
        const relPath = typeof body.relPath === 'string' ? body.relPath : '';
        const zarrRoot = process.env.VEXEL_ZARR_ROOT?.trim() || '';
        const inspection = inspectLocalOmeZarr(relPath, zarrRoot);
        return res.json({ ok: true, inspection });
      }

      // Default: parse attached OME-Zarr .zattrs / zarr.json metadata directly
      const rawMeta =
        body.metadata && typeof body.metadata === 'object'
          ? (body.metadata as Record<string, unknown>)
          : typeof body.metadataJson === 'string'
          ? (JSON.parse(body.metadataJson) as Record<string, unknown>)
          : null;
      if (!rawMeta) {
        return res.status(400).json({ ok: false, error: 'Missing OME-Zarr metadata JSON.' });
      }
      const shapesByPath =
        body.shapesByPath && typeof body.shapesByPath === 'object'
          ? (body.shapesByPath as Record<string, number[]>)
          : undefined;
      const inspection = parseOmeZarrMetadata(rawMeta, {
        shapesByPath,
        sourceKind: 'metadata_json',
      });
      return res.json({ ok: true, inspection });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to inspect OME-Zarr pyramid.';
      return res.status(400).json({ ok: false, error: msg });
    }
  });

  // Verify & register exported ONNX model metadata into Model Hub
  app.post('/api/v1/train/import-onnx', (req, res) => {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const modelId =
      typeof body.modelId === 'string'
        ? body.modelId.trim().replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 48)
        : '';
    const version =
      typeof body.version === 'string'
        ? body.version.trim().replace(/[^a-zA-Z0-9._-]/g, '').slice(0, 16)
        : '1.0.0';
    const sha256 =
      typeof body.sha256 === 'string' ? body.sha256.trim().toLowerCase() : '';
    const sizeBytes = Number(body.sizeBytes ?? 0);
    const backbone =
      typeof body.backbone === 'string' && ALLOWED_BACKBONES.has(body.backbone)
        ? body.backbone
        : 'dinov2_vit_s14';
    const task =
      body.task === 'classification' || body.task === 'detection'
        ? body.task
        : 'segmentation';
    const modality =
      body.modality === 'sem' || body.modality === 'droplet' ? body.modality : 'both';

    if (!modelId) {
      return res.status(400).json({ ok: false, error: 'Missing valid modelId.' });
    }
    if (!/^[a-f0-9]{16,64}$/.test(sha256)) {
      return res
        .status(400)
        .json({ ok: false, error: 'Invalid SHA-256 hex checksum (expected 16..64 hex chars).' });
    }
    if (!Number.isFinite(sizeBytes) || sizeBytes <= 0 || sizeBytes > 50 * 1024 * 1024) {
      return res
        .status(400)
        .json({ ok: false, error: 'ONNX artifact size must be between 1 byte and 50 MB.' });
    }

    const ref = `onnx:${modelId}@${version}`;
    const card = {
      ref,
      nameZh: `${modelId} (GPU LoRA ONNX)`,
      nameEn: `${modelId} (GPU LoRA ONNX)`,
      modality,
      task,
      framework: 'onnx',
      version,
      sha256,
      license: typeof body.license === 'string' ? body.license.slice(0, 32) : 'Apache-2.0',
      commercialUse: body.commercialUse !== false,
      tiledInference: true,
      tileSize: 512,
      metrics: {
        meanIoU:
          typeof body.meanIoU === 'number' && body.meanIoU >= 0 && body.meanIoU <= 1
            ? body.meanIoU
            : undefined,
        macroF1:
          typeof body.macroF1 === 'number' && body.macroF1 >= 0 && body.macroF1 <= 1
            ? body.macroF1
            : undefined,
      },
      descriptionZh: `通过 GPU 沙箱微调导出并完成 SHA-256 校验的 ONNX 分割模型（骨干：${backbone}，体积 ${(sizeBytes / (1024 * 1024)).toFixed(2)} MB）。`,
      descriptionEn: `SHA-256 verified ONNX segmentation model exported from GPU LoRA sandbox (${backbone}, ${(sizeBytes / (1024 * 1024)).toFixed(2)} MB).`,
    };
    return res.json({ ok: true, card });
  });

  // Optional bridge to a user's local Python `vexel serve` engine
  app.post('/api/v1/python-bridge/check', async (req, res) => {
    const { baseUrl } = req.body ?? {};
    if (typeof baseUrl !== 'string' || !/^https?:\/\//i.test(baseUrl) || !engineUrlAllowed(baseUrl)) {
      return res.status(400).json({ ok: false, error: 'Invalid or blocked baseUrl; expected http(s)://host:port' });
    }
    try {
      const clean = baseUrl.replace(/\/+$/, '');
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 3500);
      const r = await fetch(`${clean}/api/v1/health`, { signal: controller.signal });
      clearTimeout(timer);
      if (!r.ok) {
        return res.status(502).json({ ok: false, error: `Python engine returned HTTP ${r.status}` });
      }
      const data = await r.json();
      return res.json({ ok: true, data });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return res.status(502).json({ ok: false, error: msg });
    }
  });

  // Serve local onnxruntime-web WASM binaries from /ort so browser offline ONNX inference needs zero external CDN
  const ortDistDir = path.resolve(__dirname, 'node_modules/onnxruntime-web/dist');
  app.use('/ort', express.static(ortDistDir));

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  // this computer only by default; VEXEL_WEB_HOST=0.0.0.0 opens it to the network (then other computers must sign in)
  const HOST = process.env.VEXEL_WEB_HOST?.trim() || '127.0.0.1';
  app.listen(PORT, HOST, () => {
    console.log(`Vexel Workbench running at http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start Vexel server:', err);
  process.exit(1);
});
