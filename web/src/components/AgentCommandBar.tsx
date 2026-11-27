import { useRef, useState } from 'react';
import {
  Bot,
  ChevronDown,
  Eye,
  KeyRound,
  Loader2,
  Send,
  Settings2,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react';
import type {
  LLMProviderConfig,
  LLMProviderId,
  LLMProviderInfo,
  Lang,
  Modality,
  PlanProposal,
} from '../types/vexel';
import { useClickOutside } from '../hooks/useClickOutside';

interface AgentCommandBarProps {
  lang: Lang;
  modality: Modality;
  offlinePolicy: boolean;
  aiLoading: boolean;
  plan: PlanProposal | null;
  providers: LLMProviderInfo[];
  providerConfig: LLMProviderConfig;
  onChangeProviderConfig: (cfg: LLMProviderConfig) => void;
  onSubmitGoal: (goal: string, sendImage: boolean) => Promise<void>;
}

const DEFAULT_FALLBACK_PROVIDERS: LLMProviderInfo[] = [
  {
    id: 'gemini',
    name: 'Google Gemini',
    envVar: 'GEMINI_API_KEY',
    configured: false,
    supportsVision: true,
    defaultModel: 'gemini-4-hydrogen',
    models: ['gemini-4-hydrogen', 'gemini-2.5-flash', 'gemini-2.5-pro'],
    noteZh: '默认支持图像视觉识别、标尺 OCR 与确定性工作流规划',
    noteEn: 'Default vision + scale-bar OCR + deterministic workflow planner',
  },
  {
    id: 'claude',
    name: 'Anthropic Claude',
    envVar: 'ANTHROPIC_API_KEY',
    configured: false,
    supportsVision: true,
    defaultModel: 'claude-sonnet-4-20250514',
    models: ['claude-sonnet-4-20250514', 'claude-3-7-sonnet-latest', 'claude-3-5-haiku-latest'],
    noteZh: '支持多模态显微图像理解与严谨科学流程规划',
    noteEn: 'Multimodal microscopy reasoning and structured workflow planning',
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    envVar: 'DEEPSEEK_API_KEY',
    configured: false,
    supportsVision: false,
    defaultModel: 'deepseek-chat',
    models: ['deepseek-chat', 'deepseek-reasoner'],
    noteZh: '高性价比中文自然语言工作流编排与参数优化',
    noteEn: 'Fast workflow planning & parameter optimization',
  },
  {
    id: 'openai',
    name: 'OpenAI / Compatible',
    envVar: 'OPENAI_API_KEY',
    configured: false,
    supportsVision: true,
    defaultModel: 'gpt-4.1-mini',
    models: ['gpt-4.1-mini', 'gpt-4o', 'o4-mini'],
    noteZh: '支持 OpenAI 及自定义 OPENAI_BASE_URL 兼容网关',
    noteEn: 'Supports OpenAI or any OpenAI-compatible gateway',
  },
  {
    id: 'ollama',
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

export function AgentCommandBar({
  lang,
  modality,
  offlinePolicy,
  aiLoading,
  plan,
  providers,
  providerConfig,
  onChangeProviderConfig,
  onSubmitGoal,
}: AgentCommandBarProps) {
  const [goal, setGoal] = useState('');
  const [sendImage, setSendImage] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsTriggerRef = useRef<HTMLButtonElement>(null);
  const settingsPopoverRef = useRef<HTMLDivElement>(null);
  useClickOutside(settingsOpen, [settingsTriggerRef, settingsPopoverRef], () =>
    setSettingsOpen(false)
  );

  const catalog = providers.length > 0 ? providers : DEFAULT_FALLBACK_PROVIDERS;
  const activeProvider =
    catalog.find((p) => p.id === providerConfig.provider) ?? catalog[0];

  const quickPrompts =
    modality === 'sem'
      ? [
          {
            zh: '去除扫描条纹并校平背景',
            en: 'Destripe & level background',
          },
          {
            zh: '改用训练模型分割目标',
            en: 'Switch to Trained Model',
          },
          {
            zh: '滤除微小杂散噪点',
            en: 'Filter small debris noise',
          },
        ]
      : [
          {
            zh: '过滤过小或粘连液滴',
            en: 'Filter small/merged droplets',
          },
          {
            zh: '启用训练模型提升分割精度',
            en: 'Use Trained Model for segmentation',
          },
          {
            zh: '提高细胞识别对比度阈值',
            en: 'Raise cell contrast threshold',
          },
        ];

  const handleSelectProvider = (id: LLMProviderId) => {
    const info = catalog.find((p) => p.id === id);
    if (!info) return;
    onChangeProviderConfig({
      ...providerConfig,
      provider: id,
      model: info.defaultModel,
    });
    if (!info.supportsVision) setSendImage(false);
  };

  const handleSubmit = async (textOverride?: string) => {
    const q = (textOverride ?? goal).trim();
    if (!q || aiLoading) return;
    setGoal(q);
    await onSubmitGoal(q, sendImage && activeProvider.supportsVision && !offlinePolicy);
  };

  const statusTone = offlinePolicy
    ? 'text-slate-500 dark:text-slate-400'
    : activeProvider.configured
    ? 'text-emerald-600 dark:text-emerald-400'
    : 'text-amber-600 dark:text-amber-400';

  const statusText = offlinePolicy
    ? lang === 'zh'
      ? '● 离线规则'
      : '● Local Rules'
    : activeProvider.configured
    ? lang === 'zh'
      ? '● 已就绪'
      : '● Ready'
    : lang === 'zh'
    ? '▲ 未配密钥'
    : '▲ Key Needed';

  return (
    <div className="relative shrink-0 border-b border-slate-200 dark:border-slate-800 bg-slate-50/90 dark:bg-[#0D1322] px-4 py-2.5 z-20">
      <div className="flex flex-wrap lg:flex-nowrap items-center gap-2.5">
        {/* Left: Clear Natural Language Agent Identity */}
        <div className="flex items-center gap-2 shrink-0">
          <div className="w-8 h-8 rounded-md border border-blue-600/35 dark:border-cyan-500/40 bg-blue-600/12 dark:bg-cyan-500/15 flex items-center justify-center text-blue-600 dark:text-cyan-400">
            <Bot className="w-4 h-4" />
          </div>
          <div className="leading-tight">
            <div className="text-[12px] font-semibold text-slate-900 dark:text-slate-100 whitespace-nowrap flex items-center gap-1.5">
              <span>{lang === 'zh' ? '自然语言工作流助手' : 'Natural Language Agent'}</span>
              <span className={`text-[10.5px] font-mono font-normal ${statusTone}`}>
                {statusText}
              </span>
            </div>
            <div className="text-[10.5px] font-mono text-slate-500 dark:text-slate-400 truncate max-w-[200px]">
              {activeProvider.name} · {providerConfig.model}
            </div>
          </div>
        </div>

        {/* Center: Prominent Natural-Language Prompt Form */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void handleSubmit();
          }}
          className="flex-1 min-w-[280px] flex items-center gap-2"
        >
          <div className="relative flex-1 min-w-0 group">
            <Sparkles className="w-4 h-4 text-blue-600 dark:text-cyan-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder={
                lang === 'zh'
                  ? '点击此处输入自然语言指令：如“去除扫描条纹并改用训练模型分割”或“过滤小尺寸噪声”...'
                  : 'Click here to enter a command (e.g., "destripe SEM & use trained model", "filter small noise")...'
              }
              className="w-full h-9 pl-9 pr-16 rounded-md border-[1.5px] border-blue-400/80 dark:border-cyan-500/55 bg-white dark:bg-[#070B14] text-[13px] text-slate-900 dark:text-slate-100 placeholder:text-slate-500 dark:placeholder:text-slate-400 shadow-xs hover:border-blue-500 dark:hover:border-cyan-400 focus:outline-none focus:border-blue-600 dark:focus:border-cyan-400 focus:ring-2 focus:ring-blue-500/20 dark:focus:ring-cyan-400/25 transition-all"
            />
            {!goal && (
              <span className="pointer-events-none hidden sm:inline-flex items-center px-1.5 py-0.5 rounded border border-slate-200 dark:border-slate-700 bg-slate-100/90 dark:bg-slate-800/90 font-mono text-[10px] text-slate-500 dark:text-slate-400 absolute right-2 top-1/2 -translate-y-1/2">
                Enter ↵
              </span>
            )}
          </div>

          <button
            type="button"
            disabled={offlinePolicy || !activeProvider.supportsVision}
            onClick={() => setSendImage((v) => !v)}
            title={
              !activeProvider.supportsVision
                ? lang === 'zh'
                  ? '当前模型仅支持文本指纹规划'
                  : 'Selected provider is text-only'
                : lang === 'zh'
                ? '附带当前降采样图像给视觉大模型参考'
                : 'Attach downsampled image for multimodal vision reasoning'
            }
            className={`h-9 px-2.5 rounded-md border text-[11.5px] font-medium flex items-center gap-1.5 whitespace-nowrap shrink-0 shadow-xs transition-colors cursor-pointer disabled:opacity-40 ${
              sendImage && activeProvider.supportsVision && !offlinePolicy
                ? 'border-blue-600 dark:border-cyan-400 bg-blue-600/12 dark:bg-cyan-500/20 text-blue-700 dark:text-cyan-200'
                : 'border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800/90 text-slate-700 dark:text-slate-200 hover:border-blue-500 dark:hover:border-cyan-400 hover:bg-blue-50/60 dark:hover:bg-slate-700'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">{lang === 'zh' ? '看图' : 'Vision'}</span>
          </button>

          <button
            type="submit"
            disabled={aiLoading || !goal.trim()}
            className="h-9 px-3.5 rounded-md bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[12.5px] font-semibold shadow-xs flex items-center gap-1.5 whitespace-nowrap shrink-0 disabled:opacity-45 disabled:cursor-not-allowed cursor-pointer"
          >
            {aiLoading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Send className="w-3.5 h-3.5" />
            )}
            <span>{aiLoading ? (lang === 'zh' ? '规划中' : 'Planning') : lang === 'zh' ? '执行指令' : 'Plan'}</span>
          </button>
        </form>

        {/* Right: Quick Action Suggestions + LLM Provider Settings Trigger */}
        <div className="flex items-center gap-1.5 shrink-0">
          <div className="hidden xl:flex items-center gap-1.5">
            {quickPrompts.map((qp, idx) => (
              <button
                key={idx}
                type="button"
                disabled={aiLoading}
                onClick={() => void handleSubmit(lang === 'zh' ? qp.zh : qp.en)}
                className="h-8 px-2.5 rounded-md border border-slate-300 dark:border-slate-600/90 hover:border-blue-500 dark:hover:border-cyan-400 bg-white dark:bg-slate-800/85 hover:bg-blue-50/70 dark:hover:bg-slate-700/90 shadow-xs text-[11px] font-medium text-slate-700 dark:text-slate-200 hover:text-blue-700 dark:hover:text-cyan-200 whitespace-nowrap cursor-pointer transition-colors"
              >
                {lang === 'zh' ? qp.zh : qp.en}
              </button>
            ))}
          </div>

          <button
            ref={settingsTriggerRef}
            type="button"
            onClick={() => setSettingsOpen((v) => !v)}
            className={`h-9 px-3 rounded-md border text-[11.5px] font-medium flex items-center gap-1.5 whitespace-nowrap shrink-0 shadow-xs cursor-pointer transition-colors ${
              settingsOpen
                ? 'border-blue-600 dark:border-cyan-400 bg-blue-600/12 dark:bg-cyan-500/20 text-blue-700 dark:text-cyan-200'
                : 'border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800/90 hover:border-blue-500 dark:hover:border-cyan-400 hover:bg-blue-50/60 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-100'
            }`}
          >
            <Settings2 className="w-3.5 h-3.5" />
            <span>{lang === 'zh' ? '模型提供商' : 'LLM Provider'}</span>
            <ChevronDown className="w-3 h-3 opacity-75" />
          </button>
        </div>
      </div>

      {/* Compact feedback line if the last plan had a fallback notice or active LLM info */}
      {plan?.fallbackNotice && (
        <div className="mt-1.5 text-[11px] font-mono text-amber-600 dark:text-amber-400 flex items-center gap-1.5 truncate">
          <span>▲</span>
          <span className="truncate">{plan.fallbackNotice}</span>
        </div>
      )}

      {/* Multi-LLM Provider & Server-Side Secret Configuration Popover */}
      {settingsOpen && (
        <div
          ref={settingsPopoverRef}
          className="absolute right-4 top-12 w-[400px] max-w-[calc(100vw-2rem)] rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-[#111827] shadow-2xl p-4 space-y-3.5 z-50 text-[12px]"
        >
          <div className="flex items-center justify-between gap-2 pb-2 border-b border-slate-200 dark:border-slate-800">
            <div className="flex items-center gap-2 font-semibold text-slate-900 dark:text-slate-100">
              <Settings2 className="w-4 h-4 text-blue-600 dark:text-cyan-400" />
              <span>{lang === 'zh' ? 'LLM 提供商与服务端密钥配置' : 'LLM Provider & Server Key Status'}</span>
            </div>
            <button
              type="button"
              onClick={() => setSettingsOpen(false)}
              className="text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Provider Radio Cards */}
          <div className="space-y-1.5">
            <div className="text-[11px] uppercase tracking-wider font-semibold text-slate-500">
              {lang === 'zh' ? '选择大模型提供商' : 'Select Provider'}
            </div>
            <div className="divide-y divide-slate-200 dark:divide-slate-800 border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden">
              {catalog.map((p) => {
                const selected = p.id === providerConfig.provider;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => handleSelectProvider(p.id)}
                    className={`w-full px-3 py-2 text-left flex items-center justify-between gap-2 transition-colors cursor-pointer ${
                      selected
                        ? 'bg-blue-50/90 dark:bg-cyan-500/15'
                        : 'bg-white dark:bg-[#0D121F] hover:bg-slate-50 dark:hover:bg-slate-800/50'
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`font-semibold ${
                            selected
                              ? 'text-blue-700 dark:text-cyan-300'
                              : 'text-slate-900 dark:text-slate-100'
                          }`}
                        >
                          {p.name}
                        </span>
                        {p.supportsVision && (
                          <span className="text-[10px] font-mono text-slate-400">· Vision</span>
                        )}
                      </div>
                      <div className="text-[10.5px] text-slate-500 dark:text-slate-400 truncate">
                        {lang === 'zh' ? p.noteZh : p.noteEn}
                      </div>
                    </div>
                    <span
                      className={`text-[10.5px] font-mono shrink-0 ${
                        p.configured
                          ? 'text-emerald-600 dark:text-emerald-400'
                          : 'text-amber-600 dark:text-amber-400'
                      }`}
                    >
                      {p.configured
                        ? lang === 'zh'
                          ? '● 已配置'
                          : '● Ready'
                        : `▲ ${p.envVar}`}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Model ID Selection */}
          <div className="space-y-1.5">
            <label className="text-[11px] uppercase tracking-wider font-semibold text-slate-500 block">
              {lang === 'zh' ? '模型版本 (Model ID)' : 'Model ID'}
            </label>
            <div className="grid grid-cols-2 gap-2">
              <select
                value={
                  activeProvider.models.includes(providerConfig.model)
                    ? providerConfig.model
                    : '__custom__'
                }
                onChange={(e) => {
                  if (e.target.value !== '__custom__') {
                    onChangeProviderConfig({ ...providerConfig, model: e.target.value });
                  }
                }}
                className="h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11.5px] text-slate-900 dark:text-slate-100"
              >
                {activeProvider.models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
                <option value="__custom__">
                  {lang === 'zh' ? '自定义模型名...' : 'Custom model...'}
                </option>
              </select>
              <input
                type="text"
                value={providerConfig.model}
                onChange={(e) =>
                  onChangeProviderConfig({ ...providerConfig, model: e.target.value })
                }
                placeholder={activeProvider.defaultModel}
                className="h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11.5px] text-slate-900 dark:text-slate-100"
              />
            </div>
          </div>

          {/* Local Ollama Base URL */}
          {providerConfig.provider === 'ollama' && (
            <div className="space-y-1">
              <label className="text-[11px] uppercase tracking-wider font-semibold text-slate-500 block">
                {lang === 'zh' ? '本地 Ollama 服务地址' : 'Local Ollama Base URL'}
              </label>
              <input
                type="text"
                value={providerConfig.ollamaBaseUrl ?? 'http://127.0.0.1:11434'}
                onChange={(e) =>
                  onChangeProviderConfig({ ...providerConfig, ollamaBaseUrl: e.target.value })
                }
                placeholder="http://127.0.0.1:11434"
                className="w-full h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[11.5px] text-slate-900 dark:text-slate-100"
              />
            </div>
          )}

          {/* Security-first API Key Management Notice (Server-side Secrets / .env only) */}
          <div className="p-2.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60 space-y-1.5 text-[11px]">
            <div className="flex items-center gap-1.5 font-semibold text-slate-800 dark:text-slate-200">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
              <span>
                {lang === 'zh'
                  ? 'API Key 安全托管（仅服务端代理）'
                  : 'Secure Server-Side API Key Proxy'}
              </span>
            </div>
            <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
              {lang === 'zh'
                ? '为防止密钥在浏览器泄露，所有 API Key 均通过服务端路由 /api/v1/ai/* 代理。请在 AI Studio 左侧 Secrets 面板或部署环境 .env 中设置对应环境变量：'
                : 'To keep credentials out of browser storage, all LLM calls proxy through /api/v1/ai/*. Configure keys in the AI Studio Secrets panel or server .env:'}
            </p>
            <div className="flex items-center gap-1.5 font-mono text-[10.5px] text-blue-700 dark:text-cyan-300">
              <KeyRound className="w-3 h-3 shrink-0" />
              <span className="truncate">{activeProvider.envVar}</span>
              <span className="text-slate-400">·</span>
              <span className="text-slate-500 dark:text-slate-400">
                {activeProvider.configured
                  ? lang === 'zh'
                    ? '当前环境已检测到有效密钥'
                    : 'Detected in server environment'
                  : lang === 'zh'
                  ? '尚未设置（当前自动回退本地规则引擎）'
                  : 'Not set (falls back to local rules)'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
