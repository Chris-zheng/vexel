import { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  ChevronDown,
  ChevronUp,
  Cpu,
  KeyRound,
  LogOut,
  ShieldAlert,
  ShieldCheck,
  UserCheck,
  Users,
} from 'lucide-react';
import type { Lang } from '../types/vexel';
import { probeHardwareAcceleration, type HardwareAccelProfile } from '../engine/webgpuProbe';

interface AuthUserView {
  id: string;
  email: string;
  role: 'admin' | 'researcher';
}

interface GovernanceSummaryView {
  policy: {
    dailyBudgetUsd: number;
    monthlyBudgetUsd: number;
    userDailyGpuSec: number;
    guestDailyGpuSec: number;
    circuitBreakerTripped: boolean;
    updatedAt: string;
  };
  todaySpendUsd: number;
  monthSpendUsd: number;
  todayGpuSecTotal: number;
  userTodayGpuSec: number;
  userDailyGpuLimit: number;
  dailyBudgetUtilization: number;
  monthlyBudgetUtilization: number;
  totalEvents: number;
  circuitBreakerActive: boolean;
}

interface LedgerEventView {
  id: string;
  jobId: string | null;
  userEmail: string | null;
  userId: string;
  kind: string;
  provider: string | null;
  backbone: string | null;
  gpuSeconds: number;
  estCostUsd: number;
  tokenCostUsd: number;
  status: string;
  createdAt: string;
}

interface GovernanceQuotaPanelProps {
  lang: Lang;
  refreshSignal?: number;
  onAuthTokenChange?: (token: string | null) => void;
}

/**
 * G-09 Multi-User Auth + Persistent SQLite Quota & Cost Circuit-Breaker Control Panel.
 * Session Bearer tokens are held strictly in React JS memory (never written to localStorage/sessionStorage).
 */
export function GovernanceQuotaPanel({
  lang,
  refreshSignal = 0,
  onAuthTokenChange,
}: GovernanceQuotaPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<AuthUserView | null>(null);
  const [needsAdminSetup, setNeedsAdminSetup] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register' | 'bootstrap'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [authMsg, setAuthMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [summary, setSummary] = useState<GovernanceSummaryView | null>(null);
  const [ledger, setLedger] = useState<LedgerEventView[]>([]);
  const [showLedger, setShowLedger] = useState(false);
  const [hwProfile, setHwProfile] = useState<HardwareAccelProfile | null>(null);

  // Admin budget draft fields
  const [draftDailyUsd, setDraftDailyUsd] = useState('15');
  const [draftMonthlyUsd, setDraftMonthlyUsd] = useState('180');
  const [draftUserGpuSec, setDraftUserGpuSec] = useState('1800');

  const updateToken = useCallback(
    (nextToken: string | null) => {
      setToken(nextToken);
      onAuthTokenChange?.(nextToken);
    },
    [onAuthTokenChange]
  );

  const loadGovernance = useCallback(async (activeToken: string | null) => {
    const headers: Record<string, string> = {};
    if (activeToken) headers.Authorization = `Bearer ${activeToken}`;
    try {
      const [statusResp, sumResp] = await Promise.all([
        fetch('/api/v1/auth/status', { headers }),
        fetch('/api/v1/governance/summary', { headers }),
      ]);
      if (statusResp.ok) {
        const st = await statusResp.json();
        setNeedsAdminSetup(Boolean(st.needsAdminSetup));
        if (st.needsAdminSetup) setAuthMode('bootstrap');
        setUser(st.user ?? null);
      }
      if (sumResp.ok) {
        const sm = await sumResp.json();
        if (sm.summary) {
          setSummary(sm.summary);
          setDraftDailyUsd(String(sm.summary.policy.dailyBudgetUsd));
          setDraftMonthlyUsd(String(sm.summary.policy.monthlyBudgetUsd));
          setDraftUserGpuSec(String(sm.summary.policy.userDailyGpuSec));
        }
      }
    } catch {
      // keep silent in offline static preview
    }
  }, []);

  useEffect(() => {
    void loadGovernance(token);
  }, [loadGovernance, token, refreshSignal]);

  useEffect(() => {
    void probeHardwareAcceleration().then(setHwProfile);
  }, []);

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    setAuthMsg(null);
    setBusy(true);
    try {
      const endpoint =
        authMode === 'bootstrap'
          ? '/api/v1/auth/bootstrap-admin'
          : authMode === 'register'
          ? '/api/v1/auth/register'
          : '/api/v1/auth/login';

      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), password }),
      });
      const data = await resp.json();
      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || `HTTP ${resp.status}`);
      }
      updateToken(String(data.token));
      setUser(data.user as AuthUserView);
      setPassword('');
      setNeedsAdminSetup(false);
      setAuthMode('login');
      setAuthMsg(
        lang === 'zh'
          ? `已登录为 ${data.user.email} (${data.user.role === 'admin' ? '管理员' : '研究员'})，会话凭证仅存于内存。`
          : `Signed in as ${data.user.email} (${data.user.role}). Session held in memory only.`
      );
      await loadGovernance(String(data.token));
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleSignOut = () => {
    updateToken(null);
    setUser(null);
    setLedger([]);
    setShowLedger(false);
    setAuthMsg(lang === 'zh' ? '已退出登录，当前使用访客配额。' : 'Signed out (guest quota active).');
  };

  const handleSaveBudget = async (tripOverride?: boolean) => {
    if (!token || user?.role !== 'admin') return;
    setAuthError(null);
    setAuthMsg(null);
    setBusy(true);
    try {
      const resp = await fetch('/api/v1/governance/budget', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          dailyBudgetUsd: Number(draftDailyUsd),
          monthlyBudgetUsd: Number(draftMonthlyUsd),
          userDailyGpuSec: Number(draftUserGpuSec),
          ...(typeof tripOverride === 'boolean' ? { circuitBreakerTripped: tripOverride } : {}),
        }),
      });
      const data = await resp.json();
      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || `HTTP ${resp.status}`);
      }
      if (data.summary) setSummary(data.summary);
      setAuthMsg(
        typeof tripOverride === 'boolean'
          ? tripOverride
            ? lang === 'zh'
              ? '已手动触发成本熔断：暂停云端 GPU 与云端 AI 支出。'
              : 'Cost circuit breaker tripped: cloud GPU & AI spend paused.'
            : lang === 'zh'
            ? '已恢复成本熔断状态。'
            : 'Cost circuit breaker reset.'
          : lang === 'zh'
          ? '配额与预算策略已持久化写入 SQLite。'
          : 'Quota & budget policy persisted to SQLite.'
      );
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleLoadLedger = async () => {
    if (!token || user?.role !== 'admin') return;
    if (showLedger) {
      setShowLedger(false);
      return;
    }
    try {
      const resp = await fetch('/api/v1/governance/ledger?limit=15', {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await resp.json();
      if (resp.ok && Array.isArray(data.events)) {
        setLedger(data.events);
        setShowLedger(true);
      }
    } catch {
      // ignore
    }
  };

  const dailyPct = Math.round((summary?.dailyBudgetUtilization ?? 0) * 100);
  const gpuPct = summary
    ? Math.min(100, Math.round((summary.userTodayGpuSec / Math.max(1, summary.userDailyGpuLimit)) * 100))
    : 0;

  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E1320] overflow-hidden">
      {/* Summary Header Bar */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 text-left hover:bg-slate-50/70 dark:hover:bg-slate-900/50 cursor-pointer"
      >
        <div className="flex items-center gap-2 text-[12.5px] font-semibold text-slate-900 dark:text-slate-100">
          {summary?.circuitBreakerActive ? (
            <ShieldAlert className="w-4 h-4 text-rose-500 shrink-0" />
          ) : (
            <ShieldCheck className="w-4 h-4 text-emerald-500 shrink-0" />
          )}
          <span>
            {lang === 'zh'
              ? '多用户鉴权 · 配额与成本熔断台账 (SQLite)'
              : 'Multi-User Auth · Quota & Cost Circuit-Breaker (SQLite)'}
          </span>
          {needsAdminSetup && (
            <span className="font-mono text-[10px] px-1.5 py-0.5 rounded border border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300">
              {lang === 'zh' ? '首次启动待认领管理员' : 'Claim Admin Setup'}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 text-[11px] font-mono text-slate-500 dark:text-slate-400">
          {hwProfile && (
            <span className="hidden sm:inline-flex items-center gap-1 text-blue-600 dark:text-cyan-400">
              <Cpu className="w-3 h-3" />
              <span>{hwProfile.webgpuSupported ? 'WebGPU + WASM' : 'WebGL2 + WASM'}</span>
            </span>
          )}
          {summary && (
            <span>
              ${summary.todaySpendUsd.toFixed(2)} / ${summary.policy.dailyBudgetUsd.toFixed(0)} ·{' '}
              {summary.userTodayGpuSec}/{summary.userDailyGpuLimit}s GPU
            </span>
          )}
          <span className="inline-flex items-center gap-1 text-slate-700 dark:text-slate-200">
            {user ? (
              <>
                <UserCheck className="w-3.5 h-3.5 text-emerald-500" />
                <span>{user.role}</span>
              </>
            ) : (
              <>
                <Users className="w-3.5 h-3.5" />
                <span>{lang === 'zh' ? '访客' : 'Guest'}</span>
              </>
            )}
          </span>
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 pt-2 border-t border-slate-200 dark:border-slate-800 space-y-3.5 text-[12px]">
          {/* Hardware Acceleration Probe Summary */}
          {hwProfile && (
            <div className="px-3 py-2 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0B0F1A] flex items-center justify-between gap-2 text-[11px] font-mono text-slate-600 dark:text-slate-300">
              <div className="flex items-center gap-1.5 min-w-0">
                <Activity className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400 shrink-0" />
                <span className="truncate">{lang === 'zh' ? hwProfile.summaryZh : hwProfile.summaryEn}</span>
              </div>
              <span className="shrink-0 text-slate-400">24ch Pyramid</span>
            </div>
          )}

          {/* Budget & Quota Progress Meters */}
          {summary && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-[11px]">
              <div className="p-2.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-[#0B0F1A] space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-500">{lang === 'zh' ? '全站日预算' : 'Daily USD Cap'}</span>
                  <span className="font-semibold text-slate-800 dark:text-slate-100">
                    ${summary.todaySpendUsd.toFixed(2)} / ${summary.policy.dailyBudgetUsd.toFixed(2)}
                  </span>
                </div>
                <div className="h-1.5 rounded bg-slate-200 dark:bg-slate-800 overflow-hidden">
                  <div
                    className={`h-full ${dailyPct >= 90 ? 'bg-rose-500' : 'bg-blue-600 dark:bg-cyan-400'}`}
                    style={{ width: `${Math.min(100, dailyPct)}%` }}
                  />
                </div>
              </div>

              <div className="p-2.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-[#0B0F1A] space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-500">{lang === 'zh' ? '全站月预算' : 'Monthly USD Cap'}</span>
                  <span className="font-semibold text-slate-800 dark:text-slate-100">
                    ${summary.monthSpendUsd.toFixed(2)} / ${summary.policy.monthlyBudgetUsd.toFixed(2)}
                  </span>
                </div>
                <div className="h-1.5 rounded bg-slate-200 dark:bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-emerald-500"
                    style={{ width: `${Math.min(100, Math.round(summary.monthlyBudgetUtilization * 100))}%` }}
                  />
                </div>
              </div>

              <div className="p-2.5 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-[#0B0F1A] space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-500">{lang === 'zh' ? '当前账号日 GPU 额' : 'Account GPU Quota'}</span>
                  <span className="font-semibold text-slate-800 dark:text-slate-100">
                    {summary.userTodayGpuSec}s / {summary.userDailyGpuLimit}s
                  </span>
                </div>
                <div className="h-1.5 rounded bg-slate-200 dark:bg-slate-800 overflow-hidden">
                  <div
                    className={`h-full ${gpuPct >= 90 ? 'bg-amber-500' : 'bg-cyan-500'}`}
                    style={{ width: `${gpuPct}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Auth Form OR Active Session Bar */}
          {!user ? (
            <form onSubmit={handleAuthSubmit} className="space-y-2.5 p-3 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-[#0B0F1A]">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 text-[11.5px] font-semibold text-slate-800 dark:text-slate-200">
                  <KeyRound className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
                  <span>
                    {authMode === 'bootstrap'
                      ? lang === 'zh'
                        ? '首次启动：设置管理员账号（完成后自动锁定注册入口）'
                        : 'First-Run Setup: Claim Admin Account (auto-locks once claimed)'
                      : authMode === 'register'
                      ? lang === 'zh'
                        ? '注册研究员账号（解锁更高每日 GPU 配额）'
                        : 'Register Researcher Account'
                      : lang === 'zh'
                      ? '账号登录（会话令牌仅驻留内存）'
                      : 'Sign In (In-Memory Session Only)'}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-[11px]">
                  {needsAdminSetup && (
                    <button
                      type="button"
                      onClick={() => setAuthMode('bootstrap')}
                      className={`px-2 py-0.5 rounded border cursor-pointer ${
                        authMode === 'bootstrap'
                          ? 'border-amber-500/60 bg-amber-500/15 text-amber-700 dark:text-amber-300'
                          : 'border-slate-300 dark:border-slate-700 text-slate-500'
                      }`}
                    >
                      {lang === 'zh' ? '认领管理员' : 'Claim Admin'}
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setAuthMode('login')}
                    className={`px-2 py-0.5 rounded border cursor-pointer ${
                      authMode === 'login'
                        ? 'border-blue-500/60 bg-blue-500/15 text-blue-700 dark:text-cyan-300'
                        : 'border-slate-300 dark:border-slate-700 text-slate-500'
                    }`}
                  >
                    {lang === 'zh' ? '登录' : 'Sign In'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setAuthMode('register')}
                    className={`px-2 py-0.5 rounded border cursor-pointer ${
                      authMode === 'register'
                        ? 'border-blue-500/60 bg-blue-500/15 text-blue-700 dark:text-cyan-300'
                        : 'border-slate-300 dark:border-slate-700 text-slate-500'
                    }`}
                  >
                    {lang === 'zh' ? '注册研究员' : 'Register'}
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder={authMode === 'bootstrap' ? 'admin@lab.org' : 'researcher@lab.org'}
                  className="h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] text-slate-900 dark:text-slate-100"
                />
                <input
                  type="password"
                  required
                  minLength={authMode === 'bootstrap' ? 10 : 8}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={
                    authMode === 'bootstrap'
                      ? lang === 'zh'
                        ? '管理员密码 (≥10 位)'
                        : 'Admin password (≥10 chars)'
                      : lang === 'zh'
                      ? '密码 (≥8 位)'
                      : 'Password (≥8 chars)'
                  }
                  className="h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] text-slate-900 dark:text-slate-100"
                />
                <button
                  type="submit"
                  disabled={busy}
                  className="h-8 px-3 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 font-medium text-[12px] cursor-pointer disabled:opacity-50"
                >
                  {busy
                    ? '...'
                    : authMode === 'bootstrap'
                    ? lang === 'zh'
                      ? '认领并登录管理员'
                      : 'Claim Admin'
                    : authMode === 'register'
                    ? lang === 'zh'
                      ? '注册研究员账号'
                      : 'Create Account'
                    : lang === 'zh'
                    ? '登录'
                    : 'Sign In'}
                </button>
              </div>
            </form>
          ) : (
            <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 rounded border border-emerald-500/35 bg-emerald-500/8">
              <div className="flex items-center gap-2 font-mono text-[11.5px] text-emerald-800 dark:text-emerald-200">
                <UserCheck className="w-4 h-4 shrink-0" />
                <span>{user.email}</span>
                <span className="px-1.5 py-0.5 rounded bg-emerald-600/15 uppercase text-[10px] font-semibold">
                  {user.role}
                </span>
              </div>
              <button
                type="button"
                onClick={handleSignOut}
                className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[11px] flex items-center gap-1 cursor-pointer"
              >
                <LogOut className="w-3 h-3" />
                <span>{lang === 'zh' ? '退出' : 'Sign Out'}</span>
              </button>
            </div>
          )}

          {/* Admin-only Budget & Circuit-Breaker Controls */}
          {user?.role === 'admin' && summary && (
            <div className="p-3 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-[#0B0F1A] space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  {lang === 'zh' ? '管理员成本与熔断策略' : 'Admin Cost & Circuit-Breaker Policy'}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void handleSaveBudget(!summary.policy.circuitBreakerTripped)}
                    className={`h-7 px-2.5 rounded text-[11px] font-medium cursor-pointer border ${
                      summary.policy.circuitBreakerTripped
                        ? 'border-emerald-500/50 bg-emerald-600 text-white'
                        : 'border-rose-500/50 bg-rose-500/15 text-rose-700 dark:text-rose-300'
                    }`}
                  >
                    {summary.policy.circuitBreakerTripped
                      ? lang === 'zh'
                        ? '解除成本熔断'
                        : 'Reset Circuit Breaker'
                      : lang === 'zh'
                      ? '立即手动熔断'
                      : 'Trip Circuit Breaker'}
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleLoadLedger()}
                    className="h-7 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[11px] font-medium cursor-pointer"
                  >
                    {showLedger
                      ? lang === 'zh'
                        ? '收起审计台账'
                        : 'Hide Ledger'
                      : lang === 'zh'
                      ? `查看 SQLite 审计台账 (${summary.totalEvents})`
                      : `Inspect SQLite Ledger (${summary.totalEvents})`}
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
                <label className="space-y-1">
                  <span className="text-[10px] text-slate-500 block">Daily USD</span>
                  <input
                    type="number"
                    step={1}
                    min={1}
                    value={draftDailyUsd}
                    onChange={(e) => setDraftDailyUsd(e.target.value)}
                    className="w-full h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-[10px] text-slate-500 block">Monthly USD</span>
                  <input
                    type="number"
                    step={5}
                    min={5}
                    value={draftMonthlyUsd}
                    onChange={(e) => setDraftMonthlyUsd(e.target.value)}
                    className="w-full h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-[10px] text-slate-500 block">User GPU Sec/Day</span>
                  <input
                    type="number"
                    step={60}
                    min={60}
                    value={draftUserGpuSec}
                    onChange={(e) => setDraftUserGpuSec(e.target.value)}
                    className="w-full h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900"
                  />
                </label>
                <div className="flex items-end">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void handleSaveBudget()}
                    className="w-full h-7 rounded bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 font-medium cursor-pointer"
                  >
                    {lang === 'zh' ? '保存策略' : 'Save Policy'}
                  </button>
                </div>
              </div>

              {showLedger && (
                <div className="rounded border border-slate-200 dark:border-slate-800 overflow-x-auto max-h-44 overflow-y-auto">
                  <table className="w-full text-left border-collapse font-mono text-[10.5px]">
                    <thead className="bg-slate-100 dark:bg-slate-900 text-slate-500">
                      <tr>
                        <th className="py-1 px-2">Kind</th>
                        <th className="py-1 px-2">Actor</th>
                        <th className="py-1 px-2">Provider</th>
                        <th className="py-1 px-2">GPU s</th>
                        <th className="py-1 px-2">Cost $</th>
                        <th className="py-1 px-2">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                      {ledger.length === 0 ? (
                        <tr>
                          <td colSpan={6} className="py-2 px-2 text-slate-400 text-center">
                            {lang === 'zh' ? '暂无持久化台账记录' : 'No ledger events yet'}
                          </td>
                        </tr>
                      ) : (
                        ledger.map((ev) => (
                          <tr key={ev.id}>
                            <td className="py-1 px-2">{ev.kind}</td>
                            <td className="py-1 px-2 truncate max-w-[120px]">{ev.userEmail || ev.userId}</td>
                            <td className="py-1 px-2">{ev.provider || '—'}</td>
                            <td className="py-1 px-2">{ev.gpuSeconds.toFixed(0)}</td>
                            <td className="py-1 px-2">${(ev.estCostUsd + ev.tokenCostUsd).toFixed(4)}</td>
                            <td className="py-1 px-2">{ev.status}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {authError && (
            <div className="p-2 rounded border border-rose-500/40 bg-rose-500/10 text-[11.5px] text-rose-700 dark:text-rose-300">
              {authError}
            </div>
          )}
          {authMsg && (
            <div className="p-2 rounded border border-emerald-500/35 bg-emerald-500/10 text-[11.5px] text-emerald-700 dark:text-emerald-300">
              {authMsg}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
