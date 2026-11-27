import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { DatabaseSync } from 'node:sqlite';

export type UserRole = 'admin' | 'researcher';

export interface AuthUser {
  id: string;
  email: string;
  role: UserRole;
  createdAt: string;
}

export interface GovernancePolicy {
  dailyBudgetUsd: number;
  monthlyBudgetUsd: number;
  userDailyGpuSec: number;
  guestDailyGpuSec: number;
  circuitBreakerTripped: boolean;
  updatedAt: string;
}

export interface LedgerEventRecord {
  id: string;
  jobId: string | null;
  userId: string;
  userEmail: string | null;
  kind: 'gpu_train' | 'ai_plan' | 'onnx_import';
  provider: string | null;
  backbone: string | null;
  imageSha256: string | null;
  gpuSeconds: number;
  estCostUsd: number;
  tokenCostUsd: number;
  status: string;
  summary: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
}

export interface GovernanceSummary {
  policy: GovernancePolicy;
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

export interface QuotaCheckResult {
  allowed: boolean;
  statusCode: number;
  reason?: string;
  summary: GovernanceSummary;
}

export interface BootstrapStatus {
  needsAdminSetup: boolean;
  adminCount: number;
  mode: 'env_passcode' | 'claimed_db' | 'unclaimed';
}

const DEFAULT_POLICY: Omit<GovernancePolicy, 'updatedAt'> = {
  dailyBudgetUsd: 15.0,
  monthlyBudgetUsd: 180.0,
  userDailyGpuSec: 1800,
  guestDailyGpuSec: 360,
  circuitBreakerTripped: false,
};

const EMAIL_RE = /^[^\s@]{1,64}@[^\s@]{1,190}\.[^\s@]{2,24}$/;

// Ephemeral in-memory fallback signing key if VEXEL_SESSION_SECRET is not provided in env
const EPHEMERAL_SESSION_SECRET = crypto.randomBytes(32).toString('hex');

function getSessionSecret(): string {
  const fromEnv = process.env.VEXEL_SESSION_SECRET?.trim();
  if (fromEnv && fromEnv.length >= 16) return fromEnv;
  return EPHEMERAL_SESSION_SECRET;
}

export function hashPassword(password: string, saltHex?: string): { hashHex: string; saltHex: string } {
  const salt = saltHex || crypto.randomBytes(16).toString('hex');
  const derived = crypto.scryptSync(password, salt, 64, { N: 16384, r: 8, p: 1 });
  return { hashHex: derived.toString('hex'), saltHex: salt };
}

export function verifyPassword(password: string, expectedHashHex: string, saltHex: string): boolean {
  try {
    const { hashHex } = hashPassword(password, saltHex);
    const a = Buffer.from(hashHex, 'hex');
    const b = Buffer.from(expectedHashHex, 'hex');
    if (a.length !== b.length) return false;
    return crypto.timingSafeEqual(a, b);
  } catch {
    return false;
  }
}

export function timingSafeStringEqual(a: string, b: string): boolean {
  const aBuf = crypto.createHash('sha256').update(a).digest();
  const bBuf = crypto.createHash('sha256').update(b).digest();
  return crypto.timingSafeEqual(aBuf, bBuf);
}

export function estimateGpuCostUsd(provider: string, maxWallClockSec: number): number {
  // Approximate per-second USD rates for cost circuit-breaker reservation
  const ratePerHour: Record<string, number> = {
    local_cuda: 0.0,
    modal_serverless: 0.85,
    runpod_serverless: 0.69,
    autodl_cn: 0.45,
    gcp_vertex: 1.15,
    aws_sagemaker: 1.25,
  };
  const hourly = ratePerHour[provider] ?? 0.9;
  return Number(((hourly * Math.max(0, maxWallClockSec)) / 3600).toFixed(4));
}

export class GovernanceStore {
  private db: DatabaseSync;
  public readonly dbPath: string;

  constructor(dbPath?: string) {
    const resolved =
      dbPath ||
      process.env.VEXEL_DB_PATH?.trim() ||
      path.resolve(process.cwd(), 'data', 'vexel-governance.sqlite');
    this.dbPath = resolved;

    if (resolved !== ':memory:') {
      const dir = path.dirname(resolved);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }

    this.db = new DatabaseSync(resolved);
    this.initSchema();
  }

  private initSchema(): void {
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA foreign_keys = ON;

      CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT NOT NULL UNIQUE,
        role TEXT NOT NULL CHECK(role IN ('admin', 'researcher')),
        password_hash TEXT NOT NULL,
        password_salt TEXT NOT NULL,
        created_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS governance_config (
        id INTEGER PRIMARY KEY CHECK(id = 1),
        daily_budget_usd REAL NOT NULL,
        monthly_budget_usd REAL NOT NULL,
        user_daily_gpu_sec INTEGER NOT NULL,
        guest_daily_gpu_sec INTEGER NOT NULL,
        circuit_breaker_tripped INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS ledger_events (
        id TEXT PRIMARY KEY,
        job_id TEXT,
        user_id TEXT NOT NULL,
        user_email TEXT,
        kind TEXT NOT NULL,
        provider TEXT,
        backbone TEXT,
        image_sha256 TEXT,
        gpu_seconds REAL NOT NULL DEFAULT 0,
        est_cost_usd REAL NOT NULL DEFAULT 0,
        token_cost_usd REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        summary_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_ledger_created ON ledger_events(created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_ledger_user_created ON ledger_events(user_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_ledger_job_id ON ledger_events(job_id);
    `);

    const existingCfg = this.db
      .prepare('SELECT id FROM governance_config WHERE id = 1')
      .get() as { id: number } | undefined;

    if (!existingCfg) {
      const now = new Date().toISOString();
      this.db
        .prepare(
          `INSERT INTO governance_config (
            id, daily_budget_usd, monthly_budget_usd, user_daily_gpu_sec, guest_daily_gpu_sec, circuit_breaker_tripped, updated_at
          ) VALUES (1, ?, ?, ?, ?, 0, ?)`
        )
        .run(
          DEFAULT_POLICY.dailyBudgetUsd,
          DEFAULT_POLICY.monthlyBudgetUsd,
          DEFAULT_POLICY.userDailyGpuSec,
          DEFAULT_POLICY.guestDailyGpuSec,
          now
        );
    }
  }

  public close(): void {
    this.db.close();
  }

  public createSessionToken(user: AuthUser, ttlSec = 12 * 3600): string {
    const nowSec = Math.floor(Date.now() / 1000);
    const payload = {
      sub: user.id,
      email: user.email,
      role: user.role,
      iat: nowSec,
      exp: nowSec + ttlSec,
    };
    const bodyB64 = Buffer.from(JSON.stringify(payload), 'utf8').toString('base64url');
    const sig = crypto
      .createHmac('sha256', getSessionSecret())
      .update(bodyB64)
      .digest('base64url');
    return `${bodyB64}.${sig}`;
  }

  public verifySessionToken(token: string | undefined | null): AuthUser | null {
    if (!token || typeof token !== 'string') return null;
    const parts = token.split('.');
    if (parts.length !== 2) return null;
    const [bodyB64, sigB64] = parts;
    if (!bodyB64 || !sigB64) return null;

    const expectedSig = crypto
      .createHmac('sha256', getSessionSecret())
      .update(bodyB64)
      .digest('base64url');
    if (!timingSafeStringEqual(sigB64, expectedSig)) return null;

    try {
      const parsed = JSON.parse(Buffer.from(bodyB64, 'base64url').toString('utf8')) as {
        sub?: string;
        email?: string;
        role?: string;
        exp?: number;
      };
      if (!parsed.sub || !parsed.email || (parsed.role !== 'admin' && parsed.role !== 'researcher')) {
        return null;
      }
      const nowSec = Math.floor(Date.now() / 1000);
      if (typeof parsed.exp !== 'number' || parsed.exp <= nowSec) return null;

      // Verify env-based admin or persisted user record still exists
      if (parsed.sub === 'env_admin') {
        return {
          id: 'env_admin',
          email: parsed.email,
          role: 'admin',
          createdAt: new Date(0).toISOString(),
        };
      }
      const row = this.db
        .prepare('SELECT id, email, role, created_at FROM users WHERE id = ?')
        .get(parsed.sub) as
        | { id: string; email: string; role: UserRole; created_at: string }
        | undefined;
      if (!row) return null;
      return {
        id: row.id,
        email: row.email,
        role: row.role,
        createdAt: row.created_at,
      };
    } catch {
      return null;
    }
  }

  public getBootstrapStatus(): BootstrapStatus {
    const envPasscode = process.env.VEXEL_ADMIN_PASSCODE?.trim();
    const row = this.db
      .prepare("SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin'")
      .get() as { cnt: number };
    const adminCount = Number(row?.cnt ?? 0);

    if (envPasscode && envPasscode.length >= 6) {
      return { needsAdminSetup: false, adminCount, mode: 'env_passcode' };
    }
    if (adminCount > 0) {
      return { needsAdminSetup: false, adminCount, mode: 'claimed_db' };
    }
    return { needsAdminSetup: true, adminCount: 0, mode: 'unclaimed' };
  }

  public bootstrapAdmin(
    emailRaw: string,
    password: string
  ): { ok: true; user: AuthUser; token: string } | { ok: false; statusCode: number; error: string } {
    const envPasscode = process.env.VEXEL_ADMIN_PASSCODE?.trim();
    if (envPasscode && envPasscode.length >= 6) {
      return {
        ok: false,
        statusCode: 409,
        error: 'Admin access is managed by VEXEL_ADMIN_PASSCODE environment secret. Sign in directly.',
      };
    }

    const email = emailRaw.trim().toLowerCase();
    if (!EMAIL_RE.test(email)) {
      return { ok: false, statusCode: 400, error: 'Please provide a valid admin email address.' };
    }
    if (typeof password !== 'string' || password.length < 10 || password.length > 128) {
      return {
        ok: false,
        statusCode: 400,
        error: 'Admin password must be between 10 and 128 characters.',
      };
    }

    this.db.exec('BEGIN IMMEDIATE');
    try {
      const existingAdmin = this.db
        .prepare("SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin'")
        .get() as { cnt: number };
      if (Number(existingAdmin?.cnt ?? 0) > 0) {
        this.db.exec('ROLLBACK');
        return {
          ok: false,
          statusCode: 409,
          error: 'An admin account has already been claimed for this Vexel instance.',
        };
      }

      const id = `usr_${crypto.randomBytes(8).toString('hex')}`;
      const { hashHex, saltHex } = hashPassword(password);
      const now = new Date().toISOString();

      this.db
        .prepare(
          `INSERT INTO users (id, email, role, password_hash, password_salt, created_at)
           VALUES (?, ?, 'admin', ?, ?, ?)`
        )
        .run(id, email, hashHex, saltHex, now);

      this.db.exec('COMMIT');

      const user: AuthUser = { id, email, role: 'admin', createdAt: now };
      return { ok: true, user, token: this.createSessionToken(user) };
    } catch (err) {
      try {
        this.db.exec('ROLLBACK');
      } catch {
        // ignore rollback error
      }
      const msg = err instanceof Error ? err.message : 'Admin setup failed.';
      return { ok: false, statusCode: 409, error: msg };
    }
  }

  public registerResearcher(
    emailRaw: string,
    password: string
  ): { ok: true; user: AuthUser; token: string } | { ok: false; statusCode: number; error: string } {
    const email = emailRaw.trim().toLowerCase();
    if (!EMAIL_RE.test(email)) {
      return { ok: false, statusCode: 400, error: 'Please provide a valid email address.' };
    }
    if (typeof password !== 'string' || password.length < 8 || password.length > 128) {
      return {
        ok: false,
        statusCode: 400,
        error: 'Password must be between 8 and 128 characters.',
      };
    }

    const exists = this.db
      .prepare('SELECT id FROM users WHERE email = ?')
      .get(email) as { id: string } | undefined;
    if (exists) {
      return { ok: false, statusCode: 409, error: 'An account with this email already exists.' };
    }

    const id = `usr_${crypto.randomBytes(8).toString('hex')}`;
    const { hashHex, saltHex } = hashPassword(password);
    const now = new Date().toISOString();

    // Always force 'researcher' role on public registration — never accept role from caller
    this.db
      .prepare(
        `INSERT INTO users (id, email, role, password_hash, password_salt, created_at)
         VALUES (?, ?, 'researcher', ?, ?, ?)`
      )
      .run(id, email, hashHex, saltHex, now);

    const user: AuthUser = { id, email, role: 'researcher', createdAt: now };
    return { ok: true, user, token: this.createSessionToken(user) };
  }

  public loginUser(
    emailRaw: string,
    password: string
  ): { ok: true; user: AuthUser; token: string } | { ok: false; statusCode: number; error: string } {
    const email = emailRaw.trim().toLowerCase();
    if (!email || typeof password !== 'string' || password.length === 0) {
      return { ok: false, statusCode: 400, error: 'Email and password are required.' };
    }

    // VEXEL_ADMIN_PASSCODE environment variable overrides any claimed fallback hash
    const envPasscode = process.env.VEXEL_ADMIN_PASSCODE?.trim();
    if (envPasscode && envPasscode.length >= 6 && timingSafeStringEqual(password, envPasscode)) {
      const envAdmin: AuthUser = {
        id: 'env_admin',
        email: EMAIL_RE.test(email) ? email : 'admin@vexel.local',
        role: 'admin',
        createdAt: new Date(0).toISOString(),
      };
      return { ok: true, user: envAdmin, token: this.createSessionToken(envAdmin) };
    }

    const row = this.db
      .prepare('SELECT id, email, role, password_hash, password_salt, created_at FROM users WHERE email = ?')
      .get(email) as
      | {
          id: string;
          email: string;
          role: UserRole;
          password_hash: string;
          password_salt: string;
          created_at: string;
        }
      | undefined;

    if (!row || !verifyPassword(password, row.password_hash, row.password_salt)) {
      return { ok: false, statusCode: 401, error: 'Invalid email or password.' };
    }

    const user: AuthUser = {
      id: row.id,
      email: row.email,
      role: row.role,
      createdAt: row.created_at,
    };
    return { ok: true, user, token: this.createSessionToken(user) };
  }

  public getPolicy(): GovernancePolicy {
    const row = this.db
      .prepare(
        `SELECT daily_budget_usd, monthly_budget_usd, user_daily_gpu_sec, guest_daily_gpu_sec,
                circuit_breaker_tripped, updated_at
         FROM governance_config WHERE id = 1`
      )
      .get() as {
      daily_budget_usd: number;
      monthly_budget_usd: number;
      user_daily_gpu_sec: number;
      guest_daily_gpu_sec: number;
      circuit_breaker_tripped: number;
      updated_at: string;
    };

    return {
      dailyBudgetUsd: Number(row.daily_budget_usd),
      monthlyBudgetUsd: Number(row.monthly_budget_usd),
      userDailyGpuSec: Number(row.user_daily_gpu_sec),
      guestDailyGpuSec: Number(row.guest_daily_gpu_sec),
      circuitBreakerTripped: Boolean(row.circuit_breaker_tripped),
      updatedAt: row.updated_at,
    };
  }

  public updatePolicy(patch: Partial<Omit<GovernancePolicy, 'updatedAt'>>): GovernancePolicy {
    const current = this.getPolicy();
    const dailyBudgetUsd =
      typeof patch.dailyBudgetUsd === 'number' && Number.isFinite(patch.dailyBudgetUsd)
        ? Math.max(0.5, Math.min(5000, Number(patch.dailyBudgetUsd.toFixed(2))))
        : current.dailyBudgetUsd;
    const monthlyBudgetUsd =
      typeof patch.monthlyBudgetUsd === 'number' && Number.isFinite(patch.monthlyBudgetUsd)
        ? Math.max(dailyBudgetUsd, Math.min(50000, Number(patch.monthlyBudgetUsd.toFixed(2))))
        : Math.max(dailyBudgetUsd, current.monthlyBudgetUsd);
    const userDailyGpuSec =
      typeof patch.userDailyGpuSec === 'number' && Number.isFinite(patch.userDailyGpuSec)
        ? Math.max(60, Math.min(86400, Math.round(patch.userDailyGpuSec)))
        : current.userDailyGpuSec;
    const guestDailyGpuSec =
      typeof patch.guestDailyGpuSec === 'number' && Number.isFinite(patch.guestDailyGpuSec)
        ? Math.max(0, Math.min(3600, Math.round(patch.guestDailyGpuSec)))
        : current.guestDailyGpuSec;
    const circuitBreakerTripped =
      typeof patch.circuitBreakerTripped === 'boolean'
        ? patch.circuitBreakerTripped
        : current.circuitBreakerTripped;
    const now = new Date().toISOString();

    this.db
      .prepare(
        `UPDATE governance_config
         SET daily_budget_usd = ?, monthly_budget_usd = ?, user_daily_gpu_sec = ?,
             guest_daily_gpu_sec = ?, circuit_breaker_tripped = ?, updated_at = ?
         WHERE id = 1`
      )
      .run(
        dailyBudgetUsd,
        monthlyBudgetUsd,
        userDailyGpuSec,
        guestDailyGpuSec,
        circuitBreakerTripped ? 1 : 0,
        now
      );

    return this.getPolicy();
  }

  public getSummary(userId = 'guest', role: UserRole | 'guest' = 'guest'): GovernanceSummary {
    const policy = this.getPolicy();
    const now = new Date();
    const dayStartIso = new Date(
      Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), 0, 0, 0)
    ).toISOString();
    const monthStartIso = new Date(
      Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1, 0, 0, 0)
    ).toISOString();

    const dayRow = this.db
      .prepare(
        `SELECT COALESCE(SUM(est_cost_usd + token_cost_usd), 0) AS spend,
                COALESCE(SUM(gpu_seconds), 0) AS gpu_sec
         FROM ledger_events
         WHERE created_at >= ? AND status != 'rejected'`
      )
      .get(dayStartIso) as { spend: number; gpu_sec: number };

    const monthRow = this.db
      .prepare(
        `SELECT COALESCE(SUM(est_cost_usd + token_cost_usd), 0) AS spend
         FROM ledger_events
         WHERE created_at >= ? AND status != 'rejected'`
      )
      .get(monthStartIso) as { spend: number };

    const userDayRow = this.db
      .prepare(
        `SELECT COALESCE(SUM(gpu_seconds), 0) AS gpu_sec
         FROM ledger_events
         WHERE user_id = ? AND created_at >= ? AND status != 'rejected'`
      )
      .get(userId, dayStartIso) as { gpu_sec: number };

    const totalRow = this.db
      .prepare('SELECT COUNT(*) AS cnt FROM ledger_events')
      .get() as { cnt: number };

    const todaySpendUsd = Number(Number(dayRow?.spend ?? 0).toFixed(4));
    const monthSpendUsd = Number(Number(monthRow?.spend ?? 0).toFixed(4));
    const todayGpuSecTotal = Math.round(Number(dayRow?.gpu_sec ?? 0));
    const userTodayGpuSec = Math.round(Number(userDayRow?.gpu_sec ?? 0));
    const userDailyGpuLimit =
      role === 'admin'
        ? policy.userDailyGpuSec * 4
        : role === 'researcher'
        ? policy.userDailyGpuSec
        : policy.guestDailyGpuSec;

    const circuitBreakerActive =
      policy.circuitBreakerTripped ||
      todaySpendUsd >= policy.dailyBudgetUsd ||
      monthSpendUsd >= policy.monthlyBudgetUsd;

    return {
      policy,
      todaySpendUsd,
      monthSpendUsd,
      todayGpuSecTotal,
      userTodayGpuSec,
      userDailyGpuLimit,
      dailyBudgetUtilization: Number(
        Math.min(1, todaySpendUsd / Math.max(0.01, policy.dailyBudgetUsd)).toFixed(4)
      ),
      monthlyBudgetUtilization: Number(
        Math.min(1, monthSpendUsd / Math.max(0.01, policy.monthlyBudgetUsd)).toFixed(4)
      ),
      totalEvents: Number(totalRow?.cnt ?? 0),
      circuitBreakerActive,
    };
  }

  public checkQuotaAndBudget(input: {
    userId: string;
    role: UserRole | 'guest';
    estimatedGpuSec: number;
    estimatedUsd: number;
  }): QuotaCheckResult {
    const summary = this.getSummary(input.userId, input.role);
    const { policy } = summary;

    if (policy.circuitBreakerTripped) {
      return {
        allowed: false,
        statusCode: 503,
        reason:
          'Cost circuit breaker is tripped by admin policy. Cloud GPU/AI spend is paused until an admin resets it.',
        summary,
      };
    }

    if (input.estimatedUsd > 0 && summary.todaySpendUsd + input.estimatedUsd > policy.dailyBudgetUsd) {
      // Auto-trip circuit breaker so subsequent spend fails fast
      this.updatePolicy({ circuitBreakerTripped: true });
      return {
        allowed: false,
        statusCode: 429,
        reason: `Daily USD budget cap ($${policy.dailyBudgetUsd.toFixed(2)}) reached (spent $${summary.todaySpendUsd.toFixed(2)}). Circuit breaker tripped.`,
        summary: this.getSummary(input.userId, input.role),
      };
    }

    if (
      input.estimatedUsd > 0 &&
      summary.monthSpendUsd + input.estimatedUsd > policy.monthlyBudgetUsd
    ) {
      this.updatePolicy({ circuitBreakerTripped: true });
      return {
        allowed: false,
        statusCode: 429,
        reason: `Monthly USD budget cap ($${policy.monthlyBudgetUsd.toFixed(2)}) reached (spent $${summary.monthSpendUsd.toFixed(2)}). Circuit breaker tripped.`,
        summary: this.getSummary(input.userId, input.role),
      };
    }

    if (
      input.estimatedGpuSec > 0 &&
      summary.userTodayGpuSec + input.estimatedGpuSec > summary.userDailyGpuLimit
    ) {
      return {
        allowed: false,
        statusCode: 429,
        reason: `Daily GPU quota exceeded (${summary.userTodayGpuSec}/${summary.userDailyGpuLimit}s used today). Sign in as a researcher or request higher quota from admin.`,
        summary,
      };
    }

    return { allowed: true, statusCode: 200, summary };
  }

  public recordLedgerEvent(input: {
    jobId?: string | null;
    userId: string;
    userEmail?: string | null;
    kind: 'gpu_train' | 'ai_plan' | 'onnx_import';
    provider?: string | null;
    backbone?: string | null;
    imageSha256?: string | null;
    gpuSeconds?: number;
    estCostUsd?: number;
    tokenCostUsd?: number;
    status: string;
    summary?: Record<string, unknown> | null;
  }): LedgerEventRecord {
    const id = `evt_${Date.now().toString(36)}_${crypto.randomBytes(4).toString('hex')}`;
    const now = new Date().toISOString();
    const gpuSeconds = Math.max(0, Number(input.gpuSeconds ?? 0));
    const estCostUsd = Math.max(0, Number(input.estCostUsd ?? 0));
    const tokenCostUsd = Math.max(0, Number(input.tokenCostUsd ?? 0));
    const summaryJson = input.summary ? JSON.stringify(input.summary) : null;

    this.db
      .prepare(
        `INSERT INTO ledger_events (
          id, job_id, user_id, user_email, kind, provider, backbone, image_sha256,
          gpu_seconds, est_cost_usd, token_cost_usd, status, summary_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        id,
        input.jobId ?? null,
        input.userId,
        input.userEmail ?? null,
        input.kind,
        input.provider ?? null,
        input.backbone ?? null,
        input.imageSha256 ?? null,
        gpuSeconds,
        estCostUsd,
        tokenCostUsd,
        input.status,
        summaryJson,
        now,
        now
      );

    return {
      id,
      jobId: input.jobId ?? null,
      userId: input.userId,
      userEmail: input.userEmail ?? null,
      kind: input.kind,
      provider: input.provider ?? null,
      backbone: input.backbone ?? null,
      imageSha256: input.imageSha256 ?? null,
      gpuSeconds,
      estCostUsd,
      tokenCostUsd,
      status: input.status,
      summary: input.summary ?? null,
      createdAt: now,
      updatedAt: now,
    };
  }

  public updateJobFromCallback(input: {
    jobId: string;
    status: 'completed' | 'failed' | 'running';
    actualGpuSeconds?: number;
    actualCostUsd?: number;
    onnxSha256?: string;
    onnxSizeBytes?: number;
    valMeanIoU?: number;
    message?: string;
  }): LedgerEventRecord | null {
    const row = this.db
      .prepare('SELECT * FROM ledger_events WHERE job_id = ? ORDER BY created_at DESC LIMIT 1')
      .get(input.jobId) as Record<string, unknown> | undefined;
    if (!row) return null;

    let mergedSummary: Record<string, unknown> = {};
    if (typeof row.summary_json === 'string') {
      try {
        mergedSummary = JSON.parse(row.summary_json) as Record<string, unknown>;
      } catch {
        mergedSummary = {};
      }
    }
    mergedSummary = {
      ...mergedSummary,
      status: input.status,
      ...(input.message ? { message: input.message.slice(0, 280) } : {}),
      ...(input.onnxSha256 ? { onnxSha256: input.onnxSha256 } : {}),
      ...(typeof input.onnxSizeBytes === 'number' ? { onnxSizeBytes: input.onnxSizeBytes } : {}),
      ...(typeof input.valMeanIoU === 'number' ? { valMeanIoU: input.valMeanIoU } : {}),
      callbackReceivedAt: new Date().toISOString(),
    };

    const nextGpuSec =
      typeof input.actualGpuSeconds === 'number' && Number.isFinite(input.actualGpuSeconds)
        ? Math.max(0, input.actualGpuSeconds)
        : Number(row.gpu_seconds ?? 0);
    const nextCost =
      typeof input.actualCostUsd === 'number' && Number.isFinite(input.actualCostUsd)
        ? Math.max(0, input.actualCostUsd)
        : Number(row.est_cost_usd ?? 0);
    const now = new Date().toISOString();

    this.db
      .prepare(
        `UPDATE ledger_events
         SET status = ?, gpu_seconds = ?, est_cost_usd = ?, summary_json = ?, updated_at = ?
         WHERE id = ?`
      )
      .run(input.status, nextGpuSec, nextCost, JSON.stringify(mergedSummary), now, String(row.id));

    return this.getLedgerEventById(String(row.id));
  }

  public getLedgerEventById(id: string): LedgerEventRecord | null {
    const row = this.db
      .prepare('SELECT * FROM ledger_events WHERE id = ?')
      .get(id) as Record<string, unknown> | undefined;
    return row ? this.mapLedgerRow(row) : null;
  }

  public listLedgerEvents(limit = 30, kind?: LedgerEventRecord['kind']): LedgerEventRecord[] {
    const safeLimit = Math.max(1, Math.min(200, Math.round(limit)));
    const rows = kind
      ? (this.db
          .prepare('SELECT * FROM ledger_events WHERE kind = ? ORDER BY created_at DESC LIMIT ?')
          .all(kind, safeLimit) as Array<Record<string, unknown>>)
      : (this.db
          .prepare('SELECT * FROM ledger_events ORDER BY created_at DESC LIMIT ?')
          .all(safeLimit) as Array<Record<string, unknown>>);
    return rows.map((r) => this.mapLedgerRow(r));
  }

  private mapLedgerRow(row: Record<string, unknown>): LedgerEventRecord {
    let summary: Record<string, unknown> | null = null;
    if (typeof row.summary_json === 'string' && row.summary_json) {
      try {
        summary = JSON.parse(row.summary_json) as Record<string, unknown>;
      } catch {
        summary = null;
      }
    }
    return {
      id: String(row.id),
      jobId: row.job_id ? String(row.job_id) : null,
      userId: String(row.user_id),
      userEmail: row.user_email ? String(row.user_email) : null,
      kind: (row.kind as LedgerEventRecord['kind']) || 'gpu_train',
      provider: row.provider ? String(row.provider) : null,
      backbone: row.backbone ? String(row.backbone) : null,
      imageSha256: row.image_sha256 ? String(row.image_sha256) : null,
      gpuSeconds: Number(row.gpu_seconds ?? 0),
      estCostUsd: Number(row.est_cost_usd ?? 0),
      tokenCostUsd: Number(row.token_cost_usd ?? 0),
      status: String(row.status ?? 'unknown'),
      summary,
      createdAt: String(row.created_at),
      updatedAt: String(row.updated_at),
    };
  }
}
