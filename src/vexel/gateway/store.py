"""The gateway's records, in one SQLite file (``VEXEL_GATEWAY_DB``, default
``~/.vexel/gateway.db``): organisations and their limits, invite codes, users and their tokens,
the usage ledger and the response cache.

Nothing here holds an image, a goal's text or a model's answer text except the cache, which
keeps the validated, structured answer (method and settings) under a hash of its request.
Tokens and invite codes are stored as SHA-256 hashes: the database alone does not let anyone in.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS orgs (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
    region TEXT NOT NULL DEFAULT 'intl',             -- 'cn': DeepSeek first; 'intl': Gemini first
    monthly_budget_usd REAL NOT NULL DEFAULT 50.0,
    allow_images INTEGER NOT NULL DEFAULT 0,         -- pictures may reach a model only when 1
    daily_requests INTEGER NOT NULL DEFAULT 30,      -- per user
    daily_tokens INTEGER NOT NULL DEFAULT 50000,     -- per user, input + output
    active INTEGER NOT NULL DEFAULT 1, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS invites (
    code_hash TEXT PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES orgs(id),
    uses_left INTEGER NOT NULL, expires REAL NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES orgs(id), label TEXT NOT NULL,
    token_hash TEXT UNIQUE NOT NULL, active INTEGER NOT NULL DEFAULT 1, created REAL NOT NULL, last_seen REAL);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, org_id INTEGER, user_id INTEGER, plan_id TEXT,
    task TEXT, stage TEXT, provider TEXT, model TEXT, status TEXT NOT NULL,   -- ok | cached | refused | failed | quota
    reason TEXT, tokens_in INTEGER NOT NULL DEFAULT 0, tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0, image INTEGER NOT NULL DEFAULT 0, ms INTEGER);
CREATE INDEX IF NOT EXISTS usage_user_ts ON usage(user_id, ts);
CREATE INDEX IF NOT EXISTS usage_org_ts ON usage(org_id, ts);
CREATE INDEX IF NOT EXISTS usage_plan ON usage(plan_id);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, answer TEXT NOT NULL, provider TEXT, model TEXT, created REAL NOT NULL);
"""

CACHE_DAYS = 7


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def db_path() -> Path:
    env = os.environ.get("VEXEL_GATEWAY_DB")
    return Path(env) if env else Path(os.environ.get("VEXEL_HOME", Path.home() / ".vexel")) / "gateway.db"


@dataclass(frozen=True)
class Org:
    id: int
    name: str
    region: str
    monthly_budget_usd: float
    allow_images: bool
    daily_requests: int
    daily_tokens: int
    active: bool


@dataclass(frozen=True)
class User:
    id: int
    org: Org
    label: str


class Store:
    """Thread-safe access to the gateway database (one connection, serialised)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.commit()

    def _q(self, sql: str, args: tuple = (), *, commit: bool = False) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
            if commit:
                self._db.commit()
            return rows

    # -- organisations and people ---------------------------------------------------------------

    def add_org(self, name: str, *, region: str = "intl", monthly_budget_usd: float = 50.0, allow_images: bool = False,
                daily_requests: int = 30, daily_tokens: int = 50_000) -> Org:  # fmt: skip
        if region not in ("cn", "intl"):
            raise ValueError("region must be 'cn' or 'intl'")
        self._q("INSERT INTO orgs (name, region, monthly_budget_usd, allow_images, daily_requests, daily_tokens, created) VALUES (?,?,?,?,?,?,?)",
                (name, region, monthly_budget_usd, int(allow_images), daily_requests, daily_tokens, time.time()), commit=True)  # fmt: skip
        return self.org(name)

    def org(self, name_or_id: str | int) -> Org:
        col = "id" if isinstance(name_or_id, int) else "name"
        rows = self._q(f"SELECT * FROM orgs WHERE {col} = ?", (name_or_id,))
        if not rows:
            raise KeyError(f"no organisation {name_or_id!r}")
        r = rows[0]
        return Org(r["id"], r["name"], r["region"], r["monthly_budget_usd"], bool(r["allow_images"]), r["daily_requests"], r["daily_tokens"], bool(r["active"]))

    def set_org(self, name: str, **fields: Any) -> Org:
        allowed = {"region", "monthly_budget_usd", "allow_images", "daily_requests", "daily_tokens", "active"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot set {sorted(bad)}")
        for k, v in fields.items():
            self._q(f"UPDATE orgs SET {k} = ? WHERE name = ?", (int(v) if isinstance(v, bool) else v, name), commit=True)
        return self.org(name)

    def orgs(self) -> list[Org]:
        return [self.org(int(r["id"])) for r in self._q("SELECT id FROM orgs ORDER BY name")]

    def new_invite(self, org: str, *, uses: int = 10, days: float = 30) -> str:
        """An invite code (shown once): each use signs one person (one computer) in."""
        o = self.org(org)
        code = "VX-" + "-".join(secrets.token_hex(2).upper() for _ in range(4))
        self._q("INSERT INTO invites (code_hash, org_id, uses_left, expires, created) VALUES (?,?,?,?,?)",
                (_hash(code), o.id, uses, time.time() + days * 86400, time.time()), commit=True)  # fmt: skip
        return code

    def redeem(self, code: str, label: str) -> str:
        """Use an invite: a new user of its organisation and that user's token (shown once)."""
        code = code.strip().upper()
        with self._lock:
            row = self._db.execute("SELECT * FROM invites WHERE code_hash = ?", (_hash(code),)).fetchone()
            if row is None or row["uses_left"] <= 0 or row["expires"] < time.time():
                raise PermissionError("邀请码无效或已用完 / invalid or used-up invite code")
            token = "vxg_" + secrets.token_urlsafe(32)
            self._db.execute("UPDATE invites SET uses_left = uses_left - 1 WHERE code_hash = ?", (row["code_hash"],))
            self._db.execute("INSERT INTO users (org_id, label, token_hash, created) VALUES (?,?,?,?)",
                             (row["org_id"], label[:80] or "user", _hash(token), time.time()))  # fmt: skip
            self._db.commit()
        return token

    def user_for(self, token: str) -> User | None:
        rows = self._q("SELECT * FROM users WHERE token_hash = ? AND active = 1", (_hash(token or ""),))
        if not rows:
            return None
        r = rows[0]
        org = self.org(int(r["org_id"]))
        if not org.active:
            return None
        self._q("UPDATE users SET last_seen = ? WHERE id = ?", (time.time(), r["id"]), commit=True)
        return User(r["id"], org, r["label"])

    def revoke(self, user_id: int) -> None:
        self._q("UPDATE users SET active = 0 WHERE id = ?", (user_id,), commit=True)

    def users(self, org: str) -> list[dict[str, Any]]:
        o = self.org(org)
        return [dict(r) | {"token_hash": "…"} for r in self._q("SELECT id, label, active, created, last_seen FROM users WHERE org_id = ?", (o.id,))]

    # -- usage ----------------------------------------------------------------------------------

    def record(self, *, user: User | None, plan_id: str | None, task: str, stage: str, provider: str = "", model: str = "",
               status: str, reason: str = "", tokens_in: int = 0, tokens_out: int = 0, cost_usd: float = 0.0, image: bool = False,
               ms: int | None = None) -> None:  # fmt: skip
        self._q("INSERT INTO usage (ts, org_id, user_id, plan_id, task, stage, provider, model, status, reason, tokens_in, tokens_out, cost_usd, image, ms)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), user.org.id if user else None, user.id if user else None, plan_id, task, stage, provider, model, status,
                 reason[:200], tokens_in, tokens_out, cost_usd, int(image), ms), commit=True)  # fmt: skip

    def user_today(self, user: User) -> tuple[int, int]:
        """Model calls (not cached, not refused) and tokens of the user since 00:00 UTC."""
        start = time.time() - (time.time() % 86400)
        r = self._q("SELECT COUNT(*) n, COALESCE(SUM(tokens_in + tokens_out), 0) t FROM usage WHERE user_id = ? AND ts >= ? AND status = 'ok'",
                    (user.id, start))[0]  # fmt: skip
        return int(r["n"]), int(r["t"])

    def org_month_cost(self, org: Org) -> float:
        import calendar

        t = time.gmtime()
        start = calendar.timegm((t.tm_year, t.tm_mon, 1, 0, 0, 0))  # the 1st of this month, UTC
        return float(self._q("SELECT COALESCE(SUM(cost_usd), 0) c FROM usage WHERE org_id = ? AND ts >= ?", (org.id, start))[0]["c"])

    def plan_calls(self, plan_id: str) -> int:
        return int(self._q("SELECT COUNT(*) n FROM usage WHERE plan_id = ? AND status IN ('ok', 'cached')", (plan_id,))[0]["n"])

    def usage_summary(self, org: str, days: int = 30) -> dict[str, Any]:
        o = self.org(org)
        since = time.time() - days * 86400
        rows = self._q("SELECT status, provider, COUNT(*) n, SUM(tokens_in) ti, SUM(tokens_out) tout, SUM(cost_usd) c FROM usage"
                       " WHERE org_id = ? AND ts >= ? GROUP BY status, provider", (o.id, since))  # fmt: skip
        return {"org": o.name, "days": days, "month_cost_usd": round(self.org_month_cost(o), 4),
                "rows": [{k: r[k] for k in r.keys()} for r in rows]}  # fmt: skip

    # -- cache ----------------------------------------------------------------------------------

    def cached(self, key: str) -> dict[str, Any] | None:
        rows = self._q("SELECT * FROM cache WHERE key = ? AND created >= ?", (key, time.time() - CACHE_DAYS * 86400))
        return {"answer": json.loads(rows[0]["answer"]), "provider": rows[0]["provider"], "model": rows[0]["model"]} if rows else None

    def put_cache(self, key: str, answer: dict[str, Any], provider: str, model: str) -> None:
        self._q("INSERT OR REPLACE INTO cache (key, answer, provider, model, created) VALUES (?,?,?,?,?)",
                (key, json.dumps(answer, ensure_ascii=False), provider, model, time.time()), commit=True)  # fmt: skip


__all__ = ["CACHE_DAYS", "Org", "Store", "User", "db_path"]
