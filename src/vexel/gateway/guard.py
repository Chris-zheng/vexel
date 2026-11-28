"""What may reach a language model through the gateway, and what may come back.

**The goal** (what the person wants measured, at most ``GOAL_MAX`` characters) is checked before
any model sees it, in this order, and refused with a category:

1. ``abuse``: too long, or asking for something other than a plan (explanations, essays,
   image descriptions, many alternatives, long answers);
2. ``injection``: attempts to change the model's instructions or role;
3. ``exfiltration``: asking for the system prompt, keys, configuration, other users' data, or
   which model or budget is behind the service;
4. ``off_topic``: writing, translation, coding, news, general questions - also when a
   measurement word is used as cover;
5. ``off_topic``: no image-analysis vocabulary at all in a goal of more than a few characters.

**The context** comes from the desktop's planner (numbers from the image's fingerprint, the
rule-based draft, trial results). It is cut to the fields each task has, to ``CONTEXT_MAX``
bytes as JSON; strings are shortened to ``STRING_MAX`` characters and checked like the goal
(a client could hide instructions in them).

**The answer** must be a JSON object matching the task's schema; settings are kept only from
the allowlist built on the server from Vexel's own recipe definitions; an answer flagged
``out_of_scope`` by the model is a refusal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

GOAL_MAX = 200
STRING_MAX = 300
CONTEXT_MAX = 16_000
REASON_MAX = 160  # characters of each note in an answer

ABUSE = [
    re.compile(p, re.I)
    for p in (
        r"(详细|具体|一步步)?(解释|讲解|介绍|说明)(一下)?.{0,20}(原理|理论|概念|是什么|为什么)",
        r"(explain|describe|tell me about|teach me).{0,40}(principle|theory|how .{0,20} works|what is|in detail)",
        r"(describe|caption|summari[sz]e|write up).{0,30}(this )?(image|picture|figure|photo)",
        r"(描述|总结|概括|写).{0,10}(这张|该)?(图|图片|照片).{0,10}(内容|说明|图注|标题)",
        r"\b(\d{2,}|many|several|all possible)\s+(alternative|different|other)\s+(workflows?|plans?|options?)",
        r"(long|lengthy|detailed|\d{3,}[- ]?word)\s+(answer|explanation|essay|text|report)",
        r"(\d{3,}\s*字|长篇|详细的(文章|报告|说明))",
        r"(不要|别).{0,4}(方案|计划).{0,10}(直接|而是|告诉|教)",
        r"(\d+|几十|很多|多)\s*种.{0,12}(方案|方法|流程).{0,12}(解释|比较|说明|介绍)|逐一(解释|说明|比较)",
        r"(原理|机理)是什么|是什么原理|(怎么|如何)工作|工作原理|为什么会.{0,20}(原理|机理)?",
        r"\bexplain\b|why (does|do|is|are) .{0,30}(matter|happen|work)",
        r"(output|write|put).{0,30}(in|into) the (reasons?|notes?|caution) field",
        r"(reasons?|notes?|caution)\s*(字段|field).{0,20}(输出|写|生成)",
    )
]
INJECTION = [
    re.compile(p, re.I)
    for p in (
        r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|preceding|system)\b",
        r"forget\s+(all\s+)?(your|the|previous|prior)\s+(instructions?|rules?|prompts?)",
        r"disregard\s+(all\s+)?(the\s+)?(prior|previous|above|system)",
        r"(you are|you're|act as|pretend to be|from now on you)\s+(now\s+)?(a|an|in)?\s*(general|unrestricted|chatgpt|dan|assistant|developer mode|jailbroken)",
        r"developer\s*mode|jailbreak|bypass\s+(the\s+)?(safety|filter|rules)|prompt\s+injection",
        r"</?\s*(system|assistant|user|goal|instructions?)\s*>",
        r"(new|updated|override)\s+(system\s+)?(rules?|instructions?|prompt)\s*:",
        r"系统提示\s*[:：]|新的?(规则|指令)\s*[:：]",
        r"忽略(之前|以上|上面|所有|前面)(的)?(全部|所有)?(指令|规则|提示|要求)",
        r"(你现在是|现在你是|扮演|假装你是).{0,10}(助手|通用|chatgpt|任何)",
        r"(允许|可以)(回答|讨论)(任何|所有)(问题|话题)|越狱|绕过(限制|规则|安全)",
        r"answer\s+(freely|anything)|no\s+(restrictions|limits|rules)",
    )
]
EXFILTRATION = [
    re.compile(p, re.I)
    for p in (
        r"(system|hidden|initial)\s+prompt|your\s+(instructions|rules|configuration|config)",
        r"api[\s_-]?keys?|secret|password|token\s+budget|credentials?",
        r"(which|what)\s+(model|llm|provider|ai)\s+(are you|is this|do you use|powers)",
        r"(gateway|server)\s+(config|configuration|logs?)|other\s+users?'?s?\s+(requests?|data|images?)",
        r"系统提示词?|提示词原样|(你的|使用的).{0,6}(密钥|口令|配置)|api\s*密钥|(其他|别的|其它)(用户|人|公司|组织)",
        r"(你是|你用的是).{0,6}(哪个|什么)(模型|大模型|服务商)",
    )
]
OFF_TOPIC = [
    re.compile(p, re.I)
    for p in (
        r"(write|compose|draft|tell)\s+(me\s+)?.{0,20}\b(poem|story|essay|novel|email|letter|resume|cv|cover letter|song|joke|speech)\b",
        r"\b(translate|translation)\b",
        r"(weather|stock|bitcoin|crypto|news|recipe for|horoscope|lottery|football|movie)",
        r"(capital of|who is|who was|when did|what year|meaning of life)",
        r"\b(python|javascript|java|sql|html|css|c\+\+|regex)\b.{0,30}(code|script|program|function|crawler|scraper)",
        r"(code|script|program|function|crawler|scraper).{0,20}\b(python|javascript|java|sql)\b",
        r"(homework|leetcode|interview question|quantum mechanics|philosophy)",
        r"(翻译|写一?(封|篇|首|个|份)|作文|诗|小说|故事|邮件|简历|笑话|年终总结|周报|演讲稿|文案)",
        r"(天气|股票|比特币|新闻|菜谱|做法|烹饪|红烧|星座|算命|彩票|旅游攻略|电影)",
        r"(爬虫|写代码|编程|程序|脚本|量子力学|哲学|作业)",
        r"(首都|是谁|哪一年)",
    )
]
DOMAIN = re.compile(
    r"(measur|cell|nucle|droplet|microfluid|\bsem\b|\btem\b|pcb|trace|\bpads?\b|particle|pore|porosit|fib(re|er)|\bline|\bcd\b|ler|lwr|"
    r"hole|via|layer|thick|sidewall|profile|trench|\btip|apex|radius|roughness|scale|calibrat|segment|threshold|denois|destrip|contrast|"
    r"background|area|diameter|width|size|count|distribution|volume|fluoresc|ddpcr|positive|encapsul|poisson|frequen|flow|channel|"
    r"defect|grain|coverage|spacing|pitch|edge|angle|"
    r"细胞|核|液滴|微流控|电镜|线宽|线条|孔|颗粒|粒径|纤维|层|膜厚|侧壁|截面|尖端|曲率|粗糙|分割|阈值|去噪|条纹|背景|对比度|标定|比例尺|"
    r"面积|直径|宽度|尺寸|数量|计数|分布|体积|荧光|阳性|包裹|泊松|频率|流速|通道|缺陷|晶粒|覆盖|间距|周期|边缘|角度|测|量|区域|图|视野|region|roi|image)",
    re.I,
)


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    category: str = ""  # abuse | injection | exfiltration | off_topic
    reason: str = ""


def check_goal(goal: str) -> Verdict:
    """Whether a goal may go to a model (see the module docstring for the order)."""
    g = (goal or "").strip()
    if not g:
        return Verdict(True)
    if len(g) > GOAL_MAX:
        return Verdict(False, "abuse", f"目标超过 {GOAL_MAX} 个字符：请用一句话写想测什么 / goal over {GOAL_MAX} characters")
    for cat, pats in (("abuse", ABUSE), ("injection", INJECTION), ("exfiltration", EXFILTRATION), ("off_topic", OFF_TOPIC)):
        if any(p.search(g) for p in pats):
            return Verdict(False, cat, REFUSAL[cat])
    if len(g) > 6 and not DOMAIN.search(g):
        return Verdict(False, "off_topic", REFUSAL["off_topic"])
    return Verdict(True)


REFUSAL = {
    "abuse": "这里只能规划图像测量（选方法和设置），不提供解释、长文或其他回答 / only measurement plans here, no explanations or long answers",
    "injection": "请求含有改变规划服务规则的内容，已拒绝 / the request tries to change the service's rules: refused",
    "exfiltration": "规划服务不提供配置、密钥或其他用户的信息 / no configuration, keys or other users' data",
    "off_topic": "这里只能规划图像测量；请写想在这张图上测什么 / only image-measurement planning: say what to measure in this image",
}

# fields each task's context may carry (the desktop's planners send these)
FIELDS = {
    "sem": {"task", "stage", "goal", "image", "draft", "attempts", "valid_keys"},
    "droplet": {"task", "stage", "goal", "fingerprint", "draft", "trials", "valid_keys"},
}
STAGES = {"draft", "check", "revise"}


def _clean(value: Any, depth: int = 0) -> Any:
    """Numbers, booleans and None as they are; strings shortened; lists and dicts bounded."""
    if depth > 6:
        return None
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value[:STRING_MAX]
    if isinstance(value, (list, tuple)):
        return [_clean(v, depth + 1) for v in list(value)[:60]]
    if isinstance(value, dict):
        return {str(k)[:60]: _clean(v, depth + 1) for k, v in list(value.items())[:80]}
    return str(value)[:STRING_MAX]


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)] + [k for k in value if isinstance(k, str)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def sanitize_context(context: dict[str, Any], allowed_keys: set[str]) -> tuple[dict[str, Any] | None, Verdict]:
    """The context cut to what its task may carry, or a refusal."""
    if not isinstance(context, dict):
        return None, Verdict(False, "abuse", "context must be an object")
    task = "sem" if context.get("task") == "sem" else "droplet"
    stage = context.get("stage")
    if stage not in STAGES:
        return None, Verdict(False, "abuse", f"unknown stage {stage!r}")
    ctx = {k: _clean(v) for k, v in context.items() if k in FIELDS[task]}
    ctx["task"] = task
    v = check_goal(str(ctx.get("goal") or ""))
    if not v.allowed:
        return None, v
    # instructions hidden in other fields: the same checks (goal length aside)
    for s in _strings({k: x for k, x in ctx.items() if k != "goal"}):
        for cat, pats in (("injection", INJECTION), ("exfiltration", EXFILTRATION)):
            if any(p.search(s) for p in pats):
                return None, Verdict(False, cat, REFUSAL[cat])
    ctx["valid_keys"] = sorted(set(ctx.get("valid_keys") or []) & allowed_keys)
    if len(json.dumps(ctx, ensure_ascii=False, default=str)) > CONTEXT_MAX:
        return None, Verdict(False, "abuse", f"context over {CONTEXT_MAX} bytes")
    return ctx, Verdict(True)


def check_answer(text: str, task: str, allowed: dict[str, tuple], valid_keys: set[str]) -> tuple[dict[str, Any] | None, str]:
    """The model's answer as a validated object, or why it is rejected. Settings outside the
    allowlist are dropped (and said so), notes shortened."""
    from vexel.agent.models import check_settings

    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return None, "answer is not JSON"
    if not isinstance(data, dict):
        return None, "answer is not an object"
    if data.get("out_of_scope") is True:
        return None, "out_of_scope"
    settings = data.get("settings") or []
    if not isinstance(settings, list):
        return None, "settings is not a list"
    raw = {str(s.get("key")): s.get("value") for s in settings if isinstance(s, dict) and "key" in s}
    reasons = {str(s.get("key")): str(s.get("reason", ""))[:REASON_MAX] for s in settings if isinstance(s, dict) and "key" in s}
    kept, dropped = check_settings(raw, valid_keys, allowed)
    out: dict[str, Any] = {
        "settings": [{"key": k, "value": v, "reason": reasons.get(k, "")} for k, v in kept.items()],
        "notes": [str(n)[:REASON_MAX] for n in (data.get("notes") or [])[:6] if isinstance(n, (str, int, float))],
        "dropped": dropped,
    }  # fmt: skip
    if task == "sem":
        from vexel.agent.models import SEM_SCHEMA

        recipe = data.get("recipe")
        out["recipe"] = recipe if recipe in SEM_SCHEMA["properties"]["recipe"]["enum"] else None
    else:
        for k in ("cells", "fluorescence"):
            out[k] = data.get(k) if isinstance(data.get(k), bool) else None
    return out, ""


__all__ = ["CONTEXT_MAX", "GOAL_MAX", "REFUSAL", "Verdict", "check_answer", "check_goal", "sanitize_context"]
