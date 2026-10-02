"""Decide which Claude tier a task needs, using Jev as the classifier.

Jev only classifies. The mapping from its answers to a model is local policy, and
the model names come from models/registry.json, never from Jev.
"""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx

JEV_URL = os.environ.get("JEV_URL", "https://openrouter.ai/api/alpha/decisions")
JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
JEV_TIMEOUT_S = float(os.environ.get("JEV_TIMEOUT_S", "3"))
MIN_CONFIDENCE = float(os.environ.get("JEV_MIN_CONFIDENCE", "0.6"))
SECURITY_THRESHOLD = float(os.environ.get("JEV_SECURITY_THRESHOLD", "0.7"))
MAX_TEXT_CHARS = 2000

REGISTRY = json.loads((Path(__file__).parent / "models" / "registry.json").read_text())
DEFAULT_TIER = REGISTRY["default_tier"]

QUESTIONS = {
    "complexity": {
        "type": "choice",
        "instructions": "How demanding is the software task the user is asking for?",
        "criteria": {
            "simple": "Small, well-defined: a quick question, a rename, a one-line or one-file tweak, formatting.",
            "standard": "Typical coding work: a feature, a bug fix, or a refactor in a few files.",
            "hard": "Needs deep reasoning: architecture or design, tricky debugging, large multi-file changes, subtle trade-offs.",
        },
    },
    "touches_security": {
        "type": "noul",
        "instructions": "Does the task involve security-sensitive work such as credentials, authentication, permissions, or deleting data?",
        "criteria": {
            "true": "It deals with secrets, auth, access control, or destructive operations.",
            "false": "It does not.",
        },
    },
}

_cache = {}  # task key -> decision, so one task keeps one model


def human_text(body: dict) -> Optional[str]:
    """The latest thing the human typed (not tool results, not injected reminders)."""
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts = [content]
        else:
            parts = [b.get("text", "") for b in content or [] if b.get("type") == "text"]
        text = "\n".join(parts)
        text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S).strip()
        if text:
            return text[:MAX_TEXT_CHARS]
    return None


def policy(answers: dict) -> tuple:
    """Map Jev's answers to (tier, reason). Anything unsure goes to the default."""
    c = answers.get("complexity") or {}
    choice, conf = c.get("choice"), c.get("confidence")
    if choice not in ("simple", "standard", "hard") or conf is None:
        return DEFAULT_TIER, "unreadable answer"
    if conf < MIN_CONFIDENCE:
        return DEFAULT_TIER, "low confidence (%.2f)" % conf
    tier = {"simple": "haiku", "standard": "sonnet", "hard": "opus"}[choice]
    sec = (answers.get("touches_security") or {}).get("noul")
    if tier == "haiku" and sec is not None and sec >= SECURITY_THRESHOLD:
        return "sonnet", "security-sensitive, not using haiku"
    return tier, "%s (%.2f)" % (choice, conf)


_key_cache = {}


def get_jev_key() -> Optional[str]:
    """OPENROUTER_API_KEY from the environment, else the macOS Keychain item
    'jev-openrouter' (a background job does not inherit shell variables).
    The value is never printed or logged."""
    key = os.environ.get("OPENROUTER_API_KEY") or _key_cache.get("k")
    if not key:
        try:
            out = subprocess.run(
                ["security", "find-generic-password", "-s", "jev-openrouter", "-w"],
                capture_output=True, text=True, timeout=5,
            )
            key = out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            key = None
        if key:
            _key_cache["k"] = key
    return key or None


async def ask_jev(text: str) -> dict:
    key = get_jev_key()
    if not key:
        raise RuntimeError("no Jev key found")
    async with httpx.AsyncClient(timeout=JEV_TIMEOUT_S) as client:
        r = await client.post(
            JEV_URL,
            headers={"Authorization": "Bearer " + key},
            json={"model": JEV_MODEL, "state": {"user_request": text}, "questions": QUESTIONS},
        )
        r.raise_for_status()
        return r.json()


async def _route(body: dict) -> Optional[dict]:
    """Return a decision for this request, or None if it shouldn't be routed."""
    if "haiku" in str(body.get("model", "")):
        return None  # Claude Code's own small background calls: leave alone
    text = human_text(body)
    if not text:
        return None
    forced = os.environ.get("JEV_FORCE_TIER")  # testing only: skip Jev, use this tier
    if forced in REGISTRY["tiers"]:
        return {"tier": forced, "model": REGISTRY["tiers"][forced]["model"],
                "reason": "forced for testing", "ms": 0, "cost": None, "cached": False}
    task_key = hashlib.sha256(text.encode()).hexdigest()[:16]
    if task_key in _cache:
        return dict(_cache[task_key], cached=True)

    t0 = time.time()
    try:
        data = await ask_jev(text)
        tier, reason = policy(data.get("answers") or {})
        cost = (data.get("usage") or {}).get("cost")
    except Exception as e:  # fail open: never block the user's request
        detail = type(e).__name__
        if isinstance(e, httpx.HTTPStatusError):
            detail = "HTTP %d: %s" % (e.response.status_code, e.response.text[:200].replace("\n", " "))
        tier, reason, cost = DEFAULT_TIER, "jev failed: " + detail, None
    decision = {
        "tier": tier,
        "model": REGISTRY["tiers"][tier]["model"],
        "reason": reason,
        "ms": int((time.time() - t0) * 1000),
        "cost": cost,
    }
    if len(_cache) > 256:
        _cache.clear()
    _cache[task_key] = decision
    return dict(decision, cached=False)


def adapt_for_model(body: dict, tier: str) -> None:
    """Make a request that was written for another model valid for this tier."""
    strip_per_turn_effort(body, tier)
    fold_system_messages(body, tier)
    limit = REGISTRY["tiers"][tier].get("max_output_tokens")
    if limit and (body.get("max_tokens") or 0) > limit:
        body["max_tokens"] = limit
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("budget_tokens", 0) >= body.get("max_tokens", 0):
        thinking["budget_tokens"] = max(1024, body["max_tokens"] - 1)
    if (
        isinstance(body.get("thinking"), dict)
        and body["thinking"].get("type") == "adaptive"
        and REGISTRY["tiers"][tier].get("supports_adaptive_thinking") is False
    ):
        del body["thinking"]
        cm = body.get("context_management")
        if isinstance(cm, dict):  # clear_thinking edits need thinking to be on
            cm["edits"] = [e for e in cm.get("edits") or [] if not str(e.get("type", "")).startswith("clear_thinking")]
            if not cm["edits"]:
                del body["context_management"]
    if REGISTRY["tiers"][tier].get("supports_effort") is False:
        body.pop("effort", None)
        oc = body.get("output_config")
        if isinstance(oc, dict):
            oc.pop("effort", None)
            if not oc:
                del body["output_config"]


def adapt_beta_header(value: str, tier: str) -> str:
    """Remove anthropic-beta features this tier doesn't support."""
    drop = REGISTRY["tiers"][tier].get("drop_betas") or []
    keep = [b for b in value.split(",") if not any(b.strip().startswith(d) for d in drop)]
    return ",".join(keep)


def strip_per_turn_effort(body: dict, tier: str) -> None:
    """Per-turn effort rides on individual messages; remove it where unsupported."""
    if REGISTRY["tiers"][tier].get("supports_effort") is False:
        for msg in body.get("messages") or []:
            if isinstance(msg, dict):
                msg.pop("output_config", None)


def fold_system_messages(body: dict, tier: str) -> None:
    """Turn in-conversation role:'system' messages into system-reminder user text."""
    if REGISTRY["tiers"][tier].get("supports_system_role_messages") is not False:
        return
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            blocks = []
            for b in content or []:
                if isinstance(b, dict) and b.get("type") == "text":
                    blocks.append({"type": "text", "text": "<system-reminder>\n" + b.get("text", "") + "\n</system-reminder>"})
            msg["role"] = "user"
            msg["content"] = blocks or [{"type": "text", "text": "<system-reminder></system-reminder>"}]


TIER_ORDER = ["haiku", "sonnet", "opus"]


async def route(body: dict) -> Optional[dict]:
    """Jev's decision, bumped up a tier if the conversation has outgrown it.

    The decision is made once per task, but the conversation keeps growing, so the
    size check runs on every request.
    """
    decision = await _route(body)
    if not decision:
        return None
    est_tokens = len(json.dumps(body)) // 3  # rough and on the cautious side
    decision = dict(decision, est_tokens=est_tokens)
    tier = decision["tier"]
    while True:
        limit = REGISTRY["tiers"][tier].get("max_input_tokens")
        i = TIER_ORDER.index(tier)
        if limit and est_tokens > 0.75 * limit and i + 1 < len(TIER_ORDER):
            tier = TIER_ORDER[i + 1]
        else:
            break
    if tier != decision["tier"]:
        decision = dict(
            decision,
            tier=tier,
            model=REGISTRY["tiers"][tier]["model"],
            reason="conversation too long for %s (~%dk tokens), using %s"
            % (decision["tier"], est_tokens // 1000, tier),
        )
    return decision
