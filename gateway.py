"""Local Anthropic-compatible gateway that routes Claude Code requests with Jev.

Everything Claude Code sends is forwarded to Anthropic. Only POST /v1/messages is
inspected, and only when the mode is "shadow" or "on".

Modes (switch live with jevctl or the /jev command):
  off    - pure pass-through, no Jev call
  shadow - ask Jev and log the decision, change nothing
  on     - ask Jev and switch the model
"""
import json
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from routing import adapt_beta_header, adapt_for_model, get_jev_key, route

UPSTREAM = "https://api.anthropic.com"
STATE_DIR = Path(os.environ.get("JEV_STATE_DIR", str(Path.home() / ".jev-gateway")))
MODE_FILE = STATE_DIR / "mode"
LOG_FILE = STATE_DIR / "decisions.jsonl"
MODES = ("off", "shadow", "on")

# Request headers we pass to Anthropic, and response headers we pass back.
# Nothing else is forwarded, and none of these are ever logged.
SEND_EXACT = {"authorization", "x-api-key", "content-type", "accept", "user-agent"}
BACK_EXACT = {"content-type", "request-id", "retry-after", "cache-control", "x-should-retry"}

app = FastAPI()


def load_mode() -> str:
    env = os.environ.get("JEV_ROUTE_MODE", "").lower()
    if env in MODES:
        return env
    try:
        saved = MODE_FILE.read_text().strip()
        if saved in MODES:
            return saved
    except OSError:
        pass
    return "off"


state = {"mode": load_mode()}


def request_headers(request: Request) -> dict:
    out = {}
    for name, value in request.headers.items():
        n = name.lower()
        if n.startswith("anthropic-") or n in SEND_EXACT:
            out[n] = value
    # httpx asks for gzip by default and we relay raw bytes without the
    # content-encoding header, so ask Anthropic for plain bytes.
    out["accept-encoding"] = "identity"
    return out


def log_decision(**fields) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(dict(ts=int(time.time()), **fields)) + "\n")
    except OSError:
        pass


async def call_upstream(method, path, params, body, headers, fallback=None):
    """Open the upstream call first so its real status can be passed back.

    fallback = (body, headers) of the original request, retried once if a
    switched request is rejected.
    """
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    def send(content, hdrs):
        return client.send(
            client.build_request(method, UPSTREAM + path, params=params, content=content or None, headers=hdrs),
            stream=True,
        )

    try:
        upstream = await send(body, headers)
        if fallback is not None and upstream.status_code in (400, 404, 422):
            err = (await upstream.aread()).decode(errors="replace")[:400]
            print("[route] switched request rejected (%d): %s" % (upstream.status_code, err))
            print("[route] retrying original model")
            log_decision(event="fallback", status=upstream.status_code, error=err)
            await upstream.aclose()
            upstream = await send(*fallback)
    except httpx.HTTPError as e:
        await client.aclose()
        print("Upstream connection failed:", type(e).__name__)
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": "Gateway could not reach Anthropic (" + type(e).__name__ + ")"},
            },
        )

    async def close_upstream():
        await upstream.aclose()
        await client.aclose()

    back = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() in BACK_EXACT or k.lower().startswith("anthropic-")
    }
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=back,
        background=BackgroundTask(close_upstream),
    )


# ---- health and control (localhost only; a web page cannot call these) ----

@app.get("/health")
async def health():
    return {"ok": True}


def control_forbidden(request: Request):
    # A custom header forces browsers to do a CORS preflight we never answer,
    # and a real Origin header means the call came from a web page.
    if request.headers.get("x-jev-control") != "1" or request.headers.get("origin"):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return None


@app.get("/jev/status")
async def jev_status(request: Request):
    denied = control_forbidden(request)
    if denied:
        return denied
    return {"mode": state["mode"], "jev_key_found": bool(get_jev_key()), "log": str(LOG_FILE)}


@app.post("/jev/mode")
async def jev_mode(request: Request):
    denied = control_forbidden(request)
    if denied:
        return denied
    try:
        mode = str((await request.json()).get("mode", "")).lower()
    except Exception:
        mode = ""
    if mode not in MODES:
        return JSONResponse({"error": "mode must be one of " + ", ".join(MODES)}, status_code=400)
    if mode != "off" and not get_jev_key():
        return JSONResponse({"error": "no Jev key found (OPENROUTER_API_KEY or Keychain item jev-openrouter)"}, status_code=409)
    state["mode"] = mode
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        MODE_FILE.write_text(mode)
    except OSError:
        pass
    print("[jev] mode ->", mode)
    return {"mode": mode}


# ---- the one request we route ----

@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.body()
    headers = request_headers(request)
    fallback = None
    mode = state["mode"]

    if mode != "off":
        try:
            parsed = json.loads(body)
            decision = await route(parsed)
        except Exception as e:  # routing must never break the request
            parsed, decision = None, None
            print("[route] error:", type(e).__name__)
        if decision:
            asked = parsed.get("model")
            applied = mode == "on"
            print(
                "[route] %s | asked=%s -> %s%s | %s | %dms%s%s"
                % (
                    mode,
                    asked,
                    decision["model"],
                    "" if applied else " (not applied)",
                    decision["reason"],
                    decision["ms"],
                    " | cost $%.6f" % decision["cost"] if decision["cost"] else "",
                    " | cached" if decision["cached"] else "",
                )
            )
            log_decision(
                event="route", mode=mode, asked=asked, chosen=decision["model"], tier=decision["tier"],
                reason=decision["reason"], jev_ms=decision["ms"], jev_cost=decision["cost"],
                cached=decision["cached"], applied=applied, est_tokens=decision.get("est_tokens"),
            )
            if applied:
                fallback = (body, headers)
                parsed["model"] = decision["model"]
                adapt_for_model(parsed, decision["tier"])
                body = json.dumps(parsed).encode()
                headers = dict(headers)
                if "anthropic-beta" in headers:
                    headers["anthropic-beta"] = adapt_beta_header(headers["anthropic-beta"], decision["tier"])

    return await call_upstream("POST", "/v1/messages", dict(request.query_params), body, headers, fallback)


# ---- everything else Claude Code calls goes straight through ----

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])
async def passthrough(path: str, request: Request):
    if ".." in path.split("/"):
        return JSONResponse({"error": "bad path"}, status_code=400)
    body = await request.body()
    return await call_upstream(
        request.method, "/" + path, dict(request.query_params), body, request_headers(request)
    )
