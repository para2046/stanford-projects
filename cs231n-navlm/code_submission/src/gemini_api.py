"""Gemini API client for NavLM v2.

One module owns the Gemini call. Two backends, picked by
`config.GEMINI_BACKEND`:

  - "vertex"   — Vertex AI (Gemini 2.5 Pro). OAuth via `gcloud`, billed
                 to the GCP project, so the Education credit applies and
                 Pro is reachable.
  - "aistudio" — the generativelanguage API-key endpoint. Free tier —
                 Flash works, Pro is `limit: 0`.

**Every call is logged** to `logs/gemini_api.jsonl` — one JSON line with
the input/output token counts, USD cost, finishReason and the **full
response text** — the inspectable conversation log.

429s are retried honouring the server `retryDelay`; an expired Vertex
token (401) is refreshed once.

    from src.gemini_api import call_gemini
    text = call_gemini(img, sys_prompt, user_msg, model=..., label=...)
"""

import base64
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# API call log — tokens in/out + full response, one JSON line per call.
API_LOG = config.REPO_ROOT / "logs" / "gemini_api.jsonl"

# USD per 1M tokens (input, output) — edit if Google changes pricing.
PRICE = {
    "gemini-2.5-flash":      (0.30,  2.50),
    "gemini-2.5-flash-lite": (0.10,  0.40),
    "gemini-2.5-pro":        (1.25, 10.00),
}

# ── Vertex auth ──────────────────────────────────────────────────────
_TOKEN = {"value": "", "fetched": 0.0}


def _access_token(force=False):
    """OAuth access token for Vertex AI.

    Resolution order:
      1. If env var ``GOOGLE_APPLICATION_CREDENTIALS`` points to a
         service-account JSON, use google-auth to load it directly
         (supports per-process credentials → safe for parallel runs).
      2. Otherwise fall back to ``gcloud auth print-access-token``
         (the historical behaviour; uses the ambient gcloud config).

    Tokens last ~1h; refreshes every 50 min (or on ``force``)."""
    if force or not _TOKEN["value"] or time.time() - _TOKEN["fetched"] > 3000:
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if sa_path and Path(sa_path).is_file():
            # service-account JSON — per-process auth, safe for parallel
            try:
                from google.oauth2 import service_account     # type: ignore
                from google.auth.transport.requests import Request  # type: ignore
                creds = service_account.Credentials.from_service_account_file(
                    sa_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"])
                creds.refresh(Request())
                if not creds.token:
                    raise RuntimeError("service-account creds refresh "
                                        "returned empty token")
                _TOKEN["value"], _TOKEN["fetched"] = creds.token, time.time()
                return _TOKEN["value"]
            except ImportError:
                # google-auth not installed → fall through to gcloud
                pass
        r = subprocess.run("gcloud auth print-access-token", shell=True,
                           capture_output=True, text=True)
        token = r.stdout.strip()
        if r.returncode != 0 or not token:
            raise RuntimeError("gcloud auth print-access-token failed: "
                               + (r.stderr.strip()[:200] or "no output"))
        _TOKEN["value"], _TOKEN["fetched"] = token, time.time()
    return _TOKEN["value"]


# ── pure helpers (unit-tested) ───────────────────────────────────────
def cost_usd(model, prompt_tokens, output_tokens):
    """USD cost of one call from its token counts. Pure."""
    pin, pout = PRICE.get(model, (1.25, 10.00))
    return prompt_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def retry_delay_s(err_body, default=30.0):
    """Seconds to back off, read from a 429 body's RetryInfo. Pure."""
    for d in (err_body.get("error", {}).get("details", []) or []):
        if str(d.get("@type", "")).endswith("RetryInfo"):
            s = str(d.get("retryDelay", "")).strip()
            if s.endswith("s"):
                try:
                    return float(s[:-1])
                except ValueError:
                    pass
    return default


def vertex_url(model, project, location):
    """Vertex AI generateContent REST URL. Pure."""
    host = ("aiplatform.googleapis.com" if location == "global"
            else f"{location}-aiplatform.googleapis.com")
    return (f"https://{host}/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:generateContent")


# ── call ─────────────────────────────────────────────────────────────
def _endpoint_and_auth(model, backend):
    """(url, headers, params) for the chosen backend."""
    if backend == "vertex":
        # env-var overrides for per-process parallel runs
        project = os.environ.get("GCP_PROJECT") or config.GCP_PROJECT
        location = (os.environ.get("VERTEX_LOCATION")
                    or config.VERTEX_LOCATION)
        url = vertex_url(model, project, location)
        return url, {"Authorization": f"Bearer {_access_token()}"}, {}
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY (see .env)")
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    return url, {}, {"key": key}


def _log(record):
    """Append one timestamped record to logs/gemini_api.jsonl (flushed)."""
    API_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
              **record}
    with API_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def call_gemini(image_path, sys_prompt, user_msg, model="gemini-2.5-pro",
                max_tokens=2048, temperature=0.3, timeout=180, retries=5,
                backend=None, label=""):
    """One single-image Gemini call. Returns the response text.

    Logs every attempt to `logs/gemini_api.jsonl` (tokens in/out, cost,
    finishReason, full response text, or the error). Retries 429s with
    the server-suggested delay and refreshes the Vertex token on 401;
    raises RuntimeError once `retries` are exhausted or on another HTTP
    error.
    """
    backend = backend or config.GEMINI_BACKEND
    p = Path(image_path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    img_b64 = base64.b64encode(p.read_bytes()).decode()
    payload = {
        "systemInstruction": {"parts": [{"text": sys_prompt}]},
        "contents": [{"role": "user", "parts": [
            {"text": user_msg},
            {"inlineData": {"mimeType": mime, "data": img_b64}},
        ]}],
        "generationConfig": {"temperature": temperature,
                             "maxOutputTokens": max_tokens},
    }

    last_err = ""
    for attempt in range(retries):
        url, headers, params = _endpoint_and_auth(model, backend)
        try:
            r = requests.post(url, json=payload, headers=headers,
                              params=params, timeout=timeout)
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            _log({"label": label, "backend": backend, "model": model,
                  "image": p.name, "attempt": attempt, "http_status": None,
                  "error": last_err})
            time.sleep(5 * (attempt + 1))
            continue

        data = r.json() if r.content else {}

        if r.status_code == 401 and backend == "vertex":
            _log({"label": label, "backend": backend, "model": model,
                  "image": p.name, "attempt": attempt, "http_status": 401,
                  "error": "token expired — refreshing gcloud token"})
            last_err = "401 — refreshed gcloud token"
            _access_token(force=True)
            continue

        if r.status_code == 429:
            delay = retry_delay_s(data) + 2.0
            last_err = f"429 rate-limited (waited {delay:.0f}s)"
            _log({"label": label, "backend": backend, "model": model,
                  "image": p.name, "attempt": attempt, "http_status": 429,
                  "retry_after_s": delay,
                  "error": str(data.get("error", {}).get("message", ""))[:400]})
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise RuntimeError(last_err)

        if r.status_code != 200:
            last_err = f"HTTP {r.status_code}"
            _log({"label": label, "backend": backend, "model": model,
                  "image": p.name, "attempt": attempt,
                  "http_status": r.status_code,
                  "error": (json.dumps(data)[:400] or r.text[:400])})
            raise RuntimeError(f"{last_err}: {json.dumps(data)[:200]}")

        # ── 200 OK ───────────────────────────────────────────────────
        usage = data.get("usageMetadata", {})
        ptok = usage.get("promptTokenCount", 0) or 0
        otok = usage.get("candidatesTokenCount", 0) or 0
        ttok = usage.get("totalTokenCount", 0) or 0
        # billed output = everything that is not prompt (incl. thinking)
        billed_out = max(otok, ttok - ptok)
        cands = data.get("candidates", [])
        finish = cands[0].get("finishReason") if cands else None
        parts = cands[0].get("content", {}).get("parts", []) if cands else []
        text = "".join(pt.get("text", "") for pt in parts).strip()
        _log({"label": label, "backend": backend, "model": model,
              "image": p.name, "attempt": attempt, "http_status": 200,
              "prompt_tokens": ptok, "output_tokens": otok,
              "total_tokens": ttok,
              "cost_usd": round(cost_usd(model, ptok, billed_out), 6),
              "finish_reason": finish,
              "prompt_feedback": data.get("promptFeedback"),
              "response": text})
        if not cands:
            raise RuntimeError(
                f"Gemini: no candidates ({data.get('promptFeedback')})")
        return text

    raise RuntimeError(last_err or "Gemini call failed")
