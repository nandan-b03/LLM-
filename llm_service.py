# -*- coding: utf-8 -*-
"""
llm_service.py
--------------
Core LLM provider abstraction.
Supports Google Gemini (primary) and OpenAI.
Uses Python's standard library (urllib.request / json) for HTTP requests,
ensuring 100% compatibility across all Python versions (including Python 3.7)
without OpenSSL / urllib3 compatibility issues.
"""

import os
import json
import logging
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _get_provider() -> str:
    """Return which LLM provider is configured."""
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


def _call_gemini(prompt: str, system: str = "") -> str:
    """Call Google Gemini API using urllib (safe for Python 3.7+)."""
    api_key = os.environ["GEMINI_API_KEY"].strip()
    primary_model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    
    # Try primary model, with fallback to other modern flash models
    model_candidates = [primary_model, "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash"]
    
    payload_dict = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    if system:
        payload_dict["systemInstruction"] = {"parts": [{"text": system}]}
    
    payload_bytes = json.dumps(payload_dict).encode("utf-8")
    last_err = None

    for model_name in model_candidates:
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}".format(
            model_name, api_key
        )
        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            logger.warning("Gemini model %s returned HTTP %s: %s", model_name, e.code, err_body)
            last_err = e
            # If 404 or 503, try next candidate
            if e.code in (404, 503):
                continue
            raise
        except Exception as e:
            logger.warning("Gemini request error on %s: %s", model_name, e)
            last_err = e
            continue

    if last_err:
        raise last_err
    raise RuntimeError("All Gemini models failed")


def _call_openai(prompt: str, system: str = "") -> str:
    """Call OpenAI-compatible API using urllib."""
    api_key = os.environ["OPENAI_API_KEY"].strip()
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload_dict = {
        "model": model_name,
        "messages": messages
    }
    payload_bytes = json.dumps(payload_dict).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload_bytes,
        headers={
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(prompt: str, system: str = "") -> Optional[str]:
    """
    Send a prompt to the configured LLM and return the text response.
    Returns None on failure so callers can show a graceful fallback.
    """
    provider = _get_provider()
    try:
        if provider == "gemini":
            return _call_gemini(prompt, system)
        elif provider == "openai":
            return _call_openai(prompt, system)
        else:
            logger.warning("No LLM API key configured.")
            return None
    except Exception as exc:
        logger.error("LLM call failed (%s): %s", provider, exc)
        return None


def is_llm_available() -> bool:
    """Check whether any LLM provider is configured."""
    return _get_provider() != "none"


def get_provider_name() -> str:
    """Return a human-readable provider name for UI display."""
    p = _get_provider()
    return {"gemini": "Google Gemini", "openai": "OpenAI", "none": "Offline (no API key)"}.get(p, p)
