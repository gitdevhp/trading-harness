"""Shared OpenAI-compatible chat client. Same server your solver script
uses (local vLLM by default) — override via env vars if you want the
Critic/Consolidator on a different (e.g. cheaper) model than the Solver.
"""
import os
from openai import OpenAI

API_BASE_URL = os.environ.get("ACE_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("ACE_LLM_API_KEY", "EMPTY")

SOLVER_MODEL = os.environ.get("ACE_SOLVER_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
DEBATER_MODEL = os.environ.get("ACE_DEBATER_MODEL", SOLVER_MODEL)
CONSOLIDATOR_MODEL = os.environ.get("ACE_CONSOLIDATOR_MODEL", SOLVER_MODEL)

_client = None

# Cumulative token usage across all chat() calls in this process.
_token_counts = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _client


def get_token_counts() -> dict:
    """Return a copy of cumulative token counts since process start (or last reset)."""
    return dict(_token_counts)


def reset_token_counts():
    """Reset counters to zero (call between backtests if running multiple in one process)."""
    _token_counts.update({"prompt": 0, "completion": 0, "total": 0, "calls": 0})


def chat(model, messages, temperature=0.0, max_tokens=700, stop=None):
    _env_temp = os.environ.get("ACE_TEMPERATURE")
    if _env_temp is not None:
        temperature = float(_env_temp)
    response = get_client().chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stop=stop,
    )
    # Accumulate token usage if the server reports it (vLLM always does).
    usage = getattr(response, "usage", None)
    if usage is not None:
        _token_counts["prompt"]     += getattr(usage, "prompt_tokens", 0) or 0
        _token_counts["completion"] += getattr(usage, "completion_tokens", 0) or 0
        _token_counts["total"]      += getattr(usage, "total_tokens", 0) or 0
    _token_counts["calls"] += 1
    msg = response.choices[0].message
    # Primary: content field.
    text = (msg.content or "").strip()
    # Fallback for vLLM builds that use --reasoning-parser and put the actual
    # response in reasoning_content (vLLM ≥0.6) or reasoning (older builds).
    if not text:
        text = (getattr(msg, "reasoning_content", None) or "").strip()
    if not text:
        text = (getattr(msg, "reasoning", None) or "").strip()
    # Qwen3 models emit <think>...</think> blocks when reasoning-parser is NOT
    # used (tokens appear inline in content). Strip the thinking block and keep
    # only the response that follows it.
    if "<think>" in text:
        import re as _re
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
    return text
