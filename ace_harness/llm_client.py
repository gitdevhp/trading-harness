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


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _client


def chat(model, messages, temperature=0.0, max_tokens=700, stop=None):
    response = get_client().chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stop=stop,
    )
    msg = response.choices[0].message
    # Qwen3.6 with --reasoning-parser qwen3 may place the response in the
    # reasoning field and leave content=None for some output modes.
    text = (msg.content or "").strip()
    if not text:
        text = (getattr(msg, "reasoning", None) or "").strip()
    return text
