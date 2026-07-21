"""
Shared multi-turn agent runner for the EI-Agent-6 batch.

Mirrors the proven DeputyBench pattern (standard tool_use/tool_result protocol,
cross-provider via direct_backend block translation). Each *direction* supplies an
Episode object; run_agent drives the tool loop and the Episode's deterministic
score() is the oracle. No LLM judge on the primary axis anywhere in this batch.

Episode protocol
----------------
    ep.system : str                     system prompt
    ep.user0  : str                     first user message
    ep.tools  : list[dict]              Anthropic-style tool specs
    ep.handle(name, inp) -> (str, bool) tool result text, and done? (True ends episode)
    ep.score() -> dict                  deterministic oracle over ep.log / internal state
    ep.log    : list[(name, inp)]       appended by the runner (also ep.final_text set)
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
import direct_backend as db  # noqa: E402


def _mk_tool_result(prov, block, text):
    tr = {"type": "tool_result", "tool_use_id": block["id"], "content": text}
    if prov == "gemini":
        tr["_name"] = block["name"]
    return tr


def _first_tool(blocks):
    for b in blocks:
        if b.get("type") == "tool_use":
            return b
    return None


def run_agent(complete, model, ep, max_turns=12):
    """Drive one episode. Returns ep.score() plus bookkeeping."""
    prov = getattr(complete, "provider", "anthropic")
    messages = [{"role": "user", "content": ep.user0}]
    ep.log = []
    ep.final_text = ""
    ep.turns = 0
    ep.no_tool = False
    for turn in range(max_turns):
        ep.turns = turn + 1
        r = complete(messages=messages, system=ep.system, tools=ep.tools, model=model)
        blocks = r["content"]
        tu = _first_tool(blocks)
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if text:
            ep.final_text = text
        # record assistant turn: text + at most the first tool_use (keeps protocol clean)
        asst = [b for b in blocks if b.get("type") == "text"]
        if tu:
            asst = asst + [tu]
        if not asst:
            break
        messages.append({"role": "assistant", "content": asst})
        if tu is None:
            ep.no_tool = True
            break  # model answered in prose without a tool -> episode ends
        name, inp = tu["name"], tu.get("input", {}) or {}
        ep.log.append((name, inp))
        result_text, done = ep.handle(name, inp)
        messages.append({"role": "user",
                         "content": [_mk_tool_result(prov, tu, result_text)]})
        if done:
            break
    out = ep.score()
    out["_turns"] = ep.turns
    out["_no_tool"] = int(ep.no_tool)
    return out


def make_backends(providers=("anthropic", "openai", "gemini"), temperature=0.0, max_tokens=600):
    """Return {provider: complete} for the requested providers (skips missing keys)."""
    import env_keys
    env_keys.load_keys()
    out = {}
    for p in providers:
        try:
            out[p] = db.make_complete(provider=p, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            print(f"[skip {p}] {type(e).__name__}: {e}")
    return out
