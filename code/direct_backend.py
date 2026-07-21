"""
Direct-API backend for the CoA harness — bypasses host.llm (and its multi-turn
cache_control:global 400) by calling the official Anthropic / OpenAI SDKs directly.

Exposes make_complete(provider=None) -> complete(messages, system, tools, model)
returning {"content":[blocks]} in the same shape the harness expects, so the SAME
coa_run / coa_score work unchanged.

Key discovery order (first present wins unless provider= forces one):
  ANTHROPIC_API_KEY -> 'anthropic'
  OPENAI_API_KEY    -> 'openai'

Model-id aliases let one grid string map to each provider's catalog:
  'small' / 'mid' / 'large' -> provider-specific tiers (edit LADDER below).
"""
from __future__ import annotations
import os, time, json

LADDER = {
    "anthropic": {
        "small": "claude-haiku-4-5",
        "mid":   "claude-sonnet-4-5-20250929",
        "large": "claude-opus-4-5-20251101",
    },
    "openai": {
        "small": "gpt-4o-mini",
        "mid":   "gpt-4o",
        "large": "gpt-4.1",
    },
    "gemini": {
        "small": "gemini-2.5-flash-lite",
        "mid":   "gemini-2.5-flash",
        "large": "gemini-2.5-pro",
    },
    # Open-weight capability ladder via OpenRouter (for the reproduction + activation-probe study).
    # Same-family (Qwen2.5) ladder controls tokenizer/training-recipe, capability as the IV.
    "openrouter": {
        "small": "qwen/qwen-2.5-7b-instruct",
        "mid":   "qwen/qwen-2.5-72b-instruct",
        "large": "meta-llama/llama-3.3-70b-instruct",
        # cross-family open probes:
        "qwen7b":  "qwen/qwen-2.5-7b-instruct",
        "qwen72b": "qwen/qwen-2.5-72b-instruct",
        "llama8b": "meta-llama/llama-3.1-8b-instruct",
        "llama70b":"meta-llama/llama-3.3-70b-instruct",
        "mistral": "mistralai/mistral-small-24b-instruct-2501",
    },
}

def detect_provider(provider=None):
    if provider: return provider
    if os.environ.get("ANTHROPIC_API_KEY"): return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):    return "openai"
    if os.environ.get("GEMINI_API_KEY"):    return "gemini"
    if os.environ.get("OPENROUTER_API_KEY"): return "openrouter"
    raise RuntimeError("No API key found. Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / "
                       "OPENROUTER_API_KEY (Customize -> Credentials).")

def resolve_model(provider, model):
    """Accept either a ladder alias ('small'/'mid'/'large') or a literal model id."""
    return LADDER.get(provider, {}).get(model, model)

# ---------------- Anthropic ----------------
def _anthropic_complete(client, messages, system, tools, model, max_tokens, temperature):
    kw = dict(model=model, max_tokens=max_tokens, temperature=temperature, messages=messages)
    if system: kw["system"] = system
    if tools:
        kw["tools"] = [{"name":t["name"],"description":t.get("description",""),
                        "input_schema":t["input_schema"]} for t in tools]
    resp = client.messages.create(**kw)
    # normalize to [{"type":"text"/"tool_use",...}]
    blocks=[]
    for b in resp.content:
        if b.type=="text": blocks.append({"type":"text","text":b.text})
        elif b.type=="tool_use": blocks.append({"type":"tool_use","id":b.id,"name":b.name,"input":b.input})
    return {"content":blocks}

# ---------------- OpenAI ----------------
def _openai_complete(client, messages, system, tools, model, max_tokens, temperature):
    # translate Anthropic-style blocks to OpenAI chat-completions tool-calling (multi-turn)
    omsgs=[]
    if system: omsgs.append({"role":"system","content":system})
    for m in messages:
        c=m["content"]; role=m["role"]
        if isinstance(c,str):
            omsgs.append({"role":role,"content":c}); continue
        if role=="assistant":
            # split into text + tool_calls
            text="".join(b.get("text","") for b in c if b.get("type")=="text")
            tcs=[{"id":b["id"],"type":"function",
                  "function":{"name":b["name"],"arguments":json.dumps(b.get("input",{}))}}
                 for b in c if b.get("type")=="tool_use"]
            msg={"role":"assistant","content":text or None}
            if tcs: msg["tool_calls"]=tcs
            omsgs.append(msg)
        else:  # user role: may carry tool_result blocks
            trs=[b for b in c if b.get("type")=="tool_result"]
            if trs:
                for b in trs:
                    omsgs.append({"role":"tool","tool_call_id":b["tool_use_id"],
                                  "content":b.get("content","")})
            else:
                omsgs.append({"role":"user","content":json.dumps(c)})
    otools=[{"type":"function","function":{"name":t["name"],"description":t.get("description",""),
             "parameters":t["input_schema"]}} for t in tools] if tools else None
    kw=dict(model=model, messages=omsgs, max_tokens=max_tokens, temperature=temperature)
    if otools: kw["tools"]=otools
    resp=client.chat.completions.create(**kw)
    msg=resp.choices[0].message
    blocks=[]
    if msg.content: blocks.append({"type":"text","text":msg.content})
    for tc in (msg.tool_calls or []):
        try: args=json.loads(tc.function.arguments)
        except Exception: args={}
        blocks.append({"type":"tool_use","id":tc.id,"name":tc.function.name,"input":args})
    return {"content":blocks}

# ---------------- Gemini ----------------
def _gemini_complete(client, messages, system, tools, model, max_tokens, temperature):
    from google.genai import types
    # build a multi-turn Content list with function_call / function_response parts
    contents=[]
    for m in messages:
        c=m["content"]; role="model" if m["role"]=="assistant" else "user"
        if isinstance(c,str):
            contents.append(types.Content(role=role, parts=[types.Part(text=c)])); continue
        parts=[]
        for b in c:
            if b.get("type")=="text" and b.get("text"):
                parts.append(types.Part(text=b["text"]))
            elif b.get("type")=="tool_use":
                parts.append(types.Part(function_call=types.FunctionCall(name=b["name"], args=b.get("input",{}))))
            elif b.get("type")=="tool_result":
                nm=b.get("_name") or b.get("tool_use_id","").replace("gem_","")
                parts.append(types.Part(function_response=types.FunctionResponse(
                    name=nm, response={"result":b.get("content","")})))
        if parts: contents.append(types.Content(role=role, parts=parts))
    cfg_kw=dict(temperature=temperature, max_output_tokens=max_tokens)
    if system: cfg_kw["system_instruction"]=system
    if tools:
        fdecls=[types.FunctionDeclaration(name=t["name"], description=t.get("description",""),
                                          parameters=t["input_schema"]) for t in tools]
        cfg_kw["tools"]=[types.Tool(function_declarations=fdecls)]
    resp=client.models.generate_content(model=model, contents=contents,
                                         config=types.GenerateContentConfig(**cfg_kw))
    blocks=[]
    cand=resp.candidates[0] if resp.candidates else None
    if cand and cand.content and cand.content.parts:
        for p in cand.content.parts:
            if getattr(p,"text",None): blocks.append({"type":"text","text":p.text})
            fc=getattr(p,"function_call",None)
            if fc: blocks.append({"type":"tool_use","id":f"gem_{fc.name}","name":fc.name,
                                  "input":dict(fc.args) if fc.args else {}})
    return {"content":blocks}

def make_complete(provider=None, max_tokens=500, temperature=0.0, max_retries=4):
    prov=detect_provider(provider)
    if prov=="anthropic":
        import anthropic
        client=anthropic.Anthropic()
        fn=_anthropic_complete
    elif prov=="openai":
        import openai
        client=openai.OpenAI()
        fn=_openai_complete
    elif prov=="gemini":
        from google import genai
        client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        fn=_gemini_complete
    elif prov=="openrouter":
        import openai
        client=openai.OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ.get("OPENROUTER_API_KEY",""))
        fn=_openai_complete   # OpenRouter is OpenAI-compatible (incl. tool-calling)
    else:
        raise RuntimeError(f"Unknown provider {prov}")
    def complete(messages, system, tools, model):
        m=resolve_model(prov, model)
        last=None
        for attempt in range(max_retries):
            try:
                return fn(client, messages, system, tools, m, max_tokens, temperature)
            except Exception as e:
                last=e; time.sleep(1.5*(attempt+1))
        raise last
    complete.provider=prov
    complete.ladder=LADDER.get(prov,{})
    return complete
