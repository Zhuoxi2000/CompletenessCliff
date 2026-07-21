"""
Full-scale runner: adds a capability-tier sweep (small/mid/large) and wider axes on top
of the pilot harnesses. Writes data/<Direction>_full.jsonl with a 'tier' field.

Usage:
  python ei_fullrun.py CompletenessCliff --seeds 15 --tiers small,mid --ns 1,2,4,8,16,32 --temp 0.0
  python ei_fullrun.py UnitChain        --seeds 15 --tiers small,mid --hops 1,2,3,4,6,8
"""
import os, sys, json, argparse, importlib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from ei_common import run_agent, make_backends  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODMAP = {"RetryLedger": "ei_retryledger", "UnitChain": "ei_unitchain",
          "LedgerHarm": "ei_ledgerharm", "CompletenessCliff": "ei_completenesscliff",
          "HintPlacebo": "ei_hintplacebo"}


def axis_kwargs(name, a):
    if name == "CompletenessCliff" and a.ns:
        return {"ns": tuple(int(x) for x in a.ns.split(","))}
    if name == "UnitChain" and a.hops:
        return {"hops": tuple(int(x) for x in a.hops.split(","))}
    return {}


def run(name, seeds, providers, tiers, sys_variant, temp, max_workers=8, max_turns=None):
    mod = importlib.import_module(MODMAP[name])
    kw = axis_kwargs(name, ARGS)
    path = f"{DATA}/{name}_full.jsonl"
    rows = []
    jobs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex, open(path, "a") as fh:
        for tier in tiers:
            backs = make_backends(providers, temperature=temp, max_tokens=700)
            for prov, comp in backs.items():
                for meta, ep in mod.episodes(range(seeds), sys_variant=sys_variant, **kw):
                    jobs.append(ex.submit(_one, comp, prov, tier, meta, ep, temp, max_turns))
        for i, f in enumerate(as_completed(jobs), 1):
            r = f.result()
            rows.append(r)
            fh.write(json.dumps(r) + "\n")     # incremental: resilient to interruption
            fh.flush()
            if i % 25 == 0:
                print(f"[{name}] {i}/{len(jobs)} done", flush=True)
    errs = sum(1 for r in rows if r.get("error"))
    print(f"[{name}] {len(rows)} rows, {errs} errors, tiers={tiers} -> {path}")
    if errs:
        print("  sample error:", next(r["error"] for r in rows if r.get("error")))


def _one(comp, prov, tier, meta, ep, temp, max_turns):
    row = {**meta, "provider": prov, "tier": tier, "temp": temp}
    try:
        mt = max_turns or (max(12, int(meta.get("n_pages", 8)) + 6) if "n_pages" in meta else 12)
        row.update(run_agent(comp, tier, ep, max_turns=mt))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("direction")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--providers", default="anthropic,openai,gemini")
    ap.add_argument("--tiers", default="small,mid")
    ap.add_argument("--sys", default="neutral")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--ns", default=None)
    ap.add_argument("--hops", default=None)
    ARGS = ap.parse_args()
    run(ARGS.direction, ARGS.seeds, tuple(ARGS.providers.split(",")),
        tuple(ARGS.tiers.split(",")), ARGS.sys, ARGS.temp)
