"""
Pilot runner for the EI-Agent-6 batch. Runs a small matrix per direction across vendors,
writes data/<direction>_pilot.jsonl, and prints a signal summary.

Usage:
  python ei_pilot.py <direction> [--seeds N] [--providers a,b,c] [--sys neutral] [--temp 0.0]
  python ei_pilot.py all         # base pilot (neutral, temp 0) for all six
"""
import os, sys, json, argparse, importlib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from ei_common import run_agent, make_backends  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DIRECTIONS = {
    "RetryLedger": "ei_retryledger", "UnitChain": "ei_unitchain",
    "LedgerHarm": "ei_ledgerharm", "CompletenessCliff": "ei_completenesscliff",
    "HintPlacebo": "ei_hintplacebo",
}  # MemoryRot handled separately


def run_direction(name, seeds, providers, sys_variant, temp, max_workers=6):
    rows = []
    if name == "MemoryRot":
        import ei_memoryrot as mr
        backs = make_backends(providers, temperature=temp)
        jobs = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for prov, comp in backs.items():
                for h in (0, 2, 4):
                    for s in range(seeds):
                        jobs.append(ex.submit(_safe_mr, mr, comp, prov, h, s, sys_variant, temp))
            for f in as_completed(jobs):
                rows.append(f.result())
        _dump(name, rows)
        return rows
    mod = importlib.import_module(DIRECTIONS[name])
    backs = make_backends(providers, temperature=temp)
    eps = list(mod.episodes(range(seeds), sys_variant=sys_variant))
    jobs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for prov, comp in backs.items():
            for meta, ep in mod.episodes(range(seeds), sys_variant=sys_variant):
                jobs.append(ex.submit(_safe_run, comp, prov, meta, ep, temp))
        for f in as_completed(jobs):
            rows.append(f.result())
    _dump(name, rows)
    return rows


def _safe_run(comp, prov, meta, ep, temp):
    row = {**meta, "provider": prov, "temp": temp}
    try:
        row.update(run_agent(comp, "small", ep))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    return row


def _safe_mr(mr, comp, prov, h, s, sys_variant, temp):
    try:
        r = mr.memoryrot_run(comp, "small", h, s, sys_variant)
        r["provider"] = prov; r["temp"] = temp
        return r
    except Exception as e:
        return {"direction": "MemoryRot", "horizon": h, "seed": s, "provider": prov,
                "temp": temp, "error": f"{type(e).__name__}: {str(e)[:150]}"}


def _dump(name, rows):
    os.makedirs(DATA, exist_ok=True)
    path = f"{DATA}/{name}_pilot.jsonl"
    # append mode so multiple sys/temp runs accumulate
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    errs = [r for r in rows if r.get("error")]
    print(f"[{name}] {len(rows)} rows, {len(errs)} errors -> {path}")
    if errs:
        print("  sample error:", errs[0]["error"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("direction")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--providers", default="anthropic,openai,gemini")
    ap.add_argument("--sys", default="neutral")
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()
    provs = tuple(a.providers.split(","))
    names = list(DIRECTIONS) + ["MemoryRot"] if a.direction == "all" else [a.direction]
    for nm in names:
        run_direction(nm, a.seeds, provs, a.sys, a.temp)
