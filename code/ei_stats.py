"""Per-cell rates with Wilson 95% CI for a direction's full-run data. Prints a table and
dumps data/<Direction>_facts.json (used by the figure scripts and paper).

Usage: python ei_stats.py CompletenessCliff --axis tier,provider,n_pages --fields correct,overconfident_gap,claimed_complete
"""
import os, sys, json, math, argparse
from collections import defaultdict

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def load(name):
    path = f"{DATA}/{name}_full.jsonl"
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if not r.get("error")], sum(1 for r in rows if r.get("error"))


def main(name, axis, fields):
    rows, errs = load(name)
    g = defaultdict(list)
    for r in rows:
        g[tuple(r.get(k) for k in axis)].append(r)
    facts = {"direction": name, "n_rows": len(rows), "errors": errs, "axis": axis, "cells": []}
    print(f"\n===== {name}  n={len(rows)} errors={errs} =====")
    header = "  " + "/".join(axis).ljust(30) + "  n   " + "  ".join(f"{f}[95%CI]" for f in fields)
    print(header)
    for key in sorted(g, key=lambda k: tuple((x is None, x) for x in k)):
        rs = g[key]
        cell = {"key": dict(zip(axis, key)), "n": len(rs)}
        parts = []
        for f in fields:
            vals = [r[f] for r in rs if r.get(f) is not None]
            k = sum(vals); n = len(vals)
            p, lo, hi = wilson(k, n)
            cell[f] = {"rate": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": n}
            parts.append(f"{p:.2f}[{lo:.2f},{hi:.2f}]")
        facts["cells"].append(cell)
        print("  " + "/".join(str(x) for x in key).ljust(30) + f"  {len(rs):<3} " + "  ".join(parts))
    out = f"{DATA}/{name}_facts.json"
    json.dump(facts, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("direction")
    ap.add_argument("--axis", default="tier,provider,n_pages")
    ap.add_argument("--fields", default="correct,overconfident_gap,claimed_complete")
    a = ap.parse_args()
    main(a.direction, a.axis.split(","), a.fields.split(","))
