"""Analyze CompletenessCliff revision runs -> per-slice tables (Wilson 95% CI).

    python3 analyze_cc_revision.py
"""
import json, math, os
from collections import defaultdict

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
NAME = {"anthropic": "Claude", "openai": "GPT", "gemini": "Gemini"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0, c - h), min(1, c + h))


def cell(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return g


def fmt(rows, field="correct"):
    k = sum(r[field] for r in rows)
    p, lo, hi = wilson(k, len(rows))
    return f"{p:.2f} [{lo:.2f},{hi:.2f}] (n={len(rows)})"


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/CompletenessCliff_rev.jsonl")]
    rows = [r for r in rows if not r.get("error")]
    main_rows = [json.loads(l) for l in open(f"{DATA}/CompletenessCliff_full.jsonl")]
    main_rows = [r for r in main_rows if not r.get("error") and r.get("tier") == "small"]
    print(f"rev rows: {len(rows)}   errors excluded: ok")

    print("\n=== Slice A: aggregation-family generalization (small tier) ===")
    for variant in ("grouped", "dedup", "condsum", "temporal"):
        vr = [r for r in rows if r["variant"] == variant]
        for n in (4, 16):
            nr = [r for r in vr if r["n_pages"] == n]
            for prov in ("anthropic", "openai", "gemini"):
                pr = [r for r in nr if r["provider"] == prov]
                if pr:
                    claim = sum(r["claimed_complete"] for r in pr)
                    fa = sum(r["fetched_all"] for r in pr)
                    print(f"  {variant:9} N={n:2} {NAME[prov]:7} acc {fmt(pr)}  "
                          f"claim {claim}/{len(pr)}  fetched_all {fa}/{len(pr)}")

    print("\n=== Slice B: precomputed-subtotal control vs raw counting ===")
    for n in (8, 16, 32):
        for prov in ("anthropic", "openai", "gemini"):
            sub = [r for r in rows if r["variant"] == "subtotal"
                   and r["n_pages"] == n and r["provider"] == prov]
            raw = [r for r in main_rows if r.get("n_pages") == n and r["provider"] == prov]
            if sub:
                print(f"  N={n:2} {NAME[prov]:7} subtotal {fmt(sub)}   raw-count {fmt(raw)}")

    print("\n=== Slice C: elicitation wording (wrong-answer claim rates) ===")
    for n in (8, 32):
        for prov in ("anthropic", "openai", "gemini"):
            base = [r for r in main_rows if r.get("n_pages") == n and r["provider"] == prov]
            for variant, label in (("recordclaim", "record-level bool"),
                                   ("confidence", "confidence>=90")):
                vr = [r for r in rows if r["variant"] == variant
                      and r["n_pages"] == n and r["provider"] == prov]
                if not vr:
                    continue
                wrong = [r for r in vr if not r["correct"]]
                wc = sum(r["claimed_complete"] for r in wrong)
                basew = [r for r in base if not r["correct"]]
                bc = sum(r["claimed_complete"] for r in basew)
                extra = ""
                if variant == "confidence":
                    cs = [r["claim_raw"] for r in wrong if r.get("claim_raw") is not None]
                    if cs:
                        extra = f"  mean-conf-when-wrong {sum(cs)/len(cs):.0f}"
                print(f"  N={n:2} {NAME[prov]:7} {label:17} acc {fmt(vr)}  "
                      f"claim-when-wrong {wc}/{len(wrong)}  "
                      f"(orig bool: {bc}/{len(basew)}){extra}")

    print("\n=== Slice D: fixed total records (160), pagination varies ===")
    print("  (N=32,R=5) from the main run; new cells hold records fixed")
    for (n, per) in ((2, 80), (8, 20), (16, 10), (32, 5)):
        for prov in ("anthropic", "openai", "gemini"):
            if (n, per) == (32, 5):
                pr = [r for r in main_rows if r.get("n_pages") == 32 and r["provider"] == prov]
            else:
                pr = [r for r in rows if r["variant"] == "count" and r["n_pages"] == n
                      and r["per_page"] == per and r["provider"] == prov]
            if pr:
                print(f"  N={n:2} R={per:2} {NAME[prov]:7} acc {fmt(pr)}")


if __name__ == "__main__":
    main()
