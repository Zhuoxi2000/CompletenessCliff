import os
"""Turn CompletenessCliff_full.jsonl into (a) results_table.tex and (b) the macro values
to paste into main.tex. Run after the full run completes."""
import json, math
from collections import defaultdict

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
PAPER = os.path.join(os.path.dirname(__file__), "..")
NAME = {"anthropic": "Claude", "openai": "GPT", "gemini": "Gemini"}
ORDER = ["anthropic", "openai", "gemini"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), 0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


rows = [json.loads(l) for l in open(f"{DATA}/CompletenessCliff_full.jsonl") if '"error"' not in l]


def rate(tier, prov, N, field):
    v = [r[field] for r in rows if r.get("tier") == tier and r.get("provider") == prov
         and r.get("n_pages") == N and r.get(field) is not None]
    return wilson(sum(v), len(v)) if v else (float("nan"), 0, 0)


Ns = sorted({r["n_pages"] for r in rows})

# ---- results_table.tex : accuracy (small tier) + overconfidence gap, per vendor, by N ----
lines = [r"\begin{table}[t]", r"\caption{Aggregation accuracy and overconfidence gap versus page count $N$ (small tier), with Wilson 95\% confidence intervals. Overconf.\ = wrong while claiming complete.}",
         r"\label{tab:main}", r"\centering", r"\footnotesize",
         r"\begin{tabular}{r" + "c" * len(ORDER) + "c}", r"\toprule",
         r"$N$ & " + " & ".join(f"{NAME[p]} acc." for p in ORDER) + r" & GPT overconf.\\", r"\midrule"]
for N in Ns:
    cells = []
    for p in ORDER:
        a, lo, hi = rate("small", p, N, "correct")
        cells.append(f"{a:.2f}" if not math.isnan(a) else "--")
    g, _, _ = rate("small", "openai", N, "overconfident_gap")
    lines.append(f"{N} & " + " & ".join(cells) + f" & {g:.2f}" + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{PAPER}/results_table.tex", "w").write("\n".join(lines) + "\n")
print("wrote results_table.tex")

# ---- macro values for main.tex ----
def pct(t, p, N, f="correct"):
    a, lo, hi = rate(t, p, N, f)
    return f"{a:.2f}" if not math.isnan(a) else "NA"

Nmax = max(Ns)
macros = {
    "accGPTone": pct("small", "openai", 1), "accGPTeight": pct("small", "openai", 8),
    "accGPTtt": pct("small", "openai", Nmax),
    "accGemone": pct("small", "gemini", 1), "accGemtt": pct("small", "gemini", Nmax),
    "accClaudett": pct("small", "anthropic", Nmax),
    "gapGPTtt": pct("small", "openai", Nmax, "overconfident_gap"),
}
# flat claimed-complete: average claimed rate across N>=8 pooled vendors, small tier
cl = [r["claimed_complete"] for r in rows if r.get("tier") == "small" and r.get("n_pages", 0) >= 8]
macros["claimflat"] = f"{sum(cl)/len(cl):.2f}" if cl else "NA"
# mid gain: GPT small vs mid at the largest N both tiers share (mid is capped)
mid_Ns = sorted({r["n_pages"] for r in rows if r.get("tier") == "mid"})
if mid_Ns:
    Ncmp = max(mid_Ns)
    ms = rate("mid", "openai", Ncmp, "correct")[0]
    ss = rate("small", "openai", Ncmp, "correct")[0]
    macros["midgain"] = f"{ms:.2f} vs {ss:.2f} at $N{{=}}{Ncmp}$"
else:
    macros["midgain"] = "PENDING (mid tier still running)"
print("\n=== MACRO VALUES (paste into main.tex \\newcommand block) ===")
for k, v in macros.items():
    print(f"  {k} = {v}")
json.dump(macros, open(f"{DATA}/cc_macros.json", "w"), indent=2)
print("\nwrote data/cc_macros.json")
