"""CompletenessCliff figures (no Chinese, CVD-safe palette, embedded-ready PNG).
fig_cc_scissors.png : per-vendor accuracy (solid) vs self-reported completeness (dashed) vs N.
fig_cc_tier.png     : accuracy vs N, small vs mid tier (capability effect).
"""
import os, sys, json, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

# CVD-safe (Wong 2011)
COL = {"anthropic": "#0072B2", "openai": "#D55E00", "gemini": "#009E73"}
NAME = {"anthropic": "Claude", "openai": "GPT", "gemini": "Gemini"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), 0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def load():
    return [json.loads(l) for l in open(f"{DATA}/CompletenessCliff_full.jsonl") if '"error"' not in l]


def cell(rows, tier, prov, field):
    """return sorted [(N, p, lo, hi)] for a field."""
    g = defaultdict(list)
    for r in rows:
        if r.get("tier") == tier and r.get("provider") == prov and r.get(field) is not None:
            g[r["n_pages"]].append(r[field])
    out = []
    for n in sorted(g):
        p, lo, hi = wilson(sum(g[n]), len(g[n]))
        out.append((n, p, lo, hi))
    return out


def scissors(rows, tier="small"):
    provs = [p for p in COL if any(r.get("provider") == p for r in rows)]
    fig, axes = plt.subplots(1, len(provs), figsize=(3.1 * len(provs), 3.0), sharey=True)
    if len(provs) == 1:
        axes = [axes]
    for ax, prov in zip(axes, provs):
        acc = cell(rows, tier, prov, "correct")
        cla = cell(rows, tier, prov, "claimed_complete")
        if acc:
            xs = [n for n, *_ in acc]
            ax.plot(xs, [p for _, p, _, _ in acc], "-o", color=COL[prov], lw=2, ms=4, label="Accuracy")
            ax.fill_between(xs, [lo for *_, lo, _ in acc], [hi for *_, _, hi in acc],
                            color=COL[prov], alpha=0.15)
        if cla:
            xs = [n for n, *_ in cla]
            ax.plot(xs, [p for _, p, _, _ in cla], "--s", color="#555555", lw=1.6, ms=3,
                    label="Claims complete")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({n for n, *_ in acc}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(NAME[prov], fontsize=11)
        ax.set_xlabel("Pages N", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Rate", fontsize=10)
            ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_cc_scissors.png", dpi=200, bbox_inches="tight")
    print("wrote fig_cc_scissors.png")


def tier_panel(rows):
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    styles = {"small": ("-o", 1.0), "mid": ("--^", 0.85)}
    for prov in [p for p in COL if any(r.get("provider") == p for r in rows)]:
        for tier, (st, al) in styles.items():
            c = cell(rows, tier, prov, "correct")
            if c:
                ax.plot([n for n, *_ in c], [p for _, p, _, _ in c], st, color=COL[prov],
                        alpha=al, lw=1.8, ms=4, label=f"{NAME[prov]} ({tier})")
    ax.set_xscale("log", base=2)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Pages N", fontsize=10); ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(-0.03, 1.03); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=1, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_cc_tier.png", dpi=200, bbox_inches="tight")
    print("wrote fig_cc_tier.png")


if __name__ == "__main__":
    import matplotlib.ticker
    rows = load()
    scissors(rows, tier="small")
    tier_panel(rows)
