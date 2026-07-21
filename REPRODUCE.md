# Reproduce — CompletenessCliff

Self-contained bundle. API keys are read from the environment: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY`.

## Files
- `code/ei_completenesscliff.py` — episode + deterministic oracle (the benchmark logic)
- `code/ei_common.py`, `code/direct_backend.py` — multi-turn agent loop + 4-vendor SDK backend
- `code/ei_fullrun.py` — matrix runner (incremental JSONL, tier sweep)
- `code/finalize_cc.py` — stats (Wilson 95% CI) -> results_table.tex + macros json
- `code/make_cc_figure.py` — publication figures
- `data/CompletenessCliff_full.jsonl` — full-run episode records ; `data/CompletenessCliff_pilot.jsonl` — pilots
- `figures/` — the figures used in the paper

## Regenerate (from repo root ~/paper/EI-agent-6/code)
```bash
# full run (writes data/CompletenessCliff_full.jsonl incrementally)
python ei_fullrun.py CompletenessCliff --seeds 18 --tiers small,mid --temp 0.7   # (CompletenessCliff/UnitChain also take --ns/--hops)
# stats + table + macros, then figures
python finalize_cc.py && python make_cc_figure.py
# compile the paper
cd ../papers/CompletenessCliff && pdflatex main && pdflatex main
```
See ../README.md for the headline results and the exact axes used.
