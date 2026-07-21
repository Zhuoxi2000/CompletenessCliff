# CompletenessCliff

Benchmark for **paginated-aggregation collapse and completeness overconfidence** in tool-using LLM agents.

An agent pages through N blocks of records and reports an aggregate (a thresholded count) plus a
self-report of whether it covered every page. The oracle is exact: the true aggregate is computed
programmatically, so an answer is right or wrong with no judge.

**Headline (3 vendors x 2 tiers, N in {1,2,4,8,16,32}, 15 seeds/cell, 450 episodes/tier):**
- Accuracy collapses as N grows (GPT small tier: 0.87 at N=1 -> 0.07 at N=8 -> 0.00 at N=32;
  Claude holds 0.73 at N=16, then fails at 32).
- Self-reported completeness stays ~0.94 while accuracy hits zero -- the accuracy/self-report
  "scissors". Wrong-but-claims-complete reaches 1.00 at N=32.
- Agents fetch **every** page in the collapsed cells (fetch coverage 1.00): the failure is
  aggregation, not retrieval.
- Capability shifts the cliff rightward (GPT 0.00 -> 0.17, Claude 0.73 -> 0.88 at N=16) but does
  not remove it.

## Layout
- `code/ei_completenesscliff.py` -- episode generator + exact-count oracle
- `code/ei_common.py`, `code/direct_backend.py` -- multi-turn agent loop, multi-vendor API backend
- `code/ei_fullrun.py` -- matrix runner (incremental JSONL, resumable)
- `code/finalize_cc.py`, `code/ei_stats.py` -- Wilson-CI stats, results table
- `code/make_cc_figure.py` -- paper figures
- `data/CompletenessCliff_full.jsonl` -- all episode records (count, self-report, pages fetched)
- `figures/` -- paper figures

## Reproduce
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...
cd code
python ei_fullrun.py CompletenessCliff --seeds 15 --tiers small,mid
python finalize_cc.py && python make_cc_figure.py
```
Scoring is exact integer equality against the machine-computed aggregate; no LLM judge.

## Citation
Paper under review ("When Agents Stop Counting: Measuring Paginated-Aggregation Collapse and
Completeness Overconfidence in Tool-Using Language-Model Agents"). Citation will be added on acceptance.
