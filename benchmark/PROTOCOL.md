# Benchmark protocol and metric definitions

## Methods and inputs

The comparison uses the same 100 target structures in `targets.json`. RetroAgent is the ten-batch outcome-memory run with an initially empty memory and the configuration in `config.json`. DeepRetro uses the author repository at commit `dc8fcad` with local GPT-6.1 Sol Medium execution transport. Specialized-model metrics are the 14 valid archived methods in `specialist_archive.csv` after its method-level execution-error rule. Four excluded specialist methods are omitted. Local adaptations of other agent methods are not included.

DeepRetro completed 97 targets and had 3 execution errors. Every primary metric retains the 100-target denominator. Missing or invalid first routes contribute zero. Method-specific archived search protocols supply the recorded results.

## Metrics

- **Inventory closure:** an audited candidate route with every leaf in the exact inventory.
- **Strict closure:** inventory closure plus structural and source validity, permitted route depth and size, and ChemCensor `score_with_fg > 0` for every reaction.
- **First-route CC:** mean step CC of the selected first route, averaged across all 100 targets. Missing or invalid first routes contribute zero.
- **Diversity:** the archived per-target stock-closed route diversity, averaged equally across the 100 targets.
- **Complete-yield routes:** total inventory-closed output routes for which every reaction has yield evidence under the archived relaxed USPTO/Reaxys matching rules.
- **Yield proxy:** conditional mean route proxy across those completely covered inventory-closed routes. It is a reaction-analog proxy, not a measured synthesis yield. Uncovered routes are excluded from this conditional mean.

A dash means the metric is unavailable from the DeepRetro export. See `figures/CAPTIONS.md` for panel-level definitions and `comparison.csv` for the plotted values.

## Runtime limits and output

RetroAgent shares 96 specialist calls, 18 LLM calls, 64 exact-reaction queries, 36 material queries, 4 support queries and 12,288 search-work units across two phases. The initial phase receives 48 model calls and 6,144 search work; continuation uses the same cumulative counters. Output is capped at 20 routes per target. `summary.json` records observed compute rather than substituting the caps.

Cross-target memory stores strategy outcome counts. It updates after each batch of ten targets and changes only subsequent inference. Per-target route examples under `examples/` are exported results.

## Controller and direct-LLM comparison

`controller_comparison.csv` and `controller_per_case.csv` add the MCTS substitution and saved direct-LLM outputs on the same 100 target structures.

- **RetroAgent (MCTS):** UCT replaces the LLM decision node. All 100 cases pass the output and physical-call audit. Chemistry services, pinned CC, stock, retrieval limits, route limits and the total 96-model-call / 12,288-search-work caps are shared with RetroAgent. MCTS simulation states debit that search-work budget. There are no LLM calls or cross-target LLM repair-role updates.
- **Pure LLM (6.1 Sol)** and **Pure LLM (6.2 Astra):** Codex generates one candidate route per target from model knowledge, with no chemistry tools, retrieval or stock/CC feedback during generation. Saved routes are evaluated using the common normalization, exact-stock pruning, structural/source checks, pinned CC and relaxed yield matching. Missing or rejected routes contribute zero to the all-100 CC. The updated post-evaluation supersedes the earlier direct-route summary.
- Pure-LLM generation uses one call per target. Diversity is zero with at most one output route. Its recorded elapsed time is generation-only; other elapsed values cover their online inference runs. These modes have different LLM call counts and candidate-generation capabilities.
