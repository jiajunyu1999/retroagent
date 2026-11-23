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
