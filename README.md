# RetroAgent 

RetroAgent is a LangGraph controller for multistep retrosynthesis. Given a target SMILES, it starts with an empty search graph, expands candidate reactions through a resident prediction service, and checks exact stock membership and reaction support. A second phase uses the first phase's observations to repair unresolved branches and improve the route portfolio within a shared budget. The CLI exports candidate routes, metrics, budget usage, and a per-case audit.

## 100-target benchmark

| Metric | RetroAgent  | ReTRACE-style | RetrOrchestrator-style | LARC-style | RetroPlanner-style | DeepRetro-style |
|---|---:|---:|---:|---:|---:|---:|
| Stock closure | 99% | 95% | 98% | 88% | 81% | 87% |
| Strict closure | 80% | 71% | 67% | 60% | 70% | 59% |
| Mean first-route CC over 100 targets | 2.6918 | 2.3870 | 2.2223 | 2.2117 | 2.2292 | 2.3250 |
| Mean all-route CC, averaged per target | 2.5164 | 2.2056 | 2.0827 | 1.9953 | 1.9826 | 2.0454 |
| All-route diversity | 0.4439 | 0.4370 | 0.4944 | 0.5242 | 0.5182 | 0.5328 |
| Routes with complete yield evidence | 320 | 589 | 611 | 748 | 1,262 | 623 |
| Mean cumulative yield proxy on covered routes | 8.04% | 22.08% | 15.76% | 22.93% | 19.05% | 24.89% |
| Mean native model calls per target | 96 | 96 | 96 | 96 | 96 | 96 |
| Mean LLM calls per target | 13.38 | 17.57 | 17.01 | 10.50 | 17.10 | 18.00 |
| Mean search work per target | 10,030.11 | 5,942.98 | 5,850.21 | 6,076.78 | 1,778.09 | 8,046.67 |

The baseline columns represent local adaptations of the five control strategies using shared chemistry services. The 100 targets were used during  development. Strict closure checks exact stock, route structure, atom sources, search depth and size limits, and positive ChemCensor (CC) support at every step. The yield proxy multiplies step estimates and is reported only on routes with complete evidence.

[Full results](benchmark/RESULTS.md) · [CSV](benchmark/comparison.csv) · [Protocol](benchmark/PROTOCOL.md) · [Audit summary](benchmark/audit-summary.json)

## Run

Use Linux and Python 3.12 from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m retroagent_langgraph --help
```

The controller connects to two local services through authenticated Unix sockets. Supply each service's directory containing `ready.json`.

- `--service` points to the resident chemistry backend. Its interface and tool schema are in [resident_function_tools_v1.py](scripts/resident_function_tools_v1.py). The backend provides LocalRetro, RetroChimeraEdit, AiZynthFinder, Graph2Edits, and RetroChimeraDeNovo predictions, exact stock checks, inventory neighbors, and exact reaction retrieval.
- `--cc-service` points to a ChemCensor scoring service with the benchmark database. The CC server entry point is [resident_ram_cc_v86.py](scripts/resident_ram_cc_v86.py). It needs PyTorch, ChemCensor 1.2.0, the pinned `ChemCensor-DB-U2-1.0.0.sqlite`, and enough shared memory for a verified copy. Run `python scripts/resident_ram_cc_v86.py --out /path/to/new-cc-service --workers 4` in that service environment.
- Set `RETROAGENT_ASSETS` to the directory containing the external reference assets, or use the default `assets/`. Reaction analogs use `product_index.sqlite`. Yield reporting uses `train_RC1_contexts.sqlite` and `yield_reference.sqlite`. Use `--no-yield` to disable yield reporting without changing the search budget.

The release ships the controller, service client, CC server, tests, and aggregate results. Model weights, stock and reference databases, benchmark targets, and credentials are external runtime inputs.

```bash
python -m retroagent_langgraph \
  --smiles 'CCOCC' --id example --out runs/example \
  --service /path/to/native-service --cc-service /path/to/cc-service \
  --workflow --no-yield
```

`--workflow` runs without LLM calls. Agent mode requests `gpt-6.1-sol` with medium effort via local Codex CLI. Its Linux Landlock launcher needs a runtime that can change ownership and drop to UID 65534. Supply the CLI executable through `RETROAGENT_CODEX_BIN` and a local CLI login file through `RETROAGENT_CODEX_AUTH` when their defaults are unsuitable.

Use `--targets` for a JSON array of objects with `Structure ID` and `SMILES` fields. The output directory must be new. For a benchmark subset, `--blocked-targets-csv` can block direct exact retrieval on the complete cohort. Pass `--config benchmark/config.json` for the recorded benchmark limits.

## Tests

The controller tests use a simulated backend and run without service credentials or model assets.

```bash
python -m unittest discover -s tests
```
