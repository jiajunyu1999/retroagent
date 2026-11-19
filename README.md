# RetroAgent

**Recursive self-improvement for multistep retrosynthesis**

RetroAgent plans synthesis routes from a target SMILES. It uses **recursive self-improvement (RSI)** during inference. Each round of search produces evidence about which reactions, intermediates, and tools helped. The agent uses that evidence to decide where to backtrack, which tool to call next, and which promising branches to keep.

## The RSI loop

```mermaid
flowchart LR
    Target["Target SMILES"] --> Search["Expand route graph"]
    Search --> Verify["Check inventory and reaction support"]
    Verify -->|At stop| Output["Rank and audit routes"]
    Verify --> Memory["Update episode memory"]
    Memory --> Repair["Choose backtrack point and tool"]
    Repair -->|Next round| Search

    classDef input fill:#eef2ff,stroke:#6366f1,color:#1e1b4b
    classDef process fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef feedback fill:#ecfdf5,stroke:#059669,color:#064e3b
    class Target,Output input
    class Search,Verify process
    class Memory,Repair feedback
```

1. **Search.** Build a graph of candidate reactions using specialist retrosynthesis models and retrieved precedents. Verify exact inventory membership, route structure, atom sources, and per-step ChemCensor (CC) support.
2. **Reflect and revise.** Record supported branches, missing precursors, failed reaction centers, and tool outcomes in memory scoped to the current target. The LLM uses those observations to choose a repair point and request a different expansion or propose a local change.
3. **Recheck and rank.** Validate the revised candidates and retain the best route found so far. Both phases share one budget. The first route is ranked by closure tier and CC. For the remaining slots, RetroAgent uses yield coverage and diversity to select among routes close to the best CC in each tier.

## Benchmark

All methods share the same 100 development targets, chemistry services, and budget ceilings. Baseline rows use local adaptations of the published control strategies.

| Method | Stock closure | Strict closure | First-route CC | All-route CC | Diversity |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **RetroAgent** | **99%** | **80%** | **2.6918** | **2.5164** | 0.4439 |
| ReTRACE | 95% | 71% | 2.3870 | 2.2056 | 0.4370 |
| RetroPlanner | 81% | 70% | 2.2292 | 1.9826 | 0.5182 |
| RetrOrchestrator | 98% | 67% | 2.2223 | 2.0827 | 0.4944 |
| LARC | 88% | 60% | 2.2117 | 1.9953 | 0.5242 |
| DeepRetro | 87% | 59% | 2.3250 | 2.0454 | **0.5328** |

CC and diversity are averaged per target. Strict closure requires exact stock membership at the leaves, valid route structure and atom sources, positive CC at each step, and a route within the depth and size limits.

[Full results, including yield and search cost](benchmark/RESULTS.md) · [CSV](benchmark/comparison.csv) · [Protocol](benchmark/PROTOCOL.md) · [Audit summary](benchmark/audit-summary.json)

## Run RetroAgent

Install the controller on Linux with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a first run, connect the two services and use the workflow mode to check the setup.

```bash
python -m retroagent_langgraph \
  --smiles 'CCOCC' --id example --out runs/example \
  --service /path/to/native-service --cc-service /path/to/cc-service \
  --workflow --no-yield
```

In agent mode, the RSI loop requests `gpt-6.1-sol` with medium reasoning effort through the local Codex CLI. Omit `--workflow` to enable LLM calls. The Linux launcher uses Landlock and needs permission to change file ownership and drop to UID 65534. Use `RETROAGENT_CODEX_BIN` and `RETROAGENT_CODEX_AUTH` to specify the local binary and login file if needed.

For batch inference, pass `--targets` with a JSON array containing `Structure ID` and `SMILES` fields. Use a new output directory for each run. The run exports the candidate routes, measured budget use, and per-target audit. Pass `--config benchmark/config.json` for the recorded limits and `--blocked-targets-csv` to exclude direct exact retrieval for a full benchmark cohort.

<details>
<summary>Services and reference assets</summary>

The controller connects to authenticated local Unix sockets. Each service directory contains a `ready.json` connection manifest.

- **Chemistry service.** `--service` connects to the [resident tool interface](scripts/resident_function_tools_v1.py), which provides specialist predictions, exact inventory, inventory neighbors, and exact reaction retrieval. The benchmark used LocalRetro, RetroChimeraEdit, AiZynthFinder, Graph2Edits, and RetroChimeraDeNovo.
- **Reaction scoring.** `--cc-service` connects to the [ChemCensor server](scripts/resident_ram_cc_v86.py). The server needs PyTorch, ChemCensor 1.2.0, the pinned `ChemCensor-DB-U2-1.0.0.sqlite`, and sufficient shared memory. Start it with `python scripts/resident_ram_cc_v86.py --out /path/to/new-cc-service --workers 4`.
- **Reference assets.** Set `RETROAGENT_ASSETS` to the reference directory or use the default `assets/`. Reaction analog retrieval uses `product_index.sqlite`. Yield reporting uses `train_RC1_contexts.sqlite` and `yield_reference.sqlite` and can be disabled with `--no-yield`.

Model weights, stock and reference databases, benchmark targets, and credentials are supplied separately.

</details>

## Tests

The controller tests use a simulated chemistry backend.

```bash
python -m unittest discover -s tests
```
