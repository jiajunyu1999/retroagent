# Benchmark results

| Method | Inventory closure | Strict closure | First-route CC |
| :--- | ---: | ---: | ---: |
| **RetroAgent** | **99%** | **82%** | **2.7349** |
| RetroAgent (MCTS) | 83% | 65% | 2.6096 |
| Pure LLM (6.1 Sol) | 10% | 3% | 0.9857 |
| Pure LLM (6.2 Astra) | 16% | 5% | 1.0199 |
| DeepRetro | 57% | 16% | 0.8002 |
| RetroChimera | 73% | 33% | 1.6107 |
| GLN | 46% | 32% | 1.5349 |
| LocalRetro | 65% | 31% | 1.5185 |
| AiZynthFinder | 52% | 28% | 1.6236 |
| RetroChimeraDeNovo | 75% | 28% | 1.5886 |
| RetroKNN | 64% | 28% | 1.5506 |
| RetroChimeraEdit | 56% | 27% | 1.6347 |
| MHNreact | 38% | 24% | 1.5030 |
| Graph2Edits | 59% | 23% | 1.4883 |
| MEGAN | 57% | 16% | 1.4630 |
| Chemformer | 38% | 16% | 1.4620 |
| RetroSim | 41% | 16% | 1.1169 |
| NeuralSym | 21% | 12% | 1.5001 |
| RingBreaker | 1% | 0% | 0.2433 |

## RetroAgent

| Metric | Result |
| :--- | ---: |
| Targets | 100 |
| Inventory closure | 99% |
| Strict closure | 82% |
| First-route CC (all targets) | 2.734881 |
| All-route CC (per-target mean) | 2.561054 |
| Stock-route diversity | 0.439292 |
| Mean output routes / target | 20.00 |
| Mean stock-closed routes / target | 19.63 |
| Mean strict routes / target | 15.98 |
| Complete-yield stock routes | 203 |
| Conditional yield proxy | 9.1697% |
| Mean specialist calls / target | 96.00 |
| Mean LLM calls / target | 13.56 |
| Mean search work / target | 10,124 |
| Mean elapsed seconds / target | 441.69 |

[Definitions and protocol](PROTOCOL.md) · [Per-target RetroAgent results](retroagent_per_case.csv) · [Per-target DeepRetro results](deepretro_per_case.csv) · [Per-target specialist results](specialist_per_case.csv)

[Four-way controller and pure-LLM comparison](controller_comparison.csv) · [Per-target outcomes](controller_per_case.csv)
