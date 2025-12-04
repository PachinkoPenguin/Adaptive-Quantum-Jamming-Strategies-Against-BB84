# Adaptive Quantum Jamming Strategies Against BB84

## Overview
This repository provides a complete, research-grade BB84 simulator with integrated eavesdropping strategies (Eve), multi-test attack detection, information-theoretic metrics, and publication-quality visualization. It supports a fast classical backend and an optional Qiskit backend.

Highlights:
- Full BB84 protocol with image-based outputs and JSON metadata
- Pluggable eavesdropping strategies via a clean strategy pattern
- Statistical AttackDetector (QBER CI, chi-square, mutual information, runs test)
- Information theory utilities (von Neumann entropy, Holevo bound, mutual information)
- AdaptiveJammingSimulator for strategy comparison and CSV/plot exports
- VisualizationManager for figures and LaTeX tables

As of 2025-10-24: full test suite passing (155 passed, 1 skipped) on Linux with Python 3.13.

## Project structure
```
Adaptive-Quantum-Jamming-Strategies-Against-BB84/
├── src/
│   └── main/
│       └── bb84_main.py          # Core protocol, Eve, detector, simulator, visualization
├── tests/                        # Unit/integration tests (attack, holevo, visualization, etc.)
├── examples/
│   └── visualization_demo.py     # End-to-end demo generating all figures and a LaTeX table
├── bb84_output/                  # Timestamped run artifacts (images, logs, summary.json)
├── *.md                          # Documentation (this README and topic guides)
└── requirements*.txt             # Dependencies
```

## Key components

- Protocol and channel
  - `BB84Protocol`: 5-phase flow (transmit → sift → estimate errors → correct → privacy amp); invokes detection and provides Eve feedback after error estimation; saves images and `summary.json`.
  - `QuantumChannel`: applies loss and noise, optionally integrates `EveController` and forwards transmission metadata (index, timestamp, atmospheric Cn², loss/error rates).

- Eve and strategies
  - `AttackStrategy` (abstract) and `EveController` (orchestration).
  - Built-ins: `InterceptResendAttack`, `AdaptiveAttack`, `QBERAdaptiveStrategy` (PID), `GradientDescentQBERAdaptive`, `BasisLearningStrategy`, `PhotonNumberSplittingAttack` (PNS) and others present in `bb84_main.py`.

- Detection and information theory
  - `AttackDetector`: QBER Hoeffding CI, chi-square basis balance, basis mutual information, runs test; aggregated verdict with serializable history.
  - `von_neumann_entropy`, `holevo_bound`, `bb84_holevo_bound`, `binary_entropy`, `calculate_eve_information`, `mutual_information_eve_alice`.

- Simulation and visualization
  - `BB84Simulator`: parameter sweeps, backend comparisons, summary reports.
  - `AdaptiveJammingSimulator`: single-run strategy orchestration, comparison, CSV export, t-tests.
  - `VisualizationManager`: QBER/time, intercept probability, info leakage, ROC/AUC, atmospheric visuals (Rytov, phase screens, beam frames), error/runs/MI, Holevo charts, LaTeX table export.

## Installation

```bash
pip install -r requirements.txt
# Optional (quantum backend):
pip install qiskit qiskit-aer
```

## Quick start

Run a complete BB84 protocol with detection and optional Eve:

```python
from src.main.bb84_main import BB84Protocol, ClassicalBackend, InterceptResendAttack, EveController

backend = ClassicalBackend()
attack = InterceptResendAttack(backend, intercept_probability=0.5)
protocol = BB84Protocol(backend=backend, attack_strategy=attack)
protocol.setup(channel_loss=0.05, channel_error=0.02)
stats = protocol.run_protocol(num_bits=1000)
print({k: stats[k] for k in ['qber','secure','attack_detected','eve_interceptions']})
```

Compare strategies and plot results:

```python
from src.main.bb84_main import AdaptiveJammingSimulator, ClassicalBackend, InterceptResendAttack

sim = AdaptiveJammingSimulator(n_qubits=1000, backend=ClassicalBackend())
res_map = sim.compare_strategies([InterceptResendAttack(sim.backend, 0.3)])
fig = sim.plot_results(res_map)  # Returns a matplotlib Figure
csv = sim.save_results_csv(res_map)  # Writes a CSV summary
```

Generate publication-ready figures and a LaTeX table:

```python
from src.main.bb84_main import VisualizationManager
viz = VisualizationManager(output_dir='bb84_output/figs')
viz.export_results_table({'Baseline': {'QBER': {'mean':0.08,'std':0.01,'significant':False}}}, filename='table.tex')
```

See `examples/visualization_demo.py` for a comprehensive end-to-end demo that produces a full figure set and LaTeX export, even if a simulation step fails late.

## Outputs

Each run creates a timestamped directory under `bb84_output/`:

- Images: protocol log, summary table, visualizations (pipeline, parameter sweep, backend comparison), and any visualization/demo outputs
- Logs: `run.log` per run directory
- Metadata: `summary.json` with run stats and parameters

`summary.json` contains at minimum:
- `stats`: backend, bits sent, raw/final key length, qber, efficiency, channel loss/error rates, `secure` flag
- Detector and Eve fields: `attack_detected`, `eve_interceptions`, `eve_information` (when applicable)
- `params`: run arguments (num_bits, channel_loss, channel_error, backend)

## Notes

- Backends: classical backend is always available; Qiskit backend is optional and auto-detected.
- Tests: run `pytest -q` from repo root. Current status: PASS (155 passed, 1 skipped).
- Reproducibility: for simulator comparisons, prefer fixed seeds and consistent presets.

## References
- Bennett, C. H., & Brassard, G. (1984). Quantum cryptography: Public key distribution and coin tossing.
- Scarani, V., et al. (2009). The security of practical quantum key distribution.

## License
Academic research project. See repository for details.