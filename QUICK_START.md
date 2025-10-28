# Quick Start Guide

## Install dependencies

```bash
pip install -r requirements.txt
# Optional (for quantum backend):
pip install qiskit qiskit-aer
```

## Run a protocol with detection (and optional Eve)

```python
from src.main.bb84_main import BB84Protocol, ClassicalBackend, InterceptResendAttack

backend = ClassicalBackend()
attack = InterceptResendAttack(backend, intercept_probability=0.5)
protocol = BB84Protocol(backend=backend, attack_strategy=attack)
protocol.setup(channel_loss=0.05, channel_error=0.02)
stats = protocol.run_protocol(num_bits=1000)
print(stats['qber'], stats['attack_detected'], stats['final_key_length'])
```

Artifacts are written to `bb84_output/<backend>_<run>_run_<timestamp>/` (images, run.log, summary.json).

## Compare strategies quickly

```python
from src.main.bb84_main import AdaptiveJammingSimulator, InterceptResendAttack, ClassicalBackend

sim = AdaptiveJammingSimulator(n_qubits=1000, backend=ClassicalBackend())
results = sim.compare_strategies([InterceptResendAttack(sim.backend, 0.3)])
fig = sim.plot_results(results)         # matplotlib Figure
csv = sim.save_results_csv(results)     # path to CSV
```

## Create figures and LaTeX tables

```python
from src.main.bb84_main import VisualizationManager
viz = VisualizationManager(output_dir='figs')
viz.plot_qber_evolution([0.05, 0.08, 0.13, 0.11], threshold=0.11)
viz.export_results_table({'Baseline': {'QBER': {'mean':0.10, 'std':0.01, 'significant': False}}})
```

See `examples/visualization_demo.py` for a full demo that generates a figure suite and a LaTeX table.

## Good to know

- Detection: `AttackDetector` aggregates QBER CI, chi-square basis balance, basis MI, and runs test.
- Eve feedback: After error estimation, `BB84Protocol` gives Eve QBER and public bases to adapt strategies.
- Info theory: von Neumann entropy and Holevo bound utilities are available for analysis.
- Qiskit: If installed, the quantum backend is auto-detected.

## Run tests

```bash
pytest -q
```

As of 2025-10-24, all tests pass (155 passed, 1 skipped).
