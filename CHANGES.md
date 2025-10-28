# Changelog

This document summarizes the major changes across phases as the project evolved into a complete, integrated BB84 research stack.

## 2025-10-28 — Integrated stack polish and docs refresh

- Documentation updates across README, IMPLEMENTATION_SUMMARY, COMPLETE_SUMMARY, QUICK_START, QUICK_REFERENCE, EVE_IMPLEMENTATION, VISUALIZATION_REFERENCE, and VISUALIZATION_GALLERY
- Visualization gallery aligned with `examples/visualization_demo.py` (demo-driven figure regeneration)
- Project status updated; tests snapshot reflected (155 passed, 1 skipped as of 2025-10-24)
- Added a strategy smoke test script (`examples/strategy_smoke_test.py`) for quick QBER/detection/info checks with fixed seeds

## 2025-10-24 — Protocol + Eve integration; detectors; info theory; simulators; viz

- Protocol/channel
    - `BB84Protocol` invokes `AttackDetector` after error estimation and feeds back QBER+bases to Eve
    - `QuantumChannel` forwards rich metadata (index, timestamp, atmospheric Cn², loss/error) to `EveController`

- Eve framework
    - `AttackStrategy` (abstract) and `EveController` (orchestration/stats)
    - Implemented strategies: Intercept-Resend, Adaptive, QBER-Adaptive (PID), GradientDescent QBER-Adaptive, BasisLearning (Beta), ParticleFilter learner, Photon Number Splitting (PNS), Channel/Atmospheric Adaptive

- Detection and information theory
    - `AttackDetector`: QBER (Hoeffding CI), chi-square basis balance, basis mutual information, runs test; unified verdict and history
    - Info-theory: `von_neumann_entropy`, `holevo_bound`, `bb84_holevo_bound`, `calculate_eve_information`, `mutual_information_eve_alice`

- Simulation and visualization
    - `AdaptiveJammingSimulator`: single-run orchestration, comparisons, CSV export, Welch t-tests (SciPy fallback)
    - `VisualizationManager`: QBER/time, intercept probability, info leakage, ROC/AUC, atmospheric visuals (Rytov, phase screens, beam frames), error/runs/MI, Holevo charts, LaTeX table export
    - Demos: `src/main/*_demo.py` per topic; end-to-end `examples/visualization_demo.py`

- Tests
    - Expanded coverage for Eve strategies, Bayesian learning, atmospheric modeling, PNS, visualization, and Holevo helpers
    - Snapshot: 155 passed, 1 skipped on Linux (Python 3.13)

## 2025-10-20 — Detection and information-theory foundations

- Initial `AttackDetector` with QBER CI, chi-square, basis MI, runs test
- Info-theory utilities (von Neumann entropy, Holevo bound, MI helpers)

## 2025-10-15 — Initial Eve and channel wiring

- `AttackStrategy` base class and `EveController`
- Intercept-Resend and Adaptive strategies
- `QuantumChannel` updated to optionally call Eve during transmit

---

## Notes

- Breaking changes: None; the channel accepts `eve` or `eavesdropper` for backwards compatibility
- Optional dependencies: Qiskit for the quantum backend; SciPy for eigensolvers/stats (with fallbacks)
- Outputs: Images and `summary.json` saved under `bb84_output/<run>/`

