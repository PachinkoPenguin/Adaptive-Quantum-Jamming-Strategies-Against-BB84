# AI Coding Agent Instructions - BB84 Quantum Jamming Project

## Architecture Overview

This is a **monolithic quantum cryptography research codebase** focused on BB84 protocol simulation with adaptive eavesdropping strategies. All core logic lives in `src/main/bb84_main.py` (~4735 lines). The monolith is intentional for educational/research purposes - avoid splitting unless explicitly requested.

### Core Components (in order of dependency)

1. **Quantum Backends** (`QuantumBackend`, `ClassicalBackend`, `QiskitBackend`): Abstract quantum simulation layer. ClassicalBackend is always available; QiskitBackend is optional and auto-detected.

2. **Protocol Core** (`BB84Protocol`, `QuantumChannel`, `Alice`, `Bob`): Implements 5-phase BB84 flow:
   - Phase 1: Quantum transmission (Alice → Channel → Bob)
   - Phase 2: Basis sifting (public channel, basis matching)
   - Phase 3: Error estimation (QBER calculation) → **triggers AttackDetector**
   - Phase 4: Error correction (CASCADE algorithm)
   - Phase 5: Privacy amplification (universal hashing)

3. **Eve Framework** (`AttackStrategy`, `EveController`): Strategy pattern for eavesdropping attacks. `QuantumChannel` integrates `EveController` and forwards rich metadata (index, timestamp, atmospheric Cn², loss/error rates) to strategies during transmission.

4. **Detection** (`AttackDetector`): Multi-test statistical detector using QBER Hoeffding CI, chi-square basis balance, basis mutual information, and runs test. Returns aggregated verdict after error estimation phase.

5. **Simulators** (`BB84Simulator`, `AdaptiveJammingSimulator`): Orchestrate parameter sweeps, strategy comparisons, and statistical analysis (t-tests, CSV exports).

6. **Visualization** (`VisualizationManager`): Publication-quality plotting (QBER evolution, ROC curves, atmospheric effects, Holevo bounds) and LaTeX table export.

7. **Information Theory**: Standalone functions (`von_neumann_entropy`, `holevo_bound`, `bb84_holevo_bound`, `calculate_eve_information`, `mutual_information_eve_alice`). Used by strategies and visualization.

### Data Flow & Integration Points

```
BB84Protocol.run_protocol()
  ├─> Alice.generate_key() → states
  ├─> QuantumChannel.transmit()
  │     ├─> apply loss
  │     ├─> apply channel noise
  │     └─> EveController.intercept_transmission() [if Eve present]
  │           └─> AttackStrategy.should_intercept() → intercept() → update_strategy()
  ├─> Bob.measure() → raw key
  ├─> Sifting (public basis exchange)
  ├─> Error estimation → QBER
  ├─> AttackDetector.detect_attack() [4 tests + aggregation]
  ├─> Send feedback to EveController (QBER + sifted bases)
  └─> Error correction + Privacy amplification → final key
```

**Key insight**: Eve intercepts *after* channel noise (realistic), and receives public feedback (QBER + bases) after error estimation to adapt strategies.

## Attack Strategy Development Pattern

When implementing new `AttackStrategy` subclasses:

1. **Required methods**:
   - `should_intercept(metadata: Dict) -> bool`: Decision function using transmission metadata
   - `intercept(photon, alice_basis=None) -> Tuple[photon, intercepted: bool]`: Measure & resend logic
   - `update_strategy(feedback: Dict) -> None`: Adapt based on public feedback (QBER, bases)

2. **Metadata available** (from `QuantumChannel`):
   - `index`, `timestamp`, `atmospheric_Cn2`, `channel_loss_rate`, `channel_error_rate`

3. **Common patterns**:
   - **QBER-based adaptation**: Target QBER ~10% (below 11% threshold). See `QBERAdaptiveStrategy` (PID controller) or `GradientDescentQBERAdaptive`.
   - **Basis learning**: Bayesian inference from public basis announcements. See `BasisLearningStrategy` or `ParticleFilterBasisLearner`.
   - **Atmospheric exploitation**: Adjust intercept probability with Rytov variance. See `ChannelAdaptiveStrategy` and `ATMOSPHERIC_IMPLEMENTATION.md`.

4. **Testing**: Add unit tests to `tests/test_eve.py` covering interception logic, feedback loops, and statistical properties.

## Development Workflows

### Running tests
```bash
# All tests (requires pytest from requirements-dev.txt)
pytest tests/

# Specific test file
pytest tests/test_eve.py -v

# With coverage
pytest --cov=src --cov-report=html tests/
```

### Running simulations
```bash
# Basic protocol run
python -c "
from src.main.bb84_main import BB84Protocol, ClassicalBackend, InterceptResendAttack
backend = ClassicalBackend()
protocol = BB84Protocol(backend=backend, attack_strategy=InterceptResendAttack(backend, 0.5))
protocol.setup(channel_loss=0.05, channel_error=0.02)
stats = protocol.run_protocol(num_bits=1000)
print(stats['qber'], stats['attack_detected'])
"

# Full visualization demo
python examples/visualization_demo.py
```

### Output artifacts
- **Timestamped directories**: `bb84_output/<backend>_<strategy>_run_<timestamp>/`
- **Per-run files**:
  - `protocol_log.png`, `summary_table.png`: Protocol visualization
  - `run.log`: Detailed logging
  - `summary.json`: Stats (QBER, efficiency, attack_detected, eve_interceptions, eve_information)
- **Demo outputs**: `demo_visualizations/` (from `examples/visualization_demo.py`)

## Project-Specific Conventions

### Naming patterns
- **Classes**: PascalCase (`BB84Protocol`, `InterceptResendAttack`, `EveController`)
- **Strategies**: End with `Attack` or `Strategy` (`QBERAdaptiveStrategy`, `PhotonNumberSplittingAttack`)
- **Functions**: snake_case (`von_neumann_entropy`, `calculate_eve_information`)
- **Metadata keys**: snake_case (`channel_loss_rate`, `atmospheric_Cn2`)

### Basis representation
- Use `Basis.RECTILINEAR` (+ basis: |0⟩, |1⟩) or `Basis.DIAGONAL` (× basis: |+⟩, |−⟩)
- **Never use strings** ("rectilinear" / "diagonal") - always use the enum

### QBER thresholds
- **Detection threshold**: 11% (industry standard)
- **Target for adaptive attacks**: ~10% (just below threshold)
- **Theoretical clean channel**: ~5%

### Backend handling
```python
# Check Qiskit availability
from src.main.bb84_main import QISKIT_AVAILABLE
if QISKIT_AVAILABLE:
    backend = QiskitBackend()
else:
    backend = ClassicalBackend()  # Fallback
```

### Information theory calculations
- Use `base=2` for bits (default in functions)
- BB84 Holevo bound is ~1 bit for optimal eavesdropping
- For intercept-resend: Eve's mutual information ≈ 0.5 bits

## Documentation References

- **`README.md`**: Quick start, project structure, installation
- **`QUICK_START.md`**: Minimal code examples
- **`EVE_IMPLEMENTATION.md`**: Attack strategy API, patterns, feedback loops
- **`VISUALIZATION_REFERENCE.md`**: VisualizationManager API, plot types
- **`QBER_ADAPTIVE_STRATEGIES.md`**: PID and gradient descent controllers
- **`ATMOSPHERIC_IMPLEMENTATION.md`**: Rytov variance, Hufnagel-Valley model
- **`PROJECT_STATUS.md`**: Test status, feature completeness

## Common Tasks

### Adding a new attack strategy
1. Subclass `AttackStrategy` in `src/main/bb84_main.py`
2. Implement required methods (see pattern above)
3. Add unit tests to `tests/test_eve.py`
4. Update strategy list in `AdaptiveJammingSimulator.compare_strategies()` call sites
5. Document in `EVE_IMPLEMENTATION.md` if novel

### Adding visualization plots
1. Add method to `VisualizationManager` class
2. Follow publication settings (DPI=150 display, 300 saved; font sizes 12/14/16pt)
3. Add test to `tests/test_visualization.py`
4. Update `VISUALIZATION_REFERENCE.md` with usage example

### Modifying detection logic
1. Edit `AttackDetector` class methods
2. Update aggregation logic in `detect_attack()` if adding new tests
3. Ensure `get_detection_history()` returns serializable dict
4. Add tests to `tests/test_attack_detection.py`

## Testing Philosophy

- **Unit tests**: Individual components (strategies, detector tests, entropy functions)
- **Integration tests**: Full protocol runs with Eve, backend comparisons
- **Deterministic**: Seed `np.random.seed()` and `random.seed()` for reproducibility
- **Current status**: 155 passed, 1 skipped (Linux Python 3.13, 2025-10-24)

## Dependencies

- **Core**: numpy, matplotlib, python-dateutil
- **Optional**: qiskit (≥1.0.0), qiskit-aer (≥0.12.0) - auto-detected
- **Dev**: pytest, pytest-cov, flake8, mypy, ruff

Install: `pip install -r requirements.txt` (add `-r requirements-dev.txt` for testing)
