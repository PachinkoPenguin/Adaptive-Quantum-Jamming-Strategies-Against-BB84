# BB84 Quantum Key Distribution Protocol Implementation

## Research Project Overview
This repository contains the implementation and analysis of the BB84 Quantum Key Distribution Protocol as part of a research paper investigating quantum cryptography and its security implications. The project focuses on both classical and quantum simulations using Qiskit, providing a comprehensive framework for studying the protocol's behavior under various conditions.

## Project structure
```
Adaptive-Quantum-Jamming-Strategies-Against-BB84/
├── src/
│   ├── main/                 # Canonical implementation (primary entrypoint)
│   │   └── bb84_main.py      # Canonical BB84 implementation and CLI
│   ├── bb84_main.py          # Compatibility wrapper -> re-exports from main.bb84_main
│   └── bb84_education.py     # Educational/legacy implementation (kept for reference)
├── tests/                    # Test files
│   ├── test_bb84_qiskit.py   # BB84 backend tests
│   └── test_qiskit.py        # Qiskit environment tests
├── docs/                     # Documentation
└── requirements.txt          # Project dependencies
```

## Main Components

### 1. BB84 Foundation (Main Implementation)
- `src/main/bb84_main.py` is the core implementation featuring:
- Hybrid backend support (Classical and Qiskit)
- Complete BB84 protocol implementation
- Extensive visualization and analysis tools
- Quantum channel simulation with noise and loss
- Performance metrics and security analysis
- Parameter sweep capabilities

Key features:
- Modular architecture with separate Alice, Bob, and Channel classes
- Support for both classical and quantum backends
- Comprehensive error analysis and QBER calculation
- Detailed visualization of protocol stages
- Automated report generation

### 2. Supporting components
- `src/bb84_education.py`: Educational implementation and notes
- `src/bb84_main.py`: Lightweight compatibility wrapper (re-exports canonical symbols from `main.bb84_main`)
- `tests/test_bb84_qiskit.py`: Backend comparison tests
- `tests/test_qiskit.py`: Qiskit environment verification

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command line usage

The canonical runner is `src/main/bb84_main.py`. There is also a small compatibility wrapper at `src/bb84_main.py` so older invocation paths continue to work.

Examples:

```bash
# Run with classical backend (default)
python src/main/bb84_main.py

# Run with Qiskit backend (if available)
python src/main/bb84_main.py --use-qiskit

# Show help and available options
python src/main/bb84_main.py --help

# Using compatibility wrapper (same behaviour)
python src/bb84_main.py --help
```

Each run performs three types of analysis:
1. Basic simulation with selected backend
2. Parameter sweep across different error and loss rates
3. Comparison between classical and quantum backends

### Output Structure

The simulation generates several output files in the `bb84_output` directory:

1. Basic Simulation Outputs:
   - `protocol_execution.png`: Detailed log of the protocol execution
   - `protocol_summary.png`: Summary statistics table
   - `protocol_visualization.png`: Visualization of key generation pipeline
   - `summary_report_TIMESTAMP.png`: Comprehensive summary report

2. Parameter Sweep Outputs:
   - `parameter_sweep_heatmap.png`: Three heatmaps showing QBER, Efficiency, and Security Status

3. Backend Comparison Outputs:
   - `backend_comparison.png`: Boxplots comparing QBER and Efficiency between backends

Each run creates a descriptive timestamped directory under `bb84_output/` using the pattern:

```
{backend}_{run_type}_run_{YYYYMMDD_HHMMSS}
```

Example: `classical_demo_run_20251021_165101` or `qiskit_standard_run_20251021_165341`.

Inside each run directory you'll typically find image files, a `run.log` containing run-time logs, and `summary.json` with run metadata and statistics.

### CLI options (new)

The main runner supports a few useful flags to control output and logging:

- `--output-dir`: Path where run artifacts are saved (default: `bb84_output/`).
- `--run-type`: Short label for the run (e.g., `standard`, `sweep`, `comparison`) used in output folder names.
- `--log-level`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Default is `INFO`.

Example:

```bash
python src/main/bb84_main.py --output-dir bb84_output --run-type standard --log-level INFO
```

Each run creates a descriptive folder name that includes the backend, run type, and timestamp. Inside the folder you'll find image files, a `run.log` containing run-time logs, and `summary.json` with run metadata and statistics.

### summary.json schema

Each run saves a small JSON file `summary.json` with the following keys:

- `backend`: `classical` or `qiskit`
- `run_type`: The `--run-type` value
- `timestamp`: ISO 8601 timestamp of the run
- `total_bits_sent`, `raw_key_length`, `final_key_length`, `qber`, `efficiency`
- `channel_loss_rate`, `channel_error_rate`

This file is intended for quick programmatic inspection of run results.

### Programmatic Usage

For programmatic use, import the canonical symbols from the `main` package. A couple of valid examples:

```python
from main.bb84_main import BB84Protocol, BB84Simulator

# or, since src/main/__init__.py exports the main classes:
from main import BB84Protocol, BB84Simulator

# Create and run protocol (classical backend)
protocol = BB84Protocol(use_qiskit=False, output_dir='bb84_output', run_type='demo')
protocol.setup(channel_loss=0.1, channel_error=0.02)
stats = protocol.run_protocol(num_bits=1000)

# Using the simulator helper
sim = BB84Simulator(use_qiskit=False, output_dir='bb84_output', run_type='demo')
stats = sim.run_single_simulation(num_bits=200)
```

### Parameter Sweep
```python
from main.bb84_main import BB84Simulator

simulator = BB84Simulator()
results = simulator.run_parameter_sweep(
   num_bits=500,
   loss_rates=[0.0, 0.1, 0.2],
   error_rates=[0.0, 0.05, 0.10]
)
```

### Understanding Results

1. **QBER (Quantum Bit Error Rate)**:
   - Values below 11% indicate secure key distribution
   - Higher values may indicate eavesdropping or channel noise

2. **Efficiency**:
   - Shows the ratio of final key length to initial bits sent
   - Affected by loss rate and error rate
   - Typically ranges from 15% to 25%

3. **Security Status**:
   - Determined by QBER threshold of 11%
   - Automated security assessment in reports
   - Visual indicators in parameter sweep heatmaps

### Noise modeling and channel effects

1. **Classical backend**
   - Simple bit-flip error model
   - Probabilistic photon loss
   - Fast execution for initial testing

2. **Qiskit backend**
   - Realistic quantum noise modeling using `NoiseModel`
   - Depolarizing channel errors
   - Hardware-inspired noise effects
   - Quantum circuit-level simulation

### Visualization Features

1. **Protocol Execution**:
   - Real-time logging of protocol stages
   - Quantum state preparation and measurement
   - Basis reconciliation statistics
   - Error detection and correction phases

2. **Statistical Analysis**:
   - Box plots for backend comparison
   - Heatmaps for parameter sensitivity
   - Key generation pipeline visualization
   - Security threshold indicators

3. **Report Generation**:
   - Comprehensive summary reports
   - Statistical metrics and analysis
   - Security assessment
   - Performance comparisons

## Research Context
This implementation is part of a research paper investigating:
1. BB84 protocol security under various channel conditions
2. Comparison between classical and quantum simulations
3. Impact of noise and loss on key generation
4. Security thresholds and QBER analysis
5. Potential vulnerabilities and attack scenarios

## Dependencies
- numpy>=1.19.0
- matplotlib>=3.3.0
- qiskit>=1.0.0 (optional)
- qiskit-aer>=0.12.0 (optional)
- python-dateutil>=2.8.0

## Future Work
- Implementation of quantum jamming scenarios
- Extended analysis of eavesdropping attacks
- Integration with hardware quantum devices
- Advanced error correction mechanisms
- Privacy amplification improvements

## References
1. Bennett, C. H., & Brassard, G. (1984). Quantum cryptography: Public key distribution and coin tossing.
2. Additional research paper references will be added here...

## License
This project is part of an academic research paper. All rights reserved.