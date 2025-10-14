# BB84 Quantum Key Distribution Protocol Implementation

## Research Project Overview
This repository contains the implementation and analysis of the BB84 Quantum Key Distribution Protocol as part of a research paper investigating quantum cryptography and its security implications. The project focuses on both classical and quantum simulations using Qiskit, providing a comprehensive framework for studying the protocol's behavior under various conditions.

## Project Structure
```
Investigation/
├── src/                  # Source code files
│   ├── bb84.py          # Basic BB84 implementation
│   └── bb84_foundation.py # Advanced hybrid implementation (main code)
├── tests/               # Test files
│   ├── test_bb84_qiskit.py # BB84 backend tests
│   └── test_qiskit.py   # Qiskit environment tests
├── docs/               # Documentation
└── requirements.txt    # Project dependencies
```

## Main Components

### 1. BB84 Foundation (Main Implementation)
`bb84_foundation.py` is the core implementation featuring:
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

### 2. Supporting Components
- `bb84.py`: Basic educational implementation
- `test_bb84_qiskit.py`: Backend comparison tests
- `test_qiskit.py`: Qiskit environment verification

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation
```python
from src.bb84_foundation import BB84Protocol

# Create and run protocol
protocol = BB84Protocol(use_qiskit=False)  # Use classical backend
protocol.setup(channel_loss=0.1, channel_error=0.02)
stats = protocol.run_protocol(num_bits=1000)
```

### Parameter Sweep
```python
from src.bb84_foundation import BB84Simulator

simulator = BB84Simulator()
results = simulator.run_parameter_sweep(
    num_bits=500,
    loss_rates=[0.0, 0.1, 0.2],
    error_rates=[0.0, 0.05, 0.10]
)
```

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