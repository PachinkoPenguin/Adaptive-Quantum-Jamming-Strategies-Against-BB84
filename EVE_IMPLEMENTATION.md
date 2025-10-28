# Eve (Eavesdropper) Implementation for BB84 Protocol

## Overview

This implementation adds quantum eavesdropping capabilities to the existing BB84 quantum key distribution protocol simulator. The design follows object-oriented principles with an abstract strategy pattern, allowing for flexible implementation of various attack strategies.

## Architecture

### 1. Abstract Base Class: `AttackStrategy`

The `AttackStrategy` class defines the interface for all eavesdropping strategies.

```python
class AttackStrategy(ABC):
    """Abstract base class for Eve's attack strategies"""
    
    def __init__(self):
        self.statistics = {
            'interceptions': 0,
            'successful_measurements': 0,
            'information_gained': 0.0,
            'detected_interceptions': 0
        }
    
    @abstractmethod
    def intercept(self, qubit, alice_basis=None) -> Tuple[Any, bool]:
        """Intercept and potentially modify a qubit"""
        pass
    
    @abstractmethod
    def should_intercept(self, metadata: Dict) -> bool:
        """Decide whether to intercept based on metadata"""
        pass
    
    @abstractmethod
    def update_strategy(self, feedback: Dict) -> None:
        """Update strategy based on feedback"""
        pass
    
    def get_statistics(self) -> Dict:
        """Get attack statistics"""
        return self.statistics.copy()
```

**Key Methods:**
- `intercept()`: Core interception logic - measures and resends qubits
- `should_intercept()`: Decision function for selective interception
- `update_strategy()`: Adapts strategy based on observed QBER
- `get_statistics()`: Returns metrics about the attack

### 2. Controller Class: `EveController`

The `EveController` manages Eve's operations and coordinates with the attack strategy.

```python
class EveController:
    """Controller for Eve's eavesdropping operations"""
    
    def __init__(self, attack_strategy: AttackStrategy, 
                 backend: QuantumBackend, name: str = "Eve"):
        self.attack_strategy = attack_strategy
        self.backend = backend
        self.qber_history: List[float] = []
        self.basis_observations: Dict[str, int] = {}
        self.intercepted_qubits: List[Dict] = []
```

**Key Features:**
- Tracks QBER history for adaptive strategies
- Records basis observations from public announcements
- Maintains detailed logs of intercepted qubits
- Provides feedback loop for strategy adaptation

### 3. Modified `QuantumChannel`

The quantum channel now accepts an optional `EveController` parameter (also available as `eavesdropper` for clarity) and forwards rich metadata to strategies.

```python
class QuantumChannel:
    def __init__(self, backend: QuantumBackend,
                 loss_rate: float = 0.0,
                 error_rate: float = 0.0,
                 eve: Optional[EveController] = None,
                 name: str = "Quantum Channel"):
        # ... initialization
        self.eve = eve
```

**Transmission Order:**
1. Loss: probabilistic photon loss according to `loss_rate`
2. Noise: natural channel errors via backend `apply_channel_noise`
3. Eve: if present, `EveController.intercept_transmission` is invoked with metadata:
    - `index`, `timestamp` (deterministic placeholder), `atmospheric_Cn2`, `channel_loss_rate`, `channel_error_rate`
4. Return (possibly modified) state or `None` if lost

This order ensures that:
- Natural channel conditions affect the qubit first
- Eve intercepts already-noisy qubits (realistic)
- Eve's attacks are additive to natural errors

## Implemented Attack Strategies

### 1. InterceptResendAttack

Classic intercept-resend attack where Eve measures each qubit in a random basis and resends.

**Characteristics:**
- Simple but detectable
- 100% interception causes ~25% QBER
- Can be tuned with `intercept_probability` parameter

**Usage:**
```python
backend = ClassicalBackend()
eve_strategy = InterceptResendAttack(backend, intercept_probability=1.0)
eve = EveController(eve_strategy, backend)
channel = QuantumChannel(backend, eve=eve)
```

**Expected QBER:**
- 100% interception: ~25% QBER (detected)
- 50% interception: ~12.5% QBER (likely detected)
- 30% interception: ~7.5% QBER (undetected)

### 2. AdaptiveAttack

Sophisticated attack that adjusts interception rate to maintain target QBER below detection threshold.

**Characteristics:**
- Adaptive to observed QBER
- Tries to maximize information gain while staying undetected
- Uses recent QBER history for decision making

**Usage:**
```python
eve_strategy = AdaptiveAttack(
    backend, 
    target_qber=0.08,           # Target 8% QBER
    initial_intercept_rate=0.5   # Start at 50%
)
eve = EveController(eve_strategy, backend)
```

**Adaptation Logic:**
- If QBER > target + 2%: Reduce interception by 15%
- If QBER < target - 2%: Increase interception by 15%
- Bounded between 1% and 100%

## Integration Points

### With Existing BB84 Protocol

The implementation integrates seamlessly with the protocol and detector:

```python
from src.main.bb84_main import BB84Protocol, ClassicalBackend, InterceptResendAttack
backend = ClassicalBackend()
protocol = BB84Protocol(backend=backend, attack_strategy=InterceptResendAttack(backend, 0.5))
protocol.setup(channel_loss=0.05, channel_error=0.02)
stats = protocol.run_protocol(num_bits=1000)  # Runs detection and sends feedback to Eve
```

### Feedback Loop and Detector Interaction

After error estimation (phase 3), the protocol:
- Runs `AttackDetector.detect_attack` over the observed data
- Sends Eve feedback with `qber` and public bases (and key length)

```python
eve_public_info = {
    'alice_bases': protocol.alice.bases,
    'bob_bases': protocol.bob.bases,
    'key_length': len(protocol.raw_key_alice)
}
# Protocol already calls:
# protocol.channel.eve.receive_feedback(protocol.qber, eve_public_info)
```

## Statistics and Metrics

### AttackStrategy Statistics

```python
strategy_stats = eve.get_statistics()
# Returns:
{
    'interceptions': 250,              # Total interceptions
    'successful_measurements': 250,     # Successful measurements
    'information_gained': 125.0,        # Bits of information
    'detected_interceptions': 0         # Detected attacks
}
```

### EveController Statistics

```python
eve_stats = eve.get_statistics()
# Returns:
{
    'name': 'Eve',
    'total_intercepted': 250,
    'qber_history': [0.08, 0.09, 0.08],
    'basis_observations': {
        'rectilinear_observed': 500,
        'diagonal_observed': 500
    },
    'attack_strategy_stats': {...},
    'avg_qber': 0.083
}
```

## Creating Custom Attack Strategies

### Step 1: Inherit from AttackStrategy

```python
class MyCustomAttack(AttackStrategy):
    def __init__(self, backend: QuantumBackend, **params):
        super().__init__()
        self.backend = backend
        # Custom initialization
```

### Step 2: Implement Required Methods

```python
    def should_intercept(self, metadata: Dict) -> bool:
        # Custom decision logic
        return True  # or False based on metadata
    
    def intercept(self, qubit: Any, alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        # Measure qubit
        eve_basis = Basis.random()  # or custom basis selection
        measured_bit = self.backend.measure_state(qubit, eve_basis)
        
        # Resend qubit
        new_qubit = self.backend.prepare_state(measured_bit, eve_basis)
        
        # Update statistics
        self.statistics['interceptions'] += 1
        
        return (new_qubit, True)
    
    def update_strategy(self, feedback: Dict) -> None:
        # Adapt based on feedback
        qber = feedback.get('qber', 0.0)
        # Custom adaptation logic
```

### Step 3: Use Your Custom Strategy

```python
backend = ClassicalBackend()
custom_strategy = MyCustomAttack(backend, param1=value1)
eve = EveController(custom_strategy, backend)
channel = QuantumChannel(backend, eve=eve)
```

## Example Use Cases

### 1. Security Testing

Test protocol resilience against various attack levels:

```python
for intercept_rate in [0.1, 0.3, 0.5, 0.7, 1.0]:
    strategy = InterceptResendAttack(backend, intercept_rate)
    eve = EveController(strategy, backend)
    channel = QuantumChannel(backend, eve=eve)
    # Run protocol and measure QBER
```

### 2. Threshold Detection

Find the maximum undetectable interception rate:

```python
target_qber = 0.10  # Just below 11% threshold
strategy = AdaptiveAttack(backend, target_qber=target_qber)
eve = EveController(strategy, backend)
# Run multiple rounds to converge
```

### 3. Comparing Backends

Test how quantum vs classical simulation affects Eve's success:

```python
for backend_type in [ClassicalBackend(), QiskitBackend()]:
    strategy = InterceptResendAttack(backend_type, 0.5)
    eve = EveController(strategy, backend_type)
    # Compare QBER in different backends
```

## Demo

See `examples/visualization_demo.py` for a full-stack example including simulations and figures. For targeted strategy tests, use the unit tests in `tests/`.

## Performance Considerations

### Classical Backend
- Fast simulation
- Suitable for large-scale parameter sweeps
- Approximate quantum behavior

### Qiskit Backend
- Realistic quantum simulation
- Slower but more accurate
- Includes quantum noise models

### Scalability
- Attack strategies scale linearly with number of qubits
- Statistics tracking has O(1) per-qubit overhead
- Adaptive strategies require O(k) memory for k-round history

## Security Analysis

### Detection Probability

For intercept-resend attack on n qubits with interception rate p:

- **Expected QBER**: `QBER ≈ p × 0.25`
- **Detection threshold**: 11%
- **Maximum undetectable rate**: ~44% (theoretical)
- **Practical safe rate**: ~30% (accounting for variance)

### Information Gain

Eve's information gain per intercepted qubit:

- **Correct basis (50% chance)**: 1 bit of information
- **Wrong basis (50% chance)**: 0 bits of information
- **Average**: 0.5 bits per interception
- **Net gain**: `0.5 × interception_rate × key_length`

## Future Extensions

### Possible Enhancements

1. **Photon Number Splitting (PNS) Attack**
   - Exploit multi-photon pulses
   - More sophisticated than intercept-resend

2. **Trojan Horse Attack**
   - Probe Alice's and Bob's equipment
   - Requires device model integration

3. **Basis-Choice Attack**
   - Try to infer basis from side channels
   - Metadata-driven strategy

4. **Collective Measurements**
   - Store qubits and measure after basis announcement
   - Requires quantum memory model

5. **Machine Learning Strategies**
   - Learn optimal interception patterns
   - Predict basis choices

### Implementation Template

```python
class FutureAttack(AttackStrategy):
    """Template for new attack strategy"""
    
    def __init__(self, backend: QuantumBackend):
        super().__init__()
        self.backend = backend
        # Add attack-specific parameters
    
    def should_intercept(self, metadata: Dict) -> bool:
        # Implement decision logic
        pass
    
    def intercept(self, qubit: Any, alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        # Implement interception logic
        pass
    
    def update_strategy(self, feedback: Dict) -> None:
        # Implement adaptation logic
        pass
```

## References

### BB84 Protocol
- Bennett, C. H., & Brassard, G. (1984). "Quantum cryptography: Public key distribution and coin tossing"

### Quantum Eavesdropping
- Fuchs, C. A., et al. (1997). "Optimal eavesdropping in quantum cryptography"
- Lütkenhaus, N. (2000). "Security against individual attacks for realistic quantum key distribution"

### Attack Strategies
- Brassard, G., et al. (2000). "Limitations on practical quantum cryptography"
- Scarani, V., et al. (2009). "The security of practical quantum key distribution"

## License

This implementation extends the existing BB84 simulator codebase and follows the same licensing terms.

## Contact

For questions or contributions related to the Eve implementation, please refer to the main project repository.
