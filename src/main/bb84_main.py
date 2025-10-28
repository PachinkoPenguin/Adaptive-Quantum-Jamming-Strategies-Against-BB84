"""
BB84 Quantum Key Distribution Protocol - Hybrid Foundation Implementation
Author: [Your Name]
Date: 2024
Description: Complete educational implementation with Classical/Qiskit backends
             and image-based output for terminal compatibility
"""

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from collections import Counter
from abc import ABC, abstractmethod
from datetime import datetime
import os
import logging
import json
from typing import Iterable

# Optional SciPy import for high-quality eigenvalue routines
try:
    from scipy.linalg import eigvalsh as scipy_eigvalsh  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:
    scipy_eigvalsh = None
    _SCIPY_AVAILABLE = False

# Module logger
logger = logging.getLogger("bb84")

# Try to import Qiskit (optional)
QISKIT_AVAILABLE = False
QuantumCircuit = None  # Default to None

try:
    # Test with Qiskit 1.0+ imports
    from qiskit import QuantumCircuit
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    QISKIT_AVAILABLE = True
    print("✓ Qiskit detected and loaded successfully! (Version 1.0+)")
except ImportError as e:
    # Try older Qiskit API
    try:
        from qiskit import QuantumCircuit, Aer, execute
        from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
        QISKIT_AVAILABLE = True
        print("✓ Qiskit detected (older version)")
    except ImportError:
        print("✗ Qiskit not available. Using classical simulation only.")
        print("  To enable Qiskit: pip install qiskit qiskit-aer")

# Create output directory for images
OUTPUT_DIR = "bb84_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# FUNDAMENTAL QUANTUM COMPONENTS
# ============================================================================

class Basis(Enum):
    """Measurement basis for quantum states"""
    RECTILINEAR = 0  # + basis: {|0⟩, |1⟩} = {|H⟩, |V⟩}
    DIAGONAL = 1     # × basis: {|+⟩, |−⟩} = {|45°⟩, |135°⟩}
    
    @staticmethod
    def random():
        """Generate random basis choice"""
        return Basis(random.randint(0, 1))
    
    def __str__(self):
        return "+" if self == Basis.RECTILINEAR else "×"


@dataclass
class Photon:
    """
    Represents a quantum photon with polarization state
    In reality, this would be a quantum state |ψ⟩
    """
    bit_value: int  # 0 or 1
    basis: Basis    # Preparation basis
    polarization: float  # Angle in degrees
    
    def __init__(self, bit: int, basis: Basis):
        self.bit_value = bit
        self.basis = basis
        
        # Set polarization based on bit and basis
        if basis == Basis.RECTILINEAR:
            self.polarization = 0.0 if bit == 0 else 90.0  # H or V
        else:  # DIAGONAL
            self.polarization = 45.0 if bit == 0 else 135.0  # + or -


# ============================================================================
# QUANTUM BACKEND ABSTRACTION
# ============================================================================

class QuantumBackend(ABC):
    """Abstract base class for quantum simulation backends"""
    
    @abstractmethod
    def prepare_state(self, bit: int, basis: Basis) -> Any:
        """Prepare a quantum state"""
        pass
    
    @abstractmethod
    def measure_state(self, state: Any, basis: Basis) -> int:
        """Measure a quantum state"""
        pass
    
    @abstractmethod
    def apply_channel_noise(self, state: Any, error_rate: float) -> Any:
        """Apply channel noise to state"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return backend name"""
        pass


class ClassicalBackend(QuantumBackend):
    """Classical simulation backend - fast and simple"""
    
    def prepare_state(self, bit: int, basis: Basis) -> Photon:
        """Prepare a classical representation of quantum state"""
        return Photon(bit, basis)
    
    def measure_state(self, state: Photon, basis: Basis) -> int:
        """Measure state in given basis"""
        if basis == state.basis:
            # Same basis - perfect correlation
            return state.bit_value
        else:
            # Different basis - random result
            return random.randint(0, 1)
    
    def apply_channel_noise(self, state: Photon, error_rate: float) -> Photon:
        """Apply bit flip error with given probability"""
        if random.random() < error_rate:
            state.bit_value = 1 - state.bit_value
        return state
    
    def get_name(self) -> str:
        return "Classical Simulation"


# Only define QiskitBackend if Qiskit is available
if QISKIT_AVAILABLE:
    class QiskitBackend(QuantumBackend):
        """Qiskit quantum simulation backend - realistic but slower"""
        
        def __init__(self):
            # Use the new Qiskit 1.0+ API
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator()
            self.noise_model = None
        
        def prepare_state(self, bit: int, basis: Basis) -> QuantumCircuit:
            """Prepare BB84 quantum state using Qiskit"""
            qc = QuantumCircuit(1, 1)
            
            # Prepare state based on bit value
            if bit == 1:
                qc.x(0)  # |1⟩ state
                
            # Apply basis rotation
            if basis == Basis.DIAGONAL:
                qc.h(0)  # Convert to diagonal basis
                
            return qc
        
        def measure_state(self, state: QuantumCircuit, basis: Basis) -> int:
            """Measure BB84 state in given basis"""
            # Copy circuit to avoid modifying original
            qc = state.copy()
            
            # Rotate to measurement basis
            if basis == Basis.DIAGONAL:
                qc.h(0)
            
            # Add measurement
            qc.measure(0, 0)
            
            # Execute circuit using new API
            from qiskit import transpile
            transpiled = transpile(qc, self.backend)
            
            # Run with noise model if configured
            if self.noise_model:
                job = self.backend.run(transpiled, shots=1, noise_model=self.noise_model)
            else:
                job = self.backend.run(transpiled, shots=1)
                
            result = job.result()
            counts = result.get_counts(qc)
            
            # Return measured bit
            return int(list(counts.keys())[0])
        
        def apply_channel_noise(self, state: QuantumCircuit, error_rate: float) -> QuantumCircuit:
            """Configure noise model for simulation"""
            if error_rate > 0:
                from qiskit_aer.noise import NoiseModel, depolarizing_error
                self.noise_model = NoiseModel()
                error = depolarizing_error(error_rate, 1)
                self.noise_model.add_all_qubit_quantum_error(error, ['x', 'h'])
            else:
                self.noise_model = None
            return state
        
        def get_name(self) -> str:
            return "Qiskit Quantum Simulation"
else:
    # Define a placeholder if Qiskit is not available
    class QiskitBackend(QuantumBackend):
        """Placeholder for when Qiskit is not installed"""
        
        def __init__(self):
            raise ImportError("Qiskit is not installed. Please run: pip install qiskit")
        
        def prepare_state(self, bit: int, basis: Basis):
            raise NotImplementedError("Qiskit not available")
        
        def measure_state(self, state, basis: Basis):
            raise NotImplementedError("Qiskit not available")
        
        def apply_channel_noise(self, state, error_rate: float):
            raise NotImplementedError("Qiskit not available")
        
        def get_name(self) -> str:
            return "Qiskit (Not Installed)"


# ============================================================================
# OUTPUT MANAGER FOR IMAGE-BASED RESULTS
# ============================================================================

class OutputManager:
    """Manages all output as images instead of terminal text"""
    
    def __init__(self, output_dir: str = OUTPUT_DIR, backend_type: str = "classical", run_type: str = "standard"):
        """
        Initialize OutputManager with more descriptive directory structure
        Args:
            output_dir: Base output directory
            backend_type: Type of simulation backend ('classical' or 'qiskit')
            run_type: Type of run ('standard', 'sweep', 'comparison', etc.)
        """
        self.output_dir = output_dir
        self.backend_type = backend_type
        self.run_type = run_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a more descriptive run directory name
        run_name = f"{backend_type}_{run_type}_run_{timestamp}"
        self.run_dir = os.path.join(output_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_data = []
        # Setup a logger for this run
        logger_name = f"bb84.{self.backend_type}.{self.run_type}.{timestamp}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler(os.path.join(self.run_dir, "run.log"))
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
    def log(self, text: str, category: str = "info"):
        """Add text to log buffer"""
        self.log_data.append((category, text))
    
    def save_log_image(self, filename: str = "protocol_log"):
        """Save log as an image"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Group log entries by category
        y_position = 0.95
        colors = {
            'header': 'blue',
            'phase': 'darkgreen',
            'info': 'black',
            'warning': 'orange',
            'error': 'red',
            'success': 'green',
            'result': 'purple'
        }
        
        for category, text in self.log_data:
            color = colors.get(category, 'black')
            weight = 'bold' if category in ['header', 'phase', 'result'] else 'normal'
            size = 12 if category == 'header' else 10
            
            ax.text(0.05, y_position, text, 
                   fontsize=size, color=color, weight=weight,
                   transform=ax.transAxes, 
                   fontfamily='monospace')
            y_position -= 0.03
            
            if y_position < 0.05:
                break
        
        title = f"BB84 Protocol Log - {self.backend_type.capitalize()} {self.run_type.capitalize()}"
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Create descriptive filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        descriptive_filename = f"{self.backend_type}_{self.run_type}_{filename}_{timestamp}.png"
        filepath = os.path.join(self.run_dir, descriptive_filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        try:
            self.logger.info(f"Saved log image: {filepath}")
        except Exception:
            pass
        return filepath
    
    def create_summary_table(self, stats: Dict, filename: str = "summary"):
        """Create summary table as image"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for key, value in stats.items():
            if isinstance(value, float):
                if 'rate' in key.lower() or 'qber' in key.lower() or 'efficiency' in key.lower():
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            # Make key more readable
            readable_key = key.replace('_', ' ').title()
            table_data.append([readable_key, formatted_value])
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data) + 1):
            if i == 0:  # Header row
                table[(i, 0)].set_facecolor('#40466e')
                table[(i, 1)].set_facecolor('#40466e')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                color = '#f1f1f2' if i % 2 == 0 else 'white'
                table[(i, 0)].set_facecolor(color)
                table[(i, 1)].set_facecolor(color)
        
        title = f"BB84 Protocol Summary - {self.backend_type.capitalize()} {self.run_type.capitalize()}"
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        filepath = os.path.join(self.run_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        try:
            self.logger.info(f"Saved summary table image: {filepath}")
        except Exception:
            pass
        return filepath

    def save_metadata(self, stats: Dict, params: Dict = None, filename: str = "summary.json"):
        """Save run metadata and statistics to JSON file next to outputs"""
        data = {
            "stats": stats,
            "params": params or {},
            "generated_at": datetime.now().isoformat(),
        }
        filepath = os.path.join(self.run_dir, filename)
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved metadata JSON: {filepath}")
        except Exception as e:
            try:
                self.logger.error(f"Failed to save metadata JSON: {e}")
            except Exception:
                pass
        return filepath


# ============================================================================
# ALICE - QUANTUM TRANSMITTER (Modified for backends)
# ============================================================================

class Alice:
    """
    Alice: The sender in QKD protocol
    """
    
    def __init__(self, backend: QuantumBackend, name: str = "Alice"):
        self.name = name
        self.backend = backend
        self.bits: List[int] = []
        self.bases: List[Basis] = []
        self.sent_states: List = []
        
    def generate_random_bits(self, n: int) -> List[int]:
        """Generate n random bits for transmission"""
        self.bits = [random.randint(0, 1) for _ in range(n)]
        return self.bits
    
    def choose_random_bases(self, n: int) -> List[Basis]:
        """Choose random basis for each bit"""
        self.bases = [Basis.random() for _ in range(n)]
        return self.bases
    
    def prepare_states(self) -> List:
        """Prepare quantum states based on bits and bases"""
        assert len(self.bits) == len(self.bases), "Bits and bases must have same length"
        
        self.sent_states = []
        for bit, basis in zip(self.bits, self.bases):
            state = self.backend.prepare_state(bit, basis)
            self.sent_states.append(state)
        
        return self.sent_states
    
    def announce_bases(self) -> List[Basis]:
        """Publicly announce bases used"""
        return self.bases


# ============================================================================
# BOB - QUANTUM RECEIVER (Modified for backends)
# ============================================================================


# ============================================================================
# INTEGRATED ADAPTIVE JAMMING SIMULATOR
# ============================================================================

class AdaptiveJammingSimulator:
    """
    Integrated BB84 simulator combining attacks and detection.
    """

    PRESETS = {
        'laboratory':   {'loss': 0.0,  'error': 0.01},
        'urban':        {'loss': 0.05, 'error': 0.03},
        'satellite':    {'loss': 0.20, 'error': 0.08},
        'adversarial':  {'loss': 0.05, 'error': 0.12},
    }

    def __init__(self,
                 n_qubits: int = 1000,
                 attack_strategy: Optional["AttackStrategy"] = None,
                 detector_params: Optional[Dict] = None,
                 backend: Optional["QuantumBackend"] = None,
                 preset: Optional[str] = None,
                 loss_rate: float = 0.0,
                 error_rate: float = 0.02):
        self.n_qubits = n_qubits
        self.backend = backend or ClassicalBackend()
        self.detector = AttackDetector(**(detector_params or {'qber_threshold': 0.11}))

        # Channel conditions
        if preset and preset in self.PRESETS:
            cfg = self.PRESETS[preset]
            loss_rate = cfg['loss']
            error_rate = cfg['error']
        self.loss_rate = loss_rate
        self.error_rate = error_rate

        # Eve setup
        self.attack_strategy = attack_strategy
        self.eve: Optional["EveController"] = None
        if self.attack_strategy is not None:
            self.eve = EveController(self.attack_strategy, self.backend)

        # Quantum channel
        self.channel = QuantumChannel(self.backend, loss_rate=self.loss_rate, error_rate=self.error_rate, eve=self.eve)

        # Parties
        self.alice = Alice(self.backend)
        self.bob = Bob(self.backend)

        # Results storage
        self.results_history: List[Dict] = []

    def run_simulation(self) -> Dict:
        """
        Run the BB84 simulation with optional eavesdropping and detection.
        Returns a comprehensive results dictionary.
        """
        n = self.n_qubits

        # Alice prepares
        alice_bits = self.alice.generate_random_bits(n)
        alice_bases = self.alice.choose_random_bases(n)
        states = self.alice.prepare_states()

        # Transmit through channel (Eve may intercept inside channel)
        received = self.channel.transmit(states)

        # Map valid indices (not lost)
        valid_indices = [i for i, s in enumerate(received) if s is not None]
        valid_states = [received[i] for i in valid_indices]

        # Bob measures only valid states
        self.bob.choose_random_bases(len(valid_states))
        bob_bits_valid = self.bob.measure_states(valid_states)

        # Build full-length bob bits and bases with None for losses
        bob_bits: List[Optional[int]] = [None] * n
        bob_bases_full: List[Optional[Basis]] = [None] * n
        for j, idx in enumerate(valid_indices):
            bob_bits[idx] = bob_bits_valid[j]
            bob_bases_full[idx] = self.bob.bases[j]

        # Basis reconciliation (sifting)
        matching_indices = [i for i in valid_indices if bob_bases_full[i] == alice_bases[i] and bob_bits[i] is not None]
        raw_key_alice = [alice_bits[i] for i in matching_indices]
        raw_key_bob = [bob_bits[i] for i in matching_indices]

        # QBER and confidence intervals via detector
        qber_result = self.detector.detect_qber(alice_bits, bob_bits, alice_bases, bob_bases_full, sample_size=300)
        qber = qber_result.get('qber', 0.0) or 0.0

        # Attack detection
        detection = self.detector.detect_attack(alice_bits, bob_bits, alice_bases, bob_bases_full, sample_size=300)

        # Feedback to Eve (basis announcement)
        if self.eve is not None:
            public_info = {'alice_bases': alice_bases}
            self.eve.receive_feedback(qber, public_info=public_info)

        # Build results
        eve_stats = self.eve.get_statistics() if self.eve is not None else None
        channel_stats = {
            'transmitted': self.channel.transmitted_count,
            'lost': self.channel.lost_count,
            'error_events': self.channel.error_count,
            'loss_rate': self.loss_rate,
            'error_rate': self.error_rate,
        }

        results = {
            'n_qubits': n,
            'alice_bits': alice_bits,
            'alice_bases': alice_bases,
            'bob_bases': bob_bases_full,
            'bob_bits': bob_bits,
            'valid_indices': valid_indices,
            'matching_indices': matching_indices,
            'raw_key_alice': raw_key_alice,
            'raw_key_bob': raw_key_bob,
            'qber': qber,
            'qber_ci': (qber_result.get('ci_lower'), qber_result.get('ci_upper')),
            'qber_result': qber_result,
            'detection': detection,
            'channel': channel_stats,
            'eve': eve_stats,
        }

        self.results_history.append(results)
        return results

    def compare_strategies(self, strategies_list: List["AttackStrategy"]) -> Dict[str, Dict]:
        """Run simulation for each strategy and return results keyed by class name."""
        summary = {}
        for strategy in strategies_list:
            # Rebuild simulator per strategy to reset state
            sim = AdaptiveJammingSimulator(n_qubits=self.n_qubits,
                                           attack_strategy=strategy,
                                           detector_params={'qber_threshold': self.detector.qber_threshold},
                                           backend=self.backend,
                                           loss_rate=self.loss_rate,
                                           error_rate=self.error_rate)
            res = sim.run_simulation()
            name = strategy.__class__.__name__
            summary[name] = res
        return summary

    def plot_results(self, results_map: Dict[str, Dict]) -> Any:
        """Plot QBER and detection flags across strategies."""
        names = list(results_map.keys())
        qbers = [results_map[n]['qber'] * 100 for n in names]
        detected = [1 if results_map[n]['detection'].get('attack_detected') else 0 for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(names, qbers, color=['red' if q > 11 else 'green' for q in qbers])
        axes[0].axhline(11, color='r', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('QBER (%)')
        axes[0].set_title('QBER by Strategy')
        axes[0].tick_params(axis='x', rotation=15)

        axes[1].bar(names, detected, color=['orange' if d else 'blue' for d in detected])
        axes[1].set_ylabel('Detected (1=yes)')
        axes[1].set_title('Detection Verdicts')
        axes[1].tick_params(axis='x', rotation=15)
        plt.tight_layout()
        return fig

    def save_results_csv(self, results_map: Dict[str, Dict], filepath: Optional[str] = None) -> str:
        """Save summarized results to CSV."""
        import csv
        if filepath is None:
            filepath = os.path.join(OUTPUT_DIR, 'simulator_results.csv')
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Strategy', 'QBER', 'Detected', 'Lost', 'Errors', 'InfoGained'])
            for name, res in results_map.items():
                detected = 1 if res['detection'].get('attack_detected') else 0
                lost = res['channel']['lost']
                errors = res['channel']['error_events']
                info = 0.0
                if res.get('eve') and res['eve'].get('attack_strategy_stats'):
                    info = res['eve']['attack_strategy_stats'].get('information_gained', 0.0)
                writer.writerow([name, f"{res['qber']:.6f}", detected, lost, errors, f"{info:.4f}"])
        return filepath

    def statistical_analysis(self, qber_samples_map: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Pairwise Welch's t-tests between strategy QBER samples."""
        try:
            from scipy.stats import ttest_ind  # type: ignore
            use_scipy = True
        except Exception:
            ttest_ind = None
            use_scipy = False

        names = list(qber_samples_map.keys())
        results: Dict[str, Dict[str, float]] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = np.array(qber_samples_map[names[i]], dtype=float)
                b = np.array(qber_samples_map[names[j]], dtype=float)
                if use_scipy and ttest_ind is not None:
                    stat, p = ttest_ind(a, b, equal_var=False)
                else:
                    # Fallback: Welch's t-test
                    ma, mb = float(a.mean()), float(b.mean())
                    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
                    na, nb = len(a), len(b)
                    t_num = ma - mb
                    t_den = math.sqrt(va/na + vb/nb) if na > 1 and nb > 1 else float('inf')
                    stat = t_num / t_den if t_den > 0 else 0.0
                    # No p-value without SciPy; set to NaN
                    p = float('nan')
                key = f"{names[i]}_vs_{names[j]}"
                results[key] = {'t_stat': float(stat), 'p_value': float(p)}
        return results


class Bob:
    """
    Bob: The receiver in QKD protocol
    """
    
    def __init__(self, backend: QuantumBackend, name: str = "Bob"):
        self.name = name
        self.backend = backend
        self.bases: List[Basis] = []
        self.measured_bits: List[int] = []
        self.received_states: List = []
        
    def choose_random_bases(self, n: int) -> List[Basis]:
        """Choose random measurement basis for each photon"""
        self.bases = [Basis.random() for _ in range(n)]
        return self.bases
    
    def measure_states(self, states: List) -> List[int]:
        """Measure received states in chosen bases"""
        assert len(states) == len(self.bases), "Number of states and bases must match"
        
        self.received_states = states
        self.measured_bits = []
        
        for state, basis in zip(states, self.bases):
            if state is not None:
                measured_bit = self.backend.measure_state(state, basis)
                self.measured_bits.append(measured_bit)
            else:
                self.measured_bits.append(None)
        
        return self.measured_bits
    
    def announce_bases(self) -> List[Basis]:
        """Publicly announce measurement bases"""
        return self.bases


# ============================================================================
# EVE - QUANTUM EAVESDROPPER
# ============================================================================

class AttackStrategy(ABC):
    """
    Abstract base class for Eve's attack strategies.
    Defines the interface for different eavesdropping approaches.
    """
    
    def __init__(self):
        self.statistics = {
            'interceptions': 0,
            'successful_measurements': 0,
            'information_gained': 0.0,
            'detected_interceptions': 0
        }
    
    @abstractmethod
    def intercept(self, qubit: Any, alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        """
        Intercept and potentially modify a qubit.
        
        Args:
            qubit: The quantum state to intercept
            alice_basis: Alice's basis (if known to Eve)
            
        Returns:
            Tuple of (modified_qubit, was_intercepted)
        """
        pass
    
    @abstractmethod
    def should_intercept(self, metadata: Dict) -> bool:
        """
        Decide whether to intercept based on metadata.
        
        Args:
            metadata: Dictionary containing contextual information
                     (e.g., qubit_index, channel_conditions, etc.)
        
        Returns:
            bool: True if should intercept, False otherwise
        """
        pass
    
    @abstractmethod
    def update_strategy(self, feedback: Dict) -> None:
        """
        Update attack strategy based on feedback.
        
        Args:
            feedback: Dictionary containing QBER, public_info, etc.
        """
        pass
    
    def get_statistics(self) -> Dict:
        """
        Get attack statistics.
        
        Returns:
            Dictionary of attack metrics
        """
        return self.statistics.copy()


class EveController:
    """
    Controller for Eve's eavesdropping operations.
    Manages attack strategy execution and tracks observations.
    """
    
    def __init__(self, 
                 attack_strategy: AttackStrategy,
                 backend: QuantumBackend,
                 name: str = "Eve"):
        """
        Initialize Eve's controller.
        
        Args:
            attack_strategy: The attack strategy to use
            backend: Quantum backend for measurements
            name: Eve's identifier
        """
        self.name = name
        self.attack_strategy = attack_strategy
        self.backend = backend
        self.qber_history: List[float] = []
        self.basis_observations: Dict[str, int] = {
            'rectilinear_observed': 0,
            'diagonal_observed': 0
        }
        self.intercepted_qubits: List[Dict] = []
        
    def intercept_transmission(self, 
                              qubit: Any, 
                              metadata: Optional[Dict] = None) -> Any:
        """
        Intercept a qubit transmission.
        
        Args:
            qubit: The quantum state being transmitted
            metadata: Contextual information about the transmission
        
        Returns:
            The (potentially modified) quantum state
        """
        if metadata is None:
            metadata = {}
        
        # Decide whether to intercept
        if not self.attack_strategy.should_intercept(metadata):
            return qubit
        
        # Perform interception
        modified_qubit, was_intercepted = self.attack_strategy.intercept(qubit)
        
        if was_intercepted:
            # Track the interception
            self.intercepted_qubits.append({
                'index': metadata.get('qubit_index', -1),
                'timestamp': metadata.get('timestamp', None),
                'modified': modified_qubit is not qubit
            })
        
        return modified_qubit
    
    def receive_feedback(self, qber: float, public_info: Optional[Dict] = None) -> None:
        """
        Receive feedback from the public channel.
        
        Args:
            qber: Quantum Bit Error Rate observed
            public_info: Public information exchanged (basis announcements, etc.)
        """
        self.qber_history.append(qber)
        
        if public_info is None:
            public_info = {}
        
        # Update basis observations if available
        if 'alice_bases' in public_info:
            for basis in public_info['alice_bases']:
                if basis == Basis.RECTILINEAR:
                    self.basis_observations['rectilinear_observed'] += 1
                else:
                    self.basis_observations['diagonal_observed'] += 1
        
        # Update strategy based on feedback
        feedback = {
            'qber': qber,
            'qber_history': self.qber_history.copy(),
            'public_info': public_info,
            'basis_observations': self.basis_observations.copy()
        }
        self.attack_strategy.update_strategy(feedback)
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about Eve's operations.
        
        Returns:
            Dictionary of statistics
        """
        strategy_stats = self.attack_strategy.get_statistics()
        
        return {
            'name': self.name,
            'total_intercepted': len(self.intercepted_qubits),
            'qber_history': self.qber_history.copy(),
            'basis_observations': self.basis_observations.copy(),
            'attack_strategy_stats': strategy_stats,
            'avg_qber': np.mean(self.qber_history) if self.qber_history else 0.0
        }


# ============================================================================
# EXAMPLE ATTACK STRATEGIES
# ============================================================================

class InterceptResendAttack(AttackStrategy):
    """
    Classic intercept-resend attack strategy.
    Eve intercepts qubits, measures them in a random basis, and resends.
    This introduces ~25% QBER when attacking all qubits.
    """
    
    def __init__(self, 
                 backend: QuantumBackend,
                 intercept_probability: float = 1.0):
        """
        Initialize intercept-resend attack.
        
        Args:
            backend: Quantum backend for measurements
            intercept_probability: Probability of intercepting each qubit
        """
        super().__init__()
        self.backend = backend
        self.intercept_probability = intercept_probability
        self.measured_bits: List[int] = []
        self.measurement_bases: List[Basis] = []
        
    def should_intercept(self, metadata: Dict) -> bool:
        """
        Decide whether to intercept based on probability.
        
        Args:
            metadata: Context information (unused in basic strategy)
        
        Returns:
            bool: True if should intercept
        """
        return random.random() < self.intercept_probability
    
    def intercept(self, qubit: Any, alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        """
        Intercept qubit: measure in random basis and resend.
        
        Args:
            qubit: The quantum state to intercept
            alice_basis: Alice's basis (ignored - unknown to Eve)
        
        Returns:
            Tuple of (resent_qubit, was_intercepted=True)
        """
        # Choose random measurement basis
        eve_basis = Basis.random()
        
        # Measure the qubit
        measured_bit = self.backend.measure_state(qubit, eve_basis)
        
        # Store measurement
        self.measured_bits.append(measured_bit)
        self.measurement_bases.append(eve_basis)
        
        # Prepare new qubit with measured value
        resent_qubit = self.backend.prepare_state(measured_bit, eve_basis)
        
        # Update statistics
        self.statistics['interceptions'] += 1
        self.statistics['successful_measurements'] += 1
        self.statistics['information_gained'] += 0.5  # 50% chance correct basis
        
        return (resent_qubit, True)
    
    def update_strategy(self, feedback: Dict) -> None:
        """
        Update strategy based on QBER feedback.
        
        Args:
            feedback: Dictionary with qber, public_info, etc.
        """
        qber = feedback.get('qber', 0.0)
        
        # If QBER is too high, reduce interception rate
        if qber > 0.20:
            self.intercept_probability *= 0.8
        elif qber < 0.10:
            # QBER is low, can intercept more
            self.intercept_probability = min(1.0, self.intercept_probability * 1.1)
    
    def get_statistics(self) -> Dict:
        """Get attack statistics with additional metrics."""
        stats = super().get_statistics()
        stats['intercept_probability'] = self.intercept_probability
        stats['total_measurements'] = len(self.measured_bits)
        stats['basis_distribution'] = {
            'rectilinear': sum(1 for b in self.measurement_bases if b == Basis.RECTILINEAR),
            'diagonal': sum(1 for b in self.measurement_bases if b == Basis.DIAGONAL)
        }
        return stats


# ============================================================================
# ATTACK DETECTION (Alice & Bob side)
# ============================================================================

class AttackDetector:
    """
    Statistical attack detection toolbox for BB84.
    Implements multiple independent tests and aggregates them.
    """

    def __init__(self, qber_threshold: float = 0.11):
        self.qber_threshold = qber_threshold
        self.detection_history: List[Dict] = []

    # ------------------------------ Utilities ------------------------------ #
    @staticmethod
    def _encode_bases(bases: List[Any]) -> List[int]:
        """Convert Basis enums or 0/1 to integers 0/1."""
        enc = []
        for b in bases:
            if isinstance(b, Basis):
                enc.append(0 if b == Basis.RECTILINEAR else 1)
            else:
                enc.append(int(b))
        return enc

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Standard normal CDF using erf."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # --------------------------- Individual tests -------------------------- #
    def detect_qber(self,
                    alice_bits: List[Optional[int]],
                    bob_bits: List[Optional[int]],
                    alice_bases: Optional[List[Any]] = None,
                    bob_bases: Optional[List[Any]] = None,
                    sample_size: int = 300,
                    confidence: float = 0.95) -> Dict:
        """
        Detect elevated QBER using Hoeffding's inequality for CI.
        CI radius r = sqrt(-log((1-confidence)/2)/(2*n))
        """
        n = min(len(alice_bits), len(bob_bits))
        indices = list(range(n))

        # Restrict to matching bases if provided
        if alice_bases is not None and bob_bases is not None:
            m = min(len(alice_bases), len(bob_bases), n)
            indices = [i for i in range(m)
                       if alice_bases[i] is not None and bob_bases[i] is not None
                       and ((alice_bases[i] == bob_bases[i])
                            or (isinstance(alice_bases[i], Basis) and isinstance(bob_bases[i], Basis)
                                and alice_bases[i].value == bob_bases[i].value))]

        # Filter to positions with both bits observed
        indices = [i for i in indices if alice_bits[i] is not None and bob_bits[i] is not None]

        if not indices:
            return {
                'method': 'qber_hoeffding',
                'status': 'insufficient_data',
                'qber': None,
                'ci_lower': None,
                'ci_upper': None,
                'n': 0,
                'flag': False
            }

        # Sample without replacement up to sample_size
        if len(indices) > sample_size:
            indices = random.sample(indices, sample_size)

        # Compute QBER
        errors = sum(1 for i in indices if alice_bits[i] != bob_bits[i])
        qber = errors / len(indices)

        # Hoeffding CI
        n_samp = len(indices)
        radius = math.sqrt(-math.log((1.0 - confidence) / 2.0) / (2.0 * n_samp))
        ci_lower = max(0.0, qber - radius)
        ci_upper = min(1.0, qber + radius)

        return {
            'method': 'qber_hoeffding',
            'qber': qber,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': n_samp,
            'threshold': self.qber_threshold,
            'flag': qber > self.qber_threshold
        }

    def detect_basis_randomness_chi_square(self, bases: List[Any]) -> Dict:
        """
        Chi-square goodness-of-fit test for 50/50 basis usage.
        df=1, p = 1 - CDF_chi2(chi2,1) = 1 - erf(sqrt(chi2/2))
        """
        enc = self._encode_bases(bases)
        n = len(enc)
        if n == 0:
            return {'method': 'chi_square_bases', 'status': 'insufficient_data', 'flag': False}

        count0 = enc.count(0)
        count1 = enc.count(1)
        expected = n / 2.0
        # Avoid division by zero if n=0 is already handled
        chi2 = ((count0 - expected) ** 2) / expected + ((count1 - expected) ** 2) / expected
        p_value = 1.0 - math.erf(math.sqrt(chi2 / 2.0))

        return {
            'method': 'chi_square_bases',
            'n': n,
            'observed': {'0': count0, '1': count1},
            'expected_each': expected,
            'chi2': chi2,
            'p_value': p_value,
            'flag': p_value < 0.05
        }

    def detect_basis_correlation_mi(self, alice_bases: List[Any], bob_bases: List[Any]) -> Dict:
        """
        Mutual information between Alice and Bob bases (natural log units).
        Approximates sklearn.metrics.mutual_info_score without extra deps.
        """
        a = self._encode_bases(alice_bases)
        b = self._encode_bases(bob_bases)
        n = min(len(a), len(b))
        if n == 0:
            return {'method': 'basis_mutual_information', 'status': 'insufficient_data', 'flag': False}

        a = a[:n]
        b = b[:n]

        # Joint counts
        n00 = sum(1 for i in range(n) if a[i] == 0 and b[i] == 0)
        n01 = sum(1 for i in range(n) if a[i] == 0 and b[i] == 1)
        n10 = sum(1 for i in range(n) if a[i] == 1 and b[i] == 0)
        n11 = sum(1 for i in range(n) if a[i] == 1 and b[i] == 1)

        # Convert to probabilities with small epsilon to avoid log(0)
        eps = 1e-12
        p00 = max(eps, n00 / n)
        p01 = max(eps, n01 / n)
        p10 = max(eps, n10 / n)
        p11 = max(eps, n11 / n)

        pa0 = (n00 + n01) / n
        pa1 = (n10 + n11) / n
        pb0 = (n00 + n10) / n
        pb1 = (n01 + n11) / n

        mi = 0.0
        mi += p00 * math.log(p00 / (pa0 * pb0 + eps))
        mi += p01 * math.log(p01 / (pa0 * pb1 + eps))
        mi += p10 * math.log(p10 / (pa1 * pb0 + eps))
        mi += p11 * math.log(p11 / (pa1 * pb1 + eps))

        return {
            'method': 'basis_mutual_information',
            'n': n,
            'mi': mi,
            'threshold': 0.1,
            'flag': mi > 0.1
        }

    def detect_runs_test(self, errors: List[int]) -> Dict:
        """
        Wald–Wolfowitz runs test on error sequence (0: correct, 1: error).
        Two-tailed p-value via normal approximation.
        """
        if not errors:
            return {'method': 'runs_test', 'status': 'insufficient_data', 'flag': False}

        n = len(errors)
        n1 = sum(errors)
        n0 = n - n1
        if n0 == 0 or n1 == 0 or n < 2:
            # Not enough variability for runs test
            return {
                'method': 'runs_test',
                'n': n,
                'n0': n0,
                'n1': n1,
                'status': 'degenerate',
                'flag': False
            }

        # Count runs
        runs = 1
        for i in range(1, n):
            if errors[i] != errors[i - 1]:
                runs += 1

        # Expected runs and variance
        expected_runs = 1 + (2 * n0 * n1) / (n0 + n1)
        var_runs = (2 * n0 * n1 * (2 * n0 * n1 - (n0 + n1))) / (((n0 + n1) ** 2) * (n0 + n1 - 1))
        if var_runs <= 0:
            z = 0.0
            p_value = 1.0
        else:
            z = (runs - expected_runs) / math.sqrt(var_runs)
            # Two-tailed
            p_value = 2.0 * (1.0 - self._normal_cdf(abs(z)))

        return {
            'method': 'runs_test',
            'n': n,
            'n0': n0,
            'n1': n1,
            'runs': runs,
            'expected_runs': expected_runs,
            'z': z,
            'p_value': p_value,
            'flag': p_value < 0.05
        }

    # ------------------------------ Aggregation ---------------------------- #
    def detect_attack(self,
                      alice_bits: List[Optional[int]],
                      bob_bits: List[Optional[int]],
                      alice_bases: Optional[List[Any]] = None,
                      bob_bases: Optional[List[Any]] = None,
                      confidence: float = 0.95,
                      sample_size: int = 300,
                      errors_sequence: Optional[List[int]] = None) -> Dict:
        """
        Run all available detectors and produce an overall verdict.
        """
        # QBER detector
        qber_res = self.detect_qber(alice_bits, bob_bits, alice_bases, bob_bases,
                                    sample_size=sample_size, confidence=confidence)

        # Basis randomness chi-square (for Alice and Bob separately)
        chi_res_alice = self.detect_basis_randomness_chi_square(alice_bases or []) if alice_bases is not None else {
            'method': 'chi_square_bases', 'status': 'no_input', 'flag': False}
        chi_res_bob = self.detect_basis_randomness_chi_square(bob_bases or []) if bob_bases is not None else {
            'method': 'chi_square_bases', 'status': 'no_input', 'flag': False}

        # Basis correlation MI
        mi_res = self.detect_basis_correlation_mi(alice_bases or [], bob_bases or []) if (alice_bases and bob_bases) else {
            'method': 'basis_mutual_information', 'status': 'no_input', 'flag': False}

        # Runs test on errors
        if errors_sequence is None and alice_bases is not None and bob_bases is not None:
            # Derive errors from matching bases
            n = min(len(alice_bits), len(bob_bits), len(alice_bases), len(bob_bases))
            match_indices = [i for i in range(n)
                             if alice_bits[i] is not None and bob_bits[i] is not None
                             and ((alice_bases[i] == bob_bases[i])
                                  or (isinstance(alice_bases[i], Basis) and isinstance(bob_bases[i], Basis)
                                      and alice_bases[i].value == bob_bases[i].value))]
            errors_sequence = [1 if alice_bits[i] != bob_bits[i] else 0 for i in match_indices]

        runs_res = self.detect_runs_test(errors_sequence or [])

        result = {
            'qber': qber_res,
            'chi_square': {
                'alice': chi_res_alice,
                'bob': chi_res_bob
            },
            'basis_correlation': mi_res,
            'runs_test': runs_res
        }

        result['attack_detected'] = any([
            qber_res.get('flag', False),
            chi_res_alice.get('flag', False),
            chi_res_bob.get('flag', False),
            mi_res.get('flag', False),
            runs_res.get('flag', False)
        ])

        self.detection_history.append({
            'timestamp': datetime.now().isoformat(),
            'result': json.loads(json.dumps(result, default=str))  # ensure serializable
        })

        return result


# ============================================================================
# INFORMATION THEORY HELPERS (von Neumann entropy, Holevo bound, MI)
# ============================================================================

def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """
    Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ).

    Args:
        rho: Density matrix (Hermitian, PSD, trace≈1)
        base: Logarithm base (2 for bits, e for nats)

    Returns:
        Entropy S(ρ) in the requested units.
    """
    # Ensure array
    rho = np.asarray(rho, dtype=np.complex128)
    # Symmetrize to reduce numerical asymmetry
    rho = 0.5 * (rho + rho.conjugate().T)

    # Eigenvalues of Hermitian matrix
    if _SCIPY_AVAILABLE:
        vals = scipy_eigvalsh(rho)  # type: ignore
    else:
        vals = np.linalg.eigvalsh(rho)

    # Numerical cleanup
    vals = np.real(vals)
    vals[vals < 0] = 0.0
    if vals.sum() > 0:
        vals = vals / vals.sum()  # normalize trace to 1 if slightly off

    eps = 1e-12
    vals = vals[vals > eps]
    if vals.size == 0:
        return 0.0

    log_base = math.log(base) if base not in (math.e, 0) else 1.0
    return float(-np.sum(vals * (np.log(vals) / log_base)))


def holevo_bound(ensemble: Iterable[np.ndarray], probabilities: Iterable[float], base: float = 2.0) -> float:
    """
    Compute Holevo information χ for an ensemble {p_i, ρ_i}:
      χ = S(ρ_avg) - Σ p_i S(ρ_i),  where ρ_avg = Σ p_i ρ_i

    Args:
        ensemble: Iterable of density matrices ρ_i
        probabilities: Iterable of probabilities p_i (sum≈1)
        base: entropy base (2 for bits)

    Returns:
        Holevo bound χ
    """
    rhos = [np.asarray(r, dtype=np.complex128) for r in ensemble]
    ps = np.asarray(list(probabilities), dtype=float)
    ps = ps / ps.sum()

    # Average state
    rho_avg = np.zeros_like(rhos[0])
    for p, r in zip(ps, rhos):
        rho_avg = rho_avg + p * r

    s_avg = von_neumann_entropy(rho_avg, base=base)
    s_components = sum(float(p * von_neumann_entropy(r, base=base)) for p, r in zip(ps, rhos))
    return s_avg - s_components


def bb84_holevo_bound(base: float = 2.0) -> float:
    """
    Holevo bound for the BB84 ensemble with equal priors.
    States: |0>, |1>, |+>, |-> with p=1/4 each.
    Expected χ = 1 bit.
    """
    # Computational basis states
    ket0 = np.array([[1.0], [0.0]], dtype=complex)
    ket1 = np.array([[0.0], [1.0]], dtype=complex)
    # Diagonal basis
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    ket_plus = inv_sqrt2 * (ket0 + ket1)
    ket_minus = inv_sqrt2 * (ket0 - ket1)

    def dm(ket: np.ndarray) -> np.ndarray:
        return ket @ ket.conjugate().T

    ensemble = [dm(ket0), dm(ket1), dm(ket_plus), dm(ket_minus)]
    probs = [0.25, 0.25, 0.25, 0.25]
    return holevo_bound(ensemble, probs, base=base)


def binary_entropy(p: float, base: float = 2.0) -> float:
    """Binary entropy H(p) with chosen log base."""
    p = min(max(p, 0.0), 1.0)
    if p in (0.0, 1.0):
        return 0.0
    q = 1.0 - p
    log_base = math.log(base) if base not in (math.e, 0) else 1.0
    return -(p * (math.log(p) / log_base) + q * (math.log(q) / log_base))


def calculate_eve_information(attack_type: str, parameters: Dict[str, float], base: float = 2.0) -> float:
    """
    Heuristic Eve information per pulse depending on attack type.

    attack_type:
      - 'intercept_resend': 0.5 * intercept_probability
      - 'pns': P(n≥2) where P(n≥2) = 1 - (1+μ) e^{−μ}
      - 'adaptive': intercept_prob * (1 - H(success_prob)) with binary entropy H
    """
    t = attack_type.lower()
    if t == 'intercept_resend':
        p_int = float(parameters.get('intercept_probability', 1.0))
        return 0.5 * p_int
    if t == 'pns':
        mu = float(parameters.get('mean_photon_number', 0.1))
        return 1.0 - (1.0 + mu) * math.exp(-mu)
    if t == 'adaptive':
        p_int = float(parameters.get('intercept_probability', 1.0))
        p_succ = float(parameters.get('success_probability', 0.5))
        return p_int * (1.0 - binary_entropy(p_succ, base=base))
    raise ValueError(f"Unknown attack_type: {attack_type}")


def mutual_information_eve_alice(basis_match_prob: float = 0.5,
                                 eve_success_prob: float = 1.0,
                                 alice_flip_prob: float = 0.0,
                                 prior_one_prob: float = 0.5,
                                 base: float = 2.0) -> float:
    """
    Approximate mutual information I(E:A) = H(A) - H(A|E).

    Inputs:
      - basis_match_prob: P(Eve uses/knows the correct basis)
      - eve_success_prob: P(Eve's estimate is correct when bases match)
      - alice_flip_prob: Effective bit flip noise on Alice relative to Eve
      - prior_one_prob: P(A=1) prior; default 0.5

    Model:
      Accuracy before noise = p_match * p_succ + (1 - p_match) * 0.5
      Noise flips the bit with prob a_flip, so
      accuracy_eff = acc*(1-a_flip) + (1-acc)*a_flip
      error_rate = 1 - accuracy_eff
      I ≈ H(A) - H_b(error_rate)
    """
    pA1 = min(max(prior_one_prob, 0.0), 1.0)
    hA = binary_entropy(pA1, base=base)

    acc = basis_match_prob * eve_success_prob + (1.0 - basis_match_prob) * 0.5
    accuracy_eff = acc * (1.0 - alice_flip_prob) + (1.0 - acc) * alice_flip_prob
    error_rate = 1.0 - accuracy_eff
    cond = binary_entropy(error_rate, base=base)
    I = max(0.0, hA - cond)
    return min(I, hA)

class AdaptiveAttack(AttackStrategy):
    """
    Adaptive attack strategy that adjusts interception rate based on QBER.
    Tries to stay below detection threshold while maximizing information gain.
    """
    
    def __init__(self, 
                 backend: QuantumBackend,
                 target_qber: float = 0.08,
                 initial_intercept_rate: float = 0.5):
        """
        Initialize adaptive attack.
        
        Args:
            backend: Quantum backend for measurements
            target_qber: Target QBER to maintain
            initial_intercept_rate: Starting interception probability
        """
        super().__init__()
        self.backend = backend
        self.target_qber = target_qber
        self.intercept_rate = initial_intercept_rate
        self.qber_samples: List[float] = []
        
    def should_intercept(self, metadata: Dict) -> bool:
        """Intercept based on current adaptive rate."""
        return random.random() < self.intercept_rate
    
    def intercept(self, qubit: Any, alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        """Intercept and resend qubit."""
        eve_basis = Basis.random()
        measured_bit = self.backend.measure_state(qubit, eve_basis)
        resent_qubit = self.backend.prepare_state(measured_bit, eve_basis)
        
        self.statistics['interceptions'] += 1
        self.statistics['successful_measurements'] += 1
        self.statistics['information_gained'] += 0.5
        
        return (resent_qubit, True)
    
    def update_strategy(self, feedback: Dict) -> None:
        """
        Adapt interception rate to maintain target QBER.
        
        Args:
            feedback: Contains qber and history
        """
        qber = feedback.get('qber', 0.0)
        self.qber_samples.append(qber)
        
        # Calculate recent average QBER
        recent_window = min(5, len(self.qber_samples))
        recent_qber = np.mean(self.qber_samples[-recent_window:]) if self.qber_samples else 0.0
        
        # Adjust interception rate to approach target QBER
        if recent_qber > self.target_qber + 0.02:
            # QBER too high, reduce interception
            self.intercept_rate *= 0.85
        elif recent_qber < self.target_qber - 0.02:
            # QBER too low, can intercept more
            self.intercept_rate = min(1.0, self.intercept_rate * 1.15)
        
        # Keep within bounds
        self.intercept_rate = max(0.01, min(1.0, self.intercept_rate))
    
    def get_statistics(self) -> Dict:
        """Get attack statistics."""
        stats = super().get_statistics()
        stats['intercept_rate'] = self.intercept_rate
        stats['target_qber'] = self.target_qber
        stats['qber_samples'] = self.qber_samples.copy()
        return stats


class QBERAdaptiveStrategy(AttackStrategy):
    """
    QBER-Adaptive attack strategy using PID control.
    Maintains QBER just below detection threshold (11%) while maximizing
    information gain. Uses PID controller to dynamically adjust interception
    probability based on observed QBER.
    
    Mathematical Model:
    - QBER ≈ 0.25 × intercept_probability (for random basis attack)
    - PID control: p(t+1) = p(t) + Kp*e + Ki*∫e + Kd*de/dt
    """
    
    def __init__(self,
                 backend: QuantumBackend,
                 target_qber: float = 0.10,
                 threshold: float = 0.11,
                 kp: float = 2.0,
                 ki: float = 0.5,
                 kd: float = 0.1):
        """
        Initialize QBER-adaptive strategy with PID controller.
        
        Args:
            backend: Quantum backend for measurements
            target_qber: Target QBER to maintain (default: 0.10, just below threshold)
            threshold: Detection threshold (default: 0.11)
            kp: Proportional gain for PID controller
            ki: Integral gain for PID controller
            kd: Derivative gain for PID controller
        """
        super().__init__()
        self.backend = backend
        self.target_qber = target_qber
        self.threshold = threshold
        
        # PID controller parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # PID controller state
        self.intercept_probability = 0.3  # Initial value
        self.qber_history: List[float] = []
        self.error_integral = 0.0
        self.previous_error = 0.0
        
        # Measurement tracking
        self.measured_bits: List[int] = []
        self.measurement_bases: List[Basis] = []
        
    def should_intercept(self, metadata: Dict) -> bool:
        """
        Decide whether to intercept based on current probability.
        
        Args:
            metadata: Context information (unused in this strategy)
        
        Returns:
            bool: True if should intercept
        """
        return random.random() < self.intercept_probability
    
    def measure_qubit(self, qubit: Any, basis: Basis) -> int:
        """
        Measure qubit in specified basis (backend-agnostic).
        
        Args:
            qubit: Quantum state to measure
            basis: Measurement basis
        
        Returns:
            Measured bit value (0 or 1)
        """
        return self.backend.measure_state(qubit, basis)
    
    def prepare_qubit(self, bit: int, basis: Basis) -> Any:
        """
        Prepare qubit in specified state (backend-agnostic).
        
        Args:
            bit: Bit value (0 or 1)
            basis: Preparation basis
        
        Returns:
            Prepared quantum state
        """
        return self.backend.prepare_state(bit, basis)
    
    def intercept(self, qubit: Any, alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        """
        Intercept qubit: measure in random basis and resend.
        
        Args:
            qubit: The quantum state to intercept
            alice_basis: Alice's basis (unknown to Eve)
        
        Returns:
            Tuple of (resent_qubit, was_intercepted=True)
        """
        # Eve randomly chooses measurement basis
        eve_basis = Basis.random()
        
        # Measure qubit in chosen basis
        measured_bit = self.measure_qubit(qubit, eve_basis)
        
        # Store measurement
        self.measured_bits.append(measured_bit)
        self.measurement_bases.append(eve_basis)
        
        # Prepare new qubit in measured state
        resent_qubit = self.prepare_qubit(measured_bit, eve_basis)
        
        # Update statistics
        self.statistics['interceptions'] += 1
        self.statistics['successful_measurements'] += 1
        self.statistics['information_gained'] += 0.5  # 50% chance correct basis
        
        return (resent_qubit, True)
    
    def update_strategy(self, feedback: Dict) -> None:
        """
        Update interception probability using PID control.
        
        PID Control Law:
        p(t+1) = p(t) + Kp*e(t) + Ki*∫e(t) + Kd*de/dt
        
        where:
        - e(t) = target_qber - current_qber (error)
        - ∫e(t) = cumulative error (integral term)
        - de/dt = e(t) - e(t-1) (derivative term)
        
        Args:
            feedback: Dictionary containing qber and other information
        """
        current_qber = feedback.get('qber', 0.0)
        self.qber_history.append(current_qber)
        
        # Calculate error: positive if QBER too low, negative if too high
        error = self.target_qber - current_qber
        
        # Integral term (cumulative error)
        self.error_integral += error
        
        # Derivative term (rate of change)
        error_derivative = error - self.previous_error
        
        # PID correction
        correction = (
            self.kp * error +
            self.ki * self.error_integral +
            self.kd * error_derivative
        )
        
        # Update intercept probability
        self.intercept_probability += correction
        
        # Clip to valid range [0.0, 0.9]
        self.intercept_probability = max(0.0, min(0.9, self.intercept_probability))
        
        # Safety mechanism: if QBER > 0.09, reduce aggressively
        if current_qber > 0.09:
            self.intercept_probability *= 0.9
        
        # Store current error for next iteration
        self.previous_error = error
        
        # Anti-windup: reset integral if saturated
        if self.intercept_probability <= 0.0 or self.intercept_probability >= 0.9:
            self.error_integral = 0.0
    
    def get_statistics(self) -> Dict:
        """Get attack statistics with PID controller state."""
        stats = super().get_statistics()
        stats.update({
            'intercept_probability': self.intercept_probability,
            'target_qber': self.target_qber,
            'threshold': self.threshold,
            'qber_history': self.qber_history.copy(),
            'error_integral': self.error_integral,
            'previous_error': self.previous_error,
            'pid_gains': {'kp': self.kp, 'ki': self.ki, 'kd': self.kd},
            'total_measurements': len(self.measured_bits),
            'basis_distribution': {
                'rectilinear': sum(1 for b in self.measurement_bases if b == Basis.RECTILINEAR),
                'diagonal': sum(1 for b in self.measurement_bases if b == Basis.DIAGONAL)
            }
        })
        return stats


class GradientDescentQBERAdaptive(AttackStrategy):
    """
    QBER-Adaptive strategy using gradient descent optimization.
    
    Mathematical Model:
    - Loss function: L = (QBER - target_qber)²
    - Empirical gradient: ∂QBER/∂p ≈ 0.25 (for intercept-resend)
    - Update rule: p(t+1) = p(t) - learning_rate × ∂L/∂p
    - ∂L/∂p = 2(QBER - target) × ∂QBER/∂p ≈ 2(QBER - target) × 0.25
    """
    
    def __init__(self,
                 backend: QuantumBackend,
                 target_qber: float = 0.10,
                 threshold: float = 0.11,
                 learning_rate: float = 0.01):
        """
        Initialize gradient descent QBER-adaptive strategy.
        
        Args:
            backend: Quantum backend for measurements
            target_qber: Target QBER to maintain
            threshold: Detection threshold
            learning_rate: Learning rate for gradient descent
        """
        super().__init__()
        self.backend = backend
        self.target_qber = target_qber
        self.threshold = threshold
        self.learning_rate = learning_rate
        
        # State
        self.intercept_probability = 0.3  # Initial value
        self.qber_history: List[float] = []
        self.loss_history: List[float] = []
        
        # Empirical gradient constant for intercept-resend attack
        # ∂QBER/∂p ≈ 0.25 (theoretical value for random basis)
        self.qber_gradient = 0.25
        
        # Measurement tracking
        self.measured_bits: List[int] = []
        self.measurement_bases: List[Basis] = []
    
    def should_intercept(self, metadata: Dict) -> bool:
        """Decide whether to intercept based on current probability."""
        return random.random() < self.intercept_probability
    
    def measure_qubit(self, qubit: Any, basis: Basis) -> int:
        """Measure qubit in specified basis."""
        return self.backend.measure_state(qubit, basis)
    
    def prepare_qubit(self, bit: int, basis: Basis) -> Any:
        """Prepare qubit in specified state."""
        return self.backend.prepare_state(bit, basis)
    
    def intercept(self, qubit: Any, alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        """Intercept qubit: measure in random basis and resend."""
        # Eve randomly chooses measurement basis
        eve_basis = Basis.random()
        
        # Measure qubit
        measured_bit = self.measure_qubit(qubit, eve_basis)
        
        # Store measurement
        self.measured_bits.append(measured_bit)
        self.measurement_bases.append(eve_basis)
        
        # Prepare and resend
        resent_qubit = self.prepare_qubit(measured_bit, eve_basis)
        
        # Update statistics
        self.statistics['interceptions'] += 1
        self.statistics['successful_measurements'] += 1
        self.statistics['information_gained'] += 0.5
        
        return (resent_qubit, True)
    
    def update_strategy(self, feedback: Dict) -> None:
        """
        Update interception probability using gradient descent.
        
        Gradient Descent Update:
        L = (QBER - target)²
        ∂L/∂p = 2(QBER - target) × ∂QBER/∂p
        p(t+1) = p(t) - learning_rate × ∂L/∂p
        
        Args:
            feedback: Dictionary containing qber
        """
        current_qber = feedback.get('qber', 0.0)
        self.qber_history.append(current_qber)
        
        # Calculate loss
        loss = (current_qber - self.target_qber) ** 2
        self.loss_history.append(loss)
        
        # Calculate gradient of loss w.r.t. intercept probability
        # ∂L/∂p = 2(QBER - target) × ∂QBER/∂p
        error = current_qber - self.target_qber
        gradient = 2.0 * error * self.qber_gradient
        
        # Gradient descent update
        self.intercept_probability -= self.learning_rate * gradient
        
        # Clip to valid range [0.0, 0.9]
        self.intercept_probability = max(0.0, min(0.9, self.intercept_probability))
        
        # Safety mechanism: if QBER > 0.09, reduce aggressively
        if current_qber > 0.09:
            self.intercept_probability *= 0.9
    
    def get_statistics(self) -> Dict:
        """Get attack statistics with gradient descent state."""
        stats = super().get_statistics()
        stats.update({
            'intercept_probability': self.intercept_probability,
            'target_qber': self.target_qber,
            'threshold': self.threshold,
            'learning_rate': self.learning_rate,
            'qber_history': self.qber_history.copy(),
            'loss_history': self.loss_history.copy(),
            'qber_gradient': self.qber_gradient,
            'total_measurements': len(self.measured_bits),
            'basis_distribution': {
                'rectilinear': sum(1 for b in self.measurement_bases if b == Basis.RECTILINEAR),
                'diagonal': sum(1 for b in self.measurement_bases if b == Basis.DIAGONAL)
            }
        })
        return stats


class BasisLearningStrategy(AttackStrategy):
    """
    Bayesian inference attack that learns patterns in Alice's basis selection.
    Uses Beta distribution (conjugate prior for Bernoulli) to model basis choices.
    
    THEORY:
    - Prior: Beta(alpha=1, beta=1) = Uniform(0,1) 
    - Update: Observe rectilinear → alpha += 1, diagonal → beta += 1
    - Posterior: P(rectilinear) = alpha / (alpha + beta)
    - Variance: (alpha*beta) / ((alpha+beta)²*(alpha+beta+1))
    
    ADAPTIVE INTERCEPTION:
    - Base probability: 0.3
    - Confidence bonus: 0.4 * confidence
    - More confident → intercept more often in predicted basis
    """
    
    def __init__(self,
                 backend: QuantumBackend,
                 base_intercept_prob: float = 0.3,
                 confidence_threshold: float = 0.8,
                 alpha_prior: float = 1.0,
                 beta_prior: float = 1.0):
        """
        Initialize Bayesian basis learning strategy.
        
        Args:
            backend: Quantum backend for measurements
            base_intercept_prob: Baseline interception probability
            confidence_threshold: Confidence threshold for adaptation
            alpha_prior: Prior parameter for rectilinear basis
            beta_prior: Prior parameter for diagonal basis
        """
        super().__init__()
        self.backend = backend
        self.base_intercept_prob = base_intercept_prob
        self.confidence_threshold = confidence_threshold
        
        # Beta distribution parameters (conjugate prior for Bernoulli)
        self.alpha = alpha_prior  # Rectilinear observations
        self.beta = beta_prior    # Diagonal observations
        
        # Tracking
        self.observations: List[Basis] = []
        self.predicted_bases: List[Basis] = []
        self.correct_predictions: int = 0
        self.total_predictions: int = 0
        
        # Measurements
        self.measured_bits: List[int] = []
        self.measurement_bases: List[Basis] = []
        
    def observe_basis(self, basis: Basis) -> None:
        """
        Update Beta distribution parameters based on observed basis.
        
        Args:
            basis: Observed basis from public announcement
        """
        self.observations.append(basis)
        
        if basis == Basis.RECTILINEAR:
            self.alpha += 1.0
        else:  # DIAGONAL
            self.beta += 1.0
    
    def predict_basis(self) -> Basis:
        """
        Predict next basis using posterior distribution.
        
        Returns:
            Predicted basis (higher probability)
        """
        prob_rectilinear = self.alpha / (self.alpha + self.beta)
        
        # Predict basis with higher posterior probability
        predicted = Basis.RECTILINEAR if prob_rectilinear > 0.5 else Basis.DIAGONAL
        self.predicted_bases.append(predicted)
        
        return predicted
    
    def get_confidence(self) -> float:
        """
        Calculate confidence in prediction based on posterior variance.
        
        Lower variance → higher confidence
        Confidence = 1 - 2*sqrt(variance)
        
        Returns:
            Confidence score in [0, 1]
        """
        # Posterior variance
        total = self.alpha + self.beta
        variance = (self.alpha * self.beta) / ((total ** 2) * (total + 1))
        
        # Convert to confidence (lower variance = higher confidence)
        confidence = 1.0 - 2.0 * np.sqrt(variance)
        
        # Clip to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def get_basis_probability(self, basis: Basis) -> float:
        """
        Get posterior probability for a specific basis.
        
        Args:
            basis: Basis to query
            
        Returns:
            Posterior probability
        """
        prob_rectilinear = self.alpha / (self.alpha + self.beta)
        
        if basis == Basis.RECTILINEAR:
            return prob_rectilinear
        else:
            return 1.0 - prob_rectilinear
    
    def mutual_information(self, p: float) -> float:
        """
        Calculate mutual information (binary entropy).
        
        I(X;Y) = -p*log2(p) - (1-p)*log2(1-p)
        
        Args:
            p: Probability
            
        Returns:
            Mutual information in bits
        """
        if p <= 0.0 or p >= 1.0:
            return 0.0
        
        return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    
    def should_intercept(self, metadata: Dict) -> bool:
        """
        Adaptive interception based on prediction confidence.
        
        Higher confidence → higher interception probability
        
        Args:
            metadata: Contextual information (unused here)
            
        Returns:
            True if should intercept
        """
        confidence = self.get_confidence()
        
        # Adaptive probability: base + confidence bonus
        adaptive_prob = self.base_intercept_prob + 0.4 * confidence
        
        return random.random() < adaptive_prob
    
    def measure_qubit(self, qubit: Any, basis: Basis) -> int:
        """Measure qubit in specified basis."""
        return self.backend.measure_state(qubit, basis)
    
    def prepare_qubit(self, bit: int, basis: Basis) -> Any:
        """Prepare qubit in specified state."""
        return self.backend.prepare_state(bit, basis)
    
    def intercept(self, qubit: Any, alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        """
        Intercept qubit using predicted basis from Bayesian learning.
        
        Args:
            qubit: Quantum state to intercept
            alice_basis: Alice's actual basis (for evaluation only)
            
        Returns:
            Tuple of (modified_qubit, was_intercepted)
        """
        # Predict basis using Bayesian inference
        eve_basis = self.predict_basis()
        
        # Measure in predicted basis
        measured_bit = self.measure_qubit(qubit, eve_basis)
        
        # Store measurement
        self.measured_bits.append(measured_bit)
        self.measurement_bases.append(eve_basis)
        
        # Check if prediction was correct (for statistics)
        if alice_basis is not None:
            self.total_predictions += 1
            if eve_basis == alice_basis:
                self.correct_predictions += 1
                # Perfect measurement - full information
                self.statistics['information_gained'] += 1.0
            else:
                # Wrong basis - gain some information but introduce error
                prob_correct = 0.5  # Random outcome
                self.statistics['information_gained'] += self.mutual_information(prob_correct)
        
        # Prepare and resend
        resent_qubit = self.prepare_qubit(measured_bit, eve_basis)
        
        # Update statistics
        self.statistics['interceptions'] += 1
        self.statistics['successful_measurements'] += 1
        
        return (resent_qubit, True)
    
    def update_strategy(self, feedback: Dict) -> None:
        """
        Update Bayesian model based on public basis announcement.
        
        Args:
            feedback: Dictionary containing public_bases (List[Basis])
        """
        # Learn from publicly announced bases
        public_bases = None
        if 'public_bases' in feedback:
            public_bases = feedback['public_bases']
        elif isinstance(feedback, dict):
            public_info = feedback.get('public_info', {})
            if isinstance(public_info, dict):
                public_bases = public_info.get('alice_bases')
        if public_bases:
            for basis in public_bases:
                self.observe_basis(basis)
        
        # Track QBER if available
        if 'qber' in feedback:
            qber = feedback['qber']
            # Could use QBER to adjust strategy, but Bayesian learning
            # primarily relies on basis observations
    
    def get_statistics(self) -> Dict:
        """Get attack statistics with Bayesian learning metrics."""
        stats = super().get_statistics()
        
        # Posterior probabilities
        prob_rectilinear = self.alpha / (self.alpha + self.beta)
        
        stats.update({
            'base_intercept_prob': self.base_intercept_prob,
            'confidence_threshold': self.confidence_threshold,
            'alpha': self.alpha,
            'beta': self.beta,
            'prob_rectilinear': prob_rectilinear,
            'prob_diagonal': 1.0 - prob_rectilinear,
            'confidence': self.get_confidence(),
            'total_observations': len(self.observations),
            'correct_predictions': self.correct_predictions,
            'total_predictions': self.total_predictions,
            'prediction_accuracy': self.correct_predictions / max(1, self.total_predictions),
            'total_measurements': len(self.measured_bits),
            'basis_distribution': {
                'rectilinear': sum(1 for b in self.measurement_bases if b == Basis.RECTILINEAR),
                'diagonal': sum(1 for b in self.measurement_bases if b == Basis.DIAGONAL)
            }
        })
        return stats


class ParticleFilterBasisLearner(AttackStrategy):
    """
    Advanced Bayesian inference using Sequential Monte Carlo (Particle Filter).
    
    THEORY:
    - Represent posterior as weighted particles
    - Each particle is a hypothesis about P(rectilinear)
    - Sequential Importance Sampling (SIS) for updates
    - Resampling when effective sample size drops
    
    ADVANTAGES:
    - Can model non-conjugate priors
    - Handles multimodal distributions
    - More flexible than parametric Beta distribution
    
    ALGORITHM:
    1. Initialize: N particles uniformly in [0, 1]
    2. Observe: Update weights based on likelihood
    3. Resample: When ESS < N/2, resample particles
    4. Predict: Weighted average of particles
    """
    
    def __init__(self,
                 backend: QuantumBackend,
                 n_particles: int = 1000,
                 base_intercept_prob: float = 0.3,
                 confidence_threshold: float = 0.8,
                 resample_threshold: float = 0.5):
        """
        Initialize particle filter basis learner.
        
        Args:
            backend: Quantum backend for measurements
            n_particles: Number of particles
            base_intercept_prob: Baseline interception probability
            confidence_threshold: Confidence threshold for adaptation
            resample_threshold: Resample when ESS < n_particles * threshold
        """
        super().__init__()
        self.backend = backend
        self.n_particles = n_particles
        self.base_intercept_prob = base_intercept_prob
        self.confidence_threshold = confidence_threshold
        self.resample_threshold = resample_threshold
        
        # Particles: each represents P(rectilinear)
        self.particles = np.random.uniform(0.0, 1.0, n_particles)
        self.weights = np.ones(n_particles) / n_particles  # Uniform weights
        
        # Tracking
        self.observations: List[Basis] = []
        self.predicted_bases: List[Basis] = []
        self.correct_predictions: int = 0
        self.total_predictions: int = 0
        self.resampling_count: int = 0
        
        # Measurements
        self.measured_bits: List[int] = []
        self.measurement_bases: List[Basis] = []
        
    def observe_basis(self, basis: Basis) -> None:
        """
        Update particle weights based on observed basis.
        
        Uses likelihood: P(observation|particle)
        - If rectilinear observed: weight ∝ particle value
        - If diagonal observed: weight ∝ (1 - particle value)
        
        Args:
            basis: Observed basis
        """
        self.observations.append(basis)
        
        # Update weights based on likelihood
        if basis == Basis.RECTILINEAR:
            # Likelihood: particle value (probability of rectilinear)
            likelihoods = self.particles
        else:  # DIAGONAL
            # Likelihood: 1 - particle value (probability of diagonal)
            likelihoods = 1.0 - self.particles
        
        # Update weights: w_new = w_old * likelihood
        self.weights *= likelihoods
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # All weights became zero (unlikely), reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Check if resampling needed
        if self.effective_sample_size() < self.n_particles * self.resample_threshold:
            self.resample()
    
    def effective_sample_size(self) -> float:
        """
        Calculate effective sample size (ESS).
        
        ESS = 1 / sum(w_i²)
        
        Low ESS indicates weight degeneracy → need resampling
        
        Returns:
            Effective sample size
        """
        return 1.0 / np.sum(self.weights ** 2)
    
    def resample(self) -> None:
        """
        Resample particles based on weights (systematic resampling).
        
        Replaces low-weight particles with copies of high-weight particles.
        Resets weights to uniform after resampling.
        """
        # Systematic resampling
        positions = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
        cumulative_sum = np.cumsum(self.weights)
        
        i, j = 0, 0
        new_particles = np.zeros(self.n_particles)
        
        while i < self.n_particles:
            if positions[i] < cumulative_sum[j]:
                new_particles[i] = self.particles[j]
                i += 1
            else:
                j += 1
        
        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.resampling_count += 1
    
    def predict_basis(self) -> Basis:
        """
        Predict next basis using weighted average of particles.
        
        Returns:
            Predicted basis
        """
        # Weighted average: E[P(rectilinear)]
        prob_rectilinear = np.sum(self.weights * self.particles)
        
        # Predict basis with higher probability
        predicted = Basis.RECTILINEAR if prob_rectilinear > 0.5 else Basis.DIAGONAL
        self.predicted_bases.append(predicted)
        
        return predicted
    
    def get_confidence(self) -> float:
        """
        Calculate confidence based on weighted variance.
        
        Lower variance → higher confidence
        
        Returns:
            Confidence score in [0, 1]
        """
        # Weighted mean
        mean = np.sum(self.weights * self.particles)
        
        # Weighted variance
        variance = np.sum(self.weights * (self.particles - mean) ** 2)
        
        # Convert to confidence
        confidence = 1.0 - 2.0 * np.sqrt(variance)
        
        return max(0.0, min(1.0, confidence))
    
    def get_basis_probability(self, basis: Basis) -> float:
        """
        Get posterior probability for a specific basis.
        
        Args:
            basis: Basis to query
            
        Returns:
            Posterior probability (weighted average)
        """
        prob_rectilinear = np.sum(self.weights * self.particles)
        
        if basis == Basis.RECTILINEAR:
            return prob_rectilinear
        else:
            return 1.0 - prob_rectilinear
    
    def mutual_information(self, p: float) -> float:
        """Calculate mutual information (binary entropy)."""
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    
    def should_intercept(self, metadata: Dict) -> bool:
        """
        Adaptive interception based on prediction confidence.
        
        Args:
            metadata: Contextual information
            
        Returns:
            True if should intercept
        """
        confidence = self.get_confidence()
        adaptive_prob = self.base_intercept_prob + 0.4 * confidence
        return random.random() < adaptive_prob
    
    def measure_qubit(self, qubit: Any, basis: Basis) -> int:
        """Measure qubit in specified basis."""
        return self.backend.measure_state(qubit, basis)
    
    def prepare_qubit(self, bit: int, basis: Basis) -> Any:
        """Prepare qubit in specified state."""
        return self.backend.prepare_state(bit, basis)
    
    def intercept(self, qubit: Any, alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        """
        Intercept qubit using particle filter prediction.
        
        Args:
            qubit: Quantum state to intercept
            alice_basis: Alice's actual basis (for evaluation)
            
        Returns:
            Tuple of (modified_qubit, was_intercepted)
        """
        # Predict basis using particle filter
        eve_basis = self.predict_basis()
        
        # Measure in predicted basis
        measured_bit = self.measure_qubit(qubit, eve_basis)
        
        # Store measurement
        self.measured_bits.append(measured_bit)
        self.measurement_bases.append(eve_basis)
        
        # Check prediction accuracy
        if alice_basis is not None:
            self.total_predictions += 1
            if eve_basis == alice_basis:
                self.correct_predictions += 1
                self.statistics['information_gained'] += 1.0
            else:
                prob_correct = 0.5
                self.statistics['information_gained'] += self.mutual_information(prob_correct)
        
        # Prepare and resend
        resent_qubit = self.prepare_qubit(measured_bit, eve_basis)
        
        # Update statistics
        self.statistics['interceptions'] += 1
        self.statistics['successful_measurements'] += 1
        
        return (resent_qubit, True)
    
    def update_strategy(self, feedback: Dict) -> None:
        """
        Update particle filter based on public basis announcement.
        
        Args:
            feedback: Dictionary containing public_bases
        """
        # Learn from publicly announced bases
        if 'public_bases' in feedback:
            public_bases = feedback['public_bases']
            for basis in public_bases:
                self.observe_basis(basis)
    
    def get_statistics(self) -> Dict:
        """Get attack statistics with particle filter metrics."""
        stats = super().get_statistics()
        
        # Posterior statistics
        prob_rectilinear = np.sum(self.weights * self.particles)
        
        stats.update({
            'n_particles': self.n_particles,
            'base_intercept_prob': self.base_intercept_prob,
            'confidence_threshold': self.confidence_threshold,
            'prob_rectilinear': prob_rectilinear,
            'prob_diagonal': 1.0 - prob_rectilinear,
            'confidence': self.get_confidence(),
            'effective_sample_size': self.effective_sample_size(),
            'resampling_count': self.resampling_count,
            'total_observations': len(self.observations),
            'correct_predictions': self.correct_predictions,
            'total_predictions': self.total_predictions,
            'prediction_accuracy': self.correct_predictions / max(1, self.total_predictions),
            'total_measurements': len(self.measured_bits),
            'basis_distribution': {
                'rectilinear': sum(1 for b in self.measurement_bases if b == Basis.RECTILINEAR),
                'diagonal': sum(1 for b in self.measurement_bases if b == Basis.DIAGONAL)
            }
        })
        return stats


class AtmosphericChannelModel:
    """
    Atmospheric turbulence model for free-space quantum channels.
    
    Implements:
    - Hufnagel-Valley Cn² model
    - Rytov variance calculation
    - Phase screen generation (Kolmogorov spectrum)
    """
    
    def __init__(self,
                 distance_km: float = 10.0,
                 wavelength_nm: float = 1550.0,
                 time_of_day: str = 'day'):
        """
        Initialize atmospheric channel model.
        
        Args:
            distance_km: Propagation distance in kilometers
            wavelength_nm: Wavelength in nanometers (1550nm typical for QKD)
            time_of_day: 'day' or 'night' for Hufnagel-Valley model
        """
        self.distance_km = distance_km
        self.distance_m = distance_km * 1000.0
        self.wavelength_nm = wavelength_nm
        self.wavelength_m = wavelength_nm * 1e-9
        self.time_of_day = time_of_day
        
        # Hufnagel-Valley model parameters
        if time_of_day == 'day':
            self.A = 1.7e-14  # Structure constant at ground
            self.w = 27.0     # Wind speed (m/s)
        else:  # night
            self.A = 1.28e-14
            self.w = 21.0
    
    def cn2_hufnagel_valley(self, altitude_m: float) -> float:
        """
        Calculate Cn² at given altitude using Hufnagel-Valley model.
        
        Cn²(h) = A*exp(-h/1) + 2.7e-16*exp(-h/1.5) + 3.5e-13*w²*exp(-h/10)
        
        Args:
            altitude_m: Altitude in meters
            
        Returns:
            Cn² value (m^(-2/3))
        """
        h_km = altitude_m / 1000.0
        
        term1 = self.A * np.exp(-h_km / 1.0)
        term2 = 2.7e-16 * np.exp(-h_km / 1.5)
        term3 = 3.5e-13 * (self.w ** 2) * np.exp(-h_km / 10.0)
        
        return term1 + term2 + term3
    
    def calculate_rytov_variance(self, cn2: Optional[float] = None) -> float:
        """
        Calculate Rytov variance for atmospheric turbulence.
        
        σ²_R = 0.492 * Cn² * k^(7/6) * z^(11/6)
        
        where:
        - k = 2π/λ (wave number)
        - z = propagation distance
        
        Args:
            cn2: Structure constant. If None, use ground-level value.
            
        Returns:
            Rytov variance (dimensionless)
        """
        if cn2 is None:
            cn2 = self.cn2_hufnagel_valley(0)  # Ground level
        
        # Wave number
        k = 2.0 * np.pi / self.wavelength_m
        
        # Rytov variance
        sigma_r_squared = 0.492 * cn2 * (k ** (7.0/6.0)) * (self.distance_m ** (11.0/6.0))
        
        return sigma_r_squared
    
    def get_turbulence_regime(self, rytov_variance: float) -> str:
        """
        Classify turbulence regime based on Rytov variance.
        
        Args:
            rytov_variance: Rytov variance value
            
        Returns:
            Regime classification string
        """
        if rytov_variance < 0.5:
            return 'very_weak'
        elif rytov_variance < 1.0:
            return 'weak'
        elif rytov_variance < 3.0:
            return 'moderate'
        else:
            return 'strong'
    
    def generate_phase_screen(self, grid_size: int = 256, r0: float = 0.1) -> np.ndarray:
        """
        Generate atmospheric phase screen using Kolmogorov spectrum.
        
        Kolmogorov spectrum: Φ(f) ∝ f^(-11/3)
        
        Args:
            grid_size: Size of square grid (default 256x256)
            r0: Fried parameter (coherence length in meters)
            
        Returns:
            Phase screen (grid_size x grid_size array)
        """
        # Create frequency grid
        fx = np.fft.fftfreq(grid_size)
        fy = np.fft.fftfreq(grid_size)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        # Radial frequency
        f_radial = np.sqrt(fx_grid**2 + fy_grid**2)
        
        # Avoid division by zero
        f_radial[0, 0] = 1e-10
        
        # Kolmogorov spectrum: Φ(f) ∝ f^(-11/3)
        # Power spectral density
        psd = f_radial ** (-11.0/3.0)
        
        # Scale by r0
        psd *= (r0 ** (-5.0/3.0))
        
        # Generate random complex field
        random_complex = (np.random.randn(grid_size, grid_size) + 
                         1j * np.random.randn(grid_size, grid_size))
        
        # Apply spectrum
        field_fourier = np.sqrt(psd) * random_complex
        
        # Inverse FFT to get phase screen
        phase_screen = np.real(np.fft.ifft2(field_fourier))
        
        return phase_screen
    
    def get_scintillation_index(self, rytov_variance: float) -> float:
        """
        Calculate scintillation index from Rytov variance.
        
        For weak turbulence (σ²_R < 1):
            σ²_I ≈ σ²_R
        For moderate/strong turbulence:
            σ²_I ≈ exp(σ²_R) - 1
        
        Args:
            rytov_variance: Rytov variance
            
        Returns:
            Scintillation index
        """
        if rytov_variance < 1.0:
            return rytov_variance
        else:
            return np.exp(rytov_variance) - 1.0


class ChannelAdaptiveStrategy(AttackStrategy):
    """
    Attack strategy that adapts to atmospheric turbulence conditions.
    
    Uses Rytov variance to determine optimal interception probability:
    - Strong turbulence masks eavesdropping → aggressive attack
    - Weak turbulence increases detection risk → conservative attack
    
    PHYSICS:
    Rytov variance quantifies atmospheric turbulence strength:
        σ²_R = 0.492 * Cn² * k^(7/6) * z^(11/6)
    
    where Cn² is the atmospheric structure constant.
    
    STRATEGY:
    - σ²_R < 0.5: Intercept 10% (very weak turbulence)
    - 0.5 ≤ σ²_R < 1.0: Intercept 20% (weak turbulence)
    - 1.0 ≤ σ²_R < 3.0: Intercept 40% (moderate turbulence)
    - σ²_R ≥ 3.0: Intercept 70% (strong turbulence)
    """
    
    def __init__(self,
                 backend: QuantumBackend,
                 distance_km: float = 10.0,
                 wavelength_nm: float = 1550.0,
                 cn2: float = 1.7e-14,
                 time_of_day: str = 'day'):
        """
        Initialize channel-adaptive strategy.
        
        Args:
            backend: Quantum backend for measurements
            distance_km: Free-space link distance in km
            wavelength_nm: Wavelength in nanometers (1550nm typical for QKD)
            cn2: Atmospheric structure constant (default: ground level daytime)
            time_of_day: 'day' or 'night' for atmospheric model
        """
        super().__init__()
        self.backend = backend
        self.distance_km = distance_km
        self.wavelength_nm = wavelength_nm
        self.cn2 = cn2
        self.time_of_day = time_of_day
        
        # Atmospheric model
        self.atmos_model = AtmosphericChannelModel(
            distance_km=distance_km,
            wavelength_nm=wavelength_nm,
            time_of_day=time_of_day
        )
        
        # Calculate initial Rytov variance
        self.current_rytov = self.atmos_model.calculate_rytov_variance(cn2)
        self.turbulence_regime = self.atmos_model.get_turbulence_regime(self.current_rytov)
        
        # History tracking
        self.rytov_history: List[float] = [self.current_rytov]
        self.regime_history: List[str] = [self.turbulence_regime]
        self.intercept_prob_history: List[float] = []
        
        # Measurements
        self.measured_bits: List[int] = []
        self.measurement_bases: List[Basis] = []
        
    def update_rytov_variance(self, cn2: Optional[float] = None) -> None:
        """
        Update Rytov variance based on current atmospheric conditions.
        
        Args:
            cn2: New atmospheric structure constant. If None, keep current.
        """
        if cn2 is not None:
            self.cn2 = cn2
        
        self.current_rytov = self.atmos_model.calculate_rytov_variance(self.cn2)
        self.turbulence_regime = self.atmos_model.get_turbulence_regime(self.current_rytov)
        
        self.rytov_history.append(self.current_rytov)
        self.regime_history.append(self.turbulence_regime)
    
    def get_intercept_probability(self, rytov_variance: Optional[float] = None) -> float:
        """
        Calculate optimal interception probability based on turbulence.
        
        Strategy:
        - Very weak (σ²_R < 0.5): 10% (high detection risk)
        - Weak (0.5 ≤ σ²_R < 1.0): 20%
        - Moderate (1.0 ≤ σ²_R < 3.0): 40%
        - Strong (σ²_R ≥ 3.0): 70% (turbulence masks attack)
        
        Args:
            rytov_variance: Rytov variance. If None, use current value.
            
        Returns:
            Interception probability [0, 1]
        """
        if rytov_variance is None:
            rytov_variance = self.current_rytov
        
        if rytov_variance < 0.5:
            return 0.1
        elif rytov_variance < 1.0:
            return 0.2
        elif rytov_variance < 3.0:
            return 0.4
        else:  # σ²_R ≥ 3.0
            return 0.7
    
    def should_intercept(self, metadata: Dict) -> bool:
        """
        Decide whether to intercept based on atmospheric conditions.
        
        Args:
            metadata: May contain 'atmospheric_Cn2' for dynamic updates
            
        Returns:
            True if should intercept
        """
        # Update Cn2 if provided in metadata
        if 'atmospheric_Cn2' in metadata:
            self.update_rytov_variance(metadata['atmospheric_Cn2'])
        
        # Get current interception probability
        intercept_prob = self.get_intercept_probability()
        self.intercept_prob_history.append(intercept_prob)
        
        return random.random() < intercept_prob
    
    def measure_qubit(self, qubit: Any, basis: Basis) -> int:
        """Measure qubit in specified basis."""
        return self.backend.measure_state(qubit, basis)
    
    def prepare_qubit(self, bit: int, basis: Basis) -> Any:
        """Prepare qubit in specified state."""
        return self.backend.prepare_state(bit, basis)
    
    def intercept(self, qubit: Any, alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        """
        Intercept qubit using random basis measurement.
        
        Args:
            qubit: Quantum state to intercept
            alice_basis: Alice's actual basis (for statistics)
            
        Returns:
            Tuple of (modified_qubit, was_intercepted)
        """
        # Choose random measurement basis
        eve_basis = Basis.random()
        
        # Measure qubit
        measured_bit = self.measure_qubit(qubit, eve_basis)
        
        # Store measurement
        self.measured_bits.append(measured_bit)
        self.measurement_bases.append(eve_basis)
        
        # Calculate information gained
        if alice_basis is not None:
            if eve_basis == alice_basis:
                # Correct basis → full information
                self.statistics['information_gained'] += 1.0
            else:
                # Wrong basis → partial information
                self.statistics['information_gained'] += 0.5
        
        # Prepare and resend
        resent_qubit = self.prepare_qubit(measured_bit, eve_basis)
        
        # Update statistics
        self.statistics['interceptions'] += 1
        self.statistics['successful_measurements'] += 1
        
        return (resent_qubit, True)
    
    def update_strategy(self, feedback: Dict) -> None:
        """
        Update strategy based on atmospheric feedback.
        
        Args:
            feedback: May contain:
                - 'atmospheric_Cn2': Structure constant update
                - 'qber': Observed QBER (for validation)
                - 'time_of_day': Day/night for model adjustment
        """
        # Update Cn2 if provided
        if 'atmospheric_Cn2' in feedback:
            self.update_rytov_variance(feedback['atmospheric_Cn2'])
        
        # Update time of day model
        if 'time_of_day' in feedback:
            new_time = feedback['time_of_day']
            if new_time != self.time_of_day:
                self.time_of_day = new_time
                self.atmos_model = AtmosphericChannelModel(
                    distance_km=self.distance_km,
                    wavelength_nm=self.wavelength_nm,
                    time_of_day=new_time
                )
                self.update_rytov_variance()
    
    def get_statistics(self) -> Dict:
        """Get attack statistics with atmospheric parameters."""
        stats = super().get_statistics()
        
        # Calculate average Rytov variance
        avg_rytov = np.mean(self.rytov_history) if self.rytov_history else 0.0
        
        # Count regime occurrences
        regime_counts = {}
        for regime in self.regime_history:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        stats.update({
            'distance_km': self.distance_km,
            'wavelength_nm': self.wavelength_nm,
            'cn2': self.cn2,
            'time_of_day': self.time_of_day,
            'current_rytov_variance': self.current_rytov,
            'current_turbulence_regime': self.turbulence_regime,
            'current_intercept_probability': self.get_intercept_probability(),
            'avg_rytov_variance': avg_rytov,
            'rytov_history': self.rytov_history.copy(),
            'regime_history': self.regime_history.copy(),
            'regime_counts': regime_counts,
            'intercept_prob_history': self.intercept_prob_history.copy(),
            'scintillation_index': self.atmos_model.get_scintillation_index(self.current_rytov),
            'total_measurements': len(self.measured_bits),
            'basis_distribution': {
                'rectilinear': sum(1 for b in self.measurement_bases if b == Basis.RECTILINEAR),
                'diagonal': sum(1 for b in self.measurement_bases if b == Basis.DIAGONAL)
            }
        })
        return stats


# ============================================================================
# PHOTON NUMBER SPLITTING (PNS) ATTACK
# ============================================================================

class PhotonNumberSplittingAttack(AttackStrategy):
    """
    Photon Number Splitting (PNS) Attack for Weak Coherent Pulse BB84.
    
    CONTEXT:
    Real BB84 implementations use weak coherent pulses (WCP), not true single
    photons. WCPs follow Poisson statistics, occasionally producing multi-photon
    pulses. Eve can non-destructively split off extra photons from these pulses,
    store them, and measure later after basis reconciliation is announced.
    
    ATTACK MECHANISM:
    1. Monitor all pulses (no interception decision needed)
    2. For each pulse, sample photon number n ~ Poisson(μ)
    3. If n ≥ 2:
       - Store one photon for later measurement
       - Let remaining (n-1) photons continue to Bob
       - No disturbance to Bob's measurement!
    4. After basis reconciliation:
       - Measure stored photons in the correct basis
       - Gain perfect information without introducing QBER
    
    THEORETICAL FOUNDATION:
    - Multi-photon probability: P(n≥2) = 1 - (1 + μ)e^(-μ)
    - For typical μ = 0.1: P(n≥2) ≈ 0.0047 (0.47%)
    - Information gain: P(n≥2) × 1 bit per pulse
    - QBER introduced: 0% (undetectable!)
    
    This is the most dangerous known attack on practical BB84 systems.
    """
    
    def __init__(self, 
                 backend: QuantumBackend,
                 mean_photon_number: float = 0.1):
        """
        Initialize PNS attack.
        
        Args:
            backend: Quantum simulation backend
            mean_photon_number: Mean photon number μ for Poisson distribution
                               Typical values: 0.1 (secure) to 0.5 (insecure)
        """
        super().__init__()
        self.backend = backend
        self.mean_photon_number = mean_photon_number
        
        # Storage for photons awaiting basis announcement
        self.stored_photons = []  # List of (timestamp, qubit, n_photons)
        
        # Measurement results after basis reconciliation
        self.final_measurements = []  # List of (timestamp, bit, basis)
        
        # Statistics
        self.pulses_monitored = 0
        self.multi_photon_pulses = 0
        self.photons_stored = 0
        self.photons_measured = 0
        self.successful_extractions = 0
        
    @staticmethod
    def probability_multi_photon(mu: float) -> float:
        """
        Calculate probability of multi-photon pulse.
        
        P(n≥2) = 1 - P(n=0) - P(n=1) = 1 - (1 + μ)e^(-μ)
        
        Args:
            mu: Mean photon number
            
        Returns:
            Probability of n≥2 photons
        """
        return 1.0 - (1.0 + mu) * np.exp(-mu)
    
    @staticmethod
    def expected_information_gain(mu: float) -> float:
        """
        Calculate expected information gain per pulse.
        
        Eve gains 1 full bit for each multi-photon pulse,
        0 bits for single-photon pulses.
        
        Args:
            mu: Mean photon number
            
        Returns:
            Expected bits per pulse
        """
        return PhotonNumberSplittingAttack.probability_multi_photon(mu)
    
    @staticmethod
    def optimal_mu_for_distance(loss_db: float) -> float:
        """
        Calculate optimal mean photon number for given channel loss.
        
        For maximum key rate:
        μ_opt ≈ channel_transmittance = 10^(-loss_db/10)
        
        Args:
            loss_db: Channel loss in dB
            
        Returns:
            Optimal μ value
        """
        transmittance = 10**(-loss_db / 10.0)
        return transmittance
    
    def should_intercept(self, metadata: Optional[Dict] = None) -> bool:
        """
        PNS attack always monitors (non-destructive).
        
        Args:
            metadata: Unused (always monitor)
            
        Returns:
            Always True (monitor all pulses)
        """
        return True
    
    def intercept(self, 
                  qubit: Any,
                  alice_basis: Optional[Basis] = None) -> Tuple[Any, bool]:
        """
        Intercept photon pulse and split if multi-photon.
        
        Algorithm:
        1. Sample n ~ Poisson(μ)
        2. If n ≥ 2: store one photon and let the rest pass through
        3. If n < 2: let through unchanged
        
        Returns:
            Tuple of (modified_qubit, was_intercepted)
        """
        self.pulses_monitored += 1
        timestamp = self.pulses_monitored
        
        # Sample photon number from Poisson distribution
        n_photons = np.random.poisson(self.mean_photon_number)
        
        # Check for multi-photon pulse
        if n_photons >= 2:
            # SUCCESS! Store one photon for later measurement
            self.multi_photon_pulses += 1
            self.photons_stored += 1
            
            # Store photon with metadata
            self.stored_photons.append({
                'timestamp': timestamp,
                'qubit': qubit,
                'n_photons': n_photons,
                'alice_basis': alice_basis  # Usually unknown at this stage
            })
            
            # Record successful interception
            self.statistics['interceptions'] += 1
            
            # Qubit passes through to Bob (with n-1 photons)
            # No disturbance to Bob's measurement!
            return qubit, True
        else:
            # Single photon or empty pulse - let through unchanged
            return qubit, False
    
    def update_strategy(self, feedback: Optional[Dict] = None):
        """
        Update strategy with public information from basis reconciliation.
        
        After Alice and Bob announce their bases, Eve can measure
        stored photons in the correct basis to extract perfect information.
        
        Args:
            feedback: Should contain:
                - 'alice_bases': List of Alice's announced bases
                - 'bob_bases': List of Bob's announced bases (optional)
                - 'matching_indices': Indices where bases matched (optional)
        """
        if feedback is None:
            return

        # Accept both top-level and nested (inside 'public_info') formats
        public_info = feedback.get('public_info', {}) if isinstance(feedback, dict) else {}
        alice_bases = None
        if isinstance(feedback, dict):
            alice_bases = feedback.get('alice_bases') or public_info.get('alice_bases')
        if not alice_bases:
            return
        
        # Measure stored photons now that bases are known
        for stored in self.stored_photons:
            timestamp = stored['timestamp']
            qubit = stored['qubit']
            
            # Get Alice's basis for this pulse
            if timestamp <= len(alice_bases):
                correct_basis = alice_bases[timestamp - 1]
                
                # Measure in correct basis (perfect information!)
                measured_bit = self.backend.measure_state(qubit, correct_basis)
                
                # Record final measurement
                self.final_measurements.append({
                    'timestamp': timestamp,
                    'bit': measured_bit,
                    'basis': correct_basis,
                    'n_photons': stored['n_photons']
                })
                
                self.photons_measured += 1
                self.successful_extractions += 1
                
                # Update information gained
                self.statistics['information_gained'] += 1.0  # Full bit!
        
        # Clear stored photons after measurement
        self.stored_photons.clear()
    
    def get_statistics(self) -> Dict:
        """Get PNS attack statistics."""
        stats = super().get_statistics()
        
        # Calculate probabilities
        multi_photon_prob_actual = (self.multi_photon_pulses / self.pulses_monitored 
                                   if self.pulses_monitored > 0 else 0.0)
        multi_photon_prob_theory = self.probability_multi_photon(self.mean_photon_number)
        
        # Information metrics
        info_per_pulse = (self.successful_extractions / self.pulses_monitored 
                         if self.pulses_monitored > 0 else 0.0)
        expected_info = self.expected_information_gain(self.mean_photon_number)
        
        stats.update({
            'attack_type': 'Photon Number Splitting (PNS)',
            'mean_photon_number': self.mean_photon_number,
            'pulses_monitored': self.pulses_monitored,
            'multi_photon_pulses': self.multi_photon_pulses,
            'multi_photon_probability_actual': multi_photon_prob_actual,
            'multi_photon_probability_theory': multi_photon_prob_theory,
            'photons_stored': self.photons_stored,
            'photons_measured': self.photons_measured,
            'successful_extractions': self.successful_extractions,
            'stored_photons_pending': len(self.stored_photons),
            'information_per_pulse_actual': info_per_pulse,
            'information_per_pulse_expected': expected_info,
            'detection_probability': 0.0,  # UNDETECTABLE!
            'qber_introduced': 0.0,  # NO ERRORS INTRODUCED
            'efficiency': (self.successful_extractions / self.multi_photon_pulses
                          if self.multi_photon_pulses > 0 else 0.0),
            'final_measurements_count': len(self.final_measurements)
        })
        
        return stats


# ============================================================================
# QUANTUM CHANNEL (Backend-aware with Eve integration)
# ============================================================================

class QuantumChannel:
    """Quantum channel for state transmission with optional eavesdropping"""
    
    def __init__(self, 
                 backend: QuantumBackend,
                 loss_rate: float = 0.0,
                 error_rate: float = 0.0,
                 eve: Optional["EveController"] = None,
                 eavesdropper: Optional["EveController"] = None,
                 name: str = "Quantum Channel"):
        """
        Initialize quantum channel.
        
        Args:
            backend: Quantum simulation backend
            loss_rate: Probability of photon loss
            error_rate: Natural channel error rate
            eve: Optional EveController for eavesdropping (legacy/compat)
            eavesdropper: Optional EveController for eavesdropping
            name: Channel identifier
        """
        self.backend = backend
        self.loss_rate = loss_rate
        self.error_rate = error_rate
        # Accept either parameter name for backwards compatibility
        self.eve = eavesdropper if eavesdropper is not None else eve
        self.name = name
        self.transmitted_count = 0
        self.lost_count = 0
        self.error_count = 0
    
    def calculate_current_Cn2(self) -> float:
        """
        Estimate current atmospheric structure constant Cn^2.
        If an atmospheric model is attached to the channel as `atmospheric_model`,
        use it; otherwise, return a reasonable daytime ground-level default.
        """
        try:
            model = getattr(self, 'atmospheric_model', None)
            if model is not None and hasattr(model, 'cn2_hufnagel_valley'):
                return float(model.cn2_hufnagel_valley(0.0))
        except Exception:
            pass
        # Default ground-level daytime Cn^2 (m^(-2/3))
        return 1.7e-14
        
    def transmit(self, states: List) -> List:
        """
        Transmit quantum states through channel.
        Order of operations:
        1. Apply natural channel noise
        2. Apply Eve's interception (if present)
        3. Check for photon loss
        
        Args:
            states: List of quantum states to transmit
            
        Returns:
            List of transmitted (possibly modified) states
        """
        transmitted_states = []
        
        for idx, state in enumerate(states):
            # Check for photon loss first
            if random.random() < self.loss_rate:
                transmitted_states.append(None)  # State lost
                self.lost_count += 1
                continue

            # Apply natural channel noise
            if self.error_rate > 0:
                state = self.backend.apply_channel_noise(state, self.error_rate)
                if random.random() < self.error_rate:
                    self.error_count += 1

            # Apply Eve's interception if present
            if state is not None and self.eve is not None:
                metadata = {
                    'index': idx,
                    'qubit_index': idx,  # keep legacy key for internal stats
                    'timestamp': idx,    # deterministic timestamp placeholder
                    'atmospheric_Cn2': self.calculate_current_Cn2(),
                    'channel_loss_rate': self.loss_rate,
                    'channel_error_rate': self.error_rate,
                }
                state = self.eve.intercept_transmission(state, metadata)

            transmitted_states.append(state)
            self.transmitted_count += 1
        
        return transmitted_states


# ============================================================================
# BB84 PROTOCOL IMPLEMENTATION (Hybrid)
# ============================================================================

class BB84Protocol:
    """Complete BB84 QKD Protocol with hybrid backend support"""
    
    def __init__(self, 
                 backend: Optional[QuantumBackend] = None,
                 use_qiskit: bool = False,
                 output_dir: str = OUTPUT_DIR,
                 run_type: str = "standard",
                 attack_strategy: Optional["AttackStrategy"] = None,
                 enable_detection: bool = True):
        """
        Initialize BB84 protocol
        
        Args:
            backend: Custom backend or None to auto-select
            use_qiskit: Use Qiskit if available
        """
        # Select backend
        if backend is not None:
            self.backend = backend
        elif use_qiskit and QISKIT_AVAILABLE:
            self.backend = QiskitBackend()
        else:
            self.backend = ClassicalBackend()

        # Output manager for image generation
        backend_name = 'qiskit' if use_qiskit else 'classical'
        self.output_manager = OutputManager(output_dir=output_dir, backend_type=backend_name, run_type=run_type)

        # Protocol components (initialized in setup)
        self.alice = None
        self.bob = None
        self.channel = None
        
        # Eve/detector integration
        self.attack_strategy = attack_strategy
        self.detector: Optional[AttackDetector] = AttackDetector() if enable_detection else None
        self.detection_results: Optional[Dict] = None
        
        # Protocol results
        self.raw_key_alice: List[int] = []
        self.raw_key_bob: List[int] = []
        self.final_key: List[int] = []
        self.qber: float = 0.0
        self.stats: Dict = {}
        
    def setup(self, channel_loss: float = 0.1, channel_error: float = 0.02):
        """Setup protocol components"""
        self.alice = Alice(self.backend)
        self.bob = Bob(self.backend)
        # Create Eve if attack strategy provided
        eve = EveController(self.attack_strategy, self.backend) if self.attack_strategy else None
        self.channel = QuantumChannel(self.backend, channel_loss, channel_error, eavesdropper=eve)
        
        # Log setup
        self.output_manager.log("="*60, "header")
        self.output_manager.log(f"BB84 PROTOCOL INITIALIZATION", "header")
        self.output_manager.log(f"Backend: {self.backend.get_name()}", "info")
        self.output_manager.log("\nCHANNEL CONDITIONS", "phase")
        self.output_manager.log(f"Photon Loss Rate: {channel_loss:.1%}", "info")
        self.output_manager.log(f"  → Effect: {channel_loss:.1%} of photons are lost during transmission", "info")
        self.output_manager.log(f"Error Rate: {channel_error:.1%}", "info")
        self.output_manager.log(f"  → Effect: {channel_error:.1%} probability of bit flip per qubit", "info")
        if channel_error > 0.11:
            self.output_manager.log(f"  ⚠ Warning: Error rate exceeds security threshold (11%)", "warning")
        self.output_manager.log("="*60, "header")
        
    def run_protocol(self, num_bits: int = 1000) -> Dict:
        """Execute complete BB84 protocol"""
        
        self.output_manager.log(f"\nStarting BB84 with {num_bits} bits", "phase")
        
        # Phase 1: Quantum Transmission
        self._quantum_transmission_phase(num_bits)
        
        # Phase 2: Basis Reconciliation (Sifting)
        self._basis_reconciliation_phase()
        
        # Phase 3: Error Estimation
        self._error_estimation_phase()
        
        # After Phase 3: Error Estimation — Attack detection and Eve feedback
        if self.detector:
            try:
                # Use raw keys for QBER-based detection; pass full bases for basis-only tests
                alice_bases = [b for b in (self.alice.bases or []) if b is not None]
                bob_bases = [b for b in (self.bob.bases or []) if b is not None]
                self.detection_results = self.detector.detect_attack(
                    self.raw_key_alice, self.raw_key_bob,
                    alice_bases, bob_bases
                )
                if self.detection_results.get('attack_detected'):
                    self.output_manager.log("⚠️ ATTACK DETECTED!", "error")
                    self.output_manager.log(f"Detection method: {self.detection_results}", "warning")
            except Exception as e:
                # Robustness: log and continue protocol
                self.output_manager.log(f"Detection error: {e}", "warning")

        # Provide feedback to Eve for adaptation
        if self.channel and self.channel.eve:
            try:
                self.channel.eve.receive_feedback(self.qber, {
                    'alice_bases': self.alice.bases if self.alice else [],
                    'bob_bases': self.bob.bases if self.bob else [],
                    'key_length': len(self.raw_key_alice)
                })
            except Exception:
                pass

        # Phase 4: Error Correction (simplified)
        self._error_correction_phase()
        
        # Phase 5: Privacy Amplification (simplified)
        self._privacy_amplification_phase()
        
        # Calculate statistics
        self.stats = self._calculate_statistics()
        
        self.output_manager.log("\n" + "="*60, "header")
        self.output_manager.log("BB84 PROTOCOL COMPLETED", "success")
        self.output_manager.log("="*60, "header")
        
        # Save outputs as images
        self.output_manager.save_log_image("protocol_execution")
        self.output_manager.create_summary_table(self.stats, "protocol_summary")
        # Save metadata JSON with run parameters
        try:
            params = {
                'num_bits': num_bits if 'num_bits' in locals() else None,
                'channel_loss': self.channel.loss_rate if self.channel else None,
                'channel_error': self.channel.error_rate if self.channel else None,
                'backend': self.backend.get_name()
            }
            self.output_manager.save_metadata(self.stats, params=params)
        except Exception:
            pass
        
        return self.stats
    
    def _quantum_transmission_phase(self, num_bits: int):
        """Phase 1: Quantum transmission"""
        self.output_manager.log("\n--- PHASE 1: QUANTUM TRANSMISSION ---", "phase")
        
        # Alice prepares and sends
        alice_bits = self.alice.generate_random_bits(num_bits)
        alice_bases = self.alice.choose_random_bases(num_bits)
        states = self.alice.prepare_states()
        
        self.output_manager.log(f"Alice: Generated {num_bits} bits", "info")
        self.output_manager.log(f"Alice: Prepared {len(states)} quantum states", "info")
        
        # Transmission through quantum channel
        received_states = self.channel.transmit(states)
        
        # Filter out lost states
        valid_states = [s for s in received_states if s is not None]
        valid_indices = [i for i, s in enumerate(received_states) if s is not None]
        
        self.output_manager.log(f"Channel: Transmitted {len(valid_states)}/{len(states)} states", "info")
        self.output_manager.log(f"Channel: Lost {self.channel.lost_count} states", "warning")
        
        # Bob measures
        bob_bases = self.bob.choose_random_bases(len(valid_states))
        bob_bits = self.bob.measure_states(valid_states)
        
        self.output_manager.log(f"Bob: Measured {len(bob_bits)} states", "info")
        
        # Store valid indices
        self.valid_indices = valid_indices
        
    def _basis_reconciliation_phase(self):
        """Phase 2: Public comparison of bases"""
        self.output_manager.log("\n--- PHASE 2: BASIS RECONCILIATION ---", "phase")
        
        # Get bases for valid states
        alice_bases_valid = [self.alice.bases[i] for i in self.valid_indices]
        bob_bases = self.bob.bases
        
        # Find matching bases
        matching_bases = []
        for i, (a_basis, b_basis) in enumerate(zip(alice_bases_valid, bob_bases)):
            if a_basis == b_basis:
                matching_bases.append(i)
        
        # Extract raw keys
        self.raw_key_alice = [self.alice.bits[self.valid_indices[i]] 
                              for i in matching_bases]
        self.raw_key_bob = [self.bob.measured_bits[i] 
                           for i in matching_bases if self.bob.measured_bits[i] is not None]
        
        efficiency = len(matching_bases)/len(self.alice.bits) if self.alice.bits else 0
        
        self.output_manager.log(f"Matching bases: {len(matching_bases)}/{len(alice_bases_valid)}", "info")
        self.output_manager.log(f"Sifting efficiency: {efficiency:.2%}", "result")
        self.output_manager.log(f"Raw key length: {len(self.raw_key_alice)} bits", "result")
        
    def _error_estimation_phase(self):
        """Phase 3: Estimate QBER"""
        self.output_manager.log("\n--- PHASE 3: ERROR ESTIMATION ---", "phase")
        
        if len(self.raw_key_alice) == 0:
            self.output_manager.log("No matching bases found!", "error")
            self.qber = 1.0
            return
        
        # Sample subset for error estimation
        sample_size = max(1, len(self.raw_key_alice) // 5)
        sample_indices = random.sample(range(len(self.raw_key_alice)), 
                                      min(sample_size, len(self.raw_key_alice)))
        
        # Count errors
        errors = sum(1 for idx in sample_indices 
                    if self.raw_key_alice[idx] != self.raw_key_bob[idx])
        
        # Calculate QBER
        self.qber = errors / len(sample_indices) if sample_indices else 0
        
        self.output_manager.log(f"Sampled {len(sample_indices)} bits", "info")
        self.output_manager.log(f"Found {errors} errors", "info")
        self.output_manager.log(f"QBER = {self.qber:.2%}", "result")
        
        # Security check
        if self.qber > 0.11:
            self.output_manager.log("WARNING: QBER > 11% - Protocol may be insecure!", "warning")
        else:
            self.output_manager.log("QBER < 11% - Protocol secure", "success")
        
        # Remove sampled bits
        for idx in sorted(sample_indices, reverse=True):
            if idx < len(self.raw_key_alice):
                del self.raw_key_alice[idx]
            if idx < len(self.raw_key_bob):
                del self.raw_key_bob[idx]
        
    def _error_correction_phase(self):
        """Phase 4: Error correction"""
        self.output_manager.log("\n--- PHASE 4: ERROR CORRECTION ---", "phase")
        
        # Count remaining errors
        min_len = min(len(self.raw_key_alice), len(self.raw_key_bob))
        errors = sum(1 for i in range(min_len) 
                    if self.raw_key_alice[i] != self.raw_key_bob[i])
        
        self.output_manager.log(f"Found {errors} errors in {min_len} bits", "info")
        
        # Simplified: perfect error correction
        self.raw_key_bob = self.raw_key_alice[:min_len].copy()
        self.raw_key_alice = self.raw_key_alice[:min_len]
        
        self.output_manager.log("Error correction completed (simplified)", "success")
        
    def _privacy_amplification_phase(self):
        """Phase 5: Privacy amplification"""
        self.output_manager.log("\n--- PHASE 5: PRIVACY AMPLIFICATION ---", "phase")
        
        # Compression ratio based on QBER
        if self.qber == 0:
            compression_ratio = 0.9
        else:
            compression_ratio = max(0.1, 1 - 3 * self.qber)
        
        final_length = int(len(self.raw_key_alice) * compression_ratio)
        
        # Simple privacy amplification: XOR adjacent bits
        self.final_key = []
        for i in range(0, min(final_length * 2, len(self.raw_key_alice) - 1), 2):
            new_bit = self.raw_key_alice[i] ^ self.raw_key_alice[i + 1]
            self.final_key.append(new_bit)
        
        self.output_manager.log(f"Compression ratio: {compression_ratio:.2%}", "info")
        self.output_manager.log(f"Final key length: {len(self.final_key)} bits", "result")
        
    def _calculate_statistics(self) -> Dict:
        """Calculate protocol statistics"""
        stats = {
            'backend': self.backend.get_name(),
            'total_bits_sent': len(self.alice.bits) if self.alice else 0,
            'raw_key_length': len(self.raw_key_alice),
            'final_key_length': len(self.final_key),
            'qber': self.qber,
            'efficiency': len(self.final_key) / len(self.alice.bits) if self.alice and self.alice.bits else 0,
            'channel_loss_rate': self.channel.loss_rate if self.channel else 0,
            'channel_error_rate': self.channel.error_rate if self.channel else 0,
            'secure': self.qber < 0.11,
            'attack_detected': bool(self.detection_results.get('attack_detected')) if self.detection_results else False,
            'eve_interceptions': (self.channel.eve.attack_strategy.statistics.get('interceptions', 0)
                                  if self.channel and self.channel.eve else 0),
            'eve_information': (self.channel.eve.attack_strategy.statistics.get('information_gained', 0.0)
                                if self.channel and self.channel.eve else 0.0)
        }
        return stats
    
    def visualize_results(self, save_path: Optional[str] = None):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'BB84 Protocol Analysis - Backend: {self.backend.get_name()}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Key generation pipeline
        ax1 = fig.add_subplot(gs[0, :2])
        stages = ['Sent', 'Received', 'Sifted', 'Corrected', 'Final']
        counts = [
            len(self.alice.bits) if self.alice else 0,
            self.channel.transmitted_count if self.channel else 0,
            len(self.raw_key_alice) + len(self.final_key),
            len(self.raw_key_alice),
            len(self.final_key)
        ]
        colors = ['blue', 'green', 'orange', 'yellow', 'red']
        bars = ax1.bar(stages, counts, color=colors)
        ax1.set_ylabel('Number of Bits')
        ax1.set_title('Key Generation Pipeline')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom')
        
        # Plot 2: Protocol metrics
        ax2 = fig.add_subplot(gs[0, 2])
        metrics = ['QBER', 'Efficiency', 'Loss']
        values = [
            self.qber * 100,
            self.stats['efficiency'] * 100,
            self.channel.loss_rate * 100 if self.channel else 0
        ]
        colors = ['red' if self.qber > 0.11 else 'green', 'blue', 'orange']
        ax2.bar(metrics, values, color=colors)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Key Metrics')
        ax2.axhline(y=11, color='r', linestyle='--', alpha=0.5, label='QBER Threshold')
        ax2.legend()
        
        # Plot 3: Basis distribution
        ax3 = fig.add_subplot(gs[1, 0])
        if self.alice and self.alice.bases:
            alice_basis_counts = Counter([str(b) for b in self.alice.bases])
            labels = list(alice_basis_counts.keys())
            sizes = list(alice_basis_counts.values())
            ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Alice Basis Distribution')
        
        # Plot 4: Bob's basis distribution
        ax4 = fig.add_subplot(gs[1, 1])
        if self.bob and self.bob.bases:
            bob_basis_counts = Counter([str(b) for b in self.bob.bases])
            labels = list(bob_basis_counts.keys())
            sizes = list(bob_basis_counts.values())
            ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Bob Basis Distribution')
        
        # Plot 5: Key bits comparison (sample)
        ax5 = fig.add_subplot(gs[1, 2])
        if len(self.raw_key_alice) > 0 and len(self.raw_key_bob) > 0:
            sample_size = min(50, len(self.raw_key_alice))
            alice_sample = self.raw_key_alice[:sample_size]
            bob_sample = self.raw_key_bob[:sample_size]
            
            x = range(sample_size)
            ax5.scatter(x, alice_sample, c='blue', label='Alice', alpha=0.6, s=20)
            ax5.scatter(x, bob_sample, c='red', label='Bob', alpha=0.6, s=20, marker='x')
            
            # Highlight errors
            for i in range(sample_size):
                if alice_sample[i] != bob_sample[i]:
                    ax5.axvspan(i-0.3, i+0.3, alpha=0.3, color='red')
            
            ax5.set_xlabel('Bit Position')
            ax5.set_ylabel('Bit Value')
            ax5.set_title(f'First {sample_size} Raw Key Bits')
            ax5.legend()
            ax5.set_ylim(-0.1, 1.1)
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Protocol timeline
        ax6 = fig.add_subplot(gs[2, :])
        phases = ['Transmission', 'Sifting', 'Error Est.', 'Correction', 'Privacy Amp.']
        phase_times = [1, 0.5, 0.3, 0.4, 0.3]  # Relative durations
        colors_timeline = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        cumulative = 0
        for i, (phase, duration, color) in enumerate(zip(phases, phase_times, colors_timeline)):
            ax6.barh(0, duration, left=cumulative, color=color, edgecolor='black', height=0.5)
            ax6.text(cumulative + duration/2, 0, phase, ha='center', va='center', 
                    fontsize=10, fontweight='bold')
            cumulative += duration
        
        ax6.set_xlim(0, cumulative)
        ax6.set_ylim(-0.5, 0.5)
        ax6.set_title('Protocol Execution Timeline')
        ax6.axis('off')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        else:
            filepath = os.path.join(self.output_manager.run_dir, "protocol_visualization.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        
        plt.close()
        return filepath if not save_path else save_path


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

class BB84Simulator:
    """High-level simulator for running BB84 experiments"""
    
    def __init__(self, use_qiskit: bool = False, output_dir: str = OUTPUT_DIR, run_type: str = "simulator"):
        self.use_qiskit = use_qiskit and QISKIT_AVAILABLE
        backend_name = 'qiskit' if self.use_qiskit else 'classical'
        self.output_manager = OutputManager(output_dir=output_dir, backend_type=backend_name, run_type=run_type)
        self.results_history = []
        
    def run_single_simulation(self,
                             num_bits: int = 1000,
                             channel_loss: float = 0.1,
                             channel_error: float = 0.02) -> Dict:
        """Run a single BB84 simulation"""
        logger.info(f"Using {'Qiskit' if self.use_qiskit else 'Classical'} backend")
        protocol = BB84Protocol(use_qiskit=self.use_qiskit, output_dir=self.output_manager.output_dir, run_type=self.output_manager.run_type)
        # Ensure protocol writes into the same output tree
        protocol.output_manager = self.output_manager
        protocol.setup(channel_loss, channel_error)
        stats = protocol.run_protocol(num_bits)
        
        # Generate visualization
        protocol.visualize_results()
        
        # Store results
        self.results_history.append(stats)
        
        return stats
    
    def run_parameter_sweep(self,
                           num_bits: int = 1000,
                           loss_rates: List[float] = None,
                           error_rates: List[float] = None,
                           trials_per_config: int = 3) -> Dict:
        """Run simulations across parameter space"""
        logger.info(f"Using {'Qiskit' if self.use_qiskit else 'Classical'} backend for parameter sweep")
        logger.info(f"Running {trials_per_config} trials for each configuration")
        
        if loss_rates is None:
            loss_rates = [0.0, 0.05, 0.1, 0.15, 0.2]
        if error_rates is None:
            error_rates = [0.0, 0.02, 0.05, 0.08, 0.10]
        
        results = {}
        
        for loss in loss_rates:
            for error in error_rates:
                config_results = []
                for trial in range(trials_per_config):
                    stats = self.run_single_simulation(num_bits, loss, error)
                    config_results.append(stats)
                
                # Average results
                avg_stats = {
                    'qber': np.mean([r['qber'] for r in config_results]),
                    'efficiency': np.mean([r['efficiency'] for r in config_results]),
                    'secure': all(r['secure'] for r in config_results)
                }
                results[(loss, error)] = avg_stats
        
        # Create heatmap visualization
        self._create_heatmap(results, loss_rates, error_rates)
        
        return results
    
    def _create_heatmap(self, results: Dict, loss_rates: List, error_rates: List):
        """Create heatmap of parameter sweep results"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Prepare data matrices
        qber_matrix = np.zeros((len(error_rates), len(loss_rates)))
        efficiency_matrix = np.zeros((len(error_rates), len(loss_rates)))
        security_matrix = np.zeros((len(error_rates), len(loss_rates)))
        
        for i, error in enumerate(error_rates):
            for j, loss in enumerate(loss_rates):
                stats = results[(loss, error)]
                qber_matrix[i, j] = stats['qber']
                efficiency_matrix[i, j] = stats['efficiency']
                security_matrix[i, j] = 1 if stats['secure'] else 0
        
        # Plot QBER heatmap
        im1 = axes[0].imshow(qber_matrix, cmap='RdYlGn_r', aspect='auto')
        axes[0].set_title('QBER (%)')
        axes[0].set_xlabel('Loss Rate')
        axes[0].set_ylabel('Error Rate')
        axes[0].set_xticks(range(len(loss_rates)))
        axes[0].set_xticklabels([f'{l:.0%}' for l in loss_rates])
        axes[0].set_yticks(range(len(error_rates)))
        axes[0].set_yticklabels([f'{e:.0%}' for e in error_rates])
        plt.colorbar(im1, ax=axes[0])
        
        # Plot Efficiency heatmap
        im2 = axes[1].imshow(efficiency_matrix, cmap='viridis', aspect='auto')
        axes[1].set_title('Efficiency (%)')
        axes[1].set_xlabel('Loss Rate')
        axes[1].set_ylabel('Error Rate')
        axes[1].set_xticks(range(len(loss_rates)))
        axes[1].set_xticklabels([f'{l:.0%}' for l in loss_rates])
        axes[1].set_yticks(range(len(error_rates)))
        axes[1].set_yticklabels([f'{e:.0%}' for e in error_rates])
        plt.colorbar(im2, ax=axes[1])
        
        # Plot Security heatmap
        im3 = axes[2].imshow(security_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[2].set_title('Security Status')
        axes[2].set_xlabel('Loss Rate')
        axes[2].set_ylabel('Error Rate')
        axes[2].set_xticks(range(len(loss_rates)))
        axes[2].set_xticklabels([f'{l:.0%}' for l in loss_rates])
        axes[2].set_yticks(range(len(error_rates)))
        axes[2].set_yticklabels([f'{e:.0%}' for e in error_rates])
        
        # Custom colorbar for security
        cbar = plt.colorbar(im3, ax=axes[2], ticks=[0, 1])
        cbar.set_ticklabels(['Insecure', 'Secure'])
        
        plt.suptitle('BB84 Parameter Sweep Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_manager.run_dir, "parameter_sweep_heatmap.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath
    
    def compare_backends(self, num_bits: int = 1000, trials: int = 5):
        """Compare classical vs Qiskit backends"""
        results = {'Classical': [], 'Qiskit': []}

        logger.info(f"Running {trials} trials for each backend")
        logger.info("Starting Classical backend trials...")
        # Run with classical backend
        for i in range(trials):
            logger.info(f"Classical trial {i+1}/{trials}")
            protocol = BB84Protocol(use_qiskit=False)
            protocol.setup(0.1, 0.02)
            stats = protocol.run_protocol(num_bits)
            results['Classical'].append(stats)
        
        # Run with Qiskit backend if available
        if QISKIT_AVAILABLE:
            logger.info("Starting Qiskit backend trials...")
            for i in range(trials):
                logger.info(f"Qiskit trial {i+1}/{trials}")
                protocol = BB84Protocol(use_qiskit=True)
                protocol.setup(0.1, 0.02)
                stats = protocol.run_protocol(num_bits)
                results['Qiskit'].append(stats)
        
        # Create comparison visualization
        self._create_backend_comparison(results)
        
        return results
    
    def _create_backend_comparison(self, results: Dict):
        """Create visualization comparing backends"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        backends = list(results.keys())
        metrics = ['QBER', 'Efficiency']
        
        for idx, metric in enumerate(metrics):
            data = []
            for backend in backends:
                if metric == 'QBER':
                    values = [r['qber'] * 100 for r in results[backend]]
                else:
                    values = [r['efficiency'] * 100 for r in results[backend]]
                data.append(values)
            
            bp = axes[idx].boxplot(data, tick_labels=backends)
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_ylabel(f'{metric} (%)')
            axes[idx].grid(True, alpha=0.3)
            
            # Add mean markers
            for i, d in enumerate(data):
                mean_val = np.mean(d)
                axes[idx].plot(i+1, mean_val, 'r^', markersize=10, label='Mean' if i == 0 else '')
            
            if idx == 0:
                axes[idx].axhline(y=11, color='r', linestyle='--', alpha=0.5, label='Security Threshold')
            
            axes[idx].legend()
        
        plt.suptitle('Backend Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_manager.run_dir, "backend_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_summary_report(stats: Dict, output_dir: str = OUTPUT_DIR):
    """Create a comprehensive summary report as an image"""
    
    fig = plt.figure(figsize=(11, 8.5))  # Letter size
    fig.suptitle('BB84 Quantum Key Distribution - Summary Report', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Create text summary
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    report_text = f"""
    SIMULATION PARAMETERS
    ═══════════════════════════════════════════════════════════════
    Backend:                {stats.get('backend', 'Classical')}
    Total Bits Sent:        {stats.get('total_bits_sent', 0):,}
    Channel Loss Rate:      {stats.get('channel_loss_rate', 0):.1%}
    Channel Error Rate:     {stats.get('channel_error_rate', 0):.1%}
    
    PROTOCOL RESULTS
    ═══════════════════════════════════════════════════════════════
    Raw Key Length:         {stats.get('raw_key_length', 0):,} bits
    Final Key Length:       {stats.get('final_key_length', 0):,} bits
    QBER:                   {stats.get('qber', 0):.2%}
    Overall Efficiency:     {stats.get('efficiency', 0):.2%}
    
    SECURITY ASSESSMENT
    ═══════════════════════════════════════════════════════════════
    Security Status:        {'SECURE ✓' if stats.get('secure', False) else 'INSECURE ✗'}
    QBER Threshold:         11.00%
    Below Threshold:        {'Yes' if stats.get('qber', 0) < 0.11 else 'No'}
    
    RECOMMENDATIONS
    ═══════════════════════════════════════════════════════════════
    """
    
    if stats.get('qber', 0) > 0.11:
        report_text += """
    ⚠ High QBER detected. Possible causes:
      • Eavesdropping attempt
      • High channel noise
      • Equipment malfunction
    
    Recommended Actions:
      • Abort key distribution
      • Check equipment calibration
      • Verify channel integrity
    """
    else:
        report_text += """
    ✓ Protocol executed successfully
    ✓ Key can be used for secure communication
    ✓ No signs of eavesdropping detected
    
    Next Steps:
      • Proceed with encrypted communication
      • Monitor QBER for changes
      • Schedule regular key refresh
    """
    
    ax.text(0.1, 0.9, report_text, transform=ax.transAxes,
           fontsize=11, fontfamily='monospace',
           verticalalignment='top')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.5, 0.02, f"Report Generated: {timestamp}",
           transform=ax.transAxes, fontsize=9,
           horizontalalignment='center', style='italic')
    
    filepath = os.path.join(output_dir, f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filepath


# ============================================================================
# VISUALIZATION AND ANALYSIS TOOLS
# ============================================================================

class VisualizationManager:
    """
    Comprehensive visualization and analysis tools for quantum attack results.
    Generates publication-quality figures and LaTeX tables for research papers.
    """
    
    # Matplotlib configuration for publication quality
    PLOT_CONFIG = {
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    }
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualization manager.
        
        Args:
            output_dir: Directory to save generated figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply publication-quality settings
        for key, value in self.PLOT_CONFIG.items():
            plt.rcParams[key] = value
    
    def plot_qber_evolution(self, 
                           qber_history: List[float], 
                           threshold: float = 0.11,
                           title: str = "QBER Evolution Over Time",
                           save_name: Optional[str] = None) -> str:
        """
        Plot QBER time series with detection threshold line.
        
        Args:
            qber_history: List of QBER values over time
            threshold: Detection threshold line to overlay
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        time_steps = np.arange(len(qber_history))
        ax.plot(time_steps, qber_history, 'b-', linewidth=2, label='Measured QBER')
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                   label=f'Detection Threshold ({threshold:.2%})')
        
        # Highlight regions above threshold
        above_threshold = np.array(qber_history) > threshold
        if np.any(above_threshold):
            ax.fill_between(time_steps, 0, 1, where=above_threshold, 
                           alpha=0.2, color='red', label='Attack Detected')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('QBER')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, min(1.0, max(qber_history) * 1.1))
        
        filename = save_name or f"qber_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_intercept_probability(self,
                                   prob_history: List[float],
                                   title: str = "Intercept Probability Adaptation",
                                   save_name: Optional[str] = None) -> str:
        """
        Plot adaptive intercept probability evolution over time.
        
        Args:
            prob_history: List of intercept probability values
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        time_steps = np.arange(len(prob_history))
        ax.plot(time_steps, prob_history, 'g-', linewidth=2, marker='o', 
               markersize=4, label='Intercept Probability')
        
        # Add moving average
        if len(prob_history) > 10:
            window = min(20, len(prob_history) // 5)
            moving_avg = np.convolve(prob_history, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(prob_history)), moving_avg, 
                   'orange', linewidth=2, linestyle='--', label=f'{window}-step Moving Avg')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Intercept Probability')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
        
        filename = save_name or f"intercept_prob_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_information_leakage(self,
                                strategies_dict: Dict[str, float],
                                title: str = "Information Leakage Comparison",
                                save_name: Optional[str] = None) -> str:
        """
        Generate bar chart comparing information leakage across strategies.
        
        Args:
            strategies_dict: Dictionary mapping strategy names to information values (bits)
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategies = list(strategies_dict.keys())
        information = list(strategies_dict.values())
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(strategies)))
        bars = ax.bar(strategies, information, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Attack Strategy')
        ax.set_ylabel('Information Leakage (bits)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        filename = save_name or f"info_leakage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_detection_roc_curve(self,
                                results: Dict[str, Dict[str, List[float]]],
                                title: str = "Detection ROC Curves",
                                save_name: Optional[str] = None) -> str:
        """
        Plot ROC curves for different detection methods.
        
        Args:
            results: Dict mapping method names to {'fpr': [...], 'tpr': [...]} lists
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for idx, (method_name, roc_data) in enumerate(results.items()):
            fpr = roc_data.get('fpr', [])
            tpr = roc_data.get('tpr', [])
            
            if len(fpr) > 0 and len(tpr) > 0:
                # Calculate AUC using trapezoidal rule
                try:
                    from numpy import trapezoid
                    auc = trapezoid(tpr, fpr) if len(fpr) > 1 else 0.0
                except ImportError:
                    # Fallback for older numpy
                    auc = np.trapz(tpr, fpr) if len(fpr) > 1 else 0.0
                ax.plot(fpr, tpr, linewidth=2, color=colors[idx],
                       label=f'{method_name} (AUC={auc:.3f})')
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        
        filename = save_name or f"detection_roc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_adaptive_strategy_dashboard(self,
                                        qber_history: List[float],
                                        intercept_prob_history: List[float],
                                        info_gain_history: List[float],
                                        detection_metrics: Dict[str, List[float]],
                                        qber_threshold: float = 0.11,
                                        title: str = "Adaptive Strategy Evolution Dashboard",
                                        save_name: Optional[str] = None) -> str:
        """
        Create 4-subplot dashboard showing adaptive strategy evolution.
        
        Args:
            qber_history: QBER values over time
            intercept_prob_history: Intercept probability adaptation
            info_gain_history: Cumulative information gain
            detection_metrics: Dict of detection metric time series
            qber_threshold: QBER detection threshold
            title: Overall figure title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        time_steps = np.arange(len(qber_history))
        
        # Subplot 1: QBER with confidence bands
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_steps, qber_history, 'b-', linewidth=2, label='QBER')
        ax1.axhline(y=qber_threshold, color='r', linestyle='--', linewidth=2,
                   label=f'Threshold ({qber_threshold:.2%})')
        
        # Add confidence bands (using moving std if enough data)
        if len(qber_history) > 20:
            window = min(50, len(qber_history) // 4)
            qber_array = np.array(qber_history)
            moving_mean = np.convolve(qber_array, np.ones(window)/window, mode='valid')
            moving_std = np.array([np.std(qber_array[max(0, i-window):i+1]) 
                                  for i in range(len(qber_array))])
            
            if len(moving_std) >= window:
                offset = (len(qber_array) - len(moving_mean)) // 2
                x_band = time_steps[offset:offset+len(moving_mean)]
                ax1.fill_between(x_band, 
                               np.maximum(0, moving_mean - moving_std[offset:offset+len(moving_mean)]),
                               np.minimum(1, moving_mean + moving_std[offset:offset+len(moving_mean)]),
                               alpha=0.3, color='blue', label='±1σ Confidence')
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('QBER')
        ax1.set_title('QBER Evolution with Confidence Bands')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, min(1.0, max(qber_history) * 1.1) if qber_history else 1.0)
        
        # Subplot 2: Intercept probability adaptation
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_steps[:len(intercept_prob_history)], intercept_prob_history, 
                'g-', linewidth=2, marker='o', markersize=3, label='Intercept Prob.')
        
        if len(intercept_prob_history) > 10:
            window = min(20, len(intercept_prob_history) // 5)
            moving_avg = np.convolve(intercept_prob_history, np.ones(window)/window, mode='valid')
            ax2.plot(np.arange(window-1, len(intercept_prob_history)), moving_avg,
                    'orange', linewidth=2, linestyle='--', label='Moving Avg')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Intercept Probability')
        ax2.set_title('Intercept Probability Adaptation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # Subplot 3: Information gain accumulation
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_steps[:len(info_gain_history)], info_gain_history,
                'purple', linewidth=2, label='Cumulative Info Gain')
        ax3.fill_between(time_steps[:len(info_gain_history)], 0, info_gain_history,
                        alpha=0.3, color='purple')
        
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Information Gain (bits)')
        ax3.set_title('Cumulative Information Gain')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot 4: Detection metrics evolution
        ax4 = fig.add_subplot(gs[1, 1])
        colors_metrics = plt.cm.Set2(np.linspace(0, 1, len(detection_metrics)))
        
        for idx, (metric_name, metric_values) in enumerate(detection_metrics.items()):
            steps = np.arange(len(metric_values))
            ax4.plot(steps, metric_values, linewidth=2, color=colors_metrics[idx],
                    label=metric_name, marker='s', markersize=3)
        
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Detection Metric Value')
        ax4.set_title('Detection Metrics Evolution')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        filename = save_name or f"adaptive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_rytov_variance(self,
                           time: List[float],
                           rytov_values: List[float],
                           title: str = "Atmospheric Turbulence (Rytov Variance)",
                           save_name: Optional[str] = None) -> str:
        """
        Plot Rytov variance evolution showing atmospheric turbulence strength.
        
        Args:
            time: Time values
            rytov_values: Rytov variance values
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(time, rytov_values, 'darkblue', linewidth=2, label='Rytov Variance')
        ax.fill_between(time, 0, rytov_values, alpha=0.3, color='skyblue')
        
        # Mark turbulence regimes
        weak_threshold = 0.3
        moderate_threshold = 1.0
        
        ax.axhline(y=weak_threshold, color='orange', linestyle='--', 
                  linewidth=1.5, label=f'Weak Turbulence ({weak_threshold})')
        ax.axhline(y=moderate_threshold, color='red', linestyle='--',
                  linewidth=1.5, label=f'Strong Turbulence ({moderate_threshold})')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rytov Variance σ²ᴿ')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')
        
        filename = save_name or f"rytov_variance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_attack_aggression_vs_turbulence(self,
                                            turbulence_levels: List[float],
                                            aggression_levels: List[float],
                                            qber_values: List[float],
                                            title: str = "Attack Aggression vs Atmospheric Turbulence",
                                            save_name: Optional[str] = None) -> str:
        """
        Create scatter plot showing relationship between attack aggression and turbulence.
        
        Args:
            turbulence_levels: Rytov variance or cn2 values
            aggression_levels: Attack intercept probabilities
            qber_values: Resulting QBER (for color coding)
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(turbulence_levels, aggression_levels, 
                           c=qber_values, s=100, cmap='plasma',
                           edgecolors='black', linewidth=1.5, alpha=0.8)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('QBER', rotation=270, labelpad=20)
        
        ax.set_xlabel('Turbulence Level (Rytov Variance)')
        ax.set_ylabel('Attack Aggression (Intercept Probability)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        filename = save_name or f"aggression_turbulence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_phase_screen(self,
                         screen_array: np.ndarray,
                         title: str = "Atmospheric Phase Screen",
                         save_name: Optional[str] = None) -> str:
        """
        Visualize 2D atmospheric phase screen.
        
        Args:
            screen_array: 2D array of phase values
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 9))
        
        im = ax.imshow(screen_array, cmap='twilight', interpolation='bilinear',
                      extent=[0, screen_array.shape[1], 0, screen_array.shape[0]])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Phase (radians)', rotation=270, labelpad=20)
        
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(title)
        
        filename = save_name or f"phase_screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def animate_beam_propagation(self,
                                intensity_frames: List[np.ndarray],
                                save_name: Optional[str] = None,
                                fps: int = 10) -> str:
        """
        Create animation of beam propagation through turbulence.
        Saves as series of frames (animation requires additional dependencies).
        
        Args:
            intensity_frames: List of 2D intensity arrays
            save_name: Optional base filename
            fps: Frames per second (metadata only for future animation)
            
        Returns:
            Path to saved frame directory
        """
        base_name = save_name or f"beam_anim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        frame_dir = os.path.join(self.output_dir, base_name)
        os.makedirs(frame_dir, exist_ok=True)
        
        for idx, frame in enumerate(intensity_frames):
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(frame, cmap='hot', interpolation='gaussian')
            ax.set_title(f'Beam Intensity - Frame {idx+1}/{len(intensity_frames)}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Intensity', rotation=270, labelpad=20)
            
            frame_path = os.path.join(frame_dir, f"frame_{idx:04d}.png")
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save metadata
        metadata = {
            'num_frames': len(intensity_frames),
            'fps': fps,
            'frame_dir': frame_dir,
            'created': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(frame_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return frame_dir
    
    def plot_error_pattern_analysis(self,
                                   errors: List[int],
                                   title: str = "Error Pattern Analysis",
                                   save_name: Optional[str] = None) -> str:
        """
        Analyze error patterns using autocorrelation and runs test visualization.
        
        Args:
            errors: Binary error sequence (0=correct, 1=error)
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Error sequence visualization
        time_steps = np.arange(len(errors))
        ax1.scatter(time_steps, errors, c=errors, cmap='bwr', s=10, alpha=0.6)
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Error (0/1)')
        ax1.set_title('Error Sequence')
        ax1.set_ylim(-0.1, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Autocorrelation
        if len(errors) > 10:
            max_lag = min(100, len(errors) // 4)
            errors_array = np.array(errors)
            mean_err = np.mean(errors_array)
            
            autocorr = []
            for lag in range(max_lag):
                if lag == 0:
                    autocorr.append(1.0)
                else:
                    corr = np.corrcoef(errors_array[:-lag], errors_array[lag:])[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0.0)
            
            ax2.stem(range(max_lag), autocorr, basefmt=' ')
            ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax2.axhline(y=1.96/np.sqrt(len(errors)), color='r', linestyle='--', 
                       linewidth=1, label='95% Confidence')
            ax2.axhline(y=-1.96/np.sqrt(len(errors)), color='r', linestyle='--', linewidth=1)
            ax2.set_xlabel('Lag')
            ax2.set_ylabel('Autocorrelation')
            ax2.set_title('Autocorrelation Function')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Run length distribution
        runs = []
        current_val = errors[0] if errors else 0
        current_run = 1
        
        for e in errors[1:]:
            if e == current_val:
                current_run += 1
            else:
                runs.append(current_run)
                current_val = e
                current_run = 1
        runs.append(current_run)
        
        run_counts = Counter(runs)
        run_lengths = sorted(run_counts.keys())
        counts = [run_counts[rl] for rl in run_lengths]
        
        ax3.bar(run_lengths, counts, color='steelblue', edgecolor='black')
        ax3.set_xlabel('Run Length')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Run Length Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = save_name or f"error_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_basis_bias_detection(self,
                                  basis_sequence: List[int],
                                  expected_prob: float = 0.5,
                                  title: str = "Basis Bias Detection (Chi-Square Test)",
                                  save_name: Optional[str] = None) -> str:
        """
        Visualize basis randomness and chi-square test results.
        
        Args:
            basis_sequence: Sequence of basis choices (0 or 1)
            expected_prob: Expected probability of each basis
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Cumulative proportion
        cumsum = np.cumsum(basis_sequence)
        positions = np.arange(1, len(basis_sequence) + 1)
        cumulative_prop = cumsum / positions
        
        ax1.plot(positions, cumulative_prop, 'b-', linewidth=2, label='Observed Proportion')
        ax1.axhline(y=expected_prob, color='r', linestyle='--', linewidth=2,
                   label=f'Expected ({expected_prob:.2f})')
        
        # Confidence bounds
        std_dev = np.sqrt(expected_prob * (1 - expected_prob) / positions)
        ax1.fill_between(positions, 
                        expected_prob - 1.96 * std_dev,
                        expected_prob + 1.96 * std_dev,
                        alpha=0.3, color='red', label='95% CI')
        
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Cumulative Proportion of Basis 1')
        ax1.set_title('Cumulative Basis Proportion')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chi-square statistic over sliding windows
        window_size = min(100, len(basis_sequence) // 10) if len(basis_sequence) > 100 else len(basis_sequence)
        chi_square_values = []
        window_positions = []
        
        for i in range(window_size, len(basis_sequence) + 1, window_size // 2):
            window = basis_sequence[i-window_size:i]
            observed_ones = sum(window)
            expected_ones = window_size * expected_prob
            
            chi_square = ((observed_ones - expected_ones) ** 2) / expected_ones + \
                        ((window_size - observed_ones - window_size + expected_ones) ** 2) / (window_size - expected_ones)
            
            chi_square_values.append(chi_square)
            window_positions.append(i)
        
        if chi_square_values:
            ax2.plot(window_positions, chi_square_values, 'g-', linewidth=2, marker='o')
            ax2.axhline(y=3.841, color='r', linestyle='--', linewidth=2,
                       label='Critical Value (p=0.05, df=1)')
            ax2.set_xlabel('Window End Position')
            ax2.set_ylabel('χ² Statistic')
            ax2.set_title(f'Chi-Square Test (Window Size = {window_size})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = save_name or f"basis_bias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_mutual_information_matrix(self,
                                      mi_matrix: np.ndarray,
                                      labels: List[str] = None,
                                      title: str = "Mutual Information Matrix (Alice-Bob-Eve)",
                                      save_name: Optional[str] = None) -> str:
        """
        Plot mutual information matrix between parties.
        
        Args:
            mi_matrix: Square matrix of mutual information values
            labels: Party labels (default: ["Alice", "Bob", "Eve"])
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        if labels is None:
            n = mi_matrix.shape[0]
            labels = ["Alice", "Bob", "Eve"] if n == 3 else [f"Party {i+1}" for i in range(n)]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        im = ax.imshow(mi_matrix, cmap='YlOrRd', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mutual Information (bits)', rotation=270, labelpad=20)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{mi_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(title)
        plt.tight_layout()
        
        filename = save_name or f"mi_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_holevo_bound_comparison(self,
                                    strategies: List[str],
                                    theoretical_bounds: List[float],
                                    actual_information: List[float],
                                    title: str = "Holevo Bound: Theoretical vs Actual",
                                    save_name: Optional[str] = None) -> str:
        """
        Compare theoretical Holevo bounds with actual information leakage.
        
        Args:
            strategies: Strategy names
            theoretical_bounds: Theoretical maximum information (Holevo bound)
            actual_information: Actual measured information leakage
            title: Plot title
            save_name: Optional custom filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, theoretical_bounds, width, 
                      label='Holevo Bound (Theoretical)', 
                      color='skyblue', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, actual_information, width,
                      label='Actual Information',
                      color='salmon', edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Attack Strategy')
        ax.set_ylabel('Information (bits)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = save_name or f"holevo_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def export_results_table(self,
                            results: Dict[str, Dict[str, Any]],
                            filename: str = None,
                            caption: str = "Experimental Results Summary") -> str:
        """
        Export results as LaTeX table formatted for IEEE two-column papers.
        
        Args:
            results: Dictionary mapping strategy names to metric dictionaries
            filename: Output filename (default: results_table_<timestamp>.tex)
            caption: Table caption
            
        Returns:
            Path to saved LaTeX file
        """
        if filename is None:
            filename = f"results_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Extract metric names
        first_strategy = list(results.values())[0]
        metrics = list(first_strategy.keys())
        
        with open(filepath, 'w') as f:
            # LaTeX table header
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{" + caption + "}\n")
            f.write("\\label{tab:results}\n")
            f.write("\\begin{tabular}{l" + "c" * len(metrics) + "}\n")
            f.write("\\hline\n")
            
            # Column headers
            header = "Strategy & " + " & ".join(metrics) + " \\\\\n"
            f.write(header)
            f.write("\\hline\n")
            
            # Data rows
            for strategy_name, metric_dict in results.items():
                row_data = [strategy_name.replace('_', '\\_')]
                
                for metric in metrics:
                    value = metric_dict.get(metric, 0)
                    
                    # Format based on type
                    if isinstance(value, dict) and 'mean' in value and 'std' in value:
                        # Mean ± std format
                        mean = value['mean']
                        std = value['std']
                        significance = value.get('significant', False)
                        marker = '$^*$' if significance else ''
                        formatted = f"${mean:.3f} \\pm {std:.3f}${marker}"
                    elif isinstance(value, float):
                        formatted = f"{value:.3f}"
                    elif isinstance(value, int):
                        formatted = f"{value}"
                    else:
                        formatted = str(value)
                    
                    row_data.append(formatted)
                
                row = " & ".join(row_data) + " \\\\\n"
                f.write(row)
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\vspace{2mm}\n")
            f.write("\\\\{\\footnotesize $^*$ indicates statistical significance at $p < 0.05$}\n")
            f.write("\\end{table}\n")
        
        return filepath


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    import argparse
    parser = argparse.ArgumentParser(description='BB84 Quantum Key Distribution Protocol Simulator')
    parser.add_argument('--use-qiskit', action='store_true', 
                      help='Use Qiskit backend instead of classical simulation')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                      help='Base output directory for run artifacts')
    parser.add_argument('--run-type', type=str, default='standard',
                      help='Run type descriptor (standard, sweep, comparison)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level')
    args = parser.parse_args()
    # Configure module logger level
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    # Root logger basic config to ensure console output
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("BB84 QUANTUM KEY DISTRIBUTION - HYBRID IMPLEMENTATION")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Qiskit available: {QISKIT_AVAILABLE}")
    logger.info(f"Selected backend: {'Qiskit' if args.use_qiskit else 'Classical'}")
    
    # Create simulator
    simulator = BB84Simulator(use_qiskit=args.use_qiskit, output_dir=args.output_dir, run_type=args.run_type)
    
    # Run basic simulation
    logger.info("1. Running basic simulation...")
    stats = simulator.run_single_simulation(
        num_bits=1000,
        channel_loss=0.1,
        channel_error=0.02
    )
    
    # Create summary report
    report_path = create_summary_report(stats)
    logger.info(f"Summary report saved: {report_path}")
    
    # Run parameter sweep
    logger.info("2. Running parameter sweep...")
    sweep_results = simulator.run_parameter_sweep(
        num_bits=500,
        loss_rates=[0.0, 0.1, 0.2],
        error_rates=[0.0, 0.05, 0.10],
        trials_per_config=2
    )
    logger.info("Parameter sweep complete")
    
    # Compare backends if Qiskit available
    if QISKIT_AVAILABLE:
        logger.info("3. Comparing backends...")
        comparison = simulator.compare_backends(num_bits=100, trials=3)
        logger.info("Backend comparison complete")
    
    logger.info("SIMULATION COMPLETE")
    logger.info(f"All results saved in: {OUTPUT_DIR}")
    logger.info("Ready for quantum jamming implementation!")
    logger.info("Next step: Extend QuantumChannel class with Eve's intervention")


if __name__ == "__main__":
    main()