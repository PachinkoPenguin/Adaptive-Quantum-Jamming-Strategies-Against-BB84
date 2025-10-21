"""
BB84 Quantum Key Distribution Protocol - Hybrid Foundation Implementation
Author: [Your Name]
Date: 2024
Description: Complete educational implementation with Classical/Qiskit backends
             and image-based output for terminal compatibility
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
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
    def prepare_state(self, bit: int, basis: Basis) -> any:
        """Prepare a quantum state"""
        pass
    
    @abstractmethod
    def measure_state(self, state: any, basis: Basis) -> int:
        """Measure a quantum state"""
        pass
    
    @abstractmethod
    def apply_channel_noise(self, state: any, error_rate: float) -> any:
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
# QUANTUM CHANNEL (Backend-aware)
# ============================================================================

class QuantumChannel:
    """Quantum channel for state transmission"""
    
    def __init__(self, 
                 backend: QuantumBackend,
                 loss_rate: float = 0.0,
                 error_rate: float = 0.0,
                 name: str = "Quantum Channel"):
        self.backend = backend
        self.loss_rate = loss_rate
        self.error_rate = error_rate
        self.name = name
        self.transmitted_count = 0
        self.lost_count = 0
        self.error_count = 0
        
    def transmit(self, states: List) -> List:
        """Transmit quantum states through channel"""
        transmitted_states = []
        
        for state in states:
            # Check for photon loss
            if random.random() < self.loss_rate:
                transmitted_states.append(None)  # State lost
                self.lost_count += 1
            else:
                # Apply channel noise
                if self.error_rate > 0:
                    state = self.backend.apply_channel_noise(state, self.error_rate)
                    self.error_count += 1 if random.random() < self.error_rate else 0
                
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
                 run_type: str = "standard"):
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
        self.channel = QuantumChannel(self.backend, channel_loss, channel_error)
        
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
            'secure': self.qber < 0.11
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