"""
BB84 Quantum Key Distribution Protocol - Foundation Implementation
Author: Adair Virgilio Figueroa Medina
Date: 2025-10-01
Description: Complete educational implementation of BB84 protocol with detailed comments
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================================
# FUNDAMENTAL QUANTUM COMPONENTS
# ============================================================================


class Basis(Enum):
    """Measurement basis for quantum states"""

    RECTILINEAR = 0  # + basis: {|0⟩, |1⟩} = {|H⟩, |V⟩}
    DIAGONAL = 1  # × basis: {|+⟩, |−⟩} = {|45°⟩, |135°⟩}

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
    basis: Basis  # Preparation basis
    polarization: float  # Angle in degrees

    def __init__(self, bit: int, basis: Basis):
        self.bit_value = bit
        self.basis = basis

        # Set polarization based on bit and basis
        if basis == Basis.RECTILINEAR:
            self.polarization = 0.0 if bit == 0 else 90.0  # H or V
        else:  # DIAGONAL
            self.polarization = 45.0 if bit == 0 else 135.0  # + or -

    def measure(self, measurement_basis: Basis) -> int:
        """
        Measure photon in given basis
        Returns measured bit value
        """
        if measurement_basis == self.basis:
            # Same basis - perfect correlation
            return self.bit_value
        else:
            # Different basis - random result (50% probability)
            return random.randint(0, 1)


# ============================================================================
# ALICE - QUANTUM TRANSMITTER
# ============================================================================


class Alice:
    """
    Alice: The sender in QKD protocol
    - Generates random bits
    - Chooses random bases
    - Prepares and sends photons
    """

    def __init__(self, name: str = "Alice"):
        self.name = name
        self.bits: List[int] = []
        self.bases: List[Basis] = []
        self.sent_photons: List[Photon] = []

    def generate_random_bits(self, n: int) -> List[int]:
        """Generate n random bits for transmission"""
        self.bits = [random.randint(0, 1) for _ in range(n)]
        print(f"{self.name}: Generated {n} random bits")
        return self.bits

    def choose_random_bases(self, n: int) -> List[Basis]:
        """Choose random basis for each bit"""
        self.bases = [Basis.random() for _ in range(n)]
        print(f"{self.name}: Chose {n} random bases")
        return self.bases

    def prepare_photons(self) -> List[Photon]:
        """
        Prepare photons based on bits and bases
        This simulates quantum state preparation
        """
        assert len(self.bits) == len(self.bases), "Bits and bases must have same length"

        self.sent_photons = []
        for bit, basis in zip(self.bits, self.bases):
            photon = Photon(bit, basis)
            self.sent_photons.append(photon)

        print(f"{self.name}: Prepared {len(self.sent_photons)} photons")
        return self.sent_photons

    def announce_bases(self) -> List[Basis]:
        """Publicly announce bases used (classical channel)"""
        print(f"{self.name}: Announcing bases publicly")
        return self.bases


# ============================================================================
# BOB - QUANTUM RECEIVER
# ============================================================================


class Bob:
    """
    Bob: The receiver in QKD protocol
    - Chooses random measurement bases
    - Measures received photons
    - Records measurement results
    """

    def __init__(self, name: str = "Bob"):
        self.name = name
        self.bases: List[Basis] = []
        self.measured_bits: List[int] = []
        self.received_photons: List[Photon] = []

    def choose_random_bases(self, n: int) -> List[Basis]:
        """Choose random measurement basis for each photon"""
        self.bases = [Basis.random() for _ in range(n)]
        print(f"{self.name}: Chose {n} random measurement bases")
        return self.bases

    def measure_photons(self, photons: List[Photon]) -> List[int]:
        """
        Measure received photons in chosen bases
        Simulates quantum measurement with basis-dependent outcomes
        """
        assert len(photons) == len(self.bases), "Number of photons and bases must match"

        self.received_photons = photons
        self.measured_bits = []

        for photon, basis in zip(photons, self.bases):
            measured_bit = photon.measure(basis)
            self.measured_bits.append(measured_bit)

        print(f"{self.name}: Measured {len(self.measured_bits)} photons")
        return self.measured_bits

    def announce_bases(self) -> List[Basis]:
        """Publicly announce measurement bases (classical channel)"""
        print(f"{self.name}: Announcing measurement bases publicly")
        return self.bases


# ============================================================================
# QUANTUM CHANNEL
# ============================================================================


class QuantumChannel:
    """
    Quantum channel for photon transmission
    Can introduce errors, losses, and noise
    """

    def __init__(
        self,
        loss_rate: float = 0.0,
        error_rate: float = 0.0,
        name: str = "Quantum Channel",
    ):
        """
        Initialize quantum channel with imperfections

        Args:
            loss_rate: Probability of photon loss (0-1)
            error_rate: Probability of bit flip error (0-1)
        """
        self.loss_rate = loss_rate
        self.error_rate = error_rate
        self.name = name
        self.transmitted_count = 0
        self.lost_count = 0
        self.error_count = 0

    def transmit(self, photons: List[Photon]) -> List[Optional[Photon]]:
        """
        Transmit photons through channel
        May introduce losses and errors
        """
        transmitted_photons = []

        for photon in photons:
            # Check for photon loss
            if random.random() < self.loss_rate:
                transmitted_photons.append(None)  # Photon lost
                self.lost_count += 1
            else:
                # Check for bit flip error
                if random.random() < self.error_rate:
                    # Flip the bit value (error)
                    photon.bit_value = 1 - photon.bit_value
                    self.error_count += 1

                transmitted_photons.append(photon)
                self.transmitted_count += 1

        print(
            f"{self.name}: Transmitted {self.transmitted_count}/{len(photons)} photons"
        )
        print(f"  - Lost: {self.lost_count}, Errors: {self.error_count}")

        return transmitted_photons


# ============================================================================
# BB84 PROTOCOL IMPLEMENTATION
# ============================================================================


class BB84Protocol:
    """
    Complete BB84 Quantum Key Distribution Protocol
    Coordinates Alice, Bob, and the quantum channel
    """

    def __init__(
        self, alice: Alice, bob: Bob, channel: QuantumChannel, verbose: bool = True
    ):
        """
        Initialize BB84 protocol

        Args:
            alice: Sender object
            bob: Receiver object
            channel: Quantum channel for transmission
            verbose: Print detailed progress
        """
        self.alice = alice
        self.bob = bob
        self.channel = channel
        self.verbose = verbose

        # Protocol results
        self.raw_key_alice: List[int] = []
        self.raw_key_bob: List[int] = []
        self.final_key: List[int] = []
        self.qber: float = 0.0

    def run_protocol(self, num_bits: int = 1000) -> Dict:
        """
        Execute complete BB84 protocol

        Args:
            num_bits: Number of bits to transmit

        Returns:
            Dictionary with protocol results and statistics
        """
        print("=" * 60)
        print("STARTING BB84 PROTOCOL")
        print("=" * 60)

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
        stats = self._calculate_statistics()

        print("\n" + "=" * 60)
        print("BB84 PROTOCOL COMPLETED")
        print("=" * 60)

        return stats

    def _quantum_transmission_phase(self, num_bits: int):
        """Phase 1: Quantum transmission of photons"""
        print("\n--- PHASE 1: QUANTUM TRANSMISSION ---")

        # Alice prepares and sends
        alice_bits = self.alice.generate_random_bits(num_bits)
        alice_bases = self.alice.choose_random_bases(num_bits)
        photons = self.alice.prepare_photons()

        # Transmission through quantum channel
        received_photons = self.channel.transmit(photons)

        # Filter out lost photons
        valid_photons = [p for p in received_photons if p is not None]
        valid_indices = [i for i, p in enumerate(received_photons) if p is not None]

        # Bob measures
        bob_bases = self.bob.choose_random_bases(len(valid_photons))
        bob_bits = self.bob.measure_photons(valid_photons)

        # Store valid indices for later use
        self.valid_indices = valid_indices

    def _basis_reconciliation_phase(self):
        """Phase 2: Public comparison of bases (sifting)"""
        print("\n--- PHASE 2: BASIS RECONCILIATION (SIFTING) ---")

        # Get bases for valid (non-lost) photons
        alice_bases_valid = [self.alice.bases[i] for i in self.valid_indices]
        bob_bases = self.bob.bases

        # Find matching bases
        matching_bases = []
        for i, (a_basis, b_basis) in enumerate(zip(alice_bases_valid, bob_bases)):
            if a_basis == b_basis:
                matching_bases.append(i)

        # Extract raw keys (only matching bases)
        self.raw_key_alice = [
            self.alice.bits[self.valid_indices[i]] for i in matching_bases
        ]
        self.raw_key_bob = [self.bob.measured_bits[i] for i in matching_bases]

        print(f"Matching bases: {len(matching_bases)}/{len(alice_bases_valid)}")
        print(f"Sifting efficiency: {len(matching_bases) / len(self.alice.bits):.2%}")
        print(f"Raw key length: {len(self.raw_key_alice)} bits")

    def _error_estimation_phase(self):
        """Phase 3: Estimate quantum bit error rate (QBER)"""
        print("\n--- PHASE 3: ERROR ESTIMATION ---")

        if len(self.raw_key_alice) == 0:
            print("No matching bases found!")
            self.qber = 1.0
            return

        # Sample subset for error estimation (typically 10-20%)
        sample_size = max(1, len(self.raw_key_alice) // 5)
        sample_indices = random.sample(range(len(self.raw_key_alice)), sample_size)

        # Count errors in sample
        errors = 0
        for idx in sample_indices:
            if self.raw_key_alice[idx] != self.raw_key_bob[idx]:
                errors += 1

        # Calculate QBER
        self.qber = errors / sample_size if sample_size > 0 else 0
        print(f"Sampled {sample_size} bits for error estimation")
        print(f"Found {errors} errors")
        print(f"QBER = {self.qber:.2%}")

        # Security check
        if self.qber > 0.11:
            print("WARNING: QBER > 11% - Protocol may be insecure!")

        # Remove sampled bits from keys
        for idx in sorted(sample_indices, reverse=True):
            del self.raw_key_alice[idx]
            del self.raw_key_bob[idx]

    def _error_correction_phase(self):
        """Phase 4: Error correction (simplified CASCADE)"""
        print("\n--- PHASE 4: ERROR CORRECTION ---")

        # Simplified: Just identify and remove error positions
        # In practice, would use CASCADE, LDPC, or Turbo codes

        error_positions = []
        for i in range(len(self.raw_key_alice)):
            if self.raw_key_alice[i] != self.raw_key_bob[i]:
                error_positions.append(i)

        print(f"Found {len(error_positions)} errors in remaining key")

        # For simulation: assume perfect error correction
        # Bob corrects his key to match Alice
        for i in error_positions:
            self.raw_key_bob[i] = self.raw_key_alice[i]

        print("Error correction completed (simplified)")

    def _privacy_amplification_phase(self):
        """Phase 5: Privacy amplification (simplified)"""
        print("\n--- PHASE 5: PRIVACY AMPLIFICATION ---")

        # Calculate how much key to keep based on QBER
        # Simplified version - in practice use universal hashing

        if self.qber == 0:
            compression_ratio = 0.9  # Keep 90% if no errors
        else:
            # More conservative with higher QBER
            compression_ratio = max(0.1, 1 - 3 * self.qber)

        final_length = int(len(self.raw_key_alice) * compression_ratio)

        # Simple privacy amplification: XOR adjacent bits
        self.final_key = []
        for i in range(0, final_length * 2, 2):
            if i + 1 < len(self.raw_key_alice):
                # XOR adjacent bits
                new_bit = self.raw_key_alice[i] ^ self.raw_key_alice[i + 1]
                self.final_key.append(new_bit)

        print(f"Compression ratio: {compression_ratio:.2%}")
        print(f"Final secure key length: {len(self.final_key)} bits")

    def _calculate_statistics(self) -> Dict:
        """Calculate protocol statistics"""
        stats = {
            "total_bits_sent": len(self.alice.bits),
            "raw_key_length": len(self.raw_key_alice) + len(self.final_key),
            "final_key_length": len(self.final_key),
            "qber": self.qber,
            "efficiency": len(self.final_key) / len(self.alice.bits)
            if self.alice.bits
            else 0,
            "channel_loss_rate": self.channel.loss_rate,
            "channel_error_rate": self.channel.error_rate,
        }

        return stats

    def visualize_results(self):
        """Create visualization of protocol results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Key generation stages
        stages = ["Sent", "Received", "Sifted", "Final"]
        counts = [
            len(self.alice.bits),
            self.channel.transmitted_count,
            len(self.raw_key_alice),
            len(self.final_key),
        ]
        axes[0, 0].bar(stages, counts, color=["blue", "green", "orange", "red"])
        axes[0, 0].set_ylabel("Number of Bits")
        axes[0, 0].set_title("Key Generation Pipeline")

        # Plot 2: Basis distribution
        alice_basis_counts = Counter([str(b) for b in self.alice.bases])
        bob_basis_counts = Counter([str(b) for b in self.bob.bases])

        x = np.arange(2)
        width = 0.35
        axes[0, 1].bar(x - width / 2, alice_basis_counts.values(), width, label="Alice")
        axes[0, 1].bar(x + width / 2, bob_basis_counts.values(), width, label="Bob")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(["+", "×"])
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Basis Selection Distribution")
        axes[0, 1].legend()

        # Plot 3: Error analysis
        if len(self.raw_key_alice) > 0:
            sample_size = min(100, len(self.raw_key_alice))
            alice_sample = self.raw_key_alice[:sample_size]
            bob_sample = self.raw_key_bob[:sample_size]

            axes[1, 0].plot(alice_sample, "b-", label="Alice", alpha=0.7)
            axes[1, 0].plot(bob_sample, "r--", label="Bob", alpha=0.7)
            axes[1, 0].set_xlabel("Bit Position")
            axes[1, 0].set_ylabel("Bit Value")
            axes[1, 0].set_title(f"First {sample_size} Raw Key Bits")
            axes[1, 0].legend()
            axes[1, 0].set_ylim(-0.1, 1.1)

        # Plot 4: Protocol efficiency
        metrics = ["QBER", "Efficiency", "Loss Rate", "Error Rate"]
        values = [
            self.qber * 100,
            (len(self.final_key) / len(self.alice.bits)) * 100,
            self.channel.loss_rate * 100,
            self.channel.error_rate * 100,
        ]
        axes[1, 1].bar(metrics, values, color=["red", "green", "orange", "yellow"])
        axes[1, 1].set_ylabel("Percentage (%)")
        axes[1, 1].set_title("Protocol Metrics")

        plt.suptitle("BB84 Protocol Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig("output.png")


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================


def run_bb84_simulation(
    num_bits: int = 1000,
    channel_loss: float = 0.1,
    channel_error: float = 0.02,
    visualize: bool = True,
) -> Dict:
    """
    Run a complete BB84 simulation

    Args:
        num_bits: Number of bits to transmit
        channel_loss: Probability of photon loss (0-1)
        channel_error: Probability of bit flip (0-1)
        visualize: Show visualization plots

    Returns:
        Statistics dictionary
    """

    print(f"\nBB84 SIMULATION PARAMETERS:")
    print(f"  - Bits to transmit: {num_bits}")
    print(f"  - Channel loss rate: {channel_loss:.1%}")
    print(f"  - Channel error rate: {channel_error:.1%}")
    print("-" * 60)

    # Create participants
    alice = Alice()
    bob = Bob()
    channel = QuantumChannel(loss_rate=channel_loss, error_rate=channel_error)

    # Create and run protocol
    protocol = BB84Protocol(alice, bob, channel)
    stats = protocol.run_protocol(num_bits)

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total bits sent: {stats['total_bits_sent']}")
    print(f"Final key length: {stats['final_key_length']}")
    print(f"Overall efficiency: {stats['efficiency']:.2%}")
    print(f"Measured QBER: {stats['qber']:.2%}")
    print("-" * 60)

    # Security assessment
    if stats["qber"] < 0.11:
        print("✓ Protocol SECURE (QBER < 11%)")
    else:
        print("✗ Protocol INSECURE (QBER ≥ 11%)")

    if visualize:
        protocol.visualize_results()

    return stats


def test_different_conditions():
    """Test BB84 under various channel conditions"""

    print("\n" + "=" * 60)
    print("TESTING BB84 UNDER DIFFERENT CONDITIONS")
    print("=" * 60)

    conditions = [
        ("Ideal Channel", 0.0, 0.0),
        ("Low Noise", 0.05, 0.01),
        ("Moderate Noise", 0.1, 0.03),
        ("High Noise", 0.2, 0.05),
        ("Very High Noise", 0.3, 0.08),
        ("Near Threshold", 0.15, 0.10),
    ]

    results = []
    for name, loss, error in conditions:
        print(f"\nTesting: {name}")
        print("-" * 40)

        # Run simulation
        alice = Alice()
        bob = Bob()
        channel = QuantumChannel(loss_rate=loss, error_rate=error)
        protocol = BB84Protocol(alice, bob, channel, verbose=False)
        stats = protocol.run_protocol(1000)

        results.append(
            {
                "condition": name,
                "loss_rate": loss,
                "error_rate": error,
                "qber": stats["qber"],
                "efficiency": stats["efficiency"],
                "secure": stats["qber"] < 0.11,
            }
        )

        print(f"  QBER: {stats['qber']:.2%}")
        print(f"  Efficiency: {stats['efficiency']:.2%}")
        print(f"  Secure: {'Yes' if stats['qber'] < 0.11 else 'No'}")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run single simulation with visualization
    print("\n" + "=" * 60)
    print("BB84 QUANTUM KEY DISTRIBUTION - FOUNDATION")
    print("=" * 60)

    # Basic simulation
    stats = run_bb84_simulation(
        num_bits=1000, channel_loss=0.1, channel_error=0.02, visualize=True
    )

    # Test different conditions
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE TESTS")
    print("=" * 60)
    test_results = test_different_conditions()

    # Summary table
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    print(
        f"{'Condition':<20} {'Loss':<10} {'Error':<10} {'QBER':<10} {'Efficiency':<12} {'Secure':<10}"
    )
    print("-" * 80)
    for r in test_results:
        print(
            f"{r['condition']:<20} {r['loss_rate']:<10.1%} {r['error_rate']:<10.1%} "
            f"{r['qber']:<10.2%} {r['efficiency']:<12.2%} "
            f"{'✓' if r['secure'] else '✗':<10}"
        )

    print("\n" + "=" * 60)
    print("BB84 FOUNDATION COMPLETE")
    print("=" * 60)
