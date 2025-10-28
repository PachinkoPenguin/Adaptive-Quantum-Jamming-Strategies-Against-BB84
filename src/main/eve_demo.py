"""
Demonstration of Eve (Eavesdropper) Integration in BB84 Protocol
Shows how to use the AttackStrategy, EveController, and modified QuantumChannel
"""

from bb84_main import (
    BB84Protocol,
    ClassicalBackend,
    Alice,
    Bob,
    QuantumChannel,
    EveController,
    InterceptResendAttack,
    AdaptiveAttack,
    OUTPUT_DIR
)
import os
from typing import Dict
import matplotlib.pyplot as plt


def demo_intercept_resend():
    """
    Demonstration 1: Classic Intercept-Resend Attack
    Eve intercepts all qubits, measures in random basis, and resends.
    Expected QBER: ~25%
    """
    print("="*70)
    print("DEMO 1: Intercept-Resend Attack (100% interception)")
    print("="*70)
    
    # Setup backend
    backend = ClassicalBackend()
    
    # Create Alice and Bob
    alice = Alice(backend)
    bob = Bob(backend)
    
    # Create Eve with intercept-resend strategy
    eve_strategy = InterceptResendAttack(backend, intercept_probability=1.0)
    eve = EveController(eve_strategy, backend, name="Eve (Intercept-Resend)")
    
    # Create quantum channel WITH Eve
    channel = QuantumChannel(
        backend=backend,
        loss_rate=0.0,  # No natural loss for clean demo
        error_rate=0.0,  # No natural errors
        eve=eve,
        name="Quantum Channel with Eve"
    )
    
    # Run protocol
    num_bits = 500
    print(f"\nAlice generates {num_bits} bits...")
    alice_bits = alice.generate_random_bits(num_bits)
    alice_bases = alice.choose_random_bases(num_bits)
    states = alice.prepare_states()
    
    print("Transmitting through channel (Eve is listening)...")
    received_states = channel.transmit(states)
    
    print("Bob measures received states...")
    bob_bases = bob.choose_random_bases(num_bits)
    bob_bits = bob.measure_states(received_states)
    
    # Basis reconciliation
    print("\nBasis reconciliation (public channel)...")
    alice_bases_pub = alice.announce_bases()
    bob_bases_pub = bob.announce_bases()
    
    # Eve receives public information
    eve.receive_feedback(0.0, {'alice_bases': alice_bases_pub})
    
    # Sift key
    matching_indices = [i for i in range(num_bits) 
                       if alice_bases[i] == bob_bases[i]]
    
    alice_key = [alice_bits[i] for i in matching_indices]
    bob_key = [bob_bits[i] for i in matching_indices]
    
    # Calculate QBER
    errors = sum(1 for i in range(len(alice_key)) if alice_key[i] != bob_key[i])
    qber = errors / len(alice_key) if alice_key else 0
    
    print(f"\nRESULTS:")
    print(f"  Sifted key length: {len(alice_key)} bits")
    print(f"  Errors detected: {errors}")
    print(f"  QBER: {qber:.2%}")
    print(f"  Security threshold: 11%")
    print(f"  Status: {'INSECURE - Eve detected!' if qber > 0.11 else 'Secure'}")
    
    # Eve's statistics
    eve_stats = eve.get_statistics()
    print(f"\nEVE'S STATISTICS:")
    print(f"  Total intercepted: {eve_stats['total_intercepted']}")
    print(f"  Strategy stats: {eve_stats['attack_strategy_stats']}")
    
    return qber, eve_stats


def demo_adaptive_attack():
    """
    Demonstration 2: Adaptive Attack
    Eve adjusts interception rate to stay below detection threshold.
    Target QBER: 8% (below 11% threshold)
    """
    print("\n" + "="*70)
    print("DEMO 2: Adaptive Attack (Target QBER: 8%)")
    print("="*70)
    
    backend = ClassicalBackend()
    
    # Run multiple rounds to show adaptation
    num_rounds = 5
    qber_history = []
    intercept_rate_history = []
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Create participants
        alice = Alice(backend)
        bob = Bob(backend)
        
        # Create Eve with adaptive strategy
        eve_strategy = AdaptiveAttack(
            backend, 
            target_qber=0.08, 
            initial_intercept_rate=0.5 if round_num == 0 else intercept_rate_history[-1]
        )
        eve = EveController(eve_strategy, backend)
        
        # Create channel with Eve
        channel = QuantumChannel(backend, loss_rate=0.0, error_rate=0.0, eve=eve)
        
        # Run protocol
        num_bits = 500
        alice_bits = alice.generate_random_bits(num_bits)
        alice_bases = alice.choose_random_bases(num_bits)
        states = alice.prepare_states()
        
        received_states = channel.transmit(states)
        
        bob_bases = bob.choose_random_bases(num_bits)
        bob_bits = bob.measure_states(received_states)
        
        # Basis reconciliation
        matching_indices = [i for i in range(num_bits) 
                           if alice_bases[i] == bob_bases[i]]
        
        alice_key = [alice_bits[i] for i in matching_indices]
        bob_key = [bob_bits[i] for i in matching_indices]
        
        # Calculate QBER
        errors = sum(1 for i in range(len(alice_key)) if alice_key[i] != bob_key[i])
        qber = errors / len(alice_key) if alice_key else 0
        
        # Eve receives feedback and adapts
        eve.receive_feedback(qber, {'alice_bases': alice.bases})
        
        eve_stats = eve.get_statistics()
        current_intercept_rate = eve_stats['attack_strategy_stats'].get('intercept_rate', 0)
        
        qber_history.append(qber)
        intercept_rate_history.append(current_intercept_rate)
        
        print(f"  QBER: {qber:.2%}")
        print(f"  Intercept rate: {current_intercept_rate:.2%}")
        print(f"  Intercepted: {eve_stats['total_intercepted']} qubits")
        print(f"  Status: {'DETECTED' if qber > 0.11 else 'Undetected'}")
    
    print(f"\nADAPTATION SUMMARY:")
    print(f"  Initial intercept rate: {intercept_rate_history[0]:.2%}")
    print(f"  Final intercept rate: {intercept_rate_history[-1]:.2%}")
    print(f"  Average QBER: {sum(qber_history)/len(qber_history):.2%}")
    print(f"  Stayed below threshold: {all(q < 0.11 for q in qber_history)}")
    
    # Visualize adaptation
    visualize_adaptive_attack(qber_history, intercept_rate_history)
    
    return qber_history, intercept_rate_history


def demo_partial_interception():
    """
    Demonstration 3: Partial Interception
    Eve only intercepts 30% of qubits.
    Expected QBER: ~7.5%
    """
    print("\n" + "="*70)
    print("DEMO 3: Partial Interception (30% of qubits)")
    print("="*70)
    
    backend = ClassicalBackend()
    
    alice = Alice(backend)
    bob = Bob(backend)
    
    # Create Eve with 30% interception rate
    eve_strategy = InterceptResendAttack(backend, intercept_probability=0.3)
    eve = EveController(eve_strategy, backend)
    
    channel = QuantumChannel(backend, loss_rate=0.0, error_rate=0.0, eve=eve)
    
    # Run protocol
    num_bits = 1000
    print(f"\nRunning BB84 with {num_bits} bits...")
    
    alice_bits = alice.generate_random_bits(num_bits)
    alice_bases = alice.choose_random_bases(num_bits)
    states = alice.prepare_states()
    
    received_states = channel.transmit(states)
    
    bob_bases = bob.choose_random_bases(num_bits)
    bob_bits = bob.measure_states(received_states)
    
    # Basis reconciliation
    matching_indices = [i for i in range(num_bits) 
                       if alice_bases[i] == bob_bases[i]]
    
    alice_key = [alice_bits[i] for i in matching_indices]
    bob_key = [bob_bits[i] for i in matching_indices]
    
    # Calculate QBER
    errors = sum(1 for i in range(len(alice_key)) if alice_key[i] != bob_key[i])
    qber = errors / len(alice_key) if alice_key else 0
    
    eve_stats = eve.get_statistics()
    
    print(f"\nRESULTS:")
    print(f"  Sifted key length: {len(alice_key)} bits")
    print(f"  QBER: {qber:.2%}")
    print(f"  Eve intercepted: {eve_stats['total_intercepted']} qubits")
    print(f"  Interception rate: 30%")
    print(f"  Status: {'DETECTED' if qber > 0.11 else 'Undetected'}")
    
    return qber, eve_stats


def visualize_adaptive_attack(qber_history, intercept_rate_history):
    """Create visualization of adaptive attack behavior"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    rounds = list(range(1, len(qber_history) + 1))
    
    # Plot QBER over rounds
    ax1.plot(rounds, [q * 100 for q in qber_history], 'ro-', linewidth=2, markersize=8)
    ax1.axhline(y=11, color='red', linestyle='--', label='Detection Threshold (11%)', linewidth=2)
    ax1.axhline(y=8, color='orange', linestyle='--', label='Target QBER (8%)', linewidth=2)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('QBER (%)', fontsize=12)
    ax1.set_title('Adaptive Attack: QBER Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot interception rate over rounds
    ax2.plot(rounds, [r * 100 for r in intercept_rate_history], 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Interception Rate (%)', fontsize=12)
    ax2.set_title('Adaptive Attack: Interception Rate Evolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "eve_adaptive_attack_demo.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n  Visualization saved: {output_path}")
    plt.close()


def compare_attack_strategies():
    """Compare different attack strategies side-by-side"""
    print("\n" + "="*70)
    print("DEMO 4: Comparing Attack Strategies")
    print("="*70)
    
    backend = ClassicalBackend()
    num_bits = 1000
    
    strategies = {
        'No Attack': None,
        '100% Intercept': InterceptResendAttack(backend, 1.0),
        '50% Intercept': InterceptResendAttack(backend, 0.5),
        '30% Intercept': InterceptResendAttack(backend, 0.3),
        'Adaptive (target 8%)': AdaptiveAttack(backend, target_qber=0.08)
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\nTesting: {strategy_name}")
        
        alice = Alice(backend)
        bob = Bob(backend)
        
        if strategy is not None:
            eve = EveController(strategy, backend)
            channel = QuantumChannel(backend, loss_rate=0.0, error_rate=0.0, eve=eve)
        else:
            channel = QuantumChannel(backend, loss_rate=0.0, error_rate=0.0)
        
        # Run protocol
        alice_bits = alice.generate_random_bits(num_bits)
        alice_bases = alice.choose_random_bases(num_bits)
        states = alice.prepare_states()
        
        received_states = channel.transmit(states)
        
        bob_bases = bob.choose_random_bases(num_bits)
        bob_bits = bob.measure_states(received_states)
        
        # Sift key
        matching_indices = [i for i in range(num_bits) 
                           if alice_bases[i] == bob_bases[i]]
        
        alice_key = [alice_bits[i] for i in matching_indices]
        bob_key = [bob_bits[i] for i in matching_indices]
        
        # Calculate QBER
        errors = sum(1 for i in range(len(alice_key)) if alice_key[i] != bob_key[i])
        qber = errors / len(alice_key) if alice_key else 0
        
        results[strategy_name] = {
            'qber': qber,
            'key_length': len(alice_key),
            'detected': qber > 0.11
        }
        
        print(f"  QBER: {qber:.2%}")
        print(f"  Status: {'DETECTED' if qber > 0.11 else 'Undetected'}")
    
    # Visualize comparison
    visualize_strategy_comparison(results)
    
    return results


def visualize_strategy_comparison(results: Dict):
    """Create bar chart comparing strategies"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strategies = list(results.keys())
    qbers = [results[s]['qber'] * 100 for s in strategies]
    colors = ['green' if results[s]['detected'] == False else 'red' for s in strategies]
    
    bars = ax.bar(strategies, qbers, color=colors, alpha=0.7, edgecolor='black')
    
    # Add threshold line
    ax.axhline(y=11, color='red', linestyle='--', linewidth=2, label='Detection Threshold (11%)')
    
    ax.set_ylabel('QBER (%)', fontsize=12)
    ax.set_title('Attack Strategy Comparison: QBER Results', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    
    # Rotate x labels for readability
    plt.xticks(rotation=15, ha='right')
    
    # Add value labels on bars
    for bar, qber in zip(bars, qbers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{qber:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "eve_strategy_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n  Comparison visualization saved: {output_path}")
    plt.close()


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("EVE (EAVESDROPPER) DEMONSTRATION FOR BB84 PROTOCOL")
    print("="*70)
    print("\nThis demo shows how to integrate quantum eavesdropping attacks")
    print("into the BB84 quantum key distribution protocol.\n")
    
    # Demo 1: Full interception
    demo_intercept_resend()
    
    # Demo 2: Adaptive attack
    demo_adaptive_attack()
    
    # Demo 3: Partial interception
    demo_partial_interception()
    
    # Demo 4: Strategy comparison
    compare_attack_strategies()
    
    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("="*70)
    print(f"\nVisualizations saved in: {OUTPUT_DIR}/")
    print("\nKey Findings:")
    print("  • 100% interception causes ~25% QBER (easily detected)")
    print("  • Adaptive attacks can stay below 11% threshold")
    print("  • Partial interception reduces detectability")
    print("  • QBER threshold of 11% is critical for security")


if __name__ == "__main__":
    main()
