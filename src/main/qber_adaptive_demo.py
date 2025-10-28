"""
Demonstration of QBER-Adaptive Attack Strategies
Shows PID control and gradient descent approaches to maintain QBER below detection threshold
"""

from bb84_main import (
    ClassicalBackend,
    Alice,
    Bob,
    QuantumChannel,
    EveController,
    QBERAdaptiveStrategy,
    GradientDescentQBERAdaptive,
    InterceptResendAttack,
    OUTPUT_DIR
)
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def demo_pid_controller():
    """
    Demonstration 1: PID Controller QBER-Adaptive Strategy
    Shows how PID control maintains QBER at target value (10%) below threshold (11%)
    """
    print("="*70)
    print("DEMO 1: PID Controller QBER-Adaptive Strategy")
    print("="*70)
    print("\nObjective: Maintain QBER at 10% (below 11% detection threshold)")
    print("Method: PID control with Kp=2.0, Ki=0.5, Kd=0.1\n")
    
    backend = ClassicalBackend()
    
    # Run multiple rounds to show convergence
    num_rounds = 15
    qber_history = []
    intercept_prob_history = []
    
    # Create strategy with PID controller
    eve_strategy = QBERAdaptiveStrategy(
        backend,
        target_qber=0.10,  # Target: 10% QBER
        threshold=0.11,     # Detection: 11% QBER
        kp=2.0,            # Proportional gain
        ki=0.5,            # Integral gain
        kd=0.1             # Derivative gain
    )
    eve = EveController(eve_strategy, backend, name="Eve-PID")
    
    print(f"Initial intercept probability: {eve_strategy.intercept_probability:.3f}")
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Create fresh participants
        alice = Alice(backend)
        bob = Bob(backend)
        
        # Create channel with Eve
        channel = QuantumChannel(backend, loss_rate=0.0, error_rate=0.0, eve=eve)
        
        # Run BB84 protocol
        num_bits = 1000
        alice.generate_random_bits(num_bits)
        alice.choose_random_bases(num_bits)
        states = alice.prepare_states()
        
        received_states = channel.transmit(states)
        
        bob.choose_random_bases(num_bits)
        bob.measure_states(received_states)
        
        # Basis reconciliation
        matching_indices = [i for i in range(num_bits) 
                           if alice.bases[i] == bob.bases[i]]
        
        alice_key = [alice.bits[i] for i in matching_indices]
        bob_key = [bob.measured_bits[i] for i in matching_indices]
        
        # Calculate QBER
        errors = sum(1 for i in range(len(alice_key)) if alice_key[i] != bob_key[i])
        qber = errors / len(alice_key) if alice_key else 0
        
        # Eve receives feedback and adapts using PID
        eve.receive_feedback(qber, {'alice_bases': alice.bases})
        
        eve_stats = eve.get_statistics()
        current_prob = eve_stats['attack_strategy_stats']['intercept_probability']
        error_integral = eve_stats['attack_strategy_stats']['error_integral']
        
        qber_history.append(qber)
        intercept_prob_history.append(current_prob)
        
        # Display results
        status = "✓ UNDETECTED" if qber < 0.11 else "✗ DETECTED"
        print(f"  QBER: {qber:.4f} ({qber*100:.2f}%)")
        print(f"  Intercept Probability: {current_prob:.4f}")
        print(f"  Error Integral: {error_integral:.4f}")
        print(f"  Status: {status}")
    
    # Summary
    print(f"\n{'='*70}")
    print("PID CONTROLLER SUMMARY")
    print(f"{'='*70}")
    print(f"Target QBER: {eve_strategy.target_qber:.2%}")
    print(f"Detection Threshold: {eve_strategy.threshold:.2%}")
    print(f"Final QBER: {qber_history[-1]:.4f}")
    print(f"Average QBER: {np.mean(qber_history):.4f}")
    print(f"QBER Std Dev: {np.std(qber_history):.4f}")
    print(f"Stayed below threshold: {all(q < 0.11 for q in qber_history)}")
    print(f"Final intercept probability: {intercept_prob_history[-1]:.4f}")
    
    # Visualize
    visualize_pid_results(qber_history, intercept_prob_history, eve_strategy.target_qber)
    
    return qber_history, intercept_prob_history


def demo_gradient_descent():
    """
    Demonstration 2: Gradient Descent QBER-Adaptive Strategy
    Shows gradient descent optimization to maintain QBER at target
    """
    print("\n" + "="*70)
    print("DEMO 2: Gradient Descent QBER-Adaptive Strategy")
    print("="*70)
    print("\nObjective: Maintain QBER at 10% using gradient descent")
    print("Method: Gradient descent with learning_rate=0.01\n")
    print("Loss Function: L = (QBER - target)²")
    print("Gradient: ∂L/∂p ≈ 2(QBER - target) × 0.25\n")
    
    backend = ClassicalBackend()
    
    num_rounds = 15
    qber_history = []
    intercept_prob_history = []
    loss_history = []
    
    # Create gradient descent strategy
    eve_strategy = GradientDescentQBERAdaptive(
        backend,
        target_qber=0.10,
        threshold=0.11,
        learning_rate=0.01
    )
    eve = EveController(eve_strategy, backend, name="Eve-GradDesc")
    
    print(f"Initial intercept probability: {eve_strategy.intercept_probability:.3f}")
    print(f"Learning rate: {eve_strategy.learning_rate}")
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Create fresh participants
        alice = Alice(backend)
        bob = Bob(backend)
        
        channel = QuantumChannel(backend, loss_rate=0.0, error_rate=0.0, eve=eve)
        
        # Run BB84
        num_bits = 1000
        alice.generate_random_bits(num_bits)
        alice.choose_random_bases(num_bits)
        states = alice.prepare_states()
        
        received_states = channel.transmit(states)
        
        bob.choose_random_bases(num_bits)
        bob.measure_states(received_states)
        
        # Calculate QBER
        matching_indices = [i for i in range(num_bits) 
                           if alice.bases[i] == bob.bases[i]]
        alice_key = [alice.bits[i] for i in matching_indices]
        bob_key = [bob.measured_bits[i] for i in matching_indices]
        
        errors = sum(1 for i in range(len(alice_key)) if alice_key[i] != bob_key[i])
        qber = errors / len(alice_key) if alice_key else 0
        
        # Calculate loss before update
        loss = (qber - eve_strategy.target_qber) ** 2
        
        # Eve adapts using gradient descent
        eve.receive_feedback(qber, {'alice_bases': alice.bases})
        
        eve_stats = eve.get_statistics()
        current_prob = eve_stats['attack_strategy_stats']['intercept_probability']
        
        qber_history.append(qber)
        intercept_prob_history.append(current_prob)
        loss_history.append(loss)
        
        status = "✓ UNDETECTED" if qber < 0.11 else "✗ DETECTED"
        print(f"  QBER: {qber:.4f} ({qber*100:.2f}%)")
        print(f"  Loss: {loss:.6f}")
        print(f"  Intercept Probability: {current_prob:.4f}")
        print(f"  Status: {status}")
    
    # Summary
    print(f"\n{'='*70}")
    print("GRADIENT DESCENT SUMMARY")
    print(f"{'='*70}")
    print(f"Target QBER: {eve_strategy.target_qber:.2%}")
    print(f"Final QBER: {qber_history[-1]:.4f}")
    print(f"Average QBER: {np.mean(qber_history):.4f}")
    print(f"Final Loss: {loss_history[-1]:.6f}")
    print(f"Stayed below threshold: {all(q < 0.11 for q in qber_history)}")
    
    # Visualize
    visualize_gradient_descent_results(qber_history, intercept_prob_history, 
                                       loss_history, eve_strategy.target_qber)
    
    return qber_history, intercept_prob_history, loss_history


def compare_adaptive_strategies():
    """
    Demonstration 3: Compare PID vs Gradient Descent vs Simple Adaptive
    Shows performance comparison of different adaptive strategies
    """
    print("\n" + "="*70)
    print("DEMO 3: Comparing Adaptive Strategies")
    print("="*70)
    
    backend = ClassicalBackend()
    num_rounds = 15
    num_bits = 1000
    
    strategies = {
        'PID Control': QBERAdaptiveStrategy(backend, target_qber=0.10),
        'Gradient Descent': GradientDescentQBERAdaptive(backend, target_qber=0.10),
        'Simple Adaptive': InterceptResendAttack(backend, intercept_probability=0.4)
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\n--- Testing: {strategy_name} ---")
        
        eve = EveController(strategy, backend)
        qber_history = []
        prob_history = []
        
        for round_num in range(num_rounds):
            alice = Alice(backend)
            bob = Bob(backend)
            channel = QuantumChannel(backend, loss_rate=0.0, error_rate=0.0, eve=eve)
            
            # Run BB84
            alice.generate_random_bits(num_bits)
            alice.choose_random_bases(num_bits)
            states = alice.prepare_states()
            
            received_states = channel.transmit(states)
            
            bob.choose_random_bases(num_bits)
            bob.measure_states(received_states)
            
            # Calculate QBER
            matching = [i for i in range(num_bits) if alice.bases[i] == bob.bases[i]]
            alice_key = [alice.bits[i] for i in matching]
            bob_key = [bob.measured_bits[i] for i in matching]
            
            errors = sum(1 for i in range(len(alice_key)) if alice_key[i] != bob_key[i])
            qber = errors / len(alice_key) if alice_key else 0
            
            eve.receive_feedback(qber, {'alice_bases': alice.bases})
            
            eve_stats = eve.get_statistics()
            current_prob = eve_stats['attack_strategy_stats'].get('intercept_probability', 
                          eve_stats['attack_strategy_stats'].get('intercept_rate', 0))
            
            qber_history.append(qber)
            prob_history.append(current_prob)
        
        # Store results
        results[strategy_name] = {
            'qber_history': qber_history,
            'prob_history': prob_history,
            'avg_qber': np.mean(qber_history),
            'std_qber': np.std(qber_history),
            'final_qber': qber_history[-1],
            'always_undetected': all(q < 0.11 for q in qber_history)
        }
        
        print(f"  Average QBER: {results[strategy_name]['avg_qber']:.4f}")
        print(f"  Std Dev: {results[strategy_name]['std_qber']:.4f}")
        print(f"  Final QBER: {results[strategy_name]['final_qber']:.4f}")
        print(f"  Always Undetected: {results[strategy_name]['always_undetected']}")
    
    # Visualize comparison
    visualize_strategy_comparison(results)
    
    return results


def analyze_pid_gains():
    """
    Demonstration 4: Analyze effect of different PID gains
    Shows how Kp, Ki, Kd affect convergence behavior
    """
    print("\n" + "="*70)
    print("DEMO 4: PID Gain Analysis")
    print("="*70)
    
    backend = ClassicalBackend()
    num_rounds = 12
    num_bits = 1000
    
    # Test different PID gain configurations
    pid_configs = {
        'High Kp (aggressive)': {'kp': 4.0, 'ki': 0.5, 'kd': 0.1},
        'Balanced (default)': {'kp': 2.0, 'ki': 0.5, 'kd': 0.1},
        'High Ki (integral)': {'kp': 2.0, 'ki': 1.0, 'kd': 0.1},
        'High Kd (derivative)': {'kp': 2.0, 'ki': 0.5, 'kd': 0.5},
        'Conservative': {'kp': 1.0, 'ki': 0.2, 'kd': 0.05}
    }
    
    results = {}
    
    for config_name, gains in pid_configs.items():
        print(f"\n--- Testing: {config_name} ---")
        print(f"    Kp={gains['kp']}, Ki={gains['ki']}, Kd={gains['kd']}")
        
        strategy = QBERAdaptiveStrategy(
            backend,
            target_qber=0.10,
            kp=gains['kp'],
            ki=gains['ki'],
            kd=gains['kd']
        )
        eve = EveController(strategy, backend)
        
        qber_history = []
        
        for round_num in range(num_rounds):
            alice = Alice(backend)
            bob = Bob(backend)
            channel = QuantumChannel(backend, loss_rate=0.0, error_rate=0.0, eve=eve)
            
            alice.generate_random_bits(num_bits)
            alice.choose_random_bases(num_bits)
            states = alice.prepare_states()
            received_states = channel.transmit(states)
            
            bob.choose_random_bases(num_bits)
            bob.measure_states(received_states)
            
            matching = [i for i in range(num_bits) if alice.bases[i] == bob.bases[i]]
            alice_key = [alice.bits[i] for i in matching]
            bob_key = [bob.measured_bits[i] for i in matching]
            
            errors = sum(1 for i in range(len(alice_key)) if alice_key[i] != bob_key[i])
            qber = errors / len(alice_key) if alice_key else 0
            
            eve.receive_feedback(qber, {'alice_bases': alice.bases})
            qber_history.append(qber)
        
        results[config_name] = {
            'qber_history': qber_history,
            'gains': gains,
            'avg_qber': np.mean(qber_history[-5:]),  # Average of last 5 rounds
            'convergence_round': next((i for i, q in enumerate(qber_history) 
                                      if abs(q - 0.10) < 0.01), num_rounds)
        }
        
        print(f"    Average QBER (last 5): {results[config_name]['avg_qber']:.4f}")
        print(f"    Convergence round: {results[config_name]['convergence_round'] + 1}")
    
    # Visualize
    visualize_pid_gain_analysis(results)
    
    return results


def visualize_pid_results(qber_history: List[float], 
                          prob_history: List[float],
                          target_qber: float):
    """Create visualization for PID controller results"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    rounds = list(range(1, len(qber_history) + 1))
    
    # Plot 1: QBER evolution
    ax1.plot(rounds, [q * 100 for q in qber_history], 'b-o', linewidth=2, markersize=6, label='Actual QBER')
    ax1.axhline(y=11, color='red', linestyle='--', linewidth=2, label='Detection Threshold (11%)')
    ax1.axhline(y=target_qber * 100, color='green', linestyle='--', linewidth=2, label=f'Target ({target_qber*100:.0f}%)')
    ax1.fill_between(rounds, 0, 11, alpha=0.1, color='green', label='Safe Zone')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('QBER (%)', fontsize=12)
    ax1.set_title('PID Controller: QBER Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Intercept probability evolution
    ax2.plot(rounds, prob_history, 'g-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Intercept Probability', fontsize=12)
    ax2.set_title('PID Controller: Intercept Probability Adaptation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Error from target
    errors = [(q - target_qber) * 100 for q in qber_history]
    ax3.plot(rounds, errors, 'r-o', linewidth=2, markersize=6)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.fill_between(rounds, 0, errors, where=[e > 0 for e in errors], 
                     alpha=0.3, color='red', label='Above Target')
    ax3.fill_between(rounds, 0, errors, where=[e < 0 for e in errors], 
                     alpha=0.3, color='blue', label='Below Target')
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Error from Target (%)', fontsize=12)
    ax3.set_title('PID Controller: Error Evolution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "pid_controller_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved: {output_path}")
    plt.close()


def visualize_gradient_descent_results(qber_history: List[float],
                                       prob_history: List[float],
                                       loss_history: List[float],
                                       target_qber: float):
    """Create visualization for gradient descent results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    rounds = list(range(1, len(qber_history) + 1))
    
    # Plot 1: QBER evolution
    ax1.plot(rounds, [q * 100 for q in qber_history], 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=11, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax1.axhline(y=target_qber * 100, color='green', linestyle='--', linewidth=2, label='Target')
    ax1.fill_between(rounds, 0, 11, alpha=0.1, color='green')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('QBER (%)', fontsize=12)
    ax1.set_title('Gradient Descent: QBER Evolution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Loss evolution
    ax2.plot(rounds, loss_history, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Loss: (QBER - target)²', fontsize=12)
    ax2.set_title('Gradient Descent: Loss Function', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Intercept probability
    ax3.plot(rounds, prob_history, 'g-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Intercept Probability', fontsize=12)
    ax3.set_title('Gradient Descent: Probability Adaptation', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Gradient visualization
    gradients = [2 * (qber - target_qber) * 0.25 for qber in qber_history]
    ax4.plot(rounds, gradients, 'm-o', linewidth=2, markersize=6)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Gradient ∂L/∂p', fontsize=12)
    ax4.set_title('Gradient Descent: Gradient Values', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "gradient_descent_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved: {output_path}")
    plt.close()


def visualize_strategy_comparison(results: Dict):
    """Create comparison visualization for different strategies"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: QBER trajectories
    for strategy_name, data in results.items():
        rounds = list(range(1, len(data['qber_history']) + 1))
        ax1.plot(rounds, [q * 100 for q in data['qber_history']], 
                marker='o', linewidth=2, label=strategy_name)
    
    ax1.axhline(y=11, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax1.axhline(y=10, color='green', linestyle='--', linewidth=1, label='Target')
    ax1.fill_between(rounds, 0, 11, alpha=0.1, color='green')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('QBER (%)', fontsize=12)
    ax1.set_title('QBER Evolution Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Average QBER comparison
    strategies = list(results.keys())
    avg_qbers = [results[s]['avg_qber'] * 100 for s in strategies]
    colors = ['green' if results[s]['always_undetected'] else 'red' for s in strategies]
    
    bars = ax2.bar(strategies, avg_qbers, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=11, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax2.axhline(y=10, color='green', linestyle='--', linewidth=1, label='Target')
    ax2.set_ylabel('Average QBER (%)', fontsize=12)
    ax2.set_title('Average QBER Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Plot 3: Standard deviation
    std_qbers = [results[s]['std_qber'] * 100 for s in strategies]
    ax3.bar(strategies, std_qbers, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('QBER Std Dev (%)', fontsize=12)
    ax3.set_title('QBER Stability Comparison', fontsize=13, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Plot 4: Probability trajectories
    for strategy_name, data in results.items():
        rounds = list(range(1, len(data['prob_history']) + 1))
        ax4.plot(rounds, data['prob_history'], 
                marker='o', linewidth=2, label=strategy_name)
    
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Intercept Probability', fontsize=12)
    ax4.set_title('Intercept Probability Adaptation', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "strategy_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved: {output_path}")
    plt.close()


def visualize_pid_gain_analysis(results: Dict):
    """Create visualization for PID gain analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: QBER trajectories for different gains
    for config_name, data in results.items():
        rounds = list(range(1, len(data['qber_history']) + 1))
        ax1.plot(rounds, [q * 100 for q in data['qber_history']], 
                marker='o', linewidth=2, label=config_name)
    
    ax1.axhline(y=11, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax1.axhline(y=10, color='green', linestyle='--', linewidth=1, label='Target')
    ax1.fill_between(rounds, 0, 11, alpha=0.1, color='green')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('QBER (%)', fontsize=12)
    ax1.set_title('PID Gain Impact on Convergence', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    # Plot 2: Convergence speed
    configs = list(results.keys())
    convergence_rounds = [results[c]['convergence_round'] + 1 for c in configs]
    avg_qbers = [results[c]['avg_qber'] * 100 for c in configs]
    
    x = np.arange(len(configs))
    width = 0.35
    
    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x - width/2, convergence_rounds, width, label='Convergence Round', 
                    color='skyblue', alpha=0.7, edgecolor='black')
    bars2 = ax2_twin.bar(x + width/2, avg_qbers, width, label='Avg QBER (%)', 
                         color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('PID Configuration', fontsize=12)
    ax2.set_ylabel('Convergence Round', fontsize=12, color='skyblue')
    ax2_twin.set_ylabel('Average QBER (%)', fontsize=12, color='lightcoral')
    ax2.set_title('Convergence Performance', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=15, ha='right', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add legends
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "pid_gain_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved: {output_path}")
    plt.close()


def main():
    """Run all QBER-adaptive demonstrations"""
    print("\n" + "="*70)
    print("QBER-ADAPTIVE ATTACK STRATEGY DEMONSTRATIONS")
    print("="*70)
    print("\nThese demonstrations show advanced eavesdropping strategies that")
    print("maintain QBER just below the 11% detection threshold using:")
    print("  1. PID Control (Proportional-Integral-Derivative)")
    print("  2. Gradient Descent Optimization")
    print("  3. Strategy Comparison")
    print("  4. PID Gain Analysis\n")
    
    # Demo 1: PID Controller
    qber_pid, prob_pid = demo_pid_controller()
    
    # Demo 2: Gradient Descent
    qber_gd, prob_gd, loss_gd = demo_gradient_descent()
    
    # Demo 3: Strategy Comparison
    comparison_results = compare_adaptive_strategies()
    
    # Demo 4: PID Gain Analysis
    gain_results = analyze_pid_gains()
    
    # Final Summary
    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("="*70)
    print(f"\nVisualizations saved in: {OUTPUT_DIR}/")
    print("\nKey Findings:")
    print("  • PID control provides stable convergence to target QBER")
    print("  • Gradient descent minimizes loss function effectively")
    print("  • Both methods stay below 11% detection threshold")
    print("  • PID gains affect convergence speed and stability")
    print("  • Proper tuning allows maximum information gain while undetected")
    
    print("\nMathematical Principles:")
    print("  • QBER ≈ 0.25 × intercept_probability (theoretical)")
    print("  • PID: p(t+1) = p(t) + Kp*e + Ki*∫e + Kd*de/dt")
    print("  • GD: p(t+1) = p(t) - α × ∂L/∂p")
    print("  • Both approaches converge to optimal intercept probability")


if __name__ == "__main__":
    main()
