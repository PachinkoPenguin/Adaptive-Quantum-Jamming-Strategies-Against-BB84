"""
Bayesian Inference Attack Demonstration
Shows how Eve can learn patterns in Alice's basis selection using:
1. Beta distribution (conjugate prior for Bernoulli)
2. Particle filter (Sequential Monte Carlo)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.bb84_main import (
    ClassicalBackend,
    Alice,
    Bob,
    QuantumChannel,
    EveController,
    BasisLearningStrategy,
    ParticleFilterBasisLearner,
    Basis
)
import matplotlib.pyplot as plt
import numpy as np


def demo_beta_bayesian_learning():
    """
    Demo 1: Beta Distribution Bayesian Learning
    
    Shows how Eve learns Alice's basis pattern over multiple rounds.
    """
    print("\n" + "="*70)
    print("DEMO 1: Beta Distribution Bayesian Learning")
    print("="*70)
    
    # Setup
    backend = ClassicalBackend()
    strategy = BasisLearningStrategy(
        backend,
        base_intercept_prob=0.3,
        confidence_threshold=0.8
    )
    eve = EveController(strategy, backend)
    
    print("\nInitial State:")
    print(f"  Prior: Beta(α={strategy.alpha:.1f}, β={strategy.beta:.1f})")
    print(f"  P(Rectilinear) = {strategy.get_basis_probability(Basis.RECTILINEAR):.3f}")
    print(f"  Confidence = {strategy.get_confidence():.3f}")
    
    # Simulate Alice with slight bias (60% rectilinear)
    np.random.seed(42)
    rounds = 10
    
    results = {
        'round': [],
        'alpha': [],
        'beta': [],
        'prob_rectilinear': [],
        'confidence': [],
        'prediction_accuracy': []
    }
    
    print("\n" + "-"*70)
    print("Learning Progress:")
    print("-"*70)
    
    for round_num in range(1, rounds + 1):
        # Alice generates bases (60% rectilinear, 40% diagonal)
        n_bases = 50
        alice_bases = []
        for _ in range(n_bases):
            if np.random.random() < 0.6:
                alice_bases.append(Basis.RECTILINEAR)
            else:
                alice_bases.append(Basis.DIAGONAL)
        
        # Eve observes from public announcement
        for basis in alice_bases:
            strategy.observe_basis(basis)
        
        # Track results
        results['round'].append(round_num)
        results['alpha'].append(strategy.alpha)
        results['beta'].append(strategy.beta)
        results['prob_rectilinear'].append(
            strategy.get_basis_probability(Basis.RECTILINEAR)
        )
        results['confidence'].append(strategy.get_confidence())
        
        # Test prediction accuracy
        correct = 0
        for _ in range(20):
            true_basis = Basis.RECTILINEAR if np.random.random() < 0.6 else Basis.DIAGONAL
            predicted = strategy.predict_basis()
            if true_basis == predicted:
                correct += 1
        accuracy = correct / 20
        results['prediction_accuracy'].append(accuracy)
        
        print(f"Round {round_num:2d}: "
              f"α={strategy.alpha:6.1f}, β={strategy.beta:6.1f} | "
              f"P(Rect)={results['prob_rectilinear'][-1]:.3f} | "
              f"Conf={results['confidence'][-1]:.3f} | "
              f"Acc={accuracy:.3f}")
    
    print("-"*70)
    
    # Final statistics
    stats = strategy.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Observations: {stats['total_observations']}")
    print(f"  Learned P(Rectilinear): {stats['prob_rectilinear']:.3f}")
    print(f"  Learned P(Diagonal): {stats['prob_diagonal']:.3f}")
    print(f"  Confidence: {stats['confidence']:.3f}")
    print(f"  Final Prediction Accuracy: {results['prediction_accuracy'][-1]:.3f}")
    
    # Visualization
    visualize_beta_learning(results)
    
    return results


def demo_particle_filter_learning():
    """
    Demo 2: Particle Filter Bayesian Learning
    
    Shows Sequential Monte Carlo approach to learning basis patterns.
    """
    print("\n" + "="*70)
    print("DEMO 2: Particle Filter Bayesian Learning")
    print("="*70)
    
    # Setup
    backend = ClassicalBackend()
    strategy = ParticleFilterBasisLearner(
        backend,
        n_particles=1000,
        base_intercept_prob=0.3,
        confidence_threshold=0.8
    )
    
    print("\nInitial State:")
    print(f"  Particles: {strategy.n_particles}")
    print(f"  P(Rectilinear) = {strategy.get_basis_probability(Basis.RECTILINEAR):.3f}")
    print(f"  Confidence = {strategy.get_confidence():.3f}")
    print(f"  ESS = {strategy.effective_sample_size():.1f}")
    
    # Simulate Alice with slight bias (60% rectilinear)
    np.random.seed(42)
    rounds = 10
    
    results = {
        'round': [],
        'prob_rectilinear': [],
        'confidence': [],
        'ess': [],
        'resampling_count': [],
        'prediction_accuracy': []
    }
    
    print("\n" + "-"*70)
    print("Learning Progress:")
    print("-"*70)
    
    for round_num in range(1, rounds + 1):
        # Alice generates bases (60% rectilinear, 40% diagonal)
        n_bases = 50
        alice_bases = []
        for _ in range(n_bases):
            if np.random.random() < 0.6:
                alice_bases.append(Basis.RECTILINEAR)
            else:
                alice_bases.append(Basis.DIAGONAL)
        
        # Eve observes from public announcement
        for basis in alice_bases:
            strategy.observe_basis(basis)
        
        # Track results
        results['round'].append(round_num)
        results['prob_rectilinear'].append(
            strategy.get_basis_probability(Basis.RECTILINEAR)
        )
        results['confidence'].append(strategy.get_confidence())
        results['ess'].append(strategy.effective_sample_size())
        results['resampling_count'].append(strategy.resampling_count)
        
        # Test prediction accuracy
        correct = 0
        for _ in range(20):
            true_basis = Basis.RECTILINEAR if np.random.random() < 0.6 else Basis.DIAGONAL
            predicted = strategy.predict_basis()
            if true_basis == predicted:
                correct += 1
        accuracy = correct / 20
        results['prediction_accuracy'].append(accuracy)
        
        print(f"Round {round_num:2d}: "
              f"P(Rect)={results['prob_rectilinear'][-1]:.3f} | "
              f"Conf={results['confidence'][-1]:.3f} | "
              f"ESS={results['ess'][-1]:6.1f} | "
              f"Resample={results['resampling_count'][-1]:2d} | "
              f"Acc={accuracy:.3f}")
    
    print("-"*70)
    
    # Final statistics
    stats = strategy.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Observations: {stats['total_observations']}")
    print(f"  Learned P(Rectilinear): {stats['prob_rectilinear']:.3f}")
    print(f"  Learned P(Diagonal): {stats['prob_diagonal']:.3f}")
    print(f"  Confidence: {stats['confidence']:.3f}")
    print(f"  Total Resamplings: {stats['resampling_count']}")
    print(f"  Final ESS: {stats['effective_sample_size']:.1f}")
    print(f"  Final Prediction Accuracy: {results['prediction_accuracy'][-1]:.3f}")
    
    # Visualization
    visualize_particle_filter_learning(results, strategy)
    
    return results, strategy


def compare_bayesian_methods():
    """
    Demo 3: Compare Beta vs Particle Filter
    
    Side-by-side comparison of both Bayesian learning approaches.
    """
    print("\n" + "="*70)
    print("DEMO 3: Compare Beta Distribution vs Particle Filter")
    print("="*70)
    
    backend = ClassicalBackend()
    
    # Setup both strategies
    beta_strategy = BasisLearningStrategy(backend, base_intercept_prob=0.3)
    particle_strategy = ParticleFilterBasisLearner(backend, n_particles=1000)
    
    # Simulate Alice with bias
    np.random.seed(42)
    rounds = 15
    
    results = {
        'round': [],
        'beta_prob': [],
        'beta_conf': [],
        'beta_acc': [],
        'particle_prob': [],
        'particle_conf': [],
        'particle_acc': [],
        'particle_ess': []
    }
    
    print("\n" + "-"*70)
    print(f"{'Round':>5} | {'Beta P(R)':>10} {'Conf':>6} {'Acc':>6} | "
          f"{'PF P(R)':>10} {'Conf':>6} {'Acc':>6} {'ESS':>7}")
    print("-"*70)
    
    for round_num in range(1, rounds + 1):
        # Alice generates bases (60% rectilinear)
        n_bases = 50
        alice_bases = []
        for _ in range(n_bases):
            if np.random.random() < 0.6:
                alice_bases.append(Basis.RECTILINEAR)
            else:
                alice_bases.append(Basis.DIAGONAL)
        
        # Both strategies observe
        for basis in alice_bases:
            beta_strategy.observe_basis(basis)
            particle_strategy.observe_basis(basis)
        
        # Test accuracy for both
        beta_correct = 0
        particle_correct = 0
        for _ in range(20):
            true_basis = Basis.RECTILINEAR if np.random.random() < 0.6 else Basis.DIAGONAL
            if beta_strategy.predict_basis() == true_basis:
                beta_correct += 1
            if particle_strategy.predict_basis() == true_basis:
                particle_correct += 1
        
        beta_acc = beta_correct / 20
        particle_acc = particle_correct / 20
        
        # Track results
        results['round'].append(round_num)
        results['beta_prob'].append(beta_strategy.get_basis_probability(Basis.RECTILINEAR))
        results['beta_conf'].append(beta_strategy.get_confidence())
        results['beta_acc'].append(beta_acc)
        results['particle_prob'].append(particle_strategy.get_basis_probability(Basis.RECTILINEAR))
        results['particle_conf'].append(particle_strategy.get_confidence())
        results['particle_acc'].append(particle_acc)
        results['particle_ess'].append(particle_strategy.effective_sample_size())
        
        print(f"{round_num:5d} | "
              f"{results['beta_prob'][-1]:10.3f} {results['beta_conf'][-1]:6.3f} {beta_acc:6.3f} | "
              f"{results['particle_prob'][-1]:10.3f} {results['particle_conf'][-1]:6.3f} "
              f"{particle_acc:6.3f} {results['particle_ess'][-1]:7.1f}")
    
    print("-"*70)
    
    # Comparison
    print("\nComparison:")
    print(f"  Beta Distribution:")
    print(f"    Final P(Rectilinear): {results['beta_prob'][-1]:.3f}")
    print(f"    Final Confidence: {results['beta_conf'][-1]:.3f}")
    print(f"    Final Accuracy: {results['beta_acc'][-1]:.3f}")
    print(f"  Particle Filter:")
    print(f"    Final P(Rectilinear): {results['particle_prob'][-1]:.3f}")
    print(f"    Final Confidence: {results['particle_conf'][-1]:.3f}")
    print(f"    Final Accuracy: {results['particle_acc'][-1]:.3f}")
    print(f"    Total Resamplings: {particle_strategy.resampling_count}")
    
    # Visualization
    visualize_comparison(results)
    
    return results


def demo_full_bb84_with_bayesian():
    """
    Demo 4: Full BB84 Protocol with Bayesian Eve
    
    Complete BB84 run with Bayesian eavesdropping over multiple rounds.
    """
    print("\n" + "="*70)
    print("DEMO 4: Full BB84 Protocol with Bayesian Eavesdropper")
    print("="*70)
    
    backend = ClassicalBackend()
    
    # Setup with Bayesian Eve
    strategy = BasisLearningStrategy(
        backend,
        base_intercept_prob=0.3,
        confidence_threshold=0.8
    )
    eve = EveController(strategy, backend)
    channel = QuantumChannel(backend, loss_rate=0.0, error_rate=0.0, eve=eve)
    
    alice = Alice(backend, name="Alice")
    bob = Bob(backend, name="Bob")
    
    rounds = 8
    n_qubits = 100
    
    results = {
        'round': [],
        'qber': [],
        'key_rate': [],
        'eve_info_gained': [],
        'eve_confidence': [],
        'eve_accuracy': []
    }
    
    print("\n" + "-"*70)
    print(f"{'Round':>5} | {'QBER':>6} {'KeyRate':>8} | "
          f"{'EveInfo':>8} {'Conf':>6} {'Acc':>6}")
    print("-"*70)
    
    for round_num in range(1, rounds + 1):
        # Alice prepares qubits
        alice.generate_random_bits(n_qubits)
        alice.choose_random_bases(n_qubits)
        states = alice.prepare_states()
        
        # Transmit through channel (Eve intercepts)
        received = channel.transmit(states)
        
        # Bob measures
        bob.choose_random_bases(len(received))
        bob.measure_states(received)
        
        # Public basis reconciliation
        alice_bases_pub = alice.announce_bases()
        bob_bases_pub = bob.announce_bases()
        
        # Key sifting
        matching_indices = [i for i in range(len(alice_bases_pub))
                           if alice_bases_pub[i] == bob_bases_pub[i]]
        
        key_alice = [alice.bits[i] for i in matching_indices]
        key_bob = [bob.measured_bits[i] for i in matching_indices if bob.measured_bits[i] is not None]
        
        # Calculate QBER
        if len(key_alice) > 0 and len(key_bob) == len(key_alice):
            errors = sum(1 for i in range(len(key_alice)) if key_alice[i] != key_bob[i])
            qber = errors / len(key_alice)
        else:
            qber = 0.0
        
        # Eve learns from public bases and QBER
        eve.receive_feedback(qber, {'alice_bases': alice_bases_pub})
        
        # Statistics
        key_rate = len(key_alice) / n_qubits if n_qubits > 0 else 0
        eve_stats = eve.get_statistics()
        strategy_stats = eve_stats['attack_strategy_stats']
        
        results['round'].append(round_num)
        results['qber'].append(qber)
        results['key_rate'].append(key_rate)
        results['eve_info_gained'].append(strategy_stats['information_gained'])
        results['eve_confidence'].append(strategy_stats['confidence'])
        results['eve_accuracy'].append(strategy_stats['prediction_accuracy'])
        
        print(f"{round_num:5d} | "
              f"{qber:6.3f} {key_rate:8.3f} | "
              f"{strategy_stats['information_gained']:8.1f} "
              f"{strategy_stats['confidence']:6.3f} "
              f"{strategy_stats['prediction_accuracy']:6.3f}")
    
    print("-"*70)
    
    print("\nFinal Results:")
    print(f"  Average QBER: {np.mean(results['qber']):.3f}")
    print(f"  Average Key Rate: {np.mean(results['key_rate']):.3f}")
    print(f"  Total Information Gained by Eve: {results['eve_info_gained'][-1]:.1f} bits")
    print(f"  Eve's Final Confidence: {results['eve_confidence'][-1]:.3f}")
    print(f"  Eve's Final Prediction Accuracy: {results['eve_accuracy'][-1]:.3f}")
    
    # Visualization
    visualize_bb84_with_bayesian(results)
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_beta_learning(results):
    """Visualize Beta distribution learning progress."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Beta Distribution Bayesian Learning', fontsize=16, fontweight='bold')
    
    # Plot 1: Alpha and Beta parameters
    ax = axes[0, 0]
    ax.plot(results['round'], results['alpha'], 'b-o', label='α (Rectilinear)', linewidth=2)
    ax.plot(results['round'], results['beta'], 'r-s', label='β (Diagonal)', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Prior')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Beta Distribution Parameters', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Learned probability
    ax = axes[0, 1]
    ax.plot(results['round'], results['prob_rectilinear'], 'g-o', linewidth=2, markersize=8)
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='True P(Rect)=0.6')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Uniform Prior')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('P(Rectilinear)', fontsize=12)
    ax.set_title('Learned Basis Probability', fontsize=13, fontweight='bold')
    ax.set_ylim([0.4, 0.7])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Confidence
    ax = axes[1, 0]
    ax.plot(results['round'], results['confidence'], 'm-o', linewidth=2, markersize=8)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Prediction Confidence', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Prediction accuracy
    ax = axes[1, 1]
    ax.plot(results['round'], results['prediction_accuracy'], 'c-o', linewidth=2, markersize=8)
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Expected Accuracy')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Prediction Accuracy', fontsize=13, fontweight='bold')
    ax.set_ylim([0.4, 0.8])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = 'bb84_output/beta_bayesian_learning.png'
    os.makedirs('bb84_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def visualize_particle_filter_learning(results, strategy):
    """Visualize particle filter learning progress."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Particle Filter Bayesian Learning', fontsize=16, fontweight='bold')
    
    # Plot 1: Learned probability
    ax = axes[0, 0]
    ax.plot(results['round'], results['prob_rectilinear'], 'g-o', linewidth=2, markersize=8)
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='True P(Rect)=0.6')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('P(Rectilinear)', fontsize=12)
    ax.set_title('Learned Basis Probability', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Confidence
    ax = axes[0, 1]
    ax.plot(results['round'], results['confidence'], 'm-o', linewidth=2, markersize=8)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Prediction Confidence', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Effective Sample Size
    ax = axes[0, 2]
    ax.plot(results['round'], results['ess'], 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=strategy.n_particles/2, color='red', linestyle='--', 
               alpha=0.7, label=f'Resample Threshold ({strategy.n_particles//2})')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('ESS', fontsize=12)
    ax.set_title('Effective Sample Size', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Resampling count
    ax = axes[1, 0]
    ax.plot(results['round'], results['resampling_count'], 'r-s', linewidth=2, markersize=8)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Resampling Count', fontsize=12)
    ax.set_title('Cumulative Resamplings', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Prediction accuracy
    ax = axes[1, 1]
    ax.plot(results['round'], results['prediction_accuracy'], 'c-o', linewidth=2, markersize=8)
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Expected Accuracy')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Prediction Accuracy', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Particle distribution histogram
    ax = axes[1, 2]
    ax.hist(strategy.particles, bins=30, weights=strategy.weights, 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.6, color='red', linestyle='--', linewidth=2, label='True P(Rect)=0.6')
    ax.set_xlabel('P(Rectilinear)', fontsize=12)
    ax.set_ylabel('Weighted Count', fontsize=12)
    ax.set_title('Final Particle Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_path = 'bb84_output/particle_filter_learning.png'
    os.makedirs('bb84_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def visualize_comparison(results):
    """Compare Beta vs Particle Filter methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Beta Distribution vs Particle Filter Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Probability convergence
    ax = axes[0, 0]
    ax.plot(results['round'], results['beta_prob'], 'b-o', 
            label='Beta Distribution', linewidth=2, markersize=6)
    ax.plot(results['round'], results['particle_prob'], 'r-s', 
            label='Particle Filter', linewidth=2, markersize=6)
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='True P(Rect)=0.6')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('P(Rectilinear)', fontsize=12)
    ax.set_title('Probability Convergence', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Confidence comparison
    ax = axes[0, 1]
    ax.plot(results['round'], results['beta_conf'], 'b-o', 
            label='Beta Distribution', linewidth=2, markersize=6)
    ax.plot(results['round'], results['particle_conf'], 'r-s', 
            label='Particle Filter', linewidth=2, markersize=6)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Confidence Growth', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy comparison
    ax = axes[1, 0]
    ax.plot(results['round'], results['beta_acc'], 'b-o', 
            label='Beta Distribution', linewidth=2, markersize=6)
    ax.plot(results['round'], results['particle_acc'], 'r-s', 
            label='Particle Filter', linewidth=2, markersize=6)
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='Expected')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Prediction Accuracy', fontsize=12)
    ax.set_title('Prediction Accuracy', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Particle Filter ESS
    ax = axes[1, 1]
    ax.plot(results['round'], results['particle_ess'], 'r-s', linewidth=2, markersize=6)
    ax.axhline(y=500, color='orange', linestyle='--', alpha=0.7, 
               label='Resample Threshold (500)')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Effective Sample Size', fontsize=12)
    ax.set_title('Particle Filter ESS', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = 'bb84_output/bayesian_comparison.png'
    os.makedirs('bb84_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def visualize_bb84_with_bayesian(results):
    """Visualize full BB84 protocol with Bayesian Eve."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BB84 Protocol with Bayesian Eavesdropper', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: QBER over rounds
    ax = axes[0, 0]
    ax.plot(results['round'], results['qber'], 'r-o', linewidth=2, markersize=8)
    ax.axhline(y=0.11, color='orange', linestyle='--', alpha=0.7, 
               label='Detection Threshold (11%)')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('QBER', fontsize=12)
    ax.set_title('Quantum Bit Error Rate', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Key rate
    ax = axes[0, 1]
    ax.plot(results['round'], results['key_rate'], 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Key Rate', fontsize=12)
    ax.set_title('Key Generation Rate', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 0.6])
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Eve's information gain
    ax = axes[1, 0]
    ax.plot(results['round'], results['eve_info_gained'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Information Gained (bits)', fontsize=12)
    ax.set_title("Eve's Cumulative Information", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Eve's confidence and accuracy
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(results['round'], results['eve_confidence'], 'm-o', 
                 label='Confidence', linewidth=2, markersize=8)
    l2 = ax2.plot(results['round'], results['eve_accuracy'], 'c-s', 
                  label='Accuracy', linewidth=2, markersize=8)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12, color='m')
    ax2.set_ylabel('Prediction Accuracy', fontsize=12, color='c')
    ax.set_title("Eve's Learning Progress", fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='m')
    ax2.tick_params(axis='y', labelcolor='c')
    
    # Combine legends
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = 'bb84_output/bb84_with_bayesian_eve.png'
    os.makedirs('bb84_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE ATTACK DEMONSTRATIONS")
    print("="*70)
    print("\nThis demonstrates how Eve can learn patterns in Alice's basis")
    print("selection using Bayesian inference methods:")
    print("  1. Beta Distribution (conjugate prior)")
    print("  2. Particle Filter (Sequential Monte Carlo)")
    
    # Run all demos
    try:
        demo_beta_bayesian_learning()
        demo_particle_filter_learning()
        compare_bayesian_methods()
        demo_full_bb84_with_bayesian()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated visualizations:")
        print("  • bb84_output/beta_bayesian_learning.png")
        print("  • bb84_output/particle_filter_learning.png")
        print("  • bb84_output/bayesian_comparison.png")
        print("  • bb84_output/bb84_with_bayesian_eve.png")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
