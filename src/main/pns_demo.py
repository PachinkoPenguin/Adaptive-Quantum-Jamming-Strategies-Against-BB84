"""
Photon Number Splitting (PNS) Attack Demonstrations

This module demonstrates the powerful PNS attack on weak coherent pulse BB84.
Shows how Eve can extract perfect information without introducing any QBER.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os
from math import factorial

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.main.bb84_main import (
    ClassicalBackend,
    PhotonNumberSplittingAttack,
    EveController,
    Alice,
    Bob,
    QuantumChannel,
    Basis
)


def demo_poisson_statistics():
    """
    Demonstrate Poisson statistics for photon number distribution.
    Shows how multi-photon probability varies with μ.
    """
    print("\n" + "="*70)
    print("DEMO 1: Poisson Statistics for Photon Number Distribution")
    print("="*70)
    
    # Range of mean photon numbers
    mu_values = np.linspace(0.01, 1.0, 100)
    
    # Calculate probabilities
    prob_multi = [PhotonNumberSplittingAttack.probability_multi_photon(mu) 
                  for mu in mu_values]
    info_gain = [PhotonNumberSplittingAttack.expected_information_gain(mu) 
                 for mu in mu_values]
    
    # Typical values
    typical_mu = [0.1, 0.2, 0.5]
    typical_probs = [PhotonNumberSplittingAttack.probability_multi_photon(mu) 
                     for mu in typical_mu]
    
    print("\nTypical μ values and multi-photon probabilities:")
    print("-" * 70)
    for mu, prob in zip(typical_mu, typical_probs):
        print(f"μ = {mu:.2f}: P(n≥2) = {prob:.4f} ({prob*100:.2f}%)")
        print(f"         Info gain: {prob:.4f} bits/pulse")
    
    # Calculate photon number distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Multi-photon probability vs μ
    ax = axes[0, 0]
    ax.plot(mu_values, prob_multi, 'b-', linewidth=2, label='P(n≥2)')
    ax.axhline(y=0.11, color='r', linestyle='--', label='QBER threshold')
    
    # Mark typical values
    for mu, prob in zip(typical_mu, typical_probs):
        ax.plot(mu, prob, 'ro', markersize=8)
        ax.annotate(f'μ={mu:.1f}\n{prob*100:.2f}%', 
                   xy=(mu, prob), xytext=(10, 10),
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Mean Photon Number (μ)', fontsize=11)
    ax.set_ylabel('Probability P(n≥2)', fontsize=11)
    ax.set_title('Multi-Photon Pulse Probability', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Information gain vs μ
    ax = axes[0, 1]
    ax.plot(mu_values, info_gain, 'g-', linewidth=2, label='Info per pulse')
    
    # Mark typical values
    for mu, info in zip(typical_mu, typical_probs):
        ax.plot(mu, info, 'ro', markersize=8)
    
    ax.set_xlabel('Mean Photon Number (μ)', fontsize=11)
    ax.set_ylabel('Information Gain (bits/pulse)', fontsize=11)
    ax.set_title('Expected Information Gain', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Photon number distribution for μ=0.1
    ax = axes[1, 0]
    n_photons = np.arange(0, 8)
    mu = 0.1
    probs = [np.exp(-mu) * mu**n / factorial(n) for n in n_photons]
    
    colors = ['red' if n < 2 else 'green' for n in n_photons]
    bars = ax.bar(n_photons, probs, color=colors, alpha=0.7, edgecolor='black')
    
    # Annotate
    total_multi = sum(probs[2:])
    ax.axvspan(1.5, 7.5, alpha=0.2, color='green', 
              label=f'Multi-photon: {total_multi*100:.2f}%')
    
    ax.set_xlabel('Number of Photons (n)', fontsize=11)
    ax.set_ylabel('Probability P(n)', fontsize=11)
    ax.set_title(f'Poisson Distribution (μ={mu})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Photon number distribution for μ=0.5
    ax = axes[1, 1]
    mu = 0.5
    probs = [np.exp(-mu) * mu**n / factorial(n) for n in n_photons]
    
    colors = ['red' if n < 2 else 'green' for n in n_photons]
    bars = ax.bar(n_photons, probs, color=colors, alpha=0.7, edgecolor='black')
    
    # Annotate
    total_multi = sum(probs[2:])
    ax.axvspan(1.5, 7.5, alpha=0.2, color='green',
              label=f'Multi-photon: {total_multi*100:.2f}%')
    
    ax.set_xlabel('Number of Photons (n)', fontsize=11)
    ax.set_ylabel('Probability P(n)', fontsize=11)
    ax.set_title(f'Poisson Distribution (μ={mu})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('bb84_output/pns_poisson_statistics.png', dpi=100, bbox_inches='tight')
    print("\n✓ Saved: bb84_output/pns_poisson_statistics.png")
    plt.close()


def demo_pns_attack_basic():
    """
    Demonstrate basic PNS attack on BB84.
    Shows perfect information extraction with zero QBER.
    """
    print("\n" + "="*70)
    print("DEMO 2: Basic PNS Attack on BB84")
    print("="*70)
    
    # Setup
    backend = ClassicalBackend()
    mu = 0.1  # Typical mean photon number
    
    strategy = PhotonNumberSplittingAttack(backend, mean_photon_number=mu)
    eve = EveController(strategy, backend)
    channel = QuantumChannel(backend, eve=eve)
    
    alice = Alice(backend)
    bob = Bob(backend)
    
    # Run protocol
    n_qubits = 1000
    print(f"\nRunning BB84 with {n_qubits} qubits (μ={mu})...")
    
    alice.generate_random_bits(n_qubits)
    alice.choose_random_bases(n_qubits)
    states = alice.prepare_states()
    
    # Transmit (Eve monitors all)
    received = channel.transmit(states)
    
    # Bob measures
    bob.choose_random_bases(len(received))
    bob.measure_states(received)
    
    # Basis reconciliation
    matching_indices = []
    for i in range(min(len(alice.bases), len(bob.bases))):
        if alice.bases[i] == bob.bases[i]:
            matching_indices.append(i)
    
    # PUBLIC ANNOUNCEMENT: Alice and Bob announce bases
    # Eve can now measure stored photons!
    feedback = {
        'alice_bases': alice.bases,
        'bob_bases': bob.bases,
        'matching_indices': matching_indices
    }
    strategy.update_strategy(feedback)
    
    # Get statistics
    eve_stats = eve.get_statistics()
    strategy_stats = strategy.get_statistics()
    
    print("\n" + "-"*70)
    print("RESULTS:")
    print("-"*70)
    print(f"Pulses monitored:        {strategy_stats['pulses_monitored']}")
    print(f"Multi-photon pulses:     {strategy_stats['multi_photon_pulses']}")
    print(f"Photons stored:          {strategy_stats['photons_stored']}")
    print(f"Successful extractions:  {strategy_stats['successful_extractions']}")
    print(f"\nMulti-photon probability:")
    print(f"  Theoretical:           {strategy_stats['multi_photon_probability_theory']:.4f}")
    print(f"  Actual:                {strategy_stats['multi_photon_probability_actual']:.4f}")
    print(f"\nInformation gain:")
    print(f"  Per pulse (expected):  {strategy_stats['information_per_pulse_expected']:.4f} bits")
    print(f"  Per pulse (actual):    {strategy_stats['information_per_pulse_actual']:.4f} bits")
    print(f"  Total:                 {strategy_stats['information_gained']:.1f} bits")
    print(f"\nDetection risk:")
    print(f"  QBER introduced:       {strategy_stats['qber_introduced']:.1%}")
    print(f"  Detection probability: {strategy_stats['detection_probability']:.1%}")
    print(f"\nEfficiency:              {strategy_stats['efficiency']:.1%}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Attack overview
    ax = axes[0, 0]
    categories = ['Monitored', 'Multi-photon', 'Stored', 'Measured']
    values = [
        strategy_stats['pulses_monitored'],
        strategy_stats['multi_photon_pulses'],
        strategy_stats['photons_stored'],
        strategy_stats['successful_extractions']
    ]
    colors = ['blue', 'orange', 'green', 'red']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(val)}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('PNS Attack Overview', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Probability comparison
    ax = axes[0, 1]
    probs = [
        strategy_stats['multi_photon_probability_theory'],
        strategy_stats['multi_photon_probability_actual']
    ]
    labels = ['Theoretical', 'Actual']
    colors = ['blue', 'green']
    
    bars = ax.bar(labels, probs, color=colors, alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('Multi-Photon Probability', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(probs) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Information gain
    ax = axes[1, 0]
    info_types = ['Expected', 'Actual', 'Total']
    info_values = [
        strategy_stats['information_per_pulse_expected'] * strategy_stats['pulses_monitored'],
        strategy_stats['information_per_pulse_actual'] * strategy_stats['pulses_monitored'],
        strategy_stats['information_gained']
    ]
    colors = ['blue', 'orange', 'green']
    
    bars = ax.bar(info_types, info_values, color=colors, alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, info_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Information (bits)', fontsize=11)
    ax.set_title('Total Information Gained', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Stealth metrics
    ax = axes[1, 1]
    ax.text(0.5, 0.8, 'STEALTH ANALYSIS', 
           ha='center', va='top', fontsize=16, fontweight='bold',
           transform=ax.transAxes)
    
    metrics_text = f"""
    QBER Introduced: {strategy_stats['qber_introduced']:.1%}
    
    Detection Probability: {strategy_stats['detection_probability']:.1%}
    
    Status: UNDETECTABLE ✓
    
    Key Advantage:
    • No measurement disturbance
    • Perfect basis knowledge
    • Zero QBER contribution
    • Information gain: {strategy_stats['information_gained']:.0f} bits
    """
    
    ax.text(0.5, 0.6, metrics_text,
           ha='center', va='top', fontsize=11,
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('bb84_output/pns_basic_attack.png', dpi=100, bbox_inches='tight')
    print("\n✓ Saved: bb84_output/pns_basic_attack.png")
    plt.close()


def demo_mu_comparison():
    """
    Compare PNS attack effectiveness for different μ values.
    Shows trade-off between key rate and vulnerability.
    """
    print("\n" + "="*70)
    print("DEMO 3: PNS Attack vs. Mean Photon Number")
    print("="*70)
    
    backend = ClassicalBackend()
    mu_values = [0.05, 0.1, 0.2, 0.5]
    n_qubits = 1000
    
    results = []
    
    for mu in mu_values:
        print(f"\nTesting μ = {mu}...")
        
        strategy = PhotonNumberSplittingAttack(backend, mean_photon_number=mu)
        eve = EveController(strategy, backend)
        channel = QuantumChannel(backend, eve=eve)
        
        alice = Alice(backend)
        bob = Bob(backend)
        
        # Run protocol
        alice.generate_random_bits(n_qubits)
        alice.choose_random_bases(n_qubits)
        states = alice.prepare_states()
        
        received = channel.transmit(states)
        bob.choose_random_bases(len(received))
        bob.measure_states(received)
        
        # Basis reconciliation
        matching_indices = []
        for i in range(min(len(alice.bases), len(bob.bases))):
            if alice.bases[i] == bob.bases[i]:
                matching_indices.append(i)
        
        # Eve measures after announcement
        strategy.update_strategy({
            'alice_bases': alice.bases,
            'bob_bases': bob.bases
        })
        
        stats = strategy.get_statistics()
        results.append({
            'mu': mu,
            'multi_photon_prob': stats['multi_photon_probability_actual'],
            'info_per_pulse': stats['information_per_pulse_actual'],
            'total_info': stats['successful_extractions'],
            'efficiency': stats['efficiency']
        })
        
        print(f"  Multi-photon: {stats['multi_photon_pulses']}")
        print(f"  Info gained: {stats['successful_extractions']} bits")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    mu_plot = [r['mu'] for r in results]
    
    # Plot 1: Multi-photon probability
    ax = axes[0, 0]
    probs = [r['multi_photon_prob'] for r in results]
    ax.plot(mu_plot, probs, 'bo-', linewidth=2, markersize=8, label='Actual')
    
    # Theoretical curve
    mu_theory = np.linspace(0.05, 0.5, 100)
    probs_theory = [PhotonNumberSplittingAttack.probability_multi_photon(m) 
                    for m in mu_theory]
    ax.plot(mu_theory, probs_theory, 'r--', linewidth=1, label='Theoretical')
    
    ax.set_xlabel('Mean Photon Number (μ)', fontsize=11)
    ax.set_ylabel('Multi-Photon Probability', fontsize=11)
    ax.set_title('Attack Opportunity vs μ', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Information per pulse
    ax = axes[0, 1]
    info = [r['info_per_pulse'] for r in results]
    ax.plot(mu_plot, info, 'go-', linewidth=2, markersize=8)
    
    ax.set_xlabel('Mean Photon Number (μ)', fontsize=11)
    ax.set_ylabel('Info per Pulse (bits)', fontsize=11)
    ax.set_title('Information Gain Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Total information
    ax = axes[1, 0]
    total_info = [r['total_info'] for r in results]
    colors = ['blue', 'green', 'orange', 'red']
    bars = ax.bar(mu_plot, total_info, color=colors, alpha=0.7, 
                  edgecolor='black', width=0.04)
    
    for bar, val in zip(bars, total_info):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(val)}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Mean Photon Number (μ)', fontsize=11)
    ax.set_ylabel('Total Information (bits)', fontsize=11)
    ax.set_title(f'Total Info from {n_qubits} Pulses', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['μ', 'P(n≥2)', 'Info/pulse', 'Total', 'Eff'])
    for r in results:
        table_data.append([
            f"{r['mu']:.2f}",
            f"{r['multi_photon_prob']:.4f}",
            f"{r['info_per_pulse']:.4f}",
            f"{int(r['total_info'])}",
            f"{r['efficiency']:.1%}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(5):
            table[(i, j)].set_facecolor(color)
    
    ax.set_title('PNS Attack Effectiveness Summary', 
                fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('bb84_output/pns_mu_comparison.png', dpi=100, bbox_inches='tight')
    print("\n✓ Saved: bb84_output/pns_mu_comparison.png")
    plt.close()


def demo_optimal_mu():
    """
    Demonstrate optimal μ calculation for different channel losses.
    Shows security trade-off with distance.
    """
    print("\n" + "="*70)
    print("DEMO 4: Optimal μ vs Channel Loss")
    print("="*70)
    
    # Channel losses (dB)
    losses_db = np.linspace(0, 30, 100)
    
    # Calculate optimal μ and vulnerability
    optimal_mus = [PhotonNumberSplittingAttack.optimal_mu_for_distance(loss) 
                   for loss in losses_db]
    vulnerabilities = [PhotonNumberSplittingAttack.probability_multi_photon(mu) 
                      for mu in optimal_mus]
    
    # Typical fiber: 0.2 dB/km at 1550nm
    distances_km = losses_db / 0.2
    
    # Key points
    key_distances = [10, 25, 50, 100]
    key_losses = [d * 0.2 for d in key_distances]
    key_mus = [PhotonNumberSplittingAttack.optimal_mu_for_distance(loss) 
               for loss in key_losses]
    key_vulns = [PhotonNumberSplittingAttack.probability_multi_photon(mu) 
                 for mu in key_mus]
    
    print("\nOptimal μ and PNS vulnerability by distance:")
    print("-" * 70)
    for dist, loss, mu, vuln in zip(key_distances, key_losses, key_mus, key_vulns):
        print(f"{dist:3.0f} km: Loss={loss:5.1f}dB, μ_opt={mu:.4f}, "
              f"P(n≥2)={vuln:.4f} ({vuln*100:.2f}%)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Optimal μ vs distance
    ax = axes[0, 0]
    ax.plot(distances_km, optimal_mus, 'b-', linewidth=2)
    
    # Mark key points
    for dist, mu in zip(key_distances, key_mus):
        idx = np.argmin(np.abs(distances_km - dist))
        ax.plot(dist, mu, 'ro', markersize=8)
        ax.annotate(f'{dist}km\nμ={mu:.3f}',
                   xy=(dist, mu), xytext=(10, -20),
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('Distance (km)', fontsize=11)
    ax.set_ylabel('Optimal μ', fontsize=11)
    ax.set_title('Optimal Mean Photon Number', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 150)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: PNS vulnerability vs distance
    ax = axes[0, 1]
    ax.plot(distances_km, vulnerabilities, 'r-', linewidth=2, label='P(n≥2)')
    ax.axhline(y=0.01, color='orange', linestyle='--', label='1% threshold')
    ax.axhline(y=0.05, color='red', linestyle='--', label='5% threshold')
    
    # Mark key points
    for dist, vuln in zip(key_distances, key_vulns):
        idx = np.argmin(np.abs(distances_km - dist))
        ax.plot(dist, vuln, 'bo', markersize=8)
    
    ax.set_xlabel('Distance (km)', fontsize=11)
    ax.set_ylabel('PNS Vulnerability P(n≥2)', fontsize=11)
    ax.set_title('Multi-Photon Probability', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 150)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Transmittance vs distance
    ax = axes[1, 0]
    transmittance = [10**(-loss/10) for loss in losses_db]
    ax.semilogy(distances_km, transmittance, 'g-', linewidth=2)
    
    # Mark key points
    for dist, loss in zip(key_distances, key_losses):
        trans = 10**(-loss/10)
        ax.plot(dist, trans, 'ro', markersize=8)
        ax.annotate(f'{dist}km',
                   xy=(dist, trans), xytext=(10, 10),
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax.set_xlabel('Distance (km)', fontsize=11)
    ax.set_ylabel('Channel Transmittance', fontsize=11)
    ax.set_title('Optical Fiber Transmission', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 150)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Distance', 'Loss', 'μ_opt', 'P(n≥2)', 'Vuln'])
    for dist, loss, mu, vuln in zip(key_distances, key_losses, key_mus, key_vulns):
        table_data.append([
            f'{dist} km',
            f'{loss:.1f} dB',
            f'{mu:.4f}',
            f'{vuln:.4f}',
            f'{vuln*100:.2f}%'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    bbox=[0, 0.2, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style
    for i in range(5):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(5):
            table[(i, j)].set_facecolor(color)
    
    # Add note
    note_text = ("Note: Assumes standard fiber\n"
                "Loss: 0.2 dB/km @ 1550nm\n"
                "μ_opt ≈ channel transmittance")
    ax.text(0.5, 0.05, note_text,
           ha='center', va='bottom', fontsize=9,
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    ax.set_title('PNS Vulnerability Analysis', 
                fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('bb84_output/pns_optimal_mu.png', dpi=100, bbox_inches='tight')
    print("\n✓ Saved: bb84_output/pns_optimal_mu.png")
    plt.close()


def main():
    """Run all PNS attack demonstrations."""
    print("\n" + "="*70)
    print(" PHOTON NUMBER SPLITTING (PNS) ATTACK DEMONSTRATIONS")
    print("="*70)
    print("\nThe PNS attack exploits weak coherent pulses in practical BB84.")
    print("Eve gains perfect information with ZERO QBER - completely undetectable!")
    
    # Run all demos
    demo_poisson_statistics()
    demo_pns_attack_basic()
    demo_mu_comparison()
    demo_optimal_mu()
    
    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  1. pns_poisson_statistics.png - Photon number distributions")
    print("  2. pns_basic_attack.png       - Basic PNS attack results")
    print("  3. pns_mu_comparison.png      - Attack vs mean photon number")
    print("  4. pns_optimal_mu.png         - Optimal μ and vulnerability")
    print("\nKey Findings:")
    print("  • PNS introduces 0% QBER (undetectable)")
    print("  • Information gain proportional to P(n≥2)")
    print("  • For μ=0.1: ~0.47% of pulses are vulnerable")
    print("  • Higher μ → better key rate but more vulnerable")
    print("  • Countermeasure: Use true single photons or decoy states")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
