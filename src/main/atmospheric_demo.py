"""
Atmospheric Turbulence-Adaptive Attack Demonstration
Shows how Eve exploits atmospheric turbulence to mask eavesdropping
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
    ChannelAdaptiveStrategy,
    AtmosphericChannelModel
)
import matplotlib.pyplot as plt
import numpy as np


def demo_rytov_variance_calculation():
    """
    Demo 1: Rytov Variance Calculation
    
    Shows how Rytov variance changes with distance and wavelength.
    """
    print("\n" + "="*70)
    print("DEMO 1: Rytov Variance Calculation")
    print("="*70)
    
    # Test different distances
    distances = [1, 5, 10, 20, 50]  # km
    wavelengths = [850, 1310, 1550]  # nm (common QKD wavelengths)
    
    print("\nRytov Variance vs Distance and Wavelength:")
    print("-"*70)
    print(f"{'Distance':>10} | {'λ=850nm':>12} {'λ=1310nm':>12} {'λ=1550nm':>12}")
    print("-"*70)
    
    results = {
        'distances': distances,
        'wavelengths': wavelengths,
        'rytov_values': {}
    }
    
    for wavelength in wavelengths:
        results['rytov_values'][wavelength] = []
        
    for distance in distances:
        row = f"{distance:>10} km |"
        
        for wavelength in wavelengths:
            model = AtmosphericChannelModel(
                distance_km=distance,
                wavelength_nm=wavelength,
                time_of_day='day'
            )
            
            rytov = model.calculate_rytov_variance()
            results['rytov_values'][wavelength].append(rytov)
            regime = model.get_turbulence_regime(rytov)
            
            row += f" {rytov:>10.3f} "
        
        print(row)
    
    print("-"*70)
    
    # Show turbulence regimes
    print("\nTurbulence Classification:")
    print("  σ²_R < 0.5:  Very Weak")
    print("  0.5-1.0:     Weak")
    print("  1.0-3.0:     Moderate")
    print("  > 3.0:       Strong")
    
    # Visualize
    visualize_rytov_vs_distance(results)
    
    return results


def demo_hufnagel_valley_model():
    """
    Demo 2: Hufnagel-Valley Atmospheric Model
    
    Shows Cn² variation with altitude for day vs night.
    """
    print("\n" + "="*70)
    print("DEMO 2: Hufnagel-Valley Atmospheric Model")
    print("="*70)
    
    altitudes = np.linspace(0, 20, 100)  # 0-20 km
    
    day_model = AtmosphericChannelModel(distance_km=10, time_of_day='day')
    night_model = AtmosphericChannelModel(distance_km=10, time_of_day='night')
    
    cn2_day = [day_model.cn2_hufnagel_valley(h*1000) for h in altitudes]
    cn2_night = [night_model.cn2_hufnagel_valley(h*1000) for h in altitudes]
    
    print("\nCn² Profile (selected altitudes):")
    print("-"*70)
    print(f"{'Altitude':>12} | {'Day Cn²':>15} {'Night Cn²':>15}")
    print("-"*70)
    
    for h in [0, 2, 5, 10, 15]:
        idx = int(h * len(altitudes) / 20)
        if idx < len(cn2_day):
            print(f"{h:>10} km | {cn2_day[idx]:>15.3e} {cn2_night[idx]:>15.3e}")
    
    print("-"*70)
    
    print("\nKey Parameters:")
    print(f"  Day:   A={day_model.A:.2e}, w={day_model.w:.1f} m/s")
    print(f"  Night: A={night_model.A:.2e}, w={night_model.w:.1f} m/s")
    
    # Visualize
    results = {
        'altitudes': altitudes,
        'cn2_day': cn2_day,
        'cn2_night': cn2_night
    }
    visualize_hufnagel_valley(results)
    
    return results


def demo_adaptive_interception():
    """
    Demo 3: Turbulence-Adaptive Interception
    
    Shows how interception probability adapts to turbulence conditions.
    """
    print("\n" + "="*70)
    print("DEMO 3: Turbulence-Adaptive Interception Strategy")
    print("="*70)
    
    backend = ClassicalBackend()
    
    # Test different turbulence conditions
    test_conditions = [
        {'distance_km': 1, 'cn2': 1e-15, 'label': 'Very Weak (1km, low Cn²)'},
        {'distance_km': 5, 'cn2': 1.7e-14, 'label': 'Weak (5km, moderate Cn²)'},
        {'distance_km': 10, 'cn2': 1.7e-14, 'label': 'Moderate (10km, moderate Cn²)'},
        {'distance_km': 20, 'cn2': 3e-14, 'label': 'Strong (20km, high Cn²)'},
    ]
    
    print("\nInterception Strategy by Turbulence Condition:")
    print("-"*70)
    print(f"{'Condition':>35} | {'σ²_R':>8} {'Regime':>12} {'P(int)':>8}")
    print("-"*70)
    
    results = []
    
    for cond in test_conditions:
        strategy = ChannelAdaptiveStrategy(
            backend,
            distance_km=cond['distance_km'],
            wavelength_nm=1550,
            cn2=cond['cn2']
        )
        
        rytov = strategy.current_rytov
        regime = strategy.turbulence_regime
        prob = strategy.get_intercept_probability()
        
        results.append({
            'label': cond['label'],
            'distance': cond['distance_km'],
            'cn2': cond['cn2'],
            'rytov': rytov,
            'regime': regime,
            'prob': prob
        })
        
        print(f"{cond['label']:>35} | {rytov:>8.3f} {regime:>12} {prob:>8.2f}")
    
    print("-"*70)
    
    print("\nStrategy Summary:")
    print("  Weak turbulence (σ²_R < 1):    Conservative (10-20% intercept)")
    print("  Moderate turbulence (1-3):     Standard (40% intercept)")
    print("  Strong turbulence (> 3):       Aggressive (70% intercept)")
    
    # Visualize
    visualize_adaptive_strategy(results)
    
    return results


def demo_dynamic_atmospheric_conditions():
    """
    Demo 4: Dynamic Atmospheric Conditions
    
    Simulates changing atmospheric conditions over time.
    """
    print("\n" + "="*70)
    print("DEMO 4: Dynamic Atmospheric Conditions")
    print("="*70)
    
    backend = ClassicalBackend()
    strategy = ChannelAdaptiveStrategy(
        backend,
        distance_km=10,
        wavelength_nm=1550,
        cn2=1.7e-14,
        time_of_day='day'
    )
    
    # Simulate changing conditions over 24 hours
    hours = 24
    cn2_values = []
    
    print("\nSimulating 24-hour atmospheric variation:")
    print("-"*70)
    print(f"{'Hour':>6} | {'Time':>6} {'Cn²':>12} {'σ²_R':>8} {'Regime':>12} {'P(int)':>8}")
    print("-"*70)
    
    results = {
        'hours': [],
        'cn2': [],
        'rytov': [],
        'regime': [],
        'prob': []
    }
    
    for hour in range(hours):
        # Determine time of day
        if 6 <= hour < 18:
            time = 'day'
            # Day: higher Cn² (more turbulence)
            cn2 = 1.7e-14 * (1.0 + 0.5 * np.sin(np.pi * (hour - 6) / 12))
        else:
            time = 'night'
            # Night: lower Cn² (less turbulence)
            cn2 = 1.0e-14 * (1.0 + 0.3 * np.sin(np.pi * (hour - 18) / 12))
        
        cn2_values.append(cn2)
        
        # Update strategy
        strategy.update_rytov_variance(cn2)
        
        rytov = strategy.current_rytov
        regime = strategy.turbulence_regime
        prob = strategy.get_intercept_probability()
        
        results['hours'].append(hour)
        results['cn2'].append(cn2)
        results['rytov'].append(rytov)
        results['regime'].append(regime)
        results['prob'].append(prob)
        
        if hour % 3 == 0:  # Print every 3 hours
            print(f"{hour:>6} | {time:>6} {cn2:>12.2e} {rytov:>8.3f} "
                  f"{regime:>12} {prob:>8.2f}")
    
    print("-"*70)
    
    stats = strategy.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Average Rytov Variance: {stats['avg_rytov_variance']:.3f}")
    print(f"  Regime Distribution: {stats['regime_counts']}")
    
    # Visualize
    visualize_dynamic_conditions(results)
    
    return results


def demo_phase_screen_generation():
    """
    Demo 5: Phase Screen Generation
    
    Shows Kolmogorov turbulence phase screens.
    """
    print("\n" + "="*70)
    print("DEMO 5: Atmospheric Phase Screen Generation")
    print("="*70)
    
    model = AtmosphericChannelModel(distance_km=10, time_of_day='day')
    
    # Generate phase screens for different Fried parameters
    r0_values = [0.05, 0.1, 0.2]  # meters
    
    print("\nGenerating Kolmogorov phase screens:")
    print(f"  Grid size: 256x256")
    print(f"  Spectrum: Φ(f) ∝ f^(-11/3)")
    
    results = {'r0_values': r0_values, 'screens': []}
    
    for r0 in r0_values:
        print(f"\n  r₀ = {r0} m (coherence length)")
        screen = model.generate_phase_screen(grid_size=256, r0=r0)
        results['screens'].append(screen)
        
        print(f"    Phase RMS: {np.std(screen):.3f} rad")
        print(f"    Phase range: [{np.min(screen):.3f}, {np.max(screen):.3f}] rad")
    
    print("\nNote: Smaller r₀ → stronger turbulence → larger phase distortions")
    
    # Visualize
    visualize_phase_screens(results)
    
    return results


def demo_full_attack_simulation():
    """
    Demo 6: Full Attack Simulation with Atmospheric Adaptation
    
    Complete BB84 attack under varying atmospheric conditions.
    """
    print("\n" + "="*70)
    print("DEMO 6: Full BB84 Attack with Atmospheric Adaptation")
    print("="*70)
    
    backend = ClassicalBackend()
    
    # Setup adaptive strategy
    strategy = ChannelAdaptiveStrategy(
        backend,
        distance_km=10,
        wavelength_nm=1550,
        cn2=1.7e-14,
        time_of_day='day'
    )
    eve = EveController(strategy, backend)
    channel = QuantumChannel(backend, loss_rate=0.0, error_rate=0.0, eve=eve)
    
    alice = Alice(backend)
    bob = Bob(backend)
    
    # Simulate varying atmospheric conditions
    rounds = 10
    n_qubits = 100
    
    results = {
        'round': [],
        'cn2': [],
        'rytov': [],
        'regime': [],
        'intercept_prob': [],
        'qber': [],
        'key_rate': [],
        'eve_info': []
    }
    
    print("\n" + "-"*70)
    print(f"{'Round':>5} | {'Cn²':>10} {'σ²_R':>7} {'Regime':>10} "
          f"{'P(int)':>7} {'QBER':>7} {'KeyRate':>8}")
    print("-"*70)
    
    np.random.seed(42)
    
    for round_num in range(1, rounds + 1):
        # Vary atmospheric conditions
        # Simulate weather changes
        if round_num < 4:
            cn2 = 1e-14  # Calm conditions
        elif round_num < 7:
            cn2 = 2.5e-14  # Moderate turbulence
        else:
            cn2 = 4e-14  # Strong turbulence
        
        # Update strategy
        strategy.update_rytov_variance(cn2)
        
        # Alice prepares qubits
        alice.generate_random_bits(n_qubits)
        alice.choose_random_bases(n_qubits)
        states = alice.prepare_states()
        
        # Transmit through channel (Eve intercepts)
        received = channel.transmit(states)
        
        # Bob measures
        bob.choose_random_bases(len(received))
        bob.measure_states(received)
        
        # Basis reconciliation
        alice_bases = alice.announce_bases()
        bob_bases = bob.announce_bases()
        
        matching_indices = [i for i in range(len(alice_bases))
                           if alice_bases[i] == bob_bases[i]]
        
        key_alice = [alice.bits[i] for i in matching_indices]
        key_bob = [bob.measured_bits[i] for i in matching_indices 
                   if bob.measured_bits[i] is not None]
        
        # Calculate QBER
        if len(key_alice) > 0 and len(key_bob) == len(key_alice):
            errors = sum(1 for i in range(len(key_alice)) if key_alice[i] != key_bob[i])
            qber = errors / len(key_alice)
        else:
            qber = 0.0
        
        # Statistics
        key_rate = len(key_alice) / n_qubits if n_qubits > 0 else 0
        eve_stats = eve.get_statistics()['attack_strategy_stats']
        
        results['round'].append(round_num)
        results['cn2'].append(cn2)
        results['rytov'].append(eve_stats['current_rytov_variance'])
        results['regime'].append(eve_stats['current_turbulence_regime'])
        results['intercept_prob'].append(eve_stats['current_intercept_probability'])
        results['qber'].append(qber)
        results['key_rate'].append(key_rate)
        results['eve_info'].append(eve_stats['information_gained'])
        
        print(f"{round_num:>5} | {cn2:>10.2e} {results['rytov'][-1]:>7.2f} "
              f"{results['regime'][-1]:>10} {results['intercept_prob'][-1]:>7.2f} "
              f"{qber:>7.3f} {key_rate:>8.3f}")
    
    print("-"*70)
    
    print("\nFinal Results:")
    print(f"  Average QBER: {np.mean(results['qber']):.3f}")
    print(f"  Average Key Rate: {np.mean(results['key_rate']):.3f}")
    print(f"  Total Information Gained by Eve: {results['eve_info'][-1]:.1f} bits")
    print(f"  Average Intercept Probability: {np.mean(results['intercept_prob']):.3f}")
    
    # Visualize
    visualize_full_attack(results)
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_rytov_vs_distance(results):
    """Visualize Rytov variance vs distance for different wavelengths."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, wavelength in enumerate(results['wavelengths']):
        ax.plot(results['distances'], results['rytov_values'][wavelength],
                color=colors[i], marker=markers[i], linewidth=2, markersize=8,
                label=f'λ = {wavelength} nm')
    
    # Add regime boundaries
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Very Weak|Weak')
    ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Weak|Moderate')
    ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, label='Moderate|Strong')
    
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Rytov Variance σ²_R', fontsize=12)
    ax.set_title('Rytov Variance vs Propagation Distance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    output_path = 'bb84_output/rytov_vs_distance.png'
    os.makedirs('bb84_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def visualize_hufnagel_valley(results):
    """Visualize Hufnagel-Valley Cn² profile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results['cn2_day'], results['altitudes'], 
            'r-', linewidth=2, label='Day (A=1.7e-14, w=27 m/s)')
    ax.plot(results['cn2_night'], results['altitudes'],
            'b-', linewidth=2, label='Night (A=1.28e-14, w=21 m/s)')
    
    ax.set_xlabel('Structure Constant Cn² (m^(-2/3))', fontsize=12)
    ax.set_ylabel('Altitude (km)', fontsize=12)
    ax.set_title('Hufnagel-Valley Atmospheric Model', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'bb84_output/hufnagel_valley_profile.png'
    os.makedirs('bb84_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def visualize_adaptive_strategy(results):
    """Visualize adaptive interception strategy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    labels = [r['label'] for r in results]
    rytov = [r['rytov'] for r in results]
    probs = [r['prob'] for r in results]
    
    # Rytov variance
    colors = ['green', 'yellow', 'orange', 'red']
    ax1.barh(labels, rytov, color=colors)
    ax1.set_xlabel('Rytov Variance σ²_R', fontsize=12)
    ax1.set_title('Turbulence Strength', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Interception probability
    ax2.barh(labels, probs, color=colors)
    ax2.set_xlabel('Interception Probability', fontsize=12)
    ax2.set_title('Adaptive Attack Strategy', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = 'bb84_output/adaptive_interception_strategy.png'
    os.makedirs('bb84_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def visualize_dynamic_conditions(results):
    """Visualize 24-hour atmospheric variation."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    hours = results['hours']
    
    # Cn²
    ax = axes[0]
    ax.plot(hours, results['cn2'], 'b-o', linewidth=2, markersize=6)
    ax.axvspan(0, 6, alpha=0.2, color='blue', label='Night')
    ax.axvspan(6, 18, alpha=0.2, color='yellow', label='Day')
    ax.axvspan(18, 24, alpha=0.2, color='blue')
    ax.set_ylabel('Cn² (m^(-2/3))', fontsize=11)
    ax.set_title('24-Hour Atmospheric Variation', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Rytov variance
    ax = axes[1]
    ax.plot(hours, results['rytov'], 'g-o', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Strong threshold')
    ax.set_ylabel('Rytov Variance σ²_R', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Interception probability
    ax = axes[2]
    ax.plot(hours, results['prob'], 'r-o', linewidth=2, markersize=6)
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('P(intercept)', fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'bb84_output/dynamic_atmospheric_conditions.png'
    os.makedirs('bb84_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def visualize_phase_screens(results):
    """Visualize Kolmogorov phase screens."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (r0, screen) in enumerate(zip(results['r0_values'], results['screens'])):
        ax = axes[i]
        im = ax.imshow(screen, cmap='twilight', origin='lower')
        ax.set_title(f'r₀ = {r0} m\nRMS = {np.std(screen):.2f} rad', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('x (pixels)', fontsize=10)
        ax.set_ylabel('y (pixels)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Phase (rad)')
    
    fig.suptitle('Kolmogorov Atmospheric Phase Screens', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = 'bb84_output/kolmogorov_phase_screens.png'
    os.makedirs('bb84_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def visualize_full_attack(results):
    """Visualize full BB84 attack simulation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rounds = results['round']
    
    # Rytov variance
    ax = axes[0, 0]
    ax.plot(rounds, results['rytov'], 'g-o', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Rytov Variance σ²_R', fontsize=11)
    ax.set_title('Atmospheric Turbulence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Interception probability
    ax = axes[0, 1]
    ax.plot(rounds, results['intercept_prob'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('P(intercept)', fontsize=11)
    ax.set_title('Adaptive Interception', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # QBER
    ax = axes[1, 0]
    ax.plot(rounds, results['qber'], 'r-o', linewidth=2, markersize=8)
    ax.axhline(y=0.11, color='orange', linestyle='--', alpha=0.7, 
               label='Detection threshold')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('QBER', fontsize=11)
    ax.set_title('Quantum Bit Error Rate', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Eve's information
    ax = axes[1, 1]
    ax.plot(rounds, results['eve_info'], 'm-o', linewidth=2, markersize=8)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Information Gained (bits)', fontsize=11)
    ax.set_title("Eve's Cumulative Information", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('BB84 Attack with Atmospheric Adaptation', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = 'bb84_output/atmospheric_attack_simulation.png'
    os.makedirs('bb84_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ATMOSPHERIC TURBULENCE-ADAPTIVE ATTACK DEMONSTRATIONS")
    print("="*70)
    print("\nThis demonstrates how Eve exploits atmospheric turbulence to mask")
    print("eavesdropping in free-space quantum key distribution:")
    print("  • Rytov variance quantifies turbulence strength")
    print("  • Strong turbulence → aggressive attack (70% intercept)")
    print("  • Weak turbulence → conservative attack (10% intercept)")
    
    try:
        demo_rytov_variance_calculation()
        demo_hufnagel_valley_model()
        demo_adaptive_interception()
        demo_dynamic_atmospheric_conditions()
        demo_phase_screen_generation()
        demo_full_attack_simulation()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated visualizations:")
        print("  • bb84_output/rytov_vs_distance.png")
        print("  • bb84_output/hufnagel_valley_profile.png")
        print("  • bb84_output/adaptive_interception_strategy.png")
        print("  • bb84_output/dynamic_atmospheric_conditions.png")
        print("  • bb84_output/kolmogorov_phase_screens.png")
        print("  • bb84_output/atmospheric_attack_simulation.png")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
