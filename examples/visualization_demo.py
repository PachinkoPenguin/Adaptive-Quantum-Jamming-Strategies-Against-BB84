#!/usr/bin/env python3
"""
Demonstration of VisualizationManager capabilities for BB84 attack analysis.

This script shows how to use the comprehensive visualization tools to:
1. Analyze QBER evolution and detection
2. Compare attack strategies
3. Visualize atmospheric effects
4. Generate statistical analysis plots
5. Export publication-ready LaTeX tables

Run this script to generate example figures in the 'demo_visualizations' directory.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main.bb84_main import (
    VisualizationManager,
    AdaptiveJammingSimulator,
    InterceptResendAttack,
    QBERAdaptiveStrategy,
    BasisLearningStrategy,
    PhotonNumberSplittingAttack,
    ClassicalBackend,
    Basis
)


def demo_basic_plots():
    """Demonstrate basic plotting capabilities."""
    print("\n=== DEMO 1: Basic Visualization Tools ===\n")
    
    viz = VisualizationManager(output_dir="demo_visualizations/basic")
    
    # 1. QBER Evolution
    print("1. Generating QBER evolution plot...")
    np.random.seed(42)
    qber_history = list(0.05 + 0.1 * np.abs(np.sin(np.linspace(0, 6*np.pi, 200))) + 
                       np.random.normal(0, 0.01, 200))
    filepath = viz.plot_qber_evolution(qber_history, threshold=0.11)
    print(f"   Saved: {filepath}")
    
    # 2. Intercept Probability Adaptation
    print("2. Generating intercept probability plot...")
    intercept_history = list(np.clip(0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, 150)), 0, 1))
    filepath = viz.plot_intercept_probability(intercept_history)
    print(f"   Saved: {filepath}")
    
    # 3. Information Leakage Comparison
    print("3. Generating information leakage comparison...")
    strategies = {
        'Intercept-Resend': 0.50,
        'QBER-Adaptive': 0.35,
        'Basis-Learning': 0.42,
        'PNS Attack': 0.28,
        'Channel-Adaptive': 0.31
    }
    filepath = viz.plot_information_leakage(strategies)
    print(f"   Saved: {filepath}")
    
    # 4. Detection ROC Curves
    print("4. Generating detection ROC curves...")
    fpr = np.linspace(0, 1, 100)
    roc_data = {
        'QBER Detection': {
            'fpr': list(fpr),
            'tpr': list(np.power(fpr, 0.5) * 0.95)
        },
        'Chi-Square Test': {
            'fpr': list(fpr),
            'tpr': list(np.power(fpr, 0.6) * 0.9)
        },
        'MI Detection': {
            'fpr': list(fpr),
            'tpr': list(np.power(fpr, 0.7) * 0.85)
        }
    }
    filepath = viz.plot_detection_roc_curve(roc_data)
    print(f"   Saved: {filepath}")


def demo_adaptive_dashboard():
    """Demonstrate the comprehensive 4-subplot dashboard."""
    print("\n=== DEMO 2: Adaptive Strategy Dashboard ===\n")
    
    viz = VisualizationManager(output_dir="demo_visualizations/dashboard")
    
    np.random.seed(42)
    n_steps = 200
    
    # Generate synthetic adaptive behavior
    qber_history = list(0.05 + 0.08 * (1 - np.exp(-np.arange(n_steps)/50)) + 
                       np.random.normal(0, 0.01, n_steps))
    
    intercept_prob = list(0.8 * np.exp(-np.arange(n_steps)/80) + 0.2 + 
                         np.random.normal(0, 0.05, n_steps))
    intercept_prob = [max(0, min(1, p)) for p in intercept_prob]
    
    info_gain = list(np.cumsum(0.01 * np.array(intercept_prob)))
    
    detection_metrics = {
        'QBER Flag': [1 if q > 0.11 else 0 for q in qber_history[:100]],
        'Chi-Square p-value': list(np.random.beta(2, 5, 100)),
        'MI Anomaly': [1 if np.random.random() > 0.8 else 0 for _ in range(100)]
    }
    
    print("Generating adaptive strategy dashboard...")
    filepath = viz.plot_adaptive_strategy_dashboard(
        qber_history,
        intercept_prob,
        info_gain,
        detection_metrics,
        qber_threshold=0.11
    )
    print(f"Saved: {filepath}")


def demo_atmospheric_effects():
    """Demonstrate atmospheric turbulence visualization."""
    print("\n=== DEMO 3: Atmospheric Effects Visualization ===\n")
    
    viz = VisualizationManager(output_dir="demo_visualizations/atmospheric")
    
    np.random.seed(42)
    
    # 1. Rytov Variance Evolution
    print("1. Generating Rytov variance plot...")
    time = list(np.linspace(0, 100, 300))
    rytov = list(0.1 + 0.5 * np.abs(np.sin(np.linspace(0, 6*np.pi, 300))) + 
                np.random.exponential(0.1, 300))
    filepath = viz.plot_rytov_variance(time, rytov)
    print(f"   Saved: {filepath}")
    
    # 2. Attack Aggression vs Turbulence
    print("2. Generating aggression vs turbulence scatter plot...")
    turbulence = list(np.random.exponential(0.5, 80))
    aggression = list(np.clip(0.9 - 0.4 * np.array(turbulence) + 
                             np.random.normal(0, 0.1, 80), 0, 1))
    qber = list(0.05 + 0.25 * np.array(aggression) + 0.15 * np.array(turbulence))
    filepath = viz.plot_attack_aggression_vs_turbulence(turbulence, aggression, qber)
    print(f"   Saved: {filepath}")
    
    # 3. Phase Screen
    print("3. Generating atmospheric phase screen...")
    size = 128
    phase_screen = np.random.randn(size, size)
    # Add spatial correlation
    try:
        from scipy.ndimage import gaussian_filter
        phase_screen = gaussian_filter(phase_screen, sigma=5)
    except ImportError:
        pass
    phase_screen = phase_screen * 2 * np.pi
    filepath = viz.plot_phase_screen(phase_screen)
    print(f"   Saved: {filepath}")
    
    # 4. Beam Propagation Animation Frames
    print("4. Generating beam propagation animation frames...")
    size = 64
    n_frames = 15
    frames = []
    for i in range(n_frames):
        x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
        r2 = x**2 + y**2
        # Simulate beam spreading and turbulence
        intensity = np.exp(-r2 / (0.5 + 0.2*i)) * (1 + 0.3*np.random.randn(size, size))
        intensity = np.clip(intensity, 0, None)
        frames.append(intensity)
    frame_dir = viz.animate_beam_propagation(frames, fps=5)
    print(f"   Saved frames to: {frame_dir}")


def demo_statistical_analysis():
    """Demonstrate statistical analysis plots."""
    print("\n=== DEMO 4: Statistical Analysis Plots ===\n")
    
    viz = VisualizationManager(output_dir="demo_visualizations/statistics")
    
    np.random.seed(42)
    
    # 1. Error Pattern Analysis
    print("1. Generating error pattern analysis...")
    errors = []
    for _ in range(800):
        if np.random.random() < 0.12:  # 12% error rate
            # Create clusters
            cluster_size = np.random.randint(1, 5)
            errors.extend([1] * cluster_size)
        else:
            errors.append(0)
    errors = errors[:800]
    filepath = viz.plot_error_pattern_analysis(errors)
    print(f"   Saved: {filepath}")
    
    # 2. Basis Bias Detection
    print("2. Generating basis bias detection plot...")
    basis_sequence = [1 if np.random.random() < 0.58 else 0 for _ in range(1500)]
    filepath = viz.plot_basis_bias_detection(basis_sequence, expected_prob=0.5)
    print(f"   Saved: {filepath}")
    
    # 3. Mutual Information Matrix
    print("3. Generating mutual information matrix...")
    mi_matrix = np.array([
        [1.000, 0.920, 0.085],  # Alice
        [0.920, 1.000, 0.072],  # Bob
        [0.085, 0.072, 1.000]   # Eve
    ])
    filepath = viz.plot_mutual_information_matrix(mi_matrix, labels=['Alice', 'Bob', 'Eve'])
    print(f"   Saved: {filepath}")
    
    # 4. Holevo Bound Comparison
    print("4. Generating Holevo bound comparison...")
    strategies = ['Intercept\nResend', 'QBER\nAdaptive', 'Basis\nLearning', 'PNS\nAttack']
    theoretical = [1.0, 0.85, 0.92, 0.75]
    actual = [0.50, 0.35, 0.42, 0.28]
    filepath = viz.plot_holevo_bound_comparison(strategies, theoretical, actual)
    print(f"   Saved: {filepath}")


def demo_latex_export():
    """Demonstrate LaTeX table export."""
    print("\n=== DEMO 5: LaTeX Table Export ===\n")
    
    viz = VisualizationManager(output_dir="demo_visualizations/tables")
    
    # Create comprehensive results table
    results = {
        'Intercept-Resend': {
            'QBER': {'mean': 0.250, 'std': 0.012, 'significant': True},
            'Key Rate (bits/s)': {'mean': 125.3, 'std': 8.2, 'significant': False},
            'Detection Rate': {'mean': 0.954, 'std': 0.028, 'significant': True},
            'Eve Info (bits)': {'mean': 0.502, 'std': 0.031, 'significant': True}
        },
        'QBER-Adaptive': {
            'QBER': {'mean': 0.108, 'std': 0.005, 'significant': False},
            'Key Rate (bits/s)': {'mean': 218.7, 'std': 11.3, 'significant': False},
            'Detection Rate': {'mean': 0.456, 'std': 0.082, 'significant': True},
            'Eve Info (bits)': {'mean': 0.348, 'std': 0.027, 'significant': True}
        },
        'Basis-Learning': {
            'QBER': {'mean': 0.095, 'std': 0.007, 'significant': False},
            'Key Rate (bits/s)': {'mean': 235.1, 'std': 9.8, 'significant': False},
            'Detection Rate': {'mean': 0.382, 'std': 0.091, 'significant': False},
            'Eve Info (bits)': {'mean': 0.421, 'std': 0.035, 'significant': True}
        },
        'PNS-Attack': {
            'QBER': {'mean': 0.072, 'std': 0.004, 'significant': False},
            'Key Rate (bits/s)': {'mean': 268.4, 'std': 12.1, 'significant': False},
            'Detection Rate': {'mean': 0.215, 'std': 0.067, 'significant': False},
            'Eve Info (bits)': {'mean': 0.284, 'std': 0.022, 'significant': False}
        }
    }
    
    print("Generating LaTeX results table...")
    filepath = viz.export_results_table(
        results,
        filename="attack_comparison_results.tex",
        caption="Performance Comparison of Quantum Jamming Strategies Against BB84 QKD"
    )
    print(f"Saved: {filepath}")
    
    # Display table preview
    print("\nLaTeX Table Preview:")
    print("-" * 80)
    with open(filepath, 'r') as f:
        print(f.read())
    print("-" * 80)


def demo_integrated_workflow():
    """Demonstrate integrated workflow with actual simulations."""
    print("\n=== DEMO 6: Integrated Workflow Example ===\n")
    print("Running small BB84 simulations with different attacks...")
    
    backend = ClassicalBackend()
    viz = VisualizationManager(output_dir="demo_visualizations/integrated")
    
    strategies_info = {}
    
    # Run simulations with different strategies
    attack_configs = [
        ('Intercept-Resend', InterceptResendAttack(backend=backend, intercept_probability=0.5)),
        ('QBER-Adaptive', QBERAdaptiveStrategy(backend=backend, target_qber=0.10, kp=2.0, ki=0.5)),
    ]
    
    print("\nRunning simulations...")
    for name, attack in attack_configs:
        print(f"  - {name}...", end=" ")
        sim = AdaptiveJammingSimulator(
            n_qubits=500,
            attack_strategy=attack,
            backend=backend,
            loss_rate=0.05,
            error_rate=0.02
        )
        
        try:
            results = sim.run_simulation()
            strategies_info[name] = results.get('eve_info', 0.0)
            print(f"QBER={results.get('qber', 0):.3f}, Eve Info={results.get('eve_info', 0):.3f} bits")
        except Exception as e:
            print(f"Failed: {e}")
            strategies_info[name] = 0.0
    
    # Create visualization
    if strategies_info:
        print("\nGenerating information leakage comparison...")
        filepath = viz.plot_information_leakage(
            strategies_info,
            title="Information Leakage: Simulated Attack Results"
        )
        print(f"Saved: {filepath}")


def main():
    """Run all demonstration examples."""
    print("=" * 80)
    print("BB84 VISUALIZATION MANAGER DEMONSTRATION")
    print("=" * 80)
    print("\nThis script demonstrates all visualization and analysis capabilities")
    print("for quantum jamming attack research.\n")
    
    try:
        demo_basic_plots()
        demo_adaptive_dashboard()
        demo_atmospheric_effects()
        demo_statistical_analysis()
        demo_latex_export()
        demo_integrated_workflow()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("\nAll figures have been saved to: demo_visualizations/")
        print("\nYou can now:")
        print("  1. View the generated plots")
        print("  2. Use the LaTeX tables in your research papers")
        print("  3. Integrate these tools into your BB84 analysis workflow")
        print("\n")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
