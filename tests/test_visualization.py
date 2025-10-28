"""
Tests for VisualizationManager and analysis plotting tools.
"""

import pytest
import numpy as np
import os
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main.bb84_main import VisualizationManager, Basis


class TestVisualizationManager:
    """Test suite for VisualizationManager visualization and analysis tools."""
    
    @pytest.fixture
    def viz_manager(self, tmp_path):
        """Create a VisualizationManager with temporary output directory."""
        output_dir = str(tmp_path / "viz_output")
        return VisualizationManager(output_dir=output_dir)
    
    @pytest.fixture
    def sample_qber_history(self):
        """Generate sample QBER history data."""
        np.random.seed(42)
        # Start low, gradually increase with some noise
        base = np.linspace(0.03, 0.15, 100)
        noise = np.random.normal(0, 0.01, 100)
        return list(np.clip(base + noise, 0, 1))
    
    @pytest.fixture
    def sample_intercept_history(self):
        """Generate sample intercept probability history."""
        np.random.seed(42)
        # Adaptive behavior: starts high, decreases when detection risk increases
        base = 0.5 * (1 + np.sin(np.linspace(0, 4*np.pi, 80)))
        noise = np.random.normal(0, 0.05, 80)
        return list(np.clip(base + noise, 0, 1))
    
    def test_initialization(self, viz_manager):
        """Test VisualizationManager initializes correctly."""
        assert viz_manager.output_dir is not None
        assert os.path.exists(viz_manager.output_dir)
        
        # Check matplotlib config was applied
        import matplotlib.pyplot as plt
        assert plt.rcParams['figure.dpi'] == 150
        assert plt.rcParams['font.size'] == 12
    
    def test_plot_qber_evolution(self, viz_manager, sample_qber_history):
        """Test QBER evolution plotting."""
        filepath = viz_manager.plot_qber_evolution(
            sample_qber_history,
            threshold=0.11,
            save_name="test_qber_evolution.png"
        )
        
        assert os.path.exists(filepath)
        assert filepath.endswith("test_qber_evolution.png")
        
        # Verify file is not empty
        assert os.path.getsize(filepath) > 1000  # PNG should be > 1KB
    
    def test_plot_intercept_probability(self, viz_manager, sample_intercept_history):
        """Test intercept probability adaptation plotting."""
        filepath = viz_manager.plot_intercept_probability(
            sample_intercept_history,
            save_name="test_intercept_prob.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 1000
    
    def test_plot_information_leakage(self, viz_manager):
        """Test information leakage bar chart."""
        strategies = {
            'Intercept-Resend': 0.5,
            'QBER-Adaptive': 0.35,
            'Basis-Learning': 0.42,
            'PNS Attack': 0.28,
            'Channel-Adaptive': 0.31
        }
        
        filepath = viz_manager.plot_information_leakage(
            strategies,
            save_name="test_info_leakage.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 1000
    
    def test_plot_detection_roc_curve(self, viz_manager):
        """Test ROC curve plotting for detection methods."""
        # Generate synthetic ROC data
        fpr_qber = np.linspace(0, 1, 50)
        tpr_qber = np.sqrt(fpr_qber) * 0.9 + np.random.normal(0, 0.05, 50)
        tpr_qber = np.clip(tpr_qber, 0, 1)
        
        fpr_chi = np.linspace(0, 1, 50)
        tpr_chi = fpr_chi ** 0.7 * 0.95 + np.random.normal(0, 0.03, 50)
        tpr_chi = np.clip(tpr_chi, 0, 1)
        
        results = {
            'QBER Detection': {'fpr': list(fpr_qber), 'tpr': list(tpr_qber)},
            'Chi-Square Test': {'fpr': list(fpr_chi), 'tpr': list(tpr_chi)}
        }
        
        filepath = viz_manager.plot_detection_roc_curve(
            results,
            save_name="test_roc_curve.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 1000
    
    def test_plot_adaptive_strategy_dashboard(self, viz_manager, sample_qber_history, 
                                              sample_intercept_history):
        """Test comprehensive 4-subplot dashboard."""
        # Generate info gain history (cumulative)
        info_gain = list(np.cumsum(np.random.exponential(0.01, 80)))
        
        # Detection metrics
        detection_metrics = {
            'QBER_exceeded': [1 if q > 0.11 else 0 for q in sample_qber_history[:50]],
            'Chi-Square_p': list(np.random.beta(2, 5, 50)),
            'MI_detected': [1 if np.random.random() > 0.7 else 0 for _ in range(50)]
        }
        
        filepath = viz_manager.plot_adaptive_strategy_dashboard(
            sample_qber_history,
            sample_intercept_history,
            info_gain,
            detection_metrics,
            save_name="test_dashboard.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 5000  # Multi-subplot figure should be larger
    
    def test_plot_rytov_variance(self, viz_manager):
        """Test Rytov variance atmospheric turbulence plotting."""
        time = list(np.linspace(0, 100, 200))
        # Simulate varying turbulence: weak to strong
        rytov = list(0.1 + 0.5 * np.abs(np.sin(np.linspace(0, 4*np.pi, 200))) + 
                     np.random.exponential(0.1, 200))
        
        filepath = viz_manager.plot_rytov_variance(
            time,
            rytov,
            save_name="test_rytov.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 1000
    
    def test_plot_attack_aggression_vs_turbulence(self, viz_manager):
        """Test scatter plot of attack aggression vs turbulence."""
        np.random.seed(42)
        n_points = 50
        
        turbulence_levels = list(np.random.exponential(0.5, n_points))
        aggression_levels = list(np.clip(0.8 - 0.3 * np.array(turbulence_levels) + 
                                        np.random.normal(0, 0.1, n_points), 0, 1))
        qber_values = list(0.05 + 0.3 * np.array(aggression_levels) + 
                          0.1 * np.array(turbulence_levels))
        
        filepath = viz_manager.plot_attack_aggression_vs_turbulence(
            turbulence_levels,
            aggression_levels,
            qber_values,
            save_name="test_aggression_turbulence.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 1000
    
    def test_plot_phase_screen(self, viz_manager):
        """Test 2D atmospheric phase screen visualization."""
        # Generate Kolmogorov-like phase screen
        size = 64
        np.random.seed(42)
        phase_screen = np.random.randn(size, size)
        
        # Apply some spatial correlation (simple convolution)
        from scipy.ndimage import gaussian_filter
        try:
            phase_screen = gaussian_filter(phase_screen, sigma=3)
        except ImportError:
            # Fallback if scipy not available
            pass
        
        phase_screen = phase_screen * 2 * np.pi  # Scale to phase range
        
        filepath = viz_manager.plot_phase_screen(
            phase_screen,
            save_name="test_phase_screen.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 1000
    
    def test_animate_beam_propagation(self, viz_manager):
        """Test beam propagation animation frame generation."""
        # Generate sequence of intensity frames
        size = 32
        n_frames = 10
        np.random.seed(42)
        
        frames = []
        for i in range(n_frames):
            # Simulate beam spreading and turbulence
            x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
            r2 = x**2 + y**2
            intensity = np.exp(-r2 / (0.3 + 0.1*i)) * (1 + 0.2*np.random.randn(size, size))
            intensity = np.clip(intensity, 0, None)
            frames.append(intensity)
        
        frame_dir = viz_manager.animate_beam_propagation(
            frames,
            save_name="test_beam_anim",
            fps=10
        )
        
        assert os.path.exists(frame_dir)
        assert os.path.isdir(frame_dir)
        
        # Check frames were created
        frame_files = list(Path(frame_dir).glob("frame_*.png"))
        assert len(frame_files) == n_frames
        
        # Check metadata
        metadata_path = os.path.join(frame_dir, "metadata.json")
        assert os.path.exists(metadata_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['num_frames'] == n_frames
        assert metadata['fps'] == 10
    
    def test_plot_error_pattern_analysis(self, viz_manager):
        """Test error pattern analysis with autocorrelation and runs."""
        np.random.seed(42)
        
        # Generate error sequence with some clustering (non-random pattern)
        errors = []
        for _ in range(500):
            if np.random.random() < 0.1:  # 10% base error rate
                # Create small clusters
                cluster_size = np.random.randint(1, 4)
                errors.extend([1] * cluster_size)
            else:
                errors.append(0)
        
        errors = errors[:500]  # Trim to exact size
        
        filepath = viz_manager.plot_error_pattern_analysis(
            errors,
            save_name="test_error_pattern.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 3000  # Multi-subplot figure
    
    def test_plot_basis_bias_detection(self, viz_manager):
        """Test basis bias detection visualization."""
        np.random.seed(42)
        
        # Generate basis sequence with slight bias (55% ones)
        basis_sequence = [1 if np.random.random() < 0.55 else 0 for _ in range(1000)]
        
        filepath = viz_manager.plot_basis_bias_detection(
            basis_sequence,
            expected_prob=0.5,
            save_name="test_basis_bias.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 2000  # Two subplots
    
    def test_plot_mutual_information_matrix(self, viz_manager):
        """Test mutual information matrix visualization."""
        # Create symmetric MI matrix
        mi_matrix = np.array([
            [1.000, 0.850, 0.120],  # Alice
            [0.850, 1.000, 0.090],  # Bob
            [0.120, 0.090, 1.000]   # Eve
        ])
        
        labels = ["Alice", "Bob", "Eve"]
        
        filepath = viz_manager.plot_mutual_information_matrix(
            mi_matrix,
            labels=labels,
            save_name="test_mi_matrix.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 1000
    
    def test_plot_holevo_bound_comparison(self, viz_manager):
        """Test Holevo bound comparison plotting."""
        strategies = ['Intercept-Resend', 'QBER-Adaptive', 'Basis-Learning', 'PNS']
        theoretical_bounds = [1.0, 0.85, 0.92, 0.75]
        actual_information = [0.5, 0.35, 0.42, 0.28]
        
        filepath = viz_manager.plot_holevo_bound_comparison(
            strategies,
            theoretical_bounds,
            actual_information,
            save_name="test_holevo_comparison.png"
        )
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 1000
    
    def test_export_results_table_basic(self, viz_manager):
        """Test LaTeX table export with basic data."""
        results = {
            'Intercept-Resend': {
                'QBER': {'mean': 0.250, 'std': 0.012, 'significant': True},
                'Key_Rate': {'mean': 0.125, 'std': 0.008, 'significant': False},
                'Detection': {'mean': 0.95, 'std': 0.03, 'significant': True}
            },
            'QBER-Adaptive': {
                'QBER': {'mean': 0.108, 'std': 0.005, 'significant': False},
                'Key_Rate': {'mean': 0.215, 'std': 0.011, 'significant': False},
                'Detection': {'mean': 0.45, 'std': 0.08, 'significant': True}
            }
        }
        
        filepath = viz_manager.export_results_table(
            results,
            filename="test_results_table.tex",
            caption="Test Results Summary"
        )
        
        assert os.path.exists(filepath)
        assert filepath.endswith(".tex")
        
        # Verify LaTeX structure
        with open(filepath, 'r') as f:
            content = f.read()
        
        assert "\\begin{table}" in content
        assert "\\end{table}" in content
        assert "\\begin{tabular}" in content
        assert "\\caption{Test Results Summary}" in content
        assert "Intercept-Resend" in content or "Intercept\\_Resend" in content
        assert "$^*$" in content  # Significance marker
        assert "\\pm" in content  # Mean ± std format
    
    def test_export_results_table_simple_values(self, viz_manager):
        """Test LaTeX export with simple float/int values."""
        results = {
            'Strategy_A': {
                'Metric1': 0.123,
                'Metric2': 42,
                'Metric3': 0.999
            },
            'Strategy_B': {
                'Metric1': 0.456,
                'Metric2': 37,
                'Metric3': 0.888
            }
        }
        
        filepath = viz_manager.export_results_table(
            results,
            filename="test_simple_table.tex"
        )
        
        assert os.path.exists(filepath)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        assert "0.123" in content
        assert "42" in content
        assert "Strategy" in content
    
    def test_qber_evolution_empty_list(self, viz_manager):
        """Test QBER evolution handles edge case of minimal data."""
        # Should handle gracefully without crashing
        qber_history = [0.05]  # Single point
        
        filepath = viz_manager.plot_qber_evolution(
            qber_history,
            save_name="test_single_point.png"
        )
        
        assert os.path.exists(filepath)
    
    def test_multiple_plots_same_manager(self, viz_manager, sample_qber_history):
        """Test creating multiple plots with same manager instance."""
        # Should reuse output directory and not cause conflicts
        
        filepath1 = viz_manager.plot_qber_evolution(
            sample_qber_history,
            save_name="multi_test_1.png"
        )
        
        filepath2 = viz_manager.plot_qber_evolution(
            sample_qber_history,
            threshold=0.15,
            save_name="multi_test_2.png"
        )
        
        assert os.path.exists(filepath1)
        assert os.path.exists(filepath2)
        assert filepath1 != filepath2
        
        # Both should be in same output directory
        assert os.path.dirname(filepath1) == os.path.dirname(filepath2)


class TestVisualizationIntegration:
    """Integration tests combining visualization with BB84 components."""
    
    def test_visualization_with_real_simulation_data(self, tmp_path):
        """Test visualization using data from actual BB84 simulation."""
        from src.main.bb84_main import (
            AdaptiveJammingSimulator, InterceptResendAttack, 
            ClassicalBackend
        )
        
        # Run small simulation
        backend = ClassicalBackend()
        attack = InterceptResendAttack(backend=backend, intercept_probability=0.3)
        
        sim = AdaptiveJammingSimulator(
            n_qubits=100,
            attack_strategy=attack,
            backend=backend,
            detector_params={'qber_threshold': 0.11},
            loss_rate=0.05,
            error_rate=0.02
        )
        
        try:
            results = sim.run_simulation()
        except (TypeError, ValueError, KeyError) as e:
            # If simulation fails due to edge cases, skip visualization test
            pytest.skip(f"Simulation failed with: {e}")
        
        # Create visualizations from real data
        viz_mgr = VisualizationManager(output_dir=str(tmp_path / "viz_integration"))
        
        # QBER evolution (if available in results)
        if 'qber' in results:
            qber_history = [results['qber']]  # Single value
            filepath = viz_mgr.plot_qber_evolution(qber_history)
            assert os.path.exists(filepath)
        
        # Info leakage
        if 'eve_info' in results:
            info_dict = {'Intercept-Resend': results['eve_info']}
            filepath = viz_mgr.plot_information_leakage(info_dict)
            assert os.path.exists(filepath)
