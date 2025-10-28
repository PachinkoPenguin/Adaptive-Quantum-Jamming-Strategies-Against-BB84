"""
Unit Tests for Atmospheric Turbulence-Adaptive Attack Strategy
Tests ChannelAdaptiveStrategy and AtmosphericChannelModel
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main.bb84_main import (
    ClassicalBackend,
    Basis,
    ChannelAdaptiveStrategy,
    AtmosphericChannelModel,
    EveController,
    Alice,
    Bob,
    QuantumChannel
)


class TestAtmosphericChannelModel(unittest.TestCase):
    """Test atmospheric channel model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_day = AtmosphericChannelModel(
            distance_km=10.0,
            wavelength_nm=1550.0,
            time_of_day='day'
        )
        self.model_night = AtmosphericChannelModel(
            distance_km=10.0,
            wavelength_nm=1550.0,
            time_of_day='night'
        )
    
    def test_initialization_day(self):
        """Test day model initialization."""
        self.assertEqual(self.model_day.distance_km, 10.0)
        self.assertEqual(self.model_day.wavelength_nm, 1550.0)
        self.assertEqual(self.model_day.time_of_day, 'day')
        self.assertEqual(self.model_day.A, 1.7e-14)
        self.assertEqual(self.model_day.w, 27.0)
    
    def test_initialization_night(self):
        """Test night model initialization."""
        self.assertEqual(self.model_night.time_of_day, 'night')
        self.assertEqual(self.model_night.A, 1.28e-14)
        self.assertEqual(self.model_night.w, 21.0)
    
    def test_cn2_hufnagel_valley_ground(self):
        """Test Cn² at ground level."""
        cn2_ground = self.model_day.cn2_hufnagel_valley(0)
        
        # Should be positive
        self.assertGreater(cn2_ground, 0)
        
        # Should be on order of 1e-14 to 1e-10 (ground level can be strong)
        self.assertGreater(cn2_ground, 1e-15)
        self.assertLess(cn2_ground, 1e-9)
    
    def test_cn2_decreases_with_altitude(self):
        """Test that Cn² decreases with altitude."""
        cn2_0km = self.model_day.cn2_hufnagel_valley(0)
        cn2_5km = self.model_day.cn2_hufnagel_valley(5000)
        cn2_10km = self.model_day.cn2_hufnagel_valley(10000)
        
        self.assertGreater(cn2_0km, cn2_5km)
        self.assertGreater(cn2_5km, cn2_10km)
    
    def test_day_stronger_than_night(self):
        """Test that day turbulence is typically stronger."""
        cn2_day = self.model_day.cn2_hufnagel_valley(0)
        cn2_night = self.model_night.cn2_hufnagel_valley(0)
        
        self.assertGreater(cn2_day, cn2_night)
    
    def test_rytov_variance_calculation(self):
        """Test Rytov variance calculation."""
        cn2 = 1.7e-14
        rytov = self.model_day.calculate_rytov_variance(cn2)
        
        # Should be positive
        self.assertGreater(rytov, 0)
        
        # For 10km at 1550nm with typical Cn², expect strong turbulence
        self.assertGreater(rytov, 1.0)
    
    def test_rytov_increases_with_distance(self):
        """Test that Rytov variance increases with distance."""
        model_1km = AtmosphericChannelModel(distance_km=1.0)
        model_10km = AtmosphericChannelModel(distance_km=10.0)
        model_50km = AtmosphericChannelModel(distance_km=50.0)
        
        cn2 = 1.7e-14
        rytov_1 = model_1km.calculate_rytov_variance(cn2)
        rytov_10 = model_10km.calculate_rytov_variance(cn2)
        rytov_50 = model_50km.calculate_rytov_variance(cn2)
        
        self.assertGreater(rytov_10, rytov_1)
        self.assertGreater(rytov_50, rytov_10)
    
    def test_rytov_wavelength_dependence(self):
        """Test Rytov variance wavelength dependence."""
        model_850 = AtmosphericChannelModel(distance_km=10, wavelength_nm=850)
        model_1550 = AtmosphericChannelModel(distance_km=10, wavelength_nm=1550)
        
        cn2 = 1.7e-14
        rytov_850 = model_850.calculate_rytov_variance(cn2)
        rytov_1550 = model_1550.calculate_rytov_variance(cn2)
        
        # Shorter wavelength → stronger scattering
        self.assertGreater(rytov_850, rytov_1550)
    
    def test_turbulence_regime_classification(self):
        """Test turbulence regime classification."""
        self.assertEqual(self.model_day.get_turbulence_regime(0.3), 'very_weak')
        self.assertEqual(self.model_day.get_turbulence_regime(0.7), 'weak')
        self.assertEqual(self.model_day.get_turbulence_regime(1.5), 'moderate')
        self.assertEqual(self.model_day.get_turbulence_regime(5.0), 'strong')
    
    def test_phase_screen_generation(self):
        """Test phase screen generation."""
        screen = self.model_day.generate_phase_screen(grid_size=64, r0=0.1)
        
        # Check dimensions
        self.assertEqual(screen.shape, (64, 64))
        
        # Should be real values
        self.assertTrue(np.all(np.isreal(screen)))
        
        # Should have non-zero variance
        self.assertGreater(np.std(screen), 0)
    
    def test_phase_screen_r0_dependence(self):
        """Test phase screen dependence on r0."""
        screen_r0_small = self.model_day.generate_phase_screen(grid_size=64, r0=0.05)
        screen_r0_large = self.model_day.generate_phase_screen(grid_size=64, r0=0.2)
        
        # Smaller r0 → stronger turbulence → larger variance
        var_small = np.var(screen_r0_small)
        var_large = np.var(screen_r0_large)
        
        self.assertGreater(var_small, var_large)
    
    def test_scintillation_index_weak(self):
        """Test scintillation index in weak turbulence."""
        rytov = 0.5
        scint = self.model_day.get_scintillation_index(rytov)
        
        # In weak turbulence: σ²_I ≈ σ²_R
        self.assertAlmostEqual(scint, rytov, places=5)
    
    def test_scintillation_index_strong(self):
        """Test scintillation index in strong turbulence."""
        rytov = 3.0
        scint = self.model_day.get_scintillation_index(rytov)
        
        # In strong turbulence: σ²_I ≈ exp(σ²_R) - 1
        expected = np.exp(rytov) - 1
        self.assertAlmostEqual(scint, expected, places=2)


class TestChannelAdaptiveStrategy(unittest.TestCase):
    """Test channel-adaptive attack strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = ClassicalBackend()
        self.strategy = ChannelAdaptiveStrategy(
            self.backend,
            distance_km=10.0,
            wavelength_nm=1550.0,
            cn2=1.7e-14,
            time_of_day='day'
        )
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.distance_km, 10.0)
        self.assertEqual(self.strategy.wavelength_nm, 1550.0)
        self.assertEqual(self.strategy.cn2, 1.7e-14)
        self.assertEqual(self.strategy.time_of_day, 'day')
        
        # Should have calculated initial Rytov variance
        self.assertGreater(self.strategy.current_rytov, 0)
        self.assertIn(self.strategy.turbulence_regime, 
                     ['very_weak', 'weak', 'moderate', 'strong'])
    
    def test_update_rytov_variance(self):
        """Test Rytov variance update."""
        initial_rytov = self.strategy.current_rytov
        
        # Update with new Cn²
        new_cn2 = 3.0e-14
        self.strategy.update_rytov_variance(new_cn2)
        
        self.assertEqual(self.strategy.cn2, new_cn2)
        self.assertNotEqual(self.strategy.current_rytov, initial_rytov)
        self.assertGreater(len(self.strategy.rytov_history), 1)
    
    def test_intercept_probability_very_weak(self):
        """Test interception probability for very weak turbulence."""
        prob = self.strategy.get_intercept_probability(rytov_variance=0.3)
        self.assertEqual(prob, 0.1)
    
    def test_intercept_probability_weak(self):
        """Test interception probability for weak turbulence."""
        prob = self.strategy.get_intercept_probability(rytov_variance=0.7)
        self.assertEqual(prob, 0.2)
    
    def test_intercept_probability_moderate(self):
        """Test interception probability for moderate turbulence."""
        prob = self.strategy.get_intercept_probability(rytov_variance=2.0)
        self.assertEqual(prob, 0.4)
    
    def test_intercept_probability_strong(self):
        """Test interception probability for strong turbulence."""
        prob = self.strategy.get_intercept_probability(rytov_variance=5.0)
        self.assertEqual(prob, 0.7)
    
    def test_should_intercept_metadata_update(self):
        """Test that should_intercept updates from metadata."""
        initial_rytov = self.strategy.current_rytov
        
        # Provide new Cn² in metadata
        metadata = {'atmospheric_Cn2': 5.0e-14}
        decision = self.strategy.should_intercept(metadata)
        
        # Cn² should be updated
        self.assertEqual(self.strategy.cn2, 5.0e-14)
        self.assertNotEqual(self.strategy.current_rytov, initial_rytov)
        
        # Decision should be boolean
        self.assertIsInstance(decision, bool)
    
    def test_should_intercept_probability(self):
        """Test interception probability distribution."""
        np.random.seed(42)
        
        # Set specific Rytov variance for predictable probability
        self.strategy.current_rytov = 2.0  # Should give p=0.4
        
        decisions = [self.strategy.should_intercept({}) for _ in range(1000)]
        intercept_rate = sum(decisions) / len(decisions)
        
        # Should be close to 0.4
        self.assertAlmostEqual(intercept_rate, 0.4, delta=0.05)
    
    def test_intercept_qubit(self):
        """Test qubit interception."""
        qubit = self.backend.prepare_state(0, Basis.RECTILINEAR)
        modified_qubit, was_intercepted = self.strategy.intercept(qubit)
        
        self.assertTrue(was_intercepted)
        self.assertEqual(len(self.strategy.measured_bits), 1)
        self.assertEqual(len(self.strategy.measurement_bases), 1)
        self.assertEqual(self.strategy.statistics['interceptions'], 1)
    
    def test_intercept_with_alice_basis(self):
        """Test interception with Alice's basis known."""
        qubit = self.backend.prepare_state(1, Basis.RECTILINEAR)
        
        modified_qubit, was_intercepted = self.strategy.intercept(
            qubit,
            alice_basis=Basis.RECTILINEAR
        )
        
        self.assertTrue(was_intercepted)
        self.assertGreater(self.strategy.statistics['information_gained'], 0)
    
    def test_update_strategy_cn2(self):
        """Test strategy update with new Cn²."""
        initial_rytov = self.strategy.current_rytov
        
        feedback = {'atmospheric_Cn2': 2.0e-14}
        self.strategy.update_strategy(feedback)
        
        self.assertEqual(self.strategy.cn2, 2.0e-14)
        self.assertNotEqual(self.strategy.current_rytov, initial_rytov)
    
    def test_update_strategy_time_of_day(self):
        """Test strategy update with time of day change."""
        initial_time = self.strategy.time_of_day
        
        feedback = {'time_of_day': 'night'}
        self.strategy.update_strategy(feedback)
        
        self.assertEqual(self.strategy.time_of_day, 'night')
        self.assertNotEqual(self.strategy.time_of_day, initial_time)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        # Make some measurements
        for _ in range(5):
            qubit = self.backend.prepare_state(0, Basis.RECTILINEAR)
            self.strategy.intercept(qubit)
        
        stats = self.strategy.get_statistics()
        
        # Check required fields
        self.assertIn('distance_km', stats)
        self.assertIn('wavelength_nm', stats)
        self.assertIn('cn2', stats)
        self.assertIn('current_rytov_variance', stats)
        self.assertIn('current_turbulence_regime', stats)
        self.assertIn('current_intercept_probability', stats)
        self.assertIn('scintillation_index', stats)
        self.assertIn('rytov_history', stats)
        
        self.assertEqual(stats['total_measurements'], 5)
    
    def test_regime_tracking(self):
        """Test turbulence regime tracking over updates."""
        # Update multiple times with different Cn²
        cn2_values = [1e-14, 2e-14, 5e-14, 1e-14]
        
        for cn2 in cn2_values:
            self.strategy.update_rytov_variance(cn2)
        
        # Should have tracked all regimes
        self.assertEqual(len(self.strategy.regime_history), len(cn2_values) + 1)
        self.assertEqual(len(self.strategy.rytov_history), len(cn2_values) + 1)


class TestAtmosphericIntegration(unittest.TestCase):
    """Test atmospheric strategy in full BB84 protocol."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = ClassicalBackend()
    
    def test_atmospheric_in_bb84(self):
        """Test atmospheric strategy in full BB84 protocol."""
        # Setup
        strategy = ChannelAdaptiveStrategy(
            self.backend,
            distance_km=10.0,
            wavelength_nm=1550,
            cn2=1.7e-14
        )
        eve = EveController(strategy, self.backend)
        channel = QuantumChannel(self.backend, eve=eve)
        
        alice = Alice(self.backend)
        bob = Bob(self.backend)
        
        # Run protocol
        n_qubits = 50
        alice.generate_random_bits(n_qubits)
        alice.choose_random_bases(n_qubits)
        states = alice.prepare_states()
        
        # Transmit with Eve
        received = channel.transmit(states)
        
        # Bob measures
        bob.choose_random_bases(len(received))
        bob.measure_states(received)
        
        # Check Eve statistics
        eve_stats = eve.get_statistics()
        strategy_stats = eve_stats['attack_strategy_stats']
        
        self.assertIn('current_rytov_variance', strategy_stats)
        self.assertIn('current_turbulence_regime', strategy_stats)
        self.assertGreater(strategy_stats['current_rytov_variance'], 0)
    
    def test_varying_atmospheric_conditions(self):
        """Test strategy adaptation to varying conditions."""
        strategy = ChannelAdaptiveStrategy(self.backend, distance_km=10)
        
        # Simulate varying Cn² with wider range to ensure different probabilities
        cn2_sequence = [1e-15, 1e-14, 3e-14, 5e-14]
        probs = []
        
        for cn2 in cn2_sequence:
            strategy.update_rytov_variance(cn2)
            prob = strategy.get_intercept_probability()
            probs.append(prob)
        
        # Probabilities should vary with conditions
        # With this range, we should get at least 2 different probability levels
        unique_probs = len(set(probs))
        self.assertGreater(unique_probs, 1)
        
        # History should be tracked
        self.assertEqual(len(strategy.rytov_history), len(cn2_sequence) + 1)
    
    def test_extreme_distances(self):
        """Test strategy at extreme distances."""
        # Very short distance
        strategy_short = ChannelAdaptiveStrategy(
            self.backend,
            distance_km=0.1,
            cn2=1.7e-14
        )
        
        # Very long distance
        strategy_long = ChannelAdaptiveStrategy(
            self.backend,
            distance_km=100.0,
            cn2=1.7e-14
        )
        
        # Long distance should have higher Rytov variance
        self.assertGreater(
            strategy_long.current_rytov,
            strategy_short.current_rytov
        )
        
        # Long distance should have higher intercept probability
        self.assertGreater(
            strategy_long.get_intercept_probability(),
            strategy_short.get_intercept_probability()
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
