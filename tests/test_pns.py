"""
Unit Tests for Photon Number Splitting (PNS) Attack
Tests PhotonNumberSplittingAttack strategy implementation
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main.bb84_main import (
    ClassicalBackend,
    Basis,
    PhotonNumberSplittingAttack,
    EveController,
    Alice,
    Bob,
    QuantumChannel
)


class TestPhotonNumberSplittingAttack(unittest.TestCase):
    """Test PNS attack strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = ClassicalBackend()
        self.strategy = PhotonNumberSplittingAttack(
            self.backend,
            mean_photon_number=0.1
        )
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.mean_photon_number, 0.1)
        self.assertEqual(len(self.strategy.stored_photons), 0)
        self.assertEqual(len(self.strategy.final_measurements), 0)
        self.assertEqual(self.strategy.pulses_monitored, 0)
        self.assertEqual(self.strategy.multi_photon_pulses, 0)
    
    def test_probability_multi_photon(self):
        """Test multi-photon probability calculation."""
        # For μ=0.1
        prob = PhotonNumberSplittingAttack.probability_multi_photon(0.1)
        expected = 1.0 - (1.0 + 0.1) * np.exp(-0.1)
        self.assertAlmostEqual(prob, expected, places=6)
        self.assertAlmostEqual(prob, 0.0047, places=4)
        
        # For μ=0.5
        prob = PhotonNumberSplittingAttack.probability_multi_photon(0.5)
        expected = 1.0 - (1.0 + 0.5) * np.exp(-0.5)
        self.assertAlmostEqual(prob, expected, places=6)
        self.assertAlmostEqual(prob, 0.0902, places=4)
    
    def test_expected_information_gain(self):
        """Test expected information gain calculation."""
        # Information gain equals multi-photon probability
        mu = 0.1
        info = PhotonNumberSplittingAttack.expected_information_gain(mu)
        prob = PhotonNumberSplittingAttack.probability_multi_photon(mu)
        self.assertAlmostEqual(info, prob, places=10)
    
    def test_optimal_mu_for_distance(self):
        """Test optimal μ calculation."""
        # For 10 dB loss: transmittance = 10^(-10/10) = 0.1
        mu = PhotonNumberSplittingAttack.optimal_mu_for_distance(10.0)
        self.assertAlmostEqual(mu, 0.1, places=4)
        
        # For 20 dB loss: transmittance = 0.01
        mu = PhotonNumberSplittingAttack.optimal_mu_for_distance(20.0)
        self.assertAlmostEqual(mu, 0.01, places=4)
        
        # For 0 dB loss: transmittance = 1.0
        mu = PhotonNumberSplittingAttack.optimal_mu_for_distance(0.0)
        self.assertAlmostEqual(mu, 1.0, places=4)
    
    def test_should_intercept_always_true(self):
        """Test that PNS always monitors."""
        # PNS is non-destructive, always monitor
        self.assertTrue(self.strategy.should_intercept({}))
        self.assertTrue(self.strategy.should_intercept({'any': 'metadata'}))
        self.assertTrue(self.strategy.should_intercept(None))
    
    def test_intercept_single_photon(self):
        """Test interception with single photon (no splitting)."""
        np.random.seed(42)
        
        # Prepare qubit
        qubit = self.backend.prepare_state(0, Basis.RECTILINEAR)
        
        # Simulate multiple intercepts (most will be single photon)
        single_photon_count = 0
        multi_photon_count = 0
        
        for _ in range(1000):
            strategy = PhotonNumberSplittingAttack(self.backend, mean_photon_number=0.1)
            modified_qubit, was_intercepted = strategy.intercept(qubit)
            
            if was_intercepted:
                multi_photon_count += 1
            else:
                single_photon_count += 1
        
        # Should match Poisson statistics approximately
        expected_multi = PhotonNumberSplittingAttack.probability_multi_photon(0.1) * 1000
        self.assertAlmostEqual(multi_photon_count, expected_multi, delta=30)
    
    def test_intercept_multi_photon(self):
        """Test interception with multi-photon pulse."""
        np.random.seed(123)  # Seed for reproducibility
        
        # Use higher μ for more multi-photon events
        strategy = PhotonNumberSplittingAttack(self.backend, mean_photon_number=0.5)
        qubit = self.backend.prepare_state(1, Basis.DIAGONAL)
        
        # Run multiple times to catch multi-photon events
        intercepted_count = 0
        for _ in range(100):
            modified_qubit, was_intercepted = strategy.intercept(qubit)
            if was_intercepted:
                intercepted_count += 1
        
        # Should have some interceptions
        self.assertGreater(intercepted_count, 0)
        self.assertGreater(strategy.multi_photon_pulses, 0)
        self.assertEqual(strategy.photons_stored, intercepted_count)
    
    def test_intercept_updates_statistics(self):
        """Test that intercept updates statistics correctly."""
        np.random.seed(42)
        
        qubit = self.backend.prepare_state(0, Basis.RECTILINEAR)
        
        initial_monitored = self.strategy.pulses_monitored
        initial_interceptions = self.strategy.statistics['interceptions']
        
        # Monitor many pulses
        for _ in range(100):
            self.strategy.intercept(qubit)
        
        # Pulses monitored should increase
        self.assertEqual(self.strategy.pulses_monitored, initial_monitored + 100)
        
        # Should have some interceptions (with μ=0.1, expect ~0.47%)
        self.assertGreater(self.strategy.statistics['interceptions'], initial_interceptions)
    
    def test_update_strategy_without_bases(self):
        """Test update with no basis information."""
        initial_stored = len(self.strategy.stored_photons)
        
        # Update without alice_bases
        self.strategy.update_strategy(None)
        self.strategy.update_strategy({})
        self.strategy.update_strategy({'other': 'data'})
        
        # Stored photons should remain unchanged
        self.assertEqual(len(self.strategy.stored_photons), initial_stored)
    
    def test_update_strategy_with_bases(self):
        """Test measurement after basis announcement."""
        np.random.seed(42)
        
        # Intercept many pulses to get some multi-photon events
        qubits = []
        for i in range(1000):
            qubit = self.backend.prepare_state(i % 2, Basis.RECTILINEAR)
            qubits.append(qubit)
            self.strategy.intercept(qubit)
        
        initial_stored = len(self.strategy.stored_photons)
        self.assertGreater(initial_stored, 0)
        
        # Announce bases (Eve can now measure!)
        alice_bases = [Basis.RECTILINEAR] * 1000
        self.strategy.update_strategy({'alice_bases': alice_bases})
        
        # Stored photons should be measured and cleared
        self.assertEqual(len(self.strategy.stored_photons), 0)
        self.assertEqual(len(self.strategy.final_measurements), initial_stored)
        self.assertEqual(self.strategy.photons_measured, initial_stored)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        stats = self.strategy.get_statistics()
        
        # Check required fields
        self.assertIn('attack_type', stats)
        self.assertIn('mean_photon_number', stats)
        self.assertIn('pulses_monitored', stats)
        self.assertIn('multi_photon_pulses', stats)
        self.assertIn('multi_photon_probability_actual', stats)
        self.assertIn('multi_photon_probability_theory', stats)
        self.assertIn('photons_stored', stats)
        self.assertIn('photons_measured', stats)
        self.assertIn('successful_extractions', stats)
        self.assertIn('information_per_pulse_actual', stats)
        self.assertIn('information_per_pulse_expected', stats)
        self.assertIn('detection_probability', stats)
        self.assertIn('qber_introduced', stats)
        
        self.assertEqual(stats['attack_type'], 'Photon Number Splitting (PNS)')
        self.assertEqual(stats['mean_photon_number'], 0.1)
        self.assertEqual(stats['detection_probability'], 0.0)
        self.assertEqual(stats['qber_introduced'], 0.0)
    
    def test_zero_qber(self):
        """Test that PNS introduces zero QBER."""
        np.random.seed(42)
        
        # Run attack
        for _ in range(100):
            qubit = self.backend.prepare_state(0, Basis.RECTILINEAR)
            self.strategy.intercept(qubit)
        
        stats = self.strategy.get_statistics()
        
        # PNS should introduce ZERO QBER
        self.assertEqual(stats['qber_introduced'], 0.0)
        self.assertEqual(stats['detection_probability'], 0.0)
    
    def test_information_gain_scaling(self):
        """Test that information gain scales with μ."""
        results = []
        
        for mu in [0.05, 0.1, 0.2, 0.5]:
            strategy = PhotonNumberSplittingAttack(self.backend, mean_photon_number=mu)
            stats = strategy.get_statistics()
            
            results.append({
                'mu': mu,
                'expected_info': stats['information_per_pulse_expected']
            })
        
        # Information gain should increase with μ
        for i in range(len(results) - 1):
            self.assertLess(results[i]['expected_info'], 
                          results[i+1]['expected_info'])


class TestPNSIntegration(unittest.TestCase):
    """Test PNS attack in full BB84 protocol."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = ClassicalBackend()
    
    def test_pns_in_bb84(self):
        """Test PNS attack in full BB84 protocol."""
        np.random.seed(42)
        
        # Setup
        strategy = PhotonNumberSplittingAttack(self.backend, mean_photon_number=0.1)
        eve = EveController(strategy, self.backend)
        channel = QuantumChannel(self.backend, eve=eve)
        
        alice = Alice(self.backend)
        bob = Bob(self.backend)
        
        # Run protocol
        n_qubits = 500
        alice.generate_random_bits(n_qubits)
        alice.choose_random_bases(n_qubits)
        states = alice.prepare_states()
        
        # Transmit with Eve monitoring
        received = channel.transmit(states)
        
        # Bob measures
        bob.choose_random_bases(len(received))
        bob.measure_states(received)
        
        # Check Eve's statistics
        eve_stats = eve.get_statistics()
        strategy_stats = strategy.get_statistics()
        
        self.assertEqual(strategy_stats['pulses_monitored'], n_qubits)
        self.assertGreater(strategy_stats['multi_photon_pulses'], 0)
        self.assertEqual(strategy_stats['qber_introduced'], 0.0)
    
    def test_pns_with_basis_reconciliation(self):
        """Test PNS measurement after basis reconciliation."""
        np.random.seed(123)
        
        strategy = PhotonNumberSplittingAttack(self.backend, mean_photon_number=0.2)
        eve = EveController(strategy, self.backend)
        channel = QuantumChannel(self.backend, eve=eve)
        
        alice = Alice(self.backend)
        bob = Bob(self.backend)
        
        # Protocol
        n_qubits = 1000
        alice.generate_random_bits(n_qubits)
        alice.choose_random_bases(n_qubits)
        states = alice.prepare_states()
        
        received = channel.transmit(states)
        bob.choose_random_bases(len(received))
        bob.measure_states(received)
        
        # Before basis announcement
        stored_before = len(strategy.stored_photons)
        measured_before = len(strategy.final_measurements)
        
        # Basis reconciliation (public announcement)
        strategy.update_strategy({'alice_bases': alice.bases})
        
        # After basis announcement
        stored_after = len(strategy.stored_photons)
        measured_after = len(strategy.final_measurements)
        
        # Eve should have measured stored photons
        self.assertEqual(stored_after, 0)
        self.assertGreater(measured_after, measured_before)
        self.assertEqual(measured_after, stored_before)
    
    def test_pns_efficiency(self):
        """Test PNS attack efficiency."""
        np.random.seed(42)
        
        strategy = PhotonNumberSplittingAttack(self.backend, mean_photon_number=0.1)
        
        # Run attack
        for _ in range(1000):
            qubit = self.backend.prepare_state(0, Basis.RECTILINEAR)
            strategy.intercept(qubit)
        
        # Update with bases
        strategy.update_strategy({'alice_bases': [Basis.RECTILINEAR] * 1000})
        
        stats = strategy.get_statistics()
        
        # Efficiency should be 100% (all stored photons measured successfully)
        if stats['multi_photon_pulses'] > 0:
            self.assertEqual(stats['efficiency'], 1.0)
    
    def test_pns_vs_higher_mu(self):
        """Test that higher μ leads to more vulnerability."""
        np.random.seed(42)
        
        results = []
        
        for mu in [0.1, 0.5]:
            strategy = PhotonNumberSplittingAttack(self.backend, mean_photon_number=mu)
            
            for _ in range(1000):
                qubit = self.backend.prepare_state(0, Basis.RECTILINEAR)
                strategy.intercept(qubit)
            
            strategy.update_strategy({'alice_bases': [Basis.RECTILINEAR] * 1000})
            stats = strategy.get_statistics()
            
            results.append({
                'mu': mu,
                'extractions': stats['successful_extractions']
            })
        
        # Higher μ should result in more extractions
        self.assertGreater(results[1]['extractions'], results[0]['extractions'])


class TestPNSStatistics(unittest.TestCase):
    """Test PNS attack statistics and probabilities."""
    
    def test_poisson_statistics(self):
        """Test Poisson distribution statistics."""
        # Verify probability calculations match Poisson formula
        for mu in [0.1, 0.2, 0.5]:
            prob = PhotonNumberSplittingAttack.probability_multi_photon(mu)
            
            # P(n≥2) = 1 - P(n=0) - P(n=1) = 1 - (1+μ)e^(-μ)
            expected = 1.0 - (1.0 + mu) * np.exp(-mu)
            
            self.assertAlmostEqual(prob, expected, places=10)
    
    def test_information_theory(self):
        """Test information theory calculations."""
        # Information gain should equal multi-photon probability
        for mu in [0.05, 0.1, 0.2, 0.5, 1.0]:
            info = PhotonNumberSplittingAttack.expected_information_gain(mu)
            prob = PhotonNumberSplittingAttack.probability_multi_photon(mu)
            
            # Eve gains 1 full bit per multi-photon pulse
            self.assertAlmostEqual(info, prob, places=10)
    
    def test_channel_loss_scaling(self):
        """Test optimal μ scaling with channel loss."""
        # Optimal μ should decrease exponentially with loss
        losses = [0, 5, 10, 15, 20]
        optimal_mus = [PhotonNumberSplittingAttack.optimal_mu_for_distance(loss) 
                      for loss in losses]
        
        # Each 10 dB should reduce by factor of 10
        self.assertAlmostEqual(optimal_mus[0] / optimal_mus[2], 10.0, places=4)
        self.assertAlmostEqual(optimal_mus[2] / optimal_mus[4], 10.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
