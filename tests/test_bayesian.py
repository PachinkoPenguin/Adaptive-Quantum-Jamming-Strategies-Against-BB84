"""
Unit Tests for Bayesian Inference Attack Strategies
Tests both BasisLearningStrategy and ParticleFilterBasisLearner
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main.bb84_main import (
    ClassicalBackend,
    Basis,
    BasisLearningStrategy,
    ParticleFilterBasisLearner,
    EveController,
    Alice,
    Bob,
    QuantumChannel
)


class TestBasisLearningStrategy(unittest.TestCase):
    """Test Beta distribution Bayesian learning strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = ClassicalBackend()
        self.strategy = BasisLearningStrategy(
            self.backend,
            base_intercept_prob=0.3,
            confidence_threshold=0.8
        )
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.base_intercept_prob, 0.3)
        self.assertEqual(self.strategy.confidence_threshold, 0.8)
        self.assertEqual(self.strategy.alpha, 1.0)
        self.assertEqual(self.strategy.beta, 1.0)
        self.assertEqual(len(self.strategy.observations), 0)
        self.assertEqual(self.strategy.correct_predictions, 0)
        self.assertEqual(self.strategy.total_predictions, 0)
    
    def test_observe_basis_rectilinear(self):
        """Test observation of rectilinear basis."""
        initial_alpha = self.strategy.alpha
        self.strategy.observe_basis(Basis.RECTILINEAR)
        self.assertEqual(self.strategy.alpha, initial_alpha + 1.0)
        self.assertEqual(len(self.strategy.observations), 1)
        self.assertEqual(self.strategy.observations[0], Basis.RECTILINEAR)
    
    def test_observe_basis_diagonal(self):
        """Test observation of diagonal basis."""
        initial_beta = self.strategy.beta
        self.strategy.observe_basis(Basis.DIAGONAL)
        self.assertEqual(self.strategy.beta, initial_beta + 1.0)
        self.assertEqual(len(self.strategy.observations), 1)
        self.assertEqual(self.strategy.observations[0], Basis.DIAGONAL)
    
    def test_multiple_observations(self):
        """Test multiple basis observations."""
        # Observe 10 rectilinear, 5 diagonal
        for _ in range(10):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        for _ in range(5):
            self.strategy.observe_basis(Basis.DIAGONAL)
        
        self.assertEqual(self.strategy.alpha, 11.0)  # 1 + 10
        self.assertEqual(self.strategy.beta, 6.0)    # 1 + 5
        self.assertEqual(len(self.strategy.observations), 15)
    
    def test_predict_basis(self):
        """Test basis prediction."""
        # With uniform prior, prediction should be random
        prediction = self.strategy.predict_basis()
        self.assertIn(prediction, [Basis.RECTILINEAR, Basis.DIAGONAL])
        
        # After observing mostly rectilinear, should predict rectilinear
        for _ in range(20):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        for _ in range(5):
            self.strategy.observe_basis(Basis.DIAGONAL)
        
        prediction = self.strategy.predict_basis()
        self.assertEqual(prediction, Basis.RECTILINEAR)
    
    def test_get_confidence(self):
        """Test confidence calculation."""
        # Initial confidence should be low (uniform prior)
        initial_conf = self.strategy.get_confidence()
        self.assertLess(initial_conf, 0.5)
        
        # After many observations, confidence should increase
        for _ in range(100):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        
        final_conf = self.strategy.get_confidence()
        self.assertGreater(final_conf, initial_conf)
        self.assertGreater(final_conf, 0.9)
        
        # Confidence should be in [0, 1]
        self.assertGreaterEqual(final_conf, 0.0)
        self.assertLessEqual(final_conf, 1.0)
    
    def test_get_basis_probability(self):
        """Test basis probability calculation."""
        # Initially should be 0.5 (uniform prior)
        prob_rect = self.strategy.get_basis_probability(Basis.RECTILINEAR)
        self.assertAlmostEqual(prob_rect, 0.5, places=2)
        
        # After observing 60% rectilinear
        for _ in range(60):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        for _ in range(40):
            self.strategy.observe_basis(Basis.DIAGONAL)
        
        prob_rect = self.strategy.get_basis_probability(Basis.RECTILINEAR)
        prob_diag = self.strategy.get_basis_probability(Basis.DIAGONAL)
        
        # Should converge to ~0.6
        self.assertAlmostEqual(prob_rect, 0.6, places=1)
        self.assertAlmostEqual(prob_diag, 0.4, places=1)
        
        # Probabilities should sum to 1
        self.assertAlmostEqual(prob_rect + prob_diag, 1.0, places=5)
    
    def test_mutual_information(self):
        """Test mutual information calculation."""
        # Maximum entropy at p=0.5
        mi_half = self.strategy.mutual_information(0.5)
        self.assertAlmostEqual(mi_half, 1.0, places=5)
        
        # Zero entropy at extremes
        mi_zero = self.strategy.mutual_information(0.0)
        mi_one = self.strategy.mutual_information(1.0)
        self.assertEqual(mi_zero, 0.0)
        self.assertEqual(mi_one, 0.0)
        
        # Symmetric around 0.5
        mi_03 = self.strategy.mutual_information(0.3)
        mi_07 = self.strategy.mutual_information(0.7)
        self.assertAlmostEqual(mi_03, mi_07, places=5)
    
    def test_should_intercept(self):
        """Test adaptive interception decision."""
        # Test base probability
        np.random.seed(42)
        intercepts = [self.strategy.should_intercept({}) for _ in range(1000)]
        intercept_rate = sum(intercepts) / len(intercepts)
        
        # Should be around base_intercept_prob + confidence_bonus
        # With uniform prior, confidence ~0.42, so expected ~0.3 + 0.4*0.42 = 0.47
        self.assertAlmostEqual(intercept_rate, 0.47, delta=0.05)
        
        # After learning, intercept rate should increase
        for _ in range(100):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        
        intercepts = [self.strategy.should_intercept({}) for _ in range(1000)]
        new_intercept_rate = sum(intercepts) / len(intercepts)
        
        # Should be higher due to confidence bonus
        self.assertGreater(new_intercept_rate, intercept_rate)
    
    def test_intercept(self):
        """Test qubit interception."""
        # Prepare a qubit
        qubit = self.backend.prepare_state(0, Basis.RECTILINEAR)
        
        # Intercept without knowing Alice's basis
        modified_qubit, was_intercepted = self.strategy.intercept(qubit, None)
        
        self.assertTrue(was_intercepted)
        self.assertEqual(len(self.strategy.measured_bits), 1)
        self.assertEqual(len(self.strategy.measurement_bases), 1)
        self.assertEqual(self.strategy.statistics['interceptions'], 1)
    
    def test_intercept_with_alice_basis(self):
        """Test interception with Alice's basis knowledge."""
        # Prepare qubit
        qubit = self.backend.prepare_state(1, Basis.RECTILINEAR)
        
        # Intercept with Alice's basis known
        modified_qubit, was_intercepted = self.strategy.intercept(
            qubit, 
            alice_basis=Basis.RECTILINEAR
        )
        
        self.assertTrue(was_intercepted)
        self.assertEqual(self.strategy.total_predictions, 1)
        
        # Information gained depends on whether basis matched
        self.assertGreater(self.strategy.statistics['information_gained'], 0)
    
    def test_update_strategy(self):
        """Test strategy update from feedback."""
        # Simulate public basis announcement
        public_bases = [Basis.RECTILINEAR] * 30 + [Basis.DIAGONAL] * 20
        
        feedback = {'public_bases': public_bases}
        self.strategy.update_strategy(feedback)
        
        # Should have observed all bases
        self.assertEqual(len(self.strategy.observations), 50)
        self.assertEqual(self.strategy.alpha, 31.0)  # 1 + 30
        self.assertEqual(self.strategy.beta, 21.0)   # 1 + 20
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        # Make some observations
        for _ in range(10):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        
        stats = self.strategy.get_statistics()
        
        # Check all required fields
        self.assertIn('alpha', stats)
        self.assertIn('beta', stats)
        self.assertIn('prob_rectilinear', stats)
        self.assertIn('prob_diagonal', stats)
        self.assertIn('confidence', stats)
        self.assertIn('total_observations', stats)
        self.assertIn('prediction_accuracy', stats)
        
        self.assertEqual(stats['total_observations'], 10)
        self.assertEqual(stats['alpha'], 11.0)


class TestParticleFilterBasisLearner(unittest.TestCase):
    """Test particle filter Bayesian learning strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.backend = ClassicalBackend()
        self.strategy = ParticleFilterBasisLearner(
            self.backend,
            n_particles=1000,
            base_intercept_prob=0.3,
            confidence_threshold=0.8,
            resample_threshold=0.5
        )
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.n_particles, 1000)
        self.assertEqual(self.strategy.base_intercept_prob, 0.3)
        self.assertEqual(len(self.strategy.particles), 1000)
        self.assertEqual(len(self.strategy.weights), 1000)
        
        # Weights should be uniform initially
        self.assertAlmostEqual(self.strategy.weights[0], 1.0/1000, places=5)
        
        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(self.strategy.weights), 1.0, places=5)
    
    def test_observe_basis_rectilinear(self):
        """Test observation of rectilinear basis."""
        initial_ess = self.strategy.effective_sample_size()
        
        self.strategy.observe_basis(Basis.RECTILINEAR)
        
        self.assertEqual(len(self.strategy.observations), 1)
        # ESS should change after observation
        new_ess = self.strategy.effective_sample_size()
        self.assertNotEqual(initial_ess, new_ess)
    
    def test_observe_basis_diagonal(self):
        """Test observation of diagonal basis."""
        self.strategy.observe_basis(Basis.DIAGONAL)
        
        self.assertEqual(len(self.strategy.observations), 1)
        self.assertEqual(self.strategy.observations[0], Basis.DIAGONAL)
    
    def test_effective_sample_size(self):
        """Test ESS calculation."""
        # Initially with uniform weights, ESS should equal n_particles
        ess = self.strategy.effective_sample_size()
        self.assertAlmostEqual(ess, 1000, places=0)
        
        # After observations, ESS should decrease
        for _ in range(10):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        
        new_ess = self.strategy.effective_sample_size()
        self.assertLess(new_ess, ess)
    
    def test_resampling(self):
        """Test particle resampling."""
        initial_resampling_count = self.strategy.resampling_count
        
        # Force resampling by making ESS very low
        # Set one weight very high, others very low
        self.strategy.weights = np.zeros(1000)
        self.strategy.weights[0] = 1.0
        
        self.strategy.resample()
        
        # Resampling count should increase
        self.assertEqual(self.strategy.resampling_count, initial_resampling_count + 1)
        
        # Weights should be uniform after resampling
        self.assertAlmostEqual(self.strategy.weights[0], 1.0/1000, places=5)
    
    def test_automatic_resampling(self):
        """Test that resampling triggers automatically."""
        initial_count = self.strategy.resampling_count
        
        # Make many observations that will reduce ESS
        for _ in range(50):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        
        # Should have triggered at least one resampling
        self.assertGreater(self.strategy.resampling_count, initial_count)
    
    def test_predict_basis(self):
        """Test basis prediction."""
        # Initial prediction (should be close to 0.5)
        prediction = self.strategy.predict_basis()
        self.assertIn(prediction, [Basis.RECTILINEAR, Basis.DIAGONAL])
        
        # After learning pattern, should predict correctly
        for _ in range(100):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        
        # Should consistently predict rectilinear now
        prediction = self.strategy.predict_basis()
        self.assertEqual(prediction, Basis.RECTILINEAR)
    
    def test_get_confidence(self):
        """Test confidence calculation."""
        # Initial confidence should be low
        initial_conf = self.strategy.get_confidence()
        self.assertLess(initial_conf, 0.5)
        
        # After many observations, confidence should increase
        for _ in range(100):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        
        final_conf = self.strategy.get_confidence()
        self.assertGreater(final_conf, initial_conf)
        
        # Confidence should be in [0, 1]
        self.assertGreaterEqual(final_conf, 0.0)
        self.assertLessEqual(final_conf, 1.0)
    
    def test_get_basis_probability(self):
        """Test basis probability calculation."""
        # Initially around 0.5
        prob_rect = self.strategy.get_basis_probability(Basis.RECTILINEAR)
        self.assertAlmostEqual(prob_rect, 0.5, delta=0.1)
        
        # After observations, should converge
        for _ in range(60):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        for _ in range(40):
            self.strategy.observe_basis(Basis.DIAGONAL)
        
        prob_rect = self.strategy.get_basis_probability(Basis.RECTILINEAR)
        prob_diag = self.strategy.get_basis_probability(Basis.DIAGONAL)
        
        # Particle filter can be noisier, so use wider tolerance
        self.assertGreater(prob_rect, 0.5)  # Should be > 0.5 since more rectilinear
        self.assertAlmostEqual(prob_rect + prob_diag, 1.0, places=5)
    
    def test_intercept(self):
        """Test qubit interception."""
        qubit = self.backend.prepare_state(0, Basis.RECTILINEAR)
        modified_qubit, was_intercepted = self.strategy.intercept(qubit)
        
        self.assertTrue(was_intercepted)
        self.assertEqual(len(self.strategy.measured_bits), 1)
        self.assertEqual(self.strategy.statistics['interceptions'], 1)
    
    def test_update_strategy(self):
        """Test strategy update."""
        public_bases = [Basis.RECTILINEAR] * 30 + [Basis.DIAGONAL] * 20
        feedback = {'public_bases': public_bases}
        
        self.strategy.update_strategy(feedback)
        
        self.assertEqual(len(self.strategy.observations), 50)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        # Make observations
        for _ in range(10):
            self.strategy.observe_basis(Basis.RECTILINEAR)
        
        stats = self.strategy.get_statistics()
        
        # Check required fields
        self.assertIn('n_particles', stats)
        self.assertIn('prob_rectilinear', stats)
        self.assertIn('confidence', stats)
        self.assertIn('effective_sample_size', stats)
        self.assertIn('resampling_count', stats)
        self.assertIn('total_observations', stats)
        
        self.assertEqual(stats['n_particles'], 1000)
        self.assertEqual(stats['total_observations'], 10)


class TestBayesianIntegration(unittest.TestCase):
    """Test Bayesian strategies in full BB84 protocol."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = ClassicalBackend()
    
    def test_beta_bayesian_in_bb84(self):
        """Test Beta Bayesian strategy in full BB84 protocol."""
        # Setup
        strategy = BasisLearningStrategy(self.backend, base_intercept_prob=0.3)
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
        
        # Public announcement
        alice_bases = alice.announce_bases()
        bob_bases = bob.announce_bases()
        
        # Eve learns from public bases
        feedback = {'public_bases': alice_bases}
        strategy.update_strategy(feedback)
        
        # Check Eve learned
        eve_stats = eve.get_statistics()
        self.assertGreater(eve_stats['attack_strategy_stats']['total_observations'], 0)
    
    def test_particle_filter_in_bb84(self):
        """Test Particle Filter strategy in full BB84 protocol."""
        # Setup
        strategy = ParticleFilterBasisLearner(self.backend, n_particles=500)
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
        
        # Public announcement
        alice_bases = alice.announce_bases()
        
        # Eve learns from public bases
        feedback = {'public_bases': alice_bases}
        strategy.update_strategy(feedback)
        
        # Check statistics
        eve_stats = eve.get_statistics()
        strategy_stats = eve_stats['attack_strategy_stats']
        
        self.assertGreater(strategy_stats['total_observations'], 0)
        self.assertIn('effective_sample_size', strategy_stats)
    
    def test_learning_convergence(self):
        """Test that learning converges over multiple rounds."""
        strategy = BasisLearningStrategy(self.backend)
        
        # Simulate biased Alice (70% rectilinear)
        np.random.seed(42)
        for round_num in range(10):
            bases = []
            for _ in range(50):
                if np.random.random() < 0.7:
                    bases.append(Basis.RECTILINEAR)
                else:
                    bases.append(Basis.DIAGONAL)
            
            for basis in bases:
                strategy.observe_basis(basis)
        
        # Should converge to true probability
        prob_rect = strategy.get_basis_probability(Basis.RECTILINEAR)
        self.assertAlmostEqual(prob_rect, 0.7, delta=0.05)
        
        # Confidence should be high
        confidence = strategy.get_confidence()
        self.assertGreater(confidence, 0.9)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
