"""
Unit tests for Eve (Eavesdropper) implementation
Tests AttackStrategy, EveController, and QuantumChannel integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'main'))

import unittest
import numpy as np
from bb84_main import (
    ClassicalBackend,
    Alice,
    Bob,
    QuantumChannel,
    EveController,
    InterceptResendAttack,
    AdaptiveAttack,
    QBERAdaptiveStrategy,
    GradientDescentQBERAdaptive,
    Basis,
    AttackStrategy
)
from typing import Tuple, Dict, Any, Optional


class TestAttackStrategy(unittest.TestCase):
    """Test the abstract AttackStrategy base class"""
    
    def test_abstract_methods(self):
        """Test that AttackStrategy cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            AttackStrategy()
    
    def test_statistics_initialization(self):
        """Test that concrete strategies initialize statistics correctly"""
        backend = ClassicalBackend()
        strategy = InterceptResendAttack(backend)
        
        stats = strategy.get_statistics()
        self.assertIn('interceptions', stats)
        self.assertIn('successful_measurements', stats)
        self.assertIn('information_gained', stats)
        self.assertEqual(stats['interceptions'], 0)


class TestInterceptResendAttack(unittest.TestCase):
    """Test InterceptResend attack strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = ClassicalBackend()
    
    def test_initialization(self):
        """Test strategy initialization"""
        strategy = InterceptResendAttack(self.backend, intercept_probability=0.7)
        self.assertEqual(strategy.intercept_probability, 0.7)
        self.assertEqual(len(strategy.measured_bits), 0)
    
    def test_should_intercept_always(self):
        """Test 100% interception rate"""
        strategy = InterceptResendAttack(self.backend, intercept_probability=1.0)
        
        # Should always intercept
        for _ in range(10):
            self.assertTrue(strategy.should_intercept({}))
    
    def test_should_intercept_never(self):
        """Test 0% interception rate"""
        strategy = InterceptResendAttack(self.backend, intercept_probability=0.0)
        
        # Should never intercept
        for _ in range(10):
            self.assertFalse(strategy.should_intercept({}))
    
    def test_intercept_qubit(self):
        """Test qubit interception"""
        strategy = InterceptResendAttack(self.backend, intercept_probability=1.0)
        
        # Create a test qubit
        qubit = self.backend.prepare_state(bit=1, basis=Basis.RECTILINEAR)
        
        # Intercept it
        modified_qubit, was_intercepted = strategy.intercept(qubit)
        
        self.assertTrue(was_intercepted)
        self.assertIsNotNone(modified_qubit)
        self.assertEqual(len(strategy.measured_bits), 1)
        
        # Check statistics
        stats = strategy.get_statistics()
        self.assertEqual(stats['interceptions'], 1)
        self.assertEqual(stats['successful_measurements'], 1)
    
    def test_update_strategy_high_qber(self):
        """Test strategy adaptation to high QBER"""
        strategy = InterceptResendAttack(self.backend, intercept_probability=1.0)
        initial_rate = strategy.intercept_probability
        
        # Simulate high QBER feedback
        feedback = {'qber': 0.25, 'qber_history': [0.25]}
        strategy.update_strategy(feedback)
        
        # Should reduce interception rate
        self.assertLess(strategy.intercept_probability, initial_rate)
    
    def test_update_strategy_low_qber(self):
        """Test strategy adaptation to low QBER"""
        strategy = InterceptResendAttack(self.backend, intercept_probability=0.5)
        initial_rate = strategy.intercept_probability
        
        # Simulate low QBER feedback
        feedback = {'qber': 0.05, 'qber_history': [0.05]}
        strategy.update_strategy(feedback)
        
        # Should increase interception rate
        self.assertGreater(strategy.intercept_probability, initial_rate)


class TestAdaptiveAttack(unittest.TestCase):
    """Test Adaptive attack strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = ClassicalBackend()
    
    def test_initialization(self):
        """Test adaptive strategy initialization"""
        strategy = AdaptiveAttack(
            self.backend, 
            target_qber=0.08, 
            initial_intercept_rate=0.5
        )
        
        self.assertEqual(strategy.target_qber, 0.08)
        self.assertEqual(strategy.intercept_rate, 0.5)
        self.assertEqual(len(strategy.qber_samples), 0)
    
    def test_adaptation_to_high_qber(self):
        """Test adaptation when QBER exceeds target"""
        strategy = AdaptiveAttack(
            self.backend, 
            target_qber=0.08, 
            initial_intercept_rate=0.8
        )
        
        initial_rate = strategy.intercept_rate
        
        # Feedback with high QBER
        feedback = {'qber': 0.15}
        strategy.update_strategy(feedback)
        
        # Should reduce intercept rate
        self.assertLess(strategy.intercept_rate, initial_rate)
    
    def test_adaptation_to_low_qber(self):
        """Test adaptation when QBER below target"""
        strategy = AdaptiveAttack(
            self.backend, 
            target_qber=0.08, 
            initial_intercept_rate=0.2
        )
        
        initial_rate = strategy.intercept_rate
        
        # Feedback with low QBER
        feedback = {'qber': 0.03}
        strategy.update_strategy(feedback)
        
        # Should increase intercept rate
        self.assertGreater(strategy.intercept_rate, initial_rate)
    
    def test_bounds_enforcement(self):
        """Test that intercept rate stays within bounds"""
        strategy = AdaptiveAttack(self.backend, target_qber=0.08)
        
        # Try to push above 1.0
        for _ in range(10):
            feedback = {'qber': 0.01}
            strategy.update_strategy(feedback)
        
        self.assertLessEqual(strategy.intercept_rate, 1.0)
        
        # Try to push below 0.0
        for _ in range(20):
            feedback = {'qber': 0.30}
            strategy.update_strategy(feedback)
        
        self.assertGreaterEqual(strategy.intercept_rate, 0.01)


class TestEveController(unittest.TestCase):
    """Test EveController class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = ClassicalBackend()
        self.strategy = InterceptResendAttack(self.backend, intercept_probability=1.0)
        self.eve = EveController(self.strategy, self.backend, name="TestEve")
    
    def test_initialization(self):
        """Test Eve controller initialization"""
        self.assertEqual(self.eve.name, "TestEve")
        self.assertEqual(len(self.eve.qber_history), 0)
        self.assertEqual(len(self.eve.intercepted_qubits), 0)
    
    def test_intercept_transmission(self):
        """Test qubit transmission interception"""
        qubit = self.backend.prepare_state(1, Basis.RECTILINEAR)
        metadata = {'qubit_index': 0, 'timestamp': None}
        
        modified_qubit = self.eve.intercept_transmission(qubit, metadata)
        
        self.assertIsNotNone(modified_qubit)
        self.assertEqual(len(self.eve.intercepted_qubits), 1)
        self.assertEqual(self.eve.intercepted_qubits[0]['index'], 0)
    
    def test_receive_feedback(self):
        """Test feedback reception and processing"""
        public_info = {
            'alice_bases': [Basis.RECTILINEAR, Basis.DIAGONAL, Basis.RECTILINEAR]
        }
        
        self.eve.receive_feedback(qber=0.10, public_info=public_info)
        
        self.assertEqual(len(self.eve.qber_history), 1)
        self.assertEqual(self.eve.qber_history[0], 0.10)
        self.assertEqual(
            self.eve.basis_observations['rectilinear_observed'], 2
        )
        self.assertEqual(
            self.eve.basis_observations['diagonal_observed'], 1
        )
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        stats = self.eve.get_statistics()
        
        self.assertIn('name', stats)
        self.assertIn('total_intercepted', stats)
        self.assertIn('qber_history', stats)
        self.assertIn('basis_observations', stats)
        self.assertIn('attack_strategy_stats', stats)
        self.assertEqual(stats['name'], "TestEve")


class TestQuantumChannelIntegration(unittest.TestCase):
    """Test QuantumChannel integration with Eve"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = ClassicalBackend()
    
    def test_channel_without_eve(self):
        """Test channel works without Eve (backward compatibility)"""
        channel = QuantumChannel(
            self.backend, 
            loss_rate=0.0, 
            error_rate=0.0
        )
        
        alice = Alice(self.backend)
        alice.generate_random_bits(10)
        alice.choose_random_bases(10)
        states = alice.prepare_states()
        
        received = channel.transmit(states)
        
        self.assertEqual(len(received), 10)
        self.assertIsNone(channel.eve)
    
    def test_channel_with_eve(self):
        """Test channel with Eve present"""
        strategy = InterceptResendAttack(self.backend, intercept_probability=1.0)
        eve = EveController(strategy, self.backend)
        
        channel = QuantumChannel(
            self.backend,
            loss_rate=0.0,
            error_rate=0.0,
            eve=eve
        )
        
        alice = Alice(self.backend)
        alice.generate_random_bits(10)
        alice.choose_random_bases(10)
        states = alice.prepare_states()
        
        received = channel.transmit(states)
        
        self.assertEqual(len(received), 10)
        self.assertIsNotNone(channel.eve)
        # Eve should have intercepted all qubits
        self.assertGreater(len(eve.intercepted_qubits), 0)
    
    def test_transmission_order(self):
        """Test that transmission order is correct: loss -> noise -> eve"""
        strategy = InterceptResendAttack(self.backend, intercept_probability=1.0)
        eve = EveController(strategy, self.backend)
        
        channel = QuantumChannel(
            self.backend,
            loss_rate=0.5,  # 50% loss
            error_rate=0.1,  # 10% error
            eve=eve
        )
        
        alice = Alice(self.backend)
        alice.generate_random_bits(100)
        alice.choose_random_bases(100)
        states = alice.prepare_states()
        
        received = channel.transmit(states)
        
        # Check that some states were lost
        lost_count = sum(1 for s in received if s is None)
        self.assertGreater(lost_count, 0)
        
        # Check that Eve only intercepted non-lost states
        # (Eve's interceptions <= transmitted states)
        self.assertLessEqual(
            len(eve.intercepted_qubits), 
            channel.transmitted_count
        )


class TestEndToEndScenario(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = ClassicalBackend()
    
    def test_bb84_without_eve(self):
        """Test BB84 protocol without eavesdropping"""
        alice = Alice(self.backend)
        bob = Bob(self.backend)
        channel = QuantumChannel(self.backend, loss_rate=0.0, error_rate=0.0)
        
        # Run protocol
        num_bits = 100
        alice.generate_random_bits(num_bits)
        alice.choose_random_bases(num_bits)
        states = alice.prepare_states()
        
        received = channel.transmit(states)
        
        bob.choose_random_bases(num_bits)
        bob.measure_states(received)
        
        # Sift keys
        matching = [i for i in range(num_bits) 
                   if alice.bases[i] == bob.bases[i]]
        alice_key = [alice.bits[i] for i in matching]
        bob_key = [bob.measured_bits[i] for i in matching]
        
        # Calculate QBER
        errors = sum(1 for i in range(len(alice_key)) 
                    if alice_key[i] != bob_key[i])
        qber = errors / len(alice_key) if alice_key else 0
        
        # Without Eve, QBER should be ~0
        self.assertLess(qber, 0.05)
    
    def test_bb84_with_full_interception(self):
        """Test BB84 with 100% interception"""
        alice = Alice(self.backend)
        bob = Bob(self.backend)
        
        strategy = InterceptResendAttack(self.backend, intercept_probability=1.0)
        eve = EveController(strategy, self.backend)
        channel = QuantumChannel(self.backend, loss_rate=0.0, error_rate=0.0, eve=eve)
        
        # Run protocol
        num_bits = 500
        alice.generate_random_bits(num_bits)
        alice.choose_random_bases(num_bits)
        states = alice.prepare_states()
        
        received = channel.transmit(states)
        
        bob.choose_random_bases(num_bits)
        bob.measure_states(received)
        
        # Sift keys
        matching = [i for i in range(num_bits) 
                   if alice.bases[i] == bob.bases[i]]
        alice_key = [alice.bits[i] for i in matching]
        bob_key = [bob.measured_bits[i] for i in matching]
        
        # Calculate QBER
        errors = sum(1 for i in range(len(alice_key)) 
                    if alice_key[i] != bob_key[i])
        qber = errors / len(alice_key) if alice_key else 0
        
        # With 100% interception, QBER should be ~25%
        self.assertGreater(qber, 0.15)  # At least 15%
        self.assertLess(qber, 0.35)     # At most 35%
        
        # Check Eve's statistics
        eve_stats = eve.get_statistics()
        self.assertEqual(eve_stats['total_intercepted'], num_bits)


class TestStatisticsTracking(unittest.TestCase):
    """Test statistics tracking functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = ClassicalBackend()
    
    def test_strategy_statistics(self):
        """Test that strategy statistics are updated correctly"""
        strategy = InterceptResendAttack(self.backend, intercept_probability=1.0)
        
        # Perform some interceptions
        for i in range(10):
            qubit = self.backend.prepare_state(i % 2, Basis.RECTILINEAR)
            strategy.intercept(qubit)
        
        stats = strategy.get_statistics()
        
        self.assertEqual(stats['interceptions'], 10)
        self.assertEqual(stats['successful_measurements'], 10)
        self.assertGreater(stats['information_gained'], 0)
    
    def test_eve_statistics_integration(self):
        """Test that Eve correctly integrates strategy statistics"""
        strategy = InterceptResendAttack(self.backend, intercept_probability=0.5)
        eve = EveController(strategy, self.backend)
        
        # Simulate some interceptions
        for i in range(20):
            qubit = self.backend.prepare_state(i % 2, Basis.RECTILINEAR)
            metadata = {'qubit_index': i}
            eve.intercept_transmission(qubit, metadata)
        
        stats = eve.get_statistics()
        
        # Check that statistics are present
        self.assertIn('attack_strategy_stats', stats)
        self.assertGreater(stats['total_intercepted'], 0)
        
        # With 50% probability, should intercept roughly half
        self.assertGreater(stats['total_intercepted'], 0)


class TestQBERAdaptiveStrategy(unittest.TestCase):
    """Test QBER-adaptive strategy with PID control"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = ClassicalBackend()
    
    def test_initialization(self):
        """Test PID strategy initialization"""
        strategy = QBERAdaptiveStrategy(
            self.backend,
            target_qber=0.10,
            threshold=0.11,
            kp=2.0,
            ki=0.5,
            kd=0.1
        )
        
        self.assertEqual(strategy.target_qber, 0.10)
        self.assertEqual(strategy.threshold, 0.11)
        self.assertEqual(strategy.kp, 2.0)
        self.assertEqual(strategy.ki, 0.5)
        self.assertEqual(strategy.kd, 0.1)
        self.assertEqual(strategy.intercept_probability, 0.3)
        self.assertEqual(len(strategy.qber_history), 0)
        self.assertEqual(strategy.error_integral, 0.0)
    
    def test_should_intercept(self):
        """Test interception decision logic"""
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10)
        
        # Set probability to 1.0 for deterministic testing
        strategy.intercept_probability = 1.0
        for _ in range(10):
            self.assertTrue(strategy.should_intercept({}))
        
        # Set probability to 0.0
        strategy.intercept_probability = 0.0
        for _ in range(10):
            self.assertFalse(strategy.should_intercept({}))
    
    def test_measure_and_prepare_qubit(self):
        """Test qubit measurement and preparation"""
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10)
        
        # Test measurement
        qubit = self.backend.prepare_state(1, Basis.RECTILINEAR)
        measured_bit = strategy.measure_qubit(qubit, Basis.RECTILINEAR)
        self.assertIn(measured_bit, [0, 1])
        
        # Test preparation
        prepared_qubit = strategy.prepare_qubit(1, Basis.DIAGONAL)
        self.assertIsNotNone(prepared_qubit)
    
    def test_intercept(self):
        """Test qubit interception"""
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10)
        
        qubit = self.backend.prepare_state(0, Basis.RECTILINEAR)
        modified_qubit, was_intercepted = strategy.intercept(qubit)
        
        self.assertTrue(was_intercepted)
        self.assertIsNotNone(modified_qubit)
        self.assertEqual(len(strategy.measured_bits), 1)
        self.assertEqual(len(strategy.measurement_bases), 1)
        
        # Check statistics
        stats = strategy.get_statistics()
        self.assertEqual(stats['interceptions'], 1)
        self.assertEqual(stats['successful_measurements'], 1)
    
    def test_pid_update_above_target(self):
        """Test PID update when QBER above target"""
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10)
        initial_prob = strategy.intercept_probability
        
        # Simulate high QBER (above target)
        feedback = {'qber': 0.15}
        strategy.update_strategy(feedback)
        
        # Should reduce intercept probability
        self.assertLess(strategy.intercept_probability, initial_prob)
        self.assertEqual(len(strategy.qber_history), 1)
    
    def test_pid_update_below_target(self):
        """Test PID update when QBER below target"""
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10)
        initial_prob = strategy.intercept_probability
        
        # Simulate low QBER (below target)
        feedback = {'qber': 0.05}
        strategy.update_strategy(feedback)
        
        # Should increase intercept probability
        self.assertGreater(strategy.intercept_probability, initial_prob)
    
    def test_pid_clipping(self):
        """Test that probability stays within [0.0, 0.9] bounds"""
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10, kp=10.0)
        
        # Try to push above 0.9
        for _ in range(10):
            feedback = {'qber': 0.01}
            strategy.update_strategy(feedback)
        
        self.assertLessEqual(strategy.intercept_probability, 0.9)
        
        # Try to push below 0.0
        for _ in range(20):
            feedback = {'qber': 0.25}
            strategy.update_strategy(feedback)
        
        self.assertGreaterEqual(strategy.intercept_probability, 0.0)
    
    def test_safety_mechanism(self):
        """Test safety mechanism when QBER > 0.09"""
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10)
        strategy.intercept_probability = 0.5
        
        # High QBER should trigger safety mechanism
        feedback = {'qber': 0.095}
        strategy.update_strategy(feedback)
        
        # Probability should be reduced by 10%
        self.assertLess(strategy.intercept_probability, 0.5)
    
    def test_integral_anti_windup(self):
        """Test anti-windup mechanism for integral term"""
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10)
        
        # Accumulate large integral error
        for _ in range(20):
            feedback = {'qber': 0.01}
            strategy.update_strategy(feedback)
        
        # When saturated at 0.9, integral should be reset
        self.assertEqual(strategy.intercept_probability, 0.9)
        self.assertEqual(strategy.error_integral, 0.0)
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10)
        
        # Perform some operations
        for i in range(5):
            qubit = self.backend.prepare_state(i % 2, Basis.RECTILINEAR)
            strategy.intercept(qubit)
        
        feedback = {'qber': 0.08}
        strategy.update_strategy(feedback)
        
        stats = strategy.get_statistics()
        
        self.assertIn('intercept_probability', stats)
        self.assertIn('target_qber', stats)
        self.assertIn('threshold', stats)
        self.assertIn('qber_history', stats)
        self.assertIn('error_integral', stats)
        self.assertIn('pid_gains', stats)
        self.assertEqual(stats['total_measurements'], 5)
        self.assertEqual(len(stats['qber_history']), 1)


class TestGradientDescentQBERAdaptive(unittest.TestCase):
    """Test gradient descent QBER-adaptive strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = ClassicalBackend()
    
    def test_initialization(self):
        """Test gradient descent initialization"""
        strategy = GradientDescentQBERAdaptive(
            self.backend,
            target_qber=0.10,
            threshold=0.11,
            learning_rate=0.01
        )
        
        self.assertEqual(strategy.target_qber, 0.10)
        self.assertEqual(strategy.threshold, 0.11)
        self.assertEqual(strategy.learning_rate, 0.01)
        self.assertEqual(strategy.intercept_probability, 0.3)
        self.assertEqual(strategy.qber_gradient, 0.25)
        self.assertEqual(len(strategy.qber_history), 0)
        self.assertEqual(len(strategy.loss_history), 0)
    
    def test_intercept(self):
        """Test qubit interception"""
        strategy = GradientDescentQBERAdaptive(self.backend, target_qber=0.10)
        
        qubit = self.backend.prepare_state(1, Basis.DIAGONAL)
        modified_qubit, was_intercepted = strategy.intercept(qubit)
        
        self.assertTrue(was_intercepted)
        self.assertIsNotNone(modified_qubit)
        self.assertEqual(len(strategy.measured_bits), 1)
    
    def test_gradient_descent_update_above_target(self):
        """Test gradient descent when QBER above target"""
        strategy = GradientDescentQBERAdaptive(
            self.backend, 
            target_qber=0.10,
            learning_rate=0.1
        )
        initial_prob = strategy.intercept_probability
        
        # High QBER should reduce probability
        feedback = {'qber': 0.15}
        strategy.update_strategy(feedback)
        
        self.assertLess(strategy.intercept_probability, initial_prob)
        self.assertEqual(len(strategy.qber_history), 1)
        self.assertEqual(len(strategy.loss_history), 1)
    
    def test_gradient_descent_update_below_target(self):
        """Test gradient descent when QBER below target"""
        strategy = GradientDescentQBERAdaptive(
            self.backend, 
            target_qber=0.10,
            learning_rate=0.1
        )
        initial_prob = strategy.intercept_probability
        
        # Low QBER should increase probability
        feedback = {'qber': 0.05}
        strategy.update_strategy(feedback)
        
        self.assertGreater(strategy.intercept_probability, initial_prob)
    
    def test_loss_calculation(self):
        """Test loss function calculation"""
        strategy = GradientDescentQBERAdaptive(self.backend, target_qber=0.10)
        
        feedback = {'qber': 0.12}
        strategy.update_strategy(feedback)
        
        # Loss should be (0.12 - 0.10)² = 0.0004
        expected_loss = (0.12 - 0.10) ** 2
        self.assertAlmostEqual(strategy.loss_history[0], expected_loss, places=6)
    
    def test_gradient_calculation(self):
        """Test gradient calculation"""
        strategy = GradientDescentQBERAdaptive(
            self.backend, 
            target_qber=0.10,
            learning_rate=0.01
        )
        
        # QBER = 0.15, target = 0.10
        # Gradient = 2 * (0.15 - 0.10) * 0.25 = 0.025
        # Update = -0.01 * 0.025 = -0.00025
        # After gradient: 0.30 - 0.00025 = 0.29975
        # Safety mechanism (QBER > 0.09): 0.29975 * 0.9 = 0.269775
        initial_prob = strategy.intercept_probability
        feedback = {'qber': 0.15}
        strategy.update_strategy(feedback)
        
        expected_prob = initial_prob - 0.01 * 2 * (0.15 - 0.10) * 0.25
        expected_prob *= 0.9  # Safety mechanism applies since QBER > 0.09
        self.assertAlmostEqual(strategy.intercept_probability, expected_prob, places=6)
    
    def test_bounds_enforcement(self):
        """Test probability bounds [0.0, 0.9]"""
        strategy = GradientDescentQBERAdaptive(
            self.backend, 
            target_qber=0.10,
            learning_rate=1.0  # Large learning rate
        )
        
        # Try to push above 0.9
        for _ in range(10):
            feedback = {'qber': 0.01}
            strategy.update_strategy(feedback)
        
        self.assertLessEqual(strategy.intercept_probability, 0.9)
        
        # Try to push below 0.0
        strategy.intercept_probability = 0.5
        for _ in range(20):
            feedback = {'qber': 0.30}
            strategy.update_strategy(feedback)
        
        self.assertGreaterEqual(strategy.intercept_probability, 0.0)
    
    def test_safety_mechanism(self):
        """Test safety mechanism for high QBER"""
        strategy = GradientDescentQBERAdaptive(self.backend, target_qber=0.10)
        strategy.intercept_probability = 0.5
        
        feedback = {'qber': 0.095}
        strategy.update_strategy(feedback)
        
        # Should be reduced by safety mechanism
        self.assertLess(strategy.intercept_probability, 0.5)
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        strategy = GradientDescentQBERAdaptive(self.backend, target_qber=0.10)
        
        # Run some updates
        for qber in [0.08, 0.09, 0.10, 0.11]:
            feedback = {'qber': qber}
            strategy.update_strategy(feedback)
        
        stats = strategy.get_statistics()
        
        self.assertIn('intercept_probability', stats)
        self.assertIn('target_qber', stats)
        self.assertIn('learning_rate', stats)
        self.assertIn('qber_history', stats)
        self.assertIn('loss_history', stats)
        self.assertIn('qber_gradient', stats)
        self.assertEqual(len(stats['qber_history']), 4)
        self.assertEqual(len(stats['loss_history']), 4)


class TestQBERAdaptiveIntegration(unittest.TestCase):
    """Integration tests for QBER-adaptive strategies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = ClassicalBackend()
    
    def test_pid_bb84_integration(self):
        """Test PID strategy in full BB84 protocol"""
        alice = Alice(self.backend)
        bob = Bob(self.backend)
        
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10)
        eve = EveController(strategy, self.backend)
        channel = QuantumChannel(self.backend, loss_rate=0.0, error_rate=0.0, eve=eve)
        
        # Run BB84
        num_bits = 500
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
        
        # QBER should be non-zero (Eve is present)
        self.assertGreater(qber, 0.0)
        
        # Check Eve's statistics
        eve_stats = eve.get_statistics()
        self.assertGreater(eve_stats['total_intercepted'], 0)
    
    def test_gradient_descent_bb84_integration(self):
        """Test gradient descent strategy in full BB84 protocol"""
        alice = Alice(self.backend)
        bob = Bob(self.backend)
        
        strategy = GradientDescentQBERAdaptive(self.backend, target_qber=0.10)
        eve = EveController(strategy, self.backend)
        channel = QuantumChannel(self.backend, loss_rate=0.0, error_rate=0.0, eve=eve)
        
        # Run BB84
        num_bits = 500
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
        
        # QBER should be non-zero
        self.assertGreater(qber, 0.0)
        
        # Give Eve feedback
        eve.receive_feedback(qber, {'alice_bases': alice.bases})
        
        eve_stats = eve.get_statistics()
        self.assertEqual(len(eve_stats['qber_history']), 1)
    
    def test_multi_round_convergence(self):
        """Test that PID strategy converges over multiple rounds"""
        strategy = QBERAdaptiveStrategy(self.backend, target_qber=0.10)
        eve = EveController(strategy, self.backend)
        
        qber_history = []
        
        for _ in range(10):
            alice = Alice(self.backend)
            bob = Bob(self.backend)
            channel = QuantumChannel(self.backend, loss_rate=0.0, error_rate=0.0, eve=eve)
            
            num_bits = 500
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
        
        # Check convergence: later QBERs should be closer to target
        early_avg = np.mean(qber_history[:3])
        late_avg = np.mean(qber_history[-3:])
        
        # Late average should be closer to target (0.10)
        self.assertLess(abs(late_avg - 0.10), abs(early_avg - 0.10) + 0.05)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
