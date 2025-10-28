import unittest
import numpy as np

from src.main.bb84_main import AttackDetector, Basis


class TestAttackDetector(unittest.TestCase):
    def setUp(self):
        self.detector = AttackDetector(qber_threshold=0.11)

    def test_qber_detection_high_qber(self):
        # Construct matching bases and bits with 20% error rate
        n = 500
        alice_bits = np.random.randint(0, 2, size=n).tolist()
        bob_bits = alice_bits.copy()
        # Flip 20% of bits
        flip_idx = np.random.choice(n, size=int(0.2 * n), replace=False)
        for i in flip_idx:
            bob_bits[i] = 1 - bob_bits[i]
        bases = [Basis.RECTILINEAR] * n

        res = self.detector.detect_qber(alice_bits, bob_bits, bases, bases, sample_size=300)
        self.assertTrue(res['flag'])
        self.assertGreater(res['qber'], self.detector.qber_threshold)

    def test_chi_square_basis_bias(self):
        # 80/20 split should be detected
        n = 1000
        bases = [Basis.RECTILINEAR] * int(0.8 * n) + [Basis.DIAGONAL] * int(0.2 * n)
        res = self.detector.detect_basis_randomness_chi_square(bases)
        self.assertTrue(res['flag'])
        self.assertLess(res['p_value'], 0.05)

    def test_basis_correlation_mi(self):
        # Perfectly correlated bases should yield high MI
        n = 1000
        alice_bases = np.random.randint(0, 2, size=n).tolist()
        bob_bases = alice_bases.copy()
        # Convert to Basis enums for realism
        alice_bases = [Basis.RECTILINEAR if b == 0 else Basis.DIAGONAL for b in alice_bases]
        bob_bases = [Basis.RECTILINEAR if b == 0 else Basis.DIAGONAL for b in bob_bases]

        res = self.detector.detect_basis_correlation_mi(alice_bases, bob_bases)
        self.assertTrue(res['flag'])
        self.assertGreater(res['mi'], 0.1)

    def test_runs_test_clustered_errors(self):
        # Build clustered error pattern: long run of 0s then 1s
        errors = [0] * 50 + [1] * 50
        res = self.detector.detect_runs_test(errors)
        self.assertTrue(res['flag'])
        self.assertLess(res['p_value'], 0.05)

    def test_detect_attack_aggregate(self):
        # Combine scenarios to ensure overall detection
        n = 600
        # Bases
        alice_bases = [Basis.RECTILINEAR] * int(0.8 * n) + [Basis.DIAGONAL] * (n - int(0.8 * n))
        bob_bases = alice_bases.copy()  # correlated

        # Bits with 15% error
        alice_bits = np.random.randint(0, 2, size=n).tolist()
        bob_bits = alice_bits.copy()
        flip_idx = np.random.choice(n, size=int(0.15 * n), replace=False)
        for i in flip_idx:
            bob_bits[i] = 1 - bob_bits[i]

        result = self.detector.detect_attack(alice_bits, bob_bits, alice_bases, bob_bases, sample_size=300)
        self.assertTrue(result['attack_detected'])
        self.assertTrue(result['qber']['flag'])
        self.assertTrue(result['chi_square']['alice']['flag'])
        self.assertTrue(result['basis_correlation']['flag'])

    def test_detect_attack_clean(self):
        # Nearly clean scenario should not flag
        n = 2000
        rng = np.random.default_rng(123)
        alice_bases = rng.integers(0, 2, size=n).tolist()
        bob_bases = rng.integers(0, 2, size=n).tolist()
        alice_bases = [Basis.RECTILINEAR if b == 0 else Basis.DIAGONAL for b in alice_bases]
        bob_bases = [Basis.RECTILINEAR if b == 0 else Basis.DIAGONAL for b in bob_bases]

        alice_bits = rng.integers(0, 2, size=n).tolist()
        bob_bits = alice_bits.copy()
        # 1% random errors
        flip_idx = rng.choice(n, size=int(0.01 * n), replace=False)
        for i in flip_idx:
            bob_bits[i] = 1 - bob_bits[i]

        result = self.detector.detect_attack(alice_bits, bob_bits, alice_bases, bob_bases, sample_size=300)
        # It's possible some random tests might barely trigger; assert majority benign
        self.assertFalse(result['qber']['flag'])
        self.assertFalse(result['chi_square']['alice']['flag'])
        self.assertFalse(result['chi_square']['bob']['flag'])
        self.assertFalse(result['basis_correlation']['flag'])
        self.assertFalse(result['runs_test']['flag'])
        self.assertFalse(result['attack_detected'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
