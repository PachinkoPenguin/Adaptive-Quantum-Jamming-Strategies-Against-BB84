import unittest
import numpy as np

from src.main.bb84_main import (
    von_neumann_entropy,
    holevo_bound,
    bb84_holevo_bound,
    binary_entropy,
    calculate_eve_information,
    mutual_information_eve_alice,
)


class TestEntropyAndHolevo(unittest.TestCase):
    def test_entropy_pure_and_mixed(self):
        # |0><0| has zero entropy
        ket0 = np.array([[1.0], [0.0]], dtype=complex)
        rho_pure = ket0 @ ket0.conjugate().T
        self.assertAlmostEqual(von_neumann_entropy(rho_pure, base=2), 0.0, places=8)

        # Maximally mixed qubit I/2 has entropy 1 bit
        rho_mixed = 0.5 * np.eye(2, dtype=complex)
        self.assertAlmostEqual(von_neumann_entropy(rho_mixed, base=2), 1.0, places=6)

    def test_bb84_holevo(self):
        chi = bb84_holevo_bound(base=2)
        self.assertAlmostEqual(chi, 1.0, places=3)

    def test_holevo_identical_states(self):
        # Two identical pure states => zero Holevo information
        ket0 = np.array([[1.0], [0.0]], dtype=complex)
        rho = ket0 @ ket0.conjugate().T
        chi = holevo_bound([rho, rho], [0.5, 0.5], base=2)
        self.assertAlmostEqual(chi, 0.0, places=8)

    def test_binary_entropy_edges(self):
        self.assertAlmostEqual(binary_entropy(0.0), 0.0, places=8)
        self.assertAlmostEqual(binary_entropy(1.0), 0.0, places=8)
        self.assertAlmostEqual(binary_entropy(0.5), 1.0, places=6)


class TestEveInformation(unittest.TestCase):
    def test_intercept_resend_info(self):
        info = calculate_eve_information('intercept_resend', {'intercept_probability': 0.4})
        self.assertAlmostEqual(info, 0.2, places=6)

    def test_pns_info(self):
        mu = 0.1
        info = calculate_eve_information('pns', {'mean_photon_number': mu})
        expected = 1.0 - (1.0 + mu) * np.exp(-mu)
        self.assertAlmostEqual(info, expected, places=8)

    def test_adaptive_info(self):
        # If success_prob=1, info equals intercept_prob
        info = calculate_eve_information('adaptive', {
            'intercept_probability': 0.3,
            'success_probability': 1.0,
        })
        self.assertAlmostEqual(info, 0.3, places=6)
        # If success_prob=0.5, binary entropy = 1 => zero information
        info2 = calculate_eve_information('adaptive', {
            'intercept_probability': 0.7,
            'success_probability': 0.5,
        })
        self.assertAlmostEqual(info2, 0.0, places=6)

    def test_mutual_information_eve_alice(self):
        # With basis match 0.5 and perfect success, expect modest MI
        I = mutual_information_eve_alice(basis_match_prob=0.5, eve_success_prob=1.0, alice_flip_prob=0.0)
        # Should be positive and < 1
        self.assertGreater(I, 0.0)
        self.assertLess(I, 1.0)
        # More noise should reduce MI
        I_noisy = mutual_information_eve_alice(basis_match_prob=0.5, eve_success_prob=1.0, alice_flip_prob=0.2)
        self.assertLess(I_noisy, I)


if __name__ == '__main__':
    unittest.main(verbosity=2)
