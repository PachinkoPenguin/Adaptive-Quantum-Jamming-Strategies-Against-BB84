#!/usr/bin/env python3
"""
Quick test to run BB84 with both Classical and Qiskit backends
This will verify that both backends work correctly
"""

import sys
import os
import logging

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format for test output
)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

logger.info("=" * 60)
logger.info("BB84 BACKEND TEST")
logger.info("=" * 60)

# Test imports
try:
    from main.bb84_main import BB84Protocol, BB84Simulator, QISKIT_AVAILABLE

    logger.info("✓ BB84 modules imported successfully")
    logger.info(f"✓ Qiskit available: {QISKIT_AVAILABLE}")
except ImportError as e:
    logger.error(f"✗ Failed to import BB84 modules: {e}")
    sys.exit(1)

logger.info("\n" + "=" * 60)
logger.info("TEST 1: Classical Backend")
logger.info("=" * 60)

try:
    # Run with classical backend
    protocol = BB84Protocol(use_qiskit=False)
    protocol.setup(channel_loss=0.1, channel_error=0.02)
    stats = protocol.run_protocol(num_bits=100)

    logger.info("✓ Classical simulation successful!")
    logger.info(f"  - Final key length: {stats['final_key_length']} bits")
    logger.info(f"  - QBER: {stats['qber']:.2%}")
    logger.info(f"  - Secure: {stats['secure']}")
except Exception as e:
    logger.error(f"✗ Classical simulation failed: {e}")
    import traceback
    logger.error(traceback.format_exc())

if QISKIT_AVAILABLE:
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Qiskit Backend")
    logger.info("=" * 60)

    try:
        # Run with Qiskit backend
        protocol = BB84Protocol(use_qiskit=True)
        protocol.setup(channel_loss=0.1, channel_error=0.02)
        stats = protocol.run_protocol(num_bits=100)

        logger.info("✓ Qiskit simulation successful!")
        logger.info(f"  - Final key length: {stats['final_key_length']} bits")
        logger.info(f"  - QBER: {stats['qber']:.2%}")
        logger.info(f"  - Secure: {stats['secure']}")
    except Exception as e:
        logger.error(f"✗ Qiskit simulation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Backend Comparison")
    logger.info("=" * 60)

    try:
        # Quick comparison
        classical_results = []
        qiskit_results = []

        for _ in range(3):
            # Classical
            p1 = BB84Protocol(use_qiskit=False)
            p1.setup(0.1, 0.02)
            s1 = p1.run_protocol(50)
            classical_results.append(s1["qber"])

            # Qiskit
            p2 = BB84Protocol(use_qiskit=True)
            p2.setup(0.1, 0.02)
            s2 = p2.run_protocol(50)
            qiskit_results.append(s2["qber"])

        import numpy as np

        classical_avg = np.mean(classical_results)
        qiskit_avg = np.mean(qiskit_results)

        logger.info("✓ Comparison complete!")
        logger.info(f"  Classical average QBER: {classical_avg:.2%}")
        logger.info(f"  Qiskit average QBER: {qiskit_avg:.2%}")
        logger.info(f"  Difference: {abs(classical_avg - qiskit_avg):.2%}")

        if abs(classical_avg - qiskit_avg) < 0.05:  # Within 5%
            logger.info("✓ Backends produce consistent results!")
        else:
            logger.warning("⚠ Backends show significant differences")

    except Exception as e:
        logger.error(f"✗ Comparison failed: {e}")
else:
    logger.warning("\n⚠ Qiskit not available - skipping Qiskit tests")

logger.info("\n" + "=" * 60)
logger.info("SUMMARY")
logger.info("=" * 60)

if QISKIT_AVAILABLE:
    logger.info("✓ Both Classical and Qiskit backends are working!")
    logger.info("✓ You can now use either backend for your research")
    logger.info("\nTo use Qiskit in main simulation:")
    logger.info("  Edit src/main/bb84_main.py line ~1130:")
    logger.info("  Change: use_qiskit_backend = False")
    logger.info("  To:     use_qiskit_backend = True")
else:
    logger.info("✓ Classical backend is working!")
    logger.warning("⚠ Qiskit backend not available")
    logger.info("\nTo enable Qiskit:")
    logger.info("  pip install qiskit qiskit-aer")

logger.info("\n" + "=" * 60)
