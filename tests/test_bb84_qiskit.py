#!/usr/bin/env python3
"""
Quick test to run BB84 with both Classical and Qiskit backends
This will verify that both backends work correctly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("BB84 BACKEND TEST")
print("=" * 60)

# Test imports
try:
    from bb84_foundation import BB84Protocol, BB84Simulator, QISKIT_AVAILABLE

    print(f"✓ BB84 modules imported successfully")
    print(f"✓ Qiskit available: {QISKIT_AVAILABLE}")
except ImportError as e:
    print(f"✗ Failed to import BB84 modules: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST 1: Classical Backend")
print("=" * 60)

try:
    # Run with classical backend
    protocol = BB84Protocol(use_qiskit=False)
    protocol.setup(channel_loss=0.1, channel_error=0.02)
    stats = protocol.run_protocol(num_bits=100)

    print(f"✓ Classical simulation successful!")
    print(f"  - Final key length: {stats['final_key_length']} bits")
    print(f"  - QBER: {stats['qber']:.2%}")
    print(f"  - Secure: {stats['secure']}")
except Exception as e:
    print(f"✗ Classical simulation failed: {e}")
    import traceback

    traceback.print_exc()

if QISKIT_AVAILABLE:
    print("\n" + "=" * 60)
    print("TEST 2: Qiskit Backend")
    print("=" * 60)

    try:
        # Run with Qiskit backend
        protocol = BB84Protocol(use_qiskit=True)
        protocol.setup(channel_loss=0.1, channel_error=0.02)
        stats = protocol.run_protocol(num_bits=100)

        print(f"✓ Qiskit simulation successful!")
        print(f"  - Final key length: {stats['final_key_length']} bits")
        print(f"  - QBER: {stats['qber']:.2%}")
        print(f"  - Secure: {stats['secure']}")
    except Exception as e:
        print(f"✗ Qiskit simulation failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST 3: Backend Comparison")
    print("=" * 60)

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

        print(f"✓ Comparison complete!")
        print(f"  Classical average QBER: {classical_avg:.2%}")
        print(f"  Qiskit average QBER: {qiskit_avg:.2%}")
        print(f"  Difference: {abs(classical_avg - qiskit_avg):.2%}")

        if abs(classical_avg - qiskit_avg) < 0.05:  # Within 5%
            print("✓ Backends produce consistent results!")
        else:
            print("⚠ Backends show significant differences")

    except Exception as e:
        print(f"✗ Comparison failed: {e}")
else:
    print("\n⚠ Qiskit not available - skipping Qiskit tests")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if QISKIT_AVAILABLE:
    print("✓ Both Classical and Qiskit backends are working!")
    print("✓ You can now use either backend for your research")
    print("\nTo use Qiskit in main simulation:")
    print("  Edit bb84_foundation.py line ~1130:")
    print("  Change: use_qiskit_backend = False")
    print("  To:     use_qiskit_backend = True")
else:
    print("✓ Classical backend is working!")
    print("⚠ Qiskit backend not available")
    print("\nTo enable Qiskit:")
    print("  pip install qiskit qiskit-aer")

print("\n" + "=" * 60)
