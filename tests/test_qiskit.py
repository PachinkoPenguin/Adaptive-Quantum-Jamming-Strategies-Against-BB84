#!/usr/bin/env python3
"""
Test script to verify Qiskit installation and compatibility
Run this to check if Qiskit is properly installed and working
"""

import sys

print(f"Python version: {sys.version}")
print("-" * 60)

# Test 1: Basic Qiskit import
try:
    import qiskit

    print(f"✓ Qiskit version: {qiskit.__version__}")
except ImportError as e:
    print(f"✗ Qiskit import failed: {e}")
    sys.exit(1)

# Test 2: Core components
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    print("✓ Core Qiskit components imported")
except ImportError as e:
    print(f"✗ Core components import failed: {e}")

# Test 3: New Qiskit 1.0+ API (Aer moved to separate package)
try:
    from qiskit_aer import AerSimulator

    print("✓ Qiskit Aer (new API) available")
    use_new_api = True
except ImportError:
    print("✗ Qiskit Aer (new API) not found, trying old API...")
    try:
        from qiskit import Aer, execute

        print("✓ Qiskit Aer (old API) available")
        use_new_api = False
    except ImportError as e:
        print(f"✗ No Aer backend available: {e}")
        print("\nTo fix: pip install qiskit-aer")
        sys.exit(1)

# Test 4: Create and run a simple circuit
print("\n" + "=" * 60)
print("Testing circuit creation and execution...")
print("=" * 60)

try:
    # Create a simple Bell state circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    print("✓ Circuit created successfully")

    # Try to execute
    if use_new_api:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        transpiled = transpile(qc, backend)
        job = backend.run(transpiled, shots=1000)
        result = job.result()
        counts = result.get_counts()
        print(f"✓ Circuit executed (new API)")
        print(f"  Results: {counts}")
    else:
        from qiskit import Aer, execute

        backend = Aer.get_backend("qasm_simulator")
        job = execute(qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        print(f"✓ Circuit executed (old API)")
        print(f"  Results: {counts}")

except Exception as e:
    print(f"✗ Circuit execution failed: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Check optional components
print("\n" + "=" * 60)
print("Optional components check:")
print("=" * 60)

optional_packages = [
    ("qiskit_aer.noise", "Noise models"),
    ("qiskit.quantum_info", "Quantum information"),
    ("qiskit.visualization", "Visualization tools"),
    ("qiskit_ibm_runtime", "IBM Quantum Runtime"),
]

for module_name, description in optional_packages:
    try:
        module = __import__(module_name, fromlist=[""])
        print(f"✓ {description}: Available")
    except ImportError:
        print(f"○ {description}: Not installed (optional)")

# Test 6: BB84-specific test
print("\n" + "=" * 60)
print("BB84 Compatibility Test:")
print("=" * 60)

try:
    # Create a BB84 state
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Create superposition
    qc.measure(0, 0)

    if use_new_api:
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        transpiled = transpile(qc, backend)
        job = backend.run(transpiled, shots=100)
    else:
        backend = Aer.get_backend("qasm_simulator")
        job = execute(qc, backend, shots=100)

    result = job.result()
    counts = result.get_counts()

    print("✓ BB84-style circuit works")
    print(f"  Measurement distribution: {counts}")

    # Check if distribution is roughly 50/50
    if "0" in counts and "1" in counts:
        ratio = counts["0"] / (counts["0"] + counts["1"])
        if 0.3 < ratio < 0.7:
            print("✓ Quantum superposition verified (good randomness)")
        else:
            print("⚠ Unexpected measurement distribution")

except Exception as e:
    print(f"✗ BB84 test failed: {e}")

# Final summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)

if use_new_api:
    print("✓ Qiskit is properly installed with NEW API (1.0+)")
    print("✓ Ready to use Qiskit backend in BB84 simulation")
    print("\nTo use in BB84 simulation:")
    print("  python bb84_foundation.py")
    print("  # Then select Qiskit backend when prompted")
else:
    print("✓ Qiskit is installed but using OLD API")
    print("⚠ The BB84 code may need adjustments for old API")
    print("\nTo update to new API:")
    print("  pip install --upgrade qiskit qiskit-aer")

print("\n" + "=" * 60)
