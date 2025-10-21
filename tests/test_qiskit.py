#!/usr/bin/env python3
"""
Test script to verify Qiskit installation and compatibility
Run this to check if Qiskit is properly installed and working
"""

import sys
import logging

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format for test output
)
logger = logging.getLogger(__name__)

logger.info(f"Python version: {sys.version}")
logger.info("-" * 60)

# Test 1: Basic Qiskit import
try:
    import qiskit

    logger.info(f"✓ Qiskit version: {qiskit.__version__}")
except ImportError as e:
    logger.error(f"✗ Qiskit import failed: {e}")
    sys.exit(1)

# Test 2: Core components
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    logger.info("✓ Core Qiskit components imported")
except ImportError as e:
    logger.error(f"✗ Core components import failed: {e}")

# Test 3: New Qiskit 1.0+ API (Aer moved to separate package)
try:
    from qiskit_aer import AerSimulator

    logger.info("✓ Qiskit Aer (new API) available")
    use_new_api = True
except ImportError:
    logger.warning("✗ Qiskit Aer (new API) not found, trying old API...")
    try:
        from qiskit import Aer, execute

        logger.info("✓ Qiskit Aer (old API) available")
        use_new_api = False
    except ImportError as e:
        logger.error(f"✗ No Aer backend available: {e}")
        logger.error("\nTo fix: pip install qiskit-aer")
        sys.exit(1)

# Test 4: Create and run a simple circuit
logger.info("\n" + "=" * 60)
logger.info("Testing circuit creation and execution...")
logger.info("=" * 60)

try:
    # Create a simple Bell state circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    logger.info("✓ Circuit created successfully")

    # Try to execute
    if use_new_api:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        transpiled = transpile(qc, backend)
        job = backend.run(transpiled, shots=1000)
        result = job.result()
        counts = result.get_counts()
        logger.info(f"✓ Circuit executed (new API)")
        logger.info(f"  Results: {counts}")
    else:
        from qiskit import Aer, execute

        backend = Aer.get_backend("qasm_simulator")
        job = execute(qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        logger.info(f"✓ Circuit executed (old API)")
        logger.info(f"  Results: {counts}")

except Exception as e:
    logger.error(f"✗ Circuit execution failed: {e}")
    import traceback
    logger.error(traceback.format_exc())

# Test 5: Check optional components
logger.info("\n" + "=" * 60)
logger.info("Optional components check:")
logger.info("=" * 60)

optional_packages = [
    ("qiskit_aer.noise", "Noise models"),
    ("qiskit.quantum_info", "Quantum information"),
    ("qiskit.visualization", "Visualization tools"),
    ("qiskit_ibm_runtime", "IBM Quantum Runtime"),
]

for module_name, description in optional_packages:
    try:
        module = __import__(module_name, fromlist=[""])
        logger.info(f"✓ {description}: Available")
    except ImportError:
        logger.info(f"○ {description}: Not installed (optional)")

# Test 6: BB84-specific test
logger.info("\n" + "=" * 60)
logger.info("BB84 Compatibility Test:")
logger.info("=" * 60)

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

    logger.info("✓ BB84-style circuit works")
    logger.info(f"  Measurement distribution: {counts}")

    # Check if distribution is roughly 50/50
    if "0" in counts and "1" in counts:
        ratio = counts["0"] / (counts["0"] + counts["1"])
        if 0.3 < ratio < 0.7:
            logger.info("✓ Quantum superposition verified (good randomness)")
        else:
            logger.warning("⚠ Unexpected measurement distribution")

except Exception as e:
    logger.error(f"✗ BB84 test failed: {e}")

# Final summary
logger.info("\n" + "=" * 60)
logger.info("SUMMARY:")
logger.info("=" * 60)

if use_new_api:
    logger.info("✓ Qiskit is properly installed with NEW API (1.0+)")
    logger.info("✓ Ready to use Qiskit backend in BB84 simulation")
    logger.info("\nTo use in BB84 simulation:")
    logger.info("  python bb84_foundation.py")
    logger.info("  # Then select Qiskit backend when prompted")
else:
    logger.info("✓ Qiskit is installed but using OLD API")
    logger.warning("⚠ The BB84 code may need adjustments for old API")
    logger.info("\nTo update to new API:")
    logger.info("  pip install --upgrade qiskit qiskit-aer")

logger.info("\n" + "=" * 60)
