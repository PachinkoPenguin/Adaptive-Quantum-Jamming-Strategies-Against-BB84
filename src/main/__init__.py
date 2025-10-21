"""
Main BB84 implementation package

Expose the main public classes and constants to allow importing from `main`.
"""

# Explicit exports
from .bb84_main import BB84Protocol, BB84Simulator, QISKIT_AVAILABLE

__all__ = [
	"BB84Protocol",
	"BB84Simulator",
	"QISKIT_AVAILABLE",
]