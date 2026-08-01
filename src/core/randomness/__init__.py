"""Randomness subsystem: provider, bit stream, chunking and decoding.

The subsystem is deliberately decoupled from the dashboard and from any
specific entropy source. A ``RandomnessProvider`` is the only object that
knows where bits come from (NumPy PRNG or Qiskit Aer quantum-circuit
simulation); everything downstream works purely on binary strings.
"""
