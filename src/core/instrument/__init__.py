"""Instrument domain: state and smooth evolution of synthesis parameters.

The instrument state object is the immutable snapshot the dashboard and the
future DSP engine read from. The manager owns the current/target evolution
and produces a fresh snapshot on every update frame.
"""
