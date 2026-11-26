"""The imagination detector: does a network build and use a model of the scene?

Behaviour, linear probes, and interchange interventions, calibrated on subjects
with known internals. See docs/IMAGINATION_DETECTOR.md and run
``python -m hyperphantasia.detect``. The battery and calibration subjects need
only numpy; the trainable networks need JAX (``pip install -e ".[detect]"``).
"""
