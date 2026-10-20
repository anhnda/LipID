"""LipID v1.1 reproducibility package (protocol lipid-v3).

The historical modules at the repository root (phase1.py, nested_anchors_v2.py, envelope.py, bounds.py,
t6_directional.py, ...) are kept unchanged: they produced the P0'' runs reported in the draft.
Everything new lives here and only *imports* them, so the historical arithmetic is reused bit for bit.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PROTOCOL = "lipid-v3.0"
