"""Halcyon Falsification Battery v3.1.3 — Halcyon-side Python
orchestrator + gate logic + sidecar emission.

Pre-registration:
- SPEC: HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.3.md, commit 44c70b1
- Zenodo: https://doi.org/10.5281/zenodo.20785681
- Git tag: spec-v3.1.3-zenodo-20785681

This module is the thin delegation wrapper per v3.1.3 §4.6. It contains
NO substrate-relevant computation; the only physics-bearing call is to
the GIGI substrate's LOOP_TRANSPORT verb via the LoopTransportClient
interface.
"""
from inertia_damping.holonomy_battery.loops import (
    GAMMA_DEGENERATE,
    GAMMA_UNIT,
)
from inertia_damping.holonomy_battery.gates import (
    V313_CONSTANTS,
    Verdict,
    V313Constants,
    classify,
)
from inertia_damping.holonomy_battery.sidecar import (
    Section12Sidecar,
    write_sidecar,
)

__all__ = [
    "GAMMA_DEGENERATE",
    "GAMMA_UNIT",
    "Section12Sidecar",
    "V313_CONSTANTS",
    "V313Constants",
    "Verdict",
    "classify",
    "write_sidecar",
]
