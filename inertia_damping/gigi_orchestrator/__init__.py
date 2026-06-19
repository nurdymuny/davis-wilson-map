"""inertia_damping.gigi_orchestrator -- GIGI-backed phase runners for
the Halcyon orchestrator's ``--use-gigi`` flag.

This package is the swap layer between ``run_validation_report.py``
and GIGI's HTTP/GQL substrate. Each phase runner in
``phase_runners.py`` mirrors the return shape of its Python-kernel
counterpart so the orchestrator's downstream phases (5 / 8c / 9 / 10
/ 11-13) and the falsifiability validator
(``inertia_damping/validation_report.py``) stay unchanged.

See ``inertia_damping/HALCYON_USE_GIGI_FLAG_SPEC.md`` for the design.
"""
from inertia_damping.gigi_orchestrator.phase_runners import (
    LATTICE_NAME,
    U_THERM_NAME,
    E_CANONICAL_NAME,
    U_CANONICAL_NAME,
    declare_lattice_phase,
    thermalize_via_gigi,
    initialize_E_via_gigi,
    leapfrog_via_gigi,
    microcanonical_xcheck_via_gigi,
    canonical_heatbath_via_gigi,
    beta_scan_via_gigi,
    beta_envelope_via_gigi,
    snapshot_canonical_via_gigi,
)
from inertia_damping.gigi_orchestrator.translate_chains import (
    MEASUREMENT_KEY_MAP,
    LEGACY_SHORT_KEY,
    normalize_measurement_history,
    to_legacy_chains,
    gauge_buffer_to_torch,
    gauss_history_to_2d,
)

__all__ = [
    # phase_runners
    "LATTICE_NAME",
    "U_THERM_NAME",
    "E_CANONICAL_NAME",
    "U_CANONICAL_NAME",
    "declare_lattice_phase",
    "thermalize_via_gigi",
    "initialize_E_via_gigi",
    "leapfrog_via_gigi",
    "microcanonical_xcheck_via_gigi",
    "canonical_heatbath_via_gigi",
    "beta_scan_via_gigi",
    "beta_envelope_via_gigi",
    "snapshot_canonical_via_gigi",
    # translate_chains
    "MEASUREMENT_KEY_MAP",
    "LEGACY_SHORT_KEY",
    "normalize_measurement_history",
    "to_legacy_chains",
    "gauge_buffer_to_torch",
    "gauss_history_to_2d",
]
