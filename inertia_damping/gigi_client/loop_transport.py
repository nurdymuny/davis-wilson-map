"""LOOP_TRANSPORT verb — Halcyon-side typed contract + mock client.

This module is the Halcyon Python orchestrator's view of the GIGI
substrate's `LOOP_TRANSPORT` verb (per Halcyon SPEC v3.1.3 §4.4 and the
substrate-side GIGI→Halcyon reply v2). The verb does not exist in the
live GIGI engine yet; this module defines what its surface looks like
so the orchestrator and gate logic can be developed and tested today.

When GIGI's verb ships:
- The live binding implements the `LoopTransportClient` Protocol.
- `MockLoopTransportClient` is swapped for the live client in
  `run_holonomy_battery.py`. Nothing else changes.

The pre-registration commitment for the verb signature is at
HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.3.md commit 44c70b1,
Zenodo DOI 10.5281/zenodo.20785681. The `ParameterPackKind::Halcyon`
field list comes from the v1 reply §B.2 / GIGI v2 §1 acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Loop registry (CC-LT-1 path (a), per GIGI v2 §7 + Halcyon v1 §C.1)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LoopHandle:
    """First-class declarable loop object. v3.1.3 declares two loops
    (`gamma_unit` rectangular in (Q, β_W); `gamma_degenerate` zero-area).
    γ_unit⁻¹ is NOT a separate handle; the substrate time-reverses
    `gamma_unit` in the executor per CC-LT-7."""

    name: str
    control_manifold_axes: Tuple[str, ...]   # ("Q", "beta_wilson")
    vertices: Tuple[Tuple[float, float], ...]  # corners; closed loop
    t_per_segment: float                      # 50.0 for v3.1.3
    enclosed_area: float                      # 1.0 for γ_unit; 0.0 for γ_degenerate

    def n_segments(self) -> int:
        return max(0, len(self.vertices) - 1)


# ----------------------------------------------------------------------
# ParameterPackKind::Halcyon (CC-LT-2, locked per v1 §B.2)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HalcyonParameterPack:
    """The first registered variant of `ParameterPackKind` per CC-LT-2.
    Canonical field list locked at v1 §B.2 / GIGI v2 §1.

    All fields are kind-specific (Halcyon-shaped). `drive_omega` and
    `drive_F0` are inside the Halcyon variant — the lock-in protocol is
    Halcyon-specific, not a substrate primitive (future variants like
    StaggeredFermion or AURORA won't have them).

    Pre-registered values from v3.1.3 §4.4 are the defaults below.
    """

    # Coupling and substrate model (the falsifiable physics)
    alpha: float = 1.0          # ALPHA_HALCYON; v3.1.3 runs at {1, 1000}
    tau_0: float = 1.0
    beta_tau: float = 2.0       # τ_Q model coefficient (NOT Wilson β_W)
    mu_baseline: float = 1.0
    # Test-mass dynamics (Halcyon lock-in protocol)
    K_spring: float = 1.0
    c_damp: float = 0.1
    drive_omega: float = 1.0
    drive_F0: float = 0.01
    # Active-pinning gates (v3.1.3 §4.3, regime B)
    pin_lambda_Q: float = 1.0
    pin_lambda_beta_W: float = 1.0
    eps_Q: float = 0.05
    eps_beta_W: float = 0.05


# ----------------------------------------------------------------------
# Sham flags (v3.1.3 §5 + design-closeout §D.1 disambiguation)
# ----------------------------------------------------------------------


class ShamFlag(str, Enum):
    """v3.1.3's five sham controls (S₁..S₅) plus GIGI's substrate-internal
    audit-story flags. S₄ has no flag — folded into the antisymmetric
    primary observable per §3.1.

    The required science-gate set for v3.1.3 is:
        NONE, FLAT_FIELD, ALPHA_ZERO, MASS_SCALED, BACKTRACK_LOOP,
        FROZEN_FIELD.
    EMPTY_LOOP and OPEN_LOOP are substrate-internal contracts
    (GC₄ companion + parser-rejection test); Halcyon does not gate on
    them, but the substrate may ship them under the same SHAM block.
    """

    NONE = "NONE"                       # primary call, no sham
    FLAT_FIELD = "FLAT_FIELD"           # S₁: κ_Q ≡ 0 on all edges
    ALPHA_ZERO = "ALPHA_ZERO"           # S₂: ALPHA_HALCYON = 0
    MASS_SCALED = "MASS_SCALED"         # S₃: MU_BASELINE scaled (set sham_mass_scale)
    BACKTRACK_LOOP = "BACKTRACK_LOOP"   # S₅ canonical mapping (zero-area path)
    FROZEN_FIELD = "FROZEN_FIELD"       # S₆: U held; (Q_target, β_W_target) updated
    # Substrate-internal audit-story flags (not science-gate shams)
    EMPTY_LOOP = "EMPTY_LOOP"           # GC₄ companion: no-segment loop
    OPEN_LOOP = "OPEN_LOOP"             # parser-rejection test


# ----------------------------------------------------------------------
# ADIABATICITY_CHECK return shape (v3.1.3 §4.2 + Halcyon v1 §C.3)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AdiabaticityCheck:
    """Substrate-side adiabaticity diagnostics. v3.1.3 has two
    observables:

    Observable A: `tau_pin_over_t_segment` is the active-pinning
        equilibration ratio. Halcyon's Python applies the pre-registered
        gate `>= 0.1` to force AMBIGUOUS (per v3.1.3 §4.2).
    Observable B: `ramp_rate_over_relaxation_rate` is the t013
        three-constraint diagnostic (substrate's analytical formula —
        candidate (a): `||dU/dt||_op = g² · sup_edges ||E_edge||`).
        v3.1.3 does NOT pre-register a numerical threshold for this;
        sidecar records it as a diagnostic.

    `warnings_count` and `warning_substep_indices` are substrate-side
    self-sanity instrumentation (e.g., NaN propagation, negative τ_pin),
    NOT Halcyon-protocol gates per Halcyon v1 §C.3.
    """

    tau_pin_over_t_segment: float
    ramp_rate_over_relaxation_rate: float
    warnings_count: int = 0
    warning_substep_indices: Tuple[int, ...] = ()


# ----------------------------------------------------------------------
# Request + Result (v3.1.3 §4.4 verb call)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LoopTransportRequest:
    """One LOOP_TRANSPORT call. The Halcyon orchestrator issues one of
    these per (α calibration × loop × direction × sham flag). v3.1.3
    runs ~12-32 calls per α (depending on which shams are exercised in a
    given session).

    `direction` is "FORWARD" or "REVERSED"; substrate time-reverses the
    loop traversal for REVERSED (CC-LT-7). The Halcyon orchestrator
    forms `H_geom = ½(H_fwd - H_rev)` and `H_sys = ½(H_fwd + H_rev)`
    at the response boundary."""

    gauge_field_name: str
    loop: LoopHandle
    direction: str = "FORWARD"  # "FORWARD" | "REVERSED"
    ramp_rate_Q: float = 0.04
    ramp_rate_beta_W: float = 0.01
    n_discretization: int = 10000
    pack: HalcyonParameterPack = field(default_factory=HalcyonParameterPack)
    seeds: Tuple[int, ...] = tuple(range(20260616, 20260624))
    sham: ShamFlag = ShamFlag.NONE
    sham_mass_scale: Optional[float] = None  # only when sham=MASS_SCALED

    def __post_init__(self):
        if self.direction not in ("FORWARD", "REVERSED"):
            raise ValueError(
                f"direction must be FORWARD or REVERSED; got {self.direction!r}"
            )
        if self.sham is ShamFlag.MASS_SCALED and self.sham_mass_scale is None:
            raise ValueError("sham=MASS_SCALED requires sham_mass_scale")
        if self.sham is not ShamFlag.MASS_SCALED and self.sham_mass_scale is not None:
            raise ValueError(
                f"sham_mass_scale only valid with sham=MASS_SCALED; got sham={self.sham}"
            )


@dataclass(frozen=True)
class LoopTransportResult:
    """Substrate's return from one LOOP_TRANSPORT call. Per v3.1.3 §4.4:
    per-seed holonomies, Flyvbjerg-Petersen blocked σ, per-axis
    tracking-error max values, the adiabaticity check, a run identifier
    for the WAL receipt.

    `per_seed_h_scalar` is the abelianized scalar projection of each
    per-seed `GroupElement` (the substrate's choice of projection — the
    SU(2) trace divided by 2 minus 1 is the canonical one). The
    orchestrator uses the scalar for verdict classification and sign
    coherence; the quaternion is preserved for the sidecar audit trail.
    """

    per_seed_holonomy: np.ndarray             # shape (n_seeds, 4) - SU(2) quaternion
    per_seed_h_scalar: np.ndarray             # shape (n_seeds,) - real abelianized
    sigma_h_blocked: float                     # FP-blocked SEM across seeds
    tracking_error_max_Q: float
    tracking_error_max_beta_W: float
    adiabaticity_check: AdiabaticityCheck
    run_id: str

    def __post_init__(self):
        if self.per_seed_holonomy.ndim != 2 or self.per_seed_holonomy.shape[1] != 4:
            raise ValueError(
                f"per_seed_holonomy must be (n_seeds, 4) SU(2) quaternions; "
                f"got shape {self.per_seed_holonomy.shape}"
            )
        if self.per_seed_h_scalar.ndim != 1:
            raise ValueError(
                f"per_seed_h_scalar must be 1D; got shape {self.per_seed_h_scalar.shape}"
            )
        if self.per_seed_h_scalar.shape[0] != self.per_seed_holonomy.shape[0]:
            raise ValueError(
                f"per_seed length mismatch: holonomy has {self.per_seed_holonomy.shape[0]} "
                f"seeds, scalar has {self.per_seed_h_scalar.shape[0]}"
            )

    def mean_h_scalar(self) -> float:
        """Ensemble mean of the abelianized scalar projection."""
        return float(np.mean(self.per_seed_h_scalar))


# ----------------------------------------------------------------------
# Abstract client interface
# ----------------------------------------------------------------------


class LoopTransportClient(Protocol):
    """Halcyon's view of the LOOP_TRANSPORT verb. Single-method protocol
    per CC-LT-5. The live GIGI binding satisfies this implicitly
    (structural typing); the mock implements it explicitly for testing.

    The substrate-side acceptance battery `GC₁`–`GC₆` (v3.1.3 §7.4)
    is the substrate's responsibility, not this Protocol's. Clients
    that fail `GC₁`–`GC₆` are out-of-contract."""

    def declare_loop(self, loop: LoopHandle) -> str:
        """`DECLARE LOOP foo ...` → registered loop name.
        Idempotent: declaring the same loop twice returns the same name;
        declaring a different shape under the same name errors."""
        ...

    def loop_transport(
        self,
        request: LoopTransportRequest,
    ) -> LoopTransportResult:
        """One `LOOP_TRANSPORT` call per v3.1.3 §4.4."""
        ...
