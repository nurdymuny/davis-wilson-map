"""Protocol types matching the GQL grammar in HALCYON_PART_I_GATES.md.

Every shape here corresponds to a GQL surface the real GIGI substrate
will expose. Field names match the response shapes in the sprint spec
verbatim so the mock and the live client are wire-compatible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Group(str, Enum):
    """Structure group tag. Only SU(2) has verb math at launch (Part II
    locked the group-erased storage layer with SU(N) / U(1) / Z(N) as
    compile-time tags that fail loudly until implemented)."""

    SU2 = "SU(2)"
    SU3 = "SU(3)"
    U1 = "U(1)"
    Z2 = "Z(2)"
    Z3 = "Z(3)"


class GaugeFieldInit(str, Enum):
    """INIT clause options. IDENTITY and HAAR_RANDOM both required at
    launch (Halcyon thermalization starts at IDENTITY; gauge-invariance
    gate starts at HAAR_RANDOM)."""

    IDENTITY = "IDENTITY"
    HAAR_RANDOM = "HAAR_RANDOM"
    FROM_FIELD = "FROM"


class ObservableId(str, Enum):
    """Observables exposed on MEASURE clauses at launch."""

    MEAN_PLAQUETTE = "MEAN(PLAQUETTE)"
    Q_SURROGATE = "Q_SURROGATE"
    H_TOTAL = "H_TOTAL"           # rejected pre-Part-IV (no E field yet)
    GAUSS_RESIDUAL_MAX = "GAUSS_RESIDUAL_MAX"


@dataclass(frozen=True)
class LatticeSpec:
    """Output of ``DECLARE LATTICE``. Pinned by Part I's incidence
    round-trip receipt."""

    name: str
    vertices: int
    edges: Tuple[Tuple[int, int], ...]
    faces: Tuple[Tuple[int, ...], ...]
    topology: str
    euler_characteristic: int


@dataclass(frozen=True)
class GaugeFieldHandle:
    """Handle returned by ``DECLARE GAUGE_FIELD``. The link buffer is
    NOT included in the handle (it lives in the GIGI engine); the
    handle is the cheap identifier the rest of the GQL surface threads
    through."""

    name: str
    lattice_name: str
    group: Group
    repr_dim: int
    init_kind: GaugeFieldInit
    init_seed: Optional[int] = None
    init_from: Optional[str] = None


@dataclass
class HolonomyResult:
    """Output of ``SELECT HOLONOMY ... ON LOOP (...)``.

    holonomies: ndarray of shape (n_loops, repr_dim). For SU(2)
    repr_dim = 4 (quaternion). Bit-identical reproducibility contract
    per ``HALCYON_TO_GIGI_REPLY_2026-06-17.md`` § A2.
    """

    holonomies: np.ndarray
    loop_count: int

    def __post_init__(self):
        if self.holonomies.ndim != 2:
            raise ValueError(
                f"holonomies must be (n_loops, repr_dim); got shape {self.holonomies.shape}"
            )
        if self.holonomies.shape[0] != self.loop_count:
            raise ValueError(
                f"loop_count mismatch: declared {self.loop_count}, "
                f"holonomies has {self.holonomies.shape[0]}"
            )


@dataclass
class GibbsSampleResult:
    """Output of ``GIBBS_SAMPLE``.

    final_field: GaugeFieldHandle to the post-sweep state.
    measurement_history: per-observable arrays of shape (n_records,);
        keys are ObservableId values.
    diagnostics: orchestrator-facing per-sweep counters (acceptance
        rates, wall-time per sweep, etc.).
    """

    final_field: GaugeFieldHandle
    measurement_history: Dict[str, np.ndarray]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class EFieldInit(str, Enum):
    """E_FIELD INIT clause (Part IV).

    ZERO              every Lie-row is (0,0,0,0). No seed required.
    MAXWELL_BOLTZMANN per-edge Lie sample with σ = sqrt(1/(β·1.5)).
                      SEED mandatory.
    FROM_FIELD        clone an already-registered E field's buffer.
    """

    ZERO = "ZERO"
    MAXWELL_BOLTZMANN = "MAXWELL_BOLTZMANN"
    FROM_FIELD = "FROM"


@dataclass(frozen=True)
class EFieldHandle:
    """Handle returned by ``DECLARE E_FIELD``."""

    name: str
    source_gauge_field: str
    source_lattice: str
    init_kind: EFieldInit
    init_beta: Optional[float] = None
    init_seed: Optional[int] = None
    init_from: Optional[str] = None


@dataclass(frozen=True)
class ProjectGaussConfig:
    """PROJECT_GAUSS struct clause for SYMPLECTIC_FLOW (Part IV, Q3).

    Defaults match GIGI's release-profile production canonical:
    ``{ tikhonov: 1e-14, cg_tol: 1e-10, cg_max_iter: 200 }``.
    """

    tikhonov: float = 1e-14
    cg_tol: float = 1e-10
    cg_max_iter: int = 200

    @classmethod
    def default(cls) -> "ProjectGaussConfig":
        return cls()


@dataclass
class SymplecticFlowDiagnostics:
    """Per-trajectory receipts from a SYMPLECTIC_FLOW call.

    cg_iterations_per_step_p99 is a *diagnostic*, not a regression
    gate (CG iteration count varies with β at fixed seed per the A2
    contract).
    """

    run_id: str
    beta: float
    dt: float
    n_steps_completed: int
    cg_iterations_per_step_p99: float
    max_energy_drift_rel: float
    gauss_residual_max: float
    seed: Optional[int] = None


@dataclass
class SymplecticFlowResult:
    """Output of ``SYMPLECTIC_FLOW`` (Part IV).

    The flow mutates the source U field in place; ``final_field`` is
    the echoed handle. Measurement chains live in
    ``measurement_history`` keyed by ObservableId.value.
    """

    final_field: GaugeFieldHandle
    final_e_field: EFieldHandle
    measurement_history: Dict[str, np.ndarray]
    diagnostics: SymplecticFlowDiagnostics


@dataclass(frozen=True)
class BitIdentityContract:
    """Receipt for what same-seed re-runs must reproduce, per A2.

    Used by the gate tests to assert the right thing without hard-
    coding bit-identity for parameters (β, dt, n_steps) that change
    the integrator's intermediate state.
    """

    same_process: bool = True
    cross_process_same_blas: bool = True
    cross_os: bool = False              # documented, not enforced
    same_seed_different_beta: bool = False
    same_seed_different_dt: bool = False
    prefix_for_extended_n_steps: bool = True
