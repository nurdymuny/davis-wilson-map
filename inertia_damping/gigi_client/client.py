"""Abstract GIGIClient — the seam Halcyon's orchestrator queries.

This is a typing.Protocol (structural), not an ABC, so the live
GIGI Python binding can satisfy it without importing this module.
The mock implements it explicitly; the live client just needs to
expose the same call shapes.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

from inertia_damping.gigi_client.protocol import (
    GaugeFieldHandle,
    GaugeFieldInit,
    GibbsSampleResult,
    Group,
    HolonomyResult,
    LatticeSpec,
    ObservableId,
)


class GIGIClient(Protocol):
    """Halcyon's view of the GIGI substrate. Every method maps 1:1
    onto a GQL verb in HALCYON_PART_I_GATES.md. The Halcyon orchestrator
    only ever talks to this surface."""

    # -----------------------------------------------------------------
    # Part I: LATTICE + generalized HOLONOMY
    # -----------------------------------------------------------------
    def declare_lattice(self, spec: LatticeSpec) -> str:
        """``DECLARE LATTICE name VERTICES n EDGES (...) FACES (...)
        TOPOLOGY "S2";`` -> returns the registered lattice name.
        Idempotent at the same spec: declaring twice returns the same
        name; declaring with a different shape errors."""
        ...

    def holonomy_on_faces(
        self,
        field_name: str,
    ) -> HolonomyResult:
        """``SELECT HOLONOMY OF field ON FACES OF LATTICE OF field;``
        -> per-face holonomies, shape (n_faces, 4) for SU(2). The Part
        I gate is this call's bit-identity against the Python kernel's
        ``buckyball_observables.all_face_holonomies``."""
        ...

    def holonomy_on_loop(
        self,
        field_name: str,
        edge_list: Sequence[int],
    ) -> HolonomyResult:
        """``SELECT HOLONOMY OF field ON LOOP (e0, e1, e2, ...);``
        -> single-loop holonomy. Used by the Section 4 gauge-invariance
        gate and by future Wilson-loop observables."""
        ...

    # -----------------------------------------------------------------
    # Part II: GAUGE_FIELD
    # -----------------------------------------------------------------
    def declare_gauge_field(
        self,
        name: str,
        lattice_name: str,
        group: Group,
        init: GaugeFieldInit,
        seed: Optional[int] = None,
        from_field: Optional[str] = None,
    ) -> GaugeFieldHandle:
        """``GAUGE_FIELD name ON LATTICE lat GROUP group INIT ...;``"""
        ...

    def introspect_gauge_field(self, name: str) -> np.ndarray:  # type: ignore[name-defined]
        """Read back the per-edge link buffer for a declared GAUGE_FIELD.

        Returns shape ``(n_edges, repr_dim)``. The Part II round-trip
        gate is: declare -> introspect -> redeclare from the buffer ->
        introspect -> bit-identical to the first introspection.

        This call is NOT how Halcyon normally reads field state — it's
        a diagnostic for the gate. In production, observables flow
        through MEASURE clauses.
        """
        ...

    # -----------------------------------------------------------------
    # Part III: PLAQUETTE + GIBBS_SAMPLE + observable battery
    # -----------------------------------------------------------------
    def select_plaquette(
        self,
        field_name: str,
        reduction: str = "per_face",  # "per_face" | "sum" | "mean"
    ) -> "np.ndarray | float":  # noqa: F821
        """``SELECT PLAQUETTE OF field;`` / ``SUM(...)`` / ``MEAN(...)``.

        per_face -> shape (n_faces,) of Re tr(U_f) / 2
        sum      -> scalar Wilson action / β
        mean     -> ⟨P⟩, the published observable
        """
        ...

    def select_observable(
        self,
        field_name: str,
        observable: ObservableId,
    ) -> float:
        """``SELECT Q_SURROGATE OF field;`` etc. Scalar observables
        only at launch."""
        ...

    def gibbs_sample(
        self,
        field_name: str,
        beta: float,
        n_sweeps: int = 1,
        measure_every: int = 1,
        measure: Sequence[ObservableId] = (),
        seed: Optional[int] = None,
    ) -> GibbsSampleResult:
        """``GIBBS_SAMPLE field BETA β N_SWEEPS n MEASURE_EVERY k
        MEASURE (...) SEED s;``"""
        ...


# Make numpy available for the type-hint string above.
import numpy as np  # noqa: E402
