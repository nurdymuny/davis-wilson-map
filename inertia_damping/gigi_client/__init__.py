"""Halcyon -> GIGI substrate client.

This package is the seam between Halcyon's existing Python kernel
(buckyball_graph / buckyball_action / buckyball_heatbath /
buckyball_integrator / buckyball_observables) and the future GIGI
GQL substrate that will host the same gauge-field math.

The protocol is locked against ``HALCYON_PART_I_GATES.md`` in the
GIGI repo (gates Part I -> Part II -> Part III -> measurement gate).

Until the real GIGI substrate ships Parts I-III, ``MockGIGIClient``
is the implementation under test. The mock is deliberately backed
by *frozen reference goldens* (not the live Python kernel), so the
gate tests assert against the same bit-identical truth a real GIGI
implementation will need to reproduce. When GIGI ships, replacing
``MockGIGIClient`` with the live client is the only change Halcyon
needs to make; the test suite stays.

Wire choice for the swap: **embedded binding (PyO3 / native CFFI)**,
NOT the HTTP surface. Two reasons: (1) the A2 same-process bit-
identity contract is cheapest to honor in-process; (2) GIGI's HTTP
``POST /v1/gauge_field`` declares in-memory only and would not
survive a gigi-stream restart, while embedded GQL is the path to
``Engine::declare_*_durable``. The HTTP surface is for consumers
(Marcella, dashboards, REPL exploration) — not the Halcyon
production path. See ``theory/halcyon/HALCYON_PART_I_GATES.md``.
"""
from inertia_damping.gigi_client.protocol import (
    Group,
    LatticeSpec,
    GaugeFieldHandle,
    GaugeFieldInit,
    ObservableId,
    GibbsSampleResult,
    HolonomyResult,
    EFieldInit,
    EFieldHandle,
    ProjectGaussConfig,
    SymplecticFlowDiagnostics,
    SymplecticFlowResult,
)
from inertia_damping.gigi_client.client import GIGIClient
from inertia_damping.gigi_client.mock import MockGIGIClient
from inertia_damping.gigi_client.live import LiveGIGIClient
from inertia_damping.gigi_client.matched_mock import MatchedRNGMockGIGIClient
from inertia_damping.gigi_client.matched_rng import (
    XorshiftSmallRng,
    haar_random_su2,
    maxwell_boltzmann_su2,
    haar_random_links,
    maxwell_boltzmann_links,
)

__all__ = [
    "Group",
    "LatticeSpec",
    "GaugeFieldHandle",
    "GaugeFieldInit",
    "ObservableId",
    "GibbsSampleResult",
    "HolonomyResult",
    "GIGIClient",
    "MockGIGIClient",
    "LiveGIGIClient",
    "EFieldInit",
    "EFieldHandle",
    "ProjectGaussConfig",
    "SymplecticFlowDiagnostics",
    "SymplecticFlowResult",
    "MatchedRNGMockGIGIClient",
    "XorshiftSmallRng",
    "haar_random_su2",
    "maxwell_boltzmann_su2",
    "haar_random_links",
    "maxwell_boltzmann_links",
]
