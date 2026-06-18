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
"""
from inertia_damping.gigi_client.protocol import (
    Group,
    LatticeSpec,
    GaugeFieldHandle,
    GaugeFieldInit,
    ObservableId,
    GibbsSampleResult,
    HolonomyResult,
)
from inertia_damping.gigi_client.client import GIGIClient
from inertia_damping.gigi_client.mock import MockGIGIClient

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
]
