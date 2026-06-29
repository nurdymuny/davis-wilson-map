"""The two loops v3.1.3 declares on the (Q, β_W) control manifold.

Per v3.1.3 §4.1 and §5. Both loops carry the same axes
(Q, beta_wilson), same T_loop = 200, T_segment = 50. γ_unit is a closed
rectangle enclosing area 1.0 inside the validated SU(2) operating
window (β_W ∈ [2.5, 3.0]). γ_degenerate is a zero-area loop used for
the SHAM_BACKTRACK_LOOP (S₅) test.

The reverse traversal γ_unit⁻¹ is NOT declared as a separate loop.
The substrate time-reverses γ_unit in the executor per CC-LT-7 (GIGI v2
§7 + Halcyon v1 §C.1). The orchestrator passes `direction="REVERSED"`
on the LoopTransportRequest; the substrate handles the rest.
"""
from __future__ import annotations

from inertia_damping.gigi_client.loop_transport import LoopHandle


# γ_unit: closed rectangle in (Q, β_W) per v3.1.3 §4.1.
# Traversal order: (0, 2.5) → (2, 2.5) → (2, 3.0) → (0, 3.0) → (0, 2.5).
# Loop name matches the substrate's VI.5 gold fixture
# (`tests/fixtures/halcyon/part_vi/loop_transport_canonical.json` —
# "loop": "gamma_unit_in_Q_beta_W") + Gigi's calling guide §Preconditions.
GAMMA_UNIT = LoopHandle(
    name="gamma_unit_in_Q_beta_W",
    control_manifold_axes=("Q", "beta_wilson"),
    vertices=(
        (0.0, 2.5),
        (2.0, 2.5),
        (2.0, 3.0),
        (0.0, 3.0),
        (0.0, 2.5),
    ),
    t_per_segment=50.0,
    enclosed_area=1.0,  # Q_max · Δβ_W = 2 × 0.5
)


# γ_degenerate: zero-area loop for SHAM_BACKTRACK_LOOP (S₅).
# Single-point "loop" at (Q=0, β_W=2.5) — Migdal-Witten canonical point.
# Substrate's executor reads this as a zero-segment path; holonomy = identity
# by construction. Any non-machine-ε reading indicates a substrate bug.
GAMMA_DEGENERATE = LoopHandle(
    name="gamma_degenerate",
    control_manifold_axes=("Q", "beta_wilson"),
    vertices=(
        (0.0, 2.5),
        (0.0, 2.5),
    ),
    t_per_segment=50.0,
    enclosed_area=0.0,
)
