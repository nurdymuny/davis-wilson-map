"""Live-substrate smoke test for `LiveLoopTransportClient`.

SKIP-BY-DEFAULT. Tests run only when both of the following are set:

    HALCYON_LIVE_SMOKE=1
    GIGI_URL=http://...    (or the default localhost:3142 if local)

This test is INTEGRATION TESTING, not science. It verifies that:

  1. The HTTP path from `LiveLoopTransportClient` → `POST /v1/gql` →
     parsed `LoopTransportResult` works end-to-end.
  2. The GQL grammar this client emits matches the substrate's parser
     (catches any mismatch between Halcyon's best-guess GQL and
     Gigi's actual VI.2 parser arm).
  3. The orchestrator can swap mock → live and produce *some*
     `CompositeVerdict` (likely AMBIGUOUS-because-shams-fail until VI.4,
     which is the right behavior — meaningful verdicts require shams
     that actually do something, and the smoke test is not the
     publication-bound run).

The publication-bound run waits for VI.4 + VI.5 per Halcyon's design
closeout. This file's purpose is to fail fast on integration bugs
before they contaminate the publication call.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from inertia_damping.gigi_client.loop_transport import (
    HalcyonParameterPack,
    LoopTransportRequest,
    ShamFlag,
)
from inertia_damping.holonomy_battery.loops import GAMMA_UNIT


SMOKE_ENABLED = os.environ.get("HALCYON_LIVE_SMOKE", "").lower() in ("1", "true", "yes")
SKIP_REASON = (
    "Live smoke disabled. Set HALCYON_LIVE_SMOKE=1 to enable; "
    "ensure GIGI_URL points at the live engine (or leave unset for "
    "localhost:3142 with a running `cargo run --release --bin gigi-stream`)."
)


pytestmark = pytest.mark.skipif(not SMOKE_ENABLED, reason=SKIP_REASON)


# A tiny seed set + small N for the smoke test — exercise the wire
# round-trip without paying for the full v3.1.3 canonical run cost.
SMOKE_SEEDS = (20260616, 20260617)
SMOKE_N = 1000  # cheap; gates not meaningful at this N (GC₅ requires 10000)


def _build_client():
    """Lazy import so collection still works even when SMOKE_ENABLED is
    false and requests may not be installed."""
    from inertia_damping.gigi_client.live_loop_transport import (
        LiveLoopTransportClient,
    )
    return LiveLoopTransportClient(ping=False)


def test_smoke_declare_loop_round_trips():
    """`declare_loop(GAMMA_UNIT)` succeeds and is idempotent on the
    live substrate. First call: substrate accepts the DECLARE LOOP
    GQL grammar; second call: returns the same handle name without
    re-declaring (substrate-side idempotency or client-side cache)."""
    client = _build_client()
    name1 = client.declare_loop(GAMMA_UNIT)
    name2 = client.declare_loop(GAMMA_UNIT)
    assert name1 == GAMMA_UNIT.name
    assert name1 == name2


def test_smoke_loop_transport_minimal_call_round_trips():
    """One minimal `LOOP_TRANSPORT` call. Verifies the GQL string
    Halcyon generates matches the substrate's parser, the response
    Rows envelope unpacks into a `LoopTransportResult` with the right
    shape, and the per-seed scalar vector has the expected length.

    The numerical values produced here are NOT trusted — N is small,
    seeds are small, and any sham gates would fail at this scale. This
    test only verifies the integration path is wired correctly."""
    client = _build_client()
    client.declare_loop(GAMMA_UNIT)

    request = LoopTransportRequest(
        gauge_field_name="halcyon_canonical_buckyball",
        loop=GAMMA_UNIT,
        direction="FORWARD",
        pack=HalcyonParameterPack(),
        seeds=SMOKE_SEEDS,
        n_discretization=SMOKE_N,
    )
    result = client.loop_transport(request)

    # Shape contract
    assert result.per_seed_h_scalar.shape == (len(SMOKE_SEEDS),)
    assert result.per_seed_holonomy.shape == (len(SMOKE_SEEDS), 4)
    assert result.sigma_h_blocked >= 0.0
    # Tracking errors and adiabaticity are substrate diagnostics —
    # they should be present and finite.
    assert result.tracking_error_max_Q >= 0.0
    assert result.tracking_error_max_beta_W >= 0.0
    assert result.adiabaticity_check.tau_pin_over_t_segment >= 0.0
    # The substrate should return a non-empty run_id so the WAL receipt
    # is locatable post-hoc.
    assert result.run_id != ""


def test_smoke_orchestrator_swaps_mock_for_live_end_to_end():
    """The orchestrator's `run_one_calibration` succeeds against the
    live client. Verdict will likely be AMBIGUOUS until VI.4 lands —
    that's expected. The test only verifies the gate-application path
    survives the live client substitution; the verdict numerical
    correctness is the responsibility of VI.4 + VI.5."""
    from inertia_damping.run_holonomy_battery import run_one_calibration

    client = _build_client()
    composite, forward, reversed_, sham_results = run_one_calibration(
        client,
        alpha_halcyon=1.0,
        seeds=SMOKE_SEEDS,
    )

    # Verdict is in the enum set (don't assert which one — it may
    # legitimately be AMBIGUOUS until VI.4)
    assert composite.overall.value in {"POSITIVE", "NULL", "AMBIGUOUS"}
    # Substrate emitted diagnostics for both directions
    assert forward.run_id != ""
    assert reversed_.run_id != ""
    # All 5 science-gate shams returned data
    assert len(sham_results) == 5
