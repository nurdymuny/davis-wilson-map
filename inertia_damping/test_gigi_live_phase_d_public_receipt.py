"""Phase D live test — public-receipt verification path.

Exercises ``inertia_damping/scripts/verify_canonical_receipt.py``
against the live GIGI engine. This is the test that confirms a
chapter reader holding ``gigi-stream.fly.dev`` can run the verifier
and watch the band check pass against Halcyon's spine
(⟨P⟩ = 0.5068472 ± 0.0014580).

The receipt this pins:
  - The 5-statement chapter GQL block accepts cleanly on the live
    engine.
  - The 100-sweep tail mean of MEAN(PLAQUETTE) lands within 3σ of
    the Halcyon spine on the GIGI substrate's CSPRNG path.
  - The verifier returns a SHA-256 of the measurement chain that a
    reader can cite for any independently-derived numerical claim.

Skips cleanly when ``gigi-stream`` isn't reachable. To run against
production from the Halcyon side:

    GIGI_URL=https://gigi-stream.fly.dev \\
    GIGI_API_KEY=$YOUR_KEY \\
    python -m pytest inertia_damping/test_gigi_live_phase_d_public_receipt.py -v
"""
from __future__ import annotations

import os
import time

import pytest

from inertia_damping.scripts.verify_canonical_receipt import verify


@pytest.fixture(scope="module")
def base_url() -> str:
    url = os.environ.get("GIGI_URL", "http://localhost:3142")
    # Reachability probe — same shape as Phase A's fixture.
    import requests
    try:
        requests.get(
            f"{url.rstrip('/')}/v1/lattice/__phase_d_probe__",
            timeout=10.0,
        )
    except requests.exceptions.ConnectionError as ex:
        pytest.skip(f"gigi-stream not reachable at {url}: {ex}")
    return url


# ---------------------------------------------------------------------
# G_LIVE_D0 — verifier roundtrips the 5-statement block + lands in band
# ---------------------------------------------------------------------
def test_G_LIVE_D0_verifier_band_pass(base_url: str):
    """The headline public-receipt test. Runs the verifier (which
    fires the chapter's GQL block) against the live engine and
    asserts the tail-mean canonical lands within 3σ of the Halcyon
    spine. Uses unique lattice/field names so it doesn't collide
    with concurrent tests."""
    suffix = f"phase_d_{int(time.time())}"
    result = verify(
        base_url=base_url,
        persist=False,
        api_key=os.environ.get("GIGI_API_KEY"),
        lattice_name=f"halcyon_canonical_buckyball_{suffix}",
        field_name=f"halcyon_canonical_U_{suffix}",
    )
    assert result["band_pass"], (
        f"\n"
        f"  LIVE verifier tail mean ⟨P⟩ = {result['tail_mean']:.6f}\n"
        f"  Halcyon spine              = {result['halcyon_spine_P']:.6f}\n"
        f"  delta                      = {result['delta_from_spine']:.6f}\n"
        f"  tolerance (3σ + margin)    = "
        f"{result['tolerance_3sigma_plus_margin']:.6f}\n"
        f"\n"
        f"  The chapter's GQL block fired against the live engine, but\n"
        f"  the 100-sweep tail mean is outside the 3σ band around the\n"
        f"  Halcyon spine. Either the engine has regressed against its\n"
        f"  III.8a release-profile receipt, or the chapter's spine\n"
        f"  citation has drifted from the deployed verdict."
    )
    # Sanity on the diagnostic fields
    assert result["P_chain_sha256"]
    assert len(result["P_chain_sha256"]) == 64  # SHA-256 hex
    assert result["thermalization_wall_seconds"] > 0
