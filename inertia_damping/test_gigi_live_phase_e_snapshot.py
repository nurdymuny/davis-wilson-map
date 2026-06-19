"""Phase E live tests — SNAPSHOT verb against the real engine.

GIGI's Part V landed 2026-06-19 (V.4-V.6 deploy on machine
``683961dbe9ee38``). The substrate now exposes:

  * ``SNAPSHOT GAUGE_FIELD U PERSIST;`` writes the SU(2) link buffer
    to the engine's WAL under ``OP_GAUGE_FIELD_SNAPSHOT`` (``0x0B``)
    and returns a Rows envelope with ``sha256`` + ``wal_offset`` +
    ``n_edges`` + ``repr_dim``.

  * Bare ``SNAPSHOT GAUGE_FIELD U;`` parse-errors with a message
    naming ``PERSIST`` as required (D-V-D PERSIST-required pushback,
    ratified on the production binary).

This file is the live regression contract for both behaviors.

To run::

    GIGI_URL=https://gigi-stream.fly.dev \\
    GIGI_API_KEY=$KEY \\
    python -m pytest inertia_damping/test_gigi_live_phase_e_snapshot.py -v

Skips cleanly when gigi-stream isn't reachable.
"""
from __future__ import annotations

import os
import re
import time

import pytest

from inertia_damping.scripts.verify_canonical_receipt import verify


@pytest.fixture(scope="module")
def base_url() -> str:
    url = os.environ.get("GIGI_URL", "http://localhost:3142")
    import requests
    try:
        requests.get(
            f"{url.rstrip('/')}/v1/lattice/__phase_e_probe__",
            timeout=10.0,
        )
    except requests.exceptions.ConnectionError as ex:
        pytest.skip(f"gigi-stream not reachable at {url}: {ex}")
    return url


# ---------------------------------------------------------------------
# G_LIVE_E0 — SNAPSHOT returns sha256 + wal_offset of the right shape
# ---------------------------------------------------------------------
def test_G_LIVE_E0_snapshot_writes_wal_and_returns_buffer_sha(base_url: str):
    """SNAPSHOT GAUGE_FIELD U PERSIST; writes to the WAL and returns a
    64-char lowercase hex SHA-256 plus a wal_offset. This is the
    durable citation handle the chapter Appendix A.4 cites."""
    suffix = f"phase_e_{int(time.time())}"
    result = verify(
        base_url=base_url,
        persist=False,
        snapshot=True,
        api_key=os.environ.get("GIGI_API_KEY"),
        lattice_name=f"halcyon_canonical_buckyball_{suffix}",
        field_name=f"halcyon_canonical_U_{suffix}",
    )
    # Band check must still pass — Part V can't break Part III's contract
    assert result["band_pass"], (
        f"thermalization band check failed: tail_mean={result['tail_mean']}, "
        f"delta={result['delta_from_spine']}, tol={result['tolerance_3sigma_plus_margin']}"
    )
    # SNAPSHOT receipt fields
    sha = result.get("snapshot_sha256")
    assert sha is not None, "SNAPSHOT response did not return sha256"
    assert isinstance(sha, str), f"sha256 is not a string: {type(sha)}"
    assert re.fullmatch(r"[0-9a-f]{64}", sha), (
        f"sha256 is not 64-char lowercase hex: {sha!r}"
    )
    assert isinstance(result.get("snapshot_wal_offset"), int), (
        f"wal_offset is not an int: {result.get('snapshot_wal_offset')!r}"
    )
    assert result.get("snapshot_wal_offset") >= 0
    assert result.get("snapshot_n_edges") == 90, (
        f"snapshot n_edges = {result.get('snapshot_n_edges')}; want 90 (buckyball)"
    )
    assert result.get("snapshot_repr_dim") == 4, (
        f"snapshot repr_dim = {result.get('snapshot_repr_dim')}; want 4 (SU(2) quaternion)"
    )


# ---------------------------------------------------------------------
# G_LIVE_E1 — bare SNAPSHOT (no PERSIST) is rejected (D-V-D)
# ---------------------------------------------------------------------
def test_G_LIVE_E1_bare_snapshot_rejected(base_url: str):
    """D-V-D: bare ``SNAPSHOT GAUGE_FIELD U;`` parse-errors with a
    message naming ``PERSIST`` as required. Empirical enforcement of
    Bee's pushback ratified in the Part V reply letter (2026-06-19).
    When ``TRANSIENT`` ships in a future sprint, this gate keeps
    existing callers from silently changing meaning."""
    import requests
    suffix = f"phase_e_bare_{int(time.time())}"
    field_name = f"halcyon_canonical_U_bare_{suffix}"
    headers = {"Accept": "application/json"}
    if os.environ.get("GIGI_API_KEY"):
        headers["X-API-Key"] = os.environ["GIGI_API_KEY"]

    # Declare lattice + field first so the SNAPSHOT has a real target
    # (the parse-error must come from the grammar arm, not from a
    # missing-field error path that would also 400).
    requests.post(
        f"{base_url.rstrip('/')}/v1/gql",
        json={"query": (
            f"LATTICE buckyball_bare_{suffix} FROM TRUNCATED_ICOSAHEDRON "
            f"TOPOLOGY 'S2';"
        )},
        headers=headers, timeout=30.0,
    ).raise_for_status()
    requests.post(
        f"{base_url.rstrip('/')}/v1/gql",
        json={"query": (
            f"GAUGE_FIELD {field_name} ON LATTICE buckyball_bare_{suffix} "
            f"GROUP SU(2) INIT IDENTITY;"
        )},
        headers=headers, timeout=30.0,
    ).raise_for_status()

    # Now fire the bare SNAPSHOT and assert the parse error
    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/gql",
        json={"query": f"SNAPSHOT GAUGE_FIELD {field_name};"},
        headers=headers, timeout=30.0,
    )
    assert resp.status_code == 400, (
        f"bare SNAPSHOT got {resp.status_code}, expected 400. "
        f"D-V-D PERSIST-required clause may not be enforced; bare form "
        f"would silently accept whatever default the engine picks."
    )
    body = resp.json()
    error_text = (body.get("error") or "").lower()
    assert "persist" in error_text, (
        f"400 error message must name PERSIST as required; got: "
        f"{body.get('error')!r}"
    )
