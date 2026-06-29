"""Halcyon-side CSPRNG sentinel test for the gigi substrate.

SKIP-BY-DEFAULT. Tests run only when:

    HALCYON_LIVE_SMOKE=1
    GIGI_URL=http://...    (or the default localhost:3142 if local)

This file is the Halcyon-side counterpart to gigi's substrate-side
gold fixtures (IV.6 + VI.5). Together they pin the cross-team
CSPRNG-determinism contract from both sides of the wire, completing
the audit chain.

WHY THIS IS LOAD-BEARING
------------------------
Halcyon's `LiveLoopTransportClient` (per-seed thermalization landed
at commit 5add5da) decomposes every multi-seed LOOP_TRANSPORT request
into N single-seed sub-calls, with a fresh GIBBS_SAMPLE
re-thermalization between each one. That decomposition is correct
ONLY because gigi's GIBBS_SAMPLE is deterministic per seed: the same
SEED value MUST always produce the same gauge configuration, byte
for byte, across processes and across rebuilds -- *given the same
starting U_lt configuration*.

The determinism guarantee is implemented by gigi's xorshift64* CSPRNG
at `gigi/src/gauge/marsaglia_haar.rs:33`. If gigi ever swaps that
implementation (different PRNG, different seeding scheme, different
state-mixing) the contract silently breaks:

  * Halcyon's FORWARD sub-call and REVERSED sub-call for the same
    seed would land on DIFFERENT configurations.
  * The antisymmetric primary observable
    H_anti = (H_forward - H_reversed) / 2 would become meaningless.
  * The v3.1.3 pre-registered protocol (Zenodo DOI
    10.5281/zenodo.20785681, commit 44c70b1) would produce
    publication-bound numbers that look fine and are silently wrong.

This sentinel is part of the HALCYON_LIVE_SMOKE=1 battery; the
operator is expected to fire it after every gigi-stream rebuild as
part of the normal smoke-test workflow. There is no auto-hook into
`cargo build`; rebuild -> smoke is an operator discipline, not a
build-system invariant. (Wiring a Makefile or git hook is out of
scope for this artifact; if added later, document in JOURNAL.md.)

FINGERPRINT PROVENANCE
----------------------
Both captured MeanPlaquette chains below were measured against gigi
commit 3f7d42e (post-VI.6b: tau_pin measurement, tracking_error
measurement, alpha=1000 BETA_WILSON amplitude formula using tau_0 as
the implicit time-scale per v3.1.3 section 3.6's dual-alpha
calibration). The values were verified byte-identical across repeat
firings before being pinned here.

CAPTURE STATE (load-bearing -- re-pinning requires reproducing this):

  * gigi commit 3f7d42e
  * LATTICE halcyon_canonical_buckyball FROM TRUNCATED_ICOSAHEDRON
    TOPOLOGY 'S2'
  * GAUGE_FIELD (any name) ON LATTICE halcyon_canonical_buckyball
    GROUP SU(2) INIT IDENTITY    -- freshly declared, no prior
    GIBBS_SAMPLE applied; the CSPRNG output for a given SEED depends
    on the starting U_lt configuration, and INIT IDENTITY is the
    canonical starting state. The fingerprint is FIELD-NAME-INDEPENDENT
    under these conditions (same lattice topology, same group, same
    INIT mode -> identical Markov chain regardless of the field's
    name), which is why the sentinel uses UUID-suffixed scratch field
    names rather than the canonical `U_lt`.
  * GQL emitted verbatim:
        GIBBS_SAMPLE <FIELD> BETA 2.5 N_SWEEPS 200 MEASURE_EVERY 50
        MEASURE (MEAN(PLAQUETTE)) SEED <SEED>;

If this test fails after a gigi rebuild, DO NOT update the
fingerprints without first auditing:

  1. Did gigi's CSPRNG implementation change? (Check
     `gigi/src/gauge/marsaglia_haar.rs` and anything touching the
     xorshift64* state initialization or stream.)
  2. Did the GIBBS_SAMPLE update rule, plaquette measurement, or
     measurement-cadence schedule change?
  3. If yes to either: this is a cross-team breaking change. Sync with
     gigi-side gold fixtures (IV.6, VI.5) before re-pinning anything.
  4. If no to both: the failure is a real bug in the new gigi build
     and must be diagnosed, not papered over.

Record any re-pinning event in `inertia_damping/JOURNAL.md` with the
new gigi commit hash and the reason the previous fingerprint became
stale.
"""
from __future__ import annotations

import os
import uuid

import pytest


SMOKE_ENABLED = os.environ.get("HALCYON_LIVE_SMOKE", "").lower() in ("1", "true", "yes")
SKIP_REASON = (
    "Live smoke disabled. Set HALCYON_LIVE_SMOKE=1 to enable; "
    "ensure GIGI_URL points at the live engine (or leave unset for "
    "localhost:3142 with a running `cargo run --release --bin gigi-stream`)."
)

pytestmark = pytest.mark.skipif(not SMOKE_ENABLED, reason=SKIP_REASON)


# Default substrate endpoint - overridable via GIGI_URL env var. Matches
# the convention used by `LiveLoopTransportClient` so the sentinel
# always hits the same engine the rest of the live battery hits.
DEFAULT_GIGI_URL = "http://localhost:3142"
GQL_ENDPOINT_SUFFIX = "/v1/gql"

# Canonical lattice - must match precondition.py's CANONICAL_LATTICE.
# We declare it (idempotently) per test so the sentinel doesn't
# silently depend on the rest of the battery having run first.
CANONICAL_LATTICE = "halcyon_canonical_buckyball"

# GIBBS_SAMPLE template parameterized by the scratch field name and
# seed. Held as a template (not a constant) so each test invocation
# can substitute its UUID-suffixed scratch field and the
# canonical/alt seed without ambiguity about what wire string fired.
GIBBS_SAMPLE_TEMPLATE = (
    "GIBBS_SAMPLE {field} BETA 2.5 N_SWEEPS 200 MEASURE_EVERY 50 "
    "MEASURE (MEAN(PLAQUETTE)) SEED {seed};"
)

# Captured fingerprints - verified byte-identical across repeat firings
# at gigi commit 3f7d42e (post-VI.6b), measured against a freshly
# declared SU(2) GAUGE_FIELD at INIT IDENTITY on the canonical
# halcyon_canonical_buckyball lattice. See module docstring CAPTURE
# STATE block for the full pinning protocol.
CANONICAL_FINGERPRINT_SEED_20260616 = [
    0.4748770176956066,
    0.45682045092345325,
    0.5671546613446052,
    0.5125429110231062,
]
CANONICAL_FINGERPRINT_SEED_20260617 = [
    0.4891038593457839,
    0.5344809007981545,
    0.6216678882452541,
    0.4399333197466609,
]

CANONICAL_SEED = 20260616
ALT_SEED = 20260617


def _gigi_url() -> str:
    return os.environ.get("GIGI_URL", DEFAULT_GIGI_URL).rstrip("/")


def _post_gql(gql: str) -> dict:
    """POST a GQL statement to the live substrate and return the
    parsed JSON envelope. Kept inline (no shared client) so this
    sentinel is independent of `LiveLoopTransportClient`'s evolution:
    a regression in that client should never mask a CSPRNG drift, and
    a CSPRNG drift should never look like a client-side parsing bug.

    Wire format: `{"query": gql}` per gigi-stream's
    `body.get("query")` and Halcyon-side convention
    (precondition.py, live_loop_transport.py)."""
    import requests  # lazy: keep collection cheap when smoke is off

    url = _gigi_url() + GQL_ENDPOINT_SUFFIX
    response = requests.post(url, json={"query": gql}, timeout=120)
    response.raise_for_status()
    return response.json()


@pytest.fixture(scope="module", autouse=True)
def _engine_reachable():
    """Module-level pre-flight ping. If `gigi-stream` is not running,
    convert the ConnectionError into a clean pytest.skip with
    actionable launch instructions rather than letting every test in
    the module surface a network traceback as a test ERROR."""
    import requests

    url = _gigi_url() + GQL_ENDPOINT_SUFFIX
    try:
        # A trivially-empty query; the substrate will reject it but
        # the TCP/HTTP layer will respond, which is all we need to
        # know the engine is up. A 4xx is fine - a ConnectionError is
        # what we're guarding against.
        requests.post(url, json={"query": ";"}, timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip(
            f"gigi-stream not reachable at {_gigi_url()}; start with "
            f"`cargo run --release --bin gigi-stream` and retry."
        )
    except requests.exceptions.Timeout:
        pytest.skip(
            f"gigi-stream at {_gigi_url()} did not respond within 5s; "
            f"is the engine deadlocked or under heavy load?"
        )


def _declare_canonical_lattice_idempotent() -> None:
    """Declare the canonical lattice if not already present. The
    substrate treats redeclaration as an error; we tolerate any 4xx
    response with an 'already declared' marker, matching
    `_try_post_with_idempotency` in precondition.py."""
    import requests

    gql = (
        f"LATTICE {CANONICAL_LATTICE} FROM TRUNCATED_ICOSAHEDRON "
        f"TOPOLOGY 'S2';"
    )
    url = _gigi_url() + GQL_ENDPOINT_SUFFIX
    resp = requests.post(url, json={"query": gql}, timeout=30)
    if resp.ok:
        return
    body_lower = (resp.text or "").lower()
    if any(
        marker in body_lower
        for marker in ("already declared", "already exists", "duplicate", "redeclaration")
    ):
        return
    resp.raise_for_status()


def _declare_fresh_identity_scratch_field() -> str:
    """Declare a UUID-suffixed scratch GAUGE_FIELD at INIT IDENTITY
    and return its name. Each test invocation gets a unique field
    name, eliminating cross-test state contamination -- gigi has no
    RESET / DROP GAUGE_FIELD verb, so reusing `U_lt` across tests
    would mean the second test runs against a thermalized
    configuration left over from the first.

    The fingerprint is field-name-independent (see module docstring
    CAPTURE STATE), so the captured values apply regardless of which
    UUID this returns."""
    _declare_canonical_lattice_idempotent()
    name = f"U_sentinel_{uuid.uuid4().hex[:12]}"
    gql = (
        f"GAUGE_FIELD {name} ON LATTICE {CANONICAL_LATTICE} "
        f"GROUP SU(2) INIT IDENTITY;"
    )
    _post_gql(gql)
    return name


def _extract_mean_plaquette_chain(envelope: dict) -> list[float]:
    """Pull the MeanPlaquette vector out of the Rows response. The
    substrate returns ONE row whose `MeanPlaquette` field is a
    `Value::Vector(chain)` carrying all 4 measurements (after sweeps
    50, 100, 150, 200). This mirrors precondition.py's chain
    extraction."""
    rows = envelope.get("rows") or envelope.get("Rows") or []
    if not rows:
        raise AssertionError(
            f"GIBBS_SAMPLE envelope had no rows: {envelope!r}. "
            "Substrate response shape may have changed; update "
            "_extract_mean_plaquette_chain rather than papering over."
        )
    row = rows[0]
    chain_value = (
        row.get("MeanPlaquette")
        or row.get("mean_plaquette")
        or row.get("MEAN(PLAQUETTE)")
    )
    if chain_value is None or not isinstance(chain_value, (list, tuple)):
        raise AssertionError(
            f"GIBBS_SAMPLE row did not contain a MeanPlaquette vector: "
            f"{row!r}. Substrate envelope shape may have changed; update "
            "_extract_mean_plaquette_chain rather than papering over."
        )
    return [float(x) for x in chain_value]


def _gibbs_sample_chain(field_name: str, seed: int) -> list[float]:
    """Fire GIBBS_SAMPLE against the given scratch field + seed and
    return the extracted 4-element MeanPlaquette chain."""
    gql = GIBBS_SAMPLE_TEMPLATE.format(field=field_name, seed=seed)
    envelope = _post_gql(gql)
    return _extract_mean_plaquette_chain(envelope)


def _assert_chain_matches_fingerprint(
    chain: list,
    expected: list,
    *,
    seed: int,
) -> None:
    """Bit-exact equality of `chain` against `expected`. Python's
    `==` on float is bit-exact for normalized f64, so this catches
    legitimate single-bit drift that a tolerance-based check would
    hide. JSON round-trip of f64 is shortest-round-trip by serde
    convention; the captured values were observed verbatim from the
    wire."""
    assert len(chain) == len(expected), (
        f"Expected {len(expected)} MeanPlaquette measurements "
        f"(N_SWEEPS=200, MEASURE_EVERY=50) for SEED {seed}, "
        f"got {len(chain)}: {chain!r}. Substrate measurement "
        "cadence may have changed."
    )
    for i, (actual, want) in enumerate(zip(chain, expected)):
        assert actual == want, (
            f"CSPRNG fingerprint MISMATCH for SEED {seed} at element {i}: "
            f"actual={actual!r}, expected={want!r}, "
            f"|diff|={abs(actual - want):.3e}. "
            "Gigi's CSPRNG output has changed. Audit "
            "gigi/src/gauge/marsaglia_haar.rs and the GIBBS_SAMPLE "
            "update rule before touching this test. See module "
            "docstring for the re-pinning protocol."
        )


def test_csprng_fingerprint_pins_both_seeds():
    """Pin gigi's xorshift64* output for BOTH the canonical and
    alt-seed channels to bit-exact equality with the captured
    4-element MeanPlaquette fingerprints.

    Pinning both seeds catches a class of CSPRNG drift that a
    single-channel pin would miss -- e.g., a re-implementation that
    happens to land the canonical seed on the same trajectory but
    diverges on others.

    Each seed is fired against an independently declared scratch
    field at INIT IDENTITY (no cross-test state). If this fires,
    gigi's CSPRNG behavior has changed and Halcyon's per-seed
    LOOP_TRANSPORT decomposition can no longer be trusted -- DO NOT
    update the fingerprints without auditing per the module
    docstring."""
    field_canonical = _declare_fresh_identity_scratch_field()
    chain_canonical = _gibbs_sample_chain(field_canonical, CANONICAL_SEED)
    _assert_chain_matches_fingerprint(
        chain_canonical,
        CANONICAL_FINGERPRINT_SEED_20260616,
        seed=CANONICAL_SEED,
    )

    field_alt = _declare_fresh_identity_scratch_field()
    chain_alt = _gibbs_sample_chain(field_alt, ALT_SEED)
    _assert_chain_matches_fingerprint(
        chain_alt,
        CANONICAL_FINGERPRINT_SEED_20260617,
        seed=ALT_SEED,
    )


def test_csprng_fingerprint_varies_across_seeds():
    """Sanity check: a different seed produces a materially different
    chain. Without this assertion, the bit-exact pin above could
    trivially pass against a degenerate CSPRNG that collapses all
    seeds to a single trajectory -- which would be catastrophic for
    Halcyon's per-seed LOOP_TRANSPORT decomposition.

    This test is intentionally weaker than test 1: it only requires
    that the two captured fingerprints DIFFER, which they do by
    construction (the captured values diverge by ~0.05 on multiple
    elements). The bit-exact pin in test 1 carries the heavy lifting;
    this test exists to make the seed-collapse failure mode
    explicit."""
    diffs = [
        abs(a - b)
        for a, b in zip(
            CANONICAL_FINGERPRINT_SEED_20260616,
            CANONICAL_FINGERPRINT_SEED_20260617,
        )
    ]
    max_abs_diff = max(diffs)
    # Threshold is generous (1e-6); the captured fingerprints
    # actually differ by ~0.05 on multiple elements. A failure here
    # means the captured fingerprints themselves have been changed
    # to collapse onto each other, which would only happen via
    # operator error during a re-pinning event.
    assert max_abs_diff > 1e-6, (
        f"CAPTURED FINGERPRINTS appear to be collapsed: SEED "
        f"{ALT_SEED} fingerprint {CANONICAL_FINGERPRINT_SEED_20260617!r} "
        f"agrees with SEED {CANONICAL_SEED} fingerprint "
        f"{CANONICAL_FINGERPRINT_SEED_20260616!r} on every element to "
        f"within 1e-6 (max |diff| = {max_abs_diff:.3e}). This is a "
        "test-design failure (someone re-pinned both seeds to the same "
        "values), not a live-substrate failure. Audit the recent edits "
        "to this file and the JOURNAL.md entries around the last "
        "re-pinning event."
    )
