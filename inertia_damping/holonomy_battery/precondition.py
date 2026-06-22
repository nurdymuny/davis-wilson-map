"""Canonical-state precondition for the v3.1.3 LOOP_TRANSPORT verb.

The substrate's executor arm (`gigi/src/parser.rs:10338`) hardcodes
four naming conventions Halcyon's `run_holonomy_battery.py` must
satisfy before LOOP_TRANSPORT will run:

  - LATTICE  name = "halcyon_canonical_buckyball" (v3.1.3 §4.4 canonical)
  - GAUGE_FIELD name = "U_lt" (substrate executor hardcoded)
  - E_FIELD name = "E_lt" (substrate executor hardcoded)
  - LOOP name = "gamma_unit_in_Q_beta_W" (VI.5 gold fixture canonical),
    declared as `FACE 0` on the lattice

This helper declares all four on a running gigi-stream instance and
optionally thermalizes the gauge field to the canonical Migdal-Witten
operating point (β = 2.5, 200 sweeps). It is **idempotent**: if any
of the four already exists from a prior invocation, the helper detects
it via the substrate's "already declared" error and continues without
failing. This matters because a single v3.1.3 battery run produces
≥ 12 LOOP_TRANSPORT calls (5 shams × 2 α calibrations + 2 primary
calls per α), and Halcyon may want a fresh thermalization between
shams without re-declaring the LATTICE / GAUGE_FIELD / E_FIELD / LOOP.

Pattern matches Gigi's `LOOP_TRANSPORT_CALLING_GUIDE.md` §Preconditions
verbatim. When VI.2b ships and the HTTP `/v1/gql` endpoint starts
returning the Rows envelope, this helper is the canonical pre-step
between `cargo run --release --bin gigi-stream` and the orchestrator's
`run_holonomy_battery.py` invocation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# Canonical names — locked at v3.1.3 §4.4 + Gigi's calling guide.
CANONICAL_LATTICE: str = "halcyon_canonical_buckyball"
CANONICAL_U_FIELD: str = "U_lt"
CANONICAL_E_FIELD: str = "E_lt"
CANONICAL_LOOP: str = "gamma_unit_in_Q_beta_W"
CANONICAL_FACE_INDEX: int = 0


# Sentinel strings the substrate emits when a redeclaration would clash.
# Conservative; matches the GIGI lexer's error message conventions seen
# in `inertia_damping/scripts/verify_canonical_receipt.py` + the
# 2026-06-21 smoke session.
_ALREADY_DECLARED_MARKERS = (
    "already declared",
    "already exists",
    "duplicate",
    "redeclaration",
    "redeclare",
)


class PreconditionError(RuntimeError):
    """Raised when a precondition step fails for a reason other than
    'already declared'. Carries the HTTP status + substrate body +
    originating GQL so the failure mode is locatable."""

    def __init__(self, step: str, status_code: int, body: str, query: str):
        super().__init__(
            f"precondition step {step!r} failed: HTTP {status_code} {body}\n"
            f"  query: {query!r}"
        )
        self.step = step
        self.status_code = status_code
        self.body = body
        self.query = query


@dataclass(frozen=True)
class PreconditionResult:
    """What the helper produced. Returned for the sidecar audit
    trail + for downstream tests that want to verify the substrate
    reached the canonical state."""

    base_url: str
    lattice_name: str
    u_field_name: str
    e_field_name: str
    loop_name: str
    face_index: int

    # Was each step a fresh declaration or a no-op (already existed)?
    lattice_was_fresh: bool
    u_field_was_fresh: bool
    e_field_was_fresh: bool
    loop_was_fresh: bool

    # Thermalization diagnostics (only populated when init="thermalized")
    init_mode: str  # "identity" or "thermalized"
    seed: Optional[int] = None
    beta: Optional[float] = None
    n_sweeps_requested: int = 0
    n_sweeps_completed: int = 0
    mean_plaquette_chain: List[float] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """One-line summary of what the precondition produced."""
        steps = []
        if self.lattice_was_fresh:
            steps.append(f"LATTICE {self.lattice_name}=fresh")
        else:
            steps.append(f"LATTICE {self.lattice_name}=existing")
        if self.u_field_was_fresh:
            steps.append(f"GAUGE {self.u_field_name}=fresh")
        else:
            steps.append(f"GAUGE {self.u_field_name}=existing")
        if self.e_field_was_fresh:
            steps.append(f"E {self.e_field_name}=fresh")
        else:
            steps.append(f"E {self.e_field_name}=existing")
        if self.loop_was_fresh:
            steps.append(f"LOOP {self.loop_name}=fresh")
        else:
            steps.append(f"LOOP {self.loop_name}=existing")
        if self.init_mode == "thermalized":
            tail_mean = (
                self.mean_plaquette_chain[-1] if self.mean_plaquette_chain else float("nan")
            )
            steps.append(
                f"thermalized at β={self.beta}, "
                f"{self.n_sweeps_completed} sweeps, "
                f"<P>_last={tail_mean:.4f}"
            )
        else:
            steps.append(f"init=identity")
        return "; ".join(steps)


def precondition_canonical_buckyball(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    lattice_name: str = CANONICAL_LATTICE,
    u_field: str = CANONICAL_U_FIELD,
    e_field: str = CANONICAL_E_FIELD,
    loop_name: str = CANONICAL_LOOP,
    face_index: int = CANONICAL_FACE_INDEX,
    init: Literal["identity", "thermalized"] = "thermalized",
    seed: int = 20260616,
    beta: float = 2.5,
    n_sweeps: int = 200,
    measure_every: int = 50,
    timeout_s: float = 600.0,
) -> PreconditionResult:
    """Bring a running gigi-stream instance to v3.1.3 canonical state.

    Posts four GQL declarations + optionally one thermalization
    against the engine at `base_url` (default: GIGI_URL env var, or
    http://localhost:3142). Idempotent: re-declarations are detected
    and treated as no-ops; the thermalization always re-runs (since
    GIBBS_SAMPLE doesn't persist to the WAL per
    `verify_canonical_receipt.py` line 14-18).

    Returns a `PreconditionResult` carrying the per-step outcomes +
    GIBBS_SAMPLE diagnostics if applicable.
    """
    try:
        import requests  # noqa: F401 — checked at call time
    except ImportError as ex:
        raise RuntimeError(
            "precondition_canonical_buckyball requires `pip install requests`. "
            "Install requests only when running live preconditions; the "
            "mock-client path has no such dependency."
        ) from ex

    if init not in ("identity", "thermalized"):
        raise ValueError(
            f"init must be 'identity' or 'thermalized'; got {init!r}"
        )

    base_url = (
        base_url or os.environ.get("GIGI_URL") or "http://localhost:3142"
    ).rstrip("/")
    api_key = api_key or os.environ.get("GIGI_API_KEY")

    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    # Step 1: LATTICE
    lattice_query = (
        f"LATTICE {lattice_name} FROM TRUNCATED_ICOSAHEDRON TOPOLOGY 'S2';"
    )
    lattice_was_fresh = _try_post_with_idempotency(
        base_url, headers, lattice_query, timeout_s, step="LATTICE",
    )

    # Step 2: GAUGE_FIELD (always INIT IDENTITY; thermalization is step 5)
    gauge_field_query = (
        f"GAUGE_FIELD {u_field} ON LATTICE {lattice_name} "
        f"GROUP SU(2) INIT IDENTITY;"
    )
    u_field_was_fresh = _try_post_with_idempotency(
        base_url, headers, gauge_field_query, timeout_s, step="GAUGE_FIELD",
    )

    # Step 3: E_FIELD INIT ZERO (LOOP_TRANSPORT inherits momentum from
    # this field's initial state; INIT ZERO is the v3.1.3 §4.4 canonical
    # starting momentum state)
    e_field_query = (
        f"E_FIELD {e_field} ON GAUGE_FIELD {u_field} INIT ZERO;"
    )
    e_field_was_fresh = _try_post_with_idempotency(
        base_url, headers, e_field_query, timeout_s, step="E_FIELD",
    )

    # Step 4: LOOP declaration — FACE 0 (the first pentagon of the
    # buckyball, matching Gigi's smoke test and VI.5 gold fixture)
    loop_query = (
        f"LOOP {loop_name} ON {lattice_name} FACE {face_index};"
    )
    loop_was_fresh = _try_post_with_idempotency(
        base_url, headers, loop_query, timeout_s, step="LOOP",
    )

    # Step 5 (optional): GIBBS_SAMPLE thermalization
    chain: List[float] = []
    n_sweeps_completed = 0
    if init == "thermalized":
        thermalize_query = (
            f"GIBBS_SAMPLE {u_field} BETA {beta} "
            f"N_SWEEPS {n_sweeps} "
            f"MEASURE_EVERY {measure_every} "
            f"MEASURE (MEAN(PLAQUETTE)) "
            f"SEED {seed};"
        )
        body = _post_or_raise(
            base_url, headers, thermalize_query, timeout_s, step="THERMALIZE",
        )
        row = _first_row(body)
        if row is not None:
            chain_value = (
                row.get("MeanPlaquette")
                or row.get("mean_plaquette")
                or row.get("MEAN(PLAQUETTE)")
            )
            if chain_value is not None:
                chain = [float(x) for x in chain_value]
            n_sweeps_completed = int(row.get("n_sweeps_completed", n_sweeps))
        else:
            n_sweeps_completed = n_sweeps  # best-effort

    return PreconditionResult(
        base_url=base_url,
        lattice_name=lattice_name,
        u_field_name=u_field,
        e_field_name=e_field,
        loop_name=loop_name,
        face_index=face_index,
        lattice_was_fresh=lattice_was_fresh,
        u_field_was_fresh=u_field_was_fresh,
        e_field_was_fresh=e_field_was_fresh,
        loop_was_fresh=loop_was_fresh,
        init_mode=init,
        seed=seed if init == "thermalized" else None,
        beta=beta if init == "thermalized" else None,
        n_sweeps_requested=n_sweeps if init == "thermalized" else 0,
        n_sweeps_completed=n_sweeps_completed,
        mean_plaquette_chain=chain,
    )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _try_post_with_idempotency(
    base_url: str,
    headers: Dict[str, str],
    query: str,
    timeout_s: float,
    step: str,
) -> bool:
    """Post a declaration GQL. Returns True if it was a fresh
    declaration; False if the substrate said it was already declared
    (idempotent no-op). Raises PreconditionError on any other error."""
    import requests
    resp = requests.post(
        f"{base_url}/v1/gql",
        json={"query": query},
        headers=headers,
        timeout=timeout_s,
    )
    if resp.ok:
        return True
    body_text = resp.text or ""
    lower = body_text.lower()
    if any(marker in lower for marker in _ALREADY_DECLARED_MARKERS):
        return False
    raise PreconditionError(
        step=step,
        status_code=resp.status_code,
        body=body_text,
        query=query,
    )


def _post_or_raise(
    base_url: str,
    headers: Dict[str, str],
    query: str,
    timeout_s: float,
    step: str,
) -> Any:
    """Post a GQL statement that must succeed. No idempotency shortcut."""
    import requests
    resp = requests.post(
        f"{base_url}/v1/gql",
        json={"query": query},
        headers=headers,
        timeout=timeout_s,
    )
    if not resp.ok:
        raise PreconditionError(
            step=step,
            status_code=resp.status_code,
            body=resp.text or "",
            query=query,
        )
    return resp.json()


def _first_row(body: Any) -> Optional[Dict[str, Any]]:
    """Pull the first row out of a Rows envelope; returns None on a
    bare-status response."""
    if isinstance(body, dict):
        inner = body.get("rows", body)
        if isinstance(inner, dict):
            inner = inner.get("rows", [])
        if isinstance(inner, list) and inner:
            return inner[0]
    if isinstance(body, list) and body:
        return body[0]
    return None
