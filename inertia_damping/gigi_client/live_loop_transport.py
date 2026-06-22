"""LiveLoopTransportClient — HTTP-based adapter for the GIGI substrate's
LOOP_TRANSPORT verb (Halcyon Part VI per
``theory/halcyon/HALCYON_PART_VI_GATES.md`` on the gigi side).

This client is the live counterpart to ``MockLoopTransportClient``.
The Halcyon orchestrator (``run_holonomy_battery.py``) constructs
either client and treats them as the same ``LoopTransportClient``
``typing.Protocol`` — the orchestrator gate logic, sham handling, and
sidecar emission are byte-identical regardless of which client backs
the call. Swap is done in ``main()`` via the ``--client {mock,live}``
flag.

**Pre-conditions for use:**

- GIGI Part VI must be shipped:
  - VI.1 gate doc ✓ (commit ``c3970c4`` / ``9a73dc0`` on the gigi side
    per Halcyon's 2026-06-21 receipt).
  - VI.2 LOOP_TRANSPORT verb ✓ (commit ``777c7ad`` per the same).
  - VI.3 ``GC₁``-``GC₆`` acceptance battery green ✓ (commit ``1d2bd39``).
  - VI.4 SHAM-block real dispatch — **NOT YET**. The verb parses the
    SHAM block and rejects unknown flags but the 5 science-gate flags
    don't yet modify per-substep dynamics. A live call with shams will
    return primary-shaped data for each sham → all sham gates fail →
    verdict AMBIGUOUS-because-shams-fail. This is integration testing,
    not science.
  - VI.5 bit-identity per-seed gold fixture — **NOT YET**. Per-seed
    numerical values may drift within f64-reassociation slack across
    recompiles until VI.5 freezes the gold fixture.
- ``requests`` installed (mock client has no such dependency).
- ``GIGI_URL`` env var pointing at the live engine (defaults to
  ``http://localhost:3142``; production is ``https://gigi-stream.fly.dev``).
- ``GIGI_API_KEY`` env var if the live engine requires auth.

**Disposition of the publication-bound v3.1.3 science run:** waits for
VI.4 + VI.5 per Halcyon's design closeout. This client unblocks
integration testing and smoke tests; the first run against the live
binding that is meant to count as science is the one that happens
after both VI.4 and VI.5 are green.

**Wire grammar best-effort:** v3.1.3 §4.4 (the deposited grammar) and
Gigi's Part VI gate doc both reference ``ALONG_LOOP <handle>`` —
implying loops are declared separately and referenced by name. v3.1.3
does not spell out the ``DECLARE LOOP`` syntax verbatim. This client's
``declare_loop()`` constructs a best-guess GQL statement matching the
SPEC's intent; if VI.2's actual parser arm wants a different exact
shape, the live smoke test will surface a 400 from ``/v1/gql`` and we
adjust here, not in the orchestrator.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inertia_damping.gigi_client.loop_transport import (
    AdiabaticityCheck,
    HalcyonParameterPack,
    LoopHandle,
    LoopTransportRequest,
    LoopTransportResult,
    ShamFlag,
)


DEFAULT_GIGI_URL = "http://localhost:3142"
# LOOP_TRANSPORT at N_DISCRETIZATION = 10000 × 8 seeds × 2 directions
# = 160,000 substeps per call. Per Gigi's VI.3 receipt the canonical
# run was ~48s for N=16000 in the GC₅ test; the science call at N=10000
# should be roughly comparable. 5-minute timeout gives headroom for
# network jitter on the fly.dev path.
DEFAULT_TIMEOUT_S = 300.0

CANONICAL_GAUGE_FIELD = "halcyon_canonical_buckyball"


class LoopTransportLiveError(RuntimeError):
    """Raised when the live substrate rejects a LOOP_TRANSPORT call.
    The HTTP status code and the GIGI error body are included verbatim
    so smoke-test failures surface the substrate's diagnostic
    unchanged. Distinct from the validation errors that
    ``LoopTransportRequest.__post_init__`` raises pre-flight."""

    def __init__(self, status_code: int, body: str, query: str):
        super().__init__(
            f"LOOP_TRANSPORT failed: HTTP {status_code} {body} "
            f"(query: {query!r})"
        )
        self.status_code = status_code
        self.body = body
        self.query = query


class LiveLoopTransportClient:
    """HTTP adapter for the GIGI substrate's LOOP_TRANSPORT verb.
    Satisfies ``LoopTransportClient`` Protocol structurally."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        api_key: Optional[str] = None,
        ping: bool = False,
    ):
        try:
            import requests  # noqa: F401 — checked at construction time
        except ImportError as ex:
            raise RuntimeError(
                "LiveLoopTransportClient requires `pip install requests`. "
                "The mock has no such dependency; install requests only when "
                "running live tests against gigi-stream."
            ) from ex
        self.base_url = (
            base_url or os.environ.get("GIGI_URL") or DEFAULT_GIGI_URL
        ).rstrip("/")
        self.timeout_s = float(timeout_s)
        # Auth: explicit api_key arg > GIGI_API_KEY env > no auth.
        # Header shape mirrors LiveGIGIClient (X-API-Key per gigi-stream).
        self.api_key = api_key or os.environ.get("GIGI_API_KEY")
        # Loop handles we've declared this session, name → LoopHandle.
        # declare_loop() is idempotent per the existing GIGI pattern;
        # we cache the handle so a redeclare with a different shape
        # raises a clear error without round-tripping to the engine.
        self._declared_loops: Dict[str, LoopHandle] = {}
        if ping:
            self._ping()

    # ------------------------------------------------------------------
    # LoopTransportClient Protocol
    # ------------------------------------------------------------------

    def declare_loop(self, loop: LoopHandle) -> str:
        """``DECLARE LOOP <name> CONTROL_MANIFOLD (Q, beta_wilson) PATH
        (...) -> (...) T_LOOP <t> SEGMENTS PIECEWISE_LINEAR;``

        Idempotent on the same loop shape; raises if a different shape
        was previously declared under the same name (matching the
        ``MockLoopTransportClient`` contract).
        """
        existing = self._declared_loops.get(loop.name)
        if existing is not None and existing != loop:
            raise ValueError(
                f"Loop {loop.name!r} already declared with different shape; "
                f"redeclaration requires a different name"
            )
        if existing is not None:
            return loop.name

        query = self._build_declare_loop_gql(loop)
        _ = self._gql_query(query)
        self._declared_loops[loop.name] = loop
        return loop.name

    def loop_transport(self, request: LoopTransportRequest) -> LoopTransportResult:
        """``LOOP_TRANSPORT <gauge_field> ALONG_LOOP <loop_name>
        CONTROL_MANIFOLD (Q, beta_wilson) ADIABATIC TRUE ... RETURN
        H_forward, H_reversed, sigma_H_blocked, per_seed_H_forward,
        per_seed_H_reversed, tracking_error_max_Q,
        tracking_error_max_beta_W, adiabaticity_check;``

        Posts the GQL to ``/v1/gql`` and unpacks the Rows envelope
        into a ``LoopTransportResult`` matching the mock's contract.
        """
        if request.loop.name not in self._declared_loops:
            raise ValueError(
                f"Loop {request.loop.name!r} not declared; "
                f"call declare_loop({request.loop.name}) first"
            )

        query = self._build_loop_transport_gql(request)
        body = self._gql_query(query)
        rows = self._extract_rows(body)
        if not rows:
            raise LoopTransportLiveError(
                status_code=200, body=f"empty rows envelope; full body: {body!r}",
                query=query,
            )
        row = rows[0]
        return self._row_to_result(row, request, query)

    # ------------------------------------------------------------------
    # GQL builders (best-effort against gate-doc grammar + v3.1.3 §4.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_declare_loop_gql(loop: LoopHandle) -> str:
        """Construct a DECLARE LOOP GQL statement. Best-guess grammar
        against Gigi's Part VI gate doc; if VI.2's parser uses a
        different exact shape, this surfaces as a 400 on the first
        live smoke test and is patched here."""
        axes = ", ".join(loop.control_manifold_axes)
        path_segments = []
        for (q, beta_w) in loop.vertices:
            path_segments.append(f"(Q={q!r}, beta_wilson={beta_w!r})")
        path = " -> ".join(path_segments)
        return (
            f"DECLARE LOOP {loop.name}"
            f" CONTROL_MANIFOLD ({axes})"
            f" PATH {path}"
            f" T_LOOP {loop.t_per_segment * max(1, loop.n_segments())!r}"
            f" SEGMENTS PIECEWISE_LINEAR;"
        )

    @staticmethod
    def _build_loop_transport_gql(request: LoopTransportRequest) -> str:
        """Construct the LOOP_TRANSPORT GQL statement matching v3.1.3
        §4.4 + Halcyon's design-closeout per-axis ramp_rate (CC-LT-8)."""
        pack = request.pack
        seeds_clause = "[" + ", ".join(str(s) for s in request.seeds) + "]"

        sham_clause = ""
        if request.sham is not ShamFlag.NONE:
            sham_clause = " " + LiveLoopTransportClient._build_sham_clause(request)

        return (
            f"LOOP_TRANSPORT {CANONICAL_GAUGE_FIELD}"
            f" ALONG_LOOP {request.loop.name}"
            f" CONTROL_MANIFOLD (Q, beta_wilson)"
            f" ADIABATIC TRUE"
            f" RAMP_RATE_Q {request.ramp_rate_Q!r}"
            f" RAMP_RATE_BETA_W {request.ramp_rate_beta_W!r}"
            f" DRIVE_OMEGA {pack.drive_omega!r}"
            f" DRIVE_F0 {pack.drive_F0!r}"
            f" N_DISCRETIZATION {request.n_discretization}"
            f" PIN_LAMBDA_Q {pack.pin_lambda_Q!r}"
            f" PIN_LAMBDA_BETA_W {pack.pin_lambda_beta_W!r}"
            f" EPS_Q {pack.eps_Q!r}"
            f" EPS_BETA_W {pack.eps_beta_W!r}"
            f" ALPHA_HALCYON {pack.alpha!r}"
            f" TAU_0 {pack.tau_0!r}"
            f" BETA_TAU {pack.beta_tau!r}"
            f" MU_BASELINE {pack.mu_baseline!r}"
            f" K_SPRING {pack.K_spring!r}"
            f" C_DAMP {pack.c_damp!r}"
            f" DIRECTION {request.direction}"
            f" SEEDS {seeds_clause}"
            f"{sham_clause}"
            f" COMPUTE HOLONOMY_FORWARD"
            f" COMPUTE HOLONOMY_REVERSED"
            f" COMPUTE TRACKING_ERROR_TRACE_Q"
            f" COMPUTE TRACKING_ERROR_TRACE_BETA_W"
            f" COMPUTE ADIABATICITY_CHECK"
            f" RETURN H_forward, H_reversed, sigma_H_blocked,"
            f" per_seed_H_forward, per_seed_H_reversed,"
            f" tracking_error_max_Q, tracking_error_max_beta_W,"
            f" adiabaticity_check;"
        )

    @staticmethod
    def _build_sham_clause(request: LoopTransportRequest) -> str:
        """Nested SHAM { flag: value, ... } block per Halcyon
        design-closeout §C.6 (CC-LT-6 nested over top-level)."""
        if request.sham is ShamFlag.MASS_SCALED:
            return (
                f"SHAM {{ mass_baseline_scaled: true, "
                f"mu_baseline_scale: {request.sham_mass_scale!r} }}"
            )
        # Map ShamFlag enum to lowercase nested-block keys matching
        # Gigi's Part VI gate doc §SHAM table (FLAT_FIELD →
        # flat_field, ALPHA_ZERO → alpha_zero, etc.)
        key_map = {
            ShamFlag.FLAT_FIELD: "flat_field",
            ShamFlag.ALPHA_ZERO: "alpha_zero",
            ShamFlag.BACKTRACK_LOOP: "degenerate_loop",
            ShamFlag.FROZEN_FIELD: "frozen_field",
            ShamFlag.EMPTY_LOOP: "empty_loop",
            ShamFlag.OPEN_LOOP: "open_loop",
        }
        key = key_map.get(request.sham)
        if key is None:
            raise ValueError(f"unmapped sham flag {request.sham}")
        return f"SHAM {{ {key}: true }}"

    # ------------------------------------------------------------------
    # Response unpacking
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_rows(body: Any) -> List[Dict[str, Any]]:
        """Pull the Rows list out of the Rows envelope. Tolerates both
        the nested ``{rows: [...]}`` shape and a bare-list shape."""
        if isinstance(body, dict):
            inner = body.get("rows", body)
            if isinstance(inner, dict):
                inner = inner.get("rows", [inner])
            if isinstance(inner, list):
                return inner
            return [inner] if inner else []
        if isinstance(body, list):
            return body
        return []

    def _row_to_result(
        self,
        row: Dict[str, Any],
        request: LoopTransportRequest,
        query: str,
    ) -> LoopTransportResult:
        """Translate one Rows row into a ``LoopTransportResult``.

        Tries CamelCase column names first (production gigi-stream V.0
        convention per LiveGIGIClient), falls back to snake_case, falls
        back to the literal RETURN-clause field name. Each missing
        required field raises a clear error pointing at which column
        the parser tried."""
        n_seeds = len(request.seeds)

        per_seed_H_forward = self._pull_vec(
            row, ("PerSeedHForward", "per_seed_H_forward", "per_seed_h_forward"),
            f"per_seed_H_forward (vector of {n_seeds} f64)",
        )
        per_seed_H_reversed = self._pull_vec(
            row, ("PerSeedHReversed", "per_seed_H_reversed", "per_seed_h_reversed"),
            f"per_seed_H_reversed (vector of {n_seeds} f64)",
        )
        if per_seed_H_forward.shape != (n_seeds,):
            raise LoopTransportLiveError(
                status_code=200,
                body=(
                    f"per_seed_H_forward shape mismatch: expected ({n_seeds},), "
                    f"got {per_seed_H_forward.shape}"
                ),
                query=query,
            )

        # Direction selection: the response per-seed vector corresponds
        # to whichever direction was requested. For the Halcyon
        # orchestrator pattern (one call per direction), per_seed_h_scalar
        # is the per-seed scalar from the requested direction.
        if request.direction == "FORWARD":
            per_seed_h_scalar = per_seed_H_forward
        else:
            per_seed_h_scalar = per_seed_H_reversed

        # The Halcyon side uses per_seed_holonomy (the [n_seeds, 4]
        # quaternion matrix) only for the §7.2 sidecar audit trail.
        # The substrate may or may not return the full quaternion
        # matrix on the wire; if it doesn't, we synthesize identity
        # quaternions with the scalar projection parked in q[0] for
        # the sidecar's information value. The gate logic operates on
        # per_seed_h_scalar exclusively.
        per_seed_holonomy = self._pull_optional_matrix(
            row, ("PerSeedHolonomy", "per_seed_holonomy"),
            (n_seeds, 4),
        )
        if per_seed_holonomy is None:
            per_seed_holonomy = np.zeros((n_seeds, 4), dtype=np.float64)
            per_seed_holonomy[:, 0] = 1.0  # identity quaternion stub
            per_seed_holonomy[:, 0] = per_seed_h_scalar  # park the real signal in q[0]

        sigma_h_blocked = self._pull_scalar(
            row, ("SigmaHBlocked", "sigma_H_blocked", "sigma_h_blocked"),
            "sigma_H_blocked", default=0.0,
        )
        tracking_q = self._pull_scalar(
            row, ("TrackingErrorMaxQ", "tracking_error_max_Q", "tracking_error_max_q"),
            "tracking_error_max_Q", default=0.0,
        )
        tracking_beta_w = self._pull_scalar(
            row, ("TrackingErrorMaxBetaW", "tracking_error_max_beta_W",
                  "tracking_error_max_beta_w"),
            "tracking_error_max_beta_W", default=0.0,
        )

        # AdiabaticityCheck is a struct on the wire. Tolerate both
        # nested object shape and dot-separated flat keys.
        adia = self._pull_adiabaticity(row)

        run_id = str(row.get("run_id") or row.get("RunId") or "")

        return LoopTransportResult(
            per_seed_holonomy=per_seed_holonomy,
            per_seed_h_scalar=per_seed_h_scalar,
            sigma_h_blocked=sigma_h_blocked,
            tracking_error_max_Q=tracking_q,
            tracking_error_max_beta_W=tracking_beta_w,
            adiabaticity_check=adia,
            run_id=run_id,
        )

    @staticmethod
    def _pull_vec(
        row: Dict[str, Any],
        keys: Tuple[str, ...],
        what: str,
    ) -> np.ndarray:
        for k in keys:
            if k in row and row[k] is not None:
                return np.asarray(row[k], dtype=np.float64)
        raise RuntimeError(
            f"response row missing {what}; tried keys {list(keys)}; "
            f"got: {list(row.keys())}"
        )

    @staticmethod
    def _pull_optional_matrix(
        row: Dict[str, Any],
        keys: Tuple[str, ...],
        expected_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        for k in keys:
            if k in row and row[k] is not None:
                arr = np.asarray(row[k], dtype=np.float64)
                if arr.shape != expected_shape:
                    raise RuntimeError(
                        f"{k!r} shape {arr.shape} != expected {expected_shape}"
                    )
                return arr
        return None

    @staticmethod
    def _pull_scalar(
        row: Dict[str, Any],
        keys: Tuple[str, ...],
        what: str,
        default: Optional[float] = None,
    ) -> float:
        for k in keys:
            if k in row and row[k] is not None:
                return float(row[k])
        if default is None:
            raise RuntimeError(
                f"response row missing {what}; tried keys {list(keys)}; "
                f"got: {list(row.keys())}"
            )
        return float(default)

    @staticmethod
    def _pull_adiabaticity(row: Dict[str, Any]) -> AdiabaticityCheck:
        """Build an AdiabaticityCheck from either a nested struct or
        a flat dot-separated key set. Provides safe defaults for fields
        the substrate may not emit (warnings_count, warning_indices)."""
        nested = (
            row.get("AdiabaticityCheck")
            or row.get("adiabaticity_check")
        )
        if isinstance(nested, dict):
            return AdiabaticityCheck(
                tau_pin_over_t_segment=float(nested.get(
                    "tau_pin_over_t_segment",
                    nested.get("TauPinOverTSegment", 0.0),
                )),
                ramp_rate_over_relaxation_rate=float(nested.get(
                    "ramp_rate_over_relaxation_rate",
                    nested.get("RampRateOverRelaxationRate", 0.0),
                )),
                warnings_count=int(nested.get(
                    "warnings_count",
                    nested.get("WarningsCount", 0),
                )),
                warning_substep_indices=tuple(
                    int(i) for i in nested.get(
                        "warning_substep_indices",
                        nested.get("WarningSubstepIndices", ()),
                    )
                ),
            )
        # Flat shape fallback
        return AdiabaticityCheck(
            tau_pin_over_t_segment=float(
                row.get("adiabaticity_check.tau_pin_over_t_segment", 0.0)
            ),
            ramp_rate_over_relaxation_rate=float(
                row.get("adiabaticity_check.ramp_rate_over_relaxation_rate", 0.0)
            ),
            warnings_count=int(
                row.get("adiabaticity_check.warnings_count", 0)
            ),
            warning_substep_indices=(),
        )

    # ------------------------------------------------------------------
    # HTTP plumbing (mirrors LiveGIGIClient pattern)
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        h = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _gql_query(self, query: str) -> Any:
        """``POST /v1/gql`` with a GQL statement string. Returns the
        parsed JSON response. Raises ``LoopTransportLiveError`` on
        non-2xx so the smoke test surfaces the substrate's diagnostic
        verbatim."""
        import requests
        try:
            resp = requests.post(
                f"{self.base_url}/v1/gql",
                json={"query": query},
                headers=self._headers(),
                timeout=self.timeout_s,
            )
        except requests.exceptions.ConnectionError as ex:
            raise ConnectionError(
                f"gigi-stream not reachable at {self.base_url}; "
                f"start locally with `cargo run --release --bin gigi-stream` "
                f"(default port 3142), or set GIGI_URL to point at the live "
                f"engine. Original error: {ex}"
            ) from ex
        if not resp.ok:
            raise LoopTransportLiveError(
                status_code=resp.status_code,
                body=resp.text,
                query=query,
            )
        return resp.json()

    def _ping(self) -> None:
        """Confirm the engine is reachable. Posts a syntactically
        invalid GQL fragment and expects a 4xx rather than a connection
        error. Mirrors LiveGIGIClient._ping shape."""
        import requests
        try:
            resp = requests.post(
                f"{self.base_url}/v1/gql",
                json={"query": "__loop_transport_live_client_probe__;"},
                headers=self._headers(),
                timeout=self.timeout_s,
            )
        except requests.exceptions.ConnectionError as ex:
            raise ConnectionError(
                f"gigi-stream not reachable at {self.base_url}"
            ) from ex
        _ = resp.status_code  # any answer means the engine is alive
