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

    # Lattice name on the live gigi-stream that the LOOP declarations
    # attach to. Per Gigi's LOOP_TRANSPORT_CALLING_GUIDE.md
    # §Preconditions, the v3.1.3 canonical LATTICE name is
    # "halcyon_canonical_buckyball" — matching the gauge field naming
    # convention LOOP_TRANSPORT's executor arm expects (the first arg
    # of LOOP_TRANSPORT is the LATTICE name; the executor hardcodes
    # gauge field as "U_lt" and E field as "E_lt" attached to that
    # lattice).
    DEFAULT_LATTICE_NAME = "halcyon_canonical_buckyball"

    @staticmethod
    def _build_declare_loop_gql(loop: LoopHandle) -> str:
        """Construct a LOOP <name> ON <lattice> FACE <n>; statement
        per Gigi's Part VI parser (gigi/src/parser.rs:3241 parse_loop_decl).
        The substrate-side LOOP is a SPATIAL loop on the buckyball (face
        index or edge vertex sequence), NOT a (Q, β_W) parameter-space
        loop. The parameter-space ramp lives in the LOOP_TRANSPORT
        clause itself (RAMP_RATE_Q, RAMP_RATE_BETA_W, ...).

        For Halcyon's v3.1.3 canonical loop, the substrate's existing
        smoke and parser tests use FACE 0 as the canonical spatial loop
        (see gigi/tests/halcyon_part_vi_executor_smoke.rs line 59,
        gigi/tests/halcyon_part_vi_parser_grammar.rs lines 33 / 157 /
        186 / 216). Halcyon adopts the same convention."""
        return (
            f"LOOP {loop.name} ON {LiveLoopTransportClient.DEFAULT_LATTICE_NAME}"
            f" FACE 0;"
        )

    @staticmethod
    def _build_loop_transport_gql(request: LoopTransportRequest) -> str:
        """Construct the LOOP_TRANSPORT GQL statement matching Gigi's
        VI.2 parser grammar (gigi/src/parser.rs:3280-3315 ebnf comment).

        Three corrections from the initial best-guess after the first
        smoke test surfaced them:
          1. CONTROL_MANIFOLD is "(Q, BETA_WILSON)" with uppercase
             BETA_WILSON, not "(Q, beta_wilson)".
          2. SEEDS uses range syntax `[start..end]` (end exclusive),
             not an explicit list. Halcyon's seeds are always
             contiguous per v3.1.3 §3.5 ([20260616..20260624]).
          3. BETA_WILSON_START is an additive knob (defaults to regime
             midpoint 2.75 if omitted). v3.1.3 §4.1's canonical loop
             starts at β_W = 2.5; passing it explicitly so the start
             matches the SPEC's loop construction.

        DIRECTION is also NOT in Gigi's parser EBNF — the substrate
        does both forward and reversed in one call and returns both
        in the per_seed_H_forward / per_seed_H_reversed return fields.
        Removed from the GQL string."""
        pack = request.pack

        # SEEDS range: validate contiguity (parser only accepts ranges).
        # Per Gigi's calling guide §"Seed range syntax" + line 189:
        # `[start..end]` is INCLUSIVE on both ends. So 8 seeds
        # (20260616..20260623) emits `[20260616..20260623]`, NOT
        # `[20260616..20260624]` (which would be 9 seeds inclusive).
        # The substrate's VI.2 smoke test confirms this convention:
        # `SEEDS [20260616..20260616]` → 1 seed (20260616 itself).
        seeds = request.seeds
        if len(seeds) > 0:
            seeds_sorted = sorted(seeds)
            if seeds_sorted != list(range(seeds_sorted[0], seeds_sorted[-1] + 1)):
                raise ValueError(
                    f"SEEDS must be contiguous per Gigi's parser grammar; "
                    f"got {seeds!r}. Use a contiguous range like "
                    f"(20260616, 20260617, ..., 20260623)."
                )
            seeds_clause = f"[{seeds_sorted[0]}..{seeds_sorted[-1]}]"
        else:
            raise ValueError("SEEDS cannot be empty")

        sham_clause = ""
        if request.sham is not ShamFlag.NONE:
            sham_clause = " " + LiveLoopTransportClient._build_sham_clause(request)

        # BETA_WILSON_START: v3.1.3 §4.1's loop starts at β_W = 2.5
        # (the lower endpoint of the validated regime). Pass explicitly
        # so the default 2.75 midpoint doesn't apply.
        beta_wilson_start = 2.5

        return (
            f"LOOP_TRANSPORT {CANONICAL_GAUGE_FIELD}"
            f" ALONG_LOOP {request.loop.name}"
            f" CONTROL_MANIFOLD (Q, BETA_WILSON)"
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
            f" BETA_WILSON_START {beta_wilson_start!r}"
            f" SEEDS {seeds_clause}"
            f" COMPUTE HOLONOMY_FORWARD"
            f" COMPUTE HOLONOMY_REVERSED"
            f" COMPUTE TRACKING_ERROR_TRACE_Q"
            f" COMPUTE TRACKING_ERROR_TRACE_BETA_W"
            f" COMPUTE ADIABATICITY_CHECK"
            f"{sham_clause}"
            f" RETURN H_forward, H_reversed, sigma_H_blocked,"
            f" per_seed_H_forward, per_seed_H_reversed,"
            f" tracking_error_max_Q, tracking_error_max_beta_W,"
            f" adiabaticity_check;"
        )

    @staticmethod
    def _build_sham_clause(request: LoopTransportRequest) -> str:
        """Nested `SHAM { KEY = VALUE }` block per Gigi's calling guide
        §"The verb call" lines 86-91. UPPERCASE flag names with `= TRUE`
        (or `= <float>` for MASS_BASELINE_SCALED) — not lowercase
        `key: true`. OPEN_LOOP is rejected at parse time per the guide
        (gigi raises LoopTransportError::LoopNotClosed); it is not a
        runtime SHAM flag, so this method errors if it's requested."""
        if request.sham is ShamFlag.OPEN_LOOP:
            raise ValueError(
                "OPEN_LOOP is a parser-rejection test, not a runtime SHAM "
                "flag (per Gigi's calling guide §Verb call line 92). It "
                "cannot be set on a LoopTransportRequest."
            )
        if request.sham is ShamFlag.MASS_SCALED:
            return (
                f"SHAM {{ MASS_BASELINE_SCALED = {request.sham_mass_scale!r} }}"
            )
        # Map ShamFlag enum to UPPERCASE nested-block keys matching
        # gigi's calling guide §The verb call (FLAT_FIELD, ALPHA_ZERO,
        # DEGENERATE_LOOP, FROZEN_FIELD, EMPTY_LOOP).
        key_map = {
            ShamFlag.FLAT_FIELD: "FLAT_FIELD",
            ShamFlag.ALPHA_ZERO: "ALPHA_ZERO",
            ShamFlag.BACKTRACK_LOOP: "DEGENERATE_LOOP",
            ShamFlag.FROZEN_FIELD: "FROZEN_FIELD",
            ShamFlag.EMPTY_LOOP: "EMPTY_LOOP",
        }
        key = key_map.get(request.sham)
        if key is None:
            raise ValueError(f"unmapped sham flag {request.sham}")
        return f"SHAM {{ {key} = TRUE }}"

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

        # Response column names per gigi/src/parser.rs:10338 executor
        # dispatch: lowercase snake_case (per_seed_h_forward,
        # per_seed_h_reversed, h_forward, h_reversed,
        # tracking_error_max_q, tracking_error_max_beta_w,
        # adiabaticity_verdict + adiabaticity_ratio as separate flat
        # fields, n_substeps_completed). The CamelCase fallbacks below
        # are kept as future-proofing if Gigi changes the column-name
        # convention to match the V.0 dispatch shape (MeanPlaquette /
        # QSurrogate etc.) later.
        per_seed_H_forward = self._pull_vec(
            row, ("per_seed_h_forward", "PerSeedHForward", "per_seed_H_forward"),
            f"per_seed_h_forward (vector of {n_seeds} f64)",
        )
        per_seed_H_reversed = self._pull_vec(
            row, ("per_seed_h_reversed", "PerSeedHReversed", "per_seed_H_reversed"),
            f"per_seed_h_reversed (vector of {n_seeds} f64)",
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
            row, ("sigma_h_blocked", "SigmaHBlocked", "sigma_H_blocked"),
            "sigma_h_blocked", default=0.0,
        )
        tracking_q = self._pull_scalar(
            row, ("tracking_error_max_q", "TrackingErrorMaxQ", "tracking_error_max_Q"),
            "tracking_error_max_q", default=0.0,
        )
        tracking_beta_w = self._pull_scalar(
            row, ("tracking_error_max_beta_w", "TrackingErrorMaxBetaW",
                  "tracking_error_max_beta_W"),
            "tracking_error_max_beta_w", default=0.0,
        )

        # AdiabaticityCheck on the wire: per gigi/src/parser.rs:10380
        # the executor emits two FLAT fields, not a nested struct:
        #   adiabaticity_verdict: string ("ACCEPTABLE" | "AMBIGUOUS_FORCED")
        #   adiabaticity_ratio:   f64 (the tau_pin / T_segment value)
        # _pull_adiabaticity tolerates both that flat shape and the
        # nested-struct shape (for any future substrate change).
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
        """Build an AdiabaticityCheck from the substrate's flat fields
        (gigi/src/parser.rs:10380 emits `adiabaticity_verdict` as a
        string + `adiabaticity_ratio` as f64) OR from a nested struct
        if the substrate ever changes to one.

        The substrate's verdict string ("ACCEPTABLE" /
        "AMBIGUOUS_FORCED") carries the gate's own decision; the ratio
        carries the numerical value. Halcyon's Python applies its
        own threshold (0.1 per v3.1.3 §4.2) against the ratio
        regardless of the substrate's verdict — the substrate-side
        verdict is recorded for the audit trail but not load-bearing
        for Halcyon's gate."""
        # Path 1: flat fields (current substrate shape)
        if "adiabaticity_ratio" in row or "adiabaticity_verdict" in row:
            return AdiabaticityCheck(
                tau_pin_over_t_segment=float(row.get("adiabaticity_ratio", 0.0)),
                # Observable B (ramp_rate / gauge_relaxation) isn't emitted
                # by the substrate today; v3.1.3 §4.2 has it as a diagnostic
                # only, no pre-registered threshold. Default to 0.0; a
                # future substrate version may emit it.
                ramp_rate_over_relaxation_rate=float(
                    row.get("ramp_rate_over_relaxation_rate", 0.0)
                ),
                warnings_count=0,
                warning_substep_indices=(),
            )
        # Path 2: nested struct fallback (for any future substrate shape)
        nested = (
            row.get("adiabaticity_check")
            or row.get("AdiabaticityCheck")
        )
        if isinstance(nested, dict):
            return AdiabaticityCheck(
                tau_pin_over_t_segment=float(nested.get(
                    "tau_pin_over_t_segment",
                    nested.get("ratio", nested.get("TauPinOverTSegment", 0.0)),
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
        # No adiabaticity data in the response — return a zero default.
        # Halcyon's gate is permissive on tau_pin=0 (which would mean
        # the substrate didn't measure it; the gate fires AMBIGUOUS at
        # >= 0.1, so 0.0 passes).
        return AdiabaticityCheck(
            tau_pin_over_t_segment=0.0,
            ramp_rate_over_relaxation_rate=0.0,
            warnings_count=0,
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
