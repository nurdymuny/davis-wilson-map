"""LiveGIGIClient — HTTP-based adapter for the real gigi-stream engine.

**Phase A only.** This client targets the Part II HTTP surface that
GIGI shipped on 2026-06-17 (the Marsaglia-on-xorshift64* CSPRNG
decision). It covers:

  - ``POST /v1/lattice``       → ``declare_lattice``
  - ``GET  /v1/lattice/{name}`` → (used as the round-trip receipt)
  - ``POST /v1/gauge_field``   → ``declare_gauge_field``
  - ``GET  /v1/gauge_field/{name}`` → ``introspect_gauge_field``

Methods that the Part II HTTP surface doesn't yet expose
(``holonomy_on_faces``, ``holonomy_on_loop``, ``select_plaquette``,
``select_observable``, ``gibbs_sample``) raise ``NotImplementedError``
with a message naming which GIGI part will close the gap.

Wire decisions (locked by Gigi's Part II completeness critic):
  - Default URL: ``http://localhost:3142`` (matches ``fly.toml`` PORT).
    Override via the ``GIGI_URL`` env var.
  - HAAR_RANDOM buffers are NOT byte-equal to ``MockGIGIClient`` at
    the same seed by design (Bee's CSPRNG decision c): GIGI uses
    xorshift64*; the mock uses NumPy PCG64. The contract is intra-
    engine reproducibility, not cross-engine byte-identity.
  - ``persist`` defaults to False; this client does not yet exercise
    the durable persistence path.

This is NOT the production wire — the architectural decision recorded
in ``gigi_client/__init__.py`` is that the live binding for the
Halcyon production validator MUST be embedded (PyO3 / native CFFI),
not HTTP. The HTTP path is for Phase A end-to-end verification of
Parts I+II against the real engine. See
``theory/halcyon/HALCYON_PART_I_GATES.md`` and the
``wyqh19me8`` audit synthesis.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from inertia_damping.gigi_client.protocol import (
    EFieldHandle,
    EFieldInit,
    GaugeFieldHandle,
    GaugeFieldInit,
    GibbsSampleResult,
    Group,
    HolonomyResult,
    LatticeSpec,
    ObservableId,
    ProjectGaussConfig,
    SymplecticFlowDiagnostics,
    SymplecticFlowResult,
)

DEFAULT_GIGI_URL = "http://localhost:3142"
DEFAULT_TIMEOUT_S = 30.0


class LiveGIGIClient:
    """HTTP client for gigi-stream's Part II surface.

    Construct with an explicit ``base_url`` or let it read the
    ``GIGI_URL`` env var (defaults to localhost:3142). All requests
    use the ``requests`` library; pings the ``GET /v1/lattice/{name}``
    endpoint with a known-absent name to detect reachability at
    construction time when ``ping=True``.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        ping: bool = False,
        api_key: Optional[str] = None,
    ):
        try:
            import requests  # noqa: F401 — checked at construction time
        except ImportError as ex:
            raise RuntimeError(
                "LiveGIGIClient requires `pip install requests`. The mock "
                "client has no such dependency; install requests only when "
                "running Phase A live tests against gigi-stream."
            ) from ex
        self.base_url = (base_url or os.environ.get("GIGI_URL") or DEFAULT_GIGI_URL).rstrip("/")
        self.timeout_s = float(timeout_s)
        # Auth: explicit api_key arg > GIGI_API_KEY env > no auth.
        # Header shape is X-API-Key (per gigi/src/bin/gigi_stream.rs).
        self.api_key = api_key or os.environ.get("GIGI_API_KEY")
        if ping:
            self._ping()

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    # =================================================================
    # Reachability
    # =================================================================
    def _ping(self) -> None:
        """Confirm the engine is reachable. Hits a known-absent lattice
        and expects a 4xx (not a connection error)."""
        import requests
        try:
            resp = requests.get(
                f"{self.base_url}/v1/lattice/__live_client_probe__",
                headers=self._headers(), timeout=self.timeout_s,
            )
        except requests.exceptions.ConnectionError as ex:
            raise ConnectionError(
                f"gigi-stream not reachable at {self.base_url}. "
                f"Start it with `cargo run --release --bin gigi-stream` "
                f"(default port 3142), or override GIGI_URL."
            ) from ex
        # Anything that comes back from the engine is fine — even a 400
        # means the engine is alive and routing.
        _ = resp.status_code

    # =================================================================
    # Part I: LATTICE
    # =================================================================
    def declare_lattice(self, spec: LatticeSpec) -> str:
        """``POST /v1/lattice`` — declare a canonical lattice.

        Part II only supports the canonical constructor name
        ``TRUNCATED_ICOSAHEDRON`` (alias ``BUCKYBALL``). Arbitrary
        edge/face lists are a later sprint; if ``spec`` doesn't look
        like the canonical buckyball, this raises a clear error rather
        than silently failing.
        """
        import requests
        if not (spec.vertices == 60 and len(spec.edges) == 90 and len(spec.faces) == 32):
            raise NotImplementedError(
                "Part II HTTP only supports the canonical "
                "TRUNCATED_ICOSAHEDRON. Arbitrary-graph LATTICE is a "
                "later sprint."
            )
        body = {
            "name": spec.name,
            "topology": "TRUNCATED_ICOSAHEDRON",
            "topology_hint": spec.topology,
            "persist": False,
        }
        resp = requests.post(
            f"{self.base_url}/v1/lattice",
            json=body,
            headers=self._headers(), timeout=self.timeout_s,
        )
        if not resp.ok:
            raise RuntimeError(
                f"POST /v1/lattice failed: {resp.status_code} {resp.text}"
            )
        view = resp.json()
        # Sanity round-trip
        if view["n_vertices"] != spec.vertices:
            raise ValueError(
                f"engine returned n_vertices={view['n_vertices']}; "
                f"declared {spec.vertices}"
            )
        return view["name"]

    def lattice_view(self, name: str) -> Dict[str, Any]:
        """``GET /v1/lattice/{name}`` — used as the round-trip receipt."""
        import requests
        resp = requests.get(
            f"{self.base_url}/v1/lattice/{name}",
            headers=self._headers(), timeout=self.timeout_s,
        )
        if not resp.ok:
            raise RuntimeError(
                f"GET /v1/lattice/{name} failed: {resp.status_code} {resp.text}"
            )
        return resp.json()

    def holonomy_on_faces(self, field_name: str) -> HolonomyResult:
        raise NotImplementedError(
            "Part II HTTP does not expose HOLONOMY ON FACES. The kernel-side "
            "walker can compute it from the buffer returned by "
            "introspect_gauge_field(...); Part III is the gating event for "
            "an HTTP-native holonomy endpoint."
        )

    def holonomy_on_loop(self, field_name: str, edge_list: Sequence[int]) -> HolonomyResult:
        raise NotImplementedError(
            "Part II HTTP does not expose HOLONOMY ON LOOP. Walk the loop "
            "client-side from introspect_gauge_field(...) for Phase A."
        )

    # =================================================================
    # Part II: GAUGE_FIELD
    # =================================================================
    def declare_gauge_field(
        self,
        name: str,
        lattice_name: str,
        group: Group,
        init: GaugeFieldInit,
        seed: Optional[int] = None,
        from_field: Optional[str] = None,
    ) -> GaugeFieldHandle:
        """``POST /v1/gauge_field`` — declare with INIT spec."""
        import requests
        if init == GaugeFieldInit.IDENTITY:
            init_body: Dict[str, Any] = {"kind": "identity"}
        elif init == GaugeFieldInit.HAAR_RANDOM:
            if seed is None:
                raise ValueError("INIT HAAR_RANDOM requires SEED")
            init_body = {"kind": "haar_random", "seed": int(seed)}
        elif init == GaugeFieldInit.FROM_FIELD:
            if from_field is None:
                raise ValueError("INIT FROM_FIELD requires source field")
            init_body = {"kind": "from_field", "source": from_field}
        else:
            raise NotImplementedError(f"INIT kind {init} not supported")

        body = {
            "name": name,
            "lattice": lattice_name,
            "group": group.value,
            "init": init_body,
            "persist": False,
        }
        resp = requests.post(
            f"{self.base_url}/v1/gauge_field",
            json=body,
            headers=self._headers(), timeout=self.timeout_s,
        )
        if not resp.ok:
            raise self._error_from_response(resp, group)
        view = resp.json()
        return GaugeFieldHandle(
            name=view["name"],
            lattice_name=view["lattice"],
            group=group,
            repr_dim=view["repr_dim"],
            init_kind=init,
            init_seed=view.get("init_seed"),
            init_from=from_field,
        )

    def introspect_gauge_field(self, name: str) -> np.ndarray:
        """``GET /v1/gauge_field/{name}`` — returns the per-edge buffer."""
        import requests
        resp = requests.get(
            f"{self.base_url}/v1/gauge_field/{name}",
            headers=self._headers(), timeout=self.timeout_s,
        )
        if not resp.ok:
            raise RuntimeError(
                f"GET /v1/gauge_field/{name} failed: {resp.status_code} {resp.text}"
            )
        view = resp.json()
        # Wire shape: data is List[List[float]] with shape (n_edges, repr_dim).
        buf = np.asarray(view["data"], dtype=np.float64)
        if buf.shape != (view["n_edges"], view["repr_dim"]):
            raise ValueError(
                f"buffer shape {buf.shape} disagrees with view "
                f"(n_edges={view['n_edges']}, repr_dim={view['repr_dim']})"
            )
        return buf

    # =================================================================
    # Part III: HTTP read routes + GIBBS_SAMPLE via POST /v1/gql
    # =================================================================
    def select_plaquette(
        self,
        field_name: str,
        reduction: str = "per_face",
    ) -> "np.ndarray | float":
        """``GET /v1/gauge_field/{name}/plaquette?reduction=...``

        per_face -> shape (n_faces,) of Re tr(U_f) / 2
        mean / sum -> scalar reductions
        """
        import requests
        if reduction not in ("per_face", "mean", "sum"):
            raise ValueError(
                f"unknown reduction {reduction!r}; expected per_face / mean / sum"
            )
        resp = requests.get(
            f"{self.base_url}/v1/gauge_field/{field_name}/plaquette",
            params={"reduction": reduction},
            headers=self._headers(), timeout=self.timeout_s,
        )
        if not resp.ok:
            raise RuntimeError(
                f"GET .../plaquette failed: {resp.status_code} {resp.text}"
            )
        body = resp.json()
        if reduction == "per_face":
            return np.asarray(body["values"], dtype=np.float64)
        return float(body["value"])

    def select_observable(
        self,
        field_name: str,
        observable: ObservableId,
    ) -> float:
        """``GET /v1/gauge_field/{name}/q_surrogate`` for Q_SURROGATE.

        For MEAN_PLAQUETTE this delegates to select_plaquette(reduction='mean').
        H_TOTAL / GAUSS_RESIDUAL_MAX raise because they require an E field
        (Part IV).
        """
        import requests
        if observable == ObservableId.Q_SURROGATE:
            resp = requests.get(
                f"{self.base_url}/v1/gauge_field/{field_name}/q_surrogate",
                headers=self._headers(), timeout=self.timeout_s,
            )
            if not resp.ok:
                raise RuntimeError(
                    f"GET .../q_surrogate failed: {resp.status_code} {resp.text}"
                )
            return float(resp.json()["value"])
        if observable == ObservableId.MEAN_PLAQUETTE:
            return float(self.select_plaquette(field_name, reduction="mean"))
        if observable == ObservableId.H_TOTAL:
            raise ValueError(
                "H_TOTAL requires an E field; not available until Part IV "
                "(SYMPLECTIC_FLOW)."
            )
        if observable == ObservableId.GAUSS_RESIDUAL_MAX:
            raise ValueError(
                "GAUSS_RESIDUAL_MAX requires (U, E) state; Part IV not shipped."
            )
        raise ValueError(f"unknown observable {observable}")

    def select_observables_batch(
        self,
        field_name: str,
        observables: Sequence[str],
    ) -> Dict[str, float]:
        """``POST /v1/gauge_field/{name}/observables`` — batched read.

        Accepts ``["mean_plaquette", "sum_plaquette", "q_surrogate"]`` and
        returns a dict keyed by observable name.
        """
        import requests
        resp = requests.post(
            f"{self.base_url}/v1/gauge_field/{field_name}/observables",
            json={"observables": list(observables)},
            timeout=self.timeout_s,
        )
        if not resp.ok:
            raise RuntimeError(
                f"POST .../observables failed: {resp.status_code} {resp.text}"
            )
        return resp.json()

    def gibbs_sample(
        self,
        field_name: str,
        beta: float,
        n_sweeps: int = 1,
        measure_every: int = 1,
        measure: Sequence[ObservableId] = (),
        seed: Optional[int] = None,
    ) -> GibbsSampleResult:
        """``POST /v1/gql`` carrying a ``GIBBS_SAMPLE`` statement.

        This IS the Phase B verb: the thermalization algorithm runs on
        GIGI's Rust engine. Measurement chains come back as one column
        per observable in the Rows response.
        """
        if seed is None:
            raise ValueError("GIBBS_SAMPLE requires SEED (bit-identity contract)")
        # Build the GQL surface form
        obs_clause = ""
        if measure:
            obs_tokens = []
            for obs in measure:
                if obs == ObservableId.MEAN_PLAQUETTE:
                    obs_tokens.append("MEAN(PLAQUETTE)")
                elif obs == ObservableId.Q_SURROGATE:
                    obs_tokens.append("Q_SURROGATE")
                else:
                    raise ValueError(
                        f"observable {obs} not on the GQL surface for Part III; "
                        "MEAN_PLAQUETTE and Q_SURROGATE are available"
                    )
            obs_clause = f" MEASURE ({', '.join(obs_tokens)})"
        query = (
            f"GIBBS_SAMPLE {field_name}"
            f" BETA {beta}"
            f" N_SWEEPS {n_sweeps}"
            f" MEASURE_EVERY {measure_every}"
            f"{obs_clause}"
            f" SEED {int(seed)};"
        )
        body = self._gql_query(query)
        # The executor returns Rows = [{field, seed, beta, n_sweeps_completed,
        # <observable_label>: Vector}]. Pull the one row.
        rows = body.get("rows", body) if isinstance(body, dict) else body
        if isinstance(rows, dict):
            rows = rows.get("rows", [rows])
        if not rows:
            raise RuntimeError(
                f"GIBBS_SAMPLE returned no rows; full response: {body!r}"
            )
        row = rows[0]
        history: Dict[str, np.ndarray] = {}
        for obs in measure:
            # Try both label conventions: ObservableId.value (e.g. "MEAN(PLAQUETTE)")
            # and the snake_case label the GIGI executor emits.
            # gigi-stream returns measurement chains under CamelCase
            # enum-variant column names (MeanPlaquette / QSurrogate) per
            # the V.0 dispatch shape; snake_case + parser-string forms
            # are kept as fallbacks for older / local engines.
            camel = {
                ObservableId.MEAN_PLAQUETTE: "MeanPlaquette",
                ObservableId.Q_SURROGATE: "QSurrogate",
            }.get(obs)
            snake = {
                ObservableId.MEAN_PLAQUETTE: "mean_plaquette",
                ObservableId.Q_SURROGATE: "q_surrogate",
            }.get(obs)
            chain = None
            for key in (camel, snake, obs.value):
                if key and key in row:
                    chain = row[key]
                    break
            if chain is None:
                raise RuntimeError(
                    f"observable {obs} missing from response row; keys: "
                    f"{list(row.keys()) if isinstance(row, dict) else row}"
                )
            history[obs.value] = np.asarray(chain, dtype=np.float64)
        # Reconstruct a stub handle for the response
        handle = GaugeFieldHandle(
            name=field_name,
            lattice_name="",            # not provided in the Rows response
            group=Group.SU2,
            repr_dim=4,
            init_kind=GaugeFieldInit.IDENTITY,
            init_seed=None,
            init_from=None,
        )
        return GibbsSampleResult(
            final_field=handle,
            measurement_history=history,
            diagnostics={
                "n_sweeps_completed": int(row.get("n_sweeps_completed", n_sweeps)),
                "beta": float(row.get("beta", beta)),
                "seed": int(row.get("seed", seed)),
            },
        )

    # =================================================================
    # Part IV: E_FIELD + SYMPLECTIC_FLOW + (U, E) observables via /v1/gql
    # =================================================================
    def declare_e_field(
        self,
        name: str,
        gauge_field: str,
        init: EFieldInit,
        beta: Optional[float] = None,
        seed: Optional[int] = None,
        from_field: Optional[str] = None,
    ) -> EFieldHandle:
        """``E_FIELD name ON GAUGE_FIELD U INIT <init> [SEED s];``"""
        if init == EFieldInit.ZERO:
            init_clause = "ZERO"
        elif init == EFieldInit.MAXWELL_BOLTZMANN:
            if beta is None:
                raise ValueError("INIT MAXWELL_BOLTZMANN requires BETA")
            if seed is None:
                raise ValueError("INIT MAXWELL_BOLTZMANN requires SEED")
            init_clause = f"MAXWELL_BOLTZMANN BETA {beta}"
        elif init == EFieldInit.FROM_FIELD:
            if from_field is None:
                raise ValueError("INIT FROM requires source")
            init_clause = f"FROM_FIELD {from_field}"
        else:
            raise NotImplementedError(f"E_FIELD INIT {init}")
        seed_clause = f" SEED {int(seed)}" if seed is not None else ""
        query = (
            f"E_FIELD {name} ON GAUGE_FIELD {gauge_field} "
            f"INIT {init_clause}{seed_clause};"
        )
        _ = self._gql_query(query)
        return EFieldHandle(
            name=name,
            source_gauge_field=gauge_field,
            source_lattice="",  # not returned by the engine; reserved
            init_kind=init,
            init_beta=beta,
            init_seed=seed,
            init_from=from_field,
        )

    def select_h_total(self, gauge_field: str, e_field: str) -> float:
        body = self._gql_query(
            f"SELECT H_TOTAL OF ({gauge_field}, {e_field});"
        )
        return self._scalar_from_rows(body, "h_total", "H_TOTAL")

    def select_gauss_residual_max(
        self,
        gauge_field: str,
        e_field: str,
        reduction: str = "covariant",
    ) -> float:
        body = self._gql_query(
            f"SELECT GAUSS_RESIDUAL_MAX OF ({gauge_field}, {e_field});"
        )
        return self._scalar_from_rows(
            body, "gauss_residual_max", "GAUSS_RESIDUAL_MAX",
        )

    def symplectic_flow(
        self,
        field: str,
        e_field: str,
        beta: float,
        dt: float,
        n_steps: int,
        project_gauss: Optional[ProjectGaussConfig] = None,
        measure_every: int = 1,
        measure: Sequence[ObservableId] = (),
        seed: Optional[int] = None,
    ) -> SymplecticFlowResult:
        """``POST /v1/gql`` carrying a ``SYMPLECTIC_FLOW`` statement.

        Returns the Rows envelope lifted into a SymplecticFlowResult.
        """
        if project_gauss is None:
            pg_clause = " PROJECT_GAUSS FALSE"
        else:
            pg_clause = (
                " PROJECT_GAUSS { "
                f"tikhonov: {project_gauss.tikhonov}, "
                f"cg_tol: {project_gauss.cg_tol}, "
                f"cg_max_iter: {project_gauss.cg_max_iter}"
                " }"
            )
        obs_clause = ""
        if measure:
            obs_tokens = []
            for obs in measure:
                if obs == ObservableId.MEAN_PLAQUETTE:
                    obs_tokens.append("MEAN(PLAQUETTE)")
                elif obs == ObservableId.Q_SURROGATE:
                    obs_tokens.append("Q_SURROGATE")
                elif obs == ObservableId.H_TOTAL:
                    obs_tokens.append("H_TOTAL")
                elif obs == ObservableId.GAUSS_RESIDUAL_MAX:
                    obs_tokens.append("GAUSS_RESIDUAL_MAX")
                else:
                    raise ValueError(f"observable {obs} not in Part IV grammar")
            obs_clause = f" MEASURE ({', '.join(obs_tokens)})"
        seed_clause = f" SEED {int(seed)}" if seed is not None else ""
        query = (
            f"SYMPLECTIC_FLOW {field}"
            f" FROM (U={field}, E={e_field})"
            f" BETA {beta}"
            f" DT {dt}"
            f" N_STEPS {n_steps}"
            f"{pg_clause}"
            f" MEASURE_EVERY {measure_every}"
            f"{obs_clause}"
            f"{seed_clause};"
        )
        body = self._gql_query(query)
        rows = body.get("rows", body) if isinstance(body, dict) else body
        if isinstance(rows, dict):
            rows = rows.get("rows", [rows])
        if not rows:
            raise RuntimeError(f"SYMPLECTIC_FLOW returned no rows: {body!r}")
        row = rows[0]

        # Production gigi-stream returns CamelCase column names per the
        # V.0 dispatch (MeanPlaquette / QSurrogate / HTotal /
        # GaussResidualMax); snake_case + ObservableId.value forms kept
        # as fallbacks for older/local engines. Mirrors the same pattern
        # in gibbs_sample (commit 3f2652a).
        camel_map = {
            ObservableId.MEAN_PLAQUETTE: "MeanPlaquette",
            ObservableId.Q_SURROGATE: "QSurrogate",
            ObservableId.H_TOTAL: "HTotal",
            ObservableId.GAUSS_RESIDUAL_MAX: "GaussResidualMax",
        }
        snake_map = {
            ObservableId.MEAN_PLAQUETTE: "mean_plaquette",
            ObservableId.Q_SURROGATE: "q_surrogate",
            ObservableId.H_TOTAL: "h_total",
            ObservableId.GAUSS_RESIDUAL_MAX: "gauss_residual_max",
        }
        history: Dict[str, np.ndarray] = {}
        for obs in measure:
            camel = camel_map.get(obs)
            snake = snake_map.get(obs)
            chain = None
            for key in (camel, snake, obs.value):
                if key and key in row:
                    chain = row[key]
                    break
            if chain is None:
                raise RuntimeError(
                    f"observable {obs} missing from response; keys: {list(row.keys())}"
                )
            history[obs.value] = np.asarray(chain, dtype=np.float64)

        u_handle = GaugeFieldHandle(
            name=field, lattice_name="", group=Group.SU2, repr_dim=4,
            init_kind=GaugeFieldInit.IDENTITY,
        )
        e_handle = EFieldHandle(
            name=e_field, source_gauge_field=field, source_lattice="",
            init_kind=EFieldInit.MAXWELL_BOLTZMANN,
        )
        diagnostics = SymplecticFlowDiagnostics(
            run_id=str(row.get("run_id", "")),
            beta=float(row.get("beta", beta)),
            dt=float(row.get("dt", dt)),
            n_steps_completed=int(row.get("n_steps_completed", n_steps)),
            cg_iterations_per_step_p99=float(row.get("cg_iterations_per_step_p99", 0.0)),
            max_energy_drift_rel=float(row.get("max_energy_drift_rel", 0.0)),
            gauss_residual_max=float(row.get("gauss_residual_max", 0.0)),
            seed=int(row["seed"]) if row.get("seed") is not None else None,
        )
        return SymplecticFlowResult(
            final_field=u_handle,
            final_e_field=e_handle,
            measurement_history=history,
            diagnostics=diagnostics,
        )

    def _scalar_from_rows(self, body: Any, snake: str, ucase: str) -> float:
        rows = body.get("rows", body) if isinstance(body, dict) else body
        if isinstance(rows, dict):
            rows = rows.get("rows", [rows])
        if not rows:
            raise RuntimeError(f"empty response; expected scalar {ucase}")
        row = rows[0]
        for key in (snake, ucase, "value"):
            if key in row:
                return float(row[key])
        raise RuntimeError(
            f"scalar response missing expected key ({snake} or {ucase} or value); "
            f"keys: {list(row.keys())}"
        )

    def _gql_query(self, query: str) -> Any:
        """``POST /v1/gql`` with a GQL statement string. Returns the
        parsed JSON response."""
        import requests
        resp = requests.post(
            f"{self.base_url}/v1/gql",
            json={"query": query},
            headers=self._headers(), timeout=self.timeout_s,
        )
        if not resp.ok:
            raise RuntimeError(
                f"POST /v1/gql failed: {resp.status_code} {resp.text} "
                f"(query: {query!r})"
            )
        return resp.json()

    # =================================================================
    # Internals
    # =================================================================
    def _error_from_response(self, resp, group: Group) -> Exception:
        """Translate a non-2xx response into the same exception type the
        mock raises, so tests that share assertions across Mock/Live
        clients have one error contract."""
        try:
            body = resp.json()
            msg = body.get("error", resp.text)
        except Exception:
            msg = resp.text
        if "SU(2)" not in str(msg) and group != Group.SU2:
            # GIGI's group-erased storage rejects non-SU(2) groups; map to
            # the same NotImplementedError the mock raises so G2.D's regex
            # match keeps working across clients.
            return NotImplementedError(f"{msg} (group={group})")
        return RuntimeError(f"gigi-stream {resp.status_code}: {msg}")
