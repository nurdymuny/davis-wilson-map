"""Kernel-backed mock implementation of GIGIClient.

The mock delegates every protocol method to Halcyon's existing
Python kernel (``buckyball_graph`` / ``buckyball_action`` /
``buckyball_heatbath`` / ``buckyball_observables``). It is the
implementation under test for the Halcyon GIGI substrate gates.

Bit-identity contract (per HALCYON_TO_GIGI_REPLY_2026-06-17.md § A2):
the mock's per-method output, at fixed seed in the same process,
is the contract real GIGI must reproduce. The frozen goldens in
``gigi_client/golden/`` are derived from the mock and committed as
the gate-time reference.

When real GIGI ships Parts I-III, swap MockGIGIClient for the live
binding in Halcyon's orchestrator. The gate tests must still pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

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


@dataclass
class _GaugeFieldState:
    """In-memory mirror of what the GIGI engine holds for a registered
    GAUGE_FIELD. Test-only; the live binding never exposes this."""

    handle: GaugeFieldHandle
    link_buffer: torch.Tensor          # shape (n_edges, repr_dim)


class MockGIGIClient:
    """The mock that satisfies GIGIClient.

    Delegates to the buckyball_* kernel modules; carries an in-memory
    registry of declared lattices and gauge fields.
    """

    def __init__(self):
        self._lattices: Dict[str, "_LatticeState"] = {}
        self._fields: Dict[str, _GaugeFieldState] = {}
        self._e_fields: Dict[str, "_EFieldState"] = {}
        self._run_counter: int = 0

    # =================================================================
    # Part I: LATTICE + generalized HOLONOMY
    # =================================================================
    def declare_lattice(self, spec: LatticeSpec) -> str:
        from inertia_damping import buckyball_graph

        # Idempotency: re-declaring with the same shape returns the
        # same name; declaring with a different shape on the same
        # name is a contract violation.
        if spec.name in self._lattices:
            prev = self._lattices[spec.name].spec
            if (prev.vertices, prev.edges, prev.faces, prev.topology) != (
                spec.vertices, spec.edges, spec.faces, spec.topology,
            ):
                raise ValueError(
                    f"LATTICE {spec.name!r} already declared with a different shape"
                )
            return spec.name

        # Build the kernel-side graph object so observables can run.
        if (
            spec.vertices == 60
            and len(spec.edges) == 90
            and len(spec.faces) == 32
            and spec.topology == "S2"
        ):
            graph = buckyball_graph.build_truncated_icosahedron()
        else:
            raise NotImplementedError(
                "Mock only supports the truncated-icosahedron LATTICE at "
                "launch; real GIGI will support arbitrary graphs."
            )

        # Sanity: declared incidence must match what the kernel built.
        # (This is the round-trip receipt the gate enforces.)
        if graph.n_vertices != spec.vertices:
            raise ValueError(
                f"LATTICE vertices mismatch: declared {spec.vertices}, kernel {graph.n_vertices}"
            )
        if graph.n_edges != len(spec.edges):
            raise ValueError(
                f"LATTICE edges mismatch: declared {len(spec.edges)}, kernel {graph.n_edges}"
            )
        if graph.n_faces != len(spec.faces):
            raise ValueError(
                f"LATTICE faces mismatch: declared {len(spec.faces)}, kernel {graph.n_faces}"
            )
        euler = graph.n_vertices - graph.n_edges + graph.n_faces
        if euler != spec.euler_characteristic:
            raise ValueError(
                f"Euler characteristic mismatch: declared {spec.euler_characteristic}, "
                f"kernel V-E+F = {euler}"
            )

        self._lattices[spec.name] = _LatticeState(spec=spec, graph=graph)
        return spec.name

    def holonomy_on_faces(self, field_name: str) -> HolonomyResult:
        from inertia_damping import buckyball_action

        state = self._field_or_raise(field_name)
        lat = self._lattices[state.handle.lattice_name]
        # buckyball_action.all_face_holonomies returns (F, 4) of
        # quaternions; this IS the GQL response shape for SU(2).
        Uf = buckyball_action.all_face_holonomies(
            state.link_buffer, lat.graph,
        )
        Uf_np = Uf.detach().cpu().numpy().astype(np.float64)
        return HolonomyResult(holonomies=Uf_np, loop_count=Uf_np.shape[0])

    def holonomy_on_loop(
        self,
        field_name: str,
        edge_list: Sequence[int],
    ) -> HolonomyResult:
        from inertia_damping import buckyball_observables

        state = self._field_or_raise(field_name)
        lat = self._lattices[state.handle.lattice_name]
        # Single-loop holonomy. The kernel has face-holonomy; for an
        # arbitrary edge_list we walk the product directly. The GQL
        # surface will offer this natively in Part I.
        U = state.link_buffer
        n_edges = U.shape[0]
        for e in edge_list:
            if not (0 <= e < n_edges):
                raise ValueError(f"edge index {e} out of range [0, {n_edges})")
        # SU(2) quaternion product: identity = (1, 0, 0, 0).
        H = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=U.dtype)
        for e in edge_list:
            H = _quat_mul(H, U[e])
        H_np = H.detach().cpu().numpy().reshape(1, 4).astype(np.float64)
        return HolonomyResult(holonomies=H_np, loop_count=1)

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
        from inertia_damping import buckyball_action

        if lattice_name not in self._lattices:
            raise ValueError(f"LATTICE {lattice_name!r} not declared")
        lat = self._lattices[lattice_name]
        if group != Group.SU2:
            raise NotImplementedError(
                f"Mock only implements SU(2) at launch; got {group}. "
                "The group-erased storage layer exists but the verb math "
                "is SU(2)-only per Part II of the sprint gates."
            )
        repr_dim = 4  # SU(2) quaternion

        if init == GaugeFieldInit.IDENTITY:
            buf = buckyball_action.identity_links(lat.graph.n_edges)
        elif init == GaugeFieldInit.HAAR_RANDOM:
            if seed is None:
                raise ValueError("INIT HAAR_RANDOM requires SEED")
            buf = _haar_random_links(lat.graph.n_edges, seed=int(seed))
        elif init == GaugeFieldInit.FROM_FIELD:
            if from_field is None or from_field not in self._fields:
                raise ValueError(f"INIT FROM requires a registered source field; got {from_field!r}")
            buf = self._fields[from_field].link_buffer.detach().clone()
        else:
            raise NotImplementedError(f"INIT kind {init}")

        handle = GaugeFieldHandle(
            name=name,
            lattice_name=lattice_name,
            group=group,
            repr_dim=repr_dim,
            init_kind=init,
            init_seed=seed,
            init_from=from_field,
        )
        self._fields[name] = _GaugeFieldState(handle=handle, link_buffer=buf)
        return handle

    def introspect_gauge_field(self, name: str) -> np.ndarray:
        state = self._field_or_raise(name)
        return state.link_buffer.detach().cpu().numpy().astype(np.float64)

    # =================================================================
    # Part III: PLAQUETTE + observables + GIBBS_SAMPLE
    # =================================================================
    def select_plaquette(
        self,
        field_name: str,
        reduction: str = "per_face",
    ) -> "np.ndarray | float":
        from inertia_damping import buckyball_heatbath

        state = self._field_or_raise(field_name)
        lat = self._lattices[state.handle.lattice_name]
        per_face = buckyball_heatbath.plaquette_per_face(
            state.link_buffer, lat.graph,
        )
        per_face_np = per_face.detach().cpu().numpy().astype(np.float64)
        if reduction == "per_face":
            return per_face_np
        if reduction == "sum":
            return float(per_face_np.sum())
        if reduction == "mean":
            return float(per_face_np.mean())
        raise ValueError(f"unknown reduction {reduction!r}; "
                         "expected 'per_face' | 'sum' | 'mean'")

    def select_observable(
        self,
        field_name: str,
        observable: ObservableId,
    ) -> float:
        from inertia_damping import buckyball_observables

        state = self._field_or_raise(field_name)
        lat = self._lattices[state.handle.lattice_name]
        if observable == ObservableId.MEAN_PLAQUETTE:
            return float(buckyball_observables._plaquette_mean_from_U(
                state.link_buffer, lat.graph,
            ))
        if observable == ObservableId.Q_SURROGATE:
            return float(buckyball_observables.Q_surrogate(
                state.link_buffer, lat.graph,
            ))
        if observable == ObservableId.GAUSS_RESIDUAL_MAX:
            # No E field at Part III; mock returns 0.0 as a sentinel.
            # The actual gate test should not ask for this observable
            # outside Part IV.
            raise ValueError(
                "GAUSS_RESIDUAL_MAX requires (U, E) state; not available "
                "until SYMPLECTIC_FLOW (Part IV) ships."
            )
        if observable == ObservableId.H_TOTAL:
            raise ValueError(
                "H_TOTAL requires an E field; not available until Part IV."
            )
        raise ValueError(f"unknown observable {observable}")

    def gibbs_sample(
        self,
        field_name: str,
        beta: float,
        n_sweeps: int = 1,
        measure_every: int = 1,
        measure: Sequence[ObservableId] = (),
        seed: Optional[int] = None,
    ) -> GibbsSampleResult:
        from inertia_damping import buckyball_heatbath

        state = self._field_or_raise(field_name)
        lat = self._lattices[state.handle.lattice_name]
        if seed is None:
            raise ValueError("GIBBS_SAMPLE requires SEED for the bit-identity contract")
        rng = np.random.default_rng(int(seed))
        U = state.link_buffer
        history: Dict[str, List[float]] = {obs.value: [] for obs in measure}
        for s in range(n_sweeps):
            buckyball_heatbath.heatbath_sweep(U, lat.graph, beta, generator=rng)
            if (s + 1) % measure_every == 0:
                for obs in measure:
                    if obs == ObservableId.MEAN_PLAQUETTE:
                        from inertia_damping import buckyball_observables
                        history[obs.value].append(float(
                            buckyball_observables._plaquette_mean_from_U(U, lat.graph),
                        ))
                    elif obs == ObservableId.Q_SURROGATE:
                        from inertia_damping import buckyball_observables
                        history[obs.value].append(float(
                            buckyball_observables.Q_surrogate(U, lat.graph),
                        ))
                    else:
                        raise ValueError(
                            f"observable {obs} not available in Part III "
                            "(no E field; SYMPLECTIC_FLOW is Part IV)"
                        )
        state.link_buffer = U
        return GibbsSampleResult(
            final_field=state.handle,
            measurement_history={k: np.asarray(v, dtype=np.float64)
                                  for k, v in history.items()},
            diagnostics={
                "n_sweeps_completed": n_sweeps,
                "beta": float(beta),
                "seed": int(seed),
            },
        )

    # =================================================================
    # Part IV: E_FIELD + SYMPLECTIC_FLOW + (U, E) observables
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
        """``E_FIELD name ON GAUGE_FIELD U INIT <init> [SEED s];``

        ZERO              -> q0=0 everywhere, no seed.
        MAXWELL_BOLTZMANN -> Maxwell-Boltzmann Lie sample at β. SEED
                            required; mock delegates to
                            buckyball_integrator.initialize_E_canonical.
        FROM              -> clone an existing E field.
        """
        from inertia_damping import buckyball_integrator

        u_state = self._field_or_raise(gauge_field)
        lat = self._lattices[u_state.handle.lattice_name]

        if init == EFieldInit.ZERO:
            buf = torch.zeros((lat.graph.n_edges, 4), dtype=torch.float64)
        elif init == EFieldInit.MAXWELL_BOLTZMANN:
            if seed is None:
                raise ValueError("INIT MAXWELL_BOLTZMANN requires SEED")
            if beta is None:
                raise ValueError("INIT MAXWELL_BOLTZMANN requires BETA")
            gen_E = torch.Generator()
            gen_E.manual_seed(int(seed))
            E = buckyball_integrator.initialize_E_canonical(
                lat.graph, float(beta),
                generator=gen_E, project_gauss=True, U=u_state.link_buffer,
            )
            buf = E.detach().clone()
        elif init == EFieldInit.FROM_FIELD:
            if from_field is None or from_field not in self._e_fields:
                raise ValueError(f"INIT FROM requires a registered source; got {from_field!r}")
            buf = self._e_fields[from_field].buffer.detach().clone()
        else:
            raise NotImplementedError(f"E_FIELD INIT {init}")

        handle = EFieldHandle(
            name=name,
            source_gauge_field=gauge_field,
            source_lattice=u_state.handle.lattice_name,
            init_kind=init,
            init_beta=beta,
            init_seed=seed,
            init_from=from_field,
        )
        self._e_fields[name] = _EFieldState(handle=handle, buffer=buf)
        return handle

    def introspect_e_field(self, name: str) -> np.ndarray:
        if name not in self._e_fields:
            raise ValueError(f"E_FIELD {name!r} not declared")
        return self._e_fields[name].buffer.detach().cpu().numpy().astype(np.float64)

    def select_h_total(self, gauge_field: str, e_field: str) -> float:
        from inertia_damping import buckyball_integrator
        u_state = self._field_or_raise(gauge_field)
        if e_field not in self._e_fields:
            raise ValueError(f"E_FIELD {e_field!r} not declared")
        e_state = self._e_fields[e_field]
        lat = self._lattices[u_state.handle.lattice_name]
        H, _K, _V = buckyball_integrator.compute_hamiltonian(
            u_state.link_buffer, e_state.buffer, lat.graph, beta=1.0,
        )
        return float(H)

    def select_gauss_residual_max(
        self,
        gauge_field: str,
        e_field: str,
        reduction: str = "covariant",
    ) -> float:
        from inertia_damping import buckyball_integrator
        u_state = self._field_or_raise(gauge_field)
        if e_field not in self._e_fields:
            raise ValueError(f"E_FIELD {e_field!r} not declared")
        e_state = self._e_fields[e_field]
        lat = self._lattices[u_state.handle.lattice_name]
        G = buckyball_integrator.compute_gauss_residual(
            e_state.buffer, u_state.link_buffer, lat.graph,
        )
        return float(G.abs().max())

    def symplectic_flow(
        self,
        field: str,
        e_field: str,
        beta: float,
        dt: float,
        n_steps: int,
        project_gauss: Optional[ProjectGaussConfig] = ProjectGaussConfig.default(),
        measure_every: int = 1,
        measure: Sequence[ObservableId] = (),
        seed: Optional[int] = None,
    ) -> SymplecticFlowResult:
        """``SYMPLECTIC_FLOW`` — leapfrog with covariant Gauss projection.

        Delegates to buckyball_observables.integrate_with_states (the
        Halcyon kernel that produced the v1.2 production canonical).
        ``project_gauss=None`` disables projection; a struct enables it
        with the given tolerances.
        """
        from inertia_damping import buckyball_observables, buckyball_integrator

        u_state = self._field_or_raise(field)
        if e_field not in self._e_fields:
            raise ValueError(f"E_FIELD {e_field!r} not declared")
        e_state = self._e_fields[e_field]
        lat = self._lattices[u_state.handle.lattice_name]

        U_init = u_state.link_buffer.detach().clone()
        E_init = e_state.buffer.detach().clone()
        H_init_tuple = buckyball_integrator.compute_hamiltonian(
            U_init, E_init, lat.graph, beta=float(beta),
        )
        H_init = float(H_init_tuple[0])

        res = buckyball_observables.integrate_with_states(
            U_init, E_init,
            dt=float(dt), n_steps=int(n_steps), graph=lat.graph, beta=float(beta),
            measure_every=max(1, int(measure_every)),
            include_initial=False,
        )
        # Mutate the source U + E in place to match the GIGI semantics
        # (the flow mutates the source U in place; E is replaced).
        u_state.link_buffer = res["U_final"]
        e_state.buffer = res["E_final"]

        # Diagnostics
        Hs = np.asarray(res["H_history"], dtype=np.float64)
        max_energy_drift_rel = float(np.abs((Hs - H_init) / H_init).max()) \
            if H_init != 0.0 else 0.0
        G_final = buckyball_integrator.compute_gauss_residual(
            res["E_final"], res["U_final"], lat.graph,
        )
        gauss_residual_max = float(G_final.abs().max())

        # Measurement history
        history: Dict[str, np.ndarray] = {}
        n_records = len(res["U_traj"])
        for obs in measure:
            chain = np.zeros(n_records, dtype=np.float64)
            for i in range(n_records):
                U_i = res["U_traj"][i]
                E_i = res["E_traj"][i]
                if obs == ObservableId.MEAN_PLAQUETTE:
                    chain[i] = float(buckyball_observables._plaquette_mean_from_U(
                        U_i, lat.graph,
                    ))
                elif obs == ObservableId.Q_SURROGATE:
                    chain[i] = float(buckyball_observables.Q_surrogate(
                        U_i, lat.graph,
                    ))
                elif obs == ObservableId.H_TOTAL:
                    H_i, _K, _V = buckyball_integrator.compute_hamiltonian(
                        U_i, E_i, lat.graph, beta=float(beta),
                    )
                    chain[i] = float(H_i)
                elif obs == ObservableId.GAUSS_RESIDUAL_MAX:
                    G_i = buckyball_integrator.compute_gauss_residual(
                        E_i, U_i, lat.graph,
                    )
                    chain[i] = float(G_i.abs().max())
                else:
                    raise ValueError(f"unsupported observable {obs}")
            history[obs.value] = chain

        self._run_counter += 1
        diagnostics = SymplecticFlowDiagnostics(
            run_id=f"mock-symplectic-{self._run_counter:06d}",
            beta=float(beta),
            dt=float(dt),
            n_steps_completed=int(n_steps),
            cg_iterations_per_step_p99=float(res.get("cg_iter_p99", 0.0)),
            max_energy_drift_rel=max_energy_drift_rel,
            gauss_residual_max=gauss_residual_max,
            seed=int(seed) if seed is not None else None,
        )
        return SymplecticFlowResult(
            final_field=u_state.handle,
            final_e_field=e_state.handle,
            measurement_history=history,
            diagnostics=diagnostics,
        )

    # =================================================================
    # Helpers
    # =================================================================
    def _field_or_raise(self, name: str) -> _GaugeFieldState:
        if name not in self._fields:
            raise ValueError(f"GAUGE_FIELD {name!r} not declared")
        return self._fields[name]


@dataclass
class _LatticeState:
    spec: LatticeSpec
    graph: Any                  # buckyball_graph.Graph; opaque to the mock


@dataclass
class _EFieldState:
    """In-memory mirror of what the GIGI engine holds for a registered
    E_FIELD. The buffer is (n_edges, 4) row-major su(2) Lie algebra
    elements with q0 = 0."""

    handle: EFieldHandle
    buffer: torch.Tensor


# ---------------------------------------------------------------------
# Helpers shared with the heatbath internals
# ---------------------------------------------------------------------
def _haar_random_links(n_edges: int, seed: int) -> torch.Tensor:
    """Per-edge Haar-random SU(2) draw. Same convention as the
    orchestrator's GAUGE_FIELD INIT HAAR_RANDOM."""
    rng = np.random.default_rng(seed)
    # Marsaglia method: pick u1, u2 in unit disks until both are inside,
    # then build the unit quaternion (q0, q1, q2, q3).
    out = np.empty((n_edges, 4), dtype=np.float64)
    for e in range(n_edges):
        while True:
            x1, x2 = rng.uniform(-1.0, 1.0, size=2)
            s1 = x1 * x1 + x2 * x2
            if s1 < 1.0:
                break
        while True:
            x3, x4 = rng.uniform(-1.0, 1.0, size=2)
            s2 = x3 * x3 + x4 * x4
            if s2 < 1.0:
                break
        factor = math.sqrt((1.0 - s1) / s2)
        out[e] = (x1, x2, x3 * factor, x4 * factor)
    return torch.from_numpy(out)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SU(2) quaternion product. Matches buckyball_observables convention."""
    a0, a1, a2, a3 = a[0], a[1], a[2], a[3]
    b0, b1, b2, b3 = b[0], b[1], b[2], b[3]
    return torch.stack([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ])
