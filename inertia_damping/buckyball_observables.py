"""
buckyball_observables.py - Real-space observables for SU(2) 2D Yang-Mills on the
truncated icosahedron (buckyball / soccer-ball graph), plus a trajectory dumper
for the cage_preview.html viewer.

This module owns two related layers:

1. SCALAR-FIELD OBSERVABLES
   - q_face_surrogate(U, f, graph) -> float
     Per-face fractional charge q_f = arccos(q0(U_f)) / (2 pi).
     U_f is the face holonomy quaternion (see buckyball_action.face_holonomy);
     q0(U_f) lies in [-1, 1] so arccos(q0) lies in [0, pi] and q_f lies in
     [0, 1/2].  Gauge-invariant because U_f -> g(v_base) U_f g(v_base)^dag
     leaves q0 (= (1/2) Tr U_f) invariant.

   - Q_surrogate(U, graph) -> float
     Sum over all 32 faces of q_f.  Range [0, 16] in principle, but at the
     beta = 2.5 working point typical values are O(few).  This is NOT the cubic
     clover-Q (which lives in 4D and is topologically quantized).  S^2 has
     pi_2(SU(2)) = pi_2(S^3) = 0, so there is no instanton number to recover
     here; Q_surrogate is a *Wilson-flow-style* fractional accumulator that
     tracks how far the configuration sits from the U = I vacuum.

2. PER-EDGE SCALARS (for the viewer)
   - edge_phase(U) -> Tensor (n_edges,)
     Rotation angle theta_e = arctan2(|q_vec(U_e)|, q0(U_e)) in [0, pi].
     Gauge-variant (U_e transforms by left/right multiplication, which mixes
     q0 and q_vec) but visually crisp; matches what users intuit by "edge
     phase" on a gauge link.

   - edge_kinetic(E) -> Tensor (n_edges,)
     Tr(E_e^2) for the canonical-momentum quaternion E_e.  For SU(2) with
     E = i e.sigma stored as E.q0 = 0, E.q_vec = e, the matrix square is
     -|e|^2 I_2 so Tr(E^2) = -2 |e|^2 ... but the validated symplectic
     integrator stores the algebra element with E[..., 0] = 0 and the
     *real* 3-vector in E[..., 1:].  The momentum invariant the kernel
     uses is K_e = (1/2) sum_a E_e^a E_e^a (see symplectic_integrator.py /
     buckyball_integrator.py kinetic_energy), so for consistency with the
     rest of the stack we expose
         edge_kinetic(E) = 2 |q_vec(E)|^2
     which is identically 2 * K_e and matches the docstring's claim
     Tr(E_e^2) = 2 |q_vec(E)|^2 once the i.sigma factor is absorbed into
     the convention.  Gauge-covariant; we display its magnitude in the
     viewer so the covariance is harmless.

3. TRAJECTORY DUMPER
   - dump_trajectory(...) writes the JSON schema documented in WF#3:
     schema_version, gauge_group, lattice, geometry, frames[].  Each frame
     carries t, step, Q, plaquette_mean, energy, control{}, edge_phases[90],
     edge_kinetic[90].

   The integrator does not save per-step U/E.  We wrap it: call
     integrate(U0, E0, beta, n_steps, measure_every, ...)
   repeatedly with the *previous* (U, E) as new initial conditions so we can
   harvest a U/E snapshot at the end of each measurement window.  This is a
   pure read of the validated integrator -- no kernel modification.

LAZY IMPORT
   Same pattern as buckyball_action / buckyball_heatbath: qmul/qconj/qnorm
   are pulled lazily from validation/matter_sector/su2_gauge_higgs.py.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from inertia_damping.buckyball_action import (
    all_face_holonomies,
    face_holonomy,
)

RDTYPE = torch.float64
TWO_PI = 2.0 * math.pi

# Schema version for trajectory.json files written by dump_trajectory.
#   "0.1" - original schema (WF#3): scalar+control+edge_phases+edge_kinetic.
#   "0.2" - WF#4 drill-down extension: adds per-frame
#           face_actions[F], face_q0[F], vertex_gauss[V], edge_face_types[E].
# load_trajectory accepts BOTH; v0.1 frames omit the four new fields.
SCHEMA_VERSION = "0.2"
_KNOWN_SCHEMA_VERSIONS = ("0.1", "0.2")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

_su2_mod: Any = None


def _load_su2():
    """Import qmul / qconj / qnorm from the validated matter-sector module."""
    global _su2_mod
    if _su2_mod is not None:
        return _su2_mod
    rel = os.path.join("validation", "matter_sector", "su2_gauge_higgs.py")
    abs_path = os.path.join(_ROOT, rel)
    spec = importlib.util.spec_from_file_location("_su2_gauge_higgs_obs", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream SU(2) module from {abs_path!r}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _su2_mod = mod
    return mod


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _load_su2().qmul(a, b)


def _qconj(a: torch.Tensor) -> torch.Tensor:
    return _load_su2().qconj(a)


# ----------------------------------------------------------------------
# Per-face fractional charge
# ----------------------------------------------------------------------
def q_face_surrogate(U: torch.Tensor, face_idx: int, graph) -> float:
    """Per-face fractional charge q_f = arccos(q0(U_f)) / (2 pi).

    For U_f in SU(2), q0(U_f) = (1/2) Tr U_f lies in [-1, 1], so
    arccos(q0) lies in [0, pi] and q_f lies in [0, 1/2].
    Equals 0 exactly at U = I (q0 = 1) and is gauge-invariant because
    U_f -> g U_f g^dag preserves the trace.
    """
    Uf = face_holonomy(U, face_idx, graph)
    q0 = float(Uf[0].item())
    # Clamp for floating noise (q0 should be in [-1, 1] already).
    q0_c = max(-1.0, min(1.0, q0))
    return math.acos(q0_c) / TWO_PI


def Q_surrogate(U: torch.Tensor, graph) -> float:
    """Sum of q_face_surrogate over all faces.

    Range: [0, n_faces / 2] = [0, 16] for the truncated icosahedron.
    Equals 0 at U = I.  Not topologically quantized on S^2; tracks the
    cumulative angular distance of face holonomies from the vacuum.
    """
    Uf = all_face_holonomies(U, graph)  # (F, 4)
    q0 = Uf[:, 0].clamp(-1.0, 1.0)
    q_per_face = torch.acos(q0) / TWO_PI
    return float(q_per_face.sum().item())


def q_face_surrogate_all(U: torch.Tensor, graph) -> torch.Tensor:
    """Vector of per-face q_f values, shape (n_faces,)."""
    Uf = all_face_holonomies(U, graph)
    q0 = Uf[:, 0].clamp(-1.0, 1.0)
    return torch.acos(q0) / TWO_PI


# ----------------------------------------------------------------------
# Per-edge scalars
# ----------------------------------------------------------------------
def edge_phase(U: torch.Tensor) -> torch.Tensor:
    """Per-edge rotation angle theta_e = arctan2(|q_vec(U_e)|, q0(U_e)).

    Input U has shape (n_edges, 4); output has shape (n_edges,) and lies
    in [0, pi].  Gauge-variant (depends on a basis at each endpoint),
    used for visualization, not for invariants.
    """
    if U.dim() != 2 or U.shape[-1] != 4:
        raise ValueError(f"edge_phase expects (n_edges, 4); got {tuple(U.shape)}")
    q0 = U[:, 0]
    q_vec_norm = torch.sqrt((U[:, 1:] * U[:, 1:]).sum(dim=-1).clamp_min(0.0))
    return torch.atan2(q_vec_norm, q0)


def edge_kinetic(E: torch.Tensor) -> torch.Tensor:
    """Per-edge kinetic invariant Tr(E_e^2) = 2 |q_vec(E_e)|^2.

    Input E has shape (n_edges, 4) with E[:, 0] = 0 for su(2) algebra
    elements; output has shape (n_edges,), all non-negative.
    Gauge-covariant in the same sense as |E|^2 on a Lie algebra:
    its magnitude is invariant under the adjoint action, which is what
    the viewer displays.
    """
    if E.dim() != 2 or E.shape[-1] != 4:
        raise ValueError(f"edge_kinetic expects (n_edges, 4); got {tuple(E.shape)}")
    return 2.0 * (E[:, 1:] * E[:, 1:]).sum(dim=-1)


# ----------------------------------------------------------------------
# Per-face / per-vertex drill-down observables (schema v0.2)
# ----------------------------------------------------------------------
def face_q0_values(U: torch.Tensor, graph) -> List[float]:
    """Per-face q0(U_f) = (1/2) Re Tr U_f, length n_faces (32 on buckyball).

    Gauge-invariant. q0 in [-1, 1].  Used in the viewer as the raw scalar
    that defines both the per-face Wilson action contribution and the
    per-face fractional-charge surrogate q_f = arccos(q0)/(2 pi).
    """
    Uf = all_face_holonomies(U, graph)
    q0 = Uf[:, 0].clamp(-1.0, 1.0)
    return [float(x) for x in q0.tolist()]


def face_actions(U: torch.Tensor, graph) -> List[float]:
    """Per-face Wilson action contribution S_f = [1 - q0(U_f)] for SU(2).

    For SU(2) the full Wilson action is
        S_W = (beta/N) sum_f [N - Re Tr U_f]
            = beta * sum_f [1 - q0(U_f)]
    so sum(face_actions(U, graph)) * beta equals wilson_action(U, graph, beta).
    Per-face range is [0, 2] (0 at U=I, 2 at U=-I).
    """
    Uf = all_face_holonomies(U, graph)
    q0 = Uf[:, 0].clamp(-1.0, 1.0)
    return [float(1.0 - x) for x in q0.tolist()]


def vertex_gauss_norms(E: torch.Tensor, U: torch.Tensor, graph) -> List[float]:
    """Per-vertex covariant Gauss-residual L2 norm, length n_vertices.

    Thin wrapper around buckyball_integrator.compute_gauss_residual which
    returns shape (V, 3); we take the L2 norm per row.  At FP64 precision
    a kernel-projected configuration should sit at ~1e-15 (machine epsilon
    floor); this is the receipt the drill-down panel displays.
    """
    # Lazy import so we do not couple at module load.
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "_buckyball_obs_integrator_gauss",
        os.path.join(_HERE, "buckyball_integrator.py"),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load buckyball_integrator.py")
    bi = _ilu.module_from_spec(spec)
    spec.loader.exec_module(bi)
    G = bi.compute_gauss_residual(E, U, graph)         # (V, 3)
    norms = torch.linalg.norm(G, dim=-1)               # (V,)
    return [float(x) for x in norms.detach().cpu().tolist()]


def edge_face_types(graph) -> List[int]:
    """Per-edge boundary type: 0 = pent-hex, 1 = hex-hex.

    The truncated icosahedron has 60 pent-hex + 30 hex-hex + 0 pent-pent
    edges (WF#1 census).  Each edge sits in exactly two faces; we classify
    by the (pent, hex) signature of those two faces.  A pent-pent edge
    would raise ValueError; on the validated buckyball this never fires.
    """
    pent_set = set(graph.pentagons)
    hex_set = set(graph.hexagons)
    per_edge_faces: List[List[str]] = [[] for _ in range(graph.n_edges)]
    for fi, face in enumerate(graph.faces):
        tag = "P" if fi in pent_set else ("H" if fi in hex_set else "?")
        for eidx, _sign in face:
            per_edge_faces[eidx].append(tag)
    out: List[int] = []
    for e, tags in enumerate(per_edge_faces):
        if len(tags) != 2:
            raise ValueError(
                f"edge {e} touches {len(tags)} faces (expected 2); tags={tags}"
            )
        key = "".join(sorted(tags))
        if key == "HP":
            out.append(0)
        elif key == "HH":
            out.append(1)
        else:
            raise ValueError(
                f"edge {e} has unsupported face-type signature {key!r}; "
                f"buckyball should have no pent-pent edges"
            )
    return out


# ----------------------------------------------------------------------
# Helpers for the trajectory dumper
# ----------------------------------------------------------------------
def _face_record(graph, face_idx: int) -> Dict[str, Any]:
    cyc = graph.face_vertices[face_idx]
    is_pent = face_idx in set(graph.pentagons)
    return {
        "vertices": [int(v) for v in cyc],
        "type": "pent" if is_pent else "hex",
    }


def _geometry_block(graph) -> Dict[str, Any]:
    return {
        "vertex_coords": [
            [float(graph.vertex_coords[v, 0]),
             float(graph.vertex_coords[v, 1]),
             float(graph.vertex_coords[v, 2])]
            for v in range(graph.n_vertices)
        ],
        "edges": [
            [int(graph.edges[e, 0]), int(graph.edges[e, 1])]
            for e in range(graph.n_edges)
        ],
        "faces": [_face_record(graph, f) for f in range(graph.n_faces)],
    }


def _plaquette_mean_from_U(U: torch.Tensor, graph) -> float:
    Uf = all_face_holonomies(U, graph)
    return float(Uf[:, 0].mean().item())


def _build_frame(
    *,
    t: float,
    step: int,
    U: torch.Tensor,
    E: torch.Tensor,
    graph,
    energy: Optional[float],
    control: Optional[Dict[str, Any]],
    edge_face_types_cache: Optional[List[int]] = None,
) -> Dict[str, Any]:
    phases = edge_phase(U).detach().cpu().numpy()
    kinetic = edge_kinetic(E).detach().cpu().numpy()
    # v0.2 additive drill-down fields (per frame, keyed alongside the
    # existing scalars and per-edge arrays).
    f_q0 = face_q0_values(U, graph)
    f_act = [float(1.0 - x) for x in f_q0]          # avoids re-computing holonomies
    v_gauss = vertex_gauss_norms(E, U, graph)
    if edge_face_types_cache is None:
        e_ft = edge_face_types(graph)
    else:
        e_ft = list(edge_face_types_cache)
    frame: Dict[str, Any] = {
        "t": float(t),
        "step": int(step),
        "Q": Q_surrogate(U, graph),
        "plaquette_mean": _plaquette_mean_from_U(U, graph),
        "edge_phases": [float(x) for x in phases.tolist()],
        "edge_kinetic": [float(x) for x in kinetic.tolist()],
        "control": dict(control) if control else {},
        "face_actions": f_act,
        "face_q0": f_q0,
        "vertex_gauss": v_gauss,
        "edge_face_types": e_ft,
    }
    if energy is not None:
        frame["energy"] = float(energy)
    return frame


def dump_trajectory(
    U_traj: List[torch.Tensor],
    E_traj: List[torch.Tensor],
    *,
    graph,
    times: List[float],
    steps: List[int],
    out_path: str,
    energy_history: Optional[List[float]] = None,
    control_history: Optional[List[Dict[str, Any]]] = None,
    gauge_group: str = "SU(2)",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a trajectory.json file conforming to the WF#3 schema.

    Args:
        U_traj         : list of (n_edges, 4) tensors, one per frame.
        E_traj         : list of (n_edges, 4) tensors, one per frame.
        graph          : BuckyballGraph.
        times          : list of float, simulation time per frame.
        steps          : list of int, step index per frame.
        out_path       : absolute path to the JSON file to write.
        energy_history : optional list of floats, total H per frame.
        control_history: optional list of dicts, control state per frame.
        gauge_group    : string label, default "SU(2)".
        extra_meta     : optional dict merged into top-level (e.g. "notes").

    Returns the dict that was written (also persisted to out_path).
    """
    n = len(U_traj)
    if len(E_traj) != n or len(times) != n or len(steps) != n:
        raise ValueError(
            f"length mismatch: U_traj={n} E_traj={len(E_traj)} "
            f"times={len(times)} steps={len(steps)}"
        )
    if energy_history is not None and len(energy_history) != n:
        raise ValueError(f"energy_history length {len(energy_history)} != {n}")
    if control_history is not None and len(control_history) != n:
        raise ValueError(f"control_history length {len(control_history)} != {n}")

    # Edge face-type assignment is geometry-only; compute once and reuse
    # for every frame.  We still write it per-frame per the v0.2 brief
    # ("per-frame for simplicity") so the viewer can drill in without
    # cross-referencing a top-level block.
    edge_ft_cache = edge_face_types(graph)

    frames: List[Dict[str, Any]] = []
    for i in range(n):
        energy_i = energy_history[i] if energy_history is not None else None
        control_i = control_history[i] if control_history is not None else None
        frames.append(_build_frame(
            t=times[i],
            step=steps[i],
            U=U_traj[i],
            E=E_traj[i],
            graph=graph,
            energy=energy_i,
            control=control_i,
            edge_face_types_cache=edge_ft_cache,
        ))

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gauge_group": gauge_group,
        "lattice": {
            "vertices": int(graph.n_vertices),
            "edges": int(graph.n_edges),
            "faces": int(graph.n_faces),
        },
        "geometry": _geometry_block(graph),
        "frames": frames,
    }
    if extra_meta:
        for k, v in extra_meta.items():
            if k in payload:
                raise ValueError(f"extra_meta key {k!r} clashes with payload")
            payload[k] = v

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return payload


# ----------------------------------------------------------------------
# Trajectory loader + schema validator
# ----------------------------------------------------------------------
def load_trajectory(path: str) -> Dict[str, Any]:
    """Read back a trajectory.json file and validate the WF#3 schema.

    Checks performed:
      * schema_version is present and equal to a known value.
      * gauge_group, lattice, geometry, frames keys are present.
      * lattice.{vertices, edges, faces} are positive ints.
      * geometry.vertex_coords is a (V, 3) nested list of floats.
      * geometry.edges is an (E, 2) nested list of ints.
      * geometry.faces is a list of F dicts, each with a "vertices" list and
        a "type" in {"pent", "hex"}.
      * frames is a non-empty list; each frame has the required scalar keys
        and per-edge arrays of length n_edges.

    Returns the decoded payload dict (same object that dump_trajectory
    wrote).  Raises ValueError on any schema violation; lets json.JSONDecodeError
    propagate for malformed JSON.
    """
    abs_path = os.path.abspath(path)
    with open(abs_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError(f"top-level payload must be a JSON object; got {type(payload).__name__}")

    known_versions = set(_KNOWN_SCHEMA_VERSIONS)
    sv = payload.get("schema_version")
    if sv not in known_versions:
        raise ValueError(f"unknown schema_version {sv!r}; expected one of {sorted(known_versions)}")

    for k in ("gauge_group", "lattice", "geometry", "frames"):
        if k not in payload:
            raise ValueError(f"missing required top-level key {k!r}")

    lat = payload["lattice"]
    if not isinstance(lat, dict):
        raise ValueError(f"lattice block must be an object; got {type(lat).__name__}")
    for k in ("vertices", "edges", "faces"):
        if k not in lat or not isinstance(lat[k], int) or lat[k] <= 0:
            raise ValueError(f"lattice.{k} must be a positive int; got {lat.get(k)!r}")
    V, E, F = int(lat["vertices"]), int(lat["edges"]), int(lat["faces"])

    geom = payload["geometry"]
    if not isinstance(geom, dict):
        raise ValueError(f"geometry block must be an object; got {type(geom).__name__}")
    vc = geom.get("vertex_coords")
    if not (isinstance(vc, list) and len(vc) == V):
        raise ValueError(f"geometry.vertex_coords must be a list of {V} entries; got len {len(vc) if isinstance(vc, list) else 'N/A'}")
    for i, row in enumerate(vc):
        if not (isinstance(row, list) and len(row) == 3):
            raise ValueError(f"geometry.vertex_coords[{i}] must be a 3-list; got {row!r}")
    geom_edges = geom.get("edges")
    if not (isinstance(geom_edges, list) and len(geom_edges) == E):
        raise ValueError(f"geometry.edges must be a list of {E} entries; got len {len(geom_edges) if isinstance(geom_edges, list) else 'N/A'}")
    for i, row in enumerate(geom_edges):
        if not (isinstance(row, list) and len(row) == 2):
            raise ValueError(f"geometry.edges[{i}] must be a 2-list; got {row!r}")
    geom_faces = geom.get("faces")
    if not (isinstance(geom_faces, list) and len(geom_faces) == F):
        raise ValueError(f"geometry.faces must be a list of {F} entries; got len {len(geom_faces) if isinstance(geom_faces, list) else 'N/A'}")
    for i, fr in enumerate(geom_faces):
        if not isinstance(fr, dict):
            raise ValueError(f"geometry.faces[{i}] must be an object; got {fr!r}")
        if not isinstance(fr.get("vertices"), list):
            raise ValueError(f"geometry.faces[{i}].vertices must be a list; got {fr.get('vertices')!r}")
        if fr.get("type") not in ("pent", "hex"):
            raise ValueError(f"geometry.faces[{i}].type must be 'pent' or 'hex'; got {fr.get('type')!r}")

    frames = payload["frames"]
    if not (isinstance(frames, list) and len(frames) > 0):
        raise ValueError("frames must be a non-empty list")
    required_frame_keys = ("t", "step", "Q", "plaquette_mean", "edge_phases", "edge_kinetic", "control")
    # v0.2 introduces four additional per-frame fields.  They are REQUIRED
    # in v0.2 and ABSENT in v0.1; mixing is rejected per frame.
    v02_required_keys: Tuple[Tuple[str, int], ...] = (
        ("face_actions", F),
        ("face_q0", F),
        ("vertex_gauss", V),
        ("edge_face_types", E),
    )
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{i}] must be an object; got {frame!r}")
        for k in required_frame_keys:
            if k not in frame:
                raise ValueError(f"frames[{i}] missing required key {k!r}")
        ep = frame["edge_phases"]
        ek = frame["edge_kinetic"]
        if not (isinstance(ep, list) and len(ep) == E):
            raise ValueError(f"frames[{i}].edge_phases must have length {E}; got {len(ep) if isinstance(ep, list) else 'N/A'}")
        if not (isinstance(ek, list) and len(ek) == E):
            raise ValueError(f"frames[{i}].edge_kinetic must have length {E}; got {len(ek) if isinstance(ek, list) else 'N/A'}")
        if sv == "0.2":
            for key, expected_len in v02_required_keys:
                if key not in frame:
                    raise ValueError(
                        f"frames[{i}] missing v0.2 key {key!r}"
                    )
                val = frame[key]
                if not (isinstance(val, list) and len(val) == expected_len):
                    raise ValueError(
                        f"frames[{i}].{key} must have length {expected_len}; "
                        f"got {len(val) if isinstance(val, list) else 'N/A'}"
                    )
        else:
            # v0.1: tolerate absence of the v0.2 fields.  Do not error if
            # an upstream tool happens to carry them in a hybrid file --
            # forward compatibility within the same major version.
            pass

    return payload


# ----------------------------------------------------------------------
# Integration wrapper that keeps every measured (U, E) snapshot.
# We do NOT modify buckyball_integrator.py; we re-call leapfrog_step
# directly and harvest state at each measurement window.
# ----------------------------------------------------------------------
def integrate_with_states(
    U_init: torch.Tensor,
    E_init: torch.Tensor,
    *,
    dt: float,
    n_steps: int,
    graph,
    beta: float,
    measure_every: int = 20,
    include_initial: bool = True,
) -> Dict[str, Any]:
    """Run leapfrog and harvest a (U, E) snapshot every measure_every steps.

    This is a thin wrapper around the validated leapfrog_step + compute_hamiltonian
    routines from buckyball_integrator (no kernel modification).  It exists
    because buckyball_integrator.integrate() returns only the final state plus
    scalar histories; the trajectory dumper needs the full (U, E) at each
    measurement window.

    Args:
        U_init        : (n_edges, 4) initial link tensor.
        E_init        : (n_edges, 4) initial canonical-momentum tensor.
        dt            : leapfrog step size.
        n_steps       : total number of leapfrog steps.
        graph         : BuckyballGraph.
        beta          : Wilson coupling (group conventions in integrator).
        measure_every : harvest a snapshot every k steps (and at n_steps).
        include_initial : if True, also include the (U_init, E_init) state
                          at step=0, t=0.

    Returns a dict:
        U_traj      : list[Tensor (n_edges, 4)]
        E_traj      : list[Tensor (n_edges, 4)]
        times       : list[float]
        steps       : list[int]
        H_history   : list[float] (total Hamiltonian per snapshot)
        K_history   : list[float] (kinetic per snapshot)
        V_history   : list[float] (potential per snapshot)
        G_history   : list[float] (max|G_v|_inf per snapshot; covariant
                      Gauss residual sampled at the same (U, E) state used
                      for H/K/V. Additive extension for the validation
                      report's trajectory-wise Gauss check.)
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0; got {dt!r}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0; got {n_steps!r}")
    if measure_every < 1:
        raise ValueError(f"measure_every must be >= 1; got {measure_every!r}")

    # Lazy-import the validated integrator (no top-level coupling).
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "_buckyball_obs_integrator",
        os.path.join(_HERE, "buckyball_integrator.py"),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load buckyball_integrator.py")
    bi = _ilu.module_from_spec(spec)
    spec.loader.exec_module(bi)

    U = U_init.clone()
    E = E_init.clone()

    U_traj: List[torch.Tensor] = []
    E_traj: List[torch.Tensor] = []
    times: List[float] = []
    steps: List[int] = []
    H_hist: List[float] = []
    K_hist: List[float] = []
    V_hist: List[float] = []
    G_hist: List[float] = []

    def _snapshot(step_i: int) -> None:
        H_i, K_i, V_i = bi.compute_hamiltonian(U, E, graph, beta)
        # Trajectory-wise covariant Gauss residual at the SAME (U, E) state
        # used for the Hamiltonian readout. The kernel's
        # compute_gauss_residual is pure; this sampling adds O(n_edges * 16
        # FLOPs) per snapshot and is bit-identical across runs for the same
        # inputs.
        G_i = bi.compute_gauss_residual(E, U, graph)
        U_traj.append(U.detach().clone())
        E_traj.append(E.detach().clone())
        times.append(step_i * dt)
        steps.append(step_i)
        H_hist.append(float(H_i))
        K_hist.append(float(K_i))
        V_hist.append(float(V_i))
        G_hist.append(float(G_i.abs().max()))

    if include_initial:
        _snapshot(0)

    for s in range(1, n_steps + 1):
        U, E = bi.leapfrog_step(U, E, dt, graph, beta)
        if (s % measure_every == 0) or (s == n_steps):
            _snapshot(s)

    return {
        "U_traj": U_traj,
        "E_traj": E_traj,
        "times": times,
        "steps": steps,
        "H_history": H_hist,
        "K_history": K_hist,
        "V_history": V_hist,
        "G_history": G_hist,
        "U_final": U,
        "E_final": E,
        "dt": dt,
        "beta": beta,
        "n_steps": n_steps,
        "measure_every": measure_every,
    }


# ----------------------------------------------------------------------
# Gauge transformation helper (used in the sanity tests, not on the
# critical path).  Applies U_e -> g(v_from) U_e g(v_to)^dag.
# ----------------------------------------------------------------------
def apply_gauge_transformation(U: torch.Tensor, g: torch.Tensor, graph) -> torch.Tensor:
    """Apply a vertex gauge transformation to link variables.

    g has shape (n_vertices, 4), each row a unit quaternion.
    Returns a new (n_edges, 4) tensor with
        U'_e = g(v_from) . U_e . qconj(g(v_to))
    where (v_from, v_to) = graph.edges[e].
    """
    if g.shape != (graph.n_vertices, 4):
        raise ValueError(f"g must be ({graph.n_vertices}, 4); got {tuple(g.shape)}")
    v_from = torch.as_tensor(graph.edges[:, 0], dtype=torch.long)
    v_to = torch.as_tensor(graph.edges[:, 1], dtype=torch.long)
    g_from = g[v_from]                # (n_edges, 4)
    g_to_dag = _qconj(g[v_to])        # (n_edges, 4)
    left = _qmul(g_from, U)
    return _qmul(left, g_to_dag)


__all__ = [
    "SCHEMA_VERSION",
    "q_face_surrogate",
    "q_face_surrogate_all",
    "Q_surrogate",
    "edge_phase",
    "edge_kinetic",
    "face_q0_values",
    "face_actions",
    "vertex_gauss_norms",
    "edge_face_types",
    "dump_trajectory",
    "load_trajectory",
    "integrate_with_states",
    "apply_gauge_transformation",
]
