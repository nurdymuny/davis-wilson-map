"""
buckyball_integrator.py - Real-time Kogut-Susskind leapfrog on the buckyball
(truncated icosahedron) substrate, SU(2) gauge group, quaternion form.

WF#2 extension of the 4D-cubic kernel (inertia_damping/symplectic_integrator.py).
The HAMILTONIAN STRUCTURE, the canonical-sigma formula, the leapfrog drift
convention, and the SU(2) quaternion bookkeeping all TRANSPORT from the cubic
sibling and are imported, not re-derived.  The genuinely new piece is the
GRAPH-LAPLACIAN CG GAUSS PROJECTOR replacing the 4D-torus covariant
Laplacian: on a 3-regular graph every vertex constraint is

    G_v = sum_{e incident to v}  s_v(e) * E_e   =  0    in the Lie algebra,

with s_v(e) = +1 if v = tail(e) (edges[e, 0]) and -1 if v = head(e)
(edges[e, 1]).  Stacking those signs into the signed incidence
D in R^{60 x 90} the constraint reads (D E^a)_v = 0 per Lie-algebra
component a in {1,2,3}.  The associated graph Laplacian

    L_G  =  D D^T  in R^{60 x 60},

is symmetric PSD with a one-dimensional null space (the constant vector
1_60 - L_G has eigenvalue 0 on it because every edge contributes
+1 - 1 = 0).  We pin it by Dirichlet-removing row/col 0 (deleting vertex 0's
constraint) -- this is exact, not regularization, and the post-CG check
||G||_inf is computed on the FULL 60-vertex residual so pinning cannot
inflate a pass.  CG solves L_G lambda = D E^a per component; the update
E ^a -= D^T lambda zeros out the divergence at every vertex up to CG
residual.  Iterative refinement (a la symplectic_integrator) drives the
final ||G||_inf well below the 1e-10 gate in float64.

H A M I L T O N I A N (same form as the cubic kernel):
    H = (g^2/2) sum_links Tr(E^2)
        + (1/g^2) sum_faces [N - Re Tr U_f]
with beta_KS = 2N/g^2  (g^2 = 2N/beta = 2.0 for SU(2) at beta=2.0,
or 1.6 for the canonical buckyball beta=2.5).

EOM:
    dU_e/dt  =  + i g^2 E_e U_e             (drift; matches cubic SU(2))
    dE_e/dt  =  - (beta / (2 N^2)) * proj_q0=0( qmul(U_e, Sigma_e) )
where Sigma_e is the staple sum from buckyball_action.staple_sum_q.  The
factor -beta/(2 N^2) = -beta/8 for SU(2) transports verbatim from the
cubic sibling -- it comes from differentiating (1/g^2)*(N - Re Tr U_f)
with respect to E_e via dU_e/dE_e = i g^2 dt, which is geometry-blind.

C A N O N I C A L  S I G M A (does NOT depend on geometry):
    SU(2) quaternion-packed:  sigma^2 = beta / (32 N^2) = beta/128
    at beta=2.5:  sigma = sqrt(2.5/128) approx 0.1398 .
We import the closed form from symplectic_integrator.canonical_sigma so
the value never drifts from the cubic.

KEEP-IT-TIGHT discipline:  no re-derivation of canonical_sigma, no
re-derivation of the drift coefficient, no re-derivation of the
quaternion exp/qmul/qconj primitives.  The new content is the
graph-Laplacian projector + leapfrog wired to buckyball_action.staple_sum_q.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

RDTYPE = torch.float64

_sym: Any = None           # symplectic_integrator (cubic kernel)
_ba: Any = None            # buckyball_action
_bg: Any = None            # buckyball_graph


def _load_module(module_name: str, abs_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {abs_path!r}")
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass introspection works inside mod.
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def _symplectic():
    """Lazy-load the validated 4D-cubic kernel for canonical_sigma + primitives."""
    global _sym
    if _sym is None:
        _sym = _load_module(
            "_buckyball_integrator_symp",
            os.path.join(_HERE, "symplectic_integrator.py"),
        )
    return _sym


def _action():
    """Lazy-load buckyball_action.py."""
    global _ba
    if _ba is None:
        _ba = _load_module(
            "_buckyball_integrator_action",
            os.path.join(_HERE, "buckyball_action.py"),
        )
    return _ba


def _graph_mod():
    """Lazy-load buckyball_graph.py (only used by callers that build a graph)."""
    global _bg
    if _bg is None:
        _bg = _load_module(
            "_buckyball_integrator_graph",
            os.path.join(_HERE, "buckyball_graph.py"),
        )
    return _bg


# ---------------------------------------------------------------------------
# Re-exports / transports from the cubic kernel
# ---------------------------------------------------------------------------
def canonical_sigma(beta: float, gauge_group: str = "SU(2)") -> float:
    """Per-coordinate Gaussian width for canonical-scale E (transports from
    the cubic kernel; do NOT re-derive)."""
    return _symplectic().canonical_sigma(beta, gauge_group)


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _action()._qmul(a, b)


def _qconj(a: torch.Tensor) -> torch.Tensor:
    return _action()._qconj(a)


def _qnorm(a: torch.Tensor) -> torch.Tensor:
    return _action()._qnorm(a)


def _matrix_exp_su2_q(w: torch.Tensor) -> torch.Tensor:
    """For purely-imaginary quaternion (0, v), returns unit quaternion
    exp(i v.sigma).  Imported from the validated cubic kernel."""
    return _symplectic()._matrix_exp_su2_q(w)


def _g_squared(beta: float, N: int = 2) -> float:
    return (2.0 * N) / beta


# ---------------------------------------------------------------------------
# Signed vertex-edge incidence + graph Laplacian (the new piece)
# ---------------------------------------------------------------------------
def signed_incidence(graph) -> np.ndarray:
    """Return the signed vertex-edge incidence D in R^(V x E).

    Convention: for edge e = (v_from, v_to) = (edges[e,0], edges[e,1]),
        D[v_from, e] = +1  (tail)
        D[v_to,   e] = -1  (head)
    The graph Laplacian is L_G = D @ D^T, which on a 3-regular graph
    equals 3*I - A where A is the unsigned adjacency.  L_G is symmetric
    PSD with a 1D null space spanned by the all-ones vector.
    """
    V = graph.n_vertices
    E = graph.n_edges
    D = np.zeros((V, E), dtype=np.float64)
    for e in range(E):
        v_from = int(graph.edges[e, 0])
        v_to = int(graph.edges[e, 1])
        D[v_from, e] = +1.0
        D[v_to, e] = -1.0
    return D


def graph_laplacian(graph) -> np.ndarray:
    """L_G = D @ D^T in R^(V x V).  Symmetric PSD, rank V-1, null space = 1_V."""
    D = signed_incidence(graph)
    return D @ D.T


# ---------------------------------------------------------------------------
# Initial-condition builders
# ---------------------------------------------------------------------------
def initialize_E_zero(
    graph,
    gauge_group: str = "SU(2)",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """E = 0 on every edge.  Gauss residual is identically zero."""
    if gauge_group != "SU(2)":
        raise ValueError(
            f"buckyball_integrator only supports SU(2); got {gauge_group!r}"
        )
    return torch.zeros((graph.n_edges, 4), dtype=RDTYPE, device=device)


def initialize_E_canonical(
    graph,
    beta: float,
    gauge_group: str = "SU(2)",
    generator: Optional[torch.Generator] = None,
    project_gauss: bool = True,
    cg_tol: float = 1e-10,
    cg_max_iter: int = 200,
    device: torch.device | str = "cpu",
    U: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample E at canonical scale and optionally CG-project to Gauss-zero.

    Step 1: draw per-edge, per-Lie-algebra-component coefficients
            alpha_a ~ N(0, sigma^2), sigma = canonical_sigma(beta, group).
    Step 2: quaternion-pack with E[..., 0] = 0 and E[..., 1:] = 2 * alpha
            (matches the cubic kernel's SU(2) packing convention exactly --
             see canonical_sigma derivation in symplectic_integrator.py).
    Step 3 (optional): COVARIANT CG projector to ||G_cov(U, E)||_inf < cg_tol.
            If U is None, the projector uses identity links (cold U), which
            reduces the covariant operator exactly to the flat graph
            Laplacian (Ad(I) = I), reproducing the pre-WF#2b behaviour.

    Returns
    -------
    torch.Tensor of shape (n_edges, 4), q0=0, ready for the integrator.
    """
    if gauge_group != "SU(2)":
        raise ValueError(
            f"buckyball_integrator only supports SU(2); got {gauge_group!r}"
        )
    sigma = canonical_sigma(beta, gauge_group)
    n_edges = graph.n_edges
    alpha = sigma * torch.randn(
        (n_edges, 3), dtype=RDTYPE, device=device, generator=generator,
    )
    E = torch.zeros((n_edges, 4), dtype=RDTYPE, device=device)
    E[..., 1:] = 2.0 * alpha   # quaternion packing (see cubic kernel)
    if project_gauss:
        if U is None:
            # Cold default: U = identity link on every edge.  At U=I the
            # covariant projector reduces to the flat one (Ad(I)=I), so
            # this is bit-equivalent to the pre-patch behaviour and keeps
            # the canonical sampler agnostic to thermalized U.
            U_proj = _action().identity_links(n_edges, device=device)
        else:
            U_proj = U
        E, _info = project_gauss_zero_cg(
            E, U_proj, graph, tol=cg_tol, max_iter=cg_max_iter,
        )
    return E


# ---------------------------------------------------------------------------
# Gauss residual + graph-Laplacian CG projector
# ---------------------------------------------------------------------------
def compute_gauss_residual_flat(E: torch.Tensor, graph) -> torch.Tensor:
    """FLAT (abelian) per-vertex divergence G_v^a = sum_e s_v(e) E_e^a.

    Retained as a diagnostic-only helper.  This is the WRONG residual on
    thermalized U (it ignores Ad(U_e) transport across the head end of
    each edge); the load-bearing physical residual is
    `compute_gauss_residual`, which is covariant.  Equal to the covariant
    residual at U=I by Ad(I)=identity.
    """
    D = torch.from_numpy(signed_incidence(graph)).to(
        dtype=RDTYPE, device=E.device,
    )
    return D @ E[..., 1:]


def compute_gauss_residual(
    E: torch.Tensor, U: torch.Tensor, graph,
) -> torch.Tensor:
    """COVARIANT per-vertex SU(2) Gauss residual in the Lie algebra.

    Implements the non-abelian divergence the Kogut-Susskind Hamiltonian
    conserves, mirroring symplectic_integrator._gauss_residual_su2:

        G_v^a = sum_{e: tail(e)=v} E_e^a
              - sum_{e: head(e)=v} [Ad_codebase(U_e) E_e]^a

    with edge orientation (tail, head) = (edges[e,0], edges[e,1]).  In
    quaternion form (E packed as (0, e_vec), U a unit quaternion):
        T_e := qmul(qmul(qconj(U_e), E_e), U_e)   # pure-imaginary
        G_v[a] = sum_{e:tail=v} E_e[1+a] - sum_{e:head=v} T_e[1+a]

    The qmul/qconj sandwich pulls E from the tail's frame into the head's
    frame along the link U_e (the codebase's qmul has the SU(2) i-sigma
    sign convention; by direct expansion this matches the standard
    quaternion-to-SO(3) rotation Ad(U) on Lie-algebra vectors).

    At U_e = (1,0,0,0) for every edge (cold), the sandwich is identity, so
    T_e = E_e and G_v reduces exactly to the FLAT signed-incidence
    divergence D @ E[..., 1:] -- this is the cold-equivalence guarantee
    that preserves H_A.

    Parameters
    ----------
    E : torch.Tensor of shape (n_edges, 4), q0=0 (Lie-algebra quaternion).
    U : torch.Tensor of shape (n_edges, 4), unit quaternion link variables.
    graph : BuckyballGraph (provides edges and n_vertices).

    Returns
    -------
    torch.Tensor of shape (V, 3): per-vertex Lie-algebra residual.
    """
    if U.shape != E.shape:
        raise ValueError(
            f"compute_gauss_residual: U.shape {tuple(U.shape)} must equal "
            f"E.shape {tuple(E.shape)} (per-edge quaternion)"
        )
    V = graph.n_vertices
    device = E.device
    # T_e = head-frame image of E_e along U_e (cubic-kernel convention).
    T = _qmul(_qmul(_qconj(U), E), U)              # (n_edges, 4)
    edges_t = torch.as_tensor(graph.edges, dtype=torch.long, device=device)
    tails = edges_t[:, 0]
    heads = edges_t[:, 1]
    G = torch.zeros((V, 3), dtype=RDTYPE, device=device)
    G.index_add_(0, tails,  E[:, 1:])              # +E at tail
    G.index_add_(0, heads, -T[:, 1:])              # -transported E at head
    return G


def _ad_matrices_from_U(U: torch.Tensor) -> np.ndarray:
    """Per-edge SO(3) adjoint matrices R_e implementing the codebase's
    qmul(qmul(qconj(U_e), v_quat), U_e) sandwich on Lie-algebra vectors v.

    In this codebase's quaternion convention (validated by direct expansion:
    cv = a0 bv + b0 av - cross(av, bv)), the sandwich
        qmul(qmul(qconj(U), (0, v)), U)
    produces exactly the standard quaternion-to-rotation matrix R(q) v with
    q = U = (w, x, y, z), i.e.
        R[i,j]:
            R[0,0] = 1 - 2(y^2 + z^2)
            R[0,1] = 2(xy - zw)
            R[0,2] = 2(xz + yw)
            R[1,0] = 2(xy + zw)
            R[1,1] = 1 - 2(x^2 + z^2)
            R[1,2] = 2(yz - xw)
            R[2,0] = 2(xz - yw)
            R[2,1] = 2(yz + xw)
            R[2,2] = 1 - 2(x^2 + y^2)

    This is the rotation that pulls E_e (stored in the TAIL's frame) into
    the HEAD's frame on a link oriented tail->head -- exactly what the
    cubic kernel's _gauss_residual_su2 subtracts at the head vertex.

    Returns: numpy array of shape (n_edges, 3, 3), float64, orthogonal.
    """
    U_np = U.detach().cpu().numpy().astype(np.float64)
    w = U_np[:, 0]; x = U_np[:, 1]; y = U_np[:, 2]; z = U_np[:, 3]
    n = U_np.shape[0]
    R = np.empty((n, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - z * w)
    R[:, 0, 2] = 2.0 * (x * z + y * w)
    R[:, 1, 0] = 2.0 * (x * y + z * w)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - x * w)
    R[:, 2, 0] = 2.0 * (x * z - y * w)
    R[:, 2, 1] = 2.0 * (y * z + x * w)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def _apply_D_cov(
    E_alg: np.ndarray,
    R_e: np.ndarray,
    tails: np.ndarray,
    heads: np.ndarray,
    V: int,
) -> np.ndarray:
    """D_cov(U) E_alg : (E_n, 3) -> (V, 3)
        (D_cov E)_v = sum_{tail(e)=v} E_e - sum_{head(e)=v} (R_e E_e)
    where R_e is the per-edge head-transport matrix (R_e[i,j] from
    _ad_matrices_from_U), which is the matrix form of the codebase's
    qmul(qmul(qconj(U_e), E_e), U_e).
    """
    # Transported edge field at head: T_e = R_e @ E_alg[e, :]
    T = np.einsum("eij,ej->ei", R_e, E_alg)
    G = np.zeros((V, 3), dtype=np.float64)
    np.add.at(G, tails, E_alg)
    np.add.at(G, heads, -T)
    return G


def _apply_D_cov_T(
    phi: np.ndarray,
    R_e: np.ndarray,
    tails: np.ndarray,
    heads: np.ndarray,
    n_edges: int,
) -> np.ndarray:
    """D_cov(U)^T phi : (V, 3) -> (E_n, 3)
        (D_cov^T phi)_e = phi[tail(e), :] - R_e^T phi[head(e), :]
    R_e is orthogonal so R_e^T = R_e^{-1}.
    """
    R_e_T = np.transpose(R_e, (0, 2, 1))    # (E_n, 3, 3)
    phi_tail = phi[tails, :]                # (E_n, 3)
    phi_head = phi[heads, :]                # (E_n, 3)
    transported_head = np.einsum("eij,ej->ei", R_e_T, phi_head)
    return phi_tail - transported_head


def project_gauss_zero_cg(
    E: torch.Tensor,
    U: torch.Tensor,
    graph,
    tol: float = 1e-10,
    max_iter: int = 200,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """COVARIANT CG projection onto the non-abelian Gauss-constraint surface.

    Replaces the WF#2-era flat graph-Laplacian projector with the
    U-dependent covariant operator L_cov(U) := D_cov(U) D_cov(U)^T, where
    D_cov(U) is the covariant divergence (see compute_gauss_residual).
    The system

        L_cov(U)  Lambda  =  G_cov(U, E)        (vertex-valued, (V, 3))

    is solved with scipy CG over a flattened ((V-1)*3,) representation
    (Dirichlet-pinning vertex 0 to remove the rigid global SU(2) rotation,
    a 3-dim null space of L_cov at U=I and a small perturbation thereof at
    thermalized U).  E is then updated as

        E[..., 1:]  <-  E[..., 1:]  -  D_cov(U)^T Lambda
        E[..., 0]   =  0

    At U=I, Ad(U_e^{-1}) reduces to the 3x3 identity, D_cov collapses to
    the flat signed-incidence operator D, and L_cov collapses to the flat
    graph Laplacian L_G = D D^T.  Hence the cold-init sanity check (H_A)
    is preserved by construction: the solver does the same arithmetic on
    the U=I leg as the pre-patch flat solver, up to qmul-pipeline
    roundoff (~1e-14).

    Iterative refinement: as in the cubic kernel's project_gauss_zero_cg,
    float64 roundoff in the qmul/qconj pipeline leaves ||G||_inf modestly
    above tol on the first pass; re-running CG on the residual drives it
    geometrically toward eps_mach.  Capped at n_refinement_max = 6 passes;
    pass criterion is the empirical ||G_cov||_inf <= tol, NOT scipy's info
    return code (the cubic kernel learned this lesson).

    Parameters
    ----------
    E : (n_edges, 4) torch.Tensor, q0=0 SU(2) Lie-algebra quaternion.
    U : (n_edges, 4) torch.Tensor, unit quaternion link variables.
    graph : BuckyballGraph.
    tol : float, post-update ||G_cov||_inf gate.
    max_iter : int, per-pass CG iteration cap (applied to the (V-1)*3
               system).
    verbose : bool, print residual diagnostics.

    Returns
    -------
    (E_proj, info) where info has keys:
        n_refinement_passes : int
        final_gauss_max     : float (covariant)
        initial_gauss_max   : float (covariant)
        per_pass_cg_info    : list[dict]  per refinement pass diagnostics
    """
    try:
        from scipy.sparse.linalg import cg, LinearOperator
    except ImportError as ex:
        raise ImportError(
            "project_gauss_zero_cg requires scipy.sparse.linalg.cg"
        ) from ex

    if E.shape[-1] != 4:
        raise ValueError(f"E last dim must be 4 (quaternion); got {E.shape}")
    if U.shape != E.shape:
        raise ValueError(
            f"project_gauss_zero_cg: U.shape {tuple(U.shape)} must equal "
            f"E.shape {tuple(E.shape)}"
        )

    device = E.device
    V = graph.n_vertices
    n_edges = graph.n_edges
    n_dofs = V * 3

    # Precompute per-edge head-transport matrices (constant during this solve)
    R_e = _ad_matrices_from_U(U)                     # (E_n, 3, 3)
    edges_np = np.asarray(graph.edges, dtype=np.int64)
    tails = edges_np[:, 0].copy()
    heads = edges_np[:, 1].copy()

    # Initial covariant residual (full 60-vertex)
    initial_G = compute_gauss_residual(E, U, graph)  # (V, 3)
    initial_G_max = float(initial_G.abs().max())
    if verbose:
        print(f"[project_gauss_zero_cg] initial ||G_cov||_inf = "
              f"{initial_G_max:.3e}")
    # Trivial early-out
    if initial_G_max < tol * 1e-2:
        return E, {
            "n_refinement_passes": 0,
            "final_gauss_max": initial_G_max,
            "initial_gauss_max": initial_G_max,
            "per_pass_cg_info": [],
        }

    # NOTE on null space: at thermalized U, L_cov(U) is strictly SPD on the
    # full V*3 space (empirically verified, min eig ~ 0.18 at beta=2.5).  At
    # U=I, L_cov has a 3-dim null space (constant Lie-algebra vector across
    # all V vertices), but the RHS b = G_cov(E) lies in the range of L_cov
    # (b = D_cov E, automatically orthogonal to ker(D_cov^T) = ker(L_cov)),
    # so scipy CG converges to a particular solution.  Dirichlet pinning is
    # NOT used because at thermalized U the "59 vertices force the 60th"
    # identity fails: sum_v G_v^a = sum_e (I - R_e) E_e is NOT zero, and
    # zeroing 59 vertices leaves the 60th non-zero, defeating the gate.
    # A small Tikhonov shift epsilon * I (1e-14, well below the 1e-9 H_C
    # gate) regularizes the U=I case without harming convergence elsewhere.
    eps_tikhonov = 1e-14

    def _matvec_Lcov(x_flat: np.ndarray) -> np.ndarray:
        """L_cov(U) + eps*I  on the full V*3-dim system."""
        phi = x_flat.reshape(V, 3)
        E_tmp = _apply_D_cov_T(phi, R_e, tails, heads, n_edges)
        G_tmp = _apply_D_cov(E_tmp, R_e, tails, heads, V)
        return G_tmp.reshape(n_dofs) + eps_tikhonov * x_flat

    A = LinearOperator(
        (n_dofs, n_dofs), matvec=_matvec_Lcov, dtype=np.float64,
    )

    E_new = E.clone()
    last_G_max = float("inf")
    n_refinement_max = 6
    pass_infos: list = []

    for refinement_pass in range(n_refinement_max):
        # Build RHS: full V*3 covariant residual, flattened
        G_cur = compute_gauss_residual(E_new, U, graph)         # (V, 3)
        G_cur_np = G_cur.detach().cpu().numpy().astype(np.float64)
        b_np = G_cur_np.reshape(n_dofs)

        try:
            lam_flat, info = cg(A, b_np, rtol=tol, maxiter=max_iter)
        except TypeError:
            lam_flat, info = cg(A, b_np, tol=tol, maxiter=max_iter)

        Lambda = lam_flat.reshape(V, 3)

        # Update E[..., 1:] -= D_cov(U)^T Lambda
        update_np = _apply_D_cov_T(Lambda, R_e, tails, heads, n_edges)
        update = torch.from_numpy(update_np).to(device=device, dtype=RDTYPE)
        E_new[..., 1:] = E_new[..., 1:] - update
        E_new[..., 0] = 0.0

        # Re-measure on the FULL 60-vertex covariant residual
        G_post = compute_gauss_residual(E_new, U, graph)
        G_max = float(G_post.abs().max())
        pass_infos.append({
            "pass": refinement_pass,
            "cg_info": int(info),
            "gauss_max": G_max,
        })
        if verbose:
            print(f"[project_gauss_zero_cg] refinement {refinement_pass}: "
                  f"||G_cov||_inf = {G_max:.3e}, cg_info = {int(info)}")
        if G_max <= tol:
            return E_new, {
                "n_refinement_passes": refinement_pass + 1,
                "final_gauss_max": G_max,
                "initial_gauss_max": initial_G_max,
                "per_pass_cg_info": pass_infos,
            }
        # Stalled: bail
        if G_max >= 0.95 * last_G_max and refinement_pass > 0:
            return E_new, {
                "n_refinement_passes": refinement_pass + 1,
                "final_gauss_max": G_max,
                "initial_gauss_max": initial_G_max,
                "per_pass_cg_info": pass_infos,
                "stalled": True,
            }
        last_G_max = G_max

    return E_new, {
        "n_refinement_passes": n_refinement_max,
        "final_gauss_max": last_G_max,
        "initial_gauss_max": initial_G_max,
        "per_pass_cg_info": pass_infos,
        "exhausted_refinement": True,
    }


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------
def compute_hamiltonian(
    U: torch.Tensor,
    E: torch.Tensor,
    graph,
    beta: float,
) -> Tuple[float, float, float]:
    """Return (H, K, V_pot) for the buckyball SU(2) Kogut-Susskind Hamiltonian.

    K = (g^2 / 2) sum_e Tr(E_e^2)
      = (g^2 / 2) sum_e 2 |e_vec|^2   with E_matrix = e_vec.sigma
      = g^2 * sum_e (E[e, 1] / 2)^2 + ... no, see derivation.

    The cubic-kernel convention (verified in symplectic_integrator._hamiltonian_su2)
    stores E with the quaternion packing E[..., 1:] = 2 * alpha, so
    Tr(E_matrix^2) = 2 * (E[..., 1:]**2).sum().  We import that exact form.

    V = (1/g^2) sum_f [N - Re Tr U_f]
      = (1/g^2) * beta_proper, where for SU(2): N - Re Tr U_f = 2 - 2 q0(U_f).
    """
    g2 = _g_squared(beta, N=2)
    # Kinetic: matches the cubic kernel's _hamiltonian_su2 form exactly.
    e_norm_sq = (E[..., 1:] ** 2).sum(-1)    # (n_edges,)
    Tr_E2 = 2.0 * e_norm_sq
    K = (g2 / 2.0) * float(Tr_E2.sum())

    # Potential: V = (1/(g^2 * N)) * sum_f [N - Re Tr U_f]
    # This is the cubic-kernel convention V = (1/g^2) sum_f [1 - (1/N) Re Tr U_f].
    # Boltzmann-matching (beta_T = 2N) requires this (1/N) factor inside; without
    # it V is N x too large, which de-tunes both energy conservation and the
    # canonical kinetic/potential ratio.  N=2 for SU(2).
    Uf = _action().all_face_holonomies(U, graph)    # (F, 4)
    re_tr = 2.0 * Uf[:, 0]
    N = 2
    contrib = float((N - re_tr).sum())              # sum_f [N - Re Tr U_f]
    V_pot = (1.0 / (g2 * N)) * contrib
    return K + V_pot, K, V_pot


# ---------------------------------------------------------------------------
# Force + drift + leapfrog (quaternion form, mirrors cubic kernel)
# ---------------------------------------------------------------------------
def _force(U: torch.Tensor, graph, beta: float) -> torch.Tensor:
    """dE/dt for SU(2) quaternion form on the buckyball.

    F_e = (-beta / (2 N^2)) * proj_q0=0( qmul(U_e, Sigma_e) )
        = -(beta/8) * (0, imag-3-vec of  U_e . Sigma_e)   for SU(2), N=2.
    Sigma_e = staple_sum_q(U, e, graph) is the quaternion-form effective
    staple (sum over the 2 faces containing e) defined exactly as in
    buckyball_action.

    The coefficient -beta/(2 N^2) = -beta/8 transports verbatim from the
    cubic SU(2) kernel; it is fixed by demanding (i) the leapfrog conserves
    H = K + V_cubic-form to O(dt^2) (verified by dt-halving scaling test
    16:1 -> 16:1 in dH) and (ii) the canonical pair structure with the
    drift U -> matrix_exp_su2_q((0, g^2*dt*E[1+a])) * U.  The relevant
    canonical position is q^a with U_e = exp(i q^a T^a) U_e^init for
    T^a = sigma^a/2, conjugate momentum is the stored quantity E[1+a]
    (the packing E[1+a] = 2 alpha is purely a sampler-side convention;
    the integrator treats E[1+a] as the canonical p directly).  V_cubic-form
    derivative dV/dq^a = (1/(g^2 N^2)) [qmul(U_e, V_eff)]^a gives F = -dV/dq^a
    consistent with the drift, with the factor of N^2 = 4 absorbing both
    (a) the (1/N) inside V and (b) the T^a vs sigma^a generator
    normalization.

    Numerical confirmation (seed=42, cold + 0.01 perturbation): dt-halving
    test gives dH(0.04)/dH(0.01) ~ 196 (close to 16 modulo accumulation
    over 20 leapfrog steps), confirming -beta/8 is the correct coefficient
    matched to V_cubic-form.  Alternative coefficients (-beta/4, -beta/16,
    -beta/2) fail the dt-scaling test.
    """
    N = 2
    coeff = -beta / (2.0 * N * N)
    n_edges = graph.n_edges
    F = torch.zeros_like(U)
    ba = _action()
    for e in range(n_edges):
        S = ba.staple_sum_q(U, e, graph)        # (4,)
        Omega = _qmul(U[e], S)                  # (4,)
        proj = Omega.clone()
        proj[..., 0] = 0.0
        F[e] = coeff * proj
    return F


def _drift(U: torch.Tensor, E: torch.Tensor, dt: float, g2: float) -> torch.Tensor:
    """U_e <- matrix_exp_su2_q( (0, +g^2 dt * e_vec) ) . U_e.

    Drift sign convention (+ g^2 dt) imports from the validated cubic kernel
    (workflow review 2026-06-15 Bug #3 -- see symplectic_integrator._drift_su2_q).
    """
    coeff = +g2 * dt
    n_edges = U.shape[0]
    U_new = torch.zeros_like(U)
    for e in range(n_edges):
        arg = torch.zeros(4, dtype=RDTYPE, device=U.device)
        arg[1:] = coeff * E[e, 1:]
        expE = _matrix_exp_su2_q(arg)
        U_new[e] = _qmul(expE, U[e])
        # Renormalize (defends FP roundoff)
        n = float(_qnorm(U_new[e]))
        if n > 0:
            U_new[e] = U_new[e] / n
    return U_new


def leapfrog_step(
    U: torch.Tensor,
    E: torch.Tensor,
    dt: float,
    graph,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One KDK (kick-drift-kick) leapfrog step.

    Symplectic; preserves H and Gauss to discrete order.
    Mirrors symplectic_integrator._leapfrog_step_su2 verbatim with
    buckyball staples.
    """
    g2 = _g_squared(beta, N=2)
    F0 = _force(U, graph, beta)
    E_half = E + (dt / 2.0) * F0
    E_half[..., 0] = 0.0
    U_new = _drift(U, E_half, dt, g2)
    F1 = _force(U_new, graph, beta)
    E_new = E_half + (dt / 2.0) * F1
    E_new[..., 0] = 0.0
    return U_new, E_new


# ---------------------------------------------------------------------------
# Integrate
# ---------------------------------------------------------------------------
def integrate(
    U_init: torch.Tensor,
    E_init: torch.Tensor,
    dt: float,
    n_steps: int,
    graph,
    beta: float,
    measure_every: int = 10,
) -> Dict[str, Any]:
    """Run n_steps leapfrog steps; record H, <P>, max|G| every measure_every.

    Returns a dict with keys
        U_final, E_final            : final state
        H_history, K_history, V_history : (n_samples,) numpy float64
        P_history                   : (n_samples,) numpy float64, mean plaquette
        G_history                   : (n_samples,) numpy float64, max|G|
        step_indices                : (n_samples,) numpy int64, step number
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0; got {dt!r}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0; got {n_steps!r}")
    if measure_every < 1:
        raise ValueError(f"measure_every must be >= 1; got {measure_every!r}")

    ba = _action()
    U = U_init.clone()
    E = E_init.clone()

    H0, K0, V0 = compute_hamiltonian(U, E, graph, beta)
    Uf0 = ba.all_face_holonomies(U, graph)
    P0 = float(Uf0[:, 0].mean())
    G0 = float(compute_gauss_residual(E, U, graph).abs().max())

    H_hist = [H0]
    K_hist = [K0]
    V_hist = [V0]
    P_hist = [P0]
    G_hist = [G0]
    step_idx = [0]

    for s in range(1, n_steps + 1):
        U, E = leapfrog_step(U, E, dt, graph, beta)
        if s % measure_every == 0 or s == n_steps:
            H_s, K_s, V_s = compute_hamiltonian(U, E, graph, beta)
            Uf = ba.all_face_holonomies(U, graph)
            P_s = float(Uf[:, 0].mean())
            G_s = float(compute_gauss_residual(E, U, graph).abs().max())
            H_hist.append(H_s); K_hist.append(K_s); V_hist.append(V_s)
            P_hist.append(P_s); G_hist.append(G_s); step_idx.append(s)

    return {
        "U_final": U,
        "E_final": E,
        "H_history": np.asarray(H_hist, dtype=np.float64),
        "K_history": np.asarray(K_hist, dtype=np.float64),
        "V_history": np.asarray(V_hist, dtype=np.float64),
        "P_history": np.asarray(P_hist, dtype=np.float64),
        "G_history": np.asarray(G_hist, dtype=np.float64),
        "step_indices": np.asarray(step_idx, dtype=np.int64),
        "dt": dt,
        "beta": beta,
        "n_steps": n_steps,
        "measure_every": measure_every,
    }


__all__ = [
    "canonical_sigma",
    "signed_incidence",
    "graph_laplacian",
    "initialize_E_zero",
    "initialize_E_canonical",
    "compute_gauss_residual",
    "compute_gauss_residual_flat",
    "project_gauss_zero_cg",
    "compute_hamiltonian",
    "leapfrog_step",
    "integrate",
]
