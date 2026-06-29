"""su2_4d_heatbath.py - 4D SU(2) pure-gauge Wilson heatbath, CPU vectorized.

Goal: generate thermalized 4D SU(2) configurations for plaquette-correlator
extraction of the holonomy-continuum mass gap (per
gigi/theory/YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md, M1 + M3 + null controls).

Conventions match lattice/gauge_heatbath_gpu.py (which is the validated 4D SU(3)
version):
    - Wilson action: S = beta * sum_p [1 - (1/N) Re Tr U_p], N = 2.
    - Link field U shape (4, L, L, L, L, 4) with the last axis the quaternion
      (q0, q1, q2, q3) of an SU(2) element A = q0*I + i*(q1*sigma_1 + q2*sigma_2 + q3*sigma_3).
    - Periodic boundary conditions in all 4 directions.
    - Direction 0 = "time" (the extent in this direction sets the correlator window).
    - Cabibbo-Marinari simplifies to a single Kennedy-Pendleton SU(2) heatbath
      per link.
    - Checkerboard (red-black) parallelism per direction: links updated together
      have staples built from links not being updated.

Validation: <P>(beta=2.3) ~ 0.50 (published SU(2) Wilson; e.g. Mack, Phys Rep 1986
Table 1.1, and Fingberg-Heller-Karsch 1993). <P>(beta=2.5) ~ 0.58.

Output: U tensor of shape (4, L, L, L, L, 4) - 4 directions, L^4 sites,
quaternion fiber.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


RDTYPE = np.float64
EPS_K = 1e-12
MAX_KP_ITERS = 400


@dataclass
class HeatbathReport:
    L: int
    beta: float
    n_thermalize: int
    n_measure: int
    measure_every: int
    plaquette_history: List[float]
    plaquette_per_config_t_slices: Optional[List[np.ndarray]]
    wilson_per_config: Optional[List[Dict[Tuple[int, int], float]]]
    wall_seconds_thermalize: float
    wall_seconds_measure: float
    seed: int
    final_plaquette: float


def cold_start(L: int) -> np.ndarray:
    """Identity links (classical vacuum), quaternion form (q0=1, q1=q2=q3=0)."""
    U = np.zeros((4, L, L, L, L, 4), dtype=RDTYPE)
    U[..., 0] = 1.0
    return U


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two batched quaternions, broadcast on leading axes.

    a, b have shape (..., 4). Returns shape (..., 4).
    """
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ], axis=-1)


def qconj(a: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (q0, -q1, -q2, -q3); also SU(2) dagger."""
    out = a.copy()
    out[..., 1:] = -out[..., 1:]
    return out


def qnorm(a: np.ndarray) -> np.ndarray:
    """Quaternion norm sqrt(q0^2 + q1^2 + q2^2 + q3^2)."""
    return np.sqrt(np.sum(a * a, axis=-1))


def qproject(a: np.ndarray) -> np.ndarray:
    """Re-normalize quaternion to unit (numerical drift correction)."""
    n = qnorm(a)[..., None].clip(min=1e-15)
    return a / n


def avg_plaquette(U: np.ndarray) -> float:
    """<P> = (1/N_p) sum_p (1/2) Re Tr U_p = (1/N_p) sum_p q0(U_p), all 6 planes."""
    L = U.shape[1]
    planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    total = 0.0
    for mu, nu in planes:
        Um = U[mu]
        Un = U[nu]
        Un_shift = np.roll(Un, -1, axis=mu)
        Um_shift = np.roll(Um, -1, axis=nu)
        Uplaq = qmul(qmul(qmul(Um, Un_shift), qconj(Um_shift)), qconj(Un))
        total += float(np.mean(Uplaq[..., 0]))
    return total / 6.0


def link_in_direction(U: np.ndarray, mu: int, length: int) -> np.ndarray:
    """Parallel-transport product over `length` consecutive links in direction mu.

    Returns U_mu(x) * U_mu(x+mu) * ... * U_mu(x+(length-1)*mu) at every site x.
    Shape (L, L, L, L, 4) quaternion-valued.
    """
    if length < 1:
        raise ValueError(f"link length must be >= 1, got {length}")
    result = U[mu].copy()
    for k in range(1, length):
        shifted = np.roll(U[mu], -k, axis=mu)
        result = qmul(result, shifted)
    return result


def wilson_loop_RT(U: np.ndarray, R: int, T: int) -> float:
    """<W(R, T)> = (1/N) Re Tr <U_(R x T rectangle)>, averaged over all
    sites and all 6 ordered (mu_space, mu_time) plane pairs.

    Wilson loop path at base site x:
        forward R steps in mu_space
        forward T steps in mu_time
        backward R steps in mu_space (= dagger of the forward R-step starting
            from x + T*mu_time)
        backward T steps in mu_time (= dagger of the original T-step at x)

    For SU(2) the trace normalization is Tr/2 = q0.
    """
    L = U.shape[1]
    total = 0.0
    count = 0
    for mu_s in range(4):
        for mu_t in range(4):
            if mu_s == mu_t:
                continue
            # Path legs at every site x
            H_R = link_in_direction(U, mu_s, R)                      # x -> x + R*mu_s
            H_T_at_R = link_in_direction(U, mu_t, T)                 # x -> x + T*mu_t
            H_T_at_R = np.roll(H_T_at_R, -R, axis=mu_s)               # at x + R*mu_s
            H_R_back = link_in_direction(U, mu_s, R)                  # x -> x + R*mu_s
            H_R_back = np.roll(H_R_back, -T, axis=mu_t)               # at x + T*mu_t
            H_T_back = link_in_direction(U, mu_t, T)                  # x -> x + T*mu_t
            loop = qmul(qmul(qmul(H_R, H_T_at_R), qconj(H_R_back)), qconj(H_T_back))
            total += float(loop[..., 0].mean())
            count += 1
    return total / max(count, 1)


def wilson_loop_table(U: np.ndarray, R_max: int, T_max: int) -> Dict[Tuple[int, int], float]:
    """Compute <W(R, T)> for R in 1..R_max, T in 1..T_max. Returns dict keyed (R, T)."""
    out: Dict[Tuple[int, int], float] = {}
    for R in range(1, R_max + 1):
        for T in range(1, T_max + 1):
            out[(R, T)] = wilson_loop_RT(U, R, T)
    return out


def plaquette_t_slice_density(U: np.ndarray) -> np.ndarray:
    """For each t in 0..L-1, return the spatial-volume-averaged plaquette density
    on the t-slice. Only spatial plaquettes (1,2), (1,3), (2,3) are used to get
    the equal-time density needed for plaquette-plaquette correlator C_PP(t).

    Returns shape (L,).
    """
    L = U.shape[1]
    spatial_planes = [(1, 2), (1, 3), (2, 3)]
    out = np.zeros(L, dtype=RDTYPE)
    for mu, nu in spatial_planes:
        Um = U[mu]
        Un = U[nu]
        Un_shift = np.roll(Un, -1, axis=mu)
        Um_shift = np.roll(Um, -1, axis=nu)
        Uplaq = qmul(qmul(qmul(Um, Un_shift), qconj(Um_shift)), qconj(Un))
        # average over spatial dims (1, 2, 3); keep time axis 0
        out += Uplaq[..., 0].mean(axis=(1, 2, 3))
    return out / 3.0  # average over the 3 spatial planes


def staple_sum_q(U: np.ndarray, mu: int) -> np.ndarray:
    """Vectorized staple sum at every site for direction mu, quaternion form.

    Returns shape (L, L, L, L, 4) with V_eff(x) = sum of 6 staples.
    """
    L = U.shape[1]
    Umu = U[mu]
    V = np.zeros_like(Umu)
    for nu in range(4):
        if nu == mu:
            continue
        Unu = U[nu]
        Unu_pmu = np.roll(Unu, -1, axis=mu)        # U_nu(x+mu)
        Umu_pnu = np.roll(Umu, -1, axis=nu)        # U_mu(x+nu)
        # forward staple: U_nu(x+mu) . U_mu(x+nu)^dag . U_nu(x)^dag
        fwd = qmul(qmul(Unu_pmu, qconj(Umu_pnu)), qconj(Unu))
        Unu_mnu = np.roll(Unu, 1, axis=nu)         # U_nu(x-nu)
        Umu_mnu = np.roll(Umu, 1, axis=nu)         # U_mu(x-nu)
        Unu_pmu_mnu = np.roll(Unu_pmu, 1, axis=nu) # U_nu(x+mu-nu)
        # backward staple: U_nu(x+mu-nu)^dag . U_mu(x-nu)^dag . U_nu(x-nu)
        bwd = qmul(qmul(qconj(Unu_pmu_mnu), qconj(Umu_mnu)), Unu_mnu)
        V = V + fwd + bwd
    return V


def sample_kp_x0_vec(xi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized Kennedy-Pendleton x0 sampler.

    xi has shape (...,). Returns x0 with same shape, x0 in [-1, 1] sampled from
    rho(x0) ~ sqrt(1 - x0^2) * exp(xi * x0).

    Uses serial rejection per site (acceptable at small L). Per-site iteration
    cap is MAX_KP_ITERS.
    """
    flat = xi.ravel()
    out = np.empty_like(flat)
    for i, x in enumerate(flat):
        x_safe = max(float(x), 1e-12)
        accepted = False
        for _ in range(MAX_KP_ITERS):
            r1 = max(rng.random(), 1e-300)
            r2 = rng.random()
            r3 = max(rng.random(), 1e-300)
            r4 = rng.random()
            c = math.cos(2.0 * math.pi * r2) ** 2
            delta = (-math.log(r1) * c - math.log(r3)) / x_safe
            if delta < 2.0 and (r4 * r4) <= (1.0 - 0.5 * delta):
                out[i] = 1.0 - delta
                accepted = True
                break
        if not accepted:
            # Haar fallback (essentially never triggered for xi > 1e-6)
            while True:
                xx = 2.0 * rng.random() - 1.0
                if rng.random() <= math.sqrt(max(1.0 - xx * xx, 0.0)):
                    out[i] = xx
                    break
    return out.reshape(xi.shape)


def sample_new_link_su2_vec(
    V_eff: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample new SU(2) link at every site, given staple V_eff (shape ..., 4).

    P(U) ~ exp(beta * q0(qmul(U, V_eff)))   (Wilson SU(2), N=2)
    """
    # V_eff has shape (L, L, L, L, 4). k = |V_eff| per site.
    k = qnorm(V_eff)  # (L, L, L, L)
    safe_k = k.clip(min=EPS_K)
    xi = beta * safe_k  # (L, L, L, L)

    y0 = sample_kp_x0_vec(xi, rng)
    r_perp = np.sqrt(np.clip(1.0 - y0 * y0, 0.0, None))

    # uniform on 2-sphere of radius r_perp per site
    cth = 2.0 * rng.random(size=y0.shape) - 1.0
    sth = np.sqrt(np.clip(1.0 - cth * cth, 0.0, None))
    phi = 2.0 * math.pi * rng.random(size=y0.shape)
    y1 = r_perp * sth * np.cos(phi)
    y2 = r_perp * sth * np.sin(phi)
    y3 = r_perp * cth

    Y_q = np.stack([y0, y1, y2, y3], axis=-1)  # (L,L,L,L,4)

    # Where k < EPS_K, use uniform Haar (the V_eff direction is undetermined).
    haar_mask = (k < EPS_K)
    if haar_mask.any():
        # Replace at those sites with a fresh Haar sample.
        haar_q = np.zeros((int(haar_mask.sum()), 4), dtype=RDTYPE)
        for i in range(haar_q.shape[0]):
            while True:
                xx = 2.0 * rng.random() - 1.0
                if rng.random() <= math.sqrt(max(1.0 - xx * xx, 0.0)):
                    break
            r2_ = math.sqrt(max(1.0 - xx * xx, 0.0))
            cth_ = 2.0 * rng.random() - 1.0
            sth_ = math.sqrt(max(1.0 - cth_ * cth_, 0.0))
            phi_ = 2.0 * math.pi * rng.random()
            haar_q[i] = [xx, r2_ * sth_ * math.cos(phi_), r2_ * sth_ * math.sin(phi_), r2_ * cth_]
        Y_q[haar_mask] = haar_q

    # U_new = Y . V_hat^dag, V_hat = V_eff / k
    V_hat = V_eff / np.where(haar_mask[..., None], 1.0, safe_k[..., None])
    U_new = qmul(Y_q, qconj(V_hat))
    return qproject(U_new)


def checkerboard_mask(L: int, mu: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mask_red, mask_black) boolean (L,L,L,L) for parity update of
    direction mu. Sites with (x_mu + sum of others) even = red.
    """
    coords = np.indices((L, L, L, L))
    parity = coords.sum(axis=0) % 2
    red = (parity == 0)
    black = (parity == 1)
    return red, black


def heatbath_sweep(U: np.ndarray, beta: float, rng: np.random.Generator) -> None:
    """One full sweep: update each direction with checkerboard parity.

    Updates U IN PLACE.
    """
    L = U.shape[1]
    for mu in range(4):
        for parity in (0, 1):
            mask = (np.indices((L, L, L, L)).sum(axis=0) % 2) == parity
            V = staple_sum_q(U, mu)   # (L,L,L,L,4) - depends on links not in this update
            # Sample for all sites, then write back only the masked ones.
            U_new = sample_new_link_su2_vec(V, beta, rng)
            U[mu, mask] = U_new[mask]


def run_heatbath(
    L: int,
    beta: float,
    n_thermalize: int = 200,
    n_measure: int = 100,
    measure_every: int = 2,
    seed: int = 20260628,
    keep_t_slices: bool = True,
    wilson_max: int = 0,
    verbose: bool = True,
) -> HeatbathReport:
    """Cold-start, thermalize, then run measurement sweeps capturing per-t-slice
    plaquette density per config.

    Returns a HeatbathReport. Each measurement config contributes one t-slice
    array of shape (L,) representing the spatial-volume-averaged plaquette
    density at each time slice. These feed C_PP(t) extraction downstream.
    """
    rng = np.random.default_rng(seed)
    U = cold_start(L)
    if verbose:
        print(f"4D SU(2) heatbath: L={L}, beta={beta}, n_therm={n_thermalize}, "
              f"n_meas={n_measure}, measure_every={measure_every}, seed={seed}")
        print(f"  Initial <P> = {avg_plaquette(U):.4f} (should be ~1.0 cold)")

    t0 = time.perf_counter()
    for s in range(1, n_thermalize + 1):
        heatbath_sweep(U, beta, rng)
        if verbose and s % max(1, n_thermalize // 5) == 0:
            print(f"  thermalize sweep {s}/{n_thermalize}  <P>={avg_plaquette(U):.4f}")
    t_therm = time.perf_counter() - t0

    plaquette_history: List[float] = []
    t_slices: List[np.ndarray] = []
    wilson_per_cfg: List[Dict[Tuple[int, int], float]] = []
    t0 = time.perf_counter()
    for s in range(1, n_measure + 1):
        for _ in range(measure_every):
            heatbath_sweep(U, beta, rng)
        p = avg_plaquette(U)
        plaquette_history.append(p)
        if keep_t_slices:
            t_slices.append(plaquette_t_slice_density(U))
        if wilson_max > 0:
            wilson_per_cfg.append(wilson_loop_table(U, wilson_max, wilson_max))
        if verbose and s % max(1, n_measure // 5) == 0:
            print(f"  measure sweep {s}/{n_measure}  <P>={p:.4f}")
    t_meas = time.perf_counter() - t0

    final_p = avg_plaquette(U)
    if verbose:
        print(f"  Thermalization: {t_therm:.1f}s. Measurement: {t_meas:.1f}s.")
        print(f"  Final <P> = {final_p:.4f}")
        if plaquette_history:
            arr = np.array(plaquette_history)
            print(f"  <P> over measurement: mean={arr.mean():.4f}, std={arr.std():.4f}, "
                  f"N={len(arr)}")

    return HeatbathReport(
        L=L,
        beta=beta,
        n_thermalize=n_thermalize,
        n_measure=n_measure,
        measure_every=measure_every,
        plaquette_history=plaquette_history,
        plaquette_per_config_t_slices=t_slices if keep_t_slices else None,
        wilson_per_config=wilson_per_cfg if wilson_max > 0 else None,
        wall_seconds_thermalize=t_therm,
        wall_seconds_measure=t_meas,
        seed=seed,
        final_plaquette=final_p,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="4D SU(2) Wilson heatbath validation")
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--beta", type=float, default=2.3)
    ap.add_argument("--n-therm", type=int, default=200)
    ap.add_argument("--n-measure", type=int, default=100)
    ap.add_argument("--measure-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260628)
    args = ap.parse_args()

    rep = run_heatbath(
        L=args.L, beta=args.beta,
        n_thermalize=args.n_therm, n_measure=args.n_measure,
        measure_every=args.measure_every, seed=args.seed,
    )
    print(f"\nFinal report:")
    print(f"  L={rep.L} beta={rep.beta}")
    print(f"  mean <P> over measurement = {np.mean(rep.plaquette_history):.4f}")
    print(f"  published Wilson <P>(beta=2.3, 4D SU(2)) ~ 0.50 (approx)")
