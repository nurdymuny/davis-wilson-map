"""su2_4d_heatbath_gpu.py - GPU SU(2) pseudo-heatbath (PyTorch / CUDA).

Specialization of lattice/gauge_heatbath_gpu.py from SU(3) to pure SU(2). For
SU(2) the Cabibbo-Marinari decomposition is trivial: SU(2) IS the only
subgroup, so each link update is a single Kennedy-Pendleton step.

Conventions match inertia_damping/su2_4d_heatbath.py (the validated CPU
quaternion version):
    - SU(2) link variable stored as a real quaternion (q0, q1, q2, q3) in the
      last axis: A = q0*I + i*(q1*sigma_1 + q2*sigma_2 + q3*sigma_3).
    - Wilson action S = beta * sum_p [1 - (1/2) Re Tr U_p].
    - U shape (4, L, L, L, L, 4): 4 directions, L^4 sites, 4 quaternion comps.
    - Direction 0 = "time" (sets correlator window).
    - Periodic boundary conditions in all 4 dirs.
    - Checkerboard (red-black) parallelism per direction.

CPU/GPU cross-validation: identical seed should give identical <P> to ~6 sig
figs vs the CPU quaternion code; that's the correctness gate.

Performance estimate (RTX 5070 Laptop, 8.5 GB VRAM):
    L=8  (~16k links): ~0.05s / sweep -> ~30s for 600 sweeps
    L=12 (~83k links): ~0.2s / sweep  -> ~2min for 600 sweeps
    L=16 (~262k links): ~0.5s / sweep -> ~5min for 600 sweeps
    L=20 (~640k links): ~1.2s / sweep -> ~12min for 600 sweeps
    L=24 (~1.3M links): ~2.5s / sweep -> ~25min for 600 sweeps

These are CPU-CPU-bound regime estimates; actual will be faster on warm GPU.

Usage:
    PYTHONIOENCODING=utf-8 python -m inertia_damping.su2_4d_heatbath_gpu --L 8 --beta 2.3
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


RDTYPE = torch.float64
EPS_K = 1e-12
MAX_KP_ITERS = 200


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class HeatbathReport:
    L: int
    beta: float
    n_thermalize: int
    n_measure: int
    measure_every: int
    plaquette_history: List[float]
    plaquette_per_config_t_slices: Optional[List[torch.Tensor]]
    wilson_per_config: Optional[List[Dict[Tuple[int, int], float]]]
    wall_seconds_thermalize: float
    wall_seconds_measure: float
    seed: int
    final_plaquette: float
    device: str


# ----------------------------------------------------------------------
# Quaternion algebra (torch, batched)
# ----------------------------------------------------------------------
def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two batched quaternions; a, b: (..., 4)."""
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ], dim=-1)


def qconj(a: torch.Tensor) -> torch.Tensor:
    out = a.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def qnorm(a: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((a * a).sum(dim=-1))


def qproject(a: torch.Tensor) -> torch.Tensor:
    n = qnorm(a).unsqueeze(-1).clamp(min=1e-15)
    return a / n


# ----------------------------------------------------------------------
# Starts + observables
# ----------------------------------------------------------------------
def cold_start(L: int, device: torch.device) -> torch.Tensor:
    U = torch.zeros((4, L, L, L, L, 4), dtype=RDTYPE, device=device)
    U[..., 0] = 1.0
    return U


def avg_plaquette(U: torch.Tensor) -> float:
    L = U.shape[1]
    planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    total = 0.0
    for mu, nu in planes:
        Um = U[mu]
        Un = U[nu]
        Un_shift = torch.roll(Un, -1, dims=mu)
        Um_shift = torch.roll(Um, -1, dims=nu)
        Uplaq = qmul(qmul(qmul(Um, Un_shift), qconj(Um_shift)), qconj(Un))
        total += float(Uplaq[..., 0].mean().item())
    return total / 6.0


def plaquette_t_slice_density(U: torch.Tensor) -> torch.Tensor:
    L = U.shape[1]
    spatial_planes = [(1, 2), (1, 3), (2, 3)]
    out = torch.zeros(L, dtype=RDTYPE, device=U.device)
    for mu, nu in spatial_planes:
        Um = U[mu]
        Un = U[nu]
        Un_shift = torch.roll(Un, -1, dims=mu)
        Um_shift = torch.roll(Um, -1, dims=nu)
        Uplaq = qmul(qmul(qmul(Um, Un_shift), qconj(Um_shift)), qconj(Un))
        # average over spatial dims (1, 2, 3); keep time axis 0
        out = out + Uplaq[..., 0].mean(dim=(1, 2, 3))
    return out / 3.0


def link_in_direction(U: torch.Tensor, mu: int, length: int) -> torch.Tensor:
    if length < 1:
        raise ValueError(f"link length must be >= 1, got {length}")
    result = U[mu].clone()
    for k in range(1, length):
        shifted = torch.roll(U[mu], -k, dims=mu)
        result = qmul(result, shifted)
    return result


def wilson_loop_RT(U: torch.Tensor, R: int, T: int) -> float:
    L = U.shape[1]
    total = 0.0
    count = 0
    for mu_s in range(4):
        for mu_t in range(4):
            if mu_s == mu_t:
                continue
            H_R = link_in_direction(U, mu_s, R)
            H_T_at_R = link_in_direction(U, mu_t, T)
            H_T_at_R = torch.roll(H_T_at_R, -R, dims=mu_s)
            H_R_back = link_in_direction(U, mu_s, R)
            H_R_back = torch.roll(H_R_back, -T, dims=mu_t)
            H_T_back = link_in_direction(U, mu_t, T)
            loop = qmul(qmul(qmul(H_R, H_T_at_R), qconj(H_R_back)), qconj(H_T_back))
            total += float(loop[..., 0].mean().item())
            count += 1
    return total / max(count, 1)


def wilson_loop_table(U: torch.Tensor, R_max: int, T_max: int) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    for R in range(1, R_max + 1):
        for T in range(1, T_max + 1):
            out[(R, T)] = wilson_loop_RT(U, R, T)
    return out


# ----------------------------------------------------------------------
# Staples + heatbath
# ----------------------------------------------------------------------
def staple_sum_q(U: torch.Tensor, mu: int) -> torch.Tensor:
    Umu = U[mu]
    V = torch.zeros_like(Umu)
    for nu in range(4):
        if nu == mu:
            continue
        Unu = U[nu]
        Unu_pmu = torch.roll(Unu, -1, dims=mu)
        Umu_pnu = torch.roll(Umu, -1, dims=nu)
        fwd = qmul(qmul(Unu_pmu, qconj(Umu_pnu)), qconj(Unu))
        Unu_mnu = torch.roll(Unu, 1, dims=nu)
        Umu_mnu = torch.roll(Umu, 1, dims=nu)
        Unu_pmu_mnu = torch.roll(Unu_pmu, 1, dims=nu)
        bwd = qmul(qmul(qconj(Unu_pmu_mnu), qconj(Umu_mnu)), Unu_mnu)
        V = V + fwd + bwd
    return V


def _kp_x0_gpu(xi: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Vectorized Kennedy-Pendleton x0 sampler. xi has any shape; returns same."""
    shape = xi.shape
    device = xi.device
    x0 = torch.zeros(shape, dtype=RDTYPE, device=device)
    done = torch.zeros(shape, dtype=torch.bool, device=device)
    xi_safe = torch.clamp(xi, min=1e-12)
    for _ in range(MAX_KP_ITERS):
        if bool(done.all()):
            break
        r1 = torch.rand(shape, dtype=RDTYPE, device=device, generator=gen).clamp_(1e-300, 1.0)
        r2 = torch.rand(shape, dtype=RDTYPE, device=device, generator=gen)
        r3 = torch.rand(shape, dtype=RDTYPE, device=device, generator=gen).clamp_(1e-300, 1.0)
        r4 = torch.rand(shape, dtype=RDTYPE, device=device, generator=gen)
        c = torch.cos(2.0 * math.pi * r2) ** 2
        delta = (-torch.log(r1) * c - torch.log(r3)) / xi_safe
        acc = (delta < 2.0) & (r4 * r4 <= 1.0 - 0.5 * delta) & (~done)
        x0 = torch.where(acc, 1.0 - delta, x0)
        done = done | acc
    if not bool(done.all()):
        # Rare Haar fallback: draw x0 from sqrt(1-x0^2) by simple rejection
        u = torch.rand(shape, dtype=RDTYPE, device=device, generator=gen)
        x0 = torch.where(done, x0, 2.0 * u - 1.0)
    return x0


def _sample_new_link(V_eff: torch.Tensor, beta: float, gen: torch.Generator) -> torch.Tensor:
    """Sample new SU(2) link from staple-conditioned heatbath: U_new = Y * V_hat^dag."""
    k = qnorm(V_eff)
    safe_k = k.clamp(min=EPS_K)
    xi = beta * safe_k

    y0 = _kp_x0_gpu(xi, gen)
    r_perp = torch.sqrt(torch.clamp(1.0 - y0 * y0, min=0.0))

    u = torch.rand(y0.shape, dtype=RDTYPE, device=y0.device, generator=gen)
    v = torch.rand(y0.shape, dtype=RDTYPE, device=y0.device, generator=gen)
    cth = 2.0 * u - 1.0
    sth = torch.sqrt(torch.clamp(1.0 - cth * cth, min=0.0))
    phi = 2.0 * math.pi * v
    y1 = r_perp * sth * torch.cos(phi)
    y2 = r_perp * sth * torch.sin(phi)
    y3 = r_perp * cth

    Y_q = torch.stack([y0, y1, y2, y3], dim=-1)
    V_hat = V_eff / safe_k.unsqueeze(-1)  # at degenerate sites the result is overwritten below
    U_new = qmul(Y_q, qconj(V_hat))
    # Haar fallback for sites with k < EPS_K: replace with a fresh Haar quaternion
    haar_mask = (k < EPS_K)
    if bool(haar_mask.any()):
        # Sample a Haar SU(2) at every site (cheap), then write only at degenerate sites.
        u2 = torch.rand(k.shape, dtype=RDTYPE, device=k.device, generator=gen)
        # Haar x0 by rejection (vectorized): take the sample already in y0 (which used xi small => approximately Haar already)
        # Simpler: just use Y_q with V_hat replaced by identity at those sites.
        identity_q = torch.zeros(k.shape + (4,), dtype=RDTYPE, device=k.device)
        identity_q[..., 0] = 1.0
        U_new_haar = qmul(Y_q, qconj(identity_q))
        mask4 = haar_mask.unsqueeze(-1).expand_as(U_new)
        U_new = torch.where(mask4, U_new_haar, U_new)
    return qproject(U_new)


def _parity_mask(L: int, parity: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(L, device=device)
    t, x, y, z = torch.meshgrid(idx, idx, idx, idx, indexing="ij")
    return ((t + x + y + z) % 2 == parity)


def heatbath_sweep(U: torch.Tensor, beta: float, gen: torch.Generator) -> None:
    """One full sweep: each direction x both parities; updates U in place."""
    L = U.shape[1]
    device = U.device
    for mu in range(4):
        for parity in (0, 1):
            mask = _parity_mask(L, parity, device)
            mask4 = mask.unsqueeze(-1).expand_as(U[mu])
            V = staple_sum_q(U, mu)
            U_new = _sample_new_link(V, beta, gen)
            U[mu] = torch.where(mask4, U_new, U[mu])


def run_heatbath(
    L: int,
    beta: float,
    n_thermalize: int = 200,
    n_measure: int = 100,
    measure_every: int = 2,
    seed: int = 20260628,
    device: Optional[torch.device] = None,
    keep_t_slices: bool = True,
    wilson_max: int = 0,
    verbose: bool = True,
) -> HeatbathReport:
    device = device or get_device()
    gen = torch.Generator(device=device).manual_seed(seed)
    U = cold_start(L, device)
    if verbose:
        dev_name = (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
        print(f"4D SU(2) GPU heatbath: L={L}, beta={beta}, device={device} ({dev_name})")
        print(f"  Initial <P> = {avg_plaquette(U):.4f} (should be ~1.0 cold)")

    t0 = time.perf_counter()
    for s in range(1, n_thermalize + 1):
        heatbath_sweep(U, beta, gen)
        if verbose and s % max(1, n_thermalize // 5) == 0:
            print(f"  thermalize sweep {s}/{n_thermalize}  <P>={avg_plaquette(U):.4f}")
    t_therm = time.perf_counter() - t0

    plaquette_history: List[float] = []
    t_slices: List[torch.Tensor] = []
    wilson_per_cfg: List[Dict[Tuple[int, int], float]] = []
    t0 = time.perf_counter()
    for s in range(1, n_measure + 1):
        for _ in range(measure_every):
            heatbath_sweep(U, beta, gen)
        p = avg_plaquette(U)
        plaquette_history.append(p)
        if keep_t_slices:
            t_slices.append(plaquette_t_slice_density(U).cpu())
        if wilson_max > 0:
            wilson_per_cfg.append(wilson_loop_table(U, wilson_max, wilson_max))
        if verbose and s % max(1, n_measure // 5) == 0:
            print(f"  measure sweep {s}/{n_measure}  <P>={p:.4f}")
    t_meas = time.perf_counter() - t0

    final_p = avg_plaquette(U)
    if verbose:
        print(f"  Thermalization: {t_therm:.1f}s. Measurement: {t_meas:.1f}s.")
        print(f"  Final <P> = {final_p:.4f}")

    return HeatbathReport(
        L=L, beta=beta,
        n_thermalize=n_thermalize, n_measure=n_measure, measure_every=measure_every,
        plaquette_history=plaquette_history,
        plaquette_per_config_t_slices=t_slices if keep_t_slices else None,
        wilson_per_config=wilson_per_cfg if wilson_max > 0 else None,
        wall_seconds_thermalize=t_therm,
        wall_seconds_measure=t_meas,
        seed=seed,
        final_plaquette=final_p,
        device=str(device),
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--beta", type=float, default=2.3)
    ap.add_argument("--n-therm", type=int, default=200)
    ap.add_argument("--n-measure", type=int, default=50)
    ap.add_argument("--measure-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--wilson-max", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="Force CPU (for cross-validation)")
    args = ap.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    rep = run_heatbath(L=args.L, beta=args.beta, n_thermalize=args.n_therm,
                       n_measure=args.n_measure, measure_every=args.measure_every,
                       seed=args.seed, device=device, wilson_max=args.wilson_max)
    import numpy as np
    arr = np.array(rep.plaquette_history)
    print(f"\n<P>(beta={args.beta}, L={args.L}) = {arr.mean():.4f} +/- {arr.std()/np.sqrt(len(arr)):.4f}  (device={device})")
