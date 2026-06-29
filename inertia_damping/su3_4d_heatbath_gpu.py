"""su3_4d_heatbath_gpu.py - GPU SU(3) pseudo-heatbath orchestrator for the
holonomy-continuum mass-gap program (4D, M1 + M3 channels).

The HEATBATH KERNEL is the validated Cabibbo-Marinari implementation in
lattice/gauge_heatbath_gpu.py (matches published <P>(beta=6.0) = 0.5937 to <1%).
This module reuses that kernel and adds:
    - plaquette t-slice density   (for M1: connected plaquette-plaquette correlator)
    - SU(3) Wilson loops W(R, T)  (for M3: Creutz-ratio string tension)

Conventions match lattice/gauge_heatbath_gpu.py:
    - Wilson action: S = beta * sum_p [1 - (1/3) Re Tr U_p],  beta = 6/g^2
    - Link field U shape (4, L, L, L, L, 3, 3) complex128 on CUDA
    - Direction 0 = time (sets correlator window)
    - Periodic boundary conditions
    - Cabibbo-Marinari sequential heatbath over 3 SU(2) subgroups (0,1),(0,2),(1,2)
    - Each subgroup uses Kennedy-Pendleton x0 sampler
    - Checkerboard parallelism (red-black) per direction

Validation: <P>(beta=6.0) ~ 0.5937 (Wilson SU(3), published). Compare against
gauge_heatbath_gpu.py::_validate() which already tests this.

Usage:
    PYTHONIOENCODING=utf-8 python -m inertia_damping.su3_4d_heatbath_gpu \\
        --L 8 --beta 6.0 --n-therm 100 --n-measure 50
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# Reuse the validated SU(3) kernel (import the module file directly to avoid
# pulling in lattice/__init__.py which has a numba dependency we don't need).
import importlib.util as _importlib_util
import os as _os
_GHGPU_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "lattice", "gauge_heatbath_gpu.py")
_spec = _importlib_util.spec_from_file_location("_ghgpu_kernel", _GHGPU_PATH)
_ghgpu = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_ghgpu)

CDTYPE = _ghgpu.CDTYPE
RDTYPE = _ghgpu.RDTYPE
get_device = _ghgpu.get_device
cold_start_gpu = _ghgpu.cold_start_gpu
hot_start_gpu = _ghgpu.hot_start_gpu
dag = _ghgpu.dag
retr = _ghgpu.retr
heatbath_sweep_gpu = _ghgpu.heatbath_sweep_gpu
avg_plaquette = _ghgpu.avg_plaquette
project_su3 = _ghgpu.project_su3


@dataclass
class SU3HeatbathReport:
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
# SU(3) observables on top of the validated kernel
# ----------------------------------------------------------------------
def plaquette_at_plane(U: torch.Tensor, mu: int, nu: int) -> torch.Tensor:
    """Return the plaquette matrix U_p at every site for the (mu, nu) plane.

    Shape (L, L, L, L, 3, 3) complex.
    """
    Um = U[mu]
    Un = U[nu]
    Un_mu = torch.roll(Un, -1, dims=mu)
    Um_nu = torch.roll(Um, -1, dims=nu)
    return Um @ Un_mu @ dag(Um_nu) @ dag(Un)


def plaquette_t_slice_density(U: torch.Tensor) -> torch.Tensor:
    """Spatial-volume-averaged plaquette density per time slice.

    <P>(t) = (1/(3 * V_spatial * n_planes)) sum_{spatial planes, x_spatial} Re Tr U_p(x_t)

    For the connected plaquette-plaquette correlator C_PP(t) = <P(0) P(t)> - <P>^2.

    Returns shape (L,) float, on CPU for downstream numpy use.
    """
    L = U.shape[1]
    spatial_planes = [(1, 2), (1, 3), (2, 3)]
    out = torch.zeros(L, dtype=RDTYPE, device=U.device)
    for mu, nu in spatial_planes:
        Uplaq = plaquette_at_plane(U, mu, nu)            # (L,L,L,L,3,3)
        re_tr_over_N = retr(Uplaq) / 3.0                  # (L,L,L,L) real
        # average over spatial dims (1, 2, 3); keep time axis 0
        out = out + re_tr_over_N.mean(dim=(1, 2, 3))
    return out / 3.0


def link_in_direction(U: torch.Tensor, mu: int, length: int) -> torch.Tensor:
    """Product U_mu(x) * U_mu(x+mu) * ... * U_mu(x+(length-1)*mu) at every site.

    Returns shape (L, L, L, L, 3, 3) complex.
    """
    if length < 1:
        raise ValueError(f"link length must be >= 1, got {length}")
    result = U[mu].clone()
    for k in range(1, length):
        result = result @ torch.roll(U[mu], -k, dims=mu)
    return result


def wilson_loop_RT(U: torch.Tensor, R: int, T: int) -> float:
    """<W(R, T)> = (1/3) Re Tr <U_(R x T rectangle)>, averaged over all
    sites and all 6 ordered (mu_space, mu_time) plane pairs.

    For SU(3) the trace normalization is Tr/3.
    """
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
            loop = H_R @ H_T_at_R @ dag(H_R_back) @ dag(H_T_back)
            total += float((retr(loop) / 3.0).mean().item())
            count += 1
    return total / max(count, 1)


def wilson_loop_table(U: torch.Tensor, R_max: int, T_max: int) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    for R in range(1, R_max + 1):
        for T in range(1, T_max + 1):
            out[(R, T)] = wilson_loop_RT(U, R, T)
    return out


# ----------------------------------------------------------------------
# Driver: thermalize + measure with M1 + M3 observables
# ----------------------------------------------------------------------
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
    start: str = "hot",
    reunit_every: int = 1,
    verbose: bool = True,
) -> SU3HeatbathReport:
    device = device or get_device()
    gen = torch.Generator(device=device).manual_seed(seed)

    if start == "hot":
        U = hot_start_gpu(L, device, seed=seed)
    else:
        U = cold_start_gpu(L, device)

    if verbose:
        dev_name = (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
        print(f"4D SU(3) GPU heatbath: L={L}, beta={beta}, device={device} ({dev_name})")
        print(f"  Start: {start}.  Initial <P> = {avg_plaquette(U):.4f}")

    t0 = time.perf_counter()
    for s in range(1, n_thermalize + 1):
        heatbath_sweep_gpu(U, beta, gen, reunitarize=((s % reunit_every) == 0))
        if verbose and s % max(1, n_thermalize // 5) == 0:
            print(f"  thermalize sweep {s}/{n_thermalize}  <P>={avg_plaquette(U):.4f}")
    t_therm = time.perf_counter() - t0

    plaquette_history: List[float] = []
    t_slices: List[torch.Tensor] = []
    wilson_per_cfg: List[Dict[Tuple[int, int], float]] = []
    t0 = time.perf_counter()
    for s in range(1, n_measure + 1):
        for _ in range(measure_every):
            heatbath_sweep_gpu(U, beta, gen, reunitarize=True)
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
        print(f"  Thermalization: {t_therm:.1f}s.  Measurement: {t_meas:.1f}s.")
        print(f"  Final <P> = {final_p:.4f}")

    return SU3HeatbathReport(
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
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--n-therm", type=int, default=100)
    ap.add_argument("--n-measure", type=int, default=50)
    ap.add_argument("--measure-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--wilson-max", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    rep = run_heatbath(L=args.L, beta=args.beta, n_thermalize=args.n_therm,
                       n_measure=args.n_measure, measure_every=args.measure_every,
                       seed=args.seed, device=device, wilson_max=args.wilson_max)
    arr = np.array(rep.plaquette_history)
    pub = {5.7: 0.5490, 6.0: 0.5937, 6.2: 0.6136}
    pub_str = (f"  published <P>({args.beta}) ~ {pub[args.beta]:.4f}"
               if args.beta in pub else "")
    print(f"\n<P>(beta={args.beta}, L={args.L}) = {arr.mean():.4f} +/- {arr.std()/np.sqrt(len(arr)):.4f}")
    print(pub_str)
