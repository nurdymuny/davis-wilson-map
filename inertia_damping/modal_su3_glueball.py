"""modal_su3_glueball.py - Run 4D SU(3) Wilson heatbath + M1 + M3 observables
on Modal A100, so local crashes can't kill the long-running thermalization.

This is the cloud-native version of inertia_damping/push_ym4_su3_glueball_bundle.py.
Different transport:
  - Runs on Modal A100 GPU (4-hour timeout, $0.30-0.50/hour as of 2026)
  - Writes per-ensemble JSON receipts to Modal volume "halcyon-ym4-su3-results"
  - Does NOT push to gigi directly (no network reach back); pull receipts
    next session and run a local push from them.

Per the YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1 paper-readiness verdict:
the fresh-GPU-laptop sweep at n_therm=150 was under-thermalized at weak coupling
(σ_22 over Bali-Schilling 1992 by 36%/48%/78% at β=6.0/6.2/6.4). Going to
n_therm=1000 minimum (verdict recommendation) requires Cabibbo-Marinari decorrelation
times longer than the laptop has reliably stayed alive in any session today.

This script targets the Modal-grade thermalization:
  n_therm = 1500   (10x the laptop sweep, 15x the under-thermalized one)
  n_meas  = 50     (matches Dec 2025 Modal ensemble that hit published)
  sep     = 10     (between measurements; matches Dec 2025 separation field)
  betas   = [5.7, 6.0, 6.2, 6.4]
  L       = 8      (matches existing SU(2) cross-L set + Dec 2025 SU(3) anchor)
  seed    = 20260628 (deterministic; same as laptop seed for crosscheck)

Cost estimate (Modal A100):
  - SU(3) Cabibbo-Marinari heatbath at L=8 on A100: roughly 0.3-0.5 sec/sweep
    (call it 0.4 sec/sweep)
  - Per ensemble: (n_therm + n_meas * sep) * 0.4 sec = (1500 + 500) * 0.4 = 800 sec = ~13 min
  - 4 ensembles sequential: ~55 min
  - + measurement overhead (Wilson loops, plaquette correlator): ~5 min
  - Total wall: ~60 min A100 = $0.30-0.50

If Modal credit is exhausted, defer; the local laptop with n_therm=1500 can
also do this, it just takes ~80 min and risks another crash.

Usage:
    # 1. Make sure you're authed as bee-davis (already true per `modal token current`)
    # 2. Run:
    modal run inertia_damping/modal_su3_glueball.py

    # 3. After it completes, pull results to local:
    modal volume get halcyon-ym4-su3-results . --force

    # 4. Push to local gigi:
    PYTHONIOENCODING=utf-8 python -m inertia_damping.push_ym4_su3_glueball_from_modal_receipts \\
        --results-dir _modal_pull/halcyon-ym4-su3-results

Cross-check: at beta=6.0 the new sigma_22 should land near 0.193 (Bali-Schilling 1992)
and match Dec 2025 Modal sigma_22=0.1985 within statistical errors. If yes, the
under-thermalization finding is confirmed and the proper-thermalization data is
the trustworthy reference. If no, deeper investigation needed.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime

import modal


app = modal.App("halcyon-ym4-su3-glueball")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy>=1.24", "scipy>=1.10")
)

volume = modal.Volume.from_name("halcyon-ym4-su3-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Cabibbo-Marinari SU(3) heatbath kernel (lifted from
# extended_capabilities/tvr_harvest.py, which is validated against published
# <P>(beta=5.7,6.0,6.2) to <1%)
# ---------------------------------------------------------------------------

CDTYPE = "complex128"
RDTYPE = "float64"
_SUBGROUPS = [(0, 1), (0, 2), (1, 2)]


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={"/results": volume},
)
def run_su3_4d_ensemble(
    L: int = 8,
    beta: float = 6.0,
    n_thermalize: int = 1500,
    n_measure: int = 50,
    separation: int = 10,
    seed: int = 20260628,
    wilson_max: int = 3,
) -> dict:
    """Run one SU(3) ensemble: thermalize, then measure plaquette correlator
    + Wilson loops per config. Returns observable summary; writes per-config
    raw observables to /results/{ensemble_id}/configs.jsonl.
    """
    import math
    import time
    from typing import Dict, List, Tuple
    import numpy as np
    import torch

    cdtype = torch.complex128
    rdtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)

    ensemble_id = f"su3_4d_L{L}_beta{int(beta*100):03d}_modal_th{n_thermalize}"
    print(f"[modal] {ensemble_id}: device={device}, GPU={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}")
    print(f"[modal]   n_thermalize={n_thermalize}, n_measure={n_measure}, separation={separation}, seed={seed}")

    # ------------------------------------------------------------------
    # Helpers (matches lattice/gauge_heatbath_gpu.py)
    # ------------------------------------------------------------------
    def _eye():
        I = torch.zeros((L, L, L, L, 3, 3), dtype=cdtype, device=device)
        for a in range(3):
            I[..., a, a] = 1.0
        return I

    def cold_start():
        return torch.stack([_eye() for _ in range(4)], dim=0)

    def hot_start():
        A = (torch.randn((4, L, L, L, L, 3, 3), dtype=rdtype, device=device, generator=gen)
             + 1j * torch.randn((4, L, L, L, L, 3, 3), dtype=rdtype, device=device, generator=gen))
        Q, _ = torch.linalg.qr(A)
        d = torch.linalg.det(Q)
        Q = Q / (d[..., None, None] ** (1.0 / 3.0))
        return Q.to(cdtype)

    def dag(M):
        return M.conj().transpose(-1, -2)

    def retr(M):
        return torch.einsum('...ii->...', M).real

    def project_su3(U):
        w, _, vh = torch.linalg.svd(U)
        P = w @ vh
        d = torch.linalg.det(P)
        P = P / (d[..., None, None] ** (1.0 / 3.0))
        return P

    def staple_sum(U, mu):
        Umu = U[mu]
        S = torch.zeros_like(Umu)
        for nu in range(4):
            if nu == mu:
                continue
            Unu = U[nu]
            Unu_pmu = torch.roll(Unu, -1, dims=mu)
            Umu_pnu = torch.roll(Umu, -1, dims=nu)
            fwd = Unu_pmu @ dag(Umu_pnu) @ dag(Unu)
            Unu_mnu = torch.roll(Unu, 1, dims=nu)
            Umu_mnu = torch.roll(Umu, 1, dims=nu)
            Unu_pmu_mnu = torch.roll(Unu_pmu, 1, dims=nu)
            bwd = dag(Unu_pmu_mnu) @ dag(Umu_mnu) @ Unu_mnu
            S = S + fwd + bwd
        return S

    def avg_plaquette(U):
        planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        tot = 0.0
        for mu, nu in planes:
            Um, Un = U[mu], U[nu]
            Un_mu = torch.roll(Un, -1, dims=mu)
            Um_nu = torch.roll(Um, -1, dims=nu)
            P = Um @ Un_mu @ dag(Um_nu) @ dag(Un)
            tot = tot + retr(P).mean() / 3.0
        return float(tot / len(planes))

    def plaquette_t_slice(U):
        spatial_planes = [(1, 2), (1, 3), (2, 3)]
        out = torch.zeros(L, dtype=rdtype, device=device)
        for mu, nu in spatial_planes:
            Um, Un = U[mu], U[nu]
            Un_mu = torch.roll(Un, -1, dims=mu)
            Um_nu = torch.roll(Um, -1, dims=nu)
            P = Um @ Un_mu @ dag(Um_nu) @ dag(Un)
            out = out + (retr(P) / 3.0).mean(dim=(1, 2, 3))
        return (out / 3.0).cpu().numpy().tolist()

    def link_in_dir(U, mu, length):
        result = U[mu].clone()
        for k in range(1, length):
            result = result @ torch.roll(U[mu], -k, dims=mu)
        return result

    def wilson_loop_RT(U, R, T):
        total, count = 0.0, 0
        for mu_s in range(4):
            for mu_t in range(4):
                if mu_s == mu_t:
                    continue
                H_R = link_in_dir(U, mu_s, R)
                H_T_at_R = torch.roll(link_in_dir(U, mu_t, T), -R, dims=mu_s)
                H_R_back = torch.roll(link_in_dir(U, mu_s, R), -T, dims=mu_t)
                H_T_back = link_in_dir(U, mu_t, T)
                loop = H_R @ H_T_at_R @ dag(H_R_back) @ dag(H_T_back)
                total += float((retr(loop) / 3.0).mean().item())
                count += 1
        return total / max(count, 1)

    def wilson_loop_table(U, R_max):
        out = {}
        for R in range(1, R_max + 1):
            for T in range(1, R_max + 1):
                out[f"{R}_{T}"] = wilson_loop_RT(U, R, T)
        return out

    # ------------------------------------------------------------------
    # Cabibbo-Marinari heatbath
    # ------------------------------------------------------------------
    def _kp_x0(xi):
        shape, dev = xi.shape, xi.device
        x0 = torch.zeros(shape, dtype=rdtype, device=dev)
        done = torch.zeros(shape, dtype=torch.bool, device=dev)
        xi_safe = torch.clamp(xi, min=1e-12)
        for _ in range(200):
            if bool(done.all()):
                break
            r1 = torch.rand(shape, dtype=rdtype, device=dev, generator=gen).clamp_(1e-300, 1.0)
            r2 = torch.rand(shape, dtype=rdtype, device=dev, generator=gen)
            r3 = torch.rand(shape, dtype=rdtype, device=dev, generator=gen).clamp_(1e-300, 1.0)
            r4 = torch.rand(shape, dtype=rdtype, device=dev, generator=gen)
            c = torch.cos(2 * math.pi * r2) ** 2
            delta = (-(torch.log(r1)) * c - torch.log(r3)) / xi_safe
            acc = (delta < 2.0) & (r4 ** 2 <= 1.0 - 0.5 * delta) & (~done)
            x0 = torch.where(acc, 1.0 - delta, x0)
            done = done | acc
        if not bool(done.all()):
            u = torch.rand(shape, dtype=rdtype, device=dev, generator=gen)
            x0 = torch.where(done, x0, 2.0 * u - 1.0)
        return x0

    def _random_unit3(r, shape):
        u = torch.rand(shape, dtype=rdtype, device=device, generator=gen)
        v = torch.rand(shape, dtype=rdtype, device=device, generator=gen)
        cth = 2 * u - 1.0
        sth = torch.sqrt(torch.clamp(1 - cth ** 2, min=0.0))
        phi = 2 * math.pi * v
        return r * sth * torch.cos(phi), r * sth * torch.sin(phi), r * cth

    def _parity_mask(parity):
        idx = torch.arange(L, device=device)
        t, x, y, z = torch.meshgrid(idx, idx, idx, idx, indexing='ij')
        return ((t + x + y + z) % 2 == parity)

    def _heatbath_dir_parity(U, mu, parity):
        staple = staple_sum(U, mu)
        mask = _parity_mask(parity)
        m = mask[..., None]
        Umu = U[mu]
        for (i, j) in _SUBGROUPS:
            W = Umu @ staple
            w00, w01 = W[..., i, i], W[..., i, j]
            w10, w11 = W[..., j, i], W[..., j, j]
            b0 = (w00 + w11).real
            b1 = (w01 + w10).imag
            b2 = (w01 - w10).real
            b3 = (w00 - w11).imag
            k = torch.sqrt(b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3)
            k_safe = torch.clamp(k, min=1e-12)
            xi = (beta / 3.0) * k
            y0 = _kp_x0(xi)
            rr = torch.sqrt(torch.clamp(1 - y0 * y0, min=0.0))
            y1, y2, y3 = _random_unit3(rr, y0.shape)
            p0, p1, p2, p3 = b0 / k_safe, -b1 / k_safe, -b2 / k_safe, -b3 / k_safe
            X0 = y0 * p0 - (y1 * p1 + y2 * p2 + y3 * p3)
            X1 = y0 * p1 + p0 * y1 + (y2 * p3 - y3 * p2)
            X2 = y0 * p2 + p0 * y2 + (y3 * p1 - y1 * p3)
            X3 = y0 * p3 + p0 * y3 + (y1 * p2 - y2 * p1)
            R00 = (X0 + 1j * X3).to(cdtype)
            R01 = (X2 + 1j * X1).to(cdtype)
            R10 = (-X2 + 1j * X1).to(cdtype)
            R11 = (X0 - 1j * X3).to(cdtype)
            Ui = Umu[..., i, :].clone()
            Uj = Umu[..., j, :].clone()
            new_i = R00[..., None] * Ui + R01[..., None] * Uj
            new_j = R10[..., None] * Ui + R11[..., None] * Uj
            Umu[..., i, :] = torch.where(m, new_i, Ui)
            Umu[..., j, :] = torch.where(m, new_j, Uj)
        U[mu] = Umu

    def heatbath_sweep(U, reunit=True):
        for mu in range(4):
            for parity in (0, 1):
                _heatbath_dir_parity(U, mu, parity)
        if reunit:
            for mu in range(4):
                U[mu] = project_su3(U[mu])
        return U

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    U = hot_start()
    p_init = avg_plaquette(U)
    print(f"[modal]   initial <P> = {p_init:.4f}")

    t0 = time.perf_counter()
    plaquette_thermalize_curve: list[tuple[int, float]] = []
    for s in range(1, n_thermalize + 1):
        heatbath_sweep(U)
        if s % max(1, n_thermalize // 10) == 0:
            p = avg_plaquette(U)
            plaquette_thermalize_curve.append((s, p))
            print(f"[modal]   thermalize {s}/{n_thermalize}  <P>={p:.4f}  ({time.perf_counter()-t0:.0f}s)")
    t_therm = time.perf_counter() - t0
    print(f"[modal]   thermalization: {t_therm:.0f}s")

    t0 = time.perf_counter()
    per_config_p_bar_t: list[list[float]] = []
    per_config_wilson: list[Dict[str, float]] = []
    per_config_plaquette_global: list[float] = []
    for cfg in range(n_measure):
        for _ in range(separation):
            heatbath_sweep(U)
        p = avg_plaquette(U)
        per_config_plaquette_global.append(p)
        per_config_p_bar_t.append(plaquette_t_slice(U))
        per_config_wilson.append(wilson_loop_table(U, wilson_max))
        if (cfg + 1) % max(1, n_measure // 5) == 0:
            print(f"[modal]   measure {cfg+1}/{n_measure}  <P>={p:.4f}  ({time.perf_counter()-t0:.0f}s)")
    t_meas = time.perf_counter() - t0

    p_global_mean = float(np.mean(per_config_plaquette_global))
    print(f"[modal]   measurement: {t_meas:.0f}s.  <P> over ensemble = {p_global_mean:.4f}")

    # Save per-config raw observables (so anything can be reanalyzed)
    out_dir = f"/results/{ensemble_id}"
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/configs.jsonl", "w") as f:
        for cfg in range(n_measure):
            f.write(json.dumps({
                "cfg_idx": cfg,
                "P_global": per_config_plaquette_global[cfg],
                "P_bar_t": per_config_p_bar_t[cfg],
                "wilson_table": per_config_wilson[cfg],
            }) + "\n")

    # Summary receipt
    summary = {
        "ensemble_id": ensemble_id,
        "L": L, "beta": beta, "dimension": 4, "gauge_group": "SU(3)",
        "n_thermalize": n_thermalize, "n_measure": n_measure, "separation": separation,
        "seed": seed, "wilson_max": wilson_max,
        "p_global_mean": p_global_mean,
        "p_global_std": float(np.std(per_config_plaquette_global)),
        "wall_seconds": {"thermalize": t_therm, "measure": t_meas, "total": t_therm + t_meas},
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "modal_volume": "halcyon-ym4-su3-results",
        "thermalize_curve": plaquette_thermalize_curve,
        "framing_doc_version": "YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1",
        "completed_utc": datetime.utcnow().isoformat() + "Z",
    }
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    print(f"[modal]   saved /results/{ensemble_id}/ to volume halcyon-ym4-su3-results")
    return summary


@app.local_entrypoint()
def main(
    betas: str = "5.7,6.0,6.2,6.4",
    L: int = 8,
    n_thermalize: int = 1500,
    n_measure: int = 50,
    separation: int = 10,
    seed: int = 20260628,
):
    """Run the 4-beta SU(3) sweep on Modal sequentially. Each ensemble is one
    Modal function call, so each has its own 4-hour timeout. Sequential
    (not parallel) because we have one GPU at a time anyway and the cost
    is the same."""
    beta_list = [float(b) for b in betas.split(",")]
    print(f"Running SU(3) sweep on Modal: betas={beta_list}, L={L}, n_therm={n_thermalize}")
    results = []
    for beta in beta_list:
        print(f"\n=== beta = {beta} ===")
        result = run_su3_4d_ensemble.remote(
            L=L, beta=beta,
            n_thermalize=n_thermalize, n_measure=n_measure,
            separation=separation, seed=seed,
        )
        results.append(result)
        print(f"  done: <P>={result['p_global_mean']:.4f}, wall={result['wall_seconds']['total']:.0f}s")

    print(f"\n=== All {len(beta_list)} ensembles complete ===")
    for r in results:
        print(f"  {r['ensemble_id']}: <P>={r['p_global_mean']:.4f}")
    print(f"\nPull results to local with:")
    print(f"  modal volume get halcyon-ym4-su3-results . --force")
