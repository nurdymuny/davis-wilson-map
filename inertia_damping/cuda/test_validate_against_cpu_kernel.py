"""Validate the batched CUDA leapfrog against the CPU buckyball_integrator
on a small N=100 trajectory at fixed (β, seed, U0, E0).

We pass IDENTICAL initial conditions to both kernels. Differences come from:
  - CPU: sequential Python ``for e in range(n_edges)`` loops
  - GPU: batched gather + chain-product, scatter_add for Gauss residual

float64 mults at the elementary level are bit-identical between CPU/CUDA
torch; the only non-bit-identical step is ``scatter_add`` on CUDA, which
only affects the Gauss diagnostic, not the trajectory.

Gates:
  PASS-1: per-step plaquette agrees to relative tol 1e-8
  PASS-2: per-step Hamiltonian conservation rel-diff agrees to abs tol 1e-8
  PASS-3: final mean plaquette agrees to abs tol 1e-10
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from inertia_damping import buckyball_graph as bgraph
from inertia_damping import buckyball_integrator as cpu_kernel
from inertia_damping.cuda import batched_leapfrog as gpu_kernel


def main() -> int:
    beta = 2.5
    dt = 0.01
    n_steps = 100
    seed = 20260616
    measure_every = 10
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device_str}")
    print(f"beta={beta} dt={dt} n_steps={n_steps} seed={seed}")

    g = bgraph.build_truncated_icosahedron()
    assert g.verify_euler() and g.verify_three_regular()
    print(f"graph: V={g.n_vertices} E={g.n_edges} F={g.n_faces}")

    # ---- shared initial conditions (sample on CPU) -----------------------
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    sigma = gpu_kernel.matrix_exp_su2_q  # noqa: F841 (sanity touch)
    sigma_v = (beta / 128.0) ** 0.5
    alpha = sigma_v * torch.randn(
        (g.n_edges, 3), dtype=torch.float64, device="cpu", generator=gen,
    )
    U0_cpu = torch.zeros((g.n_edges, 4), dtype=torch.float64)
    U0_cpu[..., 0] = 1.0
    E0_cpu = torch.zeros((g.n_edges, 4), dtype=torch.float64)
    E0_cpu[..., 1:] = 2.0 * alpha
    # Initial Hamiltonian + Gauss
    H0_cpu, _, _ = cpu_kernel.compute_hamiltonian(U0_cpu, E0_cpu, g, beta)
    G0_cpu = float(cpu_kernel.compute_gauss_residual(E0_cpu, U0_cpu, g).abs().max())
    print(f"  H0 (cpu): {H0_cpu:.12f}  |G0|: {G0_cpu:.3e}")

    # ---- CPU kernel reference --------------------------------------------
    print("CPU kernel: integrating 100 steps...")
    cpu_out = cpu_kernel.integrate(
        U0_cpu, E0_cpu, dt=dt, n_steps=n_steps, graph=g, beta=beta,
        measure_every=measure_every,
    )
    print(f"  CPU P_history[-1]: {cpu_out['P_history'][-1]:.12f}")
    print(f"  CPU H rel drift max: "
          f"{(np.abs(cpu_out['H_history'] - H0_cpu) / abs(H0_cpu)).max():.3e}")
    print(f"  CPU max |G|: {cpu_out['G_history'].max():.3e}")

    # ---- GPU kernel ------------------------------------------------------
    device = torch.device(device_str)
    topo = gpu_kernel.build_topology_from_graph(g, device=device)
    U0_gpu = U0_cpu.to(device).unsqueeze(0)        # (1, E, 4)
    E0_gpu = E0_cpu.to(device).unsqueeze(0)        # (1, E, 4)
    H0_gpu, _, _ = gpu_kernel.compute_hamiltonian(U0_gpu, E0_gpu, topo, beta)
    G0_gpu = gpu_kernel.compute_gauss_residual_max(E0_gpu, U0_gpu, topo)
    print(f"  H0 (gpu): {float(H0_gpu):.12f}  |G0|: {float(G0_gpu):.3e}")

    # Manual step loop to record per-step plaquette + per-measure H, G
    U, E = U0_gpu.clone(), E0_gpu.clone()
    g2 = (2.0 * 2) / beta
    gpu_P_history = []
    gpu_H_meas = [float(H0_gpu)]
    gpu_G_meas = [float(G0_gpu)]
    gpu_step_idx = [0]
    for s in range(1, n_steps + 1):
        U, E = gpu_kernel.leapfrog_step(U, E, dt, topo, beta)
        gpu_P_history.append(float(gpu_kernel.compute_mean_plaquette(U, topo)))
        if s % measure_every == 0 or s == n_steps:
            Hs, _, _ = gpu_kernel.compute_hamiltonian(U, E, topo, beta)
            Gs = gpu_kernel.compute_gauss_residual_max(E, U, topo)
            gpu_H_meas.append(float(Hs))
            gpu_G_meas.append(float(Gs))
            gpu_step_idx.append(s)

    gpu_P = np.asarray(gpu_P_history)
    gpu_H = np.asarray(gpu_H_meas)
    gpu_G = np.asarray(gpu_G_meas)
    print(f"  GPU P_history[-1]: {gpu_P[-1]:.12f}")
    print(f"  GPU H rel drift max: "
          f"{(np.abs(gpu_H - H0_cpu) / abs(H0_cpu)).max():.3e}")
    print(f"  GPU max |G|: {gpu_G.max():.3e}")

    # ---- Compare ---------------------------------------------------------
    # The CPU "P_history" is sampled at measure_every steps + step 0; the GPU
    # P_history above is every step. Compare at the CPU's step indices.
    cpu_step_idx = cpu_out["step_indices"]
    cpu_P = cpu_out["P_history"]
    # cpu_step_idx[0] = 0 → corresponds to P at the initial state, not in gpu_P.
    cpu_P_dyn = cpu_P[1:]                          # one entry per cpu step idx >= 1
    cpu_step_idx_dyn = cpu_step_idx[1:]
    gpu_P_at_cpu_idx = gpu_P[cpu_step_idx_dyn - 1] # gpu_P[s-1] is step s
    p_diff = np.abs(cpu_P_dyn - gpu_P_at_cpu_idx)
    print()
    print("=== Compare ===")
    print(f"  max |delta P_history|:  {p_diff.max():.3e}  (gate 1e-8)")
    h_diff = np.abs(cpu_out["H_history"] - gpu_H)
    print(f"  max |delta H|:          {h_diff.max():.3e}  (gate 1e-8)")
    g_diff = np.abs(cpu_out["G_history"] - gpu_G)
    print(f"  max |delta G|:          {g_diff.max():.3e}")
    pf_diff = abs(cpu_out["P_history"][-1] - gpu_P[-1])
    print(f"  |delta P_final|:        {pf_diff:.3e}  (gate 1e-10)")
    pass1 = p_diff.max() < 1e-8
    pass2 = h_diff.max() < 1e-8
    pass3 = pf_diff < 1e-10
    print()
    print(f"  PASS-1 (P history):  {'PASS' if pass1 else 'FAIL'}")
    print(f"  PASS-2 (H history):  {'PASS' if pass2 else 'FAIL'}")
    print(f"  PASS-3 (P final):    {'PASS' if pass3 else 'FAIL'}")
    return 0 if (pass1 and pass2 and pass3) else 1


if __name__ == "__main__":
    sys.exit(main())
