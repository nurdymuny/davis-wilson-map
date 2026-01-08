"""
NS-005: 2D vs 3D Regularity Comparison
======================================
2D NS is known to be globally regular. 3D may blow up.

Author: Bee Rosa Davis  
Date: January 8, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")


@dataclass
class DimensionResult:
    times_2d: np.ndarray
    vort_2d: np.ndarray
    times_3d: np.ndarray
    vort_3d: np.ndarray
    bounded_2d: bool
    bounded_3d: bool


def run_2d_ns(N=128, Re=2000, T=3.0, dt=0.001):
    """2D Navier-Stokes (vorticity formulation)."""
    nu = 1.0 / Re
    k = torch.fft.fftfreq(N, d=1/N, device=device) * 2 * np.pi
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    k_sq = kx**2 + ky**2
    k_sq[0, 0] = 1.0
    
    # Initial: random vorticity
    omega = torch.randn(N, N, device=device) * 5
    omega_hat = torch.fft.fft2(omega)
    
    times, max_vorts = [], []
    t = 0.0
    
    while t < T:
        omega = torch.fft.ifft2(omega_hat).real
        max_vorts.append(omega.abs().max().item())
        times.append(t)
        
        # Stream function: ∇²ψ = -ω
        psi_hat = -omega_hat / k_sq
        u = torch.fft.ifft2(1j * ky * psi_hat).real
        v = torch.fft.ifft2(-1j * kx * psi_hat).real
        
        # Advection: (v·∇)ω
        domegadx = torch.fft.ifft2(1j * kx * omega_hat).real
        domegady = torch.fft.ifft2(1j * ky * omega_hat).real
        advection = u * domegadx + v * domegady
        adv_hat = torch.fft.fft2(advection)
        
        # Time step
        omega_hat = (omega_hat - dt * adv_hat) * torch.exp(-nu * k_sq * dt)
        t += dt
    
    return np.array(times), np.array(max_vorts)


def run_3d_ns(N=48, Re=2000, T=1.5, dt=0.001):
    """3D Navier-Stokes."""
    nu = 1.0 / Re
    k = torch.fft.fftfreq(N, d=1/N, device=device) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0
    
    # Initial: Taylor-Green
    x = torch.linspace(0, 2*np.pi, N, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w = torch.zeros_like(X)
    
    u_hat = torch.fft.fftn(u)
    v_hat = torch.fft.fftn(v)
    w_hat = torch.fft.fftn(w)
    
    times, max_vorts = [], []
    t = 0.0
    
    while t < T:
        ox_hat = 1j * (ky * w_hat - kz * v_hat)
        oy_hat = 1j * (kz * u_hat - kx * w_hat)
        oz_hat = 1j * (kx * v_hat - ky * u_hat)
        
        ox = torch.fft.ifftn(ox_hat).real
        oy = torch.fft.ifftn(oy_hat).real
        oz = torch.fft.ifftn(oz_hat).real
        vort_mag = torch.sqrt(ox**2 + oy**2 + oz**2)
        
        max_vorts.append(vort_mag.max().item())
        times.append(t)
        
        if max_vorts[-1] > 1e5:
            break
        
        u_p = torch.fft.ifftn(u_hat).real
        v_p = torch.fft.ifftn(v_hat).real
        w_p = torch.fft.ifftn(w_hat).real
        
        # Convective terms
        dudx = torch.fft.ifftn(1j * kx * u_hat).real
        dudy = torch.fft.ifftn(1j * ky * u_hat).real
        dudz = torch.fft.ifftn(1j * kz * u_hat).real
        dvdx = torch.fft.ifftn(1j * kx * v_hat).real
        dvdy = torch.fft.ifftn(1j * ky * v_hat).real
        dvdz = torch.fft.ifftn(1j * kz * v_hat).real
        dwdx = torch.fft.ifftn(1j * kx * w_hat).real
        dwdy = torch.fft.ifftn(1j * ky * w_hat).real
        dwdz = torch.fft.ifftn(1j * kz * w_hat).real
        
        cu = torch.fft.fftn(u_p*dudx + v_p*dudy + w_p*dudz)
        cv = torch.fft.fftn(u_p*dvdx + v_p*dvdy + w_p*dvdz)
        cw = torch.fft.fftn(u_p*dwdx + v_p*dwdy + w_p*dwdz)
        
        div = kx*cu + ky*cv + kz*cw
        cu -= kx * div / k_sq
        cv -= ky * div / k_sq
        cw -= kz * div / k_sq
        
        exp_v = torch.exp(-nu * k_sq * dt)
        u_hat = (u_hat - dt * cu) * exp_v
        v_hat = (v_hat - dt * cv) * exp_v
        w_hat = (w_hat - dt * cw) * exp_v
        
        t += dt
    
    return np.array(times), np.array(max_vorts)


if __name__ == "__main__":
    print("NS-005: 2D vs 3D Regularity Comparison")
    
    print("\nRunning 2D simulation...")
    t2d, v2d = run_2d_ns(N=128, Re=2000, T=3.0)
    print(f"  2D: max ω = {v2d.max():.2f}")
    
    print("\nRunning 3D simulation...")
    t3d, v3d = run_3d_ns(N=48, Re=2000, T=1.5)
    print(f"  3D: max ω = {v3d.max():.2f}")
    
    bounded_2d = v2d.max() < 1000
    bounded_3d = v3d.max() < 1000
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(t2d, v2d, 'b-', lw=2, label='2D (bounded)')
    ax.semilogy(t3d, v3d, 'r-', lw=2, label='3D (unbounded?)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Vorticity')
    ax.set_title('NS-005: 2D vs 3D Regularity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs('../../results/navier_stokes', exist_ok=True)
    plt.savefig('../../results/navier_stokes/ns_005_2d3d.png', dpi=150)
    print(f"Saved: ../../results/navier_stokes/ns_005_2d3d.png")
    
    np.savez('../../results/navier_stokes/ns_005_data.npz',
             times_2d=t2d, vort_2d=v2d, times_3d=t3d, vort_3d=v3d)
    
    print("\n" + "=" * 60)
    if bounded_2d and not bounded_3d:
        print("✓ NS-005 PASS: 2D bounded, 3D shows growth")
    elif bounded_2d and bounded_3d:
        print("~ NS-005 PARTIAL: Both bounded (need higher Re)")
    else:
        print("? NS-005 UNEXPECTED: 2D unbounded")
    print("=" * 60)
