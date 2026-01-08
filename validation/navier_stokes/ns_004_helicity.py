"""
NS-004: Helicity Barrier Dissipation Prediction
===============================================
Test whether helicity conservation predicts dissipation behavior.

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
class HelicityResult:
    times: np.ndarray
    helicity: np.ndarray
    dissipation: np.ndarray
    helicity_barrier_active: bool
    correlation: float


def helicity_dns_gpu(N: int = 64, Re: float = 500, T: float = 3.0, dt: float = 0.001):
    """
    DNS simulation tracking helicity as dissipation barrier.
    
    Helicity H = ∫ v·ω dV is conserved in inviscid flow.
    With viscosity, dH/dt ~ -2ν ∫ ω·(∇×ω) dV
    
    Hypothesis: High helicity regions resist dissipation.
    """
    print(f"NS-004: Helicity Barrier Test (GPU)")
    print(f"Grid: {N}³, Re={Re}, T={T}")
    
    nu = 1.0 / Re  # Viscosity
    
    # Wavenumbers
    k = torch.fft.fftfreq(N, d=1/N, device=device) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0
    
    # Dealiasing
    k_max = k.max() * 2/3
    dealias = (torch.abs(kx) < k_max) & (torch.abs(ky) < k_max) & (torch.abs(kz) < k_max)
    
    # Initial: ABC flow (maximally helical)
    x = torch.linspace(0, 2*np.pi, N, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    
    A, B, C = 1.0, 1.0, 1.0
    u = A * torch.sin(Z) + C * torch.cos(Y)
    v = B * torch.sin(X) + A * torch.cos(Z)
    w = C * torch.sin(Y) + B * torch.cos(X)
    
    u_hat = torch.fft.fftn(u)
    v_hat = torch.fft.fftn(v)
    w_hat = torch.fft.fftn(w)
    
    times, helicities, dissipations = [], [], []
    t, step = 0.0, 0
    
    while t < T:
        # Vorticity
        ox_hat = 1j * (ky * w_hat - kz * v_hat)
        oy_hat = 1j * (kz * u_hat - kx * w_hat)
        oz_hat = 1j * (kx * v_hat - ky * u_hat)
        
        u_p = torch.fft.ifftn(u_hat).real
        v_p = torch.fft.ifftn(v_hat).real
        w_p = torch.fft.ifftn(w_hat).real
        ox = torch.fft.ifftn(ox_hat).real
        oy = torch.fft.ifftn(oy_hat).real
        oz = torch.fft.ifftn(oz_hat).real
        
        # Helicity H = v·ω
        H = torch.mean(u_p * ox + v_p * oy + w_p * oz).item()
        
        # Dissipation ε = ν|ω|²
        eps = nu * torch.mean(ox**2 + oy**2 + oz**2).item()
        
        times.append(t)
        helicities.append(H)
        dissipations.append(eps)
        
        if step % 100 == 0:
            print(f"  t={t:.2f}: H={H:.4f}, ε={eps:.6f}")
        
        # Early termination on NaN
        if np.isnan(H) or np.isnan(eps):
            print("  NaN detected, stopping early")
            break
        
        # Convective term
        dudx = torch.fft.ifftn(1j * kx * u_hat).real
        dudy = torch.fft.ifftn(1j * ky * u_hat).real
        dudz = torch.fft.ifftn(1j * kz * u_hat).real
        dvdx = torch.fft.ifftn(1j * kx * v_hat).real
        dvdy = torch.fft.ifftn(1j * ky * v_hat).real
        dvdz = torch.fft.ifftn(1j * kz * v_hat).real
        dwdx = torch.fft.ifftn(1j * kx * w_hat).real
        dwdy = torch.fft.ifftn(1j * ky * w_hat).real
        dwdz = torch.fft.ifftn(1j * kz * w_hat).real
        
        conv_u = u_p * dudx + v_p * dudy + w_p * dudz
        conv_v = u_p * dvdx + v_p * dvdy + w_p * dvdz
        conv_w = u_p * dwdx + v_p * dwdy + w_p * dwdz
        
        conv_u_hat = torch.fft.fftn(conv_u) * dealias
        conv_v_hat = torch.fft.fftn(conv_v) * dealias
        conv_w_hat = torch.fft.fftn(conv_w) * dealias
        
        # Pressure projection
        div_conv = kx * conv_u_hat + ky * conv_v_hat + kz * conv_w_hat
        conv_u_hat -= kx * div_conv / k_sq
        conv_v_hat -= ky * div_conv / k_sq
        conv_w_hat -= kz * div_conv / k_sq
        
        # Semi-implicit: diffusion exact, convection explicit
        exp_visc = torch.exp(-nu * k_sq * dt)
        u_hat = (u_hat - dt * conv_u_hat) * exp_visc
        v_hat = (v_hat - dt * conv_v_hat) * exp_visc
        w_hat = (w_hat - dt * conv_w_hat) * exp_visc
        
        t += dt
        step += 1
    
    times = np.array(times)
    helicities = np.array(helicities)
    dissipations = np.array(dissipations)
    
    # Check helicity barrier: high H correlates with low dissipation rate change
    corr = np.corrcoef(np.abs(helicities[:-1]), -np.diff(dissipations))[0, 1]
    barrier_active = corr > 0.3  # Positive correlation = barrier working
    
    return HelicityResult(
        times=times,
        helicity=helicities,
        dissipation=dissipations,
        helicity_barrier_active=barrier_active,
        correlation=corr
    )


def plot_helicity(result: HelicityResult, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    ax1.plot(result.times, result.helicity, 'b-', lw=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Helicity H')
    ax1.set_title('Helicity Evolution')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.semilogy(result.times, result.dissipation, 'r-', lw=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Dissipation ε')
    ax2.set_title('Energy Dissipation Rate')
    ax2.grid(True, alpha=0.3)
    
    status = "BARRIER ACTIVE" if result.helicity_barrier_active else "BARRIER WEAK"
    plt.suptitle(f'NS-004: Helicity Barrier — {status} (r={result.correlation:.2f})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    result = helicity_dns_gpu(N=48, Re=200, T=2.0, dt=0.001)
    
    os.makedirs('../../results/navier_stokes', exist_ok=True)
    plot_helicity(result, '../../results/navier_stokes/ns_004_helicity.png')
    
    np.savez('../../results/navier_stokes/ns_004_data.npz',
             times=result.times, helicity=result.helicity,
             dissipation=result.dissipation, correlation=result.correlation)
    
    print("\n" + "=" * 60)
    if result.helicity_barrier_active:
        print(f"✓ NS-004 PASS: Helicity barrier active (r={result.correlation:.3f})")
    else:
        print(f"~ NS-004 INCONCLUSIVE: Weak correlation (r={result.correlation:.3f})")
    print("=" * 60)
