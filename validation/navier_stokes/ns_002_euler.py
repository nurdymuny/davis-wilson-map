"""
NS-002: Euler Blow-up Candidates via Geometric Roughness
========================================================
Test whether Δ → ∞ signals loss of regularity in Euler equations.

Author: Bee Rosa Davis
Date: January 8, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

# GPU setup for Blackwell (RTX 5070)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")


@dataclass
class EulerResult:
    times: np.ndarray
    max_vorticity: np.ndarray
    delta_values: np.ndarray
    blowup_detected: bool
    blowup_time: float


def euler_simulation_gpu(N: int = 64, T: float = 2.0, dt: float = 0.005):
    """
    Simulate 3D Euler equations (inviscid NS) and track Δ.
    
    Euler: ∂v/∂t + (v·∇)v = -∇p, ∇·v = 0
    
    Without viscosity, finite-time blowup is possible.
    We track whether Δ diverges before vorticity does.
    """
    print(f"NS-002: Euler Blow-up Detection (GPU)")
    print(f"Grid: {N}³, T_max={T}, dt={dt}")
    print(f"Device: {device}")
    
    # Wavenumbers
    k = torch.fft.fftfreq(N, d=1/N, device=device) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0  # Avoid div/0
    
    # Dealiasing
    k_max = k.max() * 2/3
    dealias = (torch.abs(kx) < k_max) & (torch.abs(ky) < k_max) & (torch.abs(kz) < k_max)
    
    # Initial condition: Kida vortex (known to produce strong vortex stretching)
    x = torch.linspace(0, 2*np.pi, N, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    
    # Kida-like initial condition with strong shear
    u = torch.sin(X) * (torch.cos(3*Y) * torch.cos(Z))
    v = torch.sin(Y) * (torch.cos(3*Z) * torch.cos(X))
    w = torch.sin(Z) * (torch.cos(3*X) * torch.cos(Y))
    
    # To Fourier space
    u_hat = torch.fft.fftn(u)
    v_hat = torch.fft.fftn(v)
    w_hat = torch.fft.fftn(w)
    
    # Project to divergence-free
    k_dot_v = kx * u_hat + ky * v_hat + kz * w_hat
    u_hat -= kx * k_dot_v / k_sq
    v_hat -= ky * k_dot_v / k_sq
    w_hat -= kz * k_dot_v / k_sq
    
    # Time stepping
    times = []
    max_vorts = []
    deltas = []
    
    t = 0.0
    step = 0
    
    while t < T:
        # Vorticity in Fourier space
        ox_hat = 1j * (ky * w_hat - kz * v_hat)
        oy_hat = 1j * (kz * u_hat - kx * w_hat)
        oz_hat = 1j * (kx * v_hat - ky * u_hat)
        
        # Physical space values
        u_phys = torch.fft.ifftn(u_hat).real
        v_phys = torch.fft.ifftn(v_hat).real
        w_phys = torch.fft.ifftn(w_hat).real
        ox = torch.fft.ifftn(ox_hat).real
        oy = torch.fft.ifftn(oy_hat).real
        oz = torch.fft.ifftn(oz_hat).real
        
        # Max vorticity (BKM criterion)
        vort_mag = torch.sqrt(ox**2 + oy**2 + oz**2)
        max_vort = vort_mag.max().item()
        
        # Geometric roughness Δ = enstrophy variance + gradient energy
        enstrophy = 0.5 * torch.mean(ox**2 + oy**2 + oz**2).item()
        enstrophy_var = torch.var(vort_mag).item()
        delta = enstrophy + enstrophy_var
        
        times.append(t)
        max_vorts.append(max_vort)
        deltas.append(delta)
        
        if step % 50 == 0:
            print(f"  t={t:.3f}: ω_max={max_vort:.2f}, Δ={delta:.4f}")
        
        # Check for blowup
        if max_vort > 1e6 or delta > 1e6:
            print(f"  BLOWUP at t={t:.4f}!")
            break
        
        # Euler step (RK2 would be better but this is simpler)
        # Nonlinear term: (v·∇)v in Fourier space via convolution
        # We compute in physical space and transform back
        
        # ∂u/∂x, etc in Fourier
        dudx = torch.fft.ifftn(1j * kx * u_hat).real
        dudy = torch.fft.ifftn(1j * ky * u_hat).real
        dudz = torch.fft.ifftn(1j * kz * u_hat).real
        dvdx = torch.fft.ifftn(1j * kx * v_hat).real
        dvdy = torch.fft.ifftn(1j * ky * v_hat).real
        dvdz = torch.fft.ifftn(1j * kz * v_hat).real
        dwdx = torch.fft.ifftn(1j * kx * w_hat).real
        dwdy = torch.fft.ifftn(1j * ky * w_hat).real
        dwdz = torch.fft.ifftn(1j * kz * w_hat).real
        
        # (v·∇)v
        conv_u = u_phys * dudx + v_phys * dudy + w_phys * dudz
        conv_v = u_phys * dvdx + v_phys * dvdy + w_phys * dvdz
        conv_w = u_phys * dwdx + v_phys * dwdy + w_phys * dwdz
        
        # Transform nonlinear term
        conv_u_hat = torch.fft.fftn(conv_u) * dealias
        conv_v_hat = torch.fft.fftn(conv_v) * dealias
        conv_w_hat = torch.fft.fftn(conv_w) * dealias
        
        # Pressure projection
        div_conv = kx * conv_u_hat + ky * conv_v_hat + kz * conv_w_hat
        conv_u_hat -= kx * div_conv / k_sq
        conv_v_hat -= ky * div_conv / k_sq
        conv_w_hat -= kz * div_conv / k_sq
        
        # Update
        u_hat = u_hat - dt * conv_u_hat
        v_hat = v_hat - dt * conv_v_hat
        w_hat = w_hat - dt * conv_w_hat
        
        t += dt
        step += 1
    
    times = np.array(times)
    max_vorts = np.array(max_vorts)
    deltas = np.array(deltas)
    
    # Did we detect blowup tendency?
    delta_growth = deltas[-1] / (deltas[0] + 1e-10)
    vort_growth = max_vorts[-1] / (max_vorts[0] + 1e-10)
    
    blowup = delta_growth > 100 or vort_growth > 100
    blowup_time = times[-1] if blowup else -1
    
    return EulerResult(
        times=times,
        max_vorticity=max_vorts,
        delta_values=deltas,
        blowup_detected=blowup,
        blowup_time=blowup_time
    )


def plot_euler_result(result: EulerResult, save_path: str = None):
    """Plot Euler simulation results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    ax1.semilogy(result.times, result.max_vorticity, 'b-', linewidth=2)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Max Vorticity |ω|_max', fontsize=12)
    ax1.set_title('Vorticity Evolution (BKM)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.semilogy(result.times, result.delta_values, 'r-', linewidth=2)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Geometric Roughness Δ', fontsize=12)
    ax2.set_title('Davis Δ Evolution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    status = "BLOWUP DETECTED" if result.blowup_detected else "REGULAR"
    plt.suptitle(f'NS-002: Euler Blowup Test — {status}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    result = euler_simulation_gpu(N=64, T=2.0, dt=0.005)
    
    os.makedirs('../../results/navier_stokes', exist_ok=True)
    plot_euler_result(result, '../../results/navier_stokes/ns_002_euler.png')
    
    np.savez('../../results/navier_stokes/ns_002_data.npz',
             times=result.times,
             max_vorticity=result.max_vorticity,
             delta_values=result.delta_values,
             blowup_detected=result.blowup_detected)
    
    print("\n" + "=" * 60)
    if result.blowup_detected:
        print(f"✓ NS-002 PASS: Δ signals blowup tendency at t={result.blowup_time:.3f}")
    else:
        print("~ NS-002: No blowup in simulation window (may need longer T)")
    print("=" * 60)
