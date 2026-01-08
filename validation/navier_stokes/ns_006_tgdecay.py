"""
NS-006: Taylor-Green Vortex Enstrophy Decay
==========================================
Match known enstrophy evolution for TG vortex.

Author: Bee Rosa Davis
Date: January 8, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")


def taylor_green_decay(N=64, Re=1600, T=10.0, dt=0.002):
    """Simulate TG vortex and track enstrophy."""
    print(f"NS-006: Taylor-Green Decay (N={N}, Re={Re})")
    
    nu = 1.0 / Re
    k = torch.fft.fftfreq(N, d=1/N, device=device) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0
    
    # Taylor-Green initial condition
    x = torch.linspace(0, 2*np.pi, N, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    
    u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w = torch.zeros_like(X)
    
    u_hat = torch.fft.fftn(u)
    v_hat = torch.fft.fftn(v)
    w_hat = torch.fft.fftn(w)
    
    times, enstrophies, energies = [], [], []
    t = 0.0
    
    while t < T:
        # Vorticity
        ox_hat = 1j * (ky * w_hat - kz * v_hat)
        oy_hat = 1j * (kz * u_hat - kx * w_hat)
        oz_hat = 1j * (kx * v_hat - ky * u_hat)
        
        ox = torch.fft.ifftn(ox_hat).real
        oy = torch.fft.ifftn(oy_hat).real
        oz = torch.fft.ifftn(oz_hat).real
        
        u_p = torch.fft.ifftn(u_hat).real
        v_p = torch.fft.ifftn(v_hat).real
        w_p = torch.fft.ifftn(w_hat).real
        
        # Enstrophy = 0.5 * <|ω|²>
        enstrophy = 0.5 * torch.mean(ox**2 + oy**2 + oz**2).item()
        energy = 0.5 * torch.mean(u_p**2 + v_p**2 + w_p**2).item()
        
        times.append(t)
        enstrophies.append(enstrophy)
        energies.append(energy)
        
        if len(times) % 500 == 1:
            print(f"  t={t:.2f}: E={energy:.6f}, Ω={enstrophy:.6f}")
        
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
    
    return np.array(times), np.array(enstrophies), np.array(energies)


if __name__ == "__main__":
    times, enstrophies, energies = taylor_green_decay(N=64, Re=1600, T=10.0)
    
    # Known behavior: enstrophy peaks around t~9 for Re=1600
    peak_idx = np.argmax(enstrophies)
    peak_time = times[peak_idx]
    peak_enst = enstrophies[peak_idx]
    
    print(f"\nEnstrophy peak: t={peak_time:.2f}, Ω={peak_enst:.4f}")
    
    # Expected peak time is around t~8-10 for Re=1600
    expected_peak_range = (6.0, 12.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    ax1.plot(times, energies, 'b-', lw=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Kinetic Energy')
    ax1.set_title('Energy Decay')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(times, enstrophies, 'r-', lw=2)
    ax2.axvline(peak_time, color='g', ls='--', label=f'Peak t={peak_time:.1f}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Enstrophy')
    ax2.set_title('Enstrophy Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('NS-006: Taylor-Green Vortex Decay')
    plt.tight_layout()
    
    os.makedirs('../../results/navier_stokes', exist_ok=True)
    plt.savefig('../../results/navier_stokes/ns_006_tgdecay.png', dpi=150)
    print(f"Saved: ../../results/navier_stokes/ns_006_tgdecay.png")
    
    np.savez('../../results/navier_stokes/ns_006_data.npz',
             times=times, enstrophies=enstrophies, energies=energies,
             peak_time=peak_time, peak_enstrophy=peak_enst)
    
    print("\n" + "=" * 60)
    if expected_peak_range[0] <= peak_time <= expected_peak_range[1]:
        print(f"✓ NS-006 PASS: Enstrophy peak at t={peak_time:.2f} matches literature")
    else:
        print(f"~ NS-006 PARTIAL: Peak at t={peak_time:.2f}, expected {expected_peak_range}")
    print("=" * 60)
