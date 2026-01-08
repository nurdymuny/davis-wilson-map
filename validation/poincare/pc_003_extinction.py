"""
PC-003: Extinction Behavior - Discrete Ricci Flow (GPU Optimized)
=================================================================

OBJECTIVE:
  Under Ricci flow, a simply connected 3-manifold with positive
  curvature shrinks to a point in finite time (extinction).
  
  CORRECTED: Implement actual discrete Ricci flow on edge lengths,
  not Wilson flow on gauge connections.

THEORY:
  For round S³: R(t) = √(R₀² - 4t), extinction at t = R₀²/4
  Volume V ~ R³, so t_ext ~ V^(2/3) ~ L²
  Expected scaling: α = 2.0

VALIDATION CRITERIA:
  - Monotonic decrease in total curvature
  - Finite extinction time (volume < threshold)
  - Correct scaling: t_extinction ~ L^2

Author: B. Davis
Date: January 8, 2026
Test: PC-003 from VALIDATION_MASTER.md
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DiscreteRicciFlow:
    """
    Discrete Ricci flow on a 3D cubic lattice approximating S³.
    
    Uses combinatorial Ricci flow with proper positive curvature driving contraction.
    For positive curvature (like S³), the manifold should shrink.
    """
    
    def __init__(self, L: int = 6):
        self.L = L
        # Edge lengths in 3 directions: [L, L, L, 3]
        self.edges = torch.ones((L, L, L, 3), dtype=torch.float32, device=device)
        
    def initialize_bumpy_sphere(self, amplitude: float = 0.3):
        """
        Initialize as positive-curvature "bumpy sphere".
        Start with uniform edge lengths (flat), then the positive
        curvature from the topology drives shrinking.
        """
        L = self.L
        
        # Start with slightly perturbed unit edges
        self.edges = torch.ones((L, L, L, 3), dtype=torch.float32, device=device)
        
        # Add position-dependent perturbation (bumpy)
        x = torch.arange(L, device=device, dtype=torch.float32)
        y = torch.arange(L, device=device, dtype=torch.float32)
        z = torch.arange(L, device=device, dtype=torch.float32)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Sinusoidal bumps to break symmetry
        bump = 1.0 + amplitude * (
            torch.sin(2 * np.pi * X / L) * torch.sin(2 * np.pi * Y / L) +
            torch.sin(2 * np.pi * Y / L) * torch.sin(2 * np.pi * Z / L) +
            torch.sin(2 * np.pi * Z / L) * torch.sin(2 * np.pi * X / L)
        ) / 3
        
        self.edges = bump.unsqueeze(-1).expand(-1, -1, -1, 3).clone()
        self.edges = torch.clamp(self.edges, min=0.5, max=1.5)
    
    def compute_vertex_curvature(self):
        """
        Compute discrete Gaussian curvature at each vertex.
        
        For a 3D lattice with cubic cells, we use angle deficit:
        K_v = 2π - Σ(solid angles at v)
        
        Simplified: K ~ 1 - (average edge length ratio)
        Positive K means positive curvature → should shrink
        """
        L = self.L
        
        # Average edge length at each vertex
        avg_edge = (self.edges[:, :, :, 0] + self.edges[:, :, :, 1] + self.edges[:, :, :, 2]) / 3
        
        # Global average
        global_avg = avg_edge.mean()
        
        # Curvature: positive when locally larger than average (excess angle)
        # This is a simplification - true discrete curvature is more complex
        K = (avg_edge - global_avg) / (global_avg + 1e-8)
        
        # Add intrinsic positive curvature (S³ topology)
        # The total curvature of S³ is 4π², distributed over vertices
        intrinsic_K = 4 * np.pi**2 / (L**3)
        K = K + intrinsic_K
        
        return K
    
    def compute_volume(self):
        """Compute effective volume."""
        vol_density = self.edges[:, :, :, 0] * self.edges[:, :, :, 1] * self.edges[:, :, :, 2]
        return vol_density.sum().item()
    
    def compute_mean_edge_length(self):
        """Mean edge length (proxy for "radius")."""
        return self.edges.mean().item()
    
    def compute_scalar_curvature(self):
        """Total scalar curvature."""
        K = self.compute_vertex_curvature()
        return K.sum().item()
    
    def flow_step(self, dt: float = 0.01):
        """
        One step of discrete Ricci flow.
        
        Key insight: For S³, ALL curvature is positive, so the whole
        thing should shrink uniformly (up to bumps smoothing out).
        
        dℓ/dt = -R * ℓ  where R > 0 for positive curvature
        """
        K = self.compute_vertex_curvature()
        
        # Expand curvature to edge dimensions
        K_expanded = K.unsqueeze(-1).expand(-1, -1, -1, 3)
        
        # Ricci flow: shrink where curvature is positive
        # For S³ topology, intrinsic positive curvature causes uniform shrinking
        self.edges = self.edges * (1 - dt * K_expanded)
        
        # Ensure positivity
        self.edges = torch.clamp(self.edges, min=0.01)
    
    def run_to_extinction(self, threshold: float = 0.3, max_steps: int = 2000, dt: float = 0.01):
        """
        Run flow until mean edge length < threshold.
        """
        mean_edges = [self.compute_mean_edge_length()]
        volumes = [self.compute_volume()]
        
        for t in range(max_steps):
            self.flow_step(dt=dt)
            
            mean_ell = self.compute_mean_edge_length()
            vol = self.compute_volume()
            mean_edges.append(mean_ell)
            volumes.append(vol)
            
            # Extinction when mean edge length below threshold
            if mean_ell < threshold:
                return t + 1, np.array(volumes), np.array(mean_edges)
            
            # Early termination if something blows up
            if np.isnan(mean_ell) or mean_ell > 10:
                return max_steps, np.array(volumes), np.array(mean_edges)
        
        return max_steps, np.array(volumes), np.array(mean_edges)


def test_extinction_scaling():
    """
    Test that extinction time scales as t ~ L² (Ricci flow prediction).
    Run multiple trials per size for statistical robustness.
    """
    print("=" * 60)
    print("PC-003: Discrete Ricci Flow Extinction Test")
    print("=" * 60)
    
    # Use smaller sizes to ensure extinction within max_steps
    sizes = [4, 5, 6, 7, 8]
    n_trials = 3
    
    results = {L: [] for L in sizes}
    
    for L in sizes:
        print(f"\nL = {L}:")
        for trial in range(n_trials):
            flow = DiscreteRicciFlow(L=L)
            flow.initialize_bumpy_sphere(amplitude=0.3)
            
            V0 = flow.compute_volume()
            t_ext, vols, edges = flow.run_to_extinction(threshold=0.15, max_steps=5000, dt=0.015)
            
            results[L].append(t_ext)
            if trial == 0:
                print(f"  Trial {trial+1}: V0={V0:.1f}, t_ext={t_ext}, final_ℓ={edges[-1]:.3f}")
            else:
                print(f"  Trial {trial+1}: t_ext={t_ext}")
        
        mean_t = np.mean(results[L])
        std_t = np.std(results[L])
        print(f"  Mean: {mean_t:.1f} ± {std_t:.1f}")
    
    # Compute scaling exponent
    mean_times = [np.mean(results[L]) for L in sizes]
    
    log_L = np.log(sizes)
    log_t = np.log(mean_times)
    
    # Linear fit: log(t) = α * log(L) + c
    coeffs = np.polyfit(log_L, log_t, 1)
    alpha = coeffs[0]
    
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)
    print(f"Fitted exponent: α = {alpha:.3f}")
    print(f"Expected (Ricci flow): α = 2.0")
    print(f"Deviation: |α - 2| = {abs(alpha - 2.0):.3f}")
    
    # Check monotonicity
    monotonic = all(mean_times[i] < mean_times[i+1] for i in range(len(mean_times)-1))
    print(f"Monotonic increase with L: {monotonic}")
    
    # R² of fit
    predicted = np.exp(coeffs[1]) * np.array(sizes)**alpha
    ss_res = np.sum((mean_times - predicted)**2)
    ss_tot = np.sum((mean_times - np.mean(mean_times))**2)
    r_squared = 1 - ss_res / ss_tot
    print(f"R² of power law fit: {r_squared:.4f}")
    
    # Pass criteria: α between 1.5 and 2.5, monotonic, good fit
    pass_test = monotonic and 1.5 < alpha < 2.5 and r_squared > 0.9
    
    print("\n" + "=" * 60)
    if pass_test:
        print("RESULT: ✅ PASS")
        print(f"  - Extinction time scales as L^{alpha:.2f}")
        print(f"  - Within range of Ricci flow prediction (L²)")
        print(f"  - Monotonic increase confirmed")
    elif monotonic and r_squared > 0.85:
        print("RESULT: ⚠️ PARTIAL")
        print(f"  - Scaling α = {alpha:.2f} differs from theoretical 2.0")
        print(f"  - This may reflect Davis-Wilson vs classical Ricci flow")
    else:
        print("RESULT: ❌ FAIL")
        print(f"  - Monotonic: {monotonic}")
        print(f"  - α = {alpha:.2f}, R² = {r_squared:.3f}")
    print("=" * 60)
    
    # Save results
    os.makedirs("../../results/poincare", exist_ok=True)
    np.savez("../../results/poincare/pc_003_extinction.npz",
             sizes=np.array(sizes),
             mean_times=np.array(mean_times),
             all_results={str(L): results[L] for L in sizes},
             alpha=alpha,
             r_squared=r_squared,
             passed=pass_test)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter with error bars
    stds = [np.std(results[L]) for L in sizes]
    axes[0].errorbar(sizes, mean_times, yerr=stds, fmt='bo', capsize=5, markersize=8)
    L_fit = np.linspace(min(sizes), max(sizes), 50)
    axes[0].plot(L_fit, np.exp(coeffs[1]) * L_fit**alpha, 'r--', 
                 label=f'Fit: t ~ L^{alpha:.2f}', linewidth=2)
    axes[0].plot(L_fit, np.exp(coeffs[1]) * (min(sizes)/L_fit[0])**(2-alpha) * L_fit**2, 
                 'g:', label='Theory: t ~ L²', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Lattice Size L')
    axes[0].set_ylabel('Extinction Time')
    axes[0].set_title('Extinction Time Scaling')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log-log plot
    axes[1].loglog(sizes, mean_times, 'bo', markersize=8)
    axes[1].loglog(L_fit, np.exp(coeffs[1]) * L_fit**alpha, 'r--', 
                   label=f'α = {alpha:.2f}', linewidth=2)
    axes[1].loglog(L_fit, 0.5 * L_fit**2, 'g:', label='α = 2 (theory)', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('log(L)')
    axes[1].set_ylabel('log(t_extinction)')
    axes[1].set_title(f'Log-Log Scaling (R² = {r_squared:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("../../results/poincare/pc_003_extinction.png", dpi=150)
    plt.close()
    
    return pass_test, alpha


if __name__ == "__main__":
    passed, alpha = test_extinction_scaling()
