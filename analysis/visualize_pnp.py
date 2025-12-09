"""
PNP Visualization: P vs NP Manifold Analysis
=============================================
High-fidelity figures for the Grand Unified Report.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import os

# Ensure output directory exists
os.makedirs('results/figures', exist_ok=True)

print("Loading P vs NP manifold data...")
data = np.load('pnp_manifold_clouds.npz')
p_cloud = data['p_cloud']      # (500, 500)
np_cloud = data['np_cloud']    # (500, 500)
p_energies = data['p_energies']
np_energies = data['np_energies']

print(f"P (2-SAT): {p_cloud.shape}, NP (3-SAT): {np_cloud.shape}")

# Create figure with 4 panels
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ============================================================
# Panel 1: Energy Trajectories
# ============================================================
ax1 = axes[0, 0]
steps = np.arange(len(p_energies))

ax1.plot(steps, p_energies, 'b-', alpha=0.7, linewidth=1, label='2-SAT (P)')
ax1.plot(steps, np_energies, 'r-', alpha=0.7, linewidth=1, label='3-SAT (NP)')

ax1.set_xlabel('Metropolis Step', fontsize=12)
ax1.set_ylabel('Energy', fontsize=12)
ax1.set_title('Energy Landscape Traversal', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Add variance annotations
p_var = p_energies.std()
np_var = np_energies.std()
ax1.text(0.02, 0.98, f'σ(P) = {p_var:.0f}\nσ(NP) = {np_var:.0f}\nRatio = {np_var/p_var:.2f}×', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================
# Panel 2: Energy Distributions
# ============================================================
ax2 = axes[0, 1]

ax2.hist(p_energies, bins=30, alpha=0.6, color='blue', label='2-SAT (P)', density=True)
ax2.hist(np_energies, bins=30, alpha=0.6, color='red', label='3-SAT (NP)', density=True)

ax2.set_xlabel('Energy', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Energy Distribution (Roughness)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Add statistics
ax2.axvline(p_energies.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(np_energies.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)

# ============================================================
# Panel 3: PCA Trajectory Projection
# ============================================================
ax3 = axes[1, 0]

# Project both trajectories to 2D
pca = PCA(n_components=2)
combined = np.vstack([p_cloud, np_cloud])
pca.fit(combined)

p_2d = pca.transform(p_cloud)
np_2d = pca.transform(np_cloud)

# Plot trajectories with color gradient for time
colors_p = plt.cm.Blues(np.linspace(0.3, 1, len(p_2d)))
colors_np = plt.cm.Reds(np.linspace(0.3, 1, len(np_2d)))

for i in range(len(p_2d) - 1):
    ax3.plot(p_2d[i:i+2, 0], p_2d[i:i+2, 1], color=colors_p[i], linewidth=0.5, alpha=0.7)
    ax3.plot(np_2d[i:i+2, 0], np_2d[i:i+2, 1], color=colors_np[i], linewidth=0.5, alpha=0.7)

# Mark start and end
ax3.scatter(p_2d[0, 0], p_2d[0, 1], c='blue', s=100, marker='o', edgecolors='black', zorder=5, label='P start')
ax3.scatter(p_2d[-1, 0], p_2d[-1, 1], c='blue', s=100, marker='s', edgecolors='black', zorder=5, label='P end')
ax3.scatter(np_2d[0, 0], np_2d[0, 1], c='red', s=100, marker='o', edgecolors='black', zorder=5, label='NP start')
ax3.scatter(np_2d[-1, 0], np_2d[-1, 1], c='red', s=100, marker='s', edgecolors='black', zorder=5, label='NP end')

ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
ax3.set_title('Manifold Walk (PCA Projection)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.3)

# ============================================================
# Panel 4: Local Dimension / Displacement Analysis
# ============================================================
ax4 = axes[1, 1]

# Compute cumulative displacement (proxy for mixing)
def cumulative_displacement(trajectory):
    """Compute cumulative squared displacement from start."""
    displacements = np.sum((trajectory - trajectory[0])**2, axis=1)
    return np.sqrt(displacements)

p_disp = cumulative_displacement(p_cloud)
np_disp = cumulative_displacement(np_cloud)

ax4.plot(steps, p_disp, 'b-', linewidth=2, label='2-SAT (P)')
ax4.plot(steps, np_disp, 'r-', linewidth=2, label='3-SAT (NP)')

ax4.set_xlabel('Metropolis Step', fontsize=12)
ax4.set_ylabel('Displacement from Origin', fontsize=12)
ax4.set_title('Diffusion (Mixing Speed)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

# Final displacement ratio
final_ratio = p_disp[-1] / np_disp[-1]
ax4.text(0.98, 0.02, f'Final ratio (P/NP): {final_ratio:.2f}×', 
         transform=ax4.transAxes, fontsize=10, horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('P vs NP: Geometric Complexity Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/pnp_analysis.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("✓ Saved: results/figures/pnp_analysis.png")

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "=" * 60)
print("P vs NP SUMMARY STATISTICS")
print("=" * 60)
print(f"\nEnergy Statistics:")
print(f"  P (2-SAT):  mean={p_energies.mean():.1f}, std={p_energies.std():.1f}")
print(f"  NP (3-SAT): mean={np_energies.mean():.1f}, std={np_energies.std():.1f}")
print(f"  Roughness Ratio (NP/P): {np_energies.std()/p_energies.std():.2f}×")

print(f"\nDiffusion Statistics:")
print(f"  P final displacement:  {p_disp[-1]:.2f}")
print(f"  NP final displacement: {np_disp[-1]:.2f}")
print(f"  Mobility Ratio (P/NP): {p_disp[-1]/np_disp[-1]:.2f}×")

print(f"\nPCA Variance Explained:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
