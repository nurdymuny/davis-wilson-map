"""
NS Visualization: Navier-Stokes Turbulence Topology
====================================================
High-fidelity figures for the Grand Unified Report.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure output directory exists
os.makedirs('results/figures', exist_ok=True)

print("Loading Navier-Stokes turbulence data...")
data = np.load('ns_topology_cloud.npz')
pts = data['point_cloud']           # (10000, 3)
vort = data['sampled_vorticity']    # (10000,)
field = data['field']               # (64, 64, 64) downsampled
enstrophy_history = data['enstrophy_history']  # (8, 2)

print(f"Point cloud: {pts.shape}")
print(f"Vorticity range: [{vort.min():.2f}, {vort.max():.2f}]")
print(f"Field shape: {field.shape}")

# Create figure with 4 panels
fig = plt.figure(figsize=(16, 12))

# ============================================================
# Panel 1: 3D Vortex Core Visualization
# ============================================================
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# Subsample for visualization
n_plot = 3000
idx = np.random.choice(len(pts), n_plot, replace=False)
pts_plot = pts[idx]
vort_plot = vort[idx]

# Normalize vorticity for coloring
vort_norm = (vort_plot - vort_plot.min()) / (vort_plot.max() - vort_plot.min())

scatter = ax1.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], 
                      c=vort_norm, cmap='hot', s=5, alpha=0.6)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Vortex Core Point Cloud\n(colored by vorticity)', fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.1)
cbar.set_label('Normalized Vorticity')

# ============================================================
# Panel 2: Enstrophy Evolution
# ============================================================
ax2 = fig.add_subplot(2, 2, 2)

times = enstrophy_history[:, 0]
enstrophy = enstrophy_history[:, 1]

ax2.plot(times, enstrophy, 'b-o', linewidth=2, markersize=8)
ax2.axvline(0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Peak (t=0.8)')
ax2.fill_between(times, 0, enstrophy, alpha=0.3)

ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Enstrophy', fontsize=12)
ax2.set_title('Enstrophy Growth (Taylor-Green Vortex)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Growth factor
growth = enstrophy[-1] / enstrophy[0]
ax2.text(0.02, 0.98, f'Growth: {growth:.1f}×', transform=ax2.transAxes, 
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================
# Panel 3: Vorticity Distribution
# ============================================================
ax3 = fig.add_subplot(2, 2, 3)

ax3.hist(vort, bins=50, alpha=0.7, color='darkblue', edgecolor='black', density=True)
ax3.axvline(vort.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {vort.mean():.1f}')
ax3.axvline(np.percentile(vort, 90), color='orange', linestyle='--', linewidth=2, 
            label=f'90th %ile: {np.percentile(vort, 90):.1f}')

ax3.set_xlabel('Vorticity Magnitude', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title('Vorticity Distribution (Sampled Points)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# ============================================================
# Panel 4: Correlation Dimension Analysis
# ============================================================
ax4 = fig.add_subplot(2, 2, 4)

print("\nComputing correlation dimension...")
tree = cKDTree(pts)

# Sample for speed
n_sample = 1000
sample_idx = np.random.choice(len(pts), n_sample, replace=False)
sample = pts[sample_idx]

# Compute pair counts at various scales
r_values = np.logspace(-1.5, 0.5, 15) * (pts.max() - pts.min()) / 10
pair_counts = []

for r in r_values:
    counts = tree.query_ball_point(sample, r)
    avg_count = np.mean([len(c) for c in counts])
    pair_counts.append(avg_count)

pair_counts = np.array(pair_counts)

# Plot log-log
log_r = np.log10(r_values)
log_c = np.log10(pair_counts + 1)  # +1 to avoid log(0)

ax4.plot(log_r, log_c, 'bo-', linewidth=2, markersize=8)

# Fit linear region
fit_mask = (log_r > log_r[3]) & (log_r < log_r[-3])
slope, intercept = np.polyfit(log_r[fit_mask], log_c[fit_mask], 1)
fit_line = slope * log_r + intercept
ax4.plot(log_r, fit_line, 'r--', linewidth=2, label=f'D = {slope:.2f}')

ax4.set_xlabel('log₁₀(r)', fontsize=12)
ax4.set_ylabel('log₁₀(C(r))', fontsize=12)
ax4.set_title('Correlation Dimension Analysis', fontsize=12, fontweight='bold')
ax4.legend(fontsize=12, loc='lower right')
ax4.grid(True, alpha=0.3)

# Add interpretation
if slope < 2.5:
    interp = "Vortex Sheets (2D)"
elif slope < 3:
    interp = "Partial Volume"
else:
    interp = "Volume Filling"
    
ax4.text(0.02, 0.98, f'D_corr = {slope:.2f}\n→ {interp}', transform=ax4.transAxes, 
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if slope < 2.5 else 'wheat', alpha=0.5))

plt.suptitle('Navier-Stokes: Turbulence Topology Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/ns_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("✓ Saved: results/figures/ns_analysis.png")

# ============================================================
# Create 2D slice visualization
# ============================================================
print("\nCreating vorticity field slice...")

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

# XY slice (middle Z)
z_mid = field.shape[2] // 2
im1 = axes2[0].imshow(field[:, :, z_mid].T, cmap='inferno', origin='lower')
axes2[0].set_title(f'XY Slice (z={z_mid})', fontsize=12, fontweight='bold')
axes2[0].set_xlabel('X')
axes2[0].set_ylabel('Y')
plt.colorbar(im1, ax=axes2[0], label='|ω|')

# XZ slice (middle Y)
y_mid = field.shape[1] // 2
im2 = axes2[1].imshow(field[:, y_mid, :].T, cmap='inferno', origin='lower')
axes2[1].set_title(f'XZ Slice (y={y_mid})', fontsize=12, fontweight='bold')
axes2[1].set_xlabel('X')
axes2[1].set_ylabel('Z')
plt.colorbar(im2, ax=axes2[1], label='|ω|')

# YZ slice (middle X)
x_mid = field.shape[0] // 2
im3 = axes2[2].imshow(field[x_mid, :, :].T, cmap='inferno', origin='lower')
axes2[2].set_title(f'YZ Slice (x={x_mid})', fontsize=12, fontweight='bold')
axes2[2].set_xlabel('Y')
axes2[2].set_ylabel('Z')
plt.colorbar(im3, ax=axes2[2], label='|ω|')

plt.suptitle('Vorticity Field at Peak Turbulence (t=0.8)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/ns_vorticity_slices.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("✓ Saved: results/figures/ns_vorticity_slices.png")

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "=" * 60)
print("NAVIER-STOKES SUMMARY STATISTICS")
print("=" * 60)
print(f"\nVorticity Statistics:")
print(f"  Max: {vort.max():.2f}")
print(f"  Mean: {vort.mean():.2f}")
print(f"  Std: {vort.std():.2f}")

print(f"\nEnstrophy Evolution:")
print(f"  Initial (t=0): {enstrophy[0]:.2f}")
print(f"  Peak (t=0.8): {enstrophy[-1]:.2f}")
print(f"  Growth factor: {enstrophy[-1]/enstrophy[0]:.2f}×")

print(f"\nCorrelation Dimension:")
print(f"  D_corr = {slope:.2f}")
print(f"  Interpretation: {interp}")

# Clustering analysis
print(f"\nClustering Analysis:")
high_vort = vort > np.percentile(vort, 90)
high_pts = pts[high_vort]
low_vort = vort < np.percentile(vort, 50)
low_pts = pts[low_vort]

tree_high = cKDTree(high_pts)
tree_low = cKDTree(low_pts)

nn_high = tree_high.query(high_pts, k=2)[0][:, 1].mean()
nn_low = tree_low.query(low_pts, k=2)[0][:, 1].mean()

print(f"  High-vort NN distance: {nn_high:.4f}")
print(f"  Low-vort NN distance: {nn_low:.4f}")
print(f"  Clustering ratio: {nn_low/nn_high:.2f}× (high vort more clustered)")
