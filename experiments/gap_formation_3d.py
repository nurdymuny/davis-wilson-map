"""
Interactive 3D Gap Formation Visualization
==========================================
Creates an interactive Plotly 3D visualization showing the thermalization
trajectory spiraling toward the vacuum attractor.
"""

import modal
import numpy as np

app = modal.App("gap-formation-3d")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "numba", "scipy", "scikit-learn", "plotly", "kaleido", "h5py")
    .add_local_dir("./lattice", remote_path="/root/lattice")
    .add_local_dir("./analysis", remote_path="/root/analysis")
)

@app.function(image=image, gpu="T4", timeout=1800)
def create_3d_visualization():
    import numpy as np
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go
    
    import sys
    sys.path.insert(0, "/root")
    
    from lattice.gauge_config import GaugeConfig, hot_start, heatbath_sweep
    from lattice.wilson_loops import average_plaquette
    from lattice.topological import compute_topological_charge
    
    print("=" * 60)
    print("CREATING INTERACTIVE 3D GAP FORMATION")
    print("=" * 60)
    
    # Parameters - reduced for faster execution
    L, beta = 6, 6.0  # Smaller lattice
    total_sweeps = 150  # Fewer sweeps
    snapshot_interval = 3
    
    print(f"Lattice: {L}^4, β = {beta}")
    print(f"Total sweeps: {total_sweeps}, snapshot every {snapshot_interval}")
    print("=" * 60)
    
    # Initialize from hot start
    config = hot_start(L, beta)
    
    # Collect snapshots
    snapshots = []
    plaquettes = []
    charges = []
    
    # Plane indices for per-plane features
    planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    print("\nCollecting snapshots...")
    
    for sweep in range(total_sweeps + 1):
        if sweep % snapshot_interval == 0:
            # Simplified feature extraction using vectorized ops
            features = []
            U = config.U
            L_cfg = config.L
            
            # Link trace statistics as fingerprint (fast, vectorized)
            for mu in range(4):
                link_traces = np.real(np.trace(U[mu], axis1=-2, axis2=-1))
                features.append(np.mean(link_traces))
                features.append(np.std(link_traces))
                features.append(np.min(link_traces))
                features.append(np.max(link_traces))
            
            # Plaquette (uses optimized numba function)
            plaq = average_plaquette(config.U)
            features.append(plaq)
            plaquettes.append(plaq)
            
            # Topological charge
            Q = compute_topological_charge(config)
            features.append(Q)
            charges.append(Q)
            
            snapshots.append(features)
            
            if sweep % 30 == 0:
                print(f"Sweep {sweep}: <P> = {plaq:.4f}, Q = {Q:.2f}")
        
        if sweep < total_sweeps:
            config = heatbath_sweep(config)
    
    print(f"\nCollected {len(snapshots)} snapshots")
    
    # Stack and reduce to 3D
    features_matrix = np.vstack(snapshots)
    n_snapshots = len(snapshots)
    
    # PCA to 3D
    n_components = min(3, n_snapshots - 1, features_matrix.shape[1])
    pca = PCA(n_components=n_components)
    coords_3d = pca.fit_transform(features_matrix)
    
    # Pad if needed
    if coords_3d.shape[1] < 3:
        padding = np.zeros((coords_3d.shape[0], 3 - coords_3d.shape[1]))
        coords_3d = np.hstack([coords_3d, padding])
    
    # Compute radial distances from final (vacuum) position
    vacuum_pos = coords_3d[-1]
    radii = np.linalg.norm(coords_3d - vacuum_pos, axis=1)
    
    # Normalize for color mapping
    sweeps = np.arange(0, total_sweeps + 1, snapshot_interval)[:n_snapshots]
    
    print(f"\nInitial radius: {radii[0]:.2f}")
    print(f"Final radius: {radii[-1]:.2f}")
    print(f"Radius reduction: {radii[0]/max(radii[-1], 0.01):.1f}×")
    
    # Create the interactive plot
    fig = go.Figure()
    
    # 1. Main trajectory line (colored by sweep)
    fig.add_trace(go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1],
        z=coords_3d[:, 2],
        mode='lines',
        line=dict(
            color=sweeps,
            colorscale='Inferno',
            width=4,
            colorbar=dict(
                title='Sweep',
                x=1.02,
                len=0.5,
                y=0.75
            )
        ),
        name='Trajectory',
        hovertemplate='Sweep %{customdata[0]}<br>Plaquette: %{customdata[1]:.3f}<br>Q: %{customdata[2]:.2f}<extra></extra>',
        customdata=np.column_stack([sweeps, plaquettes, charges])
    ))
    
    # 2. Points colored by radius (distance from vacuum)
    fig.add_trace(go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1],
        z=coords_3d[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=radii,
            colorscale='Plasma',
            opacity=0.8,
            colorbar=dict(
                title='Radius',
                x=1.02,
                len=0.5,
                y=0.25
            )
        ),
        name='Configurations',
        hovertemplate='Sweep %{customdata[0]}<br>Radius: %{customdata[1]:.2f}<br><P>: %{customdata[2]:.3f}<extra></extra>',
        customdata=np.column_stack([sweeps, radii, plaquettes])
    ))
    
    # 3. Start point (chaos)
    fig.add_trace(go.Scatter3d(
        x=[coords_3d[0, 0]],
        y=[coords_3d[0, 1]],
        z=[coords_3d[0, 2]],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='diamond'),
        text=['START (Chaos)'],
        textposition='top center',
        textfont=dict(size=14, color='red'),
        name='Hot Start',
        showlegend=True
    ))
    
    # 4. End point (vacuum attractor)
    fig.add_trace(go.Scatter3d(
        x=[coords_3d[-1, 0]],
        y=[coords_3d[-1, 1]],
        z=[coords_3d[-1, 2]],
        mode='markers+text',
        marker=dict(size=15, color='cyan', symbol='diamond'),
        text=['VACUUM'],
        textposition='bottom center',
        textfont=dict(size=14, color='cyan'),
        name='Vacuum Attractor',
        showlegend=True
    ))
    
    # 5. Add thermalization boundary sphere (approximate)
    thermalized_idx = np.argmax(np.array(plaquettes) > 1.9)  # When plaquette stabilizes
    if thermalized_idx > 0:
        therm_radius = radii[thermalized_idx]
        # Create sphere mesh
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = vacuum_pos[0] + therm_radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = vacuum_pos[1] + therm_radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = vacuum_pos[2] + therm_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.1,
            colorscale=[[0, 'green'], [1, 'green']],
            showscale=False,
            name='Thermalization Boundary'
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'<b>Davis-Wilson Gap Formation</b><br><sup>Thermalization Trajectory: 8⁴ Lattice, β=6.0 | Radius Collapse: {radii[0]/max(radii[-1], 0.01):.1f}×</sup>',
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2', 
            zaxis_title='PC3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='cube'
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=150, t=80, b=0)
    )
    
    # Add animation frames for auto-rotation (optional)
    frames = []
    for angle in range(0, 360, 5):
        rad = np.radians(angle)
        frames.append(go.Frame(
            layout=dict(
                scene_camera=dict(
                    eye=dict(
                        x=2*np.cos(rad),
                        y=2*np.sin(rad),
                        z=1.2
                    )
                )
            ),
            name=str(angle)
        ))
    
    fig.frames = frames
    
    # Add play/pause buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0.02,
                x=0.02,
                xanchor='left',
                buttons=[
                    dict(
                        label='▶ Rotate',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=100, redraw=True),
                            fromcurrent=True,
                            mode='immediate'
                        )]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate'
                        )]
                    )
                ]
            )
        ]
    )
    
    # Generate HTML content (don't write to file on remote)
    html_content = fig.to_html(include_plotlyjs=True, full_html=True)
    
    print(f"\n✅ Generated HTML visualization")
    print(f"   Snapshots: {n_snapshots}")
    print(f"   Variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    
    return {
        'n_snapshots': n_snapshots,
        'initial_radius': float(radii[0]),
        'final_radius': float(radii[-1]),
        'radius_ratio': float(radii[0]/max(radii[-1], 0.01)),
        'variance_explained': float(sum(pca.explained_variance_ratio_)),
        'html_content': html_content
    }


@app.local_entrypoint()
def main():
    result = create_3d_visualization.remote()
    
    # Save the HTML locally
    html_path = "results/gap_formation_3d.html"
    with open(html_path, 'w') as f:
        f.write(result['html_content'])
    print(f"\n✅ Saved: {html_path}")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in result.items():
        if k != 'html_content':
            print(f"  {k}: {v}")
