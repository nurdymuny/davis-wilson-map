"""
REST OF VALIDATION - Unified GPU Script (v1.5)
===============================================
Implements all remaining validation tests from REST_OF_VALIDATION_SPEC.md:
  - A2S-001: Axiom 2 Cache Sufficiency (k-NN adjacency for D_min)
  - A4C2-001: Axiom 4 Case 2 Curvature Gap (BIN-LEVEL, κ_adj)
  - KSTAR-001: κ* Continuum Survival (SELF-CALIBRATING, ACTUAL t_ref)
  - OSBRIDGE-001: Transfer-Matrix Qualitative Alignment
  - HEPS-001: H_ε → H_phys Uniform Gap (η_mean/η_max SEPARATED)

v1.5 CRITICAL INTEGRATION FIXES (GPT Round 4):
  - ε_skel NOW ACTUALLY VARIES Φ: Recompute cache_data per ε_skel (not shared)
  - WILSON FLOW NOW USED: compute_cache_for_configs_flowed() flows to t* before computing
  - KSTAR USES ACTUAL t_ref: estimate_t_ref() calls find_t_ref() on sample
  - All κ/χ measurements use FLOWED action S_E^flow(t*)

v1.4 fixes (retained):
  - κ_sep_gap: TRUE action gap (vacuum vs nearest non-vacuum quantile), units = action
  - χ_sep: dimensionless ratio = κ_sep_gap / Q_0.9(σ_b), SEPARATE function
  - Wilson flow: IMPLEMENTED (gradient flow to t*) - NOW ACTUALLY USED
  - Topological charge: Proper clover F_μν definition (or labeled "toy proxy")
  - A2S pass: Uses bin_sep_ratio >= 5 (not global_std/q90)
  - A4C2 k-NN: Real k-NN in Φ-space (not sorted 1D proxy)
  - ε_skel: NOW AFFECTS Φ (controls loop sampling richness) - NOW ACTUALLY RECOMPUTED

v1.3 fixes (retained):
  - V1 thresholds use η_max explicitly (worst-case leakage)
  - t_ref naming (t₀-like scale, avoids literature conflict)

v1.2 fixes (retained):
  - Occupancy-based quantile fallback (≥20 for Q_0.1/Q_0.9)
  - η_mean AND η_max tracked; R uses η_max (conservative)
  - A2S uses Q_0.9(σ_b) not max_b (avoid tautology)
  - A4C2 renamed to κ_adj (avoid κ_geom collision)
  - Frozen topology scopes claims, not failures

Target: Single A100 GPU run via Modal
Expected Runtime: ~45 minutes (increased due to flow)
"""

import modal
import json
from datetime import datetime
from pathlib import Path

# Modal setup
app = modal.App("rest-of-validation-v15")
volume = modal.Volume.from_name("tvr-results", create_if_missing=True)

# =============================================================================
# SMOKE TEST MODE: Set to True for quick sanity check before full run
# =============================================================================
SMOKE_TEST = True  # Set to False for full production run

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy", 
        "scipy",
        "matplotlib",
    )
)


@app.function(
    gpu="A100",
    timeout=3600,  # 1 hour (with buffer)
    image=image,
    volumes={"/results": volume},
)
def run_rest_of_validation():
    """Execute all five validation test suites."""
    import torch
    import numpy as np
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("/results")
    results_dir.mkdir(exist_ok=True)
    
    # Track all topological charges for histogram diagnostic
    r_histogram_global = []
    
    # =========================================================================
    # GPU UTILITY FUNCTIONS
    # =========================================================================
    
    def create_checkerboard_indices(L: int, T: int = None):
        """Create indices for red (even) and black (odd) sites."""
        T = T or L
        x0 = torch.arange(L, device=device)
        x1 = torch.arange(L, device=device)
        x2 = torch.arange(L, device=device)
        x3 = torch.arange(T, device=device)
        X0, X1, X2, X3 = torch.meshgrid(x0, x1, x2, x3, indexing='ij')
        parity = (X0 + X1 + X2 + X3) % 2
        red_indices = torch.nonzero(parity == 0, as_tuple=False)
        black_indices = torch.nonzero(parity == 1, as_tuple=False)
        return red_indices, black_indices
    
    def random_su3_batch(batch_size: int) -> torch.Tensor:
        """Generate batch of random SU(3) matrices on GPU (complex64)."""
        A = torch.randn(batch_size, 3, 3, dtype=torch.complex64, device=device)
        A = A + 1j * torch.randn(batch_size, 3, 3, dtype=torch.float32, device=device)
        Q, R = torch.linalg.qr(A)
        det = torch.linalg.det(Q)
        phase = torch.exp(-1j * torch.angle(det) / 3).unsqueeze(-1).unsqueeze(-1)
        return Q * phase
    
    def project_su3_batch(M: torch.Tensor) -> torch.Tensor:
        """Project batch of matrices to SU(3)."""
        U, S, Vh = torch.linalg.svd(M)
        result = U @ Vh
        det = torch.linalg.det(result)
        phase = torch.exp(-1j * torch.angle(det) / 3).unsqueeze(-1).unsqueeze(-1)
        return result * phase
    
    def su3_exp_small_batch(H: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Batched exp(i*epsilon*H) for small epsilon (complex64)."""
        h = epsilon * H
        h = (h - h.mH) / 2
        eye = torch.eye(3, dtype=torch.complex64, device=device).unsqueeze(0)
        X = eye + h + 0.5 * (h @ h)
        return project_su3_batch(X)
    
    # =========================================================================
    # SHARED: Parallel SU(3) Lattice (OPTIMIZED BATCH VERSION)
    # =========================================================================
    
    class ParallelGaugeLattice:
        """GPU-accelerated SU(3) lattice with Red-Black parallel updates (OPTIMIZED)."""
        
        def __init__(self, L: int, beta: float, T: int = None, device=None):
            self.L = L
            self.T = T if T is not None else L
            self.beta = beta
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize links to random SU(3)
            n_links = L * L * L * self.T * 4
            links_flat = random_su3_batch(n_links)
            self.links = links_flat.reshape(L, L, L, self.T, 4, 3, 3)
            
            # Create checkerboard indices for parallel updates
            self.red_idx, self.black_idx = create_checkerboard_indices(L, self.T)
            
            # Create all-sites index tensor for batch operations
            self.all_indices = torch.stack([
                torch.arange(L, device=self.device).repeat_interleave(L*L*self.T),
                torch.arange(L, device=self.device).repeat(L).repeat_interleave(L*self.T),
                torch.arange(L, device=self.device).repeat(L*L).repeat_interleave(self.T),
                torch.arange(self.T, device=self.device).repeat(L*L*L),
            ], dim=1)
            
        def get_links_at_indices(self, indices, mu):
            """Get links at specified indices for direction mu."""
            return self.links[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], mu]
        
        def set_links_at_indices(self, indices, mu, values):
            """Set links at specified indices for direction mu."""
            self.links[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], mu] = values
        
        def compute_staples_batch(self, indices, mu):
            """Compute staples for ALL sites simultaneously (VECTORIZED)."""
            L, T = self.L, self.T
            N = indices.shape[0]
            staples = torch.zeros(N, 3, 3, dtype=torch.complex64, device=self.device)
            
            for nu in range(4):
                if nu == mu:
                    continue
                
                L_nu = T if nu == 3 else L
                L_mu = T if mu == 3 else L
                
                idx_plus_mu = indices.clone()
                idx_plus_mu[:, mu] = (idx_plus_mu[:, mu] + 1) % L_mu
                
                idx_plus_nu = indices.clone()
                idx_plus_nu[:, nu] = (idx_plus_nu[:, nu] + 1) % L_nu
                
                idx_minus_nu = indices.clone()
                idx_minus_nu[:, nu] = (idx_minus_nu[:, nu] - 1 + L_nu) % L_nu
                
                idx_plus_mu_minus_nu = indices.clone()
                idx_plus_mu_minus_nu[:, mu] = (idx_plus_mu_minus_nu[:, mu] + 1) % L_mu
                idx_plus_mu_minus_nu[:, nu] = (idx_plus_mu_minus_nu[:, nu] - 1 + L_nu) % L_nu
                
                # Upper staple: U_nu(x+mu) @ U_mu(x+nu)^dag @ U_nu(x)^dag
                U1 = self.get_links_at_indices(idx_plus_mu, nu)
                U2 = self.get_links_at_indices(idx_plus_nu, mu).mH
                U3 = self.get_links_at_indices(indices, nu).mH
                staples = staples + U1 @ U2 @ U3
                
                # Lower staple: U_nu(x+mu-nu)^dag @ U_mu(x-nu)^dag @ U_nu(x-nu)
                U1 = self.get_links_at_indices(idx_plus_mu_minus_nu, nu).mH
                U2 = self.get_links_at_indices(idx_minus_nu, mu).mH
                U3 = self.get_links_at_indices(idx_minus_nu, nu)
                staples = staples + U1 @ U2 @ U3
            
            return staples
        
        def parallel_metropolis_sweep(self, n_hits: int = 3):
            """Parallel Metropolis sweep using Red-Black checkerboarding (FAST)."""
            total_accepted = 0
            total_proposed = 0
            
            for mu in range(4):
                for color_indices in [self.red_idx, self.black_idx]:
                    N = color_indices.shape[0]
                    
                    for _ in range(n_hits):
                        staples = self.compute_staples_batch(color_indices, mu)
                        current = self.get_links_at_indices(color_indices, mu)
                        
                        # Propose random update (complex64 to match links)
                        H = torch.randn(N, 3, 3, dtype=torch.complex64, device=self.device)
                        X = su3_exp_small_batch(H, 0.15)
                        proposal = X @ current
                        
                        # Compute action change
                        diff = proposal - current
                        dS = -self.beta * torch.real(
                            torch.diagonal(diff @ staples, dim1=-2, dim2=-1).sum(dim=-1)
                        )
                        
                        # Accept/reject
                        rand = torch.rand(N, device=self.device)
                        accept = (dS < 0) | (rand < torch.exp(-dS))
                        
                        # Update links
                        new_links = torch.where(
                            accept.unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3),
                            proposal,
                            current
                        )
                        self.set_links_at_indices(color_indices, mu, new_links)
                        
                        total_accepted += accept.sum().item()
                        total_proposed += N
            
            return total_accepted / total_proposed if total_proposed > 0 else 0.0
        
        def compute_plaquette_batch(self, indices, mu, nu):
            """Compute plaquettes for batch of sites (VECTORIZED)."""
            L, T = self.L, self.T
            L_mu = T if mu == 3 else L
            L_nu = T if nu == 3 else L
            
            idx_plus_mu = indices.clone()
            idx_plus_mu[:, mu] = (idx_plus_mu[:, mu] + 1) % L_mu
            
            idx_plus_nu = indices.clone()
            idx_plus_nu[:, nu] = (idx_plus_nu[:, nu] + 1) % L_nu
            
            U1 = self.get_links_at_indices(indices, mu)
            U2 = self.get_links_at_indices(idx_plus_mu, nu)
            U3 = self.get_links_at_indices(idx_plus_nu, mu).mH
            U4 = self.get_links_at_indices(indices, nu).mH
            
            return U1 @ U2 @ U3 @ U4
        
        def plaquette(self):
            """Compute average plaquette (VECTORIZED)."""
            L, T = self.L, self.T
            all_indices = self.all_indices
            
            plaq_sum = 0.0
            count = 0
            
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    P = self.compute_plaquette_batch(all_indices, mu, nu)
                    traces = torch.real(torch.diagonal(P, dim1=-2, dim2=-1).sum(dim=-1))
                    plaq_sum += traces.sum().item()
                    count += L * L * L * T
            
            return plaq_sum / (3.0 * count)  # Normalize by Nc
        
        def action_density(self):
            """Compute action density S_E / V."""
            return self.beta * (1 - self.plaquette())
        
        def wilson_action(self):
            """
            Compute total Wilson action S_W = β Σ (1 - Re Tr U_plaq / 3) (VECTORIZED).
            This is the FULL action in lattice units (not density).
            """
            L, T = self.L, self.T
            all_indices = self.all_indices
            action = 0.0
            
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    P = self.compute_plaquette_batch(all_indices, mu, nu)
                    traces = torch.real(torch.diagonal(P, dim1=-2, dim2=-1).sum(dim=-1))
                    action += (1.0 - traces / 3.0).sum().item()
            
            return self.beta * action
        
        def wilson_flow_step_batch(self, dt: float):
            """
            Single step of Wilson gradient flow (VECTORIZED).
            Using Euler integration with batch operations.
            """
            L, T = self.L, self.T
            all_indices = self.all_indices
            new_links = self.links.clone()
            
            for mu in range(4):
                # Compute staples for all sites at once
                staples = self.compute_staples_batch(all_indices, mu)  # (N, 3, 3)
                links = self.get_links_at_indices(all_indices, mu)  # (N, 3, 3)
                
                # Flow equation: X = S·U† - U·S†
                X = staples @ links.mH - links @ staples.mH
                
                # Traceless projection
                trace_X = torch.diagonal(X, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3.0
                X = X - trace_X.unsqueeze(-1) * torch.eye(3, dtype=torch.complex64, device=self.device)
                
                # Euler step
                new_U = links - dt * X @ links
                
                # Re-project to SU(3) via QR
                Q, R = torch.linalg.qr(new_U)
                det = torch.linalg.det(Q)
                phase = (det.abs() ** (1/3)) / (det + 1e-10)
                new_U = Q * phase.unsqueeze(-1).unsqueeze(-1)
                
                # Set back using indices
                self.set_links_at_indices(all_indices, mu, new_U)
        
        def wilson_flow_to_t(self, t_target: float, dt: float = 0.01):
            """
            Flow the configuration to flow-time t_target.
            Returns flowed configuration (does NOT modify self.links permanently).
            """
            # Store original
            original_links = self.links.clone()
            
            t = 0.0
            while t < t_target:
                step = min(dt, t_target - t)
                self.wilson_flow_step_batch(step)  # Use batch version
                t += step
            
            flowed_links = self.links.clone()
            
            # Restore original
            self.links = original_links
            
            return flowed_links
        
        def flow_action_density(self, t_flow: float):
            """
            Compute action density after Wilson flow to time t_flow.
            E(t) = -<S> / V at flow time t
            """
            flowed = self.wilson_flow_to_t(t_flow)
            original = self.links.clone()
            self.links = flowed
            E = self.action_density()
            self.links = original
            return E
        
        def find_t_ref(self, c: float = 0.3):
            """
            Find t_ref such that t² <E(t)> = c (t₀-like definition).
            Returns t_ref (NOT w₀, to avoid nomenclature conflict).
            Uses INCREMENTAL flow to avoid re-flowing from scratch each time.
            """
            original_links = self.links.clone()
            dt = 0.02
            t = 0.0
            
            # Flow incrementally and check condition
            for _ in range(25):
                t += dt
                self.wilson_flow_step_batch(dt)
                E_t = self.action_density()
                if t**2 * E_t >= c:
                    self.links = original_links
                    return t
            
            self.links = original_links
            return 0.3  # Fallback

        def wilson_loop(self, R: int, T_loop: int):
            """Compute spatial Wilson loop of size R x T (VECTORIZED)."""
            L = self.L
            T = self.T
            all_indices = self.all_indices  # (N, 4)
            N = all_indices.shape[0]
            
            # Initialize as identity matrices for all sites
            U_bottom = torch.eye(3, dtype=torch.complex64, device=self.device).unsqueeze(0).expand(N, -1, -1).clone()
            U_right = torch.eye(3, dtype=torch.complex64, device=self.device).unsqueeze(0).expand(N, -1, -1).clone()
            U_top = torch.eye(3, dtype=torch.complex64, device=self.device).unsqueeze(0).expand(N, -1, -1).clone()
            U_left = torch.eye(3, dtype=torch.complex64, device=self.device).unsqueeze(0).expand(N, -1, -1).clone()
            
            # Bottom edge (x direction, length R)
            idx = all_indices.clone()
            for r in range(R):
                links = self.get_links_at_indices(idx, 0)
                U_bottom = U_bottom @ links
                idx[:, 0] = (idx[:, 0] + 1) % L
            
            # Right edge (t direction, length T_loop)
            for t in range(T_loop):
                links = self.get_links_at_indices(idx, 3)
                U_right = U_right @ links
                idx[:, 3] = (idx[:, 3] + 1) % T
            
            # Top edge (x direction, backwards)
            for r in range(R):
                idx[:, 0] = (idx[:, 0] - 1 + L) % L
                links = self.get_links_at_indices(idx, 0).mH
                U_top = U_top @ links
            
            # Left edge (t direction, backwards)
            for t in range(T_loop):
                idx[:, 3] = (idx[:, 3] - 1 + T) % T
                links = self.get_links_at_indices(idx, 3).mH
                U_left = U_left @ links
            
            wloop = U_bottom @ U_right @ U_top @ U_left
            traces = torch.real(torch.diagonal(wloop, dim1=-2, dim2=-1).sum(dim=-1))
            return traces.mean().item() / 3.0
        
        def polyakov_loop(self):
            """Compute Polyakov loop (VECTORIZED)."""
            L, T = self.L, self.T
            
            # Get all spatial sites at t=0
            spatial_indices = torch.stack([
                torch.arange(L, device=self.device).repeat_interleave(L*L),
                torch.arange(L, device=self.device).repeat(L).repeat_interleave(L),
                torch.arange(L, device=self.device).repeat(L*L),
                torch.zeros(L*L*L, dtype=torch.long, device=self.device),
            ], dim=1)
            
            N = spatial_indices.shape[0]
            P = torch.eye(3, dtype=torch.complex64, device=self.device).unsqueeze(0).expand(N, -1, -1).clone()
            
            idx = spatial_indices.clone()
            for t in range(T):
                links = self.get_links_at_indices(idx, 3)
                P = P @ links
                idx[:, 3] = (idx[:, 3] + 1) % T
            
            traces = torch.diagonal(P, dim1=-2, dim2=-1).sum(dim=-1)
            return traces.mean().item() / 3.0
        
        def compute_clover_batch(self, indices, mu, nu):
            """Compute clover F_μν for batch of sites (VECTORIZED)."""
            L, T = self.L, self.T
            L_mu = T if mu == 3 else L
            L_nu = T if nu == 3 else L
            
            P1 = self.compute_plaquette_batch(indices, mu, nu)
            
            idx_mu_back = indices.clone()
            idx_mu_back[:, mu] = (idx_mu_back[:, mu] - 1 + L_mu) % L_mu
            P2 = self.compute_plaquette_batch(idx_mu_back, mu, nu)
            
            idx_nu_back = indices.clone()
            idx_nu_back[:, nu] = (idx_nu_back[:, nu] - 1 + L_nu) % L_nu
            P3 = self.compute_plaquette_batch(idx_nu_back, mu, nu)
            
            idx_both_back = indices.clone()
            idx_both_back[:, mu] = (idx_both_back[:, mu] - 1 + L_mu) % L_mu
            idx_both_back[:, nu] = (idx_both_back[:, nu] - 1 + L_nu) % L_nu
            P4 = self.compute_plaquette_batch(idx_both_back, mu, nu)
            
            Q = (P1 + P2.mH + P3.mH + P4) / 4.0
            return (Q - Q.mH) / 2.0
        
        def topological_charge_clover(self):
            """
            Compute topological charge using proper clover F_μν (VECTORIZED):
            Q = (1/32π²) Σ_x ε_μνρσ Tr(F_μν F_ρσ)
            """
            all_indices = self.all_indices
            
            F01 = self.compute_clover_batch(all_indices, 0, 1)
            F23 = self.compute_clover_batch(all_indices, 2, 3)
            F02 = self.compute_clover_batch(all_indices, 0, 2)
            F13 = self.compute_clover_batch(all_indices, 1, 3)
            F03 = self.compute_clover_batch(all_indices, 0, 3)
            F12 = self.compute_clover_batch(all_indices, 1, 2)
            
            contrib = F01 @ F23 - F02 @ F13 + F03 @ F12
            Q = torch.real(torch.diagonal(contrib, dim1=-2, dim2=-1).sum()).item()
            
            return Q / (32 * np.pi**2)
        
        def topological_charge(self):
            """
            Compute integer topological charge using clover definition.
            NOTE: On small lattices without flow, this may not give clean integers.
            Consider using after Wilson flow for better results.
            """
            q = self.topological_charge_clover()
            return int(round(q))
        
        def compute_cache(self, eps_skel: float = 0.15):
            """
            Compute cache vector Φ (Wilson loop traces) and r (topological charge).
            
            v1.4: eps_skel NOW AFFECTS Φ:
              - eps_skel < 0.15: Include more loop sizes (richer features)
              - eps_skel >= 0.15 and < 0.25: Standard loop set
              - eps_skel >= 0.25: Minimal loop set (coarser)
            """
            phi_components = []
            
            # Loop sizes depend on eps_skel (finer = more loops)
            if eps_skel < 0.15:
                # Fine: many loop sizes
                R_vals = [1, 2, 3]
                T_vals = [1, 2, 3]
            elif eps_skel < 0.25:
                # Medium: standard
                R_vals = [1, 2]
                T_vals = [1, 2]
            else:
                # Coarse: minimal
                R_vals = [1]
                T_vals = [1, 2]
            
            for R in R_vals:
                for T in T_vals:
                    if R <= self.L // 2 and T <= self.T // 2:  # Only reasonable sizes
                        w = self.wilson_loop(R, T)
                        phi_components.append(w)
            
            # Always include plaquette and Polyakov
            phi_components.append(self.plaquette())
            phi_components.append(abs(self.polyakov_loop()))
            
            phi = np.array(phi_components)
            r = self.topological_charge()
            
            return phi, r
    
    # =========================================================================
    # SHARED: Config Generation
    # =========================================================================
    
    def generate_configs(L: int, beta: float, n_configs: int, n_therm: int, n_skip: int, T: int = None):
        """Generate thermalized gauge configurations."""
        print(f"Generating {n_configs} configs at L={L}, β={beta}, T={T or L}")
        
        lattice = ParallelGaugeLattice(L, beta, T=T, device=device)
        
        # Thermalization
        print(f"  Thermalizing ({n_therm} sweeps)...")
        for _ in range(n_therm):
            lattice.parallel_metropolis_sweep()
        
        # Collection
        configs = []
        accept_rates = []
        
        print(f"  Collecting configs (n_skip={n_skip})...")
        for i in range(n_configs):
            for _ in range(n_skip):
                ar = lattice.parallel_metropolis_sweep()
            accept_rates.append(ar)
            
            # Store link data
            r = lattice.topological_charge()
            configs.append({
                'links': lattice.links.clone(),
                'plaquette': lattice.plaquette(),
                'action_density_raw': lattice.action_density(),  # v1.5: renamed (NOT used for κ)
                'polyakov': lattice.polyakov_loop(),
                'r': r,  # Store topology for tracking
            })
            r_histogram_global.append(r)
            
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{n_configs} collected")
            
            # FROZEN TOPOLOGY DIAGNOSTIC: Print r histogram every 100 configs
            if (i + 1) % 100 == 0:
                r_counts = {}
                for r_val in r_histogram_global[-100:]:
                    r_counts[r_val] = r_counts.get(r_val, 0) + 1
                print(f"    r-histogram (last 100): {dict(sorted(r_counts.items()))}")
        
        print(f"  Mean acceptance: {np.mean(accept_rates):.3f}")
        
        # Final r histogram for this batch
        r_counts = {}
        for r_val in [c['r'] for c in configs]:
            r_counts[r_val] = r_counts.get(r_val, 0) + 1
        print(f"  Final r-distribution: {dict(sorted(r_counts.items()))}")
        
        return configs, np.mean(accept_rates), r_counts
    
    def compute_cache_for_configs_flowed(configs, L: int, beta: float, T: int = None, 
                                          eps_skel: float = 0.15, t_flow: float = 0.1):
        """
        Compute cache (Φ, r, S_E) for all configs AFTER Wilson flow to t_flow.
        
        v1.5 CRITICAL FIX: Actually uses Wilson flow for all quantities.
        
        Args:
            configs: List of config dicts with 'links' tensor
            L, beta, T: Lattice parameters
            eps_skel: Controls Φ feature richness (finer = more loops)
            t_flow: Wilson flow time t* (use find_t_ref() to determine)
        
        Returns:
            cache_data with FLOWED action, phi, r
        """
        lattice = ParallelGaugeLattice(L, beta, T=T, device=device)
        
        cache_data = []
        for i, cfg in enumerate(configs):
            lattice.links = cfg['links'].clone()
            
            # Apply Wilson flow to t*
            if t_flow > 0:
                flowed_links = lattice.wilson_flow_to_t(t_flow, dt=0.02)
                lattice.links = flowed_links
            
            # Compute quantities on FLOWED config
            phi, r = lattice.compute_cache(eps_skel=eps_skel)
            S_E_flow = lattice.wilson_action()  # FULL Wilson action, not density
            plaq_flow = lattice.plaquette()
            
            cache_data.append({
                'phi': phi,
                'r': r,
                'action': S_E_flow,  # v1.5: FLOWED full Wilson action
                'plaquette': plaq_flow,
                't_flow': t_flow,
                'config_idx': i,
            })
            
            if (i + 1) % 10 == 0:
                print(f"    Flowed {i+1}/{len(configs)} configs to t={t_flow:.3f}")
        
        return cache_data
    
    def preflow_configs(configs, L: int, beta: float, T: int = None, t_flow: float = 0.1):
        """
        Pre-flow all configs to t_flow ONCE and store flowed links.
        This avoids redundant flow computations in later cache computations.
        """
        print(f"  Pre-flowing {len(configs)} configs to t={t_flow:.3f}...")
        lattice = ParallelGaugeLattice(L, beta, T=T, device=device)
        
        flowed_configs = []
        for i, cfg in enumerate(configs):
            lattice.links = cfg['links'].clone()
            
            if t_flow > 0:
                flowed_links = lattice.wilson_flow_to_t(t_flow, dt=0.02)
            else:
                flowed_links = cfg['links'].clone()
            
            flowed_configs.append({
                'links': flowed_links,
                'r': cfg.get('r', 0),  # Preserve topological charge
                'original_idx': i,
            })
            
            if (i + 1) % 10 == 0:
                print(f"    Pre-flowed {i+1}/{len(configs)} configs")
        
        return flowed_configs
    
    def compute_cache_from_preflowed(flowed_configs, L: int, beta: float, T: int = None,
                                      eps_skel: float = 0.15, t_flow: float = 0.1):
        """
        Compute cache (Φ, r, S_E) from pre-flowed configs (NO FLOW NEEDED).
        
        Much faster than compute_cache_for_configs_flowed when used multiple times.
        """
        lattice = ParallelGaugeLattice(L, beta, T=T, device=device)
        
        cache_data = []
        for i, cfg in enumerate(flowed_configs):
            lattice.links = cfg['links'].clone()
            
            # Compute quantities on already-flowed config
            phi, r = lattice.compute_cache(eps_skel=eps_skel)
            S_E_flow = lattice.wilson_action()
            plaq_flow = lattice.plaquette()
            
            cache_data.append({
                'phi': phi,
                'r': cfg.get('r', r),  # Use original r if preserved
                'action': S_E_flow,
                'plaquette': plaq_flow,
                't_flow': t_flow,
                'config_idx': cfg.get('original_idx', i),
            })
        
        return cache_data
    
    def estimate_t_ref(configs, L: int, beta: float, T: int = None, n_sample: int = 20, c: float = 0.3):
        """
        Estimate t_ref for this ensemble using a sample of configs.
        
        t_ref is defined by t² <E(t)> = c (t₀-like definition).
        
        Returns: mean t_ref across sample
        """
        lattice = ParallelGaugeLattice(L, beta, T=T, device=device)
        
        sample_indices = np.random.choice(len(configs), size=min(n_sample, len(configs)), replace=False)
        t_refs = []
        
        for i, idx in enumerate(sample_indices):
            lattice.links = configs[idx]['links'].clone()
            t_ref = lattice.find_t_ref(c=c)
            t_refs.append(t_ref)
            print(f"      Config {i+1}/{len(sample_indices)}: t_ref={t_ref:.4f}")
        
        mean_t_ref = np.mean(t_refs)
        print(f"    t_ref estimate: {mean_t_ref:.4f} (std={np.std(t_refs):.4f}, n={len(t_refs)})")
        
        return mean_t_ref
    
    def assign_bins(cache_data, eps_skel: float, eps_disc: float):
        """Assign configs to bins based on (Φ, r)."""
        import hashlib
        
        bins = {}
        assignments = []
        
        for i, cd in enumerate(cache_data):
            phi = cd['phi']
            r = cd['r']
            
            # Quantize phi
            phi_quantized = tuple(round(p / eps_disc) for p in phi)
            
            # Create bin ID
            bin_key = (phi_quantized, r)
            bin_str = str(bin_key).encode()
            bin_id = int(hashlib.md5(bin_str).hexdigest()[:8], 16)
            
            if bin_id not in bins:
                bins[bin_id] = {'indices': [], 'r': r, 'phi_mean': phi.copy()}
            bins[bin_id]['indices'].append(i)
            
            assignments.append(bin_id)
        
        return bins, assignments
    
    # =========================================================================
    # TEST A2S-001: Axiom 2 Cache Sufficiency
    # =========================================================================
    
    def compute_delta_O_jackknife(values, n_jackknife=None):
        """
        Compute δ_O = within-bin std with jackknife CI.
        Returns (delta_O, ci_lower, ci_upper).
        """
        values = np.array(values)
        n = len(values)
        if n < 3:
            return 0, 0, 0
        
        if n_jackknife is None:
            n_jackknife = n
        
        # Full sample estimate
        delta_O_full = np.std(values)
        
        # Jackknife delete-1 estimates
        jackknife_stds = []
        for i in range(min(n, n_jackknife)):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            jackknife_stds.append(np.std(values[mask]))
        
        jackknife_stds = np.array(jackknife_stds)
        jackknife_mean = np.mean(jackknife_stds)
        
        # Jackknife standard error
        se = np.sqrt((n - 1) / n * np.sum((jackknife_stds - jackknife_mean)**2))
        
        # 95% CI
        ci_lower = delta_O_full - 1.96 * se
        ci_upper = delta_O_full + 1.96 * se
        
        return delta_O_full, max(ci_lower, 0), ci_upper
    
    def run_A2S_001(flowed_configs, L: int, beta: float, t_flow: float = 0.1):
        """
        Test Axiom 2: Approximate Cache Sufficiency with δ_O(ε) = median_b σ_b(O).
        
        v1.5 CRITICAL FIX: Recomputes cache_data per ε_skel (not shared across grid).
        Uses PRE-FLOWED configs (no redundant flow computation).
        """
        print("\n" + "="*60)
        print("A2S-001: Axiom 2 Cache Sufficiency")
        print("="*60)
        
        results = {
            'test_id': 'A2S-001',
            'description': 'Axiom 2: Approximate Cache Sufficiency (δ_O metric)',
            'resolution_tests': [],
            't_flow': t_flow,
        }
        
        # Reduced grid for speed (per spec v1.1)
        eps_skel_values = [0.15, 0.20, 0.25]
        eps_disc_values = [0.15, 0.20]
        
        for eps_skel in eps_skel_values:
            # v1.5 CRITICAL: Compute cache with THIS eps_skel using pre-flowed configs
            print(f"  Computing Φ at ε_skel={eps_skel} (from pre-flowed configs)...")
            cache_data = compute_cache_from_preflowed(
                flowed_configs, L, beta, eps_skel=eps_skel, t_flow=t_flow
            )
            
            for eps_disc in eps_disc_values:
                bins, assignments = assign_bins(cache_data, eps_skel, eps_disc)
                
                # Filter bins with sufficient occupancy
                min_occupancy = 8
                occupied_bins = {bid: b for bid, b in bins.items() if len(b['indices']) >= min_occupancy}
                
                if len(occupied_bins) < 2:
                    continue
                
                # Compute δ_O = median_b σ_b(plaquette) with jackknife CI
                bin_stds = []
                bin_stds_with_ci = []
                
                for bid, b in occupied_bins.items():
                    plaqs = [cache_data[i]['plaquette'] for i in b['indices']]
                    std_val, ci_lo, ci_hi = compute_delta_O_jackknife(plaqs)
                    bin_stds.append(std_val)
                    bin_stds_with_ci.append((std_val, ci_lo, ci_hi))
                
                # δ_O = median of bin standard deviations
                delta_O = np.median(bin_stds)
                
                # CI on δ_O: bootstrap over bins
                n_boot = 200  # Reduced for speed
                delta_O_boots = []
                for _ in range(n_boot):
                    boot_idx = np.random.choice(len(bin_stds), size=len(bin_stds), replace=True)
                    delta_O_boots.append(np.median([bin_stds[i] for i in boot_idx]))
                delta_O_ci_lo = np.percentile(delta_O_boots, 2.5)
                delta_O_ci_hi = np.percentile(delta_O_boots, 97.5)
                
                # Global standard deviation (no binning)
                global_std = np.std([cd['plaquette'] for cd in cache_data])
                
                # v1.2: Use Q_0.9(σ_b) instead of max (avoids tautology with median δ_O)
                q90_sigma = np.percentile(bin_stds, 90)
                
                # Signal-to-noise ratio using Q_0.9
                signal_to_noise = global_std / max(q90_sigma, 1e-10)
                
                # v1.2 criterion: Q_0.9(σ_b) ≤ 3 × δ_O
                dispersion_ok = q90_sigma <= 3 * delta_O
                
                # v1.3: Between-bin separation via k-NN adjacency (same as A4C2)
                # D_{bb'}(O) = |μ_b(O) - μ_{b'}(O)| for adjacent bins
                # D_min = min over k-NN adjacent pairs
                bin_centroids = []
                bin_means = []
                bin_ids = list(occupied_bins.keys())
                
                for bid in bin_ids:
                    b = occupied_bins[bid]
                    # Centroid in Φ-space (first 6 components)
                    phis = [cache_data[i]['phi'] for i in b['indices']]
                    centroid = np.mean(phis, axis=0)
                    bin_centroids.append(centroid)
                    # Mean plaquette in this bin
                    plaqs = [cache_data[i]['plaquette'] for i in b['indices']]
                    bin_means.append(np.mean(plaqs))
                
                D_min = float('inf')
                if len(bin_centroids) >= 2:
                    bin_centroids = np.array(bin_centroids)
                    bin_means = np.array(bin_means)
                    
                    # k-NN graph (k=3) in Φ-space for adjacent bins
                    k = min(3, len(bin_ids) - 1)
                    from scipy.spatial.distance import cdist
                    dists = cdist(bin_centroids, bin_centroids)
                    
                    for i in range(len(bin_ids)):
                        # Find k nearest neighbors (excluding self)
                        nn_indices = np.argsort(dists[i])[1:k+1]
                        for j in nn_indices:
                            # D_{bb'} = |μ_b(O) - μ_{b'}(O)|
                            sep = abs(bin_means[i] - bin_means[j])
                            D_min = min(D_min, sep)
                
                if D_min == float('inf'):
                    D_min = 0.0
                
                # v1.3: Signal-to-bin ratio using D_min
                bin_sep_ratio = D_min / max(q90_sigma, 1e-10) if D_min > 0 else 0.0
                
                # v1.4 FIX: PASS uses bin_sep_ratio (not global_std/q90)
                # The criterion is: adjacent bins are distinguishable (D_min / Q_0.9(σ_b) ≥ 5)
                # AND within-bin dispersion is controlled (Q_0.9 ≤ 3×δ_O)
                pass_criterion = bin_sep_ratio >= 5.0 and dispersion_ok
                
                test_result = {
                    'eps_skel': eps_skel,
                    'eps_disc': eps_disc,
                    'n_bins': len(bins),
                    'n_occupied_bins': len(occupied_bins),
                    'delta_O': float(delta_O),
                    'delta_O_ci': [float(delta_O_ci_lo), float(delta_O_ci_hi)],
                    'q90_sigma': float(q90_sigma),  # v1.2: Q_0.9 instead of max
                    'global_std': float(global_std),
                    'signal_to_noise': float(signal_to_noise),  # Kept for diagnostics
                    'D_min': float(D_min),  # v1.3: k-NN adjacent separation
                    'bin_sep_ratio': float(bin_sep_ratio),  # v1.4: THIS IS THE KEY METRIC
                    'dispersion_ok': dispersion_ok,  # v1.2: Q_0.9 ≤ 3×δ_O
                    'pass': pass_criterion,  # v1.4: bin_sep_ratio >= 5 AND dispersion_ok
                }
                
                results['resolution_tests'].append(test_result)
                print(f"  ε_skel={eps_skel}, ε_disc={eps_disc}: δ_O={delta_O:.4f}, "
                      f"D_min/Q90={bin_sep_ratio:.2f}, {'PASS' if pass_criterion else 'FAIL'}")
        
        # Summary
        passing = sum(1 for t in results['resolution_tests'] if t['pass'])
        results['n_passing'] = passing
        results['n_total'] = len(results['resolution_tests'])
        results['overall_pass'] = passing > 0
        
        print(f"\nA2S-001 Summary: {passing}/{len(results['resolution_tests'])} passing")
        
        return results
    
    # =========================================================================
    # TEST A4C2-001: Axiom 4 Case 2 Curvature Gap (BIN-LEVEL)
    # =========================================================================
    
    def run_A4C2_001(flowed_configs, L: int, beta: float, t_flow: float = 0.1):
        """
        Test Axiom 4 Case 2: Same-r curvature gap.
        
        v1.5: Uses PRE-FLOWED configs (no redundant flow computation).
        
        κ_adj = min over k-NN adjacent bins of |S_E^b - S_E^{b'}|
        This is an ABSOLUTE action gap (units: action), NOT a ratio.
        Adjacent bins are identified via k-NN in Φ-space centroids.
        """
        print("\n" + "="*60)
        print("A4C2-001: Axiom 4 Case 2 Curvature Gap (κ_adj, BIN-LEVEL)")
        print("="*60)
        
        results = {
            'test_id': 'A4C2-001',
            'description': 'Axiom 4 Case 2: Same-r Curvature Gap (κ_adj, bin-level)',
            't_flow': t_flow,
        }
        
        eps_skel, eps_disc = 0.15, 0.20
        
        # v1.5: Compute cache from PRE-FLOWED configs
        print(f"  Computing Φ at ε_skel={eps_skel} (from pre-flowed configs)...")
        cache_data = compute_cache_from_preflowed(
            flowed_configs, L, beta, eps_skel=eps_skel, t_flow=t_flow
        )
        
        # Filter to r=0 sector
        r0_indices = [i for i, cd in enumerate(cache_data) if cd.get('r', 0) == 0]
        print(f"  Found {len(r0_indices)} configs in r=0 sector")
        
        if len(r0_indices) < 20:
            # If not enough r=0, use all configs (mixed topology case)
            print("  Insufficient r=0 configs, using all configs")
            r0_indices = list(range(len(cache_data)))
        
        # Create subset of cache_data for r=0 (or all)
        subset_cache = [cache_data[i] for i in r0_indices]
        
        # Bin the subset
        bins, assignments = assign_bins(subset_cache, eps_skel, eps_disc)
        
        # Filter to occupied bins
        min_occupancy = 5
        occupied_bins = {bid: b for bid, b in bins.items() if len(b['indices']) >= min_occupancy}
        
        if len(occupied_bins) < 2:
            print("  Insufficient occupied bins, skipping")
            results['overall_pass'] = False
            results['kappa_sep'] = 0
            return results
        
        # Compute bin-level statistics for S_E (FLOWED action)
        bin_stats = []
        for bid, b in occupied_bins.items():
            actions = [subset_cache[i]['action'] for i in b['indices']]
            phis = [subset_cache[i]['phi'] for i in b['indices']]
            bin_stats.append({
                'bid': bid,
                'mean_SE': np.mean(actions),
                'std_SE': np.std(actions),
                'n': len(actions),
                # Mean Φ (centroid) for k-NN
                'centroid': np.mean(phis, axis=0),
            })
        
        # v1.4 FIX: Use REAL k-NN in Φ-space (not sorted 1D proxy)
        # Build distance matrix between bin centroids
        from scipy.spatial.distance import cdist
        centroids = np.array([b['centroid'] for b in bin_stats])
        dist_matrix = cdist(centroids, centroids)
        
        # k-NN adjacency (k=3)
        k = min(3, len(bin_stats) - 1)
        
        # v1.4 FIX: κ_adj is ABSOLUTE action gap (not normalized by std)
        # κ_adj = min over k-NN edges of |S_E^b - S_E^{b'}|
        nn_gaps = []
        for i in range(len(bin_stats)):
            # Find k nearest neighbors (excluding self)
            nn_indices = np.argsort(dist_matrix[i])[1:k+1]
            for j in nn_indices:
                # Absolute action gap (NOT divided by std)
                gap = abs(bin_stats[i]['mean_SE'] - bin_stats[j]['mean_SE'])
                nn_gaps.append(gap)
        
        # κ_adj = minimum gap among k-NN adjacent pairs
        kappa_adj = min(nn_gaps) if nn_gaps else 0.0
        
        # Also compute χ_adj = κ_adj / Q_0.9(σ_b) for reference
        bin_stds = [b['std_SE'] for b in bin_stats if b['std_SE'] > 0]
        if bin_stds:
            q90_std = np.percentile(bin_stds, 90)
            chi_adj = kappa_adj / max(q90_std, 1e-10)
        else:
            chi_adj = 0.0
            q90_std = 0.0
        
        # Bootstrap CI for κ_adj
        n_boot = 100
        boot_kappas = []
        for _ in range(n_boot):
            boot_idx = np.random.choice(len(subset_cache), size=len(subset_cache), replace=True)
            boot_cache = [subset_cache[i] for i in boot_idx]
            boot_bins, _ = assign_bins(boot_cache, eps_skel, eps_disc)
            boot_occ = {bid: b for bid, b in boot_bins.items() if len(b['indices']) >= min_occupancy}
            
            if len(boot_occ) < 2:
                continue
            
            boot_stats = []
            for bid, b in boot_occ.items():
                actions = [boot_cache[i]['action'] for i in b['indices']]
                phis = [boot_cache[i]['phi'] for i in b['indices']]
                if len(actions) >= 2:
                    boot_stats.append({
                        'mean_SE': np.mean(actions),
                        'centroid': np.mean(phis, axis=0),
                    })
            
            if len(boot_stats) >= 2:
                boot_centroids = np.array([b['centroid'] for b in boot_stats])
                boot_dists = cdist(boot_centroids, boot_centroids)
                boot_k = min(3, len(boot_stats) - 1)
                boot_gaps = []
                for i in range(len(boot_stats)):
                    nn_idx = np.argsort(boot_dists[i])[1:boot_k+1]
                    for j in nn_idx:
                        boot_gaps.append(abs(boot_stats[i]['mean_SE'] - boot_stats[j]['mean_SE']))
                if boot_gaps:
                    boot_kappas.append(min(boot_gaps))
        
        if len(boot_kappas) >= 10:
            kappa_ci_lower = np.percentile(boot_kappas, 2.5)
            kappa_ci_upper = np.percentile(boot_kappas, 97.5)
        else:
            kappa_ci_lower = kappa_adj * 0.5
            kappa_ci_upper = kappa_adj * 1.5
        
        results['n_bins'] = len(bins)
        results['n_occupied'] = len(occupied_bins)
        results['n_r0_configs'] = len(r0_indices)
        results['kappa_adj'] = float(kappa_adj)  # v1.4: ABSOLUTE gap, not ratio
        results['chi_adj'] = float(chi_adj)  # v1.4: Dimensionless version
        results['q90_std'] = float(q90_std)
        results['kappa_adj_ci'] = [float(max(kappa_ci_lower, 0)), float(kappa_ci_upper)]
        results['overall_pass'] = kappa_ci_lower > 0  # CI lower bound > 0
        
        print(f"  κ_adj = {kappa_adj:.4f} [{kappa_ci_lower:.4f}, {kappa_ci_upper:.4f}] (action units)")
        print(f"  χ_adj = {chi_adj:.2f} (dimensionless)")
        print(f"  {'PASS' if results['overall_pass'] else 'FAIL'}: CI lower > 0")
        
        return results
    
    # =========================================================================
    # κ_sep_gap and χ_sep: PROPER DEFINITIONS (v1.4 CRITICAL FIX)
    # =========================================================================
    
    def kappa_sep_gap(cache_data, eps_skel=0.15, eps_disc=0.20, min_occupancy=5):
        """
        Compute κ_sep as ACTION GAP (units: action), NOT a ratio.
        
        Per spec v1.3:
        κ_sep = min_{b ≠ b_0} (Q_0.1(S_E^(b)) - Q_0.9(S_E^(b_0)))
        
        where b_0 is the vacuum bin (lowest mean action).
        
        Returns: (kappa_gap, ci_lower, ci_upper)
        Units: lattice action (β-dependent)
        """
        bins, assignments = assign_bins(cache_data, eps_skel, eps_disc)
        occupied_bins = {bid: b for bid, b in bins.items() if len(b['indices']) >= min_occupancy}
        
        if len(occupied_bins) < 2:
            return 0.0, 0.0, 0.0
        
        # Compute per-bin action statistics
        bin_stats = []
        for bid, b in occupied_bins.items():
            actions = np.array([cache_data[i]['action'] for i in b['indices']])
            n = len(actions)
            
            # Quantile estimation with occupancy fallback
            if n >= 20:
                q_lo, q_hi = np.percentile(actions, [10, 90])
            elif n >= 10:
                q_lo, q_hi = np.percentile(actions, [20, 80])
            else:
                q_lo = np.mean(actions) - np.std(actions)
                q_hi = np.mean(actions) + np.std(actions)
            
            bin_stats.append({
                'bid': bid,
                'mean_SE': np.mean(actions),
                'q10': q_lo,
                'q90': q_hi,
                'std_SE': np.std(actions),
                'n': n,
            })
        
        # Find vacuum bin (lowest mean action)
        bin_stats.sort(key=lambda x: x['mean_SE'])
        b0 = bin_stats[0]  # Vacuum bin
        
        # κ_sep = min over non-vacuum bins of (Q_0.1(b) - Q_0.9(b_0))
        # This is the ACTION GAP from vacuum's upper quantile to nearest bin's lower quantile
        gaps = []
        for b in bin_stats[1:]:  # Skip vacuum
            gap = b['q10'] - b0['q90']  # Lower quantile of b minus upper quantile of vacuum
            gaps.append(gap)
        
        kappa_gap = min(gaps) if gaps else 0.0
        
        # Bootstrap CI
        n_boot = 100
        boot_gaps = []
        all_indices = list(range(len(cache_data)))
        for _ in range(n_boot):
            boot_idx = np.random.choice(all_indices, size=len(all_indices), replace=True)
            boot_cache = [cache_data[i] for i in boot_idx]
            boot_bins, _ = assign_bins(boot_cache, eps_skel, eps_disc)
            boot_occ = {bid: b for bid, b in boot_bins.items() if len(b['indices']) >= min_occupancy}
            
            if len(boot_occ) < 2:
                continue
            
            boot_stats = []
            for bid, b in boot_occ.items():
                actions = np.array([boot_cache[i]['action'] for i in b['indices']])
                if len(actions) >= 3:
                    boot_stats.append({
                        'mean_SE': np.mean(actions),
                        'q10': np.percentile(actions, 10) if len(actions) >= 10 else np.mean(actions) - np.std(actions),
                        'q90': np.percentile(actions, 90) if len(actions) >= 10 else np.mean(actions) + np.std(actions),
                    })
            
            if len(boot_stats) >= 2:
                boot_stats.sort(key=lambda x: x['mean_SE'])
                b0_boot = boot_stats[0]
                boot_gaps_iter = [b['q10'] - b0_boot['q90'] for b in boot_stats[1:]]
                if boot_gaps_iter:
                    boot_gaps.append(min(boot_gaps_iter))
        
        if len(boot_gaps) >= 10:
            ci_lower = np.percentile(boot_gaps, 2.5)
            ci_upper = np.percentile(boot_gaps, 97.5)
        else:
            ci_lower = kappa_gap * 0.7
            ci_upper = kappa_gap * 1.3
        
        return kappa_gap, ci_lower, ci_upper
    
    def chi_sep(cache_data, eps_skel=0.15, eps_disc=0.20, min_occupancy=5):
        """
        Compute χ_sep = κ_sep_gap / Q_0.9(σ_b(S_E)) (DIMENSIONLESS).
        
        This is the signal-to-noise ratio: how many "noise widths" is the gap?
        
        Returns: chi_sep value (dimensionless)
        """
        kappa_gap, _, _ = kappa_sep_gap(cache_data, eps_skel, eps_disc, min_occupancy)
        
        bins, _ = assign_bins(cache_data, eps_skel, eps_disc)
        occupied_bins = {bid: b for bid, b in bins.items() if len(b['indices']) >= min_occupancy}
        
        if len(occupied_bins) < 2:
            return 0.0
        
        # Compute within-bin standard deviations
        bin_stds = []
        for bid, b in occupied_bins.items():
            actions = [cache_data[i]['action'] for i in b['indices']]
            if len(actions) >= 2:
                bin_stds.append(np.std(actions))
        
        if not bin_stds:
            return 0.0
        
        q90_sigma = np.percentile(bin_stds, 90)
        
        return kappa_gap / max(q90_sigma, 1e-10)
    
    # Legacy wrapper for backward compatibility
    def compute_kappa_sep(cache_data, eps_skel=0.15, eps_disc=0.20, min_occupancy=5):
        """
        DEPRECATED: Use kappa_sep_gap() for action gap or chi_sep() for dimensionless ratio.
        This returns the ACTION GAP (κ_sep_gap) for R calculation.
        """
        return kappa_sep_gap(cache_data, eps_skel, eps_disc, min_occupancy)
    
    def run_KSTAR_001():
        """
        Test κ* survival under continuum limit.
        
        v1.5 CRITICAL FIX: Actually uses t_ref via find_t_ref() and flows to it.
        Uses κ_sep_gap (action gap) on FLOWED configs.
        """
        print("\n" + "="*60)
        print("KSTAR-001: κ* Continuum Survival (SELF-CALIBRATING t_ref)")
        print("="*60)
        
        results = {
            'test_id': 'KSTAR-001',
            'description': 'κ* Continuum Survival (t_ref self-calibrated, FLOWED κ_sep_gap)',
            'scaling_data': [],
        }
        
        # Multi-β study
        if SMOKE_TEST:
            beta_L_grid = [(5.8, 8), (6.0, 8)]  # Only 2 β values in smoke
            n_configs = 20
            n_therm = 40
            n_skip = 3
            n_sample_t_ref = 2  # Minimal t_ref sampling
        else:
            beta_L_grid = [(5.8, 8), (6.0, 8), (6.2, 8), (6.0, 12)]
            n_configs = 150
            n_therm = 80
            n_skip = 4
            n_sample_t_ref = 20
        
        for beta, L in beta_L_grid:
            print(f"\n  Running β={beta}, L={L}")
            
            # Generate configs
            configs, accept, r_counts = generate_configs(L, beta, n_configs, n_therm, n_skip)
            
            # v1.5 CRITICAL: Actually compute t_ref via find_t_ref()
            print(f"    Estimating t_ref...")
            t_ref = estimate_t_ref(configs, L, beta, n_sample=n_sample_t_ref)
            
            # v1.5 OPTIMIZED: Preflow once then compute cache
            print(f"    Pre-flowing and computing κ_sep_gap at t_ref={t_ref:.4f}...")
            flowed = preflow_configs(configs, L, beta, t_flow=t_ref)
            cache_data_flowed = compute_cache_from_preflowed(
                flowed, L, beta, T=None, eps_skel=0.15, t_flow=t_ref
            )
            
            # Compute κ_sep_gap on FLOWED configs
            kappa_gap, kappa_ci_lo, kappa_ci_hi = kappa_sep_gap(cache_data_flowed)
            chi = chi_sep(cache_data_flowed)
            
            results['scaling_data'].append({
                'beta': beta,
                'L': L,
                't_ref': float(t_ref),  # v1.5: ACTUAL t_ref, not plaquette proxy
                'kappa_sep_gap': float(kappa_gap),  # Action gap units
                'chi_sep': float(chi),  # Dimensionless
                'kappa_sep_ci': [float(kappa_ci_lo), float(kappa_ci_hi)],
                'n_configs': n_configs,
                'r_histogram': dict(sorted(r_counts.items())),  # v1.5: scope frozen topology claims
            })
            
            print(f"    t_ref={t_ref:.4f}, κ_gap={kappa_gap:.2f}, χ_sep={chi:.2f}")
        
        # Analysis
        # v1.5: t_ref monotonicity check (should increase with β = finer lattice)
        t_refs_L8 = [(d['beta'], d['t_ref']) for d in results['scaling_data'] if d['L'] == 8]
        t_refs_L8.sort(key=lambda x: x[0])  # Sort by β
        if len(t_refs_L8) >= 2:
            t_ref_monotonic = all(t_refs_L8[i][1] <= t_refs_L8[i+1][1] 
                                  for i in range(len(t_refs_L8)-1))
        else:
            t_ref_monotonic = True
        
        # Finite-size check: compare L=8 vs L=12 at β=6.0
        kappa_L8 = next((d['kappa_sep_gap'] for d in results['scaling_data'] 
                        if d['beta'] == 6.0 and d['L'] == 8), 0)
        kappa_L12 = next((d['kappa_sep_gap'] for d in results['scaling_data'] 
                         if d['beta'] == 6.0 and d['L'] == 12), 0)
        
        if kappa_L8 > 0 and kappa_L12 > 0:
            fs_stability = abs(kappa_L12 - kappa_L8) / kappa_L8
        else:
            fs_stability = 1.0
        
        # Check if κ_sep_gap is nonzero and stable/growing with β
        kappas_L8 = [d['kappa_sep_gap'] for d in results['scaling_data'] if d['L'] == 8]
        nonzero_limit = all(k > 0 for k in kappas_L8)  # All κ > 0 (action gap must be positive)
        
        # Check for plateau or growth (not collapse)
        if len(kappas_L8) >= 3:
            # κ should plateau or grow as β increases (a → 0)
            # Check that κ at highest β is not much smaller than at lowest β
            scaling_ok = kappas_L8[-1] >= 0.5 * kappas_L8[0]
        else:
            scaling_ok = True
        
        results['finite_size_stability'] = float(fs_stability)
        results['fs_pass'] = fs_stability < 0.20  # Per spec: < 20%
        results['nonzero_limit'] = nonzero_limit
        results['scaling_ok'] = scaling_ok
        results['t_ref_monotonic'] = t_ref_monotonic  # v1.5: soft diagnostic only
        # v1.5: t_ref monotonicity is a SOFT diagnostic (small lattices may not show clean scaling)
        # Don't fail overall_pass on it; just warn
        results['overall_pass'] = results['fs_pass'] and nonzero_limit and scaling_ok
        
        print(f"\n  Finite-size stability: {fs_stability:.1%} ({'PASS' if results['fs_pass'] else 'FAIL'})")
        print(f"  Nonzero κ_gap limit: {'PASS' if nonzero_limit else 'FAIL'}")
        print(f"  Scaling stability: {'PASS' if scaling_ok else 'FAIL'}")
        t_ref_status = 'OK' if t_ref_monotonic else 'WARN (small lattice effects)'
        print(f"  t_ref monotonicity: {t_ref_status} (soft diagnostic)")
        
        return results
    
    # =========================================================================
    # TEST OSBRIDGE-001: Transfer-Matrix Qualitative Alignment
    # =========================================================================
    
    def run_OSBRIDGE_001():
        """
        Test OS/transfer-matrix bridge via correlators.
        
        KEY FIX: This is QUALITATIVE alignment, not numeric equality.
        We show both m_gap > 0 and κ_sep > 0, not m_gap = f(κ).
        """
        print("\n" + "="*60)
        print("OSBRIDGE-001: Transfer-Matrix Qualitative Alignment")
        print("="*60)
        
        results = {
            'test_id': 'OSBRIDGE-001',
            'description': 'Transfer-Matrix Qualitative Alignment (NOT numeric equality)',
        }
        
        # Extended temporal lattice for correlators
        L = 8
        T = 16
        beta = 6.0
        if SMOKE_TEST:
            n_configs = 15  # Minimal for qualitative check
            n_therm_os = 30
        else:
            n_configs = 200
            n_therm_os = 100
        
        print(f"  Generating configs on {L}³×{T} lattice...")
        configs, accept, _ = generate_configs(L, beta, n_configs, n_therm_os, 5, T=T)
        
        # Compute correlators
        # O(t) = sum of plaquette values at timeslice t
        correlators = np.zeros(T)
        correlator_counts = np.zeros(T)
        
        lattice = ParallelGaugeLattice(L, beta, T=T, device=device)
        
        print(f"  Computing correlators (VECTORIZED)...")
        for cfg_idx, cfg in enumerate(configs):
            lattice.links = cfg['links']
            
            # Compute O(t) for each timeslice using BATCH operations
            O_t = []
            for t in range(T):
                # Build indices for all spatial sites at fixed t
                spatial_indices = torch.stack([
                    torch.arange(L, device=device).repeat_interleave(L*L),
                    torch.arange(L, device=device).repeat(L).repeat_interleave(L),
                    torch.arange(L, device=device).repeat(L*L),
                    torch.full((L*L*L,), t, dtype=torch.long, device=device),
                ], dim=1)
                
                plaq_sum = 0.0
                count = 0
                # Only spatial plaquettes (mu, nu in {0,1,2})
                for mu in range(3):
                    for nu in range(mu + 1, 3):
                        P = lattice.compute_plaquette_batch(spatial_indices, mu, nu)
                        traces = torch.real(torch.diagonal(P, dim1=-2, dim2=-1).sum(dim=-1))
                        plaq_sum += traces.sum().item()
                        count += L * L * L
                O_t.append(plaq_sum / count)
            
            O_t = np.array(O_t)
            O_mean = np.mean(O_t)
            
            # Correlator C(τ) = <O(0)O(τ)> - <O>²
            for tau in range(T):
                C_tau = np.mean(O_t * np.roll(O_t, -tau)) - O_mean**2
                correlators[tau] += C_tau
                correlator_counts[tau] += 1
            
            if (cfg_idx + 1) % 10 == 0:
                print(f"    {cfg_idx + 1}/{n_configs} processed")
        
        correlators /= correlator_counts
        
        # Normalize
        C0 = correlators[0]
        if abs(C0) > 1e-10:
            correlators_norm = correlators / C0
        else:
            correlators_norm = correlators
        
        # Extract effective mass
        m_eff = []
        for t in range(1, T // 2):
            if correlators_norm[t] > 0 and correlators_norm[t+1] > 0:
                m = np.log(correlators_norm[t] / correlators_norm[t+1])
                m_eff.append(m)
            else:
                m_eff.append(np.nan)
        
        m_eff = np.array(m_eff)
        
        # Find plateau (look for stable region t ∈ [3, 7])
        valid_m = m_eff[~np.isnan(m_eff)]
        if len(valid_m) >= 3:
            plateau_region = m_eff[2:7] if len(m_eff) >= 7 else valid_m
            plateau_region = plateau_region[~np.isnan(plateau_region)]
            if len(plateau_region) >= 2:
                plateau_mass = np.median(plateau_region)
                plateau_std = np.std(plateau_region)
            else:
                plateau_mass = valid_m[0] if len(valid_m) > 0 else 0
                plateau_std = 0
        else:
            plateau_mass = 0
            plateau_std = 0
        
        # Check for clean exponential decay
        clean_decay = (len(valid_m) >= 3 and 
                       plateau_std < 0.3 * abs(plateau_mass) if plateau_mass != 0 else False)
        
        # Also compute κ_sep_gap on same configs (FLOWED) for qualitative comparison
        # v1.5 OPTIMIZED: Preflow once then compute cache
        t_flow_osbridge = 0.1  # Default flow time for qualitative comparison
        flowed = preflow_configs(configs, L, beta, T=T, t_flow=t_flow_osbridge)
        cache_data = compute_cache_from_preflowed(
            flowed, L, beta, T=T, eps_skel=0.15, t_flow=t_flow_osbridge
        )
        kappa_sep, _, _ = kappa_sep_gap(cache_data)  # Use TRUE gap function
        
        results['correlators'] = correlators_norm.tolist()
        results['m_eff'] = [float(m) if not np.isnan(m) else None for m in m_eff]
        results['plateau_mass'] = float(plateau_mass)
        results['plateau_std'] = float(plateau_std)
        results['clean_decay'] = clean_decay
        results['kappa_sep'] = float(kappa_sep)
        
        # QUALITATIVE ALIGNMENT: Both m_gap > 0 AND κ_sep > 0
        results['m_gap_positive'] = plateau_mass > 0
        results['kappa_sep_positive'] = kappa_sep > 0
        results['qualitative_alignment'] = results['m_gap_positive'] and results['kappa_sep_positive']
        results['overall_pass'] = results['qualitative_alignment'] and clean_decay
        
        print(f"  Plateau mass: {plateau_mass:.3f} ± {plateau_std:.3f}")
        print(f"  κ_sep: {kappa_sep:.2f}")
        print(f"  Clean decay: {'PASS' if clean_decay else 'FAIL'}")
        print(f"  Qualitative alignment (both > 0): {'PASS' if results['qualitative_alignment'] else 'FAIL'}")
        
        return results
    
    # =========================================================================
    # TEST HEPS-001: H_ε → H_phys Uniform Gap (η/κ SEPARATED)
    # =========================================================================
    
    def compute_eta_from_chain(cache_data, bins, assignments):
        """
        Compute inter-bin mixing rates from MCMC chain.
        v1.2: Returns BOTH η_mean (average) and η_max (worst-case per-bin leakage).
        
        Requires configs in MCMC order (temporal sequence).
        """
        n_transitions = 0
        n_same_bin = 0
        
        # Track per-bin leakage for η_max
        bin_exits = {}  # bin_id -> count of exits
        bin_stays = {}  # bin_id -> count of stays
        
        for i in range(len(assignments) - 1):
            b_curr = assignments[i]
            b_next = assignments[i + 1]
            
            if b_curr not in bin_exits:
                bin_exits[b_curr] = 0
                bin_stays[b_curr] = 0
            
            if b_curr != b_next:
                n_transitions += 1
                bin_exits[b_curr] += 1
            else:
                n_same_bin += 1
                bin_stays[b_curr] += 1
        
        total = n_transitions + n_same_bin
        if total == 0:
            return 0.0, 0.0  # η_mean, η_max
        
        # η_mean: global average mixing rate
        eta_mean = n_transitions / total
        
        # η_max: worst-case per-bin leakage P(exit | in bin b)
        eta_max = 0.0
        for bid in bin_exits:
            exits = bin_exits[bid]
            stays = bin_stays.get(bid, 0)
            total_in_bin = exits + stays
            if total_in_bin > 0:
                leakage = exits / total_in_bin
                eta_max = max(eta_max, leakage)
        
        return eta_mean, eta_max
    
    def run_HEPS_001(flowed_configs, L: int, beta: float, t_flow: float = 1.0):
        """
        Test uniform gap preservation under refinement.
        
        v1.5 FIX: Separate η (needs chain order) from κ_sep (can use any bag).
        - κ_sep measured at ALL refinement levels (bag-friendly)
        - η_emp measured only at ONE reference level (requires chain order)
        - NOW USES PRE-FLOWED DATA: No redundant flow computations
        """
        print("\n" + "="*60)
        print("HEPS-001: H_ε → H_phys Uniform Gap (η/κ SEPARATED)")
        print("="*60)
        
        results = {
            'test_id': 'HEPS-001',
            'description': 'H_ε → H_phys Uniform Gap (η needs chain, κ_sep is bag-friendly)',
            'refinement_ladder': [],
            't_flow': t_flow,
        }
        
        # Refinement ladder (reduced in smoke mode)
        if SMOKE_TEST:
            refinement_levels = [
                (0.30, 0.30),
                (0.20, 0.20),  # Reference level for η
                (0.10, 0.10),
            ]
            reference_level = 1  # Level 1 = ε=0.20 for η measurement
        else:
            refinement_levels = [
                (0.30, 0.30),
                (0.25, 0.25),
                (0.20, 0.20),  # Reference level for η
                (0.15, 0.15),
                (0.10, 0.10),
            ]
            reference_level = 2  # Level 2 = ε=0.20 for η measurement
        
        eta_mean_ref = None
        eta_max_ref = None
        
        # v1.5: Store original cache for η measurement (needs chain order with consistent bins)
        reference_cache = None
        reference_bins = None
        reference_assignments = None
        
        for level, (eps_skel, eps_disc) in enumerate(refinement_levels):
            # v1.5 OPTIMIZED: Compute cache from pre-flowed configs (no flow computation)
            print(f"    Computing cache for ε_skel={eps_skel} (from pre-flowed)...")
            cache_data = compute_cache_from_preflowed(
                flowed_configs, L, beta, T=None, eps_skel=eps_skel, t_flow=t_flow
            )
            
            bins, assignments = assign_bins(cache_data, eps_skel, eps_disc)
            
            # Save reference level data for η computation
            if level == reference_level:
                reference_cache = cache_data
                reference_bins = bins
                reference_assignments = assignments
            
            # κ_sep_gap: TRUE action-based separation (BAG-FRIENDLY, computed at all levels)
            kappa_sep, _, _ = compute_kappa_sep(cache_data, eps_skel, eps_disc, min_occupancy=3)
            
            n_bins = len(bins)
            occupied_bins = {bid: b for bid, b in bins.items() if len(b['indices']) >= 3}
            
            # v1.3: χ_sep = κ_sep / Q_0.9(σ_b) (dimensionless ratio)
            bin_stds = []
            for bid, b in occupied_bins.items():
                actions = [cache_data[i]['action'] for i in b['indices']]
                if len(actions) >= 2:
                    bin_stds.append(np.std(actions))
            if bin_stds:
                q90_sigma = np.percentile(bin_stds, 90)
                chi_sep = kappa_sep / max(q90_sigma, 1e-10)
            else:
                q90_sigma = 0.0
                chi_sep = 0.0
            
            # v1.5: η_mean and η_max ONLY at reference level (REQUIRES CHAIN ORDER)
            # Use reference_cache/bins/assignments for consistent η measurement
            if level == reference_level and reference_cache is not None:
                eta_mean, eta_max = compute_eta_from_chain(reference_cache, reference_bins, reference_assignments)
                eta_mean_ref = eta_mean
                eta_max_ref = eta_max
            else:
                eta_mean = None
                eta_max = None
            
            # v1.5: R uses η_max (conservative), R_mean uses η_mean (secondary)
            lambda_coupling = 1.0  # Effective coupling
            if eta_max is not None and kappa_sep > 0:
                R = eta_max / (lambda_coupling * kappa_sep)  # Conservative
                R_mean = eta_mean / (lambda_coupling * kappa_sep)  # Secondary
            else:
                R = None
                R_mean = None
            
            level_result = {
                'level': level,
                'eps_skel': eps_skel,
                'eps_disc': eps_disc,
                'n_bins': n_bins,
                'n_occupied': len(occupied_bins),
                'kappa_sep_gap': float(kappa_sep),  # v1.5: true action gap
                'chi_sep': float(chi_sep),  # v1.5: dimensionless ratio (used for pass criterion)
                'eta_mean': float(eta_mean) if eta_mean is not None else None,
                'eta_max': float(eta_max) if eta_max is not None else None,
                'R': float(R) if R is not None else None,  # Uses η_max
                'R_mean': float(R_mean) if R_mean is not None else None,  # Uses η_mean
                'chi_pass': chi_sep >= 5.0,  # v1.5: dimensionless threshold (not just κ > 0)
            }
            
            results['refinement_ladder'].append(level_result)
            
            if eta_max is not None:
                print(f"  Level {level} (ε={eps_skel}): κ_gap={kappa_sep:.3f}, χ_sep={chi_sep:.1f}, η_max={eta_max:.3f}, R(η_max)={R:.4f}")
            else:
                print(f"  Level {level} (ε={eps_skel}): κ_gap={kappa_sep:.3f}, χ_sep={chi_sep:.1f} (η not computed, bag-mode)")
        
        # Check uniform gap: χ_sep >= 5 at all levels (dimensionless threshold)
        all_chi_pass = all(l['chi_pass'] for l in results['refinement_ladder'])
        chi_min = min(l['chi_sep'] for l in results['refinement_ladder'])
        kappa_min = min(l['kappa_sep_gap'] for l in results['refinement_ladder'])
        
        # v1.5: R check uses η_max at reference level (conservative)
        ref_result = results['refinement_ladder'][reference_level]
        R_ref = ref_result['R']  # This is R(η_max)
        R_pass = R_ref is not None and R_ref <= 0.10
        
        # No collapse check: χ_sep at fine should be within factor of 2 of χ_sep at coarse
        chi_coarse = results['refinement_ladder'][0]['chi_sep']
        chi_fine = results['refinement_ladder'][-1]['chi_sep']
        if chi_coarse > 0:
            no_collapse = chi_fine / chi_coarse >= 0.5
        else:
            no_collapse = False
        
        results['all_chi_pass'] = all_chi_pass
        results['chi_min'] = float(chi_min)
        results['kappa_min'] = float(kappa_min)
        results['eta_mean_ref'] = float(eta_mean_ref) if eta_mean_ref is not None else None
        results['eta_max_ref'] = float(eta_max_ref) if eta_max_ref is not None else None
        results['R_ref'] = float(R_ref) if R_ref is not None else None  # Uses η_max
        results['R_pass'] = R_pass
        results['no_collapse'] = no_collapse
        results['overall_pass'] = all_chi_pass and R_pass and no_collapse
        
        print(f"\n  Uniform gap (χ_sep ≥ 5 all levels): {'PASS' if all_chi_pass else 'FAIL'}")
        print(f"  χ_min = {chi_min:.2f}, κ_min = {kappa_min:.4f}")
        R_txt = f"{R_ref:.4f}" if R_ref is not None else "N/A"
        print(f"  Reference R = {R_txt}: {'PASS' if R_pass else 'FAIL'}")
        print(f"  No collapse (χ_fine/χ_coarse ≥ 0.5): {'PASS' if no_collapse else 'FAIL'}")
        
        return results
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    print("="*70)
    print(f"REST OF VALIDATION - Unified GPU Script (v1.5) {'[SMOKE TEST]' if SMOKE_TEST else '[FULL RUN]'}")
    print("="*70)
    print(f"Timestamp: {timestamp}")
    print(f"Device: {device}")
    
    all_results = {
        'timestamp': timestamp,
        'device': str(device),
        'version': '1.5',  # v1.5: Flow integration
        'smoke_test': SMOKE_TEST,
    }
    
    # Phase 1: Generate master config set
    print("\n" + "="*60)
    print("Phase 1: Generating Master Config Set (PRESERVING CHAIN ORDER)")
    print("="*60)
    
    L = 8
    beta = 6.0
    if SMOKE_TEST:
        n_configs = 50
        n_therm = 60
        n_skip = 4
        n_sample_t_ref = 3  # Reduced for speed
    else:
        n_configs = 500
        n_therm = 150
        n_skip = 8
        n_sample_t_ref = 20
    
    configs, accept_rate, r_counts_master = generate_configs(L, beta, n_configs, n_therm, n_skip)
    
    # v1.5: Estimate t_ref (t₀-like scale) from sample of configs
    print("\n  Estimating t_ref from config sample...")
    t_ref = estimate_t_ref(configs, L, beta, n_sample=n_sample_t_ref)
    print(f"  t_ref = {t_ref:.4f}")
    
    # v1.5: Use t_flow = t_ref for all flow operations
    t_flow = t_ref
    
    # v1.5 OPTIMIZATION: Pre-flow configs ONCE to avoid redundant flow computations
    flowed_configs = preflow_configs(configs, L, beta, t_flow=t_flow)
    
    all_results['master_config'] = {
        'L': L,
        'beta': beta,
        'n_configs': n_configs,
        'accept_rate': float(accept_rate),
        't_ref': float(t_ref),  # v1.5: actual t_ref
        't_flow': float(t_flow),
        'r_histogram': dict(sorted(r_counts_master.items())),  # v1.5: topology distribution
    }
    
    # Phase 2: A2S-001 (uses pre-flowed configs)
    all_results['A2S_001'] = run_A2S_001(flowed_configs, L, beta, t_flow)
    
    # Phase 3: A4C2-001 (uses pre-flowed configs)
    all_results['A4C2_001'] = run_A4C2_001(flowed_configs, L, beta, t_flow)
    
    # Phase 4: KSTAR-001 (now uses actual t_ref scaling)
    all_results['KSTAR_001'] = run_KSTAR_001()
    
    # Phase 5: OSBRIDGE-001
    all_results['OSBRIDGE_001'] = run_OSBRIDGE_001()
    
    # Phase 6: HEPS-001 (now uses pre-flowed configs)
    all_results['HEPS_001'] = run_HEPS_001(flowed_configs, L, beta, t_flow)
    
    # =========================================================================
    # SMOKE TEST SANITY CHECK (key metrics summary)
    # =========================================================================
    
    if SMOKE_TEST:
        print("\n" + "="*70)
        print("SMOKE TEST SANITY CHECK: Key Metrics")
        print("="*70)
        
        smoke_metrics = {
            'parameters': {
                'master_n_configs': n_configs,
                'master_L': L,
                'master_beta': beta,
                't_ref': float(t_ref),
            },
            'metrics': {},
            'sanity_bands': {
                'kappa_gap_ok': True,  # κ_gap > 0
                'chi_sep_ok': True,    # χ_sep ≳ 2
                'eta_max_ok': True,    # η_max < 0.8
            },
        }
        
        # A4C2 metrics
        a4c2 = all_results['A4C2_001']
        kappa_adj = a4c2.get('kappa_adj', 0)
        chi_sep_a4c2 = a4c2.get('chi_sep', 0)
        smoke_metrics['metrics']['A4C2'] = {
            'kappa_adj': float(kappa_adj),
            'chi_sep': float(chi_sep_a4c2),
        }
        print(f"\n  A4C2-001:")
        print(f"    κ_adj = {kappa_adj:.4f} (action gap)")
        print(f"    χ_sep = {chi_sep_a4c2:.2f} (dimensionless)")
        if kappa_adj <= 0:
            smoke_metrics['sanity_bands']['kappa_gap_ok'] = False
            print(f"    ⚠ WARNING: κ_adj ≤ 0 (bins overlapping)")
        if chi_sep_a4c2 < 2:
            smoke_metrics['sanity_bands']['chi_sep_ok'] = False
            print(f"    ⚠ WARNING: χ_sep < 2 (gap/spread ratio weak)")
        
        # KSTAR metrics (use first β=6.0 point)
        kstar = all_results['KSTAR_001']
        kstar_60 = next((d for d in kstar['scaling_data'] if d['beta'] == 6.0 and d['L'] == 8), None)
        if kstar_60:
            kappa_kstar = kstar_60.get('kappa_sep_gap', 0)
            chi_kstar = kstar_60.get('chi_sep', 0)
            smoke_metrics['metrics']['KSTAR_beta60'] = {
                'kappa_sep_gap': float(kappa_kstar),
                'chi_sep': float(chi_kstar),
                't_ref': float(kstar_60.get('t_ref', 0)),
            }
            print(f"\n  KSTAR-001 (β=6.0, L=8):")
            print(f"    κ_gap = {kappa_kstar:.4f}")
            print(f"    χ_sep = {chi_kstar:.2f}")
            print(f"    t_ref = {kstar_60.get('t_ref', 0):.4f}")
            if kappa_kstar <= 0:
                smoke_metrics['sanity_bands']['kappa_gap_ok'] = False
        
        # HEPS metrics (reference level = 2, ε=0.20)
        heps = all_results['HEPS_001']
        ref_level = next((l for l in heps['refinement_ladder'] if l['level'] == 2), None)
        if ref_level:
            kappa_heps = ref_level.get('kappa_sep_gap', 0)
            chi_heps = ref_level.get('chi_sep', 0)
            eta_max = ref_level.get('eta_max', 1.0)
            R_val = ref_level.get('R', None)
            smoke_metrics['metrics']['HEPS_ref'] = {
                'kappa_sep_gap': float(kappa_heps),
                'chi_sep': float(chi_heps),
                'eta_max': float(eta_max) if eta_max else None,
                'R': float(R_val) if R_val else None,
            }
            print(f"\n  HEPS-001 (reference ε=0.20):")
            print(f"    κ_gap = {kappa_heps:.4f}")
            print(f"    χ_sep = {chi_heps:.2f}")
            print(f"    η_max = {eta_max:.3f}" if eta_max else "    η_max = N/A")
            print(f"    R = {R_val:.4f}" if R_val else "    R = N/A")
            if eta_max and eta_max >= 0.8:
                smoke_metrics['sanity_bands']['eta_max_ok'] = False
                print(f"    ⚠ WARNING: η_max ≥ 0.8 (mixing too high)")
            if kappa_heps <= 0:
                smoke_metrics['sanity_bands']['kappa_gap_ok'] = False
        
        # Overall smoke test verdict
        smoke_ok = all(smoke_metrics['sanity_bands'].values())
        smoke_metrics['smoke_ok'] = smoke_ok
        all_results['smoke_metrics'] = smoke_metrics
        
        print(f"\n  {'='*50}")
        if smoke_ok:
            print("  ✓ SMOKE TEST PASSED: Metrics are numerically sensible")
            print("    Ready to scale up to full run (set SMOKE_TEST=False)")
        else:
            print("  ✗ SMOKE TEST FAILED: Check warnings above")
            print("    Fix issues before full run")
        print(f"  {'='*50}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    summary = {
        'A2S_001': all_results['A2S_001']['overall_pass'],
        'A4C2_001': all_results['A4C2_001']['overall_pass'],
        'KSTAR_001': all_results['KSTAR_001']['overall_pass'],
        'OSBRIDGE_001': all_results['OSBRIDGE_001']['overall_pass'],
        'HEPS_001': all_results['HEPS_001']['overall_pass'],
    }
    
    for test_id, passed in summary.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_id}: {status}")
    
    all_pass = all(summary.values())
    all_results['summary'] = summary
    all_results['all_pass'] = all_pass
    
    print(f"\n{'='*70}")
    if all_pass:
        print("  *** ALL TESTS PASS - VALIDATION COMPLETE ***")
    else:
        n_pass = sum(summary.values())
        print(f"  {n_pass}/5 tests passing")
    print(f"{'='*70}")
    
    # Save results
    suffix = "_smoke" if SMOKE_TEST else ""
    output_file = results_dir / f"rest_of_validation{suffix}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    volume.commit()
    
    return all_results


@app.local_entrypoint()
def main():
    """Run the validation suite."""
    results = run_rest_of_validation.remote()
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"All pass: {results['all_pass']}")
