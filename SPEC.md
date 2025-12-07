# Technical Specification: Davis-Wilson Lattice Verification

## 1. Mathematical Definitions

### 1.1 Lattice Gauge Configuration

A lattice gauge configuration is a set of SU(3) matrices (link variables) on a 4D hypercubic lattice:

```
U_μ(x) ∈ SU(3)  for each site x and direction μ ∈ {0,1,2,3}
```

Storage format: `U[μ][t][x][y][z]` as complex 3×3 matrices.

For lattice size L⁴, total storage: `4 × L⁴ × 9 × 16 bytes` (9 complex128 per link)
- 8⁴ lattice: ~2.4 MB per config
- 16⁴ lattice: ~38 MB per config  
- 24⁴ lattice: ~190 MB per config

(Note: Using complex64 halves these values.)

### 1.2 Wilson Loop

A Wilson loop is the trace of the path-ordered product of link variables around a closed loop:

```
W(C) = Tr[ P ∏_{(x,μ)∈C} U_μ(x) ]
```

For a plaquette (elementary 1×1 loop) in the μν plane at site x:

```
P_μν(x) = U_μ(x) · U_ν(x+μ̂) · U_μ†(x+ν̂) · U_ν†(x)
W_μν(x) = Re[Tr(P_μν(x))]
```

### 1.3 Lüscher Topological Charge

The lattice topological charge using the Lüscher definition:

```
Q = (1/32π²) Σ_x ε_μνρσ Tr[P_μν(x) P_ρσ(x)]
```

In practice, use the "clover" or "5Li" improved definition for better continuum limit:

```
Q_clover = (1/32π²) Σ_x Tr[G̃_μν(x) G_μν(x)]
```

where G_μν is the clover-leaf field strength tensor.

### 1.4 Davis-Wilson Map

```
Γ: A/G → C = ℝ^{d_Φ} × ℤ
Γ(U) = (Φ(U), r(U))
```

**Continuous component Φ:**
- Select a "skeleton" of representative plaquettes/loops
- For each loop i in skeleton: Φ_{2i} = Re[W_i]/3, Φ_{2i+1} = Im[W_i]/3
- Dimension: d_Φ = 2 × (number of skeleton loops)
- **Normalization:** Divide traces by N=3 so |W| ≤ 1, making Φ comparable across volumes and β

**Discrete component r:**
- r = round(Q) where Q is the topological charge
- r ∈ ℤ (instanton number)

**Binning (quantization):**
- Quantizer: q_ε(Φ, r) = (Round_Λ(Φ), r) where Λ_ε = ε·ℤ^{d_Φ}
- Bins are fibers of q_ε, labeled by (ℓ, r) ∈ ℤ^{d_Φ} × ℤ
- **Lemma:** For configs in the same bin: ‖Φ(U) − Φ(U')‖ ≤ √(d_Φ)·ε/2

### 1.5 Skeleton Construction

The skeleton Σ_ε is a subset of loops chosen to cover configuration space at resolution ε.

**Simple skeleton (v1):** All plaquettes at stride s
- s = 1: Every plaquette (maximum resolution)
- s = 2: Every other plaquette
**Geodesic skeleton (v2):** Hierarchical multi-scale
- Level 0: 1×1 plaquettes at stride s₀
- Level 1: 2×2 Wilson loops at stride s₁
- Level 2: 4×4 Wilson loops at stride s₂
- Captures both UV and IR structure

For L=24 lattice with stride s=4:
- 6⁴ × 6 plaquettes = 7,776 loops
- d_Φ = 15,552 dimensions

### 1.6 Wilson Flow

Gradient flow smoothing to suppress UV fluctuations for cleaner topological charge:

```
∂_t U_μ(x,t) = -g₀² {∂S/∂U_μ(x)} U_μ(x,t)
```

Flow time t → 0.1-1.0 for smoothing. Implemented via `apply_wilson_flow(U, flow_time)`.

### 1.7 Topological Data Analysis (TDA)

Persistent homology detects topological features across scales:

- **H₀ (connected components):** Long-lived barcodes indicate well-separated clusters
- **H₁ (loops):** Holes/voids in the configuration space
- **Gap persistence:** Longest interval between vacuum and first excited state

## 2. Algorithm Specifications

### 2.1 Wilson Loop Computation

```python
def compute_plaquette(U: GaugeConfig, x: Site, mu: int, nu: int) -> np.ndarray:
    """
    Compute the plaquette P_μν(x) = U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x)
    
    Args:
        U: Gauge configuration with shape (4, L, L, L, L, 3, 3) complex128
        x: Site coordinates (t, x, y, z)
        mu: First direction (0-3)
        nu: Second direction (0-3), must be != mu
    
    Returns:
        3x3 complex matrix (the plaquette)
    
    Implementation notes:
        - Use periodic boundary conditions: (x + L) % L
        - Matrix multiplication order matters (path ordering)
        - Numba JIT compiled for performance
    """
    # Implemented in lattice/wilson_loops.py

def compute_wilson_loop(U: GaugeConfig, path: List[Tuple[Site, int]]) -> complex:
    """
    Compute Wilson loop W(C) = Tr[∏ U_μ(x)]
    
    Args:
        U: Gauge configuration
        path: List of (site, direction) pairs defining the loop
              Direction can be negative for backward links (U†)
    
    Returns:
        Complex trace of the path-ordered product
    
    Implementation notes:
        - Negative direction d means U_{-d-1}†(x-d̂)
        - Start with identity, accumulate products
        - Path must be closed (returns to starting site)
    """
    # Implemented in lattice/wilson_loops.py
```

### 2.2 Topological Charge

```python
def compute_field_strength_clover(U: GaugeConfig, x: Site, mu: int, nu: int) -> np.ndarray:
    """
    Compute clover-leaf field strength tensor G_μν(x)
    
    G_μν = (1/8) × Im[ P_μν(x) + P_μν(x-μ) + P_μν(x-ν) + P_μν(x-μ-ν)
                      - P_νμ(x) - P_νμ(x-μ) - P_νμ(x-ν) - P_νμ(x-μ-ν) ]
    
    This is the standard "clover" discretization of F_μν.
    
    Returns:
        3x3 traceless anti-Hermitian matrix (su(3) Lie algebra element)
    
    Note:
        G_μν must be traceless anti-Hermitian to live in su(3).
        Implementation projects onto this subspace.
    """
    # Implemented in lattice/topological.py

def compute_topological_charge(U: GaugeConfig, method: str = "clover") -> float:
    """
    Compute the topological charge Q of the configuration.
    
    Args:
        U: Gauge configuration
        method: "clover" (default), "plaquette", or "5Li"
    
    Returns:
        Float topological charge (should be near-integer for smooth configs)
    
    Implementation:
        Q = (1/32π²) Σ_x ε_μνρσ Tr[G_μν(x) G_ρσ(x)]
        
        The sum over ε_μνρσ reduces to:
        Q = (1/16π²) Σ_x Tr[G_01 G_23 + G_02 G_31 + G_03 G_12]
        
        (Factor of 2 from antisymmetry)
    
    Notes:
        - Apply Wilson flow or APE smearing before measuring for cleaner signal
        - Result should be close to integer; round for discrete r
    """
    # Implemented in lattice/topological.py
    # Also: apply_wilson_flow(U, flow_time) for gradient flow smoothing
```

### 2.3 Davis-Wilson Map

```python
def build_skeleton(lattice_size: int, stride: int, levels: int = 1) -> List[Loop]:
    """
    Construct the geodesic skeleton for Wilson loop sampling.
    
    Args:
        lattice_size: L for L⁴ lattice
        stride: Spacing between sampled plaquettes
        levels: Number of hierarchical levels (1 = plaquettes only)
    
    Returns:
        List of Loop objects, each specifying a closed path
    
    Level 0 (plaquettes): 1×1 loops
    Level 1: 2×2 rectangular loops  
    Level 2: 4×4 rectangular loops
    
    Total loops ≈ 6 × (L/stride)⁴ × levels
    """
    # Implemented in lattice/skeleton.py
    # Also: build_block_skeleton() for coarse-graining

def davis_wilson_map(U: GaugeConfig, skeleton: List[Loop]) -> Tuple[np.ndarray, int]:
    """
    Compute the Davis-Wilson cache Γ(U) = (Φ, r)
    
    Args:
        U: Gauge configuration
        skeleton: List of loops defining the continuous cache
    
    Returns:
        Φ: numpy array of shape (2 * len(skeleton),) with real/imag traces
        r: Integer topological charge
    
    Implementation:
        1. For each loop in skeleton:
           - Compute W = Tr[Wilson loop]
           - Append (Re(W), Im(W)) to Φ
        2. Compute Q = topological_charge(U)
        3. r = round(Q)
    """
    # Implemented in analysis/davis_wilson.py
```

### 2.4 Clustering Analysis

```python
def compute_gap_visibility(
    cache_points: np.ndarray,  # Shape (N, d_Φ)
    topological_charges: np.ndarray,  # Shape (N,)
    sector: int = 0  # Which topological sector to analyze
) -> Dict[str, Any]:
    """
    Compute the gap visibility metric and clustering statistics.
    
    Args:
        cache_points: Φ vectors for all configurations
        topological_charges: r values for all configurations
        sector: Topological sector to analyze (default: r=0 vacuum sector)
    
    Returns:
        {
            'gap_visibility': float,  # G metric (0 = no gap, >0 = gap detected)
            'n_clusters': int,        # Number of detected clusters
            'cluster_labels': array,  # Cluster assignment for each point
            'cluster_centers': array, # Centroids of each cluster
            'min_inter_cluster_distance': float,
            'density_in_void': float,
            'silhouette_score': float,
            'umap_embedding': array,  # 3D projection for visualization
            'tda_distance_scale': float,  # ε range used for TDA (for matching to physical length)
        }
    
    Algorithm:
        1. Filter to specified topological sector
        2. Cluster directly in high-dimensional space (avoid UMAP distortion)
        3. Apply block averaging for error estimation
        4. Use pairwise distance histogram analysis
        5. UMAP used ONLY for visualization (not gap detection)
        6. G = min_distance / (void_density + ε)
    
    Key parameters:
        - HDBSCAN: min_cluster_size=50, min_samples=10 (on raw Φ)
        - Block averaging: n_blocks=10 for jackknife errors
        - UMAP: n_neighbors=15, min_dist=0.1 (visualization only)
        - TDA: Store distance scale (ε_range) for later physical interpretation
    
    Statistical validation:
        - Permutation test against null hypothesis
        - Report G with confidence intervals
    """
    # Implemented in analysis/clustering.py
```

### 2.5 Topological Data Analysis

```python
def compute_persistent_homology(Phi: np.ndarray, max_dimension: int = 1) -> PersistenceResult:
    """
    Compute persistent homology via Vietoris-Rips complex.
    
    Uses ripser for efficient computation.
    
    Returns:
        PersistenceResult with betti numbers, barcodes, and gap persistence
    """
    # Implemented in analysis/tda.py
```

## 3. Data Formats

### 3.1 Gauge Configuration File Format

Use HDF5 for efficient storage:

```
config_NNNN.h5
├── /gauge_field          # (4, L, L, L, L, 3, 3) complex128 (or complex64 for storage)
├── /metadata
│   ├── lattice_size      # int
│   ├── beta              # float (coupling)
│   ├── trajectory        # int (HMC trajectory number)
│   └── action            # float (Wilson action value)
└── /measurements
    ├── plaquette         # float (average plaquette)
    └── topological_charge # float
```

**Note:** For large configs, consider storing gauge_field as complex64 to halve disk/RAM usage, while computing Γ in float64 for numerical stability.

### 3.2 Cache Output Format

```
cache_results.h5
├── /Phi                  # (N_configs, d_Phi) float64
├── /r                    # (N_configs,) int32
├── /config_ids           # (N_configs,) int32
└── /skeleton_info
    ├── n_loops           # int
    ├── stride            # int
    └── levels            # int
```

### 3.3 Analysis Output Format

```
analysis_results.json
{
    "gap_visibility": 4.73,
    "n_clusters": 3,
    "silhouette_score": 0.67,
    "sectors": {
        "0": {"n_points": 8432, "n_clusters": 2, "gap_visibility": 5.1},
        "1": {"n_points": 1203, "n_clusters": 1, "gap_visibility": null},
        "-1": {"n_points": 365, "n_clusters": 1, "gap_visibility": null}
    },
    "parameters": {
        "lattice_size": 24,
        "n_configs": 10000,
        "skeleton_stride": 4,
        "beta": 6.0
    }
}
```

### 3.4 Production Backend Support

Interface with battle-tested lattice QCD codes for configuration generation:

| Backend | Format | Reader Function |
|---------|--------|-----------------|
| openQCD | Binary | `read_openqcd_config()` |
| Grid | Binary | `read_grid_config()` |
| MILC | Binary | `read_milc_config()` |
| ILDG | XML+Binary | `read_ildg_config()` |

Utilities:
- `load_configs(directory)`: Auto-detect format, load all configs
- `convert_to_hdf5(input_dir, output_dir)`: Batch conversion
- `run_openqcd(...)`: Generate configs via openQCD subprocess
- `download_milc_configs(...)`: Fetch public MILC ensembles

See `BACKENDS.md` for detailed usage.

## 4. Computational Requirements

### 4.1 Per-Configuration Costs

| Operation | Time (24⁴ lattice) | Memory |
|-----------|-------------------|--------|
| Load config | ~0.3s | 190 MB |
| All plaquettes | ~2s | +50 MB |
| Topological charge | ~1s | +10 MB |
| Skeleton (stride=4) | ~5s | +100 MB |
| **Total Γ(U)** | **~10s** | **~400 MB peak** |

### 4.2 Full Experiment

| Phase | 10k configs | Time | Cost (Modal A100) |
|-------|-------------|------|-------------------|
| Generate configs | 10,000 | 4h | $15 |
| Compute Γ | 10,000 | 28h (parallelized: 1h) | $4 |
| Clustering | 1 | 10min | $0.50 |
| **Total** | | **~5h wall** | **~$20** |

### 4.3 Parallelization Strategy

- **Config generation**: Embarrassingly parallel (independent Markov chains)
- **Cache computation**: Embarrassingly parallel (independent configs)
- **Clustering**: Sequential, but fast

Modal deployment: 28 A100s × 1 hour = 28 GPU-hours

## 5. Test Cases

### 5.1 Unit Tests (tests/test_su3.py)

```python
def test_su3_unitarity():
    """Random SU(3) matrices should satisfy U†U = I"""
    
def test_su3_determinant():
    """det(U) should equal 1"""
    
def test_random_su3_haar_measure():
    """Generated matrices should follow Haar measure distribution"""

def test_su3_multiply():
    """Product of SU(3) matrices is SU(3)"""
```

### 5.2 Integration Tests (tests/test_integration.py)

```python
def test_plaquette_cold_start():
    """For U=I everywhere, average plaquette = 1"""
    
def test_plaquette_bounds():
    """Plaquette values in valid range for all configs"""

def test_topological_charge_cold_start():
    """Cold start config has Q ≈ 0"""

def test_small_lattice_pipeline():
    """Full pipeline on 4⁴ lattice, 10 configs"""
    
def test_clustering_synthetic_gap():
    """Synthetic clustered data should give G > 0"""
    
def test_clustering_continuous_no_gap():
    """Synthetic uniform data should give G ≈ 0"""

def test_wilson_flow_reduces_charge_variance():
    """Wilson flow should smooth topological charge"""

def test_block_averaging_consistency():
    """Block averaging gives consistent error estimates"""

def test_beta_scan_monotonicity():
    """Gap visibility varies monotonically with β in transition region"""
```

## 6. Success Criteria

### 6.0 What We're Actually Testing

**Important caveat:** This experiment tests a **weaker, empirical version** of Axiom 4 (Curvature–Information Duality):

> "If bins are genuinely separated in curvature (∫‖F‖²), they should appear as separated clusters in Φ-space at finite ε."

We are **not** directly estimating ∫‖F‖². We use Wilson loops as a proxy for geometry, then cluster in Φ-space. This is appropriate for numerical verification, but should not be oversold as a direct test of the exact inequality.

Additionally: **"Discrete clusters vs continuous cloud" is a strong visual metaphor, but high-dimensional distributions can look continuous even when there is a mass gap.** The clustering analysis provides evidence, not proof. Future work should include:
- Correlator measurements for direct mass extraction
- Off-diagonal mixing studies

### 6.1 Primary Hypothesis Test

**Null hypothesis H₀:** Configurations are uniformly distributed in cache space (no gap)
**Alternative H₁:** Configurations cluster discretely (gap exists)

**Statistical Test (implemented in `compute_gap_visibility`):**
1. Compute observed gap visibility G on real data
2. Generate null distribution by:
   - Random permutation of Φ components (destroys clustering structure)
   - Recompute G for each permutation (n_permutations=100)
3. Compute p-value = fraction of null samples with G ≥ observed
4. Reject H₀ if p < 0.05 (95% confidence)
5. Report G with bootstrap confidence intervals

### 6.2 Quantitative Thresholds

| Metric | Weak Evidence | Strong Evidence |
|--------|---------------|-----------------|
| Gap visibility G | G > 1 | G > 5 |
| p-value | p < 0.05 | p < 0.01 |
| Silhouette score | s > 0.3 | s > 0.6 |
| Cluster separation | d > 2σ | d > 5σ |
| TDA gap persistence | > median | > 90th percentile |

### 6.3 Robustness Checks

1. **Scale dependence:** Vary skeleton stride, verify G is stable
2. **Lattice size:** Compare L=16, 24, 32; G should persist or increase
3. **β dependence:** Beta scan across confinement/deconfinement transition
4. **Sector dependence:** Analyze r=0, ±1 separately
5. **Block averaging:** Error estimates via jackknife resampling

## 7. Project Structure

```
davis-wilson-lattice/
├── lattice/                    # Core lattice QCD library
│   ├── su3.py                  # SU(3) matrix operations
│   ├── gauge_config.py         # Configuration I/O and HMC generation
│   ├── wilson_loops.py         # Wilson loop/plaquette computation
│   ├── topological.py          # Topological charge, Wilson flow, smearing
│   ├── skeleton.py             # Skeleton construction for Davis-Wilson map
│   └── backends.py             # Production backend readers (openQCD, Grid, MILC, ILDG)
├── analysis/                   # Analysis tools
│   ├── davis_wilson.py         # The Davis-Wilson map Γ: A/G → C
│   ├── clustering.py           # Gap visibility, HDBSCAN, block averaging
│   ├── tda.py                  # Topological Data Analysis (persistent homology)
│   └── visualization.py        # Plotting and visualization
├── tests/                      # Test suite
│   ├── test_su3.py             # SU(3) unit tests
│   └── test_integration.py     # Full pipeline integration tests
├── modal_app.py                # Modal cloud deployment (A100 GPU)
├── run_analysis.py             # Standalone analysis script
├── SPEC.md                     # This specification
├── BACKENDS.md                 # Production backend usage guide
└── pyproject.toml              # Python dependencies
```

## 8. Running the Experiment

### Local Testing
```bash
# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v

# Small test run
python run_analysis.py --lattice-size 8 --n-configs 100
```

### Modal Cloud (Production)
```bash
# Full experiment on A100 GPUs
modal run modal_app.py

# Beta scan across transition
modal run modal_app.py::run_beta_scan
```
