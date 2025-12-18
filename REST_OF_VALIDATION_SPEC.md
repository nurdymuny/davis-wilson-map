# REST OF VALIDATION + MATH SPEC (Post-ASS-001) — v1.4

**Project:** Davis–Wilson Yang–Mills Mass Gap (v3.0)  
**Owner:** Bee Rosa Davis (she/her)  
**Date:** December 18, 2025  
**Status:** Axiom 7 almost-superselection validated by ASS-001 with R=0.00138 (72× below threshold).

**v1.4 Implementation Fixes:**
- κ_sep_gap: TRUE action gap (vacuum vs nearest non-vacuum quantile), units = action
- χ_sep: SEPARATE function for dimensionless ratio = κ_sep_gap / Q_0.9(σ_b)
- Wilson flow: IMPLEMENTED (gradient flow to t*)
- Topological charge: Proper clover F_μν definition
- A2S pass: Uses bin_sep_ratio >= 5 (not global_std/q90)
- A4C2 k-NN: Real k-NN in Φ-space (not sorted 1D proxy)
- ε_skel: NOW AFFECTS Φ (controls loop sampling richness)

---

## Executive Summary

This spec defines the **final validation suite** to close all open items in the v3.0 paper's "What Remains" section:

| Test ID | Target | Status |
|---------|--------|--------|
| A2S-001 | Axiom 2: Approximate Cache Sufficiency | Open |
| A4C2-001 | Axiom 4 Case 2: Same-r Curvature Gap | Open |
| KSTAR-001 | κ* Continuum Survival | Open |
| OSBRIDGE-001 | OS/Transfer-Matrix Bridge | Open |
| HEPS-001 | H_ε → H_phys Uniform Gap | Open |

**Compute Target:** Single A100 GPU session with cached configs  
**Expected Runtime:** 20-40 minutes (GPU-optimized)

---

## Critical Definitions (Consistent Across All Tests)

### κ_sep: Action Separation Scale (PRIMARY)

The **canonical κ** used in R = η/(λκ) is defined as:

$$
\kappa_{\text{sep}} = \min_{b \neq b_0} \left( Q_{0.1}(S_E^{\text{flow}}{}^{(b)}) - Q_{0.9}(S_E^{\text{flow}}{}^{(b_0)}) \right)
$$

where:
- $b_0$ = vacuum bin (lowest mean action)
- $Q_{p}(S_E^{(b)})$ = p-th quantile of action values in bin b
- $S_E^{\text{flow}}$ = action computed **after** Wilson flow to the test's specified scale $t^*$
- This measures the **action gap** between vacuum and nearest non-vacuum bin

**Preprocessing clause:** All κ_sep computations use $S_E^{\text{flow}}(t^*)$ evaluated after Wilson flow 
to the test's specified scale; raw action is logged for diagnostics only.

**Occupancy requirements for quantile estimation:**
- Minimum per-bin occupancy for Q_0.1/Q_0.9: ≥ 20 configs
- If bin has 10-19 configs: fall back to Q_0.2/Q_0.8
- If bin has < 10 configs: use mean ± std instead of quantiles

**Units:** Lattice action units (dimensionless, β-dependent)

### χ_sep: Dimensionless Separation Ratio (DERIVED)

When we need a **dimensionless** signal-to-noise measure:

$$
\chi_{\text{sep}} := \frac{\kappa_{\text{sep}}}{Q_{0.9}(\sigma_b(S_E))}
$$

where Q_0.9(σ_b) is the 90th percentile of within-bin action standard deviations.

**Usage:** χ_sep ≥ 1 means the gap exceeds typical within-bin noise. Use χ_sep (not κ_sep) 
when comparing gap to spread. The primary R formula uses κ_sep (action units), not χ_sep.

### κ_geom: Geometric Centroid Separation (SECONDARY)

For geometric analysis only (never used in R calculation):

$$
\kappa_{\text{geom}} = \text{condition number of bin centroid matrix}
$$

**Usage:** Only in KSTAR for geometric scaling checks. Always labeled explicitly.

### η_emp: Inter-bin Mixing Rates (DYNAMIC)

Requires **time-ordered MCMC chain**. We track TWO statistics:

**η_mean (global average mixing rate):**
$$
\eta_{\text{mean}} = \frac{1}{N-1} \sum_{t=1}^{N-1} \mathbf{1}[b(A_t) \neq b(A_{t+1})]
$$

**η_max (worst-case per-bin leakage):**
$$
\eta_{\max} = \max_b \mathbb{P}(b_{t+1} \neq b \mid b_t = b)
$$

where b(A_t) is the bin assignment at MCMC step t.

**Usage:**
- **R** is computed using η_max (conservative, what diagonal dominance arguments require)
- R_mean = η_mean / (λκ) is reported as secondary diagnostic
- Both η_mean and η_max are reported in all outputs

**Critical:** Cannot be computed from unordered config bags.

### δ_O(ε): Within-Bin Observable Dispersion

$$
\delta_O(\varepsilon) = \text{median}_b \, \sigma_b(O) \quad \text{with 95% bootstrap CI}
$$

Report as function of ε_skel.

### V1 Thresholds

- **V1_loose:** η_max < 0.5 (worst-case mixing is sub-dominant)
- **V1_strict:** η_max < 0.10 (worst-case mixing is rare)

V1 gates on **η_max** (worst-case per-bin leakage) because that's what threatens diagonal dominance.
Report η_mean as secondary diagnostic. Use V1_loose as pass criterion for conservative claims.

---

## Topology Diagnostic (Required in ALL Tests)

Every test must report:

```
r_histogram: {r: count for r in observed charges}
topology_frozen: True if only one r value observed
r_diversity: number of distinct r values
```

**Interpretation guidelines:**
- Always report r_histogram (mandatory diagnostic)
- If topology is frozen (single r value): scope claims to "within fixed r-sector"
- Topological diversity is **only required** when explicitly claiming cross-sector results
- On small lattices (L ≤ 10) with short runs, frozen topology is a **known lattice artifact**, 
  not a failure of the physics—validate passes within that sector

**Note:** "Should show transitions" is aspirational, not a hard pass/fail criterion. 
The test validates physics within whatever sector(s) are sampled.

---

## Part A — A2S-001: Axiom 2 Approximate Cache Sufficiency

### A.1 Formal Target

Axiom 2 states: configurations with identical cache yield approximately identical gauge-invariant observables.

$$
\Gamma_\varepsilon(A) = \Gamma_\varepsilon(A') \implies |\langle O \rangle_A - \langle O \rangle_{A'}| < \delta_O(\varepsilon)
$$

where δ_O(ε) → 0 as the skeleton refines.

### A.2 Controlled Observable Family

Per v3.0, observables controlled at resolution ε:

1. **Plaquette average** P = (1/6V) Σ Re Tr(U_μν)
2. **Action density** S_E/V

### A.3 Experimental Design (GPU-Optimized)

```
Parameters:
  L: 8
  beta: 6.0
  n_configs: 300 (GPU batched)
  n_therm: 100
  n_skip: 5
  
Resolution grid (reduced):
  epsilon_skel: [0.15, 0.20, 0.25]
  epsilon_disc: [0.15, 0.20]

Minimum bin occupancy: M = 10
Bootstrap samples: 200 (jackknife for speed)
```

### A.4 δ_O(ε) Definition (Concrete)

$$
\delta_O(\varepsilon) = \text{median}_b \, \sigma_b(O)
$$

with 95% CI from jackknife resampling.

### A.5 Algorithm (Vectorized)

```python
# All bins computed in parallel via tensor ops
for each (eps_skel, eps_disc):
    1. Quantize all Φ vectors (vectorized)
    2. Group by bin (sparse tensor indexing)
    3. Compute σ_b(O) for all bins simultaneously
    4. δ_O = median(σ_b) with jackknife CI
    5. Build k-NN graph of bin centroids in Φ-space (k=3)
    6. Compute between-bin separations on ADJACENT bins only:
       D_{bb'}(O) = |μ_b(O) - μ_{b'}(O)| for (b,b') in k-NN graph
       D_min(O) = min over adjacent pairs
       (Same graph structure as A4C2, O(n_bins·k) complexity)
```

### A.5 Acceptance Criteria

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Within-bin dispersion | Q_0.9(σ_b(O)) ≤ 3 × δ_O(ε) | Most bins are tight (allow outliers) |
| Signal-to-bin ratio | min_{b≠b'} D_{bb'}(O) / Q_0.9(σ_b(O)) ≥ 5 | Bins are distinguishable |
| Refinement monotonicity | δ_O(ε) decreases as ε_skel ↓ | Finer resolution → tighter bins |

**Note:** We use Q_0.9 (90th percentile) instead of max to avoid sensitivity to outlier bins.

### A.6 Deliverables

- `results/A2S_001_cache_sufficiency.json`
- `results/figures/A2S_001_sufficiency_plots.png`

---

## Part B — A4C2-001: Axiom 4 Case 2 (Same-r Curvature Gap)

### B.1 Formal Target

Case 1 (different r) is rigorous via BPS bound. Case 2 is the open problem:

For r(A) = r(A') but different Φ-bins:

$$
\kappa_{\text{adj}}^{(r)} = \min_{\substack{b \sim b' \\ r_b = r_{b'}}} \left| \bar{S}_E^{(b)} - \bar{S}_E^{(b')} \right| > 0
$$

where $b \sim b'$ denotes adjacent (nearest-neighbor) bins in Φ-space.

**Naming note:** We use κ_adj (adjacent-bin action gap) here, NOT κ_geom (which is reserved 
for the centroid condition number defined earlier). This avoids namespace collision.

### B.2 Experimental Design (BIN-BASED, NOT ALL-PAIRS)

```
Parameters:
  L: 8
  beta: 6.0
  n_configs: 300 (reuse from A2S)
  
Method: Bin-level statistics (NOT O(N²) pairs)
  - Compute mean action per bin
  - Find nearest-neighbor bins in Φ-space
  - κ_adj = min action gap between adjacent bins within same r-sector
```

### B.3 Algorithm (GPU-Optimized)

```python
# Vectorized bin-level computation
1. Assign all configs to bins (vectorized)
2. Compute bin centroids and mean actions (scatter_add)
3. Build k-NN graph of bins in Φ-space (k=3)
4. For each bin pair (b, b') in NN graph with r_b = r_b':
   κ_bb' = |S_E_mean(b) - S_E_mean(b')|
5. κ_adj = min(κ_bb') with jackknife CI
```

### B.4 Acceptance Criteria

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Nontrivial gap | κ_adj > 0 with 95% CI lower > 0 | Curvature gap exists |
| Jackknife stability | CI width < 0.5 × κ_adj | Robust estimate |

### B.5 Deliverables

- `results/A4C2_001_curvature_info_gap.json`
- `results/figures/A4C2_001_bin_action_gaps.png`

---

## Part C — KSTAR-001: Continuum Survival of κ

### C.1 Formal Target

The critical question from v3.0: does κ_sep(ε) → 0 as a → 0, or renormalize to κ* > 0?

**Using κ_sep (action separation), NOT κ_geom (centroids).**

### C.2 Experimental Design (GPU-Optimized)

```
Multi-β scaling study:
  beta_values: [5.8, 6.0, 6.2]  # coarse → fine
  L_values: [8, 10]  # finite-size check (10 not 12 for speed)
  
Per (β, L):
  n_configs: 150 (reduced for multi-β)
  n_therm: 80
  n_skip: 4
```

### C.3 Self-Calibrating Scale (No External a(β) Table)

**Key principle**: We never claim to know `a(β)` in fm. Instead, we use a **t₀-like flow reference scale** 
`t_ref` as an internal ruler. This is defined via `t²⟨E(t)⟩ = c` with c=0.3, NOT the derivative-based 
w₀ definition from literature (to avoid nomenclature disputes).

**Self-calibration procedure**:
```python
def find_t_ref_scale(config):
    """Find t such that t² <E(t)> = 0.3 (t₀-like definition)"""
    for t in linspace(0.01, 0.5, 50):
        E_t = flow_action_density(config, t)
        if t**2 * E_t > 0.3:
            return t  # This is t_ref
    return 0.3  # Fallback
```

**Usage**: At each β, measure t_ref from first 20 configs. Use t_ref as the common scale:
- κ_sep(β) in units of t_ref (dimensionless)
- Compare across β without knowing a(β) in fm

### C.4 Algorithm

```python
for each (beta, L):
    1. Generate 20 calibration configs
    2. Measure t_ref = t where t²<E(t)> = 0.3 (t₀-like flow scale)
    3. Generate 150 production configs
    4. Apply flow to t = t_ref (one common physical scale)
    5. Compute κ_sep using ASS-001 methodology:
       - Bin configs by (Φ, r)  
       - κ_sep = action gap (see Critical Definitions)
       - χ_sep = κ_sep / Q_0.9(σ_b(S_E)) (dimensionless ratio)
       - NOT κ_geom (centroids), but κ_sep (action separation)
    6. Record κ_sep(β, L) with bootstrap CI
    
Analysis:
    - Plot κ_sep vs β for each L (β is proxy for a→0)
    - Check finite-size stability: |κ(L=12) - κ(L=8)| / κ < 20%
    - Key question: does κ_sep plateau/grow as β increases?
```

### C.5 Acceptance Criteria

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Finite-size stability | Δκ_sep/κ_sep < 20% when L doubles | Volume-independent |
| Nonzero scaling limit | κ_sep(β→∞) plateau or growth | κ* survives continuum |
| Consistency | κ_sep ~ O(100) matching ASS-001 | Same physics |
| Self-calibration | t_ref measured within 10% across runs | Stable internal scale |
| t_ref/a monotonicity | t_ref/a increases as β increases | Confirms continuum direction |

**Internal monotonicity check:** Since we avoid external a(β) tables, we verify the continuum 
direction by checking that t_ref/a (measured in lattice units) is monotonically increasing with β. 
This confirms that larger β corresponds to finer lattice spacing, as expected from asymptotic freedom.

### C.6 Deliverables

- `results/KSTAR_001_scaling.json`
- `results/figures/KSTAR_001_continuum_extrapolation.png`

---

## Part D — OSBRIDGE-001: Transfer-Matrix Qualitative Alignment

### D.1 Formal Target

**What we claim**: The spectral gap from transfer-matrix (Euclidean correlator decay) is 
**qualitatively consistent** with the bin gap κ_sep from ASS-001.

**What we DO NOT claim**: Numerical equality `m_gap = λ·κ_sep` with specific units mapping.
The relationship is conceptual: both reflect gap structure in different formalisms.

### D.2 Experimental Design

```
Parameters:
  L: 8 (spatial), T: 16 (temporal, extended for correlators)
  beta: 6.0
  n_configs: 200
  
Glueball operator (0++ channel):
  O(t) = Σ_x Tr(U_plaq(x,t))  # Plaquette sum at timeslice t
  
Correlator:
  C(t) = <O(0) O(t)> - <O>²
```

### D.3 Algorithm

```python
1. Generate configs on 8³×16 lattice
2. For each config:
   a. Compute O(t) = Σ_x Re[Tr(U_plaq(x,t))] at each timeslice
   b. Compute C(t) = <O(0)O(t)> - <O>² (connected correlator)
3. Average C(t) over configs with jackknife errors
4. Fit effective mass:
   m_eff(t) = ln(C(t)/C(t+1))
5. Extract plateau mass m_gap from t ∈ [3, 7]
6. Run bin analysis on same configs to get κ_sep

Qualitative check (NOT a quantitative pass/fail):
   - Both m_gap > 0 and κ_sep > 0 present
   - Both in "gapped" regime (not collapsing to zero)
```

### D.4 Acceptance Criteria

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Clean exponential decay | C(t) > 0 for t ≤ T/2, monotonic decrease | Gap exists |
| Plateau formation | σ(m_eff) / m_eff < 30% in plateau region | Physical mass extracted |
| Qualitative alignment | m_gap > 0 AND κ_sep > 0 | Both formalisms see gap |
| Stability under flow | m_gap and κ_sep stable under ±20% flow time change | Not artifacts of t* choice |
| Consistent β-trend | If multi-β: both m_gap and κ_sep trend same direction | No contradictory scaling |

**Note**: This test is QUALITATIVE. We report both quantities but do not claim `m_gap = f(κ)` 
for any specific function f. The criteria are:
1. Both are nonzero
2. Both are stable under modest changes in flow / resolution
3. Both trend consistently with β (if multi-β data available)

### D.5 Deliverables

- `results/OSBRIDGE_001_correlators.json`
- `results/figures/OSBRIDGE_001_effective_mass.png`

---

## Part E — HEPS-001: H_ε → H_phys Uniform Gap

### E.1 Formal Target

Show that the effective Hilbert space construction preserves a uniform gap under refinement:

$$
\inf_n \kappa_{\text{sep}}^{(n)} \geq \kappa_{\min} > 0
$$

**CRITICAL: Separating η and κ measurements**

The GPT review correctly notes that η_emp requires temporal MCMC ordering (for mixing rates),
while κ_sep can be computed from any bag of configs. We handle this:

| Quantity | Requires Chain Order? | Measurement Strategy |
|----------|----------------------|---------------------|
| κ_sep | NO | Can use any config bag |
| η_emp | YES | Must track MCMC sequence |
| R = η/(λκ) | YES (through η) | Full chain needed |

**For HEPS-001**: We measure κ_sep at each refinement level (bag-friendly).
We measure η_emp and R only at the original chain's resolution, then show κ_sep remains 
bounded below as ε → 0.

### E.2 Refinement Ladder

```
Level 0 (coarse):   ε_skel = 0.30, ε_disc = 0.30
Level 1:           ε_skel = 0.25, ε_disc = 0.25
Level 2:           ε_skel = 0.20, ε_disc = 0.20
Level 3:           ε_skel = 0.15, ε_disc = 0.15
Level 4 (fine):    ε_skel = 0.10, ε_disc = 0.10
```

### E.3 Algorithm

```python
# First: Generate a SINGLE MCMC chain (preserving order)
chain = generate_mcmc_chain(n_configs=500, beta=6.0, L=8)
# chain maintains temporal ordering for η computation

for each refinement level n:
    ε = (0.30, 0.25, 0.20, 0.15, 0.10)[n]
    
    1. Compute bins at resolution (ε_skel=ε, ε_disc=ε)
    2. κ_sep^(n) = action gap (see Critical Definitions)
       χ_sep^(n) = κ_sep / Q_0.9(σ_b(S_E)) (dimensionless ratio)
       - This ONLY needs a bag of configs (no ordering)
    3. n_bins = count of occupied bins
    4. Record κ_sep^(n), χ_sep^(n), n_bins, bin occupancy distribution

# η measurement at ONE reference resolution (requires chain order)
reference_level = 2  # ε = 0.20
η_mean, η_max = compute_mixing_rate(chain, ε_skel=0.20, ε_disc=0.20)
R_ref = η_max / (λ * κ_sep^(reference_level))  # Uses η_max (conservative)

Analysis:
    - Plot κ_sep vs refinement level
    - Check: κ_sep^(n) ≥ κ_min > 0 for all n (uniform gap)
    - Check: R_ref << 1 at reference (almost-superselection holds)
    - Report: η_mean/η_max only at reference, κ at all levels
```

### E.4 Acceptance Criteria

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Uniform gap | κ_sep^(n) ≥ 50 for all n | Gap doesn't collapse under refinement |
| Reference almost-superselection | R_ref ≤ 0.10 | Consistent with ASS-001 |
| Meaningful refinement | n_bins^(fine) > n_bins^(coarse) | Resolution increases bin count |
| No collapse | κ_sep^(fine) / κ_sep^(coarse) ∈ [0.5, 2.0] | Gap remains O(1) |

### E.5 Deliverables

- `results/HEPS_001_nested_resolution.json`
- `results/figures/HEPS_001_refinement_ladder.png`

---

## Implementation: Unified GPU Script

### GPU Optimization Strategy (Critical for Speed)

**Target**: Complete all 5 tests in 20-40 minutes on single A100.

**Key optimizations**:

1. **Parallel Checkerboard HMC**: All HMC steps use even/odd site parallelism
2. **Vectorized Observables**: Wilson loops, plaquettes computed in batch
3. **Config Caching**: Generate once, reuse across tests (A2S, A4C2, HEPS share configs)
4. **Jackknife over Bootstrap**: Single-pass delete-1 jackknife (n passes) vs bootstrap (B×n passes)
5. **Bin-Level Statistics**: O(n_bins) operations, never O(n_configs²)
6. **Async I/O**: Save results while next phase runs
7. **Memory Reuse**: Pre-allocate gauge field tensors, reuse across configs

**Timing budget**:
```
Phase 1 (config generation): ~8 min (500 configs, shared)
Phase 2 (A2S-001):          ~3 min (cache analysis, δ_O computation)
Phase 3 (A4C2-001):         ~2 min (bin-level, not O(N²))
Phase 4 (KSTAR-001):        ~12 min (multi-β, but reduced counts)
Phase 5 (OSBRIDGE-001):     ~5 min (correlator analysis, extended T)
Phase 6 (HEPS-001):         ~5 min (refinement ladder, bag-mode κ only)
─────────────────────────────────────
Total:                      ~35 min
```

### File Structure

```
extended_capabilities/
  rest_of_validation_gpu.py    # Main GPU script (this spec)
  
results/
  A2S_001_cache_sufficiency.json
  A4C2_001_curvature_info_gap.json
  KSTAR_001_scaling.json
  OSBRIDGE_001_correlators.json
  HEPS_001_nested_resolution.json
  figures/
    A2S_001_sufficiency_plots.png
    A4C2_001_envelope_scatter.png
    KSTAR_001_continuum_extrapolation.png
    OSBRIDGE_001_effective_mass.png
    HEPS_001_refinement_ladder.png
```

### Modal Configuration

```python
@app.function(
    gpu="A100",
    timeout=3600,  # 1 hour (with buffer)
    image=modal.Image.debian_slim()
        .pip_install("torch", "numpy", "scipy", "matplotlib")
)
def run_rest_of_validation():
    ...
```

### Execution Order (with Config Sharing)

1. **Phase 1: Shared Config Generation**
   - Generate master set: 500 configs at β=6.0, L=8 (for A2S, A4C2, HEPS)
   - Preserve MCMC chain order for η computation
   - Print r_histogram after every 100 configs (frozen topology diagnostic)

2. **Phase 2: A2S-001** (Axiom 2)
   - Bin configs, compute within/between-bin observables
   - δ_O = median_b σ_b(O) with jackknife CI

3. **Phase 3: A4C2-001** (Axiom 4 Case 2)
   - Bin-level analysis: bin centroids in (Φ, S_E) space
   - Nearest-neighbor ΔΦ vs ΔS_E (NOT all pairs)

4. **Phase 4: KSTAR-001** (κ* scaling)
   - Multi-β runs: 5.8, 6.0, 6.2 × L=8,12
   - Self-calibrating w₀ scale, 150 configs each
   - κ_sep (not κ_geom) at each point

5. **Phase 5: OSBRIDGE-001** (correlators)
   - Extended 8³×16 lattice, 200 configs
   - Glueball correlator + effective mass
   - Qualitative alignment (not numeric equality)

6. **Phase 6: HEPS-001** (nested resolution)
   - Refinement ladder on shared configs
   - κ_sep at all levels (bag-mode, fast)
   - η_emp only at reference level (requires chain)

---

## Success Criteria Summary

| Test | Key Metric | Pass Condition |
|------|------------|----------------|
| A2S-001 | δ_O(ε) | Signal-to-noise ≥ 5, Q_0.9(σ_b) ≤ 3×δ_O |
| A4C2-001 | κ_adj(r) | κ_adj > 0 with 95% CI, bin-level analysis |
| KSTAR-001 | κ_sep(β, L) | Plateau or growth as β↑, w₀/a monotonic, finite-size stable |
| OSBRIDGE-001 | m_eff plateau | Clean decay, m_gap > 0, stable under flow variation |
| HEPS-001 | κ_sep^(n) | κ_sep ≥ 50 at all refinement levels, R(η_max) ≤ 0.10 |

**Mandatory diagnostic for ALL tests**: Report r_histogram. Frozen topology scopes claims 
to fixed-r sector but is not a failure condition.

---

## Addressing Reviewer Feedback Summary

### v1.1 Issues (Resolved)
| Issue | Resolution |
|-------|------------|
| κ definition drift | κ_sep (action) is primary everywhere; κ_geom noted but never in R formula |
| Units mismatch in OSBRIDGE | Reframed as qualitative alignment; no numeric equality claim |
| HEPS η requires chain | η measured only at reference level; κ_sep (bag-friendly) at all levels |
| A4C2 O(N²) pairs | Changed to bin-level nearest-neighbor approach |
| δ_O(ε) undefined | Defined: δ_O = median_b σ_b(O) with jackknife CI |
| V1 threshold inconsistency | Defined V1_loose (0.5) and V1_strict (0.10), use loose for MCMC |
| KSTAR a(β) table | Self-calibrating via Wilson flow w₀ scale |
| Runtime concerns | Config sharing, jackknife, vectorization, bin-level (not O(N²)) |
| Frozen topology | Mandatory r_histogram every 100 configs |

### v1.2 Issues (Resolved)
| Issue | Resolution |
|-------|------------|
| κ_sep preprocessing ambiguity | Added explicit clause: uses S_E^flow(t*) everywhere |
| κ_sep quantile occupancy | Added minimum occupancy requirements (≥20 for Q_0.1/Q_0.9) |
| η_emp mean vs max | Added both η_mean and η_max; R uses η_max (conservative) |
| A2S tautological criterion | Changed max_b to Q_0.9(σ_b) with factor 3× allowance |
| A4C2 κ_geom name collision | Renamed to κ_adj (adjacent-bin action gap) |
| KSTAR continuum direction | Added w₀/a monotonicity check |
| OSBRIDGE "same magnitude" vague | Tightened to: stability under flow, consistent β-trend |
| Frozen topology too strict | Softened: frozen is scoped, not failed |

---

## Final Deliverable

Upon completion, v3.0's "What Remains" section transforms from:

> "Prove superselection, prove Axioms 2 & 4, prove κ survives continuum, construct measure, show H_ε → H_phys"

To:

> "**All five validation packages delivered.** Axioms 2, 4 empirically verified. κ* scaling confirmed. Transfer-matrix qualitative alignment demonstrated. Uniform gap preserved under refinement."

**This closes the loop.**

---

*Spec version: 1.2 (surgical fixes from GPT reviewer round 2)*  
*Created: December 18, 2025*  
*Target: Single A100 GPU run via Modal, ~35 minutes*
