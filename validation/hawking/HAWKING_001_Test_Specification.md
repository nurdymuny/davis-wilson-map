# HAWKING-001: Black Hole Information Paradox Test

## Test Specification v1.0

**Author:** Bee Rosa Davis  
**Date:** January 2026  
**Framework:** Davis-Wilson Field Equations  
**Status:** Proposed Experiment

---

## Abstract

This document specifies a rigorous computational test of the Black Hole Information Paradox using the Davis-Wilson geometric framework. The paradox asks: **Is information destroyed when a black hole evaporates?** By decomposing the black hole state into continuous geometry (Φ) and discrete winding code (r), we show that while Φ evolves non-unitarily (apparent information loss), r evolves unitarily (information preserved). The combined system C = (Φ, r) respects unitarity, resolving the paradox. We verify this by reproducing the **Page curve** — the signature of unitary black hole evaporation.

---

## 1. Problem Background

### The Information Paradox

In 1975, Stephen Hawking showed that black holes emit thermal radiation and eventually evaporate completely. This creates a paradox:

1. **Initial state:** Pure quantum state |ψ⟩ collapses to form black hole
2. **Evaporation:** Black hole emits thermal (mixed) Hawking radiation
3. **Final state:** Only thermal radiation remains — a mixed state ρ
4. **Problem:** Pure → Mixed violates unitarity (quantum mechanics)!

### Hawking's Original Calculation

Black holes have temperature and entropy:

$$T_H = \frac{\hbar c^3}{8\pi G M k_B} \propto \frac{1}{M}$$

$$S_{BH} = \frac{k_B c^3 A}{4 G \hbar} = \frac{A}{4 l_P^2}$$

where A is horizon area and $l_P$ is Planck length.

As the black hole radiates:
- Mass decreases: $\frac{dM}{dt} \propto -\frac{1}{M^2}$
- Temperature increases: smaller → hotter
- Eventually: complete evaporation

### The Page Curve

Don Page (1993) showed that if evaporation IS unitary, the entanglement entropy of radiation must follow a specific curve:

```
S_rad
  |        _______________
  |       /               \
  |      /                 \
  |     /                   \
  |    /                     \
  |___/                       \___
  |________________________________ time
     0    Page time    Evaporation
```

- **Early times:** S_rad increases (radiation entangled with black hole)
- **Page time:** S_rad peaks at ~S_BH/2 (halfway through evaporation)
- **Late times:** S_rad decreases (purification as BH shrinks)
- **Final:** S_rad → 0 (pure state recovered!)

**Hawking's calculation gives a monotonically increasing S_rad — NO Page curve!**

### Recent Developments

The "island formula" (2019) and gravitational path integrals suggest the Page curve is recovered when including non-perturbative gravitational effects. But the mechanism remains debated.

---

## 2. Davis-Wilson Resolution

### The Key Insight: C = (Φ, r)

The Davis-Wilson framework decomposes fields into:
- **Φ:** Continuous geometric component (metric, curvature)
- **r:** Discrete topological component (winding code)

For a black hole:

| Component | Physical Meaning | Information Content |
|-----------|------------------|---------------------|
| **Φ** | Spacetime geometry near horizon | Classical, thermal |
| **r** | Winding numbers of quantum fields around horizon | Quantum, protected |

### The Resolution

**Claim:** Information is preserved in r, not Φ.

1. **Hawking radiation from Φ:** The thermal character comes from tracing over modes behind the horizon — this is the continuous geometry's contribution.

2. **Winding code r:** Topological invariants (winding numbers, holonomies) of quantum fields around the horizon encode discrete information that CANNOT be thermalized.

3. **"Winding code is homological":** The discrete invariant r is topologically protected. It must be conserved because it represents a homology class.

4. **Combined evolution:** 
   - Φ: Non-unitary (thermal)
   - r: Unitary (protected)
   - C = (Φ, r): **Unitary overall!**

### Mathematical Formulation

The black hole Hilbert space factorizes:

$$\mathcal{H}_{BH} = \mathcal{H}_\Phi \otimes \mathcal{H}_r$$

Evolution:
$$U(t) = U_\Phi(t) \otimes U_r(t)$$

where:
- $U_\Phi(t)$ is non-unitary (appears thermal when horizon traced out)
- $U_r(t)$ is strictly unitary (discrete, protected)

The entropy calculation:

$$S_{rad}(t) = S_\Phi(t) + S_r(t) - I(\Phi : r)$$

where $I(\Phi : r)$ is the mutual information between components.

The Page curve emerges because:
- Early: $S_r$ is small, $S_\Phi$ dominates (thermal increase)
- Late: $S_r$ correlations dominate, purifying the radiation

---

## 3. Lattice Implementation

### Spacetime Discretization

We discretize the near-horizon region on a lattice:

**Coordinates:** (t, r, θ, φ) → lattice sites

**Near-horizon metric (Schwarzschild):**
$$ds^2 = -\left(1 - \frac{r_s}{r}\right)dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1}dr^2 + r^2 d\Omega^2$$

**Lattice parameters:**
- $N_t$: Temporal sites (Euclidean time for thermal)
- $N_r$: Radial sites (from horizon to asymptotic)
- $N_\theta, N_\phi$: Angular sites (spherical shell)

### Field Content

**Scalar field φ(x):** Test field for Hawking radiation

**Gauge field A_μ(x):** For winding number computation

**Metric perturbation h_μν(x):** Gravitational degrees of freedom

### Winding Number on Lattice

The winding number around the horizon:

$$r = \frac{1}{2\pi} \oint_\gamma A \cdot dl$$

On the lattice:
$$r = \frac{1}{2\pi} \sum_{links \in \gamma} \theta_{link}$$

where $\theta_{link}$ is the U(1) phase on each link.

This is an **integer** — topologically quantized!

---

## 4. Test Protocol

### TEST HAWKING-001-A: Bekenstein-Hawking Entropy

**Purpose:** Verify lattice reproduces correct black hole entropy.

```
1. Initialize black hole of mass M on lattice
2. Compute horizon area A = 4πr_s² = 16πG²M²/c⁴
3. Count microstates via partition function Z
4. Compute entropy S = log(Z)
5. Compare to S_BH = A/4

PASS CRITERION: |S_lattice - S_BH| / S_BH < 0.1
```

---

### TEST HAWKING-001-B: Hawking Temperature

**Purpose:** Verify thermal radiation has correct temperature.

```
1. Initialize black hole at mass M
2. Evolve quantum fields on lattice
3. Measure radiation spectrum at r >> r_s
4. Fit to Planck spectrum, extract T
5. Compare to T_H = ℏc³/(8πGMk_B)

PASS CRITERION: |T_measured - T_H| / T_H < 0.05
```

---

### TEST HAWKING-001-C: Page Curve — Φ Component Only

**Purpose:** Show that Φ alone gives monotonic entropy (Hawking's result).

```
1. Track only continuous geometry Φ
2. Compute entanglement entropy S_Φ(t) of radiation
3. Evolve through Page time

EXPECTED: S_Φ increases monotonically (NO Page curve)
This is the "information loss" scenario.
```

---

### TEST HAWKING-001-D: Page Curve — Full C = (Φ, r)

**Purpose:** Show full system recovers Page curve.

```
1. Track both Φ and discrete winding code r
2. Compute total entropy S_C(t) = S_Φ + S_r - I(Φ:r)
3. Evolve through full evaporation

EXPECTED: S_C follows Page curve
- Rises to Page time
- Falls after Page time  
- Returns to ~0 at end

PASS CRITERION: Page curve shape recovered
```

---

### TEST HAWKING-001-E: Unitarity Verification

**Purpose:** Verify evolution is unitary when including r.

```
1. Initialize pure state |ψ⟩ on lattice
2. Evolve through partial evaporation
3. Compute purity P = Tr(ρ²) of full system

For unitary evolution: P = 1 (pure state)
For non-unitary: P < 1 (mixed state)

PASS CRITERION: 
- P_Φ < 1 (geometry alone is non-unitary)
- P_{Φ,r} ≈ 1 (full system is unitary)
```

---

### TEST HAWKING-001-F: Winding Number Conservation

**Purpose:** Verify topological protection of r.

```
1. Initialize state with winding number r_0
2. Evolve through evaporation
3. Track total winding r(t) = r_BH(t) + r_rad(t)

PASS CRITERION: r(t) = r_0 for all t (conserved)
```

---

### TEST HAWKING-001-G: Island Emergence

**Purpose:** Show that "islands" emerge naturally from the r contribution.

The island formula says:
$$S_{rad} = \min_I \left[ \frac{Area(\partial I)}{4G} + S_{bulk}(I \cup rad) \right]$$

In Davis-Wilson:
- The minimization corresponds to r finding optimal winding configuration
- Islands are regions where r contributes to radiation entropy

```
1. Compute entropy with and without island contribution
2. Verify island dominates after Page time
3. This emerges automatically from r dynamics

PASS CRITERION: Island contribution matches winding code entropy
```

---

## 5. Numerical Parameters

### Black Hole Parameters

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Initial mass | M_0 | 100 | Planck masses |
| Schwarzschild radius | r_s | 2GM/c² | Planck lengths |
| Initial entropy | S_0 | 4πM² | Dimensionless |
| Hawking temperature | T_H | 1/(8πM) | Planck temperature |
| Page time | t_Page | ~M³ | Planck times |
| Evaporation time | t_evap | ~M³ | Planck times |

### Lattice Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Temporal sites | N_t | 64 | Euclidean time |
| Radial sites | N_r | 128 | Horizon to infinity |
| Angular sites | N_θ × N_φ | 32 × 64 | Spherical shell |
| Lattice spacing | a | 0.1 r_s | Resolution |
| UV cutoff | Λ | 10/r_s | High frequency cutoff |
| IR cutoff | L | 100 r_s | Box size |

### Evolution Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Time steps | 10000 | Through evaporation |
| dt | 0.01 t_Page | Time resolution |
| Monte Carlo sweeps | 1000 | Per time step |
| Thermalization | 100 | Initial sweeps |

---

## 6. GPU Implementation Strategy

### Parallelization Opportunities

| Component | Parallelization | Expected Speedup |
|-----------|-----------------|------------------|
| Lattice field updates | Per-site parallel | ~1000x |
| Entropy computation | Eigenvalue parallel | ~100x |
| Monte Carlo sampling | Independent chains | ~100x |
| Winding number sum | Reduction kernel | ~50x |
| Correlation functions | FFT-based | ~200x |

### Memory Requirements

- Lattice fields: N_t × N_r × N_θ × N_φ × (fields) × 8 bytes
- For default parameters: ~2 GB
- Fits comfortably on RTX 5070 (8 GB VRAM)

### CUDA Kernels Needed

1. `update_field_kernel`: Metropolis updates for scalar field
2. `winding_number_kernel`: Compute winding on angular slices
3. `entropy_kernel`: Compute entanglement entropy
4. `correlation_kernel`: Two-point functions for temperature
5. `holonomy_kernel`: Parallel transport around horizon

---

## 7. Theoretical Predictions

### From Davis-Wilson Framework

**Prediction 1:** Φ-only entropy is monotonic (Hawking)
$$\frac{dS_\Phi}{dt} > 0 \quad \text{always}$$

**Prediction 2:** Full entropy follows Page curve
$$S_C(t) = S_\Phi(t) + S_r(t) - I(\Phi:r)(t)$$
with turnover at Page time.

**Prediction 3:** Winding number is conserved
$$\frac{dr}{dt} = 0 \quad \text{(topologically protected)}$$

**Prediction 4:** Late-time purity approaches 1
$$\lim_{t \to t_{evap}} Tr(\rho_C^2) = 1$$

**Prediction 5:** Information rate bounded by winding
$$\frac{dI}{dt} \leq \frac{c \cdot \Delta r}{r_s}$$

### Quantitative Benchmarks

| Time | S_Φ/S_0 | S_C/S_0 | Purity |
|------|---------|---------|--------|
| 0 | 0 | 0 | 1.0 |
| 0.25 t_evap | 0.25 | 0.25 | 0.95 |
| 0.5 t_evap (Page) | 0.5 | 0.5 | 0.90 |
| 0.75 t_evap | 0.75 | 0.25 | 0.95 |
| t_evap | 1.0 | ~0 | ~1.0 |

---

## 8. Connection to Other Tests

### Yang-Mills (YM-001)

The lattice technology is similar:
- Same gauge field discretization
- Same winding number computation
- Same Monte Carlo methods

### Schoenflies (SCHOEN-001)

The topological protection of r is the same principle:
- "Winding code is homological"
- Discrete invariants are conserved
- Topology protects information

### Riemann Hypothesis (RH-001)

The partition function approach connects:
- Z_BH related to zeta function
- Entropy from log(Z)
- Spectral methods apply

---

## 9. Success Criteria

| Test | Metric | Pass | Strong Pass |
|------|--------|------|-------------|
| HAWKING-001-A | S_lattice vs S_BH | < 10% error | < 5% error |
| HAWKING-001-B | T_measured vs T_H | < 5% error | < 2% error |
| HAWKING-001-C | S_Φ monotonic | Yes | No turnover |
| HAWKING-001-D | Page curve shape | Turnover visible | Quantitative match |
| HAWKING-001-E | Purity P_{Φ,r} | > 0.9 | > 0.99 |
| HAWKING-001-F | Winding conservation | Δr < 0.1 | Δr < 0.01 |
| HAWKING-001-G | Island emergence | Visible | Matches formula |

### Overall Verdict

| Result | Interpretation |
|--------|----------------|
| **7/7 PASS** | Information paradox RESOLVED in Davis-Wilson |
| **6/7 PASS** | Strong evidence, minor refinement needed |
| **5/7 PASS** | Framework supported, investigate failures |
| **< 5/7** | Reconsider model assumptions |

---

## 10. Physical Interpretation

### What the Test Proves

If all tests pass, we have demonstrated:

1. **Information is NOT lost:** It's preserved in the discrete winding code r, which is topologically protected.

2. **Page curve is natural:** The turnover comes from the r component dominating at late times.

3. **Unitarity holds:** The full system (Φ, r) evolves unitarily even though Φ alone appears thermal.

4. **Islands are winding contributions:** The mysterious "island" regions are simply where the winding code stores radiation entropy.

5. **No firewall needed:** Information escapes via r, not through violent modifications to the horizon geometry.

### Resolution of the Paradox

The paradox arose from considering only Φ (continuous geometry).

**Hawking's view:** Track Φ → thermal radiation → information lost

**Davis-Wilson view:** Track C = (Φ, r) → r is conserved → information preserved

The discrete winding code is the "hidden channel" through which information escapes!

---

## Appendix A: Lattice Action

The Euclidean action on the lattice:

$$S = S_{gravity} + S_{matter} + S_{gauge}$$

**Gravity (linearized):**
$$S_{gravity} = \sum_x \frac{1}{16\pi G} \left[ (\partial h)^2 + \text{curvature terms} \right]$$

**Matter (scalar field):**
$$S_{matter} = \sum_x \left[ \frac{1}{2}(\partial\phi)^2 + \frac{1}{2}m^2\phi^2 + V(\phi) \right]$$

**Gauge (for winding):**
$$S_{gauge} = \sum_{plaquettes} \frac{1}{g^2} \left[ 1 - \cos(\theta_P) \right]$$

---

## Appendix B: Page Curve Derivation

The Page curve follows from random matrix theory.

For a bipartite system with dimensions $d_A$ and $d_B$:

$$\langle S_A \rangle = \log(\min(d_A, d_B)) - \frac{\min(d_A, d_B)}{2 \max(d_A, d_B)}$$

For black hole + radiation:
- $d_{BH} \propto e^{S_{BH}} \propto e^{M^2}$
- $d_{rad} \propto e^{S_{rad}}$
- Page time: when $d_{BH} = d_{rad}$

---

## Appendix C: Winding Number and Holonomy

The winding number around the horizon:

$$r = \frac{1}{2\pi i} \oint_{S^1} d\log\psi$$

where ψ is a charged field.

Equivalently, the holonomy:
$$W[\gamma] = \mathcal{P} \exp\left( i \oint_\gamma A \right)$$

For a single winding: $W = e^{2\pi i r}$

This is **quantized** because $\pi_1(S^1) = \mathbb{Z}$.

---

*"Information cannot be destroyed. It can only change form — from Φ to r."*

**The Davis Law: C = (Φ, r)**

When winding is homological, information is eternal.

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Status:** Ready for Implementation
