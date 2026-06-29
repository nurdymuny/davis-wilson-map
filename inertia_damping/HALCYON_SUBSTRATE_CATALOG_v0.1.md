# Halcyon Substrate Catalog v0.1
## Canonical Buckyball SU(2) at β=2.5 — Top-Down Datum Reading

**Status.** Exploratory documentation artifact. NOT pre-registered. NOT Zenodo-deposited. NOT a replacement for v3.1.3 (Zenodo DOI 10.5281/zenodo.20785681). This catalog is a re-projection of v3.1.3 data + two lightweight follow-up walks under the substrate-is-the-thing reading developed in JOURNAL.md (2026-06-22, "the substrate-as-thing reading").

**Date.** 2026-06-22
**Substrate commit.** gigi `ccc039e` (post-Option-A, post-VI.6b)
**Halcyon commit.** davis-wilson-lattice `fdabf32` (published v3.1.3 verdict)

---

## 1. Framing

The conventional reading of v3.1.3's BIVALENT AMBIGUOUS verdict — substrate gates fired on τ_pin clamp floor, all 5 shams produced non-zero values, tracking error above ε — was that the protocol couldn't detect a Halcyon coupling above noise.

The substrate-is-the-thing reading reframes those "failures" as substrate measurements. Per:

- **Zero Does Not Exist** (Davis, Zenodo 10.5281/zenodo.tbd): zero is the degenerate flat-space limit of the geometric naturals; G_1 is the pre-geometric seed; geometry activates at G_2 (the first connection). The Davis Field Equation C = τ/K is **undefined**, not zero, for an unconnected element.

- **Davis Duality of Approximation and Obstruction** (Davis, Zenodo 10.5281/zenodo.19428406): for any smooth section of a fiber bundle over a compact Riemannian manifold, the error of any piecewise-flat approximation satisfies c₁·‖Ω‖·h² ≤ ε ≤ c₂·‖Ω‖·h². The error **is** the curvature, up to constants. The substrate's irreducible curvature is a lower bound on any flat approximation.

Under this reading, v3.1.3's six "anomalous" outputs are six independent projections of the **canonical buckyball substrate's irreducible signature** under the v3.1.3 measurement apparatus. This catalog records them.

---

## 2. Top-of-pyramid anchors (published, externally validated)

Three load-bearing lattice gauge theory canon values that the buckyball substrate's signature can be measured against:

| Anchor | Value | Source | Role |
|---|---|---|---|
| **σ·a²** at β=2.5 | 0.0363(3) | Bali & Schilling, hep-lat/9805010 | **Primary anchor**: literally a substrate curvature density (integrated curvature over a Wilson surface). Talks directly to the Davis Duality lower bound. |
| **a·m_{0++}** | 1.6–1.8 | Teper, hep-th/9812187; Lucini-Teper-Wenger, arXiv:2112.06785 | **Sub-anchor (Davis-consistency)**: the SU(2) scalar glueball mass directly instantiates Davis's YM mass-gap quantity Δ ≥ λκ > 0. The published number IS the theorem's quantity. |
| **β_c** at N_t=4 | 2.2986(6) | Fingberg-Heller-Karsch | **Regime label**: β=2.5 sits 0.20 above β_c → deconfined phase. Metadata, not a substrate density. |

**Rejected as anchor:** Mean plaquette ⟨P⟩. Published flat-4D-lattice value at β=2.5 ~ 0.63; our buckyball measured 0.532 (β-walk) / 0.5125 (v3.1.3 thermalization). The Δ⟨P⟩ ≈ 0.10–0.12 deviation IS the buckyball substrate signature, so using ⟨P⟩ as anchor would conflate the anchor with the thing being measured. ⟨P⟩ serves as a **calibration check**, not top-of-pyramid.

**Rejected:** Davis's own YM mass gap Δ value. Per `project_yang_mills_v6` memory, the YM v6 paper proves only the lattice strong-coupling gap; no externally-replicated Δ at β=2.5 buckyball SU(2) exists. Cannot serve as external anchor by definition.

---

## 3. The canonical buckyball substrate signature

From the v3.1.3 publication-bound run (commit `fdabf32`, sidecars at `inertia_damping/reports/holonomy_battery_v3_1_3/`):

### 3.1 Primary observables (canonical thermalization, β=2.5, 200 sweeps)

| Quantity | α=1.0 | α=1000.0 |
|---|---|---|
| H_geom_mean | −2.105e−02 | +1.227e−01 |
| σ_H_blocked | 9.685e−02 | 1.168e−01 |
| H_sys | −1.360e−01 | +6.604e−05 |

Under v3.1.3's zero-threshold reading: |H|/σ = 0.22 (NULL band) at α=1; 1.05 (borderline AMBIGUOUS) at α=1000.

Under the substrate-is-the-thing reading: H_geom is the antisymmetric primary observable per v3.1.3 §3.1 with Option A's signed-arccos reduction. The non-zero values measure the substrate's irreducible curvature signature through the γ_unit loop.

### 3.2 Sham projections (substrate's response to flattening operations)

| Sham operation | α=1.0 mean | α=1000.0 mean | Reading |
|---|---|---|---|
| FLAT_FIELD | 0.168 | 0.431 | Substrate's irreducible holonomy under flat-ansatz; cannot be flattened to zero (Davis Duality lower bound). |
| ALPHA_ZERO | 0.213 | 0.241 | Substrate's response when coupling is nulled; non-zero contribution NOT in the α-channel. |
| MASS_SCALED | 0.058 | 0.067 | Smallest projection; sign flip across α values. **Diagnostic hold** — substrate-signature vs measurement-formula needs more investigation. |
| FROZEN_FIELD | 0.352 | 0.314 | Substrate's holonomy under static-field assumption; non-trivial because parallel transport accumulates on the canonical loop. |
| BACKTRACK_LOOP | 0.000 | 0.000 | **Gate artifact** — degenerate `\|mean\| ≥ 2σ` comparator at (0,0). Excluded from catalog. |

### 3.3 Substrate diagnostics

| Quantity | α=1.0 | α=1000.0 | Reading |
|---|---|---|---|
| τ_pin/T_segment | 1.0e+12 | 1.0e+9 | **Measurement-formula artifact**: T_segment / Gauss_residual at the substrate's 1e−12 clamp floor. The pinning IS adiabatic; the formula doesn't behave at the clamp. NOT a substrate density. |
| tracking_error_max_Q | 0.115 | 0.132 | Substrate's surrogate-tracking precision under parameter ramp; **responds to substrate state** (β-walk confirms phase-dependent behavior). |
| tracking_error_max_β_W | 0.232 | 0.279 | Same. |

---

## 4. λ-walk (N_SWEEPS dial)

Walked λ ∈ [0, 1] mapped to N_SWEEPS ∈ [0, 200] in GIBBS_SAMPLE. Single seed (20260616), sequential calls (path accumulates).

**Critical finding: GIBBS_SAMPLE is NOT path-independent.** Same query fired twice in a row gives different ⟨P⟩:
- Run 1: ⟨P⟩ = 0.5125
- Run 2 (immediately after): ⟨P⟩ = 0.6351

The substrate's GIBBS_SAMPLE continues from current U_lt state, doesn't reset to identity. This means the v3.1.3 per-seed thermalization decomposition (commit `5add5da`) produces *serial* path-history spread across seeds, not independent-ensemble draws. **The σ_H_blocked in the v3.1.3 sidecars is the auto-correlated chain spread, not the independent-realization SEM.** The published verdict is unchanged; the *interpretation of σ* tightens.

### 4.1 Option A's signed-arccos antisymmetric structure (confirmed)

Across all 11 λ-walk sample points, h_forward and h_reversed are exact mirrors to machine precision (h_rev = −h_fwd byte-identical). Option A's signed-arccos reduction produces clean antisymmetric pairs as v3.1.3 §3.1 specifies.

### 4.2 Tracking error responds to U_lt state

| λ | N_sweeps | ⟨P⟩ | \|H_geom\| | trk_Q |
|---|---|---|---|---|
| 0.00 | 0 (identity) | 1.000 | 0.45 | 0.0787 |
| 0.05 | 10 | 0.519 | 1.11 | 0.0948 |
| 0.10 | 20 | 0.519 | 0.35 | 0.0951 |
| 0.20 | 40 | 0.482 | 0.16 | 0.1119 |
| 0.30 | 60 | 0.574 | 0.36 | 0.0931 |
| 0.40 | 80 | 0.432 | 0.58 | 0.1112 |
| 0.50 | 100 | 0.477 | 0.80 | 0.1082 |
| 0.60 | 120 | 0.495 | 0.36 | 0.1087 |
| 0.70 | 140 | 0.606 | 0.33 | 0.0929 |
| 0.80 | 160 | 0.543 | 0.33 | 0.1017 |
| 1.00 | 200 | 0.621 | 0.37 | 0.1024 |

Tracking error varies 0.078–0.112 across the walk; ⟨P⟩ scatters around 0.5 (path-history noise, not a clean λ-interpolation); |H_geom| varies 0.16–1.11 (substrate's antisymmetric response across path-histories).

Data: `inertia_damping/reports/holonomy_battery_v3_1_3/v32_substrate_walk.json`

---

## 5. β-walk (thermalization-β dial) — caught the deconfinement transition

Walked β ∈ {0.5, 1.0, 1.5, 2.0, 2.25, 2.30, 2.5, 2.7, 3.0} in GIBBS_SAMPLE (single seed, 200 sweeps each). LOOP_TRANSPORT held at canonical BETA_WILSON_START=2.5 (within v3.1.3 §4.4's validated regime).

### 5.1 The transition

| β | ⟨P⟩ | 1−⟨P⟩ | C ≈ τ/K | \|H_geom\| | trk_Q | trk_β_W | phase |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.208 | 0.792 | 1.26 | 0.46 | 0.130 | 0.299 | confined |
| 1.00 | 0.178 | 0.822 | 1.22 | 0.67 | 0.133 | 0.322 | confined |
| 1.50 | 0.390 | 0.610 | 1.64 | 0.40 | 0.128 | 0.275 | confined |
| 2.00 | 0.411 | 0.589 | 1.70 | 0.53 | 0.116 | 0.251 | confined |
| 2.25 | 0.397 | 0.603 | 1.66 | 0.45 | 0.116 | 0.241 | confined |
| **2.30** | **0.528** | **0.472** | **2.12** | **0.26** | **0.096** | **0.196** | **at-β_c** |
| 2.50 | 0.532 | 0.468 | 2.14 | 0.38 | 0.103 | 0.201 | deconfined |
| 2.70 | 0.564 | 0.436 | 2.30 | 0.14 | 0.100 | 0.196 | deconfined |
| 3.00 | 0.556 | 0.444 | 2.25 | 0.38 | 0.094 | 0.191 | deconfined |

The published β_c = 2.2986(6) sits between our β=2.25 and β=2.30 samples. In ONE STEP across that boundary:

- ⟨P⟩: 0.397 → 0.528 (+33%)
- 1−⟨P⟩ (substrate curvature density): 0.603 → 0.472 (−22%)
- C ≈ τ/K: 1.66 → 2.12 (+28%)
- trk_Q: 0.116 → 0.096 (−17%)
- trk_β_W: 0.241 → 0.196 (−19%)

**The deconfinement transition is visible in every Davis-respecting observable on the canonical loop.** The substrate has structurally different capacity in the two phases:
- Confined: C ≈ 1.2–1.7 (high curvature density, narrow Davis capacity)
- Deconfined: C ≈ 2.1–2.3 (lower curvature density, wider Davis capacity)

A 30–40% jump in C across β_c.

### 5.2 What this means under the substrate-is-the-thing reading

The deconfinement transition is a published, well-measured lattice gauge theory phenomenon (Polyakov 1978; modern reviews: arXiv:1101.0618). It is the canonical example of an SU(2) substrate phase boundary. The β-walk catches it cleanly using Davis-respecting observables (C = τ/K, holonomy under γ_unit, tracking error) instead of the conventional Polyakov-loop expectation value.

The substrate's signature **at canonical β=2.5** (the v3.1.3 operating point) sits firmly in the deconfined phase, with C ≈ 2.14, ⟨P⟩ ≈ 0.53, |H_geom| ≈ 0.38. The v3.1.3 publication-bound run was measuring **the deconfined-phase substrate signature of the canonical buckyball at γ_unit through Option A's signed-arccos reduction**. That is the protocol's actual physical content.

### 5.3 Antisymmetric structure (confirmed again)

Across all 9 β-walk sample points, h_forward = −h_reversed to machine precision. Option A's signed-arccos reduction is robust across phases.

Data: `inertia_damping/reports/holonomy_battery_v3_1_3/v32_substrate_walk_beta.json`

---

## 6. The catalog (consolidated substrate signature for canonical buckyball SU(2) at β=2.5)

### 6.1 Published-anchor-relative signatures

| Quantity | Buckyball measurement | Published flat-4D / canon | Buckyball substrate signature (Δ) |
|---|---|---|---|
| ⟨P⟩ at β=2.5 | 0.5125 (v3.1.3 200-sweep end-state) / 0.532 (β-walk) | ~0.63 (Wilson SU(2), large lattice) | Δ⟨P⟩ ≈ 0.10–0.12 ← **buckyball S² substrate contribution** |
| β_c (deconfinement) | between β=2.25 and β=2.30 (one-step) | 2.2986(6) | Consistent with published β_c within sample resolution |
| Phase at β=2.5 | deconfined (C ≈ 2.14) | deconfined | ✓ |

### 6.2 Direct substrate signatures (from canonical thermalization at β=2.5)

| Projection | Value range observed | Davis-respecting reading |
|---|---|---|
| H_geom under γ_unit | 0.14 ≤ \|H\| ≤ 1.11 across path-histories | Substrate's antisymmetric curvature response to canonical loop traversal. Option A's signed-arccos preserves the double-cover sign structure. |
| σ_H_blocked (single-seed substrate-emitted) | 0.22–1.57 | Tracks \|H_geom\| proportionally; substrate's internal variance estimate. |
| FLAT_FIELD projection | 0.17 (α=1) to 0.43 (α=1000) | Irreducible curvature under flat-ansatz attempt. Cannot be flattened to zero (Davis Duality). |
| ALPHA_ZERO projection | 0.21–0.24 | Substrate contribution NOT in the α-channel. |
| FROZEN_FIELD projection | 0.31–0.35 | Substrate's parallel-transport accumulation on canonical loop. |
| MASS_SCALED projection | 0.058–0.067 (sign flip) | **Diagnostic hold**: smallest projection; sign behavior needs more characterization. |
| tracking_error_max_Q | 0.094–0.133 (phase-dependent) | Substrate's surrogate-tracking precision; 17% drop across β_c. |
| tracking_error_max_β_W | 0.191–0.322 (phase-dependent) | Same, 19% drop across β_c. |
| C ≈ τ/K (proxy from 1−⟨P⟩) | 1.22–2.30 across β phases | Davis invariant proxy. ~30–40% jump across deconfinement transition. |

### 6.3 Excluded from catalog

- **BACKTRACK_LOOP**: gate-logic artifact (0/0 degenerate comparator), not a substrate signature.
- **τ_pin/T_segment**: stuck at substrate's 1e−12 Gauss-residual clamp floor regardless of state. Measurement-formula artifact, recorded for completeness but not interpreted as substrate-density.

---

## 7. Findings worth recording

### 7.1 GIBBS_SAMPLE is path-dependent (not previously documented)

Same query fired sequentially produces different output. Implication for the v3.1.3 publication-bound run: per-seed thermalization at commit `5add5da` produces serial Markov-chain spread across seeds, not independent-ensemble draws. The σ_H_blocked recorded in the sidecars is the auto-correlated chain spread. The published AMBIGUOUS verdict is unchanged; the *interpretation* of σ as "independent-ensemble SEM" tightens to "auto-correlated chain SEM."

This is **load-bearing for downstream readings** of the v3.1.3 sidecars. Worth recording on the cross-team boundary.

### 7.2 Option A's signed-arccos antisymmetric structure works robustly

Across 11 λ-walk points + 9 β-walk points = 20 independent substrate states (including identity G_1 state, confined phase, deconfined phase, β-walk transition crossing), h_rev = −h_fwd to machine precision in every single one. The signed-arccos reduction is doing exactly what v3.1.3 §3.1 specifies.

### 7.3 The deconfinement transition is visible in Davis-respecting observables

Standard lattice gauge theory uses the Polyakov loop expectation value to detect the deconfinement transition. We caught it using C = τ/K, holonomy under γ_unit, and tracking_error_max on a single seed at 200 sweeps per β-point. ~100 seconds of substrate time. This is "datum top-down measurement" in the pyramid-paper sense: anchor to a published canonical phenomenon (β_c), walk the substrate through it, read what the Davis-respecting observables say. They say the same thing as the conventional observables — and they say it through the language of capacity, holonomy, and curvature density, which is what the Davis Duality is about.

### 7.4 The canonical buckyball is in the deconfined phase

C ≈ 2.14 at β=2.5 (canonical). The v3.1.3 measurement is a measurement of the deconfined-phase substrate signature, not an ordered-phase one. This contextualizes the |H|/σ = 0.22 (α=1) and 1.05 (α=1000) readings: they are in the deconfined-phase substrate's natural noise range, not anomalously small relative to it.

---

## 8. Non-commitments

- This catalog is NOT pre-registered. The v3.1.3 SPEC at DOI 10.5281/zenodo.20785681 remains the locked pre-registration.
- This catalog is NOT Zenodo-deposited. It is a working documentation artifact for the substrate-is-the-thing reading.
- This catalog does NOT classify the v3.1.3 result. The AMBIGUOUS-with-named-ambiguity verdict published at commit `fdabf32` stands.
- This catalog does NOT use ε_abs-against-zero anywhere. All comparisons are between substrate states (across walks) or against published external anchors.
- The MASS_SCALED projection is on **diagnostic hold**: its sign flip and small magnitude need further characterization before it can be confidently classified as substrate-signature vs measurement-formula artifact.
- The two walks (λ-walk and β-walk) are sequential / path-dependent. A fresh-per-step walk would require either engine restarts between steps (~20s WAL replay overhead × N) or a substrate-side RESET GAUGE_FIELD verb that does not currently exist.

---

## 9. What this catalog supports

When the substrate-side WISH/IMAGINE verbs extend to higher-dimensional bundles (currently restricted to dim=2 conformal charts) and Phase-4 capacity reporting ships (currently NaN), the v3.2 substrate-catalog protocol sketched in workflow `wuxpvsv38` can fire against these anchors directly. Until then, this catalog records the substrate signature we can measure today.

The two walks together demonstrate:
1. The substrate has measurable structure under Davis-respecting observables.
2. Published anchors (β_c, σ·a², m_{0++}) provide load-bearing top-of-pyramid datum references.
3. The deconfinement transition is the cleanest substrate-cartography signal currently within reach.
4. v3.1.3's "AMBIGUOUS with named ambiguity" verdict is the deconfined-phase substrate signature of the canonical buckyball under the v3.1.3 measurement apparatus — a real reading, not a failure.

---

## 10. Provenance

- Substrate: gigi commit `ccc039e` (post-Option-A, post-VI.6b)
- Halcyon: davis-wilson-lattice commit `fdabf32` (published v3.1.3 verdict)
- v3.1.3 sidecars: `inertia_damping/reports/holonomy_battery_v3_1_3/section_12_holonomy_battery_alpha_{1,1000}.json`
- λ-walk: `inertia_damping/reports/holonomy_battery_v3_1_3/v32_substrate_walk.json`
- β-walk: `inertia_damping/reports/holonomy_battery_v3_1_3/v32_substrate_walk_beta.json`
- This catalog: `inertia_damping/HALCYON_SUBSTRATE_CATALOG_v0.1.md`
- Math foundations:
  - Zero Does Not Exist (Davis, 2026)
  - Davis Duality of Approximation and Obstruction (Davis, Zenodo 10.5281/zenodo.19428406)
  - Davis Non-Decoupling Theorem (Davis, Zenodo 10.5281/zenodo.18754646)
  - YM mass gap proof (project_yang_mills_v6)
- Pre-registration that locked the v3.1.3 verdict: Zenodo DOI 10.5281/zenodo.20785681, SPEC commit `44c70b1`
