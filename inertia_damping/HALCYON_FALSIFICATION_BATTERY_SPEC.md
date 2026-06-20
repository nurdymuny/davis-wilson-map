# Halcyon Falsification Battery — SPEC (v2)

**Status:** design-locked, ready to implement
**Date:** 2026-06-20
**Predecessors:** Solves Vol. 4 Appendix A.6 (the thread experiment + dynamic
transfer-function target), HALCYON_USE_GIGI_FLAG_SPEC.md (the integration pattern this follows)

## v2 changelog (responses to the 2026-06-20 adversarial review)

Four critics reviewed v1: math-dim, sim-realism, sudoku-completeness, adversarial-skeptic. Their blockers and high-priority should-fixes are all addressed here. The major shape changes:

- **τ_Q model** rewritten: `τ_Q(e) = τ_0 / (1 + β_τ · s_Q(e))` instead of the v1 `√(1/staple)` placeholder. Recovers Newtonian limit at Q=0 (`s_Q=0 → τ_Q=τ_0` uniform) and stays bounded at strong field. Dimensions stated explicitly.
- **Discrete sum** carries an explicit `1/E` per-edge weight (the uniform measure valid under the buckyball's icosahedral symmetry) and α_Halcyon is given dimensions.
- **Implementation architecture** revised: new `test_mass_dynamics.py` module with state `(U, E, x, v)`, not an extension of the existing CUDA Section 5 kernel. CUDA batching across seeds is dropped for driven evolution (seeds are not a parallelism axis under non-equilibrium dynamics). Wall budget revised to ~2–3 hr CPU or ~30 min with multiprocessing over (ω, Q) cells.
- **χ²/dof fit** redefined: 8 seeds × 5 ω × 3 Q yields 15 weighted means with SEM; 9 free parameters (K_Q, μ_Q, c_Q × 3) leaves dof=6, not dof=0. The fit operates on seed-collapsed means with FP-blocked SEM weights.
- **α_predicted** moved from "match within 2σ" gate (which was circular) to a separate self-consistency check that is NOT load-bearing for falsifiability. The load-bearing gates are H₀–H₉ only.
- **μ_proxy** defined as an explicit multiplicative scale factor on the **bare material baseline** `μ_total = μ_proxy · μ_baseline + μ_eff(Q)`. μ_proxy does NOT enter μ_eff: the Halcyon coupling is by claim material-independent. H₁ test asks: vary μ_proxy → does α extracted from (Q, μ_total(Q, μ_proxy)) pairs stay constant? If yes, the slope is geometric. (The earlier v2 draft had μ_proxy multiplying μ_eff directly; that made the test tautologically fail and was the wrong physics.)
- **H₉ added**: τ_Q model robustness — alternative functional forms should produce α within tolerance. If they don't, the model is overfitted.
- **Q_surrogate** defined explicitly as the normalized plaquette deviation from canonical, with the Q-sector labels {0,1,2} mapped to specific surrogate ranges.
- **Gate robustness slack** added: H₀ requires 6σ (not 5σ); per-seed gate independence required (≥5/8 seeds must independently strike each gate).
- **Ergodicity caveat** added explicitly, citing the Section 5 closure receipt's 16% shell-vs-ensemble gap.

## 1. The premise

The Halcyon experimental claim, per A.6 Eq. (5):

$$
\partial_Q \mu_Q \;\stackrel{?}{\ne}\; 0
\quad\text{at fixed } K_Q,\, c_Q,\, \text{drive amplitude, thermal \& EM systematics.}
$$

Falsifiability is supplied by enumerating the null space and striking through what the apparatus can rule out (Davis Method Principle 5: *elimination, not construction*). The completion invariant is the count of struck rows in the simulatable subset. Halcyon's claim is load-bearing only when the entire simulatable subset is struck *in advance* — gates fire deterministically on the JSON output of every run, not in post-hoc interpretation.

## 2. The grid (H₀…H₉)

| # | Null hypothesis | Protocol | Realm |
|---|---|---|---|
| H₀ | nothing happens (α = 0) | drive sweep at multiple Q; gate `|α| > 6 σ_α` (tightened from 5σ for robustness slack) | simulation ✓ |
| H₁ | a material effect (Eötvös-class) | vary `μ_proxy ∈ {0.5, 1.0, 2.0}` as multiplicative scale on `μ_baseline` (material mass, NOT on μ_eff); verify α extracted from (Q, μ_eff(Q)) pairs satisfies `|dα/dμ_proxy| / |α| < 0.05` | simulation ✓ |
| H₂ | thermal pickup | thermal sham — drive heater profile with Q=0 → no shift | **hardware only** |
| H₃ | EM pickup | vary test-mass `χ_mag`; predicted slope does NOT scale with `χ_mag` | **hardware only** |
| H₄ | mechanical pickup | accelerometer on mount; subtract correlated motion | **hardware only** |
| H₅ | drive-amplitude artifact (nonlinear saturation) | linearity sweep at fixed (Q, ω); F-test vs. quadratic, `F < F_crit(α=0.05)` | simulation ✓ |
| H₆ | single-frequency resonance | full χ_Q(ω) sweep; 3-param linear-response fit; `χ²/dof < 1.5` over 15 data points (dof=6) | simulation ✓ |
| H₇ | statistical fluctuation | FP-blocked SEM at plateau ≤ 3% of α; ≥5/8 seeds individually strike | simulation ✓ |
| H₈ | Q drift (sector not held) | `std(Q_surrogate)/⟨Q_surrogate⟩ < 0.03` over flat window | simulation ✓ |
| **H₉** | **τ_Q model error (overfitted functional form)** | **re-fit α with alternative `τ_Q^{alt}(e) = τ_0 · exp(-β_τ · s_Q(e))`; verify α stable within `±20%`** | **simulation ✓** |

Seven simulatable, three hardware-only. The three hardware-only rows are predictions to be checked against the eventual apparatus, not "open items" for the simulation.

## 3. The math the simulation needs

### Equations of motion

Per A.6 Eq. (3), the test mass obeys

$$
\mu_Q\,\ddot{x} + c_Q\,\dot{x} + K_Q\,x \;=\; F(t)
$$

with `μ_Q = K_Q · τ_μ²` (A.6 Eq. (2)) — the squared *inertial* relaxation time, not the viscous one.

### τ_Q model (revised — v1 placeholder rejected)

Per-edge local inertial relaxation time:

$$
\boxed{\;\tau_Q(e) \;=\; \frac{\tau_0}{1 + \beta_\tau\, s_Q(e)}\;}
$$

where

- `τ_0`: base inertial time at trivial vacuum, dimensions [time], default `τ_0 = 1.0` in lattice units
- `β_τ`: dimensionless inertial-coupling strength, default `β_τ = 2.0` (calibrates to ~30% τ-reduction at typical s_Q values)
- `s_Q(e)`: normalized local Wilson-action density at edge e:
  $$
  s_Q(e) \;=\; \frac{1}{4}\left[ (1 - q_0(U_{f_1(e)})) + (1 - q_0(U_{f_2(e)})) \right]
  $$
  where `f_1(e), f_2(e)` are the two faces sharing edge `e`, and `q_0(U_f)` is the scalar (real) part of the face holonomy quaternion. By construction `s_Q(e) ∈ [0, 1]`.

**Limits verified:**
- Q=0 trivial vacuum: `U_f = identity → q_0 = 1 → s_Q(e) = 0 → τ_Q(e) = τ_0` uniform ✓ (Newtonian limit recovered)
- Strong field: `s_Q(e) → 1 → τ_Q(e) → τ_0 / (1 + β_τ) > 0` (bounded, no infinite stiffness) ✓
- Monotone decreasing in field strength ✓
- Always finite, always positive ✓
- C¹ smooth (no kinks) ✓

This functional form is justified by analogy with the dynamical-mean-field response of a coupled oscillator in a slowly relaxing bath: the bath stiffens with local field strength, decreasing the inertial relaxation time. The β_τ coefficient is the analog of the polaron coupling and is in principle calibratable against bench data; in the simulation it is a free parameter the SPEC sets by convention.

### Mode-effective inertial coefficient

Per A.6 Eq. (4), discretized over the 90 buckyball edges with the uniform per-edge measure (valid because the buckyball is icosahedrally symmetric, so all edges are equivalent under graph automorphism):

$$
\boxed{\;\mu_{\rm eff}^{(n)}(Q) \;=\; \alpha_{\rm Halcyon} \cdot \frac{1}{E} \sum_{e=1}^{90} \kappa_Q(e)\, \tau_Q^2(e)\, |\phi_n(e)|^2\;}
$$

where

- `α_Halcyon`: the predicted Halcyon coupling constant. Dimensions: `[mass · time⁻²]` so that `μ_eff` has dimensions [mass] (since `τ²` carries [time²] and `κ_Q`, `|φ_n|²` are dimensionless). Default `α_Halcyon = 1.0` in lattice units.
- `κ_Q(e)`: dimensionless local curvature, identified with the Wilson-action local density at edge e:
  $$\kappa_Q(e) = (1 - q_0(U_{f_1(e)})) \cdot (1 - q_0(U_{f_2(e)}))$$
  (product, not sum, so κ_Q distinguishes from s_Q used in τ_Q; this prevents the two factors from collapsing into a single observable)
- `|φ_n(e)|²`: squared n-th eigenvector of the buckyball signed-incidence graph Laplacian `L_G = D Dᵀ`, where D is the 60×90 signed incidence matrix. Default `n = 1` is the lowest non-trivial eigenmode (first nonzero eigenvalue of L_G, excluding the constant null vector).
- `E = 90`: total edge count.

The eigenmodes are precomputed once at startup via `scipy.sparse.linalg.eigsh(L_G, k=10, sigma=0)` and cached as a 90-entry array per mode.

For the icosahedral symmetry caveat (sudoku-completeness finding #2): the lowest non-trivial eigenspace of L_G on the buckyball is multi-dimensional (icosahedrally degenerate). For the SPEC the default φ_1 is the lex-first eigenvector returned by scipy's solver; the H₆ protocol's broadband-consistency check implicitly validates that the choice within the degenerate subspace does not bias the fit, and the H₉ test explicitly validates that the choice of mode is not load-bearing.

### Q_surrogate definition

The buckyball at SU(2) on S² has `π₂(SU(2)) = 0`, so there is no genuine topological charge. Q_surrogate is therefore an *operational* sector label:

$$
Q_{\rm surrogate}(t) \;=\; \frac{\bar{P}(t) - P_{\rm canonical}}{P_{\rm canonical}}
$$

where `P̄(t)` is the instantaneous mean plaquette across all 32 faces and `P_canonical = 0.5072` (Migdal–Witten at β=2.5). The Q-sector labels are mapped:

| Q label | Q_surrogate range | Achieved by |
|---|---|---|
| 0 | `[-0.05, +0.05]` | canonical thermalized init (baseline) |
| 1 | `[+0.10, +0.20]` | "quench up" — biased link initialization toward q_0 < 1 (rougher field) |
| 2 | `[+0.30, +0.50]` | "quench down" reverse-biased; uses the existing scenario protocol from the cage simulator |

The Q labels are operational, not topological. The sector-separation gate (added per sudoku-completeness finding #3) verifies that the three Q clusters are linearly separable in Q_surrogate space at >3σ.

### α_predicted (NOT load-bearing for falsifiability)

α_predicted is computed from the SPEC's own model:

$$
\alpha_{\rm predicted} \;=\; \left.\frac{\partial \mu_{\rm eff}}{\partial Q}\right|_{\rm default\ parameters}
$$

via finite differences over the three Q labels. **This prediction comes from the same model that generates the simulation, and the "α_measured matches α_predicted within 2σ" check is therefore a self-consistency check, not an independent falsifier.** It is reported in the JSON but does NOT contribute to the load-bearing PASS/FAIL chain.

The independent prediction required for hardware-stage falsification (a closed-form derivation of α_Halcyon from the Davis Field Equations at β=2.5) is a separate work item, tracked in the open-items list and Appendix A.7. The simulation cannot perform that derivation; it can only verify the model's internal consistency, which is what it does.

This is an honest concession from v1 and is the load-bearing repair to the "circularity" blocker.

## 4. The protocols (simulation-side)

Each H_i corresponds to a protocol function in `inertia_damping/falsification_battery.py`:

```python
def protocol_H0_nothing(runs, ...) -> Verdict
def protocol_H1_material(runs, ...) -> Verdict
def protocol_H5_amplitude(runs, ...) -> Verdict
def protocol_H6_resonance(runs, ...) -> Verdict
def protocol_H7_statistics(runs, ...) -> Verdict
def protocol_H8_q_drift(runs, ...) -> Verdict
def protocol_H9_tau_model(runs, ...) -> Verdict
```

Each returns:

```python
@dataclass
class Verdict:
    h_id: str
    struck: bool | str           # True | False | "n/a"
    reason: str
    evidence: Dict[str, Any]
    per_seed_strikes: int        # how many of 8 seeds independently struck this gate
```

A gate is struck overall iff `per_seed_strikes ≥ 5` (majority; per adversarial-skeptic finding on gate independence).

### Drive shape (used by H₀, H₁, H₅, H₆, H₉)

The lock-in drive is a sinusoid with a Tukey window envelope (cosine taper at both ends, flat middle):

```python
F(t) = F_0 · w_tukey(t; t_total, α=0.1) · cos(ω t)
```

`α=0.1` window: 5% taper each end. Pre-drive equilibration of `N_equil = 1000` steps with no drive applied (per sim-realism finding on symplecticity / transient bias) before the Tukey-windowed phase begins.

Demodulation: multiply x(t) by `cos(ωt)` and `sin(ωt)`, block-average over the flat middle 90% to extract in-phase and quadrature amplitudes (`X_I, X_Q`); then

$$
|\chi(\omega)| \;=\; \sqrt{X_I^2 + X_Q^2}/F_0, \qquad
\arg\chi(\omega) \;=\; \mathrm{atan2}(X_Q, X_I).
$$

Both `|χ|` and `arg χ` are recorded per (ω, Q, seed) cell. The H₆ fit uses both (10 real numbers per Q × 3 Q = 30 measurements vs. 9 parameters → dof = 21), addressing math-dim's identifiability concern.

### Sweep design

- **ω grid:** 5 frequencies log-spaced across `ω_c · {0.1, 0.3, 1.0, 3.0, 10.0}` where `ω_c = √(K_0 / μ_0)`. Range justified by Deborah-number framing: ω = 0.1 ω_c → De ≪ 1 (quasi-static, stiffness-dominated, constrains K); ω = ω_c → De ≈ 1 (inertial-damping balance, constrains all three); ω = 10 ω_c → De ≫ 1 (inertia-dominated, constrains μ + c).
- **Q grid:** {0, 1, 2}.
- **Amplitude grid (H₅):** `F_0 ∈ {0.1, 0.3, 1.0, 3.0} × F_*` where `F_*` is set to give a Q=0 amplitude response 1% of the test-mass linear-displacement scale.
- **Seed grid:** 8 seeds `{20260616…20260623}` per (ω, Q) cell.
- **τ_Q alternative model (H₉):** `τ_Q^{alt}(e) = τ_0 · exp(-β_τ · s_Q(e))` re-run at 3 ω × 3 Q × 4 seeds.

Total simulation cells (excluding H₉): `5 ω × 3 Q × 8 seeds = 120` main lock-in measurements + `4 amp × 1 ω × 3 Q × 4 seeds = 48` for H₅ + `3 ω × 3 Q × 4 seeds = 36` for H₉ = **204 lock-in runs**.

### Revised wall budget

Per sim-realism finding: each lock-in is one CPU trajectory of N_steps ≈ 4000 with test-mass DOF and demod, ≈ 40s wall on CPU. CUDA batching across seeds is NOT used for driven dynamics (different reason from the Section 5 case). Parallelism across (ω, Q) cells via Python multiprocessing on 8 cores:

```
204 runs × 40 s/run / 8 cores ≈ 17 min wall (multi-process)
204 runs × 40 s/run             ≈ 2.3 hr  wall (single-process)
```

The validation-report default is multi-process; the CI smoke test (`--battery-fast`) uses 1 seed × 3 ω × 2 Q × 2 H values = 12 runs ≈ 1 min.

### Gate definitions (tightened per adversarial-skeptic)

| Gate | v1 threshold | v2 threshold | Justification |
|---|---|---|---|
| H₀ struck | `|α| > 5 σ_α` | `|α| > 6 σ_α` AND `(α_max − α_min) / (2 σ_α_blocked) < 4` | robustness slack + no-outlier-seed |
| H₁ struck | `|dα/dμ_proxy|/|α| < 0.1` | `|α| > 3 σ_α` (precondition) AND `|dα/dμ_proxy|/|α| < 0.05` | precision guard + tightened |
| H₅ struck | χ² ratio < 0.3 | F-test: `F < F_crit(0.05, 1, dof_lin)` | principled statistical test |
| H₆ struck | χ²/dof < 2.0 | χ²/dof < 1.5 with dof = 21 (15 real |χ| + 15 real arg χ − 9 params) | tighter + dof properly accounted |
| H₇ struck | block-SEM ≤ 5% | block-SEM at plateau ≤ 3% of α; seed-to-seed std ≤ 2× blocked-SEM; plateau-detection algorithm specified | tighter + algorithm defined |
| H₈ struck | std(Q)/Q < 0.05 | std(Q_surrogate)/⟨Q_surrogate⟩ < 0.03 over flat window; uses Q_surrogate defined in §3 | tighter + definition supplied |
| H₉ struck | (new) | `|α_alt − α_default| / |α_default| < 0.2` across two τ_Q functional forms | model-robustness test |
| **per-seed independence** | (none) | `per_seed_strikes ≥ 5` for every gate | required for all simulatable gates |

### Plateau-detection algorithm (per math-dim H₇ finding)

```
Compute blocked SEM at K = 2, 4, 8, 16, … until K > N/2.
If SEM(K=2^n) and SEM(K=2^{n+1}) differ by < 2%, mark plateau at K = 2^n
  and report SEM_plateau = SEM(K=2^n).
If no plateau detected, emit struck = "n/a" with reason
  "no SEM plateau detected in blocking range".
```

### Sector-separation gate (per sudoku-completeness Q-aliasing finding)

Before the lock-in measurements, verify the three Q labels actually correspond to three distinct gauge-field configurations:

```
At each Q ∈ {0, 1, 2}, sample Q_surrogate from N_seed thermalized states.
Compute (mean_Q1 − mean_Q0) / SEM_Q0 and similar pairs.
Sector separation passes if all three pairwise distances > 3 σ.
If not: emit "Q labels not separable" as a top-level FAIL on the battery
  (no need to run individual gates; the experiment is ill-posed).
```

## 5. Ergodicity caveat (per adversarial-skeptic finding)

Per `SECTION5_CLOSURE_RECEIPT.md` (the v1.2.1 closure receipt), the buckyball substrate exhibits a ~16% irreducible microcanonical-vs-canonical gap (`P_time ≈ 0.66`, `P_canonical ≈ 0.506`) that does not close under refinement of trajectory length up to 16000 steps. The battery operates on the same substrate and inherits this caveat:

- The battery gates pass or fail *within the microcanonical energy shell sampled by the seed-and-canonical-init combination*, not on the canonical ensemble.
- A `PASS_SIMULATION_ONLY` verdict does **not** imply thermodynamic equilibrium has been reached; it implies the dynamic signal in the shell that was actually sampled is consistent with the predicted form.
- For hardware-stage extension to a larger lattice (e.g. 4D cubic, V >> 60), the ergodicity gap is expected to vanish under the law of large numbers and the battery should re-validate at the new scale; the SPEC does not certify cross-scale invariance.

This caveat is recorded in the JSON output (`ergodicity_caveat` field) and rendered in the Section 11 markdown.

## 6. The output schema

`section_11_falsification_battery` in the JSON report:

```json
{
  "section_11_falsification_battery": {
    "available": true,
    "alpha_measured": 1.42e-6,
    "alpha_sem_blocked": 8.3e-8,
    "alpha_predicted_self_consistency": 1.50e-6,
    "alpha_predicted_independent": null,
    "alpha_predicted_note": "self-consistency value is from the same model that generates the simulation and is NOT load-bearing; independent prediction is open work tracked in A.7 §3",
    "sector_separation": {
      "passed": true,
      "pairwise_sigmas": [4.2, 6.1, 5.7]
    },
    "battery": {
      "H0_nothing": {
        "struck": true,
        "reason": "alpha > 6 sigma_alpha; no outlier seed",
        "per_seed_strikes": 8,
        "evidence": {"alpha_over_sem": 17.1, "alpha_max_min_over_2sem": 1.4}
      },
      "H1_material": {
        "struck": true,
        "reason": "slope geometric, not mu_proxy-scaling",
        "per_seed_strikes": 7,
        "evidence": {"d_alpha_d_mu_proxy_rel": 0.034}
      },
      "H2_thermal":    {"struck": "n/a", "reason": "simulation has no thermal DOF"},
      "H3_em_pickup":  {"struck": "n/a", "reason": "simulation has no EM DOF"},
      "H4_mechanical": {"struck": "n/a", "reason": "rigid substrate, no mount DOF"},
      "H5_amplitude": {
        "struck": true,
        "reason": "F-test passes; linear regime confirmed",
        "per_seed_strikes": 8,
        "evidence": {"F_stat": 2.4, "F_crit": 18.5}
      },
      "H6_resonance": {
        "struck": true,
        "reason": "broadband consistent, chi2/dof = 1.27 (dof=21)",
        "per_seed_strikes": 6,
        "evidence": {"chi2_per_dof": 1.27, "dof": 21}
      },
      "H7_statistics": {
        "struck": true,
        "reason": "block-SEM plateau at K=8, ratio 2.4%; seed std 1.7x blocked SEM",
        "per_seed_strikes": 8,
        "evidence": {"sem_at_plateau_rel": 0.024, "seed_std_over_block_sem": 1.7, "plateau_block_size": 8}
      },
      "H8_q_drift": {
        "struck": true,
        "reason": "Q_surrogate held within 1.2%",
        "per_seed_strikes": 8,
        "evidence": {"q_std_over_mean": 0.012}
      },
      "H9_tau_model": {
        "struck": true,
        "reason": "alpha stable under alternative tau_Q form",
        "per_seed_strikes": 6,
        "evidence": {"alpha_alt": 1.39e-6, "alpha_diff_rel": 0.021}
      }
    },
    "completion_count": 7,
    "applicable_count": 7,
    "completion_invariant_simulation": "7/7 simulatable nulls struck",
    "hardware_only_nulls": ["H2_thermal", "H3_em_pickup", "H4_mechanical"],
    "ergodicity_caveat": "Battery operates within the microcanonical energy shell. Per SECTION5_CLOSURE_RECEIPT, the buckyball substrate exhibits a ~16% irreducible shell-vs-ensemble gap. PASS does NOT imply thermodynamic equilibrium.",
    "sudoku_verdict": "PASS_SIMULATION_ONLY",
    "interpretation": "All simulatable nulls struck with per-seed independence. The H2/H3/H4 rows are predictions to be checked on hardware: the simulation predicts that thermal, EM, and mechanical pickup are NOT the explanation."
  }
}
```

`sudoku_verdict` is one of:
- `PASS_SIMULATION_ONLY`: every simulatable null struck (per-seed ≥5/8); signal present at >6σ
- `FAIL_NULL_SURVIVES`: at least one simulatable null was not struck
- `FAIL_SIGNAL_MISSING`: H₀ fails (no signal above noise)
- `FAIL_PREDICTION_INCONSISTENT`: H₉ fails (model is overfitted; alternative τ_Q form gives different α)
- `FAIL_SECTOR_SEPARATION`: Q labels do not actually correspond to distinct gauge sectors

## 7. The CLI surface

```
  --battery               run Section 11 (default: on)
  --battery-omega-grid    comma-separated (default: derived from omega_c)
  --battery-q-grid        default 0,1,2
  --battery-seeds         default 20260616..20260623
  --battery-amplitudes    H5 amplitude grid (default 0.1,0.3,1.0,3.0)
  --battery-n-cores       multiprocessing pool size (default: 8 or cpu_count)
  --battery-fast          quick smoke: 1 seed, 3 omegas, 2 Q values, no H9 (~1 min)
```

## 8. The paper hook (Appendix A.7)

Sibling to A.5 (closure receipt) and A.6 (thread experiment / transfer function). A.7 wraps the battery output:

1. **The grid table** — H₀…H₉ mechanical render from the JSON.
2. **The predicted-slope extraction** — α_measured ± SEM; the self-consistency check vs. the model-generated α_predicted, with explicit note that this is not an independent falsifier and the independent prediction is open.
3. **The hardware-only rows as predictions** — H₂/H₃/H₄ flagged with the simulation's prediction-of-absence.
4. **The ergodicity caveat** — explicit.
5. **The receipt** — SHA-256 of the battery JSON; reproducibility command.

Estimated 3–4 pages.

## 9. Implementation order

1. **This SPEC** (you are reading v2) — design contract.
2. `inertia_damping/test_mass_dynamics.py` — NEW module with state `(U, E, x, v)`, coupled symplectic leapfrog, τ_Q model, μ_eff computation, mode-shape cached at startup, time-dependent drive hook. Not an extension of Section 5 kernel.
3. `inertia_damping/falsification_battery.py` — protocols H₀..H₉, drive shape, lock-in demod, gate functions, multiprocessing orchestrator, sector-separation pre-check.
4. Unit tests in `inertia_damping/test_test_mass_dynamics.py` and `test_falsification_battery.py` — recover Newtonian limit at Q=0; recover analytic χ(ω) for an uncoupled oscillator; verify gates strike in synthetic-Halcyon-true case.
5. Integration into `validation_report.py` Section 11 emission + markdown render.
6. Integration into `run_validation_report.py` with the `--battery` CLI surface above.
7. `--battery-fast` smoke test passes.
8. Full battery run on the buckyball at β=2.5 emits a complete grid.
9. Solves Vol. 4 Appendix A.7 lands with the receipt.

## 10. What this does NOT do

- This SPEC does not propose a hardware experiment.
- The placeholder `τ_Q = τ_0 / (1 + β_τ s_Q)` form is justified by analogy, not first-principles derivation. A better functional form (with a closed-form coupling constant derived from the Davis Field Equations) is open work tracked in A.7 §3 and is the source of the *independent* `α_predicted` value the simulation cannot supply.
- A `PASS_SIMULATION_ONLY` verdict is a **necessary** condition for the hardware claim; it is not sufficient.
- Cross-scale invariance from the 60-vertex buckyball to a larger lattice is not certified.

## 11. Definitions of done

- [x] SPEC v2 (this document) — design contract closing all v1 review blockers
- [ ] `test_mass_dynamics.py` shipped with unit tests
- [ ] `falsification_battery.py` shipped with unit tests
- [ ] `run_validation_report.py --battery` produces a JSON sidecar matching §6 schema
- [ ] `--battery-fast` smoke test passes in <2 min
- [ ] Full battery run emits a complete grid
- [ ] Solves Vol. 4 Appendix A.7 lands with the receipt
- [ ] Halcyon site links the existing "A thread, pulled two ways" section to the A.7 results
