# Halcyon Falsification Battery — SPEC v3.1.3 (wording + audit-tightness patch, deposit-ready)

**Status:** PRE-REGISTRATION, deposit-ready. Supersedes
`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.2.md` (commit
`f4cfa1444a72e94c67f5cc2b7bfee51aeaf4666a`) **before** that document
reached Zenodo deposit. v3.0, v3.1, v3.1.1, and v3.1.2 stay preserved
as the four drafts the multi-round review caught. v3.1.3 is the
document that goes to Zenodo.

**Date written:** 2026-06-21
**Implementation status at time of writing:** none. No v3 code exists yet.
**Predecessors retained as first-class artefacts:**
- `HALCYON_FALSIFICATION_BATTERY_SPEC.md` (v2.0 / v2.1)
- `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md` (v3.0, commit `0fe654d`)
- `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.md` (v3.1, commit `7121094`)
- `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.1.md` (v3.1.1, commit `1165d63`)
- `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.2.md` (v3.1.2, commit `f4cfa14`)

**Author commitment.** This is the fifth and final pre-deposit
iteration. v3.0 fixed the philosophical posture. v3.1 fixed two
mathematical defects. v3.1.1 fixed seven executability issues.
v3.1.2 fixed the validity-window blocker plus three smaller issues.
v3.1.3 fixes three wording and audit-tightness issues from the
fourth round of pre-deposit technical review. After this commit,
v3.1.3 is the contract; the Zenodo DOI fires on this commit hash;
§3's falsification criteria do not move.

## §0 — v3.1.2 → v3.1.3 changelog (the three round-4 patches)

Fourth-round pre-deposit technical review of v3.1.2 (ChatGPT,
2026-06-21) cleared the document as deposit-ready *with* three
surgical fixes. v3.1.3 lands them. The full v3.0 → v3.1 → v3.1.1
→ v3.1.2 architecture (connection 1-form on Λ, antisymmetric H_geom
primary observable, five-sham control set with S₄ absorbed,
six-contract verb acceptance battery, named stopping rule with
calibration-local NULLs, β_W ∈ [2.5, 3.0] inside the validated
SU(2) regime, three-tier publication commitment) all carry forward
unchanged.

### §0.1 The three patches

| # | v3.1.2 issue | v3.1.3 patch | Lives in |
|---|---|---|---|
| 1 | The SPEC repeatedly called the GPT review rounds "external review." For a public Zenodo deposit, that phrase risks being misread as outside human peer review (which has not yet occurred). | The act-language for the four GPT rounds is now **"pre-deposit technical review"**. The person-language for the human stopping-rule committee (the lattice-gauge-theory peer reviewer named in §8.5 and the journal peer reviewer) stays as "external reviewer / external review" because those references are about future human review that has not yet happened. The two senses are now distinguished consistently. | §0 (this changelog), §11 acknowledgments, §1.5 review-history sidebar |
| 2 | `N_DISCRETIZATION = 10000` is the v3.1.3 science value, but GC₅ tests convergence at `N ∈ {1000, 2000, 4000, 8000, 16000}`. The relationship between the gate and the science call was implicit. | §7.4 GC₅ now explicitly says the science value `N = 10000` lies inside the GC₅ convergence bracket and is **accepted only if the 8000→16000 relative change in H is < 1%**; otherwise the substrate blocks v3.1.3 science calls until the bracket is widened or the substrate is patched. | §7.4 GC₅ |
| 3 | §4.2 stated `τ_pin ≈ 1 at λ_pin = 1.0` as a numerical fact in prose. This either needs an empirical citation or it needs to become a gate the substrate is responsible for verifying. | §4.2 softens the prose to a **nominal design target** (`τ_pin ~ 1`); the substrate's `ADIABATICITY_CHECK` is required to verify `τ_pin << T_segment` at runtime. A violation forces AMBIGUOUS regardless of the H values. The claim is now the gate's responsibility, not prose. | §4.2 |

### §0.2 What did NOT change

- The connection-1-form definition of holonomy (§2.3).
- The antisymmetric primary observable `H_geom` (§3.1).
- The five-sham set after S₄ absorption (§3.2 / §5).
- The two-layer audit story (§7).
- The six-contract verb acceptance battery `GC₁`–`GC₆` (§7.4 — its
  GC₅ row gets the science-value gate added, but the test itself
  is unchanged).
- The named stopping rule with calibration-local NULLs (§3.3).
- The active-pinning timing regime with tracking-error gates (§4.3).
- The publication commitment of §8.
- The numerical values for Q range, β_W range `[2.5, 3.0]`, T_loop,
  T_segment, ω_drive, N_DISCRETIZATION, ramp rates, pin lambdas,
  ε tolerances, ε_abs. *Only the prose around `τ_pin` softens; no
  numbers change.*

---

## §1 — Why v2's measurement was wrong for the dynamics being measured

(Self-contained restatement, retained verbatim from v3.1.2.)

### 1.1 What v2 established (load-bearing into v3.1.3)

- **H₁ (material independence) strikes**: `|dα/dμ_proxy|/|α| =
  0.005–0.028` across three independent runs. The Halcyon coupling
  is geometric, not material-dependent.
- **H₅ (drive-amplitude linearity) strikes** at machine precision.
- **H₉ (τ_Q model robustness) strikes**: `|α_alt − α|/|α| = 0.025–0.13`.
- **Newtonian limit at Q=0 is exact**: `μ_eff(trivial vacuum) = 0`
  by construction.
- **Smoke-mode internal signal real**: `|α|/σ = 15.6` at the model's
  own internal `μ_eff(U(t))` state.

These five findings are evidence that the framework is doing
geometric physics rather than material physics. They do not depend
on whether v2's external observability succeeded.

### 1.2 What v2 did not establish

v2 did not establish `∂_Q μ_Q ≠ 0` in observation space. At
`α_Halcyon ∈ {1, 1000}` the observation-space extractor returned
`FAIL_SIGNAL_MISSING` in both runs.

### 1.3 The load-bearing diagnostic that forces redesign

Bumping `α_Halcyon` by 1000× moved observation-space SNR by only
2.5×. When noise scales with signal, the measurement is reading the
wrong observable. The fixed-Q lock-in is the adiabatic limit of a
measurement the apparatus does diabatically.

### 1.4–1.6 v2 sidecars preserved

`battery_fast_20260620_104846.json`,
`battery_full_20260620_181227.json`,
`battery_calibrated_20260621_011304.json` are preserved with their
SHA-256s. v2.1 SPEC remains authoritative for its own design.

---

## §2 — The framework's native observable is holonomy

### 2.1 Three-sentence anchor

> The framework's native object is holonomy.
> The apparatus measures holonomy.
> The simulation should compute holonomy.

### 2.2 The bundle and the control manifold

The test mass is a section of a bundle whose base is a
multi-dimensional programmed control manifold Λ and whose fiber is
the test mass's configuration space. Λ is at least two-dimensional.
In v3.1.3:

$$
\Lambda \;=\; \bigl\{(Q,\,\beta_W) \;:\; Q \in [0, 2],\, \beta_W \in [2.5, 3.0]\bigr\}
$$

where:

- **Q** is the surrogate sector coordinate. Q is a programmed-control
  coordinate, not a smooth topological label.
- **β_W** is the Wilson gauge-action coupling appearing in
  `S_W = (β_W / N) Σ_f [N − Re Tr U_f]`. Range `[2.5, 3.0]` keeps
  the loop strictly inside the SU(2) Q-observable regime prior
  validation work has trusted. β = 2.5 is at the lower endpoint
  (Migdal-Witten canonical operating point); β = 3.0 is at the
  upper edge of the validated envelope.

The constraint `Δβ_W = 0.5` means the loop encloses area 1.0 in Λ
while staying inside configurations already covered by validation
receipts.

**Why β_W is the right second coordinate.** It is physically
meaningful (cage controls the junction biases that set the Wilson
coupling); it already exists in GIGI's verb grammar (no new
conceptual introduction); it genuinely couples to the test mass
through the substrate's existing `κ_Q τ_Q² |φ_n|²` integrand. A pure
gauge-rotation parameter would trivialize the holonomy; β_W avoids
that trap.

**The v3.1.3 commitment.** The second coordinate is locked to β_W,
range `[2.5, 3.0]`, at this commit. Run-time selection is prohibited.
Extension below β = 2.5 additionally requires an independent SU(2)
Q-tracking validation at the proposed lower endpoint *before* it
can be used; this requires a separate v3.1.x amendment committed
and pushed before execution.

### 2.3 The holonomy as a connection 1-form on Λ

The Halcyon coupling defines a connection 1-form on Λ:

$$
A \;=\; A_Q(\lambda)\,dQ \;+\; A_{\beta_W}(\lambda)\,d\beta_W
$$

The closed-loop holonomy is the ordered exponential of A along γ:

$$
U[\gamma] \;=\; \mathcal{P}\exp\left(\oint_\gamma A\right)
$$

For the test mass's abelianized response the holonomy reduces to

$$
\mathcal{H}[\gamma] \;=\; \oint_\gamma A_i\, d\lambda^i
\;=\; \int_\Sigma F \;+\; O(\text{area}^2)
$$

for a small loop bounding area Σ, where `F = dA + A ∧ A` is the
curvature. The Halcyon prediction is that **F on Λ is non-zero**;
this is the falsifiable observable.

### 2.4 The pulled-back connection on the worldline

The bundle the test mass lives on is the pull-back of the
gauge-field-coupling connection to the test mass's worldline times
Λ. As the cage drives (Q, β_W) along a programmed path, the
connection's components in Λ become observable through the test
mass's response.

### 2.5 GIGI's native verbs

The v3.1.3 substrate call is
`SAMPLE_TRANSPORT … ALONG_LOOP … CONTROL_MANIFOLD (Q, beta_wilson) …
ADIABATIC … COMPUTE HOLONOMY`.
See §4.4 below and the v3.1.3 Halcyon→Gigi letter. The Python
orchestrator is a thin wrapper that constructs the loop, calls the
verb, parses the result, and applies the §3 gates.

---

## §3 — Pre-registered falsification criteria

This section is written before §4. The protocol of §4 is designed
to satisfy the criteria of §3, not the other way around.

### 3.1 The primary observable and its three regimes

Primary observable:
$$
H_{\rm geom}[\gamma_{\rm unit}] \;=\; \tfrac{1}{2}\bigl(H[\gamma_{\rm unit}] - H[\gamma_{\rm unit}^{-1}]\bigr).
$$
Systematic-offset diagnostic:
$$
H_{\rm sys} \;=\; \tfrac{1}{2}\bigl(H[\gamma_{\rm unit}] + H[\gamma_{\rm unit}^{-1}]\bigr).
$$

| Outcome | Criterion |
|---|---|
| **POSITIVE** | `|H_geom_mean| > 5 σ_H` AND `|H_sys| < 1 σ_H` AND every sham passes its §3.2 threshold AND per-seed sign-coherence ≥ 5/8 on the primary loop |
| **NULL** | `|H_geom_mean| < 1 σ_H` AND `|H_sys| < 1 σ_H` AND every sham passes its §3.2 threshold (*no sign-coherence requirement on NULL — random signs are expected*) |
| **AMBIGUOUS** | Any of: `1σ ≤ |H_geom_mean| ≤ 5 σ`; `|H_sys| ≥ 1 σ_H`; any sham fails its threshold; any sham shows consistent-sign pattern across ≥ 6/8 seeds with `|mean| > 0.5 σ_sham` (the anti-fishing rule of §3.4); per-seed sign-coherence < 5/8 in the POSITIVE branch |

`σ_H` is the Flyvbjerg–Petersen blocked SEM of `H_geom_mean` across
the 8 seeds.

### 3.2 Required sham controls

| Sham | Gate |
|---|---|
| **S₁** flat field (κ_Q ≡ 0) | `|H_S₁_mean| < 2 σ_S₁` AND `|H_S₁_mean| < ε_abs` |
| **S₂** α_Halcyon = 0 | `|H_S₂_mean| < ε_abs` (load-bearing); 2σ check is sanity |
| **S₃** μ_baseline scaled (×0.1, ×1, ×10) | **POSITIVE branch:** baseline-subtracted H invariant within 10% of `H_geom_mean`. **NULL / AMBIGUOUS branches:** `|H_S₃,raw at μ_baseline=1| < 2 σ_S₃` AND `< ε_abs`. Per-scaling H values reported as diagnostics. |
| **S₄** absorbed into H_geom | (no separate flag) |
| **S₅** degenerate loop (zero area in Λ) | `|H_S₅_mean| < 2 σ_S₅` AND `|H_S₅_mean| < ε_abs` |
| **S₆** frozen field | `|H_S₆_mean| < 2 σ_S₆` AND `|H_S₆_mean| < ε_abs` |

**`ε_abs = 1 × 10⁻¹⁰`** is the pre-registered empirical-numerical
floor. *Rationale:* this value is the floor below which the
substrate's own GC₂ (Abelian area-law to 1%) and GC₆ (gauge
invariance to machine ε) acceptance tests, run before v3.1.3 calls
the verb for science, empirically demonstrate the verb is
numerically clean on its own controls. This is not a worst-case
`N · ε_machine` bound; it is the operating threshold the GC tests
confirm. If the substrate's GC tests show the floor sits higher,
v3.1.3 still treats 10⁻¹⁰ as the absolute gate and blocks until the
substrate cleans up. If the floor sits lower, 10⁻¹⁰ remains the
pre-registered gate (pre-registration commits to the gate, not to
the tightest possible empirical floor).

### 3.3 Stopping rule

The framework is declared *falsified by simulation* if all four of
the following are met:

1. v3.1.3 returns NULL at the preregistered calibration set
   (α_Halcyon ∈ {1, 1000, and the eventual Field-Equations
   derivation if it lands before run time}), with sham controls
   passing.
2. A second independent v3-class measurement design — to be
   specified in a hypothetical SPEC v4 with its own
   pre-registration — also returns NULL on the same substrate at
   the same calibration.
3. The two measurement designs are not trivially equivalent
   (independence verified by an external reviewer per §8.5; this
   is human peer review, not a pre-deposit technical review pass).
4. No subsequent re-scaling of α invalidates the prior NULL unless
   the observable's scaling law in α was itself preregistered.

The framework remains *not yet falsified* if v3.1.3 returns
POSITIVE, or if v3.1.3 returns AMBIGUOUS and the §3.7 re-run
criteria are met.

### 3.4 The consistent-sign anti-fishing rule (on shams)

For each sham, compute the per-seed `H_sham,i` for i = 1…8. The
sham *fails* (regardless of the |mean| gate) if `sign(H_sham,i)` is
the same for at least 6 of 8 seeds AND `|mean H_sham| > 0.5 σ_sham`.

### 3.5 Per-seed sign-coherence (POSITIVE only)

For **POSITIVE classification**, at least 5 of 8 primary-loop seeds
must share the sign of `H_geom_mean`.

For **NULL classification**, no sign-coherence requirement is
imposed. A true null has random signs across seeds — that is the
expected pattern, not a defect.

For **AMBIGUOUS classification**, sign-coherence is a contributing
factor: if `|H_geom_mean|` is in the 1σ–5σ range AND fewer than 5/8
seeds share a sign, the verdict is AMBIGUOUS specifically because
the ensemble is incoherent.

The per-seed sign-coherence rule does not require per-seed σ_H,i.
v3.1.3 does not require within-seed loop repeats.

### 3.6 Calibration commitments

The v3.1.3 simulation will run at:
- `α_Halcyon = 1`
- `α_Halcyon = 1000`
- `α_Halcyon = TBD from the Davis Field Equations` (if derived
  before execution; else the simulation runs at α=1 and α=1000 only).

### 3.7 Ambiguity-resolution re-run criteria

If v3.1.3 returns AMBIGUOUS, a re-run is allowed only under all of
the following:

- The specific ambiguity (which gate failed and by how much) is
  named in the re-run's amendment.
- The re-run uses *more seeds or longer T_segment*, not different
  ω, F₀, T_loop, α, or β_W range.
- The amendment is committed and pushed before the re-run.
- The re-run's verdict is published per §8 regardless of direction.

---

## §4 — Q/β_W-ramp protocol via GIGI's HOLONOMY and TRANSPORT verbs

### 4.1 The loop γ_unit on Λ

$$
\gamma_{\rm unit}:\quad
(Q=0,\, \beta_W=2.5)
\to (Q=2,\, \beta_W=2.5)
\to (Q=2,\, \beta_W=3.0)
\to (Q=0,\, \beta_W=3.0)
\to (Q=0,\, \beta_W=2.5)
$$

Closed rectangle on (Q, β_W) inside the validated SU(2) operating
window. Enclosed area `= Q_max · Δβ_W = 2 × 0.5 = 1` (mixed control
units). `T_loop = 200`, `T_segment = T_loop / 4 = 50`. Ramp rates
in §4.4.

### 4.2 The adiabaticity regime (active pinning; substrate-gated τ_pin)

Two distinct regimes exist in principle. v3.1.3 explicitly declares
which one it uses.

**Regime A — passive adiabaticity (NOT used by v3.1.3):**
$$
T_{\rm drive} \;\ll\; T_{\rm segment}, \qquad
\tau_{\rm local\ eq} \;\ll\; T_{\rm segment} \;\ll\; \tau_{\rm unpinned\ drift}.
$$
At v2's measured values (`τ_unpinned_drift ≈ 10` time units), the
right inequality fails for any `T_segment > 10`. v3.1.3 does NOT
attempt passive adiabaticity.

**Regime B — active-pinning adiabaticity (the v3.1.3 regime):**
$$
T_{\rm drive} \;\ll\; T_{\rm segment}, \qquad
\tau_{\rm pin} \;\ll\; T_{\rm segment},
$$
and the tracking-error gates of §4.3.

At v3.1.3's values: `T_drive = 2π ≈ 6.28`, `T_segment = 50`, so
`T_drive / T_segment ≈ 0.126` (genuinely `<<`; ~8 drive cycles per
segment).

For the second inequality, `τ_pin` is the pinning equilibration
time at the v3.1.3 pre-registered `λ_pin = 1.0`. The **nominal
design target is `τ_pin ~ 1`**, giving `τ_pin / T_segment ~ 0.02`
(genuinely `<<`). Whether the substrate actually realizes this
nominal target at run time is the substrate's responsibility: the
`ADIABATICITY_CHECK` return (per the v3.1.3 Halcyon→Gigi letter)
**must verify `τ_pin << T_segment`** with a quantitative
substrate-side measurement. If `ADIABATICITY_CHECK` reports
`τ_pin / T_segment ≥ 0.1` (one order of magnitude tighter than `<<`),
the verdict is forced AMBIGUOUS regardless of the H values, on the
same grounds as a tracking-error violation. The prose target is
documentation; the substrate gate is the load-bearing claim.

### 4.3 The tracking-error gates

Pre-registered tolerances: `ε_Q = 0.05`, `ε_{β_W} = 0.05`.

$$
\max_t |Q_{\rm surr}(t) - Q_{\rm target}(t)| \;<\; \epsilon_Q,
\qquad
\max_t |\beta_{W,{\rm surr}}(t) - \beta_{W,{\rm target}}(t)| \;<\; \epsilon_{\beta_W}.
$$

Tracking-error violation forces AMBIGUOUS regardless of the H values.

### 4.4 The GIGI call

```
SAMPLE_TRANSPORT halcyon_canonical_buckyball
  ALONG_LOOP gamma_unit_in_Q_beta_W
  CONTROL_MANIFOLD (Q, beta_wilson)
  ADIABATIC TRUE
  RAMP_RATE_Q 0.04            // (2.0 - 0.0) / T_segment = 2 / 50
  RAMP_RATE_BETA_W 0.01        // (3.0 - 2.5) / T_segment = 0.5 / 50
  DRIVE_OMEGA 1.0
  DRIVE_F0 0.01
  N_DISCRETIZATION 10000       // dt = 0.02 over T_loop = 200
                               // ~31.8 drive cycles per loop, ~314 substeps per drive cycle
                               // ~8 drive cycles per T_segment (regime B satisfied)
                               // N=10000 is inside the GC₅ convergence bracket; gated by §7.4
  PIN_LAMBDA_Q 1.0
  PIN_LAMBDA_BETA_W 1.0
  EPS_Q 0.05
  EPS_BETA_W 0.05
  ALPHA_HALCYON 1.0            // also runs at 1000
  TAU_0 1.0  BETA_TAU 2.0       // tau_Q model parameters (distinct from beta_wilson)
  MU_BASELINE 1.0  K_SPRING 1.0  C_DAMP 0.1
  SEEDS [20260616..20260623]
  COMPUTE HOLONOMY_FORWARD
  COMPUTE HOLONOMY_REVERSED
  COMPUTE TRACKING_ERROR_TRACE_Q
  COMPUTE TRACKING_ERROR_TRACE_BETA_W
  COMPUTE ADIABATICITY_CHECK   // returns tau_pin/T_segment per §4.2; AMBIGUOUS if >= 0.1
  RETURN H_forward, H_reversed, sigma_H_blocked,
         per_seed_H_forward, per_seed_H_reversed,
         tracking_error_max_Q, tracking_error_max_beta_W,
         adiabaticity_check
```

The Python orchestrator computes:
```python
H_geom = 0.5 * (H_forward - H_reversed)
H_sys  = 0.5 * (H_forward + H_reversed)
```
and applies the §3 gates.

**Disambiguation:** `β_W` is the **Wilson gauge coupling** appearing
in `S_W = (β_W / N) Σ_f [N − Re Tr U_f]`. `BETA_TAU` is the v2.1
τ_Q model coefficient appearing in
`τ_Q(e) = τ₀ / (1 + β_τ s_Q(e))`. They are different parameters and
must not be confused.

### 4.5 The integration (substrate-side)

The closed-loop holonomy of the connection A on Λ is the
substrate's responsibility. The Python orchestrator does not
re-implement it. The substrate's HOLONOMY verb computes
`U[γ] = P exp ∮ A` via discretized parallel transport along γ. The
substrate-side correctness contract is the v3.1.3 verb acceptance
battery of §7.4.

### 4.6 The orchestrator surface (specification only)

```python
def run_holonomy_battery(alpha_halcyon, seeds, log_path):
    """SPEC v3.1.3 §3 + §4 + §5; pure delegation to GIGI."""
    forward = gigi_sample_transport(loop=GAMMA_UNIT_FORWARD,
                                     alpha_halcyon=alpha_halcyon,
                                     seeds=seeds, ...)
    reversed_ = gigi_sample_transport(loop=GAMMA_UNIT_REVERSED, ...)
    H_geom = 0.5 * (forward.H - reversed_.H)
    H_sys  = 0.5 * (forward.H + reversed_.H)
    shams = {name: gigi_sample_transport(loop=..., sham_flag=name, ...)
             for name in SHAM_FLAGS}
    return apply_v3_1_3_gates(H_geom, H_sys, shams,
                              forward.tracking_error_Q,
                              forward.tracking_error_beta_W,
                              forward.adiabaticity_check)
```

No leapfrog. No demodulation. No force computation. All substrate.

---

## §5 — Sham controls (full specifications)

| Sham | Implementation flag (verb-side) | Gate |
|---|---|---|
| S₁ flat field | `SHAM_FLAT_FIELD = true` (κ_Q ≡ 0 on all edges, all times) | `|H_S₁| < 2σ_S₁` AND `|H_S₁| < 10⁻¹⁰` |
| S₂ α=0 | `ALPHA_HALCYON = 0` | `|H_S₂| < 10⁻¹⁰` (load-bearing); 2σ check is sanity |
| S₃ mass scaled | `MU_BASELINE ∈ {0.1, 1.0, 10.0}`; substrate fits baseline-subtracted H | **POSITIVE branch:** baseline-subtracted H invariant within 10%. **NULL/AMBIGUOUS branches:** `|H_S₃ at μ_baseline=1| < 2σ_S₃` AND `< 10⁻¹⁰`. |
| S₄ absorbed into H_geom | (no separate flag) | folded into §3.1 antisymmetric primary observable |
| S₅ degenerate loop | `LOOP gamma_degenerate` (zero area in Λ) | `|H_S₅| < 2σ_S₅` AND `|H_S₅| < 10⁻¹⁰` |
| S₆ frozen field | `SHAM_FROZEN_FIELD = true` | `|H_S₆| < 2σ_S₆` AND `|H_S₆| < 10⁻¹⁰` |

---

## §6 — What v3.1.3's results mean for v2

### 6.1 POSITIVE v3.1.3 + NULL v2

v2's null is **consistent with the adiabatic/fixed-point limit** of
the v3.1.3 holonomy model. This is not a "confirmed prediction" of
v2's null — v2 was not pre-registered against the v3.1.3 model. It
is a consistency check.

### 6.2 NULL v3.1.3 + NULL v2

Stopping rule per §3.3 triggers if the conditions are met
(independent v4 measurement design also NULL, peer review by the
named external committee). v2 is the adiabatic-limit complementary
control; the joint null is stronger than either alone.

### 6.3 NULL v3.1.3 + (smoke-mode internal pass on v2)

The v2 internal-extractor signal (`|α|/σ = 15.6`) is not
contradicted by a v3.1.3 null. v2's internal extractor confirmed
model mechanics; v3.1.3 (and v2's external extractor) tests
observability.

### 6.4 POSITIVE v3.1.3 + (POSITIVE v2 internal)

Strongest positive case: model's internal state and observable
holonomy both match prediction.

### 6.5 AMBIGUOUS v3.1.3

Per §3.7, re-run is allowed only with named ambiguity, more seeds
or longer T_segment, no protocol parameter changes, amendment
committed before re-run.

---

## §7 — GIGI audit surface

### 7.1 Two-layer independent auditability

- **Layer 1 — substrate computation.** `SAMPLE_TRANSPORT`,
  `HOLONOMY`, `TRANSPORT`, `SPECTRAL` are GIGI verbs. Their
  correctness is the domain of GIGI's test suite. A reviewer who
  wants to verify the substrate math runs the GIGI tests
  (`cargo test --features halcyon`) and inspects the WAL receipts.
- **Layer 2 — protocol design.** The choice of loop γ_unit, the
  adiabaticity condition, the sham controls, the gate thresholds,
  and the stopping rule are the domain of this SPEC. A reviewer
  who wants to verify the experimental design reads this document.

Neither layer needs to inspect the other's internals to perform
its review. The interface is the GIGI verb signature.

### 7.2 The receipt model

v3.1.3 emits a `section_12_holonomy_battery_v3_1_3` JSON sidecar
with:

- The SHA-256 of this SPEC at execution time
- The GIGI deploy hash used for the substrate calls
- Per-seed `H_forward` and `H_reversed` and the sham results
- Per-gate verdict and the σ_H values
- Tracking-error traces per axis
- The `adiabaticity_check` result (numerical τ_pin/T_segment ratio)
- The overall verdict (POSITIVE / NULL / AMBIGUOUS)

Reproducibility: anyone with this SPEC's SHA-256, the GIGI deploy
hash, and the seed list can re-run the experiment and verify the
sidecar.

### 7.3 What this changes for the auditor's job

In v2, the auditor had to read ~1150 lines of Python to verify the
measurement. In v3.1.3, the auditor reads this SPEC and trusts the
substrate's correctness audit (amortized across every chapter
using the substrate). The audit surface is smaller and more
orthogonal.

### 7.4 GIGI verb acceptance battery (with science-value gate on GC₅)

Before the v3.1.3 simulation calls `SAMPLE_TRANSPORT ALONG_LOOP`
for science, the substrate-side verb must pass the following six
contracts. These are testable by GIGI's `cargo test --features
halcyon` suite at the verb's introduction, before any
Halcyon-side integration:

| # | Contract | Test |
|---|---|---|
| **GC₁** | **Flat connection returns zero.** | Construct a known-flat connection (`A ≡ 0` in synthetic mode); verify `H[any loop] = 0` to machine ε across at least 4 loop shapes. |
| **GC₂** | **Known area law for an Abelian constant-curvature connection.** | Construct a connection with constant curvature `F₀` in `(Q, β_W)`; verify `H[γ] = F₀ · Area(γ)` to 1% across 3 loop sizes. |
| **GC₃** | **Reversed loop inverts/sign-flips.** | For an arbitrary connection, verify `H[γ⁻¹] = −H[γ]` (Abelian) or `H[γ]⁻¹` (non-Abelian) to 1% across at least 3 connections. |
| **GC₄** | **Zero-size loop returns zero.** | Construct a degenerate loop bounding zero area; verify `H = 0` to machine ε. |
| **GC₅** | **Discretization convergence + science-value gate (v3.1.3 patch).** | Compute H at `N_discretization ∈ {1000, 2000, 4000, 8000, 16000}`; verify monotone convergence with relative change `< 1%` between 8000 and 16000 substeps. **The v3.1.3 science call uses `N = 10000`, which lies inside this convergence bracket.** Science calls are *accepted only if* the 8000→16000 relative change is `< 1%`; otherwise science calls are blocked at the substrate side until the bracket is widened, the verb is patched, or a v3.1.x amendment moves N. |
| **GC₆** | **Gauge invariance.** | Apply a known gauge transformation to the substrate's connection; verify H is invariant to machine ε. |

The substrate must pass GC₁–GC₆ before v3.1.3 science calls are
made. If any contract fails (including GC₅'s science-value gate),
v3.1.3 is gated on the substrate-side patch. GC₁–GC₆ are the
**new** contracts the new verb introduces; the existing
1373-assertion GIGI test suite is necessary but not sufficient.

---

## §8 — Publication commitment

### 8.1 The commitment

This document is committed to the public record at:
- **GitHub:** `nurdymuny/davis-wilson-map`, branch
  `feat/halcyon-gigi-substrate`, file
  `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.3.md`.
- **Zenodo:** a DOI will be minted from this SPEC's commit hash
  before the v3.1.3 protocol runs. The git commit hash is the
  implementation-level pre-registration timestamp; the Zenodo DOI
  is the publication-level pre-registration.

### 8.2 What gets published when v3.1.3 completes

All of the following are published, regardless of verdict:
- The `section_12_holonomy_battery_v3_1_3` sidecar JSON.
- A revision of Solves Vol. 4 with Appendix A.8 reporting the
  v3.1.3 result (POSITIVE, NULL, or AMBIGUOUS).
- The log file from the run, with per-seed verdicts.
- The GIGI deploy hash and WAL receipt for the substrate calls.

There is no version in which the result is suppressed if
unfavorable.

### 8.3 If v3.1.3 is NULL

Per §3.3 and §6.2, NULL is publishable. The chapter v5 reports the
null, names v2's null as the complementary control, and either (a)
declares the stopping condition per §3.3 (if a second independent
measurement design also returns null, under the external committee
peer review of §8.5), or (b) names the conditions for a v4
measurement design under external committee review. The null is
*not* re-run in a v3.1.3a without external committee peer review.

### 8.4 If v3.1.3 is POSITIVE

The chapter v5 reports the positive, names v2's null as the
adiabatic-limit complementary case, and proceeds to design the
apparatus-side measurement that would replicate the holonomy on
hardware. Simulation positive is necessary for the apparatus claim
but not sufficient.

### 8.5 The external review committee (humans, not GPT)

The stopping-rule committee of §3.3 — required for declaring the
framework falsified — is to be assembled at the time the second
NULL is recorded, not in advance. The committee consists of:

- (a) Gigi (project PI),
- (b) one external lattice-gauge-theory peer reviewer chosen by
  Gigi from outside the program,
- (c) one peer reviewer from a journal submission process.

The committee's role is to verify that the two measurement designs
(v3.1.3 and the hypothetical v4) are not trivially equivalent.
This is *human* peer review and is distinct from the four rounds
of *pre-deposit technical review* (§11) that produced this SPEC
before deposit.

---

## §9 — What this SPEC does not commit to

- No specific `α_Halcyon` from Davis Field Equations (open work).
- No commitment to the v3.1.3 result.
- No hardware apparatus design.
- v2 not deprecated.
- No promise that v3.1.3 will succeed in detecting the predicted
  holonomy.
- No extension of the β_W range below 2.5 without a v3.1.x
  amendment AND an independent SU(2) Q-tracking validation at the
  proposed lower endpoint.

---

## §10 — Definitions of done

- [ ] This v3.1.3 SPEC committed to GitHub.
- [ ] v3.1.3 SPEC deposited on Zenodo (with v3.0, v3.1, v3.1.1,
  and v3.1.2 attached for chain-of-custody transparency).
- [ ] GIGI verb (per the v3.1.3 Halcyon→Gigi letter) lands with
  `CONTROL_MANIFOLD (Q, beta_wilson)`, `RAMP_RATE_BETA_W` for the
  `[2.5, 3.0]` window, `PIN_LAMBDA_BETA_W`, `EPS_BETA_W`,
  `TRACKING_ERROR_TRACE_BETA_W`, the five-sham flag set, and the
  `ADIABATICITY_CHECK` that emits a numerical `τ_pin/T_segment`.
- [ ] GIGI verb passes the GC₁–GC₆ acceptance battery of §7.4
  *including* GC₅'s science-value gate at N=10000.
- [ ] Python orchestrator implemented as a thin delegation wrapper.
- [ ] v3.1.3 run produces sidecar matching §7.2.
- [ ] Verdict published per §8 regardless of direction.
- [ ] Solves Vol. 4 Appendix A.8 reports the v3.1.3 result.

---

## §11 — Acknowledgments

v3.0 → v3.1 was driven by the first round of *pre-deposit
technical review* (the two mathematical defects: scalar holonomy
vanishing by FTC, adiabaticity inequality reversed). v3.1 →
v3.1.1 was driven by the second round of pre-deposit technical
review (seven executability defects). v3.1.1 → v3.1.2 was driven
by the third round (the validity-window blocker — β_W traversal
outside the SU(2) operating regime — plus three smaller patches:
self-containedness, ε_abs rationale, NULL-branch sign-coherence).
v3.1.2 → v3.1.3 was driven by the fourth round (wording
distinction between pre-deposit technical review and human peer
review; science-value gate on GC₅; substrate-gated `τ_pin` claim).

All four rounds of pre-deposit technical review were load-bearing
for the protocol. Each version's §0 changelog names every patch
and which round surfaced it. **The pre-deposit technical review
rounds are model-assisted reviews of the SPEC's mathematical
content and protocol executability; they are not a substitute for
external human peer review.** External human peer review is
reserved for §8.5's stopping-rule committee and any journal
submission process and has not yet occurred.

Pre-registration that does not admit correction before deposit is
brittle; pre-registration that admits correction after deposit is
meaningless. v3.1.3 reflects five review iterations (Gigi's
methodological intervention, plus four rounds of pre-deposit
technical review) all completed before deposit. The Zenodo deposit
timestamp will fire on v3.1.3, and from that moment §3 cannot move.

The discipline that produced this — four preserved drafts plus the
canonical v3.1.3 — demonstrates pre-registration's intended
property: each review pass caught real issues that a one-pass
pre-registration would have locked in. The cost of admission to
either credibility, positive or negative, is paid in the patches
recorded above, not in the protocol that runs.
