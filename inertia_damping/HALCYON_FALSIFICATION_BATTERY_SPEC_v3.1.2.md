# Halcyon Falsification Battery — SPEC v3.1.2 (validity-window patch, deposit-ready)

**Status:** PRE-REGISTRATION, deposit-ready. Supersedes
`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.1.md` (commit
`1165d63dbaffe30b55438cb82c1fa80aaf1f9ce0`) **before** that document
reached Zenodo deposit. v3.0, v3.1, and v3.1.1 stay preserved as the
three drafts the multi-round review caught. v3.1.2 is the document
that goes to Zenodo.

**Date written:** 2026-06-21
**Implementation status at time of writing:** none. No v3 code exists yet.
**Predecessors retained as first-class artefacts:**
- `HALCYON_FALSIFICATION_BATTERY_SPEC.md` (v2.0 / v2.1)
- `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md` (v3.0, commit `0fe654d`)
- `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.md` (v3.1, commit `7121094`)
- `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.1.md` (v3.1.1, commit `1165d63`)

**Author commitment.** This is the fourth and final pre-deposit
iteration. v3.0 fixed the philosophical posture. v3.1 fixed two
mathematical defects. v3.1.1 fixed seven executability issues. v3.1.2
fixes the four remaining issues from the third review round (validity
window, self-containedness, ε_abs rationale, NULL-branch sign coherence).
After this commit, the v3.1.2 document is the contract; the Zenodo DOI
fires on this commit hash; §3's falsification criteria do not move.

## §0 — v3.1.1 → v3.1.2 changelog (the four validity / executability patches)

Third-round external review of v3.1.1 (ChatGPT, 2026-06-21) caught
four issues between v3.1.1's GitHub commit and its planned Zenodo
deposit. Each patch is locally surgical; none requires a structural
redesign. The full v3.0 → v3.1 → v3.1.1 architecture (connection
1-form on Λ, antisymmetric H_geom primary observable, six-sham control
set with S₄ absorbed, six-contract verb acceptance battery, named
stopping rule with calibration-local NULLs, three-tier publication
commitment) all carry forward unchanged.

### §0.1 The four patches

| # | v3.1.1 issue | v3.1.2 patch | Lives in |
|---|---|---|---|
| 1 | **The blocker.** Loop traversed `β_W ∈ [2.0, 3.0]`, but the validated SU(2) Q-observable regime per the JOURNAL is `β ≥ 2.5`. β=2.3 marginally failed validation; β=2.0 is well outside the trusted operating window. | `β_W` range tightened to `[2.5, 3.0]`. Loop now starts and ends at `(Q=0, β_W=2.5)`. Area halves from 2 to 1, but the loop stays inside the regime the program has validated. | §2.2, §4.1, §4.4 |
| 2 | v3.1.1 had several "unchanged from v3.1" cross-references back to commit `7121094`. For the canonical deposit document, this asks the reader to read three files. | v3.1.2 is fully self-contained. All sections include their authoritative text. The only cross-version reference is the §0 changelog. | throughout |
| 3 | `ε_abs = 1×10⁻¹⁰` rationale was internally inconsistent: §0 changelog said "10⁹ FLOPs", §3.2 said "10⁵–10⁶ operations per loop", and the worst-case `N · ε_machine` bound at 10⁹ FLOPs is ~10⁻⁷ not 10⁻¹⁰. | The ε_abs value is kept at 1×10⁻¹⁰ as the pre-registered empirical floor, with the rationale rephrased as "empirically validated by GC₂ + GC₆ acceptance tests; not a worst-case `Nε` bound." | §3.2, §5 |
| 4 | Per-seed sign-coherence rule penalized NULL classification for lacking sign coherence, but random sign distribution is *expected* in a true NULL. | Sign-coherence applies only to POSITIVE classification. NULL is achieved by global `|H_geom_mean| / σ_H < 1` regardless of seed sign distribution. The anti-fishing rule on shams is unchanged. | §3.5 |

### §0.2 What did NOT change

- The connection-1-form definition of holonomy (§2.3).
- The antisymmetric primary observable `H_geom` (§3.1).
- The five-sham set after S₄ absorption (§3.2 / §5).
- The two-layer audit story (§7).
- The six-contract verb acceptance battery `GC₁`–`GC₆` (§7.4 — now inlined).
- The named stopping rule with calibration-local NULLs (§3.3).
- The active-pinning timing regime with tracking-error gates (§4.2 / §4.3).
- The publication commitment of §8.
- The numerical values for T_loop, T_segment, ω_drive, N_DISCRETIZATION,
  pin lambdas, ε tolerances. *Only β_W's range changes; everything else
  in §4.4 stays put.*

---

## §1 — Why v2's measurement was wrong for the dynamics being measured

(Self-contained restatement of v3.1 §1, retained verbatim for v3.1.2.)

### 1.1 What v2 established (load-bearing into v3.1.2)

The v2 battery's positive findings are not consequences of the
measurement type and survive into v3.1.2 as established properties of
the underlying framework:

- **H₁ (material independence) strikes**: `|dα/dμ_proxy|/|α| =
  0.005–0.028` across three independent runs. The Halcyon coupling is
  geometric, not material-dependent.
- **H₅ (drive-amplitude linearity) strikes** at machine precision
  (`rel slope of χ vs F₀ < 0.001`).
- **H₉ (τ_Q model robustness) strikes**: `|α_alt − α|/|α| = 0.025–0.13`.
- **Newtonian limit at Q=0 is exact**: `μ_eff(trivial vacuum) = 0` by
  construction.
- **Smoke-mode internal signal real**: `|α|/σ = 15.6` at the model's
  own internal `μ_eff(U(t))` state.

These five findings are evidence that the framework is doing geometric
physics rather than material physics. They do not depend on whether
v2's external observability succeeded.

### 1.2 What v2 did not establish

v2 did not establish `∂_Q μ_Q ≠ 0` in observation space. At
`α_Halcyon ∈ {1, 1000}` the observation-space extractor returned
`FAIL_SIGNAL_MISSING` in both runs.

### 1.3 The load-bearing diagnostic that forces redesign

Bumping `α_Halcyon` by 1000× moved observation-space SNR by only 2.5×.
When noise scales with signal, the measurement is reading the wrong
observable. The fixed-Q lock-in is the adiabatic limit of a measurement
the apparatus does diabatically.

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

### 2.2 The bundle and the control manifold (patched: validated β_W range)

The test mass is a section of a bundle whose base is a multi-dimensional
programmed control manifold Λ and whose fiber is the test mass's
configuration space. Λ is at least two-dimensional. In v3.1.2:

$$
\Lambda \;=\; \bigl\{(Q,\,\beta_W) \;:\; Q \in [0, 2],\, \beta_W \in [2.5, 3.0]\bigr\}
$$

where:

- **Q** is the surrogate sector coordinate (the v2 Q-label, unchanged).
  Q is a programmed-control coordinate, not a smooth topological label.
- **β_W** is the Wilson gauge-action coupling appearing in
  `S_W = (β_W / N) Σ_f [N − Re Tr U_f]`. *Range patched in v3.1.2 to
  `[2.5, 3.0]`* to keep the loop strictly inside the SU(2) Q-observable
  regime the program's earlier validation work has trusted (see the
  JOURNAL: clean at β=2.5 and 2.7, marginal failure at β=2.3, untested
  at β=2.0). The Migdal–Witten canonical operating point β=2.5 is the
  lower endpoint of the range, not the midpoint. β=3.0 is at the upper
  edge of the operating envelope from v6.

The constraint `Δβ_W = 0.5` (half of v3.1.1's `Δβ_W = 1.0`) means the
loop encloses half the area in Λ that v3.1.1 prescribed, but it
prescribes only configurations already covered by validation receipts.
This is the right tradeoff at the pre-registration stage: a smaller
honest area beats a larger area that drives the substrate into
regions whose holonomy verb output is not independently trusted.

**Why β_W is the right second coordinate.** It is physically meaningful
(the cage controls the junction biases that set the Wilson coupling),
it already exists in GIGI's verb grammar (no new conceptual
introduction), and it genuinely couples to the test mass through the
substrate's existing `κ_Q τ_Q² |φ_n|²` integrand. A pure gauge-rotation
parameter would trivialize the holonomy; β_W avoids that trap.

**The v3.1.2 commitment.** The second coordinate is locked to β_W,
range `[2.5, 3.0]`, at this commit. If GIGI's eventual implementation
requires a different second coordinate or a range that extends below
β=2.5, that requires a **v3.1.3 amendment committed and pushed before
execution**. Run-time selection is prohibited. Extension below β=2.5
additionally requires an independent SU(2) Q-tracking validation at
the proposed lower endpoint *before* it can be used.

### 2.3 The holonomy as a connection 1-form on Λ

The Halcyon coupling defines a connection 1-form on Λ:

$$
A \;=\; A_Q(\lambda)\,dQ \;+\; A_{\beta_W}(\lambda)\,d\beta_W
$$

The closed-loop holonomy is the ordered exponential of A along γ:

$$
U[\gamma] \;=\; \mathcal{P}\exp\left(\oint_\gamma A\right)
$$

(non-Abelian, general case). For the test mass's abelianized response
the holonomy reduces to

$$
\mathcal{H}[\gamma] \;=\; \oint_\gamma A_i\, d\lambda^i
\;=\; \int_\Sigma F \;+\; O(\text{area}^2)
$$

for a small loop bounding area Σ, where `F = dA + A ∧ A` is the
curvature. The Halcyon prediction is that **F on Λ is non-zero**; this
is the falsifiable observable. A trivial (flat-A) control connection
gives H[γ] = 0 for every closed loop on Λ; the prediction is `F ≠ 0`,
with sign and magnitude determined by the substrate's specific
geometry.

A one-dimensional retraced path bounds zero area in Λ regardless of
the connection's curvature, so v3.0's `0 → 1 → 2 → 1 → 0` could not
have detected non-trivial F even if it existed. v3.1.2 fixes this by
requiring loops with non-zero enclosed area in (Q, β_W).

### 2.4 The pulled-back connection on the worldline

The bundle the test mass lives on is the pull-back of the
gauge-field-coupling connection to the test mass's worldline times Λ.
As the cage drives (Q, β_W) along a programmed path, the connection's
components in Λ become observable through the test mass's response.

### 2.5 GIGI's native verbs

The v3.1.2 substrate call is `SAMPLE_TRANSPORT … ALONG_LOOP …
CONTROL_MANIFOLD (Q, beta_wilson) … ADIABATIC … COMPUTE HOLONOMY`.
See §4.4 below and the v3.1.2 Halcyon→Gigi letter for the verb
signature. The simulation delegates the substrate-side math to GIGI;
the Python orchestrator is a thin wrapper that constructs the loop,
calls the verb, parses the result, and applies the §3 gates.

---

## §3 — Pre-registered falsification criteria

This section is written before §4. The protocol of §4 is designed to
satisfy the criteria of §3, not the other way around.

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

`σ_H` is the Flyvbjerg–Petersen blocked SEM of `H_geom_mean` across the
8 seeds.

### 3.2 Required sham controls

A verdict requires all five shams to satisfy their gates (S₄ is
absorbed into the antisymmetric primary observable per §0.2; not a
separate sham).

| Sham | Gate (v3.1.2) |
|---|---|
| **S₁** flat field (κ_Q ≡ 0) | `|H_S₁_mean| < 2 σ_S₁` AND `|H_S₁_mean| < ε_abs` |
| **S₂** α_Halcyon = 0 | `|H_S₂_mean| < ε_abs` (load-bearing); 2σ check is sanity |
| **S₃** μ_baseline scaled (×0.1, ×1, ×10) | **POSITIVE branch:** baseline-subtracted H invariant within 10% of `H_geom_mean`. **NULL / AMBIGUOUS branches:** `|H_S₃,raw at μ_baseline=1| < 2 σ_S₃` AND `< ε_abs`. Per-scaling H values reported as diagnostics. |
| **S₄** absorbed into H_geom | (no separate flag) |
| **S₅** degenerate loop (zero area in Λ) | `|H_S₅_mean| < 2 σ_S₅` AND `|H_S₅_mean| < ε_abs` |
| **S₆** frozen field | `|H_S₆_mean| < 2 σ_S₆` AND `|H_S₆_mean| < ε_abs` |

**`ε_abs = 1 × 10⁻¹⁰`** is the pre-registered empirical-numerical
floor. *Rationale (patched in v3.1.2):* this value is the floor below
which the substrate's own GC₂ (Abelian area-law to 1%) and GC₆ (gauge
invariance to machine ε) acceptance tests, run before v3.1.2 calls
the verb for science, empirically demonstrate the verb is
numerically clean on its own controls. *This is not a worst-case
`N · ε_machine` bound* (a worst-case bound at the loop's ~10⁵–10⁶
substrate-side FLOPs would be closer to ~10⁻¹⁰ for well-conditioned
arithmetic but could be ~10⁻⁷ for adversarial cancellation patterns).
The pre-registered choice of 10⁻¹⁰ is the empirical operating value
the GC tests confirm; if the substrate's GC tests show the floor sits
higher (e.g., 10⁻⁸), v3.1.2 still treats 10⁻¹⁰ as the absolute gate —
the substrate must clean up to that level on its sham-equivalent
inputs before science calls fire. If the GC tests show it sits lower
(e.g., 10⁻¹²), the deposit value 10⁻¹⁰ remains the pre-registered
gate; this is intentional, since pre-registration commits to the
gate, not to the tightest possible empirical floor.

### 3.3 Stopping rule

The framework is declared *falsified by simulation* if all four of
the following are met:

1. v3.1.2 returns NULL at the preregistered calibration set
   (α_Halcyon ∈ {1, 1000, and the eventual Field-Equations
   derivation if it lands before run time}), with sham controls
   passing.
2. A second independent v3-class measurement design — to be
   specified in a hypothetical SPEC v4 with its own pre-registration —
   also returns NULL on the same substrate at the same calibration.
3. The two measurement designs are not trivially equivalent
   (independence verified by an external reviewer per §8.5).
4. No subsequent re-scaling of α invalidates the prior NULL unless
   the observable's scaling law in α was itself preregistered in
   this document or in v4's. A later derivation of α may motivate a
   new sensitivity target for a hypothetical v5; it does not erase
   the v3.1.2 NULL at the calibration it was tested at.

The framework remains *not yet falsified* if v3.1.2 returns POSITIVE,
or if v3.1.2 returns AMBIGUOUS and the §3.7 re-run criteria are met.

### 3.4 The consistent-sign anti-fishing rule (on shams)

For each sham, compute the per-seed `H_sham,i` for i = 1…8. The
sham *fails* (regardless of the |mean| gate) if `sign(H_sham,i)` is
the same for at least 6 of 8 seeds AND `|mean H_sham| > 0.5 σ_sham`.
This catches the case where a sham's mean stays under 2σ but its
sign pattern shows a real but small bias.

### 3.5 Per-seed sign-coherence (patched: POSITIVE only)

For **POSITIVE classification**, at least 5 of 8 primary-loop seeds
must share the sign of `H_geom_mean`. The verdict's POSITIVE branch
requires both the global threshold (`|H_geom_mean| > 5 σ_H`) and the
per-seed sign-coherence majority.

For **NULL classification**, no sign-coherence requirement is
imposed. A true null has random signs across seeds — that is the
expected pattern, not a defect. NULL is achieved by
`|H_geom_mean| < 1 σ_H` regardless of the sign distribution.

For **AMBIGUOUS classification**, sign-coherence is a contributing
factor: if `|H_geom_mean|` is in the 1σ–5σ range AND fewer than 5/8
seeds share a sign, the verdict is AMBIGUOUS specifically because the
ensemble is incoherent. If `|H_geom_mean|` is in the 1σ–5σ range
*with* 5/8 sign-coherence, the verdict is still AMBIGUOUS (because
the global threshold for POSITIVE is not met) but the sign coherence
is reported as evidence pointing toward POSITIVE; the §3.7 re-run
would be motivated by either improving statistics or refining
calibration.

The per-seed sign-coherence rule **does not require per-seed σ_H,i**.
v3.1.2 does not require within-seed loop repeats. Individual seeds
are classified by sign only; the global classification uses the
blocked SEM across the 8-seed ensemble.

### 3.6 Calibration commitments

The v3.1.2 simulation will run at three calibrations:

- `α_Halcyon = 1` (v2 default — reproduces the v2 calibration so
  results are directly comparable)
- `α_Halcyon = 1000` (v2 calibrated bump)
- `α_Halcyon = TBD from the Davis Field Equations closed-form
  derivation` (the independent prediction; open work). If not yet
  derived at the time of v3.1.2 execution, the simulation runs at
  α=1 and α=1000 only.

`σ_H` is computed at the calibration corresponding to the relevant
verdict. A POSITIVE at α=1000 with NULL at α=1 is the predicted
pattern if the framework is right and α_Halcyon scales the signal
proportionally above the noise floor.

### 3.7 Ambiguity-resolution re-run criteria

If v3.1.2 returns AMBIGUOUS, a re-run is allowed only under all of
the following:

- The specific ambiguity (which gate failed and by how much) is
  named in the re-run's amendment.
- The re-run uses *more seeds or longer T_segment*, not different
  ω, F₀, T_loop, α, or β_W range. The protocol parameters are
  locked at v3.1.2's values.
- The amendment is committed and pushed before the re-run.
- The re-run's verdict is published per §8 regardless of direction.

---

## §4 — Q/β_W-ramp protocol via GIGI's HOLONOMY and TRANSPORT verbs

### 4.1 The loop γ_unit on Λ (patched: β_W range [2.5, 3.0])

$$
\gamma_{\rm unit}:\quad
(Q=0,\, \beta_W=2.5)
\to (Q=2,\, \beta_W=2.5)
\to (Q=2,\, \beta_W=3.0)
\to (Q=0,\, \beta_W=3.0)
\to (Q=0,\, \beta_W=2.5)
$$

Closed rectangle on (Q, β_W) inside the validated SU(2) operating
window. Enclosed area `= Q_max · Δβ_W = 2 × 0.5 = 1` (in mixed
control units; half of v3.1.1's value). `T_loop = 200`, `T_segment =
T_loop / 4 = 50`. The piecewise-linear ramp rate per axis is set in
§4.4.

### 4.2 The adiabaticity regime (active pinning explicit)

Two distinct regimes exist in principle. v3.1.2 explicitly declares
which one it uses.

**Regime A — passive adiabaticity (NOT used by v3.1.2):**
$$
T_{\rm drive} \;\ll\; T_{\rm segment}, \qquad
\tau_{\rm local\ eq} \;\ll\; T_{\rm segment} \;\ll\; \tau_{\rm unpinned\ drift}.
$$
At v2's measured values (`τ_unpinned_drift ≈ 10` time units), the
right inequality fails for any `T_segment > 10`. v3.1.2 does NOT
attempt passive adiabaticity.

**Regime B — active-pinning adiabaticity (the v3.1.2 regime):**
$$
T_{\rm drive} \;\ll\; T_{\rm segment}, \qquad
\tau_{\rm pin} \;\ll\; T_{\rm segment},
$$
and the tracking-error gates of §4.3.

At v3.1.2's values: `T_drive = 2π ≈ 6.28`, `T_segment = 50`, so
`T_drive / T_segment ≈ 0.126` (genuinely `<<`, ~8 drive cycles per
segment). `τ_pin ≈ 1` at `λ_pin = 1.0`, so `τ_pin / T_segment ≈ 0.02`
(genuinely `<<`).

### 4.3 The tracking-error gates

Pre-registered tolerances: `ε_Q = 0.05`, `ε_{β_W} = 0.05`.

$$
\max_t |Q_{\rm surr}(t) - Q_{\rm target}(t)| \;<\; \epsilon_Q,
\qquad
\max_t |\beta_{W,{\rm surr}}(t) - \beta_{W,{\rm target}}(t)| \;<\; \epsilon_{\beta_W}.
$$

Tracking-error violation forces AMBIGUOUS regardless of the H values.
The substrate is required (per the v3.1.2 Halcyon→Gigi letter) to
compute and emit the tracking error per substep per axis.

### 4.4 The GIGI call (patched: β_W range [2.5, 3.0])

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
  COMPUTE ADIABATICITY_CHECK
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
τ_Q model's coupling coefficient appearing in
`τ_Q(e) = τ₀ / (1 + β_τ s_Q(e))`. They are different parameters and
must not be confused. The call carries both because the τ_Q model
uses BETA_TAU (held fixed at 2.0) while the loop traverses values of
BETA_WILSON (varying along the loop's second axis).

### 4.5 The integration (substrate-side)

The closed-loop holonomy of the connection A on Λ is the substrate's
responsibility. The Python orchestrator does not re-implement it. The
substrate's HOLONOMY verb computes
`U[γ] = P exp ∮ A` via discretized parallel transport along γ. For
the abelianized test-mass response this reduces to a path-ordered
exponential of a scalar 1-form. The substrate-side correctness
contract is the v3.1.2 verb acceptance battery of §7.4.

### 4.6 The orchestrator surface (specification only; no implementation)

```python
def run_holonomy_battery(alpha_halcyon, seeds, log_path):
    """SPEC v3.1.2 §3 + §4 + §5; pure delegation to GIGI."""
    forward = gigi_sample_transport(loop=GAMMA_UNIT_FORWARD,
                                     alpha_halcyon=alpha_halcyon,
                                     seeds=seeds, ...)
    reversed_ = gigi_sample_transport(loop=GAMMA_UNIT_REVERSED, ...)
    H_geom = 0.5 * (forward.H - reversed_.H)
    H_sys  = 0.5 * (forward.H + reversed_.H)
    shams = {name: gigi_sample_transport(loop=..., sham_flag=name, ...)
             for name in SHAM_FLAGS}
    return apply_v3_1_2_gates(H_geom, H_sys, shams,
                              forward.tracking_error_Q, forward.tracking_error_beta_W,
                              forward.adiabaticity_check)
```

No leapfrog. No demodulation. No force computation. All substrate.

---

## §5 — Sham controls (full specifications)

| Sham | Implementation flag (verb-side) | Gate |
|---|---|---|
| S₁ flat field | `SHAM_FLAT_FIELD = true` (κ_Q ≡ 0 on all edges, all times) | `|H_S₁| < 2σ_S₁` AND `|H_S₁| < 10⁻¹⁰` |
| S₂ α=0 | `ALPHA_HALCYON = 0` (substrate zeros the Halcyon coupling) | `|H_S₂| < 10⁻¹⁰` (load-bearing); 2σ check is sanity |
| S₃ mass scaled | `MU_BASELINE ∈ {0.1, 1.0, 10.0}`; substrate fits baseline-subtracted H via linear fit | **POSITIVE branch:** baseline-subtracted H invariant within 10%. **NULL/AMBIGUOUS branches:** `|H_S₃ at μ_baseline=1| < 2σ_S₃` AND `< 10⁻¹⁰`. Per-scaling H values reported as diagnostics. |
| S₄ absorbed into H_geom | (no separate flag; substrate must compute both H_forward and H_reversed) | folded into §3.1 antisymmetric primary observable |
| S₅ degenerate loop | `LOOP gamma_degenerate` (zero area in Λ) | `|H_S₅| < 2σ_S₅` AND `|H_S₅| < 10⁻¹⁰` |
| S₆ frozen field | `SHAM_FROZEN_FIELD = true` (U held; Q_target and β_W_target updated) | `|H_S₆| < 2σ_S₆` AND `|H_S₆| < 10⁻¹⁰` |

The S₄ absorption is the v3.1 patch (carried into v3.1.2) that makes
the orientation-reversal test *unfakeable by construction*: any
v3.1.2 simulation that does not compute `H_forward` AND `H_reversed`
is non-compliant with the SPEC.

---

## §6 — What v3.1.2's results mean for v2

### 6.1 POSITIVE v3.1.2 + NULL v2

v2's null is **consistent with the adiabatic/fixed-point limit** of
the v3.1.2 holonomy model. The fixed-Q lock-in trivialized the
connection's pullback to a single point in Λ, where the loop encloses
zero area and the holonomy is identically zero regardless of
curvature. v3.1.2's positive is the first evidence that the bundle
has non-trivial curvature on Λ.

This is *not* a "confirmed prediction" of v2's null — v2 was not
pre-registered against the v3.1.2 model. It is a consistency check
that survives both ways: the v3.1.2 model also predicts the v2 limit.

### 6.2 NULL v3.1.2 + NULL v2

Stopping rule per §3.3 triggers if the conditions are met (independent
v4 measurement design also NULL, external review). v2 is the
adiabatic-limit complementary control; the joint null is stronger
than either alone.

### 6.3 NULL v3.1.2 + (smoke-mode internal pass on v2)

The v2 internal-extractor signal (`|α|/σ = 15.6`) is not contradicted
by a v3.1.2 null. v2's internal extractor confirmed model mechanics;
v3.1.2 (and v2's external extractor) tests observability. A v3.1.2
NULL means the framework's mechanics produce no observable holonomy
on Λ within the validated regime, which is the load-bearing
falsification.

### 6.4 POSITIVE v3.1.2 + (POSITIVE v2 internal)

Strongest positive case: model's internal state and observable
holonomy both match prediction.

### 6.5 AMBIGUOUS v3.1.2

Per §3.7, re-run is allowed only with named ambiguity, more seeds or
longer T_segment, no protocol parameter changes, amendment committed
before re-run.

---

## §7 — GIGI audit surface (inlined: full GC₁–GC₆ acceptance battery)

### 7.1 Two-layer independent auditability

- **Layer 1 — substrate computation.** `SAMPLE_TRANSPORT`,
  `HOLONOMY`, `TRANSPORT`, `SPECTRAL` are GIGI verbs. Their
  correctness is the domain of GIGI's test suite. A reviewer who
  wants to verify the substrate math runs the GIGI tests
  (`cargo test --features halcyon`) and inspects the WAL receipts.
- **Layer 2 — protocol design.** The choice of loop γ_unit, the
  adiabaticity condition, the sham controls, the gate thresholds,
  the stopping rule are the domain of this SPEC. A reviewer who
  wants to verify the experimental design reads this document.

Neither layer needs to inspect the other's internals to perform its
review. The interface is the GIGI verb signature.

### 7.2 The receipt model

v3.1.2 emits a `section_12_holonomy_battery_v3_1_2` JSON sidecar
with:

- The SHA-256 of this SPEC at execution time
- The GIGI deploy hash used for the substrate calls
- Per-seed `H_forward` and `H_reversed` and the sham results
- Per-gate verdict
- The σ_H values used in the gate evaluation
- The tracking-error traces (max values per axis)
- The adiabaticity-check result
- The overall verdict (POSITIVE / NULL / AMBIGUOUS)

Reproducibility: anyone with this SPEC's SHA-256, the GIGI deploy
hash, and the seed list can re-run the experiment and verify the
sidecar.

### 7.3 What this changes for the auditor's job

In v2, the auditor had to read ~1150 lines of Python to verify the
measurement. In v3.1.2, the auditor reads this SPEC and trusts the
substrate's correctness audit (amortized across every chapter using
the substrate). The audit surface is smaller and more orthogonal.

### 7.4 GIGI verb acceptance battery (inlined for self-containedness)

Before the v3.1.2 simulation calls `SAMPLE_TRANSPORT ALONG_LOOP` for
science, the substrate-side verb must pass the following six
contracts. These are testable by GIGI's `cargo test` suite at the
verb's introduction, before any Halcyon-side integration:

| # | Contract | Test |
|---|---|---|
| **GC₁** | **Flat connection returns zero.** | Construct a known-flat connection (`A ≡ 0` in synthetic mode); verify `H[any loop] = 0` to machine ε across at least 4 loop shapes. |
| **GC₂** | **Known area law for an Abelian constant-curvature connection.** | Construct a connection with constant curvature `F₀` in `(Q, β_W)`; verify `H[γ] = F₀ · Area(γ)` to 1% across 3 loop sizes. |
| **GC₃** | **Reversed loop inverts/sign-flips.** | For an arbitrary connection, verify `H[γ⁻¹] = −H[γ]` (Abelian) or `H[γ]⁻¹` (non-Abelian) to 1% across at least 3 connections. |
| **GC₄** | **Zero-size loop returns zero.** | Construct a degenerate loop (any closed curve bounding zero area); verify `H = 0` to machine ε. |
| **GC₅** | **Discretization convergence.** | Compute H at `N_discretization ∈ {1000, 2000, 4000, 8000, 16000}`; verify monotone convergence with relative change `< 1%` between 8000 and 16000 substeps. |
| **GC₆** | **Gauge invariance.** | Apply a known gauge transformation to the substrate's connection; verify H is invariant to machine ε. |

The substrate must pass GC₁–GC₆ before v3.1.2 science calls are made.
If any contract fails, v3.1.2 is gated on the substrate-side patch.
The existing 1373-assertion GIGI test suite is necessary but not
sufficient; GC₁–GC₆ are the **new** contracts the new verb
introduces, and they are the substrate-side correctness audit that
closes v3.1.2's two-layer auditability story.

---

## §8 — Publication commitment

### 8.1 The commitment

This document is committed to the public record at:

- **GitHub:** `nurdymuny/davis-wilson-map`, branch
  `feat/halcyon-gigi-substrate`, file
  `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.2.md`.
- **Zenodo:** a DOI will be minted from this SPEC's commit hash
  before the v3.1.2 protocol runs. The Zenodo deposit is mandatory;
  the GitHub commit alone is the implementation-level
  pre-registration; the Zenodo DOI is the publication-level
  pre-registration.

The git commit hash of this SPEC's first push IS the
pre-registration timestamp. The simulation will not run against
this spec until both the git push and the Zenodo deposit are
complete.

### 8.2 What gets published when the v3.1.2 run completes

All of the following are published, regardless of the verdict:

- The `section_12_holonomy_battery_v3_1_2` sidecar JSON.
- A revision of Solves Vol. 4 with an Appendix A.8 reporting the
  v3.1.2 result (POSITIVE, NULL, or AMBIGUOUS).
- The log file from the run, with per-seed verdicts.
- The GIGI deploy hash and WAL receipt for the substrate calls.

There is no version of this in which the result is suppressed if
unfavorable.

### 8.3 If v3.1.2 is NULL

Per §3.3 and §6.2, NULL is publishable. The chapter v5 reports the
null, names v2's null as the complementary control, and either (a)
declares the stopping condition per §3.3 (if a second independent
measurement design also returns null), or (b) names the conditions
for a v4 measurement design under external review. The null is *not*
re-run in a v3.1.2a without external review.

### 8.4 If v3.1.2 is POSITIVE

The chapter v5 reports the positive, names v2's null as the
adiabatic-limit complementary case, and proceeds to design the
apparatus-side measurement that would replicate the holonomy on
hardware. The simulation's positive is necessary for the apparatus
claim but not sufficient.

### 8.5 The committee

The stopping-rule committee of §3.3 — required for declaring the
framework falsified — is to be assembled at the time the second
NULL is recorded, not in advance. The committee consists of: (a)
Gigi, (b) one external lattice-gauge-theory reviewer chosen by Gigi
from outside the program, (c) one peer reviewer from a journal
submission process. The committee's role is to verify that the two
measurement designs (v3.1.2 and the hypothetical v4) are not
trivially equivalent.

---

## §9 — What this SPEC does not commit to

- No specific `α_Halcyon` from Davis Field Equations (open work).
- No commitment to the v3.1.2 result.
- No hardware apparatus design.
- v2 not deprecated.
- No promise that v3.1.2 will succeed in detecting the predicted
  holonomy.
- No extension of the β_W range below 2.5 without a v3.1.3 amendment
  AND an independent SU(2) Q-tracking validation at the proposed
  lower endpoint.

---

## §10 — Definitions of done

- [ ] This v3.1.2 SPEC committed to GitHub.
- [ ] v3.1.2 SPEC deposited on Zenodo (with v3.0, v3.1, and v3.1.1
  attached for chain-of-custody transparency).
- [ ] GIGI verb (per the v3.1.2 Halcyon→Gigi letter) lands with
  `CONTROL_MANIFOLD (Q, beta_wilson)`, `RAMP_RATE_BETA_W` for the
  `[2.5, 3.0]` window, `PIN_LAMBDA_BETA_W`, `EPS_BETA_W`,
  `TRACKING_ERROR_TRACE_BETA_W`, and the five-sham flag set.
- [ ] GIGI verb passes the GC₁–GC₆ acceptance battery of §7.4.
- [ ] Python orchestrator implemented as a thin delegation wrapper.
- [ ] v3.1.2 run produces sidecar matching §7.2.
- [ ] Verdict published per §8 regardless of direction.
- [ ] Solves Vol. 4 Appendix A.8 reports the v3.1.2 result.

---

## §11 — Acknowledgments

v3.0 → v3.1 was driven by the first round of external review (the
two mathematical defects: scalar holonomy vanishing by FTC,
adiabaticity inequality reversed). v3.1 → v3.1.1 was driven by the
second round of external review (seven executability defects).
v3.1.1 → v3.1.2 was driven by the third round of external review
(the validity-window blocker — β_W traversal outside the SU(2)
operating regime — plus three smaller patches: self-containedness,
ε_abs rationale, NULL-branch sign-coherence).

All three rounds of external review were load-bearing for the
protocol. Each version's §0 changelog names every patch and which
review surfaced it. Pre-registration that does not admit correction
before deposit is brittle; pre-registration that admits correction
after deposit is meaningless. v3.1.2 reflects four review iterations
(Gigi's methodological intervention, GPT rounds 1, 2, and 3) all
completed before deposit. The Zenodo deposit timestamp will fire on
v3.1.2, and from that moment §3 cannot move.

The discipline that produced this — three preserved drafts plus the
canonical v3.1.2 — demonstrates pre-registration's intended property:
each review pass caught real issues that a one-pass pre-registration
would have locked in. The cost of admission to either credibility,
positive or negative, is paid in the patches recorded above, not in
the protocol that runs.
