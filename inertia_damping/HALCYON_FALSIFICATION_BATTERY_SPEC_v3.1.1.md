# Halcyon Falsification Battery — SPEC v3.1.1 (executability patch)

**Status:** PRE-REGISTRATION, deposit-ready. Supersedes
`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.md` (commit
`712109488d43cf2fcd43b8d2bc8b5a1b053579ec`) **before** that document
reached Zenodo deposit. v3.1 stays preserved as the second-draft
pre-registration that the second-round external review caught;
v3.1.1 is the document that actually goes to Zenodo and is cited as
the canonical pre-registered protocol.

**Date written:** 2026-06-21 (same calendar day as v3.0 and v3.1)
**Implementation status at time of writing:** none. No v3 code exists yet.
**Predecessors retained as first-class artefacts:**
- `HALCYON_FALSIFICATION_BATTERY_SPEC.md` (v2.0 / v2.1)
- `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md` (v3.0; first-draft pre-registration)
- `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.md` (v3.1; second-draft pre-registration)

**Author commitment:** v3.0 fixed the philosophical posture but had two
mathematical defects. v3.1 fixed those defects but left seven
executability issues. v3.1.1 fixes the executability issues. The Zenodo
deposit, the Halcyon→Gigi letter, and the v3.1.1 implementation are
all gated on *this* document's commit hash. v3.1.1 is the contract.

## §0 — v3.1 → v3.1.1 changelog (the seven executability patches)

Second-round external review of v3.1 (ChatGPT, 2026-06-21) caught seven
executability issues between v3.1's GitHub commit and its planned
Zenodo deposit. Each patch is locally surgical; none requires a
structural redesign. The five protocol-architecture decisions of v3.1
(connection 1-form on Λ, antisymmetric H_geom primary observable,
six-sham control set with S₄ absorbed into the antisymmetric primary,
two-layer audit story with the GC₁–GC₆ verb acceptance battery, named
stopping rule with calibration-local nulls) all carry forward
unchanged.

### §0.1 The seven patches

| # | v3.1 issue | v3.1.1 patch | Lives in |
|---|---|---|---|
| 1 | The second control coordinate `θ` was left selectable at run time, weakening pre-registration | `θ` locked to `β_W` (Wilson gauge-action coupling); selection at deposit time, not run time | §2.2, §4.4 |
| 2 | Timing chain `τ_local_eq << T_segment << τ_unpinned_drift` still failed passively at the v3.1 numerical values | Split into two named regimes: (a) passive (which v3.1.1 does NOT use); (b) active-pinning (which v3.1.1 uses); explicit declaration | §4.2 |
| 3 | "~4 cycles per segment" at v3.1 numbers is not really `<<` | T_segment bumped 25 → 50 (T_loop 100 → 200), giving 8 cycles per segment; `<<` notation kept | §4.2, §4.4 |
| 4 | `N_DISCRETIZATION 4000` comment said "400 drive cycles" but T_loop=100, ω=1 actually means 15.9 drive cycles | N updated to 10000 to match dt=0.02 over T_loop=200; comment corrected to "~31.8 drive cycles, ~314 steps per cycle" | §4.4 |
| 5 | S₂ absolute-ε threshold was named but not specified | `ε_abs = 1×10⁻¹⁰` declared as the absolute floor (10⁻¹⁰ is float64 propagated through ~10⁹ FLOPs at machine-ε per FLOP) | §3.2, §5 |
| 6 | S₃ relative-invariance gate divides by `|H_geom|` and blows up in the NULL branch | S₃ relative gate applies only in POSITIVE branch; in NULL/AMBIGUOUS branches S₃ uses the same 2σ + absolute-ε threshold as the other shams | §3.2, §5 |
| 7 | Per-seed "individual classification" requires a per-seed σ but none was defined | Per-seed classification is by **sign only**; global POSITIVE/NULL/AMBIGUOUS uses the blocked SEM across seeds. No within-seed loop repeats required | §3.5 |

The §0 changelog is the contract: v3.1.1's pre-registration commits to
the corrected §2.2 / §3 / §4 / §5 below, not to v3.1's. Anyone citing
the pre-registration cites *this* commit hash.

### §0.2 What did NOT change

- The connection-1-form definition of holonomy (§2.3).
- The antisymmetric primary observable H_geom (§3.1).
- The five-sham set after S₄ absorption (§3.2 / §5).
- The two-layer audit story and the GC₁–GC₆ verb acceptance battery (§7).
- The named stopping rule with calibration-local NULLs (§3.3).
- The publication commitment of §8.

If you read v3.1 carefully, you have read v3.1.1's structure. The
patches are surgical.

---

## §1 — Why v2's measurement was wrong for the dynamics being measured

(Unchanged from v3.1. H₁/H₅/H₉ stand; α-scaling-vs-SNR is the
load-bearing diagnostic forcing the redesign; v2 sidecars preserved.
See v3.1 §1 in the predecessor commit `7121094` for the full text;
v3.1.1 inherits §1 verbatim.)

---

## §2 — The framework's native observable is holonomy

### 2.1 Three-sentence anchor (unchanged)

> The framework's native object is holonomy.
> The apparatus measures holonomy.
> The simulation should compute holonomy.

### 2.2 The bundle and the control manifold (patched: θ → β_W)

The test mass is a section of a bundle whose base is a multi-dimensional
**programmed control manifold Λ** and whose fiber is the test mass's
configuration space. Λ is at least two-dimensional. In v3.1.1:

$$
\Lambda \;=\; \bigl\{(Q,\,\beta_W) \;:\; Q \in [0, Q_{\max}],\, \beta_W \in [\beta_{W,\min},\, \beta_{W,\max}]\bigr\}
$$

where:

- **Q** is the surrogate sector coordinate (the v2 Q-label, unchanged).
  Operational range: `Q ∈ [0, 2]`. Q is a *programmed-control
  coordinate*, not a smooth topological label — see v3.1 §0 patch #3.
- **β_W** is the **Wilson gauge-action coupling**, i.e. the β that
  appears in the Wilson action `S_W = (β_W/N) Σ_f [N − Re Tr U_f]`.
  Operational range: `β_W ∈ [2.0, 3.0]`. The Migdal–Witten canonical
  operating point β_W = 2.5 sits at the midpoint of the range.
  *β_W is the same β that GIGI's existing gauge_field declarations
  already carry.* Driving β_W during a `SAMPLE_TRANSPORT` call is
  the substrate-side ask of the v3.1.1 Halcyon→Gigi letter §4.

**Why β_W is the right second coordinate.** It is physically meaningful
(the cage controls the junction biases that set the Wilson coupling),
it already exists in the substrate's verb grammar (no new conceptual
introduction), and it genuinely couples to the test mass through the
substrate's existing `κ_Q τ_Q² |φ_n|²` integrand (κ depends on β_W
through the face-holonomy distribution; τ_Q depends on β_W through
s_Q, which depends on β_W through the staple field strengths). A pure
gauge-rotation parameter would trivialize the holonomy (gauge
transformations don't change observables); β_W avoids that trap.

**The v3.1.1 commitment.** The second coordinate is locked to β_W at
this commit. If GIGI's eventual implementation requires a different
second coordinate for technical reasons (e.g., the substrate cannot
drive β_W within a single `SAMPLE_TRANSPORT` call and must use an
alternative knob), that requires a **v3.1.2 amendment committed and
pushed before execution**. Run-time selection is prohibited.

### 2.3 The holonomy as a connection 1-form on Λ (unchanged from v3.1)

$$
A \;=\; A_Q(\lambda)\,dQ \;+\; A_{\beta_W}(\lambda)\,d\beta_W
$$

$$
U[\gamma] \;=\; \mathcal{P}\exp\left(\oint_\gamma A\right), \qquad
\mathcal{H}[\gamma] \;=\; \oint_\gamma A_i\, d\lambda^i \;=\; \int_\Sigma F + O(\mathrm{area}^2)
$$

where $F = dA + A \wedge A$ is the curvature, and the Halcyon
prediction is `F ≠ 0` on Λ. A trivial (flat-A) control connection
gives H[γ] = 0 for every closed loop on Λ; the prediction is that the
buckyball substrate's specific geometry produces non-zero curvature on
the (Q, β_W) plane.

### 2.4 The pulled-back connection on the worldline (unchanged from v3.1)

### 2.5 GIGI's native verbs (unchanged target; second axis now β_W)

The substrate call is `SAMPLE_TRANSPORT … ALONG_LOOP … CONTROL_MANIFOLD
(Q, beta_wilson) … ADIABATIC … COMPUTE HOLONOMY`. See §4.4 below and
the v3.1.1 Halcyon→Gigi letter.

---

## §3 — Pre-registered falsification criteria (patched in §3.2 #5–#6, §3.5 #7)

This section is written before §4. The protocol of §4 is designed to
satisfy the criteria of §3, not the other way around.

### 3.1 The primary observable and its three regimes (unchanged from v3.1)

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
| **POSITIVE** | `|H_geom| > 5σ_H` AND `|H_sys| < 1σ_H` AND every sham passes its threshold |
| **NULL** | `|H_geom| < 1σ_H` AND `|H_sys| < 1σ_H` AND every sham passes its threshold |
| **AMBIGUOUS** | Any of: `1σ ≤ |H_geom| ≤ 5σ`; `|H_sys| ≥ 1σ_H`; any sham fails its threshold; any sham shows consistent-sign pattern across ≥ 6/8 seeds with `|mean| > 0.5σ_sham` (the anti-fishing rule of §3.4, unchanged) |

σ_H is the Flyvbjerg–Petersen blocked SEM of H_geom across 8 seeds.

### 3.2 Required sham controls (patched: S₂ absolute ε, S₃ NULL-branch)

| Sham | Patched gate (v3.1.1) |
|---|---|
| **S₁** flat field (κ_Q ≡ 0) | `|H_S₁| < 2σ_S₁` AND `|H_S₁| < ε_abs = 1×10⁻¹⁰` |
| **S₂** α_Halcyon = 0 | `|H_S₂| < 2σ_S₂` AND `|H_S₂| < ε_abs = 1×10⁻¹⁰`. *The absolute floor matters more than the relative for S₂* because α=0 zeros the coupling at the multiplication level, so any non-machine-ε result is a substrate bug or a parametric artifact. |
| **S₃** μ_baseline scaled (×0.1, ×1, ×10) | **POSITIVE branch:** baseline-subtracted H invariant within 10% (the v3.1 gate). **NULL / AMBIGUOUS branches:** the relative-invariance gate is not evaluated (the denominator is near zero); S₃ instead satisfies `|H_S₃,raw at μ_baseline=1| < 2σ_S₃` AND `< ε_abs`, with the per-scaling H values reported as diagnostics. |
| **S₄** absorbed into H_geom | (no separate sham; the primary observable IS the test) |
| **S₅** degenerate loop (zero area in Λ) | `|H_S₅| < 2σ_S₅` AND `|H_S₅| < ε_abs = 1×10⁻¹⁰` |
| **S₆** frozen field | `|H_S₆| < 2σ_S₆` AND `|H_S₆| < ε_abs = 1×10⁻¹⁰` |

**ε_abs = 1×10⁻¹⁰** is the pre-registered absolute floor. Rationale:
float64 machine ε ≈ 2.2×10⁻¹⁶; the v3.1.1 substrate computation involves
~10⁵–10⁶ floating-point operations per loop traversal; the propagated
round-off after that many operations is bounded by ~10⁻¹⁰ in absolute
terms. Anything larger by an order of magnitude at α=0 or on a
flat/frozen/degenerate field is a bug or a real signal, not numerics.

### 3.3 Stopping rule (unchanged from v3.1)

Calibration-local NULLs. The four-part rule (NULL at preregistered α,
second independent measurement design also NULL, non-trivial-
equivalence, no α-rescaling escape unless preregistered) carries
forward verbatim.

### 3.4 The consistent-sign anti-fishing rule (unchanged from v3.1)

### 3.5 Per-seed independence (patched: sign-only individual classification)

A primary or sham gate is "struck" only if **at least 5 of 8 seeds
individually have the same sign on H**, AND the global classification
(POSITIVE / NULL / AMBIGUOUS) is computed from the blocked SEM across
the full 8-seed ensemble.

That is:

- *Per-seed criterion (individual):* sign of `H_geom` at each seed.
  Struck iff ≥ 5/8 seeds share a common sign.
- *Global criterion (ensemble):* `|H_geom_mean| / σ_H_blocked` evaluated
  against the §3.1 thresholds, with σ_H_blocked from Flyvbjerg-Petersen
  on the 8-seed distribution.

The per-seed classification *does not require* a per-seed σ_H,i. v3.1.1
does not require within-seed loop repeats. (If the substrate-side
implementation gets within-seed repeats cheaply, the verb may return
per-seed σ; the SPEC does not require it.)

The sign-coherence majority is what catches the failure mode where a
single seed dominates the mean — it does not require an absolute
classification per seed. The ensemble-level σ_H is what enforces the
global thresholds. The two-level test is internally consistent and
executable from a single 8-seed run.

### 3.6 Calibration commitments (unchanged from v3.1)

### 3.7 Ambiguity-resolution re-run criteria (unchanged from v3.1)

---

## §4 — Q-ramp protocol via GIGI's HOLONOMY and TRANSPORT verbs (patched in §4.2 #2–#3, §4.4 #4)

### 4.1 The loop γ_unit on Λ (patched: T_loop bumped 100 → 200)

$$
\gamma_{\rm unit}:\quad
(Q=0, \beta_W=2.0) \to (Q=2, \beta_W=2.0)
\to (Q=2, \beta_W=3.0) \to (Q=0, \beta_W=3.0) \to (Q=0, \beta_W=2.0)
$$

Closed rectangle on (Q, β_W). Enclosed area
`= Q_max · Δβ_W = 2 × 1 = 2` (in mixed control units). T_loop = 200
(doubled from v3.1's 100). T_segment = T_loop / 4 = 50 (doubled from
v3.1's 25). The piecewise-linear ramp rate per axis is set in §4.4.

### 4.2 The adiabaticity regime (patched: two regimes explicit; v3.1.1 uses active)

There are two distinct regimes the measurement could in principle
satisfy. v3.1.1 explicitly declares which one it uses.

**Regime A — passive adiabaticity (NOT used by v3.1.1, listed for
completeness):**
$$
T_{\rm drive} \;\ll\; T_{\rm segment}, \qquad
\tau_{\rm local\ eq} \;\ll\; T_{\rm segment} \;\ll\; \tau_{\rm unpinned\ drift}.
$$
At v2's measured values (τ_unpinned_drift ≈ 10 time units), the right
inequality fails for any T_segment > 10. v3.1.1 does NOT attempt
passive adiabaticity.

**Regime B — active-pinning adiabaticity (the v3.1.1 regime):**
$$
T_{\rm drive} \;\ll\; T_{\rm segment}, \qquad
\tau_{\rm pin} \;\ll\; T_{\rm segment},
$$
and the tracking-error gates
$$
\max_t |Q_{\rm surr}(t) - Q_{\rm target}(t)| \;<\; \epsilon_Q,
\qquad
\max_t |\beta_{W,{\rm surr}}(t) - \beta_{W,{\rm target}}(t)| \;<\; \epsilon_{\beta_W}.
$$

The active-pinning regime replaces the τ_unpinned_drift constraint
with the τ_pin constraint and the tracking-error gates. τ_pin is the
equilibration timescale of the pinning potential under v3.1.1's
λ_pin values (committed in §4.4). The pinning is part of the
simulation, mirroring the cage's role in the apparatus.

At v3.1.1's values: T_drive = 2π ≈ 6.28, T_segment = 50 (so
T_drive/T_segment ≈ 0.126, ~8 drive cycles per segment, genuinely
in the ≪ regime); τ_pin ≈ 1 (set by λ_pin = 1.0 in §4.4), so
τ_pin/T_segment ≈ 0.02 (genuinely ≪). Both regime-B inequalities
hold with comfortable margins.

### 4.3 The tracking-error gates (carried forward from v3.1)

Pre-registered tolerances: `ε_Q = 0.05`, `ε_{β_W} = 0.05`. Tracking-
error violation forces AMBIGUOUS regardless of the H values. The
substrate is required (per the Halcyon→Gigi letter) to compute and
emit the tracking error per substep.

### 4.4 The GIGI call (patched: β_W as second axis, T_loop = 200, N = 10000)

```
SAMPLE_TRANSPORT halcyon_canonical_buckyball
  ALONG_LOOP gamma_unit_in_Q_beta_W
  CONTROL_MANIFOLD (Q, beta_wilson)
  ADIABATIC TRUE
  RAMP_RATE_Q 0.04            // (Q_max - Q_min) / T_segment = 2 / 50
  RAMP_RATE_BETA_W 0.02        // (beta_W_max - beta_W_min) / T_segment = 1 / 50
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

(Disambiguation: `β_W` is the Wilson gauge coupling appearing in
`S_W = (β_W/N) Σ_f [N − Re Tr U_f]`. `BETA_TAU` is the v2.1 τ_Q
model's coupling coefficient appearing in `τ_Q(e) = τ₀/(1 + β_τ s_Q(e))`.
They are different parameters and must not be confused. The
`SAMPLE_TRANSPORT` call carries both because the τ_Q model uses
`BETA_TAU` while the loop traverses values of `BETA_WILSON`.)

### 4.5 The integration (unchanged from v3.1; substrate-side)

### 4.6 The orchestrator surface (unchanged from v3.1; specification only)

---

## §5 — Sham controls (patched: S₂ absolute ε, S₃ NULL-branch)

| Sham | Implementation flag (verb-side) | Gate (v3.1.1) |
|---|---|---|
| S₁ flat field | `SHAM_FLAT_FIELD = true` (κ_Q ≡ 0) | `|H_S₁| < 2σ_S₁` AND `|H_S₁| < 10⁻¹⁰` |
| S₂ α=0 | `ALPHA_HALCYON = 0` | `|H_S₂| < 10⁻¹⁰` (the absolute floor is the load-bearing gate here; the 2σ check is sanity) |
| S₃ mass scaled | `MU_BASELINE ∈ {0.1, 1.0, 10.0}`; substrate fits baseline-subtracted H | **POSITIVE branch:** baseline-subtracted H invariant within 10%. **NULL / AMBIGUOUS branches:** the v3.1 relative gate is replaced with `|H_S₃ at μ_baseline=1| < 2σ_S₃` AND `< 10⁻¹⁰`. The per-scaling H values are reported as diagnostics. |
| S₄ absorbed into H_geom | (no separate flag) | n/a |
| S₅ degenerate loop | `LOOP gamma_degenerate` (zero area) | `|H_S₅| < 2σ_S₅` AND `|H_S₅| < 10⁻¹⁰` |
| S₆ frozen field | `SHAM_FROZEN_FIELD = true` | `|H_S₆| < 2σ_S₆` AND `|H_S₆| < 10⁻¹⁰` |

---

## §6 — What v3.1.1's results mean for v2 (unchanged from v3.1)

POSITIVE_v3.1.1 + NULL_v2 = consistent with adiabatic-limit. NULL+NULL
= stopping rule per §3.3. POSITIVE + (internal v2 pass) = strongest
positive case. AMBIGUOUS = §3.7 re-run criteria. v2 is not deprecated.

---

## §7 — GIGI audit surface (unchanged from v3.1)

Two-layer auditability; GC₁–GC₆ verb acceptance battery the substrate
must pass before v3.1.1 calls it for science. See v3.1 §7 for the full
six-contract specification.

---

## §8 — Publication commitment (unchanged from v3.1; v3.1.1 is the canonical deposit)

The Zenodo deposit, when minted, targets v3.1.1 as the canonical
pre-registered protocol. v3.0 and v3.1 are included in the deposit
for transparency of the review-and-patch process, marked as
"first/second-draft pre-registrations that the review-before-deposit
process caught."

---

## §9 — What this SPEC does not commit to (unchanged from v3.1)

---

## §10 — Definitions of done (refreshed for v3.1.1)

- [ ] This v3.1.1 SPEC committed to GitHub.
- [ ] v3.1.1 SPEC deposited on Zenodo (with v3.0 and v3.1 attached for
  chain-of-custody transparency).
- [ ] GIGI verb (per the patched Halcyon→Gigi letter) lands with
  `CONTROL_MANIFOLD (Q, beta_wilson)`, `RAMP_RATE_BETA_W`,
  `PIN_LAMBDA_BETA_W`, `EPS_BETA_W`, `TRACKING_ERROR_TRACE_BETA_W`,
  and the five-sham flag set.
- [ ] GIGI verb passes the GC₁–GC₆ acceptance battery of §7.4.
- [ ] Python orchestrator implemented as a thin delegation wrapper.
- [ ] v3.1.1 run produces sidecar.
- [ ] Verdict published per §8.
- [ ] Solves Vol. 4 Appendix A.8 reports the v3.1.1 result.

---

## §11 — Acknowledgments

v3.0 → v3.1 was driven by the first round of external review
(mathematical defects). v3.1 → v3.1.1 was driven by the second round
of external review (executability defects: under-specified θ, internal
timing-regime inconsistency, weak `<<` separation, wrong
N_DISCRETIZATION comment, missing absolute ε for S₂, S₃ NULL-branch
division-by-zero, under-defined per-seed σ).

Both rounds of external review were load-bearing for the protocol. The
§0 changelog of each version names every patch and which review
surfaced it. Pre-registration that does not admit correction before
deposit is brittle; pre-registration that admits correction after
deposit is meaningless. v3.1.1 reflects three review iterations
(Gigi's methodological intervention, GPT round 1, GPT round 2) all
completed before deposit. The Zenodo deposit timestamp will fire on
v3.1.1, and from that moment §3 cannot move.
