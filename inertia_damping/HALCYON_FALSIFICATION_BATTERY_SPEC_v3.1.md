# Halcyon Falsification Battery — SPEC v3.1 (post-review patch)

**Status:** PRE-REGISTRATION. Supersedes `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md`
(commit `0fe654d556e4f6878c439df64d1ff20599c9c733`) **before** that document
reached Zenodo deposit. v3.0 remains preserved as the first-draft
pre-registration the v3.1 review caught; v3.1 is the document that
actually fixes the falsification criteria of the protocol in the public
record.

**Date written:** 2026-06-20 (same calendar day as v3.0, immediately after
external-review feedback)
**Implementation status at time of writing:** none. No v3 code exists yet.
**Predecessors retained as first-class artefacts:**
- `HALCYON_FALSIFICATION_BATTERY_SPEC.md` (v2.0/2.1)
- `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md` (v3.0; pre-review draft)

**Author commitment:** The Zenodo deposit, the Halcyon→Gigi letter, and the
v3 implementation are gated on *this* document's commit hash, not v3.0's.
v3.0 is preserved for the chain of custody; v3.1 is the contract.

## §0 — v3.0 → v3.1 changelog (the GPT-review patches)

Two mathematical defects and five protocol-discipline issues were caught
by external review of v3.0 between commit and Zenodo deposit. v3.1
patches them. The structure of §1–§8 is unchanged; the patches are
substantive and live inside the affected sections.

### §0.1 Defects patched (load-bearing)

1. **Scalar holonomy was identically zero by FTC.** v3.0 §2.3 and §4.5
   defined `H[γ] = ∮ ∂μ_eff/∂Q dQ`. For any single-valued scalar
   function μ_eff(Q), this integral vanishes for every closed loop by
   the fundamental theorem of calculus. v3.0 itself noticed this and
   then framed the "disagreement between telescoping and Wilson-loop
   methods" as the signal — but those are not two ways to compute the
   same observable; they are *one diagnostic that must vanish* and
   *one separate connection observable that may not*. v3.1 replaces
   the scalar derivative with a genuine connection 1-form on a
   multi-dimensional control manifold (§2.3 patched, §4.5 rewritten).

2. **Adiabaticity inequality was written backwards.** v3.0 §4.2 said
   `τ_drive >> T_loop >> τ_relax`, then assigned `T_drive≈6.28`,
   `T_loop=100`, `τ_relax≈10`. The numbers contradict the inequality
   in both directions. v3.1 §4.2 replaces the chain with the correct
   discipline: `T_drive << T_segment`, and `τ_local_eq << T_segment <<
   τ_unpinned_drift`, with an active-pinning tracking-error gate when
   the second inequality cannot be met passively.

### §0.2 Protocol-discipline corrections

3. **Q is a programmed-control coordinate, not a continuous topological
   label.** In SU(2) on S² the topological charge Q has π₂ = 0, and
   even where Q is integer-valued it labels disconnected sectors that
   no continuous path connects. v3.0 spoke about "ramping Q from
   0 → 1 → 2 → 1 → 0" as though Q were a smooth coordinate. v3.1
   renames the loop's home as the **programmed control manifold Λ**,
   whose projection onto the surrogate sector coordinate follows
   `Q_cmd : 0 → 1 → 2 → 1 → 0`. The control manifold has at least
   two dimensions, so non-trivial loops enclose area.

4. **The S₃ mass-scaling sham needs baseline subtraction.** The raw
   transfer-function response *does* scale with μ_baseline; only the
   baseline-subtracted coupling holonomy should be invariant. v3.0's
   gate as written would fail trivially. v3.1 §5 fixes the predicate.

5. **The S₄ reversed-loop sham needs the antisymmetric component.**
   Dissipative driven dynamics generate even-in-time artefacts that
   contaminate `H[γ]` and `H[γ⁻¹]` symmetrically. The geometric
   observable is `H_geom = ½(H[γ] − H[γ⁻¹])`; the symmetric component
   `H_sys = ½(H[γ] + H[γ⁻¹])` is the named systematic. v3.1 §5 gates
   on `H_geom`, reports `H_sys`.

6. **The all-six-shams-below-1σ gate has high false-ambiguous rate.**
   Each sham has independent noise; six independent draws *will*
   wander above 1σ occasionally. v3.1 §3 relaxes the sham threshold
   to 2σ_sham per sham (with family-wise discipline) and adds an
   anti-fishing rule on consistent sign patterns.

7. **The per-seed-strikes criterion was under-specified.** "≥ 5/8
   seeds individually strike" did not say whether that meant *same
   sign*, *individually above 1σ*, or *individually passing the same
   primary classification*. v3.1 §3.5 names it: same primary/sham
   verdict and same sign at ≥ 5/8 seeds.

8. **The stopping rule had a calibration-escape clause.** v3.0 §3.3
   said "the framework remains not yet falsified if NULL but α is
   refined downward." This is the slide into protocol-shopping the
   pre-registration is designed to prevent. v3.1 §3.3 makes the
   calibration result local: a NULL at preregistered α is a NULL
   *for that calibration on that substrate*. A later derivation of α
   motivates a new measurement sensitivity target; it does not erase
   the prior NULL unless the observable's scaling in α was itself
   preregistered.

9. **The v2-reinterpretation in v3.0 §6.1 was too strong.** "v2's
   null is no longer a failure of detection; it is a confirmed
   prediction" claimed independent confirmation that v2's null cannot
   provide (v2 was not pre-registered against the v3 model). v3.1 §6.1
   replaces "confirmed prediction" with "consistent with the adiabatic
   / fixed-point limit of the v3 holonomy model."

10. **The GIGI substrate audit needs an explicit verb acceptance
    battery.** v3.0 §7 said "trust GIGI's 1373 assertions" without
    listing which contracts the new HOLONOMY verb must satisfy. v3.1
    §7.4 adds a six-contract acceptance battery the verb must pass
    before v3 calls it for science.

The §0 changelog is the contract: v3.1's pre-registration commits to
the corrected §3 and §4 below, not to v3.0's. Anyone citing the
pre-registration cites *this* commit hash.

---

## §1 — Why v2's measurement was wrong for the dynamics being measured

(Unchanged from v3.0. The architectural diagnosis, the H₁/H₅/H₉
positive findings that stand, the α-scaling-vs-SNR load-bearing
diagnostic, and the v2 ↔ v3 cross-validation framing all survive
into v3.1. Repeated here in summary so v3.1 is self-contained.)

### 1.1 What v2 established (load-bearing into v3.1)

- **H₁ (material independence) strikes**: `|dα/dμ_proxy|/|α| = 0.005–0.028`.
  Halcyon coupling is geometric, not material-dependent. Structural
  property of the framework, not the measurement type. *Survives v3.1.*
- **H₅ (drive-amplitude linearity) strikes** at machine precision.
  Lock-in regime valid. *Survives v3.1.*
- **H₉ (τ_Q model robustness) strikes**: `|α_alt − α|/|α| = 0.025–0.13`.
  Not overfitted to the specific τ_Q form. *Survives v3.1.*
- **Newtonian limit exact at Q=0**: μ_eff = 0 by construction at
  trivial vacuum. *Survives v3.1.*
- **Smoke-mode internal signal real**: |α|/σ = 15.6 on internal
  μ_eff(U(t)) state. Davis mechanics work at the model level.
  *Survives v3.1.*

### 1.2 The load-bearing diagnostic that forces redesign

Bumping α_Halcyon 1000× moved observation-space SNR by only 2.5×.
Noise is intrinsic to the gauge-field dynamics being measured, not
extrinsic to the lock-in apparatus. When noise scales with signal, the
measurement is reading the wrong observable. The fixed-Q lock-in is
the adiabatic limit of a measurement the apparatus does diabatically.

### 1.3 What v2 sidecars remain

`battery_fast_20260620_104846.json`,
`battery_full_20260620_181227.json`,
`battery_calibrated_20260621_011304.json` are preserved with their
SHA-256s. v2.1 SPEC remains authoritative for its own design.

---

## §2 — The framework's native observable is holonomy

### 2.1 Three-sentence anchor (unchanged)

> The framework's native object is holonomy.
> The apparatus measures holonomy.
> The simulation should compute holonomy.

### 2.2 The bundle and the control manifold

The test mass is a section of a bundle whose base is a **multi-dimensional
programmed control manifold Λ** and whose fiber is the test mass's
configuration space. Λ is at least two-dimensional. Specifically, in
v3.1 the control manifold is

$$
\Lambda \;=\; \{(Q,\,\theta) : Q \in [0, Q_{\max}],\, \theta \in [0, 2\pi)\}
$$

where Q is the surrogate sector coordinate and θ is a **sector
orientation phase** — a second independent cage control corresponding,
in the apparatus, to a programmable Bloch-sphere axis around which the
sector winding is realized. A one-dimensional ramp in Q alone is *not*
a closed loop in Λ (it retraces itself in a single coordinate and
encloses no area); a closed loop in (Q, θ) encloses area and therefore
can pick up holonomy from a non-flat connection.

(If GIGI's substrate exposes a more natural second control coordinate
than θ — e.g., a β-coupling knob, a drive-phase parameter, or a
substrate-side gauge-rotation parameter — the v3.1 implementation may
choose that as the second axis. The constraint is that Λ has dim ≥ 2
and that loops on Λ project to the desired Q_cmd path. The second
coordinate is named in the verb call at run time.)

### 2.3 The holonomy as a connection 1-form on Λ (patched from v3.0)

The Halcyon coupling defines a connection 1-form on Λ:

$$
A \;=\; A_Q(\lambda)\,dQ \;+\; A_\theta(\lambda)\,d\theta
$$

where the components are derived from the substrate-level computation
of how the test mass's effective inertial coefficient transports as
the configuration moves through Λ. The closed-loop holonomy is the
ordered exponential of A along γ:

$$
U[\gamma] \;=\; \mathcal{P}\exp\left(\oint_\gamma A\right)
$$

(non-Abelian, general case). For the test mass's abelianized response
the holonomy reduces to

$$
\mathcal{H}[\gamma] \;=\; \oint_\gamma A_i\, d\lambda^i
\;=\; \int_\Sigma F \;+\; O(\text{area}^2)
$$

for a small loop bounding area Σ, where $F = dA + A \wedge A$ is the
curvature. The Halcyon prediction is that **the curvature F on Λ is
non-zero**; this is the falsifiable observable. A trivial (flat-A)
control connection gives H[γ] = 0 for every closed loop; the
prediction is `F ≠ 0`, with sign and magnitude determined by the
substrate's specific geometry.

For a one-dimensional path that retraces itself (the v3.0 mistake),
the loop bounds *zero area* in Λ regardless of the connection's
curvature, so `H[γ] = 0` by area alone. The v3.0 protocol could not
have detected non-trivial F even if it existed. v3.1 fixes this by
requiring loops with non-zero enclosed area in (Q, θ).

### 2.4 The pulled-back connection on the worldline

The bundle the test mass lives on is the pullback of the
gauge-field-coupling connection to the test mass's worldline-times-Λ
product. The cage drives the (Q, θ) coordinates as functions of time;
the connection's components in Λ become observable through the
test mass's response. The fixed-Q lock-in of v2 trivialized the
pullback to a single point in Λ; v3.1 measures the full transport.

### 2.5 GIGI's native verbs (unchanged target)

The v3.1 substrate call is `SAMPLE_TRANSPORT … ALONG_LOOP … ADIABATIC
… COMPUTE HOLONOMY` where the loop is now a closed curve in (Q, θ)
space, parameterized by its projection onto each axis. See §4 and the
companion Halcyon→Gigi letter for the verb's signature.

---

## §3 — Pre-registered falsification criteria (patched)

This section is written before §4. The protocol of §4 is designed to
satisfy the criteria of §3, not the other way around.

### 3.1 The primary observable and its three regimes (patched gates)

The primary observable is `H_geom[γ_unit]`, the **antisymmetric**
component of the holonomy of a unit (Q, θ) loop:

$$
H_{\rm geom}[\gamma_{\rm unit}]
  \;=\; \tfrac{1}{2}\bigl(H[\gamma_{\rm unit}] - H[\gamma_{\rm unit}^{-1}]\bigr).
$$

The symmetric component
`H_sys = ½(H[γ] + H[γ⁻¹])` is reported separately as the
systematic-offset diagnostic; it is not the load-bearing observable
but it must satisfy `|H_sys| < 1σ_H` for the verdict to be valid.

| Outcome | Criterion | Interpretation |
|---|---|---|
| **POSITIVE** | `|H_geom[γ_unit]| > 5σ_H` AND `|H_sys| < 1σ_H` AND each of the six shams returns `|H_sham| < 2σ_sham` AND no consistent-sign pattern across the sham seeds (defined below) | The framework's predicted non-trivial bundle curvature on Λ is detected. |
| **NULL** | `|H_geom[γ_unit]| < 1σ_H` AND `|H_sys| < 1σ_H` AND all shams below their 2σ thresholds | The framework's predicted holonomy is absent at this calibration and substrate. |
| **AMBIGUOUS** | Any of: `1σ ≤ |H_geom| ≤ 5σ`; or `|H_sys| ≥ 1σ_H`; or any sham fails 2σ; or any sham passes 2σ but shows a consistent sign across ≥6/8 seeds | The result is not interpretable; re-run criteria are §3.5. |

σ_H is the Flyvbjerg–Petersen blocked SEM of `H_geom` across the 8
seeds. σ_sham is computed identically per sham. Per-sham thresholds
are *not* corrected for multiplicity (since shams are independent
diagnostics, not a single hypothesis test); the family-wise
discipline is the consistent-sign rule defined in §3.4 instead.

### 3.2 Required sham controls (patched specifications)

A POSITIVE verdict requires all six sham controls to satisfy
`|H_sham| < 2σ_sham` AND the consistent-sign rule of §3.4.

| Sham | What it tests | Patched gate |
|---|---|---|
| **S₁** Loop on a flat field (κ_Q ≡ 0) | Loop isn't an artifact of the ramp + test mass alone | `|H_S₁| < 2σ_S₁` |
| **S₂** Loop with α_Halcyon = 0 | Loop is driven by the framework coupling | `|H_S₂| < 2σ_S₂`, must be machine-ε small in absolute terms |
| **S₃** Loop with μ_baseline scaled (×0.1 and ×10) | **Baseline-subtracted** coupling holonomy is invariant under material scaling | `|H_S₃,baseline-subtracted - H_geom| / |H_geom| < 0.10`. Raw response amplitude is allowed to scale; only the baseline-subtracted holonomy must be invariant. |
| **S₄** Already absorbed into H_geom | The antisymmetric primary observable IS the S₄ test | Built into the H_geom definition; the S₄ gate fires automatically as part of §3.1 |
| **S₅** Degenerate loop (zero area in Λ) | Trivial-loop holonomy is zero | `|H_S₅| < 2σ_S₅` |
| **S₆** Frozen-field loop (U held constant during traversal) | Holonomy requires field transport, not just Q_target updates | `|H_S₆| < 2σ_S₆` |

The S₃ baseline-subtraction protocol is: extract `H` at three values
of μ_baseline (×0.1, ×1, ×10), fit a linear `H(μ_baseline)` model,
the intercept at μ_baseline = 0 is the baseline-subtracted holonomy
and is the quantity that must be invariant under scaling within 10%.
Raw `H(μ_baseline)` may scale.

### 3.3 Stopping rule (patched: calibration-locked)

The framework is declared *falsified by simulation* if all four of
the following are met:

1. v3.1 returns NULL at the preregistered calibration set
   (α_Halcyon ∈ {1, 1000, and the eventual Field-Equations
   derivation if it lands before run time}), with sham controls
   passing.
2. A second independent v3-class measurement design — to be
   specified in a hypothetical SPEC v4 with its own pre-registration —
   *also* returns NULL on the same substrate at the same calibration.
3. The two measurement designs are not trivially equivalent
   (independence verified by an external reviewer per §8.5).
4. **No subsequent re-scaling of α invalidates the prior NULL**
   unless the observable's scaling law in α was itself preregistered
   in this document or in v4's. A later derivation of α may motivate
   a new sensitivity target for a hypothetical v5; it does not erase
   the v3.1 NULL at the calibration it was tested at.

The framework remains *not yet falsified* if v3.1 returns POSITIVE,
or if v3.1 returns AMBIGUOUS and the ambiguity-resolution rerun
criteria of §3.5 are met.

### 3.4 The consistent-sign anti-fishing rule

For each sham, compute the per-seed `H_sham,i` for i = 1…8. The
sham *fails* (regardless of the |mean| gate) if `sign(H_sham,i)` is
the same for at least 6 of 8 seeds AND `|mean H_sham| > 0.5 σ_sham`.
This catches the case where a sham's mean stays under 2σ but its
sign pattern shows a real but small bias — which would imply the
sham is detecting something rather than returning random noise.

### 3.5 Per-seed independence (patched: explicit definition)

A primary or sham gate is "struck" only if at least 5 of 8 seeds
*individually* satisfy:

- Same sign on H (sign-coherent at the 5/8 majority), AND
- Same primary/sham classification (each of the 5 seeds individually
  returns POSITIVE / NULL / AMBIGUOUS in the same bucket).

This is the discipline that prevents a single lucky seed from
dominating the verdict.

### 3.6 Calibration commitments (unchanged from v3.0)

α_Halcyon ∈ {1, 1000, TBD-from-Field-Equations}. The verdict is
reported per calibration. Each α is a separate pre-registered test
and a NULL at one α does not transfer to another by default unless
the observable's α-scaling law was preregistered.

### 3.7 Ambiguity-resolution re-run criteria

If v3.1 returns AMBIGUOUS, a re-run is allowed only under all of
the following:

- The specific ambiguity (which gate failed and by how much) is
  named in the re-run's amendment.
- The re-run uses *more seeds or longer t_segment*, not different
  ω, F₀, T_loop, or α. The protocol parameters are locked.
- The amendment is committed and pushed before the re-run.
- The re-run's verdict is published per §8 regardless of direction.

---

## §4 — Q-ramp protocol via GIGI's HOLONOMY and TRANSPORT verbs (patched)

### 4.1 The loop γ_unit on Λ

The unit loop is a closed curve in (Q, θ) space:

```
γ_unit:  (Q, θ) traces a closed rectangle:
         (Q=0, θ=0) → (Q=2, θ=0) → (Q=2, θ=π) → (Q=0, θ=π) → (Q=0, θ=0)
```

Total path length in Λ is 2·(Q_max + π) = 2·(2 + π) ≈ 10.28 in mixed
units; the *area enclosed* is `Q_max · π = 2π` ≈ 6.28 square units.
The projection onto Q follows `Q_cmd : 0 → 2 → 2 → 0 → 0`, which is
*not* the v3.0 `0 → 1 → 2 → 1 → 0` retracing path. It is a genuine
two-dimensional loop enclosing finite area.

(The substrate may parameterize the loop differently — e.g., circular
in (Q, θ) — without changing the pre-registered observable as long as
the enclosed area in Λ is approximately the same.)

### 4.2 The adiabaticity condition (patched: correct direction)

The required inequalities are:

$$
T_{\rm drive} \;\ll\; T_{\rm segment}, \qquad
\tau_{\rm local\ eq} \;\ll\; T_{\rm segment} \;\ll\; \tau_{\rm unpinned\ drift}
$$

where:
- `T_drive = 2π/ω_drive` is the test-mass lock-in carrier period.
- `T_segment = T_loop / 4` is the duration of each of the four edges
  of γ_unit.
- `τ_local_eq` is the gauge field's local equilibration time at a
  fixed (Q, θ).
- `τ_unpinned_drift` is the gauge field's intrinsic drift timescale
  when not actively pinned (measured from v2 H₈ data: ~10 time units).

The first inequality (`T_drive << T_segment`) ensures the lock-in
demodulates many cycles within each path segment. With ω_drive = 1.0
(period 6.28) and T_segment chosen as 25 time units (loop T_loop = 100),
the ratio is ~4 cycles per segment.

The second inequality (`τ_local_eq << T_segment << τ_unpinned_drift`)
is the one v2 violated. v3.1 satisfies it by **adding an active
Q-pinning potential** to the gauge Hamiltonian:

$$
V_{\rm pin}(U; Q_{\rm target}, \theta_{\rm target}) =
\lambda_{\rm pin}\bigl[(Q_{\rm surrogate}(U) - Q_{\rm target})^2 +
                       w_\theta(\theta_{\rm surrogate}(U) - \theta_{\rm target})^2\bigr].
$$

`λ_pin = 1.0` and `w_θ = 1.0` are pre-registered (not tuneable). The
pinning replaces v2's passive-IC-bias protocol with the apparatus's
actual continuous-cage-drive model. With pinning active, the unpinned
drift timescale is replaced by the pinning equilibration time, which
is set to ~1 time unit, so the segment can be many pinning-times long
and still satisfy the local equilibration constraint.

### 4.3 The tracking-error gate (new in v3.1)

Active pinning could itself become the source of a spurious signal —
if `λ_pin` couples to the test mass in a way that mimics the predicted
holonomy. To prevent this:

$$
\max_t |Q_{\rm surrogate}(t) - Q_{\rm target}(t)| < \epsilon_Q,
\qquad
\max_t |\theta_{\rm surrogate}(t) - \theta_{\rm target}(t)| < \epsilon_\theta.
$$

with pre-registered `ε_Q = 0.05` and `ε_θ = 0.1`. The substrate is
required (per the Halcyon→Gigi letter §4 update) to compute the
tracking error and emit it in the response. A tracking-error violation
forces AMBIGUOUS regardless of the H values.

### 4.4 The GIGI call (patched: loop on Λ, tracking error)

```
SAMPLE_TRANSPORT halcyon_canonical_buckyball
  ALONG_LOOP gamma_unit_in_Q_theta
  CONTROL_MANIFOLD (Q, theta)
  ADIABATIC TRUE
  RAMP_RATE_Q 0.08            // 2 Q-units / T_segment = 2/25
  RAMP_RATE_THETA 0.126        // π / T_segment = π/25
  DRIVE_OMEGA 1.0
  DRIVE_F0 0.01
  N_DISCRETIZATION 4000        // ~10 timesteps per drive cycle * 400 drive cycles
  PIN_LAMBDA_Q 1.0
  PIN_LAMBDA_THETA 1.0
  EPS_Q 0.05                   // tracking-error tolerance
  EPS_THETA 0.10
  ALPHA_HALCYON 1.0            // (also runs at 1000)
  TAU_0 1.0  BETA_TAU 2.0
  MU_BASELINE 1.0  K_SPRING 1.0  C_DAMP 0.1
  SEEDS [20260616..20260623]
  COMPUTE HOLONOMY_FORWARD
  COMPUTE HOLONOMY_REVERSED
  COMPUTE TRACKING_ERROR_TRACE
  COMPUTE ADIABATICITY_CHECK
  RETURN H_forward, H_reversed, sigma_H, per_seed_H,
         tracking_error_max, adiabaticity_check
```

The Python orchestrator constructs `H_geom = ½(H_forward − H_reversed)`
and `H_sys = ½(H_forward + H_reversed)`, then applies the §3 gates.

### 4.5 The integration (patched: real connection, real curvature)

The closed-loop holonomy of the connection A on Λ is the
substrate's responsibility. The Python orchestrator does not
re-implement it. The substrate's HOLONOMY verb computes:

$$
U[\gamma] \;=\; \mathcal{P}\exp\left(\oint_\gamma A\right)
$$

via discretized parallel transport along γ. For the abelianized
test-mass response, this reduces to a path-ordered exponential of a
scalar 1-form, computed via Wilson-loop-style discretization on the
N substeps. The substrate-side correctness contract is the v3.1
verb acceptance battery of §7.4.

The Python side performs only:
1. Loop construction (γ_forward and γ_reversed).
2. Verb call with the parameters of §4.4.
3. Gate evaluation per §3.
4. Sidecar emission per §7.

### 4.6 The orchestrator surface (specification only; no implementation)

```python
def run_holonomy_battery(alpha_halcyon, seeds, log_path):
    """SPEC v3.1 §3 + §4 + §5; pure delegation to GIGI."""
    forward = gigi_sample_transport(loop=GAMMA_UNIT_FORWARD,
                                     alpha_halcyon=alpha_halcyon,
                                     seeds=seeds, ...)
    reversed_ = gigi_sample_transport(loop=GAMMA_UNIT_REVERSED, ...)
    H_geom = 0.5 * (forward.H - reversed_.H)
    H_sys  = 0.5 * (forward.H + reversed_.H)
    shams = {name: gigi_sample_transport(loop=..., sham_flag=name, ...)
             for name in SHAM_FLAGS}
    return apply_v3_1_gates(H_geom, H_sys, shams, forward.tracking_error,
                            forward.adiabaticity_check)
```

No leapfrog. No demodulation. No force computation. All substrate.

---

## §5 — Sham controls (patched specifications)

| Sham | Implementation flag (verb-side) | Predicted output |
|---|---|---|
| S₁ flat field | `SHAM_FLAT_FIELD = true` (substrate forces κ_Q ≡ 0) | `H = 0` |
| S₂ α=0 | `ALPHA_HALCYON = 0` | `H = 0` to machine ε |
| S₃ mass scaled (×0.1 and ×10) | `MU_BASELINE = 0.1, 10.0`; compute baseline-subtracted H via linear fit | baseline-subtracted H invariant within 10% |
| S₄ absorbed into H_geom | (no separate sham — built into the primary observable) | n/a |
| S₅ degenerate loop | `LOOP gamma_degenerate` with zero area in Λ | `H = 0` |
| S₆ frozen field | `SHAM_FROZEN_FIELD = true` (U held; Q_target updated) | `H = 0` |

S₄ is no longer a separate sham; it has been folded into the primary
observable's antisymmetric definition (§3.1). This makes S₄ unfakeable
by construction: any v3.1 simulation that does not also compute
`H[γ_reversed]` is non-compliant.

---

## §6 — What v3.1's results mean for v2 (softened)

### 6.1 POSITIVE v3.1 + NULL v2

v2's null is **consistent with the adiabatic/fixed-point limit** of
the v3.1 holonomy model. The fixed-Q lock-in trivialized the
connection's pullback to a single point in Λ, where the loop encloses
zero area and the holonomy is identically zero regardless of
curvature. v3.1's positive is the first evidence that the bundle has
non-trivial curvature on Λ.

This is *not* a "confirmed prediction" of v2's null in the strict
sense — v2 was not pre-registered against the v3.1 model. It is a
*consistency* check that survives both ways: the v3.1 model also
predicts the v2 limit.

### 6.2 NULL v3.1 + NULL v2

Stopping rule per §3.3 triggers if the conditions are met (independent
v4 measurement design also NULL, external review). v2 is the
adiabatic-limit complementary control; the joint null is stronger
than either alone.

### 6.3 NULL v3.1 + (smoke-mode internal pass on v2)

The v2 internal-extractor signal (|α|/σ = 15.6) is not contradicted
by a v3.1 null. v2's internal extractor confirmed model mechanics;
v3.1 (and v2's external extractor) tests observability. A v3.1 NULL
means the framework's mechanics produce no observable holonomy on Λ,
which is the load-bearing falsification.

### 6.4 POSITIVE v3.1 + (POSITIVE v2 internal)

Strongest positive case: model's internal state and observable
holonomy both match prediction.

### 6.5 AMBIGUOUS v3.1

Per §3.7, re-run is allowed only with named ambiguity, more seeds or
longer T_segment, no protocol parameter changes, amendment committed
before re-run.

---

## §7 — GIGI audit surface (patched: add verb acceptance battery)

### 7.1 Two-layer independent auditability (unchanged)

- Layer 1: substrate computation = GIGI's test suite.
- Layer 2: protocol design = this SPEC.

### 7.2 The receipt model (unchanged)

v3.1 emits `section_12_holonomy_battery_v3_1` JSON sidecar with the
SHA-256 of this SPEC, the GIGI deploy hash, per-seed H values, gate
verdicts.

### 7.3 What this changes for the auditor's job (unchanged)

v3.1 audit reads this SPEC and trusts the substrate's correctness
audit (which is amortized across every chapter using the substrate).

### 7.4 GIGI verb acceptance battery (new in v3.1)

Before the v3.1 simulation calls `SAMPLE_TRANSPORT ALONG_LOOP` for
science, the substrate-side verb must pass the following six
contracts. These are testable by GIGI's `cargo test` suite at the
verb's introduction, before any Halcyon-side integration:

| # | Contract | Test |
|---|---|---|
| GC₁ | **Flat connection returns zero.** | Construct a known-flat connection (A ≡ 0 in synthetic mode); verify H[any loop] = 0 to machine ε across at least 4 loop shapes. |
| GC₂ | **Known area law for an Abelian constant-curvature connection.** | Construct a connection with constant curvature F₀ in (Q, θ); verify H[γ] = F₀ · Area(γ) to 1% across 3 loop sizes. |
| GC₃ | **Reversed loop inverts/sign-flips.** | For an arbitrary connection, verify H[γ⁻¹] = −H[γ] (Abelian) or H[γ]⁻¹ (non-Abelian) to 1% across at least 3 connections. |
| GC₄ | **Zero-size loop returns zero.** | Construct a degenerate loop (any closed curve bounding zero area); verify H = 0 to machine ε. |
| GC₅ | **Discretization convergence.** | Compute H at N_discretization ∈ {1000, 2000, 4000, 8000}; verify monotone convergence with relative change < 1% between 4000 and 8000 substeps. |
| GC₆ | **Gauge invariance.** | Apply a known gauge transformation to the substrate's connection; verify H is invariant to machine ε. |

The substrate must pass GC₁–GC₆ before v3.1 science calls are made.
If any contract fails, v3.1 is gated on the substrate-side patch. The
existing 1373-assertion GIGI test suite is necessary but not
sufficient; GC₁–GC₆ are the *new* contracts the new verb introduces.

---

## §8 — Publication commitment (unchanged from v3.0)

GitHub commit hash of *this* document is the implementation-level
pre-registration timestamp; Zenodo DOI minted from that hash is the
publication-level pre-registration. v3.1's deposit supersedes v3.0's
(which was caught in review before deposit).

All v3.1 outcomes published, regardless of direction. Solves Vol. 4
Appendix A.8 reports the result. The stopping-rule committee of v3.0
§8.5 carries over unchanged.

---

## §9 — What this SPEC does not commit to (unchanged from v3.0)

- No specific α_Halcyon from Davis Field Equations (open work).
- No commitment to the v3.1 result.
- No hardware apparatus design.
- v2 not deprecated.
- No promise that v3.1 will succeed in detecting the predicted
  holonomy.

---

## §10 — Definitions of done (patched)

- [ ] This v3.1 SPEC committed to GitHub.
- [ ] v3.1 SPEC deposited on Zenodo (the v3.0 deposit is abandoned;
  the v3.1 DOI is the canonical pre-registration).
- [ ] GIGI verb (per the patched Halcyon→Gigi letter) lands with
  `ALONG_LOOP`, `CONTROL_MANIFOLD`, tracking-error reporting, and the
  six-sham flag set.
- [ ] GIGI verb passes the GC₁–GC₆ acceptance battery of §7.4.
- [ ] Python orchestrator `run_holonomy_battery.py` is implemented as
  a thin delegation wrapper.
- [ ] v3.1 run produces sidecar matching §7.2.
- [ ] Verdict published per §8.
- [ ] Solves Vol. 4 Appendix A.8 reports the v3.1 result.

---

## §11 — Acknowledgments

v3.0 was written 2026-06-20 immediately after Gigi's methodological
intervention naming pre-registration as the discipline preventing
protocol-shopping. v3.1 was written ~hours later, before v3.0's
Zenodo deposit, after external review (ChatGPT, also same day) caught
two mathematical defects (scalar holonomy vanishing by FTC; adiabaticity
inequality reversed) and five protocol-discipline issues
(Q-as-continuous, S₃ baseline subtraction, S₄ antisymmetry, sham gate
brittleness, calibration-escape in stopping rule, v2-reinterpretation
overclaim, missing verb acceptance battery).

Both reviews count as load-bearing inputs to the pre-registered
protocol. The §0 changelog names every patch and which review
surfaced it. The discipline that makes pre-registration credible
*also* requires that pre-registration be allowed to be patched in
response to substantive review *before* deposit; the same discipline
prohibits patching *after* deposit.

The cost of admission to either credibility — positive or negative —
is the same: two independent measurements, publicly committed in
advance, with an outside reviewer, and a protocol that passes its
own peer review *before* the run.
