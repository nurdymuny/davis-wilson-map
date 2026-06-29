# Halcyon -> GIGI reply, LOOP_TRANSPORT design review (2026-06-21)

**From:** Halcyon team (Bee + Claude)
**To:** GIGI engine team (Bee + Claude)
**Subject:** Reply accepted on the verb shape, the rename, and the two-clocks discipline. Pre-registration reference UPDATED to v3.1.3 at commit `44c70b1` (the v3.0 / `0fe654d` referenced in your reply was an early draft caught by four rounds of pre-deposit technical review). Per-CC answers below.
**Companion to:** `GIGI_TO_HALCYON_2026-06-20_SAMPLE_TRANSPORT_REPLY.md` (hosted on the gigi/ side at `theory/halcyon/`).
**Supersedes (as the operative ask document):** `HALCYON_TO_GIGI_2026-06-20_HOLONOMY_VERB_REQUEST.md` (preserved as historical first contact). The 2026-06-20 letter's substrate ask is reasserted by this letter with the LOOP_TRANSPORT rename and the v3.1.3 commit reference; that letter remains in the chain of custody, this letter is the one the implementation team should read first.

---

## Letter

GIGI —

Your reply reads cleanly. The verb scope, the rename, the generalization framing, the integrator-reuse decision, and the gate-doc-before-code discipline all match how we want to see the substrate evolve. The two-clocks framing in particular — substrate timeline does not move what Halcyon accepts as a result, pre-registration does not move what the substrate ships — is exactly what we want to lock as the load-bearing methodological commitment between the two repos. We are mirroring it from this side.

One commit-hash correction up front before we get to the design questions. You cited the v3 SPEC pre-registration at `0fe654d`. That hash points at the v3.0 first-draft pre-registration which was caught by four rounds of pre-deposit technical review between commit and the planned Zenodo deposit. The actual deposit-ready contract is **v3.1.3 at commit `44c70b1b76501b4b66c6f9ace6bccd8b5bd14c4a`**, pushed today. The four review rounds caught real defects:

- **v3.0 → v3.1** (round 1, commit `7121094`): scalar holonomy `H = ∮ ∂μ/∂Q dQ` vanishes identically by FTC; reversed adiabaticity inequality. v3.1 replaces the scalar with a connection 1-form on a multi-dimensional control manifold and corrects the timing chain.
- **v3.1 → v3.1.1** (round 2, commit `1165d63`): seven executability issues (under-specified second control coordinate, internal timing inconsistency, weak `<<` separation, wrong `N_DISCRETIZATION` comment arithmetic, missing absolute ε for S₂, S₃ NULL-branch division-by-zero, under-defined per-seed σ).
- **v3.1.1 → v3.1.2** (round 3, commit `f4cfa14`): the validity-window blocker — v3.1.1's β_W range `[2.0, 3.0]` traversed below the SU(2) Q-observable's validated regime (β ≥ 2.5 per the inertia_damping JOURNAL); tightened to `[2.5, 3.0]`. Plus three smaller patches.
- **v3.1.2 → v3.1.3** (round 4, commit `44c70b1`): wording / audit-tightness — the substrate-gated `τ_pin` claim, the GC₅ science-value gate, distinction between "pre-deposit technical review" (the GPT rounds) and "external review" (the §8.5 stopping-rule human peer committee).

The differences between v3.0 and v3.1.3 are non-trivial and affect what the substrate must implement. Most importantly:

1. **Second control coordinate is locked to `β_W` (Wilson coupling)**, range `[2.5, 3.0]`, with the loop strictly inside the validated SU(2) regime. v3.0 had no second coordinate (the loop was a 1D backtrack in Q alone, which is the FTC vanishing bug).
2. **The five-sham set after S₄ absorption.** v3.0 had six top-level shams. v3.1.3 has five active shams; S₄ (the reversed-loop test) is folded into the antisymmetric primary observable `H_geom = ½(H[γ] − H[γ⁻¹])`, which makes the orientation-reversal test unfakeable by construction.
3. **Active-pinning regime with tracking-error gates.** v3.1.3 explicitly does not attempt passive adiabaticity; it uses an active Q- and β_W-pinning potential with `λ_pin = 1.0`, and the tracking-error gates `ε_Q = 0.05`, `ε_β_W = 0.05`. A tracking-error violation forces AMBIGUOUS regardless of H values.
4. **The six-contract GIGI verb acceptance battery `GC₁`–`GC₆`** that the substrate must pass before v3.1.3 science calls fire. v3.0 said "trust the 1373 assertions"; v3.1.3 says the 1373 are necessary but not sufficient and names the six new contracts.
5. **ε_abs = 10⁻¹⁰** absolute floor on sham gates (v3.1.3 §3.2 / §5).
6. **The `N_DISCRETIZATION = 10000` science value is gated by GC₅** — the substrate blocks science calls if the 8000→16000 relative change in H exceeds 1%.

All five chain-of-custody drafts (v3.0, v3.1, v3.1.1, v3.1.2, v3.1.3) are preserved in the repository at `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.x.md`. v3.1.3 is the canonical contract; v3.0 is the historical first draft. The substrate should build to v3.1.3.

That commit-hash correction made, your verb scope review and design-question framing stand verbatim. Below: methodological reciprocation (§A), rename acceptance and parameter pack (§B), per-CC answers (§C — six decisions), disambiguations on the three open shape questions (§D), what Halcyon is not asking back (§E), pre-registration reciprocation (§F). Pushback welcome on every clause.

—Bee + Claude

---

## §A — Methodological reciprocation

The two-clocks framing is exactly the discipline we want between the two repos. To make it concrete from Halcyon's side:

- **The v3.1.3 falsification criteria at commit `44c70b1` are the independent referee on the Halcyon side.** Whatever §3 says at that hash is what Halcyon accepts as a result, independent of when the substrate ships LOOP_TRANSPORT, independent of which CC-LT decisions land which way, independent of how long the GC₁–GC₆ acceptance battery takes to pass.
- **The substrate's three-constraint contract is the independent referee on the GIGI side.** Whatever GIGI's t013 contract says (gauge-invariant observable, local per-step updates, no-tunable-tolerance analytical target) is what GIGI builds to, independent of what Halcyon's pre-registered SPEC threshold says.
- The two clocks are locked separately. If GIGI's t013 contract conflicts with a Halcyon-pre-registered threshold, the **conflict is documented**, not papered over: the substrate emits the analytical observable per its own contract; Halcyon's v3.1.3 either gates on it with a pre-registered threshold or reports it as a diagnostic without a threshold. v3.1.x amendments are the mechanism for adding new pre-registered thresholds; substrate amendments are the mechanism for adding new observables. Neither side amends the other's commitments retroactively.

This is the same property the prior letter-pair (the SPEC review rounds and the AURORA v0.1 reply) was building toward. Reciprocated and reinforced here as the operative discipline.

---

## §B — Rename + parameter-pack acceptance

### B.1 LOOP_TRANSPORT rename

Adopted verbatim. The original ask used `SAMPLE_TRANSPORT … ALONG_LOOP` as the verb name because Halcyon was reading from earlier GQL pattern examples without checking the existing namespace. Your point that `src/geometry/sample_transport.rs` already exists as a 696-LOC bundle-side curvature-bounded neighborhood sampler (a completely different semantics) is correct, and overloading would conflate two unrelated computations.

**The Halcyon-side ask document is hereby restated as: `LOOP_TRANSPORT`, a gauge-side peer to `SYMPLECTIC_FLOW`.** The 2026-06-20 letter's `SAMPLE_TRANSPORT … ALONG_LOOP` references are superseded by this rename. We will not protest reading the letter as if it had said `LOOP_TRANSPORT` from the start.

### B.2 ParameterPackKind::Halcyon variant — canonical field list

Yes to `ParameterPackKind::Halcyon` as the first registered variant under a kind-agnostic outer shape. The renaming `alpha_halcyon` → `alpha` inside the variant is correct and we adopt it.

The full canonical field list for the Halcyon variant (matching the v3.1.3 §4.4 GQL parameter set):

```rust
pub enum ParameterPackKind {
    Halcyon {
        // Coupling and substrate model
        alpha: f64,           // formerly alpha_halcyon; the Halcyon coupling calibration
        tau_0: f64,           // tau_Q model: tau_Q(e) = tau_0 / (1 + beta_tau * s_Q(e))
        beta_tau: f64,        // tau_Q model coefficient
        mu_baseline: f64,     // baseline inertial coefficient (the v2.1 H1 material test scaled
                              //   THIS, not mu_eff, after the H1 decoupling patch)

        // Test-mass dynamics (Halcyon-specific; not kind-agnostic — the lock-in
        //   demodulation protocol is a Halcyon-shaped probe, not a substrate primitive)
        K_spring: f64,        // test-mass spring constant
        c_damp: f64,          // test-mass damping coefficient
        drive_omega: f64,     // lock-in carrier angular frequency
        drive_F0: f64,        // lock-in carrier drive amplitude

        // Active-pinning gates (v3.1.3 §4.3, regime B)
        pin_lambda_Q: f64,        // pinning strength on Q
        pin_lambda_beta_W: f64,   // pinning strength on beta_W
        eps_Q: f64,               // tracking-error tolerance on Q
        eps_beta_W: f64,          // tracking-error tolerance on beta_W
    },
    // future: StaggeredFermion { ... },
    // future: ShidokuQuotient { ... },
    // future: AuroraAtmospheric { ... },
}
```

The kind-agnostic outer fields are:

```rust
pub struct LoopTransportConfig {
    pub loop_id: LoopHandle,              // see CC-LT-1
    pub ramp_rate: PerAxisRampRate,        // see §D.4 — per-axis, not scalar
    pub n_substeps: usize,                 // 10000 per v3.1.3 §4.4
    pub seeds: Vec<u64>,                   // ensemble replication — see §D.2
    pub sham: ShamConfig,                  // nested block — see §C.6 / §D.1
    pub parameter_pack: ParameterPackKind, // dispatches per-variant
}
```

Two points where this differs from your §6 sketch:

1. **`drive_omega` is inside the Halcyon variant, not kind-agnostic.** A future `StaggeredFermion` variant won't have a lock-in carrier; the Halcyon test-mass-coupling protocol is the only context where a single carrier ω makes sense. If a future variant does need a carrier (AURORA might), it declares its own. The verb's kind-agnostic surface stays clean.
2. **`ramp_rate` is per-axis, not scalar.** v3.1.3's loop has `RAMP_RATE_Q = 0.04` and `RAMP_RATE_BETA_W = 0.01` — different on each axis because Q ranges over 2.0 units and β_W over 0.5 units in the same `T_segment = 50` time. See §D.4 below.

---

## §C — Per-CC decisions

### CC-LT-1 — Loop declarability: **first-class `DECLARE LOOP`**

Recommendation: **path (a) — first-class declarable loop object with a `LoopRegistry` mirroring `GaugeFieldRegistry`.**

Reasoning:

- **Reuse pattern.** v3.1.3 declares **two** loops:
  - `gamma_unit`: closed rectangle in (Q, β_W) traversing four corners (Q=0, β_W=2.5) → (Q=2, β_W=2.5) → (Q=2, β_W=3.0) → (Q=0, β_W=3.0) → (Q=0, β_W=2.5).
  - `gamma_degenerate`: zero-area loop for sham S₅ (substrate's choice of degenerate-loop shape per the §D.1 disambiguation below).
  - The reverse traversal `gamma_unit^{-1}` is computed by the substrate by traversing `gamma_unit` time-reversed, not by declaring a separate loop. This is what makes the antisymmetric primary observable `H_geom = ½(H[γ] − H[γ^{-1}])` cheap.
- **Call count.** v3.1.3 calls LOOP_TRANSPORT with `gamma_unit` at minimum 8 (seeds) × 2 (forward/reverse) × 2 (α=1, α=1000) = **32 times** per α-calibration set, plus the same multiplier for the 5 sham flag variants → ~160 calls per session. With `gamma_unit` declared once and re-referenced by handle, this reads cleanly in the GQL transcript and in the WAL. With an opaque string handle resolved inline, every call duplicates the loop construction in the GQL stream.
- **Audit story.** The §8.5 stopping-rule reviewers (when they're named) will read the WAL receipt. A named `DECLARE LOOP gamma_unit FROM (Q, beta_wilson) PATH ...` statement is auditable; an opaque-string-handle pattern hides the loop's structure inside the verb's inline cache and forces the auditor to grep for the construction site.

**Adopted: path (a). Mirror the `GaugeFieldRegistry` pattern.**

Counter-suggestion if substrate cost is an issue: ship a minimal `DECLARE LOOP` with just the registry plumbing (handle resolution, WAL emission) and defer the richer loop-construction grammar (e.g., parametric path specs) to a v0.2 of the verb. v3.1.3 only needs two loops; the registry doesn't have to support arbitrary expressivity in v0.1.

### CC-LT-2 — Parameter-pack registry: **adopt as proposed**

`ParameterPackKind::Halcyon { ... }` with kind-agnostic outer fields. Full canonical field list in §B.2 above.

Two clarifications:

- Future variants (StaggeredFermion, ShidokuQuotient, AURORA) plug in as separate enum variants. The parser dispatches on the variant tag. Halcyon does not expect or require the verb to know about other variants.
- The verb signature stays clean of Halcyon-domain identifiers. `alpha_halcyon` becomes `alpha`; `K_spring` stays `K_spring` (it's not Halcyon-specific terminology, it's the test-mass spring constant); `mu_baseline` stays the same (general inertial-baseline concept). No other variant has to inherit these names.

If a future letter ever describes LOOP_TRANSPORT as "the Halcyon verb," we agree with your §7 framing that that's drift and welcome pushback the same way you would push back on a parser bug.

### CC-LT-3 — Adiabaticity threshold: **substrate emits the rate; Halcyon's v3.1.3 does NOT pre-register a threshold for this observable (yet)**

This is the question with the most nuance. Halcyon's reply has three parts.

**(i) Two distinct adiabaticity observables, not one.** Your §3 letter and Halcyon's 2026-06-20 letter conflated two separate adiabaticity checks. v3.1.3 §4.2 distinguishes them:

- **Observable A — pinning equilibration ratio: `τ_pin / T_segment`.** This is the active-pinning regime check. The substrate is required to verify that the pinning potential's equilibration timescale `τ_pin` is much less than the per-segment duration `T_segment`. This is the gate v3.1.3 §4.2 pre-registers: **AMBIGUOUS fires if `τ_pin / T_segment ≥ 0.1`.** This is Halcyon's pre-registered threshold for the active-pinning sub-question.
- **Observable B — gauge-relaxation ratio: `ramp_rate / gauge_relaxation_rate`.** This is the broader t013 three-constraint diagnostic. v3.1.3 does **not** pre-register a numerical threshold for this observable. The substrate emits the ratio; Halcyon's sidecar reports it as a diagnostic for the stopping-rule committee's review; v3.1.3 §3 does not currently force AMBIGUOUS on it.

The two observables are different physics. Observable A measures whether the active control catches up with the parameter drive; Observable B measures whether the substrate's intrinsic equilibration outpaces the parameter drive. They can both pass, both fail, or one pass and the other fail. The substrate emits both.

**(ii) For Observable B, the analytical formula must come from theory.** Halcyon does not have a theoretical preference between your three candidates and is happy to defer to the substrate's t013 three-constraint contract. With that said: candidate (a), `||dU/dt||_op = g² · sup_edges ||E_edge||`, is the cheapest meaningful proxy and reads what the integrator already computes at every substep. We would not push back on that as the v0.1 formula. If the substrate later wants to upgrade to candidate (b) (linearized-curvature eigenvalue) when a future use case needs the tighter bound, that's a follow-up improvement that does not affect v3.1.3's gates (because v3.1.3 doesn't gate on Observable B).

**(iii) For Observable A, Halcyon's `0.1` threshold is pre-registered, not tunable.** Your §3 question about whether the threshold is `κ = 1` (strict) or `κ < 1` (safety factor) is the question that triggers the three-constraint conflict. Halcyon's stance:

- `0.1` is a pre-registered numerical choice on the Halcyon side. It is not "tunable" in the sense that an operator can vary it at runtime — it is locked in v3.1.3 §4.2 at commit `44c70b1`, and any change would require a v3.1.4 amendment with its own pre-registration.
- From the substrate's perspective, the threshold is applied **outside** the substrate. The substrate emits `τ_pin / T_segment` as an `f64`; Halcyon's Python orchestrator applies the comparison `>= 0.1` against the substrate's emitted value. The substrate itself has no tunable threshold baked into its code.
- This keeps the substrate's t013 three-constraint contract clean: the substrate's output is the analytical observable, no operator parameter inside it. Halcyon's gate is layered on top.

If you prefer a stricter framing — substrate emits the observable; Halcyon's threshold lives in the SPEC, not in either codebase — that's the same thing. Either way, no `κ` knob inside the substrate.

**Open question for the GIGI side:** what `adiabaticity_warnings_count` mechanism do you want? Halcyon's preference is that the substrate emit warnings to the WAL when **its own** internal sanity checks fire (e.g., negative `tau_pin`, NaN propagation, divergent norms) — these are substrate-correctness signals, not Halcyon-protocol gates. Halcyon does not need (or want) the substrate to fire warnings against the v3.1.3 `0.1` threshold — Halcyon's Python applies that threshold from the SPEC, not from substrate state. The WAL warnings are useful as substrate-side instrumentation; they should not be conflated with Halcyon's pre-registered gates.

### CC-LT-4 — Integrator reuse vs duplication: **duplicate for v0.1, agreed**

Adopted as you proposed. Sprint B revert lesson plus the IV.6 gold-gate test being bit-identity-locked against the current `symplectic_flow.rs:294-330` body = duplicate the per-substep KDK orchestration into `loop_transport.rs`. Extract the shared helper as a follow-up commit gated by a third consumer (e.g., StaggeredFermion variant) materializing.

The Halcyon-side mirror of this discipline: **do not** touch `symplectic_flow.rs` on the LOOP_TRANSPORT introduction commit. If the IV.6 hot path needs surgery later, that surgery is its own commit, separate from any LOOP_TRANSPORT change.

### CC-LT-5 — Name collision: **resolution (a), LOOP_TRANSPORT**

Already adopted in §B.1. No further pushback.

### CC-LT-6 — Sham-flag API shape: **nested `SHAM { ... }` block**

Adopted. v3.1.3's five active shams (S₄ is absorbed into the antisymmetric primary observable, so it has no flag) map to:

```
SHAM {
    flat_field:      bool,                    // S1
    alpha_zero:      bool,                    // S2
    mass_scale:      Option<f64>,             // S3 — set to 0.1, 1.0, or 10.0 across runs
    degenerate_loop: bool,                    // S5 (substrate's choice of degenerate shape — see §D.1)
    frozen_field:    bool,                    // S6
}
```

Default for every flag is "off" / `None`. Setting more than one at a time is not used in v3.1.3 (each sham is run independently), but is not prohibited by the grammar.

Rationale agrees with yours: nested is more grammar-stable for adding future flags. Halcyon does not anticipate adding shams beyond v3.1.3's set, but if a v3.1.x amendment ever introduces one, the nested block absorbs it cleanly. The 5 shams + S₄ absorbed into the antisymmetric primary is the v3.1.3 commitment.

---

## §D — Disambiguations on the three open shape questions

### D.1 SHAM_DEGENERATE_LOOP disambiguation

v3.1.3 §5's S₅ specification is: "degenerate loop (zero area in Λ)". The substrate should return identity holonomy for any closed path that encloses zero area in the (Q, β_W) control manifold.

Of your three readings:

1. **Out-and-back** — a path that traverses a segment and immediately retraces. Encloses zero area in Λ. ✅ Maps to v3.1.3 S₅.
2. **Zero-length** — a path with no segments (or a single-point "loop"). Also encloses zero area. ✅ Equivalent to S₅ in the limit.
3. **Non-closing** — last vertex doesn't match first. Parser rejection at declaration time. ❌ Not v3.1.3 S₅.

**Recommendation, adopting your split-into-three suggestion:**

- The substrate ships all three as **separate flags** (`SHAM_BACKTRACK_LOOP`, `SHAM_EMPTY_LOOP`, `SHAM_OPEN_LOOP`) under the nested `SHAM { ... }` block. Each tests a different substrate invariant and the three flags are the more honest shape per your §2 framing.
- **Halcyon's v3.1.3 S₅ maps to `SHAM_BACKTRACK_LOOP`.** That's the canonical mapping for the pre-registered protocol — a backtracking path in (Q, β_W) that the substrate cannot distinguish from a non-backtracking path at the integrator level. The S₅ test asserts: substrate returns identity, no spurious gauge-field contribution from the path-construction code.
- `SHAM_EMPTY_LOOP` is a stronger sanity check (the no-substep edge case) and is useful for the GC₄ verb acceptance contract. v3.1.3 does not require it as a science-gate sham; the substrate ships it for the audit story.
- `SHAM_OPEN_LOOP` is a parser-rejection test, useful for the verb's input-validation correctness but not a runtime-fired sham. v3.1.3 does not require it; the substrate ships it as a parser-correctness test.

So: v3.1.3's required sham set is `{flat_field, alpha_zero, mass_scale, BACKTRACK_LOOP (was S5), frozen_field}`. The substrate may ship additional substrate-internal flags (`EMPTY_LOOP`, `OPEN_LOOP`) for its own audit story; Halcyon's pre-registered gates do not depend on them.

### D.2 Seed shape: ensemble replication (reading 1)

**Confirmed: ensemble replication.** v3.1.3 §3.5 pre-registers 8 independent trajectories with the Flyvbjerg–Petersen blocked SEM `σ_H_blocked` computed across the 8-seed ensemble. Per-seed sign-coherence is the discriminator on the POSITIVE branch (≥ 5/8 same sign required); NULL has no sign-coherence requirement.

The substrate runs `len(seeds) = 8` independent trajectories per LOOP_TRANSPORT call. The trajectories are parallelizable across cores; the per-seed RNG threading inherits from your `gibbs_sample.rs:226` seed-from-u64 pattern. Aggregation is done in the substrate's return (per-seed holonomies + blocked-SEM summary stats).

**The bit-identity reproducibility gate:**

> Same `seeds` vec + same config (parameter pack, loop, sham, n_substeps) → byte-identical `per_seed_holonomy` GroupElements (component-by-component f64) AND byte-identical `measurement_history` chains.

This is the mirror of the IV.6 gate. Lock it in `theory/halcyon/HALCYON_PART_VI_GATES.md` as you suggested.

### D.3 Test placement: three-assertion gate suite minimal, physics regression separate

**Confirmed:** the three-assertion gate suite (`tests/halcyon_part_vi_loop_transport.rs`) stays minimal at your proposed shape:

- (a) Six (or eight, after the SHAM split) tests asserting each sham flag returns identity holonomy on a known-non-trivial field.
- (b) One test asserting trivial-bundle (identity-everywhere) holonomy = identity on the buckyball face loops.
- (c) One test asserting known-non-trivial holonomy recovers `walk_loop`'s answer when the parameter loop is trivial (no ramping).

Total: ~300–400 LOC, runs in `cargo test --features halcyon`.

The strong-coupling Wilson-loop physics regression is **separate** as you proposed, behind `#[ignore = "physics regression, runs in nightly CI"]`. Halcyon's v3.1.3 SPEC pre-registers GC₂ (Abelian area law to 1% across 3 loop sizes) as the substrate-correctness contract for non-trivial holonomy; the physics regression test is a stronger statement intended for nightly CI, not the gate suite that gates v3.1.3 science calls.

### D.4 Outer-loop shape: continuous, per-axis ramp rate

**Confirmed: continuous parameter evolution.** Per-substep, the substrate updates `(Q_target, β_W_target)` along the piecewise-linear path. KDK substep then drives the gauge field with the pinning potential anchored to the updated targets.

**Per-axis ramp rate, not scalar.** v3.1.3's loop has four segments:

1. (Q=0, β_W=2.5) → (Q=2, β_W=2.5): Q ramps at `0.04 = 2.0 / 50`, β_W held.
2. (Q=2, β_W=2.5) → (Q=2, β_W=3.0): β_W ramps at `0.01 = 0.5 / 50`, Q held.
3. (Q=2, β_W=3.0) → (Q=0, β_W=3.0): Q ramps at `−0.04`, β_W held.
4. (Q=0, β_W=3.0) → (Q=0, β_W=2.5): β_W ramps at `−0.01`, Q held.

Each segment is `T_segment = 50` time units, so `T_loop = 200`. The substrate's executor enumerates segments and applies the appropriate per-axis rate per segment. The `RampRate` config field should carry both axis values; the substrate's per-substep update reads "which segment are we in" and applies the corresponding `(dQ/dt, dβ_W/dt)`.

A clean shape for this:

```rust
pub enum LoopShape {
    PiecewiseLinear {
        vertices: Vec<(f64, f64)>,  // (Q, beta_W) corners in traversal order
        t_per_segment: f64,         // 50.0 for v3.1.3
    },
    // future: Circular, BezierClosed, etc.
}
```

The PiecewiseLinear variant is sufficient for v3.1.3. Future loop shapes (circles, smooth curves) can plug in later if a different protocol needs them.

---

## §E — What Halcyon is not asking back (mirroring your §8)

Per your discipline, reciprocated from the Halcyon side:

- **No turnaround date.** The verb ships when the design questions in this letter and your §6 settle. Halcyon is not committing to a week or a milestone for the GIGI implementation; we operate on the substrate's clock.
- **No v3.1.3 outcome guarantee.** Halcyon does not predict whether v3.1.3 will return POSITIVE, NULL, or AMBIGUOUS. The pre-registration commits to publishing whatever result comes out, regardless of direction.
- **No guarantee the verb will pass GC₁–GC₆.** Halcyon will run the acceptance battery; whether the substrate passes is what GIGI builds to. If a contract fails, v3.1.3 is gated on the substrate-side patch — Halcyon does not negotiate the contract down to "almost passes."
- **No guarantee Halcyon's Python orchestrator (`run_holonomy_battery.py`) is ready when GIGI's verb is ready.** They are two independent implementation clocks. Halcyon's Python orchestrator is a thin delegation wrapper (per v3.1.3 §4.6) and ships when the verb is callable.

What Halcyon **is** committing to:

- **v3.1.3 §3 falsification criteria stay locked at commit `44c70b1`.** They will not move to fit the verb's eventual behavior. If the verb's output reveals an unanticipated need for a new threshold or a new observable, that triggers a v3.1.4 pre-registration with its own commit hash and (after the Zenodo deposit fires) its own DOI. Pre-registration commitments are not amendable after deposit.
- **The Halcyon-side Python orchestrator will be a thin delegation wrapper.** No substrate-relevant logic on the Halcyon side beyond gate application and sidecar emission. If the substrate's emission contract changes, the Python wrapper changes. The substrate is the source of truth for the holonomy computation.
- **The §8.5 stopping-rule committee is the human peer review locus.** We will not substitute "GIGI's pre-deposit technical review" for human peer review of measurement-design independence. The two are different things and v3.1.3 keeps them distinct.
- **The cross-cutting questions in §C and §D are decisions, not announcements.** We want your read before the verb's parser arm or executor body lands; if you push back on any clause, the question reopens and gets re-pinned before code.

---

## §F — Pre-registration reciprocation

Commit `44c70b1b76501b4b66c6f9ace6bccd8b5bd14c4a` (Halcyon SPEC v3.1.3) is the independent referee on the Halcyon side. Whatever v3.1.3 §3 says at that hash is what Halcyon accepts as a result — independent of when the substrate ships LOOP_TRANSPORT, independent of the verb's parser surface, independent of which of the §C and §D decisions land which way.

The substrate timeline does not move the pre-registration. The pre-registration does not move the substrate timeline either. They are two clocks, locked separately. That is the methodological commitment we want to reciprocate, and it is the reason every design question in §C and §D is "pin this before LOC lands" rather than "we will iterate on this in the implementation log." Pre-registered work on this side earns up-front design discipline on yours; substrate-side pre-deposit technical review of v3.0 → v3.1.3 earns the same on ours.

Two informational points for the GIGI implementation team:

- The Zenodo deposit of v3.1.3 (with v3.0, v3.1, v3.1.1, v3.1.2 attached for chain-of-custody transparency) is the next pending Halcyon-side action. When the DOI mints, the v3.1.3 SPEC file gets a single-line `Zenodo DOI:` pointer added at the top via a post-deposit commit. That post-deposit commit is not part of the pre-registration — it merely records the DOI.
- The §8.5 stopping-rule committee (Gigi + one external lattice-gauge-theory peer reviewer + one journal peer reviewer) assembles only when the second NULL is recorded, not in advance. If the v3.1.3 verdict comes back POSITIVE or AMBIGUOUS, the committee is not invoked. If it comes back NULL and a hypothetical v4 measurement design also returns NULL, the committee verifies that the two designs are not trivially equivalent.

---

## §G — WAL revert transparency acknowledgment

Acknowledged on the substrate parallel-WAL-replay regression and the `claude_substrate_v0` bundle loss. Thank you for surfacing it; nothing on Halcyon's side to disclose in reciprocation (no parallel incident). The bit-identity contracts on IV.10 / III.8b / V.* being unaffected because gauge primitives go through a separate code path is the result we'd want; the Part IV gold-gate re-run at HEAD passing byte-identically is the corroboration we'd want.

The discipline this surfaces — "verb introduction commits do not touch existing hot paths; LOC that touches the IV.6-gated body gets its own commit, separate from LOOP_TRANSPORT introduction" — is the right mirror of the Sprint B revert lesson and we'll respect it from this side as well. The Halcyon Python orchestrator changes (when they happen) will similarly be in their own commits, separate from substrate-side LOOP_TRANSPORT changes.

---

## §H — Closing

The verb scope, the rename, the generalization framing, and the discipline are all aligned. The CC-LT decisions in §C and the disambiguations in §D are Halcyon's read of your design questions. Pushback welcome on every clause; if you push back, the question reopens and gets re-pinned before code.

Two-clocks discipline reciprocated. Pre-registration commit `44c70b1` is the independent referee on the Halcyon side. v3.1.3 is the contract; v3.0 was an early draft caught by four review rounds and is preserved for the chain of custody, not the spec the substrate should build to.

When the gate doc lands at `theory/halcyon/HALCYON_PART_VI_GATES.md`, ping us and we'll read it against v3.1.3 §7.4's GC₁–GC₆ to confirm the mapping. When the parser arm lands, ping us and we'll read it against the §C / §D decisions. When the verb is callable, the Halcyon Python orchestrator implementation begins on the Halcyon side.

—Bee + Claude

---

## Companion-letter index (for the GIGI implementation team)

If you're reading just this reply and the corresponding GIGI letter, here's the minimum-set of Halcyon-side documents the substrate implementation should know about:

| Document | Path | Role |
|---|---|---|
| **v3.1.3 SPEC** (canonical) | `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.3.md` | The pre-registered protocol. §3 falsification criteria are load-bearing. §7.4 GC₁–GC₆ is the verb acceptance battery. |
| v3.1.2 SPEC | `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.2.md` | Fourth-draft, superseded by v3.1.3 after round-4 pre-deposit technical review. Preserved for chain of custody. |
| v3.1.1 SPEC | `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.1.md` | Third-draft, superseded by v3.1.2. |
| v3.1 SPEC | `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.md` | Second-draft, superseded by v3.1.1. |
| v3.0 SPEC | `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md` | First-draft (the one cited in your reply as `0fe654d`). Superseded by v3.1. |
| v2.1 SPEC | `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC.md` | The fixed-Q lock-in predecessor protocol. Not deprecated; it is the adiabatic-limit complementary control case for v3.1.3. |
| Original ask | `inertia_damping/HALCYON_TO_GIGI_2026-06-20_HOLONOMY_VERB_REQUEST.md` | The 2026-06-20 first contact letter. Superseded by this reply as the operative ask document; preserved for the chain of custody. |
| **This reply** | `inertia_damping/HALCYON_TO_GIGI_2026-06-21_LOOP_TRANSPORT_REPLY.md` | The operative ask document. Implementation team should read this first. |
| Zenodo metadata | `inertia_damping/HALCYON_SPEC_v3_ZENODO_METADATA.md` | Paste-ready Zenodo deposit form values for v3.1.3 + four predecessor drafts. |
