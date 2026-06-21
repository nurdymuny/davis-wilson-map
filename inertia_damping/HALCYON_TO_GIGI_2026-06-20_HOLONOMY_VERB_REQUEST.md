# Halcyon → Gigi — Holonomy verb for the v3.1 falsification battery

**Date:** 2026-06-20 (amended same day to v3.1 after external review)
**Pattern:** same as the `--use-gigi` flag spec and the Part V SNAPSHOT
gates — Halcyon writes the substrate request, Gigi designs the engine-
side implementation, the two coordinate via this letter pattern.
**Authoritative SPEC:** `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.1.md`
(the v3.0 and v3.1 drafts were caught in external review before Zenodo
deposit; v3.1.1 is the patched contract that goes to deposit). Read
§2.2, §4.4, and §7.4 of v3.1.1 first; this letter is the
implementation request that falls out of those sections.

**Key v3.1.1 patches that change this letter from its earlier drafts:**
- The loop lives on a **multi-dimensional control manifold Λ = (Q, β_W)**,
  not on Q alone. A 1D Q-only path encloses zero area and trivially
  returns zero holonomy by FTC; v3.1 needed a 2D loop and v3.1.1
  locks the second coordinate to the **Wilson gauge coupling β_W**
  (same β that GIGI's existing `gauge_field` declarations carry —
  no new conceptual introduction). Range: β_W ∈ [2.0, 3.0] with
  the Migdal–Witten canonical 2.5 at the midpoint.
- The substrate must compute a **real connection 1-form `A` on Λ**, not
  the scalar derivative `∂μ/∂Q` (which is exact and vanishes).
- The substrate must compute `H_forward` AND `H_reversed` so the
  Python side can form `H_geom = ½(H_fwd − H_rev)` (the geometric
  observable) and `H_sys = ½(H_fwd + H_rev)` (the systematic-offset
  diagnostic).
- The substrate must emit a **tracking-error trace per axis** (Q and
  β_W independently) so that active pinning cannot become a hidden
  signal source.
- The substrate must pass a six-contract acceptance battery
  (`GC₁`–`GC₆` in v3.1.1 §7) before Halcyon calls it for science.

**Disambiguation up front:** `β_W` is the **Wilson gauge coupling**
appearing in `S_W = (β_W / N) Σ_f [N − Re Tr U_f]`. `BETA_TAU` is the
v2.1 τ_Q model's coupling coefficient appearing in
`τ_Q(e) = τ₀ / (1 + β_τ s_Q(e))`. They are different parameters and
must not be confused. The `SAMPLE_TRANSPORT` call carries both because
the τ_Q model uses `BETA_TAU` (held fixed at 2.0) while the loop
**traverses** values of `BETA_WILSON` (varying along the loop's
second axis).

## TL;DR

The v3 falsification battery needs a substrate-level holonomy
computation along a programmed Q-loop. Specifically: a new variant of
`SAMPLE_TRANSPORT` with an `ALONG_LOOP` clause that drives Q through a
closed path on the buckyball substrate while the substrate computes the
holonomy of the pulled-back inertial-coupling connection. The Python
side stays thin — it constructs the loop, sets the seeds, parses the
result, and applies the §3 gate criteria. The substrate does the math.

The audit story v3 commits to (§7 of the SPEC): the substrate's
computational correctness lives in your 1373-assertion test suite, the
protocol's design correctness lives in the v3 SPEC, and the two layers
are independently reviewable. The Python orchestrator becomes ~200
lines of glue code instead of v2's ~750 lines of inline integrator,
because the heavy lifting is yours.

## The verb, written down (v3.1 shape)

```
SAMPLE_TRANSPORT <gauge_field_name>
  ALONG_LOOP <loop_id>
  CONTROL_MANIFOLD (Q, beta_wilson)        // v3.1: must be >= 2D
  ADIABATIC TRUE
  RAMP_RATE_Q <float>                // |dQ/dt| in Q-units per sim time unit
  RAMP_RATE_BETA_W <float>            // |d(beta_W)/dt| per sim time unit
  DRIVE_OMEGA <float>                 // lock-in carrier frequency
  DRIVE_F0 <float>                    // test-mass drive amplitude
  N_DISCRETIZATION <int>              // substeps along the loop
  PIN_LAMBDA_Q <float>                // soft-pin strength on Q
  PIN_LAMBDA_BETA_W <float>           // soft-pin strength on beta_W
  EPS_Q <float>                       // tracking-error tolerance on Q
  EPS_BETA_W <float>                  // tracking-error tolerance on beta_W
  ALPHA_HALCYON <float>               // coupling calibration
  TAU_0 <float> BETA_TAU <float>       // tau_Q model parameters
  MU_BASELINE <float> K_SPRING <float> C_DAMP <float>  // test-mass parameters
  SEEDS <int_list>
  // v3.1: substrate computes connection 1-form A on (Q, theta), holonomy
  // is its closed-loop transport. Substrate must compute BOTH directions
  // so the Python side can form the antisymmetric primary observable.
  COMPUTE HOLONOMY_FORWARD           // primary: traverse loop in nominal direction
  COMPUTE HOLONOMY_REVERSED           // companion: traverse same loop reversed
  COMPUTE TRACKING_ERROR_TRACE_Q       // per-substep |Q_surrogate - Q_target|
  COMPUTE TRACKING_ERROR_TRACE_BETA_W  // per-substep |beta_W_surrogate - beta_W_target|
  COMPUTE ADIABATICITY_CHECK          // T_drive vs T_segment vs tau_relax check
  RETURN H_forward, H_reversed,
         sigma_H_blocked,
         per_seed_H_forward, per_seed_H_reversed,
         tracking_error_max_Q, tracking_error_max_beta_W,
         adiabaticity_check
```

**Concrete v3.1.1 numerical values** (from SPEC v3.1.1 §4.4, locked
at the pre-registration commit hash):

| parameter | value |
|---|---|
| `CONTROL_MANIFOLD` | `(Q, beta_wilson)` |
| Q range | `[0.0, 2.0]` |
| β_W range | `[2.0, 3.0]` |
| T_loop | 200.0 |
| T_segment | 50.0 |
| `RAMP_RATE_Q` | 0.04 |
| `RAMP_RATE_BETA_W` | 0.02 |
| `DRIVE_OMEGA` | 1.0 |
| `DRIVE_F0` | 0.01 |
| `N_DISCRETIZATION` | 10000 (dt = 0.02 over T_loop = 200) |
| `PIN_LAMBDA_Q` | 1.0 |
| `PIN_LAMBDA_BETA_W` | 1.0 |
| `EPS_Q` | 0.05 |
| `EPS_BETA_W` | 0.05 |
| `TAU_0` / `BETA_TAU` (τ_Q model) | 1.0 / 2.0 |
| `MU_BASELINE` / `K_SPRING` / `C_DAMP` | 1.0 / 1.0 / 0.1 |
| `ALPHA_HALCYON` | 1.0 and 1000.0 (two pre-registered calibrations) |
| seeds | `[20260616 .. 20260623]` (8 seeds) |

The Python orchestrator then constructs:

```python
H_geom = 0.5 * (H_forward - H_reversed)
H_sys  = 0.5 * (H_forward + H_reversed)
```

and applies the v3.1 §3 gates. The substrate does not need to know
about `H_geom`/`H_sys` — it just returns both directions and the
Python side combines.

Loop specs are first-class objects, declared earlier in the GQL block:

```
LOOP gamma_unit:
  // v3.1.1: closed rectangle on (Q, beta_W), encloses area 2*1 = 2
  CONTROL_MANIFOLD (Q, beta_wilson)
  PATH:  (Q=0.0, beta_W=2.0)
      -> (Q=2.0, beta_W=2.0)
      -> (Q=2.0, beta_W=3.0)
      -> (Q=0.0, beta_W=3.0)
      -> (Q=0.0, beta_W=2.0)
  T_LOOP 200.0
  SEGMENTS PIECEWISE_LINEAR

LOOP gamma_degenerate:
  // zero-area loop for sham S_5
  CONTROL_MANIFOLD (Q, beta_wilson)
  PATH: (Q=0.0, beta_W=2.5) -> (Q=0.0, beta_W=2.5)
  T_LOOP 200.0
```

(β_W = 2.5 for the degenerate loop is the Migdal–Witten canonical
operating point; any single point in Λ would do.)

The reversed loop is generated substrate-side by traversing
`gamma_unit` time-reversed; the Python side does not need to declare
`gamma_reversed` separately because v3.1's primary observable is
already antisymmetric.

If a different second axis is more natural for the substrate (a
β-coupling knob, a drive-phase parameter, a substrate-side
gauge-rotation parameter), that's fine — the constraint is dim(Λ) ≥ 2
and that loops enclose finite area. Name the chosen second axis in
your reply.

## The six sham controls v3 requires

Per the SPEC v3 §5, the verb needs to support the following six sham
modes. Each is a flag on the existing `SAMPLE_TRANSPORT` call, *not* a
new verb. Specifically:

1. **`SHAM_FLAT_FIELD TRUE`** — substrate forces `κ_Q(e) ≡ 0` on all
   edges, all times. Predicted output: `H = 0`. Use: distinguishes
   genuine bundle curvature from ramp-induced artefact.
2. **`SHAM_ALPHA_ZERO TRUE`** — substrate sets `α_Halcyon = 0` for the
   inertial coupling computation while everything else runs normally.
   Predicted output: `H = 0` to machine precision.
3. **`SHAM_MASS_SCALED <float>`** — substrate replaces `μ_baseline →
   μ_baseline · <factor>`. Used at factors 0.1 and 10.0. Predicted
   output: `H` invariant under the rescaling (gauge invariance).
4. **`SHAM_REVERSED_LOOP TRUE`** (or equivalently, `LOOP gamma_reversed`)
   — substrate traverses the loop time-reversed. Predicted output:
   `H_reversed = −H` for an Abelian holonomy (the test mass's
   abelianized response).
5. **`SHAM_DEGENERATE_LOOP TRUE`** (or `LOOP gamma_degenerate`) —
   substrate holds Q constant at Q=0 for the full T_LOOP. Predicted
   output: `H = 0`.
6. **`SHAM_FROZEN_FIELD TRUE`** — substrate freezes `U(t)` while
   updating `Q_target(t)`. Predicted output: `H = 0` (no transport
   without field evolution).

These six are gate-checks: the v3 verdict is only valid if all six
shams return `< 1 σ_H`. Any sham failing to return zero invalidates the
primary measurement regardless of its value.

## The adiabaticity check

The substrate is in a better position than the Python wrapper to detect
when the loop's ramp rate violates the adiabatic-transport condition.
Specifically: if at any substep the gauge field's instantaneous
relaxation rate exceeds the ramp rate by more than some factor (say 3×),
the substrate should emit a warning in the response:

```
"adiabaticity_check": {
  "passed": false,
  "violation_substep": 47,
  "ramp_rate": 0.02,
  "instantaneous_relaxation_rate": 0.083,
  "ratio": 4.15
}
```

This is the substrate diagnosing its own measurement-condition
violation. v3's gates (§3) treat `adiabaticity_check.passed == false`
as an AMBIGUOUS verdict regardless of the H value.

## What the Python orchestrator does (so you know where the seam is)

The `run_holonomy_battery.py` wrapper (to be written *after* the
substrate verb lands, not before) does this and only this:

```python
def run_holonomy_battery(alpha_halcyon, seeds, log_path):
    """SPEC v3 §4 + §5; calls SAMPLE_TRANSPORT for the primary loop
    and each sham, applies the §3 gates to the substrate's H values."""
    primary = gigi_call_sample_transport(
        loop="gamma_unit", alpha_halcyon=alpha_halcyon, seeds=seeds, ...)
    shams = {name: gigi_call_sample_transport(
                loop="gamma_unit" if name not in LOOP_OVERRIDES else LOOP_OVERRIDES[name],
                sham_flag=name, ...)
             for name in ["flat_field", "alpha_zero", "mass_scaled_10x",
                          "mass_scaled_0p1x", "reversed_loop",
                          "degenerate_loop", "frozen_field"]}
    return apply_v3_gates(primary, shams)
```

No integrator, no leapfrog, no demodulation, no force computation. All
of that is yours.

## The wider audit story v3 commits to

Per SPEC v3 §7, v3's audit surface is the inverse of v2's. v2's audit
trail was: read 750 lines of Python in `falsification_battery.py`, read
400 lines in `test_mass_dynamics.py`, walk through the leapfrog and the
demod and the per-Q-fit logic, decide if it's right. That's the kind
of audit no one actually does well.

v3's audit trail is: read 500 lines of the SPEC and decide if the
*protocol* is right; trust the substrate's correctness because your
test suite is already in place. The protocol audit is a methodology
review, not a code walkthrough. The substrate audit is the GIGI test
suite, amortized across every Halcyon chapter that uses the substrate.

This is the architectural inversion you flagged in the post-Sprint-A
letter — re-run the GIGI query rather than read the Python — and v3 is
the first place where it shows up as a Halcyon SPEC requirement, not
just an aspiration. If the verb lands, the audit story is clean.

## What I'm asking for

In rough sprint-sized chunks:

1. **The verb itself** — `SAMPLE_TRANSPORT … ALONG_LOOP …
   CONTROL_MANIFOLD … ADIABATIC` landing on `gigi-stream.fly.dev`
   and the local build with the v3.1 parameter surface above. The
   substrate computes the connection 1-form A on the 2D control
   manifold, then the closed-loop holonomy via discretized parallel
   transport. **Both `HOLONOMY_FORWARD` and `HOLONOMY_REVERSED`
   must be computed** so the Python side can form the antisymmetric
   primary observable.
2. **The five sham flags** (S₄ is absorbed into the antisymmetric
   primary observable, not a separate flag):
   - `SHAM_FLAT_FIELD` — substrate forces κ_Q ≡ 0
   - `SHAM_ALPHA_ZERO` — substrate sets α_Halcyon = 0
   - `SHAM_MASS_SCALED <float>` — substrate scales μ_baseline (called
     three times at 0.1, 1.0, 10.0 for the baseline-subtraction fit)
   - `SHAM_DEGENERATE_LOOP` — substrate uses the degenerate loop
   - `SHAM_FROZEN_FIELD` — substrate freezes U(t) while updating
     (Q_target, θ_target)
3. **Tracking-error reporting** — substrate-side computation of
   `max_t |Q_surrogate(t) − Q_target(t)|` and the same for θ, emitted
   in the response. v3.1 needs this because active Q-pinning could
   otherwise become a hidden signal source. v3.1 §4.3 makes
   tracking-error violation force AMBIGUOUS regardless of the H
   values.
4. **The adiabaticity self-check** — substrate-side computation of
   the instantaneous gauge-field local-equilibration rate vs the
   segment timescale, with a warning emitted when the inequality
   chain of v3.1 §4.2 is violated.
5. **Per-seed independent computation** — confirming the seed model
   propagates per-seed independence through the loop transport on Λ.
6. **The six-contract verb acceptance battery (`GC₁`–`GC₆`)** from
   v3.1 §7.4, as a substrate-side `cargo test --features halcyon`
   regression suite *that must pass before Halcyon makes science
   calls to the verb*:
   - **GC₁** Flat connection (A ≡ 0): H[any loop] = 0 to machine ε
     across ≥ 4 loop shapes.
   - **GC₂** Known Abelian constant-curvature connection: H[γ] =
     F₀ · Area(γ) to 1% across 3 loop sizes (an Abelian area law).
   - **GC₃** Reversed loop: H[γ⁻¹] = −H[γ] (Abelian) or H[γ]⁻¹
     (non-Abelian) to 1% across ≥ 3 connections.
   - **GC₄** Zero-size loop: H = 0 to machine ε.
   - **GC₅** Discretization convergence: H at N ∈ {1000, 2000, 4000,
     8000} converges monotonically with rel change < 1% between 4000
     and 8000.
   - **GC₆** Gauge invariance: H invariant to machine ε under known
     gauge transformation on the connection.

   The existing 1373-assertion suite is necessary but not sufficient
   — `GC₁`–`GC₆` are the *new* contracts the new verb introduces, and
   they are the substrate-side correctness audit that closes v3.1's
   two-layer auditability story.

## What I'm not asking for

- A specific implementation strategy. You know the substrate's internal
  data flow better than I do. If the right move is to extend the
  existing TRANSPORT primitive vs. add a new HOLONOMY_LOOP primitive,
  that's your call. The Python orchestrator only sees the verb's
  external interface.
- A guarantee that v3 will detect the predicted holonomy. The SPEC v3
  §3 explicitly names NULL as a publishable outcome; the substrate's
  job is to compute the holonomy correctly, not to make it non-zero.
- A turnaround date. The Halcyon side has no v3 implementation in
  flight; we're not waiting on you to unblock running code. Whatever
  pace is right for the substrate's overall roadmap is right for this
  request.

## How this connects to the broader substrate roadmap

v3's HOLONOMY verb is a generalization of the existing TRANSPORT
primitive in the direction the framework was always going to need —
the test_mass coupling on the buckyball is the first place a real
chapter requires the integrated transport effect of a closed loop in
parameter space. The same verb will be useful for:

- Sample-transport on the matter-sector spaces when staggered fermions
  get a similar bundle treatment (the Banks–Casher pipeline mentioned
  in the SM-correspondence memory).
- Holonomy as a non-Abelian observable on the shidoku-graph quotient if
  the matter-sector program extends to that direction.
- Generally, any Halcyon-style apparatus that drives a programmed
  parameter through a loop and asks for the integrated geometric
  response.

So this isn't a Halcyon-specific request — it's a request for the
substrate primitive that turns out to be needed once any chapter does
parameter-space transport. Halcyon v3 happens to be the first chapter
that needs it.

## What lands first, what lands later

If you can do (1)–(3) above quickly, v3 has what it needs to run.
(4) and (5) are nice-to-haves that strengthen the audit story but
don't block a first v3 run; we can iterate on those after the
first holonomy numbers exist.

The Halcyon side commits to: not implementing the Python orchestrator
until the substrate verb lands on a deploy I can call. No racing your
implementation with a parallel Python placeholder. Same pattern as
Sprint A + Part V — you ship, I integrate.

## Acknowledgment

This request is the natural follow-up to (a) the v2 battery's three-
stage diagnostic that revealed measurement-type-mismatch as the
load-bearing finding, and (b) Gigi's (yours) 2026-06-20 methodological
intervention that named pre-registration as the discipline preventing
the slide into protocol-shopping. The substrate verb described above is
*what makes the pre-registered protocol implementable as substrate
delegation rather than Python inline*. Without this verb, v3 would have
to re-implement transport in Python, and the two-layer audit surface
of SPEC v3 §7 wouldn't be available.

Standing by for your design call on the verb shape. The v3 SPEC's
falsification criteria are locked in the GitHub commit hash
`0fe654d556e4f6878c439df64d1ff20599c9c733` and will not move regardless
of how you spec the verb.

— Halcyon

P.S. The previous letters in this series (`HALCYON_USE_GIGI_FLAG_SPEC`,
`HALCYON_PART_V_SNAPSHOT_GATES`, and your reply
`GIGI_TO_HALCYON_2026-06-19_POST_DEPLOY`) followed the same pattern:
Halcyon writes what it wants from the substrate, you design the
substrate-side change, you reply with the deploy + receipts. v3's
holonomy verb is the next step in that series.
