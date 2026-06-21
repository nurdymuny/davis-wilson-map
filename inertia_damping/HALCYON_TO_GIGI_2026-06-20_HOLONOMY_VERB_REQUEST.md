# Halcyon → Gigi — Holonomy verb for the v3 falsification battery

**Date:** 2026-06-20
**Pattern:** same as the `--use-gigi` flag spec and the Part V SNAPSHOT
gates — Halcyon writes the substrate request, Gigi designs the engine-
side implementation, the two coordinate via this letter pattern.
**Predecessor:** `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md`,
committed at `0fe654d556e4f6878c439df64d1ff20599c9c733`. Read §2 and §4
of that document first; this letter is the implementation request that
falls out of those sections.

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

## The verb, written down

```
SAMPLE_TRANSPORT <gauge_field_name>
  ALONG_LOOP <loop_id>
  ADIABATIC TRUE
  RAMP_RATE <float>           // |dQ/dt|, in Q-units per simulation time unit
  DRIVE_OMEGA <float>          // test-mass lock-in carrier frequency (rad / time-unit)
  DRIVE_F0 <float>             // test-mass drive amplitude
  N_DISCRETIZATION <int>       // number of substeps along the loop
  PIN_LAMBDA <float>           // soft-constraint strength keeping Q on the programmed path
  ALPHA_HALCYON <float>        // coupling calibration (currently free, 1 or 1000)
  TAU_0 <float> BETA_TAU <float>  // tau_Q model parameters per SPEC v2 §3
  MU_BASELINE <float> K_SPRING <float> C_DAMP <float>  // test-mass parameters
  SEEDS <int_list>             // RNG seeds; one independent realisation per seed
  COMPUTE HOLONOMY            // primary observable: closed-loop integrated coupling
  COMPUTE TRANSPORT_TRACE      // per-substep trace for verification
  COMPUTE PER_SEED_HOLONOMY    // disaggregated for gate evaluation
  RETURN H_mean, sigma_H_blocked, per_seed_H, trace_path, adiabaticity_check
```

Loop specs are first-class objects, declared earlier in the GQL block:

```
LOOP gamma_unit:
  Q_PATH [0.0, 1.0, 2.0, 1.0, 0.0]
  SEGMENTS_PER_EDGE LINEAR
  T_LOOP 100.0

LOOP gamma_reversed:
  Q_PATH [0.0, 1.0, 2.0, 1.0, 0.0]
  SEGMENTS_PER_EDGE LINEAR
  T_LOOP 100.0
  TRAVERSE REVERSED

LOOP gamma_degenerate:
  Q_PATH [0.0, 0.0]
  T_LOOP 100.0
```

If the LOOP DSL form bothers you, an equivalent JSON-side
parameterization works too — what matters is that loops are
substrate-side objects, not strings reconstructed on every call.

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

1. **The verb itself** — `SAMPLE_TRANSPORT … ALONG_LOOP … ADIABATIC`
   landing on `gigi-stream.fly.dev` and the local build, with the
   parameter surface above. This is the load-bearing piece; v3 cannot
   run without it. If the existing `SAMPLE_TRANSPORT` verb is close to
   this, the work is mostly extending its grammar; if it's not, this
   is a new verb sibling of HOLONOMY.
2. **The six sham flags** — implementable as a single
   `SHAM_MODE <enum>` parameter if cleaner. Each variant changes one
   piece of the computation (zero out κ, zero out α, scale μ_baseline,
   reverse traversal direction, hold Q at 0, freeze U). Most of these
   are 1-5 line changes inside the per-substep loop.
3. **The adiabaticity self-check** — substrate-side computation of the
   instantaneous gauge-field relaxation rate vs the prescribed ramp
   rate, with a warning emitted in the response when violated. Useful
   for the substrate's own diagnostics regardless of Halcyon's use.
4. **Per-seed independent computation** — confirming that the substrate's
   existing seed model (PCG64 in Python, xorshift64* in Rust per the
   v1.2.1 substrate-consolidation receipt) propagates per-seed
   independence through the loop transport, so the per-seed
   distribution is what the §3 gates need.
5. **A regression test in your `cargo test --features halcyon` suite**
   that runs the verb on a small loop and verifies (a) sham flags
   return zero to machine precision, (b) the holonomy of a trivial
   trivial-bundle loop returns zero, (c) the holonomy of a known
   non-trivial test connection returns its analytical value. This is
   the substrate-side correctness check that makes the audit story
   real.

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
