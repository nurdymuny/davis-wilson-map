# Halcyon Falsification Battery — SPEC v3

**Status:** PRE-REGISTRATION. This document is the timestamped record of the
falsification criteria and protocol design *before any v3 code runs*. Once
committed to GitHub (and subsequently archived on Zenodo), the criteria
below cannot be tuned after the fact. Any deviation in the implementation
constitutes a different spec, not a refinement of this one.

**Date written:** 2026-06-20
**Implementation status at time of writing:** none
**Predecessor:** `HALCYON_FALSIFICATION_BATTERY_SPEC.md` (v2.1), retained
in its entirety as a first-class artefact, not deprecated.
**Author commitment:** The simulation will not run against this spec until
this document is committed and pushed. The git hash of this commit IS the
pre-registration record.

## §1 — Why v2's measurement was wrong for the dynamics being measured

**This section is not "v2 failed."** It is an architectural diagnosis of
what v2 succeeded at, what it did not, and which of v2's findings stand
independently of v3's redesign.

### 1.1 What v2 established (load-bearing into v3)

The v2 battery's positive findings are not consequences of the measurement
type and survive into v3 as established properties of the underlying
framework:

- **H₁ (material independence) strikes**: `|dα/dμ_proxy|/|α| = 0.005–0.028`
  across three independent runs (3-seed, 8-seed, 8-seed calibrated). The
  Halcyon coupling is geometric, not material-dependent. *This is a
  structural property of the framework, not the measurement type.* It
  holds regardless of whether α is extracted via internal `mu_eff` or
  observed `χ(ω)`.
- **H₅ (drive-amplitude linearity) strikes** at machine precision
  (`rel slope of χ vs F₀ < 0.001`). The lock-in regime is valid as a
  measurement technique within its domain of applicability. *This
  survives into v3 because any frequency-domain probe inside v3 still
  needs linearity in its instantaneous drive.*
- **H₉ (τ_Q model robustness) strikes**: `|α_alt − α|/|α| = 0.025–0.13`
  across alternative `τ_Q` functional forms. The conclusion is not
  overfitted to the specific `τ_Q = τ₀/(1+β_τ s_Q)` form. *This is a
  property of the underlying model, not of the protocol that reads it.*
- **The Newtonian limit at Q=0 is exact**: `μ_eff(trivial vacuum) = 0`
  by construction because `κ_Q(e) = (1−q₀)²` vanishes at the identity.
  *Geometric, not stamped on.*
- **The smoke-mode internal signal is real**: `|α|/σ = 15.6` at the
  model's own internal `μ_eff(U(t))` state. *The Davis mechanics work.
  μ_eff genuinely couples to Q in the predicted way at the model level.*

These five findings are evidence that the framework is doing geometric
physics rather than material physics, is operationally well-defined,
and is robust to reasonable variation in its parameters. They do not
depend on whether v2's *external observability* succeeded.

### 1.2 What v2 did not establish (and why)

The v2 battery did not establish `∂_Q μ_Q ≠ 0` in observation space.
That was the load-bearing claim, and at α_Halcyon ∈ {1, 1000} the
observation-space extractor returned `FAIL_SIGNAL_MISSING` in both runs.

The architectural diagnosis is that the *measurement type* was wrong
for the *dynamics being measured*. The apparatus runs in the *diabatic
regime* (cage continuously drives Q); v2 measured the *adiabatic
limit* (Q held by initial-condition bias, decaying back to trivial
vacuum under the Hamiltonian's natural dynamics). The mismatch is
structural and was not caused by calibration error or implementation
error.

### 1.3 The load-bearing diagnostic — α scaling vs. SNR scaling

**This single finding is what forces the v3 redesign.** Bumping
α_Halcyon by 1000× moved the observation-space SNR by only 2.5×
(from 0.21σ to 0.52σ). If the noise were dominated by lock-in
shot-noise or seed dispersion, the SNR would have grown linearly
with α. It did not. The noise is therefore *intrinsic to the
dynamics being measured*, not extrinsic to the measurement.

When the noise scales with the signal, the measurement is reading
the wrong observable. In v2's case, the noise tracks the H₈
Q-drift: the same gauge-field dynamics that produce μ_eff(Q) also
produce the relaxation of Q toward the trivial vacuum during the
measurement window. Pushing α harder does not separate the two;
only changing the measurement type does. SNR ~ `τ_Q-relaxation /
t_total` is the asymptotic ratio; α does not enter.

This is the diagnostic that distinguishes "v2 needs better
calibration" from "v2 is measuring the wrong thing." The 1000× bump
ruled out the first. v3 addresses the second.

### 1.4 What v2's null is *if* v3 also returns null

This is named here, in §1, before v3's protocol is written: if the
framework is wrong, v2's null result combined with v3's null result
is *stronger evidence against the framework than either alone*. v2
becomes a control case for v3 in this scenario — two independent
measurement protocols both failing to detect the predicted effect.
v2 is not deprecated, it is *complementary*.

### 1.5 What v2's null is *if* v3 detects holonomy

If the framework is right, v2's null is itself a confirming prediction
of v3's model: in the adiabatic limit at this calibration, the
fixed-Q lock-in *should* return below threshold because the noise is
dominated by Q-drift that the apparatus would have actively
suppressed. v2 becomes a measurement-design control for v3 — the
fixed-Q approach is the *wrong* probe of a holonomy observable, and
its null result is exactly what the framework predicts when you
measure the wrong object.

### 1.6 The v2 sidecars and what they preserve

Three sidecars are preserved in the permanent record:

- `battery_fast_20260620_104846.json` (3-seed and 8-seed smoke,
  internal extractor, 15.6σ "internal pass")
- `battery_full_20260620_181227.json` (full battery, observation
  extractor at α=1, FAIL_SIGNAL_MISSING at 0.21σ)
- `battery_calibrated_20260621_011304.json` (calibrated to α=1000,
  observation extractor, FAIL_SIGNAL_MISSING at 0.52σ, H₈ revealed
  as the dominant blocker)

Each carries the SHA-256 in its metadata. None are deleted, none
refactored. v2.1 of the SPEC remains the authoritative document for
its own measurement design.

## §2 — The framework's native observable is holonomy

**Three-sentence anchor.**

> The framework's native object is holonomy.
> The apparatus measures holonomy.
> The simulation should compute holonomy.

This is the load-bearing architectural claim of v3. The fixed-Q lock-in
of v2 measured a *susceptibility* (the response amplitude at a fixed
point in parameter space); the apparatus actually produces a *holonomy*
(the integrated transport effect of a programmed closed loop in
parameter space); the framework's bundle structure makes holonomy the
natural gauge-invariant observable. The three layers should match. They
did not in v2.

### 2.1 The bundle structure

The test mass is not a free particle that happens to interact with a
gauge field at fixed Q. It is a section of a bundle whose base is the
parameter manifold of programmed gauge sectors and whose fiber is the
configuration space of the test mass. The Halcyon coupling
α_Halcyon · (1/E) Σ κ_Q τ_Q² |φ_n|² defines a *connection* on this
bundle — it specifies how the test mass's effective inertial coefficient
transports as the gauge field configuration is moved along a path in
parameter space. The fixed-Q lock-in of v2 measured the connection's
local value at three isolated points; the holonomy is the integrated
transport effect of moving around a closed loop.

### 2.2 The pulled-back connection on the worldline

Equivalently — and this is the dual statement Gigi flagged in the
session — the bundle the test mass lives on is *not* the product
(test_mass × Q-space). It is the *pull-back* of the connection on the
gauge-field bundle to the test mass's worldline. As the cage drives Q
along a programmed path, the pull-back connection acquires the
non-trivial structure that turns it into a measurable holonomy. The
fixed-Q protocol of v2 trivialized this — it pulled back the
connection to a *point* in Q-space, which makes the holonomy
identically zero by triviality of the contractible loop.

### 2.3 The native observable, written down

The load-bearing falsifiable observable for v3 is the closed-loop
holonomy of the inertial coupling:

$$
\mathcal{H}[\gamma] \;=\; \oint_\gamma \frac{\partial\mu_{\rm eff}}{\partial Q}\, dQ
$$

for a closed loop γ in Q-space, evaluated via *adiabatic transport* of
the gauge field along γ while the test mass is being driven.

For a topologically trivial loop on a flat bundle, `H = 0` by Stokes.
For a non-trivial loop or a non-flat bundle, `H ≠ 0` and its magnitude
and sign carry the geometric content of the framework. The Halcyon
prediction is that the bundle is *not* flat — that the gauge-field
configuration as the cage cycles through programmed sectors produces a
non-zero curvature on the (test_mass, Q) bundle — and therefore that
`H[γ] ≠ 0` for closed Q-loops of finite area.

### 2.4 GIGI's native verbs

GIGI exposes HOLONOMY, TRANSPORT, SPECTRAL, and BETTI as substrate
primitives. The v3 simulation's measurement is *literally* a GIGI
HOLONOMY call:

```
HOLONOMY OF mu_eff
  ALONG loop_Q_0_1_2_1_0
  ON GAUGE_FIELD halcyon_canonical_buckyball
  BETA 2.5
```

The simulation becomes a thin Python wrapper that constructs the loop,
calls the GIGI verb, and parses the result. The substrate computation
is delegated entirely to GIGI; the Python side handles only protocol,
gates, and reporting.

This is the architectural anchor that distinguishes v3 from v2: v2
implemented its own integrator and its own susceptibility measurement
inline in Python; v3 calls the substrate's native HOLONOMY verb and
the substrate handles the math.

## §3 — Pre-registered falsification criteria

**This section is written before §4. The protocol of §4 is designed to
satisfy the criteria of §3, not the other way around.**

### 3.1 The primary observable and its three regimes

The primary observable is `H[γ_unit]`, the holonomy of the inertial
coupling around the unit Q-loop γ_unit = 0 → 1 → 2 → 1 → 0 traversed
adiabatically.

Three valid outcomes are pre-declared, with numerical thresholds:

| Outcome | Criterion | Interpretation |
|---|---|---|
| **POSITIVE** | `|H[γ_unit]| > 5 σ_H` AND sham controls return `< 1 σ_H` | The framework's predicted non-trivial bundle curvature is detected. |
| **NULL** | `|H[γ_unit]| < 1 σ_H` AND all sham controls return `< 1 σ_H` | The framework's predicted holonomy is absent at this calibration and substrate. |
| **AMBIGUOUS** | `1 σ_H ≤ |H[γ_unit]| ≤ 5 σ_H` OR any sham control fails to return zero | The result is not interpretable; investigate or re-run. |

σ_H is the Flyvbjerg–Petersen blocked SEM of `H[γ_unit]` measured
across the 8 seeds, computed identically to v2's blocked SEM (no
post-hoc adjustment).

### 3.2 Required sham controls

A POSITIVE verdict requires *all* of the following sham controls to
return `< 1 σ_H` (within noise of zero):

| Sham | What it tests | Expected if framework right | Expected if framework wrong |
|---|---|---|---|
| **S₁** Q-ramp on flat field (κ_Q ≡ 0 by hand) | The holonomy isn't an artifact of the ramp itself | `H = 0` | `H = 0` |
| **S₂** Q-ramp with α_Halcyon set to 0 | The holonomy is driven by the framework coupling, not the test-mass dynamics | `H = 0` | `H = 0` |
| **S₃** Q-ramp on a control test mass (μ_baseline scaled 10×) | The holonomy is gauge-invariant under material change | `H invariant under scaling` | `H = 0` for both |
| **S₄** Q-ramp reversed (0 → 1 → 2 → 1 → 0 traversed backwards in time) | The holonomy is orientation-dependent (sign flip) | `H → −H` | `H = 0` for both |
| **S₅** "Loop" of zero size (Q held constant at Q=0) | The holonomy of a degenerate loop is zero | `H = 0` | `H = 0` |
| **S₆** Loop traversal at the same rate but with the gauge field artificially frozen | Distinguishes geometric holonomy from kinetic artifact | `H = 0` (no transport without field evolution) | `H = 0` |

Note that S₁, S₂, S₅, S₆ predict `H = 0` regardless of which way the
framework goes. They are *plumbing checks* — they verify the
measurement apparatus correctly returns zero for null inputs. If any
of them fail (return non-zero), the result is not interpretable
regardless of the primary observable.

S₃ and S₄ are the *load-bearing falsifiers within the POSITIVE
branch*: if `H[γ_unit] > 5 σ_H` but `H[scaled mass]` differs by more
than the framework predicts under gauge invariance, the positive
signal is contaminated by an artifact. If `H[reversed γ] ≠ −H[γ]`
within `σ_H`, the holonomy is not orientation-dependent and is not
genuinely geometric.

### 3.3 Stopping rule

**This is the discipline that prevents the slide into v4, v5, …**

The framework is declared *falsified by simulation* if all three of
the following conditions are met:

1. v3 returns NULL (`|H[γ_unit]| < 1 σ_H` with all sham controls
   passing).
2. A second independent v3-class measurement design — to be
   specified in a hypothetical SPEC v4 — *also* returns NULL on the
   same substrate at the same α_Halcyon calibration.
3. The two measurement designs are not trivially equivalent to each
   other (independence verified by an outside reviewer).

If those three conditions are met, the simulation does not support
the framework's prediction at the buckyball substrate, and the
program does not proceed to a v5 measurement design without an
externally reviewed reason. *The committee that declares this
falsification is named in advance: any two of the following — Gigi,
an external lattice-gauge-theory reviewer of her choosing, a peer
reviewer from a journal submission process.* Internal iteration
without external review does not satisfy condition (3).

The framework remains *not yet falsified* if:

- v3 returns POSITIVE: the prediction is supported on this substrate.
- v3 returns NULL but the prediction's calibration `α_Halcyon` is
  refined downward (i.e., the independent prediction from the Davis
  Field Equations turns out to be even smaller than v2's calibrated
  α=1000 was probing). In this case the program proceeds to design
  a measurement sensitive at the lower calibration.
- v3 returns AMBIGUOUS and the ambiguity is resolved by a re-run
  with explicitly named systematics improvements.

The stopping rule is *not* "the simulation has tried enough times."
It is *the simulation has tried two independent measurement designs
under reviewed conditions and neither detected the predicted effect.*
The cost of admission to "framework wrong" is the same as the cost
of admission to "framework right": two independent measurements,
publicly committed in advance, with an outside reviewer.

### 3.4 Specific calibration commitments

The v3 simulation will run at three calibrations:

- α_Halcyon = 1 (v2 default — reproduces the v2 calibration so the
  results are directly comparable)
- α_Halcyon = 1000 (v2 calibrated bump — direct comparison to
  v2's calibrated run)
- α_Halcyon = TBD from the Davis Field Equations closed-form
  derivation (the independent prediction A.6 flagged as open work).
  *If this value is not yet derived at the time of v3 execution,
  the simulation runs at α=1 and α=1000 only, and the third
  calibration is deferred.*

`σ_H` is computed at the calibration corresponding to the relevant
verdict — i.e., the gates fire per calibration. A POSITIVE verdict
at α=1000 with NULL at α=1 is the predicted pattern if the framework
is right *and* α_Halcyon scales the signal but not the noise (the
opposite of v2's adiabatic regime).

### 3.5 Per-seed independence

A holonomy verdict requires per-seed independence ≥ 5/8 (same
threshold as v2). The primary observable `H[γ_unit]` is computed
per seed, and the POSITIVE/NULL/AMBIGUOUS gates apply to the per-seed
distribution, not just the mean.

## §4 — Q-ramp protocol via GIGI's HOLONOMY and TRANSPORT verbs

The protocol delegates the substrate computation to GIGI. The Python
side specifies the loop, calls the substrate verb, and parses the
result. The Python code path is reviewable here; the substrate's
correctness is reviewable through GIGI's 1373-assertion test suite
(§7).

### 4.1 The loop γ_unit

The unit Q-loop is a piecewise-linear path in Q-space:

```
γ_unit:  Q(s) = ramp(0 → 1 → 2 → 1 → 0)
         with s ∈ [0, 1], traversed at constant dQ/ds rate
         in four equal segments of duration t_seg = T_loop / 4
```

T_loop is the total loop duration (a free parameter to be set by the
adiabaticity condition below). Each segment is a linear ramp in Q.
The loop closes at Q(1) = Q(0) = 0.

### 4.2 The adiabaticity condition

The loop must be traversed slowly enough that the gauge field is
quasi-statically equilibrated at each Q value, but not so slowly
that the gauge field's intrinsic relaxation drives Q-drift away
from the programmed path. The condition is:

```
τ_drive >> T_loop >> τ_Q-relax
```

where τ_drive is the test-mass drive period (the lock-in carrier),
T_loop is the loop traversal time, and τ_Q-relax is the gauge
field's intrinsic Q-relaxation timescale measured in v2 (~10–30
time units in v2's units).

This is *the* tradeoff v3 has to thread. The protocol pre-registers
specific values: T_loop = 100 time units, drive period = 6.28 time
units (16 cycles per loop), τ_Q-relax (estimated from v2's H₈ data)
= ~10 time units (loop is ~10× faster than relaxation, fast enough
to outrun drift). These are *committed* values, not tuneable.

### 4.3 The GIGI call

The substrate computation is delegated to GIGI via:

```
SAMPLE_TRANSPORT halcyon_canonical_buckyball
  ALONG_LOOP gamma_unit
  ADIABATIC TRUE
  RAMP_RATE 0.04        // (Q_max − Q_min) / T_loop = 2/100 (units of Q per time unit)
  DRIVE_OMEGA 1.0       // test-mass lock-in carrier frequency
  COMPUTE HOLONOMY
  COMPUTE TRANSPORT_TRACE
  RETURN H, sigma_H, per_seed_H, trace_path
```

The GIGI verb computes the holonomy in the substrate's native
representation. The Python orchestrator (`run_holonomy_battery.py`,
TBD) wraps the call, supplies the loop parameterization, and parses
the result.

If `SAMPLE_TRANSPORT` does not yet expose an `ALONG_LOOP ADIABATIC`
flag, that is a GIGI-side spec update — to be coordinated with the
Halcyon→GIGI letter pattern used in earlier SPECs. The v3
implementation does not implement the substrate math in Python.

### 4.4 The discretization

The continuous Q(t) ramp is discretized at the simulation timestep
dt = 0.02 (matching v2's dt for cross-comparison). Each leapfrog
step receives an updated `Q_target(t)` and the gauge field is
pinned toward Q_target via a soft-constraint potential added to
H_gauge:

```
V_pin(U, Q_target) = lambda_pin * (Q_surrogate(U) - Q_target)^2
```

with `lambda_pin = 1.0` (committed value, not tuneable). The pinning
potential is what allows the simulation to actively hold Q on the
programmed path — replacing the v2 passive initial-condition
approach. The pinning is *part of the simulation*, mirroring the
cage's role in the apparatus. (In GIGI's substrate, the equivalent
mechanism is the `ADIABATIC` flag's enforcement of slow transport.)

### 4.5 The integration

H[γ] is computed as a path integral:

```
H[γ] = ∮ (∂μ_eff/∂Q)(Q(s)) · (dQ/ds) · ds
     ≈ Σ_n (μ_eff(Q_{n+1}) − μ_eff(Q_n))   (finite-difference along discretization)
```

For a closed loop this telescopes to:

```
H[γ] = μ_eff(Q_end) − μ_eff(Q_start) = 0 (by closure)
```

— *unless* the bundle has non-zero curvature, in which case the
path-dependent contributions accumulate to a non-zero holonomy.
The integral is computed in two equivalent ways:

(a) Direct sum of (μ_eff(Q_{n+1}) − μ_eff(Q_n)) along the
discretized loop, expected to be zero for a trivial bundle by
telescoping.

(b) Wilson-loop-like integral of the connection 1-form along the
loop, computed by GIGI's HOLONOMY verb.

A non-trivial bundle is detected by **disagreement between (a) and
(b)** at finite step size. If the connection is flat, both methods
return zero; if the connection has curvature, the Wilson-loop method
captures it while the naive telescoping does not.

This is the substantive measurement v3 makes that v2 could not:
**curvature of the (μ_eff, Q) bundle, not pointwise μ_eff(Q)
values.**

### 4.6 The Python orchestrator surface

(For implementation reference; the orchestrator does not need to be
written until after this spec is committed.)

```python
def run_holonomy_battery(graph, alpha_halcyon, seeds, log_path):
    """Compute H[γ_unit] and the six sham controls per SPEC v3."""
    primary = compute_holonomy_via_gigi(graph, alpha_halcyon, GAMMA_UNIT, seeds)
    shams = {name: compute_holonomy_via_gigi(graph, alpha_halcyon, sham, seeds)
             for name, sham in SHAM_LOOPS.items()}
    verdict = apply_v3_gates(primary, shams)
    return {"verdict": verdict, "H_primary": primary, "shams": shams}
```

## §5 — Sham controls (specified in detail)

The six sham controls of §3.2 expanded with their concrete
implementations:

### S₁ flat-field Q-ramp

Run the protocol with `κ_Q(e)` replaced by 0 for all edges, at all
times. This zeros out the Halcyon coupling at the κ-factor level,
not the α-factor level (S₂ is the α-factor zero). Different
diagnostic: S₁ tests whether the holonomy is an artifact of the
ramp + the test-mass dynamics alone; S₂ tests whether it is an
artifact of α scaling.

### S₂ α=0

Run the protocol with `α_Halcyon = 0`. The test mass has only its
baseline inertia; no Halcyon coupling enters the dynamics. Should
return H = 0 to machine precision.

### S₃ control test mass

Run the protocol with `μ_baseline` scaled 10× (or 0.1×). The
gauge-invariant prediction is that the holonomy is invariant under
this scaling (it depends on `μ_eff(Q)` not on `μ_total = μ_baseline
+ μ_eff(Q)`). A non-invariant H signals contamination by the
baseline mass — i.e., the measurement isn't truly geometric.

### S₄ reversed loop

Run the protocol with `γ_unit` traversed in the time-reversed
direction: Q(s) follows 0 → 1 → 2 → 1 → 0 with s decreasing. A
genuine holonomy of a non-Abelian connection is orientation-
dependent: `H[γ⁻¹] = −H[γ]` (for Abelian) or `H[γ⁻¹] = H[γ]⁻¹`
(for non-Abelian; here we work at the level of the test mass's
abelianized response). The simpler check is `H + H_reversed ≈ 0`
to within `σ_H`.

### S₅ zero-size loop

Hold Q constant at Q=0 for the full duration T_loop with the drive
running. The "loop" is a degenerate point; its holonomy is
identically zero by triviality. If non-zero, the apparatus has a
bias.

### S₆ frozen-field loop traversal

Run the loop traversal but freeze the gauge field (no dynamics; U
held at U(t=0) throughout). The `Q_target` is updated; the gauge
field is not. A non-zero H from this control indicates the
"holonomy" is being generated by the ramp's update of `Q_target`
alone, not by genuine gauge-field transport.

## §6 — What v3's results mean for v2

### 6.1 POSITIVE v3 + NULL v2

This is the *predicted pattern if the framework is right*. The
fixed-Q lock-in of v2 measures the wrong observable (point value of
the connection); the closed-loop holonomy of v3 measures the right
observable (integrated curvature of the bundle). The two are
independent observations and disagree because they ask different
questions. v2's null is no longer a failure of detection; it is a
*confirmed prediction* that the adiabatic limit returns zero on a
contractible loop (which a fixed-Q is, trivially). v3's positive is
the first evidence that the bundle has non-trivial curvature.

In this scenario, the chapter v4.6 (or v5, depending on naming)
reports: "v2 measured the adiabatic limit and correctly returned
zero (as the framework predicts); v3 measured the holonomy and
detected the predicted geometric coupling. The two together support
the framework more strongly than either alone."

### 6.2 NULL v3 + NULL v2

This is the *stopping condition*. Per §3.3, two independent
measurement designs both returning null, with the stopping rule
committee assembled, declares the framework not supported by
simulation at this substrate and calibration. v2 and v3 are
*complementary controls* in this scenario, not redundant. The
chapter reports both nulls and the program enters its declared
stopping state.

### 6.3 NULL v3 + (smoke-mode internal pass on v2)

The internal-extractor signal of v2 (|α|/σ = 15.6) is *not*
contradicted by a v3 null. v2's internal extractor confirmed the
model's mechanics; v3 (and v2's external extractor) tests
observability. A NULL v3 means the framework's mechanics produce
no observable holonomy, which is the *load-bearing falsification*.
The internal mechanics' working is not enough.

### 6.4 POSITIVE v3 + POSITIVE v2 internal

This would be the strongest positive case: both the model's
internal state and an observable holonomy match the prediction.
Chapter v5 would report this as joint confirmation across two
extractor types.

### 6.5 AMBIGUOUS v3

Per §3.1, ambiguous means `1σ ≤ |H| ≤ 5σ` *or* a sham control
failed. The published outcome is "ambiguous, ambiguous-resolution
re-run named in v3a." The re-run is allowed if and only if the
ambiguity-resolution criteria are written *before* the re-run.
This is the same discipline as §3.3 — the protocol cannot be tuned
to resolve the ambiguity in either direction.

## §7 — GIGI audit surface

The architectural inversion that v3 commits to: the substrate's
computational correctness is reviewable through GIGI's test
instrumentation (1373 assertions, 46.36 seconds, currently
deployed at `gigi-stream.fly.dev` and locally buildable per the
v1.2.1 closure receipt). The v3 spec is reviewable independently
in this document.

### 7.1 Two-layer independent auditability

- **Layer 1 — substrate computation.** `SAMPLE_TRANSPORT`, `HOLONOMY`,
  `TRANSPORT`, `SPECTRAL` are GIGI verbs. Their correctness is the
  domain of GIGI's test suite, not the Halcyon protocol. A reviewer
  who wants to verify the substrate math runs the GIGI tests
  (`cargo test --features halcyon`) and inspects the WAL receipts.
- **Layer 2 — protocol design.** The choice of loop γ_unit, the
  adiabaticity condition, the sham controls, the gate thresholds,
  the stopping rule are the domain of this SPEC. A reviewer who
  wants to verify the experimental design reads this document.

Neither layer needs to inspect the other's internals to perform
its review. The interface is the GIGI verb signature.

### 7.2 The receipt model

v3 emits a `section_12_holonomy_battery` JSON sidecar with:

- The SHA-256 of this SPEC at the time of execution
- The GIGI deploy hash used for the substrate calls
- Per-seed `H[γ_unit]` and the six shams
- Per-gate verdict
- The σ_H values used in the gate evaluation
- The overall verdict (POSITIVE / NULL / AMBIGUOUS)

Reproducibility: anyone with this SPEC's SHA-256, the GIGI deploy
hash, and the seed list can re-run the experiment and verify the
sidecar. The two layers are independently re-runnable.

### 7.3 What this changes for the auditor's job

In v2, the auditor had to read Python code (~750 lines in
`falsification_battery.py` + ~400 in `test_mass_dynamics.py`) to
verify the measurement was correct. In v3, the auditor reads this
SPEC (~500 lines) and trusts the GIGI test suite for the substrate.
The audit surface is *smaller* and *more orthogonal* — the
substrate's correctness audit is amortized across every chapter
that uses the substrate, not re-done per protocol.

## §8 — Publication commitment

### 8.1 The commitment

This document is committed to the public record at:

- GitHub: `nurdymuny/davis-wilson-map`, branch
  `feat/halcyon-gigi-substrate`, file
  `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md`.
- Zenodo: a DOI will be minted from this SPEC's commit hash before
  the v3 protocol runs. The Zenodo deposit is mandatory; the GitHub
  commit alone is the implementation-level pre-registration; the
  Zenodo DOI is the publication-level pre-registration.

The git commit hash of this SPEC's first push IS the
pre-registration timestamp. The simulation will not run against this
spec until both the git push and the Zenodo deposit are complete.

### 8.2 What gets published when the v3 run completes

All of the following are published, regardless of the verdict:

- The `section_12_holonomy_battery` sidecar JSON.
- A revision of Solves Vol. 4 with an Appendix A.8 reporting the v3
  result (POSITIVE, NULL, or AMBIGUOUS).
- The log file from the run, with per-seed verdicts.
- The GIGI deploy hash and WAL receipt for the substrate calls.

There is no version of this in which the result is suppressed if
unfavorable. The pre-registration commits to publication *of
whatever happens*, not publication of a result that confirms.

### 8.3 If v3 is NULL

Per §3.3 and §6.2, NULL is publishable. The chapter v5 (or whatever
version label applies) reports the null, names v2's null as the
complementary control, and either (a) declares the stopping
condition per §3.3 (if a second independent measurement design also
returns null), or (b) names the conditions for a v4 measurement
design under external review. The null is *not* re-run in a v3a
without external review.

### 8.4 If v3 is POSITIVE

The chapter v5 reports the positive, names v2's null as the
adiabatic-limit confirming prediction, and proceeds to design the
apparatus-side measurement that would replicate the holonomy on
hardware. The simulation's positive is *necessary* for the
apparatus claim but not *sufficient*; the apparatus must also
demonstrate the holonomy on its own substrate.

### 8.5 The committee

The stopping-rule committee of §3.3 — required for declaring the
framework falsified — is to be assembled at the time the second
NULL is recorded, not in advance. The committee consists of: (a)
Gigi, (b) one external lattice-gauge-theory reviewer chosen by
Gigi from outside the program, (c) one peer reviewer from a journal
submission process. The committee's role is to verify that the two
measurement designs (v3 and the hypothetical v4) are not trivially
equivalent. They do not vote on the framework's correctness;
they vote on the *independence* of the two measurement designs.

## §9 — What this SPEC does not commit to

For symmetry with the other SPECs in the series:

- This SPEC does not commit to a particular `α_Halcyon` value derived
  from the Davis Field Equations. That derivation is open work.
- This SPEC does not commit to the v3 result. The result is whatever
  the protocol returns.
- This SPEC does not commit to a hardware apparatus design. v3 is
  simulation-only; the hardware protocol is a sibling spec.
- This SPEC does not declare v2 deprecated. v2 is preserved as a
  first-class artefact.
- This SPEC does not promise that v3 will succeed in detecting the
  predicted holonomy. It promises that v3 will *test* for the
  predicted holonomy under the conditions of §3 and publish the
  result of §8.

## §10 — Definitions of done

The v3 program reaches "done" when *all* of the following are true:

- [ ] This SPEC is committed to GitHub.
- [ ] This SPEC is deposited on Zenodo with a DOI.
- [ ] The GIGI `SAMPLE_TRANSPORT ... ALONG_LOOP ADIABATIC` verb is
  available (either already in GIGI or added per the Halcyon→GIGI
  letter pattern).
- [ ] The Python orchestrator `run_holonomy_battery.py` is implemented.
- [ ] The v3 run produces a sidecar matching §7.2.
- [ ] The verdict is published per §8.2, regardless of which of
  POSITIVE / NULL / AMBIGUOUS it is.
- [ ] Solves Vol. 4 receives an Appendix A.8 reporting the v3 result.
- [ ] If NULL: the stopping-rule conditions of §3.3 are evaluated;
  the committee of §8.5 is assembled if appropriate.
- [ ] If POSITIVE: the apparatus-side replication design is named
  as the next chapter's open work.

## §11 — Acknowledgments

This SPEC was written immediately after a methodological intervention
from Gigi (2026-06-20) naming the danger of protocol-shopping and
specifying the six structural requirements (§1–§8 in this document's
outline). The intervention is the load-bearing reason the SPEC is
written *before* implementation rather than after. Pre-registration is
the cost of admission to either credibility — positive or negative — and
the discipline is hers.
