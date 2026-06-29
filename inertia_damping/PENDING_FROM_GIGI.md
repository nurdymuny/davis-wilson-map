# What Halcyon needs from GIGI before the v3.1.3 protocol can run

A personal-reference note. The cross-team letters carry the full
design; this file is just the "what am I waiting on?" summary.

**TL;DR:** GIGI ships the `LOOP_TRANSPORT` verb with its gate doc on
disk first, the verb passes the `GC₁`–`GC₆` substrate-correctness
acceptance battery, and then Halcyon's `run_holonomy_battery.py`
swaps one line (mock → live client) and runs the v3.1.3 protocol.
Nothing else on the Halcyon side is blocked on Halcyon.

---

## The five deliverables, in dependency order

### 1. Gate doc on disk, BEFORE any code lands

**Where:** `gigi/theory/halcyon/HALCYON_PART_VI_GATES.md`
**Why first:** Sprint B revert lesson — verb-introduction commits do
not touch existing hot paths, and the design-question answers live in
the gate doc, not in the implementation log. This rule was named on
both sides of the letter exchange.

**Must include** (per the design closeout letter §9):
- The full `GC₁`–`GC₆` acceptance battery table.
- The β_W ∈ [2.5, 3.0] parser validation rule with the convenient
  inheritance from Halcyon canonical β = 2.5.
- The loop time-reversal mechanism (CC-LT-7): one `DECLARE LOOP` per
  logical loop in the registry, reverse-traversal as an inline option
  on the `LoopTransport` WAL entry.
- The per-axis `ramp_rate` schema (CC-LT-8):
  `LoopShape::PiecewiseLinear { vertices, t_per_segment }` with
  v3.1.3's four-segment rectangle as the v0.1 consumer.
- The `LoopTransportDiagnostics` struct shape with the tuple return
  (`h_geom`, `h_sys`, per-seed components, σ_blocked, tracking-error
  maxes, `AdiabaticityCheck`, `run_id`).
- The `ADIABATICITY_CHECK` gate per v3.1.3 §4.2: violation forces
  AMBIGUOUS (Halcyon-side, not substrate-side enforcement).

**What this unblocks on the Halcyon side:** I can read it and confirm
it matches v3.1.3 §7.4 verbatim. If it does, I nod and stay out of
the substrate's lane. If it doesn't, I write back before any LOC.

---

### 2. The verb itself — `LOOP_TRANSPORT`

**Where:** `gigi/src/gauge/loop_transport.rs` (~280–350 LOC per the
substrate v1 reply scope estimate).

**Must satisfy:**
- The grammar shape Halcyon committed to in v3.1.3 §4.4:
  `LOOP_TRANSPORT <gauge_field> ALONG_LOOP <loop> CONTROL_MANIFOLD
  (Q, beta_wilson) ADIABATIC TRUE RAMP_RATE_Q ... RAMP_RATE_BETA_W
  ... DRIVE_OMEGA ... N_DISCRETIZATION ... PIN_LAMBDA_Q ... EPS_Q ...
  ALPHA_HALCYON ... TAU_0 ... BETA_TAU ... MU_BASELINE ... K_SPRING
  ... C_DAMP ... SEEDS [...] COMPUTE HOLONOMY_FORWARD COMPUTE
  HOLONOMY_REVERSED COMPUTE TRACKING_ERROR_TRACE_{Q,BETA_W} COMPUTE
  ADIABATICITY_CHECK RETURN ...`
- The `ParameterPackKind::Halcyon` variant with the canonical 12-field
  list from the design closeout §B.2 (alpha, tau_0, beta_tau,
  mu_baseline, K_spring, c_damp, drive_omega, drive_F0,
  pin_lambda_{Q,beta_W}, eps_{Q,beta_W}).
- The **five science-gate sham flags** inside a nested `SHAM { ... }`
  block — these are the Halcyon-side ask, gated by v3.1.3 §5:
  `FLAT_FIELD`, `ALPHA_ZERO`, `MASS_SCALED` (with
  `sham_mass_scale: Option<f64>`), `BACKTRACK_LOOP` (canonical S₅
  mapping), `FROZEN_FIELD`. These five are required for v3.1.3
  science calls.
- **Two optional substrate-internal audit-story flags** named in
  Halcyon v1 reply §D.1 but **NOT in the v3.1.3 deposited SPEC and
  NOT required by Halcyon's gates**: `EMPTY_LOOP` (GC₄ companion —
  stronger zero-area sanity than `DEGENERATE_LOOP` alone) and
  `OPEN_LOOP` (parser-rejection test for input validation). Ship at
  substrate discretion; the Halcyon orchestrator never sets them and
  the v3.1.3 verdict logic never reads them. **VI.3 SHAM-block
  implementation is not blocked on enumerating these** — the five
  flags above are the complete Halcyon-side ask.
- A `LoopRegistry` mirroring `GaugeFieldRegistry` (CC-LT-1 path (a)).

**What this unblocks:** GC tests can be written against it.

---

### 3. The GC₁–GC₆ acceptance battery — GREEN before any science call

**Where:** `gigi/tests/halcyon_part_vi_loop_transport_gc.rs` (new
file, `#[cfg(feature = "halcyon")]`).

**Per v3.1.3 §7.4 (verbatim):**
- **GC₁** — flat connection (A ≡ 0) returns H = 0 across ≥ 4 loop
  shapes, to machine ε.
- **GC₂** — Abelian constant-curvature connection returns H = F₀ ·
  Area(γ) to 1% across 3 loop sizes (area law).
- **GC₃** — reversed loop returns H[γ⁻¹] = −H[γ] to 1% across ≥ 3
  connections.
- **GC₄** — zero-size loop returns H = 0 to machine ε.
- **GC₅** — N_discretization ∈ {1000, 2000, 4000, 8000, 16000}
  converges monotonically with rel change < 1% between 8000 and 16000.
  **Plus the science-value gate:** N = 10000 is accepted only if the
  8000→16000 rel change is < 1%. Substrate **blocks** science calls
  otherwise. The 1% threshold is not negotiable; if the verb can't
  meet it, the verb is patched (or N moves by a v3.1.x amendment),
  not the threshold.
- **GC₆** — gauge transformation leaves H invariant to machine ε.

**What this unblocks:** Halcyon's mock client gets retired.

---

### 4. The sham flag implementations behind the nested `SHAM { ... }` block

**Per the substrate v1 reply §2** (~15–20 LOC for the five clean ones):
- `SHAM_FLAT_FIELD` — bypass `wilson_force_per_edge`, return zero
  force at F0 / F1 sites.
- `SHAM_ALPHA_ZERO` — override `pack.alpha` (and any equivalent inside
  the Halcyon variant) to 0.0 before the per-substep loop.
- `SHAM_MASS_SCALED <float>` — multiply drift step exponent argument
  by the scale: `drift_step(&mut e, dt * sham_mass_scale, g2)`.
- `SHAM_BACKTRACK_LOOP` — substrate's chosen degenerate-loop
  implementation (the `gamma_degenerate` loop in the registry is the
  Halcyon-side consumer).
- `SHAM_FROZEN_FIELD` — skip the U-update entirely in `drift_step`.

**Plus the two optional audit-story flags** (substrate-side
discretion; not Halcyon-side asks; not required for v3.1.3):
- `SHAM_EMPTY_LOOP` — GC₄ companion (no segments).
- `SHAM_OPEN_LOOP` — parser-rejection test.

**What this unblocks:** the orchestrator's per-sham `LoopTransport`
calls return clean data (mean ~ 0, sigma > 0) at the f64 level
Halcyon's Python gates can apply ε_abs = 10⁻¹⁰ against.

---

### 5. The bit-identity contract for per-seed reproducibility

**Where:** the gate doc, plus the test file.

**The contract:** "same `seeds` vec + same config (parameter pack,
loop, sham, n_discretization) → byte-identical `per_seed_holonomy`
GroupElements (component-by-component f64) AND byte-identical
diagnostic chains across re-runs."

This mirrors the IV.6 gold-gate. The orchestrator depends on it so
that the §3 verdict is reproducible across runs of the same protocol.

---

## When all 5 are done, Halcyon's single action is

Open `inertia_damping/run_holonomy_battery.py`, find `main()`, and
change:

```python
client = MockLoopTransportClient(scenario=args.mock_scenario)
```

to:

```python
client = LiveLoopTransportClient(...)  # whatever the binding looks like
```

The entire 35-test orchestrator suite should still pass against the
live client because the contract is `typing.Protocol` — structural
typing, no inheritance required.

Then:

```
python -m inertia_damping.run_holonomy_battery \
  --alpha 1.0 1000.0 \
  --output-dir inertia_damping/reports/holonomy_battery_v3_1_3/
```

The verdict (POSITIVE / NULL / AMBIGUOUS at each α) lands as a
`section_12_holonomy_battery_v3_1_3` sidecar JSON per v3.1.3 §7.2,
with every sidecar carrying the SPEC commit hash + Zenodo DOI at
the top.

---

## Where to look if you need the full design context

| What you want | Where it lives |
|---|---|
| The pre-registered SPEC (canonical) | `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.3.md` at git tag `spec-v3.1.3-zenodo-20785681` (commit `44c70b1`) |
| The Zenodo deposit | DOI [10.5281/zenodo.20785681](https://doi.org/10.5281/zenodo.20785681) |
| First contact letter (the original ask) | `inertia_damping/HALCYON_TO_GIGI_2026-06-20_HOLONOMY_VERB_REQUEST.md` |
| GIGI's v1 reply (rename + CC-LT questions) | `gigi/theory/halcyon/GIGI_TO_HALCYON_2026-06-20_SAMPLE_TRANSPORT_REPLY.md` |
| Halcyon v1 reply (per-CC answers + disambiguations) | `inertia_damping/HALCYON_TO_GIGI_2026-06-21_LOOP_TRANSPORT_REPLY.md` |
| GIGI v2 reply (refresh against v3.1.3 + two new substrate pins) | `gigi/theory/halcyon/GIGI_TO_HALCYON_2026-06-21_LOOP_TRANSPORT_REPLY_2.md` |
| **Halcyon design-phase closeout (operative)** | `inertia_damping/HALCYON_TO_GIGI_2026-06-21_DESIGN_CLOSEOUT.md` |
| The Halcyon-side scaffold (what's already built) | `inertia_damping/holonomy_battery/`, `inertia_damping/gigi_client/loop_transport.py`, `inertia_damping/gigi_client/mock_loop_transport.py`, `inertia_damping/run_holonomy_battery.py`, `inertia_damping/test_holonomy_battery.py` |

---

## What is NOT in this list (and why)

- **The α_Halcyon derivation from the Davis Field Equations** —
  v3.1.3 runs at α = 1 and α = 1000 without it. If the derivation
  lands before the substrate verb does, v3.1.3 §3.6 includes it as a
  third calibration; if not, the protocol runs at the two
  pre-registered values.
- **The §8.5 stopping-rule committee** — only assembled when the
  second NULL is recorded. Not a prerequisite for the first run.
- **A v3.1.4 amendment** — not anticipated. v3.1.3 is the deposited
  contract. Any change to §3 requires its own pre-registration with
  its own commit hash and its own Zenodo DOI.

---

## What you tell Gigi if she asks "what do you need from me?"

> The gate doc on disk first per the Sprint B lesson, then the verb
> with the v3.1.3 §4.4 grammar shape, then the GC₁–GC₆ acceptance
> battery green. Once those three are landed, my Python orchestrator
> swaps one line and runs. No other Halcyon-side blockers.
> Pre-registration commit is `44c70b1`, Zenodo DOI is
> `10.5281/zenodo.20785681`, git tag is `spec-v3.1.3-zenodo-20785681`.
> The design closeout letter at
> `inertia_damping/HALCYON_TO_GIGI_2026-06-21_DESIGN_CLOSEOUT.md`
> is the single source of truth for what I'm waiting on.
