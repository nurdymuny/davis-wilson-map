# Halcyon → GIGI design-phase closeout (2026-06-21)

**From:** Halcyon team (Bee + Claude)
**To:** GIGI engine team (Bee + Claude)
**Subject:** v2 accepted in full. No pushback. Design phase closes here; the next interactions are gate-doc review, parser-arm review, then the Halcyon Python orchestrator implementation. Three small Halcyon-side gate-application notes that v2 didn't have to call out but are worth naming for the gate doc.
**In reply to:** `GIGI_TO_HALCYON_2026-06-21_LOOP_TRANSPORT_REPLY_2.md`.
**Predecessors:**
- `HALCYON_TO_GIGI_2026-06-20_HOLONOMY_VERB_REQUEST.md` (first contact)
- `GIGI_TO_HALCYON_2026-06-20_SAMPLE_TRANSPORT_REPLY.md` (substrate v1)
- `HALCYON_TO_GIGI_2026-06-21_LOOP_TRANSPORT_REPLY.md` (Halcyon v1)
- `GIGI_TO_HALCYON_2026-06-21_LOOP_TRANSPORT_REPLY_2.md` (substrate v2 — being acknowledged here)

This is the design-phase closeout from the Halcyon side. After this letter, the next things on disk are substrate-side artifacts (gate doc, then implementation) and Halcyon-side artifacts (Zenodo deposit, then Python orchestrator). The cross-team letter stream stays as the audit trail.

---

## Letter

GIGI —

v2 lands cleanly. No clause warrants pushback. The five updates to v1 commitments are all the right move:

1. Pre-registration anchor `0fe654d` → `44c70b1` + the imminent Zenodo DOI: yes. v3.0 was the first draft; v3.1.3 is the contract. The five-round chain (Gigi's methodological intervention + four pre-deposit technical review rounds) caught real defects each time. Your sharpening of v1's methodology framing is correct: every round catching something is the property pre-registration is *supposed to enable*, not evidence that the process is broken. v1's praise was directional; v2's sharpening is exact. Accepted.
2. LOOP_TRANSPORT lock: yes. v1 already adopted the rename in §B.1; v2 records it as final. Halcyon does not re-open path (b) (target-type dispatcher on the SPEC's `ALONG_LOOP CONTROL_MANIFOLD` suffix) — your documentation of it as "the alternative we would have built" is the right archival shape.
3. Five sham controls S₁–S₅ + six GC contracts GC₁–GC₆ disambiguation: yes. v1's "six sham flags" framing was the conflation v2 corrected. The two surfaces are independent: GC₁–GC₆ gate verb introduction (substrate correctness); S₁–S₅ gate Halcyon-protocol science-call interpretation (sham controls). Both ship in separate test files (`tests/halcyon_part_vi_loop_transport_gc.rs` and `tests/halcyon_part_vi_loop_transport.rs`). The gate doc carrying both tables side-by-side is the right discipline so they cannot be conflated again.
4. Tuple return shape `{ h_geom, h_sys, ... }` from `COMPUTE HOLONOMY`: yes. The antisymmetric primary observable `H_geom = ½(H[γ] − H[γ⁻¹])` and the symmetric systematic-offset diagnostic `H_sys = ½(H[γ] + H[γ⁻¹])` are both load-bearing per v3.1.3 §3.1. Building both into the verb's return shape (not a query-side derivation, not an option flag) is what makes the S₄ reversed-loop sham assertion unfakeable by construction. The `LoopTransportDiagnostics` struct shape in your §5 reads correctly against v3.1.3 §4.6's orchestrator-thin-wrapper contract.
5. β_W ∈ [2.5, 3.0] strict parser validation: yes. Parser-level enforcement (cannot be bypassed in Python) is the right shape for the substrate-side gate. The convenient inheritance from Halcyon's canonical thermalization β = 2.5 is a real win — gauge regime, Q-tracking validation receipts, and bit-identity-compatible RNG state on shared seeds all transfer cleanly. β_W < 2.5 routes through the v3.1.x amendment door with independent validation attached; Halcyon-side, this means we don't try to push past it without paying the cost.

CC-LT-7 (loop time-reversal mechanism) and CC-LT-8 (per-axis `ramp_rate`): accepted as substrate-side pins, not questions back. The shape v2 names is the same shape v3.1.3 §4.1 and §4.4 commit to:

- One `DECLARE LOOP` per logical loop in the `LoopRegistry`. Two declared loops for v3.1.3: `gamma_unit` (closed rectangle in (Q, β_W)) and `gamma_degenerate` (zero-area, mapped to `SHAM_BACKTRACK_LOOP` per v1 §D.1). `γ_unit⁻¹` is **not** separately declared; the substrate's executor traverses `gamma_unit` time-reversed. The WAL gets one `DeclareLoop` entry per logical loop, with the reverse-traversal as an inline option on the `LoopTransport` WAL entry. This is the right shape; it keeps the H_geom antisymmetric primary observable cheap (single loop registration, two walks, two combinations).
- `LoopShape::PiecewiseLinear { vertices: Vec<(f64, f64)>, t_per_segment: f64 }` for v0.1 is correct and sufficient. v3.1.3's four-segment rectangle is the v0.1 consumer. Future curved-loop variants (`Circular`, `BezierClosed`) plug in when a different consumer needs them; not v0.1 work.

Pre-registration reciprocation: the substrate's three-constraint contract (gauge-invariant observable, local per-step updates, no-tunable-tolerance analytical target) is the independent referee on the GIGI side. Halcyon's `44c70b1` plus the imminent Zenodo DOI is the independent referee on the Halcyon side. The two clocks are locked separately. Pre-registration's intended property is doing what it's supposed to do — your sharpening of this in v2 reads correctly.

Three small Halcyon-side gate-application notes follow (§A). The §B section names what Halcyon is committing to next. §C closes the loop on the design phase from this side.

—Bee + Claude

---

## §A — Three small Halcyon-side gate-application notes for the gate doc

These are not pushback. They are details v2 didn't explicitly call out because each one is the *Halcyon Python orchestrator's* responsibility, not the substrate's. They are listed here so the gate doc captures them on the audit-story side.

### A.1 `ε_abs = 10⁻¹⁰` is a Halcyon-side gate on the sham f64 outputs

Per v3.1.3 §3.2 and §5, four of the five sham controls carry an absolute-ε floor of `10⁻¹⁰`:

- S₁ (flat field): `|H_S₁| < 2σ_S₁` AND `|H_S₁| < 10⁻¹⁰`
- S₂ (α=0): `|H_S₂| < 10⁻¹⁰` (load-bearing; 2σ is sanity)
- S₅ (degenerate loop / BACKTRACK_LOOP): `|H_S₅| < 2σ_S₅` AND `|H_S₅| < 10⁻¹⁰`
- S₆ (frozen field): `|H_S₆| < 2σ_S₆` AND `|H_S₆| < 10⁻¹⁰`
- S₃ (mass scaled): in the NULL/AMBIGUOUS branch, `|H_S₃ at μ_baseline=1| < 2σ_S₃` AND `< 10⁻¹⁰`

The pattern matches Observable A (τ_pin / T_segment) and the tracking-error gates: **substrate emits the f64 holonomy values; Halcyon's Python applies the `< 10⁻¹⁰` comparison**. This keeps the substrate's three-constraint contract clean — no `epsilon_abs` parameter inside the verb call, no operator-tunable tolerance baked into the substrate code. The `10⁻¹⁰` value is pre-registered in v3.1.3 §3.2 as the empirical operating floor validated by GC₂ + GC₆; if a future v3.1.x amendment moves it, the threshold moves in the SPEC, not in the substrate.

Worth noting in the gate doc because the substrate's GC contracts test against machine-ε (for GC₁ and GC₄ and GC₆) whereas Halcyon's sham gates test against `10⁻¹⁰`. The substrate's GC tests are stricter than Halcyon's sham gates by ~6 orders of magnitude; that is intentional, not a discrepancy. The substrate audits *its own correctness* at machine-ε; Halcyon audits the *protocol's observability floor* at the empirical numerical regime where the substrate has been validated to operate.

### A.2 Per-seed sign-coherence rule lives in Halcyon's Python, not in the substrate

Per v3.1.3 §3.5, POSITIVE classification requires ≥ 5/8 of the primary-loop seeds to share the sign of `H_geom_mean`. NULL has no sign-coherence requirement (random signs are expected in a true null). AMBIGUOUS triggers if sign-coherence fails in the 1σ–5σ range.

The substrate's `LoopTransportDiagnostics` exposes the per-seed inputs:

```rust
pub per_seed_h_forward: Vec<GroupElement>,
pub per_seed_h_reversed: Vec<GroupElement>,
```

Halcyon's Python forms `per_seed_h_geom[i] = ½(per_seed_h_forward[i] - per_seed_h_reversed[i])`, takes the sign (or the abelianized scalar projection of each `GroupElement`), counts the majority, and applies the 5/8 rule. **The substrate does not compute the sign-coherence statistic**; it emits the per-seed vectors and Halcyon's orchestrator applies the rule.

This keeps the substrate stateless about Halcyon's gate structure and keeps the rule in the SPEC where it can be amended (with a new pre-registration) without substrate code changes. Same pattern as ε_abs and the τ_pin / T_segment threshold.

### A.3 Tracking-error gates use the substrate's emitted `max` values

v3.1.3 §4.3 pre-registers `ε_Q = 0.05` and `ε_β_W = 0.05` as tracking-error tolerances. The `LoopTransportDiagnostics` struct in your §5 has the right fields:

```rust
pub tracking_error_max_q: f64,
pub tracking_error_max_beta_w: f64,
```

These are the per-axis `max_t` values the substrate computes during the loop traversal. Halcyon's Python applies the comparisons `tracking_error_max_q < 0.05` and `tracking_error_max_beta_w < 0.05`. A failure on either axis forces AMBIGUOUS regardless of H values. Same pattern again: substrate emits f64; Halcyon applies the pre-registered threshold.

One small substrate-side ask that v2 already covered implicitly: the gate doc should specify how `tracking_error_max_q` is computed — Halcyon's read of v3.1.3 §4.3 is the supremum of `|Q_surrogate(t) - Q_target(t)|` over all substeps in the loop traversal, where `Q_surrogate(t)` is the substrate's surrogate sector reading at substep `t` and `Q_target(t)` is the value of the piecewise-linear ramp at that substep. If the substrate's computation is the same (or a known refinement, e.g., the L² norm over substeps), the gate doc names which one v0.1 uses. Halcyon's Python applies the same threshold regardless; this is purely an audit-trail clarification.

---

## §B — What Halcyon commits to next

Updated for the design-phase closeout:

1. **Zenodo deposit of v3.1.3** with v3.0, v3.1, v3.1.1, v3.1.2 attached for chain-of-custody transparency. Paste-ready metadata at `inertia_damping/HALCYON_SPEC_v3_ZENODO_METADATA.md`. The Zenodo DOI when minted becomes the publication-level pre-registration; a post-deposit single-line commit adds the `Zenodo DOI:` pointer to the top of `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.3.md`. That post-deposit commit is *not* part of the pre-registration — it merely records the DOI for cross-reference.
2. **Read the gate doc when it lands** at `theory/halcyon/HALCYON_PART_VI_GATES.md`. Halcyon will check it against v3.1.3 §7.4's GC₁–GC₆ table for the substrate-correctness gates and against v3.1.3 §5 for the sham-control assertions. If the gate doc reads cleanly, Halcyon nods and stays out of the substrate's lane.
3. **Read the parser arm when it lands.** Halcyon will check the `LoopTransport` AST shape against v3.1.3 §4.4's `RETURN` clause and the `LoopTransportDiagnostics` struct against §5's tuple return. If the parser arm reads cleanly, Halcyon nods.
4. **Begin Halcyon Python orchestrator implementation** (`run_holonomy_battery.py`) when the verb is callable and GC₁–GC₆ are green. The orchestrator is a thin delegation wrapper per v3.1.3 §4.6 — no substrate-relevant logic, just loop construction (`DECLARE LOOP` statements for `gamma_unit` and `gamma_degenerate`), verb calls (per α-calibration × 8 seeds × forward+reverse × 5 sham variants), gate application (the three patterns in §A above), and sidecar emission per v3.1.3 §7.2.
5. **Stopping-rule committee assembly is deferred.** Per v3.1.3 §3.3 and §8.5, the human external review committee (Gigi + one external lattice-gauge-theory peer reviewer + one journal peer reviewer) assembles only when the second NULL is recorded, not in advance. If v3.1.3's verdict comes back POSITIVE or AMBIGUOUS, the committee is not invoked.

**Halcyon does NOT commit to:**

- A turnaround date for the Python orchestrator. It ships when the verb is callable and GC₁–GC₆ are green.
- A v3.1.3 outcome. POSITIVE, NULL, AMBIGUOUS are all publishable per v3.1.3 §8.2.
- A v3.1.4 amendment to lower β_W's range below 2.5. Future protocols that need it come back through the amendment door with independent validation receipts.

---

## §C — Design phase closes here

Six letter exchanges (three substrate-side, three Halcyon-side) over two calendar days have pinned every cross-cutting design question for the v0.1 verb shape:

| # | Date | From | Subject |
|---|---|---|---|
| 1 | 2026-06-20 | Halcyon | First contact: the five-piece ask (verb, sham flags, adiabaticity check, per-seed independence, regression test) |
| 2 | 2026-06-20 | GIGI | v1 reply: rename to LOOP_TRANSPORT, six CC-LT questions, scope review per ask |
| 3 | 2026-06-21 | Halcyon | v1 reply to v1: pre-registration commit hash update, CC-LT-1 through CC-LT-6 answers, three disambiguations |
| 4 | 2026-06-21 | GIGI | v2 reply: five v1 commitments updated, two new substrate-side pins (CC-LT-7, CC-LT-8), no new questions back |
| 5 | 2026-06-21 | Halcyon | **This letter:** design-phase closeout, three §A gate-doc notes, §B Halcyon-side commitments |

After this letter, the next artifacts on disk are *substrate-side build* (gate doc, then parser arm + executor + GC test file) and *Halcyon-side deposit* (Zenodo + post-deposit DOI-pointer commit). The cross-team letter stream is the audit trail; from here, each side ships against its own clock per the two-clocks methodology.

If a substantive issue surfaces during implementation that requires the design phase to re-open, either side can write to the other. The convention is the same: pin the question, document the resolution, commit the letter, then proceed with the LOC change. Re-opening the design phase on a non-substantive issue costs both sides more than just writing the LOC; the letter chain is not the place to relitigate.

The two-clocks methodology stays the operative discipline. Substrate timeline does not move the pre-registration; pre-registration does not move the substrate timeline. Five rounds of review against v3.0 → v3.1.3 demonstrated the property is real. Both clocks are locked separately; both are ticking now against their own contracts.

Pushback welcome on any clause in this letter, but the substantive design questions are resolved. Anything that surfaces now is in the noise; v0.1 LOC is the next signal.

—Bee + Claude

---

## Companion-letter index (operative state after this letter)

| Document | Path | Status |
|---|---|---|
| **v3.1.3 SPEC** (canonical pre-registration) | `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.3.md` at commit `44c70b1` | Locked; awaiting Zenodo deposit |
| v3.0 / v3.1 / v3.1.1 / v3.1.2 SPECs | `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC_v3*.md` | Preserved drafts; chain of custody |
| v2.0 / v2.1 SPEC | `inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC.md` | Adiabatic-limit predecessor; not deprecated |
| Halcyon → GIGI letter 1 (first contact) | `inertia_damping/HALCYON_TO_GIGI_2026-06-20_HOLONOMY_VERB_REQUEST.md` | Superseded by Halcyon v1 reply (operative) but preserved |
| Halcyon → GIGI letter 2 (Halcyon v1 reply) | `inertia_damping/HALCYON_TO_GIGI_2026-06-21_LOOP_TRANSPORT_REPLY.md` | Operative until v2 (this letter) supersedes it for design-phase closeout |
| **Halcyon → GIGI letter 3** (design-phase closeout) | `inertia_damping/HALCYON_TO_GIGI_2026-06-21_DESIGN_CLOSEOUT.md` | **Operative.** This document. |
| Zenodo metadata | `inertia_damping/HALCYON_SPEC_v3_ZENODO_METADATA.md` | Paste-ready for v3.1.3 deposit |
| GIGI → Halcyon letter 1 (substrate v1) | `gigi/theory/halcyon/GIGI_TO_HALCYON_2026-06-20_SAMPLE_TRANSPORT_REPLY.md` | Superseded by GIGI v2 (operative) |
| GIGI → Halcyon letter 2 (substrate v2) | `gigi/theory/halcyon/GIGI_TO_HALCYON_2026-06-21_LOOP_TRANSPORT_REPLY_2.md` | Operative substrate-side ask document |

After this letter, both teams' operative documents are stable. The next disk artifacts are implementation, not design.
