# Zenodo Deposit Metadata — Halcyon Falsification Battery SPEC v3.1

**Purpose of this file.** Paste-ready metadata for the Zenodo deposit of
`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.md` (commit hash TBD —
the v3.1 commit hash supersedes v3.0's `0fe654d` as the canonical
pre-registration). The Zenodo DOI minted from this deposit becomes the
publication-level pre-registration of the v3.1 falsification criteria,
to be cited in Solves Vol.\ 4 Appendix A.8 when the v3.1 protocol runs.

**v3.0 vs v3.1.** The v3.0 SPEC (`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md`,
commit `0fe654d`) was the first-draft pre-registration. Before its Zenodo
deposit, external review caught two mathematical defects (scalar holonomy
vanishing by FTC; adiabaticity inequality reversed) and five
protocol-discipline issues. v3.1 patches them. The v3.0 draft is preserved
as a first-class artefact in the repository for the chain of custody;
the v3.1 commit hash is the document that goes to Zenodo.

## What to upload

Both files, as a single Zenodo deposit (Zenodo supports multi-file
uploads):

1. `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.md` — the canonical
   pre-registered protocol (the actual contract).
2. `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md` — the v3.0 first-draft,
   included for transparency of the review-and-patch process. Mark as
   "v3.0 — superseded by v3.1 before deposit; preserved for the
   pre-registration chain of custody."

No accompanying figures, datasets, or supplementary code. The SPECs
are self-contained.

## Recommended Zenodo form values

### Resource type
`Publication` → `Preprint` (or `Working paper` if Zenodo's form
distinguishes; the SPEC is a pre-registered protocol document,
publishable either way).

### Title
```
Halcyon Falsification Battery — SPEC v3.1 (Pre-Registration):
Closed-Loop Holonomy on a Two-Dimensional Control Manifold
as the Load-Bearing Falsifier on the Buckyball Substrate
```

### Creators

| Name | Affiliation | ORCID |
|---|---|---|
| Bee Rosa Davis | Davis Geometric, Inc. | (fill from your record) |

### Publication date
2026-06-20

### Description (paste into the "Description" field, supports HTML)

```html
<p>
  This document is the pre-registered protocol specification for v3 of the
  Halcyon Falsification Battery — a simulation-side falsifiability harness
  for the experimental claim that programmed gauge sectors of a buckyball
  Josephson-junction substrate modify the dynamic inertial response of a
  test mass at fixed gravitational load. v3 supersedes v2's fixed-Q lock-in
  protocol after the v2 full-battery run returned <code>FAIL_SIGNAL_MISSING</code>
  in observation space across two calibration scans, with a diagnostic
  signature — α-coupling scaled 1000× while signal-to-noise scaled only
  2.5× — indicating that the noise was intrinsic to the gauge-field
  dynamics being measured, not extrinsic to the lock-in apparatus. The
  architectural diagnosis is that the apparatus runs in the diabatic regime
  (continuous Q-drive via the cage) while v2 measured the adiabatic limit
  (fixed-Q lock-in).
</p>
<p>
  <strong>v3.1 redesigns the measurement to match the framework's native
  observable and the apparatus's native protocol: closed-loop holonomy of
  a real connection 1-form on a two-dimensional programmed control
  manifold Λ = (Q, θ), measured via discretized parallel transport.</strong>
  The control manifold has dimension ≥ 2 so non-trivial loops enclose
  finite area; the primary observable is the <em>antisymmetric</em>
  component H_geom = ½(H[γ] − H[γ⁻¹]) of the forward and reversed
  holonomies, with the symmetric H_sys reported as a systematic-offset
  diagnostic. The simulation delegates the substrate computation to the
  GIGI engine's <code>SAMPLE_TRANSPORT … ALONG_LOOP … CONTROL_MANIFOLD …
  ADIABATIC</code> verb (pending GIGI spec extension), making the v3.1
  audit surface a two-layer inversion: substrate correctness is reviewable
  through the engine's test suite plus a six-contract verb acceptance
  battery (GC₁–GC₆: flat-connection-zero, area-law, reversed-loop
  inversion, zero-size-zero, discretization convergence, gauge
  invariance); protocol design is reviewable in this SPEC.
</p>
<p>
  <strong>This deposit is a pre-registration, not a result.</strong>
  Section 3 specifies the numerical thresholds (POSITIVE / NULL / AMBIGUOUS),
  the six sham controls, and the stopping rule (two independent measurement
  designs both returning NULL with external review of independence
  constitutes simulation-level falsification, and the program does not
  iterate further without external review). Section 8 commits to
  publication of whatever v3 returns, regardless of outcome, with the
  Solves Vol.&nbsp;4 chapter incorporating the result as Appendix A.8.
  The git commit hash on the GitHub repository
  (<code>nurdymuny/davis-wilson-map</code>,
  <code>0fe654d556e4f6878c439df64d1ff20599c9c733</code>) is the
  implementation-level pre-registration timestamp; this Zenodo DOI is
  the publication-level pre-registration.
</p>
<p>
  Predecessor SPECs (v2.0, v2.1) and the associated full-battery
  sidecars (<code>battery_fast_20260620_104846.json</code>,
  <code>battery_full_20260620_181227.json</code>,
  <code>battery_calibrated_20260621_011304.json</code>) are preserved as
  first-class artefacts in the repository, not deprecated. v2 becomes
  the adiabatic-limit control case for v3 in either result direction.
</p>
<p>
  The methodological discipline that produced this pre-registration —
  writing the falsification criteria <em>before</em> any v3 implementation
  exists, naming the stopping rule, committing to publish all outcomes —
  follows the standard set by the Halcyon program's pre-registered
  apparatus-side seven-gate kill chain and is documented in the SPEC's
  acknowledgments (§11). The cost of admission to either credibility —
  positive or negative — is the same: two independent measurements,
  publicly committed in advance, externally reviewed.
</p>
```

### Keywords (semicolon-separated)
```
pre-registration; falsifiability; lattice gauge theory; Yang–Mills;
buckyball substrate; truncated icosahedron; holonomy; Wilson loop;
Davis Field Equations; inertia coupling; gauge sectors; SU(2);
Halcyon; Davis Geometric; methodological discipline; reproducibility
```

### Communities
- `geometric-physics` (if available)
- Any others you maintain on Zenodo (e.g. Davis Geometric collection
  if you have one)

### License
**Recommended: `Creative Commons Attribution 4.0 International (CC-BY-4.0)`**

Rationale: pre-registration documents are most useful when they can be
cited and discussed freely. CC-BY-4.0 is the academic-standard choice
for protocol pre-registrations. If you prefer non-commercial
restrictions for consistency with the inertia_damping repo's
non-commercial-default, use `CC-BY-NC-4.0`, but note that this slightly
restricts who can cite it in industry research contexts.

### Related identifiers

| Relation | Identifier | Resource type |
|---|---|---|
| `is supplement to` | https://doi.org/10.5281/zenodo.17942784 | Publication / Preprint (Yang–Mills v6) |
| `is part of` | https://github.com/nurdymuny/davis-wilson-map/tree/0fe654d556e4f6878c439df64d1ff20599c9c733 | Software (the davis-wilson-map repo at the pre-registration commit) |
| `is documented by` | (the eventual Solves Vol. 4 v5 DOI when minted) | Publication |
| `references` | The v2.1 SPEC's own commit hash for v2 (look it up in `git log -- inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC.md`) | Software / Other |
| `references` | https://doi.org/10.5281/zenodo.20438796 | Publication (Geometric Encryption — establishes the bundle/connection framing) |

### Funding (optional)
Davis Geometric, Inc. internal R&D.

### Notes (form's "Additional notes" field)

```
This is a pre-registered protocol specification. The simulation has not
been run against v3 at the time of deposit. The deposit's purpose is
to fix the falsification criteria of §3 in the public record before any
v3 implementation exists, distinguishing honest measurement-design
redesign from post-hoc protocol-shopping. Pre-registration follows the
standard set by the Halcyon program's apparatus-side seven-gate kill
chain.

Companion SPEC (Halcyon → GIGI verb request) lives in the repository at
inertia_damping/HALCYON_TO_GIGI_2026-06-20_HOLONOMY_VERB_REQUEST.md.

The git commit hash 0fe654d556e4f6878c439df64d1ff20599c9c733 is the
canonical timestamp.
```

## After the Zenodo DOI is minted

1. Add a `Zenodo DOI:` line near the top of the SPEC file referencing
   the new DOI.
2. Commit and push that single-line edit. The post-deposit commit is
   *not* part of the pre-registration — it merely records the DOI
   pointer.
3. Solves Vol.\ 4 Appendix A.7.2 (or v5 Appendix A.8 when the v3 run
   lands) cites the Zenodo DOI as the pre-registration record.

## What this deposit is NOT

- It is not a results paper. The protocol has not run.
- It is not a software release. The Python orchestrator does not exist
  yet.
- It is not a deprecation of v2. v2 remains a first-class artefact.
- It is not a substitute for peer review. The SPEC has not been
  reviewed by anyone outside the program.

## What this deposit IS

A timestamped, publicly accessible, citable commitment to:

1. The falsification criteria of §3 (POSITIVE / NULL / AMBIGUOUS
   thresholds, six sham controls, stopping rule).
2. The publication policy of §8 (publish all outcomes, regardless of
   direction).
3. The architectural choice of §2 (holonomy as the load-bearing
   observable, GIGI as the substrate audit surface).

These three commitments cannot be edited after the deposit without
breaking the pre-registration chain. Any subsequent SPEC document that
deviates from them is, by construction, a different protocol and
requires its own pre-registration.
