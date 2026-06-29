# Zenodo Deposit Metadata — Halcyon Falsification Battery SPEC v3.1.3

**Purpose of this file.** Paste-ready metadata for the Zenodo deposit of
`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.3.md`. The Zenodo DOI minted
from this deposit becomes the publication-level pre-registration of the
v3.1.3 falsification criteria, to be cited in Solves Vol.\ 4
Appendix A.8 when the v3.1.3 protocol runs.

**Five-stage review history before deposit (all preserved):**

The reviews that produced v3.1.3 are *pre-deposit technical reviews* —
model-assisted reviews of the SPEC's mathematical content and protocol
executability, performed before deposit to catch defects in the
falsification criteria themselves. They are explicitly *not* a
substitute for human peer review, which is reserved for the §8.5
stopping-rule committee and any journal submission process.

- **v3.0** (`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md`, commit
  `0fe654d556e4f6878c439df64d1ff20599c9c733`): first-draft
  pre-registration. Round-1 pre-deposit technical review caught two
  mathematical defects (scalar holonomy vanishing by FTC; adiabaticity
  inequality reversed) and five protocol-discipline issues.
- **v3.1** (`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.md`, commit
  `712109488d43cf2fcd43b8d2bc8b5a1b053579ec`): patched the two
  mathematical defects and the five protocol issues. Round 2 caught
  seven executability issues.
- **v3.1.1** (`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.1.md`, commit
  `1165d63dbaffe30b55438cb82c1fa80aaf1f9ce0`): patched the seven
  executability issues. Round 3 caught the validity-window blocker
  (β_W range traversed below the SU(2) Q-observable's validated
  regime) plus three smaller patches (self-containedness, ε_abs
  rationale, NULL-branch sign-coherence).
- **v3.1.2** (`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.2.md`, commit
  `f4cfa1444a72e94c67f5cc2b7bfee51aeaf4666a`): patched the
  validity-window blocker (β_W now `[2.5, 3.0]`, inside the validated
  regime) and the three smaller issues. Round 4 caught three
  wording / audit-tightness issues (act-language distinction between
  pre-deposit technical review and human peer review; science-value
  gate on GC₅; substrate-gated τ_pin claim).
- **v3.1.3** (commit `44c70b1b76501b4b66c6f9ace6bccd8b5bd14c4a`,
  git tag `spec-v3.1.3-zenodo-20785681`): patched those three issues.
  The canonical pre-registered protocol. **DEPOSITED at Zenodo
  2026-06-21 as DOI [10.5281/zenodo.20785681](https://doi.org/10.5281/zenodo.20785681).**

The v3.0, v3.1, v3.1.1, and v3.1.2 drafts are preserved as first-class
artefacts in the repository for the chain of custody; the v3.1.3
commit hash is the document that goes to Zenodo as the cited
pre-registration.

## What to upload

All five SPEC files, as a single Zenodo deposit (Zenodo supports
multi-file uploads):

1. **`HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.3.md`** — the canonical
   pre-registered protocol (the actual contract; the SPEC §3
   falsification criteria committed to publication-level
   pre-registration).
2. `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.2.md` — fourth-draft,
   marked "fourth-draft pre-registration; superseded by v3.1.3 before
   deposit after round-4 pre-deposit technical review. Preserved for
   the chain of custody."
3. `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.1.md` — third-draft,
   marked "third-draft pre-registration; superseded by v3.1.2 before
   deposit after round-3 pre-deposit technical review."
4. `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.1.md` — second-draft,
   marked "second-draft pre-registration; superseded by v3.1.1 before
   deposit after round-2 pre-deposit technical review."
5. `HALCYON_FALSIFICATION_BATTERY_SPEC_v3.md` — first-draft,
   marked "first-draft pre-registration; superseded by v3.1 before
   deposit after round-1 pre-deposit technical review."

No accompanying figures, datasets, or supplementary code. The SPECs
are self-contained. The five-stage history demonstrates that
pre-registration with allowed-before-deposit correction is working:
each review pass caught real issues that a one-pass pre-registration
would have locked in.

## Recommended Zenodo form values

### Resource type
`Publication` → `Preprint` (or `Working paper` if Zenodo's form
distinguishes; the SPEC is a pre-registered protocol document,
publishable either way).

### Title
```
Halcyon Falsification Battery — SPEC v3.1.3 (Pre-Registration):
Closed-Loop Holonomy on the (Q, β_W) Control Manifold
within the Validated SU(2) Operating Regime
as the Load-Bearing Falsifier on the Buckyball Substrate
```

### Creators

| Name | Affiliation | ORCID |
|---|---|---|
| Bee Rosa Davis | Davis Geometric, Inc. | (fill from your record) |

### Publication date
2026-06-21

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
  <strong>v3.1.3 redesigns the measurement to match the framework's
  native observable and the apparatus's native protocol: closed-loop
  holonomy of a real connection 1-form on a two-dimensional programmed
  control manifold Λ = (Q, β_W), where Q is the surrogate sector
  coordinate and β_W is the Wilson gauge-action coupling with range
  [2.5, 3.0] strictly inside the SU(2) Q-observable regime that
  prior validation work has trusted. The holonomy is measured via
  discretized parallel transport.</strong>
  The control manifold has dimension ≥ 2 so non-trivial loops enclose
  finite area; the primary observable is the <em>antisymmetric</em>
  component H_geom = ½(H[γ] − H[γ⁻¹]) of the forward and reversed
  holonomies, with the symmetric H_sys reported as a systematic-offset
  diagnostic. The simulation delegates the substrate computation to the
  GIGI engine's <code>SAMPLE_TRANSPORT … ALONG_LOOP … CONTROL_MANIFOLD …
  ADIABATIC</code> verb (pending GIGI spec extension), making the v3.1.3
  audit surface a two-layer inversion: substrate correctness is reviewable
  through the engine's test suite plus a six-contract verb acceptance
  battery (GC₁–GC₆: flat-connection-zero, area-law, reversed-loop
  inversion, zero-size-zero, discretization convergence, gauge
  invariance); protocol design is reviewable in this SPEC.
</p>
<p>
  <strong>This deposit is a pre-registration, not a result.</strong>
  Section 3 specifies the numerical thresholds (POSITIVE / NULL / AMBIGUOUS),
  the five sham controls (S₄ folded into the antisymmetric primary
  observable), and the stopping rule (two independent measurement
  designs both returning NULL with external human peer review of
  independence (the §8.5 stopping-rule committee) constitutes
  simulation-level falsification, and the program does not iterate
  further without that human peer review). Section 8 commits to
  publication of whatever v3.1.3 returns, regardless of outcome, with the
  Solves Vol.&nbsp;4 chapter incorporating the result as Appendix A.8.
  The git commit hash on the GitHub repository
  (<code>nurdymuny/davis-wilson-map</code>, commit hash from v3.1.3's
  initial push) is the implementation-level pre-registration timestamp;
  this Zenodo DOI is the publication-level pre-registration.
</p>
<p>
  Predecessor SPECs (v2.0, v2.1) and the associated full-battery
  sidecars (<code>battery_fast_20260620_104846.json</code>,
  <code>battery_full_20260620_181227.json</code>,
  <code>battery_calibrated_20260621_011304.json</code>) are preserved as
  first-class artefacts in the repository, not deprecated. v2 becomes
  the adiabatic-limit control case for v3.1.3 in either result direction.
</p>
<p>
  Predecessor draft SPECs (v3.0, v3.1, v3.1.1, v3.1.2) are included in
  this deposit for transparency of the five-round review-and-patch
  process that produced v3.1.3. Each draft's §0 changelog names every
  patch and which round surfaced it. The reviews are *pre-deposit
  technical reviews* — model-assisted reviews of the SPEC's mathematical
  content and protocol executability, completed before deposit and
  distinct from human peer review (the latter is reserved for the §8.5
  stopping-rule committee).
</p>
<p>
  The methodological discipline that produced this pre-registration —
  writing the falsification criteria <em>before</em> any v3 implementation
  exists, naming the stopping rule, committing to publish all outcomes,
  iterating five times across pre-deposit technical review with every
  draft preserved — follows the standard set by the Halcyon program's
  pre-registered apparatus-side seven-gate kill chain and is documented
  in the SPEC's acknowledgments (§11). The cost of admission to either
  credibility — positive or negative — is the same: two independent
  measurements, publicly committed in advance, externally peer-reviewed
  by the named §8.5 human committee.
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
**Recommended: `PolyForm Noncommercial License 1.0.0`**

SPDX identifier: `PolyForm-Noncommercial-1.0.0`
Canonical URL: <https://polyformproject.org/licenses/noncommercial/1.0.0/>

Rationale: matches the canonical license at the repository root
(`LICENSE`). Permits academic citation, research extension, and
noncommercial reuse — including by educational institutions, public
research organizations, and government institutions — without
requiring further authorization. Commercial use of the protocol or
its derivatives requires separate licensing arranged via
`bee_davis@alumni.brown.edu`.

**If Zenodo's license picker does not list PolyForm-Noncommercial-1.0.0
directly:**

- First, search the Zenodo license picker for "PolyForm" — Zenodo
  imports the SPDX list and may have it under a slight variant.
- If still not present, choose `Other (Non-Open)` from the license
  dropdown and enter the canonical URL
  <https://polyformproject.org/licenses/noncommercial/1.0.0/> in the
  "License URL" / custom license field.
- In the deposit's "Additional notes" field, add the line:
  `License: PolyForm Noncommercial License 1.0.0
  (SPDX: PolyForm-Noncommercial-1.0.0).`

Why not CC-BY-4.0 / CC-BY-NC-4.0 for the SPEC text specifically: the
Creative Commons family is more idiomatic for prose documents, but
keeping a single canonical license across the project (code, SPECs,
papers, letters) reduces audit-trail confusion and matches what the
LICENSE file at the repository root already establishes. PolyForm-NC's
"noncommercial purposes" + "noncommercial organizations" clauses
cover the academic-citation use case explicitly, so the freedom-of-
citation rationale for CC-BY does not require a CC license.

### Related identifiers

| Relation | Identifier | Resource type |
|---|---|---|
| `is supplement to` | https://doi.org/10.5281/zenodo.17942784 | Publication / Preprint (Yang–Mills v6) |
| `is part of` | https://github.com/nurdymuny/davis-wilson-map (at the v3.1.3 commit hash; insert once known) | Software |
| `is documented by` | (the eventual Solves Vol. 4 v5 DOI when minted) | Publication |
| `references` | The v2.1 SPEC's own commit hash for v2 (look it up in `git log -- inertia_damping/HALCYON_FALSIFICATION_BATTERY_SPEC.md`) | Software / Other |
| `references` | https://doi.org/10.5281/zenodo.20438796 | Publication (Geometric Encryption — establishes the bundle/connection framing) |

### Funding (optional)
Davis Geometric, Inc. internal R&D.

### Notes (form's "Additional notes" field)

```
This is a pre-registered protocol specification. The simulation has not
been run against v3.1.3 at the time of deposit. The deposit's purpose
is to fix the falsification criteria of §3 in the public record before
any v3 implementation exists, distinguishing honest measurement-design
redesign from post-hoc protocol-shopping. Pre-registration follows the
standard set by the Halcyon program's apparatus-side seven-gate kill
chain.

Five review iterations completed before deposit:
- Gigi's methodological intervention establishing pre-registration discipline
- Round 1 (pre-deposit technical review): caught the two math defects
  (scalar holonomy = 0 by FTC; reversed adiabaticity inequality)
- Round 2: caught seven executability issues
- Round 3: caught the β_W validity-window blocker (range tightened to
  [2.5, 3.0] from [2.0, 3.0] to stay inside the SU(2) Q-observable
  validated regime), plus three smaller patches (self-containedness,
  ε_abs rationale, NULL-branch sign-coherence)
- Round 4: caught three wording / audit-tightness issues
  (act-language distinction between pre-deposit technical review and
  human peer review; science-value gate on GC₅; substrate-gated τ_pin
  claim)

The rounds are *pre-deposit technical reviews* — model-assisted
reviews of the SPEC's mathematical content and protocol executability,
performed before deposit. They are not a substitute for human peer
review, which is reserved for the §8.5 stopping-rule committee and
any subsequent journal submission process.

All four predecessor draft SPECs (v3.0, v3.1, v3.1.1, v3.1.2) are
included in this deposit for chain-of-custody transparency. v3.1.3 is
the canonical contract.

Companion SPEC (Halcyon → GIGI verb request) lives in the repository at
inertia_damping/HALCYON_TO_GIGI_2026-06-20_HOLONOMY_VERB_REQUEST.md.

The v3.1.3 git commit hash (insert when known) is the canonical
implementation-level timestamp.
```

## After the Zenodo DOI is minted (executed 2026-06-21)

1. ✅ Added a `Zenodo DOI:` callout near the top of the v3.1.3 SPEC
   file referencing [10.5281/zenodo.20785681](https://doi.org/10.5281/zenodo.20785681).
2. ✅ Committed and pushed that single-line edit. The post-deposit
   commit is *not* part of the pre-registration — it merely records
   the DOI pointer. Canonical pre-registration commit `44c70b1`
   tagged `spec-v3.1.3-zenodo-20785681` for permanent reference.
3. ⏳ Solves Vol.\ 4 Appendix A.8 (when the v3.1.3 run lands) will
   cite the Zenodo DOI as the pre-registration record. Open until
   the v3.1.3 protocol runs and the chapter v5 is published.

## What this deposit is NOT

- It is not a results paper. The protocol has not run.
- It is not a software release. The Python orchestrator does not exist
  yet.
- It is not a deprecation of v2. v2 remains a first-class artefact.
- It is not a substitute for peer review. The SPEC has not been
  reviewed by anyone outside the program (the GPT reviews are useful
  but are not a substitute for peer review of the substantive
  scientific content).

## What this deposit IS

A timestamped, publicly accessible, citable commitment to:

1. The falsification criteria of §3 (POSITIVE / NULL / AMBIGUOUS
   thresholds, five sham controls, stopping rule).
2. The publication policy of §8 (publish all outcomes, regardless of
   direction).
3. The architectural choice of §2 (holonomy as the load-bearing
   observable, GIGI as the substrate audit surface, β_W ∈ [2.5, 3.0]
   as the validated operating window).

These three commitments cannot be edited after the deposit without
breaking the pre-registration chain. Any subsequent SPEC document that
deviates from them is, by construction, a different protocol and
requires its own pre-registration.
