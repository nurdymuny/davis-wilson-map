// synthesis_workflow.js — cross-channel + multi-L + cross-group synthesis of
// halcyon_ym4_glueball_demo after the SU(3) GPU sweep lands.
//
// Invoke:
//   Workflow({ scriptPath:
//     "C:\\Users\\nurdm\\OneDrive\\Documents\\davis-wilson-lattice\\inertia_damping\\scripts\\synthesis_workflow.js"
//   })

export const meta = {
  name: 'ym_mass_gap_synthesis_after_su3',
  description: 'Cross-channel + multi-L + cross-group synthesis of halcyon_ym4_glueball_demo, producing a paper-readiness verdict.',
  phases: [
    { title: 'Snapshot' },
    { title: 'CrossChannel' },
    { title: 'MultiL' },
    { title: 'CrossGroup' },
    { title: 'PaperReadiness' },
  ],
}

const BUNDLE = 'halcyon_ym4_glueball_demo'
const GIGI_URL = 'http://localhost:3142/v1/gql'
const REPORT_PATH = 'C:\\Users\\nurdm\\OneDrive\\Documents\\davis-wilson-lattice\\inertia_damping\\reports\\PAPER_READINESS_AFTER_SU3.md'
const FRAMING_DOC = 'gigi/theory/YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md'

const SCHEMA_SNAPSHOT = {
  type: 'object',
  properties: {
    n_ensembles: { type: 'integer' },
    rows_by_group_L_beta: { type: 'array', items: { type: 'object' } },
    notes: { type: 'string' },
  },
  required: ['n_ensembles', 'notes'],
  additionalProperties: false,
}

const SCHEMA_LENS = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['strong', 'mixed', 'weak'] },
    summary: { type: 'string' },
    specific_findings: { type: 'array', items: { type: 'string' } },
    concerns_or_flags: { type: 'array', items: { type: 'string' } },
  },
  required: ['verdict', 'summary'],
  additionalProperties: false,
}

phase('Snapshot')

const snap = await agent(
  `Pull the current bundle state from gigi. Use Bash to run a curl POST against ${GIGI_URL}
with the query:

  SELECT ensemble_id, gauge_group, dimension, L, beta, n_configurations,
         plateau_fit_mass, plateau_fit_error, plateau_fit_t_lo, plateau_fit_t_hi,
         sigma_creutz_22, sigma_creutz_22_error,
         sigma_creutz_32, sigma_creutz_32_error,
         m_eff_null_t, P_bar_t, measurement_channel
  FROM ${BUNDLE}
  WHERE t = 0;

Parse the response JSON and report:
  - n_ensembles (count of distinct ensemble_id values)
  - rows_by_group_L_beta: one row per ensemble with ALL fields above
  - notes: any anomalies (NaN sigma, sentinel -9999.0 values, borderline nulls)

If gigi is unreachable, fall back to reading these on-disk receipts:
  inertia_damping/reports/ym4_glueball_in_gigi_receipt.json
  inertia_damping/reports/ym4_su3_glueball_in_gigi_receipt.json
  inertia_damping/reports/ym4_su3_dec2025_configs_receipt.json

Report under 300 words plus the rows array. Do not analyze — that's the next phases' job.`,
  { label: 'snapshot', schema: SCHEMA_SNAPSHOT, agentType: 'Explore' }
)

phase('CrossChannel')

const crossChannel = await parallel([
  () => agent(
    `Cross-channel agreement audit on the SU(2) ensembles.

For each (L, beta) compute the ratio m_g(M1) / sqrt(sigma_creutz_22) and compare against
the Teper 1998 SU(2) continuum value m_0++/sqrt(sigma) ~ 3.7.

A ratio close to 3.7 means the M1 plateau is dominated by the lowest 0++ state. A ratio
much larger (say > 6) means the unsmeared M1 has heavy excited-state contamination —
expected without APE smearing, but worth flagging per (L, beta).

Snapshot data:
${JSON.stringify(snap, null, 2)}

Report verdict (strong/mixed/weak), per-(L,beta) ratios in specific_findings, and any
concerns_or_flags about M1 instability. Note explicitly that without M2 (smearing) the
expected ratio drift is well-documented; do not call M1 broken.`,
    { label: 'cc-su2-ratio', schema: SCHEMA_LENS }
  ),

  () => agent(
    `M3 string-tension asymptotic-freedom + magnitude check.

For SU(2): verify sigma_creutz_22(beta) is MONOTONIC DECREASING in beta (asymptotic-freedom).
For SU(3): same check + compare published Bali-Schilling 1992 chi(2,2) values at the
canonical betas (beta=6.0 -> ~0.193, beta=6.2 -> ~0.16, beta=6.4 -> ~0.12).

CRITICAL CHECK: the fresh GPU SU(3) sweep at L=8 with n_therm=150 may be UNDER-THERMALIZED
because Cabibbo-Marinari decorrelates slower than SU(2). Compare:
  - Dec 2025 Modal SU(3) at L=8 beta=6.0: sigma ~0.1985 (published: ~0.193, MATCH)
  - Fresh GPU SU(3) at L=8 beta=6.0: report value vs published
If the fresh value is >20% high, FLAG as under-thermalized.

Snapshot data:
${JSON.stringify(snap, null, 2)}

Report verdict, specific per-beta findings, and any concerns. Be specific about the
under-thermalization risk on the fresh GPU SU(3) sweep.`,
    { label: 'cc-asymptotic-freedom', schema: SCHEMA_LENS }
  ),

  () => agent(
    `Falsification Criterion 7 status across all ensembles.

For each row, classify the null control:
  PASS  = m_eff_null is NaN or sentinel (-9999) at every t (clean kill)
  PARTIAL = m_eff_null has a finite value at one or two t's but well outside the real plateau range
  BORDERLINE = m_eff_null finite plateau lies within (~2 sigma of) the real plateau
  FAIL = m_eff_null plateau indistinguishable from real

The Dec 2025 SU(3) L=16 ensemble is unthermalized (hot-start) and pushed as a known-noise
row — real and null will both look like noise. Document but do not count as FAIL.

Snapshot data:
${JSON.stringify(snap, null, 2)}

Report verdict, per-ensemble status, and any cases needing reshuffle with new seed.`,
    { label: 'cc-null-control', schema: SCHEMA_LENS }
  ),
])

phase('MultiL')

const multiL = await parallel([
  () => agent(
    `SU(2) M3 sigma volume-convergence diagnostic.

For each beta in {2.0, 2.3, 2.5, 2.7}, report sigma_creutz_22(L) for L in {6, 8, 12, 16}
and compute the max relative spread (max - min) / mean. Convergence to <5% spread is the
H5 "stable under refinement" signature.

Snapshot data:
${JSON.stringify(snap, null, 2)}

Report verdict, per-beta convergence numbers, and flag any beta where the spread is >5%.`,
    { label: 'multil-su2-sigma', schema: SCHEMA_LENS }
  ),

  () => agent(
    `SU(2) M1 effective mass volume dependence + convergence.

For each beta, compute m_g(L) for L in {6, 8, 12, 16}. Report:
  - % change from L=6 to L=16
  - whether the trend is monotonic
  - approximate asymptotic value (extrapolation, not a fit)

The cleanest case is expected to be beta=2.3 (sweet spot between strong/weak coupling).
For others, plateau fitting from short t-window introduces noise.

Snapshot data:
${JSON.stringify(snap, null, 2)}

Report verdict, per-beta numbers, and which betas are "volume saturated" vs "still drifting".`,
    { label: 'multil-su2-mg', schema: SCHEMA_LENS }
  ),
])

phase('CrossGroup')

const crossGroup = await agent(
  `Cross-group SU(2) vs SU(3) at matched L=8 — the headline novelty claim.

For L=8:
  - SU(2) M3: report sigma_creutz_22(beta) across the SU(2) sweep
  - SU(3) M3 from Dec 2025 Modal (trustworthy): sigma at beta=6.0
  - SU(3) M3 from fresh GPU: sigma at beta=5.7, 6.0, 6.2, 6.4

Compute the Davis-language cross-group invariant m_g / sqrt(sigma) per (group, beta) where
both M1 plateau and M3 sigma are available. The hope is that for matched physical scale
(set by sigma) both groups show m_g/sqrt(sigma) in the published 3.5-4.5 ballpark
(Teper 1998 SU(N) interpolation).

Also try the Casimir-rescaled view: multiply SU(3) sigma by 2/3 (the Casimir ratio
C_2_F[SU(2)]/C_2_F[SU(3)] = (3/4)/(4/3) = 9/16... wait, the relevant rescaling for
the fundamental string tension under N-changing is via the Casimir; document the actual
factor you use and don't fudge it).

Snapshot data:
${JSON.stringify(snap, null, 2)}

Report verdict, the per-group m_g/sqrt(sigma) numbers, and whether cross-group collapse
onto a single curve is even visible at this lattice size + thermalization quality.`,
  { label: 'cross-group', schema: SCHEMA_LENS, effort: 'high' }
)

phase('PaperReadiness')

const synthesis = await agent(
  `Synthesize the paper-readiness verdict for Bee's "Yang-Mills mass gap in gigi" project.

Six lens reports from the prior phases:

CROSS-CHANNEL (3 reports):
${JSON.stringify(crossChannel, null, 2)}

MULTI-L (2 reports):
${JSON.stringify(multiL, null, 2)}

CROSS-GROUP (1 report):
${JSON.stringify(crossGroup, null, 2)}

SNAPSHOT META:
n_ensembles=${snap?.n_ensembles}
notes=${snap?.notes}

Produce a paper-readiness verdict markdown document with these sections (use h2):
  ## TL;DR
  ## Channel-by-channel verdict
  ## Multi-L volume convergence
  ## Cross-group SU(2) vs SU(3)
  ## Honest scope gaps
  ## Concrete next 3 moves (ranked)
  ## What this changes vs the prior paper-readiness verdict

Length 1500-2000 words. Bee's voice — direct, pattern-matcher, scope-honest, no marketing
language. End-target framing: substrate-as-publication, multi-channel + multi-L + multi-group
mass-gap evidence in gigi, NOT a Clay-problem proof.

If any lens reported the fresh GPU SU(3) is under-thermalized, name it explicitly as a
"re-run with longer thermalization" next move.

Save the markdown to ${REPORT_PATH} via the Write tool.

Return ONE PARAGRAPH (under 200 words) summarizing what landed and the file path.`,
  { label: 'paper-readiness', effort: 'high', model: 'opus' }
)

return { snap, crossChannel, multiL, crossGroup, synthesis }
