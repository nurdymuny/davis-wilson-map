# Paper-Readiness Verdict: YM Mass Gap in gigi (post SU(3) sweep)

## TL;DR

The SU(2) side of the bundle is paper-ready as a substrate-as-publication artifact: cross-channel agreement is clean at beta=2.3 across L=6,8,12,16, the asymptotic-freedom signature (Creutz sigma_22 monotone in beta) is volume-stable to under 3% spread at every beta, and Falsification Criterion 7 (shuffled-time null) is a clean kill on all 22 ensembles. That is the multi-channel + multi-L mass-gap evidence we set out to embed as queryable observables in gigi, and it survives adversarial reading.

The SU(3) side is **not** ready. The fresh GPU sweep at L=8 for beta in {6.0, 6.2, 6.4} is under-thermalized — Creutz sigma_22 is 36%, 48%, 78% above Bali-Schilling 1992, the deviation grows with beta (the Cabibbo-Marinari autocorrelation-time signature), and the Dec 2025 Modal ensemble at the same lattice point reproduces published to 3%. The two SU(3) beta=6.0 L=8 ensembles disagree by 18-32% on sigma alone, which means the cross-group "single curve" headline cannot stand at L=8 with current data.

End-target framing stays exactly where it has been: multi-channel + multi-L + multi-group mass-gap evidence embedded in gigi as queryable observables, **not** a Clay-problem proof. The SU(2) substrate is the publishable piece today. The cross-group claim needs one re-run.

## Channel-by-channel verdict

**M1 (plaquette plateau effective mass).** Honest, beta-dependent. At beta=2.0 and 2.3 (strong/intermediate coupling) the M1 plateau ratio m_g/sqrt(sigma) sits in the 3.5-4.5 band that brackets Teper's continuum 0++ value ~3.7. At beta=2.5 and 2.7 the ratios drift up to 4.8-8.3 — the textbook excited-state contamination signature of an unsmeared plaquette correlator at fixed small t. This is not M1 being broken; it is M1 doing exactly what M1 does without smearing. Eleven of the 22 rows fit on a single time-slice (t_lo=t_hi=1); those inherit early-t bias and should be footnoted, not headlined.

**M3 (Creutz string-tension asymptotic freedom).** Strong PASS on SU(2). Both sigma_creutz_22 and sigma_creutz_32 are monotone decreasing in beta at all four lattice sizes, and the inter-volume spread is under 3% at every beta (max 2.97% at beta=2.5). This is the cleanest channel in the bundle and the right one to lead with.

**M2 (APE/HYP smearing + variational basis).** Absent. The whole beta>=2.5 contamination story is what M2 is designed to fix. The honest path is: caveat-flag beta>=2.5 rows in the receipt, lead with beta=2.3, and queue M2 as the next channel to add. Re-running the cross-channel ratio audit after M2 lands should collapse the beta>=2.5 ratios from 5-8 down toward 3.7.

**M7 (shuffled-time null).** Strong PASS on all 22 ensembles, single seed (20260628). Twenty are diagnostic PASSes; two are non-diagnostic for unrelated reasons (the failed real-side plateau at SU(2) L=6 beta=2.5, and the unthermalized Dec 2025 SU(3) L=16 hot-start). No reshuffle with a new seed is required to clear the criterion. For paper-quality defense it is still worth running 2 additional seeds on the four tightest-real-error rows (SU(2) L=16 beta=2.3, L=12 beta=2.3, L=12 beta=2.5, L=12 beta=2.7) to convert PASS into PASS-with-seed-robustness.

## Multi-L volume convergence

**Creutz sigma_22 (H5 stable-under-refinement).** Strong PASS at all four SU(2) beta values. Spread across L in {6,8,12,16}: 1.33% at beta=2.0, 0.97% at beta=2.3, 2.97% at beta=2.5, 2.49% at beta=2.7. No beta exceeds the 5% threshold. L=6 is already within 3% of L=16 everywhere. This is the cross-volume invariant that actually works.

**M1 effective mass.** Mixed. Only beta=2.3 shows clean volume convergence: m_g(L) = 2.240, 2.047, 1.981, 1.949 across L=6,8,12,16 — monotone decreasing with shrinking step sizes (-0.193, -0.066, -0.032), errors also shrinking (0.035 -> 0.015), tail extrapolation m_inf ~ 1.92. The other three betas are non-monotonic, and the non-monotonicity tracks the t-window flipping between t_hi=1 and t_hi=4 across volumes, not a real finite-V effect — Creutz sigma at those same betas is L-stable, so the underlying gauge dynamics ARE saturated. The L=16 jumps at beta=2.5 and 2.7 are single-time-slice fits and should not be reported at face value.

The honest reading: present Creutz sigma as the cross-volume invariant at every beta, and present M1 volume saturation only at beta=2.3.

## Cross-group SU(2) vs SU(3)

The cross-group bundle is the weakest piece of the deliverable today, and the reason is sampling, not group structure.

At L=8 the dimensionless ratio m_g/sqrt(sigma) does not collapse onto a single curve. Only two points land in the Teper 3.5-4.5 band (SU(2) beta=2.3 at 3.64; SU(3) beta=5.7 at 2.66 just below). SU(3) ratios rise monotonically with beta (5.24 -> 6.18 -> 6.24 -> 6.53 across beta=6.0,6.0,6.2,6.4), which is consistent with M1 plateau contamination plus under-thermalization compounding as a -> 0.

The headline blocker: two SU(3) beta=6.0 L=8 ensembles disagree at high significance. Dec 2025 Modal (n=50) gives sigma_22 = 0.1985 and m_g = 2.33; fresh GPU (n=80, n_therm=150) gives 0.2624 and 3.16. The Modal ensemble matches Bali-Schilling 1992 to 3%; the fresh GPU is 36% high. P_bar in the fresh GPU sweep is also low across the board (0.548, 0.595, 0.613, 0.631 at beta=5.7,6.0,6.2,6.4), where Modal at beta=6.0 hits 0.6417. **The fresh GPU SU(3) sweep at n_therm=150 is under-thermalized at weak coupling.** This is the single most actionable finding in the whole audit.

Casimir rescaling (multiply SU(3) sigma by C2_F(SU(2))/C2_F(SU(3)) = 9/16) makes the disagreement worse, which is theoretically correct — m_glueball/sqrt(sigma_F) is already the standard dimensionless ratio and should be N-stable without a Casimir fudge. We document the factor but do not present it as a fix.

The one suggestive comparison: at matched lattice spacing (SU(2) beta=2.7 sqrt(sigma)=0.408; SU(3) beta=6.4 sqrt(sigma)=0.462) the ratios agree to ~7% (6.10 vs 6.53), but both are ~50% above Teper continuum. Suggestive, not load-bearing.

## Honest scope gaps

- **No M2 channel.** Until APE/HYP smearing + a variational basis lands, the beta>=2.5 M1 ratios are excited-state-contaminated and cannot be cited as continuum-style evidence.
- **No SU(3) anchor at L>=12.** The only L=16 SU(3) row (Dec 2025 hot-start) is corrupt — P_bar = 4.5e-7, sigma sentinel -9999, plaquette never thermalized. No SU(3) volume scan exists.
- **Fresh GPU SU(3) at weak coupling is under-thermalized.** n_therm=150 is not enough for Cabibbo-Marinari at beta>=6.0. See next-moves.
- **Single shuffle seed for Criterion 7.** Passes are clean, but seed-robustness is not yet on file.
- **No continuum limit.** This is intentional — substrate-as-publication, not Clay-problem proof. Worth being explicit in the paper.
- **One failed plateau fit.** SU(2) L=6 beta=2.5 returns sentinel; row excluded from M1 aggregates, retained for Creutz channel.

## Concrete next 3 moves (ranked)

1. **Re-run SU(3) at beta >= 6.0 with longer thermalization.** This is the explicit ask flagged by the cross-channel lens. Target n_therm >= 1000 (preferably 2000) Cabibbo-Marinari sweeps, decorrelate by ~50 sweeps between measurements, report plaquette-vs-sweep thermalization curves, and cross-check by reproducing the Dec 2025 Modal beta=6.0 result with the GPU code. Without this, the cross-group bundle cannot stand. Highest leverage; nothing else moves until this lands.
2. **Implement M2 (APE smearing + 2x2 variational basis) and re-run the cross-channel ratio audit.** Expectation is the beta>=2.5 ratios collapse from 5-8 down toward 3.7. This converts the SU(2) bundle from "clean at beta=2.3, contaminated at weak coupling" to "clean across the full beta range" and unlocks honest continuum-style language.
3. **Add 2 additional shuffle seeds to Criterion 7 on the four tightest-real-error SU(2) rows** (L=16 beta=2.3, L=12 beta=2.3, L=12 beta=2.5, L=12 beta=2.7). Cheap, defensive, converts PASS into PASS-with-seed-robustness. Belt-and-suspenders, but the right thing to have on file before the paper goes out.

## What this changes vs the prior paper-readiness verdict

The prior verdict treated the fresh GPU SU(3) sweep as a candidate cross-group anchor. It is not. The under-thermalization signature is unambiguous (chi_22 over published by 36-78%, deviation growing with beta, P_bar systematically low, Modal reference at the same lattice point matching published to 3%). The cross-group "both groups collapse onto Teper" headline is off the table until move #1 above completes.

What is **stronger** than the prior verdict: the SU(2) multi-L bundle. The H5 Creutz-sigma volume-saturation check is a strong PASS at all four betas with sub-3% spread, and the beta=2.3 M1 column gives geometric convergence to m_inf ~ 1.92 across L=6,8,12,16. The shuffled-time null sentinel is a clean kill on every ensemble. These three pieces together make the SU(2) substrate-as-publication artifact defensible today, independently of the SU(3) work.

Net: scope tightens, claim shape sharpens. Lead with SU(2) multi-channel + multi-L. Footnote SU(3) as in-progress. Re-run SU(3) with proper thermalization, add M2, then re-open the cross-group claim. End-target — multi-channel + multi-L + multi-group mass-gap evidence in gigi as queryable observables — is two moves away, not zero, and the two moves are both standard lattice hygiene, not new physics.
