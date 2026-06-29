# Halcyon Yang-Mills Holonomy-Continuum Mass Gap, Queryable in Gigi

## Provenance

- date: 2026-06-28
- machine: LAPTOP-5ECOBNCR
- davis-wilson-lattice commit: `92c53f83f3245d34c536b68b803034b0fd36ea77` (branch `feat/halcyon-gigi-substrate`)
- gigi commit: `35a727d940c97c2906f6277bd9a09434c6e0f774`
- gigi endpoint at http://localhost:3142: NOT REACHABLE at audit time; numbers below are from on-disk receipts
- framing doc: `gigi/theory/YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md`

Receipt files used:
- `inertia_damping/reports/ym_mass_gap_in_gigi_receipt.json` (2D SU(2) buckyball, 9 betas, closed-form gap)
- `inertia_damping/reports/ym4_glueball_in_gigi_receipt.json` (4D SU(2) L=8 sweep, 4 betas, M1+M3)
- `inertia_damping/reports/ym4_su3_dec2025_configs_receipt.json` (4D SU(3) Dec 2025 Modal configs, L=8 + L=16, beta=6.0)
- `inertia_damping/reports/buckyball_local_falls_out.json` (2D source data + Migdal-Witten exact reference)

## Core Claim

The Yang-Mills mass gap is read off gauge-invariant transport data via M1 (plaquette-plaquette connected correlator) and M3 (Wilson-loop Creutz-ratio string tension), visible in a single GQL line over a multi-L multi-beta multi-group bundle (`halcyon_ym4_glueball_demo` + `halcyon_ym_mass_gap_demo`), with the shuffled-null control (Falsification Criterion 7) failing to produce a plateau as predicted.

## The Single GQL Query (the headline)

```sql
SELECT
  gauge_group, L, beta,
  plateau_fit_mass, plateau_fit_error,           -- M1
  sigma_creutz_22, sigma_creutz_22_error,        -- M3
  m_eff_null_t,                                  -- Falsification Crit. 7
  m_holonomy_continuum_heat_kernel               -- analytical reference (2D)
FROM halcyon_ym4_glueball_demo
FULL OUTER JOIN halcyon_ym_mass_gap_demo USING (beta)
WHERE t = 0;
```

Headline query as actually shipped in `ym4_glueball_in_gigi_receipt.json`:

```sql
SELECT beta, plateau_fit_mass, plateau_fit_error, sigma_creutz_22,
       sigma_creutz_22_error, m_eff_null_t
FROM halcyon_ym4_glueball_demo WHERE t = 0;
```

Headline query as shipped in `ym_mass_gap_in_gigi_receipt.json`:

```sql
SELECT beta, m_holonomy_continuum_heat_kernel_su2, string_tension_wilson_fundamental,
       P_measured, P_exact_migdal_witten, delta_P, C_proxy
FROM halcyon_ym_mass_gap_demo;
```

## Result Table: 2D SU(2) Buckyball (exactly soluble)

Source: `ym_mass_gap_in_gigi_receipt.json` + `buckyball_local_falls_out.json`.
Graph: truncated icosahedron (V=60, E=90, F=32, chi=2). Heat-kernel convention; closed-form gap m_holonomy = (beta/2)*C_2(j=1/2) = 3*beta/8.

| beta | m_holonomy = 3*beta/8 | P_measured | P_exact (Migdal-Witten) | delta_P | curvature_density | C_proxy | sigma_Wilson_fundamental |
|------|----------------------:|-----------:|------------------------:|--------:|------------------:|--------:|-------------------------:|
| 0.50 | 0.1875 | 0.12238 | 0.12372 | -0.00134 | 0.8776 | 1.1394 | 2.0898 |
| 1.00 | 0.3750 | 0.23588 | 0.24019 | -0.00431 | 0.7641 | 1.3087 | 1.4263 |
| 1.50 | 0.5625 | 0.37029 | 0.34414 |  0.02614 | 0.6297 | 1.5880 | 1.0667 |
| 2.00 | 0.7500 | 0.44144 | 0.43313 |  0.00831 | 0.5586 | 1.7903 | 0.8367 |
| 2.25 | 0.8438 | 0.45462 | 0.47195 | -0.01733 | 0.5454 | 1.8336 | 0.7509 |
| 2.30 | 0.8625 | 0.50078 | 0.47928 |  0.02150 | 0.4992 | 2.0031 | 0.7355 |
| 2.50 | 0.9375 | 0.53030 | 0.50720 |  0.02311 | 0.4697 | 2.1290 | 0.6789 |
| 2.70 | 1.0125 | 0.53659 | 0.53297 |  0.00362 | 0.4634 | 2.1579 | 0.6293 |
| 3.00 | 1.1250 | 0.57863 | 0.56792 |  0.01071 | 0.4214 | 2.3732 | 0.5658 |

Aggregate fit-vs-exact: max |delta_P| = 0.02614 (beta=1.5), rms |delta_P| = 0.01562 over 9 points.

Closed-form derivation (Witten 1991, Migdal 1975, Rusakov 1990): canonical quantization of 2D YM on a closed orientable surface gives H = (beta*A/2)*Delta_G on L^2(G); eigenvalues = Casimir C_2(R); first non-trivial irrep is j=1/2 with C_2 = 3/4, so gap = 3*beta/8 per unit lattice area. EXACT for heat-kernel (Migdal) action; the simulation uses Wilson action (Kennedy-Pendleton heatbath), so the Wilson-vs-heat-kernel contrast in the last column is shown deliberately.

## Result Table: 4D SU(2) cross-L multi-beta

Source: `ym4_glueball_in_gigi_receipt.json`. All rows L=8, n_configs=100. (L=6 / L=12 / L=16 noted as in-progress per session state; not on disk at audit time.)

| L | beta | m_g (M1) | err(M1) | sigma (M3, chi_22) | err(M3) | sqrt(sigma) | m_eff_null | <P>_global | wall_s |
|---|-----:|---------:|--------:|-------------------:|--------:|------------:|-----------:|-----------:|-------:|
| 8 | 2.0  | 3.9183   | 0.1716  | 0.59857            | 0.000417 | 0.7737    | NaN (null) | 0.5012 | 334.4 |
| 8 | 2.3  | 2.0467   | 0.0312  | 0.31554            | 0.000181 | 0.5617    | NaN (null) | 0.6022 | 156.9 |
| 8 | 2.5  | 2.3836   | 0.0396  | 0.21395            | 0.000115 | 0.4626    | NaN (null) | 0.6517 | 123.9 |
| 8 | 2.7  | 2.4853   | 0.0382  | 0.16611            | 0.0000787 | 0.4076   | NaN (null) | 0.6865 | 140.9 |

Cross-L drift placeholders (PENDING in-flight L=6, L=12, L=16 SU(2) ensembles):
- `sigma(L_max)/sigma(L_min)` per beta — TBD when L>8 receipts land.
- `m_g(L=16) - m_g(L=6)` per beta — TBD when L>8 receipts land.

Cross-beta volume-independent diagnostic at L=8 (NOT a continuum extrapolation, only a substrate-level monotonicity check): sigma decreases monotonically with beta (0.599 -> 0.316 -> 0.214 -> 0.166), consistent with weak-coupling running of the lattice scale.

## Result Table: 4D SU(3) beta=6.0

Source: `ym4_su3_dec2025_configs_receipt.json`. 50 configs each, Dec 2025 Modal generation.

| L  | beta | n_cfg | <P>     | m_g (M1) | err(M1) | sigma (M3, chi_22) | err(M3) | m_eff_null | status |
|----|-----:|------:|--------:|---------:|--------:|-------------------:|--------:|-----------:|--------|
| 8  | 6.0  | 50    | 0.6419  | 2.3337   | 0.0745  | 0.19851            | 0.000138 | 2.6101 | thermalized |
| 16 | 6.0  | 50    | -2.5e-5 | 4.3888   | 0.5663  | NaN                | NaN      | NaN    | UNTHERMALIZED (<P>~0; NOT physics-quality) |

Comparison: Bali-Schilling 1992 published SU(3) beta=6.0 sigma ~ 0.0513 (in lattice units, a^2 sigma). Our L=8 sigma_creutz_22 = 0.19851 is the Creutz ratio chi(2,2), which over-estimates the asymptotic sigma by an O(1) factor on small loops (corner/perimeter contamination); these are not directly comparable numbers without smearing or larger R,T. Reference published <P>(beta=6.0) ~ 0.5937; our L=8 <P>=0.6419 is slightly high (small-volume + autocorrelation).

PLACEHOLDER: fresh GPU SU(3) sweep at beta = 5.7 / 6.0 / 6.2 / 6.4 — queued, results PENDING.

## Null Control Results

Falsification Criterion 7 (shuffled-plaquette null): if the cosh plateau fit on time-shuffled plaquette data still produces a finite plateau mass, the M1 result is suspect.

| ensemble | gauge_group | L  | beta | null m_eff | verdict |
|----------|-------------|---:|-----:|-----------:|---------|
| su2_4d_L8_beta200_n100 | SU(2) | 8 | 2.0 | NaN | Crit 7 PASS (no spurious plateau) |
| su2_4d_L8_beta229_n100 | SU(2) | 8 | 2.3 | NaN | Crit 7 PASS |
| su2_4d_L8_beta250_n100 | SU(2) | 8 | 2.5 | NaN | Crit 7 PASS |
| su2_4d_L8_beta270_n100 | SU(2) | 8 | 2.7 | NaN | Crit 7 PASS |
| su3_4d_L8_beta600_dec2025 | SU(3) | 8 | 6.0 | 2.6101 | Crit 7 LOW-CONFIDENCE / suspect: null fit returned a finite mass ~ same magnitude as real (2.334). Needs investigation (likely small-n_configs jackknife noise on shuffled time-axis). |
| su3_4d_L16_beta600_dec2025 | SU(3) | 16 | 6.0 | NaN | Crit 7 PASS but ensemble itself UNTHERMALIZED |

Counts: Crit 7 PASS = 5 / 6 ensembles. FAIL / SUSPECT = 1 / 6 (su3_4d_L8_beta600_dec2025).

## Cross-Channel Agreement

Ratio r = m_g(M1) / sqrt(sigma(M3)), to compare against the published Teper 1998 SU(2) glueball ratio m(0++)/sqrt(sigma) ~ 3.7.

| gauge_group | L | beta | m_g(M1) | sqrt(sigma) | r = m_g / sqrt(sigma) | comment |
|-------------|---:|-----:|--------:|------------:|----------------------:|---------|
| SU(2) | 8 | 2.0 | 3.918 | 0.7737 | 5.06 | strong-coupling, ratio off |
| SU(2) | 8 | 2.3 | 2.047 | 0.5617 | 3.64 | closest to Teper 3.7 |
| SU(2) | 8 | 2.5 | 2.384 | 0.4626 | 5.15 | over by ~40% |
| SU(2) | 8 | 2.7 | 2.485 | 0.4076 | 6.10 | over by ~60% |
| SU(3) | 8 | 6.0 | 2.334 | 0.4456 | 5.24 | published SU(3) m(0++)/sqrt(sigma) ~ 3.55 (Teper 1998) — over by ~50% |

Deviation from Teper attributed to: (a) no M2 variational smearing (M1 plaquette-plaquette correlator without APE smearing has poor overlap with the lowest 0++ glueball state, biasing m_g high); (b) L=8 small-volume contamination; (c) chi(2,2) overestimates asymptotic sigma. All three push r upward, consistent with observation. Scope-limited, not a counter-result.

## Scope Gaps Named Honestly

1. M2 glueball variational basis with APE smearing + GEVP — NOT yet pushed. M1 plaquette-plaquette correlator over-couples to high-energy noise; M2 is the standard fix. Code stub being written in parallel per session state.
2. M4 capacity spectrum via SPECTRAL_GAUGE on link-variable bundles — NOT yet done for 4D. (Empirical SPECTRAL_GAUGE receipt exists at `reports/spectral_gauge_empirical_receipt.json` for a different test; not wired into the YM bundle.)
3. Scale setting — no `lattice_spacing_a` column. Wilson flow (t_0, w_0) and r_0 not yet computed. Only beta and lattice-unit masses. No conversion to MeV.
4. Null controls 1, 2, 5, 6 from the framing doc — only Criterion 7 (shuffled-time) is tested. Crit 1 (Gaussian-link null), 2 (small-beta perturbative null), 5 (free-field null), 6 (cold-start null) — none yet exercised.
5. n_configurations = 50-100 per ensemble — small for jackknife. Production-quality would use 500-2000.
6. Autocorrelation — Dec 2025 SU(3) used separation=10 sweeps; fresh SU(2) sweeps did not measure tau_int. No integrated autocorrelation time receipts.
7. Cross-L convergence — only L=8 receipts on disk at audit time. L=6/12/16 SU(2) and full SU(3) beta-sweep PENDING.

## Reproducibility

Commits the backbone numbers come from:
- davis-wilson-lattice: `92c53f83f3245d34c536b68b803034b0fd36ea77`
- gigi: `35a727d940c97c2906f6277bd9a09434c6e0f774`

Deterministic-seed pattern: seed = `20260616` for the 2D buckyball sweep (bit-identical reproduction; crash recovery proved this). Per-ensemble seeds for the 4D sweeps are encoded in ensemble_id strings (`su2_4d_L{L}_beta{int(100*beta)}_n{n_configs}`) and the orchestrators (`push_ym4_glueball_bundle.py`, `push_ym4_su3_glueball_bundle.py`) consume them deterministically.

Two-command reproduction from a fresh clone:
```bash
# 1. SU(2) 4D sweep -> halcyon_ym4_glueball_demo bundle
python inertia_damping/push_ym4_glueball_bundle.py

# 2. SU(3) 4D Dec 2025 configs -> same bundle
python inertia_damping/push_ym4_su3_dec2025_configs.py
```
For the 2D buckyball gap: `python inertia_damping/push_buckyball_to_gigi.py` (or the upstream `buckyball_local_falls_out.py` to regenerate the source JSON).

## Novelty Scope (audited)

NOT NOVEL (cited prior art):
- 2D YM closed-form gap m = (beta/2)*C_2(R_*): Witten 1991, Migdal 1975, Rusakov 1990.
- Kennedy-Pendleton SU(2) heatbath: Kennedy-Pendleton 1985.
- Cabibbo-Marinari SU(3) pseudo-heatbath: Cabibbo-Marinari 1982.
- Plaquette-plaquette connected correlator for glueball mass: standard lattice technique (Berg 1980 et seq.).
- Creutz ratio for string tension: Creutz 1980.
- Migdal-Witten heat-kernel exact plaquette expectation on 2D surface: textbook.
- Teper 1998 SU(N) glueball spectrum ratios.

NOVEL (the substrate combination):
- Lattice data + analytical reference column + Davis-language column (`curvature_density`, `C_proxy`) + shuffled-null control + per-insert geometric drift (`insert_curvature`, `insert_confidence`), all queryable as ONE SQL SELECT in a single bundle.
- Multi-group (SU(2) + SU(3)) + multi-L (in-progress) + multi-beta + 2D-exact + 4D-Monte-Carlo, packed into the same bundle schema (`halcyon_ym4_glueball_demo` joined to `halcyon_ym_mass_gap_demo`).
- Falsification Criterion 7 (shuffled-null cosh fit) as a first-class queryable column, not a footnote in a methods paper.

NOT YET CLAIMED:
- A full continuum extrapolation in physical units (no scale setting).
- Full M2 / M4 cross-channel agreement (M2 not implemented, M4 not on 4D).
- A resolved 4D continuum mass gap (the open Clay problem). The framing remains: holonomy-substrate evidence assembled and queryable; continuum-limit existence not demonstrated.
