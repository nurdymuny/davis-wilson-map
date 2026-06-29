# Yang-Mills mass gap embedded in a geometric-query substrate (work in progress)

## Who I am and what I'm working on

I'm Bee Rosa Davis (Gigi). Independent mathematician, learning physics as I go. Two repos matter here:

- `davis-wilson-lattice`, Monte Carlo lattice gauge code (SU(2) and SU(3), 2D and 4D) plus the orchestrators that push results into a substrate.
- `gigi`, a geometric-query layer written in Rust (~25 modules, native HOLONOMY / TRANSPORT / SPECTRAL / BETTI verbs). Runs locally on `http://localhost:3142`.

The end-target I've named for this thread of work: **maximum solid evidence for the Yang-Mills mass gap, embedded as queryable observables in `gigi`.** Multi-channel times multi-L times multi-group bundle is the deliverable. Substrate-as-publication: the paper claims should be re-runnable as `SELECT` queries.

I am **not** claiming to have solved the Clay problem. I'm claiming I can produce a clean cross-L, cross-group, cross-β data product where the gap is visible, the null control kills it, and the analytic 2D reference sits in the same table for honest contrast. The novelty is the substrate combination, not the physics.

## The framing doc (holonomy-continuum hypothesis)

Lives at `gigi/theory/YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md`. Short version of the operative hypotheses and channels:

- **H1**. Continuum YM is the limit of holonomy/transport data, not of pointwise field values. The substrate-natural object is the holonomy bundle, not the connection.
- **H4**. The mass gap is the finite-capacity propagation scale of that holonomy substrate.
- **H5**. `gigi` should detect the gap via effective masses on gauge-invariant correlators of holonomy observables.

Measurement channels:

| ID | Observable | Status |
|----|-----------|--------|
| M1 | Connected plaquette correlator, cosh effective mass plateau | **Wired in**, every ensemble |
| M2 | 0⁺⁺ glueball via APE smearing + variational basis + GEVP | Code stub exists (`su2_4d_glueball_M2.py`, `su3_4d_glueball_M2.py`), 5/5 tests pass, not yet pushed into the bundle |
| M3 | Wilson loop area-law via Creutz ratio χ(R,T), σ | **Wired in**, every 4D ensemble |
| M4 | `SPECTRAL_GAUGE` capacity spectrum on link bundles | Verb shipped; 4D link bundles not yet pushed |

Falsification criteria: only **Crit 7** is currently tested. Shuffle plaquette values across config-axis and t-axis, run the same pipeline, the gap should die. Crits 1, 2, 5, 6 (free Gaussian baseline, abelian U(1), gauge-transformed copies, synthetic injected modes) are not done.

## What lives in gigi right now

Two bundles, both queryable on the local instance.

**`halcyon_ym_mass_gap_demo`**. 2D SU(2) buckyball, 9 β-points from β=0.5 to β=3.0. Closed-form holonomy gap sits in the table alongside the Migdal–Witten exact ⟨P⟩ and the Davis capacity proxy (a definition, not derived from a physics principle; see the 2D section).

**`halcyon_ym4_glueball_demo`**. 4D, the working bundle. 18 ensembles as of last check; the SU(3) sweep just fired and will bring the count up. Schema columns:

```
ensemble_id, gauge_group, dimension, L, beta, n_configurations,
t, P_bar_t, C_PP_t, C_PP_error_t, m_eff_t, m_eff_error_t, m_eff_null_t,
plateau_fit_mass, plateau_fit_error, plateau_fit_t_lo, plateau_fit_t_hi,
sigma_creutz_22, sigma_creutz_22_error, sigma_creutz_32, sigma_creutz_32_error,
W_11, W_12, W_22, W_23, W_33,
measurement_channel, framing_doc_version
```

`SPECTRAL_GAUGE` Phase 1 verb is live (dense fiber-weighted Laplacian, returns gap + `n_records_used` + `group_used`). Not yet hooked to 4D link bundles, that's the M4 gap.

## The main GQL query

The whole point of substrate-as-publication is that a reviewer can ask one question and get the cross-channel picture for an ensemble:

```sql
SELECT ensemble_id, gauge_group, L, beta, n_configurations,
       plateau_fit_mass, plateau_fit_error, plateau_fit_t_lo, plateau_fit_t_hi,
       sigma_creutz_22, sigma_creutz_22_error,
       sigma_creutz_32, sigma_creutz_32_error,
       measurement_channel
FROM halcyon_ym4_glueball_demo
WHERE gauge_group = 'SU(2)' AND beta = 2.3
ORDER BY L;
```

That returns the M1 effective mass and the M3 Creutz σ on the same row for L = 6, 8, 12, 16. Convergence in L is then a `GROUP BY` away.

## The 2D SU(2) buckyball case (exactly soluble)

Truncated icosahedron substrate: V=60, E=90, F=32, χ=2. Why this is the calibration case: 2D Yang-Mills is exactly soluble (Migdal 1975, Witten 1991, 35-year-old result, I am not claiming this part). It gives me an analytic reference column the lattice MC has to hit.

- Heat-kernel holonomy gap, per unit lattice area: **m = 3β/8**, exact. This is the heat-kernel-convention string tension in the fundamental irrep (σ_R = (β/2)·C_2(R) with C_2(j=1/2)=3/4), equivalent to the canonical-quantization spatial-circle energy gap with H = (β·L/2)·Δ_G. It is not a dimensionless Hamiltonian eigenvalue in lattice units.
- Wilson-action plaquette: **exact ⟨P⟩ from the Migdal–Witten character-expansion sum on a closed orientable surface** (χ=2, P=32), computed in `inertia_damping/buckyball_yangmills_exact.py:exact_mean_plaquette_su2_2d` as ⟨c_j′(β)/c_j(β)⟩ over p_j ∝ (2j+1)^χ · c_j(β)^P with c_j(β) = (2/β)·I_{2j+1}(β). The j=1/2 leading-order ratio is 1/β + I_3(β)/I_2(β); the full sum is what lands in the bundle's `<P>_exact` column. I_2(β)/I_1(β) is the *infinite-plane* Wilson-loop expectation per unit area, not the closed-surface ⟨P⟩.
- Wilson fundamental string tension (different convention, same row for contrast): **σ_W = −log(I_2(β) / I_1(β))**.

Worked Wilson-vs-heat-kernel numerical contrast (from the docstring in `push_ym_mass_gap_bundle.py`, surfaced here so a downstream reader can't accidentally cross-apply): at β=1, heat-kernel m=0.375 vs Wilson σ ≈ 0.69; at β=2, heat-kernel m=0.75 vs Wilson σ ≈ 0.51. They are not the same function of β.

Lattice MC measurement (SU(2) Kennedy–Pendleton heatbath on the buckyball) vs analytic ⟨P⟩ across the 9 β-points: **RMS deviation 0.0156 (~3%)**. Bundle reproduces bit-identically from the deterministic seed.

The 2D bundle row carries: closed-form `m_holonomy_continuum_heat_kernel_su2`, measured `<P>`, analytic `<P>_exact` (Migdal–Witten character sum, not I_2/I_1), `sigma_Wilson_fundamental` (Wilson convention, in the same row for honest contrast), and the **Davis capacity proxy C = 1/(1 − ⟨P⟩)**. The capacity proxy is a definition I am using, not a derived physics quantity; flagging it explicitly so it isn't taken as published.

## The 4D SU(2) sweep (cross-L volume convergence)

SU(2) Wilson action, quaternion representation, Kennedy–Pendleton x₀ sampler, red-black checkerboard parallelism. β ∈ {2.0, 2.3, 2.5, 2.7} at four lattice extents. L=6 on CPU; L=8, 12, 16 on local RTX 5070 (8.5 GB VRAM, CUDA 12.0). Thermalization 200 sweeps, measurement 80–100 sweeps, `measure_every=2`.

**M3, Creutz χ(2,2) cross-L** (the strongest cross-L signal):

| β | L=6 | L=8 | L=12 | L=16 | max spread |
|---|-----|-----|------|------|------------|
| 2.0 | 0.5932 | 0.5986 | 0.5957 | 0.6011 | ~1% |
| 2.3 | 0.3186 | 0.3155 | 0.3156 | 0.3177 | ~1% |
| 2.5 | 0.2081 | 0.2140 | 0.2124 | 0.2144 | ~3% |
| 2.7 | 0.1650 | 0.1661 | 0.1692 | 0.1687 | ~3% |

**M1, effective mass at L=12** (the most stable extent in our scan):

| β | m_g (lattice units) |
|---|---------------------|
| 2.0 | 3.48 ± 0.09 |
| 2.3 | 1.98 ± 0.02 |
| 2.5 | 2.21 ± 0.03 |
| 2.7 | 2.19 ± 0.03 |

**β=2.3 M1 cross-L convergence** (the cleanest case):

| L | m_g | Δ vs L=16 |
|---|-----|-----------|
| 6 | 2.24 | +0.29 |
| 8 | 2.05 | +0.10 |
| 12 | 1.98 | +0.03 |
| 16 | 1.95 | — |

Monotonic, converging to ~1.9. Published SU(2) 0⁺⁺ glueball mass in this β range (Teper 1998, roughly β=2.3–2.5 for SU(2) Wilson) is 1.4–1.8 in lattice units. Mine sits high by ~10–30%, which is the expected signature of no smearing: M1 is the unsmeared plaquette correlator, so it has heavy excited-state contamination at small t. M2 (smearing + GEVP) is the right tool to drive that down. Code stub is in, push not done.

## The 4D SU(3) (Dec 2025 Modal + fresh GPU)

SU(3) is Cabibbo–Marinari: 3 SU(2) subgroup steps per link. Two sources of configs.

**Dec 2025 Modal A100 configs** (pulled back to `_modal_pull/davis-wilson-data/configs/`, 1.9 GB volume, authed as `bee-davis` profile):

- L=8 thermalized (sep=10, therm=100), n_configs=50: physics-quality.
- L=16 unthermalized hot-start, ⟨P⟩ ~ 0: garbage. Pushed with caveat. Useful only as a check that the null control actually fails to discriminate when both real and shuffled streams are noise, which it does, by design.

L=8 β=6.0 numbers:

| observable | value | published reference |
|------------|-------|---------------------|
| χ(2,2) (Creutz) | 0.1985 ± 0.0001 (n=50) | Bali–Schilling 1992 χ(2,2) at β=6.0 ≈ 0.193 (~3% match) |
| m_g (M1) | 2.33 ± 0.07 | Teper 1998 0⁺⁺ ≈ 1.7, high, no smearing |

Note on the χ(2,2) reference number: 0.193 is the published *Creutz ratio* at β=6.0, not the asymptotic string tension. The asymptotic σa² at β=6.0 from Bali–Schilling is ≈ 0.05 (√σ a ≈ 0.22). The like-for-like comparison is χ(2,2) vs χ(2,2); applying 0.193 as the asymptotic tension in scale-setting would give the wrong lattice spacing.

**Fresh local GPU SU(3) sweep** (running now): β ∈ {5.7, 6.0, 6.2, 6.4} at L=8, validated against published ⟨P⟩ at β=5.7/6.0/6.2 to <1% in standalone tests.

## Null control results

This is Falsification Crit 7: per-config plaquette values shuffled across the config-axis and the t-axis, destroying time-correlation. Same pipeline (cosh effective mass, plateau fit, jackknife) on the shuffled stream.

| Ensemble subset | Null result |
|----------------|-------------|
| 16 / 18 SU(2) ensembles | `m_eff_null = NaN` at every t, clean pass |
| SU(3) L=8 Dec 2025 β=6.0 | null `plateau_fit_mass = 2.61` vs real 2.33, borderline; caught one noisy point in the same range. Worth reshuffling with a new seed. |
| SU(3) L=16 unthermalized | null ≈ real, because both streams are essentially random. Meaningless by design; this row is in the bundle to demonstrate the null is actually working as a discriminator and not silently passing. |

Only Crit 7 is wired in. The other five aren't done.

## What's measured per channel

For each 4D ensemble:

- **Plaquette correlator**. Spatial-volume-averaged plaquette density per t-slice. Connected correlator `C_PP(t) = ⟨P̄(0)·P̄(t)⟩ − ⟨P̄⟩²` with origin-translation averaging, jackknife errors.
- **Effective mass**. Cosh form: `m_eff(t) = arccosh((C(t−1) + C(t+1)) / (2·C(t)))`.
- **Plateau fit**. Weighted constant fit in finite-mass window, auto-selected from the first finite t. `plateau_fit_t_lo` and `plateau_fit_t_hi` are persisted so anyone can re-fit on a different window.
- **Wilson loops**. Link-product chains, all 12 ordered (μ_s, μ_t) plane pairs averaged, normalization (1/N) Re Tr. Sizes W_11 through W_33 stored.
- **Creutz ratio**. `χ(R,T) = −log(W(R,T)·W(R−1,T−1) / (W(R−1,T)·W(R,T−1)))` with jackknife. Stored at (2,2) and (3,2).
- **Null**. Shuffled-stream effective mass at every t. If the shuffled pipeline produces a finite mass plateau in the same range as real, the row is flagged borderline.

For each 2D ensemble: lattice ⟨P⟩, Migdal–Witten ⟨P⟩_exact (character sum), σ_Wilson_fundamental, m_holonomy heat-kernel closed form (per unit lattice area), Davis capacity proxy (definition).

## What's missing (honest scope gaps)

I want these named explicitly, not buried.

1. **M2 not in the bundle.** Glueball variational basis + APE smearing + GEVP. Code stubs exist (`inertia_damping/su2_4d_glueball_M2.py`, `su3_4d_glueball_M2.py`, tests 5/5). Not yet wired into the orchestrators. This is the cleanest path to bringing M1 down toward published Teper numbers.
2. **M4 not pushed.** `SPECTRAL_GAUGE` Phase 1 verb exists and is live, but 4D link-variable bundles aren't in `gigi` yet. The verb has nothing to chew on.
3. **No scale setting.** No Wilson flow, no r_0, no Sommer scale. There is no `lattice_spacing_a` column. All masses are in lattice units only. I cannot quote anything in MeV.
4. **Only Crit 7 null control.** Crits 1 (free Gaussian baseline), 2 (abelian U(1), should not gap in 4D), 5 (gauge-transformed copies, should give identical), 6 (synthetic injected modes, should be recovered) are not done.
5. **n_configurations small.** 80–100 per ensemble for the 4D SU(2) sweep, 50 for the SU(3) L=8 Dec 2025 set. Production-grade lattice work would use 500–2000. Statistical errors reflect this.
6. **Continuum extrapolation not attempted.** No a → 0 limit. The cross-L convergence I show is volume convergence at fixed bare coupling, not continuum convergence.
7. **2D mass gap formula is 35 years old.** Witten 1991. I am not claiming the physics. I'm claiming the substrate placement.

## What's running right now

Task `bzh0d8sij`: 4D SU(3) Wilson heatbath GPU sweep at L=8, β ∈ {5.7, 6.0, 6.2, 6.4}, ETA ~25 min from the last check. On finish, the orchestrator (`push_ym4_su3_glueball_bundle.py`) pushes M1 + M3 + Crit 7 null rows to `halcyon_ym4_glueball_demo`. That brings the SU(3) cross-β picture into the same bundle as the SU(2) cross-L picture.

## Reproducibility

Everything reproduces from deterministic seed. Two proof points already on disk:

- 2D bundle: bit-identical regeneration from the seed in `push_ym_mass_gap_bundle.py`.
- 4D SU(2) crash recovery: my computer hard-rebooted mid-work; every pre-crash number reproduced bit-identically by re-firing the heatbath.

Key files, all absolute paths under `C:\Users\nurdm\OneDrive\Documents\davis-wilson-lattice\`:

| Path | Purpose |
|------|---------|
| `inertia_damping/buckyball_heatbath.py` | 2D SU(2) Kennedy–Pendleton on buckyball |
| `inertia_damping/buckyball_yangmills_exact.py` | Migdal–Witten character-expansion exact ⟨P⟩ |
| `inertia_damping/su2_4d_heatbath.py` | CPU 4D SU(2) Wilson heatbath, quaternion repr |
| `inertia_damping/su2_4d_heatbath_gpu.py` | GPU (torch/CUDA) 4D SU(2) |
| `inertia_damping/su3_4d_heatbath_gpu.py` | GPU SU(3) wrapper, Cabibbo–Marinari |
| `inertia_damping/su2_4d_glueball.py` | Correlator extraction, jackknife, Creutz ratio |
| `inertia_damping/su2_4d_glueball_M2.py` | M2 stub (APE + GEVP), tests pass |
| `inertia_damping/su3_4d_glueball_M2.py` | M2 stub SU(3), tests pass |
| `inertia_damping/push_ym_mass_gap_bundle.py` | 2D SU(2) bundle push, closed-form gap |
| `inertia_damping/push_ym4_glueball_bundle.py` | 4D SU(2) M1 + M3 + null orchestrator |
| `inertia_damping/push_ym4_su3_glueball_bundle.py` | 4D SU(3) orchestrator |
| `inertia_damping/push_ym4_su3_dec2025_configs.py` | Reprocess Dec 2025 Modal configs |
| `inertia_damping/REPRODUCTION.md` | Third-party reproduction guide |
| `inertia_damping/PAPER_DRAFT_BACKBONE.md` | Backbone tables (1736 words) from receipts |

The framing doc is at `gigi/theory/YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md` and the bundle name lives in the `framing_doc_version` column on every row, so anyone can trace a number back to the hypothesis it was supposed to test.

Hardware: local NVIDIA RTX 5070 Laptop GPU, 8.5 GB VRAM, CUDA 12.0. Dec 2025 SU(3) configs originally generated on Modal A100, pulled back to `_modal_pull/davis-wilson-data/configs/`.

`gigi` runs locally as a Rust binary on `http://localhost:3142`. The schema is fixed; the SPECTRAL_GAUGE Phase 1 verb is shipped on prod and local.

## What I'd want help with

Ranked by what would most strengthen the bundle:

1. **Wire M2 in.** The code stubs and tests are there. Push APE-smeared 0⁺⁺ variational-basis GEVP masses into `halcyon_ym4_glueball_demo` alongside M1. Expectation: SU(2) β=2.3 m_g drops from 1.98 toward the published 1.4–1.8 band.
2. **More null controls.** At minimum Crit 5 (gauge-transformed copies must give bit-identical observables, sanity check the pipeline) and Crit 2 (abelian U(1) Wilson action in 4D, should *not* show a mass gap, so the pipeline reports a non-gap).
3. **Push the 4D link bundles** so `SPECTRAL_GAUGE` (M4) has something to act on. Even Phase 1 dense Laplacian on one ensemble would close the M4 loop.
4. **Reshuffle the SU(3) L=8 Dec 2025 null** with a new seed. The 2.61-vs-2.33 borderline is probably a single noisy shuffle realization, not a real failure; want to confirm by averaging over shuffle seeds.
5. **Increase n_configurations** on the cleanest SU(2) ensemble (β=2.3, L=12) toward 500. The error bar is already 0.02 in lattice units; tightening it would let me see if the residual gap to published Teper numbers is statistical or systematic-from-no-smearing.
6. **Decide on a scale-setting plan.** Wilson flow on the existing configs is the cheap path. Adds an `a` column, lets me quote things in MeV and quote a continuum-extrapolation roadmap honestly even if I haven't run it.

If you want to poke at it: point `gigi` at `http://localhost:3142` (or stand up a local copy from the `gigi` repo), run the query above, look at the rows. Everything else is the framing doc and the orchestrators.
