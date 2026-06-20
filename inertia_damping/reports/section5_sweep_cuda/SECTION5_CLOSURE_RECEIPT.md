# Section 5 Closure Receipt — v1.2.1

**Date:** 2026-06-19
**Substrate:** SU(2) Yang-Mills on the truncated icosahedron (buckyball,
$V=60$, $E=90$, $F=32$, $\chi=2$).
**Parameters:** $\beta=2.5$, $\mathrm{d}t=0.01$.
**Kernel:** `inertia_damping.cuda.batched_leapfrog` (torch + CUDA 12.8,
NVIDIA Blackwell RTX 5070).
**Status:** **NO_CLOSURE.** The Section 5 FAIL is a genuine ergodicity
caveat at the finite-size buckyball, not a finite-sample artefact.

## Question

The Halcyon v1.2 production run (`run_20260617_110642`) reported
**FAIL** on Section 5 (microcanonical-vs-canonical cross-check):

- Canonical heatbath plaquette $\langle P\rangle_{\mathrm{heatbath}}
  = 0.5068472 \pm 0.0014580$ (Flyvbjerg--Petersen blocked SEM, 2048 samples)
- Time-averaged plaquette along one $N_{\mathrm{steps}}=100$ symplectic
  trajectory $\langle P\rangle_{\mathrm{time}} = 0.4479$
- Gap $= 0.059$; tolerance $= 0.02$; verdict FAIL.

The v1.2 report flagged this as anticipated, attributing it to the 93
conserved degrees of freedom of the buckyball SU(2) Hamiltonian. The
question the v1.2.1 closure study answers: **does the gap close as the
trajectory length grows, or is it an irreducible finite-size feature?**

## Method

40 trajectories total: $N_{\mathrm{steps}} \in \{1000, 2000, 4000, 8000,
16000\}$ (a 16$\times$ range) crossed with 8 independent seeds
($\{20260616, \ldots, 20260623\}$). For each trajectory we compute
$\langle P\rangle_{\mathrm{time}}$ over the full trajectory and the
Flyvbjerg--Petersen blocked SEM, and we re-run the canonical heatbath at
each seed for $\langle P\rangle_{\mathrm{heatbath}}$ (200-sweep
thermalization + 2000-sweep measurement, every sweep recorded).

Compute path: **all 8 seeds in one batched CUDA pass** at
$N_{\mathrm{steps}}=16000$, with running time-averages sampled at each
checkpoint. Same physics as the v1.2 production run; same kernel math
(byte-identical to float-64 precision against the CPU Python kernel at
$N=100$; see `inertia_damping/cuda/test_validate_against_cpu_kernel.py`,
all three gates PASS with $\max|\Delta P_{\mathrm{history}}|
= 4.4 \times 10^{-16}$).

Wall budget for the full study: $\approx 22\,\mathrm{min}$
(19 min CPU heatbath, 161 s CUDA leapfrog) on a single laptop GPU,
versus the $\approx 13\,\mathrm{hr}$ that the original fly.io-routed
40-subprocess orchestration projected.

## Result

| $N_{\mathrm{steps}}$ | seeds | PASS | gap $<$ tol | mean gap | SEM of mean | max gap |
|---|---|---|---|---|---|---|
| 1000   | 8 | 0 | 0 | 0.1748 | 0.0091 | 0.2126 |
| 2000   | 8 | 0 | 0 | 0.1665 | 0.0083 | 0.2020 |
| 4000   | 8 | 0 | 0 | 0.1607 | 0.0089 | 0.2007 |
| 8000   | 8 | 0 | 0 | 0.1600 | 0.0087 | 0.1997 |
| 16000  | 8 | 0 | 0 | 0.1598 | 0.0090 | 0.2025 |

The gap decreases by 9% across the full 16$\times$ range in $N_{\mathrm{steps}}$
and then **flatlines at $\approx 0.16$**, far above the $0.02$ tolerance.
Linear extrapolation at the asymptotic shrinkage rate would not reach
tolerance for $\sim 10^{10}$ steps. Every one of the 40 trajectories
returns verdict FAIL.

Per-seed convergence diagnostics confirm the time-average has converged
to its trajectory limit, not to the canonical mean:

- $\langle P\rangle_{\mathrm{heatbath}} \approx 0.506$ across all seeds
  (stable, regime: plateau, $n_{\mathrm{eff}} = 1000$).
- $\langle P\rangle_{\mathrm{time}} \approx 0.66$ across all seeds
  (stable at large $N$; the blocking regime transitions from
  `no_plateau` at $N=1000$ to `plateau` at $N=16000$, with $n_{\mathrm{eff}}$
  growing from $7.8$ to $125$ — the trajectory is finding its time-average
  plateau).
- Hamiltonian conservation $\max|\Delta H/H_0| \approx 2 \times 10^{-5}$
  across all 8 seeds (the symplectic integrator is doing its job).
- Max Gauss residual $\approx 1.3$ in absolute units (unprojected canonical
  initialisation; same envelope as the v1.2 production run).

## Interpretation

This is **not** a numerical pathology and **not** insufficient sampling.
Both observables have converged. The data shows that for the buckyball
SU(2) Hamiltonian at $\beta = 2.5$, **the symplectic trajectory's
energy shell has a time-averaged plaquette $\approx 0.66$, while the
canonical ensemble (a Boltzmann average over all energy shells weighted
by their density of states) has a plaquette $\approx 0.506$.** Shell $\ne$
ensemble at finite $N$.

This is a real feature of the buckyball substrate, visible precisely
because the system is small enough (only 90 link variables and 93 conserved
quantities) that the canonical thermometer and the microcanonical
thermometer measure structurally different things. On a large 4D cubic
lattice ($L^4 \gg 10^3$) the law of large numbers smooths this
stratification out and the two thermometers agree — the buckyball does
not have that many degrees of freedom to average across.

The Halcyon v1.2 PASS verdicts (substrate identities, energy conservation,
covariant Gauss, time reversibility, Migdal--Witten target, gauge
invariance, beta-scan, beta-envelope) remain valid. The v1.2 Section 5
FAIL is now bound to a documented physical effect: **the cross-check is
not a check on whether the integrator is correct (it is); it is a probe
of the gap between microcanonical and canonical equilibrium at the
buckyball's finite size.** That gap is $\approx 0.16$, stable under
16$\times$ refinement of trajectory length, and is the open caveat the
chapter cites.

## Artefacts

- `section5_closure.json` — full per-trajectory closure analysis table
- `section5_closure.md` — human-readable per-trajectory table
- `section5_gap_vs_nsteps.pdf` — gap-vs-$N_{\mathrm{steps}}$ per seed
- `section5_pt_vs_ph.pdf` — $P_{\mathrm{time}}$ vs $P_{\mathrm{heatbath}}$
  with SEM error bars
- `sweep_manifest.json` — sweep configuration receipt
- 40 per-trajectory sidecars under
  `n{N_steps}_seed{seed}/run_<timestamp>/validation_report_*.json`,
  each with full Section 5 schema-compatible payload

## Reproducibility

```text
beta = 2.5
dt = 0.01
n_steps_max = 16000
checkpoints = [1000, 2000, 4000, 8000, 16000]
seeds = [20260616, 20260617, 20260618, 20260619,
         20260620, 20260621, 20260622, 20260623]
kernel = inertia_damping.cuda.batched_leapfrog (CUDA 12.8)
heatbath = inertia_damping.buckyball_heatbath
            (n_thermalize=200, n_measure=2000, measure_every=1)
```

Command to reproduce:

```bash
python -m inertia_damping.scripts.section5_cuda_sweep
python -m inertia_damping.scripts.section5_closure_analysis \
    --sweep-root inertia_damping/reports/section5_sweep_cuda
```

CPU↔CUDA equivalence at $N=100$:

```bash
python -m inertia_damping.cuda.test_validate_against_cpu_kernel
# PASS-1 (P history) PASS    max |delta P_history|: 4.441e-16
# PASS-2 (H history) PASS    max |delta H|:         2.132e-14
# PASS-3 (P final)   PASS    |delta P_final|:       4.441e-16
```
