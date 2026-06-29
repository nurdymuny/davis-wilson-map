# `--use-gigi` flag spec — route the Halcyon orchestrator through GIGI's substrate

**Status:** Design lock, awaiting Sprint A+B deploy on `gigi-stream.fly.dev`
**Author:** Bee Rosa Davis, with Claude (Anthropic)
**Date:** 19 June 2026
**Depends on:** GIGI Parts I–V live on production (deployed); GIGI Sprint A+B deploy (pending — currently on `origin/main` at commit `7d8f6e4`, not yet on fly.dev)
**Companion:** ``papers/solves_vol4_ym_mass_gap.tex`` (the chapter whose v1.3 verdict this unlocks), ``gigi/theory/halcyon/HALCYON_PART_I_GATES.md`` § MEASUREMENT GATE (the coverage analysis this implements)
**Goal:** add ``--use-gigi`` to ``run_validation_report.py`` so the orchestrator's gauge-field phases run on GIGI's Rust engine instead of the Python kernel. Unlocks the convergence study that would close the Section 5 honest FAIL.

---

## 0. Letter to future-Bee

We shipped the substrate consolidation as a *queryable surface* (Solves Vol. 4) but the orchestrator that produces Halcyon's deployed verdict still uses the Python kernel. Every Stage 2 run is 46 minutes wall. Two assumptions made sense at the time: that the Python kernel was fast enough for production runs, and that the GIGI substrate was a parallel substrate, not the substrate.

Both are wrong now. The Python kernel is ~50× slower than the Rust engine, and the substrate IS the substrate — Halcyon's verdict reads off the same memory as Marcella and Solves Vol. 4. Running the orchestrator on Python costs us the v1.3 verdict that would close the honest FAIL on Section 5.

This spec adds ``--use-gigi`` to ``run_validation_report.py``. The flag routes the gauge-field phases through GIGI's GQL surface; the validator stays Python (per the original non-ask in the substrate-consolidation letter). After Sprint A+B deploys, the convergence study that was deferred to "the apparatus characterisation phase" is one ``--use-gigi`` invocation away.

—Bee + Claude

---

## 1. Phase-by-phase routing

Run ``run_validation_report.py`` is structured as numbered ``_phase()`` calls. Per the measurement-gate findings doc (``inertia_damping/HALCYON_PART_I_MEASUREMENT_GATE_FINDINGS.md``, currently rendering 100 % coverage), every gauge-field phase has a GQL home; the validator + JSON/markdown rendering stay Python. The routing table:

| Phase (orchestrator line) | Current (Python kernel) | After ``--use-gigi`` (GIGI GQL) | Notes |
|---|---|---|---|
| ``build_graph`` (168) | ``buckyball_graph.build_truncated_icosahedron()`` | ``LATTICE halcyon_canonical_buckyball FROM TRUNCATED_ICOSAHEDRON TOPOLOGY 'S2' PERSIST;`` | Persist so subsequent phases reuse the declaration. Repeat declarations are idempotent. |
| ``heatbath_thermalize`` (175) | ``buckyball_heatbath.heatbath_sweep`` × N | ``GAUGE_FIELD U INIT IDENTITY;`` then ``GIBBS_SAMPLE U BETA β N_SWEEPS N MEASURE_EVERY 1 MEASURE (MEAN(PLAQUETTE), Q_SURROGATE) SEED s;`` | Single GQL call; chain comes back in the Rows envelope. |
| ``initialize_E_canonical`` (183) | ``buckyball_integrator.initialize_E_canonical`` | ``E_FIELD E ON GAUGE_FIELD U INIT MAXWELL_BOLTZMANN BETA β SEED s;`` | Mirrors Part IV. |
| ``leapfrog`` (197) | ``buckyball_observables.integrate_with_states`` | ``SYMPLECTIC_FLOW U FROM (U=U, E=E) BETA β DT dt N_STEPS n PROJECT_GAUSS {tikhonov: 1e-14, cg_tol: 1e-10, cg_max_iter: 200} MEASURE_EVERY k MEASURE (H_TOTAL, MEAN(PLAQUETTE), Q_SURROGATE, GAUSS_RESIDUAL_MAX) SEED s;`` | The load-bearing speed-up phase. |
| ``dump_trajectory`` (226) | local serialization | introspect U, E via ``GET /v1/gauge_field/{name}`` + local serialize | The sidecar still gets written for reproducibility. |
| ``microcanonical cross-check`` (273) | second leapfrog | ``SYMPLECTIC_FLOW`` from a fresh ``IDENTITY`` start | Same shape as the main leapfrog. |
| ``canonical heatbath`` (286) | ``buckyball_heatbath.thermalize`` | ``GIBBS_SAMPLE`` at the canonical β, long n_sweeps | The 2048-sample chain that feeds Section 5's `P_heatbath`. |
| ``beta-scan`` (305) | per-β Gibbs over 6 betas | ``GIBBS_SAMPLE`` loop over the β list | Cheap; the loop is sequential GQL calls. |
| ``beta-envelope sweep`` (335) | per-β leapfrog over 5 betas | ``SYMPLECTIC_FLOW`` loop over the β list | The dominant cost today. |
| ``sector-classifier`` (422) | Python | Python (reads observables off the GIGI trajectory) | Validator-side; no routing change. |
| ``write sidecar`` (447) | local serialization | local serialization (state introspected from GIGI) | No routing change. |
| ``generate_report`` (479) | Python validator | Python validator | **By design (the original non-ask).** The falsifiability validator stays external. |

**The wire is HTTP via `/v1/gql`.** Per the Phase A audit, embedded PyO3 is the right shape for per-sweep production calls; we don't need it here because the orchestrator issues one GQL call per phase (or per β), not per sweep. The audit's HTTP latency concern (round-trip dominates) only bites when you batch by sweep; batching by phase amortizes the round-trip over thousands of sweeps.

## 2. Receipts the flag must produce

Two regression contracts and one new gate.

### R1 — bit-identity of the public verdict at fixed seed

The flag must reproduce Halcyon's published v1.2 verdict bit-for-bit at fixed seed AND fixed CSPRNG. Since GIGI uses xorshift64* and Halcyon's Python kernel uses NumPy PCG64, byte-equality is structurally impossible. The contract is therefore *engine-side* bit-identity:

- ``--use-gigi`` run at seed ``s`` against gigi-stream produces a verdict JSON whose category verdicts and gate counts (8 PASS, 1 NOT_APPLICABLE, 1 FAIL) match Halcyon's deployed v1.2.
- The numerical values (canonical ⟨P⟩, Gauss residual max, Migdal-Witten gap, β-scan and β-envelope tables) match to within the documented Flyvbjerg-Petersen blocked SEM band, not to the bit.
- The buffer SHA (the SNAPSHOT receipt) reproduces between consecutive ``--use-gigi`` runs at the same seed on the same engine, by Part V's bit-identity contract.

### R2 — tolerance-band agreement against the Python-kernel run

A second receipt: the ``--use-gigi`` numerics agree with a same-seed Python-kernel run inside the same blocked SEM. This is the Bee-CSPRNG-decision-(c) contract carried forward to the orchestrator level. The receipt is:

  - ``ratio = |gigi_result.canonical_P - python_result.canonical_P| / SEM`` stays under 3.0 across all 6 β-scan points.

### G_LIVE_F (new) — the orchestrator routes cleanly through GIGI

A new live-test gate, mirroring Phase B/E:

  - ``test_gigi_live_phase_f_orchestrator.py::test_G_LIVE_F0_use_gigi_smoke``
    Run ``run_validation_report.py --use-gigi --steps 100 --canonical-sweeps 200`` (tiny problem) against gigi-stream; assert the report JSON returns 10 categories and the expected schema_version (1.2 or 1.3).
  - ``test_G_LIVE_F1_use_gigi_bit_identity_seed_run``
    Two consecutive ``--use-gigi`` runs at the same seed produce identical category verdicts + same buffer SHA from the SNAPSHOT step.
  - ``test_G_LIVE_F2_use_gigi_band_agreement_with_python``
    Same seed, ``--use-gigi`` vs Python-kernel, canonical ⟨P⟩ within 3σ of the documented SEM.

## 3. CLI surface

```
python -m inertia_damping.run_validation_report \
    --beta 2.5 --steps 1000 --seed 20260616 \
    --use-gigi \
    --gigi-url https://gigi-stream.fly.dev \
    --gigi-api-key $GIGI_API_KEY \
    --output inertia_damping/reports/
```

New flags:

- ``--use-gigi`` — route the gauge-field phases through GIGI's GQL surface.
- ``--gigi-url <url>`` — engine URL; defaults to ``GIGI_URL`` env then ``http://localhost:3142``.
- ``--gigi-api-key <key>`` — bearer auth; defaults to ``GIGI_API_KEY`` env. Required for fly.dev.
- ``--gigi-snapshot`` — additionally snapshot the post-thermalization canonical buffer to GIGI's WAL (writes one ``OP_GAUGE_FIELD_SNAPSHOT`` entry per ``--use-gigi`` run; ``--i-confirm-this-writes-to-the-engine`` required for non-localhost).

## 4. Implementation shape (when Sprint A+B deploys)

Five small modules:

1. ``inertia_damping/gigi_orchestrator/__init__.py`` — package init.
2. ``inertia_damping/gigi_orchestrator/phase_runners.py`` — one function per phase ID, takes the GIGI client + the orchestrator's args namespace, returns the same shape the existing Python phase produces.
3. ``inertia_damping/gigi_orchestrator/translate_chains.py`` — converts GIGI's CamelCase Rows envelopes into the orchestrator's existing dict-of-lists shape, so the validator doesn't know whether the chain came from Python or GIGI.
4. ``run_validation_report.py`` — thin dispatcher: when ``--use-gigi`` is set, each ``_phase()`` call routes to the corresponding ``gigi_orchestrator.phase_runners.*`` function; otherwise the existing Python kernel path runs unchanged.
5. ``test_gigi_live_phase_f_orchestrator.py`` — the three gates above.

The validator (``validation_report.py``) does not change. The sidecar (``final_state.npz``) does not change. The report JSON shape does not change. The only difference is *where* the gauge-field numbers come from.

## 5. The Section 5 convergence study (what this flag unlocks)

With ``--use-gigi`` shipped, the trajectory-length sweep that closes Section 5 becomes:

```bash
for n in 1000 2000 4000 8000 16000; do
  for seed in 20260616 20260617 20260618 20260619 20260620 \
              20260621 20260622 20260623; do
    python -m inertia_damping.run_validation_report \
      --use-gigi --beta 2.5 --steps $n --seed $seed \
      --output inertia_damping/reports/sweep/n${n}_seed${seed}/
  done
done
```

40 trajectories. At ~50 ms substrate compute per 1000 steps post-Sprint-A+B, the SYMPLECTIC_FLOW load is ~50 s total engine wall; verifier + JSON + auth overhead dominates at ~30–60 minutes wall. Per the wall-time three-number breakdown that lands in the chapter when Sprint A+B deploys.

Then a single analysis script reads the 40 sidecars and produces:

- ``P_time_n_eff`` vs ``n_steps`` curve, per seed.
- ``microcanonical_canonical_gap`` vs ``n_steps`` curve, per seed.
- Seed-aggregated convergence plot: does the gap close as ``n_steps → ∞``?

If the answer is **yes**: Section 5 promotes from FAIL to PASS. Verdict goes 8/10 → 9/10. Chapter promotes to v4.1 with the closure receipt. Davis Geometric ships a substantively-stronger result.

If the answer is **no**: we learn something real about the 93-DOF apparatus. The chapter cites the failed-to-close attempt, which is also a receipt and arguably more interesting (the framework knows where its claim ends, and is willing to test it).

Either outcome is strictly better than holding the FAIL static.

## 6. What this does NOT enable

Worth naming so nobody asks later:

- **Continuum extrapolation.** The open Clay inequality is still open. Cheaper data points don't change scope.
- **Hardware kill-chain validation.** Engineering, not compute.
- **The continuum limit of the lattice gap.** v6's strong-coupling theorem still holds at strong coupling; the continuum is still open.
- **Marcella's gauge corpus.** Different sprint; ``--use-gigi`` is orchestrator-side, not Marcella-side.

The flag closes Section 5. It does not extend v6.

## 7. Receipts at sprint completion

Phase F live gates green; bit-identity smoke test passes; one example ``--use-gigi`` run reproduces the v1.2 verdict against Halcyon's deployed JSON within the SEM band; the convergence-study script is staged but not yet fired.

## 8. Sprint sequence after this lands

1. **`--use-gigi` flag ships (this spec).** Phase F live gates green against fly.dev.
2. **Convergence study fires.** 40-trajectory sweep, ~30–60 min wall total.
3. **Analysis script.** Aggregate gap vs n_steps; decide Section 5 closure.
4. **v1.3 validation report** if Section 5 closes; v1.2.1 if it doesn't (with the closure attempt cited as evidence).
5. **Solves Vol. 4 v4.1** with the new verdict + convergence-study appendix.
6. **davisgeometric.com/halcyon** redeploys with the new verdict.

—Bee + Claude
