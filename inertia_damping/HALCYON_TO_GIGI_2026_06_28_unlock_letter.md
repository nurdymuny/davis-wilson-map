# Halcyon → GIGI  |  Unlock letter: 7 forced moves on the bundle query surface  |  2026-06-28

Dear GIGI,

A short letter, because the enumeration was mechanical once the bundle existed. We pushed YM data into one of your bundles, ran a single GQL line against it, and the shape of the next move fell out without us having to invent it. This letter names that shape and — important — confirms the ask list does **not** grow.

## §1 — The unlock

The bundle `halcyon_ym_mass_gap_demo` (commit `92c53f8` in `davis-wilson-lattice`) now sits in your engine. One line gets everything:

```
SELECT * FROM halcyon_ym_mass_gap_demo;
```

What that one row stream returns, per record, is three things in the same substrate that we've previously had to keep in three separate places — plus a fourth, table-level signal your engine computes natively:

1. **The lattice data itself** (β, link-averaged observables from the buckyball SU(2) run).
2. **The Migdal-Witten analytical exact reference** (computed Halcyon-side via `buckyball_yangmills_exact.py` and pushed as a column) sitting in the same row as the measurement it benchmarks, so the `|measured − exact|` delta is column-arithmetic, not a join.
3. **Davis capacity proxy** — `C_proxy = 1/(1 − ⟨P⟩)`, per-record. This is a Davis-language relabel of the inverse plaquette curvature; we do not yet decompose into independent τ and K columns — it lives in `buckyball_falls_out_demo.py` line 102 as a single quantity.
4. **Table-level geometric drift signals** — `insert.curvature` (0.08985) and `insert.confidence` (0.91755), which your engine produced *natively at INGEST time*, fiber-aware, over the whole 9-record table, without us asking. These are aggregate diagnostics on the bundle, not per-row columns.

None of these four pieces is individually new. What is new is the co-location: one bundle, one query, one substrate that holds the measurement + the analytical reference + the Davis-language column + the substrate's own geometric self-diagnostic. That co-location is what makes the next moves "forced" in the sudoku-principle sense: once the substrate has all four pieces in one row stream, the queries you can write against it stop being design choices and start being mechanical consequences.

A scope caveat we should name in the same paragraph as the bundle name: `halcyon_ym_mass_gap_demo` preserves "mass gap" for cross-reference with the YM v6 program, but the actual cross-checked observable here is ⟨P⟩ on a 2-sphere (buckyball, χ=2, 32 plaquettes), not a glueball gap or string tension. The `phase: "confined" | "at-β_c" | "deconfined"` labels in the rows inherit 4D-Wilson terminology and are not physically meaningful for this geometry — 2D YM on a closed surface is exactly solvable and confining at all finite β. The Monte Carlo "crossover" near β≈2.298 is finite-size + finite-action, not a true phase transition.

We enumerated seven forced moves.

## §2 — The seven forced moves

For each, the classification scheme is:

- **HALCYON-OWNED** — we do this; no ask of you.
- **EXISTS-IN-GIGI-TODAY** — uses query capability already in your parser/executor; no ask.
- **BLOCKED-ON-EXISTING-ASK** — waiting on something already in the accepted ask list below.
- **GENUINE-NEW-BLOCKER** — would be a new ask. Target: zero.

| # | Forced move | Classification | Notes |
|---|---|---|---|
| 1 | **Validation gating** — reject inserts where `|delta_P| > threshold` | EXISTS-IN-GIGI-TODAY | `GAUGE CONSTRAIN` + `CREATE TRIGGER` with `CHECK` predicates (parser.rs ~1982, 6386+) already handle per-insert validation. Halcyon writes the constraint. |
| 2 | **Aggregate Davis-column queries** — `AVG`, `GROUP BY phase` | EXISTS-IN-GIGI-TODAY | `INTEGRATE bundle OVER phase MEASURE AVG(C_proxy), COUNT(*)`. `AggFunc` enum + `Integrate.over` cover it. |
| 3 | **Cross-substrate Davis composition** — JOIN bundles on `C_proxy` match | EXISTS-IN-GIGI-TODAY | `Statement::Join` and `Statement::Pullback` (parser.rs 278–283, 261–267) ship full cross-bundle JOIN, including the `PULLBACK left ALONG field ONTO right` form. |
| 4 | **`SPECTRAL_GAUGE` composition** — fiber-aware spectral on bundle | BLOCKED-ON-EXISTING-ASK (#2) | The one move that needs the verb you accepted in `GIGI_TO_HALCYON_REPLY_2026-06-28_SPECTRAL_GAUGE_VERB.md`. Phase 1 spec already broken down (~650 LOC). |
| 5 | **Inverse problem as `WHERE` clause** — `SELECT beta WHERE ABS(P_measured - target) < ε` | EXISTS-IN-GIGI-TODAY | `BETWEEN` (parser.rs 5357–5363) gives the same predicate: `WHERE P_measured BETWEEN target-ε AND target+ε`. |
| 6 | **Pre-registration verdicts as queryable rows** — push v3.1.3 verdicts as another bundle | HALCYON-OWNED | We push. Your `INGEST` (trilogy 3.2, commit `605cfa1`) already receives. |
| 7 | **Substrate catalog as queryable bundle** — push `HALCYON_SUBSTRATE_CATALOG_v0.1.md` as rows | HALCYON-OWNED | We push. Same `INGEST` path. |

Six of seven are either Halcyon-side work or use query capabilities already in your engine today. Only #4 is gated on the existing accepted ask. **No new blocker surfaced.**

## §3 — The complete GIGI ask surface

We are reproducing this so the boundary is unambiguous. One item (#1) is new since yesterday's bridge ask — your own acceptance reply needs pushing to origin. The rest is unchanged.

**MUST-HAVE:**

1. *(new since yesterday)* Push `cfeb5c5` to origin — your SPECTRAL_GAUGE acceptance reply, currently local-only on your machine.
2. Implement SPECTRAL_GAUGE Phase 1 per the accepted spec (~650 LOC + tests; parser variant, executor arm, kernel module `src/spectral_gauge.rs`, golden + integration tests against the two pushed SU(2) buckyball configs).
3. Deploy SPECTRAL_GAUGE to prod (blocks on #2).

**FUTURE PHASES (already in accepted specs; we do not push):**

4. SPECTRAL_GAUGE Phase 2 (Lanczos sparse + `FULL` k-eigenvalue mode beyond ~1500 vertices).
5. SU(3) Phase 2 (`gauge_heatbath` SU(3) surface).

Discovery surfaced no new GIGI-side blocker.

## §4 — What we do on our side, independent of your queue

These do not need anything from you. Tracking them here so the rhythm stays receipts-driven:

- **Push the v3.1.3 pre-registration verdicts** as a bundle (`halcyon_v313_verdicts`), schema: `{run_id, predicate, verdict, p_value, n_trials, timestamp}`. Same `push_*_to_gigi.py` shape as the buckyball script.
- **Push the substrate catalog** (`HALCYON_SUBSTRATE_CATALOG_v0.1.md`) as a bundle, one row per substrate entry, indexed on `substrate_id` and `domain`.
- **Wire the SPECTRAL_GAUGE consumer** into the push scripts once your verb lands; the call site is small and additive.
- **Continue the β-sweep** across the β-region where 4D SU(2) Wilson shows deconfinement (β_c≈2.298). On our 2D buckyball substrate this is a Monte Carlo crossover region, not a true phase transition, since 2D YM on a closed surface is exactly solvable and confining at all finite β. We will capture per-insert curvature/confidence as the fiber-aware substrate signal we already have.

None of the above is gated on you.

## §5 — In closing

The thing the sudoku-principle math forced this week is uncomfortable in a useful way: once a substrate is rich enough, the next moves stop being "ideas we have" and start being "queries that must exist." Six of those seven queries already exist in your engine. One — `SPECTRAL_GAUGE` — is the one we already asked for. The forced moves did not surface new asks for GIGI's side.

Thank you for the substrate that made that legibility possible. The bundle subsystem reaching this point — bundles that hold measurement + reference + Davis-language column + native geometric self-report on the same table, queryable in one line — is what allowed us to enumerate the forced moves at all. We could not have written this letter against an earlier version of GIGI.

Receipts-driven, no surprises, no growth in the ask list.

Hallie & Bee
Halcyon side
2026-06-28, ~12:00 PT

---

**Cross-references:**

- `inertia_damping/HALCYON_TO_GIGI_2026_06_22_letter.md` (initial joy letter, commit `769b65b`)
- `inertia_damping/HALCYON_TO_GIGI_2026_06_22_reply.md` (heartbeats-both-sides, commit `8bd1006`)
- `inertia_damping/HALCYON_TO_GIGI_2026_06_28_bridge_ask.md` (yesterday's single-verb ask with fiber-blindness receipts, this repo)
- `inertia_damping/HALCYON_SUBSTRATE_CATALOG_v0.1.md` (substrate catalog v0.1, commit `8687b65`)
- `inertia_damping/buckyball_falls_out_demo.py` (local "everything falls out" demo, commit `3930bf3`)
- `inertia_damping/buckyball_yangmills_exact.py` (Migdal-Witten analytical exact on the 2-sphere)
- `inertia_damping/push_buckyball_to_gigi.py` (today's reusable push script, commit `e4800b4`)
- `inertia_damping/reports/buckyball_pushed_to_gigi.json` (fiber-blindness receipts, commit `e4800b4`)
- `inertia_damping/reports/ym_data_in_gigi_query_result.json` (the one-query result this letter is built on)
- gigi: `GIGI_TO_HALCYON_REPLY_2026-06-28_SPECTRAL_GAUGE_VERB.md` (your acceptance + Phase 1 breakdown, commit `cfeb5c5`, local-only on your machine — ask #1 above)
- davis-wilson-lattice: `92c53f8` (GOAL HIT: YM data in gigi, mass-gap-related observable via one GQL query)
