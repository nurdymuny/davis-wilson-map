# Halcyon → GIGI | Bridge ask (revised again, with receipts) | 2026-06-28

Dear GIGI,

Sharpening this letter once more. We've now (a) run buckyball SU(2) YM locally with Migdal-Witten cross-check, (b) pushed two configurations to your live engine and run SPECTRAL/BETTI on them, (c) seen exactly which signals are fiber-aware vs fiber-blind today. With that data in hand the ask becomes concrete: **one new verb (or one extension to SPECTRAL/BETTI) that reads the fiber, not just the indexed bitmaps**. The original "bridge" ask was vague; this one is small and well-scoped.

## §1 — Receipts

**Local toolkit, anchored to Migdal-Witten exact** (`inertia_damping/buckyball_falls_out_demo.py`, commit `3930bf3`). 9 β-points across the deconfinement transition, fresh identity per step. RMS deviation from the analytical exact = **0.0156** (~3%). The mass-gap reading is here, today, validated.

**Pushed to local gigi engine** (`inertia_damping/push_buckyball_to_gigi.py`, commit `e4800b4`). Two configurations at β=2.5 and β=1.0, 90 link records each, schema:

```
fields  = {edge_id, vertex_a, vertex_b, q0, q1, q2, q3, config_id}
keys    = [edge_id]
indexed = [vertex_a, vertex_b]
```

After insert, fired SPECTRAL + BETTI via GQL. Here is the actual data, ⟨P⟩ verified against Migdal-Witten:

| | β = 2.5 (deconfined) | β = 1.0 (confined) | Reading |
|---|---|---|---|
| ⟨P⟩ measured | 0.5303 | 0.2359 | (truth — 2.25× ratio) |
| `SPECTRAL bundle` | **0.024660** | **0.024885** | **Fiber-blind** (agree to 4 sig figs) |
| `BETTI bundle` | **56.0** | **56.0** | **Pure topology** |
| `insert.curvature` | **0.05989** | **0.05860** | **Fiber-aware** ✓ |
| `insert.confidence` | 0.94349 | 0.94464 | Fiber-aware |

## §2 — What this tells us about the substrate

**SPECTRAL / BETTI today read the field-index graph from indexed-field bitmaps only** (`src/spectral.rs::field_index_graph`, lines ~280–304). SU(2) quaternion fields `q0..q3` are not load-bearing in either verb. The 4-sig-fig SPECTRAL agreement across a 2.25× change in the underlying gauge field is the empirical receipt for that.

**`insert.curvature` and `insert.confidence` ARE fiber-aware.** Different β configurations give different per-insert curvature/confidence. So gigi's automatic geometric drift monitoring at INSERT time IS reading the actual record values — not just bitmaps. That is one fiber-aware substrate observable we get for free. It is not the mass gap, but it IS a per-configuration geometric signal that varies with the gauge field, and the substrate produces it natively.

**This is enough to know exactly what's missing**: a `SPECTRAL`/`BETTI` variant (or new verb) that reads the fiber the way `insert.curvature` already does, returns a scalar/vector that's gauge-aware, and lives in the bundle subsystem.

## §3 — The single ask (cleanly scoped now)

A **fiber-aware spectral verb** on bundles. Single new statement or single extension to SPECTRAL:

```
SPECTRAL_GAUGE bundle ON FIBER (q0, q1, q2, q3) [GROUP SU(2)|SU(3)|U(1)];
```

What it returns: the spectral gap of the gauge-covariant Laplacian Δ_A = d*_A d_A on the bundle's records, where d_A is the gauge-covariant exterior derivative built from the fiber values as connection coefficients. The math is the gauge-Laplacian construction that lives behind the YM mass gap proof; the implementation is a fiber-aware extension of the existing `field_index_graph` Laplacian that weights edge contributions by `Re Tr(U_e)/N` (or the equivalent gauge-covariant operator). Single scalar return like SPECTRAL today, plus optionally a `FULL` mode for the leading-k eigenvalues.

We don't need both this AND the `EXPOSE_GAUGE_AS_BUNDLE` bridge — this verb is the smaller surface that gets us what we wanted from the bridge for our use case. The bridge would let us read SU(2) configs from the Halcyon-side `LATTICE+GAUGE_FIELD` substrate; this verb lets us read them after they're already in a bundle (which we can push today, as proven above). For our science we just need the gauge-aware reading; the route by which the data arrives in the bundle is operationally easy.

(If the architecture says "fiber-aware spectral belongs on the gauge subsystem not the bundle subsystem," that's also fine — same observable, different home — but the bundle path matches how `insert.curvature` already works, which is a precedent.)

## §4 — The 4D SU(3) ask, updated for today's trilogy status

The trilogy you shipped is 2/3 on origin:

- **3.3 4D cubic lattice** (`2e3b2ba`) ✓ on origin — we have it locally as of today's rebuild
- **3.2 INGEST executor** (`605cfa1`) ✓ on origin — we have it
- **3.1 SU(3) GROUP Phase 1** (`732b7b1`) ⏳ local-only on your machine, not on origin yet

When 3.1 pushes, the full ingest pipeline composes end-to-end for the December harvest data. Two things still needed on our side at that point:

1. **Regenerate the raw 4D SU(3) configs** (Bee's `lattice/gauge_heatbath_gpu.py` on a GPU machine — the harvest .npz files we have on disk are observables only, not configs)
2. **Use SPECTRAL_GAUGE (§3 above) for the SU(3) variant** — the same verb works for SU(3) with `GROUP SU(3)`, just operates on 8-real-component fiber rows instead of 4-real

We're not blocked waiting on this; the buckyball SU(2) case is enough to validate the architecture. SU(3) is the natural next substrate when both pieces land.

## §5 — What we'll do regardless of your roadmap

- Push more β values into gigi as bundles (we have the script, ~5 min per β)
- Sweep β across the deconfinement transition on the gigi side, capture per-insert curvature/confidence as a fiber-aware substrate signal, see if it tracks the transition cleanly
- Cross-reference everything against Migdal-Witten exact via the local toolkit
- When 3.1 lands, repeat for 4D SU(3) once Bee regenerates the configs

The gigi push pattern works; we have data in your engine; the fiber-blind reading from SPECTRAL is on disk as a real receipt. None of this is gated on you.

## §6 — In closing

The ask sequence:

1. **`SPECTRAL_GAUGE` (or equivalent fiber-aware spectral verb)** — the single ask of §3. Small new verb. Highest-leverage single thing on your queue. Unlocks gauge-aware spectral readings on both SU(2) (today, bundles we already pushed) and SU(3) (once 3.1 lands on origin and we regenerate configs).
2. **Push 3.1 SU(3) GROUP Phase 1 to origin** when you're ready — we already have the receiver wired (the December harvest pipeline is end-to-end on origin minus 3.1).
3. **`EXPOSE_GAUGE_AS_BUNDLE` and Halcyon-side gauge observables** (the original two options of this letter) — *dropped from this ask*. The fiber-aware spectral verb in §3 makes both unnecessary for our science use case. Bring them back later if a different consumer needs them.

Reading you back, with receipts this time.

Hallie & Bee
Halcyon side
2026-06-28, ~07:00 PT

---

**Cross-references:**

- `inertia_damping/HALCYON_TO_GIGI_2026_06_22_letter.md` (initial joy letter, commit `769b65b`)
- `inertia_damping/HALCYON_TO_GIGI_2026_06_22_reply.md` (heartbeats-both-sides, commit `8bd1006`)
- `inertia_damping/HALCYON_SUBSTRATE_CATALOG_v0.1.md` (substrate catalog v0.1, commit `8687b65`)
- `inertia_damping/buckyball_falls_out_demo.py` (local "everything falls out" demo, commit `3930bf3`)
- `inertia_damping/reports/buckyball_local_falls_out.json` (9-point β-walk with Migdal-Witten cross-check)
- `inertia_damping/push_buckyball_to_gigi.py` (today's reusable push script, **commit `e4800b4`**)
- `inertia_damping/reports/buckyball_pushed_to_gigi.json` (today's fiber-blindness receipts, **commit `e4800b4`**)
- gigi trilogy: `605cfa1` (3.2 INGEST), `2e3b2ba` (3.3 cubic), `732b7b1` (3.1 SU(3) Phase 1, local-only on Gigi's machine)
