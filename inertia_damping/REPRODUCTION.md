# Reproduction Guide: YM Mass Gap in Gigi

This document is the third-party reproduction path for the result documented
in `inertia_damping/reports/ym_mass_gap_in_gigi_receipt.json`. Following
these steps from a fresh clone should produce the same bundle and the same
query output, bit-for-bit (heatbath is deterministic on a fixed seed).

## What you get

After running the steps below, the gigi engine will hold a bundle
`halcyon_ym_mass_gap_demo` with 9 rows (one per β-point). A single GQL
query returns the **holonomy-continuum mass gap for 2D SU(2) Yang-Mills**
(heat-kernel convention, `m = (β/2)·C_2(j=1/2) = 3β/8`), the Wilson-action
string tension (different formula, honest contrast), the Monte-Carlo
measured plaquette, the Migdal-Witten analytical exact plaquette, and the
Davis-framework curvature density + capacity per row.

Theoretical framing:
[gigi/theory/YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md](../../gigi/theory/YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md).

## Prerequisites

- Python 3.11+ with `numpy`, `torch`, `requests`, `scipy` (optional, for
  exact Bessel-function string tension)
- A built local `gigi-stream` binary with the `halcyon` feature flag and
  origin/main >= commit `e37ae9e`
- ~5 GB free disk for the gigi WAL + bundle storage

## Steps

### 1. Build and start gigi-stream locally

```sh
# In the gigi repo
cd /path/to/gigi
cargo build --release --features halcyon --bin gigi-stream
./target/release/gigi-stream.exe &
# wait until http://localhost:3142/v1/health returns {"status":"ok"}
```

### 2. Generate the 9-point β-walk (if not already in repo)

```sh
# In the davis-wilson-lattice repo
cd /path/to/davis-wilson-lattice
PYTHONIOENCODING=utf-8 python -m inertia_damping.buckyball_falls_out_demo
# writes inertia_damping/reports/buckyball_local_falls_out.json
# uses seed=20260616, 9 β-points: 0.5, 1.0, 1.5, 2.0, 2.25, 2.30, 2.50, 2.70, 3.00
# wall-clock ~3-4 minutes per β-point (CPU), ~30 minutes total
```

The committed `buckyball_local_falls_out.json` was generated with this seed
and protocol; if you re-run, you should get byte-identical output. If you
get different P_measured values, check that you are on the same git commit
and that `inertia_damping/buckyball_heatbath.py` `thermalize()` default seed
is `20260616`.

### 3. Push the bundle and run the headline query

```sh
PYTHONIOENCODING=utf-8 python -m inertia_damping.push_ym_mass_gap_bundle
# This script:
#  - drops any existing halcyon_ym_mass_gap_demo bundle
#  - creates the bundle with the holonomy-continuum schema
#  - inserts 9 rows
#  - runs a post-push SELECT * verification (must return 9 rows or the
#    script exits with stage='verification_count_mismatch')
#  - fires the headline query and pretty-prints the result table
#  - writes inertia_damping/reports/ym_mass_gap_in_gigi_receipt.json
```

Expected output table (last column is the Davis capacity):

```
   beta  m_hol(heat)  sigma_W(Wil)   <P>_meas  <P>_exact   C_proxy
  ----- ------------ ------------- ---------- ---------- ---------
   0.50       0.1875        2.0898     0.1224     0.1237     1.139
   1.00       0.3750        1.4263     0.2359     0.2402     1.309
   1.50       0.5625        1.0667     0.3703     0.3441     1.588
   2.00       0.7500        0.8367     0.4414     0.4331     1.790
   2.25       0.8438        0.7509     0.4546     0.4720     1.834
   2.30       0.8625        0.7355     0.5008     0.4793     2.003
   2.50       0.9375        0.6789     0.5303     0.5072     2.129
   2.70       1.0125        0.6293     0.5366     0.5330     2.158
   3.00       1.1250        0.5658     0.5786     0.5679     2.373
```

### 4. The headline GQL query (reproducible by hand)

```sh
curl -s http://localhost:3142/v1/gql -H "Content-Type: application/json" \
  -d '{"query": "SELECT beta, m_holonomy_continuum_heat_kernel_su2, string_tension_wilson_fundamental, P_measured, P_exact_migdal_witten, delta_P, C_proxy FROM halcyon_ym_mass_gap_demo;"}'
```

Returns a JSON `{"rows": [...], "count": 9}` body.

## Composition queries (verified working today)

```sh
# Inverse problem: which beta gives target gap in [0.9, 1.0]?
curl -s http://localhost:3142/v1/gql -H "Content-Type: application/json" \
  -d '{"query": "SELECT beta, m_holonomy_continuum_heat_kernel_su2 FROM halcyon_ym_mass_gap_demo WHERE m_holonomy_continuum_heat_kernel_su2 BETWEEN 0.9 AND 1.0;"}'
# returns beta=2.5 only

# Aggregation: bundle-level statistics on the gap column
curl -s http://localhost:3142/v1/gql -H "Content-Type: application/json" \
  -d '{"query": "INTEGRATE halcyon_ym_mass_gap_demo OVER C2_of_j_half MEASURE AVG(m_holonomy_continuum_heat_kernel_su2), MIN(m_holonomy_continuum_heat_kernel_su2), MAX(m_holonomy_continuum_heat_kernel_su2), COUNT(*);"}'
# returns global stats since C2 is constant across rows
```

## Known limitations (be honest about scope)

1. **2D only.** This bundle reflects 2D SU(2) Yang-Mills on the truncated
   icosahedron (buckyball, χ=2). The 4D Yang-Mills mass gap is the open
   Clay problem; the closed-form `3β/8` does NOT apply in 4D. The 4D
   companion bundle `halcyon_ym4_glueball_demo` extracts the gap from
   Monte-Carlo plaquette correlators (per H5 / M1-M4 of the hypothesis
   doc) and carries error bars.

2. **Convention pinned.** The `3β/8` formula uses the **Migdal heat-kernel
   convention** where `β = 1/(g²a²)`. Wilson-action convention has
   `β_W = 2N/(g²a²)` and the same physics reads `m·a² = 3/(2β_W)`. The
   bundle stores both readings as separate columns.

3. **No phase column.** The v1 bundle had a `phase` column with
   confined/at-β_c/deconfined labels; these are 4D Wilson terminology and
   physically meaningless for 2D YM (which is confined at all finite β by
   Polyakov/Mermin-Wagner). The column was dropped.

4. **`gigi-stream` is single-instance, no clustering.** This reproduction
   assumes you can run a local gigi-stream binary. Pushing to a deployed
   instance (e.g. `gigi-stream.fly.dev`) requires `GIGI_API_KEY` env var
   and changes nothing else about the script behavior.

5. **No artifact verification today.** A future hardening step would add
   a sha256 hash of the bundle export to the receipt and a CI step that
   regenerates the bundle and diffs. Today the verification is the
   3rd-party re-run of these steps.

## Cross-references

- [push_ym_mass_gap_bundle.py](push_ym_mass_gap_bundle.py) — the push script
- [buckyball_falls_out_demo.py](buckyball_falls_out_demo.py) — the β-walk
- [buckyball_yangmills_exact.py](buckyball_yangmills_exact.py) — Migdal-Witten reference
- [buckyball_heatbath.py](buckyball_heatbath.py) — Kennedy-Pendleton SU(2) heatbath
- [buckyball_graph.py](buckyball_graph.py) — truncated icosahedron (V=60, E=90, F=32)
- [reports/ym_mass_gap_in_gigi_receipt.json](reports/ym_mass_gap_in_gigi_receipt.json) — receipt JSON
- [reports/buckyball_local_falls_out.json](reports/buckyball_local_falls_out.json) — source data
