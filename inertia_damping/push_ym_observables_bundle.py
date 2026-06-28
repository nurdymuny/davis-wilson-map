"""push_ym_observables_bundle.py — YM mass-gap-related data in gigi, accessible via one GQL query.

Goal: "YM data in gigi + the mass gap visible via a simple GQL query" (Bee, 2026-06-28).

Approach: take the local-toolkit β-walk results
(`inertia_damping/reports/buckyball_local_falls_out.json` — 9 β-points, each cross-checked
against the Migdal-Witten analytical exact ⟨P⟩ formula via `buckyball_yangmills_exact.py`)
and push them as a bundle in gigi with one row per β value. The bundle holds the
mass-gap-related observable (`P_measured`), the analytical reference (`P_exact_migdal_witten`),
the deviation (`delta_P`), Davis capacity (`C_proxy`), topological charge (`Q_surrogate`),
and a phase label.

Then any consumer (Halcyon, Marcella, another Claude instance, an external researcher)
can run **one GQL query** to see the entire YM dataset and the mass-gap-related reading:

    SELECT * FROM halcyon_ym_mass_gap_demo;

That's the goal hit. The mass gap signature is visible:
  - β=0.5 confined:   P_measured = 0.122, C_proxy = 1.14, Q = 7.57
  - β=2.30 at β_c:    P_measured = 0.501, C_proxy = 2.00, Q = 4.58
  - β=3.0 deconfined: P_measured = 0.579, C_proxy = 2.37, Q = 4.45

The capacity column jumping ~30% across the deconfinement transition (β=2.25 → β=2.50)
is the deconfinement signature visible directly in the GQL result.

This is the "everything falls out" demo from buckyball_falls_out_demo.py, but now living
in gigi and queryable. The Migdal-Witten validation (3% RMS) is preserved in the table:
each row shows P_measured next to P_exact, so any reader can see immediately whether the
substrate's reading matches the analytical reference.

For SU(3): same pattern. Push observables computed from regenerated harvest configs as
rows, query as above. The trilogy (3.1 SU(3) GROUP + 3.2 INGEST + 3.3 4D cubic) is live
on prod as of v230 (gigi commit 732b7b1), so the receiver pipeline is ready.

Usage:
    python -m inertia_damping.push_ym_observables_bundle
    # Or with custom bundle name:
    python -m inertia_damping.push_ym_observables_bundle --bundle my_ym_data \\
        --source inertia_damping/reports/buckyball_local_falls_out.json

Requires:
    - gigi-stream running on http://localhost:3142 (or set GIGI_URL)
    - GIGI_API_KEY env var for prod
    - inertia_damping/reports/buckyball_local_falls_out.json (or alternative source)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import requests


DEFAULT_GIGI_URL = "http://localhost:3142"
DEFAULT_SOURCE = "inertia_damping/reports/buckyball_local_falls_out.json"
DEFAULT_BUNDLE = "halcyon_ym_mass_gap_demo"


def push_observables_bundle(
    *,
    source_path: str,
    bundle_name: str,
    gigi_url: str,
    api_key: Optional[str] = None,
) -> dict:
    with open(source_path) as f:
        walk = json.load(f)
    results = walk["results"]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    print(f"Loaded {len(results)} observations from {source_path}")
    print(f"  Validation: RMS deviation from Migdal-Witten exact = {walk['rms_deviation_vs_exact']:.4f}")

    print(f"\n=== CREATE BUNDLE {bundle_name} ===")
    create_response = requests.post(
        f"{gigi_url}/v1/bundles",
        json={
            "name": bundle_name,
            "schema": {
                "fields": {
                    "beta": "float",
                    "P_measured": "float",
                    "P_exact_migdal_witten": "float",
                    "delta_P": "float",
                    "curvature_density": "float",
                    "C_proxy": "float",
                    "Q_surrogate": "float",
                    "face_q0_mean": "float",
                    "face_q0_std": "float",
                    "n_samples": "integer",
                    "phase": "text",
                },
                "keys": ["beta"],
            },
        },
        headers=headers,
        timeout=30,
    )
    print(f"  HTTP {create_response.status_code}: {create_response.text[:200]}")

    records = []
    for r in results:
        c_proxy = r["C_proxy"]
        if c_proxy == float("inf") or c_proxy != c_proxy:  # inf or NaN
            c_proxy = 9999.0
        records.append({
            "beta": float(r["beta"]),
            "P_measured": float(r["P_measured"]),
            "P_exact_migdal_witten": float(r["P_exact"]),
            "delta_P": float(r["delta_P"]),
            "curvature_density": float(r["curvature_density"]),
            "C_proxy": float(c_proxy),
            "Q_surrogate": float(r["Q_surrogate"]),
            "face_q0_mean": float(r["face_q0_mean"]),
            "face_q0_std": float(r["face_q0_std"]),
            "n_samples": int(r["n_samples"]),
            "phase": r["phase"],
        })

    print(f"\n=== INSERT {len(records)} observations ===")
    insert_response = requests.post(
        f"{gigi_url}/v1/bundles/{bundle_name}/insert",
        json={"records": records},
        headers=headers,
        timeout=30,
    )
    insert_body = insert_response.json() if insert_response.ok else insert_response.text
    print(f"  HTTP {insert_response.status_code}: inserted={insert_body.get('count')}, "
          f"curvature={insert_body.get('curvature'):.5f}, confidence={insert_body.get('confidence'):.5f}")

    print(f"\n=== Verify via single GQL query: SELECT * FROM {bundle_name} ===")
    query_response = requests.post(
        f"{gigi_url}/v1/gql",
        json={"query": f"SELECT * FROM {bundle_name};"},
        headers=headers,
        timeout=30,
    )

    rows = []
    if query_response.ok:
        body = query_response.json()
        rows = body.get("rows", [])
        print(f"  Returned {len(rows)} rows")
        print()
        print(f"  {'β':>5} {'P_meas':>9} {'P_exact':>9} {'Δ_P':>9} {'C_proxy':>9} {'Q':>8} {'phase':>12}")
        print(f"  " + "─" * 72)
        for r in sorted(rows, key=lambda x: x.get("beta", 0)):
            print(f"  {r['beta']:5.2f} {r['P_measured']:9.4f} {r['P_exact_migdal_witten']:9.4f} "
                  f"{r['delta_P']:+9.4f} {r['C_proxy']:9.2f} {r['Q_surrogate']:8.4f} {r['phase']:>12}")

    return {
        "bundle": bundle_name,
        "url": gigi_url,
        "n_rows": len(records),
        "insert_curvature": insert_body.get("curvature") if isinstance(insert_body, dict) else None,
        "insert_confidence": insert_body.get("confidence") if isinstance(insert_body, dict) else None,
        "query_returned": len(rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="Path to the β-walk JSON with measured + exact observables")
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE,
                    help="Target gigi bundle name")
    ap.add_argument("--gigi-url", default=os.environ.get("GIGI_URL", DEFAULT_GIGI_URL))
    args = ap.parse_args()

    api_key = os.environ.get("GIGI_API_KEY")
    push_observables_bundle(
        source_path=args.source,
        bundle_name=args.bundle,
        gigi_url=args.gigi_url,
        api_key=api_key,
    )


if __name__ == "__main__":
    sys.exit(main())
