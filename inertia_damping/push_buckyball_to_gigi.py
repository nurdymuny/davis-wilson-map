"""push_buckyball_to_gigi.py — thermalize buckyball SU(2) and push as gigi bundle.

Workflow:
  1. Thermalize buckyball at given β with the local Python toolkit.
  2. POST /v1/bundles to create a 90-record link bundle with vertex_a/vertex_b INDEXED
     (drives gigi's field-index graph adjacency for SPECTRAL/BETTI).
  3. POST /v1/bundles/{name}/insert with 90 records (one per edge), capturing the
     substrate's per-insert curvature + confidence (these ARE fiber-aware).
  4. Fire SPECTRAL + BETTI via GQL.

Known caveat: SPECTRAL on this bundle is FIBER-BLIND. It returns λ₁ of the
field-index graph from indexed bitmaps only (vertex_a / vertex_b), not the SU(2)
quaternion fields. Empirically: SPECTRAL ≈ 0.0247 at β=2.5 vs ≈ 0.0249 at β=1.0
(identical within numerical tolerance) even though ⟨P⟩ differs by 2× across the
deconfinement transition. BETTI is pure topology (56.0 either way). Both are
useful as topology sanity-checks; neither is the YM mass gap. The mass-gap
reading requires the EXPOSE_GAUGE_AS_BUNDLE bridge ask (see
HALCYON_TO_GIGI_2026_06_22_bridge_ask.md §2) or a new gauge-aware Laplacian verb.

Bonus discovery: the substrate's per-insert `curvature` and `confidence` fields
ARE fiber-aware. Different β values give different per-insert curvature (β=2.5
→ 0.0599, β=1.0 → 0.0586). The substrate computes a geometric drift signal on
each insert that reads the record values, not just indexed bitmaps. This is one
fiber-aware substrate observable we get for free without the bridge.

Usage:
    python -m inertia_damping.push_buckyball_to_gigi --beta 2.5
    python -m inertia_damping.push_buckyball_to_gigi --beta 1.0 --seed 20260617

Requires:
    - gigi-stream running on http://localhost:3142 (or set GIGI_URL)
    - GIGI_API_KEY env var if prod (omit for local dev)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import requests

from inertia_damping import (
    buckyball_graph,
    buckyball_heatbath,
    buckyball_yangmills_exact,
)


DEFAULT_GIGI_URL = "http://localhost:3142"
DEFAULT_SEED = 20260616
DEFAULT_N_SWEEPS = 1000
DEFAULT_N_MEASURE = 100
DEFAULT_MEASURE_EVERY = 5


def bundle_name(beta: float, suffix: str = "links") -> str:
    """Stable bundle name from β. Uses underscore-separated rep, e.g. β=2.5 → 'beta_2_5'."""
    beta_tag = str(beta).replace(".", "_").replace("-", "neg")
    return f"halcyon_ym_buckyball_beta_{beta_tag}_{suffix}"


def post(base_url: str, path: str, body: dict, api_key: str | None = None) -> tuple[int, dict | str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    r = requests.post(f"{base_url}{path}", json=body, headers=headers, timeout=120)
    return r.status_code, (r.json() if r.ok else r.text)


def push(beta: float, *, gigi_url: str, api_key: str | None, seed: int,
         n_sweeps: int, n_measure: int, measure_every: int) -> dict:
    graph = buckyball_graph.build_truncated_icosahedron()
    V, E, F = graph.n_vertices, graph.n_edges, graph.n_faces
    edges = np.asarray(graph.edges)

    print(f"Buckyball: V={V}, E={E}, F={F}, chi={V-E+F}")
    print(f"Thermalizing β={beta}, {n_sweeps} sweeps, seed {seed}...")

    t0 = time.perf_counter()
    r = buckyball_heatbath.thermalize(
        graph, beta=beta, n_sweeps=n_sweeps, n_measure=n_measure,
        n_measure_every=measure_every, seed=seed,
    )
    U = r["U_final"].cpu().numpy()
    p_meas = float(r["P_mean"])
    p_exact = buckyball_yangmills_exact.exact_mean_plaquette_su2_buckyball(beta)
    therm_wall = time.perf_counter() - t0
    print(f"  Done in {therm_wall:.0f}s. <P>={p_meas:.4f} (vs exact {p_exact:.4f})")

    name = bundle_name(beta)

    print(f"\n=== CREATE {name} ===")
    sc, body = post(gigi_url, "/v1/bundles", {
        "name": name,
        "schema": {
            "fields": {
                "edge_id": "integer",
                "vertex_a": "integer",
                "vertex_b": "integer",
                "q0": "float", "q1": "float", "q2": "float", "q3": "float",
                "config_id": "integer",
            },
            "keys": ["edge_id"],
            "indexed": ["vertex_a", "vertex_b"],
        },
    }, api_key=api_key)
    print(f"  HTTP {sc}: {body}")
    if sc not in (200, 201):
        return {"status": "failed", "stage": "create", "body": body}

    records = []
    for i in range(E):
        records.append({
            "edge_id": i,
            "vertex_a": int(edges[i, 0]),
            "vertex_b": int(edges[i, 1]),
            "q0": float(U[i, 0]), "q1": float(U[i, 1]),
            "q2": float(U[i, 2]), "q3": float(U[i, 3]),
            "config_id": 0,
        })

    print(f"\n=== INSERT {E} link records ===")
    sc, body = post(gigi_url, f"/v1/bundles/{name}/insert", {"records": records}, api_key=api_key)
    insert_curvature = body.get("curvature") if isinstance(body, dict) else None
    insert_confidence = body.get("confidence") if isinstance(body, dict) else None
    print(f"  HTTP {sc}: count={body.get('count')}, curvature={insert_curvature}, confidence={insert_confidence}")

    print(f"\n=== SPECTRAL + BETTI via GQL ===")
    sc, spec = post(gigi_url, "/v1/gql", {"query": f"SPECTRAL {name};"}, api_key=api_key)
    sc, betti = post(gigi_url, "/v1/gql", {"query": f"BETTI {name};"}, api_key=api_key)
    spec_val = spec.get("value") if isinstance(spec, dict) else None
    betti_val = betti.get("value") if isinstance(betti, dict) else None
    print(f"  SPECTRAL = {spec_val}   (fiber-blind line-graph λ₁)")
    print(f"  BETTI    = {betti_val}  (pure topology, β₀+β₁ of field-index graph)")

    return {
        "status": "ok",
        "bundle": name,
        "beta": beta,
        "seed": seed,
        "thermalization": {
            "n_sweeps": n_sweeps, "n_measure": n_measure, "measure_every": measure_every,
            "P_measured": p_meas, "P_exact_migdal_witten": p_exact,
            "wall_seconds": therm_wall,
        },
        "graph": {"V": V, "E": E, "F": F, "chi": V - E + F},
        "insert": {
            "count": body.get("count") if isinstance(body, dict) else None,
            "curvature": insert_curvature,
            "confidence": insert_confidence,
        },
        "spectral_fiber_blind": spec_val,
        "betti_topology": betti_val,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--gigi-url", default=os.environ.get("GIGI_URL", DEFAULT_GIGI_URL))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-sweeps", type=int, default=DEFAULT_N_SWEEPS)
    ap.add_argument("--n-measure", type=int, default=DEFAULT_N_MEASURE)
    ap.add_argument("--measure-every", type=int, default=DEFAULT_MEASURE_EVERY)
    ap.add_argument("--out", default=None,
                    help="optional JSON output path; default skips writing")
    args = ap.parse_args()

    api_key = os.environ.get("GIGI_API_KEY")

    result = push(
        args.beta,
        gigi_url=args.gigi_url, api_key=api_key, seed=args.seed,
        n_sweeps=args.n_sweeps, n_measure=args.n_measure, measure_every=args.measure_every,
    )

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved result to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
