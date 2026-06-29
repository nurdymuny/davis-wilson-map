"""spectral_gauge_empirical_receipt.py — fire SPECTRAL vs SPECTRAL_GAUGE on the two
buckyball SU(2) bundles and capture the comparison as the empirical receipt for
the verb GIGI shipped at e37ae9e (live on prod 2026-06-28).

Hypothesis under test (from cfeb5c5 + bridge-ask receipts):
    SPECTRAL on these bundles is fiber-blind (line-graph Laplacian; reads only
    vertex_a/vertex_b indexed bitmaps). Empirically (e4800b4): 0.024660 at β=2.5
    vs 0.024885 at β=1.0 — agreement to 4 sig figs across 2.25× change in ⟨P⟩.

    SPECTRAL_GAUGE should be fiber-AWARE (weights edges by Re Tr(U_e)/N). The
    prediction: the gap should differ measurably between β=2.5 (deconfined,
    plaquettes close to identity, edge weights ~ +0.5) and β=1.0 (confined,
    plaquettes near random, edge weights ~ +0.23).

Schema fix (vs the older push_buckyball_to_gigi.py):
    SPECTRAL_GAUGE reads `store.schema.base_fields`. The REST POST /v1/bundles
    handler (src/bin/gigi_stream.rs:1922-1944) puts fields named in `keys` into
    base_fields and everything else into fiber_fields. The older script had only
    `edge_id` in keys, which made vertex_a/vertex_b fiber fields and caused
    SPECTRAL_GAUGE to error with "missing edge endpoint fields vertex_a/vertex_b".
    Fix: composite keys [edge_id, vertex_a, vertex_b] so all three are base.

This is the "receipts-before-letters" discipline: do the technical work, capture
the numbers, THEN write the reply to GIGI.

Usage:
    PYTHONIOENCODING=utf-8 python -m inertia_damping.spectral_gauge_empirical_receipt
    # against running local gigi-stream on http://localhost:3142

Requires:
    - gigi-stream running locally on http://localhost:3142
      (with --features halcyon, built from origin/main >= e37ae9e)
"""
from __future__ import annotations

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
OUT_PATH = "inertia_damping/reports/spectral_gauge_empirical_receipt.json"


def post(gigi_url: str, path: str, body: dict) -> tuple[int, object]:
    r = requests.post(f"{gigi_url}{path}", json=body, timeout=120)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)


def delete(gigi_url: str, path: str) -> tuple[int, object]:
    r = requests.delete(f"{gigi_url}{path}", timeout=30)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)


def bundle_name(beta: float) -> str:
    return f"halcyon_ym_buckyball_beta_{str(beta).replace('.', '_')}_links"


def push_with_composite_keys(beta: float, *, gigi_url: str, seed: int = 20260616,
                             n_sweeps: int = 1000, n_measure: int = 100,
                             measure_every: int = 5) -> dict:
    name = bundle_name(beta)

    graph = buckyball_graph.build_truncated_icosahedron()
    V, E, F = graph.n_vertices, graph.n_edges, graph.n_faces
    edges = np.asarray(graph.edges)
    print(f"Buckyball: V={V}, E={E}, F={F}, chi={V-E+F}")
    print(f"Thermalizing beta={beta}, {n_sweeps} sweeps, seed {seed}...")

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

    print(f"\n=== DROP existing {name} (if present) ===")
    sc, body = delete(gigi_url, f"/v1/bundles/{name}")
    print(f"  HTTP {sc}: {str(body)[:200]}")

    print(f"\n=== CREATE {name} with composite keys [edge_id, vertex_a, vertex_b] ===")
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
            "keys": ["edge_id", "vertex_a", "vertex_b"],
            "indexed": [],
        },
    })
    print(f"  HTTP {sc}: {str(body)[:200]}")
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
    sc, body = post(gigi_url, f"/v1/bundles/{name}/insert", {"records": records})
    print(f"  HTTP {sc}: count={body.get('count') if isinstance(body, dict) else '?'}, "
          f"curvature={body.get('curvature') if isinstance(body, dict) else '?'}, "
          f"confidence={body.get('confidence') if isinstance(body, dict) else '?'}")
    insert_body = body if isinstance(body, dict) else {}

    return {
        "status": "ok",
        "bundle": name,
        "beta": beta,
        "seed": seed,
        "graph": {"V": V, "E": E, "F": F, "chi": V - E + F},
        "thermalization": {
            "n_sweeps": n_sweeps, "n_measure": n_measure, "measure_every": measure_every,
            "P_measured": p_meas, "P_exact_migdal_witten": p_exact,
            "wall_seconds": therm_wall,
        },
        "insert": {
            "count": insert_body.get("count"),
            "curvature": insert_body.get("curvature"),
            "confidence": insert_body.get("confidence"),
        },
    }


def fire_spectral(gigi_url: str, bundle: str) -> dict:
    sc, body = post(gigi_url, "/v1/gql", {"query": f"SPECTRAL {bundle};"})
    # SPECTRAL returns either {"value": X} (single-scalar form) or {"rows":[{...}],...}
    val = None
    if isinstance(body, dict):
        if "value" in body:
            val = body["value"]
        elif "rows" in body and body["rows"]:
            row0 = body["rows"][0]
            val = row0.get("value") or row0.get("gap") or row0.get("lambda_1")
    return {"http_status": sc, "value": val, "raw": body}


def fire_spectral_gauge(gigi_url: str, bundle: str, group: str = "SU(2)") -> dict:
    query = f"SPECTRAL_GAUGE {bundle} ON FIBER (q0, q1, q2, q3) GROUP {group};"
    sc, body = post(gigi_url, "/v1/gql", {"query": query})
    gap, n, grp, err = None, None, None, None
    if isinstance(body, dict):
        if "error" in body:
            err = body["error"]
        elif "rows" in body and body["rows"]:
            row0 = body["rows"][0]
            gap = row0.get("gap")
            n = row0.get("n_records_used")
            grp = row0.get("group_used")
        else:
            err = f"unexpected response shape: {list(body.keys())}"
    else:
        err = str(body)
    return {"query": query, "http_status": sc, "gap": gap,
            "n_records_used": n, "group_used": grp, "error": err, "raw": body}


def main():
    gigi_url = os.environ.get("GIGI_URL", DEFAULT_GIGI_URL)
    try:
        h = requests.get(f"{gigi_url}/v1/health", timeout=5)
        print(f"GIGI health: HTTP {h.status_code} -- {h.text[:120]}")
    except Exception as e:
        print(f"FATAL: gigi-stream not reachable at {gigi_url}: {e}")
        return 2

    BETAS = [2.5, 1.0]
    SEED = 20260616
    N_SWEEPS = 1000

    pushes = []
    for beta in BETAS:
        print(f"\n{'='*72}\n=== PUSH beta={beta} ===\n{'='*72}")
        result = push_with_composite_keys(beta, gigi_url=gigi_url, seed=SEED, n_sweeps=N_SWEEPS)
        pushes.append(result)

    print(f"\n{'='*72}\n=== EMPIRICAL FIRE: SPECTRAL vs SPECTRAL_GAUGE ===\n{'='*72}")
    comparisons = []
    for p in pushes:
        bundle = p["bundle"]
        beta = p["beta"]
        print(f"\n--- {bundle} (beta={beta}) ---")
        spec = fire_spectral(gigi_url, bundle)
        print(f"  SPECTRAL          value={spec['value']}")
        sg = fire_spectral_gauge(gigi_url, bundle, group="SU(2)")
        print(f"  SPECTRAL_GAUGE    gap={sg['gap']}, n={sg['n_records_used']}, group={sg['group_used']}")
        if sg["error"]:
            print(f"  SPECTRAL_GAUGE error: {sg['error']}")
        comparisons.append({
            "bundle": bundle,
            "beta": beta,
            "P_measured": p["thermalization"]["P_measured"],
            "P_exact_migdal_witten": p["thermalization"]["P_exact_migdal_witten"],
            "insert_curvature": p["insert"]["curvature"],
            "insert_confidence": p["insert"]["confidence"],
            "spectral_fiber_blind": spec,
            "spectral_gauge_fiber_aware": sg,
        })

    summary = None
    if len(comparisons) == 2:
        c25, c10 = comparisons[0], comparisons[1]
        sg25 = c25["spectral_gauge_fiber_aware"]["gap"]
        sg10 = c10["spectral_gauge_fiber_aware"]["gap"]
        sp25 = c25["spectral_fiber_blind"]["value"]
        sp10 = c10["spectral_fiber_blind"]["value"]
        summary = {
            "P_ratio_beta_2_5_to_beta_1_0": (
                c25["P_measured"] / c10["P_measured"] if c10["P_measured"] else None
            ),
            "SPECTRAL_fiber_blind": {
                "beta_2_5": sp25, "beta_1_0": sp10,
                "abs_difference": abs(sp25 - sp10) if (sp25 is not None and sp10 is not None) else None,
                "reads_as": "FIBER-BLIND" if (sp25 is not None and sp10 is not None and abs(sp25 - sp10) < 1e-3) else "FIBER-AWARE",
            },
            "SPECTRAL_GAUGE_fiber_aware": {
                "beta_2_5": sg25, "beta_1_0": sg10,
                "abs_difference": abs(sg25 - sg10) if (sg25 is not None and sg10 is not None) else None,
                "ratio_2_5_to_1_0": (sg25 / sg10) if (sg25 is not None and sg10 not in (None, 0)) else None,
                "reads_as": "FIBER-BLIND" if (sg25 is not None and sg10 is not None and abs(sg25 - sg10) < 1e-3) else
                            "FIBER-AWARE" if (sg25 is not None and sg10 is not None) else
                            "ERROR",
            },
            "verb_does_what_we_hoped": (
                sg25 is not None and sg10 is not None and abs(sg25 - sg10) > 1e-3
            ),
        }

    receipt = {
        "purpose": "Empirical receipt: SPECTRAL (fiber-blind) vs SPECTRAL_GAUGE (fiber-aware) on two buckyball SU(2) bundles with composite-keys schema.",
        "ship_commit_under_test": "e37ae9e (gigi origin/main HEAD 35a727d, built locally and live on http://localhost:3142)",
        "gigi_url": gigi_url,
        "schema_used": {
            "fields": ["edge_id", "vertex_a", "vertex_b", "q0", "q1", "q2", "q3", "config_id"],
            "keys (-> base_fields)": ["edge_id", "vertex_a", "vertex_b"],
            "fiber_fields_inferred": ["q0", "q1", "q2", "q3", "config_id"],
            "note": "composite keys put vertex_a/vertex_b in base_fields as SPECTRAL_GAUGE expects (src/spectral.rs:1146-1150)",
        },
        "thermalization_protocol": {"n_sweeps": N_SWEEPS, "seed": SEED},
        "pushes": pushes,
        "comparisons": comparisons,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt saved to {OUT_PATH}")

    if summary:
        print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
