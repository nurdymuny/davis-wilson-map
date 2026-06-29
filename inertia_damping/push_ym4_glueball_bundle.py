"""push_ym4_glueball_bundle.py - generate 4D SU(2) configs across a beta sweep,
extract holonomy-continuum mass gap from plaquette correlators (M1) AND string
tension from Wilson-loop Creutz ratios (M3), with a null control per
Falsification Crit. 7. Push everything to gigi as halcyon_ym4_glueball_demo
with one ensemble per beta.

Framing: gigi/theory/YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md
  - H4: mass gap = finite-capacity propagation scale of holonomy substrate
  - H5: gigi should detect the gap from gauge-invariant transport data
  - M1: plaquette-plaquette correlator -> effective mass plateau
  - M3: Wilson loop area-law -> string tension (via Creutz ratio)
  - Null control: shuffled plaquette data must NOT give a plateau

Cross-channel agreement (the strongest H5 test): if M1 plaquette-gap and
M3 sqrt(string tension) track each other across the beta sweep, that's the
"same scale appears across independent gauge-invariant channels" prediction.

Bundle schema (one row per (ensemble_id, t); each ensemble has L rows):

    ensemble_id, gauge_group, dimension, L, beta, n_configurations, t,
    P_bar_t, C_PP_t, C_PP_error_t,
    m_eff_t, m_eff_error_t, m_eff_null_t,
    plateau_fit_mass, plateau_fit_error, plateau_fit_t_lo, plateau_fit_t_hi,
    sigma_creutz_22, sigma_creutz_22_error,
    sigma_creutz_32, sigma_creutz_32_error,
    W_11, W_12, W_22, W_23, W_33,
    measurement_channel, framing_doc_version

After the push, the headline cross-channel query:

    SELECT beta, plateau_fit_mass, plateau_fit_error,
           sigma_creutz_22, sigma_creutz_22_error
    FROM halcyon_ym4_glueball_demo
    WHERE t = 0;

returns one row per beta with the M1 mass gap next to the M3 string tension.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import requests

from inertia_damping.su2_4d_glueball import (
    extract_correlator_and_mass,
    shuffled_null_control,
    average_wilson_table,
    creutz_ratio_value,
    jackknife_creutz,
)


DEFAULT_GIGI_URL = "http://localhost:3142"
DEFAULT_BUNDLE = "halcyon_ym4_glueball_demo"
OUT_PATH = "inertia_damping/reports/ym4_glueball_in_gigi_receipt.json"
WILSON_MAX = 3   # compute W(R,T) for R, T in 1..3 -> 9 loops per config


def post(gigi_url: str, path: str, body: dict) -> tuple[int, object]:
    r = requests.post(f"{gigi_url}{path}", json=body, timeout=120)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)


def delete(gigi_url: str, path: str) -> tuple[int, object]:
    r = requests.delete(f"{gigi_url}{path}", timeout=30)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)


def _safe_float(x) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return -9999.0
        return v
    except Exception:
        return -9999.0


def run_one_ensemble(L: int, beta: float, n_therm: int, n_meas: int,
                     measure_every: int, seed: int, use_gpu: bool = False) -> dict:
    print(f"\n{'='*72}\n=== 4D SU(2) HEATBATH: L={L}, beta={beta}, gpu={use_gpu} ===\n{'='*72}")
    if use_gpu:
        from inertia_damping.su2_4d_heatbath_gpu import run_heatbath as _run
        rep_hb = _run(
            L=L, beta=beta,
            n_thermalize=n_therm, n_measure=n_meas,
            measure_every=measure_every, seed=seed,
            keep_t_slices=True, wilson_max=WILSON_MAX, verbose=True,
        )
        # t_slices come back as torch CPU tensors; convert to numpy
        t_slices_np = [t.numpy() if hasattr(t, "numpy") else t
                        for t in rep_hb.plaquette_per_config_t_slices]
    else:
        from inertia_damping.su2_4d_heatbath import run_heatbath as _run
        rep_hb = _run(
            L=L, beta=beta,
            n_thermalize=n_therm, n_measure=n_meas,
            measure_every=measure_every, seed=seed,
            keep_t_slices=True, wilson_max=WILSON_MAX, verbose=True,
        )
        t_slices_np = rep_hb.plaquette_per_config_t_slices

    p_bar = np.array(t_slices_np)  # (n_cfg, L)
    print(f"\nCollected {p_bar.shape[0]} configs.")

    print(f"\n--- M1: plaquette correlator + effective mass plateau ---")
    rep_c = extract_correlator_and_mass(p_bar, L=L, beta=beta)
    print(f"  Plateau fit: m = {rep_c.plateau_fit_mass} +/- {rep_c.plateau_fit_error}  in window {rep_c.plateau_fit_window}")

    print(f"\n--- M3: Wilson loops + Creutz ratio string tension ---")
    W_per_cfg = rep_hb.wilson_per_config
    W_avg = average_wilson_table(W_per_cfg)
    for k in sorted(W_avg):
        print(f"  W{k} = {W_avg[k]:.4f}")
    chi_22, chi_22_err = jackknife_creutz(W_per_cfg, 2, 2)
    chi_32, chi_32_err = jackknife_creutz(W_per_cfg, 3, 2)
    print(f"  chi(2,2) = {chi_22:.4f} +/- {chi_22_err:.4f}   (= sigma_lattice)")
    print(f"  chi(3,2) = {chi_32:.4f} +/- {chi_32_err:.4f}")

    print(f"\n--- NULL CONTROL: shuffled plaquette data ---")
    p_shuf = shuffled_null_control(p_bar, seed=seed + 1)
    rep_null = extract_correlator_and_mass(p_shuf, L=L, beta=beta)
    null_finite_count = int(np.isfinite(rep_null.m_eff_t).sum())
    print(f"  Null plateau fit: {rep_null.plateau_fit_mass} +/- {rep_null.plateau_fit_error}, "
          f"finite m_eff(t) = {null_finite_count}/{L}")

    return {
        "real": rep_c,
        "null": rep_null,
        "wilson_avg": W_avg,
        "wilson_per_cfg": W_per_cfg,
        "chi_22": (chi_22, chi_22_err),
        "chi_32": (chi_32, chi_32_err),
        "heatbath": {
            "L": L, "beta": beta, "n_thermalize": n_therm, "n_measure": n_meas,
            "measure_every": measure_every, "seed": seed,
            "wall_seconds_thermalize": rep_hb.wall_seconds_thermalize,
            "wall_seconds_measure": rep_hb.wall_seconds_measure,
            "final_plaquette": rep_hb.final_plaquette,
            "plaquette_history_mean": float(np.mean(rep_hb.plaquette_history)),
            "plaquette_history_std": float(np.std(rep_hb.plaquette_history)),
        },
    }


def build_records(results: dict) -> list[dict]:
    L = results["heatbath"]["L"]
    beta = results["heatbath"]["beta"]
    n_cfg = results["heatbath"]["n_measure"]
    ens_id = f"su2_4d_L{L}_beta{int(beta*100):03d}_n{n_cfg}"

    real = results["real"]
    null = results["null"]
    W_avg = results["wilson_avg"]
    chi_22, chi_22_err = results["chi_22"]
    chi_32, chi_32_err = results["chi_32"]

    records = []
    for t in range(L):
        records.append({
            "ensemble_id": ens_id, "gauge_group": "SU(2)", "dimension": 4,
            "L": L, "beta": beta, "n_configurations": n_cfg, "t": t,
            "P_bar_t": _safe_float(real.P_bar_mean_t[t]),
            "C_PP_t": _safe_float(real.C_PP_connected_t[t]),
            "C_PP_error_t": _safe_float(real.C_PP_error_t[t]),
            "m_eff_t": _safe_float(real.m_eff_t[t]),
            "m_eff_error_t": _safe_float(real.m_eff_error_t[t]),
            "m_eff_null_t": _safe_float(null.m_eff_t[t]),
            "plateau_fit_mass": _safe_float(real.plateau_fit_mass) if real.plateau_fit_mass is not None else -9999.0,
            "plateau_fit_error": _safe_float(real.plateau_fit_error) if real.plateau_fit_error is not None else -9999.0,
            "plateau_fit_t_lo": int(real.plateau_fit_window[0]) if real.plateau_fit_window else -1,
            "plateau_fit_t_hi": int(real.plateau_fit_window[1]) if real.plateau_fit_window else -1,
            "sigma_creutz_22": _safe_float(chi_22),
            "sigma_creutz_22_error": _safe_float(chi_22_err),
            "sigma_creutz_32": _safe_float(chi_32),
            "sigma_creutz_32_error": _safe_float(chi_32_err),
            "W_11": _safe_float(W_avg.get((1, 1), float("nan"))),
            "W_12": _safe_float(W_avg.get((1, 2), float("nan"))),
            "W_22": _safe_float(W_avg.get((2, 2), float("nan"))),
            "W_23": _safe_float(W_avg.get((2, 3), float("nan"))),
            "W_33": _safe_float(W_avg.get((3, 3), float("nan"))),
            "measurement_channel": "plaquette_correlator_M1_plus_wilson_loop_M3",
            "framing_doc_version": "YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1",
        })
    return records, ens_id


def create_bundle(gigi_url: str, bundle_name: str) -> dict:
    print(f"\n=== DROP existing {bundle_name} ===")
    sc, body = delete(gigi_url, f"/v1/bundles/{bundle_name}")
    print(f"  HTTP {sc}: {str(body)[:200]}")

    print(f"\n=== CREATE {bundle_name} (composite key, multi-ensemble) ===")
    sc, body = post(gigi_url, "/v1/bundles", {
        "name": bundle_name,
        "schema": {
            "fields": {
                "ensemble_id": "text", "gauge_group": "text", "dimension": "integer",
                "L": "integer", "beta": "float", "n_configurations": "integer", "t": "integer",
                "P_bar_t": "float", "C_PP_t": "float", "C_PP_error_t": "float",
                "m_eff_t": "float", "m_eff_error_t": "float", "m_eff_null_t": "float",
                "plateau_fit_mass": "float", "plateau_fit_error": "float",
                "plateau_fit_t_lo": "integer", "plateau_fit_t_hi": "integer",
                "sigma_creutz_22": "float", "sigma_creutz_22_error": "float",
                "sigma_creutz_32": "float", "sigma_creutz_32_error": "float",
                "W_11": "float", "W_12": "float", "W_22": "float", "W_23": "float", "W_33": "float",
                "measurement_channel": "text", "framing_doc_version": "text",
            },
            "keys": ["ensemble_id", "t"],
        },
    })
    print(f"  HTTP {sc}: {str(body)[:200]}")
    return {"http": sc, "body": body}


def insert_records(gigi_url: str, bundle_name: str, records: list[dict]) -> dict:
    sc, body = post(gigi_url, f"/v1/bundles/{bundle_name}/insert", {"records": records})
    insert_body = body if isinstance(body, dict) else {}
    print(f"  INSERT HTTP {sc}: count={insert_body.get('count')}, "
          f"curvature={insert_body.get('curvature')}, confidence={insert_body.get('confidence')}")
    return {"http": sc, "body": insert_body}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--betas", type=float, nargs="+", default=[2.0, 2.3, 2.5, 2.7])
    ap.add_argument("--n-therm", type=int, default=200)
    ap.add_argument("--n-meas", type=int, default=100)
    ap.add_argument("--measure-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--gigi-url", default=os.environ.get("GIGI_URL", DEFAULT_GIGI_URL))
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--use-gpu", action="store_true", help="Use GPU heatbath (torch CUDA)")
    ap.add_argument("--append", action="store_true",
                    help="Do NOT drop existing bundle; append ensembles instead")
    args = ap.parse_args()

    try:
        h = requests.get(f"{args.gigi_url}/v1/health", timeout=5)
        print(f"GIGI health: {h.text[:120]}")
    except Exception as e:
        print(f"FATAL: gigi-stream not reachable at {args.gigi_url}: {e}")
        return 2

    # Create the bundle ONCE (drop+create), unless --append (then keep prior ensembles)
    if not args.append:
        create_result = create_bundle(args.gigi_url, args.bundle)
        if create_result["http"] not in (200, 201):
            print(f"CREATE FAILED: {create_result}")
            return 1
    else:
        print(f"\n=== APPEND MODE: keeping existing {args.bundle} ensembles ===")

    all_ensembles = []
    t_overall_0 = time.perf_counter()

    for beta in args.betas:
        results = run_one_ensemble(
            L=args.L, beta=beta,
            n_therm=args.n_therm, n_meas=args.n_meas,
            measure_every=args.measure_every,
            seed=args.seed + int(beta * 100),  # per-beta seed offset for independence
            use_gpu=args.use_gpu,
        )
        records, ens_id = build_records(results)
        insert_result = insert_records(args.gigi_url, args.bundle, records)
        if insert_result["http"] not in (200, 201):
            print(f"INSERT FAILED for beta={beta}: {insert_result}")
            return 1
        all_ensembles.append({
            "ensemble_id": ens_id,
            "beta": beta,
            "real_plateau_fit_mass": results["real"].plateau_fit_mass,
            "real_plateau_fit_error": results["real"].plateau_fit_error,
            "null_plateau_fit_mass": results["null"].plateau_fit_mass,
            "chi_22": results["chi_22"],
            "chi_32": results["chi_32"],
            "P_bar_global": results["real"].P_bar_mean_global,
            "heatbath_wall_seconds": (
                results["heatbath"]["wall_seconds_thermalize"]
                + results["heatbath"]["wall_seconds_measure"]
            ),
        })

    t_overall = time.perf_counter() - t_overall_0
    print(f"\n{'='*72}\n=== HEADLINE CROSS-CHANNEL QUERY (M1 + M3 per beta) ===\n{'='*72}")
    query = (
        f"SELECT beta, plateau_fit_mass, plateau_fit_error, "
        f"sigma_creutz_22, sigma_creutz_22_error, m_eff_null_t "
        f"FROM {args.bundle} WHERE t = 0;"
    )
    print(f"GQL: {query}")
    sc, body = post(args.gigi_url, "/v1/gql", {"query": query})
    if isinstance(body, dict) and "rows" in body:
        rows = sorted(body["rows"], key=lambda r: r["beta"])
        print(f"\n{len(rows)} ensembles in bundle.\n")
        print(f"  {'beta':>5} {'m_g (M1)':>11} {'+/- err':>10} {'sigma (M3)':>12} {'+/- err':>10} {'null':>10}")
        print(f"  {'-'*5} {'-'*11} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
        for r in rows:
            null_str = f"{r['m_eff_null_t']:.4f}" if r['m_eff_null_t'] > -1000 else "NaN"
            m_str = f"{r['plateau_fit_mass']:.4f}" if r['plateau_fit_mass'] > -1000 else "  --"
            me_str = f"{r['plateau_fit_error']:.4f}" if r['plateau_fit_error'] > -1000 else "  --"
            s_str = f"{r['sigma_creutz_22']:.4f}" if r['sigma_creutz_22'] > -1000 else "  --"
            se_str = f"{r['sigma_creutz_22_error']:.4f}" if r['sigma_creutz_22_error'] > -1000 else "  --"
            print(f"  {r['beta']:5.2f} {m_str:>11} {me_str:>10} {s_str:>12} {se_str:>10} {null_str:>10}")

    receipt = {
        "goal": "4D SU(2) YM holonomy-continuum mass gap visible via simple GQL query, with M1 + M3 cross-channel agreement test across a beta-sweep",
        "framing_doc": "gigi/theory/YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md (M1 + M3 + Falsification Crit. 7)",
        "bundle": args.bundle,
        "lattice": {"L": args.L, "dimension": 4, "n_sites": args.L ** 4, "n_links": 4 * args.L ** 4},
        "betas_swept": args.betas,
        "n_ensembles": len(args.betas),
        "headline_query": query,
        "overall_wall_seconds": t_overall,
        "ensembles": all_ensembles,
        "scope_caveats": [
            f"L={args.L} is small; this is a SUBSTRATE DEMONSTRATION of the H5 cross-channel test, not a continuum mass-gap number. Continuum extrapolation needs multiple L + scale-setting via Wilson flow or r_0.",
            f"n_configurations = {args.n_meas} per ensemble is small for jackknife error bars (production would use 500-2000).",
            "M2 (glueball variational with APE smearing) is not yet pushed. Cross-channel agreement is currently M1 vs M3.",
            "Creutz chi(2,2) is the lattice string tension; sqrt(sigma) sets a mass scale that should agree QUALITATIVELY with m_g(M1) up to a numerical factor. Exact equality is not expected.",
        ],
        "discipline": "Receipts before letters. Per Bee, 2026-06-28.",
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(receipt, f, indent=2, default=str)
    print(f"\nReceipt saved to {OUT_PATH}")
    print(f"Total wall-clock for the beta sweep: {t_overall:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
