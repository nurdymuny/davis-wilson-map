"""push_ym4_su3_dec2025_configs.py - process the Dec 2025 4D SU(3) Modal-generated
configs (pulled to _modal_pull/davis-wilson-data/configs/) and push observables
to the gigi bundle halcyon_ym4_glueball_demo.

The configs:
    100 SU(3) Wilson gauge configurations at beta=6.0, generated Dec 2025 on
    Modal (A100 GPU) by the davis-wilson-lattice HMC pipeline:
        0-49  : L=8  (2.4 MB each)
        50-99 : L=16 (37.5 MB each)
    All at beta=6.0, separation=10 sweeps, thermalization=100.

Per YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md M1 + M3 measurement on existing
gauge-invariant transport data — exactly what H5 specifies as the GIGI detector
input. This is the SAME analysis we run on freshly-generated SU(2) configs.

Output: two SU(3) ensemble row-sets in halcyon_ym4_glueball_demo with
ensemble_id starting su3_4d_L{L}_beta600_dec2025 — distinguishable from any
fresh SU(3) sweep ensembles via the "_dec2025" suffix.

Usage:
    PYTHONIOENCODING=utf-8 python -m inertia_damping.push_ym4_su3_dec2025_configs
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np
import requests
import torch

# Reuse the validated SU(3) observable extractors
from inertia_damping.su3_4d_heatbath_gpu import (
    plaquette_t_slice_density, wilson_loop_table,
    CDTYPE, RDTYPE,
)
from inertia_damping.su2_4d_glueball import (
    extract_correlator_and_mass,
    shuffled_null_control,
    average_wilson_table,
    jackknife_creutz,
)


DEFAULT_GIGI_URL = "http://localhost:3142"
DEFAULT_BUNDLE = "halcyon_ym4_glueball_demo"
DEFAULT_CONFIGS_DIR = "_modal_pull/davis-wilson-data/configs"
OUT_PATH = "inertia_damping/reports/ym4_su3_dec2025_configs_receipt.json"
WILSON_MAX = 3


def post(gigi_url: str, path: str, body: dict) -> tuple[int, object]:
    r = requests.post(f"{gigi_url}{path}", json=body, timeout=180)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)


def _safe_float(x) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return -9999.0
        return v
    except Exception:
        return -9999.0


def load_config_h5(path: str, device: torch.device) -> Tuple[torch.Tensor, dict]:
    """Return (U_torch, metadata_dict). U_torch shape (4, L, L, L, L, 3, 3) complex128."""
    with h5py.File(path, "r") as f:
        U_np = np.array(f["gauge_field"], dtype=np.complex128)
        meta = dict(f["metadata"].attrs)
    U = torch.from_numpy(U_np).to(device=device, dtype=CDTYPE)
    return U, meta


def process_ensemble(config_paths: List[str], L: int, beta: float,
                     device: torch.device, verbose: bool = True) -> dict:
    """For a list of config paths, compute per-config t-slice plaquette density
    and Wilson loop table; aggregate into M1 + M3 + null control."""
    n = len(config_paths)
    if verbose:
        print(f"\n{'='*72}")
        print(f"=== SU(3) DEC 2025 CONFIGS: L={L}, beta={beta}, n_configs={n}")
        print(f"{'='*72}")

    t_slices_per_cfg: List[np.ndarray] = []
    wilson_per_cfg: List[Dict[Tuple[int, int], float]] = []
    plaquette_per_cfg: List[float] = []
    t0 = time.perf_counter()
    for i, p in enumerate(config_paths):
        U, meta = load_config_h5(p, device)
        # t-slice spatial plaquette density (for M1)
        t_slices_per_cfg.append(plaquette_t_slice_density(U).cpu().numpy())
        # Wilson loops (for M3)
        wilson_per_cfg.append(wilson_loop_table(U, WILSON_MAX, WILSON_MAX))
        # Mean plaquette for sanity
        from inertia_damping.su3_4d_heatbath_gpu import avg_plaquette
        plaquette_per_cfg.append(avg_plaquette(U))
        if verbose and (i + 1) % max(1, n // 5) == 0:
            elapsed = time.perf_counter() - t0
            print(f"  processed {i+1}/{n} configs  <P>={plaquette_per_cfg[-1]:.4f}  ({elapsed:.0f}s)")
        del U
    t_wall = time.perf_counter() - t0

    p_bar = np.array(t_slices_per_cfg)  # (n_cfg, L)
    p_mean = float(np.mean(plaquette_per_cfg))
    if verbose:
        print(f"\n  Done. wall={t_wall:.0f}s. <P> mean over ensemble = {p_mean:.4f}")

    print(f"\n--- M1: plaquette correlator + effective mass ---")
    rep_c = extract_correlator_and_mass(p_bar, L=L, beta=beta)
    print(f"  C_PP(t):   {rep_c.C_PP_connected_t}")
    print(f"  m_eff(t):  {rep_c.m_eff_t}")
    print(f"  Plateau fit: m = {rep_c.plateau_fit_mass} +/- {rep_c.plateau_fit_error}  in window {rep_c.plateau_fit_window}")

    print(f"\n--- M3: SU(3) Wilson loops + Creutz string tension ---")
    W_avg = average_wilson_table(wilson_per_cfg)
    for k in sorted(W_avg):
        print(f"  W{k} = {W_avg[k]:.4f}")
    chi_22, chi_22_err = jackknife_creutz(wilson_per_cfg, 2, 2)
    chi_32, chi_32_err = jackknife_creutz(wilson_per_cfg, 3, 2)
    print(f"  chi(2,2) = {chi_22:.4f} +/- {chi_22_err:.4f}")
    print(f"  chi(3,2) = {chi_32:.4f} +/- {chi_32_err:.4f}")

    print(f"\n--- NULL CONTROL: shuffled plaquette data ---")
    p_shuf = shuffled_null_control(p_bar, seed=20260628)
    rep_null = extract_correlator_and_mass(p_shuf, L=L, beta=beta)
    print(f"  Null plateau: {rep_null.plateau_fit_mass}")

    return {
        "real": rep_c,
        "null": rep_null,
        "wilson_avg": W_avg,
        "chi_22": (chi_22, chi_22_err),
        "chi_32": (chi_32, chi_32_err),
        "L": L, "beta": beta, "n_configs": n,
        "P_mean": p_mean,
        "wall_seconds": t_wall,
        "source": "dec_2025_modal_davis_wilson_data",
    }


def build_records(results: dict, ens_id: str) -> list[dict]:
    L = results["L"]
    beta = results["beta"]
    n_cfg = results["n_configs"]
    real = results["real"]
    null = results["null"]
    W_avg = results["wilson_avg"]
    chi_22, chi_22_err = results["chi_22"]
    chi_32, chi_32_err = results["chi_32"]
    records = []
    for t in range(L):
        records.append({
            "ensemble_id": ens_id, "gauge_group": "SU(3)", "dimension": 4,
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
            "measurement_channel": "plaquette_correlator_M1_plus_wilson_loop_M3_dec2025_modal",
            "framing_doc_version": "YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1",
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default=DEFAULT_CONFIGS_DIR)
    ap.add_argument("--gigi-url", default=os.environ.get("GIGI_URL", DEFAULT_GIGI_URL))
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--cpu", action="store_true", help="Force CPU (default uses GPU if available)")
    args = ap.parse_args()

    try:
        h = requests.get(f"{args.gigi_url}/v1/health", timeout=5)
        print(f"GIGI health: {h.text[:120]}")
    except Exception as e:
        print(f"FATAL: gigi-stream not reachable at {args.gigi_url}: {e}")
        return 2

    # Force CPU to avoid contending with the running SU(2) GPU sweep
    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Processing on device: {device}")

    # Group configs by L based on metadata
    all_configs = sorted(glob.glob(os.path.join(args.configs_dir, "config_*.h5")))
    if not all_configs:
        print(f"FATAL: no configs found in {args.configs_dir}")
        return 2
    print(f"Found {len(all_configs)} config files in {args.configs_dir}")

    # Partition by L
    L_to_configs: Dict[int, List[str]] = {}
    beta_per_L: Dict[int, float] = {}
    for p in all_configs:
        with h5py.File(p, "r") as f:
            L = int(f["metadata"].attrs["lattice_size"])
            beta = float(f["metadata"].attrs["beta"])
        L_to_configs.setdefault(L, []).append(p)
        beta_per_L[L] = beta
    for L in L_to_configs:
        print(f"  L={L}: {len(L_to_configs[L])} configs at beta={beta_per_L[L]}")

    all_receipts = {}
    for L in sorted(L_to_configs):
        beta = beta_per_L[L]
        results = process_ensemble(L_to_configs[L], L=L, beta=beta, device=device)
        ens_id = f"su3_4d_L{L}_beta{int(beta*100):03d}_dec2025"
        records = build_records(results, ens_id)
        sc, body = post(args.gigi_url, f"/v1/bundles/{args.bundle}/insert", {"records": records})
        insert_body = body if isinstance(body, dict) else {}
        print(f"\n  INSERT {ens_id}: HTTP {sc}, count={insert_body.get('count')}, "
              f"curvature={insert_body.get('curvature')}, confidence={insert_body.get('confidence')}")
        all_receipts[ens_id] = {
            "L": L, "beta": beta, "n_configs": results["n_configs"],
            "P_mean": results["P_mean"],
            "plateau_fit_mass": results["real"].plateau_fit_mass,
            "plateau_fit_error": results["real"].plateau_fit_error,
            "sigma_creutz_22": results["chi_22"],
            "sigma_creutz_32": results["chi_32"],
            "wall_seconds": results["wall_seconds"],
            "null_plateau_fit_mass": results["null"].plateau_fit_mass,
        }

    print(f"\n{'='*72}\n=== SU(3) DEC 2025 HEADLINE: cross-L at beta=6.0 ===\n{'='*72}")
    q = (f"SELECT L, beta, plateau_fit_mass, plateau_fit_error, "
         f"sigma_creutz_22, sigma_creutz_22_error "
         f"FROM {args.bundle} WHERE gauge_group = 'SU(3)' AND t = 0;")
    print(f"GQL: {q}")
    sc, body = post(args.gigi_url, "/v1/gql", {"query": q})
    if isinstance(body, dict) and "rows" in body:
        rows = sorted(body["rows"], key=lambda r: r["L"])
        print(f"\n{len(rows)} SU(3) ensemble(s).\n")
        print(f"  {'L':>2} {'beta':>5} {'m_g (M1)':>11} {'± err':>9} {'sigma (M3)':>12} {'± err':>9}")
        for r in rows:
            m = f"{r['plateau_fit_mass']:.4f}" if r['plateau_fit_mass'] > -1000 else "  --"
            me = f"{r['plateau_fit_error']:.4f}" if r['plateau_fit_error'] > -1000 else "  --"
            s = f"{r['sigma_creutz_22']:.4f}" if r['sigma_creutz_22'] > -1000 else "  --"
            se = f"{r['sigma_creutz_22_error']:.4f}" if r['sigma_creutz_22_error'] > -1000 else "  --"
            print(f"  {r['L']:>2} {r['beta']:5.2f} {m:>11} {me:>9} {s:>12} {se:>9}")

    receipt = {
        "goal": "Process Bee's Dec 2025 Modal-generated 4D SU(3) configs, push observables to gigi",
        "source": "Modal volume davis-wilson-data/configs (50 L=8 + 50 L=16, all beta=6.0)",
        "framing_doc": "gigi/theory/YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md",
        "bundle": args.bundle,
        "ensembles_processed": all_receipts,
        "scope_notes": [
            "Same observables as fresh GPU SU(2) work (M1 + M3 + null control).",
            "Two L (8, 16) at same beta=6.0 -> two-point cross-L convergence diagnostic for SU(3).",
            "beta=6.0 is the canonical published reference point for SU(3) Wilson; <P> published ~ 0.5937.",
            "Per the Dec 2025 separation=10 sweeps between configs, autocorrelation is presumed handled.",
        ],
        "discipline": "Receipts before letters. Per Bee, 2026-06-28.",
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(receipt, f, indent=2, default=str)
    print(f"\nReceipt saved to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
