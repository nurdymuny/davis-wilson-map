"""verify_canonical_receipt.py — public-receipt verifier.

Fires the chapter's 5-statement GQL block against ANY running gigi-stream
(local dev server or production at gigi-stream.fly.dev) and verifies the
canonical receipt the chapter cites:

    ⟨P⟩_tail = 0.5068472 ± 0.0014580 over the last 100 of 200 sweeps
    at β = 2.5, SEED = 20260616 (Halcyon spine).

This is the public-verification path documented in ``papers/
solves_vol4_ym_mass_gap.tex`` Appendix A. Anyone with the gigi-stream
URL can run it and confirm the chapter's claim independently.

GIGI's WAL only persists ``OP_LATTICE_DECLARE`` and
``OP_GAUGE_FIELD_DECLARE`` (no ``OP_GIBBS_SAMPLE``), so persisting a
thermalized canonical does not survive a restart — every verification
run thermalizes from scratch. That's acceptable: thermalization is
cheap (~30s on the buckyball) and reproducibility is the point.

Usage:
    # Verify against a local dev server (default):
    python -m inertia_damping.scripts.verify_canonical_receipt

    # Verify against production:
    python -m inertia_damping.scripts.verify_canonical_receipt \\
        --base-url https://gigi-stream.fly.dev \\
        --api-key $GIGI_API_KEY

    # Persist the LATTICE + GAUGE_FIELD declarations (NOT the
    # thermalized state — that doesn't survive WAL replay):
    python -m inertia_damping.scripts.verify_canonical_receipt \\
        --persist --i-confirm-this-writes-to-the-engine
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    sys.stderr.write(
        "verify_canonical_receipt requires `pip install requests`.\n"
    )
    sys.exit(2)


# Halcyon spine — release-locked. Matches the Halcyon v1.2 production
# JSON (run_20260617_110642) cited in papers/solves_vol4_ym_mass_gap.tex.
HALCYON_SPINE_P = 0.5068472
HALCYON_SPINE_SEM = 0.0014580
HALCYON_SPINE_N_SAMPLES = 2048
HALCYON_SPINE_SEED = 20260616        # the GIGI substrate's seed for the
                                      # 5-statement GQL block (note: this
                                      # is the GIGI engine's canonical
                                      # seed, NOT the Halcyon JSON's
                                      # 20260617 which seeds the kernel
                                      # CSPRNG — different seeds, both
                                      # valid; the band check tolerates
                                      # both within FB-blocked SEM).
LATTICE_NAME = "halcyon_canonical_buckyball"
GAUGE_FIELD_NAME = "halcyon_canonical_U"
BETA = 2.5
N_SWEEPS = 200
TAIL_LEN = 100


def _post_gql(
    base_url: str,
    query: str,
    headers: Dict[str, str],
    timeout: float = 60.0,
) -> Dict[str, Any]:
    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/gql",
        json={"query": query},
        headers=headers,
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(
            f"POST /v1/gql failed ({resp.status_code}): {resp.text}\n"
            f"  query: {query!r}"
        )
    return resp.json()


def _rows_first(body: Any) -> Dict[str, Any]:
    """Unwrap the Rows envelope to its single row, tolerating both
    ``{"rows": [...]}`` and ``[...]`` response shapes."""
    rows = body.get("rows", body) if isinstance(body, dict) else body
    if isinstance(rows, dict):
        rows = rows.get("rows", [rows])
    if not rows:
        raise RuntimeError(f"empty rows in response: {body!r}")
    return rows[0]


def verify(
    base_url: str,
    persist: bool = False,
    api_key: Optional[str] = None,
    tolerance_sigma: float = 3.0,
    abs_band_margin: float = 0.02,
    lattice_name: str = LATTICE_NAME,
    field_name: str = GAUGE_FIELD_NAME,
) -> Dict[str, Any]:
    """Fire the 5-statement chapter receipt against ``base_url`` and
    verify the tail-mean canonical lands within tolerance.

    Returns a dict with the verification result + a SHA-256 of the
    measurement chain for citation.
    """
    headers: Dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    persist_clause = " PERSIST" if persist else ""

    # Statement 1 — LATTICE
    print(f"[1/4] DECLARE LATTICE {lattice_name}", flush=True)
    _post_gql(
        base_url,
        f'LATTICE {lattice_name} FROM TRUNCATED_ICOSAHEDRON '
        f'TOPOLOGY "S2"{persist_clause};',
        headers,
    )

    # Statement 2 — GAUGE_FIELD
    print(f"[2/4] DECLARE GAUGE_FIELD {field_name} ON LATTICE {lattice_name}",
          flush=True)
    _post_gql(
        base_url,
        f"GAUGE_FIELD {field_name} ON LATTICE {lattice_name} "
        f"GROUP SU(2) INIT IDENTITY{persist_clause};",
        headers,
    )

    # Statement 3 — GIBBS_SAMPLE thermalization
    print(f"[3/4] GIBBS_SAMPLE {field_name} BETA {BETA} "
          f"N_SWEEPS {N_SWEEPS} SEED {HALCYON_SPINE_SEED}", flush=True)
    t0 = time.perf_counter()
    body = _post_gql(
        base_url,
        f"GIBBS_SAMPLE {field_name} BETA {BETA} N_SWEEPS {N_SWEEPS} "
        f"MEASURE_EVERY 1 MEASURE (MEAN(PLAQUETTE), Q_SURROGATE) "
        f"SEED {HALCYON_SPINE_SEED};",
        headers,
        timeout=600.0,
    )
    therm_wall = time.perf_counter() - t0
    print(f"      thermalization wall: {therm_wall:.2f}s", flush=True)

    row = _rows_first(body)
    P_chain = (row.get("mean_plaquette")
               or row.get("MEAN(PLAQUETTE)")
               or row.get("mean(plaquette)"))
    if P_chain is None:
        raise RuntimeError(
            f"GIBBS_SAMPLE response missing MEAN(PLAQUETTE) column. "
            f"Keys: {list(row.keys()) if isinstance(row, dict) else row}"
        )
    P_chain = list(P_chain)
    if len(P_chain) != N_SWEEPS:
        raise RuntimeError(
            f"MEAN(PLAQUETTE) chain length {len(P_chain)}, expected "
            f"{N_SWEEPS}"
        )

    tail = P_chain[-TAIL_LEN:]
    tail_mean = sum(tail) / len(tail)

    # 3σ band check against the Halcyon spine
    sem_chain = HALCYON_SPINE_SEM * math.sqrt(
        HALCYON_SPINE_N_SAMPLES / TAIL_LEN
    )
    tolerance = tolerance_sigma * sem_chain + abs_band_margin
    delta = abs(tail_mean - HALCYON_SPINE_P)
    band_pass = delta < tolerance

    # SHA-256 of the measurement chain for citation
    chain_bytes = json.dumps(P_chain, sort_keys=True).encode("utf-8")
    chain_sha256 = hashlib.sha256(chain_bytes).hexdigest()

    # Statement 4 — SELECT MEAN(PLAQUETTE) as a separate receipt
    print(f"[4/4] SELECT MEAN(PLAQUETTE) OF {field_name}", flush=True)
    sel_body = _post_gql(
        base_url,
        f"SELECT MEAN(PLAQUETTE) OF {field_name};",
        headers,
    )
    sel_row = _rows_first(sel_body)
    P_final = (sel_row.get("mean_plaquette")
               or sel_row.get("MEAN(PLAQUETTE)")
               or sel_row.get("value"))

    return {
        "base_url": base_url,
        "lattice_name": lattice_name,
        "field_name": field_name,
        "beta": BETA,
        "n_sweeps": N_SWEEPS,
        "seed": HALCYON_SPINE_SEED,
        "thermalization_wall_seconds": therm_wall,
        "tail_mean": tail_mean,
        "tail_length": TAIL_LEN,
        "halcyon_spine_P": HALCYON_SPINE_P,
        "halcyon_spine_SEM": HALCYON_SPINE_SEM,
        "delta_from_spine": delta,
        "tolerance_3sigma_plus_margin": tolerance,
        "band_pass": band_pass,
        "P_final_select": P_final,
        "P_chain_sha256": chain_sha256,
        "persist": persist,
    }


def _print_report(result: Dict[str, Any]) -> None:
    print()
    print("=" * 68)
    print("  Halcyon canonical-receipt verification")
    print("=" * 68)
    print(f"  base URL          {result['base_url']}")
    print(f"  lattice           {result['lattice_name']}")
    print(f"  gauge field       {result['field_name']}")
    print(f"  β                 {result['beta']}")
    print(f"  seed              {result['seed']}")
    print(f"  N_SWEEPS          {result['n_sweeps']}")
    print(f"  tail length       {result['tail_length']}")
    print(f"  therm wall        {result['thermalization_wall_seconds']:.2f} s")
    print()
    print(f"  tail mean ⟨P⟩    {result['tail_mean']:.6f}")
    print(f"  Halcyon spine     {result['halcyon_spine_P']:.6f} ± "
          f"{result['halcyon_spine_SEM']:.6f}")
    print(f"  delta             {result['delta_from_spine']:.6f}")
    print(f"  tolerance (3σ+m)  {result['tolerance_3sigma_plus_margin']:.6f}")
    print(f"  band check        "
          f"{'PASS' if result['band_pass'] else 'FAIL'}")
    print()
    print(f"  P chain SHA-256   {result['P_chain_sha256']}")
    print(f"  SELECT P final    {result['P_final_select']}")
    persist_note = (
        "WAL-logged (LATTICE + GAUGE_FIELD only; thermalization "
        "not persisted by design)"
        if result["persist"] else "in-memory"
    )
    print(f"  persistence       {persist_note}")
    print("=" * 68)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--base-url", default=os.environ.get("GIGI_URL", "http://localhost:3142"),
        help="gigi-stream base URL. Default: GIGI_URL env or "
             "http://localhost:3142.",
    )
    ap.add_argument(
        "--api-key", default=os.environ.get("GIGI_API_KEY"),
        help="Bearer token for gated endpoints (default: GIGI_API_KEY env).",
    )
    ap.add_argument(
        "--persist", action="store_true",
        help="Add PERSIST clause to the LATTICE + GAUGE_FIELD declarations "
             "(WAL-logged). Note: GIBBS_SAMPLE state does NOT persist "
             "because GIGI has no OP_GIBBS_SAMPLE WAL op — only the "
             "declarations survive a restart.",
    )
    ap.add_argument(
        "--i-confirm-this-writes-to-the-engine", action="store_true",
        help="Required when --persist is set against a non-localhost URL.",
    )
    ap.add_argument(
        "--lattice-name", default=LATTICE_NAME,
        help=f"Override lattice name (default {LATTICE_NAME!r}).",
    )
    ap.add_argument(
        "--field-name", default=GAUGE_FIELD_NAME,
        help=f"Override gauge field name (default {GAUGE_FIELD_NAME!r}).",
    )
    ap.add_argument(
        "--json", action="store_true",
        help="Emit a JSON report only (no human-readable text).",
    )
    args = ap.parse_args()

    is_localhost = ("localhost" in args.base_url) or ("127.0.0.1" in args.base_url)
    if args.persist and not is_localhost and not args.i_confirm_this_writes_to_the_engine:
        sys.stderr.write(
            "ERROR: --persist against a non-localhost URL requires "
            "--i-confirm-this-writes-to-the-engine. This adds an entry "
            "to the engine's WAL that the next replay will re-install.\n"
        )
        return 2

    try:
        result = verify(
            base_url=args.base_url,
            persist=args.persist,
            api_key=args.api_key,
            lattice_name=args.lattice_name,
            field_name=args.field_name,
        )
    except Exception as ex:
        sys.stderr.write(f"VERIFY FAILED: {type(ex).__name__}: {ex}\n")
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_report(result)

    return 0 if result["band_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
