"""Phase F live tests -- ``--use-gigi`` orchestrator routing.

Fires the orchestrator (``run_validation_report.py``) end-to-end with
``--use-gigi`` against the live gigi-stream engine. Validates the
architectural promise from HALCYON_USE_GIGI_FLAG_SPEC.md:

  * F0 smoke -- orchestrator returns the v1.2 schema (10 categories,
    same JSON shape as the Python-kernel path).
  * F1 bit-identity -- two consecutive ``--use-gigi`` runs at the same
    seed produce identical category verdicts AND identical
    ``snapshot_sha256`` from the SNAPSHOT step. Requires
    ``--gigi-snapshot`` + ``--i-confirm-this-writes-to-the-engine``
    (per spec § 3 + spec § open Q on F1).
  * F2 band-agreement -- ``--use-gigi`` vs Python-kernel at same seed,
    canonical ⟨P⟩ within 3σ of the documented blocked SEM across the
    6 β-scan points.

Test sizing: defaults to a tiny problem (--steps 100 --canonical-sweeps
200). Re-run at full sizes with HALCYON_PHASE_F_SLOW=1 in the env.

Skips cleanly when gigi-stream isn't reachable -- mirrors
test_gigi_live_phase_e_snapshot.py:36-47.

To run::

    GIGI_URL=https://gigi-stream.fly.dev \\
    GIGI_API_KEY=$KEY \\
    python -m pytest \\
        inertia_damping/test_gigi_live_phase_f_orchestrator.py -v
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def base_url() -> str:
    url = os.environ.get("GIGI_URL", "https://gigi-stream.fly.dev")
    import requests
    try:
        requests.get(
            f"{url.rstrip('/')}/v1/lattice/__phase_f_probe__",
            timeout=10.0,
        )
    except requests.exceptions.ConnectionError as ex:
        pytest.skip(f"gigi-stream not reachable at {url}: {ex}")
    return url


@pytest.fixture(scope="module")
def api_key(base_url: str) -> Optional[str]:
    key = os.environ.get("GIGI_API_KEY")
    is_localhost = ("localhost" in base_url) or ("127.0.0.1" in base_url)
    if not key and not is_localhost:
        pytest.skip(
            "GIGI_API_KEY env var not set; required for non-localhost runs."
        )
    return key


@pytest.fixture
def tmp_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("halcyon_phase_f")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _slow_mode() -> bool:
    return os.environ.get("HALCYON_PHASE_F_SLOW", "0") not in ("", "0", "false", "False")


def _problem_size_args() -> List[str]:
    """Small by default; full when HALCYON_PHASE_F_SLOW=1."""
    if _slow_mode():
        return ["--steps", "1000", "--canonical-sweeps", "2048"]
    return ["--steps", "100", "--canonical-sweeps", "200"]


def _run_orchestrator(
    base_url: str,
    api_key: Optional[str],
    output_dir: Path,
    *,
    use_gigi: bool,
    seed: int = 20260616,
    extra_flags: Sequence[str] = (),
) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    """Run ``python -m inertia_damping.run_validation_report``.

    Returns ``(report_json, sidecar_npz_as_dict, run_dir_path)``.
    """
    argv: List[str] = [
        sys.executable,
        "-m",
        "inertia_damping.run_validation_report",
        "--beta",
        "2.5",
        "--seed",
        str(int(seed)),
        "--output",
        str(output_dir),
        "--no-beta-envelope",
    ]
    argv += _problem_size_args()
    if use_gigi:
        argv += ["--use-gigi", "--gigi-url", base_url]
        if api_key:
            argv += ["--gigi-api-key", api_key]
    argv += list(extra_flags)
    # NB: check=False — the orchestrator returns exit 2 when any verdict
    # category FAILs (validation_report.py:734). At Phase F smoke
    # problem sizes (--steps 100 --canonical-sweeps 200) Section 5
    # microcanonical-vs-canonical is statistically guaranteed to FAIL on
    # 50-sample blocking — that's expected, not a wire-shape bug. We
    # validate the wire by reading the produced JSON below, not by exit
    # code. A truly non-functional --use-gigi flag would crash BEFORE
    # writing the JSON (no candidates dir or no report file), which is
    # caught explicitly.
    proc = subprocess.run(
        argv,
        cwd=str(Path(__file__).resolve().parents[1]),
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    # Locate the run_<ts>/ subdirectory the orchestrator just wrote.
    candidates = sorted(
        (p for p in Path(output_dir).iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(
            f"orchestrator produced no run_<ts>/ subdir under {output_dir}. "
            f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
        )
    run_dir = candidates[-1]
    # Parse validation_report_*.json
    report_jsons = sorted(run_dir.glob("validation_report_*.json"))
    if not report_jsons:
        raise RuntimeError(
            f"no validation_report_*.json under {run_dir}; "
            f"stdout: {proc.stdout!r}"
        )
    with open(report_jsons[-1], "r", encoding="utf-8") as fh:
        report = json.load(fh)
    # Load sidecar as plain dict
    sidecar_path = run_dir / "final_state.npz"
    sidecar: Dict[str, Any] = {}
    if sidecar_path.exists():
        with np.load(sidecar_path, allow_pickle=True) as data:
            for key in data.files:
                arr = data[key]
                if arr.shape == ():
                    sidecar[key] = arr.item()
                else:
                    sidecar[key] = arr
    return report, sidecar, run_dir


# ---------------------------------------------------------------------
# G_LIVE_F0 -- --use-gigi smoke
# ---------------------------------------------------------------------
def test_G_LIVE_F0_use_gigi_smoke(
    base_url: str,
    api_key: Optional[str],
    tmp_output_dir: Path,
):
    """Smallest problem-size --use-gigi run completes; report JSON
    matches the v1.2 schema (10 categories)."""
    t0 = time.perf_counter()
    report, sidecar, run_dir = _run_orchestrator(
        base_url, api_key, tmp_output_dir,
        use_gigi=True,
        extra_flags=["--no-sector-classifier"],
    )
    wall = time.perf_counter() - t0
    schema_v = report.get("schema_version", "")
    assert schema_v in ("1.2", "1.3"), (
        f"schema_version {schema_v!r} not in expected (1.2, 1.3); "
        f"orchestrator may not be routing categories cleanly."
    )
    categories = report.get("overall_verdict", {}).get("categories", [])
    assert len(categories) == 10, (
        f"--use-gigi report had {len(categories)} categories; expected 10. "
        f"Categories: {[c.get('name') for c in categories]}"
    )
    total = report["overall_verdict"]["total"]
    assert total == 10, f"overall_verdict.total = {total}, expected 10"
    # Run-dir artifacts present
    for fname in ("trajectory.json", "final_state.npz", "SHA256SUMS"):
        assert (run_dir / fname).exists(), f"missing {fname} in {run_dir}"
    md_reports = list(run_dir.glob("validation_report_*.md"))
    json_reports = list(run_dir.glob("validation_report_*.json"))
    assert md_reports, f"no validation_report_*.md in {run_dir}"
    assert json_reports, f"no validation_report_*.json in {run_dir}"
    print(f"[F0] --use-gigi wall: {wall:.2f}s ({len(categories)} categories)")


# ---------------------------------------------------------------------
# G_LIVE_F1 -- engine-side bit-identity at fixed seed
# ---------------------------------------------------------------------
def test_G_LIVE_F1_use_gigi_bit_identity_at_seed(
    base_url: str,
    api_key: Optional[str],
    tmp_output_dir: Path,
):
    """Two consecutive --use-gigi runs at the same seed produce:
        - identical category-by-category verdicts
        - identical snapshot_sha256 from the SNAPSHOT step

    Per spec § R1: bit-identity is engine-side (two GIGI runs match
    each other), NOT cross-engine. Requires --gigi-snapshot, which
    triggers --i-confirm-this-writes-to-the-engine for non-localhost.
    """
    is_localhost = ("localhost" in base_url) or ("127.0.0.1" in base_url)
    snapshot_flags = ["--gigi-snapshot"]
    if not is_localhost:
        snapshot_flags.append("--i-confirm-this-writes-to-the-engine")
    snapshot_flags.append("--no-sector-classifier")

    run_a_dir = tmp_output_dir / "run_a"
    run_b_dir = tmp_output_dir / "run_b"
    run_a_dir.mkdir(parents=True, exist_ok=True)
    run_b_dir.mkdir(parents=True, exist_ok=True)

    report_a, sidecar_a, _ = _run_orchestrator(
        base_url, api_key, run_a_dir,
        use_gigi=True,
        seed=20260616,
        extra_flags=snapshot_flags,
    )
    report_b, sidecar_b, _ = _run_orchestrator(
        base_url, api_key, run_b_dir,
        use_gigi=True,
        seed=20260616,
        extra_flags=snapshot_flags,
    )

    # Category-by-category verdict equality (drop runtime-only keys)
    cats_a = [
        {"name": c.get("name"), "verdict": c.get("verdict")}
        for c in report_a["overall_verdict"]["categories"]
    ]
    cats_b = [
        {"name": c.get("name"), "verdict": c.get("verdict")}
        for c in report_b["overall_verdict"]["categories"]
    ]
    assert cats_a == cats_b, (
        f"--use-gigi at the same seed produced different category "
        f"verdicts.\nrun_a: {cats_a}\nrun_b: {cats_b}\n"
        f"engine-side determinism contract is broken."
    )

    # Snapshot SHA equality + format
    sha_a = sidecar_a.get("snapshot_sha256")
    sha_b = sidecar_b.get("snapshot_sha256")
    assert sha_a is not None, (
        "sidecar_a missing snapshot_sha256; --gigi-snapshot must populate it."
    )
    assert isinstance(sha_a, str), (
        f"snapshot_sha256 is not a str: {type(sha_a)}"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", sha_a), (
        f"snapshot_sha256 not 64-char lowercase hex: {sha_a!r}"
    )
    assert sha_a == sha_b, (
        f"--use-gigi snapshot SHA differs across same-seed runs:\n"
        f"  run_a: {sha_a}\n  run_b: {sha_b}\n"
        f"Engine-side bit-identity contract R1 broken."
    )


# ---------------------------------------------------------------------
# G_LIVE_F2 -- band agreement vs Python kernel
# ---------------------------------------------------------------------
def test_G_LIVE_F2_use_gigi_band_agreement_vs_python_kernel(
    base_url: str,
    api_key: Optional[str],
    tmp_output_dir: Path,
):
    """Same seed, --use-gigi vs Python kernel; canonical ⟨P⟩ within 3σ
    of the documented blocked SEM across the 6 β-scan points.

    Spec § R2: ratio = |gigi_result.canonical_P - python_result.
    canonical_P| / SEM stays under 3.0 across all 6 β-scan points.

    Slow-mode flag HALCYON_PHASE_F_SLOW=1 promotes to intermediate
    sizes (steps=400, canonical-sweeps=1024) for a more reliable
    band check (see spec § open Q on F2 wall budget).
    """
    run_g_dir = tmp_output_dir / "gigi"
    run_p_dir = tmp_output_dir / "python"
    run_g_dir.mkdir(parents=True, exist_ok=True)
    run_p_dir.mkdir(parents=True, exist_ok=True)
    extra = ["--no-sector-classifier"]
    _, sidecar_g, _ = _run_orchestrator(
        base_url, api_key, run_g_dir,
        use_gigi=True, seed=20260616, extra_flags=extra,
    )
    _, sidecar_p, _ = _run_orchestrator(
        base_url, api_key, run_p_dir,
        use_gigi=False, seed=20260616, extra_flags=extra,
    )

    # beta_scan is stored as a JSON string in the sidecar
    bs_g_raw = sidecar_g.get("beta_scan")
    bs_p_raw = sidecar_p.get("beta_scan")
    assert bs_g_raw is not None, "--use-gigi run missing beta_scan in sidecar"
    assert bs_p_raw is not None, "Python-kernel run missing beta_scan in sidecar"

    bs_g = json.loads(bs_g_raw) if isinstance(bs_g_raw, str) else list(bs_g_raw)
    bs_p = json.loads(bs_p_raw) if isinstance(bs_p_raw, str) else list(bs_p_raw)
    assert len(bs_g) == len(bs_p), (
        f"beta_scan length mismatch: gigi={len(bs_g)} python={len(bs_p)}"
    )
    for i, (g_row, p_row) in enumerate(zip(bs_g, bs_p)):
        beta = g_row["beta"]
        assert beta == pytest.approx(p_row["beta"]), (
            f"beta mismatch at index {i}: {beta} vs {p_row['beta']}"
        )
        denom = max(g_row["P_sem"], p_row["P_sem"], 1e-12)
        ratio = abs(g_row["P_measured"] - p_row["P_measured"]) / denom
        assert ratio < 3.0, (
            f"beta={beta}: |gigi.P - python.P| / SEM = {ratio:.3f} >= 3.0.\n"
            f"  gigi.P   = {g_row['P_measured']:.6f}\n"
            f"  python.P = {p_row['P_measured']:.6f}\n"
            f"  SEM      = {denom:.6f}\n"
            f"Bee-CSPRNG-decision-(c) band contract broken at index {i}."
        )
