"""Halcyon orchestrator entry point — `run_holonomy_battery.py`.

Per v3.1.3 §4.6:

    def run_holonomy_battery(alpha_halcyon, seeds, log_path):
        forward = gigi_loop_transport(loop=GAMMA_UNIT, direction=FORWARD, ...)
        reversed_ = gigi_loop_transport(loop=GAMMA_UNIT, direction=REVERSED, ...)
        H_geom = 0.5 * (forward.H - reversed_.H)
        H_sys = 0.5 * (forward.H + reversed_.H)
        shams = {name: gigi_loop_transport(loop=..., sham_flag=name, ...)
                 for name in SHAM_FLAGS}
        return apply_v3_1_3_gates(H_geom, H_sys, shams, ...)

"No leapfrog. No demodulation. No force computation. All substrate."

This module is the thin delegation wrapper. The substantive work is
delegated to:
- LoopTransportClient (substrate via GIGI verb, mock for tests)
- holonomy_battery.gates (the §3 verdict logic)
- holonomy_battery.sidecar (the §7.2 sidecar emission)

Pre-registration: SPEC v3.1.3, commit 44c70b1, Zenodo DOI
10.5281/zenodo.20785681. The orchestrator does NOT compute any of the
pre-registered thresholds — it applies them as constants from
holonomy_battery.gates.V313_CONSTANTS.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from inertia_damping.gigi_client.loop_transport import (
    HalcyonParameterPack,
    LoopTransportClient,
    LoopTransportRequest,
    LoopTransportResult,
    ShamFlag,
)
from inertia_damping.gigi_client.mock_loop_transport import MockLoopTransportClient
from inertia_damping.holonomy_battery.gates import (
    CompositeVerdict,
    V313_CONSTANTS,
    Verdict,
    classify,
)
from inertia_damping.holonomy_battery.loops import GAMMA_DEGENERATE, GAMMA_UNIT
from inertia_damping.holonomy_battery.sidecar import Section12Sidecar, build_sidecar, write_sidecar


# v3.1.3 §3.2's science-gate sham set (S₁, S₂, S₃, S₅, S₆;
# S₄ folded into the antisymmetric primary observable).
SCIENCE_GATE_SHAMS: Tuple[ShamFlag, ...] = (
    ShamFlag.FLAT_FIELD,
    ShamFlag.ALPHA_ZERO,
    ShamFlag.MASS_SCALED,
    ShamFlag.BACKTRACK_LOOP,
    ShamFlag.FROZEN_FIELD,
)

# Default v3.1.3 calibration sweep (per §3.6)
DEFAULT_CALIBRATIONS: Tuple[float, ...] = (1.0, 1000.0)
DEFAULT_SEEDS: Tuple[int, ...] = tuple(range(20260616, 20260624))  # 8 seeds


def _run_one_call(
    client: LoopTransportClient,
    *,
    gauge_field_name: str,
    loop_handle,
    direction: str,
    pack: HalcyonParameterPack,
    seeds: Tuple[int, ...],
    sham: ShamFlag = ShamFlag.NONE,
    sham_mass_scale: Optional[float] = None,
) -> LoopTransportResult:
    """Build one LoopTransportRequest, send it, return the result."""
    request = LoopTransportRequest(
        gauge_field_name=gauge_field_name,
        loop=loop_handle,
        direction=direction,
        pack=pack,
        seeds=seeds,
        sham=sham,
        sham_mass_scale=sham_mass_scale,
    )
    return client.loop_transport(request)


def run_one_calibration(
    client: LoopTransportClient,
    *,
    alpha_halcyon: float,
    seeds: Tuple[int, ...] = DEFAULT_SEEDS,
    gauge_field_name: str = "halcyon_canonical_buckyball",
    base_pack: Optional[HalcyonParameterPack] = None,
) -> Tuple[CompositeVerdict, LoopTransportResult, LoopTransportResult, Dict[ShamFlag, LoopTransportResult]]:
    """Run one α calibration end-to-end. Returns the composite verdict
    plus the per-call results needed for the sidecar."""
    pack = (base_pack or HalcyonParameterPack()).__class__(
        **{**(base_pack or HalcyonParameterPack()).__dict__, "alpha": alpha_halcyon}
    )

    # Ensure loops are declared (idempotent)
    client.declare_loop(GAMMA_UNIT)
    client.declare_loop(GAMMA_DEGENERATE)

    # Primary forward + reversed traversals on γ_unit
    forward = _run_one_call(
        client,
        gauge_field_name=gauge_field_name,
        loop_handle=GAMMA_UNIT,
        direction="FORWARD",
        pack=pack,
        seeds=seeds,
    )
    reversed_ = _run_one_call(
        client,
        gauge_field_name=gauge_field_name,
        loop_handle=GAMMA_UNIT,
        direction="REVERSED",
        pack=pack,
        seeds=seeds,
    )

    # Each sham: one forward call on the appropriate loop
    sham_results: Dict[ShamFlag, LoopTransportResult] = {}
    for sham in SCIENCE_GATE_SHAMS:
        sham_loop = GAMMA_DEGENERATE if sham is ShamFlag.BACKTRACK_LOOP else GAMMA_UNIT
        if sham is ShamFlag.MASS_SCALED:
            # S₃ runs three sub-calls; the orchestrator picks the
            # mu_baseline = 1.0 case as the gate input (per §3.2
            # NULL/AMBIGUOUS branch handling). The other two scales
            # (×0.1, ×10) are recorded but not gate-applied unless
            # the primary verdict is POSITIVE — for v0.1 we exercise
            # only the canonical scale; baseline-subtraction is a
            # v0.2 elaboration.
            sham_results[sham] = _run_one_call(
                client,
                gauge_field_name=gauge_field_name,
                loop_handle=sham_loop,
                direction="FORWARD",
                pack=pack,
                seeds=seeds,
                sham=sham,
                sham_mass_scale=1.0,
            )
        else:
            sham_results[sham] = _run_one_call(
                client,
                gauge_field_name=gauge_field_name,
                loop_handle=sham_loop,
                direction="FORWARD",
                pack=pack,
                seeds=seeds,
                sham=sham,
            )

    composite = classify(forward, reversed_, sham_results, V313_CONSTANTS)
    return composite, forward, reversed_, sham_results


def run_holonomy_battery(
    client: LoopTransportClient,
    *,
    alpha_halcyon_values: Tuple[float, ...] = DEFAULT_CALIBRATIONS,
    seeds: Tuple[int, ...] = DEFAULT_SEEDS,
    output_dir: Optional[Path] = None,
    gigi_deploy_hash: str = "unknown",
    run_timestamp_utc: str = "",
) -> Dict[float, Tuple[CompositeVerdict, Section12Sidecar]]:
    """Run the v3.1.3 battery across the calibration sweep.

    Returns a dict mapping α → (composite verdict, sidecar). If
    `output_dir` is provided, writes one sidecar JSON per α.
    """
    results: Dict[float, Tuple[CompositeVerdict, Section12Sidecar]] = {}
    for alpha in alpha_halcyon_values:
        composite, forward, reversed_, sham_results = run_one_calibration(
            client,
            alpha_halcyon=alpha,
            seeds=seeds,
        )
        sidecar = build_sidecar(
            composite=composite,
            forward=forward,
            reversed_=reversed_,
            sham_results=sham_results,
            alpha_halcyon=alpha,
            loop_name=GAMMA_UNIT.name,
            seeds=seeds,
            gigi_deploy_hash=gigi_deploy_hash,
            run_timestamp_utc=run_timestamp_utc,
        )
        if output_dir is not None:
            output_path = output_dir / f"section_12_holonomy_battery_alpha_{alpha:g}.json"
            write_sidecar(sidecar, output_path)
        results[alpha] = (composite, sidecar)
    return results


# ----------------------------------------------------------------------
# CLI entry point — primary use is `python -m inertia_damping.run_holonomy_battery --mock-scenario primary_null`
# ----------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Halcyon v3.1.3 holonomy battery against a "
            "LoopTransportClient (mock by default; live client when GIGI's "
            "LOOP_TRANSPORT verb ships)."
        )
    )
    parser.add_argument(
        "--mock-scenario",
        default="primary_null",
        help=(
            "MockLoopTransportClient scenario for testing without the live "
            "substrate. Examples: primary_null, primary_positive, "
            "primary_ambiguous_sigma, tracking_error_violation, "
            "adiabaticity_violation, sham_failure_FLAT_FIELD, ..."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=list(DEFAULT_CALIBRATIONS),
        help="α_Halcyon calibration values to sweep (default: 1.0 1000.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write section_12 sidecar JSONs (default: stdout-only)",
    )
    parser.add_argument(
        "--client",
        choices=("mock", "live"),
        default="mock",
        help=(
            "LoopTransportClient backend. mock = deterministic scenario data "
            "(no network, default); live = HTTP POST /v1/gql against GIGI_URL. "
            "Live REQUIRES VI.4 shipped before sham gates are meaningful and "
            "VI.5 shipped before per-seed values are reproducible across "
            "recompiles. Use --client live for integration testing only until "
            "those land; the publication-bound run waits for VI.5."
        ),
    )
    parser.add_argument(
        "--gigi-url",
        default=None,
        help=(
            "Override the live-client base URL (defaults to GIGI_URL env var "
            "or http://localhost:3142). Only used when --client live."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=(
            "Per-seed list for the ensemble. v3.1.3 §3.5 pre-registers 8 seeds "
            "[20260616..20260623]; overriding for smoke tests is OK but the "
            "publication-bound run uses the pre-registered set."
        ),
    )
    return parser.parse_args(argv)


def _make_client(args: argparse.Namespace):
    """Build either a Mock or Live client per --client flag."""
    if args.client == "live":
        # Lazy import — the mock-only path must not require `requests`.
        from inertia_damping.gigi_client.live_loop_transport import (
            LiveLoopTransportClient,
        )
        print(
            "WARNING: --client live talks to a real GIGI substrate.\n"
            "         Halcyon's gate logic produces meaningful POSITIVE/NULL/\n"
            "         AMBIGUOUS verdicts only when VI.4 (SHAM block dispatch)\n"
            "         AND VI.5 (bit-identity per-seed gold fixture) are both\n"
            "         green on the substrate side. Use --client live for\n"
            "         integration testing only until those land; the\n"
            "         publication-bound run waits for VI.5.\n",
        )
        return LiveLoopTransportClient(base_url=args.gigi_url)
    return MockLoopTransportClient(scenario=args.mock_scenario)


def _reconfigure_stdout_utf8() -> None:
    """Windows consoles default to cp1252 and choke on the Greek
    letters (α, σ, β_W) the print output uses. Reconfigure stdout to
    UTF-8 if it isn't already; fall back gracefully on platforms where
    sys.stdout has no reconfigure() method."""
    if getattr(sys.stdout, "encoding", "").lower() == "utf-8":
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def main(argv: Optional[List[str]] = None) -> int:
    _reconfigure_stdout_utf8()
    args = _parse_args(argv)
    client = _make_client(args)
    results = run_holonomy_battery(
        client,
        alpha_halcyon_values=tuple(args.alpha),
        seeds=tuple(args.seeds),
        output_dir=args.output_dir,
    )
    for alpha, (composite, sidecar) in results.items():
        print(f"\n=== α = {alpha} ===")
        print(f"  Verdict: {composite.overall.value}")
        print(f"  Reason:  {composite.reason}")
        print(f"  H_geom_mean = {composite.primary.h_geom_mean:.3e}")
        print(f"  σ_H_blocked = {composite.primary.sigma_h_blocked:.3e}")
        print(f"  |H_geom|/σ  = {composite.primary.sigma_ratio:.2f}")
        print(f"  H_sys       = {composite.primary.h_sys:.3e}")
        print(f"  sign coherence: {composite.primary.sign_coherence_count}/{composite.primary.sign_coherence_total}")
        print(f"  shams:")
        for sv in composite.shams:
            print(f"    {sv.flag.value:18s} passed={sv.passed} — {sv.reason}")
        print(f"  substrate gates: passed={composite.substrate_gates.passed} — {composite.substrate_gates.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
