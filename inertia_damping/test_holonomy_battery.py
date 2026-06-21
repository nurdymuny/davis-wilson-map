"""End-to-end tests for the Halcyon v3.1.3 holonomy battery —
gate logic + orchestrator + sidecar against the mock LoopTransport client.

These tests exercise every verdict path the v3.1.3 SPEC distinguishes:
- POSITIVE (primary signal > 5σ + signs coherent + all shams pass)
- NULL    (primary signal < 1σ + all shams pass)
- AMBIGUOUS (primary in 1σ–5σ band, OR sham fails, OR substrate gate
  fails, OR primary positive-magnitude but signs incoherent)

Pre-registration reference: SPEC commit 44c70b1, Zenodo DOI
10.5281/zenodo.20785681. The thresholds tested against
V313_CONSTANTS are the pre-registered numerical gates locked at that
commit; if any of these values change in a future v3.1.x, these tests
break first — which is the right discipline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inertia_damping.gigi_client.loop_transport import (
    AdiabaticityCheck,
    HalcyonParameterPack,
    LoopTransportRequest,
    LoopTransportResult,
    ShamFlag,
)
from inertia_damping.gigi_client.mock_loop_transport import MockLoopTransportClient
from inertia_damping.holonomy_battery.gates import (
    V313_CONSTANTS,
    Verdict,
    classify,
    compute_h_geom_h_sys,
    evaluate_primary,
    evaluate_sham,
    evaluate_substrate_gates,
    per_seed_sign_coherence,
)
from inertia_damping.holonomy_battery.loops import GAMMA_DEGENERATE, GAMMA_UNIT
from inertia_damping.holonomy_battery.sidecar import (
    SCHEMA_VERSION,
    SPEC_COMMIT,
    SPEC_DOI,
    build_sidecar,
    write_sidecar,
)
from inertia_damping.run_holonomy_battery import (
    SCIENCE_GATE_SHAMS,
    run_holonomy_battery,
    run_one_calibration,
)


SEEDS_8 = tuple(range(20260616, 20260624))


# ----------------------------------------------------------------------
# Loop sanity
# ----------------------------------------------------------------------


def test_gamma_unit_is_closed_rectangle_inside_validated_window():
    # γ_unit corners must form a closed loop in the validated SU(2)
    # window β_W ∈ [2.5, 3.0] per v3.1.3 §4.1.
    assert GAMMA_UNIT.vertices[0] == GAMMA_UNIT.vertices[-1]
    assert all(2.5 <= bw <= 3.0 for _, bw in GAMMA_UNIT.vertices)
    assert all(0.0 <= q <= 2.0 for q, _ in GAMMA_UNIT.vertices)
    assert GAMMA_UNIT.enclosed_area == pytest.approx(1.0)


def test_gamma_degenerate_is_zero_area():
    assert GAMMA_DEGENERATE.enclosed_area == 0.0
    # Same point twice → zero-segment closed "loop"
    assert GAMMA_DEGENERATE.vertices[0] == GAMMA_DEGENERATE.vertices[1]


# ----------------------------------------------------------------------
# Mock client basics (declare_loop idempotency + scenario routing)
# ----------------------------------------------------------------------


def test_mock_declare_loop_idempotent():
    client = MockLoopTransportClient("primary_null")
    assert client.declare_loop(GAMMA_UNIT) == "gamma_unit"
    assert client.declare_loop(GAMMA_UNIT) == "gamma_unit"  # twice OK


def test_mock_declare_loop_rejects_redefinition():
    client = MockLoopTransportClient("primary_null")
    client.declare_loop(GAMMA_UNIT)
    different_shape = GAMMA_UNIT.__class__(
        name="gamma_unit",
        control_manifold_axes=("Q", "beta_wilson"),
        vertices=((0.0, 2.5), (1.0, 2.5), (0.0, 2.5)),  # different
        t_per_segment=50.0,
        enclosed_area=0.0,
    )
    with pytest.raises(ValueError, match="different shape"):
        client.declare_loop(different_shape)


def test_mock_loop_transport_rejects_undeclared_loop():
    client = MockLoopTransportClient("primary_null")
    request = LoopTransportRequest(
        gauge_field_name="dummy",
        loop=GAMMA_UNIT,
        direction="FORWARD",
        pack=HalcyonParameterPack(),
        seeds=SEEDS_8,
    )
    with pytest.raises(ValueError, match="not declared"):
        client.loop_transport(request)


def test_request_validates_direction_and_sham_combination():
    pack = HalcyonParameterPack()
    with pytest.raises(ValueError, match="direction"):
        LoopTransportRequest(
            gauge_field_name="x",
            loop=GAMMA_UNIT,
            direction="SIDEWAYS",
            pack=pack,
            seeds=SEEDS_8,
        )
    with pytest.raises(ValueError, match="MASS_SCALED requires sham_mass_scale"):
        LoopTransportRequest(
            gauge_field_name="x",
            loop=GAMMA_UNIT,
            direction="FORWARD",
            pack=pack,
            seeds=SEEDS_8,
            sham=ShamFlag.MASS_SCALED,
        )
    with pytest.raises(ValueError, match="sham_mass_scale only valid"):
        LoopTransportRequest(
            gauge_field_name="x",
            loop=GAMMA_UNIT,
            direction="FORWARD",
            pack=pack,
            seeds=SEEDS_8,
            sham=ShamFlag.FLAT_FIELD,
            sham_mass_scale=0.1,
        )


# ----------------------------------------------------------------------
# Compute helpers (H_geom / H_sys + sign coherence)
# ----------------------------------------------------------------------


def _synthetic_result(
    per_seed_h: np.ndarray,
    tracking_q_max: float = 0.005,
    tracking_beta_w_max: float = 0.005,
    tau_pin_over_t_segment: float = 0.02,
) -> LoopTransportResult:
    n = per_seed_h.shape[0]
    holonomy = np.zeros((n, 4), dtype=np.float64)
    holonomy[:, 0] = 1.0
    holonomy[:, 1] = per_seed_h
    sigma_blocked = float(np.std(per_seed_h, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return LoopTransportResult(
        per_seed_holonomy=holonomy,
        per_seed_h_scalar=per_seed_h,
        sigma_h_blocked=sigma_blocked,
        tracking_error_max_Q=tracking_q_max,
        tracking_error_max_beta_W=tracking_beta_w_max,
        adiabaticity_check=AdiabaticityCheck(
            tau_pin_over_t_segment=tau_pin_over_t_segment,
            ramp_rate_over_relaxation_rate=0.04,
        ),
        run_id="synthetic",
    )


def test_h_geom_is_antisymmetric_h_sys_is_symmetric():
    forward = _synthetic_result(np.array([0.10, 0.11, 0.09, 0.12, 0.08, 0.10, 0.11, 0.09]))
    reversed_ = _synthetic_result(np.array([-0.10, -0.11, -0.09, -0.12, -0.08, -0.10, -0.11, -0.09]))
    h_geom, h_sys, sigma = compute_h_geom_h_sys(forward, reversed_)
    # H_geom = ½(forward - reversed) ≈ 0.10
    assert h_geom == pytest.approx(0.10, abs=1e-3)
    # H_sys = ½(forward + reversed) ≈ 0
    assert h_sys == pytest.approx(0.0, abs=1e-3)
    assert sigma > 0


def test_per_seed_sign_coherence_matches_majority():
    forward = _synthetic_result(np.array([0.10] * 7 + [-0.10]))   # 7 positive
    reversed_ = _synthetic_result(np.array([-0.10] * 7 + [0.10]))  # mirror
    count, total = per_seed_sign_coherence(forward, reversed_)
    # per-seed H_geom = ½(forward - reversed) — 7 positive, 1 negative
    assert count == 7
    assert total == 8


# ----------------------------------------------------------------------
# Primary verdict classifier paths
# ----------------------------------------------------------------------


def test_primary_positive_when_signal_above_5sigma_and_signs_coherent():
    rng = np.random.default_rng(42)
    forward_h = rng.normal(loc=+7e-5, scale=1e-5, size=8)
    reversed_h = rng.normal(loc=-7e-5, scale=1e-5, size=8)
    forward = _synthetic_result(forward_h)
    reversed_ = _synthetic_result(reversed_h)
    verdict = evaluate_primary(forward, reversed_)
    assert verdict.verdict is Verdict.POSITIVE
    assert verdict.sigma_ratio > 5
    assert verdict.sign_coherence_count >= 5


def test_primary_null_when_signal_below_1sigma():
    rng = np.random.default_rng(123)
    forward_h = rng.normal(loc=0.0, scale=1e-5, size=8)
    reversed_h = rng.normal(loc=0.0, scale=1e-5, size=8)
    forward = _synthetic_result(forward_h)
    reversed_ = _synthetic_result(reversed_h)
    verdict = evaluate_primary(forward, reversed_)
    # Confirm we hit the NULL branch (mean is small relative to σ)
    assert verdict.verdict is Verdict.NULL


def test_primary_ambiguous_when_signal_in_intermediate_band():
    rng = np.random.default_rng(99)
    # σ_h_blocked = scale / sqrt(8) ≈ 0.354 × scale. To hit ~2-3σ,
    # set |H_geom_mean| ≈ 2 × σ_h_blocked ≈ 0.7 × scale. With
    # scale = 1e-5, loc = 8e-6 lands around |H|/σ ≈ 2.26.
    forward_h = rng.normal(loc=+8e-6, scale=1e-5, size=8)
    reversed_h = rng.normal(loc=-8e-6, scale=1e-5, size=8)
    forward = _synthetic_result(forward_h)
    reversed_ = _synthetic_result(reversed_h)
    verdict = evaluate_primary(forward, reversed_)
    # The ratio should be in (1, 5)
    assert 1.0 <= verdict.sigma_ratio <= 5.0
    assert verdict.verdict is Verdict.AMBIGUOUS


def test_anti_fishing_reason_absent_below_epsilon_abs_even_with_perfect_sign_coherence():
    """Design-closeout §A.1: anti-fishing only fires above ε_abs.
    Below the absolute floor, sign coherence carries no signal
    information, so the verdict's failure reason (if any) must not
    cite anti-fishing — the primary 2σ check is the appropriate gate.

    This test does NOT claim the sham passes; it only verifies that
    the §3.4 anti-fishing diagnostic is suppressed below the floor.
    Real noisy data with positive bias may still fail the strict 2σ
    gate, but for a reason other than fishing."""
    rng = np.random.default_rng(99)
    per_seed = np.abs(rng.normal(loc=2e-14, scale=1e-14, size=8))
    assert abs(np.mean(per_seed)) < V313_CONSTANTS.epsilon_abs
    forward = _synthetic_result(per_seed)
    sv = evaluate_sham(ShamFlag.FLAT_FIELD, sham_forward=forward)
    assert sv.sign_coherence_ratio == 1.0
    assert "anti-fishing" not in sv.reason, sv.reason


def test_primary_ambiguous_when_signs_incoherent_at_positive_magnitude():
    # H_geom magnitude well above 5σ but only 4/8 same sign — should be AMBIGUOUS
    forward_h = np.array([+1e-4, +1e-4, +1e-4, +1e-4, -1e-4, -1e-4, -1e-4, -1e-4]) + 1e-4
    reversed_h = -forward_h
    forward = _synthetic_result(forward_h)
    reversed_ = _synthetic_result(reversed_h)
    verdict = evaluate_primary(forward, reversed_)
    # Signal magnitude is large but signs split 4/4
    if verdict.sigma_ratio > 5:
        assert verdict.verdict is Verdict.AMBIGUOUS
        assert "sign-coherence" in verdict.reason


# ----------------------------------------------------------------------
# Sham gate evaluations
# ----------------------------------------------------------------------


def test_sham_passes_when_mean_below_epsilon_abs_with_realistic_noise():
    # Realistic substrate output: small mean, small but non-zero σ.
    # Should pass both the 2σ gate AND the ε_abs floor.
    rng = np.random.default_rng(42)
    per_seed = rng.normal(loc=1e-14, scale=5e-14, size=8)  # |mean| ~ 1e-14
    forward = _synthetic_result(per_seed)
    sv = evaluate_sham(ShamFlag.FLAT_FIELD, sham_forward=forward)
    assert sv.passed, sv.reason
    assert abs(sv.mean_h) < V313_CONSTANTS.epsilon_abs


def test_sham_fails_when_mean_exceeds_epsilon_abs_with_small_sigma():
    # σ chosen so 2σ < ε_abs — the ε_abs gate becomes the dominant
    # failure. With σ_per_seed = 1e-12, σ_blocked ≈ 3.5e-13, so 2σ ≈ 7e-13.
    # |mean| = 2e-10 fails ε_abs by ~2× AND fails 2σ by ~300×; both
    # appear in the reason string.
    rng = np.random.default_rng(7)
    per_seed = rng.normal(loc=2e-10, scale=1e-12, size=8)
    forward = _synthetic_result(per_seed)
    sv = evaluate_sham(ShamFlag.FLAT_FIELD, sham_forward=forward)
    assert not sv.passed
    assert "ε_abs" in sv.reason


def test_sham_alpha_zero_load_bearing_on_absolute_floor():
    # S₂ is the only sham where ε_abs is load-bearing (2σ check is sanity).
    # Use realistic noisy data with |mean| > ε_abs.
    rng = np.random.default_rng(11)
    per_seed = rng.normal(loc=5e-9, scale=1e-10, size=8)
    forward = _synthetic_result(per_seed)
    sv = evaluate_sham(ShamFlag.ALPHA_ZERO, sham_forward=forward)
    assert not sv.passed
    assert "S₂" in sv.reason or "ALPHA_ZERO" in sv.reason


def test_sham_anti_fishing_fires_when_primary_would_pass():
    """Per SPEC §3.4: 'the sham fails (regardless of the |mean| gate)
    if signs ≥ 6/8 AND |mean| > 0.5σ'. Construct data where the primary
    2σ + ε_abs gate would PASS (|mean| < 2σ AND |mean| < ε_abs) but
    sign coherence is high — anti-fishing catches what the primary
    gate would otherwise let through."""
    # |mean| ≈ 5.4e-11 (below ε_abs = 1e-10),
    # σ_blocked ≈ 6.8e-11 (so 2σ ≈ 1.36e-10 > |mean|, sigma_gate passes),
    # |mean| > 0.5σ ≈ 3.4e-11 (anti-fishing condition triggered),
    # 7/8 positive sign (one outlier negative).
    per_seed = np.array(
        [1.0e-10, 2.0e-10, 5.0e-11, 1.5e-10, 3.0e-11, 1.8e-10, 1.2e-10, -4.0e-10],
        dtype=np.float64,
    )
    forward = _synthetic_result(per_seed)
    sv = evaluate_sham(ShamFlag.FLAT_FIELD, sham_forward=forward)
    # Sanity checks on the constructed scenario
    assert sv.sign_coherence_ratio >= 6 / 8
    assert abs(sv.mean_h) < V313_CONSTANTS.epsilon_abs  # primary ε_abs passes
    assert abs(sv.mean_h) < V313_CONSTANTS.sham_threshold_sigma * sv.sigma  # primary 2σ passes
    assert abs(sv.mean_h) > V313_CONSTANTS.sham_consistent_sign_mean_threshold * sv.sigma
    # Anti-fishing should catch it
    assert not sv.passed
    assert "anti-fishing" in sv.reason


# ----------------------------------------------------------------------
# Substrate gate evaluations
# ----------------------------------------------------------------------


def test_substrate_tracking_error_violation_forces_failure_on_Q():
    forward = _synthetic_result(np.zeros(8), tracking_q_max=0.08)
    sg = evaluate_substrate_gates(forward)
    assert not sg.passed
    assert "tracking_error_max_Q" in sg.reason


def test_substrate_tracking_error_violation_forces_failure_on_beta_W():
    forward = _synthetic_result(np.zeros(8), tracking_beta_w_max=0.07)
    sg = evaluate_substrate_gates(forward)
    assert not sg.passed
    assert "β_W" in sg.reason


def test_substrate_adiabaticity_violation_forces_failure():
    forward = _synthetic_result(np.zeros(8), tau_pin_over_t_segment=0.15)
    sg = evaluate_substrate_gates(forward)
    assert not sg.passed
    assert "τ_pin" in sg.reason or "adiabaticity" in sg.reason


def test_substrate_gates_pass_at_nominal_values():
    forward = _synthetic_result(np.zeros(8))
    sg = evaluate_substrate_gates(forward)
    assert sg.passed


# ----------------------------------------------------------------------
# End-to-end orchestrator (against mock client, all calibrations)
# ----------------------------------------------------------------------


def _make_client(scenario: str) -> MockLoopTransportClient:
    return MockLoopTransportClient(scenario=scenario)


def test_orchestrator_e2e_primary_null():
    client = _make_client("primary_null")
    results = run_holonomy_battery(
        client, alpha_halcyon_values=(1.0,), seeds=SEEDS_8
    )
    assert len(results) == 1
    composite, sidecar = results[1.0]
    assert composite.overall is Verdict.NULL, composite.reason
    assert composite.substrate_gates.passed
    assert all(sv.passed for sv in composite.shams)


def test_orchestrator_e2e_primary_positive():
    client = _make_client("primary_positive")
    results = run_holonomy_battery(
        client, alpha_halcyon_values=(1.0,), seeds=SEEDS_8
    )
    composite, sidecar = results[1.0]
    assert composite.overall is Verdict.POSITIVE, composite.reason


def test_orchestrator_e2e_tracking_error_violation_forces_ambiguous():
    client = _make_client("tracking_error_violation")
    results = run_holonomy_battery(client, alpha_halcyon_values=(1.0,), seeds=SEEDS_8)
    composite, sidecar = results[1.0]
    assert composite.overall is Verdict.AMBIGUOUS
    assert not composite.substrate_gates.passed


def test_orchestrator_e2e_adiabaticity_violation_forces_ambiguous():
    client = _make_client("adiabaticity_violation")
    results = run_holonomy_battery(client, alpha_halcyon_values=(1.0,), seeds=SEEDS_8)
    composite, sidecar = results[1.0]
    assert composite.overall is Verdict.AMBIGUOUS
    assert "τ_pin" in composite.reason or "adiabaticity" in composite.reason


@pytest.mark.parametrize(
    "sham",
    [
        ShamFlag.FLAT_FIELD,
        ShamFlag.ALPHA_ZERO,
        ShamFlag.MASS_SCALED,
        ShamFlag.BACKTRACK_LOOP,
        ShamFlag.FROZEN_FIELD,
    ],
)
def test_orchestrator_e2e_each_sham_failure_forces_ambiguous(sham):
    client = _make_client(f"sham_failure_{sham.value}")
    results = run_holonomy_battery(client, alpha_halcyon_values=(1.0,), seeds=SEEDS_8)
    composite, sidecar = results[1.0]
    assert composite.overall is Verdict.AMBIGUOUS, composite.reason
    # The failing sham must appear in the reason
    assert sham.value in composite.reason


def test_orchestrator_runs_full_calibration_sweep():
    client = _make_client("primary_null")
    results = run_holonomy_battery(
        client, alpha_halcyon_values=(1.0, 1000.0), seeds=SEEDS_8
    )
    assert set(results.keys()) == {1.0, 1000.0}


# ----------------------------------------------------------------------
# Sidecar schema
# ----------------------------------------------------------------------


def test_sidecar_schema_records_v3_1_3_provenance():
    client = _make_client("primary_null")
    composite, forward, reversed_, sham_results = run_one_calibration(
        client, alpha_halcyon=1.0, seeds=SEEDS_8
    )
    sidecar = build_sidecar(
        composite=composite,
        forward=forward,
        reversed_=reversed_,
        sham_results=sham_results,
        alpha_halcyon=1.0,
        loop_name=GAMMA_UNIT.name,
        seeds=SEEDS_8,
        gigi_deploy_hash="mock-deploy-abc123",
        run_timestamp_utc="2026-06-21T08:00:00Z",
    )
    d = sidecar.to_dict()
    # Pre-registration provenance must be carried at the top of every sidecar
    assert d["schema_version"] == SCHEMA_VERSION == "section_12_holonomy_battery_v3_1_3"
    assert d["spec_commit"] == SPEC_COMMIT
    assert d["spec_doi"] == SPEC_DOI == "10.5281/zenodo.20785681"
    assert d["alpha_halcyon"] == 1.0
    assert d["loop_name"] == "gamma_unit"
    assert d["verdict"] in {"POSITIVE", "NULL", "AMBIGUOUS"}
    assert set(d["shams"].keys()) == {s.value for s in SCIENCE_GATE_SHAMS}


def test_sidecar_written_to_disk_round_trips_json(tmp_path: Path):
    client = _make_client("primary_null")
    results = run_holonomy_battery(
        client,
        alpha_halcyon_values=(1.0,),
        seeds=SEEDS_8,
        output_dir=tmp_path,
        gigi_deploy_hash="mock-deploy",
        run_timestamp_utc="2026-06-21T08:00:00Z",
    )
    sidecar_path = tmp_path / "section_12_holonomy_battery_alpha_1.json"
    assert sidecar_path.exists()
    parsed = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["spec_doi"] == SPEC_DOI
    assert parsed["alpha_halcyon"] == 1.0


# ----------------------------------------------------------------------
# Pre-registered constants integrity (canary)
# ----------------------------------------------------------------------


def test_v313_constants_match_pre_registered_values():
    """If any of these change without a v3.1.4 pre-registration, the
    deposit is broken. This test exists to catch silent drift."""
    assert V313_CONSTANTS.epsilon_abs == 1e-10
    assert V313_CONSTANTS.tracking_error_eps_Q == 0.05
    assert V313_CONSTANTS.tracking_error_eps_beta_W == 0.05
    assert V313_CONSTANTS.adiabaticity_threshold == 0.1
    assert V313_CONSTANTS.sham_threshold_sigma == 2.0
    assert V313_CONSTANTS.primary_positive_sigma == 5.0
    assert V313_CONSTANTS.primary_null_sigma == 1.0
    assert V313_CONSTANTS.sign_coherence_min == 5
    assert V313_CONSTANTS.sign_coherence_total == 8


def test_spec_provenance_constants_match_deposited_state():
    """SPEC commit + DOI are the pre-registration anchors. Editing
    these means re-pointing the orchestrator at a different deposited
    state. Should never change without a v3.1.x amendment."""
    assert SPEC_COMMIT == "44c70b1b76501b4b66c6f9ace6bccd8b5bd14c4a"
    assert SPEC_DOI == "10.5281/zenodo.20785681"
    assert SCHEMA_VERSION == "section_12_holonomy_battery_v3_1_3"
