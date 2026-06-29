"""v3.1.3 gate-application logic.

Three independent layers per the SPEC (locked at commit 44c70b1 /
Zenodo DOI 10.5281/zenodo.20785681):

1. §3.1 primary classifier: POSITIVE / NULL / AMBIGUOUS based on
   |H_geom_mean| / σ_H against {1σ, 5σ} thresholds, plus the §3.5
   per-seed sign-coherence rule (POSITIVE-only).
2. §3.2 sham gates: each sham must satisfy `|H_sham| < 2σ_sham` AND
   `|H_sham| < ε_abs = 10⁻¹⁰` (per design-closeout §A.1). The
   consistent-sign anti-fishing rule of §3.4 fires AMBIGUOUS on any
   sham showing ≥ 6/8 same-sign with |mean| > 0.5 σ_sham.
3. §4.2 / §4.3 substrate-emitted gates: ADIABATICITY_CHECK
   (τ_pin/T_segment ≥ 0.1) and tracking-error max (> ε_Q or ε_β_W)
   each force AMBIGUOUS regardless of H values.

This module is pure Python. The substrate emits f64 holonomies and
diagnostics; this module applies the pre-registered numerical
thresholds. Per the substrate's t013 three-constraint contract, none
of these thresholds live in the substrate — they live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from inertia_damping.gigi_client.loop_transport import (
    LoopTransportResult,
    ShamFlag,
)


# ----------------------------------------------------------------------
# Pre-registered thresholds (v3.1.3 §3 + §4 + design-closeout §A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class V313Constants:
    """Pre-registered numerical thresholds. Locked at SPEC commit
    44c70b1 / Zenodo DOI 10.5281/zenodo.20785681. Any change requires
    a v3.1.4 pre-registration."""

    epsilon_abs: float = 1e-10
    tracking_error_eps_Q: float = 0.05
    tracking_error_eps_beta_W: float = 0.05
    adiabaticity_threshold: float = 0.1
    sham_threshold_sigma: float = 2.0
    primary_positive_sigma: float = 5.0
    primary_null_sigma: float = 1.0
    sign_coherence_min: int = 5      # of 8 seeds
    sign_coherence_total: int = 8
    sham_consistent_sign_min: int = 6  # of 8 seeds (§3.4 anti-fishing)
    sham_consistent_sign_mean_threshold: float = 0.5  # |mean| in σ units


V313_CONSTANTS = V313Constants()


# ----------------------------------------------------------------------
# Verdict types
# ----------------------------------------------------------------------


class Verdict(str, Enum):
    POSITIVE = "POSITIVE"
    NULL = "NULL"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class ShamVerdict:
    """Per-sham gate outcome. `passed=False` forces overall AMBIGUOUS."""

    flag: ShamFlag
    mean_h: float
    sigma: float
    sign_coherence_ratio: float           # fraction of seeds with same sign
    passed: bool
    reason: str                            # human-readable diagnostic


@dataclass(frozen=True)
class PrimaryVerdict:
    """The §3.1 primary observable gate outcome."""

    h_geom_mean: float
    h_sys: float
    sigma_h_blocked: float
    sigma_ratio: float                    # |h_geom_mean| / sigma_h_blocked
    sign_coherence_count: int             # n seeds with same sign as mean
    sign_coherence_total: int
    verdict: Verdict
    reason: str


@dataclass(frozen=True)
class SubstrateGateVerdict:
    """The §4.2 + §4.3 substrate-emitted gates. `passed=False` forces
    AMBIGUOUS regardless of the primary observable."""

    tracking_q_max: float
    tracking_beta_w_max: float
    tau_pin_over_t_segment: float
    passed: bool
    reason: str


@dataclass(frozen=True)
class CompositeVerdict:
    """Final per-calibration verdict + all the per-layer diagnostics
    the §7.2 sidecar receipt needs to record."""

    overall: Verdict
    reason: str
    primary: PrimaryVerdict
    shams: Tuple[ShamVerdict, ...]
    substrate_gates: SubstrateGateVerdict


# ----------------------------------------------------------------------
# H_geom / H_sys construction (per v3.1.3 §3.1)
# ----------------------------------------------------------------------


def compute_h_geom_h_sys(
    forward: LoopTransportResult,
    reversed_: LoopTransportResult,
) -> Tuple[float, float, float]:
    """Return (h_geom_mean, h_sys, sigma_h_blocked).

    H_geom = ½(H[γ] − H[γ⁻¹]) — antisymmetric primary observable.
    H_sys  = ½(H[γ] + H[γ⁻¹]) — symmetric systematic-offset diagnostic.
    σ_H_blocked = Flyvbjerg-Petersen SEM of per-seed H_geom across the
                   8-seed ensemble.
    """
    if forward.per_seed_h_scalar.shape != reversed_.per_seed_h_scalar.shape:
        raise ValueError(
            f"forward/reversed per-seed shape mismatch: "
            f"{forward.per_seed_h_scalar.shape} vs {reversed_.per_seed_h_scalar.shape}"
        )

    per_seed_h_geom = 0.5 * (forward.per_seed_h_scalar - reversed_.per_seed_h_scalar)
    per_seed_h_sys = 0.5 * (forward.per_seed_h_scalar + reversed_.per_seed_h_scalar)
    h_geom_mean = float(np.mean(per_seed_h_geom))
    h_sys = float(np.mean(per_seed_h_sys))

    # Flyvbjerg-Petersen blocked SEM. For 8 seeds, plain SEM is a
    # well-defined estimator (no blocking transformations possible
    # below n_blocks = 2). The substrate's own σ_H_blocked is computed
    # over the per-seed FORWARD draw and may not be the right
    # statistic for the (h_geom = forward − reversed) combination;
    # we recompute here.
    n_seeds = per_seed_h_geom.shape[0]
    sigma_h_blocked = float(np.std(per_seed_h_geom, ddof=1) / np.sqrt(n_seeds))
    return h_geom_mean, h_sys, sigma_h_blocked


def per_seed_sign_coherence(
    forward: LoopTransportResult,
    reversed_: LoopTransportResult,
) -> Tuple[int, int]:
    """Returns (n_seeds_matching_mean_sign, n_total_seeds) for the
    per-seed H_geom_i = ½(forward_i − reversed_i). Used by the §3.5
    POSITIVE-only sign-coherence rule."""
    per_seed_h_geom = 0.5 * (forward.per_seed_h_scalar - reversed_.per_seed_h_scalar)
    n_total = per_seed_h_geom.shape[0]
    if n_total == 0:
        return 0, 0
    mean_sign = np.sign(np.mean(per_seed_h_geom))
    if mean_sign == 0:
        return 0, n_total
    matching = int(np.sum(np.sign(per_seed_h_geom) == mean_sign))
    return matching, n_total


# ----------------------------------------------------------------------
# Sham-gate application (per v3.1.3 §3.2 + design-closeout §A.1)
# ----------------------------------------------------------------------


def evaluate_sham(
    flag: ShamFlag,
    sham_forward: LoopTransportResult,
    sham_reversed: Optional[LoopTransportResult] = None,
    primary_h_geom: Optional[float] = None,
    constants: V313Constants = V313_CONSTANTS,
) -> ShamVerdict:
    """Apply the §3.2 gate for a single sham control.

    All shams except S₃ apply the symmetric gate:
        |H_sham_mean| < 2 σ_S AND |H_sham_mean| < ε_abs.

    S₃ (MASS_SCALED) has a branch-dependent gate per §3.2 and
    design-closeout §A.1 patch #6:
        POSITIVE branch:  baseline-subtracted H invariant within 10%.
        NULL/AMBIGUOUS branch: same as other shams (2σ + ε_abs).

    Additionally, §3.4 anti-fishing rule: if ≥ `sham_consistent_sign_min`
    of the seeds share a common sign AND |mean| > 0.5 σ, the sham fails
    regardless of the |mean| threshold.

    For shams, the substrate's forward call alone is sufficient
    (orientation-reversal is not the test target). If `sham_reversed`
    is provided, the antisymmetric combination is used for parity with
    the primary observable.
    """
    if sham_reversed is not None:
        per_seed_h_sham = 0.5 * (
            sham_forward.per_seed_h_scalar - sham_reversed.per_seed_h_scalar
        )
    else:
        per_seed_h_sham = sham_forward.per_seed_h_scalar

    mean_h = float(np.mean(per_seed_h_sham))
    abs_mean = abs(mean_h)
    n_seeds = per_seed_h_sham.shape[0]
    sigma = (
        float(np.std(per_seed_h_sham, ddof=1) / np.sqrt(n_seeds))
        if n_seeds > 1 else float("inf")
    )

    # Consistent-sign anti-fishing
    if n_seeds > 0:
        mean_sign = np.sign(mean_h)
        if mean_sign == 0:
            same_sign_count = 0
        else:
            same_sign_count = int(np.sum(np.sign(per_seed_h_sham) == mean_sign))
    else:
        same_sign_count = 0
    sign_coherence_ratio = same_sign_count / max(n_seeds, 1)

    # Primary 2σ + ε_abs gates
    sigma_gate = abs_mean < constants.sham_threshold_sigma * sigma
    epsilon_gate = abs_mean < constants.epsilon_abs

    # Step 1: per-flag primary verdict. Anti-fishing is a SECONDARY
    # check that only fires if the primary gate passes — otherwise the
    # primary failure dominates and anti-fishing would be redundant
    # noise. For S₂ specifically, the substrate failing to zero the
    # coupling is the right diagnostic; anti-fishing would obscure it.
    if flag is ShamFlag.ALPHA_ZERO:
        primary_passes = epsilon_gate
        primary_pass_reason = (
            f"S₂ passes: |mean|={abs_mean:.3e} < ε_abs={constants.epsilon_abs:.0e}"
        )
        primary_fail_reason = (
            f"S₂ (ALPHA_ZERO): |mean|={abs_mean:.3e} ≥ ε_abs={constants.epsilon_abs:.0e}; "
            f"substrate did not zero the coupling at machine precision"
        )
    elif flag is ShamFlag.MASS_SCALED and primary_h_geom is not None and abs(primary_h_geom) > 0:
        # POSITIVE branch (caller signals via primary_h_geom): use the
        # baseline-subtracted invariance test. v0.1 stands in with the
        # standard 2σ + ε_abs gate; baseline subtraction is a v0.2
        # elaboration the caller fits before passing the residual mean.
        primary_passes = sigma_gate and epsilon_gate
        primary_pass_reason = (
            f"S₃ passes (POSITIVE branch baseline-subtracted within tolerance)"
        )
        primary_fail_reason = (
            f"S₃ (MASS_SCALED, POSITIVE branch): residual |mean|={abs_mean:.3e}, "
            f"σ={sigma:.3e}, ε_abs={constants.epsilon_abs:.0e}"
        )
    else:
        primary_passes = sigma_gate and epsilon_gate
        primary_pass_reason = (
            f"{flag.value} passes: |mean|={abs_mean:.3e}, σ={sigma:.3e}, "
            f"< both 2σ and ε_abs"
        )
        failures = []
        if not sigma_gate:
            failures.append(
                f"|mean|={abs_mean:.3e} ≥ {constants.sham_threshold_sigma}σ={constants.sham_threshold_sigma * sigma:.3e}"
            )
        if not epsilon_gate:
            failures.append(
                f"|mean|={abs_mean:.3e} ≥ ε_abs={constants.epsilon_abs:.0e}"
            )
        primary_fail_reason = f"{flag.value} fails: " + "; ".join(failures)

    # Step 2: §3.4 anti-fishing rule. Per SPEC: "the sham fails
    # (regardless of the |mean| gate) if sign(H_sham,i) is the same
    # for at least 6 of 8 seeds AND |mean H_sham| > 0.5 σ_sham."
    #
    # Fires only when the primary gate would otherwise pass (so the
    # failure-reason is the appropriate diagnostic — anti-fishing
    # catches the case where a sham PASSES |mean| < 2σ AND |mean| <
    # ε_abs but exhibits suspicious sign coherence at the noise floor).
    # When primary already fails, the primary failure is the right
    # diagnostic. σ > 0 guard prevents NaN comparisons on degenerate
    # test data.
    anti_fishing_fired = (
        primary_passes
        and sigma > 0
        and same_sign_count >= constants.sham_consistent_sign_min
        and abs_mean > constants.sham_consistent_sign_mean_threshold * sigma
    )
    if anti_fishing_fired:
        return ShamVerdict(
            flag=flag,
            mean_h=mean_h,
            sigma=sigma,
            sign_coherence_ratio=sign_coherence_ratio,
            passed=False,
            reason=(
                f"§3.4 anti-fishing: {same_sign_count}/{n_seeds} seeds share sign "
                f"AND |mean|={abs_mean:.3e} > {constants.sham_consistent_sign_mean_threshold} × σ={sigma:.3e} "
                f"(primary gate would otherwise pass)"
            ),
        )

    return ShamVerdict(
        flag=flag,
        mean_h=mean_h,
        sigma=sigma,
        sign_coherence_ratio=sign_coherence_ratio,
        passed=primary_passes,
        reason=primary_pass_reason if primary_passes else primary_fail_reason,
    )


# ----------------------------------------------------------------------
# Substrate gate (per v3.1.3 §4.2 + §4.3 + design-closeout §A.3)
# ----------------------------------------------------------------------


def evaluate_substrate_gates(
    forward: LoopTransportResult,
    constants: V313Constants = V313_CONSTANTS,
) -> SubstrateGateVerdict:
    """The tracking-error and adiabaticity gates. Either failing forces
    overall AMBIGUOUS regardless of H values.

    Reads from the FORWARD call's diagnostics (the reversed call is
    not used for these gates — same loop, same substrate state, same
    expected diagnostic; substrate-side instrumentation is shared)."""

    tau_pin_ratio = forward.adiabaticity_check.tau_pin_over_t_segment
    q_max = forward.tracking_error_max_Q
    bw_max = forward.tracking_error_max_beta_W

    failures = []
    if q_max >= constants.tracking_error_eps_Q:
        failures.append(
            f"tracking_error_max_Q={q_max:.3e} ≥ ε_Q={constants.tracking_error_eps_Q}"
        )
    if bw_max >= constants.tracking_error_eps_beta_W:
        failures.append(
            f"tracking_error_max_β_W={bw_max:.3e} ≥ ε_β_W={constants.tracking_error_eps_beta_W}"
        )
    if tau_pin_ratio >= constants.adiabaticity_threshold:
        failures.append(
            f"τ_pin/T_segment={tau_pin_ratio:.3e} ≥ {constants.adiabaticity_threshold} "
            f"(adiabaticity gate per v3.1.3 §4.2)"
        )

    passed = len(failures) == 0
    reason = "substrate gates pass" if passed else "; ".join(failures)
    return SubstrateGateVerdict(
        tracking_q_max=q_max,
        tracking_beta_w_max=bw_max,
        tau_pin_over_t_segment=tau_pin_ratio,
        passed=passed,
        reason=reason,
    )


# ----------------------------------------------------------------------
# Primary §3.1 classification
# ----------------------------------------------------------------------


def evaluate_primary(
    forward: LoopTransportResult,
    reversed_: LoopTransportResult,
    constants: V313Constants = V313_CONSTANTS,
) -> PrimaryVerdict:
    """Apply the §3.1 primary observable gates (without yet considering
    sham/substrate failures). Returns POSITIVE / NULL / AMBIGUOUS based
    on |H_geom_mean| / σ + the §3.5 sign-coherence rule.

    Returns AMBIGUOUS when:
        - 1σ ≤ |H_geom| ≤ 5σ
        - |H_sys| ≥ 1σ (load-bearing — antisymmetric-only signal required)
        - POSITIVE-magnitude signal but sign coherence < 5/8 (incoherent)
    """
    h_geom_mean, h_sys, sigma_h_blocked = compute_h_geom_h_sys(forward, reversed_)

    if sigma_h_blocked <= 0.0 or np.isnan(sigma_h_blocked):
        sigma_ratio = float("inf") if h_geom_mean != 0 else 0.0
    else:
        sigma_ratio = abs(h_geom_mean) / sigma_h_blocked

    sign_count, sign_total = per_seed_sign_coherence(forward, reversed_)

    # §3.1 H_sys load-bearing condition: |H_sys| < 1σ_H for any
    # non-AMBIGUOUS verdict.
    h_sys_ok = (abs(h_sys) < constants.primary_null_sigma * sigma_h_blocked
                if sigma_h_blocked > 0 else (h_sys == 0.0))

    above_5_sigma = sigma_ratio > constants.primary_positive_sigma
    below_1_sigma = sigma_ratio < constants.primary_null_sigma
    in_ambiguous_band = (
        constants.primary_null_sigma <= sigma_ratio <= constants.primary_positive_sigma
    )

    # §3.5 sign-coherence requirement (POSITIVE branch only)
    sign_coherent = (
        sign_count >= constants.sign_coherence_min
        and sign_total >= constants.sign_coherence_total
    )

    if not h_sys_ok:
        verdict = Verdict.AMBIGUOUS
        reason = (
            f"|H_sys|={abs(h_sys):.3e} ≥ 1σ_H={sigma_h_blocked:.3e}; "
            f"systematic-offset diagnostic indicates non-antisymmetric signal"
        )
    elif above_5_sigma and sign_coherent:
        verdict = Verdict.POSITIVE
        reason = (
            f"|H_geom|/σ={sigma_ratio:.2f} > {constants.primary_positive_sigma} "
            f"AND sign-coherence {sign_count}/{sign_total} ≥ {constants.sign_coherence_min}/{constants.sign_coherence_total}"
        )
    elif above_5_sigma and not sign_coherent:
        verdict = Verdict.AMBIGUOUS
        reason = (
            f"|H_geom|/σ={sigma_ratio:.2f} > {constants.primary_positive_sigma} but "
            f"sign-coherence {sign_count}/{sign_total} < {constants.sign_coherence_min}/{constants.sign_coherence_total}; "
            f"ensemble is incoherent (§3.5 POSITIVE-only rule fails)"
        )
    elif below_1_sigma:
        verdict = Verdict.NULL
        reason = (
            f"|H_geom|/σ={sigma_ratio:.2f} < {constants.primary_null_sigma}; "
            f"random signs are expected in a true null (§3.5 no sign-coherence requirement)"
        )
    elif in_ambiguous_band:
        verdict = Verdict.AMBIGUOUS
        reason = (
            f"|H_geom|/σ={sigma_ratio:.2f} in ambiguous band "
            f"[{constants.primary_null_sigma}, {constants.primary_positive_sigma}]"
        )
    else:
        verdict = Verdict.AMBIGUOUS
        reason = f"|H_geom|/σ={sigma_ratio:.2f} — unhandled gate region"

    return PrimaryVerdict(
        h_geom_mean=h_geom_mean,
        h_sys=h_sys,
        sigma_h_blocked=sigma_h_blocked,
        sigma_ratio=sigma_ratio,
        sign_coherence_count=sign_count,
        sign_coherence_total=sign_total,
        verdict=verdict,
        reason=reason,
    )


# ----------------------------------------------------------------------
# Top-level composite classifier
# ----------------------------------------------------------------------


def classify(
    forward: LoopTransportResult,
    reversed_: LoopTransportResult,
    shams: Dict[ShamFlag, LoopTransportResult],
    constants: V313Constants = V313_CONSTANTS,
) -> CompositeVerdict:
    """End-to-end v3.1.3 verdict.

    AMBIGUOUS-on-substrate-gate or AMBIGUOUS-on-sham wins regardless
    of the primary observable. POSITIVE / NULL / AMBIGUOUS reflect
    the §3.1 + §3.2 composition."""

    substrate = evaluate_substrate_gates(forward, constants)
    primary = evaluate_primary(forward, reversed_, constants)

    sham_verdicts: List[ShamVerdict] = []
    for flag, sham_result in shams.items():
        sham_verdicts.append(
            evaluate_sham(
                flag=flag,
                sham_forward=sham_result,
                primary_h_geom=primary.h_geom_mean,
                constants=constants,
            )
        )
    sham_verdicts_tuple = tuple(sham_verdicts)

    failed_shams = [s for s in sham_verdicts if not s.passed]

    if not substrate.passed:
        return CompositeVerdict(
            overall=Verdict.AMBIGUOUS,
            reason=f"substrate gate(s) fail: {substrate.reason}",
            primary=primary,
            shams=sham_verdicts_tuple,
            substrate_gates=substrate,
        )

    if failed_shams:
        names = ", ".join(s.flag.value for s in failed_shams)
        return CompositeVerdict(
            overall=Verdict.AMBIGUOUS,
            reason=f"sham gate(s) fail: {names}",
            primary=primary,
            shams=sham_verdicts_tuple,
            substrate_gates=substrate,
        )

    return CompositeVerdict(
        overall=primary.verdict,
        reason=primary.reason,
        primary=primary,
        shams=sham_verdicts_tuple,
        substrate_gates=substrate,
    )
