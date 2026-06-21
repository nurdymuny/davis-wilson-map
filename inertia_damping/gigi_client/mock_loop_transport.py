"""Mock LoopTransportClient — scenario-driven plausible data for
testing the Halcyon orchestrator before the GIGI substrate verb lands.

The mock implements the `LoopTransportClient` Protocol exactly. Each
constructor `scenario` exercises a different verdict path:

    - "primary_positive"           — H_geom large, σ small, signs coherent
    - "primary_null"               — H_geom small, σ small (random signs)
    - "primary_ambiguous_sigma"    — |H_geom| / σ_H in the 1σ–5σ range
    - "primary_ambiguous_signs"    — |H_geom| > 5σ but only 4/8 same sign
    - "sham_failure_<FLAG>"        — that specific sham above threshold
    - "tracking_error_violation"   — substrate reports max > ε on Q
    - "adiabaticity_violation"     — substrate reports τ_pin/T_segment ≥ 0.1

The mock is seeded-reproducible (uses a deterministic NumPy generator
keyed on (scenario, request)) so test failures are debuggable.

The mock does NOT actually compute anything physical. It returns shaped
noise + a controlled mean to drive the orchestrator's gate paths.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np

from inertia_damping.gigi_client.loop_transport import (
    AdiabaticityCheck,
    LoopHandle,
    LoopTransportRequest,
    LoopTransportResult,
    ShamFlag,
)


# ----------------------------------------------------------------------
# Scenario knobs (test-side, not part of the substrate contract)
# ----------------------------------------------------------------------


_SCENARIOS = {
    "primary_positive",
    "primary_null",
    "primary_ambiguous_sigma",
    "primary_ambiguous_signs",
    "tracking_error_violation",
    "adiabaticity_violation",
}
# Plus dynamic "sham_failure_<FLAG>" for each ShamFlag value.


def _deterministic_rng(scenario: str, request: LoopTransportRequest) -> np.random.Generator:
    """Build an RNG keyed on the (scenario, request) tuple so test runs
    are reproducible AND each request gets a different draw within a
    scenario. NOT the substrate's real seed-propagation pattern — the
    real client uses request.seeds via xorshift/PCG; the mock just
    needs reproducible scenario noise."""
    key_parts = (
        scenario.encode("utf-8"),
        request.loop.name.encode("utf-8"),
        request.direction.encode("utf-8"),
        request.sham.value.encode("utf-8"),
        str(request.pack.alpha).encode("utf-8"),
        str(request.sham_mass_scale or 0.0).encode("utf-8"),
    )
    seed_int = abs(hash(key_parts)) % (2**32 - 1)
    return np.random.default_rng(seed_int)


def _identity_quaternion(n: int) -> np.ndarray:
    """SU(2) identity broadcast to (n, 4)."""
    out = np.zeros((n, 4), dtype=np.float64)
    out[:, 0] = 1.0
    return out


class MockLoopTransportClient:
    """Scenario-driven mock. Satisfies `LoopTransportClient` Protocol."""

    def __init__(self, scenario: str = "primary_null"):
        if not (scenario in _SCENARIOS or scenario.startswith("sham_failure_")):
            raise ValueError(
                f"Unknown scenario {scenario!r}; expected one of {_SCENARIOS} "
                f"or 'sham_failure_<FLAG>'."
            )
        self.scenario = scenario
        self._declared_loops: Dict[str, LoopHandle] = {}

    # ------------------------------------------------------------------
    # LoopTransportClient Protocol
    # ------------------------------------------------------------------

    def declare_loop(self, loop: LoopHandle) -> str:
        existing = self._declared_loops.get(loop.name)
        if existing is not None and existing != loop:
            raise ValueError(
                f"Loop {loop.name!r} already declared with different shape; "
                f"redeclaration requires a different name"
            )
        self._declared_loops[loop.name] = loop
        return loop.name

    def loop_transport(self, request: LoopTransportRequest) -> LoopTransportResult:
        if request.loop.name not in self._declared_loops:
            raise ValueError(
                f"Loop {request.loop.name!r} not declared; call declare_loop first"
            )

        rng = _deterministic_rng(self.scenario, request)
        n_seeds = len(request.seeds)

        # Decide mean H scalar + per-seed σ + coherence based on
        # scenario + sham + direction + (for primary) which scenario.
        if self.scenario.startswith("sham_failure_"):
            flag_name = self.scenario.removeprefix("sham_failure_")
            forced_flag = ShamFlag(flag_name) if flag_name in ShamFlag.__members__ else None
            mean_h = self._sham_failure_mean(request, forced_flag)
            sigma_per_seed = 1e-12
        elif request.sham is not ShamFlag.NONE:
            # Sham requested AND scenario is not sham_failure_* —
            # return clean sham. Centered draw (mean - sample_mean) so
            # |sample_mean| ≈ 0 and the §3.4 anti-fishing rule cannot
            # mis-fire on coincidentally coherent sign patterns in
            # finite-N noise. Real substrate noise has σ > 0 across
            # seeds; centering is the mock idealization of "the
            # substrate happened to draw a statistically clean null."
            sigma_per_seed = 1e-13
            raw = rng.normal(loc=0.0, scale=sigma_per_seed, size=n_seeds)
            centered = raw - np.mean(raw)
            per_seed_holonomy = _identity_quaternion(n_seeds)
            per_seed_holonomy[:, 1] = centered
            sigma_h_blocked = float(np.std(centered, ddof=1) / np.sqrt(n_seeds))
            return LoopTransportResult(
                per_seed_holonomy=per_seed_holonomy,
                per_seed_h_scalar=centered,
                sigma_h_blocked=sigma_h_blocked,
                tracking_error_max_Q=0.005,
                tracking_error_max_beta_W=0.005,
                adiabaticity_check=AdiabaticityCheck(
                    tau_pin_over_t_segment=0.02,
                    ramp_rate_over_relaxation_rate=0.04,
                    warnings_count=0,
                    warning_substep_indices=(),
                ),
                run_id=f"mock-{self.scenario}-{request.loop.name}-{request.direction}",
            )
        elif self.scenario == "primary_positive":
            sigma_per_seed = 1e-5
            mean_h = self._directional_signed(request, magnitude=7e-5)
        elif self.scenario == "primary_ambiguous_sigma":
            sigma_per_seed = 1e-5
            mean_h = self._directional_signed(request, magnitude=2e-5)  # ~2σ_blocked
        elif self.scenario == "primary_ambiguous_signs":
            # |H_geom| > 5σ but signs incoherent (only 4/8 same sign)
            sigma_per_seed = 1e-5
            mean_h = self._directional_signed(request, magnitude=7e-5)
            return self._with_incoherent_signs(rng, request, mean_h, sigma_per_seed)
        elif self.scenario == "primary_null":
            # Mock idealization: center the per-seed draw to force
            # |sample_mean| ≈ 0 so the test reliably hits the NULL band.
            # With 8 iid Gaussian samples the un-centered |t| statistic
            # lands in [0, 1] only ~64% of the time; centering is the
            # mock-side equivalent of "the substrate happened to draw a
            # statistically clean null", giving the orchestrator a
            # reproducible NULL path to test against.
            sigma_per_seed = 1e-5
            raw = rng.normal(loc=0.0, scale=sigma_per_seed, size=n_seeds)
            centered = raw - np.mean(raw)
            per_seed_holonomy = _identity_quaternion(n_seeds)
            per_seed_holonomy[:, 1] = centered
            sigma_h_blocked = float(np.std(centered, ddof=1) / np.sqrt(n_seeds))
            return LoopTransportResult(
                per_seed_holonomy=per_seed_holonomy,
                per_seed_h_scalar=centered,
                sigma_h_blocked=sigma_h_blocked,
                tracking_error_max_Q=0.005,
                tracking_error_max_beta_W=0.005,
                adiabaticity_check=AdiabaticityCheck(
                    tau_pin_over_t_segment=0.02,
                    ramp_rate_over_relaxation_rate=0.04,
                    warnings_count=0,
                    warning_substep_indices=(),
                ),
                run_id=f"mock-{self.scenario}-{request.loop.name}-{request.direction}",
            )
        elif self.scenario == "tracking_error_violation":
            sigma_per_seed = 1e-5
            mean_h = 0.0
            return self._with_tracking_violation(rng, request, mean_h, sigma_per_seed)
        elif self.scenario == "adiabaticity_violation":
            sigma_per_seed = 1e-5
            mean_h = 0.0
            return self._with_adiabaticity_violation(rng, request, mean_h, sigma_per_seed)
        else:
            # Should be unreachable due to constructor validation
            raise RuntimeError(f"Unhandled scenario {self.scenario!r}")

        per_seed_h_scalar = rng.normal(loc=mean_h, scale=sigma_per_seed, size=n_seeds)
        per_seed_holonomy = _identity_quaternion(n_seeds)  # cosmetic for mock
        per_seed_holonomy[:, 1] = per_seed_h_scalar         # park scalar in q_x slot
        sigma_h_blocked = float(np.std(per_seed_h_scalar, ddof=1) / np.sqrt(n_seeds))

        return LoopTransportResult(
            per_seed_holonomy=per_seed_holonomy,
            per_seed_h_scalar=per_seed_h_scalar,
            sigma_h_blocked=sigma_h_blocked,
            tracking_error_max_Q=0.005,
            tracking_error_max_beta_W=0.005,
            adiabaticity_check=AdiabaticityCheck(
                tau_pin_over_t_segment=0.02,
                ramp_rate_over_relaxation_rate=0.04,
                warnings_count=0,
                warning_substep_indices=(),
            ),
            run_id=f"mock-{self.scenario}-{request.loop.name}-{request.direction}",
        )

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _directional_signed(request: LoopTransportRequest, magnitude: float) -> float:
        """Forward/reverse traversal produces opposite-sign means so the
        orchestrator's H_geom = ½(H_fwd - H_rev) sees the full
        magnitude as the antisymmetric primary observable."""
        sign = +1.0 if request.direction == "FORWARD" else -1.0
        return sign * magnitude

    def _with_incoherent_signs(
        self,
        rng: np.random.Generator,
        request: LoopTransportRequest,
        mean_h: float,
        sigma_per_seed: float,
    ) -> LoopTransportResult:
        n_seeds = len(request.seeds)
        per_seed_h_scalar = rng.normal(loc=mean_h, scale=sigma_per_seed, size=n_seeds)
        # Force exactly 4/8 to flip sign by overriding the first half
        target_sign = np.sign(mean_h) if mean_h != 0 else 1.0
        per_seed_h_scalar[: n_seeds // 2] *= -target_sign / max(np.sign(per_seed_h_scalar[: n_seeds // 2]).mean(), 1e-12)
        per_seed_holonomy = _identity_quaternion(n_seeds)
        per_seed_holonomy[:, 1] = per_seed_h_scalar
        sigma_h_blocked = float(np.std(per_seed_h_scalar, ddof=1) / np.sqrt(n_seeds))
        return LoopTransportResult(
            per_seed_holonomy=per_seed_holonomy,
            per_seed_h_scalar=per_seed_h_scalar,
            sigma_h_blocked=sigma_h_blocked,
            tracking_error_max_Q=0.005,
            tracking_error_max_beta_W=0.005,
            adiabaticity_check=AdiabaticityCheck(
                tau_pin_over_t_segment=0.02,
                ramp_rate_over_relaxation_rate=0.04,
            ),
            run_id=f"mock-{self.scenario}-{request.loop.name}-{request.direction}",
        )

    def _with_tracking_violation(
        self,
        rng: np.random.Generator,
        request: LoopTransportRequest,
        mean_h: float,
        sigma_per_seed: float,
    ) -> LoopTransportResult:
        n_seeds = len(request.seeds)
        per_seed_h_scalar = rng.normal(loc=mean_h, scale=sigma_per_seed, size=n_seeds)
        per_seed_holonomy = _identity_quaternion(n_seeds)
        per_seed_holonomy[:, 1] = per_seed_h_scalar
        sigma_h_blocked = float(np.std(per_seed_h_scalar, ddof=1) / np.sqrt(n_seeds))
        return LoopTransportResult(
            per_seed_holonomy=per_seed_holonomy,
            per_seed_h_scalar=per_seed_h_scalar,
            sigma_h_blocked=sigma_h_blocked,
            tracking_error_max_Q=0.08,    # > ε_Q = 0.05
            tracking_error_max_beta_W=0.01,
            adiabaticity_check=AdiabaticityCheck(
                tau_pin_over_t_segment=0.02,
                ramp_rate_over_relaxation_rate=0.04,
            ),
            run_id=f"mock-{self.scenario}-{request.loop.name}-{request.direction}",
        )

    def _with_adiabaticity_violation(
        self,
        rng: np.random.Generator,
        request: LoopTransportRequest,
        mean_h: float,
        sigma_per_seed: float,
    ) -> LoopTransportResult:
        n_seeds = len(request.seeds)
        per_seed_h_scalar = rng.normal(loc=mean_h, scale=sigma_per_seed, size=n_seeds)
        per_seed_holonomy = _identity_quaternion(n_seeds)
        per_seed_holonomy[:, 1] = per_seed_h_scalar
        sigma_h_blocked = float(np.std(per_seed_h_scalar, ddof=1) / np.sqrt(n_seeds))
        return LoopTransportResult(
            per_seed_holonomy=per_seed_holonomy,
            per_seed_h_scalar=per_seed_h_scalar,
            sigma_h_blocked=sigma_h_blocked,
            tracking_error_max_Q=0.005,
            tracking_error_max_beta_W=0.005,
            adiabaticity_check=AdiabaticityCheck(
                tau_pin_over_t_segment=0.25,   # >> 0.1 → AMBIGUOUS
                ramp_rate_over_relaxation_rate=0.04,
            ),
            run_id=f"mock-{self.scenario}-{request.loop.name}-{request.direction}",
        )

    @staticmethod
    def _sham_failure_mean(
        request: LoopTransportRequest,
        forced_flag: Optional[ShamFlag],
    ) -> float:
        """If this request matches the failure-target flag, return a
        sham value above ε_abs that will trip the gate. Otherwise
        return a clean sham."""
        if forced_flag is None or request.sham is not forced_flag:
            return 0.0
        # Trip the gate by ~3× ε_abs:
        return 3e-10
