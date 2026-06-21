"""The `section_12_holonomy_battery_v3_1_3` sidecar receipt per v3.1.3 §7.2.

Schema (locked at SPEC commit 44c70b1 / Zenodo DOI
10.5281/zenodo.20785681):

    {
      "schema_version": "section_12_holonomy_battery_v3_1_3",
      "spec_commit": "44c70b1...",
      "spec_doi": "10.5281/zenodo.20785681",
      "gigi_deploy_hash": "<sha>",
      "run_timestamp_utc": "2026-06-21T08:00:00Z",
      "alpha_halcyon": 1.0,
      "loop_name": "gamma_unit",
      "seeds": [20260616, ..., 20260623],
      "per_seed_forward": [[q0, q1, q2, q3], ...],
      "per_seed_forward_scalar": [float, ...],
      "per_seed_reversed": [[q0, q1, q2, q3], ...],
      "per_seed_reversed_scalar": [float, ...],
      "h_geom_mean": float,
      "h_sys": float,
      "sigma_h_blocked": float,
      "sigma_ratio": float,
      "sign_coherence_count": int,
      "sign_coherence_total": int,
      "shams": {
        "FLAT_FIELD": { "mean_h": float, "sigma": float, "passed": bool, "reason": str, ... },
        ...
      },
      "substrate_gates": {
        "tracking_q_max": float,
        "tracking_beta_w_max": float,
        "tau_pin_over_t_segment": float,
        "ramp_rate_over_relaxation_rate": float,
        "passed": bool,
        "reason": str
      },
      "verdict": "POSITIVE" | "NULL" | "AMBIGUOUS",
      "reason": str
    }

The sidecar is the load-bearing artefact for downstream reproducibility:
anyone with this SPEC's commit hash, the GIGI deploy hash, and the
seed list can re-run the experiment and verify the sidecar (per
v3.1.3 §7.2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inertia_damping.gigi_client.loop_transport import (
    LoopTransportResult,
    ShamFlag,
)
from inertia_damping.holonomy_battery.gates import CompositeVerdict


SCHEMA_VERSION = "section_12_holonomy_battery_v3_1_3"
SPEC_COMMIT = "44c70b1b76501b4b66c6f9ace6bccd8b5bd14c4a"
SPEC_DOI = "10.5281/zenodo.20785681"


@dataclass(frozen=True)
class Section12Sidecar:
    """In-memory representation of one calibration's sidecar."""

    schema_version: str = SCHEMA_VERSION
    spec_commit: str = SPEC_COMMIT
    spec_doi: str = SPEC_DOI
    gigi_deploy_hash: str = "unknown"
    run_timestamp_utc: str = ""
    alpha_halcyon: float = 1.0
    loop_name: str = "gamma_unit"
    seeds: Tuple[int, ...] = ()
    per_seed_forward: List[List[float]] = field(default_factory=list)
    per_seed_forward_scalar: List[float] = field(default_factory=list)
    per_seed_reversed: List[List[float]] = field(default_factory=list)
    per_seed_reversed_scalar: List[float] = field(default_factory=list)
    h_geom_mean: float = 0.0
    h_sys: float = 0.0
    sigma_h_blocked: float = 0.0
    sigma_ratio: float = 0.0
    sign_coherence_count: int = 0
    sign_coherence_total: int = 0
    shams: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    substrate_gates: Dict[str, Any] = field(default_factory=dict)
    verdict: str = "AMBIGUOUS"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy types to plain python for JSON serialization
        d["seeds"] = list(self.seeds)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


def build_sidecar(
    *,
    composite: CompositeVerdict,
    forward: LoopTransportResult,
    reversed_: LoopTransportResult,
    sham_results: Dict[ShamFlag, LoopTransportResult],
    alpha_halcyon: float,
    loop_name: str,
    seeds: Tuple[int, ...],
    gigi_deploy_hash: str = "unknown",
    run_timestamp_utc: str = "",
) -> Section12Sidecar:
    """Construct the v3.1.3 §7.2 sidecar from one calibration's results."""
    shams_dict: Dict[str, Dict[str, Any]] = {}
    sham_verdicts_by_flag = {sv.flag: sv for sv in composite.shams}
    for flag, sham_result in sham_results.items():
        sv = sham_verdicts_by_flag.get(flag)
        shams_dict[flag.value] = {
            "mean_h": sv.mean_h if sv else float("nan"),
            "sigma": sv.sigma if sv else float("nan"),
            "sign_coherence_ratio": sv.sign_coherence_ratio if sv else float("nan"),
            "passed": sv.passed if sv else False,
            "reason": sv.reason if sv else "no verdict computed",
            "per_seed_h_scalar": [float(x) for x in sham_result.per_seed_h_scalar],
            "run_id": sham_result.run_id,
        }

    substrate_gates_dict = {
        "tracking_q_max": composite.substrate_gates.tracking_q_max,
        "tracking_beta_w_max": composite.substrate_gates.tracking_beta_w_max,
        "tau_pin_over_t_segment": composite.substrate_gates.tau_pin_over_t_segment,
        "ramp_rate_over_relaxation_rate": forward.adiabaticity_check.ramp_rate_over_relaxation_rate,
        "passed": composite.substrate_gates.passed,
        "reason": composite.substrate_gates.reason,
    }

    return Section12Sidecar(
        schema_version=SCHEMA_VERSION,
        spec_commit=SPEC_COMMIT,
        spec_doi=SPEC_DOI,
        gigi_deploy_hash=gigi_deploy_hash,
        run_timestamp_utc=run_timestamp_utc,
        alpha_halcyon=alpha_halcyon,
        loop_name=loop_name,
        seeds=tuple(seeds),
        per_seed_forward=[list(row) for row in forward.per_seed_holonomy.tolist()],
        per_seed_forward_scalar=[float(x) for x in forward.per_seed_h_scalar],
        per_seed_reversed=[list(row) for row in reversed_.per_seed_holonomy.tolist()],
        per_seed_reversed_scalar=[float(x) for x in reversed_.per_seed_h_scalar],
        h_geom_mean=composite.primary.h_geom_mean,
        h_sys=composite.primary.h_sys,
        sigma_h_blocked=composite.primary.sigma_h_blocked,
        sigma_ratio=composite.primary.sigma_ratio,
        sign_coherence_count=composite.primary.sign_coherence_count,
        sign_coherence_total=composite.primary.sign_coherence_total,
        shams=shams_dict,
        substrate_gates=substrate_gates_dict,
        verdict=composite.overall.value,
        reason=composite.reason,
    )


def write_sidecar(sidecar: Section12Sidecar, output_path: Path) -> None:
    """Write the sidecar to disk as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(sidecar.to_json(), encoding="utf-8")
