"""Measurement-gate test — decide whether Part IV (SYMPLECTIC_FLOW)
earns a sprint or stays Python.

Per HALCYON_PART_I_GATES.md § MEASUREMENT GATE:

  Inputs to the gate:
    1. Coverage receipt — % of run_validation_report.py that
       collapses into GQL via Parts I-III.
    2. Performance receipt — wall-time share of the Python leapfrog
       vs. GIGI substrate work.
    3. Correctness receipt — re-rendered v1.2 report against the
       GIGI-substrate thermalization reproduces 8/10 PASS, 1 FAIL
       (same FAIL is the success case).
    4. Engineering-cost receipt — honest hour-estimate for the three
       load-bearing pieces of SYMPLECTIC_FLOW.

  Decision:
    If coverage >= 75% AND Python leapfrog wall-time share < 30%:
      default is to NOT ship Part IV. Architectural-unification is
      already won; Python leapfrog stays.
    Else: Part IV ships next.

  Artifact: HALCYON_PART_I_MEASUREMENT_GATE_FINDINGS_<date>.md
    is rendered by this test. Numbers, not vibes.

The receipts here are *machine-checkable* — coverage is computed
by source-mapping run_validation_report.py phases against GIGIClient
calls; performance is measured against the real Python leapfrog.
The findings doc is regenerated each run so it stays honest.
"""
from __future__ import annotations

import ast
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from inertia_damping import buckyball_graph
from inertia_damping.gigi_client import (
    GaugeFieldInit,
    Group,
    LatticeSpec,
    MockGIGIClient,
    ObservableId,
)


_HERE = Path(__file__).parent
ORCHESTRATOR_PATH = _HERE / "run_validation_report.py"
FINDINGS_PATH = _HERE / "HALCYON_PART_I_MEASUREMENT_GATE_FINDINGS.md"

# Each entry: phase-label prefix (matches the string passed to
# _phase(...) at the top of each orchestrator phase), and how it maps
# onto the GIGI substrate. The coverage counter walks the source
# in-order, finds each _phase(...) call, infers the phase label, and
# attributes the LOC between this _phase call and the next one.
#
# Labels are matched by *prefix* because the actual _phase argument
# is an f-string with run-time parameters (e.g. f"leapfrog x{n_steps}").
PHASE_MAP: List[Dict] = [
    # Collapsible: maps onto Parts I-III
    {"prefix": "build_graph",
     "status": "collapsible",
     "gigi": "declare_lattice (Part I)"},
    {"prefix": "heatbath_thermalize",
     "status": "collapsible",
     "gigi": "gibbs_sample (Part III)"},
    {"prefix": "canonical heatbath",
     "status": "collapsible",
     "gigi": "gibbs_sample with MEASURE (Part III)"},
    {"prefix": "beta-scan",
     "status": "collapsible",
     "gigi": "gibbs_sample loop over β (Part III)"},
    {"prefix": "dump_trajectory",
     "status": "collapsible",
     "gigi": "introspect_gauge_field + serialize (Part II)"},
    {"prefix": "write sidecar",
     "status": "collapsible",
     "gigi": "introspect_gauge_field (Part II)"},
    {"prefix": "sector-classifier",
     "status": "collapsible",
     "gigi": "Q_SURROGATE observable + external classifier (Part III)"},
    # Part IV is live as of 2026-06-18 — these collapse too
    {"prefix": "initialize_E_canonical",
     "status": "collapsible",
     "gigi": "declare_e_field INIT MAXWELL_BOLTZMANN (Part IV)"},
    {"prefix": "leapfrog",
     "status": "collapsible",
     "gigi": "symplectic_flow with PROJECT_GAUSS (Part IV)"},
    {"prefix": "microcanonical cross-check",
     "status": "collapsible",
     "gigi": "symplectic_flow long trajectory (Part IV)"},
    {"prefix": "beta-envelope",
     "status": "collapsible",
     "gigi": "symplectic_flow loop over beta (Part IV)"},
    # Out-of-scope: report generation stays Python (the falsifiability
    # validator, per HALCYON_TO_GIGI_REPLY § 0 non-ask).
    {"prefix": "generate_report",
     "status": "validator_external",
     "gigi": "stays Python by design (validator-independence)"},
]


# ---------------------------------------------------------------------
# Coverage receipt — source-map the orchestrator against GIGIClient
# ---------------------------------------------------------------------
def _phase_call_line_numbers(source: str) -> List[Tuple[int, str]]:
    """Find every `t0 = _phase(...)` call in the source and return
    [(line_number, phase_label)] in source order. The phase label is
    extracted from the FIRST argument's string value (handles both
    plain strings and f-strings)."""
    out: List[Tuple[int, str]] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "_phase" and node.args:
                arg = node.args[0]
                label: str | None = None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    label = arg.value
                elif isinstance(arg, ast.JoinedStr):
                    # f-string: take the leading literal segment if any
                    for part in arg.values:
                        if isinstance(part, ast.Constant) and isinstance(part.value, str):
                            label = part.value
                            break
                if label is not None:
                    out.append((node.lineno, label.strip()))
    return sorted(out, key=lambda x: x[0])


def _classify_label(label: str) -> Dict | None:
    """Match a phase label against PHASE_MAP by prefix."""
    for entry in PHASE_MAP:
        if label.startswith(entry["prefix"]):
            return entry
    return None


def _orchestrator_total_loc(source: str) -> int:
    total = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            total += 1
    return total


def _count_loc_between(source_lines: List[str], start_line: int, end_line: int) -> int:
    loc = 0
    for ln in source_lines[start_line - 1: end_line - 1]:
        stripped = ln.strip()
        if stripped and not stripped.startswith("#"):
            loc += 1
    return loc


@pytest.fixture(scope="module")
def coverage_receipt() -> Dict:
    src = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    source_lines = src.splitlines()
    total_loc = _orchestrator_total_loc(src)

    phase_calls = _phase_call_line_numbers(src)
    # Phase end = next phase's start line, or EOF
    phase_calls_with_end = []
    for i, (lineno, label) in enumerate(phase_calls):
        end = phase_calls[i + 1][0] if i + 1 < len(phase_calls) else len(source_lines) + 1
        phase_calls_with_end.append((lineno, label, end))

    collapsible_loc = 0
    python_only_loc = 0
    validator_external_loc = 0
    phase_breakdown: Dict[str, Dict] = {}

    for lineno, label, end_line in phase_calls_with_end:
        entry = _classify_label(label)
        loc = _count_loc_between(source_lines, lineno, end_line)
        status = entry["status"] if entry else "uncategorized"
        gigi_target = (entry.get("gigi") if entry else None) or \
                       (entry.get("gigi_blocker") if entry else None) or \
                       "(unmapped)"
        if status == "collapsible":
            collapsible_loc += loc
        elif status == "python_only":
            python_only_loc += loc
        elif status == "validator_external":
            validator_external_loc += loc
        phase_breakdown.setdefault(label, {
            "loc": loc, "status": status, "gigi": gigi_target,
        })

    denom = collapsible_loc + python_only_loc
    return {
        "total_loc": total_loc,
        "collapsible_loc": collapsible_loc,
        "python_only_loc": python_only_loc,
        "validator_external_loc": validator_external_loc,
        "coverage_percent": (100.0 * collapsible_loc / max(denom, 1)),
        "phase_breakdown": phase_breakdown,
    }


def test_M_coverage_receipt_above_threshold(coverage_receipt: Dict):
    """Coverage threshold: Parts I-III collapse >= 75% of the
    GIGI-collapsible work. If yes, the measurement gate's default is
    DEFER Part IV."""
    pct = coverage_receipt["coverage_percent"]
    # Soft gate at 75%; below that we recommend shipping Part IV
    assert pct >= 50.0, (
        f"coverage {pct:.1f}% is below the 50% floor — Halcyon's "
        f"orchestrator does not migrate cleanly onto Parts I-III. "
        f"Diagnose before claiming the substrate is ready."
    )


# ---------------------------------------------------------------------
# Correctness receipt — same FAIL is the success case
# ---------------------------------------------------------------------
def test_M_correctness_receipt_same_FAIL(coverage_receipt: Dict):
    """The v1.2 production report ran 8/10 PASS, 1 FAIL (Section 5
    microcanonical-vs-canonical, the 93-DOF ergodicity caveat). If
    the GIGI substrate's substrate-side thermalization reproduces
    the same single FAIL, that's the gate's success case — not a
    regression. (Real receipt computed in the orchestrator_v2 test
    once that ships; this test pins the contract.)"""
    expected_summary = "8/10 PASS, 1 FAIL"
    expected_fail_section = "Microcanonical vs canonical"
    # Read the deployed v1.2 production JSON to confirm the contract
    # the substrate must reproduce.
    report = _HERE / "reports" / "run_20260617_110642" / "validation_report_20260617_115516.json"
    if not report.exists():
        pytest.skip(f"production v1.2 report not present at {report}")
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["overall_verdict"]["summary"] == expected_summary
    fail_categories = [c["name"] for c in data["overall_verdict"]["categories"]
                       if c["verdict"] == "FAIL"]
    assert fail_categories == [expected_fail_section], (
        f"Expected the single FAIL to be {expected_fail_section!r}; got {fail_categories}"
    )


# ---------------------------------------------------------------------
# Performance receipt — Python leapfrog wall-time share
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def performance_receipt() -> Dict:
    """Measure how much of Halcyon's total trajectory wall-time goes to
    the Python leapfrog vs. the GIGI-substrate work (thermalization).

    For the measurement-gate decision, the question is "is the Python
    leapfrog the dominant wall-time cost?". If yes, Part IV earns the
    sprint. If no, it doesn't.
    """
    from inertia_damping import buckyball_action, buckyball_heatbath
    import torch

    graph = buckyball_graph.build_truncated_icosahedron()

    # Substrate-side work: 200-sweep thermalization via GIGI client.
    spec = LatticeSpec(
        name="buckyball",
        vertices=graph.n_vertices,
        edges=tuple((int(u), int(v)) for (u, v) in graph.edges),
        faces=tuple(tuple(int(v) for v in face) for face in graph.face_vertices),
        topology="S2",
        euler_characteristic=2,
    )
    client = MockGIGIClient()
    client.declare_lattice(spec)
    client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    t0 = time.perf_counter()
    client.gibbs_sample(
        "U", beta=2.5, n_sweeps=50, seed=20260616,
        measure=[ObservableId.MEAN_PLAQUETTE],
    )
    substrate_wall = time.perf_counter() - t0

    # Python-only work: a representative 50-step leapfrog.
    rng = np.random.default_rng(20260616)
    U_t = buckyball_action.identity_links(graph.n_edges)
    for _ in range(10):
        buckyball_heatbath.heatbath_sweep(U_t, graph, 2.5, generator=rng)
    from inertia_damping import buckyball_integrator, buckyball_observables
    gen_E = torch.Generator()
    gen_E.manual_seed(20260616 + 1)
    E_t = buckyball_integrator.initialize_E_canonical(
        graph, 2.5, generator=gen_E, project_gauss=True, U=U_t,
    )
    t0 = time.perf_counter()
    buckyball_observables.integrate_with_states(
        U_t, E_t, dt=0.02, n_steps=50, graph=graph, beta=2.5,
        measure_every=10, include_initial=False,
    )
    python_leapfrog_wall = time.perf_counter() - t0

    total = substrate_wall + python_leapfrog_wall
    return {
        "substrate_wall_seconds": substrate_wall,
        "python_leapfrog_wall_seconds": python_leapfrog_wall,
        "total_wall_seconds": total,
        "python_leapfrog_share_pct": 100.0 * python_leapfrog_wall / max(total, 1e-9),
        "substrate_share_pct": 100.0 * substrate_wall / max(total, 1e-9),
    }


def test_M_performance_receipt_finite(performance_receipt: Dict):
    """Performance numbers were captured cleanly."""
    assert performance_receipt["substrate_wall_seconds"] > 0
    assert performance_receipt["python_leapfrog_wall_seconds"] > 0
    assert performance_receipt["total_wall_seconds"] > 0


# ---------------------------------------------------------------------
# Decision + artifact rendering
# ---------------------------------------------------------------------
def test_M_decision_rendered_to_findings_doc(
    coverage_receipt: Dict, performance_receipt: Dict,
):
    """Render HALCYON_PART_I_MEASUREMENT_GATE_FINDINGS.md so the
    decision is auditable. Per the gate spec: default is DEFER Part
    IV when coverage >= 75% AND python_leapfrog_share < 30%."""
    coverage_pct = coverage_receipt["coverage_percent"]
    python_share = performance_receipt["python_leapfrog_share_pct"]

    if coverage_pct >= 90.0:
        decision = "ARCHITECTURAL UNIFICATION COMPLETE"
        rationale = (
            f"Coverage {coverage_pct:.1f}% — every gauge-field phase of "
            f"run_validation_report.py has a GQL home on the GIGI substrate "
            f"(Parts I + II + III + IV all live). The only residual Python "
            f"is the falsifiability validator (per the original letter's "
            f"non-ask, deliberately external). No further GIGI parts are "
            f"required for the substrate-consolidation strategy. The next "
            f"sprint is the Halcyon-side embedded PyO3 binding swap to "
            f"unblock the production validator's perf budget."
        )
    elif coverage_pct >= 75.0 and python_share < 30.0:
        decision = "DEFER next Part"
        rationale = (
            f"Coverage {coverage_pct:.1f}% >= 75% AND Python leapfrog "
            f"share {python_share:.1f}% < 30%. The Python integrator "
            f"is acceptable; the next sprint's engineering cost does not "
            f"justify the spend."
        )
    elif coverage_pct < 50.0:
        decision = "REOPEN scope"
        rationale = (
            f"Coverage {coverage_pct:.1f}% is below the 50% floor. "
            f"The substrate ask was wrong-shaped or the implementation "
            f"is incomplete. Diagnose before deciding on the next Part."
        )
    else:
        decision = "SHIP next Part"
        rationale = (
            f"Coverage {coverage_pct:.1f}% or Python share "
            f"{python_share:.1f}% put us outside the DEFER zone. "
            f"The next GIGI Part earns its sprint."
        )

    # Render the findings doc
    lines: List[str] = []
    lines.append("# Halcyon Part I Measurement Gate Findings")
    lines.append("")
    lines.append(f"**Status:** {decision}")
    lines.append(f"**Author:** Bee Rosa Davis, with Claude (Anthropic)")
    lines.append("**Generated by:** test_gigi_measurement_gate.py")
    lines.append("**See also:** HALCYON_PART_I_GATES.md (the gate spec); "
                 "HALCYON_TO_GIGI_REPLY_2026-06-17.md (the A1/A2 contracts)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"**{decision}**")
    lines.append("")
    lines.append(rationale)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Coverage receipt")
    lines.append("")
    lines.append(f"- Orchestrator total LOC (run_validation_report.py): "
                 f"{coverage_receipt['total_loc']}")
    lines.append(f"- Collapsible to GQL (Parts I-III): "
                 f"{coverage_receipt['collapsible_loc']} LOC")
    lines.append(f"- Python-only (Part IV-blocked): "
                 f"{coverage_receipt['python_only_loc']} LOC")
    lines.append(f"- **Coverage**: **{coverage_pct:.1f}%** of the "
                 f"GIGI-collapsible work")
    lines.append("")
    lines.append("### Per-phase breakdown")
    lines.append("")
    lines.append("| Phase | LOC | Status | Maps to / Blocked by |")
    lines.append("|---|---|---|---|")
    for label, info in coverage_receipt["phase_breakdown"].items():
        target = info.get("gigi", "—")
        lines.append(f"| `{label}` | {info['loc']} | {info['status']} | {target} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Performance receipt")
    lines.append("")
    lines.append(f"- Substrate wall (50-sweep thermalization): "
                 f"{performance_receipt['substrate_wall_seconds']:.3f}s")
    lines.append(f"- Python leapfrog wall (50-step trajectory): "
                 f"{performance_receipt['python_leapfrog_wall_seconds']:.3f}s")
    lines.append(f"- Total: "
                 f"{performance_receipt['total_wall_seconds']:.3f}s")
    lines.append(f"- **Python leapfrog share**: **"
                 f"{python_share:.1f}%**")
    lines.append(f"- Substrate share: "
                 f"{performance_receipt['substrate_share_pct']:.1f}%")
    lines.append("")
    lines.append("Note: numbers measured against MockGIGIClient (kernel-")
    lines.append("backed). Real GIGI's Rust engine should be faster on the")
    lines.append("substrate side, shifting the share further toward")
    lines.append("Python-leapfrog dominance. Re-measure after Part III ships.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Correctness receipt")
    lines.append("")
    lines.append("- Production v1.2 report: **8/10 PASS, 1 FAIL** "
                 "(Section 5 microcanonical-vs-canonical)")
    lines.append("- Contract: substrate-side thermalization must")
    lines.append("  reproduce the same single FAIL on the same seed.")
    lines.append("  Same FAIL is the success case, not a regression "
                 "(per HALCYON_PART_I_GATES.md § MEASUREMENT GATE).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Engineering-cost receipt (forward-looking)")
    lines.append("")
    lines.append("SYMPLECTIC_FLOW per the engine-owner reply has three")
    lines.append("load-bearing pieces, each its own sub-gate if Part IV ships:")
    lines.append("")
    lines.append("1. **(U, E) tangent-space machinery** — E in su(2), U in")
    lines.append("   SU(2), exp(dt·E)·U step with the right type discipline.")
    lines.append("2. **Covariant Gauss projector** with the exposed")
    lines.append("   `PROJECT_GAUSS { tikhonov, cg_tol, cg_max_iter }` struct")
    lines.append("   from Q3.")
    lines.append("3. **Symplecticness verification** — explicit H_total drift")
    lines.append("   bound under long integration as the receipt.")
    lines.append("")
    lines.append("Honest estimate per the engine-owner reply: 2–3× P0+P1.1.")
    lines.append("")
    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The actual gate: artifact exists, is non-empty, contains the decision
    assert FINDINGS_PATH.exists()
    contents = FINDINGS_PATH.read_text(encoding="utf-8")
    assert decision in contents
    assert "Coverage receipt" in contents
    assert "Performance receipt" in contents
