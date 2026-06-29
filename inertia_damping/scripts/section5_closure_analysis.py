"""section5_closure_analysis.py — does Section 5's FAIL close under refinement?

Reads the per-trajectory sidecars produced by
``inertia_damping/scripts/section5_convergence_sweep.py`` and decides
whether the microcanonical-vs-canonical disagreement is a genuine
ergodicity caveat (gap persists) or a finite-sample artefact (gap
shrinks below tolerance as ``N_steps`` grows).

Outputs:
- ``section5_closure.json`` — per-(N, seed) table + aggregated verdict
- ``section5_closure.md`` — human-readable summary
- ``section5_gap_vs_nsteps.pdf`` — gap-vs-N_steps curves with tolerance line
- ``section5_pt_vs_ph.pdf`` — P_time and P_heatbath vs N_steps with error bars

Closure criterion (per ``HALCYON_USE_GIGI_FLAG_SPEC.md`` § 5):
- HARD closure: at the largest N_steps in the sweep, every seed has
  ``gap < tolerance`` AND ``margin_factor < 1.0``.
- SOFT closure: mean gap at largest N_steps falls below tolerance even
  if individual seeds occasionally exceed it.
- NO closure: gap is flat or grows with N_steps.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

DEFAULT_SWEEP_ROOT = _REPO / "inertia_damping" / "reports" / "section5_sweep"
DEFAULT_OUTPUT_DIR = DEFAULT_SWEEP_ROOT

SWEEP_DIR_RE = re.compile(r"^n(?P<n>\d+)_seed(?P<seed>\d+)$")


def _find_sidecar(run_dir: Path) -> Optional[Path]:
    runs = sorted([p for p in run_dir.iterdir()
                   if p.is_dir() and p.name.startswith("run_")],
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        return None
    last = runs[-1]
    jsons = sorted(last.glob("validation_report_*.json"))
    return jsons[-1] if jsons else None


def _load_section5(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"WARN: failed to load {path}: {exc}\n")
        return None
    return report.get("section_5_method_crosscheck")


def _extract_row(n_steps: int, seed: int, s5: dict) -> dict:
    return {
        "n_steps": n_steps,
        "seed": seed,
        "available": bool(s5.get("available", False)),
        "verdict": s5.get("verdict"),
        "gap": s5.get("gap"),
        "tolerance": s5.get("tolerance"),
        "margin_factor": s5.get("margin_factor"),
        "P_heatbath": s5.get("P_heatbath"),
        "P_time": s5.get("P_time"),
        "P_heatbath_sem": s5.get("P_heatbath_sem"),
        "P_time_sem": s5.get("P_time_sem"),
        "P_heatbath_n_eff": s5.get("P_heatbath_n_eff"),
        "P_time_n_eff": s5.get("P_time_n_eff"),
        "P_heatbath_blocking_regime": s5.get("P_heatbath_blocking_regime"),
        "P_time_blocking_regime": s5.get("P_time_blocking_regime"),
        "n_heatbath_samples": s5.get("n_heatbath_samples"),
        "n_time_samples": s5.get("n_time_samples"),
    }


def _collect(sweep_root: Path) -> List[dict]:
    rows: List[dict] = []
    for child in sorted(sweep_root.iterdir()):
        if not child.is_dir():
            continue
        m = SWEEP_DIR_RE.match(child.name)
        if not m:
            continue
        n = int(m.group("n"))
        seed = int(m.group("seed"))
        sidecar = _find_sidecar(child)
        if sidecar is None:
            rows.append({"n_steps": n, "seed": seed, "available": False,
                         "verdict": "MISSING", "gap": None})
            continue
        s5 = _load_section5(sidecar)
        if s5 is None:
            rows.append({"n_steps": n, "seed": seed, "available": False,
                         "verdict": "BAD_JSON", "gap": None,
                         "json_path": str(sidecar)})
            continue
        row = _extract_row(n, seed, s5)
        row["json_path"] = str(sidecar)
        rows.append(row)
    return rows


def _aggregate_by_n(rows: List[dict]) -> List[dict]:
    by_n: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("available"):
            by_n[r["n_steps"]].append(r)
    out = []
    for n in sorted(by_n):
        bucket = by_n[n]
        gaps = [r["gap"] for r in bucket if r["gap"] is not None]
        margins = [r["margin_factor"] for r in bucket
                   if r["margin_factor"] is not None]
        pass_count = sum(1 for r in bucket if r.get("verdict") == "PASS")
        tol = bucket[0].get("tolerance")
        n_below = sum(1 for g in gaps if tol is not None and g < tol)
        mean_gap = sum(gaps) / len(gaps) if gaps else None
        max_gap = max(gaps) if gaps else None
        min_gap = min(gaps) if gaps else None
        sem_gap = None
        if gaps and len(gaps) > 1:
            mu = mean_gap
            var = sum((g - mu) ** 2 for g in gaps) / (len(gaps) - 1)
            sem_gap = math.sqrt(var / len(gaps))
        out.append({
            "n_steps": n,
            "n_seeds": len(bucket),
            "n_pass": pass_count,
            "n_gap_below_tol": n_below,
            "tolerance": tol,
            "mean_gap": mean_gap,
            "sem_gap": sem_gap,
            "min_gap": min_gap,
            "max_gap": max_gap,
            "mean_margin_factor": (sum(margins) / len(margins)
                                   if margins else None),
            "max_margin_factor": max(margins) if margins else None,
        })
    return out


def _closure_verdict(agg: List[dict]) -> dict:
    if not agg:
        return {"verdict": "NO_DATA", "rationale": "no completed runs"}
    largest = agg[-1]
    smallest = agg[0]
    tol = largest.get("tolerance")
    rationale_bits: List[str] = []
    hard = (largest["n_pass"] == largest["n_seeds"]
            and largest["max_margin_factor"] is not None
            and largest["max_margin_factor"] < 1.0)
    soft = (tol is not None and largest["mean_gap"] is not None
            and largest["mean_gap"] < tol)
    # Trend: does mean_gap decrease as N_steps grows?
    if (smallest["mean_gap"] is not None and largest["mean_gap"] is not None
            and smallest["mean_gap"] > 0):
        ratio = largest["mean_gap"] / smallest["mean_gap"]
    else:
        ratio = None
    if ratio is not None:
        rationale_bits.append(
            f"mean_gap shrinks from {smallest['mean_gap']:.4f} "
            f"(N={smallest['n_steps']}) to {largest['mean_gap']:.4f} "
            f"(N={largest['n_steps']}); ratio={ratio:.2f}"
        )
    if hard:
        verdict = "HARD_CLOSURE"
        rationale_bits.append(
            f"all {largest['n_seeds']} seeds PASS at N={largest['n_steps']} "
            f"with max margin_factor={largest['max_margin_factor']:.2f}"
        )
    elif soft:
        verdict = "SOFT_CLOSURE"
        rationale_bits.append(
            f"mean_gap={largest['mean_gap']:.4f} < tolerance={tol} "
            f"at N={largest['n_steps']}, but only "
            f"{largest['n_pass']}/{largest['n_seeds']} seeds individually PASS"
        )
    elif ratio is not None and ratio < 0.9:
        verdict = "PARTIAL_CLOSURE"
        rationale_bits.append(
            f"gap trending down but mean_gap={largest['mean_gap']:.4f} "
            f"still > tolerance={tol}; would need larger N_steps to close"
        )
    else:
        verdict = "NO_CLOSURE"
        rationale_bits.append(
            "gap is flat or non-monotone; the FAIL is a real ergodicity "
            "caveat at these sweep sizes"
        )
    return {
        "verdict": verdict,
        "hard_closure": hard,
        "soft_closure": soft,
        "mean_gap_ratio_largest_over_smallest": ratio,
        "tolerance": tol,
        "largest_n_steps": largest["n_steps"],
        "largest_n_pass": largest["n_pass"],
        "largest_n_seeds": largest["n_seeds"],
        "largest_mean_gap": largest["mean_gap"],
        "largest_max_gap": largest["max_gap"],
        "largest_max_margin_factor": largest["max_margin_factor"],
        "rationale": "; ".join(rationale_bits),
    }


def _emit_markdown(rows: List[dict], agg: List[dict],
                   verdict: dict, sweep_root: Path) -> str:
    lines = []
    lines.append("# Section 5 closure analysis\n")
    lines.append(f"Sweep root: `{sweep_root}`\n")
    lines.append(f"Closure verdict: **{verdict['verdict']}**\n")
    lines.append(f"Rationale: {verdict['rationale']}\n")
    lines.append("\n## Per-N aggregate\n")
    lines.append("| N_steps | seeds | PASS | gap<tol | mean_gap | sem_gap | max_gap | max margin |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a in agg:
        def fmt(x, d=4):
            return f"{x:.{d}f}" if isinstance(x, float) else "—"
        lines.append(
            f"| {a['n_steps']} | {a['n_seeds']} | "
            f"{a['n_pass']} | {a['n_gap_below_tol']} | "
            f"{fmt(a['mean_gap'])} | {fmt(a['sem_gap'])} | "
            f"{fmt(a['max_gap'])} | {fmt(a['max_margin_factor'], 2)} |"
        )
    lines.append("\n## Per-trajectory rows\n")
    lines.append("| N_steps | seed | verdict | gap | margin | P_heatbath | P_time | P_t n_eff | P_t regime |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["n_steps"], r["seed"])):
        def fmt(x, d=4):
            return f"{x:.{d}f}" if isinstance(x, float) else "—"
        lines.append(
            f"| {r['n_steps']} | {r['seed']} | "
            f"{r.get('verdict','?')} | {fmt(r.get('gap'))} | "
            f"{fmt(r.get('margin_factor'), 2)} | "
            f"{fmt(r.get('P_heatbath'))} | {fmt(r.get('P_time'))} | "
            f"{fmt(r.get('P_time_n_eff'), 1)} | "
            f"{r.get('P_time_blocking_regime','—')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _emit_figures(rows: List[dict], agg: List[dict],
                  output_dir: Path) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("WARN: matplotlib not available; skipping figures\n")
        return []
    paths: List[Path] = []
    by_seed: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if r.get("available") and r.get("gap") is not None:
            by_seed[r["seed"]].append((r["n_steps"], r["gap"]))
    # Figure 1: gap vs N_steps, one line per seed
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, pts in sorted(by_seed.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=f"seed {seed}", alpha=0.7)
    tol = next((a["tolerance"] for a in agg if a["tolerance"] is not None),
               None)
    if tol is not None:
        ax.axhline(tol, color="red", linestyle="--",
                   label=f"tolerance = {tol}")
    ax.set_xscale("log")
    ax.set_xlabel("N_steps (symplectic leapfrog)")
    ax.set_ylabel("|P_heatbath - P_time|")
    ax.set_title("Section 5 gap vs N_steps (per seed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = output_dir / "section5_gap_vs_nsteps.pdf"
    fig.savefig(p1)
    plt.close(fig)
    paths.append(p1)
    # Figure 2: P_heatbath and P_time vs N_steps with SEM error bars
    fig, ax = plt.subplots(figsize=(8, 5))
    by_n_ph: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    by_n_pt: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if not r.get("available"):
            continue
        if r.get("P_heatbath") is not None:
            by_n_ph[r["n_steps"]].append(
                (r["P_heatbath"], r.get("P_heatbath_sem") or 0.0))
        if r.get("P_time") is not None:
            by_n_pt[r["n_steps"]].append(
                (r["P_time"], r.get("P_time_sem") or 0.0))
    ns = sorted(set(by_n_ph) | set(by_n_pt))
    ph_means = [sum(v[0] for v in by_n_ph[n]) / len(by_n_ph[n])
                if by_n_ph[n] else None for n in ns]
    pt_means = [sum(v[0] for v in by_n_pt[n]) / len(by_n_pt[n])
                if by_n_pt[n] else None for n in ns]
    ph_sems = [(sum(v[1] ** 2 for v in by_n_ph[n]) ** 0.5
                / max(len(by_n_ph[n]), 1))
               if by_n_ph[n] else 0.0 for n in ns]
    pt_sems = [(sum(v[1] ** 2 for v in by_n_pt[n]) ** 0.5
                / max(len(by_n_pt[n]), 1))
               if by_n_pt[n] else 0.0 for n in ns]
    ax.errorbar(ns, ph_means, yerr=ph_sems, marker="o",
                label="P_heatbath (canonical)", capsize=4)
    ax.errorbar(ns, pt_means, yerr=pt_sems, marker="s",
                label="P_time (microcanonical)", capsize=4)
    ax.set_xscale("log")
    ax.set_xlabel("N_steps (symplectic leapfrog)")
    ax.set_ylabel("mean plaquette")
    ax.set_title("Microcanonical P_time approaches canonical P_heatbath?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = output_dir / "section5_pt_vs_ph.pdf"
    fig.savefig(p2)
    plt.close(fig)
    paths.append(p2)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=str, default=str(DEFAULT_SWEEP_ROOT))
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    output_dir = (Path(args.output_dir).resolve() if args.output_dir
                  else sweep_root)
    if not sweep_root.exists():
        sys.stderr.write(f"ERROR: sweep root not found: {sweep_root}\n")
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect(sweep_root)
    if not rows:
        sys.stderr.write("ERROR: no n*_seed* directories found\n")
        return 2
    agg = _aggregate_by_n(rows)
    verdict = _closure_verdict(agg)

    closure = {
        "sweep_root": str(sweep_root),
        "n_trajectories": len(rows),
        "n_available": sum(1 for r in rows if r.get("available")),
        "closure": verdict,
        "by_n_steps": agg,
        "rows": rows,
    }
    json_path = output_dir / "section5_closure.json"
    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(closure, fh, indent=2)
    md_path = output_dir / "section5_closure.md"
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_emit_markdown(rows, agg, verdict, sweep_root))
    figs = _emit_figures(rows, agg, output_dir)

    print(f"Closure verdict: {verdict['verdict']}")
    print(f"  rationale: {verdict['rationale']}")
    print(f"  JSON:      {json_path}")
    print(f"  Markdown:  {md_path}")
    for f in figs:
        print(f"  Figure:    {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
