"""
sector_classifier.py -- Blind classifier gate for the Halcyon validation
discipline.

This is NOT a topological-charge test. pi_2(SU(2)) = 0 on S^2; Q_surrogate is
the cumulative arccos-of-trace observable defined in
buckyball_observables.Q_surrogate, range [0, 16]. The three bands
(B0 = [0, 0.5), B1 = [0.5, 1.5), B2 = [1.5, 2.5)) are operational labels for
calibration ensembles -- NOT topological sectors.

MODULE PURPOSE
--------------
Discrimination test on calibration windows: does an LOO k-NN classifier
reading 4 gauge-invariant features assign trajectory windows to their
observed Q_surrogate band better than a label-permutation null?

The B3-reframed version (1.2 schema) drops synthetic ensemble construction
entirely. The classifier reads windows from the EXISTING load-bearing
leapfrog trajectory (or a long heatbath chain), bins each window by mean
Q_surrogate, and tests whether the 4 gauge-invariant features carry
discriminative information about the bin label.

Determinism: pure numpy; cross-OS reproducibility is NOT claimed (LOO
ordering and Mahalanobis covariance computation may differ at FP64 epsilon
across BLAS implementations). Intra-process is bit-identical at fixed seed.

Discipline rules: NEVER write "Q=k sector". Use "B0/B1/B2 band" or
"plaquette-angle-sum band". NEVER per-edge Haar scramble. NEVER sigma_z
nudge. The label-permutation null is the canonical statistical contract.
"""
from __future__ import annotations

import importlib.util
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FEATURE_NAMES: Tuple[str, ...] = (
    "total_plaq_action",
    "Q_surrogate",
    "P_mean",
    "max_plaq_holonomy_dist",
)

BAND_EDGES: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.5),
    (0.5, 1.5),
    (1.5, 2.5),
)
BAND_NAMES: Tuple[str, ...] = ("B0", "B1", "B2")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_kernel(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_sc_{name}", os.path.join(_HERE, f"{name}.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def feature_vector(U, graph) -> np.ndarray:
    """4 gauge-invariant features extracted from a single (n_edges, 4) U state.

    Returns
    -------
    np.ndarray shape (4,):
        [total_plaq_action, Q_surrogate, P_mean, max_plaq_holonomy_dist]
    """
    import torch
    action = _load_kernel("buckyball_action")
    observables = _load_kernel("buckyball_observables")
    if not isinstance(U, torch.Tensor):
        U = torch.as_tensor(U, dtype=torch.float64)
    Uf = action.all_face_holonomies(U, graph)              # (F, 4)
    q0 = Uf[:, 0].clamp(-1.0, 1.0)
    F = graph.n_faces
    # total Wilson plaquette action = sum_f [1 - q0(U_f)] (in our convention)
    total_plaq_action = float((1.0 - q0).sum().item())
    # Q_surrogate from observables (range [0, 16] on the buckyball)
    Q_surr = float(observables.Q_surrogate(U, graph))
    # Mean plaquette
    P_mean = float(q0.mean().item())
    # Max per-face holonomy distance from identity: max arccos(q0) / pi in [0,1]
    angles = torch.acos(q0)
    max_plaq_holonomy_dist = float((angles / math.pi).max().item())
    return np.asarray(
        [total_plaq_action, Q_surr, P_mean, max_plaq_holonomy_dist],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Window collection on real trajectory
# ---------------------------------------------------------------------------
def _band_for_q(q: float) -> Optional[int]:
    """Map a Q_surrogate value to a band label in {0, 1, 2}. Returns None outside."""
    for i, (lo, hi) in enumerate(BAND_EDGES):
        if lo <= q < hi:
            return i
    return None


def collect_real_windows(
    U_traj: List[Any], graph, window_size: int = 10,
) -> Dict[int, List[np.ndarray]]:
    """Walk an existing trajectory, bin each window by mean Q_surrogate.

    Returns {label: [feature_vectors]} if all 3 bands have >= 10 windows;
    otherwise returns {} (the gate verdict will become NOT_APPLICABLE).
    """
    observables = _load_kernel("buckyball_observables")
    if not U_traj or len(U_traj) < window_size:
        return {}
    ensembles: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    n_frames = len(U_traj)
    for start in range(0, n_frames - window_size + 1, window_size):
        window = U_traj[start:start + window_size]
        # Per-window mean Q_surrogate -> band label.
        q_vals = [float(observables.Q_surrogate(U_i, graph)) for U_i in window]
        q_mean = float(np.mean(q_vals))
        label = _band_for_q(q_mean)
        if label is None:
            continue
        # Use the LAST frame's feature vector to avoid intra-window averaging
        # of features (keeps n_eff honest; correlated windows reduce n_eff).
        ensembles[label].append(feature_vector(window[-1], graph))
    if min(len(ensembles[k]) for k in (0, 1, 2)) < 10:
        return {}
    return ensembles


def _ensembles_to_matrix(
    ensembles: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack ensembles into (X, y). Deterministic label order 0, 1, 2."""
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for label in (0, 1, 2):
        for fv in ensembles[label]:
            X_list.append(fv)
            y_list.append(label)
    return np.asarray(X_list, dtype=np.float64), np.asarray(y_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# k-NN LOO classifier with Mahalanobis distance
# ---------------------------------------------------------------------------
def _mahalanobis_distance(
    X_train: np.ndarray, X_query: np.ndarray, ridge: float = 1e-8,
) -> np.ndarray:
    """Mahalanobis distance from each query to each training row.

    Covariance is computed on the TRAINING fold only (no leakage).
    Ridge-regularised so collinear features don't blow up the inverse.
    """
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    cov = (Xc.T @ Xc) / max(X_train.shape[0] - 1, 1)
    cov += ridge * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)
    diffs = X_query[:, None, :] - X_train[None, :, :]    # (n_q, n_train, d)
    # Quadratic form: (n_q, n_train)
    d2 = np.einsum("qti,ij,qtj->qt", diffs, cov_inv, diffs)
    return np.sqrt(np.maximum(d2, 0.0))


def knn_loo_classify(
    feature_matrix: np.ndarray, labels: np.ndarray, k: int = 5,
) -> Dict[str, Any]:
    """k-NN LOO classifier with Mahalanobis distance. No leakage in the
    covariance estimate: each LOO fold recomputes mu/cov on the training fold.

    Returns accuracy + per-class accuracy + confusion matrix + n_eff.
    """
    N = feature_matrix.shape[0]
    if N < k + 1:
        raise ValueError(f"need N > k = {k}; got N = {N}")
    predictions = np.empty(N, dtype=np.int64)
    for i in range(N):
        mask = np.arange(N) != i
        X_tr = feature_matrix[mask]
        y_tr = labels[mask]
        X_te = feature_matrix[i:i + 1]
        dists = _mahalanobis_distance(X_tr, X_te)[0]
        nn = np.argpartition(dists, k)[:k]
        nn_labels = y_tr[nn]
        # Majority vote, tie-break by smallest mean distance.
        counts = np.bincount(nn_labels, minlength=3)
        max_count = counts.max()
        top = [c for c in range(3) if counts[c] == max_count]
        if len(top) == 1:
            predictions[i] = top[0]
        else:
            # Tie -> argmin mean distance among tied classes.
            best, best_d = top[0], float("inf")
            for c in top:
                mean_d = dists[nn[nn_labels == c]].mean()
                if mean_d < best_d:
                    best, best_d = c, mean_d
            predictions[i] = best
    accuracy = float((predictions == labels).mean())
    per_class_accuracy: Dict[str, float] = {}
    confusion = np.zeros((3, 3), dtype=np.int64)
    for true_lab in range(3):
        mask = labels == true_lab
        if mask.any():
            per_class_accuracy[BAND_NAMES[true_lab]] = float((predictions[mask] == true_lab).mean())
        for pred_lab in range(3):
            confusion[true_lab, pred_lab] = int(((labels == true_lab) & (predictions == pred_lab)).sum())
    # n_eff: estimate via lag-1 autocorrelation of the binary correct/incorrect
    # series. Conservative: n_eff = N / (1 + 2 * sum of positive autocorrelations).
    correct = (predictions == labels).astype(np.float64)
    if correct.std() > 0 and N >= 4:
        c_center = correct - correct.mean()
        denom = (c_center * c_center).sum()
        rho1 = float((c_center[1:] * c_center[:-1]).sum() / max(denom, 1e-30))
        # Single lag estimator; n_eff = N (1 - rho) / (1 + rho) clamped.
        rho1 = max(0.0, min(0.99, rho1))
        n_eff = float(N * (1.0 - rho1) / (1.0 + rho1))
        n_eff = max(1.0, min(float(N), n_eff))
    else:
        n_eff = float(N)
    return {
        "predictions": predictions.tolist(),
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
        "n_eff": n_eff,
    }


# ---------------------------------------------------------------------------
# Label-permutation null
# ---------------------------------------------------------------------------
def permutation_null(
    feature_matrix: np.ndarray, labels: np.ndarray,
    n_permutations: int = 200, k: int = 5, seed: int = 20260617,
) -> Dict[str, Any]:
    """200 random shuffles of the real labels; for each, run LOO and record
    accuracy. p_value = (1 + #(shuffled_acc >= real_acc)) / (1 + n_permutations).
    """
    real = knn_loo_classify(feature_matrix, labels, k=k)
    real_acc = real["accuracy"]
    rng = np.random.default_rng(seed)
    null_accs: List[float] = []
    for i in range(n_permutations):
        perm = rng.permutation(labels.size)
        shuffled = labels[perm]
        out = knn_loo_classify(feature_matrix, shuffled, k=k)
        null_accs.append(out["accuracy"])
    null_accs_arr = np.asarray(null_accs, dtype=np.float64)
    p_value = float((1 + int((null_accs_arr >= real_acc).sum())) / (1 + n_permutations))
    return {
        "null_accuracy_mean": float(null_accs_arr.mean()),
        "null_accuracy_std": float(null_accs_arr.std(ddof=1)) if n_permutations > 1 else 0.0,
        "null_accuracy_distribution": null_accs,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Feature ablation (MANDATORY per adversarial finding)
# ---------------------------------------------------------------------------
def feature_ablation(
    feature_matrix: np.ndarray, labels: np.ndarray, k: int = 5,
) -> Dict[str, Any]:
    """Leave-one-feature-out LOO accuracy AND single-feature LOO accuracy.

    The 'single_feature_max' field catches gauge-leak: if any one feature
    alone reaches >= TOL_real_accuracy_min, the classifier is reading a
    surface label rather than a gauge-invariant content gradient. The
    top-level gate marks FAIL in that case.
    """
    out: Dict[str, Any] = {"leave_one_out": {}, "single_feature": {}}
    n_features = feature_matrix.shape[1]
    for i, name in enumerate(FEATURE_NAMES[:n_features]):
        # Leave-one-feature-out
        mask = [j for j in range(n_features) if j != i]
        if mask:
            res = knn_loo_classify(feature_matrix[:, mask], labels, k=k)
            out["leave_one_out"][name] = res["accuracy"]
        # Single feature
        res_single = knn_loo_classify(feature_matrix[:, i:i + 1], labels, k=k)
        out["single_feature"][name] = res_single["accuracy"]
    out["single_feature_max"] = max(out["single_feature"].values()) if out["single_feature"] else 0.0
    return out


# ---------------------------------------------------------------------------
# Top-level gate
# ---------------------------------------------------------------------------
def run_classifier_gate(
    graph, U_traj: List[Any], beta: float,
    k: int = 5, seed: int = 20260617,
) -> Dict[str, Any]:
    """Top-level entry point. Reads thresholds via classifier_thresholds()
    so the orchestrator and the report builder are not duplicates.
    """
    # Lazy import to avoid circular dep
    from inertia_damping import validation_report
    real_min, null_alpha = validation_report.classifier_thresholds()

    ensembles = collect_real_windows(U_traj, graph)
    if not ensembles:
        return {
            "available": True,
            "verdict": "NOT_APPLICABLE",
            "reason": ("natural Q_surrogate dispersion at beta = "
                       f"{beta} does not populate all 3 bands; "
                       "classifier is meaningless here"),
            "real_accuracy_threshold": real_min,
            "null_alpha_threshold": null_alpha,
        }
    X, y = _ensembles_to_matrix(ensembles)
    real = knn_loo_classify(X, y, k=k)
    null = permutation_null(X, y, k=k, seed=seed)
    ablation = feature_ablation(X, y, k=k)

    # Pass criteria:
    #   (a) real accuracy >= TOL_real_min
    #   (b) permutation p-value <= TOL_null_alpha (real classifier reads labels)
    #   (c) feature_ablation.single_feature_max < TOL_real_min (no single
    #       feature alone hits the gate -- catches gauge-leak)
    real_accuracy = real["accuracy"]
    p_value = null["p_value"]
    single_max = ablation["single_feature_max"]

    fail_reasons: List[str] = []
    if real_accuracy < real_min:
        fail_reasons.append(
            f"real_accuracy {real_accuracy:.3f} < threshold {real_min:.3f}"
        )
    if p_value > null_alpha:
        fail_reasons.append(
            f"permutation p_value {p_value:.3f} > null_alpha {null_alpha:.3f} "
            "(classifier is not statistically distinguishable from a "
            "label-blind baseline)"
        )
    if single_max >= real_min:
        fail_reasons.append(
            f"single_feature_max {single_max:.3f} >= real_min {real_min:.3f} "
            "(gate cleared by ONE feature alone; gauge-leak suspected)"
        )

    verdict = "PASS" if not fail_reasons else "FAIL"

    return {
        "available": True,
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "real_accuracy": real_accuracy,
        "real_accuracy_threshold": real_min,
        "per_class_accuracy": real["per_class_accuracy"],
        "confusion_matrix": real["confusion_matrix"],
        "n_eff": real["n_eff"],
        "n_samples": int(X.shape[0]),
        "n_samples_per_band": {BAND_NAMES[i]: len(ensembles[i]) for i in (0, 1, 2)},
        "permutation_p_value": p_value,
        "permutation_null_alpha": null_alpha,
        "permutation_null_accuracy_mean": null["null_accuracy_mean"],
        "permutation_null_accuracy_std": null["null_accuracy_std"],
        "feature_ablation": ablation,
        "feature_names": list(FEATURE_NAMES),
        "band_edges": [list(e) for e in BAND_EDGES],
        "band_names": list(BAND_NAMES),
        "k": k,
        "seed": seed,
    }


__all__ = [
    "FEATURE_NAMES",
    "BAND_EDGES",
    "BAND_NAMES",
    "feature_vector",
    "collect_real_windows",
    "knn_loo_classify",
    "permutation_null",
    "feature_ablation",
    "run_classifier_gate",
]
