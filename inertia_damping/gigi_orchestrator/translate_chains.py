"""translate_chains.py -- normalize gigi-stream's Rows envelopes.

GIGI's HTTP/GQL surface returns measurement chains under CamelCase
column names per the V.0 dispatch shape (MeanPlaquette, QSurrogate,
HTotal, GaussResidualMax); snake_case forms are kept as fallbacks.
The orchestrator and the falsifiability validator already consume
ObservableId.value strings ("MEAN(PLAQUETTE)", "Q_SURROGATE", ...) or
the historic short names ("P_history", "Q_history", "H_history",
"G_history"). This module is the seam that lets the GIGI phase
runners hand the existing orchestrator dicts whose keys match the
Python-kernel shape so phases 5 / 8c / 9 / 10 / 11-13 don't know the
data came from GIGI.

Three small layers:

* ``MEASUREMENT_KEY_MAP``       — CamelCase / snake_case -> ObservableId.value
* ``normalize_measurement_history`` — applied to ``GibbsSampleResult``
  /  ``SymplecticFlowResult`` ``measurement_history`` dicts; yields a
  dict keyed by ObservableId.value with ndarray values.
* ``gauge_buffer_to_torch``     — reshape (n_edges, repr_dim) numpy
  -> contiguous torch float64 tensor of shape (n_edges, 4) the kernel
  consumers expect.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

# Map from any wire-format column name to the canonical ObservableId.value.
# Order: CamelCase (V.0 dispatch) > snake_case > ObservableId.value.
MEASUREMENT_KEY_MAP: Dict[str, str] = {
    "MeanPlaquette": "MEAN(PLAQUETTE)",
    "mean_plaquette": "MEAN(PLAQUETTE)",
    "MEAN(PLAQUETTE)": "MEAN(PLAQUETTE)",
    "QSurrogate": "Q_SURROGATE",
    "q_surrogate": "Q_SURROGATE",
    "Q_SURROGATE": "Q_SURROGATE",
    "HTotal": "H_TOTAL",
    "h_total": "H_TOTAL",
    "H_TOTAL": "H_TOTAL",
    "GaussResidualMax": "GAUSS_RESIDUAL_MAX",
    "gauss_residual_max": "GAUSS_RESIDUAL_MAX",
    "GAUSS_RESIDUAL_MAX": "GAUSS_RESIDUAL_MAX",
}


# Map ObservableId.value -> the legacy short orchestrator keys the
# Python kernel uses downstream (e.g. for sidecar fields). Kept as a
# convenience for phase_runners.
LEGACY_SHORT_KEY: Dict[str, str] = {
    "MEAN(PLAQUETTE)": "P_history",
    "Q_SURROGATE": "Q_history",
    "H_TOTAL": "H_history",
    "GAUSS_RESIDUAL_MAX": "G_history",
}


def normalize_measurement_history(
    history: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Return ``history`` with keys canonicalised to ObservableId.value.

    Accepts dicts whose keys are any of the wire column names; passes
    through any keys not in MEASUREMENT_KEY_MAP unchanged. Values are
    coerced to ``np.ndarray(dtype=float64)`` if not already.
    """
    out: Dict[str, np.ndarray] = {}
    for k, v in history.items():
        canonical = MEASUREMENT_KEY_MAP.get(k, k)
        out[canonical] = np.asarray(v, dtype=np.float64)
    return out


def to_legacy_chains(
    history: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Return a dict keyed by legacy short names (P_history, Q_history,
    H_history, G_history). Unknown observables pass through with their
    canonical key. Use when handing chains to phase 5 / phase 9.
    """
    norm = normalize_measurement_history(history)
    out: Dict[str, np.ndarray] = {}
    for canonical_key, arr in norm.items():
        legacy = LEGACY_SHORT_KEY.get(canonical_key, canonical_key)
        out[legacy] = arr
    return out


def gauge_buffer_to_torch(buf: np.ndarray) -> Any:
    """Reshape an (n_edges, repr_dim) numpy buffer from
    ``LiveGIGIClient.introspect_gauge_field`` into the contiguous
    torch float64 link tensor the kernel consumers (phase 5 dumper,
    phase 9 sidecar) expect.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError(
            "translate_chains.gauge_buffer_to_torch requires torch; "
            "install torch in the orchestrator environment."
        )
    arr = np.ascontiguousarray(np.asarray(buf, dtype=np.float64))
    return torch.from_numpy(arr).contiguous()


def gauss_history_to_2d(
    g_chain: Sequence[float],
    steps: Sequence[int],
) -> np.ndarray:
    """Build the (n_records, 2) ndarray of ``(step, value)`` rows that
    the sidecar's ``gauss_history`` field stores. The Python kernel
    builds this from ``result['steps']`` + ``result['G_history']`` at
    lines 213-217 of run_validation_report.py; GIGI's chains come back
    as scalar lists, so we zip them client-side.
    """
    if len(g_chain) != len(steps):
        raise ValueError(
            f"gauss_history_to_2d: len(steps)={len(steps)} != "
            f"len(g_chain)={len(g_chain)}"
        )
    return np.asarray(
        [(int(steps[i]), float(g_chain[i])) for i in range(len(steps))],
        dtype=np.float64,
    )


__all__ = [
    "MEASUREMENT_KEY_MAP",
    "LEGACY_SHORT_KEY",
    "normalize_measurement_history",
    "to_legacy_chains",
    "gauge_buffer_to_torch",
    "gauss_history_to_2d",
]
