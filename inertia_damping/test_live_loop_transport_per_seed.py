"""Unit tests for `LiveLoopTransportClient`'s per-seed thermalization
decomposition (no live engine required — stubs out the HTTP layer).

Background: per Gigi's substrate-determinism stance (2026-06-21 VI.6
readout), the substrate's LOOP_TRANSPORT verb is deterministic given
fixed `(U, E)` input. To produce the per-seed variance v3.1.3 §3.5
requires, Halcyon's live client decomposes any multi-seed request into
N single-seed sub-calls, each preceded by a `GIBBS_SAMPLE U_lt SEED
<per_seed>` re-thermalization. These tests verify the decomposition
fires the right substrate queries in the right order, aggregates the
per-seed scalars correctly, and computes σ_H_blocked as the unbiased
SEM across the per-seed ensemble.

The live smoke test (`test_live_loop_transport_smoke.py`) is skip-by-
default and requires `HALCYON_LIVE_SMOKE=1` + a running gigi-stream;
these tests run on every `pytest inertia_damping/` invocation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
import pytest

from inertia_damping.gigi_client.live_loop_transport import (
    LiveLoopTransportClient,
)
from inertia_damping.gigi_client.loop_transport import (
    HalcyonParameterPack,
    LoopTransportRequest,
)
from inertia_damping.holonomy_battery.loops import GAMMA_UNIT


# ---------------------------------------------------------------------------
# Test fixture: build a client with HTTP stubbed out.
# ---------------------------------------------------------------------------


class _GqlStub:
    """Records the GQL queries the client sends and returns canned
    rows. Per-seed `h_scalar` lookup is keyed on the seed extracted
    from the SEEDS clause; per-seed thermalizations get a no-op
    success response.
    """

    def __init__(self, seed_to_h_scalar: Dict[int, float]):
        self.seed_to_h_scalar = seed_to_h_scalar
        self.sent_queries: List[str] = []

    def __call__(self, query: str) -> Any:
        self.sent_queries.append(query)
        if "GIBBS_SAMPLE" in query:
            return {
                "rows": [
                    {
                        "mean_plaquette": [0.51],
                        "n_sweeps_completed": 200,
                    }
                ]
            }
        if "LOOP_TRANSPORT" in query:
            m = re.search(r"SEEDS \[(\d+)\.\.(\d+)\]", query)
            assert m is not None, f"no SEEDS range in query: {query!r}"
            lo, hi = int(m.group(1)), int(m.group(2))
            assert lo == hi, (
                "per-seed decomposition should emit single-seed ranges; "
                f"got [{lo}..{hi}]"
            )
            seed = lo
            assert seed in self.seed_to_h_scalar, (
                f"unexpected per-seed call for seed {seed}; "
                f"known seeds: {sorted(self.seed_to_h_scalar)}"
            )
            h = self.seed_to_h_scalar[seed]
            return {
                "rows": [
                    {
                        "per_seed_h_forward": [h],
                        "per_seed_h_reversed": [-h],
                        "h_forward": h,
                        "h_reversed": -h,
                        "sigma_h_blocked": 0.0,
                        "tracking_error_max_q": 0.01,
                        "tracking_error_max_beta_w": 0.02,
                        "adiabaticity_ratio": 0.05,
                        "adiabaticity_verdict": "ACCEPTABLE",
                        "n_substeps_completed": 1000,
                    }
                ]
            }
        # Unknown query — shouldn't happen in these tests.
        return {"rows": []}


def _build_stubbed_client(
    seed_to_h_scalar: Dict[int, float],
    per_seed_thermalize: bool = True,
) -> "tuple[LiveLoopTransportClient, _GqlStub]":
    client = LiveLoopTransportClient(
        per_seed_thermalize=per_seed_thermalize, ping=False
    )
    # Bypass the HTTP-roundtripping declare_loop by populating the cache
    # directly. The decomposition's loop-name check looks up against
    # _declared_loops, so a manual seed is enough.
    client._declared_loops[GAMMA_UNIT.name] = GAMMA_UNIT
    stub = _GqlStub(seed_to_h_scalar)
    client._gql_query = stub  # type: ignore[method-assign]
    return client, stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multi_seed_request_decomposes_into_per_seed_subcalls():
    """A 3-seed request fires 3 GIBBS_SAMPLE re-thermalizations followed
    by 3 single-seed LOOP_TRANSPORT calls, in matching order."""
    seed_h = {20260616: 0.7, 20260617: 0.8, 20260618: 0.9}
    client, stub = _build_stubbed_client(seed_h)

    request = LoopTransportRequest(
        gauge_field_name="halcyon_canonical_buckyball",
        loop=GAMMA_UNIT,
        direction="FORWARD",
        pack=HalcyonParameterPack(),
        seeds=(20260616, 20260617, 20260618),
        n_discretization=1000,
    )
    result = client.loop_transport(request)

    # Per-seed scalar vector preserves order of request.seeds.
    assert result.per_seed_h_scalar.shape == (3,)
    np.testing.assert_allclose(
        result.per_seed_h_scalar, np.array([0.7, 0.8, 0.9])
    )

    # Per-seed holonomy stack has shape (3, 4).
    assert result.per_seed_holonomy.shape == (3, 4)

    # σ_H_blocked is the unbiased SEM across per-seed values.
    expected_sigma = float(np.std([0.7, 0.8, 0.9], ddof=1) / np.sqrt(3))
    assert abs(result.sigma_h_blocked - expected_sigma) < 1e-12

    # Substrate queries: 3 GIBBS_SAMPLEs + 3 LOOP_TRANSPORTs = 6 total.
    gibbs_queries = [q for q in stub.sent_queries if "GIBBS_SAMPLE" in q]
    lt_queries = [q for q in stub.sent_queries if "LOOP_TRANSPORT" in q]
    assert len(gibbs_queries) == 3
    assert len(lt_queries) == 3

    # Each thermalization carries its per-seed SEED.
    assert "SEED 20260616" in gibbs_queries[0]
    assert "SEED 20260617" in gibbs_queries[1]
    assert "SEED 20260618" in gibbs_queries[2]

    # Interleaving: GIBBS_SAMPLE before each LOOP_TRANSPORT for the same seed.
    # Expected ordering: GIBBS(616), LT(616), GIBBS(617), LT(617), GIBBS(618), LT(618).
    assert "GIBBS_SAMPLE" in stub.sent_queries[0]
    assert "LOOP_TRANSPORT" in stub.sent_queries[1]
    assert "SEEDS [20260616..20260616]" in stub.sent_queries[1]
    assert "GIBBS_SAMPLE" in stub.sent_queries[2]
    assert "LOOP_TRANSPORT" in stub.sent_queries[3]
    assert "SEEDS [20260617..20260617]" in stub.sent_queries[3]


def test_multi_seed_reversed_direction_picks_reversed_scalar():
    """When direction=REVERSED, the per_seed_h_scalar carries the
    per_seed_h_reversed field from each sub-call (which the stub
    returns as -h for each seed)."""
    seed_h = {20260616: 0.7, 20260617: 0.8}
    client, _ = _build_stubbed_client(seed_h)

    request = LoopTransportRequest(
        gauge_field_name="halcyon_canonical_buckyball",
        loop=GAMMA_UNIT,
        direction="REVERSED",
        pack=HalcyonParameterPack(),
        seeds=(20260616, 20260617),
        n_discretization=1000,
    )
    result = client.loop_transport(request)

    np.testing.assert_allclose(result.per_seed_h_scalar, np.array([-0.7, -0.8]))


def test_single_seed_request_uses_single_shot_path():
    """A 1-seed request doesn't trigger the decomposition — it goes
    through the original single-shot LOOP_TRANSPORT call directly."""
    seed_h = {20260616: 0.42}
    client, stub = _build_stubbed_client(seed_h)

    request = LoopTransportRequest(
        gauge_field_name="halcyon_canonical_buckyball",
        loop=GAMMA_UNIT,
        direction="FORWARD",
        pack=HalcyonParameterPack(),
        seeds=(20260616,),
        n_discretization=1000,
    )
    result = client.loop_transport(request)

    # Single seed, single LOOP_TRANSPORT, no GIBBS_SAMPLE.
    assert result.per_seed_h_scalar.shape == (1,)
    np.testing.assert_allclose(result.per_seed_h_scalar, np.array([0.42]))
    assert all("GIBBS_SAMPLE" not in q for q in stub.sent_queries)
    assert sum("LOOP_TRANSPORT" in q for q in stub.sent_queries) == 1


def test_per_seed_thermalize_off_uses_single_shot_path():
    """Disabling per_seed_thermalize at construction time falls back to
    the original single-call multi-seed behavior — useful for
    integration tests that want to exercise the substrate's
    seed-handling in isolation, separate from Halcyon's thermalization
    discipline."""
    seed_h = {20260616: 0.7, 20260617: 0.8}
    client, stub = _build_stubbed_client(seed_h, per_seed_thermalize=False)

    # Bypass the SEEDS-range single-stub assertion by patching the stub
    # to expect a multi-seed range.
    def stub_multi_seed(query: str):
        if "LOOP_TRANSPORT" in query:
            return {
                "rows": [
                    {
                        "per_seed_h_forward": [0.7, 0.8],
                        "per_seed_h_reversed": [-0.7, -0.8],
                        "h_forward": 0.75,
                        "h_reversed": -0.75,
                        "sigma_h_blocked": 1e-16,
                        "tracking_error_max_q": 0.0,
                        "tracking_error_max_beta_w": 0.0,
                        "adiabaticity_ratio": 1.0,
                        "adiabaticity_verdict": "AMBIGUOUS_FORCED",
                        "n_substeps_completed": 1000,
                    }
                ]
            }
        return {"rows": []}

    client._gql_query = stub_multi_seed  # type: ignore[method-assign]

    request = LoopTransportRequest(
        gauge_field_name="halcyon_canonical_buckyball",
        loop=GAMMA_UNIT,
        direction="FORWARD",
        pack=HalcyonParameterPack(),
        seeds=(20260616, 20260617),
        n_discretization=1000,
    )
    result = client.loop_transport(request)
    # The single-shot path returns the substrate's raw per_seed_h_forward
    # vector (which in the original buggy substrate would be the
    # placeholder; the test only verifies the path was single-shot).
    assert result.per_seed_h_scalar.shape == (2,)
    np.testing.assert_allclose(result.per_seed_h_scalar, np.array([0.7, 0.8]))


def test_decomposition_aggregates_diagnostics_as_max_across_seeds():
    """Tracking-error and adiabaticity-ratio fields are aggregated as
    MAX across per-seed sub-calls — Halcyon's gates fire on the
    worst-case across the ensemble, not the per-call value."""
    seed_h = {20260616: 0.7, 20260617: 0.8}

    client = LiveLoopTransportClient(per_seed_thermalize=True, ping=False)
    client._declared_loops[GAMMA_UNIT.name] = GAMMA_UNIT

    # Per-seed varying diagnostic values to verify the MAX aggregation.
    per_seed_diagnostics = {
        20260616: {"tracking_q": 0.02, "tracking_b": 0.03, "adia": 0.05},
        20260617: {"tracking_q": 0.04, "tracking_b": 0.01, "adia": 0.08},
    }

    def stub_gql_query(query: str):
        if "GIBBS_SAMPLE" in query:
            return {"rows": [{"mean_plaquette": [0.51]}]}
        if "LOOP_TRANSPORT" in query:
            m = re.search(r"SEEDS \[(\d+)\.\.(\d+)\]", query)
            seed = int(m.group(1))
            d = per_seed_diagnostics[seed]
            return {
                "rows": [
                    {
                        "per_seed_h_forward": [seed_h[seed]],
                        "per_seed_h_reversed": [-seed_h[seed]],
                        "h_forward": seed_h[seed],
                        "h_reversed": -seed_h[seed],
                        "sigma_h_blocked": 0.0,
                        "tracking_error_max_q": d["tracking_q"],
                        "tracking_error_max_beta_w": d["tracking_b"],
                        "adiabaticity_ratio": d["adia"],
                        "adiabaticity_verdict": "ACCEPTABLE",
                        "n_substeps_completed": 1000,
                    }
                ]
            }
        return {"rows": []}

    client._gql_query = stub_gql_query  # type: ignore[method-assign]

    request = LoopTransportRequest(
        gauge_field_name="halcyon_canonical_buckyball",
        loop=GAMMA_UNIT,
        direction="FORWARD",
        pack=HalcyonParameterPack(),
        seeds=(20260616, 20260617),
        n_discretization=1000,
    )
    result = client.loop_transport(request)

    assert result.tracking_error_max_Q == pytest.approx(0.04)  # max(0.02, 0.04)
    assert result.tracking_error_max_beta_W == pytest.approx(0.03)  # max(0.03, 0.01)
    assert result.adiabaticity_check.tau_pin_over_t_segment == pytest.approx(0.08)


def test_decomposition_run_id_concatenates_subrun_ids():
    """When per-seed sub-calls return run_ids, the aggregated result's
    run_id is a `;`-joined string of the sub-run-ids. Audit trail
    locates each per-seed receipt independently."""
    seed_h = {20260616: 0.7, 20260617: 0.8}
    client = LiveLoopTransportClient(per_seed_thermalize=True, ping=False)
    client._declared_loops[GAMMA_UNIT.name] = GAMMA_UNIT

    def stub_gql_query(query: str):
        if "GIBBS_SAMPLE" in query:
            return {"rows": [{"mean_plaquette": [0.51]}]}
        if "LOOP_TRANSPORT" in query:
            m = re.search(r"SEEDS \[(\d+)\.\.(\d+)\]", query)
            seed = int(m.group(1))
            return {
                "rows": [
                    {
                        "per_seed_h_forward": [seed_h[seed]],
                        "per_seed_h_reversed": [-seed_h[seed]],
                        "h_forward": seed_h[seed],
                        "h_reversed": -seed_h[seed],
                        "sigma_h_blocked": 0.0,
                        "tracking_error_max_q": 0.0,
                        "tracking_error_max_beta_w": 0.0,
                        "adiabaticity_ratio": 0.05,
                        "adiabaticity_verdict": "ACCEPTABLE",
                        "n_substeps_completed": 1000,
                        "run_id": f"run-{seed}",
                    }
                ]
            }
        return {"rows": []}

    client._gql_query = stub_gql_query  # type: ignore[method-assign]

    result = client.loop_transport(
        LoopTransportRequest(
            gauge_field_name="halcyon_canonical_buckyball",
            loop=GAMMA_UNIT,
            direction="FORWARD",
            pack=HalcyonParameterPack(),
            seeds=(20260616, 20260617),
            n_discretization=1000,
        )
    )
    assert result.run_id == "run-20260616;run-20260617"
