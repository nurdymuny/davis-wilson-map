"""MatchedRNGMockGIGIClient — the byte-identity demonstration mock.

Same shape as ``MockGIGIClient`` except every random-field initializer
(HAAR_RANDOM, MAXWELL_BOLTZMANN) routes through ``matched_rng``'s port
of GIGI's xorshift64* + Marsaglia algorithm. At the same seed, the
resulting link buffer is BYTE-IDENTICAL to what the live GIGI engine
produces.

This is the OPT-IN third receipt. The default mock ``MockGIGIClient``
keeps NumPy PCG64 (per CSPRNG decision (c) — two independent CSPRNGs
catching bugs is a feature). The matched mock proves byte-identity
is achievable when the chapter / spec wants the strongest possible
form of agreement.

GIBBS_SAMPLE and SYMPLECTIC_FLOW are NOT yet byte-matched — they
delegate to the kernel which uses its own RNG path for the heatbath
+ leapfrog. Adding byte-identity for the dynamical path is a
separate sprint (would require porting GIGI's Kennedy-Pendleton kernel
exactly). For the matched-RNG demo, byte-identity on field
initialization is enough to prove the principle.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from inertia_damping.gigi_client.mock import MockGIGIClient
from inertia_damping.gigi_client.protocol import (
    EFieldInit,
    EFieldHandle,
    GaugeFieldHandle,
    GaugeFieldInit,
    Group,
)
from inertia_damping.gigi_client.matched_rng import (
    haar_random_links,
    maxwell_boltzmann_links,
)


class MatchedRNGMockGIGIClient(MockGIGIClient):
    """Same as MockGIGIClient with HAAR_RANDOM + MAXWELL_BOLTZMANN
    routed through GIGI's xorshift64* + Marsaglia path."""

    def declare_gauge_field(
        self,
        name: str,
        lattice_name: str,
        group: Group,
        init: GaugeFieldInit,
        seed: Optional[int] = None,
        from_field: Optional[str] = None,
    ) -> GaugeFieldHandle:
        # Intercept HAAR_RANDOM; everything else falls through.
        if init == GaugeFieldInit.HAAR_RANDOM:
            if seed is None:
                raise ValueError("INIT HAAR_RANDOM requires SEED")
            if lattice_name not in self._lattices:
                raise ValueError(f"LATTICE {lattice_name!r} not declared")
            if group != Group.SU2:
                raise NotImplementedError(
                    "MatchedRNGMockGIGIClient supports SU(2) only at launch"
                )
            lat = self._lattices[lattice_name]
            buf = haar_random_links(lat.graph.n_edges, seed=int(seed))
            handle = GaugeFieldHandle(
                name=name, lattice_name=lattice_name, group=group,
                repr_dim=4, init_kind=init, init_seed=seed,
            )
            # Reuse parent's _GaugeFieldState container shape
            from inertia_damping.gigi_client.mock import _GaugeFieldState
            self._fields[name] = _GaugeFieldState(handle=handle, link_buffer=buf)
            return handle
        return super().declare_gauge_field(
            name=name, lattice_name=lattice_name, group=group,
            init=init, seed=seed, from_field=from_field,
        )

    def declare_e_field(
        self,
        name: str,
        gauge_field: str,
        init: EFieldInit,
        beta: Optional[float] = None,
        seed: Optional[int] = None,
        from_field: Optional[str] = None,
    ) -> EFieldHandle:
        if init == EFieldInit.MAXWELL_BOLTZMANN:
            if seed is None:
                raise ValueError("INIT MAXWELL_BOLTZMANN requires SEED")
            if beta is None:
                raise ValueError("INIT MAXWELL_BOLTZMANN requires BETA")
            u_state = self._field_or_raise(gauge_field)
            lat = self._lattices[u_state.handle.lattice_name]
            buf = maxwell_boltzmann_links(
                lat.graph.n_edges, seed=int(seed), beta=float(beta),
            )
            handle = EFieldHandle(
                name=name, source_gauge_field=gauge_field,
                source_lattice=u_state.handle.lattice_name,
                init_kind=init, init_beta=beta, init_seed=seed,
            )
            from inertia_damping.gigi_client.mock import _EFieldState
            self._e_fields[name] = _EFieldState(handle=handle, buffer=buf)
            return handle
        return super().declare_e_field(
            name=name, gauge_field=gauge_field, init=init,
            beta=beta, seed=seed, from_field=from_field,
        )
