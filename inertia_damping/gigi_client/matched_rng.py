"""Halcyon's xorshift64* SmallRng — byte-for-byte port of GIGI's
``src/gauge/marsaglia_haar.rs::SmallRng``.

This module exists for the matched-RNG receipt: when Halcyon's mock
client is configured with ``MatchedRNGMockGIGIClient``, every random
draw (HAAR_RANDOM, MAXWELL_BOLTZMANN) goes through the SAME xorshift64*
stream as GIGI's Rust engine. Same seed -> byte-identical buffer.

By default Halcyon's MockGIGIClient uses NumPy PCG64, and CSPRNG
decision (c) explicitly drops mock-vs-live byte-equality for that
path. The matched-RNG path is an OPT-IN third receipt: it doesn't
replace the "tolerance-band agreement is the architectural guarantee"
contract; it adds a "look, the strongest possible agreement is
achievable when wanted" demonstration alongside it.

Source of truth: ``gigi/src/gauge/marsaglia_haar.rs``. Any change to
the Rust algorithm requires a mirror here, and the byte-identity
test ``test_gigi_matched_rng.py::test_matched_rng_haar_byte_identical_to_live``
is the receipt that catches divergence.
"""
from __future__ import annotations

import math
import sys
from typing import Tuple

import torch


_U64_MASK = (1 << 64) - 1
_TWO53 = float(1 << 53)
_MULT = 0x2545_F491_4F6C_DD1D


class XorshiftSmallRng:
    """xorshift64* PRNG byte-for-byte mirroring GIGI's SmallRng.

    Algorithm (line-for-line):
        state = max(1, seed)            # avoid zero-state lockup
        next_u64:
            x = state
            x ^= x >> 12
            x ^= x << 25
            x ^= x >> 27
            state = x
            return (x * 0x2545_F491_4F6C_DD1D) & 0xFFFFFFFFFFFFFFFF
        uniform = (next_u64() >> 11) / 2**53   # 53-bit precision in [0, 1)
    """

    __slots__ = ("state", "draws")

    def __init__(self, seed: int):
        self.state = max(1, int(seed) & _U64_MASK)
        self.draws = 0

    def next_u64(self) -> int:
        x = self.state
        x ^= (x >> 12) & _U64_MASK
        x = (x ^ ((x << 25) & _U64_MASK)) & _U64_MASK
        x ^= (x >> 27) & _U64_MASK
        self.state = x
        return (x * _MULT) & _U64_MASK

    def uniform(self) -> float:
        """Uniform in [0, 1) with 53 bits of precision. Matches the
        Rust ``(self.next_u64() >> 11) as f64 / (1u64 << 53) as f64``."""
        self.draws = (self.draws + 1) & _U64_MASK
        return float(self.next_u64() >> 11) / _TWO53


def haar_random_su2(rng: XorshiftSmallRng) -> Tuple[float, float, float, float]:
    """Marsaglia 4-uniforms-with-rejection — byte-for-byte mirror of
    ``gauge::marsaglia_haar::haar_random_su2``.

    Consumes RNG state for every draw including rejected ones; this
    is the bit-identity invariant the Part-II gold gate pins.
    """
    while True:
        x1 = 2.0 * rng.uniform() - 1.0
        x2 = 2.0 * rng.uniform() - 1.0
        s1 = x1 * x1 + x2 * x2
        if s1 < 1.0:
            break
    while True:
        x3 = 2.0 * rng.uniform() - 1.0
        x4 = 2.0 * rng.uniform() - 1.0
        s2 = x3 * x3 + x4 * x4
        if s2 < 1.0:
            break
    factor = math.sqrt((1.0 - s1) / s2)
    return (x1, x2, x3 * factor, x4 * factor)


def maxwell_boltzmann_su2(
    rng: XorshiftSmallRng, beta: float,
) -> Tuple[float, float, float, float]:
    """Maxwell-Boltzmann su(2) Lie sample — byte-for-byte mirror of
    ``gauge::marsaglia_haar::maxwell_boltzmann_su2``.

    Per-edge draw cadence is exactly 4 uniforms — the discarded g4
    is REQUIRED so the RNG state advance matches GIGI's regardless
    of which g_k components we keep.
    """
    sigma = math.sqrt(1.0 / (beta * 1.5))
    # Pair 1: g1, g2
    u1 = max(rng.uniform(), sys.float_info.min)
    u2 = rng.uniform()
    r1 = math.sqrt(-2.0 * math.log(u1))
    theta1 = 2.0 * math.pi * u2
    g1 = r1 * math.cos(theta1)
    g2 = r1 * math.sin(theta1)
    # Pair 2: g3 (g4 discarded for byte-identity)
    u3 = max(rng.uniform(), sys.float_info.min)
    u4 = rng.uniform()
    r2 = math.sqrt(-2.0 * math.log(u3))
    theta2 = 2.0 * math.pi * u4
    g3 = r2 * math.cos(theta2)
    _g4_discarded = r2 * math.sin(theta2)
    return (0.0, sigma * g1, sigma * g2, sigma * g3)


def haar_random_links(n_edges: int, seed: int) -> torch.Tensor:
    """Byte-identical replacement for ``mock._haar_random_links`` that
    uses GIGI's xorshift64* stream + Marsaglia algorithm."""
    rng = XorshiftSmallRng(seed)
    out = torch.empty((n_edges, 4), dtype=torch.float64)
    for e in range(n_edges):
        q = haar_random_su2(rng)
        out[e, 0] = q[0]
        out[e, 1] = q[1]
        out[e, 2] = q[2]
        out[e, 3] = q[3]
    return out


def maxwell_boltzmann_links(
    n_edges: int, seed: int, beta: float,
) -> torch.Tensor:
    """Byte-identical Maxwell-Boltzmann E-field init using GIGI's stream."""
    rng = XorshiftSmallRng(seed)
    out = torch.empty((n_edges, 4), dtype=torch.float64)
    for e in range(n_edges):
        e_vec = maxwell_boltzmann_su2(rng, beta)
        out[e, 0] = e_vec[0]
        out[e, 1] = e_vec[1]
        out[e, 2] = e_vec[2]
        out[e, 3] = e_vec[3]
    return out
