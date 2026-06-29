"""
buckyball_graph.py — Combinatorial + geometric data for the truncated icosahedron
(a.k.a. C60 / soccer-ball graph), used as the spatial lattice for 2D-YM on S^2.

Derivation (kept terse; full notes live in inertia_damping/JOURNAL.md):
  V = 60, E = 90, F = 32 (12 pentagons + 20 hexagons), chi = V - E + F = 2.
  Cartesian embedding uses phi = (1+sqrt(5))/2: cyclic perms of
    Orbit A: (0, +/-1, +/-3 phi)            -> 12 vertices
    Orbit B: (+/-1, +/-(2+phi), +/-2 phi)   -> 24 vertices
    Orbit C: (+/-phi, +/-2, +/-(2 phi + 1)) -> 24 vertices
  Sphere radius^2 = 10 + 9 phi; nearest-neighbour edge length^2 = 4 exactly.
  Faces traced via the planar rotation system (CCW neighbour ordering around
  each outward normal) + combinatorial-map next_in_face rule. Edge-type census
  is 60 pent-hex + 30 hex-hex + 0 pent-pent (the prior task brief was wrong
  about every edge being pent-hex; see lattice-research note).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0
EDGE_LEN_SQ = 4.0
SPHERE_R2 = 10.0 + 9.0 * PHI
TOL_DIST = 1e-9


@dataclass
class BuckyballGraph:
    vertex_coords: np.ndarray
    edges: np.ndarray
    faces: List[List[Tuple[int, int]]]
    pentagons: List[int]
    hexagons: List[int]
    adj: List[List[int]] = field(default_factory=list)
    face_vertices: List[Tuple[int, ...]] = field(default_factory=list)

    @property
    def n_vertices(self) -> int:
        return int(self.vertex_coords.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def verify_euler(self) -> bool:
        return self.n_vertices - self.n_edges + self.n_faces == 2

    def verify_three_regular(self) -> bool:
        return all(len(self.adj[v]) == 3 for v in range(self.n_vertices))

    def verify_pent_hex_edges(self) -> bool:
        """Every edge belongs to exactly 2 faces. The truncated icosahedron has
        60 pent-hex + 30 hex-hex + 0 pent-pent edges; the looser claim "every
        edge in exactly 2 faces" is what the gauge theory actually needs.
        """
        seen = [0] * self.n_edges
        for face in self.faces:
            for eidx, _sign in face:
                seen[eidx] += 1
        return all(c == 2 for c in seen)

    def edge_type_census(self) -> Tuple[int, int, int]:
        pent_set = set(self.pentagons)
        hex_set = set(self.hexagons)
        per_edge: List[List[str]] = [[] for _ in range(self.n_edges)]
        for fi, face in enumerate(self.faces):
            tag = "P" if fi in pent_set else ("H" if fi in hex_set else "?")
            for eidx, _sign in face:
                per_edge[eidx].append(tag)
        n_PH = n_HH = n_PP = 0
        for tags in per_edge:
            key = "".join(sorted(tags))
            if key == "HP":
                n_PH += 1
            elif key == "HH":
                n_HH += 1
            elif key == "PP":
                n_PP += 1
        return n_PH, n_HH, n_PP


def _build_vertices() -> np.ndarray:
    verts: List[Tuple[float, float, float]] = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            base = (0.0, s1 * 1.0, s2 * 3.0 * PHI)
            for shift in range(3):
                verts.append(tuple(base[(i - shift) % 3] for i in range(3)))
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                base = (s1 * 1.0, s2 * (2.0 + PHI), s3 * 2.0 * PHI)
                for shift in range(3):
                    verts.append(tuple(base[(i - shift) % 3] for i in range(3)))
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                base = (s1 * PHI, s2 * 2.0, s3 * (2.0 * PHI + 1.0))
                for shift in range(3):
                    verts.append(tuple(base[(i - shift) % 3] for i in range(3)))
    V = np.asarray(verts, dtype=np.float64)
    assert V.shape == (60, 3), V.shape
    r2 = np.einsum("ij,ij->i", V, V)
    assert np.allclose(r2, SPHERE_R2, atol=1e-9), (float(r2.min()), float(r2.max()))
    return V


def _build_edges(V: np.ndarray):
    pairs: List[Tuple[int, int]] = []
    for i in range(60):
        for j in range(i + 1, 60):
            d2 = float(np.sum((V[i] - V[j]) ** 2))
            if abs(d2 - EDGE_LEN_SQ) < TOL_DIST:
                pairs.append((i, j))
    assert len(pairs) == 90, f"got {len(pairs)} edges, expected 90"
    adj: List[List[int]] = [[] for _ in range(60)]
    for i, j in pairs:
        adj[i].append(j)
        adj[j].append(i)
    for v in range(60):
        assert len(adj[v]) == 3, f"vertex {v} has degree {len(adj[v])}"
        adj[v].sort()
    edge_index = {(i, j): k for k, (i, j) in enumerate(pairs)}
    return pairs, adj, edge_index


def _build_rotation_system(V: np.ndarray, adj: List[List[int]]):
    rot: List[Tuple[int, int, int]] = [(0, 0, 0)] * 60
    for v in range(60):
        nv = V[v] / np.linalg.norm(V[v])
        u_first = V[adj[v][0]] - V[v]
        e1 = u_first - np.dot(u_first, nv) * nv
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(nv, e1)
        scored = []
        for u in adj[v]:
            d = V[u] - V[v]
            scored.append((np.arctan2(float(np.dot(d, e2)), float(np.dot(d, e1))), u))
        scored.sort()
        rot[v] = tuple(u for _, u in scored)
    return rot


def _trace_faces(rot):
    def succ(u: int, v: int) -> int:
        ru = rot[u]
        k = ru.index(v)
        return ru[(k + 1) % 3]

    visited = set()
    cycles: List[Tuple[int, ...]] = []
    for v0 in range(60):
        for u0 in rot[v0]:
            if (v0, u0) in visited:
                continue
            cyc: List[int] = []
            v, u = v0, u0
            while True:
                visited.add((v, u))
                cyc.append(v)
                w = succ(u, v)
                v, u = u, w
                if (v, u) == (v0, u0):
                    break
            cycles.append(tuple(cyc))
    assert len(cycles) == 32, len(cycles)
    return cycles


def _orient_outward(V: np.ndarray, cycle: Tuple[int, ...]) -> Tuple[int, ...]:
    centroid = np.mean(V[list(cycle)], axis=0)
    n_out = centroid / np.linalg.norm(centroid)
    n = np.zeros(3)
    k = len(cycle)
    for i in range(k):
        a = V[cycle[i]]
        b = V[cycle[(i + 1) % k]]
        n += np.cross(a, b)
    if float(np.dot(n, n_out)) < 0.0:
        return tuple(reversed(cycle))
    return cycle


def build_truncated_icosahedron() -> BuckyballGraph:
    V = _build_vertices()
    pairs, adj, edge_index = _build_edges(V)
    rot = _build_rotation_system(V, adj)
    raw_cycles = _trace_faces(rot)

    pent_cycles: List[Tuple[int, ...]] = []
    hex_cycles: List[Tuple[int, ...]] = []
    for c in raw_cycles:
        oc = _orient_outward(V, c)
        if len(oc) == 5:
            pent_cycles.append(oc)
        elif len(oc) == 6:
            hex_cycles.append(oc)
        else:
            raise AssertionError(f"unexpected face length {len(oc)}")
    assert len(pent_cycles) == 12 and len(hex_cycles) == 20

    face_vertices: List[Tuple[int, ...]] = []
    faces: List[List[Tuple[int, int]]] = []
    pentagons: List[int] = []
    hexagons: List[int] = []

    def _emit(cyc: Tuple[int, ...]) -> int:
        es: List[Tuple[int, int]] = []
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            if a < b:
                es.append((edge_index[(a, b)], +1))
            else:
                es.append((edge_index[(b, a)], -1))
        fi = len(faces)
        faces.append(es)
        face_vertices.append(cyc)
        return fi

    for c in pent_cycles:
        pentagons.append(_emit(c))
    for c in hex_cycles:
        hexagons.append(_emit(c))

    edge_sign_sum = [0] * len(pairs)
    for face in faces:
        for eidx, s in face:
            edge_sign_sum[eidx] += s
    assert all(v == 0 for v in edge_sign_sum), "face orientation inconsistent"

    edges_arr = np.asarray(pairs, dtype=np.int64)

    g = BuckyballGraph(
        vertex_coords=V,
        edges=edges_arr,
        faces=faces,
        pentagons=pentagons,
        hexagons=hexagons,
        adj=adj,
        face_vertices=face_vertices,
    )

    assert g.verify_euler(), "Euler characteristic != 2"
    assert g.verify_three_regular(), "graph is not 3-regular"
    assert g.verify_pent_hex_edges(), "edge-face incidence broken"
    census = g.edge_type_census()
    assert census == (60, 30, 0), f"edge-type census {census} != (60, 30, 0)"
    return g


if __name__ == "__main__":
    g = build_truncated_icosahedron()
    print(f"V={g.n_vertices} E={g.n_edges} F={g.n_faces} "
          f"pent={len(g.pentagons)} hex={len(g.hexagons)} "
          f"census(PH,HH,PP)={g.edge_type_census()} euler={g.verify_euler()}")
