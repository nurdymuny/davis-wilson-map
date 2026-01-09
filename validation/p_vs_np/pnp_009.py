#!/usr/bin/env python3
"""
NS-012: Algorithm-Induced Embedding Test (v3 - PRODUCTION, FIXED)
=================================================================

Validates Axiom 3.1: Polynomial-time algorithms induce low-Γ embeddings.

Key properties:
- Stable seeds (hashlib-based), multi-seed statistics
- SCC-DAG depth embedding for 2-SAT (algorithm-induced)
- Γ computed on CPU (SciPy): transport-curvature proxy + geodesic distortion
- kNN graph: undirected union + MIN weight across directions (true metric-preserving)
- LCC-only distortion to avoid disconnected artifacts
- Polynomial vs exponential model fits with significance check
- Strategy-wise low-Γ detection

Author: Bee Rosa Davis
Date: January 2026
"""

import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist, pdist

# -----------------------------------------------------------------------------
# Device / environment notes
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
print("Note: Γ computation uses CPU (SciPy) regardless of device")


def stable_hash_int(s: str, mod: int = 100000) -> int:
    """Stable hash that doesn't change between Python runs."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % mod


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# SAT Problem Generators
# =============================================================================

@dataclass
class SATInstance:
    """A SAT instance with n variables and m clauses."""
    n_vars: int
    clauses: List[List[int]]

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)


def generate_random_2sat(n: int, clause_ratio: float = 2.0, seed: Optional[int] = None) -> SATInstance:
    """Generate random 2-SAT instance."""
    if seed is not None:
        np.random.seed(seed)
    m = int(n * clause_ratio)
    clauses: List[List[int]] = []
    for _ in range(m):
        vars_chosen = np.random.choice(n, size=2, replace=False) + 1
        signs = np.random.choice([-1, 1], size=2)
        clause = [int(signs[0] * vars_chosen[0]), int(signs[1] * vars_chosen[1])]
        clauses.append(clause)
    return SATInstance(n_vars=n, clauses=clauses)


def generate_random_3sat(n: int, clause_ratio: float = 4.26, seed: Optional[int] = None) -> SATInstance:
    """Generate random 3-SAT at critical ratio."""
    if seed is not None:
        np.random.seed(seed)
    m = int(n * clause_ratio)
    clauses: List[List[int]] = []
    for _ in range(m):
        vars_chosen = np.random.choice(n, size=3, replace=False) + 1
        signs = np.random.choice([-1, 1], size=3)
        clause = [int(signs[i] * vars_chosen[i]) for i in range(3)]
        clauses.append(clause)
    return SATInstance(n_vars=n, clauses=clauses)


# =============================================================================
# 2-SAT Solver with SCC-DAG Depth Embedding
# =============================================================================

class TwoSATSolver:
    """
    2-SAT solver using implication graph + SCC decomposition.
    Depth computed on SCC DAG (acyclic), not raw cyclic graph.
    """

    def __init__(self, instance: SATInstance):
        self.instance = instance
        self.n = instance.n_vars
        self.graph: Dict[int, List[int]] = defaultdict(list)
        self.reverse_graph: Dict[int, List[int]] = defaultdict(list)
        self._build_implication_graph()

        self.component, self.n_components = self._kosaraju_scc()
        self.scc_dag_depth = self._compute_scc_dag_depth()

    def _lit_to_node(self, lit: int) -> int:
        var = abs(lit) - 1
        return 2 * var if lit > 0 else 2 * var + 1

    def _neg_node(self, node: int) -> int:
        return node ^ 1

    def _build_implication_graph(self) -> None:
        for clause in self.instance.clauses:
            if len(clause) != 2:
                continue
            a, b = clause
            node_a = self._lit_to_node(a)
            node_b = self._lit_to_node(b)
            node_not_a = self._neg_node(node_a)
            node_not_b = self._neg_node(node_b)

            self.graph[node_not_a].append(node_b)
            self.graph[node_not_b].append(node_a)
            self.reverse_graph[node_b].append(node_not_a)
            self.reverse_graph[node_a].append(node_not_b)

    def _kosaraju_scc(self) -> Tuple[List[int], int]:
        """Kosaraju's algorithm for SCC decomposition."""
        n_nodes = 2 * self.n
        visited = [False] * n_nodes
        order: List[int] = []

        def dfs1(start: int) -> None:
            stack: List[Tuple[int, bool]] = [(start, False)]
            while stack:
                v, processed = stack.pop()
                if processed:
                    order.append(v)
                    continue
                if visited[v]:
                    continue
                visited[v] = True
                stack.append((v, True))
                for u in self.graph[v]:
                    if not visited[u]:
                        stack.append((u, False))

        for i in range(n_nodes):
            if not visited[i]:
                dfs1(i)

        component = [-1] * n_nodes
        comp_id = 0

        def dfs2(start: int, cid: int) -> None:
            stack = [start]
            while stack:
                v = stack.pop()
                if component[v] != -1:
                    continue
                component[v] = cid
                for u in self.reverse_graph[v]:
                    if component[u] == -1:
                        stack.append(u)

        for node in reversed(order):
            if component[node] == -1:
                dfs2(node, comp_id)
                comp_id += 1

        return component, comp_id

    def _compute_scc_dag_depth(self) -> Dict[int, int]:
        """Compute depth on SCC DAG (acyclic)."""
        scc_edges: Dict[int, set] = defaultdict(set)
        for u in range(2 * self.n):
            for v in self.graph[u]:
                su = self.component[u]
                sv = self.component[v]
                if su != sv:
                    scc_edges[su].add(sv)

        in_degree = defaultdict(int)
        for scc in range(self.n_components):
            for target in scc_edges[scc]:
                in_degree[target] += 1

        depth = {scc: 0 for scc in range(self.n_components)}
        queue = deque([scc for scc in range(self.n_components) if in_degree[scc] == 0])

        while queue:
            scc = queue.popleft()
            for target in scc_edges[scc]:
                depth[target] = max(depth[target], depth[scc] + 1)
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        return depth

    def solve(self) -> Optional[List[bool]]:
        """Solve 2-SAT instance."""
        assignment = [False] * self.n
        for i in range(self.n):
            node_pos = 2 * i
            node_neg = 2 * i + 1
            if self.component[node_pos] == self.component[node_neg]:
                return None
            assignment[i] = self.component[node_pos] > self.component[node_neg]
        return assignment

    def get_algorithm_embedding(self) -> torch.Tensor:
        """Embedding from algorithm structure (SCC + DAG depth)."""
        dim = 4
        embedding = torch.zeros(self.n, dim, dtype=torch.float64)

        max_depth = max(self.scc_dag_depth.values()) if self.scc_dag_depth else 1

        for i in range(self.n):
            node_pos = 2 * i
            node_neg = 2 * i + 1

            scc_pos = self.component[node_pos] / max(self.n_components, 1)
            scc_neg = self.component[node_neg] / max(self.n_components, 1)
            embedding[i, 0] = scc_pos - scc_neg

            depth_pos = self.scc_dag_depth[self.component[node_pos]]
            depth_neg = self.scc_dag_depth[self.component[node_neg]]
            embedding[i, 1] = (depth_pos - depth_neg) / max(max_depth, 1)

            in_pos = len(self.reverse_graph[node_pos])
            in_neg = len(self.reverse_graph[node_neg])
            embedding[i, 2] = (in_pos - in_neg) / max(in_pos + in_neg, 1)

            out_pos = len(self.graph[node_pos])
            out_neg = len(self.graph[node_neg])
            embedding[i, 3] = (out_pos - out_neg) / max(out_pos + out_neg, 1)

        return embedding


# =============================================================================
# Γ Computation (CPU / SciPy)
# =============================================================================

class GammaComputer:
    """
    Computes Γ functional using a transport-curvature proxy + geodesic distortion.

    All computation on CPU via SciPy for numerical stability.
    """

    def __init__(self, instance: SATInstance, embedding: torch.Tensor):
        self.instance = instance
        self.embedding = embedding.cpu().numpy()
        self.n = instance.n_vars
        self.dim = embedding.shape[1]

    def _adaptive_k(self) -> int:
        """Adaptive k for kNN: k = max(5, ceil(log2(n+1)))."""
        return max(5, int(np.ceil(np.log2(self.n + 1))))

    def compute_knn_graph(self, k: Optional[int] = None) -> csr_matrix:
        """
        Build symmetric kNN graph from embedding.

        Symmetrization policy:
        - Edge exists if either direction exists (union)
        - Undirected weight = min(w_ij, w_ji) if both directions exist
        - Minimum edge weight enforced to avoid degeneracy
        - Final adjacency is symmetric (required for undirected shortest paths)
        """
        if self.n <= 1:
            return csr_matrix((self.n, self.n))

        if k is None:
            k = self._adaptive_k()

        dists = cdist(self.embedding, self.embedding)
        n = self.n
        k = min(k, n - 1)

        MIN_EDGE_WEIGHT = 1e-6

        # Build undirected edge set with min weights across directions.
        # Use undirected key (u,v) where u < v.
        edge_min: Dict[Tuple[int, int], float] = {}

        for i in range(n):
            neigh = np.argsort(dists[i])[1:k + 1]
            for j in neigh:
                if i == j:
                    continue
                u, v = (i, j) if i < j else (j, i)
                w = max(float(dists[i, j]), MIN_EDGE_WEIGHT)
                prev = edge_min.get((u, v))
                if prev is None or w < prev:
                    edge_min[(u, v)] = w

        # Emit symmetric adjacency (both directions) with same undirected weight.
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for (u, v), w in edge_min.items():
            rows.extend([u, v])
            cols.extend([v, u])
            data.extend([w, w])

        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
        return adj

    def compute_neighbor_transport_curvature(self, k: Optional[int] = None) -> np.ndarray:
        """
        Neighbor transport curvature proxy (OR-inspired, not true Ollivier–Ricci).

        κ(x,y) ≈ 1 - (avg transport cost between neighbor sets) / d(x,y)

        Returns per-node average curvature.
        """
        if self.n < 3:
            return np.zeros(self.n)

        if k is None:
            k = self._adaptive_k()

        dists = cdist(self.embedding, self.embedding)
        k = min(k, self.n - 1)

        neighbors: List[np.ndarray] = []
        for i in range(self.n):
            nn = np.argsort(dists[i])[1:k + 1]
            neighbors.append(nn)

        node_curvatures = np.zeros(self.n, dtype=np.float64)
        node_counts = np.zeros(self.n, dtype=np.float64)

        for i in range(self.n):
            for j in neighbors[i]:
                if i >= j:
                    continue

                d_ij = float(dists[i, j])
                if d_ij < 1e-10:
                    continue

                ni = neighbors[i]
                nj = neighbors[j]

                cross = dists[np.ix_(ni, nj)]
                w1_approx = 0.5 * (cross.min(axis=1).mean() + cross.min(axis=0).mean())
                kappa = 1.0 - float(w1_approx) / d_ij

                node_curvatures[i] += kappa
                node_curvatures[j] += kappa
                node_counts[i] += 1
                node_counts[j] += 1

        node_counts = np.maximum(node_counts, 1.0)
        return node_curvatures / node_counts

    def compute_geodesic_distortion(self, k: Optional[int] = None) -> Tuple[float, int]:
        """
        Distortion between Euclidean and graph-geodesic distances.

        - Computes within largest connected component only.
        - Uses larger k than curvature for better connectivity.

        Returns: (distortion_std, lcc_size).
        """
        if self.n < 3:
            return 0.0, self.n

        if k is None:
            k = max(8, int(np.ceil(2.0 * np.log2(self.n + 1))))

        euclidean = cdist(self.embedding, self.embedding)
        adj = self.compute_knn_graph(k)

        n_components, labels = connected_components(adj, directed=False)
        if n_components > 1:
            sizes = np.bincount(labels)
            largest = int(np.argmax(sizes))
            mask_nodes = labels == largest
            lcc_size = int(sizes[largest])

            idx = np.where(mask_nodes)[0]
            adj_lcc = adj[np.ix_(idx, idx)]
            euclidean_lcc = euclidean[np.ix_(idx, idx)]
        else:
            lcc_size = self.n
            adj_lcc = adj
            euclidean_lcc = euclidean

        geodesic = shortest_path(adj_lcc, directed=False)

        mask = np.isfinite(geodesic) & (euclidean_lcc > 1e-10) & (geodesic > 1e-10)
        if mask.sum() < 10:
            return 0.0, lcc_size

        ratios = geodesic[mask] / euclidean_lcc[mask]
        distortion = float(np.std(ratios))
        return distortion, lcc_size

    def compute_gamma(self) -> Dict[str, float]:
        """Compute Γ using transport curvature proxy + geodesic distortion."""
        k_curv = self._adaptive_k()

        curvatures = self.compute_neighbor_transport_curvature(k_curv)
        mean_curvature = float(np.abs(curvatures).mean())
        total_curvature = float(np.abs(curvatures).sum())

        distortion, lcc_size = self.compute_geodesic_distortion()

        if self.n > 1:
            pairwise = pdist(self.embedding)
            diameter = float(pairwise.max()) if pairwise.size else 0.0
        else:
            diameter = 0.0

        if self.n > self.dim and self.n > 1:
            centered = self.embedding - self.embedding.mean(axis=0, keepdims=True)
            cov = (centered.T @ centered) / float(self.n)
            eig = np.linalg.eigvalsh(cov)
            eig = np.maximum(eig, 0.0)
            total_var = float(eig.sum())
            if total_var > 1e-10:
                eff_dim = float((total_var ** 2) / float((eig ** 2).sum()))
            else:
                eff_dim = 1.0
        else:
            eff_dim = float(min(self.n, self.dim))

        gamma_total = (total_curvature + distortion * float(self.n)) * diameter / max(eff_dim, 1.0)
        connectivity = float(lcc_size) / float(self.n)

        return {
            "gamma_total": float(gamma_total),
            "mean_curvature": mean_curvature,
            "total_curvature": total_curvature,
            "distortion": float(distortion),
            "diameter": float(diameter),
            "effective_dim": float(eff_dim),
            "connectivity": float(connectivity),
            "lcc_size": int(lcc_size),
            "k_used_curv": int(k_curv),
        }


# =============================================================================
# Embedding Strategies for 3-SAT
# =============================================================================

class EmbeddingStrategies:
    """Different embedding strategies to test for 3-SAT."""

    @staticmethod
    def random_embedding(instance: SATInstance, dim: int = 10) -> torch.Tensor:
        return torch.randn(instance.n_vars, dim, dtype=torch.float64)

    @staticmethod
    def spectral_embedding(instance: SATInstance, dim: int = 10) -> torch.Tensor:
        n = instance.n_vars
        adj = np.zeros((n, n), dtype=np.float64)

        for clause in instance.clauses:
            vars_in_clause = [abs(lit) - 1 for lit in clause]
            for i, v1 in enumerate(vars_in_clause):
                for v2 in vars_in_clause[i + 1:]:
                    adj[v1, v2] += 1
                    adj[v2, v1] += 1

        degree = adj.sum(axis=1)
        laplacian = np.diag(degree) - adj

        try:
            _, eigvecs = np.linalg.eigh(laplacian)
            k = min(dim, max(n - 1, 1))
            emb = eigvecs[:, 1:k + 1]  # skip constant
            if emb.shape[1] < dim:
                emb = np.hstack([emb, np.zeros((n, dim - emb.shape[1]))])
        except Exception:
            emb = np.random.randn(n, dim)

        return torch.tensor(emb, dtype=torch.float64)

    @staticmethod
    def clause_structure_embedding(instance: SATInstance, dim: int = 10) -> torch.Tensor:
        n = instance.n_vars
        m = len(instance.clauses)
        participation = np.zeros((n, m), dtype=np.float64)

        for c_idx, clause in enumerate(instance.clauses):
            for lit in clause:
                var = abs(lit) - 1
                participation[var, c_idx] = 1.0 if lit > 0 else -1.0

        try:
            U, S, _ = np.linalg.svd(participation, full_matrices=False)
            k = min(dim, U.shape[1])
            emb = U[:, :k] * S[:k]
            if emb.shape[1] < dim:
                emb = np.hstack([emb, np.zeros((n, dim - emb.shape[1]))])
        except Exception:
            emb = np.random.randn(n, dim)

        return torch.tensor(emb, dtype=torch.float64)

    @staticmethod
    def greedy_embedding(instance: SATInstance, dim: int = 10) -> torch.Tensor:
        n = instance.n_vars
        influence = np.zeros(n, dtype=np.float64)
        polarity = np.zeros(n, dtype=np.float64)

        for clause in instance.clauses:
            for lit in clause:
                var = abs(lit) - 1
                influence[var] += 1
                polarity[var] += 1 if lit > 0 else -1

        order = np.argsort(influence)[::-1]
        emb = np.zeros((n, dim), dtype=np.float64)
        for rank, var in enumerate(order):
            emb[var, 0] = rank / max(n, 1)
            emb[var, 1] = influence[var] / max(influence.max(), 1.0)
            emb[var, 2] = polarity[var] / max(influence[var], 1.0)

        for d in range(3, dim):
            emb[:, d] = np.sin(emb[:, 0] * d * np.pi)

        return torch.tensor(emb, dtype=torch.float64)


# =============================================================================
# Model Fitting with Proper Statistics
# =============================================================================

def fit_scaling_models(ns: np.ndarray, gammas: np.ndarray) -> Dict[str, Any]:
    """
    Fit both polynomial and exponential models, compare R².
    Also checks if exponential rate is statistically significant.
    """
    valid = (gammas > 0) & np.isfinite(gammas) & (ns > 0)
    ns = ns[valid]
    gammas = gammas[valid]

    if len(ns) < 3:
        return {
            "polynomial_exp": 0.0,
            "exponential_rate": 0.0,
            "poly_r2": 0.0,
            "exp_r2": 0.0,
            "best_model": "insufficient_data",
            "exp_significant": False,
            "poly_stderr": np.nan,
            "exp_stderr": np.nan,
        }

    log_n = np.log(ns)
    log_gamma = np.log(gammas)

    poly_slope, _, poly_r, _, poly_stderr = stats.linregress(log_n, log_gamma)
    poly_r2 = float(poly_r ** 2)

    exp_slope, _, exp_r, _, exp_stderr = stats.linregress(ns, log_gamma)
    exp_r2 = float(exp_r ** 2)

    exp_significant = bool(abs(exp_slope) > 2.0 * exp_stderr) if exp_stderr and exp_stderr > 0 else False

    if exp_r2 > poly_r2 + 0.05 and exp_significant:
        best_model = "exponential"
    elif poly_r2 > exp_r2 + 0.05:
        best_model = "polynomial"
    elif exp_r2 > poly_r2 and not exp_significant:
        best_model = "polynomial"
    else:
        best_model = "ambiguous"

    return {
        "polynomial_exp": float(poly_slope),
        "poly_r2": float(poly_r2),
        "poly_stderr": float(poly_stderr) if poly_stderr is not None else np.nan,
        "exponential_rate": float(exp_slope),
        "exp_r2": float(exp_r2),
        "exp_stderr": float(exp_stderr) if exp_stderr is not None else np.nan,
        "exp_significant": exp_significant,
        "best_model": best_model,
    }


def is_low_gamma_scaling(scaling: Dict[str, Any]) -> bool:
    """
    STRICT: Low-Γ = polynomial with small exponent and good fit.
    Ambiguous requires higher R².
    """
    if scaling["best_model"] == "insufficient_data":
        return False

    small_exponent = scaling["polynomial_exp"] <= 1.5

    if scaling["best_model"] == "polynomial":
        return small_exponent and scaling["poly_r2"] >= 0.8
    if scaling["best_model"] == "ambiguous":
        return small_exponent and scaling["poly_r2"] >= 0.9
    return False


# =============================================================================
# Main Tests
# =============================================================================

def test_2sat_algorithm_embedding(
    n_range: List[int] = [20, 40, 80, 160, 320],
    n_seeds: int = 5
) -> Dict[str, Any]:
    print("=" * 70)
    print("Test 1: 2-SAT Algorithm-Induced Embedding")
    print("Expectation: Γ = O(n^k) for small k (polynomial)")
    print(f"Seeds per n: {n_seeds}")
    print("=" * 70)

    results = []
    for n in n_range:
        gammas = []
        connectivities = []
        for seed in range(n_seeds):
            full_seed = seed * 10000 + n
            set_seed(full_seed)

            instance = generate_random_2sat(n, seed=full_seed)
            solver = TwoSATSolver(instance)
            emb = solver.get_algorithm_embedding()

            g = GammaComputer(instance, emb).compute_gamma()
            gammas.append(g["gamma_total"])
            connectivities.append(g["connectivity"])

        gammas = np.array(gammas, dtype=np.float64)
        conn = float(np.mean(connectivities))
        print(f"  n={n:4d}: Γ = {gammas.mean():.2f} ± {gammas.std():.2f} (conn={conn:.2f})")

        results.append({
            "n": n,
            "gamma_mean": float(gammas.mean()),
            "gamma_std": float(gammas.std()),
            "connectivity": conn,
        })

    ns = np.array([r["n"] for r in results], dtype=np.float64)
    gmeans = np.array([r["gamma_mean"] for r in results], dtype=np.float64)

    scaling = fit_scaling_models(ns, gmeans)

    print(f"\n  Polynomial:  Γ ~ n^{scaling['polynomial_exp']:.2f} (R²={scaling['poly_r2']:.3f})")
    print(f"  Exponential: Γ ~ exp({scaling['exponential_rate']:.4f} n) (R²={scaling['exp_r2']:.3f})")
    print(f"  Best model: {scaling['best_model']}")

    is_polynomial = scaling["best_model"] in ["polynomial", "ambiguous"] and scaling["polynomial_exp"] < 3.0
    print(f"\n  Result: {'✓ POLYNOMIAL' if is_polynomial else '✗ NOT POLYNOMIAL'}")

    return {"results": results, "scaling": scaling, "is_polynomial": is_polynomial}


def test_3sat_multiple_embeddings(
    n_range: List[int] = [20, 40, 80, 160, 320],
    n_seeds: int = 5
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("Test 2: 3-SAT Multiple Embedding Strategies")
    print("Checking if ANY strategy achieves low Γ")
    print(f"Seeds per n: {n_seeds}")
    print("=" * 70)

    strategies = {
        "random": EmbeddingStrategies.random_embedding,
        "spectral": EmbeddingStrategies.spectral_embedding,
        "clause": EmbeddingStrategies.clause_structure_embedding,
        "greedy": EmbeddingStrategies.greedy_embedding,
    }

    all_results: Dict[str, List[Dict[str, float]]] = {name: [] for name in strategies}

    for n in n_range:
        print(f"\n  n = {n}:")
        for name, strat in strategies.items():
            gammas = []
            conns = []
            for seed in range(n_seeds):
                full_seed = seed * 10000 + n + stable_hash_int(name)
                set_seed(full_seed)

                instance = generate_random_3sat(n, seed=full_seed)
                emb = strat(instance)
                g = GammaComputer(instance, emb).compute_gamma()
                gammas.append(g["gamma_total"])
                conns.append(g["connectivity"])

            gammas = np.array(gammas, dtype=np.float64)
            conn = float(np.mean(conns))
            print(f"    {name:12s}: Γ = {gammas.mean():8.2f} ± {gammas.std():5.2f} (conn={conn:.2f})")

            all_results[name].append({
                "n": float(n),
                "gamma_mean": float(gammas.mean()),
                "gamma_std": float(gammas.std()),
                "connectivity": conn,
            })

    print("\n  Scaling Analysis:")
    scalings: Dict[str, Dict[str, Any]] = {}
    low_gamma_candidates: List[str] = []

    for name, rows in all_results.items():
        ns = np.array([r["n"] for r in rows], dtype=np.float64)
        gmeans = np.array([r["gamma_mean"] for r in rows], dtype=np.float64)
        scaling = fit_scaling_models(ns, gmeans)
        scalings[name] = scaling

        is_low = is_low_gamma_scaling(scaling)
        if is_low:
            low_gamma_candidates.append(name)

        status = "⚠ LOW-Γ" if is_low else ""
        print(
            f"    {name:12s}: n^{scaling['polynomial_exp']:.2f} (R²={scaling['poly_r2']:.2f}) | "
            f"exp({scaling['exponential_rate']:.3f}n) (R²={scaling['exp_r2']:.2f}) | "
            f"{scaling['best_model']} {status}"
        )

    no_low_gamma_found = len(low_gamma_candidates) == 0
    print(f"\n  Low-Γ candidates found: {low_gamma_candidates if low_gamma_candidates else 'None'}")
    print(f"\n  Result: {'✓ NO LOW-Γ EMBEDDING FOUND' if no_low_gamma_found else '✗ LOW-Γ EMBEDDING EXISTS'}")

    return {
        "all_results": all_results,
        "scalings": scalings,
        "low_gamma_candidates": low_gamma_candidates,
        "no_low_gamma_found": no_low_gamma_found,
    }


def test_separation(
    n_range: List[int] = [20, 40, 80, 160, 320],
    n_seeds: int = 5
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("Test 3: P vs NP-complete Separation")
    print("=" * 70)

    results_2sat = []
    results_3sat = []

    for n in n_range:
        g2 = []
        for seed in range(n_seeds):
            full_seed = seed * 10000 + n
            set_seed(full_seed)
            instance = generate_random_2sat(n, seed=full_seed)
            emb = TwoSATSolver(instance).get_algorithm_embedding()
            g = GammaComputer(instance, emb).compute_gamma()
            g2.append(g["gamma_total"])

        g3 = []
        for seed in range(n_seeds):
            full_seed = seed * 10000 + n + 999999
            set_seed(full_seed)
            instance = generate_random_3sat(n, seed=full_seed)
            emb = EmbeddingStrategies.spectral_embedding(instance)
            g = GammaComputer(instance, emb).compute_gamma()
            g3.append(g["gamma_total"])

        g2_mean = float(np.mean(g2))
        g3_mean = float(np.mean(g3))
        ratio = g3_mean / max(g2_mean, 1e-8)

        print(f"  n={n:4d}: 2-SAT Γ={g2_mean:8.2f}, 3-SAT Γ={g3_mean:8.2f}, ratio={ratio:.1f}x")

        results_2sat.append({"n": float(n), "gamma": g2_mean, "std": float(np.std(g2))})
        results_3sat.append({"n": float(n), "gamma": g3_mean, "std": float(np.std(g3))})

    ns = np.array([r["n"] for r in results_2sat], dtype=np.float64)

    scaling_2 = fit_scaling_models(ns, np.array([r["gamma"] for r in results_2sat], dtype=np.float64))
    scaling_3 = fit_scaling_models(ns, np.array([r["gamma"] for r in results_3sat], dtype=np.float64))

    print(f"\n  2-SAT: Γ ~ n^{scaling_2['polynomial_exp']:.2f} ({scaling_2['best_model']})")
    print(f"  3-SAT: Γ ~ n^{scaling_3['polynomial_exp']:.2f} ({scaling_3['best_model']})")

    exp_gap = float(scaling_3["polynomial_exp"] - scaling_2["polynomial_exp"])
    separation = exp_gap > 0.5 or scaling_3["best_model"] == "exponential"

    print(f"\n  Exponent gap: {exp_gap:.2f}")
    print(f"  Result: {'✓ SEPARATION DETECTED' if separation else '✗ NO CLEAR SEPARATION'}")

    return {
        "results_2sat": results_2sat,
        "results_3sat": results_3sat,
        "scaling_2sat": scaling_2,
        "scaling_3sat": scaling_3,
        "exp_gap": exp_gap,
        "separation": separation,
    }


def run_full_test() -> Dict[str, Any]:
    print("=" * 70)
    print("NS-012: Algorithm-Induced Embedding Test (v3 - PRODUCTION, FIXED)")
    print("Validating Axiom 3.1 for P ≠ NP")
    print("=" * 70)
    print()

    start = time.time()

    result_2sat = test_2sat_algorithm_embedding()
    result_3sat = test_3sat_multiple_embeddings()
    result_sep = test_separation()

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("FINAL RESULTS - NS-012")
    print("=" * 70)

    test1_pass = bool(result_2sat["is_polynomial"])
    test2_pass = bool(result_3sat["no_low_gamma_found"])
    test3_pass = bool(result_sep["separation"])

    print(f"  Test 1 (2-SAT has polynomial Γ):              {'✓ PASS' if test1_pass else '✗ FAIL'}")
    print(f"  Test 2 (No low-Γ 3-SAT embedding found):      {'✓ PASS' if test2_pass else '✗ FAIL'}")
    print(f"  Test 3 (2-SAT vs 3-SAT separation):           {'✓ PASS' if test3_pass else '✗ FAIL'}")

    all_pass = test1_pass and test2_pass and test3_pass

    if all_pass:
        print("\n  ✓ AXIOM 3.1 SUPPORTED BY EVIDENCE")
        print("    - Poly-time algorithm (2-SAT) induces low-Γ embedding")
        print("    - No tested embedding achieves low Γ for 3-SAT")
        print("    - Clear scaling separation between P and NP-complete")
        print("\n    Note: This supports but does not prove Axiom 3.1.")
        print("    Proof requires showing NO embedding can achieve low Γ for NP-complete.")
    else:
        print("\n  ⚠ AXIOM 3.1 NEEDS INVESTIGATION")

    print(f"\n  Time: {elapsed:.1f}s")
    print("=" * 70)

    return {
        "passed": all_pass,
        "result_2sat": result_2sat,
        "result_3sat": result_3sat,
        "result_separation": result_sep,
        "elapsed": float(elapsed),
    }


if __name__ == "__main__":
    results = run_full_test()
