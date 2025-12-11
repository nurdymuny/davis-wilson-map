"""
PNP-004: 3-SAT Phase Transition via Geometric Roughness
========================================================

OBJECTIVE:
  Predict the famous 3-SAT phase transition at clause/variable ratio α ≈ 4.267
  using the Davis geometric framework.

BACKGROUND:
  Random 3-SAT exhibits a sharp phase transition:
  - α < 4.267: Almost all instances are SAT (satisfiable)
  - α > 4.267: Almost all instances are UNSAT
  
  This is one of the most well-studied phenomena in computational complexity.
  The transition point has been computed to high precision: α_c ≈ 4.267

DAVIS FRAMEWORK CONNECTION:
  Δ measures "geometric roughness" of the solution space:
  - Low Δ: Smooth landscape, solutions easy to find (P-like)
  - High Δ: Rough landscape, solutions hard to find (NP-hard)
  
  Hypothesis: Δ should diverge near α_c, signaling the phase transition.

VALIDATION:
  1. Generate random 3-SAT instances at various α
  2. Compute geometric roughness Δ for each
  3. Find where Δ diverges or changes sharply
  4. Compare transition point with known α_c ≈ 4.267

Author: B. Davis
Date: December 11, 2025
Test: PNP-004 from millennium_validation_plan.md
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
from itertools import product


@dataclass
class SATInstance:
    """A 3-SAT instance."""
    n_vars: int
    clauses: List[Tuple[int, int, int]]  # Each clause is 3 literals (signed ints)
    
    @property
    def n_clauses(self):
        return len(self.clauses)
    
    @property
    def alpha(self):
        return self.n_clauses / self.n_vars


@dataclass 
class PhaseTransitionResult:
    """Results from phase transition analysis."""
    alphas: np.ndarray
    delta_values: np.ndarray
    sat_fractions: np.ndarray
    predicted_alpha_c: float
    known_alpha_c: float
    error: float


def generate_random_3sat(n_vars: int, n_clauses: int) -> SATInstance:
    """
    Generate a random 3-SAT instance.
    
    Each clause has 3 distinct variables, each negated with probability 0.5.
    """
    clauses = []
    for _ in range(n_clauses):
        # Pick 3 distinct variables
        vars = np.random.choice(n_vars, size=3, replace=False) + 1
        # Negate each with probability 0.5
        signs = np.random.choice([-1, 1], size=3)
        clause = tuple(int(v * s) for v, s in zip(vars, signs))
        clauses.append(clause)
    
    return SATInstance(n_vars=n_vars, clauses=clauses)


def evaluate_clause(clause: Tuple[int, int, int], assignment: np.ndarray) -> bool:
    """Evaluate a clause given a variable assignment."""
    for lit in clause:
        var_idx = abs(lit) - 1
        val = assignment[var_idx]
        if lit > 0 and val:
            return True
        if lit < 0 and not val:
            return True
    return False


def count_satisfied_clauses(instance: SATInstance, assignment: np.ndarray) -> int:
    """Count how many clauses are satisfied by an assignment."""
    return sum(evaluate_clause(c, assignment) for c in instance.clauses)


def is_satisfiable_brute_force(instance: SATInstance, max_vars: int = 20) -> Optional[bool]:
    """
    Check satisfiability by brute force (only for small instances).
    Returns None if instance is too large.
    """
    if instance.n_vars > max_vars:
        return None
    
    for bits in product([False, True], repeat=instance.n_vars):
        assignment = np.array(bits)
        if count_satisfied_clauses(instance, assignment) == instance.n_clauses:
            return True
    return False


def compute_solution_space_roughness(instance: SATInstance, n_samples: int = 1000) -> float:
    """
    Compute Davis Δ: geometric roughness of the solution space.
    
    Method: Sample random assignments, compute the "energy" (unsatisfied clauses).
    Δ = variance of energy landscape + gradient roughness.
    
    This measures how "jagged" the optimization landscape is.
    """
    n_vars = instance.n_vars
    
    energies = []
    gradients = []
    
    for _ in range(n_samples):
        # Random assignment
        assignment = np.random.randint(0, 2, size=n_vars).astype(bool)
        energy = instance.n_clauses - count_satisfied_clauses(instance, assignment)
        energies.append(energy)
        
        # Compute local gradient (flip each variable, measure energy change)
        local_gradient = []
        for i in range(n_vars):
            flipped = assignment.copy()
            flipped[i] = not flipped[i]
            new_energy = instance.n_clauses - count_satisfied_clauses(instance, flipped)
            local_gradient.append(abs(new_energy - energy))
        
        gradients.append(np.mean(local_gradient))
    
    energies = np.array(energies)
    gradients = np.array(gradients)
    
    # Δ combines:
    # 1. Energy variance (how spread out are the energies?)
    # 2. Gradient magnitude (how steep is the landscape?)
    # 3. Gradient variance (how unpredictable are the gradients?)
    
    energy_var = np.var(energies) / (instance.n_clauses + 1)
    gradient_mean = np.mean(gradients)
    gradient_var = np.var(gradients)
    
    # Normalize by problem size
    delta = (energy_var + gradient_mean + gradient_var) / instance.n_vars
    
    return delta


def compute_solution_space_delta_spectral(instance: SATInstance, n_samples: int = 500) -> float:
    """
    Compute Δ using spectral properties of the solution space.
    
    Build a local neighborhood graph and measure its spectral gap.
    Small spectral gap = rough landscape = high Δ.
    """
    n_vars = instance.n_vars
    
    # Sample random assignments and their neighbors
    samples = []
    for _ in range(n_samples):
        assignment = np.random.randint(0, 2, size=n_vars).astype(bool)
        energy = instance.n_clauses - count_satisfied_clauses(instance, assignment)
        samples.append((assignment.copy(), energy))
    
    # Compute energy correlations between nearby points
    # This measures the "smoothness" of the landscape
    
    correlations = []
    for i in range(min(200, len(samples))):
        base_assignment, base_energy = samples[i]
        
        # Sample neighbors (1-flip distance)
        neighbor_energies = []
        for j in range(n_vars):
            neighbor = base_assignment.copy()
            neighbor[j] = not neighbor[j]
            neighbor_energy = instance.n_clauses - count_satisfied_clauses(instance, neighbor)
            neighbor_energies.append(neighbor_energy)
        
        # Correlation: how predictable are neighbor energies from base energy?
        if np.std(neighbor_energies) > 0:
            corr = np.corrcoef([base_energy] * len(neighbor_energies), neighbor_energies)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
    
    # Low correlation = rough landscape = high Δ
    mean_corr = np.mean(correlations) if correlations else 0
    delta = 1 - mean_corr  # Transform so high Δ = rough
    
    return delta


def run_phase_transition_analysis(n_vars: int = 50, 
                                   alpha_range: Tuple[float, float] = (3.0, 5.5),
                                   n_alpha_points: int = 25,
                                   n_instances: int = 20) -> PhaseTransitionResult:
    """
    Run the phase transition analysis.
    
    For each α (clause/variable ratio):
    1. Generate random 3-SAT instances
    2. Compute Δ for each
    3. Check satisfiability (for small instances)
    """
    print("=" * 70)
    print("PNP-004: 3-SAT PHASE TRANSITION ANALYSIS")
    print("Davis Framework: Geometric Roughness Δ")
    print("=" * 70)
    
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha_points)
    delta_means = []
    delta_stds = []
    sat_fractions = []
    
    print(f"\nParameters: n_vars={n_vars}, n_instances={n_instances}")
    print(f"Scanning α from {alpha_range[0]} to {alpha_range[1]}")
    print(f"Known transition: α_c ≈ 4.267\n")
    
    for alpha in alphas:
        n_clauses = int(alpha * n_vars)
        
        deltas = []
        n_sat = 0
        
        for _ in range(n_instances):
            instance = generate_random_3sat(n_vars, n_clauses)
            
            # Compute Δ
            delta = compute_solution_space_roughness(instance, n_samples=500)
            deltas.append(delta)
            
            # Check SAT (only for small instances)
            if n_vars <= 20:
                if is_satisfiable_brute_force(instance):
                    n_sat += 1
        
        delta_means.append(np.mean(deltas))
        delta_stds.append(np.std(deltas))
        sat_fractions.append(n_sat / n_instances if n_vars <= 20 else np.nan)
        
        print(f"  α={alpha:.3f}: Δ={delta_means[-1]:.4f} ± {delta_stds[-1]:.4f}")
    
    delta_means = np.array(delta_means)
    delta_stds = np.array(delta_stds)
    sat_fractions = np.array(sat_fractions)
    
    # Find predicted transition point (where Δ changes most rapidly)
    # Use the derivative of Δ
    d_delta = np.gradient(delta_means, alphas)
    max_gradient_idx = np.argmax(np.abs(d_delta))
    predicted_alpha_c = alphas[max_gradient_idx]
    
    # Alternative: find inflection point
    d2_delta = np.gradient(d_delta, alphas)
    # Look for sign change in second derivative near the max gradient
    
    known_alpha_c = 4.267
    error = abs(predicted_alpha_c - known_alpha_c) / known_alpha_c
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nKnown transition point:    α_c = {known_alpha_c}")
    print(f"Predicted transition point: α_c = {predicted_alpha_c:.3f}")
    print(f"Error: {error:.1%}")
    
    if error < 0.1:
        print("\n✓ PNP-004 PASSED: Phase transition predicted within 10%")
    elif error < 0.2:
        print("\n~ PNP-004 MARGINAL: Transition approximately located")
    else:
        print("\n✗ PNP-004 FAILED: Transition not accurately predicted")
    
    print("=" * 70)
    
    return PhaseTransitionResult(
        alphas=alphas,
        delta_values=delta_means,
        sat_fractions=sat_fractions,
        predicted_alpha_c=predicted_alpha_c,
        known_alpha_c=known_alpha_c,
        error=error
    )


def plot_phase_transition(result: PhaseTransitionResult, save_path: str = None):
    """Visualize the phase transition analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Δ vs α
    ax = axes[0]
    ax.plot(result.alphas, result.delta_values, 'b.-', linewidth=2, markersize=8)
    ax.axvline(x=result.known_alpha_c, color='r', linestyle='--', linewidth=2, 
               label=f'Known α_c = {result.known_alpha_c}')
    ax.axvline(x=result.predicted_alpha_c, color='g', linestyle=':', linewidth=2,
               label=f'Predicted α_c = {result.predicted_alpha_c:.3f}')
    ax.set_xlabel('Clause/Variable Ratio α', fontsize=12)
    ax.set_ylabel('Geometric Roughness Δ', fontsize=12)
    ax.set_title('Davis Δ vs 3-SAT Constraint Density\n'
                 'Transition = where Δ changes rapidly', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Derivative of Δ (shows transition more clearly)
    ax = axes[1]
    d_delta = np.gradient(result.delta_values, result.alphas)
    ax.plot(result.alphas, d_delta, 'g.-', linewidth=2, markersize=8)
    ax.axvline(x=result.known_alpha_c, color='r', linestyle='--', linewidth=2,
               label=f'Known α_c = {result.known_alpha_c}')
    ax.axvline(x=result.predicted_alpha_c, color='b', linestyle=':', linewidth=2,
               label=f'Max gradient at {result.predicted_alpha_c:.3f}')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Clause/Variable Ratio α', fontsize=12)
    ax.set_ylabel('dΔ/dα (Roughness Gradient)', fontsize=12)
    ax.set_title('Rate of Change of Δ\n'
                 'Peak = Phase Transition', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'PNP-004: 3-SAT Phase Transition via Geometric Roughness\n'
                 f'Error: {result.error:.1%}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    
    plt.close()


def main():
    """Run PNP-004 validation test."""
    print("\n" + "=" * 70)
    print("DAVIS FRAMEWORK - MILLENNIUM VALIDATION")
    print("Test PNP-004: 3-SAT Phase Transition")
    print("=" * 70 + "\n")
    
    # Run analysis
    result = run_phase_transition_analysis(
        n_vars=50,           # Variables per instance
        alpha_range=(3.0, 5.5),  # Scan around known transition
        n_alpha_points=25,   # Resolution
        n_instances=20       # Instances per alpha
    )
    
    # Plot
    plot_phase_transition(result, save_path='results/p_vs_np/pnp_004_phase_transition.png')
    
    # Save data
    os.makedirs('results/p_vs_np', exist_ok=True)
    np.savez('results/p_vs_np/pnp_004_data.npz',
             alphas=result.alphas,
             delta_values=result.delta_values,
             predicted_alpha_c=result.predicted_alpha_c,
             known_alpha_c=result.known_alpha_c,
             error=result.error)
    
    print("\nResults saved to results/p_vs_np/")
    
    return result


if __name__ == "__main__":
    result = main()
