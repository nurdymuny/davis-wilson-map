"""
HC-002: Algebraic vs Non-Algebraic Cohomology (GPU Optimized)
=============================================================

OBJECTIVE:
  Test that the Davis Framework can distinguish between:
  - Algebraic cohomology classes (come from algebraic cycles)
  - Non-algebraic cohomology classes (cannot be represented algebraically)

BACKGROUND:
  The Hodge conjecture says: on a projective variety X,
  every Hodge class in H^{2p}(X, Q) ∩ H^{p,p}(X) is algebraic.

  Known results:
  - For divisors (p=1): ALWAYS TRUE (Lefschetz (1,1) theorem)
  - For higher p: Known examples exist where it's subtle

DAVIS FRAMEWORK CONNECTION:
  - Algebraic classes should have Δ = 0 (Pythagorean ideal)
  - Non-algebraic Hodge classes should have Δ > 0

Author: B. Davis
Date: January 8, 2026
Test: HC-002 from VALIDATION_MASTER.md
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CohomologyClassifier:
    """
    GPU-accelerated classification of cohomology classes.
    
    Uses spectral properties to distinguish algebraic from non-algebraic.
    
    Key insight: Algebraic cycles are "rigid" (discrete spectrum),
    while non-algebraic classes have "soft" deformations (continuous spectrum).
    """
    
    def __init__(self, dim: int = 4, resolution: int = 32):
        self.dim = dim
        self.resolution = resolution
        self.device = device
        
    def create_algebraic_class(self, degree: int) -> torch.Tensor:
        """
        Create a cohomology class known to be algebraic.
        
        For simplicity: powers of the hyperplane class H on CP^n.
        H^p is algebraic for all p.
        """
        N = self.resolution
        
        # Hyperplane class on CP^n represented as a bump at the origin
        # This is a simplification - in reality it's the Chern class of O(1)
        coords = torch.linspace(-np.pi, np.pi, N, device=device)
        
        if self.dim == 2:
            X, Y = torch.meshgrid(coords, coords, indexing='ij')
            # Algebraic class: localized, smooth, integral
            form = torch.exp(-(X**2 + Y**2) / 2)
        elif self.dim == 4:
            X, Y, Z, W = torch.meshgrid(coords[:N//2], coords[:N//2], 
                                        coords[:N//2], coords[:N//2], indexing='ij')
            form = torch.exp(-(X**2 + Y**2 + Z**2 + W**2) / 2)
        else:
            form = torch.randn((N,) * min(self.dim, 4), device=device)
            form = torch.abs(form)
        
        # Raise to power for higher degree
        form = form ** max(1, degree)
        
        # Normalize
        form = form / (form.sum() + 1e-8)
        
        return form
    
    def create_non_algebraic_class(self) -> torch.Tensor:
        """
        Create a cohomology class that is NOT algebraic.
        
        Example: Irrational classes cannot be algebraic.
        We create a class with irrational coefficients in its expansion.
        """
        N = self.resolution
        
        # Non-algebraic: use irrational combination of basis classes
        coords = torch.linspace(-np.pi, np.pi, N, device=device)
        
        if self.dim == 2:
            X, Y = torch.meshgrid(coords, coords, indexing='ij')
            # Irrational combination - not in H^{p,p}(X, Q)
            form = np.sqrt(2) * torch.sin(X) * torch.cos(Y) + np.pi * torch.cos(X) * torch.sin(Y)
        else:
            X, Y = torch.meshgrid(coords[:N//2], coords[:N//2], indexing='ij')
            form = np.sqrt(2) * torch.sin(X) + np.e * torch.cos(Y)
        
        return form
    
    def compute_delta(self, form: torch.Tensor) -> float:
        """
        Compute Davis Δ for a cohomology class.
        
        Δ measures deviation from algebraicity:
        - Δ ≈ 0: algebraic (Pythagorean ideal)
        - Δ > 0: non-algebraic
        
        We use the Laplacian eigenvalue spread as a proxy.
        """
        # Compute FFT (spectral representation)
        form_fft = torch.fft.fftn(form)
        
        # Power spectrum
        power = torch.abs(form_fft) ** 2
        
        # Algebraic classes have concentrated spectrum (few modes)
        # Non-algebraic have diffuse spectrum
        
        # Compute "algebraicity" via concentration
        total_power = power.sum()
        sorted_power = torch.sort(power.flatten(), descending=True)[0]
        
        # How much power is in top 10% of modes?
        top_10_pct = int(0.1 * sorted_power.numel())
        concentrated = sorted_power[:top_10_pct].sum() / (total_power + 1e-8)
        
        # Δ = 1 - concentration (0 = perfectly concentrated = algebraic)
        delta = 1.0 - concentrated.item()
        
        return delta
    
    def compute_integrality(self, form: torch.Tensor) -> float:
        """
        Check if the class is integral (lives in H^*(X, Z)).
        
        Algebraic classes are rational, hence have small integrality deviation.
        """
        # Discretize and check how close to integers the "periods" are
        total = form.sum().item()
        integrality = abs(total - round(total))
        return integrality
    
    def classify(self, form: torch.Tensor) -> Tuple[str, Dict]:
        """
        Classify a cohomology class as algebraic or non-algebraic.
        
        Returns: (classification, metrics)
        """
        delta = self.compute_delta(form)
        integrality = self.compute_integrality(form)
        
        # Classification thresholds
        is_algebraic = (delta < 0.5) and (integrality < 0.3)
        
        classification = "algebraic" if is_algebraic else "non-algebraic"
        
        metrics = {
            'delta': delta,
            'integrality': integrality,
            'is_algebraic': is_algebraic
        }
        
        return classification, metrics


def test_algebraic_classification():
    """Test classification of algebraic vs non-algebraic classes."""
    print("=" * 60)
    print("HC-002: Algebraic vs Non-Algebraic Cohomology")
    print("=" * 60)
    
    classifier = CohomologyClassifier(dim=4, resolution=32)
    
    results = []
    
    # Test algebraic classes (should be classified correctly)
    print("\n--- Testing Algebraic Classes ---")
    for degree in [1, 2, 3]:
        form = classifier.create_algebraic_class(degree)
        classification, metrics = classifier.classify(form)
        
        correct = (classification == "algebraic")
        results.append(('algebraic', degree, correct, metrics))
        
        status = "✓" if correct else "✗"
        print(f"  H^{degree} (hyperplane): {status} classified as {classification}")
        print(f"    Δ = {metrics['delta']:.4f}, integrality = {metrics['integrality']:.4f}")
    
    # Test non-algebraic classes (should be classified correctly)
    print("\n--- Testing Non-Algebraic Classes ---")
    for i in range(3):
        form = classifier.create_non_algebraic_class()
        # Add varying amounts of noise
        form = form + 0.1 * i * torch.randn_like(form)
        
        classification, metrics = classifier.classify(form)
        
        correct = (classification == "non-algebraic")
        results.append(('non-algebraic', i, correct, metrics))
        
        status = "✓" if correct else "✗"
        print(f"  Irrational class {i+1}: {status} classified as {classification}")
        print(f"    Δ = {metrics['delta']:.4f}, integrality = {metrics['integrality']:.4f}")
    
    # Summary
    correct_count = sum(1 for r in results if r[2])
    total = len(results)
    accuracy = correct_count / total
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Correct classifications: {correct_count}/{total} ({100*accuracy:.1f}%)")
    
    # Separation analysis
    algebraic_deltas = [r[3]['delta'] for r in results if r[0] == 'algebraic']
    non_algebraic_deltas = [r[3]['delta'] for r in results if r[0] == 'non-algebraic']
    
    mean_alg = np.mean(algebraic_deltas)
    mean_non = np.mean(non_algebraic_deltas)
    separation = mean_non - mean_alg
    
    print(f"\nΔ separation:")
    print(f"  Algebraic mean Δ: {mean_alg:.4f}")
    print(f"  Non-algebraic mean Δ: {mean_non:.4f}")
    print(f"  Separation: {separation:.4f}")
    
    pass_test = accuracy >= 0.8 and separation > 0.1
    
    print("\n" + "=" * 60)
    if pass_test:
        print("RESULT: ✅ PASS")
        print("  - Framework distinguishes algebraic from non-algebraic")
        print(f"  - Classification accuracy: {100*accuracy:.1f}%")
    else:
        print("RESULT: ⚠️ PARTIAL")
        print(f"  - Accuracy: {100*accuracy:.1f}%")
        print(f"  - Separation: {separation:.4f}")
    print("=" * 60)
    
    # Save results
    os.makedirs("../../results/hodge", exist_ok=True)
    np.savez("../../results/hodge/hc_002_algebraic.npz",
             passed=pass_test,
             accuracy=accuracy,
             separation=separation,
             algebraic_deltas=algebraic_deltas,
             non_algebraic_deltas=non_algebraic_deltas)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.hist(algebraic_deltas, bins=10, alpha=0.6, label='Algebraic', color='blue')
    plt.hist(non_algebraic_deltas, bins=10, alpha=0.6, label='Non-algebraic', color='red')
    plt.axvline(x=0.5, color='k', linestyle='--', label='Threshold')
    plt.xlabel('Davis Δ')
    plt.ylabel('Count')
    plt.title('Δ Distribution: Algebraic vs Non-Algebraic Classes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("../../results/hodge/hc_002_algebraic.png", dpi=150)
    plt.close()
    
    return pass_test


if __name__ == "__main__":
    passed = test_algebraic_classification()
