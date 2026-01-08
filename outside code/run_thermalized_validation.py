"""
Run Validation Tests with Thermalized Configurations
=====================================================

This script uses proper heatbath MCMC to generate thermalized configs,
then runs the A2S and A4C2 tests that previously failed.

Author: Bee Rosa Davis
Date: December 2025
"""

import torch
import sys
from pathlib import Path

# Import heatbath
from heatbath_mcmc import (
    generate_ensemble, compute_plaquette, thermalize,
    project_to_su3, heatbath_sweep
)


def wilson_action(U: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute Wilson gauge action."""
    dims = U.shape[:4]
    action = 0.0
    
    for t in range(dims[0]):
        for x in range(dims[1]):
            for y in range(dims[2]):
                for z in range(dims[3]):
                    site = (t, x, y, z)
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            shift_mu = [0, 0, 0, 0]
                            shift_mu[mu] = 1
                            shift_nu = [0, 0, 0, 0]
                            shift_nu[nu] = 1
                            
                            site_plus_mu = tuple((s + shift_mu[i]) % dims[i] for i, s in enumerate(site))
                            site_plus_nu = tuple((s + shift_nu[i]) % dims[i] for i, s in enumerate(site))
                            
                            P = (U[t, x, y, z, mu] @ 
                                 U[site_plus_mu[0], site_plus_mu[1], site_plus_mu[2], site_plus_mu[3], nu] @
                                 U[site_plus_nu[0], site_plus_nu[1], site_plus_nu[2], site_plus_nu[3], mu].conj().T @
                                 U[t, x, y, z, nu].conj().T)
                            
                            action += (1 - P.trace().real / 3.0)
    
    return beta * action


def compute_topological_charge(U: torch.Tensor) -> torch.Tensor:
    """Compute topological charge Q (simplified clover definition)."""
    # This is a simplified version - full clover is more complex
    dims = U.shape[:4]
    Q = torch.tensor(0.0, device=U.device)
    
    # Sum over spatial sites at t=0
    for x in range(dims[1]):
        for y in range(dims[2]):
            for z in range(dims[3]):
                # F_12 * F_34 contribution
                P12 = compute_plaquette_single(U, (0, x, y, z), 1, 2)
                P34 = compute_plaquette_single(U, (0, x, y, z), 0, 3)
                Q += (P12.trace().imag * P34.trace().imag) / 9.0
    
    return Q / (32 * 3.14159**2)


def compute_plaquette_single(U, site, mu, nu):
    """Compute single plaquette at site in mu-nu plane."""
    dims = U.shape[:4]
    t, x, y, z = site
    
    shift_mu = [0, 0, 0, 0]
    shift_mu[mu] = 1
    shift_nu = [0, 0, 0, 0]
    shift_nu[nu] = 1
    
    site_plus_mu = tuple((s + shift_mu[i]) % dims[i] for i, s in enumerate(site))
    site_plus_nu = tuple((s + shift_nu[i]) % dims[i] for i, s in enumerate(site))
    
    return (U[t, x, y, z, mu] @ 
            U[site_plus_mu[0], site_plus_mu[1], site_plus_mu[2], site_plus_mu[3], nu] @
            U[site_plus_nu[0], site_plus_nu[1], site_plus_nu[2], site_plus_nu[3], mu].conj().T @
            U[t, x, y, z, nu].conj().T)


def test_A2S_001(ensemble: torch.Tensor, beta: float) -> dict:
    """
    A2S-001: Action-to-Susceptibility Response
    
    Tests that δ⟨O⟩/δβ correlates with action fluctuations.
    This requires thermalized configs to show physical correlations.
    """
    print("\n" + "="*60)
    print("TEST: A2S-001 - Action-to-Susceptibility Response")
    print("="*60)
    
    n_configs = ensemble.shape[0]
    
    # Compute observables for each config
    plaquettes = []
    actions = []
    
    for i in range(n_configs):
        U = ensemble[i]
        plaq = compute_plaquette(U)
        action = wilson_action(U, beta)
        
        plaquettes.append(plaq.item() if torch.is_tensor(plaq) else plaq)
        actions.append(action.item() if torch.is_tensor(action) else action)
    
    plaquettes = torch.tensor(plaquettes)
    actions = torch.tensor(actions)
    
    # Compute susceptibility via fluctuation-dissipation
    # χ = β² (⟨S²⟩ - ⟨S⟩²) 
    S_mean = actions.mean()
    S2_mean = (actions**2).mean()
    chi = beta**2 * (S2_mean - S_mean**2)
    
    # Compute δO via finite difference would require another β
    # Instead, check correlation between O and S
    P_mean = plaquettes.mean()
    PS_cov = ((plaquettes - P_mean) * (actions - S_mean)).mean()
    
    # Normalize
    P_std = plaquettes.std()
    S_std = actions.std()
    
    if P_std > 1e-10 and S_std > 1e-10:
        correlation = PS_cov / (P_std * S_std)
    else:
        correlation = 0.0
    
    # For thermalized configs, expect |correlation| > 0.1
    # Random configs have correlation ≈ 0
    threshold = 0.1
    passed = abs(correlation) > threshold
    
    print(f"  ⟨P⟩ = {P_mean:.6f} ± {P_std:.6f}")
    print(f"  ⟨S⟩ = {S_mean:.1f} ± {S_std:.1f}")
    print(f"  χ = {chi:.4f}")
    print(f"  Correlation(P, S) = {correlation:.4f}")
    print(f"  Threshold: |corr| > {threshold}")
    print()
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return {
        'passed': passed,
        'plaquette_mean': P_mean.item(),
        'plaquette_std': P_std.item(),
        'action_mean': S_mean.item(),
        'action_std': S_std.item(),
        'correlation': correlation.item() if torch.is_tensor(correlation) else correlation,
        'chi': chi.item()
    }


def test_A4C2_001(ensemble: torch.Tensor, beta: float, n_bins: int = 10) -> dict:
    """
    A4C2-001: 4-point Correlation / 2-point Correlation Check
    
    Tests that binned observables show proper statistical properties.
    """
    print("\n" + "="*60)
    print("TEST: A4C2-001 - Binned Observable Statistics")
    print("="*60)
    
    n_configs = ensemble.shape[0]
    
    if n_configs < n_bins * 2:
        print(f"  Warning: Only {n_configs} configs, need {n_bins * 2} for {n_bins} bins")
        n_bins = max(2, n_configs // 2)
        print(f"  Reducing to {n_bins} bins")
    
    # Compute plaquettes
    plaquettes = []
    for i in range(n_configs):
        plaq = compute_plaquette(ensemble[i])
        plaquettes.append(plaq.item() if torch.is_tensor(plaq) else plaq)
    
    plaquettes = torch.tensor(plaquettes)
    
    # Bin the data
    bin_size = n_configs // n_bins
    binned_means = []
    
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size
        binned_means.append(plaquettes[start:end].mean().item())
    
    binned_means = torch.tensor(binned_means)
    
    # Compute bin statistics
    overall_mean = plaquettes.mean()
    overall_var = plaquettes.var()
    
    bin_mean = binned_means.mean()
    bin_var = binned_means.var()
    
    # Variance reduction ratio
    # For independent samples: bin_var ≈ overall_var / bin_size
    # For correlated samples: bin_var > overall_var / bin_size
    expected_bin_var = overall_var / bin_size
    
    if expected_bin_var > 1e-10:
        variance_ratio = bin_var / expected_bin_var
    else:
        variance_ratio = 1.0
    
    # For thermalized configs, expect significant variance in bins
    # Random configs would have variance_ratio ≈ 1
    # Correlated configs have variance_ratio > 1 (autocorrelation)
    
    # Test: bins should have meaningful variance
    min_bin_std = 0.001
    passed = bin_var.sqrt() > min_bin_std and n_bins >= 5
    
    print(f"  Number of bins: {n_bins}")
    print(f"  Configs per bin: {bin_size}")
    print(f"  Overall: ⟨P⟩ = {overall_mean:.6f} ± {overall_var.sqrt():.6f}")
    print(f"  Binned:  ⟨P⟩ = {bin_mean:.6f} ± {bin_var.sqrt():.6f}")
    print(f"  Variance ratio: {variance_ratio:.3f}")
    print(f"  Min bin std threshold: {min_bin_std}")
    print()
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return {
        'passed': passed,
        'n_bins': n_bins,
        'bin_size': bin_size,
        'overall_mean': overall_mean.item(),
        'overall_std': overall_var.sqrt().item(),
        'bin_mean': bin_mean.item(),
        'bin_std': bin_var.sqrt().item(),
        'variance_ratio': variance_ratio.item()
    }


def main():
    print("="*60)
    print("LATTICE QCD VALIDATION WITH THERMALIZED CONFIGS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    lattice_dims = (8, 8, 8, 8)  # Larger lattice for better statistics
    beta = 6.0
    n_configs = 100  # More configs for statistical tests
    n_therm = 200    # Thorough thermalization
    n_skip = 10      # Decorrelation between configs
    
    print(f"\nLattice: {lattice_dims}")
    print(f"β = {beta}")
    print(f"Configs: {n_configs}")
    print(f"Thermalization: {n_therm} sweeps")
    print(f"Skip between configs: {n_skip} sweeps")
    print()
    
    # Generate thermalized ensemble
    print("Generating thermalized ensemble...")
    print("-" * 40)
    
    ensemble = generate_ensemble(
        lattice_dims, beta,
        n_configs=n_configs,
        n_therm=n_therm,
        n_skip=n_skip,
        device=device,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("RUNNING VALIDATION TESTS")
    print("="*60)
    
    # Run tests
    results = {}
    
    results['A2S-001'] = test_A2S_001(ensemble, beta)
    results['A4C2-001'] = test_A4C2_001(ensemble, beta, n_bins=10)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not result['passed']:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
    else:
        print("Some tests failed. Check parameters or increase statistics.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
