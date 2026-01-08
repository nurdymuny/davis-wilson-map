"""
Test heatbath from COLD START (identity matrices).

If the algorithm is correct:
- Cold start plaquette = 1.0
- After thermalization at β=6.0, should drop to ~0.59

This is a sanity check to see if the heatbath is working at all.
"""

import torch
from heatbath_mcmc import heatbath_sweep, compute_plaquette, project_to_su3

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Small lattice
    dims = (4, 4, 4, 4)
    beta = 6.0
    
    # COLD START: all links = identity
    U = torch.eye(3, dtype=torch.complex64, device=device)
    U = U.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    U = U.expand(dims[0], dims[1], dims[2], dims[3], 4, 3, 3).clone()
    
    print(f"\nCOLD START: all links = identity")
    print(f"Initial plaquette: {compute_plaquette(U):.6f}")
    print(f"(Should be 1.0 for identity links)")
    print()
    
    print(f"Thermalizing at β={beta}...")
    print("-" * 40)
    
    for i in range(100):
        U = heatbath_sweep(U, beta)
        
        if (i + 1) % 5 == 0:
            plaq = compute_plaquette(U)
            print(f"  Sweep {i+1:4d}: <P> = {plaq:.6f}")
    
    print("-" * 40)
    final_plaq = compute_plaquette(U)
    print(f"Final plaquette: {final_plaq:.6f}")
    print(f"Expected at β=6.0: ~0.59")
    print()
    
    if final_plaq > 0.55:
        print("✓ Looks reasonable!")
    else:
        print("✗ Something is wrong with the heatbath")


if __name__ == "__main__":
    main()
