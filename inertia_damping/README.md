# Davis Inertia Damping Control System

**Pseudocode Draft v0.1 — June 2026**

Software architecture for the inertia damping experiment built on Davis Field Equations / Yang-Mills mass gap framework. Six modules, each in its own file, designed to be implemented and verified independently before integration.

## Module index

1. **Module 1: Forward Model (Hamiltonian + Connection)** — `module1_forward_model.md`
   The dynamical simulator. Propagates the gauge field forward in time given current state and control inputs. Symplectic integrator on the lattice Hamiltonian.

2. **Module 2: Boundary Problem Solver** — `module2_boundary_solver.md`
   Given a target inertia profile, computes the gauge field configuration that realizes it on the lattice. Constrained optimization with integer topological charge constraint.

3. **Module 3: Wilson Loop Inversion** — `module3_wilson_inversion.md`
   Converts abstract gauge field setpoints into concrete junction phase and bias current setpoints for the hardware. Matrix logarithm on the principal branch.

4. **Module 4: State Estimation** — `module4_state_estimation.md`
   Reconstructs the current gauge field from SQUID magnetometer measurements. Extended Kalman filter on a Lie group manifold with phase unwrapping.

5. **Module 5: Optimal Control** — `module5_optimal_control.md`
   Plans the time-varying control schedule to drive the field from current state to target. Pontryagin minimum principle, bang-bang protocols for linear-in-u dynamics.

6. **Module 6: Stability Monitor** — `module6_stability_monitor.md`
   Continuously verifies that the current configuration is stable. Sparse eigenvalue computation on the Hessian, fast probe checks between full evaluations, emergency rollback if instability detected.

## Data flow

The modules form a control loop:

- Module 2 produces a *target* (offline, slow, called when a new experiment is configured)
- Module 5 plans a *trajectory* from current state to target (offline planning + online MPC)
- Module 3 converts schedule into *hardware setpoints* (DAC commands)
- Hardware executes; Module 4 *measures* the result via SQUIDs
- Module 1 *predicts* what should be happening for comparison with Module 4
- Module 6 *monitors* stability throughout
- Module 5 *replans* based on Module 4 + Module 6 feedback
- Loop repeats

## Implementation order

Recommended sequence for actually building this:

1. Start with Module 1 on a small test lattice (cube, 8 vertices). Verify symplectic conservation properties with no driving.
2. Add Module 2's action and topological charge routines (you'll need them anyway). Verify on known instantons.
3. Module 6's eigenvalue analysis next — needed to validate Module 2's outputs are actually stable.
4. Module 3 is straightforward once you have a target field; build it after Module 2 is solid.
5. Module 4 with synthetic SQUID measurements generated from Module 1. Verify filter convergence.
6. Module 5 last. Use the others as the simulator it plans against.

Test each module against the next before scaling up the lattice. Move to buckyball geometry only after all six pass on the cube.

## Framework gaps to fill from your work

The following are flagged in individual module files; consolidated here:

- **Variable-beta coupling functional form.** Module 2 uses `m^2 -> m^2 - alpha^2 A^mu A_mu` per Benioff's fiber bundle scaling result as a placeholder. Replace with your Branch XIII / Davis Duality expression.
- **Gauge group choice.** U(1) is the simplest; SU(2) is the standard Yang-Mills minimal case. Pick based on which symmetry your physical platform realizes.
- **Coupling constants and prefactors** in the relationship between mass gap, lattice spacing, junction parameters, and the inertia coupling. These are framework-specific dimensional analysis.
- **Cage geometry specifics.** Buckyball is a placeholder; the optimal cage shape depends on the variational solution to Module 2's boundary problem in your framework.

## Honest scope statement

This is software architecture, not a working implementation. Every module has been written at the level of "informed pseudocode that explains the algorithm" — enough that someone with strong numerical experience could turn it into working code, not enough that the code itself exists.

The mathematics in Modules 1, 3, 4, 5, 6 is well-established (lattice gauge theory, Lie group Kalman filters, Pontryagin optimal control, sparse eigenvalue methods). What is novel is the combination and the target: testing whether the variable-beta coupling produces a measurable change in effective inertia. This experiment, as far as can be determined, has not been performed.

The framework dependencies in Module 2 (and to a lesser extent in the coupling parameters appearing elsewhere) are where the specific predictions of Davis Field Equations enter. Those are the load-bearing claims that the experiment would actually test.

## Status and next steps

Status: pseudocode complete for first draft. No tested code. No hardware design beyond cage geometry placeholder. No experimental cost estimate.

Next steps depend on collaborators and resources:

- *If solo*: implement Module 1 in Python on small lattice, verify, post as open-source preprint with reproducibility instructions. Use this to attract physics collaborators.
- *If with a numerical physicist*: pair-implement Modules 1, 2, 6 first. Aim for a paper showing the boundary problem solver works on simulated data with a known coupling.
- *If with an experimentalist*: do the above first, then engage on hardware design (cage geometry, junction parameters, cryogenic constraints) once the software predicts something specific to measure.

## License

**PolyForm Noncommercial License 1.0.0** — free for research and
noncommercial use; commercial use requires a separate license.

See the repository-root [`LICENSE`](../LICENSE) file for the canonical
license text and project-specific notice. SPDX identifier:
`PolyForm-Noncommercial-1.0.0`. Canonical license URL:
<https://polyformproject.org/licenses/noncommercial/1.0.0/>.

Author: Bee Davis, with software architecture assistance from Claude
(Anthropic). Copyright (c) 2026 Bee Rosa Davis (Davis Geometric, Inc.)
and contributors.

For commercial-licensing inquiries: `bee_davis@alumni.brown.edu`.
