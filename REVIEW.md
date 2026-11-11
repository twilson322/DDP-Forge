# Code Review: ddp-forge

Reviewed as a staff engineer on an optimization or controls team would review a pull request that adds a constrained iLQR/DDP solver. Scope: `include/ddp` (solver, dual numbers, systems, problems), the experiment app, the tests, the CasADi cross-check and the build.

## 1. Overall assessment

A compact, readable solver that is verified the right way: an exact Riccati comparison on a linear-quadratic problem, nested dual numbers for second derivatives, and an independent IPOPT solve warm-started at every C++ solution. The README is candid about where DDP loses to iLQR.

The review found two correctness defects in what the solver reports and enforces, both on paths the benchmark problems happened not to exercise. `converged` could be reported after a numerical failure, and the terminal state was never checked against the obstacles. Both are fixed, each with a test that fails on the old code. Rerunning every experiment gave identical iteration counts, costs and convergence flags, so no README number changed.

## 2. Scorecard

| Area | Score | Basis |
|---|---|---|
| Correctness | 8 | Exact on the LQ problem, matches IPOPT to 3.2e-8 relative cost. Two reporting and enforcement defects fixed in this pass. |
| Architecture | 8 | Dynamics written once and templated on the scalar type, one solver for iLQR and DDP, augmented Lagrangian wrapped around the unconstrained core. |
| Readability | 7 | The backward pass reads like the paper. Single-letter Eigen names (`Qx`, `Quu`, `K`) are standard here but the 140-line `solve` would read better split into backward pass, line search and multiplier update. |
| Self-documentation | 8 | Comments explain the choices (control-space regularization, stopping on predicted rather than actual improvement). One stale comment fixed. |
| Maintainability | 7 | Header-only template, so every change recompiles every user. Solver state (`lam_`, `mu_`) lives in members and is reset inside `solve`. |
| Testing | 8 | LQ optimality against Riccati, DDP against iLQR, AL feasibility, dual-number arithmetic, and now termination reporting and terminal constraints. |
| Performance | 6 | `constraints()` heap-allocates an `Eigen::VectorXd` and `MatrixXd` per stage per iteration. DDP's Hessian costs $(n_x+n_u)(n_x+n_u+1)/2$ nested-dual passes per stage, which the README measures. |
| Reliability | 8 | Regularization escalation, line-search fallback, finite-cost checks, AL multiplier projection. |
| API design | 7 | `Problem`, `Options`, `Result` are plain structs with defaults. `solve` takes the initial controls by value and mutates solver members, so one `Solver` cannot solve two problems concurrently. |
| Domain-specific correctness | 7 | RK4 discretization replicated in the cross-check, control-space Levenberg-Marquardt justified. Obstacles are enforced at the knots only. |
| Safety and robustness | Not assessed | An offline optimizer and an MPC study in simulation. |

## 3. Problems

### Fixed in this review

| Severity | Location | What | Why | Fix and evidence |
|---|---|---|---|---|
| High | `Solver::solve` | For unconstrained problems, `converged` was set whenever the inner loop exited, including when $\rho$ passed `rho_max` after a failed backward pass. For constrained problems a feasible run that hit the iteration cap was also reported converged. | The README's convergence claims rest on this flag. A numerical failure could have been reported as success. No benchmark hit this path, so no reported number was wrong. | `Termination {Converged, MaxIterations, RegularizationLimit}` on every exit, and `converged` requires stationarity and feasibility. `Solver.IterationCapIsNotReportedAsConvergence`. |
| Medium | `constraints`, `al_cost`, `max_violation`, backward pass | Constraints were evaluated for states 0 to N-1 only. | A goal inside an obstacle, or a final state that drifts into one, was reported feasible. | Terminal obstacle constraints with their own multipliers, added to the terminal value function. `Solver.TerminalStateIsCheckedAgainstObstacles` fails without them (the final state sits at the goal, inside the obstacle). |
| Medium | `apps/ddp_experiments.cpp` | `fopen` results were never checked, and the output directory was never created. | Running with a fresh `--out` directory segfaulted in the first `fprintf`. Found while rerunning the experiments for this review. | `open_for_writing` exits with the path on failure, and `main` creates the directory. |
| Low | `Options::tol` | The comment said "relative cost change". The code stops on the predicted first-order improvement $\lvert dV_1\rvert / (1 + \lvert J\rvert)$. | The comment described the bug the README says was fixed. | Comment corrected. |
| Low | `Solver::solve` | Line-search floor (1e-4), Armijo fraction (1e-4) and the regularization floor after a failed factorization (1e-6) were literals. | Tuning knobs hidden in the loop. | `Options::min_alpha`, `armijo_fraction`, `rho_after_failure`. |
| Low | `Acrobot::I2` | Used `1.0 / 12.0` with the second link length hard-coded as 1 m. | Changing the link length would silently leave the inertia wrong. | `l2` member, $m_2 l_2^2/12$. |
| Low | CI | The crosscheck job ran `pip install` against Ubuntu 24.04's system Python. | PEP 668 rejects it. | `actions/setup-python`. |

### Open

| Severity | Location | What | Why | Recommended fix |
|---|---|---|---|---|
| Medium | obstacle constraints | Enforced at the knots only, 20 ms apart. | The quadrotor moves several centimetres between knots, so a straight segment can clip an obstacle's edge while every knot is outside. | Inflate each radius by $v_{max}\,\Delta t / 2$, or add mid-interval constraint points. |
| Low | `constraints`, `al_cost`, `max_violation` | Dynamic Eigen allocations per stage per iteration. | Allocation in the solver's inner loop. Matters for the MPC use with a 1-iteration budget. | Preallocate per-stage buffers once in `solve`, or use fixed-capacity types. |
| Low | `Solver` | Multipliers and penalty are members mutated by `solve`. | Not reentrant, and a second `solve` silently resets them. | Move them into a local solve state, make `solve` const. |
| Low | `solve` | 140 lines in one function. | Harder to review than it needs to be. | Split into `backwardPass`, `lineSearch`, `updateMultipliers`. |
| Low | tests | MPC mode (`always_accept`) has no unit test. | The real-time iteration path is exercised only by the experiment. | A test that one always-accept iteration reduces the cost on a near-optimal warm start. |

## 4. Readability and self-documentation

```cpp
// before
double tol{1e-7};          // relative cost change for inner convergence
...
for (double alpha = 1.0; alpha > 1e-4; alpha *= 0.5) {
...
double I2() const { return m2 * lc2 * lc2 + m2 * 1.0 / 12.0; }      // about the elbow (l2 = 1)

// after
double tol{1e-7};          // inner convergence: predicted first-order improvement |dV1| / (1 + |J|)
...
for (double alpha = 1.0; alpha > o.min_alpha; alpha *= 0.5) {
...
// uniform rods: m l^2 / 12 about the centre of mass, shifted to the joint
double I2() const { return m2 * lc2 * lc2 + m2 * l2 * l2 / 12.0; }  // about the elbow
```

The comments on control-space regularization and on the stopping rule are the kind worth keeping: each records a choice, the alternative, and the measured reason.

## 5. Architecture and design

Writing the dynamics once as `template <typename S> f(x, u)` and evaluating them in `double`, `Dual<double>` and `Dual<Dual<double>>` is the central design decision, and it is a good one: Jacobians and second-order terms cannot drift from the model. Wrapping the PHR augmented Lagrangian around an unchanged unconstrained core keeps both parts testable. The CasADi script replicates the RK4 map exactly, so the IPOPT comparison checks the optimizer, not the discretization.

## 6. Testing and verification

| Claim | Verified by |
|---|---|
| iLQR is exact on linear-quadratic problems | Math and test: discrete Riccati recursion |
| Dual-number derivatives are correct | Tests: analytic derivatives of elementary functions, finite differences on all three systems |
| DDP and iLQR reach the same optimum | Test on the acrobot |
| Solutions are local optima | Independent IPOPT solve warm-started at each solution (`results/crosscheck.txt`) |
| Bounds and obstacles hold at the knots, including the terminal state | Tests, with a mutation check on the terminal constraint |
| `converged` means converged | Test on the iteration cap |
| Memory safety | All 10 tests under ASan and UBSan |
| Behavior unchanged by this review | Every experiment rerun: identical iterations, costs and flags |
| Feasibility between knots | Not verified |

## 7. Prioritized action plan

Must fix before commit: none outstanding.

Before sharing as a library: preallocate the constraint buffers, make `solve` reentrant, and add an MPC-mode test.

Recommended: continuous-time obstacle clearance (radius inflation or extra constraint points).

Optional: split `solve`, and compute the DDP Hessian with fewer passes (one nested-dual pass per column with a vector-valued inner dual).

## 8. Final verdict

| Context | Verdict |
|---|---|
| Personal or portfolio project | STRONG |
| Shared codebase | ACCEPTABLE, after the allocation and reentrancy items |
| Simulation | STRONG |
| Hardware | Not applicable on its own. As an MPC back end it would need the allocation fix and knot-to-knot obstacle clearance first. |
