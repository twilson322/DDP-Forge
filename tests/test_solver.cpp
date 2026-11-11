#include <gtest/gtest.h>

#include "ddp/problems.hpp"

using namespace ddp;

// Linear dynamics + quadratic cost: the Gauss-Newton model is exact, so iLQR must land on the
// optimum in a single iteration and every later iteration must change nothing.
struct DoubleIntegrator {
  static constexpr std::size_t NX = 2, NU = 1;
  template <typename S>
  Arr<S, 2> f(const Arr<S, 2>& x, const Arr<S, 1>& u) const { return {x[1], u[0]}; }
};

TEST(Solver, LinearQuadraticConvergesInOneStep) {
  Problem<DoubleIntegrator> p;
  p.dt = 0.05;
  p.N = 60;
  p.x0 << 1.0, -0.5;
  p.x_goal << 0.0, 0.0;
  p.Q.diagonal() << 1.0, 0.1;
  p.R(0, 0) = 0.01;
  p.Qf.diagonal() << 10.0, 1.0;
  Solver<DoubleIntegrator> s(p);
  Options o;
  o.max_iter = 1;
  const auto r1 = s.solve(zeros<DoubleIntegrator>(p.N), o);
  o.max_iter = 20;
  const auto r20 = s.solve(zeros<DoubleIntegrator>(p.N), o);
  // Not bit-exact: the initial Levenberg-Marquardt term (rho = 1e-6 on Q_uu) perturbs the
  // first step slightly. The gap is ~4e-10 relative.
  EXPECT_NEAR(r1.cost, r20.cost, 1e-8 * r20.cost);

  // Independent check: finite-horizon discrete Riccati recursion on the exact discretization.
  Eigen::Matrix2d A;
  A << 1, p.dt, 0, 1;
  Eigen::Vector2d B(0.5 * p.dt * p.dt, p.dt);
  Eigen::Matrix2d P = p.Qf;
  for (int k = 0; k < p.N; ++k) {
    const double S = p.R(0, 0) + B.dot(P * B);
    P = p.Q + A.transpose() * P * A - (A.transpose() * P * B) * (B.transpose() * P * A) / S;
  }
  EXPECT_NEAR(r20.cost, 0.5 * p.x0.dot(P * p.x0), 1e-9);
}

TEST(Solver, DdpMatchesIlqrOptimumOnAcrobot) {
  const auto p = acrobot_swingup(true);
  Options o;
  Solver<Acrobot> s(p);
  const auto a = s.solve(zeros<Acrobot>(p.N), o);
  o.second_order = true;
  const auto b = s.solve(zeros<Acrobot>(p.N), o);
  EXPECT_TRUE(a.converged);
  EXPECT_TRUE(b.converged);
  EXPECT_NEAR(a.cost, b.cost, 1e-3);
  EXPECT_LT(b.iterations, a.iterations);  // second-order model needs fewer iterations
  EXPECT_NEAR(b.xs.back()(0), M_PI, 0.05);
}

TEST(Solver, AugmentedLagrangianRespectsBoundsAndObstacles) {
  const auto p = quad_obstacles(true);
  Solver<PlanarQuad> s(p);
  const auto r = s.solve(hover(p), Options{});
  ASSERT_TRUE(r.converged);
  EXPECT_LT(r.max_violation, 1e-4);
  for (const auto& u : r.us) {
    EXPECT_GE(u.minCoeff(), -1e-4);
    EXPECT_LE(u.maxCoeff(), 8.0 + 1e-4);
  }
  for (const auto& x : r.xs)
    for (const auto& o : p.obstacles) EXPECT_GE(std::hypot(x(0) - o.cx, x(1) - o.cy), o.r - 1e-4);
  EXPECT_NEAR(r.xs.back()(0), 6.0, 0.05);
  EXPECT_NEAR(r.xs.back()(1), 3.0, 0.05);
}

TEST(Solver, IterationCapIsNotReportedAsConvergence) {
  const auto p = cartpole_swingup(false);
  Solver<CartPole> s(p);
  Options o;
  o.max_iter = 5;
  const auto r = s.solve(zeros<CartPole>(p.N), o);
  EXPECT_FALSE(r.converged);
  EXPECT_EQ(r.termination, Termination::MaxIterations);
}

TEST(Solver, TerminalStateIsCheckedAgainstObstacles) {
  // A goal inside an obstacle cannot be reached feasibly. Before the terminal constraint
  // existed, only states 0..N-1 were checked, so this was reported as converged and feasible.
  auto p = quad_obstacles(true);
  const Circle& o = p.obstacles.front();
  p.x_goal(0) = o.cx;
  p.x_goal(1) = o.cy;
  Solver<PlanarQuad> s(p);
  const auto r = s.solve(hover(p), Options{});
  const double final_distance = std::hypot(r.xs.back()(0) - o.cx, r.xs.back()(1) - o.cy);
  // The heavy terminal cost pulls into the obstacle and the constraint pushes back, so the final
  // state ends on the boundary (to within the augmented Lagrangian's tolerance), not at the goal.
  EXPECT_GT(final_distance, 0.95 * o.r);
  // and whatever violation is left is reported, so `converged` cannot claim feasibility
  const double terminal_violation = o.r * o.r - final_distance * final_distance;
  EXPECT_GE(r.max_violation, terminal_violation - 1e-12);
  EXPECT_EQ(r.converged, r.max_violation < Options{}.viol_tol && r.termination == Termination::Converged);
}
