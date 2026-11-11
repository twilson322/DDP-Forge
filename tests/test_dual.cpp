#include <gtest/gtest.h>

#include <cmath>

#include "ddp/problems.hpp"

using namespace ddp;

TEST(Dual, ElementaryDerivatives) {
  const double x = 0.7;
  const Dual<double> d(x, 1.0);
  EXPECT_NEAR(sin(d).d, std::cos(x), 1e-15);
  EXPECT_NEAR(cos(d).d, -std::sin(x), 1e-15);
  EXPECT_NEAR(exp(d).d, std::exp(x), 1e-15);
  EXPECT_NEAR(sqrt(d).d, 0.5 / std::sqrt(x), 1e-15);
  EXPECT_NEAR(tanh(d).d, 1.0 - std::tanh(x) * std::tanh(x), 1e-15);
  EXPECT_NEAR((d * d * d / (d + 1.0)).d, (3 * x * x * (x + 1) - x * x * x) / ((x + 1) * (x + 1)), 1e-14);
}

TEST(Dual, NestedGivesSecondDerivative) {
  // f(x) = sin(x) * x^2, f'' = -sin(x) x^2 + 4 x cos(x) + 2 sin(x)
  const double x = 1.3;
  using DD = Dual<Dual<double>>;
  const DD z(Dual<double>(x, 1.0), Dual<double>(1.0, 0.0));
  const DD f = sin(z) * z * z;
  EXPECT_NEAR(f.d.d, -std::sin(x) * x * x + 4 * x * std::cos(x) + 2 * std::sin(x), 1e-13);
}

template <typename Sys>
void check_derivatives(const Problem<Sys>& p, const typename Problem<Sys>::VX& x,
                       const typename Problem<Sys>::VU& u, const typename Problem<Sys>::VX& lam) {
  constexpr int NX = Problem<Sys>::NX, NU = Problem<Sys>::NU, NZ = NX + NU;
  Solver<Sys> s(p);
  Eigen::Matrix<double, NX, NX> Fx;
  Eigen::Matrix<double, NX, NU> Fu;
  s.linearize(x, u, Fx, Fu);
  const double e = 1e-6;
  for (int j = 0; j < NZ; ++j) {  // Jacobian vs central differences
    auto xp = x, xm = x;
    auto up = u, um = u;
    if (j < NX) { xp(j) += e; xm(j) -= e; } else { up(j - NX) += e; um(j - NX) -= e; }
    // Concrete type, not auto: with auto this is an Eigen expression holding references to
    // two temporaries that die at the semicolon (GCC's -Wmaybe-uninitialized caught it).
    const Eigen::Matrix<double, NX, 1> col = (s.step(xp, up) - s.step(xm, um)) / (2 * e);
    for (int i = 0; i < NX; ++i) EXPECT_NEAR(j < NX ? Fx(i, j) : Fu(i, j - NX), col(i), 1e-6);
  }
  const auto H = s.second_order(x, u, lam);  // Hessian of lam' F vs differences of the Jacobian
  for (int j = 0; j < NZ; ++j) {
    auto xp = x, xm = x;
    auto up = u, um = u;
    if (j < NX) { xp(j) += e; xm(j) -= e; } else { up(j - NX) += e; um(j - NX) -= e; }
    Eigen::Matrix<double, NX, NX> Fxp, Fxm;
    Eigen::Matrix<double, NX, NU> Fup, Fum;
    s.linearize(xp, up, Fxp, Fup);
    s.linearize(xm, um, Fxm, Fum);
    for (int k = 0; k < NZ; ++k) {
      const double dp = (k < NX ? lam.dot(Fxp.col(k)) : lam.dot(Fup.col(k - NX)));
      const double dm = (k < NX ? lam.dot(Fxm.col(k)) : lam.dot(Fum.col(k - NX)));
      EXPECT_NEAR(H(j, k), (dp - dm) / (2 * e), 1e-5);
    }
  }
}

TEST(Dual, CartPoleDerivativesMatchFiniteDifferences) {
  Problem<CartPole>::VX x, lam;
  x << 0.2, 2.5, -0.4, 1.1;
  lam << 1.0, -3.0, 0.5, 2.0;
  check_derivatives(cartpole_swingup(false), x, Problem<CartPole>::VU::Constant(3.0), lam);
}
TEST(Dual, AcrobotDerivativesMatchFiniteDifferences) {
  Problem<Acrobot>::VX x, lam;
  x << 1.0, -0.7, 2.0, -1.5;
  lam << -2.0, 1.0, 0.3, 4.0;
  check_derivatives(acrobot_swingup(false), x, Problem<Acrobot>::VU::Constant(-2.0), lam);
}
TEST(Dual, QuadDerivativesMatchFiniteDifferences) {
  Problem<PlanarQuad>::VX x, lam;
  x << 0.3, -0.2, 0.4, 0.5, -0.3, 0.7;
  lam << 1, -2, 0.5, 3, 1, -1;
  Problem<PlanarQuad>::VU u;
  u << 4.0, 6.0;
  check_derivatives(quad_obstacles(false), x, u, lam);
}
