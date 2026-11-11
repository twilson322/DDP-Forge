#pragma once
#include <cmath>
#include <vector>

#include "ddp/solver.hpp"

// The benchmark problems. scripts/crosscheck_casadi.py builds the same discrete problems
// (same RK4, dt, horizon, weights, bounds) so the optimal costs can be compared directly.
namespace ddp {

inline Problem<CartPole> cartpole_swingup(bool bounded) {
  Problem<CartPole> p;
  p.dt = 0.02;
  p.N = 200;  // 4 s
  p.x0.setZero();
  p.x_goal << 0.0, M_PI, 0.0, 0.0;
  p.Q.diagonal() << 0.1, 0.1, 0.01, 0.01;
  p.R(0, 0) = 0.05;
  p.Qf.diagonal() << 100.0, 200.0, 20.0, 20.0;
  p.u_bounds = bounded;
  p.u_min << -10.0;
  p.u_max << 10.0;
  return p;
}

inline Problem<Acrobot> acrobot_swingup(bool bounded) {
  Problem<Acrobot> p;
  p.dt = 0.02;
  p.N = 250;  // 5 s
  p.x0.setZero();
  p.x_goal << M_PI, 0.0, 0.0, 0.0;
  p.Q.diagonal() << 0.05, 0.05, 0.005, 0.005;
  p.R(0, 0) = 0.01;
  p.Qf.diagonal() << 500.0, 500.0, 50.0, 50.0;
  p.u_bounds = bounded;
  p.u_min << -12.0;
  p.u_max << 12.0;
  return p;
}

inline Problem<PlanarQuad> quad_obstacles(bool with_obstacles) {
  Problem<PlanarQuad> p;
  p.dt = 0.02;
  p.N = 150;  // 3 s
  p.x0.setZero();
  p.x_goal << 6.0, 3.0, 0.0, 0.0, 0.0, 0.0;
  p.Q.diagonal() << 0.01, 0.01, 0.1, 0.01, 0.01, 0.01;
  p.R.diagonal() << 0.02, 0.02;
  p.Qf.diagonal() << 500.0, 500.0, 200.0, 50.0, 50.0, 20.0;
  p.u_ref << 0.5 * p.sys.m * p.sys.g, 0.5 * p.sys.m * p.sys.g;
  p.u_bounds = true;
  p.u_min << 0.0, 0.0;
  p.u_max << 8.0, 8.0;
  if (with_obstacles) p.obstacles = {{2.0, 1.2, 0.8}, {3.6, 2.3, 0.7}, {4.6, 0.8, 0.6}};
  return p;
}

template <typename Sys>
std::vector<typename Problem<Sys>::VU> zeros(int N) {
  return std::vector<typename Problem<Sys>::VU>(static_cast<std::size_t>(N), Problem<Sys>::VU::Zero());
}
template <typename Sys>
std::vector<typename Problem<Sys>::VU> hover(const Problem<Sys>& p) {
  return std::vector<typename Problem<Sys>::VU>(static_cast<std::size_t>(p.N), p.u_ref);
}

}  // namespace ddp
