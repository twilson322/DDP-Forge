#pragma once
#include <array>
#include <cmath>
#include <string>

#include "ddp/dual.hpp"

// SI units throughout (kg, m, s, N, N m, rad).
// Continuous-time dynamics written once, templated on the scalar type, so the same code is
// evaluated in double, Dual<double> (Jacobians) and Dual<Dual<double>> (second derivatives).
namespace ddp {

template <typename S, std::size_t N>
using Arr = std::array<S, N>;

// Classic 4th-order Runge-Kutta, used as the discrete-time model F(x, u) for every solver
// (and replicated exactly in the CasADi cross-check).
template <typename Sys, typename S>
Arr<S, Sys::NX> rk4(const Sys& sys, const Arr<S, Sys::NX>& x, const Arr<S, Sys::NU>& u, double dt) {
  using X = Arr<S, Sys::NX>;
  auto axpy = [](const X& a, const X& b, double h) {
    X r;
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i] * h;
    return r;
  };
  const X k1 = sys.f(x, u);
  const X k2 = sys.f(axpy(x, k1, 0.5 * dt), u);
  const X k3 = sys.f(axpy(x, k2, 0.5 * dt), u);
  const X k4 = sys.f(axpy(x, k3, dt), u);
  X r;
  for (std::size_t i = 0; i < x.size(); ++i) r[i] = x[i] + (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0);
  return r;
}

// Cart-pole, theta = 0 hanging down, theta = pi upright (Tedrake, Underactuated Robotics, ch. 3).
// x = [cart position, pole angle, cart velocity, pole rate], u = [cart force].
struct CartPole {
  static constexpr std::size_t NX = 4, NU = 1;
  static constexpr const char* name = "cartpole";
  double mc{1.0}, mp{0.3}, l{0.5}, g{9.81}, b_cart{0.0};
  template <typename S>
  Arr<S, 4> f(const Arr<S, 4>& x, const Arr<S, 1>& u) const {
    using std::sin;
    using std::cos;
    const S s = sin(x[1]), c = cos(x[1]);
    const S den = S(mc) + S(mp) * s * s;
    const S force = u[0] - S(b_cart) * x[2];
    const S pdd = (force + S(mp) * s * (S(l) * x[3] * x[3] + S(g) * c)) / den;
    const S tdd = (-(force * c) - S(mp * l) * x[3] * x[3] * c * s - S((mc + mp) * g) * s) / (S(l) * den);
    return {x[2], x[3], pdd, tdd};
  }
};

// Acrobot (Spong 1995): actuated elbow, passive shoulder. q = 0 hanging down.
// x = [q1, q2, dq1, dq2], u = [elbow torque].
struct Acrobot {
  static constexpr std::size_t NX = 4, NU = 1;
  static constexpr const char* name = "acrobot";
  double m1{1.0}, m2{1.0}, l1{1.0}, l2{1.0}, lc1{0.5}, lc2{0.5}, g{9.81};  // kg, m, m/s^2
  // uniform rods: m l^2 / 12 about the centre of mass, shifted to the joint
  double I1() const { return m1 * lc1 * lc1 + m1 * l1 * l1 / 12.0; }  // about the shoulder
  double I2() const { return m2 * lc2 * lc2 + m2 * l2 * l2 / 12.0; }  // about the elbow
  template <typename S>
  Arr<S, 4> f(const Arr<S, 4>& x, const Arr<S, 1>& u) const {
    using std::sin;
    using std::cos;
    const S c2 = cos(x[1]), s1 = sin(x[0]), s2 = sin(x[1]), s12 = sin(x[0] + x[1]);
    const S M11 = S(I1() + I2() + m2 * l1 * l1) + S(2.0 * m2 * l1 * lc2) * c2;
    const S M12 = S(I2()) + S(m2 * l1 * lc2) * c2;
    const S M22 = S(I2());
    const S h = S(m2 * l1 * lc2) * s2;
    // C(q, dq) dq
    const S Cq1 = -(h * x[3] * (x[0 + 2] * 2.0 + x[3]));
    const S Cq2 = h * x[2] * x[2];
    const S G1 = -(S(m1 * g * lc1) * s1) - S(m2 * g) * (S(l1) * s1 + S(lc2) * s12);
    const S G2 = -(S(m2 * g * lc2) * s12);
    const S r1 = G1 - Cq1;
    const S r2 = G2 - Cq2 + u[0];
    const S det = M11 * M22 - M12 * M12;
    return {x[2], x[3], (M22 * r1 - M12 * r2) / det, (M11 * r2 - M12 * r1) / det};
  }
};

// Planar quadrotor. x = [px, py, theta, vx, vy, omega], u = [left thrust, right thrust].
struct PlanarQuad {
  static constexpr std::size_t NX = 6, NU = 2;
  static constexpr const char* name = "quad2d";
  double m{1.0}, I{0.02}, r{0.2}, g{9.81};
  template <typename S>
  Arr<S, 6> f(const Arr<S, 6>& x, const Arr<S, 2>& u) const {
    using std::sin;
    using std::cos;
    const S T = u[0] + u[1];
    return {x[3], x[4], x[5], -(T * sin(x[2])) / m, T * cos(x[2]) / m - S(g), S(r / I) * (u[1] - u[0])};
  }
};

}  // namespace ddp
