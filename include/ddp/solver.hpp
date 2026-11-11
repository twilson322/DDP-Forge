#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ddp/dual.hpp"
#include "ddp/systems.hpp"

// Constrained iLQR / DDP.
//   * Discrete dynamics x+ = RK4(f, x, u, dt); Jacobians by forward-mode dual numbers.
//   * DDP adds the second-order dynamics terms sum_i Vx'_i * d2F_i/dz2, evaluated with nested
//     duals. iLQR drops them (Gauss-Newton).
//   * Inequality constraints c(x, u) <= 0 (input bounds, circular obstacles) are handled with
//     a Powell-Hestenes-Rockafellar augmented Lagrangian around the unconstrained solver,
//     in the spirit of ALTRO (Howell, Jackson, Manchester 2019).
//   * Levenberg-Marquardt regularization on V_xx (Tassa, Erez, Todorov 2012) and a backtracking
//     line search on the ratio of actual to expected cost reduction.
namespace ddp {

struct Circle { double cx, cy, r; };

template <typename Sys>
struct Problem {
  static constexpr int NX = static_cast<int>(Sys::NX), NU = static_cast<int>(Sys::NU);
  using VX = Eigen::Matrix<double, NX, 1>;
  using VU = Eigen::Matrix<double, NU, 1>;
  Sys sys{};
  double dt{0.02};
  int N{100};
  VX x0 = VX::Zero(), x_goal = VX::Zero();
  Eigen::Matrix<double, NX, NX> Q = Eigen::Matrix<double, NX, NX>::Zero();
  Eigen::Matrix<double, NX, NX> Qf = Eigen::Matrix<double, NX, NX>::Identity();
  Eigen::Matrix<double, NU, NU> R = Eigen::Matrix<double, NU, NU>::Identity();
  VU u_ref = VU::Zero();  // running cost penalizes u - u_ref (hover thrust for the quad)
  // Optional time-varying state reference (N + 1 entries) for tracking problems; when empty
  // the running and terminal costs use x_goal.
  std::vector<VX> x_ref;
  const VX& xr(int k) const { return x_ref.empty() ? x_goal : x_ref[static_cast<std::size_t>(k)]; }
  bool u_bounds{false};
  VU u_min = VU::Zero(), u_max = VU::Zero();
  std::vector<Circle> obstacles;  // on state indices (0, 1)
  int n_con() const { return (u_bounds ? 2 * NU : 0) + static_cast<int>(obstacles.size()); }
};

struct Options {
  bool second_order{false};  // false = iLQR, true = DDP
  int max_iter{1000};        // total inner iterations across all AL phases
  double tol{1e-7};          // inner convergence: predicted first-order improvement |dV1| / (1 + |J|)
  double rho_init{1e-6}, rho_min{1e-9}, rho_max{1e10};
  double rho_after_failure{1e-6};  // floor once Q_uu has failed to factor: rho_min is too small to help
  double mu_init{10.0}, mu_factor{10.0}, mu_max{1e8};
  double viol_tol{1e-4};
  int max_outer{12};
  // Line search: halve alpha down to min_alpha, accept on an Armijo-style sufficient decrease.
  double min_alpha{1e-4};
  double armijo_fraction{1e-4};  // required fraction of the predicted decrease
  // Real-time iteration: take the full step (alpha = 1) without a line search. Used with a
  // one-iteration budget in MPC, where returning the unchanged warm start (no feedback at all)
  // is worse than an unverified Newton step.
  bool always_accept{false};
};

/// Why the last inner (unconstrained) solve stopped.
enum class Termination { Converged, MaxIterations, RegularizationLimit };
inline const char* to_string(Termination t) {
  switch (t) {
    case Termination::Converged: return "converged";
    case Termination::MaxIterations: return "max_iterations";
    case Termination::RegularizationLimit: return "regularization_limit";
  }
  return "?";
}

template <typename Sys>
struct Result {
  using P = Problem<Sys>;
  std::vector<typename P::VX> xs;
  std::vector<typename P::VU> us;
  std::vector<Eigen::Matrix<double, P::NU, P::NX>> K;  // feedback gains (TVLQR around the solution)
  double cost{0};            // task cost (without the augmented-Lagrangian terms)
  double max_violation{0};
  int iterations{0}, outer{0};
  /// The last inner solve reached stationarity AND every constraint (including the terminal
  /// state's) is within viol_tol. A run that stopped on the iteration cap or because the
  /// regularization blew up is never reported as converged, even if it happens to be feasible.
  bool converged{false};
  Termination termination{Termination::MaxIterations};
  double time_ms{0};
  std::vector<double> cost_history;  // task cost after every accepted iteration
};

template <typename Sys>
class Solver {
 public:
  using P = Problem<Sys>;
  static constexpr int NX = P::NX, NU = P::NU, NZ = NX + NU;
  using VX = typename P::VX;
  using VU = typename P::VU;
  using MXX = Eigen::Matrix<double, NX, NX>;
  using MXU = Eigen::Matrix<double, NX, NU>;
  using MUU = Eigen::Matrix<double, NU, NU>;
  using MUX = Eigen::Matrix<double, NU, NX>;
  using MZZ = Eigen::Matrix<double, NZ, NZ>;
  using VZ = Eigen::Matrix<double, NZ, 1>;

  explicit Solver(const P& p) : p_(p) {}

  VX step(const VX& x, const VU& u) const {
    Arr<double, NX> xa;
    Arr<double, NU> ua;
    for (int i = 0; i < NX; ++i) xa[i] = x(i);
    for (int i = 0; i < NU; ++i) ua[i] = u(i);
    const auto y = rk4(p_.sys, xa, ua, p_.dt);
    VX r;
    for (int i = 0; i < NX; ++i) r(i) = y[i];
    return r;
  }

  // Jacobians of the discrete dynamics: one Dual<double> pass per input direction.
  void linearize(const VX& x, const VU& u, MXX& Fx, MXU& Fu) const {
    for (int j = 0; j < NZ; ++j) {
      Arr<Dual<double>, NX> xd;
      Arr<Dual<double>, NU> ud;
      for (int i = 0; i < NX; ++i) xd[i] = Dual<double>(x(i), i == j ? 1.0 : 0.0);
      for (int i = 0; i < NU; ++i) ud[i] = Dual<double>(u(i), NX + i == j ? 1.0 : 0.0);
      const auto y = rk4(p_.sys, xd, ud, p_.dt);
      for (int i = 0; i < NX; ++i) (j < NX ? Fx(i, j) : Fu(i, j - NX)) = y[i].d;
    }
  }

  // Hessian of lambda' F(z) with respect to z = [x; u], one Dual<Dual<double>> pass per pair.
  MZZ second_order(const VX& x, const VU& u, const VX& lambda) const {
    using DD = Dual<Dual<double>>;
    MZZ H;
    for (int j = 0; j < NZ; ++j)
      for (int k = j; k < NZ; ++k) {
        Arr<DD, NX> xd;
        Arr<DD, NU> ud;
        auto seed = [&](int idx, double val) {
          return DD(Dual<double>(val, idx == j ? 1.0 : 0.0), Dual<double>(idx == k ? 1.0 : 0.0, 0.0));
        };
        for (int i = 0; i < NX; ++i) xd[i] = seed(i, x(i));
        for (int i = 0; i < NU; ++i) ud[i] = seed(NX + i, u(i));
        const auto y = rk4(p_.sys, xd, ud, p_.dt);
        double h = 0.0;
        for (int i = 0; i < NX; ++i) h += lambda(i) * y[i].d.d;
        H(j, k) = H(k, j) = h;
      }
    return H;
  }

  // Constraint values c(x, u) <= 0 and their gradients with respect to z = [x; u].
  int constraints(const VX& x, const VU& u, Eigen::VectorXd& c, Eigen::MatrixXd& J) const {
    const int m = p_.n_con();
    c.resize(m);
    J.setZero(m, NZ);
    int r = 0;
    if (p_.u_bounds) {
      for (int i = 0; i < NU; ++i) {
        c(r) = u(i) - p_.u_max(i);
        J(r++, NX + i) = 1.0;
        c(r) = p_.u_min(i) - u(i);
        J(r++, NX + i) = -1.0;
      }
    }
    for (const Circle& o : p_.obstacles) {
      const double dx = x(0) - o.cx, dy = x(1) - o.cy;
      c(r) = o.r * o.r - dx * dx - dy * dy;
      J(r, 0) = -2.0 * dx;
      J(r, 1) = -2.0 * dy;
      ++r;
    }
    return m;
  }

  double task_cost(const std::vector<VX>& xs, const std::vector<VU>& us) const {
    double J = 0.0;
    for (int k = 0; k < p_.N; ++k) {
      const VX e = xs[k] - p_.xr(k);
      const VU du = us[k] - p_.u_ref;
      J += 0.5 * e.dot(p_.Q * e) + 0.5 * du.dot(p_.R * du);
    }
    const VX e = xs[p_.N] - p_.xr(p_.N);
    return J + 0.5 * e.dot(p_.Qf * e);
  }

  double al_cost(const std::vector<VX>& xs, const std::vector<VU>& us) const {
    double J = 0.0;
    Eigen::VectorXd c;
    Eigen::MatrixXd Jc;
    for (int k = 0; k < p_.N; ++k) {
      const int m = constraints(xs[k], us[k], c, Jc);
      for (int i = 0; i < m; ++i) {
        const double a = std::max(0.0, lam_[k](i) + mu_ * c(i));
        J += (a * a - lam_[k](i) * lam_[k](i)) / (2.0 * mu_);
      }
    }
    const int mN = terminal_constraints(xs[p_.N], c, Jc);
    for (int i = 0; i < mN; ++i) {
      const double a = std::max(0.0, lam_terminal_(i) + mu_ * c(i));
      J += (a * a - lam_terminal_(i) * lam_terminal_(i)) / (2.0 * mu_);
    }
    return J;
  }

  // State-only constraints on the terminal state (the obstacles). Without these the final state
  // was never checked: a goal inside an obstacle would have been reported feasible.
  int terminal_constraints(const VX& x, Eigen::VectorXd& c, Eigen::MatrixXd& J) const {
    const int m = static_cast<int>(p_.obstacles.size());
    c.resize(m);
    J.setZero(m, NX);
    for (int r = 0; r < m; ++r) {
      const Circle& o = p_.obstacles[static_cast<std::size_t>(r)];
      const double dx = x(0) - o.cx, dy = x(1) - o.cy;
      c(r) = o.r * o.r - dx * dx - dy * dy;
      J(r, 0) = -2.0 * dx;
      J(r, 1) = -2.0 * dy;
    }
    return m;
  }

  double max_violation(const std::vector<VX>& xs, const std::vector<VU>& us) const {
    double v = 0.0;
    Eigen::VectorXd c;
    Eigen::MatrixXd Jc;
    for (int k = 0; k < p_.N; ++k) {
      const int m = constraints(xs[k], us[k], c, Jc);
      for (int i = 0; i < m; ++i) v = std::max(v, c(i));
    }
    const int mN = terminal_constraints(xs[p_.N], c, Jc);
    for (int i = 0; i < mN; ++i) v = std::max(v, c(i));
    return v;
  }

  Result<Sys> solve(std::vector<VU> us, const Options& o) {
    const auto t0 = std::chrono::steady_clock::now();
    Result<Sys> R;
    const int N = p_.N, m = p_.n_con();
    lam_.assign(N, Eigen::VectorXd::Zero(m));
    lam_terminal_ = Eigen::VectorXd::Zero(static_cast<int>(p_.obstacles.size()));
    mu_ = m > 0 ? o.mu_init : 1.0;
    std::vector<VX> xs(N + 1);
    xs[0] = p_.x0;
    for (int k = 0; k < N; ++k) xs[k + 1] = step(xs[k], us[k]);
    std::vector<VU> kff(N);
    std::vector<MUX> Kfb(N);
    std::vector<MXX> Fx(N);
    std::vector<MXU> Fu(N);
    double rho = o.rho_init;
    double J = task_cost(xs, us) + al_cost(xs, us);
    R.cost_history.push_back(task_cost(xs, us));

    for (int outer = 0; outer < (m > 0 ? o.max_outer : 1); ++outer) {
      R.outer = outer + 1;
      J = task_cost(xs, us) + al_cost(xs, us);
      R.termination = Termination::MaxIterations;
      for (int it = 0; R.iterations < o.max_iter; ++it) {
        ++R.iterations;
        for (int k = 0; k < N; ++k) linearize(xs[k], us[k], Fx[k], Fu[k]);
        // ------------------------------------------------ backward pass
        double dV1 = 0.0, dV2 = 0.0;
        bool ok = false;
        while (!ok) {
          ok = true;
          dV1 = dV2 = 0.0;
          VX Vx = p_.Qf * (xs[N] - p_.xr(N));
          MXX Vxx = p_.Qf;
          {
            Eigen::VectorXd c;
            Eigen::MatrixXd Jc;
            const int mN = terminal_constraints(xs[N], c, Jc);
            for (int i = 0; i < mN; ++i) {
              const double a = lam_terminal_(i) + mu_ * c(i);
              if (a <= 0.0) continue;  // inactive
              Vx += a * Jc.row(i).transpose();
              Vxx += mu_ * Jc.row(i).transpose() * Jc.row(i);
            }
          }
          for (int k = N - 1; k >= 0; --k) {
            // Stage cost derivatives including the augmented-Lagrangian terms.
            VZ lz = VZ::Zero();
            MZZ lzz = MZZ::Zero();
            lz.template head<NX>() = p_.Q * (xs[k] - p_.xr(k));
            lz.template tail<NU>() = p_.R * (us[k] - p_.u_ref);
            lzz.template topLeftCorner<NX, NX>() = p_.Q;
            lzz.template bottomRightCorner<NU, NU>() = p_.R;
            if (m > 0) {
              Eigen::VectorXd c;
              Eigen::MatrixXd Jc;
              constraints(xs[k], us[k], c, Jc);
              for (int i = 0; i < m; ++i) {
                const double a = lam_[k](i) + mu_ * c(i);
                if (a <= 0.0) continue;  // inactive
                lz += a * Jc.row(i).transpose();
                lzz += mu_ * Jc.row(i).transpose() * Jc.row(i);  // Gauss-Newton on the penalty
              }
            }
            VX Qx = lz.template head<NX>() + Fx[k].transpose() * Vx;
            VU Qu = lz.template tail<NU>() + Fu[k].transpose() * Vx;
            MXX Qxx = lzz.template topLeftCorner<NX, NX>() + Fx[k].transpose() * Vxx * Fx[k];
            MUU Quu = lzz.template bottomRightCorner<NU, NU>() + Fu[k].transpose() * Vxx * Fu[k];
            MUX Qux = lzz.template bottomLeftCorner<NU, NX>() + Fu[k].transpose() * Vxx * Fx[k];
            if (o.second_order) {
              const MZZ H = second_order(xs[k], us[k], Vx);
              Qxx += H.template topLeftCorner<NX, NX>();
              Quu += H.template bottomRightCorner<NU, NU>();
              Qux += H.template bottomLeftCorner<NU, NX>();
            }
            // Levenberg-Marquardt in control space. (Regularizing V_xx instead, Tassa's
            // preferred variant, only adds rho * Fu' Fu ~ rho * dt^2 here, far too weak to
            // cure the negative curvature the DDP second-order terms introduce.)
            Quu += rho * MUU::Identity();
            const Eigen::LLT<MUU> llt(Quu);
            if (llt.info() != Eigen::Success) {  // not positive definite: regularize harder
              rho = std::max(rho * 10.0, o.rho_after_failure);
              ok = false;
              break;
            }
            kff[k] = -llt.solve(Qu);
            Kfb[k] = -llt.solve(Qux);
            dV1 += kff[k].dot(Qu);
            dV2 += 0.5 * kff[k].dot(Quu * kff[k]);
            Vx = Qx + Kfb[k].transpose() * Quu * kff[k] + Kfb[k].transpose() * Qu + Qux.transpose() * kff[k];
            Vxx = Qxx + Kfb[k].transpose() * Quu * Kfb[k] + Kfb[k].transpose() * Qux + Qux.transpose() * Kfb[k];
            Vxx = 0.5 * (Vxx + Vxx.transpose());
          }
          if (!ok && rho > o.rho_max) break;
        }
        if (!ok) {
          R.termination = Termination::RegularizationLimit;
          break;
        }
        // ------------------------------------------------ forward pass with line search
        bool accepted = false;
        double Jnew = J;
        std::vector<VX> xn(N + 1);
        std::vector<VU> un(N);
        for (double alpha = 1.0; alpha > o.min_alpha; alpha *= 0.5) {
          xn[0] = p_.x0;
          for (int k = 0; k < N; ++k) {
            un[k] = us[k] + alpha * kff[k] + Kfb[k] * (xn[k] - xs[k]);
            xn[k + 1] = step(xn[k], un[k]);
          }
          Jnew = task_cost(xn, un) + al_cost(xn, un);
          const double expected = -(alpha * dV1 + alpha * alpha * dV2);
          if (o.always_accept || (std::isfinite(Jnew) && (J - Jnew) > o.armijo_fraction * std::max(expected, 0.0) && Jnew < J)) {
            accepted = true;
            if (verbose_) std::printf("  it %3d J %.6f alpha %.4f rho %.2e dV1 %.3e\n", R.iterations, Jnew, alpha, rho, dV1);
            break;
          }
        }
        if (accepted) {
          xs.swap(xn);
          us.swap(un);
          const double dJ = J - Jnew;
          J = Jnew;
          R.cost_history.push_back(task_cost(xs, us));
          rho = std::max(o.rho_min, rho / 4.0);
          // Stop on a small predicted first-order improvement, i.e. near stationarity.
          // An earlier version stopped on a small *actual* cost change; iLQR then quit at a
          // point 5.9 % above the optimum on the cart-pole, crawling with tiny line-search
          // steps while the predicted improvement was still large (IPOPT, warm-started there,
          // walked on to the true local optimum). dJ alone only detects stagnation.
          if (std::abs(dV1) < o.tol * (1.0 + std::abs(J)) || dJ < 1e-12 * (1.0 + std::abs(J))) {
            R.termination = Termination::Converged;
            break;
          }
        } else {
          rho *= 10.0;
          if (std::abs(dV1) < 1e-10) {  // already at a stationary point
            R.termination = Termination::Converged;
            break;
          }
          if (rho > o.rho_max) {
            R.termination = Termination::RegularizationLimit;
            break;
          }
        }
      }
      // ------------------------------------------------ augmented-Lagrangian update
      const double viol = max_violation(xs, us);
      if (m == 0 || viol < o.viol_tol) {
        R.converged = R.termination == Termination::Converged;
        break;
      }
      Eigen::VectorXd c;
      Eigen::MatrixXd Jc;
      for (int k = 0; k < N; ++k) {
        constraints(xs[k], us[k], c, Jc);
        for (int i = 0; i < m; ++i) lam_[k](i) = std::max(0.0, lam_[k](i) + mu_ * c(i));
      }
      const int mN = terminal_constraints(xs[N], c, Jc);
      for (int i = 0; i < mN; ++i) lam_terminal_(i) = std::max(0.0, lam_terminal_(i) + mu_ * c(i));
      mu_ = std::min(mu_ * o.mu_factor, o.mu_max);
      rho = o.rho_init;
    }
    R.xs = xs;
    R.us = us;
    R.K = Kfb;
    R.cost = task_cost(xs, us);
    R.max_violation = max_violation(xs, us);
    R.time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return R;
  }

  void set_verbose(bool v) { verbose_ = v; }
  const P& problem() const { return p_; }
  P& problem() { return p_; }

 private:
  P p_;
  std::vector<Eigen::VectorXd> lam_;
  Eigen::VectorXd lam_terminal_;
  double mu_{1.0};
  bool verbose_{false};
};

}  // namespace ddp
