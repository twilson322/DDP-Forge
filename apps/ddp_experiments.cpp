// ddp_experiments <converge|mpc|all> [--out results]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "ddp/problems.hpp"

using namespace ddp;

namespace {

/// fopen that never returns null: a missing output directory used to crash in the first fprintf.
std::FILE* open_for_writing(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(1);
  }
  return f;
}

template <typename Sys>
void dump_traj(const std::string& path, const Result<Sys>& r, double dt) {
  std::FILE* f = open_for_writing(path.c_str());
  std::fprintf(f, "t");
  for (std::size_t i = 0; i < Sys::NX; ++i) std::fprintf(f, ",x%zu", i);
  for (std::size_t i = 0; i < Sys::NU; ++i) std::fprintf(f, ",u%zu", i);
  std::fprintf(f, "\n");
  for (std::size_t k = 0; k < r.xs.size(); ++k) {
    std::fprintf(f, "%.4f", k * dt);
    for (std::size_t i = 0; i < Sys::NX; ++i) std::fprintf(f, ",%.6f", r.xs[k](static_cast<int>(i)));
    for (std::size_t i = 0; i < Sys::NU; ++i)
      std::fprintf(f, ",%.6f", k < r.us.size() ? r.us[k](static_cast<int>(i)) : 0.0);
    std::fprintf(f, "\n");
  }
  std::fclose(f);
}

template <typename Sys>
void run_both(const std::string& tag, const Problem<Sys>& p, const std::vector<typename Problem<Sys>::VU>& u0,
              const std::string& out, std::FILE* summary, std::FILE* hist) {
  for (int so = 0; so <= 1; ++so) {
    Options o;
    o.second_order = so == 1;
    Solver<Sys> s(p);
    const auto r = s.solve(u0, o);
    const char* algo = so ? "ddp" : "ilqr";
    std::fprintf(summary, "%s,%s,%d,%d,%d,%.6f,%.3e,%.2f,%.3f\n", tag.c_str(), algo, r.converged, r.iterations,
                 r.outer, r.cost, r.max_violation, r.time_ms, r.time_ms / std::max(1, r.iterations));
    for (std::size_t i = 0; i < r.cost_history.size(); ++i)
      std::fprintf(hist, "%s,%s,%zu,%.8f\n", tag.c_str(), algo, i, r.cost_history[i]);
    std::printf("%-22s %-5s converged=%d (%s) iters=%3d outer=%2d cost=%10.4f viol=%.1e time=%8.1f ms\n", tag.c_str(),
                algo, r.converged, ddp::to_string(r.termination), r.iterations, r.outer, r.cost, r.max_violation, r.time_ms);
    dump_traj(out + "/traj_" + tag + "_" + algo + ".csv", r, p.dt);
  }
}

int exp_converge(const std::string& out) {
  std::FILE* summary = open_for_writing((out + "/converge.csv").c_str());
  std::fprintf(summary, "problem,algo,converged,iterations,outer,cost,max_violation,time_ms,ms_per_iter\n");
  std::FILE* hist = open_for_writing((out + "/cost_history.csv").c_str());
  std::fprintf(hist, "problem,algo,iter,cost\n");
  {
    auto p = cartpole_swingup(false);
    run_both("cartpole", p, zeros<CartPole>(p.N), out, summary, hist);
    auto pb = cartpole_swingup(true);
    run_both("cartpole_bounded", pb, zeros<CartPole>(pb.N), out, summary, hist);
  }
  {
    auto p = acrobot_swingup(true);
    run_both("acrobot_bounded", p, zeros<Acrobot>(p.N), out, summary, hist);
  }
  {
    auto p = quad_obstacles(false);
    run_both("quad2d_free", p, hover<PlanarQuad>(p), out, summary, hist);
    auto pc = quad_obstacles(true);
    run_both("quad2d_obstacles", pc, hover<PlanarQuad>(pc), out, summary, hist);
  }
  std::fclose(summary);
  std::fclose(hist);
  return 0;
}

// Plan offline, track online. The offline iLQR swing-up (4 s, +/-10 N) is tracked on a plant
// with a 20 % heavier pole and 0.5 N s/m cart friction, starting from a perturbed angle, with a
// push of 2, 4 or 6 N s on the cart at t = 5 s. Controllers:
//   tvlqr - the iLQR feedback gains around the offline plan (time-varying LQR), then an
//           infinite-horizon LQR at upright, saturated at the force limit
//   mpc   - receding-horizon iLQR (1.5 s horizon) on a tracking cost around the same plan,
//           warm-started from the shifted previous solution, capped at 1 (real-time
//           iteration, full step), 2, 5 or 50 iterations
int exp_mpc(const std::string& out) {
  std::FILE* f = open_for_writing((out + "/mpc.csv").c_str());
  std::fprintf(f, "push_Ns,controller,max_iters,seed,success,final_angle_err_deg,max_solve_ms,mean_solve_ms,p99_solve_ms,cost\n");
  CartPole plant;
  plant.mp = 0.36;
  plant.b_cart = 0.5;
  const auto offline_p = cartpole_swingup(true);
  Solver<CartPole> offline(offline_p);
  Options oo;
  oo.second_order = true;
  const auto ref = offline.solve(zeros<CartPole>(offline_p.N), oo);
  const double dt = offline_p.dt;
  const int T = 400;  // 8 s
  const int H = 75;   // 1.5 s horizon
  using VX = Problem<CartPole>::VX;
  using VU = Problem<CartPole>::VU;
  auto xref_at = [&](int k) -> VX { return k <= offline_p.N ? ref.xs[static_cast<std::size_t>(k)] : offline_p.x_goal; };
  auto uref_at = [&](int k) -> VU { return k < offline_p.N ? ref.us[static_cast<std::size_t>(k)] : VU::Zero(); };
  // Infinite-horizon LQR at the upright equilibrium (same Q, R as the MPC), used by the TVLQR
  // controller once the plan has ended. The first version kept using the last finite-horizon
  // gain there, which made the baseline weaker than it needs to be.
  Eigen::Matrix<double, 1, 4> K_inf;
  {
    Solver<CartPole> lin(offline_p);
    Eigen::Matrix4d A;
    Eigen::Matrix<double, 4, 1> B;
    lin.linearize(offline_p.x_goal, VU::Zero(), A, B);
    const Eigen::Vector4d qd(10.0, 20.0, 1.0, 1.0);
    const Eigen::Matrix4d Q = qd.asDiagonal();
    Eigen::Matrix4d P = Q;
    for (int i = 0; i < 20000; ++i) {
      const double S = 0.05 + B.dot(P * B);
      P = Q + A.transpose() * P * A - (A.transpose() * P * B) * (B.transpose() * P * A) / S;
    }
    K_inf = -(B.transpose() * P * A) / (0.05 + B.dot(P * B));
  }
  for (double push : {2.0, 4.0, 6.0}) {
  for (const char* ctrl : {"tvlqr", "mpc"}) {
    for (int iters : {1, 2, 5, 50}) {
      if (std::string(ctrl) == "tvlqr" && iters != 1) continue;
      for (int seed = 0; seed < 20; ++seed) {
        std::mt19937_64 rng(static_cast<std::uint64_t>(100 + seed));
        std::normal_distribution<double> N01(0.0, 1.0);
        VX x = offline_p.x0;
        x(1) += 0.05 * N01(rng);  // perturbed start
        std::vector<VU> u_warm;
        for (int k = 0; k < H; ++k) u_warm.push_back(uref_at(k));
        double cost = 0;
        std::vector<double> times;
        std::FILE* log = (seed == 0) ? open_for_writing((out + "/mpc_" + ctrl + "_" + std::to_string(iters) + "_push" +
                                                   std::to_string(static_cast<int>(push)) + ".csv").c_str())
                                     : nullptr;
        if (log) std::fprintf(log, "t,p,theta,u\n");
        for (int t = 0; t < T; ++t) {
          VU u;
          if (std::string(ctrl) == "tvlqr") {
            if (t < offline_p.N) {
              u = uref_at(t) + ref.K[static_cast<std::size_t>(t)] * (x - xref_at(t));
            } else {
              VX e = x - offline_p.x_goal;
              e(1) = std::remainder(e(1), 2 * M_PI);
              u = K_inf * e;
            }
          } else {
            Problem<CartPole> p = offline_p;
            p.x0 = x;
            p.N = H;
            p.Q.diagonal() << 10.0, 20.0, 1.0, 1.0;
            p.R(0, 0) = 0.05;
            p.Qf.diagonal() << 100.0, 200.0, 20.0, 20.0;
            p.x_ref.clear();
            for (int k = 0; k <= H; ++k) p.x_ref.push_back(xref_at(t + k));
            Options o;
            o.max_iter = iters;
            o.max_outer = 3;
            o.always_accept = iters == 1;
            Solver<CartPole> s(p);
            std::vector<VU> warm(static_cast<std::size_t>(H));
            const int sh = t == 0 ? 0 : 1;
            for (int k = 0; k < H; ++k)
              warm[static_cast<std::size_t>(k)] = k + sh < H ? u_warm[static_cast<std::size_t>(k + sh)] : uref_at(t + k);
            const auto r = s.solve(warm, o);
            u = r.us[0];
            u_warm = r.us;
            times.push_back(r.time_ms);
          }
          u(0) = std::clamp(u(0), offline_p.u_min(0), offline_p.u_max(0));
          Arr<double, 4> xa{x(0), x(1), x(2), x(3)};
          const auto y = rk4(plant, xa, Arr<double, 1>{u(0)}, dt);
          x << y[0], y[1], y[2], y[3];
          if (t == 250) x(2) += push / (plant.mc + plant.mp);  // impulse on the cart
          const VX e = x - xref_at(t + 1);
          cost += (0.5 * (e(0) * e(0) + 10 * e(1) * e(1)) + 0.5 * 0.1 * u(0) * u(0)) * dt;
          if (log) std::fprintf(log, "%.3f,%.4f,%.4f,%.4f\n", (t + 1) * dt, x(0), x(1), u(0));
        }
        if (log) std::fclose(log);
        const double final_err = std::abs(std::remainder(x(1) - M_PI, 2 * M_PI)) * 180.0 / M_PI;
        std::sort(times.begin(), times.end());
        auto pct = [&](double q) { return times.empty() ? 0.0 : times[static_cast<std::size_t>(q * (times.size() - 1))]; };
        double mean = 0;
        for (double v : times) mean += v;
        mean = times.empty() ? 0.0 : mean / times.size();
        std::fprintf(f, "%.0f,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f\n", push, ctrl, iters, seed, final_err < 5.0 && std::abs(x(0)) < 1.0,
                     final_err, pct(1.0), mean, pct(0.99), cost);
      }
      std::printf("mpc push=%.0f %-6s iters=%2d done\n", push, ctrl, iters);
    }
  }
  }
  std::fclose(f);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string what = argc > 1 ? argv[1] : "all";
  const std::string out = (argc >= 4 && std::string(argv[2]) == "--out") ? argv[3] : "results";
  std::filesystem::create_directories(out);
  if (what == "converge" || what == "all") exp_converge(out);
  if (what == "mpc" || what == "all") exp_mpc(out);
  return 0;
}
