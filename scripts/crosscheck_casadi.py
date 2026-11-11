#!/usr/bin/env python3
"""Independent check of the C++ solutions with CasADi + IPOPT.

Each problem is rebuilt as a multiple-shooting NLP with exactly the same discrete dynamics
(RK4, same dt and horizon), cost weights, bounds and obstacles as include/ddp/problems.hpp.
IPOPT is run twice: from the same naive initial guess the C++ solvers used, and warm-started
at the C++ solution. If the C++ result is a genuine local optimum, the warm-started IPOPT run
stays at the same cost.
"""
import argparse
import pathlib
import time

import casadi as ca
import numpy as np
import pandas as pd


def cartpole_f(x, u, mc=1.0, mp=0.3, l=0.5, g=9.81):
    s, c = ca.sin(x[1]), ca.cos(x[1])
    den = mc + mp * s * s
    pdd = (u[0] + mp * s * (l * x[3] ** 2 + g * c)) / den
    tdd = (-u[0] * c - mp * l * x[3] ** 2 * c * s - (mc + mp) * g * s) / (l * den)
    return ca.vertcat(x[2], x[3], pdd, tdd)


def acrobot_f(x, u, m1=1.0, m2=1.0, l1=1.0, lc1=0.5, lc2=0.5, g=9.81):
    I1 = m1 * lc1**2 + m1 * l1**2 / 12.0
    I2 = m2 * lc2**2 + m2 / 12.0
    c2, s1, s2, s12 = ca.cos(x[1]), ca.sin(x[0]), ca.sin(x[1]), ca.sin(x[0] + x[1])
    M11 = I1 + I2 + m2 * l1**2 + 2 * m2 * l1 * lc2 * c2
    M12 = I2 + m2 * l1 * lc2 * c2
    M22 = I2
    h = m2 * l1 * lc2 * s2
    Cq1 = -h * x[3] * (2 * x[2] + x[3])
    Cq2 = h * x[2] ** 2
    G1 = -m1 * g * lc1 * s1 - m2 * g * (l1 * s1 + lc2 * s12)
    G2 = -m2 * g * lc2 * s12
    r1, r2 = G1 - Cq1, G2 - Cq2 + u[0]
    det = M11 * M22 - M12 * M12
    return ca.vertcat(x[2], x[3], (M22 * r1 - M12 * r2) / det, (M11 * r2 - M12 * r1) / det)


def quad_f(x, u, m=1.0, I=0.02, r=0.2, g=9.81):
    T = u[0] + u[1]
    return ca.vertcat(x[3], x[4], x[5], -T * ca.sin(x[2]) / m, T * ca.cos(x[2]) / m - g, r / I * (u[1] - u[0]))


def rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


PROBLEMS = {
    "cartpole": dict(f=cartpole_f, nx=4, nu=1, dt=0.02, N=200, x0=[0, 0, 0, 0], xg=[0, np.pi, 0, 0],
                     Q=[0.1, 0.1, 0.01, 0.01], R=[0.05], Qf=[100, 200, 20, 20], ub=None, uref=[0]),
    "cartpole_bounded": dict(f=cartpole_f, nx=4, nu=1, dt=0.02, N=200, x0=[0, 0, 0, 0], xg=[0, np.pi, 0, 0],
                             Q=[0.1, 0.1, 0.01, 0.01], R=[0.05], Qf=[100, 200, 20, 20], ub=([-10], [10]), uref=[0]),
    "acrobot_bounded": dict(f=acrobot_f, nx=4, nu=1, dt=0.02, N=250, x0=[0, 0, 0, 0], xg=[np.pi, 0, 0, 0],
                            Q=[0.05, 0.05, 0.005, 0.005], R=[0.01], Qf=[500, 500, 50, 50], ub=([-12], [12]), uref=[0]),
    "quad2d_free": dict(f=quad_f, nx=6, nu=2, dt=0.02, N=150, x0=[0] * 6, xg=[6, 3, 0, 0, 0, 0],
                        Q=[0.01, 0.01, 0.1, 0.01, 0.01, 0.01], R=[0.02, 0.02], Qf=[500, 500, 200, 50, 50, 20],
                        ub=([0, 0], [8, 8]), uref=[4.905, 4.905]),
    "quad2d_obstacles": dict(f=quad_f, nx=6, nu=2, dt=0.02, N=150, x0=[0] * 6, xg=[6, 3, 0, 0, 0, 0],
                             Q=[0.01, 0.01, 0.1, 0.01, 0.01, 0.01], R=[0.02, 0.02], Qf=[500, 500, 200, 50, 50, 20],
                             ub=([0, 0], [8, 8]), uref=[4.905, 4.905],
                             obstacles=[(2.0, 1.2, 0.8), (3.6, 2.3, 0.7), (4.6, 0.8, 0.6)]),
}


def solve(spec, X_init, U_init):
    nx, nu, N, dt = spec["nx"], spec["nu"], spec["N"], spec["dt"]
    opti = ca.Opti()
    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)
    Q, R, Qf = np.diag(spec["Q"]), np.diag(spec["R"]), np.diag(spec["Qf"])
    xg, uref = np.array(spec["xg"]), np.array(spec["uref"])
    J = 0
    for k in range(N):
        e = X[:, k] - xg
        du = U[:, k] - uref
        J += 0.5 * ca.mtimes([e.T, Q, e]) + 0.5 * ca.mtimes([du.T, R, du])
        opti.subject_to(X[:, k + 1] == rk4(spec["f"], X[:, k], U[:, k], dt))
        if spec["ub"] is not None:
            opti.subject_to(opti.bounded(np.array(spec["ub"][0]), U[:, k], np.array(spec["ub"][1])))
        for (cx, cy, r) in spec.get("obstacles", []):
            opti.subject_to((X[0, k] - cx) ** 2 + (X[1, k] - cy) ** 2 >= r * r)
    e = X[:, N] - xg
    J += 0.5 * ca.mtimes([e.T, Qf, e])
    opti.subject_to(X[:, 0] == np.array(spec["x0"]))
    opti.minimize(J)
    opti.set_initial(X, X_init)
    opti.set_initial(U, U_init)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 3000, "tol": 1e-9})
    t0 = time.time()
    try:
        sol = opti.solve()
        return float(sol.value(J)), True, opti.stats()["iter_count"], time.time() - t0
    except RuntimeError:
        return float(opti.debug.value(J)), False, opti.stats()["iter_count"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    res = pathlib.Path(ap.parse_args().results)
    cpp = pd.read_csv(res / "converge.csv")
    rows = []
    for name, spec in PROBLEMS.items():
        nx, nu, N = spec["nx"], spec["nu"], spec["N"]
        # 1. naive initial guess, same as the C++ solvers: states at x0, controls at u_ref
        X0 = np.tile(np.array(spec["x0"], float)[:, None], (1, N + 1))
        U0 = np.tile(np.array(spec["uref"], float)[:, None], (1, N))
        J_cold, ok_cold, it_cold, t_cold = solve(spec, X0, U0)
        for algo in ["ilqr", "ddp"]:
            tr = pd.read_csv(res / f"traj_{name}_{algo}.csv")
            Xc = tr[[f"x{i}" for i in range(nx)]].to_numpy().T
            Uc = tr[[f"u{i}" for i in range(nu)]].to_numpy().T[:, :N]
            J_warm, ok_warm, it_warm, _ = solve(spec, Xc, Uc)
            J_cpp = float(cpp[(cpp.problem == name) & (cpp.algo == algo)].cost.iloc[0])
            rows.append({"problem": name, "solver": algo, "C++ cost": J_cpp,
                         "IPOPT warm-started at C++ solution": J_warm, "IPOPT iterations (warm)": it_warm,
                         "relative gap": (J_cpp - J_warm) / J_warm,
                         "IPOPT from naive guess": J_cold, "IPOPT time naive [s]": t_cold, "IPOPT ok": ok_cold and ok_warm})
        print(f"{name:18s} IPOPT cold {J_cold:10.4f} ({it_cold} it, {t_cold:.2f} s)")
    df = pd.DataFrame(rows)
    df.to_csv(res / "crosscheck.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(5))


if __name__ == "__main__":
    main()
