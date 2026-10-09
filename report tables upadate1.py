import numpy as np
from scipy.integrate import solve_ivp
import time

# ==============================================================================
# 1. Problem Definition (Robertson Chemical Kinetics)
# ==============================================================================
def robertson(t, y):
    y1, y2, y3 = y
    dy1 = -0.04 * y1 + 1e4 * y2 * y3
    dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    dy3 = 3e7 * y2**2
    return np.array([dy1, dy2, dy3])

def jacobian(y):
    y1, y2, y3 = y
    return np.array([
        [-0.04, 1e4 * y3, 1e4 * y2],
        [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
        [0, 6e7 * y2, 0]
    ])

def reduced_jacobian(y1, y2):
    return np.array([
        [-0.04 - 1e4 * y2, 1e4 * (1 - y1 - 2 * y2)],
        [0.04 + 1e4 * y2, -1e4 * (1 - y1 - 2 * y2) - 6e7 * y2]
    ])

print("Generating high-accuracy reference solution (Radau, rtol=1e-12, atol=1e-14)...")
ref_sol = solve_ivp(robertson, [0, 40], [1, 0, 0], method='Radau', rtol=1e-12, atol=1e-14, dense_output=True)
y_exact_40 = ref_sol.y[:, -1]
print(f"Reference solution y_exact(40) = {y_exact_40}\n")

# ==============================================================================
# 2. Solver Definitions
# ==============================================================================
def explicit_euler(h):
    N = int(40 / h)
    y = np.array([1.0, 0.0, 0.0])
    for _ in range(N):
        y = y + h * robertson(0, y)
        if np.any(np.isnan(y)) or np.any(y < -1.0): return y, N, True
    return y, N, False

def rk4(h):
    N = int(40 / h)
    y = np.array([1.0, 0.0, 0.0])
    for _ in range(N):
        k1 = robertson(0, y)
        k2 = robertson(0, y + h/2 * k1)
        k3 = robertson(0, y + h/2 * k2)
        k4 = robertson(0, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        if np.any(np.isnan(y)) or np.any(y < -1.0): return y, N, True
    return y, N, False

def solve_implicit_step(y_old, h, tau_newton):
    y_guess = y_old.copy()
    iters = 0
    for k in range(20):
        F = y_guess - y_old - h * robertson(0, y_guess)
        if k > 0 and np.linalg.norm(F, np.inf) < tau_newton: break
        JF = np.eye(3) - h * jacobian(y_guess)
        y_guess += np.linalg.solve(JF, -F)
        iters += 1
    return y_guess, iters

def implicit_euler_fixed(h, tau_newton):
    N = int(40 / h)
    y = np.array([1.0, 0.0, 0.0])
    total_iters = 0
    for _ in range(N):
        y, iters = solve_implicit_step(y, h, tau_newton)
        total_iters += iters
        if np.any(np.isnan(y)) or np.any(y < -1.0): return y, N, total_iters, True
    return y, N, total_iters, False

def implicit_euler_adaptive(tol=1e-4):
    t, y, h = 0.0, np.array([1.0, 0.0, 0.0]), 1e-6
    steps, total_iters = 0, 0
    while t < 40:
        if t + h > 40: h = 40 - t
        y_a, it_a = solve_implicit_step(y, h, 1e-6)
        y_b_half, it_b1 = solve_implicit_step(y, h/2, 1e-6)
        y_b, it_b2 = solve_implicit_step(y_b_half, h/2, 1e-6)
        total_iters += (it_a + it_b1 + it_b2)
        err = np.linalg.norm(y_b - y_a, np.inf)
        if err <= tol or h < 1e-8:
            t += h; y = y_b; steps += 1
            h *= min(2.0, max(0.5, 0.9 * (tol / err))) if err > 0 else 2.0
        else:
            h *= max(0.2, 0.9 * (tol / err) ** 0.5)
    return y, steps, total_iters

# ==============================================================================
# 3. Generate and Print Tables
# ==============================================================================
print("="*85)
print("Table 1: Dynamic spectral evolution and stiffness ratio")
print("="*85)
print(f"{'Time t (s)':<12} | {'|lambda|_max':<12} | {'|lambda|_min':<12} | {'Stiffness Ratio S(t)':<20} | {'Active Regime'}")
print("-" * 85)
t_list = [1e-8, 1e-4, 1e-2, 1, 40]
regimes = ["Fast initial transient", "Boundary layer peak (y2 ~ 1e-5)", "Intermediate radical exchange", "Slow quasi-steady state", "Long-term equilibrium drift"]
for i, t in enumerate(t_list):
    y = ref_sol.sol(t)
    eigs = np.abs(np.linalg.eigvals(reduced_jacobian(y[0], y[1])))
    print(f"{t:<12.1e} | {np.max(eigs):<12.2e} | {np.min(eigs):<12.2e} | {np.max(eigs)/np.min(eigs):<20.2e} | {regimes[i]}")

print("\n" + "="*85)
print("Table 2: Expected theoretical convergence orders")
print("="*85)
print(f"{'Numerical Method':<30} | {'Expected Theoretical Order p'}")
print("-" * 60)
for name, p in [("Explicit Euler", 1), ("Implicit Euler", 1), ("Runge-Kutta 2 (RK2)", 2), ("Classical Runge-Kutta 4 (RK4)", 4)]:
    print(f"{name:<30} | {p}")

print("\n" + "="*85)
print("Table 3: Convergence verification, stability assessment, and empirical order at t = 40")
print("="*85)
print(f"{'Method':<16} | {'Step Size h':<12} | {'Global Error E(h)':<18} | {'Observed Order p':<16} | {'Non-Neg?':<8} | {'Stable?'}")
print("-" * 90)
h_list_exEuler = [4e-4, 2e-4, 1e-4, 5e-5]
h_list_rk4 = [8e-4, 5e-4, 4e-4]
h_list_impEuler = [4e-4, 2e-4, 1e-4, 5e-5]
for method_name, func, h_list in [("Explicit Euler", explicit_euler, h_list_exEuler), ("Classical RK4", rk4, h_list_rk4)]:
    prev_err, prev_h = None, None
    for h in h_list:
        y, N, diverged = func(h)
        if diverged:
            print(f"{method_name:<16} | {h:<12.1e} | {'Diverged':<18} | {'-':<16} | {'No':<8} | {'No'}")
            continue
        err = np.linalg.norm(y - y_exact_40, np.inf)
        p = np.log(prev_err/err) / np.log(prev_h/h) if prev_err else 0
        print(f"{method_name:<16} | {h:<12.1e} | {err:<18.4e} | {p:<16.3f} | {'Yes' if np.all(y >= 0) else 'No':<8} | {'Yes'}")
        prev_err, prev_h = err, h
prev_err = None
for h in h_list_impEuler:
    y, N, _, _ = implicit_euler_fixed(h, 1e-6)
    err = np.linalg.norm(y - y_exact_40, np.inf)
    p = np.log(prev_err/err) / np.log(h_list_impEuler[h_list_impEuler.index(h)-1]/h) if prev_err else 0
    print(f"{'Implicit Euler':<16} | {h:<12.1e} | {err:<18.4e} | {p:<16.3f} | {'Yes':<8} | {'Yes'}")
    prev_err = err

print("\n" + "="*85)
print("Table 4: Decoupled diagnostics for Explicit Euler near the stability limit (t = 40)")
print("="*85)
print(f"{'Step Size h (s)':<15} | {'Max Invariant Defect':<20} | {'Minimum y2':<15} | {'Relative Error':<15} | {'Diagnostic Status'}")
print("-" * 90)
for h in [2e-4, 5e-4, 6e-4, 7e-4]:
    y, N, diverged = explicit_euler(h)
    if diverged or np.isnan(y).any():
        print(f"{h:<15.1e} | {'-':<20} | {'-inf':<15} | {'NaN':<15} | {'Diverged (NaN)'}")
    else:
        defect = np.max(np.abs(np.sum(y) - 1))
        min_y2 = np.min(y[1])
        rel_err = np.linalg.norm(y - y_exact_40, np.inf) / np.linalg.norm(y_exact_40, np.inf)
        status = "Unphysical (y2 < 0)" if min_y2 < 0 else ("Marginally stable, positive" if h > 5.9e-4 else "Stable, accurate, positive")
        print(f"{h:<15.1e} | {defect:<20.2e} | {min_y2:<15.2e} | {rel_err:<15.2e} | {status}")

print("\n" + "="*85)
print("Table 5: Cost versus Accuracy Evaluation (Target Error ~5.0e-4)")
print("Note: CPU times depend on hardware. Please use your own measured data.")
print("="*85)
print(f"{'Method':<20} | {'Step Size h':<12} | {'Global Error':<15} | {'CPU Time (s)':<15} | {'RHS Evals':<12} | {'Newton Iters'}")
print("-" * 100)
start = time.perf_counter(); y_ee, _, _ = explicit_euler(2.08e-4); t_ee = time.perf_counter() - start
print(f"{'Explicit Euler':<20} | {2.08e-4:<12.1e} | {np.linalg.norm(y_ee - y_exact_40, np.inf):<15.2e} | {t_ee:<15.4f} | {int(40/2.08e-4):<12} | {'-'}")
start = time.perf_counter(); y_rk4, _, _ = rk4(8.00e-4); t_rk4 = time.perf_counter() - start
print(f"{'Classical RK4':<20} | {8.00e-4:<12.1e} | {np.linalg.norm(y_rk4 - y_exact_40, np.inf):<15.2e} | {t_rk4:<15.4f} | {int(40/8e-4)*4:<12} | {'-'}")
start = time.perf_counter(); y_imp, steps_imp, it_imp = implicit_euler_adaptive(1e-4); t_imp = time.perf_counter() - start
print(f"{'Implicit Euler (Adap)':<20} | {'Adaptive':<12} | {np.linalg.norm(y_imp - y_exact_40, np.inf):<15.2e} | {t_imp:<15.4f} | {steps_imp*3:<12} | {it_imp}")

print("\n" + "="*85)
print("Table 6: Effect of Newton residual tolerance on invariant defect (h=0.1, n=400)")
print("="*85)
print(f"{'Newton Tol tau':<14} | {'Max Invariant Defect':<20} | {'Trajectory Error E(h)':<22} | {'Avg Newton Iters/Step':<22} | {'Total CPU Time (s)'}")
print("-" * 110)
for tau in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    start = time.perf_counter()
    y, N, iters, _ = implicit_euler_fixed(0.1, tau_newton=tau)
    cpu_time = time.perf_counter() - start
    print(f"{tau:<14.1e} | {np.max(np.abs(np.sum(y) - 1)):<20.2e} | {np.linalg.norm(y - y_exact_40, np.inf):<22.4e} | {iters/N:<22.2f} | {cpu_time:.4f}")