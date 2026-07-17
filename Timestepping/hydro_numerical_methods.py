"""
Numerical Time-Stepping Methods for Hydrology & Geology Simulations
=====================================================================

Almost every process-based hydrology or geology model boils down to
solving one or more Ordinary Differential Equations (ODEs) through time:

        dy/dt = f(t, y)

"y" might be reservoir volume, water-table height, solute concentration,
or the amount of a radioactive isotope left in a mineral. Because we can't
usually solve these equations exactly, we "step" forward in time in small
increments (dt, often called h) using a numerical integration scheme.
The CHOICE of scheme and the SIZE of the time step control how accurate,
stable, and expensive the simulation is.

This script implements and compares three classic explicit schemes:

    1. Forward Euler       - 1st-order accurate, cheapest, least stable
    2. Heun's Method (RK2) - 2nd-order accurate ("improved Euler")
    3. Classic RK4         - 4th-order accurate, the workhorse of
                              hydrologic/geologic ODE solvers

...applied to three physically motivated examples:

    A. Linear reservoir drainage      dV/dt = Qin - k*V
       (a lake, farm pond, or a linear "bucket" model of a watershed)
       Has an exact analytical solution -> lets us measure true error.

    B. Nonlinear unconfined-aquifer recession   dh/dt = -alpha * h^2
       (Boussinesq-type baseflow recession of a water table after
       rain stops -- classic nonlinear groundwater hydrograph problem)
       No simple closed form -> shows RK4's advantage on curvature.

    C. Radioactive decay chain (geochronology, e.g. simplified U->Pb)
       A COUPLED system of two ODEs -> shows how the same machinery
       extends to multiple state variables, as in reactive-transport
       or multi-reservoir hydrologic models.

Run this file directly to print a numeric comparison table and save
comparison plots to ./outputs/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# =====================================================================
# 1. GENERIC ODE STEPPERS
# =====================================================================
# Each "step" function advances the state y by one time step h, given
# the current time t and the derivative function f(t, y).

def euler_step(f, t, y, h):
    """Forward Euler: use the slope at the START of the interval only.
    Local error ~ O(h^2) per step, global error ~ O(h). Cheapest (1
    function evaluation) but drifts fastest, especially for curving
    (nonlinear) processes or long time steps."""
    return y + h * f(t, y)


def rk2_step(f, t, y, h):
    """Heun's Method / RK2 ("predictor-corrector"): estimate the slope
    at the start (k1), use it to predict the endpoint, evaluate the
    slope there too (k2), then average the two slopes. Global error
    ~ O(h^2) -- roughly squares the accuracy of Euler for the same h,
    at the cost of 2 function evaluations per step."""
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + (h / 2.0) * (k1 + k2)


def rk4_step(f, t, y, h):
    """Classic 4th-order Runge-Kutta: samples the slope FOUR times per
    step (start, two midpoint estimates, and the end) and combines them
    with Simpson-like weights. Global error ~ O(h^4): doubling the step
    size only costs a factor of ~16 in accuracy, not 2. This is why RK4
    is the default choice in most rainfall-runoff, groundwater, and
    reaction-kinetics codes -- it lets you take much bigger time steps
    than Euler for the same accuracy."""
    k1 = f(t, y)
    k2 = f(t + h / 2.0, y + h / 2.0 * k1)
    k3 = f(t + h / 2.0, y + h / 2.0 * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


STEPPERS = {"euler": euler_step, "rk2": rk2_step, "rk4": rk4_step}


def integrate(f, y0, t0, t_end, h, method="rk4"):
    """March the ODE forward from t0 to t_end in fixed steps of size h
    using the requested method. Returns arrays of time and state."""
    step_fn = STEPPERS[method]
    n_steps = int(round((t_end - t0) / h))
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1,) + np.shape(y0), dtype=float)
    t[0], y[0] = t0, y0
    for i in range(n_steps):
        y[i + 1] = step_fn(f, t[i], y[i], h)
        t[i + 1] = t[i] + h
    return t, y


# =====================================================================
# EXAMPLE A: Linear reservoir drainage
# =====================================================================
# dV/dt = Qin - k*V
#
# A classic "linear bucket" model used everywhere in hydrology: a lake,
# farm pond, or an entire watershed's storage draining through an
# outlet proportional to how full it is (like Darcy flow through a
# porous dam, or baseflow recession). Qin is inflow (rain/snowmelt/
# streamflow in), k is a linear outlet/recession constant (1/day).
#
# Because it's LINEAR, we know the exact solution, so we can measure
# each method's true numerical error directly.

def reservoir_rhs(Qin, k):
    def f(t, V):
        return Qin - k * V
    return f


def reservoir_analytical(t, V0, Qin, k):
    """Exact solution: V(t) = Veq + (V0 - Veq) * exp(-k t), Veq = Qin/k"""
    Veq = Qin / k
    return Veq + (V0 - Veq) * np.exp(-k * t)


# =====================================================================
# EXAMPLE B: Nonlinear unconfined aquifer (Boussinesq) recession
# =====================================================================
# dh/dt = -alpha * h^2
#
# After rain stops, the water table in an unconfined aquifer feeding a
# stream declines nonlinearly (Boussinesq's approximation to Dupuit
# groundwater flow gives roughly a 1/t recession rather than the
# exponential recession of the linear-reservoir case). This is the
# standard "nonlinear baseflow recession" used to interpret streamflow
# recession curves in hydrogeology. There's no simple closed-form h(t)
# once you add recharge, so this is where Euler starts to visibly lag
# behind RK4, especially with larger time steps.

def aquifer_rhs(alpha):
    def f(t, h):
        return -alpha * h ** 2
    return f


def aquifer_analytical(t, h0, alpha):
    """Exact solution when recharge = 0: h(t) = h0 / (1 + alpha*h0*t)"""
    return h0 / (1.0 + alpha * h0 * t)


# =====================================================================
# EXAMPLE C: Coupled radioactive decay chain (geochronology)
# =====================================================================
# dN_parent/dt = -lambda1 * N_parent
# dN_daughter/dt = lambda1 * N_parent - lambda2 * N_daughter
#
# The basis of radiometric dating (e.g. a simplified two-step decay
# chain analogous to U -> intermediate -> Pb). This shows the same RK4
# machinery working on a VECTOR state, which is exactly how coupled
# reactive-transport or multi-box hydrologic models (soil moisture +
# groundwater + streamflow, all interacting) are solved in practice.

def decay_chain_rhs(lam1, lam2):
    def f(t, N):
        Np, Nd = N
        dNp = -lam1 * Np
        dNd = lam1 * Np - lam2 * Nd
        return np.array([dNp, dNd])
    return f


# =====================================================================
# DEMONSTRATION / COMPARISON
# =====================================================================

def run_example_A():
    Qin, k, V0 = 2.0, 0.15, 30.0     # m^3/day inflow, 1/day, m^3 initial
    t0, t_end = 0.0, 40.0
    h_values = [4.0, 2.0, 0.5]        # days -- coarse to fine

    fig, axes = plt.subplots(1, len(h_values), figsize=(15, 4.2), sharey=True)
    t_fine = np.linspace(t0, t_end, 400)
    V_exact = reservoir_analytical(t_fine, V0, Qin, k)

    print("\n=== EXAMPLE A: Linear reservoir drainage ===")
    print(f"{'h (days)':>10} {'method':>8} {'final V':>12} {'abs error':>12}")
    for ax, h in zip(axes, h_values):
        ax.plot(t_fine, V_exact, "k-", lw=2, label="exact")
        for method, style in [("euler", "o--"), ("rk2", "s--"), ("rk4", "^-")]:
            t, V = integrate(reservoir_rhs(Qin, k), V0, t0, t_end, h, method)
            err = abs(V[-1] - reservoir_analytical(t_end, V0, Qin, k))
            print(f"{h:10.2f} {method:>8} {V[-1]:12.4f} {err:12.5f}")
            ax.plot(t, V, style, ms=3, lw=1, label=method.upper())
        ax.set_title(f"time step h = {h} days")
        ax.set_xlabel("time (days)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("reservoir volume V (m$^3$)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Example A -- Linear reservoir drainage: dV/dt = Qin - kV")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "A_linear_reservoir.png"), dpi=140)
    plt.close(fig)


def run_example_B():
    alpha, h0 = 0.01, 5.0             # 1/(m*day), initial water table height (m)
    t0, t_end = 0.0, 60.0
    h_values = [6.0, 3.0, 1.0]         # days

    fig, axes = plt.subplots(1, len(h_values), figsize=(15, 4.2), sharey=True)
    t_fine = np.linspace(t0, t_end, 400)
    h_exact = aquifer_analytical(t_fine, h0, alpha)

    print("\n=== EXAMPLE B: Nonlinear aquifer (Boussinesq) recession ===")
    print(f"{'dt (days)':>10} {'method':>8} {'final h':>12} {'abs error':>12}")
    for ax, dt in zip(axes, h_values):
        ax.plot(t_fine, h_exact, "k-", lw=2, label="exact")
        for method, style in [("euler", "o--"), ("rk2", "s--"), ("rk4", "^-")]:
            t, hstate = integrate(aquifer_rhs(alpha), h0, t0, t_end, dt, method)
            err = abs(hstate[-1] - aquifer_analytical(t_end, h0, alpha))
            print(f"{dt:10.2f} {method:>8} {hstate[-1]:12.4f} {err:12.5f}")
            ax.plot(t, hstate, style, ms=3, lw=1, label=method.upper())
        ax.set_title(f"time step = {dt} days")
        ax.set_xlabel("time (days)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("water-table height h (m)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Example B -- Nonlinear aquifer recession: dh/dt = -alpha h^2")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "B_nonlinear_aquifer.png"), dpi=140)
    plt.close(fig)


def run_example_C():
    lam1, lam2 = 0.20, 0.05           # decay constants (1/Myr, illustrative)
    N0 = np.array([100.0, 0.0])       # start with pure parent isotope
    t0, t_end, h = 0.0, 60.0, 0.5

    fig, ax = plt.subplots(figsize=(7, 4.5))
    print("\n=== EXAMPLE C: Coupled radioactive decay chain ===")
    for method, style in [("euler", "--"), ("rk4", "-")]:
        t, N = integrate(decay_chain_rhs(lam1, lam2), N0, t0, t_end, h, method)
        ax.plot(t, N[:, 0], style, label=f"parent ({method.upper()})")
        ax.plot(t, N[:, 1], style, label=f"daughter ({method.upper()})")
        print(f"{method:>6}: final parent={N[-1,0]:.3f}  final daughter={N[-1,1]:.3f}")
    ax.set_xlabel("time (Myr, illustrative units)")
    ax.set_ylabel("amount remaining")
    ax.set_title("Example C -- Coupled decay chain (geochronology)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "C_decay_chain.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run_example_A()
    run_example_B()
    run_example_C()
    print(f"\nPlots saved to: {OUTDIR}")
