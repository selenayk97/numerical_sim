"""
Stochastic / Monte Carlo Methods for Hydrology & Geology
=====================================================================

Deterministic models (like the RK4 solver from earlier) give you ONE
answer for ONE set of inputs. But real rainfall, aquifer properties,
and model parameters are all uncertain. Stochastic methods let us
propagate that uncertainty through a model to answer questions like:

    "What's the 1%-annual-chance (100-year) peak flow, given that we
     aren't sure exactly how much it will rain or how the watershed
     will respond?"
    "Given streamflow observations, what recession-constant values
     are actually consistent with the data -- and how uncertain
     are we?"
    "Where will a contaminant plume end up, given that groundwater
     flow paths are never perfectly smooth?"

This script covers three classic techniques:

    1. Monte Carlo simulation
       -- randomize rainfall + model parameters thousands of times,
          run a simple rainfall-runoff model each time, and build up
          a distribution of peak flows (basic flood-risk propagation).

    2. Markov Chain Monte Carlo (MCMC) via Metropolis-Hastings
       -- Bayesian calibration: given observed streamflow, find the
          probability distribution of a model parameter (not just a
          single "best" value), which is how modern hydrologic model
          uncertainty analysis (e.g. DREAM, GLUE-style workflows) works.

    3. Random-walk particle tracking
       -- simulate solute/contaminant transport in groundwater as many
          independent random-walking particles, an alternative to
          solving the advection-dispersion PDE directly. This is the
          method behind particle-tracking codes like MODPATH/RWPT.

Run this file directly to see printed statistics and saved plots in
./outputs/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)
OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTDIR, exist_ok=True)


# =====================================================================
# 1. MONTE CARLO SIMULATION -- FLOOD RISK PROPAGATION
# =====================================================================
# A very simple "unit hydrograph"-style rainfall-runoff model:
#     peak flow Qp = C * i * A
# where:
#     i = rainfall intensity (mm/hr)      -- uncertain, drawn from a PDF
#     A = watershed area (km^2)           -- fixed, known
#     C = runoff coefficient (0-1)        -- uncertain: depends on
#         antecedent soil moisture, land use, etc.
#
# Instead of picking single "design" values for i and C, we draw
# thousands of random combinations from their plausible distributions
# and see what distribution of peak flows comes out. This is the
# essence of Monte Carlo flood-risk analysis.

def monte_carlo_flood_risk(n_sims=20000, area_km2=15.0):
    # Rainfall intensity: right-skewed -> use a Gamma distribution
    # (mean ~40 mm/hr, similar shape to Example-A style storms)
    intensity = RNG.gamma(shape=3.0, scale=13.0, size=n_sims)  # mm/hr

    # Runoff coefficient: bounded 0-1, use a Beta distribution centered
    # around 0.4 (a mixed urban/rural watershed)
    runoff_coeff = RNG.beta(a=4, b=6, size=n_sims)

    # Simple peak-flow formula (rational method), converting units so
    # Qp comes out in m^3/s
    #   Qp [m3/s] = C * i [mm/hr] * A [km2] / 3.6
    Qp = runoff_coeff * intensity * area_km2 / 3.6

    print("\n=== 1. Monte Carlo flood-risk propagation ===")
    print(f"  simulations run           : {n_sims}")
    print(f"  mean peak flow            : {Qp.mean():.1f} m3/s")
    print(f"  median peak flow          : {np.median(Qp):.1f} m3/s")
    print(f"  std dev of peak flow      : {Qp.std():.1f} m3/s")

    percentiles = [50, 90, 95, 99, 99.9]
    print(f"\n  {'Percentile':>12} {'Peak flow (m3/s)':>18} {'~Return period':>16}")
    for p in percentiles:
        val = np.percentile(Qp, p)
        # crude mapping: annual-max exceedance percentile -> return period
        T = 1.0 / (1.0 - p / 100.0) if p < 100 else np.inf
        print(f"  {p:>12} {val:>18.1f} {T:>13.0f} yr")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.hist(Qp, bins=60, color="steelblue", alpha=0.7, density=True)
    ax1.axvline(np.median(Qp), color="k", ls="--", label="median")
    ax1.axvline(np.percentile(Qp, 99), color="red", ls="--",
                label="99th percentile\n(~100-yr proxy)")
    ax1.set_xlabel("simulated peak flow (m$^3$/s)")
    ax1.set_ylabel("probability density")
    ax1.set_title(f"Monte Carlo flood peaks (n={n_sims})")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    sorted_Q = np.sort(Qp)[::-1]
    exceed_prob = np.arange(1, n_sims + 1) / (n_sims + 1)
    ax2.plot(1 / exceed_prob, sorted_Q, color="darkorange")
    ax2.set_xscale("log")
    ax2.set_xlabel("return period (years, log scale)")
    ax2.set_ylabel("peak flow (m$^3$/s)")
    ax2.set_title("Empirical flood-frequency curve\nfrom Monte Carlo output")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "1_monte_carlo_flood_risk.png"), dpi=140)
    plt.close(fig)
    return Qp


# =====================================================================
# 2. MCMC (METROPOLIS-HASTINGS) -- BAYESIAN MODEL CALIBRATION
# =====================================================================
# We observe noisy streamflow recession data from a linear reservoir
# (dV/dt = -k*V, same model family as before) and want the POSTERIOR
# distribution of the recession constant k -- not just a single
# best-fit value, but the full range of k values consistent with the
# noisy data, weighted by how well each explains the observations.
#
# Metropolis-Hastings algorithm:
#   1. Start at some k
#   2. Propose a new k' = k + small random jump
#   3. Compute how much more (or less) likely the data are under k'
#      versus k (the likelihood ratio)
#   4. Accept the jump with a probability equal to that ratio (always
#      accept if it's better; sometimes accept if it's worse -- this
#      is what lets the chain explore the full posterior, not just
#      hill-climb to a single optimum)
#   5. Repeat thousands of times; the resulting chain of accepted k
#      values approximates the posterior distribution of k.

def simulate_observed_recession(true_k=0.08, V0=100.0, t_end=40,
                                 noise_sd=2.0):
    t_obs = np.arange(0, t_end + 1, 2)  # observe every 2 days
    V_true = V0 * np.exp(-true_k * t_obs)
    V_obs = V_true + RNG.normal(0, noise_sd, size=t_obs.shape)
    return t_obs, V_obs


def log_likelihood(k, t_obs, V_obs, V0, noise_sd):
    V_model = V0 * np.exp(-k * t_obs)
    resid = V_obs - V_model
    return -0.5 * np.sum((resid / noise_sd) ** 2)


def metropolis_hastings(t_obs, V_obs, V0, noise_sd,
                         n_iter=15000, proposal_sd=0.006, k_init=0.15):
    k_current = k_init
    ll_current = log_likelihood(k_current, t_obs, V_obs, V0, noise_sd)
    chain = np.zeros(n_iter)
    n_accepted = 0

    for i in range(n_iter):
        k_proposed = k_current + RNG.normal(0, proposal_sd)
        if k_proposed <= 0:
            chain[i] = k_current
            continue
        ll_proposed = log_likelihood(k_proposed, t_obs, V_obs, V0, noise_sd)
        # Accept if better; accept probabilistically if worse
        if np.log(RNG.random()) < (ll_proposed - ll_current):
            k_current, ll_current = k_proposed, ll_proposed
            n_accepted += 1
        chain[i] = k_current

    print("\n=== 2. MCMC (Metropolis-Hastings) calibration of recession constant k ===")
    print(f"  iterations           : {n_iter}")
    print(f"  acceptance rate      : {n_accepted/n_iter:.1%}  (aim for ~20-40%)")
    return chain


def run_mcmc_example():
    V0, true_k, noise_sd = 100.0, 0.08, 2.0
    t_obs, V_obs = simulate_observed_recession(true_k, V0, noise_sd=noise_sd)
    chain = metropolis_hastings(t_obs, V_obs, V0, noise_sd)

    burn_in = 3000
    posterior = chain[burn_in:]
    print(f"  true k               : {true_k:.4f}")
    print(f"  posterior mean k     : {posterior.mean():.4f}")
    print(f"  posterior std k      : {posterior.std():.4f}")
    print(f"  95% credible interval: [{np.percentile(posterior,2.5):.4f}, "
          f"{np.percentile(posterior,97.5):.4f}]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(chain, lw=0.5, color="teal")
    ax1.axvline(burn_in, color="red", ls="--", label="end of burn-in")
    ax1.set_xlabel("MCMC iteration")
    ax1.set_ylabel("k (recession constant)")
    ax1.set_title("MCMC trace plot")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.hist(posterior, bins=50, density=True, color="teal", alpha=0.6)
    ax2.axvline(true_k, color="red", ls="--", label="true k")
    ax2.axvline(posterior.mean(), color="black", ls="--", label="posterior mean")
    ax2.set_xlabel("k (recession constant)")
    ax2.set_ylabel("posterior density")
    ax2.set_title("Posterior distribution of k")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "2_mcmc_calibration.png"), dpi=140)
    plt.close(fig)


# =====================================================================
# 3. RANDOM-WALK PARTICLE TRACKING -- SOLUTE TRANSPORT IN GROUNDWATER
# =====================================================================
# The advection-dispersion equation describes how a contaminant plume
# moves and spreads in groundwater:
#     dC/dt = -v * dC/dx + D * d2C/dx2   (1D form)
# Rather than solve this PDE on a grid, we can represent the plume as
# many independent particles. Each particle:
#   - moves with the mean groundwater velocity (advection)
#   - gets an extra random "kick" each step (dispersion), drawn from
#     a normal distribution with variance = 2*D*dt
# The particle CLOUD's spatial distribution then approximates the
# concentration field -- this is exactly how codes like MODPATH-RW
# or RWPT engines work, and it's often cheaper and more stable than
# grid-based PDE solvers for advection-dominated transport.

def particle_tracking_transport(n_particles=3000, velocity=0.5,
                                 dispersion=0.05, dt=1.0, n_steps=80):
    """velocity in m/day, dispersion coefficient D in m^2/day."""
    x = np.zeros(n_particles)  # all particles start at the source (x=0)
    positions_over_time = [x.copy()]

    for _ in range(n_steps):
        advective_step = velocity * dt
        dispersive_kick = RNG.normal(0, np.sqrt(2 * dispersion * dt),
                                      size=n_particles)
        x = x + advective_step + dispersive_kick
        positions_over_time.append(x.copy())

    print("\n=== 3. Random-walk particle tracking (solute transport) ===")
    print(f"  particles             : {n_particles}")
    print(f"  mean velocity         : {velocity} m/day")
    print(f"  dispersion coefficient: {dispersion} m^2/day")
    print(f"  simulated time        : {n_steps*dt:.0f} days")
    print(f"  final plume centroid  : {x.mean():.2f} m  "
          f"(expected ~{velocity*n_steps*dt:.2f} m from pure advection)")
    print(f"  final plume std dev   : {x.std():.2f} m  "
          f"(expected ~{np.sqrt(2*dispersion*n_steps*dt):.2f} m from theory)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    snapshot_steps = [n_steps // 3, 2 * n_steps // 3, n_steps]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(snapshot_steps)))
    for s, c in zip(snapshot_steps, colors):
        ax1.hist(positions_over_time[s], bins=30, density=True, alpha=0.5,
                 color=c, label=f"t = {s*dt:.0f} days")
    ax1.axvline(0, color="red", ls="--", lw=1.5, label="source (t=0)")
    ax1.set_xlabel("distance downgradient (m)")
    ax1.set_ylabel("particle density (proxy for concentration)")
    ax1.set_title("Plume spreading over time")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # show a handful of individual particle trajectories
    traj = np.array(positions_over_time)  # shape (n_steps+1, n_particles)
    t_axis = np.arange(n_steps + 1) * dt
    for p in range(15):
        ax2.plot(t_axis, traj[:, p], lw=0.8, alpha=0.6)
    ax2.plot(t_axis, traj.mean(axis=1), "k-", lw=2.5, label="plume centroid")
    ax2.set_xlabel("time (days)")
    ax2.set_ylabel("distance downgradient (m)")
    ax2.set_title("Individual particle paths vs. plume centroid")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "3_particle_tracking_transport.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    monte_carlo_flood_risk()
    run_mcmc_example()
    particle_tracking_transport()
    print(f"\nPlots saved to: {OUTDIR}")
