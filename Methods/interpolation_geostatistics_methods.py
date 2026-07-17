"""
Interpolation & Geostatistics Methods for Hydrology & Geology
=====================================================================

You almost never have data everywhere you need it: rain gauges,
monitoring wells, and boreholes are scattered points, but you need a
continuous map of rainfall, water-table elevation, or an aquifer
property to run a model or draw a contour map. Spatial interpolation
methods fill in the gaps between known points.

This script covers three methods, from simplest to most rigorous:

    1. Inverse Distance Weighting (IDW)
       -- the simplest spatial interpolator: nearby points get more
          influence than far points, using a simple 1/distance^p
          weighting. Fast, intuitive, no statistical assumptions --
          but it has no concept of spatial correlation structure or
          uncertainty.

    2. Semivariogram analysis
       -- before kriging, you need to characterize HOW SIMILAR nearby
          points tend to be as a function of separation distance. The
          semivariogram is the geostatistical tool for this, and it's
          the key diagnostic every hydrogeologist looks at before
          interpolating well/gauge data.

    3. Ordinary Kriging
       -- the geostatistical "best linear unbiased predictor": uses
          the fitted semivariogram model to produce not just an
          interpolated surface, but also a MAP OF ESTIMATION
          UNCERTAINTY (kriging variance) -- telling you where you can
          trust the interpolation and where you can't (e.g. far from
          any well). This is the standard method for mapping aquifer
          properties, contaminant concentrations, and rainfall fields
          in real hydrogeologic practice.

Run this file directly to see printed diagnostics and saved plots
(including side-by-side IDW vs Kriging maps) in ./outputs/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

RNG = np.random.default_rng(3)
OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTDIR, exist_ok=True)


# =====================================================================
# 0. SYNTHETIC "TRUE" FIELD + SPARSE MONITORING WELLS
# =====================================================================
# We build a smooth, spatially-correlated "true" water-table elevation
# surface (so we have ground truth to check against), then sample it
# at a handful of scattered well locations -- mimicking a real
# groundwater monitoring network. The rest of the script pretends we
# only know the well values and must reconstruct the surface.

def make_true_field(grid_n=80, domain=100.0):
    xg = np.linspace(0, domain, grid_n)
    yg = np.linspace(0, domain, grid_n)
    Xg, Yg = np.meshgrid(xg, yg)
    # A smooth trend (regional groundwater gradient) plus two
    # "mound/depression" features (e.g. recharge mound, pumping cone)
    Z = (100 - 0.08 * Xg - 0.04 * Yg
         + 15 * np.exp(-((Xg - 25) ** 2 + (Yg - 70) ** 2) / (2 * 15 ** 2))
         - 12 * np.exp(-((Xg - 70) ** 2 + (Yg - 30) ** 2) / (2 * 12 ** 2)))
    return xg, yg, Xg, Yg, Z


def sample_wells(xg, yg, Xg, Yg, Z, n_wells=25, noise_sd=0.4):
    well_x = RNG.uniform(0, xg.max(), n_wells)
    well_y = RNG.uniform(0, yg.max(), n_wells)
    # interpolate the "true" field at well locations, then add
    # measurement noise -- exactly what a real well network gives you
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((yg, xg), Z)
    well_z = interp(np.column_stack([well_y, well_x])) + RNG.normal(0, noise_sd, n_wells)
    return well_x, well_y, well_z


# =====================================================================
# 1. INVERSE DISTANCE WEIGHTING (IDW)
# =====================================================================
#           sum_i ( z_i / d_i^p )
#   z(x0) = ---------------------
#           sum_i ( 1 / d_i^p )
#
# where d_i is the distance from the prediction point x0 to observation
# i, and p is a power (commonly 2) controlling how quickly influence
# fades with distance.

def idw_interpolate(well_x, well_y, well_z, Xg, Yg, power=2.0):
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    well_pts = np.column_stack([well_x, well_y])
    d = cdist(grid_pts, well_pts)          # distance from every grid cell to every well
    d[d == 0] = 1e-10                       # avoid divide-by-zero at well locations
    weights = 1.0 / d ** power
    z_interp = (weights * well_z).sum(axis=1) / weights.sum(axis=1)
    return z_interp.reshape(Xg.shape)


# =====================================================================
# 2. SEMIVARIOGRAM ANALYSIS
# =====================================================================
# The experimental semivariogram measures how DISSIMILAR pairs of
# points are, on average, as a function of their separation distance h:
#
#       gamma(h) = 0.5 * average[ (z_i - z_j)^2 ]  for all pairs
#                  with distance ~ h
#
# Typically gamma(h) rises from near 0 (nearby points are similar) up
# to a plateau called the SILL (beyond which points are essentially
# uncorrelated) at a distance called the RANGE. We fit a simple
# spherical model, the classic default in hydrogeology:
#
#       gamma(h) = nugget + sill*(1.5*h/range - 0.5*(h/range)^3)  for h < range
#       gamma(h) = nugget + sill                                   for h >= range

def experimental_semivariogram(well_x, well_y, well_z, n_bins=10):
    pts = np.column_stack([well_x, well_y])
    d = cdist(pts, pts)
    z_diff_sq = (well_z[:, None] - well_z[None, :]) ** 2

    iu = np.triu_indices(len(well_z), k=1)  # unique pairs only
    dist_pairs = d[iu]
    gamma_pairs = 0.5 * z_diff_sq[iu]

    max_dist = dist_pairs.max()
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers, gamma_exp = [], []
    for i in range(n_bins):
        mask = (dist_pairs >= bin_edges[i]) & (dist_pairs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(dist_pairs[mask].mean())
            gamma_exp.append(gamma_pairs[mask].mean())
    return np.array(bin_centers), np.array(gamma_exp)


def spherical_model(h, nugget, sill, rng):
    h = np.asarray(h, dtype=float)
    gamma = np.where(
        h < rng,
        nugget + sill * (1.5 * h / rng - 0.5 * (h / rng) ** 3),
        nugget + sill,
    )
    return gamma


def fit_spherical_variogram(bin_centers, gamma_exp):
    from scipy.optimize import curve_fit
    # initial guesses: nugget ~ small, sill ~ plateau value, range ~ half max distance
    p0 = [0.5, gamma_exp.max(), bin_centers.max() * 0.5]
    bounds = ([0, 0, 1e-3], [gamma_exp.max(), gamma_exp.max() * 2, bin_centers.max()])
    popt, _ = curve_fit(spherical_model, bin_centers, gamma_exp, p0=p0, bounds=bounds)
    return popt  # nugget, sill, range


# =====================================================================
# 3. ORDINARY KRIGING
# =====================================================================
# Ordinary Kriging predicts z at an unsampled location as a WEIGHTED
# AVERAGE of the observations, where the weights are chosen (by
# solving a linear system built from the semivariogram model) to be
# the "best linear unbiased" estimate -- minimizing expected error
# while requiring the weights sum to 1. Crucially it also returns
# the KRIGING VARIANCE at every location: a genuine, model-based
# measure of interpolation uncertainty (small near wells, large in
# data-sparse areas), which IDW simply cannot provide.

def ordinary_kriging(well_x, well_y, well_z, Xg, Yg, variogram_params):
    nugget, sill, rng_ = variogram_params
    n = len(well_z)
    well_pts = np.column_stack([well_x, well_y])

    # Build the kriging matrix: semivariances between all well pairs,
    # bordered with a row/col of 1's to enforce weights summing to 1
    # (the "Lagrange multiplier" trick for ordinary kriging).
    D = cdist(well_pts, well_pts)
    Gamma = spherical_model(D, nugget, sill, rng_)
    K = np.ones((n + 1, n + 1))
    K[:n, :n] = Gamma
    K[-1, -1] = 0
    K_inv = np.linalg.inv(K)

    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    d0 = cdist(grid_pts, well_pts)
    gamma0 = spherical_model(d0, nugget, sill, rng_)
    rhs = np.hstack([gamma0, np.ones((grid_pts.shape[0], 1))])  # add constraint row

    weights_and_mu = rhs @ K_inv  # shape (n_grid_pts, n+1)
    weights = weights_and_mu[:, :n]
    mu = weights_and_mu[:, -1]

    z_pred = weights @ well_z
    # Kriging variance: sigma^2 = weights . gamma0 + mu  (Lagrange term)
    krig_var = (weights * gamma0).sum(axis=1) + mu

    return z_pred.reshape(Xg.shape), krig_var.reshape(Xg.shape)


# =====================================================================
# DEMONSTRATION
# =====================================================================

def run_demo():
    xg, yg, Xg, Yg, Z_true = make_true_field()
    well_x, well_y, well_z = sample_wells(xg, yg, Xg, Yg, Z_true, n_wells=25)

    print("\n=== 0. Setup ===")
    print(f"  monitoring wells: {len(well_z)}")
    print(f"  well head range : {well_z.min():.1f} to {well_z.max():.1f} m")

    # --- 1. IDW ---
    Z_idw = idw_interpolate(well_x, well_y, well_z, Xg, Yg, power=2.0)
    idw_rmse = np.sqrt(np.mean((Z_idw - Z_true) ** 2))
    print("\n=== 1. Inverse Distance Weighting ===")
    print(f"  RMSE vs true field: {idw_rmse:.3f} m")

    # --- 2. Semivariogram ---
    bin_centers, gamma_exp = experimental_semivariogram(well_x, well_y, well_z)
    nugget, sill, rng_ = fit_spherical_variogram(bin_centers, gamma_exp)
    print("\n=== 2. Semivariogram model (spherical) ===")
    print(f"  nugget = {nugget:.3f}")
    print(f"  sill   = {sill:.3f}")
    print(f"  range  = {rng_:.2f} m  (distance beyond which points are ~uncorrelated)")

    # --- 3. Ordinary Kriging ---
    Z_krig, krig_var = ordinary_kriging(well_x, well_y, well_z, Xg, Yg,
                                         (nugget, sill, rng_))
    krig_rmse = np.sqrt(np.mean((Z_krig - Z_true) ** 2))
    print("\n=== 3. Ordinary Kriging ===")
    print(f"  RMSE vs true field: {krig_rmse:.3f} m")
    print(f"  mean kriging std  : {np.sqrt(krig_var).mean():.3f} m")
    print(f"  (IDW gives no uncertainty estimate at all -- kriging does)")

    # --- Plot 1: semivariogram fit ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(bin_centers, gamma_exp, color="black", label="experimental semivariogram")
    h_fine = np.linspace(0, bin_centers.max(), 200)
    ax.plot(h_fine, spherical_model(h_fine, nugget, sill, rng_), "r-", lw=2,
            label="fitted spherical model")
    ax.axhline(nugget + sill, color="gray", ls=":", label="sill")
    ax.axvline(rng_, color="gray", ls="--", label="range")
    ax.set_xlabel("separation distance h (m)")
    ax.set_ylabel("semivariance γ(h)")
    ax.set_title("Semivariogram: spatial correlation of well heads")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "1_semivariogram.png"), dpi=140)
    plt.close(fig)

    # --- Plot 2: true field, IDW, Kriging, kriging variance ---
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    vmin, vmax = Z_true.min(), Z_true.max()

    for ax, field, title in zip(
        axes[:3], [Z_true, Z_idw, Z_krig],
        ["'True' water-table (unknown\nin practice)", "IDW interpolation", "Ordinary Kriging"]
    ):
        im = ax.pcolormesh(Xg, Yg, field, shading="auto", cmap="viridis",
                            vmin=vmin, vmax=vmax)
        ax.scatter(well_x, well_y, c="red", s=18, edgecolor="white", lw=0.5,
                   label="wells")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (m)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="head (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].legend(fontsize=7, loc="lower right")

    im4 = axes[3].pcolormesh(Xg, Yg, np.sqrt(krig_var), shading="auto", cmap="magma")
    axes[3].scatter(well_x, well_y, c="cyan", s=18, edgecolor="black", lw=0.5)
    axes[3].set_title("Kriging std. dev.\n(uncertainty -- IDW has no equivalent)", fontsize=10)
    axes[3].set_xlabel("x (m)")
    fig.colorbar(im4, ax=axes[3], shrink=0.8, label="std dev (m)")

    fig.suptitle(f"IDW (RMSE={idw_rmse:.2f} m) vs Ordinary Kriging (RMSE={krig_rmse:.2f} m)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "2_idw_vs_kriging_maps.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run_demo()
    print(f"\nPlots saved to: {OUTDIR}")
