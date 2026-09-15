#!/usr/bin/env python3
"""
Recreate the fractional-residual bias plots (Δpx/px, Δpy/py, Δpz/pz, ΔpT/pT)
from the Pythia truth-closure output, for BOTH the real neutrino and the
"lost" hadronic-W quark, side by side -- this is the actual comparison the
whole closure test was built for: does the same PDF-weight argmax selection
bias we saw in the lost-jet stand-in also show up in the real neutrino?

Input:  closure_sl_truth_vs_reco.root  (written by solve_sl_pythia_closure.py)
Output: sl_closure_bias_1d.png   -- 2 rows (nu, lost) x 4 cols (dpx/dpy/dpz/dpt, all fractional)
        sl_closure_bias_2d.png   -- Δpy/py vs Δpx/px, nu vs lost, side by side
                                     (this is the same correlation plot whose
                                     4-quadrant symmetry was left unresolved
                                     earlier -- worth checking if it reproduces
                                     here too)
"""

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_FILE = "closure_sl_truth_vs_reco.root"

FRAC_RANGE = (-3, 3)
FRAC_BINS = 60


def frac(reco, true):
    """Fractional residual, NaN where true is ~0 (avoids exploding denominators)."""
    true = np.asarray(true, dtype=float)
    reco = np.asarray(reco, dtype=float)
    out = np.full_like(true, np.nan)
    ok = np.abs(true) > 1e-6
    out[ok] = (reco[ok] - true[ok]) / true[ok]
    return out


def main():
    f = uproot.open(INPUT_FILE)
    t = f["closure"]
    d = t.arrays(library="np")

    sel = d["has_solution"].astype(bool)
    print(f"{sel.sum()}/{len(sel)} events with a solution (plotting these only).")

    # ── neutrino: true vs reco ──────────────────────────────────────────────
    nu_px_t, nu_py_t, nu_pz_t = d["nu_px_true"][sel], d["nu_py_true"][sel], d["nu_pz_true"][sel]
    nu_px_r, nu_py_r, nu_pz_r = d["nu_px_reco"][sel], d["nu_py_reco"][sel], d["nu_pz_reco"][sel]
    nu_pt_t = np.hypot(nu_px_t, nu_py_t)
    nu_pt_r = np.hypot(nu_px_r, nu_py_r)

    nu_dpx = frac(nu_px_r, nu_px_t)
    nu_dpy = frac(nu_py_r, nu_py_t)
    nu_dpz = frac(nu_pz_r, nu_pz_t)
    nu_dpt = frac(nu_pt_r, nu_pt_t)

    # ── lost quark: true vs reco ────────────────────────────────────────────
    lost_px_t, lost_py_t, lost_pz_t = d["lost_px_true"][sel], d["lost_py_true"][sel], d["lost_pz_true"][sel]
    lost_px_r, lost_py_r, lost_pz_r = d["lost_px_reco"][sel], d["lost_py_reco"][sel], d["lost_pz_reco"][sel]
    lost_pt_t = np.hypot(lost_px_t, lost_py_t)
    lost_pt_r = np.hypot(lost_px_r, lost_py_r)

    lost_dpx = frac(lost_px_r, lost_px_t)
    lost_dpy = frac(lost_py_r, lost_py_t)
    lost_dpz = frac(lost_pz_r, lost_pz_t)
    lost_dpt = frac(lost_pt_r, lost_pt_t)

    # ── 1D fractional residuals: nu (top row) vs lost (bottom row) ──────────
    rows = [
        ("neutrino", [nu_dpx, nu_dpy, nu_dpz, nu_dpt]),
        ("lost quark", [lost_dpx, lost_dpy, lost_dpz, lost_dpt]),
    ]
    labels = [r"$\Delta p_x/p_x$", r"$\Delta p_y/p_y$", r"$\Delta p_z/p_z$", r"$\Delta p_T/p_T$"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for row_i, (row_name, arrs) in enumerate(rows):
        for col_i, (arr, label) in enumerate(zip(arrs, labels)):
            ax = axes[row_i, col_i]
            good = np.isfinite(arr)
            ax.hist(arr[good], bins=FRAC_BINS, range=FRAC_RANGE, histtype="stepfilled",
                    alpha=0.7, color="C0" if row_i == 0 else "C1")
            mean, median = np.mean(arr[good]), np.median(arr[good])
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_title(f"{row_name}: {label}\nmean={mean:+.2f}  median={median:+.2f}",
                         fontsize=10)
            ax.set_xlabel(label)
    fig.tight_layout()
    fig.savefig("sl_closure_bias_1d.png", dpi=150)
    print("Wrote sl_closure_bias_1d.png")

    # ── 2D: Δpy/py vs Δpx/px, nu vs lost side by side ───────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (row_name, dpx, dpy) in zip(
        axes2, [("neutrino", nu_dpx, nu_dpy), ("lost quark", lost_dpx, lost_dpy)]
    ):
        good = np.isfinite(dpx) & np.isfinite(dpy)
        h = ax.hist2d(dpx[good], dpy[good], bins=50,
                       range=[FRAC_RANGE, FRAC_RANGE], cmap="viridis")
        ax.axhline(0, color="w", lw=0.6, ls="--")
        ax.axvline(0, color="w", lw=0.6, ls="--")
        ax.set_xlabel(r"$\Delta p_x/p_x$")
        ax.set_ylabel(r"$\Delta p_y/p_y$")
        ax.set_title(row_name)
        fig2.colorbar(h[3], ax=ax)
    fig2.tight_layout()
    fig2.savefig("sl_closure_bias_2d.png", dpi=150)
    print("Wrote sl_closure_bias_2d.png")

    # ── numeric summary ──────────────────────────────────────────────────────
    print("\n=== summary (mean / median of fractional residual) ===")
    for row_name, arrs in rows:
        print(f"  {row_name}:")
        for arr, label in zip(arrs, labels):
            good = np.isfinite(arr)
            print(f"    {label:22s} mean={np.mean(arr[good]):+7.3f}"
                  f"   median={np.median(arr[good]):+7.3f}"
                  f"   n={good.sum()}")


if __name__ == "__main__":
    main()
