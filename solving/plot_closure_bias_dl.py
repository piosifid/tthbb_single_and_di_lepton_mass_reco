#!/usr/bin/env python3
"""
Recreate the fractional-residual bias plots (Δpx/px, Δpy/py, Δpz/pz, ΔpT/pT)
from the dileptonic Pythia truth-closure output, for BOTH neutrinos side by
side -- same style as the semileptonic version (plot_sl_closure_bias.py), so
the two channels can be compared directly.

Input:  closure_dl_truth_vs_reco.root  (written by solve_dl_pythia_closure.py)
Output: dl_closure_bias_1d.png   -- 2 rows (nu1=top's nu, nu2=antitop's nu)
                                     x 4 cols (dpx/dpy/dpz/dpt, all fractional)
        dl_closure_bias_2d.png   -- Δpy/py vs Δpx/px, nu1 vs nu2, side by side
"""

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_FILE = "closure_dl_truth_vs_reco.root"
FRAC_RANGE = (-3, 3)
FRAC_BINS = 60


def frac(reco, true):
    true = np.asarray(true, dtype=float)
    reco = np.asarray(reco, dtype=float)
    out = np.full_like(true, np.nan)
    ok = np.abs(true) > 1e-6
    out[ok] = (reco[ok] - true[ok]) / true[ok]
    return out


def main():
    f = uproot.open(INPUT_FILE)
    d = f["closure"].arrays(library="np")
    sel = d["has_solution"].astype(bool)
    print(f"{sel.sum()}/{len(sel)} events with a solution (plotting these only).")

    nu1_px_t, nu1_py_t, nu1_pz_t = d["nu1_px_true"][sel], d["nu1_py_true"][sel], d["nu1_pz_true"][sel]
    nu1_px_r, nu1_py_r, nu1_pz_r = d["nu1_px_reco"][sel], d["nu1_py_reco"][sel], d["nu1_pz_reco"][sel]
    nu1_pt_t = np.hypot(nu1_px_t, nu1_py_t)
    nu1_pt_r = np.hypot(nu1_px_r, nu1_py_r)

    nu2_px_t, nu2_py_t, nu2_pz_t = d["nu2_px_true"][sel], d["nu2_py_true"][sel], d["nu2_pz_true"][sel]
    nu2_px_r, nu2_py_r, nu2_pz_r = d["nu2_px_reco"][sel], d["nu2_py_reco"][sel], d["nu2_pz_reco"][sel]
    nu2_pt_t = np.hypot(nu2_px_t, nu2_py_t)
    nu2_pt_r = np.hypot(nu2_px_r, nu2_py_r)

    nu1_dpx, nu1_dpy, nu1_dpz, nu1_dpt = (
        frac(nu1_px_r, nu1_px_t), frac(nu1_py_r, nu1_py_t),
        frac(nu1_pz_r, nu1_pz_t), frac(nu1_pt_r, nu1_pt_t),
    )
    nu2_dpx, nu2_dpy, nu2_dpz, nu2_dpt = (
        frac(nu2_px_r, nu2_px_t), frac(nu2_py_r, nu2_py_t),
        frac(nu2_pz_r, nu2_pz_t), frac(nu2_pt_r, nu2_pt_t),
    )

    rows = [
        ("neutrino (top, W+)", [nu1_dpx, nu1_dpy, nu1_dpz, nu1_dpt]),
        ("antineutrino (antitop, W-)", [nu2_dpx, nu2_dpy, nu2_dpz, nu2_dpt]),
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
    fig.savefig("dl_closure_bias_1d.png", dpi=150)
    print("Wrote dl_closure_bias_1d.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (row_name, dpx, dpy) in zip(
        axes2, [("neutrino (top)", nu1_dpx, nu1_dpy), ("antineutrino (antitop)", nu2_dpx, nu2_dpy)]
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
    fig2.savefig("dl_closure_bias_2d.png", dpi=150)
    print("Wrote dl_closure_bias_2d.png")

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
