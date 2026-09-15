#!/usr/bin/env python3
"""
Plots the DL all-solutions closure output: overlays the fractional-residual
distribution across EVERY candidate solution against the distribution
restricted to only the per-event argmax-weight winner (is_best==1).
See plot_sl_closure_bias_allsol.py's docstring for the rationale -- same
idea, dileptonic channel, both neutrinos.

Also plots the reconstructed Higgs mass (from whichever two pool jets are
NOT used as topb/atopb in a given candidate) with all three views
overlapped: the TRUE Higgs mass, the distribution across every candidate,
and the argmax-winner-only distribution.

Input:  closure_dl_allsolutions.root
Output: dl_closure_bias_allsol_vs_best.png
        dl_closure_higgs_mass.png
"""

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_FILE = "closure_dl_allsolutions.root"
FRAC_RANGE = (-3, 3)
FRAC_BINS = 30
HIGGS_RANGE = (0, 300)
HIGGS_BINS = 60


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
    is_best = d["is_best"].astype(bool)
    n_events = len(np.unique(d["eventNumber"]))
    print(f"{len(d['eventNumber'])} total candidate rows across {n_events} events "
          f"({is_best.sum()} argmax-winner rows).")

    nu1_pt_t = np.hypot(d["nu1_px_true"], d["nu1_py_true"])
    nu1_pt_r = np.hypot(d["nu1_px_reco"], d["nu1_py_reco"])
    nu2_pt_t = np.hypot(d["nu2_px_true"], d["nu2_py_true"])
    nu2_pt_r = np.hypot(d["nu2_px_reco"], d["nu2_py_reco"])

    panels = [
        ("nu(top) Δpx/px", frac(d["nu1_px_reco"], d["nu1_px_true"])),
        ("nu(top) Δpy/py", frac(d["nu1_py_reco"], d["nu1_py_true"])),
        ("nu(top) Δpz/pz", frac(d["nu1_pz_reco"], d["nu1_pz_true"])),
        ("nu(top) ΔpT/pT", frac(nu1_pt_r, nu1_pt_t)),
        ("nu(atop) Δpx/px", frac(d["nu2_px_reco"], d["nu2_px_true"])),
        ("nu(atop) Δpy/py", frac(d["nu2_py_reco"], d["nu2_py_true"])),
        ("nu(atop) Δpz/pz", frac(d["nu2_pz_reco"], d["nu2_pz_true"])),
        ("nu(atop) ΔpT/pT", frac(nu2_pt_r, nu2_pt_t)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (title, arr) in zip(axes.flat, panels):
        good_all = np.isfinite(arr)
        good_best = good_all & is_best
        ax.hist(arr[good_all], bins=FRAC_BINS, range=FRAC_RANGE, density=True,
                histtype="stepfilled", alpha=0.35, color="gray",
                label=f"all candidates (n={good_all.sum()})")
        ax.hist(arr[good_best], bins=FRAC_BINS, range=FRAC_RANGE, density=True,
                histtype="step", linewidth=1.8, color="crimson",
                label=f"argmax winner only (n={good_best.sum()})")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(title.split(" ")[-1])
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("dl_closure_bias_allsol_vs_best.png", dpi=150)
    print("Wrote dl_closure_bias_allsol_vs_best.png")

    print("\n=== mean / median: all candidates vs argmax winner ===")
    for title, arr in panels:
        good_all = np.isfinite(arr)
        good_best = good_all & is_best
        print(f"  {title:20s} all: mean={np.mean(arr[good_all]):+7.3f} median={np.median(arr[good_all]):+7.3f}"
              f"   | best: mean={np.mean(arr[good_best]):+7.3f} median={np.median(arr[good_best]):+7.3f}")

    # ── Reconstructed Higgs mass: true vs all candidates vs argmax winner ──
    event_true_higgs = {}
    for ev, m in zip(d["eventNumber"], d["higgs_mass_true"]):
        event_true_higgs.setdefault(ev, m)
    true_higgs_vals = np.array(list(event_true_higgs.values()))

    reco_all = d["higgs_mass_reco"]
    reco_best = d["higgs_mass_reco"][is_best]

    fig3, ax3 = plt.subplots(figsize=(8, 5.5))
    ax3.hist(true_higgs_vals, bins=HIGGS_BINS, range=HIGGS_RANGE, density=True,
             histtype="step", linewidth=2.2, color="forestgreen",
             label=f"true (n={len(true_higgs_vals)} events)")
    ax3.hist(reco_all, bins=HIGGS_BINS, range=HIGGS_RANGE, density=True,
             histtype="stepfilled", alpha=0.35, color="gray",
             label=f"all candidates (n={len(reco_all)})")
    ax3.hist(reco_best, bins=HIGGS_BINS, range=HIGGS_RANGE, density=True,
             histtype="step", linewidth=1.8, color="crimson",
             label=f"argmax winner only (n={len(reco_best)})")
    ax3.set_xlabel("Higgs candidate mass [GeV]")
    ax3.set_title("Dileptonic: reconstructed Higgs mass")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig("dl_closure_higgs_mass.png", dpi=150)
    print("Wrote dl_closure_higgs_mass.png")

    print(f"\n  Higgs mass: true mean={true_higgs_vals.mean():.2f}"
          f"   all-candidates mean={np.nanmean(reco_all):.2f}"
          f"   argmax-winner mean={np.nanmean(reco_best):.2f}")


if __name__ == "__main__":
    main()
