#!/usr/bin/env python3
"""
Plots the SL all-solutions closure output: overlays the fractional-residual
distribution across EVERY candidate solution (all jet-pairings x mass-grid
points x quartic roots) against the distribution restricted to only the
per-event argmax-weight winner (is_best==1) -- the latter should reproduce
plot_sl_closure_bias.py's neutrino/lost-quark panels as a cross-check.

Also plots the reconstructed Higgs mass (from whichever two pool jets are
NOT used as blep/bhad in a given candidate) with all three views overlapped:
the TRUE Higgs mass (from the real higgsb1+higgsb2), the distribution across
every candidate, and the argmax-winner-only distribution -- shows directly
how much the mass resolution/pairing degrades once you're not looking only
at the winning combination.

Input:  closure_sl_allsolutions.root
Output: sl_closure_bias_allsol_vs_best.png
        sl_closure_higgs_mass.png
"""

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_FILE = "closure_sl_allsolutions.root"
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

    nu_pt_t = np.hypot(d["nu_px_true"], d["nu_py_true"])
    nu_pt_r = np.hypot(d["nu_px_reco"], d["nu_py_reco"])
    lost_pt_t = np.hypot(d["lost_px_true"], d["lost_py_true"])
    lost_pt_r = np.hypot(d["lost_px_reco"], d["lost_py_reco"])

    panels = [
        ("neutrino Δpx/px", frac(d["nu_px_reco"], d["nu_px_true"])),
        ("neutrino Δpy/py", frac(d["nu_py_reco"], d["nu_py_true"])),
        ("neutrino Δpz/pz", frac(d["nu_pz_reco"], d["nu_pz_true"])),
        ("neutrino ΔpT/pT", frac(nu_pt_r, nu_pt_t)),
        ("lost quark Δpx/px", frac(d["lost_px_reco"], d["lost_px_true"])),
        ("lost quark Δpy/py", frac(d["lost_py_reco"], d["lost_py_true"])),
        ("lost quark Δpz/pz", frac(d["lost_pz_reco"], d["lost_pz_true"])),
        ("lost quark ΔpT/pT", frac(lost_pt_r, lost_pt_t)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (title, arr) in zip(axes.flat, panels):
        good_all = np.isfinite(arr)
        good_best = good_all & is_best
        # normalize each to a density so the "all candidates" histogram
        # (which has ~n_solutions x more entries) is shape-comparable to
        # the "winner only" one, not just dwarfing it in raw counts.
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
    fig.savefig("sl_closure_bias_allsol_vs_best.png", dpi=150)
    print("Wrote sl_closure_bias_allsol_vs_best.png")

    print("\n=== mean / median: all candidates vs argmax winner ===")
    for title, arr in panels:
        good_all = np.isfinite(arr)
        good_best = good_all & is_best
        print(f"  {title:20s} all: mean={np.mean(arr[good_all]):+7.3f} median={np.median(arr[good_all]):+7.3f}"
              f"   | best: mean={np.mean(arr[good_best]):+7.3f} median={np.median(arr[good_best]):+7.3f}")

    # ── Reconstructed Higgs mass: true vs all candidates vs argmax winner ──
    # higgs_mass_true is duplicated across every row of the same event (it
    # doesn't depend on the candidate), so dedupe to one value per event
    # before histogramming it -- otherwise it'd get weighted by
    # n_solutions per event instead of by event count.
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
    ax3.set_title("Semileptonic: reconstructed Higgs mass")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig("sl_closure_higgs_mass.png", dpi=150)
    print("Wrote sl_closure_higgs_mass.png")

    print(f"\n  Higgs mass: true mean={true_higgs_vals.mean():.2f}"
          f"   all-candidates mean={np.nanmean(reco_all):.2f}"
          f"   argmax-winner mean={np.nanmean(reco_best):.2f}")


if __name__ == "__main__":
    main()
