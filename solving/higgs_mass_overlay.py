#!/usr/bin/env python3
"""
Just the Higgs mass plot: true vs reco (max-weight winner) vs reco (all
candidates), overlaid -- pulled out of summarize_solutions_higgs.py on its
own, no n_solutions panel, no correct-pairing stats.

Input: closure_sl_allsolutions.root / closure_dl_allsolutions.root

Usage:
    python3 higgs_mass_overlay.py sl
    python3 higgs_mass_overlay.py dl
"""
import sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    "sl": "closure_sl_allsolutions.root",
    "dl": "closure_dl_allsolutions.root",
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIG:
        print(f"Usage: {sys.argv[0]} sl|dl")
        sys.exit(1)

    channel = sys.argv[1]
    input_file = CONFIG[channel]

    f = uproot.open(input_file)
    d = f["closure"].arrays(library="np")

    is_best = d["is_best"].astype(bool)
    n_events = int(is_best.sum())

    higgs_true_per_event = d["higgs_mass_true"][is_best]
    higgs_reco_winner = d["higgs_mass_reco"][is_best]
    higgs_reco_all = d["higgs_mass_reco"]

    def stats(name, arr):
        print(f"  {name:28s} n={len(arr):7d}  mean={np.mean(arr):7.2f}  "
              f"median={np.median(arr):7.2f}  std={np.std(arr):6.2f}")

    print(f"[{channel}] Higgs mass [GeV]:")
    stats("true (per event)", higgs_true_per_event)
    stats("reco, max-weight winner", higgs_reco_winner)
    stats("reco, all candidates", higgs_reco_all)

    lo = min(higgs_true_per_event.min(), higgs_reco_winner.min(), higgs_reco_all.min())
    hi = max(higgs_true_per_event.max(), higgs_reco_winner.max(), higgs_reco_all.max())
    pad = 0.05 * (hi - lo)
    rng = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(higgs_reco_all, bins=60, range=rng, histtype="stepfilled", alpha=0.35,
            color="gray", density=True, label=f"reco, all candidates (n={len(higgs_reco_all)})")
    ax.hist(higgs_reco_winner, bins=60, range=rng, histtype="step", linewidth=2,
            color="crimson", density=True, label=f"reco, winner (n={n_events})")
    ax.hist(higgs_true_per_event, bins=60, range=rng, histtype="step", linewidth=2,
            color="k", density=True, label=f"true (n={n_events})")
    ax.set_xlabel("Higgs mass [GeV]")
    ax.set_ylabel("density")
    ax.set_title(f"{channel.upper()}: Higgs mass, true vs reco")
    ax.legend(fontsize=9)

    outname = f"{channel}_higgs_mass_overlay.png"
    fig.tight_layout()
    fig.savefig(outname, dpi=150)
    print(f"\nWrote {outname}")


if __name__ == "__main__":
    main()
