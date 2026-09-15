#!/usr/bin/env python3
"""
Plots the real-root-count-per-hypothesis distribution -- now read straight
out of the "hyp_roots" tree that solve_sl_pythia_closure_allsol.py /
solve_dl_pythia_closure_allsol.py write alongside the usual "closure" tree
(after applying HYP_N_ROOTS_instrumentation.txt). No resimulation needed:
just rerun the solve script once (it now records this for free in the same
pass) and this script only reads the output.

Usage:
    python3 n_roots_per_hypothesis.py sl
    python3 n_roots_per_hypothesis.py dl
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
    if "hyp_roots" not in f:
        print(f"ERROR: '{input_file}' has no 'hyp_roots' tree yet -- rerun "
              f"solve_{channel}_pythia_closure_allsol.py after applying the "
              f"HYP_N_ROOTS instrumentation patch.")
        sys.exit(1)

    n_roots = f["hyp_roots"]["n_roots"].array(library="np")

    vals, counts = np.unique(n_roots, return_counts=True)
    total = len(n_roots)
    print(f"\n=== real pnux roots per (jet-pairing, mt, mW) hypothesis point, {channel.upper()} ===")
    for v, c in zip(vals, counts):
        print(f"  {int(v)} roots: {c} hypothesis points ({100*c/total:.1f}%)")
    print(f"  mean={n_roots.mean():.3f}  RMS={n_roots.std():.4f}  entries={total}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar(vals, counts / total, color="white", edgecolor="k", linewidth=1.2)
    ax.set_xlabel("N(solutions)")
    ax.set_ylabel("Fraction(Events)")
    ax.set_title(f"{channel.upper()}: real roots per (jet-pairing, mt, mW) hypothesis")
    stats_txt = (f"Entries  {total}\n"
                 f"Mean     {n_roots.mean():.3f}\n"
                 f"RMS      {n_roots.std():.4f}")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="square", facecolor="white", edgecolor="k"))

    outname = f"{channel}_n_roots_per_hypothesis.png"
    fig.tight_layout()
    fig.savefig(outname, dpi=150)
    print(f"\nWrote {outname}")


if __name__ == "__main__":
    main()