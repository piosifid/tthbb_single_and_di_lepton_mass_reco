#!/usr/bin/env python3
"""
Checks the "small-denominator" hypothesis for the Δpx/px, Δpy/py positive
tail: does a large |fractional residual| happen specifically when the TRUE
px/py is small (so a modest absolute error blows up when divided by it), for
BOTH the neutrino and the lost quark, independent of correct_pairing (which
we already ruled out as the explanation)?

Input:  closure_sl_truth_vs_reco.root
Output: sl_closure_denom_check.png
        -- scatter of |Δp/p| (y, log scale) vs |p_true| (x), nu and lost
           quark, px and py -- if the tail is a small-denominator artifact,
           the large-|Δp/p| points should cluster at small |p_true| and
           thin out as |p_true| grows.
"""

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_FILE = "closure_sl_truth_vs_reco.root"


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

    panels = [
        ("neutrino px", d["nu_px_true"], frac(d["nu_px_reco"], d["nu_px_true"])),
        ("neutrino py", d["nu_py_true"], frac(d["nu_py_reco"], d["nu_py_true"])),
        ("lost quark px", d["lost_px_true"], frac(d["lost_px_reco"], d["lost_px_true"])),
        ("lost quark py", d["lost_py_true"], frac(d["lost_py_reco"], d["lost_py_true"])),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (title, true_val, dfrac) in zip(axes, panels):
        true_val = np.asarray(true_val, dtype=float)
        good = sel & np.isfinite(dfrac)
        x = np.abs(true_val[good])
        y = np.abs(dfrac[good])
        ax.scatter(x, y, s=6, alpha=0.35)
        ax.set_yscale("log")
        ax.set_xlabel("|p_true|  [GeV]")
        ax.set_ylabel("|Δp/p|  (log)")
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0, color="r", lw=0.8, ls="--", label="|Δp/p|=1")
        ax.legend(fontsize=8)

        # quantify: mean |p_true| for the "tail" (|Δp/p|>1) vs the "core" (<1)
        tail = y > 1.0
        core = ~tail
        if tail.sum() and core.sum():
            print(f"  {title:16s}  tail(|Δp/p|>1): n={tail.sum():4d}  "
                  f"mean|p_true|={x[tail].mean():7.2f}   "
                  f"core(|Δp/p|<=1): n={core.sum():4d}  mean|p_true|={x[core].mean():7.2f}")

    fig.tight_layout()
    fig.savefig("sl_closure_denom_check.png", dpi=150)
    print("Wrote sl_closure_denom_check.png")


if __name__ == "__main__":
    main()