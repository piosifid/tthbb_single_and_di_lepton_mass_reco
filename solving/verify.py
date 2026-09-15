#!/usr/bin/env python3
"""
v2 -- fixes a real bug in verify_mirror_identity_sl.py: that script split
events by |ratio|~1 vs far, but the identity actually predicts on the
SIGNED ratio:

    lost_dpx_frac = -nu_dpx_frac * ratio          [EXACT, ratio = nu_px_true/lost_px_true]

For a collapsed neutrino (nu_dpx_frac ~ -1):

    lost_dpx_frac ~ +ratio     (signed!)

So:
    ratio ~ +1  ->  lost_dpx_frac ~ +1   (this is the ONLY source of the +1 peak)
    ratio ~ -1  ->  lost_dpx_frac ~ -1   (reinforces the existing -1 peak, not a
                                          new feature)

Binning on |ratio|~1 (v1's mistake) merges these two opposite-sign cases and
can wash out the +1 signal if the sample happens to have more collapsed
events with ratio~-1 than ratio~+1 -- which is exactly consistent with a
histogram that shows a sharp -1 peak and only a faint +1 blip.

This version splits into three signed groups and tests each separately.
Works on either closure_sl_truth_vs_reco.root (winner-only) or
closure_sl_allsolutions.root (long format, one row per candidate) -- just
change INPUT_FILE. The identity is per-candidate algebra, so it should hold
in both.
"""

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_FILE = "closure_sl_truth_vs_reco.root"   # or "closure_sl_allsolutions.root"


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

    if "has_solution" in d:
        sel = d["has_solution"].astype(bool)
    else:
        sel = np.ones(len(d["nu_px_true"]), dtype=bool)  # allsolutions file: every row has a solution

    nu_px_t, nu_px_r = d["nu_px_true"][sel], d["nu_px_reco"][sel]
    lost_px_t, lost_px_r = d["lost_px_true"][sel], d["lost_px_reco"][sel]

    # ── 1. Exact identity check (unchanged from v1) ─────────────────────────
    residual = (lost_px_r - lost_px_t) + (nu_px_r - nu_px_t)
    print("=== Exact identity check: Δ(lost_px) + Δ(nu_px) should be ~0 ===")
    print(f"  mean={np.mean(residual):.3e}  std={np.std(residual):.3e}"
          f"  max|residual|={np.max(np.abs(residual)):.3e}\n")

    # ── 2. SIGNED ratio split (the fix) ─────────────────────────────────────
    nu_dpx_frac = frac(nu_px_r, nu_px_t)
    lost_dpx_frac = frac(lost_px_r, lost_px_t)
    ratio = nu_px_t / lost_px_t  # signed, nu_px_true / lost_px_true

    good = np.isfinite(nu_dpx_frac) & np.isfinite(lost_dpx_frac) & np.isfinite(ratio)
    collapsed = good & (nu_dpx_frac > -1.3) & (nu_dpx_frac < -0.7)

    ratio_plus1 = collapsed & (ratio > 0.7) & (ratio < 1.3)
    ratio_minus1 = collapsed & (ratio > -1.3) & (ratio < -0.7)
    ratio_other = collapsed & ~ratio_plus1 & ~ratio_minus1

    print("=== Collapsed events (nu_dpx_frac ~ -1), split by SIGNED ratio ===")
    for name, mask in [("ratio ~ +1  (predict lost_dpx_frac ~ +1)", ratio_plus1),
                        ("ratio ~ -1  (predict lost_dpx_frac ~ -1)", ratio_minus1),
                        ("ratio elsewhere (predict scattered)", ratio_other)]:
        n = int(mask.sum())
        if n == 0:
            print(f"  {name}: n=0")
            continue
        print(f"  {name}: n={n}  mean lost_dpx_frac={np.mean(lost_dpx_frac[mask]):+.3f}"
              f"  median={np.median(lost_dpx_frac[mask]):+.3f}")

    # ── 3. Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.scatter(ratio[~collapsed], lost_dpx_frac[~collapsed], s=4, alpha=0.15,
               color="gray", label="nu not collapsed")
    ax.scatter(ratio[ratio_plus1], lost_dpx_frac[ratio_plus1], s=10, alpha=0.7,
               color="crimson", label="collapsed & ratio~+1")
    ax.scatter(ratio[ratio_minus1], lost_dpx_frac[ratio_minus1], s=10, alpha=0.7,
               color="steelblue", label="collapsed & ratio~-1")
    ax.scatter(ratio[ratio_other], lost_dpx_frac[ratio_other], s=6, alpha=0.4,
               color="darkorange", label="collapsed & ratio elsewhere")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.axhline(-1, color="k", lw=0.8, ls=":")
    ax.axvline(1, color="k", lw=0.8, ls="--")
    ax.axvline(-1, color="k", lw=0.8, ls=":")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("nu_px_true / lost_px_true (signed ratio)")
    ax.set_ylabel("lost_dpx_frac")
    ax.set_title("Fixed: split on SIGNED ratio, not |ratio|")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.hist(lost_dpx_frac[ratio_plus1], bins=50, range=(-3, 3), histtype="step",
              linewidth=2, color="crimson", density=True,
              label=f"ratio~+1 (n={int(ratio_plus1.sum())})")
    ax2.hist(lost_dpx_frac[ratio_minus1], bins=50, range=(-3, 3), histtype="step",
              linewidth=2, color="steelblue", density=True,
              label=f"ratio~-1 (n={int(ratio_minus1.sum())})")
    ax2.hist(lost_dpx_frac[ratio_other], bins=50, range=(-3, 3), histtype="step",
              linewidth=2, color="darkorange", density=True,
              label=f"ratio elsewhere (n={int(ratio_other.sum())})")
    ax2.axvline(1, color="k", lw=0.8, ls="--")
    ax2.axvline(-1, color="k", lw=0.8, ls=":")
    ax2.set_xlabel("lost_dpx_frac")
    ax2.set_title("Now the +1 and -1 peaks should separate cleanly by ratio sign")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("sl_mirror_identity_check_v2.png", dpi=150)
    print("\nWrote sl_mirror_identity_check_v2.png")


if __name__ == "__main__":
    main()