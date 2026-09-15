# %%
"""
Intersection test: correct_pairing AND near-true mass hypothesis together
-- the one candidate per event that is the actual true kinematic solution.
If the -1 shoulder in dpz survives even here, it's a genuine algebraic
two-fold solution ambiguity (e.g. the quadratic/quartic neutrino pz
solve), not a selection, mass-scan, or pairing artifact.
"""
import uproot
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_FILE = "closure_dl_allsolutions.root"
TREE_NAME  = "tree"   # change if your TTree has a different name

NEAR_TRUE_MT = {170}
NEAR_TRUE_MW = {80}

OUT_DIR = "plots_dl_exact_solve"
os.makedirs(OUT_DIR, exist_ok=True)

# %%
f = uproot.open(INPUT_FILE)
print(f.keys())
tree = f[TREE_NAME] if TREE_NAME in f else f[f.keys()[0].split(";")[0]]
arr = tree.arrays(library="np")

correct = arr["correct_pairing"].astype(bool)
near = np.isin(arr["mt_hyp"], list(NEAR_TRUE_MT)) & np.isin(arr["mW_hyp"], list(NEAR_TRUE_MW))
exact = correct & near

print(f"total candidates      = {len(exact)}")
print(f"correct_pairing only  = {correct.sum()}")
print(f"near-true mass only   = {near.sum()}")
print(f"exact (both together) = {exact.sum()}")

# %%
def dfrac(reco, true):
    true = np.asarray(true, dtype=float)
    reco = np.asarray(reco, dtype=float)
    out = np.full_like(true, np.nan)
    ok = true != 0
    out[ok] = (reco[ok] - true[ok]) / true[ok]
    return out

CHECKS = [
    ("dpx", dfrac(arr["nu1_px_reco"], arr["nu1_px_true"]), r"$\Delta p_x/p_x^{true}$"),
    ("dpy", dfrac(arr["nu1_py_reco"], arr["nu1_py_true"]), r"$\Delta p_y/p_y^{true}$"),
    ("dpz", dfrac(arr["nu1_pz_reco"], arr["nu1_pz_true"]), r"$\Delta p_z/p_z^{true}$"),
]

for name, vals, xlabel in CHECKS:
    v_exact = vals[exact & np.isfinite(vals)]
    v_rest  = vals[~exact & np.isfinite(vals)]

    plt.figure(figsize=(7, 5))
    bins = np.linspace(-4, 4, 100)
    plt.hist(v_rest, bins=bins, density=True, histtype="step", color="gray",
              label=f"everything else (n={len(v_rest)})")
    plt.hist(v_exact, bins=bins, density=True, histtype="step", color="tab:green",
              linewidth=1.5, label=f"exact solve: correct pairing + true mass (n={len(v_exact)})")
    plt.axvline(0.0, color="black", linestyle=":", linewidth=1)
    plt.axvline(-1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel(xlabel + " (nu1)")
    plt.ylabel("Normalized entries")
    plt.title(f"{name}: exact-solve subset vs. rest")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{OUT_DIR}/{name}_exact_solve.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Wrote {fname}")

    # zoom in on the exact subset alone -- is there really a bimodal shape
    # even here, or does it just look small next to the giant "rest" pool?
    plt.figure(figsize=(7, 5))
    plt.hist(v_exact, bins=bins, color="tab:green", alpha=0.7)
    plt.axvline(0.0, color="black", linestyle=":", linewidth=1)
    plt.axvline(-1.0, color="red", linestyle="--", linewidth=1)
    plt.xlabel(xlabel + " (nu1)")
    plt.ylabel("Entries")
    plt.title(f"{name}: exact-solve subset ONLY (n={len(v_exact)})")
    plt.tight_layout()
    fname2 = f"{OUT_DIR}/{name}_exact_solve_only.png"
    plt.savefig(fname2, dpi=150)
    plt.show()
    print(f"Wrote {fname2}")