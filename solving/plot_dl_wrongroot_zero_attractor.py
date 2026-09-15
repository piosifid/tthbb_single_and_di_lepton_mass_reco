# %%
"""
Root-spread magnitude turned out equal for px/py/pz -- so that's not it.
But the observed bump is specifically at ratio=-1, i.e. reco==0, not just
"far from truth". This tests a sharper claim: among multi-candidate exact
-solve events (correct pairing + true mt/mW), does the NON-closest-to-truth
candidate's raw pz cluster near 0, while its raw px/py do not?
"""
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_FILE = "closure_dl_allsolutions.root"
TREE_NAME  = "tree"

NEAR_TRUE_MT = {170}
NEAR_TRUE_MW = {80}

OUT_DIR = "plots_dl_wrongroot_zero_attractor"
os.makedirs(OUT_DIR, exist_ok=True)

# %%
f = uproot.open(INPUT_FILE)
tree = f[TREE_NAME] if TREE_NAME in f else f[f.keys()[0].split(";")[0]]
arr = tree.arrays(library="np")
df = pd.DataFrame(arr)

correct = df["correct_pairing"].astype(bool)
near = df["mt_hyp"].isin(NEAR_TRUE_MT) & df["mW_hyp"].isin(NEAR_TRUE_MW)
sub = df[correct & near].copy()

sub["dpz_abs"] = (sub["nu1_pz_reco"] - sub["nu1_pz_true"]).abs()

records = []
for ev, g in sub.groupby("eventNumber"):
    if len(g) < 2:
        continue
    g = g.sort_values("dpz_abs")
    best = g.iloc[0]
    for _, other in g.iloc[1:].iterrows():
        records.append({
            "wrong_px": other["nu1_px_reco"], "true_px": other["nu1_px_true"],
            "wrong_py": other["nu1_py_reco"], "true_py": other["nu1_py_true"],
            "wrong_pz": other["nu1_pz_reco"], "true_pz": other["nu1_pz_true"],
        })
r = pd.DataFrame(records)
print(f"'wrong root' candidates collected: {len(r)}")

# %%
# raw wrong-root values, normalized by |true| so events at different scales
# are comparable -- 0 means "this coordinate collapsed to zero", 1 means
# "landed right on truth".
for comp in ["px", "py", "pz"]:
    r[f"wrong_{comp}_over_true"] = r[f"wrong_{comp}"] / r[f"true_{comp}"].abs()

plt.figure(figsize=(7, 5))
bins = np.linspace(-3, 3, 100)
for comp, color in [("px", "tab:blue"), ("py", "tab:orange"), ("pz", "tab:green")]:
    vals = r[f"wrong_{comp}_over_true"].replace([np.inf, -np.inf], np.nan).dropna()
    plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5,
              color=color, label=f"wrong-root {comp} (median={vals.median():.2f})")
plt.axvline(0.0, color="black", linestyle=":", linewidth=1)
plt.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("wrong-root reco value / |true value|  (0 = collapsed to zero, 1 = matches truth)")
plt.ylabel("Normalized entries")
plt.title("Where does the WRONG root actually land? px vs py vs pz")
plt.legend()
plt.tight_layout()
fname = f"{OUT_DIR}/wrongroot_landing_point.png"
plt.savefig(fname, dpi=150)
plt.show()
print(f"Wrote {fname}")

# %%
frac_near_zero = {}
for comp in ["px", "py", "pz"]:
    vals = r[f"wrong_{comp}_over_true"].replace([np.inf, -np.inf], np.nan).dropna()
    frac_near_zero[comp] = ((vals.abs() < 0.3).sum() / len(vals))
print("fraction of wrong-root candidates landing within 0.3 of true*0 (i.e. near raw-zero):")
print(frac_near_zero)

# %%
# Now the ACTUAL Delta-p/p quantity used in every earlier plot (real jets,
# true-level, all-candidates): (reco-true)/true, for the wrong-root
# candidates only. This is what directly tests: does px/py's wrong root
# show up as a broad tail near -2 (sign flip) instead of a peak at -1
# (collapse to zero) the way pz does?
for comp in ["px", "py", "pz"]:
    r[f"wrong_{comp}_dfrac"] = (r[f"wrong_{comp}"] - r[f"true_{comp}"]) / r[f"true_{comp}"]

plt.figure(figsize=(7, 5))
bins2 = np.linspace(-4, 4, 100)
for comp, color in [("px", "tab:blue"), ("py", "tab:orange"), ("pz", "tab:green")]:
    vals = r[f"wrong_{comp}_dfrac"].replace([np.inf, -np.inf], np.nan).dropna()
    plt.hist(vals, bins=bins2, density=True, histtype="step", linewidth=1.5,
              color=color, label=f"wrong-root {comp} (median={vals.median():.2f})")
plt.axvline(0.0, color="black", linestyle=":", linewidth=1)
plt.axvline(-1.0, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(-2.0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel(r"wrong-root $\Delta p/p^{true}$  (same quantity as all earlier plots)")
plt.ylabel("Normalized entries")
plt.title("Wrong-root failure mode in actual dp/p units: px/py (~-2 tail) vs pz (-1 peak)")
plt.legend()
plt.tight_layout()
fname2 = f"{OUT_DIR}/wrongroot_dpfrac.png"
plt.savefig(fname2, dpi=150)
plt.show()
print(f"Wrote {fname2}")