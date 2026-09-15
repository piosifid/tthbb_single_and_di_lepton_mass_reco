# %%
"""
Decisive test: within the same event, same pairing, same mt/mW hypothesis,
how much does pnux spread across its multiple root-candidates vs how much
does pnuz spread? If z spreads much more than x for the SAME events, that
directly proves the asymmetry is in how the multi-valued root maps onto
each coordinate -- not a population-level artifact.
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

OUT_DIR = "plots_dl_per_event_spread"
os.makedirs(OUT_DIR, exist_ok=True)

# %%
f = uproot.open(INPUT_FILE)
print(f.keys())
tree = f[TREE_NAME] if TREE_NAME in f else f[f.keys()[0].split(";")[0]]
arr = tree.arrays(library="np")
df = pd.DataFrame(arr)

correct = df["correct_pairing"].astype(bool)
near = df["mt_hyp"].isin(NEAR_TRUE_MT) & df["mW_hyp"].isin(NEAR_TRUE_MW)
sub = df[correct & near].copy()
print(f"exact-solve rows = {len(sub)}")

# %%
# group by event: same pairing+mass hypothesis already selected, so within
# one event's group, remaining spread comes only from the multiple roots
# of the quartic (pnux) and whatever pnuy/pnuz that root maps to.
groups = sub.groupby("eventNumber")
sizes = groups.size()
print(f"events with >=2 candidates in exact-solve subset: {(sizes >= 2).sum()} / {len(sizes)}")

multi = groups.filter(lambda g: len(g) >= 2).groupby("eventNumber")

records = []
for ev, g in multi:
    records.append({
        "n": len(g),
        "range_px": g["nu1_px_reco"].max() - g["nu1_px_reco"].min(),
        "range_py": g["nu1_py_reco"].max() - g["nu1_py_reco"].min(),
        "range_pz": g["nu1_pz_reco"].max() - g["nu1_pz_reco"].min(),
        "true_px": g["nu1_px_true"].iloc[0],
        "true_py": g["nu1_py_true"].iloc[0],
        "true_pz": g["nu1_pz_true"].iloc[0],
    })
r = pd.DataFrame(records)
print(f"multi-candidate events used: {len(r)}")

# normalize spread by |true| so it's comparable across events (same idea
# as the dpz_frac fields elsewhere)
r["rel_range_px"] = r["range_px"] / r["true_px"].abs()
r["rel_range_py"] = r["range_py"] / r["true_py"].abs()
r["rel_range_pz"] = r["range_pz"] / r["true_pz"].abs()

print("median relative root-spread within an event:")
print(f"  px: {r['rel_range_px'].median():.3f}")
print(f"  py: {r['rel_range_py'].median():.3f}")
print(f"  pz: {r['rel_range_pz'].median():.3f}")

# %%
plt.figure(figsize=(7, 5))
bins = np.linspace(0, 5, 80)
for col, color, label in [
    ("rel_range_px", "tab:blue", "px"),
    ("rel_range_py", "tab:orange", "py"),
    ("rel_range_pz", "tab:green", "pz"),
]:
    vals = r[col].replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[vals < 5]
    plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5,
              color=color, label=f"{label} (median={vals.median():.2f})")
plt.xlabel("Per-event candidate spread / |true value|")
plt.ylabel("Normalized entries")
plt.title("Within-event root spread: px vs py vs pz (same pairing + mass hyp)")
plt.legend()
plt.tight_layout()
fname = f"{OUT_DIR}/per_event_root_spread.png"
plt.savefig(fname, dpi=150)
plt.show()
print(f"Wrote {fname}")
