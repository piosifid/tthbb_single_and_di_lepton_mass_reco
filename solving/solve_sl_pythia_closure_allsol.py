#!/usr/bin/env python3
"""
Semileptonic Pythia truth-closure test -- ALL-SOLUTIONS variant.

Same physics and combinatorics as solve_sl_pythia_closure.py (full
permutations(range(4), 2) jet-pairing over the true b-quark pool, full
(mt, mW) grid scan, same reused solver functions from mass_reco_functions.py)
-- but instead of keeping only the argmax-weight winner per event, this
writes EVERY valid candidate solution to the output tree, in "long format":
one row per (event, candidate), with a `is_best` flag marking which row was
the per-event argmax winner. This lets the downstream plots compare "the
bias across every candidate the solver considered" against "the bias only
in what it actually picked" -- exactly the additional test requested
alongside the existing max-weight-only pipeline (kept as its own separate
script/output, not replaced).

Input:  pythiaOutput_ttH_sl.root
Output: closure_sl_allsolutions.root  (tree "closure", one row per candidate)

Warning: this is much bigger than the max-weight-only output -- typically
a few hundred valid candidates survive per event, so N_EVENTS=200 can mean
several hundred thousand output rows. Still small/fast for a flat ROOT
tree, but don't casually crank N_EVENTS to the full 10000 without checking
runtime and output size first.
"""

import sys
import math

import numpy as np
import uproot
from itertools import permutations

MASS_RECO_DIR = "."
sys.path.insert(0, MASS_RECO_DIR)

from mass_reco_functions_sl import (          # noqa: E402
    inv_mass,
    _dedup_roots,
    quartic_solver,
    count_quartic_real_roots,
    algebraic_pz,
    _leg_a_geometry,
    _leg_a_scan_dependent,
    _leg_b_geometry,
    _leg_b_scan_dependent,
    _dp_family_with_mass,
    _combine_to_quartic,
    MT_MIN, MT_MAX, MT_STEP,
    MW_MIN, MW_MAX, MW_STEP,
    ECM, Q_SCALE, _pdf_set,
)

HYP_N_ROOTS = []  # real root count at every (jet-pairing, mt, mW) hypothesis point that had >=1 root

INPUT_FILE  = "../generation/pythiaOutput_ttH_sl.root"
OUTPUT_FILE = "closure_sl_allsolutions.root"
N_EVENTS    = 2000
PRINT_EVERY = 20


def _p4(branches, prefix, i):
    return {
        "E":  float(branches[f"{prefix}_E"][i]),
        "px": float(branches[f"{prefix}_px"][i]),
        "py": float(branches[f"{prefix}_py"][i]),
        "pz": float(branches[f"{prefix}_pz"][i]),
    }


def solve_event_all(lep, nu_true, qvis, qlost_true, pool, pool_labels):
    """Same algebra as solve_sl_pythia_closure.py's solve_event, but returns
    the FULL list of valid candidates (each tagged with its role labels and
    whether that candidate's pairing was physically correct), not just the
    argmax winner."""
    lep_E, lep_px, lep_py, lep_pz = lep["E"], lep["px"], lep["py"], lep["pz"]
    mlep = inv_mass(lep_E, lep_px, lep_py, lep_pz)

    qvis_E, qvis_px, qvis_py, qvis_pz = qvis["E"], qvis["px"], qvis["py"], qvis["pz"]
    mqvis = inv_mass(qvis_E, qvis_px, qvis_py, qvis_pz)

    m_lost = inv_mass(qlost_true["E"], qlost_true["px"], qlost_true["py"], qlost_true["pz"])
    mlost2 = m_lost ** 2

    MET_x = nu_true["px"] + qlost_true["px"]
    MET_y = nu_true["py"] + qlost_true["py"]

    pool_masses = [inv_mass(j["E"], j["px"], j["py"], j["pz"]) for j in pool]

    sqr_lp_E, sqr_lp_pz = lep_E ** 2, lep_pz ** 2
    sqr_lm_E = qvis_E ** 2

    all_solutions = []

    for b_had_idx, b_lep_idx in permutations(range(4), 2):
        b_had, b_lep = pool[b_had_idx], pool[b_lep_idx]
        bhad_E, bhad_px, bhad_py, bhad_pz = b_had["E"], b_had["px"], b_had["py"], b_had["pz"]
        blep_E, blep_px, blep_py, blep_pz = b_lep["E"], b_lep["px"], b_lep["py"], b_lep["pz"]
        mb_had = pool_masses[b_had_idx]
        mb_lep = pool_masses[b_lep_idx]

        correct_pairing = (
            pool_labels[b_had_idx] == "bhad" and pool_labels[b_lep_idx] == "blep"
        )

        a2, a3, a4, c20, c10, c00, lp_dot_b = _leg_a_geometry(
            lep_E, lep_px, lep_py, lep_pz, blep_E, blep_px, blep_py, blep_pz
        )
        b2c, b3c, b4c, lm_dot_bb = _leg_b_geometry(
            bhad_E, bhad_px, bhad_py, bhad_pz, qvis_E, qvis_px, qvis_py, qvis_pz
        )

        for mt_int in range(MT_MIN, MT_MAX, MT_STEP):
            mt = float(mt_int)
            for mW_int in range(MW_MIN, MW_MAX, MW_STEP):
                if mt_int < mW_int:
                    continue
                mW = float(mW_int)

                a1v, c22, c21, c11 = _leg_a_scan_dependent(
                    lep_E, lep_px, lep_py, lep_pz, blep_E, mt, mW,
                    mlep, mb_lep, a2, a3, a4, lp_dot_b, sqr_lp_E, sqr_lp_pz
                )
                b1v = _leg_b_scan_dependent(
                    bhad_E, qvis_E, mt, mW, mqvis, mb_had, mlost2, lm_dot_bb, sqr_lm_E
                )
                dp20, dp10, dp00, dp21, dp11, dp22 = _dp_family_with_mass(
                    bhad_E, bhad_px, bhad_py, bhad_pz,
                    qvis_E, qvis_px, qvis_py, qvis_pz,
                    mt, mW, mlost2
                )

                polx, d0, d11, d21, d22, c0 = _combine_to_quartic(
                    c00, c10, c20, c11, c21, c22,
                    dp00, dp10, dp20, dp11, dp21, dp22,
                    MET_x, MET_y
                )
                if abs(polx[4]) < 1e-10:
                    continue
                polx_n = [c / polx[4] for c in polx]
                pnux_roots = _dedup_roots(quartic_solver(polx_n))
                if not pnux_roots:
                    continue
                HYP_N_ROOTS.append(count_quartic_real_roots(polx_n))

                for pnux in pnux_roots:
                    c1 = c10 * pnux + c11
                    c2 = c20 * pnux ** 2 + c21 * pnux + c22
                    d1 = dp10 * pnux + d11
                    d2 = dp20 * pnux ** 2 + d21 * pnux + d22
                    denom = c1 * d0 - c0 * d1
                    if abs(denom) < 1e-6:
                        continue
                    pnuy = (c0 * d2 - c2 * d0) / denom

                    lpbz_diff = lep_E * blep_pz - blep_E * lep_pz
                    if abs(lpbz_diff) < 1e-6:
                        pnuz = algebraic_pz(
                            {"E": blep_E, "px": blep_px, "py": blep_py, "pz": blep_pz},
                            {"E": lep_E, "px": lep_px, "py": lep_py, "pz": lep_pz},
                            mW, mt, mb_lep, mlep, pnux, pnuy
                        )
                        if pnuz is None:
                            continue
                    else:
                        pnuz = (-a1v - a2 * pnux - a3 * pnuy) / a4

                    if abs(b4c) < 1e-6:
                        continue
                    pbx = MET_x - pnux
                    pby = MET_y - pnuy
                    pbz = (-b1v - b2c * pbx - b3c * pby) / b4c

                    if not all(math.isfinite(v) for v in (pnux, pnuy, pnuz, pbx, pby, pbz)):
                        continue

                    pnu_E = math.sqrt(pnux ** 2 + pnuy ** 2 + pnuz ** 2)
                    Wlep_E, Wlep_px, Wlep_py, Wlep_pz = (
                        pnu_E + lep_E, pnux + lep_px, pnuy + lep_py, pnuz + lep_pz
                    )
                    tlep_E, tlep_px, tlep_py, tlep_pz = (
                        Wlep_E + blep_E, Wlep_px + blep_px, Wlep_py + blep_py, Wlep_pz + blep_pz
                    )

                    lost_E = math.sqrt(pbx ** 2 + pby ** 2 + pbz ** 2 + mlost2)
                    What_E, What_px, What_py, What_pz = (
                        lost_E + qvis_E, pbx + qvis_px, pby + qvis_py, pbz + qvis_pz
                    )
                    thad_E, thad_px, thad_py, thad_pz = (
                        What_E + bhad_E, What_px + bhad_px, What_py + bhad_py, What_pz + bhad_pz
                    )

                    higgs_idx = [k for k in range(4) if k not in (b_had_idx, b_lep_idx)]
                    bH1, bH2 = pool[higgs_idx[0]], pool[higgs_idx[1]]
                    H_E  = bH1["E"]  + bH2["E"]
                    H_px = bH1["px"] + bH2["px"]
                    H_py = bH1["py"] + bH2["py"]
                    H_pz = bH1["pz"] + bH2["pz"]
                    higgs_mass_reco = inv_mass(H_E, H_px, H_py, H_pz)

                    ttH_E  = tlep_E + thad_E + H_E
                    ttH_px = tlep_px + thad_px + H_px
                    ttH_py = tlep_py + thad_py + H_py
                    ttH_pz = tlep_pz + thad_pz + H_pz
                    if ttH_E > ECM:
                        continue

                    x1 = (ttH_E + ttH_pz) / ECM
                    x2 = (ttH_E - ttH_pz) / ECM
                    if not (0.0 < x1 < 1.0 and 0.0 < x2 < 1.0):
                        continue

                    pdf1 = _pdf_set.xfxQ(21, x1, Q_SCALE)
                    pdf2 = _pdf_set.xfxQ(21, x2, Q_SCALE)
                    weight = (pdf1 * pdf2) / (x1 * x2)

                    all_solutions.append({
                        "weight": weight, "mt": mt, "mW": mW,
                        "correct_pairing": correct_pairing,
                        "pnux": pnux, "pnuy": pnuy, "pnuz": pnuz,
                        "pbx": pbx, "pby": pby, "pbz": pbz,
                        "higgs_mass_reco": higgs_mass_reco,
                    })

    return all_solutions


def main():
    print(f"Reading {INPUT_FILE} ...")
    f = uproot.open(INPUT_FILE)
    tree = f["events"]

    wanted = []
    for prefix in ("blep", "lep", "nu", "bhad", "qvis", "qlost", "higgsb1", "higgsb2"):
        for comp in ("E", "px", "py", "pz"):
            wanted.append(f"{prefix}_{comp}")
    wanted.append("eventNumber")

    arrs = tree.arrays(wanted, library="np")
    n_available = len(arrs["eventNumber"])
    n_do = min(N_EVENTS, n_available)
    print(f"{n_available} events in file; processing the first {n_do}.")

    rows = {
        "eventNumber": [], "is_best": [], "weight": [], "mt_hyp": [], "mW_hyp": [],
        "correct_pairing": [], "n_solutions": [],
        "nu_px_true": [], "nu_py_true": [], "nu_pz_true": [],
        "nu_px_reco": [], "nu_py_reco": [], "nu_pz_reco": [],
        "lost_px_true": [], "lost_py_true": [], "lost_pz_true": [],
        "lost_px_reco": [], "lost_py_reco": [], "lost_pz_reco": [],
        "higgs_mass_true": [], "higgs_mass_reco": [],
    }

    n_with_solution = 0
    total_rows = 0
    examples_shown = 0
    EXAMPLE_MAX = 5
    for i in range(n_do):
        lep        = _p4(arrs, "lep", i)
        nu_true    = _p4(arrs, "nu", i)
        qvis       = _p4(arrs, "qvis", i)
        qlost_true = _p4(arrs, "qlost", i)
        blep_true    = _p4(arrs, "blep", i)
        bhad_true    = _p4(arrs, "bhad", i)
        higgsb1_true = _p4(arrs, "higgsb1", i)
        higgsb2_true = _p4(arrs, "higgsb2", i)

        pool = [blep_true, bhad_true, higgsb1_true, higgsb2_true]
        pool_labels = ["blep", "bhad", "higgsb1", "higgsb2"]

        solutions = solve_event_all(lep, nu_true, qvis, qlost_true, pool, pool_labels)
        ev_num = int(arrs["eventNumber"][i])
        higgs_mass_true = inv_mass(
            higgsb1_true["E"] + higgsb2_true["E"],
            higgsb1_true["px"] + higgsb2_true["px"],
            higgsb1_true["py"] + higgsb2_true["py"],
            higgsb1_true["pz"] + higgsb2_true["pz"],
        )

        correct_rows = [s for s in solutions if s["correct_pairing"]]
        if correct_rows and examples_shown < EXAMPLE_MAX:
            examples_shown += 1
            hm_recos = sorted(set(round(s["higgs_mass_reco"], 6) for s in correct_rows))
            print(f"\n--- Example event {ev_num}: correct-pairing closure check "
                  f"({len(correct_rows)} correct-pairing rows across the mt/mW grid) ---")
            print(f"  higgs_mass_true = {higgs_mass_true:.4f} GeV")
            print(f"  higgs_mass_reco (correct pairing -- identical for all of them, "
                  f"since it only depends on which jets were picked, not mt/mW/root): {hm_recos}")
            best = min(correct_rows, key=lambda s: (s["pnux"] - nu_true["px"]) ** 2
                       + (s["pnuy"] - nu_true["py"]) ** 2 + (s["pnuz"] - nu_true["pz"]) ** 2)
            print(f"  closest-matching correct-pairing neutrino solution "
                  f"(mt={best['mt']:.0f}, mW={best['mW']:.0f}):")
            print(f"    nu_px: true={nu_true['px']:+8.3f}  reco={best['pnux']:+8.3f}")
            print(f"    nu_py: true={nu_true['py']:+8.3f}  reco={best['pnuy']:+8.3f}")
            print(f"    nu_pz: true={nu_true['pz']:+8.3f}  reco={best['pnuz']:+8.3f}")

        if solutions:
            n_with_solution += 1
            best_w = max(s["weight"] for s in solutions)
            for s in solutions:
                rows["eventNumber"].append(ev_num)
                rows["is_best"].append(int(s["weight"] == best_w))
                rows["weight"].append(s["weight"])
                rows["mt_hyp"].append(s["mt"])
                rows["mW_hyp"].append(s["mW"])
                rows["correct_pairing"].append(int(s["correct_pairing"]))
                rows["n_solutions"].append(len(solutions))
                rows["nu_px_true"].append(nu_true["px"])
                rows["nu_py_true"].append(nu_true["py"])
                rows["nu_pz_true"].append(nu_true["pz"])
                rows["nu_px_reco"].append(s["pnux"])
                rows["nu_py_reco"].append(s["pnuy"])
                rows["nu_pz_reco"].append(s["pnuz"])
                rows["lost_px_true"].append(qlost_true["px"])
                rows["lost_py_true"].append(qlost_true["py"])
                rows["lost_pz_true"].append(qlost_true["pz"])
                rows["lost_px_reco"].append(s["pbx"])
                rows["lost_py_reco"].append(s["pby"])
                rows["lost_pz_reco"].append(s["pbz"])
                rows["higgs_mass_true"].append(higgs_mass_true)
                rows["higgs_mass_reco"].append(s["higgs_mass_reco"])
            total_rows += len(solutions)

        if (i + 1) % PRINT_EVERY == 0 or i == n_do - 1:
            print(f"  processed {i + 1}/{n_do} events "
                  f"({n_with_solution} with >=1 solution, {total_rows} candidate rows so far)")

    print(f"\nDone: {n_with_solution}/{n_do} events had at least one valid solution.")
    print(f"Total candidate rows written: {total_rows}")

    hyp_n_roots = np.asarray(HYP_N_ROOTS, dtype=np.int32)
    vals, counts = np.unique(hyp_n_roots, return_counts=True)
    print(f"\n=== real pnux roots per (jet-pairing, mt, mW) hypothesis point ===")
    for v, c in zip(vals, counts):
        print(f"  {int(v)} roots: {c} hypothesis points ({100*c/len(hyp_n_roots):.1f}%)")
    print(f"  mean={hyp_n_roots.mean():.3f}  RMS={hyp_n_roots.std():.4f}  entries={len(hyp_n_roots)}")

    print(f"Writing {OUTPUT_FILE} ...")
    with uproot.recreate(OUTPUT_FILE) as fout:
        fout["closure"] = {k: np.asarray(v) for k, v in rows.items()}
        fout["hyp_roots"] = {"n_roots": hyp_n_roots}
    print("Done.")


if __name__ == "__main__":
    main()