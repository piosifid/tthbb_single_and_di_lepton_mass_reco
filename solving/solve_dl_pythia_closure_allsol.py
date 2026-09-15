#!/usr/bin/env python3
"""
Dileptonic Pythia truth-closure test -- ALL-SOLUTIONS variant.

Same physics/combinatorics as solve_dl_pythia_closure.py, but writes EVERY
valid candidate solution per event (long format: one row per event x
candidate, with an `is_best` flag marking the per-event argmax winner)
instead of only the winner. See solve_sl_pythia_closure_allsol.py's
docstring for the rationale -- same idea, dileptonic channel.

Input:  pythiaOutput_ttH.root
Output: closure_dl_allsolutions.root  (tree "closure", one row per candidate)

Same size/runtime warning as the SL all-solutions variant: don't jump
N_EVENTS straight to the full sample without checking timing first.
"""

import sys
import math

import numpy as np
import uproot
from itertools import permutations

MASS_RECO_DL_DIR = "."
sys.path.insert(0, MASS_RECO_DL_DIR)

from mass_reco_functions_dl import (       # noqa: E402
    compute_coefficients,
    quartic_solver,
    count_quartic_real_roots,
    filter_close_solutions,
    algebraic_pz,
    _pdf_set,
)

HYP_N_ROOTS = []  # real root count at every (jet-pairing, mt, mW) hypothesis point that had >=1 root

ECM = 13600.0
Q_SCALE = 234.0

MT_MIN, MT_MAX, MT_STEP = 150, 201, 5
MW_MIN, MW_MAX, MW_STEP = 60, 101, 5

INPUT_FILE  = "../generation/pythiaOutput_ttH.root"
OUTPUT_FILE = "closure_dl_allsolutions.root"
N_EVENTS    = 2000
PRINT_EVERY = 20


def inv_mass(E, px, py, pz):
    m2 = E ** 2 - px ** 2 - py ** 2 - pz ** 2
    return math.sqrt(max(0.0, m2))


def sqr(x):
    return x * x


def _p4(branches, prefix, i):
    return {
        "E":  float(branches[f"{prefix}_E"][i]),
        "px": float(branches[f"{prefix}_px"][i]),
        "py": float(branches[f"{prefix}_py"][i]),
        "pz": float(branches[f"{prefix}_pz"][i]),
    }

BRANCH_COUNTS = {"algebraic_pz_top": 0, "algebraic_pz_atop": 0, "linear_pz_total": 0}
 
def solve_event_all(lp, lm, MET_x, MET_y, pool, pool_labels):
    """Same algebra as solve_dl_pythia_closure.py's solve_event, but returns
    the FULL list of valid candidates instead of only the argmax winner."""
    lp_px, lp_py, lp_pz, lp_E = lp["px"], lp["py"], lp["pz"], lp["E"]
    lm_px, lm_py, lm_pz, lm_E = lm["px"], lm["py"], lm["pz"], lm["E"]
    mlp = inv_mass(lp_E, lp_px, lp_py, lp_pz)
    mlm = inv_mass(lm_E, lm_px, lm_py, lm_pz)

    sqr_lp_E, sqr_lp_pz = sqr(lp_E), sqr(lp_pz)
    sqr_lm_E, sqr_lm_pz = sqr(lm_E), sqr(lm_pz)
    sqr_mlp, sqr_mlm = sqr(mlp), sqr(mlm)

    pool_masses = [inv_mass(j["E"], j["px"], j["py"], j["pz"]) for j in pool]

    all_solutions = []

    for i, j in permutations(range(4), 2):
        b, bb = pool[i], pool[j]
        b_px, b_py, b_pz, b_E = b["px"], b["py"], b["pz"], b["E"]
        bb_px, bb_py, bb_pz, bb_E = bb["px"], bb["py"], bb["pz"], bb["E"]
        mb, mbb = pool_masses[i], pool_masses[j]
        sqr_mb, sqr_mbb = sqr(mb), sqr(mbb)

        correct_pairing = (
            pool_labels[i] == "topb" and pool_labels[j] == "atopb"
        )

        higgs_idx = [k for k in range(4) if k not in (i, j)]
        m_idx, n_idx = higgs_idx[0], higgs_idx[1]
        bh1, bh2 = pool[m_idx], pool[n_idx]
        E_higgs  = bh1["E"]  + bh2["E"]
        Px_higgs = bh1["px"] + bh2["px"]
        Py_higgs = bh1["py"] + bh2["py"]
        Pz_higgs = bh1["pz"] + bh2["pz"]
        higgs_mass_reco = inv_mass(E_higgs, Px_higgs, Py_higgs, Pz_higgs)

        a2 = 2 * (b_E * lp_px - lp_E * b_px)
        a3 = 2 * (b_E * lp_py - lp_E * b_py)
        a4 = 2 * (b_E * lp_pz - lp_E * b_pz)

        b2 = 2 * (bb_E * lm_px - lm_E * bb_px)
        b3 = 2 * (bb_E * lm_py - lm_E * bb_py)
        b4 = 2 * (bb_E * lm_pz - lm_E * bb_pz)

        c20 = (-4 * (sqr_lp_E - sqr(lp_px)) * sqr(a4)
               - 4 * (sqr_lp_E - sqr_lp_pz) * sqr(a2)
               - 8 * lp_px * lp_pz * a2 * a4)
        c10 = (-8 * (sqr_lp_E - sqr_lp_pz) * a2 * a3
               + 8 * lp_px * lp_py * sqr(a4)
               - 8 * lp_px * lp_pz * a3 * a4
               - 8 * lp_py * lp_pz * a2 * a4)
        c00 = (-4 * (sqr_lp_E - sqr(lp_py)) * sqr(a4)
               - 4 * (sqr_lp_E - sqr_lp_pz) * sqr(a3)
               - 8 * lp_py * lp_pz * a3 * a4)

        dp20 = (-4 * (sqr_lm_E - sqr(lm_px)) * sqr(b4)
                - 4 * (sqr_lm_E - sqr_lm_pz) * sqr(b2)
                - 8 * lm_px * lm_pz * b2 * b4)
        dp10 = (-8 * (sqr_lm_E - sqr_lm_pz) * b2 * b3
                + 8 * lm_px * lm_py * sqr(b4)
                - 8 * lm_px * lm_pz * b3 * b4
                - 8 * lm_py * lm_pz * b2 * b4)
        dp00 = (-4 * (sqr_lm_E - sqr(lm_py)) * sqr(b4)
                - 4 * (sqr_lm_E - sqr_lm_pz) * sqr(b3)
                - 8 * lm_py * lm_pz * b3 * b4)

        lp_dot_b  = b_px * lp_px + b_py * lp_py + b_pz * lp_pz
        lm_dot_bb = bb_px * lm_px + bb_py * lm_py + bb_pz * lm_pz

        for mt_int in range(MT_MIN, MT_MAX, MT_STEP):
            mt = float(mt_int)
            for mW_int in range(MW_MIN, MW_MAX, MW_STEP):
                if mt_int < mW_int:
                    continue
                mW = float(mW_int)

                result = compute_coefficients(
                    mt, mW, sqr_mlp, sqr_mlm, sqr_mb, sqr_mbb, sqr_lp_E, sqr_lp_pz, sqr_lm_E, sqr_lm_pz,
                    b_E, lp_E, bb_E, lm_E, lp_dot_b, lp_px, lp_py, lp_pz, a2, a3, a4, lm_dot_bb,
                    lm_px, lm_py, lm_pz, b2, b3, b4, dp00, dp10, dp20, c00, c10, c20, MET_x, MET_y
                )
                a1v = result[0]
                c22, c21, c11 = result[1], result[2], result[3]
                b1v = result[4]
                d22, d21, d11 = result[8], result[9], result[10]
                c0 = result[11]
                d0 = result[12]
                polx = list(result[13:18])

                if abs(polx[4]) < 1e-10:
                    continue
                polx_n = [c / polx[4] for c in polx]
                pnux_roots = filter_close_solutions(quartic_solver(polx_n), tol=1e-2)
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
                    pnubx = MET_x - pnux
                    pnuby = MET_y - pnuy

                    
                    lpbz_diff  = lp_E * b_pz  - b_E  * lp_pz
                    lmbbz_diff = lm_E * bb_pz - bb_E * lm_pz
 
                    if abs(lpbz_diff) < 1e-6:
                        BRANCH_COUNTS["algebraic_pz_top"] += 1
                        pnuz = algebraic_pz(b, lp, mW, mt, mb, mlp, pnux, pnuy)
                        if pnuz is None:
                            continue
                    else:
                        BRANCH_COUNTS["linear_pz_total"] += 1
                        pnuz = (-a1v - a2 * pnux - a3 * pnuy) / a4
 
                    if abs(lmbbz_diff) < 1e-6:
                        BRANCH_COUNTS["algebraic_pz_atop"] += 1
                        pnubz = algebraic_pz(bb, lm, mW, mt, mbb, mlm, pnubx, pnuby)
                        if pnubz is None:
                            continue
                    else:
                        BRANCH_COUNTS["linear_pz_total"] += 1
                        pnubz = (-b1v - b2 * pnubx - b3 * pnuby) / b4


                    if not all(math.isfinite(v) for v in (pnux, pnuy, pnuz, pnubx, pnuby, pnubz)):
                        continue

                    pnu_E = math.sqrt(pnux ** 2 + pnuy ** 2 + pnuz ** 2)
                    Wp_E, Wp_px, Wp_py, Wp_pz = pnu_E + lp_E, pnux + lp_px, pnuy + lp_py, pnuz + lp_pz
                    t_E, t_px, t_py, t_pz = Wp_E + b_E, Wp_px + b_px, Wp_py + b_py, Wp_pz + b_pz

                    pnub_E = math.sqrt(pnubx ** 2 + pnuby ** 2 + pnubz ** 2)
                    Wm_E, Wm_px, Wm_py, Wm_pz = pnub_E + lm_E, pnubx + lm_px, pnuby + lm_py, pnubz + lm_pz
                    tb_E, tb_px, tb_py, tb_pz = Wm_E + bb_E, Wm_px + bb_px, Wm_py + bb_py, Wm_pz + bb_pz

                    E_ttH  = t_E + tb_E + E_higgs
                    Px_ttH = t_px + tb_px + Px_higgs
                    Py_ttH = t_py + tb_py + Py_higgs
                    Pz_ttH = t_pz + tb_pz + Pz_higgs
                    if E_ttH > ECM:
                        continue

                    x1 = (E_ttH + Pz_ttH) / ECM
                    x2 = (E_ttH - Pz_ttH) / ECM
                    if not (0.0 < x1 < 1.0 and 0.0 < x2 < 1.0):
                        continue

                    w1 = _pdf_set.xfxQ(21, x1, Q_SCALE)
                    w2 = _pdf_set.xfxQ(21, x2, Q_SCALE)
                    weight = (w1 * w2) / (x1 * x2)

                    all_solutions.append({
                        "weight": weight, "mt": mt, "mW": mW,
                        "correct_pairing": correct_pairing,
                        "pnux": pnux, "pnuy": pnuy, "pnuz": pnuz,
                        "pnubx": pnubx, "pnuby": pnuby, "pnubz": pnubz,
                        "higgs_mass_reco": higgs_mass_reco,
                    })

    return all_solutions


def main():
    print(f"Reading {INPUT_FILE} ...")
    f = uproot.open(INPUT_FILE)
    tree = f["events"]

    wanted = []
    for prefix in ("toplep", "topnu", "topb", "atoplep", "atopnu", "atopb", "higgsb1", "higgsb2"):
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
        "nu1_px_true": [], "nu1_py_true": [], "nu1_pz_true": [],
        "nu1_px_reco": [], "nu1_py_reco": [], "nu1_pz_reco": [],
        "nu2_px_true": [], "nu2_py_true": [], "nu2_pz_true": [],
        "nu2_px_reco": [], "nu2_py_reco": [], "nu2_pz_reco": [],
        "higgs_mass_true": [], "higgs_mass_reco": [],
    }

    n_with_solution = 0
    total_rows = 0
    examples_shown = 0
    EXAMPLE_MAX = 5
    for i in range(n_do):
        lp = _p4(arrs, "toplep", i)
        lm = _p4(arrs, "atoplep", i)
        nu1_true = _p4(arrs, "topnu", i)
        nu2_true = _p4(arrs, "atopnu", i)
        topb_true    = _p4(arrs, "topb", i)
        atopb_true   = _p4(arrs, "atopb", i)
        higgsb1_true = _p4(arrs, "higgsb1", i)
        higgsb2_true = _p4(arrs, "higgsb2", i)

        pool = [topb_true, atopb_true, higgsb1_true, higgsb2_true]
        pool_labels = ["topb", "atopb", "higgsb1", "higgsb2"]

        MET_x = nu1_true["px"] + nu2_true["px"]
        MET_y = nu1_true["py"] + nu2_true["py"]

        solutions = solve_event_all(lp, lm, MET_x, MET_y, pool, pool_labels)
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
            best = min(correct_rows, key=lambda s: (s["pnux"] - nu1_true["px"]) ** 2
                       + (s["pnuy"] - nu1_true["py"]) ** 2 + (s["pnuz"] - nu1_true["pz"]) ** 2
                       + (s["pnubx"] - nu2_true["px"]) ** 2 + (s["pnuby"] - nu2_true["py"]) ** 2
                       + (s["pnubz"] - nu2_true["pz"]) ** 2)
            print(f"  closest-matching correct-pairing neutrino solution "
                  f"(mt={best['mt']:.0f}, mW={best['mW']:.0f}):")
            print(f"    nu1_px: true={nu1_true['px']:+8.3f}  reco={best['pnux']:+8.3f}")
            print(f"    nu1_py: true={nu1_true['py']:+8.3f}  reco={best['pnuy']:+8.3f}")
            print(f"    nu1_pz: true={nu1_true['pz']:+8.3f}  reco={best['pnuz']:+8.3f}")
            print(f"    nu2_px: true={nu2_true['px']:+8.3f}  reco={best['pnubx']:+8.3f}")
            print(f"    nu2_py: true={nu2_true['py']:+8.3f}  reco={best['pnuby']:+8.3f}")
            print(f"    nu2_pz: true={nu2_true['pz']:+8.3f}  reco={best['pnubz']:+8.3f}")

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
                rows["nu1_px_true"].append(nu1_true["px"])
                rows["nu1_py_true"].append(nu1_true["py"])
                rows["nu1_pz_true"].append(nu1_true["pz"])
                rows["nu1_px_reco"].append(s["pnux"])
                rows["nu1_py_reco"].append(s["pnuy"])
                rows["nu1_pz_reco"].append(s["pnuz"])
                rows["nu2_px_true"].append(nu2_true["px"])
                rows["nu2_py_true"].append(nu2_true["py"])
                rows["nu2_pz_true"].append(nu2_true["pz"])
                rows["nu2_px_reco"].append(s["pnubx"])
                rows["nu2_py_reco"].append(s["pnuby"])
                rows["nu2_pz_reco"].append(s["pnubz"])
                rows["higgs_mass_true"].append(higgs_mass_true)
                rows["higgs_mass_reco"].append(s["higgs_mass_reco"])
            total_rows += len(solutions)

        if (i + 1) % PRINT_EVERY == 0 or i == n_do - 1:
            print(f"  processed {i + 1}/{n_do} events "
                  f"({n_with_solution} with >=1 solution, {total_rows} candidate rows so far)")

    print(f"\nDone: {n_with_solution}/{n_do} events had at least one valid solution.")
    print(f"Total candidate rows written: {total_rows}")
    print("BRANCH_COUNTS:", BRANCH_COUNTS)

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