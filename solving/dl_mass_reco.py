#!/usr/bin/env python3
"""
Full-combinatorics dileptonic truth-closure test on the Pythia8 sample.

Input:  pythiaOutput_ttH.root   (written by pythiattH_dl.C -- both W's are
        forced leptonic, so every kept event is already pure dileptonic)
Output: closure_dl_truth_vs_reco.root -- true vs. solved (argmax-weight)
        momentum for BOTH neutrinos, feeding the same Δpx/px-style
        bias-plot pipeline built for the semileptonic channel.

Mirrors solve_ttbar_dilepton() in your real production DL solver file --
same jet-pairing combinatorics over the 4 true b-quarks (topb, atopb,
higgsb1, higgsb2) treated as an UNORDERED pool (the solver does not use the
truth labels to pick out which b belongs to which top), same (mt, mW) grid
scan, and it reuses your real solver's own low-level algebra
(compute_coefficients, quartic_solver, filter_close_solutions,
algebraic_pz) rather than reimplementing it.

One thing is added here that solve_ttbar_dilepton itself does not do: it
only tracks weight/masses/combination per candidate, never the actual
solved neutrino momentum components. Recording those or nothing else is
the entire point of this closure test, so the per-candidate loop below
also stores (pnux, pnuy, pnuz, pnubx, pnuby, pnubz) for every solution and
keeps only the argmax-weight winner per event (same "winner only, not all
solutions" choice already made for the semileptonic adapter).

Unlike the semileptonic channel, dileptonic has no "lost jet" to drop --
both b's and both leptons are fully visible, so the true MET is exactly
the sum of the two real truth neutrinos:
    MET_x = topnu_px + atopnu_px
    MET_y = topnu_py + atopnu_py
(exact at parton level, no ISR/FSR/MPI/detector smearing in this sample).
"""

import sys
import math

import numpy as np
import uproot
from itertools import permutations

# ─── EDIT ME: directory + module name for your real DL solver file ───────────
# (the file with solve_ttbar_dilepton / compute_coefficients in it --
# uploaded to this conversation as mass_reco_functions_mc_8.py; rename the
# import below if your deployed copy uses a different filename)
MASS_RECO_DL_DIR = "/eos/user/p/piosifid/Main_An_MassReco/mass_reco_jec"
sys.path.insert(0, MASS_RECO_DL_DIR)

from mass_reco_functions_mc import (     # noqa: E402  (import after sys.path edit)
    compute_coefficients,
    quartic_solver,
    filter_close_solutions,
    algebraic_pz,
    _pdf_set,
)

# ─── config ─────────────────────────────────────────────────────────────────
ECM = 13600.0
Q_SCALE = 234.0

MT_MIN, MT_MAX, MT_STEP = 150, 201, 5
MW_MIN, MW_MAX, MW_STEP = 60, 101, 5

INPUT_FILE  = "../generation/pythiaOutput_ttH.root"
OUTPUT_FILE = "closure_dl_truth_vs_reco.root"
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


def solve_event(lp, lm, MET_x, MET_y, pool, pool_labels):
    """
    lp : positive lepton (from the top's W+) -- pairs with the "b" role
    lm : negative lepton (from the antitop's W-) -- pairs with the "bb" role
    MET_x, MET_y : true total invisible momentum (topnu + atopnu)
    pool : [topb_true, atopb_true, higgsb1_true, higgsb2_true] -- UNORDERED,
           the solver does not know which is which, exactly like the real
           jet-pairing combinatorics between the 4 real b-jets.
    pool_labels : ["topb", "atopb", "higgsb1", "higgsb2"], same order as pool.

    Returns the winning (max-weight) solution dict, or None.
    """
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

        higgs_idx = [k for k in range(4) if k not in (i, j)]
        m_idx, n_idx = higgs_idx[0], higgs_idx[1]
        bh1, bh2 = pool[m_idx], pool[n_idx]
        E_higgs  = bh1["E"]  + bh2["E"]
        Px_higgs = bh1["px"] + bh2["px"]
        Py_higgs = bh1["py"] + bh2["py"]
        Pz_higgs = bh1["pz"] + bh2["pz"]

        # ── mass-independent geometry (verbatim from solve_ttbar_dilepton) ──
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
                        pnuz = algebraic_pz(b, lp, mW, mt, mb, mlp, pnux, pnuy)
                        if pnuz is None:
                            continue
                    else:
                        pnuz = (-a1v - a2 * pnux - a3 * pnuy) / a4

                    if abs(lmbbz_diff) < 1e-6:
                        pnubz = algebraic_pz(bb, lm, mW, mt, mbb, mlm, pnubx, pnuby)
                        if pnubz is None:
                            continue
                    else:
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
                    # NOTE: your production solve_ttbar_dilepton only checks
                    # x1<1 and x2<1 (no lower bound). Tightened here to match
                    # the semileptonic adapter's stricter 0<x<1 physical
                    # requirement -- harmless since x1,x2 should be positive
                    # anyway for a system with positive invariant mass.
                    if not (0.0 < x1 < 1.0 and 0.0 < x2 < 1.0):
                        continue

                    w1 = _pdf_set.xfxQ(21, x1, Q_SCALE)
                    w2 = _pdf_set.xfxQ(21, x2, Q_SCALE)
                    weight = (w1 * w2) / (x1 * x2)

                    all_solutions.append({
                        "weight": weight,
                        "mt": mt, "mW": mW,
                        "b_idx": i, "bb_idx": j,
                        "pnux": pnux, "pnuy": pnuy, "pnuz": pnuz,
                        "pnubx": pnubx, "pnuby": pnuby, "pnubz": pnubz,
                    })

    if not all_solutions:
        return None
    best = max(all_solutions, key=lambda s: s["weight"])
    best["n_solutions"] = len(all_solutions)
    best["role_b_label"]  = pool_labels[best["b_idx"]]
    best["role_bb_label"] = pool_labels[best["bb_idx"]]
    best["correct_pairing"] = (
        best["role_b_label"] == "topb" and best["role_bb_label"] == "atopb"
    )
    return best


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

    out = {
        "eventNumber": [], "has_solution": [], "n_solutions": [],
        "weight": [], "mt_hyp": [], "mW_hyp": [], "correct_pairing": [],
        "nu1_px_true": [], "nu1_py_true": [], "nu1_pz_true": [],
        "nu1_px_reco": [], "nu1_py_reco": [], "nu1_pz_reco": [],
        "nu2_px_true": [], "nu2_py_true": [], "nu2_pz_true": [],
        "nu2_px_reco": [], "nu2_py_reco": [], "nu2_pz_reco": [],
    }

    n_with_solution = 0
    for i in range(n_do):
        lp = _p4(arrs, "toplep", i)        # positive lepton, from top's W+
        lm = _p4(arrs, "atoplep", i)       # negative lepton, from antitop's W-
        nu1_true = _p4(arrs, "topnu", i)   # TRUE neutrino (top's W+)
        nu2_true = _p4(arrs, "atopnu", i)  # TRUE antineutrino (antitop's W-)
        topb_true    = _p4(arrs, "topb", i)
        atopb_true   = _p4(arrs, "atopb", i)
        higgsb1_true = _p4(arrs, "higgsb1", i)
        higgsb2_true = _p4(arrs, "higgsb2", i)

        pool = [topb_true, atopb_true, higgsb1_true, higgsb2_true]
        pool_labels = ["topb", "atopb", "higgsb1", "higgsb2"]

        MET_x = nu1_true["px"] + nu2_true["px"]
        MET_y = nu1_true["py"] + nu2_true["py"]

        best = solve_event(lp, lm, MET_x, MET_y, pool, pool_labels)

        out["eventNumber"].append(int(arrs["eventNumber"][i]))
        out["nu1_px_true"].append(nu1_true["px"])
        out["nu1_py_true"].append(nu1_true["py"])
        out["nu1_pz_true"].append(nu1_true["pz"])
        out["nu2_px_true"].append(nu2_true["px"])
        out["nu2_py_true"].append(nu2_true["py"])
        out["nu2_pz_true"].append(nu2_true["pz"])

        if best is None:
            out["has_solution"].append(0)
            out["n_solutions"].append(0)
            out["weight"].append(0.0)
            out["mt_hyp"].append(-1.0)
            out["mW_hyp"].append(-1.0)
            out["correct_pairing"].append(0)
            for k in ("nu1_px_reco", "nu1_py_reco", "nu1_pz_reco",
                      "nu2_px_reco", "nu2_py_reco", "nu2_pz_reco"):
                out[k].append(np.nan)
        else:
            n_with_solution += 1
            out["has_solution"].append(1)
            out["n_solutions"].append(best["n_solutions"])
            out["weight"].append(best["weight"])
            out["mt_hyp"].append(best["mt"])
            out["mW_hyp"].append(best["mW"])
            out["correct_pairing"].append(int(best["correct_pairing"]))
            out["nu1_px_reco"].append(best["pnux"])
            out["nu1_py_reco"].append(best["pnuy"])
            out["nu1_pz_reco"].append(best["pnuz"])
            out["nu2_px_reco"].append(best["pnubx"])
            out["nu2_py_reco"].append(best["pnuby"])
            out["nu2_pz_reco"].append(best["pnubz"])

        if (i + 1) % PRINT_EVERY == 0 or i == n_do - 1:
            print(f"  processed {i + 1}/{n_do} events "
                  f"({n_with_solution} with a solution so far)")

    print(f"\nDone: {n_with_solution}/{n_do} events had at least one valid solution.")
    if n_with_solution:
        correct = sum(out["correct_pairing"])
        print(f"  Correct (topb/atopb) role pairing in the winning combo: "
              f"{correct}/{n_with_solution} ({100*correct/n_with_solution:.1f}%)")

    print(f"Writing {OUTPUT_FILE} ...")
    with uproot.recreate(OUTPUT_FILE) as fout:
        fout["closure"] = {k: np.asarray(v) for k, v in out.items()}
    print("Done.")


if __name__ == "__main__":
    main()
