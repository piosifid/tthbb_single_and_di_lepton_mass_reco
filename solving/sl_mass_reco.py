#!/usr/bin/env python3
"""
Full-combinatorics semileptonic truth-closure test on the Pythia8 sample.

Input:  pythiaOutput_ttH_sl.root   (written by pythiattH_sl.C)
Output: a new .root file with, per event, the true vs. solved (argmax-weight)
        neutrino and "lost quark" momenta -- feeds the same Δpx/px-style
        bias-plot pipeline built earlier for the real analysis.

This is NOT a simplified adapter. Exactly like the real production solver
(solve_ttbar_semileptonic_v5 in mass_reco_functions.py), it:

  * treats the four true b-quarks (blep, bhad, higgsb1, higgsb2) as an
    UNORDERED pool of 4 -- it does NOT use the truth labels to pick out
    "the" leptonic/hadronic b. It runs the full permutations(range(4), 2)
    jet-role search over the pool, exactly like the real jet-pairing
    combinatorics between the 4 real b-jets.
  * uses qvis (the Pythia macro's randomly-kept hadronic-W daughter) as the
    visible leg-B parton, and folds qlost (the randomly-dropped one) into
    MET, using qlost's own real mass as the "lost" invisible particle's
    known mass -- mirroring exactly how the real analysis treats its
    dropped jet.
  * scans the full (mt, mW) mass-hypothesis grid.
  * selects ONE winning solution per event: the single (jet-pairing x
    mass-hypothesis x quartic-root) combination with the highest PDF
    weight, overall -- i.e. mass selection is part of the same argmax,
    not a separate step.

The real neutrino (nu_*) is truth -- it is never given to the solver. It is
only used afterwards to compute the true MET (nu + qlost) and to compare
against the solver's solved leg-A (leptonic) invisible momentum.

REQUIRES: the real production module mass_reco_functions.py importable
(same one used by solve_ttbar_semileptonic_v5) -- edit MASS_RECO_DIR below
to point at the directory that contains it on your system. Also requires
uproot, numpy, numba, and whatever PDF package mass_reco_functions.py
itself needs (parton.mkPDF) -- if solve_ttbar_semileptonic_v5 already runs
in your environment, this will too.
"""

import sys
import math

import numpy as np
import uproot
from itertools import permutations

# ─── EDIT ME: directory containing your real mass_reco_functions.py ───────────
MASS_RECO_DIR = "../../mass_reco_sl/lost_jet/"
sys.path.insert(0, MASS_RECO_DIR)

from mass_reco_functions import (          # noqa: E402  (import after sys.path edit)
    inv_mass,
    _dedup_roots,
    quartic_solver,
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

# ─── config ─────────────────────────────────────────────────────────────────
INPUT_FILE  = "../generation/pythiaOutput_ttH_sl.root"
OUTPUT_FILE = "closure_sl_truth_vs_reco.root"
N_EVENTS    = 2000          # "lets do it for like 200 events at the beginning"
PRINT_EVERY = 20


def _p4(branches, prefix, i):
    """Pull one event's 4-vector out of the flat-branch uproot arrays."""
    return {
        "E":  float(branches[f"{prefix}_E"][i]),
        "px": float(branches[f"{prefix}_px"][i]),
        "py": float(branches[f"{prefix}_py"][i]),
        "pz": float(branches[f"{prefix}_pz"][i]),
    }


def solve_event(lep, nu_true, qvis, qlost_true, pool, pool_labels):
    """
    Full jet-pairing x mass-grid x quartic-root search, mirroring
    solve_ttbar_semileptonic_v5 exactly.

    lep, qvis   : visible 4-vectors (dict E/px/py/pz)
    nu_true     : TRUE neutrino 4-vector -- used only to build the true MET
                  and for the final truth-vs-reco comparison, never fed to
                  the algebra as an unknown.
    qlost_true  : TRUE "lost" hadronic-W daughter -- its momentum is folded
                  into MET and hidden; only its real MASS is used by the
                  solver (exactly like the real analysis's dropped jet).
    pool        : list of 4 true b-quark 4-vectors [blep, bhad, higgsb1, higgsb2]
                  fed to the solver as an UNORDERED pool -- role identity is
                  never used to pick out b_had/b_lep, only recovered
                  afterwards for bookkeeping.
    pool_labels : ["blep", "bhad", "higgsb1", "higgsb2"] -- same order as pool,
                  used only to check afterwards whether the winning
                  combination happened to pick the physically-correct roles.

    Returns the winning solution dict, or None if no candidate solved.
    """
    lep_E, lep_px, lep_py, lep_pz = lep["E"], lep["px"], lep["py"], lep["pz"]
    mlep = inv_mass(lep_E, lep_px, lep_py, lep_pz)

    qvis_E, qvis_px, qvis_py, qvis_pz = qvis["E"], qvis["px"], qvis["py"], qvis["pz"]
    mqvis = inv_mass(qvis_E, qvis_px, qvis_py, qvis_pz)

    m_lost = inv_mass(qlost_true["E"], qlost_true["px"], qlost_true["py"], qlost_true["pz"])
    mlost2 = m_lost ** 2

    # True total invisible momentum (parton-level "MET"): real neutrino +
    # the hadronic-W daughter we're hiding. At pure parton level with no
    # ISR/FSR/MPI (as generated), this is exact by momentum conservation.
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

        higgs_idx = [i for i in range(4) if i not in (b_had_idx, b_lep_idx)]
        bH1_idx, bH2_idx = higgs_idx[0], higgs_idx[1]
        bH1, bH2 = pool[bH1_idx], pool[bH2_idx]
        H_E  = bH1["E"]  + bH2["E"]
        H_px = bH1["px"] + bH2["px"]
        H_py = bH1["py"] + bH2["py"]
        H_pz = bH1["pz"] + bH2["pz"]

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
                        "weight": weight,
                        "mt": mt, "mW": mW,
                        "b_had_idx": b_had_idx, "b_lep_idx": b_lep_idx,
                        "pnux": pnux, "pnuy": pnuy, "pnuz": pnuz,
                        "pbx": pbx, "pby": pby, "pbz": pbz,
                    })

    if not all_solutions:
        return None
    best = max(all_solutions, key=lambda s: s["weight"])
    best["n_solutions"] = len(all_solutions)
    best["role_bhad_label"] = pool_labels[best["b_had_idx"]]
    best["role_blep_label"] = pool_labels[best["b_lep_idx"]]
    best["correct_pairing"] = (
        best["role_bhad_label"] == "bhad" and best["role_blep_label"] == "blep"
    )
    return best


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

    out = {
        "eventNumber":      [],
        "has_solution":     [],
        "n_solutions":      [],
        "weight":           [],
        "mt_hyp":           [],
        "mW_hyp":           [],
        "correct_pairing":  [],
        "nu_px_true": [], "nu_py_true": [], "nu_pz_true": [],
        "nu_px_reco": [], "nu_py_reco": [], "nu_pz_reco": [],
        "lost_px_true": [], "lost_py_true": [], "lost_pz_true": [],
        "lost_px_reco": [], "lost_py_reco": [], "lost_pz_reco": [],
        "m_lost_true": [],
    }

    n_with_solution = 0
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

        best = solve_event(lep, nu_true, qvis, qlost_true, pool, pool_labels)

        out["eventNumber"].append(int(arrs["eventNumber"][i]))
        out["m_lost_true"].append(
            inv_mass(qlost_true["E"], qlost_true["px"], qlost_true["py"], qlost_true["pz"])
        )
        out["nu_px_true"].append(nu_true["px"])
        out["nu_py_true"].append(nu_true["py"])
        out["nu_pz_true"].append(nu_true["pz"])
        out["lost_px_true"].append(qlost_true["px"])
        out["lost_py_true"].append(qlost_true["py"])
        out["lost_pz_true"].append(qlost_true["pz"])

        if best is None:
            out["has_solution"].append(0)
            out["n_solutions"].append(0)
            out["weight"].append(0.0)
            out["mt_hyp"].append(-1.0)
            out["mW_hyp"].append(-1.0)
            out["correct_pairing"].append(0)
            out["nu_px_reco"].append(np.nan)
            out["nu_py_reco"].append(np.nan)
            out["nu_pz_reco"].append(np.nan)
            out["lost_px_reco"].append(np.nan)
            out["lost_py_reco"].append(np.nan)
            out["lost_pz_reco"].append(np.nan)
        else:
            n_with_solution += 1
            out["has_solution"].append(1)
            out["n_solutions"].append(best["n_solutions"])
            out["weight"].append(best["weight"])
            out["mt_hyp"].append(best["mt"])
            out["mW_hyp"].append(best["mW"])
            out["correct_pairing"].append(int(best["correct_pairing"]))
            out["nu_px_reco"].append(best["pnux"])
            out["nu_py_reco"].append(best["pnuy"])
            out["nu_pz_reco"].append(best["pnuz"])
            out["lost_px_reco"].append(best["pbx"])
            out["lost_py_reco"].append(best["pby"])
            out["lost_pz_reco"].append(best["pbz"])

        if (i + 1) % PRINT_EVERY == 0 or i == n_do - 1:
            print(f"  processed {i + 1}/{n_do} events "
                  f"({n_with_solution} with a solution so far)")

    print(f"\nDone: {n_with_solution}/{n_do} events had at least one valid solution.")
    if n_with_solution:
        correct = sum(out["correct_pairing"])
        print(f"  Correct (blep/bhad) role pairing in the winning combo: "
              f"{correct}/{n_with_solution} ({100*correct/n_with_solution:.1f}%)")

    print(f"Writing {OUTPUT_FILE} ...")
    with uproot.recreate(OUTPUT_FILE) as fout:
        fout["closure"] = {k: np.asarray(v) for k, v in out.items()}
    print("Done.")


if __name__ == "__main__":
    main()
