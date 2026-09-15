"""
SL mass reconstruction v5
==========================
Single method (v4's full 6-jet combinatorial branch has been dropped —
tracked separately in another chat).

Every event is required to have 6 good jets, ranked by b-score:
    ranks 1-4  -> the b/Higgs pool. We search over which of these 4 is
                  b_had and which is b_lep; the remaining 2 are the
                  Higgs pair, by elimination.
    ranks 5-6  -> the two hadronic-W daughter candidates. One is chosen
                  at random and KEPT as the visible "qvis" jet; the OTHER
                  is DROPPED — but before dropping it, its real
                  (measured) mass is recorded, and only its momentum is
                  folded into MET (treated as invisible from that point
                  on).

The event is then solved as a two-invisible-momentum system, reusing the
Sonnenschein-style quartic solver originally written for the ttbar
DILEPTON channel:
    leg A (leptonic top) : lepton         + real neutrino  + b_lep
    leg B (hadronic top) : surviving qvis + "lost quark"    + b_had
The "lost quark" is invisible (its momentum is solved for, like the
neutrino), but — unlike the original DL solver's massless antineutrino —
it is assigned the REAL mass of the jet that was dropped, since we
always have that jet in hand before discarding its momentum info. This
is why this file requires an updated derivation: the standard DL solver
algebra hard-codes both invisible particles as exactly massless, so a
new dp-family (leg B's quadratic-in-momentum coefficients) had to be
re-derived to allow a nonzero, known m_lost. See `_dp_family_with_mass`
below — three of the six coefficients are unaffected by mass, two shift
linearly in m_lost^2, and one (the constant term) shifts quadratically;
the derivation was done symbolically and validated against synthetic
on-shell events (exact recovery to numerical precision).

Weight = PDF only (no Breit-Wigner terms), matching the original DL
solver convention — there is no fully-visible hadronic W to
Breit-Wigner against here, by construction.
"""

import math
import random
import numpy as np
import awkward as ak
from itertools import permutations
from collections import Counter
from numba import njit
from parton import mkPDF
from custom_function import *

# ─── Configuration ─────────────────────────────────────────────────────────────
PDF_DIR        = "/eos/user/p/piosifid/PocketCoffea/ANN_newCoffea/INF_DNN_new"
ECM            = 13600      # centre-of-mass energy [GeV]
Q_SCALE        = 234        # PDF evaluation scale [GeV]
MT_MIN         = 150        # top mass scan start [GeV]
MT_MAX         = 201        # top mass scan end (exclusive) [GeV]
MT_STEP        = 5          # top mass scan step [GeV]
MW_MIN         = 60         # W mass scan start [GeV]
MW_MAX         = 101        # W mass scan end (exclusive) [GeV]
MW_STEP        = 5          # W mass scan step [GeV]
N_JETS         = 6          # every event must have 6 good jets (ranked by b-score)
DEBUG_N_EVENTS = 5          # print weight table for first N events
DEBUG_N_NOSOL_EVENTS = 5    # print failure-reason breakdown for first N no-solution events
DEBUG_N_MOMENTUM_EVENTS = 100  # print truth-vs-reco lost-jet momentum for first N events
RNG_SEED       = 12345      # seed for the random rank5/rank6 drop choice
# ───────────────────────────────────────────────────────────────────────────────
MW_STEP        = 5          # W mass scan step [GeV]
NEAR_TRUE_MT   = {170}   # grid points bracketing the generator top mass (~172.5 GeV)
NEAR_TRUE_MW   = {80}         # grid point nearest the generator W mass (~80.4 GeV)
_pdf_set = mkPDF("CT10", 0, pdfdir=PDF_DIR)

TOL = 1e-5


# ─── Utility ────────────────────────────────────────────────────────────────────

def inv_mass(E, px, py, pz):
    m2 = E**2 - px**2 - py**2 - pz**2
    return math.sqrt(max(0.0, m2))


def sqr(x):
    return x**2


def sign(x):
    return 1 if x >= 0 else -1


def delta_phi(a, b):
    """Difference between two phi angles, wrapped to [-pi, pi).
    Ported from the dilepton (mc) mass_reco_functions.py — same convention,
    used here to build calculate_dr for the SL v5 Higgs-pair DR check."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def calculate_dr(eta1, phi1, eta2, phi2):
    """Delta R between two objects, with proper phi wrapping. Ported
    verbatim from the dilepton (mc) mass_reco_functions.py, where an
    analogous DR-based selection criterion (dr_criterion_for_max_weight)
    was already shown to improve on PDF-weight-only combinatorics."""
    d_eta = eta1 - eta2
    d_phi = delta_phi(phi1, phi2)
    return math.sqrt(d_eta**2 + d_phi**2)


def _dedup_roots(solutions, tol=1e-2):
    unique = []
    for sol in solutions:
        if not any(abs(sol - u) < tol for u in unique):
            unique.append(sol)
    return unique


# ─── Jet index mapping (v5 slots) ──────────────────────────────────────────────

_JET_SLOTS = ["1st_jet_v5", "2nd_jet_v5", "3rd_jet_v5",
               "4th_jet_v5", "5th_jet_v5", "6th_jet_v5"]
_IDX_SLOTS = ["1st_jet_v5_idx", "2nd_jet_v5_idx", "3rd_jet_v5_idx",
               "4th_jet_v5_idx", "5th_jet_v5_idx", "6th_jet_v5_idx"]

def get_original_jet_idx_v5(events, event_idx, local_idx):
    """Map local jet index (0-based within the 6 b-score-ranked slots) to
    the original JetGood index."""
    if local_idx < len(_IDX_SLOTS):
        val = events[_IDX_SLOTS[local_idx]][event_idx]
        try:
            return int(val)
        except (TypeError, ValueError):
            return int(ak.to_numpy(val))
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Sonnenschein-style quartic solver — ported from the DL mass_reco_functions.py,
# generic in the identity of the two visible+invisible legs.
# ═════════════════════════════════════════════════════════════════════════════

def quadratic_solver(polx):
    """Real roots of polx[0] + polx[1]*x + polx[2]*x^2."""
    solutions = []
    if abs(polx[2]) < TOL:
        if abs(polx[1]) > TOL:
            solutions.append(-polx[0] / polx[1])
        return solutions
    discriminant = polx[1]**2 - 4 * polx[2] * polx[0]
    if abs(discriminant) < TOL:
        solutions.append(-0.5 * polx[1] / polx[2])
    elif discriminant > TOL:
        sqrt_disc = math.sqrt(discriminant)
        q = -0.5 * (polx[1] + sign(polx[1]) * sqrt_disc)
        solutions.extend([q / polx[2], polx[0] / q])
    return solutions


def cubic_solver(polx):
    """Real roots of polx[0] + polx[1]*x + polx[2]*x^2 + polx[3]*x^3."""
    if abs(polx[3]) < TOL:
        return quadratic_solver(polx)
    q = (polx[2]**2 - 3 * polx[1]) / 9
    r = (2 * polx[2]**3 - 9 * polx[1] * polx[2] + 27 * polx[0]) / 54
    solutions = []
    if abs(q) < TOL:
        solutions.append(-polx[2] / 3)
    elif q**3 > r**2:
        arg = max(-1.0, min(1.0, r / math.sqrt(q**3)))
        theta = math.acos(arg)
        solutions.extend([
            -2 * math.sqrt(q) * math.cos(theta / 3) - polx[2] / 3,
            -2 * math.sqrt(q) * math.cos((theta + 2 * math.pi) / 3) - polx[2] / 3,
            -2 * math.sqrt(q) * math.cos((theta + 4 * math.pi) / 3) - polx[2] / 3,
        ])
    else:
        sqrt_term = math.sqrt(r**2 - q**3)
        powthrd = abs(-r + sqrt_term) ** (1.0 / 3.0)
        a = sign(-r + sqrt_term) * powthrd
        b = q / a if abs(a) > TOL else 0
        solutions.append(a + b - polx[2] / 3)
    return solutions


def quartic_solver(polx):
    """Real roots of polx[0] + polx[1]*x + ... + polx[4]*x^4."""
    if abs(polx[4]) < TOL:
        return cubic_solver(polx[:4])
    coeffs = [c / polx[4] for c in polx]
    solutions = []
    if abs(coeffs[0]) < TOL:
        solutions.append(0)
        solutions.extend(cubic_solver(coeffs[1:5]))
        return solutions
    e = coeffs[2] - 3 * coeffs[3]**2 / 8
    f = coeffs[1] + coeffs[3]**3 / 8 - coeffs[2] * coeffs[3] / 2
    g = coeffs[0] - 3 * coeffs[3]**4 / 256 + coeffs[3]**2 * coeffs[2] / 16 - coeffs[3] * coeffs[1] / 4
    shift = -coeffs[3] / 4
    if abs(g) < TOL:
        solutions.append(shift)
        solutions.extend([root + shift for root in cubic_solver([f, e, 0, 1])])
    elif abs(f) < TOL:
        for z in quadratic_solver([g, e, 1]):
            if z >= 0:
                solutions.append(math.sqrt(z) + shift)
                solutions.append(-math.sqrt(z) + shift)
    else:
        resolvent = [-f**2, e**2 - 4 * g, 2 * e, 1]
        for h_squared in cubic_solver(resolvent):
            if h_squared > 0:
                h = math.sqrt(h_squared)
                j = (e + h_squared - f / h) / 2
                solutions.extend([root + shift for root in quadratic_solver([j, h, 1])])
                solutions.extend([root + shift for root in quadratic_solver([g / j, -h, 1])])
    return solutions


def algebraic_pz(b, lp, mWp, mt, mb, mlp, pnux, pnuy):
    """
    Fallback pz solver for LEG A (real, massless neutrino) only, used when
    the standard linear formula is numerically singular. Ported verbatim
    from the DL solver — assumes a massless invisible particle, which is
    correct for the real neutrino but NOT re-derived for a massive lost
    quark. For leg B, singular cases are skipped rather than using this
    (see the main loop) to avoid silently applying the wrong mass
    assumption in that rare edge case.
    """
    epsilon = 1e-6

    def evalterm1(a1, pnux, pnuy):
        return a1[0] + a1[1] * pnux + a1[2] * pnuy

    def evalterm2(a2, pnux, pnuy):
        return a2[0] + a2[1] * pnux + a2[2] * pnuy + a2[3] * pnux**2 + a2[4] * pnux * pnuy + a2[5] * pnuy**2

    lpE2_minus_lpz2 = lp["E"]**2 - lp["pz"]**2
    mblp = np.sqrt(np.maximum(0, (b["E"] + lp["E"])**2 - (b["px"] + lp["px"])**2 - (b["py"] + lp["py"])**2 - (b["pz"] + lp["pz"])**2))

    a1 = [(mWp**2 - mlp**2) * lp["pz"] * 0.5 / lpE2_minus_lpz2, lp["px"] * lp["pz"] / lpE2_minus_lpz2, lp["py"] * lp["pz"] / lpE2_minus_lpz2]
    a2 = [(mWp**4 + mlp**4 - 2 * mWp**2 * mlp**2) * 0.25 / lpE2_minus_lpz2,
          (mWp**2 - mlp**2) * lp["px"] / lpE2_minus_lpz2, (mWp**2 - mlp**2) * lp["py"] / lpE2_minus_lpz2,
          -(lp["E"]**2 - lp["px"]**2) / lpE2_minus_lpz2, 2 * lp["px"] * lp["py"] / lpE2_minus_lpz2,
          -(lp["E"]**2 - lp["py"]**2) / lpE2_minus_lpz2]

    b1 = [(mt**2 - mblp**2) * (b["pz"] + lp["pz"]) * 0.5 / ((b["E"] + lp["E"])**2 - (b["pz"] + lp["pz"])**2),
          (b["px"] + lp["px"]) * (b["pz"] + lp["pz"]) / ((b["E"] + lp["E"])**2 - (b["pz"] + lp["pz"])**2),
          (b["py"] + lp["py"]) * (b["pz"] + lp["pz"]) / ((b["E"] + lp["E"])**2 - (b["pz"] + lp["pz"])**2)]
    b2 = [(mt**4 + mblp**4 - 2 * mt**2 * mblp**2) * 0.25 / ((b["E"] + lp["E"])**2 - (b["pz"] + lp["pz"])**2),
          (mt**2 - mblp**2) * (b["px"] + lp["px"]) / ((b["E"] + lp["E"])**2 - (b["pz"] + lp["pz"])**2),
          (mt**2 - mblp**2) * (b["py"] + lp["py"]) / ((b["E"] + lp["E"])**2 - (b["pz"] + lp["pz"])**2),
          -((b["E"] + lp["E"])**2 - (b["px"] + lp["px"])**2) / ((b["E"] + lp["E"])**2 - (b["pz"] + lp["pz"])**2),
          2 * (b["px"] + lp["px"]) * (b["py"] + lp["py"]) / ((b["E"] + lp["E"])**2 - (b["pz"] + lp["pz"])**2),
          -((b["E"] + lp["E"])**2 - (b["py"] + lp["py"])**2) / ((b["E"] + lp["E"])**2 - (b["pz"] + lp["pz"])**2)]

    a1val = evalterm1(a1, pnux, pnuy)
    a2val = evalterm2(a2, pnux, pnuy)
    b1val = evalterm1(b1, pnux, pnuy)
    b2val = evalterm2(b2, pnux, pnuy)

    pnuz_a, pnuz_b = [], []
    radicant_a = a1val**2 + a2val
    if radicant_a >= 0:
        pnuz_a.extend([a1val + np.sqrt(radicant_a), a1val - np.sqrt(radicant_a)])
    elif np.abs(radicant_a) < epsilon:
        pnuz_a.append(a1val)
    radicant_b = b1val**2 + b2val
    if radicant_b >= 0:
        pnuz_b.extend([b1val + np.sqrt(radicant_b), b1val - np.sqrt(radicant_b)])
    elif np.abs(radicant_b) < epsilon:
        pnuz_b.append(b1val)

    if len(pnuz_a) == 0 or len(pnuz_b) == 0:
        return None

    min_diff = float('inf')
    best_pnuz = None
    for a in pnuz_a:
        for b_ in pnuz_b:
            diff = abs(a - b_)
            if diff < min_diff:
                min_diff = diff
                best_pnuz = 0.5 * (a + b_)

    if min_diff < np.sqrt(epsilon):
        return best_pnuz
    return None


@njit
def _dp_family_with_mass(bb_E, bb_px, bb_py, bb_pz,
                          lm_E, lm_px, lm_py, lm_pz,
                          mt, mW, mlost2):
    """
    Leg-B quadratic-in-(pbx,pby) coefficients, generalized to a lost quark
    with mass^2 = mlost2 (0.0 exactly reproduces the original DL-solver
    massless case). "bb" = b_had jet, "lm" = surviving qvis jet (both
    visible, real measured 4-vectors); the invisible "lost" 4-momentum is
    the unknown being eliminated here.

    Derivation: the two leg-B mass-shell equations
        (lm + lost)^2 = mW^2         [pseudo-hadronic-W]
        (bb + lm + lost)^2 = mt^2    [hadronic top]
    are linear in the lost quark's (E, px, py, pz) once E_lost^2 is
    replaced via its own mass shell (E_lost^2 = px^2+py^2+pz^2+mlost2).
    Solving that 2x2 linear system gives E_lost and pbz as linear
    functions of (pbx, pby); substituting those back into the lost
    quark's mass shell gives ONE quadratic constraint in (pbx, pby) —
    that quadratic's six coefficients are exactly dp20, dp10, dp00, dp21,
    dp11, dp22 below.

    Symbolically derived (sympy) and numerically validated: reproduces
    the original massless dp-family exactly at mlost2=0, and recovers
    the true momenta to numerical precision (~1e-12) for synthetic
    on-shell events with a genuinely massive lost quark. Three of the six
    coefficients (dp20, dp10, dp00) turn out to be completely independent
    of mlost2 (pure geometry); dp21, dp11 shift linearly in mlost2; dp22
    shifts quadratically. Auto-generated via sympy CSE from that
    derivation — do not hand-edit.
    """
    t0 = bb_pz*lm_pz
    t1 = bb_E*lm_E
    t2 = 2*t1
    t3 = bb_E**2
    t4 = lm_pz**2
    t5 = t3*t4
    t6 = bb_pz**2
    t7 = lm_E**2
    t8 = t6*t7
    t9 = -t0*t2 + t5 + t8
    t10 = 1/t9
    t11 = lm_px**2
    t12 = t11*t3
    t13 = bb_px**2
    t14 = t13*t7
    t15 = t13*t4
    t16 = t11*t6
    t17 = 2*lm_px
    t18 = bb_px*t1
    t19 = bb_px*t0
    t20 = lm_py**2
    t21 = t20*t3
    t22 = bb_py**2
    t23 = t22*t7
    t24 = t22*t4
    t25 = t20*t6
    t26 = 2*lm_py
    t27 = bb_py*t1
    t28 = bb_py*t0
    t29 = bb_px*lm_py
    t30 = bb_py*lm_px
    t31 = lm_px*lm_py
    t32 = bb_px*bb_py
    t33 = t32*t7
    t34 = t32*t4
    t35 = mlost2*t18
    t36 = bb_E**3
    t37 = lm_E*t36
    t38 = lm_px*t6
    t39 = bb_pz**3
    t40 = lm_pz*t39
    t41 = mlost2*t19
    t42 = lm_px**3
    t43 = bb_px**3
    t44 = t43*t7
    t45 = lm_E**3
    t46 = bb_E*t45
    t47 = bb_px*t46
    t48 = lm_px*t3
    t49 = lm_pz**3
    t50 = bb_pz*t49
    t51 = bb_px*t50
    t52 = bb_px*t4
    t53 = t1*t52
    t54 = mW**2
    t55 = lm_px*t1
    t56 = bb_px*t7
    t57 = t0*t56
    t58 = t19*t20
    t59 = t19*t54
    t60 = lm_px*t0
    t61 = mt**2
    t62 = 2*t31
    t63 = 4*t0
    t64 = t4*t43
    t65 = bb_px*t5
    t66 = bb_px*t23
    t67 = bb_px*t8
    t68 = t56*t61
    t69 = t52*t54
    t70 = 3*t11
    t71 = t18*t20
    t72 = t18*t54
    t73 = bb_px*t24
    t74 = t54*t56
    t75 = t52*t61
    t76 = mlost2*t27
    t77 = lm_py*t6
    t78 = mlost2*t28
    t79 = lm_py**3
    t80 = bb_py**3
    t81 = t7*t80
    t82 = bb_py*t46
    t83 = lm_py*t3
    t84 = bb_py*t50
    t85 = bb_py*t4
    t86 = t1*t85
    t87 = lm_py*t1
    t88 = lm_py*t0
    t89 = bb_py*t7
    t90 = t0*t89
    t91 = t11*t28
    t92 = t28*t54
    t93 = t4*t80
    t94 = bb_py*t5
    t95 = bb_py*t14
    t96 = bb_py*t8
    t97 = t61*t89
    t98 = t54*t85
    t99 = t11*t27
    t100 = 3*t20
    t101 = t27*t54
    t102 = bb_py*t15
    t103 = t54*t89
    t104 = t61*t85
    t105 = lm_E**4
    t106 = lm_px**4
    t107 = lm_py**4
    t108 = lm_pz**4
    t109 = mW**4
    t110 = mlost2**2
    t111 = bb_E**4
    t112 = bb_px**4
    t113 = bb_py**4
    t114 = bb_pz**4
    t115 = mt**4
    t116 = 2*t109
    t117 = 2*t37
    t118 = 2*mlost2
    t119 = 4*lm_px
    t120 = 4*lm_py
    t121 = 4*t42
    t122 = t1*t13
    t123 = 4*t79
    t124 = t1*t22
    t125 = t118*t54
    t126 = t1*t61
    t127 = t118*t3
    t128 = t0*t118
    t129 = 2*t46
    t130 = 2*t50
    t131 = t4*t6
    t132 = 2*t7
    t133 = 2*t11
    t134 = 2*t20
    t135 = 2*t54
    t136 = t0*t11
    t137 = 8*t1
    t138 = t0*t54
    t139 = 8*t31
    t140 = 2*t15
    t141 = 2*t24
    t142 = t1*t135
    t143 = 2*t61
    t144 = t1*t143
    t145 = t135*t4
    t146 = 2*t3
    t147 = t146*t7
    t148 = 2*t0
    t149 = t0*t13
    t150 = t0*t135
    t151 = t0*t22
    t152 = t132*t54
    t153 = t0*t61
    t154 = 2*t5
    t155 = 4*t54
    t156 = 2*t22
    t157 = 2*t8
    t158 = 4*t11
    t159 = 4*t20
    t160 = 2*t4
    dp20 = -t10*(t12 + t14 - t15 - t16 - t17*t18 + t17*t19 + t9)
    dp00 = -t10*(t21 + t23 - t24 - t25 - t26*t27 + t26*t28 + t9)
    dp10 = -2*t10*(t0*t29 + t0*t30 - t1*t29 - t1*t30 + t3*t31 - t31*t6 + t33 - t34)
    dp21 = -t10*(lm_px*t21 - lm_px*t25 + lm_px*t37 + lm_px*t40 + lm_px*t5 + lm_px*t8 + mlost2*t38 - mlost2*t48 - t0*t48 - t1*t38 - t13*t55 + t13*t60 + t14*t17 - t15*t17 - t18*t70 + t19*t70 - t22*t55 + t22*t60 + t26*t33 - t26*t34 - t27*t62 + t28*t62 + t3*t42 - t3*t56 + t35 + t38*t4 - t38*t54 - t41 - t42*t6 + t44 - t47 + t48*t54 + t48*t7 - t51 - t52*t6 + t53 + t54*t55 - t54*t60 - t55*t61 - t55*t63 + t57 + t58 + t59 + t60*t61 - t64 + t65 + t66 + t67 + t68 + t69 - t71 - t72 - t73 - t74 - t75)
    dp11 = -t10*(lm_py*t12 - lm_py*t16 + lm_py*t37 + lm_py*t40 + lm_py*t5 + lm_py*t8 + mlost2*t77 - mlost2*t83 - t0*t83 - t1*t77 - t100*t27 + t100*t28 - t101 - t102 - t103 - t104 - t13*t87 + t13*t88 + t17*t33 - t17*t34 - t18*t62 + t19*t62 - t22*t87 + t22*t88 + t23*t26 - t24*t26 + t3*t79 - t3*t89 + t4*t77 - t54*t77 + t54*t83 + t54*t87 - t54*t88 - t6*t79 - t6*t85 - t61*t87 + t61*t88 - t63*t87 + t7*t83 + t76 - t78 + t81 - t82 - t84 + t86 + t90 + t91 + t92 - t93 + t94 + t95 + t96 + t97 + t98 - t99)
    dp22 = -1/4*t10*(-mlost2*t117 - t0*t116 + t0*t125 + t0*t127 - t0*t137*t20 - t0*t147 - t0*t152 + t1*t116 + t1*t118*t6 - t1*t125 + t1*t140 + t1*t141 - t1*t145 - t101*t120 - t102*t120 - t103*t120 - t104*t120 + t105*t3 - t105*t6 + t106*t3 - t106*t6 + t107*t3 - t107*t6 + t108*t3 - t108*t6 + t109*t3 - t109*t4 - t109*t6 + t109*t7 + t11*t117 + t11*t142 - t11*t144 - t11*t150 + t110*t3 - t110*t6 - t111*t4 + t111*t7 - t112*t4 + t112*t7 - t113*t4 + t113*t7 - t114*t4 + t114*t7 - t115*t4 + t115*t7 + t117*t20 - t117*t4 + t117*t54 - t118*t12 + t118*t122 + t118*t124 + t118*t126 - t118*t131 + t118*t16 - t118*t21 + t118*t25 - t118*t40 + t118*t5 + t118*t8 + t119*t35 - t119*t41 + t119*t44 - t119*t47 - t119*t51 + t119*t53 + t119*t57 + t119*t58 + t119*t59 - t119*t64 + t119*t65 + t119*t66 + t119*t67 + t119*t68 + t119*t69 - t119*t71 - t119*t72 - t119*t73 - t119*t74 - t119*t75 + t12*t132 + t12*t134 + t12*t135 - t12*t148 + t120*t76 - t120*t78 + t120*t81 - t120*t82 - t120*t84 + t120*t86 + t120*t90 + t120*t91 + t120*t92 - t120*t93 + t120*t94 + t120*t95 + t120*t96 + t120*t97 + t120*t98 - t120*t99 - t121*t18 + t121*t19 - t122*t133 - t122*t134 - t122*t135 - t123*t27 + t123*t28 - t124*t133 - t124*t134 - t124*t135 + t125*t6 - t126*t134 - t127*t54 - t127*t7 - t128*t13 - t128*t22 - t128*t61 - t129*t13 - t129*t22 + t129*t54 - t129*t6 - t129*t61 - t13*t130 + t13*t150 + t13*t154 + t13*t157 - t130*t22 + t130*t3 + t130*t54 - t130*t61 - t131*t143 + t131*t155 + t131*t2 + t132*t153 + t132*t21 + t132*t40 - t132*t5 + t133*t149 + t133*t151 + t133*t40 + t133*t5 + t133*t8 + t134*t149 + t134*t151 + t134*t153 - t134*t16 + t134*t40 + t134*t5 + t134*t8 - t135*t14 - t135*t16 + t135*t21 - t135*t23 + t135*t24 - t135*t25 + t135*t40 - t136*t137 + t136*t143 - t137*t138 - t138*t146 + t139*t33 - t139*t34 + t14*t143 - t14*t146 + t14*t148 + t14*t156 + t14*t158 - t140*t22 + t140*t54 - t140*t6 - t140*t61 - t141*t6 - t141*t61 + t142*t20 - t142*t6 - t142*t61 + t143*t23 + t143*t5 + t143*t8 + t144*t4 + t145*t61 - t146*t23 - t146*t8 - t147*t61 - t148*t21 + t148*t23 - t15*t158 - t150*t20 + t150*t22 + t150*t61 - t152*t61 + t154*t22 + t154*t6 + t155*t3*t7 + t156*t8 + t157*t4 + t159*t23 - t159*t24 + t16*t160 - t16*t2 + t160*t25 - t2*t25 + 2*t36*t45 - 4*t38*t52 - 2*t39*t49 - 4*t48*t56 - 4*t77*t85 - 4*t83*t89)
    return dp20, dp10, dp00, dp21, dp11, dp22


# ═════════════════════════════════════════════════════════════════════════════
# Main solver
# ═════════════════════════════════════════════════════════════════════════════

def _leg_a_geometry(lep_E, lep_px, lep_py, lep_pz, blep_E, blep_px, blep_py, blep_pz):
    """Leg A (leptonic top) slopes -- independent of the (mt, mW) scan."""
    a2 = 2 * (blep_E * lep_px - lep_E * blep_px)
    a3 = 2 * (blep_E * lep_py - lep_E * blep_py)
    a4 = 2 * (blep_E * lep_pz - lep_E * blep_pz)
    sqr_lp_E, sqr_lp_pz = lep_E**2, lep_pz**2
    c20 = -4*(sqr_lp_E-lep_px**2)*a4**2 - 4*(sqr_lp_E-sqr_lp_pz)*a2**2 - 8*lep_px*lep_pz*a2*a4
    c10 = -8*(sqr_lp_E-sqr_lp_pz)*a2*a3 + 8*lep_px*lep_py*a4**2 - 8*lep_px*lep_pz*a3*a4 - 8*lep_py*lep_pz*a2*a4
    c00 = -4*(sqr_lp_E-lep_py**2)*a4**2 - 4*(sqr_lp_E-sqr_lp_pz)*a3**2 - 8*lep_py*lep_pz*a3*a4
    lp_dot_b = blep_px*lep_px + blep_py*lep_py + blep_pz*lep_pz
    return a2, a3, a4, c20, c10, c00, lp_dot_b


def _leg_a_scan_dependent(lep_E, lep_px, lep_py, lep_pz, blep_E, mt, mW,
                           mlep, mb_lep, a2, a3, a4, lp_dot_b,
                           sqr_lp_E, sqr_lp_pz):
    """Leg A (leptonic top) pieces that DO depend on the (mt, mW) scan point.
    Real neutrino -- always exactly massless, unaffected by this file's work."""
    delta_q_mlp = mW**2 - mlep**2
    a1v = (blep_E+lep_E)*delta_q_mlp - lep_E*(mt**2-mb_lep**2-mlep**2) + 2*blep_E*sqr_lp_E - 2*lep_E*lp_dot_b
    c22 = (delta_q_mlp*a4)**2 - 4*(sqr_lp_E-sqr_lp_pz)*a1v**2 - 4*delta_q_mlp*lep_pz*a1v*a4
    c21 = -8*(sqr_lp_E-sqr_lp_pz)*a1v*a2 + 4*delta_q_mlp*(lep_px*a4**2-lep_pz*a2*a4) - 8*lep_px*lep_pz*a1v*a4
    c11 = -8*(sqr_lp_E-sqr_lp_pz)*a1v*a3 + 4*delta_q_mlp*(lep_py*a4**2-lep_pz*a3*a4) - 8*lep_py*lep_pz*a1v*a4
    return a1v, c22, c21, c11


def _leg_b_geometry(bhad_E, bhad_px, bhad_py, bhad_pz, qvis_E, qvis_px, qvis_py, qvis_pz):
    """Leg B (hadronic top) slopes -- independent of the (mt, mW) scan."""
    b2 = 2 * (bhad_E * qvis_px - qvis_E * bhad_px)
    b3 = 2 * (bhad_E * qvis_py - qvis_E * bhad_py)
    b4 = 2 * (bhad_E * qvis_pz - qvis_E * bhad_pz)
    lm_dot_bb = bhad_px*qvis_px + bhad_py*qvis_py + bhad_pz*qvis_pz
    return b2, b3, b4, lm_dot_bb


def _leg_b_scan_dependent(bhad_E, qvis_E, mt, mW, mqvis, mb_had, mlost2, lm_dot_bb,
                           sqr_lm_E):
    """Leg B (hadronic top) pieces that DO depend on the (mt, mW) scan point,
    generalized for a lost quark of mass^2 = mlost2 (verified: this shift,
    applied wherever mqvis^2 previously appeared, exactly matches the
    from-scratch derivation for b1 -- only dp22/dp21/dp11 needed the fuller
    treatment in _dp_family_with_mass)."""
    delta_q_mlm = mW**2 - mqvis**2 - mlost2
    b1v = (bhad_E+qvis_E)*delta_q_mlm - qvis_E*(mt**2-mb_had**2-mqvis**2-mlost2) + 2*bhad_E*sqr_lm_E - 2*qvis_E*lm_dot_bb
    return b1v


def _combine_to_quartic(c00, c10, c20, c11, c21, c22,
                         dp00, dp10, dp20, dp11, dp21, dp22,
                         MET_x, MET_y):
    """MET substitution + quartic-in-pnux combination. Mass-independent —
    unchanged from the original DL solver, just fed the new dp-family."""
    d22 = dp22 + MET_x**2*dp20 + MET_y**2*dp00 + MET_x*MET_y*dp10 + MET_x*dp21 + MET_y*dp11
    d21 = -dp21 - 2*MET_x*dp20 - MET_y*dp10
    d11 = -dp11 - 2*MET_y*dp00 - MET_x*dp10
    c0, d0 = c00, dp00

    polx_0 = c0**2*d22**2 + c11*d22*(c11*d0-c0*d11) + c0*c22*(d11**2-2*d0*d22) + c22*d0*(c22*d0-c11*d11)
    polx_1 = c0*d21*(2*c0*d22-c11*d11) + c0*d11*(2*c22*dp10+c21*d11) + c22*d0*(2*c21*d0-c11*dp10) - c0*d22*(c11*dp10+c10*d11) - 2*c0*d0*(c22*d21+c21*d22) - d0*d11*(c11*c21+c10*c22) + c11*d0*(c11*d21+2*c10*d22)
    polx_2 = c0**2*(2*d22*dp20+d21**2) - c0*d21*(c11*dp10+c10*d11) + c11*dp20*(c11*d0-c0*d11) + c0*dp10*(c22*dp10-c10*d22) + c0*d11*(2*c21*dp10+c20*d11) + d0**2*(2*c22*c20+c21**2) - 2*c0*d0*(c22*dp20+c21*d21+c20*d22) + c10*d0*(2*c11*d21+c10*d22) - d0*dp10*(c11*c21+c10*c22) - d0*d11*(c11*c20+c10*c21)
    polx_3 = c0*d21*(2*c0*dp20-c10*dp10) - c0*dp20*(c11*dp10+c10*d11) + c0*dp10*(c21*dp10+2*c20*d11) - 2*c0*d0*(c21*dp20+c20*d21) + c10*d0*(2*c11*dp20+c10*d21) + c20*d0*(2*c21*d0-c10*d11) - d0*dp10*(c11*c20+c10*c21)
    polx_4 = c0**2*dp20**2 + c10*dp20*(c10*d0-c0*dp10) + c20*dp10*(c0*dp10-c10*d0) + c20*d0*(c20*d0-2*c0*dp20)

    return [polx_0, polx_1, polx_2, polx_3, polx_4], d0, d11, d21, d22, c0


def solve_ttbar_semileptonic_v5(events, rng_seed=RNG_SEED):
    """
    v5 SL mass reconstruction.

    Requires 6 good jets per event. Jets ranked 1-4 (by b-score) form the
    b/Higgs pool; jets ranked 5-6 are the two hadronic-W daughter
    candidates. One of those two is randomly kept as the visible "qvis"
    jet; the other is dropped -- its real (measured) mass is recorded,
    and only its momentum is folded into MET, then the event is solved
    treating it as an invisible particle of that known mass.
    """
    rng = random.Random(rng_seed)

    # "second/third/fourth_max_weight" are NOT separate discriminants like
    # max_mean_weight/max_sum_weight -- they're the 2nd/3rd/4th-best PDF
    # weight among DISTINCT Higgs-pair partitions (ties to max_weight's own
    # ranking, not a different weighting scheme). This mirrors the dilepton
    # channel's best/second/third/fourth ladder, and exists so a DR-based
    # selection criterion (dr_criterion_for_max_weight_sl_v5, below) can
    # fall through to the next-best distinct partition when the top-weight
    # one fails a DeltaR cut -- exactly like the dilepton channel already
    # does successfully.
    DISCRIMINANTS = ["max_weight", "max_mean_weight", "max_sum_weight",
                      "second_max_weight", "third_max_weight", "fourth_max_weight"]
    out = {}
    for disc in DISCRIMINANTS:
        for field in ["weight", "higgs_mass", "ttH_mass",
                      "m_t_lep_reco", "m_t_had_reco", "m_W_lep_reco",
                      "m_W_had_reco", "combination"]:
            out[f"{disc}_{field}_per_event"] = []
    out["kept_rank_per_event"]    = []   # which of rank5/rank6 was used as qvis
    out["dropped_rank_per_event"] = []   # which was folded into MET
    out["m_lost_per_event"]       = []   # the real mass used for the invisible particle
    # lost-jet momentum: truth (from the real jet, before we hid it) vs. solved
    # (from the winning max-weight combination) -- for closure/validation plots.
    out["lost_px_true_per_event"]  = []
    out["lost_py_true_per_event"]  = []
    out["lost_pz_true_per_event"]  = []
    out["lost_px_reco_per_event"]  = []
    out["lost_py_reco_per_event"]  = []
    out["lost_pz_reco_per_event"]  = []
    out["all_lost_dpx_per_event"] = []
    out["all_lost_dpy_per_event"] = []
    out["all_lost_dpz_per_event"] = []
    out["all_lost_dpt_per_event"] = []
    out["all_combinations_per_event"] = []   
    out["all_near_true_mass_per_event"] = []
    out["all_weight_per_event"] = []
    out["all_lost_pt_per_event"] = []
    out["all_ttH_pz_per_event"] = []
    # per-event dict: (b_had_idx, b_lep_idx) pool-local ordered role assignment
    # -> its best (max) weight across the whole mt/mW grid. Lets us ask, after
    # the fact, whether the weight argmax actually discriminates the correct
    # role assignment from the wrong ones, or is roughly no better than a
    # random guess over the 12 possible orderings / 6 possible pool partitions.
    out["role_weight_max_per_event"] = []

    n_events = len(events)

    # ── No-solution diagnostics ─────────────────────────────────────────────
    # Tracks WHY an event ends up with no solution at all: how many never had
    # 6 good jets to begin with, and -- for those that did -- which failure
    # mode(s) in the combinatorial/quartic search were responsible. Counted
    # per-event as "did this reason occur at least once for this event"
    # (not total combo-level occurrences), since what matters for
    # understanding a no-solution event is which walls it hit, not how many
    # times it hit them.
    n_lt6jets = 0
    n_nosol_with6jets = 0
    nosol_reason_event_counts = Counter()
    n_nosol_printed = 0

    # per-event truth-vs-reco lost-jet momentum, kept for the first
    # DEBUG_N_MOMENTUM_EVENTS events -- used for the summary mean printed
    # after the event loop.
    momentum_debug = {"px_true": [], "px_reco": [], "py_true": [], "py_reco": [],
                       "pz_true": [], "pz_reco": [], "pt_true": [], "pt_reco": []}

    for event_idx in range(n_events):

        def _append_empty():
            for disc in DISCRIMINANTS:
                out[f"{disc}_weight_per_event"].append(0)
                for field in ["higgs_mass", "ttH_mass", "m_t_lep_reco",
                              "m_t_had_reco", "m_W_lep_reco",
                              "m_W_had_reco", "combination"]:
                    out[f"{disc}_{field}_per_event"].append(None)
            out["kept_rank_per_event"].append(None)
            out["dropped_rank_per_event"].append(None)
            out["m_lost_per_event"].append(None)
            out["lost_px_true_per_event"].append(None)
            out["lost_py_true_per_event"].append(None)
            out["lost_pz_true_per_event"].append(None)
            out["lost_px_reco_per_event"].append(None)
            out["lost_py_reco_per_event"].append(None)
            out["lost_pz_reco_per_event"].append(None)
            out["role_weight_max_per_event"].append(None)

        # ── Lepton ────────────────────────────────────────────────────────────
        lep    = events["lepton"][event_idx]
        lep_px = float(lep["px"]); lep_py = float(lep["py"])
        lep_pz = float(lep["pz"]); lep_E  = float(lep["E"])
        mlep   = inv_mass(lep_E, lep_px, lep_py, lep_pz)

        # ── MET ───────────────────────────────────────────────────────────────
        MET     = events["MET_few"][event_idx]
        MET_pt  = float(MET["pt"]); MET_phi = float(MET["phi"])
        MET_x0  = MET_pt * math.cos(MET_phi)
        MET_y0  = MET_pt * math.sin(MET_phi)

        # ── 6 jets required ──────────────────────────────────────────────────
        jets = []
        for slot in _JET_SLOTS:
            jt = events[slot][event_idx]
            if jt is not None and abs(float(jt["pt"])) > 1e-3:
                jets.append(jt)

        if len(jets) < 6:
            n_lt6jets += 1
            _append_empty()
            continue

        pool = jets[0:4]          # ranks 1-4: b/Higgs pool
        w_candidates = jets[4:6]  # ranks 5-6: the two W_had daughter candidates

        keep_local = rng.choice([0, 1])   # 0 -> rank5 kept, 1 -> rank6 kept
        drop_local = 1 - keep_local
        qvis_jet    = w_candidates[keep_local]
        dropped_jet = w_candidates[drop_local]
        kept_rank    = 5 + keep_local
        dropped_rank = 5 + drop_local

        qvis_E, qvis_px, qvis_py, qvis_pz = (
            float(qvis_jet["E"]), float(qvis_jet["px"]),
            float(qvis_jet["py"]), float(qvis_jet["pz"])
        )
        mqvis = inv_mass(qvis_E, qvis_px, qvis_py, qvis_pz)

        dropped_E, dropped_px, dropped_py, dropped_pz = (
            float(dropped_jet["E"]), float(dropped_jet["px"]),
            float(dropped_jet["py"]), float(dropped_jet["pz"])
        )
        m_lost  = inv_mass(dropped_E, dropped_px, dropped_py, dropped_pz)
        mlost2  = m_lost**2

        # fold the dropped jet's momentum into MET
        MET_x = MET_x0 + dropped_px
        MET_y = MET_y0 + dropped_py

        pool_masses = [inv_mass(float(j["E"]), float(j["px"]), float(j["py"]), float(j["pz"]))
                       for j in pool]

        sqr_lp_E, sqr_lp_pz = lep_E**2, lep_pz**2
        sqr_lm_E = qvis_E**2

        all_solutions   = []
        weights_by_key  = {}
        kinematics_by_key = {}
        reason_counts   = Counter()   # this event's combinatorial failure modes
        role_weight_max = {}          # (b_had_idx, b_lep_idx) -> best weight seen

        for b_had_idx, b_lep_idx in permutations(range(4), 2):
            b_had, b_lep = pool[b_had_idx], pool[b_lep_idx]
            bhad_E, bhad_px, bhad_py, bhad_pz = (
                float(b_had["E"]), float(b_had["px"]), float(b_had["py"]), float(b_had["pz"])
            )
            blep_E, blep_px, blep_py, blep_pz = (
                float(b_lep["E"]), float(b_lep["px"]), float(b_lep["py"]), float(b_lep["pz"])
            )
            mb_had = pool_masses[b_had_idx]
            mb_lep = pool_masses[b_lep_idx]

            higgs_idx = [i for i in range(4) if i not in (b_had_idx, b_lep_idx)]
            bH1_idx, bH2_idx = higgs_idx[0], higgs_idx[1]
            bH1, bH2 = pool[bH1_idx], pool[bH2_idx]
            H_E  = float(bH1["E"])  + float(bH2["E"])
            H_px = float(bH1["px"]) + float(bH2["px"])
            H_py = float(bH1["py"]) + float(bH2["py"])
            H_pz = float(bH1["pz"]) + float(bH2["pz"])
            higgs_mass = inv_mass(H_E, H_px, H_py, H_pz)

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
                        reason_counts["quartic_degenerate"] += 1
                        continue
                    polx_n = [c / polx[4] for c in polx]
                    pnux_roots = _dedup_roots(quartic_solver(polx_n))
                    if not pnux_roots:
                        reason_counts["no_real_roots"] += 1
                        continue

                    for pnux in pnux_roots:

                        c1 = c10*pnux + c11
                        c2 = c20*pnux**2 + c21*pnux + c22
                        d1 = dp10*pnux + d11
                        d2 = dp20*pnux**2 + d21*pnux + d22
                        denom = c1*d0 - c0*d1
                        if abs(denom) < 1e-6:
                            reason_counts["leg_a_denom_singular"] += 1
                            continue
                        pnuy = (c0*d2 - c2*d0) / denom

                        # leg A pz: real neutrino, always massless, algebraic_pz fallback OK
                        lpbz_diff = lep_E*blep_pz - blep_E*lep_pz
                        if abs(lpbz_diff) < 1e-6:
                            pnuz = algebraic_pz(
                                {"E": blep_E, "px": blep_px, "py": blep_py, "pz": blep_pz},
                                {"E": lep_E, "px": lep_px, "py": lep_py, "pz": lep_pz},
                                mW, mt, mb_lep, mlep, pnux, pnuy
                            )
                            if pnuz is None:
                                reason_counts["leg_a_algebraic_pz_failed"] += 1
                                continue
                        else:
                            pnuz = (-a1v - a2*pnux - a3*pnuy) / a4

                        # leg B pz: massive lost quark -- singular case is SKIPPED
                        # (no verified massive-mass fallback formula; see algebraic_pz docstring)
                        if abs(b4c) < 1e-6:
                            reason_counts["leg_b_singular_skipped"] += 1
                            continue
                        pbx = MET_x - pnux
                        pby = MET_y - pnuy
                        pbz = (-b1v - b2c*pbx - b3c*pby) / b4c

                        if not all(math.isfinite(v) for v in (pnux, pnuy, pnuz, pbx, pby, pbz)):
                            reason_counts["nonfinite_momentum"] += 1
                            continue

                        pnu_E = math.sqrt(pnux**2 + pnuy**2 + pnuz**2)
                        Wlep_E, Wlep_px, Wlep_py, Wlep_pz = (
                            pnu_E+lep_E, pnux+lep_px, pnuy+lep_py, pnuz+lep_pz
                        )
                        tlep_E, tlep_px, tlep_py, tlep_pz = (
                            Wlep_E+blep_E, Wlep_px+blep_px, Wlep_py+blep_py, Wlep_pz+blep_pz
                        )
                        mt_lep_reco = inv_mass(tlep_E, tlep_px, tlep_py, tlep_pz)
                        mW_lep_reco = inv_mass(Wlep_E, Wlep_px, Wlep_py, Wlep_pz)

                        # lost quark: NOT massless -- uses its real (known) mass
                        lost_E = math.sqrt(pbx**2 + pby**2 + pbz**2 + mlost2)
                        What_E, What_px, What_py, What_pz = (
                            lost_E+qvis_E, pbx+qvis_px, pby+qvis_py, pbz+qvis_pz
                        )
                        thad_E, thad_px, thad_py, thad_pz = (
                            What_E+bhad_E, What_px+bhad_px, What_py+bhad_py, What_pz+bhad_pz
                        )
                        mt_had_reco = inv_mass(thad_E, thad_px, thad_py, thad_pz)
                        mW_had_reco = inv_mass(What_E, What_px, What_py, What_pz)

                        ttH_E  = tlep_E + thad_E + H_E
                        ttH_px = tlep_px + thad_px + H_px
                        ttH_py = tlep_py + thad_py + H_py
                        ttH_pz = tlep_pz + thad_pz + H_pz
                        if ttH_E > ECM:
                            reason_counts["ttH_E_exceeds_ECM"] += 1
                            continue
                        ttH_mass = inv_mass(ttH_E, ttH_px, ttH_py, ttH_pz)

                        x1 = (ttH_E+ttH_pz) / ECM
                        x2 = (ttH_E-ttH_pz) / ECM
                        if not (0.0 < x1 < 1.0 and 0.0 < x2 < 1.0):
                            reason_counts["x1x2_out_of_range"] += 1
                            continue

                        pdf1 = _pdf_set.xfxQ(21, x1, Q_SCALE)
                        pdf2 = _pdf_set.xfxQ(21, x2, Q_SCALE)
                        weight = (pdf1*pdf2) / (x1*x2)

                        role_key = (b_had_idx, b_lep_idx)
                        if role_key not in role_weight_max or weight > role_weight_max[role_key]:
                            role_weight_max[role_key] = weight

                        key = (mt_int, mW_int, b_had_idx, b_lep_idx, bH1_idx, bH2_idx)
                        kin = {
                            "weight": weight, "higgs_mass": higgs_mass, "ttH_mass": ttH_mass,
                            "m_t_lep_reco": mt_lep_reco, "m_t_had_reco": mt_had_reco,
                            "m_W_lep_reco": mW_lep_reco, "m_W_had_reco": mW_had_reco,
                            "combination": key,
                            # solved momentum of the "lost" quark -- compared against the
                            # real dropped jet's momentum (pbx/pby/pbz_true) below, since we
                            # actually know the truth here (we deliberately hid it).
                            "pbx": pbx, "pby": pby, "pbz": pbz,
                            "ttH_pz": ttH_pz,
                        }
                        if key not in weights_by_key:
                            weights_by_key[key] = []
                            kinematics_by_key[key] = kin
                        weights_by_key[key].append(weight)
                        all_solutions.append(kin)

        # ── Select best combination ───────────────────────────────────────────
        best_max = max(all_solutions, key=lambda s: s["weight"]) if all_solutions else None

        # ── Rank distinct Higgs-pair partitions by weight (best/2nd/3rd/4th) ────
        # Same "walk the weight-ranked list, dedup by Higgs pair" logic the
        # dilepton channel already uses -- needed so a DR-based selection
        # criterion can fall through to the next-best distinct partition when
        # the top-weight one fails a DeltaR cut (dr_criterion_for_max_weight_
        # sl_v5, below). A combination's Higgs pair is the (bH1_idx, bH2_idx)
        # tail of its "combination" key -- two combos share a Higgs pair iff
        # that tail matches, regardless of (mt, mW, b_had_idx, b_lep_idx).
        second_sol = third_sol = fourth_sol = None
        if all_solutions:
            used_higgs_pairs = set()
            for sol in sorted(all_solutions, key=lambda s: -s["weight"]):
                hp = frozenset(sol["combination"][-2:])
                if hp in used_higgs_pairs:
                    continue
                if not used_higgs_pairs:
                    used_higgs_pairs.add(hp)   # this is best_max's own pair -- skip, don't store again
                    continue
                if second_sol is None:
                    second_sol = sol
                elif third_sol is None:
                    third_sol = sol
                elif fourth_sol is None:
                    fourth_sol = sol
                    break
                used_higgs_pairs.add(hp)

        # ── No-solution diagnostics ─────────────────────────────────────────────
        # Event had 6 good jets but every one of the 12 role-assignments x
        # (mt,mW) grid points failed somewhere in the chain. Record which
        # failure mode(s) it hit, and print a breakdown for the first few
        # such events so the pattern is visible, not just the final count.
        if best_max is None:
            n_nosol_with6jets += 1
            for reason in reason_counts:
                nosol_reason_event_counts[reason] += 1
            if n_nosol_printed < DEBUG_N_NOSOL_EVENTS:
                n_nosol_printed += 1
                dropped_pt = math.hypot(dropped_px, dropped_py)
                total_attempts = sum(reason_counts.values())
                print(f"\n====== EVENT {event_idx} — NO SOLUTION (kept rank{kept_rank}, "
                      f"dropped rank{dropped_rank}, m_lost={m_lost:.2f} GeV) ======")
                print(f"  dropped-jet truth: px={dropped_px:.2f} py={dropped_py:.2f} "
                      f"pz={dropped_pz:.2f} pt={dropped_pt:.2f}")
                print(f"  {total_attempts} failed attempts across all role-assignments x (mt,mW) grid points:")
                for reason, cnt in reason_counts.most_common():
                    print(f"    {reason:28s}: {cnt:5d}  ({100*cnt/max(1,total_attempts):.1f}%)")

        # ── Debug printout ────────────────────────────────────────────────────
        # Since the "lost" jet's real 4-vector was known before we hid it, we can
        # directly compare its true px/py/pz/pt against the solver's solved
        # (pbx, pby, pbz) for the winning (max-weight) combination -- this is the
        # real closure test of the mass-aware solver on genuine, non-synthetic
        # events (not just the throwaway validation scripts).
        if event_idx < DEBUG_N_EVENTS and all_solutions:
            best_per_key = {}
            for sol in all_solutions:
                k = sol["combination"]
                if k not in best_per_key or sol["weight"] > best_per_key[k]["weight"]:
                    best_per_key[k] = sol
            print(f"\n====== EVENT {event_idx} — weight table v5 (kept rank{kept_rank}, "
                  f"dropped rank{dropped_rank}, m_lost={m_lost:.2f} GeV) ======")
            for k, sol in sorted(best_per_key.items(), key=lambda x: -x[1]["weight"])[:20]:
                print(f"  {str(k):45s}"
                      f"  mWh={sol['m_W_had_reco']:6.1f}  mth={sol['m_t_had_reco']:6.1f}"
                      f"  mtl={sol['m_t_lep_reco']:6.1f}  mttH={sol['ttH_mass']:8.1f}"
                      f"  mH={sol['higgs_mass']:6.1f}  weight={sol['weight']:12.4f}"
                      f"  pbz={sol['pbz']:7.2f}")
            best_w = max(s["weight"] for s in best_per_key.values())
            print(f"  → {len(best_per_key)} unique combinations, best weight={best_w:.4f}")

            dropped_pt = math.hypot(dropped_px, dropped_py)
            if best_max is not None:
                reco_pt = math.hypot(best_max["pbx"], best_max["pby"])
                print(f"  ── lost-jet momentum: truth vs. solved (winning combo) ──")
                print(f"     px:   truth={dropped_px:8.2f}   solved={best_max['pbx']:8.2f}"
                      f"   Δ={best_max['pbx']-dropped_px:+7.2f}")
                print(f"     py:   truth={dropped_py:8.2f}   solved={best_max['pby']:8.2f}"
                      f"   Δ={best_max['pby']-dropped_py:+7.2f}")
                print(f"     pz:   truth={dropped_pz:8.2f}   solved={best_max['pbz']:8.2f}"
                      f"   Δ={best_max['pbz']-dropped_pz:+7.2f}")
                print(f"     pt:   truth={dropped_pt:8.2f}   solved={reco_pt:8.2f}"
                      f"   Δ={reco_pt-dropped_pt:+7.2f}")
            else:
                print(f"  ── lost-jet momentum: truth px={dropped_px:.2f} py={dropped_py:.2f} "
                      f"pz={dropped_pz:.2f} pt={dropped_pt:.2f}  (no solution found)")

        # ── Momentum debug: truth vs. reco lost-jet px/py/pz/pt, one line per
        # event, for the first DEBUG_N_MOMENTUM_EVENTS events -- independent of
        # DEBUG_N_EVENTS/all_solutions above, kept deliberately compact since
        # the full weight table already covers the first few events in detail.
        # Feeds the mean summary printed after the event loop.
        if event_idx < DEBUG_N_MOMENTUM_EVENTS and best_max is not None:
            dropped_pt_dbg = math.hypot(dropped_px, dropped_py)
            reco_pt_dbg = math.hypot(best_max["pbx"], best_max["pby"])
            momentum_debug["px_true"].append(dropped_px)
            momentum_debug["px_reco"].append(best_max["pbx"])
            momentum_debug["py_true"].append(dropped_py)
            momentum_debug["py_reco"].append(best_max["pby"])
            momentum_debug["pz_true"].append(dropped_pz)
            momentum_debug["pz_reco"].append(best_max["pbz"])
            momentum_debug["pt_true"].append(dropped_pt_dbg)
            momentum_debug["pt_reco"].append(reco_pt_dbg)
            dpx_dbg = best_max['pbx'] - dropped_px
            dpy_dbg = best_max['pby'] - dropped_py
            dpz_dbg = best_max['pbz'] - dropped_pz
            dpt_dbg = reco_pt_dbg - dropped_pt_dbg

            print(f"[momentum debug] event {event_idx}:")
            print(f"  px  true={dropped_px:8.2f} reco={best_max['pbx']:8.2f}  dpx={dpx_dbg:8.2f}")
            print(f"  py  true={dropped_py:8.2f} reco={best_max['pby']:8.2f}  dpy={dpy_dbg:8.2f}")
            print(f"  pz  true={dropped_pz:8.2f} reco={best_max['pbz']:8.2f}  dpz={dpz_dbg:8.2f}")
            print(f"  pt  true={dropped_pt_dbg:8.2f} reco={reco_pt_dbg:8.2f}  dpt={dpt_dbg:8.2f}")

        best_mean_w = best_sum_w = -999
        best_mean_k = best_sum_k = None
        for key, wlist in weights_by_key.items():
            mean_w = float(np.mean(wlist)); sum_w = float(np.sum(wlist))
            if mean_w > best_mean_w:
                best_mean_w = mean_w; best_mean_k = key
            if sum_w > best_sum_w:
                best_sum_w = sum_w; best_sum_k = key

        def _store(disc, sol, extra_w=None):
            w = extra_w if extra_w is not None else (sol["weight"] if sol else 0)
            out[f"{disc}_weight_per_event"].append(w)
            for field in ["higgs_mass", "ttH_mass", "m_t_lep_reco",
                          "m_t_had_reco", "m_W_lep_reco", "m_W_had_reco", "combination"]:
                out[f"{disc}_{field}_per_event"].append(sol[field] if sol else None)

        _store("max_weight", best_max)
        _store("max_mean_weight",
               kinematics_by_key.get(best_mean_k) if best_mean_k else None,
               best_mean_w if best_mean_k else 0)
        _store("max_sum_weight",
               kinematics_by_key.get(best_sum_k) if best_sum_k else None,
               best_sum_w if best_sum_k else 0)

        # 2nd/3rd/4th-best DISTINCT Higgs-pair partitions, ranked by weight --
        # feeds dr_criterion_for_max_weight_sl_v5's fallback ladder.
        _store("second_max_weight", second_sol)
        _store("third_max_weight",  third_sol)
        _store("fourth_max_weight", fourth_sol)

        out["kept_rank_per_event"].append(kept_rank if best_max else None)
        out["dropped_rank_per_event"].append(dropped_rank if best_max else None)
        out["m_lost_per_event"].append(m_lost if best_max else None)
        dropped_pt = math.hypot(dropped_px, dropped_py)
        out["all_lost_dpx_per_event"].append([s["pbx"] - dropped_px for s in all_solutions])
        out["all_lost_dpy_per_event"].append([s["pby"] - dropped_py for s in all_solutions])
        out["all_lost_dpz_per_event"].append([s["pbz"] - dropped_pz for s in all_solutions])
        out["all_lost_dpt_per_event"].append([math.hypot(s["pbx"], s["pby"]) - dropped_pt for s in all_solutions])
        out["all_combinations_per_event"].append([s["combination"] for s in all_solutions])   # NEW
        out["all_near_true_mass_per_event"].append([
            int(s["combination"][0] in NEAR_TRUE_MT and s["combination"][1] in NEAR_TRUE_MW)
            for s in all_solutions
        ])
        if best_max is not None and best_max["weight"] != 0:
            out["all_weight_per_event"].append(
                [s["weight"] / best_max["weight"] for s in all_solutions]
            )
        else:
            out["all_weight_per_event"].append([None] * len(all_solutions))
        out["all_lost_pt_per_event"].append([math.hypot(s["pbx"], s["pby"]) for s in all_solutions])
        out["all_ttH_pz_per_event"].append([s["ttH_pz"] for s in all_solutions])
        out["lost_px_true_per_event"].append(dropped_px)
        out["lost_py_true_per_event"].append(dropped_py)
        out["lost_pz_true_per_event"].append(dropped_pz)
        out["lost_px_reco_per_event"].append(best_max["pbx"] if best_max else None)
        out["lost_py_reco_per_event"].append(best_max["pby"] if best_max else None)
        out["lost_pz_reco_per_event"].append(best_max["pbz"] if best_max else None)
        out["role_weight_max_per_event"].append(role_weight_max if role_weight_max else None)

    # ── Aggregate no-solution summary ───────────────────────────────────────
    n_total_nosol = n_lt6jets + n_nosol_with6jets
    print(f"\n{'='*70}")
    print(f"=== NO-SOLUTION DIAGNOSTICS ===")
    print(f"  Total events                        : {n_events}")
    print(f"  No solution, total                  : {n_total_nosol}"
          f" ({100*n_total_nosol/max(1,n_events):.1f}%)")
    print(f"    ...fewer than 6 good jets         : {n_lt6jets}"
          f" ({100*n_lt6jets/max(1,n_events):.1f}%)")
    print(f"    ...had 6 jets, but every combo failed: {n_nosol_with6jets}"
          f" ({100*n_nosol_with6jets/max(1,n_events):.1f}%)")
    if n_nosol_with6jets > 0:
        print(f"  Of those {n_nosol_with6jets} events, fraction that hit each failure "
              f"mode at least once:")
        for reason, cnt in nosol_reason_event_counts.most_common():
            print(f"    {reason:28s}: {cnt:5d}/{n_nosol_with6jets}"
                  f" ({100*cnt/n_nosol_with6jets:.1f}%)")
    print(f"{'='*70}\n")

    out["n_events_lt6jets"]            = n_lt6jets
    out["n_events_nosol_with6jets"]    = n_nosol_with6jets
    out["nosol_reason_event_counts"]   = dict(nosol_reason_event_counts)

    # ── Momentum debug summary: mean truth/reco/delta over the events
    # collected above (see DEBUG_N_MOMENTUM_EVENTS). ──────────────────────────
    if momentum_debug["px_true"]:
        n_dbg = len(momentum_debug["px_true"])
        print(f"\n{'='*70}")
        print(f"=== MOMENTUM DEBUG SUMMARY (first {n_dbg} events with a solution) ===")
        for label, key in [("px", "px"), ("py", "py"), ("pz", "pz"), ("pt", "pt")]:
            true_vals = np.array(momentum_debug[f"{key}_true"])
            reco_vals = np.array(momentum_debug[f"{key}_reco"])
            mean_true = true_vals.mean()
            mean_reco = reco_vals.mean()
            mean_delta = (reco_vals - true_vals).mean()
            print(f"  {label:3s}: mean true={mean_true:8.2f}   mean reco={mean_reco:8.2f}"
                  f"   mean Δ(reco-true)={mean_delta:+7.2f}")
        print(f"{'='*70}\n")

    return out


# ─── Truth matching ────────────────────────────────────────────────────────────

def compute_truth_matching_sl_v5(events, results, j_matched, q_matched,
                                  both_in_topN_mask=None, both_in_pool4_mask=None):
    """
    Truth matching for v5. Only the Higgs jets have gen-level truth
    available; combination = (mt, mW, b_had_idx, b_lep_idx, bH1_idx, bH2_idx),
    Higgs local indices are always the last two elements, referencing
    positions within the b/Higgs pool (ranks 1-4), so the standard
    slot-index mapping applies directly.

    both_in_topN_mask: both truth Higgs jets found among ALL N_JETS good
    jets (pool of 4 + the 2 W-daughter candidates at ranks 5-6). Looser --
    a truth jet landing at rank 5/6 still counts here even though it can
    NEVER be picked as a Higgs jet (only the pool of 4 is ever searched
    for the Higgs pair).

    both_in_pool4_mask: both truth Higgs jets found specifically within
    the 4-jet pool (ranks 1-4). This is the actually-correct denominator
    for what the Higgs-pair combinatorial search can ever reach -- use
    this "TRUE efficiency (pool-4)" number, not the top-N one, when you
    want the real ceiling for this search.
    """
    t = len(events)
    correct_matches, higgs_mass_correct, higgs_mass_wrong = [], [], []
    # Higgs-pair DR for the winning (max-weight) combination -- computed for
    # EVERY solved event regardless of correctness (higgs_dr), plus split by
    # truth-correctness (higgs_dr_correct / higgs_dr_wrong), mirroring the
    # higgs_mass_correct/wrong split above. This is the discrimination check
    # requested before porting the dilepton channel's dr_criterion_for_
    # max_weight idea to SL v5: if the two distributions genuinely separate,
    # DR is a usable extra handle on top of PDF weight for this combinatorics
    # problem too.
    higgs_dr, higgs_dr_correct, higgs_dr_wrong = [], [], []

    for event_idx in range(t):
        comb = results["max_weight_combination_per_event"][event_idx]
        hm   = results["max_weight_higgs_mass_per_event"][event_idx]
        jm, qm = j_matched[event_idx], q_matched[event_idx]

        if comb is None or jm is None or qm is None:
            correct_matches.append(None); higgs_mass_correct.append(None); higgs_mass_wrong.append(None)
            higgs_dr.append(None); higgs_dr_correct.append(None); higgs_dr_wrong.append(None)
            continue

        bH1_local, bH2_local = comb[-2], comb[-1]
        orig_bH1 = get_original_jet_idx_v5(events, event_idx, bH1_local)
        orig_bH2 = get_original_jet_idx_v5(events, event_idx, bH2_local)

        if orig_bH1 is None or orig_bH2 is None:
            correct_matches.append(None); higgs_mass_correct.append(None); higgs_mass_wrong.append(None)
            higgs_dr.append(None); higgs_dr_correct.append(None); higgs_dr_wrong.append(None)
            continue

        # DR between the two Higgs-candidate jets of the winning combination
        try:
            eta1 = float(events["JetGood"]["eta"][event_idx][orig_bH1])
            phi1 = float(events["JetGood"]["phi"][event_idx][orig_bH1])
            eta2 = float(events["JetGood"]["eta"][event_idx][orig_bH2])
            phi2 = float(events["JetGood"]["phi"][event_idx][orig_bH2])
            dr = calculate_dr(eta1, phi1, eta2, phi2)
        except Exception:
            dr = None

        truth_pair = {int(jm), int(qm)}
        reco_pair  = {int(orig_bH1), int(orig_bH2)}
        n_correct  = len(truth_pair & reco_pair)

        higgs_dr.append(dr)
        if n_correct == 2:
            correct_matches.append(2.); higgs_mass_correct.append(hm); higgs_mass_wrong.append(None)
            higgs_dr_correct.append(dr); higgs_dr_wrong.append(None)
        elif n_correct == 1:
            correct_matches.append(1.); higgs_mass_correct.append(None); higgs_mass_wrong.append(hm)
            higgs_dr_correct.append(None); higgs_dr_wrong.append(dr)
        else:
            correct_matches.append(0.); higgs_mass_correct.append(None); higgs_mass_wrong.append(hm)
            higgs_dr_correct.append(None); higgs_dr_wrong.append(dr)

    n2 = sum(1 for x in correct_matches if x == 2.)
    n1 = sum(1 for x in correct_matches if x == 1.)
    n0 = sum(1 for x in correct_matches if x == 0.)
    nt = n2 + n1 + n0

    n_both_matched = n_both_solution = n_one_solution = 0
    n2_t = n1_t = n0_t = nt_top = 0
    n2_p4 = n1_p4 = n0_p4 = nt_pool4 = 0

    for event_idx in range(t):
        jm, qm = j_matched[event_idx], q_matched[event_idx]
        comb = results["max_weight_combination_per_event"][event_idx]
        jm_ok = jm is not None and int(jm) >= 0
        qm_ok = qm is not None and int(qm) >= 0
        has_sol = comb is not None
        if jm_ok and qm_ok:
            n_both_matched += 1
            if has_sol:
                n_both_solution += 1
        elif (jm_ok ^ qm_ok) and has_sol:
            n_one_solution += 1

        if both_in_topN_mask is not None and both_in_topN_mask[event_idx] and comb is not None:
            nt_top += 1
            cm = correct_matches[event_idx]
            if cm == 2.:   n2_t += 1
            elif cm == 1.: n1_t += 1
            else:          n0_t += 1

        if both_in_pool4_mask is not None and both_in_pool4_mask[event_idx] and comb is not None:
            nt_pool4 += 1
            cm = correct_matches[event_idx]
            if cm == 2.:   n2_p4 += 1
            elif cm == 1.: n1_p4 += 1
            else:          n0_p4 += 1

    ns = n_both_solution
    print(f"\n=== Truth matching SL v5 (drop-one-W-jet, mass-aware solver) ===")
    print(f"  Total events                                    : {t}")
    print(f"  Events with solution                            : {nt}")
    print(f"  Events with both truth jets matched             : {n_both_matched}")
    print(f"  Events: solution + both matched                 : {ns}")
    print(f"  Events: solution + one matched                  : {n_one_solution}")
    if both_in_topN_mask is not None:
        print(f"  Events: sol + both matched + both in top{N_JETS}     : {nt_top}  <- TOP-{N_JETS} denominator (loose)")
    if both_in_pool4_mask is not None:
        print(f"  Events: sol + both matched + both in POOL-4     : {nt_pool4}  <- TRUE denominator "
              f"(only the pool of 4 is ever searched for the Higgs pair)")
    print(f"")
    print(f"  Out of {nt} events WITH solution:")
    print(f"    both correct : {n2}/{nt} ({100*n2/max(1,nt):.1f}%)")
    print(f"    one correct  : {n1}/{nt} ({100*n1/max(1,nt):.1f}%)")
    print(f"    none correct : {n0}/{nt} ({100*n0/max(1,nt):.1f}%)")
    if both_in_topN_mask is not None:
        print(f"")
        print(f"  Out of {nt_top} events WITH solution + both matched + both in top{N_JETS}:")
        print(f"    both correct : {n2_t}/{nt_top} ({100*n2_t/max(1,nt_top):.1f}%)  <- loose (top-{N_JETS}) efficiency")
        print(f"    one correct  : {n1_t}/{nt_top} ({100*n1_t/max(1,nt_top):.1f}%)")
        print(f"    none correct : {n0_t}/{nt_top} ({100*n0_t/max(1,nt_top):.1f}%)")
    if both_in_pool4_mask is not None:
        print(f"")
        print(f"  Out of {nt_pool4} events WITH solution + both matched + both in POOL-4:")
        print(f"    both correct : {n2_p4}/{nt_pool4} ({100*n2_p4/max(1,nt_pool4):.1f}%)  <- TRUE efficiency (pool-4)")
        print(f"    one correct  : {n1_p4}/{nt_pool4} ({100*n1_p4/max(1,nt_pool4):.1f}%)")
        print(f"    none correct : {n0_p4}/{nt_pool4} ({100*n0_p4/max(1,nt_pool4):.1f}%)")
    print(f"={'='*55}\n")

    # ── DR discrimination check (correct vs wrong Higgs pair) ────────────────
    dr_correct_vals = [x for x in higgs_dr_correct if x is not None]
    dr_wrong_vals   = [x for x in higgs_dr_wrong if x is not None]
    print(f"=== HIGGS-PAIR DR: correct vs wrong (winning combination) ===")
    if dr_correct_vals:
        print(f"  correct pair  (n={len(dr_correct_vals):5d}): "
              f"mean DR={np.mean(dr_correct_vals):.3f}  median={np.median(dr_correct_vals):.3f}")
    else:
        print(f"  correct pair  (n=0): no events")
    if dr_wrong_vals:
        print(f"  wrong pair    (n={len(dr_wrong_vals):5d}): "
              f"mean DR={np.mean(dr_wrong_vals):.3f}  median={np.median(dr_wrong_vals):.3f}")
    else:
        print(f"  wrong pair    (n=0): no events")
    print(f"  (if these two distributions separate, DR is a usable extra "
          f"handle for the combinatorics, same as in the dilepton channel)")
    print(f"={'='*55}\n")

    # Per-event correct_match value, but ONLY for events where both truth
    # Higgs jets actually sit inside the pool of 4 (the only jets ever
    # searched) -- None everywhere else, so histogramming this field
    # automatically reproduces the "TRUE efficiency (pool-4)" numbers
    # printed above.
    correct_match_pool4 = [
        (correct_matches[i] if (both_in_pool4_mask is not None and both_in_pool4_mask[i]) else None)
        for i in range(t)
    ]

    return {
        "correct_matches": correct_matches,
        "higgs_mass_correct": higgs_mass_correct,
        "higgs_mass_wrong": higgs_mass_wrong,
        "higgs_dr": higgs_dr,
        "higgs_dr_correct": higgs_dr_correct,
        "higgs_dr_wrong": higgs_dr_wrong,
        "correct_match_pool4": correct_match_pool4,
    }


# ─── DR-based selection criterion (SL v5 analog of the dilepton channel's ──────
# dr_criterion_for_max_weight) ───────────────────────────────────────────────

_DR_CHOSEN_FIELDS = ["weight", "higgs_mass", "ttH_mass", "m_t_lep_reco",
                     "m_t_had_reco", "m_W_lep_reco", "m_W_had_reco", "combination"]
_DR_RANK_KEYS = {1: "max_weight", 2: "second_max_weight",
                 3: "third_max_weight", 4: "fourth_max_weight"}


def dr_criterion_for_max_weight_sl_v5(events, results, dr_threshold=1.9):
    """
    SL v5 analog of the dilepton channel's dr_criterion_for_max_weight.

    For each event, walks the four weight-ranked, DISTINCT-Higgs-pair
    combinations already stored by solve_ttbar_semileptonic_v5 (max ->
    second -> third -> fourth) and picks the first whose Higgs jet pair has
    DeltaR < dr_threshold. Falls back to the raw max_weight (PDF-only)
    combination if none of the four pass the cut.

    Motivation: the SL v5 "TRUE efficiency" (correctly identifying the
    Higgs pair / b_had-b_lep partition) sits around 25-30% on PDF weight
    alone -- only modestly better than the 1-in-6 random-chance floor for
    picking the right pool partition. The dilepton channel already showed
    that a DeltaR cut on the Higgs jet pair, applied on top of the PDF
    weight ranking, improves on weight-only combinatorics there. This
    checks whether the same idea transfers to SL v5, whose default
    threshold (1.9) is carried over unchanged for now, per the observed
    correct/wrong DR split on the full-stats sample (correct: mean 1.51,
    median 1.37; wrong: mean 2.09, median 2.13 -- 1.9 sits between them,
    same as it does in the dilepton channel's own tuning).

    IMPORTANT: reads combination tuples from `results` (the raw dict
    solve_ttbar_semileptonic_v5 returns), NOT from `events`. Once a list of
    Python tuples gets wired onto `events` via ak.Array(...), each tuple
    becomes an awkward Record indexable only by string field name ("0",
    "1", ... not -2/-1) -- exactly the pattern compute_truth_matching_sl_v5
    already relies on `results` (not `events`) to avoid. `events` is still
    used here, but only for JetGood eta/phi (never Record-wrapped tuples).
    """
    t = len(events)

    res = {f"dr_chosen_{f}": [] for f in _DR_CHOSEN_FIELDS}
    res["dr_rank_used"] = []
    res["dr_chosen_value"] = []
    for r in [1, 2, 3, 4]:
        res[f"dr_rank{r}"] = []

    n_fallback = n_promoted = n_no_reco = 0
    rank_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    def get_dr(event_idx, comb):
        if comb is None:
            return None
        bH1_local, bH2_local = comb[-2], comb[-1]
        orig1 = get_original_jet_idx_v5(events, event_idx, bH1_local)
        orig2 = get_original_jet_idx_v5(events, event_idx, bH2_local)
        if orig1 is None or orig2 is None:
            return None
        n_jg = len(events["JetGood"]["eta"][event_idx])
        if orig1 >= n_jg or orig2 >= n_jg:
            return None
        eta1 = float(events["JetGood"]["eta"][event_idx][orig1])
        phi1 = float(events["JetGood"]["phi"][event_idx][orig1])
        eta2 = float(events["JetGood"]["eta"][event_idx][orig2])
        phi2 = float(events["JetGood"]["phi"][event_idx][orig2])
        return calculate_dr(eta1, phi1, eta2, phi2)

    for event_idx in range(t):
        max_comb = results["max_weight_combination_per_event"][event_idx]

        if max_comb is None:
            n_no_reco += 1
            for f in _DR_CHOSEN_FIELDS:
                res[f"dr_chosen_{f}"].append(None)
            res["dr_rank_used"].append(None)
            res["dr_chosen_value"].append(None)
            for r in [1, 2, 3, 4]:
                res[f"dr_rank{r}"].append(None)
            continue

        combs = {r: results[f"{key}_combination_per_event"][event_idx]
                  for r, key in _DR_RANK_KEYS.items()}
        drs = {r: get_dr(event_idx, combs[r]) for r in [1, 2, 3, 4]}
        for r in [1, 2, 3, 4]:
            res[f"dr_rank{r}"].append(drs[r])

        rank = 0
        for r in [1, 2, 3, 4]:
            if drs[r] is not None and drs[r] < dr_threshold:
                rank = r
                if r != 1:
                    n_promoted += 1
                break
        if rank == 0:
            n_fallback += 1
        rank_counts[rank] += 1
        res["dr_rank_used"].append(rank)
        res["dr_chosen_value"].append(drs.get(rank, drs[1]) if rank else drs[1])

        rank_key = _DR_RANK_KEYS[rank] if rank != 0 else "max_weight"
        for f in _DR_CHOSEN_FIELDS:
            res[f"dr_chosen_{f}"].append(results[f"{rank_key}_{f}_per_event"][event_idx])

    n_with_reco = t - n_no_reco
    print(f"\n=== dr_criterion_for_max_weight_sl_v5 (threshold={dr_threshold}) ===")
    print(f"  Total events              : {t}")
    print(f"  No reco solution          : {n_no_reco}  ({100*n_no_reco/max(1,t):.1f}%)")
    print(f"  --- Of events with reco ({n_with_reco}) ---")
    print(f"  Rank 1 used (DR < {dr_threshold})  : {rank_counts[1]}  ({100*rank_counts[1]/max(1,n_with_reco):.1f}%)")
    print(f"  Rank 2 promoted           : {rank_counts[2]}  ({100*rank_counts[2]/max(1,n_with_reco):.1f}%)")
    print(f"  Rank 3 promoted           : {rank_counts[3]}  ({100*rank_counts[3]/max(1,n_with_reco):.1f}%)")
    print(f"  Rank 4 promoted           : {rank_counts[4]}  ({100*rank_counts[4]/max(1,n_with_reco):.1f}%)")
    print(f"  Fallback (no rank < {dr_threshold}) : {rank_counts[0]}  ({100*rank_counts[0]/max(1,n_with_reco):.1f}%)")
    print(f"  Total promoted (2/3/4)    : {n_promoted}  ({100*n_promoted/max(1,n_with_reco):.1f}%)")
    print(f"  DR criterion overrides    : {n_promoted + rank_counts[0]}  ({100*(n_promoted+rank_counts[0])/max(1,n_with_reco):.1f}%)")
    print(f"=======================================================\n")

    return res


def compute_dr_criterion_efficiency_sl_v5(events, dr_res, j_matched, q_matched,
                                           both_in_topN_mask=None,
                                           both_in_pool4_mask=None):
    """
    Truth-matching efficiency for whichever combination the DR criterion
    actually picked -- computed the same way as compute_truth_matching_sl_v5's
    "TRUE efficiency" number, but reading the DR-selected combination
    instead of the raw PDF-weight argmax. Run this after
    dr_criterion_for_max_weight_sl_v5 to see whether the DR criterion
    actually raises the ~25-30% PDF-only baseline, or not.

    IMPORTANT: reads combination tuples from `dr_res` (the raw dict
    dr_criterion_for_max_weight_sl_v5 returns), NOT from
    events["dr_chosen_combination"] -- once wired onto `events` via
    ak.Array(...), each tuple becomes an awkward Record indexable only by
    string field name, not -2/-1 (see dr_criterion_for_max_weight_sl_v5's
    docstring for the same issue on the source side).

    both_in_pool4_mask: both truth Higgs jets found specifically within
    the 4-jet pool (ranks 1-4) -- see compute_truth_matching_sl_v5's
    docstring for why this, not both_in_topN_mask, is the actually-correct
    denominator for the true ceiling of this search.
    """
    t = len(events)
    n2 = n1 = n0 = nt = 0
    n2_t = n1_t = n0_t = nt_top = 0
    n2_p4 = n1_p4 = n0_p4 = nt_pool4 = 0

    # Per-event plotting fields, mirroring compute_truth_matching_sl_v5's
    # correct_matches / higgs_mass_correct / higgs_mass_wrong shape, but for
    # the DR-selected combination instead of the raw PDF-weight argmax --
    # needed to actually plot the "did DR help" comparison, not just print
    # the summary numbers.
    correct_match_dr    = []
    higgs_mass_correct_dr = []
    higgs_mass_wrong_dr   = []

    for event_idx in range(t):
        comb = dr_res["dr_chosen_combination"][event_idx]
        hm   = dr_res["dr_chosen_higgs_mass"][event_idx]
        jm, qm = j_matched[event_idx], q_matched[event_idx]
        if comb is None or jm is None or qm is None:
            correct_match_dr.append(None)
            higgs_mass_correct_dr.append(None)
            higgs_mass_wrong_dr.append(None)
            continue

        bH1_local, bH2_local = comb[-2], comb[-1]
        orig_bH1 = get_original_jet_idx_v5(events, event_idx, bH1_local)
        orig_bH2 = get_original_jet_idx_v5(events, event_idx, bH2_local)
        if orig_bH1 is None or orig_bH2 is None:
            correct_match_dr.append(None)
            higgs_mass_correct_dr.append(None)
            higgs_mass_wrong_dr.append(None)
            continue

        nt += 1
        truth_pair = {int(jm), int(qm)}
        reco_pair  = {int(orig_bH1), int(orig_bH2)}
        n_correct  = len(truth_pair & reco_pair)
        if n_correct == 2:
            n2 += 1
            correct_match_dr.append(2.); higgs_mass_correct_dr.append(hm); higgs_mass_wrong_dr.append(None)
        elif n_correct == 1:
            n1 += 1
            correct_match_dr.append(1.); higgs_mass_correct_dr.append(None); higgs_mass_wrong_dr.append(hm)
        else:
            n0 += 1
            correct_match_dr.append(0.); higgs_mass_correct_dr.append(None); higgs_mass_wrong_dr.append(hm)

        if both_in_topN_mask is not None and both_in_topN_mask[event_idx]:
            nt_top += 1
            if n_correct == 2:
                n2_t += 1
            elif n_correct == 1:
                n1_t += 1
            else:
                n0_t += 1

        if both_in_pool4_mask is not None and both_in_pool4_mask[event_idx]:
            nt_pool4 += 1
            if n_correct == 2:
                n2_p4 += 1
            elif n_correct == 1:
                n1_p4 += 1
            else:
                n0_p4 += 1

    print(f"\n=== DR-CRITERION EFFICIENCY (dr_chosen_combination) ===")
    print(f"  Out of {nt} events with a DR-selected combination:")
    print(f"    both correct : {n2}/{nt} ({100*n2/max(1,nt):.1f}%)")
    print(f"    one correct  : {n1}/{nt} ({100*n1/max(1,nt):.1f}%)")
    print(f"    none correct : {n0}/{nt} ({100*n0/max(1,nt):.1f}%)")
    if both_in_topN_mask is not None:
        print(f"")
        print(f"  Out of {nt_top} events (+ both truth jets in top{N_JETS}, loose):")
        print(f"    both correct : {n2_t}/{nt_top} ({100*n2_t/max(1,nt_top):.1f}%)"
              f"  <- compare to the PDF-only loose (top-{N_JETS}) efficiency")
        print(f"    one correct  : {n1_t}/{nt_top} ({100*n1_t/max(1,nt_top):.1f}%)")
        print(f"    none correct : {n0_t}/{nt_top} ({100*n0_t/max(1,nt_top):.1f}%)")
    if both_in_pool4_mask is not None:
        print(f"")
        print(f"  Out of {nt_pool4} events (+ both truth jets in POOL-4, the real denominator):")
        print(f"    both correct : {n2_p4}/{nt_pool4} ({100*n2_p4/max(1,nt_pool4):.1f}%)"
              f"  <- compare directly to the PDF-only TRUE efficiency (pool-4) above")
        print(f"    one correct  : {n1_p4}/{nt_pool4} ({100*n1_p4/max(1,nt_pool4):.1f}%)")
        print(f"    none correct : {n0_p4}/{nt_pool4} ({100*n0_p4/max(1,nt_pool4):.1f}%)")
    print(f"={'='*55}\n")

    # Same pool-4-conditioned per-event field as compute_truth_matching_sl_v5's
    # correct_match_pool4, but for the DR-selected combination -- lets the
    # plotting script histogram the PDF-only vs DR-criterion TRUE efficiency
    # (pool-4) comparison directly instead of just printing it.
    correct_match_dr_pool4 = [
        (correct_match_dr[i] if (both_in_pool4_mask is not None and both_in_pool4_mask[i]) else None)
        for i in range(t)
    ]

    return {"n2": n2, "n1": n1, "n0": n0, "nt": nt,
            "n2_t": n2_t, "n1_t": n1_t, "n0_t": n0_t, "nt_top": nt_top,
            "n2_p4": n2_p4, "n1_p4": n1_p4, "n0_p4": n0_p4, "nt_pool4": nt_pool4,
            "correct_match_dr": correct_match_dr,
            "higgs_mass_correct_dr": higgs_mass_correct_dr,
            "higgs_mass_wrong_dr": higgs_mass_wrong_dr,
            "correct_match_dr_pool4": correct_match_dr_pool4}