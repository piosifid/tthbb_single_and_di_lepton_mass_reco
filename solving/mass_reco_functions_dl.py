import math
import numpy as np
import awkward as ak
from numba import njit, float64
from parton import mkPDF
from custom_function import *

_pdf_set = mkPDF("CT10", 0, pdfdir="/eos/user/p/piosifid/PocketCoffea/ANN_newCoffea/INF_DNN_new")

def sign(x):
    """Return +1 for x >= 0, -1 for x < 0.
    Differs from np.sign in that sign(0) = +1 (needed by the quartic solver
    convention where the sign of zero discriminant terms defaults positive)."""
    return 1 if x >= 0 else -1


def filter_close_solutions(solutions, tol=1e-6):
    """Deduplicate a list of scalar solutions: any two within `tol` are merged
    (first one wins). O(n^2), intended for small lists (n <= 4 from the quartic solver)."""
    unique = []
    for sol in solutions:
        if not any(abs(sol - u) < tol for u in unique):
            unique.append(sol)
    return unique


@njit
def delta_phi(a, b):
    """Difference between two phi angles, wrapped to [-pi, pi)."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


@njit
def calculate_dr(eta1, phi1, eta2, phi2):
    """Delta R between two objects, with proper phi wrapping."""
    d_eta = eta1 - eta2
    d_phi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(d_eta**2 + d_phi**2)


TOL = 1e-5  # Module-level tolerance for "effectively zero" checks


def quadratic_solver(polx):
    """Real roots of polx[0] + polx[1]*x + polx[2]*x^2.
    Returns list of real roots (0, 1, or 2 entries)."""
    solutions = []
    if abs(polx[2]) < TOL:  # Linear case
        if abs(polx[1]) > TOL:
            solutions.append(-polx[0] / polx[1])
        return solutions

    discriminant = polx[1]**2 - 4 * polx[2] * polx[0]
    if abs(discriminant) < TOL:  # Double root
        solutions.append(-0.5 * polx[1] / polx[2])
    elif discriminant > TOL:  # Two distinct real roots
        sqrt_disc = math.sqrt(discriminant)
        # Numerically stable form (avoids cancellation): see Numerical Recipes §5.6
        q = -0.5 * (polx[1] + sign(polx[1]) * sqrt_disc)
        solutions.extend([q / polx[2], polx[0] / q])
    return solutions


def cubic_solver(polx):
    """Real roots of polx[0] + polx[1]*x + polx[2]*x^2 + polx[3]*x^3.
    Returns list of real roots (1 or 3 entries for a real cubic)."""
    if abs(polx[3]) < TOL:
        return quadratic_solver(polx)  # quadratic only reads polx[0:3]

    # Cardano's method (Numerical Recipes §5.6 convention)
    q = (polx[2]**2 - 3 * polx[1]) / 9
    r = (2 * polx[2]**3 - 9 * polx[1] * polx[2] + 27 * polx[0]) / 54

    solutions = []
    if abs(q) < TOL:
        solutions.append(-polx[2] / 3)
    elif q**3 > r**2:  # Three real roots
        # Clamp to handle floating-point excursions outside acos's domain
        arg = max(-1.0, min(1.0, r / math.sqrt(q**3)))
        theta = math.acos(arg)
        solutions.extend([
            -2 * math.sqrt(q) * math.cos(theta / 3) - polx[2] / 3,
            -2 * math.sqrt(q) * math.cos((theta + 2 * math.pi) / 3) - polx[2] / 3,
            -2 * math.sqrt(q) * math.cos((theta + 4 * math.pi) / 3) - polx[2] / 3,
        ])
    else:  # One real root
        sqrt_term = math.sqrt(r**2 - q**3)
        powthrd = abs(-r + sqrt_term) ** (1.0 / 3.0)
        a = sign(-r + sqrt_term) * powthrd
        b = q / a if abs(a) > TOL else 0
        solutions.append(a + b - polx[2] / 3)
    return solutions


def quartic_solver(polx):
    """Real roots of polx[0] + polx[1]*x + ... + polx[4]*x^4.
    Returns list of real roots (0 to 4 entries)."""
    if abs(polx[4]) < TOL:
        return cubic_solver(polx[:4])

    # Normalize to monic form
    coeffs = [c / polx[4] for c in polx]

    solutions = []
    if abs(coeffs[0]) < TOL:
        # x = 0 is a root; the rest reduce to a cubic in x
        solutions.append(0)
        solutions.extend(cubic_solver(coeffs[1:5]))
        return solutions

    # Depressed quartic substitution: x = y - coeffs[3]/4
    # gives y^4 + e*y^2 + f*y + g = 0
    e = coeffs[2] - 3 * coeffs[3]**2 / 8
    f = coeffs[1] + coeffs[3]**3 / 8 - coeffs[2] * coeffs[3] / 2
    g = coeffs[0] - 3 * coeffs[3]**4 / 256 + coeffs[3]**2 * coeffs[2] / 16 - coeffs[3] * coeffs[1] / 4
    shift = -coeffs[3] / 4

    if abs(g) < TOL:
        # y = 0 is a root of the depressed quartic
        solutions.append(shift)
        # The rest factors out a cubic: y^3 + e*y + f = 0
        solutions.extend([root + shift for root in cubic_solver([f, e, 0, 1])])

    elif abs(f) < TOL:
        # Biquadratic: solve y^2 = z from z^2 + e*z + g = 0
        for z in quadratic_solver([g, e, 1]):
            if z >= 0:
                solutions.append(math.sqrt(z) + shift)
                solutions.append(-math.sqrt(z) + shift)

    else:
        # General case: solve the resolvent cubic for h^2
        # h^6 + 2e*h^4 + (e^2 - 4g)*h^2 - f^2 = 0  =>  cubic in h^2
        resolvent = [-f**2, e**2 - 4 * g, 2 * e, 1]
        for h_squared in cubic_solver(resolvent):
            if h_squared > 0:
                h = math.sqrt(h_squared)
                j = (e + h_squared - f / h) / 2
                # Factor depressed quartic as (y^2 + h*y + j)(y^2 - h*y + g/j)
                solutions.extend(
                    [root + shift for root in quadratic_solver([j, h, 1])]
                )
                solutions.extend(
                    [root + shift for root in quadratic_solver([g / j, -h, 1])]
                )

    return solutions

def algebraic_pz(b, lp, mWp, mt, mb, mlp, pnux, pnuy):
    """
    Computes the z-component for the neutrino using algebraic methods.
    
    Parameters:
        b, lp: Dictionaries with 'E', 'px', 'py', 'pz' as keys.
        mWp, mt, mb, mlp: Masses of W boson, top quark, b-jet, and lepton.
        pnux, pnuy: Neutrino momentum components in the x and y directions.
    
    Returns:
        pnuz: Computed z-component for the neutrino momentum, or None if no real solution.
    """
    epsilon = 1e-6
    # Define some helper functions to evaluate terms
    def evalterm1(a1, pnux, pnuy):
        return a1[0] + a1[1] * pnux + a1[2] * pnuy
    
    def evalterm2(a2, pnux, pnuy):
        return a2[0] + a2[1] * pnux + a2[2] * pnuy + a2[3] * pnux**2 + a2[4] * pnux * pnuy + a2[5] * pnuy**2

    # Kinematic terms
    lpE2_minus_lpz2 = lp["E"]**2 - lp["pz"]**2
    mblp = np.sqrt(np.maximum(0, (b["E"] + lp["E"])**2 - (b["px"] + lp["px"])**2 - (b["py"] + lp["py"])**2 - (b["pz"] + lp["pz"])**2))
    
    # Terms for the neutrino momentum
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

    # Evaluate terms
    a1val = evalterm1(a1, pnux, pnuy)
    a2val = evalterm2(a2, pnux, pnuy)
    b1val = evalterm1(b1, pnux, pnuy)
    b2val = evalterm2(b2, pnux, pnuy)

    # Calculate solutions for pnuz using the radicant method
    pnuz_a = []
    pnuz_b = []

    # First set of solutions
    radicant_a = a1val**2 + a2val
    if radicant_a >= 0:
        pnuz_a.extend([a1val + np.sqrt(radicant_a), a1val - np.sqrt(radicant_a)])
    elif np.abs(radicant_a) < epsilon:
        pnuz_a.append(a1val)

    # Second set of solutions
    radicant_b = b1val**2 + b2val
    if radicant_b >= 0:
        pnuz_b.extend([b1val + np.sqrt(radicant_b), b1val - np.sqrt(radicant_b)])
    elif np.abs(radicant_b) < epsilon:
        pnuz_b.append(b1val)

    # Check for solution pairs
    if len(pnuz_a) == 0 or len(pnuz_b) == 0:
        return None  # No valid solution found

    # Find the closest matching pair between pnuz_a and pnuz_b
    min_diff = float('inf')
    best_pnuz = None
    for a in pnuz_a:
        for b in pnuz_b:
            diff = abs(a - b)
            if diff < min_diff:
                min_diff = diff
                best_pnuz = 0.5 * (a + b)

    if min_diff < np.sqrt(epsilon):
        return best_pnuz
    else:
        return None  

def sqr(x):
    return x**2
    

@njit(
    float64[:](float64, float64, float64, float64, float64, float64, float64, float64, 
               float64, float64, float64, float64, float64, float64, float64, 
               float64, float64, float64, float64, float64, float64, float64,
               float64, float64, float64, float64, float64, float64, float64,
               float64, float64, float64, float64, float64, float64, float64 )
)
def compute_coefficients(
    p, q, sqr_mlp, sqr_mlm, sqr_mb, sqr_mbb, sqr_lp_E, sqr_lp_pz, sqr_lm_E, sqr_lm_pz,
    b_E, lp_E, bb_E, lm_E, lp_dot_b, lp_px, lp_py, lp_pz, a2, a3, a4, lm_dot_bb,
    lm_px, lm_py, lm_pz, b2, b3, b4, dp00, dp10, dp20, c00, c10, c20, ETmiss_pt, ETmiss_phi
):
    # Precompute reusable terms
    sqr_q = q ** 2
    sqr_p = p ** 2
    delta_q_mlp = sqr_q - sqr_mlp
    delta_q_mlm = sqr_q - sqr_mlm

    b_E2 = 2 * b_E
    lp_E2 = 2 * lp_E
    bb_E2 = 2 * bb_E
    lm_E2 = 2 * lm_E
    
    # a1, c22, c21, c11
    a1 = (b_E + lp_E) * delta_q_mlp - lp_E * (sqr_p - sqr_mb - sqr_mlp) + b_E2 * sqr_lp_E - lp_E2 * lp_dot_b
    c22 = (delta_q_mlp * a4) ** 2 - 4 * (sqr_lp_E - sqr_lp_pz) * (a1 ** 2) - 4 * delta_q_mlp * lp_pz * a1 * a4
    c21 = -8 * (sqr_lp_E - sqr_lp_pz) * a1 * a2 + 4 * delta_q_mlp * (lp_px * (a4 ** 2) - lp_pz * a2 * a4) - 8 * lp_px * lp_pz * a1 * a4
    c11 = -8 * (sqr_lp_E - sqr_lp_pz) * a1 * a3 + 4 * delta_q_mlp * (lp_py * (a4 ** 2) - lp_pz * a3 * a4) - 8 * lp_py * lp_pz * a1 * a4

    # b1
    b1 = (bb_E + lm_E) * delta_q_mlm - lm_E * (sqr_p - sqr_mbb - sqr_mlm) + bb_E2 * sqr_lm_E - lm_E2 * lm_dot_bb

    # dp22, dp21, dp11
    dp22 = ((sqr_q - sqr_mlm) * b4 )** 2 - 4 * (sqr_lm_E - sqr_lm_pz) * (b1 ** 2) - 4 * (sqr_q - sqr_mlm) * lm_pz * b1 * b4
    dp21 = -8 * (sqr_lm_E - sqr_lm_pz) * b1 * b2 + 4 * (sqr_q - sqr_mlm) * (lm_px * b4 ** 2 - lm_pz * b2 * b4) - 8 * lm_px * lm_pz * b1 * b4
    dp11 = -8 * (sqr_lm_E - sqr_lm_pz) * b1 * b3 + 4 * (sqr_q - sqr_mlm) * (lm_py * b4 ** 2 - lm_pz * b3 * b4) - 8 * lm_py * lm_pz * b1 * b4

    # d coefficients
    d22 = dp22 + (ETmiss_pt**2) * dp20 + (ETmiss_phi**2) * dp00 + ETmiss_pt * ETmiss_phi * dp10 + ETmiss_pt * dp21 + ETmiss_phi * dp11
    d21 = -dp21 - 2 * ETmiss_pt * dp20 - ETmiss_phi * dp10
    d11 = -dp11 - 2 * ETmiss_phi * dp00 - ETmiss_pt * dp10
    
    # polx coefficients
    polx_0 = c00 ** 2 * d22 ** 2 + c11 * d22 * (c11 * dp00 - c00 * d11) + \
             c00 * c22 * (d11 ** 2 - 2 * dp00 * d22) + c22 * dp00 * (c22 * dp00 - c11 * d11)
    
    polx_1 = c00 * d21 * (2 * c00 * d22 - c11 * d11) + c00 * d11 * (2 * c22 * dp10 + c21 * d11) + \
             c22 * dp00 * (2 * c21 * dp00 - c11 * dp10) - c00 * d22 * (c11 * dp10 + c10 * d11) - \
             2 * c00 * dp00 * (c22 * d21 + c21 * d22) - dp00 * d11 * (c11 * c21 + c10 * c22) + \
             c11 * dp00 * (c11 * d21 + 2 * c10 * d22)
    
    polx_2 = c00 ** 2 * (2 * d22 * dp20 + d21 ** 2) - c00 * d21 * (c11 * dp10 + c10 * d11) + \
             c11 * dp20 * (c11 * dp00 - c00 * d11) + c00 * dp10 * (c22 * dp10 - c10 * d22) + \
             c00 * d11 * (2 * c21 * dp10 + c20 * d11) + dp00 ** 2 * (2 * c22 * c20 + c21 ** 2) - \
             2 * c00 * dp00 * (c22 * dp20 + c21 * d21 + c20 * d22) + c10 * dp00 * (2 * c11 * d21 + c10 * d22) - \
             dp00 * dp10 * (c11 * c21 + c10 * c22) - dp00 * d11 * (c11 * c20 + c10 * c21)
    
    polx_3 = c00 * d21 * (2 * c00 * dp20 - c10 * dp10) - c00 * dp20 * (c11 * dp10 + c10 * d11) + \
             c00 * dp10 * (c21 * dp10 + 2 * c20 * d11) - 2 * c00 * dp00 * (c21 * dp20 + c20 * d21) + \
             c10 * dp00 * (2 * c11 * dp20 + c10 * d21) + c20 * dp00 * (2 * c21 * dp00 - c10 * d11) - \
             dp00 * dp10 * (c11 * c20 + c10 * c21)
    
    polx_4 = c00 ** 2 * dp20 ** 2 + c10 * dp20 * (c10 * dp00 - c00 * dp10) + \
             c20 * dp10 * (c00 * dp10 - c10 * dp00) + c20 * dp00 * (c20 * dp00 - 2 * c00 * dp20)

    # Combine polynomial coefficients
    polx_coeffs = np.array([polx_0, polx_1, polx_2, polx_3, polx_4], dtype=np.float64)

    return np.concatenate((
    np.array([a1, c22, c21, c11, b1, dp22, dp21, dp11, d22, d21, d11, c00, dp00], dtype=np.float64),
    polx_coeffs
    ))


def solve_ttbar_dilepton(events):

  RED = "\033[31m"
  GREEN = "\033[32m"
  YELLOW = "\033[33m"
  RESET = "\033[0m"
  BLUE = "\033[34m" 
  ttbar_masses = []
  ttH_masses = []  

  all_higgs_masses_per_event = []  
  solutions_number = []
  max_weight_per_event = []
  max_weight_higgs_mass_per_event = []
  max_weight_ttH_mass_per_event = []
  max_weight_combination_per_event = []
  max_mean_weight_per_event = []
  max_mean_weight_higgs_mass_per_event = []
  max_mean_weight_combination_per_event = []
  all_event_weights_flat = []
  max_sum_weight_per_event = []
  max_sum_weight_higgs_mass_per_event = []
  max_sum_weight_ttH_mass_per_event = []
  max_mean_weight_ttH_mass_per_event = []
  max_sum_weight_combination_per_event = []
  all_top_masses_per_event = []  
  max_weight_top_mass_per_event = []  
  max_mean_weight_top_mass_per_event = []
  max_sum_weight_top_mass_per_event = []
  all_W_masses_per_event = []  
  max_weight_W_mass_per_event = []  
  max_mean_weight_W_mass_per_event = []
  max_sum_weight_W_mass_per_event = []
  second_max_weight_per_event = []
  second_max_weight_higgs_mass_per_event = []
  second_max_weight_ttH_mass_per_event = []
  second_max_weight_combination_per_event = []
  second_max_weight_top_mass_per_event = []
  second_max_weight_W_mass_per_event = []
  third_max_weight_per_event = []
  third_max_weight_higgs_mass_per_event = []
  third_max_weight_ttH_mass_per_event = []
  third_max_weight_combination_per_event = []
  third_max_weight_top_mass_per_event = []
  third_max_weight_W_mass_per_event = []
  fourth_max_weight_per_event = []
  fourth_max_weight_higgs_mass_per_event = []
  fourth_max_weight_ttH_mass_per_event = []
  fourth_max_weight_top_mass_per_event = []
  fourth_max_weight_W_mass_per_event = []
  fourth_max_weight_combination_per_event = []
  t = len(events)
  ############## Loop over events ###########################
  for event_idx in range(t):
    #print(f"Processing event {event_idx + 1}/{len(events)}")
 
    max_sum_weight = -999
    max_sum_weight_higgs_mass = None
    max_sum_weight_ttH_mass = None
    max_sum_weight_top_mass = None
    max_sum_weight_W_mass = None
    max_sum_weight_combination = None
    max_weight = -999  # Initialize to a low value
    max_weight_higgs_mass = None
    max_weight_ttH_mass = None
    max_weight_combination = None 
    second_max_weight = -999 
    third_max_weight = -999 
    max_higgs_mass = -999
    second_max_weight_higgs_mass = None
    second_max_weight_ttH_mass = None
    second_max_weight_combination = None
    second_max_top_mass = None
    second_max_W_mass = None
    second_max_ttbar_mass = None
    third_max_weight_higgs_mass = None
    third_max_weight_ttH_mass = None
    third_max_weight_combination = None
    third_max_top_mass = None
    third_max_W_mass = None
    third_max_ttbar_mass = None
    max_weight = -999  
    all_valid_solutions = []
    max_weight_combination = None
    max_higgs_mass = -999
    max_ttH_mass = -999
    max_ttbar_mass = -999
    max_top_mass = -999
    max_W_mass = -999
    max_ttbar_mass = -999
    event_max_weight = -999

      # Track jets for max weight
    max_mean_weight = -999
    max_mean_weight_higgs_mass = None
    max_mean_weight_ttH_mass = None
    max_mean_weight_top_mass = None
    max_mean_weight_W_mass = None
    max_mean_weight_combination = None  # Track jets for max mean weight
    all_solution_counts = []
    higgs_masses = []    
    event_weights = []
    weights_by_higgs_jets = {}
    higgs_masses_by_jets = {}
    ttH_masses_by_key = {}
    sum_weights_by_higgs_jets = {}
    mtop_by_jet = {}
    mW_by_jet = {}

    ECM = 13600
    Q = 234


    lp = events["lepton_pos"][event_idx]  # Positive lepton (from W+)
    lm = events["lepton_neg"][event_idx]  # Negative lepton (from W-)
    lp_px = float(lp["px"][0])
    lp_py = float(lp["py"][0])
    lp_pz = float(lp["pz"][0])
    lp_E = float(lp["E"][0])
    lm_px = float(lm["px"][0])
    lm_py = float(lm["py"][0])
    lm_pz = float(lm["pz"][0])
    lm_E = float(lm["E"][0])

    # Calculate squared values
    sqr_lp_E = float(sqr(lp_E))  
    sqr_lp_pz = float(sqr(lp_pz))
    sqr_lp_px = float(sqr(lp_px))
    sqr_lp_py = float(sqr(lp_py))
    sqr_lm_E = float(sqr(lm_E))
    sqr_lm_pz = float(sqr(lm_pz))
    sqr_lm_px = float(sqr(lm_px))
    sqr_lm_py = float(sqr(lm_py))

    # Convert masses
    mlp = float(lp["mass"][0])
    mlm = float(lm["mass"][0])
    sqr_mlp = float(sqr(mlp))
    sqr_mlm = float(sqr(mlm))
    ETmiss = events["MET_few"][event_idx]  # Missing transverse energy (MET)
    ETmiss_pt_1 = ETmiss["pt"]
    ETmiss_phi_1 = ETmiss["phi"]
    ETmiss_pt = ETmiss_pt_1 * np.cos(ETmiss_phi_1)
    ETmiss_phi = ETmiss_pt_1 * np.sin(ETmiss_phi_1)
    sqr_ETmiss_pt = sqr(ETmiss_pt)
    sqr_ETmiss_phi = sqr(ETmiss_phi)
    # print("nBJets", events["BJetGood"]["pt"][event_idx])
    # Define jets array
    jets = [
    events["1st_jet"][event_idx],
    events["2nd_jet"][event_idx],
    events["3rd_jet"][event_idx],
    events["4th_jet"][event_idx]
     ]

    fifth_jet = events["5th_jet"][event_idx]
    if fifth_jet is not None :
        if abs(fifth_jet["pt"]) > 1e-3:
            jets.append(fifth_jet)
   
    n_jets = 4
    ############## Loop over bJets ###########################
    for i in range(n_jets):  # Loop over b-jet (index i)
        for j in range(n_jets):  # Loop over bb-jet (index j)
            if i == j:
               continue  # Skip cases where b and bb are the same jet
            b = jets[i]
            bb = jets[j]
            # Convert Awkward Array fields to scalars
            mb = b["mass"]
            mbb = bb["mass"]
            b_px = b["px"]
            b_py = b["py"]
            b_pz = b["pz"]
            b_E = b["E"]
            bb_px = bb["px"]
            bb_py = bb["py"]
            bb_pz = bb["pz"]
            bb_E = bb["E"]

            # Calculate squared values
            sqr_mb = float(sqr(mb))  # Assuming `sqr` is a function that computes the square
            sqr_mbb = float(sqr(mbb))
            b_E2 = 2 * b_E
            lp_E2 = 2 * lp_E
            bb_E2 = 2 * bb_E
            lm_E2 = 2 * lm_E
            
            a2 = 2 * (b_E * lp_px - lp_E * b_px)
            a3 = 2 * (b_E * lp_py - lp_E * b_py)
            a4 = 2 * (b_E * lp_pz - lp_E * b_pz) 

            b2 = 2 * (bb_E * lm_px - lm_E * bb_px) 
            b3 = 2 * (bb_E * lm_py - lm_E * bb_py) 
            b4 = 2 * (bb_E * lm_pz - lm_E * bb_pz)  
            c20 = -4 * (sqr_lp_E - sqr_lp_px) * sqr(a4) - \
                  4 * (sqr_lp_E - sqr_lp_pz) * sqr(a2) - \
                  8 * lp_px * lp_pz * a2 * a4

            c10 = -8 * (sqr_lp_E - sqr_lp_pz) * a2 * a3 + \
                  8 * lp_px * lp_py * sqr(a4) - \
                  8 * lp_px * lp_pz * a3 * a4 - \
                  8 * lp_py * lp_pz * a2 * a4

            c00 = -4 * (sqr_lp_E - sqr_lp_py) * sqr(a4) - \
                  4 * (sqr_lp_E - sqr_lp_pz) * sqr(a3) - \
                  8 * lp_py * lp_pz * a3 * a4
            
            # Refactored dp20, dp10, dp00
            dp20 = -4 * (sqr_lm_E - sqr_lm_px) * sqr(b4) - \
                   4 * (sqr_lm_E - sqr_lm_pz) * sqr(b2) - \
                   8 * lm_px * lm_pz * b2 * b4

            dp10 = -8 * (sqr_lm_E - sqr_lm_pz) * b2 * b3 + \
                   8 * lm_px * lm_py * sqr(b4) - \
                   8 * lm_px * lm_pz * b3 * b4 - \
                   8 * lm_py * lm_pz * b2 * b4

            dp00 = -4 * (sqr_lm_E - sqr_lm_py) * sqr(b4) - \
                   4 * (sqr_lm_E - sqr_lm_pz) * sqr(b3) - \
                   8 * lm_py * lm_pz * b3 * b4
            lp_dot_b = b_px * lp_px + b_py * lp_py + b_pz * lp_pz
            lm_dot_bb = bb_px * lm_px + bb_py * lm_py + bb_pz * lm_pz
            
            ############## Loop over bJet from Higgs ###########################
            for m in range(n_jets):
              if m == i or m == j:
                continue  # Skip if it's already used as b or bb
              for n in range(n_jets):
                if n == i or n == j or n <= m:
                  continue  # Skip if already used as b, bb, or bh1
                #print(f"Using jet {i+1} as b, jet {j+1} as bb, jet {m+1} as bh1, and jet {n+1} as bh2")

                # bh1 and bh2 are the remaining two jets
                bh1 = jets[m]
                bh2 = jets[n] 
                # Higgs system from bh1 and bh2
                E_higgs = bh1["E"] + bh2["E"]
                Px_higgs = bh1["px"] + bh2["px"]
                Py_higgs = bh1["py"] + bh2["py"]
                Pz_higgs = bh1["pz"] + bh2["pz"]
                higgs_mass = np.sqrt(E_higgs**2 - Px_higgs**2 - Py_higgs**2 - Pz_higgs**2)
                  
                ############## Loop over top & W masses ###########################
                for p in range(150, 201, 5):
                  for q in range(60, 101, 5):
                      if p < q:
                        continue
                      p = float(p)
                      q = float(q)
                      result = compute_coefficients(
                          p, q, sqr_mlp, sqr_mlm, sqr_mb, sqr_mbb, sqr_lp_E, sqr_lp_pz, sqr_lm_E, sqr_lm_pz,
                          b_E, lp_E, bb_E, lm_E, lp_dot_b, lp_px, lp_py, lp_pz, a2, a3, a4, lm_dot_bb,
                          lm_px, lm_py, lm_pz, b2, b3, b4, dp00, dp10, dp20, c00, c10, c20, ETmiss_pt, ETmiss_phi
                         )
                      
      
                      polx_coeffs = result[-5:]
                      c22 = result[1]
                      c21 = result[2]
                      c11 = result[3]
                      b1 = result[4]
                      dp22 = result[5]
                      dp21 = result[6]
                      dp11 = result[7]
                      d22 = result[8]
                      d21 = result[9]
                      d11 = result[10]
                      c00 = result[11]
                      dp00 = result[12]
                      a1 = result[0]
                      # Normalize coefficients before calling quartic_solver
                      if abs(polx_coeffs[-1]) > 1e-10:  # Ensure nonzero leading coefficient
                         polx_coeffs = [p / polx_coeffs[-1] for p in polx_coeffs]  # Normalize
                      # Pass this list to quartic_solver
                      pnuxt = quartic_solver(polx_coeffs)
                      pnuxt = filter_close_solutions(pnuxt, tol=1e-2)  
                      #if len(pnuxt) == 8:
                        # print(f"Unusual case detected in event {event_idx}: {pnuxt}")
                
                      
                      #Iterate over the solutions for neutrino and anti-neutrino momenta
                      pnux_list, pnuy_list, pnuz_list = [], [], []
                      pnubx_list, pnuby_list, pnubz_list = [], [], []
                      num_solutions = 0
                      
                      cd_diff = []
                      c0 = c00
                      d0 = dp00

                      weight = 0.

                   
                      #print(f"Using jet {i+1} as b, jet {j+1} as bb, jet {m+1} as bh1, and jet {n+1} as bh2")
                      #if len(pnuxt) == 0:
                      #print("No solutions found for neutrino/anti-neutrino momenta!")

                      #print(f"{BLUE}solutions number {len(pnuxt)}{RESET}")
                      ############## Loop over neutrino solutions ###########################          
                      for k in range(len(pnuxt)):
                                  #print(f"{YELLOW}Processing solution {k+1}/{len(pnuxt)}{RESET}")
                                  # Calculate the other components (py, pz) for neutrino and anti-neutrino
                                   
                                  num_solutions = 0
                                  thispnux = pnuxt[k]
                                  num_solutions += 1
                                  
                                  
                                  # Initialize the unPhysicalSolution flag
                                  unPhysicalSolution = False

                                  # Calculate neutrino py using the formula
                                  c1 = c10 * thispnux + c11
                                  c2 = c20 * sqr(thispnux) + c21 * thispnux + c22
                                  d1 = dp10 * thispnux + d11
                                  d2 = dp20 * sqr(thispnux) + d21 * thispnux + d22
        
                                  denom = c1 * d0 - c0 * d1
                                  cd_diff.append(denom)
                                  if np.abs(denom) < 1e-6:  # Avoid division by zero
                                      continue
        
                                  thispnuy = (c0 * d2 - c2 * d0) / denom
                                  thispnubx = ETmiss_pt - thispnux
                                  thispnuby = ETmiss_phi - thispnuy
        
                                  # Calculate neutrino and anti-neutrino pz components (handle singularities)
                                  lpbz_diff = lp_E * b_pz - b_E * lp_pz
                                  lmbbz_diff = lm_E * bb_pz - bb_E * lm_pz
                                  if np.abs(lpbz_diff) < 1e-6:  # Handle neutrino pz singularity
                                      thispnuz = algebraic_pz(b, lp, q, p, mb, mlp, thispnux, thispnuy)
                                      if thispnuz is None:
                                          continue
                                  else:
                                      thispnuz = (-a1 - a2 * thispnux - a3 * thispnuy) / a4

                                  if np.abs(lmbbz_diff) < 1e-6:  # Handle anti-neutrino pz singularity
                                      thispnubz = algebraic_pz(bb, lm, q, p, mbb, mlm, thispnubx, thispnuby)
                                      if thispnubz is None:
                                          continue
                                  else:
                                      thispnubz = (-b1 - b2 * thispnubx - b3 * thispnuby) / b4
                      
                                  #print(f"Neutrino momentum (px, py, pz): ({thispnux}, {thispnuy}, {thispnuz})")  
                                        # Loop over remaining jets to assign bh1 and bh2
                                  # Check for NaN or Inf in neutrino and anti-neutrino momenta components
                                  if np.isnan(thispnux) or np.isnan(thispnuy) or np.isnan(thispnuz):
                                      unPhysicalSolution = True
                                  if np.isinf(thispnux) or np.isinf(thispnuy) or np.isinf(thispnuz):
                                      unPhysicalSolution = True
                                  if np.isnan(thispnubx) or np.isnan(thispnuby) or np.isnan(thispnubz):
                                      unPhysicalSolution = True
                                  if np.isinf(thispnubx) or np.isinf(thispnuby) or np.isinf(thispnubz):
                                     unPhysicalSolution = True

                                  # Skip the current pnuxt iteration if it's an unphysical solution
                                  if unPhysicalSolution:
                                      continue

                                  # W+ reconstruction (nnu and mup)
                                  pnu_E = np.sqrt(thispnux**2 + thispnuy**2 + thispnuz**2)
                                  Wp_E = pnu_E + lp_E
                                  Wp_px = thispnux + lp_px
                                  Wp_py = thispnuy + lp_py
                                  Wp_pz = thispnuz + lp_pz

                                  # t reconstruction (bq and Wp)
                                  t_E = Wp_E + b_E
                                  t_px = Wp_px + b_px
                                  t_py = Wp_py + b_py
                                  t_pz = Wp_pz + b_pz

                                  # W- reconstruction (nnub and mum)
                                  pnub_E = np.sqrt(thispnubx**2 + thispnuby**2 + thispnubz**2)
                                  Wm_E = pnub_E + lm_E
                                  Wm_px = thispnubx + lm_px
                                  Wm_py = thispnuby + lm_py
                                  Wm_pz = thispnubz + lm_pz

                                  # tb reconstruction (Wm and bb)
                                  tb_E = Wm_E + bb_E
                                  tb_px = Wm_px + bb_px
                                  tb_py = Wm_py + bb_py
                                  tb_pz = Wm_pz + bb_pz

                                  # ttbar system
                                  E_tt = t_E + tb_E
                                  Px_tt = t_px + tb_px
                                  Py_tt = t_py + tb_py
                                  Pz_tt = t_pz + tb_pz
                                  ttbar_mass = np.sqrt(E_tt**2 - Px_tt**2 - Py_tt**2 - Pz_tt**2)

                                  # ttH system
                                  E_ttH = E_tt + E_higgs
                                  Px_ttH = Px_tt + Px_higgs
                                  Py_ttH = Py_tt + Py_higgs
                                  Pz_ttH = Pz_tt + Pz_higgs
                                  ttH_mass = np.sqrt(E_ttH**2 - Px_ttH**2 - Py_ttH**2 - Pz_ttH**2)
                           
                                  if E_ttH > ECM:
                                      continue

                                  higgs_masses.append(higgs_mass)

                                  # Skip the current solution if unphysical
                                  if unPhysicalSolution:    
                                      print ("found one")
                                      continue  
                        
                        
                                  # x1 and x2 calculation for PDF weights
                                  x1 = (E_ttH + Pz_ttH) / ECM
                                  x2 = (E_ttH - Pz_ttH) / ECM
                                  #print(x1)

                                  # Check if x1 and x2 are within valid range
                                  if (x1 < 1.) and (x2 < 1.):



                                      # Call the LHAPDF function with the scalar values
                                      w1 = _pdf_set.xfxQ(21, x1, Q)
                                      w2 = _pdf_set.xfxQ(21, x2, Q)
                                      #w1 = 1
                                      #w2 = 1
                                      # Combine the weights
                                      weight = (w1 * w2) / (x1 * x2)

                                      
                                      
                                      key = (p, q, i, j, m , n)  # Index of the Higgs jets (bh1, bh2)
                                      if key not in weights_by_higgs_jets:
                                          weights_by_higgs_jets[key] = []
                                          higgs_masses_by_jets[key] = higgs_mass
                                          ttH_masses_by_key[key] = ttH_mass
                                          mtop_by_jet[key] = p
                                          mW_by_jet[key] = q

                                      
                                      weights_by_higgs_jets[key].append(weight)
                                      all_valid_solutions.append({"weight": weight,
                                          "combination": (i, j, m, n),
                                          "higgs_pair": set((int(m), int(n))),
                                          "higgs_mass": higgs_mass,
                                          "ttbar_mass": ttbar_mass,
                                          "ttH_mass": ttH_mass,
                                          "top_mass": p,
                                          "W_mass": q,
                                      })

                                   
                                  else:                            
                                      weight = 0.  # Assign zero if unphysical
                      event_weights.append(weight)
                  all_solution_counts.append(len(pnuxt))
    max_mean_weight = -999 
    #print ("mass reco", max_weight)
    # Sort all valid solutions by descending weight
    sorted_solutions = sorted(all_valid_solutions, key=lambda s: s["weight"], reverse=True)

    # Initialize
    best = second = third = fourth = None
    used_higgs_pairs = set()

    for sol in sorted_solutions:
        hp = frozenset(sol["higgs_pair"])
    
        if best is None:
            best = sol
            used_higgs_pairs.add(hp)
        elif second is None and hp not in used_higgs_pairs:
            second = sol
            used_higgs_pairs.add(hp)
        elif third is None and hp not in used_higgs_pairs:
            third = sol
            used_higgs_pairs.add(hp)
        elif fourth is None and hp not in used_higgs_pairs:
            fourth = sol
            used_higgs_pairs.add(hp)

        if fourth:  # all 4 found
            break
      
    if best:
        max_weight_per_event.append(best["weight"])
        max_weight_combination_per_event.append(best["combination"])
        max_weight_higgs_mass_per_event.append(best["higgs_mass"])
        max_weight_ttH_mass_per_event.append(best["ttH_mass"])
        max_weight_top_mass_per_event.append(best["top_mass"])
        max_weight_W_mass_per_event.append(best["W_mass"])
    else:
        max_weight_per_event.append(0)
        max_weight_combination_per_event.append(None)
        max_weight_higgs_mass_per_event.append(None)
        max_weight_ttH_mass_per_event.append(None)
        max_weight_top_mass_per_event.append(None)
        max_weight_W_mass_per_event.append(None)

    # Second
    if second:
        second_max_weight_per_event.append(second["weight"])
        second_max_weight_combination_per_event.append(second["combination"])
        second_max_weight_higgs_mass_per_event.append(second["higgs_mass"])
        second_max_weight_ttH_mass_per_event.append(second["ttH_mass"])
        second_max_weight_top_mass_per_event.append(second["top_mass"])
        second_max_weight_W_mass_per_event.append(second["W_mass"])
    else:
        second_max_weight_per_event.append(0)
        second_max_weight_combination_per_event.append(None)
        second_max_weight_higgs_mass_per_event.append(None)
        second_max_weight_ttH_mass_per_event.append(None)
        second_max_weight_top_mass_per_event.append(None)
        second_max_weight_W_mass_per_event.append(None)

    # Third
    if third:
        third_max_weight_per_event.append(third["weight"])
        third_max_weight_combination_per_event.append(third["combination"])
        third_max_weight_higgs_mass_per_event.append(third["higgs_mass"])
        third_max_weight_ttH_mass_per_event.append(third["ttH_mass"])
        third_max_weight_top_mass_per_event.append(third["top_mass"])
        third_max_weight_W_mass_per_event.append(third["W_mass"])
    else:
        third_max_weight_per_event.append(0)
        third_max_weight_combination_per_event.append(None)
        third_max_weight_higgs_mass_per_event.append(None)
        third_max_weight_ttH_mass_per_event.append(None)
        third_max_weight_top_mass_per_event.append(None)
        third_max_weight_W_mass_per_event.append(None)


    if fourth:
        fourth_max_weight_per_event.append(fourth["weight"])
        fourth_max_weight_combination_per_event.append(fourth["combination"])
        fourth_max_weight_higgs_mass_per_event.append(fourth["higgs_mass"])
        fourth_max_weight_ttH_mass_per_event.append(fourth["ttH_mass"])
        fourth_max_weight_top_mass_per_event.append(fourth["top_mass"])
        fourth_max_weight_W_mass_per_event.append(fourth["W_mass"])
    else:
        fourth_max_weight_per_event.append(0)
        fourth_max_weight_combination_per_event.append(None)
        fourth_max_weight_higgs_mass_per_event.append(None)
        fourth_max_weight_ttH_mass_per_event.append(None)
        fourth_max_weight_top_mass_per_event.append(None)
        fourth_max_weight_W_mass_per_event.append(None)

    #print(f"\n--- Event {event_idx} ---")
    #print(f"[Post-loop] best_combination        : {best['combination'] if best else None}")
    #print(f"[Post-loop] second_combination      : {second['combination'] if second else None}")
    #print(f"[Post-loop] third_combination       : {third['combination'] if third else None}")
    #print(f"[Post-loop] best_weight             : {best['weight'] if best else None}")
    #print(f"[Post-loop] second_weight           : {second['weight'] if second else None}")
    #print(f"[Post-loop] third_weight            : {third['weight'] if third else None}")
      
    for key, weights in weights_by_higgs_jets.items():
        mean_weight = np.mean(weights)  # Calculate mean weight for this jet assignment
        sum_weight = np.sum(weights)
        sum_weights_by_higgs_jets[key] = sum_weight  # Store sum for each combination
        if mean_weight > max_mean_weight:
           max_mean_weight = mean_weight
           max_mean_weight_higgs_mass = higgs_masses_by_jets[key]
           max_mean_weight_ttH_mass = ttH_masses_by_key[key]
           max_mean_weight_top_mass = mtop_by_jet[key]
           max_mean_weight_W_mass = mW_by_jet[key]
           max_mean_weight_combination = key  # Store the best jet assignment
           
        if sum_weight > max_sum_weight:
           max_sum_weight = sum_weight
           max_sum_weight_higgs_mass = higgs_masses_by_jets[key]
           max_sum_weight_ttH_mass = ttH_masses_by_key[key]
           max_sum_weight_top_mass = mtop_by_jet[key]
           max_sum_weight_W_mass = mW_by_jet[key]
           max_sum_weight_combination = key

    


    max_mean_weight_per_event.append(max_mean_weight)
    max_mean_weight_higgs_mass_per_event.append(max_mean_weight_higgs_mass)
    max_mean_weight_ttH_mass_per_event.append(max_mean_weight_ttH_mass)
    max_mean_weight_top_mass_per_event.append(max_mean_weight_top_mass)
    max_mean_weight_W_mass_per_event.append(max_mean_weight_W_mass)
    max_mean_weight_combination_per_event.append(max_mean_weight_combination)
    all_higgs_masses_per_event.append(higgs_masses)
    solutions_number.append(all_solution_counts)
    event_weights_flat = ak.flatten(event_weights, axis=None).tolist()  
    all_event_weights_flat.append(event_weights_flat)
    max_sum_weight_per_event.append(max_sum_weight)
    max_sum_weight_higgs_mass_per_event.append(max_sum_weight_higgs_mass)
    max_sum_weight_ttH_mass_per_event.append(max_sum_weight_ttH_mass)
    max_sum_weight_top_mass_per_event.append(max_sum_weight_top_mass)
    max_sum_weight_W_mass_per_event.append(max_sum_weight_W_mass)
    max_sum_weight_combination_per_event.append(max_sum_weight_combination)

  results = {
        "all_higgs_masses_per_event": all_higgs_masses_per_event,
        "solutions_number": solutions_number, 
        "all_weights_per_event": all_event_weights_flat,
        "max_weight_per_event": max_weight_per_event,
        "max_weight_higgs_mass_per_event": max_weight_higgs_mass_per_event,
        "max_weight_ttH_mass_per_event": max_weight_ttH_mass_per_event,
        "max_weight_top_mass_per_event": max_weight_top_mass_per_event,
        "max_weight_W_mass_per_event": max_weight_W_mass_per_event,
        "max_weight_combination_per_event": max_weight_combination_per_event,
        "second_max_weight_per_event": second_max_weight_per_event,
        "second_max_weight_higgs_mass_per_event": second_max_weight_higgs_mass_per_event,
        "second_max_weight_ttH_mass_per_event": second_max_weight_ttH_mass_per_event,
        "second_max_weight_top_mass_per_event": second_max_weight_top_mass_per_event,
        "second_max_weight_W_mass_per_event": second_max_weight_W_mass_per_event,
        "second_max_weight_combination_per_event": second_max_weight_combination_per_event,
        "third_max_weight_per_event": third_max_weight_per_event,
        "third_max_weight_higgs_mass_per_event": third_max_weight_higgs_mass_per_event,
        "third_max_weight_ttH_mass_per_event": third_max_weight_ttH_mass_per_event,
        "third_max_weight_top_mass_per_event": third_max_weight_top_mass_per_event,
        "third_max_weight_W_mass_per_event": third_max_weight_W_mass_per_event,
        "third_max_weight_combination_per_event": third_max_weight_combination_per_event,
        "fourth_max_weight_per_event": fourth_max_weight_per_event,
        "fourth_max_weight_higgs_mass_per_event": fourth_max_weight_higgs_mass_per_event,
        "fourth_max_weight_ttH_mass_per_event": fourth_max_weight_ttH_mass_per_event,
        "fourth_max_weight_top_mass_per_event": fourth_max_weight_top_mass_per_event,
        "fourth_max_weight_W_mass_per_event": fourth_max_weight_W_mass_per_event,
        "fourth_max_weight_combination_per_event": fourth_max_weight_combination_per_event,
        "max_mean_weight_per_event": max_mean_weight_per_event,
        "max_mean_weight_higgs_mass_per_event": max_mean_weight_higgs_mass_per_event,
        "max_mean_weight_ttH_mass_per_event": max_mean_weight_ttH_mass_per_event,
        "max_mean_weight_top_mass_per_event": max_mean_weight_top_mass_per_event,
        "max_mean_weight_W_mass_per_event": max_mean_weight_W_mass_per_event,
        "max_mean_weight_combination_per_event": max_mean_weight_combination_per_event,
        "max_sum_weight_per_event": max_sum_weight_per_event,
        "max_sum_weight_higgs_mass_per_event": max_sum_weight_higgs_mass_per_event,
        "max_sum_weight_ttH_mass_per_event": max_sum_weight_ttH_mass_per_event,
        "max_sum_weight_top_mass_per_event": max_sum_weight_top_mass_per_event,
        "max_sum_weight_W_mass_per_event": max_sum_weight_W_mass_per_event,
        "max_sum_weight_combination_per_event": max_sum_weight_combination_per_event
    }


  return results

    
@njit
def calculate_pt_asymmetry(pt1, pt2):
    """
    Calculate the pT asymmetry between two particles.

    Parameters:
    pt1, pt2 : float
        Transverse momenta of the two particles.

    Returns:
    float
        The pT asymmetry.
    """
    denominator = pt1 + pt2
    if denominator == 0:
        return 0.0  # or np.nan or -1 depending on your convention
    return np.abs(pt1 - pt2) / denominator    

@njit
def compute_jet_pair_properties_for_event(pt, eta, phi, mass, higgs_eta, higgs_phi, higgs_mass):
    """
    Compute jet pair properties for a single event.

    Parameters:
    - pt, eta, phi, mass: Arrays of jet properties for a single event.

    Returns:
    - phi_pairs, eta_pairs, mass_pairs, pt_pairs, indices: Pair properties and indices.
    """
    n_jets = len(pt)
    phi_pairs = []
    eta_pairs = []
    mass_pairs = []
    pt_pairs = []
    delta_r_pairs = []
    indices = []
    # Initialize variables to track the minimum ΔR
    # Initialize variables to track the minimum ΔR
    min_dr = 999.0  # Start with infinity
    min_phi_pair = -999.0
    min_eta_pair = -999.0
    min_mass_pair = -999.0
    min_pt_pair = -999.0
    min_indices = (-1, -1)  # Initialize with invalid indices


    for j in range(n_jets):
        for k in range(j + 1, n_jets):  # Avoid duplicate pairs
            # Jet 4-momenta for pair (j, k)
            px1 = pt[j] * np.cos(phi[j])
            py1 = pt[j] * np.sin(phi[j])
            pz1 = pt[j] * np.sinh(eta[j])
            e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + mass[j]**2)

            px2 = pt[k] * np.cos(phi[k])
            py2 = pt[k] * np.sin(phi[k])
            pz2 = pt[k] * np.sinh(eta[k])
            e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + mass[k]**2)

            # Sum components
            px = px1 + px2
            py = py1 + py2
            pz = pz1 + pz2
            energy = e1 + e2

            # Compute pair properties
            pt_pair = np.sqrt(px**2 + py**2)
            mass_pair = np.sqrt(max(energy**2 - px**2 - py**2 - pz**2, 0))
            phi_pair = np.arctan2(py, px)
            #eta_pair = np.arcsinh(pz/pt_pair)
            eta_pair = 0.5 * np.log((energy + pz) / (energy - pz))
            '''
            pt_pairs.append(pt_pair)
            phi_pairs.append(phi_pair)
            eta_pairs.append(eta_pair)
            indices.append((j, k))  # Store the jet pair indices
            '''
            dphi = delta_phi(phi_pair, higgs_phi)
            
            deta = eta_pair - higgs_eta
            # Compute ΔR for the pair
            dr = np.sqrt(deta**2 + dphi**2)
            #print("DR", dr)
            # Store results for this pair
                        # Check if this pair has the smallest ΔR
            if dr < min_dr and dr < 0.4:
                min_dr = dr
                min_phi_pair = phi_pair
                min_eta_pair = eta_pair
                min_mass_pair = mass_pair
                min_pt_pair = pt_pair
                min_indices = (j, k)

    
    
    pt_pairs.append(min_pt_pair)
    phi_pairs.append(min_phi_pair)
    eta_pairs.append(min_eta_pair)
    indices.append(min_indices)  # Store the jet pair indices           
    delta_r_pairs.append(min_dr)
    mass_pairs.append(min_mass_pair)
            

    return phi_pairs, eta_pairs, mass_pairs, pt_pairs, delta_r_pairs, indices

def dr_criterion_for_max_weight(events):
    t = len(events)
    massreco_chosen_pair_higgs_mass_1 = []
    massreco_chosen_pair_top_mass_1 = []
    massreco_chosen_pair_W_mass_1 = []
    massreco_chosen_pair_ttH_mass_1 = []
    massreco_chosen_pair_combination_1 = []

    for event_idx in range(t):
        jet_eta = events["JetGood"]["eta"][event_idx]
        jet_phi = events["JetGood"]["phi"][event_idx]

        dr_1 = dr_2 = dr_3 = dr_4 = None
        #hj1 = hj2 = hj1_2 = hj2_2 = hj1_3 = hj2_3 = -1

        max_comb = events["max_weight_combination"][event_idx]
        second_comb = events["second_max_weight_combination"][event_idx]
        third_comb = events["third_max_weight_combination"][event_idx]
        fourth_comb = events["fourth_max_weight_combination"][event_idx]

        if max_comb is not None:
            hj1 = max_comb["2"]
            hj2 = max_comb["3"]
            original_hj1 = get_original_jet_idx(events, event_idx, hj1)
            original_hj2 = get_original_jet_idx(events, event_idx, hj2)
            if original_hj1 < len(jet_eta) and original_hj2 < len(jet_eta):
                dr_1 = calculate_dr(jet_eta[original_hj1], jet_phi[original_hj1], jet_eta[original_hj2], jet_phi[original_hj2])
                #print(f"Event {event_idx} - Max Weight Combination: {max_comb}, Max Weight: {events['max_weight'][event_idx]}, ΔR: {dr_1}")

        if second_comb is not None:
            hj1_2 = second_comb["2"]
            hj2_2 = second_comb["3"]
            original_hj1_2 = get_original_jet_idx(events, event_idx, hj1_2)
            original_hj2_2 = get_original_jet_idx(events, event_idx, hj2_2)
            if original_hj1_2 < len(jet_eta) and original_hj2_2 < len(jet_eta):
                dr_2 = calculate_dr(jet_eta[original_hj1_2], jet_phi[original_hj1_2], jet_eta[original_hj2_2], jet_phi[original_hj2_2])
                #print(f"Event {event_idx} - 2nd Max Weight Combination: {second_comb}, 2nd Max Weight: {events['second_max_weight'][event_idx]}, ΔR: {dr_2}")

        if third_comb is not None:
            hj1_3 = third_comb["2"]
            hj2_3 = third_comb["3"]
            original_hj1_3 = get_original_jet_idx(events, event_idx, hj1_3)
            original_hj2_3 = get_original_jet_idx(events, event_idx, hj2_3)
            if original_hj1_3 < len(jet_eta) and original_hj2_3 < len(jet_eta):
                dr_3 = calculate_dr(jet_eta[original_hj1_3], jet_phi[original_hj1_3], jet_eta[original_hj2_3], jet_phi[original_hj2_3])
                #print(f"Event {event_idx} - 3rd Max Weight Combination: {third_comb}, 3rd Max Weight: {events['third_max_weight'][event_idx]}, ΔR: {dr_3}")

        if fourth_comb is not None:
            hj1_4 = fourth_comb["2"]
            hj2_4 = fourth_comb["3"]
            original_hj1_4 = get_original_jet_idx(events, event_idx, hj1_4)
            original_hj2_4 = get_original_jet_idx(events, event_idx, hj2_4)
            if original_hj1_4 < len(jet_eta) and original_hj2_4 < len(jet_eta):
                dr_4 = calculate_dr(jet_eta[original_hj1_4], jet_phi[original_hj1_4], jet_eta[original_hj2_4], jet_phi[original_hj2_4])
                #print(f"Event {event_idx} - 4th Max Weight Combination: {fourth_comb}, 3rd Max Weight: {events['fourth_max_weight'][event_idx]}, ΔR: {dr_4}")

        k = 1.9
        if dr_1 is not None and dr_1 < k:
            comb_key = "max_weight"
        elif dr_2 is not None and dr_2 < k:
            comb_key = "second_max_weight"
        elif dr_3 is not None and dr_3 < k:
            comb_key = "third_max_weight"
        elif dr_4 is not None and dr_4 < k:
            comb_key = "fourth_max_weight"
        else:
            comb_key = "max_weight"  # Default
            
        #print(f"Event {event_idx} - 4th Max Weight Combination: {comb_key}, 3rd Max Weight: { events[f'{comb_key}'][event_idx] }, ΔR1: {dr_1}, ΔR2: {dr_2}, ΔR3: {dr_3}, ΔR4: {dr_4}")
        massreco_chosen_pair_higgs_mass_1.append(events[f"{comb_key}_higgs_mass"][event_idx])
        massreco_chosen_pair_top_mass_1.append(events[f"{comb_key}_top_mass"][event_idx])
        massreco_chosen_pair_W_mass_1.append(events[f"{comb_key}_W_mass"][event_idx])
        massreco_chosen_pair_ttH_mass_1.append(events[f"{comb_key}_ttH_mass"][event_idx])
        massreco_chosen_pair_combination_1.append(events[f"{comb_key}_combination"][event_idx])

    res = {
        "massreco_chosen_pair_higgs_mass": massreco_chosen_pair_higgs_mass_1,
        "massreco_chosen_pair_top_mass": massreco_chosen_pair_top_mass_1,
        "massreco_chosen_pair_W_mass": massreco_chosen_pair_W_mass_1,
        "massreco_chosen_pair_ttH_mass": massreco_chosen_pair_ttH_mass_1,
        "massreco_chosen_pair_combination": massreco_chosen_pair_combination_1
    }
    
    return res

def get_original_jet_idx(events, event_idx, hj):
    if hj == 0:
        return ak.to_numpy(events["1st_jet_idx"])[event_idx]
    elif hj == 1:
        return ak.to_numpy(events["2nd_jet_idx"])[event_idx]
    elif hj == 2:
        return ak.to_numpy(events["3rd_jet_idx"])[event_idx]
    elif hj == 3:
        return ak.to_numpy(events["4th_jet_idx"])[event_idx]
    else:
        return None

def skip_event():
    pair_phi.append([])
    pair_eta.append([])
    pair_phi.append([])
    pair_eta.append([])
    pair_mass.append([])
    pair_pt.append([])
    pair_indices.append([])
    pair_dr.append([])
    numerators.append([])
    denominators.append([])
    correct_matches.append([])
    correct_agreements.append([])
    firstandsecondmaxweight_higgs_mass_1.append([])
    massreco_chosen_pair_correct_higgs_mass_1.append([])
    massreco_chosen_pair_all_higgs_mass_1.append([])
    massreco_chosen_pair_wrong_higgs_mass_1.append([])
    massreco_chosen_pair_wrong_higgs_mass_2_jets_1.append([])
    massreco_chosen_pair_wrong_higgs_mass_1_jets_1.append([])
    massreco_chosen_pair_correct_top_mass_1.append([])
    massreco_chosen_pair_wrong_top_mass_1.append([])
    massreco_chosen_pair_correct_W_mass_1.append([])
    massreco_chosen_pair_wrong_W_mass_1.append([])
    massreco_chosen_pair_correct_ttH_mass_1.append([])
    massreco_chosen_pair_wrong_ttH_mass_1.append([])
    massreco_chosen_pair_correct_jet_pt_1.append([])
    massreco_chosen_pair_wrong_jet_pt_1.append([])
    massreco_chosen_pair_correct_jet_phi_1.append([])
    massreco_chosen_pair_wrong_jet_phi_1.append([])
    massreco_chosen_pair_correct_jet_eta_1.append([])
    massreco_chosen_pair_wrong_jet_eta_1.append([])
    massreco_chosen_pair_correct_jet_mult_1.append([])
    massreco_chosen_pair_wrong_jet_mult_1.append([])
    massreco_chosen_pair_wrong_dr_1.append([])
    massreco_chosen_pair_correct_dr_1.append([])
    massreco_chosen_pair_wrong_delta_angle_1.append([])
    massreco_chosen_pair_correct_delta_angle_1.append([])
    massreco_chosen_pair_wrong_dr_top_1.append([])
    massreco_chosen_pair_correct_dr_top_1.append([]) 
    massreco_chosen_pair_wrong_lep_neg_1.append([]) 
    massreco_chosen_pair_correct_lep_neg_1.append([])     
    massreco_chosen_pair_wrong_lep_pos_1.append([]) 
    massreco_chosen_pair_correct_lep_pos_1.append([]) 
    massreco_chosen_pair_wrong_pt_assymetry_1.append([])
    massreco_chosen_pair_correct_pt_assymetry_1.append([])
    massreco_chosen_pair_correct_jet_btag_score_1.append([None, None])
    massreco_chosen_pair_wrong_jet_btag_score_1.append([None, None])



def compute_jet_pair_properties_all_events(events):
    massreco_chosen_pair_dr_1 = []
    massreco_chosen_pair_delta_angle_1 = []
    pull_magnitude_list = []
    pull_angle_list = []
    pull_magnitude_diff_list = []
    pull_angle_diff_list = []

    t = len(events)
    prefix = "max_weight"

    for event_idx in range(t):
        dr = None
        delta_angle = None
        pull_magnitude = [None, None]
        pull_angle = [None, None]
        pull_magnitude_diff = None
        pull_angle_diff = None

        comb = events[f"{prefix}_combination"][event_idx]
        if comb is None:
            massreco_chosen_pair_dr_1.append([None])
            massreco_chosen_pair_delta_angle_1.append([None])
            pull_magnitude_list.append(pull_magnitude)
            pull_angle_list.append(pull_angle)
            pull_magnitude_diff_list.append(pull_magnitude_diff)
            pull_angle_diff_list.append(pull_angle_diff)
            continue

        hj1 = comb["2"]
        hj2 = comb["3"]

        original_hj1 = get_original_jet_idx(events, event_idx, hj1)
        original_hj2 = get_original_jet_idx(events, event_idx, hj2)

        if original_hj1 < len(events["JetGood"]["eta"][event_idx]) and original_hj2 < len(events["JetGood"]["eta"][event_idx]):
            eta1 = events["JetGood"]["eta"][event_idx][original_hj1]
            phi1 = events["JetGood"]["phi"][event_idx][original_hj1]
            eta2 = events["JetGood"]["eta"][event_idx][original_hj2]
            phi2 = events["JetGood"]["phi"][event_idx][original_hj2]

            pull_angle1 = events["JetGood"]["pull_angle"][event_idx][original_hj1]
            pull_angle2 = events["JetGood"]["pull_angle"][event_idx][original_hj2]

            pull_magnitude1 = events["JetGood"]["pull_magnitude"][event_idx][original_hj1]
            pull_magnitude2 = events["JetGood"]["pull_magnitude"][event_idx][original_hj2]

            dr = calculate_dr(eta1, phi1, eta2, phi2)
            delta_angle = pull_angle1 - pull_angle2
            pull_magnitude_diff = pull_magnitude1 - pull_magnitude2
            pull_angle_diff = pull_angle1 - pull_angle2

            pull_magnitude = [pull_magnitude1, pull_magnitude2]
            pull_angle = [pull_angle1, pull_angle2]

        massreco_chosen_pair_dr_1.append([dr])
        massreco_chosen_pair_delta_angle_1.append([delta_angle])
        pull_magnitude_list.append(pull_magnitude)
        pull_angle_list.append(pull_angle)
        pull_magnitude_diff_list.append(pull_magnitude_diff)
        pull_angle_diff_list.append(pull_angle_diff)

    return {
        "massreco_chosen_pair_dr": massreco_chosen_pair_dr_1,
        "massreco_chosen_pair_delta_angle": massreco_chosen_pair_delta_angle_1,
        "pull_magnitude": pull_magnitude_list,
        "pull_angle": pull_angle_list,
        "pull_magnitude_diff": pull_magnitude_diff_list,
        "pull_angle_diff": pull_angle_diff_list,
    }
