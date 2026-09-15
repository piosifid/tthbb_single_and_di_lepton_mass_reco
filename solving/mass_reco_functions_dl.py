import math
import numpy as np
import awkward as ak
from numba import njit, float64
from parton import mkPDF
import vector
vector.register_awkward()

_pdf_set = mkPDF("CT10", 0, pdfdir=".")

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
                break  # any one valid h reconstructs all 4 roots of the quartic --
                       # using more than one just re-derives the same roots with
                       # extra floating-point noise, producing spurious "5th/6th/
                       # 7th/8th root" artifacts the dedup tolerance can miss
    return solutions

def count_quartic_real_roots(polx):
    """Like quartic_solver, but returns the REAL root count WITH MULTIPLICITY
    (always 0, 2, or 4 for a true quartic) instead of the list of distinct
    values. A double root counts as 2, not 1 -- physically correct, unlike
    len(quartic_solver(polx)), which dedups a double root's two coincident
    roots down to a single distinct value (so it can report 1 or 3)."""
    if abs(polx[4]) < TOL:
        return len(cubic_solver(polx[:4]))
    coeffs = [c / polx[4] for c in polx]
    if abs(coeffs[0]) < TOL:
        return 1 + len(cubic_solver(coeffs[1:5]))
    e = coeffs[2] - 3 * coeffs[3]**2 / 8
    f = coeffs[1] + coeffs[3]**3 / 8 - coeffs[2] * coeffs[3] / 2
    g = coeffs[0] - 3 * coeffs[3]**4 / 256 + coeffs[3]**2 * coeffs[2] / 16 - coeffs[3] * coeffs[1] / 4
    if abs(g) < TOL:
        return 1 + len(cubic_solver([f, e, 0, 1]))
    elif abs(f) < TOL:
        count = 0
        for z in quadratic_solver([g, e, 1]):
            if z >= 0:
                count += 2
        return count
    else:
        resolvent = [-f**2, e**2 - 4 * g, 2 * e, 1]
        for h_squared in cubic_solver(resolvent):
            if h_squared > 0:
                h = math.sqrt(h_squared)
                j = (e + h_squared - f / h) / 2
                d1 = h**2 - 4 * j
                d2 = h**2 - 4 * (g / j)
                return (2 if d1 > -TOL else 0) + (2 if d2 > -TOL else 0)
        return 0
 
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

def dr_criterion_for_max_weight(events, dr_threshold=1.9):
    """
    For each event, selects the best mass-reco Higgs pair combination by applying
    a DR cut on the Higgs jet pair. Iterates through combinations ranked by PDF
    weight (max -> second -> third -> fourth) and picks the first with DR < dr_threshold.
    If none pass the DR cut, falls back to max_weight (trusting the PDF weight).
    """
    t = len(events)

    # --- Original outputs ---
    massreco_chosen_pair_higgs_mass_1  = []
    massreco_chosen_pair_top_mass_1    = []
    massreco_chosen_pair_W_mass_1      = []
    massreco_chosen_pair_ttH_mass_1    = []
    massreco_chosen_pair_combination_1 = []

    # --- All-events per-rank DR and mass (regardless of which rank was selected) ---
    rank_used_1        = []
    dr_rank1_1         = []
    dr_rank2_1         = []
    dr_rank3_1         = []
    dr_rank4_1         = []
    dr_chosen_1        = []
    higgs_mass_rank1_1 = []
    higgs_mass_rank2_1 = []
    higgs_mass_rank3_1 = []
    higgs_mass_rank4_1 = []
    higgs_mass_chosen_1 = []

    # --- Selected: filled only when that rank was the final choice ---
    higgs_mass_chosen_from_rank1_1    = []
    higgs_mass_chosen_from_rank2_1    = []
    higgs_mass_chosen_from_rank3_1    = []
    higgs_mass_chosen_from_rank4_1    = []
    higgs_mass_chosen_from_fallback_1 = []
    dr_chosen_from_rank1_1            = []
    dr_chosen_from_rank2_1            = []
    dr_chosen_from_rank3_1            = []
    dr_chosen_from_rank4_1            = []
    dr_chosen_from_fallback_1         = []

    # --- Rejected: filled only when that rank was skipped in favour of a lower rank ---
    # rank1 rejected = rank was promoted to 2, 3, or 4
    # rank2 rejected = rank was promoted to 3 or 4
    # rank3 rejected = rank was promoted to 4
    # rank4 rejected = fallback (rank4 tried but also failed DR cut -> back to rank1)
    higgs_mass_rank1_rejected_1 = []
    higgs_mass_rank2_rejected_1 = []
    higgs_mass_rank3_rejected_1 = []
    higgs_mass_rank4_rejected_1 = []
    dr_rank1_rejected_1         = []
    dr_rank2_rejected_1         = []
    dr_rank3_rejected_1         = []
    dr_rank4_rejected_1         = []

    n_fallback  = 0
    n_promoted  = 0
    n_no_reco   = 0
    rank_counts = {1: 0, 2: 0, 3: 0, 4: 0, 0: 0}

    for event_idx in range(t):
        jet_eta = events["JetGood"]["eta"][event_idx]
        jet_phi = events["JetGood"]["phi"][event_idx]

        max_comb    = events["max_weight_combination"][event_idx]
        second_comb = events["second_max_weight_combination"][event_idx]
        third_comb  = events["third_max_weight_combination"][event_idx]
        fourth_comb = events["fourth_max_weight_combination"][event_idx]

        if max_comb is None:
            n_no_reco += 1
            massreco_chosen_pair_higgs_mass_1.append(None)
            massreco_chosen_pair_top_mass_1.append(None)
            massreco_chosen_pair_W_mass_1.append(None)
            massreco_chosen_pair_ttH_mass_1.append(None)
            massreco_chosen_pair_combination_1.append(None)
            rank_used_1.append(None)
            dr_rank1_1.append(None)
            dr_rank2_1.append(None)
            dr_rank3_1.append(None)
            dr_rank4_1.append(None)
            dr_chosen_1.append(None)
            higgs_mass_rank1_1.append(None)
            higgs_mass_rank2_1.append(None)
            higgs_mass_rank3_1.append(None)
            higgs_mass_rank4_1.append(None)
            higgs_mass_chosen_1.append(None)
            higgs_mass_chosen_from_rank1_1.append(None)
            higgs_mass_chosen_from_rank2_1.append(None)
            higgs_mass_chosen_from_rank3_1.append(None)
            higgs_mass_chosen_from_rank4_1.append(None)
            higgs_mass_chosen_from_fallback_1.append(None)
            dr_chosen_from_rank1_1.append(None)
            dr_chosen_from_rank2_1.append(None)
            dr_chosen_from_rank3_1.append(None)
            dr_chosen_from_rank4_1.append(None)
            dr_chosen_from_fallback_1.append(None)
            higgs_mass_rank1_rejected_1.append(None)
            higgs_mass_rank2_rejected_1.append(None)
            higgs_mass_rank3_rejected_1.append(None)
            higgs_mass_rank4_rejected_1.append(None)
            dr_rank1_rejected_1.append(None)
            dr_rank2_rejected_1.append(None)
            dr_rank3_rejected_1.append(None)
            dr_rank4_rejected_1.append(None)
            continue

        def get_dr(comb):
            if comb is None:
                return None
            hj1 = get_original_jet_idx(events, event_idx, comb["2"])
            hj2 = get_original_jet_idx(events, event_idx, comb["3"])
            if hj1 is None or hj2 is None:
                return None
            if hj1 >= len(jet_eta) or hj2 >= len(jet_eta):
                return None
            return calculate_dr(jet_eta[hj1], jet_phi[hj1],
                                jet_eta[hj2], jet_phi[hj2])

        # Compute DR for all four ranks
        dr_1 = get_dr(max_comb)
        dr_2 = get_dr(second_comb)
        dr_3 = get_dr(third_comb)
        dr_4 = get_dr(fourth_comb)

        # Store DR and mass for all ranks unconditionally
        dr_rank1_1.append(dr_1)
        dr_rank2_1.append(dr_2)
        dr_rank3_1.append(dr_3)
        dr_rank4_1.append(dr_4)
        m1 = events["max_weight_higgs_mass"][event_idx]
        m2 = events["second_max_weight_higgs_mass"][event_idx]
        m3 = events["third_max_weight_higgs_mass"][event_idx]
        m4 = events["fourth_max_weight_higgs_mass"][event_idx]
        higgs_mass_rank1_1.append(m1)
        higgs_mass_rank2_1.append(m2)
        higgs_mass_rank3_1.append(m3)
        higgs_mass_rank4_1.append(m4)

        # Select best combination
        if dr_1 is not None and dr_1 < dr_threshold:
            comb_key = "max_weight"
            rank = 1
        elif dr_2 is not None and dr_2 < dr_threshold:
            comb_key = "second_max_weight"
            rank = 2
            n_promoted += 1
        elif dr_3 is not None and dr_3 < dr_threshold:
            comb_key = "third_max_weight"
            rank = 3
            n_promoted += 1
        elif dr_4 is not None and dr_4 < dr_threshold:
            comb_key = "fourth_max_weight"
            rank = 4
            n_promoted += 1
        else:
            comb_key = "max_weight"
            rank = 0
            n_fallback += 1

        rank_counts[rank] += 1
        rank_used_1.append(rank)

        chosen_higgs_mass = events[f"{comb_key}_higgs_mass"][event_idx]
        chosen_dr = dr_1 if rank in (1, 0) else (dr_2 if rank == 2 else (dr_3 if rank == 3 else dr_4))

        dr_chosen_1.append(chosen_dr)
        higgs_mass_chosen_1.append(chosen_higgs_mass)

        massreco_chosen_pair_higgs_mass_1.append(chosen_higgs_mass)
        massreco_chosen_pair_top_mass_1.append(events[f"{comb_key}_top_mass"][event_idx])
        massreco_chosen_pair_W_mass_1.append(events[f"{comb_key}_W_mass"][event_idx])
        massreco_chosen_pair_ttH_mass_1.append(events[f"{comb_key}_ttH_mass"][event_idx])
        massreco_chosen_pair_combination_1.append(events[f"{comb_key}_combination"][event_idx])

        # --- Selected: only the chosen rank gets filled ---
        higgs_mass_chosen_from_rank1_1.append(chosen_higgs_mass if rank == 1 else None)
        higgs_mass_chosen_from_rank2_1.append(chosen_higgs_mass if rank == 2 else None)
        higgs_mass_chosen_from_rank3_1.append(chosen_higgs_mass if rank == 3 else None)
        higgs_mass_chosen_from_rank4_1.append(chosen_higgs_mass if rank == 4 else None)
        higgs_mass_chosen_from_fallback_1.append(chosen_higgs_mass if rank == 0 else None)
        dr_chosen_from_rank1_1.append(chosen_dr if rank == 1 else None)
        dr_chosen_from_rank2_1.append(chosen_dr if rank == 2 else None)
        dr_chosen_from_rank3_1.append(chosen_dr if rank == 3 else None)
        dr_chosen_from_rank4_1.append(chosen_dr if rank == 4 else None)
        dr_chosen_from_fallback_1.append(chosen_dr if rank == 0 else None)

        # --- Rejected: filled when that rank was skipped ---
        # rank1 rejected: when promoted to rank 2, 3, or 4
        higgs_mass_rank1_rejected_1.append(m1 if rank in (2, 3, 4) else None)
        dr_rank1_rejected_1.append(dr_1 if rank in (2, 3, 4) else None)
        # rank2 rejected: when promoted to rank 3 or 4
        higgs_mass_rank2_rejected_1.append(m2 if rank in (3, 4) else None)
        dr_rank2_rejected_1.append(dr_2 if rank in (3, 4) else None)
        # rank3 rejected: when promoted to rank 4
        higgs_mass_rank3_rejected_1.append(m3 if rank == 4 else None)
        dr_rank3_rejected_1.append(dr_3 if rank == 4 else None)
        # rank4 rejected: when fallback (rank4 tried but also failed DR cut)
        higgs_mass_rank4_rejected_1.append(m4 if rank == 0 else None)
        dr_rank4_rejected_1.append(dr_4 if rank == 0 else None)

    # --- Summary ---
    n_with_reco = t - n_no_reco
    print(f"\n=== dr_criterion_for_max_weight (threshold={dr_threshold}) ===")
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

    res = {
        # Original outputs
        "massreco_chosen_pair_higgs_mass":   massreco_chosen_pair_higgs_mass_1,
        "massreco_chosen_pair_top_mass":     massreco_chosen_pair_top_mass_1,
        "massreco_chosen_pair_W_mass":       massreco_chosen_pair_W_mass_1,
        "massreco_chosen_pair_ttH_mass":     massreco_chosen_pair_ttH_mass_1,
        "massreco_chosen_pair_combination":  massreco_chosen_pair_combination_1,
        # All-events per-rank
        "rank_used":           rank_used_1,
        "dr_rank1":            dr_rank1_1,
        "dr_rank2":            dr_rank2_1,
        "dr_rank3":            dr_rank3_1,
        "dr_rank4":            dr_rank4_1,
        "dr_chosen":           dr_chosen_1,
        "higgs_mass_rank1":    higgs_mass_rank1_1,
        "higgs_mass_rank2":    higgs_mass_rank2_1,
        "higgs_mass_rank3":    higgs_mass_rank3_1,
        "higgs_mass_rank4":    higgs_mass_rank4_1,
        "higgs_mass_chosen":   higgs_mass_chosen_1,
        # Selected-from-rank
        "higgs_mass_chosen_from_rank1":    higgs_mass_chosen_from_rank1_1,
        "higgs_mass_chosen_from_rank2":    higgs_mass_chosen_from_rank2_1,
        "higgs_mass_chosen_from_rank3":    higgs_mass_chosen_from_rank3_1,
        "higgs_mass_chosen_from_rank4":    higgs_mass_chosen_from_rank4_1,
        "higgs_mass_chosen_from_fallback": higgs_mass_chosen_from_fallback_1,
        "dr_chosen_from_rank1":            dr_chosen_from_rank1_1,
        "dr_chosen_from_rank2":            dr_chosen_from_rank2_1,
        "dr_chosen_from_rank3":            dr_chosen_from_rank3_1,
        "dr_chosen_from_rank4":            dr_chosen_from_rank4_1,
        "dr_chosen_from_fallback":         dr_chosen_from_fallback_1,
        # Rejected-from-rank
        "higgs_mass_rank1_rejected":       higgs_mass_rank1_rejected_1,
        "higgs_mass_rank2_rejected":       higgs_mass_rank2_rejected_1,
        "higgs_mass_rank3_rejected":       higgs_mass_rank3_rejected_1,
        "higgs_mass_rank4_rejected":       higgs_mass_rank4_rejected_1,
        "dr_rank1_rejected":               dr_rank1_rejected_1,
        "dr_rank2_rejected":               dr_rank2_rejected_1,
        "dr_rank3_rejected":               dr_rank3_rejected_1,
        "dr_rank4_rejected":               dr_rank4_rejected_1,
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


def get_solver_jet_indices(events, event_idx):
    """
    Returns the JetGood-array indices of the jets actually available to
    solve_ttbar_dilepton()'s combinatorics for this event.

    Reads events["1st_jet_idx".."4th_jet_idx"] -- whichever ordering
    criterion workflow_mc.py used to build them (pT now, previously
    btag) -- instead of independently recomputing a hardcoded btag-based
    top-4. This keeps the acceptance check (could the solver possibly
    have found the truth pair?) automatically consistent with whatever
    jets the solver was actually given.

    Only the first 4 are included: solve_ttbar_dilepton() hardcodes
    n_jets = 4 for its combinatorics loop, so even though a 5th jet is
    sometimes appended to its local `jets` list, it is never actually
    reached by `for i in range(n_jets)` (range(4)). If that n_jets bound
    is ever changed to use the 5th jet too, "5th_jet_idx" should be added
    here as well.
    """
    idx_fields = ["1st_jet_idx", "2nd_jet_idx", "3rd_jet_idx", "4th_jet_idx"]
    indices = []
    for f in idx_fields:
        if f not in ak.fields(events):
            continue
        val = events[f][event_idx]
        if val is None:
            continue
        indices.append(int(val))
    return indices


def compute_jet_pair_properties_all_events(events, j_matched, q_matched, prefix="max_weight", btag_branch="btagUParTAK4B"):

    pair_phi, pair_eta, pair_mass, pair_pt, pair_indices, pair_dr = [], [], [], [], [], []
    original_jet_pair_indices = []
    numerators = []
    denominators = []
    correct_matches = []
    correct_agreements = []
    higgs_mass_truth_jets_1 = []
    firstandsecondmaxweight_higgs_mass_1 = []
    massreco_chosen_pair_correct_higgs_mass_1 = []
    massreco_chosen_pair_all_higgs_mass_1 = []
    massreco_chosen_pair_wrong_higgs_mass_1 = []
    massreco_chosen_pair_wrong_higgs_mass_2_jets_1 = []
    massreco_chosen_pair_wrong_higgs_mass_1_jets_1 = []
    massreco_chosen_pair_correct_top_mass_1 = []
    massreco_chosen_pair_wrong_top_mass_1 = []
    massreco_chosen_pair_correct_W_mass_1 = []
    massreco_chosen_pair_wrong_W_mass_1 = []
    massreco_chosen_pair_correct_ttH_mass_1 = []
    massreco_chosen_pair_wrong_ttH_mass_1 = []
    massreco_chosen_pair_correct_jet_pt_1 = []
    massreco_chosen_pair_wrong_jet_pt_1 = []
    massreco_chosen_pair_correct_jet_phi_1 = []
    massreco_chosen_pair_wrong_jet_phi_1 = []
    massreco_chosen_pair_correct_jet_eta_1 = []
    massreco_chosen_pair_wrong_jet_eta_1 = []
    massreco_chosen_pair_correct_jet_mult_1 = []
    massreco_chosen_pair_wrong_jet_mult_1 = []
    massreco_chosen_pair_wrong_dr_1 = []
    massreco_chosen_pair_correct_dr_1 = []
    massreco_chosen_pair_wrong_delta_angle_1 = []
    massreco_chosen_pair_correct_delta_angle_1 = []
    massreco_chosen_pair_wrong_pull_angle_1 = []
    massreco_chosen_pair_correct_pull_angle_1 = []
    massreco_chosen_pair_wrong_delta_magn_1 = []
    massreco_chosen_pair_correct_delta_magn_1 = []
    massreco_chosen_pair_wrong_pull_magn_1 = []
    massreco_chosen_pair_correct_pull_magn_1 = []
    massreco_chosen_pair_wrong_dr_top_1 = []
    massreco_chosen_pair_correct_dr_top_1 = []
    massreco_chosen_pair_wrong_lep_neg_1 = []
    massreco_chosen_pair_correct_lep_neg_1 = []
    massreco_chosen_pair_wrong_lep_pos_1 = []
    massreco_chosen_pair_correct_lep_pos_1 = []
    massreco_chosen_pair_wrong_pt_assymetry_1 = []
    massreco_chosen_pair_correct_pt_assymetry_1 = []
    massreco_chosen_pair_correct_jet_btag_score_1 = []
    massreco_chosen_pair_wrong_jet_btag_score_1 = []

    def skip_event():
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
        higgs_mass_truth_jets_1.append([])
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
        massreco_chosen_pair_wrong_delta_magn_1.append([])
        massreco_chosen_pair_correct_delta_magn_1.append([])
        massreco_chosen_pair_wrong_pull_angle_1.append([])
        massreco_chosen_pair_correct_pull_angle_1.append([])
        massreco_chosen_pair_wrong_pull_magn_1.append([])
        massreco_chosen_pair_correct_pull_magn_1.append([])
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

    t = len(events)

    ############## Loop over events ###########################
    for event_idx in range(t):

        # --- Step 1: find truth-matched Higgs jet indices ---
        higgs_matched_indices = [
            idx for idx, parton in enumerate(q_matched[event_idx])
            if parton is not None and parton.provenance == 1
        ]
        selected_jet_pair = higgs_matched_indices if len(higgs_matched_indices) == 2 else []

        # Skip if we don't have exactly 2 truth-matched Higgs jets
        if len(selected_jet_pair) != 2:
            skip_event()
            continue

        # --- Step 2: require the solver actually had >=4 jet candidates ---
        solver_jet_indices = get_solver_jet_indices(events, event_idx)
        if len(solver_jet_indices) < 4:
            skip_event()
            continue

        # --- Step 3: require both truth jets are among the jets the solver
        #     actually received (events["1st_jet_idx".."4th_jet_idx"] --
        #     whichever ordering criterion is active in workflow_mc.py: pT
        #     now, previously btag) ---
        jet0, jet1 = selected_jet_pair
        both_in_top4 = jet0 in solver_jet_indices and jet1 in solver_jet_indices
        if event_idx < 5:
            print(f"  [Event {event_idx}] truth_jets={selected_jet_pair}, "
                  f"solver_jet_idx={solver_jet_indices}, "
                  f"both_in_top4={both_in_top4}")
        if not both_in_top4:
            skip_event()
            continue

        # --- Step 4: compute truth-pair invariant mass ---
        pt1   = events["JetGood"]["pt"][event_idx][jet0]
        eta1  = events["JetGood"]["eta"][event_idx][jet0]
        phi1  = events["JetGood"]["phi"][event_idx][jet0]
        mass1 = events["JetGood"]["mass"][event_idx][jet0]
        pt2   = events["JetGood"]["pt"][event_idx][jet1]
        eta2  = events["JetGood"]["eta"][event_idx][jet1]
        phi2  = events["JetGood"]["phi"][event_idx][jet1]
        mass2 = events["JetGood"]["mass"][event_idx][jet1]

        p4_1 = vector.obj(pt=pt1, eta=eta1, phi=phi1, mass=mass1)
        p4_2 = vector.obj(pt=pt2, eta=eta2, phi=phi2, mass=mass2)
        higgs_mass_truth_jets = (p4_1 + p4_2).mass

        # --- Step 5: check if mass reco produced a combination for this event ---
        reco_combination = events[f"{prefix}_combination"][event_idx]

        if reco_combination is None:
            # Truth jets found but mass reco had no solution: record truth mass, zeros for match
            higgs_mass_truth_jets_1.append([higgs_mass_truth_jets])
            numerators.append([0.])
            denominators.append([1.])
            correct_matches.append([0.])
            correct_agreements.append([0.])
            # All output quantities are None/empty for this event
            massreco_chosen_pair_all_higgs_mass_1.append([None])
            massreco_chosen_pair_correct_higgs_mass_1.append([None])
            massreco_chosen_pair_wrong_higgs_mass_1.append([None])
            massreco_chosen_pair_wrong_higgs_mass_2_jets_1.append([None])
            massreco_chosen_pair_wrong_higgs_mass_1_jets_1.append([None])
            massreco_chosen_pair_correct_top_mass_1.append([None])
            massreco_chosen_pair_wrong_top_mass_1.append([None])
            massreco_chosen_pair_correct_W_mass_1.append([None])
            massreco_chosen_pair_wrong_W_mass_1.append([None])
            massreco_chosen_pair_correct_ttH_mass_1.append([None])
            massreco_chosen_pair_wrong_ttH_mass_1.append([None])
            massreco_chosen_pair_correct_jet_pt_1.append([None, None])
            massreco_chosen_pair_wrong_jet_pt_1.append([None, None])
            massreco_chosen_pair_correct_jet_eta_1.append([None, None])
            massreco_chosen_pair_wrong_jet_eta_1.append([None, None])
            massreco_chosen_pair_correct_jet_phi_1.append([None, None])
            massreco_chosen_pair_wrong_jet_phi_1.append([None, None])
            massreco_chosen_pair_correct_jet_mult_1.append([None])
            massreco_chosen_pair_wrong_jet_mult_1.append([None])
            massreco_chosen_pair_correct_dr_1.append([None])
            massreco_chosen_pair_wrong_dr_1.append([None])
            massreco_chosen_pair_correct_delta_angle_1.append([None])
            massreco_chosen_pair_wrong_delta_angle_1.append([None])
            massreco_chosen_pair_correct_delta_magn_1.append([None])
            massreco_chosen_pair_wrong_delta_magn_1.append([None])
            massreco_chosen_pair_correct_pull_angle_1.append([[np.nan, np.nan]])
            massreco_chosen_pair_wrong_pull_angle_1.append([[np.nan, np.nan]])
            massreco_chosen_pair_correct_pull_magn_1.append([[np.nan, np.nan]])
            massreco_chosen_pair_wrong_pull_magn_1.append([[np.nan, np.nan]])
            massreco_chosen_pair_correct_dr_top_1.append([None])
            massreco_chosen_pair_wrong_dr_top_1.append([None])
            massreco_chosen_pair_correct_lep_pos_1.append([None])
            massreco_chosen_pair_wrong_lep_pos_1.append([None])
            massreco_chosen_pair_correct_lep_neg_1.append([None])
            massreco_chosen_pair_wrong_lep_neg_1.append([None])
            massreco_chosen_pair_correct_pt_assymetry_1.append([None])
            massreco_chosen_pair_wrong_pt_assymetry_1.append([None])
            massreco_chosen_pair_correct_jet_btag_score_1.append([None, None])
            massreco_chosen_pair_wrong_jet_btag_score_1.append([None, None])
            pair_phi.append([])
            pair_eta.append([])
            pair_mass.append([])
            pair_pt.append([])
            pair_indices.append([])
            pair_dr.append([])
            firstandsecondmaxweight_higgs_mass_1.append([])
            continue

        # --- Step 6: read mass-reco Higgs jet assignments (FIX A: outside the loop) ---
        hj1    = reco_combination["2"]
        hj2    = reco_combination["3"]
        j_top1 = reco_combination["0"]
        j_top2 = reco_combination["1"]

        original_hj1    = get_original_jet_idx(events, event_idx, hj1)
        original_hj2    = get_original_jet_idx(events, event_idx, hj2)
        original_j_top1 = get_original_jet_idx(events, event_idx, j_top1)
        original_j_top2 = get_original_jet_idx(events, event_idx, j_top2)

        n_jets_good = len(events["JetGood"]["eta"][event_idx])

        # Guard against None returns from get_original_jet_idx (e.g. 5th jet case)
        if original_hj1 is None or original_hj2 is None:
            skip_event()
            continue
        if original_j_top1 is None or original_j_top2 is None:
            skip_event()
            continue

        # --- Step 7: compute Higgs-pair kinematics from reco combination (FIX A: once, not per loop) ---
        dr = delta_angle = pull_magnitude_diff = pull_angle_diff = None
        pull_magnitude1 = pull_angle1 = pull_magnitude2 = pull_angle2 = np.nan
        pt_assymetry = None

        if original_hj1 < n_jets_good and original_hj2 < n_jets_good:
            eta_hj1  = events["JetGood"]["eta"][event_idx][original_hj1]
            phi_hj1  = events["JetGood"]["phi"][event_idx][original_hj1]
            eta_hj2  = events["JetGood"]["eta"][event_idx][original_hj2]
            phi_hj2  = events["JetGood"]["phi"][event_idx][original_hj2]

            pull_angle1     = events["JetGood"]["pull_angle"][event_idx][original_hj1]
            pull_angle2     = events["JetGood"]["pull_angle"][event_idx][original_hj2]
            pull_magnitude1 = events["JetGood"]["pull_magnitude"][event_idx][original_hj1]
            pull_magnitude2 = events["JetGood"]["pull_magnitude"][event_idx][original_hj2]

            dr                = calculate_dr(eta_hj1, phi_hj1, eta_hj2, phi_hj2)
            delta_angle       = pull_angle1 - pull_angle2
            pull_magnitude_diff = pull_magnitude1 - pull_magnitude2
            pull_angle_diff   = pull_angle1 - pull_angle2

            pt_hj1 = events["JetGood"]["pt"][event_idx][original_hj1]
            pt_hj2 = events["JetGood"]["pt"][event_idx][original_hj2]
            pt_assymetry = calculate_pt_asymmetry(pt_hj1, pt_hj2)

        # --- Step 8: compute top-b-jet kinematics ---
        dr_top = None
        pt_asymmetry_top = None
        eta_top1 = phi_top1 = eta_top2 = phi_top2 = None  # initialise before lepton block

        if original_j_top1 < n_jets_good and original_j_top2 < n_jets_good:
            eta_top1 = events["JetGood"]["eta"][event_idx][original_j_top1]
            phi_top1 = events["JetGood"]["phi"][event_idx][original_j_top1]
            eta_top2 = events["JetGood"]["eta"][event_idx][original_j_top2]
            phi_top2 = events["JetGood"]["phi"][event_idx][original_j_top2]
            dr_top = calculate_dr(eta_top1, phi_top1, eta_top2, phi_top2)

            pt_top1 = events["JetGood"]["pt"][event_idx][original_j_top1]
            pt_top2 = events["JetGood"]["pt"][event_idx][original_j_top2]
            pt_asymmetry_top = calculate_pt_asymmetry(pt_top1, pt_top2)

        # --- Step 9: compute lepton-to-top-jet ΔR (FIX B+C: guarded + corrected logic) ---
        dr_min_pos = dr_min_neg = None
        lep_pos = events["lepton_pos"][event_idx]
        lep_neg = events["lepton_neg"][event_idx]

        if (len(lep_pos) == 1 and len(lep_neg) == 1
                and eta_top1 is not None and eta_top2 is not None):  # FIX B: guard

            eta_pos, phi_pos = lep_pos["eta"][0], lep_pos["phi"][0]
            eta_neg, phi_neg = lep_neg["eta"][0], lep_neg["phi"][0]

            dr1_pos = calculate_dr(eta_top1, phi_top1, eta_pos, phi_pos)
            dr2_pos = calculate_dr(eta_top2, phi_top2, eta_pos, phi_pos)
            dr1_neg = calculate_dr(eta_top1, phi_top1, eta_neg, phi_neg)
            dr2_neg = calculate_dr(eta_top2, phi_top2, eta_neg, phi_neg)

            # Closest jet to positive lepton
            if dr1_pos < dr2_pos:
                closest_jet_to_lep_pos = original_j_top1
                dr_min_pos = dr1_pos
            else:
                closest_jet_to_lep_pos = original_j_top2
                dr_min_pos = dr2_pos

            # FIX C: assign the *other* jet to lep_neg, irrespective of which ΔR is smaller
            if closest_jet_to_lep_pos == original_j_top1:
                dr_min_neg = dr2_neg
            else:
                dr_min_neg = dr1_neg

        # --- Step 10: determine match category (FIX A: plain comparison, not inside loop) ---
        first_match  = 1. if jet0 == original_hj1 or jet1 == original_hj1 else 0.
        second_match = 1. if jet0 == original_hj2 or jet1 == original_hj2 else 0.

        # FIX B (correct_match semantics): derive cleanly from first/second match
        if first_match == 1. and second_match == 1.:
            correct_match = 2.   # both Higgs jets correctly identified
        elif first_match == 1. or second_match == 1.:
            correct_match = 1.   # one Higgs jet correctly identified
        else:
            correct_match = 0.   # neither Higgs jet correctly identified

        denominator = 1.
        numerator   = 1. if correct_match > 0. else 0.

        # --- Block 2: per-event match decision (first 10 events) ---
        if event_idx < 10:
            reco_mass = events[f"{prefix}_higgs_mass"][event_idx]
            print(f"\n--- Event {event_idx} ---")
            print(f"  Truth Higgs jets (JetGood idx) : {selected_jet_pair}")
            print(f"  Reco Higgs slots (btag-sorted) : hj1={hj1}, hj2={hj2}")
            print(f"  Reco Higgs jets (JetGood idx)  : original_hj1={original_hj1}, original_hj2={original_hj2}")
            print(f"  first_match={first_match}, second_match={second_match}, correct_match={correct_match}")
            print(f"  higgs_mass_truth = {higgs_mass_truth_jets:.2f} GeV")
            print(f"  higgs_mass_reco  = {reco_mass:.2f} GeV" if reco_mass is not None else "  higgs_mass_reco  = None")
            print(f"  DR(reco Higgs pair) = {dr:.3f}" if dr is not None else "  DR(reco Higgs pair) = None")

        # --- Step 11: fill per-case output variables ---
        # Initialise all to None / nan
        massreco_chosen_pair_wrong_higgs_mass        = None
        massreco_chosen_pair_wrong_higgs_mass_2_jets = None
        massreco_chosen_pair_wrong_higgs_mass_1_jets = None
        massreco_chosen_pair_correct_higgs_mass      = None
        massreco_chosen_pair_all_higgs_mass          = None
        massreco_chosen_pair_wrong_top_mass          = None
        massreco_chosen_pair_correct_top_mass        = None
        massreco_chosen_pair_wrong_W_mass            = None
        massreco_chosen_pair_correct_W_mass          = None
        massreco_chosen_pair_wrong_ttH_mass          = None
        massreco_chosen_pair_correct_ttH_mass        = None
        massreco_chosen_pair_correct_jet_pt          = None
        massreco_chosen_pair_wrong_jet_pt            = None
        massreco_chosen_pair_correct_jet_pt_second   = None
        massreco_chosen_pair_wrong_jet_pt_second     = None
        massreco_chosen_pair_correct_jet_eta         = None
        massreco_chosen_pair_wrong_jet_eta           = None
        massreco_chosen_pair_correct_jet_eta_second  = None
        massreco_chosen_pair_wrong_jet_eta_second    = None
        massreco_chosen_pair_correct_jet_phi         = None
        massreco_chosen_pair_wrong_jet_phi           = None
        massreco_chosen_pair_correct_jet_phi_second  = None
        massreco_chosen_pair_wrong_jet_phi_second    = None
        massreco_chosen_pair_correct_jet_btag_score        = None
        massreco_chosen_pair_correct_jet_btag_score_second = None
        massreco_chosen_pair_wrong_jet_btag_score          = None
        massreco_chosen_pair_wrong_jet_btag_score_second   = None
        massreco_chosen_pair_correct_jet_mult        = None
        massreco_chosen_pair_wrong_jet_mult          = None
        massreco_chosen_pair_correct_dr              = None
        massreco_chosen_pair_wrong_dr                = None
        massreco_chosen_pair_correct_delta_angle     = None
        massreco_chosen_pair_wrong_delta_angle       = None
        massreco_chosen_pair_correct_delta_magn      = None
        massreco_chosen_pair_wrong_delta_magn        = None
        massreco_chosen_pair_correct_pull_angle      = [np.nan, np.nan]
        massreco_chosen_pair_wrong_pull_angle        = [np.nan, np.nan]
        massreco_chosen_pair_correct_pull_magn       = [np.nan, np.nan]
        massreco_chosen_pair_wrong_pull_magn         = [np.nan, np.nan]
        massreco_chosen_pair_correct_dr_top          = None
        massreco_chosen_pair_wrong_dr_top            = None
        massreco_chosen_pair_correct_lep_pos         = None
        massreco_chosen_pair_wrong_lep_pos           = None
        massreco_chosen_pair_correct_lep_neg         = None
        massreco_chosen_pair_wrong_lep_neg           = None
        massreco_chosen_pair_correct_pt_assymetry    = None
        massreco_chosen_pair_wrong_pt_assymetry      = None

        # ----- case: first jet matched, second wrong -----
        if first_match == 1. and second_match == 0.:
            massreco_chosen_pair_wrong_higgs_mass_1_jets = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_wrong_higgs_mass   = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_all_higgs_mass     = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_wrong_top_mass     = events[f"{prefix}_top_mass"][event_idx]
            massreco_chosen_pair_wrong_W_mass       = events[f"{prefix}_W_mass"][event_idx]
            massreco_chosen_pair_wrong_ttH_mass     = events[f"{prefix}_ttH_mass"][event_idx]
            massreco_chosen_pair_wrong_dr           = dr
            massreco_chosen_pair_wrong_dr_top       = dr_top
            massreco_chosen_pair_wrong_lep_pos      = dr_min_pos
            massreco_chosen_pair_wrong_lep_neg      = dr_min_neg
            massreco_chosen_pair_wrong_pt_assymetry = pt_assymetry
            massreco_chosen_pair_wrong_delta_angle  = delta_angle
            massreco_chosen_pair_wrong_delta_magn   = pull_magnitude_diff
            massreco_chosen_pair_wrong_jet_mult     = events["nJetGood"][event_idx]
            massreco_chosen_pair_correct_pull_magn  = [pull_magnitude1, np.nan]
            massreco_chosen_pair_correct_pull_angle = [pull_angle1, np.nan]
            massreco_chosen_pair_wrong_pull_magn   = [np.nan, pull_magnitude2]
            massreco_chosen_pair_wrong_pull_angle  = [np.nan, pull_angle2]
            if original_hj1 < n_jets_good and original_hj2 < n_jets_good:
                massreco_chosen_pair_correct_jet_pt    = events["JetGood"]["pt"][event_idx][original_hj1]
                massreco_chosen_pair_wrong_jet_pt      = events["JetGood"]["pt"][event_idx][original_hj2]
                massreco_chosen_pair_correct_jet_eta   = events["JetGood"]["eta"][event_idx][original_hj1]
                massreco_chosen_pair_wrong_jet_eta     = events["JetGood"]["eta"][event_idx][original_hj2]
                massreco_chosen_pair_correct_jet_phi   = events["JetGood"]["phi"][event_idx][original_hj1]
                massreco_chosen_pair_wrong_jet_phi     = events["JetGood"]["phi"][event_idx][original_hj2]
                massreco_chosen_pair_correct_jet_btag_score = events["JetGood"][btag_branch][event_idx][original_hj1]
                massreco_chosen_pair_wrong_jet_btag_score   = events["JetGood"][btag_branch][event_idx][original_hj2]

        # ----- case: second jet matched, first wrong -----
        elif first_match == 0. and second_match == 1.:
            massreco_chosen_pair_wrong_higgs_mass_1_jets = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_wrong_higgs_mass   = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_all_higgs_mass     = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_wrong_top_mass     = events[f"{prefix}_top_mass"][event_idx]
            massreco_chosen_pair_wrong_W_mass       = events[f"{prefix}_W_mass"][event_idx]
            massreco_chosen_pair_wrong_ttH_mass     = events[f"{prefix}_ttH_mass"][event_idx]
            massreco_chosen_pair_wrong_dr           = dr
            massreco_chosen_pair_wrong_dr_top       = dr_top
            massreco_chosen_pair_wrong_lep_pos      = dr_min_pos
            massreco_chosen_pair_wrong_lep_neg      = dr_min_neg
            massreco_chosen_pair_wrong_pt_assymetry = pt_assymetry
            massreco_chosen_pair_wrong_delta_angle  = delta_angle
            massreco_chosen_pair_wrong_delta_magn   = pull_magnitude_diff
            massreco_chosen_pair_wrong_jet_mult     = events["nJetGood"][event_idx]
            massreco_chosen_pair_correct_pull_magn  = [pull_magnitude2, np.nan]
            massreco_chosen_pair_correct_pull_angle = [pull_angle2, np.nan]
            massreco_chosen_pair_wrong_pull_magn   = [np.nan, pull_magnitude1]
            massreco_chosen_pair_wrong_pull_angle  = [np.nan, pull_angle1]
            if original_hj1 < n_jets_good and original_hj2 < n_jets_good:
                massreco_chosen_pair_correct_jet_pt    = events["JetGood"]["pt"][event_idx][original_hj2]
                massreco_chosen_pair_wrong_jet_pt      = events["JetGood"]["pt"][event_idx][original_hj1]
                massreco_chosen_pair_correct_jet_eta   = events["JetGood"]["eta"][event_idx][original_hj2]
                massreco_chosen_pair_wrong_jet_eta     = events["JetGood"]["eta"][event_idx][original_hj1]
                massreco_chosen_pair_correct_jet_phi   = events["JetGood"]["phi"][event_idx][original_hj2]
                massreco_chosen_pair_wrong_jet_phi     = events["JetGood"]["phi"][event_idx][original_hj1]
                massreco_chosen_pair_correct_jet_btag_score = events["JetGood"][btag_branch][event_idx][original_hj2]
                massreco_chosen_pair_wrong_jet_btag_score   = events["JetGood"][btag_branch][event_idx][original_hj1]

        # ----- case: neither jet matched (FIX D: separate if block, not elif) -----
        # Note: this is now a separate if, correctly independent of the above elif chain
        if first_match == 0. and second_match == 0.:
            # Mass/event-level quantities: always fill, no JetGood indexing needed
            massreco_chosen_pair_wrong_higgs_mass      = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_wrong_higgs_mass_2_jets = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_all_higgs_mass        = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_wrong_top_mass        = events[f"{prefix}_top_mass"][event_idx]
            massreco_chosen_pair_wrong_W_mass          = events[f"{prefix}_W_mass"][event_idx]
            massreco_chosen_pair_wrong_ttH_mass        = events[f"{prefix}_ttH_mass"][event_idx]
            massreco_chosen_pair_wrong_jet_mult        = events["nJetGood"][event_idx]
            massreco_chosen_pair_wrong_dr              = dr
            massreco_chosen_pair_wrong_dr_top          = dr_top
            massreco_chosen_pair_wrong_lep_pos         = dr_min_pos
            massreco_chosen_pair_wrong_lep_neg         = dr_min_neg
            massreco_chosen_pair_wrong_pt_assymetry    = pt_assymetry
            massreco_chosen_pair_wrong_delta_angle     = delta_angle
            massreco_chosen_pair_wrong_delta_magn      = pull_magnitude_diff
            massreco_chosen_pair_wrong_pull_magn       = [pull_magnitude1, pull_magnitude2]
            massreco_chosen_pair_wrong_pull_angle      = [pull_angle1, pull_angle2]
            # Jet-level quantities: only fill if reco indices are within JetGood bounds
            if original_hj1 < n_jets_good and original_hj2 < n_jets_good:
                massreco_chosen_pair_wrong_jet_pt         = events["JetGood"]["pt"][event_idx][original_hj1]
                massreco_chosen_pair_wrong_jet_pt_second  = events["JetGood"]["pt"][event_idx][original_hj2]
                massreco_chosen_pair_wrong_jet_eta        = events["JetGood"]["eta"][event_idx][original_hj1]
                massreco_chosen_pair_wrong_jet_eta_second = events["JetGood"]["eta"][event_idx][original_hj2]
                massreco_chosen_pair_wrong_jet_phi        = events["JetGood"]["phi"][event_idx][original_hj1]
                massreco_chosen_pair_wrong_jet_phi_second = events["JetGood"]["phi"][event_idx][original_hj2]
                massreco_chosen_pair_wrong_jet_btag_score        = events["JetGood"][btag_branch][event_idx][original_hj1]
                massreco_chosen_pair_wrong_jet_btag_score_second = events["JetGood"][btag_branch][event_idx][original_hj2]

        # ----- case: both jets matched (FIX D: separate if, was unreachable elif) -----
        if first_match == 1. and second_match == 1.:
            massreco_chosen_pair_correct_jet_pt           = events["JetGood"]["pt"][event_idx][original_hj1]
            massreco_chosen_pair_correct_jet_pt_second    = events["JetGood"]["pt"][event_idx][original_hj2]
            massreco_chosen_pair_correct_jet_eta          = events["JetGood"]["eta"][event_idx][original_hj1]
            massreco_chosen_pair_correct_jet_eta_second   = events["JetGood"]["eta"][event_idx][original_hj2]
            massreco_chosen_pair_correct_jet_phi          = events["JetGood"]["phi"][event_idx][original_hj1]
            massreco_chosen_pair_correct_jet_phi_second   = events["JetGood"]["phi"][event_idx][original_hj2]
            massreco_chosen_pair_correct_jet_btag_score        = events["JetGood"][btag_branch][event_idx][original_hj1]
            massreco_chosen_pair_correct_jet_btag_score_second = events["JetGood"][btag_branch][event_idx][original_hj2]
            massreco_chosen_pair_correct_higgs_mass  = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_all_higgs_mass      = events[f"{prefix}_higgs_mass"][event_idx]
            massreco_chosen_pair_correct_top_mass    = events[f"{prefix}_top_mass"][event_idx]
            massreco_chosen_pair_correct_W_mass      = events[f"{prefix}_W_mass"][event_idx]
            massreco_chosen_pair_correct_ttH_mass    = events[f"{prefix}_ttH_mass"][event_idx]
            massreco_chosen_pair_correct_dr          = dr
            massreco_chosen_pair_correct_delta_angle = delta_angle
            massreco_chosen_pair_correct_delta_magn  = pull_magnitude_diff
            massreco_chosen_pair_correct_dr_top      = dr_top
            massreco_chosen_pair_correct_lep_pos     = dr_min_pos
            massreco_chosen_pair_correct_lep_neg     = dr_min_neg
            massreco_chosen_pair_correct_pt_assymetry = pt_assymetry
            massreco_chosen_pair_correct_jet_mult    = events["nJetGood"][event_idx]
            # FIX E: correct_pull_* filled here (was previously stored in wrong_pull_*)
            massreco_chosen_pair_correct_pull_magn   = [pull_magnitude1, pull_magnitude2]
            massreco_chosen_pair_correct_pull_angle  = [pull_angle1, pull_angle2]

        # --- Step 12: append results ---
        higgs_mass_truth_jets_1.append([higgs_mass_truth_jets])
        numerators.append([numerator])
        denominators.append([denominator])
        correct_matches.append([correct_match])
        correct_agreements.append([0.])   # agreement logic was unused; kept as placeholder
        firstandsecondmaxweight_higgs_mass_1.append([])
        pair_phi.append([])
        pair_eta.append([])
        pair_mass.append([])
        pair_pt.append([])
        pair_indices.append([])
        pair_dr.append([])

        massreco_chosen_pair_all_higgs_mass_1.append([massreco_chosen_pair_all_higgs_mass])
        massreco_chosen_pair_correct_higgs_mass_1.append([massreco_chosen_pair_correct_higgs_mass])
        massreco_chosen_pair_wrong_higgs_mass_1.append([massreco_chosen_pair_wrong_higgs_mass])
        massreco_chosen_pair_wrong_higgs_mass_2_jets_1.append([massreco_chosen_pair_wrong_higgs_mass_2_jets])
        massreco_chosen_pair_wrong_higgs_mass_1_jets_1.append([massreco_chosen_pair_wrong_higgs_mass_1_jets])
        massreco_chosen_pair_correct_top_mass_1.append([massreco_chosen_pair_correct_top_mass])
        massreco_chosen_pair_wrong_top_mass_1.append([massreco_chosen_pair_wrong_top_mass])
        massreco_chosen_pair_correct_W_mass_1.append([massreco_chosen_pair_correct_W_mass])
        massreco_chosen_pair_wrong_W_mass_1.append([massreco_chosen_pair_wrong_W_mass])
        massreco_chosen_pair_correct_ttH_mass_1.append([massreco_chosen_pair_correct_ttH_mass])
        massreco_chosen_pair_wrong_ttH_mass_1.append([massreco_chosen_pair_wrong_ttH_mass])
        massreco_chosen_pair_correct_jet_pt_1.append([massreco_chosen_pair_correct_jet_pt,
                                                      massreco_chosen_pair_correct_jet_pt_second])
        massreco_chosen_pair_wrong_jet_pt_1.append([massreco_chosen_pair_wrong_jet_pt,
                                                    massreco_chosen_pair_wrong_jet_pt_second])
        massreco_chosen_pair_correct_jet_eta_1.append([massreco_chosen_pair_correct_jet_eta,
                                                       massreco_chosen_pair_correct_jet_eta_second])
        massreco_chosen_pair_wrong_jet_eta_1.append([massreco_chosen_pair_wrong_jet_eta,
                                                     massreco_chosen_pair_wrong_jet_eta_second])
        massreco_chosen_pair_correct_jet_phi_1.append([massreco_chosen_pair_correct_jet_phi,
                                                       massreco_chosen_pair_correct_jet_phi_second])
        massreco_chosen_pair_wrong_jet_phi_1.append([massreco_chosen_pair_wrong_jet_phi,
                                                     massreco_chosen_pair_wrong_jet_phi_second])
        massreco_chosen_pair_correct_jet_btag_score_1.append([massreco_chosen_pair_correct_jet_btag_score,
                                                               massreco_chosen_pair_correct_jet_btag_score_second])
        massreco_chosen_pair_wrong_jet_btag_score_1.append([massreco_chosen_pair_wrong_jet_btag_score,
                                                             massreco_chosen_pair_wrong_jet_btag_score_second])
        massreco_chosen_pair_correct_jet_mult_1.append([massreco_chosen_pair_correct_jet_mult])
        massreco_chosen_pair_wrong_jet_mult_1.append([massreco_chosen_pair_wrong_jet_mult])
        massreco_chosen_pair_correct_dr_1.append([massreco_chosen_pair_correct_dr])
        massreco_chosen_pair_wrong_dr_1.append([massreco_chosen_pair_wrong_dr])
        massreco_chosen_pair_correct_delta_angle_1.append([massreco_chosen_pair_correct_delta_angle])
        massreco_chosen_pair_wrong_delta_angle_1.append([massreco_chosen_pair_wrong_delta_angle])
        massreco_chosen_pair_correct_delta_magn_1.append([massreco_chosen_pair_correct_delta_magn])
        massreco_chosen_pair_wrong_delta_magn_1.append([massreco_chosen_pair_wrong_delta_magn])
        massreco_chosen_pair_correct_pull_angle_1.append([massreco_chosen_pair_correct_pull_angle])
        massreco_chosen_pair_wrong_pull_angle_1.append([massreco_chosen_pair_wrong_pull_angle])
        massreco_chosen_pair_correct_pull_magn_1.append([massreco_chosen_pair_correct_pull_magn])
        massreco_chosen_pair_wrong_pull_magn_1.append([massreco_chosen_pair_wrong_pull_magn])
        massreco_chosen_pair_correct_dr_top_1.append([massreco_chosen_pair_correct_dr_top])
        massreco_chosen_pair_wrong_dr_top_1.append([massreco_chosen_pair_wrong_dr_top])
        massreco_chosen_pair_correct_lep_pos_1.append([massreco_chosen_pair_correct_lep_pos])
        massreco_chosen_pair_wrong_lep_pos_1.append([massreco_chosen_pair_wrong_lep_pos])
        massreco_chosen_pair_correct_lep_neg_1.append([massreco_chosen_pair_correct_lep_neg])
        massreco_chosen_pair_wrong_lep_neg_1.append([massreco_chosen_pair_wrong_lep_neg])
        massreco_chosen_pair_correct_pt_assymetry_1.append([massreco_chosen_pair_correct_pt_assymetry])
        massreco_chosen_pair_wrong_pt_assymetry_1.append([massreco_chosen_pair_wrong_pt_assymetry])

    # === Block 1: global sanity counters ===
    n_total    = t
    n_skipped  = t - len(correct_matches)
    n_no_reco  = sum(1 for x in correct_matches if len(x) == 0)
    n_correct_2 = sum(1 for x in correct_matches if len(x) > 0 and x[0] == 2.)
    n_correct_1 = sum(1 for x in correct_matches if len(x) > 0 and x[0] == 1.)
    n_correct_0 = sum(1 for x in correct_matches if len(x) > 0 and x[0] == 0.)
    n_with_reco = n_correct_2 + n_correct_1 + n_correct_0

    print(f"\n=== compute_jet_pair_properties_all_events ===")
    print(f"  Total events           : {n_total}")
    print(f"  Skipped (no truth/btag): {n_skipped}")
    print(f"  Reco had no solution   : {n_no_reco}")
    print(f"  correct_match == 2     : {n_correct_2}  (both Higgs jets right)")
    print(f"  correct_match == 1     : {n_correct_1}  (one Higgs jet right)")
    print(f"  correct_match == 0     : {n_correct_0}  (neither Higgs jet right)")
    print(f"  Efficiency (match==2)  : {n_correct_2 / max(1, n_with_reco):.3f}")

    # Cross-check: correct_higgs_mass filled iff match==2, wrong_higgs_mass filled iff match==0
    n_correct_mass_filled = sum(1 for x in massreco_chosen_pair_correct_higgs_mass_1
                                if len(x) > 0 and x[0] is not None)
    n_wrong_mass_filled   = sum(1 for x in massreco_chosen_pair_wrong_higgs_mass_1
                                if len(x) > 0 and x[0] is not None)
    n_correct_0_with_reco = sum(
        1 for i, x in enumerate(correct_matches)
        if len(x) > 0 and x[0] == 0.
        and len(massreco_chosen_pair_wrong_higgs_mass_1[i]) > 0
        and massreco_chosen_pair_wrong_higgs_mass_1[i][0] is not None
    )
    n_correct_0_no_reco = n_correct_0 - n_correct_0_with_reco

    print(f"  correct_higgs_mass filled : {n_correct_mass_filled}  (should == n_correct_2={n_correct_2})")
    print(f"  wrong_higgs_mass filled   : {n_wrong_mass_filled}  "
          f"(match==0 with reco={n_correct_0_with_reco}, match==0 no reco={n_correct_0_no_reco}, "
          f"total match==0={n_correct_0})")

    # === Block 4: pull angle label sanity check ===
    # Expected behaviour per match case:
    #   correct_match==2 : correct_pull filled (both values), wrong_pull = [nan, nan]
    #   correct_match==1 : correct_pull filled (one value + nan), wrong_pull has one nan — OK
    #   correct_match==0 : wrong_pull filled (both values) if reco jets in bounds, else nan is acceptable
    pull_warn_count = 0
    for i, (cm, cpm, wpm) in enumerate(zip(
            correct_matches,
            massreco_chosen_pair_correct_pull_magn_1,
            massreco_chosen_pair_wrong_pull_magn_1)):
        if len(cm) == 0:
            continue

        def has_any_value(lst):
            return (len(lst) > 0 and len(lst[0]) >= 1
                    and not all(np.isnan(float(v)) for v in lst[0] if v is not None))

        def both_values(lst):
            return (len(lst) > 0 and len(lst[0]) >= 2
                    and not np.isnan(float(lst[0][0]))
                    and not np.isnan(float(lst[0][1])))

        # match==2: correct_pull must have both values; wrong_pull must be all nan
        if cm[0] == 2.:
            if not both_values(cpm):
                print(f"  WARNING event {i}: correct_match==2 but correct_pull_magn missing values — "
                      f"reco Higgs jets likely out of JetGood bounds")
                pull_warn_count += 1
            if has_any_value(wpm):
                print(f"  WARNING event {i}: correct_match==2 but wrong_pull_magn is filled — check Fix E")
                pull_warn_count += 1

        # match==0: wrong_pull should have values if reco jets were in bounds
        # nan is acceptable when original_hj1/hj2 >= n_jets_good (5th jet edge case)
        # so we only warn if wrong_higgs_mass was filled but pull is nan (inconsistency)
        elif cm[0] == 0.:
            wm = massreco_chosen_pair_wrong_higgs_mass_1[i]
            mass_filled = len(wm) > 0 and wm[0] is not None
            if mass_filled and not has_any_value(wpm):
                print(f"  WARNING event {i}: correct_match==0, wrong_higgs_mass filled "
                      f"but wrong_pull_magn is nan — reco jets may be out of JetGood bounds")
                pull_warn_count += 1

    if pull_warn_count == 0:
        print(f"  Pull magn label check     : OK (no warnings)")
    else:
        print(f"  Pull magn label check     : {pull_warn_count} warnings — inspect above")
    print(f"==============================================\n")

    props = {
        "pair_phi": pair_phi,
        "pair_eta": pair_eta,
        "pair_mass": pair_mass,
        "pair_pt": pair_pt,
        "pair_indices": pair_indices,
        "pair_dr": pair_dr,
        "numerators": numerators,
        "denominators": denominators,
        "correct_matches": correct_matches,
        "correct_agreements": correct_agreements,
        "higgs_mass_truth_jets": higgs_mass_truth_jets_1,
        "massreco_chosen_pair_all_higgs_mass": massreco_chosen_pair_all_higgs_mass_1,
        "massreco_chosen_pair_correct_higgs_mass": massreco_chosen_pair_correct_higgs_mass_1,
        "massreco_chosen_pair_wrong_higgs_mass": massreco_chosen_pair_wrong_higgs_mass_1,
        "massreco_chosen_pair_wrong_higgs_mass_2_jets": massreco_chosen_pair_wrong_higgs_mass_2_jets_1,
        "massreco_chosen_pair_wrong_higgs_mass_1_jets": massreco_chosen_pair_wrong_higgs_mass_1_jets_1,
        "massreco_chosen_pair_correct_top_mass": massreco_chosen_pair_correct_top_mass_1,
        "massreco_chosen_pair_wrong_top_mass": massreco_chosen_pair_wrong_top_mass_1,
        "massreco_chosen_pair_correct_W_mass": massreco_chosen_pair_correct_W_mass_1,
        "massreco_chosen_pair_wrong_W_mass": massreco_chosen_pair_wrong_W_mass_1,
        "massreco_chosen_pair_correct_ttH_mass": massreco_chosen_pair_correct_ttH_mass_1,
        "massreco_chosen_pair_wrong_ttH_mass": massreco_chosen_pair_wrong_ttH_mass_1,
        "massreco_chosen_pair_correct_jet_pt": massreco_chosen_pair_correct_jet_pt_1,
        "massreco_chosen_pair_wrong_jet_pt": massreco_chosen_pair_wrong_jet_pt_1,
        "massreco_chosen_pair_correct_jet_eta": massreco_chosen_pair_correct_jet_eta_1,
        "massreco_chosen_pair_wrong_jet_eta": massreco_chosen_pair_wrong_jet_eta_1,
        "massreco_chosen_pair_correct_jet_phi": massreco_chosen_pair_correct_jet_phi_1,
        "massreco_chosen_pair_wrong_jet_phi": massreco_chosen_pair_wrong_jet_phi_1,
        "massreco_chosen_pair_correct_jet_mult": massreco_chosen_pair_correct_jet_mult_1,
        "massreco_chosen_pair_wrong_jet_mult": massreco_chosen_pair_wrong_jet_mult_1,
        "massreco_chosen_pair_correct_dr": massreco_chosen_pair_correct_dr_1,
        "massreco_chosen_pair_wrong_dr": massreco_chosen_pair_wrong_dr_1,
        "massreco_chosen_pair_correct_delta_angle": massreco_chosen_pair_correct_delta_angle_1,
        "massreco_chosen_pair_wrong_delta_angle": massreco_chosen_pair_wrong_delta_angle_1,
        "massreco_chosen_pair_correct_pull_magn": massreco_chosen_pair_correct_pull_magn_1,
        "massreco_chosen_pair_wrong_pull_magn": massreco_chosen_pair_wrong_pull_magn_1,
        "massreco_chosen_pair_correct_pull_angle": massreco_chosen_pair_correct_pull_angle_1,
        "massreco_chosen_pair_wrong_pull_angle": massreco_chosen_pair_wrong_pull_angle_1,
        "massreco_chosen_pair_correct_delta_magn": massreco_chosen_pair_correct_delta_magn_1,
        "massreco_chosen_pair_wrong_delta_magn": massreco_chosen_pair_wrong_delta_magn_1,
        "massreco_chosen_pair_correct_dr_top": massreco_chosen_pair_correct_dr_top_1,
        "massreco_chosen_pair_wrong_dr_top": massreco_chosen_pair_wrong_dr_top_1,
        "massreco_chosen_pair_correct_lep_pos": massreco_chosen_pair_correct_lep_pos_1,
        "massreco_chosen_pair_wrong_lep_pos": massreco_chosen_pair_wrong_lep_pos_1,
        "massreco_chosen_pair_correct_lep_neg": massreco_chosen_pair_correct_lep_neg_1,
        "massreco_chosen_pair_wrong_lep_neg": massreco_chosen_pair_wrong_lep_neg_1,
        "massreco_chosen_pair_correct_pt_assymetry": massreco_chosen_pair_correct_pt_assymetry_1,
        "massreco_chosen_pair_wrong_pt_assymetry": massreco_chosen_pair_wrong_pt_assymetry_1,
        "massreco_chosen_pair_correct_jet_btag_score": massreco_chosen_pair_correct_jet_btag_score_1,
        "massreco_chosen_pair_wrong_jet_btag_score": massreco_chosen_pair_wrong_jet_btag_score_1,
    }
    return props


def compute_pair_kinematics_simple(jet1, jet2):
    # Extract properties
    pt1, eta1, phi1, mass1 = float(jet1.pt), float(jet1.eta), float(jet1.phi), float(jet1.mass)
    pt2, eta2, phi2, mass2 = float(jet2.pt), float(jet2.eta), float(jet2.phi), float(jet2.mass)

    # ΔR
    dr = calculate_dr(eta1, phi1, eta2, phi2)

    # Invariant mass (using 4-vector components)
    px1, py1, pz1 = pt1 * np.cos(phi1), pt1 * np.sin(phi1), pt1 * np.sinh(eta1)
    px2, py2, pz2 = pt2 * np.cos(phi2), pt2 * np.sin(phi2), pt2 * np.sinh(eta2)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + mass1**2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + mass2**2)

    px, py, pz, e = px1 + px2, py1 + py2, pz1 + pz2, e1 + e2
    mass = np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0.0))

    # Average φ and η (non-periodic average for φ is ok here since jets are close)
    phi_avg = 0.5 * (phi1 + phi2)
    eta_avg = 0.5 * (eta1 + eta2)

    return phi_avg, eta_avg, mass, dr


def compute_jet_pair_properties_all_events_4(events, j_matched, q_matched, btag_branch="btagUParTAK4B"):
    """
    Diagnostic function with three goals:

    Goal 1 — Acceptance study
        pair_in_the_4leading      : are both truth Higgs jets among the top-4 pt-sorted JetGood jets?
        pair_in_the_4leadingbtag  : are both truth Higgs jets among the top-4 btag-scored JetGood jets?
        Codes: 0 = truth pair not found, 1 = found but NOT both in top-4, 2 = both in top-4

    Goal 2 — Mass reco outcome classification
        mass_reco_had_a_solution:
            0 = no truth pair AND no reco solution
            1 = truth pair found, reco had NO solution
            2 = truth pair found, reco had solution, both truth jets in top-4 btag
            3 = truth pair found, reco had solution, truth jets NOT in top-4 btag
            4 = no truth pair, reco had a solution
            5 = truth pair found, reco had solution, truth jets in top-4 btag, reco got it RIGHT
            6 = truth pair found, reco had solution, truth jets in top-4 btag, reco got it WRONG

    Goal 3 — Truth-pair kinematics
        pair_phi_truth, pair_eta_truth, pair_mass_truth, pair_dr_truth, pair_indices_truth
        Always filled when truth pair exists, regardless of reco outcome.
        Lets you compare kinematic properties across reco outcome categories.
    """

    pair_phi_truth      = []
    pair_eta_truth      = []
    pair_mass_truth     = []
    pair_dr_truth       = []
    pair_indices_truth  = []
    pair_in_the_4leading         = []
    pair_in_the_4leadingbtagscore = []
    mass_reco_had_a_solution     = []

    prefix = "max_weight"

    for event_idx in range(len(events)):

        jets = events["JetGood"][event_idx]
        n_jets_good = len(events["JetGood"]["pt"][event_idx])

        # --- Step 1: find truth-matched Higgs jet indices ---
        truth_pair = [
            idx for idx, parton in enumerate(q_matched[event_idx])
            if parton is not None and parton.provenance == 1
        ]
        has_truth = len(truth_pair) == 2
        reco_combination = events[f"{prefix}_combination"][event_idx]
        has_reco = reco_combination is not None

        # --- Goal 3: truth-pair kinematics (always, when truth exists) ---
        if has_truth:
            jet0_idx, jet1_idx = truth_pair
            jet1 = jets[jet0_idx]
            jet2 = jets[jet1_idx]
            d_eta, d_phi, mass, dr = compute_pair_kinematics_simple(jet1, jet2)
            pair_phi_truth.append([d_phi])
            pair_eta_truth.append([d_eta])
            pair_mass_truth.append([mass])
            pair_dr_truth.append([dr])
            pair_indices_truth.append([truth_pair])
        else:
            pair_phi_truth.append([])
            pair_eta_truth.append([])
            pair_mass_truth.append([])
            pair_dr_truth.append([])
            pair_indices_truth.append([])

        # --- Goal 1: acceptance study (diagnostics only, both orderings) ---
        # Check 1a: are truth jets among the top-4 pT-sorted JetGood jets?
        if has_truth:
            pt_scores = ak.to_numpy(events["JetGood"]["pt"][event_idx])
            top4_pt_idx = np.argsort(-pt_scores)[:4]
            in_top4_pt = (truth_pair[0] in top4_pt_idx and truth_pair[1] in top4_pt_idx)
            pair_in_the_4leading.append([2 if in_top4_pt else 1])
        else:
            pair_in_the_4leading.append([0])

        # Check 1b: are truth jets among the top-4 btag-scored JetGood jets?
        btag_scores = ak.to_numpy(events["JetGood"][btag_branch][event_idx])
        if len(btag_scores) >= 4:
            top4_btag_idx = np.argsort(-btag_scores)[:4]
            if has_truth:
                in_top4_btag = (truth_pair[0] in top4_btag_idx
                                and truth_pair[1] in top4_btag_idx)
                pair_in_the_4leadingbtagscore.append([2 if in_top4_btag else 1])
            else:
                pair_in_the_4leadingbtagscore.append([0])
        else:
            # fewer than 4 jets: truth jets cannot both be in top-4
            pair_in_the_4leadingbtagscore.append([0])

        # --- Acceptance gate for Goal 2: were the truth jets among the
        #     jets the solver actually received? This uses events["1st_jet_idx"
        #     .."4th_jet_idx"] -- whichever ordering criterion is active in
        #     workflow_mc.py (pT now, previously btag) -- instead of a
        #     hardcoded btag re-derivation, so it stays consistent with
        #     whatever the solver could actually see. (pair_in_the_4leading /
        #     pair_in_the_4leadingbtagscore above remain independent,
        #     ordering-specific diagnostics for comparison.)
        solver_jet_indices = get_solver_jet_indices(events, event_idx)
        in_solver_candidates = has_truth and (
            truth_pair[0] in solver_jet_indices and truth_pair[1] in solver_jet_indices
        )

        # --- Goal 2: mass reco outcome classification ---
        if not has_truth and not has_reco:
            mass_reco_had_a_solution.append([0])  # neither

        elif has_truth and not has_reco:
            mass_reco_had_a_solution.append([1])  # truth found, reco failed

        elif not has_truth and has_reco:
            mass_reco_had_a_solution.append([4])  # no truth, reco found

        elif has_truth and has_reco:
            if not in_solver_candidates:
                # truth jets weren't both given to the solver: it could not have found them
                mass_reco_had_a_solution.append([3])
            else:
                # truth jets were both given to the solver: check if reco got it right
                hj1 = reco_combination["2"]
                hj2 = reco_combination["3"]
                original_hj1 = get_original_jet_idx(events, event_idx, hj1)
                original_hj2 = get_original_jet_idx(events, event_idx, hj2)

                if original_hj1 is None or original_hj2 is None:
                    # 5th jet edge case
                    mass_reco_had_a_solution.append([3])
                else:
                    reco_correct = (
                        (truth_pair[0] == original_hj1 and truth_pair[1] == original_hj2) or
                        (truth_pair[0] == original_hj2 and truth_pair[1] == original_hj1)
                    )
                    if reco_correct:
                        mass_reco_had_a_solution.append([5])  # reco got it right
                    else:
                        mass_reco_had_a_solution.append([6])  # reco got it wrong

        # Note: code [2] from original (truth in top-4, reco found) is now split
        # into [5] (correct) and [6] (wrong) for finer discrimination.
        # If you want the old coarse code, merge [5]+[6] → [2] downstream.

    # --- Summary printout ---
    counts = {}
    for x in mass_reco_had_a_solution:
        key = x[0] if len(x) > 0 else "empty"
        counts[key] = counts.get(key, 0) + 1

    total = len(events)
    print(f"\n=== compute_jet_pair_properties_all_events_4 ===")
    print(f"  Total events : {total}")
    print(f"  Code 0 (no truth, no reco)                       : {counts.get(0,0)}  ({100*counts.get(0,0)/max(1,total):.1f}%)")
    print(f"  Code 1 (truth found, reco FAILED)                : {counts.get(1,0)}  ({100*counts.get(1,0)/max(1,total):.1f}%)")
    print(f"  Code 3 (truth found, truth NOT in solver's jets) : {counts.get(3,0)}  ({100*counts.get(3,0)/max(1,total):.1f}%)")
    print(f"  Code 4 (no truth, reco found)                    : {counts.get(4,0)}  ({100*counts.get(4,0)/max(1,total):.1f}%)")
    print(f"  Code 5 (truth in solver's jets, reco CORRECT)    : {counts.get(5,0)}  ({100*counts.get(5,0)/max(1,total):.1f}%)")
    print(f"  Code 6 (truth in solver's jets, reco WRONG)      : {counts.get(6,0)}  ({100*counts.get(6,0)/max(1,total):.1f}%)")

    n_truth_in_top4 = counts.get(1,0) + counts.get(5,0) + counts.get(6,0)
    n_reco_attempted = counts.get(5,0) + counts.get(6,0)
    print(f"  --- Acceptance ---")
    print(f"  Truth jets available to solver : {n_truth_in_top4}/{total}  ({100*n_truth_in_top4/max(1,total):.1f}%)")
    print(f"  Reco efficiency (of events with truth available to solver and a reco solution) : "
          f"{counts.get(5,0)}/{max(1,n_reco_attempted)}  "
          f"({100*counts.get(5,0)/max(1,n_reco_attempted):.1f}%)")

    n_in_top4_pt   = sum(1 for x in pair_in_the_4leading if len(x) > 0 and x[0] == 2)
    n_in_top4_btag = sum(1 for x in pair_in_the_4leadingbtagscore if len(x) > 0 and x[0] == 2)
    print(f"  Truth pair in top-4 by pT  : {n_in_top4_pt}/{total}  ({100*n_in_top4_pt/max(1,total):.1f}%)")
    print(f"  Truth pair in top-4 by btag: {n_in_top4_btag}/{total}  ({100*n_in_top4_btag/max(1,total):.1f}%)")
    print(f"================================================\n")

    param = {
        "pair_in_the_4leading":          pair_in_the_4leading,
        "pair_in_the_4leadingbtagscore": pair_in_the_4leadingbtagscore,
        "mass_reco_had_a_solution":      mass_reco_had_a_solution,
        "pair_phi_truth":                pair_phi_truth,
        "pair_eta_truth":                pair_eta_truth,
        "pair_mass_truth":               pair_mass_truth,
        "pair_dr_truth":                 pair_dr_truth,
        "pair_indices_truth":            pair_indices_truth,
    }
    return param

def compute_jet_pair_properties_per_rank(events, j_matched, q_matched):
    """
    For each event, runs truth matching for ALL four ranks independently,
    regardless of which rank was selected by the DR criterion.
    Outputs per-rank correct_match and Higgs mass for correct/wrong cases.

    This is used to answer:
    - "When rank N was rejected, would it have been correct?"
    - "What is the unconditional efficiency of each rank?"
    """

    RANK_PREFIXES = [
        "max_weight",
        "second_max_weight",
        "third_max_weight",
        "fourth_max_weight",
    ]
    RANK_NAMES = ["rank1", "rank2", "rank3", "rank4"]

    # Initialise output lists per rank
    correct_match_per_rank   = {r: [] for r in RANK_NAMES}
    higgs_mass_correct_per_rank = {r: [] for r in RANK_NAMES}
    higgs_mass_wrong_per_rank   = {r: [] for r in RANK_NAMES}

    t = len(events)

    for event_idx in range(t):

        # --- Step 1: find truth-matched Higgs jet indices ---
        higgs_matched_indices = [
            idx for idx, parton in enumerate(q_matched[event_idx])
            if parton is not None and parton.provenance == 1
        ]
        selected_jet_pair = higgs_matched_indices if len(higgs_matched_indices) == 2 else []

        # If no valid truth pair, append None for all ranks
        if len(selected_jet_pair) != 2:
            for r in RANK_NAMES:
                correct_match_per_rank[r].append(None)
                higgs_mass_correct_per_rank[r].append(None)
                higgs_mass_wrong_per_rank[r].append(None)
            continue

        # --- Step 2: require the solver actually had >=4 jet candidates ---
        solver_jet_indices = get_solver_jet_indices(events, event_idx)
        if len(solver_jet_indices) < 4:
            for r in RANK_NAMES:
                correct_match_per_rank[r].append(None)
                higgs_mass_correct_per_rank[r].append(None)
                higgs_mass_wrong_per_rank[r].append(None)
            continue

        # --- Step 3: require both truth jets are among the jets the solver
        #     actually received (pT-ordered now, previously btag) ---
        jet0, jet1 = selected_jet_pair
        if not (jet0 in solver_jet_indices and jet1 in solver_jet_indices):
            for r in RANK_NAMES:
                correct_match_per_rank[r].append(None)
                higgs_mass_correct_per_rank[r].append(None)
                higgs_mass_wrong_per_rank[r].append(None)
            continue

        n_jets_good = len(events["JetGood"]["eta"][event_idx])

        # --- Loop over all four ranks ---
        for prefix, rank_name in zip(RANK_PREFIXES, RANK_NAMES):

            reco_combination = events[f"{prefix}_combination"][event_idx]

            if reco_combination is None:
                correct_match_per_rank[rank_name].append(None)
                higgs_mass_correct_per_rank[rank_name].append(None)
                higgs_mass_wrong_per_rank[rank_name].append(None)
                continue

            hj1 = reco_combination["2"]
            hj2 = reco_combination["3"]

            original_hj1 = get_original_jet_idx(events, event_idx, hj1)
            original_hj2 = get_original_jet_idx(events, event_idx, hj2)

            if original_hj1 is None or original_hj2 is None:
                correct_match_per_rank[rank_name].append(None)
                higgs_mass_correct_per_rank[rank_name].append(None)
                higgs_mass_wrong_per_rank[rank_name].append(None)
                continue

            # Determine match
            first_match  = 1. if jet0 == original_hj1 or jet1 == original_hj1 else 0.
            second_match = 1. if jet0 == original_hj2 or jet1 == original_hj2 else 0.

            if first_match == 1. and second_match == 1.:
                correct_match = 2.
            elif first_match == 1. or second_match == 1.:
                correct_match = 1.
            else:
                correct_match = 0.

            higgs_mass = events[f"{prefix}_higgs_mass"][event_idx]

            correct_match_per_rank[rank_name].append(correct_match)
            higgs_mass_correct_per_rank[rank_name].append(higgs_mass if correct_match == 2. else None)
            higgs_mass_wrong_per_rank[rank_name].append(higgs_mass if correct_match < 2. else None)

    # --- Summary ---
    print(f"\n=== compute_jet_pair_properties_per_rank ===")
    for rank_name in RANK_NAMES:
        cm = correct_match_per_rank[rank_name]
        n2 = sum(1 for x in cm if x == 2.)
        n1 = sum(1 for x in cm if x == 1.)
        n0 = sum(1 for x in cm if x == 0.)
        nt = n2 + n1 + n0
        print(f"  {rank_name}: correct={n2}/{nt} ({100*n2/max(1,nt):.1f}%), "
              f"partial={n1}, wrong={n0}")
    print(f"==============================================\n")

    res = {}
    for rank_name in RANK_NAMES:
        res[f"correct_match_{rank_name}"]        = correct_match_per_rank[rank_name]
        res[f"higgs_mass_{rank_name}_correct"]   = higgs_mass_correct_per_rank[rank_name]
        res[f"higgs_mass_{rank_name}_wrong"]     = higgs_mass_wrong_per_rank[rank_name]
    return res


def compute_selected_rejected_correct_wrong(events, props, props_per_rank):
    """
    Combines rank_used, correct_match_dr, and per-rank correct_match to produce
    per-rank mass variables split by selected/rejected and correct/wrong.
 
    Outputs (18 variables):
    For ranks 1-4:
        higgs_mass_rankN_selected_correct  — selected by DR criterion AND correct_match==2
        higgs_mass_rankN_selected_wrong    — selected by DR criterion AND correct_match!=2
        higgs_mass_rankN_rejected_correct  — rejected by DR criterion AND rank would be correct
        higgs_mass_rankN_rejected_wrong    — rejected by DR criterion AND rank would be wrong
    For fallback (rank==0):
        higgs_mass_fallback_selected_correct
        higgs_mass_fallback_selected_wrong
    """
    t = len(events)
 
    rank_used_list    = ak.to_list(events["rank_used"])
    correct_match_dr  = ak.to_list(ak.firsts(ak.Array(props["correct_matches"])))
    mass_chosen       = ak.to_list(events["higgs_mass_chosen"])
 
    cm_rank = {
        1: ak.to_list(ak.Array(props_per_rank["correct_match_rank1"])),
        2: ak.to_list(ak.Array(props_per_rank["correct_match_rank2"])),
        3: ak.to_list(ak.Array(props_per_rank["correct_match_rank3"])),
        4: ak.to_list(ak.Array(props_per_rank["correct_match_rank4"])),
    }
    mass_rank = {
        1: ak.to_list(events["higgs_mass_rank1"]),
        2: ak.to_list(events["higgs_mass_rank2"]),
        3: ak.to_list(events["higgs_mass_rank3"]),
        4: ak.to_list(events["higgs_mass_rank4"]),
    }
 
    # Initialise output lists
    sel_correct = {r: [] for r in [0, 1, 2, 3, 4]}
    sel_wrong   = {r: [] for r in [0, 1, 2, 3, 4]}
    rej_correct = {r: [] for r in [1, 2, 3, 4]}
    rej_wrong   = {r: [] for r in [1, 2, 3, 4]}
 
    n_sel_correct = {r: 0 for r in [0,1,2,3,4]}
    n_sel_wrong   = {r: 0 for r in [0,1,2,3,4]}
    n_rej_correct = {r: 0 for r in [1,2,3,4]}
    n_rej_wrong   = {r: 0 for r in [1,2,3,4]}
 
    for i in range(t):
        ru    = rank_used_list[i]
        cm_dr = correct_match_dr[i]
        mc    = mass_chosen[i]
 
        # Selected cases — only the rank that was actually chosen
        for r in [0, 1, 2, 3, 4]:
            if ru == r:
                if cm_dr == 2.:
                    sel_correct[r].append(mc)
                    n_sel_correct[r] += 1
                    sel_wrong[r].append(None)
                else:
                    sel_correct[r].append(None)
                    sel_wrong[r].append(mc)
                    n_sel_wrong[r] += 1
            else:
                sel_correct[r].append(None)
                sel_wrong[r].append(None)
 
        # Rejected cases — all ranks that were NOT chosen
        for r in [1, 2, 3, 4]:
            if ru != r and ru is not None:
                rank_cm = cm_rank[r][i]
                mr      = mass_rank[r][i]
                if rank_cm == 2.:
                    rej_correct[r].append(mr)
                    n_rej_correct[r] += 1
                    rej_wrong[r].append(None)
                elif rank_cm is not None:
                    rej_correct[r].append(None)
                    rej_wrong[r].append(mr)
                    n_rej_wrong[r] += 1
                else:
                    rej_correct[r].append(None)
                    rej_wrong[r].append(None)
            else:
                rej_correct[r].append(None)
                rej_wrong[r].append(None)
 
    # Summary
    print(f"\n=== compute_selected_rejected_correct_wrong ===")
    rank_names = {0: 'fallback', 1: 'rank1', 2: 'rank2', 3: 'rank3', 4: 'rank4'}
    for r in [1, 2, 3, 4]:
        print(f"  {rank_names[r]}:")
        print(f"    selected correct={n_sel_correct[r]}, wrong={n_sel_wrong[r]}")
        print(f"    rejected correct={n_rej_correct[r]}, wrong={n_rej_wrong[r]}")
    print(f"  fallback: selected correct={n_sel_correct[0]}, wrong={n_sel_wrong[0]}")
    print(f"==============================================\n")
 
    res = {}
    for r in [1, 2, 3, 4]:
        res[f"higgs_mass_rank{r}_selected_correct"] = sel_correct[r]
        res[f"higgs_mass_rank{r}_selected_wrong"]   = sel_wrong[r]
        res[f"higgs_mass_rank{r}_rejected_correct"] = rej_correct[r]
        res[f"higgs_mass_rank{r}_rejected_wrong"]   = rej_wrong[r]
    res["higgs_mass_fallback_selected_correct"] = sel_correct[0]
    res["higgs_mass_fallback_selected_wrong"]   = sel_wrong[0]
    return res