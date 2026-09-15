# ttH Neutrino Closure

Truth-level closure tests for the ttH(bb) neutrino/lost-jet reconstruction algebra, in both the semileptonic (SL) and dileptonic (DL) channels. The question this repo answers: given the TRUE four-momenta of every visible particle in a simulated event (no detector effects, no jet mis-assignment from real data), does the analytic Sonnenschein-style quartic solver recover the true neutrino (and, in the SL channel, the true "lost" quark) momentum when it's handed the correct jet pairing?

The pipeline has two stages: **generation** (Pythia8 truth-level event samples) and **solving** (the actual reconstruction algebra, run over those truth events, plus the diagnostics that check it).

## Repository layout

```
generation/
    pythiattH_sl.C          ROOT/Pythia8 macro -> pythiaOutput_ttH_sl.root  (SL truth events)
    pythiattH_dilepton.C    ROOT/Pythia8 macro -> pythiaOutput_ttH.root     (DL truth events)

solving/
    mass_reco_functions_sl.py   SL solver library (quartic solver + SL-specific coefficient algebra)
    mass_reco_functions_dl.py   DL solver library (quartic solver + DL-specific coefficient algebra)
    solve_sl_pythia_closure_allsol.py   Runs the SL solver over the truth sample -> closure_sl_allsolutions.root
    solve_dl_pythia_closure_allsol.py   Runs the DL solver over the truth sample -> closure_dl_allsolutions.root
    n_roots_per_hypothesis.py   Diagnostic: real-root-multiplicity distribution of the quartic solver
    higgs_mass_overlay.py       Diagnostic: Higgs mass, true vs. reco (winner) vs. reco (all candidates)
    CT10/                       Bundled CT10 PDF grid (LHAPDF-format), used to weight candidate solutions
```

Branches: `main` is active development; `v0` is a frozen snapshot of the repo's initial state.

## The physics

Each event has a hadronic top, a leptonic top, and an H -> bb. In the **SL** channel, one W decays leptonically (giving a real, invisible neutrino) and the other hadronically; SL additionally treats one hadronic-W daughter quark as "lost" (its momentum folded into MET, exactly like a second invisible particle) so that both channels reduce to the same kind of problem: reconstruct one or two invisible momenta from missing transverse energy plus on-shell mass constraints. In the **DL** channel, both tops decay leptonically, giving two real neutrinos.

Both channels solve the same way: for a fixed hypothesis (which jets play which role, plus a trial top mass `mt` and W mass `mW`), the W-mass-shell and top-mass-shell constraints combine into a single quartic polynomial in the invisible particle's x-momentum. `quartic_solver()` (Ferrari's method: depressed quartic -> resolvent cubic -> two quadratic factors) returns the real roots; each root fixes the rest of the invisible kinematics via the MET constraint.

The full search space per event is: every jet-pairing permutation (12 orderings of the 4 non-Higgs-candidate jets into "leptonic-side" and "hadronic-side" roles) x a grid scan over `(mt, mW)` x the real roots of the quartic at each grid point. Every surviving candidate gets a parton-luminosity weight from the CT10 PDF set (via the candidate's implied `x1, x2` momentum fractions); the highest-weight candidate per event is the algorithm's actual pick.

### What each solve script writes

`solve_sl_pythia_closure_allsol.py` / `solve_dl_pythia_closure_allsol.py` write **`closure_{sl,dl}_allsolutions.root`**, containing two trees:

- **`closure`** -- one row per (event, surviving candidate), "long format". Key branches: `eventNumber`, `weight`, `mt_hyp`, `mW_hyp`, `correct_pairing` (was this candidate's jet assignment the true one?), `n_solutions` (candidates found for that event), `is_best` (1 for the per-event argmax-weight winner, else 0), the true and reconstructed invisible momentum components (`nu_px_true/reco`, ... for SL; `nu1_*`, `nu2_*` for DL), and `higgs_mass_true` / `higgs_mass_reco`. Filtering `is_best == 1` reproduces exactly what a "keep only the winner" pipeline would have written -- this repo intentionally keeps every candidate instead, so the bias in what the algorithm rejects can be compared to the bias in what it picks.
- **`hyp_roots`** -- one row per (event, jet-pairing, mt, mW) hypothesis point that had at least one real root, branch `n_roots`: the real-root count of the quartic **with multiplicity** (always 0, 2, or 4 for a genuine quartic -- see `count_quartic_real_roots()` below). This is a different question from "how many candidates survived," which is what `n_solutions` answers.

Both solve scripts also print a handful of worked examples to the console: for the first few events with at least one correctly-paired candidate, they show that `higgs_mass_reco` for every correctly-paired candidate is numerically identical to `higgs_mass_true` (expected -- the Higgs mass is built purely from which two jets were picked as "Higgs," independent of `mt`/`mW`/the solved root), plus the closest-matching correctly-paired neutrino solution vs. the true neutrino, as a sanity check that the algebra actually recovers truth when given the right jet assignment.

### The solver library (`mass_reco_functions_{sl,dl}.py`)

Both files share the same generic quartic machinery:

- `quadratic_solver`, `cubic_solver`, `quartic_solver` -- Ferrari's method. `quartic_solver` returns the **distinct** real roots (a genuine double root is returned once, not twice), which is the correct behavior for generating candidates: a duplicated candidate would carry no extra information.
- `count_quartic_real_roots` -- a separate function, added specifically for the `hyp_roots` diagnostic, that returns the real root count **with multiplicity** using the same discriminant tests `quartic_solver` makes internally. It does not touch or affect `quartic_solver`'s own candidate list. Validated against `numpy.roots` ground truth over 200,000 random quartics plus explicit double-root constructions.
- `algebraic_pz` -- fallback longitudinal-momentum solver for the rare case where the standard linear formula is numerically singular.
- SL-specific: `_leg_a_geometry`, `_leg_a_scan_dependent`, `_leg_b_geometry`, `_leg_b_scan_dependent`, `_dp_family_with_mass`, `_combine_to_quartic` build the quartic's coefficients from the leptonic and hadronic leg kinematics, generalized to a lost quark of nonzero mass.
- DL-specific: `compute_coefficients` does the analogous coefficient construction for the two-neutrino system.
- `_pdf_set` -- loaded once at import time via `parton.mkPDF("CT10", 0, pdfdir=".")`, reading the bundled `CT10/` grid.

**A note on the quartic solver's history**: the general-case branch solves a resolvent cubic for `h^2`, and any *one* valid positive root of that cubic already reconstructs all 4 roots of the original quartic. An earlier version of this code looped over *every* positive root the resolvent cubic returned (up to 3) and accumulated candidates from each, which -- due to floating-point differences between the redundant reconstructions -- could produce spurious "5th/6th/7th/8th root" artifacts that a naive dedup tolerance didn't always catch. The fix was a single `break` after the first valid branch. This is why `hyp_roots` should show only 2s and 4s (plus, extremely rarely, exact double-root degeneracies).

## Diagnostics

- **`n_roots_per_hypothesis.py {sl|dl}`** -- reads the `hyp_roots` tree, prints the real-root-multiplicity breakdown (should be ~88%/12% split between 2 and 4 roots) with mean/RMS, and saves `{channel}_n_roots_per_hypothesis.png`.
- **`higgs_mass_overlay.py {sl|dl}`** -- reads the `closure` tree, prints mean/median/std of the Higgs mass for (a) truth, (b) the max-weight winner, and (c) every candidate, and saves `{channel}_higgs_mass_overlay.png` with all three overlaid as normalized density histograms.

## How to run

**1. Generate the truth samples** (from `generation/`, needs ROOT with Pythia8 support):

```
root -l -q pythiattH_sl.C          # -> pythiaOutput_ttH_sl.root
root -l -q pythiattH_dilepton.C    # -> pythiaOutput_ttH.root
```

**2. Run the solvers** (from `solving/`; needs `uproot`, `numpy`, `awkward`, `numba`, `vector`, and `parton`):

```
python3 solve_sl_pythia_closure_allsol.py    # -> closure_sl_allsolutions.root
python3 solve_dl_pythia_closure_allsol.py    # -> closure_dl_allsolutions.root
```

Each processes the first `N_EVENTS` events in its input file (2000 by default -- edit the constant near the top of the script to change it; check runtime/output size before jumping straight to the full sample, since this scans a full `(mt, mW)` grid times all jet-pairing permutations times every quartic root per event).

**3. Run the diagnostics** (from `solving/`, after step 2):

```
python3 n_roots_per_hypothesis.py sl
python3 n_roots_per_hypothesis.py dl
python3 higgs_mass_overlay.py sl
python3 higgs_mass_overlay.py dl
```

## Repo history notes

Two earlier "winner-only" solve scripts (`solve_sl_pythia_closure.py`, `solve_dl_pythia_closure.py`) and several earlier plotting/diagnostic scripts were removed as redundant: filtering the `closure` tree's `is_best == 1` rows reproduces exactly what those winner-only scripts wrote (same deterministic algebra, same event loop), so keeping both was unnecessary duplication.
