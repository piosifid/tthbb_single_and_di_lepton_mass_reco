#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TH1F.h"
#include "TTree.h"
#include "TRandom3.h"
#include <fstream>
#include <iomanip>
#include "TMath.h"
#include <cstdio>
#include <cstdlib>
#include "TApplication.h"

using namespace Pythia8;
using namespace std;

std::ofstream outFile;

///////////////////////////////////////////////////////////////
// Particle ID List for Reference
//
// Top Quarks:      6 (t), -6 (tbar)
// W Bosons:        24 (W+), -24 (W-)
// Leptons:         11/-11 (e-/e+), 12/-12 (nu_e/nu_e_bar),
//                  13/-13 (mu-/mu+), 14/-14 (nu_mu/nu_mu_bar)
// Light quarks:    1-4 (d,u,s,c) -- the W's hadronic decay products
// Higgs:           25 (H), 5/-5 (b/bbar)
///////////////////////////////////////////////////////////////

void printMomenta(const Event& event, int index) {
    outFile << std::fixed << std::setprecision(15);
    Particle particle = event[index];
    outFile  << particle.name() << ": mass: " << particle.m()
              << ", px: " << particle.px() << ", py: " << particle.py()
              << ", pz: " << particle.pz() << ", E: " << particle.e() << std::endl;
    for (int i = particle.daughter1(); i <= particle.daughter2(); ++i) {
        if (i > 0 && i < event.size()) {
            printMomenta(event, i);
        }
    }
}

void processQuarkAndDecays(const Event& event, int quarkId) {
    for (int i = 0; i < event.size(); ++i) {
        if (event[i].id() == quarkId && event[i].status() == -62) {
            printMomenta(event, i);
        }
    }
}

// ── Semileptonic selection helpers ───────────────────────────────────────
int finalCopy(const Event& event, int idx) {
    while (idx > 0 && event[idx].daughter1() > 0 &&
           event[event[idx].daughter1()].id() == event[idx].id()) {
        idx = event[idx].daughter1();
    }
    return idx;
}

// Classifies one W's decay and returns the indices of its actual daughters.
// For a leptonic W: lepIdx/nuIdx are set, q1Idx/q2Idx stay -1.
// For a hadronic W: q1Idx/q2Idx are set, lepIdx/nuIdx stay -1.
bool classifyW(const Event& event, int Widx, bool &isLeptonic,
                int &lepIdx, int &nuIdx, int &q1Idx, int &q2Idx) {
    lepIdx = nuIdx = q1Idx = q2Idx = -1;
    isLeptonic = false;
    bool sawLepton = false, sawQuark = false;
    for (int d = event[Widx].daughter1(); d <= event[Widx].daughter2(); ++d) {
        if (d <= 0) continue;
        int id = event[d].id();
        int aid = std::abs(id);
        if (aid == 11 || aid == 13) { lepIdx = d; sawLepton = true; }
        else if (aid == 12 || aid == 14) { nuIdx = d; sawLepton = true; }
        else if (aid >= 1 && aid <= 4) {
            if (q1Idx < 0) q1Idx = d; else q2Idx = d;
            sawQuark = true;
        }
    }
    if (sawLepton == sawQuark) return false;  // neither or both -- guard
    isLeptonic = sawLepton;
    return true;
}
// ─────────────────────────────────────────────────────────────────────────

float evMax = 10000;
long attemptsMax = 20 * (long)evMax;

int pythiattH_sl() {

    std::remove("pythiaOutput_ttH_sl.txt");
    std::remove("pythiaOutput_ttH_sl.root");

    Pythia pythia;

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13600.");

    pythia.readString("HiggsSM:gg2Httbar = on ");
    pythia.readString("HiggsSM:qqbar2Httbar = on");

    pythia.readString("6:onMode = off");
    pythia.readString("6:onIfAny = 24");

    // Both leptonic (e/mu + neutrinos) and hadronic (light quarks) W decays
    // enabled -- filtered event-by-event below to keep only one-of-each.
    pythia.readString("24:onMode = off");
    pythia.readString("24:onIfAny = 1 2 3 4 11 12 13 14");

    pythia.readString("25:onMode = off");
    pythia.readString("25:onIfMatch = 5 -5");

    pythia.readString("PartonLevel:ISR = off");
    pythia.readString("PartonLevel:FSR = off");
    pythia.readString("PartonLevel:MPI = off");
    pythia.readString("HadronLevel:all = off");

    pythia.init();

    TFile* rootFile = new TFile("pythiaOutput_ttH_sl.root", "RECREATE");
    TTree* tree = new TTree("events", "ttH semileptonic events");

    Int_t eventNumber;
    Int_t topIsLeptonic;  // 1 if the top (id 6) was the leptonic side, 0 if the antitop was

    Float_t mtop_lep, mW_lep, mtop_had, mW_had;

    Float_t blep_px, blep_py, blep_pz, blep_E, blep_m;
    Float_t lep_px, lep_py, lep_pz, lep_E, lep_m;
    Float_t nu_px, nu_py, nu_pz, nu_E, nu_m;

    Float_t bhad_px, bhad_py, bhad_pz, bhad_E, bhad_m;
    Float_t qvis_px, qvis_py, qvis_pz, qvis_E, qvis_m;
    Float_t qlost_px, qlost_py, qlost_pz, qlost_E, qlost_m;

    Float_t higgs_px, higgs_py, higgs_pz, higgs_E, higgs_m;
    Float_t higgsb1_px, higgsb1_py, higgsb1_pz, higgsb1_E, higgsb1_m;
    Float_t higgsb2_px, higgsb2_py, higgsb2_pz, higgsb2_E, higgsb2_m;

    tree->Branch("eventNumber", &eventNumber, "eventNumber/I");
    tree->Branch("topIsLeptonic", &topIsLeptonic, "topIsLeptonic/I");

    tree->Branch("mtop_lep", &mtop_lep, "mtop_lep/F");
    tree->Branch("mW_lep",   &mW_lep,   "mW_lep/F");
    tree->Branch("mtop_had", &mtop_had, "mtop_had/F");
    tree->Branch("mW_had",   &mW_had,   "mW_had/F");

    tree->Branch("blep_px", &blep_px, "blep_px/F");
    tree->Branch("blep_py", &blep_py, "blep_py/F");
    tree->Branch("blep_pz", &blep_pz, "blep_pz/F");
    tree->Branch("blep_E",  &blep_E,  "blep_E/F");
    tree->Branch("blep_m",  &blep_m,  "blep_m/F");

    tree->Branch("lep_px", &lep_px, "lep_px/F");
    tree->Branch("lep_py", &lep_py, "lep_py/F");
    tree->Branch("lep_pz", &lep_pz, "lep_pz/F");
    tree->Branch("lep_E",  &lep_E,  "lep_E/F");
    tree->Branch("lep_m",  &lep_m,  "lep_m/F");

    tree->Branch("nu_px", &nu_px, "nu_px/F");
    tree->Branch("nu_py", &nu_py, "nu_py/F");
    tree->Branch("nu_pz", &nu_pz, "nu_pz/F");
    tree->Branch("nu_E",  &nu_E,  "nu_E/F");
    tree->Branch("nu_m",  &nu_m,  "nu_m/F");

    tree->Branch("bhad_px", &bhad_px, "bhad_px/F");
    tree->Branch("bhad_py", &bhad_py, "bhad_py/F");
    tree->Branch("bhad_pz", &bhad_pz, "bhad_pz/F");
    tree->Branch("bhad_E",  &bhad_E,  "bhad_E/F");
    tree->Branch("bhad_m",  &bhad_m,  "bhad_m/F");

    tree->Branch("qvis_px", &qvis_px, "qvis_px/F");
    tree->Branch("qvis_py", &qvis_py, "qvis_py/F");
    tree->Branch("qvis_pz", &qvis_pz, "qvis_pz/F");
    tree->Branch("qvis_E",  &qvis_E,  "qvis_E/F");
    tree->Branch("qvis_m",  &qvis_m,  "qvis_m/F");

    tree->Branch("qlost_px", &qlost_px, "qlost_px/F");
    tree->Branch("qlost_py", &qlost_py, "qlost_py/F");
    tree->Branch("qlost_pz", &qlost_pz, "qlost_pz/F");
    tree->Branch("qlost_E",  &qlost_E,  "qlost_E/F");
    tree->Branch("qlost_m",  &qlost_m,  "qlost_m/F");

    tree->Branch("higgs_px", &higgs_px, "higgs_px/F");
    tree->Branch("higgs_py", &higgs_py, "higgs_py/F");
    tree->Branch("higgs_pz", &higgs_pz, "higgs_pz/F");
    tree->Branch("higgs_E",  &higgs_E,  "higgs_E/F");
    tree->Branch("higgs_m",  &higgs_m,  "higgs_m/F");

    tree->Branch("higgsb1_px", &higgsb1_px, "higgsb1_px/F");
    tree->Branch("higgsb1_py", &higgsb1_py, "higgsb1_py/F");
    tree->Branch("higgsb1_pz", &higgsb1_pz, "higgsb1_pz/F");
    tree->Branch("higgsb1_E",  &higgsb1_E,  "higgsb1_E/F");
    tree->Branch("higgsb1_m",  &higgsb1_m,  "higgsb1_m/F");

    tree->Branch("higgsb2_px", &higgsb2_px, "higgsb2_px/F");
    tree->Branch("higgsb2_py", &higgsb2_py, "higgsb2_py/F");
    tree->Branch("higgsb2_pz", &higgsb2_pz, "higgsb2_pz/F");
    tree->Branch("higgsb2_E",  &higgsb2_E,  "higgsb2_E/F");
    tree->Branch("higgsb2_m",  &higgsb2_m,  "higgsb2_m/F");

    outFile.open("pythiaOutput_ttH_sl.txt", std::ios_base::app);

    TRandom3 rng(0);  // for the random kept/lost quark assignment, mirroring
                       // the real analysis's random choice of which jet is "seen"

    long kept = 0, attempts = 0, skippedIncomplete = 0;
    while (kept < (long)evMax && attempts < attemptsMax) {
        ++attempts;
        if (!pythia.next()) continue;

        int topIdx = -1, atopIdx = -1;
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (pythia.event[i].id() == 6  && pythia.event[i].status() == -62) topIdx  = i;
            if (pythia.event[i].id() == -6 && pythia.event[i].status() == -62) atopIdx = i;
        }
        if (topIdx < 0 || atopIdx < 0) continue;

        int Wp = -1, Wm = -1, bTop = -1, bAtop = -1;
        for (int d = pythia.event[topIdx].daughter1(); d <= pythia.event[topIdx].daughter2(); ++d) {
            if (d <= 0) continue;
            if (pythia.event[d].id() == 24) Wp = d;
            if (pythia.event[d].id() == 5)  bTop = d;
        }
        for (int d = pythia.event[atopIdx].daughter1(); d <= pythia.event[atopIdx].daughter2(); ++d) {
            if (d <= 0) continue;
            if (pythia.event[d].id() == -24) Wm = d;
            if (pythia.event[d].id() == -5)  bAtop = d;
        }
        if (Wp < 0 || Wm < 0 || bTop < 0 || bAtop < 0) continue;

        Wp = finalCopy(pythia.event, Wp);
        Wm = finalCopy(pythia.event, Wm);

        bool WpLeptonic, WmLeptonic;
        int lepP=-1, nuP=-1, q1P=-1, q2P=-1;
        int lepM=-1, nuM=-1, q1M=-1, q2M=-1;
        if (!classifyW(pythia.event, Wp, WpLeptonic, lepP, nuP, q1P, q2P)) continue;
        if (!classifyW(pythia.event, Wm, WmLeptonic, lepM, nuM, q1M, q2M)) continue;
        if (WpLeptonic == WmLeptonic) continue;  // keep only semileptonic

        // ── Resolve every index BEFORE touching pythia.event[...] with any of them ──
        topIsLeptonic = WpLeptonic ? 1 : 0;

        int lepTopIdx  = WpLeptonic ? topIdx  : atopIdx;
        int hadTopIdx  = WpLeptonic ? atopIdx : topIdx;
        int bLepIdx    = WpLeptonic ? bTop    : bAtop;
        int bHadIdx    = WpLeptonic ? bAtop   : bTop;
        int lepIdx     = WpLeptonic ? lepP    : lepM;
        int nuIdx      = WpLeptonic ? nuP     : nuM;
        int q1Idx      = WpLeptonic ? q1M     : q1P;
        int q2Idx      = WpLeptonic ? q2M     : q2P;
        int WlepIdx    = WpLeptonic ? Wp      : Wm;
        int WhadIdx    = WpLeptonic ? Wm      : Wp;

        if (lepTopIdx < 0 || hadTopIdx < 0 || bLepIdx < 0 || bHadIdx < 0 ||
            lepIdx < 0 || nuIdx < 0 || q1Idx < 0 || q2Idx < 0 ||
            WlepIdx < 0 || WhadIdx < 0) {
            ++skippedIncomplete;
            std::cerr << "Skipping attempt " << attempts
                      << ": incomplete decay chain (lepIdx=" << lepIdx
                      << " nuIdx=" << nuIdx << " q1Idx=" << q1Idx
                      << " q2Idx=" << q2Idx << " bLepIdx=" << bLepIdx
                      << " bHadIdx=" << bHadIdx << ")" << std::endl;
            continue;
        }

        // ── TXT output (only now that we know this event is complete) ──
        outFile <<"------------ Event ------ "<<kept<<" ------------------"<<endl;
        pythia.event.list();
        outFile  << endl;
        outFile  << "Top Decay Chain:" << endl;
        processQuarkAndDecays(pythia.event, 6);
        outFile  << endl;
        outFile  << "Anti-top Decay Chain:" << endl;
        processQuarkAndDecays(pythia.event, -6);
        outFile  << endl;
        outFile  << "Higgs Decay Chain:" << endl;
        processQuarkAndDecays(pythia.event, 25);
        outFile  << endl;

        // ── ROOT output ──
        eventNumber = kept;

        mtop_lep = pythia.event[lepTopIdx].m();
        mW_lep   = pythia.event[WlepIdx].m();
        mtop_had = pythia.event[hadTopIdx].m();
        mW_had   = pythia.event[WhadIdx].m();

        Particle& blep = pythia.event[bLepIdx];
        blep_px = blep.px(); blep_py = blep.py(); blep_pz = blep.pz();
        blep_E  = blep.e();  blep_m  = blep.m();

        Particle& lepP4 = pythia.event[lepIdx];
        lep_px = lepP4.px(); lep_py = lepP4.py(); lep_pz = lepP4.pz();
        lep_E  = lepP4.e();  lep_m  = lepP4.m();

        Particle& nuP4 = pythia.event[nuIdx];
        nu_px = nuP4.px(); nu_py = nuP4.py(); nu_pz = nuP4.pz();
        nu_E  = nuP4.e();  nu_m  = nuP4.m();

        Particle& bhad = pythia.event[bHadIdx];
        bhad_px = bhad.px(); bhad_py = bhad.py(); bhad_pz = bhad.pz();
        bhad_E  = bhad.e();  bhad_m  = bhad.m();

        // Randomly designate one hadronic-W quark as "kept" (qvis) and the
        // other as "lost" (qlost) -- mirrors the random choice your real
        // analysis makes over which W-daughter jet actually gets reconstructed.
        int keptIdx, lostIdx;
        if (rng.Integer(2) == 0) { keptIdx = q1Idx; lostIdx = q2Idx; }
        else                     { keptIdx = q2Idx; lostIdx = q1Idx; }

        Particle& qvisP4 = pythia.event[keptIdx];
        qvis_px = qvisP4.px(); qvis_py = qvisP4.py(); qvis_pz = qvisP4.pz();
        qvis_E  = qvisP4.e();  qvis_m  = qvisP4.m();

        Particle& qlostP4 = pythia.event[lostIdx];
        qlost_px = qlostP4.px(); qlost_py = qlostP4.py(); qlost_pz = qlostP4.pz();
        qlost_E  = qlostP4.e();  qlost_m  = qlostP4.m();

        Particle higgs;
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (pythia.event[i].id() == 25 && pythia.event[i].status() == -62) {
                higgs = pythia.event[i];
                break;
            }
        }
        higgs_px = higgs.px(); higgs_py = higgs.py(); higgs_pz = higgs.pz();
        higgs_E  = higgs.e();  higgs_m  = higgs.m();

        bool firstB = true;
        for (int i = higgs.daughter1(); i <= higgs.daughter2(); ++i) {
            if (i <= 0) continue;
            Particle d = pythia.event[i];
            if (std::abs(d.id()) == 5) {
                if (firstB) {
                    higgsb1_px = d.px(); higgsb1_py = d.py(); higgsb1_pz = d.pz();
                    higgsb1_E  = d.e();  higgsb1_m  = d.m();
                    firstB = false;
                } else {
                    higgsb2_px = d.px(); higgsb2_py = d.py(); higgsb2_pz = d.pz();
                    higgsb2_E  = d.e();  higgsb2_m  = d.m();
                }
            }
        }

        tree->Fill();
        ++kept;
        std::cout << std::endl;
    }

    std::cout << "Kept " << kept << " semileptonic events out of "
              << attempts << " attempts (" << skippedIncomplete
              << " skipped for incomplete decay chains)." << std::endl;

    outFile.close();
    rootFile->Write();
    rootFile->Close();

    gApplication->Terminate(0);
    return 0;
}