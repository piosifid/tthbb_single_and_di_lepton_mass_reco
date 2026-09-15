#include "Pythia8/Pythia.h" 
#include "TFile.h" // .root file
#include "TH1F.h" //histograms
#include <fstream> // outfile
#include <iomanip> //
#include "TMath.h" // Pi
#include <cstdio> // Include for std::remove
#include "TApplication.h" // exit
#include "TTree.h"

using namespace Pythia8;
using namespace std;

std::ofstream outFile;

///////////////////////////////////////////////////////////////
// Particle ID List for Reference
//
// Top Quarks:
//    6   : Top quark (t)
//   -6   : Anti-top quark (t̄)
//
// W Bosons and Related:
//   24   : W⁺ boson
//  -24   : W⁻ boson
//   11   : Electron (e⁻)
//  -11   : Positron (e⁺)
//   12   : Electron neutrino (νₑ)
//  -12   : Electron anti-neutrino (ν̄ₑ)
//   13   : Muon (μ⁻)
//  -13   : Anti-muon (μ⁺)
//   14   : Muon neutrino (νₘᵤ)
//  -14   : Muon anti-neutrino (ν̄ₘᵤ)
//
// Higgs Boson and Decay Products:
//   25   : Higgs boson (H)
//    5   : Bottom quark (b)
//   -5   : Anti-bottom quark (b̄)
///////////////////////////////////////////////////////////////


//Function to print the 4-momentum
void printMomenta(const Event& event, int index) {


    // Ensure numbers are printed with 15 digits of precision
    outFile << std::fixed << std::setprecision(15);

    // Print the four-momentum of the current particle
    Particle particle = event[index];
    // if ( event[index].status() != -62){
    outFile  << particle.name() << ": mass: " << particle.m()//<< " (index " << index << ") "
              << ", px: " << particle.px() << ", py: " << particle.py() << ", pz: " << particle.pz() << ", E: " << particle.e() << std::endl;
          //}
    // Recursively print the momenta of the daughters
    for (int i = particle.daughter1(); i <= particle.daughter2(); ++i) {
        if (i > 0 && i < event.size()) {
            printMomenta(event, i);
        }
    }
}

// Function to find particles with specific ID (quarkId) in final state (-62)
// and print their complete decay chain using printMomenta
void processQuarkAndDecays(const Event& event, int quarkId) {
    for (int i = 0; i < event.size(); ++i) {
        if (event[i].id() == quarkId && event[i].status() == -62) { // Adjust status code if necessary
            printMomenta(event, i); // Print the particle and its decay chain
        }
    }
}

//Number of generated events
float evMax=10000;

int pythiattH_dilepton() {
    // Initialize Pythia as before
    std::remove("pythiaOutput_ttH.txt");
    std::remove("pythiaOutput_ttH.root");
    
    Pythia pythia;
    Particle b;
    Vec4 p_ttbar, pTop, boosted, pAntitop, lepton_plus, lepton_minus, bq;

    // Set up for pp collision at 13 TeV
    pythia.readString("Beams:idA = 2212");  
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13600.");

    // Enable ttH production
    pythia.readString("HiggsSM:gg2Httbar = on "); // Gluon-gluon fusion producing ttH
    pythia.readString("HiggsSM:qqbar2Httbar = on"); // Quark-antiquark annihilation producing ttH

    // Disable all top decays and then enable decay to W and b
    pythia.readString("6:onMode = off");
    pythia.readString("6:onIfAny = 24");

    // Disable all W decays, then enable decay to electron, muon and their neutrinos
    pythia.readString("24:onMode = off");
    pythia.readString("24:onIfAny = 11 12 13 14"); // 11: e-, 12: electron-neutrino, 13: mu-, 14: muon-neutrino

    //  Disable all Higgs decays, then enable only to bbbar 
    pythia.readString("25:onMode = off");    // Turn off all Higgs decays
    pythia.readString("25:onIfMatch = 5 -5"); // Enable Higgs decay to b-bbar only

    // Disable Initial State Radiation, Final State Radiation, Multiple Parton Interactions, and Hadronization to study the complete process pp → ttH → (bW+)(bW-)(bb) at parton level
    pythia.readString("PartonLevel:ISR = off");
    pythia.readString("PartonLevel:FSR = off");
    pythia.readString("PartonLevel:MPI = off");
    pythia.readString("HadronLevel:all = off");

    // Initialize Pythia
    pythia.init();

    // Create ROOT file and tree
    TFile* rootFile = new TFile("pythiaOutput_ttH.root", "RECREATE");
    TTree* tree = new TTree("events", "ttH Events");

    // Define variables for ROOT tree
    int eventNumber;
    // Top variables
    Float_t top_px, top_py, top_pz, top_E, top_m;
    Float_t topW_px, topW_py, topW_pz, topW_E, topW_m;
    Float_t topb_px, topb_py, topb_pz, topb_E, topb_m;
    Float_t toplep_px, toplep_py, toplep_pz, toplep_E, toplep_m;
    Float_t topnu_px, topnu_py, topnu_pz, topnu_E, topnu_m;
    
    // Antitop variables
    Float_t atop_px, atop_py, atop_pz, atop_E, atop_m;
    Float_t atopW_px, atopW_py, atopW_pz, atopW_E, atopW_m;
    Float_t atopb_px, atopb_py, atopb_pz, atopb_E, atopb_m;
    Float_t atoplep_px, atoplep_py, atoplep_pz, atoplep_E, atoplep_m;
    Float_t atopnu_px, atopnu_py, atopnu_pz, atopnu_E, atopnu_m;
    
    // Higgs variables
    Float_t higgs_px, higgs_py, higgs_pz, higgs_E, higgs_m;
    Float_t higgsb1_px, higgsb1_py, higgsb1_pz, higgsb1_E, higgsb1_m;
    Float_t higgsb2_px, higgsb2_py, higgsb2_pz, higgsb2_E, higgsb2_m;

    // Set up tree branches
    tree->Branch("eventNumber", &eventNumber, "eventNumber/I");
    
    // Top branches
    tree->Branch("top_px", &top_px, "top_px/F");
    tree->Branch("top_py", &top_py, "top_py/F");
    tree->Branch("top_pz", &top_pz, "top_pz/F");
    tree->Branch("top_E", &top_E, "top_E/F");
    tree->Branch("top_m", &top_m, "top_m/F");
    
    // W from top branches
    tree->Branch("topW_px", &topW_px, "topW_px/F");
    tree->Branch("topW_py", &topW_py, "topW_py/F");
    tree->Branch("topW_pz", &topW_pz, "topW_pz/F");
    tree->Branch("topW_E", &topW_E, "topW_E/F");
    tree->Branch("topW_m", &topW_m, "topW_m/F");
    
    // b from top branches
    tree->Branch("topb_px", &topb_px, "topb_px/F");
    tree->Branch("topb_py", &topb_py, "topb_py/F");
    tree->Branch("topb_pz", &topb_pz, "topb_pz/F");
    tree->Branch("topb_E", &topb_E, "topb_E/F");
    tree->Branch("topb_m", &topb_m, "topb_m/F");
    
    // Lepton from top W branches
    tree->Branch("toplep_px", &toplep_px, "toplep_px/F");
    tree->Branch("toplep_py", &toplep_py, "toplep_py/F");
    tree->Branch("toplep_pz", &toplep_pz, "toplep_pz/F");
    tree->Branch("toplep_E", &toplep_E, "toplep_E/F");
    tree->Branch("toplep_m", &toplep_m, "toplep_m/F");
    
    // Neutrino from top W branches
    tree->Branch("topnu_px", &topnu_px, "topnu_px/F");
    tree->Branch("topnu_py", &topnu_py, "topnu_py/F");
    tree->Branch("topnu_pz", &topnu_pz, "topnu_pz/F");
    tree->Branch("topnu_E", &topnu_E, "topnu_E/F");
    tree->Branch("topnu_m", &topnu_m, "topnu_m/F");
    
    // Antitop branches
    tree->Branch("atop_px", &atop_px, "atop_px/F");
    tree->Branch("atop_py", &atop_py, "atop_py/F");
    tree->Branch("atop_pz", &atop_pz, "atop_pz/F");
    tree->Branch("atop_E", &atop_E, "atop_E/F");
    tree->Branch("atop_m", &atop_m, "atop_m/F");
    
    // W from antitop branches
    tree->Branch("atopW_px", &atopW_px, "atopW_px/F");
    tree->Branch("atopW_py", &atopW_py, "atopW_py/F");
    tree->Branch("atopW_pz", &atopW_pz, "atopW_pz/F");
    tree->Branch("atopW_E", &atopW_E, "atopW_E/F");
    tree->Branch("atopW_m", &atopW_m, "atopW_m/F");
    
    // b from antitop branches
    tree->Branch("atopb_px", &atopb_px, "atopb_px/F");
    tree->Branch("atopb_py", &atopb_py, "atopb_py/F");
    tree->Branch("atopb_pz", &atopb_pz, "atopb_pz/F");
    tree->Branch("atopb_E", &atopb_E, "atopb_E/F");
    tree->Branch("atopb_m", &atopb_m, "atopb_m/F");
    
    // Lepton from antitop W branches
    tree->Branch("atoplep_px", &atoplep_px, "atoplep_px/F");
    tree->Branch("atoplep_py", &atoplep_py, "atoplep_py/F");
    tree->Branch("atoplep_pz", &atoplep_pz, "atoplep_pz/F");
    tree->Branch("atoplep_E", &atoplep_E, "atoplep_E/F");
    tree->Branch("atoplep_m", &atoplep_m, "atoplep_m/F");
    
    // Neutrino from antitop W branches
    tree->Branch("atopnu_px", &atopnu_px, "atopnu_px/F");
    tree->Branch("atopnu_py", &atopnu_py, "atopnu_py/F");
    tree->Branch("atopnu_pz", &atopnu_pz, "atopnu_pz/F");
    tree->Branch("atopnu_E", &atopnu_E, "atopnu_E/F");
    tree->Branch("atopnu_m", &atopnu_m, "atopnu_m/F");
    
    // Higgs branches
    tree->Branch("higgs_px", &higgs_px, "higgs_px/F");
    tree->Branch("higgs_py", &higgs_py, "higgs_py/F");
    tree->Branch("higgs_pz", &higgs_pz, "higgs_pz/F");
    tree->Branch("higgs_E", &higgs_E, "higgs_E/F");
    tree->Branch("higgs_m", &higgs_m, "higgs_m/F");
    
    // First b from Higgs branches
    tree->Branch("higgsb1_px", &higgsb1_px, "higgsb1_px/F");
    tree->Branch("higgsb1_py", &higgsb1_py, "higgsb1_py/F");
    tree->Branch("higgsb1_pz", &higgsb1_pz, "higgsb1_pz/F");
    tree->Branch("higgsb1_E", &higgsb1_E, "higgsb1_E/F");
    tree->Branch("higgsb1_m", &higgsb1_m, "higgsb1_m/F");
    
    // Second b from Higgs branches
    tree->Branch("higgsb2_px", &higgsb2_px, "higgsb2_px/F");
    tree->Branch("higgsb2_py", &higgsb2_py, "higgsb2_py/F");
    tree->Branch("higgsb2_pz", &higgsb2_pz, "higgsb2_pz/F");
    tree->Branch("higgsb2_E", &higgsb2_E, "higgsb2_E/F");
    tree->Branch("higgsb2_m", &higgsb2_m, "higgsb2_m/F");

    // Open txt file
    outFile.open("pythiaOutput_ttH.txt", std::ios_base::app);

    // Event loop
    for (int iEvent = 0; iEvent < evMax; ++iEvent) {
        if (!pythia.next()) continue;

        // TXT output (keeping existing functionality)
        outFile << "------------ Event ------ " << iEvent << " ------------------" << endl;
        pythia.event.list();
        outFile << endl;
        outFile << "Top Decay Chain:" << endl;
        processQuarkAndDecays(pythia.event, 6);
        outFile << endl;
        outFile << "Anti-top Decay Chain:" << endl;
        processQuarkAndDecays(pythia.event, -6);
        outFile << endl;
        outFile << "Higgs Decay Chain:" << endl;
        processQuarkAndDecays(pythia.event, 25);
        outFile << endl;

        // ROOT output
        eventNumber = iEvent;

        // Helper lambda to find particles
        auto findParticle = [&](int pid, int status) -> Particle {
            for (int i = 0; i < pythia.event.size(); ++i) {
                if (pythia.event[i].id() == pid && pythia.event[i].status() == status) {
                    return pythia.event[i];
                }
            }
            return Particle();
        };

        // Store top quark and its decay products
        Particle top = findParticle(6, -62);
        if (top.id() == 6) {
            top_px = top.px(); top_py = top.py(); top_pz = top.pz();
            top_E = top.e(); top_m = top.m();
            
            // Find W and b from top decay
            for (int i = top.daughter1(); i <= top.daughter2(); ++i) {
                Particle daughter = pythia.event[i];
                if (daughter.id() == 24) { // W+
                    topW_px = daughter.px(); topW_py = daughter.py();
                    topW_pz = daughter.pz(); topW_E = daughter.e();
                    topW_m = daughter.m();
                    
                    // Find leptons from W decay
                    for (int j = daughter.daughter1(); j <= daughter.daughter2(); ++j) {
                        Particle Wdaughter = pythia.event[j];
                        if (abs(Wdaughter.id()) == 11 || abs(Wdaughter.id()) == 13) {
                            toplep_px = Wdaughter.px(); toplep_py = Wdaughter.py();
                            toplep_pz = Wdaughter.pz(); toplep_E = Wdaughter.e();
                            toplep_m = Wdaughter.m();
                        } else if (abs(Wdaughter.id()) == 12 || abs(Wdaughter.id()) == 14) {
                            topnu_px = Wdaughter.px(); topnu_py = Wdaughter.py();
                            topnu_pz = Wdaughter.pz(); topnu_E = Wdaughter.e();
                            topnu_m = Wdaughter.m();
                        }
                    }
                } else if (daughter.id() == 5) {
                    topb_px = daughter.px(); topb_py = daughter.py();
                    topb_pz = daughter.pz(); topb_E = daughter.e();
                    topb_m = daughter.m();
                }
            }
        }

        // Store antitop quark and its decay products
        Particle antitop = findParticle(-6, -62);
        if (antitop.id() == -6) {
            atop_px = antitop.px(); atop_py = antitop.py(); atop_pz = antitop.pz();
            atop_E = antitop.e(); atop_m = antitop.m();
            
            // Find W and b from antitop decay
            for (int i = antitop.daughter1(); i <= antitop.daughter2(); ++i) {
                Particle daughter = pythia.event[i];
                if (daughter.id() == -24) { // W-
                    atopW_px = daughter.px(); atopW_py = daughter.py();
                    atopW_pz = daughter.pz(); atopW_E = daughter.e();
                    atopW_m = daughter.m();
                    
                    // Find leptons from W decay
                    for (int j = daughter.daughter1(); j <= daughter.daughter2(); ++j) {
                        Particle Wdaughter = pythia.event[j];
                        if (abs(Wdaughter.id()) == 11 || abs(Wdaughter.id()) == 13) { // lepton
                            atoplep_px = Wdaughter.px(); atoplep_py = Wdaughter.py();
                            atoplep_pz = Wdaughter.pz(); atoplep_E = Wdaughter.e();
                            atoplep_m = Wdaughter.m();
                        } else if (abs(Wdaughter.id()) == 12 || abs(Wdaughter.id()) == 14) { // neutrino
                            atopnu_px = Wdaughter.px(); atopnu_py = Wdaughter.py();
                            atopnu_pz = Wdaughter.pz(); atopnu_E = Wdaughter.e();
                            atopnu_m = Wdaughter.m();
                        }
                    }
                } else if (daughter.id() == -5) { // anti-b quark
                    atopb_px = daughter.px(); atopb_py = daughter.py();
                    atopb_pz = daughter.pz(); atopb_E = daughter.e();
                    atopb_m = daughter.m();
                }
            }
        }

        // Store Higgs and its decay products
        Particle higgs = findParticle(25, -62);
        if (higgs.id() == 25) {
            higgs_px = higgs.px(); higgs_py = higgs.py(); higgs_pz = higgs.pz();
            higgs_E = higgs.e(); higgs_m = higgs.m();
            
            bool firstB = true;
            // Find b quarks from Higgs decay
            for (int i = higgs.daughter1(); i <= higgs.daughter2(); ++i) {
                Particle daughter = pythia.event[i];
                if (abs(daughter.id()) == 5) { // b or anti-b quark
                    if (firstB) {
                        higgsb1_px = daughter.px(); higgsb1_py = daughter.py();
                        higgsb1_pz = daughter.pz(); higgsb1_E = daughter.e();
                        higgsb1_m = daughter.m();
                        firstB = false;
                    } else {
                        higgsb2_px = daughter.px(); higgsb2_py = daughter.py();
                        higgsb2_pz = daughter.pz(); higgsb2_E = daughter.e();
                        higgsb2_m = daughter.m();
                    }
                }
            }
        }

        tree->Fill();
    }

    // Cleanup
    outFile.close();
    rootFile->Write();
    rootFile->Close();
    
    gApplication->Terminate(0);
    return 0;
}
