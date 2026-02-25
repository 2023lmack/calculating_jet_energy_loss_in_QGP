#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <memory>
#include <iomanip>

// Pythia8
#include "Pythia8/Pythia.h"

// FastJet
#include "fastjet/ClusterSequence.hh"

// ROOT
#include "TFile.h"
#include "TH1D.h"

// Struct for jet and constituients histograms
struct HistTriplet {
  TH1D* pt  = nullptr;
  TH1D* phi = nullptr;
  TH1D* eta = nullptr;
};

// Makes name for each histogram (not title)
static std::string makeName(const std::string& base,
                            int pthatIdx,
                            int jetIdx = -1) {
  std::ostringstream ss;
  ss << base << "_pthat" << pthatIdx;
  if (jetIdx >= 0)
    ss << "_jet" << jetIdx;
  return ss.str();
}
// Makes title
static std::string makeFullTitle(const std::string& quantity,
                                 double ptHatLow, double ptHatHigh,
                                 double jetLow, double jetHigh,
                                 const std::string& axisLabel,
                                 const std::string& yLabel) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(0);  // <-- 0 decimals (change to 1/2 if you want)

  ss << quantity
     << " | pTHat [" << ptHatLow << ", " << ptHatHigh << ")"
     << " | jet pT [" << jetLow << ", " << jetHigh << ")"
     << ";" << axisLabel << ";" << yLabel;
  return ss.str();
}

// Find which pt-bin a jet belongs to: [edge[i], edge[i+1})
static int findPtBin(double pt, const std::vector<double>& edges) {
  if (edges.size() < 2) return -1;
  for (size_t i = 0; i + 1 < edges.size(); ++i) {
    if (pt >= edges[i] && pt < edges[i + 1]) return static_cast<int>(i);
  }
  return -1; // out of range
}

// Simulation method
double run_events(int nEvents, double ptMin, double ptMax,
                TH1D& hParPt, TH1D& hParPhi, TH1D& hParEta,
                const std::vector<double>& jetPtBins,
                std::vector<HistTriplet>& jetHists,
                std::vector<HistTriplet>& constHists) {
  // ---- Settings
  double eCM         = 5020;
  double R           = 0.4;
  double jetPtMin    = 0.0;
  double eta_cut     = 2.8 + R;
  double jet_eta_cut = eta_cut - R;

  // Error check
  if (jetPtBins.size() < 2) {
    std::cerr << "jetPtBins must have at least 2 edges.\n";
    return 0.0;
  }
  const size_t nJetBins = jetPtBins.size() - 1;
  if (jetHists.size() != nJetBins || constHists.size() != nJetBins) {
    std::cerr << "jetHists/constHists size mismatch with jetPtBins.\n";
    return 0.0;
  }

  // ---- Pythia init
  Pythia8::Pythia pythia;
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");
  pythia.readString("Beams:eCM = " + std::to_string(eCM));

  pythia.readString("PhaseSpace:pTHatMin = " + std::to_string(ptMin));
  pythia.readString("PhaseSpace:pTHatMax = " + std::to_string(ptMax));

  pythia.readString("HardQCD:all = on");
  pythia.readString("HadronLevel:all = on");

  pythia.readString("Next:numberShowEvent = 0");
  pythia.readString("Next:numberShowInfo  = 0");
  pythia.readString("Next:numberShowProcess = 0");

  pythia.init();

  // ---- FastJet setup
  fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, R);

  // ---- Event loop for each particle
  for (int iEvt = 0; iEvt < nEvents; ++iEvt) {
    if (!pythia.next()) continue;

    std::vector<fastjet::PseudoJet> particles;
    particles.reserve(pythia.event.size());

    std::vector<fastjet::PseudoJet> hardPartons;
    hardPartons.reserve(16);

    for (int i = 0; i < pythia.event.size(); ++i) { // 2 particles; print if no partons found; measure leading jets
      const auto& p = pythia.event[i];
        
      if (!p.isFinal())   continue;
      if (!p.isVisible()) continue;

      fastjet::PseudoJet pj(p.px(), p.py(), p.pz(), p.e());
      if (std::abs(pj.eta()) > eta_cut) continue;

      // Adds particles to the array
      pj.set_user_index(i);
      particles.push_back(pj);

      // Fill particle histograms (per pTHat bin)
      hParPt.Fill(pj.perp());
      hParPhi.Fill(pj.phi_std());
      hParEta.Fill(pj.eta());
    }

    if (particles.empty()) continue;

    fastjet::ClusterSequence cs(particles, jetDef);
    auto jets = fastjet::sorted_by_pt(cs.inclusive_jets(jetPtMin));

    // Loops over jets
    for (const auto& j : jets) {
      if (std::abs(j.eta()) > jet_eta_cut) continue;

      const double jpt = j.perp();
      const int jetBin = findPtBin(jpt, jetPtBins);
      if (jetBin < 0) continue; // outside [jetPtBins[0], jetPtBins.back())

      // Fill jet hists for this jetPt bin
      jetHists[jetBin].pt->Fill(jpt);
      jetHists[jetBin].phi->Fill(j.phi_std());
      jetHists[jetBin].eta->Fill(j.eta());

      // Fill constituent hists *for jets in this jetPt bin*
      for (const auto& c : j.constituents()) {
        constHists[jetBin].pt->Fill(c.perp());
        constHists[jetBin].phi->Fill(c.phi_std());
        constHists[jetBin].eta->Fill(c.eta());
      }
    }
  }
  double sigmaGen = pythia.info.sigmaGen();
  int nAccepted   = pythia.info.nAccepted();
  pythia.stat();
  if (nAccepted <= 0) return 0.0;
  return sigmaGen / nAccepted;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Error: need number of events.\n"
              << "Usage: " << argv[0] << " <num_events>\n";
    return 1;
  }
  // Takes input for number of events
  int nEvents = std::stoi(argv[1]);

  // Creates directory for outputs and root file
  std::filesystem::create_directories("output");
  TFile outFile("output/jet_kinematics.root", "RECREATE");

  // Sets ptHat and jetPt bins
  std::vector<double> ptHatBins = {20, 35, 50, 65, 80, 100, 120, 140, 160, 180, 200, 220, 245, 270, 295, 320, 350, 380};
  std::vector<double> jetPtBins = {20, 35, 50, 65, 80, 100, 120, 140, 160, 180, 200, 220, 245, 270, 295, 320, 350, 380};
  //std::vector<double> ptHatBins = {20, 80, 400};
  //std::vector<double> jetPtBins = {20, 80, 400};
  const size_t nJetBins = jetPtBins.size() - 1;
  const size_t nptHatBins = ptHatBins.size() - 1;
  std::vector<int> nQuarkJets(nJetBins, 0);
  std::vector<int> nGluonJets(nJetBins, 0);

  // Makes histograms for different jetPt bins with the scaled data
  // std::vector<TH1D*> hJetPt_weighted(nJetBins);
  // for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
  //     std::string name  = "hJetPt_weighted_bin" + std::to_string(jetBin);

  //     std::ostringstream title;
  //     title << std::fixed << std::setprecision(0)
  //           << "Weighted Jet pT | jet bin ["
  //           << jetPtBins[jetBin] << ", " << jetPtBins[jetBin+1] << ")"
  //           << ";p_{T} [GeV];d#sigma/dp_{T}";

  //     hJetPt_weighted[jetBin] = new TH1D(name.c_str(),
  //                                       title.str().c_str(),
  //                                       200, 0, 400);
  // }

  std::vector<TH1D*> hJetPt_weighted_combined(nJetBins, nullptr);
  for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {

  const double jetLow  = jetPtBins[jetBin];
  const double jetHigh = jetPtBins[jetBin + 1];

  std::string name = "hJetPt_weighted_jet" + std::to_string(jetBin);
  std::ostringstream title;
  title << std::fixed << std::setprecision(0)
        << "Weighted Jet pT | jet pT [" << jetLow << ", " << jetHigh << ")"
        << ";p_{T} [GeV];d#sigma/dp_{T}";

  hJetPt_weighted_combined[jetBin] = new TH1D(name.c_str(), title.str().c_str(), 200, 0, 400);
  }

  std::vector<std::vector<TH1D*>> hJetPt_weighted(nptHatBins, std::vector<TH1D*>(nJetBins, nullptr));

  // Loops over each ptHat bin
  for (size_t ptHatBin = 0; ptHatBin < nptHatBins; ++ptHatBin) {

    const double ptHatLow  = ptHatBins[ptHatBin];
    const double ptHatHigh = ptHatBins[ptHatBin + 1];
    for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {

      const double jetLow  = jetPtBins[jetBin];
      const double jetHigh = jetPtBins[jetBin + 1];

      std::string name = "hJetPt_weighted_pthat" + std::to_string(ptHatBin) + "_jet" + std::to_string(jetBin);
      std::ostringstream title;
      title << std::fixed << std::setprecision(0)
            << "Weighted Jet pT | pTHat [" << ptHatLow << ", " << ptHatHigh << ")"
            << " | jet pT [" << jetLow << ", " << jetHigh << ")"
            << ";p_{T} [GeV];d#sigma/dp_{T}";

      hJetPt_weighted[ptHatBin][jetBin] = new TH1D(name.c_str(), title.str().c_str(), 200, 0, 400);
    }


    // Particle hists for THIS pTHat bin
    // Brackets delete anything defined inside of the code after the last bracket
    {
      // Erases decimals in title
      std::ostringstream rng;
      rng << std::fixed << std::setprecision(0)
            << "pTHat ["
            << ptHatLow << ", "
            << ptHatHigh
            << ")";

      std::string genTitle = rng.str();

    // Particle Pt
      TH1D hParPt(
        makeName("hParPt", ptHatBin).c_str(),
        ("Particle p_{T} | " + genTitle + ");p_{T};Particles").c_str(),
        200, 0, 200
      );

      // Particle phi
      TH1D hParPhi(
        makeName("hParPhi", ptHatBin).c_str(),
        ("Particle #phi | " + genTitle + ");#phi;Particles").c_str(),
        128, -M_PI, M_PI
      );

      // Particle eta
      TH1D hParEta(
        makeName("hParEta", ptHatBin).c_str(),
        ("Particle #eta | " + genTitle + ");#eta;Particles").c_str(),
        120, -6, 6
      );

      // Build jet/constituent histograms for *each jetPt bin* within this pTHat bin
      // Defines vectors for jet Pt, phi, and eta
      std::vector<std::unique_ptr<TH1D>> jetPtH, jetPhiH, jetEtaH;

      // Defines vector for constituients pt, phi, and eta
      std::vector<std::unique_ptr<TH1D>> constPtH, constPhiH, constEtaH;

      // Reserves enough space in each array for the number of histograms necessary for each jetPt bin
      jetPtH.reserve(nJetBins);  jetPhiH.reserve(nJetBins);  jetEtaH.reserve(nJetBins);
      constPtH.reserve(nJetBins); constPhiH.reserve(nJetBins); constEtaH.reserve(nJetBins);

      // Makes a vector of HistTriplets with space for each jetpt bin
      std::vector<HistTriplet> jetHists(nJetBins), constHists(nJetBins);

      // Loops over each jetPt bin
      for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
        const double jetLow  = jetPtBins[jetBin];
        const double jetHigh = jetPtBins[jetBin + 1];

        // Adds a histogram to jetPtH
        jetPtH.emplace_back(std::make_unique<TH1D>(
          makeName("hJetPt", ptHatBin, jetBin).c_str(),
          makeFullTitle("Jet p_{T}",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "p_{T}", "Jets").c_str(),
          200, 0, 400
        ));

        // Adds a histogram to JetphiH
        jetPhiH.emplace_back(std::make_unique<TH1D>(
          makeName("hJetPhi", ptHatBin, jetBin).c_str(),
          makeFullTitle("Jet #phi",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "#phi", "Jets").c_str(),
          128, -M_PI, M_PI
        ));

        // Adds a histogram to JetEtaH
        jetEtaH.emplace_back(std::make_unique<TH1D>(
          makeName("hJetEta", ptHatBin, jetBin).c_str(),
          makeFullTitle("Jet #eta",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "#eta", "Jets").c_str(),
          120, -6, 6
        ));

        // Adds a histogram to ConstituentPtH
        constPtH.emplace_back(std::make_unique<TH1D>(
          makeName("hConstPt", ptHatBin, jetBin).c_str(),
          makeFullTitle("Constituent p_{T}",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "p_{T}", "Constituents").c_str(),
          200, 0, 200
        ));

        // Adds a histogram to ConstituentPhiH
        constPhiH.emplace_back(std::make_unique<TH1D>(
          makeName("hConstPhi", ptHatBin, jetBin).c_str(),
          makeFullTitle("Constituent #phi",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "#phi", "Constituents").c_str(),
          128, -M_PI, M_PI
        ));

        // Adds a histogram to ConstituentEtaH
        constEtaH.emplace_back(std::make_unique<TH1D>(
          makeName("hConstEta", ptHatBin, jetBin).c_str(),
          makeFullTitle("Constituent #eta",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "#eta", "Constituents").c_str(),
          120, -6, 6
        ));
        // Adds a HistTriplet to jetHists (defined outside of the loop) for each jetBin
        jetHists[jetBin] = { jetPtH.back().get(),   jetPhiH.back().get(),   jetEtaH.back().get() };

        // Adds a HistTriplet to constHists (defined outside of the loop) for each jetBin
        constHists[jetBin] = { constPtH.back().get(), constPhiH.back().get(), constEtaH.back().get() };
      }


      // One Pythia run fills ALL jetPt bins for this pTHat bin
      double weight = run_events(nEvents, ptHatLow, ptHatHigh,
                          hParPt, hParPhi, hParEta,
                          jetPtBins,
                          jetHists, constHists);

      std::cout << "ptHatBin " << ptHatBin << " weight = " << weight << "\n";

      if (weight <= 0) continue; 

      for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
        hJetPt_weighted[ptHatBin][jetBin]->Add(jetHists[jetBin].pt, weight);
      }

      for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
        hJetPt_weighted_combined[jetBin]->Add(hJetPt_weighted[ptHatBin][jetBin]);
      }

      // Write particle hists for this pTHat bin
      outFile.cd();
      hParPt.Write(); hParPhi.Write(); hParEta.Write();

      // Write all jet/constituent hists for this pTHat bin
      outFile.cd();
      for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
        jetPtH[jetBin]->Write();
        jetPhiH[jetBin]->Write();
        jetEtaH[jetBin]->Write();

        constPtH[jetBin]->Write();
        constPhiH[jetBin]->Write();
        constEtaH[jetBin]->Write();
      }
    }
  }
  

  // Sets up weighted total JetPt histogram
  TH1D* hJetPt_total = (TH1D*)hJetPt_weighted[0][0]->Clone("hJetPt_total");
  hJetPt_total->Reset();
  for (size_t ptHatBin = 0; ptHatBin < nptHatBins; ++ptHatBin) {
    for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
      hJetPt_total->Add(hJetPt_weighted[ptHatBin][jetBin]);
    }
  }

  hJetPt_total->SetTitle("Total Weighted Jet p_{T};p_{T} [GeV];d#sigma/dp_{T}");

  outFile.cd();
  for (size_t ptHatBin = 0; ptHatBin < nptHatBins; ++ptHatBin) {
    for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
      hJetPt_weighted[ptHatBin][jetBin]->Write();
    }
  }

  outFile.cd();
  for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
    hJetPt_weighted_combined[jetBin]->Write();
  }

  for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
    delete hJetPt_weighted_combined[jetBin];
  }

  for (size_t ptHatBin = 0; ptHatBin < nptHatBins; ++ptHatBin) {
    for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
      delete hJetPt_weighted[ptHatBin][jetBin];
    }
  }

  // Write total weighted jetPt histogram
  outFile.cd();
  hJetPt_total->Write();
  delete hJetPt_total;

  outFile.Close();
  std::cout << "Wrote output/jet_kinematics.root\n";
  return 0;
}