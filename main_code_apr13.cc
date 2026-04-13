// C++ Packages
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
#include "TGraphErrors.h"
#include "TCanvas.h"

// =====================================================
// Sets every angle phi to its simplest form
// =====================================================
static inline double dPhi(double a, double b) {
  double d = a - b;
  while (d >  M_PI) d -= 2.0*M_PI;
  while (d < -M_PI) d += 2.0*M_PI;
  return d;
}

static inline double absDPhi(double a, double b) {
  return std::abs(dPhi(a, b));
}

// =====================================================
// Computes delta R^2 for two Fastjet Objects
// Used to find the distance between a jet and hard partons
// =====================================================
static inline double deltaR2(const fastjet::PseudoJet& a,
                             const fastjet::PseudoJet& b) {
  const double dy   = a.eta() - b.eta();
  const double dphi = dPhi(a.phi_std(), b.phi_std());
  return dy*dy + dphi*dphi;
}

// =====================================================
// Adds the pT of surrounding particles to enforce that the 
// combined pT < 3.0 GeV
// =====================================================
static inline bool photonIsIsolatedCone03(const Pythia8::Event& evt,
                                          int phoIdx,
                                          const std::vector<fastjet::PseudoJet>& particles,
                                          double Riso = 0.3,
                                          double maxSumPt = 3.0) {
  // ---- Sets photon
  const auto& pho = evt[phoIdx];

  // ---- Build a fastjet object for the photon
  fastjet::PseudoJet g(pho.px(), pho.py(), pho.pz(), pho.e());

  // ---- Radius^2 value
  const double R2max = Riso * Riso;

  // ---- Initializes pT to be zero
  double sumPt = 0.0;

  // ---- Loops through fastjet particles
  for (const auto& pj : particles) {

    // ---- pj.user_index() is the pythia index
    int idx = pj.user_index();
    if (idx < 0 || idx >= evt.size()) continue;

    // ---- exclude photons from the sum (including the photon itself)
    if (idx == phoIdx) continue;

    // ---- cone check
    if (deltaR2(pj, g) < R2max) {
      sumPt += pj.perp();
      if (sumPt >= maxSumPt) return false; // early exit
    }
  }

  return (sumPt < maxSumPt);
}

// =====================================================
// Returns: 21 for gluon, 1 - 6 for quark flavor (abs), 0 if unmatched/unknown
// =====================================================
static inline int tagJetPartonFlavor(const fastjet::PseudoJet& jet,
                                     const std::vector<fastjet::PseudoJet>& hardPartons,
                                     double Rmatch) {
  if (hardPartons.empty()) return 0;

  const double R2max = Rmatch * Rmatch;
  double bestR2 = 1e99;
  int bestPdg = 0;

  for (const auto& hp : hardPartons) {
    const double r2 = deltaR2(jet, hp);
    if (r2 < bestR2) {
      bestR2 = r2;
      bestPdg = hp.user_index(); // we stored pdg id here
    }
  }

  if (bestR2 > R2max) return 0; // no good match

  // Normalize to "quark vs gluon"
  if (bestPdg == 21) return 21;
  if (bestPdg != 0 && std::abs(bestPdg) <= 6) return std::abs(bestPdg);
  return 0;
}

// =====================================================
// Quark ids are [1, 2, 3, 4, 5, 6], so this checks if a particle with a given id is a quark
// =====================================================
static bool isQuark(int id) {
  int a = std::abs(id);
  return (a >= 1 && a <= 6);
}

// =====================================================
// Gluon id is 21, so this checks if a particle with a given id is a gluon
// =====================================================
static bool isGluon(int id) {
  return id == 21;
}

// =====================================================
// Struct for jet and constituients histograms
// =====================================================
struct HistTriplet {
  TH1D* pt  = nullptr;
  TH1D* phi = nullptr;
  TH1D* eta = nullptr;
};

// =====================================================
// Makes name for each histogram (not title)
// =====================================================
static std::string makeName(const std::string& base,
                            int pthatIdx,
                            int jetIdx = -1) {
  std::ostringstream ss;
  ss << base << "_pthat" << pthatIdx;
  if (jetIdx >= 0)
    ss << "_jet" << jetIdx;
  return ss.str();
}

// =====================================================
// Makes title for each (complicated) histogram
// =====================================================
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

// =====================================================
// Find which pt-bin a jet belongs to: [edge[i], edge[i+1})
// =====================================================
static int findPtBin(double pt, const std::vector<double>& edges) {
  if (edges.size() < 2) return -1;
  for (size_t i = 0; i + 1 < edges.size(); ++i) {
    if (pt >= edges[i] && pt < edges[i + 1]) return static_cast<int>(i);
  }
  return -1; // out of range
}

// =====================================================
// Core Analysis Loop
//  - Generate events (Pythia)
//  - Cluster jets (FastJet)
//  - Match jets to hard partons
//  - Fill histograms
//  - Return sigmaGen / nAccepted weight
// =====================================================
double run_events(int nEvents, double ptMin, double ptMax,
                TH1D& hParPt, TH1D& hParPhi, TH1D& hParEta, TH1D& numJetHist, TH1D& numPhotonHist,
                const std::vector<double>& jetPtBins,
                std::vector<HistTriplet>& jetHists,
                std::vector<HistTriplet>& constHists,
                std::vector<int>& quarkJetCount,
                std::vector<int>& totalJetCount,
                bool photonTagging = false
                ) {

  // ---- Settings
  double eCM         = 5020;
  double R           = 0.4;
  double jetPtMin    = 0.0;
  double eta_cut     = 2.8 + R;
  double jet_eta_cut = eta_cut - R;

  // ---- Error check
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

    // =====================================================
    // SETTINGS CHECK: I would check these. I used the settings
    // from Emma's code and the initial code, so I'd check
    // to make sure these are correct.
    // =====================================================
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");
  pythia.readString("Beams:eCM = " + std::to_string(eCM));
  pythia.readString("PhaseSpace:pTHatMin = " + std::to_string(ptMin));
  pythia.readString("PhaseSpace:pTHatMax = " + std::to_string(ptMax));
  pythia.readString("HardQCD:all = on");
  pythia.readString("HadronLevel:all = on");
  pythia.readString("PromptPhoton:all = on");
  pythia.readString("Next:numberShowEvent = 0");
  pythia.readString("Next:numberShowInfo  = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Photon:ProcessType = 0");

  pythia.init();

  // ---- FastJet setup
  fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, R);

  // ---- Event loop for each particle
  for (int iEvt = 0; iEvt < nEvents; ++iEvt) {
    if (!pythia.next()) continue;

    // ---- Initializes fastjet object particles w/ a size of the number of events
    std::vector<fastjet::PseudoJet> particles;
    particles.reserve(pythia.event.size());

    // ---- Initializes fastjet object hardPartons w/ arbitrary size
    std::vector<fastjet::PseudoJet> hardPartons;
    hardPartons.reserve(8);

    // ---- Initializes vector for photon candidates
    std::vector<int> photonCandidates;
    photonCandidates.reserve(2);


    // ---- Particle loop that loops over each particle
    for (int i = 0; i < pythia.event.size(); ++i) {

      // ---- Sets p as the particle
      const auto& p = pythia.event[i];

      // ------------------------------
      // Checks if the particle is an outgoing particle of the hardest subprocess
      // AND whether the particle is a quark or gluon
      // ------------------------------
      if (std::abs(p.status()) == 23 && (p.isQuark() || p.isGluon())) {

        // **p.status() == -23 always**
        fastjet::PseudoJet hp(p.px(), p.py(), p.pz(), p.e());
        hp.set_user_index(p.id());     // PDG id
        hardPartons.push_back(hp);
      }

      // ---- Continues over not final and not visible particles
      if (!p.isFinal())   continue;
      if (!p.isVisible()) continue;

      // ---- Checks if p is a photon, has a pT greater than 50, and meets the eta cut
      // ---- If so, adds the index to the photonCandidates vector
      if (p.id() == 22 && p.pT() > 50 && std::abs(p.eta()) < 2.37) {
        if (std::abs(p.eta()) > 1.37 && std::abs(p.eta()) < 1.52) continue;
        photonCandidates.push_back(i);
      }

      // ---- Sets up a fastjet object for each particle
      fastjet::PseudoJet pj(p.px(), p.py(), p.pz(), p.e());

       // ---- Eta cut
      if (std::abs(pj.eta()) > eta_cut) continue;

      // ---- Adds final and visible particles to the particles vector
      pj.set_user_index(i);
      particles.push_back(pj);

      // ---- Fill particle histograms
      hParPt.Fill(pj.perp());
      hParPhi.Fill(pj.phi_std());
      hParEta.Fill(pj.eta());
    }

    // ---- Continues if no particles
    if (particles.empty()) continue;

    // ---- Reconstructs jets
    fastjet::ClusterSequence cs(particles, jetDef);

    // ---- Sorts jets by pt
    auto jets = fastjet::sorted_by_pt(cs.inclusive_jets(jetPtMin));

    if (photonTagging == true) {
      int count = 0;

      if (photonCandidates.empty()) continue;
      const double dPhiCut = 0.3;

      // ---- Keep only photons that pass isolation
      std::vector<int> goodPhotons;
      for (int phoIdx : photonCandidates) {
        if (!photonIsIsolatedCone03(pythia.event, phoIdx, particles, 0.3, 3.0)) continue;
        goodPhotons.push_back(phoIdx);
      }

      if (goodPhotons.empty()) continue;

      // ---- For each jet, count how many good photons get associated to it
      std::vector<int> jetPhotonCount(jets.size(), 0);

      // ---- Loop over good photons and assign each one to its best jet
      for (int phoIdx : goodPhotons) {
        const auto& pho = pythia.event[phoIdx];

        int bestJetIdx = -1;
        double bestScore = 1e9;

        for (int jIdx = 0; jIdx < (int)jets.size(); ++jIdx) {
          const auto& j = jets[jIdx];

          // ---- Jet acceptance cuts
          if (std::abs(j.eta()) > jet_eta_cut) continue;

          const double jpt = j.perp();
          const int jetBin = findPtBin(jpt, jetPtBins);
          if (jetBin < 0) continue;

          // ---- Away-side requirement
          double dphi = absDPhi(j.phi_std(), pho.phi());
          if (std::abs(dphi - M_PI) > (M_PI / 8)) continue;
          if (dphi < M_PI - dPhiCut) continue;

          // ---- Pick the jet closest to back-to-back
          double score = std::abs(M_PI - dphi);
          if (score < bestScore) {
            bestScore = score;
            bestJetIdx = jIdx;
          }
        }

        // ---- If no acceptable jet matched this photon, skip it
        if (bestJetIdx < 0) continue;

        // ---- Count this photon for that jet
        jetPhotonCount[bestJetIdx]++;
      }

      // ---- Now loop over jets and fill only photon-tagged jets
      for (int jIdx = 0; jIdx < (int)jets.size(); ++jIdx) {
        const auto& j = jets[jIdx];

        if (std::abs(j.eta()) > jet_eta_cut) continue;

        const double jpt = j.perp();
        const int jetBin = findPtBin(jpt, jetPtBins);
        if (jetBin < 0) continue;

        // ---- Only keep jets with at least one associated good photon
        if (jetPhotonCount[jIdx] <= 0) continue;

        count++;

        // ---- Fill photon multiplicity per tagged jet
        numPhotonHist.Fill(jetPhotonCount[jIdx]);

        const int flavor = tagJetPartonFlavor(j, hardPartons, 0.3);

        if (flavor != 0 && std::abs(flavor) <= 6) quarkJetCount[jetBin]++;
        if (flavor != 0) totalJetCount[jetBin]++;

        jetHists[jetBin].pt->Fill(jpt);
        jetHists[jetBin].phi->Fill(j.phi_std());
        jetHists[jetBin].eta->Fill(j.eta());

        for (const auto& c : j.constituents()) {
          constHists[jetBin].pt->Fill(c.perp());
          constHists[jetBin].phi->Fill(c.phi_std());
          constHists[jetBin].eta->Fill(c.eta());
        }
      }

      // ---- Number of photon-tagged jets in this event
      numJetHist.Fill(count);
    }
    
    if (photonTagging == false) {
      int count = 0;
      // ---- Jet Loop
      for (const auto& j : jets) {

        // ---- Eta cut for the jet axis
        if (std::abs(j.eta()) > jet_eta_cut) continue;

        // ---- Sets jet pt and the respective jet pt bin
        const double jpt = j.perp();
        const int jetBin = findPtBin(jpt, jetPtBins);

        // ---- Skips if invalid jet bin
        if (jetBin < 0) continue;

        // ---- Sets flavor equal to 21 for gluon jets, [-3, 6] \ 0 for quark jets, and 0 otherwise
        const int flavor = tagJetPartonFlavor(j, hardPartons, 0.3);

        // ---- Increments quarkJetCount for the respective jet bin if jet is quark-initiated
        if (flavor != 0 && std::abs(flavor) <= 6) quarkJetCount[jetBin]++;

        // ---- Increments totalJetCount if it is a quark or gluon jet
        if (flavor != 0) totalJetCount[jetBin]++;

        // ---- Fill jet hists for this jetPt bin and ptHat bin
        jetHists[jetBin].pt->Fill(jpt);
        jetHists[jetBin].phi->Fill(j.phi_std());
        jetHists[jetBin].eta->Fill(j.eta());

        // ---- Fill constituent hists *for jets in this jetPt bin* and ptHat bin
        for (const auto& c : j.constituents()) {
          constHists[jetBin].pt->Fill(c.perp());
          constHists[jetBin].phi->Fill(c.phi_std());
          constHists[jetBin].eta->Fill(c.eta());
        }
        count++;
        // ---- Breaks after considering the leading jet
        //break;
      }
      numJetHist.Fill(count);
    }
  }
  // ---- Assign cross section and accepted-event count AFTER event generation
  double sigmaGen = pythia.info.sigmaGen();
  int nAccepted   = pythia.info.nAccepted();

  // ---- Stat message
  pythia.stat();

  // ---- Makes sure number of accepted events is positive
  if (nAccepted <= 0) return 0.0;

  // ---- Returns correct slice weight
  return sigmaGen / nAccepted;
}

int main(int argc, char* argv[]) {
  
  // ---- Error check for number of arguments
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <num_events> <photonTagging 0/1>\n";
    return 1;
  }

  // ---- Takes input for number of events
  int nEvents = std::stoi(argv[1]);

  bool photTagging = std::stoi(argv[2]);

  // Creates directory for outputs and root file
  std::filesystem::create_directories("output");
  TFile outFile("output/jet_kinematics.root", "RECREATE");

  // ---- Sets ptHat and jetPt bins
  //std::vector<double> ptHatBins = {20, 35, 50, 65, 80, 100, 120, 140, 160, 180, 200, 220, 245, 270, 295, 320, 350, 380};
  //std::vector<double> jetPtBins = {20, 35, 50, 65, 80, 100, 120, 140, 160, 180, 200, 220, 245, 270, 295, 320, 350, 380};
  std::vector<double> ptHatBins = {20, 80, 400};
  std::vector<double> jetPtBins = {20, 80, 400};

  // ---- Sets number of jetPtBins and ptHatBins
  const size_t nJetBins = jetPtBins.size() - 1;
  const size_t nptHatBins = ptHatBins.size() - 1;

  // ---- Weighted accumulators for the final quark-fraction calculation
  std::vector<double> weightedQuarkJets(nJetBins, 0.0);
  std::vector<double> weightedTotalJets(nJetBins, 0.0);

  // ---- Sum of squared weights for an effective-statistics error bar
  std::vector<double> weightedTotalJets2(nJetBins, 0.0);

  // ---- Initializes a vector for the number of jet bins for the weighted histograms after combining the data from ptHat bins
  std::vector<TH1D*> hJetPt_weighted_combined(nJetBins, nullptr);

  // ---- Number of Jets hist
  TH1D numJetHist(
    "numJetHist",
    "Number of Jets per Event",
    10, 0, 10
  );

  TH1D numPhotonHist(
    "numPhotonHist",
    "Number of Photons per Photon-Tagged Jet",
    25, 0, 25
  );

  // ------------------------------
  // Jet bin loop that creates the name and title
  // for each of the weighted and combined jet pt vectors
  // and stores a blank histogram in each index
  // ------------------------------
  for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {

  // ---- Sets jetLow and jetHigh for each jetBin
  const double jetLow  = jetPtBins[jetBin];
  const double jetHigh = jetPtBins[jetBin + 1];

  // ---- Sets up name for weighted combined histograms
  std::string name = "hJetPt_weighted_jet" + std::to_string(jetBin);
  std::ostringstream title;
  title << std::fixed << std::setprecision(0)
        << "Weighted Jet pT | jet pT [" << jetLow << ", " << jetHigh << ")"
        << ";p_{T} [GeV];d#sigma/dp_{T}";
  hJetPt_weighted_combined[jetBin] = new TH1D(name.c_str(), title.str().c_str(), 200, 0, 400);
  }
  
  // ---- Initializes a 2D array of histograms at a specific ptHat bin and jet bin
  std::vector<std::vector<TH1D*>> hJetPt_weighted(nptHatBins, std::vector<TH1D*>(nJetBins, nullptr));

  
  // ------------------------------
  // Nested loop that loops over the ptHat bins
  // AND jet bins, assigns an empty histogram to each element
  // of the 2D vector, and names/titles each histogram approprietly
  // ------------------------------
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


    // ---- Particle hists for THIS pTHat bin
    // ---- (Particle hists don't depend on jet bin)
    {
      // ---- Erases decimals in title
      std::ostringstream rng;
      rng << std::fixed << std::setprecision(0)
            << "pTHat ["
            << ptHatLow << ", "
            << ptHatHigh
            << ")";

      std::string genTitle = rng.str();

      // ---- Particle Pt
      TH1D hParPt(
        makeName("hParPt", ptHatBin).c_str(),
        ("Particle p_{T} | " + genTitle + ");p_{T};Particles").c_str(),
        200, 0, 200
      );

      // ---- Particle phi
      TH1D hParPhi(
        makeName("hParPhi", ptHatBin).c_str(),
        ("Particle #phi | " + genTitle + ");#phi;Particles").c_str(),
        128, -M_PI, M_PI
      );

      // ---- Particle eta
      TH1D hParEta(
        makeName("hParEta", ptHatBin).c_str(),
        ("Particle #eta | " + genTitle + ");#eta;Particles").c_str(),
        120, -6, 6
      );

      // ------------------------------
      // Build jet/constituient histograms for each jetPt bin* within this pTHat bin
      // ------------------------------

      // ---- Defines vectors for jet Pt, phi, and eta
      std::vector<std::unique_ptr<TH1D>> jetPtH, jetPhiH, jetEtaH;

      // ---- Defines vector for constituients pt, phi, and eta
      std::vector<std::unique_ptr<TH1D>> constPtH, constPhiH, constEtaH;

      // ---- Reserves enough space in each array for the number of histograms necessary for each jetPt bin
      jetPtH.reserve(nJetBins);  jetPhiH.reserve(nJetBins);  jetEtaH.reserve(nJetBins);
      constPtH.reserve(nJetBins); constPhiH.reserve(nJetBins); constEtaH.reserve(nJetBins);

      // ---- Makes a vector of HistTriplets with space for each jetpt bin
      std::vector<HistTriplet> jetHists(nJetBins), constHists(nJetBins);

      // ---- Loops over each jetPt bin
      for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {

        // ---- Defines jet bin min and max
        const double jetLow  = jetPtBins[jetBin];
        const double jetHigh = jetPtBins[jetBin + 1];

        // ---- Adds a histogram to jetPtH
        jetPtH.emplace_back(std::make_unique<TH1D>(
          makeName("hJetPt", ptHatBin, jetBin).c_str(),
          makeFullTitle("Jet p_{T}",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "p_{T}", "Jets").c_str(),
          200, 0, 400
        ));

        // ---- Adds a histogram to JetphiH
        jetPhiH.emplace_back(std::make_unique<TH1D>(
          makeName("hJetPhi", ptHatBin, jetBin).c_str(),
          makeFullTitle("Jet #phi",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "#phi", "Jets").c_str(),
          128, -M_PI, M_PI
        ));

        // ---- Adds a histogram to JetEtaH
        jetEtaH.emplace_back(std::make_unique<TH1D>(
          makeName("hJetEta", ptHatBin, jetBin).c_str(),
          makeFullTitle("Jet #eta",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "#eta", "Jets").c_str(),
          120, -6, 6
        ));

        // ---- Adds a histogram to ConstituentPtH
        constPtH.emplace_back(std::make_unique<TH1D>(
          makeName("hConstPt", ptHatBin, jetBin).c_str(),
          makeFullTitle("Constituent p_{T}",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "p_{T}", "Constituents").c_str(),
          200, 0, 200
        ));

        // ---- Adds a histogram to ConstituentPhiH
        constPhiH.emplace_back(std::make_unique<TH1D>(
          makeName("hConstPhi", ptHatBin, jetBin).c_str(),
          makeFullTitle("Constituent #phi",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "#phi", "Constituents").c_str(),
          128, -M_PI, M_PI
        ));

        // ---- Adds a histogram to ConstituentEtaH
        constEtaH.emplace_back(std::make_unique<TH1D>(
          makeName("hConstEta", ptHatBin, jetBin).c_str(),
          makeFullTitle("Constituent #eta",
                        ptHatLow, ptHatHigh,
                        jetLow, jetHigh,
                        "#eta", "Constituents").c_str(),
          120, -6, 6
        ));
        // ---- Adds a HistTriplet to jetHists (defined outside of the loop) for each jetBin
        jetHists[jetBin] = { jetPtH.back().get(),   jetPhiH.back().get(),   jetEtaH.back().get() };

        // ---- Adds a HistTriplet to constHists (defined outside of the loop) for each jetBin
        constHists[jetBin] = { constPtH.back().get(), constPhiH.back().get(), constEtaH.back().get() };
      }
      // ---- Per-pTHat-bin raw counters
      std::vector<int> nQuarkJets_this(nJetBins, 0);
      std::vector<int> totalJets_this(nJetBins, 0);

      // ---- One Pythia run fills ALL jetPt bins for this pTHat bin and returns the weight
      double weight = run_events(nEvents, ptHatLow, ptHatHigh,
                        hParPt, hParPhi, hParEta, numJetHist, numPhotonHist,
                        jetPtBins,
                        jetHists, constHists,
                        nQuarkJets_this, totalJets_this, photTagging);
      // ---- Error check
      if (weight <= 0) continue; 

      // ---- Accumulate weighted counts for the final quark fraction
      for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
        weightedQuarkJets[jetBin] += weight * nQuarkJets_this[jetBin];
        weightedTotalJets[jetBin] += weight * totalJets_this[jetBin];

        // ---- For approximate error bars later
        weightedTotalJets2[jetBin] += weight * weight * totalJets_this[jetBin];
      }

      // ---- Fills each weighted histogram from its raw data counterpart and applies the weight for the respective ptHat bin
      for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
        hJetPt_weighted[ptHatBin][jetBin]->Add(jetHists[jetBin].pt, weight);
      }

      // ---- Adds weighted data to the combined (over each ptHat bin) weighted data histogram
      for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
        hJetPt_weighted_combined[jetBin]->Add(hJetPt_weighted[ptHatBin][jetBin]);
      }

      // ---- Write particle hists for this pTHat bin
      outFile.cd();
      hParPt.Write(); hParPhi.Write(); hParEta.Write();

      // ---- Write all jet/constituent hists for this pTHat bin
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
  
  outFile.cd();
  numJetHist.Write();
  numPhotonHist.Write();
  // ---- Sets up weighted total JetPt histogram
  TH1D* hJetPt_total = (TH1D*)hJetPt_weighted[0][0]->Clone("hJetPt_total");

  // ---- Clears data from clone
  hJetPt_total->Reset();

  // ---- Loops over each ptHat bin and jet bin and adds all weighted data (full weighted histogram)
  for (size_t ptHatBin = 0; ptHatBin < nptHatBins; ++ptHatBin) {
    for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
      hJetPt_total->Add(hJetPt_weighted[ptHatBin][jetBin]);
    }
  }

  // ---- Sets title
  hJetPt_total->SetTitle("Total Weighted Jet p_{T};p_{T} [GeV];d#sigma/dp_{T}");

  // ---- Writes all weigted histograms
  outFile.cd();
  for (size_t ptHatBin = 0; ptHatBin < nptHatBins; ++ptHatBin) {
    for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
      hJetPt_weighted[ptHatBin][jetBin]->Write();
    }
  }
  
  // ---- Writes combined weighted histograms
  outFile.cd();
  for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
    hJetPt_weighted_combined[jetBin]->Write();
  }
  
  // ---- Frees memory
  for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
    delete hJetPt_weighted_combined[jetBin];
  }

  // ---- Frees memory
  for (size_t ptHatBin = 0; ptHatBin < nptHatBins; ++ptHatBin) {
    for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
      delete hJetPt_weighted[ptHatBin][jetBin];
    }
  }

  // ---- Write total weighted jetPt histogram
  outFile.cd();
  hJetPt_total->Write();
  delete hJetPt_total;

  // ----------------------------
  // Make quark/gluon fraction graphs vs jet pT bin
  // ----------------------------

  // ---- Initializes x and x-error vectors
  std::vector<double> x(nJetBins), ex(nJetBins, 0.0);

  // ---- Initializes quark fraction and associated error
  std::vector<double> fq(nJetBins), efq(nJetBins, 0.0);

  // ---- Jet bin loop
  for (size_t jetBin = 0; jetBin < nJetBins; ++jetBin) {
    const double low  = jetPtBins[jetBin];
    const double high = jetPtBins[jetBin + 1];

    x[jetBin]  = 0.5 * (low + high);
    ex[jetBin] = 0.5 * (high - low);

    const double Nw = weightedTotalJets[jetBin];

    if (Nw > 0.0) {
      const double pq = weightedQuarkJets[jetBin] / Nw;
      fq[jetBin] = pq;

      double Neff = 0.0;
      if (weightedTotalJets2[jetBin] > 0.0) {
        Neff = (Nw * Nw) / weightedTotalJets2[jetBin];
      }

      if (Neff > 0.0) {
        efq[jetBin] = std::sqrt(pq * (1.0 - pq) / Neff);
      } else {
        efq[jetBin] = 0.0;
      }
    } else {
      fq[jetBin] = 0.0;
      efq[jetBin] = 0.0;
    }
  }

  // ---- Sets up TGraph w/ errors
  auto grQuarkFrac = new TGraphErrors((int)nJetBins, x.data(), fq.data(), ex.data(), efq.data());

  // ---- Defines name
  grQuarkFrac->SetName("grQuarkFraction");

  // ---- Defines title
  grQuarkFrac->SetTitle("Quark-initiated jet fraction vs jet p_{T};jet p_{T} bin center [GeV];quark fraction");

  // ---- Write to file
  outFile.cd();
  grQuarkFrac->Write();

  // ---- Frees memory
  delete grQuarkFrac;


  outFile.Close();
  std::cout << "Wrote output/jet_kinematics.root\n";
  return 0;
}