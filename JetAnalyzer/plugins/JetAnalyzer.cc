//  Jet tuple producer for 13 TeV Run2 MC samples
//  Specifically aimed at studies of gluon and light quark jets
//  Data is saved to file on a jet-by-jet basis, resulting in almost flat tuples
//
//  Author: Kimmo Kallonen
//  Based on previous work by: Petra-Maria Ekroos

#include "JetAnalyzer.h"
//#include "TH1D.h"
//#include "TH2D.h"

JetAnalyzer::JetAnalyzer(const edm::ParameterSet& iConfig):
    vtxToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
    jetToken_(consumes<pat::JetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
    pfToken_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCands"))),
    EDMGenJetsToken_(consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("genJets"))),
    genEventInfoToken_(consumes <GenEventInfoProduct>(iConfig.getParameter<edm::InputTag>("genEventInfo"))),
    pileupInfoToken_(consumes <std::vector<PileupSummaryInfo>> (iConfig.getParameter<edm::InputTag>("pileupInfo"))),
    pfRhoAllToken_(consumes <double> (iConfig.getParameter<edm::InputTag>("pfRhoAll"))),
    pfRhoCentralToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("pfRhoCentral"))),
    pfRhoCentralNeutralToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("pfRhoCentralNeutral"))),
    pfRhoCentralChargedPileUpToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("pfRhoCentralChargedPileUp"))),
    qglToken_(consumes<edm::ValueMap<float>>(edm::InputTag("QGTagger", "qgLikelihood"))),
    ptDToken_(consumes<edm::ValueMap<float>>(edm::InputTag("QGTagger", "ptD"))),
    axis2Token_(consumes<edm::ValueMap<float>>(edm::InputTag("QGTagger", "axis2"))),
    multToken_(consumes<edm::ValueMap<int>>(edm::InputTag("QGTagger", "mult"))),
	executionMode (iConfig.getUntrackedParameter<int>("executionMode"))
	
{
    goodVtxNdof = iConfig.getParameter<double>("confGoodVtxNdof");
    goodVtxZ = iConfig.getParameter<double>("confGoodVtxZ");
    goodVtxRho = iConfig.getParameter<double>("confGoodVtxRho");
}

JetAnalyzer::~JetAnalyzer()
{}

void JetAnalyzer::beginJob()
{

	correctParticlesInReco=0;
	correctParticlesNotInReco=0;
	incorrectParticlesInReco=0;
	incorrectParticlesNotInReco=0;
}

void JetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    edm::Handle<reco::VertexCollection> vertices;
    iEvent.getByToken(vtxToken_, vertices);
    edm::Handle<pat::JetCollection> jets;
    iEvent.getByToken(jetToken_, jets);
    edm::Handle<pat::PackedCandidateCollection> pfs;
    iEvent.getByToken(pfToken_, pfs);
    edm::Handle<reco::GenJetCollection> genJets;
    iEvent.getByToken(EDMGenJetsToken_, genJets);
    edm::Handle<GenEventInfoProduct> genEventInfo;
    iEvent.getByToken(genEventInfoToken_, genEventInfo);
    edm::Handle<std::vector< PileupSummaryInfo >>  puInfo;
    iEvent.getByToken(pileupInfoToken_, puInfo);

    edm::Handle<double> pfRhoAllHandle;
    iEvent.getByToken(pfRhoAllToken_, pfRhoAllHandle);
    edm::Handle<double> pfRhoCentralHandle;
    iEvent.getByToken(pfRhoCentralToken_, pfRhoCentralHandle);
    edm::Handle<double> pfRhoCentralNeutralHandle;
    iEvent.getByToken(pfRhoCentralNeutralToken_, pfRhoCentralNeutralHandle);
    edm::Handle<double> pfRhoCentralChargedPileUpHandle;
    iEvent.getByToken(pfRhoCentralChargedPileUpToken_, pfRhoCentralChargedPileUpHandle);

    edm::Handle<edm::ValueMap<float>> qglHandle;
    iEvent.getByToken(qglToken_, qglHandle);
    edm::Handle<edm::ValueMap<float>> ptDHandle;
    iEvent.getByToken(ptDToken_, ptDHandle);
    edm::Handle<edm::ValueMap<float>> axis2Handle;
    iEvent.getByToken(axis2Token_, axis2Handle);
    edm::Handle<edm::ValueMap<int>> multHandle;
    iEvent.getByToken(multToken_, multHandle);

    // Create vectors for the jets
    // sortedJets include all jets of the event, while selectedJets have pT and eta cuts
    vector<JetIndexed> sortedJets;
    vector<JetIndexed> selectedJets;

    // Loop over the jets to save them to the jet vectors for pT-ordering
    //std::cout << "Here A" << std::endl;
    int iJetR = -1;
    for(pat::JetCollection::const_iterator jetIt = jets->begin(); jetIt!=jets->end(); ++jetIt) {
        const pat::Jet &jet = *jetIt;
        std::vector<std::string> jec = jet.availableJECSets();
        //for (std::vector<std::string>::const_iterator i = jec.begin(); i != jec.end(); ++i)
        //	std::cout << *i << std::endl;        

        //std::cout << std::endl;
        ++iJetR;
        sortedJets.push_back( JetIndexed( jet, iJetR) );
        // Select
        if ( (jet.pt() > 30) ) {// && (fabs(jet.eta()) < 2.5) ) {
			selectedJets.push_back( JetIndexed( jet, iJetR) );
        }
    }
    //std::cout << "Here B" << std::endl;

    // Sort the jets in pT order
    std::sort(sortedJets.begin(), sortedJets.end(), higher_pT_sort());
    std::sort(selectedJets.begin(), selectedJets.end(), higher_pT_sort());


    //math::XYZTLorentzVector rawRecoP4(0,0,0,0);
    //std::cout << "Here C" << std::endl;
    // Loop over the pT-ordered selected jets and save them to file
    for (unsigned int ptIdx = 0; ptIdx < selectedJets.size(); ++ptIdx) {
        math::XYZTLorentzVector rawRecoP4(0,0,0,0);

        //std::cout << "Here D" << std::endl;
        // Make selective cuts on the event level
        if (sortedJets.size() < 2) continue;
        //if (fabs(sortedJets[0].jet.eta()) > 2.5 || fabs(sortedJets[1].jet.eta()) > 2.5) continue;
        if (fabs(sortedJets[0].jet.pt()) < 30 || fabs(sortedJets[1].jet.pt()) < 30) continue;

        JetIndexed idxJet = selectedJets[ptIdx];
        const pat::Jet j = idxJet.jet;
        int iJetRef = idxJet.eventIndex;

        // Jet variables
        jetPt = j.pt();
        jetEta = j.eta();
        jetPhi = j.phi();
        jetMass = j.mass();
        jetArea = j.jetArea();

        jetRawPt = j.correctedJet("Uncorrected").pt();
        jetRawMass = j.correctedJet("Uncorrected").mass();

        jetChargedHadronMult = j.chargedHadronMultiplicity();
        jetNeutralHadronMult = j.neutralHadronMultiplicity();
        jetChargedMult = j.chargedMultiplicity();
        jetNeutralMult = j.neutralMultiplicity();

        jetPtOrder = ptIdx;

        // Determine jet IDs
        jetLooseID = 0;
        jetTightID = 0;

        Float_t nhf = j.neutralHadronEnergyFraction();
        Float_t nemf = j.neutralEmEnergyFraction();
        Float_t chf = j.chargedHadronEnergyFraction();
        Float_t cemf = j.chargedEmEnergyFraction();
        unsigned int numconst = j.chargedMultiplicity() + j.neutralMultiplicity();
        unsigned int chm = j.chargedMultiplicity();

        if (abs(j.eta())<=2.7 && (numconst>1 && nhf<0.99 && nemf<0.99) && ((abs(j.eta())<=2.4 && chf>0 && chm>0 && cemf<0.99) || abs(j.eta())>2.4)) {
            jetLooseID = 1;
            if (nhf<0.90 && nemf<0.90) {
                jetTightID = 1;
            }
        }
        //std::cout << "Here E" << std::endl;
        // Add variables for deltaPhi and deltaEta for the two leading jets of the event
        dPhiJetsLO = deltaPhi(sortedJets[0].jet.phi(), sortedJets[1].jet.phi());
        dEtaJetsLO = sortedJets[0].jet.eta() - sortedJets[1].jet.eta();
        //std::cout << "Here E1" << std::endl;


        // The alpha variable is the third jet's pT divided by the average of the two leading jets' pT
        alpha = 0;
        // Make sure that there are at least 3 jets in the event
        if(sortedJets.size() > 2) {
                Float_t leadingPtAvg = (sortedJets[0].jet.pt() + sortedJets[1].jet.pt()) * 0.5;
                alpha = sortedJets[2].jet.pt() / leadingPtAvg;
        }
        //std::cout << "Here E2" << std::endl;

        // Assign flavors for each jet using three different flavor definitions
        partonFlav = j.partonFlavour();
        hadronFlav = j.hadronFlavour();

        physFlav = 0;
        if (j.genParton()) physFlav = j.genParton()->pdgId();
        //std::cout << "Here E3" << std::endl;

        // For convenience, save variables distinguishing gluon, light quark and other jets
        isPartonUDS = 0;
        isPartonG = 0;
        isPartonOther = 0;
        isPhysUDS = 0;
        isPhysG = 0;
        isPhysOther = 0;

        // Physics definition for flavors
        if(abs(physFlav) == 1 || abs(physFlav) == 2 || abs(physFlav) == 3) {
            isPhysUDS = 1;
        } else if(abs(physFlav) == 21) {
            isPhysG = 1;
        } else {
            isPhysOther = 1;
        }

        // Parton definition for flavors
        if(abs(partonFlav) == 1 || abs(partonFlav) == 2 || abs(partonFlav) == 3) {
            isPartonUDS = 1;
        } else if(abs(partonFlav) == 21) {
            isPartonG = 1;
        } else {
            isPartonOther = 1;
        }

        edm::RefToBase<pat::Jet> jetRef(edm::Ref<pat::JetCollection>(jets, iJetRef));

        // Add event information to the jet-based tree
        event = iEvent.id().event();
        run = iEvent.id().run();
        lumi = iEvent.id().luminosityBlock();
        //std::cout << "Here E6" << std::endl;

        eventJetMult = selectedJets.size();



        // Loop over the PF candidates contained inside the jet, first sorting them in pT order
        std::vector<reco::CandidatePtr> pfCands = j.daughterPtrVector();
		
		std::sort(pfCands.begin(), pfCands.end(), [](const reco::CandidatePtr &p1, const reco::CandidatePtr &p2) {return p1->pt() > p2->pt(); });
        int njetpf = 0;

        // Create a PF map for easier matching later
        std::map<const pat::PackedCandidate*, const pat::PackedCandidate> pfMap;

        // Here the jet girth is also calculated
        jetGirth = 0;
		nPFR = 0;
        unsigned int pfCandsSize = pfCands.size();
        for (unsigned int i = 0; i < pfCandsSize; ++i) {
            const pat::PackedCandidate &pf = dynamic_cast<const pat::PackedCandidate &>(*pfCands[i]);
            const pat::PackedCandidate* pfPointer = &pf;
            pfMap.insert(std::pair <const pat::PackedCandidate*, const pat::PackedCandidate> (pfPointer, pf));

			rawRecoP4 += pf.p4()*pf.puppiWeightNoLep();
			++njetpf;
        }

		
        // Generator level jet variables and its constituents
        jetGenMatch = 0;
        genJetPt = 0;
        genJetEta = 0;
        genJetPhi = 0;
        genJetMass = 0;
	genJetE = 0;
        
        // Check if the jet has a matching generator level jet
        if(j.genJet()) {
            jetGenMatch = 1;
            const reco::GenJet* gj = j.genJet();
		
	    genJetPt = gj->pt();
            genJetEta = gj->eta();
            genJetPhi = gj->phi();
            genJetMass = gj->mass();
	    genJetE = gj->p4().E();
			
            
        }
        
		if(jetGenMatch==1){
			std::ofstream myfile;
			myfile.open ("mlData.txt", std::ios_base::app);
		
		
			myfile << rawRecoP4.pt() << "\t";
			myfile << rawRecoP4.eta() << "\t";
			myfile << rawRecoP4.phi() << "\t";
			myfile << rawRecoP4.M() << "\t"; 
			
			myfile << j.pt() << "\t";
			myfile << j.eta() << "\t";
			myfile << j.phi() << "\t";
			myfile << j.p4().M() << "\t"; 
			
			myfile << genJetPt << "\t";
			myfile << genJetEta << "\t";
			myfile << genJetPhi << "\t";
			myfile << genJetMass << "\t"; 
			
			myfile << partonFlav << "\t";
			myfile << hadronFlav << "\t";
			myfile << physFlav << "\n";
			myfile.close();	
		}
	}
}


void JetAnalyzer::endJob() {
	
}
