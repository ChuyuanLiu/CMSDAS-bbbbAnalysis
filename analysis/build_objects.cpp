// code used to build the objects starting from the basic inputs of the analysis

// c++ -lm -o build_objects build_objects.cpp -I include/ `root-config --glibs --cflags`

#include<iostream>
#include<string>

#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH1F.h"

#include "analysis_utils.h" // jet_t, init_jet_t
#include "input_tree.h"
#include "output_tree.h"
#include "bbbb_functions.h"

using namespace std;

int main(int argc, char** argv)
{
    // handling the I/O

    if (argc < 3)
    {
        cout << "*** usage : ./build_objects inputFile outputFile isData=0 isSignal=0" << endl;
        return 1;
    }

    string inputFile  = argv[1];
    string outputFile = argv[2];    

    cout << "[INFO] Input  file: " << inputFile << endl;
    TFile* fIn = TFile::Open(inputFile.c_str());
    TTree* tIn = (TTree*) fIn->Get("bbbbTree");

    TriggerEfficiencyCalculator theTriggerEfficiencyCalculator("../trigger/TriggerEfficiencies.root");

    cout << "[INFO] Output file: " << outputFile << endl;
    TFile* fOut = new TFile(outputFile.c_str(), "recreate");
    TTree* tOut = new TTree ("bbbbTree", "bbbbTree");

    bool isData = false;
    bool isSig  = false;
    if (argc >= 4)
        isData = (std::stoi(argv[3]) == 0 ? false : true);
    if (argc >= 5)
        isSig = (std::stoi(argv[4]) == 0 ? false : true);
    cout << "[INFO] Is data?   : " << std::boolalpha << isData << std::noboolalpha << endl;
    cout << "[INFO] Is signal? : " << std::boolalpha << isSig  << std::noboolalpha << endl;

    std::string sample_type = "bkg";
    if (isData) sample_type = "data";
    if (isSig)  sample_type = "sig";
    cout << "[INFO] The tree will be read for type : " << sample_type << endl;

    const float btag_WP_medium = 0.3093; // for DeepJet

    // declaring the output tree - use the output_tree class
    // output variables are defined inside there
    output_tree otree;

    // prepare the input tree reader and run the event loop
    input_tree itree(tIn, sample_type);

    const auto nEv = tIn->GetEntries();
    for (uint iEv = 0; iEv < nEv; ++iEv)
    {
        tIn->GetEntry(iEv);

        if (iEv % 10000 == 0)
            cout << "... doing event " << iEv << " / " << nEv << endl;

        otree.clear_vars();

        // prepare the objects of the 4 jets
        jet_t jet1, jet2, jet3, jet4;
        // the macros below copy the properties of the jet with the specified index (1,2,3,4) to the jet object (jet1,jet2,jet3,jet4)
        init_jet_t(jet1, 1, itree, sample_type) ;
        init_jet_t(jet2, 2, itree, sample_type) ;
        init_jet_t(jet3, 3, itree, sample_type) ;
        init_jet_t(jet4, 4, itree, sample_type) ;

        // compute regressed quantities and other member variables
        jet1.pt_breg = jet1.bRegCorr * jet1.pt;
        jet1.m_breg  = jet1.bRegCorr * jet1.m;
        jet1.p4      .SetPtEtaPhiM(jet1.pt,      jet1.eta, jet1.phi, jet1.m);
        jet1.p4_breg .SetPtEtaPhiM(jet1.pt_breg, jet1.eta, jet1.phi, jet1.m_breg);

        jet2.pt_breg = jet2.bRegCorr * jet2.pt;
        jet2.m_breg  = jet2.bRegCorr * jet2.m;
        jet2.p4      .SetPtEtaPhiM(jet2.pt,      jet2.eta, jet2.phi, jet2.m);
        jet2.p4_breg .SetPtEtaPhiM(jet2.pt_breg, jet2.eta, jet2.phi, jet2.m_breg);

        jet3.pt_breg = jet3.bRegCorr * jet3.pt;
        jet3.m_breg  = jet3.bRegCorr * jet3.m;
        jet3.p4      .SetPtEtaPhiM(jet3.pt,      jet3.eta, jet3.phi, jet3.m);
        jet3.p4_breg .SetPtEtaPhiM(jet3.pt_breg, jet3.eta, jet3.phi, jet3.m_breg);

        jet4.pt_breg = jet4.bRegCorr * jet4.pt;
        jet4.m_breg  = jet4.bRegCorr * jet4.m;
        jet4.p4      .SetPtEtaPhiM(jet4.pt,      jet4.eta, jet4.phi, jet4.m);
        jet4.p4_breg .SetPtEtaPhiM(jet4.pt_breg, jet4.eta, jet4.phi, jet4.m_breg);

        // ========================================
        // ========================================
        // ---- here goes your analysis code to build the high level objects
        // cout << "JET 1 : " << jet1.pt << " " << jet1.eta << " " << jet1.genjet_pt << " " << endl;
        // cout << "JET 2 : " << jet2.pt << " " << jet2.eta << " " << jet2.genjet_pt << " " << endl;
        // cout << "JET 3 : " << jet3.pt << " " << jet3.eta << " " << jet3.genjet_pt << " " << endl;
        // cout << "JET 4 : " << jet4.pt << " " << jet4.eta << " " << jet4.genjet_pt << " " << endl;
        // cout << " -------------------------------------- " << endl;

        // pair the jets
        std::vector<jet_t> jets {jet1, jet2, jet3, jet4};
        std::vector<jet_t> result = bbbb_jets_idxs_BothClosestToDiagonal(&jets);

        TLorentzVector v_H1, v_H2, v_HH;
        v_H1 = result.at(0).p4_breg + result.at(1).p4_breg;
        v_H2 = result.at(2).p4_breg + result.at(3).p4_breg;

        // order them by highest pT: pt(H1) > pt(H2)

        jet_t H1_b1 = result.at(0);
        jet_t H1_b2 = result.at(1);
        jet_t H2_b1 = result.at(2);
        jet_t H2_b2 = result.at(3);

        if (v_H1.Pt() < v_H2.Pt()){
            std::swap(v_H1, v_H2);
            std::swap(H1_b1, H2_b1);
            std::swap(H1_b2, H2_b2);
        }

        v_HH = v_H1 + v_H2;
        // ========================================
        // ========================================

        // copy to the output
        otree.H1_pt_  = v_H1.Pt();
        otree.H1_eta_ = v_H1.Eta();
        otree.H1_phi_ = v_H1.Phi();
        otree.H1_m_   = v_H1.M();

        otree.H2_pt_  = v_H2.Pt();
        otree.H2_eta_ = v_H2.Eta();
        otree.H2_phi_ = v_H2.Phi();
        otree.H2_m_   = v_H2.M();

        otree.HH_pt_  = v_HH.Pt();
        otree.HH_eta_ = v_HH.Eta();
        otree.HH_phi_ = v_HH.Phi();
        otree.HH_m_   = v_HH.M();

        otree.H1_b1_pt_  = H1_b1.p4_breg.Pt();
        otree.H1_b1_eta_ = H1_b1.p4_breg.Eta();
        otree.H1_b1_phi_ = H1_b1.p4_breg.Phi();
        otree.H1_b1_m_   = H1_b1.p4_breg.M();

        otree.H1_b2_pt_  = H1_b2.p4_breg.Pt();
        otree.H1_b2_eta_ = H1_b2.p4_breg.Eta();
        otree.H1_b2_phi_ = H1_b2.p4_breg.Phi();
        otree.H1_b2_m_   = H1_b2.p4_breg.M();

        otree.H2_b1_pt_  = H2_b1.p4_breg.Pt();
        otree.H2_b1_eta_ = H2_b1.p4_breg.Eta();
        otree.H2_b1_phi_ = H2_b1.p4_breg.Phi();
        otree.H2_b1_m_   = H2_b1.p4_breg.M();

        otree.H2_b2_pt_  = H2_b2.p4_breg.Pt();
        otree.H2_b2_eta_ = H2_b2.p4_breg.Eta();
        otree.H2_b2_phi_ = H2_b2.p4_breg.Phi();
        otree.H2_b2_m_   = H2_b2.p4_breg.M();

        otree.H1H2_deltaEta_   = std::abs( v_H1.Eta() - v_H2.Eta() );
        otree.H1H2_deltaPhi_   = v_H1.DeltaPhi(v_H2);

        // boost H1 to the bbbb CM
        TLorentzVector vH1_cm = v_H1;
        vH1_cm.Boost(-v_HH.BoostVector());
        otree.H1_costhetaCM_  = vH1_cm.CosTheta();

        // and fw from the input
        otree.run_                = **(itree.run);
        otree.luminosityBlock_    = **(itree.luminosityBlock);
        otree.event_              = **(itree.event);
        otree.xs_                 = **(itree.xs);

        otree.btag_SF_            = **(itree.btag_SF);
        otree.btag_SF_bup_        = **(itree.btag_SF_bup);
        otree.btag_SF_bdown_      = **(itree.btag_SF_bdown);
        otree.btag_SF_cup_        = **(itree.btag_SF_cup);
        otree.btag_SF_cdown_      = **(itree.btag_SF_cdown);
        otree.btag_SF_lightup_    = **(itree.btag_SF_lightup);
        otree.btag_SF_lightdown_  = **(itree.btag_SF_lightdown);
        otree.norm_weight_        = **(itree.norm_weight);

        otree.n_btag_ = 0;
        if (result.at(0).btagscore > btag_WP_medium) ++otree.n_btag_;
        if (result.at(1).btagscore > btag_WP_medium) ++otree.n_btag_;
        if (result.at(2).btagscore > btag_WP_medium) ++otree.n_btag_;
        if (result.at(3).btagscore > btag_WP_medium) ++otree.n_btag_;

        otree.rndm_1_ = **(itree.rndm_1);
        otree.rndm_2_ = **(itree.rndm_2);
        otree.rndm_3_ = **(itree.rndm_3);
        
        // calculate the triggerSF following the twiki indications
        if(isData) otree.trigger_SF_ = 1.;
        else
        {
            // Pay attention to provide the correct variable to estract the trigger efficiency!!
            std::vector<float> jetPtVector {jet1.pt, jet2.pt, jet3.pt, jet4.pt};
            // Remember to order the 4 jets by pT!! (search for std::sort)
            // std::sort...; 
            std::sort(jetPtVector.begin(), jetPtVector.end(), std::greater<>());

            // Estract the efficiency for the four filters considered in data
            float dataEfficiency_Double90Quad30_QuadCentralJet30 = theTriggerEfficiencyCalculator.getDataEfficiency_Double90Quad30_QuadCentralJet30(jetPtVector[3]);
            float dataEfficiency_Double90Quad30_DoubleCentralJet90 = theTriggerEfficiencyCalculator.getDataEfficiency_Double90Quad30_DoubleCentralJet90(jetPtVector[1]);
            float dataEfficiency_Quad45_QuadCentralJet45 = theTriggerEfficiencyCalculator.getDataEfficiency_Quad45_QuadCentralJet45(jetPtVector[3]);
            float dataEfficiency_And_QuadCentralJet45 = theTriggerEfficiencyCalculator.getDataEfficiency_And_QuadCentralJet45(jetPtVector[3]);
            // Calculate data total efficiency
            float dataEfficiency_Double90Quad30 = (dataEfficiency_Double90Quad30_QuadCentralJet30 * dataEfficiency_Double90Quad30_DoubleCentralJet90);//-999.;
	        //float dataEfficiency = -999.;
            float dataEfficiency = (dataEfficiency_Double90Quad30) + dataEfficiency_Quad45_QuadCentralJet45 - (dataEfficiency_Double90Quad30 * dataEfficiency_And_QuadCentralJet45); //-999.;

            // Estract the efficiency for the four filters considered in mc
            float mcEfficiency_Double90Quad30_QuadCentralJet30 = theTriggerEfficiencyCalculator.getMcEfficiency_Double90Quad30_QuadCentralJet30(jetPtVector[3]);
            float mcEfficiency_Double90Quad30_DoubleCentralJet90 = theTriggerEfficiencyCalculator.getMcEfficiency_Double90Quad30_DoubleCentralJet90(jetPtVector[1]);
            float mcEfficiency_Quad45_QuadCentralJet45 = theTriggerEfficiencyCalculator.getMcEfficiency_Quad45_QuadCentralJet45(jetPtVector[3]);
            float mcEfficiency_And_QuadCentralJet45 = theTriggerEfficiencyCalculator.getMcEfficiency_And_QuadCentralJet45(jetPtVector[3]);
            // Calculate data total efficiency
            float mcEfficiency_Double90Quad30 = (mcEfficiency_Double90Quad30_QuadCentralJet30 * mcEfficiency_Double90Quad30_DoubleCentralJet90);
            float mcEfficiency = (mcEfficiency_Double90Quad30) + mcEfficiency_Quad45_QuadCentralJet45 - (mcEfficiency_Double90Quad30 * mcEfficiency_And_QuadCentralJet45); //-999.;

            // Calculate the trigger scale factor (data/mc)
             otree.trigger_SF_ = dataEfficiency / mcEfficiency;
            
            otree.Jet1Pt = jetPtVector[0];
            otree.Jet2Pt = jetPtVector[1];
            otree.Jet3Pt = jetPtVector[2];
            otree.Jet4Pt = jetPtVector[3];
            
            otree.dataEfficiency_Double90Quad30_QuadCentralJet30 = dataEfficiency_Double90Quad30_QuadCentralJet30;
            otree.dataEfficiency_Double90Quad30_DoubleCentralJet90 = dataEfficiency_Double90Quad30_DoubleCentralJet90;
            otree.dataEfficiency_Quad45_QuadCentralJet45 = dataEfficiency_Quad45_QuadCentralJet45;
            otree.dataEfficiency_And_QuadCentralJet45 = dataEfficiency_And_QuadCentralJet45;
            otree.dataEfficiency = dataEfficiency;

            otree.mcEfficiency_Double90Quad30_QuadCentralJet30 = mcEfficiency_Double90Quad30_QuadCentralJet30;
            otree.mcEfficiency_Double90Quad30_DoubleCentralJet90 = mcEfficiency_Double90Quad30_DoubleCentralJet90;
            otree.mcEfficiency_Quad45_QuadCentralJet45 = mcEfficiency_Quad45_QuadCentralJet45;
            otree.mcEfficiency_And_QuadCentralJet45 = mcEfficiency_And_QuadCentralJet45;
            otree.mcEfficiency = mcEfficiency;
            
        }
        

        otree.fill();
    }

    // save to output
    fOut->cd();
    otree.write();

    fIn->Close();
    fOut->Close();

}
