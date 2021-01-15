#ifndef OUTPUT_TREE_H
#define OUTPUT_TREE_H

#include "TTree.h"

// to add a variable
// 1) add it to the list of declared member variables
// 2) add the corresponding SetBranchAddress in init()
// 3) add a clear default value in clear()

class output_tree {
    public:
        output_tree();
        ~output_tree();

        void clear_vars();
        
        int fill()  {return tree_->Fill();}
        int write() {return tree_->Write();}
        TTree* get_tree() {return tree_.get();}

        // here list all the variables
        unsigned int run_;              
        unsigned int luminosityBlock_;  
        long long    event_;            
        float        xs_;               

        float        btag_SF_;            
        float        btag_SF_bup_;        
        float        btag_SF_bdown_;      
        float        btag_SF_cup_;        
        float        btag_SF_cdown_;      
        float        btag_SF_lightup_;    
        float        btag_SF_lightdown_;  
        float        norm_weight_;      
        float        trigger_SF_;  
        
        float dataEfficiency_Double90Quad30_QuadCentralJet30 ;
        float dataEfficiency_Double90Quad30_DoubleCentralJet90 ;
        float dataEfficiency_Quad45_QuadCentralJet45 ;
        float dataEfficiency_And_QuadCentralJet45 ;
        float dataEfficiency ;

        float mcEfficiency_Double90Quad30_QuadCentralJet30 ;
        float mcEfficiency_Double90Quad30_DoubleCentralJet90 ;
        float mcEfficiency_Quad45_QuadCentralJet45 ;
        float mcEfficiency_And_QuadCentralJet45 ;
        float mcEfficiency ;
    
        float Jet1Pt;
        float Jet2Pt;
        float Jet3Pt;
        float Jet4Pt;

        int  n_btag_;

        // composite candidates
        float H1_pt_;
        float H1_eta_;
        float H1_phi_;
        float H1_m_;

        float H2_pt_;
        float H2_eta_;
        float H2_phi_;
        float H2_m_;

        float HH_pt_;
        float HH_eta_;
        float HH_phi_;
        float HH_m_;

        // other variables
        float H1H2_deltaEta_;
        float H1H2_deltaPhi_;
        float H1_costhetaCM_;

        // jets
        float H1_b1_pt_;
        float H1_b1_eta_;
        float H1_b1_phi_;
        float H1_b1_m_;

        float H1_b2_pt_;
        float H1_b2_eta_;
        float H1_b2_phi_;
        float H1_b2_m_;

        float H2_b1_pt_;
        float H2_b1_eta_;
        float H2_b1_phi_;
        float H2_b1_m_;

        float H2_b2_pt_;
        float H2_b2_eta_;
        float H2_b2_phi_;
        float H2_b2_m_;

        float rndm_1_;
        float rndm_2_;
        float rndm_3_;


    private:
        std::unique_ptr<TTree> tree_;
        void init();
};  

output_tree::output_tree()
{
    init();
}

output_tree::~output_tree()
{
    tree_.release(); // to avoid crashes at the end of execution
}

void output_tree::init()
{
    tree_ = std::unique_ptr<TTree>(new TTree("bbbbTree", "bbbbTree"));

    // create branches

    tree_ -> Branch ("run",              &run_);
    tree_ -> Branch ("luminosityBlock",  &luminosityBlock_);
    tree_ -> Branch ("event",            &event_);
    tree_ -> Branch ("xs",               &xs_);

    tree_ ->Branch ("btag_SF",           &btag_SF_);
    tree_ ->Branch ("btag_SF_bup",       &btag_SF_bup_);
    tree_ ->Branch ("btag_SF_bdown",     &btag_SF_bdown_);
    tree_ ->Branch ("btag_SF_cup",       &btag_SF_cup_);
    tree_ ->Branch ("btag_SF_cdown",     &btag_SF_cdown_);
    tree_ ->Branch ("btag_SF_lightup",   &btag_SF_lightup_);
    tree_ ->Branch ("btag_SF_lightdown", &btag_SF_lightdown_);
    tree_ ->Branch ("norm_weight",       &norm_weight_);
    tree_ ->Branch ("trigger_SF",        &trigger_SF_);
    
    tree_ ->Branch ("dataEfficiency_Double90Quad30_QuadCentralJet30",&dataEfficiency_Double90Quad30_QuadCentralJet30);
    tree_ ->Branch ("dataEfficiency_Double90Quad30_DoubleCentralJet90",&dataEfficiency_Double90Quad30_DoubleCentralJet90);
    tree_ ->Branch ("dataEfficiency_Quad45_QuadCentralJet45",&dataEfficiency_Quad45_QuadCentralJet45);
    tree_ ->Branch ("dataEfficiency_And_QuadCentralJet45",&dataEfficiency_And_QuadCentralJet45);
    tree_ ->Branch ("dataEfficiency",&dataEfficiency);

    tree_ ->Branch ("mcEfficiency_Double90Quad30_QuadCentralJet30",
        &mcEfficiency_Double90Quad30_QuadCentralJet30);
    tree_ ->Branch ("mcEfficiency_Double90Quad30_DoubleCentralJet90",&mcEfficiency_Double90Quad30_DoubleCentralJet90);
    tree_ ->Branch ("mcEfficiency_Quad45_QuadCentralJet45",&mcEfficiency_Quad45_QuadCentralJet45);
    tree_ ->Branch ("mcEfficiency_And_QuadCentralJet45",&mcEfficiency_And_QuadCentralJet45);
    tree_ ->Branch ("mcEfficiency",&mcEfficiency);
    
    tree_ ->Branch ("Jet1Pt",  &Jet1Pt);
    tree_ ->Branch ("Jet2Pt",  &Jet2Pt);
    tree_ ->Branch ("Jet3Pt",  &Jet3Pt);
    tree_ ->Branch ("Jet4Pt",  &Jet4Pt);



    tree_ ->Branch ("dataEfficiency_Double90Quad30_QuadCentralJet30",&dataEfficiency_Double90Quad30_QuadCentralJet30);
    tree_ ->Branch ("dataEfficiency_Double90Quad30_DoubleCentralJet90",&dataEfficiency_Double90Quad30_DoubleCentralJet90);
    tree_ ->Branch ("dataEfficiency_Quad45_QuadCentralJet45",&dataEfficiency_Quad45_QuadCentralJet45);
    tree_ ->Branch ("dataEfficiency_And_QuadCentralJet45",&dataEfficiency_And_QuadCentralJet45);
    tree_ ->Branch ("dataEfficiency",&dataEfficiency);

    tree_ ->Branch ("mcEfficiency_Double90Quad30_QuadCentralJet30",
    &mcEfficiency_Double90Quad30_QuadCentralJet30);
    tree_ ->Branch ("mcEfficiency_Double90Quad30_DoubleCentralJet90",&mcEfficiency_Double90Quad30_DoubleCentralJet90);
    tree_ ->Branch ("mcEfficiency_Quad45_QuadCentralJet45",&mcEfficiency_Quad45_QuadCentralJet45);
    tree_ ->Branch ("mcEfficiency_And_QuadCentralJet45",&mcEfficiency_And_QuadCentralJet45);
    tree_ ->Branch ("mcEfficiency",&mcEfficiency);



    tree_ ->Branch ("n_btag",  &n_btag_);

    tree_ -> Branch ("H1_pt",  &H1_pt_);
    tree_ -> Branch ("H1_eta", &H1_eta_);
    tree_ -> Branch ("H1_phi", &H1_phi_);
    tree_ -> Branch ("H1_m",   &H1_m_);
    
    tree_ -> Branch ("H2_pt",  &H2_pt_);
    tree_ -> Branch ("H2_eta", &H2_eta_);
    tree_ -> Branch ("H2_phi", &H2_phi_);
    tree_ -> Branch ("H2_m",   &H2_m_);
    
    tree_ -> Branch ("HH_pt",  &HH_pt_);
    tree_ -> Branch ("HH_eta", &HH_eta_);
    tree_ -> Branch ("HH_phi", &HH_phi_);
    tree_ -> Branch ("HH_m",   &HH_m_);

    tree_ -> Branch ("H1H2_deltaEta", &H1H2_deltaEta_);
    tree_ -> Branch ("H1H2_deltaPhi", &H1H2_deltaPhi_);
    tree_ -> Branch ("H1_costhetaCM", &H1_costhetaCM_);

    tree_ -> Branch ("H1_b1_pt",  &H1_b1_pt_);
    tree_ -> Branch ("H1_b1_eta", &H1_b1_eta_);
    tree_ -> Branch ("H1_b1_phi", &H1_b1_phi_);
    tree_ -> Branch ("H1_b1_m",   &H1_b1_m_);

    tree_ -> Branch ("H1_b2_pt",  &H1_b2_pt_);
    tree_ -> Branch ("H1_b2_eta", &H1_b2_eta_);
    tree_ -> Branch ("H1_b2_phi", &H1_b2_phi_);
    tree_ -> Branch ("H1_b2_m",   &H1_b2_m_);

    tree_ -> Branch ("H2_b1_pt",  &H2_b1_pt_);
    tree_ -> Branch ("H2_b1_eta", &H2_b1_eta_);
    tree_ -> Branch ("H2_b1_phi", &H2_b1_phi_);
    tree_ -> Branch ("H2_b1_m",   &H2_b1_m_);

    tree_ -> Branch ("H2_b2_pt",  &H2_b2_pt_);
    tree_ -> Branch ("H2_b2_eta", &H2_b2_eta_);
    tree_ -> Branch ("H2_b2_phi", &H2_b2_phi_);
    tree_ -> Branch ("H2_b2_m",   &H2_b2_m_);

    tree_ -> Branch ("rndm_1", &rndm_1_);
    tree_ -> Branch ("rndm_2", &rndm_2_);
    tree_ -> Branch ("rndm_3", &rndm_3_);

}

void output_tree::clear_vars()
{
    run_               = 0;
    luminosityBlock_   = 0;
    event_             = -999;
    xs_                = -999;
    
    btag_SF_           = -999;
    btag_SF_bup_       = -999;
    btag_SF_bdown_     = -999;
    btag_SF_cup_       = -999;
    btag_SF_cdown_     = -999;
    btag_SF_lightup_   = -999;
    btag_SF_lightdown_ = -999;
    norm_weight_       = -999;
    trigger_SF_        = -999;
    
    dataEfficiency_Double90Quad30_QuadCentralJet30 = -999;
    dataEfficiency_Double90Quad30_DoubleCentralJet90 = -999;
    dataEfficiency_Quad45_QuadCentralJet45 = -999;
    dataEfficiency_And_QuadCentralJet45= -999;
    dataEfficiency=-999;

    mcEfficiency_Double90Quad30_QuadCentralJet30=-999;
    mcEfficiency_Double90Quad30_DoubleCentralJet90=-999;
    mcEfficiency_Quad45_QuadCentralJet45=-999;
    mcEfficiency_And_QuadCentralJet45=-999;
    mcEfficiency=-999;
    
    Jet1Pt = -999;
    Jet2Pt = -999;
    Jet3Pt = -999;
    Jet4Pt = -999;

    n_btag_ = -999;

    H1_pt_  = -999;
    H1_eta_ = -999;
    H1_phi_ = -999;
    H1_m_   = -999;

    H2_pt_  = -999;
    H2_eta_ = -999;
    H2_phi_ = -999;
    H2_m_   = -999;

    HH_pt_  = -999;
    HH_eta_ = -999;
    HH_phi_ = -999;
    HH_m_   = -999;

    H1H2_deltaEta_ = -999;
    H1H2_deltaPhi_ = -999;
    H1_costhetaCM_ = -999;

    H1_b1_pt_  = -999;
    H1_b1_eta_ = -999;
    H1_b1_phi_ = -999;
    H1_b1_m_   = -999;

    H1_b2_pt_  = -999;
    H1_b2_eta_ = -999;
    H1_b2_phi_ = -999;
    H1_b2_m_   = -999;

    H2_b1_pt_  = -999;
    H2_b1_eta_ = -999;
    H2_b1_phi_ = -999;
    H2_b1_m_   = -999;

    H2_b2_pt_  = -999;
    H2_b2_eta_ = -999;
    H2_b2_phi_ = -999;
    H2_b2_m_   = -999;

    rndm_1_ = -999;
    rndm_2_ = -999;
    rndm_3_ = -999;
}

#endif