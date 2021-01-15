#ifndef BBBB_FUNCTIONS_H
#define BBBB_FUNCTIONS_H

#include <cmath>
#include <utility>
#include <iostream>
#include "analysis_utils.h"
#include "TMath.h"
#include "TSpline.h"

class TriggerEfficiencyCalculator
{
  public:
    TriggerEfficiencyCalculator(std::string triggerEfficiencyFileName)
    {
        TFile triggerEfficiencyFile(triggerEfficiencyFileName.c_str());
      
        fSplineDataEfficiency_Double90Quad30_QuadCentralJet30   = getSplineFromFile(triggerEfficiencyFile, "SingleMuon_Double90Quad30_Spline_QuadCentralJet30"  );
        fSplineDataEfficiency_Double90Quad30_DoubleCentralJet90 = getSplineFromFile(triggerEfficiencyFile, "SingleMuon_Double90Quad30_Spline_DoubleCentralJet90");
        fSplineDataEfficiency_Quad45_QuadCentralJet45           = getSplineFromFile(triggerEfficiencyFile, "SingleMuon_Quad45_Spline_QuadCentralJet45"          );
        fSplineDataEfficiency_And_QuadCentralJet45              = getSplineFromFile(triggerEfficiencyFile, "SingleMuon_And_Spline_QuadCentralJet45"             );

        fSplineMcEfficiency_Double90Quad30_QuadCentralJet30     = getSplineFromFile(triggerEfficiencyFile, "TTbar_Double90Quad30_Spline_QuadCentralJet30"       );
        fSplineMcEfficiency_Double90Quad30_DoubleCentralJet90   = getSplineFromFile(triggerEfficiencyFile, "TTbar_Double90Quad30_Spline_DoubleCentralJet90"     );
        fSplineMcEfficiency_Quad45_QuadCentralJet45             = getSplineFromFile(triggerEfficiencyFile, "TTbar_Quad45_Spline_QuadCentralJet45"               );
        fSplineMcEfficiency_And_QuadCentralJet45                = getSplineFromFile(triggerEfficiencyFile, "TTbar_And_Spline_QuadCentralJet45"                  );
    }

    float getDataEfficiency_Double90Quad30_QuadCentralJet30  (float fourthLeadingJet) {return getEfficiency(fSplineDataEfficiency_Double90Quad30_QuadCentralJet30   ,fourthLeadingJet);}
    float getDataEfficiency_Double90Quad30_DoubleCentralJet90(float secondLeadingJet) {return getEfficiency(fSplineDataEfficiency_Double90Quad30_DoubleCentralJet90 ,secondLeadingJet);}
    float getDataEfficiency_Quad45_QuadCentralJet45          (float fourthLeadingJet) {return getEfficiency(fSplineDataEfficiency_Quad45_QuadCentralJet45           ,fourthLeadingJet);}
    float getDataEfficiency_And_QuadCentralJet45             (float fourthLeadingJet) {return getEfficiency(fSplineDataEfficiency_And_QuadCentralJet45              ,fourthLeadingJet);}
    
    float getMcEfficiency_Double90Quad30_QuadCentralJet30    (float fourthLeadingJet) {return getEfficiency(fSplineMcEfficiency_Double90Quad30_QuadCentralJet30     ,fourthLeadingJet);}
    float getMcEfficiency_Double90Quad30_DoubleCentralJet90  (float secondLeadingJet) {return getEfficiency(fSplineMcEfficiency_Double90Quad30_DoubleCentralJet90   ,secondLeadingJet);}
    float getMcEfficiency_Quad45_QuadCentralJet45            (float fourthLeadingJet) {return getEfficiency(fSplineMcEfficiency_Quad45_QuadCentralJet45             ,fourthLeadingJet);}
    float getMcEfficiency_And_QuadCentralJet45               (float fourthLeadingJet) {return getEfficiency(fSplineMcEfficiency_And_QuadCentralJet45                ,fourthLeadingJet);}


  private:
    std::unique_ptr<TSpline> getSplineFromFile(TFile& inputFile, std::string splineName)
    {
        auto* spline = static_cast<TSpline*>(inputFile.Get(splineName.c_str()));
        if(spline == nullptr) throw std::runtime_error("Spline " + splineName + " does not exist in file " + inputFile.GetName());

        return std::unique_ptr<TSpline>(spline);
    }

    float getEfficiency(const std::unique_ptr<TSpline>& theSpline, float variableValue)
    {
        if     (variableValue > theSpline->GetXmax()) variableValue = theSpline->GetXmax();
        else if(variableValue < theSpline->GetXmin()) variableValue = theSpline->GetXmin();
        return theSpline->Eval(variableValue);
    }

    std::unique_ptr<TSpline> fSplineDataEfficiency_Double90Quad30_QuadCentralJet30  ;
    std::unique_ptr<TSpline> fSplineDataEfficiency_Double90Quad30_DoubleCentralJet90;
    std::unique_ptr<TSpline> fSplineDataEfficiency_Quad45_QuadCentralJet45          ;
    std::unique_ptr<TSpline> fSplineDataEfficiency_And_QuadCentralJet45             ;

    std::unique_ptr<TSpline> fSplineMcEfficiency_Double90Quad30_QuadCentralJet30    ;
    std::unique_ptr<TSpline> fSplineMcEfficiency_Double90Quad30_DoubleCentralJet90  ;
    std::unique_ptr<TSpline> fSplineMcEfficiency_Quad45_QuadCentralJet45            ;
    std::unique_ptr<TSpline> fSplineMcEfficiency_And_QuadCentralJet45               ;

};

std::vector<jet_t> bbbb_jets_idxs_BothClosestToDiagonal(const std::vector<jet_t> *presel_jets)
{

    //Do you add b-jet regression into the pairing?
    // bool breg = any_cast<bool>(parameterList_->at("BjetRegression"));

    bool breg = true;
    // float targetHiggsMass1 = any_cast<float>(parameterList_->at("LeadingHiggsMass"));
    // float targetHiggsMass2 = any_cast<float>(parameterList_->at("SubleadingHiggsMass"));
    float targetHiggsMass1 = 125;
    float targetHiggsMass2 = 120;

    std::vector<TLorentzVector> p4s;

    for (uint i = 0; i < presel_jets->size(); ++i)
        for (uint j = i+1; j < presel_jets->size(); ++j)
        {
            TLorentzVector p4sum;

            if (breg)
               p4sum= (presel_jets->at(i).p4_breg + presel_jets->at(j).p4_breg);
            else
               p4sum= (presel_jets->at(i).p4 + presel_jets->at(j).p4);
            p4s.emplace_back(p4sum);
        }

    float m1,m2,m3,m4,m5,m6;

    if(p4s.at(0).Pt() > p4s.at(5).Pt())
    {
      m1 = p4s.at(0).M(); m2 = p4s.at(5).M();
    }
    else
    {
      m1 = p4s.at(5).M(); m2 = p4s.at(0).M();        
    }

    if(p4s.at(1).Pt() > p4s.at(4).Pt())
    {
      m3 = p4s.at(1).M(); m4 = p4s.at(4).M();
    }
    else
    {
      m3 = p4s.at(4).M(); m4 = p4s.at(1).M();       
    }

    if(p4s.at(2).Pt() > p4s.at(3).Pt())
    {
      m5 = p4s.at(2).M(); m6 = p4s.at(3).M();
    }
    else
    {
      m5 = p4s.at(3).M(); m6 = p4s.at(2).M();       
    }

    std::pair<float, float> m_12_34 = std::make_pair(m1,m2);
    std::pair<float, float> m_13_24 = std::make_pair(m3,m4);
    std::pair<float, float> m_14_23 = std::make_pair(m5,m6);

    float d12_34 = TMath::Sqrt(pow(m_12_34.first,2) + pow(m_12_34.second,2) )*fabs( TMath::Sin( TMath::ATan(m_12_34.second/m_12_34.first) - TMath::ATan(targetHiggsMass2/targetHiggsMass1) ) );
    float d13_24 = TMath::Sqrt(pow(m_13_24.first,2) + pow(m_13_24.second,2) )*fabs( TMath::Sin( TMath::ATan(m_13_24.second/m_13_24.first) - TMath::ATan(targetHiggsMass2/targetHiggsMass1) ) );
    float d14_23 = TMath::Sqrt(pow(m_14_23.first,2) + pow(m_14_23.second,2) )*fabs( TMath::Sin( TMath::ATan(m_14_23.second/m_14_23.first) - TMath::ATan(targetHiggsMass2/targetHiggsMass1) ) );

    //NOTE: The formula below is equivalent (after some math-massaging =) )
    //float d12_34_b = fabs( m_12_34.first - ((targetHiggsMass1/targetHiggsMass1)*m_12_34.second) );
    //float d13_24_b = fabs( m_13_24.first - ((targetHiggsMass1/targetHiggsMass1)*m_13_24.second) );
    //float d14_23_b = fabs( m_14_23.first - ((targetHiggsMass1/targetHiggsMass1)*m_14_23.second) );
 
    float the_min = std::min({d12_34, d13_24, d14_23});
        
#include "analysis_utils.h"

    std::vector<jet_t> outputJets = *presel_jets;

    if (the_min == d12_34){
        outputJets.at(0) = presel_jets->at(1 - 1);
        outputJets.at(1) = presel_jets->at(2 - 1);
        outputJets.at(2) = presel_jets->at(3 - 1);
        outputJets.at(3) = presel_jets->at(4 - 1);
    }

    else if (the_min == d13_24){
        outputJets.at(0) = presel_jets->at(1 - 1);
        outputJets.at(1) = presel_jets->at(3 - 1);
        outputJets.at(2) = presel_jets->at(2 - 1);
        outputJets.at(3) = presel_jets->at(4 - 1);
    }

    else if (the_min == d14_23){
        outputJets.at(0) = presel_jets->at(1 - 1);
        outputJets.at(1) = presel_jets->at(4 - 1);
        outputJets.at(2) = presel_jets->at(2 - 1);
        outputJets.at(3) = presel_jets->at(3 - 1);
    }

    else
        std::cout << "** [WARNING] : bbbb_jets_idxs_BothClosestToDiagonal : something went wrong with finding the smallest Dhh" << std::endl;

    return outputJets;
}

#endif
