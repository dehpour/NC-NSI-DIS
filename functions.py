import pandas as pd
import numpy as np
from constants import *
from experiment_DUNE_CPOptimized import *

sigma_0p = 3.9e-32 * GF**2/np.pi * Mp*E/(1+Q2/MZ**2)**2 #m^2
sigma_0n = 3.9e-32 * GF**2/np.pi * Mn*E/(1+Q2/MZ**2)**2 #m^2

def NFD(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau):
  epsilonAu_emu    = epsilonAu_mue
  epsilonAu_etau   = epsilonAu_taue
  epsilonAu_mutau  = epsilonAu_taumu
  epsilonAd_emu    = epsilonAd_mue
  epsilonAd_etau   = epsilonAd_taue
  epsilonAd_mutau  = epsilonAd_taumu
  epsilonAs_emu    = epsilonAs_mue
  epsilonAs_etau   = epsilonAs_taue
  epsilonAs_mutau  = epsilonAs_taumu
  
  fAuee   = gAu + epsilonAu_ee
  fAdee   = gAd + epsilonAd_ee
  fAsee   = gAs + epsilonAs_ee
  fAuemu  = fAumue  = epsilonAu_emu
  fAdemu  = fAdmue  = epsilonAd_emu
  fAsemu  = fAsmue  = epsilonAs_emu
  fAuetau = fAutaue = epsilonAu_etau
  fAdetau = fAdtaue = epsilonAd_etau
  fAsetau = fAstaue = epsilonAs_etau
  
  fAumue   = fAuemu   = epsilonAu_mue
  fAdmue   = fAdemu   = epsilonAd_mue
  fAsmue   = fAsemu   = epsilonAs_mue
  fAumumu  = gAu + epsilonAu_mumu
  fAdmumu  = gAd + epsilonAd_mumu
  fAsmumu  = gAs + epsilonAs_mumu
  fAumutau = fAutaumu = epsilonAu_mutau
  fAdmutau = fAdtaumu = epsilonAd_mutau
  fAsmutau = fAstaumu = epsilonAs_mutau
  
  fAutaue   = fAuetau   = epsilonAu_taue
  fAdtaue   = fAdetau   = epsilonAd_taue
  fAstaue   = fAsetau   = epsilonAs_taue
  fAutaumu  = fAumutau  = epsilonAu_taumu
  fAdtaumu  = fAdmutau  = epsilonAd_taumu
  fAstaumu  = fAsmutau  = epsilonAs_taumu
  fAutautau = gAu + epsilonAu_tautau
  fAdtautau = gAd + epsilonAd_tautau
  fAstautau = gAs + epsilonAs_tautau
  
  fVuff = (fVuee   * A_e_FD   * np.conj(A_e_FD) + fVuemu   * A_e_FD   * np.conj(A_mu_FD) + fVuetau   * A_e_FD   * np.conj(A_tau_FD)
         + fVumue  * A_mu_FD  * np.conj(A_e_FD) + fVumumu  * A_mu_FD  * np.conj(A_mu_FD) + fVumutau  * A_mu_FD  * np.conj(A_tau_FD)
         + fVutaue * A_tau_FD * np.conj(A_e_FD) + fVutaumu * A_tau_FD * np.conj(A_mu_FD) + fVutautau * A_tau_FD * np.conj(A_tau_FD))
  
  fVdff = (fVdee   * A_e_FD   * np.conj(A_e_FD) + fVdemu   * A_e_FD   * np.conj(A_mu_FD) + fVdetau   * A_e_FD   * np.conj(A_tau_FD)
         + fVdmue  * A_mu_FD  * np.conj(A_e_FD) + fVdmumu  * A_mu_FD  * np.conj(A_mu_FD) + fVdmutau  * A_mu_FD  * np.conj(A_tau_FD)
         + fVdtaue * A_tau_FD * np.conj(A_e_FD) + fVdtaumu * A_tau_FD * np.conj(A_mu_FD) + fVdtautau * A_tau_FD * np.conj(A_tau_FD))
  
  fVsff = (fVsee   * A_e_FD   * np.conj(A_e_FD) + fVsemu   * A_e_FD   * np.conj(A_mu_FD) + fVsetau   * A_e_FD   * np.conj(A_tau_FD)
         + fVsmue  * A_mu_FD  * np.conj(A_e_FD) + fVsmumu  * A_mu_FD  * np.conj(A_mu_FD) + fVsmutau  * A_mu_FD  * np.conj(A_tau_FD)
         + fVstaue * A_tau_FD * np.conj(A_e_FD) + fVstaumu * A_tau_FD * np.conj(A_mu_FD) + fVstautau * A_tau_FD * np.conj(A_tau_FD))
  
  fVufp = (1/AFD) * (- fVuemu   * A_e_FD   * A_tau_FD + fVuetau   * A_e_FD   * A_mu_FD 
                     - fVumumu  * A_mu_FD  * A_tau_FD + fVumutau  * A_mu_FD  * A_mu_FD
                     - fVutaumu * A_tau_FD * A_tau_FD + fVutautau * A_tau_FD * A_mu_FD)
  
  fVdfp = (1/AFD) * (- fVdemu   * A_e_FD   * A_tau_FD + fVdetau   * A_e_FD   * A_mu_FD 
                     - fVdmumu  * A_mu_FD  * A_tau_FD + fVdmutau  * A_mu_FD  * A_mu_FD
                     - fVdtaumu * A_tau_FD * A_tau_FD + fVdtautau * A_tau_FD * A_mu_FD)
  
  fVsfp = (1/AFD) * (- fVsemu   * A_e_FD   * A_tau_FD + fVsetau   * A_e_FD   * A_mu_FD 
                     - fVsmumu  * A_mu_FD  * A_tau_FD + fVsmutau  * A_mu_FD  * A_mu_FD
                     - fVstaumu * A_tau_FD * A_tau_FD + fVstautau * A_tau_FD * A_mu_FD)
  
  fVuft = ((fVuee * A_e_FD + fVumue * A_mu_FD + fVutaue * A_tau_FD) * (np.conj(A_e_FD) * AFD / abs(A_e_FD))
          - ( fVuemu   * A_e_FD   * np.conj(A_mu_FD) + fVuetau   * A_e_FD   * np.conj(A_tau_FD) 
            + fVumumu  * A_mu_FD  * np.conj(A_mu_FD) + fVumutau  * A_mu_FD  * np.conj(A_tau_FD)
            + fVutaumu * A_tau_FD * np.conj(A_mu_FD) + fVutautau * A_tau_FD * np.conj(A_tau_FD)) * abs(A_e_FD)/AFD)
  
  fVdft = ((fVdee * A_e_FD + fVdmue * A_mu_FD + fVdtaue * A_tau_FD) * (np.conj(A_e_FD) * AFD / abs(A_e_FD))
          - ( fVdemu   * A_e_FD   * np.conj(A_mu_FD) + fVdetau   * A_e_FD   * np.conj(A_tau_FD) 
            + fVdmumu  * A_mu_FD  * np.conj(A_mu_FD) + fVdmutau  * A_mu_FD  * np.conj(A_tau_FD)
            + fVdtaumu * A_tau_FD * np.conj(A_mu_FD) + fVdtautau * A_tau_FD * np.conj(A_tau_FD)) * abs(A_e_FD)/AFD)
  
  fVsft = ((fVsee * A_e_FD + fVsmue * A_mu_FD + fVstaue * A_tau_FD) * (np.conj(A_e_FD) * AFD / abs(A_e_FD))
          - ( fVsemu   * A_e_FD   * np.conj(A_mu_FD) + fVsetau   * A_e_FD   * np.conj(A_tau_FD) 
            + fVsmumu  * A_mu_FD  * np.conj(A_mu_FD) + fVsmutau  * A_mu_FD  * np.conj(A_tau_FD)
            + fVstaumu * A_tau_FD * np.conj(A_mu_FD) + fVstautau * A_tau_FD * np.conj(A_tau_FD)) * abs(A_e_FD)/AFD)
  
  fAuff = (fAuee   * A_e_FD   * np.conj(A_e_FD) + fAuemu   * A_e_FD   * np.conj(A_mu_FD) + fAuetau   * A_e_FD   * np.conj(A_tau_FD)
         + fAumue  * A_mu_FD  * np.conj(A_e_FD) + fAumumu  * A_mu_FD  * np.conj(A_mu_FD) + fAumutau  * A_mu_FD  * np.conj(A_tau_FD)
         + fAutaue * A_tau_FD * np.conj(A_e_FD) + fAutaumu * A_tau_FD * np.conj(A_mu_FD) + fAutautau * A_tau_FD * np.conj(A_tau_FD))
  
  fAdff = (fAdee   * A_e_FD   * np.conj(A_e_FD) + fAdemu   * A_e_FD   * np.conj(A_mu_FD) + fAdetau   * A_e_FD   * np.conj(A_tau_FD)
         + fAdmue  * A_mu_FD  * np.conj(A_e_FD) + fAdmumu  * A_mu_FD  * np.conj(A_mu_FD) + fAdmutau  * A_mu_FD  * np.conj(A_tau_FD)
         + fAdtaue * A_tau_FD * np.conj(A_e_FD) + fAdtaumu * A_tau_FD * np.conj(A_mu_FD) + fAdtautau * A_tau_FD * np.conj(A_tau_FD))
  
  fAsff = (fAsee   * A_e_FD   * np.conj(A_e_FD) + fAsemu   * A_e_FD   * np.conj(A_mu_FD) + fAsetau   * A_e_FD   * np.conj(A_tau_FD)
         + fAsmue  * A_mu_FD  * np.conj(A_e_FD) + fAsmumu  * A_mu_FD  * np.conj(A_mu_FD) + fAsmutau  * A_mu_FD  * np.conj(A_tau_FD)
         + fAstaue * A_tau_FD * np.conj(A_e_FD) + fAstaumu * A_tau_FD * np.conj(A_mu_FD) + fAstautau * A_tau_FD * np.conj(A_tau_FD))
  
  fAufp = (1/AFD) * (- fAuemu   * A_e_FD   * A_tau_FD + fAuetau   * A_e_FD   * A_mu_FD
                     - fAumumu  * A_mu_FD  * A_tau_FD + fAumutau  * A_mu_FD  * A_mu_FD
                     - fAutaumu * A_tau_FD * A_tau_FD + fAutautau * A_tau_FD * A_mu_FD)
  
  fAdfp = (1/AFD) * (- fAdemu   * A_e_FD   * A_tau_FD + fAdetau   * A_e_FD   * A_mu_FD 
                     - fAdmumu  * A_mu_FD  * A_tau_FD + fAdmutau  * A_mu_FD  * A_mu_FD
                     - fAdtaumu * A_tau_FD * A_tau_FD + fAdtautau * A_tau_FD * A_mu_FD)
  
  fAsfp = (1/AFD) * (- fAsemu   * A_e_FD   * A_tau_FD + fAsetau   * A_e_FD   * A_mu_FD
                     - fAsmumu  * A_mu_FD  * A_tau_FD + fAsmutau  * A_mu_FD  * A_mu_FD
                     - fAstaumu * A_tau_FD * A_tau_FD + fAstautau * A_tau_FD * A_mu_FD)
  
  fAuft = ((fAuee * A_e_FD + fAumue * A_mu_FD + fAutaue * A_tau_FD) * (np.conj(A_e_FD) * AFD / abs(A_e_FD))
          - ( fAuemu   * A_e_FD   * np.conj(A_mu_FD) + fAuetau   * A_e_FD   * np.conj(A_tau_FD) 
            + fAumumu  * A_mu_FD  * np.conj(A_mu_FD) + fAumutau  * A_mu_FD  * np.conj(A_tau_FD)
            + fAutaumu * A_tau_FD * np.conj(A_mu_FD) + fAutautau * A_tau_FD * np.conj(A_tau_FD)) * abs(A_e_FD)/AFD)
  
  fAdft = ((fAdee * A_e_FD + fAdmue * A_mu_FD + fAdtaue * A_tau_FD) * (np.conj(A_e_FD) * AFD / abs(A_e_FD))
          - ( fAdemu   * A_e_FD   * np.conj(A_mu_FD) + fAdetau   * A_e_FD   * np.conj(A_tau_FD) 
            + fAdmumu  * A_mu_FD  * np.conj(A_mu_FD) + fAdmutau  * A_mu_FD  * np.conj(A_tau_FD)
            + fAdtaumu * A_tau_FD * np.conj(A_mu_FD) + fAdtautau * A_tau_FD * np.conj(A_tau_FD)) * abs(A_e_FD)/AFD)
  
  fAsft = ((fAsee * A_e_FD + fAsmue * A_mu_FD + fAstaue * A_tau_FD) * (np.conj(A_e_FD) * AFD / abs(A_e_FD))
          - ( fAsemu   * A_e_FD   * np.conj(A_mu_FD) + fAsetau   * A_e_FD   * np.conj(A_tau_FD) 
            + fAsmumu  * A_mu_FD  * np.conj(A_mu_FD) + fAsmutau  * A_mu_FD  * np.conj(A_tau_FD)
            + fAstaumu * A_tau_FD * np.conj(A_mu_FD) + fAstautau * A_tau_FD * np.conj(A_tau_FD)) * abs(A_e_FD)/AFD)
  
  sigma_p_ff = sigma_0p * ((  2/3 * x1uP - Mp/(2*E) * x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP ) * ( abs(fVuff)**2 + abs(fAuff)**2 )
                          + ( 2/3 * x1dP - Mp/(2*E) * x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP ) * ( abs(fVdff)**2 + abs(fAdff)**2 )
                          + ( 2/3 * x1sP - Mp/(2*E) * x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsff)**2 + abs(fAsff)**2 )
                          + ( 2/3 * x1uM - Mp/(2*E) * x2uM + (Mp/(2*E))**2 * x3uM ) * np.real(fVuff * np.conj(fAuff))
                          + ( 2/3 * x1dM - Mp/(2*E) * x2dM + (Mp/(2*E))**2 * x3dM ) * np.real(fVdff * np.conj(fAdff)) 
                          + ( 2/3 * x1sM - Mp/(2*E) * x2sM + (Mp/(2*E))**2 * x3sM ) * np.real(fVsff * np.conj(fAsff)))
  
  sigma_p_fp = sigma_0p * ((  2/3 * x1uP - Mp/(2*E) * x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP ) * ( abs(fVufp)**2 + abs(fAufp)**2 )
                          + ( 2/3 * x1dP - Mp/(2*E) * x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP ) * ( abs(fVdfp)**2 + abs(fAdfp)**2 )
                          + ( 2/3 * x1sP - Mp/(2*E) * x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsfp)**2 + abs(fAsfp)**2 )
                          + ( 2/3 * x1uM - Mp/(2*E) * x2uM + (Mp/(2*E))**2 * x3uM ) * np.real(fVufp * np.conj(fAufp))
                          + ( 2/3 * x1dM - Mp/(2*E) * x2dM + (Mp/(2*E))**2 * x3dM ) * np.real(fVdfp * np.conj(fAdfp))
                          + ( 2/3 * x1sM - Mp/(2*E) * x2sM + (Mp/(2*E))**2 * x3sM ) * np.real(fVsfp * np.conj(fAsfp)))
  
  sigma_p_ft = sigma_0p * ((  2/3 * x1uP - Mp/(2*E) * x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP ) * ( abs(fVuft)**2 + abs(fAuft)**2 )
                          + ( 2/3 * x1dP - Mp/(2*E) * x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP ) * ( abs(fVdft)**2 + abs(fAdft)**2 )
                          + ( 2/3 * x1sP - Mp/(2*E) * x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsft)**2 + abs(fAsft)**2 )
                          + ( 2/3 * x1uM - Mp/(2*E) * x2uM + (Mp/(2*E))**2 * x3uM ) * np.real(fVuft * np.conj(fAuft))
                          + ( 2/3 * x1dM - Mp/(2*E) * x2dM + (Mp/(2*E))**2 * x3dM ) * np.real(fVdft * np.conj(fAdft))
                          + ( 2/3 * x1sM - Mp/(2*E) * x2sM + (Mp/(2*E))**2 * x3sM ) * np.real(fVsft * np.conj(fAsft)))
  
  sigma_p_total = sigma_p_ff + sigma_p_fp + sigma_p_ft
  
  sigma_n_ff = sigma_0n * ((  2/3 * x1uP - Mn/(2*E) * x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP ) * ( abs(fVdff)**2 + abs(fAdff)**2 )
                          + ( 2/3 * x1dP - Mn/(2*E) * x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP ) * ( abs(fVuff)**2 + abs(fAuff)**2 )
                          + ( 2/3 * x1sP - Mn/(2*E) * x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsff)**2 + abs(fAsff)**2 )
                          + ( 2/3 * x1uM - Mn/(2*E) * x2uM + (Mn/(2*E))**2 * x3uM ) * np.real(fVdff * np.conj(fAdff))
                          + ( 2/3 * x1dM - Mn/(2*E) * x2dM + (Mn/(2*E))**2 * x3dM ) * np.real(fVuff * np.conj(fAuff))
                          + ( 2/3 * x1sM - Mn/(2*E) * x2sM + (Mn/(2*E))**2 * x3sM ) * np.real(fVsff * np.conj(fAsff)))
  
  sigma_n_fp = sigma_0n * ((  2/3 * x1uP - Mn/(2*E) * x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP ) * ( abs(fVdfp)**2 + abs(fAdfp)**2 )
                          + ( 2/3 * x1dP - Mn/(2*E) * x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP ) * ( abs(fVufp)**2 + abs(fAufp)**2 )
                          + ( 2/3 * x1sP - Mn/(2*E) * x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsfp)**2 + abs(fAsfp)**2 )
                          + ( 2/3 * x1uM - Mn/(2*E) * x2uM + (Mn/(2*E))**2 * x3uM ) * np.real(fVdfp * np.conj(fAdfp))
                          + ( 2/3 * x1dM - Mn/(2*E) * x2dM + (Mn/(2*E))**2 * x3dM ) * np.real(fVufp * np.conj(fAufp))
                          + ( 2/3 * x1sM - Mn/(2*E) * x2sM + (Mn/(2*E))**2 * x3sM ) * np.real(fVsfp * np.conj(fAsfp)))
  
  sigma_n_ft = sigma_0n * ((  2/3 * x1uP - Mn/(2*E) * x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP ) * ( abs(fVdft)**2 + abs(fAdft)**2 )
                          + ( 2/3 * x1dP - Mn/(2*E) * x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP ) * ( abs(fVuft)**2 + abs(fAuft)**2 )
                          + ( 2/3 * x1sP - Mn/(2*E) * x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsft)**2 + abs(fAsft)**2 )
                          + ( 2/3 * x1uM - Mn/(2*E) * x2uM + (Mn/(2*E))**2 * x3uM ) * np.real(fVdft * np.conj(fAdft))
                          + ( 2/3 * x1dM - Mn/(2*E) * x2dM + (Mn/(2*E))**2 * x3dM ) * np.real(fVuft * np.conj(fAuft))
                          + ( 2/3 * x1sM - Mn/(2*E) * x2sM + (Mn/(2*E))**2 * x3sM ) * np.real(fVsft * np.conj(fAsft)))
  
  sigma_n_total = sigma_n_ff + sigma_n_fp + sigma_n_ft
  
  Ndiff = (F_mu_FD * POT * t * NpFD * sigma_p_total + F_mu_FD * POT * t * NnFD * sigma_n_total)
  
  Nintegrated = 0
  for j in range (len(E)):
      Nintegrated = Nintegrated + Ndiff[j]*0.25
  return Nintegrated

def NbarFD(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau):
  epsilonAu_emu    = epsilonAu_mue
  epsilonAu_etau   = epsilonAu_taue
  epsilonAu_mutau  = epsilonAu_taumu
  epsilonAd_emu    = epsilonAd_mue
  epsilonAd_etau   = epsilonAd_taue
  epsilonAd_mutau  = epsilonAd_taumu
  epsilonAs_emu    = epsilonAs_mue
  epsilonAs_etau   = epsilonAs_taue
  epsilonAs_mutau  = epsilonAs_taumu
  
  fAuee   = gAu + epsilonAu_ee
  fAdee   = gAd + epsilonAd_ee
  fAsee   = gAs + epsilonAs_ee
  fAuemu  = fAumue  = epsilonAu_emu
  fAdemu  = fAdmue  = epsilonAd_emu
  fAsemu  = fAsmue  = epsilonAs_emu
  fAuetau = fAutaue = epsilonAu_etau
  fAdetau = fAdtaue = epsilonAd_etau
  fAsetau = fAstaue = epsilonAs_etau
  
  fAumue   = fAuemu   = epsilonAu_mue
  fAdmue   = fAdemu   = epsilonAd_mue
  fAsmue   = fAsemu   = epsilonAs_mue
  fAumumu  = gAu + epsilonAu_mumu
  fAdmumu  = gAd + epsilonAd_mumu
  fAsmumu  = gAs + epsilonAs_mumu
  fAumutau = fAutaumu = epsilonAu_mutau
  fAdmutau = fAdtaumu = epsilonAd_mutau
  fAsmutau = fAstaumu = epsilonAs_mutau
  
  fAutaue   = fAuetau   = epsilonAu_taue
  fAdtaue   = fAdetau   = epsilonAd_taue
  fAstaue   = fAsetau   = epsilonAs_taue
  fAutaumu  = fAumutau  = epsilonAu_taumu
  fAdtaumu  = fAdmutau  = epsilonAd_taumu
  fAstaumu  = fAsmutau  = epsilonAs_taumu
  fAutautau = gAu + epsilonAu_tautau
  fAdtautau = gAd + epsilonAd_tautau
  fAstautau = gAs + epsilonAs_tautau
  
  fVuff_bar = (fVuee   * A_ebar_FD   * np.conj(A_ebar_FD) + fVuemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fVuetau   * A_ebar_FD   * np.conj(A_taubar_FD)
             + fVumue  * A_mubar_FD  * np.conj(A_ebar_FD) + fVumumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fVumutau  * A_mubar_FD  * np.conj(A_taubar_FD)
             + fVutaue * A_taubar_FD * np.conj(A_ebar_FD) + fVutaumu * A_taubar_FD * np.conj(A_mubar_FD) + fVutautau * A_taubar_FD * np.conj(A_taubar_FD))
  
  fVdff_bar = (fVdee   * A_ebar_FD   * np.conj(A_ebar_FD) + fVdemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fVdetau   * A_ebar_FD   * np.conj(A_taubar_FD)
             + fVdmue  * A_mubar_FD  * np.conj(A_ebar_FD) + fVdmumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fVdmutau  * A_mubar_FD  * np.conj(A_taubar_FD)
             + fVdtaue * A_taubar_FD * np.conj(A_ebar_FD) + fVdtaumu * A_taubar_FD * np.conj(A_mubar_FD) + fVdtautau * A_taubar_FD * np.conj(A_taubar_FD))
  
  fVsff_bar = (fVsee   * A_ebar_FD   * np.conj(A_ebar_FD) + fVsemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fVsetau   * A_ebar_FD   * np.conj(A_taubar_FD)
             + fVsmue  * A_mubar_FD  * np.conj(A_ebar_FD) + fVsmumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fVsmutau  * A_mubar_FD  * np.conj(A_taubar_FD)
             + fVstaue * A_taubar_FD * np.conj(A_ebar_FD) + fVstaumu * A_taubar_FD * np.conj(A_mubar_FD) + fVstautau * A_taubar_FD * np.conj(A_taubar_FD))
  
  fVufp_bar = (1/AbarFD) * (- fVuemu   * A_ebar_FD   * A_taubar_FD + fVuetau   * A_ebar_FD   * A_mubar_FD 
                            - fVumumu  * A_mubar_FD  * A_taubar_FD + fVumutau  * A_mubar_FD  * A_mubar_FD
                            - fVutaumu * A_taubar_FD * A_taubar_FD + fVutautau * A_taubar_FD * A_mubar_FD)
  
  fVdfp_bar = (1/AbarFD) * (- fVdemu   * A_ebar_FD   * A_taubar_FD + fVdetau   * A_ebar_FD   * A_mubar_FD 
                            - fVdmumu  * A_mubar_FD  * A_taubar_FD + fVdmutau  * A_mubar_FD  * A_mubar_FD
                            - fVdtaumu * A_taubar_FD * A_taubar_FD + fVdtautau * A_taubar_FD * A_mubar_FD)
  
  fVsfp_bar = (1/AbarFD) * (- fVsemu   * A_ebar_FD   * A_taubar_FD + fVsetau   * A_ebar_FD   * A_mubar_FD 
                            - fVsmumu  * A_mubar_FD  * A_taubar_FD + fVsmutau  * A_mubar_FD  * A_mubar_FD
                            - fVstaumu * A_taubar_FD * A_taubar_FD + fVstautau * A_taubar_FD * A_mubar_FD)
  
  fVuft_bar = ((fVuee * A_ebar_FD + fVumue * A_mubar_FD + fVutaue * A_taubar_FD) * (np.conj(A_ebar_FD) * AbarFD / abs(A_ebar_FD))
                - ( fVuemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fVuetau   * A_ebar_FD   * np.conj(A_taubar_FD) 
                  + fVumumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fVumutau  * A_mubar_FD  * np.conj(A_taubar_FD)
                  + fVutaumu * A_taubar_FD * np.conj(A_mubar_FD) + fVutautau * A_taubar_FD * np.conj(A_taubar_FD)) * abs(A_ebar_FD)/AbarFD)
  
  fVdft_bar = ((fVdee * A_ebar_FD + fVdmue * A_mubar_FD + fVdtaue * A_taubar_FD) * (np.conj(A_ebar_FD) * AbarFD / abs(A_ebar_FD))
              - ( fVdemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fVdetau   * A_ebar_FD   * np.conj(A_taubar_FD) 
                + fVdmumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fVdmutau  * A_mubar_FD  * np.conj(A_taubar_FD)
                + fVdtaumu * A_taubar_FD * np.conj(A_mubar_FD) + fVdtautau * A_taubar_FD * np.conj(A_taubar_FD)) * abs(A_ebar_FD)/AbarFD)
  
  fVsft_bar = ((fVsee * A_ebar_FD + fVsmue * A_mubar_FD + fVstaue * A_taubar_FD) * (np.conj(A_ebar_FD) * AbarFD / abs(A_ebar_FD))
              - ( fVsemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fVsetau   * A_ebar_FD   * np.conj(A_taubar_FD) 
                + fVsmumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fVsmutau  * A_mubar_FD  * np.conj(A_taubar_FD)
                + fVstaumu * A_taubar_FD * np.conj(A_mubar_FD) + fVstautau * A_taubar_FD * np.conj(A_taubar_FD)) * abs(A_ebar_FD)/AbarFD)
  
  fAuff_bar = (fAuee   * A_ebar_FD   * np.conj(A_ebar_FD) + fAuemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fAuetau   * A_ebar_FD   * np.conj(A_taubar_FD)
             + fAumue  * A_mubar_FD  * np.conj(A_ebar_FD) + fAumumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fAumutau  * A_mubar_FD  * np.conj(A_taubar_FD)
             + fAutaue * A_taubar_FD * np.conj(A_ebar_FD) + fAutaumu * A_taubar_FD * np.conj(A_mubar_FD) + fAutautau * A_taubar_FD * np.conj(A_taubar_FD))
  
  fAdff_bar = (fAdee   * A_ebar_FD   * np.conj(A_ebar_FD) + fAdemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fAdetau   * A_ebar_FD   * np.conj(A_taubar_FD)
             + fAdmue  * A_mubar_FD  * np.conj(A_ebar_FD) + fAdmumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fAdmutau  * A_mubar_FD  * np.conj(A_taubar_FD)
             + fAdtaue * A_taubar_FD * np.conj(A_ebar_FD) + fAdtaumu * A_taubar_FD * np.conj(A_mubar_FD) + fAdtautau * A_taubar_FD * np.conj(A_taubar_FD))
  
  fAsff_bar = (fAsee   * A_ebar_FD   * np.conj(A_ebar_FD) + fAsemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fAsetau   * A_ebar_FD   * np.conj(A_taubar_FD)
             + fAsmue  * A_mubar_FD  * np.conj(A_ebar_FD) + fAsmumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fAsmutau  * A_mubar_FD  * np.conj(A_taubar_FD)
             + fAstaue * A_taubar_FD * np.conj(A_ebar_FD) + fAstaumu * A_taubar_FD * np.conj(A_mubar_FD) + fAstautau * A_taubar_FD * np.conj(A_taubar_FD))
  
  fAufp_bar = (1/AbarFD) * (- fAuemu   * A_ebar_FD   * A_taubar_FD + fAuetau   * A_ebar_FD   * A_mubar_FD 
                            - fAumumu  * A_mubar_FD  * A_taubar_FD + fAumutau  * A_mubar_FD  * A_mubar_FD
                            - fAutaumu * A_taubar_FD * A_taubar_FD + fAutautau * A_taubar_FD * A_mubar_FD)
  
  fAdfp_bar = (1/AbarFD) * (- fAdemu   * A_ebar_FD   * A_taubar_FD + fAdetau   * A_ebar_FD   * A_mubar_FD 
                            - fAdmumu  * A_mubar_FD  * A_taubar_FD + fAdmutau  * A_mubar_FD  * A_mubar_FD
                            - fAdtaumu * A_taubar_FD * A_taubar_FD + fAdtautau * A_taubar_FD * A_mubar_FD)
  
  fAsfp_bar = (1/AbarFD) * (- fAsemu   * A_ebar_FD   * A_taubar_FD + fAsetau   * A_ebar_FD   * A_mubar_FD 
                            - fAsmumu  * A_mubar_FD  * A_taubar_FD + fAsmutau  * A_mubar_FD  * A_mubar_FD
                            - fAstaumu * A_taubar_FD * A_taubar_FD + fAstautau * A_taubar_FD * A_mubar_FD)
  
  fAuft_bar = ((fAuee * A_ebar_FD + fAumue * A_mubar_FD + fAutaue * A_taubar_FD) * (np.conj(A_ebar_FD) * AbarFD / abs(A_ebar_FD))
                - ( fAuemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fAuetau   * A_ebar_FD   * np.conj(A_taubar_FD) 
                  + fAumumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fAumutau  * A_mubar_FD  * np.conj(A_taubar_FD)
                  + fAutaumu * A_taubar_FD * np.conj(A_mubar_FD) + fAutautau * A_taubar_FD * np.conj(A_taubar_FD)) * abs(A_ebar_FD)/AbarFD)
  
  fAdft_bar = ((fAdee * A_ebar_FD + fAdmue * A_mubar_FD + fAdtaue * A_taubar_FD) * (np.conj(A_ebar_FD) * AbarFD / abs(A_ebar_FD))
              - ( fAdemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fAdetau   * A_ebar_FD   * np.conj(A_taubar_FD) 
                + fAdmumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fAdmutau  * A_mubar_FD  * np.conj(A_taubar_FD)
                + fAdtaumu * A_taubar_FD * np.conj(A_mubar_FD) + fAdtautau * A_taubar_FD * np.conj(A_taubar_FD)) * abs(A_ebar_FD)/AbarFD)
  
  fAsft_bar = ((fAsee * A_ebar_FD + fAsmue * A_mubar_FD + fAstaue * A_taubar_FD) * (np.conj(A_ebar_FD) * AbarFD / abs(A_ebar_FD))
              - ( fAsemu   * A_ebar_FD   * np.conj(A_mubar_FD) + fAsetau   * A_ebar_FD   * np.conj(A_taubar_FD) 
                + fAsmumu  * A_mubar_FD  * np.conj(A_mubar_FD) + fAsmutau  * A_mubar_FD  * np.conj(A_taubar_FD)
                + fAstaumu * A_taubar_FD * np.conj(A_mubar_FD) + fAstautau * A_taubar_FD * np.conj(A_taubar_FD)) * abs(A_ebar_FD)/AbarFD)
  
  sigmabar_p_ff = sigma_0p * (( 2/3 * x1uP  - Mp/(2*E) *  x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP  ) * ( abs(fVuff_bar)**2 + abs(fAuff_bar)**2 )
                            + ( 2/3 * x1dP  - Mp/(2*E) *  x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP   ) * ( abs(fVdff_bar)**2 + abs(fAdff_bar)**2 )
                            + ( 2/3 * x1sP - Mp/(2*E) *  x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsff_bar)**2 + abs(fAsff_bar)**2 )
                            - ( 2/3 * x1uM   - Mp/(2*E) *  x2uM + (Mp/(2*E))**2 *  x3uM  ) * np.real(fVuff_bar * np.conj(fAuff_bar))
                            - ( 2/3 * x1dM  - Mp/(2*E) *  x2dM  + (Mp/(2*E))**2 *  x3dM ) * np.real(fVdff_bar * np.conj(fAdff_bar)) 
                            - ( 2/3 * x1sM - Mp/(2*E) * x2sM   + (Mp/(2*E))**2 * x3sM     ) * np.real(fVsff_bar * np.conj(fAsff_bar)))
  
  sigmabar_p_fp = sigma_0p * (( 2/3 * x1uP  - Mp/(2*E) *  x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP  ) * ( abs(fVufp_bar)**2 + abs(fAufp_bar)**2 )
                            + ( 2/3 * x1dP  - Mp/(2*E) *  x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP   ) * ( abs(fVdfp_bar)**2 + abs(fAdfp_bar)**2 )
                            + ( 2/3 * x1sP - Mp/(2*E) *  x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsfp_bar)**2 + abs(fAsfp_bar)**2 )
                            - ( 2/3 * x1uM   - Mp/(2*E) *  x2uM + (Mp/(2*E))**2 *  x3uM  ) * np.real(fVufp_bar * np.conj(fAufp_bar))
                            - ( 2/3 * x1dM  - Mp/(2*E) *  x2dM  + (Mp/(2*E))**2 *  x3dM ) * np.real(fVdfp_bar * np.conj(fAdfp_bar))
                            - ( 2/3 * x1sM - Mp/(2*E) * x2sM   + (Mp/(2*E))**2 * x3sM     ) * np.real(fVsfp_bar * np.conj(fAsfp_bar)))
  
  sigmabar_p_ft = sigma_0p * (( 2/3 * x1uP  - Mp/(2*E) *  x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP  ) * ( abs(fVuft_bar)**2 + abs(fAuft_bar)**2 )
                            + ( 2/3 * x1dP  - Mp/(2*E) *  x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP   ) * ( abs(fVdft_bar)**2 + abs(fAdft_bar)**2 )
                            + ( 2/3 * x1sP - Mp/(2*E) *  x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsft_bar)**2 + abs(fAsft_bar)**2 )
                            - ( 2/3 * x1uM   - Mp/(2*E) *  x2uM + (Mp/(2*E))**2 *  x3uM  ) * np.real(fVuft_bar * np.conj(fAuft_bar))
                            - ( 2/3 * x1dM  - Mp/(2*E) *  x2dM  + (Mp/(2*E))**2 *  x3dM ) * np.real(fVdft_bar * np.conj(fAdft_bar))
                            - ( 2/3 * x1sM - Mp/(2*E) * x2sM   + (Mp/(2*E))**2 * x3sM     ) * np.real(fVsft_bar * np.conj(fAsft_bar)))
  
  sigmabar_p_total = sigmabar_p_ff + sigmabar_p_fp + sigmabar_p_ft
  
  sigmabar_n_ff = sigma_0n * (( 2/3 * x1uP  - Mn/(2*E) *  x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP  ) * ( abs(fVdff_bar)**2 + abs(fAdff_bar)**2 )
                            + ( 2/3 * x1dP  - Mn/(2*E) *  x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP   ) * ( abs(fVuff_bar)**2 + abs(fAuff_bar)**2 )
                            + ( 2/3 * x1sP - Mn/(2*E) *  x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsff_bar)**2 + abs(fAsff_bar)**2 )
                            - ( 2/3 * x1uM   - Mn/(2*E) *  x2uM + (Mn/(2*E))**2 *  x3uM  ) * np.real(fVdff_bar * np.conj(fAdff_bar))
                            - ( 2/3 * x1dM  - Mn/(2*E) *  x2dM  + (Mn/(2*E))**2 *  x3dM ) * np.real(fVuff_bar * np.conj(fAuff_bar))
                            - ( 2/3 * x1sM - Mn/(2*E) * x2sM   + (Mn/(2*E))**2 * x3sM     ) * np.real(fVsff_bar * np.conj(fAsff_bar)))
  
  sigmabar_n_fp = sigma_0n * (( 2/3 * x1uP  - Mn/(2*E) *  x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP  ) * ( abs(fVdfp_bar)**2 + abs(fAdfp_bar)**2 )
                            + ( 2/3 * x1dP  - Mn/(2*E) *  x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP   ) * ( abs(fVufp_bar)**2 + abs(fAufp_bar)**2 )
                            + ( 2/3 * x1sP - Mn/(2*E) *  x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsfp_bar)**2 + abs(fAsfp_bar)**2 )
                            - ( 2/3 * x1uM   - Mn/(2*E) *  x2uM + (Mn/(2*E))**2 *  x3uM  ) * np.real(fVdfp_bar * np.conj(fAdfp_bar))
                            - ( 2/3 * x1dM  - Mn/(2*E) *  x2dM  + (Mn/(2*E))**2 *  x3dM ) * np.real(fVufp_bar * np.conj(fAufp_bar))
                            - ( 2/3 * x1sM - Mn/(2*E) * x2sM   + (Mn/(2*E))**2 * x3sM     ) * np.real(fVsfp_bar * np.conj(fAsfp_bar)))
  
  sigmabar_n_ft = sigma_0n * (( 2/3 * x1uP  - Mn/(2*E) *  x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP  ) * ( abs(fVdft_bar)**2 + abs(fAdft_bar)**2 )
                            + ( 2/3 * x1dP  - Mn/(2*E) *  x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP   ) * ( abs(fVuft_bar)**2 + abs(fAuft_bar)**2 )
                            + ( 2/3 * x1sP - Mn/(2*E) *  x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsft_bar)**2 + abs(fAsft_bar)**2 )
                            - ( 2/3 * x1uM   - Mn/(2*E) *  x2uM + (Mn/(2*E))**2 *  x3uM  ) * np.real(fVdft_bar * np.conj(fAdft_bar))
                            - ( 2/3 * x1dM  - Mn/(2*E) *  x2dM  + (Mn/(2*E))**2 *  x3dM ) * np.real(fVuft_bar * np.conj(fAuft_bar))
                            - ( 2/3 * x1sM - Mn/(2*E) * x2sM   + (Mn/(2*E))**2 * x3sM     ) * np.real(fVsft_bar * np.conj(fAsft_bar)))
  
  sigmabar_n_total = sigmabar_n_ff + sigmabar_n_fp + sigmabar_n_ft
  
  Ndiff = (F_mubar_FD * POT * tbar * NpFD * sigmabar_p_total + F_mubar_FD * POT * tbar * NnFD * sigmabar_n_total)
  
  Nintegrated = 0
  for j in range (len(E)):
      Nintegrated = Nintegrated + Ndiff[j]*0.25
  return Nintegrated

def NPNbarFD(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau):
  return NFD(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau) + NbarFD(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau)

def NMNbarFD(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau):
  return NFD(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau) - NbarFD(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau)

def NND(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau):
  epsilonAu_emu    = epsilonAu_mue
  epsilonAu_etau   = epsilonAu_taue
  epsilonAu_mutau  = epsilonAu_taumu
  epsilonAd_emu    = epsilonAd_mue
  epsilonAd_etau   = epsilonAd_taue
  epsilonAd_mutau  = epsilonAd_taumu
  epsilonAs_emu    = epsilonAs_mue
  epsilonAs_etau   = epsilonAs_taue
  epsilonAs_mutau  = epsilonAs_taumu
  
  fAuee   = gAu + epsilonAu_ee
  fAdee   = gAd + epsilonAd_ee
  fAsee   = gAs + epsilonAs_ee
  fAuemu  = fAumue  = epsilonAu_emu
  fAdemu  = fAdmue  = epsilonAd_emu
  fAsemu  = fAsmue  = epsilonAs_emu
  fAuetau = fAutaue = epsilonAu_etau
  fAdetau = fAdtaue = epsilonAd_etau
  fAsetau = fAstaue = epsilonAs_etau
  
  fAumue   = fAuemu   = epsilonAu_mue
  fAdmue   = fAdemu   = epsilonAd_mue
  fAsmue   = fAsemu   = epsilonAs_mue
  fAumumu  = gAu + epsilonAu_mumu
  fAdmumu  = gAd + epsilonAd_mumu
  fAsmumu  = gAs + epsilonAs_mumu
  fAumutau = fAutaumu = epsilonAu_mutau
  fAdmutau = fAdtaumu = epsilonAd_mutau
  fAsmutau = fAstaumu = epsilonAs_mutau
  
  fAutaue   = fAuetau   = epsilonAu_taue
  fAdtaue   = fAdetau   = epsilonAd_taue
  fAstaue   = fAsetau   = epsilonAs_taue
  fAutaumu  = fAumutau  = epsilonAu_taumu
  fAdtaumu  = fAdmutau  = epsilonAd_taumu
  fAstaumu  = fAsmutau  = epsilonAs_taumu
  fAutautau = gAu + epsilonAu_tautau
  fAdtautau = gAd + epsilonAd_tautau
  fAstautau = gAs + epsilonAs_tautau
  
  fVuff = (fVuee   * A_e_ND   * np.conj(A_e_ND) + fVuemu   * A_e_ND   * np.conj(A_mu_ND) + fVuetau   * A_e_ND   * np.conj(A_tau_ND)
         + fVumue  * A_mu_ND  * np.conj(A_e_ND) + fVumumu  * A_mu_ND  * np.conj(A_mu_ND) + fVumutau  * A_mu_ND  * np.conj(A_tau_ND)
         + fVutaue * A_tau_ND * np.conj(A_e_ND) + fVutaumu * A_tau_ND * np.conj(A_mu_ND) + fVutautau * A_tau_ND * np.conj(A_tau_ND))
  
  fVdff = (fVdee   * A_e_ND   * np.conj(A_e_ND) + fVdemu   * A_e_ND   * np.conj(A_mu_ND) + fVdetau   * A_e_ND   * np.conj(A_tau_ND)
         + fVdmue  * A_mu_ND  * np.conj(A_e_ND) + fVdmumu  * A_mu_ND  * np.conj(A_mu_ND) + fVdmutau  * A_mu_ND  * np.conj(A_tau_ND)
         + fVdtaue * A_tau_ND * np.conj(A_e_ND) + fVdtaumu * A_tau_ND * np.conj(A_mu_ND) + fVdtautau * A_tau_ND * np.conj(A_tau_ND))
  
  fVsff = (fVsee   * A_e_ND   * np.conj(A_e_ND) + fVsemu   * A_e_ND   * np.conj(A_mu_ND) + fVsetau   * A_e_ND   * np.conj(A_tau_ND)
         + fVsmue  * A_mu_ND  * np.conj(A_e_ND) + fVsmumu  * A_mu_ND  * np.conj(A_mu_ND) + fVsmutau  * A_mu_ND  * np.conj(A_tau_ND)
         + fVstaue * A_tau_ND * np.conj(A_e_ND) + fVstaumu * A_tau_ND * np.conj(A_mu_ND) + fVstautau * A_tau_ND * np.conj(A_tau_ND))
  
  fVufp = (1/AND) * (- fVuemu   * A_e_ND   * A_tau_ND + fVuetau   * A_e_ND   * A_mu_ND 
                     - fVumumu  * A_mu_ND  * A_tau_ND + fVumutau  * A_mu_ND  * A_mu_ND
                     - fVutaumu * A_tau_ND * A_tau_ND + fVutautau * A_tau_ND * A_mu_ND)
  
  fVdfp = (1/AND) * (- fVdemu   * A_e_ND   * A_tau_ND + fVdetau   * A_e_ND   * A_mu_ND 
                     - fVdmumu  * A_mu_ND  * A_tau_ND + fVdmutau  * A_mu_ND  * A_mu_ND
                     - fVdtaumu * A_tau_ND * A_tau_ND + fVdtautau * A_tau_ND * A_mu_ND)
  
  fVsfp = (1/AND) * (- fVsemu   * A_e_ND   * A_tau_ND + fVsetau   * A_e_ND   * A_mu_ND 
                     - fVsmumu  * A_mu_ND  * A_tau_ND + fVsmutau  * A_mu_ND  * A_mu_ND
                     - fVstaumu * A_tau_ND * A_tau_ND + fVstautau * A_tau_ND * A_mu_ND)
  
  fVuft = ((fVuee * A_e_ND + fVumue * A_mu_ND + fVutaue * A_tau_ND) * (np.conj(A_e_ND) * AND / abs(A_e_ND))
          - ( fVuemu   * A_e_ND   * np.conj(A_mu_ND) + fVuetau   * A_e_ND   * np.conj(A_tau_ND) 
            + fVumumu  * A_mu_ND  * np.conj(A_mu_ND) + fVumutau  * A_mu_ND  * np.conj(A_tau_ND)
            + fVutaumu * A_tau_ND * np.conj(A_mu_ND) + fVutautau * A_tau_ND * np.conj(A_tau_ND)) * abs(A_e_ND)/AND)
  
  fVdft = ((fVdee * A_e_ND + fVdmue * A_mu_ND + fVdtaue * A_tau_ND) * (np.conj(A_e_ND) * AND / abs(A_e_ND))
          - ( fVdemu   * A_e_ND   * np.conj(A_mu_ND) + fVdetau   * A_e_ND   * np.conj(A_tau_ND) 
            + fVdmumu  * A_mu_ND  * np.conj(A_mu_ND) + fVdmutau  * A_mu_ND  * np.conj(A_tau_ND)
            + fVdtaumu * A_tau_ND * np.conj(A_mu_ND) + fVdtautau * A_tau_ND * np.conj(A_tau_ND)) * abs(A_e_ND)/AND)
  
  fVsft = ((fVsee * A_e_ND + fVsmue * A_mu_ND + fVstaue * A_tau_ND) * (np.conj(A_e_ND) * AND / abs(A_e_ND))
        - ( fVsemu   * A_e_ND   * np.conj(A_mu_ND) + fVsetau   * A_e_ND   * np.conj(A_tau_ND) 
          + fVsmumu  * A_mu_ND  * np.conj(A_mu_ND) + fVsmutau  * A_mu_ND  * np.conj(A_tau_ND)
          + fVstaumu * A_tau_ND * np.conj(A_mu_ND) + fVstautau * A_tau_ND * np.conj(A_tau_ND)) * abs(A_e_ND)/AND)

  fAuff = (fAuee   * A_e_ND   * np.conj(A_e_ND) + fAuemu   * A_e_ND   * np.conj(A_mu_ND) + fAuetau   * A_e_ND   * np.conj(A_tau_ND)
         + fAumue  * A_mu_ND  * np.conj(A_e_ND) + fAumumu  * A_mu_ND  * np.conj(A_mu_ND) + fAumutau  * A_mu_ND  * np.conj(A_tau_ND)
         + fAutaue * A_tau_ND * np.conj(A_e_ND) + fAutaumu * A_tau_ND * np.conj(A_mu_ND) + fAutautau * A_tau_ND * np.conj(A_tau_ND))

  fAdff = (fAdee   * A_e_ND   * np.conj(A_e_ND) + fAdemu   * A_e_ND   * np.conj(A_mu_ND) + fAdetau   * A_e_ND   * np.conj(A_tau_ND)
         + fAdmue  * A_mu_ND  * np.conj(A_e_ND) + fAdmumu  * A_mu_ND  * np.conj(A_mu_ND) + fAdmutau  * A_mu_ND  * np.conj(A_tau_ND)
         + fAdtaue * A_tau_ND * np.conj(A_e_ND) + fAdtaumu * A_tau_ND * np.conj(A_mu_ND) + fAdtautau * A_tau_ND * np.conj(A_tau_ND))

  fAsff = (fAsee   * A_e_ND   * np.conj(A_e_ND) + fAsemu   * A_e_ND   * np.conj(A_mu_ND) + fAsetau   * A_e_ND   * np.conj(A_tau_ND)
         + fAsmue  * A_mu_ND  * np.conj(A_e_ND) + fAsmumu  * A_mu_ND  * np.conj(A_mu_ND) + fAsmutau  * A_mu_ND  * np.conj(A_tau_ND)
         + fAstaue * A_tau_ND * np.conj(A_e_ND) + fAstaumu * A_tau_ND * np.conj(A_mu_ND) + fAstautau * A_tau_ND * np.conj(A_tau_ND))

  fAufp = (1/AND) * (- fAuemu   * A_e_ND   * A_tau_ND + fAuetau   * A_e_ND   * A_mu_ND
                     - fAumumu  * A_mu_ND  * A_tau_ND + fAumutau  * A_mu_ND  * A_mu_ND
                     - fAutaumu * A_tau_ND * A_tau_ND + fAutautau * A_tau_ND * A_mu_ND)

  fAdfp = (1/AND) * (- fAdemu   * A_e_ND   * A_tau_ND + fAdetau   * A_e_ND   * A_mu_ND 
                     - fAdmumu  * A_mu_ND  * A_tau_ND + fAdmutau  * A_mu_ND  * A_mu_ND
                     - fAdtaumu * A_tau_ND * A_tau_ND + fAdtautau * A_tau_ND * A_mu_ND)

  fAsfp = (1/AND) * (- fAsemu   * A_e_ND   * A_tau_ND + fAsetau   * A_e_ND   * A_mu_ND
                     - fAsmumu  * A_mu_ND  * A_tau_ND + fAsmutau  * A_mu_ND  * A_mu_ND
                     - fAstaumu * A_tau_ND * A_tau_ND + fAstautau * A_tau_ND * A_mu_ND)

  fAuft = ((fAuee * A_e_ND + fAumue * A_mu_ND + fAutaue * A_tau_ND) * (np.conj(A_e_ND) * AND / abs(A_e_ND))
          - ( fAuemu   * A_e_ND   * np.conj(A_mu_ND) + fAuetau   * A_e_ND   * np.conj(A_tau_ND) 
            + fAumumu  * A_mu_ND  * np.conj(A_mu_ND) + fAumutau  * A_mu_ND  * np.conj(A_tau_ND)
            + fAutaumu * A_tau_ND * np.conj(A_mu_ND) + fAutautau * A_tau_ND * np.conj(A_tau_ND)) * abs(A_e_ND)/AND)

  fAdft = ((fAdee * A_e_ND + fAdmue * A_mu_ND + fAdtaue * A_tau_ND) * (np.conj(A_e_ND) * AND / abs(A_e_ND))
          - ( fAdemu   * A_e_ND   * np.conj(A_mu_ND) + fAdetau   * A_e_ND   * np.conj(A_tau_ND) 
            + fAdmumu  * A_mu_ND  * np.conj(A_mu_ND) + fAdmutau  * A_mu_ND  * np.conj(A_tau_ND)
            + fAdtaumu * A_tau_ND * np.conj(A_mu_ND) + fAdtautau * A_tau_ND * np.conj(A_tau_ND)) * abs(A_e_ND)/AND)

  fAsft = ((fAsee * A_e_ND + fAsmue * A_mu_ND + fAstaue * A_tau_ND) * (np.conj(A_e_ND) * AND / abs(A_e_ND))
          - ( fAsemu   * A_e_ND   * np.conj(A_mu_ND) + fAsetau   * A_e_ND   * np.conj(A_tau_ND) 
            + fAsmumu  * A_mu_ND  * np.conj(A_mu_ND) + fAsmutau  * A_mu_ND  * np.conj(A_tau_ND)
            + fAstaumu * A_tau_ND * np.conj(A_mu_ND) + fAstautau * A_tau_ND * np.conj(A_tau_ND)) * abs(A_e_ND)/AND)
  
  sigma_p_ff = sigma_0p * ((  2/3 * x1uP  - Mp/(2*E) *  x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP  ) * ( abs(fVuff)**2 + abs(fAuff)**2 )
                          + ( 2/3 * x1dP  - Mp/(2*E) *  x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP   ) * ( abs(fVdff)**2 + abs(fAdff)**2 )
                          + ( 2/3 * x1sP - Mp/(2*E) *  x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsff)**2 + abs(fAsff)**2 )
                          + ( 2/3 * x1uM   - Mp/(2*E) *  x2uM + (Mp/(2*E))**2 *  x3uM  ) * np.real(fVuff * np.conj(fAuff))
                          + ( 2/3 * x1dM  - Mp/(2*E) *  x2dM  + (Mp/(2*E))**2 *  x3dM ) * np.real(fVdff * np.conj(fAdff)) 
                          + ( 2/3 * x1sM - Mp/(2*E) * x2sM   + (Mp/(2*E))**2 * x3sM     ) * np.real(fVsff * np.conj(fAsff)))
  
  sigma_p_fp = sigma_0p * ((  2/3 * x1uP  - Mp/(2*E) *  x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP  ) * ( abs(fVufp)**2 + abs(fAufp)**2 )
                          + ( 2/3 * x1dP  - Mp/(2*E) *  x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP   ) * ( abs(fVdfp)**2 + abs(fAdfp)**2 )
                          + ( 2/3 * x1sP - Mp/(2*E) *  x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsfp)**2 + abs(fAsfp)**2 )
                          + ( 2/3 * x1uM   - Mp/(2*E) *  x2uM + (Mp/(2*E))**2 *  x3uM  ) * np.real(fVufp * np.conj(fAufp))
                          + ( 2/3 * x1dM  - Mp/(2*E) *  x2dM  + (Mp/(2*E))**2 *  x3dM ) * np.real(fVdfp * np.conj(fAdfp))
                          + ( 2/3 * x1sM - Mp/(2*E) * x2sM   + (Mp/(2*E))**2 * x3sM     ) * np.real(fVsfp * np.conj(fAsfp)))
  
  sigma_p_ft = sigma_0p * ((  2/3 * x1uP  - Mp/(2*E) *  x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP  ) * ( abs(fVuft)**2 + abs(fAuft)**2 )
                          + ( 2/3 * x1dP  - Mp/(2*E) *  x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP   ) * ( abs(fVdft)**2 + abs(fAdft)**2 )
                          + ( 2/3 * x1sP - Mp/(2*E) *  x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsft)**2 + abs(fAsft)**2 )
                          + ( 2/3 * x1uM   - Mp/(2*E) *  x2uM + (Mp/(2*E))**2 *  x3uM  ) * np.real(fVuft * np.conj(fAuft))
                          + ( 2/3 * x1dM  - Mp/(2*E) *  x2dM  + (Mp/(2*E))**2 *  x3dM ) * np.real(fVdft * np.conj(fAdft))
                          + ( 2/3 * x1sM - Mp/(2*E) * x2sM   + (Mp/(2*E))**2 * x3sM     ) * np.real(fVsft * np.conj(fAsft)))
  
  sigma_p_total = sigma_p_ff + sigma_p_fp + sigma_p_ft
  
  sigma_n_ff = sigma_0n * ((  2/3 * x1uP  - Mn/(2*E) *  x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP  ) * ( abs(fVdff)**2 + abs(fAdff)**2 )
                          + ( 2/3 * x1dP  - Mn/(2*E) *  x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP   ) * ( abs(fVuff)**2 + abs(fAuff)**2 )
                          + ( 2/3 * x1sP - Mn/(2*E) *  x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsff)**2 + abs(fAsff)**2 )
                          + ( 2/3 * x1uM   - Mn/(2*E) *  x2uM + (Mn/(2*E))**2 *  x3uM  ) * np.real(fVdff * np.conj(fAdff))
                          + ( 2/3 * x1dM  - Mn/(2*E) *  x2dM  + (Mn/(2*E))**2 *  x3dM ) * np.real(fVuff * np.conj(fAuff))
                          + ( 2/3 * x1sM - Mn/(2*E) * x2sM   + (Mn/(2*E))**2 * x3sM     ) * np.real(fVsff * np.conj(fAsff)))
  
  sigma_n_fp = sigma_0n * ((  2/3 * x1uP  - Mn/(2*E) *  x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP  ) * ( abs(fVdfp)**2 + abs(fAdfp)**2 )
                          + ( 2/3 * x1dP  - Mn/(2*E) *  x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP   ) * ( abs(fVufp)**2 + abs(fAufp)**2 )
                          + ( 2/3 * x1sP - Mn/(2*E) *  x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsfp)**2 + abs(fAsfp)**2 )
                          + ( 2/3 * x1uM   - Mn/(2*E) *  x2uM + (Mn/(2*E))**2 *  x3uM  ) * np.real(fVdfp * np.conj(fAdfp))
                          + ( 2/3 * x1dM  - Mn/(2*E) *  x2dM  + (Mn/(2*E))**2 *  x3dM ) * np.real(fVufp * np.conj(fAufp))
                          + ( 2/3 * x1sM - Mn/(2*E) * x2sM   + (Mn/(2*E))**2 * x3sM     ) * np.real(fVsfp * np.conj(fAsfp)))
  
  sigma_n_ft = sigma_0n * ((  2/3 * x1uP  - Mn/(2*E) *  x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP  ) * ( abs(fVdft)**2 + abs(fAdft)**2 )
                          + ( 2/3 * x1dP  - Mn/(2*E) *  x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP   ) * ( abs(fVuft)**2 + abs(fAuft)**2 )
                          + ( 2/3 * x1sP - Mn/(2*E) *  x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsft)**2 + abs(fAsft)**2 )
                          + ( 2/3 * x1uM   - Mn/(2*E) *  x2uM + (Mn/(2*E))**2 *  x3uM  ) * np.real(fVdft * np.conj(fAdft))
                          + ( 2/3 * x1dM  - Mn/(2*E) *  x2dM  + (Mn/(2*E))**2 *  x3dM ) * np.real(fVuft * np.conj(fAuft))
                          + ( 2/3 * x1sM - Mn/(2*E) * x2sM   + (Mn/(2*E))**2 * x3sM     ) * np.real(fVsft * np.conj(fAsft)))
  
  sigma_n_total = sigma_n_ff + sigma_n_fp + sigma_n_ft
  
  Ndiff = (F_mu_ND * POT * t * NpND * sigma_p_total + F_mu_ND * POT * t * NnND * sigma_n_total)
  
  Nintegrated = 0
  for j in range (len(E)):
      Nintegrated = Nintegrated + Ndiff[j]*0.25
  return Nintegrated

def NbarND(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau):
  epsilonAu_emu    = epsilonAu_mue
  epsilonAu_etau   = epsilonAu_taue
  epsilonAu_mutau  = epsilonAu_taumu
  epsilonAd_emu    = epsilonAd_mue
  epsilonAd_etau   = epsilonAd_taue
  epsilonAd_mutau  = epsilonAd_taumu
  epsilonAs_emu    = epsilonAs_mue
  epsilonAs_etau   = epsilonAs_taue
  epsilonAs_mutau  = epsilonAs_taumu
  
  fAuee   = gAu + epsilonAu_ee
  fAdee   = gAd + epsilonAd_ee
  fAsee   = gAs + epsilonAs_ee
  fAuemu  = fAumue  = epsilonAu_emu
  fAdemu  = fAdmue  = epsilonAd_emu
  fAsemu  = fAsmue  = epsilonAs_emu
  fAuetau = fAutaue = epsilonAu_etau
  fAdetau = fAdtaue = epsilonAd_etau
  fAsetau = fAstaue = epsilonAs_etau
  
  fAumue   = fAuemu   = epsilonAu_mue
  fAdmue   = fAdemu   = epsilonAd_mue
  fAsmue   = fAsemu   = epsilonAs_mue
  fAumumu  = gAu + epsilonAu_mumu
  fAdmumu  = gAd + epsilonAd_mumu
  fAsmumu  = gAs + epsilonAs_mumu
  fAumutau = fAutaumu = epsilonAu_mutau
  fAdmutau = fAdtaumu = epsilonAd_mutau
  fAsmutau = fAstaumu = epsilonAs_mutau
  
  fAutaue   = fAuetau   = epsilonAu_taue
  fAdtaue   = fAdetau   = epsilonAd_taue
  fAstaue   = fAsetau   = epsilonAs_taue
  fAutaumu  = fAumutau  = epsilonAu_taumu
  fAdtaumu  = fAdmutau  = epsilonAd_taumu
  fAstaumu  = fAsmutau  = epsilonAs_taumu
  fAutautau = gAu + epsilonAu_tautau
  fAdtautau = gAd + epsilonAd_tautau
  fAstautau = gAs + epsilonAs_tautau
  
  fVuff_bar = (fVuee   * A_ebar_ND    * np.conj(A_ebar_ND ) + fVuemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fVuetau   * A_ebar_ND    * np.conj(A_taubar_ND )
             + fVumue  * A_mubar_ND   * np.conj(A_ebar_ND ) + fVumumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fVumutau  * A_mubar_ND   * np.conj(A_taubar_ND )
             + fVutaue * A_taubar_ND  * np.conj(A_ebar_ND ) + fVutaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fVutautau * A_taubar_ND  * np.conj(A_taubar_ND ))
  
  fVdff_bar = (fVdee   * A_ebar_ND    * np.conj(A_ebar_ND ) + fVdemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fVdetau   * A_ebar_ND    * np.conj(A_taubar_ND )
             + fVdmue  * A_mubar_ND   * np.conj(A_ebar_ND ) + fVdmumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fVdmutau  * A_mubar_ND   * np.conj(A_taubar_ND )
             + fVdtaue * A_taubar_ND  * np.conj(A_ebar_ND ) + fVdtaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fVdtautau * A_taubar_ND  * np.conj(A_taubar_ND ))
  
  fVsff_bar = (fVsee   * A_ebar_ND    * np.conj(A_ebar_ND ) + fVsemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fVsetau   * A_ebar_ND    * np.conj(A_taubar_ND )
             + fVsmue  * A_mubar_ND   * np.conj(A_ebar_ND ) + fVsmumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fVsmutau  * A_mubar_ND   * np.conj(A_taubar_ND )
             + fVstaue * A_taubar_ND  * np.conj(A_ebar_ND ) + fVstaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fVstautau * A_taubar_ND  * np.conj(A_taubar_ND ))
  
  fVufp_bar = (1/AbarND ) * (- fVuemu   * A_ebar_ND    * A_taubar_ND  + fVuetau   * A_ebar_ND    * A_mubar_ND  
                             - fVumumu  * A_mubar_ND   * A_taubar_ND  + fVumutau  * A_mubar_ND   * A_mubar_ND 
                             - fVutaumu * A_taubar_ND  * A_taubar_ND  + fVutautau * A_taubar_ND  * A_mubar_ND )
  
  fVdfp_bar = (1/AbarND ) * (- fVdemu   * A_ebar_ND    * A_taubar_ND  + fVdetau   * A_ebar_ND    * A_mubar_ND  
                             - fVdmumu  * A_mubar_ND   * A_taubar_ND  + fVdmutau  * A_mubar_ND   * A_mubar_ND 
                             - fVdtaumu * A_taubar_ND  * A_taubar_ND  + fVdtautau * A_taubar_ND  * A_mubar_ND )
  
  fVsfp_bar = (1/AbarND ) * (- fVsemu   * A_ebar_ND    * A_taubar_ND  + fVsetau   * A_ebar_ND    * A_mubar_ND  
                             - fVsmumu  * A_mubar_ND   * A_taubar_ND  + fVsmutau  * A_mubar_ND   * A_mubar_ND 
                             - fVstaumu * A_taubar_ND  * A_taubar_ND  + fVstautau * A_taubar_ND  * A_mubar_ND )
  
  fVuft_bar = ((fVuee * A_ebar_ND  + fVumue * A_mubar_ND  + fVutaue * A_taubar_ND ) * (np.conj(A_ebar_ND ) * AbarND  / abs(A_ebar_ND ))
                - ( fVuemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fVuetau   * A_ebar_ND    * np.conj(A_taubar_ND ) 
                  + fVumumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fVumutau  * A_mubar_ND   * np.conj(A_taubar_ND )
                  + fVutaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fVutautau * A_taubar_ND  * np.conj(A_taubar_ND )) * abs(A_ebar_ND )/AbarND )
  
  fVdft_bar = ((fVdee * A_ebar_ND  + fVdmue * A_mubar_ND  + fVdtaue * A_taubar_ND ) * (np.conj(A_ebar_ND ) * AbarND  / abs(A_ebar_ND ))
              - ( fVdemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fVdetau   * A_ebar_ND    * np.conj(A_taubar_ND ) 
                + fVdmumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fVdmutau  * A_mubar_ND   * np.conj(A_taubar_ND )
                + fVdtaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fVdtautau * A_taubar_ND  * np.conj(A_taubar_ND )) * abs(A_ebar_ND )/AbarND )
  
  fVsft_bar = ((fVsee * A_ebar_ND  + fVsmue * A_mubar_ND  + fVstaue * A_taubar_ND ) * (np.conj(A_ebar_ND ) * AbarND  / abs(A_ebar_ND ))
              - ( fVsemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fVsetau   * A_ebar_ND    * np.conj(A_taubar_ND ) 
                + fVsmumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fVsmutau  * A_mubar_ND   * np.conj(A_taubar_ND )
                + fVstaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fVstautau * A_taubar_ND  * np.conj(A_taubar_ND )) * abs(A_ebar_ND )/AbarND )
  
  fAuff_bar = (fAuee   * A_ebar_ND    * np.conj(A_ebar_ND ) + fAuemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fAuetau   * A_ebar_ND    * np.conj(A_taubar_ND )
             + fAumue  * A_mubar_ND   * np.conj(A_ebar_ND ) + fAumumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fAumutau  * A_mubar_ND   * np.conj(A_taubar_ND )
             + fAutaue * A_taubar_ND  * np.conj(A_ebar_ND ) + fAutaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fAutautau * A_taubar_ND  * np.conj(A_taubar_ND ))
  
  fAdff_bar = (fAdee   * A_ebar_ND    * np.conj(A_ebar_ND ) + fAdemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fAdetau   * A_ebar_ND    * np.conj(A_taubar_ND )
             + fAdmue  * A_mubar_ND   * np.conj(A_ebar_ND ) + fAdmumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fAdmutau  * A_mubar_ND   * np.conj(A_taubar_ND )
             + fAdtaue * A_taubar_ND  * np.conj(A_ebar_ND ) + fAdtaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fAdtautau * A_taubar_ND  * np.conj(A_taubar_ND ))
  
  fAsff_bar = (fAsee   * A_ebar_ND    * np.conj(A_ebar_ND ) + fAsemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fAsetau   * A_ebar_ND    * np.conj(A_taubar_ND )
             + fAsmue  * A_mubar_ND   * np.conj(A_ebar_ND ) + fAsmumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fAsmutau  * A_mubar_ND   * np.conj(A_taubar_ND )
             + fAstaue * A_taubar_ND  * np.conj(A_ebar_ND ) + fAstaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fAstautau * A_taubar_ND  * np.conj(A_taubar_ND ))
  
  fAufp_bar = (1/AbarND ) * (- fAuemu   * A_ebar_ND    * A_taubar_ND  + fAuetau   * A_ebar_ND    * A_mubar_ND  
                             - fAumumu  * A_mubar_ND   * A_taubar_ND  + fAumutau  * A_mubar_ND   * A_mubar_ND 
                             - fAutaumu * A_taubar_ND  * A_taubar_ND  + fAutautau * A_taubar_ND  * A_mubar_ND )
  
  fAdfp_bar = (1/AbarND ) * (- fAdemu   * A_ebar_ND    * A_taubar_ND  + fAdetau   * A_ebar_ND    * A_mubar_ND  
                             - fAdmumu  * A_mubar_ND   * A_taubar_ND  + fAdmutau  * A_mubar_ND   * A_mubar_ND 
                             - fAdtaumu * A_taubar_ND  * A_taubar_ND  + fAdtautau * A_taubar_ND  * A_mubar_ND )
  
  fAsfp_bar = (1/AbarND ) * (- fAsemu   * A_ebar_ND    * A_taubar_ND  + fAsetau   * A_ebar_ND    * A_mubar_ND  
                             - fAsmumu  * A_mubar_ND   * A_taubar_ND  + fAsmutau  * A_mubar_ND   * A_mubar_ND 
                             - fAstaumu * A_taubar_ND  * A_taubar_ND  + fAstautau * A_taubar_ND  * A_mubar_ND )
  
  fAuft_bar = ((fAuee * A_ebar_ND  + fAumue * A_mubar_ND  + fAutaue * A_taubar_ND ) * (np.conj(A_ebar_ND ) * AbarND  / abs(A_ebar_ND ))
                - ( fAuemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fAuetau   * A_ebar_ND    * np.conj(A_taubar_ND ) 
                  + fAumumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fAumutau  * A_mubar_ND   * np.conj(A_taubar_ND )
                  + fAutaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fAutautau * A_taubar_ND  * np.conj(A_taubar_ND )) * abs(A_ebar_ND )/AbarND )
  
  fAdft_bar = ((fAdee * A_ebar_ND  + fAdmue * A_mubar_ND  + fAdtaue * A_taubar_ND ) * (np.conj(A_ebar_ND ) * AbarND  / abs(A_ebar_ND ))
              - ( fAdemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fAdetau   * A_ebar_ND    * np.conj(A_taubar_ND ) 
                + fAdmumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fAdmutau  * A_mubar_ND   * np.conj(A_taubar_ND )
                + fAdtaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fAdtautau * A_taubar_ND  * np.conj(A_taubar_ND )) * abs(A_ebar_ND )/AbarND )
  
  fAsft_bar = ((fAsee * A_ebar_ND  + fAsmue * A_mubar_ND  + fAstaue * A_taubar_ND ) * (np.conj(A_ebar_ND ) * AbarND  / abs(A_ebar_ND ))
              - ( fAsemu   * A_ebar_ND    * np.conj(A_mubar_ND ) + fAsetau   * A_ebar_ND    * np.conj(A_taubar_ND ) 
                + fAsmumu  * A_mubar_ND   * np.conj(A_mubar_ND ) + fAsmutau  * A_mubar_ND   * np.conj(A_taubar_ND )
                + fAstaumu * A_taubar_ND  * np.conj(A_mubar_ND ) + fAstautau * A_taubar_ND  * np.conj(A_taubar_ND )) * abs(A_ebar_ND )/AbarND )
  
  sigmabar_p_ff = sigma_0p * (( 2/3 * x1uP  - Mp/(2*E) *  x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP  ) * ( abs(fVuff_bar)**2 + abs(fAuff_bar)**2 )
                            + ( 2/3 * x1dP  - Mp/(2*E) *  x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP   ) * ( abs(fVdff_bar)**2 + abs(fAdff_bar)**2 )
                            + ( 2/3 * x1sP - Mp/(2*E) *  x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsff_bar)**2 + abs(fAsff_bar)**2 )
                            - ( 2/3 * x1uM   - Mp/(2*E) *  x2uM + (Mp/(2*E))**2 *  x3uM  ) * np.real(fVuff_bar * np.conj(fAuff_bar))
                            - ( 2/3 * x1dM  - Mp/(2*E) *  x2dM  + (Mp/(2*E))**2 *  x3dM ) * np.real(fVdff_bar * np.conj(fAdff_bar)) 
                            - ( 2/3 * x1sM - Mp/(2*E) * x2sM   + (Mp/(2*E))**2 * x3sM     ) * np.real(fVsff_bar * np.conj(fAsff_bar)))
  
  sigmabar_p_fp = sigma_0p * (( 2/3 * x1uP  - Mp/(2*E) *  x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP  ) * ( abs(fVufp_bar)**2 + abs(fAufp_bar)**2 )
                            + ( 2/3 * x1dP  - Mp/(2*E) *  x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP   ) * ( abs(fVdfp_bar)**2 + abs(fAdfp_bar)**2 )
                            + ( 2/3 * x1sP - Mp/(2*E) *  x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsfp_bar)**2 + abs(fAsfp_bar)**2 )
                            - ( 2/3 * x1uM   - Mp/(2*E) *  x2uM + (Mp/(2*E))**2 *  x3uM  ) * np.real(fVufp_bar * np.conj(fAufp_bar))
                            - ( 2/3 * x1dM  - Mp/(2*E) *  x2dM  + (Mp/(2*E))**2 *  x3dM ) * np.real(fVdfp_bar * np.conj(fAdfp_bar))
                            - ( 2/3 * x1sM - Mp/(2*E) * x2sM   + (Mp/(2*E))**2 * x3sM     ) * np.real(fVsfp_bar * np.conj(fAsfp_bar)))
  
  sigmabar_p_ft = sigma_0p * (( 2/3 * x1uP  - Mp/(2*E) *  x2uP + 3/2 * (Mp/(2*E))**2 *  x3uP  ) * ( abs(fVuft_bar)**2 + abs(fAuft_bar)**2 )
                            + ( 2/3 * x1dP  - Mp/(2*E) *  x2dP + 3/2 * (Mp/(2*E))**2 *  x3dP   ) * ( abs(fVdft_bar)**2 + abs(fAdft_bar)**2 )
                            + ( 2/3 * x1sP - Mp/(2*E) *  x2sP + 3/2 * (Mp/(2*E))**2 *  x3sP ) * ( abs(fVsft_bar)**2 + abs(fAsft_bar)**2 )
                            - ( 2/3 * x1uM   - Mp/(2*E) *  x2uM + (Mp/(2*E))**2 *  x3uM  ) * np.real(fVuft_bar * np.conj(fAuft_bar))
                            - ( 2/3 * x1dM  - Mp/(2*E) *  x2dM  + (Mp/(2*E))**2 *  x3dM ) * np.real(fVdft_bar * np.conj(fAdft_bar))
                            - ( 2/3 * x1sM - Mp/(2*E) * x2sM   + (Mp/(2*E))**2 * x3sM     ) * np.real(fVsft_bar * np.conj(fAsft_bar)))
  
  sigmabar_p_total = sigmabar_p_ff + sigmabar_p_fp + sigmabar_p_ft
  
  sigmabar_n_ff = sigma_0n * (( 2/3 * x1uP  - Mn/(2*E) *  x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP  ) * ( abs(fVdff_bar)**2 + abs(fAdff_bar)**2 )
                            + ( 2/3 * x1dP  - Mn/(2*E) *  x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP   ) * ( abs(fVuff_bar)**2 + abs(fAuff_bar)**2 )
                            + ( 2/3 * x1sP - Mn/(2*E) *  x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsff_bar)**2 + abs(fAsff_bar)**2 )
                            - ( 2/3 * x1uM   - Mn/(2*E) *  x2uM + (Mn/(2*E))**2 *  x3uM  ) * np.real(fVdff_bar * np.conj(fAdff_bar))
                            - ( 2/3 * x1dM  - Mn/(2*E) *  x2dM  + (Mn/(2*E))**2 *  x3dM ) * np.real(fVuff_bar * np.conj(fAuff_bar))
                            - ( 2/3 * x1sM - Mn/(2*E) * x2sM   + (Mn/(2*E))**2 * x3sM     ) * np.real(fVsff_bar * np.conj(fAsff_bar)))
  
  sigmabar_n_fp = sigma_0n * (( 2/3 * x1uP  - Mn/(2*E) *  x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP  ) * ( abs(fVdfp_bar)**2 + abs(fAdfp_bar)**2 )
                            + ( 2/3 * x1dP  - Mn/(2*E) *  x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP   ) * ( abs(fVufp_bar)**2 + abs(fAufp_bar)**2 )
                            + ( 2/3 * x1sP - Mn/(2*E) *  x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsfp_bar)**2 + abs(fAsfp_bar)**2 )
                            - ( 2/3 * x1uM   - Mn/(2*E) *  x2uM + (Mn/(2*E))**2 *  x3uM  ) * np.real(fVdfp_bar * np.conj(fAdfp_bar))
                            - ( 2/3 * x1dM  - Mn/(2*E) *  x2dM  + (Mn/(2*E))**2 *  x3dM ) * np.real(fVufp_bar * np.conj(fAufp_bar))
                            - ( 2/3 * x1sM - Mn/(2*E) * x2sM   + (Mn/(2*E))**2 * x3sM     ) * np.real(fVsfp_bar * np.conj(fAsfp_bar)))
  
  sigmabar_n_ft = sigma_0n * (( 2/3 * x1uP  - Mn/(2*E) *  x2uP + 3/2 * (Mn/(2*E))**2 *  x3uP  ) * ( abs(fVdft_bar)**2 + abs(fAdft_bar)**2 )
                            + ( 2/3 * x1dP  - Mn/(2*E) *  x2dP + 3/2 * (Mn/(2*E))**2 *  x3dP   ) * ( abs(fVuft_bar)**2 + abs(fAuft_bar)**2 )
                            + ( 2/3 * x1sP - Mn/(2*E) *  x2sP + 3/2 * (Mn/(2*E))**2 *  x3sP ) * ( abs(fVsft_bar)**2 + abs(fAsft_bar)**2 )
                            - ( 2/3 * x1uM   - Mn/(2*E) *  x2uM + (Mn/(2*E))**2 *  x3uM  ) * np.real(fVdft_bar * np.conj(fAdft_bar))
                            - ( 2/3 * x1dM  - Mn/(2*E) *  x2dM  + (Mn/(2*E))**2 *  x3dM ) * np.real(fVuft_bar * np.conj(fAuft_bar))
                            - ( 2/3 * x1sM - Mn/(2*E) * x2sM   + (Mn/(2*E))**2 * x3sM     ) * np.real(fVsft_bar * np.conj(fAsft_bar)))
  
  sigmabar_n_total = sigmabar_n_ff + sigmabar_n_fp + sigmabar_n_ft
  
  Ndiff = (F_mubar_ND * POT * tbar * NpND * sigmabar_p_total + F_mubar_ND * POT * tbar * NnND * sigmabar_n_total)
  
  Nintegrated = 0
  for j in range (len(E)):
      Nintegrated = Nintegrated + Ndiff[j]*0.25
  return Nintegrated

def NPNbarND(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau):
  return NND(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau) + NbarND(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau)

def NMNbarND(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau):
  return NND(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau) - NbarND(epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau)
