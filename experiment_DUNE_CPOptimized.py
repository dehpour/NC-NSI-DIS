import pandas as pd
import numpy as np
from constants import *

"""
Amplitude
"""

AMP_NU_FD = pd.read_csv('./amp/NuFIT2022_neutrino_LBNEFD_globes_amplitude.csv', header=None)

E    = AMP_NU_FD[0]

A_e_FD   = AMP_NU_FD[1]+AMP_NU_FD[2]*1j
A_mu_FD  = AMP_NU_FD[3]+AMP_NU_FD[4]*1j
A_tau_FD = AMP_NU_FD[5]+AMP_NU_FD[6]*1j

AFD = (abs(A_mu_FD)**2 + abs(A_tau_FD)**2)**(1/2)

AMP_NUBAR_FD = pd.read_csv('./amp/NuFIT2022_antineutrino_LBNEFD_globes_amplitude.csv', header=None)

A_ebar_FD   = AMP_NUBAR_FD[1]+AMP_NUBAR_FD[2]*1j
A_mubar_FD  = AMP_NUBAR_FD[3]+AMP_NUBAR_FD[4]*1j
A_taubar_FD = AMP_NUBAR_FD[5]+AMP_NUBAR_FD[6]*1j

AbarFD = (abs(A_mubar_FD)**2 + abs(A_taubar_FD)**2)**(1/2)


AMP_NU_ND = pd.read_csv('./amp/neutrino_LBNEND_globes_amplitude.csv', header=None)

A_e_ND   = AMP_NU_ND[1]+AMP_NU_ND[2]*1j
A_mu_ND  = AMP_NU_ND[3]+AMP_NU_ND[4]*1j
A_tau_ND = AMP_NU_ND[5]+AMP_NU_ND[6]*1j

AND = (abs(A_mu_ND)**2 + abs(A_tau_ND)**2)**(1/2)

AMP_NUBAR_ND = pd.read_csv('./amp/antineutrino_LBNEND_globes_amplitude.csv', header=None)

A_ebar_ND   = AMP_NUBAR_ND[1]+AMP_NUBAR_ND[2]*1j
A_mubar_ND  = AMP_NUBAR_ND[3]+AMP_NUBAR_ND[4]*1j
A_taubar_ND = AMP_NUBAR_ND[5]+AMP_NUBAR_ND[6]*1j

AbarND = (abs(A_mubar_ND)**2 + abs(A_taubar_ND)**2)**(1/2)

"""
Flux
"""

POT = 1.1e21

t    = 6.5 #years
tbar = 6.5 #years

FLUX_NU_ND    = pd.read_csv('./flux/OptimizedEngineeredNov2017_neutrino_LBNEND_globes_flux.csv', header=None)
FLUX_NUBAR_ND = pd.read_csv('./flux/OptimizedEngineeredNov2017_antineutrino_LBNEND_globes_flux.csv', header=None)

F_mu_ND    = FLUX_NU_ND[2]
F_mubar_ND = FLUX_NUBAR_ND[5]

FLUX_NU_FD    = pd.read_csv('./flux/OptimizedEngineeredNov2017_neutrino_LBNEFD_globes_flux.csv', header=None)
FLUX_NUBAR_FD = pd.read_csv('./flux/OptimizedEngineeredNov2017_antineutrino_LBNEFD_globes_flux.csv', header=None)

F_mu_FD    = FLUX_NU_FD[2]
F_mubar_FD = FLUX_NUBAR_FD[5]

"""
Detector
"""

MfidND = 67.2e-3 * 5.5e32 # kton to GeV
NnND = 22/40 * MfidND/Mn
NpND = 18/40 * MfidND/Mp

MfidFD = 40 * 5.5e32 # kton to GeV
NnFD = 22/40 * MfidFD/Mn
NpFD = 18/40 * MfidFD/Mp
