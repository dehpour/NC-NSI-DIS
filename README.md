# NC-NSI-DIS
Here is our codebase for computing the Neutral Current (NC) Deep Inelastic Scattering (DIS) of neutrino from nucleon number of events (for now, only) in the presence of axial NC Non-Standard Interaction (NSI) at Near Detector (ND) and Far Detectors (FD) of a Deep Underground Neutrino Experiment (DUNE) setup.

### Prerequisite
We just used `python3` with `numpy` and `pandas` libs.

### Install
First, you should create your own Python script in the same dir where you clone this repo. Then you can import our framework into your code via
```python
from functions import *
```

### Experiment
The loaded experiment is the `TDR version of DUNE with CP optimized flux` by default. One can use another arbitrary experiment like `EXP` by providing auxiliary files like fluxes and oscillation amplitudes and make their own `experiment_EXP.py` file and then replace
```python
from experiment_EXP import *
```
instead of
```python
from experiment_DUNE_CPOptimized import *
```
in the `functions.py`.

## NC axial NSI in DIS of neutrino-nucleon at the DUNE-like
It can be used via
```python
NFD(*arguments)
NbarFD(*arguments)
NND(*arguments)
NbarND(*arguments)
```
to calculate the number of events at DUNE-like experiment in the FD with neutrino and antineutrino mode and ND with neutrino and antineutrino mode with NC axial nonstandard parameters as follows:
```python
arguments = [epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau]
```

## Cite
If you do use anything here please cite the paper [arXiv:2312.12420](http://arxiv.org/abs/2312.12420) [[JHEP 04 (2024) 038](https://doi.org/10.1007/JHEP04(2024)038)].
```
@article{Abbaslu:2023vqk,
    author = "Abbaslu, Saeed and Dehpour, Mehran and Farzan, Yasaman and Safari, Sahar",
    title = "{Searching for axial neutral current non-standard interactions of neutrinos by DUNE-like experiments}",
    eprint = "2312.12420",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1007/JHEP04(2024)038",
    journal = "JHEP",
    volume = "04",
    pages = "038",
    year = "2024"
}
```
