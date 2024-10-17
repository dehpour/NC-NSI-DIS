# The neutrino-Nucleus NC DIS in the presence of axial NC NSI in the DUNE-like
Our codebase computing the Neutral Current Deep Inelastic Scattering of neutrino from nucleon number of events in the presence of axial NC Non-Standard Interaction at near and far detectors of a DUNE-like setup is here.

## Prerequisite
We just used `python3` with `numpy` and `pandas` libs.

## How to use
First, you should create your own Python script in the same dir where you clone this repo. Then you can import our framework into your code via
```python
from functions import *
```
Then, it can be used via
```python
NFD(*arguments)
NbarFD(*arguments)
NND(*arguments)
NbarND(*arguments)
```
to calculate the number of events at DUNE experiment in the Far Detector (FD) with neutrino and antineutrino mode and Near Detector (ND) with neutrino and antineutrino mode with NC axial nonstandard parameters as follows:
```python
arguments = [epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau]
```

### Experiments
By default, the loaded experiment is the `TDR version of DUNE with CP optimized flux`. One can use another arbitrary experiment like `EXP` by providing auxiliary files like fluxes and oscillation amplitudes and make their own `experiment_EXP.py` file and then replace
```python
from experiment_EXP import *
```
instead of
```python
from experiment_DUNE_CPOptimized import *
```
in the `functions.py`.

## Citation
If you do use anything here please cite the paper [arXiv:2312.12420](http://arxiv.org/abs/2312.12420).
```
@article{Abbaslu:2023vqk,
    author = "Abbaslu, Saeed and Dehpour, Mehran and Farzan, Yasaman and Safari, Sahar",
    title = "{Searching for Axial Neutral Current Non-Standard Interactions of neutrinos by DUNE-like experiments}",
    eprint = "2312.12420",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "12",
    year = "2023"
}
```
