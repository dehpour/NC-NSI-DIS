# Searching for Axial Neutral Current Non-Standard Interaction of neutrinos by DUNE-like experiment
Our codebase computing the number of Neutral Current Deep Inelastic Scattering events in the presence of Axial Non-Standard Interaction at near and far detectors of a DUNE-like setup is here.

## Prerequisite
```
python3
numpy
pandas
```

## How to Use 
In your own Python program, by using
```python
from functions import *
```
load the framework. Then, can be using
```python
NFD(*arguments)
NbarFD(*arguments)
NND(*arguments)
NbarND(*arguments)
```
calculated the number of events @ DUNE-like experiment in the Far Detector (FD) with neutrino and antineutrino mode and Near Detector (ND) with neutrino and antineutrino mode respectively with axial nonstandard parameters as follows:
```python
arguments = [epsilonAu_ee,epsilonAu_mue,epsilonAu_taue,epsilonAu_mumu,epsilonAu_taumu,epsilonAu_tautau,epsilonAd_ee,epsilonAd_mue,epsilonAd_taue,epsilonAd_mumu,epsilonAd_taumu,epsilonAd_tautau,epsilonAs_ee,epsilonAs_mue,epsilonAs_taue,epsilonAs_mumu,epsilonAs_taumu,epsilonAs_tautau]
```

### Changing Experiment
By default, the loaded experiment is the `TDR version of DUNE with CP Optimized flux`. One can use another arbitrary experiment like `EXP` by providing auxiliary files like fluxes and oscillation amplitudes and make their own `experiment_EXP.py` file and then replace
```python
from experiment_EXP import *
```
instead of
```python
from experiment_DUNE_CPOptimized import *
```
in the `functions.py` file.

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
