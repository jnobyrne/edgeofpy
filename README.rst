**Development Status:** 3 - Alpha. Some features still need to be added and tested.

==========
EDGE OF PY
==========

Package for assessing the distance to criticality in brain dynamics.

Features
--------
- Avalanche criticality measures
    - Avalanche detection
    - Avalanche distribution plotting
    - Power-law fitting (via `powerlaw <https://github.com/jeffalstott/powerlaw>`_)
    - Deviation from criticality coefficient (DCC)
    - Shape collapse error
    - Avalanche repertoire size and diversity
    - Branching ratios
    - Susceptibility
    - Fano factor

- Edge of chaos measures
    - 0-1 chaos test
    - Lambda max (Dahmen et al., 2019)

- Edge of synchrony measures
    - Pair correlation function (PCF)
    - Phase lag index (PLI)
    - Phase lag entropy (PLE)
    - Global lability of synchronization (GLS)
    - Phase-locking avalanche detection

- Ising model functions
    - Fitting the spin-glass model by pseudo-likelihood maximization
    - SK phase diagram inference (Ezaki et al., 2020)
    - Metropolis-Hastings algorithm
    - Heavy-tailed phase diagram inference
    - Ising energy
    - Specific heat
    - Energy landscape sample


Installation
------------
Install edgeofpy by running:

    pip3 install edgeofpy

Documentation
-------------
Coming soon.

Requirements
------------
- Python 3.10 or later
- numpy >= 1.22
- scipy >= 1.7
- matplotlib >= 3.5
- powerlaw >= 1.5

Citing edgeofpy
---------------
If you use edgeofpy, please cite it: use the "Cite this repository" button on
GitHub, or the metadata in ``CITATION.cff``. Please also cite the original
papers of the methods you use; they are listed in the function docstrings.
``fit_powerlaw`` relies on the powerlaw package:

    Alstott J, Bullmore E, Plenz D (2014). powerlaw: A Python package for
    analysis of heavy-tailed distributions. PLoS ONE 9(1): e85777.
    https://doi.org/10.1371/journal.pone.0085777

License
-------
The project is licensed under the Apache 2.0 license.
