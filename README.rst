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
    - Power law fitting (via `powerlaw <https://github.com/jeffalstott/powerlaw>`)
    - Deviation from criticality coefficients (DCC)
    - Shape collapse error
    - Shew's kappa
    - Avalanche reprtoire size and diversity
    - Branching ratios
    - Susceptibility
    - Fano factor

- Edge of chaos measures
    - 0-1 chaos tested
    - Lambda max (Dahmen et al., 2019)

- Edge of synchrony measures
    - Pair correlation function (PCF)
    - Phase lag index (PLI)
    - Phase lag entropy (PLE)
    - Global lability index (GLI)
    - Phase-locking avalanche detection


Installation
------------
Install edgeofpy by running:

    pip3 install edgeofpy

Documentation
-------------
Coming soon.

Requirements
------------
- Python 3.7 or later
- numpy
- scipy
- matplotlib
- powerlaw
- neurokit2

License
-------
The project is licensed under the Apache 2.0 license.
