edgeofpy
========

**edgeofpy** is a Python package for assessing the distance to criticality in
brain dynamics, especially from M/EEG recordings. It brings together measures
of several "edges" at which the brain is thought to operate:

- **Activity propagation**: neuronal avalanches, power-law fitting, the
  deviation from criticality coefficient (DCC), shape collapse, branching
  ratio, susceptibility.
- **Synchrony**: phase lag index (PLI), phase lag entropy (PLE), pair
  correlation function (PCF), global lability of synchronization (GLS),
  phase-locking avalanches.
- **Chaos**: the modified 0-1 chaos test and lambda max.
- **Pairwise maximum-entropy (Ising) models**: model fitting, Metropolis
  sampling and phase diagrams.

.. code-block:: bash

   pip install edgeofpy

.. toctree::
   :maxdepth: 2

   install
   quickstart
   api
   citing
   changelog
