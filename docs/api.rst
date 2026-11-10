API reference
=============

All functions listed here can be imported from the top-level package, e.g.
``eop.detect_avalanches`` after ``import edgeofpy as eop``.

.. currentmodule:: edgeofpy

Avalanches
----------

Detection of neuronal avalanches, power-law fitting, DCC, shape collapse, branching ratio and related measures.

.. autosummary::
   :toctree: generated/

   detect_avalanches
   plot_pdf
   plot_cdf
   fit_powerlaw
   fit_third_exponent
   dcc
   shape_collapse_error
   lattice_search
   dcc_collapse
   dcc_collapse_2
   avl_repertoire
   avl_pattern_dissimilarity
   avl_branching_ratio
   branching_ratio
   susceptibility
   fano_factor

Chaos
-----

Edge-of-chaos measures.

.. autosummary::
   :toctree: generated/

   z1_chaos_test
   chaos_pipeline
   lambda_max
   minmaxsig
   schreiber_denoise

Synchrony
---------

Edge-of-synchrony measures and phase-locking avalanches.

.. autosummary::
   :toctree: generated/

   pcf
   pli
   ple
   complex_phase_relationship
   phase_locking
   phase_lock_interval
   global_lability
   detect_phase_avls
   detect_phase_avls_full

Ising models
------------

Pairwise maximum-entropy (Ising) models and phase diagrams.

.. autosummary::
   :toctree: generated/

   fit_pmem
   fit_beta
   sk_diagram
   metropolis
   plot_sk
   ht_diagram
   ising_energy
   heat_capacity
   energy_landscape_2d

Utilities
---------

Event detection and time binning.

.. autosummary::
   :toctree: generated/

   binarized_events
   time_bin_events

Model systems
-------------

.. currentmodule:: edgeofpy.models

Simulations of systems with known critical points, for validating analyses
and learning how the measures behave. These are used as
``eop.models.kuramoto(...)``, for example.

.. autosummary::
   :toctree: generated/

   branching_process
   branching_network
   kuramoto
   logistic_map
   linear_network
   ising_samples
