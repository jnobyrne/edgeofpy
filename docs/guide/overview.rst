Choosing a measure
==================

The brain has been proposed to operate close to several kinds of
transitions, or "edges". edgeofpy groups its measures by transition. They
need different inputs, and they do not all measure the same thing: some
measure the distance to a transition, others measure on which side of it
a system lies, or how far into one of its sides.

.. list-table::
   :header-rows: 1
   :widths: 18 22 25 35

   * - Family
     - Transition
     - Input
     - What the measures tell you
   * - Avalanches
     - Between activity that dies out and activity that grows
       (critical branching)
     - Multichannel signals, thresholded into events
     - Power-law exponents and their scaling relation (DCC), shape collapse:
       consistency with a critical point. Branching ratio: side of the
       transition. Susceptibility, Fano factor: size of fluctuations,
       which peak at the transition.
   * - Chaos
     - Between stable (regular) and chaotic dynamics
     - Single signals (0-1 test); many units with trials (lambda max)
     - 0-1 test: whether a signal is chaotic (K close to 1) or regular
       (K close to 0). Lambda max: how close a network is to instability
       (1).
   * - Synchrony
     - Between incoherent and synchronized oscillations
     - Band-pass filtered signals
     - PCF: fluctuations of synchrony, which peak at the transition.
       PLI, PLE, order parameter: degree of synchrony. GLS: changes in the
       number of phase-locked pairs.
   * - Ising models
     - Phase transitions of a model fitted to the data (paramagnetic,
       ferromagnetic, spin glass)
     - Binarized multichannel signals (-1/+1)
     - Fitted fields and couplings; position of the data on the specific
       heat curve and on the Sherrington-Kirkpatrick phase diagram.

What the worked examples show
-----------------------------

The :doc:`worked examples <../auto_examples/index>` test the measures on
simulated systems whose distance to the transition is known. They point to
the following limits:

- **A DCC close to 0 is necessary, but not sufficient.** In a branching
  network tuned through its critical point, the DCC returns close to 0 in
  the supercritical regime (:ref:`sphx_glr_auto_examples_plot_03_criticality_sweep.py`).
  Power laws can also arise without criticality (Touboul & Destexhe, 2017).
  Combine the DCC with measures that tell the side of the transition, like
  the branching ratio.
- **The branching ratio saturates.** Above the critical point, it stays
  close to 1 instead of growing (:ref:`sphx_glr_auto_examples_plot_03_criticality_sweep.py`).
- **The 0-1 test does not measure the distance to the edge of chaos.** It
  separates regular from chaotic signals, but K stays close to 0 for all
  regular dynamics, near the edge or far from it
  (:ref:`sphx_glr_auto_examples_plot_04_edge_of_chaos.py`). It is also
  sensitive to measurement noise; :func:`edgeofpy.chaos_pipeline` with
  denoising reduces this.
- **Lambda max is best used for relative comparisons.** It tracks the true
  value in the mid range, but underestimates it close to 1, and more so when
  only a fraction of the units is recorded
  (:ref:`sphx_glr_auto_examples_plot_04_edge_of_chaos.py`).
- **The PLI and the PLE measure the degree of synchrony,** not the distance
  to the transition: they change steadily through it
  (:ref:`sphx_glr_auto_examples_plot_06_synchrony.py`).
- **The PCF peaks at the transition only with many units.** With a few
  tens of channels, finite-size fluctuations mask the peak; a low PCF still
  identifies the synchronized side (:ref:`sphx_glr_auto_examples_plot_06_synchrony.py`).
- **Ising model results depend on the size of the model.** In small
  models, the specific heat peak is broad and shifted, and the phase
  boundaries depend on the simulation length
  (:ref:`sphx_glr_auto_examples_plot_05_ising.py`).

References
----------

- Touboul & Destexhe (2017) Phys Rev E 95(1), 012413.
