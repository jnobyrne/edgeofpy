Quick start
===========

All functions are available from the top-level package:

.. code-block:: python

   import edgeofpy as eop

Most functions take a NumPy array of shape (channels, samples). With MNE,
for example, use ``data = raw.get_data()`` and ``s_freq = raw.info['sfreq']``.

Neuronal avalanches
-------------------

Detect avalanches, fit power laws to their sizes and durations, and compute
the deviation from criticality coefficient (DCC):

.. code-block:: python

   import numpy as np
   import edgeofpy as eop

   # data: channels x samples, cleaned signal; s_freq: sampling frequency
   avls, _, _, _ = eop.detect_avalanches(data, s_freq, max_iei=0.004,
                                         threshold=3)
   sizes = np.array([a['size'] for a in avls])
   durations = np.array([a['dur_bin'] for a in avls])

   fit_s = eop.fit_powerlaw(sizes)          # xmin chosen automatically
   fit_d = eop.fit_powerlaw(durations)
   third = eop.fit_third_exponent(sizes, durations,
                                  dur_range=(fit_d['xmin'], fit_d['xmax']))
   dcc = eop.dcc(fit_s['power_law_exp'], fit_d['power_law_exp'], third)

   eop.plot_pdf(sizes, binning='log')        # size distribution

``fit_powerlaw`` returns a dictionary with the fitted exponent, the fitted
range (``xmin``, ``xmax``) and likelihood-ratio comparisons with alternative
distributions, which are worth checking before interpreting the exponents.

Other measures
--------------

.. code-block:: python

   eop.pli(data)                             # phase lag index
   eop.z1_chaos_test(data[0])                # 0-1 chaos test, one channel

Measures based on phase, such as the PLI, are usually computed on
narrow-band filtered signals.

See the :doc:`api` for every function and its options.
