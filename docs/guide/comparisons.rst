Comparing conditions and subjects
=================================

Most studies ask whether one condition, group or subject is closer to a
critical point than another. The absolute values of the measures depend on
many choices and on the amount of data, so a comparison is only valid if
these are the same for everything compared. The following checklist
summarizes the recommendations of the other pages.

Analysis choices
----------------

- **Preprocess all recordings identically:** same filters, reference,
  artifact rejection and, as far as possible, the same channels.
- **Fix the parameters of the analysis in advance** and use the same for
  all recordings: event threshold, time bin, fitting ranges of the power
  laws (:doc:`avalanches`). Ranges chosen automatically for each recording
  can create differences of their own.
- **Check the robustness of the conclusions** to the choices that matter
  most, above all the time bin, and report results over a range of values
  rather than for a single one (Fontenele et al., 2024).

Amount of data
--------------

- **Match the number of avalanches** when comparing exponents or DCCs
  (``n_avalanches`` in :func:`edgeofpy.estimate_dcc`): with fewer
  avalanches, the DCC is larger, whatever the distance to criticality.
- **Match recording durations** where possible; this also helps the other
  measures, whose estimates become more variable with less data.
- **Report confidence intervals** (``n_boot``), and for the DCC, judge
  consistency with criticality from the signed deviation.

Interpretation
--------------

- **Use several measures.** Each has blind spots (:doc:`overview`): a DCC
  close to 0 does not tell on which side of the transition a system is; the
  branching ratio does, but saturates above it. Measures that agree are
  more convincing than any one of them.
- **Control for spectral power.** Oscillations affect most measures: a
  strong rhythm, such as the alpha rhythm with the eyes closed, makes events
  regular in time, which changes the avalanches, and increases the measures
  of synchrony through the quality of the phase estimates. A difference
  between conditions that differ in power may reflect the oscillation rather
  than the distance to criticality; include power as a covariate, or check
  that the result holds in frequency bands or periods with comparable power.
- **Compare relative values for approximate measures.** Some measures are
  biased in a way that preserves the order of systems but not the absolute
  value, like :func:`edgeofpy.lambda_max` close to 1 or the branching ratio
  with subsampling. They support statements like "closer in A than in B"
  better than "at a distance of x".
- **Within-subject comparisons are more sensitive** than between-subject
  comparisons, as they remove the many sources of variability between
  individuals (anatomy, electrode placement, signal-to-noise ratio).

References
----------

- Fontenele et al. (2024) Sci Adv 10(17), eadj9303.
