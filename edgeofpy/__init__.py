"""edgeofpy: tools for assessing the distance to criticality in brain
dynamics."""

from importlib.metadata import version, PackageNotFoundError

from edgeofpy.avalanche import (detect_avalanches,
                                plot_pdf,
                                plot_cdf,
                                fit_powerlaw,
                                fit_third_exponent,
                                dcc,
                                shape_collapse_error,
                                lattice_search,
                                dcc_collapse,
                                dcc_collapse_2,
                                avl_repertoire,
                                avl_pattern_dissimilarity,
                                avl_branching_ratio,
                                branching_ratio,
                                susceptibility,
                                fano_factor,
                                )
from edgeofpy.chaos import (z1_chaos_test,
                            chaos_pipeline,
                            lambda_max,
                            minmaxsig,
                            schreiber_denoise,
                            )
from edgeofpy.synchrony import (pcf,
                                pli,
                                ple,
                                complex_phase_relationship,
                                phase_locking,
                                phase_lock_interval,
                                global_lability,
                                detect_phase_avls,
                                detect_phase_avls_full,
                                )
from edgeofpy.ising import (fit_pmem,
                            fit_beta,
                            sk_diagram,
                            metropolis,
                            plot_sk,
                            ht_diagram,
                            ising_energy,
                            heat_capacity,
                            energy_landscape_2d,
                            )
from edgeofpy import models # noqa: F401  (submodule: eop.models.<name>)
from edgeofpy.utils import (binarized_events,
                            time_bin_events,
                            )

try:
    __version__ = version('edgeofpy')
except PackageNotFoundError: # not installed, e.g. run from a source checkout
    __version__ = 'unknown'

__all__ = [
    # avalanche
    'detect_avalanches', 'plot_pdf', 'plot_cdf', 'fit_powerlaw',
    'fit_third_exponent', 'dcc', 'shape_collapse_error', 'lattice_search',
    'dcc_collapse', 'dcc_collapse_2', 'avl_repertoire',
    'avl_pattern_dissimilarity', 'avl_branching_ratio', 'branching_ratio',
    'susceptibility', 'fano_factor',
    # chaos
    'z1_chaos_test', 'chaos_pipeline', 'lambda_max', 'minmaxsig',
    'schreiber_denoise',
    # synchrony
    'pcf', 'pli', 'ple', 'complex_phase_relationship', 'phase_locking',
    'phase_lock_interval', 'global_lability', 'detect_phase_avls',
    'detect_phase_avls_full',
    # ising
    'fit_pmem', 'fit_beta', 'sk_diagram', 'metropolis', 'plot_sk',
    'ht_diagram', 'ising_energy', 'heat_capacity', 'energy_landscape_2d',
    # utils
    'binarized_events', 'time_bin_events',
]
