"""edgeofpy main module"""

from edgeofpy.avalanche import (detect_avalanches,
                                plot_pdf,
                                plot_third,
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
                                shew_kappa,
                                fano_factor,
                                )
from edgeofpy.chaos import (z1_chaos_test,
                            lambda_max,
                            minmaxsig,
                            )
from edgeofpy.synchrony import (pcf,
                                pli,
                                ple,
                                complex_phase_relationship,
                                phase_locking,
                                phase_lock_interval,
                                global_lability
                                )
from edgeofpy.utils import (binarized_events,
                            time_bin_events,
                            _detect_start_end,
                            )
