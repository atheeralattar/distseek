import chisquare_tests as tests
import pandas as pd
import numpy as np
from generators import Generator

def estimate(df):
    """
    Final distribution estimator
    """

    if np.array_equal(np.unique(df), [0, 1]):
        p = np.count_nonzero(df) / len(df)
        final_results = {'guessed_dist': 'Bernoulli', 'p': p}
        final_results = pd.DataFrame.from_records([final_results], index='guessed_dist')
    else:
        final_results = {'uniform': tests.chsq_uniform(df), 'geometric': tests.chsq_geom(df),
                         'exponential': tests.chsq_expon(df), 'weibull': tests.chsq_weibull(df),
                         'normal': tests.chsq_normal(df), 'gamma': tests.chsq_gamma(df)}
        final_results = pd.DataFrame(final_results).transpose()
        final_results_tmp = final_results
        #dropping -ve chi-square
        final_results = final_results.query('chisquare>0')

        #limiting to the min chisquare
        final_results = final_results.query('chisquare==chisquare.min()')

        #selecting min p_value in case of tie


        #final_results.dropna(axis=1, inplace = True)

        print(final_results)
    return final_results
