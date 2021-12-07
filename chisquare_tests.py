from scipy.stats import chisquare, norm, gamma
from scipy.special import digamma, polygamma
import pandas as pd
import numpy as np


def chsq_normal(df):
    """
    Chi-square for normal dist.
    """
    test_result = {'chisquare': 999,
                   'p_value': 0,
                   'reject_H0': True,
                   'guessed_dist': 'Failed',
                   'p_hat': 0}
    try:
        n = len(df)
        mu = np.mean(df)
        std = np.std(df)
        folds = np.array([.2, .4, .6, .8])
        intervals = mu + norm.ppf(folds) * std
        intervals = np.sort(np.hstack([min(df), intervals, max(df)]))
        expected_freq = np.full(5, n / 5)
        observed_freq = pd.cut(df, bins=intervals).value_counts()
        diff = np.abs(observed_freq.sum() - expected_freq.sum())
        observed_freq[-1] = observed_freq[-1] + diff
        chisq = chisquare(observed_freq, expected_freq)
        test_result = {'chisquare': chisq.statistic,
                       'p_value': chisq.pvalue,
                       'reject_H0': chisq.pvalue < 0.05,
                       'guessed_dist': 'Normal',
                       'mu': mu,
                       'std': std}
    except:
        pass
    return test_result


def chsq_uniform(df):
    """
    Chi-Square estimator for uniform
    - data is divided into 5 folds
    ....

    inputs
    -------
    df   : dataset
    """
    test_result = {'chisquare': 999,
                   'p_value': 0,
                   'reject_H0': True,
                   'guessed_dist': 'Failed',
                   'a': 999,
                   'b': 999}
    try:
        binned_data = pd.cut(df, bins=5).value_counts()
        chsq = chisquare(binned_data)

        test_result = {'chisquare': chsq.statistic,
                       'p_value': chsq.pvalue,
                       'reject_H0': chsq.pvalue < 0.05,
                       'guessed_dist': 'Uniform',
                       'a': df.min(),
                       'b': df.max()}
    except:
        pass
    return test_result


def chsq_geom(df):
    """
    Chi-Square estimator for geometric
    - data is divided into 5 folds
    ....

    inputs
    -------
    df   : dataset
    """
    try:
        n = len(df)
        observed_freq = np.unique(df, return_counts=True)[1]
        categories = np.append(0, np.unique(df, return_counts=True)[0])
        binned_data = pd.cut(df, bins=categories).value_counts()
        denom = (categories[1:] * observed_freq).sum()
        p_hat = n / denom
        probability = (1 - p_hat) ** (categories[1:] - 1) * p_hat

        # combining results to 1
        probability[-1] = 1 - probability[:-1].sum()

        expected_frqs = n * probability
        chisq = chisquare(observed_freq, expected_frqs)
        test_result = {'chisquare': chisq.statistic,
                       'p_value': chisq.pvalue,
                       'reject_H0': chisq.pvalue < 0.05,
                       'guessed_dist': 'Geometric',
                       'p_hat': p_hat}

        if np.isnan(test_result['chisquare']):
            test_result = {'chisquare': 999,
                           'p_value': 0,
                           'reject_H0': True,
                           'guessed_dist': 'Failed',
                           'p_hat': 0}
        return test_result
    except:
        test_result = {'chisquare': 999,
                       'p_value': 0,
                       'reject_H0': True,
                       'guessed_dist': 'Failed',
                       'p_hat': 0}

        return test_result


def chsq_expon(df):
    """
    Chi-Square estimator for exponential
    - data is divided into 5 folds
    ....

    inputs
    -------
    df   : dataset
    """
    test_result = {'chisquare': 999,
                   'p_value': 0,
                   'reject_H0': True,
                   'guessed_dist': 'Failed',
                   'lambda': 0}
    try:
        n = len(df)
        x_bar = df.mean()
        lmbda = 1 / x_bar
        folds = np.array([1, 2, 3, 4])
        intervals = -x_bar * np.log(1 - folds / 5)
        intervals = np.sort(np.hstack([0, intervals, max(df)]))
        expected_freq = np.full(5, n / 5)
        observed_freq = pd.cut(df, bins=intervals).value_counts()
        diff = np.abs(observed_freq.sum() - expected_freq.sum())
        observed_freq[-1] = observed_freq[-1] + diff
        chisq = chisquare(observed_freq, expected_freq)
        test_result = {'chisquare': chisq.statistic,
                       'p_value': chisq.pvalue,
                       'reject_H0': chisq.pvalue < 0.05,
                       'guessed_dist': 'Exponential',
                       'lambda': lmbda}
    except:
        pass

    return test_result


def weibull_pars(df):
    """
    Helper function to calc. r for weibull
    ....

    inputs
    -------
    df   : dataset
    """
    counter = 0
    n = len(df)
    A = np.sum(np.log(df)) / n
    alpha_hat = (((6 / (np.pi ** 2)) * (np.sum(np.log(df) ** 2) - ((np.sum(np.log(df))) ** 2) / n)) / (n - 1)) ** -0.5
    B = np.sum(df ** alpha_hat)
    C = np.sum((df ** alpha_hat) * np.log(df))
    H = np.sum((df ** alpha_hat) * (np.log(df)) ** 2)
    delta = np.inf

    while delta > .0001:
        alpha_hat_new = alpha_hat + (A + (1 / alpha_hat) - (C / B)) / (
                (1 / (alpha_hat ** 2)) + (B * H - C ** 2) / (B ** 2))
        delta = np.abs(alpha_hat_new - alpha_hat)
        alpha_hat = alpha_hat_new

        B = np.sum(df ** alpha_hat)
        C = np.sum((df ** alpha_hat) * np.log(df))
        H = np.sum((df ** alpha_hat) * (np.log(df)) ** 2)

        counter += 1

        if counter > 500:
            print('Iteration is not converging')
            break
    beta_hat = (np.sum(df ** alpha_hat) / n) ** (1 / alpha_hat)
    return {"r_hat": alpha_hat, "lmbda_hat": 1 / beta_hat}


def gamma_pars(df):
    """
    Helper function to calc. gamma (k, theta)
    """
    counter = 0
    n = len(df)
    s = np.log((1 / n) * np.sum(df)) - (1 / n) * np.sum(np.log(df))
    k = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    delta = np.inf
    while delta > 0.0001:
        k_new = k - (np.log(k) - digamma(k) - s) / ((1 / k) - polygamma(1, k))
        delta = np.abs(k_new - k)
        if counter == 500:
            print('Iteration is not converging')
            break
        k = k_new
    scale = (1 / (k * n)) * np.sum(df)
    return {"shape": k, "scale": scale}


def chsq_weibull(df):
    """
        Chi-Square estimator for weibull
        - data is divided into 5 folds
        ....

        inputs
        -------
        df   : dataset
    """
    test_result = {'chisquare': 999,
                   'p_value': 0,
                   'reject_H0': True,
                   'guessed_dist': 'Failed',
                   'p_hat': 0}
    try:
        n = len(df)
        x_bar = df.mean()
        lmbda = 1 / x_bar
        r_hat = weibull_pars(df)['r_hat']
        lmbda_hat = weibull_pars(df)['lmbda_hat']
        folds = np.array([1, 2, 3, 4])
        intervals = (1 / lmbda_hat) * (-np.log(1 - folds / 5)) ** (1 / r_hat)
        intervals = np.hstack([0, intervals, max(df)])
        expected_freq = np.full(5, n / 5)
        observed_freq = pd.cut(df, bins=intervals).value_counts()
        chisq = chisquare(observed_freq, expected_freq)
        test_result = {'chisquare': chisq.statistic,
                       'p_value': chisq.pvalue,
                       'reject_H0': chisq.pvalue < 0.05,
                       'guessed_dist': 'Weibull',
                       'shape': lmbda,
                       'scale': r_hat}

        if np.isnan(test_result['chisquare']):
            test_result = {'chisquare': 999,
                           'p_value': 0,
                           'reject_H0': True,
                           'guessed_dist': 'Failed',
                           'p_hat': 0}
    except:
        pass
    return test_result


def chsq_gamma(df):
    """
    Function to calc. the gamma chi-square
    """
    test_result = {'chisquare': 999,
                   'p_value': 0,
                   'reject_H0': True,
                   'guessed_dist': 'Failed',
                   'p_hat': 0}
    try:
        n = len(df)
        shape = gamma_pars(df)['shape']
        scale = gamma_pars(df)['scale']
        folds = np.array([.2, .4, .6, .8])
        intervals = gamma.ppf(folds, a=shape, scale=scale)
        intervals = np.sort(np.hstack([min(df), intervals, max(df)]))
        expected_freq = np.full(5, n / 5)
        observed_freq = pd.cut(df, bins=intervals).value_counts()
        diff = np.abs(observed_freq.sum() - expected_freq.sum())
        observed_freq[-1] = observed_freq[-1] + diff
        chisq = chisquare(observed_freq, expected_freq)
        test_result = {'chisquare': chisq.statistic,
                       'p_value': chisq.pvalue,
                       'reject_H0': chisq.pvalue < 0.05,
                       'guessed_dist': 'Gamma',
                       'shape': shape,
                       'scale': scale}
    except:
        pass
    return test_result
