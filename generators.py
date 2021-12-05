import numpy as np
from scipy.stats import bernoulli, weibull_min
#

class Generator:
    """
    main class that generates all the dists
    """

    def __init__(self, no_samples):
        """Constructor for generator"""
        self.no_samples = no_samples

    def normal(self, mu, sigma):
        """
        Normal dist generator
        ....

        inputs
        -------
        mu    : mean
        sigma : standard deviation
        """
        data = np.random.normal(mu, sigma, size=self.no_samples)
        return data

    def bern(self, p):
        """
        Bernoulli dist generator
        ....

        inputs
        -------
        p   : probability
        """
        data = bernoulli.rvs(p, size=self.no_samples)
        return data

    def geometric(self, p):
        """
        Geometric dist generator
        ....

        inputs
        -------
        p   : probability
        """
        data = np.random.geometric(p, size=self.no_samples)
        return data

    def exponential(self, rate):
        """
        Exponential dist generator
        ....

        inputs
        -------
        rate   : lambda
        """
        data = np.random.exponential(1/rate, size=self.no_samples)
        return data

    def gamma(self, shape, scale):
        """
        Gamma dist generator
        ....

        inputs
        -------
        shape   : shape or form parameter
        scale   : scale parameter
        """
        data = np.random.gamma(shape, scale, size=self.no_samples)
        return data

    def weibull(self, shape, scale):
        """
        Weibull dist generator
        ....

        inputs
        -------
        shape   : shape or form parameter
        scale   : scale parameter
        """
        data = weibull_min.rvs(shape, scale, size=self.no_samples)
        return data
