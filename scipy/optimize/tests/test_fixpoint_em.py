"""
Test fixpoint solvers using expectation maximization problems.

"""
from __future__ import division, absolute_import, print_function

from functools import partial

import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose, TestCase

from scipy.optimize import fixpoint, basinhopping, minimize
from scipy.stats import poisson, norm
from scipy.special import expit, logit


def clo(w):
    "Closure in the jargon of compositional analysis."
    return w / w.sum()


class alr:
    "Additive log ratio transform."
    @classmethod
    def encode(cls, p):
        return np.log(p[:-1]) - np.log(p[-1])

    @classmethod
    def decode(cls, v):
        return clo(np.concatenate((np.exp(v), [1])))


class PoissonMixUnboundedTransform:
    "Unbounded parameterization of a Poisson mixture model."

    @classmethod
    def encode(cls, p, mu):
        return np.concatenate((alr.encode(p), np.log(mu)))

    @classmethod
    def decode(cls, X):
        X = np.asarray(X)
        n = X.shape[0]
        v, log_mu = X[:n//2], X[n//2:]
        p = alr.decode(v)
        mu = np.exp(log_mu)
        return p, mu


class PoissonMixSimpleTransform:
    "Simple parameterization of a Poisson mixture model."

    @classmethod
    def encode(cls, p, mu):
        ptrunc = p[:-1]
        return np.concatenate((ptrunc, mu))

    @classmethod
    def decode(cls, X):
        X = np.asarray(X)
        n = X.shape[0]
        ptrunc, mu = X[:n//2], X[n//2:]
        p = np.concatenate((ptrunc, [1-ptrunc.sum()]))
        return p, mu


class PoissonMix:
    "Poisson mixture model."

    @classmethod
    def neg_log_likelihood(cls, data, transform, X):
        p, mu = transform.decode(X)
        n = data.shape[0]
        counts = np.arange(n)
        likelihoods = poisson.pmf(counts[:, np.newaxis], mu).dot(p)
        if not np.all(likelihoods):
            return np.inf
        return -data.dot(np.log(likelihoods))

    @classmethod
    def em(cls, data, transform, X):
        p, mu = transform.decode(X)
        n = data.shape[0]
        counts = np.arange(n)
        unnormalized_pi = p * poisson.pmf(counts[:, np.newaxis], mu)
        pi = unnormalized_pi / unnormalized_pi.sum(axis=1)[:, np.newaxis]
        p = data.dot(pi) / data.sum()
        mu = (data * counts).dot(pi) / data.dot(pi)
        return transform.encode(p, mu)


class CheckPoissonMix(object):
    "Help test EM acceleration of Poisson mixture model parameter inference."

    def _check_solution(self, p, mu):
        # Make some attempt to correctly match the components.
        ind = np.argsort(p)
        des_ind = np.argsort(self.desired_p)
        assert_allclose(p[ind], self.desired_p[des_ind], atol=1e-3)
        assert_allclose(mu[ind], self.desired_mu[des_ind], atol=1e-3)

    def test_plain_em(self):
        transform = PoissonMixSimpleTransform
        X0 = transform.encode(self.initial_p, self.initial_mu)
        iterations = 0
        prev = X0
        xtol = 1e-8
        while True:
            X = PoissonMix.em(self.data, transform, prev)
            iterations += 1
            if np.linalg.norm(X - prev) < xtol * np.linalg.norm(prev):
                break
            prev = X
        self._check_solution(*transform.decode(X))
        assert_allclose(iterations, self.desired_plain_em_iterations, atol=5)

    def test_local_squarem(self):
        transform = PoissonMixSimpleTransform
        X0 = transform.encode(self.initial_p, self.initial_mu)
        f = partial(PoissonMix.em, self.data, transform)
        sol = fixpoint(f, X0, method='squarem')
        self._check_solution(*transform.decode(sol.fun))
        assert_allclose(sol.nit, self.desired_local_squarem_nit, atol=5)
        assert_allclose(sol.nfev, self.desired_local_squarem_nfev, atol=5)

    #def test_globalized_squarem(self):
        #raise Exception

    def test_direct_minimization(self):
        transform = PoissonMixUnboundedTransform
        X0 = transform.encode(self.initial_p, self.initial_mu)
        f = partial(PoissonMix.neg_log_likelihood, self.data, transform)
        low = -10
        high = 10
        bounds = [[low, high]] * len(X0)

        def check_bounds(**kwargs):
            x = kwargs['x_new']
            return bool(np.all(low < x) and np.all(x < high))

        result = basinhopping(f, X0, accept_test=check_bounds,
                minimizer_kwargs=dict(bounds=bounds))
        self._check_solution(*transform.decode(result.x))


class CheckVaradhanPoissonMix(CheckPoissonMix):
    # Ravi Varadhan and Christophe Roland.
    # Simple and Globally Convergent Methods for Accelerating
    # the Convergence of Any EM Algorithm.
    # Scandinavian Journal of Statistics, Vol 35: 335--353, 2008.
    #
    # This is a poisson mixture model, with three parameters:
    # a mixture proportion and two poisson rates.
    #
    # For the given data, the maximum likelihood estimates should be:
    # mixture proportion of the first poisson rate : 0.3599
    # the first poisson rate : 1.256
    # the second poisson rate : 2.663
    # Note that this is not distinguishable from the estimates
    # (1.0 - 0.3599, 2.663, 1.256).
    #
    # The EM strategy is to first conditionally distribute the blame
    # for each of the ten counts between the two Poisson processes
    # (this is like the expectation step),
    # and then to use these condtional distributions to re-estimate the
    # overall mixing proportion and the two Poisson rates.
    #
    data = np.array([162, 267, 271, 185, 111, 61, 27, 8, 3, 1], dtype=float)
    desired_p = np.array([0.3599, 0.6401])
    desired_mu = np.array([1.256, 2.663])


class TestVaradhanTable2(CheckVaradhanPoissonMix):
    initial_p = np.array([0.4, 0.6])
    initial_mu = np.array([1.0, 2.0])
    desired_plain_em_iterations = 2061
    desired_local_squarem_nit = 28
    desired_local_squarem_nfev = 108


class TestVaradhanFigure2(CheckVaradhanPoissonMix):
    initial_p = np.array([0.3, 0.7])
    initial_mu = np.array([1.0, 2.5])
    desired_plain_em_iterations = 2335
    desired_local_squarem_nit = 84
    desired_local_squarem_nfev = 674


class TestJamshidian1997Example1(CheckPoissonMix):
    # Acceleration of the EM Algorithm by Using Quasi-Newton Methods
    # Mortaza Jamshidian and Robert I. Jennrich
    # Journal of the Royal Statistical Society. Series B (Methodological)
    # Vol. 59, No. 3 (1997), pp. 569-587
    #
    # In example 1 the frequencies below have been sampled according to
    # mixture probability (of rate 1) : 0.3
    # rate 1 : 1.0
    # rate 2 : 1.5
    #
    data = np.array([552, 703, 454, 180, 84, 23, 4], dtype=float)
    initial_p = np.array([0.5, 0.5])
    initial_mu = np.array([1.0, 2.0])
    desired_p = np.array([0.4503, 0.5497])
    desired_mu = np.array([1.0255, 1.5486])
    desired_plain_em_iterations = 36774
    desired_local_squarem_nit = 113
    desired_local_squarem_nfev = 517
