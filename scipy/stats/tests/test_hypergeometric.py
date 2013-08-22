from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.stats
from scipy.stats import _cpa_hypergeom
from scipy.misc import comb

from numpy.testing import (run_module_suite, assert_equal, assert_allclose,
        assert_)

def test_3f2z1():
    # According to HypergeometricPFQ[{1,2,3},{4,5},1]
    # this is -12*(pi*pi - 10).
    a1, a2, a3, b1, b2 = 1.0, 2.0, 3.0, 4.0, 5.0
    actual, extra = _cpa_hypergeom.hyper3F2Z1(a1, a2, a3, b1, b2)
    desired = -12*(np.pi * np.pi - 10)
    assert_allclose(actual, desired)

def test_regularized_3f2z1():
    # According to HypergeometricPFQregularized[{1,2,3},{4,5},1]
    # this is (10 - pi*pi)/12.
    # The regularization is not the same.
    a1, a2, a3, b1, b2 = 1.0, 2.0, 3.0, 4.0, 5.0
    actual, extra = _cpa_hypergeom.hyper3F2regularizedZ1(a1, a2, a3, b1, b2)
    desired = (10 - np.pi * np.pi) / 12
    assert_(not np.allclose(actual, desired))

def test_hypergeometric_survival():
    # Check the cumulative distribution computed in a couple of ways.

    # Define the data and parameters according to scipy notation.
    scipy_k = 30
    scipy_M = 13397950
    scipy_n = 4363
    scipy_N = 12390

    # Use the scipy cumulative distribution function.
    scipy_survival = scipy.stats.hypergeom.sf(
            scipy_k, scipy_M, scipy_n, scipy_N)

    print(scipy_survival)

    # Redefine the data and parameters according to Wikipedia notation.
    N = scipy_M
    K = scipy_n
    n = scipy_N
    k = scipy_k

    # Define the arguments of the 3F2 functions.
    a1 = 1
    a2 = k + 1 - K
    a3 = k + 1 - n
    b1 = k + 2
    b2 = N + k + 2 - K - n

    # Try the explicit Wikipedia formula,
    # using the regularized hyper3f2 function.
    hyper_reg, extra = _cpa_hypergeom.hyper3F2regularizedZ1(a1, a2, a3, b1, b2)
    wiki_survival_reg = hyper_reg

    print(wiki_survival_reg)

    # Use the explicit formula from Wikipedia without tricks.
    scale = (comb(n, k+1) * comb(N-n, K-k-1)) / comb(N, K)
    a1 = 1
    a2 = k + 1 - K
    a3 = k + 1 - n
    b1 = k + 2
    b2 = N + k + 2 - K - n
    hyper, extra = _cpa_hypergeom.hyper3F2regularizedZ1(a1, a2, a3, b1, b2)
    wiki_survival = scale * hyper

    # Check that the two calculations give the same value.
    assert_allclose(wiki_survival, scipy_survival)


if __name__ == '__main__':
    run_module_suite()

