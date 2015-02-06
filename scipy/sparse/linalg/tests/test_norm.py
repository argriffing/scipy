"""Test functions for the sparse.linalg.norm module
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (
        assert_raises, assert_allclose, assert_equal, assert_,
        decorators, TestCase, run_module_suite)

from scipy._lib._version import NumpyVersion
import scipy.sparse

from scipy.sparse.linalg import norm as spnorm
from numpy.linalg import norm as npnorm


class TestNorm(TestCase):
    def test_norm(self):
        a = np.arange(9) - 4
        b = a.reshape((3, 3))
        b = scipy.sparse.csr_matrix(b)

        #Frobenius norm is the default
        assert_equal(spnorm(b), 7.745966692414834)        
        assert_equal(spnorm(b, 'fro'), 7.745966692414834)

        assert_equal(spnorm(b, np.inf), 9)
        assert_equal(spnorm(b, -np.inf), 2)
        assert_equal(spnorm(b, 1), 7)
        assert_equal(spnorm(b, -1), 6)

        #_multi_svd_norm is not implemented for sparse matrix
        assert_raises(NotImplementedError, spnorm, b, 2)
        assert_raises(NotImplementedError, spnorm, b, -2)
                
    def test_norm_axis(self):
        a = np.array([[ 1, 2, 3],
                      [-1, 1, 4]])

        c = scipy.sparse.csr_matrix(a)        
        #Frobenius norm
        assert_equal(
                spnorm(c, axis=0),
                np.sqrt(np.power(np.asmatrix(a), 2).sum(axis=0)))
        assert_equal(
                spnorm(c, axis=1),
                np.sqrt(np.power(np.asmatrix(a), 2).sum(axis=1)))

        assert_equal(
                spnorm(c, np.inf, axis=0),
                max(np.absolute(np.asmatrix(a)).sum(axis=0)))
        assert_equal(
                spnorm(c, np.inf, axis=1),
                max(np.absolute(np.asmatrix(a)).sum(axis=1)))

        assert_equal(
                spnorm(c, -np.inf, axis=0),
                min(np.absolute(np.asmatrix(a)).sum(axis=0)))
        assert_equal(
                spnorm(c, -np.inf, axis=1),
                min(np.absolute(np.asmatrix(a)).sum(axis=1)))
                        
        assert_equal(
                spnorm(c, 1, axis=0),
                np.absolute(np.asmatrix(a)).sum(axis=0))
        assert_equal(
                spnorm(c, 1, axis=1),
                np.absolute(np.asmatrix(a)).sum(axis=1))
        
        assert_equal(
                spnorm(c, -1, axis=0),
                min(np.absolute(np.asmatrix(a)).sum(axis=0))  )
        assert_equal(
                spnorm(c, -1, axis=1),
                min(np.absolute(np.asmatrix(a)).sum(axis=1))  )
        
        #_multi_svd_norm is not implemented for sparse matrix
        assert_raises(NotImplementedError, spnorm, c, 2, 0)
        assert_raises(NotImplementedError, spnorm, c, -2, 0)




class TestSparseVsDenseNorms(TestCase):
    _sparse_types = (
            scipy.sparse.bsr_matrix,
            scipy.sparse.coo_matrix,
            scipy.sparse.csc_matrix,
            scipy.sparse.csr_matrix,
            scipy.sparse.dia_matrix,
            scipy.sparse.dok_matrix,
            scipy.sparse.lil_matrix,
            )
    _test_matrices = (
            (np.arange(9) - 4).reshape((3, 3)),
            [
                [ 1, 2, 3],
                [-1, 1, 4]],
            [
                [ 1, 0, 3],
                [-1, 1, 4j]],
            )
    def test_sparse_matrix_norms(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                assert_allclose(spnorm(S), npnorm(M))
                assert_allclose(spnorm(S, 'fro'), npnorm(M, 'fro'))
                assert_allclose(spnorm(S, np.inf), npnorm(M, np.inf))
                assert_allclose(spnorm(S, -np.inf), npnorm(M, -np.inf))
                assert_allclose(spnorm(S, 1), npnorm(M, 1))
                assert_allclose(spnorm(S, -1), npnorm(M, -1))

    @decorators.skipif(NumpyVersion(np.__version__) < '1.8.0')
    def test_sparse_matrix_norms_scalar_axis(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                for axis in 0, 1:
                    wat = spnorm(S, -np.inf, axis=axis)
                    print(wat)
                    assert_allclose(
                            spnorm(S, axis=axis),
                            npnorm(M, axis=axis, keepdims=True))
                    assert_allclose(
                            spnorm(S, np.inf, axis=axis),
                            npnorm(M, np.inf, axis=axis, keepdims=True))
                    assert_allclose(
                            spnorm(S, -np.inf, axis=axis),
                            npnorm(M, -np.inf, axis=axis, keepdims=True))
                    assert_allclose(
                            spnorm(S, 1, axis=axis),
                            npnorm(M, 1, axis=axis, keepdims=True))
                    assert_allclose(
                            spnorm(S, -1, axis=axis),
                            npnorm(M, -1, axis=axis, keepdims=True))
