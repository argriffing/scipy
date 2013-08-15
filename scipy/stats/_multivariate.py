#
# Author: Joris Vankerschaver 2013
#
from __future__ import division, print_function, absolute_import

from scipy.misc import doccer
from functools import wraps
import numpy as np

__all__ = ['multivariate_normal']


_LOG_2PI = np.log(2 * np.pi)


def _process_parameters(dim, mean, cov):
    """
    Infer dimensionality from mean or covariance matrix, ensure that
    mean and covariance are full vector resp. matrix.

    """

    # Try to infer dimensionality
    if dim is None:
        if mean is None:
            if cov is None:
                dim = 1
            else:
                cov = np.asarray(cov, dtype=float)
                if cov.ndim < 2:
                    dim = 1
                else:
                    dim = cov.shape[0]
        else:
            mean = np.asarray(mean, dtype=float)
            dim = mean.size
    else:
        if not np.isscalar(dim):
            raise ValueError("Dimension of random variable must be a scalar.")

    # Check input sizes and return full arrays for mean and cov if necessary
    if mean is None:
        mean = np.zeros(dim)
    mean = np.asarray(mean, dtype=float)

    if cov is None:
        cov = 1.0
    cov = np.asarray(cov, dtype=float)

    if dim == 1:
        mean.shape = (1,)
        cov.shape = (1, 1)

    if mean.ndim != 1 or mean.shape[0] != dim:
        raise ValueError("Array 'mean' must be vector of length %d." % dim)
    if cov.ndim == 0:
        cov = cov * np.eye(dim)
    elif cov.ndim == 1:
        cov = np.diag(cov)
    else:
        if cov.shape != (dim, dim):
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                                 " but cov.ndim = %d" % cov.ndim)

    return dim, mean, cov


def _process_quantiles(x, dim):
    """
    Adjust quantiles array so that last axis labels the components of
    each data point.

    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 0:
        x = x[np.newaxis]
    elif x.ndim == 1:
        if dim == 1:
            x = x[:, np.newaxis]
        else:
            x = x[np.newaxis, :]

    return x


def process_arguments(f):
    """
    Process arguments passed to member functions of `multivariate_normal`.

    This function infers the dimensionality of the Gaussian from the mean
    or from the data, ensures that the mean and covariance are resp. a
    full vector and matrix, and that the data points are formatted
    as an ndarray whose last axis labels the components of the data points.

    """
    @wraps(f)
    def _f(self, x, mean=None, cov=1):
        dim, mean, cov = _process_parameters(None, mean, cov)
        x = _process_quantiles(x, dim)

        out = f(self, x, mean, cov).squeeze()
        if out.ndim == 0:
            out = out[()]
        return out

    return _f


def _pinv_1d(v, eps=1e-5):
    """
    A helper function for computing the pseudoinverse.

    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Elements of v smaller than eps are considered negligible.

    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.
        
    """
    return np.array([0 if abs(x) < eps else 1/x for x in v], dtype=float)


def _psd_pinv_decomposed_log_pdet(mat, eps=1e-5):
    """
    Compute a decomposition of the pseudo-inverse
    and the logarithm of the pseudo-determinant
    of a symmetric positive semi-definite matrix.

    The pseudo-determinant of a matrix is defined as the product of
    the non-zero eigenvalues, and coincides with the usual determinant
    for a full matrix. For reasons of efficiency, we (implicitly)
    assume that the matrix is symmetric positive semi-definite and use
    the non-zero singular values to compute the pseudo-determinant,
    rather than the eigenvalues.

    Parameters
    ----------
    mat : array_like
        Input array of shape (`m`, `n`)
    eps : float, optional
        Threshold below which a singular value is considered to be zero.

    Returns
    -------
    M : array_like
        The pseudo-inverse of the input matrix is np.dot(M, M.T).
    log_pdet : float
        Logarithm of the pseudo-determinant of the matrix.

    Notes
    -----
    The expression for the pseudo-determinant in terms of singular values
    rather than eigenvalues is only valid for matrices that are
    symmetric positive semi-definite, but we do not check this.

    """
    u, s, vt = np.linalg.svd(mat)
    s_pinv = _pinv_1d(s, eps)
    U = np.multiply(u, np.sqrt(s_pinv))
    log_pdet = np.sum(np.log(s[s > eps]))
    return U, log_pdet


_doc_default_callparams = \
"""Parameters
----------
x : array_like
    Quantiles, with the last axis of `x` denoting the components.
mean : array_like, optional
    Mean of the distribution (default zero)
cov : array_like, optional
    Covariance matrix of the distribution (default one)
"""

_doc_callparams_note = \
"""Setting the parameter `mean` to `None` is equivalent to having `mean`
be the zero-vector. The parameter `cov` can be a scalar, in which case
the covariance matrix is the identity times that value, a vector of
diagonal entries for the covariance matrix, or a two-dimensional
array_like.
"""

_doc_frozen_callparams = \
"""Parameters
----------
x : array_like
    Quantiles, with the last axis of `x` denoting the components.
"""

_doc_frozen_callparams_note = \
"""See class definition for a detailed description of `x`."""

docdict_params = {
    '_doc_default_callparams': _doc_default_callparams,
    '_doc_callparams_note': _doc_callparams_note
}

docdict_noparams = {
    '_doc_default_callparams': _doc_frozen_callparams,
    '_doc_callparams_note': _doc_frozen_callparams_note
}


class multivariate_normal_gen(object):
    r"""
    A multivariate normal random variable.

    The `mean` keyword specifies the mean. The `cov` keyword specifies the
    covariance matrix.

    .. versionadded:: 0.13.0

    Methods
    -------
    pdf(x, mean=None, cov=1)
        Probability density function.
    logpdf(x, mean=None, cov=1)
        Log of the probability density function.

    %(_doc_default_callparams)s

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" multivariate normal
    random variable:

    rv = multivariate_normal(mean=None, scale=1)
        - Frozen  object with the same methods but holding the given
          mean and covariance fixed.

    Notes
    -----
    %(_doc_callparams_note)s

    The covariance matrix `cov` must be a (symmetric) positive
    semi-definite matrix, but `multivariate_normal` will not check for
    this explicitly. The determinant and inverse of `cov` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `cov` does not need to have full rank.

    The probability density function for `multivariate_normal` is

    .. math::

        f(x) = \frac{1}{\sqrt{(2 \pi)^k \det \Sigma}} \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right),

    where :math:`\mu` is the mean, :math:`\Sigma` the covariance matrix,
    and :math:`k` is the dimension of the space where :math:`x` takes values.

    Examples
    --------
    >>> from scipy.stats import multivariate_normal
    >>> x = np.linspace(0, 5, 10, endpoint=False)
    >>> y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); y
    array([ 0.00108914,  0.01033349,  0.05946514,  0.20755375,  0.43939129,
            0.56418958,  0.43939129,  0.20755375,  0.05946514,  0.01033349])
    >>> plt.plot(x, y)

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.  This allows us for instance to
    display the frozen pdf for a non-isotropic random variable in 2D as
    follows:

    >>> x, y = np.mgrid[-1:1:.01, -1:1:.01]
    >>> pos = np.empty(x.shape + (2,))
    >>> pos[:, :, 0] = x; pos[:, :, 1] = y
    >>> rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    >>> plt.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self):
        self.__doc__ = doccer.docformat(self.__doc__, docdict_params)

    def __call__(self, *args, **kwargs):
        """
        Create a frozen multivariate normal distribution.

        See `multivariate_normal_frozen` for more information.

        """

        return multivariate_normal_frozen(*args, **kwargs)

    def _logpdf(self, x, mean, prec_U, log_det_cov):
        """
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function
        mean : array_like
            Mean of the distribution
        prec_U : array_like
            A decomposition such that np.dot(prec_U, prec_U.T)
            is the precision matrix, i.e. inverse of the covariance matrix.
        log_det_cov : float
            Logarithm of the determinant of the covariance matrix

        """
        dim = x.shape[-1]
        dev = x - mean
        maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
        return -0.5 * (dim * _LOG_2PI + log_det_cov + maha)

    @process_arguments
    def logpdf(self, x, mean, cov):
        """
        Log of the multivariate normal probability density function.

        %(_doc_default_callparams)s

        Notes
        -----
        %(_doc_callparams_note)s

        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `x`

        """
        prec_U, log_det_cov = _psd_pinv_decomposed_log_pdet(cov)
        return self._logpdf(x, mean, prec_U, log_det_cov)

    @process_arguments
    def pdf(self, x, mean, cov):
        """
        Multivariate normal probability density function.

        %(_doc_default_callparams)s

        Notes
        -----
        %(_doc_callparams_note)s

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        """
        prec_U, log_det_cov = _psd_pinv_decomposed_log_pdet(cov)
        return np.exp(self._logpdf(x, mean, prec_U, log_det_cov))

multivariate_normal = multivariate_normal_gen()


class multivariate_normal_frozen(object):
    def __init__(self, mean=None, cov=1):
        """
        Create a frozen multivariate normal distribution.

        Parameters
        ----------
        mean : array_like, optional
            Mean of the distribution (default zero)
        cov : array_like, optional
            Covariance matrix of the distribution (default one)

        Examples
        --------
        When called with the default parameters, this will create a 1D random 
        variable with mean 0 and covariance 1:

        >>> from scipy.stats import multivariate_normal
        >>> r = multivariate_normal()
        >>> r.mean
        array([ 0.])
        >>> r.cov
        array([[1.]])

        """
        self.dim, self.mean, self.cov = _process_parameters(None, mean, cov)
        self.prec_U, self._log_det_cov = _psd_pinv_decomposed_log_pdet(self.cov)

    def logpdf(self, x):
        # TODO: the output processing below can be made into a generator
        # just as for the multivariate_normal class.
        x = _process_quantiles(x, self.dim)
        out = multivariate_normal._logpdf(x, self.mean, self.prec_U,
                                          self._log_det_cov).squeeze()
        if out.ndim == 0:
            out = out[()]
        return out

    def pdf(self, x):
        return np.exp(self.logpdf(x))


# Set frozen generator docstrings from corresponding docstrings in
# multivariate_normal_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf']:
    method = multivariate_normal_gen.__dict__[name]
    method_frozen = multivariate_normal_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(method.__doc__, docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, docdict_params)
