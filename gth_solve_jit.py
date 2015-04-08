"""
Filename: gth_solve_jit.py

Author: Daisuke Oyama

Routine to compute the stationary distribution of an irreducible Markov
chain by the Grassmann-Taksar-Heyman (GTH) algorithm.

"""
import numpy as np
from numba import jit

try:
    xrange
except:  # python3
    xrange = range


def gth_solve(A, overwrite=False):
    r"""
    This routine computes the stationary distribution of an irreducible
    Markov transition matrix (stochastic matrix) or transition rate
    matrix (generator matrix) `A`.

    More generally, given a Metzler matrix (square matrix whose
    off-diagonal entries are all nonnegative) `A`, this routine solves
    for a nonzero solution `x` to `x (A - D) = 0`, where `D` is the
    diagonal matrix for which the rows of `A - D` sum to zero (i.e.,
    :math:`D_{ii} = \sum_j A_{ij}` for all :math:`i`). One (and only
    one, up to normalization) nonzero solution exists corresponding to
    each reccurent class of `A`, and in particular, if `A` is
    irreducible, there is a unique solution; when there are more than
    one solution, the routine returns the solution that contains in its
    support the first index `i` such that no path connects `i` to any
    index larger than `i`. The solution is normalized so that its 1-norm
    equals one. This routine implements the Grassmann-Taksar-Heyman
    (GTH) algorithm [1]_, a numerically stable variant of Gaussian
    elimination, where only the off-diagonal entries of `A` are used as
    the input data. For a nice exposition of the algorithm, see Stewart
    [2]_, Chapter 10.

    Parameters
    ----------
    A : array_like(float, ndim=2)
        Stochastic matrix or generator matrix. Must be of shape n x n.
    overwrite : bool, optional(default=False)
        Whether to overwrite `A`; may improve performance.

    Returns
    -------
    x : numpy.ndarray(float, ndim=1)
        Stationary distribution of `A`.

    References
    ----------
    .. [1] W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative
       Analysis and Steady State Distributions for Markov Chains,"
       Operations Research (1985), 1107-1116.

    .. [2] W. J. Stewart, Probability, Markov Chains, Queues, and
       Simulation, Princeton University Press, 2009.

    """
    A1 = np.array(A, dtype=float, copy=not overwrite)

    if len(A1.shape) != 2 or A1.shape[0] != A1.shape[1]:
        raise ValueError('matrix must be square')

    n = A1.shape[0]

    x = np.zeros(n)

    # === Reduction === #
    for i in xrange(n-1):
        scale = np.sum(A1[i, i+1:n])
        if scale <= 0:
            # There is one (and only one) recurrent class contained in
            # {0, ..., i};
            # compute the solution associated with that recurrent class.
            n = i+1
            break
        A1[i+1:n, i] /= scale

        A1[i+1:n, i+1:n] += np.dot(A1[i+1:n, i:i+1], A1[i:i+1, i+1:n])

    # === Backward substitution === #
    x[n-1] = 1
    for i in xrange(n-2, -1, -1):
        x[i] = np.dot(x[i+1:n], A1[i+1:n, i])

    # === Normalization === #
    x /= np.sum(x)

    return x


@jit
def gth_solve_jit(A):
    """
    JIT-compiled version of `gth_solve`.

    """
    #A1 = np.array(A, dtype=np.float64)
    A1 = np.array(A, dtype=float)

    if len(A1.shape) != 2 or A1.shape[0] != A1.shape[1]:
        # raise ValueError('matrix must be square')  # Not supported
        raise ValueError

    n = A1.shape[0]

    #x = np.zeros(n, dtype=np.float64)
    x = np.zeros(n)

    # === Reduction === #
    SCALE_NONZERO = 1
    for k in range(n-1):
        if SCALE_NONZERO == 1:
            scale = np.sum(A1[k, k+1:n])
            if scale <= 0:
                # There is one (and only one) recurrent class contained in
                # {0, ..., k};
                # compute the solution associated with that recurrent class.
                n = k+1
                # break
                # `break` statement is not supported for auto jitting
                # see github.com/numba/numba/pull/819
                # so need to use the flag `SCALE_NONZERO`
                SCALE_NONZERO = 0
            for i in range(k+1, n):
                A1[i, k] /= scale

                for j in range(k+1, n):
                    A1[i, j] += A1[i, k] * A1[k, j]

    # === Backward substitution === #
    x[n-1] = 1
    for k in range(n-2, -1, -1):
        for i in range(k+1, n):
            x[k] += x[i] * A1[i, k]

    # === Normalization === #
    norm = np.sum(x)
    for k in range(n):
        x[k] /= norm

    return x
