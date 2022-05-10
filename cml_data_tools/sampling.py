import numpy as np


def binomial_sample_curves(curves, density, rng=None):
    """Produces a sampling from the curves. The number of samples is drawn from
    a binomial distribution parametrized by the number of points in the curve
    and the float `density`, for which a typical value might be `1/365`
    (semantically, once per patient-year).

    Arguments
    ---------
    curves : pandas.DataFrame
        A dataframe of curve points to sample from. Index is dates, columns are
        data channels.  The number of curve points is the `n` parameter of the
        binomial distribution.
    density : float
        The `p` parameter of the binomial distribution.
    rng : numpy.random._generator.Generator
        A numpy random number generator or None. If None,
        `numpy.random.default_rng()` is used.

    Returns
    -------
    DataFrame or None. If the number of samples drawn from the distribution is
    zero, then this function returns None, otherwise it returns a sampling from
    the curves DataFrame.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = rng.binomial(len(curves.index), density)
    if n > 0:
        return curves.sample(n=n)


def dual_sample_curves(curves, density, rng=None):
    """Produces a stream of paired samples with datetime deltas.

    Arguments
    ---------
    curves : pandas.DataFrame
    curves : pandas.DataFrame
        A dataframe of curve points to sample from. Index is dates, columns are
        data channels.  The number of curve points is the `n` parameter of the
        binomial distribution.
    density : float
        The `p` parameter of the binomial distribution.

    Returns
    -------
    A triplet (t0, t1, dt) or None. If the number of samples to be drawn is
    zero then None is returned. Else `t0` and `t1` have the same number of
    samples drawn from the curves dataframe, and `dt` is the time deltas
    between the pairs of samples in `t0` and `t1`.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = rng.binomial(len(curves.index), density)
    if n > 0:
        t0 = df.sample(n=n)
        t1 = df.sample(n=n)
        t0_dates = t0.index.get_level_values(1)
        t1_dates = t1.index.get_level_values(1)
        dt = t0_dates - t1_dates
        return (t0, t1, dt)
