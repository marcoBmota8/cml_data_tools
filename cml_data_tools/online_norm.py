"""
Parallel calculation of curve statistics (mean and variance).

Usage:
>>> # assume we have a list of curveset dataframes
>>> stats = None
>>> for df in curves:
...     stats = update(stats, to_stats(df))
>>> to_dataframe(stats)
                                             mean            var        n
mode           channel
ANA            ANA-titer                56.045114   17110.853668   333375
Age            age                      37.929960     609.835687  2275344
Event          event                     0.042143       0.040367  2279196
Medications    abilify                   0.482493       0.249694     9882
...                                           ...            ...      ...
Top Lab Values UricA                     6.089179       6.685858   389326
               pO2/FI                  223.330774   26433.165234   214713
[4295 rows x 3 columns]
"""
import collections

import numpy
import pandas


NormStats = collections.namedtuple('NormStats', 'channels mean var n')


def to_dataframe(stats):
    """Helper function, converts NormStats objects to dataframes"""
    data = stats._asdict()
    index = data.pop('channels')
    if isinstance(index[0], tuple):
        index = pandas.MultiIndex.from_tuples(index)
    df = pandas.DataFrame(data=data, index=index)
    return df


def to_stats(curveset):
    """Helper function, converts dataframe of curves to stats on the curves"""
    # .values is faster than .to_numpy(), although the effect is very small for
    # the dataframe itself (it is mostly present in converting the columns)
    C = curveset.columns.values
    x = curveset.values
    M = numpy.nanmean(x, axis=0)
    V = numpy.nanvar(x, axis=0)
    N = numpy.full(M.shape, len(x), dtype=numpy.int64)
    return NormStats(C, M, V, N)


def update(prev, curr):
    """Updates a stats dataframe mean / variance with new values.

    Labels that are common to both `prev` and `curr` are updated using a
    variant of the parallel version of Welford's algorithm, labels that are in
    one dataframe but not the other are passed through unchanged.

    If the first argument is None, returns the second argument. This is a
    convenience to make a common coding pattern easier.  With this behavior,
    code that looks like:

    >>> stats = None
    >>> for df in stuff:
    ...     new_stats = to_stats(df)
    ...     if stats is None:
    ...         stats = new_stats
    ...     else:
    ...         stats = update(stats, new_stats)

    Becomes

    >>> stats = None
    >>> for df in stuff:
    ...     stats = update(stats, to_stats(df))
    """
    if prev is None:
        return curr

    # This code isn't particularly pretty but it is very, very fast
    # Indexing 1d arrays with index arrays (and masks where necessary) is a
    # good clip faster than doing anything with multi-dimensional arrays and
    # fancier indexing / slicing techniques.
    #
    # The two slowest operations here are intersect1d and argsort.
    prev_channels, prev_m, prev_v, prev_n = prev
    curr_channels, curr_m, curr_v, curr_n = curr

    C, prev_idx, curr_idx = numpy.intersect1d(prev_channels,
                                              curr_channels,
                                              assume_unique=True,
                                              return_indices=True)
    # Calculate the updated values
    cm = curr_m[curr_idx]
    cv = curr_v[curr_idx]
    cn = curr_n[curr_idx]

    pm = prev_m[prev_idx]
    pv = prev_v[prev_idx]
    pn = prev_n[prev_idx]

    N = pn + cn

    cf = cn / N
    pf = pn / N
    dx = cm - pm

    M = pm + (cf * dx)
    V = (pv * pf) + (cv * cf) + (pf * cf * dx * dx)

    # Mask out prev values not in current (i.e. unchanged vals)
    prev_mask = numpy.ones_like(prev_channels, dtype=numpy.bool)
    prev_mask[prev_idx] = False

    C0 = prev_channels[prev_mask]
    M0 = prev_m[prev_mask]
    V0 = prev_v[prev_mask]
    N0 = prev_n[prev_mask]

    # Mask out curr values not in prev (i.e. brand new vals)
    curr_mask = numpy.ones_like(curr_channels, dtype=numpy.bool)
    curr_mask[curr_idx] = False

    C1 = curr_channels[curr_mask]
    M1 = curr_m[curr_mask]
    V1 = curr_v[curr_mask]
    N1 = curr_n[curr_mask]

    # Recombine the unchanged, updated, and new values
    C = numpy.concatenate((C, C0, C1))
    M = numpy.concatenate((M, M0, M1))
    V = numpy.concatenate((V, V0, V1))
    N = numpy.concatenate((N, N0, N1))

    sort = numpy.argsort(C)
    C = C[sort]
    M = M[sort]
    V = V[sort]
    N = N[sort]

    return NormStats(C, M, V, N)
