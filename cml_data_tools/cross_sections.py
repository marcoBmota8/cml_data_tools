import numpy
import pandas
import scipy


def stack(xs_iter, channels):
    """Stacks cross sections into one large pandas SparseDataFrame.

    Arguments
    ---------
    xs_iter : Iterator[pandas.DataFrame]
    channels : pandas.Series
        A pandas series containing the names of all channels for each cross
        section. Any individual curve dataframe will probably have curves for
        only a small fraction of these channels, which is why this returns a
        SparseDataFrame.

    Returns
    -------
    pandas.SparseDataFrame
        A SparseDataFrame multi-indexed by (id, date), where `id` is the
        patient id of the source record and `date` is the sampling time for the
        values in the row.
    """
    # I would like to use pandas.concat to do this, which would have been much
    # cleaner, but there are two issues:

    # 1) It's very slow with a large number of dfs to concat, largely because
    # it is having to compute the new column index from the union of dfs.  This
    # problem can be avoided by passing the list of channels, which we do know
    # ahead of time, although some channels may be completely empty.

    # 2) It also chokes with 'AssertionError: invalid dtype determination in
    # get_concat_dtype' at
    # /site-packages/pandas/core/internals.py(5317)get_empty_dtype_and_na() I
    # have spent hours trying to track this down, to no avail. Therefore I'm
    # doing it my way, which is probably more efficient anyway, even after the
    # column index problem is avoided, because it incorporates the knowledge
    # that the final dataframe is very sparse.
    channel_ids = {name: i for i, name in enumerate(channels)}
    n_channels = len(channels)

    i = []
    j = []
    data = []
    ndx = []
    for df in xs_iter:
        n = len(df.index)
        if i:
            offset = i[-1] + 1
        else:
            offset = 0
        for channel in df:
            vals = df[channel].values
            channel_num = channel_ids[channel]
            i.extend(range(offset, offset + n))
            j.extend([channel_num] * n)
            data.extend(vals)
        ndx.extend(df.index.values)

    sparse_mat = scipy.sparse.coo_matrix((data, (i, j)),
                                         shape=(i[-1] + 1, n_channels),
                                         dtype=numpy.dtype('d'))
    sparse_df = pandas.SparseDataFrame(sparse_mat,
                                       default_fill_value=numpy.nan,
                                       columns=channels)
    sparse_df.index = pandas.MultiIndex.from_tuples(ndx, names=df.index.names)
    return sparse_df
