import collections
import functools
import inspect
import operator
import pathlib
import pickle
from concurrent.futures import as_completed, ProcessPoolExecutor, TimeoutError

import numpy as np
import pandas as pd

from cml_data_tools.curves import build_patient_curves
from cml_data_tools.clustering import (make_affinity_matrix, iter_clusters,
                                       AffinityPropagationClusterer)
from cml_data_tools.models import IcaPhenotypeModel
from cml_data_tools.pickle_cache import PickleCache
from cml_data_tools.plotting import plot_phenotypes_to_file
from cml_data_tools.source_ehr import (make_data_df, make_meta_df,
                                       aggregate_data, aggregate_meta)
from cml_data_tools.online_standardizer import (collect_curve_stats,
                                                update_curve_stats,
                                                OnlineCurveStandardizer)


def _drain_queue(q, timeout=None):
    try:
        for future in as_completed(q, timeout=timeout):
            yield future.result()
            q.remove(future)
    except TimeoutError:
        pass


def _worker(df, spec, resolution, calc_stats):
    curves = build_patient_curves(df, spec, resolution)
    if calc_stats:
        stats = collect_curve_stats(curves)
    else:
        stats = None
    return curves, stats


def _parallel_curve_gen(data, max_workers, spec, resolution, calc_stats):
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = set()
        for df in data:
            fut = pool.submit(_worker, df, spec, resolution, calc_stats)
            futures.add(fut)
            if len(futures) > 100:
                yield from _drain_queue(futures, timeout=1)
        # Block until all futures arrive
        yield from _drain_queue(futures)


def _sample_from_curves(curves, density):
    rng = np.random.default_rng()
    for df in curves:
        n = rng.binomial(len(df.index), density)
        if n > 0:
            samples = df.sample(n=n)
            yield samples


def _build_std_matrix(meta, cross_sections, standardizer, return_dense=True):
    meta = meta.set_index(['mode', 'channel'])
    # An error such as
    #   "TypeError: loop of ufunc does not support argument 0 of type"
    #   "float which has no callable log10 method"
    # Indicates that the series in the fill df have dtype "object,"
    # they need to have a float dtype.
    meta.fill = meta.fill.astype(np.float)
    # A KeyError during the standardization process indicates that we have
    # channels in the metadata not present in the actual patient data curves;
    # here we select only meta for channels which exist
    channels = list(standardizer.curve_stats.channels)
    channels = pd.MultiIndex.from_tuples(channels)
    meta = meta.loc[channels]
    # Use the actual channels from the patient data to reindex the cross
    # sections and then build the full, dense data matrix. The size of dense
    # should be [num cross sections, num channels across all patient data]
    dense = []
    for df in cross_sections:
        df = df.reindex(columns=channels)
        dense.append(df)
    dense = pd.concat(dense, copy=False)
    # Standardize the data matrix, then the fill values, then use the fill
    # values to remove all NaN from the standardized data matrix
    std_matrix = standardizer.transform(dense)
    fill = pd.DataFrame(meta.fill).transpose()
    fill, _ = fill.align(std_matrix, join='right', axis=1)
    fill = standardizer.transform(fill).loc['fill']
    std_matrix = std_matrix.fillna(fill)
    if return_dense:
        return dense, std_matrix
    return std_matrix


def _to_data_matrix(meta, cross_sections):
    channels = meta[['mode', 'channel']]
    channels = pd.MultiIndex.from_frame(channels)
    dense_df = pd.concat([df.reindex(columns=channels)
                          for df in cross_sections],
                         copy=False)
    dense_df = dense_df.dropna(axis='columns', how='all')
    return dense_df


def _to_std_data_matrix(meta, data_matrix, standardizer):
    # Standardize data matrix
    # NOTE: This step will drop channels that are constant or zero
    # (i.e. uninformative) from the data matrix (the steps below will
    # do the same to the metadata)
    std_data = standardizer.transform(data_matrix)
    # Meta has cols 'mode', 'channel', 'description', 'fill'
    # To do the align below we need 'mode' & 'channel' to be the index
    # (since these are the columns of the data matrix)
    meta = meta.set_index(['mode', 'channel'])
    # An error such as
    #   "TypeError: loop of ufunc does not support argument 0 of type"
    #   "float which has no callable log10 method"
    # Indicates that the series in the fill df have dtype "object,"
    # they need to have a float dtype.
    fill = pd.DataFrame(meta['fill']).transpose().astype(float)
    fill, _ = fill.align(std_data, join='right', axis=1)
    fill_series = standardizer.transform(fill).loc['fill']
    std_data = std_data.fillna(fill_series)
    return std_data


def _curve_expressions(curves, std, model, freq, agg):
    for df in curves:
        X = df.resample(freq, level='date').mean()
        X = standardizer.transform(X)
        X = model.transform(X)
        if agg is not None:
            X = X.agg(agg)
        # XXX: pickling loses the name attribute
        X.name = df.index.get_level_values('id')[0]
        yield X


def _expression_trajectories(expressions, freq, agg):
    for expr in expressions:
        X = expr.resample(freq, level='date').agg(agg)
        # XXX: pickling loses the name attribute
        X.name = expr.name
        yield df


def cached_operation(func):
    default_key = inspect.signature(func).parameters['key'].default
    @functools.wraps(func)
    def wrapper(self, *args, force=False, **kwargs):
        key = kwargs.get('key', default_key)
        if key not in self.cache or force:
            return func(self, *args, **kwargs)
    return wrapper


class Experiment:
    """
    An Experiment instance is a state machine which orchestrates the various
    phases and tasks involved in running an experiment. The experiment must be
    configured with a list (or other sortable container) of
    ``cml_data_tools.Config`` objects specifying how a data mode is transformed
    into floats and standardized prior to training any model. The experiment
    also requires a cache, which provides the working memory for the
    experiment. The default cache is a PickleCache instance pointed at the
    current working directory.

    Parameters
    ----------
    configs : Config[]
        A list or other sortable container of Config (or compatible) objects
    cache : PickleCache
        An instance of a PickleCache (or compatible) class

    Notes
    -----
    The methods which represent experiment operations follow some common
    patterns. Each returns None; these operations manipulate state (i.e. the
    cache) rather than return a value.

    Each has a keyword argument "key" which is used to set the results of the
    operation in the cache. Thus, if we wanted to compute two cross sections
    from the same set of curves, for example, we could call generate cross
    sections (via `compute_cross_sections`) at two different densities and save
    the results of both operations. I.e.

    >>> experiment.compute_cross_sections(key='xs_yearly', density=1/365)
    >>> experiment.compute_cross_sections(key='xs_monthly', density=1/30)

    Our state-machine memory now contains two entries of cross sections,
    `xs_yearly` and `xs_monthly`, taken from the curves once yearly and once
    monthly respectively.

    Many operations rely on previous computations. These are frequently
    accessed from the cache by appropriately named keys. For example, if we
    wished to compute data matrices for our previously computed sampling rates,
    we would execute:

    >>> experiment.build_data_matrix(key='dm_yearly', xs_key='xs_yearly')
    >>> experiment.build_data_matrix(key='dm_monthly', xs_key='xs_monthly')

    Check each operation's documentation for what other keys it needs in order
    to function.

    The state-machine's memory is truly called a cache. Cached operations check
    the cache before executing, and if their key exists they return.  Each
    cached operation accepts a keyword argument "force", which will force
    re-execution of the operation. For example, in the following sequence

    >>> experiment.fetch_data()
    >>> experiment.fetch_data()
    >>> experiment.fetch_data(force=True)

    The first line establishes a DB connection and downloads the data, storing
    it in the cache at the default key "data." The second line is then a noop.
    The third line re-runs `fetch_data` and overwrites "data" in the cache.
    """
    def __init__(self, configs, cache=None):
        self.configs = sorted(configs, key=operator.attrgetter('mode'))
        # Cache must have an interface compatible with PickleCache, which is
        # assumed to be the default implementation. Experiment state-machine
        # methods will not work without a proper cache to get/set results to
        if cache is None:
            self.cache = PickleCache()
        self.cache = cache

    @cached_operation
    def fetch_data(self, key='data', min_inst=2, min_span=182):
        """Download patient data

        Aggregates by patient ID and converts to pandas.DataFrame before
        caching as a stream of DataFrames. Cf. cml_data_tools.source_ehr

        Keyword Arguments
        -----------------
        key : str
            Default 'data'. Specifies cache key for the fetched data.
        min_inst : int
            Default 2. Minimum number of distinct dates a patient record must
            have in order to be used for further processing.
        min_span : int
            Default 182. Minimum distance between the earliest and latest date
            in the patient record for which there is data. Default is 6 months.
        """
        data_iter = map(make_data_df, aggregate_data(self.configs))
        # Filter out persons who don't have at least min_inst distinct
        # dates of data
        # TODO: min_span
        data_iter = (
            df for df in data_iter
            if len(df['date'][~np.isnan(df['date'])].unique()) >= min_inst
        )
        self.cache.set_stream(key, data_iter)

    @cached_operation
    def fetch_meta(self, key='meta'):
        """Download modal data metadata. Cf. cml_data_tools.source_ehr

        Keyword Arguments
        -----------------
        key : str
            Default 'meta'. Specifies cache key for the fetched metadata.
        """
        meta = make_meta_df(aggregate_meta(self.configs))
        self.cache.set(key, meta)

    @cached_operation
    def compute_curves(self, key='curves',
                       data_key='data',
                       curve_stats_key='curve_stats',
                       resolution='D',
                       extra_curve_kws=None,
                       max_workers=0,
                       calc_stats=True):
        """Compute patient curves. Transforms patient data from heterogeneous
        data types into curves of floats.

        Keyword Arguments
        -----------------
        key : str
            Default 'curves'. The key for the generated curves.
        data_key : str
            Default 'data'. The key of input patient dataframes.
        curve_stats_key : str
            Default 'curve_stats'. The key for curve statistics. Cf.
            `calc_stats`.
        resolution : str
            Default 'D', i.e. daily. The resolution of the generated curves.
        extra_curve_kws : Mapping, optional
            Extra curve gen configuration.
        max_workers : int
            Default 0. If above zero, specifies the number of subprocesses to
            use in generating curves in parallel. At default of zero, no
            parallelization is used.
        calc_stats : bool
            Default True. Flag indicating whether or not to calculate curve
            statistics alongside curve generation. If True, then the curve
            stats are stored at `curve_stats_key`. This allows us to
            parallelize collection of curve statistics, which otherwise must be
            run in serial.
        """
        xtra = extra_curve_kws or {}
        spec = {}
        for config in self.configs:
            extra_kws = xtra.get(config.mode, {})
            func = config.curve_builder(**extra_kws)
            spec[config.mode] = func

        data = self.cache.get_stream(data_key)
        if max_workers > 0:
            curves_iter = _parallel_curve_gen(data, max_workers, spec,
                                              resolution, calc_stats)
        # Single core execution
        else:
            curves_iter = (_worker(df, spec, resolution, calc_stats)
                           for df in data)

        if calc_stats:
            def _intercept_stats(curves_iter):
                stats = None
                for curves, new_stats in curves_iter:
                    stats = update_curve_stats(stats, new_stats)
                    yield curves
                self.cache.set(curve_stats_key, stats)
        else:
            def _intercept_stats(curves_iter):
                for curves, _ in curves_iter:
                    yield curves

        # Drives the iterators - i.e. this is when the work happens
        self.cache.set_stream(key, _intercept_stats(curves_iter))

    @cached_operation
    def compute_curve_stats(self, key='curve_stats', curves_key='curves'):
        """Calculate curve statistics on existing curves. Runs in serial
        (i.e. slowly). It is faster to generate stats during curve generation.

        Keyword Arguments
        -----------------
        key : str
            Default 'curve_stats'. Key for generated curve statistics.
        curves_key : str
            Default 'curves'. Key for curves to generate stats on.
        """
        curves = self.cache.get_stream(curves_key)
        stats = collect_curve_stats(next(curves))
        for curveset in curves:
            new = collect_curve_stats(curveset)
            stats = update_curve_stats(stats, new)
        self.cache.set(key, stats)

    @cached_operation
    def compute_cross_sections(self, key='curve_xs',
                               curves_key='curves',
                               density=1/365):
        """Subsamples a set of curves at `density`.

        Keyword Arguments
        -----------------
        key : str
            Default 'curve_xs'. Key for subsampled cross sections'
        curves_key : str
            Default 'curves'. Key for input stream of curves.
        density : float
            Default 1/365 (i.e. roughly yearly). Specifies how frequently to
            subsample each patient dataframe.
        """
        curves_iter = self.cache.get_stream(curves_key)
        samples = _sample_from_curves(curves_iter, density)
        self.cache.set_stream(key, samples)

    @cached_operation
    def make_standardizer(self, key='standardizer', stats_key='curve_stats'):
        """Instantiate and fit an OnlineCurveStandardizer.

        Keyword Arguments
        -----------------
        key : str
            Default 'standardizer'. Key for the OnlineCurveStandardizer
            instance.
        stats_key : str
            Default 'curve_stats'. Key for already computed curve stats.
        """
        stats = self.cache.get(stats_key)
        std = OnlineCurveStandardizer(curve_stats=stats, configs=self.configs)
        std.fit()
        self.cache.set(key, std)

    @cached_operation
    def build_standardized_data_matrix(self, key='std_matrix',
                                       dense_key='data_matrix',
                                       meta_key='meta',
                                       xs_key='curve_xs',
                                       std_key='standardizer',
                                       save_dense=True):
        """Merges a set of dataframes with heterogeneous column labels into a
        single dataframe. Prunes the column labels by what actually exists in
        the curve data (taken from the fit standardizer), and then finally
        standardizes the dataframe.

        Keyword Arguments
        -----------------
        key : str
            Default 'std_matrix'. Key for the final merged, standardized, and
            filled dataframe.
        dense_key : str
            Default 'data_matrix'. Key for merged dataframe. Cf `save_dense`.
        meta_key : str
            Default 'meta'. Key for the column label metadata.
        xs_key : str
            Default 'curve_xs'. Key for the stream of dataframes to merge.
        std_key : str
            Default 'standardizer'. Key for a fitted standardizer instance.
        save_dense : bool
            Default True. If True, then this step saves the unstandardized,
            unfilled data matrix in the cache using `dense_key` as the key.
            This can be useful for manually inspecting the merged data prior to
            filling and standardizing.
        """
        meta = self.cache.get(meta_key)
        cross_sections = self.cache.get_stream(xs_key)
        standardizer = self.cache.get(std_key)
        dense, matrix = _build_std_matrix(meta, cross_sections, standardizer)
        if save_dense:
            self.cache.set(dense_key, dense)
        self.cache.set(key, matrix)

    @cached_operation
    def build_data_matrix(self, key='data_matrix',
                          meta_key='meta',
                          xs_key='curve_xs'):
        """Merges a set of dataframes with heterogeneous column labels into a
        single dataframe.

        Keyword Arguments
        -----------------
        key : str
            Default 'data_matrix'. Key for merged dataframe.
        meta_key : str
            Default 'meta'. Key for the column label metadata.
        xs_key : str
            Default 'curve_xs'. Key for the stream of dataframes to merge.
        """
        meta = self.cache.get(meta_key)
        cross_sections = self.cache.get_stream(xs_key)
        dense_df = _to_data_matrix(meta, cross_sections)
        self.cache.set(key, dense_df)

    @cached_operation
    def standardize_data_matrix(self, key='std_matrix',
                                std_key='standardizer',
                                data_matrix_key='data_matrix',
                                meta_key='meta'):
        """Applies a fit standardizer to a data matrix, using fill values as
        needed from the column label metadata.

        Keyword Arguments
        -----------------
        key : str
            Default 'std_matrix'. Key for standardized dataframe.
        std_key : str
            Default 'standardizer'. Key for a fitted standardizer instance.
        meta_key : str
            Default 'meta'. Key for the column label metadata.
        """
        meta = self.cache.get(meta_key)
        data_matrix = self.cache.get(data_matrix_key)
        standardizer = self.cache.get(std_key)
        std_data = _to_std_data_matrix(meta, data_matrix, standardizer)
        self.cache.set(key, std_data)

    @cached_operation
    def learn_model(self, key='model',
                    train_data_key='std_matrix',
                    **model_kws):
        """Instantiate and fit an IcaPhenotypeModel from preprocessed data.

        Keyword Arguments
        -----------------
        key : str
            Default 'model'. Key for the fit model.
        train_data_key : str
            Default 'std_matrix'. Key for a processed, standardized dataframe.
        **kwargs
            All other kwargs are forwarded to IcaPhenotypeModel at
            instantiation.
        """
        data = self.cache.get(train_data_key)
        model = IcaPhenotypeModel(**model_kws)
        model.fit(data)
        self.cache.set(key, model)

    @cached_operation
    def collect_phenotypes(self, model_keys, key='phenotypes'):
        """Collect phenotypes from models specified in model_keys into a
        single file for further analysis or hand inspection.

        Arguments
        ---------
        model_keys : Iterator
            An input iterator of keys (or locations) for models to aggregate

        Keyword Arguments
        -----------------
        key : str
            Default 'phenotypes'. Key for the dict of phenotypes.
        """
        phen = {}
        for k in model_keys:
            model = self.cache.get(k)
            phen[k] = model.phenotypes_
        self.cache.set(key, phen)

    @cached_operation
    def create_affinity_matrix(self, key='affinity_matrix',
                               phenotypes_key='phenotypes'):
        """Transform the collected phenotypes from trained submodels into an
        affinity matrix.

        Keyword Arguments
        -----------------
        key : str
            Default 'affinity_matrix'. Key for the 2d ndarray of similarities.
        phenotypes_key : str
            Default 'phenotypes'. Key for the dict of phenotypes.
        """
        phenotypes = self.cache.get(phenotypes_key)
        aff_matrix = make_affinity_matrix(list(phenotypes.values()))
        self.cache.set(key, aff_matrix)

    @cached_operation
    def cluster_affinities(self, key='clustering',
                           clusters_key='clusters',
                           affinity_key='affinity_matrix',
                           preference=None,
                           threshold=None,
                           **kwargs):
        """Run affinity propagation on precomputed similarities (affinities).

        Keyword Arguments
        -----------------
        key : str
            Default 'clustering'. Key for a clustering.Clustering object.
        clusters_key : str
            Default 'clusters'. Key for secondary product, triplets of
            (submatrix, indices, center) for each cluster, where the submatrix
            is the original, unmasked values of the affinity matrix for the
            cluster centered at `center`.
        affinities_key : str
            Default 'affinity_matrix'. Key for the precomputed affinities.
        preference : Number
            Default None. The preference is passed through to the Affinity
            Propagation algorithm. If None (the default), the median of the
            unmasked affinity matrix is used.
        threshold : Number
            Default None. Used to construct a sparse affinity matrix. If None,
            the cached affinity matrix is used without any masking.
        **kwargs
            All other kwargs are forwarded to the affinity propagation
            algorithm.
        """
        S = self.cache.get(affinity_key).copy()
        # Use median as default preference for Affinity Propagation
        # median should be calculated prior to threshold masking
        if preference is None:
            preference = np.median(S)
        kwargs['preference'] = preference
        # Generate clustering on copy of affinity matrix
        S0 = S.copy()
        if threshold is not None:
            S0[S0 < threshold] = 0.0
        clusterer = AffinityPropagationClusterer(**kwargs)
        clusterer.fit(S0)
        self.cache.set(key, clusterer)
        # Pull clusters from original affinity matrix
        clusters = list(iter_clusters(S, clusterer.labels_))
        self.cache.set(clusters_key, clusters)

    def combine_models(self):
        # TODO
        pass

    def plot_model(self, pdf_path='phenotypes.pdf',
                   model_key='model',
                   meta_key='meta',
                   std_key='standardizer'):
        """Plot a model to a file of PDFs. This operation is not cached.

        Keyword Arguments
        -----------------
        pdf_path : str or pathlib.Path
            Default "phenotypes.pdf"; a pathlike object specifying where to
            plot the learned phenotypes in the model.
        model_key : str
            Default 'model'. Cache key for the fit model to plot.
        meta_key : str
            Default 'meta'. Key for the column label metadata.
        std_key : str
            Default 'standardizer'. Key for a fitted standardizer instance.
        """
        path = pathlib.Path(pdf_path).resolve()
        model = self.cache.get(model_key)
        meta = self.cache.get(meta_key)
        standardizer = self.cache.get(std_key)
        plot_phenotypes_to_file(model.phenotypes_,
                                model.expressions_,
                                path, meta, standardizer)

    @cached_operation
    def compute_expressions(self, key='expressions', curves_key='curves',
                            std_key='standardizer', model_key='model',
                            freq='6D', agg=None):
        curves = self.cache.get_stream(curves_key)
        std = self.cache.get(std_key)
        model = self.cache.get(model_key)
        expressions = _curve_expressions(curves, std, model, freq, agg)
        self.cache.set_stream(key, expressions)

    @cached_operation
    def compute_trajectories(self, key='trajectories', expr_key='expressions',
                             freq='6MS', agg='max'):
        expressions = self.cache.get_stream(expr_key)
        trajectories = _expression_trajectories(expressions, freq, agg)
        self.cache.set_stream(key, trajectories)
