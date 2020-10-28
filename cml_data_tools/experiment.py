"""
An Experiment class is a convenience for running scripts
"""
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
from cml_data_tools.models import IcaPhenotypeModel
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
    def wrapper(self, force=False, **kwargs):
        key = kwargs.get('key', default_key)
        if key not in self.cache or force:
            return func(self, **kwargs)
    return wrapper


class Experiment:
    def __init__(self, configs, cache=None):
        self.configs = sorted(configs, key=operator.attrgetter('mode'))
        # Cache must have an interface compatible with PickleCache, which is
        # assumed to be the default implementation. Experiment state-machine
        # methods will not work without a proper cache to get/set results to
        self.cache = cache

    @cached_operation
    def fetch_data(self, key='data', configs=None):
        srcs = configs or self.configs
        data_iter = map(make_data_df, aggregate_data(srcs))
        self.cache.set_stream(key, data_iter)

    @cached_operation
    def fetch_meta(self, key='meta', configs=None):
        print('Executing fetch_meta')
        srcs = configs or self.configs
        meta = make_meta_df(aggregate_meta(srcs))
        self.cache.set(key, meta)

    @cached_operation
    def compute_curves(self, key='curves', data_key='data', configs=None,
                       resolution='D', extra_curve_kws=None, max_workers=0,
                       calc_stats=True, curve_stats_key='curve_stats'):
        cfgs = configs or self.configs
        xtra = extra_curve_kws or {}
        spec = {}
        for config in cfgs:
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

        def _intercept_stats(curves_iter):
            stats = None
            for curves, new_stats in curves_iter:
                if calc_stats:
                    stats = update_curve_stats(stats, new_stats)
                yield curves

            if calc_stats:
                self.cache.set(curve_stats_key, stats)

        # Drives the iterators - i.e. this is when the work happens
        self.cache.set_stream(key, _intercept_stats(curves_iter))

    @cached_operation
    def compute_curve_stats(self, key='curve_stats', curves_key='curves'):
        # Runs in serial
        curves = self.cache.get_stream(curves_key)
        stats = collect_curve_stats(next(curves))
        for curveset in curves:
            new = collect_curve_stats(curveset)
            stats = update_curve_stats(stats, new)
        self.cache.set(key, stats)

    @cached_operation
    def compute_cross_sections(self, key='curve_xs', curves_key='curves',
                               density=1/365):
        curves_iter = self.cache.get_stream(curves_key)
        samples = _sample_from_curves(curves_iter, density)
        self.cache.set_stream(key, samples)

    @cached_operation
    def make_standardizer(self, key='standardizer',
                          fit_from_stats=True,
                          stats_key='curve_stats',
                          data_key='data_matrix',
                          configs=None, extra_std_kws=None):
        configs = configs or self.configs
        stats = self.cache.get(stats_key)
        std = OnlineCurveStandardizer(curve_stats=stats, configs=configs)
        # Fits on the previously computed curve stats provided at init
        std.fit()
        self.cache.set(key, std)

    @cached_operation
    def build_data_matrix(self, key='data_matrix',
                          meta_key='meta', xs_key='curve_xs'):
        meta = self.cache.get(meta_key)
        cross_sections = self.cache.get_stream(xs_key)
        dense_df = _to_data_matrix(meta, cross_sections)
        self.cache.set(key, dense_df)

    @cached_operation
    def standardize_data_matrix(self, key='std_matrix',
                                std_key='standardizer',
                                data_matrix_key='data_matrix',
                                meta_key='meta'):
        meta = self.cache.get(meta_key)
        data_matrix = self.cache.get(data_matrix_key)
        standardizer = self.cache.get(std_key)
        std_data = _to_std_data_matrix(meta, data_matrix, standardizer)
        self.cache.set(key, std_data)

    @cached_operation
    def learn_model(self, key='model',
                    input_data_key='std_matrix', **model_kws):
        """Create an ICA Model from the specified input data"""
        data = self.cache.get(input_data_key)
        model = IcaPhenotypeModel(**model_kws)
        model.fit(data)
        self.cache.set(key, model)

    def combine_models(self):
        # TODO
        pass

    def plot_model(self, pdf_path='phenotypes.pdf', model_key='model',
                   meta_key='meta', std_key='standardizer'):
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
