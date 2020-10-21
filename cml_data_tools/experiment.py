"""
An Experiment class is a convenience for running scripts
"""
import collections
import pathlib
import pickle
import operator
from concurrent.futures import as_completed, ProcessPoolExecutor

import pandas as pd

from cml_data_tools.curves import build_patient_curves
from cml_data_tools.models import IcaPhenotypeModel
from cml_data_tools.plotting import plot_phenotypes_to_file
from cml_data_tools.source_ehr import (make_data_df, make_meta_df,
                                       aggregate_data, aggregate_meta)
from cml_data_tools.standardizers import DataframeStandardizer


class Experiment:
    def __init__(self, configs,
                 loc=pathlib.Path(),
                 protocol=pickle.HIGHEST_PROTOCOL,
                 suffix='.pkl'):
        self.configs = sorted(configs, key=operator.attrgetter('mode'))
        self.cache = pathlib.Path(loc).resolve()
        self.cache.mkdir(exist_ok=True)
        self.protocol = protocol
        self.suffix = suffix

    def _read(self, path, post=None):
        with open(path, 'rb') as file:
            while True:
                try:
                    x = pickle.load(file)
                except EOFError:
                    break
                else:
                    if callable(post):
                        yield post(x)
                    else:
                        yield x

    @property
    def data_(self):
        yield from self._read(self.data_path_, post=make_data_df)

    @property
    def meta_(self):
        if not hasattr(self, '_meta_df'):
            with open(self.meta_path_, 'rb') as file:
                self._meta_df = pickle.load(file)
        return self._meta_df

    @property
    def curves_(self):
        yield from self._read(self.curve_path_)

    @property
    def cross_sections_(self):
        yield from self._read(self.cross_sections_path_)

    @property
    def data_matrix_(self):
        if not hasattr(self, '_data_matrix'):
            with open(self.data_matrix_path_, 'rb') as file:
                self._data_matrix = pickle.load(file)
        return self._data_matrix

    @property
    def standardizer_(self):
        """Customized via make_standardizer"""
        if not hasattr(self, '_standardizer'):
            # Make a default standardizer using the default configs
            self.make_standardizer()
        return self._standardizer

    @property
    def standardized_data_(self):
        if not hasattr(self, '_standardized_data'):
            with open(self.standardized_data_path_, 'rb') as file:
                self._standardized_data = pickle.load(file)
        return self._standardized_data

    @property
    def model_(self):
        if not hasattr(self, '_model'):
            with open(self.model_path_, 'rb') as file:
                self._model = pickle.load(file)
        return self._model

    @property
    def trajectories_(self):
        yield from self._read(self.trajectories_path_)

    def fetch_data(self, key='data', configs=None):
        """Populate self.data_"""
        srcs = configs or self.configs
        path = (self.cache/key).with_suffix(self.suffix)

        if path not in self.cache.iterdir():
            with open(path, 'wb') as file:
                for recs in aggregate_data(srcs):
                    pickle.dump(recs, file, protocol=self.protocol)

        self.data_path_ = path
        return self.data_

    def fetch_meta(self, key='meta', configs=None):
        """Populate self.meta_"""
        srcs = configs or self.configs
        path = (self.cache/key).with_suffix(self.suffix)

        if path not in self.cache.iterdir():
            meta = make_meta_df(aggregate_meta(srcs))
            with open(path, 'wb') as file:
                pickle.dump(meta, file, protocol=self.protocol)

        self.meta_path_ = path
        return self.meta_

    def compute_curves(self, key='curves',
                       configs=None,
                       resolution='D',
                       extra_curve_kws=None,
                       n_cpu=0):
        """Populate self.curves_"""
        cfgs = configs or self.configs
        xtra = extra_curve_kws or {}
        path = (self.cache/key).with_suffix(self.suffix)

        spec = {}
        for config in cfgs:
            extra_kws = xtra.get(config.mode, {})
            func = config.curve_builder(**extra_kws)
            spec[config.mode] = func

        if path not in self.cache.iterdir():
            with open(path, 'wb') as file:
                # Parallel execution
                if n_cpu > 0:
                    with ProcessPoolExecutor(max_workers=n_cpu) as pool:
                        futures = []
                        for df in self.data_:
                            futures.append(pool.submit(build_patient_curves,
                                                       df, spec, resolution))
                        for future in as_completed(futures):
                            curves = future.result()
                            pickle.dump(curves, file, protocol=self.protocol)
                            futures.remove(future)
                            del future
                # Single core execution
                else:
                    for df in self.data_:
                        curves = build_patient_curves(df, spec, resolution)
                        pickle.dump(curves, file, protocol=self.protocol)

        self.curve_path_ = path
        self.curve_spec_ = spec
        return self.curves_

    def compute_cross_sections(self, key='curve_xs',
                               configs=None,
                               density=1 / (1 * 365)):
        """Populate self.cross_sections_"""
        path = (self.cache/key).with_suffix(self.suffix)

        if path not in self.cache.iterdir():
            rng = np.random.default_rng()
            with open(path, 'wb') as file:
                for df in self.curves_:
                    n = rng.binomial(len(df.index), density)
                    if n > 0:
                        samples = df.sample(n=n)
                        pickle.dump(samples, file, protocol=self.protocol)

        self.cross_sections_path_ = path
        return self.cross_sections_

    def make_standardizer(self, configs=None, extra_std_kws=None):
        """Generate and configure a DataframeStandardizer"""
        cfgs = configs or self.configs
        xtra = extra_std_kws or {}
        std = DataframeStandardizer()
        for config in cfgs:
            kws = config.std_kws.copy()
            kws.update(xtra.get(config.mode, {}))
            std.add_standardizer(config.mode, config.std_cls, **kws)
        self._standardizer = std

    def build_data_matrix(self, key='data_matrix'):
        path = (self.cache/key).with_suffix(self.suffix)

        if path not in self.cache.iterdir():
            channel_names = self.meta_[['mode', 'channel']]
            channel_names = pd.MultiIndex.from_frame(channel_names)

            dense_df = pd.concat([df.reindex(columns=channel_names)
                                  for df in self.cross_sections_],
                                 copy=False)
            dense_df = dense_df.dropna(axis='columns', how='all')

            with open(path, 'wb') as file:
                pickle.dump(dense_df, file, protocol=self.protocol)
            self._data_matrix = dense_df

        self.data_matrix_path_ = path
        return self.data_matrix_

    def standardize_data_matrix(self, key='std_matrix'):
        """Standardize self.data_matrix_ => self.standardized_data_"""
        path = (self.cache/key).with_suffix(self.suffix)
        self.standardizer_.fit(self.data_matrix_)

        if path not in self.cache.iterdir():
            # Standardize data matrix
            # NOTE: This step will drop channels that are constant or zero
            # (i.e. uninformative) from the data matrix (the steps below will
            # do the same to the metadata)
            std_data = self.standardizer_.transform(self.data_matrix_)
            # Meta has cols 'mode', 'channel', 'description', 'fill'
            # To do the align below we need 'mode' & 'channel' to be the index
            # (since these are the columns of the data matrix)
            meta = self.meta_.set_index(['mode', 'channel'])
            # An error such as
            #   "TypeError: loop of ufunc does not support argument 0 of type"
            #   "float which has no callable log10 method"
            # Indicates that the series in the fill df have dtype "object,"
            # they need to have a float dtype.
            fill = pd.DataFrame(meta['fill']).transpose().astype(float)
            fill, _ = fill.align(std_data, join='right', axis=1)
            fill_series = self.standardizer_.transform(fill).loc['fill']
            std_data = std_data.fillna(fill_series)

            with open(path, 'wb') as file:
                pickle.dump(std_data, file, protocol=self.protocol)
            self._standardized_data = std_data

        self.standardized_data_path_ = path
        return self.standardized_data_

    def learn_model(self, key='model', **model_kws):
        """Populate self.model_ and fit with self.standardized_data_"""
        path = (self.cache/key).with_suffix(self.suffix)

        if path not in self.cache.iterdir():
            model = IcaPhenotypeModel(**model_kws)
            model.fit(self.standardized_data_)
            with open(path, 'wb') as file:
                pickle.dump(model, file, protocol=self.protocol)
            self._model = model

        self.model_path_ = path
        return self.model_

    def plot_model(self, key='phenotypes'):
        path = (self.cache/key).with_suffix('.pdf')
        plot_phenotypes_to_file(self.model_.phenotypes_,
                                self.model_.expressions_,
                                path,
                                self.meta_,
                                self.standardizer_)

    def compute_expressions(self, key='expressions', freq='6D', agg=None):
        self.expressions_ = []
        for df in self.curves_:
            X = df.resample(freq, level='date').mean()
            X = self.standardizer_.transform(X)
            X = self.model_.transform(X)
            if agg is not None:
                X = X.agg(agg)
            X.name = df.index.get_level_values('id')[0]
            self.expressions_.append(X)
        return self.expressions_

    def compute_trajectories(self, key='trajectories',
                             configs=None,
                             freq='6MS',
                             agg='max'):
        path = (self.cache/key).with_suffix(self.suffix)
        if path not in self.cache.iterdir():
            with open(path, 'wb') as file:
                for expr in self.expressions_:
                    df = expr.resample(freq, level='date').agg(agg)
                    # XXX: pickling loses the name attribute
                    df.name = expr.name
                    pickle.dump(df, file, protocol=self.protocol)
        self.trajectories_path_ = path
        return self.trajectories_
