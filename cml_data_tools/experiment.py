"""
An Experiment class is a convenience for running scripts
"""
import pathlib
import pickle
import operator

import pandas as pd


class Experiment:
    def __init__(self, configs,
                 loc=pathlib.Path(),
                 protocol=pickle.HIGHEST_PROTOCOL):
        self.configs = sorted(configs, key=operator.attrgetter('mode'))
        self.cache = pathlib.Path(loc).resolve()
        self.cache.mkdir(exist_ok=True)
        self.protocol = protocol

        # Learned or otherwise generated attributes
        self.data_ = None
        self.meta_ = None
        self.curves_ = None
        self.curves_spec_ = None
        self.cross_sections_ = None

    def _write_cache(self, key, it):
        with open(self.cache/f'{key}.pkl', 'wb') as file:
            for val in it:
                pickle.dump(val, file, protocol=self.protocol)

    def _read_cache(self, key):
        with open(self.cache/f'{key}.pkl', 'rb') as file:
            while True:
                try:
                    yield pickle.load(file)
                except EOFError:
                    break

    def _has_cached(self, key):
        return key in {p.stem for p in self.cache.iterdir()}

    def fetch_data(self, key='data', configs=None, eager=False):
        """Populate data_"""
        cols = ('ptid', 'date', 'mode', 'channel', 'value')
        srcs = configs or self.configs

        if not self._has_cached(key):
            self._write_cache(key, aggregate_data(srcs))

        data = self._read_cache(key)
        self.data_ = list(data) if eager else data

    def fetch_meta(self, key='meta', configs=None):
        """Populate meta_"""
        cols = ('mode', 'channel', 'description', 'fill')
        srcs = configs or self.configs

        if not self._has_cached(key):
            self._write_cache(key, aggregate_data(srcs))

        meta = list(self._read_cache(key))
        self.meta_ = pd.DataFrame(meta, columns=cols)

    def compute_curves(self, key='curves',
                       configs=None,
                       resolution='D',
                       eager=False):
        """Populate curves_spec_ and curves_"""
        if configs is None:
            configs = self.configs

        if self.data_ is None:
            raise RuntimeError

        spec = {c.mode: c.curve_cls(**c.curve_kws) for c in configs}
        self.curves_spec_ = spec

        curves = (compute_curves(df, spec, resolution) for df in self.data_)
        self._write_cache(key, curves)
        reader = self._read_cache(key)
        self.curves_ = list(reader) if eager else reader

    def compute_cross_sections(self, key='xs',
                               configs=None,
                               density=1 / (1* 365),
                               eager=False):
        samples = (df.sample(fac=max(1 / len(df.index), density))
                   for df in self.curves_)
        self._write_cache(key, samples)
        reader = self._read_cache(key)
        self.cross_sections_ = list(reader) if eager else reader
