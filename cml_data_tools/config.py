"""
Mode configuration for experiments.
"""
import collections


class Config:
    """Generic namespace for carrying around configuration data on a mode"""
    def __init__(self, mode=None, data=None, meta=None,
                 curve_cls=None, std_cls=None,
                 curve_kws=None, std_kws=None):
        self.mode = mode
        self.data = data
        self.meta = meta
        self.curve_cls = curve_cls
        self.curve_kws = {} if curve_kws is None else curve_kws
        self.std_cls = std_cls
        self.std_kws = {} if std_kws is None else std_kws

    def curve_builder(self, **kws):
        kwargs = collections.ChainMap(kws, self.curve_kws)
        return self.curve_cls(**kwargs)

    def standardizer(self, **kws):
        kwargs = collections.ChainMap(kws, self.std_kws)
        return self.std_cls(**kwargs)
