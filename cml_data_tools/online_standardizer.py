import collections
import numpy as np
from cml_data_tools import online_norm
from cml_data_tools.standardizers import (DataframeStandardizer,
                                          LinearStandardizer,
                                          GelmanStandardizer,
                                          LogGelmanStandardizerWithFallbacks)


CurveStats = collections.namedtuple('CurveStats', [
    'channels', 'n_tot', 'n_pos', 'base_mean', 'base_var', 'log10_mean',
    'log10_var', 'curve_min', 'curve_max'
])


def collect_curve_stats(curves):
    # We want to track, for each curve, the total number of observations
    # and the number of non-negative values, the mean and variance of the
    # curve, and the mean and variance of the curve to log10
    C = curves.columns.values
    X = curves.values

    M = np.nanmean(X, axis=0)
    V = np.nanvar(X, axis=0)
    N_tot = np.full(M.shape, len(X), dtype=np.int64)
    N_pos = (X > 0.0).sum(axis=0)

    X_log = np.log10(X)
    X_log[np.isinf(X_log)] = np.nan
    M_log = np.nanmean(X_log, axis=0)
    V_log = np.nanvar(X_log, axis=0)

    min_ = np.nanmin(X, axis=0)
    max_ = np.nanmax(X, axis=0)

    return CurveStats(C, N_tot, N_pos, M, V, M_log, V_log, min_, max_)


def update_curve_stats(prev, curr):
    if prev is None:
        return curr

    base_prev = (prev.channels, prev.base_mean, prev.base_var, prev.n_tot)
    base_curr = (curr.channels, curr.base_mean, curr.base_var, curr.n_tot)
    base = online_norm.update(base_prev, base_curr)

    log10_prev = (prev.channels, prev.log10_mean, prev.log10_var, prev.n_tot)
    log10_curr = (curr.channels, curr.log10_mean, curr.log10_var, curr.n_tot)
    log10 = online_norm.update(log10_prev, log10_curr)

    assert (base.channels == log10.channels).all()
    assert (base.n == log10.n).all()

    n_pos = prev.n_pos + curr.n_pos
    assert n_pos.shape == base.channels.shape

    curve_min = np.minimum(prev.curve_min, curr.curve_min)
    assert curve_min.shape == base.channels.shape

    curve_max = np.maximum(prev.curve_max, curr.curve_max)
    assert curve_max.shape == base.channels.shape

    return CurveStats(base.channels, base.n, n_pos, base.mean, base.var,
                      log10.mean, log10.var, curve_min, curve_max)


class OnlineCurveStandardizer(DataframeStandardizer):
    def __init__(self, curve_stats=None):
        super().__init__()
        self.curve_stats = curve_stats

    def update_from_stats(self, stats):
        self.curve_stats = update_curve_stats(self.curve_stats, stats)

    def update_from_curves(self, curves):
        stats = collect_curve_stats(curves)
        self.curve_stats = update_curve_stats(self.curve_stats, stats)

    def stats_as_df(self):
        if self.curve_stats is not None:
            return online_norm.to_dataframe(self.curve_stats)

    def fit(self, curves=None, y=None):
        # Hook to fit from curves given at last minute
        if curves is not None:
            for df in curves:
                self.update_from_curves(df)

        stats_df = self.stats_as_df().T
        if stats_df is None:
            raise RuntimeError(f'{self.__class__.__name__} is misconfigured')

        for channel_name in stats_df:
            stats = stats_df[channel_name]
            mode = channel_name[0]
            standardizer, kwargs = self._standardizer_info[mode]
            instance = standardizer(**kwargs)
            # Manually fit the standardizers
            if standardizer is LinearStandardizer:
                pass
            elif standardizer is GelmanStandardizer:
                if instance.log_transform:
                    mean = stats.log10_mean
                    stdev = np.sqrt(stats.log10_var)
                else:
                    mean = stats.base_mean
                    stdev = np.sqrt(stats.base_var)
                instance._mean = mean
                instance._stdev = stdev
                instance.eps = 0.0
            elif standardizer is LogGelmanStandardizerWithFallbacks:
                if np.isnan(stats.base_mean):
                    transformer = LinearStandardizer()
                elif stats.base_var < 1e-6:
                    transformer = LinearStandardizer(offset=stats.curve_min)
                elif stats.n_pos < (stats.n_tot * 0.99):
                    transformer = GelmanStandardizer(log_transform=False)
                    transformer._mean = stats.base_mean
                    transformer._stdev = np.sqrt(stats.base_var)
                else:
                    transformer = GelmanStandardizer(log_transform=True)
                    transformer._mean = stats.log10_mean
                    transformer._stdev = np.sqrt(stats.log10_var)
                instance._transformer = transformer
            self._transformer[channel_name] = instance
        return self
