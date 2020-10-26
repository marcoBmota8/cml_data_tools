import collections

import numpy as np
import pandas as pd

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

    log_X = np.log10(X)
    log_X[np.isinf(log_X)] = np.nan
    log_M = np.nanmean(log_X, axis=0)
    log_V = np.nanvar(log_X, axis=0)

    c_min = np.nanmin(X, axis=0)
    c_max = np.nanmax(X, axis=0)

    return CurveStats(C, N_tot, N_pos, M, V, log_M, log_V, c_min, c_max)


def update_curve_stats(prev, curr):
    if prev is None:
        return curr

    prev_chan, prev_n, prev_pos, prev_m, prev_v, prev_log_m, prev_log_v,\
        prev_min, prev_max = prev

    curr_chan, curr_n, curr_pos, curr_m, curr_v, curr_log_m, curr_log_v,\
        curr_min, curr_max = curr

    C, prev_idx, curr_idx = np.intersect1d(prev_chan, curr_chan,
                                           assume_unique=True,
                                           return_indices=True)

    cn = curr_n[curr_idx]
    pn = prev_n[prev_idx]
    N = pn + cn
    cf = cn / N
    pf = pn / N

    # Calculate the updated values for basic mean / var
    cm = curr_m[curr_idx]
    cv = curr_v[curr_idx]
    pm = prev_m[prev_idx]
    pv = prev_v[prev_idx]
    dx = cm - pm
    M = pm + (cf * dx)
    V = (pv * pf) + (cv * cf) + (pf * cf * dx * dx)

    # Updated mean / var of log transformed
    log_cm = curr_log_m[curr_idx]
    log_cv = curr_log_v[curr_idx]
    log_pm = prev_log_m[prev_idx]
    log_pv = prev_log_v[prev_idx]
    log_dx = log_cm - log_pm
    log_M = log_pm + (cf * log_dx)
    log_V = (log_pv * pf) + (log_cv * cf) + (pf * cf * log_dx * log_dx)

    # Update num positive, max & min values
    n_pos = prev_pos[prev_idx] + curr_pos[curr_idx]
    c_max = np.maximum(prev_max[prev_idx], curr_max[curr_idx])
    c_min = np.minimum(prev_min[prev_idx], curr_min[curr_idx])

    # Mask for prev values not in current (i.e. unchanged vals)
    p_mask = np.ones(prev_chan.shape, dtype=np.bool)
    p_mask[prev_idx] = False

    # Mask for curr values not in previous (i.e. unchanged vals)
    c_mask = np.ones(curr_chan.shape, dtype=np.bool)
    c_mask[curr_idx] = False

    # Recombine the unchanged, updated, and new values
    C = np.concatenate((C, prev_chan[p_mask], curr_chan[c_mask]))
    N = np.concatenate((N, prev_n[p_mask], curr_n[c_mask]))
    P = np.concatenate((n_pos, prev_pos[p_mask], curr_pos[c_mask]))
    M = np.concatenate((M, prev_m[p_mask], curr_m[c_mask]))
    V = np.concatenate((V, prev_v[p_mask], curr_v[c_mask]))
    log_M = np.concatenate((log_M, prev_log_m[p_mask], curr_log_m[c_mask]))
    log_V = np.concatenate((log_V, prev_log_v[p_mask], curr_log_v[c_mask]))
    c_min = np.concatenate((c_min, prev_min[p_mask], curr_min[c_mask]))
    c_max = np.concatenate((c_max, prev_max[p_mask], curr_max[c_mask]))

    # Stay sorted by channel
    sort = np.argsort(C)
    C = C[sort]
    N = N[sort]
    P = P[sort]
    M = M[sort]
    V = V[sort]
    log_M = log_M[sort]
    log_V = log_V[sort]
    c_min = c_min[sort]
    c_max = c_max[sort]

    return CurveStats(C, N, P, M, V, log_M, log_V, c_min, c_max)


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
            data = self.curve_stats._asdict()
            index = data.pop('channels')
            if isinstance(index[0], tuple):
                index = pd.MultiIndex.from_tuples(index)
            df = pd.DataFrame(data=data, index=index)
            return df

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
