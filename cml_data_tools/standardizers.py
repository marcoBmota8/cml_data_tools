import numpy as np
import numpy.random as nprnd
import pandas as pd
import scipy.stats as stats
from scipy import interpolate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class ChannelTransformer:
    """Pickleable wrapper for interp1d, which is not picklable on its own"""
    def __init__(self, xi, yi, **kwargs):
        self.xi = xi
        self.yi = yi
        self.args = kwargs
        self.f = interpolate.interp1d(xi, yi, **kwargs)

    def __call__(self, xnew):
        return self.f(xnew)

    def __getstate__(self):
        return self.xi, self.yi, self.args

    def __setstate__(self, state):
        self.xi = state[0]
        self.yi = state[1]
        self.args = state[2]
        self.f = interpolate.interp1d(self.xi, self.yi, **self.args)


class DataframeStandardizer(BaseEstimator, TransformerMixin):
    """A dataframe standardizer with a custom transformer for each data mode.

    This is intended to be used on the dataframe of cross sections destined for
    phenotype model fitting. It allows for transforming all curves into roughly
    the same range of values, which is necessary for getting an interpretable
    model. Curves of different modes or even different channels may otherwise
    cover vastly different scales, which would obscure interpretation of the
    model.

    Usage
    -----
    >>> dfs = DataframeStandardizer()
    >>> dfs.add_standardizer('Phecodes', PpfStandardizer(
    ...     lambda x: stats.lognorm.ppf(x, s=1), n_quantiles=1000))
    >>> dfs.add_standardizer('Lab Tests', PpfStandardizer(
    ...     stats.norm.ppf, n_quantiles=1000))
    >>> dfs.add_standardizer('Medications', lambda x: 4 * x)
    >>> dfs.fit(X)
    >>> transformed = dfs.transform(X)
    >>> new_transformed = dfss.transform(new_X)
    """
    def __init__(self):
        self._transformer = {}
        self._standardizer_info = {}

    def fit(self, X, y=None):
        """Fit a transformer for each column of X. Returns self.

        Arguments
        ---------
        X : pandas.DataFrame
            A pandas dataframe with one column per channel of data, typically a
            set of cross sections intended as input to a phenotype model.
        """
        for channel_name in X:
            channel = X[channel_name]
            mode = channel_name[0]
            # There's a different standardizer for each channel, but the
            # standardizer class and kwargs are specified per mode:
            standardizer, kwargs = self._standardizer_info[mode]
            self._transformer[channel_name] = standardizer(**kwargs).fit(channel)
        return self

    def transform(self, X, y=None):
        """Transform each column of X.

        Arguments
        ---------
        X : pandas.DataFrame
            A pandas dataframe, with columns corresponding to channels of data.
            Typically a set of cross sections intended as input to a phenotype
            model.

        Returns
        -------
        pandas.DataFrame
            The transformed dataframe, same size as X.
        """
        for col_name in X:
            result = self._transformer[col_name].transform(X[col_name])
            X.loc[:, col_name] = result
        return X

    def add_standardizer(self, mode, standardizer, **kwargs):
        """Add a standardizer for a given mode of data.

        At fitting time, one of these standardizers per column of data in the
        given mode will be created using `kwargs`.

        Arguments
        ---------
        mode : str
            Specifies the mode for which the standardizer will operate.
        standardizer : a Standardizer class
        kwargs : the set of kwargs to be passed to standardizer.__init__().
        """
        self._standardizer_info[mode] = (standardizer, kwargs)

    def get(self, mode):
        return self._standardizer_info.get(mode)

    def get_mode_scales(self):
        scales = {}
        # Lots of repeats, but probably not that slow.
        for channel_name in self._transformer:
            mode = channel_name[0]
            scales[mode] = self._transformer[channel_name].scale
        return scales

    def set_mode_scales(self, scales):
        for channel_name in self._transformer:
            mode = channel_name[0]
            self._transformer[channel_name].scale = scales[mode]


class LinearStandardizer(BaseEstimator, TransformerMixin):
    """A simple transformer that scales and offsets the input. This
    standardizer is non-data dependent.

    Parameters
    ----------
    scale : float (default 1.0)
        Constant scale factor by which to multiply each element
    offset : float (double 0.0)
        Constant offset to add to each element (default 0).
    """
    def __init__(self, scale=1.0, offset=0):
        self.offset = offset
        self.scale = scale

    def fit(self, series):
        """Fits a standardization function. Returns self.

        The fitted function transforms an input data series into a specified
        output space. A simple example would be a statistical standardizer that
        shifts and scales `series` so that its mean is zero and standard
        deviation is one, and fixes that transformation so that new data can be
        scaled and shifted by the same amount.

        The fitted function may be learned from the data in `series`, but
        simpler functions that are not learned from data are allowed.
        (Subclassed methods must retain the `series` argument, even if they
        ignore it.)

        Arguments
        ---------
        series : pandas.Series
            Contains data from which to learn the transforming function
        """
        return self

    def transform(self, series):
        """Transforms the series by the previously - fit function.

        Arguments
        ---------
        series : pandas.Series

        Returns
        -------
        pandas.Series
        """
        return self.scale * series + self.offset


class YeoJohnsonStandardizer(LinearStandardizer):
    """Creates a Yeo-Johnson transformer.

    Transforms the data x to a near-standard-normal transformation with the
    Yeo-Johnson transform. If x is all positive, this is equivalent to a
    Box-Cox transform.

    This is useful for lab values, which are mostly log normal but with
    different parameterizations. Some labs are mostly normal to begin with, and
    others are highly skewed.

    An alternative to this class might be to use the PpfStandardizer with a
    standard normal distribution, but that has been found to amplify
    floating-point quantization noise if the original values are heavily
    discretized (such as x \in {0.0, 0.1, 0.2, 0.3, ... 1.0}). When curves are
    created from these values, even if they are constant for a given patient,
    quantization noise can easily result. Yeo-Johnson doesn't have that
    problem, but the tradeoff is that it doesn't reign in outliers the same way
    that a PpfStandardizer would do. Yeo-Johnson also keeps bimodal
    distributions bimodal, which could be a good thing in some circumstances.

    See: Yeo I-K, Johnson RA. A New Family of Power Transformations to Improve
    Normality or Symmetry. Biometrika. 2000;87(4):954–9.

    This transformer falls back to an identity transformer
    (LinearStandardizer(offset=0, scale=1)) if all values are NaN, and to
    LinearStandardizer(offset=-x, scale=1) if all values are extremely close to
    x.
    """
    def fit(self, series):
        # Use LinearStandardizer as placeholder if all NaN
        if series.isna().all():
            self._transformer = LinearStandardizer()

        # Use LinearStandardizer if all values are basically the same.
        else:
            hi = series.max()
            lo = series.min()
            if hi == 0 or ((hi - lo) / hi) < 1.0e-6:
                self._transformer = LinearStandardizer(offset=-hi, scale=1)
            else:
                self._transformer = PowerTransformer(method='yeo-johnson',
                                                     standardize=True,
                                                     copy=True)
        # Standardizers are designed to work separately on each series in the
        # DataFrame, but PowerTransformer is designed to handle each column
        # separately when you pass the whole 2-d array. It therefore expects a
        # 2-d input. So we have to do this (presumably) inefficiently by
        # reshaping each series as a 2-d ndarray, and then putting it back into
        # a series in the `transform` method below.
        self._transformer.fit(series.values.reshape(-1, 1))
        return self

    def transform(self, series):
        view = series.values.reshape(-1, 1)
        vals = super().transform(self._transformer.transform(view))
        vals = vals.flatten()
        return pd.Series(vals, name=series.name, index=series.index)


class LogStandardizer(LinearStandardizer):
    """Creates a standardized log transformer.

    Transforms the data x to scale * xt, where
    xt = post_scale_factor * log10(1 + pre_scale_factor * x). If
    post_scale_factor is None (default), it is chosen such that mean(xt) = 1
    for nonzero xt. This preserves the fact that unobserved values
    can be imputed to zero, and scales things compatible with the standard
    normal. The `pre_scale_factor` transforms the units of x if necessary, so
    that 1 + x is a minimal change and can be thought of as Bayesian
    smoothing. The default of pre_scale_factor=20*365.25 is intended for use
    with event intensities, which are transformed from events per day to events
    per 20 years, so the '1 +' acts like a baseline rate of 1 event per 20
    years (with 20 years being the approximate typical length of a long
    record).

    This is useful for event intensities, which tend to follow a lognormal
    distribution, but the long tail overwhelms the variables that follow a
    normal distribution.

    Then the whole thing is scaled by `scale`, consistent with all other
    standardizers.
    """
    def __init__(self, pre_scale_factor=20*365.25,
                 post_scale_factor=None,
                 scale=1.0, offset=0.0):
        super().__init__(scale=scale, offset=offset)
        self.pre_scale_factor = pre_scale_factor
        self.post_scale_factor = post_scale_factor

    def fit(self, series):
        xt = np.log10(1 + self.pre_scale_factor * series)
        if self.post_scale_factor is None:
            xt_mean = xt.mean()
            if xt_mean == 0:
                self.post_scale_factor = 1
            else:
                self.post_scale_factor = 1 / xt_mean
        return self

    def transform(self, series):
        x = 1 + self.pre_scale_factor * series
        x = np.log10(x) / self.scale
        return super().transform(x)


class SquaredStandardizer(LinearStandardizer):
    """Creates a squared transformer.

    Transforms the data x to xt = (scale * x^2) + offset. This is useful for
    variables like age, where the difference between 70 and 80 is much more
    important than the difference between 20 and 30.
    """
    def transform(self, series):
        return super().transform(np.square(series))


class PpfStandardizer(LinearStandardizer):
    """A distribution - based series standardizer.

    The standardization happens(at least conceptually) in two steps:
    1) Transform the series to its quantiles(in [0, 1])

    2) Transform those quantiles to standard distribution values, using the
        specified output distribution ppf.

    A ppf(percent point function) is the inverse cdf, mapping from quantile in
    [0, 1] to the distribution support domain(usually[-Inf, Inf] or [0,
    Inf]). You can think of it as the "output distribution" of the
    standardizer. Usual practice would be to use the parametric distribution
    that is as close as possible to your actual data distribution.

    This is a standardization method that produces comparable ranges in each
    column, but is robust to extreme outliers. It may distort relationships
    between columns if the output distribution is very different from the input
    distribution. It maintains the rank order of data within a column.

    The final transform function computed by self.fit() is a simple
    interpolated lookup, with quantiles of the series passed to self.fit()
    mapped to the values of the specified ppf output distribution. The
    granularity of this lookup is specified with the n_quantiles parameter.

    Usage
    -----
    >>> # A standardizer that transforms the input data into a standard
    >>> # lognormal distribution would be constructed as follows:
    >>> ps=PpfStandardizer(lambda x: stats.lognorm.ppf(x, s=1),
    ...                    n_quantiles=1000)
    >>> for series in X:
    ...     transformer[series] = ps.fit(X[series])
    ...     transformed = transformer[series].transform(X[series])
    ... new_transformed = transformer[new_series.name].tranform(new_series)

    Arguments
    ---------
    output_distribution : callable
        A percent point function (aka inverse cdf) that maps [0,1] onto the
        desired data space
    n_quantiles : int
        The number of quantiles to compute for the transform function (default
        1000). The larger this number, the smoother and more accurate the
        transform, but the slower and larger the transformer.
    """
    def __init__(self, output_distribution, n_quantiles=1000,
                 scale=1.0, offset=0.0):
        super().__init__(scale=scale, offset=offset)
        self.n_quantiles = n_quantiles
        self.ppf = output_distribution
        self._transformer = None

    def fit(self, series):
        sample = self._subsample(series, self.n_quantiles)

        # sample_size may be less than requested, if there was not enough data
        sample_size = len(sample)

        # divide by sample_size + 1 because ranks begin at 1
        quantiles = (stats.rankdata(sample, method='average')
                     / (sample_size + 1))

        z = self.ppf(quantiles)
        zmin = np.min(z)
        zmax = np.max(z)

        self._transformer = ChannelTransformer(sample, z,
                                               fill_value=(zmin, zmax),
                                               bounds_error=False,
                                               assume_sorted=False)
        return self

    def transform(self, series):
        return super().transform(self._transformer(series))

    @staticmethod
    def _subsample(series, sample_size):
        """Randomly downsample `series`.

        Samples uniformly without replacement from the non-NaN values of
        `series`.

        Arguments
        ---------
        series : pandas.Series
        sample_size : int
            the requested number of elements of `series` to sample at random.

        Returns
        -------
        pandas.Series
            A series containing the randomly sampled elements. The size of this
            series is at most the number of non-NaN values in `series`,
            regardless of the value of `sample_size`.
        """
        # sample without replacement from the non-nan values.
        valid_values = series[series.notna()].values

        if valid_values.size == 0:
            valid_values = np.zeros(3)

        sample_size = min(sample_size, len(valid_values))
        sample = nprnd.choice(valid_values, size=sample_size, replace=False)

        # Add an additional minimum and maximum value, so that the average rank
        # of the duplicated values at each end will produce a cdf other than 0
        # or 1, which map to -inf, inf and cause problems.
        return np.append(sample, (min(sample), max(sample)))
