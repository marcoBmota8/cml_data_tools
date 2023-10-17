import abc
import collections
import contextlib
import heapq
import itertools
import operator

import pandas


DATA_COLS = ('ptid', 'date', 'mode', 'channel', 'value')
META_COLS = ('mode', 'channel', 'description', 'fill')


def drop_unparseable_dates(df):
    """
    If the df has a date column with non-null values, convert it to pandas
    datetime objects are drop any rows which fail to convert.
    """
    #if 'date' in df.columns and not df['date'].isna().all():
    # Remove second condition: If df['date'] are all None, we want to convert
    # them to NaT so that ufuncs down the line will handle them correctly
    if 'date' in df.columns:
        df['date'] = pandas.to_datetime(df['date'],
                                        errors='coerce')
        if not df['date'].isna().all():
            df.dropna(subset=['date'], inplace=True)
    return df


def make_data_df(rows, drop_dates_by='mode'):
    """Convert patient data tuple iterator to DataFrame"""
    df = pandas.DataFrame.from_records(list(rows), columns=DATA_COLS)
    df = df.infer_objects()
    df = df.groupby(by=drop_dates_by).apply(drop_unparseable_dates)
    return df


def make_meta_df(rows):
    """Convert channel metadata iterator to DataFrame"""
    df = pandas.DataFrame.from_records(rows, columns=META_COLS)
    df = df.infer_objects()
    return df


def moded(mode, data):
    """Utility function, cf. `aggregate_data`"""
    # Module level definition allows us to pickle aggregate_data
    for (ptid, date, channel, value) in data:
        yield (ptid, date, mode, channel, value)


def aggregate_data(sources):
    """Utility function to aggregate multiple data streams by patient ID.

    Arguments
    ---------
    sources : Iterable[Any]
        Each source object represents a distinct stream of input datapoints.
        They may be any object which contains "mode" and "data" attributes; the
        mode attribute is expected to be a string name for this particular
        input stream, and the data attribute the actual input iterator.
        Iteration over the "data" attribute is expected to yield tuples of
        (patient id, date, channel name, channel value). This function is as
        lazy as the input data streams. The input data streams are assumed to
        be presorted by patient ID (sorting within this function would require
        fully consuming each iterator before merging).

    Yields
    ------
    Tuple[Tuple]
        A tuple per patient of data tuples. Each inner tuple will have five
        fields, (patient id, date, mode, channel, value). Tuples are yielded in
        ascending order of patient id.
    """
    get_id = operator.itemgetter(0)
    moded_ = (moded(s.mode, s.data) for s in sources)
    merged = heapq.merge(*moded_, key=get_id)
    groups = itertools.groupby(merged, key=get_id)
    yield from (tuple(grp) for _, grp in groups)


def aggregate_meta(sources):
    """Utility function to aggregate channel metadata across multiple modes.

    Arguments
    ---------
    sources : Iterable[Source]
        Cf. aggregate_modes. The only requirement is that "mode" and "meta"
        attributes be present, and that the meta attribute be an iterator
        providing tuples of (channel, description, fill).

    Yields
    ------
    Tuple[str, str, str, fill value]
        Tuples of (mode, channel, description, fill value). Iteration order
        follows source order.
    """
    for s in sources:
        for (chan, desc, fill) in s.meta:
            yield (s.mode, chan, desc, fill)


class DatabaseSource:
    """
    Base class for simple database driven sources

    Arguments
    ---------
    connection_factory : function
        A zero-argument callable that must return a DB-API 2.0 compliant
        connection object. The connection is closed after use.

    arraysize : int or None
        If not None, then results are iterated over in arraysize chunks using
        cursor.fetchmany(). Otherwise cursor.fetchall() is used to get the rows
        and then the cursor is closed immediately, prior to iteration. This
        parameter exists for tuning memory usage in constrained environments.
    """
    def __init__(self, table, connection_factory, arraysize=None):
        self.table = table
        self.connection_factory = connection_factory
        self.arraysize = arraysize

    def __iter__(self):
        if self.arraysize is None:
            yield from self._fetchall(self.sql)
        else:
            yield from self._fetchmany(self.sql)

    def _fetchall(self, sql):
        # Pulls all data and immediately closes all db objs before handing
        # control back to caller
        with contextlib.closing(self.connection_factory()) as conn:
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
        yield from self.post(rows)

    def _fetchmany(self, sql):
        # Pulls data in chunks; db connection / cursor are open until all data
        # has been passed through to caller
        with contextlib.closing(self.connection_factory()) as conn:
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.arraysize = self.arraysize
                cursor.execute(sql)
                rows = cursor.fetchmany()
                while rows:
                    yield from self.post(rows)
                    rows = cursor.fetchmany()

    def post(self, rows):
        """
        Provides a hook for subclasses to postprocess the raw database tuples
        before __iter__ yields them. Use for converting values to a datatype,
        prepending a mode, etc.
        """
        yield from rows

    @property
    @abc.abstractmethod
    def sql(self):
        """Defines the SQL used in this Database source"""


class DataSource(DatabaseSource):
    """
    Provides a default base representation of a site specific, database driven
    source of EHR data points. Downstream components will assume __iter__
    provides tuples of the form (patient id, date, channel, value) sorted by
    patient id and date; if the table provided is not so organized, subclasses
    may directly subclass `DatabaseSource` and / or override the `sql`
    property.
    """
    @property
    def sql(self):
        return f'SELECT DISTINCT * FROM {self.table} ORDER BY id, date'


class MetaSource(DatabaseSource):
    """
    Provides a default base representation of a site specific, database driven
    source of channel metadata for a given mode. Downstream components will
    assume __iter__ provides tuples of the form (channel, description, fill
    value) sorted by channel; if the table provided is not so organized,
    subclasses may directly subclass `DatabaseSource` and / or override the
    `sql` property.
    """
    @property
    def sql(self):
        return f'SELECT DISTINCT * FROM {self.table} ORDER BY channel'
