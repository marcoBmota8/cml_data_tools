import contextlib
import pathlib
import pickle


class PickleCache:
    """A pickle-based caching object.

    Pickling provides a useful tool that fits well with the file paradigm but
    relatively poorly with the cache paradigm, which is the ability to
    read/write pickles to/from a stream. For operations on very large amounts
    of data (i.e. curve generation or subsampling) the ability to work with
    streams is important for memory efficiency. Accordingly, the PickleCache
    adds to the generic caching paradigm of getting and setting objects via key
    `get_stream` and `set_stream`.

    Parameters
    ----------
    loc : str or pathlib.Path
        A pathlike object specifying the location of the cache; i.e. a
        directory in which to store pickles. Defaults to pathlib.Path()
    protocol : int
        Default is pickle.HIGHEST_PROTOCOL. The pickling protocol to use in
        writing to the cache.
    suffix : str
        Default '.pkl'. The file suffix to use for cached pickles.
    """
    def __init__(self,
                 loc=pathlib.Path(),
                 protocol=pickle.HIGHEST_PROTOCOL,
                 suffix='.pkl'):
        self.loc = pathlib.Path(loc).resolve()
        self.loc.mkdir(exist_ok=True)
        self.protocol = protocol
        self.suffix = suffix

    def __contains__(self, key):
        return self._make_path(key) in self.loc.iterdir()

    def __len__(self):
        return len(self.loc.iterdir())

    def __str__(self):
        return (f'{self.__class__.__name__}(loc={self.loc}, '
                f'protocol={self.protocol}, suffix={self.suffix})')

    def __repr__(self):
        return f'<{str(self)}>'

    def set(self, key, obj):
        """Pickles `obj` to a file determined by `key`. Returns None"""
        with open(self._make_path(key), 'wb') as file:
            pickle.dump(obj, file, protocol=self.protocol)

    def get(self, key):
        """Unpickles and returns the object at `key`.

        Raises
        ------
        KeyError if the key does not exist in the cache.
        """
        try:
            with open(self._make_path(key), 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            raise KeyError(f'{key} not in cache')

    def remove(self, key):
        """Deletes the file (hence object) at `key`

        Raises
        ------
        KeyError if the key does not exist in the cache.
        """
        try:
            path = self._make_path(key)
            path.unlink()
        except FileNotFoundError:
            raise KeyError(f'{key} not in cache')

    def add(self, key, obj):
        """Appends an object to the file of pickles at `key`"""
        with open(self._make_path(key), 'ab') as file:
            pickle.dump(obj, file, protocol=self.protocol)

    def has(self, key):
        """Checks the cache for the existence of a key"""
        return key in self  # delegate to __contains__

    def set_stream(self, key, stream):
        """Flushes an iterator or other stream to a pickle file.

        Arguments
        ---------
        key : str
            Identifies the path where the stream where be flushed.
        stream : Iterator
            Any iterator which produces values.
        """
        with open(self._make_path(key), 'wb') as file:
            for item in stream:
                pickle.dump(item, file, protocol=self.protocol)

    def get_stream(self, key):
        """Yields values from a stream of pickles located by `key`.

        Yields
        ------
        Successive objects from the pickle stream at `key`.

        Raises
        ------
        KeyError if the key does not exist in the cache.
        """
        try:
            with open(self._make_path(key), 'rb') as file:
                while True:
                    try:
                        x = pickle.load(file)
                    except EOFError:
                        break
                    else:
                        yield x
        except FileNotFoundError:
            raise KeyError(f'{key} not in cache')

    def _make_path(self, key):
        """Utility for transforming a key into a filepath"""
        return (self.loc/key).resolve().with_suffix(self.suffix)

    @contextlib.contextmanager
    def relocate(self, temp_loc, exist_ok=True):
        """
        Context manager which temporarily resets the base location of the
        cache. Useful for creating subdirectory structures in the cache or in
        looping constructs.
        """
        temp_loc = pathlib.Path(temp_loc).resolve()
        temp_loc.mkdir(exist_ok=exist_ok)
        prev_loc = self.loc
        self.loc = temp_loc
        yield self
        self.loc = prev_loc
