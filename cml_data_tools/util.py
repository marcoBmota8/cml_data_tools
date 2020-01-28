import pathlib
import pickle

__all__ = ['format_size', 'slug',
           'from_glob', 'from_files',
           'pickle_to_stream', 'unpickle_stream']


def format_size(num):
    """Format bytes in human readable SI units"""
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%sB" % (num, unit)
        num /= 1024.0
    return "%.1f%sB" % (num, 'Yi')


def slug(word):
    """Converts a mode to a friendlier repr for filenames"""
    return word.lower().strip().replace(' ', '-')


def from_glob(directory, glob='*.pkl'):
    """
    Generator over the pickle files in a given directory that match a glob.
    Each file may contain multiple pickles.
    """
    # Always glob over the paths deterministically
    for path in sorted(pathlib.Path(directory).glob(glob)):
        with path.open('rb') as file:
            yield from unpickle_stream(file)


def from_files(files):
    """
    Generator over the pickle files specified by an iterable of names.
    Each file may contain multiple pickles.
    """
    for name in sorted(files):
        with open(name, 'rb') as file:
            yield from unpickle_stream(file)


def pickle_to_stream(iterator, stream, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Pickles all items in an iterator to a file-like stream.
    Yields the byte offset of the pickled object in the stream.
    """
    for obj in iterator:
        offset = stream.tell()
        pickle.dump(obj, stream, protocol=protocol)
        yield offset


def unpickle_stream(stream):
    """Generator over all pickled objects in a file-like stream"""
    while True:
        try:
            yield pickle.load(stream)
        except EOFError:
            break
