import lzma
import mmap
import pathlib
import pickle

import numpy

__all__ = ['MMapDataset']


slice_type = type(slice(0))


class MMapDataset:
    """Represents a large dataset on disk accessed via mmap.

    Indexing returns instances (cf. pf.instances).
    Slices and iterables of indices return lists of instances.

    Class instances are pickleable (they may be used in a multiprocessing
    environment).

    Arguments
    ---------
    data_filename : str or os.PathLike
    addr_filename : str or os.PathLike
        Data is stored as triplets of (id, X, y) that have been pickled, lzma
        compressed, and written to the file given in `dat_file`.  `idx_file` is
        a .npy file off integer byte offsets giving the location of each datum
        stored in `dat_file`.

    repeat : int
        Number of times to iterate over the underlying dataset

    Attributes
    ----------
    file : file
        The file backing the memory map

    buff : mmap
        A memory map

    addr : numpy.ndarray
        A 1d ndarray which contains byte offsets into buff

    n_instances : int
        The total number of instances represented in the file
    """
    def __init__(self, data_filename, addr_filename=None, repeat=1):
        self.data_fn = pathlib.Path(data_filename)
        self.addr_fn = (pathlib.Path(addr_filename)
                        if addr_filename is not None
                        else self.data_fn.with_suffix('.idx.npy'))
        if not repeat > 0:
            raise ValueError('repeat must be non-negative')
        self.repeat = repeat
        self._open = False
        self._init_buffers()

    def _init_buffers(self):
        if self._open:
            return
        self.addr = numpy.load(self.addr_fn)
        self.file = open(self.data_fn, 'rb')
        self.buff = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.n_instances = len(self.addr)
        self._open = True

    def __len__(self):
        return self.n_instances * self.repeat

    def __getitem__(self, idx):
        if not self._open:
            raise ValueError(f'Cannot index closed {self.__class__.__name__}')

        if isinstance(idx, (tuple, list, numpy.ndarray)):
            return self._get_batch(idx)

        if isinstance(idx, slice_type):
            start, stop, stride = idx.indices(self.n_instances)
            return self._get_batch(range(start, stop, stride))

        idx = int(idx)
        # Addition is correct here... remember, idx is negative
        if idx < 0:
            idx = len(self) + idx

        # Old school iteration protocol. No need to define __iter__; iter(obj)
        # will return an iterator that will call __getitem__ with successive
        # indices until... this line. Then the IndexError gets caught and
        # transformed into a StopIteration.
        if idx >= len(self):
            raise IndexError(f'{self.__class__.__name__} index out of range')

        return self._get_instance(idx)

    def __del__(self):
        if hasattr(self, 'buff') or hasattr(self, 'file'):
            self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['file']
        del state['buff']
        del state['addr']
        state['_open'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_buffers()

    def _get_batch(self, batch_indices):
        insts = [self._get_instance(i) for i in batch_indices]
        return insts

    def _get_instance(self, idx):
        # Needed in case of repetition
        idx %= self.n_instances

        i = self.addr[idx]
        try:
            j = self.addr[idx + 1]
        except IndexError:
            j = None
        inst = self.buff[i:j]
        inst = lzma.decompress(inst)
        inst = pickle.loads(inst)
        return inst

    def close(self):
        self.buff.close()
        self.file.close()
        self._open = False

    def closed(self):
        return not self._open
