import pathlib
import pickle


class PickleCache:
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

    def set(self, key, obj):
        with open(self._make_path(key), 'wb') as file:
            pickle.dump(obj, file, protocol=self.protocol)

    def get(self, key):
        try:
            with open(self._make_path(key), 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            raise KeyError

    def add(self, key, obj):
        with open(self._make_path(key), 'ab') as file:
            pickle.dump(obj, file, protocol=self.protocol)

    def has(self, key):
        # delegates to __contains__
        return key in self

    def set_stream(self, key, stream):
        with open(self._make_path(key), 'wb') as file:
            for item in stream:
                pickle.dump(item, file, protocol=self.protocol)

    def get_stream(self, key):
        with open(self._make_path(key), 'rb') as file:
            while True:
                try:
                    x = pickle.load(file)
                except EOFError:
                    break
                else:
                    yield x

    def _make_path(self, key):
        return (self.loc/key).with_suffix(self.suffix)
