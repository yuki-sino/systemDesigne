
class Record(object):
    def __init__(self, keys):
        self._data = {}

        for key in keys:
            self._data[key] = None

    def set(self, key, value):
        if key not in self._data:
            raise KeyError(f"Unknown key: {key}")

        self._data[key] = value

    def get(self, key):
        if key not in self._data:
            raise KeyError(f"Unknown key: {key}")

        return self._data[key]