import collections
import itertools


def flat_list(list_of_lists):
    return list(itertools.chain(*list_of_lists))


class LruCache:
    def __init__(self, size):
        self.size = size
        self.cache = collections.OrderedDict()

    def get(self, key):
        value = self.cache.pop(key, None)
        if value is None:
            return None
        self.cache[key] = value
        return value

    def set(self, key, value):
        self.cache[key] = value
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)