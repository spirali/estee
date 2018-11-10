import itertools


def flat_list(list_of_lists):
    return list(itertools.chain(*list_of_lists))