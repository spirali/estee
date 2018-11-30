import numpy as np


def _set_consts(graph, duration, size):
    for t in graph.tasks:
        t.expected_duration = duration

    for o in graph.outputs:
        o.expected_size = size


def process_imode_exact(graph):
    for t in graph.tasks:
        t.expected_duration = t.duration

    for o in graph.outputs:
        o.expected_size = o.size


def process_imode_blind(graph):
    _set_consts(graph, None, None)


def process_imode_user(graph):
    # Do nothing
    pass


def process_imode_mean(graph):
    durations = np.array([t.duration for t in graph.tasks])
    sizes = np.array([o.size for o in graph.outputs])
    _set_consts(graph, durations.mean(), sizes.mean())
