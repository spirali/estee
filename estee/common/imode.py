import numpy as np


def _set_consts(graph, duration, size):
    for t in graph.tasks:
        t.expected_duration = duration

    for o in graph.outputs:
        o.expected_size = size


def process_imode_exact(graph):
    for t in graph.tasks.values():
        t.expected_duration = t.duration

    for o in graph.objects.values():
        o.expected_size = o.size


def process_imode_blind(graph):
    _set_consts(graph, None, None)


def process_imode_user(graph):
    durations = np.array([t.duration for t in graph.tasks.values()])
    sizes = np.array([o.size for o in graph.outputs])
    duration = durations.mean() if graph.tasks else 0
    size = sizes.mean() if graph.outputs else 0

    for t in graph.tasks:
        if t.expected_duration is None:
            t.expected_duration = duration

    for o in graph.outputs:
        if o.expected_size is None:
            o.expected_size = size


def process_imode_mean(graph):
    durations = np.array([t.duration for t in graph.tasks])
    sizes = np.array([o.size for o in graph.outputs])
    _set_consts(graph,
                durations.mean() if graph.tasks else 0,
                sizes.mean() if graph.outputs else 0)
