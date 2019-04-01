from typing import Union

import numpy as np

from .taskgraph import TaskGraph


def _set_consts(graph: TaskGraph, duration: Union[float, None], size: Union[float, None]):
    for t in graph.tasks.values():
        t.expected_duration = duration

    for o in graph.objects.values():
        o.expected_size = size


def process_imode_exact(graph: TaskGraph):
    for t in graph.tasks.values():
        t.expected_duration = t.duration

    for o in graph.objects.values():
        o.expected_size = o.size


def process_imode_blind(graph: TaskGraph):
    _set_consts(graph, None, None)


def process_imode_user(graph: TaskGraph):
    durations = np.array([t.duration for t in graph.tasks.values()])
    sizes = np.array([o.size for o in graph.objects.values()])
    duration = durations.mean() if graph.tasks else 0
    size = sizes.mean() if graph.objects else 0

    for t in graph.tasks.values():
        if t.expected_duration is None:
            t.expected_duration = duration

    for o in graph.objects.values():
        if o.expected_size is None:
            o.expected_size = size


def process_imode_mean(graph: TaskGraph):
    durations = np.array([t.duration for t in graph.tasks.values()])
    sizes = np.array([o.size for o in graph.objects.values()])
    _set_consts(graph,
                durations.mean() if graph.tasks else 0,
                sizes.mean() if graph.objects else 0)
