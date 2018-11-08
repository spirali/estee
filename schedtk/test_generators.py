

from .taskgraph import TaskGraph
from .generators import random_dependencies, random_levels


def test_random_dependencies():
    graph = TaskGraph()
    random_dependencies(10, 0.2, lambda: graph.new_task(output_size=1))

    assert graph.task_count == 10
    graph.validate()


def test_random_levels():
    graph = TaskGraph()
    random_levels([3, 10, 5, 1], [0, 3, 2, 3], lambda: graph.new_task(output_size=1))

    graph.validate()
    assert graph.task_count == 19
    assert len(list(graph.arcs)) == 43
