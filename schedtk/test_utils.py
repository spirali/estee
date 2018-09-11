
from . import Simulator, Worker
from .taskgraph import TaskGraph
from .connectors import InstantConnector
import pytest


@pytest.fixture
def plan1():
    """
        a1/2 a2/3
        |    |
        a3/2 | a4/1
        |\  / /|
        | a5/1 a6/6 a7/1
        |  \   |   /
        |   \  |  /
         \--- a8/1
    """
    task_graph = TaskGraph()

    a1, a2, a3, a4, a5, a6, a7, a8 = [
        task_graph.new_task("a{}".format(i + 1), duration)
        for i, duration in enumerate([2, 3, 2, 1, 1, 6, 1, 1])
    ]

    a3.add_input(a1)
    a5.add_inputs([a3, a2, a4])
    a6.add_input(a4)
    a8.add_inputs([a5, a6, a7, a3])

    task_graph.validate()

    return task_graph


def do_sched_test(task_graph, n_workers, scheduler, connector=None):

    if connector is None:
        connector = InstantConnector()

    workers = [Worker() for _ in range(n_workers)]
    simulator = Simulator(task_graph, workers, scheduler, connector)
    return simulator.run()
