
from . import Simulator, Worker
from .taskgraph import TaskGraph
from .connectors import InstantConnector
import pytest


@pytest.fixture
def plan1():
    """
        a1/1 a2/3/3
        |    |
        a3/1 | a4/1/6
        |\  / /|
        | a5/1 a6/6 a7/2
        |  \   |   /
        |   \  |  /
         \--- a8/1
    """
    task_graph = TaskGraph()

    a1, a2, a3, a4, a5, a6, a7, a8 = [
        task_graph.new_task("a{}".format(i + 1), duration, size)
        for i, (duration, size) in enumerate([
            (2, 1),  # a1
            (3, 3),  # a2
            (2, 1),  # a3
            (1, 6),  # a4
            (1, 1),  # a5
            (6, 1),  # a6
            (1, 2),  # a7
            (1, 1)   # a8
        ])
    ]

    a3.add_input(a1)
    a5.add_inputs([a3, a2, a4])
    a6.add_input(a4)
    a8.add_inputs([a5, a6, a7, a3])

    task_graph.validate()

    return task_graph


def plan_reverse_cherry1():
    """
        a1/10/1  a2/10/1
          \     /
           \   /
             a3
    """
    task_graph = TaskGraph()
    a1 = task_graph.new_task("a1", 10, 1)
    a2 = task_graph.new_task("a2", 10, 1)
    a3 = task_graph.new_task("a3", 1)

    a3.add_input(a1)
    a3.add_input(a2)
    return task_graph


def do_sched_test(task_graph, n_workers, scheduler, connector=None, report=None):

    if connector is None:
        connector = InstantConnector()

    workers = [Worker() for _ in range(n_workers)]
    simulator = Simulator(task_graph, workers, scheduler, connector, trace=bool(report))
    result = simulator.run()
    if report:
        simulator.make_trace_report(report)
    return result
