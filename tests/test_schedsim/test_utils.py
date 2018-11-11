import pytest

from schedsim.common import TaskGraph
from schedsim.communication import InstantConnector
from schedsim.simulator import Simulator
from schedsim.worker import Worker


@pytest.fixture
def plan1():
    """
        a1/1 a2/3/3
        |    |
        a3/1 | a4/1/6
        |\  / /|
        o o |/ |
        | a5/1 a6/6 a7/2
        |  \   |   /
        |   \  |  /
         \--- a8/1
    """
    task_graph = TaskGraph()

    a1, a2, a3, a4, a5, a6, a7, a8 = [
        task_graph.new_task("a{}".format(i + 1), duration=duration, outputs=outputs)
        for i, (duration, outputs) in enumerate([
            (2, [1]),  # a1
            (3, [3]),  # a2
            (2, [1, 1]),  # a3
            (1, [6]),  # a4
            (1, [1]),  # a5
            (6, [1]),  # a6
            (1, [2]),  # a7
            (1, [])   # a8
        ])
    ]

    a3.add_input(a1)
    a5.add_inputs([a3.outputs[0], a2, a4])
    a6.add_input(a4)
    a8.add_inputs([a5, a6, a7, a3.outputs[1]])

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


def do_sched_test(task_graph, workers, scheduler, connector=None, report=None):

    if connector is None:
        connector = InstantConnector()

    if isinstance(workers, int):
        workers = [Worker() for _ in range(workers)]
    else:
        workers = [Worker(cpus=cpus) for cpus in workers]

    simulator = Simulator(task_graph, workers, scheduler, connector, trace=bool(report))
    result = simulator.run()
    if report:
        simulator.make_trace_report(report)
    return result


def task_by_name(plan, name):
    return [t for t in plan.tasks if t.name == name][0]
