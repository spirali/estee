import pytest

from .schedulers import AllOnOneScheduler, DoNothingScheduler, SchedulerBase
from .simulator import TaskAssignment
from .taskgraph import TaskGraph
from .test_utils import do_sched_test


def test_simulator_empty_task_graph():

    task_graph = TaskGraph()

    scheduler = DoNothingScheduler()
    assert do_sched_test(task_graph, 1, scheduler) == 0


def test_simulator_no_events():

    task_graph = TaskGraph()
    task_graph.new_task("A", duration=1)

    scheduler = DoNothingScheduler()
    with pytest.raises(RuntimeError):
        do_sched_test(task_graph, 1, scheduler)


def test_simulator_cpus1():
    test_graph = TaskGraph()
    test_graph.new_task("A", duration=1, cpus=1)
    test_graph.new_task("B", duration=1, cpus=2)

    scheduler = AllOnOneScheduler()
    assert do_sched_test(test_graph, [2], scheduler) == 2

    scheduler = AllOnOneScheduler()
    assert do_sched_test(test_graph, [3], scheduler) == 1


def test_simulator_cpus2():
    test_graph = TaskGraph()
    test_graph.new_task("A", duration=1, cpus=1)
    test_graph.new_task("B", duration=2, cpus=1)
    test_graph.new_task("C", duration=1, cpus=1)

    scheduler = AllOnOneScheduler()
    assert do_sched_test(test_graph, [2], scheduler) == 2

    scheduler = AllOnOneScheduler()
    assert do_sched_test(test_graph, [3], scheduler) == 2


def test_simulator_cpus3():
    test_graph = TaskGraph()
    test_graph.new_task("A", duration=3, cpus=1)
    test_graph.new_task("B", duration=1, cpus=2)
    test_graph.new_task("C", duration=1, cpus=1)
    test_graph.new_task("D", duration=1, cpus=3)
    test_graph.new_task("E", duration=1, cpus=1)
    test_graph.new_task("F", duration=1, cpus=1)

    scheduler = AllOnOneScheduler()
    assert do_sched_test(test_graph, [3], scheduler) == 4

    scheduler = AllOnOneScheduler()
    assert do_sched_test(test_graph, [4], scheduler) == 3

    scheduler = AllOnOneScheduler()
    assert do_sched_test(test_graph, [5], scheduler) == 3


def test_worker_freecpus():
    test_graph = TaskGraph()
    test_graph.new_task("A", duration=10, cpus=2)
    test_graph.new_task("B", duration=8, cpus=3)
    c = test_graph.new_task("C", duration=1, cpus=1)
    d = test_graph.new_task("D", duration=3, cpus=3)
    d.add_input(c)

    free_cpus = []

    class Scheduler(SchedulerBase):
        def schedule(self, new_ready, new_finished):
            worker = self.simulator.workers[0]
            free_cpus.append(worker.free_cpus)
            return (TaskAssignment(worker, t) for t in new_ready)

    scheduler = Scheduler()
    do_sched_test(test_graph, [10], scheduler)
    assert free_cpus == [10, 5, 5, 8, 10]
