import pytest

from schedsim.common import TaskGraph
from schedsim.communication import SimpleNetModel
from schedsim.schedulers import AllOnOneScheduler, DoNothingScheduler, StaticScheduler, SchedulerBase
from schedsim.simulator import TaskAssignment
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


def test_task_zerocost():
    test_graph = TaskGraph()
    a = test_graph.new_task("A", duration=1, output_size=100)
    b = test_graph.new_task("B", duration=1, output_size=50)
    c = test_graph.new_task("C", duration=8)
    c.add_inputs((a, b))

    d = test_graph.new_task("D", duration=1, outputs=[0])
    e = test_graph.new_task("E", duration=1, outputs=[0])
    e.add_input(d)

    class Scheduler(StaticScheduler):
        def static_schedule(self):
            workers = self.simulator.workers
            tasks = self.simulator.task_graph.tasks
            return [
                TaskAssignment(workers[0], tasks[0]),
                TaskAssignment(workers[1], tasks[1]),
                TaskAssignment(workers[2], tasks[3]),
                TaskAssignment(workers[0], tasks[4]),
                TaskAssignment(workers[2], tasks[2])
            ]

    scheduler = Scheduler()
    do_sched_test(test_graph, 3, scheduler, SimpleNetModel(bandwidth=2))


def test_scheduling_time():
    test_graph = TaskGraph()
    a = test_graph.new_task("A", duration=3, output_size=1)
    b = test_graph.new_task("B", duration=1, output_size=1)
    c = test_graph.new_task("C", duration=1, output_size=1)
    d = test_graph.new_task("D", duration=1, output_size=1)

    b.add_input(a)
    c.add_input(b)
    d.add_input(c)

    times = []

    class Scheduler(SchedulerBase):
        def schedule(self, new_ready, new_finished):
            times.append(self.simulator.env.now)
            workers = self.simulator.workers
            return [
                TaskAssignment(workers[0], t)
                for t in new_ready
            ]

    scheduler = Scheduler()
    simulator = do_sched_test(
            test_graph, 1, scheduler,
            SimpleNetModel(bandwidth=2),
            scheduling_time=2, return_simulator=True)

    assert times == [0, 5, 8, 11, 14]
    assert simulator.task_info(a).end_time == 5
    assert simulator.task_info(b).end_time == 8
    assert simulator.task_info(c).end_time == 11
    assert simulator.task_info(d).end_time == 14
