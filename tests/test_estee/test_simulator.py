import itertools

import pytest

from estee.common import TaskGraph
from estee.simulator import SimpleNetModel, Worker, TaskAssignment
from estee.schedulers import AllOnOneScheduler, DoNothingScheduler, SchedulerBase, \
    StaticScheduler
from estee.simulator.utils import estimate_schedule
from .test_utils import do_sched_test

from .test_utils import fixed_scheduler

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
            if not self.workers or not self.task_graph.tasks:
                return
            tasks = self.task_graph.tasks
            self.assign(self.workers[0], tasks[0])
            self.assign(self.workers[1], tasks[1])
            self.assign(self.workers[2], tasks[3])
            self.assign(self.workers[0], tasks[4])
            self.assign(self.workers[2], tasks[2])

    scheduler = Scheduler("x", "0")
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
        def schedule(self, update):
            if not self.task_graph.tasks:
                return
            simulator = self._simulator
            times.append(simulator.env.now)
            for t in update.new_ready_tasks:
                self.assign(self.workers[0], t)

    scheduler = Scheduler("x", "0")
    simulator = do_sched_test(
            test_graph, 1, scheduler,
            SimpleNetModel(bandwidth=2),
            scheduling_time=2, return_simulator=True)
    runtime_state = simulator.runtime_state

    assert times == [0, 5, 8, 11, 14]
    assert runtime_state.task_info(a).end_time == 5
    assert runtime_state.task_info(b).end_time == 8
    assert runtime_state.task_info(c).end_time == 11
    assert runtime_state.task_info(d).end_time == 14


def test_estimate_schedule(plan1):
    netmodel = SimpleNetModel(1)
    workers = [Worker(cpus=4) for _ in range(4)]

    schedule = [TaskAssignment(w, t) for (w, t) in zip(itertools.cycle(workers), plan1.tasks)]

    assert estimate_schedule(schedule, plan1, netmodel) == 16


def test_estimate_schedule_zero_expected_time(plan1):
    netmodel = SimpleNetModel(1)
    workers = [Worker(cpus=4) for _ in range(4)]

    plan1.tasks[1].expected_duration = 0
    plan1.tasks[5].expected_duration = 0

    schedule = [TaskAssignment(w, t) for (w, t) in zip(itertools.cycle(workers), plan1.tasks)]

    assert estimate_schedule(schedule, plan1, netmodel) == 15


def test_simulator_reschedule_no_download():
    test_graph = TaskGraph()
    a1 = test_graph.new_task("A1", duration=10, cpus=1)
    a2 = test_graph.new_task("A2", duration=10, cpus=1)

    b = test_graph.new_task("B", duration=1, cpus=1)
    c = test_graph.new_task("C", duration=1, cpus=1)
    d = test_graph.new_task("D", duration=1, cpus=1)
    e = test_graph.new_task("E", duration=1, cpus=1)

    assignments = [[
        (0, a1, 0),
        (1, a2, 0),
        (0, b, 10),
        (0, c, 9),
        (0, d, 8),
        (0, e, 7),
    ], [
        (1, a1, 0),
    ], [
        (None, a1, 0)
    ], [
        (0, a1, 0)
    ], [
        (2, a1, 0)
    ]]

    simulator = do_sched_test(test_graph, [1, 1, 1],
                                  fixed_scheduler(assignments, steps=True, reassigning=True),
                              trace=True, return_simulator=True)
    assert simulator.env.now == 14
    assert simulator.runtime_state.task_info(a1).assigned_workers == [simulator.workers[0]]

    # Test the same without allowed reassigning
    with pytest.raises(Exception):
        do_sched_test(test_graph, [1, 1, 1],
                      fixed_scheduler(assignments, steps=True, reassigning=False))


def test_simulator_reschedule_too_late():
    test_graph = TaskGraph()

    source = test_graph.new_task("S", duration=0, cpus=1, output_size=10)
    a1 = test_graph.new_task("A1", duration=10, cpus=1)
    b = test_graph.new_task("B", duration=1, cpus=1)

    assignments = [[
        (0, source, 100),
        (1, a1, 0),
        (0, b, 10),
    ], [
        (0, a1, 0),
    ]]

    simulator = do_sched_test(test_graph, [1, 1, 1],
                             fixed_scheduler(assignments, steps=True, reassigning=True),
                             trace=True, return_simulator=True, netmodel=SimpleNetModel(1))
    assert simulator.env.now == 10
    assert simulator.runtime_state.task_info(a1).assigned_workers == [simulator.workers[1]]


def test_simulator_reschedule_running_download():
    test_graph = TaskGraph()

    source = test_graph.new_task("S", duration=0, cpus=1, output_size=10)

    a1 = test_graph.new_task("A1", duration=10, cpus=1)
    b = test_graph.new_task("B", duration=1, cpus=1)

    a1.add_input(source)

    assignments = [[
        (0, source, 100),
        (1, a1, 0),
        (0, b, 10),
    ], [], [
        (2, a1, 0),
    ]]

    simulator = do_sched_test(test_graph, [1, 1, 1],
                             fixed_scheduler(assignments, steps=True, reassigning=True),
                             trace=True, return_simulator=True, netmodel=SimpleNetModel(1))
    assert simulator.env.now == 21
    assert simulator.runtime_state.task_info(a1).assigned_workers == [simulator.workers[2]]


def test_simulator_reschedule_scheduled_download():
    test_graph = TaskGraph()

    s = [test_graph.new_task("S{}".format(i), duration=0, cpus=1, output_size=10)
         for i in range(10)]

    a1 = test_graph.new_task("A1", duration=10, cpus=1)
    b = test_graph.new_task("B", duration=1, cpus=1)
    c = test_graph.new_task("C", duration=2, cpus=1)
    a1.add_inputs(s)

    assignments = [
       [(0, x, 100) for x in s] + [
        (1, a1, 0),
        (0, b, 10),
        (1, c, 10),
    ], [], [
        (2, a1, 0),
    ]]

    scheduler = fixed_scheduler(assignments, steps=True, reassigning=True)
    scheduler._disable_cleanup = True
    simulator = do_sched_test(test_graph, [1, 1, 1],
                              scheduler,
                              trace=True, return_simulator=True, netmodel=SimpleNetModel(1))
    assert simulator.env.now > 40

    available = set()
    for x in s:
        available.add(frozenset(w.worker_id for w in scheduler.task_graph.objects[x.output.id].availability))
    assert frozenset({0, 1, 2}) in available
    assert frozenset({0, 2}) in available
    assert simulator.runtime_state.task_info(a1).assigned_workers == [simulator.workers[2]]


def test_simulator_reassign_failed():
    test_graph = TaskGraph()

    a1 = test_graph.new_task("A1", duration=5, cpus=1)
    a2 = test_graph.new_task("A1", duration=3, cpus=1)

    test_update = []

    class Scheduler(SchedulerBase):

        def start(self):
            self.step = 0
            self.workers = {}
            return super().start()

        def schedule(self, update):
            if not self.task_graph.tasks:
                return
            self.step += 1
            if self.step == 1:
                self.assign(self.workers[0], a1)
                self.assign(self.workers[1], a2)
            elif self.step == 2:
                self.assign(self.workers[2], a1)
            elif self.step == 3:
                test_update.append(update)

    scheduler = Scheduler("test", "0", True)
    simulator = do_sched_test(test_graph, [1, 1, 1],
                              scheduler,
                              trace=True, netmodel=SimpleNetModel(1))

    assert test_update[0].reassign_failed[0].id == a1.id
    assert test_update[0].reassign_failed[0].scheduled_worker.worker_id == 0