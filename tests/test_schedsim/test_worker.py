import random

import pytest

from schedsim.common import TaskGraph
from schedsim.communication import SimpleNetModel
from schedsim.schedulers import SchedulerBase
from schedsim.simulator import TaskAssignment
from schedsim.worker import Worker
from .test_utils import do_sched_test, fixed_scheduler


def test_worker_max_downloads_per_worker():
    g = TaskGraph()

    a = g.new_task("a", duration=0, outputs=[1, 1, 1, 1])
    b = g.new_task("b", duration=0)
    b.add_inputs(a.outputs)

    s = fixed_scheduler([
        (0, a, 0),
        (1, b, 0),
    ])

    assert do_sched_test(g, [1, 1], s, SimpleNetModel()) == 2
    assert do_sched_test(
        g, [Worker(), Worker(max_downloads_per_worker=1)], s, SimpleNetModel()) == 4
    assert do_sched_test(
        g, [Worker(), Worker(max_downloads_per_worker=2)], s, SimpleNetModel()) == 2
    assert do_sched_test(
        g, [Worker(), Worker(max_downloads_per_worker=3)], s, SimpleNetModel()) == 2
    assert do_sched_test(
        g, [Worker(), Worker(max_downloads_per_worker=4)], s, SimpleNetModel()) == 1
    assert do_sched_test(
        g, [Worker(), Worker(max_downloads_per_worker=5)], s, SimpleNetModel()) == 1


def test_worker_max_downloads_global():
    g = TaskGraph()

    a1, a2, a3, a4 = [g.new_task("a{}".format(i), duration=0, output_size=1)
                      for i in range(4)]
    b = g.new_task("b", duration=0)
    b.add_inputs([a1, a2, a3, a4])

    s = fixed_scheduler([
        (0, a1, 0),
        (1, a2, 0),
        (2, a3, 0),  # worker is 2!
        (2, a4, 0),  # worker is also 2!
        (4, b, 0),
    ])

    def make_workers(max_downloads, max_downloads_per_worker=2):
        return [
            Worker(), Worker(), Worker(), Worker(),
            Worker(max_downloads=max_downloads,
                   max_downloads_per_worker=max_downloads_per_worker)
        ]

    assert do_sched_test(
        g, make_workers(1), s, SimpleNetModel()) == pytest.approx(4)
    assert do_sched_test(
        g, make_workers(2), s, SimpleNetModel()) == pytest.approx(2)
    assert do_sched_test(
        g, make_workers(3), s, SimpleNetModel()) == pytest.approx(2)
    assert do_sched_test(
        g, make_workers(3), s, SimpleNetModel()) == pytest.approx(2)
    assert do_sched_test(
        g, make_workers(4), s, SimpleNetModel()) == pytest.approx(1)
    assert do_sched_test(
        g, make_workers(4, 1), s, SimpleNetModel()) == pytest.approx(2)
    assert do_sched_test(
        g, make_workers(3, 1), s, SimpleNetModel()) == pytest.approx(2)


def test_worker_download_priorities():
    SIZE = 20
    g = TaskGraph()

    a = g.new_task("a", duration=0, outputs=[1] * SIZE)
    b = [g.new_task("b{}".format(i), duration=0) for i in range(SIZE)]
    for i, t in enumerate(b):
        t.add_input(a.outputs[i])

    r = random.Random(42)
    priorities = list(range(SIZE))
    r.shuffle(priorities)

    s = fixed_scheduler(
        [(0, a, 0)] + [(1, t, p) for t, p in zip(b, priorities)])

    w = [Worker(), Worker(max_downloads=2, max_downloads_per_worker=2)]
    simulator = do_sched_test(g, w, s, SimpleNetModel(), return_simulator=True)

    for t, p in zip(b, priorities):
        assert simulator.task_info(t).end_time == pytest.approx((SIZE - p - 1) // 2 + 1)


def test_worker_execute_priorities():
    SIZE = 20
    g = TaskGraph()
    b = [g.new_task("b{}".format(i), duration=1) for i in range(SIZE)]

    r = random.Random(42)
    priorities = list(range(SIZE))
    r.shuffle(priorities)

    s = fixed_scheduler(
        [(0, t, p) for t, p in zip(b, priorities)])
    simulator = do_sched_test(g, [Worker(cpus=2)], s, return_simulator=True)

    for t, p in zip(b, priorities):
        assert simulator.task_info(t).end_time == pytest.approx((SIZE - p - 1) // 2 + 1)


def test_worker_freecpus():
    test_graph = TaskGraph()
    test_graph.new_task("A", duration=10, cpus=2, output_size=1)
    test_graph.new_task("B", duration=8, cpus=3, output_size=1)
    c = test_graph.new_task("C", duration=1, cpus=1, output_size=1)
    d = test_graph.new_task("D", duration=3, cpus=3, output_size=1)
    d.add_input(c)

    free_cpus = []

    class Scheduler(SchedulerBase):
        def schedule(self, new_ready, new_finished):
            worker = self.simulator.workers[0]
            free_cpus.append(worker.free_cpus)
            return [TaskAssignment(worker, t) for t in new_ready]

    scheduler = Scheduler()
    do_sched_test(test_graph, [10], scheduler)
    assert free_cpus == [10, 5, 5, 8, 10]


def test_worker_running_tasks():
    test_graph = TaskGraph()
    test_graph.new_task("X", duration=10)
    a = test_graph.new_task("A", duration=1, output_size=1)
    b = test_graph.new_task("B", duration=8, output_size=1)
    b.add_input(a)

    remaining_times = []

    class Scheduler(SchedulerBase):
        def __init__(self):
            self.scheduled = False

        def schedule(self, new_ready, new_finished):
            workers = self.simulator.workers

            remaining_times.append([[t.remaining_time(self.simulator.env.now)
                                     for t
                                     in w.running_tasks.values()]
                                    for w in workers])

            if not self.scheduled:
                tasks = self.simulator.task_graph.tasks
                self.scheduled = True
                return [
                    TaskAssignment(workers[0], tasks[0]),
                    TaskAssignment(workers[1], tasks[1]),
                    TaskAssignment(workers[1], tasks[2])
                ]
            else:
                return ()

    scheduler = Scheduler()
    do_sched_test(test_graph, 2, scheduler)
    assert remaining_times == [
        [[], []],
        [[9], []],
        [[1], []],
        [[], []]
    ]
