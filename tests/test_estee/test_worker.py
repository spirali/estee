import random

import pytest

from estee.common import TaskGraph
from estee.schedulers import SchedulerBase
from estee.simulator import SimpleNetModel, Worker
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


def test_worker_download_priorities1():
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

    runtime_state = simulator.runtime_state
    for t, p in zip(b, priorities):
        assert runtime_state.task_info(t).end_time == pytest.approx((SIZE - p - 1) // 2 + 1)


def test_worker_download_priorities2():
    g = TaskGraph()

    a = g.new_task("a", duration=0, outputs=[2, 2])
    b = g.new_task("b", duration=4, output_size=2)
    d = g.new_task("d", duration=1)

    a2 = g.new_task("a2", duration=1)
    a2.add_input(a.outputs[0])

    b2 = g.new_task("b", duration=1, output_size=1)
    b2.add_input(a.outputs[1])
    b2.add_input(b)

    s = fixed_scheduler(
        [(0, a),
         (0, b),
         (1, d, 3),
         (1, a2, 1),
         (1, b2, 2)
         ])

    w = [Worker(cpus=3), Worker(cpus=1, max_downloads=1)]
    simulator = do_sched_test(g, w, s, SimpleNetModel(), return_simulator=True)

    assert simulator.runtime_state.task_info(a2).end_time == pytest.approx(3)
    assert simulator.runtime_state.task_info(b2).end_time == pytest.approx(7)


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

    runtime_state = simulator.runtime_state
    for t, p in zip(b, priorities):
        assert runtime_state.task_info(t).end_time == pytest.approx((SIZE - p - 1) // 2 + 1)


def test_worker_priority_block():
    g = TaskGraph()

    a = g.new_task("a", duration=1)
    b = g.new_task("b", duration=1, cpus=3)
    c = g.new_task("c", duration=1)

    s = fixed_scheduler(
        [(0, a, 3),
         (0, b, 2),
         (0, c, 1)
         ])

    w = [Worker(cpus=3)]
    simulator = do_sched_test(g, w, s, SimpleNetModel(), return_simulator=True)
    runtime_state = simulator.runtime_state

    assert runtime_state.task_info(a).end_time == pytest.approx(1)
    assert runtime_state.task_info(b).end_time == pytest.approx(2)
    assert runtime_state.task_info(c).end_time == pytest.approx(1)

    s = fixed_scheduler(
        [(0, a, 3),
         (0, b, 2, 2),
         (0, c, 1)
         ])

    w = [Worker(cpus=3)]
    simulator = do_sched_test(g, w, s, SimpleNetModel(), return_simulator=True)
    runtime_state = simulator.runtime_state

    assert runtime_state.task_info(a).end_time == pytest.approx(1)
    assert runtime_state.task_info(b).end_time == pytest.approx(2)
    assert runtime_state.task_info(c).end_time == pytest.approx(3)


def test_worker_freecpus():
    test_graph = TaskGraph()
    test_graph.new_task("A", duration=10, cpus=2, output_size=1)
    test_graph.new_task("B", duration=8, cpus=3, output_size=1)
    c = test_graph.new_task("C", duration=1, cpus=1, output_size=1)
    d = test_graph.new_task("D", duration=3, cpus=3, output_size=1)
    d.add_input(c)

    free_cpus = []

    class Scheduler(SchedulerBase):
        def schedule(self, update):
            if not self.task_graph.tasks:
                return
            worker = self._simulator.workers[0]
            free_cpus.append(worker.free_cpus)
            for t in update.new_ready_tasks:
                self.assign(self.workers[worker.id], t)

    scheduler = Scheduler("x", "0")
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
        scheduled = False

        def schedule(self, update):
            if not self.task_graph.tasks:
                return

            simulator = self._simulator
            remaining_times.append([[t.remaining_time(simulator.env.now)
                                     for t
                                     in w.running_tasks.values()]
                                    for w in simulator.workers])

            if not self.scheduled:
                tasks = self.task_graph.tasks
                self.scheduled = True
                self.assign(self.workers[0], tasks[0])
                self.assign(self.workers[1], tasks[1])
                self.assign(self.workers[1], tasks[2])
            else:
                return ()

    scheduler = Scheduler("x", "0")
    do_sched_test(test_graph, 2, scheduler)
    assert remaining_times == [
        [[], []],
        [[9], []],
        [[1], []],
        [[], []]
    ]


def test_more_outputs_from_same_source():
    test_graph = TaskGraph()
    a = test_graph.new_task("A", duration=1, outputs=[1, 1, 1])
    b = test_graph.new_task("B", duration=1)
    b.add_input(a.outputs[0])
    b.add_input(a.outputs[2])

    s = fixed_scheduler([
        (0, a, 0),
        (0, b, 0),
    ])

    assert do_sched_test(test_graph, [1], s) == 2
