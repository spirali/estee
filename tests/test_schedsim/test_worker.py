import pytest

from schedsim.worker import Worker
from schedsim.communication import SimpleNetModel
from schedsim.schedulers import AllOnOneScheduler, DoNothingScheduler, SchedulerBase
from schedsim.simulator import TaskAssignment
from schedsim.common import TaskGraph
from .test_utils import do_sched_test, fixed_scheduler

import random

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