
from . import Simulator, Worker
from .taskgraph import TaskGraph
from .schedulers import DoNothingScheduler

from .test_utils import do_sched_test

import pytest


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