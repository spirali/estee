
from schedsim.schedulers import (AllOnOneScheduler, RandomScheduler, RandomGtScheduler,
                                 BlevelGtScheduler, RandomAssignScheduler, Camp2Scheduler)
from schedsim.schedulers.utils import compute_independent_tasks
from schedsim.communication import SimpleConnector
from schedsim.schedulers import (AllOnOneScheduler, BlevelGtScheduler, Camp2Scheduler,
    DLSScheduler, ETFScheduler, LASTScheduler, MCPScheduler,
    RandomAssignScheduler, RandomGtScheduler, RandomScheduler)

from schedsim.schedulers.utils import (compute_alap,
    compute_b_level, compute_independent_tasks, compute_t_level)
from .test_utils import do_sched_test, task_by_name


def test_scheduler_all_on_one(plan1):

    scheduler = AllOnOneScheduler()
    assert 17 == do_sched_test(plan1, 1, scheduler)

    scheduler = AllOnOneScheduler()
    assert 17 == do_sched_test(plan1, 3, scheduler)


def test_scheduler_all(plan1):

    scheduler = AllOnOneScheduler()
    assert 17 == do_sched_test(plan1, 1, scheduler)


def test_scheduler_random(plan1):
    # 1w, instant
    assert 17 == do_sched_test(plan1, 1, RandomScheduler())

    # 2w, instant
    for _ in range(50):
        assert 9 <= do_sched_test(plan1, 2, RandomScheduler()) <= 12

    # 3w, instant
    for _ in range(50):
        assert 8 <= do_sched_test(plan1, 3, RandomScheduler()) <= 9

    # 2w, simple
    for _ in range(50):
        assert 13 <= do_sched_test(plan1, 2, RandomScheduler(), SimpleConnector()) <= 20


def test_scheduler_random_gt(plan1):

    # 2w, simple
    for _ in range(50):
        assert 13 <= do_sched_test(plan1, 2, RandomGtScheduler(), SimpleConnector()) <= 19


def test_scheduler_blevel_gt(plan1):

    # 2w, simple
    for _ in range(50):
        assert do_sched_test(plan1, 2, BlevelGtScheduler(), SimpleConnector()) in [13, 16]


def test_scheduler_random_assign(plan1):
    for _ in range(50):
        assert 10 <= do_sched_test(plan1, 2, RandomAssignScheduler(), SimpleConnector()) <= 22
        assert 9 <= do_sched_test(plan1, 3, RandomAssignScheduler(), SimpleConnector()) <= 22


def test_scheduler_camp(plan1):
    for _ in range(10):
        assert 10 <= do_sched_test(plan1, 2, Camp2Scheduler(), SimpleConnector()) <= 18


def test_scheduler_dls(plan1):
    assert do_sched_test(plan1, 2, DLSScheduler(), SimpleConnector()) == 17


def test_scheduler_last(plan1):
    assert do_sched_test(plan1, 2, LASTScheduler(), SimpleConnector()) == 17


def test_scheduler_mcp(plan1):
    assert do_sched_test(plan1, 2, MCPScheduler(), SimpleConnector()) == 17


def test_scheduler_etf(plan1):
    assert do_sched_test(plan1, 2, ETFScheduler(), SimpleConnector()) == 17


def test_compute_indepndent_tasks(plan1):
    it = compute_independent_tasks(plan1)
    a1, a2, a3, a4, a5, a6, a7, a8 = plan1.tasks
    assert it[a1] == frozenset((a2, a4, a6, a7))
    assert it[a2] == frozenset((a1, a3, a4, a6, a7))
    assert it[a3] == frozenset((a2, a4, a6, a7))
    assert it[a5] == frozenset((a6, a7))
    assert it[a7] == frozenset((a1, a2, a3, a4, a5, a6))
    assert it[a8] == frozenset()


def test_compute_t_level(plan1):
    t = compute_t_level(plan1, lambda t: t.duration + t.size)

    assert t[task_by_name(plan1, "a1")] == 0
    assert t[task_by_name(plan1, "a2")] == 0
    assert t[task_by_name(plan1, "a3")] == 3
    assert t[task_by_name(plan1, "a4")] == 0
    assert t[task_by_name(plan1, "a5")] == 7
    assert t[task_by_name(plan1, "a6")] == 7
    assert t[task_by_name(plan1, "a7")] == 0
    assert t[task_by_name(plan1, "a8")] == 14


def test_compute_b_level(plan1):
    b = compute_b_level(plan1, lambda t: t.duration + t.size)

    assert b[task_by_name(plan1, "a1")] == 10
    assert b[task_by_name(plan1, "a2")] == 10
    assert b[task_by_name(plan1, "a3")] == 7
    assert b[task_by_name(plan1, "a4")] == 16
    assert b[task_by_name(plan1, "a5")] == 4
    assert b[task_by_name(plan1, "a6")] == 9
    assert b[task_by_name(plan1, "a7")] == 5
    assert b[task_by_name(plan1, "a8")] == 2


def test_compute_alap(plan1):
    alap = compute_alap(plan1, 1)

    assert alap[task_by_name(plan1, "a1")] == 6
    assert alap[task_by_name(plan1, "a2")] == 6
    assert alap[task_by_name(plan1, "a3")] == 9
    assert alap[task_by_name(plan1, "a4")] == 0
    assert alap[task_by_name(plan1, "a5")] == 12
    assert alap[task_by_name(plan1, "a6")] == 7
    assert alap[task_by_name(plan1, "a7")] == 11
    assert alap[task_by_name(plan1, "a8")] == 14
