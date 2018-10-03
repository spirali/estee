
from .schedulers import AllOnOneScheduler, RandomScheduler, RandomGtScheduler, BlevelGtScheduler, RandomAssignScheduler
from .connectors import SimpleConnector

from .test_utils import do_sched_test


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
        assert do_sched_test(plan1, 2, BlevelGtScheduler(False), SimpleConnector()) in [13, 16]
        assert do_sched_test(plan1, 2, BlevelGtScheduler(True), SimpleConnector()) in [16]


def test_scheduler_random_assign(plan1):
    for _ in range(50):
        assert 10 <= do_sched_test(plan1, 2, RandomAssignScheduler(), SimpleConnector()) <= 22
        assert 9 <= do_sched_test(plan1, 3, RandomAssignScheduler(), SimpleConnector()) <= 22

"""
def test_xxx(plan1):
    assert do_sched_test(plan1, 3, RandomAssignScheduler(), SimpleConnector(), report="/tmp/rep.html")
"""