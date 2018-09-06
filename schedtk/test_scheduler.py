
from .schedulers import AllOnOneScheduler, RandomScheduler


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
    assert 17 == do_sched_test(plan1, 1, RandomScheduler())
    for _ in range(50):
        assert 9 <= do_sched_test(plan1, 2, RandomScheduler()) <= 12
    for _ in range(50):
        assert 8 <= do_sched_test(plan1, 3, RandomScheduler()) <= 9
