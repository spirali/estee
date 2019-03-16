from estee.common import TaskGraph
from estee.schedulers import (AllOnOneScheduler, BlevelGtScheduler,
                                 Camp2Scheduler,
                                 DLSScheduler, ETFScheduler, LASTScheduler,
                                 MCPScheduler,
                                 RandomAssignScheduler, RandomGtScheduler,
                                 RandomScheduler, WorkStealingScheduler, SchedulerBase)

from estee.schedulers.genetic import GeneticScheduler
from estee.schedulers.utils import compute_alap, compute_b_level_duration_size, \
    compute_independent_tasks, compute_t_level_duration_size, topological_sort, \
    worker_estimate_earliest_time
from estee.simulator import TaskAssignment, SimpleNetModel, Worker
from estee.simulator.worker import RunningTask
from estee.schedulers.clustering import find_critical_path, critical_path_clustering
from estee.schedulers.utils import (compute_alap,
                                       compute_independent_tasks)
from estee.schedulers.utils import compute_b_level_duration_size, \
    compute_t_level_duration_size
from .test_utils import do_sched_test, task_by_name


def test_scheduler_all_on_one(plan1):

    scheduler = AllOnOneScheduler()
    scheduler._disable_cleanup = True
    assert 17 == do_sched_test(plan1, 1, scheduler)

    for obj in scheduler.task_graph.objects.values():
        assert len(obj.placing) == 1
        assert obj.availability == obj.placing

    scheduler = AllOnOneScheduler()
    scheduler._disable_cleanup = True
    assert 17 == do_sched_test(plan1, 3, scheduler)

    for obj in scheduler.task_graph.objects.values():
        assert len(obj.placing) == 1
        assert obj.availability == obj.placing


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
        assert 13 <= do_sched_test(plan1, 2, RandomScheduler(), SimpleNetModel()) <= 20


def test_scheduler_random_gt(plan1):

    # 2w, simple
    for _ in range(50):
        assert 13 <= do_sched_test(plan1, 2, RandomGtScheduler(), SimpleNetModel()) <= 19


def test_scheduler_blevel_gt(plan1):

    # 2w, simple
    for _ in range(50):
        scheduler = BlevelGtScheduler()
        scheduler._disable_cleanup = True
        assert do_sched_test(plan1, 2, scheduler, SimpleNetModel()) in [13, 16]

        sizes = set()
        for obj in scheduler.task_graph.objects.values():
            assert len(obj.placing) == 1
            sizes.add(len(obj.availability))

        assert sizes == {1, 2}


def test_scheduler_random_assign(plan1):
    for _ in range(50):
        assert 10 <= do_sched_test(plan1, 2, RandomAssignScheduler(), SimpleNetModel()) <= 25
        assert 9 <= do_sched_test(plan1, 3, RandomAssignScheduler(), SimpleNetModel()) <= 25


def test_scheduler_camp(plan1):
    for _ in range(10):
        assert 10 <= do_sched_test(plan1, 2, Camp2Scheduler(), SimpleNetModel()) <= 18


def test_scheduler_dls(plan1):
    assert 14 <= do_sched_test(plan1, 2, DLSScheduler(), SimpleNetModel()) <= 15


def test_scheduler_last(plan1):
    assert do_sched_test(plan1, 2, LASTScheduler(), SimpleNetModel()) == 17


def test_scheduler_mcp(plan1):
    assert do_sched_test(plan1, 2, MCPScheduler(), SimpleNetModel()) == 15


def test_scheduler_etf(plan1):
    assert do_sched_test(plan1, 2, ETFScheduler(), SimpleNetModel()) == 15


def test_scheduler_genetic(plan1):
    assert 10 <= do_sched_test(plan1, 2, GeneticScheduler(), SimpleNetModel()) <= 20


def test_scheduler_ws(plan1):
    assert 12 <= do_sched_test(plan1, 2, WorkStealingScheduler(), SimpleNetModel()) <= 18


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
    t = compute_t_level_duration_size(None, plan1)

    assert t[task_by_name(plan1, "a1")] == 0
    assert t[task_by_name(plan1, "a2")] == 0
    assert t[task_by_name(plan1, "a3")] == 3
    assert t[task_by_name(plan1, "a4")] == 0
    assert t[task_by_name(plan1, "a5")] == 7
    assert t[task_by_name(plan1, "a6")] == 7
    assert t[task_by_name(plan1, "a7")] == 0
    assert t[task_by_name(plan1, "a8")] == 14


def test_compute_b_level_plan1(plan1):
    b = compute_b_level_duration_size(None, plan1)

    assert b[task_by_name(plan1, "a1")] == 9
    assert b[task_by_name(plan1, "a2")] == 9
    assert b[task_by_name(plan1, "a3")] == 6
    assert b[task_by_name(plan1, "a4")] == 15
    assert b[task_by_name(plan1, "a5")] == 3
    assert b[task_by_name(plan1, "a6")] == 8
    assert b[task_by_name(plan1, "a7")] == 4
    assert b[task_by_name(plan1, "a8")] == 1


def test_compute_b_level_multiple_outputs():
    tg = TaskGraph()
    a = tg.new_task(outputs=[2, 4], expected_duration=0)
    b = tg.new_task(outputs=[5], expected_duration=0)
    c = tg.new_task(outputs=[2], expected_duration=0)
    d = tg.new_task(expected_duration=0)

    b.add_input(a.outputs[0])
    c.add_input(a.outputs[1])
    d.add_inputs((b, c))

    blevel = compute_b_level_duration_size(None, tg)

    assert blevel[a] == 7
    assert blevel[b] == 5
    assert blevel[c] == 2
    assert blevel[d] == 0


def test_compute_alap(plan1):
    alap = compute_alap(None, plan1, 1)

    assert alap[task_by_name(plan1, "a1")] == 6
    assert alap[task_by_name(plan1, "a2")] == 9
    assert alap[task_by_name(plan1, "a3")] == 10
    assert alap[task_by_name(plan1, "a4")] == 6
    assert alap[task_by_name(plan1, "a5")] == 13
    assert alap[task_by_name(plan1, "a6")] == 8
    assert alap[task_by_name(plan1, "a7")] == 13
    assert alap[task_by_name(plan1, "a8")] == 14


def test_worker_estimate_earliest_time():
    tg = TaskGraph()
    t0 = tg.new_task(expected_duration=3, cpus=2)
    t1 = tg.new_task(expected_duration=5, cpus=1)
    t2 = tg.new_task(expected_duration=4, cpus=3)
    t4 = tg.new_task(expected_duration=4, cpus=2)
    t5 = tg.new_task(expected_duration=4, cpus=2)

    worker = Worker(cpus=4)
    worker.assignments = [TaskAssignment(worker, t0), TaskAssignment(worker, t1),
                          TaskAssignment(worker, t2), TaskAssignment(worker, t4)]
    worker.running_tasks[t0] = RunningTask(t0, 0)
    worker.running_tasks[t1] = RunningTask(t1, 0)

    assert worker_estimate_earliest_time(worker, t5, 0) == 7


def test_worker_estimate_earliest_time_multiple_at_once():
    tg = TaskGraph()
    t0 = tg.new_task(expected_duration=3, cpus=1)
    t1 = tg.new_task(expected_duration=3, cpus=1)
    t2 = tg.new_task(expected_duration=3, cpus=1)

    worker = Worker(cpus=2)
    worker.assignments = [TaskAssignment(worker, t0), TaskAssignment(worker, t1)]
    worker.running_tasks[t0] = RunningTask(t0, 0)
    worker.running_tasks[t1] = RunningTask(t1, 0)

    assert worker_estimate_earliest_time(worker, t2, 0) == 3


def test_worker_estimate_earliest_time_offset_now():
    tg = TaskGraph()
    t0 = tg.new_task(expected_duration=3, cpus=1)
    t1 = tg.new_task(expected_duration=5, cpus=1)
    t2 = tg.new_task(expected_duration=3, cpus=2)

    worker = Worker(cpus=2)
    worker.assignments = [TaskAssignment(worker, t0), TaskAssignment(worker, t1)]
    worker.running_tasks[t0] = RunningTask(t0, 0)
    worker.running_tasks[t1] = RunningTask(t1, 0)

    assert worker_estimate_earliest_time(worker, t2, 2) == 3


def test_topological_sort(plan1):
    tasks = ['a1', 'a2', 'a4', 'a7', 'a3', 'a5', 'a6', 'a8']
    assert topological_sort(plan1) == [task_by_name(plan1, t) for t in tasks]


def test_find_critical_path(plan1):
    path = find_critical_path(plan1)
    assert [t.id for t in path] == [3, 5, 7]


def test_critical_path_clustering(plan1):
    assert [[3, 5, 7], [0, 2, 4], [1], [6]] == \
           [[t.id for t in p] for p in critical_path_clustering(plan1)]


def test_simulator_local_reassign():
    test_graph = TaskGraph()

    a0 = test_graph.new_task("A0", duration=1, output_size=1)
    a1 = test_graph.new_task("A1", duration=1, output_size=1)
    a2 = test_graph.new_task("A2", duration=1, cpus=1, output_size=10)

    a2.add_inputs([a1, a0])

    class Scheduler(SchedulerBase):

        def start(self):
            self.done = False
            return super().start()

        def schedule(self, update):
            if not self.task_graph.tasks or self.done:
                return

            t = self.task_graph.tasks[a2.id]
            for o in t.inputs:
                assert not o.scheduled
            for o in t.outputs:
                assert not o.scheduled

            w1 = self.workers[1]
            w2 = self.workers[2]
            self.assign(w1, t)

            for o in t.inputs:
                assert o.scheduled == {w1}
            for o in t.outputs:
                assert o.scheduled == {w1}

            self.assign(w2, t)

            for o in t.inputs:
                assert o.scheduled == {w2}
            for o in t.outputs:
                assert o.scheduled == {w2}

            for t in self.task_graph.tasks.values():
                self.assign(w1, t)

            self.done = True

    scheduler = Scheduler("test", "0", True)
    simulator = do_sched_test(test_graph, [1, 1, 1],
                              scheduler,
                              trace=True, netmodel=SimpleNetModel(1))